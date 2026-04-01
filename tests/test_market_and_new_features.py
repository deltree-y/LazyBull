"""测试市场状态特征与新增个股特征"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.market_state import (
    compute_market_state_features,
    precompute_market_state_features,
    _compute_daily_market_stats,
)
from src.lazybull.features.builder import FeatureBuilder
from src.lazybull.factors.normalization import cross_sectional_zscore


# ---------------------------------------------------------------------------
# TestIsNewStock
# ---------------------------------------------------------------------------

class TestIsNewStock:
    """测试新股标记（is_new_stock）特征"""

    def _make_builder_and_df(self, list_days_list):
        builder = FeatureBuilder()
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(len(list_days_list))],
            'list_days': list_days_list,
        })
        result = builder._add_new_individual_features(df)
        return result

    def test_is_new_stock_within_365_days(self):
        """上市不足 365 天应标记为新股（is_new_stock=1）"""
        result = self._make_builder_and_df([100, 200, 364])
        assert list(result['is_new_stock']) == [1, 1, 1]

    def test_is_new_stock_exactly_365_days(self):
        """恰好上市 365 天不应标记为新股（is_new_stock=0，边界严格小于）"""
        result = self._make_builder_and_df([365])
        assert result['is_new_stock'].iloc[0] == 0

    def test_is_new_stock_over_365_days(self):
        """上市超过 365 天不应标记为新股（is_new_stock=0）"""
        result = self._make_builder_and_df([400, 1000])
        assert list(result['is_new_stock']) == [0, 0]


# ---------------------------------------------------------------------------
# TestZscoreSize
# ---------------------------------------------------------------------------

class TestZscoreSize:
    """测试流通市值 Z-Score 标准化（zscore_size）"""

    def _build(self, df):
        builder = FeatureBuilder()
        return builder._add_new_individual_features(df.copy())

    def test_zscore_size_large_group(self):
        """行业内股票数 >=5 时，组内 zscore 均值应接近 0"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'circ_mv': [100, 200, 300, 400, 500, 150, 250, 350, 450, 550],
            'sw_industry': ['A'] * 5 + ['B'] * 5,
            'tradable': [1] * 10,
        })
        result = self._build(df)
        assert 'zscore_size' in result.columns
        # 每个行业内部均值应接近 0
        for ind in ['A', 'B']:
            group_mean = result[result['sw_industry'] == ind]['zscore_size'].mean()
            assert abs(group_mean) < 1e-6, f"行业 {ind} zscore_size 均值应接近 0，实际={group_mean}"

    def test_zscore_size_small_group_fallback(self):
        """行业内可交易股票数 <5 时，回退至全市场统计"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(8)],
            'circ_mv': [100, 200, 300, 400, 500, 150, 250, 350],
            # A 行业 5 支，B 行业仅 3 支（小组回退）
            'sw_industry': ['A'] * 5 + ['B'] * 3,
            'tradable': [1] * 8,
        })
        result = self._build(df)
        assert 'zscore_size' in result.columns

        # B 行业使用全市场统计：用全部 tradable 的 log1p(circ_mv) 计算 z-score
        # pandas Series.std() 默认 ddof=1，与 normalization 模块一致
        log_sizes = pd.Series(np.log1p(df['circ_mv'].values.astype(float)))
        global_mean = float(log_sizes.mean())
        global_std = float(log_sizes.std())   # ddof=1

        b_mask = df['sw_industry'] == 'B'
        for idx in df[b_mask].index:
            expected = (log_sizes.iloc[idx] - global_mean) / global_std
            actual = result.loc[idx, 'zscore_size']
            assert abs(actual - expected) < 1e-6, (
                f"B 行业股票 {idx} 应使用全市场统计，期望={expected:.6f}，实际={actual:.6f}"
            )

    def test_zscore_size_tradable_only(self):
        """非可交易股票不应影响 zscore_size 的统计量"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'circ_mv': [100, 200, 300, 400, 500, 150, 250, 350, 450, 550],
            'sw_industry': ['A'] * 5 + ['B'] * 5,
            # B 行业后 2 支不可交易，满足 tradable>=5 的 A 行业正常计算
            'tradable': [1] * 5 + [1, 1, 1, 0, 0],
        })
        result = self._build(df)
        assert 'zscore_size' in result.columns
        # A 行业全部可交易，均值应接近 0
        group_mean = result[result['sw_industry'] == 'A']['zscore_size'].mean()
        assert abs(group_mean) < 1e-6

    def test_zscore_size_column_name(self):
        """输出列名应为 zscore_size（而非 size_zscore）"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(5)],
            'circ_mv': [100, 200, 300, 400, 500],
            'sw_industry': ['A'] * 5,
            'tradable': [1] * 5,
        })
        result = self._build(df)
        assert 'zscore_size' in result.columns
        assert 'size_zscore' not in result.columns


# ---------------------------------------------------------------------------
# TestSpecScore
# ---------------------------------------------------------------------------

class TestSpecScore:
    """测试综合评分（spec_score = zscore_volatility_20 * (-zscore_size)）"""

    def _build(self, df):
        builder = FeatureBuilder()
        return builder._add_new_individual_features(df.copy())

    def test_spec_score_formula(self):
        """spec_score 应等于 zscore_volatility_20 * (-zscore_size)"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(3)],
            'zscore_volatility_20': [1.5, -0.5, 0.8],
            'zscore_size': [0.3, -1.2, 0.5],
        })
        result = self._build(df)
        assert 'spec_score' in result.columns
        expected = [1.5 * (-0.3), (-0.5) * (1.2), 0.8 * (-0.5)]
        for i, exp in enumerate(expected):
            assert abs(result['spec_score'].iloc[i] - exp) < 1e-9, (
                f"第 {i} 行 spec_score 期望={exp:.9f}，实际={result['spec_score'].iloc[i]:.9f}"
            )

    def test_spec_score_missing_dependency(self):
        """缺少 zscore_volatility_20 时，spec_score 应全为 NaN"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'zscore_size': [0.3, -1.2],
        })
        result = self._build(df)
        assert 'spec_score' in result.columns
        assert result['spec_score'].isna().all()


# ---------------------------------------------------------------------------
# TestMarketStateSingleDay
# ---------------------------------------------------------------------------

class TestMarketStateSingleDay:
    """测试单日市场截面统计特征"""

    def _make_daily_data(self):
        """构造测试用单日行情数据（vol=0 表示停牌）"""
        return pd.DataFrame({
            'trade_date': ['20230101'] * 5,
            'ts_code': [f'{i:06d}.SZ' for i in range(5)],
            'pct_chg': [2.0, -1.0, 3.0, -2.0, 0.5],
            'vol': [1000, 2000, 1500, 0, 800],   # 第 4 支停牌
            'amount': [5000.0, 8000.0, 6000.0, 0.0, 3000.0],
        })

    def test_mkt_vol_cnt_basic(self):
        """mkt_vol_cnt 应为可交易股票收益率的截面标准差"""
        daily_data = self._make_daily_data()
        result = compute_market_state_features(daily_data, '20230101', ['20230101'], 0)
        # 可交易股票（vol>0）：pct_chg = [2.0, -1.0, 3.0, 0.5]，转换为小数
        # pandas Series.std() 使用 ddof=1，与实现一致
        tradable_rets = pd.Series([2.0, -1.0, 3.0, 0.5]) / 100.0
        expected_std = float(tradable_rets.std())   # ddof=1
        assert abs(result['mkt_vol_cnt'] - expected_std) < 1e-9

    def test_mkt_vol_cnt_tradable_only(self):
        """mkt_vol_cnt 计算时应排除 vol=0（停牌）股票"""
        daily_data = self._make_daily_data()
        stats = _compute_daily_market_stats(daily_data, '20230101')
        # vol=0 的股票不应纳入计算，共 4 支可交易
        # pandas Series.std() 使用 ddof=1
        all_rets = pd.Series([2.0, -1.0, 3.0, -2.0, 0.5]) / 100.0
        all_std = float(all_rets.std())
        tradable_rets = pd.Series([2.0, -1.0, 3.0, 0.5]) / 100.0
        tradable_std = float(tradable_rets.std())
        # 结果不应等于含停牌股票的全量标准差
        assert abs(stats['vol_cnt'] - tradable_std) < 1e-9
        assert abs(stats['vol_cnt'] - all_std) > 1e-9

    def test_mkt_turnover_ratio(self):
        """mkt_turnover_ratio = sum(amount) / sum(circ_mv)，仅统计可交易股票"""
        daily_data = self._make_daily_data()
        daily_basic = pd.DataFrame({
            'trade_date': ['20230101'] * 5,
            'ts_code': [f'{i:06d}.SZ' for i in range(5)],
            'circ_mv': [1e6, 2e6, 1.5e6, 1e6, 8e5],
            'turnover_rate_f': [1.5, 2.0, 1.2, 0.8, 1.8],
        })
        result = compute_market_state_features(
            daily_data, '20230101', ['20230101'], 0, daily_basic
        )
        # vol>0 的股票：索引 0,1,2,4
        tradable_codes = {'000000.SZ', '000001.SZ', '000002.SZ', '000004.SZ'}
        amount_sum = daily_data[daily_data['ts_code'].isin(tradable_codes)]['amount'].sum()
        circ_mv_sum = daily_basic[daily_basic['ts_code'].isin(tradable_codes)]['circ_mv'].sum()
        expected = amount_sum / circ_mv_sum
        assert abs(result['mkt_turnover_ratio'] - expected) < 1e-9

    def test_mkt_turnover_std(self):
        """mkt_turnover_std 应为可交易股票换手率的标准差"""
        daily_data = self._make_daily_data()
        daily_basic = pd.DataFrame({
            'trade_date': ['20230101'] * 5,
            'ts_code': [f'{i:06d}.SZ' for i in range(5)],
            'circ_mv': [1e6, 2e6, 1.5e6, 1e6, 8e5],
            'turnover_rate_f': [1.5, 2.0, 1.2, 0.8, 1.8],
        })
        result = compute_market_state_features(
            daily_data, '20230101', ['20230101'], 0, daily_basic
        )
        tradable_codes = {'000000.SZ', '000001.SZ', '000002.SZ', '000004.SZ'}
        tf_vals = daily_basic[daily_basic['ts_code'].isin(tradable_codes)]['turnover_rate_f']
        expected_std = float(tf_vals.std())
        assert abs(result['mkt_turnover_std'] - expected_std) < 1e-9


# ---------------------------------------------------------------------------
# TestMarketStateRolling
# ---------------------------------------------------------------------------

class TestMarketStateRolling:
    """测试需要历史数据的滚动市场特征"""

    def _make_multi_day_data(self, n_days, n_stocks=20, seed=42):
        """构造多日行情测试数据"""
        rng = np.random.default_rng(seed)
        # 生成连续日期字符串（YYYYMMDD 格式）
        dates = []
        base = pd.Timestamp('20230101')
        for i in range(n_days):
            dates.append((base + pd.Timedelta(days=i)).strftime('%Y%m%d'))

        rows = []
        for d in dates:
            for j in range(n_stocks):
                rows.append({
                    'trade_date': d,
                    'ts_code': f'{j:06d}.SZ',
                    'pct_chg': float(rng.normal(0, 2)),
                    'vol': 1000,
                    'amount': float(rng.uniform(1000, 10000)),
                })
        return pd.DataFrame(rows), dates

    def test_mkt_vol_20(self):
        """mkt_vol_20 应为最近 20 日 mkt_vol_cnt 的均值（min_periods=1）"""
        daily_data, dates = self._make_multi_day_data(25)
        last_date = dates[-1]
        result = compute_market_state_features(daily_data, last_date, dates, len(dates) - 1)
        assert not np.isnan(result['mkt_vol_20'])
        # 手动计算近 20 日的 vol_cnt 均值
        recent_dates = dates[-20:]
        vol_cnts = [
            _compute_daily_market_stats(daily_data, d)['vol_cnt']
            for d in recent_dates
        ]
        expected = float(np.mean([v for v in vol_cnts if not np.isnan(v)]))
        assert abs(result['mkt_vol_20'] - expected) < 1e-9

    def test_mkt_ret_avg_20(self):
        """mkt_ret_avg_20 应为最近 20 日截面平均收益率之和"""
        daily_data, dates = self._make_multi_day_data(25)
        last_date = dates[-1]
        result = compute_market_state_features(daily_data, last_date, dates, len(dates) - 1)
        assert not np.isnan(result['mkt_ret_avg_20'])
        # 手动计算最近 20 日 mean_ret 之和
        recent_dates = dates[-20:]
        mean_rets = [
            _compute_daily_market_stats(daily_data, d)['mean_ret']
            for d in recent_dates
        ]
        expected = float(sum(v for v in mean_rets if not np.isnan(v)))
        assert abs(result['mkt_ret_avg_20'] - expected) < 1e-9

    def test_mkt_adv_dec_ratio_rolling(self):
        """mkt_adv_dec_ratio 应为最近 60 日涨跌比的滚动均值"""
        daily_data, dates = self._make_multi_day_data(65)
        last_date = dates[-1]
        result = compute_market_state_features(daily_data, last_date, dates, len(dates) - 1)
        assert not np.isnan(result['mkt_adv_dec_ratio'])
        # 手动计算最近 60 日 adv_dec_ratio 均值
        recent_dates = dates[-60:]
        ratios = [
            _compute_daily_market_stats(daily_data, d)['adv_dec_ratio']
            for d in recent_dates
        ]
        expected = float(np.mean([v for v in ratios if not np.isnan(v)]))
        assert abs(result['mkt_adv_dec_ratio'] - expected) < 1e-9

    def test_rolling_window_partial(self):
        """历史数据不足 20 日时，mkt_vol_20 应使用现有数据而非返回 NaN"""
        daily_data, dates = self._make_multi_day_data(5)
        last_date = dates[-1]
        result = compute_market_state_features(daily_data, last_date, dates, len(dates) - 1)
        # 仅 5 日数据，mkt_vol_20 不应为 NaN
        assert not np.isnan(result['mkt_vol_20'])
        assert not np.isnan(result['mkt_ret_avg_20'])


# ---------------------------------------------------------------------------
# TestPrecomputeMarketStateFeatures
# ---------------------------------------------------------------------------

class TestPrecomputeMarketStateFeatures:
    """测试批量预计算市场状态特征（precompute_market_state_features）"""

    def _make_multi_day_data(self, n_days, n_stocks=20, seed=42):
        """构造多日行情测试数据（同 TestMarketStateRolling）"""
        rng = np.random.default_rng(seed)
        base = pd.Timestamp('20230101')
        dates = [(base + pd.Timedelta(days=i)).strftime('%Y%m%d') for i in range(n_days)]
        rows = []
        for d in dates:
            for j in range(n_stocks):
                rows.append({
                    'trade_date': d,
                    'ts_code': f'{j:06d}.SZ',
                    'pct_chg': float(rng.normal(0, 2)),
                    'vol': float(rng.integers(0, 1000)),
                    'amount': float(rng.uniform(1000, 10000)),
                })
        return pd.DataFrame(rows), dates

    def _make_daily_basic(self, dates, n_stocks=20, seed=99):
        """构造多日 daily_basic 测试数据"""
        rng = np.random.default_rng(seed)
        rows = []
        for d in dates:
            for j in range(n_stocks):
                rows.append({
                    'trade_date': d,
                    'ts_code': f'{j:06d}.SZ',
                    'circ_mv': float(rng.uniform(1e6, 1e8)),
                    'turnover_rate_f': float(rng.uniform(0.5, 5.0)),
                })
        return pd.DataFrame(rows)

    def test_output_shape(self):
        """批量预计算应返回行数等于 trading_dates 的 DataFrame"""
        daily_data, dates = self._make_multi_day_data(30)
        result = precompute_market_state_features(daily_data, dates)
        assert len(result) == len(dates)
        assert set(result.columns) == {
            'mkt_vol_cnt', 'mkt_vol_20', 'mkt_turnover_ratio',
            'mkt_ret_avg_20', 'mkt_turnover_std', 'mkt_adv_dec_ratio',
            'mkt_ma_trend', 'mkt_drawdown_20', 'mkt_ret_avg_60', 'mkt_ret_vol_20',
            'mkt_ma250_ratio',
            'mkt_atr_pct', 'mkt_atr_pct_ma250',
        }

    def test_parity_with_single_day_no_basic(self):
        """批量预计算结果应与逐日计算完全一致（无 daily_basic）

        新增特征（mkt_ma_trend 等）仅在批量模式下计算，逐日模式返回 NaN，
        因此 parity 比较时跳过这些列。
        """
        # 仅批量模式可用的特征，逐日模式固定返回 NaN
        batch_only_cols = {'mkt_ma_trend', 'mkt_drawdown_20', 'mkt_ret_avg_60', 'mkt_ret_vol_20', 'mkt_ma250_ratio', 'mkt_atr_pct', 'mkt_atr_pct_ma250'}

        daily_data, dates = self._make_multi_day_data(70)
        batch = precompute_market_state_features(daily_data, dates)
        # 对后 5 天逐日比对共有字段
        for i, d in enumerate(dates[-5:], len(dates) - 5):
            single = compute_market_state_features(daily_data, d, dates, i)
            for col in single:
                if col in batch_only_cols:
                    continue  # 跳过仅批量模式计算的特征
                b_val = float(batch.loc[d, col])
                s_val = float(single[col])
                if np.isnan(s_val):
                    assert np.isnan(b_val), f"[{d}] {col}: 期望 NaN，实际 {b_val}"
                else:
                    assert abs(b_val - s_val) < 1e-9, (
                        f"[{d}] {col}: 批量={b_val:.12f} 逐日={s_val:.12f}"
                    )

    def test_parity_with_single_day_with_basic(self):
        """批量预计算结果应与逐日计算完全一致（含 daily_basic）"""
        # 仅批量模式可用的特征
        batch_only_cols = {'mkt_ma_trend', 'mkt_drawdown_20', 'mkt_ret_avg_60', 'mkt_ret_vol_20', 'mkt_ma250_ratio', 'mkt_atr_pct', 'mkt_atr_pct_ma250'}

        daily_data, dates = self._make_multi_day_data(70)
        daily_basic = self._make_daily_basic(dates)
        batch = precompute_market_state_features(daily_data, dates, daily_basic)
        for i, d in enumerate(dates[-5:], len(dates) - 5):
            single = compute_market_state_features(daily_data, d, dates, i, daily_basic)
            for col in single:
                if col in batch_only_cols:
                    continue
                b_val = float(batch.loc[d, col])
                s_val = float(single[col])
                if np.isnan(s_val):
                    assert np.isnan(b_val), f"[{d}] {col}: 期望 NaN，实际 {b_val}"
                else:
                    assert abs(b_val - s_val) < 1e-9, (
                        f"[{d}] {col}: 批量={b_val:.12f} 逐日={s_val:.12f}"
                    )

    def test_rolling_min_periods_1(self):
        """rolling min_periods=1：数据不足窗口时不应返回 NaN"""
        daily_data, dates = self._make_multi_day_data(5)
        result = precompute_market_state_features(daily_data, dates)
        # 全部 5 天的 mkt_vol_20 和 mkt_ret_avg_20 均不应为 NaN（min_periods=1）
        assert not result['mkt_vol_20'].isna().any(), "min_periods=1 时 mkt_vol_20 不应有 NaN"
        assert not result['mkt_ret_avg_20'].isna().any(), "min_periods=1 时 mkt_ret_avg_20 不应有 NaN"

    def test_empty_data_returns_nan(self):
        """空 daily_data 时应返回全 NaN，不应抛出异常"""
        dates = ['20230101', '20230102']
        empty = pd.DataFrame(columns=['trade_date', 'ts_code', 'pct_chg', 'vol'])
        result = precompute_market_state_features(empty, dates)
        assert len(result) == 2
        assert result['mkt_vol_cnt'].isna().all()

    def test_no_duplicate_compute_with_cache(self):
        """FeatureBuilder 缓存：多次调用 _add_market_state_features 只触发一次批量预计算"""
        daily_data, dates = self._make_multi_day_data(10)
        builder = FeatureBuilder()
        assert builder._market_state_cache is None

        # 模拟两次调用：每次传入一个空截面 DataFrame
        result_df = pd.DataFrame({'ts_code': ['000000.SZ', '000001.SZ']})
        for i, d in enumerate(dates[:2]):
            builder._add_market_state_features(result_df.copy(), daily_data, d, dates, i)

        # 缓存应在第一次调用后建立，且不为空
        assert builder._market_state_cache is not None
        assert len(builder._market_state_cache) == len(dates)


# ---------------------------------------------------------------------------
# TestWarmupStartDateIndependence
# ---------------------------------------------------------------------------

class TestWarmupStartDateIndependence:
    """测试 warmup 机制确保同一 trade_date 特征不受历史起点（--start-date）影响"""

    def _make_daily_data(self, n_days=200, n_stocks=15, seed=42):
        """构造多日行情测试数据（含技术指标所需字段）"""
        rng = np.random.default_rng(seed)
        base = pd.Timestamp('20230101')
        dates = [(base + pd.Timedelta(days=i)).strftime('%Y%m%d') for i in range(n_days)]
        rows = []
        for d in dates:
            for j in range(n_stocks):
                close = float(rng.uniform(5.0, 50.0))
                rows.append({
                    'trade_date': d,
                    'ts_code': f'{j:06d}.SZ',
                    'pct_chg': float(rng.normal(0, 2)),
                    'vol': float(rng.integers(500, 2000)),
                    'amount': float(rng.uniform(1000, 10000)),
                    'close_adj': close,
                    'high_adj': close * float(rng.uniform(1.0, 1.05)),
                    'low_adj': close * float(rng.uniform(0.95, 1.0)),
                    'open_adj': close * float(rng.uniform(0.98, 1.02)),
                    'ret_1': float(rng.normal(0, 0.02)),
                })
        return pd.DataFrame(rows), dates

    def _make_daily_basic(self, dates, n_stocks=15, seed=99):
        """构造多日 daily_basic 测试数据"""
        rng = np.random.default_rng(seed)
        rows = []
        for d in dates:
            for j in range(n_stocks):
                rows.append({
                    'trade_date': d,
                    'ts_code': f'{j:06d}.SZ',
                    'circ_mv': float(rng.uniform(1e6, 1e8)),
                    'turnover_rate_f': float(rng.uniform(0.5, 5.0)),
                })
        return pd.DataFrame(rows)

    def test_slice_by_trading_days_basic(self):
        """_slice_by_trading_days 应正确按 warmup_days 截取数据"""
        _, dates = self._make_daily_data(n_days=200)
        daily_data, _ = self._make_daily_data(n_days=200)
        builder = FeatureBuilder()

        anchor = dates[150]
        # warmup=120 → 起点应是 dates[30]
        sliced = builder._slice_by_trading_days(daily_data, dates, anchor, warmup_days=120)
        expected_start = dates[30]
        assert sliced['trade_date'].min() == expected_start
        # 不应包含 dates[29] 或之前的数据
        assert dates[29] not in set(sliced['trade_date'].unique())

    def test_slice_by_trading_days_insufficient_history(self):
        """历史不足 warmup_days 时应从第一个交易日开始（不抛异常）"""
        daily_data, dates = self._make_daily_data(n_days=50)
        builder = FeatureBuilder()
        anchor = dates[10]
        # warmup=120 > 10，应从 dates[0] 开始
        sliced = builder._slice_by_trading_days(daily_data, dates, anchor, warmup_days=120)
        assert sliced['trade_date'].min() == dates[0]

    def test_slice_by_trading_days_unknown_anchor(self):
        """anchor 不在 trading_dates 中时应原样返回 DataFrame"""
        daily_data, dates = self._make_daily_data(n_days=50)
        builder = FeatureBuilder()
        sliced = builder._slice_by_trading_days(daily_data, dates, '99991231', warmup_days=120)
        assert len(sliced) == len(daily_data)

    def test_market_state_independent_of_start_date(self):
        """同一目标日市场状态特征在历史充足时不受历史起点影响"""
        daily_data, dates = self._make_daily_data()
        # target_date = dates[160]，warmup 起点 = dates[40]（160-120=40）
        target_date = dates[160]
        result_df = pd.DataFrame({'ts_code': ['000000.SZ']})

        # Run1：使用全量数据（dates[0..199]）
        builder1 = FeatureBuilder()
        builder1._add_market_state_features(
            result_df.copy(), daily_data, target_date, dates, dates.index(target_date)
        )

        # Run2：截断到 dates[10..199]（仍覆盖 warmup 起点 dates[40]）
        trunc_dates = set(dates[10:])
        trunc_data = daily_data[daily_data['trade_date'].isin(trunc_dates)]
        builder2 = FeatureBuilder()
        builder2._add_market_state_features(
            result_df.copy(), trunc_data, target_date, dates, dates.index(target_date)
        )

        # 目标日期的所有市场状态特征应完全一致（两次切片后输入相同）
        for col in ['mkt_vol_cnt', 'mkt_vol_20', 'mkt_adv_dec_ratio', 'mkt_ret_avg_20']:
            v1 = float(builder1._market_state_cache.loc[target_date, col])
            v2 = float(builder2._market_state_cache.loc[target_date, col])
            if np.isnan(v1):
                assert np.isnan(v2), f"市场状态 {col}: run1=NaN, run2={v2}"
            else:
                assert abs(v1 - v2) < 1e-9, (
                    f"市场状态 {col} 不一致（受 start-date 影响）: "
                    f"run1={v1:.12f}, run2={v2:.12f}"
                )

    def test_market_state_with_basic_independent_of_start_date(self):
        """含 daily_basic 时，市场状态特征在历史充足条件下与起点无关"""
        daily_data, dates = self._make_daily_data()
        daily_basic = self._make_daily_basic(dates)
        target_date = dates[160]
        result_df = pd.DataFrame({'ts_code': ['000000.SZ']})

        # Run1：全量数据
        builder1 = FeatureBuilder()
        builder1._add_market_state_features(
            result_df.copy(), daily_data, target_date, dates, dates.index(target_date),
            daily_basic_data=daily_basic,
        )

        # Run2：截断到 dates[15..199]（仍覆盖 warmup 起点 dates[40]）
        trunc_dates = set(dates[15:])
        trunc_data = daily_data[daily_data['trade_date'].isin(trunc_dates)]
        trunc_basic = daily_basic[daily_basic['trade_date'].isin(trunc_dates)]
        builder2 = FeatureBuilder()
        builder2._add_market_state_features(
            result_df.copy(), trunc_data, target_date, dates, dates.index(target_date),
            daily_basic_data=trunc_basic,
        )

        for col in ['mkt_vol_cnt', 'mkt_vol_20', 'mkt_adv_dec_ratio',
                    'mkt_turnover_ratio', 'mkt_turnover_std']:
            v1 = float(builder1._market_state_cache.loc[target_date, col])
            v2 = float(builder2._market_state_cache.loc[target_date, col])
            if np.isnan(v1):
                assert np.isnan(v2), f"市场状态(含basic) {col}: run1=NaN, run2={v2}"
            else:
                assert abs(v1 - v2) < 1e-9, (
                    f"市场状态(含basic) {col} 不一致: run1={v1:.12f}, run2={v2:.12f}"
                )

    def test_tech_factor_independent_of_start_date(self):
        """同一目标日技术指标在历史充足时不受历史起点影响"""
        daily_data, dates = self._make_daily_data(n_days=200, n_stocks=5)
        target_date = dates[160]

        # Run1：全量数据
        builder1 = FeatureBuilder()
        tech1 = builder1._get_tech_factor_today(daily_data, target_date, dates)

        # Run2：截断到 dates[10..199]（仍覆盖 warmup 起点 dates[40]）
        trunc_dates = set(dates[10:])
        trunc_data = daily_data[daily_data['trade_date'].isin(trunc_dates)]
        builder2 = FeatureBuilder()
        tech2 = builder2._get_tech_factor_today(trunc_data, target_date, dates)

        assert len(tech1) > 0, "Run1 技术指标结果不应为空"
        assert len(tech2) > 0, "Run2 技术指标结果不应为空"

        # 对每支股票验证技术指标一致
        for col in ['macd_dif', 'macd_dea', 'macd_hist', 'kdj_k', 'kdj_d', 'kdj_j',
                    'rsi_14', 'bb_pct']:
            if col not in tech1.columns or col not in tech2.columns:
                continue
            for ts_code in tech1['ts_code'].unique():
                r1 = tech1[tech1['ts_code'] == ts_code][col]
                r2 = tech2[tech2['ts_code'] == ts_code][col]
                if r1.empty or r2.empty:
                    continue
                v1 = float(r1.iloc[0])
                v2 = float(r2.iloc[0])
                if np.isnan(v1):
                    assert np.isnan(v2), f"技术指标 {col}[{ts_code}]: run1=NaN, run2={v2}"
                else:
                    assert abs(v1 - v2) < 1e-9, (
                        f"技术指标 {col}[{ts_code}] 不一致: "
                        f"run1={v1:.12f}, run2={v2:.12f}"
                    )

    def test_insufficient_warmup_produces_different_values(self):
        """当截断导致 warmup 不足时，特征值可能不同（预期行为）"""
        daily_data, dates = self._make_daily_data()
        target_date = dates[160]
        result_df = pd.DataFrame({'ts_code': ['000000.SZ']})

        # Run1：全量数据（warmup 充足，60 日滚动窗口完整）
        builder1 = FeatureBuilder()
        builder1._add_market_state_features(
            result_df.copy(), daily_data, target_date, dates, dates.index(target_date)
        )

        # Run3：截断到 dates[115..199]（深入 60 日窗口内，dates[100..114] 丢失）
        # mkt_adv_dec_ratio 的 60 日窗口为 dates[100..160]，截断后只有 45 天数据
        trunc_dates = set(dates[115:])
        trunc_data = daily_data[daily_data['trade_date'].isin(trunc_dates)]
        builder3 = FeatureBuilder()
        builder3._add_market_state_features(
            result_df.copy(), trunc_data, target_date, dates, dates.index(target_date)
        )

        # mkt_adv_dec_ratio 使用 60 日窗口：数据不足时 (min_periods=1) 用不同子集计算
        v1 = float(builder1._market_state_cache.loc[target_date, 'mkt_adv_dec_ratio'])
        v3 = float(builder3._market_state_cache.loc[target_date, 'mkt_adv_dec_ratio'])
        assert not np.isnan(v1) and not np.isnan(v3)
        # 两者应不相等（数据截断导致不同的滚动窗口覆盖范围）
        assert abs(v1 - v3) > 1e-9, (
            f"截断后 mkt_adv_dec_ratio 预期不同，实际相同: v1={v1}, v3={v3}"
        )
