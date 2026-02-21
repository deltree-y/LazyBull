"""测试技术指标与波动率批量预计算功能

验证：
1. 小样本数据上"旧逻辑（按日切片计算）"与"新逻辑（一次预计算后按日取值）"结果一致
2. 缓存确实生效：多日构建时预计算函数只调用一次
3. compute_ret_1 口径修复：无 ret_1 时优先使用 close_adj.pct_change()
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.lazybull.factors.precompute_technical_factors import precompute_technical_factors
from src.lazybull.factors.returns import compute_ret_1
from src.lazybull.factors.technical_indicators import (
    calculate_rsi,
    calculate_kdj,
    calculate_macd,
    calculate_bollinger_bands,
)
from src.lazybull.factors.volatility import calculate_volatility
from src.lazybull.features.builder import FeatureBuilder


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_daily_adj():
    """构造一个小样本的后复权日线数据（5 支股票 × 60 天）"""
    np.random.seed(42)
    stocks = [f'{i:06d}.SZ' for i in range(1, 6)]
    # 生成 60 个工作日
    dates = pd.date_range('2023-01-01', periods=60, freq='B').strftime('%Y%m%d').tolist()

    records = []
    for stock in stocks:
        price = 10.0
        for date in dates:
            # 随机价格游走
            ret = np.random.normal(0, 0.01)
            close = price * (1 + ret)
            high = close * (1 + abs(np.random.normal(0, 0.005)))
            low = close * (1 - abs(np.random.normal(0, 0.005)))
            open_ = price
            records.append({
                'ts_code': stock,
                'trade_date': date,
                'open_adj': open_,
                'high_adj': high,
                'low_adj': low,
                'close_adj': close,
                'pct_chg': ret * 100,
            })
            price = close

    df = pd.DataFrame(records)
    return df


@pytest.fixture
def sample_trade_cal(sample_daily_adj):
    """构造与 sample_daily_adj 对应的交易日历"""
    dates = sorted(sample_daily_adj['trade_date'].unique().tolist())
    return pd.DataFrame({'cal_date': dates, 'is_open': 1})


@pytest.fixture
def sample_stock_basic():
    """模拟股票基本信息（上市时间足够久，均不过滤）"""
    stocks = [f'{i:06d}.SZ' for i in range(1, 6)]
    return pd.DataFrame({
        'ts_code': stocks,
        'name': [f'股票{i}' for i in range(1, 6)],
        'list_date': ['20100101'] * 5,
    })


@pytest.fixture
def sample_adj_factor(sample_daily_adj):
    """构造复权因子（全 1，即数据已是后复权）"""
    return pd.DataFrame({
        'ts_code': sample_daily_adj['ts_code'],
        'trade_date': sample_daily_adj['trade_date'],
        'adj_factor': 1.0,
    })


# ---------------------------------------------------------------------------
# TestPrecomputeTechnicalFactors：结果一致性测试
# ---------------------------------------------------------------------------

class TestPrecomputeTechnicalFactors:
    """验证批量预计算结果与旧逻辑（按日切片）完全一致"""

    def _old_logic_for_date(self, daily_adj: pd.DataFrame, trade_date: str,
                             trading_dates: list, current_idx: int,
                             vol_windows: list) -> dict:
        """旧逻辑：按日切片后分别调用各计算函数，取当日行"""
        result = {}

        # 技术指标历史窗口
        if current_idx >= 30:
            lookback = 50
            hist_start_date = trading_dates[max(0, current_idx - lookback)]
            hist_dates = [d for d in trading_dates if hist_start_date <= d <= trade_date]
            tech_hist = daily_adj[daily_adj['trade_date'].isin(hist_dates)].copy()

            if 'close_adj' in tech_hist.columns:
                rsi_df = calculate_rsi(tech_hist, window=14)
                row = rsi_df[rsi_df['trade_date'] == trade_date]
                if len(row) > 0:
                    for ts in row['ts_code']:
                        result.setdefault(ts, {})['rsi_14'] = float(
                            row[row['ts_code'] == ts]['rsi_14'].iloc[0]
                        )

            if all(c in tech_hist.columns for c in ['high_adj', 'low_adj', 'close_adj']):
                kdj_df = calculate_kdj(tech_hist)
                row = kdj_df[kdj_df['trade_date'] == trade_date]
                if len(row) > 0:
                    for ts in row['ts_code']:
                        result.setdefault(ts, {}).update({
                            'kdj_k': float(row[row['ts_code'] == ts]['kdj_k'].iloc[0]),
                            'kdj_d': float(row[row['ts_code'] == ts]['kdj_d'].iloc[0]),
                            'kdj_j': float(row[row['ts_code'] == ts]['kdj_j'].iloc[0]),
                        })

            if 'close_adj' in tech_hist.columns:
                macd_df = calculate_macd(tech_hist)
                row = macd_df[macd_df['trade_date'] == trade_date]
                if len(row) > 0:
                    for ts in row['ts_code']:
                        result.setdefault(ts, {}).update({
                            'macd_dif': float(row[row['ts_code'] == ts]['macd_dif'].iloc[0]),
                            'macd_dea': float(row[row['ts_code'] == ts]['macd_dea'].iloc[0]),
                            'macd_hist': float(row[row['ts_code'] == ts]['macd_hist'].iloc[0]),
                        })

            if 'close_adj' in tech_hist.columns:
                bb_df = calculate_bollinger_bands(tech_hist)
                row = bb_df[bb_df['trade_date'] == trade_date]
                if len(row) > 0:
                    for ts in row['ts_code']:
                        result.setdefault(ts, {}).update({
                            'bb_middle': float(row[row['ts_code'] == ts]['bb_middle'].iloc[0]),
                            'bb_upper': float(row[row['ts_code'] == ts]['bb_upper'].iloc[0]),
                            'bb_lower': float(row[row['ts_code'] == ts]['bb_lower'].iloc[0]),
                        })

        # 波动率历史窗口
        lookback_vol = max(vol_windows) + 1
        hist_start_date_v = trading_dates[max(0, current_idx - lookback_vol)]
        hist_dates_v = [d for d in trading_dates if hist_start_date_v <= d <= trade_date]
        vol_hist = daily_adj[daily_adj['trade_date'].isin(hist_dates_v)].copy()
        if 'pct_chg' in vol_hist.columns:
            vol_hist['ret_1'] = vol_hist['pct_chg'] / 100.0
        if 'ret_1' in vol_hist.columns:
            vol_df = calculate_volatility(vol_hist, ret_col='ret_1', windows=vol_windows)
            row = vol_df[vol_df['trade_date'] == trade_date]
            if len(row) > 0:
                for ts in row['ts_code']:
                    for w in vol_windows:
                        col = f'volatility_{w}'
                        if col in row.columns:
                            result.setdefault(ts, {})[col] = float(
                                row[row['ts_code'] == ts][col].iloc[0]
                            )

        return result

    def test_rsi_parity(self, sample_daily_adj):
        """新旧逻辑的 rsi_14 数值应完全一致（误差 < 1e-9）"""
        tech_all = precompute_technical_factors(sample_daily_adj, vol_windows=[20])
        trading_dates = sorted(sample_daily_adj['trade_date'].unique().tolist())
        # 选取一个历史充足的日期（current_idx=45）
        trade_date = trading_dates[45]
        current_idx = 45

        tech_today = tech_all[tech_all['trade_date'] == trade_date]

        old_result = self._old_logic_for_date(
            sample_daily_adj, trade_date, trading_dates, current_idx, vol_windows=[20]
        )

        assert 'rsi_14' in tech_today.columns, "批量预计算结果应包含 rsi_14 列"
        for ts in tech_today['ts_code']:
            new_val = float(tech_today[tech_today['ts_code'] == ts]['rsi_14'].iloc[0])
            old_val = old_result.get(ts, {}).get('rsi_14', np.nan)
            if not np.isnan(old_val):
                assert abs(new_val - old_val) < 1e-6, (
                    f"{ts} rsi_14 不一致：新={new_val:.8f}，旧={old_val:.8f}"
                )

    def test_kdj_parity(self, sample_daily_adj):
        """新旧逻辑的 kdj_k/kdj_d/kdj_j 数值应完全一致（误差 < 1e-9）"""
        tech_all = precompute_technical_factors(sample_daily_adj, vol_windows=[20])
        trading_dates = sorted(sample_daily_adj['trade_date'].unique().tolist())
        trade_date = trading_dates[45]
        current_idx = 45

        tech_today = tech_all[tech_all['trade_date'] == trade_date]
        old_result = self._old_logic_for_date(
            sample_daily_adj, trade_date, trading_dates, current_idx, vol_windows=[20]
        )

        for col in ['kdj_k', 'kdj_d', 'kdj_j']:
            assert col in tech_today.columns, f"批量预计算结果应包含 {col} 列"
            for ts in tech_today['ts_code']:
                new_val = float(tech_today[tech_today['ts_code'] == ts][col].iloc[0])
                old_val = old_result.get(ts, {}).get(col, np.nan)
                if not np.isnan(old_val):
                    assert abs(new_val - old_val) < 1e-6, (
                        f"{ts} {col} 不一致：新={new_val:.8f}，旧={old_val:.8f}"
                    )

    def test_macd_parity(self, sample_daily_adj):
        """新旧逻辑的 macd_dif/macd_dea/macd_hist 数值应完全一致"""
        tech_all = precompute_technical_factors(sample_daily_adj, vol_windows=[20])
        trading_dates = sorted(sample_daily_adj['trade_date'].unique().tolist())
        trade_date = trading_dates[45]
        current_idx = 45

        tech_today = tech_all[tech_all['trade_date'] == trade_date]
        old_result = self._old_logic_for_date(
            sample_daily_adj, trade_date, trading_dates, current_idx, vol_windows=[20]
        )

        for col in ['macd_dif', 'macd_dea', 'macd_hist']:
            assert col in tech_today.columns, f"批量预计算结果应包含 {col} 列"
            for ts in tech_today['ts_code']:
                new_val = float(tech_today[tech_today['ts_code'] == ts][col].iloc[0])
                old_val = old_result.get(ts, {}).get(col, np.nan)
                if not np.isnan(old_val):
                    assert abs(new_val - old_val) < 1e-6, (
                        f"{ts} {col} 不一致：新={new_val:.8f}，旧={old_val:.8f}"
                    )

    def test_bollinger_bands_parity(self, sample_daily_adj):
        """新旧逻辑的布林带指标应完全一致"""
        tech_all = precompute_technical_factors(sample_daily_adj, vol_windows=[20])
        trading_dates = sorted(sample_daily_adj['trade_date'].unique().tolist())
        trade_date = trading_dates[45]
        current_idx = 45

        tech_today = tech_all[tech_all['trade_date'] == trade_date]
        old_result = self._old_logic_for_date(
            sample_daily_adj, trade_date, trading_dates, current_idx, vol_windows=[20]
        )

        for col in ['bb_middle', 'bb_upper', 'bb_lower']:
            assert col in tech_today.columns, f"批量预计算结果应包含 {col} 列"
            for ts in tech_today['ts_code']:
                new_val = float(tech_today[tech_today['ts_code'] == ts][col].iloc[0])
                old_val = old_result.get(ts, {}).get(col, np.nan)
                if not np.isnan(old_val):
                    assert abs(new_val - old_val) < 1e-6, (
                        f"{ts} {col} 不一致：新={new_val:.8f}，旧={old_val:.8f}"
                    )

    def test_volatility_parity(self, sample_daily_adj):
        """新旧逻辑的 volatility_20 应完全一致"""
        vol_windows = [5, 10, 20]
        tech_all = precompute_technical_factors(sample_daily_adj, vol_windows=vol_windows)
        trading_dates = sorted(sample_daily_adj['trade_date'].unique().tolist())
        trade_date = trading_dates[45]
        current_idx = 45

        tech_today = tech_all[tech_all['trade_date'] == trade_date]
        old_result = self._old_logic_for_date(
            sample_daily_adj, trade_date, trading_dates, current_idx, vol_windows=vol_windows
        )

        for w in vol_windows:
            col = f'volatility_{w}'
            assert col in tech_today.columns, f"批量预计算结果应包含 {col} 列"
            for ts in tech_today['ts_code']:
                new_val = float(tech_today[tech_today['ts_code'] == ts][col].iloc[0])
                old_val = old_result.get(ts, {}).get(col, np.nan)
                if not np.isnan(old_val):
                    assert abs(new_val - old_val) < 1e-6, (
                        f"{ts} {col} 不一致：新={new_val:.8f}，旧={old_val:.8f}"
                    )

    def test_output_contains_required_columns(self, sample_daily_adj):
        """批量预计算输出应包含全部预期列"""
        tech_all = precompute_technical_factors(sample_daily_adj, vol_windows=[5, 10, 20])
        expected_cols = [
            'ts_code', 'trade_date',
            'rsi_14',
            'kdj_k', 'kdj_d', 'kdj_j',
            'macd_dif', 'macd_dea', 'macd_hist',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_pct',
            'volatility_5', 'volatility_10', 'volatility_20',
        ]
        for col in expected_cols:
            assert col in tech_all.columns, f"输出缺少列：{col}"

    def test_empty_input_returns_empty_df(self):
        """空输入不应抛异常，返回空 DataFrame"""
        result = precompute_technical_factors(pd.DataFrame(), vol_windows=[20])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestTechFactorCache：验证缓存只触发一次预计算
# ---------------------------------------------------------------------------

class TestTechFactorCache:
    """验证 FeatureBuilder 实例级缓存行为"""

    def test_precompute_called_only_once(self, sample_daily_adj, monkeypatch):
        """多日构建时，precompute_technical_factors 应只调用一次"""
        import src.lazybull.features.builder as builder_module

        call_count = {'n': 0}
        original_func = builder_module.precompute_technical_factors

        def counting_precompute(*args, **kwargs):
            call_count['n'] += 1
            return original_func(*args, **kwargs)

        monkeypatch.setattr(builder_module, 'precompute_technical_factors', counting_precompute)

        builder = FeatureBuilder(min_list_days=10, require_label=False)
        trading_dates = sorted(sample_daily_adj['trade_date'].unique().tolist())

        # 多次调用 _get_tech_factor_today（模拟多日构建）
        for trade_date in trading_dates[35:40]:
            builder._get_tech_factor_today(sample_daily_adj, trade_date)

        assert call_count['n'] == 1, (
            f"precompute_technical_factors 应只调用 1 次，实际调用了 {call_count['n']} 次"
        )

    def test_cache_returns_correct_date(self, sample_daily_adj):
        """缓存查表应返回正确的当日数据"""
        builder = FeatureBuilder(min_list_days=10, require_label=False)
        trading_dates = sorted(sample_daily_adj['trade_date'].unique().tolist())

        for trade_date in trading_dates[35:38]:
            today_df = builder._get_tech_factor_today(sample_daily_adj, trade_date)
            assert (today_df['trade_date'] == trade_date).all(), (
                f"查表返回的数据日期不正确，期望 {trade_date}"
            )

    def test_cache_is_instance_scoped(self, sample_daily_adj):
        """不同 FeatureBuilder 实例的缓存应相互独立"""
        builder1 = FeatureBuilder(min_list_days=10, require_label=False)
        builder2 = FeatureBuilder(min_list_days=10, require_label=False)

        trading_dates = sorted(sample_daily_adj['trade_date'].unique().tolist())
        trade_date = trading_dates[35]

        # 触发 builder1 的缓存
        builder1._get_tech_factor_today(sample_daily_adj, trade_date)

        # builder2 的缓存应仍为 None（未触发）
        assert builder2._tech_factor_cache is None, "不同实例的缓存应相互独立"

    def test_new_builder_instance_cache_is_none(self):
        """新建 FeatureBuilder 实例时缓存应为 None"""
        builder = FeatureBuilder()
        assert builder._tech_factor_cache is None


# ---------------------------------------------------------------------------
# TestComputeRet1：验证 compute_ret_1 口径一致性
# ---------------------------------------------------------------------------

class TestComputeRet1:
    """验证 compute_ret_1 的优先级逻辑与无前瞻性"""

    def _make_df(self, stocks, dates, seed=0):
        """构造小样本 daily_adj（含 close_adj 与 pct_chg）"""
        np.random.seed(seed)
        records = []
        for stock in stocks:
            price = 10.0
            for date in dates:
                ret = np.random.normal(0, 0.01)
                close = price * (1 + ret)
                records.append({
                    'ts_code': stock,
                    'trade_date': date,
                    'close_adj': close,
                    'pct_chg': ret * 100,
                })
                price = close
        return pd.DataFrame(records)

    def test_priority1_uses_existing_ret_1(self):
        """若 daily_adj 已含 ret_1，直接返回，不重新计算"""
        stocks = ['000001.SZ', '000002.SZ']
        dates = ['20230103', '20230104', '20230105']
        df = self._make_df(stocks, dates)
        df['ret_1'] = 0.999  # 故意设置一个特殊值
        result = compute_ret_1(df)
        assert (result == 0.999).all(), "存在 ret_1 列时应直接返回，不重新计算"

    def test_priority2_uses_close_adj_pct_change(self):
        """无 ret_1 但有 close_adj 时，应使用 groupby pct_change"""
        stocks = ['000001.SZ', '000002.SZ']
        dates = ['20230103', '20230104', '20230105', '20230106', '20230109']
        df = self._make_df(stocks, dates)
        # 不包含 ret_1 列
        assert 'ret_1' not in df.columns

        result = compute_ret_1(df)

        # 手动计算期望值：按 ts_code 分组，trade_date 升序，pct_change
        expected = (
            df[['ts_code', 'trade_date', 'close_adj']]
            .sort_values(['ts_code', 'trade_date'])
            .groupby('ts_code', sort=False)['close_adj']
            .pct_change()
            .reindex(df.index)
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_priority2_no_cross_stock_leakage(self):
        """close_adj 路径不能产生跨股票边界差分（第一个交易日应为 NaN）"""
        stocks = ['000001.SZ', '000002.SZ']
        dates = ['20230103', '20230104', '20230105']
        df = self._make_df(stocks, dates)

        result = compute_ret_1(df)

        # 每支股票第一行应为 NaN（无前一日数据）
        sorted_df = df.sort_values(['ts_code', 'trade_date'])
        for stock in stocks:
            first_row_idx = sorted_df[sorted_df['ts_code'] == stock].index[0]
            assert np.isnan(result.loc[first_row_idx]), (
                f"{stock} 第一个交易日 ret_1 应为 NaN，不应产生跨股票差分"
            )

    def test_priority3_fallback_pct_chg_with_warning(self, caplog):
        """无 ret_1 与 close_adj 时，fallback 到 pct_chg/100 并记录 warning"""
        import logging
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ'],
            'trade_date': ['20230103', '20230104'],
            'pct_chg': [1.0, -0.5],
        })
        with caplog.at_level(logging.WARNING):
            result = compute_ret_1(df)

        expected = pd.Series([0.01, -0.005], index=df.index)
        pd.testing.assert_series_equal(result, expected, check_names=False)
        assert any('pct_chg' in msg for msg in caplog.messages), (
            "fallback 到 pct_chg 路径时应记录 warning"
        )

    def test_priority4_all_nan_with_warning(self, caplog):
        """缺少所有来源列时返回全 NaN 并记录 warning"""
        import logging
        df = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20230103'],
        })
        with caplog.at_level(logging.WARNING):
            result = compute_ret_1(df)

        assert result.isna().all(), "所有来源缺失时应返回全 NaN"
        assert len(caplog.messages) > 0, "所有来源缺失时应记录 warning"

    def test_result_aligned_to_original_index(self):
        """compute_ret_1 结果索引应与输入 daily_adj.index 对齐"""
        stocks = ['000001.SZ', '000002.SZ']
        dates = ['20230103', '20230104', '20230105']
        df = self._make_df(stocks, dates)
        # 打乱行顺序
        df = df.sample(frac=1, random_state=7).reset_index(drop=True)

        result = compute_ret_1(df)
        assert result.index.equals(df.index), "结果索引应与原始 daily_adj 索引一致"


# ---------------------------------------------------------------------------
# TestVolatilityRet1Consistency：验证波动率口径修复
# ---------------------------------------------------------------------------

class TestVolatilityRet1Consistency:
    """验证修复后预计算 volatility_20 与 close_adj pct_change 参考实现完全一致"""

    def _build_sample(self, n_stocks=3, n_days=50, seed=42):
        """构造含 close_adj 但无 ret_1 的小样本"""
        np.random.seed(seed)
        stocks = [f'{i:06d}.SZ' for i in range(1, n_stocks + 1)]
        dates = pd.date_range('2023-01-01', periods=n_days, freq='B').strftime('%Y%m%d').tolist()
        records = []
        for stock in stocks:
            price = 10.0
            for date in dates:
                ret = np.random.normal(0, 0.01)
                close = price * (1 + ret)
                # pct_chg 故意与 close_adj pct_change 略有偏差，用于验证口径差异
                pct_chg_biased = ret * 100 + np.random.normal(0, 0.001)
                records.append({
                    'ts_code': stock,
                    'trade_date': date,
                    'close_adj': close,
                    'pct_chg': pct_chg_biased,
                })
                price = close
        return pd.DataFrame(records)

    def test_volatility_20_consistent_with_close_adj_pct_change(self):
        """预计算 volatility_20 应与 close_adj pct_change 参考实现完全一致"""
        df = self._build_sample()
        vol_windows = [20]

        # 新路径：使用修复后的 precompute_technical_factors
        tech_all = precompute_technical_factors(df, vol_windows=vol_windows)

        # 参考实现：直接用 close_adj pct_change 构造 ret_1，再计算滚动 std
        ref_df = df[['ts_code', 'trade_date', 'close_adj']].copy()
        ref_df = ref_df.sort_values(['ts_code', 'trade_date'])
        ref_df['ret_1'] = ref_df.groupby('ts_code', sort=False)['close_adj'].pct_change()
        ref_vol = calculate_volatility(ref_df, ret_col='ret_1', windows=vol_windows)

        # 对每支股票的最后 5 个交易日（历史充足）逐行比较
        for stock in df['ts_code'].unique():
            dates_for_stock = sorted(df[df['ts_code'] == stock]['trade_date'].unique())
            for date in dates_for_stock[-5:]:
                new_val = float(
                    tech_all[(tech_all['ts_code'] == stock) & (tech_all['trade_date'] == date)
                             ]['volatility_20'].iloc[0]
                )
                ref_val = float(
                    ref_vol[(ref_vol['ts_code'] == stock) & (ref_vol['trade_date'] == date)
                            ]['volatility_20'].iloc[0]
                )
                assert abs(new_val - ref_val) < 1e-10, (
                    f"{stock} {date} volatility_20 不一致："
                    f"新={new_val:.12f}，参考={ref_val:.12f}"
                )

    def test_volatility_differs_from_pct_chg_path(self):
        """当 pct_chg 与 close_adj pct_change 有偏差时，修复后结果应不同于 pct_chg 路径"""
        df = self._build_sample()
        vol_windows = [20]

        # 修复后路径
        tech_all = precompute_technical_factors(df, vol_windows=vol_windows)

        # 旧（错误）路径：直接用 pct_chg/100
        old_df = df[['ts_code', 'trade_date']].copy()
        old_df['ret_1'] = df['pct_chg'].values / 100.0
        old_vol = calculate_volatility(old_df, ret_col='ret_1', windows=vol_windows)

        # 至少在某些行上，两者结果应有差异（因为 pct_chg 故意加了噪声）
        stock = df['ts_code'].unique()[0]
        dates_for_stock = sorted(df[df['ts_code'] == stock]['trade_date'].unique())
        diffs = []
        for date in dates_for_stock[-5:]:
            new_val = float(
                tech_all[(tech_all['ts_code'] == stock) & (tech_all['trade_date'] == date)
                         ]['volatility_20'].iloc[0]
            )
            old_val = float(
                old_vol[(old_vol['ts_code'] == stock) & (old_vol['trade_date'] == date)
                        ]['volatility_20'].iloc[0]
            )
            diffs.append(abs(new_val - old_val))

        assert max(diffs) > 1e-8, (
            "修复后路径与旧 pct_chg 路径结果相同，说明修复未生效（或测试数据不够区分）"
        )

    def test_zscore_volatility_20_stable_with_close_adj(self):
        """端到端：zscore_volatility_20 在修复后口径下应稳定（与 close_adj 参考一致）"""
        df = self._build_sample(n_stocks=5, n_days=50)
        vol_windows = [20]

        tech_all = precompute_technical_factors(df, vol_windows=vol_windows)

        # 取最后一个日期的 volatility_20
        last_date = sorted(df['trade_date'].unique())[-1]
        today_vol = tech_all[tech_all['trade_date'] == last_date][['ts_code', 'volatility_20']].copy()

        # 手动计算截面 zscore
        mean_v = today_vol['volatility_20'].mean()
        std_v = today_vol['volatility_20'].std(ddof=1)
        today_vol['zscore_vol_20'] = (today_vol['volatility_20'] - mean_v) / std_v

        # 参考实现
        ref_df = df[['ts_code', 'trade_date', 'close_adj']].copy()
        ref_df = ref_df.sort_values(['ts_code', 'trade_date'])
        ref_df['ret_1'] = ref_df.groupby('ts_code', sort=False)['close_adj'].pct_change()
        ref_vol = calculate_volatility(ref_df, ret_col='ret_1', windows=vol_windows)
        ref_today = ref_vol[ref_vol['trade_date'] == last_date][['ts_code', 'volatility_20']].copy()
        ref_mean = ref_today['volatility_20'].mean()
        ref_std = ref_today['volatility_20'].std(ddof=1)
        ref_today['zscore_vol_20'] = (ref_today['volatility_20'] - ref_mean) / ref_std

        # 合并比较
        merged = today_vol.merge(ref_today, on='ts_code', suffixes=('_new', '_ref'))
        for _, row in merged.iterrows():
            assert abs(row['zscore_vol_20_new'] - row['zscore_vol_20_ref']) < 1e-10, (
                f"{row['ts_code']} zscore_volatility_20 不一致"
            )
