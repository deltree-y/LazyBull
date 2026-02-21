"""测试技术指标与波动率批量预计算功能

验证：
1. 小样本数据上"旧逻辑（按日切片计算）"与"新逻辑（一次预计算后按日取值）"结果一致
2. 缓存确实生效：多日构建时预计算函数只调用一次
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.lazybull.factors.precompute_technical_factors import precompute_technical_factors
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
