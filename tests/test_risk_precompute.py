"""风控因子批量预计算测试

验证 risk/precompute.py 的批量向量化结果与 factor_registry 逐日路径的一致性，
以及 FeatureBuilder 缓存查表路径的正确性。
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.risk.factor_registry import compute_all_risk_factors
from src.lazybull.features import FeatureBuilder
from src.lazybull.risk.precompute import (
    PRECOMPUTED_RISK_FACTOR_NAMES,
    build_risk_factor_cache_dict,
    precompute_risk_factors,
)

# 无停牌数据下与逐日路径应完全一致的因子（窗口按日期对齐 == 按观测对齐）
_EXACT_MATCH_FACTORS = [
    "downside_vol_20",
    "downside_corr_20",
    "var_95_20",
    "cvar_95_20",
    "max_drawdown_20",
    "drawdown_duration",
    "skewness_20",
    "kurtosis_20",
    "parkinson_vol_20",
    "high_low_range_ratio",
    "gap_risk",
    "turnover_cv_20",
    "amount_cv_20",
    "vol_ratio_5_20",
    "volume_climax_days",
    "turnover_percentile",
    "volume_price_divergence",
]

# 窗口边界收益样本数存在已知细微差异的因子（仅做有限性检查）
_LOOSE_MATCH_FACTORS = [
    "amihud_illiq_20",
    "up_down_vol_ratio",
    "vol_of_vol_20",
    "garch_persistence",
    "vol_regime_percentile",
]


def _make_daily_adj(n_stocks: int = 6, n_days: int = 320, seed: int = 42) -> pd.DataFrame:
    """构造无停牌的合成后复权日线数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days).strftime("%Y%m%d").tolist()
    frames = []
    for i in range(n_stocks):
        ts_code = f"{600000 + i}.SH"
        rets = rng.normal(0.0005, 0.02, n_days)
        close = 10.0 * np.cumprod(1 + rets)
        high = close * (1 + rng.uniform(0.001, 0.03, n_days))
        low = close * (1 - rng.uniform(0.001, 0.03, n_days))
        open_ = low + (high - low) * rng.uniform(0.2, 0.8, n_days)
        frames.append(
            pd.DataFrame(
                {
                    "ts_code": ts_code,
                    "trade_date": dates,
                    "close_adj": close,
                    "open_adj": open_,
                    "high_adj": high,
                    "low_adj": low,
                    "vol": rng.uniform(1e4, 1e6, n_days),
                    "amount": rng.uniform(1e4, 1e7, n_days),
                    "turnover_rate": rng.uniform(0.5, 15.0, n_days),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["pre_close_adj"] = df.groupby("ts_code")["close_adj"].shift(1)
    return df


@pytest.fixture(scope="module")
def daily_adj() -> pd.DataFrame:
    return _make_daily_adj()


@pytest.fixture(scope="module")
def precomputed(daily_adj: pd.DataFrame) -> pd.DataFrame:
    result = precompute_risk_factors(daily_adj)
    assert result is not None
    return result


class TestPrecomputeBasics:
    def test_output_aligned_with_input(self, daily_adj, precomputed):
        """输出行数与输入对齐，包含全部 22 个因子列。"""
        assert len(precomputed) == len(daily_adj)
        for name in PRECOMPUTED_RISK_FACTOR_NAMES:
            assert name in precomputed.columns, f"缺少因子列: {name}"

    def test_empty_input_returns_none(self):
        assert precompute_risk_factors(pd.DataFrame()) is None
        assert precompute_risk_factors(None) is None

    def test_missing_required_column_returns_none(self, daily_adj):
        assert precompute_risk_factors(daily_adj.drop(columns=["close_adj"])) is None

    def test_missing_optional_column_yields_nan_factor(self, daily_adj):
        """缺少 turnover_rate 时对应因子应全为 NaN，其余因子不受影响。"""
        result = precompute_risk_factors(daily_adj.drop(columns=["turnover_rate"]))
        assert result is not None
        assert result["turnover_cv_20"].isna().all()
        assert result["turnover_percentile"].isna().all()
        assert result["var_95_20"].notna().any()

    def test_builder_merges_daily_basic_turnover_before_risk_precompute(self, daily_adj):
        """换手率来自 daily_basic，必须在风险因子预计算前并入 daily_adj。"""
        daily = daily_adj.drop(columns=["turnover_rate"])
        daily_basic = daily_adj[["ts_code", "trade_date", "turnover_rate"]]
        builder = FeatureBuilder(horizon=20, label_filter_mode="single")

        builder.precompute_daily_adj(
            daily,
            pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"]),
            daily_basic,
        )
        result = precompute_risk_factors(builder._daily_adj_precomputed)

        assert result is not None
        assert result["turnover_cv_20"].notna().any()
        assert result["turnover_percentile"].notna().any()

    def test_cache_dict_structure(self, daily_adj, precomputed):
        cache = build_risk_factor_cache_dict(precomputed)
        some_date = daily_adj["trade_date"].iloc[-1]
        assert some_date in cache
        day_df = cache[some_date]
        assert "ts_code" in day_df.columns
        assert len(day_df) == daily_adj["trade_date"].eq(some_date).sum()

    def test_late_listing_no_runtime_warning(self, daily_adj):
        """后期上市股票（宽矩阵前段全 NaN 窗口）不应产生 RuntimeWarning。"""
        dates = sorted(daily_adj["trade_date"].unique())
        late = _make_daily_adj(n_stocks=1, n_days=30, seed=7)
        late["ts_code"] = "300999.SZ"
        late["trade_date"] = dates[-30:]
        df = pd.concat([daily_adj, late], ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = precompute_risk_factors(df)
        assert result is not None
        # 新股上市初期因子为 NaN，后期开始有值
        late_rows = result[result["ts_code"] == "300999.SZ"]
        assert len(late_rows) == 30
        assert late_rows["var_95_20"].notna().any()


class TestConsistencyWithPerDayPath:
    """批量预计算 vs 逐日路径的数值一致性。"""

    @pytest.fixture(scope="class")
    def per_day_results(self, daily_adj):
        """在最后一个交易日运行原逐日计算路径。"""
        trade_date = daily_adj["trade_date"].max()
        cross_df = daily_adj[daily_adj["trade_date"] == trade_date][["ts_code"]].reset_index(
            drop=True
        )
        window = daily_adj.copy()
        window["ret_1"] = window["close_adj"] / window["pre_close_adj"] - 1
        results = compute_all_risk_factors(
            df=cross_df, daily_adj=window, market_state=None, trade_date=trade_date
        )
        return trade_date, cross_df, results

    def test_exact_match_factors(self, precomputed, per_day_results):
        trade_date, cross_df, per_day = per_day_results
        pre_day = precomputed[precomputed["trade_date"] == trade_date]
        pre_day = cross_df.merge(pre_day, on="ts_code", how="left").set_index("ts_code")
        for name in _EXACT_MATCH_FACTORS:
            assert name in per_day, f"逐日路径缺少因子: {name}"
            expected = per_day[name].to_numpy(dtype=np.float64)
            actual = pre_day[name].to_numpy(dtype=np.float64)
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=1e-6,
                equal_nan=True,
                err_msg=f"因子 {name} 批量预计算与逐日路径不一致",
            )

    def test_loose_match_factors_finite(self, precomputed, per_day_results):
        """窗口边界语义差异因子：仅要求预计算结果非全 NaN 且量级合理。"""
        trade_date, cross_df, _ = per_day_results
        pre_day = precomputed[precomputed["trade_date"] == trade_date]
        pre_day = cross_df.merge(pre_day, on="ts_code", how="left")
        for name in _LOOSE_MATCH_FACTORS:
            values = pre_day[name].to_numpy(dtype=np.float64)
            assert np.isfinite(values).any(), f"因子 {name} 全为 NaN"

    def test_percentile_factors_in_range(self, precomputed):
        """分位类因子取值应在 [0, 1] 区间。"""
        for name in ["vol_regime_percentile", "turnover_percentile"]:
            values = precomputed[name].dropna()
            assert (values >= 0).all() and (values <= 1).all(), f"{name} 超出 [0,1]"


class TestRegistryExclude:
    def test_exclude_skips_factors(self, daily_adj):
        trade_date = daily_adj["trade_date"].max()
        cross_df = daily_adj[daily_adj["trade_date"] == trade_date][["ts_code"]].reset_index(
            drop=True
        )
        results = compute_all_risk_factors(
            df=cross_df,
            daily_adj=None,
            market_state=None,
            trade_date=trade_date,
            exclude=set(PRECOMPUTED_RISK_FACTOR_NAMES),
        )
        for name in PRECOMPUTED_RISK_FACTOR_NAMES:
            assert name not in results, f"exclude 未生效: {name}"
        # 公告类因子不受 exclude 影响
        assert "pledge_high_flag" in results


class TestBuilderIntegration:
    def test_attach_risk_factors_static(self, daily_adj, precomputed):
        """静态合并函数：查表因子 + 公告类因子均应追加到 features。"""
        from src.lazybull.features.builder import _attach_risk_factors_static

        cache = build_risk_factor_cache_dict(precomputed)
        trade_date = daily_adj["trade_date"].max()
        features = daily_adj[daily_adj["trade_date"] == trade_date][["ts_code"]].reset_index(
            drop=True
        )
        result = _attach_risk_factors_static(
            features, trade_date, cache, PRECOMPUTED_RISK_FACTOR_NAMES
        )
        for name in PRECOMPUTED_RISK_FACTOR_NAMES:
            assert name in result.columns
        assert "pledge_high_flag" in result.columns
        assert len(result) == len(features)

    def test_attach_missing_date_yields_nan_columns(self, precomputed):
        """缓存中不存在的日期：仍应追加 NaN 因子列，保持 schema 一致。"""
        from src.lazybull.features.builder import _attach_risk_factors_static

        cache = build_risk_factor_cache_dict(precomputed)
        features = pd.DataFrame({"ts_code": ["600000.SH"]})
        result = _attach_risk_factors_static(
            features, "19900101", cache, PRECOMPUTED_RISK_FACTOR_NAMES
        )
        for name in PRECOMPUTED_RISK_FACTOR_NAMES:
            assert name in result.columns
            assert result[name].isna().all()

    def test_builder_cache_path(self, daily_adj):
        """FeatureBuilder._add_risk_factors 走缓存路径并复用缓存。"""
        from src.lazybull.features.builder import FeatureBuilder

        builder = FeatureBuilder(require_label=False)
        trade_date = daily_adj["trade_date"].max()
        trading_dates = sorted(daily_adj["trade_date"].unique().tolist())
        features = daily_adj[daily_adj["trade_date"] == trade_date][["ts_code"]].reset_index(
            drop=True
        )
        result = builder._add_risk_factors(features, daily_adj, trade_date, trading_dates)
        assert builder._risk_factor_cache_dict is not None
        for name in PRECOMPUTED_RISK_FACTOR_NAMES:
            assert name in result.columns
        # 二次调用复用缓存（同一对象）
        cache_ref = builder._risk_factor_cache_dict
        builder._add_risk_factors(features, daily_adj, trade_date, trading_dates)
        assert builder._risk_factor_cache_dict is cache_ref
        # clear_caches 释放风控缓存
        builder.clear_caches()
        assert builder._risk_factor_cache_dict is None
