# -*- coding: utf-8 -*-
"""审计修复回归测试：串行/并行一致性、缺失复权因子、缺失值语义、推理质量门禁。

对应 2026-08-08 代码审计的 4 个修复：
- 问题1：并行路径基本面代理回填顺序与串行路径不一致
- 问题2：离线/在线缺失复权因子处理相反
- 问题3：缺失 dv_ttm/pe_ttm 被编码成真实经济含义
- 问题5：模型推理缺少数值质量门禁
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.lazybull.data.build_clean import build_clean_data
from src.lazybull.features.builder.static_extra import _add_value_dividend_features_static
from src.lazybull.features.context import FeatureContext
from src.lazybull.features.parallel import build_features_for_day_static
from src.lazybull.signals.ml_signal import MLSignal


# ── 问题1：串行/并行回填顺序一致 ──────────────────────────────


def _make_trading_dates(n: int = 35) -> list:
    """生成 n 个递增交易日字符串（YYYYMMDD）。"""
    import datetime

    start = datetime.date(2023, 1, 2)
    dates = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:  # 跳过周末
            dates.append(d.strftime("%Y%m%d"))
        d += datetime.timedelta(days=1)
    return dates


def _make_daily_adj_dict(trading_dates, n_stocks=3):
    """构造每日复权截面字典（含 clean 层标记，避免 features 层重算）。"""
    frames = {}
    for d in trading_dates:
        n = n_stocks
        df = pd.DataFrame(
            {
                "ts_code": [f"{i:06d}.SZ" for i in range(n)],
                "trade_date": [d] * n,
                "close_adj": [10.0 + i for i in range(n)],
                "open_adj": [9.9 + i for i in range(n)],
                "high_adj": [10.2 + i for i in range(n)],
                "low_adj": [9.8 + i for i in range(n)],
                "open": [9.9 + i for i in range(n)],
                "high": [10.2 + i for i in range(n)],
                "low": [9.8 + i for i in range(n)],
                "pre_close": [9.8 + i for i in range(n)],
                "adj_factor": [1.0] * n,
                "vol": [1000.0] * n,
                "amount": [10000.0] * n,
                "pct_chg": [1.0] * n,
                # clean 层标记
                "is_st": [0] * n,
                "is_suspended": [0] * n,
                "is_limit_up": [0] * n,
                "is_limit_down": [0] * n,
                "tradable": [1] * n,
                "list_days": [1000] * n,
            }
        )
        frames[d] = df
    return frames


class _RegistryStub:
    """mock 因子处理器：仅模拟 handler 生成 q_ocf_to_sales / ocf_to_revenue 列。"""

    def apply_all(self, features, ctx, current_data):
        features = features.copy()
        features["q_ocf_to_sales"] = 0.5
        features["ocf_to_revenue"] = 0.6
        return features


def test_parallel_path_backfills_proxy_features_after_handlers():
    """并行路径须在因子处理器之后回填 cf_sales/cf_nm（与串行路径一致）。

    若回填发生在 apply_all 之前，handler 生成的 q_ocf_to_sales/ocf_to_revenue
    将无法回填 cf_sales，导致串行 cf_sales 有值、并行 NaN 的口径漂移。
    """
    trading_dates = _make_trading_dates(35)
    daily_adj_dict = _make_daily_adj_dict(trading_dates)
    target = trading_dates[28]
    current_data = daily_adj_dict[target]

    trade_cal = pd.DataFrame({"cal_date": trading_dates, "is_open": [1] * len(trading_dates)})
    stock_basic = pd.DataFrame(
        {
            "ts_code": [f"{i:06d}.SZ" for i in range(3)],
            "name": ["测试"] * 3,
            "list_date": ["20100101"] * 3,
        }
    )
    ctx = FeatureContext(
        trade_date=target,
        trade_cal=trade_cal,
        daily_data=current_data,
        adj_factor=pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"]),
        stock_basic=stock_basic,
        horizons=[5],
        lookback_windows=[5],
        require_label=False,
        min_list_days=0,
    )
    trading_date_index = {d: i for i, d in enumerate(trading_dates)}

    result = build_features_for_day_static(
        trade_date=target,
        ctx=ctx,
        daily_adj_dict=daily_adj_dict,
        tech_factor_cache_dict=None,
        market_state_cache=None,
        trading_dates_list=trading_dates,
        trading_date_index=trading_date_index,
        daily_adj_precomputed=None,
        factor_registry=_RegistryStub(),
    )

    assert result is not None
    assert "cf_sales" in result.columns
    # handler 生成的 q_ocf_to_sales=0.5 应回填到 cf_sales
    assert result["cf_sales"].notna().all()
    assert np.allclose(result["cf_sales"], 0.5)


# ── 问题2：缺失复权因子离线/在线统一保留 NaN ─────────────────


class _StubStorage:
    """build_clean_data 最小存储桩：trade_cal/stock_basic/daily 存在，adj_factor 缺失。"""

    def __init__(self, trade_cal, stock_basic, daily_raw):
        self._trade_cal = trade_cal
        self._stock_basic = stock_basic
        self._daily_raw = daily_raw
        self.saved = []

    def load_raw(self, name):
        if name == "trade_cal":
            return self._trade_cal
        if name == "stock_basic":
            return self._stock_basic
        return None

    def load_raw_by_date(self, name, trade_date):
        if name == "daily":
            return self._daily_raw.copy()
        if name == "adj_factor":
            return None  # 复权因子缺失
        return None

    def save_clean(self, *args, **kwargs):
        self.saved.append(("save_clean", args, kwargs))

    def save_clean_by_date(self, *args, **kwargs):
        self.saved.append(("save_clean_by_date", args, kwargs))

    def is_data_exists(self, *args, **kwargs):
        return False


def test_build_clean_missing_adj_factor_skips_daily_partition():
    """离线批量构建：复权因子整日缺失时不得生成 clean/daily。"""
    trade_cal_raw = pd.DataFrame(
        {"exchange": ["SSE", "SSE"], "cal_date": ["20230102", "20230103"], "is_open": [1, 1]}
    )
    stock_basic_raw = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "name": ["测试"], "list_date": ["20100101"]}
    )
    daily_raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20230102"],
            "close": [10.0],
            "open": [9.8],
            "high": [10.5],
            "low": [9.5],
            "vol": [1000000],
            "amount": [10000000],
            "pct_chg": [2.0],
        }
    )
    storage = _StubStorage(trade_cal_raw, stock_basic_raw, daily_raw)

    cleaner = Mock()
    cleaner.clean_trade_cal.return_value = pd.DataFrame({"cal_date": ["20230102"], "is_open": [1]})
    cleaner.clean_stock_basic.return_value = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "name": ["测试"]}
    )
    cleaner.clean_daily.side_effect = lambda daily, adj: daily.copy()
    cleaner.add_tradable_universe_flag.side_effect = lambda daily, **kw: daily

    build_clean_data(
        storage=storage,
        loader=Mock(),
        cleaner=cleaner,
        start_date="20230102",
        end_date="20230102",
        force=True,
        min_list_days=0,
    )

    cleaner.clean_daily.assert_not_called()
    assert not any(call[0] == "save_clean_by_date" for call in storage.saved)


# ── 问题3：缺失 dv_ttm/pe_ttm 保留 NaN + 显式缺失标记 ────────


def test_value_dividend_empty_daily_basic_returns_features():
    """daily_basic 单日整体缺失时返回原特征、不生成价值列（审计问题7，硬告警不熔断）。"""
    features = pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]})
    daily_basic = pd.DataFrame(
        columns=["ts_code", "trade_date", "pb", "pe_ttm", "dv_ttm", "total_mv"]
    )

    result = _add_value_dividend_features_static(features, daily_basic, "20230102")

    assert list(result.columns) == ["ts_code"]
    assert len(result) == 2


def test_value_dividend_empty_daily_basic_logs_error():
    """daily_basic 单日整体缺失时记录 error 级硬告警（审计问题7，评审问题5）。"""
    import io

    from loguru import logger

    sink = io.StringIO()
    handler_id = logger.add(sink, level="ERROR", format="{level}|{message}")
    try:
        features = pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]})
        daily_basic = pd.DataFrame(
            columns=["ts_code", "trade_date", "pb", "pe_ttm", "dv_ttm", "total_mv"]
        )
        _add_value_dividend_features_static(features, daily_basic, "20230102")
        logs = sink.getvalue()
    finally:
        logger.remove(handler_id)
    assert "ERROR" in logs
    assert "daily_basic" in logs
    assert "20230102" in logs


def test_value_dividend_missing_semantics_preserved():
    """缺失 dv_ttm/pe_ttm 不再被编码成"不分红/亏损"，另增显式缺失标记。"""
    features = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
        }
    )
    daily_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "trade_date": ["20230102"] * 4,
            "pe_ttm": [10.0, -5.0, np.nan, 20.0],
            "dv_ttm": [0.03, 0.0, np.nan, np.nan],
        }
    )

    result = _add_value_dividend_features_static(features, daily_basic, "20230102")

    # dv_ttm：缺失保留 NaN，不再 fillna(0)
    assert result.loc[0, "dv_ttm"] == pytest.approx(0.03)
    assert result.loc[1, "dv_ttm"] == pytest.approx(0.0)  # 真实不分红=0 保持 0
    assert np.isnan(result.loc[2, "dv_ttm"])
    assert result["dv_ttm_missing"].tolist() == [0, 0, 1, 1]

    # pe_ttm：缺失 is_loss=0（不再误判亏损），缺失单独标记
    assert result["is_loss"].tolist() == [0, 1, 0, 0]
    assert result["pe_ttm_missing"].tolist() == [0, 0, 1, 0]
    assert result.loc[0, "ep_ttm"] == pytest.approx(0.1)
    assert np.isnan(result.loc[2, "ep_ttm"])


# ── 问题5：推理数值质量门禁 ─────────────────────────────────


def test_check_feature_quality_rejects_only_all_nan():
    """仅全空列被拒绝；全零/常量/高缺失（可能是合法状态）仅警告不阻断。"""
    signal = MLSignal()
    X = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0, 4.0, 5.0],
            "all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "all_zero": [0.0, 0.0, 0.0, 0.0, 0.0],
            "constant": [5.0, 5.0, 5.0, 5.0, 5.0],
            "high_missing": [1.0, 2.0, np.nan, np.nan, np.nan],
        }
    )

    assert signal._check_feature_quality(X[["good"]]) is True
    assert signal._check_feature_quality(X[["good", "all_nan"]]) is False  # 全空拒绝
    assert signal._check_feature_quality(X[["good", "all_zero"]]) is True  # 全零仅警告
    assert signal._check_feature_quality(X[["good", "constant"]]) is True  # 常量仅警告
    # 60% 缺失 > 50% 阈值：仅警告，不拒绝
    assert signal._check_feature_quality(X[["good", "high_missing"]]) is True
    assert signal._check_feature_quality(pd.DataFrame()) is False


def test_check_feature_quality_accepts_market_state_broadcast():
    """市场环境特征（mkt_*）为单日常量广播列，截面内唯一值=1 不应被拒绝。"""
    signal = MLSignal()
    n = 50
    X = pd.DataFrame(
        {
            "mkt_adv_dec_ratio": [1.2] * n,
            "mkt_ret_avg_20": [0.005] * n,
            "mkt_turnover_std": [0.1] * n,
            "mkt_vol_20": [1.0e9] * n,
            "zscore_size": np.linspace(-2, 2, n),
        }
    )
    assert signal._check_feature_quality(X) is True


def _make_ready_signal():
    """构造跳过模型加载的 MLSignal 实例。"""
    signal = MLSignal()
    signal.model_version = 1
    signal.model = Mock()
    signal.model.predict.return_value = np.array([0.1, 0.2, 0.3])
    signal.metadata = {
        "feature_columns": ["feat_a", "feat_b"],
        "version_str": "v1",
        "feature_count": 2,
        "train_params": {"task": "regression"},
    }
    signal.feature_columns = ["feat_a", "feat_b"]
    signal.registry = Mock()
    signal.registry.check_feature_consistency.return_value = None
    return signal


def _make_features_df(feat_a_values, feat_b_values):
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "amount_ma20": [100000.0] * 3,
            "total_mv": [1000000.0] * 3,
            "feat_a": feat_a_values,
            "feat_b": feat_b_values,
        }
    )


def test_generate_all_zero_column_warns_but_predicts():
    """generate：全零特征列（可能为合法状态，如全部不分红）仅警告，不阻断预测。"""
    signal = _make_ready_signal()
    universe = ["000001.SZ", "000002.SZ", "000003.SZ"]
    data = {"features": _make_features_df([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])}

    with patch.object(MLSignal, "_load_model", return_value=None):
        result = signal.generate(pd.Timestamp("2023-01-02"), universe, data)

    assert len(result) > 0


def test_generate_ranked_rejects_all_nan_column():
    """generate_ranked：全空特征列（数据完全缺失）应拒绝预测并返回空列表。"""
    signal = _make_ready_signal()
    universe = ["000001.SZ", "000002.SZ", "000003.SZ"]
    data = {"features": _make_features_df([np.nan, np.nan, np.nan], [1.0, 2.0, 3.0])}

    with patch.object(MLSignal, "_load_model", return_value=None):
        result = signal.generate_ranked(pd.Timestamp("2023-01-02"), universe, data)

    assert result == []


def test_generate_passes_quality_gate_with_good_data():
    """generate：数值质量正常时照常预测（回归护栏）。"""
    signal = _make_ready_signal()
    universe = ["000001.SZ", "000002.SZ", "000003.SZ"]
    data = {"features": _make_features_df([1.0, 2.0, 3.0], [0.5, 0.6, 0.7])}

    with patch.object(MLSignal, "_load_model", return_value=None):
        result = signal.generate(pd.Timestamp("2023-01-02"), universe, data)

    assert len(result) > 0
