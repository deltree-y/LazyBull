#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行时可用性标记（factors/availability.py）专项测试。"""

from typing import List

import numpy as np
import pandas as pd

from src.lazybull.factors.availability import (
    AVAILABILITY_MARKER_SOURCES,
    availability_marker_names,
    derive_availability_markers,
    ensure_availability_markers,
)
from src.lazybull.ml import ModelRegistry
from src.lazybull.ml.train_core import prepare_training_data
from src.lazybull.signals import MLSignal


def _make_frame() -> pd.DataFrame:
    """构造含结构性缺失的合成截面（4 只股票 × 2 日）。"""
    return pd.DataFrame(
        {
            "trade_date": ["20240102"] * 4 + ["20240103"] * 4,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"] * 2,
            # 一致预期：仅前两只股票有覆盖
            "cons_analyst_count_30d": [5.0, 3.0, np.nan, np.nan] * 2,
            "cons_eps_yield_fy1": [0.05, 0.04, np.nan, np.nan] * 2,
            "cons_eps_yield_fy2": [np.nan, 0.03, np.nan, np.nan] * 2,
            # 两融：仅后两只股票为标的
            "rzye_chg_5": [np.nan, np.nan, 0.01, -0.02] * 2,
            # 基金持仓：仅第一只有持仓
            "fund_hold_ratio": [0.12, np.nan, np.nan, np.nan] * 2,
        }
    )


# prepare.py 的基础特征列（缺失会 KeyError，测试框架必须全部提供；镜像生产 schema 风格）
_PREPARE_BASE_FEATURE_COLUMNS = [
    "neu_ret_1",
    "neu_ret_20",
    "neu_ret_5",
    "alpha_industry_20",
    "alpha_industry_5",
    "ind_ret_avg",
    "ind_momentum_rank",
    "zscore_ma_deviation_20",
    "zscore_acceleration",
    "zscore_macd_hist",
    "bb_pct",
    "zscore_turnover_rate",
    "vol_ratio_20",
    "vol_burst_20",
    "zscore_amount_ma20",
    "zscore_net_mf_amount",
    "zscore_elg_net_amount_sum_20",
    "lg_net_amount_sum_5",
    "zscore_volatility_20",
    "zscore_volatility_5",
    "amplitude",
    "zscore_bb_width",
    "upper_shadow",
    "lower_shadow",
    "spec_score",
    "rsi_14",
    "kdj_j",
    "zscore_size",
    "zscore_bp",
    "zscore_dv_ttm",
    "zscore_pe_ttm",
    "is_loss",
    "dv_ttm_missing",
    "pe_ttm_missing",
    "list_days",
    "mkt_adv_dec_ratio",
    "mkt_ret_avg_20",
    "mkt_turnover_std",
    "mkt_vol_20",
]


def _make_prepare_frame(n_dates: int = 60, stocks_per_date: int = 4) -> pd.DataFrame:
    """构造 prepare_training_data 可用框架（含完整基础特征列 + 结构性缺失来源列）。"""
    rows = []
    for date_idx in range(n_dates):
        trade_date = f"2024{102 + date_idx:04d}"
        for stock_idx in range(stocks_per_date):
            row = {
                "trade_date": trade_date,
                "ts_code": f"{stock_idx:06d}.SZ",
                "name": f"S{stock_idx:06d}",
                "industry": "测试行业",
                "list_date": "20100101",
                "is_st": 0,
                "is_suspended": 0,
                "is_limit_up": 0,
                "is_limit_down": 0,
                # 结构性缺失来源列：前两只股票有研报覆盖、后两只为两融标的
                "cons_analyst_count_30d": np.nan if stock_idx >= 2 else 5.0 + stock_idx,
                "cons_eps_yield_fy1": np.nan if stock_idx >= 2 else 0.04,
                "rzye_chg_5": np.nan if stock_idx < 2 else 0.01 * (stock_idx + 1),
                "fund_hold_ratio": 0.1 if stock_idx == 0 else np.nan,
            }
            for feat_idx, feat_col in enumerate(_PREPARE_BASE_FEATURE_COLUMNS):
                row[feat_col] = float(date_idx * 0.01 + stock_idx * 0.001 + feat_idx * 1e-6)
            rows.append(row)
    return pd.DataFrame(rows)


def test_marker_names_stable_order():
    assert availability_marker_names() == list(AVAILABILITY_MARKER_SOURCES.keys())


def test_derive_markers_any_notna_semantics():
    """标记 = 任一来源列非空（不是全部非空）。"""
    frame = _make_frame()

    added = derive_availability_markers(frame)

    assert added == ["has_cons_coverage", "has_margin_balance", "has_fund_holding"]
    assert "has_express_data" not in added  # 来源列缺失 → 跳过，不生成常量列
    assert "has_express_data" not in frame.columns

    expect_cons = [1, 1, 0, 0] * 2  # fy2 缺但 fy1 有 → 仍算有覆盖
    expect_margin = [0, 0, 1, 1] * 2
    expect_fund = [1, 0, 0, 0] * 2
    assert frame["has_cons_coverage"].tolist() == expect_cons
    assert frame["has_margin_balance"].tolist() == expect_margin
    assert frame["has_fund_holding"].tolist() == expect_fund
    for name in added:
        assert frame[name].dtype == np.int8


def test_derive_markers_wanted_filter_only_requested():
    """推理侧只补模型 feature_columns 里出现过的标记。"""
    frame = _make_frame()

    added = ensure_availability_markers(frame, ["has_margin_balance", "other_feature"])

    assert added == ["has_margin_balance"]
    assert "has_cons_coverage" not in frame.columns


def test_derive_markers_skips_all_columns_missing():
    """来源列全部缺失时不得静默产出常量零列。"""
    frame = pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"]})

    added = derive_availability_markers(frame)

    assert added == []
    assert list(frame.columns) == ["trade_date", "ts_code"]


def test_derive_markers_keeps_existing_column_values():
    """已存在的标记列不重算、不覆盖（幂等）。"""
    frame = _make_frame()
    frame["has_cons_coverage"] = 7

    added = derive_availability_markers(frame)

    assert "has_cons_coverage" not in added
    assert frame["has_cons_coverage"].tolist() == [7] * 8


def test_training_and_serving_markers_identical():
    """同一份数据下，训练侧（全量）与推理侧（按需）产出的标记值必须逐值一致。"""
    train_frame = _make_frame()
    serve_frame = _make_frame()

    derive_availability_markers(train_frame)
    ensure_availability_markers(serve_frame, ["has_cons_coverage", "has_fund_holding"])

    for name in ("has_cons_coverage", "has_fund_holding"):
        assert train_frame[name].tolist() == serve_frame[name].tolist()
        assert train_frame[name].equals(serve_frame[name])


def test_prepare_training_data_flag_controls_markers():
    """训练入口开关：开启时标记进入 feature_columns，关闭时完全不出现。"""
    frame = _make_prepare_frame()
    rng = np.random.default_rng(11)
    frame["y_ret_5"] = rng.normal(0.01, 0.02, size=len(frame))

    on = prepare_training_data(
        frame.copy(),
        label_column="y_ret_5",
        val_ratio=0.4,
        enable_availability_markers=True,
    )
    off = prepare_training_data(frame.copy(), label_column="y_ret_5", val_ratio=0.4)

    on_feature_columns = on[4]
    off_feature_columns = off[4]
    markers = [name for name in availability_marker_names() if name in on_feature_columns]

    # prepare.py 基础特征列缺失会 KeyError，因此框架必须提供完整基础列（镜像生产 schema 风格）
    assert markers, "开启开关后可用性标记必须进入 feature_columns"
    assert not [name for name in availability_marker_names() if name in off_feature_columns]
    assert list(on[0].columns) == on_feature_columns
    assert set(on[0]["has_cons_coverage"].unique()) <= {0, 1}
    assert on[0]["has_cons_coverage"].dtype == np.float32


class _CaptureModel:
    """记录推理输入特征矩阵的模拟模型（用于断言推理侧标记已物化）。

    模型加载会反序列化为独立实例，因此用类级列表记录，避免丢失捕获。
    """

    captured_frames: List[pd.DataFrame] = []

    def predict(self, X):
        type(self).captured_frames.append(X.copy())
        return X.iloc[:, 0].to_numpy(dtype=float)


def test_ml_signal_materializes_markers_at_inference(tmp_path):
    """推理侧：模型 feature_columns 含标记时必须运行时派生（而不是补成 NaN）。"""
    _CaptureModel.captured_frames.clear()
    registry = ModelRegistry(models_dir=str(tmp_path))
    model = _CaptureModel()
    version = registry.register_model(
        model=model,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["has_cons_coverage", "cons_analyst_count_30d", "alpha_ret_20"],
        label_column="y_ret_5",
        n_samples=100,
        train_params={"n_estimators": 10},
    )
    signal = MLSignal(top_n=2, model_version=version, models_dir=str(tmp_path))
    features_df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "cons_analyst_count_30d": [5.0, np.nan, np.nan],
            "alpha_ret_20": [0.1, 0.2, 0.3],
        }
    )

    signal.generate(
        pd.Timestamp("2023-06-15"),
        list(features_df["ts_code"]),
        {"features": features_df},
    )

    assert _CaptureModel.captured_frames, "推理必须调用模型 predict"
    captured = _CaptureModel.captured_frames[-1]
    assert "has_cons_coverage" in captured.columns
    # 标记必须是运行时派生的 0/1，而不是缺失补 NaN
    assert captured["has_cons_coverage"].tolist() == [1.0, 0.0, 0.0]


def test_oos_eval_frame_derivation_matches_training():
    """OOS 评估侧（独立加载的测试集）派生标记必须与训练侧逐值一致。

    回归背景：`split_training.execute_split_training` 的测试集由 `load_features_data`
    单独加载、不经过 `prepare_training_data`，若不在 `df_test_eval[feature_columns]`
    之前派生标记，会直接 KeyError（2026-09-16 实测 14 折全灭）。
    """
    train_frame = _make_frame()
    test_frame = _make_frame()  # 同一份来源列，模拟独立加载的测试集
    feature_columns = [
        "cons_analyst_count_30d",
        "rzye_chg_5",
        "has_cons_coverage",
        "has_margin_balance",
        "has_fund_holding",
    ]

    derive_availability_markers(train_frame)  # 训练侧：全量派生
    added = ensure_availability_markers(test_frame, feature_columns)  # OOS 侧：按模型列派生

    assert added == ["has_cons_coverage", "has_margin_balance", "has_fund_holding"]
    # 派生后必须能直接按模型特征列切片（修复前此处抛 KeyError）
    selected = test_frame[feature_columns]
    assert list(selected.columns) == feature_columns
    for name in added:
        assert train_frame[name].tolist() == test_frame[name].tolist()
