#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_core 验证集尾部隔离专项测试。"""

import json

import numpy as np
import pandas as pd
import pytest

import src.lazybull.ml.train_core as train_core_module
from src.lazybull.ml.train_core import (
    evaluate_validation_daily,
    prepare_training_data,
    split_train_val_by_date,
    split_val_for_early_stopping_by_date,
    split_val_for_selection_protocol_by_date,
)


BASE_FEATURE_COLUMNS = [
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
    "list_days",
    "mkt_adv_dec_ratio",
    "mkt_ret_avg_20",
    "mkt_turnover_std",
    "mkt_vol_20",
]


def test_load_factor_exclude_list_caches_by_explicit_file(tmp_path):
    first_file = tmp_path / "first.json"
    second_file = tmp_path / "second.json"
    first_file.write_text(json.dumps({"exclude_factors": ["factor_a"]}), encoding="utf-8")
    second_file.write_text(json.dumps({"exclude_factors": ["factor_b"]}), encoding="utf-8")

    train_core_module._factor_exclude_cache.clear()
    try:
        assert train_core_module._load_factor_exclude_list(exclude_file=first_file) == {
            "factor_a"
        }
        assert train_core_module._load_factor_exclude_list(exclude_file=second_file) == {
            "factor_b"
        }
    finally:
        train_core_module._factor_exclude_cache.clear()


def _make_training_df(n_dates: int, stocks_per_date: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for date_idx in range(n_dates):
        trade_date = f"2024{(101 + date_idx):04d}"
        for stock_idx in range(stocks_per_date):
            row = {
                "trade_date": trade_date,
                "ts_code": f"{stock_idx:06d}.SZ",
                "name": f"S{stock_idx:06d}",
                "industry": "测试行业",
                "list_date": "20100101",
                "list_days": 1000 + date_idx,
                "is_st": 0,
                "is_suspended": 0,
                "is_limit_up": 0,
                "is_limit_down": 0,
                "y_ret_5": float(rng.normal(loc=0.01, scale=0.02)),
            }
            for feat_idx, feat_col in enumerate(BASE_FEATURE_COLUMNS):
                row[feat_col] = float(date_idx * 0.01 + stock_idx * 0.001 + feat_idx * 1e-6)
            rows.append(row)
    return pd.DataFrame(rows)


def test_split_val_for_early_stopping_by_date_tail_embargo():
    df = _make_training_df(n_dates=12, stocks_per_date=3)
    df_val = df[["trade_date", "ts_code", "y_ret_5"]].copy()

    df_es, df_embargo, stats = split_val_for_early_stopping_by_date(df_val, embargo_days=4)

    assert stats["val_raw_n_dates"] == 12
    assert stats["val_es_n_dates"] == 8
    assert stats["val_embargo_n_dates"] == 4
    assert len(df_es) == 8 * 3
    assert len(df_embargo) == 4 * 3
    assert set(df_es["trade_date"].unique()).isdisjoint(set(df_embargo["trade_date"].unique()))
    assert df_es["trade_date"].max() < df_embargo["trade_date"].min()


def test_split_val_for_early_stopping_by_date_all_embargo_when_short():
    df = _make_training_df(n_dates=3, stocks_per_date=2)
    df_val = df[["trade_date", "ts_code", "y_ret_5"]].copy()

    df_es, df_embargo, stats = split_val_for_early_stopping_by_date(df_val, embargo_days=5)

    assert stats["val_raw_n_dates"] == 3
    assert stats["val_es_n_dates"] == 0
    assert stats["val_embargo_n_dates"] == 3
    assert len(df_es) == 0
    assert len(df_embargo) == len(df_val)


def test_split_val_for_selection_protocol_by_date_splits_three_segments():
    df = _make_training_df(n_dates=20, stocks_per_date=2)
    df_val = df[["trade_date", "ts_code", "y_ret_5"]].copy()

    df_es, df_calib, df_embargo, stats = split_val_for_selection_protocol_by_date(
        df_val, embargo_days=4
    )

    assert stats["val_raw_n_dates"] == 20
    assert stats["val_embargo_n_dates"] == 4
    assert stats["val_calib_n_dates"] == 4
    assert stats["val_es_n_dates"] == 12
    assert len(df_es) == 12 * 2
    assert len(df_calib) == 4 * 2
    assert len(df_embargo) == 4 * 2
    assert set(df_es["trade_date"].unique()).isdisjoint(set(df_calib["trade_date"].unique()))
    assert set(df_calib["trade_date"].unique()).isdisjoint(set(df_embargo["trade_date"].unique()))
    assert df_es["trade_date"].max() < df_calib["trade_date"].min()
    assert df_calib["trade_date"].max() < df_embargo["trade_date"].min()


def test_split_val_for_selection_protocol_short_data_falls_back_to_es_only():
    df = _make_training_df(n_dates=7, stocks_per_date=2)
    df_val = df[["trade_date", "ts_code", "y_ret_5"]].copy()

    df_es, df_calib, df_embargo, stats = split_val_for_selection_protocol_by_date(
        df_val, embargo_days=6
    )

    assert stats["val_embargo_n_dates"] == 6
    assert stats["val_calib_n_dates"] == 0
    assert stats["val_es_n_dates"] == 1
    assert len(df_calib) == 0
    assert len(df_es) == 2


def test_prepare_training_data_auto_embargo_from_label_horizon():
    df = _make_training_df(n_dates=40, stocks_per_date=2)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        feature_columns,
        df_train_split,
        df_val_split,
        data_stats,
        df_val_split_original,
    ) = prepare_training_data(
        df,
        label_column="y_ret_5",
        val_ratio=0.4,
    )

    expected_delta = 6  # y_ret_5 -> horizon=5, delta=max(5+1, 5)=6
    assert data_stats["val_embargo_days"] == expected_delta
    assert data_stats["val_raw_n_dates"] > 0
    assert data_stats["val_embargo_n_dates"] == expected_delta
    assert data_stats["val_es_n_dates"] + data_stats["val_calib_n_dates"] == (
        data_stats["val_raw_n_dates"] - expected_delta
    )

    # 返回给训练器的验证集应与 early stopping 子集一致
    assert len(X_val) == len(y_val) == len(df_val_split)
    assert len(X_val) == data_stats["val_es_samples"]
    assert len(X_train) == len(y_train) == len(df_train_split)
    assert list(X_val.columns) == feature_columns

    if data_stats["val_calib_n_dates"] > 0:
        assert len(df_val_split_original) == data_stats["val_calib_samples"]
        assert set(df_val_split["trade_date"].unique()).isdisjoint(
            set(df_val_split_original["trade_date"].unique())
        )
    else:
        assert len(df_val_split_original) == len(df_val_split)

    # 验证集日期不应与训练集日期重叠
    train_dates = set(df_train_split["trade_date"].unique())
    val_dates = set(df_val_split["trade_date"].unique())
    assert train_dates.isdisjoint(val_dates)

    if data_stats["val_es_n_dates"] > 0 and data_stats["val_calib_n_dates"] > 0:
        assert data_stats["val_end_date"] < data_stats["val_calib_start_date"]
    if data_stats["val_calib_n_dates"] > 0 and data_stats["val_embargo_n_dates"] > 0:
        assert data_stats["val_calib_end_date"] < data_stats["val_embargo_start_date"]

    # 与独立计算的期望子集对齐
    _, raw_val, _ = split_train_val_by_date(df, val_ratio=0.4, delta=expected_delta)
    expected_es, expected_calib, _, _ = split_val_for_selection_protocol_by_date(
        raw_val, embargo_days=expected_delta
    )
    assert set(df_val_split["trade_date"].unique()) == set(expected_es["trade_date"].unique())
    assert set(df_val_split_original["trade_date"].unique()) == set(
        (expected_calib if len(expected_calib) > 0 else expected_es)["trade_date"].unique()
    )


class _RiskMockModel:
    def predict(self, X):
        return X["pred_feature"].values


def test_evaluate_validation_daily_uses_existing_prediction_col():
    df_val = pd.DataFrame(
        [
            {
                "trade_date": "20240102",
                "ts_code": "000001.SZ",
                "pred_feature": 0.1,
                "y_ret_5": -0.05,
                "final_score": 0.9,
            },
            {
                "trade_date": "20240102",
                "ts_code": "000002.SZ",
                "pred_feature": 0.9,
                "y_ret_5": 0.08,
                "final_score": 0.1,
            },
            {
                "trade_date": "20240103",
                "ts_code": "000003.SZ",
                "pred_feature": 0.2,
                "y_ret_5": -0.02,
                "final_score": 0.8,
            },
            {
                "trade_date": "20240103",
                "ts_code": "000004.SZ",
                "pred_feature": 0.8,
                "y_ret_5": 0.06,
                "final_score": 0.2,
            },
        ]
    )

    metrics = evaluate_validation_daily(
        model=_RiskMockModel(),
        df_val=df_val,
        feature_columns=["pred_feature"],
        original_return_col="y_ret_5",
        task="regression",
        topk_values=[1],
        emit_logs=False,
        prediction_col="final_score",
    )

    assert metrics["prediction_col"] == "final_score"
    assert metrics["top1_return_mean"] == (-0.05 - 0.02) / 2


def test_prepare_training_data_state_keep_event_decay_and_high_missing(monkeypatch):
    df = _make_training_df(n_dates=40, stocks_per_date=2)
    df["fundamental_freshness_days"] = 5.0
    df["forecast_type_score"] = 1.0
    df["forecast_chg_mid"] = 2.0
    df["forecast_freshness_days"] = np.arange(len(df)) % 20
    df["zscore_bp_sz"] = np.linspace(1.0, 2.0, len(df))
    df["vol_ratio_20"] = np.nan
    df.loc[df.index[::5], "vol_ratio_20"] = 1.0  # 缺失率 80%

    # 拆分后 prepare_training_data 在 prepare 模块，_load_factor_exclude_list 由该模块绑定引用
    monkeypatch.setattr(
        "src.lazybull.ml.train_core.prepare._load_factor_exclude_list",
        lambda models_dir=None, exclude_file=None: {"zscore_bp"},
    )

    result = prepare_training_data(
        df,
        label_column="y_ret_5",
        val_ratio=0.3,
        enable_fundamental_features=True,
        enable_alt_features=True,
        factor_prune=True,
        max_feature_missing_ratio=0.4,
        freshness_strategy="state_keep_event_decay",
        event_freshness_half_life_days=10,
    )

    feature_columns = result[4]
    df_train_split = result[5]
    data_stats = result[7]

    assert (
        "fundamental_freshness_days" in feature_columns
        or "fundamental_freshness_days" in data_stats["removed_constant_features"]
    )
    assert "forecast_freshness_days" not in feature_columns
    assert "vol_ratio_20" not in feature_columns
    assert "zscore_bp" not in feature_columns
    assert "zscore_bp_sz" not in feature_columns
    assert "forecast_freshness_days" in data_stats["removed_freshness_features"]
    assert "fundamental_freshness_days" in data_stats["kept_state_freshness_features"]
    assert "vol_ratio_20" in data_stats["removed_high_missing_features"]

    # 事件型特征应被 freshness 衰减（至少有一部分样本 < 原始值）
    assert df_train_split["forecast_type_score"].max() <= 1.0
    assert df_train_split["forecast_type_score"].min() < 1.0


def test_prepare_training_data_state_keep_event_no_decay():
    df = _make_training_df(n_dates=40, stocks_per_date=2)
    df["fundamental_freshness_days"] = np.arange(len(df)) % 10
    df["forecast_type_score"] = 1.0
    df["forecast_chg_mid"] = 2.0
    df["forecast_freshness_days"] = np.arange(len(df)) % 20

    result = prepare_training_data(
        df,
        label_column="y_ret_5",
        val_ratio=0.3,
        enable_fundamental_features=True,
        enable_alt_features=True,
        freshness_strategy="state_keep_event_no_decay",
    )

    feature_columns = result[4]
    df_train_split = result[5]
    data_stats = result[7]

    assert "fundamental_freshness_days" in feature_columns
    assert "forecast_freshness_days" not in feature_columns
    assert "forecast_freshness_days" in data_stats["removed_freshness_features"]
    assert "fundamental_freshness_days" in data_stats["kept_state_freshness_features"]
    assert df_train_split["forecast_type_score"].eq(1.0).all()
    assert df_train_split["forecast_chg_mid"].eq(2.0).all()


def test_prepare_training_data_feature_columns_override():
    """feature_columns_override 强制特征列对齐，数据缺失列补 NaN。"""
    df = _make_training_df(n_dates=40, stocks_per_date=2)
    df["vol_ratio_20"] = np.nan
    df.loc[df.index[::5], "vol_ratio_20"] = 1.0  # 缺失率 80% > 0.6

    override_cols = ["neu_ret_1", "vol_ratio_20", "express_revenue_yoy"]

    result = prepare_training_data(
        df,
        label_column="y_ret_5",
        val_ratio=0.3,
        max_feature_missing_ratio=0.6,
        feature_columns_override=override_cols,
    )

    feature_columns = result[4]
    X_train = result[0]
    df_train_split = result[5]

    # 强制对齐到 override，且保持顺序一致
    assert feature_columns == override_cols
    # 高缺失列被强制保留
    assert "vol_ratio_20" in feature_columns
    # 数据中不存在的列已补全为全 NaN 列，且参与训练数据
    assert "express_revenue_yoy" in X_train.columns
    assert X_train["express_revenue_yoy"].isna().all()
    assert df_train_split["express_revenue_yoy"].isna().all()


def test_prepare_training_data_without_override_removes_high_missing():
    """未提供 override 时，高缺失列仍按门禁移除（默认行为不变）。"""
    df = _make_training_df(n_dates=40, stocks_per_date=2)
    df["vol_ratio_20"] = np.nan
    df.loc[df.index[::5], "vol_ratio_20"] = 1.0  # 缺失率 80% > 0.6

    result = prepare_training_data(
        df,
        label_column="y_ret_5",
        val_ratio=0.3,
        max_feature_missing_ratio=0.6,
    )

    feature_columns = result[4]
    data_stats = result[7]

    assert "vol_ratio_20" not in feature_columns
    assert "vol_ratio_20" in data_stats["removed_high_missing_features"]
