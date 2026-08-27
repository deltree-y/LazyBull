#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练入口新增因子开关透传回归测试。"""

import types

import pandas as pd
import pytest

from scripts import train_ml_model as train_ml_model_module
from src.lazybull.ml.train_core import prepare_training_data
from src.lazybull.ml.walk_forward import training_core as core_module


def _sample_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "20240102",
                "ts_code": "000001.SZ",
                "neu_y_ret_20": 0.1,
            }
        ]
    )


def test_consensus_feature_flag_requires_complete_built_schema():
    with pytest.raises(ValueError, match="一致预期特征 schema 不完整"):
        prepare_training_data(
            _sample_train_df(),
            label_column="neu_y_ret_20",
            enable_consensus_features=True,
        )


def test_consensus_revision_feature_flag_requires_complete_built_schema():
    with pytest.raises(ValueError, match="一致预期修正特征 schema 不完整"):
        prepare_training_data(
            _sample_train_df(),
            label_column="neu_y_ret_20",
            enable_consensus_revision_features=True,
        )


def test_consensus_revision_feature_flag_rejects_all_nan_sentinel():
    """哨兵列全 NaN（未构建或混入旧语义分区）必须失败，不能静默退化为零因子。"""
    from src.lazybull.ml.train_core.constants import CONSENSUS_REVISION_FEATURE_COLUMNS

    df = _sample_train_df().copy()
    for col in CONSENSUS_REVISION_FEATURE_COLUMNS:
        df[col] = 0.0
    df["cons_revision_schema_v2"] = float("nan")

    with pytest.raises(ValueError, match="哨兵列"):
        prepare_training_data(
            df,
            label_column="neu_y_ret_20",
            enable_consensus_revision_features=True,
        )


def test_walk_forward_train_window_forwards_new_feature_flags(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        core_module,
        "load_features_data",
        lambda *args, **kwargs: (_sample_train_df(), 1),
    )

    def _fake_prepare_training_data(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(
        core_module,
        "prepare_training_data",
        _fake_prepare_training_data,
    )

    args = types.SimpleNamespace(
        task="regression",
        pos_quantile=None,
        pos_topk=None,
        label_column="neu_y_ret_20",
        label_transform="raw",
        winsorize_p=0.01,
        val_ratio=0.1,
        enable_fundamental_features=False,
        enable_alt_features=False,
        enable_margin_features=False,
        enable_cyq_features=False,
        enable_fund_features=False,
        enable_express_features=False,
        enable_enhanced_features=False,
        enable_north_features=False,
        enable_lhb_features=False,
        enable_consensus_features=False,
        enable_cashflow_quality_features=True,
        enable_consensus_revision_features=True,
        feature_stability_filter=False,
        factor_prune=False,
        factor_exclude_file="configs/factor_exclude_candidate_sparse_v1.json",
    )

    with pytest.raises(RuntimeError, match="stop after capture"):
        core_module._train_model_on_window(
            "20240101",
            "20240131",
            storage=None,
            loader=None,
            args=args,
            main_board_codes={"000001.SZ"},
        )

    assert captured["enable_cashflow_quality_features"] is True
    assert captured["enable_consensus_revision_features"] is True
    assert captured["factor_exclude_file"] == ("configs/factor_exclude_candidate_sparse_v1.json")


def test_walk_forward_registered_metadata_includes_consensus_feature_flag():
    args = types.SimpleNamespace(
        enable_consensus_features=True,
        enable_cashflow_quality_features=False,
        enable_consensus_revision_features=True,
    )

    metadata = core_module._build_feature_flag_train_params(args)

    assert metadata == {
        "enable_consensus_features": True,
        "enable_cashflow_quality_features": False,
        "enable_consensus_revision_features": True,
    }


def test_multi_seed_ensemble_keeps_top_30pct_with_min_three(monkeypatch):
    def _fake_train_model_on_window(
        train_start,
        train_end,
        storage,
        loader,
        args,
        main_board_codes=None,
        random_state_override=None,
        feature_columns_override=None,
    ):
        seed = int(random_state_override)
        return {
            "model": f"m{seed}",
            "feature_columns": ["f1"],
            "label_column": "neu_y_ret_20",
            "train_params": {
                "best_iteration": 500,
                "learning_rate": args.learning_rate,
                "n_estimators": args.n_estimators,
                "random_state": seed,
            },
            "train_metrics": {},
            "val_metrics": {},
            "df_val_split_original": pd.DataFrame([{"trade_date": "20240101", "x": 1}]),
            "data_stats": {},
            "train_days_count": 1,
            "total_train_samples": 1,
            "X_train_len": 1,
            "X_val_len": 1,
        }

    monkeypatch.setattr(
        core_module,
        "_train_model_on_window",
        _fake_train_model_on_window,
    )
    monkeypatch.setattr(
        core_module,
        "_evaluate_train_result_val_daily",
        lambda tr, *_args, **_kwargs: {
            "daily_rankic_ir": {
                101: 0.10,
                102: 0.12,
                103: 0.15,
                104: 0.18,
                105: 0.20,
            }[tr["train_params"]["random_state"]],
            "daily_rankic_mean": {
                101: 0.05,
                102: 0.06,
                103: 0.07,
                104: 0.08,
                105: 0.09,
            }[tr["train_params"]["random_state"]],
            "diagnostic_Top30_逐日均值_50分位": {
                101: 0.0010,
                102: 0.0012,
                103: 0.0015,
                104: 0.0018,
                105: 0.0020,
            }[tr["train_params"]["random_state"]],
        },
    )

    args = types.SimpleNamespace(
        learning_rate=0.02,
        n_estimators=5000,
        label_column="neu_y_ret_20",
        task="regression",
        random_state=42,
    )

    sub_models, base_result, _meta = core_module._build_ensemble_sub_models(
        windows=[("20120101", "20181231")],
        storage=None,
        loader=None,
        args=args,
        main_board_codes=set(),
        seeds=[101, 102, 103, 104, 105],
        topk_values=[30],
    )

    assert sub_models == ["m105", "m104", "m103"]
    assert base_result["train_params"]["random_state"] == 105


def test_multi_seed_ensemble_keep_ratio_and_min_models_are_configurable(monkeypatch):
    def _fake_train_model_on_window(
        train_start,
        train_end,
        storage,
        loader,
        args,
        main_board_codes=None,
        random_state_override=None,
        feature_columns_override=None,
    ):
        seed = int(random_state_override)
        return {
            "model": f"m{seed}",
            "feature_columns": ["f1"],
            "label_column": "neu_y_ret_20",
            "train_params": {
                "best_iteration": 500,
                "learning_rate": args.learning_rate,
                "n_estimators": args.n_estimators,
                "random_state": seed,
            },
            "train_metrics": {},
            "val_metrics": {},
            "df_val_split_original": pd.DataFrame([{"trade_date": "20240101", "x": 1}]),
            "data_stats": {},
            "train_days_count": 1,
            "total_train_samples": 1,
            "X_train_len": 1,
            "X_val_len": 1,
        }

    monkeypatch.setattr(
        core_module,
        "_train_model_on_window",
        _fake_train_model_on_window,
    )
    monkeypatch.setattr(
        core_module,
        "_evaluate_train_result_val_daily",
        lambda tr, *_args, **_kwargs: {
            "daily_rankic_ir": {
                201: 0.10,
                202: 0.12,
                203: 0.15,
                204: 0.18,
                205: 0.20,
            }[tr["train_params"]["random_state"]],
            "daily_rankic_mean": {
                201: 0.05,
                202: 0.06,
                203: 0.07,
                204: 0.08,
                205: 0.09,
            }[tr["train_params"]["random_state"]],
            "diagnostic_Top30_逐日均值_50分位": {
                201: 0.0010,
                202: 0.0012,
                203: 0.0015,
                204: 0.0018,
                205: 0.0020,
            }[tr["train_params"]["random_state"]],
        },
    )

    args = types.SimpleNamespace(
        learning_rate=0.02,
        n_estimators=5000,
        label_column="neu_y_ret_20",
        task="regression",
        random_state=42,
        ensemble_seed_keep_top_ratio=0.4,
        ensemble_seed_keep_min_models=2,
    )

    sub_models, base_result, meta = core_module._build_ensemble_sub_models(
        windows=[("20120101", "20181231")],
        storage=None,
        loader=None,
        args=args,
        main_board_codes=set(),
        seeds=[201, 202, 203, 204, 205],
        topk_values=[30],
    )

    assert sub_models == ["m205", "m204"]
    assert base_result["train_params"]["random_state"] == 205
    assert meta["_ensemble_validation_result"]["df_val_split_original"].empty


def test_ensemble_validation_uses_panel_unseen_by_all_retained_models():
    def make_result(seed, train_end, val_es_end, val_start):
        return {
            "train_params": {"random_state": seed},
            "data_stats": {
                "train_end_date": train_end,
                "val_es_end_date": val_es_end,
            },
            "df_val_split_original": pd.DataFrame(
                [{"trade_date": val_start, "ts_code": "000001.SZ", "f1": 1.0}]
            ),
        }

    early_result = make_result(42, "20230630", "20230714", "20230717")
    latest_result = make_result(61, "20230731", "20230811", "20230814")

    selected = core_module._select_ensemble_validation_result([early_result, latest_result])

    assert selected["train_params"]["random_state"] == 61
    assert selected["df_val_split_original"]["trade_date"].min() > "20230811"


def test_train_ml_model_main_forwards_new_feature_flags(monkeypatch):
    captured = {}

    monkeypatch.setattr(train_ml_model_module, "setup_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_ml_model_module, "Storage", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_ml_model_module, "DataLoader", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_ml_model_module, "ModelRegistry", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_ml_model_module, "get_models_root", lambda *args, **kwargs: "models")
    monkeypatch.setattr(
        train_ml_model_module,
        "load_features_data",
        lambda *args, **kwargs: (_sample_train_df(), 1),
    )
    monkeypatch.setattr(train_ml_model_module.traceback, "print_exc", lambda: None)

    def _fake_prepare_training_data(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capture")

    def _fake_exit(code=0):
        raise SystemExit(code)

    monkeypatch.setattr(train_ml_model_module, "prepare_training_data", _fake_prepare_training_data)
    monkeypatch.setattr(
        train_ml_model_module.sys,
        "argv",
        [
            "train_ml_model.py",
            "--start-date",
            "20240101",
            "--end-date",
            "20240131",
            "--data-root",
            "./data",
            "--enable-cashflow-quality-features",
            "--enable-consensus-revision-features",
            "--factor-prune",
            "--factor-exclude-file",
            "configs/factor_exclude_candidate_sparse_v1.json",
        ],
    )
    monkeypatch.setattr(train_ml_model_module.sys, "exit", _fake_exit)

    with pytest.raises(SystemExit) as exc_info:
        train_ml_model_module.main()

    assert exc_info.value.code == 1
    assert captured["enable_cashflow_quality_features"] is True
    assert captured["enable_consensus_revision_features"] is True
    assert captured["factor_exclude_file"] == ("configs/factor_exclude_candidate_sparse_v1.json")


def test_ensemble_sub_models_align_feature_columns_to_base_window(monkeypatch):
    """多窗口集成时，后续子模型应以基础窗口特征列为准（feature_columns_override）。"""
    calls = []

    def _fake_train_model_on_window(
        train_start,
        train_end,
        storage,
        loader,
        args,
        main_board_codes=None,
        random_state_override=None,
        feature_columns_override=None,
    ):
        seed = int(random_state_override)
        calls.append({"seed": seed, "feature_columns_override": feature_columns_override})
        return {
            "model": f"m{seed}",
            "feature_columns": ["f1", "f2"],
            "label_column": "neu_y_ret_20",
            "train_params": {"best_iteration": 100, "random_state": seed},
            "train_metrics": {},
            "val_metrics": {},
            "df_val_split_original": pd.DataFrame(
                [{"trade_date": "20240101", "ts_code": "000001.SZ", "f1": 1.0}]
            ),
            "data_stats": {"train_end_date": "20231231", "val_es_end_date": "20231201"},
            "train_days_count": 1,
            "total_train_samples": 1,
            "X_train_len": 1,
            "X_val_len": 1,
        }

    monkeypatch.setattr(core_module, "_train_model_on_window", _fake_train_model_on_window)
    monkeypatch.setattr(
        core_module,
        "_evaluate_train_result_val_daily",
        lambda tr, *_args, **_kwargs: {
            "daily_rankic_ir": 0.1,
            "daily_rankic_mean": 0.05,
            "diagnostic_Top30_逐日均值_50分位": 0.001,
        },
    )

    args = types.SimpleNamespace(
        learning_rate=0.02,
        n_estimators=5000,
        label_column="neu_y_ret_20",
        task="regression",
        random_state=42,
        ensemble_seed_keep_top_ratio=1.0,
        ensemble_seed_keep_min_models=1,
    )

    sub_models, base_result, _meta = core_module._build_ensemble_sub_models(
        windows=[("20120101", "20181231"), ("20130101", "20191231")],
        storage=None,
        loader=None,
        args=args,
        main_board_codes=set(),
        seeds=[101],
        topk_values=[30],
    )

    # 首个子模型（基础窗口）不传 override；后续子模型以基础窗口特征列为准
    assert calls[0]["feature_columns_override"] is None
    assert calls[1]["feature_columns_override"] == ["f1", "f2"]
    assert sub_models == ["m101", "m101"]
