#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练入口新增因子开关透传回归测试。"""

import types

import pandas as pd
import pytest

from scripts import train_ml_model as train_ml_model_module
from scripts import walk_forward as walk_forward_module


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


def test_walk_forward_train_window_forwards_new_feature_flags(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        walk_forward_module,
        "load_features_data",
        lambda *args, **kwargs: (_sample_train_df(), 1),
    )

    def _fake_prepare_training_data(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(walk_forward_module, "prepare_training_data", _fake_prepare_training_data)

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
    )

    with pytest.raises(RuntimeError, match="stop after capture"):
        walk_forward_module._train_model_on_window(
            "20240101", "20240131", storage=None, loader=None, args=args
        )

    assert captured["enable_cashflow_quality_features"] is True
    assert captured["enable_consensus_revision_features"] is True


def test_walk_forward_adaptive_best_iter_action_thresholds():
    assert walk_forward_module._resolve_adaptive_best_iter_action(100, 5000) == "low_iter"
    assert walk_forward_module._resolve_adaptive_best_iter_action(101, 5000) is None
    assert walk_forward_module._resolve_adaptive_best_iter_action(4749, 5000) is None
    assert walk_forward_module._resolve_adaptive_best_iter_action(4750, 5000) == "hit_cap"
    assert walk_forward_module._resolve_adaptive_best_iter_action(None, 5000) is None


def test_walk_forward_adaptive_candidate_replacement_requires_ir_gain_and_top30_median_hold():
    base = {"daily_rankic_ir": 1.20, "diagnostic_Top30_逐日均值_50分位": 0.0030}

    better = {"daily_rankic_ir": 1.25, "diagnostic_Top30_逐日均值_50分位": 0.0035}
    assert walk_forward_module._candidate_passes_adaptive_replacement(base, better) is True

    weak_ir = {"daily_rankic_ir": 1.199, "diagnostic_Top30_逐日均值_50分位": 0.0050}
    assert walk_forward_module._candidate_passes_adaptive_replacement(base, weak_ir) is False

    lower_top30 = {"daily_rankic_ir": 1.30, "diagnostic_Top30_逐日均值_50分位": 0.0025}
    assert walk_forward_module._candidate_passes_adaptive_replacement(base, lower_top30) is False


def test_walk_forward_adaptive_candidate_args_follow_requested_rules():
    args = types.SimpleNamespace(learning_rate=0.02, n_estimators=5000)

    low_iter_args = walk_forward_module._build_adaptive_candidate_args(args, "low_iter")
    assert low_iter_args.learning_rate == pytest.approx(0.02, rel=1e-2)
    assert low_iter_args.n_estimators == 5000

    hit_cap_args = walk_forward_module._build_adaptive_candidate_args(args, "hit_cap")
    assert hit_cap_args.learning_rate == pytest.approx(0.03, rel=1e-2)
    assert hit_cap_args.n_estimators == 25000


def test_walk_forward_adaptive_retry_seed_is_incremental():
    assert walk_forward_module._resolve_adaptive_retry_seed(42, 1) == 43
    assert walk_forward_module._resolve_adaptive_retry_seed(42, 10) == 52


def test_live_adaptive_updates_remaining_submodels_within_same_split(monkeypatch):
    calls = []

    def _fake_train_model_on_window(
        train_start, train_end, storage, loader, args, random_state_override=None
    ):
        calls.append((round(args.learning_rate, 6), random_state_override))
        best_iter = 4900 if args.learning_rate < 0.03 else 1200
        return {
            "model": object(),
            "feature_columns": ["f1"],
            "train_params": {
                "best_iteration": best_iter,
                "learning_rate": args.learning_rate,
                "n_estimators": args.n_estimators,
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

    monkeypatch.setattr(walk_forward_module, "_train_model_on_window", _fake_train_model_on_window)
    monkeypatch.setattr(
        walk_forward_module,
        "_evaluate_train_result_val_daily",
        lambda tr, *_args, **_kwargs: {
            "daily_rankic_ir": 1.30 if tr["train_params"]["learning_rate"] >= 0.03 else 1.00,
            "daily_rankic_mean": 0.11 if tr["train_params"]["learning_rate"] >= 0.03 else 0.10,
            "diagnostic_Top30_逐日均值_50分位": 0.0040 if tr["train_params"]["learning_rate"] >= 0.03 else 0.0030,
        },
    )

    args = types.SimpleNamespace(
        learning_rate=0.02,
        n_estimators=5000,
        label_column="neu_y_ret_20",
        task="regression",
        random_state=42,
        adaptive_low_iter_max_retries=10,
    )

    _sub_models, _base_result, meta = walk_forward_module._build_ensemble_sub_models(
        windows=[("20120101", "20181231")],
        storage=None,
        loader=None,
        args=args,
        seeds=[729, 121],
        topk_values=[30],
        enable_live_adaptive=True,
    )

    assert calls == [(0.02, 729), (0.03, 729), (0.03, 121)]
    assert meta["live_adaptive_triggered"] is True
    assert meta["live_adaptive_trigger_count"] == 1
    assert meta["live_adaptive_used_count"] == 1
    assert meta["live_adaptive_final_learning_rate"] == pytest.approx(0.03, rel=1e-6)


def test_live_adaptive_low_iter_retries_with_incremental_seed(monkeypatch):
    calls = []

    def _fake_train_model_on_window(
        train_start, train_end, storage, loader, args, random_state_override=None
    ):
        calls.append((round(args.learning_rate, 6), random_state_override))
        if random_state_override == 200:
            best_iter = 17
        elif random_state_override == 201:
            best_iter = 500
        elif random_state_override == 202:
            best_iter = 120
        else:
            best_iter = 450
        return {
            "model": object(),
            "feature_columns": ["f1"],
            "train_params": {
                "best_iteration": best_iter,
                "learning_rate": args.learning_rate,
                "n_estimators": args.n_estimators,
                "random_state": random_state_override,
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

    monkeypatch.setattr(walk_forward_module, "_train_model_on_window", _fake_train_model_on_window)
    monkeypatch.setattr(
        walk_forward_module,
        "_evaluate_train_result_val_daily",
        lambda tr, *_args, **_kwargs: {
            "daily_rankic_ir": {
                201: 1.10,
                202: 1.25,
                203: 1.18,
            }.get(tr["train_params"].get("random_state"), 1.00),
            "daily_rankic_mean": {
                201: 0.11,
                202: 0.13,
                203: 0.12,
            }.get(tr["train_params"].get("random_state"), 0.10),
            "diagnostic_Top30_逐日均值_50分位": {
                201: 0.0035,
                202: 0.0045,
                203: 0.0040,
            }.get(tr["train_params"].get("random_state"), 0.0030),
        },
    )

    args = types.SimpleNamespace(
        learning_rate=0.02,
        n_estimators=5000,
        label_column="neu_y_ret_20",
        task="regression",
        random_state=42,
        adaptive_low_iter_max_retries=3,
    )

    _sub_models, _base_result, meta = walk_forward_module._build_ensemble_sub_models(
        windows=[("20120101", "20181231")],
        storage=None,
        loader=None,
        args=args,
        seeds=[200],
        topk_values=[30],
        enable_live_adaptive=True,
    )

    assert calls == [(0.02, 200), (0.02, 201), (0.02, 202), (0.02, 203)]
    assert meta["live_adaptive_triggered"] is True
    assert meta["live_adaptive_last_action"] == "low_iter"
    assert meta["live_adaptive_retry_count"] == 3
    assert meta["live_adaptive_last_retry_seed"] == 203
    assert meta["live_adaptive_last_candidate_best_iteration"] == 120
    assert meta["live_adaptive_used_count"] == 1
    assert meta["live_adaptive_final_random_state"] == 202


def test_multi_seed_ensemble_keeps_top_30pct_with_min_three(monkeypatch):
    def _fake_train_model_on_window(
        train_start, train_end, storage, loader, args, random_state_override=None
    ):
        seed = int(random_state_override)
        return {
            "model": f"m{seed}",
            "feature_columns": ["f1"],
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

    monkeypatch.setattr(walk_forward_module, "_train_model_on_window", _fake_train_model_on_window)
    monkeypatch.setattr(
        walk_forward_module,
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
        adaptive_low_iter_max_retries=3,
    )

    sub_models, base_result, _meta = walk_forward_module._build_ensemble_sub_models(
        windows=[("20120101", "20181231")],
        storage=None,
        loader=None,
        args=args,
        seeds=[101, 102, 103, 104, 105],
        topk_values=[30],
        enable_live_adaptive=False,
    )

    assert sub_models == ["m105", "m104", "m103"]
    assert base_result["train_params"]["random_state"] == 105


def test_multi_seed_ensemble_keep_ratio_and_min_models_are_configurable(monkeypatch):
    def _fake_train_model_on_window(
        train_start, train_end, storage, loader, args, random_state_override=None
    ):
        seed = int(random_state_override)
        return {
            "model": f"m{seed}",
            "feature_columns": ["f1"],
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

    monkeypatch.setattr(walk_forward_module, "_train_model_on_window", _fake_train_model_on_window)
    monkeypatch.setattr(
        walk_forward_module,
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
        adaptive_low_iter_max_retries=3,
        ensemble_seed_keep_top_ratio=0.4,
        ensemble_seed_keep_min_models=2,
    )

    sub_models, base_result, _meta = walk_forward_module._build_ensemble_sub_models(
        windows=[("20120101", "20181231")],
        storage=None,
        loader=None,
        args=args,
        seeds=[201, 202, 203, 204, 205],
        topk_values=[30],
        enable_live_adaptive=False,
    )

    assert sub_models == ["m205", "m204"]
    assert base_result["train_params"]["random_state"] == 205


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
        ],
    )
    monkeypatch.setattr(train_ml_model_module.sys, "exit", _fake_exit)

    with pytest.raises(SystemExit) as exc_info:
        train_ml_model_module.main()

    assert exc_info.value.code == 1
    assert captured["enable_cashflow_quality_features"] is True
    assert captured["enable_consensus_revision_features"] is True