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


def test_walk_forward_adaptive_candidate_replacement_requires_ir_gain_and_rankic_hold():
    base = {"daily_rankic_ir": 1.20, "daily_rankic_mean": 0.08}

    better = {"daily_rankic_ir": 1.25, "daily_rankic_mean": 0.08}
    assert walk_forward_module._candidate_passes_adaptive_replacement(base, better) is True

    weak_ir = {"daily_rankic_ir": 1.249, "daily_rankic_mean": 0.09}
    assert walk_forward_module._candidate_passes_adaptive_replacement(base, weak_ir) is False

    lower_rankic = {"daily_rankic_ir": 1.30, "daily_rankic_mean": 0.079}
    assert walk_forward_module._candidate_passes_adaptive_replacement(base, lower_rankic) is False


def test_walk_forward_adaptive_candidate_args_follow_requested_rules():
    args = types.SimpleNamespace(learning_rate=0.02, n_estimators=5000)

    low_iter_args = walk_forward_module._build_adaptive_candidate_args(args, "low_iter")
    assert low_iter_args.learning_rate == pytest.approx(0.01, rel=1e-2)
    assert low_iter_args.n_estimators == 10000

    hit_cap_args = walk_forward_module._build_adaptive_candidate_args(args, "hit_cap")
    assert hit_cap_args.learning_rate == pytest.approx(0.03, rel=1e-2)
    assert hit_cap_args.n_estimators == 5000


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
            "daily_rankic_mean": 0.10,
        },
    )

    args = types.SimpleNamespace(
        learning_rate=0.02,
        n_estimators=5000,
        label_column="neu_y_ret_20",
        task="regression",
        random_state=42,
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