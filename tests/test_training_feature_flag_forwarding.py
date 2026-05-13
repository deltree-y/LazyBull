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
    )

    with pytest.raises(RuntimeError, match="stop after capture"):
        walk_forward_module._train_model_on_window(
            "20240101", "20240131", storage=None, loader=None, args=args
        )

    assert captured["enable_cashflow_quality_features"] is True
    assert captured["enable_consensus_revision_features"] is True


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