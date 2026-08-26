"""一致预期开关在 walk-forward 注册边界的回归测试。"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.lazybull.ml.walk_forward import deploy_training, split_training
from src.lazybull.ml.walk_forward.utils import WalkForwardSplit


class _PredictModel:
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        algorithm="xgboost",
        task="regression",
        label_column="neu_y_ret_20",
        label_transform="raw",
        winsorize_p=0.01,
        pos_quantile=None,
        pos_topk=None,
        scale_pos_weight=None,
        val_ratio=0.2,
        train_window_years=5,
        ensemble_offsets=0,
        ensemble_seeds=[42],
        enable_consensus_features=True,
        enable_cashflow_quality_features=False,
        enable_consensus_revision_features=True,
        neutral_label_blend_weight=0.0,
        oos_detail_metrics=False,
        run_log_csv=None,
        data_root="data",
    )


def _data_stats() -> dict:
    return {
        "samples_after_filter": 3,
        "val_start_date": "20231201",
        "val_end_date": "20231229",
    }


def _training_result() -> dict:
    return {
        "model": _PredictModel(),
        "feature_columns": ["f1"],
        "label_column": "neu_y_ret_20",
        "train_params": {"best_iteration": 100},
        "train_metrics": {},
        "val_metrics": {},
        "df_val_split_original": pd.DataFrame(),
        "data_stats": _data_stats(),
        "train_days_count": 10,
        "total_train_samples": 3,
        "X_train_len": 2,
        "X_val_len": 1,
    }


def _assert_registered_flags(registry: MagicMock) -> None:
    train_params = registry.register_model.call_args.kwargs["train_params"]
    assert train_params["enable_consensus_features"] is True
    assert train_params["enable_cashflow_quality_features"] is False
    assert train_params["enable_consensus_revision_features"] is True


def test_deploy_registration_persists_consensus_feature_flags(monkeypatch):
    registry = MagicMock()
    registry.register_model.return_value = "0.96.0-test"
    monkeypatch.setattr(
        deploy_training,
        "resolve_deploy_train_window",
        lambda **_kwargs: ("20190101", "20231229"),
    )
    monkeypatch.setattr(deploy_training, "_resolve_ensemble_seeds", lambda _args: [42])
    monkeypatch.setattr(
        deploy_training,
        "_train_model_on_window",
        lambda *_args, **_kwargs: _training_result(),
    )
    monkeypatch.setattr(
        deploy_training,
        "create_training_run_record_from_training_session",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(deploy_training, "write_training_run_to_csv", lambda *_args: None)

    deploy_training.execute_deploy_training(
        deploy_train_end="20231229",
        wf_run_id="test-run",
        storage=MagicMock(),
        loader=MagicMock(),
        registry=registry,
        args=_args(),
        main_board_codes={"000001.SZ"},
        topk_values=[30],
        trade_cal=pd.DataFrame(),
    )

    _assert_registered_flags(registry)


def test_split_registration_persists_consensus_feature_flags(monkeypatch):
    registry = MagicMock()
    registry.register_model.return_value = "0.96.0-test"
    candidate = _training_result()
    candidate.update(
        {
            "candidate_name": "base",
            "val_daily_metrics": {},
            "ensemble_meta": {},
        }
    )
    test_df = pd.DataFrame(
        {
            "trade_date": ["20240102"],
            "ts_code": ["000001.SZ"],
            "f1": [1.0],
            "neu_y_ret_20": [0.01],
            "is_st": [False],
            "is_suspended": [False],
            "is_limit_up": [False],
        }
    )
    monkeypatch.setattr(
        split_training,
        "_build_split_training_candidate",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(split_training, "load_features_data", lambda *_args: (test_df, 1))
    monkeypatch.setattr(
        split_training,
        "_filter_to_main_board",
        lambda frame, *_args: frame,
    )
    monkeypatch.setattr(split_training, "apply_serving_event_decay", lambda frame, **_kwargs: frame)
    monkeypatch.setattr(split_training, "evaluate_validation_daily", lambda **_kwargs: {})
    monkeypatch.setattr(
        split_training,
        "build_daily_topk_detail_df",
        lambda **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        split_training,
        "create_training_run_record_from_training_session",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(split_training, "write_training_run_to_csv", lambda *_args: None)

    split_training.execute_split_training(
        split=WalkForwardSplit(0, "20190101", "20231229", "20240102", "20240102"),
        wf_run_id="test-run",
        storage=MagicMock(),
        loader=MagicMock(),
        registry=registry,
        args=_args(),
        main_board_codes={"000001.SZ"},
        topk_values=[30],
    )

    _assert_registered_flags(registry)
