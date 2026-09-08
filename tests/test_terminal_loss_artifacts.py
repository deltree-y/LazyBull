"""terminal_loss 产物落盘 artifacts 测试（flat/版本化双模式，合成数据）"""

import json

import numpy as np
import pandas as pd
import pytest

from src.lazybull.risk.terminal_loss import (
    TERMINAL_LOSS_MODEL_TYPE,
    TerminalLossModel,
    TerminalLossModelConfig,
    build_performance_metrics,
    save_flat_artifacts,
    save_versioned_artifacts,
)
from src.lazybull.risk.terminal_loss.train import TerminalLossTrainConfig

FEATURES = ["f1", "f2", "remaining_intervals"]


def _tiny_model(tmp_path) -> TerminalLossModel:
    """构造微型 TerminalLossModel（3 特征 XGBClassifier，cpu）。"""
    from xgboost import XGBClassifier

    rng = np.random.default_rng(11)
    X = rng.normal(0, 1, (200, 3))
    y = (rng.random(200) < 0.3).astype(int)
    clf = XGBClassifier(n_estimators=10, max_depth=2, eval_metric="logloss", device="cpu")
    clf.fit(X, y)
    config = TerminalLossModelConfig(
        task_id="terminal_vol_scaled_loss",
        feature_names=FEATURES,
        train_config=TerminalLossTrainConfig(device="cpu"),
        metadata={"n_train": 200, "stage_dates": {"train": ["20240102", "20241231"]}},
    )
    return TerminalLossModel(config, clf)


def _report_payload() -> dict:
    return {
        "es": {"n": 50, "event_rate": 0.3, "logloss": 0.6},
        "train": {"n": 200, "event_rate": 0.3},
        "label_coverage": [{"h": 5, "label_status": "valid", "count": 10}],
        "isolation_dropped": 0,
    }


def _coverage_df() -> pd.DataFrame:
    return pd.DataFrame({"h": [5, 10], "label_status": ["valid", "valid"], "count": [10, 8]})


def _calibration_df() -> pd.DataFrame:
    return pd.DataFrame({"h": [5], "sigma_bin": ["q1"], "n": [10]})


class TestModelTypeGuard:
    def test_model_type_must_not_contain_classifier(self):
        """ModelRegistry.get_latest_version 跳过 model_type 含 classifier 的模型，
        terminal_loss 目录内 registry 若含该子串将导致 load_model(None) 取不到 latest。"""
        assert "classifier" not in TERMINAL_LOSS_MODEL_TYPE


class TestPerformanceMetrics:
    def test_build_performance_metrics_aligns_with_summarize(self):
        es = {"logloss": 0.6, "brier": 0.2, "pr_auc": 0.45, "event_rate": 0.3, "mean_pred": 0.28}
        train = {"logloss": 0.4, "event_rate": 0.3}
        m = build_performance_metrics(es, train)
        assert m["lift"] == pytest.approx(0.45 / 0.3)
        assert m["pred_bias"] == pytest.approx(0.28 - 0.3)
        assert m["es_logloss"] == 0.6
        assert m["train_logloss"] == 0.4

    def test_zero_event_rate_lift_is_none(self):
        es = {"logloss": 0.1, "brier": 0.05, "pr_auc": 0.0, "event_rate": 0.0, "mean_pred": 0.05}
        m = build_performance_metrics(es, {})
        assert m["lift"] is None


class TestSaveFlatArtifacts:
    def test_fixed_name_five_files(self, tmp_path):
        model = _tiny_model(tmp_path)
        save_flat_artifacts(tmp_path, model, _report_payload(), _calibration_df(), _coverage_df())
        for name in (
            "terminal_loss_model.joblib",
            "terminal_loss_model.json",
            "terminal_loss_report.json",
            "calibration_by_h_sigma.csv",
            "label_coverage.csv",
        ):
            assert (tmp_path / name).exists(), name
        assert not list(tmp_path.glob("v*_model.joblib"))


class TestSaveVersionedArtifacts:
    def test_two_saves_register_two_versions(self, tmp_path):
        model = _tiny_model(tmp_path)
        v1 = save_versioned_artifacts(
            tmp_path,
            model,
            train_start_date="20240102",
            train_end_date="20241231",
            n_samples=200,
            train_params={"n_train": 200, "stage_dates": {"train": ["a", "b"]}},
            performance_metrics={"es_logloss": 0.6},
            report_payload=_report_payload(),
            calibration_df=_calibration_df(),
            coverage_df=_coverage_df(),
        )
        v2 = save_versioned_artifacts(
            tmp_path,
            model,
            train_start_date="20240102",
            train_end_date="20241231",
            n_samples=200,
            train_params={"n_train": 200},
            performance_metrics={"es_logloss": 0.55},
            report_payload=_report_payload(),
            calibration_df=_calibration_df(),
            coverage_df=_coverage_df(),
        )
        assert (v1, v2) == (1, 2)

        # 两套 v{N} 产物共存，且无 flat 固定名产物
        for v in ("v1", "v2"):
            for suffix in (
                "_model.joblib",
                "_features.json",
                "_metadata.json",
                "_report.json",
                "_calibration_by_h_sigma.csv",
                "_label_coverage.csv",
            ):
                assert (tmp_path / f"{v}{suffix}").exists(), f"{v}{suffix}"
        assert not (tmp_path / "terminal_loss_model.joblib").exists()

        assert (tmp_path / "latest_model_version.txt").read_text(encoding="utf-8").strip() == "2"
        with open(tmp_path / "model_registry.json", encoding="utf-8") as f:
            registry = json.load(f)
        assert registry["next_version"] == 3
        assert [m["version"] for m in registry["models"]] == [1, 2]
        assert all(m["model_type"] == TERMINAL_LOSS_MODEL_TYPE for m in registry["models"])
        assert all(m["label_column"] == "loss_label" for m in registry["models"])

    def test_load_model_returns_usable_instance(self, tmp_path):
        from src.lazybull.ml.model_registry import ModelRegistry

        model = _tiny_model(tmp_path)
        save_versioned_artifacts(
            tmp_path,
            model,
            train_start_date="20240102",
            train_end_date="20241231",
            n_samples=200,
            train_params={"n_train": 200},
            performance_metrics={"es_logloss": 0.6},
            report_payload=_report_payload(),
            calibration_df=_calibration_df(),
            coverage_df=_coverage_df(),
        )
        loaded, meta = ModelRegistry(models_dir=str(tmp_path)).load_model(version=1)
        assert isinstance(loaded, TerminalLossModel)
        assert meta["model_type"] == TERMINAL_LOSS_MODEL_TYPE
        infer_df = pd.DataFrame({c: np.zeros(2) for c in FEATURES})
        proba = loaded.predict_proba(infer_df)
        assert proba.shape == (2,)
        assert ((proba >= 0) & (proba <= 1)).all()
