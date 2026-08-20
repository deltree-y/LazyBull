# -*- coding: utf-8 -*-
"""推理侧事件 freshness 衰减测试（消除 train/serve skew）。"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.lazybull.ml import ModelRegistry
from src.lazybull.ml.train_core import (
    DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS,
    apply_event_freshness_decay,
    apply_serving_event_decay,
)
from src.lazybull.signals import MLSignal


class RecordingModel:
    """记录 predict 输入的模拟模型。"""

    def __init__(self) -> None:
        self.last_X = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.last_X = X.copy()
        if len(X.columns) > 0:
            return X.iloc[:, 0].fillna(0).values * 0.1
        return np.zeros(len(X))


def _features_frame() -> pd.DataFrame:
    """构造能通过 MLSignal 选股过滤的当日特征。"""
    codes = [f"{i:06d}.SZ" for i in range(1, 9)]
    return pd.DataFrame(
        {
            "ts_code": codes,
            "forecast_type_score": [1.0] * len(codes),
            "forecast_chg_mid": [50.0] * len(codes),
            "forecast_freshness_days": [120] * len(codes),
            "amount_ma20": [60000.0] * len(codes),
            "total_mv": [1000000.0] * len(codes),
            "sw_l1_code": ["801010"] * len(codes),
        }
    )


def _register_recording_model(
    models_dir: str, train_params: dict
) -> tuple:
    """注册记录型模型，返回 (models_dir, version)。"""
    registry = ModelRegistry(models_dir=models_dir)
    model = RecordingModel()
    version = registry.register_model(
        model=model,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["forecast_type_score", "forecast_chg_mid"],
        label_column="neu_y_ret_20",
        n_samples=1000,
        train_params=train_params,
    )
    return models_dir, version


# ═══════════════════════════════════════════════════════════════
# apply_event_freshness_decay 单元
# ═══════════════════════════════════════════════════════════════


def test_decay_weight_follows_half_life():
    df = pd.DataFrame(
        {
            "forecast_freshness_days": [0, 120, 240, np.nan, -10],
            "forecast_type_score": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    df, stats = apply_event_freshness_decay(
        df, event_freshness_cols=["forecast_freshness_days"], half_life_days=120
    )

    assert stats["forecast_freshness_days"] == 5
    assert df["forecast_type_score"].iloc[0] == pytest.approx(1.0)  # 0 天 → 权重 1
    assert df["forecast_type_score"].iloc[1] == pytest.approx(0.5)  # 1 个半衰期
    assert df["forecast_type_score"].iloc[2] == pytest.approx(0.25)  # 2 个半衰期
    assert df["forecast_type_score"].iloc[3] == pytest.approx(1.0)  # 缺失 → 权重 1
    assert df["forecast_type_score"].iloc[4] == pytest.approx(1.0)  # 负值按 0 处理
    # freshness 列本身不被修改
    assert df["forecast_freshness_days"].iloc[1] == 120


def test_decay_only_touches_mapped_value_columns():
    df = pd.DataFrame(
        {
            "forecast_freshness_days": [120],
            "forecast_type_score": [1.0],
            "unrelated_col": [7.0],
        }
    )
    df, _ = apply_event_freshness_decay(
        df, event_freshness_cols=["forecast_freshness_days"], half_life_days=120
    )
    assert df["forecast_type_score"].iloc[0] == pytest.approx(0.5)
    assert df["unrelated_col"].iloc[0] == pytest.approx(7.0)


def test_decay_rejects_non_positive_half_life():
    df = pd.DataFrame({"forecast_freshness_days": [10], "forecast_type_score": [1.0]})
    with pytest.raises(ValueError):
        apply_event_freshness_decay(
            df, event_freshness_cols=["forecast_freshness_days"], half_life_days=0
        )


# ═══════════════════════════════════════════════════════════════
# apply_serving_event_decay 策略门控
# ═══════════════════════════════════════════════════════════════


def _decay_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_freshness_days": [120],
            "forecast_type_score": [1.0],
        }
    )


def test_serving_decay_applies_only_for_decay_strategy():
    df = _decay_input()
    out = apply_serving_event_decay(df, "state_keep_event_decay", 120)
    assert out["forecast_type_score"].iloc[0] == pytest.approx(0.5)

    df = _decay_input()
    out = apply_serving_event_decay(df, "state_keep_event_no_decay", 120)
    assert out["forecast_type_score"].iloc[0] == pytest.approx(1.0)

    df = _decay_input()
    out = apply_serving_event_decay(df, "drop_all", 120)
    assert out["forecast_type_score"].iloc[0] == pytest.approx(1.0)


def test_serving_decay_without_freshness_columns_returns_unchanged():
    df = pd.DataFrame({"forecast_type_score": [1.0]})
    out = apply_serving_event_decay(df, "state_keep_event_decay", 120)
    assert out["forecast_type_score"].iloc[0] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════
# MLSignal 推理侧衰减复现
# ═══════════════════════════════════════════════════════════════


def test_ml_signal_decays_event_features_before_predict():
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir, version = _register_recording_model(
            tmpdir,
            train_params={
                "task": "regression",
                "freshness_strategy": "state_keep_event_decay",
                "event_freshness_half_life_days": 120,
            },
        )
        signal = MLSignal(top_n=5, model_version=version, models_dir=models_dir)
        signal._load_model()
        model = signal.model

        features_df = _features_frame()
        universe = features_df["ts_code"].tolist()
        signals = signal.generate(pd.Timestamp("2023-06-15"), universe, {"features": features_df})

        assert len(signals) == 5
        served = model.last_X
        # freshness=120、半衰期 120 → 权重 0.5
        assert served["forecast_type_score"].iloc[0] == pytest.approx(0.5)
        assert served["forecast_chg_mid"].iloc[0] == pytest.approx(25.0)
        # freshness 列不是模型特征列，不应进入模型
        assert "forecast_freshness_days" not in served.columns


def test_ml_signal_skips_decay_for_no_decay_strategy():
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir, version = _register_recording_model(
            tmpdir,
            train_params={
                "task": "regression",
                "freshness_strategy": "state_keep_event_no_decay",
            },
        )
        signal = MLSignal(top_n=5, model_version=version, models_dir=models_dir)
        signal._load_model()
        model = signal.model

        features_df = _features_frame()
        universe = features_df["ts_code"].tolist()
        signal.generate(pd.Timestamp("2023-06-15"), universe, {"features": features_df})

        served = model.last_X
        assert served["forecast_type_score"].iloc[0] == pytest.approx(1.0)
        assert served["forecast_chg_mid"].iloc[0] == pytest.approx(50.0)


def test_ml_signal_legacy_metadata_defaults_to_decay():
    """旧模型 train_params 无 freshness 键：默认 decay 策略 + 默认半衰期 45。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir, version = _register_recording_model(
            tmpdir,
            train_params={"task": "regression", "n_estimators": 100},
        )
        signal = MLSignal(top_n=5, model_version=version, models_dir=models_dir)
        signal._load_model()
        model = signal.model

        features_df = _features_frame()
        features_df["forecast_freshness_days"] = DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS
        universe = features_df["ts_code"].tolist()
        signal.generate(pd.Timestamp("2023-06-15"), universe, {"features": features_df})

        served = model.last_X
        # freshness=45、半衰期 45 → 权重 0.5
        assert served["forecast_type_score"].iloc[0] == pytest.approx(0.5)
