"""DEF-ML-60-v1 R2 测试：交叉修正、分数与排名、等日期权重、purge 与诊断。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.defensive_ml import prediction, spec, training


def test_crossing_is_fixed_by_sorting_rows() -> None:
    raw = np.array([[0.10, 0.05, 0.20], [0.30, 0.10, 0.05], [0.01, 0.02, 0.03]])
    fixed, crossed = prediction.fix_crossing(raw)
    assert list(crossed) == [True, True, False]
    assert fixed[0].tolist() == [0.05, 0.10, 0.20]
    assert fixed[1].tolist() == [0.05, 0.10, 0.30]
    assert fixed[2].tolist() == [0.01, 0.02, 0.03]


def test_score_formula_and_rank_tiebreak() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20260105", "20260105", "20260105", "20260106", "20260106"],
            "ts_code": ["B.SZ", "A.SZ", "C.SZ", "A.SZ", "B.SZ"],
        }
    )
    raw = np.array(
        [
            [-0.05, -0.02, 0.01],  # B: score -0.035
            [-0.05, -0.02, 0.01],  # A: 同分，代码升序在前
            [-0.10, -0.08, -0.02],  # C: 更低
            [-0.30, -0.20, -0.10],  # A: 全负仍参与排序
            [-0.30, -0.20, -0.10],  # B
        ]
    )
    out = prediction.score_frame(frame, raw, "f01", {"0.1": 1, "0.5": 2, "0.9": 3}, "sha")
    first = out[out["trade_date"] == "20260105"]
    assert first["ts_code"].tolist() == ["A.SZ", "B.SZ", "C.SZ"]
    assert first["rank"].tolist() == [1, 2, 3]
    assert first["score"].iloc[0] == pytest.approx(0.5 * -0.05 + 0.5 * -0.02)
    second = out[out["trade_date"] == "20260106"]
    assert second["score"].lt(0).all()
    assert second["rank"].tolist() == [1, 2]


def test_sample_weights_equalize_dates() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20260105"] * 2 + ["20260106"] * 6,
            "y_def_ret60": [0.1] * 8,
        }
    )
    weights = training.sample_weights(frame)
    per_date = pd.Series(weights).groupby(frame["trade_date"].to_numpy()).sum()
    assert per_date.iloc[0] == pytest.approx(per_date.iloc[1])
    assert weights.mean() == pytest.approx(1.0)


def test_training_slice_purges_by_label_end() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20230103", "20230109", "20230401"],
            "label_end_date": ["20230303", "20230501", "20231001"],
            "label_status": ["valid", "valid", "valid"],
        }
    )
    fold = {"train_days": ["20230103", "20230109", "20230401"], "V": "20230401"}
    sliced = training.training_slice(frame, fold)
    assert sliced["trade_date"].tolist() == ["20230103"]


def test_validation_slice_requires_valid_and_maturity() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20230401", "20230402", "20230403"],
            "label_end_date": ["20230601", "20231001", "20230901"],
            "label_status": ["valid", "valid", "right_censored"],
        }
    )
    fold = {"val_days": ["20230401", "20230402", "20230403"], "C": "20230901"}
    sliced = training.validation_slice(frame, fold)
    assert sliced["trade_date"].tolist() == ["20230401"]


def test_model_params_are_frozen() -> None:
    params = training._model_params(0.1, 8)
    assert params["objective"] == "reg:quantileerror"
    assert params["quantile_alpha"] == 0.1
    assert params["n_estimators"] == 300
    assert spec.MODEL["early_stopping"] is False
    assert params["max_depth"] == 3
    assert params["min_child_weight"] == 200
    assert params["n_jobs"] == 8


def test_predict_matrix_rejects_non_finite() -> None:
    class _Model:
        def predict(self, matrix):  # noqa: D401 - 测试替身
            return np.array([0.1, np.nan])

    frame = pd.DataFrame({name: [0.0, 0.0] for name in spec.FEATURES})
    with pytest.raises(ValueError):
        prediction.predict_matrix(_Model(), frame)


def test_pinball_loss_hand_values() -> None:
    y = np.array([0.0, 0.0])
    pred = np.array([0.1, -0.1])
    # q=0.5: 两侧各 0.05 -> 均值 0.05
    assert prediction.pinball_loss(y, pred, 0.5) == pytest.approx(0.05)
    # q=0.9: 低估惩罚重（0.09），高估轻（0.01）-> 均值 0.05
    assert prediction.pinball_loss(y, pred, 0.9) == pytest.approx(0.05)
    # 单侧核对：低估 0.1 时 q=0.9 罚 0.09，q=0.1 罚 0.01
    assert prediction.pinball_loss(np.array([0.1]), np.array([0.0]), 0.9) == pytest.approx(0.09)
    assert prediction.pinball_loss(np.array([0.1]), np.array([0.0]), 0.1) == pytest.approx(0.01)


def test_quantile_diagnostics_reports_coverage() -> None:
    actual = pd.DataFrame(
        {"trade_date": ["20260105"], "ts_code": ["A"], "y_def_ret60": [0.02]}
    )
    predictions = pd.DataFrame(
        {
            "trade_date": ["20260105"],
            "ts_code": ["A"],
            "q10_fixed": [-0.05],
            "q50_fixed": [0.01],
            "q90_fixed": [0.08],
            "crossed": [False],
        }
    )
    result = prediction.quantile_diagnostics(actual, predictions)
    assert result["rows"] == 1
    assert result["below_q50"] == 0.0
    assert result["below_q90"] == 1.0
    assert result["coverage_width_median"] == pytest.approx(0.13)
