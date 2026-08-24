"""风控模型推理侧百分位特征闭环测试"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

import pytest

from src.lazybull.risk.position_risk import (
    PositionRiskConfig,
    PositionRiskModel,
    PositionRiskMonitor,
)


def _make_model(feature_names):
    clf = MagicMock()
    clf.predict.side_effect = lambda X: np.array([1] * len(X))
    clf.predict_proba.side_effect = lambda X: np.array([[0.2, 0.6, 0.2]] * len(X))
    config = PositionRiskConfig(model_version=1, feature_names=feature_names)
    return PositionRiskModel(config, clf)


def test_predict_batch_missing_pct_rejected():
    """模型层不得在未知范围的批次内自行计算截面百分位。"""
    model = _make_model(["ret_5", "pct_ret_5"])
    df = pd.DataFrame({"ts_code": ["A", "B", "C"], "ret_5": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="完整当日截面"):
        model.predict_batch(df)


def test_predict_batch_keeps_existing_pct_columns():
    """截面已含 pct_* 列时原样使用, 不覆盖。"""
    model = _make_model(["ret_5", "pct_ret_5"])
    df = pd.DataFrame({"ts_code": ["A", "B"], "ret_5": [1.0, 2.0], "pct_ret_5": [0.25, 0.75]})
    model.predict_batch(df)
    X = model._clf.predict.call_args[0][0]
    assert np.isclose(X[0, 1], 0.25)


def test_predict_single_missing_pct_rejected():
    """单股无完整截面时不得用 NaN 冒充训练期百分位。"""
    model = _make_model(["ret_5", "pct_ret_5"])
    features = pd.Series({"ts_code": "A", "ret_5": 1.0})
    with pytest.raises(ValueError, match="完整当日截面"):
        model.predict_single(features)


def test_monitor_evaluate_position_uses_full_cross_section_pct():
    """单股监控必须先按完整当日截面计算百分位，不能提前返回 HOLD。"""
    model = _make_model(["ret_5", "pct_ret_5"])
    monitor = PositionRiskMonitor(model)
    features_df = pd.DataFrame({"ts_code": ["A", "B", "C"], "ret_5": [1.0, 2.0, 3.0]})

    result = monitor.evaluate_position("A", "20260105", features_df)

    assert result.ts_code == "A"
    X = model._clf.predict.call_args[0][0]
    assert X.shape == (1, 2)
    assert np.isclose(X[0, 1], 1 / 3)


def test_monitor_evaluate_positions_ranks_before_holding_filter():
    """批量监控必须先在全市场排名，再筛持仓子集。"""
    model = _make_model(["ret_5", "pct_ret_5"])
    monitor = PositionRiskMonitor(model)
    features_df = pd.DataFrame({"ts_code": ["A", "B", "C"], "ret_5": [1.0, 2.0, 3.0]})

    results = monitor.evaluate_positions(
        [{"ts_code": "A"}, {"ts_code": "C"}], "20260105", features_df
    )

    assert set(results) == {"A", "C"}
    X = model._clf.predict.call_args[0][0]
    assert np.isclose(X[0, 1], 1 / 3)
    assert np.isclose(X[1, 1], 1.0)
