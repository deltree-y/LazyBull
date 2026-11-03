# -*- coding: utf-8 -*-
"""P5b T5：两个裁判的方向自检与互不抵扣（§5.4.2 末 + §5.4.4 方向 fixture）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

import p5b_t5_env
from src.lazybull.v2.hosts.testbed.referee.drawdown import (
    DIRECTIONS,
    IMPROVE_EVENT,
    WORSEN_EVENT,
    AaReadings,
    DrawdownThresholdError,
    DrawdownThresholds,
    assert_events_disjoint,
    direction_self_check,
    drawdown_look_plan,
    improve_event,
    reading_matrix,
    train_drawdown_thresholds,
    worsen_event,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import baseline_percent_by_look

PLAN = drawdown_look_plan()


def _thresholds(worsen, improve) -> DrawdownThresholds:
    return DrawdownThresholds(
        plan=PLAN,
        alpha_train=0.045,
        alpha_per_look=0.045 / PLAN.k,
        worsen=tuple(float(value) for value in worsen),
        improve=tuple(float(value) for value in improve),
        worsen_by_scene={"scene": tuple(float(value) for value in worsen)},
        improve_by_scene={"scene": tuple(float(value) for value in improve)},
        worsen_units=(),
        improve_units=(),
        determining_worsen=("scene|dd_worsen",) * PLAN.k,
        determining_improve=("scene|dd_improve",) * PLAN.k,
    )


def _reading(arm_pp: float, base_pp: float = -20.0) -> np.ndarray:
    """构造 MDD 恰为给定值的 arm/base 路径并返回 D_k 读数矩阵 (1, K)。"""
    base = p5b_t5_env.flat_path(base_pp)[None, :]
    arm = p5b_t5_env.flat_path(arm_pp)[None, :]
    return reading_matrix(arm, baseline_percent_by_look(base[0], PLAN.look_days), PLAN)


def test_improvement_fixture_only_triggers_improve():
    """base=-20%、arm=-10% ⇒ D=+10pp：只允许移向 DD-I 检出，不得触发 DD-W。"""
    d_matrix = _reading(-10.0)
    assert np.allclose(d_matrix, 10.0, atol=1e-9)
    thresholds = _thresholds([-5.0] * PLAN.k, [5.0] * PLAN.k)
    assert improve_event(d_matrix, thresholds.improve).all()
    assert not worsen_event(d_matrix, thresholds.worsen).any()
    assert_events_disjoint(d_matrix, thresholds.worsen, thresholds.improve)


def test_worsening_fixture_only_triggers_worsen():
    """base=-20%、arm=-30% ⇒ D=-10pp：只允许移向 DD-W 否决，不得触发 DD-I。"""
    d_matrix = _reading(-30.0)
    assert np.allclose(d_matrix, -10.0, atol=1e-9)
    thresholds = _thresholds([-5.0] * PLAN.k, [5.0] * PLAN.k)
    assert worsen_event(d_matrix, thresholds.worsen).all()
    assert not improve_event(d_matrix, thresholds.improve).any()


def test_no_change_fixture_triggers_neither():
    """D=0 既不算恶化否决也不算改善检出（"未触发否决" ≠ "检出成功"）。"""
    d_matrix = _reading(-20.0)
    assert np.allclose(d_matrix, 0.0, atol=1e-9)
    thresholds = _thresholds([-5.0] * PLAN.k, [5.0] * PLAN.k)
    assert not worsen_event(d_matrix, thresholds.worsen).any()
    assert not improve_event(d_matrix, thresholds.improve).any()


def test_events_are_not_complements():
    """两个事件不是互补：小改善（+1pp）既不触发检出也不触发否决。"""
    thresholds = _thresholds([-5.0] * PLAN.k, [5.0] * PLAN.k)
    d_matrix = _reading(-19.0)  # D = +1
    assert not improve_event(d_matrix, thresholds.improve).any()
    assert not worsen_event(d_matrix, thresholds.worsen).any()


def test_direction_check_requires_c_i_strictly_greater():
    assert direction_self_check([-5.0] * 4, [5.0] * 4) == []
    errors = direction_self_check([-5.0, 5.0, -5.0, -5.0], [5.0, 5.0, 5.0, 5.0])
    assert len(errors) == 1 and "look 1" in errors[0]
    assert direction_self_check([0.0] * 4, [0.0] * 4)


def test_training_rejects_degenerate_aa():
    """A/A 全部为 0 ⇒ c^W = c^I = 0 ⇒ 训练必须 fail-closed（不得产生退化阈值）。"""
    degenerate = AaReadings(scene="s", d_matrix=np.zeros((120, PLAN.k)), realization_id="r")
    with pytest.raises(DrawdownThresholdError):
        train_drawdown_thresholds([degenerate], PLAN)


def test_trained_thresholds_take_extremes_over_scene():
    rng = np.random.default_rng(7)
    wide = rng.uniform(-20.0, 20.0, size=(400, PLAN.k))
    narrow = rng.uniform(-1.0, 1.0, size=(400, PLAN.k))
    thresholds = train_drawdown_thresholds(
        [
            AaReadings(scene="wide", d_matrix=wide, realization_id="r1"),
            AaReadings(scene="narrow", d_matrix=narrow, realization_id="r2"),
        ],
        PLAN,
    )
    alpha = 0.045 / PLAN.k
    for index in range(PLAN.k):
        expected_w = min(
            float(np.quantile(wide[:, index], alpha)), float(np.quantile(narrow[:, index], alpha))
        )
        expected_i = max(
            float(np.quantile(wide[:, index], 1 - alpha)),
            float(np.quantile(narrow[:, index], 1 - alpha)),
        )
        assert math.isclose(thresholds.worsen[index], expected_w, abs_tol=1e-12)
        assert math.isclose(thresholds.improve[index], expected_i, abs_tol=1e-12)
        assert thresholds.improve[index] > thresholds.worsen[index]
    assert thresholds.determining_worsen[0] == "wide|dd_worsen"
    assert thresholds.determining_improve[0] == "wide|dd_improve"


def test_event_names_are_distinct_and_frozen():
    assert WORSEN_EVENT != IMPROVE_EVENT
    assert WORSEN_EVENT == "vetoed_by_calibrated_drawdown_worsening"
    assert IMPROVE_EVENT.startswith("detected_by_calibrated_drawdown_improvement")
    assert set(DIRECTIONS) == {"dd_worsen", "dd_improve"}


def test_threshold_dict_keeps_two_directions_apart():
    thresholds = _thresholds([-5.0] * PLAN.k, [5.0] * PLAN.k)
    payload = thresholds.as_dict()
    assert payload["direction_check"]["passed"] is True
    assert payload["events"]["worsen"] != payload["events"]["improve"]
    assert payload["c_i_by_look"] != payload["c_w_by_look"]


def test_training_rejects_degenerate_or_flipped_thresholds():
    """位置检查：A/A 中位数必须落在 (c^W, c^I) 内部（方向写反/阈值退化时先失败）。"""
    from src.lazybull.v2.hosts.testbed.referee.drawdown.thresholds import _check_median_inside

    _check_median_inside(
        {"s": [0.0] * PLAN.k}, {"s": [-5.0] * PLAN.k}, {"s": [5.0] * PLAN.k}
    )
    with pytest.raises(DrawdownThresholdError):
        _check_median_inside(
            {"s": [6.0] * PLAN.k}, {"s": [-5.0] * PLAN.k}, {"s": [5.0] * PLAN.k}
        )
    with pytest.raises(DrawdownThresholdError):
        _check_median_inside(
            {"s": [0.0] * PLAN.k}, {"s": [0.0] * PLAN.k}, {"s": [0.0] * PLAN.k}
        )
