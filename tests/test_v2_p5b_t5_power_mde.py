# -*- coding: utf-8 -*-
"""P5b T5：功效、可构造率与两个 MDE（§5.3 表 + §5.5 回撤行 + §5.4.4）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

import p5b_t5_env  # noqa: F401
from src.lazybull.v2.hosts.testbed.referee.drawdown import (
    ALPHA_TRAIN,
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
    EFFECT_GRID_PP,
    MDE_IMPROVE_KEY,
    MDE_NOT_REACHED,
    MDE_NOT_REGISTERED,
    MDE_WORSEN_KEY,
    DrawdownPowerError,
    DrawdownThresholds,
    EffectCell,
    drawdown_look_plan,
    evaluate_drawdown_power,
    register_mde,
)

PLAN = drawdown_look_plan()
SCENES = ("ma20", "ar1_06")


def _thresholds() -> DrawdownThresholds:
    return DrawdownThresholds(
        plan=PLAN,
        alpha_train=ALPHA_TRAIN,
        alpha_per_look=ALPHA_TRAIN / PLAN.k,
        worsen=(-5.0,) * PLAN.k,
        improve=(5.0,) * PLAN.k,
        worsen_by_scene={},
        improve_by_scene={},
        worsen_units=(),
        improve_units=(),
        determining_worsen=("",) * PLAN.k,
        determining_improve=("",) * PLAN.k,
    )


def _cell(
    direction: str,
    level: float,
    scene: str,
    *,
    detected_fraction: float,
    total: int = 100,
    not_constructible_fraction: float = 0.0,
) -> EffectCell:
    """构造一个单元：前 ``detected_fraction`` 的行产生事件，其余不触发。"""
    hits = int(round(detected_fraction * total))
    sign = 1.0 if direction == DIRECTION_IMPROVE else -1.0
    d_matrix = np.full((total, PLAN.k), -sign * 10.0)  # 默认不触发
    d_matrix[:hits, 0] = sign * 50.0  # 第一 look 触发
    constructible = np.ones(total, dtype=bool)
    blocked = int(round(not_constructible_fraction * total))
    if blocked:
        constructible[-blocked:] = False
        d_matrix[-blocked:] = np.nan
    return EffectCell(
        direction=direction,
        level_pp=float(level),
        scene=scene,
        d_matrix=d_matrix,
        constructible=constructible,
        parameter=np.where(constructible, 0.3, np.nan),
        pure_pp=np.where(constructible, sign * level, np.nan),
        observed_pp=np.where(constructible, sign * 1.0, np.nan),
        realization_id=f"{scene}:{direction}:{level}",
    )


def _cells(rates, *, direction: str = DIRECTION_IMPROVE, not_constructible=0.0):
    return [
        _cell(
            direction,
            level,
            scene,
            detected_fraction=rates.get((level, scene), 0.0),
            not_constructible_fraction=not_constructible,
        )
        for level in EFFECT_GRID_PP
        for scene in SCENES
    ]


def test_mde_picks_smallest_level_above_target():
    rates = {(8.0, "ma20"): 0.85, (8.0, "ar1_06"): 0.10, (12.0, "ar1_06"): 0.90}
    power = evaluate_drawdown_power(
        _cells(rates) + _cells({}, direction=DIRECTION_WORSEN),
        _thresholds(),
        plan=PLAN,
        applicable={DIRECTION_IMPROVE: True, DIRECTION_WORSEN: True},
    )
    improve = power["mde"][MDE_IMPROVE_KEY]
    assert improve["by_scene"]["ma20"] == 8.0
    assert improve["by_scene"]["ar1_06"] == 12.0
    assert improve["status_by_scene"]["ma20"] == "registered"
    worsen = power["mde"][MDE_WORSEN_KEY]
    assert worsen["by_scene"]["ma20"] is None
    assert worsen["status_by_scene"]["ma20"] == MDE_NOT_REACHED


def test_mde_marks_grid_upper_bound():
    rates = {(12.0, "ma20"): 0.80, (12.0, "ar1_06"): 0.81}
    power = evaluate_drawdown_power(
        _cells(rates) + _cells({}, direction=DIRECTION_WORSEN),
        _thresholds(),
        plan=PLAN,
        applicable={DIRECTION_IMPROVE: True, DIRECTION_WORSEN: True},
    )
    improve = power["mde"][MDE_IMPROVE_KEY]
    assert improve["at_grid_upper_bound"] == {"ma20": True, "ar1_06": True}
    assert improve["by_scene"]["ma20"] == 12.0


def test_mde_not_registered_when_aa_fails_for_that_direction():
    rates = {(12.0, "ma20"): 0.99}
    cells = _cells(rates) + _cells(rates, direction=DIRECTION_WORSEN)
    power = evaluate_drawdown_power(
        cells,
        _thresholds(),
        plan=PLAN,
        applicable={DIRECTION_IMPROVE: True, DIRECTION_WORSEN: False},
    )
    assert power["mde"][MDE_WORSEN_KEY]["registered"] is False
    assert power["mde"][MDE_WORSEN_KEY]["by_scene"]["ma20"] is None
    assert power["mde"][MDE_WORSEN_KEY]["status_by_scene"]["ma20"] == MDE_NOT_REGISTERED
    assert power["mde"][MDE_IMPROVE_KEY]["by_scene"]["ma20"] == 12.0


def test_not_constructible_counts_as_not_detected():
    """不可构造的实现按未检出计入主分母 M，C/M 单独披露（不得剔除）。"""
    cell = _cell(
        DIRECTION_IMPROVE,
        8.0,
        "ma20",
        detected_fraction=0.5,
        total=100,
        not_constructible_fraction=0.4,
    )
    cells = (
        [cell]
        + [
            _cell(DIRECTION_IMPROVE, level, scene, detected_fraction=0.0)
            for level in EFFECT_GRID_PP
            for scene in SCENES
            if not (level == 8.0 and scene == "ma20")
        ]
        + [
            _cell(DIRECTION_WORSEN, level, scene, detected_fraction=0.0)
            for level in EFFECT_GRID_PP
            for scene in SCENES
        ]
    )
    power = evaluate_drawdown_power(
        cells,
        _thresholds(),
        plan=PLAN,
        applicable={DIRECTION_IMPROVE: True, DIRECTION_WORSEN: True},
    )
    entry = power["by_direction_level_scene"][DIRECTION_IMPROVE]["8.0"]["ma20"]
    assert entry["total"] == 100
    assert entry["successes"] == 50
    assert entry["rate"] == 0.5
    assert entry["constructible_count"] == 60
    assert math.isclose(entry["constructibility_rate"], 0.6, rel_tol=1e-12)
    assert entry["not_constructible_count"] == 40
    mde = power["mde"]
    assert 8.0 in mde["not_constructible_levels"][DIRECTION_IMPROVE]["ma20"]
    assert math.isclose(
        mde["constructibility_by_scene"][DIRECTION_IMPROVE]["ma20"]["8.0"], 0.6, rel_tol=1e-12
    )
    assert "未检出" in mde["not_constructible_note"]


def test_power_rejects_grid_change_and_missing_applicable():
    cells = _cells({}) + _cells({}, direction=DIRECTION_WORSEN)
    with pytest.raises(DrawdownPowerError):
        evaluate_drawdown_power(
            cells,
            _thresholds(),
            plan=PLAN,
            applicable={DIRECTION_IMPROVE: True, DIRECTION_WORSEN: True},
            grid=(1.0, 2.0),
        )
    with pytest.raises(DrawdownPowerError):
        evaluate_drawdown_power(
            cells, _thresholds(), plan=PLAN, applicable={DIRECTION_IMPROVE: True}
        )
    with pytest.raises(DrawdownPowerError):
        evaluate_drawdown_power(
            cells,
            _thresholds(),
            plan=PLAN,
            applicable={DIRECTION_IMPROVE: True, DIRECTION_WORSEN: True},
            target=0.5,
        )


def test_power_requires_every_cell():
    """场景集由 cells 决定：某场景缺档 ⇒ 效应格缺失，必须报错（不得删困难场景）。"""
    cells = (
        [_cell(DIRECTION_IMPROVE, level, "ma20", detected_fraction=0.0) for level in EFFECT_GRID_PP]
        + [
            _cell(DIRECTION_WORSEN, level, "ma20", detected_fraction=0.0)
            for level in EFFECT_GRID_PP
        ]
        + [_cell(DIRECTION_IMPROVE, 1.0, "ar1_06", detected_fraction=0.0)]
    )
    with pytest.raises(DrawdownPowerError):
        evaluate_drawdown_power(
            cells,
            _thresholds(),
            plan=PLAN,
            applicable={DIRECTION_IMPROVE: True, DIRECTION_WORSEN: True},
        )


def test_register_mde_reports_not_reached_honestly():
    readings = {
        DIRECTION_WORSEN: {float(level): {"ma20": 0.1} for level in EFFECT_GRID_PP},
        DIRECTION_IMPROVE: {float(level): {"ma20": 0.1} for level in EFFECT_GRID_PP},
    }
    mde = register_mde(True, readings, scenes=("ma20",), target=0.8)
    assert mde[MDE_WORSEN_KEY]["status_by_scene"]["ma20"] == MDE_NOT_REACHED
    assert mde[MDE_WORSEN_KEY]["by_scene"]["ma20"] is None
    assert mde["mde_requires_aa_pass"] is True
    assert mde["pooled_average_forbidden"] is True
    assert mde["grid_upper_pp"] == 12.0


def test_effect_cell_rejects_row_count_mismatch():
    with pytest.raises(DrawdownPowerError):
        EffectCell(
            direction=DIRECTION_IMPROVE,
            level_pp=1.0,
            scene="ma20",
            d_matrix=np.zeros((4, PLAN.k)),
            constructible=np.ones(3, dtype=bool),
            parameter=np.zeros(4),
            pure_pp=np.zeros(4),
            observed_pp=np.zeros(4),
            realization_id="r",
        )


def test_mde_flags_non_monotone_power_curve():
    """更大档位检出率下降时必须打标（主分母含不可构造实现，曲线可能非单调）。"""
    rates = {(8.0, "ma20"): 0.9, (12.0, "ma20"): 0.4}
    cells = _cells(rates) + _cells({}, direction=DIRECTION_WORSEN)
    power = evaluate_drawdown_power(
        cells,
        _thresholds(),
        plan=PLAN,
        applicable={DIRECTION_IMPROVE: True, DIRECTION_WORSEN: True},
    )
    improve = power["mde"][MDE_IMPROVE_KEY]
    assert improve["by_scene"]["ma20"] == 8.0
    assert improve["power_curve_monotone"]["ma20"] is False
    assert improve["power_curve_dips"]["ma20"][0]["to_level_pp"] == 12.0
    assert improve["aa_gate_passed"] is True
    assert improve["registered"] is True


def test_mde_registered_requires_an_actual_mde():
    """`registered` 语义务必是"存在 MDE"，不能只表示"A/A 过门"。"""
    cells = _cells({}) + _cells({}, direction=DIRECTION_WORSEN)
    power = evaluate_drawdown_power(
        cells,
        _thresholds(),
        plan=PLAN,
        applicable={DIRECTION_IMPROVE: True, DIRECTION_WORSEN: True},
    )
    worsen = power["mde"][MDE_WORSEN_KEY]
    assert worsen["aa_gate_passed"] is True
    assert worsen["registered"] is False
    assert power["mde"]["registered"] is False
    assert "nominal_vs_actual" in power["mde"]


def test_power_rejects_unfrozen_scene_set():
    cells = _cells({}) + _cells({}, direction=DIRECTION_WORSEN)
    with pytest.raises(DrawdownPowerError):
        evaluate_drawdown_power(
            cells,
            _thresholds(),
            plan=PLAN,
            applicable={DIRECTION_IMPROVE: True, DIRECTION_WORSEN: True},
            scenes=("ma20", "ar1_06", "regime_hetero", "real_seed_block"),
        )
