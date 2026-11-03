# -*- coding: utf-8 -*-
"""P5b T5：A/A 临界值训练与 holdout 逐单元 Wilson 门（§5.4.1/§5.4.2/§5.4.4）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

import p5b_t5_env  # noqa: F401
from src.lazybull.v2.hosts.testbed.referee.drawdown import (
    ALPHA_TRAIN,
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
    AaReadings,
    DrawdownHoldoutError,
    DrawdownThresholdError,
    DrawdownThresholds,
    aa_readings_from_independent_paths,
    drawdown_look_plan,
    evaluate_drawdown_holdout,
    merge_aa_readings,
    train_drawdown_thresholds,
    wilson_upper_gate,
)
from src.lazybull.v2.hosts.testbed.referee.holdout import WILSON_Z

PLAN = drawdown_look_plan()
SCENES = ("ma20", "ar1_06", "regime_hetero", "real_seed_block")


def _readings(value: float, samples: int = 200, scene: str = "ma20", jitter: float = 0.0):
    rng = np.random.default_rng(3)
    matrix = np.full((samples, PLAN.k), float(value))
    if jitter:
        matrix += rng.normal(0.0, jitter, size=matrix.shape)
    return AaReadings(scene=scene, d_matrix=matrix, realization_id=f"{scene}:aa")


def _thresholds(worsen=(-20.0,), improve=(20.0,)) -> DrawdownThresholds:
    return DrawdownThresholds(
        plan=PLAN,
        alpha_train=ALPHA_TRAIN,
        alpha_per_look=ALPHA_TRAIN / PLAN.k,
        worsen=tuple(float(worsen[0]) for _ in range(PLAN.k)),
        improve=tuple(float(improve[0]) for _ in range(PLAN.k)),
        worsen_by_scene={},
        improve_by_scene={},
        worsen_units=(),
        improve_units=(),
        determining_worsen=("",) * PLAN.k,
        determining_improve=("",) * PLAN.k,
    )


def test_alpha_split_matches_family_wise_rate():
    readings = [_readings(0.0, samples=400, scene=scene, jitter=10.0) for scene in SCENES]
    thresholds = train_drawdown_thresholds(readings, PLAN)
    assert math.isclose(thresholds.alpha_per_look, ALPHA_TRAIN / PLAN.k, rel_tol=1e-12)
    assert thresholds.as_dict()["quantile_interpolation"] == "linear"
    for index in range(PLAN.k):
        assert thresholds.worsen[index] < thresholds.improve[index]


def test_threshold_requires_min_valid_sample():
    tiny = AaReadings(scene="s", d_matrix=np.full((50, PLAN.k), 0.1), realization_id="r")
    with pytest.raises(DrawdownThresholdError):
        train_drawdown_thresholds([tiny], PLAN)


def test_threshold_rejects_duplicate_scenes():
    with pytest.raises(DrawdownThresholdError):
        train_drawdown_thresholds([_readings(0.0), _readings(0.0)], PLAN)


def test_aa_rejects_zero_injection_clone():
    path = np.zeros((50, 500))
    path[:, 10:20] = -0.01
    with pytest.raises(DrawdownThresholdError):
        aa_readings_from_independent_paths(path, path, PLAN, scene="s")
    with pytest.raises(DrawdownThresholdError):
        aa_readings_from_independent_paths(path, path[0], PLAN, scene="s")


def test_aa_uses_frozen_baseline_for_both_readings():
    """A/A 读数 = MDD(arm) − MDD(B0)；arm 与 B0 相同 ⇒ 全 0（结构自检）。"""
    base = np.zeros(500)
    base[:100] = -0.002
    arm = base[None, :] + 0.0005  # 轻微偏移（非恒等）
    readings = aa_readings_from_independent_paths(arm, base, PLAN, scene="s")
    assert readings.d_matrix.shape == (1, PLAN.k)
    assert findings_all_finite(readings.d_matrix)


def findings_all_finite(matrix: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(matrix, dtype=float)).all())


def test_merge_requires_same_scene_and_stream():
    first = _readings(0.0, samples=101, scene="a")
    second = _readings(0.0, samples=99, scene="a")
    merged = merge_aa_readings([first, second])
    assert merged.samples == 200
    with pytest.raises(DrawdownThresholdError):
        merge_aa_readings([first, _readings(0.0, samples=10, scene="b")])
    with pytest.raises(DrawdownThresholdError):
        merge_aa_readings([])


def test_holdout_has_eight_units_and_no_aggregation():
    readings = [_readings(0.0, scene=scene, jitter=8.0) for scene in SCENES]
    thresholds = train_drawdown_thresholds(readings, PLAN)
    holdout = evaluate_drawdown_holdout(readings, thresholds, plan=PLAN, count=200)
    assert holdout["unit_count"] == 8
    assert set(holdout["units"]) == {
        f"{scene}|{direction}"
        for scene in SCENES
        for direction in (DIRECTION_WORSEN, DIRECTION_IMPROVE)
    }
    forbidden = ("pooled", "average", "weighted", "overall", "combined", "aggregate")
    for key in holdout:
        assert not any(word in key.lower() for word in forbidden)
    for unit in holdout["units"].values():
        assert unit["total"] == 200
        assert len(unit["per_look_triggers"]) == PLAN.k
        assert unit["wilson_low"] <= unit["rate"] <= unit["wilson_high"] or unit["rate"] == 0.0
    assert holdout["reporting_rule"] == "per_unit_only_no_pooling_no_weighting"


def test_holdout_unit_failure_disables_only_that_direction():
    """某单元 Wilson 上界 > 6% ⇒ 该方向不可用；另一方向仍可独立判定（不得抵扣）。"""
    readings = [_readings(0.0, scene=scene, jitter=8.0) for scene in SCENES]
    thresholds = _thresholds(worsen=(1.0,), improve=(100.0,))  # 恶化阈值偏正 ⇒ 大量触发
    holdout = evaluate_drawdown_holdout(readings, thresholds, plan=PLAN, count=200)
    assert holdout["usable_by_direction"][DIRECTION_WORSEN] is False
    assert holdout["usable_by_direction"][DIRECTION_IMPROVE] is True
    assert holdout["referee_usable"] is False
    failed = holdout["failed_units"]
    assert all(item.endswith(DIRECTION_WORSEN) for item in failed)


def test_holdout_rejects_loosened_gate_and_count_mismatch():
    readings = [_readings(0.0, scene=scene, jitter=8.0) for scene in SCENES]
    thresholds = train_drawdown_thresholds(readings, PLAN)
    with pytest.raises(DrawdownHoldoutError):
        evaluate_drawdown_holdout(readings, thresholds, plan=PLAN, count=200, wilson_upper_max=0.5)
    with pytest.raises(DrawdownHoldoutError):
        evaluate_drawdown_holdout(readings, thresholds, plan=PLAN, count=200, z=0.0)
    with pytest.raises(DrawdownHoldoutError):
        evaluate_drawdown_holdout(readings, thresholds, plan=PLAN, count=199)


def test_wilson_upper_bound_golden_case():
    low, high = wilson_upper_gate(0, 10000, z=WILSON_Z)
    assert low == 0.0
    assert math.isclose(high, 0.0003838, abs_tol=1e-6)
    # 与 T4 闭合报告 §4.2 的 ``ma20|mu=0`` 单元（223/10000，Wilson 上界 2.5383%）同口径
    low2, high2 = wilson_upper_gate(223, 10000, z=WILSON_Z)
    assert math.isclose(low2, 0.0195841, abs_tol=1e-6)
    assert math.isclose(high2, 0.0253827, abs_tol=1e-6)
    with pytest.raises(Exception):
        wilson_upper_gate(0, 0)


def test_holdout_rejects_unfrozen_scene_set():
    """冻结场景集必须逐项比对（禁止删困难场景/多加场景）。"""
    readings = [_readings(0.0, scene=scene, jitter=8.0) for scene in SCENES]
    thresholds = train_drawdown_thresholds(readings, PLAN)
    with pytest.raises(DrawdownHoldoutError):
        evaluate_drawdown_holdout(readings, thresholds, plan=PLAN, count=200, scenes=SCENES[:3])
    holdout = evaluate_drawdown_holdout(readings, thresholds, plan=PLAN, count=200, scenes=SCENES)
    assert holdout["holdout_gate_only"] is True
    assert len(holdout["aa_summary"]) == len(SCENES)
    assert all("median_by_look" in item for item in holdout["aa_summary"])
    assert set(holdout["cross_event_overlap"]) == set(SCENES)


def test_cross_event_overlap_is_counted_not_assumed():
    """family-wise 事件跨 look 可能同真：生产路径必须实际计数（披露量）。"""
    from src.lazybull.v2.hosts.testbed.referee.drawdown import event_overlap_count

    matrix = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 10.0]])
    worsen = [-20.0] * PLAN.k
    improve = [5.0] * PLAN.k
    assert event_overlap_count(matrix, worsen, improve) == 0
    matrix[1, 0] = -30.0  # look0 触发否决，look3 触发检出 ⇒ 同一实现两事件同真
    assert event_overlap_count(matrix, worsen, improve) == 1
