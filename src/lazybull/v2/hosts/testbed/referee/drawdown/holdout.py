# -*- coding: utf-8 -*-
"""P5b T5 回撤裁判 holdout 验收（设计 v4 §5.4.4 + §7.2「逐场景×零假设」行）。

验收口径
========
- 验收单元 = **场景 × 零假设（DD-W/DD-I）× H × 裁判**；H=20 单一取值 ⇒
  4 场景 × 2 裁判 = **8 个单元**，逐单元单独判定，**禁止跨场景/跨单元加权或混池**；
- 判定：每单元 family-wise 频率的 Wilson 95% **上界 ≤ 6%**（与收益裁判同口径；
  manifest 可冻结更严值，**不得放宽**）；任一单元不过 ⇒ 该裁判不可用；
- A/A 使用**同分布独立路径**（:mod:`thresholds` 的结构断言），事件与功效侧同一事件；
- 本模块**只读**临界值：入参是已冻结的 ``c^W_k`` / ``c^I_k``，不返回任何可写回路径。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
    DIRECTIONS,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.sampling import IMPLEMENTATION_ID
from src.lazybull.v2.hosts.testbed.referee.drawdown.thresholds import (
    AaReadings,
    DrawdownThresholds,
    aa_summary,
    event_overlap_count,
    improve_event,
    worsen_event,
)
from src.lazybull.v2.hosts.testbed.referee.holdout import WILSON_Z, wilson_interval
from src.lazybull.v2.hosts.testbed.referee.statistics import LookPlan

__all__ = [
    "HOLDOUT_SCHEMA",
    "REPORTING_RULE",
    "DrawdownHoldoutError",
    "evaluate_drawdown_holdout",
    "unit_keys",
    "wilson_upper_gate",
]

#: 产物 schema
HOLDOUT_SCHEMA = "p5b_testbed_drawdown_holdout/v1"
#: 报告口径（禁止聚合的唯一声明点）
REPORTING_RULE = "per_unit_only_no_pooling_no_weighting"
#: 冻结 Wilson 上界门（0.06；不得放宽）
WILSON_UPPER_MAX = 0.06


class DrawdownHoldoutError(TestbedError):
    """holdout 输入非法（门槛被放宽/读数与阈值不匹配/场景缺失）。"""

    __test__ = False


def _validate_gate(wilson_upper_max: float, z: float) -> Tuple[float, float]:
    gate = float(wilson_upper_max)
    if gate > WILSON_UPPER_MAX:
        raise DrawdownHoldoutError(f"Wilson 上界门不得放宽到 {WILSON_UPPER_MAX} 以上，实际 {gate}")
    if not math.isclose(float(z), WILSON_Z, rel_tol=0.0, abs_tol=1e-9):
        raise DrawdownHoldoutError(f"Wilson 分位 z 必须等于冻结值 {WILSON_Z}，实际 {z}")
    return gate, float(z)


def wilson_upper_gate(successes: int, total: int, *, z: float = WILSON_Z) -> Tuple[float, float]:
    """返回 ``(low, high)``（Wilson 95% 区间；``total<=0`` 即报错）。"""
    return wilson_interval(int(successes), int(total), z=float(z))


def _direction_flags(
    d_matrix: np.ndarray, direction: str, thresholds: DrawdownThresholds
) -> np.ndarray:
    if direction == DIRECTION_WORSEN:
        return worsen_event(d_matrix, thresholds.worsen)
    if direction == DIRECTION_IMPROVE:
        return improve_event(d_matrix, thresholds.improve)
    raise DrawdownHoldoutError(f"未知裁判方向 {direction!r}（允许 {DIRECTIONS}）")


def _per_look_counts(
    d_matrix: np.ndarray, direction: str, thresholds: DrawdownThresholds
) -> List[int]:
    values = np.asarray(d_matrix, dtype=float)
    limits = thresholds.worsen if direction == DIRECTION_WORSEN else thresholds.improve
    counts = []
    for index, limit in enumerate(limits):
        column = values[:, index]
        hits = column < limit if direction == DIRECTION_WORSEN else column > limit
        counts.append(int(np.sum(np.isfinite(column) & hits)))
    return counts


def evaluate_drawdown_holdout(
    readings: Sequence[AaReadings],
    thresholds: DrawdownThresholds,
    *,
    plan: LookPlan,
    count: int,
    scenes: Optional[Sequence[str]] = None,
    wilson_upper_max: float = WILSON_UPPER_MAX,
    z: float = WILSON_Z,
) -> Dict[str, Any]:
    """逐单元（场景 × 裁判）holdout 验收（只读阈值，无阈值回写路径）。

    Args:
        readings: holdout 阶段的 A/A 读数（逐场景；arm/base 为独立路径）。
        thresholds: 已冻结的两组临界值。
        plan: look 映射。
        count: 每单元实现数（manifest 冻结值 10000）。
        scenes: 冻结场景集（给出时与读数集合逐项比对，缺/多即拒绝）。
        wilson_upper_max: Wilson 上界门（冻结 0.06，不得放宽）。
        z: Wilson 分位（冻结值）。

    Returns:
        ``{schema, units, failed_units, usable_by_direction, referee_usable,
        holdout_gate_only, aa_summary, cross_event_overlap, ...}``。

    Raises:
        DrawdownHoldoutError: 门槛被放宽、读数缺失/多出场景、实现数与声明不符。
    """
    gate, quantile = _validate_gate(wilson_upper_max, z)
    items = list(readings)
    if not items:
        raise DrawdownHoldoutError("holdout A/A 读数为空（不得用空单元冒充通过）")
    if scenes is not None:
        expected = sorted(str(item) for item in scenes)
        actual = sorted(item.scene for item in items)
        if actual != expected:
            raise DrawdownHoldoutError(
                f"holdout 场景集与冻结值不符（禁止删困难场景/多加场景）: "
                f"冻结 {expected} 实际 {actual}"
            )
    units: Dict[str, Any] = {}
    failed: List[str] = []
    usable: Dict[str, bool] = {}
    for direction in DIRECTIONS:
        usable[direction] = True
    for item in items:
        if item.looks != int(plan.k):
            raise DrawdownHoldoutError(
                f"场景 {item.scene!r} 的探索 look 数 {item.looks} 与 K={plan.k} 不符"
            )
        if int(item.samples) != int(count):
            raise DrawdownHoldoutError(
                f"场景 {item.scene!r} 的实现数 {item.samples} 与冻结值 {count} 不符"
                "（禁止下调实现数或改写分母）"
            )
        for direction in DIRECTIONS:
            flags = _direction_flags(item.d_matrix, direction, thresholds)
            successes = int(flags.sum())
            total = int(flags.size)
            low, high = wilson_upper_gate(successes, total, z=quantile)
            reasons: List[str] = []
            if high > gate:
                reasons.append("wilson_upper_exceeds_max")
            key = f"{item.scene}|{direction}"
            units[key] = {
                "unit": key,
                "scene": item.scene,
                "null": direction,
                "H": int(plan.h),
                "referee": direction,
                "realization_id": item.realization_id,
                "successes": successes,
                "total": total,
                "rate": successes / total,
                "wilson_low": low,
                "wilson_high": high,
                "per_look_triggers": _per_look_counts(item.d_matrix, direction, thresholds),
                "thresholds_used": [
                    float(value)
                    for value in (
                        thresholds.worsen if direction == DIRECTION_WORSEN else thresholds.improve
                    )
                ],
                "event": (
                    "any_k(D_k < c^W_k)"
                    if direction == DIRECTION_WORSEN
                    else "any_k(I_k = D_k > c^I_k)"
                ),
                "pass": not reasons,
                "failure_reasons": reasons,
            }
            if reasons:
                failed.append(key)
                usable[direction] = False
    overlaps = {
        item.scene: event_overlap_count(item.d_matrix, thresholds.worsen, thresholds.improve)
        for item in items
    }
    return {
        "schema": HOLDOUT_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "plan": plan.as_dict(),
        "unit_definition": "scene × null(DD-W/DD-I) × H × referee",
        "unit_count": len(units),
        "scenes": sorted(item.scene for item in items),
        "reporting_rule": REPORTING_RULE,
        "wilson_confidence": 0.95,
        "wilson_z": quantile,
        "wilson_upper_max": gate,
        "aa_independent_paths": True,
        "counts": {"holdout_per_unit": int(count)},
        "units": units,
        "failed_units": failed,
        "usable_by_direction": {key: bool(value) for key, value in usable.items()},
        "referee_usable": all(usable.values()),
        "holdout_gate_only": True,
        "holdout_gate_note": (
            "referee_usable 只表示 A/A（Wilson 上界门）通过，**不含**注入与 MDE 是否成立；"
            "H5 三条件见判定表 h5_conditions"
        ),
        "aa_summary": aa_summary(items),
        "cross_event_overlap": overlaps,
        "cross_event_overlap_note": (
            "同一读数上 family-wise 恶化否决与改善检出同时成立的实现数（披露量，不阻断）"
        ),
        "per_direction_failure_note": (
            "DD-W 与 DD-I 分别判定：任一方向有单元不过 ⇒ 该方向不可用；" "两者不得互相抵扣"
        ),
    }


def unit_keys(scenes: Sequence[str]) -> List[str]:
    """验收单元键清单（供测试与报告核对，避免聚合口径）。"""
    return [f"{scene}|{direction}" for scene in scenes for direction in DIRECTIONS]
