# -*- coding: utf-8 -*-
"""P5b T4 收益裁判功效与 MDE（设计 v4 §5.3 效应校准）。

口径
====
- 效应网格：``0.5δ / 1δ / 1.5δ / 3δ / 5δ`` 的**日度 log 位移**（不按 look 单独改 δ）；
- 检出事件与 A/A **同一事件**：``any_k(valid_k and T_k < -c_k)``；
- 每档、每场景报检出率 + Wilson 区间 + 实现数 + block 参数；
- ``MDE`` = **A/A（误报与无效率）全部过门后**，点估计检出率首次达到 80% 的最小效应档；
  没有则如实写 ``tested_to_5delta_not_80pct``；
- **禁止**把跨场景简单平均当成统一 MDE；不得宣称 5δ 通用灵敏（r3 已实测 MA20/AR1 在 5δ
  未达 80%）。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.holdout import WILSON_Z, wilson_interval
from src.lazybull.v2.hosts.testbed.referee.sampling import IMPLEMENTATION_ID
from src.lazybull.v2.hosts.testbed.referee.scenes import SceneRegistry
from src.lazybull.v2.hosts.testbed.referee.statistics import (
    DEFAULT_CHUNK,
    DELTA_LOG_DAILY,
    STATISTIC_PSD,
    LookPlan,
    veto_flags,
)
from src.lazybull.v2.hosts.testbed.referee.training import generate_scores

__all__ = [
    "EFFECT_GRID_DELTA",
    "MDE_NOT_REACHED",
    "MDE_NOT_REGISTERED",
    "POWER_SCHEMA",
    "TARGET_POWER",
    "PowerError",
    "evaluate_power",
    "register_mde",
]

#: 效应网格（δ 倍数；§5.5 冻结）
EFFECT_GRID_DELTA: tuple = (0.5, 1.0, 1.5, 3.0, 5.0)
#: 目标检出率（点估计口径）
TARGET_POWER = 0.80
#: 未达 80% 的如实表述
MDE_NOT_REACHED = "tested_to_5delta_not_80pct"
#: A/A 未过门时的登记状态
MDE_NOT_REGISTERED = "not_registered_aa_failed"
#: 产物 schema
POWER_SCHEMA = "p5b_testbed_income_power/v1"


class PowerError(TestbedError):
    """功效计算输入非法（效应档非正、阈值非法、注册表角色错误）。"""

    __test__ = False


def register_mde(
    applicable: bool,
    rates_by_level_scene: Mapping[str, Mapping[str, float]],
    grid: Sequence[float],
    scenes: Sequence[str],
    *,
    target: float = TARGET_POWER,
) -> Dict[str, Any]:
    """登记 MDE（仅当 A/A 过门）；逐场景给出最小达标档或未达标记。

    Args:
        applicable: A/A 与无效率是否全部过门。
        rates_by_level_scene: ``{"5.0": {"ma20": 0.62, ...}}`` 检出率（点估计）。
        grid: 效应档（δ 倍数，升序）。
        scenes: 场景名。
        target: 目标检出率（默认 0.80）。

    Returns:
        ``{"registered": bool, "by_scene": {scene: float|None}, "status_by_scene": {...}}``。
    """
    ordered = sorted(float(level) for level in grid)
    by_scene: Dict[str, Optional[float]] = {}
    status: Dict[str, str] = {}
    for scene in scenes:
        if not applicable:
            by_scene[scene] = None
            status[scene] = MDE_NOT_REGISTERED
            continue
        hit = [
            level
            for level in ordered
            if float(rates_by_level_scene.get(str(level), {}).get(scene, 0.0)) >= float(target)
        ]
        if hit:
            by_scene[scene] = float(min(hit))
            status[scene] = "registered"
        else:
            by_scene[scene] = None
            status[scene] = MDE_NOT_REACHED
    # 命中落在网格上界时只能宣称 "MDE <= 上界档"，不能当作精确 MDE 读数
    upper = float(ordered[-1]) if ordered else None
    at_upper = {
        scene: bool(by_scene[scene] is not None and upper is not None and by_scene[scene] >= upper)
        for scene in scenes
    }
    return {
        "registered": bool(applicable),
        "by_scene": by_scene,
        "status_by_scene": status,
        "at_grid_upper_bound": at_upper,
        "grid_upper_level": upper,
    }


def _validate_power_inputs(
    *,
    plan: LookPlan,
    scenes: Sequence[str],
    thresholds: Sequence[float],
    grid: Sequence[float],
    scale_by_scene: Optional[Mapping[str, float]],
    target: float,
    z: float,
    invalid_rate_per_look_max: float,
) -> Tuple[List[float], List[float], List[str]]:
    """阈值/网格/钩子/分位/目标的冻结口径校验（任何偏离即拒绝）。

    Returns:
        ``(frozen_thresholds, ordered_grid, scene_names)``。
    """
    frozen = [float(value) for value in thresholds]
    if len(frozen) != plan.k or not np.isfinite(frozen).all():
        raise PowerError(f"阈值非法（必须 K={plan.k} 个有限值）: {frozen}")
    ordered = sorted(float(level) for level in grid)
    if not ordered or min(ordered) <= 0:
        raise PowerError(f"效应档必须为正: {ordered}")
    if scale_by_scene is not None:
        unknown = sorted(set(scale_by_scene) - set(scenes))
        if unknown:
            raise PowerError(f"scale_by_scene 含未知场景: {unknown}")
        non_identity = [
            value
            for value in scale_by_scene.values()
            if not math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ]
        if non_identity:
            raise PowerError(
                "scale_by_scene 只允许恒等映射（1.0）；任何缩放都会改注册的效应档，属口径篡改"
            )
    # 分位与目标功效同样属于冻结口径：偏离即拒绝（z 偏差会让区间失真、target 偏差会改 MDE 定义）
    if not math.isclose(float(z), WILSON_Z, rel_tol=0.0, abs_tol=1e-9):
        raise PowerError(f"Wilson 分位 z 必须等于冻结值 {WILSON_Z}，实际 {z}")
    if not math.isclose(float(target), TARGET_POWER, rel_tol=0.0, abs_tol=1e-12):
        raise PowerError(f"目标功效必须等于冻结值 {TARGET_POWER}，实际 {target}")
    # 无效率门同样是冻结口径：可被放大成"永不过门"从而静默废掉 10% 门
    if not math.isclose(float(invalid_rate_per_look_max), 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise PowerError(
            f"无效率门必须等于冻结值 0.10，实际 {invalid_rate_per_look_max}"
            "（禁止放大或缩小该门）"
        )
    return frozen, ordered, list(scenes)


def _score_effect_grid(
    *,
    plan: LookPlan,
    registry: SceneRegistry,
    scenes: Sequence[str],
    ordered: Sequence[float],
    frozen: Sequence[float],
    seed: int,
    count: int,
    statistic: str,
    delta: float,
    prefix: str,
    chunk: Optional[int],
    z: float,
    scale_by_scene: Optional[Mapping[str, float]],
    invalid_rate_per_look_max: float,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], List[str]]:
    """逐效应档 × 逐场景打分（返回逐格明细、检出率矩阵与退化单元列表）。

    Returns:
        ``(by_level, rates, degraded_units)``。
    """
    by_level: Dict[str, Any] = {}
    rates: Dict[str, Dict[str, float]] = {}
    degraded_units: List[str] = []
    for level in ordered:
        key = str(float(level))
        by_level[key] = {}
        rates[key] = {}
        for scene in scenes:
            if scale_by_scene is None:
                factor = 1.0
            elif scene not in scale_by_scene:
                raise PowerError(f"scale_by_scene 缺少场景 {scene!r}（必须逐场景给恒等值）")
            else:
                factor = float(scale_by_scene[scene])
            scores, realization_id = generate_scores(
                registry,
                scene,
                seed=seed,
                stage="effect",
                null="",
                mu=-level * float(delta) * factor,
                count=int(count),
                plan=plan,
                statistic=statistic,
                delta=delta,
                prefix=prefix,
                level=float(level),
                chunk=chunk,
            )
            flags = veto_flags(scores.values, frozen)
            successes = int(flags.sum())
            total = int(flags.size)
            low, high = wilson_interval(successes, total, z=z)
            invalid_by_look = [scores.invalid_rate(index) for index in range(plan.k)]
            if max(invalid_by_look) > float(invalid_rate_per_look_max):
                # 与 holdout 同一无效率门：退化单元不得静默低估检出率
                degraded_units.append(f"{scene}|{level}")
            by_level[key][scene] = {
                "scene": scene,
                "level_delta": float(level),
                "daily_log_shift": -level * float(delta),
                "successes": successes,
                "total": total,
                "rate": successes / total,
                "wilson_low": low,
                "wilson_high": high,
                "realization_id": realization_id,
                "invalid_rate_by_look": invalid_by_look,
            }
            rates[key][scene] = successes / total
    return by_level, rates, degraded_units


def evaluate_power(
    *,
    plan: LookPlan,
    registry: SceneRegistry,
    seed: int,
    count: int,
    thresholds: Sequence[float],
    scale_by_scene: Optional[Mapping[str, float]] = None,
    grid: Sequence[float] = EFFECT_GRID_DELTA,
    applicable: Optional[bool] = None,
    statistic: str = STATISTIC_PSD,
    delta: float = DELTA_LOG_DAILY,
    prefix: str = "",
    target: float = TARGET_POWER,
    z: float = WILSON_Z,
    chunk: Optional[int] = None,
    invalid_rate_per_look_max: float = 0.10,
) -> Dict[str, Any]:
    """逐场景 × 逐效应档计算检出率与 Wilson 区间，并登记 MDE。

    Args:
        plan: look 映射。
        registry: **holdout 池**场景注册表（效应档与 A/A 共用同一池）。
        seed: 命名种子根。
        count: 每档每场景路径数（manifest 冻结值 2000）。
        thresholds: 已冻结 ``c_k``。
        scale_by_scene: 仅测试用（对检出率做场景缩放）；生产路径必须为 ``None``。
        grid: 效应档（δ 倍数）。
        applicable: A/A 是否过门（**必填**，无默认值：未做 A/A 验收不得登记 MDE）。
        statistic: 注册统计量。
        delta: 日度 log 基准。
        prefix: realization 前缀。
        target: 目标检出率。
        z: Wilson 分位。
        chunk: 分块路径数。
        invalid_rate_per_look_max: 与 holdout 同一无效率门（超过即不登记 MDE）。

    Returns:
        逐档逐场景读数 + ``mde`` 段（含 ``pooled_average_forbidden``）。

    Raises:
        PowerError: 注册表角色非 holdout、``applicable`` 未显式给出、效应档非正、阈值非法、
            ``scale_by_scene`` 被用于生产、分位/目标/无效率门偏离冻结值。
    """
    if registry.pool_role != "holdout":
        raise PowerError(
            f"功效评估必须使用 holdout 池（pool_role='holdout'），实际 {registry.pool_role!r}"
        )
    if applicable is None:
        raise PowerError(
            "必须显式传入 applicable（A/A 是否过门）：无默认值，禁止在未做 A/A 验收时登记 MDE"
        )
    frozen, ordered, scenes = _validate_power_inputs(
        plan=plan,
        scenes=registry.names(),
        thresholds=thresholds,
        grid=grid,
        scale_by_scene=scale_by_scene,
        target=target,
        z=z,
        invalid_rate_per_look_max=invalid_rate_per_look_max,
    )

    by_level, rates, degraded_units = _score_effect_grid(
        plan=plan,
        registry=registry,
        scenes=scenes,
        ordered=ordered,
        frozen=frozen,
        seed=seed,
        count=count,
        statistic=statistic,
        delta=delta,
        prefix=prefix,
        chunk=chunk,
        z=z,
        scale_by_scene=scale_by_scene,
        invalid_rate_per_look_max=invalid_rate_per_look_max,
    )
    mde = register_mde(
        bool(applicable) and not degraded_units, rates, ordered, scenes, target=target
    )
    if degraded_units:
        mde["degraded_units"] = list(degraded_units)
        mde["degraded_reason"] = (
            f"invalid_rate > {float(invalid_rate_per_look_max)}（不得登记 MDE）"
        )
    return {
        "schema": POWER_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "statistic": statistic,
        "delta": float(delta),
        "plan": plan.as_dict(),
        "grid_delta": [float(level) for level in ordered],
        "target_power": float(target),
        "detection_event": "any_k(valid and T_k < -c_k)",
        "thresholds": frozen,
        "counts": {
            "effect_per_scene_per_level": int(count),
            "seed": int(seed),
            "realization_prefix": prefix,
            # 保真：None = 真正单批，不得伪记为默认分块值
            "chunk": None if chunk is None else int(chunk),
            "effective_chunk": DEFAULT_CHUNK if chunk is None else int(chunk),
        },
        "invalid_rate_per_look_max": float(invalid_rate_per_look_max),
        "wilson_confidence": 0.95,
        "wilson_z": float(z),
        "by_level_scene": by_level,
        "mde": {
            **mde,
            "definition": "smallest effect level with point-estimate detection rate >= target",
            "mde_requires_aa_pass": True,
            "pooled_average_forbidden": True,
            "per_scene_only": True,
        },
        "limitations": [
            "检出率是已测历史场景的条件读数，不是未来保证",
            "不得宣称 5δ 通用灵敏（MA20/AR1 在 5δ 未达 80% 属已知事实）",
            "同一裁判的 A/A 与功效使用同一组阈值与同一检出事件",
        ],
    }
