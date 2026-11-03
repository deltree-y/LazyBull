# -*- coding: utf-8 -*-
"""P5b T5 回撤裁判功效与 MDE（设计 v4 §5.3 表 + §5.5「回撤效应网格」行 + §5.4.4）。

口径
====
- 改善与恶化**分别**使用同一纯注入档网格 ``{1,2,3,5,8,12}pp``（展示换算：λ/ζ 与实现效应
  一并列出；DD-I 与 DD-W **分开命名**）；
- 检出事件与 A/A **同一事件**（``any_k(D_k < c^W_k)`` / ``any_k(I_k = D_k > c^I_k)``）；
- ``MDE_drawdown_worsen`` / ``MDE_drawdown_improve`` = **A/A 过门后**点估计检出率达 80% 的
  最小档；未达则如实登记 ``tested_to_12pp_not_80pct``，并标注是否命中网格上界；
- **主功效分母 = 全部实现 M**（可构造 C 单独披露）：不可构造的实现按**未检出**计入，
  不得剔除；``C/M`` 逐格给出；
- 逐格同时报告：**纯注入效应**（``MDD(arm)-MDD(x)``，锚上）、**含噪观察效应**
  （``MDD(arm)-MDD(B0)``，锚上）、**各 look 实际效应**均值。禁止把实际效应改报为目标值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
    DIRECTIONS,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.sampling import IMPLEMENTATION_ID
from src.lazybull.v2.hosts.testbed.referee.drawdown.thresholds import (
    DrawdownThresholds,
    improve_event,
    worsen_event,
)
from src.lazybull.v2.hosts.testbed.referee.holdout import WILSON_Z, wilson_interval
from src.lazybull.v2.hosts.testbed.referee.statistics import LookPlan

__all__ = [
    "EFFECT_GRID_PP",
    "MDE_IMPROVE_KEY",
    "MDE_NOT_REACHED",
    "MDE_NOT_REGISTERED",
    "MDE_WORSEN_KEY",
    "POWER_SCHEMA",
    "TARGET_POWER",
    "DrawdownPowerError",
    "EffectCell",
    "evaluate_drawdown_power",
    "register_mde",
]

#: 纯注入效应档（pp；改善与恶化共用但分别命名）
EFFECT_GRID_PP: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
#: 目标检出率（点估计口径）
TARGET_POWER = 0.80
#: 未达 80% 的如实表述
MDE_NOT_REACHED = "tested_to_12pp_not_80pct"
#: A/A 未过门时的登记状态
MDE_NOT_REGISTERED = "not_registered_aa_failed"
#: 两个 MDE 的字段名（**分别命名，不得互相抵扣**）
MDE_WORSEN_KEY = "MDE_drawdown_worsen"
MDE_IMPROVE_KEY = "MDE_drawdown_improve"
#: 产物 schema
POWER_SCHEMA = "p5b_testbed_drawdown_power/v1"


class DrawdownPowerError(TestbedError):
    """功效计算输入非法（档位越界/格缺失/目标被改/分母被改）。"""

    __test__ = False


@dataclass(frozen=True)
class EffectCell:
    """一个「方向 × 效应档 × 场景」单元的实现级读数。"""

    direction: str
    level_pp: float
    scene: str
    d_matrix: np.ndarray
    constructible: np.ndarray
    parameter: np.ndarray
    pure_pp: np.ndarray
    observed_pp: np.ndarray
    realization_id: str

    def __post_init__(self) -> None:
        values = np.asarray(self.d_matrix, dtype=float)
        if values.ndim != 2:
            raise DrawdownPowerError(f"D 矩阵必须是二维，实际 shape={values.shape}")
        rows = values.shape[0]
        for name in ("constructible", "parameter", "pure_pp", "observed_pp"):
            if np.asarray(getattr(self, name)).size != rows:
                raise DrawdownPowerError(f"{name} 行数必须等于实现数 {rows}（分母不得被改写）")

    @property
    def total(self) -> int:
        return int(np.asarray(self.d_matrix).shape[0])

    @property
    def constructible_count(self) -> int:
        return int(np.asarray(self.constructible, dtype=bool).sum())


def _stats(values: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)]
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def _cell_flags(cell: EffectCell, thresholds: DrawdownThresholds) -> np.ndarray:
    if cell.direction == DIRECTION_WORSEN:
        return worsen_event(cell.d_matrix, thresholds.worsen)
    if cell.direction == DIRECTION_IMPROVE:
        return improve_event(cell.d_matrix, thresholds.improve)
    raise DrawdownPowerError(f"未知方向 {cell.direction!r}（允许 {DIRECTIONS}）")


def register_mde(
    applicable: bool,
    readings: Mapping[str, Mapping[float, Dict[str, Any]]],
    *,
    scenes: Sequence[str],
    directions: Sequence[str] = DIRECTIONS,
    grid: Sequence[float] = EFFECT_GRID_PP,
    target: float = TARGET_POWER,
) -> Dict[str, Any]:
    """登记两个 MDE（仅当 A/A 过门）：逐方向逐场景给出最小达标档或未达标记。

    Args:
        applicable: 该方向是否 A/A 过门（``{direction: bool}`` 形态由调用方预先折叠）。
        readings: ``{direction: {level_pp: {"scene": rate}}}`` 检出率矩阵。
        scenes: 场景名。
        directions: 两个方向。
        grid: 效应档（pp，升序）。
        target: 目标检出率。

    Returns:
        ``{registered, mde_drawdown_worsen: {...}, mde_drawdown_improve: {...}, ...}``。
    """
    ordered = sorted(float(level) for level in grid)
    upper = float(ordered[-1]) if ordered else None
    result: Dict[str, Any] = {
        "registered": False,
        "aa_gate_passed": bool(applicable),
        "definition": (
            "A/A 过门后，点估计检出率达 80% 的最小纯注入档（pp）；"
            "档名为**目标值**，锚上实际纯注入效应为 目标∓tolerance（最小干预左沿）"
        ),
        "target_power": float(target),
        "grid_pp": [float(level) for level in ordered],
        "grid_upper_pp": upper,
        "nominal_vs_actual": "档名=目标；实际纯注入=|目标|-0.05pp（tolerance 下沿，见 solver）",
        "pooled_average_forbidden": True,
        "per_scene_only": True,
        "mde_requires_aa_pass": True,
    }
    any_registered = False
    for direction in directions:
        by_scene: Dict[str, Optional[float]] = {}
        status: Dict[str, str] = {}
        at_upper: Dict[str, bool] = {}
        curve: Dict[str, List[float]] = {}
        monotone: Dict[str, bool] = {}
        contradicted: Dict[str, Any] = {}
        for scene in scenes:
            levels_rates = {
                level: float(dict(readings.get(direction, {})).get(level, {}).get(scene, 0.0))
                for level in ordered
            }
            curve[scene] = [levels_rates[level] for level in ordered]
            monotone[scene] = all(
                right >= left - 1e-12 for left, right in zip(curve[scene], curve[scene][1:])
            )
            if not applicable:
                by_scene[scene] = None
                status[scene] = MDE_NOT_REGISTERED
                at_upper[scene] = False
                continue
            hits = [level for level in ordered if levels_rates[level] >= float(target)]
            if not monotone[scene]:
                dips = [
                    {
                        "from_level_pp": left,
                        "from_rate": levels_rates[left],
                        "to_level_pp": right,
                        "to_rate": levels_rates[right],
                        "delta": levels_rates[right] - levels_rates[left],
                    }
                    for left, right in zip(ordered, ordered[1:])
                    if levels_rates[right] < levels_rates[left] - 1e-12
                ]
                if dips:
                    contradicted[scene] = dips
            if hits:
                chosen = float(min(hits))
                by_scene[scene] = chosen
                status[scene] = "registered"
                at_upper[scene] = bool(upper is not None and chosen >= upper)
                any_registered = True
            else:
                by_scene[scene] = None
                status[scene] = MDE_NOT_REACHED
                at_upper[scene] = False
        key = MDE_WORSEN_KEY if direction == DIRECTION_WORSEN else MDE_IMPROVE_KEY
        result[key] = {
            "direction": direction,
            "by_scene": by_scene,
            "status_by_scene": status,
            "at_grid_upper_bound": at_upper,
            "registered": any(value is not None for value in by_scene.values()),
            "scenes_registered": sorted(
                scene for scene, value in by_scene.items() if value is not None
            ),
            "scenes_total": sorted(by_scene),
            "gate_scope": (
                "方向级 registered = 至少一个场景登记 MDE（逐场景明细见 by_scene）；"
                "H5 的方向级判定口径由用户在合并验收时选择"
            ),
            "aa_gate_passed": bool(applicable),
            "power_curve_by_scene": curve,
            "power_curve_monotone": monotone,
            "power_curve_dips": contradicted,
            "contradiction_note": (
                "主分母含不可构造实现 ⇒ 更大档位的检出率可能反而更低；"
                "'MDE=X' 只表示该档首次达标，不代表 ≥X 的档位都达标"
            ),
        }
    result["registered"] = bool(applicable) and any_registered
    return result


def _validate_power_inputs(
    grid: Sequence[float], target: float, z: float, applicable: Mapping[str, bool]
) -> List[float]:
    """冻结口径校验（网格/目标功效/Wilson 分位/逐方向门状态）。"""
    ordered = [float(level) for level in grid]
    if ordered != [float(level) for level in EFFECT_GRID_PP]:
        raise DrawdownPowerError(f"效应档必须等于冻结网格 {list(EFFECT_GRID_PP)}，实际 {ordered}")
    if not math.isclose(float(target), TARGET_POWER, rel_tol=0.0, abs_tol=1e-12):
        raise DrawdownPowerError(f"目标功效必须等于冻结值 {TARGET_POWER}，实际 {target}")
    if not math.isclose(float(z), WILSON_Z, rel_tol=0.0, abs_tol=1e-9):
        raise DrawdownPowerError(f"Wilson 分位 z 必须等于冻结值 {WILSON_Z}，实际 {z}")
    for direction in DIRECTIONS:
        if direction not in applicable:
            raise DrawdownPowerError(f"必须显式给出方向 {direction!r} 的 A/A 是否过门（无默认值）")
    return ordered


def _score_cell(
    cell: EffectCell, thresholds: DrawdownThresholds, plan: LookPlan, z: float
) -> Dict[str, Any]:
    """单个效应格的读数（检出率 + Wilson + C/M + 纯注入/含噪观察/逐 look 效应）。"""
    flags = _cell_flags(cell, thresholds)
    successes = int(flags.sum())
    total = int(flags.size)
    low, high = wilson_interval(successes, total, z=float(z))
    constructible = np.asarray(cell.constructible, dtype=bool)
    values = np.asarray(cell.d_matrix, dtype=float)
    limits = thresholds.worsen if cell.direction == DIRECTION_WORSEN else thresholds.improve
    per_look_hits = []
    for index in range(int(plan.k)):
        column = values[:, index]
        hit = (
            column < limits[index] if cell.direction == DIRECTION_WORSEN else column > limits[index]
        )
        per_look_hits.append(int(np.sum(np.isfinite(column) & hit)))
    return {
        "direction": cell.direction,
        "level_pp": float(cell.level_pp),
        "scene": cell.scene,
        "realization_id": cell.realization_id,
        "successes": successes,
        "total": total,
        "rate": successes / total,
        "wilson_low": low,
        "wilson_high": high,
        "constructible_count": int(constructible.sum()),
        "constructibility_rate": float(constructible.sum()) / total if total else 0.0,
        "not_constructible_count": int(total - constructible.sum()),
        "parameter_stats": _stats(cell.parameter, constructible),
        "pure_effect_pp": _stats(cell.pure_pp, constructible),
        "observed_effect_pp": _stats(cell.observed_pp, constructible),
        "per_look_effect_pp": [
            _stats(values[:, index], constructible) for index in range(int(plan.k))
        ],
        "per_look_triggers": per_look_hits,
        "constructibility_floor": 0.80,
        "rate_not_comparable": bool((float(constructible.sum()) / total if total else 0.0) < 0.80),
        "rate_not_comparable_note": (
            "该档可构造率 C/M < 80%：检出率被不可构造实现拉低，只作披露、不得据此外推更高档位"
        ),
        "event": (
            "any_k(D_k < c^W_k)"
            if cell.direction == DIRECTION_WORSEN
            else "any_k(I_k = D_k > c^I_k)"
        ),
    }


def _score_cells(
    cells: Sequence[EffectCell], thresholds: DrawdownThresholds, plan: LookPlan, z: float
) -> Tuple[Dict[str, Any], Dict[str, Dict[float, Dict[str, float]]], List[str]]:
    """逐格打分并收集场景清单（缺格检查由调用方完成）。"""
    by_cell: Dict[str, Dict[str, Dict[str, Any]]] = {}
    rates: Dict[str, Dict[float, Dict[str, float]]] = {}
    scenes: List[str] = []
    for cell in cells:
        if cell.scene not in scenes:
            scenes.append(cell.scene)
        entry = _score_cell(cell, thresholds, plan, z)
        by_cell.setdefault(cell.direction, {}).setdefault(str(float(cell.level_pp)), {})[
            cell.scene
        ] = entry
        rates.setdefault(cell.direction, {}).setdefault(float(cell.level_pp), {})[cell.scene] = (
            entry["rate"]
        )
    return by_cell, rates, scenes


def _mde_section(
    by_cell: Mapping[str, Any],
    rates: Mapping[str, Any],
    scenes: Sequence[str],
    ordered: Sequence[float],
    applicable_by_direction: Mapping[str, bool],
    target: float,
) -> Dict[str, Any]:
    """两个 MDE（逐方向判定）+ 可构造率矩阵 + 不可构造档清单。"""
    mde = register_mde(
        all(applicable_by_direction.values()),
        {direction: rates[direction] for direction in DIRECTIONS if direction in rates},
        scenes=scenes,
        directions=DIRECTIONS,
        grid=ordered,
        target=target,
    )
    # 逐方向分别判定：某个方向 A/A 未过门不得把另一方向的 MDE 一起作废（不得互相抵扣）
    for direction in DIRECTIONS:
        key = MDE_WORSEN_KEY if direction == DIRECTION_WORSEN else MDE_IMPROVE_KEY
        mde[key] = dict(
            register_mde(
                bool(applicable_by_direction[direction]),
                {direction: rates[direction]},
                scenes=scenes,
                directions=(direction,),
                grid=ordered,
                target=target,
            )[key]
        )
    mde["constructibility_by_scene"] = {
        direction: {
            scene: {
                str(float(level)): float(
                    by_cell[direction][str(float(level))][scene]["constructibility_rate"]
                )
                for level in ordered
            }
            for scene in scenes
        }
        for direction in DIRECTIONS
        if direction in by_cell
    }
    mde["not_constructible_levels"] = {
        direction: {
            scene: [
                float(level)
                for level in ordered
                if by_cell[direction][str(float(level))][scene]["not_constructible_count"] > 0
            ]
            for scene in scenes
        }
        for direction in DIRECTIONS
        if direction in by_cell
    }
    mde["not_constructible_note"] = (
        "不可构造档按未检出计入主分母 M；C/M 逐格披露，禁止剔除或截断档位"
    )
    mde["applicable_by_direction"] = dict(applicable_by_direction)
    return mde


def evaluate_drawdown_power(
    cells: Sequence[EffectCell],
    thresholds: DrawdownThresholds,
    *,
    plan: LookPlan,
    applicable: Mapping[str, bool],
    scenes: Optional[Sequence[str]] = None,
    grid: Sequence[float] = EFFECT_GRID_PP,
    target: float = TARGET_POWER,
    z: float = WILSON_Z,
    chunk: Optional[int] = None,
    per_scene_realizations: Optional[int] = None,
) -> Dict[str, Any]:
    """逐方向 × 逐档 × 逐场景计算检出率与 Wilson 区间，并登记两个 MDE。

    Args:
        cells: 全部效应单元（方向 × 档 × 场景）。
        thresholds: 已冻结的两组临界值。
        plan: look 映射。
        applicable: ``{direction: A/A 是否过门}``（**必须显式给出**）。
        grid: 效应档（pp）。
        target: 目标检出率。
        z: Wilson 分位（冻结值）。

    Returns:
        逐格读数 + 两个 MDE 段（无跨场景聚合字段）。

    Raises:
        DrawdownPowerError: 档位越界/格缺失/目标或分位被改/方向未给门状态。
    """
    ordered = _validate_power_inputs(grid, target, z, applicable)
    by_cell, rates, cell_scenes = _score_cells(cells, thresholds, plan, z)
    if scenes is not None:
        expected = sorted(str(item) for item in scenes)
        if sorted(cell_scenes) != expected:
            raise DrawdownPowerError(
                f"效应单元场景集与冻结值不符（禁止删困难场景）: 冻结 {expected} "
                f"实际 {sorted(cell_scenes)}"
            )
    scenes = cell_scenes
    missing = [
        f"{direction}|{level}|{scene}"
        for direction in DIRECTIONS
        for level in ordered
        for scene in scenes
        if scene not in by_cell.get(direction, {}).get(str(float(level)), {})
    ]
    if missing:
        raise DrawdownPowerError(f"效应格缺失（不得删困难场景）: {missing}")
    applicable_by_direction = {
        direction: bool(applicable.get(direction)) for direction in DIRECTIONS
    }
    mde = _mde_section(by_cell, rates, scenes, ordered, applicable_by_direction, target)
    return {
        "schema": POWER_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "plan": plan.as_dict(),
        "grid_pp": ordered,
        "target_power": float(target),
        "wilson_confidence": 0.95,
        "wilson_z": float(z),
        "by_direction_level_scene": by_cell,
        "mde": mde,
        "counts": {
            "effect_per_scene_per_level": (
                None if per_scene_realizations is None else int(per_scene_realizations)
            ),
            "chunk": None if chunk is None else int(chunk),
        },
        "limitations": [
            "检出率是已测历史场景的条件读数，不是未来保证",
            "不可构造的实现按未检出计入主分母（M），可构造率 C/M 单独披露",
            "MDE 逐场景登记，禁止跨场景平均或加权",
        ],
    }
