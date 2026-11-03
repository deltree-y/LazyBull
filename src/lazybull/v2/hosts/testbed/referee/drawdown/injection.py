# -*- coding: utf-8 -*-
"""P5b T5 崩盘窗回撤注入与 λ/ζ 求解器（设计 v4 §5.4.3 唯一锚定 + §5.5「回撤求解」行）。

注入公式（冻结）
================
窗口内**负 log 收益**才改造（正收益日不变、窗口外不改）::

    DD-I（改善）：x'_t = (1 - λ) x_t,  λ ∈ [0, 1]
    DD-W（恶化）：x'_t = (1 + ζ) x_t,  ζ ∈ [0, 2]
    重建：r'_t = exp(x'_t) - 1（读数为 log 尺度最小回撤，故不再显式 exp 路径）

反事实与目标
============
每个实现先构造 ``x = B0 + n``（配对噪声实现），再生成 ``arm = x + inj``；求解目标是
**纯注入效应** ``MDD(arm) - MDD(x)``，**不得**把含噪的 ``MDD(arm) - MDD(B0)`` 当目标。
唯一锚 ``T_anchor = H*J_max = 500``：该 horizon 上目标为 ``DD-I:+δ``、``DD-W:-δ``（pp）。
同一路径所有 look **共用一次注入**（早期 look 只报告实际实现效应），λ 与 ζ **分别独立求解**。

求解（冻结规则）
================
1. 先做 :data:`SCAN_POINTS`=101 点全域扫描；
2. 再在最接近目标的相邻区间用二分细化，**最多** :data:`MAX_REFINE_STEPS`=80 次；
3. 可行解 = ``|纯注入效应 - 目标| <= 0.05pp``；
4. 多解取**最小干预量**：DD-I 取最小 λ、DD-W 取最小 ζ（不得称最大 λ 为最少干预）；
5. 目标不可达 / 容差内无解 / 注入后路径非有限 ⇒ ``not_constructible``，
   **不得截断档位、不得把实际效应改报为目标值**。

数值实现
========
把注入写成累计曲线上的仿射平移（无需逐路径 Python 循环、也无需重新累计）::

    cum(θ) = cum(x) + (factor(θ) - 1) * cum(x · 1[support])

其中 ``support = (x < 0) ∧ window``。因为 ``x'_t = factor·x_t`` 在支撑上等价于
累计曲线加一个与 θ 线性相关的项，故 101 点扫描与二细分都是**整块向量化**的。
支撑由 ``x`` 自身的符号决定（不随 θ 变化），因此效应关于 θ 单调：DD-I 单调变浅、
DD-W 单调变深——这是二分有效性的前提（并由用例覆盖）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    ANCHOR_HORIZON,
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
    cumulative_log_path,
    log_to_percent,
    max_drawdown_log,
)

__all__ = [
    "InjectionSolution",
    "InjectionError",
    "LAMBDA_RANGE",
    "MAX_REFINE_STEPS",
    "SCAN_POINTS",
    "STATUS_CONSTRUCTIBLE",
    "STATUS_NO_SOLUTION",
    "STATUS_UNREACHABLE",
    "TOLERANCE_PP",
    "ZETA_RANGE",
    "apply_injection",
    "injection_support",
    "parameter_range",
    "pure_injection_effect",
    "solve_injection",
    "solver_section",
]

#: DD-I 的 λ 取值域（最小干预 ⇒ 取可行解中的最小 λ）
LAMBDA_RANGE: Tuple[float, float] = (0.0, 1.0)
#: DD-W 的 ζ 取值域（最小干预 ⇒ 取可行解中的最小 ζ）
ZETA_RANGE: Tuple[float, float] = (0.0, 2.0)
#: 全域扫描点数（冻结）
SCAN_POINTS = 101
#: 细化最大迭代次数（冻结上限）
MAX_REFINE_STEPS = 80
#: 目标容差（pp）
TOLERANCE_PP = 0.05
#: 二分收敛阈值（只影响数值精度，不放宽容差）
_SOLVER_EPS = 1e-12

#: 状态码
STATUS_CONSTRUCTIBLE = "constructible"
STATUS_UNREACHABLE = "unreachable_in_domain"
STATUS_NO_SOLUTION = "no_solution_within_tolerance"
#: ``not_constructible`` 的唯一机器可读标记（主功效分母仍计入这些实现）
NOT_CONSTRUCTIBLE = "not_constructible"


class InjectionError(TestbedError):
    """注入/求解输入非法（方向未知、目标符号不符、形状不符）。"""

    __test__ = False


@dataclass(frozen=True)
class InjectionSolution:
    """一次求解的完整读数（逐实现参数 + 纯注入效应 + 可行性）。"""

    direction: str
    target_pp: float
    anchor: int
    parameter: np.ndarray
    achieved_pp: np.ndarray
    constructible: np.ndarray
    status: Tuple[str, ...]
    scan_points: int
    refine_steps: int
    domain: Tuple[float, float]

    @property
    def total(self) -> int:
        return int(self.parameter.size)

    @property
    def constructible_count(self) -> int:
        return int(self.constructible.sum())

    def constructibility_rate(self) -> float:
        """可构造率 ``C/M``（分母 = 全部实现，禁止剔除不可构造）。"""
        return float(self.constructible_count) / float(self.total) if self.total else 0.0

    def parameter_stats(self) -> Dict[str, Any]:
        """λ/ζ 分布（只统计可行解；不可构造的实现不计入分布，但计入分母）。"""
        values = np.asarray(self.parameter, dtype=float)[np.asarray(self.constructible, bool)]
        if values.size == 0:
            return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
        return {
            "count": int(values.size),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }

    def effect_stats(self) -> Dict[str, Any]:
        """锚上纯注入效应分布（只统计可行解；与目标逐值可比）。"""
        values = np.asarray(self.achieved_pp, dtype=float)[np.asarray(self.constructible, bool)]
        if values.size == 0:
            return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
        return {
            "count": int(values.size),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }

    def status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.status:
            counts[item] = counts.get(item, 0) + 1
        return dict(sorted(counts.items()))


def parameter_range(direction: str) -> Tuple[float, float]:
    """方向 → 参数取值域（DD-I: λ∈[0,1]；DD-W: ζ∈[0,2]）。"""
    if direction == DIRECTION_IMPROVE:
        return LAMBDA_RANGE
    if direction == DIRECTION_WORSEN:
        return ZETA_RANGE
    raise InjectionError(
        f"未知回撤方向 {direction!r}（允许 {DIRECTION_IMPROVE!r}/{DIRECTION_WORSEN!r}）"
    )


def _sign(direction: str) -> float:
    """效应关于参数的单调方向（DD-I 变浅 +1；DD-W 变深 -1）。"""
    return 1.0 if direction == DIRECTION_IMPROVE else -1.0


def _validate_target(direction: str, target_pp: float) -> float:
    target = float(target_pp)
    if not np.isfinite(target):
        raise InjectionError(f"目标效应必须有限，实际 {target_pp!r}")
    if direction == DIRECTION_IMPROVE and target <= 0.0:
        raise InjectionError(f"DD-I 目标必须为正（改善），实际 {target}")
    if direction == DIRECTION_WORSEN and target >= 0.0:
        raise InjectionError(f"DD-W 目标必须为负（恶化），实际 {target}")
    return target


def injection_support(paths: np.ndarray, window_mask: np.ndarray) -> np.ndarray:
    """注入支撑 = 窗口内**负** log 收益（正收益日不变）。

    Args:
        paths: ``(M, N)`` 逐日 log 收益（``x = B0 + n``）。
        window_mask: ``(N,)`` 布尔窗口掩码（冻结集合的并集）。

    Raises:
        InjectionError: 维度/长度不符。
    """
    values = np.asarray(paths, dtype=float)
    mask = np.asarray(window_mask, dtype=bool)
    if values.ndim != 2:
        raise InjectionError(f"paths 必须是二维 (M, N)，实际 shape={values.shape}")
    if mask.shape != (values.shape[1],):
        raise InjectionError(f"window_mask 长度必须是 {values.shape[1]}，实际 shape={mask.shape}")
    if not np.isfinite(values).all():
        raise InjectionError("paths 含非有限值（注入后路径必须有限）")
    return (values < 0.0) & mask[None, :]


def apply_injection(
    paths: np.ndarray,
    support: np.ndarray,
    *,
    direction: str,
    parameter: np.ndarray,
) -> np.ndarray:
    """按冻结公式生成 ``arm = x + inj``（唯一注入实现；正收益/窗口外保持原值）。

    Args:
        paths: ``(M, N)`` 逐日 log 收益。
        support: :func:`injection_support` 的输出（``(M, N)`` 布尔）。
        direction: DD-I / DD-W。
        parameter: ``(M,)`` λ 或 ζ（每股一条；必须有限且落在取值域内）。

    Raises:
        InjectionError: 形状不符/参数越界/非有限（禁止静默裁剪或截断档位）。
    """
    values = np.asarray(paths, dtype=float)
    mask = np.asarray(support, dtype=bool)
    theta = np.asarray(parameter, dtype=float).reshape(-1)
    if values.ndim != 2 or mask.shape != values.shape:
        raise InjectionError(f"paths/support 形状不符：{values.shape} vs {mask.shape}")
    if theta.size != values.shape[0]:
        raise InjectionError(f"参数个数({theta.size}) 必须等于路径数({values.shape[0]})")
    low, high = parameter_range(direction)
    if not np.isfinite(theta).all() or (theta < low - 1e-12).any() or (theta > high + 1e-12).any():
        raise InjectionError(
            f"参数 {theta!r} 越界或非有限（{direction} 取值域 {parameter_range(direction)}）"
        )
    factor = (1.0 - theta) if direction == DIRECTION_IMPROVE else (1.0 + theta)
    return np.where(mask, values * factor[:, None], values)


class _Prepared:
    """累计曲线预算（扫描/二分的共用底座；不可变使用）。"""

    __slots__ = ("values", "support", "cum_base", "cum_support", "base_pct", "anchor")

    def __init__(self, values: np.ndarray, support: np.ndarray, anchor: int) -> None:
        self.values = values
        self.support = support
        self.anchor = int(anchor)
        self.cum_base = cumulative_log_path(values)[:, : self.anchor + 1]
        self.cum_support = cumulative_log_path(np.where(support, values, 0.0))[:, : self.anchor + 1]
        self.base_pct = log_to_percent(max_drawdown_log(self.cum_base))

    def effect(
        self, direction: str, theta: np.ndarray, rows: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """纯注入效应（pp）：``MDD(x + inj) - MDD(x)``（``rows`` 限定参与计算的实现）。"""
        theta = np.asarray(theta, dtype=float)
        if rows is None:
            base = self.cum_base
            support = self.cum_support
            anchor = self.base_pct
        else:
            index = np.asarray(rows, dtype=int)
            base = self.cum_base[index]
            support = self.cum_support[index]
            anchor = self.base_pct[index]
        scale = -theta if direction == DIRECTION_IMPROVE else theta
        cum = base + scale[:, None] * support
        return log_to_percent(max_drawdown_log(cum)) - anchor


def _prepare(paths: np.ndarray, window_mask: np.ndarray, anchor: int) -> _Prepared:
    values = np.asarray(paths, dtype=float)
    if values.ndim != 2:
        raise InjectionError(f"paths 必须是二维 (M, N)，实际 shape={values.shape}")
    limit = int(anchor)
    if limit <= 0 or limit > values.shape[1]:
        raise InjectionError(
            f"anchor={limit} 必须落在 1..{values.shape[1]}（唯一锚不得换成历史档位）"
        )
    return _Prepared(values, injection_support(values, window_mask), limit)


def pure_injection_effect(
    paths: np.ndarray,
    window_mask: np.ndarray,
    *,
    direction: str,
    parameter: np.ndarray,
    anchor: int = ANCHOR_HORIZON,
) -> np.ndarray:
    """逐实现的纯注入效应 ``MDD(arm) - MDD(x)``（pp，锚上）。"""
    prepared = _prepare(paths, window_mask, anchor)
    return prepared.effect(direction, parameter)


def _bracket(
    grid: np.ndarray, effects: np.ndarray, target: float, tolerance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """由 101 点扫描结果给出逐实现的细化区间（最接近目标的相邻区间）。"""
    absolute = np.abs(effects - target)
    feasible = absolute <= tolerance
    has = feasible.any(axis=0)
    first = np.argmax(feasible, axis=0)
    closest = np.argmin(absolute, axis=0)
    start = np.where(has, np.maximum(first - 1, 0), np.maximum(closest - 1, 0))
    stop = np.where(has, first, np.minimum(closest + 1, grid.size - 1))
    return grid[start], grid[stop]


def _bisect(
    prepared: _Prepared,
    direction: str,
    target: float,
    tolerance: float,
    low: np.ndarray,
    high: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """向量化二分求**最小可行参数**（左沿）；返回 (参数, 实际迭代次数)。"""
    sign = _sign(direction)
    threshold = -tolerance
    steps = 0
    for step in range(MAX_REFINE_STEPS):
        active = (high - low) > _SOLVER_EPS
        if not active.any():
            break
        steps = step + 1
        index = np.flatnonzero(active)
        mid = 0.5 * (low[index] + high[index])
        score = sign * (prepared.effect(direction, mid, rows=index) - target)
        reach = score >= threshold
        high[index[reach]] = mid[reach]
        low[index[~reach]] = mid[~reach]
    return high, steps


def solve_injection(
    paths: np.ndarray,
    window_mask: np.ndarray,
    *,
    direction: str,
    target_pp: float,
    anchor: int = ANCHOR_HORIZON,
    tolerance: float = TOLERANCE_PP,
) -> InjectionSolution:
    """逐实现求解 λ/ζ，使锚上**纯注入效应**达到目标（容差 0.05pp）。

    Args:
        paths: ``(M, N)`` 噪声实现 ``x = B0 + n``（配对噪声，非 B0 本身）。
        window_mask: ``(N,)`` 冻结崩盘窗并集。
        direction: DD-I（改善，目标为正）/ DD-W（恶化，目标为负）。
        target_pp: 目标纯注入效应（pp）。
        anchor: 唯一锚（默认 500，禁止改成历史档位）。
        tolerance: 可行容差（默认 0.05pp，不得放宽）。

    Returns:
        :class:`InjectionSolution`（``constructible`` 掩码 + 逐实现参数与效应）。

    Raises:
        InjectionError: 方向/目标/形状非法，或容差被放宽。
    """
    goal = _validate_target(direction, target_pp)
    if not np.isclose(float(tolerance), TOLERANCE_PP, rtol=0.0, atol=1e-12):
        raise InjectionError(
            f"容差必须等于冻结值 {TOLERANCE_PP}pp（不得放宽或收紧），实际 {tolerance}"
        )
    prepared = _prepare(paths, window_mask, anchor)
    low_bound, high_bound = parameter_range(direction)
    grid = np.linspace(low_bound, high_bound, SCAN_POINTS)
    effects = np.empty((SCAN_POINTS, prepared.values.shape[0]), dtype=float)
    for index, value in enumerate(grid):
        effects[index] = prepared.effect(direction, np.full(prepared.values.shape[0], value))
    low, high = _bracket(grid, effects, goal, float(tolerance))
    parameter, steps = _bisect(prepared, direction, goal, float(tolerance), low, high)
    achieved = prepared.effect(direction, parameter)
    sign = _sign(direction)
    score = sign * (achieved - goal)
    constructible = np.abs(score) <= float(tolerance)
    end_score = sign * (effects[-1] - goal)
    status = []
    for row in range(parameter.size):
        if constructible[row]:
            status.append(STATUS_CONSTRUCTIBLE)
        elif end_score[row] < -float(tolerance):
            status.append(STATUS_UNREACHABLE)
        else:
            status.append(STATUS_NO_SOLUTION)
    return InjectionSolution(
        direction=direction,
        target_pp=goal,
        anchor=int(anchor),
        parameter=parameter,
        achieved_pp=achieved,
        constructible=constructible,
        status=tuple(status),
        scan_points=int(SCAN_POINTS),
        refine_steps=int(steps),
        domain=(float(low_bound), float(high_bound)),
    )


def solver_section() -> Dict[str, Any]:
    """manifest 的求解器冻结段（§5.5「回撤求解」行逐字落地）。"""
    return {
        "scan_points": int(SCAN_POINTS),
        "max_refine_steps": int(MAX_REFINE_STEPS),
        "tolerance_pp": float(TOLERANCE_PP),
        "feasible_rule": "|纯注入效应 - 目标| <= tolerance_pp",
        "multi_solution_rule": "DD-I 取最小 λ、DD-W 取最小 ζ（均为最小干预；不得称最大值为最少干预）",
        "search_rule": "先 101 点全域扫描，再在最接近目标的相邻区间二分细化（最多 80 次）",
        "lambda_range": [float(LAMBDA_RANGE[0]), float(LAMBDA_RANGE[1])],
        "zeta_range": [float(ZETA_RANGE[0]), float(ZETA_RANGE[1])],
        "independent_parameters": True,
        "not_constructible_semantics": (
            "目标不可达 / 容差内无解 / 注入后路径非有限 ⇒ not_constructible；"
            "不得截断档位、不得把实际效应改报为目标值；主功效分母仍计入该实现（按未检出）"
        ),
        "truncation_forbidden": True,
        "report_target_as_achieved_forbidden": True,
    }
