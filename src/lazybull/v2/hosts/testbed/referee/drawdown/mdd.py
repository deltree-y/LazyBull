# -*- coding: utf-8 -*-
"""P5b T5 回撤读数：MDD 与 ``D_k`` 的唯一实现点（设计 v4 §5.4 头部 + §5.4.4）。

口径（写死在代码里，禁止下游复制公式）
====================================
- 路径 = 逐日 **log 收益** ``x_t = ln(1+r_t)``，累计曲线 ``cum[t] = Σ_{i<t} x_i``、
  ``cum[0] = 0``（起点必须计入：窗口第一天就有回撤时不能漏掉）；
- 峰到谷最大回撤 ``MDD = min_t (cum[t] - max_{s<=t} cum[s])``，恒 ``<= 0``（log 尺度）；
- 对外读数为**百分点（pp）**：``MDD_pct = (exp(MDD_log) - 1) * 100``；
- 统一读数 ``D_k = MDD_arm(T_k) - MDD_base(T_k)``（同时取 pp）。MDD 为负值 ⇒
  ``D_k > 0`` 表示候选回撤更浅（改善），``D_k < 0`` 表示更深（恶化）；
  **DD-I 不得再对 D_k 取负**（``I_k = D_k``，设计 §5.4.2）。

逐 look 独立 horizon
====================
每个 look 用自己的 ``T_k`` 在**同一路径的前 T_k 个收益**上重算 MDD（H=20 ⇒
``T_k ∈ {120,240,360,500}``），**禁止**把 1714 日全链档位套到 500 日锚上（§5.4.4）。
look 映射复用收益裁判的 §5.1 唯一实现 :func:`referee.statistics.make_look_plan`，
本模块只校验其与冻结值一致，不另写一份映射。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.statistics import LookPlan, make_look_plan

__all__ = [
    "ANCHOR_HORIZON",
    "DIRECTION_IMPROVE",
    "DIRECTION_WORSEN",
    "DIRECTIONS",
    "DrawdownReadingError",
    "HOLD_H",
    "LOOK_DAYS",
    "WINDOW_SAMPLES",
    "baseline_percent_by_look",
    "cumulative_log_path",
    "drawdown_look_plan",
    "drawdown_reading",
    "horizon_mdd_log",
    "horizon_mdd_percent",
    "log_to_percent",
    "max_drawdown_log",
    "percent_to_log",
    "reading_from_cum",
    "validate_horizons",
]

#: 持有期（与 §5.1 一致；T5 只用 H=20）
HOLD_H = 20
#: 冻结的逐 look 终点（H=20 ⇒ 120/240/360/500）
LOOK_DAYS: Tuple[int, ...] = (120, 240, 360, 500)
#: 唯一锚定 horizon（= H*J_max = 500，与 T0.5b 已批口径一致）
ANCHOR_HORIZON = 500
#: 历史评估窗口的收益样本数（B0 最后 500 个收益样本；全链 1714 仅作历史描述）
WINDOW_SAMPLES = 500
#: DD-W：回撤恶化否决（D_k < c^W_k 触发）
DIRECTION_WORSEN = "dd_worsen"
#: DD-I：回撤改善检出（I_k = D_k > c^I_k 触发）
DIRECTION_IMPROVE = "dd_improve"
#: 两个方向（阈值/A/A/holdout/MDE 分别命名，不得互相抵扣）
DIRECTIONS: Tuple[str, ...] = (DIRECTION_WORSEN, DIRECTION_IMPROVE)


class DrawdownReadingError(TestbedError):
    """回撤读数输入非法（形状/非有限/horizon 越界/映射不符冻结值）。"""

    __test__ = False


def cumulative_log_path(log_returns: np.ndarray) -> np.ndarray:
    """累计 log 收益曲线（首点 ``0`` 显式前置）。

    Args:
        log_returns: ``(N,)`` 或 ``(M, N)`` 逐日 log 收益。

    Returns:
        ``(N+1,)`` 或 ``(M, N+1)``，首列/首元素为 ``0``。

    Raises:
        DrawdownReadingError: 维度非法或含非有限值（回撤读数不成立）。
    """
    values = np.asarray(log_returns, dtype=float)
    if values.ndim not in (1, 2):
        raise DrawdownReadingError(
            f"log 收益必须是 1D/2D，实际 shape={values.shape}（禁止按列/按格错位）"
        )
    if not np.isfinite(values).all():
        raise DrawdownReadingError("log 收益含非有限值（回撤读数不成立，不得静默跳过）")
    if values.ndim == 1:
        return np.concatenate((np.zeros(1, dtype=float), np.cumsum(values, dtype=float)))
    head = np.zeros((values.shape[0], 1), dtype=float)
    return np.concatenate((head, np.cumsum(values, axis=1, dtype=float)), axis=1)


def max_drawdown_log(cumulative: np.ndarray) -> np.ndarray:
    """累计 log 曲线的峰到谷最大回撤（``min_t (cum[t] - max_{s<=t} cum[s])``，<= 0）。

    Args:
        cumulative: :func:`cumulative_log_path` 的输出（含首点 0）。

    Returns:
        ``()``／``(M,)`` 的 log 尺度最大回撤（<= 0）。
    """
    arr = np.asarray(cumulative, dtype=float)
    if arr.ndim not in (1, 2):
        raise DrawdownReadingError(f"累计曲线必须是 1D/2D，实际 shape={arr.shape}")
    peak = np.maximum.accumulate(arr, axis=-1)
    return (arr - peak).min(axis=-1)


def horizon_mdd_log(paths: np.ndarray, horizon: int) -> np.ndarray:
    """路径**前 ``horizon`` 个收益**上的最大回撤（log 尺度）。

    Raises:
        DrawdownReadingError: horizon 非正或超过路径长度（禁止按较短长度截断）。
    """
    values = np.asarray(paths, dtype=float)
    if values.ndim not in (1, 2):
        raise DrawdownReadingError(f"路径必须是 1D/2D，实际 shape={values.shape}")
    length = int(horizon)
    available = int(values.shape[-1])
    if length <= 0 or length > available:
        raise DrawdownReadingError(
            f"horizon={length} 非法（必须落在 1..{available}；禁止用更短长度冒充 look 终点）"
        )
    return max_drawdown_log(cumulative_log_path(values[..., :length]))


def log_to_percent(values: np.ndarray) -> np.ndarray:
    """log 尺度 → 百分点（``(exp(v) - 1) * 100``）。"""
    return (np.exp(np.asarray(values, dtype=float)) - 1.0) * 100.0


def percent_to_log(percent: np.ndarray) -> np.ndarray:
    """百分点 → log 尺度（``ln(1 + p/100)``；``p <= -100`` 非法）。"""
    values = np.asarray(percent, dtype=float)
    if (values <= -100.0).any():
        raise DrawdownReadingError("百分点读数 <= -100%（净值非正），无法转回 log 尺度")
    return np.log1p(values / 100.0)


def horizon_mdd_percent(paths: np.ndarray, horizon: int) -> np.ndarray:
    """前 ``horizon`` 个收益上的最大回撤（pp）。"""
    return log_to_percent(horizon_mdd_log(paths, horizon))


def drawdown_reading(arm: np.ndarray, base: np.ndarray, horizon: int) -> np.ndarray:
    """``D_k = MDD_arm(T_k) - MDD_base(T_k)``（pp；正值=改善，负值=恶化）。"""
    return horizon_mdd_percent(arm, horizon) - horizon_mdd_percent(base, horizon)


def reading_from_cum(cum_arm: np.ndarray, cum_base: np.ndarray, horizon: int) -> np.ndarray:
    """已预算累计曲线时的 ``D_k``（列截断到 ``horizon+1``，含首点 0）。

    Raises:
        DrawdownReadingError: horizon 越界（禁止按更短长度截断）。
    """
    first = np.asarray(cum_arm, dtype=float)
    second = np.asarray(cum_base, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise DrawdownReadingError(f"两臂累计曲线必须同形 2D，实际 {first.shape} vs {second.shape}")
    stop = int(horizon) + 1
    if stop > first.shape[1]:
        raise DrawdownReadingError(f"horizon={horizon} 超出累计曲线长度 {first.shape[1] - 1}")
    return log_to_percent(max_drawdown_log(first[:, :stop])) - log_to_percent(
        max_drawdown_log(second[:, :stop])
    )


def baseline_percent_by_look(base: np.ndarray, look_days: Sequence[int]) -> np.ndarray:
    """单条**基准路径**在各 look 终点上的 MDD（pp）。

    基准 = B0 冻结窗口（基准臂）；A/A 与效应单元的 ``D_k`` 都以它为减数
    （设计 §5.4 头部 ``MDD_base``），因此必须在**同一实现集**上复用同一组基准读数
    （禁止按实现重算基准窗口）。

    Args:
        base: ``(N,)`` 或 ``(1, N)`` 逐日 log 收益。
        look_days: look 终点（``{120,240,360,500}``）。

    Returns:
        ``(K,)`` 基准 MDD（pp，负值）。

    Raises:
        DrawdownReadingError: 形状非法或 look 越界。
    """
    values = np.asarray(base, dtype=float).reshape(-1)
    if values.size == 0:
        raise DrawdownReadingError("基准路径为空（无法给出 MDD_base）")
    cumulative = cumulative_log_path(values)
    columns = []
    for day in look_days:
        stop = int(day)
        if stop <= 0 or stop >= cumulative.size:
            raise DrawdownReadingError(
                f"look 终点 {stop} 越界（基准路径只有 {cumulative.size - 1} 个收益）"
            )
        raw = np.asarray([max_drawdown_log(cumulative[: stop + 1])], dtype=float)
        columns.append(float(log_to_percent(raw)[0]))
    return np.asarray(columns, dtype=float)


def drawdown_look_plan() -> LookPlan:
    """H=20 的 look 映射（复用 §5.1 唯一实现），并校验与冻结值一致。"""
    plan = make_look_plan(HOLD_H)
    validate_horizons(plan)
    return plan


def validate_horizons(plan: LookPlan) -> LookPlan:
    """校验 look 映射 = 冻结值（``{120,240,360,500}``、K=4、锚 = H*J_max = 500）。

    Raises:
        DrawdownReadingError: 与冻结值不符（禁止把 1714 全链档位套到本仪器上）。
    """
    if int(plan.h) != HOLD_H:
        raise DrawdownReadingError(f"H 必须为 {HOLD_H}（本仪器冻结值），实际 {plan.h}")
    if tuple(int(day) for day in plan.look_days) != LOOK_DAYS:
        raise DrawdownReadingError(
            f"look_days 必须等于冻结值 {list(LOOK_DAYS)}，实际 {list(plan.look_days)}"
        )
    if int(plan.max_horizon()) != ANCHOR_HORIZON:
        raise DrawdownReadingError(
            f"唯一锚必须等于 H*J_max={ANCHOR_HORIZON}，实际 {plan.max_horizon()}"
        )
    return plan
