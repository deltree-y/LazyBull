# -*- coding: utf-8 -*-
"""P5b T6 演练效应注入（**基准路径的纯函数**；只改 NAV/日收益）。

契约来源：设计 v4 §6.2（五臂构造：基准 / 机械 A/A / 已知收益效应 / DD-I 改善 / DD-W 恶化）、
§6.3（机械 A/A 逐值一致；只查方向与链路）、§5.5「H6 数值」行（收益 −5δ、DD-I +12pp、
DD-W −12pp 各 1 条）、§5.4.3（崩盘窗与 λ/ζ 求解语义）、§5.6（限制声明）。

三条纪律
========
1. **纯函数**：``同样的 base + 同样的冻结参数 ⇒ 逐字节同样的结果``（无随机、无墙钟、
   无全局状态）；注入参数一律来自 T6 manifest（盲态预登记），运行期不得按读数调整；
2. **只改 NAV**：变换只作用于日 ``log`` 收益，再据初始资金重建 NAV；成交、现金、持仓、
   日期一律不动（由 :mod:`hosts` 在宿主层落实）；
3. **fail-closed**：非有限、越界参数、路径断裂（非正/非有限 NAV）一律报错，禁止静默
   降级为 0/NaN。

窗口口径（唯一可实现的「同一组窗」）
====================================
T5 冻结的崩盘窗是 **B0 最后 500 个收益样本上的样本序号区间** ``[peak, trough)``
（``date_semantics=fold_local_ordinal``，**没有自然日**）。因此演练路径只能按**样本序号对齐**
（``index_aligned``）复用它：第 ``i`` 个收益样本（D1 起算，0 基）对应冻结序号 ``i``。
禁止按日期重选窗、禁止按深度重排、禁止按路径长度截断窗（越界即 ``not_constructible``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.drawdown.injection import (
    InjectionSolution,
    solve_injection,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    ANCHOR_HORIZON,
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
)
from src.lazybull.v2.hosts.testbed.referee.statistics import (
    DELTA_LOG_DAILY,
    delta_annualized_ok,
)

__all__ = [
    "ANCHOR_TARGET_PP",
    "DEGENERATE_SPREAD_TOL",
    "EFFECT_INCOME",
    "EFFECT_NONE",
    "EFFECT_WINDOW",
    "INCOME_DELTA_MULTIPLE",
    "InjectionError",
    "RehearsalEffect",
    "apply_single_return",
    "arm_effect_plan",
    "build_arm_effect",
    "income_daily_shift",
    "log_returns_from_nav",
    "nav_from_log_returns",
    "solve_anchor_parameter",
    "transform_log_returns",
    "window_mask_from_frozen",
]

#: 收益臂的冻结位移倍数（设计 §5.3 网格的 5δ 档；不得按 look 或按结果改倍数）
INCOME_DELTA_MULTIPLE = 5.0
#: 两个回撤臂的冻结目标（pp；DD-I 为正=改善、DD-W 为负=恶化）
ANCHOR_TARGET_PP: Mapping[str, float] = {
    DIRECTION_IMPROVE: 12.0,
    DIRECTION_WORSEN: -12.0,
}
#: 效应种类（与 manifest 的 ``arms[*].injection.kind`` 一一对应）
EFFECT_NONE = "none"
EFFECT_INCOME = "income_log_shift"
EFFECT_WINDOW = "crash_window_transform"
#: 退化方差判据（相对）：中心化 block 均值的最大绝对值低于该比例即判「路径退化为确定性」
DEGENERATE_SPREAD_TOL = 1e-12


class InjectionError(TestbedError):
    """注入参数/路径非法（越界、非有限、非正 NAV、窗口越界）。"""

    __test__ = False


@dataclass(frozen=True)
class RehearsalEffect:
    """单臂的冻结效应计划（manifest 冻结值在运行期的只读投影）。"""

    arm_id: str
    kind: str
    status: str
    direction: Optional[str]
    target_pp: Optional[float]
    anchor: Optional[int]
    parameter: Optional[float]
    parameter_name: Optional[str]
    required_return_days: int
    reason: Optional[str]
    window_bounds: Optional[List[int]]

    @property
    def active(self) -> bool:
        """是否真正施加注入（``constructible`` 且参数已冻结）。"""
        return self.status == "constructible" and self.kind != EFFECT_NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "kind": self.kind,
            "status": self.status,
            "direction": self.direction,
            "target_pp": self.target_pp,
            "anchor": self.anchor,
            "parameter": self.parameter,
            "parameter_name": self.parameter_name,
            "required_return_days": int(self.required_return_days),
            "reason": self.reason,
            "window_bounds": self.window_bounds,
            "active": self.active,
        }


def income_daily_shift(multiple: float = INCOME_DELTA_MULTIPLE) -> float:
    """收益臂的冻结日度 log 位移（``-multiple * δ``，δ = ln(1.05)/252）。

    Raises:
        InjectionError: δ 与冻结字面值不符，或倍数不为冻结值（禁止另立口径）。
    """
    if not delta_annualized_ok(DELTA_LOG_DAILY):
        raise InjectionError(
            f"收益裁判冻结 δ 与 ln(1.05)/252 不一致（禁止另立口径）: {DELTA_LOG_DAILY!r}"
        )
    value = float(multiple)
    if not math.isclose(value, INCOME_DELTA_MULTIPLE, rel_tol=0.0, abs_tol=1e-12):
        raise InjectionError(
            f"收益效应倍数必须等于冻结值 {INCOME_DELTA_MULTIPLE}，实际 {multiple!r}"
        )
    return -value * float(DELTA_LOG_DAILY)


def window_mask_from_frozen(
    frozen_list: Sequence[Mapping[str, Any]], length: int
) -> Tuple[np.ndarray, List[int]]:
    """把 T5 冻结窗清单转成演练路径上的布尔掩码（**样本序号对齐**，不重选）。

    Args:
        frozen_list: T5 manifest ``drawdown_referee.windows.frozen_list``。
        length: 演练路径的收益样本数。

    Returns:
        ``(mask, stops)``：``mask`` 为 ``(length,)`` 布尔；``stops`` 为各窗 ``trough`` 序号。

    Raises:
        InjectionError: 窗清单为空、序号非法或越出演练路径（禁止截断）。
    """
    total = int(length)
    if total <= 0:
        raise InjectionError(f"演练路径收益样本必须为正，实际 {length!r}")
    if not frozen_list:
        raise InjectionError("冻结崩盘窗清单为空（回撤臂无法构造）")
    mask = np.zeros(total, dtype=bool)
    stops: List[int] = []
    for entry in frozen_list:
        start = int(entry["window_start_ordinal"])
        stop = int(entry["window_stop_ordinal"])
        if start < 0 or stop <= start:
            raise InjectionError(f"冻结窗序号非法（须 0 <= peak < trough）: {entry!r}")
        if stop > total:
            raise InjectionError(
                f"冻结窗 [{start}, {stop}) 越出演练路径 {total} 个收益样本（禁止截断窗）"
            )
        mask[start:stop] = True
        stops.append(stop)
    return mask, stops


def solve_anchor_parameter(
    anchor_returns: np.ndarray,
    frozen_list: Sequence[Mapping[str, Any]],
    *,
    direction: str,
    target_pp: float,
) -> InjectionSolution:
    """在**冻结 B0 锚窗**上按 T5 求解器语义求 λ/ζ（盲态；不读演练读数）。

    复用 ``referee.drawdown.injection.solve_injection``：101 点扫描 + 最多 80 次细化、
    0.05pp 容差、多解取最小干预；锚固定 ``H*J_max=500``。

    Raises:
        InjectionError: 锚窗长度与 ``ANCHOR_HORIZON`` 不符，或方向/目标非冻结组合。
    """
    values = np.asarray(anchor_returns, dtype=float).reshape(-1)
    if values.size != ANCHOR_HORIZON:
        raise InjectionError(
            f"锚窗收益样本必须等于 {ANCHOR_HORIZON}，实际 {values.size}（禁止换成其它档位）"
        )
    if str(direction) not in ANCHOR_TARGET_PP:
        raise InjectionError(f"未知回撤方向: {direction!r}")
    # 目标不再钉死 ±12pp：由 manifest 预登记推导（DD-W 必须越过 T5 冻结阈值才有事件）
    if str(direction) == DIRECTION_IMPROVE and not (0.0 < float(target_pp) <= 40.0):
        raise InjectionError(f"DD-I 目标必须为正且 ≤40pp，实际 {target_pp!r}")
    if str(direction) == DIRECTION_WORSEN and not (-40.0 <= float(target_pp) < 0.0):
        raise InjectionError(f"DD-W 目标必须为负且 ≥-40pp，实际 {target_pp!r}")
    mask, _ = window_mask_from_frozen(frozen_list, values.size)
    return solve_injection(
        values[None, :],
        mask,
        direction=str(direction),
        target_pp=float(target_pp),
        anchor=ANCHOR_HORIZON,
    )


def build_arm_effect(
    arm_id: str,
    spec: Mapping[str, Any],
    *,
    return_days: int,
    frozen_list: Sequence[Mapping[str, Any]],
) -> RehearsalEffect:
    """按 manifest 冻结的臂效应声明构造运行期效应（含可构造性门）。

    可构造性门（**在任何读数之前**判定，判定规则已写进 manifest）：

    - DD 臂：``return_days >= T_anchor(500)`` 且各冻结窗 ``trough <= return_days``；
    - 收益臂：``return_days >= 1``。

    不满足即 ``not_constructible``（不改报目标值、不截断档位、不换更短锚）。
    """
    kind = str(spec["kind"])
    total = int(return_days)
    if kind == EFFECT_NONE:
        return RehearsalEffect(
            arm_id=arm_id,
            kind=kind,
            status="constructible",
            direction=None,
            target_pp=None,
            anchor=None,
            parameter=None,
            parameter_name=None,
            required_return_days=0,
            reason=None,
            window_bounds=None,
        )
    if kind == EFFECT_INCOME:
        status = "constructible" if total >= 1 else "not_constructible"
        return RehearsalEffect(
            arm_id=arm_id,
            kind=kind,
            status=status,
            direction=None,
            target_pp=None,
            anchor=None,
            parameter=income_daily_shift(float(spec["delta_multiple"])),
            parameter_name="delta_multiple",
            required_return_days=1,
            reason=None if status == "constructible" else "no_return_sample",
            window_bounds=None,
        )
    if kind != EFFECT_WINDOW:
        raise InjectionError(
            f"未知效应种类: {kind!r}（允许 {EFFECT_NONE}/{EFFECT_INCOME}/{EFFECT_WINDOW}）"
        )
    return _window_effect(arm_id, spec, total, frozen_list)


def _window_effect(
    arm_id: str,
    spec: Mapping[str, Any],
    return_days: int,
    frozen_list: Sequence[Mapping[str, Any]],
) -> RehearsalEffect:
    """DD 臂：冻结窗越界或样本不足锚 ⇒ ``not_constructible``（原因可机器读）。"""
    direction = str(spec["direction"])
    target = float(spec["target_pp"])
    anchor = int(spec["anchor"])
    frozen = [dict(entry) for entry in frozen_list]
    bounds = [int(entry["window_stop_ordinal"]) for entry in frozen]
    needed = max([anchor, *bounds])
    parameter = spec.get("parameter")
    status = "constructible"
    reason: Optional[str] = None
    if return_days < needed:
        status = "not_constructible"
        reason = "anchor_horizon_exceeds_rehearsal_window"
    elif parameter is None:
        status = "not_constructible"
        reason = "anchor_parameter_not_constructible"
    return RehearsalEffect(
        arm_id=arm_id,
        kind=EFFECT_WINDOW,
        status=status,
        direction=direction,
        target_pp=target,
        anchor=anchor,
        parameter=None if parameter is None else float(parameter),
        parameter_name="lambda" if direction == DIRECTION_IMPROVE else "zeta",
        required_return_days=int(needed),
        reason=reason,
        window_bounds=bounds,
    )


def log_returns_from_nav(nav: Sequence[float]) -> np.ndarray:
    """NAV 序列 → 逐日 log 收益（``D0`` 只作估值点，不产生收益）。

    Raises:
        InjectionError: 长度不足 2、含非正/非有限 NAV（路径断裂）。
    """
    values = np.asarray(list(nav), dtype=float).reshape(-1)
    if values.size < 2:
        raise InjectionError(f"NAV 序列至少需要 2 个估值点（D0 + D1），实际 {values.size}")
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise InjectionError("NAV 序列含非有限或非正值（注入路径不成立，禁止降级）")
    return np.diff(np.log(values))


def nav_from_log_returns(initial: float, returns: Sequence[float]) -> np.ndarray:
    """由初始 NAV + 逐日 log 收益重建 NAV 序列（含首点）。

    Raises:
        InjectionError: 初始 NAV 非正/非有限，或收益非有限、重建后非正。
    """
    start = float(initial)
    if not math.isfinite(start) or start <= 0.0:
        raise InjectionError(f"初始 NAV 必须为有限正值，实际 {initial!r}")
    values = np.asarray(list(returns), dtype=float).reshape(-1)
    if values.size and not np.isfinite(values).all():
        raise InjectionError("逐日 log 收益含非有限值（注入结果必须有限）")
    path = start * np.exp(np.concatenate(([0.0], np.cumsum(values))))
    if not np.isfinite(path).all() or (path <= 0.0).any():
        raise InjectionError("重建后的 NAV 路径出现非有限或非正值（fail-closed）")
    return path


def _window_factor(effect: RehearsalEffect) -> float:
    """DD 臂的冻结乘子（并校验参数取值域）。"""
    parameter = effect.parameter
    if parameter is None:
        raise InjectionError(f"臂 {effect.arm_id!r} 缺冻结参数（禁止以 0 代替）")
    value = float(parameter)
    if effect.direction == DIRECTION_IMPROVE:
        if not (0.0 <= value <= 1.0):
            raise InjectionError(f"λ 越界（须落在 [0,1]）: {value}")
        return 1.0 - value
    if effect.direction == DIRECTION_WORSEN:
        if not (0.0 <= value <= 2.0):
            raise InjectionError(f"ζ 越界（须落在 [0,2]）: {value}")
        return 1.0 + value
    raise InjectionError(f"未知回撤方向: {effect.direction!r}")


def apply_single_return(
    value: float,
    index: int,
    effect: RehearsalEffect,
    *,
    mask: Optional[np.ndarray] = None,
) -> float:
    """对**单日** log 收益施加冻结效应（宿主逐日注入的唯一实现）。

    与 :func:`transform_log_returns` 的整条路径版本逐值等价（由用例锁定）。

    Args:
        value: 当日 log 收益（``x_t``）。
        index: 该收益在样本序列中的 0 基序号（D1 ⇒ 0）。
        effect: 冻结效应。
        mask: DD 臂的窗口掩码（``None`` ⇒ 未提供，按越界处理）。

    Raises:
        InjectionError: 非有限输入/输出，或 DD 臂缺掩码。
    """
    current = float(value)
    if not math.isfinite(current):
        raise InjectionError("当日 log 收益非有限（注入不成立）")
    if not effect.active:
        return current
    if effect.kind == EFFECT_INCOME:
        out = current + float(effect.parameter or 0.0)
    else:
        if mask is None:
            raise InjectionError(f"臂 {effect.arm_id!r} 缺窗口掩码（禁止无窗注入）")
        flags = np.asarray(mask, dtype=bool)
        position = int(index)
        if position < 0 or position >= flags.size:
            raise InjectionError(
                f"收益序号 {position} 越出窗口掩码长度 {flags.size}（禁止越界注入）"
            )
        out = current * _window_factor(effect) if (flags[position] and current < 0.0) else current
    if not math.isfinite(out):
        raise InjectionError("注入后的 log 收益非有限（fail-closed）")
    return out


def transform_log_returns(
    returns: np.ndarray, effect: RehearsalEffect, *, frozen_list: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """把冻结效应施加到整条 log 收益路径（纯函数；与宿主逐日施加等价）。

    - ``income``：全样本 ``x'_t = x_t - 5δ``；
    - ``crash_window``：**仅窗口内负收益** ``x'_t = (1-λ)x_t`` / ``(1+ζ)x_t``，
      正收益日与窗口外不变（与 T5 注入公式逐字一致）；
    - ``none`` / 未激活：原样返回（拷贝）。

    Raises:
        InjectionError: 形状非法、参数越界、非有限。
    """
    values = np.asarray(returns, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise InjectionError("基准 log 收益含非有限值")
    if not effect.active:
        return values.copy()
    if effect.kind == EFFECT_INCOME:
        return values + float(effect.parameter)
    mask, _ = window_mask_from_frozen(frozen_list, values.size)
    support = (values < 0.0) & mask
    factor = _window_factor(effect)
    out = np.where(support, values * factor, values)
    if not np.isfinite(out).all():
        raise InjectionError("注入后的 log 收益含非有限值（fail-closed）")
    return out


def arm_effect_plan(document: Mapping[str, Any], return_days: int) -> Dict[str, RehearsalEffect]:
    """按 manifest 生成全部臂的效应计划（键 = ``arm_id``，顺序稳定）。"""
    arms = dict(document["arms"])
    frozen = list(document["injection_window"]["frozen_list"])
    return {
        str(arm_id): build_arm_effect(
            arm_id,
            dict(arms[arm_id]["injection"]),
            return_days=return_days,
            frozen_list=frozen,
        )
        for arm_id in sorted(arms)
    }
