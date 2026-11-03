# -*- coding: utf-8 -*-
"""P5b T6 账本 → 配对收益 → 裁判读数（**只读**，阈值只来自 T4/T5 冻结产物）。

契约来源：设计 v4 §5.5「H6 数值」行（收益 −5δ、DD-I +12pp、DD-W −12pp 各 1 条；只查点
估计方向与链路，不作检出率/MDE）、§6.3（单次演练只查方向与链路；两方向分别检查）、
§5.1/§5.2（look 映射与 block 统计量）、§5.4（``D_k`` 与两个方向的阈值）。

三条硬口径
==========
1. **通路不改**：配对收益只用 ``referee.ledger_source.paired_returns_from_ledger``
   （内部走 T3 ``query_shards`` 只读门与校验分片读）；阈值只从 T4/T5 冻结产物读；
2. **fail-closed**：样本不足 look、方差非正、路径非有限一律记无效原因，不得当作有利结论，
   也不得剔除样本；
3. **退化方差不得伪造 Z**：确定性位移臂（clone + 常数位移）的配对路径方差为零，block 统计量
   在该样本上**不可识别**；此时只披露原始数值伪影并标记 ``identifiable=false``，
   不得把它当作 Z 点估计。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.lazybull.v2.hosts.testbed.layout import CampaignLayout
from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.drawdown.injection import pure_injection_effect
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    ANCHOR_HORIZON,
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
    drawdown_look_plan,
    drawdown_reading,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.thresholds import (
    IMPROVE_EVENT,
    WORSEN_EVENT,
)
from src.lazybull.v2.hosts.testbed.referee.statistics import (
    DELTA_LOG_DAILY,
    block_means,
    variance_psd_lag1,
)
from src.lazybull.v2.hosts.testbed.rehearsal.injection import (
    DEGENERATE_SPREAD_TOL,
    RehearsalEffect,
    income_daily_shift,
    window_mask_from_frozen,
)

__all__ = [
    "ReadingsError",
    "drawdown_look_reading",
    "frozen_thresholds",
    "h6_drawdown_item",
    "h6_income_item",
    "income_look_reading",
    "paired_paths",
]

#: look 级状态码
LOOK_OK = "ok"
LOOK_INSUFFICIENT = "insufficient_blocks"
LOOK_NON_POSITIVE_VARIANCE = "non_positive_variance"
LOOK_DEGENERATE = "degenerate_variance"
#: H6 条目状态码
ITEM_PASS = "pass"
ITEM_PARTIAL = "partial"
ITEM_FAIL = "fail"
ITEM_NOT_EXECUTED = "not_executed"


class ReadingsError(TestbedError):
    """读数输入非法（配对路径不齐、阈值缺失、方向未知）。"""

    __test__ = False


def frozen_thresholds(
    t4_thresholds: Mapping[str, Any], t5_thresholds: Mapping[str, Any]
) -> Dict[str, Any]:
    """从 T4/T5 冻结临界值产物读取读数参数（**不重训、不改值**）。"""
    c_k = [float(value) for value in t4_thresholds["c_by_look"]]
    c_w = [float(value) for value in t5_thresholds["c_w_by_look"]]
    c_i = [float(value) for value in t5_thresholds["c_i_by_look"]]
    plan = drawdown_look_plan()
    if len(c_k) != int(plan.k) or len(c_w) != int(plan.k) or len(c_i) != int(plan.k):
        raise ReadingsError(
            "临界值个数与冻结 look 数不符（拒绝运行）: "
            f"c_k={len(c_k)} c_w={len(c_w)} c_i={len(c_i)} K={plan.k}"
        )
    return {
        "look_days": [int(day) for day in plan.look_days],
        "look_blocks": [int(j) for j in plan.j_looks],
        "h": int(plan.h),
        "c_k": c_k,
        "c_w": c_w,
        "c_i": c_i,
        "delta_log_daily": float(DELTA_LOG_DAILY),
        "anchor": ANCHOR_HORIZON,
        "source": "T4 calibration_thresholds + T5 drawdown_thresholds（只读引用）",
    }


def paired_paths(
    layout: CampaignLayout,
    arm_candidate: str,
    arm_base: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """取配对路径（收益差 + 两臂 NAV 路径），只读、不改账本。"""
    from src.lazybull.v2.hosts.testbed.referee.ledger_source import (
        nav_series_from_ledger,
        paired_returns_from_ledger,
    )

    pair = paired_returns_from_ledger(layout, arm_candidate, arm_base, start=start, end=end)
    series_candidate = nav_series_from_ledger(layout, arm_candidate, start=start, end=end)
    series_base = nav_series_from_ledger(layout, arm_base, start=start, end=end)
    return {
        "pair": pair,
        "returns_candidate": np.diff(np.log(np.asarray(series_candidate["nav"], dtype=float))),
        "returns_base": np.diff(np.log(np.asarray(series_base["nav"], dtype=float))),
        "nav_candidate": [float(value) for value in series_candidate["nav"]],
        "nav_base": [float(value) for value in series_base["nav"]],
        "dates": list(series_base["dates"]),
        "return_dates": list(series_base["dates"][1:]),
        "d0_date": series_base["dates"][0],
        "published_generation": dict(pair["published_generation"]),
        "chain_digests": dict(pair["chain_digests"]),
        "read_path": pair["read_path"],
    }


def _look_block_stats(returns: np.ndarray, blocks: int, height: int) -> Dict[str, Any]:
    """单 look 的 block 均值与 PSD 方差（含退化判据）。"""
    needed = int(blocks) * int(height)
    if returns.size < needed:
        return {
            "status": LOOK_INSUFFICIENT,
            "needed_samples": needed,
            "available_samples": int(returns.size),
            "parameter": None,
            "variance": None,
            "spread": None,
            "raw_statistic": None,
            "identifiable": False,
        }
    means = block_means(returns[None, :], blocks, height)[0]
    parameter = float(means.mean())
    centered = means - parameter
    variance = float(variance_psd_lag1(centered[None, :])[0])
    spread = float(np.abs(centered).max()) if centered.size else 0.0
    if not np.isfinite(variance) or variance <= 0.0:
        return {
            "status": LOOK_NON_POSITIVE_VARIANCE,
            "needed_samples": needed,
            "available_samples": int(returns.size),
            "parameter": parameter,
            "variance": variance,
            "spread": spread,
            "raw_statistic": None,
            "identifiable": False,
        }
    raw = (parameter + float(DELTA_LOG_DAILY)) / float(np.sqrt(variance / blocks))
    degenerate = spread <= DEGENERATE_SPREAD_TOL * max(1.0, abs(parameter))
    return {
        "status": LOOK_DEGENERATE if degenerate else LOOK_OK,
        "needed_samples": needed,
        "available_samples": int(returns.size),
        "parameter": parameter,
        "variance": variance,
        "spread": spread,
        "raw_statistic": raw,
        "identifiable": not degenerate,
    }


def income_look_reading(returns: np.ndarray, thresholds: Mapping[str, Any]) -> Dict[str, Any]:
    """收益臂逐 look 读数（冻结统计量；退化方差不算 Z）。"""
    values = np.asarray(returns, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ReadingsError("配对收益含非有限值（fail-closed）")
    rows: List[Dict[str, Any]] = []
    for index, day in enumerate(thresholds["look_days"]):
        stats = _look_block_stats(
            values, int(thresholds["look_blocks"][index]), int(thresholds["h"])
        )
        z_value = stats["raw_statistic"] if stats["identifiable"] else None
        veto = bool(z_value is not None and z_value < -float(thresholds["c_k"][index]))
        rows.append(
            {
                "look": index,
                "look_day": int(day),
                "blocks": int(thresholds["look_blocks"][index]),
                "c_k": float(thresholds["c_k"][index]),
                **stats,
                "z_point_estimate": None if z_value is None else float(z_value),
                "veto": veto,
            }
        )
    return {
        "statistic": "psd_bartlett_lag1_boundary_centered",
        "delta_log_daily": float(thresholds["delta_log_daily"]),
        "sample_days": int(values.size),
        "paired_mean": float(values.mean()) if values.size else None,
        "rows": rows,
        "executable_looks": [row["look"] for row in rows if row["status"] != LOOK_INSUFFICIENT],
        "identifiable_looks": [row["look"] for row in rows if row["identifiable"]],
    }


def drawdown_look_reading(
    returns_arm: np.ndarray,
    returns_base: np.ndarray,
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    """回撤臂逐 look 的 ``D_k`` 与两个方向的事件（分别判定，互不抵扣）。"""
    arm = np.asarray(returns_arm, dtype=float).reshape(-1)
    base = np.asarray(returns_base, dtype=float).reshape(-1)
    if arm.size != base.size:
        raise ReadingsError(f"两臂收益长度不一致（禁止错臂/错日配对）: {arm.size} vs {base.size}")
    if not np.isfinite(arm).all() or not np.isfinite(base).all():
        raise ReadingsError("回撤读数输入含非有限值（fail-closed）")
    rows: List[Dict[str, Any]] = []
    for index, day in enumerate(thresholds["look_days"]):
        if arm.size < int(day):
            rows.append(
                {
                    "look": index,
                    "look_day": int(day),
                    "status": LOOK_INSUFFICIENT,
                    "needed_samples": int(day),
                    "available_samples": int(arm.size),
                    "d_pp": None,
                    "c_w": float(thresholds["c_w"][index]),
                    "c_i": float(thresholds["c_i"][index]),
                    "worsen_event": False,
                    "improve_event": False,
                }
            )
            continue
        reading = float(drawdown_reading(arm, base, int(day)))
        worsen = bool(reading < float(thresholds["c_w"][index]))
        improve = bool(reading > float(thresholds["c_i"][index]))
        if worsen and improve:
            raise ReadingsError(
                f"同一 look 同时触发两个方向（数值互斥被破坏）: look={index} D={reading}"
            )
        rows.append(
            {
                "look": index,
                "look_day": int(day),
                "status": LOOK_OK,
                "needed_samples": int(day),
                "available_samples": int(arm.size),
                "d_pp": reading,
                "c_w": float(thresholds["c_w"][index]),
                "c_i": float(thresholds["c_i"][index]),
                "worsen_event": worsen,
                "improve_event": improve,
            }
        )
    return {
        "rows": rows,
        "executable_looks": [row["look"] for row in rows if row["status"] == LOOK_OK],
    }


def h6_income_item(
    effect: RehearsalEffect, reading: Mapping[str, Any], expected_shift: float
) -> Dict[str, Any]:
    """H6 收益臂条目（只判方向与链路；Z 不可识别时如实披露）。"""
    rows = list(reading["rows"])
    residual = None
    mean = rows[0].get("parameter") if rows else None
    paired_mean = _paired_mean(reading)
    if paired_mean is not None:
        residual = abs(paired_mean - float(expected_shift))
    identifiable = [row for row in rows if row["identifiable"]]
    direction_ok = bool(paired_mean is not None and paired_mean < 0.0)
    magnitude_ok = bool(
        residual is not None and residual <= max(1e-9, abs(float(expected_shift)) * 1e-6)
    )
    status = _income_status(paired_mean, direction_ok, magnitude_ok, identifiable, rows)
    return {
        "name": "income_minus_5d",
        "arm_id": effect.arm_id,
        "target": "配对日度 log 差均值 = −5δ",
        "target_value": float(expected_shift),
        "paired_mean": paired_mean,
        "max_abs_residual": residual,
        "direction_ok": direction_ok,
        "magnitude_ok": magnitude_ok,
        "paired_sample_days": int(reading["sample_days"]),
        "looks": rows,
        "executable_looks": list(reading["executable_looks"]),
        "identifiable_looks": list(reading["identifiable_looks"]),
        "z_point_estimate": (
            None if not identifiable else float(identifiable[0]["z_point_estimate"])
        ),
        "z_identifiable": bool(identifiable),
        "mean_of_first_look": mean,
        "status": status,
        "judgment": _income_judgment(status),
        "evidence_fields": [
            "h6.items[0].paired_mean",
            "h6.items[0].max_abs_residual",
            "h6.items[0].z_point_estimate",
            "h6.items[0].looks",
        ],
    }


def _paired_mean(reading: Mapping[str, Any]) -> Optional[float]:
    """配对均值（取样本参数；不足一个 look 时由调用方从原始收益给出）。"""
    return reading.get("paired_mean")


def _income_status(
    paired_mean: Optional[float],
    direction_ok: bool,
    magnitude_ok: bool,
    identifiable: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """按**冻结判据**判定：方向为负 + 幅度一致 + 链路完整（样本在）。

    Z 点估计是否可识别（``identifiable``）只作披露字段（退化方差是确定性位移臂的固有
    后果），**不改变**方向判定；报告与判定表使用同一状态词表，避免「判据已满足却报未执行」。
    """
    if paired_mean is None or not rows:
        return ITEM_NOT_EXECUTED
    if not direction_ok:
        return ITEM_FAIL
    return ITEM_PASS if magnitude_ok else ITEM_FAIL


def _income_judgment(status: str) -> str:
    return {
        ITEM_PASS: "方向为负、位移幅度符合预登记且链路完整（Z 不可识别时只披露，不改变判定）",
        ITEM_PARTIAL: "（保留词表；收益臂不再使用：幅度不符已改判 fail）",
        ITEM_FAIL: "配对方向不符合预登记位移（判定失败）",
        ITEM_NOT_EXECUTED: "无配对收益样本（未执行）",
    }[status]


def h6_drawdown_item(
    effect: RehearsalEffect,
    reading: Optional[Mapping[str, Any]],
    *,
    direction: str,
    pure_effect_pp: Optional[float] = None,
) -> Dict[str, Any]:
    """H6 回撤臂条目（锚 look 的方向 + 事件；不可构造即未执行）。"""
    if direction not in (DIRECTION_IMPROVE, DIRECTION_WORSEN):
        raise ReadingsError(f"未知回撤方向: {direction!r}")
    anchor_index = len(drawdown_look_plan().look_days) - 1
    anchor_row = None
    if reading is not None:
        anchor_row = next(
            (row for row in reading["rows"] if int(row["look"]) == anchor_index), None
        )
    constructible = effect.status == "constructible"
    status, judgment = _drawdown_status(constructible, anchor_row, direction)
    threshold_key = "c_i" if direction == DIRECTION_IMPROVE else "c_w"
    # 目标必须取**冻结件**里该臂的 target_pp（DD-W 已按 c^W_anchor-2pp 修正），
    # 不得回退到常量：否则报告与 manifest 会互相矛盾（第四轮复审 P1）。
    target_pp = float(effect.target_pp)
    return {
        "name": "dd_improve_12pp" if direction == DIRECTION_IMPROVE else "dd_worsen_12pp",
        "arm_id": effect.arm_id,
        "direction": direction,
        "target": f"锚 look 纯注入效应 ≈ {target_pp:+.2f}pp",
        "target_pp": target_pp,
        "anchor": int(effect.anchor or ANCHOR_HORIZON),
        "constructibility": effect.status,
        "constructibility_reason": effect.reason,
        "required_return_days": int(effect.required_return_days),
        "available_return_days": (
            None if reading is None else int(reading["rows"][0]["available_samples"])
        ),
        "parameter": effect.parameter,
        "parameter_name": effect.parameter_name,
        "pure_effect_pp": pure_effect_pp,
        "d_anchor_pp": None if anchor_row is None else anchor_row["d_pp"],
        "threshold_pp": None if anchor_row is None else anchor_row[threshold_key],
        threshold_key + "_anchor_pp": None if anchor_row is None else anchor_row[threshold_key],
        "anchor_status": None if anchor_row is None else anchor_row["status"],
        "event": _drawdown_event(anchor_row, direction),
        "looks": [] if reading is None else reading["rows"],
        "executable_looks": [] if reading is None else reading["executable_looks"],
        "status": status,
        "judgment": judgment,
        "evidence_fields": [
            f"h6.items[{1 if direction == DIRECTION_IMPROVE else 2}].constructibility",
            f"h6.items[{1 if direction == DIRECTION_IMPROVE else 2}].d_anchor_pp",
            f"h6.items[{1 if direction == DIRECTION_IMPROVE else 2}].{threshold_key}_anchor_pp",
            f"h6.items[{1 if direction == DIRECTION_IMPROVE else 2}].event",
        ],
    }


def _drawdown_status(
    constructible: bool, anchor_row: Optional[Mapping[str, Any]], direction: str
) -> tuple:
    if not constructible:
        return ITEM_NOT_EXECUTED, "锚 look 不可构造（如实记录，不改报目标值/不截断档位）"
    if anchor_row is None or anchor_row["status"] != LOOK_OK:
        return ITEM_NOT_EXECUTED, "锚 look 样本不足（未执行）"
    triggered = bool(
        anchor_row["improve_event"]
        if direction == DIRECTION_IMPROVE
        else anchor_row["worsen_event"]
    )
    if not triggered:
        return ITEM_FAIL, "锚 look 未触发预登记事件（判定失败，不得用另一条抵扣）"
    return ITEM_PASS, "锚 look 触发预登记事件（仍不得据此宣称检出率/MDE）"


def _drawdown_event(anchor_row: Optional[Mapping[str, Any]], direction: str) -> str:
    if anchor_row is None or anchor_row["status"] != LOOK_OK:
        return "not_executed"
    if direction == DIRECTION_IMPROVE:
        return IMPROVE_EVENT if anchor_row["improve_event"] else "no_event"
    return WORSEN_EVENT if anchor_row["worsen_event"] else "no_event"


def expected_income_shift(effect: RehearsalEffect) -> float:
    """收益臂的冻结日度位移（含参数校验）。"""
    return income_daily_shift(5.0) if effect.kind == "income_log_shift" else 0.0


def pure_effect_at_anchor(
    returns_base: np.ndarray,
    effect: RehearsalEffect,
    frozen_list: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    """可构造时的锚上纯注入效应（``MDD(arm)-MDD(x)``，pp）。"""
    if not effect.active or effect.kind != "crash_window_transform":
        return None
    values = np.asarray(returns_base, dtype=float).reshape(-1)
    if values.size < ANCHOR_HORIZON:
        return None
    mask, _ = window_mask_from_frozen(frozen_list, values.size)
    parameter = np.asarray([float(effect.parameter or 0.0)], dtype=float)
    value = pure_injection_effect(
        values[None, :],
        mask,
        direction=str(effect.direction),
        parameter=parameter,
        anchor=ANCHOR_HORIZON,
    )
    return float(np.asarray(value).reshape(-1)[0])
