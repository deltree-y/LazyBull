# -*- coding: utf-8 -*-
"""判据纯函数（协议 §1、§7.1、§7.3、§7.4）。

只做**布尔判定**，不做指标计算：CAGR 与 ``mdd_loss`` 必须由
:func:`src.lazybull.v2.common.chain_metrics.calculate_chain_metrics` 给出
（``max_drawdown`` 为负，``mdd_loss = -max_drawdown`` 为正）。本模块不复制公式。

符号纪律（复审整改第 5 项）
==========================
- 门槛函数只接受**损失幅度**（≥0）。把公共指标的原始负回撤直接传入 → 直接报错，
  禁止静默取反、禁止静默按绝对值处理。
- 需要从原始 ``max_drawdown`` 转换时，只允许走 :func:`mdd_loss_from_raw`：它要求
  输入 ≤ 0（或全 0），否则报错。
- ``-30%`` 的损失不可能通过 ``< 20%`` 的门；``R 损失 30% / F 损失 20%`` 也不会被判成改善。

轴纪律
======
- 主窗净值轴必须与冻结的 1,839 点逐日相等；多腿/多臂对照必须各自对齐预定日期轴；
- 空值、非有限值（±inf）、重复日期、未升序、单点轴一律不通过。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

from scripts.v2_p7.cashflow_sleeve_split import spec


class GateError(RuntimeError):
    """判据输入不合法（缺值、非有限、符号错、轴不符）。"""


def _finite(value: Any, label: str) -> float:
    if value is None:
        raise GateError(f"判据输入缺值，禁止通过：{label}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise GateError(f"判据输入不是数值：{label}={value!r}") from error
    if not math.isfinite(number):
        raise GateError(f"判据输入非有限：{label}={number}")
    return number


def _loss(value: Any, label: str) -> float:
    """损失幅度校验：必须 ≥ 0（原始 max_drawdown 为负，不得直接传入）。"""
    number = _finite(value, label)
    if number < 0:
        raise GateError(
            f"{label}={number} 为负：这是 max_drawdown 的符号，不是损失幅度；"
            f"请先用 mdd_loss_from_raw 转成 mdd_loss = -max_drawdown"
        )
    return number


def mdd_loss_from_raw(max_drawdown: Any, label: str = "max_drawdown") -> float:
    """公共指标 → 损失幅度（唯一允许的符号转换点）。"""
    number = _finite(max_drawdown, label)
    if number > 0:
        raise GateError(f"{label}={number} 为正：公共指标的最大回撤必须 ≤ 0，符号或实现不符")
    return -number


def r_gate(cagr: Any, mdd_loss: Any, f_mdd_loss: Any) -> Dict[str, Any]:
    """R 的单袖研究目标（协议 §7.3）：CAGR ≥26%（含等号）且回撤损失严格小于 F。"""
    r_cagr = _finite(cagr, "R.cagr")
    r_mdd = _loss(mdd_loss, "R.mdd_loss")
    f_mdd = _loss(f_mdd_loss, "F.mdd_loss")
    cagr_ok = r_cagr >= float(spec.TARGETS["r_cagr_min"])
    mdd_ok = r_mdd < f_mdd
    return {
        "gate": "r_single_sleeve",
        "inputs": {"cagr": r_cagr, "mdd_loss": r_mdd, "f_mdd_loss": f_mdd},
        "booleans": {"cagr_ge_26pct": cagr_ok, "mdd_loss_strictly_below_f": mdd_ok},
        "passed": bool(cagr_ok and mdd_ok),
        "rules": {"cagr": ">= 0.26（含等号）", "mdd": "< F.mdd_loss（严格下降）"},
        "cagr_distance_to_target": r_cagr - float(spec.TARGETS["r_cagr_min"]),
        "mdd_loss_delta_vs_f": r_mdd - f_mdd,
    }


def portfolio_gate(cagr: Any, mdd_loss: Any, capital: Optional[int] = None) -> Dict[str, Any]:
    """单档组合门（净 CAGR > 25% 且回撤损失 < 20%）。"""
    value_cagr = _finite(cagr, "portfolio.cagr")
    value_mdd = _loss(mdd_loss, "portfolio.mdd_loss")
    cagr_ok = value_cagr > float(spec.TARGETS["portfolio_cagr_min"])
    mdd_ok = value_mdd < float(spec.TARGETS["portfolio_mdd_loss_max"])
    return {
        "gate": "portfolio_single_capital",
        "capital": capital,
        "inputs": {"cagr": value_cagr, "mdd_loss": value_mdd},
        "booleans": {"cagr_gt_25pct": cagr_ok, "mdd_loss_lt_20pct": mdd_ok},
        "passed": bool(cagr_ok and mdd_ok),
        "rules": {"cagr": "> 0.25（恰好 25% 不达标）", "mdd_loss": "< 0.20（恰好 20% 不达标）"},
    }


def _strict_bool(value: Any, label: str) -> bool:
    """只接受真正的布尔值；``bool("False") is True`` 这类转换一律拒绝。"""
    if isinstance(value, bool):
        return value
    raise GateError(f"{label} 不是布尔值（{value!r}）：禁止把字符串/数值当真假")


def portfolio_conjunction(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """两本金档合取（协议 §1：两档合取，任一无原始通过即不通过）。

    只接受**布尔** ``passed``；字符串 ``"False"`` 必须被拒绝而不是判真。
    更稳的用法是 :func:`portfolio_from_metrics`（由原始指标重算）。
    """
    missing = [
        capital
        for capital in spec.TARGETS["portfolio_capital_scenarios"]
        if str(capital) not in results
    ]
    if missing:
        raise GateError(f"组合门缺本金档：{missing}")
    booleans = {
        f"{capital}_passed": _strict_bool(results[str(capital)]["passed"], f"{capital}.passed")
        for capital in spec.TARGETS["portfolio_capital_scenarios"]
    }
    return {
        "gate": "portfolio_conjunction",
        "booleans": booleans,
        "passed": all(booleans.values()),
        "rule": "两档同时满足；单档通过不作整体通过",
    }


def portfolio_from_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """从原始指标重算两档组合门（不信任上游布尔值）。"""
    per_capital = {}
    for capital in spec.TARGETS["portfolio_capital_scenarios"]:
        key = str(capital)
        if key not in metrics:
            raise GateError(f"组合门缺本金档指标：{key}")
        item = metrics[key]
        per_capital[key] = portfolio_gate(item.get("cagr"), item.get("mdd_loss"), capital)
    conjunction = portfolio_conjunction(per_capital)
    return {
        "gate": "portfolio_from_metrics",
        "per_capital": per_capital,
        "conjunction": conjunction,
        "passed": conjunction["passed"],
        "rule": "由 cagr/mdd_loss 原始值逐档重算，再取合取",
    }


def validate_nav_frame(
    frame: Any,
    label: str = "nav",
    expected_days: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """净值帧合法性：日期必备且唯一升序、NAV 有限为正、可选严格日期轴。"""
    import pandas as pd

    if frame is None or len(frame) == 0:
        raise GateError(f"{label} 帧为空，禁止通过")
    if "nav" not in getattr(frame, "columns", []):
        raise GateError(f"{label} 帧缺 nav 列")
    if "trade_date" not in getattr(frame, "columns", []):
        raise GateError(f"{label} 帧缺 trade_date 列：无法校验日期轴")
    if len(frame) < 2:
        raise GateError(f"{label} 帧只有 {len(frame)} 个点，不足以构成收益区间")

    dates = [str(value) for value in frame["trade_date"].tolist()]
    if len(set(dates)) != len(dates):
        duplicated = sorted({day for day in dates if dates.count(day) > 1})
        raise GateError(f"{label} 帧存在重复日期：{duplicated[:5]}")
    if dates != sorted(dates):
        raise GateError(f"{label} 帧日期未升序")

    nav = pd.to_numeric(frame["nav"], errors="coerce")
    numeric = nav.to_numpy(dtype=float)
    if any(not math.isfinite(value) for value in numeric):
        raise GateError(f"{label} 帧 NAV 存在非有限值（NaN/±inf）")
    if (nav <= 0).any():
        raise GateError(f"{label} 帧 NAV 存在非正值：{int((nav <= 0).sum())} 行")

    if expected_days is not None:
        expected = [str(day) for day in expected_days]
        if dates != expected:
            raise GateError(
                f"{label} 帧日期轴与冻结轴不符：got={len(dates)} expected={len(expected)}，"
                f"首个差异位置={_first_diff(dates, expected)}"
            )
    return {"rows": int(len(frame)), "first": dates[0], "last": dates[-1]}


def _first_diff(left: Sequence[str], right: Sequence[str]) -> str:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return f"{index}:{a}!={b}"
    return f"length {len(left)} vs {len(right)}"


def align_axes(
    frames: Mapping[str, Sequence[str]],
    expected_days: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """多腿/多臂日期轴必须一致（可选严格等于冻结轴）。"""
    axes = {name: [str(day) for day in days] for name, days in frames.items()}
    if not axes:
        raise GateError("无可对齐的日期轴")
    reference_name = sorted(axes)[0]
    reference = axes[reference_name]
    if len(reference) < 2:
        raise GateError(f"日期轴只有 {len(reference)} 个点（基准 {reference_name}），禁止通过")
    mismatched = {
        name: {
            "rows": len(days),
            "first": days[0] if days else None,
            "last": days[-1] if days else None,
        }
        for name, days in axes.items()
        if days != reference
    }
    if mismatched:
        raise GateError(f"日期轴不一致（基准 {reference_name}）：{mismatched}")
    if expected_days is not None:
        expected = [str(day) for day in expected_days]
        if reference != expected:
            raise GateError(
                f"日期轴与冻结轴不符：got={len(reference)} expected={len(expected)}，"
                f"首个差异位置={_first_diff(reference, expected)}"
            )
    return {"rows": len(reference), "first": reference[0], "last": reference[-1]}


__all__ = [
    "GateError",
    "align_axes",
    "mdd_loss_from_raw",
    "portfolio_conjunction",
    "portfolio_gate",
    "r_gate",
    "validate_nav_frame",
]
