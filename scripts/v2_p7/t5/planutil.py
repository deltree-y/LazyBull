# -*- coding: utf-8 -*-
"""P7 T5 计划层纯函数：预净额/最终净额的腿级签名，与执行期保留的归因拆分。

与 ``host`` 解耦以便单测与文件规模控制；不持有状态、无副作用。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from scripts.v2_p7.t4.state import stable_digest


def plan_cash(cash_open: Mapping[str, float], delta: Mapping[str, float], sid: str) -> float:
    """计划期现金（期初 + 本日有符号转账）。"""
    return float(cash_open.get(sid, 0.0)) + float(delta.get(sid, 0.0))


def plan_credit(credit_open: Mapping[str, float], delta: Mapping[str, float], sid: str) -> float:
    """计划期 credit（期初 + 本日有符号 ``L``）。"""
    return float(credit_open.get(sid, 0.0)) + float(delta.get(sid, 0.0))


def _plan_signature(plan: Any) -> Dict[str, Any]:
    """净额计划的腿级签名（预净额与最终净额逐腿对照用）。"""
    scenarios: Dict[str, int] = {}
    for group in plan.groups:
        scenarios[group.scenario] = scenarios.get(group.scenario, 0) + 1
    signature = {
        "groups": len(plan.groups),
        "offset_legs": int(sum(len(group.offsets) for group in plan.groups)),
        "offset_shares": int(plan.offset_shares()),
        "market_legs": int(sum(1 for group in plan.groups if group.market is not None)),
        "market_shares": int(plan.market_shares()),
        "scenarios": dict(sorted(scenarios.items())),
    }
    signature["digest"] = stable_digest(signature)
    return signature


def _hold_capped_notional(
    plan: Any, trades: Sequence[Mapping[str, Any]], hold: Mapping[str, float]
) -> Dict[str, float]:
    """出借方执行期保留造成的买入截断（授权名义 − 实际成交名义，按袖）。"""
    out: Dict[str, float] = {}
    for sid in sorted(hold):
        if float(hold[sid]) <= 0.0:
            continue
        decision = plan.decisions.get(sid)
        authorized = float(decision.authorized_notional) if decision is not None else 0.0
        filled = 0.0
        for trade in trades:
            if trade.get("kind") != "market_buy":
                continue
            shares = float((trade.get("sleeves") or {}).get(sid, 0) or 0)
            filled += shares * float(trade.get("price") or 0.0)
        out[sid] = float(max(0.0, authorized - filled))
    return out


def _conservation_check(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    from scripts.v2_p7.t4.netting import assert_intent_conservation

    return assert_intent_conservation(rows)


def _cross_day(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    from scripts.v2_p7.t4.netting import cross_day_conservation

    return cross_day_conservation(rows)


def _per_sleeve_min(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in rows:
        sleeve = str(row["sleeve"])
        value = float(row["cash_plus_credit"])
        current = out.get(sleeve)
        out[sleeve] = value if current is None else min(current, value)
    return dict(sorted(out.items()))
