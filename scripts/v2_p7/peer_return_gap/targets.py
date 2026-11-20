# -*- coding: utf-8 -*-
"""月末目标与最小买入核算（协议 §5）；费用口径复用公共 CostModel。

# 复杂度豁免：P7 候选一阶段专用研究工具，不入生产链路（AGENTS §7.5）。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from src.lazybull.common.cost import CostModel

from . import spec


def cost_model() -> CostModel:
    return CostModel(
        commission_rate=spec.COMMISSION_RATE,
        min_commission=spec.MIN_COMMISSION,
        stamp_tax=spec.STAMP_TAX,
        slippage=spec.SLIPPAGE,
    )


def account_params(capital: int) -> dict[str, Any]:
    """单个 B 独立账户的冻结参数；两档本金各走一次真实预算。"""
    return {
        "capital": int(capital),
        "b_capital": float(capital) * spec.B_SHARE,
        "b_share": spec.B_SHARE,
        "slots": spec.SLOTS,
        "sector_cap": spec.SECTOR_CAP,
        "commission_rate": spec.COMMISSION_RATE,
        "min_commission": spec.MIN_COMMISSION,
        "stamp_tax": spec.STAMP_TAX,
        "slippage": spec.SLIPPAGE,
        "min_notional": round(spec.MIN_NOTIONAL, 6),
        "lot_size": spec.LOT_SIZE,
    }


def unit_value(total_value: float) -> float:
    """``V_T = 当日 B 总资产/6``（协议 §5.1）。"""
    return float(total_value) / float(spec.SLOTS)


def affordable_shares(price: float, budget: float) -> tuple[int, float]:
    """先预留滑点与佣金，再整手向下取整；返回 (股数, 名义额)。"""
    if not math.isfinite(price) or price <= 0 or budget <= 0:
        return 0, 0.0
    model = cost_model()
    gross = budget / (1.0 + spec.SLIPPAGE + spec.COMMISSION_RATE)
    shares = int(gross // (price * spec.LOT_SIZE)) * spec.LOT_SIZE
    while shares > 0:
        notional = shares * price
        cost = model.calculate_buy_cost(notional)
        if cost <= budget and notional >= spec.MIN_NOTIONAL:
            return shares, notional
        shares -= spec.LOT_SIZE
    return 0, 0.0


def plan_buys(
    *,
    targets: Sequence[Mapping[str, Any]],
    prices: Mapping[str, float],
    total_value: float,
    available_cash: float,
    held: Mapping[str, Any],
    industry: Mapping[str, str],
) -> list[dict[str, Any]]:
    """T+1 收盘按 T 顺序尝试买入；预算与槽位在当日实际操作中递减。

    留任股票不补差价；槽位/行业额度不预支；买单当日失效。
    """
    value = unit_value(total_value)
    cash = float(available_cash)
    codes = list(held)
    sectors: dict[str, int] = {}
    for code in codes:
        sector = industry.get(code)
        if sector:
            sectors[sector] = sectors.get(sector, 0) + 1
    plans: list[dict[str, Any]] = []
    for order, row in enumerate(targets, start=1):
        code = str(row["ts_code"])
        sector = industry.get(code)
        price = float(prices.get(code, float("nan")))
        record = {
            "slot": order,
            "ts_code": code,
            "sector": sector,
            "budget": round(min(value, cash), 6),
            "price": price,
            "shares": 0,
            "notional": 0.0,
            "reason": "",
        }
        if code in codes:
            record["reason"] = "already_held"
            plans.append(record)
            continue
        if sector is None:
            record["reason"] = "industry_unknown"
        elif len(codes) >= spec.SLOTS:
            record["reason"] = "slots_full"
        elif sectors.get(sector, 0) >= spec.SECTOR_CAP:
            record["reason"] = "sector_full"
        elif not math.isfinite(price) or price <= 0:
            record["reason"] = "no_price"
        else:
            shares, notional = affordable_shares(price, min(value, cash))
            if notional < spec.MIN_NOTIONAL:
                record["reason"] = "below_min_notional"
            elif notional > cash:
                record["reason"] = "cash_insufficient"
            else:
                record.update({"shares": shares, "notional": round(notional, 6)})
                record["reason"] = "buy"
                cash -= cost_model().calculate_buy_cost(notional)
                codes.append(code)
                sectors[sector] = sectors.get(sector, 0) + 1
        plans.append(record)
    return plans
