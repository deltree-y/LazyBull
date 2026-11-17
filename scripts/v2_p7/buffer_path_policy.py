"""固定 Top20 到期续持政策：研究账户纯函数，不接生产、不构造收益。

每次到期仍属当期合法 Top20 则原股数续持 20 交易日，实际买入日不变。
续持占原批次槽位及预算；本模块只给新买名义预算，含费现金约束由账户层执行。
复杂度豁免：阶段专用研究政策，不入生产链路（CLAUDE §7.8）。
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from typing import Any, Protocol

from scripts.v2_p7.t25.target_layer import PlannerKernel, select_targets, sleeve_weights
from src.lazybull.trading.sell_rules import is_holding_period_exit_due


class VarianceSource(Protocol):
    """只读截至信号日的方差入口，与既有 VarianceBook 兼容。"""

    def variance(self, code: str, day: str) -> float | None:
        """返回既有历史窗口方差，无法估计返回 None。"""
        ...


def _validated_ranked(ranked: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    """保留调用方完整合法排名顺序，拒绝重复股票和非有限分数。"""
    result = [(str(code), float(score)) for code, score in ranked]
    if len({code for code, _ in result}) != len(result):
        raise ValueError("当期合法排名含重复股票")
    if any(not code or not math.isfinite(score) for code, score in result):
        raise ValueError("当期合法排名含空股票或非有限分数")
    return result


def due_decision(
    code: str,
    holding_days: int,
    renewal_count: int,
    ranked: Sequence[tuple[str, float]],
    enabled: bool,
    active_signal: bool = True,
    forced_exit: bool = False,
) -> dict[str, Any]:
    """按唯一到期函数判断续持；不改实际买入日，不取消已有退出义务。

    ``renewal_count`` 是此前成功续持次数；调用方须在 ``renew=True`` 时一次性
    采用返回的新次数。已有卖单或风险退出均应传 ``forced_exit=True``，且账户层
    不得据此撤销卖单。到期只使用共享阈值，不在本模块重算日期差。
    """
    if int(holding_days) != holding_days or holding_days < 0:
        raise ValueError(f"实际持有交易日数非法：{holding_days}")
    if int(renewal_count) != renewal_count or renewal_count < 0:
        raise ValueError(f"续持次数非法：{renewal_count}")
    due = bool(is_holding_period_exit_due(int(holding_days), 20 * (int(renewal_count) + 1)))
    renew = False
    if forced_exit:
        reason = "forced_exit"
    elif not due:
        reason = "not_due"
    elif not enabled:
        reason = "buffer_disabled"
    elif not active_signal:
        reason = "inactive_signal"
    else:
        current = _validated_ranked(ranked)
        if not current:
            reason = "missing_ranking"
        elif str(code) not in {stock for stock, _ in current[:20]}:
            reason = "outside_top20"
        else:
            renew, reason = True, "renew_top20"
    return {
        "due": due,
        "renew": renew,
        "next_renewal_count": int(renewal_count) + int(renew),
        "reason": reason,
    }


def plan_tranche(
    day: str,
    tranche_idx: int,
    ranked: Sequence[tuple[str, float]],
    held_codes: Collection[str],
    retained_codes: Collection[str],
    retained_market_value: float,
    nav: float,
    variance_book: VarianceSource | None,
) -> dict[str, Any]:
    """复用原候选选择与 half-Kelly，保留股占槽且新买预算只向下压缩。

    ``retained_codes`` 只含当前批次保留的实际持仓，``held_codes`` 含全部当前持仓；
    ``retained_market_value`` 是前者当前合计市值。保留股按当前排名顺序进入组合，
    缺当日分数者按代码升序置后、分数为 0，并显式登记；不补未来分数。
    返回 ``new_buys[].weight`` 是批内名义权重，``budget`` 已包含 50% NAV 和
    预算压缩，账户只按 budget 执行，不再乘 weight。函数不调整续持股数。
    """
    if tranche_idx not in (0, 1):
        raise ValueError(f"冻结政策只有批次 0/1：{tranche_idx}")
    nav, retained_market_value = float(nav), float(retained_market_value)
    if any(not math.isfinite(value) or value < 0 for value in (nav, retained_market_value)):
        raise ValueError(f"NAV 或保留持仓市值非法：nav={nav}, retained={retained_market_value}")
    current = _validated_ranked(ranked)
    held, retained = {str(code) for code in held_codes}, {str(code) for code in retained_codes}
    if not retained.issubset(held):
        raise ValueError("本批保留股票必须属于当前实际持仓")
    if len(retained) > 10:
        raise ValueError(f"本批保留股票超出 10 个槽位：{len(retained)}")
    scores = dict(current)
    order = {code: index for index, (code, _) in enumerate(current)}
    retained_order = sorted(retained, key=lambda code: (order.get(code, len(order)), code))
    missing_score = [code for code in retained_order if code not in scores]
    kernel = PlannerKernel()
    picks, selection = select_targets(kernel, current, day, tranche_idx, sorted(held))
    new_picks = picks[: 10 - len(retained)]
    combined = {code: scores.get(code, 0.0) for code in retained_order}
    combined.update(new_picks)
    weights = sleeve_weights(combined, day, variance_book)
    batch_budget = 0.5 * nav
    new_budget_cap = max(0.0, batch_budget - retained_market_value)
    nominal = {
        code: min(batch_budget * float(weights.get(code, 0.0)), 0.15 * nav) for code, _ in new_picks
    }
    nominal_total = sum(nominal.values())
    scale = min(1.0, new_budget_cap / nominal_total) if nominal_total > 0 else 1.0
    budgets = {code: amount * scale for code, amount in nominal.items()}
    buys = [
        {
            "ts_code": code,
            "score": score,
            "weight": float(weights.get(code, 0.0)),
            "budget": budgets[code],
            "effective_nav_weight": budgets[code] / nav if nav > 0 else 0.0,
        }
        for code, score in new_picks
    ]
    return {
        "day": day,
        "tranche_idx": tranche_idx,
        "new_buys": buys,
        "new_weights": {row["ts_code"]: row["weight"] for row in buys},
        "new_budgets": budgets,
        "composite_weights": weights,
        "composite_scores": combined,
        "retained_codes": retained_order,
        "retained_missing_score": missing_score,
        "retained_count": len(retained),
        "new_slot_count": len(new_picks),
        "unfilled_slot_count": 10 - len(retained) - len(new_picks),
        "retained_market_value": retained_market_value,
        "batch_budget": batch_budget,
        "new_budget_cap": new_budget_cap,
        "new_nominal_budget": nominal_total,
        "budget_scale": scale,
        "new_total_budget": sum(budgets.values()),
        "cashguard_applied": False,
        "selection": selection,
    }
