# -*- coding: utf-8 -*-
"""P7 T1 共同入离场执行、费用与总回报经济价值（设计 §6.2~§6.3）。

冻结口径
========
- 计划入场 = 信号日 T 的下一交易日（T+1）；只有六股在同一交易日**都**满足
  v2 T+1 开盘买入条件（真实开盘价、非停牌、非涨停）才共同入场，否则整组共同顺延，
  最长 10 个交易日；10 日内无共同入场日 ⇒ 整组 `unfilled_group_count`，收益与费用为真实 0。
- 到期日 = 共同入场后第 h 个交易日；只有六股在同一日都满足卖出条件（真实开盘价、非停牌、
  非跌停）才共同离场，否则共同顺延，最长 10 日；仍无 ⇒ `endpoint_unresolved`（进
  `unresolved_count`，不补 0）。
- 费用只消费**真实未复权成交金额**；总回报用复权开盘价之比承载持有期经济回报，
  毛/净收益共用同一 ``V_x``；不得把复权金额送入成本模型。
- 单股预算 = ``V + 100P + 预估买入费用``，再按 100 股整手向下（冻结的最低 ticket 语义）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from scripts.v2_p7.t1.dataplane import share_chain_known, share_ratio

STATUS_RESOLVED = 0
STATUS_UNFILLED = 1
STATUS_UNRESOLVED = 2

STATUS_NAMES: Dict[int, str] = {
    STATUS_RESOLVED: "resolved_filled",
    STATUS_UNFILLED: "unfilled",
    STATUS_UNRESOLVED: "unresolved",
}


@dataclass
class HorizonResult:
    """某个持期 h 下的逐组执行与收益结果。"""

    horizon: int
    status: np.ndarray
    reason: List[str]
    entry_pos: np.ndarray
    exit_pos: np.ndarray
    entry_defer: np.ndarray
    exit_defer: np.ndarray
    gross: np.ndarray
    net: np.ndarray
    shares: np.ndarray
    sell_shares: np.ndarray
    buy_amount: np.ndarray
    sell_amount: np.ndarray
    buy_cost: np.ndarray
    sell_cost: np.ndarray
    residual: np.ndarray
    gross_residual: np.ndarray
    event_net: np.ndarray
    control_net: np.ndarray
    stock_entry_ok: np.ndarray
    stock_exit_ok: np.ndarray
    entry_gap: np.ndarray
    exit_gap: np.ndarray
    one_price_entry: np.ndarray
    one_price_exit: np.ndarray

    @property
    def size(self) -> int:
        return int(self.status.size)

    def counts(self) -> Dict[str, int]:
        return {
            "intent_total": int(self.size),
            "resolved_filled_count": int((self.status == STATUS_RESOLVED).sum()),
            "unfilled_group_count": int((self.status == STATUS_UNFILLED).sum()),
            "unresolved_count": int((self.status == STATUS_UNRESOLVED).sum()),
        }


def buy_conditions(panel: Any, positions: np.ndarray, members: np.ndarray) -> np.ndarray:
    """六股共同买入条件（真实开盘价 + 非停牌 + 非涨停）。

    **只判交易可执行性**：复权链完整性属于收益可解析性，由 `return_chain_ok` 单独判定，
    不得把复权缺失当作买不进（T1 评审 R3）。
    """
    clipped = np.clip(positions, 0, panel.n_days - 1)
    safe = np.where(members >= 0, members, 0)
    raw = panel.open_u[clipped[:, None], safe]
    price_ok = np.isfinite(raw) & (raw > 0)
    tradable = (~panel.suspended[clipped[:, None], safe]) & (
        ~panel.limit_up[clipped[:, None], safe]
    )
    return price_ok & tradable


def sell_conditions(panel: Any, positions: np.ndarray, members: np.ndarray) -> np.ndarray:
    """六股共同卖出条件（真实开盘价 + 非停牌 + 非跌停）。"""
    clipped = np.clip(positions, 0, panel.n_days - 1)
    safe = np.where(members >= 0, members, 0)
    raw = panel.open_u[clipped[:, None], safe]
    price_ok = np.isfinite(raw) & (raw > 0)
    tradable = (~panel.suspended[clipped[:, None], safe]) & (
        ~panel.limit_down[clipped[:, None], safe]
    )
    return price_ok & tradable


def _first_common_day(
    panel: Any,
    planned: np.ndarray,
    members: np.ndarray,
    pending: np.ndarray,
    *,
    max_defer: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """共同顺延搜索：返回 (位置, 顺延天数, 逐股在该窗口内是否可成交)。"""
    n_groups = int(planned.size)
    position = np.full(n_groups, -1, dtype=np.int64)
    defer = np.full(n_groups, -1, dtype=np.int16)
    stock_ok_any = np.zeros_like(members, dtype=bool)
    active = pending.copy()
    last = panel.n_days - 1
    for step in range(max_defer + 1):
        if not active.any():
            break
        candidate = planned + step
        inb = candidate <= last
        conditions = buy_conditions if mode == "buy" else sell_conditions
        ok = np.zeros(n_groups, dtype=bool)
        if inb.any():
            ok[inb] = conditions(panel, candidate[inb], members[inb]).all(axis=1)
        stock_conditions = conditions(panel, np.clip(candidate, 0, last), members)
        stock_ok_any |= stock_conditions & inb[:, None]
        newly = active & ok
        position[newly] = candidate[newly]
        defer[newly] = step
        active &= ~newly
    return position, defer, stock_ok_any


def _lot_and_cost(
    price: np.ndarray,
    *,
    min_ticket: float,
    lot_size: int,
    cost: Dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """整手股数与买入金额（预算 = V + 100P + 预估买入费用）。"""
    ratio = float(lot_size)
    budget = min_ticket + ratio * price
    estimate = np.maximum(budget * cost["commission_rate"], cost["min_commission"])
    estimate = estimate + budget * cost["slippage"]
    budget = budget + estimate
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.floor(budget / price / ratio) * ratio
    shares = np.where(np.isfinite(shares) & (shares > 0), shares, 0.0)
    return shares.astype(np.int64), shares * price


def _buy_cost(amount: np.ndarray, cost: Dict[str, float]) -> np.ndarray:
    commission = np.maximum(amount * cost["commission_rate"], cost["min_commission"])
    return commission + amount * cost["slippage"]


def _sell_cost(amount: np.ndarray, cost: Dict[str, float]) -> np.ndarray:
    commission = np.maximum(amount * cost["commission_rate"], cost["min_commission"])
    return commission + amount * cost["stamp_tax"] + amount * cost["slippage"]


def _row_nanmean(values: np.ndarray) -> np.ndarray:
    """按行求 NaN 均值（无有效值时返回 NaN，不产生空切片告警）。"""
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    totals = np.where(finite, values, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def simulate_horizon(
    panel: Any,
    members: np.ndarray,
    signal_pos: np.ndarray,
    pre_status: np.ndarray,
    pre_reason: List[str],
    *,
    horizon: int,
    min_ticket: float,
    cost: Dict[str, float],
    lot_size: int = 100,
    entry_max_defer: int = 10,
    exit_max_defer: int = 10,
) -> HorizonResult:
    """对全部意图组执行共同入离场并还原扣成本总回报。"""
    n_groups, width = members.shape
    valid = (members >= 0).all(axis=1) & (pre_status == 0)
    planned = signal_pos + 1

    entry_pos, entry_defer, stock_entry_ok = _first_common_day(
        panel, planned, members, valid, max_defer=entry_max_defer, mode="buy"
    )
    stock_entry_ok &= valid[:, None]
    expiry = np.where(entry_pos >= 0, entry_pos + horizon, -1)
    exit_pending = entry_pos >= 0
    exit_pos, exit_defer, stock_exit_ok = _first_common_day(
        panel,
        np.where(exit_pending, expiry, 0),
        members,
        exit_pending,
        max_defer=exit_max_defer,
        mode="sell",
    )
    stock_exit_ok &= exit_pending[:, None]

    last = panel.n_days - 1
    safe = np.where(members >= 0, members, 0)
    entry_clip = np.clip(entry_pos, 0, last)
    exit_clip = np.clip(exit_pos, 0, last)
    price_buy = panel.open_u[entry_clip[:, None], safe]
    price_sell = panel.open_u[exit_clip[:, None], safe]
    adj_buy = panel.open_adj[entry_clip[:, None], safe]
    adj_sell = panel.open_adj[exit_clip[:, None], safe]
    ratio = share_ratio(panel, entry_clip[:, None], exit_clip[:, None], safe)
    chain_ok = share_chain_known(panel, entry_clip[:, None], exit_clip[:, None], safe)
    adj_ok = np.isfinite(adj_buy) & (adj_buy > 0) & np.isfinite(adj_sell) & (adj_sell > 0)
    return_chain_ok = adj_ok & chain_ok & np.isfinite(ratio) & (ratio > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        shares, buy_amount = _lot_and_cost(
            price_buy, min_ticket=min_ticket, lot_size=lot_size, cost=cost
        )
        sell_shares = np.floor(shares * ratio)
        sell_amount = sell_shares * price_sell
        buy_cost = _buy_cost(buy_amount, cost)
        sell_cost = _sell_cost(sell_amount, cost)
        value_exit = buy_amount * (adj_sell / adj_buy)
        gross = value_exit / buy_amount - 1.0
        net = (value_exit - sell_cost) / (buy_amount + buy_cost) - 1.0

    filled = (
        (entry_pos >= 0)
        & (exit_pos >= 0)
        & return_chain_ok.all(axis=1)
        & (shares > 0).all(axis=1)
        & np.isfinite(net).all(axis=1)
    )
    zero_lot = (entry_pos >= 0) & (exit_pos >= 0) & (shares == 0).any(axis=1)

    status = np.full(n_groups, STATUS_UNRESOLVED, dtype=np.int8)
    reason = list(pre_reason)
    for index in range(n_groups):
        if not valid[index]:
            status[index] = STATUS_UNRESOLVED
            continue
        if entry_pos[index] < 0:
            status[index] = STATUS_UNFILLED
            reason[index] = "no_common_entry_within_10d"
            continue
        if filled[index]:
            status[index] = STATUS_RESOLVED
            reason[index] = "resolved"
        elif zero_lot[index]:
            status[index] = STATUS_UNRESOLVED
            reason[index] = "zero_lot_budget"
        elif exit_pos[index] < 0:
            status[index] = STATUS_UNRESOLVED
            reason[index] = "endpoint_unresolved"
        else:
            status[index] = STATUS_UNRESOLVED
            reason[index] = "return_chain_unresolved"

    gross_clean = np.where(filled[:, None], gross, np.nan)
    net_clean = np.where(filled[:, None], net, np.nan)
    event_net = net_clean[:, 0]
    control_net = _row_nanmean(net_clean[:, 1:])
    residual = event_net - control_net
    event_gross = gross_clean[:, 0]
    control_gross = _row_nanmean(gross_clean[:, 1:])
    gross_residual = event_gross - control_gross

    with np.errstate(invalid="ignore", divide="ignore"):
        entry_gap = np.where(
            filled[:, None],
            panel.open_u[entry_clip[:, None], safe] / panel.pre_close[entry_clip[:, None], safe]
            - 1.0,
            np.nan,
        )
        exit_gap = np.where(
            filled[:, None],
            panel.open_u[exit_clip[:, None], safe] / panel.pre_close[exit_clip[:, None], safe]
            - 1.0,
            np.nan,
        )
    one_price_entry = np.zeros((n_groups, width), dtype=bool)
    one_price_exit = np.zeros((n_groups, width), dtype=bool)
    price_range_entry = panel.high[entry_clip[:, None], safe] - panel.low[entry_clip[:, None], safe]
    price_range_exit = panel.high[exit_clip[:, None], safe] - panel.low[exit_clip[:, None], safe]
    one_price_entry[filled] = price_range_entry[filled] <= 1e-9
    one_price_exit[filled] = price_range_exit[filled] <= 1e-9

    return HorizonResult(
        horizon=int(horizon),
        status=status,
        reason=reason,
        entry_pos=entry_pos.astype(np.int32),
        exit_pos=exit_pos.astype(np.int32),
        entry_defer=entry_defer,
        exit_defer=exit_defer,
        gross=gross_clean,
        net=net_clean,
        shares=shares,
        sell_shares=sell_shares,
        buy_amount=np.where(filled[:, None], buy_amount, np.nan),
        sell_amount=np.where(filled[:, None], sell_amount, np.nan),
        buy_cost=np.where(filled[:, None], buy_cost, np.nan),
        sell_cost=np.where(filled[:, None], sell_cost, np.nan),
        residual=residual,
        gross_residual=gross_residual,
        event_net=event_net,
        control_net=control_net,
        stock_entry_ok=stock_entry_ok,
        stock_exit_ok=stock_exit_ok,
        entry_gap=entry_gap,
        exit_gap=exit_gap,
        one_price_entry=one_price_entry,
        one_price_exit=one_price_exit,
    )


def intent_residual(result: HorizonResult) -> np.ndarray:
    """全部意图的净残差：已成交用真实值，未成交整组取真实 0，不可解析取 NaN。"""
    values = np.full(result.size, np.nan, dtype=np.float64)
    resolved = result.status == STATUS_RESOLVED
    unfilled = result.status == STATUS_UNFILLED
    values[resolved] = result.residual[resolved]
    values[unfilled] = 0.0
    return values


def execute_all_cells(
    panel: Any,
    tables: Dict[tuple, Any],
    params: Any,
    *,
    cost: Dict[str, float],
) -> Dict[tuple, HorizonResult]:
    """对每个 (source, horizon) 单元执行共同入离场（表按单元构造，观察窗随 h 变化）。"""
    results: Dict[tuple, HorizonResult] = {}
    for source, horizon in params.cell_keys():
        table = tables[(source, int(horizon))]
        results[(source, int(horizon))] = simulate_horizon(
            panel,
            table.members,
            table.signal_pos,
            table.status,
            table.reason,
            horizon=int(horizon),
            min_ticket=params.min_ticket,
            cost=cost,
            lot_size=100,
            entry_max_defer=10,
            exit_max_defer=10,
        )
    return results
