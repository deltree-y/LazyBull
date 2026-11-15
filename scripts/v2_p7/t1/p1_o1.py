# -*- coding: utf-8 -*-
"""P-1 重开 O1：约束账户路径读数（只报不判，无任何自动否决）。

约束：本金 300,000 / 6 槽 / 单笔 ≥ min_ticket / 现金非负 / 同股抑制。
读数（R05 修正后的**实际账户口径**）：

- **R1** 无约束资金需求 = 意图数 × min_ticket；
- **R2** 日均占用率 = 逐日 Σ 已建仓本金 / 本金 的均值；
- **R3** 超期锁定 p95 = **逐日汇总**「持有天数 > 2×持有期」仓位市值 / 本金的 95 分位
  （不是逐笔固定 min_ticket 的分位；市值按该意图事件窗口净收益 ``event_net`` 计，
  逐股逐日不重估，属已登记近似）；
- **R4** 退出延迟 p95（未平仓按已观察时长记删失下界）；
- **R5** 终局损失压力 = 截止**仍未平仓**仓位本金 / 本金（全部按 0 估值）。

R05 修正（2026-11-15，P7 整体检视）：
1. 回款用**实际** ``event_net``（不再用包络中点 ``d_point``）；
2. 建仓按**实际** ``entry_pos``（不再用 ``signal_pos+1``）；
3. 释放必须 ``holding == HOLDING_CLOSED``；``HOLDING_OPEN`` 截止不释放、单列终局；
4. R3 改逐日汇总市值/本金分位。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from scripts.v2_p7.t1.p1_bounds import (
    FILL_CONFIRMED,
    FILL_POSSIBLE,
    HOLDING_CLOSED,
    HOLDING_OPEN,
)
from scripts.v2_p7.t1.p1_params import P1RParams

#: 超期倍数（持有天数 > 该倍数 × 持有期即记超期锁定；沿用预登记 2 倍口径）
OVERDUE_HORIZON_MULTIPLE = 2.0


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mark(capital: float, ret: float) -> float:
    """仓位市值近似 = 本金 × (1 + 事件窗口净收益)；非有限值时退回本金。"""
    if not np.isfinite(ret):
        return float(capital)
    return float(capital) * (1.0 + float(ret))


def _release_closed(
    positions: Dict[int, Dict[str, Any]], cash: float, day: int
) -> float:
    """释放当日及之前**已平仓**（``holding=CLOSED``）的持仓；未平仓永不释放。"""
    for stock in [
        key
        for key, item in positions.items()
        if item["closed"] and int(item["exit"]) <= int(day)
    ]:
        item = positions.pop(stock)
        cash += float(item["capital"]) * (1.0 + float(item["ret"]))
    return cash


def _process_day(
    day: int,
    by_entry: Dict[int, List[int]],
    intents: Any,
    positions: Dict[int, Dict[str, Any]],
    cash: float,
    *,
    slots: int,
    min_ticket: float,
    result: Any,
) -> Tuple[float, Dict[str, int]]:
    """处理一个交易日的**实际入场**（``entry_pos``）计划，返回现金与计数。"""
    counters = {"bought": 0, "suppressed": 0, "no_slot": 0, "cash_blocked": 0}
    holding = np.asarray(result.holding)
    exit_pos = np.asarray(result.exit_pos)
    event_net = np.asarray(result.event_net, dtype=np.float64)
    for index in by_entry.get(day, []):
        stock = int(intents.stock[index])
        if stock in positions:
            counters["suppressed"] += 1
            continue
        if len(positions) >= slots:
            counters["no_slot"] += 1
            continue
        if cash < min_ticket:
            counters["cash_blocked"] += 1
            continue
        observed = float(event_net[index]) if np.isfinite(event_net[index]) else 0.0
        positions[stock] = {
            "entry": int(day),
            "exit": int(exit_pos[index]),
            "capital": float(min_ticket),
            "ret": observed,
            "closed": int(holding[index]) == HOLDING_CLOSED,
        }
        cash -= min_ticket
        counters["bought"] += 1
    return cash, counters


def constraint_readings(
    intents: Any,
    result: Any,
    params: P1RParams,
    *,
    cutoff_pos: int,
) -> Dict[str, Any]:
    """逐日推进约束账户，输出五组只读读数与参考线标记。"""
    n = intents.size
    initial = float(params.initial_capital)
    min_ticket = float(params.min_ticket)
    slots = int(params.max_concurrent_positions)
    horizon = int(params.horizon_primary)

    fill = np.asarray(result.fill)
    holding = np.asarray(result.holding)
    entry_pos = np.asarray(result.entry_pos, dtype=np.int64)

    confirmed = fill == FILL_CONFIRMED
    possible = fill == FILL_POSSIBLE
    counts = {
        "intent_total": int(n),
        "filled_confirmed": int(confirmed.sum()),
        "possible_not_filled": int(possible.sum()),
        "not_filled": int((~confirmed & ~possible).sum()),
        "holding_open_intents": int((confirmed & (holding == HOLDING_OPEN)).sum()),
        "holding_closed_intents": int((confirmed & (holding == HOLDING_CLOSED)).sum()),
        "holding_na_intents": int((~(confirmed)).sum()),
    }

    # 建仓日 = **实际** ``entry_pos``（R05 修正②）；顺序 = (实际入场日, 源, 股票)
    planned = np.where(confirmed & (entry_pos >= 0), entry_pos, np.iinfo(np.int64).max)
    order = np.lexsort(
        (np.asarray(intents.stock), np.asarray(intents.source_code), planned)
    )
    by_entry: Dict[int, List[int]] = {}
    for index in order.tolist():
        if not bool(confirmed[index]) or int(entry_pos[index]) < 0:
            continue
        by_entry.setdefault(int(entry_pos[index]), []).append(int(index))

    cash = initial
    positions: Dict[int, Dict[str, Any]] = {}
    occupied_series: List[float] = []
    overdue_series: List[float] = []
    totals = {"bought": 0, "suppressed": 0, "no_slot": 0, "cash_blocked": 0}

    for day in range(0, int(cutoff_pos) + 1):
        cash = _release_closed(positions, cash, day)
        cash, counters = _process_day(
            day,
            by_entry,
            intents,
            positions,
            cash,
            slots=slots,
            min_ticket=min_ticket,
            result=result,
        )
        for key in totals:
            totals[key] += counters[key]
        occupied = float(sum(float(item["capital"]) for item in positions.values()))
        occupied_series.append(occupied / initial if initial else 0.0)
        overdue = float(
            sum(
                _mark(float(item["capital"]), float(item["ret"]))
                for item in positions.values()
                if float(day - int(item["entry"])) > OVERDUE_HORIZON_MULTIPLE * horizon
            )
        )
        overdue_series.append(overdue / initial if initial else 0.0)

    open_capital = float(
        sum(float(item["capital"]) for item in positions.values() if not item["closed"])
    )
    closed_capital = float(
        sum(float(item["capital"]) for item in positions.values() if item["closed"])
    )
    delays = _path_level_delays(result, cutoff_pos, horizon)

    r1 = n * min_ticket
    r2 = float(np.mean(occupied_series)) if occupied_series else 0.0
    r3 = _percentile(overdue_series, 95)
    r4 = _percentile([float(value) for value in delays], 95)
    open_share = open_capital / initial if initial else 0.0
    reference = {
        "R2_mean_capital_utilization": 0.90,
        "R3_overdue_lock_share": 0.10,
        "R4_exit_delay_days": 20.0,
        "R5_terminal_loss_stress": 0.25,
    }
    flags = {
        "R2_above_reference": bool(r2 > reference["R2_mean_capital_utilization"]),
        "R3_above_reference": bool(r3 is not None and r3 > reference["R3_overdue_lock_share"]),
        "R4_above_reference": bool(r4 is not None and r4 > reference["R4_exit_delay_days"]),
        "R5_above_reference": bool(open_share > reference["R5_terminal_loss_stress"]),
    }
    return {
        "schema": "p7_p1r_o1_readings/v2",
        "decision_role": "readings_only",
        "auto_veto": False,
        "initial_capital": initial,
        "max_concurrent_positions": slots,
        "min_ticket": min_ticket,
        "constraint": (
            "本金固定 / 6 槽 / 现金非负 / 同股抑制 / 按实际 entry_pos 建仓 / "
            "仅 holding=CLOSED 释放 / 未平仓持有到截止"
        ),
        "R1_uncapped_capital_demand": float(r1),
        "R1_reference": None,
        "R2_mean_capital_utilization": float(r2),
        "R2_days": int(len(occupied_series)),
        "R3_overdue_lock_share_p95": r3,
        "R3_definition": (
            "逐日 Σ(持有天数 > 2×持有期 仓位市值) / 本金，取 95 分位；市值按 event_net 计"
        ),
        "R3_overdue_days": int(sum(1 for value in overdue_series if value > 0.0)),
        "R4_exit_delay_p95_days": r4,
        "R4_note": "未平仓按已观察时长记删失下界，不等同实际退出延迟",
        "R5_terminal_loss_stress": float(open_share),
        "R5_note": "截止仍未平仓仓位本金 / 本金（全部按 0 估值）；已平仓不进入",
        "R5_open_capital": open_capital,
        "R5_closed_unreleased_capital": closed_capital,
        "counts": {
            **counts,
            "bought": totals["bought"],
            "suppressed_active_position": totals["suppressed"],
            "blocked_no_slot": totals["no_slot"],
            "blocked_cash": totals["cash_blocked"],
            "open_at_cutoff": int(
                sum(1 for item in positions.values() if not item["closed"])
            ),
        },
        "cash_return_basis": "event_net",
        "entry_basis": "entry_pos",
        "reference_lines": reference,
        "flags": flags,
        "no_automatic_conclusion": True,
    }


def _path_level_delays(result: Any, cutoff_pos: int, horizon: int) -> List[int]:
    """路径级退出延迟（R4）：已平仓取实际延迟，未平仓取已观察时长下界。"""
    fill = np.asarray(result.fill)
    holding = np.asarray(result.holding)
    entry = np.asarray(result.entry_pos, dtype=np.int64)
    exit_ = np.asarray(result.exit_pos, dtype=np.int64)
    confirmed = (fill == FILL_CONFIRMED) & (entry >= 0)
    delays: List[int] = []
    for index in np.nonzero(confirmed)[0].tolist():
        due = int(entry[index]) + int(horizon)
        if int(holding[index]) == HOLDING_CLOSED:
            delays.append(int(exit_[index]) - due)
        else:
            delays.append(max(0, int(cutoff_pos) - due))
    return delays
