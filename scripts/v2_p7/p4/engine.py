# -*- coding: utf-8 -*-
"""P4 冻结算法执行：候选实际账户、同域等权毛基准与逐日审计。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Mapping

import numpy as np
import pandas as pd

from scripts.v2_p7.p2.account import ResearchAccount
from scripts.v2_p7.p2.types import (
    MarketSnapshot,
    PriceObservation,
    PriceStatus,
    money,
    price,
)
from scripts.v2_p7.p4.data import P4Data, mainboard_mask

COMMISSION_RATE = Decimal("0.0001954")
MIN_COMMISSION = Decimal("5")
STAMP_TAX = Decimal("0.0005")
SLIPPAGE = Decimal("0.0005")
ZERO = Decimal("0.00")


@dataclass(frozen=True, slots=True)
class BatchPlan:
    notice_id: str
    signal_date: str
    entry_date: str
    flow_date: str
    selected: tuple[dict, ...]
    selection_audit: tuple[dict, ...]
    membership: tuple[str, ...]
    adv20: Mapping[str, float]
    capacity_1pct: float | None
    capacity_5pct: float | None


@dataclass(slots=True)
class CandidateRun:
    label: str
    principal: float
    start_date: str
    end_date: str
    levels: pd.Series
    fills: tuple[dict, ...]
    exceptions: tuple[dict, ...]
    corporate_actions: tuple[dict, ...]
    missing_rows: tuple[dict, ...]
    final_positions: tuple[dict, ...]
    position_days: int
    missing_position_days: int
    nav_marks: dict[str, float]
    #: 逐日持仓市值 / NAV（E4-4 月资金利用率；无持仓日按 0 计入，不删零事件月）
    daily_utilization: tuple[dict, ...] = ()


@dataclass(slots=True)
class BenchmarkRun:
    start_date: str
    levels: pd.Series
    event_rows: tuple[dict, ...]


def _finite(value: object) -> bool:
    return bool(pd.notna(value) and np.isfinite(float(value)))


def _buy_fee(notional: Decimal) -> Decimal:
    commission = money(max(MIN_COMMISSION, notional * COMMISSION_RATE))
    slippage = money(notional * SLIPPAGE)
    return commission + slippage


def _sell_fee(notional: Decimal) -> Decimal:
    commission = money(max(MIN_COMMISSION, notional * COMMISSION_RATE))
    stamp = money(notional * STAMP_TAX)
    slippage = money(notional * SLIPPAGE)
    return commission + stamp + slippage


def max_lot_shares(budget: Decimal, trade_price: Decimal) -> int:
    """满足买入费用后不超预算的最大 100 股整数倍。"""
    if budget <= 0 or trade_price <= 0:
        return 0
    lots = int((budget / (trade_price * 100)).to_integral_value(rounding=ROUND_FLOOR))
    while lots > 0:
        shares = lots * 100
        notional = money(trade_price * shares)
        if notional + _buy_fee(notional) <= budget:
            return shares
        lots -= 1
    return 0


def _history_amounts(data: P4Data, signal: str, codes: list[str]) -> pd.DataFrame:
    index = data.dates.index(signal)
    if index < 19:
        raise RuntimeError(f"信号日 {signal} 不足 20 个历史交易日")
    history = data.dates[index - 19 : index + 1]  # noqa: E203
    selected = {str(code) for code in codes}
    frames = []
    for day in history:
        frame = data.read_daily(day)
        frame = frame.loc[frame["code"].isin(selected), ["code", "amount"]].copy()
        frame["day"] = day
        frames.append(frame)
    if not frames:
        return pd.DataFrame(index=codes)
    combined = pd.concat(frames, ignore_index=True)
    pivot = combined.pivot_table(index="code", columns="day", values="amount", aggfunc="last")
    pivot = pivot.reindex(index=codes, columns=history)
    return pivot


def _candidate_frame(data: P4Data, batch) -> tuple[pd.DataFrame, list[dict]]:
    events = data.batch_events(batch.notice_id)
    adds = events.loc[events["direction"].eq("in")].copy()
    adds["code"] = adds["ts_code"].astype(str)
    adds = adds.loc[mainboard_mask(adds["code"])].copy()
    t_daily = data.read_daily(batch.signal_date)
    t_daily = t_daily.loc[mainboard_mask(t_daily["code"])].copy()
    t_basic = data.read_basic(batch.signal_date)[["code", "circ_mv"]]
    merged = (
        adds[["code", "name"]]
        .merge(t_daily, on="code", how="left")
        .merge(t_basic, on="code", how="left")
    )
    merged["eligible"] = (
        merged["close"].map(_finite)
        & (pd.to_numeric(merged["close"], errors="coerce") > 0)
        & merged["is_st"].fillna(1).eq(0)
        & merged["circ_mv"].map(_finite)
    )
    audit: list[dict] = []
    for row in merged.itertuples():
        if not row.eligible:
            audit.append(
                {
                    "notice_id": batch.notice_id,
                    "ts_code": row.code,
                    "name": row.name,
                    "status": "not_selected_eligibility",
                    "reason": "missing T row/ST/nonpositive close/missing circ_mv",
                }
            )
    return merged, audit


def _rank_candidates(
    data: P4Data,
    batch,
    merged: pd.DataFrame,
    audit: list[dict],
) -> tuple[tuple[dict, ...], tuple[dict, ...], Mapping[str, float]]:
    eligible_codes = merged.loc[merged["eligible"], "code"].tolist()
    amounts = _history_amounts(data, batch.signal_date, eligible_codes)
    valid_amounts = amounts.notna().all(axis=1) & (amounts > 0).all(axis=1)
    adv20 = amounts.mean(axis=1, skipna=False)
    merged = merged.set_index("code")
    merged["adv20"] = adv20
    merged["rank_proxy"] = merged["circ_mv"] / merged["adv20"]
    merged["rank_valid"] = merged["eligible"] & valid_amounts.reindex(
        merged.index, fill_value=False
    )
    for code in merged.index[merged["eligible"] & ~merged["rank_valid"]]:
        audit.append(
            {
                "notice_id": batch.notice_id,
                "ts_code": code,
                "name": str(merged.loc[code, "name"]),
                "status": "not_selected_rank_missing",
                "reason": "未满足恰好 20 个有效正 amount",
            }
        )
    ranked = merged.loc[merged["rank_valid"]].sort_values(
        ["rank_proxy", "code"], ascending=[False, True], kind="mergesort"
    )
    selected = tuple(
        {
            "notice_id": batch.notice_id,
            "ts_code": str(code),
            "name": str(row["name"]),
            "rank": int(position + 1),
            "rank_proxy": float(row["rank_proxy"]),
            "circ_mv": float(row["circ_mv"]),
            "adv20": float(row["adv20"]),
        }
        for position, (code, row) in enumerate(ranked.head(6).iterrows())
    )
    selected_codes = {item["ts_code"] for item in selected}
    for rank, (code, row) in enumerate(ranked.iterrows(), start=1):
        if code not in selected_codes:
            audit.append(
                {
                    "notice_id": batch.notice_id,
                    "ts_code": str(code),
                    "name": str(row["name"]),
                    "status": "not_selected_top6",
                    "reason": f"排名 {rank}",
                }
            )
    adv_map = {item["ts_code"]: item["adv20"] for item in selected}
    cap1 = 6 * min(value * 0.01 * 1000 for value in adv_map.values()) if adv_map else None
    cap5 = 6 * min(value * 0.05 * 1000 for value in adv_map.values()) if adv_map else None
    return selected, tuple(audit), {"capacity_1pct": cap1, "capacity_5pct": cap5, "adv": adv_map}


def select_batch(
    data: P4Data, batch
) -> tuple[tuple[dict, ...], tuple[dict, ...], Mapping[str, float]]:
    merged, audit = _candidate_frame(data, batch)
    return _rank_candidates(data, batch, merged, audit)


def prepare_plans(data: P4Data) -> tuple[dict[str, BatchPlan], dict[str, pd.DataFrame]]:
    plans: dict[str, BatchPlan] = {}
    cache: dict[str, pd.DataFrame] = {}
    for batch in data.batches:
        t_daily = data.read_daily(batch.signal_date)
        t_daily = t_daily.loc[mainboard_mask(t_daily["code"])].copy()
        cache[batch.signal_date] = t_daily
        selected, audit, cap = select_batch(data, batch)
        membership = t_daily.loc[
            (~t_daily["is_st"].fillna(1).astype(bool))
            & (pd.to_numeric(t_daily["close"], errors="coerce") > 0),
            "code",
        ].astype(str)
        for day in data.trade_days(batch.entry_date, batch.flow_date):
            if day not in cache:
                frame = data.read_daily(day)
                cache[day] = frame.loc[mainboard_mask(frame["code"])].copy()
        plans[batch.notice_id] = BatchPlan(
            notice_id=batch.notice_id,
            signal_date=batch.signal_date,
            entry_date=batch.entry_date,
            flow_date=batch.flow_date,
            selected=selected,
            selection_audit=audit,
            membership=tuple(sorted(set(membership.tolist()))),
            adv20=cap["adv"],
            capacity_1pct=cap["capacity_1pct"],
            capacity_5pct=cap["capacity_5pct"],
        )
    return plans, cache


def _day_frame(
    data: P4Data, cache: dict[str, pd.DataFrame], day: str, codes: list[str]
) -> pd.DataFrame:
    if day in cache:
        frame = cache[day]
        return frame.loc[frame["code"].isin(codes)].copy()
    return data.read_codes(day, codes)


def _snapshot(
    data: P4Data,
    cache: dict[str, pd.DataFrame],
    day: str,
    codes: list[str],
) -> MarketSnapshot:
    frame = _day_frame(data, cache, day, codes)
    rows = frame.set_index("code") if not frame.empty else frame
    quotes: dict[str, PriceObservation] = {}
    for code in codes:
        if code not in rows.index:
            quotes[code] = PriceObservation(PriceStatus.MISSING_ROW, source="clean/daily")
            continue
        row = rows.loc[code]
        if bool(row.get("is_suspended", False)):
            quotes[code] = PriceObservation(PriceStatus.SUSPENDED, source="clean/daily")
            continue
        close = row.get("close")
        if _finite(close) and float(close) > 0:
            quotes[code] = PriceObservation(PriceStatus.OK, float(close), source="clean/daily")
        else:
            quotes[code] = PriceObservation(PriceStatus.NO_QUOTE, source="clean/daily")
    return MarketSnapshot(day, quotes, source="clean/daily")


def _buy_allowed(row: pd.Series) -> tuple[bool, str]:
    if row is None or not _finite(row.get("open")) or float(row["open"]) <= 0:
        return False, "open_missing"
    if bool(row.get("is_suspended", False)):
        return False, "suspended"
    if bool(row.get("is_st", False)):
        return False, "st_at_entry"
    if not _finite(row.get("pre_close")) or float(row["pre_close"]) <= 0:
        return False, "pre_close_missing"
    limit_up = (Decimal(str(row["pre_close"])) * Decimal("1.10")).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    if Decimal(str(row["open"])) >= limit_up:
        return False, "limit_up_open"
    return True, "ok"


def _sell_close_allowed(row: pd.Series) -> tuple[bool, str]:
    if row is None or not _finite(row.get("close")) or float(row["close"]) <= 0:
        return False, "close_missing"
    if bool(row.get("is_suspended", False)):
        return False, "suspended"
    if bool(row.get("is_limit_down", False)):
        return False, "limit_down_close"
    return True, "ok"


def _sell_open_allowed(row: pd.Series) -> tuple[bool, str]:
    if row is None or not _finite(row.get("open")) or float(row["open"]) <= 0:
        return False, "open_missing"
    if bool(row.get("is_suspended", False)):
        return False, "suspended"
    if not _finite(row.get("pre_close")) or float(row["pre_close"]) <= 0:
        return False, "pre_close_missing"
    rate = Decimal("0.05") if bool(row.get("is_st", False)) else Decimal("0.10")
    limit_down = (Decimal(str(row["pre_close"])) * (Decimal(1) - rate)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    if Decimal(str(row["open"])) <= limit_down:
        return False, "limit_down_open"
    return True, "ok"


def _corporate_schedule(data: P4Data, code: str, entry: str, end: str) -> list[dict]:
    frame = data.dividends.loc[
        data.dividends["ts_code"].eq(code) & (data.dividends["ex_date"] > entry)
    ]
    events = []
    for row in frame.itertuples():
        effective = data.effective_trade_day(str(row.ex_date))
        if effective is None or effective > end:
            continue
        cash = (
            row.cash_div
            if _finite(row.cash_div)
            else (float(row.cash_div_tax) * 0.8 if _finite(row.cash_div_tax) else 0.0)
        )
        ratio = float(row.stk_div) if _finite(row.stk_div) else 0.0
        pay = str(row.pay_date) if len(str(row.pay_date)) == 8 else effective
        events.append(
            {
                "code": code,
                "effective_date": effective,
                "ex_date": str(row.ex_date),
                "pay_date": max(pay, effective),
                "cash_per_share": max(float(cash), 0.0),
                "stk_div": max(ratio, 0.0),
                "end_date": str(row.end_date),
            }
        )
    return events


def _apply_corporate(
    account: ResearchAccount, items: list[dict], day: str, log: list[dict]
) -> None:
    for item in items:
        position = account.position(item["code"])
        if position is None:
            continue
        old_shares = position.shares
        cash_amount = money(Decimal(str(item["cash_per_share"])) * old_shares)
        if cash_amount > 0:
            account.add_receivable(
                f"DIV-{item['code']}-{item['end_date']}-{item['ex_date']}",
                item["code"],
                cash_amount,
                day,
                item["pay_date"],
            )
        if item["stk_div"] > 0:
            account.apply_stock_dividend(
                f"STK-{item['code']}-{item['end_date']}-{item['ex_date']}",
                item["code"],
                item["stk_div"],
                day,
                item["cash_per_share"],
            )
        elif cash_amount > 0:
            account.adjust_last_price_for_cash_dividend(item["code"], item["cash_per_share"], day)
        new_shares = account.position(item["code"]).shares if account.position(item["code"]) else 0
        log.append(
            {
                "date": day,
                "ts_code": item["code"],
                "ex_date": item["ex_date"],
                "pay_date": item["pay_date"],
                "cash_per_share": item["cash_per_share"],
                "cash_amount": float(cash_amount),
                "stk_div": item["stk_div"],
                "shares_before": old_shares,
                "shares_after": new_shares,
            }
        )


def _mark(
    account: ResearchAccount,
    data: P4Data,
    cache: dict[str, pd.DataFrame],
    day: str,
    missing_days: list[int],
) -> object:
    codes = list(account.position_codes)
    before = len(account.missing_ledger)
    mark = account.mark(_snapshot(data, cache, day, codes))
    missing_days.append(len(account.missing_ledger) - before)
    return mark


def _submit_and_fill_buy(
    account: ResearchAccount,
    plan: BatchPlan,
    code: str,
    price_value: float,
    target: Decimal,
) -> tuple[bool, str]:
    budget = min(target, account.cash)
    unit = price(price_value)
    shares = max_lot_shares(budget, unit)
    if shares <= 0:
        return False, "budget_too_small"
    order_id = f"B-{plan.notice_id}-{code}"
    account.submit_order(order_id, code, "buy", shares, plan.signal_date, plan.entry_date)
    notional = money(unit * shares)
    account.execute_fill(
        order_id,
        trade_date=plan.entry_date,
        quantity=shares,
        trade_price=unit,
        fee=_buy_fee(notional),
        fill_id=order_id,
    )
    return True, "filled"


def _submit_and_fill_sell(
    account: ResearchAccount,
    data: P4Data,
    plan: BatchPlan,
    code: str,
    day: str,
    price_value: float,
) -> None:
    position = account.position(code)
    if position is None:
        return
    order_id = f"S-{plan.notice_id}-{code}-{day}"
    account.submit_order(
        order_id,
        code,
        "sell",
        position.shares,
        data.prev_trade_day(day),
        day,
    )
    notional = money(price(price_value) * position.shares)
    account.execute_fill(
        order_id,
        trade_date=day,
        quantity=position.shares,
        trade_price=price_value,
        fee=_sell_fee(notional),
        fill_id=order_id,
    )


def _event_maps(plans: dict[str, BatchPlan], start: str) -> tuple[dict, dict, dict]:
    signals, entries, flows = {}, {}, {}
    for plan in plans.values():
        if plan.signal_date < start:
            continue
        signals.setdefault(plan.signal_date, []).append(plan)
        entries.setdefault(plan.entry_date, []).append(plan)
        flows.setdefault(plan.flow_date, []).append(plan)
    return signals, entries, flows


def _apply_corporate_day(
    account: ResearchAccount,
    state: dict[str, dict],
    day: str,
    corp_log: list[dict],
) -> None:
    for code in account.position_codes:
        item = state.get(code)
        if item is None:
            continue
        due = [event for event in item["corporate"] if event["effective_date"] == day]
        if due:
            _apply_corporate(account, due, day, corp_log)


def _defer_exception(
    item: dict, day: str, side: str, reason: str, status: str = "deferred"
) -> dict:
    return {
        "notice_id": item["notice_id"],
        "ts_code": item["ts_code"],
        "date": day,
        "side": side,
        "status": status,
        "reason": reason,
    }


def _attempt_delayed_sells(
    account: ResearchAccount,
    state: dict[str, dict],
    data: P4Data,
    cache: dict[str, pd.DataFrame],
    plans: dict[str, BatchPlan],
    day: str,
    exceptions: list[dict],
) -> bool:
    attempted = False
    for code, item in list(state.items()):
        if item["flow_date"] >= day:
            continue
        attempted = True
        frame = _day_frame(data, cache, day, [code]).set_index("code")
        row = frame.loc[code] if code in frame.index else None
        allowed, reason = _sell_open_allowed(row)
        if not allowed:
            exceptions.append(_defer_exception(item, day, "sell", reason))
            continue
        _submit_and_fill_sell(
            account, data, plans[item["notice_id"]], code, day, float(row["open"])
        )
        state.pop(code, None)
    return attempted


def _enter_batch(
    account: ResearchAccount,
    state: dict[str, dict],
    data: P4Data,
    cache: dict[str, pd.DataFrame],
    plan: BatchPlan,
    day: str,
    nav_signal: dict[str, Decimal],
    exceptions: list[dict],
) -> None:
    slots = 6 - len(account.positions())
    target = nav_signal[plan.notice_id] / 6
    for item in plan.selected:
        code = item["ts_code"]
        if code in account.position_codes:
            exceptions.append(
                _defer_exception(
                    {"notice_id": plan.notice_id, "ts_code": code},
                    day,
                    "buy",
                    "already held",
                    "skipped_existing",
                )
            )
            continue
        if slots <= 0:
            exceptions.append(
                _defer_exception(
                    {"notice_id": plan.notice_id, "ts_code": code},
                    day,
                    "buy",
                    "no_slot",
                    "expired_unfilled",
                )
            )
            continue
        frame = _day_frame(data, cache, day, [code]).set_index("code")
        row = frame.loc[code] if code in frame.index else None
        allowed, reason = _buy_allowed(row)
        if not allowed:
            exceptions.append(
                _defer_exception(
                    {"notice_id": plan.notice_id, "ts_code": code},
                    day,
                    "buy",
                    reason,
                    "expired_unfilled",
                )
            )
            continue
        filled, status = _submit_and_fill_buy(account, plan, code, float(row["open"]), target)
        exceptions.append(
            _defer_exception(
                {"notice_id": plan.notice_id, "ts_code": code},
                day,
                "buy",
                "ok" if filled else status,
                status,
            )
        )
        if not filled:
            continue
        slots -= 1
        state[code] = {
            "notice_id": plan.notice_id,
            "ts_code": code,
            "entry_date": plan.entry_date,
            "flow_date": plan.flow_date,
            "corporate": _corporate_schedule(
                data, code, plan.entry_date, getattr(data, "cutoff", data.dates[-1])
            ),
        }


def _exit_flow_batch(
    account: ResearchAccount,
    state: dict[str, dict],
    data: P4Data,
    cache: dict[str, pd.DataFrame],
    plans: dict[str, BatchPlan],
    plan: BatchPlan,
    day: str,
    exceptions: list[dict],
) -> None:
    for code, item in list(state.items()):
        if item["flow_date"] != day or item["notice_id"] != plan.notice_id:
            continue
        frame = _day_frame(data, cache, day, [code]).set_index("code")
        row = frame.loc[code] if code in frame.index else None
        allowed, reason = _sell_close_allowed(row)
        if not allowed:
            exceptions.append(_defer_exception(item, day, "sell", reason))
            continue
        _submit_and_fill_sell(
            account, data, plans[item["notice_id"]], code, day, float(row["close"])
        )
        state.pop(code, None)


def _candidate_fills(account: ResearchAccount) -> tuple[dict, ...]:
    return tuple(
        {
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "trade_date": fill.trade_date,
            "ts_code": fill.ts_code,
            "side": fill.side.value,
            "shares": fill.shares,
            "trade_price": str(fill.trade_price),
            "amount": str(fill.amount),
            "fee": str(fill.fee),
        }
        for fill in account.fills
    )


def _candidate_final_positions(account: ResearchAccount) -> tuple[dict, ...]:
    return tuple(
        {
            "ts_code": item.ts_code,
            "shares": item.shares,
            "last_price": str(item.last_price),
            "last_price_date": item.last_price_date,
        }
        for item in account.positions()
    )


def run_candidate(
    data: P4Data,
    plans: dict[str, BatchPlan],
    cache: dict[str, pd.DataFrame],
    *,
    principal: float,
    start_date: str,
    label: str,
) -> CandidateRun:
    account = ResearchAccount(principal)
    levels: dict[str, float] = {start_date: float(principal)}
    nav_signal: dict[str, Decimal] = {}
    state: dict[str, dict] = {}
    exceptions: list[dict] = []
    corp_log: list[dict] = []
    position_days = 0
    missing_days: list[int] = []
    utilization: list[dict] = []
    last_nav = Decimal(str(principal))
    signals, entries, flows = _event_maps(plans, start_date)
    days = data.trade_days(start_date, getattr(data, "cutoff", data.dates[-1]))
    for day in days:
        _apply_corporate_day(account, state, day, corp_log)
        delayed = _attempt_delayed_sells(account, state, data, cache, plans, day, exceptions)
        for plan in entries.get(day, []):
            _enter_batch(account, state, data, cache, plan, day, nav_signal, exceptions)
        for plan in flows.get(day, []):
            _exit_flow_batch(account, state, data, cache, plans, plan, day, exceptions)
        need_mark = (
            bool(account.position_codes)
            or day in signals
            or day in entries
            or day in flows
            or delayed
        )
        if need_mark:
            before_positions = len(account.position_codes)
            mark = _mark(account, data, cache, day, missing_days)
            position_days += before_positions
            levels[day] = float(mark.nav)
            last_nav = mark.nav
            utilization.append(
                {
                    "trade_date": str(day),
                    "market_value": float(mark.market_value),
                    "nav": float(mark.nav),
                }
            )
            for plan in signals.get(day, []):
                nav_signal[plan.notice_id] = mark.nav
                levels[plan.signal_date] = float(mark.nav)
            continue
        # 无持仓/无事件的空窗日：利用率按 0 计（不删零事件月），NAV 为现金
        utilization.append(
            {"trade_date": str(day), "market_value": 0.0, "nav": float(last_nav)}
        )
    index = pd.Index(days, name="trade_date")
    level_series = pd.Series(levels, dtype=float).reindex(index).ffill().bfill()
    if level_series.isna().any():
        raise RuntimeError("候选 NAV 序列存在无法填充的缺口")
    return CandidateRun(
        label=label,
        principal=float(principal),
        start_date=start_date,
        end_date=days[-1],
        levels=level_series,
        fills=_candidate_fills(account),
        exceptions=tuple(exceptions),
        corporate_actions=tuple(corp_log),
        missing_rows=tuple(account.missing_ledger.to_rows()),
        final_positions=_candidate_final_positions(account),
        position_days=position_days,
        missing_position_days=int(sum(missing_days)),
        nav_marks=levels,
        daily_utilization=tuple(utilization),
    )


def run_benchmark(
    data: P4Data,
    plans: dict[str, BatchPlan],
    cache: dict[str, pd.DataFrame],
    *,
    principal: float,
    start_date: str,
) -> BenchmarkRun:
    levels: dict[str, float] = {start_date: float(principal)}
    event_rows = []
    nav = float(principal)
    for plan in sorted(plans.values(), key=lambda item: item.signal_date):
        if plan.signal_date < start_date:
            continue
        members = list(plan.membership)
        if not members:
            event_rows.append(
                {"notice_id": plan.notice_id, "members": 0, "valid_entry_weight": 0.0}
            )
            continue
        entry_frame = _day_frame(data, cache, plan.entry_date, members).set_index("code")
        weight = 1.0 / len(members)
        units: dict[str, float] = {}
        last: dict[str, float] = {}
        cash = 0.0
        for code in members:
            row = entry_frame.loc[code] if code in entry_frame.index else None
            open_adj = 0.0 if row is None else float(row.get("open_adj", np.nan))
            if not np.isfinite(open_adj) or open_adj <= 0:
                cash += weight * nav
                continue
            units[code] = weight * nav / open_adj
            last[code] = open_adj
        deployed_weight = weight * len(units)
        for day in data.trade_days(plan.entry_date, plan.flow_date):
            frame = _day_frame(data, cache, day, list(units)).set_index("code")
            value = cash
            for code, unit in units.items():
                row = frame.loc[code] if code in frame.index else None
                close_adj = 0.0 if row is None else float(row.get("close_adj", np.nan))
                if np.isfinite(close_adj) and close_adj > 0:
                    last[code] = close_adj
                value += unit * last[code]
            nav = value
            levels[day] = nav
        event_rows.append(
            {
                "notice_id": plan.notice_id,
                "members": len(members),
                "invested_members": len(units),
                "valid_entry_weight": deployed_weight,
                "uninvested_weight": 1.0 - deployed_weight,
                "nav_end": nav,
            }
        )
    index = pd.Index(
        data.trade_days(start_date, getattr(data, "cutoff", data.dates[-1])),
        name="trade_date",
    )
    level_series = pd.Series(levels, dtype=float).reindex(index).ffill().bfill()
    return BenchmarkRun(
        start_date=start_date,
        levels=level_series,
        event_rows=tuple(event_rows),
    )
