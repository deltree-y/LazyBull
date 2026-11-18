"""B 袖原生回放：批次排期、收盘买入、开盘卖出、行业与槽位约束。

相位（协议 §9.2）：T 生成信号 → T+1 **收盘**买入；到期在**开盘**卖出；
每批 60 日周期，偏移 0/20/40；槽位 7/7/6；单股预算 = B 资产/20，上限 10%；
一级行业最多 5 只（含存量与拟买）；现金为硬上限，整手取整。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.lazybull.trading.sell_rules import is_holding_period_exit_due

from scripts.v2_p7.defensive_ml import b_account, spec


@dataclass
class ReplayConfig:
    """回放参数（默认即冻结规格；压力只允许改 slippage / rank_lag / 现金折扣）。"""

    top_n: int = int(spec.PORTFOLIO["top_n"])
    holding_period: int = int(spec.PORTFOLIO["holding_period"])
    rebalance: int = int(spec.PORTFOLIO["rebalance_freq"])
    tranches: tuple[int, ...] = tuple(spec.PORTFOLIO["tranche_slots"])
    offsets: tuple[int, ...] = tuple(spec.PORTFOLIO["tranche_offsets"])
    max_per_industry: int = int(spec.PORTFOLIO["max_per_industry"])
    max_single_weight: float = float(spec.PORTFOLIO["max_single_stock_weight"])
    slippage: float = float(spec.COSTS["slippage"])
    buy_retry_days: int = int(spec.PORTFOLIO["buy_retry_days"])
    buys_per_day_retry: int = int(spec.PORTFOLIO["buy_retries_per_day"])
    rank_lag_days: int = 0
    cash_equity_discount: float = 0.0
    min_scored: int = int(spec.UNIVERSE["min_scored_per_day"])
    lot_size: int = 100


@dataclass
class _PendingBuy:
    ts_code: str
    budget: float
    tranche: int
    created: str
    days_left: int
    tries_left: int
    industry: str = "UNKNOWN"


@dataclass
class _PendingSell:
    lot_id: int
    ts_code: str
    reason: str
    since: str


@dataclass
class ReplayResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    dividends: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _series_map(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """宽表 -> {date: {code: value}}（仅保留有限值）。"""
    out: dict[str, dict[str, float]] = {}
    values = frame.to_numpy(dtype=float)
    columns = list(frame.columns)
    for i, day in enumerate(frame.index):
        row = values[i]
        out[str(day)] = {
            columns[j]: float(row[j]) for j in range(len(columns)) if np.isfinite(row[j])
        }
    return out


def _flag_map(frame: pd.DataFrame) -> dict[str, set[str]]:
    """宽表 0/1 -> {date: {code}}（值为 1 的集合）。"""
    out: dict[str, set[str]] = {}
    values = frame.to_numpy(dtype=float)
    columns = list(frame.columns)
    for i, day in enumerate(frame.index):
        row = values[i]
        out[str(day)] = {columns[j] for j in range(len(columns)) if row[j] == 1.0}
    return out


def _rank_lists(ranks: pd.DataFrame, lag: int, calendar: Sequence[str]) -> dict[str, list[str]]:
    """{信号日: 按 rank 升序的代码表}；lag>0 时 T 日改用 T-lag 的排名。"""
    grouped = ranks.sort_values(["trade_date", "rank"]).groupby("trade_date")["ts_code"]
    base = {str(day): [str(code) for code in codes] for day, codes in grouped}
    if lag <= 0:
        return base
    shifted: dict[str, list[str]] = {}
    for index, day in enumerate(calendar):
        if index < lag:
            continue
        source = base.get(calendar[index - lag])
        if source is None:
            continue
        # 压力语义：在**原 T 日**使用 T−lag 的排名；调仓日期不变，绝不把未来排名搬到过去
        shifted[day] = source
    return shifted


def _industry_map(industry: pd.DataFrame) -> dict[str, dict[str, str]]:
    """{(date, code): bucket} 的日期分组视图。"""
    out: dict[str, dict[str, str]] = {}
    for row in industry.itertuples(index=False):
        out.setdefault(str(row.trade_date), {})[str(row.ts_code)] = str(row.industry_bucket)
    return out


def _assets(account: b_account.BAccount, day: str, close: Mapping[str, float],
            stale: float) -> float:
    value, _ = account.market_value(day, close)
    return account.cash + value + account.receivable + stale


def run_b_arm(
    *,
    prices: dict[str, pd.DataFrame],
    ranks: pd.DataFrame,
    industry: pd.DataFrame,
    calendar: Sequence[str],
    start: str,
    end: str,
    capital: float,
    arm: str,
    corporate: Any,
    config: ReplayConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> ReplayResult:
    """跑单臂 B 现金账户；返回逐日账、成交、分红与诊断。"""
    cfg = config or ReplayConfig()
    emit = log or (lambda message: None)
    days = [day for day in calendar if start <= day <= end]
    if not days:
        raise ValueError("回放窗口为空")
    open_map = _series_map(prices["open"])
    close_map = _series_map(prices["close"])
    suspended = _flag_map(prices["is_suspended"])
    limit_up = _flag_map(prices["is_limit_up"])
    limit_down = _flag_map(prices["is_limit_down"])
    tradable = _flag_map(prices["tradable"])
    rank_map = _rank_lists(ranks, cfg.rank_lag_days, list(calendar))
    industry_lookup = _industry_map(industry)

    account = b_account.BAccount(capital=capital, arm=arm)
    pending_buys: list[_PendingBuy] = []
    pending_sells: list[_PendingSell] = []
    stale_prices: dict[str, float] = {}
    stale_value = 0.0
    fees_total = 0.0
    stats = {
        "buy_attempts": 0,
        "buy_filled": 0,
        "buy_deferred": 0,
        "buy_expired": 0,
        "sell_signals": 0,
        "sell_filled": 0,
        "sell_deferred": 0,
        "dividend_cash": 0.0,
        "bonus_events": 0,
        "below_min_days": 0,
        "missing_price_days": 0,
        "skipped_no_rank": 0,
    }

    day_position = {day: i for i, day in enumerate(calendar)}
    start_index = day_position[days[0]]
    schedule = {}
    for tranche, offset in enumerate(cfg.offsets):
        slots = cfg.tranches[tranche]
        steps = [(k * cfg.rebalance + offset) for k in range(0, 64)]
        for step in steps:
            index = start_index + step
            if index < len(calendar):
                schedule.setdefault(calendar[index], []).append((tranche, slots))

    for position, day in enumerate(days):
        next_day = days[position + 1] if position + 1 < len(days) else "99999999"
        b_account.release_locks([account], day)

        stats["dividend_cash"] += account.settle_due(day)
        ex_events = corporate.ex_events(day) if corporate else ()
        if ex_events:
            account.apply_ex_date(day, ex_events, next_day)
            stats["bonus_events"] += sum(
                1 for event in ex_events if getattr(event, "share_ratio", 0)
            )

        # 1) 开盘卖出（到期或延期义务）
        still_pending: list[_PendingSell] = []
        for order in pending_sells:
            lot = account.lots.get(order.lot_id)
            if lot is None:
                continue
            price = open_map.get(day, {}).get(order.ts_code)
            blocked = (
                order.ts_code in suspended.get(day, set())
                or order.ts_code in limit_down.get(day, set())
                or price is None
                or price <= 0
            )
            if blocked:
                stats["sell_deferred"] += 1
                account.blocked_sells.append(
                    {"trade_date": day, "ts_code": order.ts_code, "reason": order.reason}
                )
                still_pending.append(order)
                continue
            shares = lot.sellable(day)
            if shares <= 0:
                still_pending.append(order)
                continue
            record = account.sell(day, order.lot_id, shares, price, order.reason, cfg.slippage)
            fees_total += record["fee"]
            stats["sell_filled"] += 1
        pending_sells = still_pending

        # 2) 收盘买入（前一日信号 + 候补重试）
        still_buys: list[_PendingBuy] = []
        cash_limit = account.cash * (1.0 - cfg.cash_equity_discount)
        for order in pending_buys:
            price = close_map.get(day, {}).get(order.ts_code)
            blocked = (
                order.ts_code in suspended.get(day, set())
                or order.ts_code in limit_up.get(day, set())
                or order.ts_code not in tradable.get(day, set())
                or price is None
                or price <= 0
            )
            if blocked or order.tries_left <= 0:
                order.days_left -= 1
                order.tries_left = cfg.buys_per_day_retry
                stats["buy_deferred"] += 1
                if order.days_left > 0:
                    still_buys.append(order)
                else:
                    stats["buy_expired"] += 1
                continue
            budget = min(order.budget, max(0.0, cash_limit))
            fill_price = price * (1.0 + cfg.slippage)
            shares = int(budget / fill_price // cfg.lot_size) * cfg.lot_size
            # 费用必须落在现金内：逐手回退直到「成交额 + 佣金 ≤ 可用现金」
            while shares > 0:
                amount = fill_price * shares
                fee = account._buy_cost(amount)  # noqa: SLF001 - 同模块费率唯一来源
                if amount + fee <= cash_limit + 1e-9:
                    break
                shares -= cfg.lot_size
            stats["buy_attempts"] += 1
            if shares <= 0:
                order.days_left -= 1
                order.tries_left -= 1
                if order.days_left > 0:
                    still_buys.append(order)
                else:
                    stats["buy_expired"] += 1
                continue
            record = account.buy(day, order.ts_code, shares, price, order.tranche, cfg.slippage)
            fees_total += record["fee"]
            cash_limit = account.cash * (1.0 - cfg.cash_equity_discount)
            stats["buy_filled"] += 1
        pending_buys = still_buys

        # 3) 估值（缺价沿上一真实价并登记陈旧）
        close_today = close_map.get(day, {})
        for code, price in close_today.items():
            stale_prices[code] = price
        held_shares = account.shares()
        missing = [code for code in held_shares if code not in close_today]
        if missing:
            stats["missing_price_days"] += 1
        valuation_prices = {
            code: close_today.get(code, stale_prices.get(code, 0.0)) for code in held_shares
        }
        stale_value = sum(
            stale_prices.get(code, 0.0) * held_shares[code] for code in missing
        )
        snapshot = account.snapshot(day, valuation_prices, stale_value, fees_total)
        snapshot["pending_buys"] = len(pending_buys)
        snapshot["pending_sells"] = len(pending_sells)

        # 4) 生成下一交易日订单
        for lot_id, lot in list(account.lots.items()):
            held_days = day_position[day] - day_position[lot.buy_date]
            if is_holding_period_exit_due(held_days, cfg.holding_period):
                if not any(item.lot_id == lot_id for item in pending_sells):
                    pending_sells.append(
                        _PendingSell(lot_id, lot.ts_code, "holding_period_end", day)
                    )
                    stats["sell_signals"] += 1

        entry_slots = schedule.get(day)
        if entry_slots:
            codes = rank_map.get(day)
            if not codes:
                stats["skipped_no_rank"] += 1
            elif len(codes) < cfg.min_scored:
                stats["below_min_days"] += 1
            else:
                pending_buys.extend(
                    _select_candidates(
                        day=day,
                        schedule=entry_slots,
                        codes=codes,
                        account=account,
                        close=close_map.get(day, {}),
                        industry=industry_lookup.get(day, {}),
                        pending_buys=pending_buys,
                        pending_sells=pending_sells,
                        cfg=cfg,
                        stats=stats,
                    )
                )

    daily = pd.DataFrame(account.daily)
    trades = pd.DataFrame(account.trades)
    dividends = pd.DataFrame(account.dividends)
    diagnostics = {
        "arm": arm,
        "capital": capital,
        "days": len(daily),
        "trades": len(trades),
        "stats": stats,
        "conservation": account.conservation(),
        "ending": daily.iloc[-1].to_dict() if len(daily) else {},
    }
    emit(f"{arm} {capital:.0f}: days={len(daily)} trades={len(trades)} "
         f"end={diagnostics['ending'].get('total_value')}")
    return ReplayResult(daily=daily, trades=trades, dividends=dividends, diagnostics=diagnostics)


def _select_candidates(
    *,
    day: str,
    schedule: Sequence[tuple[int, int]],
    codes: Sequence[str],
    account: b_account.BAccount,
    close: Mapping[str, float],
    industry: Mapping[str, str],
    pending_buys: Sequence[_PendingBuy],
    pending_sells: Sequence[_PendingSell],
    cfg: ReplayConfig,
    stats: dict[str, Any],
) -> list[_PendingBuy]:
    """按排名选股：排除已持仓、行业 ≤5、单股 ≤10%、槽位预算等权。"""
    held = set(account.shares())
    leaving = {order.ts_code for order in pending_sells}
    held = {code for code in held if code not in leaving}
    bucket_count: dict[str, int] = {}
    for code in held:
        bucket = industry.get(code, "UNKNOWN")
        bucket_count[bucket] = bucket_count.get(bucket, 0) + 1
    for order in pending_buys:
        bucket = order.industry
        bucket_count[bucket] = bucket_count.get(bucket, 0) + 1

    assets = max(1.0, account.cash + sum(
        account.lots[lot].shares * close.get(account.lots[lot].ts_code, 0.0)
        for lot in account.lots
    ))
    per_name = assets / float(cfg.top_n)
    cap = assets * cfg.max_single_weight
    orders: list[_PendingBuy] = []
    for tranche, slots in schedule:
        taken = 0
        for code in codes:
            if taken >= slots:
                break
            if code in held or code in leaving:
                continue
            if any(order.ts_code == code for order in pending_buys) or any(
                order.ts_code == code for order in orders
            ):
                continue
            bucket = industry.get(code, "UNKNOWN")
            if bucket_count.get(bucket, 0) >= cfg.max_per_industry:
                continue
            if close.get(code, 0.0) <= 0:
                continue
            bucket_count[bucket] = bucket_count.get(bucket, 0) + 1
            orders.append(
                _PendingBuy(
                    ts_code=code,
                    budget=min(per_name, cap),
                    tranche=tranche,
                    created=day,
                    days_left=cfg.buy_retry_days,
                    tries_left=cfg.buys_per_day_retry,
                    industry=bucket,
                )
            )
            taken += 1
    stats["candidate_orders"] = stats.get("candidate_orders", 0) + len(orders)
    return orders
