# -*- coding: utf-8 -*-
"""C1 预筛：六槽月度账户、同域基准、T2.5 A 冻结持仓重放与 70/30 组合。

r3 修复（C1-PSR2-01/02/04）：
- 逐交易日主循环 + 真正的 pending 订单；未来成交不提前进入历史状态；
- 总回报口径的持仓单位，卖出与盯市同一账，现金分红视为再投资；
- 缺价不回退买入成本，记录 last_valid 与影响；留存股按 NAV/6 调整；
- 基准在信号日冻结成员/权重，缺价保留份额并给压力带。

探索性预筛，不是正式 P-1 仪器；所有近似在报告 manifest 中显式登记。

# 复杂度豁免：scripts/v2_p7/c1/** 为阶段专用探索性预筛工具，不入生产链路；
# run_c1 与 replay_frozen_holdings 超硬限，预筛若升级为正式仪器再拆分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.v2_p7.c1.monthly import MonthlyPanel, top_codes
from scripts.v2_p7.t1.dataplane import Panel

COST = {
    "commission_rate": 0.0001954,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.0005,
}
LOT = 100
MIN_TICKET = 25588.536335721597
TOP_N = 6
MAX_DEFER = 5


def _buy_cost(amount: float) -> float:
    if amount <= 0:
        return 0.0
    commission = max(COST["min_commission"], amount * COST["commission_rate"])
    return commission + amount * COST["slippage"]


def _sell_cost(amount: float) -> float:
    if amount <= 0:
        return 0.0
    commission = max(COST["min_commission"], amount * COST["commission_rate"])
    return commission + amount * COST["stamp_tax"] + amount * COST["slippage"]


def _safe_price(panel: Panel, day: int, idx: int, field: str) -> float:
    value = getattr(panel, field)[day, idx]
    return float(value) if np.isfinite(value) and value > 0 else float("nan")


@dataclass
class Position:
    units: float  # 总回报口径的持仓价值（入场日 = 股数 × 裸价）
    entry_adj: float
    entry_day: int
    code: str
    shares_raw: int
    last_valid_adj: float


@dataclass
class C1Result:
    nav: pd.Series
    trades: pd.DataFrame
    monthly: pd.DataFrame
    metrics: Dict[str, float]
    exceptions: Dict[str, int] = field(default_factory=dict)
    rebalances: List[dict] = field(default_factory=list)
    validity: Dict[str, Any] = field(default_factory=dict)


def _position_value(panel: Panel, day: int, pos: Position, counters: Dict[str, int]) -> float:
    idx = panel.stock_index[pos.code]
    adj = panel.close_adj[day, idx]
    if np.isfinite(adj) and adj > 0:
        pos.last_valid_adj = float(adj)
    else:
        counters["missing_price_days"] += 1
        counters["missing_price_value_days"] = counters.get("missing_price_value_days", 0.0) + (
            pos.units * (pos.last_valid_adj / pos.entry_adj)
        )
        adj = pos.last_valid_adj
    return pos.units * (float(adj) / pos.entry_adj)


def _portfolio_value(
    panel: Panel, day: int, holdings: Mapping[str, Position], cash: float, counters: Dict[str, int]
) -> float:
    value = cash
    for pos in holdings.values():
        value += _position_value(panel, day, pos, counters)
    return float(value)


def _try_sell(
    panel: Panel,
    day: int,
    code: str,
    pos: Position,
    target_value: float,
    cash: float,
    trades: List[dict],
    signal_date: str,
) -> Tuple[float, bool, str]:
    """按总回报价值卖出；返回 (现金, 是否清仓, 状态 done/blocked/no_price)。

    订单完成与持仓清空分开：目标部分减持完成后由主循环移出 pending，
    不再用 remaining=0 表示“还有 0 元要卖”，避免下一日把 0 解释成全仓卖出。
    """
    idx = panel.stock_index[code]
    if panel.suspended[day, idx] or panel.limit_down[day, idx]:
        return cash, False, "blocked"
    adj = _safe_price(panel, day, idx, "open_adj")
    if not np.isfinite(adj):
        return cash, False, "no_price"
    current = pos.units * (adj / pos.entry_adj)
    value = current if target_value <= 0 or target_value >= current else target_value
    if value <= 1e-9:
        return cash, False, "done"
    cost = _sell_cost(value)
    cash += value - cost
    shares = int(np.floor(pos.shares_raw * value / current)) if current > 0 else 0
    trades.append(
        {
            "trade_date": panel.days[day],
            "signal_date": signal_date,
            "ts_code": code,
            "side": "SELL",
            "shares": shares,
            "price": float(adj),
            "amount": float(value),
            "cost": float(cost),
            "entry_day": panel.days[pos.entry_day],
            "total_return_value": float(value),
            "raw_notional": float(shares * adj),
            "account_basis": "total_return_proxy",
        }
    )
    closed = value >= current - 1e-9
    if not closed:
        pos.units *= 1.0 - value / current
    return cash, closed, "done"


def _try_buy(
    panel: Panel,
    day: int,
    code: str,
    target_value: float,
    cash: float,
    holdings: Dict[str, Position],
    trades: List[dict],
    top_n: int,
    signal_date: str,
) -> Tuple[float, float, str]:
    """按总回报价值买入；返回 (现金, 剩余目标, 状态 full/partial/blocked/no_slot/no_price/budget)。"""
    if code not in holdings and len(holdings) >= top_n:
        return cash, target_value, "no_slot"
    idx = panel.stock_index[code]
    if panel.suspended[day, idx] or panel.limit_up[day, idx]:
        return cash, target_value, "blocked"
    price = _safe_price(panel, day, idx, "open_u")
    adj = _safe_price(panel, day, idx, "open_adj")
    if not np.isfinite(price) or not np.isfinite(adj):
        return cash, target_value, "no_price"
    budget = min(target_value, cash)
    if budget < MIN_TICKET:
        return cash, target_value, "budget"
    effective = budget / (1.0 + COST["slippage"])
    shares = int(np.floor(effective / price / LOT) * LOT)
    while shares > 0:
        amount = shares * price
        cost = _buy_cost(amount)
        if amount + cost <= cash + 1e-9:
            break
        shares -= LOT
    if shares <= 0:
        return cash, target_value, "budget"
    amount = shares * price
    cost = _buy_cost(amount)
    cash -= amount + cost
    trades.append(
        {
            "trade_date": panel.days[day],
            "signal_date": signal_date,
            "ts_code": code,
            "side": "BUY",
            "shares": shares,
            "price": price,
            "amount": amount,
            "cost": cost,
            "entry_day": panel.days[day],
            "total_return_value": amount,
            "raw_notional": amount,
            "account_basis": "total_return_proxy",
        }
    )
    if code in holdings:
        pos = holdings[code]
        current = pos.units * (adj / pos.entry_adj)
        pos.units = current + amount
        pos.entry_adj = adj
        pos.shares_raw += shares
    else:
        holdings[code] = Position(
            units=amount,
            entry_adj=adj,
            entry_day=day,
            code=code,
            shares_raw=shares,
            last_valid_adj=adj,
        )
    remaining = max(0.0, target_value - amount)
    return cash, remaining, "full" if remaining <= MIN_TICKET else "partial"


def run_c1(
    panel: Panel,
    monthly: MonthlyPanel,
    stocks: Sequence[str],
    *,
    principal: float = 300_000.0,
    top_n: int = TOP_N,
) -> C1Result:
    days = list(panel.days)
    ends = [int(x) for x in monthly.month_end_days]
    first_month = 0
    while first_month < len(ends) and not np.isfinite(monthly.signals[first_month]).any():
        first_month += 1
    if first_month >= len(ends) - 1:
        raise ValueError("C1 无有效信号月")
    first_exec = ends[first_month] + 1
    if first_exec >= panel.n_days:
        raise ValueError("C1 首个执行日超出面板")

    cash = float(principal)
    holdings: Dict[str, Position] = {}
    trades: List[dict] = []
    rebalances: List[dict] = []
    counters = {
        "stale_positions": 0,
        "unfilled_buys": 0,
        "negative_cash_days": 0,
        "missing_price_days": 0,
        "missing_price_value_days": 0.0,
        "signal_day_trades": 0,
    }
    pending_sells: Dict[str, float] = {}
    pending_buys: Dict[str, float] = {}
    pending_since: Dict[str, int] = {}
    pending_signal: Dict[str, str] = {}
    exec_map = {
        ends[m] + 1: m
        for m in range(len(ends))
        if ends[m] + 1 < panel.n_days and np.isfinite(monthly.signals[m]).any()
    }

    nav_dates: List[str] = []
    nav_values: List[float] = []
    # 本金起点：首个执行日前一日
    nav_dates.append(days[first_exec - 1])
    nav_values.append(float(principal))

    for day in range(first_exec, panel.n_days):
        if day in exec_map:
            m = exec_map[day]
            targets = top_codes(monthly, stocks, m, top_n)
            signal_day = ends[m]
            nav_signal = _portfolio_value(panel, signal_day, holdings, cash, counters)
            target_value = nav_signal / float(top_n)
            pending_sells = {}
            pending_buys = {}
            pending_since = {}
            pending_signal = {}
            signal_label = days[signal_day]
            if cash < 0:
                counters["negative_cash_days"] += 1
            for code, pos in holdings.items():
                idx = panel.stock_index[code]
                signal_adj = _safe_price(panel, signal_day, idx, "close_adj")
                if not np.isfinite(signal_adj):
                    signal_adj = pos.last_valid_adj
                current = pos.units * (signal_adj / pos.entry_adj)
                if code not in targets:
                    pending_sells[code] = current
                    pending_since[code] = day
                    pending_signal[code] = signal_label
                else:
                    diff = current - target_value
                    lot_price = _safe_price(panel, signal_day, idx, "open_u")
                    lot_value = lot_price * LOT if np.isfinite(lot_price) else MIN_TICKET
                    if diff > lot_value:
                        pending_sells[code] = diff
                        pending_since[code] = day
                        pending_signal[code] = signal_label
                    elif diff < -MIN_TICKET:
                        pending_buys[code] = -diff
                        pending_since[code] = day
                        pending_signal[code] = signal_label
            for code in targets:
                if code not in holdings:
                    pending_buys[code] = target_value
                    pending_since[code] = day
                    pending_signal[code] = signal_label
            if len(holdings) > top_n:
                # 已有超槽（stale）：不再新买
                for code in list(pending_buys):
                    if code not in holdings:
                        del pending_buys[code]
                        pending_since.pop(code, None)
            rebalances.append(
                {
                    "signal_date": days[signal_day],
                    "execution_date": days[day],
                    "targets": targets,
                    "pending_sells": list(pending_sells),
                    "pending_buys": list(pending_buys),
                    "nav_signal": nav_signal,
                }
            )

        # 只处理当天可执行订单；未来订单保留在 pending
        if pending_sells:
            for code in list(pending_sells):
                pos = holdings.get(code)
                if pos is None:
                    pending_sells.pop(code, None)
                    pending_signal.pop(code, None)
                    continue
                cash, closed, status = _try_sell(
                    panel,
                    day,
                    code,
                    pos,
                    pending_sells[code],
                    cash,
                    trades,
                    pending_signal.get(code, ""),
                )
                if status == "done":
                    pending_sells.pop(code, None)
                    pending_signal.pop(code, None)
                    if closed:
                        holdings.pop(code, None)
        if pending_buys:
            for code in list(pending_buys):
                cash, remaining, status = _try_buy(
                    panel,
                    day,
                    code,
                    pending_buys[code],
                    cash,
                    holdings,
                    trades,
                    top_n,
                    pending_signal.get(code, ""),
                )
                if status == "full":
                    pending_buys.pop(code, None)
                    pending_signal.pop(code, None)
                elif status == "partial":
                    pending_buys[code] = remaining
        # 超时取消
        for code in list(pending_sells):
            if day - pending_since.get(code, day) > MAX_DEFER:
                counters["stale_positions"] += 1
                pending_sells.pop(code, None)
                pending_signal.pop(code, None)
        for code in list(pending_buys):
            if day - pending_since.get(code, day) > MAX_DEFER:
                counters["unfilled_buys"] += 1
                pending_buys.pop(code, None)
                pending_signal.pop(code, None)

        if cash < 0:
            counters["negative_cash_days"] += 1
        value = _portfolio_value(panel, day, holdings, cash, counters)
        nav_dates.append(days[day])
        nav_values.append(value)

    nav = pd.Series(nav_values, index=pd.to_datetime(nav_dates, format="%Y%m%d"), name="nav")
    trades_df = pd.DataFrame(trades)
    if len(trades_df):
        counters["signal_day_trades"] = int(
            (trades_df["trade_date"] <= trades_df["signal_date"]).sum()
        )
    monthly_nav = nav.resample("ME").last().dropna()
    monthly_ret = monthly_nav.pct_change().dropna()
    metrics = performance_metrics(nav, trades_df, counters, principal)
    validity = {
        "signal_day_trades": int(counters["signal_day_trades"]),
        "negative_cash_days": int(counters["negative_cash_days"]),
        "missing_price_days": int(counters["missing_price_days"]),
        "account_basis": "total_return_proxy",
        "executable_cash_account": False,
        "cash_flow_checked": False,
        "valid": bool(counters["signal_day_trades"] == 0 and counters["negative_cash_days"] == 0),
    }
    return C1Result(
        nav=nav,
        trades=trades_df,
        monthly=pd.DataFrame({"nav": monthly_nav, "ret": monthly_ret}),
        metrics=metrics,
        exceptions=counters,
        rebalances=rebalances,
        validity=validity,
    )


def performance_metrics(
    nav: pd.Series, trades: pd.DataFrame, exceptions: Mapping[str, int], principal: float
) -> Dict[str, float]:
    if len(nav) < 2:
        return {}
    values = nav.to_numpy(dtype=float)
    days = (nav.index[-1] - nav.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (
        (values[-1] / principal) ** (1.0 / years) - 1.0 if years > 0 and principal > 0 else np.nan
    )
    peak = np.maximum.accumulate(values)
    drawdown = values / peak - 1.0
    max_dd = float(np.nanmin(drawdown))
    ret = nav.pct_change().dropna()
    vol = float(ret.std(ddof=1) * np.sqrt(252)) if len(ret) > 1 else np.nan
    sharpe = float(ret.mean() * 252 / vol) if vol and np.isfinite(vol) and vol > 0 else np.nan
    total_cost = float(trades["cost"].sum()) if len(trades) else 0.0
    notional = float(trades["amount"].sum()) if len(trades) else 0.0
    avg_nav = float(np.nanmean(values))
    turnover = notional / avg_nav / years if years > 0 and avg_nav > 0 else np.nan
    cost_drag = total_cost / avg_nav / years if years > 0 and avg_nav > 0 else np.nan
    return {
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "years": float(years),
        "initial_capital": float(principal),
        "cagr": float(cagr),
        "max_drawdown": max_dd,
        "volatility": vol,
        "sharpe": sharpe,
        "total_return": float(values[-1] / principal - 1.0),
        "total_cost": total_cost,
        "turnover_double_sided": float(turnover),
        "cost_drag": float(cost_drag),
        "trades": int(len(trades)),
        "missing_price_value_ratio": (
            float(exceptions.get("missing_price_value_days", 0.0) / (avg_nav * len(values)))
            if avg_nav > 0 and len(values) > 0
            else float("nan")
        ),
        **{f"exception_{k}": int(v) for k, v in exceptions.items()},
    }


def _last_price_between(panel: Panel, idx: int, start: int, end: int) -> Tuple[int, float] | None:
    """只在该月 [start, end] 内找最后可估值，绝不使用未来最终价回填旧月。"""
    for day in range(int(end), int(start) - 1, -1):
        price = panel.close_adj[day, idx]
        if np.isfinite(price) and price > 0:
            return day, float(price)
    return None


def _member_return(panel: Panel, monthly: MonthlyPanel, m: int, idx: int) -> Tuple[float, str]:
    """持有月 m+1 的成员收益；缺价/退市用**当月内**最后可估值，找不到则 unresolved。"""
    ret = monthly.returns[m + 1][idx]
    if np.isfinite(ret):
        return float(ret), "ok"
    prev = int(monthly.month_end_days[m])
    target = int(monthly.month_end_days[m + 1])
    prev_price = panel.close_adj[prev, idx]
    if not np.isfinite(prev_price) or prev_price <= 0:
        return 0.0, "unresolved"
    found = _last_price_between(panel, idx, prev, target)
    if found is None:
        return 0.0, "unresolved"
    day, price = found
    if day == target:
        return float(price / prev_price - 1.0), "ok"
    return float(price / prev_price - 1.0), "stale_or_delisted"


def benchmark_series(
    panel: Panel, monthly: MonthlyPanel, *, start: str = "", end: str = ""
) -> pd.DataFrame:
    """同域等权毛机会成本序列：信号日冻结成员/权重；缺价保留份额并给压力带。"""
    rows = []
    for m in range(11, len(monthly.month_end_days) - 1):
        sig_day = int(monthly.month_end_days[m])
        hold_day = int(monthly.month_end_days[m + 1])
        members = np.flatnonzero(monthly.eligible[m])
        if members.size == 0:
            continue
        weight = 1.0 / members.size
        total = 0.0
        unresolved_weight = 0.0
        statuses = {}
        for idx in members:
            ret, status = _member_return(panel, monthly, m, int(idx))
            total += weight * ret
            statuses[status] = statuses.get(status, 0) + 1
            if status == "unresolved":
                unresolved_weight += weight
        rows.append(
            {
                "signal_date": panel.days[sig_day],
                "hold_date": panel.days[hold_day],
                "gross_return": float(total),
                "members": int(members.size),
                "unresolved_weight": float(unresolved_weight),
                "stress_lo": float(total - unresolved_weight),
                "stress_hi": float(total + unresolved_weight),
                "status": ",".join(f"{k}:{v}" for k, v in sorted(statuses.items())),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["nav"] = (1.0 + frame["gross_return"]).cumprod()
    if start:
        frame = frame[frame["hold_date"] >= start]
    if end:
        frame = frame[frame["hold_date"] <= end]
    return frame.reset_index(drop=True)


def matched_benchmark_series(
    panel: Panel, monthly: MonthlyPanel, stocks: Sequence[str], *, start: str = "", end: str = ""
) -> pd.DataFrame:
    """所选 6 只的完整参照组（L2 / L1 / 全市场）等权收益均值（诊断）。"""
    rows = []
    for m in range(11, len(monthly.month_end_days) - 1):
        sig_day = int(monthly.month_end_days[m])
        hold_day = int(monthly.month_end_days[m + 1])
        selected = top_codes(monthly, stocks, m, TOP_N)
        if not selected:
            continue
        eligible = monthly.eligible[m] & np.isfinite(monthly.returns[m + 1])
        values = []
        for code in selected:
            idx = panel.stock_index[code]
            ref_type = int(monthly.ref_types[m, idx])
            key = int(monthly.ref_keys[m, idx])
            l1 = int(monthly.l1_keys[m, idx])
            if ref_type == 2:
                mask = eligible & (monthly.ref_types[m] == 2) & (monthly.ref_keys[m] == key)
            elif ref_type == 1:
                mask = eligible & (monthly.l1_keys[m] == l1)
            else:
                mask = eligible
            if mask.sum() >= 2:
                values.append(float(np.nanmean(monthly.returns[m + 1][mask])))
        if values:
            rows.append(
                {
                    "signal_date": panel.days[sig_day],
                    "hold_date": panel.days[hold_day],
                    "matched_return": float(np.mean(values)),
                    "selected": len(selected),
                }
            )
    frame = pd.DataFrame(rows)
    if start:
        frame = frame[frame["hold_date"] >= start]
    if end:
        frame = frame[frame["hold_date"] <= end]
    return frame.reset_index(drop=True)


def replay_frozen_holdings(
    panel: Panel, holdings_path: Path, *, principal: float, start: str = "", end: str = ""
) -> Tuple[pd.Series, Dict[str, float]]:
    """T2.5 冻结持仓重放（参考用；不承诺 r3 表三合规）。"""
    raw = pd.read_parquet(holdings_path)
    raw["trade_date"] = raw["trade_date"].astype(str)
    raw = raw.sort_values(["trade_date", "ts_code"])
    by_date = {date: frame for date, frame in raw.groupby("trade_date")}
    dates = sorted(by_date)
    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]
    if not dates:
        return pd.Series(dtype=float), {}
    cash = float(principal)
    positions: Dict[str, dict] = {}
    nav_dates: List[str] = []
    nav_values: List[float] = []
    prev_codes: set[str] = set()
    total_cost = 0.0
    negative_days = 0
    for date in dates:
        day = panel.day_index.get(date)
        if day is None:
            continue
        current: Dict[str, dict] = {}
        for row in by_date[date].itertuples(index=False):
            code = str(row.ts_code)
            idx = panel.stock_index.get(code)
            if idx is None:
                continue
            buy_day = panel.day_index.get(str(row.buy_date), day)
            entry_adj = _safe_price(panel, buy_day, idx, "open_adj")
            if not np.isfinite(entry_adj):
                entry_adj = _safe_price(panel, buy_day, idx, "close_adj")
            current[code] = {
                "shares": float(row.shares),
                "buy_price": float(row.buy_price),
                "entry_adj": entry_adj,
                "buy_day": buy_day,
            }
        for code in prev_codes - set(current):
            pos = positions.get(code)
            if pos is None:
                continue
            idx = panel.stock_index[code]
            price = _safe_price(panel, day, idx, "open_u")
            if not np.isfinite(price):
                price = _safe_price(panel, day, idx, "close_adj")
            if not np.isfinite(price):
                price = pos["buy_price"]
            amount = pos["shares"] * price
            cash += amount - _sell_cost(amount)
            total_cost += _sell_cost(amount)
        for code, pos in current.items():
            if code in prev_codes:
                continue
            amount = pos["shares"] * pos["buy_price"]
            cash -= amount + _buy_cost(amount)
            total_cost += _buy_cost(amount)
        positions = current
        prev_codes = set(current)
        value = cash
        for code, pos in positions.items():
            idx = panel.stock_index[code]
            price = panel.close_adj[day, idx]
            if np.isfinite(price) and pos["entry_adj"] > 0:
                value += pos["shares"] * pos["buy_price"] * (float(price) / pos["entry_adj"])
            else:
                value += pos["shares"] * pos["buy_price"]
        if cash < 0:
            negative_days += 1
        nav_dates.append(date)
        nav_values.append(value)
    nav = pd.Series(nav_values, index=pd.to_datetime(nav_dates, format="%Y%m%d"), name="nav")
    info = {
        "principal": principal,
        "negative_cash_days": negative_days,
        "total_cost": total_cost,
        "dates": len(nav),
        "compliant_with_v31_table3": False,
        "reason": "复权盯市/裸价卖出与划拨未按 r3 总回报账统一；只作参考，不用于表三",
    }
    return nav, info


def combine_rebalanced(
    a_nav: pd.Series,
    c_nav: pd.Series,
    *,
    wa: float = 0.7,
    wc: float = 0.3,
    principal: float = 1_000_000.0,
) -> pd.Series:
    """70/30 月度恢复组合 NAV（NAV 级再平衡；不模拟划拨卖出/费用/整手）。

    r3 明确：不用于表三；若未来使用，必须替换为真实子账本。
    """
    frame = pd.concat([a_nav.rename("a"), c_nav.rename("c")], axis=1).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    frame = frame / frame.iloc[0]
    a_ret = frame["a"].pct_change().fillna(0.0)
    c_ret = frame["c"].pct_change().fillna(0.0)
    values = [principal]
    weight_a, weight_c = wa, wc
    month = frame.index[0].month
    for i in range(1, len(frame)):
        if frame.index[i].month != month:
            weight_a, weight_c = wa, wc
            month = frame.index[i].month
        port = weight_a * a_ret.iloc[i] + weight_c * c_ret.iloc[i]
        values.append(values[-1] * (1.0 + port))
        if 1.0 + port != 0:
            weight_a = weight_a * (1.0 + a_ret.iloc[i]) / (1.0 + port)
            weight_c = 1.0 - weight_a
    return pd.Series(values, index=frame.index, name="a_c1_not_compliant")
