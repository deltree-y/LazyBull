"""原生研究账户的只读守恒检查；不读取外部数据、不承担策略准入判定。"""

# 复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.8）。

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping

import pandas as pd

CASH_TOLERANCE = 1e-6
NAV_TOLERANCE = 1e-10


def _day(value: Any) -> str:
    """统一原生 Timestamp、YYYYMMDD 与 JSON ISO 日期，拒绝空日期。"""
    if value is None or pd.isna(value):
        raise ValueError("账户守恒记录缺少日期")
    text = str(value)
    stamp = (
        pd.to_datetime(text, format="%Y%m%d")
        if len(text) == 8 and text.isdigit()
        else pd.Timestamp(value)
    )
    if pd.isna(stamp):
        raise ValueError(f"账户守恒日期非法：{value!r}")
    return stamp.strftime("%Y%m%d")


def _number(value: Any, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"账户守恒数值必须有限：{context}={value!r}")
    return number


def _shares(value: Any, context: str) -> int:
    number = _number(value, context)
    if not number.is_integer():
        raise ValueError(f"账户股数必须为整数：{context}={value!r}")
    return int(number)


def _positions(value: Mapping[str, Any], context: str) -> dict[str, int]:
    return {str(code): _shares(info["shares"], f"{context}/{code}") for code, info in value.items()}


def _plain(value: Any) -> Any:
    """末端诊断转成可 JSON 保存的内容，不修改原账户。"""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


class _Checks:
    def __init__(self) -> None:
        self.violations: list[dict[str, Any]] = []
        self.max_residual: dict[str, float] = {}

    def require(self, ok: bool, code: str, day: str, **details: Any) -> None:
        if not ok:
            self.violations.append({"check": code, "trade_date": day, **details})

    def equal(
        self, actual: Any, expected: Any, code: str, day: str, tolerance: float = CASH_TOLERANCE
    ) -> None:
        residual = _number(actual, code) - _number(expected, code)
        self.max_residual[code] = max(self.max_residual.get(code, 0.0), abs(residual))
        self.require(
            abs(residual) <= tolerance,
            code,
            day,
            actual=actual,
            expected=expected,
            residual=residual,
        )


def _by_day(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_day(row[field])].append(row)
    return dict(grouped)


def _initial_point(
    result: Mapping[str, Any],
    daily: list[dict[str, Any]],
    states: list[dict[str, Any]],
    capital: float,
    checks: _Checks,
) -> dict[str, Any]:
    """显式初始点优先；缺失时只能使用实际给出的首个纯现金日。"""
    explicit = result.get("initial_point")
    if explicit is None:
        point = {**daily[0], "positions": states[0]["positions"]}
        source = "first_daily_cash_point"
    else:
        point = dict(explicit)
        source = "explicit_initial_point"
    day = _day(point["trade_date"])
    checks.require(day <= _day(daily[0]["trade_date"]), "initial_date_order", day)
    checks.equal(point["cash"], capital, "initial_cash", day)
    checks.equal(point["market_value"], 0.0, "initial_market_value", day)
    checks.equal(point["receivable"], 0.0, "initial_receivable", day)
    checks.equal(point["total_value"], capital, "initial_total_value", day)
    checks.equal(point["nav"], 1.0, "initial_nav", day, NAV_TOLERANCE)
    checks.require(not point["positions"], "initial_positions_empty", day)
    return {"source": source, "trade_date": day, "cash": capital, "positions": {}, "nav": 1.0}


def _corporate_maps(
    corporate: Mapping[str, Any], checks: _Checks
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], dict[str, float]]:
    rows = _by_day(list(corporate.get("rows", [])), "trade_date")
    paid: dict[str, float] = {}
    for key, value in corporate.get("cash_paid_by_day", {}).items():
        day = _day(key)
        if day in paid:
            raise ValueError(f"公司行为到账日期重复：{day}")
        paid[day] = _number(value, f"cash_paid_by_day/{day}")
    accrued: dict[str, float] = {}
    for day in sorted(set(rows) | set(paid)):
        paid_rows = 0.0
        accrued[day] = 0.0
        for row in rows.get(day, []):
            kind = row["kind"]
            if kind == "pay":
                paid_rows += _number(row["cash_paid"], f"{day}/cash_paid")
            elif kind == "ex":
                accrued[day] += _number(row["cash_receivable"], f"{day}/cash_receivable")
            elif kind in {"lock", "unlock"}:
                # 锁股不改变现金、应收和总股份，仍由独立恒等式约束。
                continue
            elif kind == "sell_blocked_locked":
                checks.require(False, "corporate_sell_blocked_locked", day, row=_plain(row))
            else:
                raise ValueError(f"公司行为 kind 不受支持：{day}/{kind}")
        checks.equal(paid.get(day, 0.0), paid_rows, "corporate_cash_paid_rows", day)
        checks.require(
            paid.get(day, 0.0) >= 0.0 and accrued[day] >= 0.0, "corporate_cash_nonnegative", day
        )
    return rows, paid, accrued


def _lock_identity(row: Mapping[str, Any]) -> tuple[tuple[str, str], dict[str, Any]]:
    """事件和股票为键，锁定数量及释放时点必须逐笔对应。"""
    key = (str(row["event_id"]), str(row["ts_code"]))
    return key, {
        "shares": _shares(row["shares"], f"lock/{key}"),
        "recognized_date": _day(row["recognized_date"]),
        "unlock_trade_day": _day(row["unlock_trade_day"]),
        "div_listdate": _day(row["div_listdate"]),
    }


def _apply_locks(
    day: str,
    rows: list[dict[str, Any]],
    locks: dict[tuple[str, str], dict[str, Any]],
    checks: _Checks,
) -> dict[str, int]:
    """按 API 行顺序登记/释放锁；ex 增股由独立股份账处理。"""
    bonus = {
        (str(row["event_id"]), str(row["ts_code"])): _shares(row["bonus_shares"], day)
        for row in rows
        if row["kind"] == "ex" and "event_id" in row
    }
    for row in rows:
        if row["kind"] not in {"lock", "unlock"}:
            continue
        key, value = _lock_identity(row)
        if row["kind"] == "lock":
            checks.require(
                key not in locks, "lock_event_unique", day, event_id=key[0], stock=key[1]
            )
            checks.require(
                value["shares"] > 0
                and value["recognized_date"] == day
                and value["unlock_trade_day"] > day,
                "lock_quantity_and_dates",
                day,
                event_id=key[0],
                lock=value,
            )
            checks.require(
                bonus.get(key) == value["shares"],
                "lock_matches_bonus_event",
                day,
                event_id=key[0],
                shares=value["shares"],
                bonus_shares=bonus.get(key),
            )
            if key not in locks:
                locks[key] = value
        else:
            checks.require(
                locks.get(key) == value,
                "unlock_matches_active_event",
                day,
                event_id=key[0],
                stock=key[1],
                expected=locks.get(key),
                actual=value,
            )
            checks.require(
                day >= value["unlock_trade_day"], "unlock_not_before_listing", day, lock=value
            )
            if locks.get(key) == value:
                del locks[key]
    totals: dict[str, int] = defaultdict(int)
    for (_, code), value in locks.items():
        totals[code] += value["shares"]
        checks.require(
            value["unlock_trade_day"] > day, "due_lock_released", day, stock=code, lock=value
        )
    return dict(totals)


def _check_ending_locks(
    corporate: Mapping[str, Any],
    locks: Mapping[tuple[str, str], dict[str, Any]],
    locked_shares: Mapping[str, int],
    day: str,
    checks: _Checks,
) -> None:
    """末端锁与逐事件重建一致；被守卫阻断的账户不得标记有效。"""
    lock_keys = [
        _lock_identity(row)[0] for row in corporate.get("rows", []) if row["kind"] == "lock"
    ]
    checks.require(len(lock_keys) == len(set(lock_keys)), "lock_event_once_over_history", day)
    recorded = {}
    for row in corporate.get("locks", []):
        key, value = _lock_identity(row)
        checks.require(key not in recorded, "ending_lock_event_unique", day, event_id=key[0])
        recorded[key] = value
    checks.require(recorded == dict(locks), "ending_locks_match_events", day)
    totals = {
        str(code): _shares(value, f"ending_locked/{code}")
        for code, value in corporate.get("locked_shares", {}).items()
    }
    checks.require(
        totals == dict(locked_shares),
        "ending_locked_shares_match_events",
        day,
        actual=totals,
        expected=dict(locked_shares),
    )
    blocked = sum(row["kind"] == "sell_blocked_locked" for row in corporate.get("rows", []))
    declared = _shares(corporate.get("blocked_sell_count", 0), "blocked_sell_count")
    checks.equal(declared, blocked, "blocked_sell_count_matches_rows", day, 0.0)
    checks.require(declared == 0 and blocked == 0, "no_blocked_locked_sales", day, count=declared)


def _renewal_dates(
    events: list[dict[str, Any]],
    states: list[dict[str, Any]],
    trades: Mapping[str, list],
    checks: _Checks,
) -> dict[str, int]:
    """只比较可配对的连续持仓续持事件；缺匹配记录单列，不外推实际持有期。"""
    by_day = {_day(state["trade_date"]): index for index, state in enumerate(states)}
    matched = unmatched = 0
    for event in events:
        if event.get("kind") != "expiry" or not event.get("renew"):
            continue
        day, code = _day(event["trade_date"]), str(event["ts_code"])
        index = by_day.get(day, 0)
        previous = states[index - 1]["positions"].get(code) if index else None
        current = states[index]["positions"].get(code) if day in by_day else None
        traded = any(str(trade["stock"]) == code for trade in trades.get(day, []))
        if not previous or not current or traded or "buy_date" not in event:
            unmatched += 1
            continue
        if "buy_date" not in previous or "buy_date" not in current:
            unmatched += 1
            continue
        matched += 1
        values = [
            _day(value) for value in (previous["buy_date"], current["buy_date"], event["buy_date"])
        ]
        checks.require(
            len(set(values)) == 1, "renewal_preserves_buy_date", day, stock=code, buy_dates=values
        )
    return {"matched_renewals": matched, "unmatched_renewals": unmatched}


def validate_account(result: Mapping[str, Any], initial_capital: float) -> dict[str, Any]:
    """验证逐日现金/股份/应收/总资产恒等式，返回证据；失败时 all_passed=False。

    初始点必须来自 result.initial_point 或首个实际纯现金 daily/state。缺字段、非法日期、
    非有限数值等结构错误直接报错。末端持仓与未决义务合法，不强行要求清仓。
    """
    capital = _number(initial_capital, "initial_capital")
    if capital <= 0:
        raise ValueError("初始资金必须大于零")
    daily, states = list(result["daily"]), list(result["states"])
    if not daily or len(daily) != len(states):
        raise ValueError("daily/states 必须非空且逐日等长")
    days = [_day(row["trade_date"]) for row in daily]
    state_days = [_day(row["trade_date"]) for row in states]
    if days != sorted(set(days)) or days != state_days:
        raise ValueError("daily/states 日期必须逐位一致、唯一且递增")
    checks = _Checks()
    initial = _initial_point(result, daily, states, capital, checks)
    trades = _by_day(list(result["trades"]), "date")
    corporate = result.get("corporate") or {}
    corporate_rows, paid, accrued = _corporate_maps(corporate, checks)
    for source, source_days in [
        ("trades", trades),
        ("corporate_rows", corporate_rows),
        ("cash_paid", paid),
    ]:
        outside = sorted(set(source_days) - set(days))
        checks.require(not outside, "event_date_in_daily", "", source=source, outside_days=outside)
    previous_cash, previous_receivable = capital, 0.0
    share_book: dict[str, int] = {}
    locks: dict[tuple[str, str], dict[str, Any]] = {}
    locked_shares: dict[str, int] = {}
    lock_history: list[dict[str, Any]] = []
    max_positions = 0
    for day, row, state in zip(days, daily, states):
        locked_shares = _apply_locks(day, corporate_rows.get(day, []), locks, checks)
        for action in corporate_rows.get(day, []):
            if action["kind"] == "ex":
                code = str(action["ts_code"])
                added = _shares(action["bonus_shares"], f"{day}/{code}/bonus_shares")
                checks.require(
                    added >= 0, "bonus_shares_nonnegative", day, stock=code, shares=added
                )
                share_book[code] = share_book.get(code, 0) + added
        bought, sold, fees = 0.0, 0.0, 0.0
        for index, trade in enumerate(trades.get(day, [])):
            code, side = str(trade["stock"]), trade["action"]
            shares = _shares(trade["shares"], f"{day}/{index}/shares")
            amount = _number(trade["amount"], f"{day}/{index}/amount")
            cost = _number(trade["cost"], f"{day}/{index}/cost")
            if side not in {"buy", "sell"}:
                raise ValueError(f"原生成交方向非法：{day}/{index}/{side}")
            checks.require(
                shares > 0 and amount >= 0.0 and cost >= 0.0,
                "trade_nonnegative",
                day,
                trade_index=index,
            )
            fees += cost
            if side == "buy":
                bought += amount
                share_book[code] = share_book.get(code, 0) + shares
            else:
                available = share_book.get(code, 0) - locked_shares.get(code, 0)
                checks.require(
                    shares <= available,
                    "sell_within_unlocked_shares",
                    day,
                    stock=code,
                    requested=shares,
                    available=available,
                    locked_shares=locked_shares.get(code, 0),
                )
                sold += amount
                share_book[code] = share_book.get(code, 0) - shares
            checks.require(
                share_book[code] >= 0,
                "shares_after_trade_nonnegative",
                day,
                stock=code,
                shares=share_book[code],
            )
        expected_cash = previous_cash + sold - bought - fees + paid.get(day, 0.0)
        expected_receivable = previous_receivable + accrued.get(day, 0.0) - paid.get(day, 0.0)
        checks.equal(row["cash"], expected_cash, "cash_conservation", day)
        checks.equal(row["receivable"], expected_receivable, "receivable_conservation", day)
        for field, expected in [("buy_amount", bought), ("sell_amount", sold), ("fees", fees)]:
            checks.equal(row[field], expected, f"daily_{field}_vs_trades", day)
        checks.equal(state["cash"], row["cash"], "state_cash_vs_daily", day)
        cash = _number(row["cash"], f"{day}/cash")
        market = _number(row["market_value"], f"{day}/market_value")
        receivable = _number(row["receivable"], f"{day}/receivable")
        checks.equal(row["total_value"], cash + market + receivable, "asset_identity", day)
        checks.equal(
            row["nav"],
            _number(row["total_value"], f"{day}/total_value") / capital,
            "nav_identity",
            day,
            NAV_TOLERANCE,
        )
        checks.require(cash >= -CASH_TOLERANCE, "cash_nonnegative", day, cash=cash)
        checks.require(
            market >= -CASH_TOLERANCE and receivable >= -CASH_TOLERANCE, "assets_nonnegative", day
        )
        actual = _positions(state["positions"], day)
        for code in set(actual) | set(locked_shares):
            locked, total = locked_shares.get(code, 0), actual.get(code, 0)
            checks.require(
                0 <= locked <= total,
                "locked_shares_within_total",
                day,
                stock=code,
                locked_shares=locked,
                total_shares=total,
            )
        lock_history.append({"trade_date": day, "locked_shares": dict(locked_shares)})
        for code in set(share_book) | set(actual):
            expected, held = share_book.get(code, 0), actual.get(code, 0)
            checks.require(
                held == expected,
                "shares_conservation",
                day,
                stock=code,
                actual=held,
                expected=expected,
            )
            checks.require(held >= 0, "shares_nonnegative", day, stock=code, shares=held)
        count = sum(shares > 0 for shares in actual.values())
        max_positions = max(max_positions, count)
        checks.require(count <= 20, "maximum_20_positions", day, count=count)
        if "positions" in row:
            checks.equal(row["positions"], len(actual), "position_count_vs_state", day, 0.0)
        previous_cash, previous_receivable = cash, receivable
    ending = result.get("ending", states[-1])
    checks.equal(ending["cash"], daily[-1]["cash"], "ending_cash", days[-1])
    checks.require(
        _positions(ending["positions"], "ending")
        == _positions(states[-1]["positions"], "last_state"),
        "ending_positions",
        days[-1],
    )
    if corporate:
        checks.equal(
            corporate["receivable"], daily[-1]["receivable"], "ending_receivable", days[-1]
        )
    _check_ending_locks(corporate, locks, locked_shares, days[-1], checks)
    renewal_dates = _renewal_dates(list(result.get("buffer_events", [])), states, trades, checks)
    pending_fields = (
        "pending_signals",
        "pending_condition_sells",
        "pending_stop_loss_sells",
        "unfilled_slots",
    )
    return {
        "valid": not checks.violations,
        "all_passed": not checks.violations,
        "scope": "数值守恒；不构成收益/回撤准入或历史可交易性证明",
        "initial_point": initial,
        "day_count": len(days),
        "trade_count": len(result["trades"]),
        "corporate_row_count": len(corporate.get("rows", [])),
        "buffer_event_count": len(result.get("buffer_events", [])),
        "holding_date_observations": renewal_dates,
        "locked_shares_by_day": lock_history,
        "max_positions": max_positions,
        "tolerances": {"cash_absolute": CASH_TOLERANCE, "nav_absolute": NAV_TOLERANCE, "shares": 0},
        "max_absolute_residual": checks.max_residual,
        "violations": checks.violations,
        "ending": {
            "trade_date": days[-1],
            "cash": float(daily[-1]["cash"]),
            "receivable": float(daily[-1]["receivable"]),
            "positions": _positions(states[-1]["positions"], "ending"),
            "pending": {
                name: _plain(ending.get(name, states[-1].get(name, {}))) for name in pending_fields
            },
            "receivables": _plain(corporate.get("receivables", [])),
            "locks": _plain(corporate.get("locks", [])),
            "locked_shares": dict(locked_shares),
            "open_right_endpoint": True,
            "liquidation_required": False,
        },
    }
