"""快递月经营研究单账户：真实现金、股数及公司行为，不接生产。

调用方冻结月度信号；本模块只执行。T 收盘冻结每槽名义额，T+1 开盘卖出、
收盘买入；失败买单当日作废，失败卖单逐日开盘顺延。分红税沿既有 T4
研究口径，可另施现金分红折扣压力，后者不冒充个人持有期差异税。
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.v2_p7.buffer_native_corporate import NativeCorporateAdapter, NativeCorporateBook
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.trade_status import evaluate_trade_status

FROZEN_COSTS = {
    "commission_rate": 0.0001954,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.0005,
}


class MonthlyAccountError(ValueError):
    """研究输入或账本状态无法可靠执行。"""


def _day(value: Any) -> str:
    """拒绝非 YYYYMMDD 日期，避免日期轴隐式改写。"""
    text = str(value)
    if len(text) != 8 or not text.isdigit():
        raise MonthlyAccountError(f"日期必须为 YYYYMMDD：{value!r}")
    pd.Timestamp(text)
    return text


def _positive(value: Any) -> bool:
    return pd.notna(value) and math.isfinite(float(value)) and float(value) > 0


def run_monthly_account(
    *,
    calendar: Sequence[str],
    quotes: pd.DataFrame,
    signals: pd.DataFrame,
    capital: float,
    corporate_book: NativeCorporateBook,
    arm_name: str,
    codes: Sequence[str],
    dividend_haircut: float = 0.0,
) -> dict[str, Any]:
    """执行冻结四槽信号，返回逐日净值、成交、决策及公司行为证据。

    quotes 包含 trade_date/ts_code/open/close 及五个 clean 状态列：
    is_suspended/is_limit_up/is_limit_down/is_st/tradable；不得在本层重新推算。
    signals 列为 signal_date/ts_code/eligible，每个信号日恰有四股，eligible 为布尔。
    公司行为日历必须等于 calendar；调用方负责公告时间及经营信号的形成。
    """
    account = _MonthlyAccount(
        calendar, quotes, signals, capital, corporate_book, arm_name, codes, dividend_haircut
    )
    return account.run()


class _MonthlyAccount:
    """与 NativeCorporateAdapter 兼容的最小研究状态。"""

    def __init__(
        self,
        calendar: Sequence[str],
        quotes: pd.DataFrame,
        signals: pd.DataFrame,
        capital: float,
        corporate_book: NativeCorporateBook,
        arm_name: str,
        codes: Sequence[str],
        dividend_haircut: float,
        *,
        expected_slots: int = 4,
    ) -> None:
        self.days = tuple(_day(day) for day in calendar)
        self.codes = tuple(sorted(str(code) for code in codes))
        if not self.days or self.days != tuple(sorted(set(self.days))):
            raise MonthlyAccountError("日历必须非空、严格递增且不重复")
        if (
            isinstance(expected_slots, bool)
            or not isinstance(expected_slots, int)
            or expected_slots <= 0
        ):
            raise MonthlyAccountError("预期槽数必须为正整数")
        self.slot_count = expected_slots
        if len(self.codes) != self.slot_count or len(set(self.codes)) != self.slot_count:
            if self.slot_count == 4:
                raise MonthlyAccountError("固定四槽必须提供四个不同股票代码")
            raise MonthlyAccountError(f"固定{self.slot_count}槽必须提供同数不同股票代码")
        if tuple(corporate_book._days) != self.days:
            raise MonthlyAccountError("公司行为日历与账户日历不一致")
        if not _positive(capital) or not 0 <= dividend_haircut <= 1:
            raise MonthlyAccountError("初始资金须正且分红压力折扣须在[0,1]")
        required = {
            "trade_date",
            "ts_code",
            "open",
            "close",
            "is_suspended",
            "is_limit_up",
            "is_limit_down",
            "is_st",
            "tradable",
        }
        if not required.issubset(quotes.columns):
            raise MonthlyAccountError(f"行情缺列：{sorted(required - set(quotes.columns))}")
        data = quotes.copy()
        data["trade_date"] = data["trade_date"].map(_day)
        if data.duplicated(["trade_date", "ts_code"]).any():
            raise MonthlyAccountError("行情日期与代码重复")
        for column in ("is_suspended", "is_limit_up", "is_limit_down", "is_st", "tradable"):
            if not data[column].isin([0, 1, False, True]).all():
                raise MonthlyAccountError(f"clean 状态列含未知值：{column}")
        self.quotes = {
            (str(row["trade_date"]), str(row["ts_code"])): row for row in data.to_dict("records")
        }
        self.signals = self._validate_signals(signals)
        self.current_capital = float(capital)
        self.initial_capital = float(capital)
        self.arm_name = str(arm_name)
        self.positions: dict[str, dict[str, Any]] = {}
        self.corporate = NativeCorporateAdapter(corporate_book)
        self.cost = CostModel(**FROZEN_COSTS)
        self.haircut = float(dividend_haircut)
        self.tax_liability = 0.0
        self.tax_paid = 0.0
        self.total_fees = 0.0
        self.pending_sells: dict[str, dict[str, Any]] = {}
        self.plans: dict[str, list[dict[str, Any]]] = {}
        self.daily: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.holdings: list[dict[str, Any]] = []

    def _validate_signals(self, signals: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        required = {"signal_date", "ts_code", "eligible"}
        if not required.issubset(signals.columns):
            raise MonthlyAccountError(f"信号缺列：{sorted(required - set(signals.columns))}")
        result = {}
        for day, group in signals.groupby("signal_date", sort=True):
            day = _day(day)
            if day not in self.days:
                raise MonthlyAccountError(f"信号日不在日历：{day}")
            if len(group) != self.slot_count or set(group["ts_code"]) != set(self.codes):
                if self.slot_count == 4:
                    raise MonthlyAccountError(f"{day} 信号必须恰有四个固定代码")
                raise MonthlyAccountError(f"{day} 信号必须恰有{self.slot_count}个固定代码")
            if not group["eligible"].map(lambda value: isinstance(value, bool)).all():
                raise MonthlyAccountError(f"{day} eligible 必须显式为布尔，未知不能当 False")
            result[day] = group.sort_values("ts_code").to_dict("records")
        return result

    def _note(self, day: str, code: str, kind: str, **values: Any) -> None:
        self.decisions.append({"trade_date": day, "ts_code": code, "kind": kind, **values})

    def _quote(self, day: str, code: str, side: str) -> tuple[float | None, str]:
        row = self.quotes.get((day, code))
        if row is None:
            return None, "row_absent"
        if side == "buy" and row["is_st"]:
            return None, "ST禁止买入"
        allowed, reason = evaluate_trade_status(row, action=side, require_tradable=side == "buy")
        if not allowed:
            return None, str(reason)
        field = "open" if side == "sell" else "close"
        if not _positive(row[field]):
            return None, f"no_valid_{field}"
        return float(row[field]), "ok"

    def _corporate_day(self, day: str) -> None:
        start = len(self.corporate.rows)
        self.corporate.before_day(self, day)
        for row in self.corporate.rows[start:]:
            if row["kind"] == "ex":
                self.tax_liability += float(row["cash_receivable"]) * self.haircut
                code = str(row["ts_code"])
                if code in self.pending_sells and row["bonus_shares"]:
                    order = self.pending_sells[code]
                    order["shares"] = math.floor(
                        order["shares"] * (1.0 + float(row["share_ratio"]))
                    )
            elif row["kind"] == "pay":
                tax = float(row["cash_paid"]) * self.haircut
                self.current_capital -= tax
                self.tax_liability -= tax
                self.tax_paid += tax
        if abs(self.tax_liability) < 1e-9:
            self.tax_liability = 0.0

    def _queue_sells(self, day: str, plans: Sequence[dict[str, Any]]) -> None:
        for plan in plans:
            code = plan["ts_code"]
            current = int(self.positions.get(code, {}).get("shares", 0))
            if not current:
                continue
            if not plan["eligible"]:
                requested = current
            else:
                row = self.quotes.get((day, code))
                if row is None or not _positive(row["open"]):
                    self._note(day, code, "rebalance_sell_not_estimable", reason="no_valid_open")
                    continue
                target = int(plan["notional"] / float(row["open"]) / 100) * 100
                requested = max(0, current - target)
            if code in self.pending_sells:
                # 新信号可扩大旧义务，但不能撤回此前尚未完成的卖出。
                order = self.pending_sells[code]
                order["shares"] = max(order["shares"], requested)
                self._note(day, code, "prior_sell_obligation_kept", shares=order["shares"])
                continue
            if requested:
                self.pending_sells[code] = {
                    "shares": requested,
                    "signal_date": plan["signal_date"],
                    "created": day,
                }

    def _sell(self, day: str) -> None:
        for code in sorted(tuple(self.pending_sells)):
            order = self.pending_sells[code]
            position = self.positions.get(code)
            if position is None:
                raise MonthlyAccountError(f"{day} {code} 卖出义务缺持仓")
            price, reason = self._quote(day, code, "sell")
            if price is None:
                self._note(day, code, "sell_blocked", reason=reason, shares=order["shares"])
                continue
            if _day(position["last_buy_date"]) >= day:
                self._note(day, code, "sell_blocked", reason="t_plus_one")
                continue
            sellable = self.corporate.sellable_shares(self, code)
            quantity = min(int(order["shares"]), int(position["shares"]), sellable)
            if quantity <= 0:
                self._note(day, code, "sell_blocked", reason="bonus_locked")
                continue
            amount = quantity * price
            fee = self.cost.calculate_sell_cost(amount)
            self.current_capital += amount - fee
            self.total_fees += fee
            self.trades.append(self._trade(day, code, "sell", quantity, price, fee, order))
            before = int(position["shares"])
            position["shares"] -= quantity
            position["buy_cost_cash"] *= position["shares"] / before
            order["shares"] -= quantity
            if not position["shares"]:
                del self.positions[code]
            if not order["shares"]:
                del self.pending_sells[code]

    def _trade(
        self,
        day: str,
        code: str,
        side: str,
        shares: int,
        price: float,
        fee: float,
        source: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "trade_date": day,
            "signal_date": source["signal_date"],
            "ts_code": code,
            "side": side,
            "shares": shares,
            "price": price,
            "amount": shares * price,
            "fee": fee,
            "price_kind": "open" if side == "sell" else "close",
            "cash_after": self.current_capital,
        }

    def _buy(self, day: str, plans: Sequence[dict[str, Any]]) -> None:
        eligible = []
        for plan in plans:
            code = plan["ts_code"]
            if not plan["eligible"] or code in self.pending_sells:
                continue
            price, reason = self._quote(day, code, "buy")
            if price is None:
                self._note(day, code, "buy_voided", reason=reason)
                continue
            held = int(self.positions.get(code, {}).get("shares", 0))
            budget = max(0.0, float(plan["notional"]) - held * price)
            eligible.append((plan, price, budget))
        total = sum(item[2] for item in eligible)
        scale = min(1.0, self.current_capital / total) if total > 0 else 1.0
        for plan, price, budget in eligible:
            budget = min(budget * scale, self.current_capital)
            quantity = int(budget / price / 100) * 100
            while (
                quantity
                and quantity * price + self.cost.calculate_buy_cost(quantity * price) > budget
            ):
                quantity -= 100
            if not quantity:
                self._note(day, plan["ts_code"], "buy_voided", reason="cash_or_lot", budget=budget)
                continue
            self._fill_buy(day, plan, price, quantity)

    def _fill_buy(self, day: str, plan: Mapping[str, Any], price: float, quantity: int) -> None:
        code = str(plan["ts_code"])
        amount = quantity * price
        fee = self.cost.calculate_buy_cost(amount)
        self.current_capital -= amount + fee
        self.total_fees += fee
        old = self.positions.get(code)
        if old:
            old["buy_trade_price"] = (old["buy_trade_price"] * old["shares"] + amount) / (
                old["shares"] + quantity
            )
            old["shares"] += quantity
            old["buy_cost_cash"] += amount + fee
            old["last_buy_date"] = day
        else:
            self.positions[code] = {
                "shares": quantity,
                "buy_date": day,
                "last_buy_date": day,
                "signal_date": str(plan["signal_date"]),
                "buy_trade_price": price,
                "buy_cost_cash": amount + fee,
                "last_known_price": price,
            }
        self.trades.append(self._trade(day, code, "buy", quantity, price, fee, plan))

    def _close(self, day: str, day_trades: Sequence[Mapping[str, Any]]) -> float:
        market_value = 0.0
        for code, position in sorted(self.positions.items()):
            row = self.quotes.get((day, code))
            if row is not None and _positive(row["close"]):
                price = float(row["close"])
                position["last_known_price"] = price
                position["last_price_date"] = day
                source = "day_close"
            else:
                price = float(position["last_known_price"])
                source = "last_known_adjusted_for_corporate"
            value = int(position["shares"]) * price
            market_value += value
            self.holdings.append(
                {
                    "trade_date": day,
                    "ts_code": code,
                    "shares": int(position["shares"]),
                    "price": price,
                    "market_value": value,
                    "valuation_source": source,
                    "locked_shares": self.corporate.locked_shares(code),
                }
            )
        equity = (
            self.current_capital + market_value + self.corporate.receivable - self.tax_liability
        )
        if self.current_capital < -1e-8 or not _positive(equity) or self.tax_liability < -1e-8:
            raise MonthlyAccountError(
                f"{day} 非法现金/权益/税负债：{self.current_capital}/{equity}/{self.tax_liability}"
            )
        self.daily.append(
            {
                "trade_date": day,
                "cash": self.current_capital,
                "market_value": market_value,
                "receivable": self.corporate.receivable,
                "dividend_haircut_liability": self.tax_liability,
                "equity": equity,
                "total_value": equity,
                "nav": equity / self.initial_capital,
                "buy_amount": sum(row["amount"] for row in day_trades if row["side"] == "buy"),
                "sell_amount": sum(row["amount"] for row in day_trades if row["side"] == "sell"),
                "fees": sum(row["fee"] for row in day_trades),
                "fees_cumulative": self.total_fees,
                "dividend_haircut_paid": self.tax_paid,
                "holding_count": len(self.positions),
            }
        )
        return equity

    def _plan(self, day: str, index: int, equity: float) -> None:
        rows = self.signals.get(day, [])
        if not rows:
            return
        if index + 1 >= len(self.days):
            self._note(day, "", "right_censored_signal")
            return
        execution = self.days[index + 1]
        self.plans[execution] = [
            {
                "signal_date": day,
                "ts_code": row["ts_code"],
                "eligible": row["eligible"],
                "notional": equity / self.slot_count if row["eligible"] else 0.0,
            }
            for row in rows
        ]
        for plan in self.plans[execution]:
            self._note(
                day,
                plan["ts_code"],
                "planned",
                execution_date=execution,
                signal_date=day,
                eligible=plan["eligible"],
                notional=plan["notional"],
            )

    def run(self) -> dict[str, Any]:
        for index, day in enumerate(self.days):
            trade_start = len(self.trades)
            self._corporate_day(day)
            plans = self.plans.get(day, [])
            self._queue_sells(day, plans)
            self._sell(day)
            self._buy(day, plans)
            equity = self._close(day, self.trades[trade_start:])
            self.corporate.after_day(self, day)
            self._plan(day, index, equity)
        return {
            "arm_name": self.arm_name,
            "initial_capital": self.initial_capital,
            "daily": pd.DataFrame(self.daily),
            "trades": pd.DataFrame(
                self.trades,
                columns=[
                    "trade_date",
                    "signal_date",
                    "ts_code",
                    "side",
                    "shares",
                    "price",
                    "amount",
                    "fee",
                    "price_kind",
                    "cash_after",
                ],
            ),
            "decisions": (
                pd.DataFrame(self.decisions)
                if self.decisions
                else pd.DataFrame(columns=["trade_date", "ts_code", "kind"])
            ),
            "holdings": pd.DataFrame(
                self.holdings,
                columns=[
                    "trade_date",
                    "ts_code",
                    "shares",
                    "price",
                    "market_value",
                    "valuation_source",
                    "locked_shares",
                ],
            ),
            "corporate": self.corporate.payload(),
            "final_state": {
                "cash": self.current_capital,
                "positions": copy.deepcopy(self.positions),
                "pending_sells": copy.deepcopy(self.pending_sells),
                "receivable": self.corporate.receivable,
                "dividend_haircut_liability": self.tax_liability,
            },
            "assumptions": {
                "costs": dict(FROZEN_COSTS),
                "slot_count": self.slot_count,
                "dividend_haircut": self.haircut,
                "dividend_haircut_basis": "额外扣减既有 T4 口径已入账红利的压力，非精确个人税",
                "cash_interest": 0.0,
                "not_production": True,
                "historical_only": True,
            },
        }
