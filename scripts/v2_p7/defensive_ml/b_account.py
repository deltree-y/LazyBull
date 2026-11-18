"""B 袖真实现金账：股数、费用、现金分红、送转、锁股与到期卖出。

口径（协议 §9）：
- 成交用**未复权**真实价格与真实股数；`adj` 只用于特征/标签，禁止在复权净值上再加分红；
- 佣金 `max(rate*amount, 5)`、卖出印花税、双边滑点，逐单核算；
- 现金分红按公司行为桥的税后口径直接入账，不重复扣税；
- 送转新股 `floor(shares*ratio)`，**次一交易日才可卖**（锁股），不增加当日可卖量。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from scripts.v2_p7.defensive_ml import spec


class BAccountError(RuntimeError):
    """账户层错误（超卖、现金为负、守恒破坏）。"""


@dataclass
class Lot:
    """单笔买入批次。"""

    lot_id: int
    ts_code: str
    shares: int
    buy_date: str
    tranche: int
    locked: dict[str, int] = field(default_factory=dict)  # 可卖日 -> 股数

    def sellable(self, day: str) -> int:
        blocked = sum(v for k, v in self.locked.items() if k > day)
        return max(0, self.shares - blocked)

    def release(self, day: str) -> None:
        for key in [k for k in self.locked if k <= day]:
            self.locked.pop(key, None)


@dataclass
class BAccount:
    """现金 + 批次持股 + 应收。"""

    capital: float
    arm: str
    cash: float = 0.0
    receivable: float = 0.0
    lots: dict[int, Lot] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    daily: list[dict[str, Any]] = field(default_factory=list)
    dividends: list[dict[str, Any]] = field(default_factory=list)
    blocked_sells: list[dict[str, Any]] = field(default_factory=list)
    pending_dividends: dict[str, float] = field(default_factory=dict)
    pending_pay_days: dict[str, str] = field(default_factory=dict)
    _next_lot: int = 1

    def __post_init__(self) -> None:
        self.cash = float(self.capital)

    # ---- 费用 -------------------------------------------------------------
    @staticmethod
    def _commission(amount: float) -> float:
        return max(
            float(spec.COSTS["commission_rate"]) * amount,
            float(spec.COSTS["min_commission"]),
        )

    def _buy_cost(self, amount: float) -> float:
        return self._commission(amount)

    def _sell_cost(self, amount: float) -> float:
        stamp = float(spec.COSTS["stamp_tax"]) * amount
        return self._commission(amount) + stamp

    # ---- 交易 -------------------------------------------------------------
    def buy(self, day: str, code: str, shares: int, price: float, tranche: int,
            slippage: float | None = None) -> dict[str, Any]:
        """收盘买入（含滑点与佣金），现金为硬上限。"""
        if shares <= 0:
            raise BAccountError("买入股数必须为正")
        slip = float(spec.COSTS["slippage"]) if slippage is None else float(slippage)
        fill_price = float(price) * (1.0 + slip)
        amount = fill_price * shares
        fee = self._buy_cost(amount)
        if amount + fee > self.cash + 1e-9:
            raise BAccountError(f"现金不足：{code} 需 {amount + fee:.2f}，可用 {self.cash:.2f}")
        self.cash -= amount + fee
        lot = Lot(self._next_lot, code, int(shares), day, tranche)
        self.lots[lot.lot_id] = lot
        self._next_lot += 1
        record = {
            "trade_date": day, "ts_code": code, "action": "buy", "shares": int(shares),
            "price": fill_price, "amount": amount, "fee": fee, "tranche": tranche,
            "reason": "scheduled_entry", "lot_id": lot.lot_id,
        }
        self.trades.append(record)
        return record

    def sell(self, day: str, lot_id: int, shares: int, price: float, reason: str,
             slippage: float | None = None) -> dict[str, Any]:
        """开盘卖出（含滑点、佣金、印花税）。"""
        lot = self.lots.get(lot_id)
        if lot is None:
            raise BAccountError(f"未知批次 {lot_id}")
        if shares <= 0 or shares > lot.sellable(day):
            raise BAccountError(f"可卖不足：{lot.ts_code} 申请 {shares}，可卖 {lot.sellable(day)}")
        slip = float(spec.COSTS["slippage"]) if slippage is None else float(slippage)
        fill_price = float(price) * (1.0 - slip)
        amount = fill_price * shares
        fee = self._sell_cost(amount)
        self.cash += amount - fee
        lot.shares -= shares
        record = {
            "trade_date": day, "ts_code": lot.ts_code, "action": "sell", "shares": int(shares),
            "price": fill_price, "amount": amount, "fee": fee, "tranche": lot.tranche,
            "reason": reason, "lot_id": lot.lot_id,
        }
        self.trades.append(record)
        if lot.shares == 0:
            self.lots.pop(lot_id, None)
        return record

    # ---- 公司行为 ---------------------------------------------------------
    def apply_ex_date(self, day: str, events: Iterable[Any], next_day: str) -> None:
        """除权日：现金分红计应收（按事件挂账）、送转加股并锁到次一交易日。"""
        held = self._shares_by_code()
        for event in events:
            code = str(getattr(event, "ts_code"))
            shares = held.get(code, 0)
            if shares <= 0:
                continue
            cash_per_share = float(getattr(event, "cash_per_share", 0.0) or 0.0)
            ratio = float(getattr(event, "share_ratio", 0.0) or 0.0)
            if cash_per_share > 0:
                amount = cash_per_share * shares
                self.receivable += amount
                pay_day = str(
                    getattr(event, "pay_trade_day", "")
                    or getattr(event, "pay_date", "")
                    or day
                )
                key = str(getattr(event, "end_date", "")) + "|" + code
                self.pending_dividends[key] = amount
                self.pending_pay_days[key] = pay_day
            if ratio > 0:
                bonus = int(shares * ratio)
                if bonus > 0:
                    self._add_bonus(code, bonus, next_day)
            self.dividends.append(
                {
                    "trade_date": day, "ts_code": code, "kind": "ex_date",
                    "shares": shares, "cash_per_share": cash_per_share,
                    "share_ratio": ratio,
                }
            )

    def _add_bonus(self, code: str, shares: int, sellable_from: str) -> None:
        """送转：按批次原股数比例加股，锁到次一交易日（不可当日回转）。"""
        targets = sorted(
            [lot for lot in self.lots.values() if lot.ts_code == code], key=lambda lot: lot.lot_id
        )
        if not targets:
            return
        total = sum(lot.shares for lot in targets)
        if total <= 0:
            return
        assigned = 0
        for index, lot in enumerate(targets):
            portion = (
                shares - assigned
                if index == len(targets) - 1
                else int(shares * lot.shares / total)
            )
            portion = max(0, min(portion, shares - assigned))
            if portion == 0:
                continue
            lot.shares += portion
            lot.locked[sellable_from] = lot.locked.get(sellable_from, 0) + portion
            assigned += portion

    def settle_due(self, day: str) -> float:
        """到账日：把 pay_day 已到的应收转入现金，返回本次到账金额。"""
        cash_in = 0.0
        for key in [k for k, pay in self.pending_pay_days.items() if pay <= day]:
            amount = self.pending_dividends.pop(key, 0.0)
            self.pending_pay_days.pop(key, None)
            if amount <= 0:
                continue
            paid = min(amount, self.receivable)
            self.receivable -= paid
            self.cash += paid
            cash_in += paid
            code = key.split("|")[-1]
            self.dividends.append(
                {"trade_date": day, "ts_code": code, "kind": "pay_date", "amount": paid}
            )
        return cash_in

    def apply_pay_date(self, day: str, events: Iterable[Any]) -> float:
        """派息日：把已挂账的应收转入现金，返回本次到账金额。"""
        cash_in = 0.0
        for event in events:
            code = str(getattr(event, "ts_code"))
            key = str(getattr(event, "end_date", "")) + "|" + code
            amount = self.pending_dividends.pop(key, 0.0)
            if amount <= 0:
                continue
            paid = min(amount, self.receivable)
            self.receivable -= paid
            self.cash += paid
            cash_in += paid
            self.dividends.append(
                {"trade_date": day, "ts_code": code, "kind": "pay_date", "amount": paid}
            )
        return cash_in

    # ---- 估值与守恒 -------------------------------------------------------
    def _shares_by_code(self) -> dict[str, int]:
        held: dict[str, int] = {}
        for lot in self.lots.values():
            held[lot.ts_code] = held.get(lot.ts_code, 0) + lot.shares
        return held

    def shares(self) -> dict[str, int]:
        return self._shares_by_code()

    def market_value(self, day: str, prices: Mapping[str, float]) -> tuple[float, list[str]]:
        """按当日真实收盘估值；缺价记陈旧并沿上一价（由驱动层维护）。"""
        total = 0.0
        missing: list[str] = []
        for code, shares in self._shares_by_code().items():
            price = prices.get(code)
            if price is None or price <= 0:
                missing.append(code)
                continue
            total += price * shares
        return total, missing

    def snapshot(self, day: str, prices: Mapping[str, float], stale_value: float,
                 fees_total: float) -> dict[str, Any]:
        """逐日记录（含守恒核对）。"""
        value, missing = self.market_value(day, prices)
        total = self.cash + value + self.receivable
        held = self._shares_by_code()
        record = {
            "trade_date": day,
            "arm": self.arm,
            "cash": self.cash,
            "market_value": value,
            "stale_market_value": stale_value,
            "receivable": self.receivable,
            "total_value": total,
            "positions": len(held),
            "fees_cum": fees_total,
            "missing_price_count": len(missing),
        }
        self.daily.append(record)
        return record

    def conservation(self) -> dict[str, float]:
        """现金 + 市值 + 应收 与逐日总值的对账（供验证层使用）。"""
        fees = sum(record["fee"] for record in self.trades)
        buys = sum(record["amount"] for record in self.trades if record["action"] == "buy")
        sells = sum(record["amount"] for record in self.trades if record["action"] == "sell")
        return {
            "capital": float(self.capital),
            "cash": self.cash,
            "receivable": self.receivable,
            "fees": fees,
            "buy_amount": buys,
            "sell_amount": sells,
            "shares": float(sum(self._shares_by_code().values())),
        }


def release_locks(accounts: Sequence[BAccount], day: str) -> None:
    """解除已到可卖日的锁股。"""
    for account in accounts:
        for lot in account.lots.values():
            lot.release(day)
