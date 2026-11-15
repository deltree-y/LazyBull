# -*- coding: utf-8 -*-
"""实际现金/整股/应收账与订单状态机（P2 共用计量参照）。

硬语义：
- **不是总回报代理账**：现金按分、持仓按实际股数、应收按面值分别记账；
  ``nav = cash + receivable + Σ shares × 当日可得价格``；
- 买入必须整手、现金不足拒绝；卖出以股数为单位，禁止裸卖；
- 订单完成（remaining=0）与持仓清空（shares=0）分列，部分成交后订单仍在 pending；
- 估值只接受当日快照；缺价保留最后有效价并逐日登记，不用未来价回填；
- 普通执行状态逐条记录，不因单笔未成交冻结账户或整批。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal
from enum import Enum

from scripts.v2_p7.p2.missing import (
    MissingPriceLedger,
    MissingPriceRecord,
    calendar_days_between,
)
from scripts.v2_p7.p2.types import (
    InstrumentError,
    MarketSnapshot,
    PriceStatus,
    money,
    normalize_date,
    price,
    share_count,
)

ZERO = Decimal("0.00")


class AccountError(InstrumentError):
    """账户结构/时间轴错误。"""


class InsufficientCash(AccountError):
    """买入现金不足（零副作用拒绝）。"""


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class PositionView:
    ts_code: str
    shares: int
    cost_basis: Decimal
    last_price: Decimal
    last_price_date: str
    missing_streak_days: int


@dataclass(slots=True)
class Position:
    ts_code: str
    shares: int
    cost_basis: Decimal
    last_price: Decimal
    last_price_date: str
    missing_streak_days: int = 0
    buy_dates: dict[str, int] = field(default_factory=dict)

    def sellable_shares(self, day: str) -> int:
        return sum(qty for buy_date, qty in self.buy_dates.items() if buy_date < day)


@dataclass(slots=True)
class Order:
    order_id: str
    ts_code: str
    side: OrderSide
    quantity: int
    signal_date: str
    earliest_trade_date: str
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    reason: str = ""

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def completed(self) -> bool:
        return self.remaining_quantity == 0

    def summary(self) -> dict:
        return {
            "order_id": self.order_id,
            "ts_code": self.ts_code,
            "side": self.side.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "signal_date": self.signal_date,
            "earliest_trade_date": self.earliest_trade_date,
            "status": self.status.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Receivable:
    receivable_id: str
    ts_code: str
    amount: Decimal
    recognized_date: str
    settle_date: str


@dataclass(frozen=True, slots=True)
class FillResult:
    order_id: str
    fill_quantity: int
    remaining_quantity: int
    order_completed: bool
    position_closed: bool
    position_shares_after: int
    cash_after: Decimal


@dataclass(frozen=True, slots=True)
class AccountMark:
    as_of: str
    cash: Decimal
    receivable: Decimal
    market_value: Decimal
    nav: Decimal
    unresolved_weight: float
    missing_records: tuple[MissingPriceRecord, ...]


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    as_of: str
    cash: Decimal
    receivable: Decimal
    market_value: Decimal
    nav: Decimal
    unresolved_weight: float
    executable_cash_account: bool = True
    cash_flow_checked: bool = True
    account_basis: str = "executable_cash_shares"


@dataclass(frozen=True, slots=True)
class AccountFill:
    fill_id: str
    order_id: str
    trade_date: str
    ts_code: str
    side: OrderSide
    shares: int
    trade_price: Decimal
    amount: Decimal
    fee: Decimal
    order_completed: bool
    position_closed: bool


@dataclass(frozen=True, slots=True)
class ShareAdjustment:
    adjustment_id: str
    ts_code: str
    trade_date: str
    share_ratio: Decimal
    shares_before: int
    shares_after: int


class ResearchAccount:
    """单账户实际现金/整股账（只做多、非杠杆）。"""

    def __init__(
        self,
        principal: object,
        *,
        buy_lot_size: int = 100,
        missing_ledger: MissingPriceLedger | None = None,
    ) -> None:
        initial = money(principal)
        if initial < 0:
            raise AccountError(f"本金不得为负: {initial}")
        if isinstance(buy_lot_size, bool) or not isinstance(buy_lot_size, int) or buy_lot_size <= 0:
            raise AccountError(f"buy_lot_size 必须为正整数: {buy_lot_size!r}")
        self._cash = initial
        self._buy_lot_size = buy_lot_size
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}
        self._active_orders: dict[str, Order] = {}
        self._order_by_code_side: dict[tuple[str, OrderSide], str] = {}
        self._receivables: dict[str, Receivable] = {}
        self._receivable_ids: set[str] = set()
        self._fills: list[AccountFill] = []
        self._share_adjustments: list[ShareAdjustment] = []
        self._missing_ledger = missing_ledger or MissingPriceLedger()
        self._last_mark_date: str | None = None
        self._last_snapshot: AccountSnapshot | None = None

    @property
    def cash(self) -> Decimal:
        return self._cash

    @property
    def receivable(self) -> Decimal:
        return sum((item.amount for item in self._receivables.values()), ZERO)

    @property
    def missing_ledger(self) -> MissingPriceLedger:
        return self._missing_ledger

    @property
    def fills(self) -> tuple[AccountFill, ...]:
        return tuple(self._fills)

    @property
    def position_codes(self) -> tuple[str, ...]:
        """当前持仓代码（只读，不暴露可变持仓对象）。"""
        return tuple(sorted(self._positions))

    @property
    def share_adjustments(self) -> tuple[ShareAdjustment, ...]:
        return tuple(self._share_adjustments)

    def apply_stock_dividend(
        self,
        adjustment_id: str,
        ts_code: str,
        share_ratio: object,
        trade_date: object,
        cash_per_share: object = 0,
    ) -> ShareAdjustment:
        """按 ``stk_div`` 增加股数，并按除权价调整缺价 carry。

        无当日新行情时，``last_price`` 必须按理论除权价
        ``(last_price - cash_per_share) / (1 + stk_div)`` 调整；否则
        「旧价 × 新股数」会制造机械 NAV 增值。
        """
        key = str(adjustment_id).strip()
        if not key or any(item.adjustment_id == key for item in self._share_adjustments):
            raise AccountError(f"送转调整 ID 为空或重复: {adjustment_id!r}")
        code = str(ts_code).strip().upper()
        position = self._positions.get(code)
        if position is None:
            raise AccountError(f"送转调整无持仓: {code}")
        day = normalize_date(trade_date)
        if self._last_mark_date is not None and day <= self._last_mark_date:
            raise AccountError(
                f"送转调整日 {day} 不晚于最后估值日 {self._last_mark_date}，禁止倒填"
            )
        if (code, OrderSide.SELL) in self._order_by_code_side:
            raise AccountError(f"{code} 有活动卖单，禁止直接调整股数")
        try:
            ratio = Decimal(str(share_ratio))
            cash = Decimal(str(cash_per_share or 0))
        except Exception as exc:
            raise AccountError(
                f"送转比例/现金不可解析: {share_ratio!r}/{cash_per_share!r}"
            ) from exc
        if not ratio.is_finite() or ratio < 0:
            raise AccountError(f"送转比例必须为非负有限数: {share_ratio!r}")
        if not cash.is_finite() or cash < 0:
            raise AccountError(f"每股现金必须为非负有限数: {cash_per_share!r}")
        old_shares = position.shares
        new_shares = int(
            (Decimal(old_shares) * (Decimal(1) + ratio)).to_integral_value(rounding=ROUND_FLOOR)
        )
        if new_shares < old_shares:
            raise AccountError(f"送转后股数不得减少: {old_shares} -> {new_shares}")
        gross = position.last_price - cash
        if gross <= 0:
            raise AccountError(f"{code} 除权后价格非正: {position.last_price} - {cash}")
        position.last_price = price(gross / (Decimal(1) + ratio))
        position.last_price_date = day
        added = new_shares - old_shares
        position.shares = new_shares
        if added > 0:
            position.buy_dates[day] = position.buy_dates.get(day, 0) + added
        item = ShareAdjustment(
            adjustment_id=key,
            ts_code=code,
            trade_date=day,
            share_ratio=ratio,
            shares_before=old_shares,
            shares_after=new_shares,
        )
        self._share_adjustments.append(item)
        return item

    def adjust_last_price_for_cash_dividend(
        self,
        ts_code: str,
        cash_per_share: object,
        trade_date: object,
    ) -> None:
        """纯现金分红且当日无新行情时，把 carry 价下调到除息价。"""
        code = str(ts_code).strip().upper()
        position = self._positions.get(code)
        if position is None:
            raise AccountError(f"现金分红调整无持仓: {code}")
        day = normalize_date(trade_date)
        if self._last_mark_date is not None and day <= self._last_mark_date:
            raise AccountError(f"现金分红调整日 {day} 不晚于最后估值日")
        cash = Decimal(str(cash_per_share))
        if not cash.is_finite() or cash <= 0:
            raise AccountError(f"每股现金必须为正有限数: {cash_per_share!r}")
        if position.last_price <= cash:
            raise AccountError(f"{code} 除息后价格非正: {position.last_price} - {cash}")
        position.last_price = price(position.last_price - cash)
        position.last_price_date = day

    def position(self, ts_code: str) -> PositionView | None:
        item = self._positions.get(str(ts_code).strip().upper())
        if item is None:
            return None
        return PositionView(
            ts_code=item.ts_code,
            shares=item.shares,
            cost_basis=item.cost_basis,
            last_price=item.last_price,
            last_price_date=item.last_price_date,
            missing_streak_days=item.missing_streak_days,
        )

    def positions(self) -> tuple[PositionView, ...]:
        return tuple(
            self.position(code) for code in sorted(self._positions) if code in self._positions
        )

    def active_orders(self) -> tuple[dict, ...]:
        return tuple(self._active_orders[key].summary() for key in sorted(self._active_orders))

    def order(self, order_id: str) -> dict:
        item = self._orders.get(str(order_id))
        if item is None:
            raise AccountError(f"未知订单: {order_id}")
        return item.summary()

    def add_receivable(
        self,
        receivable_id: str,
        ts_code: str,
        amount: object,
        recognized_date: object,
        settle_date: object,
    ) -> Receivable:
        """登记现金红利/合并对价等应收；结算日到达后自动转入现金。"""
        key = str(receivable_id).strip()
        if not key or key in self._receivable_ids:
            raise AccountError(f"应收 ID 为空或历史重复: {receivable_id!r}")
        recognized = normalize_date(recognized_date)
        settle = normalize_date(settle_date)
        if settle < recognized:
            raise AccountError(f"应收结算日早于确认日: {recognized} > {settle}")
        if self._last_mark_date is not None and recognized <= self._last_mark_date:
            raise AccountError(
                f"应收确认日 {recognized} 不晚于最后估值日 {self._last_mark_date}，禁止倒填"
            )
        value = money(amount)
        if value <= 0:
            raise AccountError(f"应收金额必须为正: {value}")
        item = Receivable(
            receivable_id=key,
            ts_code=str(ts_code).strip().upper(),
            amount=value,
            recognized_date=recognized,
            settle_date=settle,
        )
        self._receivables[key] = item
        self._receivable_ids.add(key)
        return item

    def submit_order(
        self,
        order_id: str,
        ts_code: str,
        side: OrderSide | str,
        quantity: int,
        signal_date: object,
        earliest_trade_date: object,
    ) -> Order:
        """登记计划订单；不移动现金/股数，成交必须显式调用 execute_fill。"""
        key = str(order_id).strip()
        if not key or key in self._orders:
            raise AccountError(f"订单 ID 为空或重复: {order_id!r}")
        code = str(ts_code).strip().upper()
        if not code:
            raise AccountError("订单 ts_code 不得为空")
        order_side = OrderSide(side)
        qty = share_count(quantity, field="order quantity")
        if qty <= 0:
            raise AccountError(f"订单数量必须为正: {qty}")
        if order_side is OrderSide.BUY and qty % self._buy_lot_size:
            raise AccountError(f"买入数量必须为 {self._buy_lot_size} 股整数倍: {qty}")
        signal = normalize_date(signal_date)
        earliest = normalize_date(earliest_trade_date)
        if earliest <= signal:
            raise AccountError(f"成交最早日 {earliest} 必须晚于信号日 {signal}（T+1）")
        active_key = (code, order_side)
        if active_key in self._order_by_code_side:
            raise AccountError(f"{code} 已有活动 {order_side.value} 订单，禁止重复排队")
        if order_side is OrderSide.SELL:
            position = self._positions.get(code)
            if position is None or position.shares < qty:
                current = 0 if position is None else position.shares
                raise AccountError(f"卖出 {code} {qty} 股超过持仓 {current} 股")
        order = Order(
            order_id=key,
            ts_code=code,
            side=order_side,
            quantity=qty,
            signal_date=signal,
            earliest_trade_date=earliest,
        )
        self._orders[key] = order
        self._active_orders[key] = order
        self._order_by_code_side[active_key] = key
        return order

    def execute_fill(
        self,
        order_id: str,
        *,
        trade_date: object,
        quantity: int,
        trade_price: object,
        fee: object = 0,
        fill_id: str | None = None,
    ) -> FillResult:
        """按显式成交事实入账；拒绝未来回填/超量/现金不足，失败零副作用。"""
        order = self._active_orders.get(str(order_id))
        if order is None:
            raise AccountError(f"订单不在 pending 集合: {order_id}")
        day, qty, unit_price, amount, cost_fee, resolved_fill_id = self._prepare_fill(
            order,
            trade_date=trade_date,
            quantity=quantity,
            trade_price=trade_price,
            fee=fee,
            fill_id=fill_id,
        )
        position = self._positions.get(order.ts_code)
        if order.side is OrderSide.BUY:
            position = self._apply_buy_fill(order, position, day, qty, unit_price, amount, cost_fee)
            closed = False
        else:
            position, closed = self._apply_sell_fill(
                position, order.ts_code, qty, amount, cost_fee, day
            )
        order.filled_quantity += qty
        order_completed = order.completed
        if order_completed:
            self._complete_order(order)
        shares_after = 0 if position is None else position.shares
        self._fills.append(
            AccountFill(
                fill_id=resolved_fill_id,
                order_id=order.order_id,
                trade_date=day,
                ts_code=order.ts_code,
                side=order.side,
                shares=qty,
                trade_price=unit_price,
                amount=amount,
                fee=cost_fee,
                order_completed=order_completed,
                position_closed=closed,
            )
        )
        self.assert_invariants(check_nav=False)
        return FillResult(
            order_id=order.order_id,
            fill_quantity=qty,
            remaining_quantity=order.remaining_quantity,
            order_completed=order_completed,
            position_closed=closed,
            position_shares_after=shares_after,
            cash_after=self._cash,
        )

    def _prepare_fill(
        self,
        order: Order,
        *,
        trade_date: object,
        quantity: int,
        trade_price: object,
        fee: object,
        fill_id: str | None,
    ) -> tuple[str, int, Decimal, Decimal, Decimal, str]:
        day = normalize_date(trade_date)
        if day < order.earliest_trade_date:
            raise AccountError(
                f"成交日 {day} 早于最早可交易日 {order.earliest_trade_date}（时间轴错误）"
            )
        if self._last_mark_date is not None and day <= self._last_mark_date:
            raise AccountError(
                f"成交日 {day} 不晚于最后估值日 {self._last_mark_date}，禁止改动已封存收盘态"
            )
        qty = share_count(quantity, field="fill quantity")
        if qty <= 0 or qty > order.remaining_quantity:
            raise AccountError(
                f"成交量 {qty} 必须位于 [1, {order.remaining_quantity}] 订单剩余量内"
            )
        if order.side is OrderSide.BUY and qty % self._buy_lot_size:
            raise AccountError(f"买入成交必须整手: {qty} 不是 {self._buy_lot_size} 的倍数")
        unit_price = price(trade_price)
        cost_fee = money(fee)
        if cost_fee < 0:
            raise AccountError(f"费用不得为负: {cost_fee}")
        resolved_id = str(fill_id or f"{order.order_id}-{len(self._fills) + 1}").strip()
        if not resolved_id:
            raise AccountError("fill_id 不得为空")
        if any(item.fill_id == resolved_id for item in self._fills):
            raise AccountError(f"fill_id 重复: {resolved_id}")
        return day, qty, unit_price, money(unit_price * qty), cost_fee, resolved_id

    def _apply_buy_fill(
        self,
        order: Order,
        position: Position | None,
        day: str,
        qty: int,
        unit_price: Decimal,
        amount: Decimal,
        fee: Decimal,
    ) -> Position:
        total = amount + fee
        if total > self._cash:
            raise InsufficientCash(
                f"买入 {order.ts_code} 需要 {total}，现金仅 {self._cash}（零副作用）"
            )
        self._cash -= total
        if position is None:
            position = Position(
                ts_code=order.ts_code,
                shares=qty,
                cost_basis=total,
                last_price=unit_price,
                last_price_date=day,
                buy_dates={day: qty},
            )
            self._positions[order.ts_code] = position
            return position
        position.shares += qty
        position.cost_basis += total
        position.last_price = unit_price
        position.last_price_date = day
        position.missing_streak_days = 0
        position.buy_dates[day] = position.buy_dates.get(day, 0) + qty
        return position

    def _apply_sell_fill(
        self,
        position: Position | None,
        ts_code: str,
        qty: int,
        amount: Decimal,
        fee: Decimal,
        day: str,
    ) -> tuple[Position | None, bool]:
        if position is None or position.shares < qty:
            held = 0 if position is None else position.shares
            raise AccountError(f"卖出 {ts_code} {qty} 股超过持仓 {held} 股（零副作用）")
        sellable = position.sellable_shares(day)
        if sellable < qty:
            raise AccountError(f"卖出 {ts_code} {qty} 股超过 {day} 可卖 {sellable} 股（股份 T+1）")
        remaining = qty
        for buy_date in sorted(position.buy_dates):
            if buy_date >= day:
                continue
            take = min(remaining, position.buy_dates[buy_date])
            position.buy_dates[buy_date] -= take
            remaining -= take
            if position.buy_dates[buy_date] == 0:
                del position.buy_dates[buy_date]
            if remaining == 0:
                break
        removed_cost = (
            position.cost_basis
            if qty == position.shares
            else money(position.cost_basis * Decimal(qty) / Decimal(position.shares))
        )
        self._cash += amount - fee
        position.shares -= qty
        position.cost_basis -= removed_cost
        if position.shares == 0:
            self._positions.pop(ts_code, None)
            return None, True
        position.missing_streak_days = 0
        return position, False

    def _complete_order(self, order: Order) -> None:
        order.status = OrderStatus.FILLED
        self._active_orders.pop(order.order_id, None)
        self._order_by_code_side.pop((order.ts_code, order.side), None)

    def cancel_order(self, order_id: str, *, reason: str = "cancelled") -> None:
        """撤单；部分成交持仓保留，订单与持仓分别记状态。"""
        self._finish_unfilled_order(order_id, OrderStatus.CANCELLED, reason)

    def expire_order(self, order_id: str, *, reason: str = "expired") -> None:
        """超时失效；仅逐订单失效，不冻结账户/批次。"""
        self._finish_unfilled_order(order_id, OrderStatus.EXPIRED, reason)

    def _finish_unfilled_order(self, order_id: str, status: OrderStatus, reason: str) -> None:
        order = self._active_orders.get(str(order_id))
        if order is None:
            raise AccountError(f"订单不在 pending 集合: {order_id}")
        order.status = status
        order.reason = str(reason)
        self._active_orders.pop(order.order_id, None)
        self._order_by_code_side.pop((order.ts_code, order.side), None)

    def mark(self, snapshot: MarketSnapshot) -> AccountMark:
        """按指定日快照估值；只使用该日观测，不读取任何未来价格。"""
        day = normalize_date(snapshot.as_of)
        if self._last_mark_date is not None and day <= self._last_mark_date:
            raise AccountError(f"估值日必须递增: {day} 不晚于 {self._last_mark_date}")
        self._settle_receivables(day)
        market_value = ZERO
        pending_missing: list[tuple[Position, PriceStatus, Decimal]] = []
        for code in sorted(self._positions):
            position = self._positions[code]
            observation = snapshot.observation(code)
            if observation.status is PriceStatus.OK:
                assert observation.value is not None
                position.last_price = observation.value
                position.last_price_date = day
                position.missing_streak_days = 0
                market_value += money(position.last_price * position.shares)
                continue
            position.missing_streak_days += 1
            carried = money(position.last_price * position.shares)
            market_value += carried
            pending_missing.append((position, observation.status, carried))

        nav = money(self._cash + self.receivable + market_value)
        missing_records: list[MissingPriceRecord] = []
        for position, reason, carried in pending_missing:
            weight = float(money(carried / nav * 100) / 100) if nav > 0 else 0.0
            record = MissingPriceRecord(
                as_of=day,
                ts_code=position.ts_code,
                reason=reason,
                shares=position.shares,
                carried_value=carried,
                last_price_date=position.last_price_date,
                days_since_last_quote=calendar_days_between(position.last_price_date, day),
                missing_streak_days=position.missing_streak_days,
                nav=nav,
                unresolved_weight=weight,
                source=snapshot.source,
            )
            stored, _ = self._missing_ledger.record(record)
            missing_records.append(stored)
        unresolved = (
            float(
                money(sum((item.carried_value for item in missing_records), ZERO) / nav * 100) / 100
            )
            if nav > 0
            else 0.0
        )
        self._last_mark_date = day
        self._last_snapshot = AccountSnapshot(
            as_of=day,
            cash=self._cash,
            receivable=self.receivable,
            market_value=money(market_value),
            nav=nav,
            unresolved_weight=unresolved,
        )
        self.assert_invariants(check_nav=True)
        return AccountMark(
            as_of=day,
            cash=self._cash,
            receivable=self.receivable,
            market_value=money(market_value),
            nav=nav,
            unresolved_weight=unresolved,
            missing_records=tuple(missing_records),
        )

    @property
    def snapshot(self) -> AccountSnapshot:
        if self._last_snapshot is None:
            raise AccountError("账户尚未完成任何收盘估值")
        return self._last_snapshot

    def _settle_receivables(self, day: str) -> None:
        for key, item in list(self._receivables.items()):
            if item.settle_date <= day:
                self._cash += item.amount
                del self._receivables[key]

    def assert_invariants(self, *, check_nav: bool = False) -> None:
        """账户结构不变量；``check_nav=True`` 时才与最后收盘快照比对。"""
        self._assert_cash_and_positions()
        if check_nav:
            self._assert_nav_identity()
        self._assert_active_orders()

    def _assert_cash_and_positions(self) -> None:
        if self._cash < 0:
            raise AccountError(f"现金为负: {self._cash}")
        if self.receivable < 0:
            raise AccountError(f"应收为负: {self.receivable}")
        for code, position in self._positions.items():
            if position.ts_code != code:
                raise AccountError(f"持仓键与代码不一致: {code} != {position.ts_code}")
            if position.shares <= 0:
                raise AccountError(f"{code} 持仓股数必须为正: {position.shares}")
            if position.cost_basis < 0:
                raise AccountError(f"{code} 成本为负: {position.cost_basis}")
            if position.last_price <= 0:
                raise AccountError(f"{code} 最后有效价非正: {position.last_price}")

    def _assert_nav_identity(self) -> None:
        if self._last_snapshot is None:
            return
        market_value = sum(
            (money(item.last_price * item.shares) for item in self._positions.values()),
            ZERO,
        )
        calculated = money(self._cash + self.receivable + market_value)
        if calculated != self._last_snapshot.nav:
            raise AccountError(
                f"账户不守恒：cash+应收+持仓={calculated}，快照 NAV={self._last_snapshot.nav}"
            )

    def _assert_active_orders(self) -> None:
        for key, order in self._active_orders.items():
            if key != order.order_id or order.status is not OrderStatus.PENDING:
                raise AccountError(f"活动订单索引损坏: {key} -> {order.status}")
            if order.remaining_quantity <= 0:
                raise AccountError(f"活动订单剩余量必须为正: {order.order_id}")
            if order.side is OrderSide.SELL:
                held = self._positions.get(order.ts_code)
                if held is None or held.shares < order.remaining_quantity:
                    raise AccountError(f"活动卖单 {order.order_id} 超出剩余持仓，禁止裸卖")

    def snapshots_for_audit(self) -> list[dict]:
        """账本审计行：订单、成交与应收的稳定视图。"""
        rows = [{"kind": "order", **order.summary()} for order in self._orders.values()]
        rows.extend(
            {
                "kind": "fill",
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "trade_date": fill.trade_date,
                "ts_code": fill.ts_code,
                "side": fill.side.value,
                "shares": fill.shares,
                "trade_price": str(fill.trade_price),
                "amount": str(fill.amount),
                "fee": str(fill.fee),
                "order_completed": fill.order_completed,
                "position_closed": fill.position_closed,
            }
            for fill in self._fills
        )
        rows.extend(
            {
                "kind": "receivable",
                "receivable_id": item.receivable_id,
                "ts_code": item.ts_code,
                "amount": str(item.amount),
                "recognized_date": item.recognized_date,
                "settle_date": item.settle_date,
            }
            for item in self._receivables.values()
        )
        rows.extend(
            {
                "kind": "share_adjustment",
                "adjustment_id": item.adjustment_id,
                "ts_code": item.ts_code,
                "trade_date": item.trade_date,
                "share_ratio": str(item.share_ratio),
                "shares_before": item.shares_before,
                "shares_after": item.shares_after,
            }
            for item in self._share_adjustments
        )
        return rows
