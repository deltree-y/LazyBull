"""P4-T3 原生单袖合并引擎（hosts 侧公共值对象边界）。

删除 P2b 旧 dict 适配器后，回测与纸面宿主在本类接缝处直接构造
``TargetPortfolio`` / ``Account`` 并调用 ``PortfolioMerger`` 协议：
归一化之后、批次限权之前。纸面 S6 在提交前消费同一指令序列；回测 S6 仅作
T3 证明性证据，不参与成交，且其成交投影禁止 T4 复用。多袖能力仍归 P6.5。
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from src.lazybull.v2.common.protocols.core import PortfolioMerger
from src.lazybull.v2.common.types import (
    Account,
    Lot,
    Money,
    Order,
    OrderReason,
    OrderSide,
    OrderType,
    Position,
    Price,
    Shares,
    TargetPortfolio,
    TradeDate,
    TSCode,
    Weight,
)
from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.core.execution.ml_signal_feed import BacktestEngineML
from src.lazybull.v2.core.execution.signal_execution import _cap_normalized_signals


def _require_position_field(info: Mapping[str, Any], key: str, stock: str) -> Any:
    """读取原生账户快照必需字段；缺失或缺失标记 fail-fast。"""
    value = info.get(key)
    if value is None or pd.isna(value):
        raise ValueError(f"原生合并账户快照拒绝缺失字段：股票 {stock} 字段 {key!r}")
    return value


def build_native_account(engine: BacktestEngineML, total_value: float) -> Account:
    """按当次接缝状态构造只读 Account；每股票单 lot 投影，多 lot 归 P6.5。

    ``last_known_price → buy_trade_price`` 仅为合并快照窄回退，不参与引擎估值。
    """
    if total_value is None:
        raise RuntimeError("原生合并接缝缺少同日组合总值 _last_rebalance_nav")
    position_map: Dict[TSCode, Position] = {}
    for stock, info in engine.positions.items():
        shares = _require_position_field(info, "shares", stock)
        if not shares:
            continue
        buy_date = _require_position_field(info, "buy_date", stock)
        buy_cost_cash = _require_position_field(info, "buy_cost_cash", stock)
        buy_trade_price = _require_position_field(info, "buy_trade_price", stock)
        last_price = info.get("last_known_price")
        if last_price is None:
            last_price = buy_trade_price
        lot_date = pd.Timestamp(buy_date).date()
        if pd.isna(lot_date):
            raise ValueError(f"原生合并账户快照拒绝非法 buy_date：股票 {stock}")
        lot = Lot(
            shares=Shares(shares),
            buy_date=TradeDate(lot_date),
            cost_basis=Money(buy_cost_cash),
        )
        position_map[TSCode(stock)] = Position(
            ts_code=TSCode(stock),
            shares=Shares(shares),
            cost_basis=Money(buy_cost_cash),
            last_price=Price(last_price),
            lots=(lot,),
        )
    return Account(
        cash=Money(engine.current_capital),
        positions=MappingProxyType(position_map),
        total_value=Money(total_value),
        nav=calculate_nav(total_value, engine.initial_capital),
    )


def _sell_order_reason(record: Mapping[str, Any]) -> OrderReason:
    """卖出记录类型到原生指令原因的单一映射。"""
    sell_type = str(record.get("sell_type", ""))
    if sell_type == "holding_period":
        return OrderReason.EXPIRY
    if sell_type == "stop_loss":
        return OrderReason.STOP_LOSS
    if sell_type == "rebalance":
        return OrderReason.REBALANCE
    if sell_type == "risk_trim":
        return OrderReason.TRIM
    return OrderReason.CONDITION_SELL


def _project_trade_order(
    record: Mapping[str, Any], current: int, sleeve_id: str
) -> tuple[Order, int]:
    """投影单条真实成交并返回 Order 与成交后目标股数。"""
    stock = str(record["stock"])
    action = str(record["action"])
    shares = int(record["shares"])
    if action == "buy":
        side = OrderSide.BUY
        target = current + shares
        reason = OrderReason.REBALANCE
    elif action == "sell":
        sell_timing = record.get("sell_timing")
        if sell_timing not in (None, "", "open"):
            raise ValueError(
                f"T3 证据投影仅支持 open 卖出，拒绝 {sell_timing!r}（{stock}）；"
                "OrderType 无 close 枚举，禁止静默伪造"
            )
        side = OrderSide.SELL
        target = max(current - shares, 0)
        reason = _sell_order_reason(record)
    else:
        raise ValueError(f"原生 Order 投影拒绝未知动作：{action!r}（{stock}）")
    order = Order(
        ts_code=TSCode(stock),
        side=side,
        target_shares=Shares(target),
        order_type=OrderType.MARKET_OPEN,
        reason=reason,
        source_sleeve=sleeve_id,
    )
    return order, target


def project_t3_evidence_orders(
    trades: Sequence[Mapping[str, Any]],
    sleeve_id: str = "value",
    *,
    initial_positions: Mapping[str, int] | None = None,
) -> list[Order]:
    """仅供 T3 回测 S6 证明性证据；不参与执行，禁止 T4/paper_replay 复用。

    输入必须是完整成交序列；空仓起算可省略 ``initial_positions``，否则必须显式
    提供期初持仓，保证 ``target_shares`` 是目标股数而非本批增量。close 卖出显式
    拒绝，未成交/顺延指令不属于本证据投影。
    """
    holdings: Dict[str, int] = dict(initial_positions or {})
    orders: list[Order] = []
    for record in trades:
        stock = str(record["stock"])
        order, target = _project_trade_order(record, holdings.get(stock, 0), sleeve_id)
        holdings[stock] = target
        orders.append(order)
    return orders


class NativeMergeBacktestEngineML(BacktestEngineML):
    """由宿主注入原生 PortfolioMerger 的单袖引擎。"""

    def __init__(
        self,
        *args,
        portfolio_merger: PortfolioMerger,
        sleeve_id: str = "value",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.portfolio_merger = portfolio_merger
        self.merger_sleeve_id = sleeve_id

    def _build_current_account(self, total_value: float) -> Account:
        """当次接缝账户快照（独立方法便于宿主验收观测）。"""
        return build_native_account(self, total_value)

    def _merge_target_weights(
        self, normalized: Dict[str, float], date: pd.Timestamp
    ) -> Dict[str, float]:
        """构造原生 TargetPortfolio/Account 并无损解包单袖合并结果。"""
        target = TargetPortfolio(
            sleeve_id=self.merger_sleeve_id,
            target_weights=MappingProxyType(
                {TSCode(code): Weight(weight) for code, weight in normalized.items()}
            ),
            as_of_date=TradeDate(pd.Timestamp(date).date()),
        )
        current = self._build_current_account(self._last_rebalance_nav)
        merged = self.portfolio_merger.merge({self.merger_sleeve_id: target}, current)
        return {code.value: weight.value for code, weight in merged.target_weights.items()}

    def _normalize_and_cap_signals(
        self, signals: Dict[str, float], date: pd.Timestamp, tranche_idx: int
    ) -> Dict[str, float]:
        """在 hosts 边界构造原生目标并合并，再沿用既有批次限权。"""
        normalized = self._normalize_signals(signals, date)
        weights = self._merge_target_weights(normalized, date)
        return _cap_normalized_signals(self, weights, tranche_idx)

    def net_native_orders(self, orders: Sequence[Order]) -> Sequence[Order]:
        """T3 回测证明性 S6 调用；返回值只入证据，不参与已经完成的成交。"""
        return self.portfolio_merger.net_across_sleeves(orders)
