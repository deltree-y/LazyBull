"""回测延迟订单执行 mixin。"""

from typing import Dict

import pandas as pd
from loguru import logger

from ..common.date_utils import to_trade_date_str
from ..common.trade_status import is_tradeable


class BacktestPendingExecutionMixin:
    """提供回测延迟订单事件记录与执行相关实现。"""

    def _record_pending_order_event(self, event: Dict) -> None:
        """记录延迟订单事件，日终统一压缩显示。"""
        event_type = str(event.get("type", "")).lower()
        if event_type == "added":
            self._daily_warning_items.setdefault("pending_order_added", []).append(
                {
                    "stock": str(event.get("stock", "-")),
                    "action": str(event.get("action", "-")),
                    "reason": str(event.get("reason") or "-"),
                }
            )
            return

        if event_type == "success":
            self._daily_warning_items.setdefault("pending_order_success", []).append(
                {
                    "stock": str(event.get("stock", "-")),
                    "action": str(event.get("action", "-")),
                    "retry_count": int(event.get("retry_count", 0) or 0),
                    "delay_days": int(event.get("delay_days", 0) or 0),
                }
            )
            return

        if event_type in {"expired_retry", "expired_days"}:
            self._daily_warning_items.setdefault("pending_order_expired", []).append(
                {
                    "stock": str(event.get("stock", "-")),
                    "action": str(event.get("action", "-")),
                    "expire_type": event_type,
                    "retry_count": int(event.get("retry_count", 0) or 0),
                    "max_retry_count": int(event.get("max_retry_count", 0) or 0),
                    "delay_days": int(event.get("delay_days", 0) or 0),
                    "max_retry_days": int(event.get("max_retry_days", 0) or 0),
                }
            )

    def _process_pending_orders(self, date: pd.Timestamp) -> None:
        """处理延迟订单队列

        Args:
            date: 当前日期
        """
        if not self.pending_order_manager:
            return

        # 获取应重试的订单列表及已放弃的订单（仅 buy 订单会过期，sell 订单持续重试直至复牌）
        orders_to_retry, expired_orders = self.pending_order_manager.get_orders_to_retry(date)

        if not orders_to_retry:
            return

        # 获取当日行情数据
        trade_date_str = to_trade_date_str(date)
        date_quote = self.price_data_cache[self.price_data_cache["trade_date"] == trade_date_str]

        for order in orders_to_retry:
            # 检查是否可交易
            if date_quote.empty:
                # 当日行情数据为空，无法判断交易状态，继续延迟
                logger.warning(f"延迟订单 {order.stock} 在 {date.date()} 无行情数据，继续延迟")
                continue
            tradeable, reason = is_tradeable(
                order.stock, trade_date_str, date_quote, action=order.action
            )

            if tradeable:
                # 可交易，尝试执行
                if order.action == "buy":
                    self._buy_stock_direct(
                        date, order.stock, order.target_value, signal_date=order.signal_date
                    )
                    self.pending_order_manager.mark_success(date, order.stock, "buy")
                elif order.action == "sell":
                    self._sell_stock_direct(date, order.stock)
                    self.pending_order_manager.mark_success(date, order.stock, "sell")
            else:
                # 仍不可交易，更新延迟订单
                self.pending_order_manager.add_order(
                    stock=order.stock,
                    action=order.action,
                    current_date=date,
                    signal_date=order.signal_date,
                    target_value=order.target_value,
                    reason=reason,
                )
