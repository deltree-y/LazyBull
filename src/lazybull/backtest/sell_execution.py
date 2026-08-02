"""回测卖出执行 mixin。"""

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from ..common.date_utils import to_trade_date_str
from ..common.trade_status import is_tradeable
from ..risk.stop_loss_checker import check_positions_stop_loss
from ..trading.sell_rules import (
    is_holding_period_expired,
    min_holding_days_for_rebalance_sell,
    select_rebalance_sell_candidates,
)


class BacktestSellExecutionMixin:
    """提供回测卖出执行相关实现。"""

    def _queue_rebalance_sells(
        self,
        date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        date_to_idx: Dict,
    ) -> None:
        """调仓日同步生成卖出信号：将当前非保护持仓排队到 T+1 卖出。

        使卖出与买入在同一交易日执行，消除调仓日卖出滞后一天的偏差。
        与 _check_and_sell 的职责互补：后者处理日常持有期到期/条件卖出，
        本方法仅在信号成功入队列的调仓日触发，针对即将到期的持仓提前排队。

        Args:
            date: 当前调仓日（信号生成日）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if not self.positions:
            return

        current_idx = date_to_idx.get(date)
        if current_idx is None:
            return

        # 获取新信号中的股票（这些股票应保留，不卖出）
        signal_data = self.pending_signals.get(date, {})
        if isinstance(signal_data, dict) and "signals" in signal_data:
            new_signal_stocks = set(signal_data["signals"].keys())
        else:
            new_signal_stocks = set()

        # 构建持有天数映射（无法计算持有天数的持仓保持既有行为：不排队卖出）
        holding_days_map: Dict[str, Optional[int]] = {}
        for stock, info in list(self.positions.items()):
            buy_date = info.get("buy_date")
            buy_idx = date_to_idx.get(buy_date) if buy_date else None
            if buy_idx is None:
                continue
            holding_days_map[stock] = current_idx - buy_idx

        # 共享调仓卖出筛选（回测阈值下限 floor=0：仅在持仓即将到期时提前触发）
        decision = select_rebalance_sell_candidates(
            holding_days_map,
            min_holding_days=min_holding_days_for_rebalance_sell(self.holding_period, floor=0),
            target_codes=new_signal_stocks,
            queued_codes=set(self.pending_condition_sells) | set(self.pending_stop_loss_sells),
        )

        for stock in decision.sells:
            # 排队到 T+1 卖出
            self.pending_condition_sells[stock] = {
                "trigger_date": date,
                "sell_type": "rebalance",
            }

        if decision.sells:
            logger.info(f"调仓日同步卖出: {len(decision.sells)} 只排队到 T+1 卖出")

    def _check_and_sell(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """检查卖出条件并生成 T0 卖出信号

        - 持有期到期：写入 pending_condition_sells 队列（Tn+1 执行）

        Args:
            date: 当前日期
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        holding_period_sell_slot_weights: List[Dict[str, float]] = []

        current_idx = date_to_idx.get(date)
        if current_idx is None:
            return

        portfolio_value_for_planning = self._calculate_portfolio_value(date)

        # 过滤已在待卖队列中的持仓（避免重复）
        positions_to_check = {
            stock: info
            for stock, info in self.positions.items()
            if stock not in self.pending_condition_sells
            and stock not in self.pending_stop_loss_sells
        }

        for stock, info in positions_to_check.items():
            buy_date = info["buy_date"]
            buy_idx = date_to_idx.get(buy_date)

            # 以实际买入日作为持有期起点
            anchor_idx = buy_idx
            if anchor_idx is None:
                signal_date = info.get("signal_date", buy_date)
                logger.warning(
                    f"股票 {stock} 买入日期 {buy_date}（信号日 {signal_date}）不在交易日映射中"
                )
                continue

            # 计算持有天数（交易日）
            holding_days = current_idx - anchor_idx

            # 持有期到期 → T0 生成卖出信号，T+1 执行（共享判定口径）
            if is_holding_period_expired(holding_days, self.holding_period):
                self.pending_condition_sells[stock] = {
                    "trigger_date": date,
                    "sell_type": "holding_period",
                }
                holding_period_sell_slot_weights.append(
                    {
                        "stock": stock,
                        "weight": self._get_position_weight_for_planning(
                            date,
                            stock,
                            portfolio_value=portfolio_value_for_planning,
                        ),
                    }
                )

        if holding_period_sell_slot_weights:
            self._queue_condition_sell_refill_signal(
                date=date,
                slot_weights=holding_period_sell_slot_weights,
                price_data=self.price_data_cache,
                date_to_idx=date_to_idx,
            )

    def _execute_pending_condition_sells(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """执行待条件卖出操作（Tn+1 日执行）

        Args:
            date: 当前日期（执行日，Tn+1）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if not self.pending_condition_sells:
            return

        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx == 0:
            return

        trigger_date = trading_dates[current_idx - 1]

        # 筛选前一交易日触发的条件卖出
        stocks_to_sell = [
            (stock, info)
            for stock, info in list(self.pending_condition_sells.items())
            if info["trigger_date"] == trigger_date
        ]
        if not stocks_to_sell:
            return

        # 执行卖出
        for stock, info in stocks_to_sell:
            if stock not in self.positions:
                self.pending_condition_sells.pop(stock, None)
                continue
            self._sell_stock(date, stock, sell_type=info["sell_type"])
            self.pending_condition_sells.pop(stock, None)

        # 实际卖出明细统一在每日总结下一行展示，这里不再单独输出卖出执行日志。

    def _check_stop_loss(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """检查止损触发条件（T 日检查，生成 T+1 卖出信号）

        Args:
            date: 当前日期（检查日）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if not self.stop_loss_monitor:
            return

        trade_date_str = to_trade_date_str(date)

        # 过滤掉已在待止损卖出队列中的持仓
        positions_to_check = {
            stock: info
            for stock, info in self.positions.items()
            if stock not in self.pending_stop_loss_sells
        }
        if not positions_to_check:
            return

        # 构建价格和跌停信息
        date_quote = self.price_data_cache[self.price_data_cache["trade_date"] == trade_date_str]
        prices: Dict[str, float] = {}
        limit_down_info: Dict[str, bool] = {}
        for stock in positions_to_check:
            price = self._get_trade_price(date, stock)
            if price is not None:
                prices[stock] = price
            if not date_quote.empty:
                stock_quote = date_quote[date_quote["ts_code"] == stock]
                if not stock_quote.empty and "is_limit_down" in stock_quote.columns:
                    limit_down_info[stock] = bool(stock_quote["is_limit_down"].iloc[0] == 1)

        # 获取停牌日历
        suspend_calendar = None
        try:
            suspend_calendar = self._get_suspend_calendar()
        except Exception as e:
            logger.warning(f"停牌日历初始化失败（{e}），将跳过停牌检查")

        # 调用公共止损检查
        actions = check_positions_stop_loss(
            positions=positions_to_check,
            stop_loss_monitor=self.stop_loss_monitor,
            prices=prices,
            limit_down_info=limit_down_info,
            suspend_calendar=suspend_calendar,
            trade_date=trade_date_str,
            verbose=self.verbose,
        )

        # 将结果转换为引擎内部的 pending_stop_loss_sells 格式
        for action in actions:
            self.pending_stop_loss_sells[action.ts_code] = {
                "trigger_date": date,
                "reason": action.reason,
                "trigger_type": action.trigger_type or "unknown",
            }
            if self.verbose:
                logger.warning(
                    f"  止损触发: {date.date()} {action.ts_code}, 原因: {action.reason}, "
                    f"将在下一交易日执行卖出"
                )

    def _execute_pending_stop_loss_sells(
        self, date: pd.Timestamp, trading_dates: List[pd.Timestamp], date_to_idx: Dict
    ) -> None:
        """执行待止损卖出操作（T+1 日执行）

        Args:
            date: 当前日期（执行日，T+1）
            trading_dates: 交易日列表
            date_to_idx: 日期到索引的映射
        """
        if not self.stop_loss_monitor or not self.pending_stop_loss_sells:
            return

        # 查找前一个交易日触发的止损
        current_idx = date_to_idx.get(date)
        if current_idx is None or current_idx == 0:
            return

        trigger_date = trading_dates[current_idx - 1]

        # 执行前一交易日触发的止损卖出
        stocks_to_sell = []
        for stock, info in list(self.pending_stop_loss_sells.items()):
            if info["trigger_date"] == trigger_date:
                stocks_to_sell.append((stock, info))

        if not stocks_to_sell:
            return

        # 执行卖出
        for stock, info in stocks_to_sell:
            # 检查股票是否还在持仓中（可能已被正常调仓卖出）
            if stock not in self.positions:
                # 从待卖出队列中移除
                self.pending_stop_loss_sells.pop(stock, None)
                continue

            # 执行止损卖出
            self._sell_stock(
                date,
                stock,
                sell_type="stop_loss",
                sell_reason=info["reason"],
                trigger_type=info["trigger_type"],
            )

            # 从待卖出队列中移除
            self.pending_stop_loss_sells.pop(stock, None)

        # 实际卖出明细统一在每日总结下一行展示，这里不再单独输出止损卖出执行日志。

    def _sell_stock(
        self,
        date: pd.Timestamp,
        stock: str,
        sell_type: str = "holding_period",
        sell_reason: Optional[str] = None,
        trigger_type: Optional[str] = None,
    ) -> None:
        """卖出股票（在 T+n 日以收盘价卖出）

        带交易状态检查的卖出方法。如果启用延迟订单功能，会检查股票是否可交易。

        Args:
            date: 卖出日期（T+n）
            stock: 股票代码
            sell_type: 卖出类型，'holding_period' 或 'stop_loss'
            sell_reason: 卖出原因描述（止损时使用）
            trigger_type: 触发类型（止损时使用）
        """
        self._sell_stock_with_status_check(date, stock, sell_type, sell_reason, trigger_type)

    def _sell_stock_with_status_check(
        self,
        date: pd.Timestamp,
        stock: str,
        sell_type: str = "holding_period",
        sell_reason: Optional[str] = None,
        trigger_type: Optional[str] = None,
    ) -> None:
        """卖出股票（带交易状态检查）

        如果启用延迟订单功能，会检查股票是否可交易（停牌或跌停）
        不可交易时加入延迟队列而非直接失败

        Args:
            date: 卖出日期
            stock: 股票代码
            sell_type: 卖出类型，'holding_period' 或 'stop_loss'
            sell_reason: 卖出原因描述（止损时使用）
            trigger_type: 触发类型（止损时使用）
        """
        # 检查交易状态
        if self.enable_pending_order and self.price_data_cache is not None:
            trade_date_str = to_trade_date_str(date)

            # 使用 SuspendCalendar 检查停牌状态
            is_suspended_flag = False
            suspend_calendar = None
            try:
                suspend_calendar = self._get_suspend_calendar()
                is_suspended_flag = suspend_calendar.is_suspended(stock, trade_date_str)
                if is_suspended_flag:
                    # 停牌，加入延迟队列
                    if self.pending_order_manager:
                        self.pending_order_manager.add_order(
                            stock=stock,
                            action="sell",
                            current_date=date,
                            signal_date=date,  # 卖出是基于持有期，用当前日期
                            target_value=None,
                            reason="停牌",
                        )
                    if self.verbose:
                        logger.info(f"  卖出延迟: {date.date()} {stock}, 原因: 停牌")
                    return
            except Exception as e:
                # 停牌数据加载失败，记录警告但继续检查（降级处理）
                logger.warning(f"停牌状态检查失败（{e}），继续检查其他交易状态")

            # 检查行情数据
            date_quote = self.price_data_cache[
                self.price_data_cache["trade_date"] == trade_date_str
            ]
            if date_quote.empty:
                # 当日行情数据为空，无法判断交易状态，加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action="sell",
                        current_date=date,
                        signal_date=date,  # 卖出是基于持有期，用当前日期
                        target_value=None,
                        reason="无行情数据",
                    )
                if self.verbose:
                    logger.info(f"  卖出延迟: {date.date()} {stock}, 原因: 无行情数据")
                return

            # 检查跌停状态
            tradeable, reason = is_tradeable(stock, trade_date_str, date_quote, action="sell")

            if not tradeable:
                # 不可交易（跌停等），加入延迟队列
                if self.pending_order_manager:
                    self.pending_order_manager.add_order(
                        stock=stock,
                        action="sell",
                        current_date=date,
                        signal_date=date,  # 卖出是基于持有期，用当前日期
                        target_value=None,
                        reason=reason,
                    )
                if self.verbose:
                    logger.info(f"  卖出延迟: {date.date()} {stock}, 原因: {reason}")
                return

        # 可交易，直接卖出
        self._sell_stock_direct(date, stock, sell_type, sell_reason, trigger_type)

    def _sell_stock_direct(
        self,
        date: pd.Timestamp,
        stock: str,
        sell_type: str = "holding_period",
        sell_reason: Optional[str] = None,
        trigger_type: Optional[str] = None,
    ) -> None:
        """直接卖出股票（不检查交易状态）

        现金流使用成交价格（trade_price）计算
        收益率使用绩效价格（pnl_price）计算

        根据 sell_timing 参数选择使用开盘价或收盘价：
        - sell_timing='close': 使用收盘价（默认）
        - sell_timing='open': 使用开盘价，如果开盘价不存在则降级到收盘价

        Args:
            date: 卖出日期（T+n）
            stock: 股票代码
            sell_type: 卖出类型，'holding_period' 或 'stop_loss'
            sell_reason: 卖出原因描述（止损时使用）
            trigger_type: 触发类型（止损时使用）
        """
        if stock not in self.positions or self.positions[stock]["shares"] == 0:
            return

        # 根据 sell_timing 参数选择价格
        if self.sell_timing == "open":
            # 尝试使用开盘价
            sell_trade_price = self._get_trade_price_open(date, stock)
            sell_pnl_price = self._get_pnl_price_open(date, stock)

            # 降级策略：如果开盘价不存在，使用收盘价
            if sell_trade_price is None:
                if self.verbose:
                    logger.warning(
                        f"股票 {stock} 在 {date.date()} 缺少开盘成交价格，" f"降级使用收盘价卖出"
                    )
                sell_trade_price = self._get_trade_price(date, stock)
                if sell_trade_price is None:
                    logger.warning(
                        f"无法获取 {stock} 在 {date.date()} 的成交价格（开盘/收盘），跳过卖出"
                    )
                    return

            if sell_pnl_price is None:
                # 开盘绩效价格不存在，尝试降级到收盘绩效价格
                sell_pnl_price = self._get_pnl_price(date, stock)
                if sell_pnl_price is None:
                    logger.warning(
                        f"无法获取 {stock} 在 {date.date()} 的绩效价格，使用成交价格代替"
                    )
                    sell_pnl_price = sell_trade_price
        else:
            # 使用收盘价（默认）
            sell_trade_price = self._get_trade_price(date, stock)
            if sell_trade_price is None:
                logger.warning(f"无法获取 {stock} 在 {date.date()} 的成交价格，跳过卖出")
                return

            sell_pnl_price = self._get_pnl_price(date, stock)
            if sell_pnl_price is None:
                logger.warning(f"无法获取 {stock} 在 {date.date()} 的绩效价格，使用成交价格代替")
                sell_pnl_price = sell_trade_price

        # 获取持仓信息
        shares = self.positions[stock]["shares"]
        buy_date = self.positions[stock]["buy_date"]
        signal_date = self.positions[stock].get("signal_date", buy_date)
        buy_trade_price = self.positions[stock]["buy_trade_price"]
        buy_pnl_price = self.positions[stock]["buy_pnl_price"]
        buy_cost_cash = self.positions[stock]["buy_cost_cash"]

        # 计算现金流（基于成交价格）
        sell_amount = shares * sell_trade_price
        sell_cost = self.cost_model.calculate_sell_cost(sell_amount)
        sell_proceeds = sell_amount - sell_cost  # 卖出后实际到手金额

        # 计算收益率（基于绩效价格）
        pnl_buy_amount = shares * buy_pnl_price  # 绩效口径买入金额
        pnl_sell_amount = shares * sell_pnl_price  # 绩效口径卖出金额

        # 买入和卖出的手续费
        buy_amount = shares * buy_trade_price
        buy_cost = self.cost_model.calculate_buy_cost(buy_amount)
        total_cost = buy_cost + sell_cost  # 总手续费

        # 绩效收益（基于绩效价格，扣除手续费）
        # 收益 = 卖出金额 - 买入金额 - 总手续费
        pnl_profit_amount = pnl_sell_amount - pnl_buy_amount - total_cost
        # 收益率 = 收益 / (买入金额 + 买入手续费)
        # 买入成本是买入金额+买入手续费，这是投资者实际付出的成本
        pnl_profit_pct = (
            pnl_profit_amount / (pnl_buy_amount + buy_cost)
            if (pnl_buy_amount + buy_cost) > 0
            else 0
        )

        # 更新持仓和资金
        del self.positions[stock]
        self.current_capital += sell_proceeds

        # 如果是止损卖出，清理止损监控器中的持仓状态
        if sell_type == "stop_loss" and self.stop_loss_monitor:
            self.stop_loss_monitor.remove_position(stock)

        # 记录交易（包含绩效收益信息和卖出类型）
        trade_record = {
            "date": date,
            "signal_date": signal_date,
            "stock": stock,
            "action": "sell",
            "price": sell_trade_price,  # 卖出成交价格
            "shares": shares,
            "amount": sell_amount,
            "cost": sell_cost,
            "buy_date": buy_date,
            "buy_price": buy_trade_price,  # 买入成交价格
            "buy_pnl_price": buy_pnl_price,  # 买入绩效价格
            "sell_pnl_price": sell_pnl_price,  # 卖出绩效价格
            "pnl_profit_amount": pnl_profit_amount,  # 绩效收益金额
            "pnl_profit_pct": pnl_profit_pct,  # 绩效收益率
            "sell_type": sell_type,  # 卖出类型
            "sell_timing": self.sell_timing,  # 新增：卖出时机（open/close）
        }

        # 如果是止损卖出，添加止损相关信息
        if sell_type == "stop_loss":
            trade_record["sell_reason"] = sell_reason
            trade_record["trigger_type"] = trigger_type

        self.trades.append(trade_record)
