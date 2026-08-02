"""回测主循环 mixin。"""

import time
from typing import List

import pandas as pd
from loguru import logger


class BacktestRunLoopMixin:
    """提供回测主循环实现。"""

    def run(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """运行回测

        Args:
            start_date: 开始日期
            end_date: 结束日期
            trading_dates: 交易日列表
            price_data: 价格数据，需包含 ts_code, trade_date, close, close_adj（可选）

        Returns:
            净值曲线DataFrame
        """
        logger.info(f"开始回测: {start_date.date()} 至 {end_date.date()}")

        # 筛选回测期间的交易日
        trading_dates = [d for d in trading_dates if start_date <= d <= end_date]
        total_days = len(trading_dates)

        # 创建日期到索引的映射，优化查找效率
        date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}

        # 准备价格索引（使用 MultiIndex，替代嵌套字典）
        self._prepare_price_index(price_data)

        # 缓存价格数据用于交易状态检查
        self.price_data_cache = price_data

        # 获取调仓日期（信号生成日期）→ {日期: tranche_idx}
        signal_dates = self._get_rebalance_dates(trading_dates)

        if self.stagger_tranches > 1:
            logger.info(
                f"数据准备完成, 调仓日期共 {len(signal_dates)} 天"
                f"（{self.stagger_tranches} 批分批调仓）"
            )
        else:
            logger.info(f"数据准备完成, 调仓日期共 {len(signal_dates)} 天")

        # 记录开始时间
        start_time = time.time()
        deferred_sink_id = logger.add(
            self._collect_deferred_log,
            format="{message}",
            level="DEBUG",
            colorize=False,
            filter=lambda record: record["extra"].get("_defer_emit", False),
        )

        try:
            # 按日推进
            # _cycle_anchor_idx 是当前调仓周期的"第1天"在 trading_dates 中的 idx
            # 初始为 0（第一天即第1轮的第1天）；每次信号成功入队列时重置为信号日 idx
            # 这样门控连续阻断的空仓期不会推进 cycle_day
            self._cycle_anchor_idx = 0
            cycle_separator = "\n================================================ 新一轮回测 ================================================="
            for idx, date in enumerate(trading_dates):
                # 新一轮首日：输出分隔线（在所有业务日志之前）
                if idx == self._cycle_anchor_idx:
                    self._emit_immediate_log("INFO", cycle_separator)
                cycle_day = idx - self._cycle_anchor_idx + 1
                trade_start_idx = len(self.trades)
                self._deferred_day_logs = []
                self._reset_daily_warning_items()

                with logger.contextualize(_defer_emit=True):
                    # 处理延迟订单（先处理延迟订单，再处理新信号）
                    if self.enable_pending_order:
                        self._process_pending_orders(date)

                    # 检查止损（T 日检查，T+1 日执行卖出）
                    if self.stop_loss_monitor:
                        self._check_stop_loss(date, trading_dates, date_to_idx)

                    # 判断是否为信号生成日
                    if date in signal_dates:
                        tranche_idx = signal_dates[date]
                        self._generate_signal(
                            date,
                            trading_dates,
                            price_data,
                            date_to_idx,
                            tranche_idx=tranche_idx,
                        )
                        # 信号成功入队列 → 本日即为新周期第1天，更新 anchor 并输出分隔线
                        if date in self.pending_signals and idx != self._cycle_anchor_idx:
                            self._cycle_anchor_idx = idx
                            cycle_day = 1
                            self._emit_immediate_log("INFO", cycle_separator)

                        # 调仓日同步生成卖出信号：将当前非保护持仓排队到 T+1 卖出，
                        # 使卖出与买入在同一交易日执行，避免卖出滞后一天。
                        if date in self.pending_signals:
                            self._queue_rebalance_sells(date, trading_dates, date_to_idx)

                    # @2026/01/18: 改为先卖出再买入, 避免当天买入的股票被误判为达到持有期而卖出
                    # 执行止损卖出（Tn+1 执行）
                    if self.stop_loss_monitor:
                        self._execute_pending_stop_loss_sells(date, trading_dates, date_to_idx)

                    # 执行条件卖出（Tn+1 执行：亏损提前换出、整体止盈、持有期到期）
                    self._execute_pending_condition_sells(date, trading_dates, date_to_idx)

                    # 检查卖出条件并生成 T0 卖出信号
                    # - 持有期到期 / 盈利延续到期：写入 pending_condition_sells，Tn+1 执行
                    # - 亏损提前换出 / 整体止盈：写入 pending_condition_sells，Tn+1 执行
                    self._check_and_sell(date, trading_dates, date_to_idx)

                    # 执行待执行的买入操作（Tn+1）
                    self._execute_pending_buys(date, trading_dates, date_to_idx)

                    # 空仓提前调仓 / 盈利延续拖尾提前调仓：
                    # 场景 A（空仓）：持仓全部卖出，资金闲置 → 立即触发新一轮信号
                    # 场景 B（盈利延续拖尾）：cycle_day >= holding_period 但仍有残留持仓（通常为盈利延续）
                    #   → 若"残留持仓占比 + 新信号目标仓位 ≤ 100%"，则提前启动新一轮；否则继续等待
                    early_rebalance_guards_ok = (
                        self.enable_early_rebalance_on_empty
                        and not self.pending_signals
                        and not any(
                            slot_info.get("unfilled_count", 0) > 0
                            for slot_info in self.unfilled_slots.values()
                        )
                        and date not in signal_dates
                    )

                    is_empty_position = not self.positions
                    is_holding_period_exceeded = (
                        bool(self.positions) and cycle_day >= self.holding_period
                    )

                    if early_rebalance_guards_ok and (
                        is_empty_position or is_holding_period_exceeded
                    ):
                        if not is_empty_position:
                            # 盈利延续拖尾场景：打印当前残留持仓占比
                            current_nav = self._calculate_portfolio_value(date)
                            residual_market_value = current_nav - self.current_capital
                            residual_ratio = (
                                residual_market_value / current_nav if current_nav > 0 else 0.0
                            )
                        else:
                            residual_ratio = 0.0

                        # 快照历史状态：提前调仓若未真正入队列则回滚，避免污染门控/质量计算基准
                        # 仅快照评估过程会追加的字段，保证启用/禁用该开关对正常调仓日的门控计算完全一致
                        gate_history_snapshot = self._snapshot_early_rebalance_state(date)

                        self._generate_signal(
                            date,
                            trading_dates,
                            price_data,
                            date_to_idx,
                            tranche_idx=0,
                        )

                        # 盈利延续拖尾场景：需额外校验 "残留仓位 + 新信号仓位 ≤ 100%"
                        # 若不满足，撤回本次信号，继续等待残留持仓到期
                        signal_accepted = date in self.pending_signals
                        if signal_accepted and is_holding_period_exceeded:
                            current_nav = self._calculate_portfolio_value(date)
                            residual_market_value = current_nav - self.current_capital
                            residual_ratio = (
                                residual_market_value / current_nav if current_nav > 0 else 0.0
                            )
                            new_signal_weight_sum = sum(
                                self.pending_signals[date].get("signals", {}).values()
                            )
                            combined_ratio = residual_ratio + new_signal_weight_sum
                            if combined_ratio > 1.0 + 1e-9:
                                # 超过上限，撤回信号
                                del self.pending_signals[date]
                                signal_accepted = False
                                self._record_early_rebalance_summary(
                                    "拖尾拒绝",
                                    f"残留{residual_ratio:.1%}+新信号{new_signal_weight_sum:.1%}"
                                    f"={combined_ratio:.1%}>100%",
                                )
                            else:
                                self._record_early_rebalance_summary(
                                    "拖尾通过",
                                    f"残留{residual_ratio:.1%}+新信号{new_signal_weight_sum:.1%}"
                                    f"={combined_ratio:.1%}",
                                )

                        # 信号未真正入队列（门控阻断或拖尾拒绝）→ 回滚历史快照，避免污染基准
                        if not signal_accepted:
                            self._restore_early_rebalance_state(date, gate_history_snapshot)
                            if is_empty_position:
                                self._record_early_rebalance_summary(
                                    "空仓未入队",
                                    "无持仓, 新信号未入队",
                                )

                        # 信号真正入队列后，才更新节奏并清理预定调仓日
                        if signal_accepted:
                            if is_empty_position:
                                self._record_early_rebalance_summary(
                                    "空仓触发",
                                    "无持仓, 新信号入队",
                                )
                            # 清除接下来一个持有期内的原预定调仓日，避免"刚买完又调仓"
                            next_rebalance_cutoff_idx = idx + self.holding_period
                            stale_dates = [
                                d
                                for d in list(signal_dates.keys())
                                if idx < date_to_idx.get(d, -1) <= next_rebalance_cutoff_idx
                            ]
                            for d in stale_dates:
                                del signal_dates[d]
                            if stale_dates:
                                logger.info(
                                    f"  已清除未来 {len(stale_dates)} 个预定调仓日（至 {stale_dates[-1].date()}），"
                                    f"避免重复调仓"
                                )
                            # 信号成功入队列 → 本日即为新周期第1天，更新 anchor 并输出分隔线
                            if idx != self._cycle_anchor_idx:
                                self._cycle_anchor_idx = idx
                                cycle_day = 1
                                self._emit_immediate_log("INFO", cycle_separator)

                    # 处理仓位补齐（在补齐窗口期内尝试补齐未满仓位）
                    if self.enable_position_completion:
                        self._process_position_completion(
                            date, trading_dates, price_data, date_to_idx
                        )

                    # 计算当日组合价值
                    portfolio_value = self._calculate_portfolio_value(date)

                trading_days = idx + 1
                buy_count, sell_count, trade_detail_logs = self._build_daily_trade_log(
                    date=date,
                    trade_start_idx=trade_start_idx,
                    date_to_idx=date_to_idx,
                )
                self._emit_daily_summary_log(
                    self._format_daily_progress_log(
                        date=date,
                        trading_days=trading_days,
                        total_days=total_days,
                        cycle_day=cycle_day,
                        portfolio_value=portfolio_value,
                        buy_count=buy_count,
                        sell_count=sell_count,
                    )
                )
                self._flush_deferred_day_logs(
                    predicate=lambda record: "调仓决策摘要:" in str(record.get("message", ""))
                )
                for trade_detail_log in trade_detail_logs:
                    self._emit_immediate_log("INFO", f"  {trade_detail_log}")
                signal_count_log = self._build_daily_signal_log(date)
                if signal_count_log:
                    self._emit_immediate_log("INFO", f"  {signal_count_log}")
                for warning_log in self._build_daily_warning_logs():
                    self._emit_immediate_log("INFO", f"  {warning_log}")
                self._flush_deferred_day_logs()

                self.portfolio_values.append(
                    {
                        "date": date,
                        "portfolio_value": portfolio_value,
                        "capital": self.current_capital,
                        "market_value": portfolio_value - self.current_capital,
                    }
                )
        finally:
            logger.remove(deferred_sink_id)
            self._deferred_day_logs = []

        # 生成净值曲线
        nav_df = self._generate_nav_curve()

        total_time = time.time() - start_time
        logger.info(
            f"回测完成: 共 {len(trading_dates)} 个交易日, {len(self.trades)} 笔交易, 总耗时 {total_time:.1f}秒"
        )

        # 输出延迟订单统计
        if self.enable_pending_order and self.pending_order_manager:
            stats = self.pending_order_manager.get_statistics()
            logger.info(
                f"延迟订单统计: 累计添加 {stats['total_added']}, "
                f"成功执行 {stats['total_succeeded']}, "
                f"过期放弃 {stats['total_expired']}, "
                f"剩余待处理 {stats['pending']}"
            )

        # 输出仓位补齐统计
        if self.enable_position_completion:
            logger.info(
                f"仓位补齐统计: 累计未满仓 {self.completion_stats['total_unfilled']} 次, "
                f"补齐成功 {self.completion_stats['total_completed']} 次, "
                f"补齐尝试 {self.completion_stats['completion_attempts']} 次, "
                f"放弃补齐 {self.completion_stats['total_abandoned']} 次"
            )

        return nav_df
