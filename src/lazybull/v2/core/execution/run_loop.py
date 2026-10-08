"""回测主循环 mixin（P2a-T5，行为冻结搬运件）。

来源：旧 ``backtest/run_loop.py`` 的 ``BacktestRunLoopMixin``（T0 规划 §3.3）。
类名、公开签名 ``run(self, ...)``、全部语句的执行顺序 / 数值操作 / 日志文本
逐字保留。

退役组件摘除点（exposure 政策族；B0 基线下这些方法内部守卫 no-op，摘除逐位等价）：
1. 旧 :124 ``self._execute_pending_exposure_trims(date, trading_dates, date_to_idx)``
   （连同上方注释「执行风控减仓…」一并删除）
2. 旧 :132 ``self._queue_exposure_trim(date, trading_dates, date_to_idx)``
   （连同注释「暴露门控每日判定…」一并删除）
3. 旧 :135 ``self._queue_exposure_replenish(date, trading_dates, date_to_idx)``
   （连同注释「暴露门控对称回补判定…」一并删除）
4. 旧 :141 ``self._execute_pending_exposure_replenishes(date, trading_dates, date_to_idx)``
   （连同注释「执行对称回补买入…」一并删除）

D8 提取式拆分：旧 ``run`` 在复杂度基线在册（func_lines / mccabe 28 / nesting 超限），
直接复制将构成新增硬超限，故按「提取 + 委托」把主循环阶段块拆为私有方法
（``_run_signal_day`` / ``_run_daily_trade_executions`` / ``_try_early_rebalance``
族 / ``_finalize_day`` / ``_log_run_stats``）；提取仅移动语句位置，公开签名与
全部语句的执行顺序、数值操作、日志逐字保留。
（``_run_daily_trade_executions`` 为 T5 收尾补提：black 稳定形态下 ``run``
仍超 150 行硬上限，把「先卖后买」当日执行序列整体移出。）

依赖：本模块无项目内 import（旧文件同样只依赖 time / typing / pandas / loguru），
无改指项。
"""

import time
from typing import Dict, List

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
        # 在线政策 provider（P2-5）需要当日交易日位置（剩余持有交易日口径与快照一致）
        self._trade_date_index = date_to_idx

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
            cycle_separator = (
                "\n================================================"
                " 新一轮回测 ================================================="
            )
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
                    cycle_day = self._run_signal_day(
                        idx,
                        date,
                        cycle_day,
                        signal_dates,
                        trading_dates,
                        price_data,
                        date_to_idx,
                        cycle_separator,
                    )

                    # @2026/01/18: 先卖后买的当日执行序列（D8 提取，见 _run_daily_trade_executions）
                    self._run_daily_trade_executions(date, trading_dates, date_to_idx)

                    # 空仓提前调仓 / 盈利延续拖尾提前调仓（D8 提取，见 _try_early_rebalance）
                    cycle_day = self._try_early_rebalance(
                        idx,
                        date,
                        cycle_day,
                        signal_dates,
                        trading_dates,
                        price_data,
                        date_to_idx,
                        cycle_separator,
                    )

                    # 处理仓位补齐（在补齐窗口期内尝试补齐未满仓位）
                    if self.enable_position_completion:
                        self._process_position_completion(
                            date, trading_dates, price_data, date_to_idx
                        )

                    # 计算当日组合价值
                    portfolio_value = self._calculate_portfolio_value(date)

                # 日终日志汇总 / 净值快照 / 持仓快照（D8 提取，见 _finalize_day）
                self._finalize_day(
                    idx,
                    date,
                    trade_start_idx,
                    total_days,
                    cycle_day,
                    portfolio_value,
                    trading_dates,
                    date_to_idx,
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

        self._log_run_stats()

        return nav_df

    def _run_daily_trade_executions(
        self,
        date: pd.Timestamp,
        trading_dates: List[pd.Timestamp],
        date_to_idx: Dict[pd.Timestamp, int],
    ) -> None:
        """先卖后买的当日执行序列（D8 提取；语句、顺序与注释逐字保留）。"""
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

    def _run_signal_day(
        self,
        idx: int,
        date: pd.Timestamp,
        cycle_day: int,
        signal_dates: Dict[pd.Timestamp, int],
        trading_dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
        date_to_idx: Dict[pd.Timestamp, int],
        cycle_separator: str,
    ) -> int:
        """信号生成日处理（D8 提取自 ``run`` 主循环；语句逐字保留）。

        Returns:
            更新后的 cycle_day（信号成功入队列且 idx 非 anchor 时重置为 1）。
        """
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
        return cycle_day

    def _try_early_rebalance(
        self,
        idx: int,
        date: pd.Timestamp,
        cycle_day: int,
        signal_dates: Dict[pd.Timestamp, int],
        trading_dates: List[pd.Timestamp],
        price_data: pd.DataFrame,
        date_to_idx: Dict[pd.Timestamp, int],
        cycle_separator: str,
    ) -> int:
        """空仓提前调仓 / 盈利延续拖尾提前调仓（D8 提取自 ``run`` 主循环；语句逐字保留）。

        场景 A（空仓）：持仓全部卖出，资金闲置 → 立即触发新一轮信号
        场景 B（盈利延续拖尾）：cycle_day >= holding_period 但仍有残留持仓（通常为盈利延续）
          → 若"残留持仓占比 + 新信号目标仓位 ≤ 100%"，则提前启动新一轮；否则继续等待

        Returns:
            更新后的 cycle_day（信号成功入队列且 idx 非 anchor 时重置为 1）。
        """
        early_rebalance_guards_ok = self._early_rebalance_guards_ok(date, signal_dates)

        is_empty_position = not self.positions
        is_holding_period_exceeded = bool(self.positions) and cycle_day >= self.holding_period

        if early_rebalance_guards_ok and (is_empty_position or is_holding_period_exceeded):
            # 旧实现此处计算残留占比但返回值从不被读取（随后重新计算覆盖）；
            # 求值本身带 last_known_price 缓存写回副作用，行为冻结保留调用、不绑定变量
            self._early_rebalance_residual_ratio(date, is_empty_position)

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
            signal_accepted = self._validate_early_rebalance_signal(
                date, is_holding_period_exceeded
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
                cycle_day = self._accept_early_rebalance(
                    idx,
                    date,
                    cycle_day,
                    signal_dates,
                    date_to_idx,
                    is_empty_position,
                    cycle_separator,
                )
        return cycle_day

    def _early_rebalance_guards_ok(
        self, date: pd.Timestamp, signal_dates: Dict[pd.Timestamp, int]
    ) -> bool:
        """提前调仓前置守卫（D8 提取；短路求值顺序逐字保留）。"""
        return (
            self.enable_early_rebalance_on_empty
            and not self.pending_signals
            and not any(
                slot_info.get("unfilled_count", 0) > 0 for slot_info in self.unfilled_slots.values()
            )
            and date not in signal_dates
        )

    def _early_rebalance_residual_ratio(self, date: pd.Timestamp, is_empty_position: bool) -> float:
        """盈利延续拖尾场景残留持仓占比（D8 提取；语句逐字保留）。

        注：旧实现中该返回值在调用点未被读取（随后会被重新计算覆盖），
        行为冻结原样保留该次 ``_calculate_portfolio_value`` 求值。
        """
        if not is_empty_position:
            # 盈利延续拖尾场景：打印当前残留持仓占比
            current_nav = self._calculate_portfolio_value(date)
            residual_market_value = current_nav - self.current_capital
            residual_ratio = residual_market_value / current_nav if current_nav > 0 else 0.0
        else:
            residual_ratio = 0.0
        return residual_ratio

    def _validate_early_rebalance_signal(
        self, date: pd.Timestamp, is_holding_period_exceeded: bool
    ) -> bool:
        """校验提前调仓信号是否真正入队列（D8 提取；语句逐字保留）。"""
        signal_accepted = date in self.pending_signals
        if signal_accepted and is_holding_period_exceeded:
            current_nav = self._calculate_portfolio_value(date)
            residual_market_value = current_nav - self.current_capital
            residual_ratio = residual_market_value / current_nav if current_nav > 0 else 0.0
            new_signal_weight_sum = sum(self.pending_signals[date].get("signals", {}).values())
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
        return signal_accepted

    def _accept_early_rebalance(
        self,
        idx: int,
        date: pd.Timestamp,
        cycle_day: int,
        signal_dates: Dict[pd.Timestamp, int],
        date_to_idx: Dict[pd.Timestamp, int],
        is_empty_position: bool,
        cycle_separator: str,
    ) -> int:
        """信号真正入队列后的节奏更新与预定调仓日清理（D8 提取；语句逐字保留）。

        Returns:
            更新后的 cycle_day（idx 非 anchor 时重置为 1）。
        """
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
        return cycle_day

    def _finalize_day(
        self,
        idx: int,
        date: pd.Timestamp,
        trade_start_idx: int,
        total_days: int,
        cycle_day: int,
        portfolio_value: float,
        trading_dates: List[pd.Timestamp],
        date_to_idx: Dict[pd.Timestamp, int],
    ) -> None:
        """日终日志汇总 / 净值快照 / 持仓快照（D8 提取自 ``run`` 主循环；语句逐字保留）。"""
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
        # 政策旁路（terminal_loss P2-1）：逐日持仓快照，仅记录不参与决策
        self._record_holdings_snapshot(
            date=date,
            portfolio_value=portfolio_value,
            trading_dates=trading_dates,
            date_to_idx=date_to_idx,
        )

    def _log_run_stats(self) -> None:
        """回测结束统计日志（D8 提取自 ``run`` 尾部；语句逐字保留）。"""
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
