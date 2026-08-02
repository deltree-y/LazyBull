"""回测报告与日志相关方法。"""

import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def _format_rebalance_decision_summary(
    decision_trace: Dict,
    execution_date: Optional[pd.Timestamp] = None,
    tranche_tag: str = "",
) -> str:
    """格式化统一的调仓决策摘要日志。"""

    def _to_optional_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            if np.isnan(value):
                return None
        except TypeError:
            return None
        return float(value)

    def _compact_summary(summary: Optional[str]) -> str:
        if not summary:
            return "-"

        compact = str(summary).strip()
        compact = compact.replace("，", ", ")
        compact = compact.replace("达到阈值 ", "档=")
        compact = compact.replace("未达到首档阈值 ", "未达首档=")
        compact = compact.replace("目标仓位 ", "目标=")
        compact = re.sub(r",?\s*市场层=[0-9.]+%", "", compact)
        compact = re.sub(r"\s+", " ", compact)
        return compact

    def _fmt_exposure(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        try:
            if np.isnan(value):
                return "N/A"
        except TypeError:
            return "N/A"
        return f"{float(value):.1%}"

    signal_date = decision_trace.get("signal_date")
    signal_label = signal_date.date() if isinstance(signal_date, pd.Timestamp) else signal_date
    execution_label = (
        execution_date.date() if isinstance(execution_date, pd.Timestamp) else execution_date
    )
    candidate_count = decision_trace.get("candidate_count")
    target_n = decision_trace.get("target_n", 0)
    queued = bool(decision_trace.get("queued", execution_date is not None))
    final_target_exposure = _to_optional_float(decision_trace.get("final_target_exposure", 1.0))

    header = f"{tranche_tag}调仓决策摘要: 信号日 {signal_label}"
    if execution_label is not None:
        execution_text = execution_label
    else:
        execution_text = "-"

    candidate_text = candidate_count if candidate_count is not None else "N/A"

    topn_text = f"目标={target_n}"

    final_action = "入队" if queued else "不入队"

    if final_target_exposure is not None and final_target_exposure <= 0:
        final_detail = "阻断, 不入队"
    else:
        final_detail = final_action

    return (
        f"{header} | 执行={execution_text} | 候选={candidate_text} | {topn_text}"
        f" | 最终={_fmt_exposure(final_target_exposure)}[{final_detail}]"
    )


class BacktestReportingMixin:
    """回测报告与日志输出能力。"""

    def _collect_deferred_log(self, message) -> None:
        """收集单个交易日内暂缓输出的日志。"""
        record = message.record
        self._deferred_day_logs.append(
            {
                "level": record["level"].name,
                "message": record["message"],
            }
        )

    def _emit_immediate_log(self, level: str, message: str, colors: bool = False) -> None:
        """绕过单日缓冲，立即输出日志。"""
        bound_logger = logger.bind(_defer_emit=False)
        if colors:
            bound_logger.opt(colors=True).log(level.upper(), message)
            return
        bound_logger.log(level.upper(), message)

    def _emit_daily_summary_log(self, message: str) -> None:
        """输出每日顶格彩色总结行。"""
        self._emit_immediate_log(
            "INFO",
            f"<bold><cyan>{message}</cyan></bold>",
            colors=True,
        )

    def _normalize_deferred_log_message(self, message: str) -> Optional[str]:
        """将缓冲日志统一整理为两空格缩进格式。"""
        lines = [line.rstrip() for line in str(message).splitlines() if line.strip()]
        if not lines:
            return None
        return "\n".join(f"  {line.lstrip()}" for line in lines)

    def _flush_deferred_day_logs(
        self,
        predicate: Optional[Callable[[Dict[str, str]], bool]] = None,
    ) -> None:
        """在每日总结之后统一回放明细日志。"""
        remaining_logs: List[Dict[str, str]] = []
        for record in self._deferred_day_logs:
            if predicate is not None and not predicate(record):
                remaining_logs.append(record)
                continue
            normalized = self._normalize_deferred_log_message(record["message"])
            if normalized:
                self._emit_immediate_log(record["level"], normalized)
        self._deferred_day_logs = remaining_logs if predicate is not None else []

    @staticmethod
    def _format_compact_items(items: List[str], limit: int = 4) -> str:
        """压缩同类股票列表，避免单行过长。"""
        if not items:
            return "-"
        visible = items[:limit]
        suffix = f", ...+{len(items) - limit}" if len(items) > limit else ""
        return ", ".join(visible) + suffix

    @staticmethod
    def _format_trade_cash_wan(trade: Dict) -> str:
        """格式化买入现金支出（万元）。"""
        amount = float(trade.get("amount", 0.0) or 0.0)
        cost = float(trade.get("cost", 0.0) or 0.0)
        total_cash = max(amount + cost, 0.0)
        return f"{total_cash / 10000:.1f}w"

    def _build_daily_trade_log(
        self,
        date: pd.Timestamp,
        trade_start_idx: int,
        date_to_idx: Dict,
    ) -> Tuple[int, int, List[str]]:
        """构建当日实际成交摘要。"""
        day_trades = [trade for trade in self.trades[trade_start_idx:] if trade.get("date") == date]
        if not day_trades:
            return 0, 0, []

        buy_items: List[str] = []
        sell_items: List[str] = []
        current_idx = date_to_idx.get(date)

        for trade in day_trades:
            stock = str(trade.get("stock", "-"))
            action = trade.get("action")
            if action == "buy":
                buy_items.append(f"{stock}({self._format_trade_cash_wan(trade)})")
                continue

            if action != "sell":
                continue

            buy_date = trade.get("buy_date")
            holding_days = 0
            if isinstance(buy_date, pd.Timestamp) and current_idx is not None:
                buy_idx = date_to_idx.get(buy_date)
                if buy_idx is not None:
                    holding_days = max(current_idx - buy_idx, 0)

            profit_pct = float(trade.get("pnl_profit_pct", 0.0) or 0.0)
            sell_items.append(f"{stock}({holding_days}d,{profit_pct:+.1%})")

        if not buy_items and not sell_items:
            return 0, 0, []

        lines = []
        if sell_items:
            lines.append(f"交易: 卖{len(sell_items)}[{', '.join(sell_items)}]")
        if buy_items:
            lines.append(f"交易: 买{len(buy_items)}[{', '.join(buy_items)}]")

        return len(buy_items), len(sell_items), lines

    def _format_completion_summary(
        self,
        success_items: List[Dict],
        delayed_slots: List[str],
        remaining_count: int,
    ) -> str:
        """压缩仓位补齐日志。"""
        parts = []
        if success_items:
            success_labels = []
            for item in success_items:
                slot = item["slot"]
                buy = item["buy"]
                success_labels.append(buy if slot == buy else f"{slot}→{buy}")
            parts.append(f"成功{len(success_items)}[{self._format_compact_items(success_labels)}]")
        if delayed_slots:
            parts.append(f"延迟{len(delayed_slots)}[{self._format_compact_items(delayed_slots)}]")
        if remaining_count > 0:
            parts.append(f"待补{remaining_count}")
        return f"补齐: {' | '.join(parts)}"

    def _reset_daily_warning_items(self) -> None:
        """重置当日需汇总展示的压缩事件。"""
        self._daily_warning_items = {
            "early_rebalance": [],
            "duplicate_buy": [],
            "position_unfilled": [],
            "completion_skipped": [],
            "completion_abandoned": [],
            "pending_order_added": [],
            "pending_order_success": [],
            "pending_order_expired": [],
        }

    def _record_completion_skip(self, label: str, detail: str) -> None:
        """记录补齐跳过原因，日终统一压缩显示。"""
        self._daily_warning_items.setdefault("completion_skipped", []).append(
            {"label": label, "detail": detail}
        )

    @staticmethod
    def _format_pending_order_group(
        items: List[Dict],
        action_label: str,
        item_formatter,
        count_prefix: str = "",
    ) -> str:
        """格式化延迟订单分组摘要。"""
        labels = [item_formatter(item) for item in items]
        prefix = f"{count_prefix}{action_label}{len(items)}"
        return f"{prefix}[{BacktestReportingMixin._format_compact_items(labels, limit=6)}]"

    def _record_duplicate_buy_skip(self, stock: str, buy_date: pd.Timestamp) -> None:
        """记录重复买入跳过，日终统一压缩显示。"""
        self._daily_warning_items.setdefault("duplicate_buy", []).append(
            {"stock": stock, "buy_date": buy_date}
        )

    def _record_position_unfilled_summary(
        self,
        tranche_tag: str,
        target_n: int,
        actually_bought: int,
        unfilled_count: int,
        unfilled_stocks: List[str],
    ) -> None:
        """记录仓位未满摘要。"""
        self._daily_warning_items.setdefault("position_unfilled", []).append(
            {
                "tranche_tag": tranche_tag.strip(),
                "target_n": int(target_n),
                "actually_bought": int(actually_bought),
                "unfilled_count": int(unfilled_count),
                "unfilled_stocks": list(unfilled_stocks),
            }
        )

    def _record_completion_abandoned_summary(
        self,
        tranche_tag: str,
        original_signal_date: pd.Timestamp,
        attempts: int,
        unfilled_stocks: List[str],
    ) -> None:
        """记录补齐放弃摘要。"""
        self._daily_warning_items.setdefault("completion_abandoned", []).append(
            {
                "tranche_tag": tranche_tag.strip(),
                "original_signal_date": original_signal_date,
                "attempts": int(attempts),
                "unfilled_stocks": list(unfilled_stocks),
            }
        )

    def _record_early_rebalance_summary(self, label: str, detail: str) -> None:
        """记录提前调仓相关事件，日终统一汇总。"""
        self._daily_warning_items.setdefault("early_rebalance", []).append(
            {"label": label, "detail": detail}
        )

    def _build_daily_signal_log(self, date: pd.Timestamp) -> Optional[str]:
        """构建当日新生成买卖信号的数量摘要。"""

        def _format_groups(groups: List[Tuple[str, int]]) -> str:
            return ", ".join(f"{label}{count}" for label, count in groups if count > 0)

        buy_groups: List[Tuple[str, int]] = []
        signal_data = self.pending_signals.get(date)
        if isinstance(signal_data, dict):
            if "signals" in signal_data:
                buy_count = len(signal_data.get("signals", {}))
                if buy_count > 0:
                    buy_label = "调仓" if signal_data.get("decision_trace") else "补槽"
                    buy_groups.append((buy_label, buy_count))
            elif signal_data:
                buy_groups.append(("调仓", len(signal_data)))

        sell_label_map = {
            "holding_period": "持有期",
            "rebalance": "调仓",
        }
        stop_loss_label_map = {
            "drawdown": "回撤止损",
            "trailing_stop": "移动止损",
            "consecutive_limit_down": "连续跌停",
            "unknown": "止损",
        }

        sell_counts: Dict[str, int] = {}
        for info in self.pending_condition_sells.values():
            if info.get("trigger_date") != date:
                continue
            label = sell_label_map.get(str(info.get("sell_type") or ""), "条件卖出")
            sell_counts[label] = sell_counts.get(label, 0) + 1

        for info in self.pending_stop_loss_sells.values():
            if info.get("trigger_date") != date:
                continue
            label = stop_loss_label_map.get(str(info.get("trigger_type") or "unknown"), "止损")
            sell_counts[label] = sell_counts.get(label, 0) + 1

        sell_groups: List[Tuple[str, int]] = []
        for label in (
            "调仓",
            "持有期",
            "回撤止损",
            "移动止损",
            "连续跌停",
            "止损",
            "条件卖出",
        ):
            count = sell_counts.get(label, 0)
            if count > 0:
                sell_groups.append((label, count))

        if not buy_groups and not sell_groups:
            return None

        parts = []
        if sell_groups:
            parts.append(f"卖[{_format_groups(sell_groups)}]")
        if buy_groups:
            parts.append(f"买[{_format_groups(buy_groups)}]")

        return f"信号: {' | '.join(parts)}"

    def _build_daily_warning_logs(self) -> List[str]:
        """构建需在每日总结下展示的日级压缩摘要。"""
        lines: List[str] = []

        early_items = self._daily_warning_items.get("early_rebalance", [])
        if early_items:
            labels = [f"{item['label']}[{item['detail']}]" for item in early_items]
            lines.append(f"提前调仓: {self._format_compact_items(labels)}")

        duplicate_buy_items = self._daily_warning_items.get("duplicate_buy", [])
        if duplicate_buy_items:
            labels = [f"{item['stock']}({item['buy_date'].date()})" for item in duplicate_buy_items]
            lines.append(
                f"重复买入跳过: {len(duplicate_buy_items)}只"
                f"[{self._format_compact_items(labels, limit=6)}]"
            )

        position_unfilled_items = self._daily_warning_items.get("position_unfilled", [])
        if position_unfilled_items:
            labels = []
            for item in position_unfilled_items:
                prefix = f"{item['tranche_tag']} " if item["tranche_tag"] else ""
                labels.append(
                    f"{prefix}目标{item['target_n']}/实买{item['actually_bought']}/待补"
                    f"{item['unfilled_count']}[{self._format_compact_items(item['unfilled_stocks'], limit=6)}]"
                    f"/{self.completion_window_days}天"
                )
            lines.append(f"仓位未满: {self._format_compact_items(labels, limit=3)}")

        completion_skipped_items = self._daily_warning_items.get("completion_skipped", [])
        if completion_skipped_items:
            groups = []
            ordered_labels = [
                "当日无行情",
                "前日无行情",
                "无数据",
                "无候选",
                "候选已持仓",
                "候选不可交易",
            ]
            for label in ordered_labels:
                matched = [item for item in completion_skipped_items if item.get("label") == label]
                if not matched:
                    continue
                details = [item["detail"] for item in matched]
                if label == "当日无行情":
                    groups.append(f"{label}{len(matched)}")
                else:
                    groups.append(
                        f"{label}{len(matched)}[{self._format_compact_items(details, limit=6)}]"
                    )
            if groups:
                lines.append(f"补齐跳过: {' | '.join(groups)}")

        completion_abandoned_items = self._daily_warning_items.get("completion_abandoned", [])
        if completion_abandoned_items:
            labels = []
            for item in completion_abandoned_items:
                prefix = f"{item['tranche_tag']} " if item["tranche_tag"] else ""
                labels.append(
                    f"{prefix}信号日{item['original_signal_date'].date()}/尝试{item['attempts']}次/"
                    f"剩{len(item['unfilled_stocks'])}[{self._format_compact_items(item['unfilled_stocks'], limit=6)}]"
                )
            lines.append(f"补齐放弃: {self._format_compact_items(labels, limit=3)}")

        pending_order_added_items = self._daily_warning_items.get("pending_order_added", [])
        if pending_order_added_items:
            groups = []
            for action in ("buy", "sell"):
                action_items = [
                    item for item in pending_order_added_items if item.get("action") == action
                ]
                if not action_items:
                    continue
                groups.append(
                    self._format_pending_order_group(
                        action_items,
                        action_label="买" if action == "buy" else "卖",
                        count_prefix="新增",
                        item_formatter=lambda item: f"{item['stock']}({item['reason']})",
                    )
                )
            if groups:
                lines.append(f"延迟订单: {' | '.join(groups)}")

        pending_order_success_items = self._daily_warning_items.get("pending_order_success", [])
        if pending_order_success_items:
            groups = []
            for action in ("buy", "sell"):
                action_items = [
                    item for item in pending_order_success_items if item.get("action") == action
                ]
                if not action_items:
                    continue
                groups.append(
                    self._format_pending_order_group(
                        action_items,
                        action_label="买" if action == "buy" else "卖",
                        count_prefix="成功",
                        item_formatter=lambda item: (
                            f"{item['stock']}(重{item['retry_count']},延{item['delay_days']}d)"
                        ),
                    )
                )
            if groups:
                lines.append(f"延迟订单成交: {' | '.join(groups)}")

        pending_order_expired_items = self._daily_warning_items.get("pending_order_expired", [])
        if pending_order_expired_items:
            groups = []
            for expire_type, label in (
                ("expired_retry", "超次"),
                ("expired_days", "超期"),
            ):
                for action in ("buy", "sell"):
                    action_items = [
                        item
                        for item in pending_order_expired_items
                        if item.get("expire_type") == expire_type and item.get("action") == action
                    ]
                    if not action_items:
                        continue
                    groups.append(
                        self._format_pending_order_group(
                            action_items,
                            action_label=("买" if action == "buy" else "卖"),
                            count_prefix=label,
                            item_formatter=lambda item: (
                                f"{item['stock']}(重{item['retry_count']}>{item['max_retry_count']})"
                                if item.get("expire_type") == "expired_retry"
                                else f"{item['stock']}(延{item['delay_days']}d>{item['max_retry_days']}d)"
                            ),
                        )
                    )
            if groups:
                lines.append(f"延迟订单放弃: {' | '.join(groups)}")

        return lines

    def _calculate_current_exposure_pct(self, portfolio_value: float) -> float:
        """按当日组合市值计算股票仓位比例。"""
        if portfolio_value <= 0:
            return 0.0

        market_value = max(portfolio_value - self.current_capital, 0.0)
        exposure_pct = market_value / portfolio_value * 100
        return min(exposure_pct, 100.0)

    def _initialize_decision_trace_for_signal(self, decision_trace: Dict) -> Dict:
        """扩展点：子类可补充市场层占位信息。"""
        return decision_trace

    def _finalize_decision_trace_for_signal_day(
        self, decision_trace: Dict, signal_date: pd.Timestamp
    ) -> Dict:
        """扩展点：子类可在信号日补齐摘要所需状态。"""
        return decision_trace

    def _build_signal_decision_trace(
        self,
        date: pd.Timestamp,
        target_n: int,
        candidate_count: int,
        tranche_idx: int,
    ) -> Dict:
        """构建调仓决策摘要所需的状态。"""
        trace = {
            "signal_date": date,
            "target_n": target_n,
            "candidate_count": candidate_count,
            "tranche_idx": tranche_idx,
            "queued": False,
            "market_regime": {
                "enabled": False,
                "exposure": 1.0,
                "summary": "未启用",
            },
            "market_layer_exposure": 1.0,
            "final_target_exposure": 1.0,
        }
        return self._initialize_decision_trace_for_signal(trace)

    def _mark_decision_trace_blocked(self, decision_trace: Dict) -> Dict:
        """标记该信号未进入待买队列。"""
        decision_trace["queued"] = False
        decision_trace["final_target_exposure"] = 0.0
        if decision_trace.get("market_regime", {}).get("enabled"):
            decision_trace["market_regime"]["exposure"] = None
            decision_trace["market_regime"]["summary"] = "未评估（信号已阻断）"
        decision_trace["market_layer_exposure"] = None
        return decision_trace

    def _log_rebalance_decision_summary(
        self,
        decision_trace: Dict,
        execution_date: Optional[pd.Timestamp] = None,
        tranche_tag: str = "",
    ) -> None:
        """统一输出调仓决策摘要。"""
        logger.info(
            _format_rebalance_decision_summary(
                decision_trace=decision_trace,
                execution_date=execution_date,
                tranche_tag=tranche_tag,
            )
        )

    def _get_current_position_atr_stats(
        self, date: pd.Timestamp
    ) -> Optional[Tuple[float, float, float]]:
        """获取当日持仓 ATR% 统计（子类可覆写）。"""
        return None

    def _format_current_position_atr_stats(self, date: pd.Timestamp) -> str:
        """格式化当日持仓 ATR% 统计。"""
        atr_stats = self._get_current_position_atr_stats(date)
        if atr_stats is None:
            return "ATR:[N/A/N/A/N/A]"

        min_atr_pct, avg_atr_pct, max_atr_pct = atr_stats
        return f"ATR:[" f"{min_atr_pct:.2%}/{avg_atr_pct:.2%}/{max_atr_pct:.2%}]"

    def _format_daily_progress_log(
        self,
        date: pd.Timestamp,
        trading_days: int,
        total_days: int,
        cycle_day: int,
        portfolio_value: float,
        buy_count: int = 0,
        sell_count: int = 0,
    ) -> str:
        """格式化每日回测进度日志。"""
        total_return = (portfolio_value / self.initial_capital - 1) * 100
        # 简单年化收益率（不假设收益再投入）
        simple_annual = (
            (total_return / 100) * (252 / trading_days) * 100 if trading_days > 0 else 0.0
        )
        ann_return = simple_annual
        rebalance_return_str = (
            f"{(portfolio_value / self._last_rebalance_nav - 1) * 100:+.2f}%"
            if self._last_rebalance_nav and self._last_rebalance_nav > 0
            else "N/A"
        )
        target_position_count = self._get_target_position_count()
        current_exposure_pct = self._calculate_current_exposure_pct(portfolio_value)
        t_index = max(cycle_day - 1, 0)

        return (
            f"T{t_index}[{date.date()}]: {trading_days:0{len(str(total_days))}}/{total_days} 天"
            f" | 本轮[{cycle_day:0{len(str(self.rebalance_freq))}}/{self.rebalance_freq}]"
            f" | 持仓/仓位[{len(self.positions):0{len(str(target_position_count))}}/{target_position_count}]"
            f"/[{current_exposure_pct:.2f}%]"
            f" | 买/卖[{buy_count}/{sell_count}]"
            f" | 收益[本调仓/本轮/年化]=[{rebalance_return_str}/{total_return:+.2f}%/{ann_return:+.2f}%]"
        )
