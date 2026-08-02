"""测试回测 reporting 模块的归属与兼容导出。"""

import src.lazybull.backtest.engine as engine_module
from src.lazybull.backtest import BacktestEngine
from src.lazybull.backtest.reporting import (
    BacktestReportingMixin,
)
from src.lazybull.backtest.reporting import (
    _format_rebalance_decision_summary as reporting_formatter,
)


def test_rebalance_formatter_is_reexported_from_engine_module():
    """engine 顶层 formatter 应与 reporting 模块保持同一对象。"""
    assert engine_module._format_rebalance_decision_summary is reporting_formatter


def test_reporting_methods_owned_by_mixin():
    """指定方法应由 BacktestReportingMixin 提供，不在 BacktestEngine 内重复定义。"""
    method_names = [
        "_collect_deferred_log",
        "_emit_immediate_log",
        "_emit_daily_summary_log",
        "_normalize_deferred_log_message",
        "_flush_deferred_day_logs",
        "_format_compact_items",
        "_format_trade_cash_wan",
        "_build_daily_trade_log",
        "_format_completion_summary",
        "_reset_daily_warning_items",
        "_record_completion_skip",
        "_format_pending_order_group",
        "_record_duplicate_buy_skip",
        "_record_position_unfilled_summary",
        "_record_completion_abandoned_summary",
        "_record_early_rebalance_summary",
        "_build_daily_signal_log",
        "_build_daily_warning_logs",
        "_calculate_current_exposure_pct",
        "_initialize_decision_trace_for_signal",
        "_finalize_decision_trace_for_signal_day",
        "_build_signal_decision_trace",
        "_mark_decision_trace_blocked",
        "_log_rebalance_decision_summary",
        "_get_current_position_atr_stats",
        "_format_current_position_atr_stats",
        "_format_daily_progress_log",
    ]

    for method_name in method_names:
        assert getattr(BacktestEngine, method_name) is getattr(BacktestReportingMixin, method_name)
        assert method_name not in BacktestEngine.__dict__


def test_dead_top_level_helpers_are_removed_from_engine_module():
    """已确认删除的死代码不应继续存在于 engine 顶层。"""
    assert not hasattr(engine_module, "_format_buy_execution_stock_list")
    assert not hasattr(engine_module, "_sum_buy_execution_weights")
    assert not hasattr(engine_module, "_format_buy_execution_summary")
