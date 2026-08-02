"""验证 BacktestEngine 延迟订单执行方法的实现归属。"""

from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.backtest.pending_execution import BacktestPendingExecutionMixin


def test_backtest_engine_pending_execution_owned_by_mixin() -> None:
    """延迟订单方法应来自 pending_execution 模块中的 mixin。"""
    assert (
        BacktestEngine._record_pending_order_event
        is BacktestPendingExecutionMixin._record_pending_order_event
    )
    assert (
        BacktestEngine._process_pending_orders
        is BacktestPendingExecutionMixin._process_pending_orders
    )
    assert (
        BacktestEngine._record_pending_order_event.__module__
        == "src.lazybull.backtest.pending_execution"
    )
    assert (
        BacktestEngine._process_pending_orders.__module__
        == "src.lazybull.backtest.pending_execution"
    )
