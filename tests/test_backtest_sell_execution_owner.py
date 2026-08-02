"""验证 BacktestEngine 卖出执行方法的实现归属。"""

from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.backtest.sell_execution import BacktestSellExecutionMixin


def test_backtest_engine_sell_execution_owned_by_mixin() -> None:
    """关键卖出执行方法应来自 sell_execution 模块中的 mixin。"""
    assert (
        BacktestEngine._queue_rebalance_sells is BacktestSellExecutionMixin._queue_rebalance_sells
    )
    assert BacktestEngine._check_and_sell is BacktestSellExecutionMixin._check_and_sell
    assert (
        BacktestEngine._execute_pending_condition_sells
        is BacktestSellExecutionMixin._execute_pending_condition_sells
    )
    assert BacktestEngine._check_stop_loss is BacktestSellExecutionMixin._check_stop_loss
    assert (
        BacktestEngine._execute_pending_stop_loss_sells
        is BacktestSellExecutionMixin._execute_pending_stop_loss_sells
    )
    assert BacktestEngine._sell_stock is BacktestSellExecutionMixin._sell_stock
    assert (
        BacktestEngine._sell_stock_with_status_check
        is BacktestSellExecutionMixin._sell_stock_with_status_check
    )
    assert BacktestEngine._sell_stock_direct is BacktestSellExecutionMixin._sell_stock_direct
