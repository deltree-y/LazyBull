"""验证 BacktestEngine 买入执行方法的实现归属。"""

from src.lazybull.backtest.buy_execution import BacktestBuyExecutionMixin
from src.lazybull.backtest.engine import BacktestEngine


def test_backtest_engine_buy_execution_owned_by_mixin() -> None:
    """关键买入执行方法应来自 buy_execution 模块中的 mixin。"""
    assert BacktestEngine._execute_pending_buys is BacktestBuyExecutionMixin._execute_pending_buys
    assert (
        BacktestEngine._process_position_completion
        is BacktestBuyExecutionMixin._process_position_completion
    )
    assert (
        BacktestEngine._buy_stock_with_status_check
        is BacktestBuyExecutionMixin._buy_stock_with_status_check
    )
    assert (
        BacktestEngine._build_position_extra_info
        is BacktestBuyExecutionMixin._build_position_extra_info
    )
    assert BacktestEngine._buy_stock_direct is BacktestBuyExecutionMixin._buy_stock_direct
    assert BacktestEngine._buy_stock is BacktestBuyExecutionMixin._buy_stock
    assert (
        BacktestEngine._update_completion_attribution
        is BacktestBuyExecutionMixin._update_completion_attribution
    )
