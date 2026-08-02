"""验证 BacktestEngine.run 的实现归属。"""

from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.backtest.run_loop import BacktestRunLoopMixin


def test_backtest_engine_run_owned_by_run_loop_mixin() -> None:
    """run 方法应来自 run_loop 模块中的 BacktestRunLoopMixin。"""
    assert BacktestEngine.run is BacktestRunLoopMixin.run
    assert BacktestEngine.run.__module__ == "src.lazybull.backtest.run_loop"
