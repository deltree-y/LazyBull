"""验证 BacktestEngine 信号执行方法的实现归属。"""

from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.backtest.engine_ml import BacktestEngineML
from src.lazybull.backtest.signal_execution import BacktestSignalExecutionMixin


def test_backtest_engine_signal_execution_owned_by_mixin() -> None:
    """关键信号执行方法应来自 signal_execution 模块中的 mixin。"""
    assert BacktestEngine._build_signal_data is BacktestSignalExecutionMixin._build_signal_data
    assert (
        BacktestEngine._post_filter_candidates
        is BacktestSignalExecutionMixin._post_filter_candidates
    )
    assert (
        BacktestEngine._get_position_weight_for_planning
        is BacktestSignalExecutionMixin._get_position_weight_for_planning
    )
    assert (
        BacktestEngine._queue_condition_sell_refill_signal
        is BacktestSignalExecutionMixin._queue_condition_sell_refill_signal
    )
    assert (
        BacktestEngine._get_holding_features_row
        is BacktestSignalExecutionMixin._get_holding_features_row
    )
    assert BacktestEngine._generate_signal is BacktestSignalExecutionMixin._generate_signal


def test_backtest_engine_ml_overrides_still_effective() -> None:
    """BacktestEngineML 的 hook 覆写应保持生效。"""
    assert BacktestEngineML._build_signal_data is not BacktestEngine._build_signal_data
    assert BacktestEngineML._post_filter_candidates is not BacktestEngine._post_filter_candidates
    assert (
        BacktestEngineML._get_holding_features_row is not BacktestEngine._get_holding_features_row
    )
