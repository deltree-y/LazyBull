"""测试每日回测进度日志。"""

import pandas as pd

from src.lazybull.backtest import BacktestEngine
from src.lazybull.signals.base import Signal
from src.lazybull.universe.base import Universe


class MockUniverse(Universe):
    """模拟股票池。"""

    def get_stocks(self, date, quote_data=None):
        return []


class MockSignal(Signal):
    """模拟信号生成器。"""

    def __init__(self, top_n=20):
        super().__init__("mock")
        self.top_n = top_n

    def generate(self, date, universe, data):
        return {}


def test_daily_progress_log_includes_position_count_and_exposure():
    """每日回测日志应展示当前持仓数、目标持仓数和仓位比例。"""
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=20),
        initial_capital=100000.0,
        rebalance_freq=20,
        verbose=False,
    )
    engine.positions = {f"stock_{i}": {} for i in range(17)}
    engine.current_capital = 15000.0
    engine._last_rebalance_nav = 100240.48

    log_line = engine._format_daily_progress_log(
        date=pd.Timestamp("2025-12-29"),
        trading_days=122,
        total_days=124,
        cycle_day=2,
        portfolio_value=115840.0,
    )

    assert log_line.startswith("回测[2025-12-29]: 122/124 天 - 本轮第[2/20]天")
    assert "持仓/仓位[17/20]/[87.05%]" in log_line
    assert "收益:本调仓/本轮/年化:+15.56%/+15.84%/+35.49%)" in log_line


def test_target_position_count_scales_with_stagger_tranches():
    """分批调仓时目标持仓数应按批次数放大。"""
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=5),
        initial_capital=100000.0,
        rebalance_freq=5,
        stagger_tranches=3,
        verbose=False,
    )

    assert engine._get_target_position_count() == 15