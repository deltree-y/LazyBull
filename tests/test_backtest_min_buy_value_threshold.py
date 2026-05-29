"""测试回测最小买入后市值阈值（与纸面交易口径一致）。"""

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


def test_buy_is_skipped_when_below_min_buy_value_threshold(monkeypatch):
    """当买入后市值低于阈值时，应跳过买入。"""
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=20),
        initial_capital=100000.0,
        rebalance_freq=20,
        min_buy_value_ratio=0.2,
        verbose=False,
    )
    trade_date = pd.Timestamp("2025-01-02")

    monkeypatch.setattr(engine, "_get_trade_price", lambda date, stock: 9.0)
    monkeypatch.setattr(engine, "_get_pnl_price", lambda date, stock: 9.0)
    monkeypatch.setattr(engine, "_calculate_portfolio_value", lambda date: 100000.0)

    # 阈值=100000/20*0.2=1000；本单按100股取整后金额=900，应被拦截
    engine._buy_stock_direct(trade_date, "000001.SZ", target_value=900.0)

    assert "000001.SZ" not in engine.positions
    assert len(engine.trades) == 0


def test_buy_executes_when_min_buy_value_ratio_disabled(monkeypatch):
    """阈值关闭时，小额但满足一手约束的买入应正常成交。"""
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=20),
        initial_capital=100000.0,
        rebalance_freq=20,
        min_buy_value_ratio=0.0,
        verbose=False,
    )
    trade_date = pd.Timestamp("2025-01-02")

    monkeypatch.setattr(engine, "_get_trade_price", lambda date, stock: 9.0)
    monkeypatch.setattr(engine, "_get_pnl_price", lambda date, stock: 9.0)

    engine._buy_stock_direct(trade_date, "000001.SZ", target_value=900.0)

    assert "000001.SZ" in engine.positions
    assert len(engine.trades) == 1
    assert engine.trades[0]["amount"] == 900.0
