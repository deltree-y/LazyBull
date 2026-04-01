"""测试每日回测进度日志。"""

import pandas as pd
import src.lazybull.backtest.engine as engine_module

from src.lazybull.backtest import BacktestEngine
from src.lazybull.backtest.engine_ml import BacktestEngineML
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
    """每日回测日志应展示当前持仓数、目标持仓数、仓位比例和 ATR 占位符。"""
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
    assert "收益:本调仓/本轮/年化:[+15.56%/+15.84%/+35.49%]" in log_line
    assert "ATR(min/avg/max):[N/A/N/A/N/A]" in log_line


def test_ml_daily_progress_log_includes_position_atr_summary():
    """ML 回测日志应展示当前持仓股票的 atr_pct_14 统计。"""
    engine = BacktestEngineML(
        features_by_date={
            "20251229": pd.DataFrame(
                {
                    "ts_code": ["stock_0", "stock_1", "stock_2", "other"],
                    "atr_pct_14": [0.0123, 0.0234, 0.0345, 0.0999],
                }
            )
        },
        universe=MockUniverse(),
        signal=MockSignal(top_n=20),
        initial_capital=100000.0,
        rebalance_freq=20,
        verbose=False,
    )
    engine.positions = {f"stock_{i}": {} for i in range(3)}
    engine._last_rebalance_nav = 100000.0

    log_line = engine._format_daily_progress_log(
        date=pd.Timestamp("2025-12-29"),
        trading_days=1,
        total_days=20,
        cycle_day=1,
        portfolio_value=100000.0,
    )

    assert "ATR(min/avg/max):[1.23%/2.34%/3.45%]" in log_line


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


def test_cycle_separator_logs_only_on_first_day(monkeypatch):
    """每轮分隔线只应在本轮第1天前输出一次。"""
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=5),
        initial_capital=100000.0,
        rebalance_freq=2,
        verbose=False,
        enable_pending_order=False,
        enable_position_completion=False,
    )

    log_messages = []
    separator_line = "\n================================================ 新一轮回测 ================================================="

    monkeypatch.setattr(engine_module.logger, "info", lambda message: log_messages.append(str(message)))
    monkeypatch.setattr(engine, "_prepare_price_index", lambda price_data: None)
    monkeypatch.setattr(engine, "_get_rebalance_dates", lambda trading_dates: {})
    monkeypatch.setattr(engine, "_calculate_portfolio_value", lambda date: 100000.0)
    monkeypatch.setattr(engine, "_generate_nav_curve", lambda: pd.DataFrame())

    trading_dates = [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-06"),
    ]
    price_data = pd.DataFrame(columns=["ts_code", "trade_date", "close"])

    engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=trading_dates,
        price_data=price_data,
    )

    assert log_messages.count(separator_line) == 2