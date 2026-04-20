"""测试每日回测进度日志。"""

import pandas as pd
import src.lazybull.backtest.engine as engine_module

from src.lazybull.backtest import BacktestEngine
from src.lazybull.backtest.engine_ml import BacktestEngineML
from src.lazybull.common.cost import CostModel
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

    assert log_line.startswith("回测[2025-12-29]: 122/124 天 - 本轮第[02/20]天")
    assert "持仓/仓位[17/20]/[87.05%]" in log_line
    assert "收益:本调仓/本轮/年化:[+15.56%/+15.84%/+35.49%]" in log_line
    assert "ATR:[N/A/N/A/N/A]" in log_line


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

    assert "ATR:[1.23%/2.34%/3.45%]" in log_line


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
    """新一轮分隔线只在回测首日 + 每次信号成功入队列时输出一次。"""
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=5),
        initial_capital=100000.0,
        rebalance_freq=2,
        verbose=False,
        enable_pending_order=False,
        enable_position_completion=False,
        enable_early_rebalance_on_empty=False,
    )

    log_messages = []
    separator_line = "\n================================================ 新一轮回测 ================================================="

    monkeypatch.setattr(engine_module.logger, "info", lambda message: log_messages.append(str(message)))
    monkeypatch.setattr(engine, "_prepare_price_index", lambda price_data: None)
    # 在 trading_dates[2] 设一个信号日，让信号进入 pending_signals 触发新一轮分隔线
    monkeypatch.setattr(
        engine,
        "_get_rebalance_dates",
        lambda trading_dates: {trading_dates[2]: 0},
    )
    monkeypatch.setattr(engine, "_calculate_portfolio_value", lambda date: 100000.0)
    monkeypatch.setattr(engine, "_generate_nav_curve", lambda: pd.DataFrame())

    def fake_generate_signal(date, *args, **kwargs):
        engine.pending_signals[date] = {"signals": {}, "ranked_candidates": [], "target_n": 0, "tranche_idx": 0, "decision_trace": {}}

    monkeypatch.setattr(engine, "_generate_signal", fake_generate_signal)

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

    # 首日1次 + 信号日重置1次 = 2次
    assert log_messages.count(separator_line) == 2


def test_cycle_separator_is_logged_before_new_cycle_signal(monkeypatch):
    """新一轮分隔线应在信号成功入队列后输出（新语义：anchor 基于信号成功而非预定调仓日）。"""
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=5),
        initial_capital=100000.0,
        rebalance_freq=2,
        verbose=False,
        enable_pending_order=False,
        enable_position_completion=False,
        enable_early_rebalance_on_empty=False,
    )

    log_messages = []
    separator_line = "\n================================================ 新一轮回测 ================================================="

    monkeypatch.setattr(engine_module.logger, "info", lambda message: log_messages.append(str(message)))
    monkeypatch.setattr(engine, "_prepare_price_index", lambda price_data: None)
    monkeypatch.setattr(
        engine,
        "_get_rebalance_dates",
        lambda trading_dates: {trading_dates[2]: 0},
    )
    monkeypatch.setattr(engine, "_calculate_portfolio_value", lambda date: 100000.0)
    monkeypatch.setattr(engine, "_generate_nav_curve", lambda: pd.DataFrame())
    def fake_generate_signal(date, *args, **kwargs):
        log_messages.append("SIGNAL_GENERATED")
        engine.pending_signals[date] = {"signals": {}, "ranked_candidates": [], "target_n": 0, "tranche_idx": 0, "decision_trace": {}}

    monkeypatch.setattr(engine, "_generate_signal", fake_generate_signal)

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

    separator_indices = [i for i, message in enumerate(log_messages) if message == separator_line]
    signal_index = log_messages.index("SIGNAL_GENERATED")

    # 首日1次 + 信号日入队列后1次 = 2次
    assert len(separator_indices) == 2
    # 第二次分隔线在信号日志之后（新语义：anchor 更新发生在 _generate_signal 之后）
    assert separator_indices[1] > signal_index


def test_rebalance_buy_warning_log_uses_three_line_summary_format(monkeypatch):
    """调仓买入 warning 日志应按三行展示汇总、成功和失败信息。"""

    signal_date = pd.Timestamp("2023-01-02")
    buy_date = pd.Timestamp("2023-01-03")

    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=3),
        initial_capital=100000.0,
        cost_model=CostModel(
            commission_rate=0,
            min_commission=0,
            stamp_tax=0,
            slippage=0,
        ),
        rebalance_freq=10,
        verbose=False,
        enable_position_completion=True,
    )
    engine.current_capital = 60000.0
    engine.positions = {
        "600001.SH": {
            "shares": 1000,
            "buy_date": signal_date,
            "signal_date": signal_date,
            "buy_trade_price": 20.0,
            "buy_pnl_price": 20.0,
            "buy_cost_cash": 20000.0,
        }
    }

    trading_dates = [signal_date, buy_date]
    date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}

    price_data = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20230103",
                "close": 10.0,
                "close_adj": 10.0,
                "open": 10.0,
                "open_adj": 10.0,
                "vol": 1000000,
                "pct_chg": 0.0,
                "filter_is_suspended": 0,
                "is_suspended": 0,
                "is_limit_up": 0,
                "is_limit_down": 0,
                "filter_is_st": 0,
                "is_st": 0,
                "filter_list_days": 100,
                "list_days": 100,
                "tradable": 1,
            },
            {
                "ts_code": "000002.SZ",
                "trade_date": "20230103",
                "close": 10.0,
                "close_adj": 10.0,
                "open": 10.0,
                "open_adj": 10.0,
                "vol": 1000000,
                "pct_chg": 9.99,
                "filter_is_suspended": 0,
                "is_suspended": 0,
                "is_limit_up": 1,
                "is_limit_down": 0,
                "filter_is_st": 0,
                "is_st": 0,
                "filter_list_days": 100,
                "list_days": 100,
                "tradable": 1,
            },
            {
                "ts_code": "000003.SZ",
                "trade_date": "20230103",
                "close": 10.0,
                "close_adj": 10.0,
                "open": 10.0,
                "open_adj": 10.0,
                "vol": 1000000,
                "pct_chg": 9.99,
                "filter_is_suspended": 0,
                "is_suspended": 0,
                "is_limit_up": 1,
                "is_limit_down": 0,
                "filter_is_st": 0,
                "is_st": 0,
                "filter_list_days": 100,
                "list_days": 100,
                "tradable": 1,
            },
            {
                "ts_code": "600001.SH",
                "trade_date": "20230103",
                "close": 20.0,
                "close_adj": 20.0,
                "open": 20.0,
                "open_adj": 20.0,
                "vol": 1000000,
                "pct_chg": 0.0,
                "filter_is_suspended": 0,
                "is_suspended": 0,
                "is_limit_up": 0,
                "is_limit_down": 0,
                "filter_is_st": 0,
                "is_st": 0,
                "filter_list_days": 100,
                "list_days": 100,
                "tradable": 1,
            },
        ]
    )
    engine._prepare_price_index(price_data)
    engine.price_data_cache = price_data

    engine.pending_signals[signal_date] = {
        "signals": {
            "000001.SZ": 0.25,
            "000002.SZ": 0.25,
        },
        "ranked_candidates": [
            ("000001.SZ", 1.0),
            ("000002.SZ", 1.0),
        ],
        "target_n": 3,
        "tranche_idx": 0,
        "decision_trace": {
            "signal_date": signal_date,
            "candidate_count": 3,
            "target_n": 3,
            "holding_bonus": {
                "enabled": True,
                "kept_count": 1,
                "kept_stocks": ["600001.SH"],
            },
        },
    }

    warning_messages = []
    monkeypatch.setattr(
        engine_module.logger,
        "warning",
        lambda message: warning_messages.append(str(message)),
    )
    monkeypatch.setattr(engine_module.logger, "info", lambda message: None)

    engine._execute_pending_buys(
        date=buy_date,
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    summary = next(message for message in warning_messages if "调仓买入汇总:" in message)
    lines = summary.splitlines()

    assert summary.endswith("\n")
    assert len(lines) == 3
    assert (
        "调仓买入汇总: 执行日 2023-01-03 | 信号日 2023-01-02 | 计划=2 | 计划资金占比=50.00% | "
        "继承上轮=1 | 继承资金占比=25.00% | 成功=1 | 失败=1"
    ) == lines[0]
    assert "成功仓位: 数量=1 | 股票=[000001.SZ] | 资金占比=25.00%" == lines[1]
    assert "失败仓位: 数量=1 | 股票=[000002.SZ(涨停)] | 资金占比=25.00%" == lines[2]
