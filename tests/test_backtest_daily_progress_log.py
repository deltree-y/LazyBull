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
    """每日回测日志应展示仓位、买卖数量和收益摘要。"""
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
        buy_count=3,
        sell_count=1,
    )

    assert log_line.startswith("T1[2025-12-29]: 122/124 天 | 本轮[02/20]")
    assert "持仓/仓位[17/20]/[87.05%]" in log_line
    assert "买/卖[3/1]" in log_line
    # 年化收益改为简单年化公式: 15.84% / (122/252) = 32.72%
    assert "收益[本调仓/本轮/年化]=[+15.56%/+15.84%/+32.72%]" in log_line
    assert "ATR:" not in log_line


def test_ml_daily_progress_log_no_longer_shows_atr_summary():
    """ML 回测日志也不再展示 ATR 摘要。"""
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

    assert "买/卖[0/0]" in log_line
    assert "ATR:" not in log_line


def test_daily_trade_log_buy_and_sell_items_are_not_compacted():
    """交易买卖明细应完整输出，不应折叠为 ...+N。"""
    trade_date = pd.Timestamp("2025-01-02")
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=20),
        initial_capital=100000.0,
        rebalance_freq=20,
        verbose=False,
    )

    engine.trades = [
        {
            "date": trade_date,
            "stock": f"00000{i}.SZ",
            "action": "buy",
            "amount": 10000.0 + i,
            "cost": 0.0,
        }
        for i in range(6)
    ]
    engine.trades.extend(
        [
            {
                "date": trade_date,
                "stock": f"00010{i}.SZ",
                "action": "sell",
                "buy_date": pd.Timestamp("2025-01-01"),
                "pnl_profit_pct": 0.01 * i,
            }
            for i in range(6)
        ]
    )

    buy_count, sell_count, lines = engine._build_daily_trade_log(
        date=trade_date,
        trade_start_idx=0,
        date_to_idx={pd.Timestamp("2025-01-01"): 0, trade_date: 1},
    )

    assert buy_count == 6
    assert sell_count == 6
    assert len(lines) == 2
    assert lines[0] == (
        "交易: 卖6[000100.SZ(1d,+0.0%), 000101.SZ(1d,+1.0%), 000102.SZ(1d,+2.0%), "
        "000103.SZ(1d,+3.0%), 000104.SZ(1d,+4.0%), 000105.SZ(1d,+5.0%)]"
    )
    assert lines[1] == (
        "交易: 买6[000000.SZ(1.0w), 000001.SZ(1.0w), 000002.SZ(1.0w), "
        "000003.SZ(1.0w), 000004.SZ(1.0w), 000005.SZ(1.0w)]"
    )


def test_daily_signal_log_summarizes_new_buy_and_sell_signals():
    """当日新生成的买卖信号应压缩为一条简洁计数日志。"""
    trade_date = pd.Timestamp("2025-01-02")
    next_date = pd.Timestamp("2025-01-03")
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=20),
        initial_capital=100000.0,
        rebalance_freq=20,
        verbose=False,
    )

    engine.pending_signals[trade_date] = {
        "signals": {"000001.SZ": 0.1, "000002.SZ": 0.1},
        "decision_trace": {"queued": True},
    }
    engine.pending_condition_sells["000003.SZ"] = {
        "trigger_date": trade_date,
        "sell_type": "holding_period",
    }
    engine.pending_stop_loss_sells["000005.SZ"] = {
        "trigger_date": trade_date,
        "trigger_type": "drawdown",
    }
    engine.pending_stop_loss_sells["000006.SZ"] = {
        "trigger_date": next_date,
        "trigger_type": "drawdown",
    }

    assert (
        engine._build_daily_signal_log(trade_date)
        == "信号: 卖[持有期1, 回撤止损1] | 买[调仓2]"
    )


def test_target_position_count_is_split_across_stagger_tranches():
    """分批调仓应拆分总目标持仓数，而不是按批次数放大。"""
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=5),
        initial_capital=100000.0,
        rebalance_freq=5,
        stagger_tranches=3,
        verbose=False,
    )

    assert engine._get_target_position_count() == 5
    assert [engine._get_tranche_target_count(index) for index in range(3)] == [2, 2, 1]
    assert [engine._get_tranche_capital_fraction(index) for index in range(3)] == [
        0.4,
        0.4,
        0.2,
    ]


def test_stagger_signal_plan_keeps_total_target_and_splits_slots(monkeypatch):
    """各批信号只生成本批槽位，同时保留总 TopN 作为组合目标。"""
    trading_dates = [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    price_data = pd.DataFrame(
        {
            "trade_date": ["20250102", "20250103"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 10.0],
        }
    )

    for tranche_idx, expected_count in enumerate([2, 2, 1]):
        engine = BacktestEngine(
            universe=MockUniverse(),
            signal=MockSignal(top_n=5),
            initial_capital=100000.0,
            rebalance_freq=20,
            stagger_tranches=3,
            verbose=False,
        )
        monkeypatch.setattr(
            engine.signal,
            "generate_ranked",
            lambda *_args: [(f"00000{index}.SZ", 10.0 - index) for index in range(1, 10)],
        )

        engine._generate_signal(
            date=trading_dates[0],
            trading_dates=trading_dates,
            price_data=price_data,
            date_to_idx={date: index for index, date in enumerate(trading_dates)},
            tranche_idx=tranche_idx,
        )

        signal_plan = engine.pending_signals[trading_dates[0]]
        assert signal_plan["target_n"] == expected_count
        assert signal_plan["desired_position_count"] == 5
        assert len(signal_plan["slot_weights"]) == expected_count


def test_stagger_tranches_build_full_position_across_batches(monkeypatch):
    """后续批次应继续占用剩余槽位，完成后达到总 TopN 满仓。"""
    trading_dates = list(pd.date_range("2025-01-02", periods=4, freq="B"))
    stocks = [f"00000{index}.SZ" for index in range(1, 9)]
    price_data = pd.DataFrame(
        [
            {
                "trade_date": date.strftime("%Y%m%d"),
                "ts_code": stock,
                "close": 10.0,
            }
            for date in trading_dates
            for stock in stocks
        ]
    )
    signal = MockSignal(top_n=4)
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(
            commission_rate=0.0,
            min_commission=0.0,
            stamp_tax=0.0,
            slippage=0.0,
        ),
        rebalance_freq=4,
        stagger_tranches=2,
        enable_pending_order=False,
        enable_position_completion=False,
        enable_early_rebalance_on_empty=False,
        verbose=False,
    )
    monkeypatch.setattr(engine.universe, "get_stocks", lambda *_args, **_kwargs: stocks)
    monkeypatch.setattr(
        signal,
        "generate_ranked",
        lambda *_args: [(stock, 10.0 - index) for index, stock in enumerate(stocks)],
    )

    engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=trading_dates,
        price_data=price_data,
    )

    assert len(engine.positions) == 4
    assert len(engine.get_trades()) == 4
    assert engine.current_capital == 0.0


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

    monkeypatch.setattr(
        engine,
        "_emit_immediate_log",
        lambda level, message, colors=False: log_messages.append(str(message)),
    )
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

    monkeypatch.setattr(
        engine,
        "_emit_immediate_log",
        lambda level, message, colors=False: log_messages.append(str(message)),
    )
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


def test_run_emits_daily_summary_before_trade_signal_and_detail_logs(monkeypatch):
    """单日回测应先输出总结，再输出调仓摘要、交易两行、信号统计，最后回放缓冲细节。"""

    trade_date = pd.Timestamp("2025-01-02")
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=5),
        initial_capital=100000.0,
        rebalance_freq=5,
        completion_window_days=5,
        verbose=False,
        enable_pending_order=False,
        enable_position_completion=False,
        enable_early_rebalance_on_empty=False,
    )

    emitted_messages = []
    monkeypatch.setattr(
        engine,
        "_emit_immediate_log",
        lambda level, message, colors=False: emitted_messages.append((level, str(message), colors)),
    )
    monkeypatch.setattr(engine, "_prepare_price_index", lambda price_data: None)
    monkeypatch.setattr(engine, "_get_rebalance_dates", lambda trading_dates: {})
    monkeypatch.setattr(engine, "_generate_nav_curve", lambda: pd.DataFrame())
    monkeypatch.setattr(engine, "_calculate_portfolio_value", lambda date: 100000.0)

    def fake_check_and_sell(date, trading_dates, date_to_idx):
        engine_module.logger.info("调仓决策摘要: 信号日 2025-01-02 | 执行=2025-01-03 | 候选=5 | 目标=2")
        engine.pending_signals[date] = {
            "signals": {"000007.SZ": 0.1, "000008.SZ": 0.1},
            "decision_trace": {"queued": True},
        }
        engine.pending_condition_sells["000009.SZ"] = {
            "trigger_date": date,
            "sell_type": "holding_period",
        }
        engine.pending_stop_loss_sells["000010.SZ"] = {
            "trigger_date": date,
            "trigger_type": "drawdown",
        }
        engine.trades.append(
            {
                "date": date,
                "stock": "000002.SZ",
                "action": "sell",
                "buy_date": pd.Timestamp("2024-12-30"),
                "pnl_profit_pct": -0.021,
            }
        )
        engine_module.logger.info("调试细节日志")

    def fake_execute_pending_buys(date, trading_dates, date_to_idx):
        engine.trades.append(
            {
                "date": date,
                "stock": "000001.SZ",
                "action": "buy",
                "amount": 103000.0,
                "cost": 0.0,
            }
        )

    monkeypatch.setattr(engine, "_check_and_sell", fake_check_and_sell)
    monkeypatch.setattr(engine, "_execute_pending_buys", fake_execute_pending_buys)

    engine.run(
        start_date=trade_date,
        end_date=trade_date,
        trading_dates=[trade_date],
        price_data=pd.DataFrame(columns=["ts_code", "trade_date", "close"]),
    )

    summary_index = next(
        i for i, (_, message, _) in enumerate(emitted_messages) if "T0[2025-01-02]" in message
    )
    decision_index = next(
        i
        for i, (_, message, _) in enumerate(emitted_messages)
        if "调仓决策摘要: 信号日 2025-01-02 | 执行=2025-01-03 | 候选=5 | 目标=2" in message
    )
    signal_index = next(
        i
        for i, (_, message, _) in enumerate(emitted_messages)
        if "信号: 卖[持有期1, 回撤止损1] | 买[调仓2]" in message
    )
    buy_index = next(
        i for i, (_, message, _) in enumerate(emitted_messages) if "交易: 买1[000001.SZ(10.3w)]" in message
    )
    sell_index = next(
        i for i, (_, message, _) in enumerate(emitted_messages) if "交易: 卖1[000002.SZ(0d,-2.1%)]" in message
    )
    detail_index = next(
        i for i, (_, message, _) in enumerate(emitted_messages) if "调试细节日志" in message
    )

    assert summary_index < decision_index < sell_index < buy_index < signal_index < detail_index
    assert emitted_messages[detail_index][1].startswith("  调试细节日志")


def test_run_emits_compact_daily_warning_summaries_as_plain_lines(monkeypatch):
    """压缩类事件应按日级白色单行展示。"""

    trade_date = pd.Timestamp("2025-01-02")
    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(top_n=5),
        initial_capital=100000.0,
        rebalance_freq=5,
        completion_window_days=5,
        verbose=False,
        enable_pending_order=False,
        enable_position_completion=False,
        enable_early_rebalance_on_empty=False,
    )

    emitted_messages = []
    monkeypatch.setattr(
        engine,
        "_emit_immediate_log",
        lambda level, message, colors=False: emitted_messages.append((level, str(message), colors)),
    )
    monkeypatch.setattr(engine, "_prepare_price_index", lambda price_data: None)
    monkeypatch.setattr(engine, "_get_rebalance_dates", lambda trading_dates: {})
    monkeypatch.setattr(engine, "_generate_nav_curve", lambda: pd.DataFrame())
    monkeypatch.setattr(engine, "_calculate_portfolio_value", lambda date: 100000.0)

    def fake_check_and_sell(date, trading_dates, date_to_idx):
        engine._record_pending_order_event(
            {"type": "added", "stock": "002701.SZ", "action": "sell", "reason": "跌停"}
        )
        engine._record_pending_order_event(
            {"type": "added", "stock": "000938.SZ", "action": "sell", "reason": "跌停"}
        )
        engine._record_pending_order_event(
            {"type": "success", "stock": "002701.SZ", "action": "sell", "retry_count": 1, "delay_days": 1}
        )
        engine._record_pending_order_event(
            {"type": "success", "stock": "000938.SZ", "action": "sell", "retry_count": 1, "delay_days": 1}
        )
        engine._record_pending_order_event(
            {
                "type": "expired_retry",
                "stock": "300001.SZ",
                "action": "buy",
                "retry_count": 4,
                "max_retry_count": 3,
            }
        )
        engine._record_pending_order_event(
            {
                "type": "expired_days",
                "stock": "300002.SZ",
                "action": "buy",
                "delay_days": 6,
                "max_retry_days": 5,
            }
        )
        engine_module.logger.info("调试细节日志")

    def fake_execute_pending_buys(date, trading_dates, date_to_idx):
        engine._record_early_rebalance_summary("空仓触发", "无持仓, 新信号入队")
        engine._record_duplicate_buy_skip("000506.SZ", pd.Timestamp("2025-08-28"))
        engine._record_position_unfilled_summary(
            tranche_tag="",
            target_n=20,
            actually_bought=3,
            unfilled_count=17,
            unfilled_stocks=[
                "002458.SZ",
                "600926.SH",
                "002670.SZ",
                "601231.SH",
                "002440.SZ",
                "002746.SZ",
                "601198.SH",
            ],
        )
        engine._record_completion_abandoned_summary(
            tranche_tag="",
            original_signal_date=pd.Timestamp("2021-01-18"),
            attempts=4,
            unfilled_stocks=["603444.SH", "601186.SH"],
        )
        engine._record_completion_skip("当日无行情", "当日无行情")
        engine._record_completion_skip("前日无行情", "信号日2021-01-18")
        engine._record_completion_skip("无数据", "批2 信号日2021-01-19")
        engine._record_completion_skip("无候选", "信号日2021-01-20")
        engine._record_completion_skip("候选已持仓", "批2 信号日2021-01-21")
        engine._record_completion_skip("候选不可交易", "603444.SH(涨停)")
        engine._record_completion_skip("候选不可交易", "601186.SH(停牌)")

    monkeypatch.setattr(engine, "_check_and_sell", fake_check_and_sell)
    monkeypatch.setattr(engine, "_execute_pending_buys", fake_execute_pending_buys)

    engine.run(
        start_date=trade_date,
        end_date=trade_date,
        trading_dates=[trade_date],
        price_data=pd.DataFrame(columns=["ts_code", "trade_date", "close"]),
    )

    early_rebalance = next(item for item in emitted_messages if "提前调仓:" in item[1])
    duplicate_buy = next(item for item in emitted_messages if "重复买入跳过:" in item[1])
    position_unfilled = next(item for item in emitted_messages if "仓位未满:" in item[1])
    completion_skipped = next(item for item in emitted_messages if "补齐跳过:" in item[1])
    completion_abandoned = next(item for item in emitted_messages if "补齐放弃:" in item[1])
    pending_order_added = next(item for item in emitted_messages if "延迟订单: " in item[1])
    pending_order_success = next(item for item in emitted_messages if "延迟订单成交: " in item[1])
    pending_order_expired = next(item for item in emitted_messages if "延迟订单放弃: " in item[1])
    detail_index = next(i for i, item in enumerate(emitted_messages) if "调试细节日志" in item[1])
    early_index = emitted_messages.index(early_rebalance)
    duplicate_buy_index = emitted_messages.index(duplicate_buy)
    position_unfilled_index = emitted_messages.index(position_unfilled)
    completion_skipped_index = emitted_messages.index(completion_skipped)
    completion_abandoned_index = emitted_messages.index(completion_abandoned)
    pending_order_added_index = emitted_messages.index(pending_order_added)
    pending_order_success_index = emitted_messages.index(pending_order_success)
    pending_order_expired_index = emitted_messages.index(pending_order_expired)

    assert early_rebalance == ("INFO", "  提前调仓: 空仓触发[无持仓, 新信号入队]", False)
    assert duplicate_buy == (
        "INFO",
        "  重复买入跳过: 1只[000506.SZ(2025-08-28)]",
        False,
    )
    assert position_unfilled == (
        "INFO",
        "  仓位未满: 目标20/实买3/待补17[002458.SZ, 600926.SH, 002670.SZ, 601231.SH, 002440.SZ, 002746.SZ, ...+1]/5天",
        False,
    )
    assert completion_skipped == (
        "INFO",
        "  补齐跳过: 当日无行情1 | 前日无行情1[信号日2021-01-18] | 无数据1[批2 信号日2021-01-19] | 无候选1[信号日2021-01-20] | 候选已持仓1[批2 信号日2021-01-21] | 候选不可交易2[603444.SH(涨停), 601186.SH(停牌)]",
        False,
    )
    assert completion_abandoned == (
        "INFO",
        "  补齐放弃: 信号日2021-01-18/尝试4次/剩2[603444.SH, 601186.SH]",
        False,
    )
    assert pending_order_added == (
        "INFO",
        "  延迟订单: 新增卖2[002701.SZ(跌停), 000938.SZ(跌停)]",
        False,
    )
    assert pending_order_success == (
        "INFO",
        "  延迟订单成交: 成功卖2[002701.SZ(重1,延1d), 000938.SZ(重1,延1d)]",
        False,
    )
    assert pending_order_expired == (
        "INFO",
        "  延迟订单放弃: 超次买1[300001.SZ(重4>3)] | 超期买1[300002.SZ(延6d>5d)]",
        False,
    )
    assert early_index < detail_index
    assert duplicate_buy_index < detail_index
    assert position_unfilled_index < detail_index
    assert completion_skipped_index < detail_index
    assert completion_abandoned_index < detail_index
    assert pending_order_added_index < detail_index
    assert pending_order_success_index < detail_index
    assert pending_order_expired_index < detail_index
