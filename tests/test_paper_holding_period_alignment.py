"""纸面交易持有期到期/盈利延续与回测口径对齐测试"""

import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd

from src.lazybull.backtest import BacktestEngine
from src.lazybull.common.cost import CostModel
from src.lazybull.signals.base import Signal
from src.lazybull.universe.base import Universe
from src.lazybull.paper.models import Position
from src.lazybull.paper.runner import PaperTradingRunner
from src.lazybull.paper.runtime import _execute_t1_if_pending


def _build_runner(tmpdir: str) -> PaperTradingRunner:
    """构建最小可测的 PaperTradingRunner。"""
    with patch("src.lazybull.paper.runner.TushareClient"):
        runner = PaperTradingRunner(
            initial_capital=1_000_000.0,
            data_root=tmpdir,
            paper_root=tmpdir,
            verbose=False,
        )
    return runner


class _MockUniverse(Universe):
    def get_stocks(self, date, quote_data=None):
        return ["000001.SZ"]


class _MockSignal(Signal):
    def generate(self, date, universe, data):
        return {"000001.SZ": 1.0}


def test_daily_holding_extension_strength_can_trigger_on_non_rebalance_day():
    """非调仓日也应可触发强势度盈利延续。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _build_runner(tmpdir)

        runner.account.state.positions = {
            "000001.SZ": Position(
                ts_code="000001.SZ",
                shares=100,
                buy_price=10.0,
                buy_cost=1.0,
                buy_date="20240102",
            )
        }

        runner.loader.load_clean_daily_by_date = MagicMock(
            return_value=pd.DataFrame(
                [{"ts_code": "000001.SZ", "close": 11.2}]
            )
        )
        runner.loader.load_clean_trade_cal = MagicMock(
            return_value=pd.DataFrame(
                [
                    {"cal_date": "20240102", "is_open": 1},
                    {"cal_date": "20240103", "is_open": 1},
                    {"cal_date": "20240104", "is_open": 1},
                    {"cal_date": "20240105", "is_open": 1},
                ]
            )
        )
        runner._score_holding_strength = MagicMock(
            return_value=type(
                "Breakdown",
                (),
                {"total": 0.80, "to_log_str": lambda self: "mock-breakdown"},
            )()
        )

        config = {
            "enable_profit_based_holding": True,
            "profit_extension_mode": "strength",
            "profit_extension_strength_threshold": 0.56,
            "rebalance_freq": 2,
            "profit_extension_days": 3,
        }

        protected, sell_actions = runner.evaluate_holding_period_actions("20240104", config)

        assert protected == {"000001.SZ"}
        assert sell_actions == []


def test_t1_should_execute_holding_period_sell_even_without_rebalance_instructions():
    """当日无调仓指令时，T1 也应执行持有期到期卖出。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _build_runner(tmpdir)

        runner.evaluate_holding_period_actions = MagicMock(
            return_value=(
                set(),
                [
                    {
                        "ts_code": "000001.SZ",
                        "shares": 100,
                        "reason": "持有期到期不延续[strength]",
                        "can_execute": True,
                    }
                ],
            )
        )

        runner.paper_storage.find_pending_instructions = MagicMock(return_value=None)
        runner.paper_storage.load_pending_buys = MagicMock(return_value=[])
        runner._load_prices = MagicMock(
            return_value=(
                {"000001.SZ": 10.0},
                {"000001.SZ": 10.0},
            )
        )

        captured = {"instructions": []}

        def _fake_execute(instructions, buy_prices, sell_prices, trade_date):
            captured["instructions"] = instructions
            return []

        runner.broker.execute_instructions = _fake_execute
        runner.broker.get_failed_buy_targets = MagicMock(return_value=[])

        config = {
            "buy_price": "close",
            "sell_price": "open",
            "rebalance_freq": 20,
            "enable_profit_based_holding": True,
        }

        _execute_t1_if_pending(runner, "20240104", config)

        assert len(captured["instructions"]) == 1
        assert captured["instructions"][0].action == "sell"
        assert captured["instructions"][0].ts_code == "000001.SZ"


def test_t0_generate_instructions_should_not_create_sell_from_target_reduction():
    """T0 指令生成不再按目标权重直接生成卖出。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _build_runner(tmpdir)

        runner.account.state.positions = {
            "000001.SZ": Position(
                ts_code="000001.SZ",
                shares=1000,
                buy_price=10.0,
                buy_cost=1.0,
                buy_date="20240102",
            )
        }
        runner.account.state.cash = 0.0

        instructions = runner._generate_instructions(
            targets=[
                type("Target", (), {"ts_code": "000001.SZ", "target_weight": 0.05, "reason": "降权"})(),
            ],
            buy_price_type="close",
            sell_price_type="open",
            current_prices={"000001.SZ": 10.0},
            source_date="20240104",
        )

        assert instructions == []


def test_profit_rate_should_use_adjusted_price_for_extension_decision():
    """盈利延续判定应使用后复权绩效价而非成交价。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _build_runner(tmpdir)

        runner.account.state.positions = {
            "000001.SZ": Position(
                ts_code="000001.SZ",
                shares=100,
                buy_price=10.0,
                buy_cost=1.0,
                buy_date="20240102",
                buy_pnl_price=100.0,
            )
        }

        runner.loader.load_clean_daily_by_date = MagicMock(
            side_effect=[
                pd.DataFrame([{"ts_code": "000001.SZ", "close": 10.0, "close_adj": 120.0}]),
            ]
        )
        runner.loader.load_clean_trade_cal = MagicMock(
            return_value=pd.DataFrame(
                [
                    {"cal_date": "20240102", "is_open": 1},
                    {"cal_date": "20240103", "is_open": 1},
                    {"cal_date": "20240104", "is_open": 1},
                ]
            )
        )

        config = {
            "enable_profit_based_holding": True,
            "profit_extension_mode": "pnl",
            "profit_extension_threshold": 0.1,
            "rebalance_freq": 2,
            "profit_extension_days": 3,
            "buy_price": "close",
        }

        protected, sell_actions = runner.evaluate_holding_period_actions("20240104", config)
        assert protected == {"000001.SZ"}
        assert sell_actions == []


def test_holding_bonus_kept_position_should_reset_holding_anchor_to_t1():
    """持仓保留奖励命中时，应将持有期锚点重置到 T+1。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _build_runner(tmpdir)

        runner.account.state.positions = {
            "000001.SZ": Position(
                ts_code="000001.SZ",
                shares=100,
                buy_price=10.0,
                buy_cost=1.0,
                buy_date="20240102",
            )
        }
        runner.account.save_state = MagicMock()
        runner._get_next_trade_date = MagicMock(return_value="20240105")

        runner._reset_holding_anchor_for_kept_positions(
            trade_date="20240104",
            kept_stocks=["000001.SZ"],
        )

        assert runner.account.state.positions["000001.SZ"].buy_date == "20240105"
        runner.account.save_state.assert_called_once()


def test_end_to_end_decision_alignment_with_backtest_on_same_window():
    """同区间下，回测与纸面在盈利延续决策上应保持一致（后复权口径）。"""
    # 回测侧
    trading_dates = [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    price_data = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": d.strftime("%Y%m%d"),
                "close": 10.0,
                "open": 10.0,
                "close_adj": 120.0 if d >= pd.Timestamp("2024-01-04") else 100.0,
                "open_adj": 120.0 if d >= pd.Timestamp("2024-01-04") else 100.0,
            }
            for d in trading_dates
        ]
    )
    bt_engine = BacktestEngine(
        universe=_MockUniverse(),
        signal=_MockSignal(),
        initial_capital=100000.0,
        cost_model=CostModel(),
        rebalance_freq=2,
        holding_period=2,
        enable_profit_based_holding=True,
        profit_extension_mode="pnl",
        profit_extension_threshold=0.1,
        profit_extension_days=2,
        verbose=False,
    )
    bt_engine._prepare_price_index(price_data)
    bt_engine.price_data_cache = price_data
    bt_engine.positions = {
        "000001.SZ": {
            "shares": 100,
            "buy_date": pd.Timestamp("2024-01-02"),
            "signal_date": pd.Timestamp("2024-01-02"),
            "buy_trade_price": 10.0,
            "buy_pnl_price": 100.0,
            "buy_cost_cash": 1.0,
        }
    }
    bt_engine._check_and_sell(pd.Timestamp("2024-01-04"), trading_dates, date_to_idx)
    assert "000001.SZ" in bt_engine.positions

    # 纸面侧
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _build_runner(tmpdir)
        runner.account.state.positions = {
            "000001.SZ": Position(
                ts_code="000001.SZ",
                shares=100,
                buy_price=10.0,
                buy_cost=1.0,
                buy_date="20240102",
                buy_pnl_price=100.0,
            )
        }
        runner.loader.load_clean_daily_by_date = MagicMock(
            return_value=pd.DataFrame(
                [{"ts_code": "000001.SZ", "close": 10.0, "close_adj": 120.0}]
            )
        )
        runner.loader.load_clean_trade_cal = MagicMock(
            return_value=pd.DataFrame(
                [
                    {"cal_date": "20240102", "is_open": 1},
                    {"cal_date": "20240103", "is_open": 1},
                    {"cal_date": "20240104", "is_open": 1},
                    {"cal_date": "20240105", "is_open": 1},
                ]
            )
        )
        cfg = {
            "enable_profit_based_holding": True,
            "profit_extension_mode": "pnl",
            "profit_extension_threshold": 0.1,
            "rebalance_freq": 2,
            "profit_extension_days": 2,
            "buy_price": "close",
        }
        protected, sell_actions = runner.evaluate_holding_period_actions("20240104", cfg)
        assert protected == {"000001.SZ"}
        assert sell_actions == []
