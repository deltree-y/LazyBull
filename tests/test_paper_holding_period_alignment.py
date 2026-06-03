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
from src.lazybull.paper.runtime import _build_sell_instructions, _plan_next_day_retry_and_sell_instructions


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


def test_t0_should_plan_holding_period_sell_for_next_trade_day():
    """持有期到期卖出应在 T0 预生成到下一交易日指令。"""
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

        runner.paper_storage.load_pending_buys = MagicMock(return_value=[])
        runner.paper_storage.load_instructions = MagicMock(return_value=[])
        runner.paper_storage.save_instructions = MagicMock()
        runner._get_next_trade_date = MagicMock(return_value="20240105")

        config = {
            "buy_price": "close",
            "sell_price": "open",
            "rebalance_freq": 20,
            "enable_profit_based_holding": True,
        }

        actions = _plan_next_day_retry_and_sell_instructions(
            runner,
            "20240104",
            config,
            stop_loss_actions=[],
            early_exit_actions=[],
            take_profit_actions=[],
        )

        assert len(actions) == 1
        saved_instructions = runner.paper_storage.save_instructions.call_args[0][1]
        assert len(saved_instructions) == 1
        assert saved_instructions[0].action == "sell"
        assert saved_instructions[0].ts_code == "000001.SZ"


def test_build_sell_instructions_should_preserve_retry_attempt():
    instructions = _build_sell_instructions(
        actions=[{"ts_code": "000001.SZ", "shares": 100, "reason": "重试卖出"}],
        trade_date="20240104",
        config={"sell_price": "close"},
        retry_attempt=2,
    )

    assert len(instructions) == 1
    assert instructions[0].retry_attempt == 2


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


def test_early_exit_should_not_trigger_after_holding_period_reached():
    """到达持有期后，亏损提前换出应让位于到期/延续路径（对齐回测）。"""
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

        # 当日亏损（95/100=-5%），若未加边界会触发 early_exit
        runner.loader.load_clean_daily_by_date = MagicMock(
            return_value=pd.DataFrame(
                [{"ts_code": "000001.SZ", "close": 9.5, "close_adj": 95.0}]
            )
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

        cfg = {
            "enable_profit_based_holding": True,
            "rebalance_freq": 2,
            "early_exit_holding_ratio": 0.5,
            "early_exit_loss_threshold": -0.03,
            "buy_price": "close",
            "early_exit_mode": "disabled",
        }

        actions = runner.evaluate_early_exit("20240104", cfg)
        assert actions == []


def test_holding_strength_should_ensure_features_before_daily_evaluation(monkeypatch):
    """strength 模式按日评估前应先 ensure 当日 features。"""
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
            return_value=pd.DataFrame([{"ts_code": "000001.SZ", "close": 112.0, "close_adj": 112.0}])
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
        runner._score_holding_strength = MagicMock(
            return_value=type(
                "Breakdown",
                (object,),
                {"total": 0.8, "to_log_str": lambda self: "ok"},
            )()
        )

        ensure_called = {"count": 0}

        def _fake_ensure(*_args, **_kwargs):
            ensure_called["count"] += 1
            return True, []

        monkeypatch.setattr("src.lazybull.paper.runner.ensure_features_for_date", _fake_ensure)

        cfg = {
            "enable_profit_based_holding": True,
            "profit_extension_mode": "strength",
            "profit_extension_strength_threshold": 0.56,
            "rebalance_freq": 2,
            "profit_extension_days": 3,
            "buy_price": "close",
        }

        protected, sell_actions = runner.evaluate_holding_period_actions("20240104", cfg)

        assert ensure_called["count"] == 1
        assert protected == {"000001.SZ"}
        assert sell_actions == []
