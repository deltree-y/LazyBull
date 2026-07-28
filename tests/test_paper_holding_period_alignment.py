"""纸面交易持有期到期与回测口径对齐测试"""

import tempfile
from unittest.mock import MagicMock, patch

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
        }

        actions = _plan_next_day_retry_and_sell_instructions(
            runner,
            "20240104",
            config,
            stop_loss_actions=[],
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


def _mock_trade_cal():
    """构建 4 个连续交易日的 mock 交易日历。"""
    import pandas as pd

    return pd.DataFrame(
        [
            {"cal_date": "20240102", "is_open": 1},
            {"cal_date": "20240103", "is_open": 1},
            {"cal_date": "20240104", "is_open": 1},
            {"cal_date": "20240105", "is_open": 1},
        ]
    )


def test_evaluate_holding_period_actions_sells_expired_position():
    """持有期到期的持仓应生成整手卖出动作。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _build_runner(tmpdir)

        runner.account.state.positions = {
            "000001.SZ": Position(
                ts_code="000001.SZ",
                shares=150,
                buy_price=10.0,
                buy_cost=1.0,
                buy_date="20240102",
            )
        }
        runner.loader.load_clean_trade_cal = MagicMock(return_value=_mock_trade_cal())

        # rebalance_freq=2 -> 阈值 max(1, 2-1)=1，20240102 -> 20240104 持有 2 天已到期
        protected, sell_actions = runner.evaluate_holding_period_actions(
            "20240104", {"rebalance_freq": 2}
        )

        assert protected == set()
        assert len(sell_actions) == 1
        assert sell_actions[0]["ts_code"] == "000001.SZ"
        assert sell_actions[0]["shares"] == 100  # 整手取整
        assert "持有期到期" in sell_actions[0]["reason"]


def test_evaluate_holding_period_actions_keeps_young_and_excluded_positions():
    """未满持有期或在排除集内的持仓不应生成卖出动作。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = _build_runner(tmpdir)

        runner.account.state.positions = {
            "000001.SZ": Position(
                ts_code="000001.SZ",
                shares=100,
                buy_price=10.0,
                buy_cost=1.0,
                buy_date="20240104",  # 当日买入，未满持有期
            ),
            "000002.SZ": Position(
                ts_code="000002.SZ",
                shares=100,
                buy_price=10.0,
                buy_cost=1.0,
                buy_date="20240102",  # 已到期但在排除集内
            ),
        }
        runner.loader.load_clean_trade_cal = MagicMock(return_value=_mock_trade_cal())

        protected, sell_actions = runner.evaluate_holding_period_actions(
            "20240104",
            {"rebalance_freq": 2},
            exclude_stocks={"000002.SZ"},
        )

        assert protected == set()
        assert sell_actions == []
