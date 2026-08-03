"""测试停牌股票处理逻辑

测试纸面交易和回测中停牌股票的处理，包括：
1. 停牌股票在止损检查中不会触发止损
2. 需要卖出但停牌/无价格时会加入延迟卖出队列
3. 回测中停牌股票不会触发止损且卖出会延迟
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.lazybull.common.cost import CostModel
from src.lazybull.paper import (
    PaperAccount,
    PaperBroker,
    PaperStorage,
    Position,
    TargetWeight,
)
from src.lazybull.paper.models import PendingSell
from src.lazybull.risk.stop_loss import StopLossConfig, StopLossMonitor


@pytest.fixture
def temp_storage():
    """临时存储目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        yield storage


@pytest.fixture
def sample_account_with_position(temp_storage):
    """带有持仓的示例账户"""
    account = PaperAccount(initial_capital=100000.0, storage=temp_storage)

    # 添加一个持仓
    account.state.positions["000001.SZ"] = Position(
        ts_code="000001.SZ", shares=1000, buy_price=10.0, buy_cost=10.0, buy_date="20260101"
    )
    account.state.cash = 90000.0  # 已用资金10000

    return account


class TestPaperTradingSuspendedHandling:
    """纸面交易停牌处理测试"""

    def test_suspended_stock_skipped_in_stop_loss_check(self, temp_storage):
        """测试：停牌股票在止损检查中被跳过"""
        from scripts.paper_trade import PaperTradingRunner
        from src.lazybull.paper.runtime import _check_stop_loss

        # 创建账户和持仓
        account = PaperAccount(initial_capital=100000.0, storage=temp_storage)
        account.state.positions["000001.SZ"] = Position(
            ts_code="000001.SZ", shares=1000, buy_price=10.0, buy_cost=10.0, buy_date="20260101"
        )

        # 创建 mock runner
        runner = MagicMock(spec=PaperTradingRunner)
        runner.account = account
        runner.storage = temp_storage

        # 创建止损监控器
        stop_loss_config = StopLossConfig(
            enabled=True, drawdown_pct=10.0, consecutive_limit_down_days=2
        )
        stop_loss_monitor = StopLossMonitor(stop_loss_config)

        # Mock DataLoader 返回停牌股票数据
        mock_daily_data = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260120"],
                "close": [0.0],  # 停牌时价格可能为0
                "is_limit_down": [0],
                "is_suspended": [1],  # 停牌标志
            }
        )

        with patch("src.lazybull.paper.runtime.DataLoader") as MockDataLoader:
            mock_loader = MagicMock()
            mock_loader.load_clean_daily_by_date.return_value = mock_daily_data
            MockDataLoader.return_value = mock_loader

            # 执行止损检查
            actions = _check_stop_loss(
                runner=runner, stop_loss_monitor=stop_loss_monitor, trade_date="20260120", config={}
            )

            # 验证：停牌股票不会触发止损
            assert len(actions) == 0, "停牌股票不应触发止损"


    def test_sell_no_price_data_added_to_pending_sells(
        self, sample_account_with_position, temp_storage
    ):
        """测试：执行指令时卖出无价格数据股票加入延迟卖出队列（无价格数据原因）"""
        from src.lazybull.paper.models import TradeInstruction

        broker = PaperBroker(
            account=sample_account_with_position, storage=temp_storage, verbose=False
        )

        # 创建卖出指令
        instructions = [
            TradeInstruction(
                ts_code="000001.SZ",
                action="sell",
                shares=1000,
                price_type="close",
                reason="退出持仓",
                source_date="20260119",
                target_weight=0.0,
            )
        ]

        # 无卖出价格：ts_code not in sell_prices 触发 "无卖出价格" 分支
        buy_prices = {}
        sell_prices = {}  # 000001.SZ 无卖出价格

        # Mock 可交易性信息（非停牌，但无价格数据）
        tradability = {
            "000001.SZ": {
                "is_suspended": 0,  # 非停牌
                "is_limit_up": 0,
                "is_limit_down": 0,
                "tradable": 1,
            }
        }

        with (
            patch.object(broker, "_load_tradability_info", return_value=tradability),
            patch.object(broker, "_get_suspend_calendar") as mock_sc,
        ):
            # Mock SuspendCalendar 返回非停牌
            mock_sc.return_value.is_suspended.return_value = False

            # 执行指令
            fills = broker.execute_instructions(
                instructions=instructions,
                buy_prices=buy_prices,
                sell_prices=sell_prices,
                trade_date="20260120",
            )

            # 验证：没有成交
            assert len(fills) == 0, "无价格数据股票不应成交"

            # 验证：加入了延迟卖出队列
            assert len(broker.pending_sells) == 1, "应该加入延迟卖出队列"

            pending_sell = broker.pending_sells[0]
            assert pending_sell.ts_code == "000001.SZ"
            assert pending_sell.shares == 1000
            assert "无价格数据" in pending_sell.reason, "原因应包含'无价格数据'"

    def test_execute_instructions_suspended_stock_added_to_pending_sells(
        self, sample_account_with_position, temp_storage
    ):
        """测试：执行指令时停牌股票加入延迟卖出队列"""
        from src.lazybull.paper.models import TradeInstruction

        broker = PaperBroker(
            account=sample_account_with_position, storage=temp_storage, verbose=False
        )

        # 创建卖出指令
        instructions = [
            TradeInstruction(
                ts_code="000001.SZ",
                action="sell",
                shares=1000,
                price_type="close",
                reason="止损卖出",
                source_date="20260119",
                target_weight=0.0,
            )
        ]

        # 无卖出价格
        buy_prices = {}
        sell_prices = {}

        # Mock 可交易性信息（停牌）
        tradability = {
            "000001.SZ": {
                "is_suspended": 1,
                "is_limit_up": 0,
                "is_limit_down": 0,
                "tradable": 0,  # 停牌股票不可交易
            }
        }

        with (
            patch.object(broker, "_load_tradability_info", return_value=tradability),
            patch.object(broker, "_get_suspend_calendar") as mock_sc,
        ):
            # Mock SuspendCalendar 返回停牌
            mock_sc.return_value.is_suspended.return_value = True

            # 执行指令
            fills = broker.execute_instructions(
                instructions=instructions,
                buy_prices=buy_prices,
                sell_prices=sell_prices,
                trade_date="20260120",
            )

            # 验证：没有成交
            assert len(fills) == 0, "停牌股票不应成交"

            # 验证：加入了延迟卖出队列
            assert len(broker.pending_sells) == 1, "应该加入延迟卖出队列"

            pending_sell = broker.pending_sells[0]
            assert pending_sell.ts_code == "000001.SZ"
            assert "停牌" in pending_sell.reason, "原因应包含'停牌'"


class TestBacktestSuspendedHandling:
    """回测停牌处理测试"""

    def test_suspended_stock_skipped_in_backtest_stop_loss(self):
        """测试：回测中停牌股票在止损检查中被跳过"""
        from src.lazybull.backtest.engine import BacktestEngine
        from src.lazybull.signals.base import Signal
        from src.lazybull.universe.base import Universe

        # 创建 mock 的 universe 和 signal
        mock_universe = MagicMock(spec=Universe)
        mock_signal = MagicMock(spec=Signal)

        # 创建回测引擎
        stop_loss_config = StopLossConfig(
            enabled=True, drawdown_pct=10.0, consecutive_limit_down_days=2
        )

        engine = BacktestEngine(
            universe=mock_universe,
            signal=mock_signal,
            initial_capital=100000.0,
            rebalance_freq=5,
            stop_loss_config=stop_loss_config,
            verbose=False,
        )

        # 设置持仓
        test_date = pd.Timestamp("2026-01-20")
        engine.positions["000001.SZ"] = {
            "shares": 1000,
            "buy_date": pd.Timestamp("2026-01-10"),
            "buy_trade_price": 10.0,
            "buy_pnl_price": 10.0,
            "signal_date": pd.Timestamp("2026-01-09"),
        }

        # Mock 价格数据缓存（停牌股票）
        engine.price_data_cache = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260120"],
                "close": [0.0],  # 停牌时价格可能为0
                "close_adj": [0.0],
                "is_limit_down": [0],
                "is_suspended": [1],  # 停牌标志
            }
        )

        # Mock _get_trade_price 返回价格，Mock _get_suspend_calendar 返回停牌
        with (
            patch.object(engine, "_get_trade_price", return_value=8.0),
            patch.object(engine, "_get_suspend_calendar") as mock_sc,
        ):
            mock_sc.return_value.is_suspended.return_value = True

            # 执行止损检查
            trading_dates = [pd.Timestamp("2026-01-19"), test_date, pd.Timestamp("2026-01-21")]
            date_to_idx = {d: i for i, d in enumerate(trading_dates)}

            engine._check_stop_loss(test_date, trading_dates, date_to_idx)

            # 验证：停牌股票不会加入待止损卖出队列
            assert "000001.SZ" not in engine.pending_stop_loss_sells, "停牌股票不应触发止损"

    def test_suspended_stock_sell_deferred_in_backtest(self):
        """测试：回测中停牌股票卖出会延迟"""
        from src.lazybull.backtest.engine import BacktestEngine
        from src.lazybull.signals.base import Signal
        from src.lazybull.universe.base import Universe

        # 创建 mock 的 universe 和 signal
        mock_universe = MagicMock(spec=Universe)
        mock_signal = MagicMock(spec=Signal)

        # 创建回测引擎（启用延迟订单）
        engine = BacktestEngine(
            universe=mock_universe,
            signal=mock_signal,
            initial_capital=100000.0,
            rebalance_freq=5,
            enable_pending_order=True,
            verbose=False,
        )

        # 设置持仓
        test_date = pd.Timestamp("2026-01-20")
        engine.positions["000001.SZ"] = {
            "shares": 1000,
            "buy_date": pd.Timestamp("2026-01-10"),
            "buy_trade_price": 10.0,
            "buy_pnl_price": 10.0,
            "signal_date": pd.Timestamp("2026-01-09"),
        }
        engine.cash = 90000.0

        # Mock 价格数据缓存（停牌股票）
        engine.price_data_cache = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260120"],
                "close": [10.0],
                "close_adj": [10.0],
                "open": [10.0],
                "open_adj": [10.0],
                "is_limit_down": [0],
                "is_suspended": [1],  # 停牌标志
            }
        )

        # 初始化延迟订单管理器
        from src.lazybull.execution.pending_order import PendingOrderManager

        engine.pending_order_manager = PendingOrderManager(max_retry_count=5, max_retry_days=10)

        # Mock _get_trade_price 和 _get_pnl_price
        with (
            patch.object(engine, "_get_trade_price", return_value=10.0),
            patch.object(engine, "_get_pnl_price", return_value=10.0),
        ):

            # 尝试卖出
            engine._sell_stock(test_date, "000001.SZ", sell_type="holding_period")

            # 验证：持仓未被卖出（因为停牌）
            assert "000001.SZ" in engine.positions, "停牌股票不应立即卖出"
            assert engine.positions["000001.SZ"]["shares"] == 1000, "持仓股数不应改变"

            # 验证：卖出被加入延迟队列
            pending_orders = engine.pending_order_manager.get_all_orders()
            assert len(pending_orders) > 0, "应该有延迟订单"

            pending_sell = pending_orders[0]
            assert pending_sell.stock == "000001.SZ"
            assert pending_sell.action == "sell"
            assert "停牌" in pending_sell.last_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
