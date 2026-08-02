"""测试卖出订单原因文案的准确性"""

import tempfile
from unittest.mock import patch

from src.lazybull.common.cost import CostModel
from src.lazybull.paper import (
    PaperAccount,
    PaperBroker,
    PaperStorage,
    TradeInstruction,
)


def test_execution_stats_new_position():
    """测试新建持仓统计"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        broker = PaperBroker(account, cost_model=CostModel(), storage=storage)

        # 买入新股票
        instructions = [
            TradeInstruction(
                ts_code="000001.SZ",
                action="buy",
                shares=1000,
                price_type="close",
                reason="新建仓位",
                source_date="20260120",
                target_weight=0.1,
            ),
        ]

        prices = {"000001.SZ": 10.0}

        with patch.object(broker, "_load_tradability_info", return_value={}):
            fills = broker.execute_instructions(instructions, prices, prices, "20260121")

        # 检查统计
        assert len(fills) == 1
        assert fills[0].action == "buy"
        assert account.get_position("000001.SZ") is not None


def test_execution_stats_add_position():
    """测试加仓统计"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)

        # 先建立持仓
        account.add_position(
            ts_code="000001.SZ", shares=1000, buy_price=10.0, buy_cost=15.0, buy_date="20260120"
        )
        account.update_cash(-10015.0)

        broker = PaperBroker(account, cost_model=CostModel(), storage=storage)

        # 加仓
        instructions = [
            TradeInstruction(
                ts_code="000001.SZ",
                action="buy",
                shares=1000,
                price_type="close",
                reason="加仓",
                source_date="20260120",
                target_weight=0.5,
            ),
        ]

        prices = {"000001.SZ": 10.0}

        with patch.object(broker, "_load_tradability_info", return_value={}):
            fills = broker.execute_instructions(instructions, prices, prices, "20260121")

        # 检查
        assert len(fills) == 1
        assert fills[0].action == "buy"
        pos = account.get_position("000001.SZ")
        assert pos.shares > 1000  # 加仓后股数增加


def test_execution_stats_liquidate():
    """测试清仓统计"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)

        # 建立持仓
        account.add_position(
            ts_code="000001.SZ", shares=1000, buy_price=10.0, buy_cost=15.0, buy_date="20260120"
        )
        account.update_cash(-10015.0)

        broker = PaperBroker(account, cost_model=CostModel(), storage=storage)

        # 清仓
        instructions = [
            TradeInstruction(
                ts_code="000001.SZ",
                action="sell",
                shares=1000,
                price_type="close",
                reason="退出持仓",
                source_date="20260120",
                target_weight=0.0,
            ),
        ]

        prices = {"000001.SZ": 10.0}

        with patch.object(broker, "_load_tradability_info", return_value={}):
            fills = broker.execute_instructions(instructions, prices, prices, "20260121")

        # 检查
        assert len(fills) == 1
        assert fills[0].action == "sell"
        assert fills[0].shares == 1000  # 卖出全部
        assert account.get_position("000001.SZ") is None  # 持仓已清空


def test_execution_stats_reduce_position():
    """测试减仓统计"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)

        # 建立持仓
        account.add_position(
            ts_code="000001.SZ", shares=5000, buy_price=10.0, buy_cost=15.0, buy_date="20260120"
        )
        account.update_cash(-50015.0)

        broker = PaperBroker(account, cost_model=CostModel(), storage=storage)

        # 减仓
        instructions = [
            TradeInstruction(
                ts_code="000001.SZ",
                action="sell",
                shares=1000,
                price_type="close",
                reason="信号调整",
                source_date="20260120",
                target_weight=0.2,
            ),
        ]

        prices = {"000001.SZ": 10.0}

        with patch.object(broker, "_load_tradability_info", return_value={}):
            fills = broker.execute_instructions(instructions, prices, prices, "20260121")

        # 检查
        assert len(fills) == 1
        assert fills[0].action == "sell"
        assert fills[0].shares < 5000  # 部分卖出
        pos = account.get_position("000001.SZ")
        assert pos is not None  # 持仓仍然存在
        assert pos.shares < 5000  # 股数减少
        assert pos.shares > 0


def test_execution_stats_mixed_operations():
    """测试混合操作的统计"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)

        # 建立两只持仓
        account.add_position(
            ts_code="000001.SZ", shares=1000, buy_price=10.0, buy_cost=15.0, buy_date="20260120"
        )
        account.add_position(
            ts_code="000002.SZ", shares=2000, buy_price=20.0, buy_cost=30.0, buy_date="20260120"
        )
        account.update_cash(-50045.0)

        broker = PaperBroker(account, cost_model=CostModel(), storage=storage)

        # 调仓：清仓000001.SZ，减仓000002.SZ，新建000003.SZ
        instructions = [
            TradeInstruction(
                ts_code="000001.SZ",
                action="sell",
                shares=1000,
                price_type="close",
                reason="退出持仓",
                source_date="20260120",
                target_weight=0.0,
            ),
            TradeInstruction(
                ts_code="000002.SZ",
                action="sell",
                shares=500,
                price_type="close",
                reason="减仓",
                source_date="20260120",
                target_weight=0.2,
            ),
            TradeInstruction(
                ts_code="000003.SZ",
                action="buy",
                shares=1000,
                price_type="close",
                reason="新建仓位",
                source_date="20260120",
                target_weight=0.3,
            ),
        ]

        prices = {
            "000001.SZ": 10.0,
            "000002.SZ": 20.0,
            "000003.SZ": 15.0,
        }

        with patch.object(broker, "_load_tradability_info", return_value={}):
            fills = broker.execute_instructions(instructions, prices, prices, "20260121")

        # 检查：应该有清仓、减仓、新建仓位各一笔
        sell_fills = [f for f in fills if f.action == "sell"]
        buy_fills = [f for f in fills if f.action == "buy"]

        assert len(sell_fills) == 2  # 清仓000001.SZ + 减仓000002.SZ
        assert len(buy_fills) == 1  # 新建000003.SZ

        # 验证持仓
        assert account.get_position("000001.SZ") is None  # 已清仓
        assert account.get_position("000002.SZ") is not None  # 仍持有但减少
        assert account.get_position("000003.SZ") is not None  # 新建
