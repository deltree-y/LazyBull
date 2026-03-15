"""测试卖出订单原因文案的准确性"""

import tempfile
from unittest.mock import patch

from src.lazybull.common.cost import CostModel
from src.lazybull.paper import (
    PaperAccount,
    PaperBroker,
    PaperStorage,
    TargetWeight,
)


def test_sell_order_reason_full_liquidation():
    """测试完全清仓时的 reason 为"退出持仓" """
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)

        # 建立持仓（100的倍数）
        account.add_position(
            ts_code="000001.SZ",
            shares=1000,  # 100的倍数
            buy_price=10.0,
            buy_cost=15.0,
            buy_date="20260120",
        )
        account.update_cash(-10015.0)

        broker = PaperBroker(account, storage=storage)

        # 清仓目标（target_weight=0，且应该卖出全部股数）
        targets = []  # 空目标列表意味着清仓所有持仓

        prices = {"000001.SZ": 10.0}

        # Mock the tradability check to return empty dict (no restrictions)
        with patch.object(broker, "_load_tradability_info", return_value={}):
            orders = broker.generate_orders(targets, prices, prices, "20260121")

        # 应该生成卖出订单
        assert len(orders) == 1
        assert orders[0].action == "sell"
        assert orders[0].ts_code == "000001.SZ"
        assert orders[0].shares == 1000  # 卖出全部
        assert orders[0].reason == "退出持仓"  # 完全清仓


def test_sell_order_reason_reduce_position():
    """测试减仓时的 reason 为"减仓" """
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)

        # 建立持仓
        account.add_position(
            ts_code="000001.SZ", shares=5000, buy_price=10.0, buy_cost=15.0, buy_date="20260120"
        )
        account.update_cash(-50015.0)

        broker = PaperBroker(account, storage=storage)

        # 减仓目标（target_weight>0但小于当前权重）
        targets = [
            TargetWeight(ts_code="000001.SZ", target_weight=0.2, reason="信号调整"),
        ]

        prices = {"000001.SZ": 10.0}

        with patch.object(broker, "_load_tradability_info", return_value={}):
            orders = broker.generate_orders(targets, prices, prices, "20260121")

        # 应该生成卖出订单
        assert len(orders) == 1
        assert orders[0].action == "sell"
        assert orders[0].ts_code == "000001.SZ"
        assert orders[0].shares < 5000  # 部分卖出
        assert orders[0].reason == "减仓"  # 减仓而非清仓


def test_execution_stats_new_position():
    """测试新建持仓统计"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = PaperStorage(tmpdir)
        account = PaperAccount(initial_capital=100000.0, storage=storage)
        broker = PaperBroker(account, cost_model=CostModel(), storage=storage)

        # 买入新股票
        targets = [
            TargetWeight(ts_code="000001.SZ", target_weight=0.1, reason="新建仓位"),
        ]

        prices = {"000001.SZ": 10.0}

        with patch.object(broker, "_load_tradability_info", return_value={}):
            orders = broker.generate_orders(targets, prices, prices, "20260121")
        fills = broker.execute_orders(orders, "20260121", "close", "close")

        # 检查统计（通过查看日志输出，这里只能间接验证）
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
        targets = [
            TargetWeight(ts_code="000001.SZ", target_weight=0.5, reason="加仓"),
        ]

        prices = {"000001.SZ": 10.0}

        with patch.object(broker, "_load_tradability_info", return_value={}):
            orders = broker.generate_orders(targets, prices, prices, "20260121")
        fills = broker.execute_orders(orders, "20260121", "close", "close")

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
        targets = []  # 空目标列表意味着清仓所有持仓

        prices = {"000001.SZ": 10.0}

        with patch.object(broker, "_load_tradability_info", return_value={}):
            orders = broker.generate_orders(targets, prices, prices, "20260121")
        fills = broker.execute_orders(orders, "20260121", "close", "close")

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
        targets = [
            TargetWeight(ts_code="000001.SZ", target_weight=0.2, reason="信号调整"),
        ]

        prices = {"000001.SZ": 10.0}

        with patch.object(broker, "_load_tradability_info", return_value={}):
            orders = broker.generate_orders(targets, prices, prices, "20260121")
        fills = broker.execute_orders(orders, "20260121", "close", "close")

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
        targets = [
            TargetWeight(ts_code="000002.SZ", target_weight=0.2, reason="减仓"),
            TargetWeight(ts_code="000003.SZ", target_weight=0.3, reason="新建仓位"),
        ]

        prices = {
            "000001.SZ": 10.0,
            "000002.SZ": 20.0,
            "000003.SZ": 15.0,
        }

        with patch.object(broker, "_load_tradability_info", return_value={}):
            orders = broker.generate_orders(targets, prices, prices, "20260121")
        fills = broker.execute_orders(orders, "20260121", "close", "close")

        # 检查：应该有清仓、减仓、新建仓位各一笔
        sell_fills = [f for f in fills if f.action == "sell"]
        buy_fills = [f for f in fills if f.action == "buy"]

        assert len(sell_fills) == 2  # 清仓000001.SZ + 减仓000002.SZ
        assert len(buy_fills) == 1  # 新建000003.SZ

        # 验证持仓
        assert account.get_position("000001.SZ") is None  # 已清仓
        assert account.get_position("000002.SZ") is not None  # 仍持有但减少
        assert account.get_position("000003.SZ") is not None  # 新建
