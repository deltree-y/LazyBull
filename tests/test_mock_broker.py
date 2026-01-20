"""测试MockBroker模块"""

import tempfile
from pathlib import Path

import pytest

from src.lazybull.live.persistence import SimplePersistence
from src.lazybull.live.mock_broker import MockBroker


@pytest.fixture
def temp_state_file():
    """创建临时状态文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "test_broker_state.json"
        yield str(state_file)


@pytest.fixture
def persistence(temp_state_file):
    """创建持久化实例"""
    return SimplePersistence(temp_state_file)


@pytest.fixture
def broker(persistence):
    """创建MockBroker实例"""
    return MockBroker(
        persistence=persistence,
        initial_cash=100000.0,
        commission_rate=0.0003,
        slippage=0.001
    )


def test_broker_initialization(broker):
    """测试券商初始化"""
    assert broker.cash == 100000.0
    assert broker.commission_rate == 0.0003
    assert broker.slippage == 0.001
    assert len(broker.positions) == 0


def test_buy_order_success(broker):
    """测试成功买入"""
    order = broker.place_order(
        symbol="000001.SZ",
        side="buy",
        qty=100,
        price=10.0,
        trade_date="20230601"
    )
    
    # 验证订单状态
    assert order["status"] == "filled"
    assert order["symbol"] == "000001.SZ"
    assert order["side"] == "buy"
    assert order["qty"] == 100
    assert order["filled_qty"] == 100
    
    # 验证价格（含滑点）
    expected_price = 10.0 * (1 + 0.001)  # 买入向上滑点
    assert abs(order["price"] - expected_price) < 0.01
    
    # 验证佣金
    trade_value = 100 * expected_price
    expected_commission = max(trade_value * 0.0003, 5.0)
    assert abs(order["commission"] - expected_commission) < 0.01
    
    # 验证持仓
    assert "000001.SZ" in broker.positions
    assert broker.positions["000001.SZ"]["qty"] == 100
    
    # 验证现金扣减
    total_cost = trade_value + expected_commission
    assert abs(broker.cash - (100000.0 - total_cost)) < 0.01


def test_buy_order_insufficient_cash(broker):
    """测试资金不足买入被拒"""
    # 尝试买入超过资金的数量
    order = broker.place_order(
        symbol="000001.SZ",
        side="buy",
        qty=100000,  # 远超可用资金
        price=10.0,
        trade_date="20230601"
    )
    
    # 验证订单被拒
    assert order["status"] == "rejected"
    assert "资金不足" in order["reason"]
    assert order["filled_qty"] == 0
    
    # 验证持仓和现金未变化
    assert len(broker.positions) == 0
    assert broker.cash == 100000.0


def test_sell_order_success(broker):
    """测试成功卖出"""
    # 先买入
    buy_order = broker.place_order(
        symbol="000001.SZ",
        side="buy",
        qty=100,
        price=10.0,
        trade_date="20230601"
    )
    assert buy_order["status"] == "filled"
    
    cash_after_buy = broker.cash
    
    # 卖出
    sell_order = broker.place_order(
        symbol="000001.SZ",
        side="sell",
        qty=100,
        price=11.0,
        trade_date="20230602"
    )
    
    # 验证订单状态
    assert sell_order["status"] == "filled"
    assert sell_order["symbol"] == "000001.SZ"
    assert sell_order["side"] == "sell"
    assert sell_order["qty"] == 100
    assert sell_order["filled_qty"] == 100
    
    # 验证价格（含滑点）
    expected_price = 11.0 * (1 - 0.001)  # 卖出向下滑点
    assert abs(sell_order["price"] - expected_price) < 0.01
    
    # 验证佣金和印花税
    trade_value = 100 * expected_price
    expected_commission = max(trade_value * 0.0003, 5.0)
    expected_stamp_duty = trade_value * 0.001
    assert abs(sell_order["commission"] - expected_commission) < 0.01
    assert abs(sell_order["stamp_duty"] - expected_stamp_duty) < 0.01
    
    # 验证持仓清空
    assert "000001.SZ" not in broker.positions
    
    # 验证现金增加
    net_proceeds = trade_value - expected_commission - expected_stamp_duty
    assert abs(broker.cash - (cash_after_buy + net_proceeds)) < 0.01


def test_sell_order_insufficient_position(broker):
    """测试持仓不足卖出被拒"""
    # 尝试卖出不存在的持仓
    order = broker.place_order(
        symbol="000001.SZ",
        side="sell",
        qty=100,
        price=10.0,
        trade_date="20230601"
    )
    
    # 验证订单被拒
    assert order["status"] == "rejected"
    assert "持仓不足" in order["reason"]
    assert order["filled_qty"] == 0


def test_sell_partial_position(broker):
    """测试卖出部分持仓"""
    # 买入200股
    buy_order = broker.place_order(
        symbol="000001.SZ",
        side="buy",
        qty=200,
        price=10.0,
        trade_date="20230601"
    )
    assert buy_order["status"] == "filled"
    
    # 卖出100股
    sell_order = broker.place_order(
        symbol="000001.SZ",
        side="sell",
        qty=100,
        price=11.0,
        trade_date="20230602"
    )
    
    # 验证订单成功
    assert sell_order["status"] == "filled"
    
    # 验证剩余持仓
    assert "000001.SZ" in broker.positions
    assert broker.positions["000001.SZ"]["qty"] == 100


def test_multiple_buys_average_cost(broker):
    """测试多次买入计算平均成本"""
    # 第一次买入
    broker.place_order(
        symbol="000001.SZ",
        side="buy",
        qty=100,
        price=10.0,
        trade_date="20230601"
    )
    
    # 第二次买入
    broker.place_order(
        symbol="000001.SZ",
        side="buy",
        qty=100,
        price=12.0,
        trade_date="20230602"
    )
    
    # 验证持仓数量
    assert broker.positions["000001.SZ"]["qty"] == 200
    
    # 验证成本价（应该是加权平均）
    # 注意：价格含滑点
    price1 = 10.0 * (1 + 0.001)
    price2 = 12.0 * (1 + 0.001)
    expected_cost = (100 * price1 + 100 * price2) / 200
    actual_cost = broker.positions["000001.SZ"]["cost_price"]
    assert abs(actual_cost - expected_cost) < 0.01


def test_get_account_info(broker):
    """测试获取账户信息"""
    # 初始状态
    info = broker.get_account_info()
    assert info["cash"] == 100000.0
    assert info["total_value"] == 100000.0
    assert len(info["positions"]) == 0
    
    # 买入后
    broker.place_order(
        symbol="000001.SZ",
        side="buy",
        qty=100,
        price=10.0,
        trade_date="20230601"
    )
    
    info = broker.get_account_info()
    assert info["cash"] < 100000.0  # 现金减少
    assert len(info["positions"]) == 1
    assert info["total_value"] > 0  # 总值包含持仓


def test_persistence_integration(broker, persistence):
    """测试持久化集成"""
    # 执行买入
    broker.place_order(
        symbol="000001.SZ",
        side="buy",
        qty=100,
        price=10.0,
        trade_date="20230601"
    )
    
    # 验证订单已持久化
    orders = persistence.get_orders()
    assert len(orders) == 1
    assert orders[0]["symbol"] == "000001.SZ"
    
    # 验证持仓已持久化
    positions = persistence.get_positions()
    assert "000001.SZ" in positions
    assert positions["000001.SZ"]["qty"] == 100
    
    # 验证账户已持久化
    account = persistence.get_account()
    assert account["cash"] < 100000.0


def test_commission_minimum(broker):
    """测试佣金最低5元"""
    # 买入少量股票，佣金应该是5元
    order = broker.place_order(
        symbol="000001.SZ",
        side="buy",
        qty=1,
        price=10.0,
        trade_date="20230601"
    )
    
    assert order["status"] == "filled"
    assert order["commission"] == 5.0  # 最低佣金


def test_invalid_side(broker):
    """测试无效的买卖方向"""
    with pytest.raises(ValueError, match="不支持的买卖方向"):
        broker.place_order(
            symbol="000001.SZ",
            side="invalid",
            qty=100,
            price=10.0,
            trade_date="20230601"
        )


def test_order_id_uniqueness(broker):
    """测试订单ID唯一性"""
    order1 = broker.place_order(
        symbol="000001.SZ",
        side="buy",
        qty=100,
        price=10.0,
        trade_date="20230601"
    )
    
    order2 = broker.place_order(
        symbol="000002.SZ",
        side="buy",
        qty=100,
        price=10.0,
        trade_date="20230601"
    )
    
    # 验证订单ID不同
    assert order1["local_order_id"] != order2["local_order_id"]
    assert order1["local_order_id"].startswith("MO")
    assert order2["local_order_id"].startswith("MO")
