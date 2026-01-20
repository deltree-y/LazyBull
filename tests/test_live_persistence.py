"""测试持久化模块"""

import json
import tempfile
from pathlib import Path

import pytest

from src.lazybull.live.persistence import SimplePersistence


@pytest.fixture
def temp_state_file():
    """创建临时状态文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "test_state.json"
        yield str(state_file)


def test_initialization_new_file(temp_state_file):
    """测试初始化新文件"""
    persistence = SimplePersistence(temp_state_file)
    
    # 检查初始状态
    state = persistence.load_state()
    assert "orders" in state
    assert "positions" in state
    assert "account" in state
    assert "pending_signals" in state
    assert len(state["orders"]) == 0
    assert len(state["positions"]) == 0
    assert len(state["pending_signals"]) == 0


def test_save_and_load_order(temp_state_file):
    """测试保存和加载订单"""
    persistence = SimplePersistence(temp_state_file)
    
    # 保存订单
    order = {
        "local_order_id": "TEST001",
        "symbol": "000001.SZ",
        "side": "buy",
        "qty": 100,
        "price": 10.5,
        "status": "filled"
    }
    persistence.save_order(order)
    
    # 重新加载
    persistence2 = SimplePersistence(temp_state_file)
    state = persistence2.load_state()
    
    assert len(state["orders"]) == 1
    assert state["orders"][0]["local_order_id"] == "TEST001"
    assert state["orders"][0]["symbol"] == "000001.SZ"


def test_update_order_status(temp_state_file):
    """测试更新订单状态"""
    persistence = SimplePersistence(temp_state_file)
    
    # 保存订单
    order = {
        "local_order_id": "TEST002",
        "symbol": "000002.SZ",
        "status": "pending"
    }
    persistence.save_order(order)
    
    # 更新状态
    persistence.update_order_status("TEST002", "filled", filled_qty=100)
    
    # 验证
    state = persistence.load_state()
    assert state["orders"][0]["status"] == "filled"
    assert state["orders"][0]["filled_qty"] == 100


def test_save_and_get_positions(temp_state_file):
    """测试保存和获取持仓"""
    persistence = SimplePersistence(temp_state_file)
    
    # 保存持仓
    positions = {
        "000001.SZ": {"qty": 100, "cost_price": 10.5},
        "000002.SZ": {"qty": 200, "cost_price": 20.3}
    }
    persistence.save_positions(positions)
    
    # 获取持仓
    loaded_positions = persistence.get_positions()
    assert len(loaded_positions) == 2
    assert loaded_positions["000001.SZ"]["qty"] == 100
    assert loaded_positions["000002.SZ"]["cost_price"] == 20.3


def test_save_and_get_account(temp_state_file):
    """测试保存和获取账户"""
    persistence = SimplePersistence(temp_state_file)
    
    # 保存账户
    persistence.save_account(cash=100000.0, total_value=150000.0)
    
    # 获取账户
    account = persistence.get_account()
    assert account["cash"] == 100000.0
    assert account["total_value"] == 150000.0
    assert "update_time" in account


def test_add_and_get_pending_signals(temp_state_file):
    """测试添加和获取待执行信号"""
    persistence = SimplePersistence(temp_state_file)
    
    # 添加信号
    signals = [
        {"symbol": "000001.SZ", "weight": 0.5, "signal_meta": {"score": 95}},
        {"symbol": "000002.SZ", "weight": 0.5, "signal_meta": {"score": 90}}
    ]
    persistence.add_pending_signals("20230601", signals)
    
    # 获取未执行的信号
    pending = persistence.get_pending_signals(executed=False)
    assert len(pending) == 1
    assert pending[0]["trade_date"] == "20230601"
    assert len(pending[0]["signals"]) == 2
    assert pending[0]["executed"] is False


def test_pop_pending_signals(temp_state_file):
    """测试弹出待执行信号"""
    persistence = SimplePersistence(temp_state_file)
    
    # 添加信号
    signals = [
        {"symbol": "000001.SZ", "weight": 0.5},
        {"symbol": "000002.SZ", "weight": 0.5}
    ]
    persistence.add_pending_signals("20230601", signals)
    
    # 弹出信号
    popped = persistence.pop_pending_signals("20230601")
    assert popped is not None
    assert len(popped) == 2
    
    # 验证已标记为执行
    pending = persistence.get_pending_signals(executed=False)
    assert len(pending) == 0
    
    executed = persistence.get_pending_signals(executed=True)
    assert len(executed) == 1
    
    # 再次弹出应返回None
    popped2 = persistence.pop_pending_signals("20230601")
    assert popped2 is None


def test_pop_nonexistent_signals(temp_state_file):
    """测试弹出不存在的信号"""
    persistence = SimplePersistence(temp_state_file)
    
    # 弹出不存在的信号
    popped = persistence.pop_pending_signals("20230601")
    assert popped is None


def test_get_orders_with_filter(temp_state_file):
    """测试带过滤的获取订单"""
    persistence = SimplePersistence(temp_state_file)
    
    # 保存多个订单
    order1 = {"local_order_id": "TEST001", "trade_date": "20230601"}
    order2 = {"local_order_id": "TEST002", "trade_date": "20230602"}
    persistence.save_order(order1)
    persistence.save_order(order2)
    
    # 按日期过滤
    orders_0601 = persistence.get_orders(date_filter="20230601")
    assert len(orders_0601) == 1
    assert orders_0601[0]["local_order_id"] == "TEST001"
    
    # 获取全部
    all_orders = persistence.get_orders()
    assert len(all_orders) == 2


def test_persistence_across_instances(temp_state_file):
    """测试多个实例间的持久化"""
    # 第一个实例保存数据
    persistence1 = SimplePersistence(temp_state_file)
    persistence1.save_account(cash=200000.0, total_value=250000.0)
    positions = {"000001.SZ": {"qty": 100, "cost_price": 10.0}}
    persistence1.save_positions(positions)
    
    # 第二个实例读取数据
    persistence2 = SimplePersistence(temp_state_file)
    account = persistence2.get_account()
    loaded_positions = persistence2.get_positions()
    
    assert account["cash"] == 200000.0
    assert loaded_positions["000001.SZ"]["qty"] == 100


def test_json_format(temp_state_file):
    """测试JSON格式正确性"""
    persistence = SimplePersistence(temp_state_file)
    
    # 保存一些数据
    persistence.save_account(cash=100000.0, total_value=100000.0)
    order = {"local_order_id": "TEST001", "symbol": "000001.SZ"}
    persistence.save_order(order)
    
    # 直接读取JSON文件验证格式
    with open(temp_state_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert "orders" in data
    assert "positions" in data
    assert "account" in data
    assert "pending_signals" in data
    assert isinstance(data["orders"], list)
    assert isinstance(data["positions"], dict)
