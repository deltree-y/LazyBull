"""测试 T+1 纸面交易功能"""

import json
import os
from pathlib import Path

import pytest

from src.lazybull.common.cost import CostModel
from src.lazybull.live.mock_broker import MockBroker, OrderResult
from src.lazybull.live.persistence import SimplePersistence


@pytest.fixture
def temp_state_file(tmp_path):
    """创建临时状态文件路径"""
    return str(tmp_path / "test_trading_state.json")


@pytest.fixture
def persistence(temp_state_file):
    """创建持久化实例"""
    return SimplePersistence(file_path=temp_state_file)


@pytest.fixture
def broker(persistence):
    """创建模拟券商实例"""
    return MockBroker(persistence=persistence)


class TestSimplePersistence:
    """测试持久化模块"""
    
    def test_init_creates_empty_state(self, temp_state_file):
        """测试初始化创建空状态"""
        p = SimplePersistence(file_path=temp_state_file)
        
        state = p.get_state()
        assert "account" in state
        assert "positions" in state
        assert "orders" in state
        assert "pending_signals" in state
        assert state["account"]["cash"] == 0.0
        assert len(state["positions"]) == 0
        assert len(state["orders"]) == 0
        assert len(state["pending_signals"]) == 0
    
    def test_init_account(self, persistence):
        """测试初始化账户"""
        persistence.init_account(100000.0)
        
        account = persistence.get_account()
        assert account["cash"] == 100000.0
        assert account["initial_cash"] == 100000.0
    
    def test_update_cash(self, persistence):
        """测试更新现金"""
        persistence.init_account(100000.0)
        persistence.update_cash(80000.0)
        
        account = persistence.get_account()
        assert account["cash"] == 80000.0
        assert account["initial_cash"] == 100000.0
    
    def test_update_position(self, persistence):
        """测试更新持仓"""
        persistence.update_position(
            code="000001.SZ",
            shares=1000,
            avg_cost=10.5,
            last_price=11.0
        )
        
        position = persistence.get_position("000001.SZ")
        assert position["code"] == "000001.SZ"
        assert position["shares"] == 1000
        assert position["avg_cost"] == 10.5
        assert position["last_price"] == 11.0
    
    def test_clear_position_when_shares_zero(self, persistence):
        """测试股数为 0 时清空持仓"""
        persistence.update_position("000001.SZ", 1000, 10.5, 11.0)
        persistence.update_position("000001.SZ", 0, 0.0, 11.0)
        
        position = persistence.get_position("000001.SZ")
        assert position is None
    
    def test_add_order(self, persistence):
        """测试添加订单"""
        order = {
            "order_id": "O001",
            "code": "000001.SZ",
            "direction": "buy",
            "shares": 1000,
            "price": 10.5,
            "amount": 10500.0,
            "cost": 31.5,
            "status": "filled",
            "create_time": "2024-01-15 10:00:00",
            "fill_time": "2024-01-15 10:00:01"
        }
        
        persistence.add_order(order)
        
        orders = persistence.get_orders()
        assert len(orders) == 1
        assert orders[0]["order_id"] == "O001"
        assert orders[0]["code"] == "000001.SZ"
    
    def test_get_orders_with_filters(self, persistence):
        """测试订单查询过滤"""
        persistence.add_order({
            "order_id": "O001",
            "code": "000001.SZ",
            "direction": "buy",
            "shares": 1000,
            "price": 10.5,
            "amount": 10500.0,
            "cost": 31.5,
            "status": "filled",
            "create_time": "2024-01-15 10:00:00",
            "fill_time": "2024-01-15 10:00:01"
        })
        
        persistence.add_order({
            "order_id": "O002",
            "code": "000002.SZ",
            "direction": "sell",
            "shares": 500,
            "price": 20.0,
            "amount": 10000.0,
            "cost": 35.0,
            "status": "filled",
            "create_time": "2024-01-16 10:00:00",
            "fill_time": "2024-01-16 10:00:01"
        })
        
        # 按代码过滤
        orders = persistence.get_orders(code="000001.SZ")
        assert len(orders) == 1
        assert orders[0]["code"] == "000001.SZ"
        
        # 按方向过滤
        orders = persistence.get_orders(direction="sell")
        assert len(orders) == 1
        assert orders[0]["direction"] == "sell"
        
        # 限制数量
        orders = persistence.get_orders(limit=1)
        assert len(orders) == 1
        assert orders[0]["order_id"] == "O002"  # 最新的
    
    def test_add_pending_signal(self, persistence):
        """测试添加待执行信号"""
        signals = {
            "000001.SZ": 0.5,
            "000002.SZ": 0.5
        }
        
        persistence.add_pending_signal(
            trade_date="20240115",
            exec_date="20240116",
            signals=signals,
            top_n=2
        )
        
        pending = persistence.get_pending_signals()
        assert len(pending) == 1
        assert pending[0]["trade_date"] == "20240115"
        assert pending[0]["exec_date"] == "20240116"
        assert pending[0]["executed"] == False
        assert len(pending[0]["signals"]) == 2
    
    def test_mark_signal_executed(self, persistence):
        """测试标记信号为已执行"""
        signals = {"000001.SZ": 1.0}
        persistence.add_pending_signal("20240115", "20240116", signals, 1)
        
        persistence.mark_signal_executed("20240115")
        
        pending = persistence.get_pending_signals(executed=False)
        assert len(pending) == 0
        
        pending = persistence.get_pending_signals(executed=True)
        assert len(pending) == 1
        assert pending[0]["executed"] == True
    
    def test_clear_executed_signals(self, persistence):
        """测试清除已执行信号"""
        persistence.add_pending_signal("20240115", "20240116", {"000001.SZ": 1.0}, 1)
        persistence.add_pending_signal("20240116", "20240117", {"000002.SZ": 1.0}, 1)
        
        persistence.mark_signal_executed("20240115")
        persistence.clear_executed_signals()
        
        all_signals = persistence.get_pending_signals()
        assert len(all_signals) == 1
        assert all_signals[0]["trade_date"] == "20240116"
    
    def test_persistence_survives_reload(self, temp_state_file):
        """测试持久化在重新加载后保持"""
        p1 = SimplePersistence(file_path=temp_state_file)
        p1.init_account(100000.0)
        p1.update_position("000001.SZ", 1000, 10.5, 11.0)
        
        # 重新加载
        p2 = SimplePersistence(file_path=temp_state_file)
        account = p2.get_account()
        assert account["cash"] == 100000.0
        
        position = p2.get_position("000001.SZ")
        assert position["shares"] == 1000
    
    def test_reset(self, persistence):
        """测试重置状态"""
        persistence.init_account(100000.0)
        persistence.update_position("000001.SZ", 1000, 10.5, 11.0)
        persistence.add_order({
            "order_id": "O001",
            "code": "000001.SZ",
            "direction": "buy",
            "shares": 1000,
            "price": 10.5,
            "amount": 10500.0,
            "cost": 31.5,
            "status": "filled",
            "create_time": "2024-01-15 10:00:00",
            "fill_time": "2024-01-15 10:00:01"
        })
        
        persistence.reset()
        
        state = persistence.get_state()
        assert state["account"]["cash"] == 0.0
        assert len(state["positions"]) == 0
        assert len(state["orders"]) == 0
        assert len(state["pending_signals"]) == 0


class TestMockBroker:
    """测试模拟券商"""
    
    def test_init_with_default_cost_model(self, persistence):
        """测试使用默认成本模型初始化"""
        broker = MockBroker(persistence=persistence)
        assert broker.cost_model is not None
        assert broker.cost_model.commission_rate == 0.0003
    
    def test_buy_order_success(self, broker, persistence):
        """测试成功的买入订单"""
        persistence.init_account(100000.0)
        
        result = broker.place_order(
            code="000001.SZ",
            direction="buy",
            shares=1000,
            price=10.0
        )
        
        assert result.is_success()
        assert result.code == "000001.SZ"
        assert result.shares == 1000
        assert result.price == 10.0
        assert result.amount == 10000.0
        
        # 检查账户
        account = persistence.get_account()
        assert account["cash"] < 100000.0  # 扣除了成本
        
        # 检查持仓
        position = persistence.get_position("000001.SZ")
        assert position["shares"] == 1000
        assert position["avg_cost"] == 10.0
    
    def test_buy_order_insufficient_cash(self, broker, persistence):
        """测试资金不足的买入订单"""
        persistence.init_account(5000.0)
        
        result = broker.place_order(
            code="000001.SZ",
            direction="buy",
            shares=1000,
            price=10.0
        )
        
        assert not result.is_success()
        assert result.status == "rejected"
        assert "资金不足" in result.message
    
    def test_buy_order_invalid_shares(self, broker, persistence):
        """测试无效股数的买入订单"""
        persistence.init_account(100000.0)
        
        result = broker.place_order(
            code="000001.SZ",
            direction="buy",
            shares=0,
            price=10.0
        )
        
        assert not result.is_success()
        assert result.status == "rejected"
        assert "股数必须大于 0" in result.message
    
    def test_buy_order_accumulates_position(self, broker, persistence):
        """测试多次买入累积持仓"""
        persistence.init_account(100000.0)
        
        # 第一次买入
        result1 = broker.place_order("000001.SZ", "buy", 1000, 10.0)
        assert result1.is_success()
        
        # 第二次买入
        result2 = broker.place_order("000001.SZ", "buy", 500, 12.0)
        assert result2.is_success()
        
        # 检查持仓
        position = persistence.get_position("000001.SZ")
        assert position["shares"] == 1500
        # 平均成本 = (1000*10 + 500*12) / 1500 = 16000 / 1500 = 10.67
        assert abs(position["avg_cost"] - 10.67) < 0.01
    
    def test_sell_order_success(self, broker, persistence):
        """测试成功的卖出订单"""
        persistence.init_account(100000.0)
        
        # 先买入
        broker.place_order("000001.SZ", "buy", 1000, 10.0)
        
        # 再卖出
        result = broker.place_order("000001.SZ", "sell", 500, 11.0)
        
        assert result.is_success()
        assert result.shares == 500
        assert result.price == 11.0
        
        # 检查持仓
        position = persistence.get_position("000001.SZ")
        assert position["shares"] == 500
    
    def test_sell_order_no_position(self, broker, persistence):
        """测试无持仓的卖出订单"""
        persistence.init_account(100000.0)
        
        result = broker.place_order("000001.SZ", "sell", 1000, 10.0)
        
        assert not result.is_success()
        assert result.status == "rejected"
        assert "无持仓" in result.message
    
    def test_sell_order_insufficient_shares(self, broker, persistence):
        """测试持仓不足的卖出订单"""
        persistence.init_account(100000.0)
        broker.place_order("000001.SZ", "buy", 1000, 10.0)
        
        result = broker.place_order("000001.SZ", "sell", 2000, 11.0)
        
        assert not result.is_success()
        assert result.status == "rejected"
        assert "持仓不足" in result.message
    
    def test_sell_order_clears_position(self, broker, persistence):
        """测试全部卖出清空持仓"""
        persistence.init_account(100000.0)
        broker.place_order("000001.SZ", "buy", 1000, 10.0)
        
        result = broker.place_order("000001.SZ", "sell", 1000, 11.0)
        
        assert result.is_success()
        
        # 检查持仓已清空
        position = persistence.get_position("000001.SZ")
        assert position is None
    
    def test_invalid_direction(self, broker, persistence):
        """测试无效的方向"""
        persistence.init_account(100000.0)
        
        result = broker.place_order("000001.SZ", "hold", 1000, 10.0)
        
        assert not result.is_success()
        assert result.status == "rejected"
        assert "不支持的方向" in result.message
    
    def test_get_account_info(self, broker, persistence):
        """测试获取账户信息"""
        persistence.init_account(100000.0)
        
        # 买入一些股票
        broker.place_order("000001.SZ", "buy", 1000, 10.0)
        broker.place_order("000002.SZ", "buy", 500, 20.0)
        
        info = broker.get_account_info()
        
        assert "cash" in info
        assert "position_value" in info
        assert "total_value" in info
        assert "initial_cash" in info
        assert "positions" in info
        
        assert info["initial_cash"] == 100000.0
        assert len(info["positions"]) == 2
        
        # 持仓市值应该是 1000*10 + 500*20 = 20000
        assert info["position_value"] == 20000.0
    
    def test_order_generates_unique_id(self, broker, persistence):
        """测试订单生成唯一 ID"""
        persistence.init_account(100000.0)
        
        result1 = broker.place_order("000001.SZ", "buy", 1000, 10.0)
        result2 = broker.place_order("000002.SZ", "buy", 1000, 10.0)
        
        assert result1.order_id != result2.order_id
    
    def test_orders_are_persisted(self, broker, persistence):
        """测试订单被持久化"""
        persistence.init_account(100000.0)
        
        broker.place_order("000001.SZ", "buy", 1000, 10.0)
        broker.place_order("000001.SZ", "sell", 500, 11.0)
        
        orders = persistence.get_orders()
        assert len(orders) == 2
        assert orders[0]["direction"] == "buy"
        assert orders[1]["direction"] == "sell"
