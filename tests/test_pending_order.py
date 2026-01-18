"""测试延迟订单管理器"""

import pandas as pd
import pytest

from src.lazybull.execution.pending_order import PendingOrder, PendingOrderManager


@pytest.fixture
def manager():
    """创建延迟订单管理器实例"""
    return PendingOrderManager(max_retry_count=3, max_retry_days=5)


@pytest.fixture
def base_date():
    """基准日期"""
    return pd.Timestamp('2023-01-10')


def test_add_order_new(manager, base_date):
    """测试添加新订单"""
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=100000.0,
        reason='涨停'
    )
    
    assert manager.get_pending_count() == 1
    assert manager.has_order('000001.SZ', 'buy')
    
    stats = manager.get_statistics()
    assert stats['pending'] == 1
    assert stats['total_added'] == 1


def test_add_order_duplicate(manager, base_date):
    """测试添加重复订单（增加重试次数）"""
    # 第一次添加
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=100000.0,
        reason='涨停'
    )
    
    # 第二次添加（同一股票和操作）
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date + pd.Timedelta(days=1),
        signal_date=base_date,
        target_value=100000.0,
        reason='涨停'
    )
    
    # 应该只有一条订单，但重试次数增加
    assert manager.get_pending_count() == 1
    orders = manager.get_all_orders()
    assert len(orders) == 1
    assert orders[0].retry_count == 2


def test_get_orders_to_retry_normal(manager, base_date):
    """测试获取可重试订单"""
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=100000.0,
        reason='涨停'
    )
    
    # 1天后，应该可以重试
    retry_orders = manager.get_orders_to_retry(base_date + pd.Timedelta(days=1))
    assert len(retry_orders) == 1
    assert retry_orders[0].stock == '000001.SZ'


def test_get_orders_to_retry_max_retry_exceeded(manager, base_date):
    """测试超过最大重试次数"""
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=100000.0,
        reason='涨停'
    )
    
    # 模拟多次重试
    for i in range(3):
        manager.add_order(
            stock='000001.SZ',
            action='buy',
            current_date=base_date + pd.Timedelta(days=i+1),
            signal_date=base_date,
            target_value=100000.0,
            reason='涨停'
        )
    
    # retry_count = 4，超过 max_retry_count = 3
    retry_orders = manager.get_orders_to_retry(base_date + pd.Timedelta(days=4))
    assert len(retry_orders) == 0
    assert manager.get_pending_count() == 0
    
    stats = manager.get_statistics()
    assert stats['total_expired'] == 1


def test_get_orders_to_retry_max_days_exceeded(manager, base_date):
    """测试超过最大延迟天数"""
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=100000.0,
        reason='停牌'
    )
    
    # 6天后，超过 max_retry_days = 5
    retry_orders = manager.get_orders_to_retry(base_date + pd.Timedelta(days=6))
    assert len(retry_orders) == 0
    assert manager.get_pending_count() == 0
    
    stats = manager.get_statistics()
    assert stats['total_expired'] == 1


def test_mark_success(manager, base_date):
    """测试标记订单成功"""
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=100000.0,
        reason='涨停'
    )
    
    manager.mark_success('000001.SZ', 'buy')
    
    assert manager.get_pending_count() == 0
    assert not manager.has_order('000001.SZ', 'buy')
    
    stats = manager.get_statistics()
    assert stats['total_succeeded'] == 1


def test_remove_order(manager, base_date):
    """测试手动移除订单"""
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=100000.0,
        reason='涨停'
    )
    
    manager.remove_order('000001.SZ', 'buy')
    
    assert manager.get_pending_count() == 0
    assert not manager.has_order('000001.SZ', 'buy')


def test_multiple_orders(manager, base_date):
    """测试管理多个订单"""
    # 添加多个订单
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=100000.0,
        reason='涨停'
    )
    
    manager.add_order(
        stock='000002.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=200000.0,
        reason='停牌'
    )
    
    manager.add_order(
        stock='000003.SZ',
        action='sell',
        current_date=base_date,
        signal_date=base_date,
        target_value=None,
        reason='跌停'
    )
    
    assert manager.get_pending_count() == 3
    
    # 获取可重试订单
    retry_orders = manager.get_orders_to_retry(base_date + pd.Timedelta(days=1))
    assert len(retry_orders) == 3


def test_clear_all(manager, base_date):
    """测试清空所有订单"""
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=100000.0,
        reason='涨停'
    )
    
    manager.add_order(
        stock='000002.SZ',
        action='sell',
        current_date=base_date,
        signal_date=base_date,
        target_value=None,
        reason='跌停'
    )
    
    manager.clear_all()
    
    assert manager.get_pending_count() == 0


def test_different_actions_same_stock(manager, base_date):
    """测试同一股票的不同操作可以共存"""
    manager.add_order(
        stock='000001.SZ',
        action='buy',
        current_date=base_date,
        signal_date=base_date,
        target_value=100000.0,
        reason='涨停'
    )
    
    manager.add_order(
        stock='000001.SZ',
        action='sell',
        current_date=base_date,
        signal_date=base_date,
        target_value=None,
        reason='跌停'
    )
    
    assert manager.get_pending_count() == 2
    assert manager.has_order('000001.SZ', 'buy')
    assert manager.has_order('000001.SZ', 'sell')
