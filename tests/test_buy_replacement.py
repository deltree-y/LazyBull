"""测试买入失败补位机制

测试场景：
1. 买入失败时生成 PendingBuy 记录
2. 重试 PendingBuy 时检查可交易性
3. 超过最大尝试次数后移除
4. 同日不重复推进 attempts
5. 生成补位目标应用一手约束
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.lazybull.paper import PaperStorage
from src.lazybull.paper.broker import PaperBroker
from src.lazybull.paper.account import PaperAccount
from src.lazybull.paper.models import Fill, Order, PendingBuy, PendingSell, TargetWeight


@pytest.fixture
def temp_storage_dir():
    """创建临时存储目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def storage(temp_storage_dir):
    """创建存储实例"""
    return PaperStorage(root_path=temp_storage_dir, verbose=False)


@pytest.fixture
def account(storage):
    """创建账户实例"""
    return PaperAccount(initial_capital=500000.0, storage=storage, verbose=False)


@pytest.fixture
def broker(account, storage):
    """创建经纪实例"""
    return PaperBroker(account=account, storage=storage, verbose=False)


def test_pending_buy_save_and_load(storage):
    """测试 PendingBuy 的保存和加载"""
    # 创建测试数据
    pending_buys = [
        PendingBuy(
            ts_code='000001.SZ',
            target_weight=0.1,
            reason='补位-涨停',
            create_date='20260121',
            attempts=1,
            last_attempt_date='20260122',
            original_signal_date='20260120'
        ),
        PendingBuy(
            ts_code='000002.SZ',
            target_weight=0.15,
            reason='补位-停牌',
            create_date='20260121',
            attempts=2,
            last_attempt_date='20260123',
            original_signal_date='20260120'
        ),
    ]
    
    # 保存
    storage.save_pending_buys(pending_buys)
    
    # 加载
    loaded = storage.load_pending_buys()
    
    # 验证
    assert len(loaded) == 2
    assert loaded[0].ts_code == '000001.SZ'
    assert loaded[0].target_weight == 0.1
    assert loaded[0].reason == '补位-涨停'
    assert loaded[0].attempts == 1
    assert loaded[0].last_attempt_date == '20260122'
    assert loaded[1].ts_code == '000002.SZ'
    assert loaded[1].attempts == 2


def test_pending_buy_empty_load(storage):
    """测试加载不存在的 PendingBuy 文件"""
    loaded = storage.load_pending_buys()
    assert loaded == []


def test_broker_tracks_failed_buy_targets(broker):
    """测试 broker 跟踪买入失败的目标"""
    # 创建目标
    targets = [
        TargetWeight(ts_code='000001.SZ', target_weight=0.2, reason='信号生成'),
        TargetWeight(ts_code='000002.SZ', target_weight=0.2, reason='信号生成'),
    ]
    
    # Mock 价格和可交易性数据
    buy_prices = {'000001.SZ': 10.0}  # 000002.SZ 无价格，应该失败
    sell_prices = {}
    
    # Mock 可交易性信息
    with patch.object(broker, '_load_tradability_info') as mock_tradability:
        mock_tradability.return_value = {
            '000001.SZ': {
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'tradable': 1
            },
        }
        
        # 生成订单
        orders = broker.generate_orders(targets, buy_prices, sell_prices, '20260121')
        
        # 获取失败目标
        failed_targets = broker.get_failed_buy_targets()
        
        # 验证：000002.SZ 应该因为无价格数据而失败
        assert len(failed_targets) == 1
        assert failed_targets[0].ts_code == '000002.SZ'
        assert '无价格数据' in failed_targets[0].reason


def test_broker_tracks_limit_up_as_failed(broker):
    """测试 broker 将涨停标记为买入失败"""
    # 创建目标
    targets = [
        TargetWeight(ts_code='000001.SZ', target_weight=0.2, reason='信号生成'),
    ]
    
    # Mock 价格
    buy_prices = {'000001.SZ': 10.0}
    sell_prices = {}
    
    # Mock 可交易性信息（涨停）
    with patch.object(broker, '_load_tradability_info') as mock_tradability:
        mock_tradability.return_value = {
            '000001.SZ': {
                'is_suspended': 0,
                'is_limit_up': 1,  # 涨停
                'is_limit_down': 0,
                'tradable': 1
            },
        }
        
        # 生成订单
        orders = broker.generate_orders(targets, buy_prices, sell_prices, '20260121')
        
        # 获取失败目标
        failed_targets = broker.get_failed_buy_targets()
        
        # 验证
        assert len(failed_targets) == 1
        assert failed_targets[0].ts_code == '000001.SZ'
        assert '涨停' in failed_targets[0].reason


def test_retry_pending_buys_success(broker):
    """测试成功重试 PendingBuy"""
    # 添加一个 pending buy
    pending_buy = PendingBuy(
        ts_code='000001.SZ',
        target_weight=0.2,
        reason='补位-涨停',
        create_date='20260121',
        attempts=1,
        last_attempt_date='20260122',
        original_signal_date='20260120'
    )
    broker.pending_buys = [pending_buy]
    
    # Mock 数据加载
    with patch('src.lazybull.paper.broker.DataLoader') as MockLoader, \
         patch('src.lazybull.paper.broker.Storage') as MockStorage:
        
        # Mock daily_data
        mock_loader = MockLoader.return_value
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'close': [10.0],
        })
        mock_loader.load_clean_daily_by_date.return_value = daily_data
        
        # Mock 可交易性（可以买入）
        with patch.object(broker, '_load_tradability_info') as mock_tradability:
            mock_tradability.return_value = {
                '000001.SZ': {
                    'is_suspended': 0,
                    'is_limit_up': 0,
                    'is_limit_down': 0,
                    'tradable': 1
                },
            }
            
            # 重试
            fills, remaining = broker.retry_pending_buys('20260123', 'close')
            
            # 验证：应该成功买入
            assert len(fills) == 1
            assert fills[0].ts_code == '000001.SZ'
            assert fills[0].action == 'buy'
            assert len(remaining) == 0


def test_retry_pending_buys_still_limit_up(broker):
    """测试重试时仍然涨停的情况"""
    # 添加一个 pending buy
    pending_buy = PendingBuy(
        ts_code='000001.SZ',
        target_weight=0.2,
        reason='补位-涨停',
        create_date='20260121',
        attempts=1,
        last_attempt_date='20260122',
        original_signal_date='20260120'
    )
    broker.pending_buys = [pending_buy]
    
    # Mock 数据加载
    with patch('src.lazybull.paper.broker.DataLoader') as MockLoader, \
         patch('src.lazybull.paper.broker.Storage') as MockStorage:
        
        # Mock daily_data
        mock_loader = MockLoader.return_value
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'close': [11.0],
        })
        mock_loader.load_clean_daily_by_date.return_value = daily_data
        
        # Mock 可交易性（仍然涨停）
        with patch.object(broker, '_load_tradability_info') as mock_tradability:
            mock_tradability.return_value = {
                '000001.SZ': {
                    'is_suspended': 0,
                    'is_limit_up': 1,  # 仍然涨停
                    'is_limit_down': 0,
                    'tradable': 1
                },
            }
            
            # 重试
            fills, remaining = broker.retry_pending_buys('20260123', 'close')
            
            # 验证：应该保留在队列中，attempts 增加
            assert len(fills) == 0
            assert len(remaining) == 1
            assert remaining[0].attempts == 2
            assert remaining[0].last_attempt_date == '20260123'


def test_retry_pending_buys_max_attempts_exceeded(broker):
    """测试超过最大尝试次数"""
    # 添加一个已经尝试 5 次的 pending buy
    pending_buy = PendingBuy(
        ts_code='000001.SZ',
        target_weight=0.2,
        reason='补位-涨停',
        create_date='20260121',
        attempts=5,
        last_attempt_date='20260127',
        original_signal_date='20260120'
    )
    broker.pending_buys = [pending_buy]
    
    # Mock 数据加载
    with patch('src.lazybull.paper.broker.DataLoader') as MockLoader, \
         patch('src.lazybull.paper.broker.Storage') as MockStorage:
        
        # Mock daily_data
        mock_loader = MockLoader.return_value
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'close': [12.0],
        })
        mock_loader.load_clean_daily_by_date.return_value = daily_data
        
        # Mock 可交易性（可以买入，但已过期）
        with patch.object(broker, '_load_tradability_info') as mock_tradability:
            mock_tradability.return_value = {
                '000001.SZ': {
                    'is_suspended': 0,
                    'is_limit_up': 0,
                    'is_limit_down': 0,
                    'tradable': 1
                },
            }
            
            # 重试（第6次，应该移除）
            fills, remaining = broker.retry_pending_buys('20260128', 'close', max_attempts=5)
            
            # 验证：应该被移除
            assert len(fills) == 0
            assert len(remaining) == 0


def test_retry_pending_buys_same_day_no_increment(broker):
    """测试同日重试不增加 attempts"""
    # 添加一个 pending buy（last_attempt_date 是今天）
    pending_buy = PendingBuy(
        ts_code='000001.SZ',
        target_weight=0.2,
        reason='补位-涨停',
        create_date='20260121',
        attempts=2,
        last_attempt_date='20260123',  # 今天
        original_signal_date='20260120'
    )
    broker.pending_buys = [pending_buy]
    
    # Mock 数据加载
    with patch('src.lazybull.paper.broker.DataLoader') as MockLoader, \
         patch('src.lazybull.paper.broker.Storage') as MockStorage:
        
        # Mock daily_data
        mock_loader = MockLoader.return_value
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'close': [11.0],
        })
        mock_loader.load_clean_daily_by_date.return_value = daily_data
        
        # Mock 可交易性（涨停）
        with patch.object(broker, '_load_tradability_info') as mock_tradability:
            mock_tradability.return_value = {
                '000001.SZ': {
                    'is_suspended': 0,
                    'is_limit_up': 1,
                    'is_limit_down': 0,
                    'tradable': 1
                },
            }
            
            # 重试（同一天）
            fills, remaining = broker.retry_pending_buys('20260123', 'close')
            
            # 验证：attempts 不应增加
            assert len(remaining) == 1
            assert remaining[0].attempts == 2  # 仍然是2，没有增加


def test_clear_failed_buy_targets(broker):
    """测试清空失败买入目标"""
    broker._failed_buy_targets = [
        TargetWeight(ts_code='000001.SZ', target_weight=0.2, reason='涨停'),
    ]
    
    broker.clear_failed_buy_targets()
    
    assert len(broker.get_failed_buy_targets()) == 0
