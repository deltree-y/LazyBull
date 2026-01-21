"""测试数据确保和 T0 打印增强功能"""

import tempfile
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.lazybull.data import DataCleaner, DataLoader, Storage, TushareClient
from src.lazybull.data.ensure import (
    ensure_basic_data,
    ensure_clean_data_for_date,
    ensure_raw_data_for_date,
)
from src.lazybull.features import FeatureBuilder, ensure_features_for_date
from src.lazybull.paper import PaperAccount, PaperStorage, TargetWeight
from src.lazybull.paper.runner import PaperTradingRunner


@pytest.fixture
def mock_client():
    """模拟 TushareClient"""
    client = Mock(spec=TushareClient)
    
    # 模拟交易日历
    trade_cal = pd.DataFrame({
        'exchange': ['SSE'] * 5,
        'cal_date': ['20250120', '20250121', '20250122', '20250123', '20250124'],
        'is_open': [1, 1, 1, 1, 1],
    })
    client.get_trade_cal.return_value = trade_cal
    
    # 模拟股票基本信息
    stock_basic = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'name': ['测试股票1', '测试股票2'],
        'list_date': ['20200101', '20200101'],
        'market': ['主板', '主板'],
    })
    client.get_stock_basic.return_value = stock_basic
    
    # 模拟日线数据
    daily = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'trade_date': ['20250121', '20250121'],
        'open': [10.0, 20.0],
        'high': [11.0, 21.0],
        'low': [9.0, 19.0],
        'close': [10.5, 20.5],
        'vol': [1000000, 2000000],
        'amount': [10000000, 40000000],
        'pct_chg': [5.0, 2.5],
        'pre_close': [10.0, 20.0],
    })
    client.get_daily.return_value = daily
    
    # 模拟复权因子
    adj_factor = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'trade_date': ['20250121', '20250121'],
        'adj_factor': [1.0, 1.0],
    })
    client.get_adj_factor.return_value = adj_factor
    
    # 模拟停复牌和涨跌停（空数据）
    client.get_suspend_d.return_value = pd.DataFrame()
    client.get_stk_limit.return_value = pd.DataFrame()
    
    return client


@pytest.fixture
def temp_storage():
    """临时存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        yield storage


def test_ensure_raw_data_for_date(mock_client, temp_storage):
    """测试确保 raw 数据存在"""
    trade_date = '20250121'
    
    # 首次调用应该下载数据
    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date)
    assert result is True
    
    # 再次调用应该跳过下载（数据已存在）
    result = ensure_raw_data_for_date(mock_client, temp_storage, trade_date, force=False)
    assert result is True


def test_ensure_basic_data(mock_client, temp_storage):
    """测试确保基础数据存在"""
    end_date = '20250121'
    
    # 首次调用应该下载数据
    result = ensure_basic_data(mock_client, temp_storage, end_date)
    assert result is True
    
    # 验证数据已保存
    trade_cal = temp_storage.load_raw("trade_cal")
    assert trade_cal is not None
    assert len(trade_cal) > 0
    
    stock_basic = temp_storage.load_raw("stock_basic")
    assert stock_basic is not None
    assert len(stock_basic) > 0


def test_ensure_clean_data_for_date(mock_client, temp_storage):
    """测试确保 clean 数据存在"""
    trade_date = '20250121'
    loader = DataLoader(temp_storage)
    cleaner = DataCleaner()
    
    # 确保基础数据存在
    ensure_basic_data(mock_client, temp_storage, trade_date)
    
    # 确保 clean 数据
    result = ensure_clean_data_for_date(
        temp_storage, loader, cleaner, mock_client, trade_date
    )
    assert result is True
    
    # 验证 clean 数据已保存
    assert temp_storage.is_data_exists("clean", "daily", trade_date)


def test_print_t0_targets():
    """测试 T0 打印信息"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 runner
        runner = PaperTradingRunner(
            initial_capital=500000.0,
            data_root=tmpdir,
            paper_root=tmpdir
        )
        
        # 创建测试数据
        targets = [
            TargetWeight(ts_code='000001.SZ', target_weight=0.2, reason='测试信号1'),
            TargetWeight(ts_code='000002.SZ', target_weight=0.3, reason='测试信号2'),
        ]
        
        stock_basic = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'name': ['测试股票1', '测试股票2'],
        })
        
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'close': [10.5, 20.5],
        })
        
        # 调用打印方法（不应抛出异常）
        try:
            runner._print_t0_targets(targets, stock_basic, daily_data)
            success = True
        except Exception as e:
            print(f"打印失败: {e}")
            success = False
        
        assert success is True


def test_enhance_target_info():
    """测试增强目标信息"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建 runner
        runner = PaperTradingRunner(
            initial_capital=500000.0,
            data_root=tmpdir,
            paper_root=tmpdir
        )
        
        # 创建测试数据
        signal_dict = {
            '000001.SZ': 0.2,
            '000002.SZ': 0.3,
        }
        
        stock_basic = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'name': ['测试股票1', '测试股票2'],
        })
        
        daily_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'close': [10.5, 20.5],
        })
        
        # 调用增强方法
        targets = runner._enhance_target_info(
            signal_dict, stock_basic, daily_data, '20250121'
        )
        
        # 验证结果
        assert len(targets) == 2
        assert targets[0].ts_code == '000001.SZ'
        assert targets[0].target_weight == 0.2
        assert '权重=0.2000' in targets[0].reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
