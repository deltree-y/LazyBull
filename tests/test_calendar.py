"""测试交易日历工具函数"""

import pandas as pd
import pytest

from src.lazybull.data import DataLoader, Storage


def test_get_trading_dates_empty():
    """测试空交易日历"""
    loader = DataLoader()
    
    # 当没有数据时应返回空列表
    dates = loader.get_trading_dates('2023-01-01', '2023-12-31')
    assert isinstance(dates, list)


def test_trading_dates_with_real_data():
    storage = Storage()
    
    # 测试加载
    loader = DataLoader(storage)
    dates = loader.get_trading_dates('2023-01-01', '2023-01-10')
    
    # 应该只返回is_open=1的日期
    assert len(dates) == 6  # 10天中有6个交易日
    
    # 验证日期排序
    assert dates == sorted(dates)


def test_date_filtering():
    """测试日期过滤功能"""
    # 创建更大范围的模拟日历
    mock_cal = pd.DataFrame({
        'exchange': ['SSE'] * 365,
        'cal_date': pd.date_range('2023-01-01', periods=365, freq='D'),
        'is_open': [1] * 365,  # 所有日期都开盘（简化）
        'pretrade_date': [None] * 365
    })
    
    storage = Storage()
    storage.save_raw(mock_cal, "trade_cal")
    
    loader = DataLoader(storage)
    
    # 测试日期范围过滤
    dates = loader.get_trading_dates('2023-06-01', '2023-06-30')
    
    # 应该只返回6月的日期
    assert all(pd.to_datetime(d).month == 6 for d in dates)
