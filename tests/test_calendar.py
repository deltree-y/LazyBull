"""测试交易日历工具函数"""

import tempfile

import pandas as pd
import pytest

from src.lazybull.data import DataLoader, Storage


def test_get_trading_dates_empty():
    """测试空交易日历"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(root_path=tmpdir)
        loader = DataLoader(storage)
        
        # 当没有数据时应返回空列表
        dates = loader.get_trading_dates('2023-01-01', '2023-12-31')
        assert isinstance(dates, list)


def test_trading_dates_with_real_data():
    """测试带有真实数据的交易日期"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(root_path=tmpdir)
        
        # 创建模拟数据
        mock_cal = pd.DataFrame({
            'exchange': ['SSE'] * 10,
            'cal_date': ['20230101', '20230102', '20230103', '20230104', '20230105',
                        '20230106', '20230107', '20230108', '20230109', '20230110'],
            'is_open': [0, 1, 1, 1, 1, 1, 0, 0, 1, 1],  # 7个交易日（周日周六休息）
            'pretrade_date': [None] * 10
        })
        storage.save_raw(mock_cal, "trade_cal")
        
        # 测试加载
        loader = DataLoader(storage)
        dates = loader.get_trading_dates('2023-01-01', '2023-01-10')
        
        # 应该只返回is_open=1的日期
        assert len(dates) == 7  # 10天中有7个交易日
        
        # 验证日期排序
        assert dates == sorted(dates)


def test_date_filtering():
    """测试日期过滤功能"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(root_path=tmpdir)
        
        # 创建更大范围的模拟日历
        dates_range = pd.date_range('2023-01-01', periods=365, freq='D')
        mock_cal = pd.DataFrame({
            'exchange': ['SSE'] * 365,
            'cal_date': dates_range.strftime('%Y%m%d'),
            'is_open': [1] * 365,  # 所有日期都开盘（简化）
            'pretrade_date': [None] * 365
        })
        
        storage.save_raw(mock_cal, "trade_cal")
        
        loader = DataLoader(storage)
        
        # 测试日期范围过滤
        dates = loader.get_trading_dates('2023-06-01', '2023-06-30')
        
        # 应该只返回6月的日期
        assert all(pd.to_datetime(d, format='%Y%m%d').month == 6 for d in dates)
