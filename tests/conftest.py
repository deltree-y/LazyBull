"""Pytest配置文件"""

import sys
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture
def mock_config():
    """提供模拟配置"""
    from src.lazybull.common.config import Config
    
    config = Config()
    config.set("data.root", "./data")
    config.set("backtest.initial_capital", 1000000)
    config.set("costs.commission_rate", 0.0003)
    
    return config


@pytest.fixture
def mock_stock_basic():
    """提供模拟股票基本信息"""
    import pandas as pd
    
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH'],
        'symbol': ['000001', '000002', '600000'],
        'name': ['平安银行', '万科A', '浦发银行'],
        'market': ['主板', '主板', '主板'],
        'list_date': ['19910403', '19910129', '19991110']
    })
