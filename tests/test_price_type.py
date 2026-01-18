"""测试价格类型选择功能（已废弃 price_type 参数，但保持测试以确保向后兼容）"""

import pandas as pd
import pytest

from src.lazybull.backtest import BacktestEngine
from src.lazybull.common.cost import CostModel
from src.lazybull.signals.base import Signal
from src.lazybull.universe.base import Universe


class MockUniverse(Universe):
    """模拟股票池"""
    
    def __init__(self, stocks):
        self.stocks = stocks
    
    def get_stocks(self, date):
        return self.stocks


class MockSignal(Signal):
    """模拟信号生成器"""
    
    def generate(self, date, universe, data):
        # 等权分配
        n = len(universe)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {stock: weight for stock in universe}


@pytest.fixture
def mock_price_data_with_adj():
    """创建包含不复权和复权价格的模拟数据"""
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ']
    
    data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': 10.0,  # 不复权价格
                'close_adj': 10.0 * 1.1,  # 后复权价格（+10%）
                'close_qfq': 10.0 * 1.05,  # 前复权价格（+5%）
            })
    
    return pd.DataFrame(data)


def test_price_index_trade_price(mock_price_data_with_adj):
    """测试成交价格索引使用不复权价格"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=1000000,
        price_type='close',  # 保留以兼容
        verbose=False
    )
    
    # 准备价格索引
    engine._prepare_price_index(mock_price_data_with_adj)
    
    # 检查成交价格是否为不复权价格
    first_date = pd.Timestamp('2023-01-02')  # 第一个交易日
    trade_price = engine._get_trade_price(first_date, '000001.SZ')
    assert trade_price == 10.0


def test_price_index_pnl_price(mock_price_data_with_adj):
    """测试绩效价格索引使用后复权价格"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=1000000,
        price_type='close_adj',  # 保留以兼容
        verbose=False
    )
    
    # 准备价格索引
    engine._prepare_price_index(mock_price_data_with_adj)
    
    # 检查绩效价格是否为后复权价格
    first_date = pd.Timestamp('2023-01-02')
    pnl_price = engine._get_pnl_price(first_date, '000001.SZ')
    assert pnl_price == 11.0  # 10.0 * 1.1


def test_price_index_separation(mock_price_data_with_adj):
    """测试价格口径分离：成交价 vs 绩效价"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=1000000,
        verbose=False
    )
    
    # 准备价格索引
    engine._prepare_price_index(mock_price_data_with_adj)
    
    # 检查两套价格体系
    first_date = pd.Timestamp('2023-01-02')
    trade_price = engine._get_trade_price(first_date, '000001.SZ')
    pnl_price = engine._get_pnl_price(first_date, '000001.SZ')
    
    assert trade_price == 10.0  # 不复权
    assert pnl_price == 11.0  # 后复权
    assert trade_price != pnl_price  # 两者应不同


def test_price_type_backward_compat():
    """测试 price_type 参数向后兼容（已废弃但不报错）"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    # price_type 参数保留但不再起作用
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        price_type='invalid_type',  # 不再验证
        verbose=False
    )
    
    # 应该正常创建，不报错
    assert engine.price_type == 'invalid_type'


def test_price_fallback_missing_close_adj():
    """测试缺少 close_adj 时的回退机制"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    # 创建只有 close 列的数据
    price_data = pd.DataFrame({
        'ts_code': ['000001.SZ', '000001.SZ'],
        'trade_date': ['20230102', '20230103'],
        'close': [10.0, 10.5]
    })
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        verbose=False
    )
    
    # 应该回退到 'close' 列（会有 warning）
    engine._prepare_price_index(price_data)
    
    first_date = pd.Timestamp('2023-01-02')
    trade_price = engine._get_trade_price(first_date, '000001.SZ')
    pnl_price = engine._get_pnl_price(first_date, '000001.SZ')
    
    # 两者都应该是 close 的值
    assert trade_price == 10.0
    assert pnl_price == 10.0


def test_price_type_default():
    """测试默认价格类型为 close（向后兼容）"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    # 不指定 price_type，应该默认为 'close'
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        verbose=False
    )
    
    assert engine.price_type == 'close'

