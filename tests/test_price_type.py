"""测试价格类型选择功能"""

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
                'close_hfq': 10.0 * 1.05,  # 前复权价格（+5%）
            })
    
    return pd.DataFrame(data)


def test_price_type_close(mock_price_data_with_adj):
    """测试使用不复权价格"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=1000000,
        price_type='close',
        verbose=False
    )
    
    price_dict = engine._prepare_price_dict(mock_price_data_with_adj)
    
    # 检查价格是否为不复权价格
    first_date = pd.Timestamp('2023-01-02')  # 第一个交易日
    assert price_dict[first_date]['000001.SZ'] == 10.0


def test_price_type_close_adj(mock_price_data_with_adj):
    """测试使用后复权价格"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=1000000,
        price_type='close_adj',
        verbose=False
    )
    
    price_dict = engine._prepare_price_dict(mock_price_data_with_adj)
    
    # 检查价格是否为后复权价格
    first_date = pd.Timestamp('2023-01-02')
    assert price_dict[first_date]['000001.SZ'] == 11.0  # 10.0 * 1.1


def test_price_type_close_hfq(mock_price_data_with_adj):
    """测试使用前复权价格"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=1000000,
        price_type='close_qfq',
        verbose=False
    )
    
    price_dict = engine._prepare_price_dict(mock_price_data_with_adj)
    
    # 检查价格是否为前复权价格
    first_date = pd.Timestamp('2023-01-02')
    assert price_dict[first_date]['000001.SZ'] == 10.5  # 10.0 * 1.05


def test_price_type_invalid():
    """测试无效的价格类型"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    with pytest.raises(ValueError, match="不支持的价格类型"):
        BacktestEngine(
            universe=universe,
            signal=signal,
            price_type='invalid_type',
            verbose=False
        )


def test_price_type_fallback():
    """测试价格列不存在时的回退机制"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    # 创建只有 close 列的数据
    price_data = pd.DataFrame({
        'ts_code': ['000001.SZ'],
        'trade_date': ['20230102'],
        'close': [10.0]
    })
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        price_type='close_adj',  # 请求不存在的列
        verbose=False
    )
    
    # 应该回退到 'close' 列
    price_dict = engine._prepare_price_dict(price_data)
    first_date = pd.Timestamp('2023-01-02')
    assert price_dict[first_date]['000001.SZ'] == 10.0


def test_price_type_default():
    """测试默认价格类型为 close（不复权）"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    # 不指定 price_type，应该默认为 'close'
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        verbose=False
    )
    
    assert engine.price_type == 'close'
