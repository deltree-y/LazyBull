"""测试自定义调仓频率功能"""

import tempfile

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
def mock_trading_dates():
    """创建模拟交易日列表"""
    dates = pd.date_range('2023-01-01', periods=50, freq='B')  # 50个交易日
    return [pd.Timestamp(d) for d in dates]


@pytest.fixture
def mock_price_data():
    """创建模拟价格数据"""
    dates = pd.date_range('2023-01-01', periods=50, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH']
    
    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': 10.0,
                'close_adj': 10.0
            })
    
    return pd.DataFrame(data)


def test_rebalance_freq_integer(mock_trading_dates, mock_price_data):
    """测试整数调仓频率（每5个交易日调仓一次）"""
    universe = MockUniverse(['000001.SZ', '000002.SZ'])
    signal = MockSignal()
    
    # 使用整数调仓频率
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=1000000,
        cost_model=CostModel(),
        rebalance_freq=5,  # 每5个交易日调仓一次
        verbose=False
    )
    
    # 验证调仓日期数量
    rebalance_dates = engine._get_rebalance_dates(mock_trading_dates)
    expected_count = (len(mock_trading_dates) + 4) // 5  # 向上取整
    assert len(rebalance_dates) == expected_count
    
    # 验证第一个调仓日是第一个交易日
    assert rebalance_dates[0] == mock_trading_dates[0]
    
    # 验证调仓间隔
    for i in range(1, len(rebalance_dates)):
        # 找到两个调仓日在原列表中的索引
        idx1 = mock_trading_dates.index(rebalance_dates[i-1])
        idx2 = mock_trading_dates.index(rebalance_dates[i])
        # 间隔应该是5个交易日（或最后一次小于5）
        assert idx2 - idx1 == 5 or i == len(rebalance_dates) - 1


def test_rebalance_freq_daily(mock_trading_dates):
    """测试日频调仓"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        rebalance_freq="D",
        verbose=False
    )
    
    rebalance_dates = engine._get_rebalance_dates(mock_trading_dates)
    assert len(rebalance_dates) == len(mock_trading_dates)


def test_rebalance_freq_weekly(mock_trading_dates):
    """测试周频调仓"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        rebalance_freq="W",
        verbose=False
    )
    
    rebalance_dates = engine._get_rebalance_dates(mock_trading_dates)
    # 50个交易日大约是10周
    assert 8 <= len(rebalance_dates) <= 12


def test_rebalance_freq_monthly(mock_trading_dates):
    """测试月频调仓"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        rebalance_freq="M",
        verbose=False
    )
    
    rebalance_dates = engine._get_rebalance_dates(mock_trading_dates)
    # 50个交易日大约是2-3个月
    assert 2 <= len(rebalance_dates) <= 4


def test_rebalance_freq_invalid_integer():
    """测试无效的整数调仓频率（负数或零）"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    with pytest.raises(ValueError, match="调仓频率必须为正整数"):
        engine = BacktestEngine(
            universe=universe,
            signal=signal,
            rebalance_freq=0,
            verbose=False
        )
        engine._get_rebalance_dates([pd.Timestamp('2023-01-01')])
    
    with pytest.raises(ValueError, match="调仓频率必须为正整数"):
        engine = BacktestEngine(
            universe=universe,
            signal=signal,
            rebalance_freq=-5,
            verbose=False
        )
        engine._get_rebalance_dates([pd.Timestamp('2023-01-01')])


def test_rebalance_freq_invalid_string():
    """测试无效的字符串调仓频率"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    with pytest.raises(ValueError, match="不支持的调仓频率"):
        engine = BacktestEngine(
            universe=universe,
            signal=signal,
            rebalance_freq="X",
            verbose=False
        )
        engine._get_rebalance_dates([pd.Timestamp('2023-01-01')])


def test_holding_period_auto_integer():
    """测试整数调仓频率时自动设置持有期"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        rebalance_freq=7,  # 每7天调仓
        verbose=False
    )
    
    # 持有期应该等于调仓频率
    assert engine.holding_period == 7


def test_holding_period_manual_override():
    """测试手动设置持有期覆盖自动计算"""
    universe = MockUniverse(['000001.SZ'])
    signal = MockSignal()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        rebalance_freq=5,
        holding_period=10,  # 手动设置持有期
        verbose=False
    )
    
    # 持有期应该使用手动设置的值
    assert engine.holding_period == 10
