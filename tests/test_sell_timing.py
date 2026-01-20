"""测试卖出时机配置功能（T+1 开盘 vs T+1 收盘）"""

import pandas as pd
import pytest

from src.lazybull.backtest import BacktestEngine
from src.lazybull.common.cost import CostModel
from src.lazybull.signals.base import Signal
from src.lazybull.universe.base import Universe


class MockUniverse(Universe):
    """模拟股票池"""
    
    def get_stocks(self, date, quote_data=None):
        """返回所有可用股票"""
        return ['000001.SZ', '000002.SZ']


class MockSignal(Signal):
    """模拟信号生成器（等权）"""
    
    def __init__(self, top_n=2):
        self.top_n = top_n
        self.weight_method = 'equal'
    
    def generate(self, date, universe, data):
        """生成等权信号"""
        if not universe:
            return {}
        return {stock: 1.0 / len(universe) for stock in universe}
    
    def generate_ranked(self, date, universe, data):
        """生成排序候选列表（为了兼容新的信号接口）"""
        if not universe:
            return []
        # 返回排序后的候选列表：[(stock, score), ...]
        return [(stock, 1.0) for stock in universe]


@pytest.fixture
def mock_price_data_with_open():
    """模拟包含开盘价的价格数据"""
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ']
    
    data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            # 开盘价略低于收盘价（模拟日内上涨）
            open_price = 10.0 + i * 0.1
            close_price = 10.0 + i * 0.1 + 0.05
            
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'open': open_price,
                'close': close_price,
                'open_adj': open_price * 1.1,  # 模拟复权后价格
                'close_adj': close_price * 1.1
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_price_data_missing_open():
    """模拟缺少部分开盘价的价格数据（测试降级策略）"""
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ']
    
    data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            close_price = 10.0 + i * 0.1
            
            # 所有记录都有基本字段
            record = {
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': close_price,
                'close_adj': close_price * 1.1
            }
            
            # 前5天有开盘价，后5天没有开盘价（但保留close/close_adj）
            if i < 5:
                record['open'] = close_price - 0.05
                record['open_adj'] = (close_price - 0.05) * 1.1
            else:
                # 后5天缺少开盘价，但仍保留close和close_adj
                # open字段不存在（模拟缺失数据）
                pass
            
            data.append(record)
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_trading_dates():
    """模拟交易日列表"""
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    return [pd.Timestamp(d) for d in dates]


def test_default_sell_timing_close(mock_price_data_with_open, mock_trading_dates):
    """测试默认卖出时机为收盘价"""
    
    # 创建回测引擎（不指定 sell_timing，应默认为 'close'）
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        cost_model=CostModel(),
        rebalance_freq=5,  # 每5天调仓
        holding_period=2,  # 持有2天
        verbose=False,
        enable_pending_order=False  # 禁用延迟订单以简化测试
    )
    
    # 验证默认值
    assert engine.sell_timing == 'close'
    
    # 运行回测
    nav_curve = engine.run(
        start_date=mock_trading_dates[0],
        end_date=mock_trading_dates[-1],
        trading_dates=mock_trading_dates,
        price_data=mock_price_data_with_open
    )
    
    # 获取交易记录
    trades_df = engine.get_trades()
    
    # 验证卖出交易都使用收盘价
    sell_trades = trades_df[trades_df['action'] == 'sell']
    if len(sell_trades) > 0:
        for _, trade in sell_trades.iterrows():
            # 验证记录中的 sell_timing 字段
            assert trade['sell_timing'] == 'close'
            
            # 验证价格是收盘价（带有小数部分 0.05）
            # 收盘价格应该是 base + index * 0.1 + 0.05
            date = trade['date']
            date_idx = mock_trading_dates.index(date)
            expected_close_price = 10.0 + date_idx * 0.1 + 0.05
            
            # 允许小的浮点误差
            assert abs(trade['price'] - expected_close_price) < 0.01


def test_sell_timing_open(mock_price_data_with_open, mock_trading_dates):
    """测试配置为开盘价卖出"""
    
    # 创建回测引擎，指定 sell_timing='open'
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        cost_model=CostModel(),
        rebalance_freq=5,
        holding_period=2,
        verbose=False,
        sell_timing='open',  # 使用开盘价卖出
        enable_pending_order=False
    )
    
    # 验证配置
    assert engine.sell_timing == 'open'
    
    # 运行回测
    nav_curve = engine.run(
        start_date=mock_trading_dates[0],
        end_date=mock_trading_dates[-1],
        trading_dates=mock_trading_dates,
        price_data=mock_price_data_with_open
    )
    
    # 获取交易记录
    trades_df = engine.get_trades()
    
    # 验证卖出交易都使用开盘价
    sell_trades = trades_df[trades_df['action'] == 'sell']
    if len(sell_trades) > 0:
        for _, trade in sell_trades.iterrows():
            # 验证记录中的 sell_timing 字段
            assert trade['sell_timing'] == 'open'
            
            # 验证价格是开盘价（不带小数部分 0.05）
            # 开盘价格应该是 base + index * 0.1
            date = trade['date']
            date_idx = mock_trading_dates.index(date)
            expected_open_price = 10.0 + date_idx * 0.1
            
            # 允许小的浮点误差
            assert abs(trade['price'] - expected_open_price) < 0.01


def test_sell_timing_open_fallback_to_close(mock_price_data_missing_open, mock_trading_dates):
    """测试开盘价缺失时降级到收盘价"""
    
    # 创建回测引擎，指定 sell_timing='open'
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        cost_model=CostModel(),
        rebalance_freq=3,
        holding_period=2,
        verbose=True,  # 开启日志以查看降级警告
        sell_timing='open',
        enable_pending_order=False
    )
    
    # 运行回测
    nav_curve = engine.run(
        start_date=mock_trading_dates[0],
        end_date=mock_trading_dates[-1],
        trading_dates=mock_trading_dates,
        price_data=mock_price_data_missing_open
    )
    
    # 获取交易记录
    trades_df = engine.get_trades()
    
    # 验证卖出交易
    sell_trades = trades_df[trades_df['action'] == 'sell']
    if len(sell_trades) > 0:
        for _, trade in sell_trades.iterrows():
            # 所有卖出都应该标记为 'open'
            assert trade['sell_timing'] == 'open'
            
            # 价格应该存在（降级到收盘价后）
            assert trade['price'] > 0
            
            # 如果卖出日期在后5天（没有开盘价），价格应该是收盘价
            date = trade['date']
            date_idx = mock_trading_dates.index(date)
            
            if date_idx >= 5:
                # 没有开盘价，应该使用收盘价
                expected_close_price = 10.0 + date_idx * 0.1
                assert abs(trade['price'] - expected_close_price) < 0.01
            else:
                # 有开盘价，应该使用开盘价
                expected_open_price = 10.0 + date_idx * 0.1 - 0.05
                assert abs(trade['price'] - expected_open_price) < 0.01


def test_invalid_sell_timing_parameter():
    """测试无效的 sell_timing 参数"""
    
    universe = MockUniverse()
    signal = MockSignal()
    
    # 尝试使用无效的 sell_timing 值
    with pytest.raises(ValueError) as excinfo:
        engine = BacktestEngine(
            universe=universe,
            signal=signal,
            initial_capital=100000,
            sell_timing='invalid'  # 无效值
        )
    
    # 验证错误消息
    assert "卖出时机参数必须为 'close' 或 'open'" in str(excinfo.value)


def test_sell_timing_performance_comparison(mock_price_data_with_open, mock_trading_dates):
    """对比开盘价卖出和收盘价卖出的性能差异"""
    
    universe = MockUniverse()
    
    # 创建两个引擎进行对比
    engine_close = BacktestEngine(
        universe=universe,
        signal=MockSignal(),
        initial_capital=100000,
        cost_model=CostModel(),
        rebalance_freq=5,
        holding_period=2,
        verbose=False,
        sell_timing='close',
        enable_pending_order=False
    )
    
    engine_open = BacktestEngine(
        universe=universe,
        signal=MockSignal(),
        initial_capital=100000,
        cost_model=CostModel(),
        rebalance_freq=5,
        holding_period=2,
        verbose=False,
        sell_timing='open',
        enable_pending_order=False
    )
    
    # 运行两个回测
    nav_close = engine_close.run(
        start_date=mock_trading_dates[0],
        end_date=mock_trading_dates[-1],
        trading_dates=mock_trading_dates,
        price_data=mock_price_data_with_open
    )
    
    nav_open = engine_open.run(
        start_date=mock_trading_dates[0],
        end_date=mock_trading_dates[-1],
        trading_dates=mock_trading_dates,
        price_data=mock_price_data_with_open
    )
    
    # 验证两个回测都成功完成
    assert len(nav_close) == len(nav_open)
    assert len(nav_close) == len(mock_trading_dates)
    
    # 获取最终净值
    final_nav_close = nav_close['nav'].iloc[-1]
    final_nav_open = nav_open['nav'].iloc[-1]
    
    # 两个净值应该不同（因为开盘价和收盘价不同）
    # 由于开盘价低于收盘价，开盘价卖出应该获得略低的收益
    # 但由于价格差异很小且有成本，实际结果可能相近
    assert final_nav_close > 0
    assert final_nav_open > 0
    
    # 获取交易记录并验证
    trades_close = engine_close.get_trades()
    trades_open = engine_open.get_trades()
    
    # 卖出次数应该相同
    sells_close = trades_close[trades_close['action'] == 'sell']
    sells_open = trades_open[trades_open['action'] == 'sell']
    
    assert len(sells_close) == len(sells_open)
    
    # 卖出价格应该不同（开盘价 < 收盘价）
    if len(sells_close) > 0 and len(sells_open) > 0:
        avg_sell_price_close = sells_close['price'].mean()
        avg_sell_price_open = sells_open['price'].mean()
        
        # 开盘价应该低于收盘价
        assert avg_sell_price_open < avg_sell_price_close
