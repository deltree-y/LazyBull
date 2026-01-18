"""测试 T+1 交易规则的回测引擎"""

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
    
    def generate(self, date, universe, data):
        """生成等权信号"""
        if not universe:
            return {}
        return {stock: 1.0 / len(universe) for stock in universe}


@pytest.fixture
def mock_price_data():
    """模拟价格数据"""
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ']
    
    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': 10.0  # 固定价格，简化测试
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_trading_dates():
    """模拟交易日列表"""
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    return [pd.Timestamp(d) for d in dates]


def test_t1_trading_logic(mock_price_data, mock_trading_dates):
    """测试 T+1 买入、T+n 卖出逻辑"""
    
    # 创建回测引擎（持有期设为2天）
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        cost_model=CostModel(),
        rebalance_freq="D",  # 每日调仓
        holding_period=2  # 持有2天
    )
    
    # 运行回测
    nav_curve = engine.run(
        start_date=mock_trading_dates[0],
        end_date=mock_trading_dates[-1],
        trading_dates=mock_trading_dates,
        price_data=mock_price_data
    )
    
    # 验证净值曲线生成
    assert len(nav_curve) == len(mock_trading_dates)
    assert 'nav' in nav_curve.columns
    assert 'portfolio_value' in nav_curve.columns
    
    # 获取交易记录
    trades_df = engine.get_trades()
    
    # 验证交易记录
    assert len(trades_df) > 0
    assert 'date' in trades_df.columns
    assert 'action' in trades_df.columns
    assert 'price' in trades_df.columns
    
    # 验证 T+1 买入逻辑：
    # 如果第0天生成信号，应该在第1天买入
    buy_trades = trades_df[trades_df['action'] == 'buy']
    if len(buy_trades) > 0:
        first_buy_date = buy_trades['date'].min()
        # 第一笔买入应该在第1天或之后（不会在第0天）
        assert first_buy_date > mock_trading_dates[0]
    
    # 验证持有期逻辑：
    # 买入后应该在 T+2 天卖出
    sell_trades = trades_df[trades_df['action'] == 'sell']
    if len(sell_trades) > 0 and len(buy_trades) > 0:
        # 每笔卖出应该在对应买入之后至少2天
        for _, sell_trade in sell_trades.iterrows():
            stock = sell_trade['stock']
            sell_date = sell_trade['date']
            
            # 找到该股票最近一次买入
            stock_buys = buy_trades[
                (buy_trades['stock'] == stock) & 
                (buy_trades['date'] < sell_date)
            ]
            
            if len(stock_buys) > 0:
                last_buy_date = stock_buys['date'].max()
                buy_idx = mock_trading_dates.index(last_buy_date)
                sell_idx = mock_trading_dates.index(sell_date)
                holding_days = sell_idx - buy_idx
                
                # 持有期应该至少为2天
                assert holding_days >= 2, f"持有期不足：{holding_days}天"


def test_pending_signals_mechanism(mock_price_data, mock_trading_dates):
    """测试信号待执行机制"""
    
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        holding_period=3
    )
    
    # 运行回测
    engine.run(
        start_date=mock_trading_dates[0],
        end_date=mock_trading_dates[4],  # 只运行前5天
        trading_dates=mock_trading_dates[:5],
        price_data=mock_price_data
    )
    
    # 验证信号生成和执行的分离
    trades_df = engine.get_trades()
    
    if len(trades_df) > 0:
        buy_trades = trades_df[trades_df['action'] == 'buy']
        
        # 验证买入发生在信号生成的次日
        for _, trade in buy_trades.iterrows():
            buy_date = trade['date']
            buy_idx = mock_trading_dates.index(buy_date)
            
            # 买入不会发生在第0天（因为需要等待T+1）
            assert buy_idx > 0


def test_position_tracking_with_buy_date(mock_price_data, mock_trading_dates):
    """测试持仓跟踪（包含买入日期、价格和成本）"""
    
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        holding_period=2
    )
    
    # 运行部分回测
    engine.run(
        start_date=mock_trading_dates[0],
        end_date=mock_trading_dates[3],
        trading_dates=mock_trading_dates[:4],
        price_data=mock_price_data
    )
    
    # 验证持仓结构
    # 注意：持仓结构现在是 {股票: {shares, buy_date, buy_trade_price, buy_pnl_price, buy_cost_cash}}
    for stock, info in engine.positions.items():
        assert 'shares' in info
        assert 'buy_date' in info
        assert 'buy_trade_price' in info
        assert 'buy_pnl_price' in info
        assert 'buy_cost_cash' in info
        assert isinstance(info['shares'], (int, float))
        assert isinstance(info['buy_date'], pd.Timestamp)
        assert isinstance(info['buy_trade_price'], (int, float))
        assert isinstance(info['buy_pnl_price'], (int, float))
        assert isinstance(info['buy_cost_cash'], (int, float))
