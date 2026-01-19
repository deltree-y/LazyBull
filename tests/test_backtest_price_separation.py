"""测试价格口径分离和风险预算功能"""

import pandas as pd
import pytest
import numpy as np

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
def mock_price_data_with_adj():
    """模拟价格数据（包含不复权和后复权价格）"""
    dates = pd.date_range('2023-01-01', periods=30, freq='B')
    stocks = ['000001.SZ', '000002.SZ']
    
    data = []
    for i, date in enumerate(dates):
        for j, stock in enumerate(stocks):
            # 不复权价格保持固定
            close = 10.0
            # 后复权价格模拟上涨趋势（用于测试收益率计算）
            close_adj = 10.0 + i * 0.1 + j * 0.05
            
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': close,
                'close_adj': close_adj
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_trading_dates_30():
    """模拟30个交易日列表"""
    dates = pd.date_range('2023-01-01', periods=30, freq='B')
    return [pd.Timestamp(d) for d in dates]


def test_price_index_creation(mock_price_data_with_adj, mock_trading_dates_30):
    """测试价格索引创建"""
    
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        holding_period=5
    )
    
    # 运行回测以初始化价格索引
    engine.run(
        start_date=mock_trading_dates_30[0],
        end_date=mock_trading_dates_30[9],
        trading_dates=mock_trading_dates_30[:10],
        price_data=mock_price_data_with_adj
    )
    
    # 验证价格索引已创建
    assert engine.trade_price_index is not None
    assert engine.pnl_price_index is not None
    
    # 验证可以访问价格
    date = mock_trading_dates_30[0]
    stock = '000001.SZ'
    
    trade_price = engine._get_trade_price(date, stock)
    pnl_price = engine._get_pnl_price(date, stock)
    
    assert trade_price is not None
    assert pnl_price is not None
    assert trade_price == 10.0  # 不复权价格
    assert pnl_price == 10.0  # 第一天的后复权价格


def test_trade_records_with_pnl_fields(mock_price_data_with_adj, mock_trading_dates_30):
    """测试交易记录包含绩效价格字段"""
    
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        rebalance_freq="D",  # 每日调仓以产生更多交易
        holding_period=5
    )
    
    # 运行回测
    engine.run(
        start_date=mock_trading_dates_30[0],
        end_date=mock_trading_dates_30[19],
        trading_dates=mock_trading_dates_30[:20],
        price_data=mock_price_data_with_adj
    )
    
    # 获取交易记录
    trades_df = engine.get_trades()
    
    # 验证有交易记录
    assert len(trades_df) > 0, "应该有交易记录"
    
    # 验证卖出记录包含绩效价格字段
    sell_trades = trades_df[trades_df['action'] == 'sell']
    
    if len(sell_trades) > 0:
        for _, trade in sell_trades.iterrows():
            assert 'buy_pnl_price' in trade
            assert 'sell_pnl_price' in trade
            assert 'pnl_profit_amount' in trade
            assert 'pnl_profit_pct' in trade
            
            # 验证绩效价格与成交价格不同（因为有趋势）
            # 后复权价格应该更高（模拟上涨）
            assert trade['sell_pnl_price'] > trade['buy_pnl_price']


def test_risk_budget_enabled(mock_price_data_with_adj, mock_trading_dates_30):
    """测试风险预算启用"""
    
    universe = MockUniverse()
    signal = MockSignal()
    
    # 启用风险预算
    engine_with_rb = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        rebalance_freq="D",  # 每日调仓以产生更多交易
        holding_period=5,
        enable_risk_budget=True,
        vol_window=10
    )
    
    # 禁用风险预算作为对照
    engine_without_rb = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        rebalance_freq="D",  # 每日调仓
        holding_period=5,
        enable_risk_budget=False
    )
    
    # 运行两次回测
    nav_with_rb = engine_with_rb.run(
        start_date=mock_trading_dates_30[0],
        end_date=mock_trading_dates_30[19],
        trading_dates=mock_trading_dates_30[:20],
        price_data=mock_price_data_with_adj
    )
    
    nav_without_rb = engine_without_rb.run(
        start_date=mock_trading_dates_30[0],
        end_date=mock_trading_dates_30[19],
        trading_dates=mock_trading_dates_30[:20],
        price_data=mock_price_data_with_adj
    )
    
    # 验证两者都产生了净值曲线
    assert len(nav_with_rb) > 0
    assert len(nav_without_rb) > 0
    
    # 验证交易数量（启用风险预算可能改变权重，但不应该改变交易数量太多）
    trades_with_rb = engine_with_rb.get_trades()
    trades_without_rb = engine_without_rb.get_trades()
    
    # 至少应该有一些交易
    assert len(trades_with_rb) > 0
    assert len(trades_without_rb) > 0


def test_volatility_calculation(mock_price_data_with_adj, mock_trading_dates_30):
    """测试波动率计算"""
    
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        holding_period=5,
        vol_window=10
    )
    
    # 初始化价格索引
    engine._prepare_price_index(mock_price_data_with_adj)
    
    # 计算波动率
    stock = '000001.SZ'
    end_date = mock_trading_dates_30[15]
    
    vol = engine._calculate_volatility(stock, end_date)
    
    # 验证波动率是合理的数值
    assert vol > 0
    assert vol >= engine.vol_epsilon  # 应该不低于最小值
    assert vol < 10.0  # 应该是合理范围内


def test_fallback_to_close_when_no_adj(mock_trading_dates_30):
    """测试缺少 close_adj 时回退到 close"""
    
    # 创建只有 close 列的价格数据
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ']
    
    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': 10.0
            })
    
    price_data = pd.DataFrame(data)
    
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        holding_period=2
    )
    
    # 运行回测（应该使用 close 作为绩效价格）
    nav_curve = engine.run(
        start_date=mock_trading_dates_30[0],
        end_date=mock_trading_dates_30[9],
        trading_dates=mock_trading_dates_30[:10],
        price_data=price_data
    )
    
    # 验证回测成功运行
    assert len(nav_curve) == 10
    
    # 验证价格索引相同（退化到 close）
    date = mock_trading_dates_30[1]
    stock = '000001.SZ'
    
    trade_price = engine._get_trade_price(date, stock)
    pnl_price = engine._get_pnl_price(date, stock)
    
    if trade_price is not None and pnl_price is not None:
        assert trade_price == pnl_price  # 应该相同


def test_position_structure_with_new_fields(mock_price_data_with_adj, mock_trading_dates_30):
    """测试持仓结构包含新字段"""
    
    universe = MockUniverse()
    signal = MockSignal()
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        holding_period=5
    )
    
    # 运行部分回测
    engine.run(
        start_date=mock_trading_dates_30[0],
        end_date=mock_trading_dates_30[4],
        trading_dates=mock_trading_dates_30[:5],
        price_data=mock_price_data_with_adj
    )
    
    # 验证持仓结构
    for stock, info in engine.positions.items():
        assert 'shares' in info
        assert 'buy_date' in info
        assert 'buy_trade_price' in info
        assert 'buy_pnl_price' in info
        assert 'buy_cost_cash' in info
        
        # 验证数值类型
        assert isinstance(info['shares'], (int, float))
        assert isinstance(info['buy_date'], pd.Timestamp)
        assert isinstance(info['buy_trade_price'], (int, float))
        assert isinstance(info['buy_pnl_price'], (int, float))
        assert isinstance(info['buy_cost_cash'], (int, float))
        
        # 验证成交价格和绩效价格都是正数
        assert info['buy_trade_price'] > 0
        assert info['buy_pnl_price'] > 0
        assert info['buy_cost_cash'] > 0
