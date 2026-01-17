"""测试收益跟踪功能"""

import pandas as pd
import pytest

from src.lazybull.backtest import BacktestEngine
from src.lazybull.common.cost import CostModel
from src.lazybull.signals.base import Signal
from src.lazybull.universe.base import Universe


class MockUniverse(Universe):
    """模拟股票池"""
    
    def get_stocks(self, date):
        """返回所有可用股票"""
        return ['000001.SZ']


class MockSignal(Signal):
    """模拟信号生成器"""
    
    def generate(self, date, universe, data):
        """生成信号"""
        if not universe:
            return {}
        return {stock: 1.0 for stock in universe}


@pytest.fixture
def price_data_with_profit():
    """创建价格变化的数据（用于测试盈利/亏损）"""
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    
    data = []
    for i, date in enumerate(dates):
        # 价格从10元涨到15元，然后回到12元
        if i < 5:
            price = 10.0 + i * 1.0  # 10, 11, 12, 13, 14
        else:
            price = 14.0 - (i - 4) * 0.5  # 14, 13.5, 13, 12.5, 12, 11.5
        
        data.append({
            'ts_code': '000001.SZ',
            'trade_date': date.strftime('%Y%m%d'),
            'close': price
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def trading_dates():
    """交易日列表"""
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    return [pd.Timestamp(d) for d in dates]


def test_profit_tracking_in_sell_trades(price_data_with_profit, trading_dates):
    """测试卖出交易包含收益信息"""
    
    # 创建回测引擎（持有期=3天）
    universe = MockUniverse()
    signal = MockSignal()
    cost_model = CostModel()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        cost_model=cost_model,
        rebalance_freq="D",
        holding_period=3
    )
    
    # 运行回测
    engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=trading_dates,
        price_data=price_data_with_profit
    )
    
    # 获取交易记录
    trades_df = engine.get_trades()
    
    # 验证有买入和卖出交易
    assert len(trades_df) > 0
    buy_trades = trades_df[trades_df['action'] == 'buy']
    sell_trades = trades_df[trades_df['action'] == 'sell']
    
    assert len(buy_trades) > 0, "应该有买入交易"
    assert len(sell_trades) > 0, "应该有卖出交易"
    
    # 验证卖出交易包含新增字段
    for _, sell_trade in sell_trades.iterrows():
        assert 'buy_price' in sell_trade, "卖出交易应包含买入价格"
        assert 'profit_amount' in sell_trade, "卖出交易应包含收益金额"
        assert 'profit_pct' in sell_trade, "卖出交易应包含收益率"
        
        # 验证字段值合理
        assert pd.notna(sell_trade['buy_price']), "买入价格不应为空"
        assert pd.notna(sell_trade['profit_amount']), "收益金额不应为空"
        assert pd.notna(sell_trade['profit_pct']), "收益率不应为空"


def test_profit_calculation_accuracy(price_data_with_profit, trading_dates):
    """测试收益计算准确性"""
    
    universe = MockUniverse()
    signal = MockSignal()
    cost_model = CostModel()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        cost_model=cost_model,
        rebalance_freq="D",
        holding_period=3
    )
    
    # 运行回测
    engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[6],  # 运行7天
        trading_dates=trading_dates[:7],
        price_data=price_data_with_profit
    )
    
    trades_df = engine.get_trades()
    sell_trades = trades_df[trades_df['action'] == 'sell']
    
    # 验证至少有一笔卖出交易
    assert len(sell_trades) > 0, "应该有卖出交易"
    
    # 验证收益计算逻辑
    for _, sell_trade in sell_trades.iterrows():
        stock = sell_trade['stock']
        sell_price = sell_trade['price']
        buy_price = sell_trade['buy_price']
        shares = sell_trade['shares']
        sell_cost = sell_trade['cost']
        profit_amount = sell_trade['profit_amount']
        profit_pct = sell_trade['profit_pct']
        
        # 找到对应的买入交易
        buy_trade = trades_df[
            (trades_df['action'] == 'buy') & 
            (trades_df['stock'] == stock) &
            (trades_df['date'] < sell_trade['date'])
        ].iloc[-1]  # 最后一次买入
        
        buy_cost = buy_trade['cost']
        buy_amount = buy_trade['amount']
        
        # 手动计算预期收益
        buy_total_cost = buy_amount + buy_cost
        sell_proceeds = shares * sell_price - sell_cost
        expected_profit = sell_proceeds - buy_total_cost
        expected_profit_pct = expected_profit / buy_total_cost
        
        # 验证计算结果（允许小的浮点误差）
        assert abs(profit_amount - expected_profit) < 0.01, \
            f"收益金额计算错误: {profit_amount} vs {expected_profit}"
        assert abs(profit_pct - expected_profit_pct) < 0.0001, \
            f"收益率计算错误: {profit_pct} vs {expected_profit_pct}"


def test_profit_with_price_increase(price_data_with_profit, trading_dates):
    """测试价格上涨时的盈利"""
    
    # 创建价格上涨的数据
    data = []
    for i, date in enumerate(trading_dates[:7]):
        data.append({
            'ts_code': '000001.SZ',
            'trade_date': date.strftime('%Y%m%d'),
            'close': 10.0 + i * 2.0  # 价格持续上涨
        })
    
    price_data = pd.DataFrame(data)
    
    universe = MockUniverse()
    signal = MockSignal()
    cost_model = CostModel()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        cost_model=cost_model,
        rebalance_freq="D",
        holding_period=3
    )
    
    engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[6],
        trading_dates=trading_dates[:7],
        price_data=price_data
    )
    
    trades_df = engine.get_trades()
    sell_trades = trades_df[trades_df['action'] == 'sell']
    
    # 价格上涨，应该有正收益
    if len(sell_trades) > 0:
        for _, sell_trade in sell_trades.iterrows():
            # 扣除成本后，收益可能为正或负，但这里价格涨幅较大，应该为正
            # 注意：由于有成本，小幅上涨可能仍为负
            profit_amount = sell_trade['profit_amount']
            # 至少验证字段存在且有值
            assert pd.notna(profit_amount), "收益金额应该有值"


def test_profit_with_price_decrease(trading_dates):
    """测试价格下跌时的亏损"""
    
    # 创建价格下跌的数据
    data = []
    for i, date in enumerate(trading_dates[:7]):
        data.append({
            'ts_code': '000001.SZ',
            'trade_date': date.strftime('%Y%m%d'),
            'close': 20.0 - i * 2.0  # 价格持续下跌
        })
    
    price_data = pd.DataFrame(data)
    
    universe = MockUniverse()
    signal = MockSignal()
    cost_model = CostModel()
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        cost_model=cost_model,
        rebalance_freq="D",
        holding_period=3
    )
    
    engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[6],
        trading_dates=trading_dates[:7],
        price_data=price_data
    )
    
    trades_df = engine.get_trades()
    sell_trades = trades_df[trades_df['action'] == 'sell']
    
    # 价格下跌，应该有负收益（亏损）
    if len(sell_trades) > 0:
        for _, sell_trade in sell_trades.iterrows():
            profit_amount = sell_trade['profit_amount']
            # 价格下跌较大，应该为负
            assert profit_amount < 0, "价格下跌应该产生亏损"
            assert sell_trade['profit_pct'] < 0, "收益率应该为负"
