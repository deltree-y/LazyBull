"""测试 T+1 交易规则的回测引擎"""

import io

import pandas as pd
import pytest
from loguru import logger

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

    def generate_ranked(self, date, universe, data):
        """生成排序候选列表。"""
        if not universe:
            return []
        return [(stock, float(len(universe) - idx)) for idx, stock in enumerate(universe)]


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
        rebalance_freq=1,  # 每日调仓
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
    # 买入后至少要满持有期，并在后续交易日卖出
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


def test_holding_period_sell_uses_t0_signal_and_t1_execution():
    """持有期到期应在 T0 生成卖出信号，并于下一交易日执行。"""

    trading_dates = [pd.Timestamp(d) for d in pd.date_range('2023-01-02', periods=6, freq='B')]
    date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
    price_data = pd.DataFrame(
        {
            'ts_code': ['000001.SZ'] * len(trading_dates),
            'trade_date': [date.strftime('%Y%m%d') for date in trading_dates],
            'close': [10.0, 10.0, 10.5, 11.0, 11.5, 12.0],
        }
    )

    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(),
        initial_capital=100000,
        cost_model=CostModel(),
        rebalance_freq=1,
        holding_period=2,
        enable_pending_order=False,
        enable_position_completion=False,
        verbose=False,
    )
    engine.price_data_cache = price_data.copy()
    engine._prepare_price_index(price_data)
    engine.positions = {
        '000001.SZ': {
            'shares': 100,
            'buy_date': trading_dates[1],
            'buy_trade_price': 10.0,
            'buy_pnl_price': 10.0,
            'buy_cost_cash': 0.0,
        }
    }

    engine._check_and_sell(
        date=trading_dates[3],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    assert '000001.SZ' in engine.pending_condition_sells
    assert engine.pending_condition_sells['000001.SZ']['trigger_date'] == trading_dates[3]
    assert engine.pending_condition_sells['000001.SZ']['sell_type'] == 'holding_period'
    assert engine.get_trades().empty

    engine._execute_pending_condition_sells(
        date=trading_dates[4],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    trades = engine.get_trades()
    assert len(trades) == 1
    assert trades.iloc[0]['action'] == 'sell'
    assert trades.iloc[0]['date'] == trading_dates[4]
    assert trades.iloc[0]['sell_type'] == 'holding_period'


def test_holding_period_sell_generates_refill_buy_on_t1_without_profit_mode():
    """未启用盈亏动态持仓时，持有期卖出也应在下一交易日先卖后补位买入。"""

    trading_dates = [pd.Timestamp(d) for d in pd.date_range('2023-01-02', periods=6, freq='B')]
    date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
    price_data = pd.DataFrame(
        [
            {
                'ts_code': '000001.SZ',
                'trade_date': date.strftime('%Y%m%d'),
                'close': close_1,
                'open': close_1,
                'close_adj': close_1,
                'open_adj': close_1,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            }
            for date, close_1 in zip(trading_dates, [10.0, 10.0, 10.5, 11.0, 11.5, 12.0])
        ]
        + [
            {
                'ts_code': '000002.SZ',
                'trade_date': date.strftime('%Y%m%d'),
                'close': 20.0,
                'open': 20.0,
                'close_adj': 20.0,
                'open_adj': 20.0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            }
            for date in trading_dates
        ]
    )

    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(),
        initial_capital=100000,
        cost_model=CostModel(commission_rate=0, min_commission=0, stamp_tax=0, slippage=0),
        rebalance_freq=20,
        holding_period=2,
        enable_pending_order=False,
        enable_position_completion=True,
        verbose=False,
    )
    engine.price_data_cache = price_data.copy()
    engine._prepare_price_index(price_data)
    engine.current_capital = 0.0
    engine.positions = {
        '000001.SZ': {
            'shares': 1000,
            'buy_date': trading_dates[1],
            'signal_date': trading_dates[1],
            'buy_trade_price': 10.0,
            'buy_pnl_price': 10.0,
            'buy_cost_cash': 0.0,
        }
    }

    engine._check_and_sell(
        date=trading_dates[3],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    assert '000001.SZ' in engine.pending_condition_sells
    assert trading_dates[3] in engine.pending_signals
    assert engine.pending_signals[trading_dates[3]]['desired_position_count'] == 1

    engine._execute_pending_condition_sells(
        date=trading_dates[4],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )
    engine._execute_pending_buys(
        date=trading_dates[4],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    trades = engine.get_trades()
    assert len(trades) == 2
    assert list(trades['action']) == ['sell', 'buy']
    assert trades.iloc[0]['stock'] == '000001.SZ'
    assert trades.iloc[1]['stock'] == '000002.SZ'
    assert '000001.SZ' not in engine.positions
    assert '000002.SZ' in engine.positions


def test_profit_extension_rejection_uses_t0_signal_and_t1_execution():
    """盈利延续未通过时，应在 T0 生成卖出信号，并于下一交易日执行。"""

    trading_dates = [pd.Timestamp(d) for d in pd.date_range('2023-01-02', periods=6, freq='B')]
    date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
    price_data = pd.DataFrame(
        {
            'ts_code': ['000001.SZ'] * len(trading_dates),
            'trade_date': [date.strftime('%Y%m%d') for date in trading_dates],
            'close': [10.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(),
        initial_capital=100000,
        cost_model=CostModel(),
        rebalance_freq=1,
        holding_period=2,
        enable_pending_order=False,
        enable_position_completion=False,
        verbose=False,
    )
    engine.price_data_cache = price_data.copy()
    engine._prepare_price_index(price_data)
    engine.positions = {
        '000001.SZ': {
            'shares': 100,
            'buy_date': trading_dates[1],
            'buy_trade_price': 10.0,
            'buy_pnl_price': 10.0,
            'buy_cost_cash': 0.0,
        }
    }

    engine._check_and_sell(
        date=trading_dates[3],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    assert '000001.SZ' in engine.pending_condition_sells
    assert engine.pending_condition_sells['000001.SZ']['trigger_date'] == trading_dates[3]
    assert engine.get_trades().empty

    engine._execute_pending_condition_sells(
        date=trading_dates[4],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    trades = engine.get_trades()
    assert len(trades) == 1
    assert trades.iloc[0]['action'] == 'sell'
    assert trades.iloc[0]['date'] == trading_dates[4]


def test_profit_extension_rejection_generates_refill_buy_on_t1():
    """盈利延续未通过时，应在下一交易日先卖再按 T0 候选补回空槽。"""

    trading_dates = [pd.Timestamp(d) for d in pd.date_range('2023-01-02', periods=6, freq='B')]
    date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
    price_data = pd.DataFrame(
        [
            {
                'ts_code': '000001.SZ',
                'trade_date': date.strftime('%Y%m%d'),
                'close': close_1,
                'open': close_1,
                'close_adj': close_1,
                'open_adj': close_1,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            }
            for date, close_1 in zip(trading_dates, [10.0, 10.0, 11.0, 12.0, 13.0, 14.0])
        ]
        + [
            {
                'ts_code': '000002.SZ',
                'trade_date': date.strftime('%Y%m%d'),
                'close': 20.0,
                'open': 20.0,
                'close_adj': 20.0,
                'open_adj': 20.0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            }
            for date in trading_dates
        ]
    )

    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(),
        initial_capital=100000,
        cost_model=CostModel(commission_rate=0, min_commission=0, stamp_tax=0, slippage=0),
        rebalance_freq=20,
        holding_period=2,
        enable_pending_order=False,
        enable_position_completion=True,
        verbose=False,
    )
    engine.price_data_cache = price_data.copy()
    engine._prepare_price_index(price_data)
    engine.current_capital = 0.0
    engine.positions = {
        '000001.SZ': {
            'shares': 1000,
            'buy_date': trading_dates[1],
            'signal_date': trading_dates[1],
            'buy_trade_price': 10.0,
            'buy_pnl_price': 10.0,
            'buy_cost_cash': 0.0,
        }
    }

    engine._check_and_sell(
        date=trading_dates[3],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    assert '000001.SZ' in engine.pending_condition_sells
    assert trading_dates[3] in engine.pending_signals
    assert engine.pending_signals[trading_dates[3]]['desired_position_count'] == 1

    engine._execute_pending_condition_sells(
        date=trading_dates[4],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )
    engine._execute_pending_buys(
        date=trading_dates[4],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    trades = engine.get_trades()
    assert len(trades) == 2
    assert list(trades['action']) == ['sell', 'buy']
    assert trades.iloc[0]['stock'] == '000001.SZ'
    assert trades.iloc[1]['stock'] == '000002.SZ'
    assert '000001.SZ' not in engine.positions
    assert '000002.SZ' in engine.positions


def test_profit_extension_expiry_uses_t0_signal_and_t1_execution():
    """盈利延续到期后，应在到期日生成卖出信号，并于下一交易日执行。"""

    trading_dates = [pd.Timestamp(d) for d in pd.date_range('2023-01-02', periods=7, freq='B')]
    date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
    price_data = pd.DataFrame(
        {
            'ts_code': ['000001.SZ'] * len(trading_dates),
            'trade_date': [date.strftime('%Y%m%d') for date in trading_dates],
            'close': [10.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )

    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(),
        initial_capital=100000,
        cost_model=CostModel(),
        rebalance_freq=1,
        holding_period=2,
        enable_pending_order=False,
        enable_position_completion=False,
        verbose=False,
    )
    engine._prepare_price_index(price_data)
    engine.positions = {
        '000001.SZ': {
            'shares': 100,
            'buy_date': trading_dates[1],
            'buy_trade_price': 10.0,
            'buy_pnl_price': 10.0,
            'buy_cost_cash': 0.0,
        }
    }

    engine._check_and_sell(
        date=trading_dates[4],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    assert '000001.SZ' in engine.pending_condition_sells
    assert engine.pending_condition_sells['000001.SZ']['trigger_date'] == trading_dates[4]
    assert engine.get_trades().empty

    engine._execute_pending_condition_sells(
        date=trading_dates[5],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    trades = engine.get_trades()
    assert len(trades) == 1
    assert trades.iloc[0]['action'] == 'sell'
    assert trades.iloc[0]['date'] == trading_dates[5]


def test_pending_buys_use_t0_priority_candidates_same_day():
    """T0 买入计划应在 T1 按优先级同日顺延。"""

    trading_dates = [pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03')]
    date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
    price_data = pd.DataFrame(
        [
            {
                'ts_code': '000001.SZ',
                'trade_date': '20230102',
                'close': 10.0,
                'close_adj': 10.0,
                'open': 10.0,
                'open_adj': 10.0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            },
            {
                'ts_code': '000002.SZ',
                'trade_date': '20230102',
                'close': 10.0,
                'close_adj': 10.0,
                'open': 10.0,
                'open_adj': 10.0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            },
            {
                'ts_code': '000001.SZ',
                'trade_date': '20230103',
                'close': 10.0,
                'close_adj': 10.0,
                'open': 10.0,
                'open_adj': 10.0,
                'is_suspended': 0,
                'is_limit_up': 1,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            },
            {
                'ts_code': '000002.SZ',
                'trade_date': '20230103',
                'close': 10.0,
                'close_adj': 10.0,
                'open': 10.0,
                'open_adj': 10.0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            },
        ]
    )

    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(),
        initial_capital=100000,
        cost_model=CostModel(commission_rate=0, min_commission=0, stamp_tax=0, slippage=0),
        rebalance_freq=1,
        holding_period=2,
        enable_pending_order=False,
        enable_position_completion=False,
        verbose=False,
    )
    engine.price_data_cache = price_data.copy()
    engine._prepare_price_index(price_data)
    engine.pending_signals[trading_dates[0]] = {
        'signals': {'000001.SZ': 1.0},
        'priority_candidates': [('000001.SZ', 1.0), ('000002.SZ', 0.9)],
        'slot_weights': [{'stock': '000001.SZ', 'weight': 1.0}],
        'target_n': 1,
        'tranche_idx': 0,
    }

    engine._execute_pending_buys(
        date=trading_dates[1],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    trades = engine.get_trades()
    assert len(trades) == 1
    assert trades.iloc[0]['action'] == 'buy'
    assert trades.iloc[0]['stock'] == '000002.SZ'
    assert trades.iloc[0]['signal_date'] == trading_dates[0]
    assert '000002.SZ' in engine.positions
    assert '000001.SZ' not in engine.positions

    attribution = engine.get_execution_attribution()
    assert len(attribution) == 1
    assert attribution.iloc[0]['planned_stock'] == '000001.SZ'
    assert attribution.iloc[0]['actual_stock'] == '000002.SZ'
    assert attribution.iloc[0]['actual_rank'] == 2
    assert attribution.iloc[0]['status'] == 'filled'
    assert attribution.iloc[0]['reason'] == '涨停'


def test_buy_plan_does_not_overbuy_when_sell_execution_fails():
    """T1 卖出失败时，不应继续按原买入计划超配加仓。"""

    trading_dates = [pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03')]
    date_to_idx = {date: idx for idx, date in enumerate(trading_dates)}
    price_data = pd.DataFrame(
        [
            {
                'ts_code': '000001.SZ',
                'trade_date': '20230102',
                'close': 10.0,
                'close_adj': 10.0,
                'open': 10.0,
                'open_adj': 10.0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            },
            {
                'ts_code': '000002.SZ',
                'trade_date': '20230102',
                'close': 10.0,
                'close_adj': 10.0,
                'open': 10.0,
                'open_adj': 10.0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            },
            {
                'ts_code': '000001.SZ',
                'trade_date': '20230103',
                'close': 10.0,
                'close_adj': 10.0,
                'open': 10.0,
                'open_adj': 10.0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 1,
                'is_st': 0,
                'list_days': 200,
            },
            {
                'ts_code': '000002.SZ',
                'trade_date': '20230103',
                'close': 10.0,
                'close_adj': 10.0,
                'open': 10.0,
                'open_adj': 10.0,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            },
        ]
    )

    engine = BacktestEngine(
        universe=MockUniverse(),
        signal=MockSignal(),
        initial_capital=100000,
        cost_model=CostModel(commission_rate=0, min_commission=0, stamp_tax=0, slippage=0),
        rebalance_freq=1,
        holding_period=2,
        enable_pending_order=True,
        enable_position_completion=False,
        verbose=False,
    )
    engine.price_data_cache = price_data.copy()
    engine._prepare_price_index(price_data)
    engine.current_capital = 50000.0
    engine.positions = {
        '000001.SZ': {
            'shares': 100,
            'buy_date': trading_dates[0],
            'signal_date': trading_dates[0],
            'buy_trade_price': 10.0,
            'buy_pnl_price': 10.0,
            'buy_cost_cash': 1000.0,
        }
    }
    engine.pending_condition_sells = {
        '000001.SZ': {
            'trigger_date': trading_dates[0],
            'sell_type': 'holding_period',
        }
    }
    engine.pending_signals[trading_dates[0]] = {
        'signals': {'000002.SZ': 1.0},
        'priority_candidates': [('000002.SZ', 1.0)],
        'slot_weights': [{'stock': '000002.SZ', 'weight': 1.0}],
        'target_n': 1,
        'tranche_idx': 0,
    }

    engine._execute_pending_condition_sells(
        date=trading_dates[1],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )
    engine._execute_pending_buys(
        date=trading_dates[1],
        trading_dates=trading_dates,
        date_to_idx=date_to_idx,
    )

    assert '000001.SZ' in engine.positions
    assert '000002.SZ' not in engine.positions
    assert engine.get_trades().empty
