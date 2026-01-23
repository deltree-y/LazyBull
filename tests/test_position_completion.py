"""测试仓位补齐机制"""

import pandas as pd
import pytest

from src.lazybull.backtest import BacktestEngine
from src.lazybull.common.cost import CostModel
from src.lazybull.signals.base import Signal
from src.lazybull.universe.base import BasicUniverse


class MockRankedSignal(Signal):
    """模拟信号生成器（支持 generate_ranked）"""
    
    def __init__(self, top_n=3):
        super().__init__("mock_ranked")
        self.top_n = top_n
        self.weight_method = "equal"
    
    def generate(self, date, universe, data):
        """生成等权信号"""
        if not universe:
            return {}
        selected = universe[:min(self.top_n, len(universe))]
        if not selected:
            return {}
        return {stock: 1.0 / len(selected) for stock in selected}
    
    def generate_ranked(self, date, universe, data):
        """生成排序候选列表（用于回填）"""
        if not universe:
            return []
        # 返回所有候选，按原顺序排序
        return [(stock, 1.0) for stock in universe]


@pytest.fixture
def completion_price_data():
    """创建测试仓位补齐的价格数据
    
    场景：
    - T日(第1天): 生成信号选出3只股票：000001, 000002, 000003
    - T+1日(第2天): 买入日，000002涨停，000003涨停 -> 只买入000001
    - T+2日(第3天): 补齐尝试1，000002仍涨停，000003开板 -> 补齐000003
    - T+3日(第4天): 补齐尝试2，000002开板 -> 补齐000002
    - 结果：3只股票全部补齐完成
    """
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
    
    data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            is_limit_up = 0
            pct_chg = 0.0
            
            # 000002在第2天(i=1)和第3天(i=2)涨停，第4天(i=3)开板
            if stock == '000002.SZ' and i in [1, 2]:
                is_limit_up = 1
                pct_chg = 9.99
            
            # 000003在第2天(i=1)涨停，第3天(i=2)开板
            if stock == '000003.SZ' and i == 1:
                is_limit_up = 1
                pct_chg = 9.99
            
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': 10.0 + i * 0.1,
                'close_adj': 10.0 + i * 0.1,
                'open': 10.0 + i * 0.1,
                'open_adj': 10.0 + i * 0.1,
                'vol': 1000000,
                'pct_chg': pct_chg,
                'filter_is_suspended': 0,
                'is_suspended': 0,
                'is_limit_up': is_limit_up,
                'is_limit_down': 0,
                'filter_is_st': 0,
                'is_st': 0,
                'filter_list_days': 100,
                'list_days': 100,
                'tradable': 1
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def completion_stock_basic():
    """股票基本信息"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ'],
        'name': ['股票1', '股票2', '股票3', '股票4', '股票5'],
        'market': ['主板', '主板', '主板', '主板', '主板'],
        'list_date': ['20200101', '20200101', '20200101', '20200101', '20200101']
    })


def test_position_completion_enabled(completion_price_data, completion_stock_basic):
    """测试仓位补齐功能启用时的行为"""
    
    # 创建股票池
    universe = BasicUniverse(
        stock_basic=completion_stock_basic,
        exclude_st=False,
        filter_suspended=False
    )
    
    # 创建信号生成器（top_n=3）
    signal = MockRankedSignal(top_n=3)
    
    # 创建回测引擎（启用补齐功能）
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(commission_rate=0, min_commission=0, stamp_tax=0, slippage=0),
        rebalance_freq=10,  # 10天调仓一次，确保只有一次调仓
        holding_period=10,
        enable_pending_order=False,  # 禁用延迟订单，简化测试
        enable_position_completion=True,  # 启用补齐功能
        completion_window_days=3,  # 3天补齐窗口
        verbose=True
    )
    
    # 运行回测
    trading_dates = pd.date_range('2023-01-01', periods=10, freq='B')
    nav_df = engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=list(trading_dates),
        price_data=completion_price_data
    )
    
    # 验证补齐统计
    assert engine.completion_stats['total_unfilled'] == 1, "应该有1次未满仓"
    assert engine.completion_stats['total_completed'] >= 2, "应该至少补齐2次（000002和000003）"
    assert engine.completion_stats['completion_attempts'] >= 2, "应该至少尝试2次补齐"
    
    # 验证最终持仓：应该成功买入3只股票
    # 注意：由于持有期=10天，在回测结束时还没有卖出
    assert len(engine.positions) == 3, "最终应该持有3只股票"
    
    # 验证买入的股票
    bought_stocks = set(engine.positions.keys())
    assert '000001.SZ' in bought_stocks, "应该成功买入000001（T+1日正常买入）"
    # 000002和000003至少有一只应该被补齐
    assert ('000002.SZ' in bought_stocks) or ('000003.SZ' in bought_stocks), \
        "至少应该补齐000002或000003中的一只"


def test_position_completion_disabled(completion_price_data, completion_stock_basic):
    """测试仓位补齐功能禁用时的行为"""
    
    # 创建股票池
    universe = BasicUniverse(
        stock_basic=completion_stock_basic,
        exclude_st=False,
        filter_suspended=False
    )
    
    # 创建信号生成器（top_n=3）
    signal = MockRankedSignal(top_n=3)
    
    # 创建回测引擎（禁用补齐功能）
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(commission_rate=0, min_commission=0, stamp_tax=0, slippage=0),
        rebalance_freq=10,
        holding_period=10,
        enable_pending_order=False,
        enable_position_completion=False,  # 禁用补齐功能
        verbose=True
    )
    
    # 运行回测
    trading_dates = pd.date_range('2023-01-01', periods=10, freq='B')
    nav_df = engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=list(trading_dates),
        price_data=completion_price_data
    )
    
    # 验证补齐统计：禁用时不应该有补齐操作
    assert engine.completion_stats['total_unfilled'] == 0, "禁用补齐时，不应该记录未满仓"
    assert engine.completion_stats['total_completed'] == 0, "禁用补齐时，不应该有补齐操作"
    
    # 验证最终持仓：只有T+1日成功买入的股票，没有补齐
    # 由于000002和000003在T+1日涨停，只能买入000001
    assert len(engine.positions) == 1, "禁用补齐时，应该只持有1只股票（T+1日成功买入的）"
    assert '000001.SZ' in engine.positions, "应该持有000001"


def test_completion_window_exceeded():
    """测试超过补齐窗口后放弃补齐"""
    
    # 创建价格数据：000002持续涨停超过补齐窗口
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
    
    data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            is_limit_up = 0
            
            # 000002在第2-5天持续涨停（超过3天补齐窗口）
            if stock == '000002.SZ' and 1 <= i <= 4:
                is_limit_up = 1
            
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': 10.0 + i * 0.1,
                'close_adj': 10.0 + i * 0.1,
                'open': 10.0 + i * 0.1,
                'open_adj': 10.0 + i * 0.1,
                'vol': 1000000,
                'pct_chg': 0.0,
                'filter_is_suspended': 0,
                'is_suspended': 0,
                'is_limit_up': is_limit_up,
                'is_limit_down': 0,
                'filter_is_st': 0,
                'is_st': 0,
                'filter_list_days': 100,
                'list_days': 100,
                'tradable': 1
            })
    
    price_data = pd.DataFrame(data)
    
    stock_basic = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
        'name': ['股票1', '股票2', '股票3'],
        'market': ['主板', '主板', '主板'],
        'list_date': ['20200101', '20200101', '20200101']
    })
    
    universe = BasicUniverse(
        stock_basic=stock_basic,
        exclude_st=False,
        filter_suspended=False
    )
    
    signal = MockRankedSignal(top_n=2)  # 选2只股票
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(commission_rate=0, min_commission=0, stamp_tax=0, slippage=0),
        rebalance_freq=10,
        holding_period=10,
        enable_pending_order=False,
        enable_position_completion=True,
        completion_window_days=3,  # 3天补齐窗口
        verbose=True
    )
    
    # 运行回测
    trading_dates = pd.date_range('2023-01-01', periods=10, freq='B')
    nav_df = engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=list(trading_dates),
        price_data=price_data
    )
    
    # 验证放弃补齐
    assert engine.completion_stats['total_unfilled'] == 1, "应该有1次未满仓"
    assert engine.completion_stats['total_abandoned'] >= 1, "应该至少放弃1次补齐"
    
    # 验证最终持仓：由于000002持续涨停，最终应该只持有1只股票（000001）
    # 或者补齐了000003
    assert 1 <= len(engine.positions) <= 2, "最终应该持有1-2只股票"


def test_completion_with_alternative_candidates():
    """测试原未成交股票不可用时，使用其他候选股票补齐"""
    
    # 创建价格数据：000002持续涨停，但000004可用
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ']
    
    data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            is_limit_up = 0
            
            # 000002持续涨停
            if stock == '000002.SZ' and i >= 1:
                is_limit_up = 1
            
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': 10.0 + i * 0.1,
                'close_adj': 10.0 + i * 0.1,
                'open': 10.0 + i * 0.1,
                'open_adj': 10.0 + i * 0.1,
                'vol': 1000000,
                'pct_chg': 0.0,
                'filter_is_suspended': 0,
                'is_suspended': 0,
                'is_limit_up': is_limit_up,
                'is_limit_down': 0,
                'filter_is_st': 0,
                'is_st': 0,
                'filter_list_days': 100,
                'list_days': 100,
                'tradable': 1
            })
    
    price_data = pd.DataFrame(data)
    
    stock_basic = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ'],
        'name': ['股票1', '股票2', '股票3', '股票4'],
        'market': ['主板', '主板', '主板', '主板'],
        'list_date': ['20200101', '20200101', '20200101', '20200101']
    })
    
    universe = BasicUniverse(
        stock_basic=stock_basic,
        exclude_st=False,
        filter_suspended=False
    )
    
    signal = MockRankedSignal(top_n=2)  # 选2只股票
    
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(commission_rate=0, min_commission=0, stamp_tax=0, slippage=0),
        rebalance_freq=10,
        holding_period=10,
        enable_pending_order=False,
        enable_position_completion=True,
        completion_window_days=3,
        verbose=True
    )
    
    # 运行回测
    trading_dates = pd.date_range('2023-01-01', periods=10, freq='B')
    nav_df = engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=list(trading_dates),
        price_data=price_data
    )
    
    # 验证补齐成功：应该用000003或000004补齐000002的槽位
    assert engine.completion_stats['total_unfilled'] == 1, "应该有1次未满仓"
    assert engine.completion_stats['total_completed'] >= 1, "应该至少补齐1次"
    
    # 验证最终持仓：应该成功买入2只股票
    assert len(engine.positions) == 2, "最终应该持有2只股票"
    assert '000001.SZ' in engine.positions, "应该持有000001"
    # 应该补齐了000003或000004
    assert ('000003.SZ' in engine.positions) or ('000004.SZ' in engine.positions), \
        "应该补齐000003或000004"
