"""测试仓位补齐机制"""

import io

import pandas as pd
import pytest
from loguru import logger

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
        - T+1日(第2天): 买入日，000002涨停，000003涨停
            -> 同日按优先级顺延买入000004、000005
        - 结果：3个槽位在T+1日全部补满，不进入跨日补齐
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
    """启用补齐时，T1 备用候选足够应优先同日补满。"""
    
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
    
    # 验证：同日顺延已补满，不应进入跨日补齐
    assert engine.completion_stats['total_unfilled'] == 0, "同日补满后，不应该记录未满仓"
    assert engine.completion_stats['total_completed'] == 0, "同日补满后，不应该走跨日补齐"
    assert engine.completion_stats['completion_attempts'] == 0, "同日补满后，不应该发起补齐尝试"
    
    # 验证最终持仓：应该成功买入3只股票
    # 注意：由于持有期=10天，在回测结束时还没有卖出
    assert len(engine.positions) == 3, "最终应该持有3只股票"
    
    # 验证买入的股票：000002/000003 涨停后，应同日顺延到 000004/000005
    assert set(engine.positions.keys()) == {'000001.SZ', '000004.SZ', '000005.SZ'}


def test_position_completion_disabled(completion_price_data, completion_stock_basic):
    """禁用跨日补齐时，T1 同日顺延仍应生效。"""
    
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
    
    # 验证最终持仓：禁用跨日补齐时，也应在 T+1 基于候选优先级顺延补满
    assert set(engine.positions.keys()) == {'000001.SZ', '000004.SZ', '000005.SZ'}


def test_completion_window_exceeded():
    """测试超过补齐窗口后放弃补齐"""
    
    # 创建价格数据：000002持续涨停超过补齐窗口，000003也涨停超过窗口
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ']
    
    data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            is_limit_up = 0
            
            # 000002在第2-5天持续涨停（超过3天补齐窗口）
            if stock == '000002.SZ' and 1 <= i <= 4:
                is_limit_up = 1
            
            # 000003也在第2-5天持续涨停，确保无法补齐
            if stock == '000003.SZ' and 1 <= i <= 4:
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
    """原计划股票不可买时，应在同日顺延到后备候选。"""
    
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
    
    # 验证：同日顺延成功后，不应进入跨日补齐
    assert engine.completion_stats['total_unfilled'] == 0, "同日已顺延成功，不应该记录未满仓"

    # 验证最终持仓：000002涨停时，应同日顺延到 000003
    assert '000001.SZ' in engine.positions, "应该持有000001"
    assert '000003.SZ' in engine.positions, "应该同日顺延买入000003"
    assert len(engine.positions) == 2, "同日顺延后应持有2只股票"


def test_completion_uses_prev_day_data():
    """同日候选耗尽后，补齐应在后续日期使用 D-1 数据生成候选。"""
    
    # 创建价格数据：测试补齐时是否使用正确的日期数据
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ']
    
    data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            is_limit_up = 0
            
            # T+1 同日顺延时，000002/000003/000004 都不可买，迫使进入跨日补齐
            if stock == '000002.SZ' and i in [1, 2]:
                is_limit_up = 1

            if stock in ['000003.SZ', '000004.SZ'] and i == 1:
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
    
    signal = MockRankedSignal(top_n=2)
    
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
    
    # 验证：T+1 同日候选耗尽后，T+2 发生跨日补齐
    assert engine.completion_stats['total_unfilled'] == 1, "应该有1次未满仓"
    assert engine.completion_stats['total_completed'] == 1, "应该补齐1次"
    assert len(engine.positions) == 2, "最终应该持有2只股票"
    
    # 验证持仓的股票
    assert '000001.SZ' in engine.positions, "应该持有000001"
    assert '000003.SZ' in engine.positions, "T+2 应补齐到 000003"
    assert engine.positions['000003.SZ']['buy_date'] == trading_dates[2], "补齐应发生在 T+2"


def test_completion_log_summarizes_success_when_cash_limited():
    """补齐日志应压缩为单行摘要，实际成交由交易记录校验。"""

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
                'ts_code': '600569.SH',
                'trade_date': '20230102',
                'close': 2.17,
                'close_adj': 2.17,
                'open': 2.17,
                'open_adj': 2.17,
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
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            },
            {
                'ts_code': '600569.SH',
                'trade_date': '20230103',
                'close': 2.17,
                'close_adj': 2.17,
                'open': 2.17,
                'open_adj': 2.17,
                'is_suspended': 0,
                'is_limit_up': 0,
                'is_limit_down': 0,
                'is_st': 0,
                'list_days': 200,
            },
        ]
    )

    stock_basic = pd.DataFrame(
        {
            'ts_code': ['000001.SZ', '600569.SH'],
            'name': ['股票1', '股票2'],
            'market': ['主板', '主板'],
            'list_date': ['20200101', '20200101'],
        }
    )

    universe = BasicUniverse(stock_basic=stock_basic, exclude_st=False, filter_suspended=False)
    signal = MockRankedSignal(top_n=2)
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000.0,
        cost_model=CostModel(commission_rate=0, min_commission=0, stamp_tax=0, slippage=0),
        rebalance_freq=5,
        holding_period=5,
        enable_pending_order=False,
        enable_position_completion=True,
        completion_window_days=3,
        verbose=True,
    )
    engine._prepare_price_index(price_data)
    engine.positions = {
        '000001.SZ': {
            'shares': 9970,
            'buy_date': trading_dates[0],
            'signal_date': trading_dates[0],
            'buy_trade_price': 10.0,
            'buy_pnl_price': 10.0,
            'buy_cost_cash': 99700.0,
        }
    }
    engine.current_capital = 300.0
    engine.unfilled_slots = {
        trading_dates[0]: {
            'unfilled_count': 1,
            'unfilled_slot_weights': [{'stock': '002119.SZ', 'weight': 0.5}],
            'target_n': 2,
            'ranked_candidates': [('000001.SZ', 1.0), ('600569.SH', 0.9)],
            'signal_date': trading_dates[0],
            'first_attempt_date': trading_dates[0],
            'attempts': 0,
            'tranche_idx': 0,
        }
    }

    stream = io.StringIO()
    sink_id = logger.add(stream, format='{message}')
    try:
        engine._process_position_completion(
            date=trading_dates[1],
            trading_dates=trading_dates,
            price_data=price_data,
            date_to_idx=date_to_idx,
        )
    finally:
        logger.remove(sink_id)

    output = stream.getvalue()
    assert '补齐: 成功1[002119.SZ→600569.SH]' in output
    assert '目标市值' not in output
    assert '600569.SH' in engine.positions
    assert engine.positions['600569.SH']['shares'] == 100

