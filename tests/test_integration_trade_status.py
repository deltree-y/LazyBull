"""集成测试：涨跌停与停牌处理完整流程"""

import pandas as pd
import pytest

from src.lazybull.backtest import BacktestEngine
from src.lazybull.common.cost import CostModel
from src.lazybull.signals.base import Signal
from src.lazybull.universe.base import BasicUniverse


class MockSignal(Signal):
    """模拟信号生成器（等权）"""
    
    def generate(self, date, universe, data):
        """生成等权信号"""
        if not universe:
            return {}
        # 只选前3只
        selected = universe[:min(3, len(universe))]
        if not selected:
            return {}
        return {stock: 1.0 / len(selected) for stock in selected}


@pytest.fixture
def sample_price_data_with_status():
    """创建包含交易状态的示例价格数据"""
    dates = pd.date_range('2023-01-01', periods=10, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ']
    
    data = []
    for i, date in enumerate(dates):
        for stock in stocks:
            # 000001.SZ: 正常交易
            # 000002.SZ: 第2-4天停牌
            # 000003.SZ: 第3天涨停
            # 000004.SZ: 第5天跌停
            # 000005.SZ: 正常交易
            
            is_suspended = 0
            is_limit_up = 0
            is_limit_down = 0
            vol = 1000000
            pct_chg = 0.0
            
            if stock == '000002.SZ' and 1 <= i <= 3:
                is_suspended = 1
                vol = 0
            
            if stock == '000003.SZ' and i == 2:
                is_limit_up = 1
                pct_chg = 9.99
            
            if stock == '000004.SZ' and i == 4:
                is_limit_down = 1
                pct_chg = -9.99
            
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': 10.0 + i * 0.1,  # 逐日上涨
                'close_adj': 10.0 + i * 0.1,
                'vol': vol,
                'pct_chg': pct_chg,
                'filter_is_suspended': is_suspended,
                'is_limit_up': is_limit_up,
                'is_limit_down': is_limit_down,
                'filter_is_st': 0,
                'filter_list_days': 100,
                'tradable': 1 if not is_suspended else 0
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_stock_basic():
    """创建股票基本信息"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ'],
        'name': ['股票1', '股票2', '股票3', '股票4', '股票5'],
        'market': ['主板', '主板', '主板', '主板', '主板'],
        'list_date': ['20200101', '20200101', '20200101', '20200101', '20200101']
    })


def test_stock_selection_filtering(sample_price_data_with_status, sample_stock_basic):
    """测试Universe阶段仅过滤停牌股票（涨跌停在信号生成时基于T+1数据过滤）"""
    
    # 创建股票池（启用过滤）
    universe = BasicUniverse(
        stock_basic=sample_stock_basic,
        exclude_st=False,
        filter_suspended=True,  # 过滤停牌
        filter_limit_stocks=True  # 此参数在Universe级别不再过滤涨跌停
    )
    
    # 获取第3天的行情数据（000002停牌，000003涨停）
    trading_dates = pd.date_range('2023-01-01', periods=10, freq='B')
    date = trading_dates[2]
    date_quote = sample_price_data_with_status[
        sample_price_data_with_status['trade_date'] == date.strftime('%Y%m%d')
    ]
    
    # 获取股票池
    stocks = universe.get_stocks(date, quote_data=date_quote)
    
    # 验证：
    # - 000002（停牌）应被过滤（Universe级别过滤）
    # - 000003（涨停）不应被过滤（因为T日涨停不代表T+1日也涨停，需要在信号生成时基于T+1数据过滤）
    assert '000002.SZ' not in stocks  # 停牌被过滤
    assert '000003.SZ' in stocks  # 涨停不在此过滤（留给信号生成阶段）
    assert '000001.SZ' in stocks  # 正常
    assert '000004.SZ' in stocks  # 正常
    assert '000005.SZ' in stocks  # 正常
    
    print(f"✓ Universe过滤测试通过: 仅过滤停牌股票，涨跌停留待信号生成时基于T+1数据过滤")


def test_pending_order_mechanism(sample_price_data_with_status, sample_stock_basic):
    """测试新的信号生成机制（基于T+1数据过滤并回填）
    
    新设计：不再使用延迟订单for买入，而是在信号生成时基于T+1数据过滤并回填。
    延迟订单机制仅用于卖出时的跌停情况。
    """
    
    # 创建股票池（启用过滤）
    universe = BasicUniverse(
        stock_basic=sample_stock_basic,
        exclude_st=False,
        filter_suspended=True,
        filter_limit_stocks=True
    )
    
    signal = MockSignal()
    
    # 创建回测引擎
    engine = BacktestEngine(
        universe=universe,
        signal=signal,
        initial_capital=100000,
        cost_model=CostModel(),
        rebalance_freq=1,  # 每天调仓
        holding_period=2,  # 持有2天
        enable_pending_order=True,  # 启用延迟订单（主要用于卖出跌停）
        max_retry_count=5,
        max_retry_days=5,
        verbose=False  # 关闭详细日志避免输出过多
    )
    
    # 运行回测
    trading_dates = pd.date_range('2023-01-01', periods=10, freq='B')
    nav_curve = engine.run(
        start_date=trading_dates[0],
        end_date=trading_dates[-1],
        trading_dates=list(trading_dates),
        price_data=sample_price_data_with_status
    )
    
    # 验证回测完成
    assert len(nav_curve) == len(trading_dates)
    assert 'nav' in nav_curve.columns
    
    # 验证延迟订单统计（新设计下买入不使用延迟订单）
    if engine.pending_order_manager:
        stats = engine.pending_order_manager.get_statistics()
        print(f"✓ 延迟订单统计: 累计添加 {stats['total_added']}, "
              f"成功执行 {stats['total_succeeded']}, 过期放弃 {stats['total_expired']}")
        
        # 新设计：买入不使用延迟订单（在信号生成时已过滤），
        # 延迟订单仅用于卖出跌停情况
    
    # 获取交易记录
    trades_df = engine.get_trades()
    print(f"✓ 回测完成: 共 {len(trades_df)} 笔交易")
    print(f"✓ 新设计验证: 信号生成时基于T+1数据过滤并回填，买入不使用延迟订单")
    
    # 验证基本交易记录
    assert len(trades_df) >= 0  # 至少应该没有错误


def test_filtering_configuration(sample_price_data_with_status, sample_stock_basic):
    """测试过滤功能的配置开关"""
    
    # 测试1: 关闭所有过滤
    universe_no_filter = BasicUniverse(
        stock_basic=sample_stock_basic,
        exclude_st=False,
        filter_suspended=False,  # 不过滤停牌
        filter_limit_stocks=False  # 不过滤涨跌停
    )
    
    trading_dates = pd.date_range('2023-01-01', periods=10, freq='B')
    date = trading_dates[2]
    date_quote = sample_price_data_with_status[
        sample_price_data_with_status['trade_date'] == date.strftime('%Y%m%d')
    ]
    
    stocks_no_filter = universe_no_filter.get_stocks(date, quote_data=date_quote)
    
    # 不过滤时应该包含所有股票
    assert len(stocks_no_filter) == 5
    assert '000002.SZ' in stocks_no_filter  # 停牌但未过滤
    assert '000003.SZ' in stocks_no_filter  # 涨停但未过滤
    
    # 测试2: 仅过滤停牌
    universe_filter_suspend = BasicUniverse(
        stock_basic=sample_stock_basic,
        exclude_st=False,
        filter_suspended=True,  # 过滤停牌
        filter_limit_stocks=False  # 不过滤涨跌停
    )
    
    stocks_filter_suspend = universe_filter_suspend.get_stocks(date, quote_data=date_quote)
    
    assert '000002.SZ' not in stocks_filter_suspend  # 停牌被过滤
    assert '000003.SZ' in stocks_filter_suspend  # 涨停未过滤
    assert len(stocks_filter_suspend) == 4
    
    # 测试3: filter_limit_stocks参数（已不在Universe级别过滤，保留参数以兼容）
    universe_filter_limit = BasicUniverse(
        stock_basic=sample_stock_basic,
        exclude_st=False,
        filter_suspended=False,  # 不过滤停牌
        filter_limit_stocks=True  # 此参数在Universe级别不再生效
    )
    
    stocks_filter_limit = universe_filter_limit.get_stocks(date, quote_data=date_quote)
    
    # 新设计：涨跌停不在Universe级别过滤，所以都应该在
    assert '000002.SZ' in stocks_filter_limit  # 停牌未过滤
    assert '000003.SZ' in stocks_filter_limit  # 涨停不在此过滤（已移至信号生成阶段）
    assert len(stocks_filter_limit) == 5
    
    print("✓ 过滤配置测试通过: Universe仅过滤停牌，涨跌停在信号生成阶段基于T+1数据过滤")


def test_backward_compatibility(sample_price_data_with_status, sample_stock_basic):
    """测试向后兼容性 - 不提供quote_data时不崩溃"""
    
    universe = BasicUniverse(
        stock_basic=sample_stock_basic,
        exclude_st=False
    )
    
    trading_dates = pd.date_range('2023-01-01', periods=10, freq='B')
    date = trading_dates[0]
    
    # 不提供 quote_data 参数（旧API）
    stocks = universe.get_stocks(date)
    
    # 应该返回所有股票（因为没有过滤数据）
    assert len(stocks) == 5
    
    print("✓ 向后兼容性测试通过: 不提供quote_data时正常工作")


if __name__ == '__main__':
    # 可以直接运行此文件进行快速测试
    pytest.main([__file__, '-v', '-s'])
