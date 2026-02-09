"""测试交易状态检查工具"""

import pandas as pd
import pytest

from src.lazybull.common.trade_status import (
    is_suspended,
    is_limit_up,
    is_limit_down,
    is_tradeable,
    get_trade_status_info
)


@pytest.fixture
def sample_quote_data():
    """创建示例行情数据"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000001.SZ', '000002.SZ', '000002.SZ', '000003.SZ', '000004.SZ'],
        'trade_date': ['20230110', '20230111', '20230110', '20230111', '20230110', '20230110'],
        'close': [10.0, 10.5, 20.0, 18.0, 30.0, 40.0],
        'pct_chg': [5.0, 5.0, 9.95, -10.0, 0.0, 0.0],
        'vol': [1000000, 1000000, 2000000, 2000000, 0, 100000],
        'filter_is_suspended': [0, 0, 0, 0, 1, 0],
        'is_limit_up': [0, 0, 1, 0, 0, 0],
        'is_limit_down': [0, 0, 0, 1, 0, 0]
    })


def test_is_suspended_normal(sample_quote_data):
    """测试正常交易的股票不停牌"""
    assert not is_suspended('000001.SZ', '20230110', sample_quote_data)
    assert not is_suspended('000002.SZ', '20230110', sample_quote_data)


def test_is_suspended_suspended(sample_quote_data):
    """测试停牌股票"""
    assert is_suspended('000003.SZ', '20230110', sample_quote_data)


def test_is_suspended_missing_data(sample_quote_data):
    """测试数据缺失情况（保守策略：假定停牌）"""
    # 当股票不在行情数据中时，保守策略假定停牌
    assert is_suspended('999999.SZ', '20230110', sample_quote_data)


def test_is_limit_up_normal(sample_quote_data):
    """测试非涨停股票"""
    assert not is_limit_up('000001.SZ', '20230110', sample_quote_data)


def test_is_limit_up_limit_up(sample_quote_data):
    """测试涨停股票"""
    assert is_limit_up('000002.SZ', '20230110', sample_quote_data)


def test_is_limit_down_normal(sample_quote_data):
    """测试非跌停股票"""
    assert not is_limit_down('000001.SZ', '20230110', sample_quote_data)


def test_is_limit_down_limit_down(sample_quote_data):
    """测试跌停股票"""
    assert is_limit_down('000002.SZ', '20230111', sample_quote_data)


def test_is_tradeable_normal_buy(sample_quote_data):
    """测试正常股票可买入"""
    tradeable, reason = is_tradeable('000001.SZ', '20230110', sample_quote_data, 'buy')
    assert tradeable
    assert reason is None


def test_is_tradeable_normal_sell(sample_quote_data):
    """测试正常股票可卖出"""
    tradeable, reason = is_tradeable('000001.SZ', '20230110', sample_quote_data, 'sell')
    assert tradeable
    assert reason is None


def test_is_tradeable_suspended_buy(sample_quote_data):
    """测试停牌股票不可买入"""
    tradeable, reason = is_tradeable('000003.SZ', '20230110', sample_quote_data, 'buy')
    assert not tradeable
    assert reason == "停牌"


def test_is_tradeable_suspended_sell(sample_quote_data):
    """测试停牌股票不可卖出"""
    tradeable, reason = is_tradeable('000003.SZ', '20230110', sample_quote_data, 'sell')
    assert not tradeable
    assert reason == "停牌"


def test_is_tradeable_limit_up_buy(sample_quote_data):
    """测试涨停股票不可买入"""
    tradeable, reason = is_tradeable('000002.SZ', '20230110', sample_quote_data, 'buy')
    assert not tradeable
    assert reason == "涨停"


def test_is_tradeable_limit_up_sell(sample_quote_data):
    """测试涨停股票可卖出"""
    tradeable, reason = is_tradeable('000002.SZ', '20230110', sample_quote_data, 'sell')
    assert tradeable
    assert reason is None


def test_is_tradeable_limit_down_buy(sample_quote_data):
    """测试跌停股票可买入"""
    tradeable, reason = is_tradeable('000002.SZ', '20230111', sample_quote_data, 'buy')
    assert tradeable
    assert reason is None


def test_is_tradeable_limit_down_sell(sample_quote_data):
    """测试跌停股票不可卖出"""
    tradeable, reason = is_tradeable('000002.SZ', '20230111', sample_quote_data, 'sell')
    assert not tradeable
    assert reason == "跌停"


def test_get_trade_status_info_normal(sample_quote_data):
    """测试获取正常股票状态信息"""
    info = get_trade_status_info('000001.SZ', '20230110', sample_quote_data)
    assert not info['is_suspended']
    assert not info['is_limit_up']
    assert not info['is_limit_down']
    assert info['can_buy']
    assert info['can_sell']
    assert info['close'] == 10.0
    assert info['pct_chg'] == 5.0


def test_get_trade_status_info_suspended(sample_quote_data):
    """测试获取停牌股票状态信息"""
    info = get_trade_status_info('000003.SZ', '20230110', sample_quote_data)
    assert info['is_suspended']
    assert not info['can_buy']
    assert not info['can_sell']


def test_get_trade_status_info_limit_up(sample_quote_data):
    """测试获取涨停股票状态信息"""
    info = get_trade_status_info('000002.SZ', '20230110', sample_quote_data)
    assert not info['is_suspended']
    assert info['is_limit_up']
    assert not info['is_limit_down']
    assert not info['can_buy']
    assert info['can_sell']


def test_get_trade_status_info_limit_down(sample_quote_data):
    """测试获取跌停股票状态信息"""
    info = get_trade_status_info('000002.SZ', '20230111', sample_quote_data)
    assert not info['is_suspended']
    assert not info['is_limit_up']
    assert info['is_limit_down']
    assert info['can_buy']
    assert not info['can_sell']


def test_get_trade_status_info_missing(sample_quote_data):
    """测试数据缺失情况（保守策略：假定停牌）"""
    info = get_trade_status_info('999999.SZ', '20230110', sample_quote_data)
    # 当股票不在行情数据中时，保守策略假定停牌
    assert info['is_suspended']
    assert not info['can_buy']
    assert not info['can_sell']
    assert info['close'] is None
    assert info['pct_chg'] is None


@pytest.fixture
def sample_suspend_data():
    """创建示例停牌数据"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
        'trade_date': ['20230110', '20230110', '20230110'],
        'suspend_type': ['R', 'S', 'S'],  # R=复牌, S=停牌
        'suspend_timing': ['0', '0', '0']
    })


def test_is_suspended_by_suspend_df_suspended():
    """测试使用suspend_df判断停牌股票"""
    from src.lazybull.common.trade_status import is_suspended_by_suspend_df
    
    suspend_df = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ'],
        'trade_date': ['20230110', '20230110'],
        'suspend_type': ['S', 'S']
    })
    
    assert is_suspended_by_suspend_df('000001.SZ', '20230110', suspend_df)
    assert is_suspended_by_suspend_df('000002.SZ', '20230110', suspend_df)


def test_is_suspended_by_suspend_df_resumed():
    """测试使用suspend_df判断复牌股票"""
    from src.lazybull.common.trade_status import is_suspended_by_suspend_df
    
    suspend_df = pd.DataFrame({
        'ts_code': ['000001.SZ'],
        'trade_date': ['20230110'],
        'suspend_type': ['R']  # R表示复牌
    })
    
    assert not is_suspended_by_suspend_df('000001.SZ', '20230110', suspend_df)


def test_is_suspended_by_suspend_df_no_record():
    """测试使用suspend_df判断无记录的股票（视为未停牌）"""
    from src.lazybull.common.trade_status import is_suspended_by_suspend_df
    
    suspend_df = pd.DataFrame({
        'ts_code': ['000001.SZ'],
        'trade_date': ['20230110'],
        'suspend_type': ['S']
    })
    
    # 不在suspend_df中的股票，视为未停牌
    assert not is_suspended_by_suspend_df('000002.SZ', '20230110', suspend_df)


def test_is_suspended_with_suspend_df_priority(sample_quote_data, sample_suspend_data):
    """测试优先使用suspend_df判断停牌状态"""
    # 000001.SZ 在 quote_data 中 is_suspended=0，但在 suspend_df 中 suspend_type='R'（复牌）
    # 应该使用 suspend_df 的结果：未停牌
    assert not is_suspended('000001.SZ', '20230110', sample_quote_data, sample_suspend_data)
    
    # 000002.SZ 在 suspend_df 中 suspend_type='S'（停牌）
    # 应该使用 suspend_df 的结果：停牌
    assert is_suspended('000002.SZ', '20230110', sample_quote_data, sample_suspend_data)
    
    # 000003.SZ 在 suspend_df 中 suspend_type='S'（停牌）
    assert is_suspended('000003.SZ', '20230110', sample_quote_data, sample_suspend_data)


def test_is_suspended_fallback_to_quote_data(sample_quote_data):
    """测试在没有suspend_df时回退到quote_data"""
    # 当没有 suspend_df 时，应该使用 quote_data 的 is_suspended 列
    # 000003.SZ 在 quote_data 中 vol=0（停牌）
    assert is_suspended('000003.SZ', '20230110', sample_quote_data, None)


def test_is_tradeable_with_suspend_df(sample_quote_data, sample_suspend_data):
    """测试使用suspend_df判断可交易性"""
    # 000002.SZ 在 suspend_df 中停牌
    tradeable, reason = is_tradeable('000002.SZ', '20230110', sample_quote_data, 'buy', sample_suspend_data)
    assert not tradeable
    assert reason == "停牌"
    
    # 000001.SZ 在 suspend_df 中复牌，应该可交易
    tradeable, reason = is_tradeable('000001.SZ', '20230110', sample_quote_data, 'buy', sample_suspend_data)
    assert tradeable
    assert reason is None


def test_get_trade_status_info_with_suspend_df(sample_quote_data, sample_suspend_data):
    """测试使用suspend_df获取交易状态信息"""
    # 000002.SZ 在 suspend_df 中停牌
    info = get_trade_status_info('000002.SZ', '20230110', sample_quote_data, sample_suspend_data)
    assert info['is_suspended']
    assert not info['can_buy']
    assert not info['can_sell']
    
    # 000001.SZ 在 suspend_df 中复牌
    info = get_trade_status_info('000001.SZ', '20230110', sample_quote_data, sample_suspend_data)
    assert not info['is_suspended']
    assert info['can_buy']
    assert info['can_sell']
