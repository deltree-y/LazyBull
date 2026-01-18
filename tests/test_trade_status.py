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
    """测试数据缺失情况"""
    assert not is_suspended('999999.SZ', '20230110', sample_quote_data)


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
    """测试数据缺失情况"""
    info = get_trade_status_info('999999.SZ', '20230110', sample_quote_data)
    assert not info['is_suspended']
    assert not info['is_limit_up']
    assert not info['is_limit_down']
    assert info['can_buy']
    assert info['can_sell']
    assert info['close'] is None
    assert info['pct_chg'] is None
