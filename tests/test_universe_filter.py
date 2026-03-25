"""测试 Universe 向量化停牌过滤"""

import pandas as pd
import pytest

from src.lazybull.universe.base import BasicUniverse


@pytest.fixture
def stock_list():
    """模拟股票列表"""
    return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '603056.SH']


@pytest.fixture
def quote_data_with_suspended():
    """模拟行情数据，包含停牌股票"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH'],
        'trade_date': ['20260317'] * 4,
        'close': [15.0, 20.0, 8.0, 45.0],
        'vol': [100000, 200000, 0, 150000],  # 600000.SH 停牌（vol=0）
        'is_suspended': [0, 0, 1, 0],  # 600000.SH 标记为停牌
    })


@pytest.fixture
def mock_stock_basic():
    """模拟股票基本信息"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '603056.SH'],
        'name': ['平安银行', '万科A', '浦发银行', '招商银行', '德邦股份'],
        'list_date': ['20100101'] * 5,
    })


@pytest.fixture
def universe(mock_stock_basic):
    """创建 BasicUniverse 实例（启用停牌过滤）"""
    return BasicUniverse(
        stock_basic=mock_stock_basic,
        filter_suspended=True,
        filter_limit_stocks=False,
        verbose=True,
    )


class TestFilterUntradeableStocks:
    """测试向量化停牌过滤"""

    def test_filter_suspended_stocks(self, universe, stock_list, quote_data_with_suspended):
        """停牌股票应被过滤"""
        date = pd.Timestamp('2026-03-17')
        result = universe._filter_untradeable_stocks(stock_list, date, quote_data_with_suspended)

        assert '000001.SZ' in result
        assert '000002.SZ' in result
        assert '600036.SH' in result
        # 600000.SH 停牌，应被过滤
        assert '600000.SH' not in result
        # 603056.SH 无行情数据，视为停牌
        assert '603056.SH' not in result

    def test_no_data_treated_as_suspended(self, universe, stock_list, quote_data_with_suspended):
        """无行情数据的股票应被视为停牌"""
        date = pd.Timestamp('2026-03-17')
        result = universe._filter_untradeable_stocks(stock_list, date, quote_data_with_suspended)

        # 603056.SH 不在行情数据中，视为停牌
        assert '603056.SH' not in result
        # 有数据且未停牌的保留
        assert len(result) == 3

    def test_empty_quote_data_returns_all(self, universe, stock_list):
        """空行情数据时保留全部股票"""
        date = pd.Timestamp('2026-03-17')
        empty_df = pd.DataFrame(columns=['ts_code', 'trade_date', 'vol', 'is_suspended'])
        result = universe._filter_untradeable_stocks(stock_list, date, empty_df)

        assert result == stock_list

    def test_filter_suspended_disabled(self, stock_list, quote_data_with_suspended, mock_stock_basic):
        """filter_suspended=False 时应保留全部"""
        universe = BasicUniverse(
            stock_basic=mock_stock_basic, filter_suspended=False, verbose=False
        )
        date = pd.Timestamp('2026-03-17')
        result = universe._filter_untradeable_stocks(
            stock_list, date, quote_data_with_suspended
        )

        assert result == stock_list

    def test_vol_fallback(self, universe):
        """没有 is_suspended 列时，使用 vol 判断停牌"""
        stock_list = ['000001.SZ', '000002.SZ']
        quote_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'trade_date': ['20260317', '20260317'],
            'close': [15.0, 20.0],
            'vol': [100000, 0],  # 000002.SZ vol=0，停牌
        })
        date = pd.Timestamp('2026-03-17')
        result = universe._filter_untradeable_stocks(stock_list, date, quote_data)

        assert '000001.SZ' in result
        assert '000002.SZ' not in result

    def test_preserves_order(self, universe, quote_data_with_suspended):
        """过滤后应保持原始顺序"""
        stock_list = ['600036.SH', '000001.SZ', '000002.SZ']
        date = pd.Timestamp('2026-03-17')
        result = universe._filter_untradeable_stocks(
            stock_list, date, quote_data_with_suspended
        )

        assert result == ['600036.SH', '000001.SZ', '000002.SZ']

    def test_no_date_data_returns_all(self, universe, stock_list):
        """当日无行情数据时保留全部"""
        quote_data = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20260316'],  # 不同日期
            'vol': [100000],
            'is_suspended': [0],
        })
        date = pd.Timestamp('2026-03-17')
        result = universe._filter_untradeable_stocks(stock_list, date, quote_data)

        assert result == stock_list
