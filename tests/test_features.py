"""测试特征构建模块"""

import pandas as pd
import pytest

from src.lazybull.features import FeatureBuilder


@pytest.fixture
def mock_trade_cal():
    """模拟交易日历"""
    # 创建20个连续交易日
    dates = pd.date_range('2023-01-01', periods=20, freq='B')  # Business days
    
    return pd.DataFrame({
        'exchange': ['SSE'] * len(dates),
        'cal_date': dates.strftime('%Y%m%d').tolist(),
        'is_open': [1] * len(dates)
    })


@pytest.fixture
def mock_stock_basic():
    """模拟股票基本信息"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
        'name': ['平安银行', '万科A', '*ST浦发', '邯郸钢铁'],
        'list_date': ['20100101', '20100101', '20100101', '20230101']  # 最后一个刚上市
    })


@pytest.fixture
def mock_daily_data():
    """模拟日线行情数据"""
    dates = pd.date_range('2023-01-01', periods=20, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH']
    
    data = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        for i, stock in enumerate(stocks):
            # 模拟价格
            base_price = 10.0 + i
            close = base_price * (1 + 0.01 * i)
            pre_close = base_price
            pct_chg = ((close - pre_close) / pre_close) * 100
            
            # 第三只股票在某些日期停牌（成交量为0）
            vol = 0 if (stock == '600000.SH' and date.day % 5 == 0) else 1000000
            
            data.append({
                'ts_code': stock,
                'trade_date': date_str,
                'close': close,
                'pre_close': pre_close,
                'pct_chg': pct_chg,
                'vol': vol,
                'amount': vol * close if vol > 0 else 0
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_adj_factor():
    """模拟复权因子"""
    dates = pd.date_range('2023-01-01', periods=20, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH']
    
    data = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        for stock in stocks:
            # 所有股票使用相同的复权因子（简化）
            data.append({
                'ts_code': stock,
                'trade_date': date_str,
                'adj_factor': 1.0
            })
    
    return pd.DataFrame(data)


class TestFeatureBuilder:
    """测试特征构建器"""
    
    def test_init(self):
        """测试初始化"""
        builder = FeatureBuilder(min_list_days=60, horizon=5)
        
        assert builder.min_list_days == 60
        assert builder.horizon == 5
        assert len(builder.lookback_windows) > 0
    
    def test_get_trading_dates(self, mock_trade_cal):
        """测试提取交易日列表"""
        builder = FeatureBuilder()
        
        trading_dates = builder._get_trading_dates(mock_trade_cal)
        
        assert len(trading_dates) == 20
        assert trading_dates == sorted(trading_dates)
        assert trading_dates[0] < trading_dates[-1]
    
    def test_calculate_adj_close(self, mock_daily_data, mock_adj_factor):
        """测试计算后复权收盘价"""
        builder = FeatureBuilder()
        
        daily_adj = builder._calculate_adj_close(mock_daily_data, mock_adj_factor)
        
        # 检查是否添加了 close_adj 列
        assert 'close_adj' in daily_adj.columns
        
        # 检查计算是否正确（复权因子为1时，close_adj应该等于close）
        assert (daily_adj['close_adj'] == daily_adj['close']).all()
    
    def test_calculate_forward_returns(
        self,
        mock_daily_data,
        mock_adj_factor,
        mock_trade_cal
    ):
        """测试计算未来5日收益"""
        builder = FeatureBuilder(horizon=5)
        
        # 准备数据
        trading_dates = builder._get_trading_dates(mock_trade_cal)
        daily_adj = builder._calculate_adj_close(mock_daily_data, mock_adj_factor)
        
        # 选择第一个交易日
        trade_date = trading_dates[0]
        current_idx = 0
        current_data = daily_adj[daily_adj['trade_date'] == trade_date].copy()
        
        # 计算标签
        labels = builder._calculate_forward_returns(
            current_data,
            daily_adj,
            trade_date,
            trading_dates,
            current_idx
        )
        
        # 检查返回结果
        assert 'y_ret_5' in labels.columns
        assert len(labels) == len(current_data)
        
        # 未来5日的数据应该存在，标签不应该全是NaN
        assert not labels['y_ret_5'].isna().all()
    
    def test_forward_returns_calculation_correctness(
        self,
        mock_daily_data,
        mock_adj_factor,
        mock_trade_cal
    ):
        """测试未来收益计算的正确性"""
        builder = FeatureBuilder(horizon=5)
        
        trading_dates = builder._get_trading_dates(mock_trade_cal)
        daily_adj = builder._calculate_adj_close(mock_daily_data, mock_adj_factor)
        
        trade_date = trading_dates[0]
        future_date = trading_dates[5]  # 第5个后续交易日
        
        current_idx = 0
        current_data = daily_adj[daily_adj['trade_date'] == trade_date].copy()
        
        labels = builder._calculate_forward_returns(
            current_data,
            daily_adj,
            trade_date,
            trading_dates,
            current_idx
        )
        
        # 手动验证第一只股票的收益计算
        stock = '000001.SZ'
        current_price = daily_adj[
            (daily_adj['trade_date'] == trade_date) & 
            (daily_adj['ts_code'] == stock)
        ]['close_adj'].iloc[0]
        
        future_price = daily_adj[
            (daily_adj['trade_date'] == future_date) & 
            (daily_adj['ts_code'] == stock)
        ]['close_adj'].iloc[0]
        
        expected_return = (future_price / current_price) - 1
        actual_return = labels[labels['ts_code'] == stock]['y_ret_5'].iloc[0]
        
        assert abs(actual_return - expected_return) < 1e-6
    
    def test_forward_returns_insufficient_future_dates(
        self,
        mock_daily_data,
        mock_adj_factor,
        mock_trade_cal
    ):
        """测试未来交易日不足的情况"""
        builder = FeatureBuilder(horizon=5)
        
        trading_dates = builder._get_trading_dates(mock_trade_cal)
        daily_adj = builder._calculate_adj_close(mock_daily_data, mock_adj_factor)
        
        # 选择最后一个交易日（没有足够的未来数据）
        trade_date = trading_dates[-1]
        current_idx = len(trading_dates) - 1
        current_data = daily_adj[daily_adj['trade_date'] == trade_date].copy()
        
        labels = builder._calculate_forward_returns(
            current_data,
            daily_adj,
            trade_date,
            trading_dates,
            current_idx
        )
        
        # 标签应该全是NaN
        assert labels['y_ret_5'].isna().all()
    
    def test_apply_filters_st_stocks(
        self,
        mock_stock_basic
    ):
        """测试ST股票过滤"""
        builder = FeatureBuilder(min_list_days=60)
        
        # 创建模拟特征数据
        df = pd.DataFrame({
            'trade_date': ['20230110'] * 4,
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'ret_1': [0.01, 0.02, 0.03, 0.01],
            'y_ret_5': [0.05, 0.06, 0.07, 0.08],
            'vol': [1000000, 1000000, 1000000, 1000000]
        })
        
        # 添加过滤标记
        result = builder._add_filter_flags(
            df,
            mock_stock_basic,
            None,
            '20230110'
        )
        
        # 检查ST标记（统一列名：is_st）
        assert result[result['ts_code'] == '600000.SH']['is_st'].iloc[0] == 1  # *ST浦发
        assert result[result['ts_code'] == '000001.SZ']['is_st'].iloc[0] == 0  # 平安银行
        
        # 应用过滤
        filtered = builder._apply_filters(result)
        
        # ST股票应该被过滤掉
        assert '600000.SH' not in filtered['ts_code'].values
        assert '000001.SZ' in filtered['ts_code'].values
        # 确认统一列名存在
        assert 'is_st' in filtered.columns
    
    def test_apply_filters_list_days(
        self,
        mock_stock_basic
    ):
        """测试上市天数过滤"""
        builder = FeatureBuilder(min_list_days=60)
        
        df = pd.DataFrame({
            'trade_date': ['20230110'] * 4,
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'ret_1': [0.01, 0.02, 0.03, 0.01],
            'y_ret_5': [0.05, 0.06, 0.07, 0.08],
            'vol': [1000000, 1000000, 1000000, 1000000]
        })
        
        result = builder._add_filter_flags(
            df,
            mock_stock_basic,
            None,
            '20230110'
        )
        
        filtered = builder._apply_filters(result)
        
        # 刚上市的股票（600001.SH，上市日期20230101）应该被过滤掉
        # 注意：这里的判断依赖于上市天数计算
        assert '600001.SH' not in filtered['ts_code'].values or len(filtered) == 0
    
    def test_apply_filters_suspend(
        self,
        mock_stock_basic
    ):
        """测试停牌过滤"""
        builder = FeatureBuilder(min_list_days=10)  # 降低上市天数要求
        
        df = pd.DataFrame({
            'trade_date': ['20230110'] * 4,
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'ret_1': [0.01, 0.02, 0.03, 0.01],
            'y_ret_5': [0.05, 0.06, 0.07, 0.08],
            'vol': [1000000, 0, 1000000, 1000000]  # 000002.SZ停牌（成交量为0）
        })
        
        result = builder._add_filter_flags(
            df,
            mock_stock_basic,
            None,
            '20230110'
        )
        
        # 检查停牌标记（统一列名：is_suspended）
        assert result[result['ts_code'] == '000002.SZ']['is_suspended'].iloc[0] == 1
        
        # 应用过滤
        filtered = builder._apply_filters(result)
        
        # 停牌股票应该被过滤掉
        assert '000002.SZ' not in filtered['ts_code'].values
        # 确认统一列名存在
        assert 'is_suspended' in filtered.columns
    
    def test_build_features_for_day_integration(
        self,
        mock_trade_cal,
        mock_daily_data,
        mock_adj_factor,
        mock_stock_basic
    ):
        """测试完整的单日特征构建流程"""
        builder = FeatureBuilder(min_list_days=10, horizon=5)
        
        # 选择一个中间的交易日（确保有历史和未来数据）
        trade_date = '20230109'  # 第5个交易日
        
        features = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal,
            daily_data=mock_daily_data,
            adj_factor=mock_adj_factor,
            stock_basic=mock_stock_basic,
            suspend_info=None,
            limit_info=None
        )
        
        # 检查返回结果
        assert len(features) > 0
        assert 'trade_date' in features.columns
        assert 'ts_code' in features.columns
        assert 'y_ret_5' in features.columns
        assert 'ret_1' in features.columns
        # 统一列名（与clean层一致）
        assert 'is_st' in features.columns
        assert 'is_suspended' in features.columns
        assert 'list_days' in features.columns
        assert 'is_limit_up' in features.columns
        assert 'is_limit_down' in features.columns
        
        # 所有样本的trade_date应该一致
        assert (features['trade_date'] == trade_date).all()
        
        # 标签不应该有缺失值（因为已经过滤）
        assert not features['y_ret_5'].isna().any()
        
        # ST股票应该被过滤掉
        assert not features['is_st'].any()
        
        # 停牌股票应该被过滤掉
        assert not features['is_suspended'].any()
    
    def test_limit_flags(self, mock_daily_data, mock_stock_basic):
        """测试涨跌停标记"""
        builder = FeatureBuilder()
        
        # 创建包含涨跌停的数据
        df = pd.DataFrame({
            'trade_date': ['20230110'] * 4,
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'ret_1': [0.01, 0.02, 0.03, 0.01],
            'y_ret_5': [0.05, 0.06, 0.07, 0.08],
            'is_st': [0, 0, 1, 0],  # 第三只是ST
            'vol': [1000000, 1000000, 1000000, 1000000]
        })
        
        # 模拟涨停和跌停的日线数据
        limit_daily = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'trade_date': ['20230110'] * 4,
            'close': [11.0, 12.0, 13.0, 14.0],
            'pct_chg': [10.0, -10.0, 5.0, 0.0]  # 涨停、跌停、ST涨停、正常
        })
        
        result = builder._add_limit_flags(
            df,
            limit_daily,
            None,
            '20230110'
        )
        
        # 检查涨跌停标记（统一列名：is_limit_up, is_limit_down）
        assert result[result['ts_code'] == '000001.SZ']['is_limit_up'].iloc[0] == 1
        assert result[result['ts_code'] == '000002.SZ']['is_limit_down'].iloc[0] == 1
        assert result[result['ts_code'] == '600000.SH']['is_limit_up'].iloc[0] == 1  # ST股票5%涨停
        assert result[result['ts_code'] == '600001.SH']['is_limit_up'].iloc[0] == 0


def test_feature_builder_with_empty_data():
    """测试空数据的处理"""
    builder = FeatureBuilder()
    
    # 空交易日历
    empty_cal = pd.DataFrame(columns=['exchange', 'cal_date', 'is_open'])
    
    trading_dates = builder._get_trading_dates(empty_cal)
    assert len(trading_dates) == 0


def test_feature_builder_reuses_clean_markers():
    """测试特征构建器复用clean层标记"""
    builder = FeatureBuilder(min_list_days=60)
    
    # 创建包含clean层标记的数据
    df = pd.DataFrame({
        'trade_date': ['20230110'] * 4,
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
        'ret_1': [0.01, 0.02, 0.03, 0.01],
        'y_ret_5': [0.05, 0.06, 0.07, 0.08],
        'vol': [1000000, 1000000, 1000000, 1000000],
        # clean层标记
        'is_st': [0, 0, 1, 0],
        'is_suspended': [0, 1, 0, 0],
        'is_limit_up': [1, 0, 0, 0],
        'is_limit_down': [0, 0, 0, 1],
        'list_days': [1000, 1000, 1000, 50],
        'tradable': [1, 0, 0, 0]
    })
    
    # 模拟stock_basic（实际不会被使用，因为有clean标记）
    stock_basic = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
        'name': ['平安银行', '万科A', '*ST浦发', '邯郸钢铁'],
        'list_date': ['20100101', '20100101', '20100101', '20230101']
    })
    
    # 添加过滤标记（应该直接复用clean层标记）
    result = builder._add_filter_flags(
        df,
        stock_basic,
        None,
        '20230110'
    )
    
    # 验证clean层标记被保留
    assert result[result['ts_code'] == '600000.SH']['is_st'].iloc[0] == 1
    assert result[result['ts_code'] == '000002.SZ']['is_suspended'].iloc[0] == 1
    assert result[result['ts_code'] == '600001.SH']['list_days'].iloc[0] == 50
    
    # 模拟daily_data用于涨跌停标记（实际不会被使用，因为有clean标记）
    daily_data = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
        'trade_date': ['20230110'] * 4,
        'close': [11.0, 12.0, 13.0, 14.0],
        'pct_chg': [10.0, -10.0, 5.0, -10.0]
    })
    
    # 添加涨跌停标记（应该直接复用clean层标记）
    result = builder._add_limit_flags(
        result,
        daily_data,
        None,
        '20230110'
    )
    
    # 验证clean层涨跌停标记被保留
    assert result[result['ts_code'] == '000001.SZ']['is_limit_up'].iloc[0] == 1
    assert result[result['ts_code'] == '600001.SH']['is_limit_down'].iloc[0] == 1
    
    # 应用过滤
    filtered = builder._apply_filters(result)
    
    # ST、停牌、上市不足60天的股票应该被过滤
    assert '000001.SZ' in filtered['ts_code'].values  # 正常，仅涨停不过滤
    assert '000002.SZ' not in filtered['ts_code'].values  # 停牌
    assert '600000.SH' not in filtered['ts_code'].values  # ST
    assert '600001.SH' not in filtered['ts_code'].values  # 上市不足60天
