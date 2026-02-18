"""测试因子模块"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.features import FeatureBuilder


@pytest.fixture
def mock_stock_basic_with_industry():
    """模拟包含行业信息的股票基本信息"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
        'name': ['平安银行', '万科A', '浦发银行', '邯郸钢铁'],
        'list_date': ['20100101', '20100101', '20100101', '20100101'],
        'industry': ['银行', '房地产', '银行', '钢铁']
    })


@pytest.fixture
def mock_daily_data_with_ohlc():
    """模拟包含OHLC的日线行情数据"""
    dates = pd.date_range('2023-01-01', periods=50, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH']
    
    data = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        for i, stock in enumerate(stocks):
            # 模拟价格
            base_price = 10.0 + i
            close = base_price * (1 + 0.001 * np.random.randn())
            open_price = close * (1 + 0.002 * np.random.randn())
            high = max(open_price, close) * (1 + 0.005 * abs(np.random.randn()))
            low = min(open_price, close) * (1 - 0.005 * abs(np.random.randn()))
            pre_close = base_price * 0.99
            pct_chg = ((close - pre_close) / pre_close) * 100
            
            vol = 1000000 + i * 100000
            amount = vol * close
            
            data.append({
                'ts_code': stock,
                'trade_date': date_str,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'pre_close': pre_close,
                'pct_chg': pct_chg,
                'vol': vol,
                'amount': amount
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_adj_factor_extended():
    """模拟扩展的复权因子"""
    dates = pd.date_range('2023-01-01', periods=50, freq='B')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH']
    
    data = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        for stock in stocks:
            data.append({
                'ts_code': stock,
                'trade_date': date_str,
                'adj_factor': 1.0
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_trade_cal_extended():
    """模拟扩展的交易日历"""
    dates = pd.date_range('2023-01-01', periods=50, freq='B')
    
    return pd.DataFrame({
        'exchange': ['SSE'] * len(dates),
        'cal_date': dates.strftime('%Y%m%d').tolist(),
        'is_open': [1] * len(dates)
    })


class TestNewFeatures:
    """测试新增特征"""
    
    def test_amount_ratio_removed(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试 amount_ratio_* 特征已删除"""
        builder = FeatureBuilder(min_list_days=0, require_label=False)
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        trade_date = trading_dates[30]  # 选择中间日期
        
        result = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        # 验证 amount_ratio_* 不存在
        amount_ratio_cols = [col for col in result.columns if col.startswith('amount_ratio_')]
        assert len(amount_ratio_cols) == 0, f"发现 amount_ratio 列: {amount_ratio_cols}"
    
    def test_vol_ma_removed(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试 vol_ma* 特征已删除"""
        builder = FeatureBuilder(min_list_days=0, require_label=False)
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        trade_date = trading_dates[30]
        
        result = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        # 验证 vol_ma* 不存在
        vol_ma_cols = [col for col in result.columns if col.startswith('vol_ma')]
        assert len(vol_ma_cols) == 0, f"发现 vol_ma 列: {vol_ma_cols}"
    
    def test_amount_ma_preserved(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试 amount_ma* 特征保留"""
        builder = FeatureBuilder(min_list_days=0, require_label=False, lookback_windows=[5, 10, 20])
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        trade_date = trading_dates[30]
        
        result = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        # 验证 amount_ma* 存在
        assert 'amount_ma5' in result.columns
        assert 'amount_ma10' in result.columns
        assert 'amount_ma20' in result.columns
    
    def test_amplitude_feature_exists(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试振幅特征存在"""
        builder = FeatureBuilder(min_list_days=0, require_label=False)
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        trade_date = trading_dates[30]
        
        result = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        assert 'amplitude' in result.columns
        # 振幅应该是非负数
        assert (result['amplitude'].dropna() >= 0).all()
    
    def test_shadow_features_exist(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试上下影线特征存在"""
        builder = FeatureBuilder(min_list_days=0, require_label=False)
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        trade_date = trading_dates[30]
        
        result = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        assert 'upper_shadow' in result.columns
        assert 'lower_shadow' in result.columns
        assert 'body_length' in result.columns
    
    def test_volatility_features_exist(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试波动率特征存在"""
        builder = FeatureBuilder(min_list_days=0, require_label=False, lookback_windows=[5, 10, 20])
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        trade_date = trading_dates[30]
        
        result = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        assert 'volatility_5' in result.columns
        assert 'volatility_10' in result.columns
        assert 'volatility_20' in result.columns
    
    def test_industry_features_exist(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试行业特征存在"""
        builder = FeatureBuilder(min_list_days=0, require_label=False)
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        trade_date = trading_dates[30]
        
        result = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        assert 'industry' in result.columns
        assert 'industry_id' in result.columns
        assert 'alpha_industry' in result.columns
        
        # industry_id 应该是正整数
        assert (result['industry_id'] > 0).all()
    
    def test_industry_id_encoding_stable(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试 industry_id 编码稳定性"""
        builder = FeatureBuilder(min_list_days=0, require_label=False)
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        
        # 构建两个不同日期的特征
        result1 = builder.build_features_for_day(
            trade_date=trading_dates[30],
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        result2 = builder.build_features_for_day(
            trade_date=trading_dates[35],
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        # 同一股票在不同日期的 industry_id 应该一致
        for ts_code in result1['ts_code'].unique():
            id1 = result1[result1['ts_code'] == ts_code]['industry_id'].iloc[0]
            id2 = result2[result2['ts_code'] == ts_code]['industry_id'].iloc[0]
            assert id1 == id2, f"{ts_code} 的 industry_id 不一致"
    
    def test_missing_industry_raises_error(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended
    ):
        """测试缺失行业字段时报错"""
        # 创建不包含 industry 的 stock_basic
        stock_basic_no_industry = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'name': ['平安银行', '万科A', '浦发银行', '邯郸钢铁'],
            'list_date': ['20100101', '20100101', '20100101', '20100101']
        })
        
        builder = FeatureBuilder(min_list_days=0, require_label=False)
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        
        # 应该抛出 ValueError
        with pytest.raises(ValueError, match="stock_basic 数据中缺少 'industry' 字段"):
            builder.build_features_for_day(
                trade_date=trading_dates[30],
                trade_cal=mock_trade_cal_extended,
                daily_data=mock_daily_data_with_ohlc,
                adj_factor=mock_adj_factor_extended,
                stock_basic=stock_basic_no_industry
            )
    
    def test_acceleration_feature_exists(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试加速度特征存在"""
        builder = FeatureBuilder(min_list_days=0, require_label=False, lookback_windows=[5, 10, 20])
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        trade_date = trading_dates[30]
        
        result = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        assert 'acceleration' in result.columns
    
    def test_volume_burst_features_exist(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试量能突变特征存在"""
        builder = FeatureBuilder(min_list_days=0, require_label=False, lookback_windows=[5, 10, 20])
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        trade_date = trading_dates[30]
        
        result = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        assert 'vol_burst_5' in result.columns
        assert 'vol_burst_10' in result.columns
        assert 'vol_burst_20' in result.columns
    
    def test_technical_indicators_exist(
        self,
        mock_daily_data_with_ohlc,
        mock_adj_factor_extended,
        mock_trade_cal_extended,
        mock_stock_basic_with_industry
    ):
        """测试技术指标特征存在"""
        builder = FeatureBuilder(min_list_days=0, require_label=False)
        
        trading_dates = builder._get_trading_dates(mock_trade_cal_extended)
        trade_date = trading_dates[40]  # 需要足够历史数据
        
        result = builder.build_features_for_day(
            trade_date=trade_date,
            trade_cal=mock_trade_cal_extended,
            daily_data=mock_daily_data_with_ohlc,
            adj_factor=mock_adj_factor_extended,
            stock_basic=mock_stock_basic_with_industry
        )
        
        # RSI
        assert 'rsi_14' in result.columns
        
        # KDJ
        assert 'kdj_k' in result.columns
        assert 'kdj_d' in result.columns
        assert 'kdj_j' in result.columns
        
        # MACD
        assert 'macd_dif' in result.columns
        assert 'macd_dea' in result.columns
        assert 'macd_hist' in result.columns
        
        # 布林带
        assert 'bb_width' in result.columns
        assert 'bb_pct' in result.columns
