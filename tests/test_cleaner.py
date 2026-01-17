"""测试数据清洗模块"""

import pandas as pd
import pytest

from src.lazybull.data import DataCleaner


@pytest.fixture
def cleaner():
    """创建数据清洗器实例"""
    return DataCleaner()


@pytest.fixture
def mock_trade_cal_raw():
    """模拟原始交易日历数据"""
    return pd.DataFrame({
        'exchange': ['SSE', 'SSE', 'SSE', 'SSE', 'SSE', 'SSE'],  # 添加重复
        'cal_date': ['20230101', '20230102', '20230103', '2023-01-04', '20230103', '20230104'],  # 混合格式+重复
        'is_open': [0, 1, 1, 1, 1, 1],
        'pretrade_date': ['20221230', '20230101', '20230102', '20230103', '20230102', '20230103']
    })


@pytest.fixture
def mock_stock_basic_raw():
    """模拟原始股票基本信息"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '000001.SZ'],  # 包含重复
        'name': ['平安银行', '万科A', '*ST浦发', '平安银行'],
        'symbol': ['000001', '000002', '600000', '000001'],
        'list_date': ['20100101', '20100101', '20100101', '20100101']
    })


@pytest.fixture
def mock_daily_raw():
    """模拟原始日线行情"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000001.SZ', '000001.SZ'],  # 包含重复
        'trade_date': ['20230102', '20230102', '20230103', '20230102'],  # 包含重复
        'open': [10.0, 11.0, 10.5, 10.0],
        'high': [10.5, 11.5, 11.0, 10.5],
        'low': [9.8, 10.8, 10.2, 9.8],
        'close': [10.2, 11.2, 10.8, 10.2],
        'pre_close': [10.0, 11.0, 10.2, 10.0],
        'pct_chg': [2.0, 1.8, 5.9, 2.0],
        'vol': [1000000, 2000000, 1500000, 1000000],
        'amount': [10200000, 22400000, 16200000, 10200000]
    })


@pytest.fixture
def mock_adj_factor_raw():
    """模拟原始复权因子"""
    return pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000001.SZ'],
        'trade_date': ['20230102', '20230102', '20230103'],
        'adj_factor': [1.0, 1.0, 1.05]
    })


class TestDataCleaner:
    """测试数据清洗器"""
    
    def test_init(self, cleaner):
        """测试初始化"""
        assert cleaner is not None
    
    def test_clean_trade_cal_deduplication(self, cleaner, mock_trade_cal_raw):
        """测试交易日历去重"""
        result = cleaner.clean_trade_cal(mock_trade_cal_raw)
        
        # 检查去重：按 (exchange, cal_date) 主键去重
        assert len(result) == 4  # 原始6条，去重后4条（20230103和20230104各有重复）
        
        # 验证无重复
        assert not result.duplicated(subset=['exchange', 'cal_date']).any()
    
    def test_clean_trade_cal_date_format(self, cleaner, mock_trade_cal_raw):
        """测试交易日历日期格式统一"""
        result = cleaner.clean_trade_cal(mock_trade_cal_raw)
        
        # 检查日期格式统一为 YYYYMMDD
        assert all(len(d) == 8 for d in result['cal_date'])
        assert all(d.isdigit() for d in result['cal_date'])
        
        # 检查混合格式的日期已统一
        assert '20230104' in result['cal_date'].values
    
    def test_clean_trade_cal_sorting(self, cleaner, mock_trade_cal_raw):
        """测试交易日历排序"""
        result = cleaner.clean_trade_cal(mock_trade_cal_raw)
        
        # 检查按 cal_date 排序
        assert list(result['cal_date']) == sorted(result['cal_date'])
    
    def test_clean_stock_basic_deduplication(self, cleaner, mock_stock_basic_raw):
        """测试股票基本信息去重"""
        result = cleaner.clean_stock_basic(mock_stock_basic_raw)
        
        # 检查去重：按 ts_code 主键去重
        assert len(result) == 3  # 原始4条，去重后3条
        assert not result.duplicated(subset=['ts_code']).any()
    
    def test_clean_stock_basic_sorting(self, cleaner, mock_stock_basic_raw):
        """测试股票基本信息排序"""
        result = cleaner.clean_stock_basic(mock_stock_basic_raw)
        
        # 检查按 ts_code 排序
        assert list(result['ts_code']) == sorted(result['ts_code'])
    
    def test_clean_daily_deduplication(self, cleaner, mock_daily_raw, mock_adj_factor_raw):
        """测试日线行情去重"""
        result = cleaner.clean_daily(mock_daily_raw, mock_adj_factor_raw)
        
        # 检查去重：按 (ts_code, trade_date) 主键去重
        assert len(result) == 3  # 原始4条，去重后3条
        assert not result.duplicated(subset=['ts_code', 'trade_date']).any()
    
    def test_clean_daily_adjusted_prices(self, cleaner, mock_daily_raw, mock_adj_factor_raw):
        """测试复权价格计算"""
        result = cleaner.clean_daily(mock_daily_raw, mock_adj_factor_raw)
        
        # 检查复权价格列存在
        assert 'close_adj' in result.columns
        assert 'open_adj' in result.columns
        assert 'high_adj' in result.columns
        assert 'low_adj' in result.columns
        
        # 验证复权价格计算正确
        # 000001.SZ 20230102: close=10.2, adj_factor=1.0 => close_adj=10.2
        row1 = result[(result['ts_code'] == '000001.SZ') & (result['trade_date'] == '20230102')]
        assert abs(row1['close_adj'].iloc[0] - 10.2) < 0.01
        
        # 000001.SZ 20230103: close=10.8, adj_factor=1.05 => close_adj=11.34
        row2 = result[(result['ts_code'] == '000001.SZ') & (result['trade_date'] == '20230103')]
        assert abs(row2['close_adj'].iloc[0] - 10.8 * 1.05) < 0.01
    
    def test_clean_daily_type_conversion(self, cleaner, mock_daily_raw, mock_adj_factor_raw):
        """测试数值类型转换"""
        result = cleaner.clean_daily(mock_daily_raw, mock_adj_factor_raw)
        
        # 检查数值列为 float 类型
        numeric_cols = ['open', 'high', 'low', 'close', 'pct_chg', 'vol', 'amount']
        for col in numeric_cols:
            if col in result.columns:
                assert pd.api.types.is_numeric_dtype(result[col])
    
    def test_clean_daily_sorting(self, cleaner, mock_daily_raw, mock_adj_factor_raw):
        """测试日线行情排序"""
        result = cleaner.clean_daily(mock_daily_raw, mock_adj_factor_raw)
        
        # 检查按 (ts_code, trade_date) 排序
        assert list(result[['ts_code', 'trade_date']].itertuples(index=False, name=None)) == \
               sorted(result[['ts_code', 'trade_date']].itertuples(index=False, name=None))
    
    def test_add_tradable_universe_flag_st_detection(self, cleaner):
        """测试 ST 检测"""
        daily_df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'trade_date': ['20230110'] * 4,
            'close': [10.0, 11.0, 12.0, 13.0],
            'vol': [1000000] * 4,
            'pct_chg': [1.0, 2.0, 3.0, 4.0]
        })
        
        stock_basic_df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'name': ['平安银行', '万科A', '*ST浦发', 'S*ST退市'],
            'list_date': ['20100101'] * 4
        })
        
        result = cleaner.add_tradable_universe_flag(daily_df, stock_basic_df)
        
        # 检查 filter_is_st 标记
        assert result[result['ts_code'] == '000001.SZ']['filter_is_st'].iloc[0] == 0
        assert result[result['ts_code'] == '000002.SZ']['filter_is_st'].iloc[0] == 0
        assert result[result['ts_code'] == '600000.SH']['filter_is_st'].iloc[0] == 1
        assert result[result['ts_code'] == '600001.SH']['filter_is_st'].iloc[0] == 1
    
    def test_add_tradable_universe_flag_suspension_detection(self, cleaner):
        """测试停牌检测"""
        daily_df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
            'trade_date': ['20230110'] * 3,
            'close': [10.0, 11.0, 12.0],
            'vol': [1000000, 0, 1500000],  # 第二只成交量为0
            'pct_chg': [1.0, 0.0, 2.0]
        })
        
        stock_basic_df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
            'name': ['股票A', '股票B', '股票C'],
            'list_date': ['20100101'] * 3
        })
        
        result = cleaner.add_tradable_universe_flag(daily_df, stock_basic_df)
        
        # 检查 filter_is_suspended 标记
        assert result[result['ts_code'] == '000001.SZ']['filter_is_suspended'].iloc[0] == 0
        assert result[result['ts_code'] == '000002.SZ']['filter_is_suspended'].iloc[0] == 1
        assert result[result['ts_code'] == '000003.SZ']['filter_is_suspended'].iloc[0] == 0
    
    def test_add_tradable_universe_flag_limit_detection(self, cleaner):
        """测试涨跌停检测"""
        daily_df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'trade_date': ['20230110'] * 4,
            'close': [11.0, 9.0, 10.5, 10.0],
            'vol': [1000000] * 4,
            'pct_chg': [10.0, -10.0, 5.0, 0.0]  # 涨停、跌停、ST涨停、正常
        })
        
        stock_basic_df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'name': ['股票A', '股票B', '*ST股票C', '股票D'],
            'list_date': ['20100101'] * 4
        })
        
        result = cleaner.add_tradable_universe_flag(daily_df, stock_basic_df)
        
        # 检查涨跌停标记
        assert result[result['ts_code'] == '000001.SZ']['is_limit_up'].iloc[0] == 1
        assert result[result['ts_code'] == '000002.SZ']['is_limit_down'].iloc[0] == 1
        assert result[result['ts_code'] == '600000.SH']['is_limit_up'].iloc[0] == 1  # ST 5%
        assert result[result['ts_code'] == '600001.SH']['is_limit_up'].iloc[0] == 0
    
    def test_add_tradable_universe_flag_tradable_flag(self, cleaner):
        """测试 tradable 标记"""
        daily_df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'trade_date': ['20230110'] * 4,
            'close': [10.0] * 4,
            'vol': [1000000, 0, 1000000, 1000000],
            'pct_chg': [1.0, 0.0, 2.0, 3.0]
        })
        
        stock_basic_df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '600000.SH', '600001.SH'],
            'name': ['股票A', '股票B', '*ST股票C', '股票D'],
            'list_date': ['20100101', '20100101', '20100101', '20230109']  # 最后一个1天前上市
        })
        
        result = cleaner.add_tradable_universe_flag(daily_df, stock_basic_df, min_list_days=2)
        
        # 检查 tradable 标记
        # 000001.SZ: 正常，可交易
        assert result[result['ts_code'] == '000001.SZ']['tradable'].iloc[0] == 1
        # 000002.SZ: 停牌，不可交易
        assert result[result['ts_code'] == '000002.SZ']['tradable'].iloc[0] == 0
        # 600000.SH: ST，不可交易
        assert result[result['ts_code'] == '600000.SH']['tradable'].iloc[0] == 0
        # 600001.SH: 上市不足2天（1天），不可交易
        assert result[result['ts_code'] == '600001.SH']['tradable'].iloc[0] == 0
    
    def test_standardize_date_columns(self, cleaner):
        """测试日期格式标准化"""
        df = pd.DataFrame({
            'date1': ['20230101', '20230102'],
            'date2': ['2023-01-01', '2023-01-02'],
            'date3': pd.to_datetime(['2023-01-01', '2023-01-02'])
        })
        
        result = cleaner._standardize_date_columns(df, ['date1', 'date2', 'date3'])
        
        # 检查所有日期都转换为 YYYYMMDD 格式
        assert all(result['date1'] == ['20230101', '20230102'])
        assert all(result['date2'] == ['20230101', '20230102'])
        assert all(result['date3'] == ['20230101', '20230102'])
    
    def test_deduplicate(self, cleaner):
        """测试去重逻辑"""
        df = pd.DataFrame({
            'key1': ['A', 'A', 'B', 'C', 'C'],
            'key2': [1, 1, 2, 3, 3],
            'value': [10, 20, 30, 40, 50]  # 重复时保留最后一个
        })
        
        result = cleaner._deduplicate(df, ['key1', 'key2'])
        
        # 检查去重后数量
        assert len(result) == 3
        
        # 检查保留最后一条记录
        assert result[(result['key1'] == 'A') & (result['key2'] == 1)]['value'].iloc[0] == 20
        assert result[(result['key1'] == 'C') & (result['key2'] == 3)]['value'].iloc[0] == 50
    
    def test_validate_uniqueness_success(self, cleaner):
        """测试唯一性验证（成功）"""
        df = pd.DataFrame({
            'key1': ['A', 'B', 'C'],
            'key2': [1, 2, 3]
        })
        
        # 不应抛出异常
        cleaner._validate_uniqueness(df, ['key1', 'key2'])
    
    def test_validate_uniqueness_failure(self, cleaner):
        """测试唯一性验证（失败）"""
        df = pd.DataFrame({
            'key1': ['A', 'A', 'B'],
            'key2': [1, 1, 2]
        })
        
        # 应抛出 ValueError
        with pytest.raises(ValueError, match="主键.*存在重复"):
            cleaner._validate_uniqueness(df, ['key1', 'key2'])
    
    def test_clean_daily_with_missing_adj_factor(self, cleaner):
        """测试缺少复权因子的情况"""
        daily_raw = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'trade_date': ['20230102', '20230102'],
            'close': [10.0, 11.0],
            'open': [9.8, 10.8],
            'high': [10.5, 11.5],
            'low': [9.5, 10.5],
            'vol': [1000000, 2000000],
            'amount': [10000000, 22000000],
            'pct_chg': [2.0, 1.8]
        })
        
        # 复权因子只有部分股票
        adj_factor_raw = pd.DataFrame({
            'ts_code': ['000001.SZ'],
            'trade_date': ['20230102'],
            'adj_factor': [1.1]
        })
        
        result = cleaner.clean_daily(daily_raw, adj_factor_raw)
        
        # 检查所有股票都有复权价格（缺失的用1.0填充）
        assert 'close_adj' in result.columns
        assert len(result) == 2
        assert not result['close_adj'].isna().any()
        
        # 验证计算
        assert abs(result[result['ts_code'] == '000001.SZ']['close_adj'].iloc[0] - 10.0 * 1.1) < 0.01
        assert abs(result[result['ts_code'] == '000002.SZ']['close_adj'].iloc[0] - 11.0 * 1.0) < 0.01
    
    def test_clean_daily_with_negative_volume(self, cleaner):
        """测试过滤负成交量"""
        daily_raw = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
            'trade_date': ['20230102'] * 3,
            'close': [10.0, 11.0, 12.0],
            'open': [9.8, 10.8, 11.8],
            'high': [10.5, 11.5, 12.5],
            'low': [9.5, 10.5, 11.5],
            'vol': [1000000, -100, 2000000],  # 第二个为负
            'amount': [10000000, 1100, 24000000],
            'pct_chg': [2.0, 1.0, 3.0]
        })
        
        adj_factor_raw = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
            'trade_date': ['20230102'] * 3,
            'adj_factor': [1.0] * 3
        })
        
        result = cleaner.clean_daily(daily_raw, adj_factor_raw)
        
        # 负成交量的记录应该被过滤
        assert len(result) == 2
        assert '000002.SZ' not in result['ts_code'].values
