"""测试行业中性化功能"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.normalization import (
    cross_sectional_zscore,
    industry_neutralization,
)


class TestCrossSectionalZscore:
    """测试截面 Z-Score 标准化"""
    
    def test_global_zscore_basic(self):
        """测试全市场 Z-Score 基本功能"""
        # 创建测试数据
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ'],
            'value': [10.0, 20.0, 30.0, 40.0, 50.0],
            'tradable': [1, 1, 1, 1, 1]
        })
        
        # 应用 zscore
        result = cross_sectional_zscore(
            df,
            columns=['value'],
            group_col=None,
            tradable_col='tradable',
            suffix='_z'
        )
        
        # 检查结果
        assert 'value_z' in result.columns
        assert not result['value_z'].isna().any()
        
        # 检查均值接近0，标准差接近1
        assert abs(result['value_z'].mean()) < 0.01
        assert abs(result['value_z'].std() - 1.0) < 0.01
    
    def test_zscore_with_tradable_filter(self):
        """测试只使用 tradable==1 的样本计算统计量"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ', '000005.SZ'],
            'value': [10.0, 20.0, 30.0, 40.0, 50.0],
            'tradable': [1, 1, 1, 0, 0]  # 后两个不可交易
        })
        
        result = cross_sectional_zscore(
            df,
            columns=['value'],
            group_col=None,
            tradable_col='tradable',
            suffix='_z'
        )
        
        # 统计量应该只基于前3个可交易样本计算
        tradable_mean = df[df['tradable'] == 1]['value'].mean()  # 20.0
        tradable_std = df[df['tradable'] == 1]['value'].std()    # 10.0
        
        # 检查第一个样本的 z-score
        expected_z = (10.0 - tradable_mean) / tradable_std
        assert abs(result.loc[0, 'value_z'] - expected_z) < 0.01
    
    def test_industry_zscore_large_group(self):
        """测试行业内 Z-Score（组内样本数>=5）"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'value': [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
            'industry': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
            'tradable': [1] * 10
        })
        
        result = cross_sectional_zscore(
            df,
            columns=['value'],
            group_col='industry',
            tradable_col='tradable',
            min_group_size=5,
            suffix='_z'
        )
        
        # 每个行业内部应该标准化
        for industry in ['A', 'B']:
            industry_data = result[result['industry'] == industry]
            mean_z = industry_data['value_z'].mean()
            std_z = industry_data['value_z'].std()
            
            # 行业内均值应接近0，标准差接近1
            assert abs(mean_z) < 0.01
            assert abs(std_z - 1.0) < 0.01
    
    def test_industry_zscore_small_group_fallback(self):
        """测试行业内样本数<5时回退到全市场统计"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(8)],
            'value': [10, 20, 30, 40, 50, 15, 25, 35],
            'industry': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B'],  # B行业只有3个样本
            'tradable': [1] * 8
        })
        
        result = cross_sectional_zscore(
            df,
            columns=['value'],
            group_col='industry',
            tradable_col='tradable',
            min_group_size=5,
            suffix='_z'
        )
        
        # 计算全市场统计量
        global_mean = df['value'].mean()
        global_std = df['value'].std()
        
        # 检查 B 行业（样本数<5）的样本是否使用了全市场统计
        b_industry = result[result['industry'] == 'B']
        for idx, row in b_industry.iterrows():
            expected_z = (row['value'] - global_mean) / global_std
            assert abs(row['value_z'] - expected_z) < 0.01


class TestIndustryNeutralization:
    """测试行业中性化函数"""
    
    def test_industry_neutralization_basic(self):
        """测试行业中性化基本功能"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'pe_ttm': [10, 20, 30, 40, 50, 15, 25, 35, 45, 55],
            'pb': [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
            'sw_name': ['农林牧渔', '农林牧渔', '农林牧渔', '农林牧渔', '农林牧渔',
                       '化工', '化工', '化工', '化工', '化工'],
            'tradable': [1] * 10
        })
        
        result = industry_neutralization(
            df,
            columns=['pe_ttm', 'pb'],
            industry_col='sw_name',
            tradable_col='tradable',
            min_group_size=5,
            prefix='neu_',
            inplace=False
        )
        
        # 检查新增列
        assert 'neu_pe_ttm' in result.columns
        assert 'neu_pb' in result.columns
        
        # 原始列应该保留
        assert 'pe_ttm' in result.columns
        assert 'pb' in result.columns
        
        # 每个行业内部应该标准化
        for industry in ['农林牧渔', '化工']:
            industry_data = result[result['sw_name'] == industry]
            assert abs(industry_data['neu_pe_ttm'].mean()) < 0.01
            assert abs(industry_data['neu_pb'].mean()) < 0.01
    
    def test_industry_neutralization_missing_column(self):
        """测试缺少列时抛出错误"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'sw_name': ['农林牧渔', '化工'],
            'tradable': [1, 1]
        })
        
        with pytest.raises(ValueError, match="以下列不存在"):
            industry_neutralization(
                df,
                columns=['pe_ttm'],  # 这个列不存在
                industry_col='sw_name',
                tradable_col='tradable'
            )
    
    def test_industry_neutralization_missing_industry_col(self):
        """测试缺少行业列时抛出错误"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'pe_ttm': [10.0, 20.0],
            'tradable': [1, 1]
        })
        
        with pytest.raises(ValueError, match="行业列 sw_name 不存在"):
            industry_neutralization(
                df,
                columns=['pe_ttm'],
                industry_col='sw_name',  # 这个列不存在
                tradable_col='tradable'
            )
    
    def test_log_total_mv_neutralization(self):
        """测试 log_total_mv 的中性化"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'log_total_mv': [5.0, 6.0, 7.0, 8.0, 9.0, 5.5, 6.5, 7.5, 8.5, 9.5],
            'sw_name': ['农林牧渔', '农林牧渔', '农林牧渔', '农林牧渔', '农林牧渔',
                       '化工', '化工', '化工', '化工', '化工'],
            'tradable': [1] * 10
        })
        
        result = industry_neutralization(
            df,
            columns=['log_total_mv'],
            industry_col='sw_name',
            tradable_col='tradable',
            min_group_size=5,
            prefix='neu_'
        )
        
        assert 'neu_log_total_mv' in result.columns
        assert not result['neu_log_total_mv'].isna().any()
        
        # 验证每个行业内标准化
        for industry in ['农林牧渔', '化工']:
            industry_data = result[result['sw_name'] == industry]
            assert abs(industry_data['neu_log_total_mv'].mean()) < 0.01


class TestIntegrationWithFeatureBuilder:
    """测试与 FeatureBuilder 的集成"""
    
    def test_whitelist_columns(self):
        """测试白名单列是否正确处理"""
        # 创建包含白名单列的数据
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'pe_ttm': np.random.randn(10) * 10 + 20,
            'pb': np.random.randn(10) * 1 + 2,
            'bp': np.random.randn(10) * 0.1 + 0.5,
            'dv_ttm': np.random.randn(10) * 2 + 5,
            'log_total_mv': np.random.randn(10) * 1 + 8,
            'amount_ma20': np.random.randn(10) * 1000 + 5000,
            'turnover_rate': np.random.randn(10) * 2 + 5,
            'volatility_5': np.random.randn(10) * 0.01 + 0.02,
            'volatility_10': np.random.randn(10) * 0.015 + 0.025,
            'volatility_20': np.random.randn(10) * 0.02 + 0.03,
            'net_mf_amount': np.random.randn(10) * 1000,
            'ret_20': np.random.randn(10) * 0.1,
            'ma_deviation_20': np.random.randn(10) * 0.05,
            'sw_name': ['农林牧渔'] * 5 + ['化工'] * 5,
            'tradable': [1] * 10
        })
        
        # 应用中性化
        result = industry_neutralization(
            df,
            columns=[
                'pe_ttm', 'pb', 'bp', 'dv_ttm', 'log_total_mv',
                'amount_ma20', 'turnover_rate', 'volatility_5',
                'volatility_10', 'volatility_20', 'net_mf_amount',
                'ret_20', 'ma_deviation_20'
            ],
            industry_col='sw_name',
            tradable_col='tradable',
            min_group_size=5
        )
        
        # 检查所有白名单列都已中性化
        expected_cols = [
            'neu_pe_ttm', 'neu_pb', 'neu_bp', 'neu_dv_ttm',
            'neu_log_total_mv', 'neu_amount_ma20', 'neu_turnover_rate',
            'neu_volatility_5', 'neu_volatility_10', 'neu_volatility_20',
            'neu_net_mf_amount', 'neu_ret_20', 'neu_ma_deviation_20'
        ]
        
        for col in expected_cols:
            assert col in result.columns, f"缺少中性化列: {col}"
            assert not result[col].isna().any(), f"{col} 包含 NaN 值"
