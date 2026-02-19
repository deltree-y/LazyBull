"""测试行业去均值（demean）功能"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.normalization import industry_demean


class TestIndustryDemean:
    """测试行业去均值函数"""
    
    def test_industry_demean_basic(self):
        """测试行业去均值基本功能"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'y_ret_20': [0.10, 0.20, 0.30, 0.40, 0.50, 0.15, 0.25, 0.35, 0.45, 0.55],
            'ret_20': [0.08, 0.18, 0.28, 0.38, 0.48, 0.13, 0.23, 0.33, 0.43, 0.53],
            'sw_name': ['农林牧渔', '农林牧渔', '农林牧渔', '农林牧渔', '农林牧渔',
                       '化工', '化工', '化工', '化工', '化工'],
            'tradable': [1] * 10
        })
        
        result = industry_demean(
            df,
            columns=['y_ret_20', 'ret_20'],
            industry_col='sw_name',
            tradable_col='tradable',
            min_group_size=5,
            prefix='neu_',
            inplace=False
        )
        
        # 检查新增列
        assert 'neu_y_ret_20' in result.columns
        assert 'neu_ret_20' in result.columns
        
        # 原始列应该保留
        assert 'y_ret_20' in result.columns
        assert 'ret_20' in result.columns
        
        # 每个行业内部应该去均值（均值接近0）
        for industry in ['农林牧渔', '化工']:
            industry_data = result[result['sw_name'] == industry]
            # 使用可交易样本计算均值
            tradable_data = industry_data[industry_data['tradable'] == 1]
            assert abs(tradable_data['neu_y_ret_20'].mean()) < 0.01
            assert abs(tradable_data['neu_ret_20'].mean()) < 0.01
    
    def test_demean_with_tradable_filter(self):
        """测试只使用 tradable==1 的样本计算均值"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(8)],
            'y_ret_20': [0.10, 0.20, 0.30, 0.40, 0.50, 0.15, 0.25, 0.35],
            'sw_name': ['农林牧渔', '农林牧渔', '农林牧渔', '农林牧渔', '农林牧渔',
                       '化工', '化工', '化工'],
            'tradable': [1, 1, 1, 1, 0, 1, 1, 0]  # 第5个和第8个不可交易
        })
        
        result = industry_demean(
            df,
            columns=['y_ret_20'],
            industry_col='sw_name',
            tradable_col='tradable',
            min_group_size=3  # 降低阈值以测试
        )
        
        # 计算农林牧渔行业可交易样本的均值（前4个）
        industry_a_tradable_mean = df[(df['sw_name'] == '农林牧渔') & (df['tradable'] == 1)]['y_ret_20'].mean()
        # 应该是 (0.10 + 0.20 + 0.30 + 0.40) / 4 = 0.25
        assert abs(industry_a_tradable_mean - 0.25) < 0.01
        
        # 检查第一个样本的去均值结果
        expected_neu = 0.10 - industry_a_tradable_mean  # 0.10 - 0.25 = -0.15
        assert abs(result.loc[0, 'neu_y_ret_20'] - expected_neu) < 0.01
    
    def test_demean_small_group_fallback(self):
        """测试行业样本数<5时回退到全市场均值"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(8)],
            'y_ret_20': [0.10, 0.20, 0.30, 0.40, 0.50, 0.15, 0.25, 0.35],
            'sw_name': ['农林牧渔', '农林牧渔', '农林牧渔', '农林牧渔', '农林牧渔',
                       '化工', '化工', '化工'],  # 化工行业只有3个样本
            'tradable': [1] * 8
        })
        
        result = industry_demean(
            df,
            columns=['y_ret_20'],
            industry_col='sw_name',
            tradable_col='tradable',
            min_group_size=5
        )
        
        # 计算全市场均值
        global_mean = df['y_ret_20'].mean()
        
        # 检查化工行业（样本数<5）的样本是否使用了全市场均值
        chemical_industry = result[result['sw_name'] == '化工']
        for idx, row in chemical_industry.iterrows():
            expected_neu = row['y_ret_20'] - global_mean
            assert abs(row['neu_y_ret_20'] - expected_neu) < 0.01
    
    def test_demean_missing_column(self):
        """测试缺少列时抛出错误"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'sw_name': ['农林牧渔', '化工'],
            'tradable': [1, 1]
        })
        
        with pytest.raises(ValueError, match="以下列不存在"):
            industry_demean(
                df,
                columns=['y_ret_20'],  # 这个列不存在
                industry_col='sw_name',
                tradable_col='tradable'
            )
    
    def test_demean_missing_industry_col(self):
        """测试缺少行业列时抛出错误"""
        df = pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'y_ret_20': [0.10, 0.20],
            'tradable': [1, 1]
        })
        
        with pytest.raises(ValueError, match="行业列 sw_name 不存在"):
            industry_demean(
                df,
                columns=['y_ret_20'],
                industry_col='sw_name',  # 这个列不存在
                tradable_col='tradable'
            )
    
    def test_demean_multiple_columns(self):
        """测试同时去均值多个列"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'y_ret_5': np.random.randn(10) * 0.05 + 0.02,
            'y_ret_10': np.random.randn(10) * 0.08 + 0.03,
            'y_ret_20': np.random.randn(10) * 0.12 + 0.05,
            'ret_5': np.random.randn(10) * 0.05 + 0.01,
            'ret_10': np.random.randn(10) * 0.08 + 0.02,
            'ret_20': np.random.randn(10) * 0.12 + 0.04,
            'sw_name': ['农林牧渔'] * 5 + ['化工'] * 5,
            'tradable': [1] * 10
        })
        
        result = industry_demean(
            df,
            columns=['y_ret_5', 'y_ret_10', 'y_ret_20', 'ret_5', 'ret_10', 'ret_20'],
            industry_col='sw_name',
            tradable_col='tradable',
            min_group_size=5
        )
        
        # 检查所有列都已去均值
        expected_cols = [
            'neu_y_ret_5', 'neu_y_ret_10', 'neu_y_ret_20',
            'neu_ret_5', 'neu_ret_10', 'neu_ret_20'
        ]
        
        for col in expected_cols:
            assert col in result.columns, f"缺少去均值列: {col}"
            assert not result[col].isna().any(), f"{col} 包含 NaN 值"
        
        # 验证每个行业内均值接近0
        for industry in ['农林牧渔', '化工']:
            industry_data = result[result['sw_name'] == industry]
            for col in expected_cols:
                mean_val = industry_data[col].mean()
                assert abs(mean_val) < 0.01, f"{industry} {col} 均值不为0: {mean_val}"
    
    def test_demean_prefix(self):
        """测试自定义前缀"""
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(6)],
            'y_ret_20': [0.10, 0.20, 0.30, 0.15, 0.25, 0.35],
            'sw_name': ['农林牧渔', '农林牧渔', '农林牧渔', '化工', '化工', '化工'],
            'tradable': [1] * 6
        })
        
        result = industry_demean(
            df,
            columns=['y_ret_20'],
            industry_col='sw_name',
            tradable_col='tradable',
            min_group_size=3,
            prefix='custom_'
        )
        
        # 检查使用了自定义前缀
        assert 'custom_y_ret_20' in result.columns
        assert 'neu_y_ret_20' not in result.columns


class TestIntegrationWithZScore:
    """测试去均值与Z-Score的集成"""
    
    def test_demean_vs_zscore_naming(self):
        """测试去均值（neu_前缀）和Z-Score（_zscore后缀）命名区分"""
        from src.lazybull.factors.normalization import industry_neutralization
        
        df = pd.DataFrame({
            'ts_code': [f'{i:06d}.SZ' for i in range(10)],
            'y_ret_20': np.random.randn(10) * 0.1 + 0.05,  # 收益率：用去均值
            'pe_ttm': np.random.randn(10) * 10 + 20,       # 估值指标：用Z-Score
            'sw_name': ['农林牧渔'] * 5 + ['化工'] * 5,
            'tradable': [1] * 10
        })
        
        # 去均值（收益率）
        result_demean = industry_demean(
            df,
            columns=['y_ret_20'],
            industry_col='sw_name',
            tradable_col='tradable',
            min_group_size=5,
            prefix='neu_'
        )
        
        # Z-Score（估值指标）
        result_zscore = industry_neutralization(
            df,
            columns=['pe_ttm'],
            industry_col='sw_name',
            tradable_col='tradable',
            min_group_size=5,
            prefix='neu_'  # 函数内部会改为 _zscore 后缀
        )
        
        # 检查命名约定
        assert 'neu_y_ret_20' in result_demean.columns  # 去均值：前缀
        assert 'neu_pe_ttm' in result_zscore.columns    # Z-Score：前缀（后续会改为后缀）
        
        # 验证去均值结果：均值为0，但标准差不为1
        industry_data = result_demean[result_demean['sw_name'] == '农林牧渔']
        assert abs(industry_data['neu_y_ret_20'].mean()) < 0.01
        # 标准差不应该是1（不是Z-Score）
        assert abs(industry_data['neu_y_ret_20'].std() - 1.0) > 0.1
        
        # 验证Z-Score结果：均值为0，标准差为1
        industry_data_zscore = result_zscore[result_zscore['sw_name'] == '农林牧渔']
        assert abs(industry_data_zscore['neu_pe_ttm'].mean()) < 0.01
        assert abs(industry_data_zscore['neu_pe_ttm'].std() - 1.0) < 0.01
