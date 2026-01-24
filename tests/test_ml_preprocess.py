"""测试机器学习预处理模块

测试按日横截面标签处理功能：
- winsorize去极值
- z-score标准化
- 组合处理流程
- 验证处理效果
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.ml.preprocess import (
    cross_sectional_winsorize,
    cross_sectional_zscore,
    process_labels_cross_sectional,
    validate_cross_sectional_standardization,
)


@pytest.fixture
def sample_data():
    """生成测试用的样本数据
    
    包含3个交易日，每日30只股票，标签值有正常值和极端值
    """
    np.random.seed(42)
    
    dates = ['20230101', '20230102', '20230103']
    n_stocks_per_day = 30
    
    data = []
    for date in dates:
        # 生成30只股票的标签
        # 正常值：均值0，标准差0.05
        normal_labels = np.random.normal(0, 0.05, n_stocks_per_day - 2)
        # 添加2个极端值
        extreme_labels = np.array([0.3, -0.25])  # 极端正负值
        labels = np.concatenate([normal_labels, extreme_labels])
        
        for i, label in enumerate(labels):
            data.append({
                'trade_date': date,
                'ts_code': f'{i:06d}.SZ',
                'y_ret_5': label
            })
    
    df = pd.DataFrame(data)
    return df


def test_cross_sectional_winsorize(sample_data):
    """测试横截面winsorize功能"""
    df = sample_data.copy()
    
    # 记录原始极端值
    original_max = df.groupby('trade_date')['y_ret_5'].max()
    original_min = df.groupby('trade_date')['y_ret_5'].min()
    
    # 执行winsorize
    df_processed = cross_sectional_winsorize(
        df,
        value_col='y_ret_5',
        date_col='trade_date',
        limits=(0.05, 0.05),  # 截断上下5%
        inplace=False
    )
    
    # 验证：处理后的极端值应该被截断
    processed_max = df_processed.groupby('trade_date')['y_ret_5'].max()
    processed_min = df_processed.groupby('trade_date')['y_ret_5'].min()
    
    # 每个日期的最大值应该小于等于原始最大值（截断生效）
    for date in df['trade_date'].unique():
        assert processed_max[date] <= original_max[date]
        assert processed_min[date] >= original_min[date]
    
    # 验证数据未丢失
    assert len(df_processed) == len(df)
    assert df_processed['y_ret_5'].notna().sum() == df['y_ret_5'].notna().sum()


def test_cross_sectional_zscore(sample_data):
    """测试横截面z-score标准化功能"""
    df = sample_data.copy()
    
    # 执行标准化
    df_processed = cross_sectional_zscore(
        df,
        value_col='y_ret_5',
        date_col='trade_date',
        inplace=False
    )
    
    # 验证：每个日期的均值应该接近0，标准差接近1
    for date in df['trade_date'].unique():
        date_data = df_processed[df_processed['trade_date'] == date]['y_ret_5']
        
        mean_val = date_data.mean()
        std_val = date_data.std(ddof=1)
        
        # 允许小的数值误差
        assert abs(mean_val) < 1e-10, f"日期 {date} 均值 {mean_val} 不接近0"
        assert abs(std_val - 1.0) < 1e-10, f"日期 {date} 标准差 {std_val} 不接近1"
    
    # 验证数据未丢失
    assert len(df_processed) == len(df)


def test_process_labels_cross_sectional_full_pipeline(sample_data):
    """测试完整的标签处理流程（winsorize + zscore）"""
    df = sample_data.copy()
    
    # 记录原始统计
    original_stats = {
        'mean': df['y_ret_5'].mean(),
        'std': df['y_ret_5'].std(),
        'max': df['y_ret_5'].max(),
        'min': df['y_ret_5'].min()
    }
    
    # 执行完整处理流程
    df_processed = process_labels_cross_sectional(
        df,
        label_col='y_ret_5',
        date_col='trade_date',
        winsorize_limits=(0.05, 0.05),
        apply_winsorize=True,
        apply_zscore=True,
        inplace=False
    )
    
    # 验证：每个日期处理后的均值≈0，标准差≈1
    for date in df['trade_date'].unique():
        date_data = df_processed[df_processed['trade_date'] == date]['y_ret_5']
        
        mean_val = date_data.mean()
        std_val = date_data.std(ddof=1)
        
        # 标准化后应该接近标准正态分布
        assert abs(mean_val) < 1e-10, f"日期 {date} 均值 {mean_val} 不接近0"
        assert abs(std_val - 1.0) < 1e-10, f"日期 {date} 标准差 {std_val} 不接近1"
    
    # 验证：处理后的总体标准差应该接近1（但不完全等于1，因为按日标准化）
    # 由于按日标准化，总体标准差会略小于1
    processed_std = df_processed['y_ret_5'].std()
    assert 0.9 < processed_std < 1.1, f"总体标准差 {processed_std} 不在合理范围"
    
    # 验证数据完整性
    assert len(df_processed) == len(df)


def test_process_labels_only_winsorize(sample_data):
    """测试仅应用winsorize，不应用zscore"""
    df = sample_data.copy()
    
    df_processed = process_labels_cross_sectional(
        df,
        label_col='y_ret_5',
        date_col='trade_date',
        winsorize_limits=(0.05, 0.05),
        apply_winsorize=True,
        apply_zscore=False,
        inplace=False
    )
    
    # 验证：极端值被截断，但不进行标准化
    for date in df['trade_date'].unique():
        date_data_original = df[df['trade_date'] == date]['y_ret_5']
        date_data_processed = df_processed[df_processed['trade_date'] == date]['y_ret_5']
        
        # 均值和标准差可能变化（因为截断），但不应该标准化到(0,1)
        mean_val = date_data_processed.mean()
        std_val = date_data_processed.std(ddof=1)
        
        # 不应该接近标准正态分布（因为没有zscore）
        # 但应该接近原始分布（除了极端值被截断）
        original_mean = date_data_original.mean()
        
        # 均值应该相对接近（截断对均值影响较小）
        assert abs(mean_val - original_mean) < 0.1


def test_process_labels_only_zscore(sample_data):
    """测试仅应用zscore，不应用winsorize"""
    df = sample_data.copy()
    
    df_processed = process_labels_cross_sectional(
        df,
        label_col='y_ret_5',
        date_col='trade_date',
        apply_winsorize=False,
        apply_zscore=True,
        inplace=False
    )
    
    # 验证：标准化到(0,1)，但极端值未截断
    for date in df['trade_date'].unique():
        date_data = df_processed[df_processed['trade_date'] == date]['y_ret_5']
        
        mean_val = date_data.mean()
        std_val = date_data.std(ddof=1)
        
        # 应该接近标准正态分布
        assert abs(mean_val) < 1e-10
        assert abs(std_val - 1.0) < 1e-10


def test_validate_cross_sectional_standardization(sample_data):
    """测试横截面标准化验证功能"""
    df = sample_data.copy()
    
    # 先进行标准化
    df_processed = cross_sectional_zscore(
        df,
        value_col='y_ret_5',
        date_col='trade_date',
        inplace=True
    )
    
    # 验证标准化效果
    validation_result = validate_cross_sectional_standardization(
        df_processed,
        value_col='y_ret_5',
        date_col='trade_date',
        tolerance=0.01
    )
    
    # 应该通过验证
    assert validation_result['is_valid'] is True
    assert abs(validation_result['mean_of_means']) < 0.01
    assert abs(validation_result['mean_of_stds'] - 1.0) < 0.01
    assert validation_result['dates_passed'] == validation_result['dates_total']
    assert len(validation_result['invalid_dates']) == 0


def test_handle_missing_values():
    """测试处理缺失值的情况"""
    # 创建包含缺失值的数据
    data = {
        'trade_date': ['20230101'] * 10,
        'ts_code': [f'{i:06d}.SZ' for i in range(10)],
        'y_ret_5': [0.01, 0.02, np.nan, 0.03, np.nan, -0.01, -0.02, 0.04, np.nan, 0.00]
    }
    df = pd.DataFrame(data)
    
    # 执行处理
    df_processed = process_labels_cross_sectional(
        df,
        label_col='y_ret_5',
        date_col='trade_date',
        apply_winsorize=True,
        apply_zscore=True,
        inplace=False
    )
    
    # 验证：缺失值位置保持不变
    assert df_processed['y_ret_5'].isna().sum() == df['y_ret_5'].isna().sum()
    
    # 验证：非缺失值被正确标准化
    valid_data = df_processed['y_ret_5'].dropna()
    assert abs(valid_data.mean()) < 1e-10
    assert abs(valid_data.std() - 1.0) < 1e-10


def test_inplace_modification():
    """测试原地修改功能"""
    data = {
        'trade_date': ['20230101'] * 5,
        'ts_code': [f'{i:06d}.SZ' for i in range(5)],
        'y_ret_5': [0.01, 0.02, 0.03, -0.01, -0.02]
    }
    df = pd.DataFrame(data)
    df_copy = df.copy()
    
    # inplace=True
    result = cross_sectional_zscore(df, value_col='y_ret_5', inplace=True)
    
    # 应该返回同一个对象
    assert result is df
    
    # 原始DataFrame应该被修改
    assert not df.equals(df_copy)
    
    # inplace=False
    df2 = df_copy.copy()
    result2 = cross_sectional_zscore(df2, value_col='y_ret_5', inplace=False)
    
    # 应该返回新对象
    assert result2 is not df2
    
    # 原始DataFrame不应该被修改
    assert df2.equals(df_copy)


def test_edge_case_single_stock_per_day():
    """测试边界情况：每天只有一只股票"""
    data = {
        'trade_date': ['20230101', '20230102', '20230103'],
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
        'y_ret_5': [0.05, -0.03, 0.02]
    }
    df = pd.DataFrame(data)
    
    # 执行处理（样本数不足，应该跳过标准化）
    df_processed = cross_sectional_zscore(
        df,
        value_col='y_ret_5',
        date_col='trade_date',
        inplace=False
    )
    
    # 样本数不足时，应该保持原值
    pd.testing.assert_frame_equal(df_processed, df)


def test_edge_case_zero_std():
    """测试边界情况：某天所有值相同（标准差为0）"""
    data = {
        'trade_date': ['20230101'] * 5 + ['20230102'] * 5,
        'ts_code': [f'{i:06d}.SZ' for i in range(10)],
        'y_ret_5': [0.05] * 5 + [0.01, 0.02, 0.03, -0.01, -0.02]  # 第一天全相同
    }
    df = pd.DataFrame(data)
    
    # 执行标准化
    df_processed = cross_sectional_zscore(
        df,
        value_col='y_ret_5',
        date_col='trade_date',
        inplace=False
    )
    
    # 第一天标准差为0，应该全部标准化为0
    day1_data = df_processed[df_processed['trade_date'] == '20230101']['y_ret_5']
    assert (day1_data == 0.0).all()
    
    # 第二天正常
    day2_data = df_processed[df_processed['trade_date'] == '20230102']['y_ret_5']
    assert abs(day2_data.mean()) < 1e-10
    assert abs(day2_data.std() - 1.0) < 1e-10
