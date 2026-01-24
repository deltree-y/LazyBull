"""机器学习数据预处理模块

提供标签和特征的标准化处理工具，支持：
- 按日横截面去极值（winsorize）
- 按日横截面标准化（z-score）
- 组合处理流程

用于提升模型IC/RankIC的稳定性。
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import mstats


def cross_sectional_winsorize(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = 'trade_date',
    limits: tuple = (0.01, 0.01),
    inplace: bool = False
) -> pd.DataFrame:
    """按日横截面去极值处理（winsorize）
    
    对每个交易日内的样本独立进行去极值处理，截断上下极端值。
    这样可以减少异常值对模型训练的影响，提升稳定性。
    
    Args:
        df: 数据DataFrame
        value_col: 需要处理的列名（如标签列 'y_ret_5'）
        date_col: 日期列名，默认 'trade_date'
        limits: 截断比例 (下界比例, 上界比例)，默认 (0.01, 0.01) 表示截断上下1%
        inplace: 是否原地修改，默认False
        
    Returns:
        处理后的DataFrame
    """
    if not inplace:
        df = df.copy()
    
    if value_col not in df.columns:
        raise ValueError(f"列 {value_col} 不存在")
    
    if date_col not in df.columns:
        raise ValueError(f"日期列 {date_col} 不存在")
    
    # 统计处理信息
    total_dates = df[date_col].nunique()
    processed_count = 0
    
    # 按日期分组处理
    for date, group in df.groupby(date_col):
        # 获取当前日期的索引
        idx = group.index
        
        # 获取原始值
        values = group[value_col].values
        
        # 过滤缺失值
        valid_mask = ~np.isnan(values)
        
        if valid_mask.sum() == 0:
            # 全部为NaN，跳过
            continue
        
        # 进行 winsorize 处理
        # 注意：mstats.winsorize 会保留NaN位置
        winsorized_values = mstats.winsorize(values, limits=limits, nan_policy='propagate')
        
        # 更新DataFrame
        df.loc[idx, value_col] = winsorized_values
        
        processed_count += 1
    
    logger.debug(
        f"横截面 winsorize 完成: 处理 {processed_count}/{total_dates} 个交易日, "
        f"截断比例={limits}"
    )
    
    return df


def cross_sectional_zscore(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = 'trade_date',
    inplace: bool = False
) -> pd.DataFrame:
    """按日横截面标准化（z-score）
    
    对每个交易日内的样本独立进行标准化，使得每日的均值为0，标准差为1。
    这样可以消除不同日期间的尺度差异，提升模型的跨期稳定性。
    
    Args:
        df: 数据DataFrame
        value_col: 需要处理的列名（如标签列 'y_ret_5'）
        date_col: 日期列名，默认 'trade_date'
        inplace: 是否原地修改，默认False
        
    Returns:
        处理后的DataFrame
    """
    if not inplace:
        df = df.copy()
    
    if value_col not in df.columns:
        raise ValueError(f"列 {value_col} 不存在")
    
    if date_col not in df.columns:
        raise ValueError(f"日期列 {date_col} 不存在")
    
    # 统计处理信息
    total_dates = df[date_col].nunique()
    processed_count = 0
    zero_std_count = 0
    
    # 按日期分组处理
    for date, group in df.groupby(date_col):
        # 获取当前日期的索引
        idx = group.index
        
        # 获取原始值
        values = group[value_col].values
        
        # 过滤缺失值
        valid_mask = ~np.isnan(values)
        
        if valid_mask.sum() <= 1:
            # 有效样本数<=1，无法标准化，保持原值
            continue
        
        # 计算均值和标准差（仅基于有效值）
        valid_values = values[valid_mask]
        mean_val = np.mean(valid_values)
        std_val = np.std(valid_values, ddof=1)  # 使用无偏估计
        
        if std_val > 1e-8:
            # 标准差足够大，进行标准化
            # 只标准化有效值，NaN保持不变
            standardized_values = values.copy()
            standardized_values[valid_mask] = (valid_values - mean_val) / std_val
            df.loc[idx, value_col] = standardized_values
            processed_count += 1
        else:
            # 标准差接近0，所有值相同，标准化为0
            standardized_values = values.copy()
            standardized_values[valid_mask] = 0.0
            df.loc[idx, value_col] = standardized_values
            zero_std_count += 1
            processed_count += 1
    
    logger.debug(
        f"横截面 z-score 标准化完成: 处理 {processed_count}/{total_dates} 个交易日, "
        f"其中 {zero_std_count} 个日期标准差为0"
    )
    
    return df


def process_labels_cross_sectional(
    df: pd.DataFrame,
    label_col: str,
    date_col: str = 'trade_date',
    winsorize_limits: tuple = (0.01, 0.01),
    apply_winsorize: bool = True,
    apply_zscore: bool = True,
    inplace: bool = False
) -> pd.DataFrame:
    """按日横截面标签处理流程（winsorize + z-score）
    
    完整的标签预处理流程，用于训练前的数据准备：
    1. （可选）去极值：截断异常值
    2. （可选）标准化：均值0、标准差1
    
    此流程在每个交易日内独立进行，不引入未来信息泄漏。
    
    Args:
        df: 数据DataFrame，需包含 date_col 和 label_col
        label_col: 标签列名，如 'y_ret_5'
        date_col: 日期列名，默认 'trade_date'
        winsorize_limits: 去极值截断比例，默认 (0.01, 0.01)
        apply_winsorize: 是否应用去极值，默认True
        apply_zscore: 是否应用z-score标准化，默认True
        inplace: 是否原地修改，默认False
        
    Returns:
        处理后的DataFrame
    """
    if not inplace:
        df = df.copy()
    
    logger.info(f"开始按日横截面标签处理: {label_col}")
    logger.info(f"  - 去极值(winsorize): {apply_winsorize}, 截断比例={winsorize_limits}")
    logger.info(f"  - 标准化(z-score): {apply_zscore}")
    
    original_count = len(df)
    original_na_count = df[label_col].isna().sum()
    logger.info(f"  - 总样本数: {original_count}, 标签缺失: {original_na_count}")
    
    # 1. 去极值
    if apply_winsorize:
        df = cross_sectional_winsorize(
            df,
            value_col=label_col,
            date_col=date_col,
            limits=winsorize_limits,
            inplace=True
        )
    
    # 2. 标准化
    if apply_zscore:
        df = cross_sectional_zscore(
            df,
            value_col=label_col,
            date_col=date_col,
            inplace=True
        )
    
    # 统计处理后的标签分布
    processed_values = df[label_col].dropna()
    if len(processed_values) > 0:
        logger.info(
            f"标签处理完成: mean={processed_values.mean():.6f}, "
            f"std={processed_values.std():.6f}, "
            f"min={processed_values.min():.4f}, max={processed_values.max():.4f}"
        )
    else:
        logger.warning("处理后无有效标签值")
    
    return df


def validate_cross_sectional_standardization(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = 'trade_date',
    tolerance: float = 0.01
) -> dict:
    """验证横截面标准化效果
    
    检查每个交易日的标签是否接近标准正态分布（均值≈0，标准差≈1）。
    用于测试和诊断。
    
    Args:
        df: 数据DataFrame
        value_col: 检查的列名
        date_col: 日期列名
        tolerance: 容差范围，默认0.01
        
    Returns:
        验证结果字典，包含：
        - is_valid: bool，是否通过验证
        - mean_of_means: 所有日期均值的均值
        - mean_of_stds: 所有日期标准差的均值
        - dates_passed: 通过验证的日期数
        - dates_total: 总日期数
        - invalid_dates: 未通过的日期列表
    """
    results = {
        'is_valid': True,
        'mean_of_means': 0.0,
        'mean_of_stds': 1.0,
        'dates_passed': 0,
        'dates_total': 0,
        'invalid_dates': []
    }
    
    daily_means = []
    daily_stds = []
    
    for date, group in df.groupby(date_col):
        values = group[value_col].dropna()
        
        if len(values) <= 1:
            # 样本数不足，跳过
            continue
        
        results['dates_total'] += 1
        
        mean_val = values.mean()
        std_val = values.std(ddof=1)
        
        daily_means.append(mean_val)
        daily_stds.append(std_val)
        
        # 检查是否接近0和1
        mean_ok = abs(mean_val) < tolerance
        std_ok = abs(std_val - 1.0) < tolerance
        
        if mean_ok and std_ok:
            results['dates_passed'] += 1
        else:
            results['invalid_dates'].append({
                'date': date,
                'mean': mean_val,
                'std': std_val
            })
    
    if len(daily_means) > 0:
        results['mean_of_means'] = np.mean(daily_means)
        results['mean_of_stds'] = np.mean(daily_stds)
    
    # 判断总体是否通过
    if results['dates_total'] > 0:
        pass_rate = results['dates_passed'] / results['dates_total']
        results['is_valid'] = pass_rate >= 0.95  # 95%的日期通过即认为有效
    
    return results
