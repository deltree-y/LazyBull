"""行业中性化与标准化模块

实现截面 Z-Score 标准化和行业内中性化功能
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def cross_sectional_zscore(
    df: pd.DataFrame,
    columns: List[str],
    group_col: Optional[str] = None,
    tradable_col: str = 'tradable',
    min_group_size: int = 5,
    suffix: str = '_z'
) -> pd.DataFrame:
    """截面 Z-Score 标准化
    
    对指定列进行截面（单日）Z-Score标准化：z = (x - mean) / std
    
    Args:
        df: 输入DataFrame，应包含单日截面数据
        columns: 需要标准化的列名列表
        group_col: 分组列（例如行业），如果指定则在组内进行标准化
        tradable_col: 可交易标记列，只使用该列为1的样本计算统计量
        min_group_size: 最小组内样本数，小于该值时回退到全市场统计
        suffix: 输出列后缀，默认'_z'
        
    Returns:
        添加了标准化列的DataFrame
    """
    result = df.copy()
    
    # 确保tradable列存在
    if tradable_col not in result.columns:
        logger.warning(f"未找到 {tradable_col} 列，将使用全部样本")
        tradable_mask = pd.Series(True, index=result.index)
    else:
        tradable_mask = (result[tradable_col] == 1)
    
    for col in columns:
        if col not in result.columns:
            logger.warning(f"列 {col} 不存在，跳过")
            continue
        
        output_col = f"{col}{suffix}"
        
        # 初始化输出列为NaN
        result[output_col] = np.nan
        
        if group_col is None or group_col not in result.columns:
            # 全市场标准化
            _apply_zscore_global(result, col, output_col, tradable_mask)
        else:
            # 分组标准化（行业内）
            _apply_zscore_by_group(
                result, col, output_col, group_col, 
                tradable_mask, min_group_size
            )
    
    return result


def _apply_zscore_global(
    df: pd.DataFrame,
    input_col: str,
    output_col: str,
    tradable_mask: pd.Series
) -> None:
    """应用全市场 Z-Score（原地修改）
    
    Args:
        df: DataFrame
        input_col: 输入列名
        output_col: 输出列名
        tradable_mask: 可交易样本的mask
    """
    # 获取可交易样本的值
    tradable_values = df.loc[tradable_mask, input_col]
    valid_values = tradable_values.dropna()
    
    if len(valid_values) == 0:
        logger.warning(f"{input_col}: 没有有效的可交易样本")
        return
    
    # 计算全市场统计量
    mean_val = valid_values.mean()
    std_val = valid_values.std()
    
    # 如果标准差为0或NaN，无法标准化
    if pd.isna(std_val) or std_val == 0:
        logger.warning(f"{input_col}: 标准差为0或NaN，无法标准化")
        return
    
    # 对所有样本（包括不可交易的）应用标准化
    df[output_col] = (df[input_col] - mean_val) / std_val


def _apply_zscore_by_group(
    df: pd.DataFrame,
    input_col: str,
    output_col: str,
    group_col: str,
    tradable_mask: pd.Series,
    min_group_size: int
) -> None:
    """应用分组 Z-Score（原地修改）
    
    对每个组内进行Z-Score标准化，如果组内样本数不足，则回退到全市场标准化
    
    Args:
        df: DataFrame
        input_col: 输入列名
        output_col: 输出列名
        group_col: 分组列名
        tradable_mask: 可交易样本的mask
        min_group_size: 最小组内样本数
    """
    # 先计算全市场统计量（用于小组回退）
    tradable_values = df.loc[tradable_mask, input_col]
    valid_values = tradable_values.dropna()
    
    if len(valid_values) == 0:
        logger.warning(f"{input_col}: 没有有效的可交易样本")
        return
    
    global_mean = valid_values.mean()
    global_std = valid_values.std()
    
    if pd.isna(global_std) or global_std == 0:
        logger.warning(f"{input_col}: 全市场标准差为0或NaN，无法标准化")
        return
    
    # 按组进行标准化
    groups = df.groupby(group_col, dropna=False)
    
    for group_name, group_df in groups:
        group_indices = group_df.index
        
        # 获取该组内可交易样本
        group_tradable_mask = tradable_mask[group_indices]
        group_tradable_values = group_df.loc[group_tradable_mask, input_col]
        group_valid_values = group_tradable_values.dropna()
        
        # 判断是否需要回退到全市场
        if len(group_valid_values) < min_group_size:
            # 样本数不足，使用全市场统计量
            df.loc[group_indices, output_col] = (
                group_df[input_col] - global_mean
            ) / global_std
        else:
            # 使用组内统计量
            group_mean = group_valid_values.mean()
            group_std = group_valid_values.std()
            
            # 如果组内标准差为0或NaN，回退到全市场
            if pd.isna(group_std) or group_std == 0:
                df.loc[group_indices, output_col] = (
                    group_df[input_col] - global_mean
                ) / global_std
            else:
                df.loc[group_indices, output_col] = (
                    group_df[input_col] - group_mean
                ) / group_std


def industry_neutralization(
    df: pd.DataFrame,
    columns: List[str],
    industry_col: str = 'sw_name',
    tradable_col: str = 'tradable',
    min_group_size: int = 5,
    prefix: str = 'neu_',
    inplace: bool = False
) -> pd.DataFrame:
    """行业中性化（行业内 Z-Score）
    
    对指定列进行行业内Z-Score标准化，实现行业中性化
    
    Args:
        df: 输入DataFrame，应包含单日截面数据
        columns: 需要中性化的列名列表
        industry_col: 行业列名，默认'sw_name'（申万行业名称）
        tradable_col: 可交易标记列，只使用该列为1的样本计算统计量
        min_group_size: 最小组内样本数，小于该值时回退到全市场统计
        prefix: 输出列前缀，默认'neu_'
        inplace: 是否原地修改（覆盖原列），默认False（新增列）
        
    Returns:
        添加了中性化列的DataFrame
    """
    result = df.copy()
    
    # 检查行业列是否存在
    if industry_col not in result.columns:
        raise ValueError(
            f"行业列 {industry_col} 不存在！\n"
            f"请先下载并加载申万行业分类数据。\n"
            f"运行命令：python scripts/update_basic_data.py --only-shenwan --force"
        )
    
    # 检查列是否存在
    missing_cols = [col for col in columns if col not in result.columns]
    if missing_cols:
        raise ValueError(
            f"以下列不存在：{missing_cols}\n"
            f"请确保这些特征已在 FeatureBuilder 中生成。\n"
            f"常见原因：\n"
            f"  1. log_total_mv: 需要 total_mv 字段（来自 daily_basic）\n"
            f"  2. volatility_x: 需要确认波动率特征的列名\n"
            f"  3. 其他特征：需要相应的数据源（daily_basic、moneyflow 等）"
        )
    
    logger.info(
        f"开始行业中性化：{len(columns)} 个特征，"
        f"行业列={industry_col}，min_group_size={min_group_size}"
    )
    
    # 对每个列进行行业内 Z-Score
    for col in columns:
        if inplace:
            output_col = col
            # 先备份原始列
            backup_col = f"_original_{col}"
            result[backup_col] = result[col].copy()
        else:
            output_col = f"{prefix}{col}"
        
        # 应用行业内 Z-Score
        temp_result = cross_sectional_zscore(
            result,
            columns=[col],
            group_col=industry_col,
            tradable_col=tradable_col,
            min_group_size=min_group_size,
            suffix='_temp'
        )
        
        # 将结果复制到目标列
        result[output_col] = temp_result[f"{col}_temp"]
        
        # 统计使用行业统计和全市场统计的样本数
        if industry_col in result.columns and tradable_col in result.columns:
            tradable_df = result[result[tradable_col] == 1]
            industry_counts = tradable_df.groupby(industry_col)[col].count()
            small_groups = (industry_counts < min_group_size).sum()
            large_groups = (industry_counts >= min_group_size).sum()
            
            logger.debug(
                f"  {col}: {large_groups} 个行业使用行业统计，"
                f"{small_groups} 个行业回退全市场统计"
            )
    
    logger.info(f"行业中性化完成")
    
    return result


def industry_demean(
    df: pd.DataFrame,
    columns: List[str],
    industry_col: str = 'sw_name',
    tradable_col: str = 'tradable',
    min_group_size: int = 5,
    prefix: str = 'neu_',
    inplace: bool = False
) -> pd.DataFrame:
    """行业去均值（demean）中性化
    
    对指定列进行行业去均值：neu_x = x - mean(x within industry same day)
    主要用于收益率/标签列的中性化，消除行业间收益差异
    
    Args:
        df: 输入DataFrame，应包含单日截面数据
        columns: 需要去均值的列名列表（通常是收益率/标签列）
        industry_col: 行业列名，默认'sw_name'（申万行业名称）
        tradable_col: 可交易标记列，只使用该列为1的样本计算均值
        min_group_size: 最小组内样本数，小于该值时回退到全市场均值
        prefix: 输出列前缀，默认'neu_'
        inplace: 是否原地修改（覆盖原列），默认False（新增列）
        
    Returns:
        添加了去均值列的DataFrame
    """
    result = df.copy()
    
    # 检查行业列是否存在
    if industry_col not in result.columns:
        raise ValueError(
            f"行业列 {industry_col} 不存在！\n"
            f"请先下载并加载申万行业分类数据。\n"
            f"运行命令：python scripts/update_basic_data.py --only-shenwan --force"
        )
    
    # 检查列是否存在
    missing_cols = [col for col in columns if col not in result.columns]
    if missing_cols:
        raise ValueError(
            f"以下列不存在：{missing_cols}\n"
            f"请确保这些特征已在 FeatureBuilder 中生成。"
        )
    
    logger.info(
        f"开始行业去均值：{len(columns)} 个列，"
        f"行业列={industry_col}，min_group_size={min_group_size}"
    )
    
    # 确保tradable列存在
    if tradable_col not in result.columns:
        logger.warning(f"未找到 {tradable_col} 列，将使用全部样本")
        tradable_mask = pd.Series(True, index=result.index)
    else:
        tradable_mask = (result[tradable_col] == 1)
    
    # 对每个列进行去均值
    for col in columns:
        if inplace:
            output_col = col
            # 先备份原始列
            backup_col = f"_original_{col}"
            result[backup_col] = result[col].copy()
        else:
            output_col = f"{prefix}{col}"
        
        # 初始化输出列为NaN
        result[output_col] = np.nan
        
        # 计算全市场均值（用于小组回退）
        tradable_values = result.loc[tradable_mask, col]
        valid_values = tradable_values.dropna()
        
        if len(valid_values) == 0:
            logger.warning(f"{col}: 没有有效的可交易样本")
            continue
        
        global_mean = valid_values.mean()
        
        if pd.isna(global_mean):
            logger.warning(f"{col}: 全市场均值为NaN，跳过")
            continue
        
        # 按行业分组进行去均值
        groups = result.groupby(industry_col, dropna=False)
        
        for group_name, group_df in groups:
            group_indices = group_df.index
            
            # 获取该组内可交易样本
            group_tradable_mask = tradable_mask[group_indices]
            group_tradable_values = group_df.loc[group_tradable_mask, col]
            group_valid_values = group_tradable_values.dropna()
            
            # 判断是否需要回退到全市场
            if len(group_valid_values) < min_group_size:
                # 样本数不足，使用全市场均值
                result.loc[group_indices, output_col] = group_df[col] - global_mean
            else:
                # 使用组内均值
                group_mean = group_valid_values.mean()
                
                # 如果组内均值为NaN，回退到全市场
                if pd.isna(group_mean):
                    result.loc[group_indices, output_col] = group_df[col] - global_mean
                else:
                    result.loc[group_indices, output_col] = group_df[col] - group_mean
        
        # 统计使用行业统计和全市场统计的样本数
        if industry_col in result.columns and tradable_col in result.columns:
            tradable_df = result[result[tradable_col] == 1]
            industry_counts = tradable_df.groupby(industry_col)[col].count()
            small_groups = (industry_counts < min_group_size).sum()
            large_groups = (industry_counts >= min_group_size).sum()
            
            logger.debug(
                f"  {col}: {large_groups} 个行业使用行业均值，"
                f"{small_groups} 个行业回退全市场均值"
            )
    
    logger.info(f"行业去均值完成")
    
    return result
