"""特征工程工具模块

提供通用的特征处理函数，包括：
- 极值处理（winsorize）
- 对数变换（log1p）
- 标准化（zscore）
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import mstats


def winsorize_series(
    series: pd.Series,
    limits: tuple = (0.01, 0.01),
    nan_policy: str = 'propagate'
) -> pd.Series:
    """对序列进行 winsorize 处理（截断极端值）
    
    Args:
        series: 输入序列
        limits: 截断比例 (下限, 上限)，例如 (0.01, 0.01) 表示截断上下1%极端值
        nan_policy: NaN 处理策略
            - 'propagate': 保留 NaN（默认）
            - 'omit': 忽略 NaN 计算阈值
            - 'raise': NaN 时抛出异常
            
    Returns:
        处理后的序列
        
    Examples:
        >>> s = pd.Series([1, 2, 3, 100, 200])
        >>> winsorize_series(s, limits=(0.2, 0.2))
        # 截断最小和最大20%的值
    """
    if len(series) == 0 or series.isna().all():
        return series
    
    result = mstats.winsorize(
        series,
        limits=limits,
        nan_policy=nan_policy
    )
    
    return pd.Series(result, index=series.index)


def log1p_transform(
    series: pd.Series,
    base: Optional[float] = None
) -> pd.Series:
    """对序列进行 log1p 变换（对数变换）
    
    log1p(x) = log(1 + x)，适用于包含 0 或接近 0 的数据
    
    Args:
        series: 输入序列
        base: 对数底数，None 表示自然对数（默认）
            
    Returns:
        变换后的序列
        
    Examples:
        >>> s = pd.Series([0, 1, 10, 100])
        >>> log1p_transform(s)
        # 返回 [log(1), log(2), log(11), log(101)]
    """
    if base is None:
        return np.log1p(series)
    else:
        return np.log1p(series) / np.log(base)


def zscore_transform(
    series: pd.Series,
    with_mean: bool = True,
    with_std: bool = True,
    ddof: int = 0
) -> pd.Series:
    """对序列进行 z-score 标准化
    
    z = (x - mean) / std
    
    Args:
        series: 输入序列
        with_mean: 是否减去均值
        with_std: 是否除以标准差
        ddof: 标准差计算的自由度，0 表示总体标准差（默认），1 表示样本标准差
            
    Returns:
        标准化后的序列
        
    Examples:
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> zscore_transform(s)
        # 返回标准化后的序列，均值=0，标准差=1
    """
    if len(series) == 0 or series.isna().all():
        return series
    
    result = series.copy()
    
    if with_mean:
        mean = series.mean()
        result = result - mean
    
    if with_std:
        std = series.std(ddof=ddof)
        if std > 1e-10:  # 避免除零
            result = result / std
        else:
            # 标准差为 0，返回全 0
            result = pd.Series(0.0, index=series.index)
    
    return result


def cross_sectional_zscore(
    df: pd.DataFrame,
    value_col: str,
    group_col: Optional[str] = None,
    winsorize_limits: Optional[tuple] = None,
    ddof: int = 0
) -> pd.Series:
    """对 DataFrame 进行截面 z-score 标准化
    
    可选先进行 winsorize 处理，然后在每个 group 内进行标准化
    
    Args:
        df: 输入 DataFrame
        value_col: 要标准化的列名
        group_col: 分组列名（例如 'trade_date'），None 表示全局标准化
        winsorize_limits: winsorize 参数，None 表示不进行 winsorize
        ddof: 标准差计算的自由度
            
    Returns:
        标准化后的序列
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'trade_date': ['20230101', '20230101', '20230102', '20230102'],
        ...     'ts_code': ['A', 'B', 'A', 'B'],
        ...     'return': [0.05, 0.10, -0.02, 0.03]
        ... })
        >>> cross_sectional_zscore(df, 'return', 'trade_date', winsorize_limits=(0.01, 0.01))
        # 每个 trade_date 内分别标准化
    """
    if value_col not in df.columns:
        raise ValueError(f"列 {value_col} 不存在")
    
    values = df[value_col].copy()
    
    # 先 winsorize（如果需要）
    if winsorize_limits is not None:
        if group_col is not None:
            # 按组 winsorize（使用 transform 避免 FutureWarning）
            values = df.groupby(group_col)[value_col].transform(
                lambda x: winsorize_series(x, limits=winsorize_limits)
            )
        else:
            # 全局 winsorize
            values = winsorize_series(values, limits=winsorize_limits)
    
    # 标准化（使用矢量化 transform 方法避免 groupby.apply FutureWarning）
    if group_col is not None:
        # 按组标准化：使用 transform 直接计算 zscore
        # 选择正确的输入数据：如果已 winsorize，使用 values；否则使用原始列
        input_data = values if winsorize_limits is not None else df[value_col]
        
        # 计算组内均值和标准差
        mean = input_data.groupby(df[group_col]).transform('mean')
        std = input_data.groupby(df[group_col]).transform('std', ddof=ddof)
        
        # 计算 zscore
        result = input_data - mean
        # 避免除零
        result = result.where(std > 1e-10, 0.0) / std.where(std > 1e-10, 1.0)
    else:
        # 全局标准化
        if winsorize_limits is not None:
            result = zscore_transform(values, ddof=ddof)
        else:
            result = zscore_transform(df[value_col], ddof=ddof)
    
    return result
