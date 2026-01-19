"""日期处理工具函数

统一日期格式转换，避免类型不匹配导致的比较错误
"""

from typing import Union

import pandas as pd
import numpy as np


def to_trade_date_str(date: Union[str, pd.Timestamp, pd.DatetimeIndex, np.datetime64]) -> str:
    """将日期转换为交易日期字符串格式 YYYYMMDD
    
    Args:
        date: 输入日期，支持多种格式
        
    Returns:
        YYYYMMDD 格式的字符串
        
    Examples:
        >>> to_trade_date_str('20230101')
        '20230101'
        >>> to_trade_date_str('2023-01-01')
        '20230101'
        >>> to_trade_date_str(pd.Timestamp('2023-01-01'))
        '20230101'
    """
    if isinstance(date, str):
        # 如果已经是字符串，标准化格式
        date = date.replace('-', '').replace('/', '')
        if len(date) == 8 and date.isdigit():
            return date
        # 尝试解析其他格式
        try:
            return pd.to_datetime(date).strftime('%Y%m%d')
        except:
            raise ValueError(f"无法解析日期字符串: {date}")
    elif isinstance(date, pd.Timestamp):
        return date.strftime('%Y%m%d')
    elif isinstance(date, (pd.DatetimeIndex, np.datetime64)):
        return pd.Timestamp(date).strftime('%Y%m%d')
    else:
        # 尝试转换为 Timestamp
        try:
            return pd.Timestamp(date).strftime('%Y%m%d')
        except:
            raise ValueError(f"不支持的日期类型: {type(date)}, 值: {date}")


def to_timestamp(date: Union[str, pd.Timestamp, np.datetime64]) -> pd.Timestamp:
    """将日期转换为 pd.Timestamp 对象
    
    Args:
        date: 输入日期，支持多种格式
        
    Returns:
        pd.Timestamp 对象
        
    Examples:
        >>> to_timestamp('20230101')
        Timestamp('2023-01-01 00:00:00')
        >>> to_timestamp('2023-01-01')
        Timestamp('2023-01-01 00:00:00')
    """
    if isinstance(date, pd.Timestamp):
        return date
    elif isinstance(date, str):
        # 尝试 YYYYMMDD 格式
        if len(date) == 8 and date.isdigit():
            return pd.to_datetime(date, format='%Y%m%d')
        # 尝试其他格式
        return pd.to_datetime(date)
    elif isinstance(date, np.datetime64):
        return pd.Timestamp(date)
    else:
        # 尝试转换
        try:
            return pd.Timestamp(date)
        except:
            raise ValueError(f"不支持的日期类型: {type(date)}, 值: {date}")


def normalize_date_column(df: pd.DataFrame, column: str, to_str: bool = True) -> pd.DataFrame:
    """规范化 DataFrame 中的日期列
    
    Args:
        df: 输入 DataFrame
        column: 日期列名
        to_str: True 转换为 YYYYMMDD 字符串，False 转换为 pd.Timestamp
        
    Returns:
        规范化后的 DataFrame（副本）
    """
    df = df.copy()
    
    if column not in df.columns:
        return df
    
    if to_str:
        # 转换为 YYYYMMDD 字符串
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = df[column].dt.strftime('%Y%m%d')
        elif pd.api.types.is_object_dtype(df[column]):
            # 字符串列，标准化格式
            df[column] = df[column].apply(lambda x: to_trade_date_str(x) if pd.notna(x) else x)
    else:
        # 转换为 pd.Timestamp
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = pd.to_datetime(df[column], format='%Y%m%d', errors='coerce')
    
    return df


def normalize_date_columns(df: pd.DataFrame, columns: list, to_str: bool = True) -> pd.DataFrame:
    """规范化 DataFrame 中的多个日期列
    
    Args:
        df: 输入 DataFrame
        columns: 日期列名列表
        to_str: True 转换为 YYYYMMDD 字符串，False 转换为 pd.Timestamp
        
    Returns:
        规范化后的 DataFrame（副本）
    """
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df = normalize_date_column(df, column, to_str=to_str)
    return df
