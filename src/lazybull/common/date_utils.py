"""日期处理工具函数

统一日期格式转换，避免类型不匹配导致的比较错误。
对跨层数据契约，统一使用 YYYYMMDD 字符串。
"""

from typing import Optional, Union

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


def normalize_to_yyyymmdd(value) -> Optional[str]:
    """将单个日期值规范化为 YYYYMMDD 字符串。

    约定：
    - 无法解析/空值返回 None，不返回字符串 "nan"
    - 仅允许 8 位数字日期
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.strftime("%Y%m%d")

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    text = text.replace("-", "").replace("/", "")
    if len(text) != 8 or not text.isdigit():
        return None
    return text


def normalize_series_to_yyyymmdd(series: pd.Series) -> pd.Series:
    """将 Series 统一规范化为 YYYYMMDD 字符串（无效值为 None）。"""
    return series.map(normalize_to_yyyymmdd)


def calc_holding_trade_days(
    buy_date: str,
    current_date: str,
    trade_dates_list: list,
) -> int:
    """按交易日口径计算持有天数（不含买入当日），回测/纸面共用。

    Args:
        buy_date: 买入日期 YYYYMMDD
        current_date: 当前日期 YYYYMMDD
        trade_dates_list: 开市交易日列表（升序）

    Returns:
        交易日口径持有天数；日期缺失或不在交易日列表中返回 0。
    """
    if not buy_date or not current_date:
        return 0
    try:
        buy_idx = trade_dates_list.index(str(buy_date))
        cur_idx = trade_dates_list.index(str(current_date))
        return max(0, cur_idx - buy_idx)
    except ValueError:
        return 0


def is_recent_date_str(date_str: str, days: int = 3) -> bool:
    """判断日期字符串是否处于"近期可能未发布"窗口内。

    用于龙虎榜等逐日接口的空响应处理：接口数据通常在收盘后晚间才发布，
    过早下载会得到空响应。若把空响应落盘为占位分区，之后 is_data_exists
    检查通过就永远不会重试，造成"假空"永久缓存、真实数据丢失。
    对窗口内的空响应不落盘、延迟重试即可避免。

    Args:
        date_str: 日期字符串 YYYYMMDD
        days: 判定窗口天数（自然日）

    Returns:
        True 表示处于近期窗口内（数据可能未发布）
    """
    try:
        d = to_timestamp(date_str)
    except ValueError:
        return False
    if d is None:
        return False
    return d >= pd.Timestamp.now() - pd.Timedelta(days=days)
