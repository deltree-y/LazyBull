"""波动率因子模块

提供基于收益率的波动率计算
"""

import logging
import numpy as np
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


def calculate_volatility(df: pd.DataFrame, ret_col: str = 'ret_1', windows: List[int] = None) -> pd.DataFrame:
    """计算滚动波动率（基于收益率的标准差）
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, ret_col（收益率列）
        ret_col: 收益率列名，默认 'ret_1'
        windows: 窗口列表，默认 [5, 10, 20]
        
    Returns:
        DataFrame，包含 ts_code, trade_date, volatility_{window}
    """
    logger.debug("计算滚动波动率(Optimized)...")
    
    if windows is None:
        windows = [5, 10, 20]
    
    # 1. 排序并备份
    df_calc = df[['ts_code', 'trade_date', ret_col]].copy()
    df_calc.sort_values(['ts_code', 'trade_date'], inplace=True)
    
    # 2. 向量化计算
    gp = df_calc.groupby('ts_code')[ret_col]
    
    for window in windows:
        col_name = f'volatility_{window}'
        # 核心：reset_index(level=0, drop=True) 确保索引对齐
        df_calc[col_name] = gp.rolling(window=window, min_periods=window).std().reset_index(level=0, drop=True)
    
    # 3. 构造返回结果 (result)
    result = df_calc.drop(columns=[ret_col])
    return result
