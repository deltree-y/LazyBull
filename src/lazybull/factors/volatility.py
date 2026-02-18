"""波动率因子模块

提供基于收益率的波动率计算
"""

import numpy as np
import pandas as pd
from typing import List


def calculate_volatility(df: pd.DataFrame, ret_col: str = 'ret_1', windows: List[int] = None) -> pd.DataFrame:
    """计算滚动波动率（基于收益率的标准差）
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, ret_col（收益率列）
        ret_col: 收益率列名，默认 'ret_1'
        windows: 窗口列表，默认 [5, 10, 20]
        
    Returns:
        DataFrame，包含 ts_code, trade_date, volatility_{window}
    """
    if windows is None:
        windows = [5, 10, 20]
    
    result = df[['ts_code', 'trade_date']].copy()
    
    # 按股票分组计算
    grouped = df.sort_values(['ts_code', 'trade_date']).groupby('ts_code')
    
    for window in windows:
        col_name = f'volatility_{window}'
        volatility_values = []
        
        for ts_code, group in grouped:
            group = group.sort_values('trade_date').copy()
            
            # 计算滚动标准差
            vol = group[ret_col].rolling(window=window, min_periods=window).std()
            
            temp_df = pd.DataFrame({
                'ts_code': ts_code,
                'trade_date': group['trade_date'].values,
                col_name: vol.values
            })
            volatility_values.append(temp_df)
        
        if volatility_values:
            vol_df = pd.concat(volatility_values, ignore_index=True)
            result = result.merge(vol_df, on=['ts_code', 'trade_date'], how='left')
        else:
            result[col_name] = np.nan
    
    return result
