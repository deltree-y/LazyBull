"""K线形态因子模块

提供K线相关的技术因子：振幅、上下影线、实体长度等
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_amplitude(df: pd.DataFrame) -> pd.DataFrame:
    """计算振幅
    
    振幅 = (最高价 - 最低价) / 前收盘价
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, high_adj, low_adj, pre_close, adj_factor
            或者包含 pre_close_adj
        
    Returns:
        DataFrame，包含 ts_code, trade_date, amplitude
    """
    logging.debug("计算振幅...")
    result = df[['ts_code', 'trade_date']].copy()
    
    # 计算前收盘复权价
    if 'pre_close_adj' in df.columns:
        pre_close_adj = df['pre_close_adj']
    else:
        # 使用 pre_close * adj_factor 近似
        pre_close_adj = df['pre_close'] * df['adj_factor']
    
    # 振幅 = (high_adj - low_adj) / pre_close_adj
    result['amplitude'] = np.where(
        pre_close_adj > 1e-6,
        (df['high_adj'] - df['low_adj']) / pre_close_adj,
        np.nan
    )
    
    return result


def calculate_shadows(df: pd.DataFrame) -> pd.DataFrame:
    """计算上下影线和实体长度
    
    - 上影线 = (最高价 - max(开盘价, 收盘价)) / 收盘价
    - 下影线 = (min(开盘价, 收盘价) - 最低价) / 收盘价
    - 实体长度 = abs(收盘价 - 开盘价) / 收盘价
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, open_adj, high_adj, low_adj, close_adj
        
    Returns:
        DataFrame，包含 ts_code, trade_date, upper_shadow, lower_shadow, body_length
    """
    logger.debug("计算上下影线和实体长度...")
    result = df[['ts_code', 'trade_date']].copy()
    
    # 计算K线上下端点
    body_high = np.maximum(df['open_adj'], df['close_adj'])
    body_low = np.minimum(df['open_adj'], df['close_adj'])
    
    # 上影线比例
    result['upper_shadow'] = np.where(
        df['close_adj'] > 1e-6,
        (df['high_adj'] - body_high) / df['close_adj'],
        np.nan
    )
    
    # 下影线比例
    result['lower_shadow'] = np.where(
        df['close_adj'] > 1e-6,
        (body_low - df['low_adj']) / df['close_adj'],
        np.nan
    )
    
    # 实体长度比例
    result['body_length'] = np.where(
        df['close_adj'] > 1e-6,
        np.abs(df['close_adj'] - df['open_adj']) / df['close_adj'],
        np.nan
    )
    
    return result
