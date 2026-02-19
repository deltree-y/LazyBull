"""动量与加速度因子模块

提供基于价格动量的二阶变化特征
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """计算加速度特征（动量的二阶变化）
    
    定义：acceleration = ret_5 - ret_10
    表示短期动量相对中期动量的变化，捕捉趋势加速或减速
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, ret_5, ret_10
        
    Returns:
        DataFrame，包含 ts_code, trade_date, acceleration
    """
    logger.debug("计算加速度特征...")
    result = df[['ts_code', 'trade_date']].copy()
    
    # 计算加速度
    if 'ret_5' in df.columns and 'ret_10' in df.columns:
        result['acceleration'] = df['ret_5'] - df['ret_10']
    else:
        result['acceleration'] = np.nan
    
    return result
