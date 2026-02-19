"""量能突变因子模块

提供基于成交量的突变特征
"""

import numpy as np
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)


def calculate_volume_burst(
    df: pd.DataFrame,
    vol_ratio_windows: List[int] = None
) -> pd.DataFrame:
    """计算量能突变特征（基于 vol_ratio 的截面 zscore）
    
    使用 vol_ratio_N 进行截面标准化，识别当日成交量相对市场的异常程度
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, vol_ratio_{window}
        vol_ratio_windows: vol_ratio 窗口列表，默认 [5, 10, 20]
        
    Returns:
        DataFrame，包含 ts_code, trade_date, vol_burst_{window}
    """
    logger.debug("计算量能突变特征...")
    if vol_ratio_windows is None:
        vol_ratio_windows = [5, 10, 20]
    
    result = df[['ts_code', 'trade_date']].copy()
    
    for window in vol_ratio_windows:
        vol_ratio_col = f'vol_ratio_{window}'
        vol_burst_col = f'vol_burst_{window}'
        
        if vol_ratio_col in df.columns:
            # 按交易日分组，计算截面 zscore
            zscore = df.groupby('trade_date')[vol_ratio_col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            result[vol_burst_col] = zscore
        else:
            result[vol_burst_col] = np.nan
    
    return result
