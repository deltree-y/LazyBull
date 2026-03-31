"""波动率因子模块

提供基于收益率的波动率计算，以及 ATR（Average True Range）计算
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


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """计算 ATR（Average True Range，平均真实波幅）

    真实波幅 TR = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = TR 的 window 日简单滚动均值

    Args:
        df: DataFrame，需包含 ts_code, trade_date, high_adj, low_adj, close_adj（后复权价格）
        window: ATR 窗口期，默认 14 日

    Returns:
        DataFrame，包含 ts_code, trade_date, atr_{window}（绝对值，单位：元）、
        atr_pct_{window}（ATR 占当日收盘价比率，无量纲）
    """
    logger.debug(f"计算 ATR（window={window}）...")

    df_calc = df[['ts_code', 'trade_date', 'high_adj', 'low_adj', 'close_adj']].copy()
    df_calc.sort_values(['ts_code', 'trade_date'], inplace=True)

    prev_close = df_calc.groupby('ts_code')['close_adj'].shift(1)

    tr = pd.concat([
        df_calc['high_adj'] - df_calc['low_adj'],
        (df_calc['high_adj'] - prev_close).abs(),
        (df_calc['low_adj'] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df_calc['_tr'] = tr
    atr_col = f'atr_{window}'
    atr_pct_col = f'atr_pct_{window}'
    df_calc[atr_col] = (
        df_calc.groupby('ts_code')['_tr']
        .transform(lambda x: x.rolling(window=window, min_periods=window).mean())
    )
    # atr_pct = atr / close_adj，消除价格量级差异，可直接用于跨股比较
    df_calc[atr_pct_col] = df_calc[atr_col] / df_calc['close_adj'].replace(0, np.nan)

    return df_calc[['ts_code', 'trade_date', atr_col, atr_pct_col]]
