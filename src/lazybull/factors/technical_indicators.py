"""技术指标因子模块

提供常见技术指标的计算：RSI、KDJ、MACD、布林带等
"""

import numpy as np
import pandas as pd
from typing import List


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """计算RSI（相对强弱指标）
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, close_adj
        window: RSI 窗口，默认14
        
    Returns:
        DataFrame，包含 ts_code, trade_date, rsi_{window}
    """
    result = df[['ts_code', 'trade_date']].copy()
    
    # 按股票分组计算
    grouped = df.sort_values(['ts_code', 'trade_date']).groupby('ts_code')
    
    rsi_values = []
    for ts_code, group in grouped:
        group = group.sort_values('trade_date').copy()
        
        # 计算价格变化
        delta = group['close_adj'].diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # 计算指数移动平均
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        
        # 计算 RS 和 RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        # 保存结果
        temp_df = pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': group['trade_date'].values,
            f'rsi_{window}': rsi.values
        })
        rsi_values.append(temp_df)
    
    if rsi_values:
        result = pd.concat(rsi_values, ignore_index=True)
    else:
        result[f'rsi_{window}'] = np.nan
    
    return result


def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """计算KDJ指标
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, high_adj, low_adj, close_adj
        n: KDJ 的 N 参数，默认9
        m1: K 的平滑参数，默认3
        m2: D 的平滑参数，默认3
        
    Returns:
        DataFrame，包含 ts_code, trade_date, kdj_k, kdj_d, kdj_j
    """
    result = df[['ts_code', 'trade_date']].copy()
    
    # 按股票分组计算
    grouped = df.sort_values(['ts_code', 'trade_date']).groupby('ts_code')
    
    kdj_values = []
    for ts_code, group in grouped:
        group = group.sort_values('trade_date').copy()
        
        # 计算 RSV (Raw Stochastic Value)
        low_n = group['low_adj'].rolling(window=n, min_periods=n).min()
        high_n = group['high_adj'].rolling(window=n, min_periods=n).max()
        
        rsv = 100 * (group['close_adj'] - low_n) / (high_n - low_n).replace(0, np.nan)
        
        # 计算 K, D, J
        k = rsv.ewm(com=m1 - 1, adjust=False).mean()
        d = k.ewm(com=m2 - 1, adjust=False).mean()
        j = 3 * k - 2 * d
        
        # 保存结果
        temp_df = pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': group['trade_date'].values,
            'kdj_k': k.values,
            'kdj_d': d.values,
            'kdj_j': j.values
        })
        kdj_values.append(temp_df)
    
    if kdj_values:
        result = pd.concat(kdj_values, ignore_index=True)
    else:
        result['kdj_k'] = np.nan
        result['kdj_d'] = np.nan
        result['kdj_j'] = np.nan
    
    return result


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """计算MACD指标
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, close_adj
        fast: 快线周期，默认12
        slow: 慢线周期，默认26
        signal: 信号线周期，默认9
        
    Returns:
        DataFrame，包含 ts_code, trade_date, macd_dif, macd_dea, macd_hist
    """
    result = df[['ts_code', 'trade_date']].copy()
    
    # 按股票分组计算
    grouped = df.sort_values(['ts_code', 'trade_date']).groupby('ts_code')
    
    macd_values = []
    for ts_code, group in grouped:
        group = group.sort_values('trade_date').copy()
        
        # 计算快慢EMA
        ema_fast = group['close_adj'].ewm(span=fast, adjust=False).mean()
        ema_slow = group['close_adj'].ewm(span=slow, adjust=False).mean()
        
        # DIF = 快线 - 慢线
        dif = ema_fast - ema_slow
        
        # DEA = DIF的信号线（EMA）
        dea = dif.ewm(span=signal, adjust=False).mean()
        
        # MACD柱 = (DIF - DEA) * 2
        hist = (dif - dea) * 2
        
        # 保存结果
        temp_df = pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': group['trade_date'].values,
            'macd_dif': dif.values,
            'macd_dea': dea.values,
            'macd_hist': hist.values
        })
        macd_values.append(temp_df)
    
    if macd_values:
        result = pd.concat(macd_values, ignore_index=True)
    else:
        result['macd_dif'] = np.nan
        result['macd_dea'] = np.nan
        result['macd_hist'] = np.nan
    
    return result


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """计算布林带指标
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, close_adj
        window: 移动平均窗口，默认20
        num_std: 标准差倍数，默认2.0
        
    Returns:
        DataFrame，包含 ts_code, trade_date, bb_middle, bb_upper, bb_lower, bb_width, bb_pct
    """
    result = df[['ts_code', 'trade_date']].copy()
    
    # 按股票分组计算
    grouped = df.sort_values(['ts_code', 'trade_date']).groupby('ts_code')
    
    bb_values = []
    for ts_code, group in grouped:
        group = group.sort_values('trade_date').copy()
        
        # 计算中轨（移动平均）
        middle = group['close_adj'].rolling(window=window, min_periods=window).mean()
        
        # 计算标准差
        std = group['close_adj'].rolling(window=window, min_periods=window).std()
        
        # 计算上下轨
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        # 带宽 = (上轨 - 下轨) / 中轨
        width = (upper - lower) / middle.replace(0, np.nan)
        
        # %B = (价格 - 下轨) / (上轨 - 下轨)
        pct_b = (group['close_adj'] - lower) / (upper - lower).replace(0, np.nan)
        
        # 保存结果
        temp_df = pd.DataFrame({
            'ts_code': ts_code,
            'trade_date': group['trade_date'].values,
            'bb_middle': middle.values,
            'bb_upper': upper.values,
            'bb_lower': lower.values,
            'bb_width': width.values,
            'bb_pct': pct_b.values
        })
        bb_values.append(temp_df)
    
    if bb_values:
        result = pd.concat(bb_values, ignore_index=True)
    else:
        result['bb_middle'] = np.nan
        result['bb_upper'] = np.nan
        result['bb_lower'] = np.nan
        result['bb_width'] = np.nan
        result['bb_pct'] = np.nan
    
    return result
