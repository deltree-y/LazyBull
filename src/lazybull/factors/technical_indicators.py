"""技术指标因子模块

提供常见技术指标的计算：RSI、KDJ、MACD、布林带等
"""

import numpy as np
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """计算RSI（相对强弱指标）
    
    Args:
        df: DataFrame，需包含 ts_code, trade_date, close_adj
        window: RSI 窗口，默认14
        
    Returns:
        DataFrame，包含 ts_code, trade_date, rsi_{window}
    """
    if False:   #原始实现：逐行计算，效率较低
        col_name = f'rsi_{window}'
        df_calc = df[['ts_code', 'trade_date', 'close_adj']].copy()
        df_calc.sort_values(['ts_code', 'trade_date'], inplace=True)
        
        # 1. 分组计算 Diff
        delta = df_calc.groupby('ts_code')['close_adj'].diff()
        
        # 2. 计算 Gain/Loss
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        # 3. 使用 transform 自动对齐索引，避免 join 报错
        avg_gain = gain.groupby(df_calc['ts_code']).rolling(window=window, min_periods=window).mean().reset_index(level=0, drop=True)
        avg_loss = loss.groupby(df_calc['ts_code']).rolling(window=window, min_periods=window).mean().reset_index(level=0, drop=True)
        
        # 4. 计算 RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df_calc[col_name] = 100 - (100 / (1 + rs))
        
        result = df_calc[['ts_code', 'trade_date', col_name]]
        return result
    else:   #优化实现：全局排序 + 向量化计算，效率更高
        col_name = f'rsi_{window}'
        # 1. 排序并提取 values，转为 Numpy 运算避开索引开销
        df = df.sort_values(['ts_code', 'trade_date'])
        close = df['close_adj'].values
        codes = df['ts_code'].values
        
        # 2. 计算全局差值
        delta = np.zeros_like(close)
        delta[1:] = np.diff(close)
        
        # 3. 处理股票切换处的 delta (不同股票之间不应计算差值)
        # 如果当前行的 ts_code 不等于前一行的，将 delta 设为 0
        mask_new_code = (codes[1:] != codes[:-1])
        delta[1:][mask_new_code] = 0
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # 4. 使用快速滑动平均 (使用 uniform filter 的向量化实现)
        # 这里用 pandas 的 rolling 但不带分组，速度会快很多
        # 或者为了极致性能，我们可以利用卷积/累加和
        def fast_rolling_mean(arr, window):
            # 我们可以利用 cumsum 实现快速滑动平均，或者直接调用无分组的 pandas
            return pd.Series(arr).rolling(window=window, min_periods=window).mean().values

        # 注意：为了处理不同股票的边界，这里仍然需要按组处理，
        # 但我们通过将 gain/loss 直接放回 DataFrame 一次性处理来优化
        df['g'] = gain
        df['l'] = loss
        
        # 优化点：合并 groupby 减少重复分组开销
        grouped = df.groupby('ts_code', sort=False)
        # 使用 engine='cython' 或直接在 Series 上调用
        avg_gain = grouped['g'].transform(lambda x: x.rolling(window).mean())
        avg_loss = grouped['l'].transform(lambda x: x.rolling(window).mean())
        
        # 5. 计算 RSI
        # 这里的 np.where 相当于原来的 replace(0, np.nan) 但更高效
        rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
        df[col_name] = 100 - (100 / (1 + rs))
        
        return df[['ts_code', 'trade_date', col_name]]

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
    # 1. 预处理：只取必要列并排序，确保计算逻辑正确
    # 使用 reset_index 确保我们有一个干净的单层索引用于最后合并
    df_calc = df[['ts_code', 'trade_date', 'high_adj', 'low_adj', 'close_adj']].copy()
    df_calc.sort_values(['ts_code', 'trade_date'], inplace=True)
    
    # 2. 向量化计算 N 日内最高价和最低价
    # 使用 groupby().rolling() 计算后，用 reset_index(level=0, drop=True) 剔除掉多余的 ts_code 索引层级
    # 这样剩下的索引就和 df_calc 的索引完全一致，可以直接赋值
    gp = df_calc.groupby('ts_code', group_keys=False)
    
    low_n = gp['low_adj'].rolling(window=n, min_periods=n).min().reset_index(level=0, drop=True)
    high_n = gp['high_adj'].rolling(window=n, min_periods=n).max().reset_index(level=0, drop=True)
    
    # 3. 计算 RSV
    denom = (high_n - low_n).replace(0, np.nan)
    rsv = 100 * (df_calc['close_adj'] - low_n) / denom
    
    # 填充初始值的 RSV (防止 ewm 报错，通常 KDJ 初始值为 50)
    df_calc['rsv_tmp'] = rsv.fillna(50.0)
    
    # 4. 计算 K, D, J (使用 transform 避免索引冲突)
    # transform 会保持与原 df_calc 相同的索引结构，彻底解决 join 失败问题
    df_calc['kdj_k'] = df_calc.groupby('ts_code')['rsv_tmp'].transform(
        lambda x: x.ewm(com=m1 - 1, adjust=False).mean()
    )
    
    df_calc['kdj_d'] = df_calc.groupby('ts_code')['kdj_k'].transform(
        lambda x: x.ewm(com=m2 - 1, adjust=False).mean()
    )
    
    df_calc['kdj_j'] = 3 * df_calc['kdj_k'] - 2 * df_calc['kdj_d']
    
    # 5. 按照原要求构造返回结果
    result = df_calc[['ts_code', 'trade_date', 'kdj_k', 'kdj_d', 'kdj_j']].copy()
    
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
    
    # 1. 预处理：只取必要列并全局排序（只需排序一次）
    df_calc = df[['ts_code', 'trade_date', 'close_adj']].copy()
    df_calc.sort_values(['ts_code', 'trade_date'], inplace=True)
    
    # 定义分组对象
    gp = df_calc.groupby('ts_code')['close_adj']
    
    # 2. 向量化计算快慢 EMA
    # transform(lambda x: ...) 会保持索引与 df_calc 一致，直接赋值不会报错
    ema_fast = gp.transform(lambda x: x.ewm(span=fast, adjust=False).mean())
    ema_slow = gp.transform(lambda x: x.ewm(span=slow, adjust=False).mean())
    
    # 3. 计算 DIF
    df_calc['macd_dif'] = ema_fast - ema_slow
    
    # 4. 计算 DEA (DIF 的信号线)
    # 针对 DIF 再次进行分组平滑
    df_calc['macd_dea'] = df_calc.groupby('ts_code')['macd_dif'].transform(
        lambda x: x.ewm(span=signal, adjust=False).mean()
    )
    
    # 5. 计算 MACD 柱状图
    df_calc['macd_hist'] = (df_calc['macd_dif'] - df_calc['macd_dea']) * 2
    
    # 6. 整理返回结果，变量名仍为 result
    result = df_calc[['ts_code', 'trade_date', 'macd_dif', 'macd_dea', 'macd_hist']].copy()
    
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
    
    # 1. 预处理：排序并提取核心数据，减少内存负担
    df_calc = df[['ts_code', 'trade_date', 'close_adj']].copy()
    df_calc.sort_values(['ts_code', 'trade_date'], inplace=True)
    
    # 2. 向量化分组滚动计算
    # 使用 reset_index(level=0, drop=True) 剔除 groupby 产生的额外索引层，解决 Join 报错
    gp = df_calc.groupby('ts_code')['close_adj']
    
    # 计算中轨（SMA）
    df_calc['bb_middle'] = gp.rolling(window=window, min_periods=window).mean().reset_index(level=0, drop=True)
    
    # 计算标准差
    rolling_std = gp.rolling(window=window, min_periods=window).std().reset_index(level=0, drop=True)
    
    # 3. 计算上下轨
    df_calc['bb_upper'] = df_calc['bb_middle'] + num_std * rolling_std
    df_calc['bb_lower'] = df_calc['bb_middle'] - num_std * rolling_std
    
    # 4. 计算带宽（Width）和 %B 指标
    # 提前处理分母为 0 的情况
    diff_upper_lower = (df_calc['bb_upper'] - df_calc['bb_lower']).replace(0, np.nan)
    
    df_calc['bb_width'] = diff_upper_lower / df_calc['bb_middle'].replace(0, np.nan)
    df_calc['bb_pct'] = (df_calc['close_adj'] - df_calc['bb_lower']) / diff_upper_lower
    
    # 5. 整理返回结果，变量名为 result
    result = df_calc[['ts_code', 'trade_date', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_pct']].copy()
    
    return result
