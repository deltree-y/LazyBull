"""波动结构因子

B 类因子（6 个）：parkinson_vol_20, vol_of_vol_20, vol_regime_percentile,
garch_persistence, high_low_range_ratio, gap_risk

所有因子通过 @register_risk_factor 装饰器注册到全局注册表。

v2：_prepare_stock_ohlc 改为返回 pivot DataFrame（MultiIndex 列），
全部 6 个因子改为全向量化计算，消除 5000 股 Python 逐股迭代。
"""

import numpy as np
import pandas as pd
from loguru import logger

from .factor_registry import register_risk_factor

_EPS = 1e-8


def _align_to_df(result_series: pd.Series, df: pd.DataFrame) -> pd.Series:
    """将计算结果对齐到 df 的 ts_code 顺序，缺失填 NaN。"""
    if result_series is None or len(result_series) == 0:
        return pd.Series(np.nan, index=df.index)
    aligned = df[['ts_code']].merge(
        result_series.rename_axis('ts_code').reset_index(name='value'),
        on='ts_code', how='left',
    )
    return aligned['value'].reset_index(drop=True)


def _prepare_stock_ohlc(
    daily_adj: pd.DataFrame, trade_date: str, window: int = 20,
) -> pd.DataFrame:
    """从 daily_adj 提取各股票的 OHLC 矩阵，返回 pivot DataFrame。

    列：MultiIndex [(open_adj, ts_code), (high_adj, ts_code), ...]
    行：trade_date（时间升序）
    内置调用级缓存：同一 (trade_date, window) 组合复用。
    """
    cache = getattr(_prepare_stock_ohlc, '_cache', None)
    if cache is not None:
        cached_result, cached_key = cache
        if cached_key == (trade_date, window):
            return cached_result

    if daily_adj is None or len(daily_adj) == 0:
        return pd.DataFrame()

    ohlc_cols = ['open_adj', 'high_adj', 'low_adj', 'close_adj']
    available = [c for c in ohlc_cols if c in daily_adj.columns]
    if len(available) < 4 or 'ts_code' not in daily_adj.columns:
        return pd.DataFrame()

    # 向量化：groupby.tail(window) → pivot（无 Python 循环）
    df = daily_adj.groupby('ts_code', sort=False).tail(window)
    result = df.pivot(index='trade_date', columns='ts_code', values=available)
    result = result.sort_index()

    _prepare_stock_ohlc._cache = (result, (trade_date, window))
    return result


# ═══════════════════════════════════════════════════════════════
# B1. Parkinson 极值波动率（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("parkinson_vol_20")
def compute_parkinson_vol_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    ohlc = _prepare_stock_ohlc(daily_adj, kwargs.get('trade_date', ''), window=20)
    if ohlc.empty:
        return pd.Series(np.nan, index=df.index)

    high = ohlc['high_adj']
    low = ohlc['low_adj']
    n_valid = high.notna().sum(axis=0)
    hi_lo_sq = np.log(high / low.clip(lower=_EPS)) ** 2
    parkinson = np.sqrt(1.0 / (4.0 * n_valid * np.log(2)) * hi_lo_sq.sum(axis=0)) * np.sqrt(252)
    parkinson[n_valid < 5] = np.nan
    return _align_to_df(parkinson.rename('parkinson_vol_20'), df)


# ═══════════════════════════════════════════════════════════════
# B2. 波动率的波动率（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("vol_of_vol_20")
def compute_vol_of_vol_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    ohlc = _prepare_stock_ohlc(daily_adj, kwargs.get('trade_date', ''), window=80)
    if ohlc.empty:
        return pd.Series(np.nan, index=df.index)

    close = ohlc['close_adj']
    daily_ret = close.pct_change(fill_method=None)
    rolling_vol = daily_ret.rolling(20, min_periods=5).std() * np.sqrt(252)
    vol_of_vol = rolling_vol.rolling(60, min_periods=20).std()
    n_valid = close.notna().sum(axis=0)
    result = vol_of_vol.iloc[-1].rename('vol_of_vol_20')
    result[n_valid < 40] = np.nan
    return _align_to_df(result, df)


# ═══════════════════════════════════════════════════════════════
# B3. 波动率分位（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("vol_regime_percentile")
def compute_vol_regime_percentile(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    ohlc = _prepare_stock_ohlc(daily_adj, kwargs.get('trade_date', ''), window=252)
    if ohlc.empty:
        return pd.Series(np.nan, index=df.index)

    close = ohlc['close_adj']
    daily_ret = close.pct_change(fill_method=None)
    rolling_vol = daily_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    current_vol = rolling_vol.iloc[-1]
    pct = (rolling_vol < current_vol).mean(axis=0)
    n_valid = close.notna().sum(axis=0)
    result = pct.rename('vol_regime_percentile')
    result[n_valid < 60] = np.nan
    return _align_to_df(result, df)


# ═══════════════════════════════════════════════════════════════
# B4. GARCH 波动持续性（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("garch_persistence")
def compute_garch_persistence(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    ohlc = _prepare_stock_ohlc(daily_adj, kwargs.get('trade_date', ''), window=80)
    if ohlc.empty:
        return pd.Series(np.nan, index=df.index)

    close = ohlc['close_adj']
    rets = close.pct_change(fill_method=None).iloc[-60:]
    sq_rets = rets ** 2
    # 逐列 corr(sq_ret[1:], sq_ret[:-1]) 用 corrwith 向量化
    sq_lag = sq_rets.shift(1).iloc[1:]
    sq_cur = sq_rets.iloc[1:]
    persistence = sq_cur.corrwith(sq_lag)
    n_valid = close.notna().sum(axis=0)
    result = persistence.rename('garch_persistence')
    result[n_valid < 30] = np.nan
    return _align_to_df(result, df)


# ═══════════════════════════════════════════════════════════════
# B5. 日内振幅均值（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("high_low_range_ratio")
def compute_high_low_range_ratio(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    ohlc = _prepare_stock_ohlc(daily_adj, kwargs.get('trade_date', ''), window=20)
    if ohlc.empty:
        return pd.Series(np.nan, index=df.index)

    range_ratio = (ohlc['high_adj'] - ohlc['low_adj']) / ohlc['close_adj'].clip(lower=_EPS)
    result = range_ratio.mean(axis=0).rename('high_low_range_ratio')
    n_valid = ohlc['high_adj'].notna().sum(axis=0)
    result[n_valid < 5] = np.nan
    return _align_to_df(result, df)


# ═══════════════════════════════════════════════════════════════
# B6. 向下跳空频率（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("gap_risk")
def compute_gap_risk(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    ohlc = _prepare_stock_ohlc(daily_adj, kwargs.get('trade_date', ''), window=21)
    if ohlc.empty:
        return pd.Series(np.nan, index=df.index)

    prev_low = ohlc['low_adj'].shift(1)
    gap_down = ohlc['open_adj'] < prev_low
    result = gap_down.tail(20).mean(axis=0).rename('gap_risk')
    n_valid = ohlc['open_adj'].notna().sum(axis=0)
    result[n_valid < 10] = np.nan
    return _align_to_df(result, df)
