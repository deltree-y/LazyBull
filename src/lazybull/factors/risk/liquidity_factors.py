"""流动性风险因子

D 类因子（8 个）：turnover_cv_20, amount_cv_20, amihud_illiq_20,
vol_ratio_5_20, up_down_vol_ratio, volume_climax_days,
turnover_percentile, volume_price_divergence

所有因子通过 @register_risk_factor 装饰器注册到全局注册表。

v2：_prepare_stock_daily 改为返回 pivot DataFrame，全部 8 个因子改为全向量化。
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


def _prepare_stock_daily(
    daily_adj: pd.DataFrame, trade_date: str, window: int = 20,
) -> pd.DataFrame:
    """从 daily_adj 提取各股票历史数据，返回 pivot DataFrame。

    列：MultiIndex [(close_adj, ts_code), (vol, ts_code), ...]
    行：trade_date（时间升序）
    内置调用级缓存：同一 (trade_date, window) 组合复用。
    """
    cache = getattr(_prepare_stock_daily, '_cache', None)
    if cache is not None:
        cached_result, cached_key = cache
        if cached_key == (trade_date, window):
            return cached_result

    if daily_adj is None or len(daily_adj) == 0:
        return pd.DataFrame()

    val_cols = ['close_adj', 'vol', 'amount', 'turnover_rate']
    available = [c for c in val_cols if c in daily_adj.columns]
    if 'ts_code' not in daily_adj.columns or len(available) < 2:
        return pd.DataFrame()

    # 向量化：groupby.tail(window) → pivot（无 Python 循环）
    df = daily_adj.groupby('ts_code', sort=False).tail(window)
    result = df.pivot(index='trade_date', columns='ts_code', values=available)
    result = result.sort_index()

    _prepare_stock_daily._cache = (result, (trade_date, window))
    return result


# ═══════════════════════════════════════════════════════════════
# D1. 换手率变异系数（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("turnover_cv_20")
def compute_turnover_cv_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    pivot = _prepare_stock_daily(daily_adj, kwargs.get('trade_date', ''), window=20)
    if pivot.empty or 'turnover_rate' not in pivot.columns.levels[0]:
        return pd.Series(np.nan, index=df.index)

    turnover = pivot['turnover_rate']
    cv = turnover.std(axis=0) / (turnover.mean(axis=0) + _EPS)
    n_valid = turnover.notna().sum(axis=0)
    cv[n_valid < 10] = np.nan
    return _align_to_df(cv.rename('turnover_cv_20'), df)


# ═══════════════════════════════════════════════════════════════
# D2. 成交额变异系数（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("amount_cv_20")
def compute_amount_cv_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    pivot = _prepare_stock_daily(daily_adj, kwargs.get('trade_date', ''), window=20)
    if pivot.empty or 'amount' not in pivot.columns.levels[0]:
        return pd.Series(np.nan, index=df.index)

    amount = pivot['amount']
    cv = amount.std(axis=0) / (amount.mean(axis=0) + _EPS)
    n_valid = amount.notna().sum(axis=0)
    cv[n_valid < 10] = np.nan
    return _align_to_df(cv.rename('amount_cv_20'), df)


# ═══════════════════════════════════════════════════════════════
# D3. Amihud 非流动性（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("amihud_illiq_20")
def compute_amihud_illiq_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    pivot = _prepare_stock_daily(daily_adj, kwargs.get('trade_date', ''), window=20)
    if pivot.empty or 'amount' not in pivot.columns.levels[0]:
        return pd.Series(np.nan, index=df.index)

    close = pivot['close_adj']
    amount = pivot['amount']
    ret = close.pct_change(fill_method=None)
    # |ret| / amount, skip NaN and zero amount
    amount_safe = amount.clip(lower=_EPS)
    illiq_daily = ret.abs() / amount_safe
    illiq = illiq_daily.mean(axis=0) * 1e6
    n_valid = close.notna().sum(axis=0)
    illiq[n_valid < 10] = np.nan
    return _align_to_df(illiq.rename('amihud_illiq_20'), df)


# ═══════════════════════════════════════════════════════════════
# D4. 量能比率 5/20（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("vol_ratio_5_20")
def compute_vol_ratio_5_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    pivot = _prepare_stock_daily(daily_adj, kwargs.get('trade_date', ''), window=20)
    if pivot.empty or 'vol' not in pivot.columns.levels[0]:
        return pd.Series(np.nan, index=df.index)

    vol = pivot['vol']
    vol_5 = vol.tail(5).mean(axis=0)
    vol_20 = vol.mean(axis=0)
    ratio = vol_5 / (vol_20 + _EPS)
    n_valid = vol.notna().sum(axis=0)
    ratio[n_valid < 10] = np.nan
    return _align_to_df(ratio.rename('vol_ratio_5_20'), df)


# ═══════════════════════════════════════════════════════════════
# D5. 涨跌量比（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("up_down_vol_ratio")
def compute_up_down_vol_ratio(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    pivot = _prepare_stock_daily(daily_adj, kwargs.get('trade_date', ''), window=20)
    if pivot.empty or 'vol' not in pivot.columns.levels[0]:
        return pd.Series(np.nan, index=df.index)

    close = pivot['close_adj']
    vol = pivot['vol']
    ret = close.pct_change(fill_method=None)
    up_vol = vol.where(ret > 0).mean(axis=0)
    down_vol = vol.where(ret < 0).mean(axis=0)
    ratio = up_vol / (down_vol + _EPS)
    n_valid = close.notna().sum(axis=0)
    ratio[n_valid < 10] = np.nan
    return _align_to_df(ratio.rename('up_down_vol_ratio'), df)


# ═══════════════════════════════════════════════════════════════
# D6. 距天量天数（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("volume_climax_days")
def compute_volume_climax_days(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    pivot = _prepare_stock_daily(daily_adj, kwargs.get('trade_date', ''), window=60)
    if pivot.empty or 'vol' not in pivot.columns.levels[0]:
        return pd.Series(np.nan, index=df.index)

    vol = pivot['vol']
    vol_recent = vol.tail(20)
    # argmax along rows (axis=0) gives the index of max per column
    max_pos = vol_recent.fillna(0).values.argmax(axis=0)
    days_since = len(vol_recent) - 1 - max_pos
    result = pd.Series(days_since.astype(float), index=vol_recent.columns,
                       name='volume_climax_days')
    n_valid = vol.notna().sum(axis=0)
    result[n_valid < 10] = np.nan
    return _align_to_df(result, df)


# ═══════════════════════════════════════════════════════════════
# D7. 换手率历史分位（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("turnover_percentile")
def compute_turnover_percentile(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    pivot = _prepare_stock_daily(daily_adj, kwargs.get('trade_date', ''), window=252)
    if pivot.empty or 'turnover_rate' not in pivot.columns.levels[0]:
        return pd.Series(np.nan, index=df.index)

    turnover = pivot['turnover_rate']
    current = turnover.iloc[-1]
    pct = (turnover < current).mean(axis=0)
    n_valid = turnover.notna().sum(axis=0)
    pct[n_valid < 60] = np.nan
    return _align_to_df(pct.rename('turnover_percentile'), df)


# ═══════════════════════════════════════════════════════════════
# D8. 量价背离（向量化）
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("volume_price_divergence")
def compute_volume_price_divergence(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    pivot = _prepare_stock_daily(daily_adj, kwargs.get('trade_date', ''), window=15)
    if pivot.empty or 'vol' not in pivot.columns.levels[0]:
        return pd.Series(np.nan, index=df.index)

    close = pivot['close_adj'].tail(10)
    vol = pivot['vol'].tail(10)
    divergence = close.corrwith(vol)
    n_valid = close.notna().sum(axis=0)
    divergence[n_valid < 8] = np.nan
    return _align_to_df(divergence.rename('volume_price_divergence'), df)
