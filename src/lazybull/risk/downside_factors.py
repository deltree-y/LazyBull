"""下行风险因子

A 类因子（8 个）：downside_vol_20, downside_corr_20, var_95_20, cvar_95_20,
max_drawdown_20, drawdown_duration, skewness_20, kurtosis_20

所有因子通过 @register_risk_factor 装饰器注册到全局注册表。
每个因子是一个独立函数，≤ 80 行，可独立单元测试。
"""

import numpy as np
import pandas as pd
from loguru import logger

from .factor_registry import register_risk_factor


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

_MAX_WINDOW = 60  # 需要的最长历史窗口（交易日）
_EPS = 1e-8


def _prepare_stock_returns(
    daily_adj: pd.DataFrame,
    trade_date: str,
    window: int = 20,
) -> pd.DataFrame:
    """从 daily_adj 中提取各股票在 [trade_date-window+1, trade_date] 的日收益序列。

    返回 DataFrame，列为 ts_code，行为日期（按时间升序），值为 ret_1。

    内置调用级缓存：同一 (trade_date, window) 组合在一次
    compute_all_risk_factors() 调用内复用，避免 6 个 A 类因子重复 pivot。
    """
    # 调用级缓存
    cache = getattr(_prepare_stock_returns, '_cache', None)
    if cache is not None:
        cached_result, cached_key = cache
        if cached_key == (trade_date, window):
            return cached_result

    if daily_adj is None or 'ret_1' not in daily_adj.columns:
        return pd.DataFrame()

    # daily_adj 已由 compute_all_risk_factors 预过滤（≤trade_date 且 ≤252 日/股），
    # 只需再按 ts_code 截尾到 window，无需额外 copy（预过滤已生成独立副本）
    if 'ts_code' not in daily_adj.columns:
        return pd.DataFrame()
    df = daily_adj.groupby('ts_code', sort=False).tail(window)

    # 转置为 ts_code × trade_date 的收益矩阵
    pivot = df.pivot_table(
        values='ret_1', index='trade_date', columns='ts_code',
        aggfunc='first',
    )
    result = pivot.sort_index()

    _prepare_stock_returns._cache = (result, (trade_date, window))
    return result


def _prepare_stock_prices(
    daily_adj: pd.DataFrame,
    trade_date: str,
    window: int = 20,
    price_col: str = 'close_adj',
) -> pd.DataFrame:
    """从 daily_adj 提取各股票的价格序列矩阵。

    内置调用级缓存：同一 (trade_date, window, price_col) 组合复用。
    """
    # 调用级缓存
    cache = getattr(_prepare_stock_prices, '_cache', None)
    if cache is not None:
        cached_result, cached_key = cache
        if cached_key == (trade_date, window, price_col):
            return cached_result

    if daily_adj is None or price_col not in daily_adj.columns:
        return pd.DataFrame()

    # daily_adj 已预过滤，只需 groupby.tail(window)
    if 'ts_code' not in daily_adj.columns:
        return pd.DataFrame()
    df = daily_adj.groupby('ts_code', sort=False).tail(window)

    pivot = df.pivot_table(
        values=price_col, index='trade_date', columns='ts_code',
        aggfunc='first',
    )
    result = pivot.sort_index()

    _prepare_stock_prices._cache = (result, (trade_date, window, price_col))
    return result


def _align_to_df(result_series: pd.Series, df: pd.DataFrame) -> pd.Series:
    """将计算结果对齐到 df 的 ts_code 顺序，缺失填 NaN。"""
    if result_series is None or len(result_series) == 0:
        return pd.Series(np.nan, index=df.index)
    # 确保 index name 为 'ts_code'（dict 构建的 Series 可能缺失 index name）
    aligned = df[['ts_code']].merge(
        result_series.rename_axis('ts_code').reset_index(name='value'),
        on='ts_code', how='left',
    )
    return aligned['value'].reset_index(drop=True)


# ---------------------------------------------------------------------------
# A1. 下行波动率
# ---------------------------------------------------------------------------

@register_risk_factor("downside_vol_20")
def compute_downside_vol_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """计算下行波动率：只统计负收益日的波动率。

    downside_vol = std(neg_returns) over 20 days
    """
    rets = _prepare_stock_returns(daily_adj, kwargs.get('trade_date', ''), window=20)
    if rets.empty:
        return pd.Series(np.nan, index=df.index)

    # 只保留负收益
    neg_rets = rets.where(rets < 0, 0)
    result = neg_rets.std(ddof=0).rename('downside_vol_20')
    return _align_to_df(result, df)


# ---------------------------------------------------------------------------
# A2. 下行相关性（下行 β）
# ---------------------------------------------------------------------------

@register_risk_factor("downside_corr_20")
def compute_downside_corr_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """计算下行相关性：市场下跌日的 stock_ret vs mkt_ret 相关系数。

    downside_corr = corr(stock_ret, mkt_ret | mkt_ret < 0) over 20 days
    """
    rets = _prepare_stock_returns(daily_adj, kwargs.get('trade_date', ''), window=20)
    if rets.empty or len(rets) < 5:
        return pd.Series(np.nan, index=df.index)

    # 市场等权收益
    mkt_ret = rets.mean(axis=1)
    down_mask = mkt_ret < 0

    if down_mask.sum() < 5:
        return pd.Series(np.nan, index=df.index)

    results = {}
    for col in rets.columns:
        stock_r = rets[col]
        valid = stock_r.notna() & mkt_ret.notna() & down_mask
        if valid.sum() >= 5:
            results[col] = stock_r[valid].corr(mkt_ret[valid])
        else:
            results[col] = np.nan

    result = pd.Series(results, name='downside_corr_20')
    return _align_to_df(result, df)


# ---------------------------------------------------------------------------
# A3. 历史 VaR 95%
# ---------------------------------------------------------------------------

@register_risk_factor("var_95_20")
def compute_var_95_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """历史模拟法 95% VaR：20 日收益的第 5 百分位数。"""
    rets = _prepare_stock_returns(daily_adj, kwargs.get('trade_date', ''), window=20)
    if rets.empty:
        return pd.Series(np.nan, index=df.index)

    result = rets.quantile(0.05).rename('var_95_20')
    return _align_to_df(result, df)


# ---------------------------------------------------------------------------
# A4. 条件 VaR 95%（CVaR / Expected Shortfall）
# ---------------------------------------------------------------------------

@register_risk_factor("cvar_95_20")
def compute_cvar_95_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """条件 VaR：收益低于 5% VaR 的那些天的平均收益。"""
    rets = _prepare_stock_returns(daily_adj, kwargs.get('trade_date', ''), window=20)
    if rets.empty:
        return pd.Series(np.nan, index=df.index)

    results = {}
    for col in rets.columns:
        series = rets[col].dropna()
        if len(series) < 10:
            results[col] = np.nan
            continue
        var_95 = series.quantile(0.05)
        tail = series[series <= var_95]
        results[col] = tail.mean() if len(tail) > 0 else var_95

    result = pd.Series(results, name='cvar_95_20')
    return _align_to_df(result, df)


# ---------------------------------------------------------------------------
# A5. 最大回撤
# ---------------------------------------------------------------------------

@register_risk_factor("max_drawdown_20")
def compute_max_drawdown_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """近 20 日最大回撤（基于 close_adj 计算）。"""
    prices = _prepare_stock_prices(daily_adj, kwargs.get('trade_date', ''), window=20)
    if prices.empty:
        return pd.Series(np.nan, index=df.index)

    # max_drawdown = min(price / cummax(price) - 1)
    cummax = prices.cummax()
    drawdowns = prices / cummax - 1
    result = drawdowns.min().rename('max_drawdown_20')
    return _align_to_df(result, df)


# ---------------------------------------------------------------------------
# A6. 回撤持续天数
# ---------------------------------------------------------------------------

@register_risk_factor("drawdown_duration")
def compute_drawdown_duration(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """当前回撤已持续天数（自最近一个最高点以来的天数）。"""
    prices = _prepare_stock_prices(daily_adj, kwargs.get('trade_date', ''), window=60)
    if prices.empty:
        return pd.Series(np.nan, index=df.index)

    results = {}
    for col in prices.columns:
        series = prices[col].dropna()
        if len(series) < 2:
            results[col] = np.nan
            continue
        peak_idx = series.idxmax()
        # 计算从 peak 到最后的天数
        all_dates = series.index.tolist()
        duration = len(all_dates) - 1 - all_dates.index(peak_idx)
        results[col] = float(duration)

    result = pd.Series(results, name='drawdown_duration')
    return _align_to_df(result, df)


# ---------------------------------------------------------------------------
# A7. 收益偏度
# ---------------------------------------------------------------------------

@register_risk_factor("skewness_20")
def compute_skewness_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """近 20 日日收益偏度（负偏 = 暴跌风险高）。"""
    rets = _prepare_stock_returns(daily_adj, kwargs.get('trade_date', ''), window=20)
    if rets.empty:
        return pd.Series(np.nan, index=df.index)

    result = rets.skew().rename('skewness_20')
    return _align_to_df(result, df)


# ---------------------------------------------------------------------------
# A8. 收益峰度
# ---------------------------------------------------------------------------

@register_risk_factor("kurtosis_20")
def compute_kurtosis_20(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """近 20 日日收益峰度（高 = 肥尾风险）。"""
    rets = _prepare_stock_returns(daily_adj, kwargs.get('trade_date', ''), window=20)
    if rets.empty:
        return pd.Series(np.nan, index=df.index)

    result = rets.kurtosis().rename('kurtosis_20')
    return _align_to_df(result, df)
