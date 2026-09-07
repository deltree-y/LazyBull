"""波动结构因子

B 类因子（6 个）：parkinson_vol_20, vol_of_vol_20, vol_regime_percentile,
garch_persistence, high_low_range_ratio, gap_risk

所有因子通过 @register_risk_factor 装饰器注册到全局注册表。

v2：_prepare_stock_ohlc 改为返回 pivot DataFrame（MultiIndex 列），
全部 6 个因子改为全向量化计算，消除 5000 股 Python 逐股迭代。
"""

from typing import List

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


# ═══════════════════════════════════════════════════════════════
# sigma_daily_20：期末异常亏损任务的严格日历对齐原始波动尺度
# （docs/plans/terminal_loss_risk_model_plan.md 2.4/3.2 契约）
# ═══════════════════════════════════════════════════════════════

def compute_sigma_daily_panel(
    daily_df: pd.DataFrame,
    calendar_dates: List[str],
    window: int = 20,
) -> pd.DataFrame:
    """计算 sigma_daily_20 面板（日期 × 股票），严格日历对齐、未年化。

    与本文件其他因子的 tail(window) 压缩窗口不同：窗口按全市场交易日历对齐，
    截至 T 的最近 window 个日历交易日槽必须全部有有效收益（close_adj 存在且
    能构成收益），停牌缺行不压缩成"最近 window 条有数据记录"；任一缺失即 NaN
    （标记不可用，不靠隐式下限制造标签）。零波动（std=0，如长期一字板）同样
    返回 0，由调用方按"无效"处理（方案 2.4.3）。

    不注册进 cs_train/cs_infer 标准特征流水线：本尺度由 terminal_loss
    训练/数据集构建侧独立计算（与 clean/daily 同源），第一轮交付不改动
    主链路特征 schema；后续接入 shadow 时再评估是否纳入公共因子路径。

    Args:
        daily_df: 长表，需含 ts_code, trade_date, close_adj（复权收盘）
        calendar_dates: 完整交易日历（升序，YYYYMMDD 字符串）
        window: 波动窗口（默认 20 个交易日）

    Returns:
        DataFrame，index=calendar_dates，columns=ts_code，值为样本标准差
        （ddof=1，小数不年化）；窗口不足/缺行为 NaN
    """
    if daily_df is None or len(daily_df) == 0:
        return pd.DataFrame(index=calendar_dates)

    close_pivot = daily_df.pivot_table(
        index='trade_date', columns='ts_code', values='close_adj', aggfunc='last'
    )
    # reindex 到完整日历：停牌缺行成为 NaN 槽位，阻止窗口压缩
    close_pivot = close_pivot.reindex(calendar_dates)
    ret = close_pivot.pct_change()
    sigma = ret.rolling(window, min_periods=window).std()
    return sigma
