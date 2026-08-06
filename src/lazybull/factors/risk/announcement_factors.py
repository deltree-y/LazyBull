"""公告类低频因子

纳入 v1 的公告类因子（4 种数据源，7 个加工后特征），采用三层加工策略：
- 新鲜度衰减：原始值 × exp(-freshness_days / half_life)
- Delta-on-Update：仅公告日非零
- 分档编码：连续值 → 2~3 档风险等级

因子清单：
  pledge_ratio_decayed, pledge_high_flag, pledge_delta  ← pledge_stat
  unlock_risk_flag, unlock_ratio                          ← share_float
  block_discount_avg_10d, block_discount_days_10d         ← block_trade
  short_balance_change_5, short_sell_ratio_5              ← margin_detail（已有）

所有因子通过 @register_risk_factor 装饰器注册到全局注册表。
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from loguru import logger

from .factor_registry import register_risk_factor

_EPS = 1e-8

# 半衰期（天）
_HALF_LIFE_PLEDGE = 30       # 质押数据
_HALF_LIFE_UNLOCK = 30       # 限售解禁
_HALF_LIFE_BLOCK_TRADE = 10  # 大宗交易


def _align_to_df(result_series: pd.Series, df: pd.DataFrame) -> pd.Series:
    if result_series is None or len(result_series) == 0:
        return pd.Series(np.nan, index=df.index)
    # 确保 index name 为 'ts_code'（dict 构建的 Series 可能缺失 index name）
    aligned = df[['ts_code']].merge(
        result_series.rename_axis('ts_code').reset_index(name='value'),
        on='ts_code', how='left',
    )
    return aligned['value'].reset_index(drop=True)


def _compute_freshness_decay(freshness_days: pd.Series, half_life: int) -> pd.Series:
    """新鲜度衰减权重：exp(-days / half_life)。"""
    return np.exp(-freshness_days.clip(lower=0) / half_life)


# ═══════════════════════════════════════════════════════════════
# 质押因子 (pledge_stat)
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("pledge_ratio_decayed")
def compute_pledge_ratio_decayed(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """质押率 × 新鲜度衰减。

    使用 pledge_stat 数据中的 pledge_ratio 和 freshness_days。
    若 pledge_stat 数据不可用，返回 NaN。
    """
    # 检查是否有质押数据列
    pledge_col = 'pledge_ratio'
    freshness_col = 'pledge_freshness_days'

    if pledge_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    raw_ratio = df[pledge_col].astype(float)
    if freshness_col in df.columns:
        decay = _compute_freshness_decay(df[freshness_col], _HALF_LIFE_PLEDGE)
        return raw_ratio * decay
    return raw_ratio


@register_risk_factor("pledge_high_flag")
def compute_pledge_high_flag(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """质押率分档编码：>50% → 1（高危），30-50% → 0，<30% → -1。

    缺数据时返回 0（视为正常）。
    """
    pledge_col = 'pledge_ratio'
    if pledge_col not in df.columns:
        return pd.Series(0, index=df.index)

    raw_ratio = df[pledge_col].astype(float)
    result = pd.Series(0, index=df.index)
    result[raw_ratio > 0.50] = 1
    result[raw_ratio < 0.30] = -1
    result[raw_ratio.isna()] = 0
    return result


@register_risk_factor("pledge_delta")
def compute_pledge_delta(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """质押率 Delta-on-Update：仅公告日有变化时非零。

    需要 pledge_ratio 和上一期的 pledge_ratio_prev 列。
    若无历史对比数据，返回全 0。
    """
    pledge_col = 'pledge_ratio'
    prev_col = 'pledge_ratio_prev'

    if pledge_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    current = df[pledge_col].astype(float)
    if prev_col in df.columns:
        previous = df[prev_col].astype(float)
        delta = current - previous
        # 只保留实质性变化（>0.5%）
        delta[delta.abs() < 0.005] = 0.0
        return delta.fillna(0.0)
    return pd.Series(0.0, index=df.index)


# ═══════════════════════════════════════════════════════════════
# 限售解禁因子 (share_float)
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("unlock_risk_flag")
def compute_unlock_risk_flag(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """限售解禁风险分档：<30天=2（危险），30-90天=1（关注），>90天或无=0（安全）。

    需要 days_to_unlock 列。
    """
    unlock_col = 'days_to_unlock'
    if unlock_col not in df.columns:
        return pd.Series(0, index=df.index)

    days = df[unlock_col].astype(float)
    result = pd.Series(0, index=df.index)
    result[(days > 0) & (days <= 30)] = 2
    result[(days > 30) & (days <= 90)] = 1
    result[days.isna() | (days <= 0) | (days > 90)] = 0
    return result


@register_risk_factor("unlock_ratio")
def compute_unlock_ratio(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """解禁比例：解禁股数 / 流通股数。需要 unlock_ratio 列。"""
    unlock_col = 'unlock_ratio'
    if unlock_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[unlock_col].astype(float)


# ═══════════════════════════════════════════════════════════════
# 大宗交易因子 (block_trade)
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("block_discount_avg_10d")
def compute_block_discount_avg_10d(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """近 10 日大宗交易平均折价率。

    需要 block_trade_discount 和 block_trade_date 列（逐笔数据聚合后的截面）。
    若无数据，返回 NaN。
    """
    discount_col = 'block_discount_avg_10d'
    if discount_col in df.columns:
        return df[discount_col].astype(float)
    return pd.Series(np.nan, index=df.index)


@register_risk_factor("block_discount_days_10d")
def compute_block_discount_days_10d(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """近 10 个交易日中出现大宗交易折价的天数。

    需要 block_discount_days 列。若无数据，返回 0。
    """
    days_col = 'block_discount_days_10d'
    if days_col in df.columns:
        return df[days_col].astype(float).fillna(0)
    return pd.Series(0, index=df.index)


# ═══════════════════════════════════════════════════════════════
# 融券因子 (margin_detail — 已下载)
# ═══════════════════════════════════════════════════════════════

@register_risk_factor("short_balance_change_5")
def compute_short_balance_change_5(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """融券余额 5 日变化率。

    需要 short_balance 和 short_balance_5d_ago 列（或从 margin_detail 预计算）。
    若无数据，返回 NaN。
    """
    if 'short_balance_change_5' in df.columns:
        return df['short_balance_change_5'].astype(float)
    # 尝试从 short_balance 计算
    if 'short_balance' in df.columns and 'short_balance_prev5' in df.columns:
        cur = df['short_balance'].astype(float)
        prev = df['short_balance_prev5'].astype(float)
        return (cur - prev) / (prev.abs() + _EPS)
    return pd.Series(np.nan, index=df.index)


@register_risk_factor("short_sell_ratio_5")
def compute_short_sell_ratio_5(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """融券卖出量 / 成交量（5 日均值）。

    需要 short_sell_vol 和 vol 列。若无数据，返回 NaN。
    """
    if 'short_sell_ratio_5' in df.columns:
        return df['short_sell_ratio_5'].astype(float)
    return pd.Series(np.nan, index=df.index)
