"""市场状态特征 —— 从 builder.py 拆出的 _add_market_state_features()。

每日一个标量，广播到所有股票。首次批量预计算并缓存，后续 O(1) 取值。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def add_market_state_features(
    result: pd.DataFrame,
    daily_adj: pd.DataFrame,
    trade_date: str,
    trading_dates: List[str],
    current_idx: int,
    daily_basic_data: Optional[pd.DataFrame] = None,
    market_state_cache: Optional[pd.DataFrame] = None,
    tech_factor_cache: Optional[pd.DataFrame] = None,
    warmup_days: int = 120,
) -> pd.DataFrame:
    """添加市场状态特征（每日一个标量，广播到所有股票）。

    Args:
        result: 当日截面 DataFrame
        daily_adj: 全量后复权日线数据
        trade_date: 目标交易日（YYYYMMDD）
        trading_dates: 已排序的交易日列表
        current_idx: trade_date 在 trading_dates 中的索引
        daily_basic_data: 全量每日指标数据（可选）
        market_state_cache: 预计算的市场状态缓存 DataFrame（index=trade_date）
        tech_factor_cache: 预计算的技术因子缓存（供切片用）
        warmup_days: warmup 天数

    Returns:
        添加了市场状态列的 DataFrame
    """
    from ..factors import compute_market_state_features, precompute_market_state_features

    try:
        if market_state_cache is not None and trade_date in market_state_cache.index:
            row = market_state_cache.loc[trade_date]
            mkt_features = row.to_dict()
        else:
            logger.warning(f"{trade_date} 不在市场状态缓存中，回退到逐日计算")
            mkt_features = compute_market_state_features(
                daily_data=daily_adj,
                trade_date=trade_date,
                trading_dates=trading_dates,
                current_idx=current_idx,
                daily_basic_data=daily_basic_data,
            )
    except Exception as e:
        logger.error(f"计算市场状态特征失败：{e}")
        mkt_features = {
            "mkt_vol_cnt": np.nan,
            "mkt_vol_20": np.nan,
            "mkt_turnover_ratio": np.nan,
            "mkt_ret_avg_20": np.nan,
            "mkt_turnover_std": np.nan,
            "mkt_adv_dec_ratio": np.nan,
            "mkt_ma250_ratio": np.nan,
        }

    for feat_name, feat_val in mkt_features.items():
        result[feat_name] = feat_val

    return result


def precompute_market_state_cache(
    daily_adj: pd.DataFrame,
    trading_dates: List[str],
    trade_date: str,
    daily_basic_data: Optional[pd.DataFrame] = None,
    tech_factor_cache: Optional[pd.DataFrame] = None,
    warmup_days: int = 120,
) -> pd.DataFrame:
    """批量预计算所有交易日市场状态特征。

    Args:
        daily_adj: 全量复权日线数据
        trading_dates: 已排序的交易日列表
        trade_date: 锚点日期（用于 warmup 切片）
        daily_basic_data: 全量每日指标数据
        tech_factor_cache: 技术因子缓存
        warmup_days: warmup 天数

    Returns:
        index=trade_date 的市场状态 DataFrame
    """
    from ..factors import precompute_market_state_features

    logger.info("首次构建：批量预计算所有交易日市场状态特征（缓存中）...")

    sliced_daily_adj = _slice_by_trading_days(daily_adj, trading_dates, trade_date, warmup_days)
    sliced_daily_basic = (
        _slice_by_trading_days(daily_basic_data, trading_dates, trade_date, warmup_days)
        if daily_basic_data is not None
        else None
    )

    return precompute_market_state_features(
        daily_data=sliced_daily_adj,
        trading_dates=trading_dates,
        daily_basic_data=sliced_daily_basic,
        tech_factor_df=tech_factor_cache,
    )


def _slice_by_trading_days(
    daily_df: Optional[pd.DataFrame],
    trading_dates: List[str],
    anchor_trade_date: str,
    warmup_days: int = 120,
) -> Optional[pd.DataFrame]:
    """按交易日历回溯 warmup_days 个交易日切片数据。"""
    if daily_df is None or len(daily_df) == 0:
        return daily_df
    if anchor_trade_date not in trading_dates:
        return daily_df

    anchor_idx = trading_dates.index(anchor_trade_date)
    warmup_start_idx = max(0, anchor_idx - warmup_days)
    window_dates = set(trading_dates[warmup_start_idx:])
    return daily_df[daily_df["trade_date"].isin(window_dates)]
