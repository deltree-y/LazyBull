"""东财人气榜因子模块

将 AKShare 拉取的东方财富个股人气排名数据转化为截面特征。

数据来源：AKShare stock_hot_rank_detail_em（免费）
更新频率：日频
数据起始：约 2021 年
限制：早期 split（2021 前）该特征全部 NaN，XGBoost 自动忽略。

因子说明：
- hot_rank: 当日人气排名（越小越热门）
- hot_rank_chg_5: 5 日排名变动（负值=关注度上升）
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger


HOT_RANK_COLS = [
    "hot_rank",
    "hot_rank_chg_5",
]


def build_hot_rank_lookup_by_date(
    hot_rank_df: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """构建人气榜日频查询表

    Args:
        hot_rank_df: 人气榜 DataFrame，需包含 ts_code, trade_date, hot_rank
        trading_dates: 交易日列表（YYYYMMDD 字符串，已排序）

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, hot_rank, hot_rank_chg_5)}
    """
    if hot_rank_df is None or len(hot_rank_df) == 0:
        logger.warning("人气榜数据为空，跳过")
        return {}

    df = hot_rank_df.copy()
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]
    df["hot_rank"] = pd.to_numeric(df["hot_rank"], errors="coerce")
    df = df.sort_values(["ts_code", "trade_date"])

    # 计算 5 日排名变动
    df["hot_rank_lag_5"] = df.groupby("ts_code")["hot_rank"].shift(5)
    df["hot_rank_chg_5"] = df["hot_rank"] - df["hot_rank_lag_5"]

    # 构建日频查询表
    date_set = set(trading_dates)
    result: Dict[str, pd.DataFrame] = {}

    for trade_date, grp in df.groupby("trade_date"):
        if trade_date not in date_set:
            continue
        keep = ["ts_code", "hot_rank", "hot_rank_chg_5"]
        available = [c for c in keep if c in grp.columns]
        result[trade_date] = grp[available].reset_index(drop=True)

    logger.info(f"人气榜查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result
