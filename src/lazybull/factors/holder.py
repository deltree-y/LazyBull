"""股东人数因子模块

将不定期公布的股东人数数据（stk_holdernumber）前向填充到日频，
构建每日筹码集中度特征。

数据来源：Tushare stk_holdernumber API（2000 积分）
更新频率：约 10 日一次（随公司公告）
核心逻辑：
- 使用 ann_date（公告日期）做 point-in-time 对齐，防止前视偏差
- 股东人数下降 → 筹码集中 → 看多信号
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from ..common.date_utils import normalize_series_to_yyyymmdd
from .announcement_utils import build_latest_announcement_lookup_by_date


HOLDER_COLS = [
    "holder_num_chg",      # 股东人数环比变动率
    "holder_num_chg_2q",   # 两期变动率
]

HOLDER_FRESHNESS_COL = "holder_freshness_days"


def build_holder_lookup_by_date(
    stk_holdernumber: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """将股东人数数据按 ann_date point-in-time 对齐到日频

    Args:
        stk_holdernumber: 股东人数 DataFrame，需包含
            ts_code, ann_date, end_date, holder_num
        trading_dates: 交易日列表（YYYYMMDD 字符串，已排序）

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, holder_num_chg, ...)}
    """
    df = stk_holdernumber.copy()

    # 日期标准化
    for col in ["ann_date", "end_date"]:
        if col in df.columns:
            df[col] = normalize_series_to_yyyymmdd(df[col])

    df["holder_num"] = pd.to_numeric(df["holder_num"], errors="coerce")
    df = df.dropna(subset=["ann_date", "holder_num"])

    # 仅去除完全重复记录，保留同一报告期多次公告版本，交由 PIT 查询按交易日选择
    df = df.sort_values(["ts_code", "end_date", "ann_date"])
    df = df.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")

    # 按股票+公告日排序，计算环比变动
    df = df.sort_values(["ts_code", "ann_date"])
    grouped = df.groupby("ts_code")
    df["holder_num_prev"] = grouped["holder_num"].shift(1)
    df["holder_num_prev2"] = grouped["holder_num"].shift(2)
    df["holder_num_chg"] = (df["holder_num"] - df["holder_num_prev"]) / df["holder_num_prev"].replace(0, np.nan)
    df["holder_num_chg_2q"] = (df["holder_num"] - df["holder_num_prev2"]) / df["holder_num_prev2"].replace(0, np.nan)

    factor_df = df[["ts_code", "ann_date"] + HOLDER_COLS].copy()
    return build_latest_announcement_lookup_by_date(
        factor_df,
        trading_dates,
        value_cols=HOLDER_COLS,
        freshness_col=HOLDER_FRESHNESS_COL,
        log_name="股东人数",
    )
