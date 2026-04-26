"""股东人数因子模块

将不定期公布的股东人数数据（stk_holdernumber）前向填充到日频，
构建每日筹码集中度特征。

数据来源：Tushare stk_holdernumber API（2000 积分）
更新频率：约 10 日一次（随公司公告）
核心逻辑：
- 使用 ann_date（公告日期）做 point-in-time 对齐，防止前视偏差
- 股东人数下降 → 筹码集中 → 看多信号
"""

import bisect
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger


HOLDER_COLS = [
    "holder_num_chg",      # 股东人数环比变动率
    "holder_num_chg_2q",   # 两期变动率
]


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
            df[col] = df[col].astype(str).str.replace("-", "").str[:8]

    df["holder_num"] = pd.to_numeric(df["holder_num"], errors="coerce")
    df = df.dropna(subset=["ann_date", "holder_num"])

    # 按股票+报告期去重，保留最新公告
    df = df.sort_values(["ts_code", "end_date", "ann_date"])
    df = df.drop_duplicates(subset=["ts_code", "end_date"], keep="last")

    # 按股票+公告日排序，计算环比变动
    df = df.sort_values(["ts_code", "ann_date"])
    grouped = df.groupby("ts_code")
    df["holder_num_prev"] = grouped["holder_num"].shift(1)
    df["holder_num_prev2"] = grouped["holder_num"].shift(2)
    df["holder_num_chg"] = (df["holder_num"] - df["holder_num_prev"]) / df["holder_num_prev"].replace(0, np.nan)
    df["holder_num_chg_2q"] = (df["holder_num"] - df["holder_num_prev2"]) / df["holder_num_prev2"].replace(0, np.nan)

    if len(trading_dates) == 1:
        trade_date = trading_dates[0]
        visible = df[df["ann_date"] <= trade_date].sort_values(["ts_code", "ann_date"])
        if visible.empty:
            result = {trade_date: pd.DataFrame(columns=["ts_code"] + HOLDER_COLS)}
        else:
            day_df = (
                visible.drop_duplicates(subset=["ts_code"], keep="last")
                [["ts_code"] + HOLDER_COLS]
                .reset_index(drop=True)
            )
            result = {trade_date: day_df}
        logger.info(f"股东人数查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
        return result

    # 构建每只股票的 ann_date 排序列表，用于 point-in-time 二分查找
    stock_ann_dates: Dict[str, list] = {}
    stock_values: Dict[str, list] = {}
    for ts_code, grp in df.groupby("ts_code"):
        grp = grp.sort_values("ann_date")
        stock_ann_dates[ts_code] = grp["ann_date"].tolist()
        stock_values[ts_code] = grp[HOLDER_COLS].values.tolist()

    # 对每个交易日查询
    result: Dict[str, pd.DataFrame] = {}
    for trade_date in trading_dates:
        rows = []
        for ts_code, ann_dates in stock_ann_dates.items():
            # 二分查找 ann_date <= trade_date 的最新记录
            idx = bisect.bisect_right(ann_dates, trade_date) - 1
            if idx >= 0:
                values = stock_values[ts_code][idx]
                rows.append({
                    "ts_code": ts_code,
                    "holder_num_chg": values[0],
                    "holder_num_chg_2q": values[1],
                })
        if rows:
            result[trade_date] = pd.DataFrame(rows)

    logger.info(f"股东人数查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result
