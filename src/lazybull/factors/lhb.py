"""龙虎榜因子模块

基于 TuShare top_list 接口的个股上榜数据构造截面特征。

数据来源：Tushare top_list（2000 积分）
- trade_date, ts_code, l_buy, l_sell, l_amount, net_amount, net_rate
- amount_rate: 龙虎榜成交额占总成交额比
- turnover_rate, float_values, reason

因子说明：
- lhb_on_list: 当日是否上榜 (0/1)
- lhb_net_amount: 当日龙虎榜净买入额（元）
- lhb_net_rate: 当日净买入占流通市值比
- lhb_amount_rate: 龙虎榜成交占比
- lhb_up_days_20: 近 20 日累计上榜次数
- lhb_net_sum_5 / lhb_net_sum_20: 近 5/20 日净买入累计
- lhb_reason_count: 当日上榜理由数（同一股票同日可有多条）

注: top_list 同一 (trade_date, ts_code) 可能出现多条记录
(不同上榜理由), 需先做 groupby 聚合。
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger


LHB_COLS = [
    "lhb_on_list",
    "lhb_net_amount",
    "lhb_net_rate",
    "lhb_amount_rate",
    "lhb_up_days_20",
    "lhb_net_sum_5",
    "lhb_net_sum_20",
    "lhb_reason_count",
]


def build_lhb_lookup_by_date(
    top_list_df: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """构建龙虎榜日频查询表

    Args:
        top_list_df: top_list 原始 DataFrame, 至少含 trade_date, ts_code,
                    net_amount (部分日期可能缺失)
        trading_dates: 交易日列表 (YYYYMMDD 字符串, 已排序)

    Returns:
        Dict[trade_date -> DataFrame(ts_code, lhb_*)]

    说明: 未上榜个股不会在 result 中出现, 由 FeatureBuilder 合并时
    以 left-join 方式保留 NaN, 再在基础特征里用 0 填充 (lhb_on_list=0,
    其余计数/额度类填 0 即可)。
    """
    if top_list_df is None or len(top_list_df) == 0:
        logger.warning("龙虎榜因子: 输入数据为空")
        return {}

    df = top_list_df.copy()
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]

    for col in ["net_amount", "net_rate", "amount_rate", "l_amount", "float_values"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 先按 (trade_date, ts_code) 聚合: 多条上榜理由合并为一条
    agg_spec = {}
    if "net_amount" in df.columns:
        agg_spec["net_amount"] = "sum"
    if "net_rate" in df.columns:
        agg_spec["net_rate"] = "mean"
    if "amount_rate" in df.columns:
        agg_spec["amount_rate"] = "mean"
    if "reason" in df.columns:
        agg_spec["reason"] = "count"

    grouped = df.groupby(["trade_date", "ts_code"], as_index=False).agg(agg_spec)
    grouped = grouped.rename(
        columns={
            "net_amount": "lhb_net_amount",
            "net_rate": "lhb_net_rate",
            "amount_rate": "lhb_amount_rate",
            "reason": "lhb_reason_count",
        }
    )
    grouped["lhb_on_list"] = 1.0

    # 按 ts_code 排序, 计算滚动指标
    grouped = grouped.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = grouped.groupby("ts_code", group_keys=False)

    if "lhb_net_amount" in grouped.columns:
        grouped["lhb_net_sum_5"] = g["lhb_net_amount"].apply(
            lambda s: s.rolling(5, min_periods=1).sum()
        )
        grouped["lhb_net_sum_20"] = g["lhb_net_amount"].apply(
            lambda s: s.rolling(20, min_periods=1).sum()
        )
    grouped["lhb_up_days_20"] = g["lhb_on_list"].apply(
        lambda s: s.rolling(20, min_periods=1).sum()
    )

    # 构建日频查询表
    date_set = set(trading_dates)
    result: Dict[str, pd.DataFrame] = {}
    keep_cols = ["ts_code"] + [c for c in LHB_COLS if c in grouped.columns]

    for trade_date, grp in grouped.groupby("trade_date"):
        if trade_date not in date_set:
            continue
        result[trade_date] = grp[keep_cols].reset_index(drop=True)

    logger.info(f"龙虎榜因子查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result
