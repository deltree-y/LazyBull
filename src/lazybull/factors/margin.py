"""融资融券因子模块

将日频融资融券明细数据（margin_detail）转化为截面特征。

数据来源：Tushare margin_detail API（2000 积分）
- rzye: 融资余额（元）
- rqye: 融券余额（元）
- rzmre: 融资买入额（元）
- rzche: 融资偿还额（元）
- rqmcl: 融券卖出量（股）
- rqchl: 融券偿还量（股）

因子说明：
- rzye_chg_5/20: 融资余额 N 日变动率，反映杠杆情绪变化
- rqye_rzye_ratio: 融券余额 / 融资余额，做空情绪指标
- margin_net_buy_ratio: 融资净买入 / 成交额，杠杆资金参与度
- short_balance_change_5: 融券余额 5 日变化率（rqye），做空力量强弱
- short_sell_vol_change_5: 融券卖出量 5 日变化率（rqmcl），做空活跃度
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd


MARGIN_COLS = [
    "rzye_chg_5",
    "rzye_chg_20",
    "rqye_rzye_ratio",
    "margin_net_buy_ratio",
    "short_balance_change_5",
    "short_sell_vol_change_5",
]


def build_margin_lookup_by_date(
    margin_detail: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """构建融资融券日频查询表

    Args:
        margin_detail: 融资融券明细 DataFrame，需包含 ts_code, trade_date,
                       rzye, rqye, rzmre, rzche 等字段
        trading_dates: 交易日列表（YYYYMMDD 字符串，已排序）

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, rzye_chg_5, ...)}
    """
    df = margin_detail.copy()

    # 确保日期格式
    df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])

    # 确保数值列
    for col in ["rzye", "rqye", "rzmre", "rzche"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["ts_code", "trade_date"])

    # 逐股计算滚动变动率
    grouped = df.groupby("ts_code")

    # 融资余额 N 日变动率
    df["rzye_lag_5"] = grouped["rzye"].shift(5)
    df["rzye_lag_20"] = grouped["rzye"].shift(20)
    df["rzye_chg_5"] = (df["rzye"] - df["rzye_lag_5"]) / df["rzye_lag_5"].replace(0, np.nan)
    df["rzye_chg_20"] = (df["rzye"] - df["rzye_lag_20"]) / df["rzye_lag_20"].replace(0, np.nan)

    # 融券/融资比
    df["rqye_rzye_ratio"] = df["rqye"] / df["rzye"].replace(0, np.nan)

    # 融资净买入/成交额（rzmre - rzche 为融资净买入额）
    if "rzmre" in df.columns and "rzche" in df.columns:
        df["margin_net_buy"] = df["rzmre"] - df["rzche"]
    else:
        df["margin_net_buy"] = np.nan

    # 融券余额 5 日变化率（rqye）：做空力量增强 = 负面信号
    if "rqye" in df.columns:
        df["rqye_lag_5"] = grouped["rqye"].shift(5)
        df["short_balance_change_5"] = (
            (df["rqye"] - df["rqye_lag_5"]) / df["rqye_lag_5"].replace(0, np.nan)
        )
    else:
        df["short_balance_change_5"] = np.nan

    # 融券卖出量 5 日变化率（rqmcl）：做空活跃度
    if "rqmcl" in df.columns:
        df["rqmcl_lag_5"] = grouped["rqmcl"].shift(5)
        df["short_sell_vol_change_5"] = (
            (df["rqmcl"] - df["rqmcl_lag_5"]) / df["rqmcl_lag_5"].replace(0, np.nan)
        )
    else:
        df["short_sell_vol_change_5"] = np.nan

    # 构建日频查询表
    date_set = set(trading_dates)
    result: Dict[str, pd.DataFrame] = {}
    keep_cols = ["ts_code"] + MARGIN_COLS + ["margin_net_buy"]

    for trade_date, grp in df.groupby("trade_date"):
        if trade_date not in date_set:
            continue
        available = [c for c in keep_cols if c in grp.columns]
        result[trade_date] = grp[available].reset_index(drop=True)

    logger.info(f"融资融券查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result
