"""筹码胜率因子模块

将日频的筹码分布表现数据（cyq_perf）构建为每日截面特征。

数据来源：Tushare cyq_perf API（5000 积分）
更新频率：日频（每天 18~19 点更新）
核心逻辑：
- winner_rate 反映当前价位上的获利盘比例
- weight_avg 加权平均成本 → 偏离度衡量当前价与筹码中心的距离
- cost_concentration 衡量筹码集中度（窄→主力控盘，宽→筹码分散）
- 变化率捕捉筹码博弈的动态趋势
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger


CYQ_PERF_COLS = [
    "winner_rate",          # 胜率
    "weight_avg_bias",      # 加权平均成本偏离度
    "cost_concentration",   # 筹码集中度
    "winner_rate_chg_5",    # 5日胜率变化
    "winner_rate_chg_20",   # 20日胜率变化
]


def build_cyq_perf_lookup_by_date(
    cyq_perf_df: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """将 cyq_perf 数据构建为日频截面 lookup

    Args:
        cyq_perf_df: cyq_perf 原始 DataFrame，需包含
            ts_code, trade_date, winner_rate, weight_avg,
            cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct
        trading_dates: 交易日列表（YYYYMMDD 字符串，已排序）

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, winner_rate, ...)}
    """
    df = cyq_perf_df.copy()

    # 日期标准化
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]

    # 数值列转换
    numeric_cols = ["winner_rate", "weight_avg", "cost_5pct", "cost_15pct",
                    "cost_50pct", "cost_85pct", "cost_95pct",
                    "his_low", "his_high"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["trade_date", "winner_rate"])

    # 按股票+日期去重
    df = df.sort_values(["ts_code", "trade_date"])
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

    # 计算衍生特征（需排序后逐股计算）
    grouped = df.groupby("ts_code")

    # 胜率变化
    df["winner_rate_chg_5"] = grouped["winner_rate"].diff(5)
    df["winner_rate_chg_20"] = grouped["winner_rate"].diff(20)

    # 构建 trade_date -> DataFrame 的 lookup
    trading_dates_set = set(trading_dates)
    result: Dict[str, pd.DataFrame] = {}

    for trade_date, day_df in df.groupby("trade_date"):
        if trade_date not in trading_dates_set:
            continue

        rows = []
        for _, row in day_df.iterrows():
            weight_avg = row.get("weight_avg", np.nan)
            cost_15 = row.get("cost_15pct", np.nan)
            cost_85 = row.get("cost_85pct", np.nan)

            # 需要当日收盘价来计算偏离度，但 cyq_perf 本身不含收盘价
            # weight_avg_bias 在 builder 中通过 close 计算，此处先存 weight_avg
            # 筹码集中度 = (85% 成本 - 15% 成本) / 加权平均成本
            cost_concentration = np.nan
            if pd.notna(cost_85) and pd.notna(cost_15) and pd.notna(weight_avg) and weight_avg > 0:
                cost_concentration = (cost_85 - cost_15) / weight_avg

            rows.append({
                "ts_code": row["ts_code"],
                "winner_rate": row["winner_rate"],
                "weight_avg": weight_avg,
                "cost_concentration": cost_concentration,
                "winner_rate_chg_5": row.get("winner_rate_chg_5", np.nan),
                "winner_rate_chg_20": row.get("winner_rate_chg_20", np.nan),
            })

        if rows:
            result[trade_date] = pd.DataFrame(rows)

    logger.info(f"筹码胜率查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result
