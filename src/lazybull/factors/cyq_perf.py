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

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd


CYQ_PERF_COLS = [
    "winner_rate",  # 胜率
    "weight_avg_bias",  # 加权平均成本偏离度
    "cost_concentration",  # 筹码集中度
    "winner_rate_chg_5",  # 5日胜率变化
    "winner_rate_chg_20",  # 20日胜率变化
]


def build_cyq_perf_lookup_by_date(
    cyq_perf_df: pd.DataFrame,
    trading_dates: List[str],
    calendar_dates: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """将 cyq_perf 数据构建为日频截面 lookup

    Args:
        cyq_perf_df: cyq_perf 原始 DataFrame，需包含
            ts_code, trade_date, winner_rate, weight_avg,
            cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct
        trading_dates: 输出交易日列表（YYYYMMDD 字符串，已排序）；
            仅用于过滤输出日期，同时并入变化率对齐日历
        calendar_dates: 可选完整交易日历（YYYYMMDD 字符串）。ensure 链路只输出
            单日截面，但历史缺失日（如下载失败导致全市场缺某日）仍需补入对齐日历，
            避免 diff 静默跨期；缺省时仅用数据内日期并集

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, winner_rate, ...)}
    """
    if cyq_perf_df is None or len(cyq_perf_df) == 0:
        return {}

    df = cyq_perf_df.copy()

    # 日期标准化
    if "trade_date" in df.columns:
        df["trade_date"] = normalize_series_to_yyyymmdd(df["trade_date"])

    # 数值列转换
    numeric_cols = [
        "winner_rate",
        "weight_avg",
        "cost_5pct",
        "cost_15pct",
        "cost_50pct",
        "cost_85pct",
        "cost_95pct",
        "his_low",
        "his_high",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["trade_date", "winner_rate"])
    if len(df) == 0:
        return {}

    # 按股票+日期去重
    df = df.sort_values(["ts_code", "trade_date"])
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")

    # 筹码集中度 = (85% 成本 - 15% 成本) / 加权平均成本（向量化）
    if all(c in df.columns for c in ["cost_85pct", "cost_15pct", "weight_avg"]):
        df["cost_concentration"] = np.where(
            (df["weight_avg"] > 0) & df["cost_85pct"].notna() & df["cost_15pct"].notna(),
            (df["cost_85pct"] - df["cost_15pct"]) / df["weight_avg"],
            np.nan,
        )
    else:
        df["cost_concentration"] = np.nan

    # 胜率变化率：按交易日历对齐滞后。日历 = 数据内日期 ∪ 传入日期（裁剪到数据范围），
    # 个股缺失数据日与全市场缺失数据日均保留空位，diff(5)/diff(20) 严格取 5/20 个
    # 交易日前的值；对齐位置缺数据则为 NaN，不静默跨越更长的日历区间。
    calendar = set(df["trade_date"].unique())
    cal_lo, cal_hi = min(calendar), max(calendar)
    calendar |= {d for d in trading_dates if cal_lo <= d <= cal_hi}
    if calendar_dates:
        calendar |= {d for d in calendar_dates if cal_lo <= d <= cal_hi}
    df = _compute_winner_rate_changes_calendar_aligned(df, sorted(calendar))

    # 构建 trade_date -> DataFrame 的 lookup（向量化按日切分，不再逐行 iterrows）
    trading_dates_set = set(trading_dates)
    output_cols = [
        "ts_code",
        "winner_rate",
        "weight_avg",
        "cost_concentration",
        "winner_rate_chg_5",
        "winner_rate_chg_20",
    ]
    keep_cols = [c for c in output_cols if c in df.columns]
    result: Dict[str, pd.DataFrame] = {}
    for trade_date, day_df in df.groupby("trade_date", sort=False):
        if trade_date not in trading_dates_set:
            continue
        result[trade_date] = day_df[keep_cols].reset_index(drop=True)

    logger.info(f"筹码胜率查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result


def _compute_winner_rate_changes_calendar_aligned(
    df: pd.DataFrame, calendar: List[str]
) -> pd.DataFrame:
    """按交易日历对齐计算 winner_rate 的 5/20 日变化。

    calendar 为对齐基准日历（已包含数据内日期与调用方补入的完整交易日），
    每只股票按日历位置重索引后再取 diff(5)/diff(20)：恰好在 5/20 个交易日前的
    值不可得时结果为 NaN，避免行级 diff 在缺数据日被删除后静默跨期。
    """
    td_pos = {d: i for i, d in enumerate(calendar)}
    df["_pos"] = df["trade_date"].map(td_pos).astype(int)

    results = []
    for code, grp in df.groupby("ts_code", sort=False):
        s = grp.set_index("_pos")["winner_rate"]
        lo, hi = int(s.index.min()), int(s.index.max())
        s_full = s.reindex(range(lo, hi + 1))
        chg5 = s_full.diff(5)
        chg20 = s_full.diff(20)
        grp = grp.copy()
        grp["winner_rate_chg_5"] = grp["_pos"].map(chg5)
        grp["winner_rate_chg_20"] = grp["_pos"].map(chg20)
        results.append(grp)

    df = pd.concat(results, ignore_index=True) if results else df
    return df.drop(columns=["_pos"])
