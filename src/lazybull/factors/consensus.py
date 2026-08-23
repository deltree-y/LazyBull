"""卖方一致预期因子模块

基于 TuShare report_rc 接口的分析师研报数据构造截面特征。

数据来源：Tushare report_rc（2000 积分）
- ts_code, report_date, quarter
- eps: 每股收益预测
- max_price, min_price: 目标价区间
- rating: 评级文本
- op_pr/tp/np: 营收/净利润预测

因子说明：
- cons_analyst_count_30d: 近 30 日覆盖的研报数
- cons_eps_mean_fy0: 近 90 日当前财年 (FY0) 每股收益预测均值
- cons_eps_mean_fy1: 近 90 日未来第一财年 (FY1) 每股收益预测均值
- cons_eps_mean_fy2: 近 90 日未来第二财年 (FY2) 每股收益预测均值
- cons_eps_revision_30d: 近 30 日 EPS 预测中值相对于前 30 日的变化率 (全预测期)
- cons_target_price_mid: 近 90 日目标价中值 (max/min 均值)
- cons_rating_score: 近 90 日评级得分 (买入=5, 增持=4, 中性=3, 减持=2, 卖出=1)

注: report_rc 的 ts_code/report_date 粒度 + 每股票每日多条研报, 需按
ts_code 逐票处理, 滚动窗口基于研报发布日期 (report_date)。为避免前视,
特征以"截至 trade_date 当天可见的所有研报"进行滚动聚合。
财年按研报中 quarter 的预测年份相对 report_date 发布年份定位
(FY0=当年, FY1=次年, FY2=后年), EPS 相关列按财年分组过滤,
避免不同预测期混入同一均值。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd


CONS_COLS = [
    "cons_analyst_count_30d",
    "cons_eps_mean_fy0",
    "cons_eps_mean_fy1",
    "cons_eps_mean_fy2",
    "cons_eps_revision_30d",
    "cons_target_price_mid",
    "cons_rating_score",
]

CONSENSUS_FRESHNESS_COL = "consensus_freshness_days"

_RATING_MAP = {
    # 覆盖常见中文评级 + 常见英文评级, 未命中返回 3.0 (中性)
    "买入": 5.0, "强烈推荐": 5.0, "Buy": 5.0, "强推": 5.0,
    "增持": 4.0, "推荐": 4.0, "Outperform": 4.0,
    "中性": 3.0, "持有": 3.0, "Hold": 3.0, "Neutral": 3.0,
    "减持": 2.0, "Underperform": 2.0,
    "卖出": 1.0, "Sell": 1.0,
}


def _rating_to_score(rating) -> float:
    if rating is None or (isinstance(rating, float) and pd.isna(rating)):
        return np.nan
    key = str(rating).strip()
    if not key:
        return np.nan
    for k, v in _RATING_MAP.items():
        if k in key:
            return v
    return 3.0


def _parse_quarter_year(quarter) -> Optional[int]:
    """从 quarter 字段解析预测财年，如 '2024Q4' -> 2024。

    无法解析（缺失/非标准格式）返回 None。
    """
    if quarter is None:
        return None
    if isinstance(quarter, float) and pd.isna(quarter):
        return None
    s = str(quarter).strip()
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    return None


def build_consensus_lookup_by_date(
    report_rc_df: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """构建卖方一致预期日频查询表

    Args:
        report_rc_df: report_rc 原始 DataFrame, 需含 ts_code, report_date, eps
        trading_dates: 交易日列表 (YYYYMMDD 字符串, 已排序)

    Returns:
        Dict[trade_date -> DataFrame(ts_code, cons_*, consensus_freshness_days)]
    """
    if report_rc_df is None or len(report_rc_df) == 0:
        logger.warning("一致预期因子: 输入数据为空")
        return {}

    df = report_rc_df.copy()
    df["report_date"] = normalize_series_to_yyyymmdd(df["report_date"])
    for col in ["eps", "max_price", "min_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 财年定位: 解析研报中 quarter 的预测年份, 相对 report_date 发布年份
    # 计算 FY 位置 (FY0=当年, FY1=次年, FY2=后年), 供 EPS 按财年分组过滤。
    df["_report_year"] = pd.to_numeric(
        df["report_date"].astype(str).str[:4], errors="coerce"
    )
    if "quarter" in df.columns:
        df["_q_year"] = df["quarter"].map(_parse_quarter_year)
    else:
        df["_q_year"] = np.nan
    df["_rel_fy"] = df["_q_year"] - df["_report_year"]

    if "max_price" in df.columns and "min_price" in df.columns:
        df["_tp_mid"] = df[["max_price", "min_price"]].mean(axis=1, skipna=True)
    elif "max_price" in df.columns:
        df["_tp_mid"] = df["max_price"]
    elif "min_price" in df.columns:
        df["_tp_mid"] = df["min_price"]
    else:
        df["_tp_mid"] = np.nan

    if "rating" in df.columns:
        df["_rating_score"] = df["rating"].map(_rating_to_score)
    else:
        df["_rating_score"] = np.nan

    # 按 report_date 排序后便于截断到 <= trade_date 的视图
    df = df.sort_values(["ts_code", "report_date"]).reset_index(drop=True)

    sorted_trading_dates = sorted({d for d in trading_dates if d is not None})
    result: Dict[str, pd.DataFrame] = {}

    # 预计算日期 dt 便于窗口比较
    df["_rd_dt"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")

    for td in sorted_trading_dates:
        td_dt = pd.to_datetime(td, format="%Y%m%d", errors="coerce")
        if pd.isna(td_dt):
            continue
        window_90 = td_dt - pd.Timedelta(days=90)
        window_30 = td_dt - pd.Timedelta(days=30)
        window_60 = td_dt - pd.Timedelta(days=60)

        visible = df[df["_rd_dt"] <= td_dt]
        if visible.empty:
            continue

        # 90 日窗口聚合
        win90 = visible[visible["_rd_dt"] > window_90]
        if win90.empty:
            continue

        agg = win90.groupby("ts_code").agg(
            cons_target_price_mid=("_tp_mid", "median"),
            cons_rating_score=("_rating_score", "mean"),
        )

        # EPS 预测均值按财年分组过滤 (FY0=当年, FY1=次年, FY2=后年),
        # 避免同一 report_date 下不同预测季度混入同一均值
        for fy_rel, fy_label in ((0, "fy0"), (1, "fy1"), (2, "fy2")):
            fy_sub = win90[win90["_rel_fy"] == fy_rel]
            fy_mean = (
                fy_sub.groupby("ts_code")["eps"].mean()
                if not fy_sub.empty
                else pd.Series(dtype="float64")
            )
            agg = agg.join(fy_mean.rename(f"cons_eps_mean_{fy_label}"), how="left")

        # 30 日分析师数
        win30 = visible[visible["_rd_dt"] > window_30]
        count30 = win30.groupby("ts_code").size().rename("cons_analyst_count_30d")
        agg = agg.join(count30, how="left")
        agg["cons_analyst_count_30d"] = agg["cons_analyst_count_30d"].fillna(0.0)

        # EPS 修正率: 最近 30 日 eps 中值 vs 前 30 日 eps 中值 (全预测期口径)
        prev_win = visible[
            (visible["_rd_dt"] > window_60) & (visible["_rd_dt"] <= window_30)
        ]
        recent_med = win30.groupby("ts_code")["eps"].median()
        prev_med = prev_win.groupby("ts_code")["eps"].median()
        rev = (recent_med - prev_med) / prev_med.replace(0, np.nan).abs()
        rev.name = "cons_eps_revision_30d"
        agg = agg.join(rev, how="left")

        latest_report = visible.groupby("ts_code")["_rd_dt"].max().rename(CONSENSUS_FRESHNESS_COL)
        agg = agg.join(latest_report, how="left")

        agg = agg.reset_index()
        if CONSENSUS_FRESHNESS_COL in agg.columns:
            agg[CONSENSUS_FRESHNESS_COL] = (
                td_dt - pd.to_datetime(agg[CONSENSUS_FRESHNESS_COL], errors="coerce")
            ).dt.days
        keep_cols = ["ts_code"] + [c for c in CONS_COLS if c in agg.columns]
        if CONSENSUS_FRESHNESS_COL in agg.columns:
            keep_cols.append(CONSENSUS_FRESHNESS_COL)
        result[td] = agg[keep_cols].reset_index(drop=True)

    logger.info(f"一致预期因子查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result
