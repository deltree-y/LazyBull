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
- cons_eps_mean_fym1: 近 90 日上一财年 (FY-1) 每股收益预测均值（兼容旧模型）
- cons_eps_mean_fy0: 近 90 日当前财年 (FY0) 每股收益预测均值
- cons_eps_mean_fy1: 近 90 日未来第一财年 (FY1) 每股收益预测均值
- cons_eps_mean_fy2: 近 90 日未来第二财年 (FY2) 每股收益预测均值
- cons_eps_yield_fym1/fy0/fy1/fy2: 对应财年 EPS / 当日未复权收盘价
- cons_eps_revision_30d: 近 30 日相对此前 90 日的有界对称修正率 (全预测期)
- cons_target_price_mid: 近 90 日目标价中值（兼容旧模型）
- cons_target_upside: 目标价中值 / 当日未复权收盘价 - 1
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
from ..data.report_rc import deduplicate_report_rc

CONS_COLS = [
    "cons_analyst_count_30d",
    "cons_eps_mean_fym1",
    "cons_eps_mean_fy0",
    "cons_eps_mean_fy1",
    "cons_eps_mean_fy2",
    "cons_eps_yield_fym1",
    "cons_eps_yield_fy0",
    "cons_eps_yield_fy1",
    "cons_eps_yield_fy2",
    "cons_eps_revision_30d",
    "cons_target_price_mid",
    "cons_target_upside",
    "cons_rating_score",
]

CONSENSUS_FRESHNESS_COL = "consensus_freshness_days"

_RATING_MAP = {
    "强烈推荐": 5.0,
    "买入": 5.0,
    "买进": 5.0,
    "强推": 5.0,
    "buy": 5.0,
    "跑赢行业": 4.0,
    "优于大市": 4.0,
    "增持": 4.0,
    "推荐": 4.0,
    "outperform": 4.0,
    "overweight": 4.0,
    "中性": 3.0,
    "持有": 3.0,
    "hold": 3.0,
    "neutral": 3.0,
    "减持": 2.0,
    "underperform": 2.0,
    "卖出": 1.0,
    "sell": 1.0,
}

_MISSING_RATINGS = {"", "无", "无评级", "未评级", "暂无评级", "none", "n/a", "nan", "-", "--"}

_STATE_AGGREGATION_DAYS = 90
_MAX_STATE_AGE_DAYS = 365
_REVISION_RECENT_DAYS = 30
_REVISION_BASELINE_DAYS = 90
_REVISION_MIN_RECENT_DATES = 2
_REVISION_MIN_BASELINE_DATES = 3


def _rating_to_score(rating) -> float:
    if rating is None or pd.isna(rating):
        return np.nan
    key = str(rating).strip().casefold()
    if key in _MISSING_RATINGS:
        return np.nan
    for k, v in _RATING_MAP.items():
        if k in key:
            return v
    return np.nan


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


def parse_quarter_year(quarter) -> Optional[int]:
    """公共入口：从 quarter 字段解析预测财年（供修正因子等复用）。"""
    return _parse_quarter_year(quarter)


def rating_to_score(rating) -> float:
    """公共入口：评级文本映射为分数（供修正因子等复用）。"""
    return _rating_to_score(rating)


def _build_close_lookup(
    daily_data_lookup: Optional[Dict[str, pd.DataFrame]],
    trading_dates: List[str],
) -> Dict[str, Dict[str, float]]:
    """构建未复权收盘价查询，供每股预测值转换为可比收益率。"""
    if daily_data_lookup is None:
        return {}

    lookup: Dict[str, Dict[str, float]] = {}
    for trade_date in trading_dates:
        price_df = daily_data_lookup.get(trade_date)
        if price_df is None or len(price_df) == 0:
            continue
        if "ts_code" not in price_df.columns or "close" not in price_df.columns:
            continue
        ts_codes = price_df["ts_code"].astype(str)
        closes = pd.to_numeric(price_df["close"], errors="coerce")
        day_map = {
            ts_code: float(close)
            for ts_code, close in zip(ts_codes, closes)
            if pd.notna(close) and float(close) > 0
        }
        if day_map:
            lookup[trade_date] = day_map
    return lookup


def _build_eps_revision(visible: pd.DataFrame, td_dt: pd.Timestamp) -> pd.Series:
    """按报告日中值计算全预测期 EPS 有界对称修正率。"""
    recent_start = td_dt - pd.Timedelta(days=_REVISION_RECENT_DAYS)
    baseline_start = recent_start - pd.Timedelta(days=_REVISION_BASELINE_DAYS)
    revision_source = visible[(visible["_rd_dt"] > baseline_start) & (visible["_rd_dt"] <= td_dt)][
        ["ts_code", "_rd_dt", "eps"]
    ].dropna(subset=["eps"])
    if revision_source.empty:
        return pd.Series(dtype="float64", name="cons_eps_revision_30d")

    daily_eps = revision_source.groupby(["ts_code", "_rd_dt"])["eps"].median()
    recent = daily_eps[daily_eps.index.get_level_values("_rd_dt") > recent_start]
    baseline = daily_eps[daily_eps.index.get_level_values("_rd_dt") <= recent_start]
    if recent.empty or baseline.empty:
        return pd.Series(dtype="float64", name="cons_eps_revision_30d")

    recent_group = recent.groupby(level="ts_code")
    baseline_group = baseline.groupby(level="ts_code")
    recent_median = recent_group.median()
    baseline_median = baseline_group.median()
    recent_count = recent_group.size()
    baseline_count = baseline_group.size()
    denominator = recent_median.abs() + baseline_median.abs()
    revision = 2.0 * (recent_median - baseline_median) / denominator.where(denominator > 0)
    valid = (recent_count >= _REVISION_MIN_RECENT_DATES) & (
        baseline_count >= _REVISION_MIN_BASELINE_DATES
    )
    revision = revision.where(valid)
    revision.name = "cons_eps_revision_30d"
    return revision


def build_consensus_lookup_by_date(
    report_rc_df: pd.DataFrame,
    trading_dates: List[str],
    daily_data_lookup: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """构建卖方一致预期日频查询表

    Args:
        report_rc_df: report_rc 原始 DataFrame，需含完整研报身份列、quarter 与 eps
        trading_dates: 交易日列表 (YYYYMMDD 字符串, 已排序)
        daily_data_lookup: 日线查询表，需包含 ts_code/close，用于经济归一化

    Returns:
        Dict[trade_date -> DataFrame(ts_code, cons_*, consensus_freshness_days)]
    """
    if report_rc_df is None or len(report_rc_df) == 0:
        logger.warning("一致预期因子: 输入数据为空")
        return {}

    df = report_rc_df.copy()
    df["report_date"] = normalize_series_to_yyyymmdd(df["report_date"])
    df = deduplicate_report_rc(
        df,
        include_quarter=True,
        require_full_identity=True,
    )
    for col in ["eps", "max_price", "min_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 财年定位: 解析研报中 quarter 的预测年份, 相对 report_date 发布年份
    # 计算 FY 位置 (FY0=当年, FY1=次年, FY2=后年), 供 EPS 按财年分组过滤。
    df["_report_year"] = pd.to_numeric(df["report_date"].astype(str).str[:4], errors="coerce")
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
    close_lookup = _build_close_lookup(daily_data_lookup, sorted_trading_dates)

    # 预计算日期 dt 便于窗口比较
    df["_rd_dt"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")

    for td in sorted_trading_dates:
        td_dt = pd.to_datetime(td, format="%Y%m%d", errors="coerce")
        if pd.isna(td_dt):
            continue
        state_cutoff = td_dt - pd.Timedelta(days=_MAX_STATE_AGE_DAYS)
        window_30 = td_dt - pd.Timedelta(days=_REVISION_RECENT_DAYS)

        visible = df[df["_rd_dt"] <= td_dt]
        if visible.empty:
            continue

        # 状态值以每只股票最新研报日为锚聚合近 90 日研报；最多保留 365 天，
        # 避免交易日第 90 天整行消失，由 freshness 衰减负责平滑降权。
        state_candidates = visible[visible["_rd_dt"] >= state_cutoff].copy()
        if state_candidates.empty:
            continue
        latest_by_stock = state_candidates.groupby("ts_code")["_rd_dt"].transform("max")
        win90 = state_candidates[
            state_candidates["_rd_dt"]
            > latest_by_stock - pd.Timedelta(days=_STATE_AGGREGATION_DAYS)
        ]

        agg = win90.groupby("ts_code").agg(
            cons_target_price_mid=("_tp_mid", "median"),
            cons_rating_score=("_rating_score", "mean"),
        )

        # EPS 预测均值按财年分组过滤 (FY0=当年, FY1=次年, FY2=后年),
        # 避免同一 report_date 下不同预测季度混入同一均值
        for fy_rel, fy_label in ((-1, "fym1"), (0, "fy0"), (1, "fy1"), (2, "fy2")):
            fy_sub = win90[win90["_rel_fy"] == fy_rel]
            fy_mean = (
                fy_sub.groupby("ts_code")["eps"].mean()
                if not fy_sub.empty
                else pd.Series(dtype="float64")
            )
            agg = agg.join(fy_mean.rename(f"cons_eps_mean_{fy_label}"), how="left")

        # 30 日研报覆盖数：同一研报通常按 FY0/FY1/FY2 展开为多条预测行，
        # 必须先按研报身份去重，避免把预测期数量误当覆盖数量。
        win30 = visible[visible["_rd_dt"] > window_30]
        reports30 = deduplicate_report_rc(win30, include_quarter=False)
        count30 = reports30.groupby("ts_code").size().rename("cons_analyst_count_30d")
        agg = agg.join(count30, how="left")
        agg["cons_analyst_count_30d"] = agg["cons_analyst_count_30d"].fillna(0.0)

        # revision 保持全预测期口径，但先按报告日取中值，并要求足够报告日样本。
        agg = agg.join(_build_eps_revision(visible, td_dt), how="left")

        close_by_stock = pd.Series(
            [close_lookup.get(td, {}).get(str(ts_code), np.nan) for ts_code in agg.index],
            index=agg.index,
            dtype="float64",
        )
        for fy_label in ("fym1", "fy0", "fy1", "fy2"):
            raw_col = f"cons_eps_mean_{fy_label}"
            agg[f"cons_eps_yield_{fy_label}"] = agg[raw_col] / close_by_stock
        agg["cons_target_upside"] = agg["cons_target_price_mid"] / close_by_stock - 1.0

        latest_report = (
            state_candidates.groupby("ts_code")["_rd_dt"].max().rename(CONSENSUS_FRESHNESS_COL)
        )
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
