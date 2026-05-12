"""一致预期修正因子模块

基于 report_rc（一致预期研报）按 report_date 聚合构建时序修正信号。
当前已有一致预期基础因子（cons_eps_mean_fy1 等），本模块补充时间序列维度的
"修正方向/加速度/分歧度"信号——这些是 A 股实证中最有效的负向预警因子。

核心信号：
- EPS 修正加速度：修正本身在加速还是减速
- 分析师分歧度：预测标准差/均值（>0.3 提示不确定性高）
- 分歧度变化：分歧度扩大 = 风险上升
- 目标价上行空间：当前价距目标价的距离
- 覆盖分析师数量变化：撤出覆盖 = 强烈负向
- 评级上调比例：边际情绪改善
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# 一致预期修正因子输出列
CONSENSUS_REVISION_COLS = [
    "cons_eps_revision_accel",     # EPS 修正加速度（修正变化率）
    "cons_eps_dispersion",         # 分析师 EPS 预测分歧度
    "cons_eps_dispersion_chg",     # 分歧度月度变化
    "cons_target_upside",          # 目标价上行空间
    "cons_target_upside_chg",      # 目标价上行空间月度变化
    "cons_analyst_count_chg",      # 覆盖分析师数月度变化
    "cons_rating_upgrade_ratio",   # 近 30 日评级上调占比
]

CONSENSUS_REVISION_FRESHNESS_COL = "cons_revision_freshness_days"


def build_consensus_revision_lookup_by_date(
    report_rc_raw: pd.DataFrame,
    trading_dates: List[str],
    daily_data_lookup: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """基于 report_rc 原始数据构建一致预期修正日频查询表

    对每只股票在每个交易日，用最近 90 日内的研报数据计算修正信号。

    Args:
        report_rc_raw: report_rc 原始数据，需包含
                       ts_code, report_date, rec_fore_Netprofit, rec_target
        trading_dates: 交易日列表
        daily_data_lookup: 日线数据查询表 {trade_date: DataFrame}，
                          用于获取 close_adj 计算目标价上行空间

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, cons_eps_revision_accel, ...)}
    """
    if report_rc_raw is None or len(report_rc_raw) == 0:
        logger.warning("report_rc 数据为空，跳过一致预期修正因子构建")
        return {}

    df = report_rc_raw.copy()
    df["report_date"] = df["report_date"].astype(str).str[:8]

    # 按 ts_code + report_date 分组聚合每日研报
    # 对同一日多份研报取均值
    agg_cols = {}
    if "rec_fore_Netprofit" in df.columns:
        df["rec_fore_Netprofit"] = pd.to_numeric(df["rec_fore_Netprofit"], errors="coerce")
        agg_cols["rec_fore_Netprofit"] = ["mean", "std", "count"]
    if "rec_target" in df.columns:
        df["rec_target"] = pd.to_numeric(df["rec_target"], errors="coerce")
        agg_cols["rec_target"] = "mean"

    if not agg_cols:
        logger.warning("report_rc 缺少 rec_fore_Netprofit 或 rec_target 列")
        return {}

    daily = df.groupby(["ts_code", "report_date"], as_index=False).agg(agg_cols)
    # 展平多级列名
    daily.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in daily.columns
    ]

    # 重命名为简洁列名
    rename_map = {
        "rec_fore_Netprofit_mean": "eps_pred_mean",
        "rec_fore_Netprofit_std": "eps_pred_std",
        "rec_fore_Netprofit_count": "analyst_count",
        "rec_target_mean": "target_price",
    }
    daily = daily.rename(columns={k: v for k, v in rename_map.items() if k in daily.columns})

    daily = daily.sort_values(["ts_code", "report_date"])

    logger.info(
        f"一致预期修正构建: {daily['ts_code'].nunique()} 只股票, "
        f"{daily['report_date'].nunique()} 个报告日"
    )

    # 按交易日遍历，对每只股票计算最近 90 日窗口的修正信号
    result_dict: Dict[str, pd.DataFrame] = {}
    daily_dates = sorted(daily["report_date"].unique())

    for trade_date in trading_dates:
        rows = []
        for ts_code, grp in daily.groupby("ts_code"):
            # 取 report_date <= trade_date 且最近 90 日内的记录
            recent = grp[
                (grp["report_date"] <= trade_date)
                & (grp["report_date"] >= _offset_date(trade_date, -90))
            ]
            if len(recent) < 3:
                continue

            # 更早 30 日的窗口（用于计算加速度）
            earlier = grp[
                (grp["report_date"] <= _offset_date(trade_date, -30))
                & (grp["report_date"] >= _offset_date(trade_date, -120))
            ]

            row = {"ts_code": ts_code}

            # EPS 预测均值和标准差
            if "eps_pred_mean" in recent.columns:
                row["cons_eps_mean_90d"] = float(recent["eps_pred_mean"].mean())
                if len(recent) >= 5 and recent["eps_pred_mean"].std() > 0:
                    row["cons_eps_dispersion"] = float(
                        recent["eps_pred_mean"].std() / abs(recent["eps_pred_mean"].mean())
                        if abs(recent["eps_pred_mean"].mean()) > 1e-6
                        else 0.0
                    )
                else:
                    row["cons_eps_dispersion"] = np.nan

            # EPS 修正方向（斜率）
            if len(recent) >= 10 and "eps_pred_mean" in recent.columns:
                recent_sorted = recent.sort_values("report_date")
                eps_vals = recent_sorted["eps_pred_mean"].values
                if len(eps_vals) >= 5 and not np.all(np.isnan(eps_vals)):
                    x = np.arange(len(eps_vals))
                    mask = ~np.isnan(eps_vals)
                    if mask.sum() >= 3:
                        slope = np.polyfit(x[mask], eps_vals[mask], 1)[0]
                        row["cons_eps_revision_accel"] = float(slope / (abs(eps_vals[mask].mean()) + 1e-6))
                    else:
                        row["cons_eps_revision_accel"] = np.nan
                else:
                    row["cons_eps_revision_accel"] = np.nan
            else:
                row["cons_eps_revision_accel"] = np.nan

            # 分歧度变化
            if len(earlier) >= 5 and "eps_pred_mean" in earlier.columns:
                earlier_std = earlier["eps_pred_mean"].std()
                earlier_mean = earlier["eps_pred_mean"].mean()
                if earlier_std > 0 and abs(earlier_mean) > 1e-6:
                    earlier_disp = earlier_std / abs(earlier_mean)
                else:
                    earlier_disp = np.nan
                if not np.isnan(earlier_disp) and not np.isnan(row.get("cons_eps_dispersion", np.nan)):
                    row["cons_eps_dispersion_chg"] = float(
                        row["cons_eps_dispersion"] - earlier_disp
                    )
                else:
                    row["cons_eps_dispersion_chg"] = np.nan
            else:
                row["cons_eps_dispersion_chg"] = np.nan

            # 目标价上行空间
            if "target_price" in recent.columns and daily_data_lookup is not None:
                price_df = daily_data_lookup.get(trade_date)
                if price_df is not None:
                    price_row = price_df[price_df["ts_code"] == ts_code]
                    if not price_row.empty:
                        close_adj = float(price_row["close_adj"].iloc[0])
                        target = float(recent["target_price"].mean())
                        if close_adj > 0 and not np.isnan(target):
                            row["cons_target_upside"] = float(target / close_adj - 1.0)
                        else:
                            row["cons_target_upside"] = np.nan
                    else:
                        row["cons_target_upside"] = np.nan
                else:
                    row["cons_target_upside"] = np.nan
            else:
                row["cons_target_upside"] = np.nan

            # 覆盖分析师数变化
            if "analyst_count" in recent.columns:
                recent_count = len(recent)
                earlier_count = len(earlier) if len(earlier) > 0 else recent_count
                row["cons_analyst_count_chg"] = float(
                    (recent_count - earlier_count) / max(earlier_count, 1)
                )
            else:
                row["cons_analyst_count_chg"] = np.nan

            # 目标价变化（30 日窗口 vs 更早 30 日）
            if "target_price" in recent.columns and len(earlier) >= 3:
                recent_target = float(recent["target_price"].mean())
                earlier_target = float(earlier["target_price"].mean())
                if not np.isnan(recent_target) and not np.isnan(earlier_target) and abs(earlier_target) > 1e-6:
                    row["cons_target_upside_chg"] = float(
                        recent_target / earlier_target - 1.0
                    )
                else:
                    row["cons_target_upside_chg"] = np.nan
            else:
                row["cons_target_upside_chg"] = np.nan

            # 评级上调比例（简化：用目标价是否上调代理）
            if "target_price" in recent.columns and len(earlier) >= 3:
                if not np.isnan(row.get("cons_target_upside_chg", np.nan)):
                    row["cons_rating_upgrade_ratio"] = float(
                        1.0 if row["cons_target_upside_chg"] > 0.02 else 0.0
                    )
                else:
                    row["cons_rating_upgrade_ratio"] = np.nan
            else:
                row["cons_rating_upgrade_ratio"] = np.nan

            # 新鲜度：最近一次研报距当日天数
            if len(recent) > 0:
                latest_report = recent["report_date"].max()
                row[CONSENSUS_REVISION_FRESHNESS_COL] = _days_between(latest_report, trade_date)
            else:
                row[CONSENSUS_REVISION_FRESHNESS_COL] = np.nan

            rows.append(row)

        if rows:
            result_dict[trade_date] = pd.DataFrame(rows)

    hit_dates = len(result_dict)
    logger.info(f"一致预期修正日频查询表构建完成: {hit_dates} 个交易日有数据")
    return result_dict


def _offset_date(date_str: str, offset_days: int) -> str:
    """日期偏移辅助函数。"""
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str[:8], "%Y%m%d")
    return (dt + timedelta(days=offset_days)).strftime("%Y%m%d")


def _days_between(date_a: str, date_b: str) -> int:
    """计算两个日期之间的天数。"""
    from datetime import datetime
    a = datetime.strptime(date_a[:8], "%Y%m%d")
    b = datetime.strptime(date_b[:8], "%Y%m%d")
    return abs((b - a).days)
