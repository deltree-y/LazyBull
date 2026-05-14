"""一致预期修正因子模块

基于 report_rc（一致预期研报）按 report_date 聚合构建时序修正信号。
当前已有一致预期基础因子（cons_eps_mean_fy1 等），本模块补充时间序列维度的
"修正方向/加速度/分歧度"信号——这些是 A 股实证中最有效的负向预警因子。

核心信号：
- 分析师分歧度：预测标准差/均值（>0.3 提示不确定性高）
- 覆盖分析师数量变化：近期覆盖密度相对历史基线的变化
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

# 一致预期修正因子输出列
CONSENSUS_REVISION_COLS = [
    "cons_eps_dispersion",         # 分析师 EPS 预测分歧度
    "cons_analyst_count_chg",      # 覆盖分析师密度变化
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
        report_rc_raw: report_rc 原始数据，需包含 ts_code, report_date 与 EPS 预测字段
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

    # 兼容两套 report_rc 口径：
    # 1) rec_fore_Netprofit / rec_target（部分环境）
    # 2) np / tp + max_price/min_price（当前主口径）
    eps_source_col = _pick_first_existing_col(df, ["rec_fore_Netprofit", "np", "tp"])
    # 按 ts_code + report_date 分组聚合每日研报
    # 对同一日多份研报取均值
    agg_cols = {}
    if eps_source_col is not None:
        df[eps_source_col] = pd.to_numeric(df[eps_source_col], errors="coerce")
        agg_cols[eps_source_col] = ["mean", "std", "count"]

    if not agg_cols:
        logger.warning("report_rc 缺少净利润预测相关列，无法构建一致预期修正因子")
        return {}

    logger.info("一致预期修正字段映射: eps={}", eps_source_col if eps_source_col is not None else "缺失")

    daily = df.groupby(["ts_code", "report_date"], as_index=False).agg(agg_cols)
    # 展平多级列名
    daily.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in daily.columns
    ]

    # 重命名为简洁列名
    rename_map = {}
    if eps_source_col is not None:
        rename_map.update(
            {
                f"{eps_source_col}_mean": "eps_pred_mean",
                f"{eps_source_col}_std": "eps_pred_std",
                f"{eps_source_col}_count": "analyst_count",
            }
        )
    daily = daily.rename(columns={k: v for k, v in rename_map.items() if k in daily.columns})

    daily = daily.sort_values(["ts_code", "report_date"])

    # 仅保留可能有数据命中的交易日范围，避免全历史无效遍历
    min_report_date = str(daily["report_date"].min())[:8]
    max_report_date = str(daily["report_date"].max())[:8]
    effective_trade_dates = [
        d
        for d in trading_dates
        if d >= min_report_date and d <= _offset_date(max_report_date, 90)
    ]

    if not effective_trade_dates:
        logger.info("一致预期修正日频查询表构建完成: 0 个交易日有数据（交易日不在报告覆盖范围）")
        return {}

    logger.info(
        f"一致预期修正构建: {daily['ts_code'].nunique()} 只股票, "
        f"{daily['report_date'].nunique()} 个报告日, "
        f"有效交易日 {len(effective_trade_dates)}/{len(trading_dates)}"
    )

    # 预计算交易日整数边界，避免循环中重复日期偏移计算
    effective_trade_ord = np.array([int(d) for d in effective_trade_dates], dtype=np.int32)
    recent_start_ord = np.array([int(_offset_date(d, -90)) for d in effective_trade_dates], dtype=np.int32)
    earlier_start_ord = np.array([int(_offset_date(d, -120)) for d in effective_trade_dates], dtype=np.int32)
    earlier_end_ord = np.array([int(_offset_date(d, -30)) for d in effective_trade_dates], dtype=np.int32)
    analyst_recent_start_ord = np.array([int(_offset_date(d, -60)) for d in effective_trade_dates], dtype=np.int32)
    analyst_prior_start_ord = np.array([int(_offset_date(d, -120)) for d in effective_trade_dates], dtype=np.int32)
    analyst_prior_end_ord = np.array([int(_offset_date(d, -60)) for d in effective_trade_dates], dtype=np.int32)

    # 改为按股票遍历，并仅处理该股票的活跃交易日期窗口
    result_rows_by_date: Dict[str, List[dict]] = {}
    grouped_by_stock = list(daily.groupby("ts_code", sort=False))

    for stock_idx, (ts_code, grp) in enumerate(grouped_by_stock, 1):
        report_ord = grp["report_date"].astype(np.int32).to_numpy()
        if report_ord.size == 0:
            continue

        stock_start_idx = int(np.searchsorted(effective_trade_ord, report_ord[0], side="left"))
        stock_end_cutoff = int(_offset_date(str(report_ord[-1]), 90))
        stock_end_idx = int(np.searchsorted(effective_trade_ord, stock_end_cutoff, side="right"))

        if stock_start_idx >= stock_end_idx:
            continue

        eps_vals = (
            pd.to_numeric(grp["eps_pred_mean"], errors="coerce").to_numpy(dtype=float)
            if "eps_pred_mean" in grp.columns
            else None
        )
        analyst_vals = (
            pd.to_numeric(grp["analyst_count"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if "analyst_count" in grp.columns
            else np.ones(report_ord.size, dtype=float)
        )

        for td_idx in range(stock_start_idx, stock_end_idx):
            recent_l = int(np.searchsorted(report_ord, recent_start_ord[td_idx], side="left"))
            recent_r = int(np.searchsorted(report_ord, effective_trade_ord[td_idx], side="right"))
            recent_count = recent_r - recent_l
            if recent_count < 3:
                continue

            earlier_l = int(np.searchsorted(report_ord, earlier_start_ord[td_idx], side="left"))
            earlier_r = int(np.searchsorted(report_ord, earlier_end_ord[td_idx], side="right"))
            earlier_count = max(earlier_r - earlier_l, 0)

            trade_date = effective_trade_dates[td_idx]
            row = {"ts_code": ts_code}

            if eps_vals is not None:
                recent_eps = eps_vals[recent_l:recent_r]
                eps_mean = _safe_nanmean(recent_eps)

                recent_std = float(np.nanstd(recent_eps, ddof=1)) if recent_eps.size >= 2 else np.nan
                if recent_count >= 5 and not np.isnan(recent_std) and recent_std > 0 and abs(eps_mean) > 1e-6:
                    row["cons_eps_dispersion"] = float(recent_std / abs(eps_mean))
                else:
                    row["cons_eps_dispersion"] = np.nan
            else:
                row["cons_eps_dispersion"] = np.nan

            recent_analyst_total = float(np.nansum(analyst_vals[recent_l:recent_r]))
            analyst_recent_l = int(np.searchsorted(report_ord, analyst_recent_start_ord[td_idx], side="left"))
            analyst_prior_l = int(np.searchsorted(report_ord, analyst_prior_start_ord[td_idx], side="left"))
            analyst_prior_r = int(np.searchsorted(report_ord, analyst_prior_end_ord[td_idx], side="right"))
            recent_analyst_total = float(np.nansum(analyst_vals[analyst_recent_l:recent_r]))
            prior_analyst_total = float(np.nansum(analyst_vals[analyst_prior_l:analyst_prior_r]))
            recent_window_days = 60.0
            prior_window_days = 60.0
            recent_density = recent_analyst_total / recent_window_days
            prior_density = prior_analyst_total / prior_window_days
            if prior_density <= 0:
                row["cons_analyst_count_chg"] = np.nan if recent_density <= 0 else float(recent_density)
            else:
                row["cons_analyst_count_chg"] = float((recent_density - prior_density) / prior_density)

            latest_report = str(report_ord[recent_r - 1])
            row[CONSENSUS_REVISION_FRESHNESS_COL] = _days_between(latest_report, trade_date)

            result_rows_by_date.setdefault(trade_date, []).append(row)

        if stock_idx % 200 == 0 or stock_idx == len(grouped_by_stock):
            logger.info(
                "一致预期修正进度: 股票 {}/{}（当前: {}）",
                stock_idx,
                len(grouped_by_stock),
                ts_code,
            )

    result_dict: Dict[str, pd.DataFrame] = {
        d: pd.DataFrame(rows) for d, rows in result_rows_by_date.items() if rows
    }

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


def _pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """按优先级返回第一个存在的列名。"""
    for col in candidates:
        if col in df.columns:
            return col
    return None

def _safe_nanmean(values: np.ndarray) -> float:
    """对全 NaN/空数组安全求均值，避免 RuntimeWarning。"""
    if values is None or values.size == 0:
        return np.nan
    mask = ~np.isnan(values)
    if not np.any(mask):
        return np.nan
    return float(values[mask].mean())
