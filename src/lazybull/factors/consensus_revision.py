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

    # 兼容两套 report_rc 口径：
    # 1) rec_fore_Netprofit / rec_target（部分环境）
    # 2) np / tp + max_price/min_price（当前主口径）
    eps_source_col = _pick_first_existing_col(df, ["rec_fore_Netprofit", "np", "tp"])
    target_source_col = _pick_first_existing_col(df, ["rec_target"])

    # 若无单列目标价，使用 max/min 目标价中位作为回退口径
    target_proxy_col = None
    if target_source_col is None and ("max_price" in df.columns or "min_price" in df.columns):
        max_price = pd.to_numeric(df.get("max_price"), errors="coerce")
        min_price = pd.to_numeric(df.get("min_price"), errors="coerce")
        target_proxy_col = "target_price_proxy"
        if "max_price" in df.columns and "min_price" in df.columns:
            df[target_proxy_col] = (max_price + min_price) / 2.0
        elif "max_price" in df.columns:
            df[target_proxy_col] = max_price
        else:
            df[target_proxy_col] = min_price

    # 按 ts_code + report_date 分组聚合每日研报
    # 对同一日多份研报取均值
    agg_cols = {}
    if eps_source_col is not None:
        df[eps_source_col] = pd.to_numeric(df[eps_source_col], errors="coerce")
        agg_cols[eps_source_col] = ["mean", "std", "count"]
    if target_source_col is not None:
        df[target_source_col] = pd.to_numeric(df[target_source_col], errors="coerce")
        agg_cols[target_source_col] = "mean"
    elif target_proxy_col is not None:
        agg_cols[target_proxy_col] = "mean"

    if not agg_cols:
        logger.warning(
            "report_rc 缺少净利润预测与目标价相关列，无法构建一致预期修正因子"
        )
        return {}

    logger.info(
        "一致预期修正字段映射: eps={}, target={}",
        eps_source_col if eps_source_col is not None else "缺失",
        target_source_col if target_source_col is not None else (target_proxy_col or "缺失"),
    )

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
    if target_source_col is not None:
        rename_map[f"{target_source_col}_mean"] = "target_price"
    elif target_proxy_col is not None:
        rename_map[f"{target_proxy_col}_mean"] = "target_price"
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

    # close_adj 按交易日+股票构建哈希查询，避免 DataFrame 反复过滤
    close_adj_lookup = _build_close_adj_lookup(daily_data_lookup, effective_trade_dates)

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
        target_vals = (
            pd.to_numeric(grp["target_price"], errors="coerce").to_numpy(dtype=float)
            if "target_price" in grp.columns
            else None
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
                row["cons_eps_mean_90d"] = eps_mean

                recent_std = float(np.nanstd(recent_eps, ddof=1)) if recent_eps.size >= 2 else np.nan
                if recent_count >= 5 and not np.isnan(recent_std) and recent_std > 0 and abs(eps_mean) > 1e-6:
                    row["cons_eps_dispersion"] = float(recent_std / abs(eps_mean))
                else:
                    row["cons_eps_dispersion"] = np.nan

                if recent_count >= 10:
                    mask = ~np.isnan(recent_eps)
                    if mask.sum() >= 3:
                        x = np.arange(recent_eps.size, dtype=float)
                        slope = np.polyfit(x[mask], recent_eps[mask], 1)[0]
                        eps_masked_mean = _safe_nanmean(recent_eps[mask])
                        row["cons_eps_revision_accel"] = float(slope / (abs(eps_masked_mean) + 1e-6))
                    else:
                        row["cons_eps_revision_accel"] = np.nan
                else:
                    row["cons_eps_revision_accel"] = np.nan

                if earlier_count >= 5:
                    earlier_eps = eps_vals[earlier_l:earlier_r]
                    earlier_std = float(np.nanstd(earlier_eps, ddof=1)) if earlier_eps.size >= 2 else np.nan
                    earlier_mean = _safe_nanmean(earlier_eps)
                    if not np.isnan(earlier_std) and earlier_std > 0 and not np.isnan(earlier_mean) and abs(earlier_mean) > 1e-6:
                        earlier_disp = earlier_std / abs(earlier_mean)
                    else:
                        earlier_disp = np.nan
                    cur_disp = row.get("cons_eps_dispersion", np.nan)
                    if not np.isnan(cur_disp) and not np.isnan(earlier_disp):
                        row["cons_eps_dispersion_chg"] = float(cur_disp - earlier_disp)
                    else:
                        row["cons_eps_dispersion_chg"] = np.nan
                else:
                    row["cons_eps_dispersion_chg"] = np.nan
            else:
                row["cons_eps_mean_90d"] = np.nan
                row["cons_eps_dispersion"] = np.nan
                row["cons_eps_revision_accel"] = np.nan
                row["cons_eps_dispersion_chg"] = np.nan

            if target_vals is not None:
                recent_target = target_vals[recent_l:recent_r]
                target_mean = _safe_nanmean(recent_target)
                close_adj = close_adj_lookup.get(trade_date, {}).get(ts_code, np.nan)
                if not np.isnan(target_mean) and not np.isnan(close_adj) and close_adj > 0:
                    row["cons_target_upside"] = float(target_mean / close_adj - 1.0)
                else:
                    row["cons_target_upside"] = np.nan

                if earlier_count >= 3:
                    earlier_target = target_vals[earlier_l:earlier_r]
                    earlier_target_mean = _safe_nanmean(earlier_target)
                    if (
                        not np.isnan(target_mean)
                        and not np.isnan(earlier_target_mean)
                        and abs(earlier_target_mean) > 1e-6
                    ):
                        row["cons_target_upside_chg"] = float(target_mean / earlier_target_mean - 1.0)
                    else:
                        row["cons_target_upside_chg"] = np.nan
                else:
                    row["cons_target_upside_chg"] = np.nan
            else:
                row["cons_target_upside"] = np.nan
                row["cons_target_upside_chg"] = np.nan

            compare_earlier = earlier_count if earlier_count > 0 else recent_count
            row["cons_analyst_count_chg"] = float((recent_count - compare_earlier) / max(compare_earlier, 1))

            target_upside_chg = row.get("cons_target_upside_chg", np.nan)
            if not np.isnan(target_upside_chg):
                row["cons_rating_upgrade_ratio"] = float(1.0 if target_upside_chg > 0.02 else 0.0)
            else:
                row["cons_rating_upgrade_ratio"] = np.nan

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


def _build_close_adj_lookup(
    daily_data_lookup: Optional[Dict[str, pd.DataFrame]],
    trade_dates: List[str],
) -> Dict[str, Dict[str, float]]:
    """构建 close_adj 的日内哈希索引: trade_date -> {ts_code: close_adj}。"""
    if daily_data_lookup is None:
        return {}

    lookup: Dict[str, Dict[str, float]] = {}
    for trade_date in trade_dates:
        price_df = daily_data_lookup.get(trade_date)
        if price_df is None or len(price_df) == 0:
            continue
        if "ts_code" not in price_df.columns or "close_adj" not in price_df.columns:
            continue

        ts_series = price_df["ts_code"].astype(str)
        close_series = pd.to_numeric(price_df["close_adj"], errors="coerce")
        day_map: Dict[str, float] = {}
        for ts_code, close_adj in zip(ts_series, close_series):
            if not np.isnan(close_adj):
                day_map[str(ts_code)] = float(close_adj)
        if day_map:
            lookup[trade_date] = day_map

    return lookup


def _safe_nanmean(values: np.ndarray) -> float:
    """对全 NaN/空数组安全求均值，避免 RuntimeWarning。"""
    if values is None or values.size == 0:
        return np.nan
    mask = ~np.isnan(values)
    if not np.any(mask):
        return np.nan
    return float(values[mask].mean())
