"""一致预期修正因子模块

基于 report_rc（一致预期研报）按 report_date 聚合构建时序修正信号。
当前已有一致预期基础因子（cons_eps_mean_fy1 等），本模块补充时间序列维度的
"修正方向/加速度/分歧度"信号——这些是 A 股实证中最有效的负向预警因子。

核心信号：
- EPS 修正加速度：修正本身在加速还是减速
- 分析师分歧度：预测标准差/均值（>0.3 提示不确定性高）
- 分歧度变化：分歧度扩大 = 风险上升
- 修正目标价上行空间：当前价距修正窗口目标价的距离
- 研报覆盖数量变化：撤出覆盖 = 强烈负向（列名保留 analyst 以兼容旧模型）
- 评级上调比例：边际情绪改善
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd
from ..data.report_rc import deduplicate_report_rc

# 一致预期修正因子输出列
CONSENSUS_REVISION_COLS = [
    "cons_eps_revision_accel",  # EPS 修正加速度（修正变化率）
    "cons_eps_dispersion",  # 分析师 EPS 预测分歧度
    "cons_eps_dispersion_chg",  # 分歧度月度变化
    "cons_revision_target_upside",  # 修正窗口目标价上行空间
    "cons_target_upside_chg",  # 目标价上行空间月度变化
    "cons_analyst_count_chg",  # 研报覆盖数月度变化（兼容旧列名）
    "cons_rating_upgrade_ratio",  # 近 30 日评级上调占比
]

CONSENSUS_REVISION_FRESHNESS_COL = "cons_revision_freshness_days"
_MAX_STATE_AGE_DAYS = 365


def _safe_nanstd(values: np.ndarray, ddof: int = 1) -> float:
    """安全计算忽略 NaN 的标准差。

    当有效样本数不足以支撑给定 ddof 时返回 NaN，避免 numpy 发出
    "Degrees of freedom <= 0 for slice" 的 RuntimeWarning。
    """
    if values is None:
        return np.nan

    valid_values = values[~np.isnan(values)]
    if valid_values.size <= ddof:
        return np.nan
    return float(np.nanstd(valid_values, ddof=ddof))


def _build_anchor_metrics(
    report_ord: np.ndarray,
    coverage_ord: np.ndarray,
    eps_vals: Optional[np.ndarray],
    target_vals: Optional[np.ndarray],
    anchor_ord: int,
) -> Optional[Dict[str, float]]:
    """按最新研报日锚定窗口，构造一只股票的修正状态。"""
    recent_start_ord = int(_offset_date(str(anchor_ord), -90))
    earlier_start_ord = int(_offset_date(str(anchor_ord), -120))
    earlier_end_ord = int(_offset_date(str(anchor_ord), -30))

    recent_l = int(np.searchsorted(report_ord, recent_start_ord, side="left"))
    recent_r = int(np.searchsorted(report_ord, anchor_ord, side="right"))
    recent_count = recent_r - recent_l
    if recent_count < 3:
        return None

    earlier_l = int(np.searchsorted(report_ord, earlier_start_ord, side="left"))
    earlier_r = int(np.searchsorted(report_ord, earlier_end_ord, side="right"))
    earlier_count = max(earlier_r - earlier_l, 0)
    row: Dict[str, float] = {}

    if eps_vals is not None:
        recent_eps = eps_vals[recent_l:recent_r]
        eps_mean = _safe_nanmean(recent_eps)
        recent_std = _safe_nanstd(recent_eps, ddof=1)
        if (
            recent_count >= 5
            and not np.isnan(recent_std)
            and recent_std > 0
            and abs(eps_mean) > 1e-6
        ):
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
            earlier_std = _safe_nanstd(earlier_eps, ddof=1)
            earlier_mean = _safe_nanmean(earlier_eps)
            if (
                not np.isnan(earlier_std)
                and earlier_std > 0
                and not np.isnan(earlier_mean)
                and abs(earlier_mean) > 1e-6
            ):
                earlier_disp = earlier_std / abs(earlier_mean)
            else:
                earlier_disp = np.nan
            current_disp = row["cons_eps_dispersion"]
            row["cons_eps_dispersion_chg"] = (
                float(current_disp - earlier_disp)
                if not np.isnan(current_disp) and not np.isnan(earlier_disp)
                else np.nan
            )
        else:
            row["cons_eps_dispersion_chg"] = np.nan
    else:
        row["cons_eps_dispersion"] = np.nan
        row["cons_eps_revision_accel"] = np.nan
        row["cons_eps_dispersion_chg"] = np.nan

    target_mean = np.nan
    if target_vals is not None:
        recent_target = target_vals[recent_l:recent_r]
        target_mean = _safe_nanmean(recent_target)
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
        row["cons_target_upside_chg"] = np.nan
    row["_target_mean"] = target_mean

    recent_report_l = int(np.searchsorted(coverage_ord, recent_start_ord, side="left"))
    recent_report_r = int(np.searchsorted(coverage_ord, anchor_ord, side="right"))
    earlier_report_l = int(np.searchsorted(coverage_ord, earlier_start_ord, side="left"))
    earlier_report_r = int(np.searchsorted(coverage_ord, earlier_end_ord, side="right"))
    recent_report_count = recent_report_r - recent_report_l
    earlier_report_count = max(earlier_report_r - earlier_report_l, 0)
    compare_earlier = earlier_report_count if earlier_report_count > 0 else recent_report_count
    row["cons_analyst_count_chg"] = float(
        (recent_report_count - compare_earlier) / max(compare_earlier, 1)
    )

    target_upside_chg = row["cons_target_upside_chg"]
    row["cons_rating_upgrade_ratio"] = (
        float(1.0 if target_upside_chg > 0.02 else 0.0)
        if not np.isnan(target_upside_chg)
        else np.nan
    )
    return row


def build_consensus_revision_lookup_by_date(
    report_rc_raw: pd.DataFrame,
    trading_dates: List[str],
    daily_data_lookup: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """基于 report_rc 原始数据构建一致预期修正日频查询表

    对每只股票按最新研报日锚定最近 90 日窗口计算修正状态，最多保留 365 日。

    Args:
        report_rc_raw: report_rc 原始数据，需包含完整研报身份列、quarter，
                   以及 rec_fore_Netprofit/rec_target 或 np/目标价字段
        trading_dates: 交易日列表
        daily_data_lookup: 日线数据查询表 {trade_date: DataFrame}，
                          用于获取未复权 close 计算目标价上行空间

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, cons_eps_revision_accel, ...)}
    """
    if report_rc_raw is None or len(report_rc_raw) == 0:
        logger.warning("report_rc 数据为空，跳过一致预期修正因子构建")
        return {}

    df = report_rc_raw.copy()
    df["report_date"] = normalize_series_to_yyyymmdd(df["report_date"])
    df = df[df["report_date"].astype("string").str.fullmatch(r"\d{8}", na=False)].copy()
    df = deduplicate_report_rc(
        df,
        include_quarter=True,
        require_full_identity=True,
    )
    report_identity_df = deduplicate_report_rc(df, include_quarter=False)

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
            df[target_proxy_col] = pd.concat([max_price, min_price], axis=1).mean(
                axis=1,
                skipna=True,
            )
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
        logger.warning("report_rc 缺少净利润预测与目标价相关列，无法构建一致预期修正因子")
        return {}

    logger.info(
        "一致预期修正字段映射: eps={}, target={}",
        eps_source_col if eps_source_col is not None else "缺失",
        target_source_col if target_source_col is not None else (target_proxy_col or "缺失"),
    )

    daily = df.groupby(["ts_code", "report_date"], as_index=False).agg(agg_cols)
    # 展平多级列名
    daily.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in daily.columns]

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
        for d in sorted(set(trading_dates))
        if d >= min_report_date and d <= _offset_date(max_report_date, _MAX_STATE_AGE_DAYS)
    ]

    if not effective_trade_dates:
        logger.info("一致预期修正日频查询表构建完成: 0 个交易日有数据（交易日不在报告覆盖范围）")
        return {}

    logger.info(
        f"一致预期修正构建: {daily['ts_code'].nunique()} 只股票, "
        f"{daily['report_date'].nunique()} 个报告日, "
        f"有效交易日 {len(effective_trade_dates)}/{len(trading_dates)}"
    )

    # 预计算交易日与唯一研报日期，避免把同一报告日误当成一位分析师。
    effective_trade_ord = np.array([int(d) for d in effective_trade_dates], dtype=np.int32)
    coverage_by_stock = {
        str(ts_code): group["report_date"].astype(np.int32).sort_values().to_numpy()
        for ts_code, group in report_identity_df.groupby("ts_code", sort=False)
    }
    close_lookup = _build_close_lookup(daily_data_lookup, effective_trade_dates)

    # 改为按股票遍历，并仅处理该股票的活跃交易日期窗口
    result_rows_by_date: Dict[str, List[dict]] = {}
    grouped_by_stock = list(daily.groupby("ts_code", sort=False))

    for stock_idx, (ts_code, grp) in enumerate(grouped_by_stock, 1):
        report_ord = grp["report_date"].astype(np.int32).to_numpy()
        if report_ord.size == 0:
            continue

        stock_start_idx = int(np.searchsorted(effective_trade_ord, report_ord[0], side="left"))
        stock_end_cutoff = int(_offset_date(str(report_ord[-1]), _MAX_STATE_AGE_DAYS))
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
        coverage_ord = coverage_by_stock.get(str(ts_code), report_ord)
        anchor_cache: Dict[int, Optional[Dict[str, float]]] = {}

        for td_idx in range(stock_start_idx, stock_end_idx):
            trade_date = effective_trade_dates[td_idx]
            visible_r = int(np.searchsorted(report_ord, effective_trade_ord[td_idx], side="right"))
            if visible_r == 0:
                continue
            anchor_ord = int(report_ord[visible_r - 1])
            freshness_days = _days_between(str(anchor_ord), trade_date)
            if freshness_days > _MAX_STATE_AGE_DAYS:
                continue
            if anchor_ord not in anchor_cache:
                anchor_cache[anchor_ord] = _build_anchor_metrics(
                    report_ord,
                    coverage_ord,
                    eps_vals,
                    target_vals,
                    anchor_ord,
                )
            anchor_metrics = anchor_cache[anchor_ord]
            if anchor_metrics is None:
                continue

            row = dict(anchor_metrics)
            target_mean = row.pop("_target_mean")
            close = close_lookup.get(trade_date, {}).get(str(ts_code), np.nan)
            row["cons_revision_target_upside"] = (
                float(target_mean / close - 1.0)
                if not np.isnan(target_mean) and not np.isnan(close) and close > 0
                else np.nan
            )
            row["ts_code"] = ts_code
            row[CONSENSUS_REVISION_FRESHNESS_COL] = freshness_days

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


def _build_close_lookup(
    daily_data_lookup: Optional[Dict[str, pd.DataFrame]],
    trade_dates: List[str],
) -> Dict[str, Dict[str, float]]:
    """构建未复权 close 的日内哈希索引。"""
    if daily_data_lookup is None:
        return {}

    lookup: Dict[str, Dict[str, float]] = {}
    for trade_date in trade_dates:
        price_df = daily_data_lookup.get(trade_date)
        if price_df is None or len(price_df) == 0:
            continue
        if "ts_code" not in price_df.columns or "close" not in price_df.columns:
            continue

        ts_series = price_df["ts_code"].astype(str)
        close_series = pd.to_numeric(price_df["close"], errors="coerce")
        day_map: Dict[str, float] = {}
        for ts_code, close in zip(ts_series, close_series):
            if not np.isnan(close):
                day_map[str(ts_code)] = float(close)
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
