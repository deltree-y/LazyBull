"""一致预期修正因子模块（v2 重做）

基于 report_rc（一致预期研报）按 report_date 构建时序修正信号，与基础一致预期因子互补：
基础因子给出水平状态，本模块给出修正速度/分歧变化/覆盖与评级边际变化。

v2 相对 v1 的经济语义修正：
- EPS 源列改回 ``eps``（v1 误用 ``np`` 净利润，且未区分预测期），并按预测财年
  （FY）分组，优先 FY1、回退 FY0，杜绝多预测期混合。
- ``cons_eps_dispersion`` 改为"同日同 FY 研报级分歧度"的时间平均
  （v1 先按报告日聚合再取窗口 std，衡量的是预测随时间的波动，并非分析师分歧）。
- ``cons_eps_revision_accel`` 改为按报告日真实日历时间的一阶斜率
  （v1 按研报行序号拟合，与时间无关）。
- ``cons_rating_upgrade_ratio`` 改为真实读取 ``rating`` 列
  （v1 是目标价变化 >= 2% 的 0/1 别名）。
- 删除 ``cons_revision_target_upside``（与基础 ``cons_target_upside`` 高度重合）。
- 输出截面 1%/99% winsorize，避免极端值牵引下游 Z-Score。
- 新增哨兵列 ``cons_revision_schema_v2``，值恒为 schema 版本号，
  用于让旧语义缓存（缺该列）在 ensure/schema 校验下强制重建。

保留 v1 契约：输出列名不变（存量模型兼容）、报告日锚定 90 日窗口、
状态保鲜 365 日、``cons_revision_freshness_days`` freshness 与事件衰减、
同一研报多预测期行不放大覆盖计数。

核心信号：
- cons_eps_revision_accel  : EPS 修正速度（近 90 日按真实日历天数斜率，锚定年 FY1 优先）
- cons_eps_dispersion      : 分析师 EPS 分歧度（同日同财年研报级，窗口均值）
- cons_eps_dispersion_chg  : 分歧度变化（近 30 日 vs 此前 90 日）
- cons_target_upside_chg   : 目标价均值变化（近 30 日 / 此前 90 日 - 1）
- cons_analyst_count_chg   : 研报覆盖数变化（近 30 日 vs 此前 90 日折算，研报身份去重）
- cons_rating_upgrade_ratio: 评级上调占比（近 30 日评级分高于此前 90 日均值的研报占比）
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd
from ..data.report_rc import deduplicate_report_rc, report_rc_key_columns
from .consensus import parse_quarter_year, rating_to_score

# 一致预期修正因子输出列（六值列；目标价水平列 v2 起移除）
CONSENSUS_REVISION_COLS = [
    "cons_eps_revision_accel",  # EPS 修正速度（近 90 日按日历时间斜率）
    "cons_eps_dispersion",  # 分析师 EPS 分歧度（同日同 FY 研报级）
    "cons_eps_dispersion_chg",  # 分歧度变化（近 30 日 vs 此前 90 日）
    "cons_target_upside_chg",  # 目标价均值变化（近 30 日 / 此前 90 日 - 1）
    "cons_analyst_count_chg",  # 研报覆盖数变化（近 30 日 vs 此前 90 日折算）
    "cons_rating_upgrade_ratio",  # 评级上调占比（真实读取 rating）
]

CONSENSUS_REVISION_FRESHNESS_COL = "cons_revision_freshness_days"
CONSENSUS_REVISION_VERSION_COL = "cons_revision_schema_v2"
CONSENSUS_REVISION_SCHEMA_VERSION = 2

_MAX_STATE_AGE_DAYS = 365
_RECENT_WINDOW_DAYS = 90
_RECENT30_WINDOW_DAYS = 30
_BASELINE_START_OFFSET_DAYS = -120
_BASELINE_END_OFFSET_DAYS = -30

_WINSOR_LIMITS = (0.01, 0.99)
_WINSOR_MIN_VALID = 20

_MIN_RECENT_REPORT_ROWS = 3
_MIN_DISPERSION_DAYS = 3
_MIN_DISPERSION_REPORTS_PER_DAY = 2
_MIN_ACCEL_DAYS = 5
_MIN_TARGET_EARLIER_VALID = 3
_MIN_RATING_EARLIER_VALID = 3


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


def _safe_nanmean(values: np.ndarray) -> float:
    """对全 NaN/空数组安全求均值，避免 RuntimeWarning。"""
    if values is None or values.size == 0:
        return np.nan
    mask = ~np.isnan(values)
    if not np.any(mask):
        return np.nan
    return float(values[mask].mean())


def _build_anchor_metrics(
    report_ord: np.ndarray,
    eps_vals: np.ndarray,
    q_year_vals: np.ndarray,
    ord_days_vals: np.ndarray,
    coverage_ord: np.ndarray,
    target_ord: np.ndarray,
    target_vals: np.ndarray,
    rating_ord: np.ndarray,
    rating_vals: np.ndarray,
    anchor_ord: int,
) -> Optional[Dict[str, float]]:
    """按最新研报日锚定窗口，构造一只股票的修正状态。

    Args:
        report_ord: 研报行级报告日期（含多预测期行，int32 升序）
        eps_vals: 研报行级 eps 预测值（可为 NaN）
        q_year_vals: 研报行级绝对预测财年（quarter 解析年份；无法解析为 -99）
        ord_days_vals: 研报行级报告日距 epoch 的天数（真实日历天数，供斜率回归）
        coverage_ord: 研报身份去重后的报告日期（升序，覆盖计数专用）
        target_ord: 研报身份去重后的报告日期（升序，目标价专用）
        target_vals: 研报级目标价（与 target_ord 对齐）
        rating_ord: 研报身份去重后的报告日期（升序，评级专用）
        rating_vals: 研报级评级得分（与 rating_ord 对齐）
        anchor_ord: 最新可见研报日期
    """
    recent_start_ord = int(_offset_date(str(anchor_ord), -_RECENT_WINDOW_DAYS))
    recent30_start_ord = int(_offset_date(str(anchor_ord), -_RECENT30_WINDOW_DAYS))
    earlier_start_ord = int(_offset_date(str(anchor_ord), _BASELINE_START_OFFSET_DAYS))
    earlier_end_ord = int(_offset_date(str(anchor_ord), _BASELINE_END_OFFSET_DAYS))

    recent_l = int(np.searchsorted(report_ord, recent_start_ord, side="left"))
    recent_r = int(np.searchsorted(report_ord, anchor_ord, side="right"))
    if recent_r - recent_l < _MIN_RECENT_REPORT_ROWS:
        return None

    # 选择目标绝对财年：锚定报告年的 FY1（anchor_year+1）与 FY0（anchor_year）按
    # 窗口内有效 eps 覆盖报告日数多者优先（持平取 FY1）。绝对财年杜绝跨年报告的不同
    # FY1（如 2025 与 2026 财年）被混入同一序列产生虚假修正；日数门槛对齐分歧度
    # 最低要求（3 个报告日），避免少量样本压掉充足回退财年。
    anchor_year = int(anchor_ord // 10000)

    def _fy_report_days(slice_l: int, slice_r: int, fy_year: int) -> int:
        mask = (q_year_vals[slice_l:slice_r] == fy_year) & ~np.isnan(eps_vals[slice_l:slice_r])
        if mask.sum() == 0:
            return 0
        return int(np.unique(report_ord[slice_l:slice_r][mask]).size)

    fy1_days = _fy_report_days(recent_l, recent_r, anchor_year + 1)
    fy0_days = _fy_report_days(recent_l, recent_r, anchor_year)
    if fy1_days >= _MIN_DISPERSION_DAYS and fy1_days >= fy0_days:
        target_fy_year = anchor_year + 1
    elif fy0_days >= _MIN_DISPERSION_DAYS:
        target_fy_year = anchor_year
    else:
        target_fy_year = None

    row: Dict[str, float] = {}

    def _eps_dispersion_in_slice(slice_l: int, slice_r: int) -> Optional[float]:
        """窗口内同日同财年研报级分歧度的时间平均。"""
        if target_fy_year is None or slice_r - slice_l <= 0:
            return None
        mask = (q_year_vals[slice_l:slice_r] == target_fy_year) & ~np.isnan(
            eps_vals[slice_l:slice_r]
        )
        if mask.sum() == 0:
            return None
        eps_v = eps_vals[slice_l:slice_r][mask]
        ord_v = report_ord[slice_l:slice_r][mask]
        disps: List[float] = []
        for day in np.unique(ord_v):
            day_vals = eps_v[ord_v == day]
            if day_vals.size < _MIN_DISPERSION_REPORTS_PER_DAY:
                continue
            day_mean = float(np.mean(day_vals))
            day_std = float(np.std(day_vals, ddof=1))
            if abs(day_mean) > 1e-6 and day_std > 0:
                disps.append(day_std / abs(day_mean))
        if len(disps) < _MIN_DISPERSION_DAYS:
            return None
        return float(np.mean(disps))

    def _eps_accel_in_slice(slice_l: int, slice_r: int) -> Optional[float]:
        """窗口内按真实日历天数拟合 EPS 预测中值的一阶斜率。"""
        if target_fy_year is None or slice_r - slice_l < _MIN_ACCEL_DAYS:
            return None
        mask = (q_year_vals[slice_l:slice_r] == target_fy_year) & ~np.isnan(
            eps_vals[slice_l:slice_r]
        )
        if mask.sum() < _MIN_ACCEL_DAYS:
            return None
        eps_v = eps_vals[slice_l:slice_r][mask]
        days_v = ord_days_vals[slice_l:slice_r][mask]
        by_day: Dict[int, List[float]] = {}
        for day, val in zip(days_v.tolist(), eps_v.tolist()):
            by_day.setdefault(int(day), []).append(float(val))
        days = np.array(sorted(by_day.keys()), dtype=float)
        medians = np.array([float(np.median(by_day[int(d)])) for d in days], dtype=float)
        if medians.size < _MIN_ACCEL_DAYS:
            return None
        mean_val = float(np.mean(medians))
        if abs(mean_val) < 1e-9:
            return None
        slope = float(np.polyfit(days, medians, 1)[0])
        return slope / abs(mean_val)

    # EPS 分歧度与修正速度（近 90 日窗口）
    row["cons_eps_dispersion"] = _eps_dispersion_in_slice(recent_l, recent_r)
    row["cons_eps_revision_accel"] = _eps_accel_in_slice(recent_l, recent_r)

    # 分歧度变化：近 30 日 vs 此前 90 日。earlier 终点用 side="left"（严格小于
    # anchor-30），与 recent30 起点不重叠，避免第 -30 日被重复计入两侧窗口。
    recent30_l = int(np.searchsorted(report_ord, recent30_start_ord, side="left"))
    earlier_l = int(np.searchsorted(report_ord, earlier_start_ord, side="left"))
    earlier_r = int(np.searchsorted(report_ord, earlier_end_ord, side="left"))
    current_disp = _eps_dispersion_in_slice(recent30_l, recent_r)
    earlier_disp = _eps_dispersion_in_slice(earlier_l, earlier_r)
    row["cons_eps_dispersion_chg"] = (
        float(current_disp - earlier_disp)
        if current_disp is not None and earlier_disp is not None
        else np.nan
    )

    # 目标价均值变化：近 30 日 vs 此前 90 日（研报级，同研报多预测期行不重复加权）
    target_recent_l = int(np.searchsorted(target_ord, recent30_start_ord, side="left"))
    target_recent_r = int(np.searchsorted(target_ord, anchor_ord, side="right"))
    target_earlier_l = int(np.searchsorted(target_ord, earlier_start_ord, side="left"))
    target_earlier_r = int(np.searchsorted(target_ord, earlier_end_ord, side="left"))
    recent_target = target_vals[target_recent_l:target_recent_r]
    earlier_target = target_vals[target_earlier_l:target_earlier_r]
    recent_target_valid = recent_target[~np.isnan(recent_target)]
    earlier_target_valid = earlier_target[~np.isnan(earlier_target)]
    if recent_target_valid.size >= 1 and earlier_target_valid.size >= _MIN_TARGET_EARLIER_VALID:
        earlier_target_mean = float(np.mean(earlier_target_valid))
        if abs(earlier_target_mean) > 1e-6:
            row["cons_target_upside_chg"] = (
                float(np.mean(recent_target_valid)) / earlier_target_mean - 1.0
            )
        else:
            row["cons_target_upside_chg"] = np.nan
    else:
        row["cons_target_upside_chg"] = np.nan

    # 研报覆盖数变化：近 30 日 vs 此前 90 日（折算为 30 日等效）
    cov_recent30_l = int(np.searchsorted(coverage_ord, recent30_start_ord, side="left"))
    cov_recent30_r = int(np.searchsorted(coverage_ord, anchor_ord, side="right"))
    cov_earlier_l = int(np.searchsorted(coverage_ord, earlier_start_ord, side="left"))
    cov_earlier_r = int(np.searchsorted(coverage_ord, earlier_end_ord, side="left"))
    recent30_count = cov_recent30_r - cov_recent30_l
    baseline_equiv = max(cov_earlier_r - cov_earlier_l, 0) / 3.0
    if baseline_equiv > 0:
        row["cons_analyst_count_chg"] = float((recent30_count - baseline_equiv) / baseline_equiv)
    else:
        row["cons_analyst_count_chg"] = float(recent30_count) if recent30_count > 0 else 0.0

    # 评级上调占比：真实读取 rating，基线为此前 90 日研报级评分均值
    rating_recent30_l = int(np.searchsorted(rating_ord, recent30_start_ord, side="left"))
    rating_recent30_r = int(np.searchsorted(rating_ord, anchor_ord, side="right"))
    rating_earlier_l = int(np.searchsorted(rating_ord, earlier_start_ord, side="left"))
    rating_earlier_r = int(np.searchsorted(rating_ord, earlier_end_ord, side="left"))
    earlier_scores = rating_vals[rating_earlier_l:rating_earlier_r]
    recent_scores = rating_vals[rating_recent30_l:rating_recent30_r]
    earlier_scores_valid = earlier_scores[~np.isnan(earlier_scores)]
    recent_scores_valid = recent_scores[~np.isnan(recent_scores)]
    if earlier_scores_valid.size >= _MIN_RATING_EARLIER_VALID and recent_scores_valid.size >= 1:
        baseline_score = float(np.mean(earlier_scores_valid))
        upgrade_count = int((recent_scores_valid > baseline_score).sum())
        row["cons_rating_upgrade_ratio"] = float(upgrade_count / recent_scores_valid.size)
    else:
        row["cons_rating_upgrade_ratio"] = np.nan

    return row


def _winsorize_cross_section(
    day_df: pd.DataFrame,
    cols: List[str],
    limits: Tuple[float, float] = _WINSOR_LIMITS,
) -> pd.DataFrame:
    """按当日股票截面对指定列做 1%/99% winsorize，NaN 保持不动。"""
    result = day_df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        values = result[col].to_numpy(dtype=float, copy=True)
        valid = values[~np.isnan(values)]
        if valid.size < _WINSOR_MIN_VALID:
            continue
        lower, upper = float(np.quantile(valid, limits[0])), float(np.quantile(valid, limits[1]))
        result[col] = np.clip(values, lower, upper)
    return result


def build_consensus_revision_lookup_by_date(
    report_rc_raw: pd.DataFrame,
    trading_dates: List[str],
    daily_data_lookup: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """基于 report_rc 原始数据构建一致预期修正日频查询表

    对每只股票按最新研报日锚定最近 90 日窗口计算修正状态，最多保留 365 日。

    Args:
        report_rc_raw: report_rc 原始数据，需包含完整研报身份列、quarter 与 eps
        trading_dates: 交易日列表（YYYYMMDD 字符串）
        daily_data_lookup: v2 起不再使用（已移除目标价水平列），仅保留参数兼容

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, cons_eps_revision_accel, ...)}
    """
    if report_rc_raw is None or len(report_rc_raw) == 0:
        logger.warning("report_rc 数据为空，跳过一致预期修正因子构建")
        return {}

    df = report_rc_raw.copy()
    df["report_date"] = normalize_series_to_yyyymmdd(df["report_date"])
    df = df[df["report_date"].astype("string").str.fullmatch(r"\d{8}", na=False)].copy()

    if "eps" not in df.columns:
        raise ValueError(
            "report_rc 缺少 eps 列，无法构建一致预期修正因子（v2 不回退净利润口径）。"
            "请使用 --download report_rc --force 重下目标区间"
        )

    df = deduplicate_report_rc(
        df,
        include_quarter=True,
        require_full_identity=True,
    )

    # ── 研报级字段（同一研报多预测期行取均值，保证研报权重为 1）──
    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    max_price = (
        pd.to_numeric(df["max_price"], errors="coerce")
        if "max_price" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    min_price = (
        pd.to_numeric(df["min_price"], errors="coerce")
        if "min_price" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["_target_price"] = pd.concat([max_price, min_price], axis=1).mean(axis=1, skipna=True)
    if "rating" in df.columns:
        df["_rating_score"] = df["rating"].map(rating_to_score)
    else:
        df["_rating_score"] = np.nan

    # 财年定位: 解析研报中 quarter 的预测年份，供 EPS 按绝对财年分组过滤。
    # 锚定报告年的 FY1（anchor_year+1）优先、FY0（anchor_year）回退，见 _build_anchor_metrics。
    if "quarter" in df.columns:
        df["_q_year"] = df["quarter"].map(parse_quarter_year)
    else:
        df["_q_year"] = np.nan

    identity_key_cols = report_rc_key_columns(df, include_quarter=False)
    identity_group_keys = [c for c in identity_key_cols if c in df.columns]
    if identity_group_keys:
        identity_group = df.groupby(identity_group_keys, sort=False)
        df["_target_price"] = identity_group["_target_price"].transform("mean")
        df["_rating_score"] = identity_group["_rating_score"].transform("mean")
    report_identity_df = deduplicate_report_rc(df, include_quarter=False)

    if len(report_identity_df) == 0:
        logger.warning("report_rc 去重后为空，跳过一致预期修正因子构建")
        return {}

    # 行级与研报级数组都必须按 report_date 升序：searchsorted 窗口定位依赖有序性，
    # 去重结果继承原始分区顺序，必须显式排序，否则未来研报会被错误计入历史窗口（前视）。
    df = df.sort_values(["ts_code", "report_date"]).reset_index(drop=True)
    df["_report_ord"] = df["report_date"].astype(np.int32)
    df["_ord_days"] = (
        pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
        - pd.Timestamp("1970-01-01")
    ).dt.days
    report_identity_df = report_identity_df.sort_values(["ts_code", "report_date"]).reset_index(
        drop=True
    )
    report_identity_df["_ident_ord"] = report_identity_df["report_date"].astype(np.int32)

    sorted_trading_dates = sorted({d for d in trading_dates if d is not None})
    if not sorted_trading_dates:
        return {}

    # 预计算研报身份去重后的研报级数组（覆盖/目标价/评级），并行数组与日期同步排序
    ident_by_stock = {}
    for ts_code, group in report_identity_df.groupby("ts_code", sort=False):
        ord_arr = group["_ident_ord"].to_numpy(dtype=np.int32)
        target_arr = pd.to_numeric(group["_target_price"], errors="coerce").to_numpy(dtype=float)
        rating_arr = pd.to_numeric(group["_rating_score"], errors="coerce").to_numpy(dtype=float)
        order = np.argsort(ord_arr, kind="stable")
        ord_arr = ord_arr[order]
        target_arr = target_arr[order]
        rating_arr = rating_arr[order]
        ident_by_stock[str(ts_code)] = {
            "coverage_ord": ord_arr,
            "target_ord": ord_arr,
            "target_vals": target_arr,
            "rating_ord": ord_arr,
            "rating_vals": rating_arr,
        }

    # 仅保留可能有数据命中的交易日范围，避免全历史无效遍历
    min_report_ord = int(df["_report_ord"].min())
    max_report_ord = int(df["_report_ord"].max())
    effective_trade_dates = [
        d
        for d in sorted_trading_dates
        if min_report_ord <= int(d) <= int(_offset_date(str(max_report_ord), _MAX_STATE_AGE_DAYS))
    ]
    if not effective_trade_dates:
        logger.info("一致预期修正日频查询表构建完成: 0 个交易日有数据（交易日不在报告覆盖范围）")
        return {}

    logger.info(
        f"一致预期修正构建: {df['ts_code'].nunique()} 只股票, "
        f"{df['report_date'].nunique()} 个报告日, "
        f"有效交易日 {len(effective_trade_dates)}/{len(trading_dates)}"
    )

    effective_trade_ord = np.array([int(d) for d in effective_trade_dates], dtype=np.int32)
    result_rows_by_date: Dict[str, List[dict]] = {}
    grouped_by_stock = list(df.groupby("ts_code", sort=False))

    for stock_idx, (ts_code, grp) in enumerate(grouped_by_stock, 1):
        report_ord = grp["_report_ord"].to_numpy(dtype=np.int32)
        eps_vals = pd.to_numeric(grp["eps"], errors="coerce").to_numpy(dtype=float)
        q_year_vals = grp["_q_year"].fillna(-99).astype(np.int32).to_numpy(dtype=np.int32)
        ord_days_vals = grp["_ord_days"].to_numpy(dtype=float)
        ident = ident_by_stock.get(str(ts_code))
        if ident is None:
            continue
        coverage_ord = ident["coverage_ord"]
        target_ord = ident["target_ord"]
        target_vals = ident["target_vals"]
        rating_ord = ident["rating_ord"]
        rating_vals = ident["rating_vals"]

        stock_start_idx = int(np.searchsorted(effective_trade_ord, report_ord[0], side="left"))
        stock_end_cutoff = int(_offset_date(str(report_ord[-1]), _MAX_STATE_AGE_DAYS))
        stock_end_idx = int(np.searchsorted(effective_trade_ord, stock_end_cutoff, side="right"))

        if stock_start_idx >= stock_end_idx:
            continue

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
                    eps_vals,
                    q_year_vals,
                    ord_days_vals,
                    coverage_ord,
                    target_ord,
                    target_vals,
                    rating_ord,
                    rating_vals,
                    anchor_ord,
                )
            anchor_metrics = anchor_cache[anchor_ord]
            if anchor_metrics is None:
                continue

            row = dict(anchor_metrics)
            row["ts_code"] = ts_code
            row[CONSENSUS_REVISION_FRESHNESS_COL] = freshness_days
            row[CONSENSUS_REVISION_VERSION_COL] = CONSENSUS_REVISION_SCHEMA_VERSION

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

    # 截面 winsorize：按当日股票截面裁剪极端值，降低下游 Z-Score 被牵引的风险
    for day, day_df in result_dict.items():
        result_dict[day] = _winsorize_cross_section(day_df, list(CONSENSUS_REVISION_COLS))

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
