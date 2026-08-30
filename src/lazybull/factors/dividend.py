# -*- coding: utf-8 -*-
"""分红政策质量因子模块（v3：归母净利润口径 + 稠密事件编码）。

将 TuShare `dividend`（分红送股）原始事件表转换为日频截面查询表，
输出分红政策质量因子（状态因子 7 个 + 事件因子 2 个 + 附列）。

PIT 契约（防前视/防泄露）：
  - 仅 `div_proc=实施` 行进入因子；预案/决案行不参与任何因子计算；
  - 状态因子（连续性/稳定性/增长率/支付率/现金占比/历史股息率）的
    可用日统一为 `ex_date`（除息日，事实落地日，天然无前视）；
  - 事件因子 `dividend_days_to_ex_date` 使用未来 `ex_date` 的前提是
    该事件已公告（`imp_ann_date <= T`，缺失回退 `ann_date`）——
    除息日历随实施公告公开，非前视；
  - `dividend_recent_imp_ann_10d` 为纯回看窗口 [T-9, T]（含 T）：
    公告只影响其发布日及之后 10 个交易日的因子值，绝不回填发布日之前；
  - 送转调整仅使用 `ex_date <= T` 的送转事件（PIT 累计因子）。

口径（每股调整 DPS）：
  - 每股现金分红采用 TuShare `cash_div_tax`（税前）；
  - 历史每股分红按后续发生的送转（`stk_div`）后向调整到最新股本口径，
    消除跨年送转导致的每股分红不可比。

缺失语义：
  - 无分红历史 → 原始列 NaN（不填 0，0 是"不分红"信号）；
  - `dividend_hist_missing` 显式区分"从未分红"与"上市不足一年"；
  - continuity 分母 = 上市后年数（`stock_basic.list_date`），新股不背锅；
  - stability 需 >=3 个有效年度、median>0；growth 需窗口内年份全部有效。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..data.financial_statement_versions import fill_actual_announcement_date

# 分红政策质量因子输出列（handler 透传到特征层）
# 审查后首轮移除 dividend_cash_ratio（现金总额"元"与送转股数"股"量纲不可比）
DIVIDEND_COLS = [
    "dividend_continuity_5y",  # 近 5 个已结束财年正分红年占比
    "dividend_stability_5y",  # 年度 DPS 的 1 - MAD/中位数（稳健稳定度）
    "dividend_growth_3y",  # 3 年 DPS 有界对称增长率
    "dividend_growth_5y",  # 5 年 DPS 有界对称增长率
    "dividend_payout_ratio",  # 最近可见年度 现金分红总额/归母净利润
    "dividend_yield_hist_12m",  # 近 12 个月每股现金分红/未复权收盘价
    "dividend_days_to_ex_date",  # 距最近已公告未除息事件的自然日数；31 表示 30 日内无事件
    "dividend_recent_imp_ann_10d",  # 近 10 交易日实施公告计数（纯回看）
]

DIVIDEND_FRESHNESS_COL = "dividend_freshness_days"  # 距最近 ex_date 自然日差
DIVIDEND_HIST_MISSING_COL = "dividend_hist_missing"  # 从未分红=1/有历史=0/上市不足一年=NaN

# 分红政策因子 schema 哨兵：handler 对当日全截面恒写当前版本号（含无数据股票），
# 训练入口校验哨兵列缺失、NaN 或版本不符必须失败；语义重做时递增哨兵值。
DIVIDEND_POLICY_SCHEMA_VERSION = 3
DIVIDEND_POLICY_VERSION_COL = "dividend_schema_v1"

# lookup 中间列（近 12 个月每股现金分红累计分子，handler 除以未复权收盘价）
_CASH_12M_ADJ_COL = "dividend_cash_12m_adj"

# 数值稳定性参数
_EPS_CASH = 1e-8  # 正分红判定阈值（元/股）
_MIN_ABS_NET_PROFIT = 1e7  # 净利润分母经济尺度下限（元，1000 万元）
_CLIP_GROWTH = (-2.0, 2.0)  # 有界对称增长率裁剪界
_CLIP_STABILITY = (-1.0, 1.0)  # 稳定度裁剪界
_CLIP_PAYOUT = (-2.0, 2.0)  # 支付率裁剪界
_CLIP_DAYS_TO_EX = 30  # days_to_ex_date 有效上限（自然日）
_NO_UPCOMING_EX_DAYS = 31  # 上市满一年且未来 30 日内无已公告除息事件
_YIELD_WINDOW_DAYS = 365  # 历史股息率回看窗口（自然日）
_MIN_STABILITY_YEARS = 3  # stability 最少有效年度样本
_RECENT_IMP_ANN_WINDOW = 10  # 近期公告计数窗口（交易日，含 T）
_MAX_SCAN_FUTURE_EX = 10  # days_to_ex 向后扫描事件数上限
_WINDOW_YEARS = 5  # 连续性/稳定性年度窗口长度（以最新可见年度为锚向前）
_OUTPUT_DATE_CHUNK_SIZE = 64  # 分块向量化，限制批量构建临时 DataFrame 峰值


def _norm_date_series(s: pd.Series) -> pd.Series:
    """将日期列统一为 YYYYMMDD 字符串（容错 20240101 / 2024-01-01 / datetime）。"""
    return s.astype(str).str.strip().str.replace("-", "", regex=False).str[:8]


def _to_int_arr(s: pd.Series) -> np.ndarray:
    """YYYYMMDD 字符串 → int64 数组（仅用于排序/区间比较）。"""
    return s.to_numpy(dtype=np.int64)


def _to_dt_arr(s: pd.Series) -> np.ndarray:
    """YYYYMMDD 字符串 → datetime64[ns] 数组（用于自然日差计算）。"""
    return pd.to_datetime(s, format="%Y%m%d").to_numpy(dtype="datetime64[ns]")


def _bounded_symmetric_growth(curr: float, prev: float) -> float:
    """有界对称增长率：g = (curr-prev)/(0.5*(|curr|+|prev|))，裁剪 [-2, 2]。"""
    denom = 0.5 * (abs(curr) + abs(prev))
    if denom <= 0:
        return np.nan
    g = (curr - prev) / denom
    return float(np.clip(g, _CLIP_GROWTH[0], _CLIP_GROWTH[1]))


IncomeQ4Events = Tuple[np.ndarray, np.ndarray, np.ndarray]


def validate_income_for_dividend_payout(income_raw: Optional[pd.DataFrame]) -> None:
    """校验 income 是否含有可用于分红支付率的有效年报。"""
    if income_raw is None or len(income_raw) == 0:
        raise ValueError("缺少 raw/income 数据")
    required = {"ts_code", "end_date", "n_income_attr_p"}
    missing = sorted(required - set(income_raw.columns))
    if missing:
        raise ValueError("income 数据缺少支付率必需列: " + ", ".join(missing))
    if not ({"f_ann_date", "ann_date"} & set(income_raw.columns)):
        raise ValueError("income 数据缺少支付率必需公告日列: f_ann_date/ann_date")

    work = income_raw.copy()
    if "report_type" in work.columns:
        work = work[pd.to_numeric(work["report_type"], errors="coerce").eq(1)]
    if "f_ann_date" in work.columns:
        if "ann_date" in work.columns:
            work = fill_actual_announcement_date(work)
        available_date = _norm_date_series(work["f_ann_date"])
    else:
        available_date = _norm_date_series(work["ann_date"])
    end_date = _norm_date_series(work["end_date"])
    profit = pd.to_numeric(work["n_income_attr_p"], errors="coerce")
    valid = (
        work["ts_code"].notna()
        & end_date.str.endswith("1231", na=False)
        & available_date.str.match(r"^\d{8}$", na=False)
        & profit.notna()
    )
    if not valid.any():
        raise ValueError("income 数据不含可用于支付率的有效合并年报归母净利润")


def _build_income_q4_lookup(
    income_raw: Optional[pd.DataFrame],
) -> Dict[str, IncomeQ4Events]:
    """构建 per-stock Q4 归母净利润事件表。

    PIT：可用日取 f_ann_date（缺失回退 ann_date）；仅使用合并报表 report_type=1，
    同 (ts_code, end_date, avail_date) 冲突优先 update_flag=1。
    """
    lookup: Dict[str, IncomeQ4Events] = {}
    if income_raw is None or len(income_raw) == 0:
        return lookup
    work = income_raw.copy()
    required = {"ts_code", "end_date", "n_income_attr_p"}
    missing = sorted(required - set(work.columns))
    if missing or not ({"f_ann_date", "ann_date"} & set(work.columns)):
        details = missing or ["f_ann_date/ann_date"]
        logger.warning(
            "income 数据缺少支付率必需列，dividend_payout_ratio 将全 NaN: " + ", ".join(details)
        )
        return lookup
    if "report_type" in work.columns:
        work = work[pd.to_numeric(work["report_type"], errors="coerce").eq(1)]
    if "f_ann_date" not in work.columns:
        work["f_ann_date"] = work["ann_date"]
    work = fill_actual_announcement_date(work)
    for col in ("ann_date", "end_date", "f_ann_date"):
        if col in work.columns:
            work[col] = _norm_date_series(work[col])
    work = work.dropna(subset=["ts_code", "end_date", "f_ann_date"])
    work = work[work["f_ann_date"].str.match(r"^\d{8}$", na=False)]
    work["_year"] = work["end_date"].str[:4]
    # 仅 Q4（年报）行：归母净利润全年累计口径与年度分红总额财年对齐
    work = work[work["end_date"].str[4:8] == "1231"]
    work["_avail"] = work["f_ann_date"]
    work = work.dropna(subset=["_avail"])
    work["_avail_num"] = _to_int_arr(work["_avail"])
    work["n_income_attr_p"] = pd.to_numeric(work["n_income_attr_p"], errors="coerce")
    sort_cols = ["ts_code", "_year", "_avail_num"]
    if "update_flag" in work.columns:
        work["_is_latest"] = pd.to_numeric(work["update_flag"], errors="coerce").eq(1)
        sort_cols.append("_is_latest")
    work = work.sort_values(sort_cols, kind="mergesort")
    work = work.drop_duplicates(subset=["ts_code", "_year", "_avail_num"], keep="last")

    for ts_code, grp in work.groupby("ts_code", sort=False):
        grp = grp.sort_values("_avail_num")
        lookup[str(ts_code)] = (
            grp["_avail_num"].to_numpy(dtype=np.int64),
            grp["_year"].astype(int).to_numpy(dtype=np.int64),
            grp["n_income_attr_p"].to_numpy(dtype=float),
        )
    return lookup


def build_dividend_lookup_by_date(
    dividend_raw: Optional[pd.DataFrame],
    trading_dates: List[str],
    income_raw: Optional[pd.DataFrame] = None,
    list_date_map: Optional[Dict[str, str]] = None,
    calendar_dates: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """构建分红政策质量日频查询表。

    仅输出"当日存在可见分红事件"（最近 ex_date <= T）或"已公告未除息"的股票行；
    无分红历史的股票由 handler 按缺失语义展开（0/NaN + 缺失标记）。

    Args:
        dividend_raw: dividend 原始数据（含 div_proc/cash_div_tax/stk_div/
                      ann_date/imp_ann_date/ex_date/end_date/base_share）
        trading_dates: 交易日列表（YYYYMMDD 字符串，已排序，lookup 输出范围）
        income_raw: 利润表数据（可选，供 payout_ratio 取 Q4 归母净利润）
        list_date_map: {ts_code: list_date YYYYMMDD}（可选，用于上市年限）
        calendar_dates: 完整预热日历（含 trading_dates 前序，用于近 10 交易日
            回看窗口，保证批量构建与单日推理口径一致）；None 时用 trading_dates

    Returns:
        {trade_date: DataFrame(ts_code, 因子列...)}
    """
    if dividend_raw is None or len(dividend_raw) == 0:
        logger.warning("dividend 数据为空，跳过分红政策日频查询表构建")
        return {}
    if not trading_dates:
        return {}

    # ── 1. 仅实施行 + 日期/数值规范化 ─────────────────────────
    df = dividend_raw.copy()
    df["ts_code"] = df["ts_code"].astype(str)
    if "div_proc" in df.columns:
        df = df[df["div_proc"].astype(str).str.contains("实施", na=False)]
        if len(df) == 0:
            logger.warning("dividend 数据无实施行，跳过分红政策日频查询表构建")
            return {}
    # 回退必须先于日期规范化（NaN 字符串化后 fillna 失效，缺失 imp_ann_date 的实施行会被删）
    if "imp_ann_date" in df.columns:
        df["imp_ann_date"] = df["imp_ann_date"].fillna(df["ann_date"])
    for col in ("ann_date", "ex_date", "imp_ann_date", "end_date"):
        if col in df.columns:
            df[col] = _norm_date_series(df[col])
    df = df.dropna(subset=["ex_date"])
    df = df[df["ex_date"].str.match(r"^\d{8}$", na=False)]
    if len(df) == 0:
        return {}
    df = df.dropna(subset=["end_date"])
    df = df[df["end_date"].str.match(r"^\d{8}$", na=False)]

    for col in ("cash_div_tax", "stk_div", "base_share"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 可用日（事件因子公告可见性）：imp_ann_date 优先、缺失回退 ann_date
    if "imp_ann_date" not in df.columns:
        df["imp_ann_date"] = df["ann_date"]
    df = df.dropna(subset=["imp_ann_date"])
    df = df[df["imp_ann_date"].str.match(r"^\d{8}$", na=False)]

    df["_ex_num"] = _to_int_arr(df["ex_date"])
    df["_ex_dt"] = _to_dt_arr(df["ex_date"])
    df["_imp_num"] = _to_int_arr(df["imp_ann_date"])
    df["_year"] = df["end_date"].str[:4].astype(int)
    df = df.sort_values(["ts_code", "_ex_num"], kind="mergesort")

    income_q4_lookup = _build_income_q4_lookup(income_raw)
    list_date_map = list_date_map or {}

    # 回看窗口基准日历：优先 calendar_dates（含前序预热），否则 trading_dates
    calendar = list(calendar_dates) if calendar_dates else list(trading_dates)
    # 每个日历日的窗口左界（自然日减 365 天，避免 YYYYMMDD 整数相减跨月错误）
    left_ord_by_day = {
        d: int(
            (pd.to_datetime(d, format="%Y%m%d") - pd.Timedelta(days=_YIELD_WINDOW_DAYS)).strftime(
                "%Y%m%d"
            )
        )
        for d in calendar
    }
    # 每个输出日的近 10 交易日窗口起点（基于完整日历定位 T 的位置）
    cal_index = {d: idx for idx, d in enumerate(calendar)}
    win_start_by_day = {
        d: int(calendar[max(0, cal_index.get(d, 0) - (_RECENT_IMP_ANN_WINDOW - 1))])
        for d in trading_dates
    }

    # 年度段边界必须覆盖到 trading_dates 最后一年，否则 T 晚于最后事件年份时
    # 成熟财年边界（每年 0101/0901）缺失导致窗口错位
    last_trading_year = int(trading_dates[-1][:4])

    states: List[dict] = []
    for ts_code, grp in df.groupby("ts_code", sort=False):
        state = _build_stock_state(
            ts_code,
            grp,
            income_q4_lookup.get(ts_code),
            list_date_map.get(ts_code),
            last_trading_year,
        )
        if state is not None:
            states.append(state)

    if not states:
        logger.warning("分红政策查询表构建失败：无有效股票状态")
        return {}

    trade_dt = pd.to_datetime(trading_dates, format="%Y%m%d").to_numpy(dtype="datetime64[ns]")

    # ── 4. 按股票向量化日期、分块生成截面 ───────────────────
    lookup: Dict[str, pd.DataFrame] = {}
    trade_ord = np.asarray(trading_dates, dtype=np.int64)
    window_start_ord = np.asarray(
        [int(win_start_by_day[date_value]) for date_value in trading_dates],
        dtype=np.int64,
    )
    yield_left_ord = np.asarray(
        [int(left_ord_by_day[date_value]) for date_value in trading_dates],
        dtype=np.int64,
    )
    for chunk_start in range(0, len(trading_dates), _OUTPUT_DATE_CHUNK_SIZE):
        chunk_end = min(chunk_start + _OUTPUT_DATE_CHUNK_SIZE, len(trading_dates))
        stock_rows: List[Dict[str, np.ndarray]] = []
        for state in states:
            rows = _stock_rows_for_dates(
                state,
                trade_ord[chunk_start:chunk_end],
                trade_dt[chunk_start:chunk_end],
                window_start_ord[chunk_start:chunk_end],
                yield_left_ord[chunk_start:chunk_end],
            )
            if rows is not None:
                stock_rows.append(rows)
        if not stock_rows:
            continue
        chunk_rows = pd.DataFrame(
            {
                column: np.concatenate([rows[column] for rows in stock_rows])
                for column in stock_rows[0]
            }
        )
        for chunk_pos, day_rows in chunk_rows.groupby("_date_pos", sort=True):
            date_index = chunk_start + int(chunk_pos)
            lookup[trading_dates[date_index]] = day_rows.drop(columns="_date_pos").reset_index(
                drop=True
            )

    if lookup:
        total = sum(len(v) for v in lookup.values())
        logger.info(
            f"分红政策日频查询表构建完成: {len(lookup)} 个交易日有数据, "
            f"覆盖 {total} 条股票-日记录"
        )
    return lookup


# ── per-stock 状态与单日计算 ──────────────────────────────────


def _mature_year(t_ord: int) -> int:
    """T 日最新成熟财年（财年 y 的分红成熟于 (y+1) 年 9 月 1 日）。

    A股年报分红实施通常在次年 4-8 月完成，9 月 1 日前财年 y 尚未实施完毕
    （可能有晚到实施/取消），其后无实施记录即可确认该财年无分红。
    """
    y = t_ord // 10000
    mmdd = t_ord % 10000
    return y - 1 if mmdd >= 901 else y - 2


def _build_stock_state(
    ts_code: str,
    events: pd.DataFrame,
    income_q4: Optional[IncomeQ4Events],
    list_date: Optional[str],
    last_trading_year: int,
) -> Optional[dict]:
    """预计算单只股票的事件数组与年度窗口分段（送转 PIT 截断）。

    送转调整（当前股本口径，方向为前复权式）：
      T 日 1 股对应历史（送转前）的 G_{i-1}/G_T 股，其中
      G_k = Π_{j<=k}(1+stk_div_j)；历史每股分红按今天股本 = D_i × G_{i-1} / G_T。
      实现：base_i = D_i × G_{i-1}（与 T 无关），T 日再除以 G_T。
      比率类因子（连续性/稳定性/增长率）与共同 G_T 约掉可直接用 base。

    段边界 = 事件 ex 前缀点 ∪ 每年 0101/0901（保证段内可见事件集合与
    成熟财年 Y(T) 均恒定）；已成熟财年缺少实施记录时作为 0，尚未成熟
    财年仅在正分红 ex_date 落地后进入窗口。

    Returns:
        state dict（含 numpy 数组与分段预计算），无法构建时返回 None
    """
    ex_num = events["_ex_num"].to_numpy(dtype=np.int64)
    imp_num = events["_imp_num"].to_numpy(dtype=np.int64)
    cash_tax = events["cash_div_tax"].to_numpy(dtype=float)
    stk_div = np.maximum(events["stk_div"].to_numpy(dtype=float), 0.0)
    base_share = events["base_share"].to_numpy(dtype=float)
    years = events["_year"].to_numpy(dtype=np.int64)

    factors = 1.0 + stk_div
    prefix_g = np.concatenate([[1.0], np.cumprod(factors)])
    # base_i = D_i × G_{i-1}（当前股本口径的未归一化历史分红）
    base_adj = cash_tax * prefix_g[:-1]
    # 12 个月窗口 base 前缀和（与 T 无关的 base 部分，T 日再除以 G_T）
    pref_base = np.concatenate([[0.0], np.cumsum(base_adj)])
    cash_events = cash_tax * base_share * 10000.0

    # 上市年份（list_date 缺失时保守取最早分红年度，避免新股分母错配）
    list_year = None
    if list_date:
        norm = str(list_date).strip().replace("-", "")[:4]
        if norm.isdigit():
            list_year = int(norm)
    if list_year is None:
        list_year = int(years.min())

    # 段边界：事件前缀点 ∪ 每年 0101/0901（覆盖至最后交易年+1）
    last_event_year = int(ex_num[-1]) // 10000
    bounds_set = set(int(e) for e in ex_num)
    if income_q4 is not None:
        bounds_set.update(int(avail) for avail in income_q4[0])
    for y in range(list_year, max(last_event_year, last_trading_year) + 2):
        bounds_set.add(y * 10000 + 101)
        bounds_set.add(y * 10000 + 901)
    bounds = sorted(b for b in bounds_set if b > 0)

    # 逐段增量维护可见年度聚合（事件级，同财年多次分红按可见前缀逐步计入）
    seg_factors: List[dict] = []
    year_base: Dict[int, float] = {}
    year_cash: Dict[int, float] = {}
    ev_idx = 0
    for bound in bounds:
        while ev_idx < len(ex_num) and ex_num[ev_idx] <= bound:
            y = int(years[ev_idx])
            year_base[y] = year_base.get(y, 0.0) + float(base_adj[ev_idx])
            year_cash[y] = year_cash.get(y, 0.0) + float(cash_events[ev_idx])
            ev_idx += 1
        seg_factors.append(
            _annual_window_factors(
                list_year, _mature_year(bound), year_base, year_cash, income_q4, bound
            )
        )

    boundaries_arr = np.array(bounds, dtype=np.int64)
    seg_cols = {
        key: np.array([f[key] for f in seg_factors], dtype=float)
        for key in ("continuity", "stability", "growth_3y", "growth_5y", "payout")
    }

    # 公告数组（recent_imp_ann 计数用，按公告日排序）
    imp_sorted = np.sort(imp_num)

    return {
        "ts_code": ts_code,
        "ex_num": ex_num,
        "ex_dt": events["_ex_dt"].to_numpy(dtype="datetime64[ns]"),
        "imp_num": imp_num,
        "imp_sorted": imp_sorted,
        "pref_base": pref_base,
        "prefix_g": prefix_g,
        "list_year": list_year,
        "boundaries": boundaries_arr,
        "seg_cols": seg_cols,
    }


def _annual_window_factors(
    list_year: int,
    y_mature: int,
    year_base: Dict[int, float],
    year_cash: Dict[int, float],
    income_q4: Optional[IncomeQ4Events],
    bound: int,
) -> dict:
    """以成熟财年边界计算窗口因子值。

    窗口锚点取最新成熟财年与最新已实施正分红财年的较大值。已成熟年份
    无实施记录时作为 0；尚未成熟年份只有正分红已在 ex_date 落地才计入，
    避免 9 月前把未完成财年误判为停发，同时让已实施分红立即更新状态。
    """
    result = {
        "continuity": np.nan,
        "stability": np.nan,
        "growth_3y": np.nan,
        "growth_5y": np.nan,
        "payout": np.nan,
    }

    visible_positive_years = [year for year, value in year_base.items() if value > _EPS_CASH]
    anchor_year = max(
        y_mature,
        max(visible_positive_years) if visible_positive_years else y_mature,
    )
    year_floor = max(list_year, anchor_year - (_WINDOW_YEARS - 1))
    if anchor_year < year_floor:
        return result
    window_years = list(range(year_floor, anchor_year + 1))
    known_years = [
        year for year in window_years if year <= y_mature or year_base.get(year, 0.0) > _EPS_CASH
    ]
    if not known_years:
        return result
    denom = len(known_years)

    # 连续性：成熟停发年=0；未成熟且尚无正分红的年份不提前进入分母
    pos_count = sum(1 for year in known_years if year_base.get(year, 0.0) > _EPS_CASH)
    result["continuity"] = pos_count / denom

    # 稳定性：DPS 序列 MAD/中位数（≥3 年且中位数>0）
    dps_seq = np.array([year_base.get(year, 0.0) for year in known_years], dtype=float)
    if denom >= _MIN_STABILITY_YEARS:
        median = float(np.median(dps_seq))
        if median > _EPS_CASH:
            mad = float(np.median(np.abs(dps_seq - median)))
            result["stability"] = float(
                np.clip(1.0 - mad / median, _CLIP_STABILITY[0], _CLIP_STABILITY[1])
            )

    # 增长率：成熟年份可取确认后的 0；未成熟年份必须已有正分红落地
    def _known_dps(year: int) -> Optional[float]:
        if year < list_year:
            return None
        value = year_base.get(year, 0.0)
        if year <= y_mature or value > _EPS_CASH:
            return value
        return None

    dps_last = _known_dps(anchor_year)
    dps_3y = _known_dps(anchor_year - 3)
    dps_5y = _known_dps(anchor_year - 5)
    if dps_last is not None and dps_3y is not None:
        result["growth_3y"] = _bounded_symmetric_growth(dps_last, dps_3y)
    if dps_last is not None and dps_5y is not None:
        result["growth_5y"] = _bounded_symmetric_growth(dps_last, dps_5y)

    # 支付率：最新可见分红年度现金总额 / 该财年 Q4 归母净利润（PIT 双可见）
    visible_years = [y for y in year_cash.keys() if year_cash[y] > 0.0]
    if visible_years:
        latest_year = max(visible_years)
        cash_total = year_cash.get(latest_year, 0.0)
        if cash_total > _EPS_CASH and income_q4 is not None:
            avail_arr, years_arr, profit_arr = income_q4
            mask = (years_arr == latest_year) & (avail_arr <= bound)
            if mask.any():
                attributable_profit = float(profit_arr[mask][-1])
                if abs(attributable_profit) >= _MIN_ABS_NET_PROFIT:
                    result["payout"] = float(
                        np.clip(
                            cash_total / attributable_profit,
                            _CLIP_PAYOUT[0],
                            _CLIP_PAYOUT[1],
                        )
                    )

    return result


def _stock_rows_for_dates(
    state: dict,
    trade_ord: np.ndarray,
    trade_dt: np.ndarray,
    window_start_ord: np.ndarray,
    yield_left_ord: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """向量化计算单只股票的一段日期；整段无可见事件时返回 None。

    行输出条件：
      - 存在可见历史事件（最近 ex_date <= T）；或
      - 存在"已公告未除息"事件（imp_ann_date <= T 且 ex_date > T）——
        此时 days_to_ex_date / recent_imp_ann_10d 有信号，状态因子为 NaN。
    """
    ex_num = state["ex_num"]
    pos_ex = np.searchsorted(ex_num, trade_ord, side="right") - 1

    # 距最近已公告未除息事件天数：最多向后检查固定事件数，整段向量化。
    days_to_ex = np.full(len(trade_ord), np.nan, dtype=float)
    scan_start = pos_ex + 1
    imp_num = state["imp_num"]
    ex_dt = state["ex_dt"]
    has_announced_future = np.zeros(len(trade_ord), dtype=bool)
    resolved = np.zeros(len(trade_ord), dtype=bool)
    for offset in range(_MAX_SCAN_FUTURE_EX):
        candidate = scan_start + offset
        in_bounds = candidate < len(ex_num)
        safe_candidate = np.minimum(candidate, len(ex_num) - 1)
        announced = in_bounds & ~resolved & (imp_num[safe_candidate] <= trade_ord)
        if announced.any():
            delta = (ex_dt[safe_candidate] - trade_dt) / np.timedelta64(1, "D")
            in_window = announced & (delta <= _CLIP_DAYS_TO_EX)
            days_to_ex[in_window] = delta[in_window].astype(float)
            has_announced_future |= announced
            resolved |= announced

    active = (pos_ex >= 0) | has_announced_future
    if not active.any():
        return None

    # 年度状态因子：段定位（无历史事件时全 NaN）。
    seg_pos = np.searchsorted(state["boundaries"], trade_ord, side="right") - 1
    seg_cols = state["seg_cols"]
    state_visible = (pos_ex >= 0) & (seg_pos >= 0)

    def _segment_values(name: str) -> np.ndarray:
        values = np.full(len(trade_ord), np.nan, dtype=float)
        values[state_visible] = seg_cols[name][seg_pos[state_visible]]
        return values

    # 近 12 个月每股现金分红累计（T 日股本口径：base 窗口和 ÷ G(T)，PIT 截断）
    # 当前股本口径为前复权式：T 日 1 股对应历史 G_{i-1}/G_T 股，
    # 历史每股分红 = D_i × G_{i-1} / G_T，送转后历史每股分红缩小。
    pref_base = state["pref_base"]
    prefix_g = state["prefix_g"]
    cash_12m = np.zeros(len(trade_ord), dtype=float)
    history_visible = pos_ex >= 0
    right = pos_ex[history_visible] + 1
    left = np.searchsorted(ex_num, yield_left_ord[history_visible], side="left")
    g_t = prefix_g[pos_ex[history_visible] + 1]
    cash_12m[history_visible] = (pref_base[right] - pref_base[left]) / g_t

    # 近期实施公告计数（纯回看窗口 [T-9, T]，含 T）
    imp_sorted = state["imp_sorted"]
    recent_count = np.searchsorted(imp_sorted, trade_ord, side="right") - np.searchsorted(
        imp_sorted, window_start_ord, side="left"
    )

    # freshness：距最近 ex_date 自然日差（无历史事件 → NaN）
    freshness = np.full(len(trade_ord), np.nan, dtype=float)
    freshness[history_visible] = (
        trade_dt[history_visible] - ex_dt[pos_ex[history_visible]]
    ) / np.timedelta64(1, "D")
    hist_missing = np.where(history_visible, 0.0, np.nan)

    active_pos = np.flatnonzero(active)
    return {
        "_date_pos": active_pos,
        "ts_code": np.full(len(active_pos), state["ts_code"], dtype=object),
        "dividend_continuity_5y": _segment_values("continuity")[active],
        "dividend_stability_5y": _segment_values("stability")[active],
        "dividend_growth_3y": _segment_values("growth_3y")[active],
        "dividend_growth_5y": _segment_values("growth_5y")[active],
        "dividend_payout_ratio": _segment_values("payout")[active],
        _CASH_12M_ADJ_COL: cash_12m[active],
        "dividend_days_to_ex_date": days_to_ex[active],
        "dividend_recent_imp_ann_10d": recent_count[active].astype(float),
        DIVIDEND_FRESHNESS_COL: freshness[active],
        DIVIDEND_HIST_MISSING_COL: hist_missing[active],
    }
