"""业绩快报因子模块

将事件驱动的业绩快报（express）数据
按 ann_date point-in-time 对齐到日频。

数据来源：Tushare express_vip API（5000 积分）
更新频率：事件驱动（季报前后密集发布）

与 forecast（业绩预告）的区别：
- forecast 是预告（预计范围），express 是快报（实际数据）
- express 发布时间更晚但数据更准确
- 两者可共存：forecast 先到给出预期方向，express 后到确认实际情况

关键防前视：只用 ann_date（公告日），不用 end_date（报告期末）。
"""

import bisect
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..common.date_utils import normalize_series_to_yyyymmdd
from .announcement_utils import build_latest_announcement_lookup_by_date


def _compute_revenue_yoy(ex: pd.DataFrame) -> None:
    """在 DataFrame 上原地计算营收同比增速（%）

    逻辑：同一 ts_code，当前 end_date 对比去年同期（年份-1、月日相同）的 revenue。
    公式：(当期revenue - 去年同期revenue) / abs(去年同期revenue) * 100

    去年同期只取本期公告日当天及之前已披露的版本，避免重述公告造成跨期前视。
    """
    required = {"revenue", "end_date", "ann_date"}
    if not required.issubset(ex.columns):
        ex["revenue_yoy"] = np.nan
        return

    # 构建 (ts_code, end_date) -> (ann_date 升序列表, revenue 列表)
    rev_lookup: Dict[Tuple[str, str], Tuple[List[str], List[float]]] = {}
    ordered = ex[["ts_code", "end_date", "ann_date", "revenue"]].sort_values(
        ["ts_code", "end_date", "ann_date"]
    )
    for ts_code, end_date, ann_date, revenue in zip(
        ordered["ts_code"], ordered["end_date"], ordered["ann_date"], ordered["revenue"]
    ):
        ann_dates, revenues = rev_lookup.setdefault((ts_code, str(end_date)), ([], []))
        ann_dates.append(str(ann_date))
        revenues.append(revenue)

    yoy_values = []
    for ts_code, end_date, ann_date, cur_rev in zip(
        ex["ts_code"], ex["end_date"], ex["ann_date"], ex["revenue"]
    ):
        end_date = str(end_date)
        if len(end_date) != 8 or not end_date.isdigit() or pd.isna(cur_rev):
            yoy_values.append(np.nan)
            continue

        # 去年同期：年份 -1，月日不变
        prev_end = str(int(end_date[:4]) - 1) + end_date[4:]
        hist = rev_lookup.get((ts_code, prev_end))
        if hist is None:
            yoy_values.append(np.nan)
            continue

        prev_ann_dates, prev_revenues = hist
        pos = bisect.bisect_right(prev_ann_dates, str(ann_date)) - 1
        if pos < 0:
            yoy_values.append(np.nan)
            continue

        prev_rev = prev_revenues[pos]
        if pd.notna(prev_rev) and abs(prev_rev) > 1e-6:
            yoy_values.append((cur_rev - prev_rev) / abs(prev_rev) * 100)
        else:
            yoy_values.append(np.nan)
    ex["revenue_yoy"] = yoy_values


EXPRESS_COLS = [
    "express_revenue_yoy",  # 营业收入同比增速
    "express_profit_yoy",  # 净利润同比增速
    "express_roe",  # 快报加权ROE
    "express_surprise",  # 业绩惊喜（净利润增速 vs 上期预告偏差）
]

EXPRESS_FRESHNESS_COL = "express_freshness_days"


def build_express_lookup_by_date(
    express_df: pd.DataFrame,
    trading_dates: List[str],
    forecast_df: pd.DataFrame = None,
) -> Dict[str, pd.DataFrame]:
    """将业绩快报数据按 ann_date point-in-time 对齐到日频

    Args:
        express_df: 业绩快报 DataFrame，需包含
            ts_code, ann_date, end_date, revenue, yoy_net_profit, diluted_roe
        trading_dates: 交易日列表（YYYYMMDD 字符串，已排序）
        forecast_df: 业绩预告 DataFrame（可选），用于计算 express_surprise

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, express_revenue_yoy, ...)}
    """
    if express_df is None or len(express_df) == 0:
        return {}

    ex = express_df.copy()

    # 日期标准化（兼容 datetime 和字符串类型）
    for col in ["ann_date", "end_date"]:
        if col in ex.columns:
            ex[col] = normalize_series_to_yyyymmdd(ex[col])

    ex = ex.dropna(subset=["ann_date"])

    # 数值列转换（TuShare express_vip 实际列名）
    for col in ["revenue", "yoy_net_profit", "diluted_roe"]:
        if col in ex.columns:
            ex[col] = pd.to_numeric(ex[col], errors="coerce")

    # 仅去除完全重复记录，保留同一报告期多次公告版本，交由 PIT 查询按交易日选择
    ex = ex.sort_values(["ts_code", "end_date", "ann_date"])
    ex = ex.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")
    ex = ex.sort_values(["ts_code", "ann_date"])

    # 计算营收同比增速（TuShare express_vip 不提供此字段，需自行计算）
    # 同一公司，当前 end_date 对比去年同期 end_date 的 revenue
    _compute_revenue_yoy(ex)

    # 构建预告 lookup（用于计算 express_surprise）
    fc_lookup: Dict[Tuple[str, str], Tuple[List[str], List[float]]] = {}
    if forecast_df is not None and len(forecast_df) > 0:
        fc = forecast_df.copy()
        for col in ["ann_date", "end_date"]:
            if col in fc.columns:
                fc[col] = normalize_series_to_yyyymmdd(fc[col])
        for col in ["p_change_min", "p_change_max"]:
            if col in fc.columns:
                fc[col] = pd.to_numeric(fc[col], errors="coerce")
        if "p_change_min" in fc.columns and "p_change_max" in fc.columns:
            fc["forecast_chg_mid"] = (fc["p_change_min"] + fc["p_change_max"]) / 2
            fc = fc.dropna(subset=["forecast_chg_mid"])
            fc = fc.sort_values(["ts_code", "end_date", "ann_date"])
            fc = fc.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")
            for (ts_code, end_date), grp in fc.groupby(["ts_code", "end_date"], sort=False):
                grp = grp.sort_values("ann_date")
                fc_lookup[(ts_code, end_date)] = (
                    grp["ann_date"].tolist(),
                    grp["forecast_chg_mid"].astype(float).tolist(),
                )

    ex["express_surprise"] = np.nan
    for idx, row in ex.iterrows():
        key = (row["ts_code"], row["end_date"])
        hist = fc_lookup.get(key)
        if hist is None:
            continue
        ann_dates, values = hist
        pos = bisect.bisect_right(ann_dates, row["ann_date"]) - 1
        if pos < 0:
            continue
        base_forecast = values[pos]
        profit_yoy = row.get("yoy_net_profit")
        if pd.notna(base_forecast) and pd.notna(profit_yoy):
            ex.at[idx, "express_surprise"] = float(profit_yoy) - float(base_forecast)

    factor_df = ex.assign(
        express_revenue_yoy=ex["revenue_yoy"],
        express_profit_yoy=ex["yoy_net_profit"],
        express_roe=ex["diluted_roe"],
    )[["ts_code", "ann_date"] + EXPRESS_COLS]

    return build_latest_announcement_lookup_by_date(
        factor_df,
        trading_dates,
        value_cols=EXPRESS_COLS,
        freshness_col=EXPRESS_FRESHNESS_COL,
        log_name="业绩快报",
    )
