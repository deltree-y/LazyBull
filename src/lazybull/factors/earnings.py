"""业绩预告/快报因子模块

将事件驱动的业绩预告（forecast）和业绩快报（express）数据
按 ann_date point-in-time 对齐到日频。

数据来源：
- Tushare forecast API（2000 积分）：业绩预告
- Tushare express API（2000 积分）：业绩快报

关键防前视：只用 ann_date（公告日），不用 end_date（报告期末）。
公告后前向填充，直到被同一报告期的新公告或下一报告期覆盖。
"""

import bisect
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger


# 业绩预告类型 → 评分
FORECAST_TYPE_SCORE = {
    "预增": 1.0,
    "略增": 0.5,
    "续盈": 0.3,
    "扭亏": 0.8,
    "不确定": 0.0,
    "略减": -0.5,
    "预减": -1.0,
    "首亏": -1.0,
    "续亏": -0.8,
}

EARNINGS_COLS = [
    "forecast_type_score",
    "forecast_chg_mid",
    "express_profit_yoy",
    "express_revenue_yoy",
]


def build_earnings_lookup_by_date(
    forecast_df: pd.DataFrame,
    express_df: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """将业绩预告和快报数据按 ann_date point-in-time 对齐到日频

    Args:
        forecast_df: 业绩预告 DataFrame，需包含
            ts_code, ann_date, end_date, type, p_change_min, p_change_max
        express_df: 业绩快报 DataFrame，需包含
            ts_code, ann_date, end_date, yoy_net_profit, yoy_sales
        trading_dates: 交易日列表（YYYYMMDD 字符串，已排序）

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, forecast_type_score, ...)}
    """
    # ── 处理业绩预告 ────────────────────────────────────────────
    fc_records: Dict[str, list] = {}  # ts_code -> [{ann_date, score, chg_mid}]

    if forecast_df is not None and len(forecast_df) > 0:
        fc = forecast_df.copy()
        fc["ann_date"] = fc["ann_date"].astype(str).str.replace("-", "").str[:8]
        fc = fc.dropna(subset=["ann_date"])

        # 类型评分
        if "type" in fc.columns:
            fc["forecast_type_score"] = fc["type"].map(FORECAST_TYPE_SCORE).fillna(0.0)
        else:
            fc["forecast_type_score"] = 0.0

        # 变动幅度中值
        for col in ["p_change_min", "p_change_max"]:
            if col in fc.columns:
                fc[col] = pd.to_numeric(fc[col], errors="coerce")
        if "p_change_min" in fc.columns and "p_change_max" in fc.columns:
            fc["forecast_chg_mid"] = (fc["p_change_min"] + fc["p_change_max"]) / 2
        else:
            fc["forecast_chg_mid"] = np.nan

        # 去重：同股同报告期保留最新公告
        fc = fc.sort_values(["ts_code", "end_date", "ann_date"])
        fc = fc.drop_duplicates(subset=["ts_code", "end_date"], keep="last")
        fc = fc.sort_values(["ts_code", "ann_date"])

        for ts_code, grp in fc.groupby("ts_code"):
            fc_records[ts_code] = [
                {
                    "ann_date": row["ann_date"],
                    "forecast_type_score": row["forecast_type_score"],
                    "forecast_chg_mid": row["forecast_chg_mid"],
                }
                for _, row in grp.iterrows()
            ]

    # ── 处理业绩快报 ────────────────────────────────────────────
    ex_records: Dict[str, list] = {}

    if express_df is not None and len(express_df) > 0:
        ex = express_df.copy()
        ex["ann_date"] = ex["ann_date"].astype(str).str.replace("-", "").str[:8]
        ex = ex.dropna(subset=["ann_date"])

        for col in ["yoy_net_profit", "yoy_sales"]:
            if col in ex.columns:
                ex[col] = pd.to_numeric(ex[col], errors="coerce")

        # 去重
        if "end_date" in ex.columns:
            ex["end_date"] = ex["end_date"].astype(str).str.replace("-", "").str[:8]
            ex = ex.sort_values(["ts_code", "end_date", "ann_date"])
            ex = ex.drop_duplicates(subset=["ts_code", "end_date"], keep="last")
        ex = ex.sort_values(["ts_code", "ann_date"])

        for ts_code, grp in ex.groupby("ts_code"):
            ex_records[ts_code] = [
                {
                    "ann_date": row["ann_date"],
                    "express_profit_yoy": row.get("yoy_net_profit", np.nan),
                    "express_revenue_yoy": row.get("yoy_sales", np.nan),
                }
                for _, row in grp.iterrows()
            ]

    # ── 对每个交易日做 point-in-time 查询 ───────────────────────
    all_codes = set(list(fc_records.keys()) + list(ex_records.keys()))
    result: Dict[str, pd.DataFrame] = {}

    for trade_date in trading_dates:
        rows = []
        for ts_code in all_codes:
            row = {"ts_code": ts_code}
            has_data = False

            # 预告查询
            if ts_code in fc_records:
                recs = fc_records[ts_code]
                ann_dates = [r["ann_date"] for r in recs]
                idx = bisect.bisect_right(ann_dates, trade_date) - 1
                if idx >= 0:
                    r = recs[idx]
                    row["forecast_type_score"] = r["forecast_type_score"]
                    row["forecast_chg_mid"] = r["forecast_chg_mid"]
                    has_data = True

            # 快报查询
            if ts_code in ex_records:
                recs = ex_records[ts_code]
                ann_dates = [r["ann_date"] for r in recs]
                idx = bisect.bisect_right(ann_dates, trade_date) - 1
                if idx >= 0:
                    r = recs[idx]
                    row["express_profit_yoy"] = r["express_profit_yoy"]
                    row["express_revenue_yoy"] = r["express_revenue_yoy"]
                    has_data = True

            if has_data:
                rows.append(row)

        if rows:
            result[trade_date] = pd.DataFrame(rows)

    logger.info(f"业绩预告/快报查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result
