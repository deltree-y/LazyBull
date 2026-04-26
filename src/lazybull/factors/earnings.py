"""业绩预告因子模块

将事件驱动的业绩预告（forecast）数据
按 ann_date point-in-time 对齐到日频。

数据来源：
- Tushare forecast_vip API（5000 积分）：业绩预告

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
]


def build_earnings_lookup_by_date(
    forecast_df: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """将业绩预告数据按 ann_date point-in-time 对齐到日频

    Args:
        forecast_df: 业绩预告 DataFrame，需包含
            ts_code, ann_date, end_date, type, p_change_min, p_change_max
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

        if len(trading_dates) == 1:
            trade_date = trading_dates[0]
            visible = fc[fc["ann_date"] <= trade_date].sort_values(["ts_code", "ann_date"])
            if visible.empty:
                result = {trade_date: pd.DataFrame(columns=["ts_code"] + EARNINGS_COLS)}
            else:
                day_df = (
                    visible.drop_duplicates(subset=["ts_code"], keep="last")
                    [["ts_code"] + EARNINGS_COLS]
                    .reset_index(drop=True)
                )
                result = {trade_date: day_df}
            logger.info(f"业绩预告查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
            return result

        stock_ann_dates: Dict[str, list] = {}
        stock_values: Dict[str, list] = {}
        for ts_code, grp in fc.groupby("ts_code"):
            grp = grp.sort_values("ann_date")
            stock_ann_dates[ts_code] = grp["ann_date"].tolist()
            stock_values[ts_code] = grp[EARNINGS_COLS].values.tolist()

        fc_records = {
            ts_code: (stock_ann_dates[ts_code], stock_values[ts_code])
            for ts_code in stock_ann_dates
        }

    # ── 对每个交易日做 point-in-time 查询 ───────────────────────
    all_codes = set(fc_records.keys())
    result: Dict[str, pd.DataFrame] = {}

    for trade_date in trading_dates:
        rows = []
        for ts_code in all_codes:
            row = {"ts_code": ts_code}
            has_data = False

            # 预告查询
            if ts_code in fc_records:
                ann_dates, values = fc_records[ts_code]
                idx = bisect.bisect_right(ann_dates, trade_date) - 1
                if idx >= 0:
                    row["forecast_type_score"] = values[idx][0]
                    row["forecast_chg_mid"] = values[idx][1]
                    has_data = True

            if has_data:
                rows.append(row)

        if rows:
            result[trade_date] = pd.DataFrame(rows)

    logger.info(f"业绩预告查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result
