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
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger


def _compute_revenue_yoy(ex: pd.DataFrame) -> None:
    """在 DataFrame 上原地计算营收同比增速（%）

    逻辑：同一 ts_code，当前 end_date 对比去年同期（年份-1、月日相同）的 revenue。
    公式：(当期revenue - 去年同期revenue) / abs(去年同期revenue) * 100
    """
    if "revenue" not in ex.columns or "end_date" not in ex.columns:
        ex["revenue_yoy"] = np.nan
        return

    # 构建 (ts_code, end_date) -> revenue 的映射
    rev_map: Dict[tuple, float] = {}
    for _, row in ex.iterrows():
        rev_map[(row["ts_code"], row["end_date"])] = row["revenue"]

    yoy_values = []
    for _, row in ex.iterrows():
        end_date = str(row["end_date"])
        if len(end_date) == 8:
            # 去年同期：年份 -1，月日不变
            prev_end = str(int(end_date[:4]) - 1) + end_date[4:]
            prev_rev = rev_map.get((row["ts_code"], prev_end))
            cur_rev = row["revenue"]
            if prev_rev is not None and pd.notna(prev_rev) and abs(prev_rev) > 1e-6 and pd.notna(cur_rev):
                yoy_values.append((cur_rev - prev_rev) / abs(prev_rev) * 100)
            else:
                yoy_values.append(np.nan)
        else:
            yoy_values.append(np.nan)
    ex["revenue_yoy"] = yoy_values


EXPRESS_COLS = [
    "express_revenue_yoy",    # 营业收入同比增速
    "express_profit_yoy",     # 净利润同比增速
    "express_roe",            # 快报加权ROE
    "express_surprise",       # 业绩惊喜（净利润增速 vs 上期预告偏差）
]


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

    # 日期标准化
    for col in ["ann_date", "end_date"]:
        if col in ex.columns:
            ex[col] = ex[col].astype(str).str.replace("-", "").str[:8]

    ex = ex.dropna(subset=["ann_date"])

    # 数值列转换（TuShare express_vip 实际列名）
    for col in ["revenue", "yoy_net_profit", "diluted_roe"]:
        if col in ex.columns:
            ex[col] = pd.to_numeric(ex[col], errors="coerce")

    # 去重：同股同报告期保留最新公告
    ex = ex.sort_values(["ts_code", "end_date", "ann_date"])
    ex = ex.drop_duplicates(subset=["ts_code", "end_date"], keep="last")
    ex = ex.sort_values(["ts_code", "ann_date"])

    # 计算营收同比增速（TuShare express_vip 不提供此字段，需自行计算）
    # 同一公司，当前 end_date 对比去年同期 end_date 的 revenue
    _compute_revenue_yoy(ex)

    # 构建预告 lookup（用于计算 express_surprise）
    fc_lookup: Dict[str, Dict[str, float]] = {}  # ts_code -> {end_date -> forecast_chg_mid}
    if forecast_df is not None and len(forecast_df) > 0:
        fc = forecast_df.copy()
        for col in ["ann_date", "end_date"]:
            if col in fc.columns:
                fc[col] = fc[col].astype(str).str.replace("-", "").str[:8]
        for col in ["p_change_min", "p_change_max"]:
            if col in fc.columns:
                fc[col] = pd.to_numeric(fc[col], errors="coerce")
        if "p_change_min" in fc.columns and "p_change_max" in fc.columns:
            fc["forecast_chg_mid"] = (fc["p_change_min"] + fc["p_change_max"]) / 2
            fc = fc.dropna(subset=["forecast_chg_mid"])
            fc = fc.sort_values(["ts_code", "end_date", "ann_date"])
            fc = fc.drop_duplicates(subset=["ts_code", "end_date"], keep="last")
            for _, row in fc.iterrows():
                fc_lookup.setdefault(row["ts_code"], {})[row["end_date"]] = row["forecast_chg_mid"]

    # 构建每只股票的 ann_date 排序列表
    stock_records: Dict[str, list] = {}
    for ts_code, grp in ex.groupby("ts_code"):
        records = []
        for _, row in grp.iterrows():
            revenue_yoy = row.get("revenue_yoy", np.nan)
            profit_yoy = row.get("yoy_net_profit", np.nan)
            roe = row.get("diluted_roe", np.nan)

            # express_surprise = 实际净利润增速 - 预告预期增速
            surprise = np.nan
            end_date = row.get("end_date", "")
            if pd.notna(profit_yoy) and ts_code in fc_lookup and end_date in fc_lookup[ts_code]:
                surprise = profit_yoy - fc_lookup[ts_code][end_date]

            records.append({
                "ann_date": row["ann_date"],
                "express_revenue_yoy": revenue_yoy,
                "express_profit_yoy": profit_yoy,
                "express_roe": roe,
                "express_surprise": surprise,
            })
        stock_records[ts_code] = records

    # 对每个交易日做 point-in-time 查询
    result: Dict[str, pd.DataFrame] = {}
    all_codes = set(stock_records.keys())

    for trade_date in trading_dates:
        rows = []
        for ts_code in all_codes:
            recs = stock_records[ts_code]
            ann_dates = [r["ann_date"] for r in recs]
            idx = bisect.bisect_right(ann_dates, trade_date) - 1
            if idx >= 0:
                r = recs[idx]
                rows.append({
                    "ts_code": ts_code,
                    "express_revenue_yoy": r["express_revenue_yoy"],
                    "express_profit_yoy": r["express_profit_yoy"],
                    "express_roe": r["express_roe"],
                    "express_surprise": r["express_surprise"],
                })
        if rows:
            result[trade_date] = pd.DataFrame(rows)

    logger.info(f"业绩快报查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result
