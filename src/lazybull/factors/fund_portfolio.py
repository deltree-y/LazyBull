"""基金持仓因子模块

将公募基金持仓数据（fund_portfolio）按个股聚合后
按 ann_date point-in-time 对齐到日频。

数据来源：Tushare fund_portfolio API（5000 积分）
更新频率：季频（随基金季报/半年报/年报披露）
核心逻辑：
- 汇总所有公募基金对同一只股票的持仓，得到机构共识
- 基金持股比例上升 + 持仓基金数量增加 → 机构看好
- fund_count_chg 捕捉"拥挤度"变化（人多→可能过热，人少→可能冷门价值）
"""

import bisect
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

FUND_PORTFOLIO_COLS = [
    "fund_hold_ratio",  # 基金持股占流通股比例（全基金汇总）
    "fund_hold_ratio_chg",  # 基金持股比例季度环比变化
    "fund_count",  # 持仓基金数量
    "fund_count_chg",  # 持仓基金数量季度变化
]


def _aggregate_fund_portfolio(raw_df: pd.DataFrame) -> pd.DataFrame:
    """将基金级持仓数据聚合到个股级

    Args:
        raw_df: fund_portfolio 原始数据，包含
            ts_code(基金代码), symbol(股票代码), ann_date, end_date,
            stk_mkv_ratio, stk_float_ratio, mkv, amount

    Returns:
        个股级汇总 DataFrame(stock_code, ann_date, end_date,
                            fund_hold_ratio, fund_count)
    """
    # 直接操作传入的 DataFrame（调用方保证传入独立副本或不再使用原始引用）
    df = raw_df

    # 日期标准化（兼容 datetime 和字符串类型）
    for col in ["ann_date", "end_date"]:
        if col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y%m%d")
            else:
                df[col] = df[col].astype(str).str.replace("-", "").str[:8]

    # 数值列转换
    for col in ["stk_float_ratio", "mkv", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["symbol", "end_date"])

    # 按个股+季度聚合
    agg_df = (
        df.groupby(["symbol", "end_date"])
        .agg(
            fund_hold_ratio=("stk_float_ratio", "sum"),  # 合计占流通股比例
            fund_count=("ts_code", "nunique"),  # 持仓基金数量
            ann_date=("ann_date", "max"),  # 取最晚公告日作为信息可用时点
        )
        .reset_index()
    )

    return agg_df


def build_fund_portfolio_lookup_by_date(
    fund_portfolio_df: pd.DataFrame,
    trading_dates: List[str],
    pre_aggregated: bool = False,
) -> Dict[str, pd.DataFrame]:
    """将基金持仓数据按 ann_date point-in-time 对齐到日频

    Args:
        fund_portfolio_df: fund_portfolio DataFrame（原始或已聚合）
        trading_dates: 交易日列表（YYYYMMDD 字符串，已排序）
        pre_aggregated: 若为 True，表示传入的已是个股级聚合数据，跳过聚合步骤

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, fund_hold_ratio, ...)}
    """
    if fund_portfolio_df is None or len(fund_portfolio_df) == 0:
        return {}

    # 聚合到个股级（若已预聚合则跳过，节省内存）
    agg = fund_portfolio_df if pre_aggregated else _aggregate_fund_portfolio(fund_portfolio_df)
    if len(agg) == 0:
        return {}

    # 按股票+报告期排序，计算季度变化
    agg = agg.sort_values(["symbol", "end_date"])
    grouped = agg.groupby("symbol")
    agg["fund_hold_ratio_prev"] = grouped["fund_hold_ratio"].shift(1)
    agg["fund_count_prev"] = grouped["fund_count"].shift(1)
    agg["fund_hold_ratio_chg"] = agg["fund_hold_ratio"] - agg["fund_hold_ratio_prev"]
    agg["fund_count_chg"] = agg["fund_count"] - agg["fund_count_prev"]

    # 构建每只股票的 ann_date 排序列表，用于 bisect
    stock_records: Dict[str, list] = {}
    for symbol, grp in agg.groupby("symbol"):
        # symbol 是6位数字，需要转换为 ts_code 格式
        ts_code = _symbol_to_ts_code(symbol)
        if ts_code is None:
            continue
        records = []
        for _, row in grp.sort_values("ann_date").iterrows():
            records.append(
                {
                    "ann_date": row["ann_date"],
                    "fund_hold_ratio": row["fund_hold_ratio"],
                    "fund_hold_ratio_chg": row["fund_hold_ratio_chg"],
                    "fund_count": row["fund_count"],
                    "fund_count_chg": row["fund_count_chg"],
                }
            )
        stock_records[ts_code] = records

    # 对每个交易日做 point-in-time 查询
    result: Dict[str, pd.DataFrame] = {}

    for trade_date in trading_dates:
        rows = []
        for ts_code, records in stock_records.items():
            ann_dates = [r["ann_date"] for r in records]
            idx = bisect.bisect_right(ann_dates, trade_date) - 1
            if idx >= 0:
                r = records[idx]
                rows.append(
                    {
                        "ts_code": ts_code,
                        "fund_hold_ratio": r["fund_hold_ratio"],
                        "fund_hold_ratio_chg": r["fund_hold_ratio_chg"],
                        "fund_count": r["fund_count"],
                        "fund_count_chg": r["fund_count_chg"],
                    }
                )
        if rows:
            result[trade_date] = pd.DataFrame(rows)

    logger.info(f"基金持仓查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result


def _symbol_to_ts_code(symbol: str) -> str:
    """将股票代码转换为 ts_code 格式

    Args:
        symbol: 股票代码，可以是纯6位数字（如 '000001'）
                或已含交易所后缀（如 '000001.SZ'）

    Returns:
        ts_code 格式，如 '000001.SZ'，无法识别返回 None
    """
    if symbol is None or (isinstance(symbol, float) and pd.isna(symbol)):
        return None
    s = str(symbol).strip()
    if not s or s == "nan":
        return None
    # 如果已含交易所后缀（如 TuShare fund_portfolio 返回的 symbol），直接返回
    if "." in s and s.split(".")[-1] in ("SH", "SZ", "BJ"):
        return s
    s = s.zfill(6)
    if s.startswith(("6", "9")):
        return f"{s}.SH"
    elif s.startswith(("0", "2", "3")):
        return f"{s}.SZ"
    elif s.startswith(("4", "8")):
        return f"{s}.BJ"
    return None
