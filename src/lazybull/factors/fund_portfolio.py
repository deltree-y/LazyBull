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

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ..common.date_utils import normalize_series_to_yyyymmdd
from .announcement_utils import build_latest_announcement_lookup_by_date

FUND_PORTFOLIO_COLS = [
    "fund_hold_ratio",  # 基金持股占流通股比例（全基金汇总）
    "fund_hold_ratio_chg",  # 基金持股比例季度环比变化
    "fund_count",  # 持仓基金数量
    "fund_count_chg",  # 持仓基金数量季度变化
]

FUND_PORTFOLIO_FRESHNESS_COL = "fund_portfolio_freshness_days"

FUND_PORTFOLIO_RAW_COLS = [
    "ts_code",
    "symbol",
    "ann_date",
    "end_date",
    "stk_float_ratio",
]

_QUARTER_ENDS = ["0331", "0630", "0930", "1231"]


def _prev_quarter_end(end_date) -> Optional[str]:
    """返回紧邻上一季度末（YYYYMMDD），非法或非季度末返回 None。"""
    text = str(end_date)
    if len(text) != 8 or not text.isdigit():
        return None
    year, month_day = text[:4], text[4:]
    if month_day not in _QUARTER_ENDS:
        return None
    idx = _QUARTER_ENDS.index(month_day)
    if idx == 0:
        return f"{int(year) - 1}1231"
    return f"{year}{_QUARTER_ENDS[idx - 1]}"


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
    available_cols = [c for c in FUND_PORTFOLIO_RAW_COLS if c in raw_df.columns]
    missing_cols = set(FUND_PORTFOLIO_RAW_COLS) - set(available_cols)
    if missing_cols:
        logger.warning(f"fund_portfolio 聚合缺少必要列: {sorted(missing_cols)}")
        return pd.DataFrame(
            columns=["symbol", "end_date", "fund_hold_ratio", "fund_count", "ann_date"]
        )

    # 只保留聚合所需列，避免季度原始明细在内存中保留无关宽列。
    df = raw_df.loc[:, available_cols].copy()

    # 日期标准化（兼容 datetime 和字符串类型）
    for col in ["ann_date", "end_date"]:
        if col in df.columns:
            df[col] = normalize_series_to_yyyymmdd(df[col])

    # 数值列转换
    df["stk_float_ratio"] = pd.to_numeric(df["stk_float_ratio"], errors="coerce")

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
    agg["prev_ann_date"] = grouped["ann_date"].shift(1)
    agg["prev_end_date"] = grouped["end_date"].shift(1)
    agg["fund_hold_ratio_chg"] = agg["fund_hold_ratio"] - agg["fund_hold_ratio_prev"]
    agg["fund_count_chg"] = agg["fund_count"] - agg["fund_count_prev"]

    # 上期延迟披露（公告日晚于本期）会造成前视；上期非紧邻季度则环比口径不可比。
    prev_ann = agg["prev_ann_date"].fillna("")
    cur_ann = agg["ann_date"].fillna("")
    prev_end = agg["prev_end_date"].fillna("")
    expected_prev_end = agg["end_date"].map(_prev_quarter_end).fillna("")
    invalid_chg = (prev_ann > cur_ann) | (prev_end != expected_prev_end)
    agg.loc[invalid_chg, ["fund_hold_ratio_chg", "fund_count_chg"]] = np.nan

    agg["ts_code"] = agg["symbol"].map(_symbol_to_ts_code)
    agg = agg.dropna(subset=["ts_code", "ann_date"])
    factor_df = agg[["ts_code", "ann_date"] + FUND_PORTFOLIO_COLS].copy()
    return build_latest_announcement_lookup_by_date(
        factor_df,
        trading_dates,
        value_cols=FUND_PORTFOLIO_COLS,
        freshness_col=FUND_PORTFOLIO_FRESHNESS_COL,
        log_name="基金持仓",
    )


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
