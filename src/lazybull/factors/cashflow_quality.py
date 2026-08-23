"""现金流量表质量因子模块

将季度频率的现金流量表数据（cashflow）前向填充到日频，
构建每日现金流质量查询表，供特征构建使用。

核心逻辑：
- 使用 ann_date（公告日期）作为数据可用时间点，防止前视偏差
- 对每个交易日，找到每只股票最近一次已公告的现金流量表数据
- 构建经营现金流稳定性、自由现金流收益率等质量因子
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from .announcement_utils import build_latest_announcement_lookup_by_date

# 现金流因子输出列（前向填充后的日频列名）
CASHFLOW_COLS = [
    "ocf",                  # 经营活动现金流净额
    "ocf_to_revenue",       # OCF / 营业收入（现金含量）
    "ocf_to_profit",        # OCF / 净利润（利润质量）
    "fcf",                  # 自由现金流
    "fcf_yield",            # FCF / 总市值（现金回报率）
    "capex_to_ocf",         # 资本支出 / OCF
]

CASHFLOW_FRESHNESS_COL = "cashflow_freshness_days"


def build_cashflow_quality_lookup_by_date(
    cashflow_raw: pd.DataFrame,
    trading_dates: List[str],
    daily_basic_lookup: Dict[str, pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """将季度现金流量表前向填充到日频，构建每日查询表

    Args:
        cashflow_raw: 现金流量表原始 DataFrame，需包含
                      ts_code, ann_date, end_date,
                      n_cashflow_act, c_pay_acq_const_fiolta, c_fr_sale_sg, net_profit
        trading_dates: 交易日列表（YYYYMMDD 格式字符串，已排序）
        daily_basic_lookup: 每日指标查询表 {trade_date: DataFrame}，
                           用于获取 total_mv 计算 fcf_yield

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, ocf, ocf_to_revenue, ...)}
    """
    if cashflow_raw is None or len(cashflow_raw) == 0:
        logger.warning("现金流量表数据为空，跳过现金流因子构建")
        return {}

    df = cashflow_raw.copy()

    # 清洗：去掉 ann_date 缺失的记录
    df = df.dropna(subset=["ann_date"])
    df["ann_date"] = df["ann_date"].astype(str).str[:8]
    df["end_date"] = df["end_date"].astype(str).str[:8]

    # 仅去除完全重复记录，保留同一报告期的多次公告版本，交由 PIT 查询按交易日选择
    df = df.sort_values(["ts_code", "end_date", "ann_date"])
    df = df.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")

    # 数值化
    numeric_cols = [
        "n_cashflow_act", "c_pay_acq_const_fiolta",
        "c_fr_sale_sg", "net_profit",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 计算派生因子
    if "n_cashflow_act" in df.columns:
        df["ocf"] = df["n_cashflow_act"]
    else:
        df["ocf"] = np.nan

    if "c_fr_sale_sg" in df.columns and df["c_fr_sale_sg"].notna().any():
        df["ocf_to_revenue"] = np.where(
            df["c_fr_sale_sg"].abs() > 1e-6,
            df["ocf"] / df["c_fr_sale_sg"],
            np.nan,
        )
    else:
        df["ocf_to_revenue"] = np.nan

    if "net_profit" in df.columns and df["net_profit"].notna().any():
        df["ocf_to_profit"] = np.where(
            df["net_profit"].abs() > 1e-6,
            df["ocf"] / df["net_profit"],
            np.nan,
        )
    else:
        df["ocf_to_profit"] = np.nan

    # 自由现金流 = OCF - 资本支出
    if "c_pay_acq_const_fiolta" in df.columns:
        df["fcf"] = df["ocf"] - df["c_pay_acq_const_fiolta"].abs()
    else:
        df["fcf"] = np.nan

    if "c_pay_acq_const_fiolta" in df.columns and df["ocf"].notna().any():
        df["capex_to_ocf"] = np.where(
            df["ocf"].abs() > 1e-6,
            df["c_pay_acq_const_fiolta"].abs() / df["ocf"],
            np.nan,
        )
    else:
        df["capex_to_ocf"] = np.nan

    available_cols = [c for c in CASHFLOW_COLS if c in df.columns]

    logger.info(
        f"现金流质量查询表构建: {df['ts_code'].nunique()} 只股票, "
        f"{len(trading_dates)} 个交易日"
    )

    factor_df = df[["ts_code", "ann_date", "end_date"] + available_cols].copy()
    result_dict = build_latest_announcement_lookup_by_date(
        factor_df,
        trading_dates,
        value_cols=available_cols,
        end_col="end_date",
        freshness_col=CASHFLOW_FRESHNESS_COL,
        log_name="现金流质量",
    )

    # 后处理：fcf_yield 需要总市值，在 feature builder 中合并时计算
    # 此处先保留 fcf 原始值

    logger.info(f"现金流质量日频查询表构建完成: {len(result_dict)} 个交易日")
    return result_dict
