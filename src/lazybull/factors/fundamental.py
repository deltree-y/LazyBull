"""基本面因子模块

将季度频率的财务指标数据（fina_indicator）前向填充到日频，
构建每日基本面查询表，供特征构建使用。

核心逻辑：
- 使用 ann_date（公告日期）而非 end_date（报告期末）作为数据可用时间点
- 防止前视偏差：Q4 报告（end_date=20231231）可能到 2024-04 才公告
- 对每个交易日，找到每只股票最近一次已公告的季报数据
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from .announcement_utils import build_latest_announcement_lookup_by_date


# 基本面因子列名
FUNDA_COLS = [
    # 盈利能力（5+5=10）
    'roe_waa', 'roe_dt', 'roa',                    # ROE/ROA 体系
    'or_yoy', 'netprofit_yoy', 'profit_dedt',      # 增速体系
    'q_gr_yoy', 'equity_yoy',                       # 单季增速 + 净资产增长
    'grossprofit_margin', 'netprofit_margin',       # 利润率体系
    # 盈利质量（2）
    'cf_sales', 'cf_nm',                            # 经营现金流/营收, /净利润
    # 偿债/流动性（2+1=3）
    'debt_to_assets', 'current_ratio', 'quick_ratio',
    # 无形资产风险（1）
    'int_to_talcap',                                 # 无形资产/总资本比（商誉风险代理）
    # 运营效率（2）
    'assets_turn', 'inv_turn',
]
FUNDA_AUX_COLS = ['q_ocf_to_sales']
FUNDAMENTAL_FRESHNESS_COL = 'fundamental_freshness_days'


def build_fundamental_lookup_by_date(
    fina_indicator: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """将季度财务指标前向填充到日频，构建每日基本面查询表

    对每只股票，在每个交易日查找 ann_date <= trade_date 的最新季报数据。

    Args:
        fina_indicator: 财务指标 DataFrame，需包含 ts_code, ann_date, end_date,
                        roe_waa, or_yoy, netprofit_yoy, debt_to_assets, q_gr_yoy
        trading_dates: 交易日列表（YYYYMMDD 格式字符串，已排序）

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, roe_waa, or_yoy, ...)}
    """
    df = fina_indicator.copy()

    # 清洗：去掉 ann_date 缺失的记录（无法确定公告时间）
    df = df.dropna(subset=['ann_date'])

    # 确保日期为字符串格式
    df['ann_date'] = df['ann_date'].astype(str).str[:8]
    df['end_date'] = df['end_date'].astype(str).str[:8]

    # 仅去除完全重复记录，保留同一报告期的多次公告版本，交由 PIT 查询按交易日选择
    df = df.sort_values(['ts_code', 'end_date', 'ann_date'])
    df = df.drop_duplicates(subset=['ts_code', 'end_date', 'ann_date'], keep='last')

    # 确保数值列为 float
    for col in FUNDA_COLS + FUNDA_AUX_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'q_ocf_to_sales' in df.columns:
        if 'cf_sales' in df.columns:
            df['cf_sales'] = df['cf_sales'].combine_first(df['q_ocf_to_sales'])
        else:
            df['cf_sales'] = df['q_ocf_to_sales']

    available_cols = [c for c in FUNDA_COLS if c in df.columns]
    available_aux_cols = [c for c in FUNDA_AUX_COLS if c in df.columns]

    logger.info(f"基本面查询表构建: {df['ts_code'].nunique()} 只股票, {len(trading_dates)} 个交易日")

    factor_df = df[['ts_code', 'ann_date'] + available_cols + available_aux_cols].copy()
    result_dict = build_latest_announcement_lookup_by_date(
        factor_df,
        trading_dates,
        value_cols=available_cols + available_aux_cols,
        freshness_col=FUNDAMENTAL_FRESHNESS_COL,
        log_name='基本面',
    )
    logger.info(f"基本面日频查询表构建完成: {len(result_dict)} 个交易日")
    return result_dict
