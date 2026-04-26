"""基本面因子模块

将季度频率的财务指标数据（fina_indicator）前向填充到日频，
构建每日基本面查询表，供特征构建使用。

核心逻辑：
- 使用 ann_date（公告日期）而非 end_date（报告期末）作为数据可用时间点
- 防止前视偏差：Q4 报告（end_date=20231231）可能到 2024-04 才公告
- 对每个交易日，找到每只股票最近一次已公告的季报数据
"""

import bisect
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger


# 基本面因子列名
FUNDA_COLS = ['roe_waa', 'or_yoy', 'netprofit_yoy', 'debt_to_assets', 'q_gr_yoy']


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

    # 去重：同一股票同一报告期，保留最新公告的修正版
    df = df.sort_values(['ts_code', 'end_date', 'ann_date'])
    df = df.drop_duplicates(subset=['ts_code', 'end_date'], keep='last')

    # 确保数值列为 float
    for col in FUNDA_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 对增长率类指标做 winsorize 截断极端值
    _winsorize_growth_cols(df)

    available_cols = [c for c in FUNDA_COLS if c in df.columns]

    logger.info(f"基本面查询表构建: {df['ts_code'].nunique()} 只股票, {len(trading_dates)} 个交易日")

    if len(trading_dates) == 1:
        trade_date = trading_dates[0]
        visible = df[df['ann_date'] <= trade_date].sort_values(['ts_code', 'ann_date'])
        if visible.empty:
            result_dict = {trade_date: pd.DataFrame(columns=['ts_code'] + available_cols)}
        else:
            day_df = (
                visible.drop_duplicates(subset=['ts_code'], keep='last')
                [['ts_code'] + available_cols]
                .reset_index(drop=True)
            )
            result_dict = {trade_date: day_df}
        logger.info(f"基本面日频查询表构建完成: {len(result_dict)} 个交易日")
        return result_dict

    # 构建每只股票的 (ann_date, values) 有序列表
    stock_reports = {}  # {ts_code: [(ann_date, val1, val2, ...)]}
    stock_ann_dates = {}  # {ts_code: [ann_date1, ann_date2, ...]}

    for ts_code, group in df.groupby('ts_code'):
        group = group.sort_values('ann_date')
        ann_dates = group['ann_date'].tolist()
        values = group[available_cols].values.tolist()
        stock_reports[ts_code] = values
        stock_ann_dates[ts_code] = ann_dates

    all_stocks = list(stock_reports.keys())

    # 构建每日查询表
    result_dict = {}
    for trade_date in trading_dates:
        rows = []
        for ts_code in all_stocks:
            ann_dates = stock_ann_dates[ts_code]
            # 二分查找：找到 ann_date <= trade_date 的最后一个
            idx = bisect.bisect_right(ann_dates, trade_date) - 1
            if idx >= 0:
                rows.append([ts_code] + stock_reports[ts_code][idx])

        if rows:
            day_df = pd.DataFrame(rows, columns=['ts_code'] + available_cols)
        else:
            day_df = pd.DataFrame(columns=['ts_code'] + available_cols)
        result_dict[trade_date] = day_df

    logger.info(f"基本面日频查询表构建完成: {len(result_dict)} 个交易日")
    return result_dict


def _winsorize_growth_cols(df: pd.DataFrame) -> None:
    """对增长率列做 winsorize（原地修改）

    增长率类指标（or_yoy, netprofit_yoy, q_gr_yoy）可能出现极端值
    （如扭亏为盈导致 >10000%），截断到 1%~99% 分位。
    """
    growth_cols = ['or_yoy', 'netprofit_yoy', 'q_gr_yoy']
    for col in growth_cols:
        if col not in df.columns:
            continue
        s = df[col]
        valid = s.dropna()
        if len(valid) == 0:
            continue
        lower = valid.quantile(0.01)
        upper = valid.quantile(0.99)
        df[col] = s.clip(lower=lower, upper=upper)
