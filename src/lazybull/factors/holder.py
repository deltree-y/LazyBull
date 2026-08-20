"""股东人数因子模块

将不定期公布的股东人数数据（stk_holdernumber）前向填充到日频，
构建每日筹码集中度特征。

数据来源：Tushare stk_holdernumber API（2000 积分）
更新频率：约 10 日一次（随公司公告）
核心逻辑：
- 使用 ann_date（公告日期）做 point-in-time 对齐，防止前视偏差
- 股东人数下降 → 筹码集中 → 看多信号
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..common.date_utils import normalize_series_to_yyyymmdd
from .announcement_utils import build_latest_announcement_lookup_by_date


HOLDER_COLS = [
    "holder_num_chg",      # 股东人数环比变动率
    "holder_num_chg_2q",   # 两期变动率
]

HOLDER_FRESHNESS_COL = "holder_freshness_days"


def _compute_holder_cross_period_changes(df: pd.DataFrame) -> pd.DataFrame:
    """为每个公告版本计算跨报告期环比基准（上一/前二报告期最新已公告值）。

    每个版本的环比基准 = 公告日不晚于本版本、报告期早于本版本的最新已公告值；
    同一报告期存在多个版本（首发+修正）时，基准取其中公告最晚（修正）的值，
    避免朴素 shift 取到同报告期修正版本导致信号被稀释。
    """
    results = []
    for _ts_code, grp in df.groupby("ts_code", sort=False):
        grp = grp.sort_values(["ann_date", "end_date"], kind="mergesort").reset_index(drop=True)
        ends = grp["end_date"].tolist()
        anns = grp["ann_date"].tolist()
        holders = grp["holder_num"].tolist()
        prev1 = [np.nan] * len(grp)
        prev2 = [np.nan] * len(grp)
        for i in range(len(grp)):
            # end_date -> (ann_date, holder)：已公告且报告期早于当前版本的最优基准
            best_by_end: Dict[str, Tuple[str, float]] = {}
            for j in range(i):
                if ends[j] >= ends[i]:
                    continue
                if ends[j] not in best_by_end or anns[j] > best_by_end[ends[j]][0]:
                    best_by_end[ends[j]] = (anns[j], holders[j])
            sorted_ends = sorted(best_by_end.keys(), reverse=True)
            if sorted_ends:
                prev1[i] = best_by_end[sorted_ends[0]][1]
            if len(sorted_ends) >= 2:
                prev2[i] = best_by_end[sorted_ends[1]][1]
        grp["holder_num_prev"] = prev1
        grp["holder_num_prev2"] = prev2
        results.append(grp)
    if not results:
        # 空输入边界：保持输出 schema（补齐基准列），避免下游列访问 KeyError
        empty = df.iloc[0:0].copy()
        empty["holder_num_prev"] = np.nan
        empty["holder_num_prev2"] = np.nan
        return empty
    return pd.concat(results, ignore_index=True)


def build_holder_lookup_by_date(
    stk_holdernumber: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, pd.DataFrame]:
    """将股东人数数据按 ann_date point-in-time 对齐到日频

    Args:
        stk_holdernumber: 股东人数 DataFrame，需包含
            ts_code, ann_date, end_date, holder_num
        trading_dates: 交易日列表（YYYYMMDD 字符串，已排序）

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(ts_code, holder_num_chg, ...)}
    """
    df = stk_holdernumber.copy()

    # 日期标准化
    for col in ["ann_date", "end_date"]:
        if col in df.columns:
            df[col] = normalize_series_to_yyyymmdd(df[col])

    df["holder_num"] = pd.to_numeric(df["holder_num"], errors="coerce")
    df = df.dropna(subset=["ann_date", "holder_num"])

    # 仅去除完全重复记录；保留同报告期多公告版本，
    # 每个版本的跨期环比基准由 _compute_holder_cross_period_changes 按 PIT 规则计算
    df = df.sort_values(["ts_code", "end_date", "ann_date"])
    df = df.drop_duplicates(subset=["ts_code", "end_date", "ann_date"], keep="last")

    df = _compute_holder_cross_period_changes(df)
    df["holder_num_chg"] = (
        df["holder_num"] - df["holder_num_prev"]
    ) / df["holder_num_prev"].replace(0, np.nan)
    df["holder_num_chg_2q"] = (
        df["holder_num"] - df["holder_num_prev2"]
    ) / df["holder_num_prev2"].replace(0, np.nan)

    factor_df = df[["ts_code", "ann_date", "end_date"] + HOLDER_COLS].copy()
    return build_latest_announcement_lookup_by_date(
        factor_df,
        trading_dates,
        value_cols=HOLDER_COLS,
        freshness_col=HOLDER_FRESHNESS_COL,
        end_col="end_date",
        log_name="股东人数",
    )
