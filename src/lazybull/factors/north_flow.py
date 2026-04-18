"""北向资金因子模块

基于 TuShare moneyflow_hsgt 接口的沪深股通日频资金流数据构造市场级另类因子,
以广播形式写入每只股票 (同一交易日所有 ts_code 共享相同的北向值)。

数据来源：Tushare moneyflow_hsgt（2000 积分）
- hgt: 沪股通当日净买入（亿元）
- sgt: 深股通当日净买入（亿元）
- north_money: 北向资金净流入（亿元）= hgt + sgt

因子说明：
- north_flow: 当日北向净流入
- north_flow_ma5 / north_flow_ma20: N 日移动均值
- north_flow_z20: 20 日滚动 z-score（剔除均值/波动影响，留下方向信号）
- north_flow_sum5: 最近 5 日北向累计净流入
- north_flow_sign_streak: 连续同方向天数（正为持续流入，负为持续流出）
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


NORTH_COLS = [
    "north_flow",
    "north_flow_ma5",
    "north_flow_ma20",
    "north_flow_z20",
    "north_flow_sum5",
    "north_flow_sign_streak",
]


def _compute_sign_streak(series: pd.Series) -> pd.Series:
    """计算连续同方向天数 (正=连续流入, 负=连续流出, 0=本日净零)"""
    sign = np.sign(series.fillna(0.0))
    streak = np.zeros(len(sign), dtype=float)
    last = 0.0
    for i, s in enumerate(sign.values):
        if s == 0:
            last = 0.0
        elif last == 0 or np.sign(last) != s:
            last = s
        else:
            last = last + s
        streak[i] = last
    return pd.Series(streak, index=series.index)


def build_north_flow_lookup_by_date(
    hsgt_df: pd.DataFrame,
    trading_dates: List[str],
) -> Dict[str, Dict[str, float]]:
    """构建北向资金市场级因子查询表

    与其他因子不同, 北向是市场级数据 (一天一条), 因此 lookup 的 value
    是 `Dict[str, float]` (列名 -> 当日值), FeatureBuilder 合并时对
    全部 ts_code 广播同一份值。

    Args:
        hsgt_df: moneyflow_hsgt 原始 DataFrame, 需含 trade_date, north_money
        trading_dates: 交易日列表 (YYYYMMDD 字符串)

    Returns:
        Dict[trade_date -> Dict[col_name -> float]]
    """
    if hsgt_df is None or len(hsgt_df) == 0:
        logger.warning("北向资金因子: 输入数据为空")
        return {}

    df = hsgt_df.copy()
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "").str[:8]

    # 统一北向净流入列名
    if "north_money" in df.columns:
        df["north_flow"] = pd.to_numeric(df["north_money"], errors="coerce")
    elif "hgt" in df.columns and "sgt" in df.columns:
        df["north_flow"] = (
            pd.to_numeric(df["hgt"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["sgt"], errors="coerce").fillna(0.0)
        )
    else:
        logger.warning("北向资金因子: 缺少 north_money/hgt/sgt 列")
        return {}

    df = df.drop_duplicates("trade_date").sort_values("trade_date").reset_index(drop=True)

    # 滚动特征
    df["north_flow_ma5"] = df["north_flow"].rolling(5, min_periods=1).mean()
    df["north_flow_ma20"] = df["north_flow"].rolling(20, min_periods=1).mean()
    roll_std = df["north_flow"].rolling(20, min_periods=5).std()
    df["north_flow_z20"] = (df["north_flow"] - df["north_flow_ma20"]) / roll_std.replace(
        0, np.nan
    )
    df["north_flow_sum5"] = df["north_flow"].rolling(5, min_periods=1).sum()
    df["north_flow_sign_streak"] = _compute_sign_streak(df["north_flow"])

    date_set = set(trading_dates)
    result: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        td = row["trade_date"]
        if td not in date_set:
            continue
        rec = {}
        for col in NORTH_COLS:
            val = row.get(col)
            rec[col] = float(val) if pd.notna(val) else np.nan
        result[td] = rec

    logger.info(f"北向资金因子查询表: 覆盖 {len(result)}/{len(trading_dates)} 个交易日")
    return result
