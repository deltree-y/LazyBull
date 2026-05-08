"""公告型 PIT 因子公共工具。

为按公告日 point-in-time 对齐到日频的因子提供统一的“最新已公告记录”查询，
并可额外输出公告新鲜度，让模型自行学习陈旧信息的折价，而不是直接硬截止。
"""

import bisect
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


def _normalize_date_str(value) -> Optional[str]:
    """将日期值标准化为 YYYYMMDD 字符串。"""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.strftime("%Y%m%d")

    text = str(value).strip().replace("-", "")
    if not text or text == "nan":
        return None
    return text[:8] if len(text) >= 8 else None


def build_latest_announcement_lookup_by_date(
    factor_df: pd.DataFrame,
    trading_dates: List[str],
    *,
    value_cols: List[str],
    code_col: str = "ts_code",
    ann_col: str = "ann_date",
    freshness_col: Optional[str] = None,
    log_name: str = "公告因子",
) -> Dict[str, pd.DataFrame]:
    """构建按公告日对齐、可附带 freshness 的日频查询表。

    Args:
        factor_df: 已完成因子计算的 DataFrame，需包含 code_col/ann_col/value_cols。
        trading_dates: 交易日列表。
        value_cols: 需要输出的因子列名。
        code_col: 证券代码列名。
        ann_col: 公告日期列名。
        freshness_col: 可选的新鲜度列名；若提供，则输出距公告日的天数。
        log_name: 日志展示名称。

    Returns:
        Dict[str, DataFrame]: {trade_date -> DataFrame(code_col, *value_cols)}
    """
    if factor_df is None or len(factor_df) == 0:
        return {}

    if not trading_dates:
        return {}

    required_cols = {code_col, ann_col, *value_cols}
    missing_cols = [col for col in required_cols if col not in factor_df.columns]
    if missing_cols:
        raise ValueError(f"{log_name} 缺少必要列: {missing_cols}")

    trade_dates = []
    for trade_date in trading_dates:
        normalized = _normalize_date_str(trade_date)
        if normalized is not None:
            trade_dates.append(normalized)

    if not trade_dates:
        return {}

    # 去重并保持输入顺序。
    ordered_trade_dates = list(dict.fromkeys(trade_dates))
    single_day = len(ordered_trade_dates) == 1

    df = factor_df.copy()
    df[ann_col] = df[ann_col].map(_normalize_date_str)
    df = df.dropna(subset=[code_col, ann_col])
    if df.empty:
        return {}

    df = df.sort_values([code_col, ann_col])
    trade_ts_map = {
        trade_date: pd.to_datetime(trade_date, format="%Y%m%d", errors="coerce")
        for trade_date in ordered_trade_dates
    }

    stock_ann_dates: Dict[str, List[str]] = {}
    stock_values: Dict[str, List[List[object]]] = {}
    stock_ann_timestamps: Dict[str, List[pd.Timestamp]] = {}

    for ts_code, grp in df.groupby(code_col, sort=False):
        grp = grp.sort_values(ann_col)
        ann_dates = grp[ann_col].tolist()
        stock_ann_dates[ts_code] = ann_dates
        stock_values[ts_code] = grp[value_cols].values.tolist()
        if freshness_col is not None:
            stock_ann_timestamps[ts_code] = pd.to_datetime(
                pd.Series(ann_dates), format="%Y%m%d", errors="coerce"
            ).tolist()

    result: Dict[str, pd.DataFrame] = {}
    output_cols = [code_col] + value_cols
    if freshness_col is not None:
        output_cols.append(freshness_col)

    for trade_date in ordered_trade_dates:
        trade_ts = trade_ts_map[trade_date]
        rows = []
        for ts_code, ann_dates in stock_ann_dates.items():
            idx = bisect.bisect_right(ann_dates, trade_date) - 1
            if idx < 0:
                continue

            values = stock_values[ts_code][idx]
            row = {code_col: ts_code}
            for value_idx, col in enumerate(value_cols):
                row[col] = values[value_idx]
            if freshness_col is not None:
                ann_ts = stock_ann_timestamps[ts_code][idx]
                if pd.isna(trade_ts) or pd.isna(ann_ts):
                    row[freshness_col] = float("nan")
                else:
                    row[freshness_col] = int((trade_ts - ann_ts).days)
            rows.append(row)

        if rows or single_day:
            result[trade_date] = pd.DataFrame(rows, columns=output_cols)

    logger.info(
        f"{log_name}查询表: 覆盖 {len(result)}/{len(ordered_trade_dates)} 个交易日"
        + (f"，附带 {freshness_col} 新鲜度列" if freshness_col is not None else "")
    )
    return result