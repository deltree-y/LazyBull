"""标签计算 —— 从 builder.py 拆出的 _calculate_forward_returns()。

标签语义对齐回测/纸面交易节奏：
- T 日（信号日）生成信号
- T+1 日收盘买入（close_adj）
- T+1+N 日开盘卖出（open_adj）
- 公式：y_ret_N = open_adj(T+1+N) / close_adj(T+1) - 1
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def compute_forward_returns(
    current_data: pd.DataFrame,
    trade_date: str,
    trading_dates: List[str],
    current_idx: int,
    horizons: List[int],
    daily_adj_dict: Optional[Dict[str, pd.DataFrame]] = None,
    daily_adj: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """计算未来 N 日收益标签（支持多个 horizon）。

    Args:
        current_data: 当日数据（仅用于 trade_date/ts_code 索引）
        trade_date: 当前交易日（信号日 T）
        trading_dates: 交易日序列
        current_idx: 当前交易日在序列中的索引
        horizons: 预测时间窗口列表
        daily_adj_dict: O(1) 按日期索引的复权日线字典（优先）
        daily_adj: 全量复权日线 DataFrame（回退路径）

    Returns:
        包含多个标签的 DataFrame（y_ret_5, y_ret_10, y_ret_20 等）
    """
    result = current_data[["trade_date", "ts_code"]].copy()

    if current_idx + 1 >= len(trading_dates):
        logger.warning(f"{trade_date} 后续无 T+1 交易日，所有 y_ret_* 标签为空")
        for horizon in horizons:
            result[f"y_ret_{horizon}"] = np.nan
        return result

    buy_date = trading_dates[current_idx + 1]
    if daily_adj_dict is not None:
        _buy_sub = daily_adj_dict.get(buy_date)
        buy_data = (
            _buy_sub[["ts_code", "close_adj"]].copy()
            if _buy_sub is not None
            else pd.DataFrame(columns=["ts_code", "close_adj"])
        )
    elif daily_adj is not None:
        buy_data = daily_adj[daily_adj["trade_date"] == buy_date][
            ["ts_code", "close_adj"]
        ].copy()
    else:
        buy_data = pd.DataFrame(columns=["ts_code", "close_adj"])

    buy_data.rename(columns={"close_adj": "buy_close_adj"}, inplace=True)
    result = result.merge(buy_data, on="ts_code", how="left")

    missing_summary = []
    for horizon in horizons:
        label_col = f"y_ret_{horizon}"

        sell_idx = current_idx + 1 + horizon
        if sell_idx >= len(trading_dates):
            logger.warning(
                f"{trade_date} 后续交易日不足 {horizon + 1} 天（T+1+{horizon} 越界），"
                f"{label_col} 标签为空"
            )
            result[label_col] = np.nan
            continue

        sell_date = trading_dates[sell_idx]
        if daily_adj_dict is not None:
            _sell_sub = daily_adj_dict.get(sell_date)
            sell_data = (
                _sell_sub[["ts_code", "open_adj"]].copy()
                if _sell_sub is not None
                else pd.DataFrame(columns=["ts_code", "open_adj"])
            )
        elif daily_adj is not None:
            sell_data = daily_adj[daily_adj["trade_date"] == sell_date][
                ["ts_code", "open_adj"]
            ].copy()
        else:
            sell_data = pd.DataFrame(columns=["ts_code", "open_adj"])

        sell_col = f"sell_open_adj_{horizon}"
        sell_data.rename(columns={"open_adj": sell_col}, inplace=True)
        result = result.merge(sell_data, on="ts_code", how="left")

        valid_mask = result["buy_close_adj"] > 1e-6
        result.loc[valid_mask, label_col] = (
            result.loc[valid_mask, sell_col] / result.loc[valid_mask, "buy_close_adj"]
        ) - 1
        result.loc[~valid_mask, label_col] = np.nan

        result.drop(columns=[sell_col], inplace=True)

        missing_labels = int(result[label_col].isna().sum())
        if missing_labels > 0:
            missing_summary.append(f"{label_col}={missing_labels}")

    if missing_summary:
        logger.debug(
            f"{trade_date} 标签缺失统计（T+1 收盘价或 T+1+N 开盘价缺失）: "
            + ", ".join(missing_summary)
        )

    result.drop(columns=["buy_close_adj"], inplace=True)
    return result
