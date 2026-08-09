# -*- coding: utf-8 -*-
"""FeatureBuilder 静态核心函数：基础特征/窗口特征/回看日期（无状态，供并行路径共用）。"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def _calculate_base_features(
    current_data: pd.DataFrame,
    daily_adj_dict: Optional[Dict[str, pd.DataFrame]],
    trade_date: str,
    trading_dates: List[str],
    current_idx: int,
    lookback_windows: List[int],
    trading_date_index: Optional[Dict[str, int]],
    daily_basic_data: Optional[pd.DataFrame] = None,
    moneyflow_data: Optional[pd.DataFrame] = None,
    daily_adj: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """模块级基础特征计算（无状态版本）。"""
    base_columns = ["trade_date", "ts_code", "vol", "amount"]
    clean_marker_columns = [
        "is_st",
        "is_suspended",
        "is_limit_up",
        "is_limit_down",
        "list_days",
        "tradable",
    ]
    columns_to_keep = base_columns.copy()
    for col in clean_marker_columns:
        if col in current_data.columns:
            columns_to_keep.append(col)

    features = current_data[columns_to_keep].copy()

    # ret_1
    features = features.merge(
        current_data[["ts_code", "pct_chg"]],
        on="ts_code",
        how="left",
        suffixes=("", "_dup"),
    )
    features.rename(columns={"pct_chg": "ret_1"}, inplace=True)
    features["ret_1"] = features["ret_1"] / 100.0

    # opening_strength
    if "open" in current_data.columns and "pre_close" in current_data.columns:
        _open = current_data[["ts_code", "open", "pre_close"]].copy()
        _open["opening_strength"] = np.where(
            _open["pre_close"] > 1e-6,
            _open["open"] / _open["pre_close"] - 1,
            np.nan,
        )
        features = features.merge(
            _open[["ts_code", "opening_strength"]],
            on="ts_code",
            how="left",
        )

    # intraday_vol_structure
    if all(c in current_data.columns for c in ["high", "open", "low"]):
        _hloc = current_data[["ts_code", "high", "open", "low"]].copy()
        _up = _hloc["high"] - _hloc["open"]
        _down = _hloc["open"] - _hloc["low"]
        _hloc["intraday_vol_structure"] = np.where(_down > 1e-6, _up / _down, np.nan)
        features = features.merge(
            _hloc[["ts_code", "intraday_vol_structure"]],
            on="ts_code",
            how="left",
        )

    # 回看特征
    for window in lookback_windows:
        hist_dates = _get_lookback_dates_static(
            trade_date,
            window,
            trading_dates,
            trading_date_index,
        )
        if not hist_dates:
            features[f"ret_{window}"] = np.nan
            features[f"vol_ratio_{window}"] = np.nan
            features[f"ma_deviation_{window}"] = np.nan
            continue
        if daily_adj_dict is not None:
            _frames = [daily_adj_dict[d] for d in hist_dates if d in daily_adj_dict]
            hist_data = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
        elif daily_adj is not None:
            hist_data = daily_adj[daily_adj["trade_date"].isin(hist_dates)]
        else:
            hist_data = pd.DataFrame()

        hist_features = _calculate_window_features_static(hist_data, current_data, window)
        features = features.merge(hist_features, on="ts_code", how="left")

    return features


# cf_nm 无 fina_indicator 侧独立代理，依赖 cashflow 因子 ocf_to_profit 回填；
# 缺失/未启用时按构建会话只提示一次，避免批量构建按交易日刷屏
_CF_NM_DEP_WARNED = False


def _backfill_fundamental_proxy_features_static(features: pd.DataFrame) -> pd.DataFrame:
    """用可稳定获取的字段回填基本面代理列（cf_sales、cf_nm）。"""
    global _CF_NM_DEP_WARNED
    features = features.copy()

    if "cf_sales" not in features.columns:
        features["cf_sales"] = np.nan
    if "q_ocf_to_sales" in features.columns:
        features["cf_sales"] = features["cf_sales"].combine_first(features["q_ocf_to_sales"])
    if "ocf_to_revenue" in features.columns:
        features["cf_sales"] = features["cf_sales"].combine_first(features["ocf_to_revenue"])

    if "cf_nm" not in features.columns:
        features["cf_nm"] = np.nan
    if "ocf_to_profit" in features.columns:
        features["cf_nm"] = features["cf_nm"].combine_first(features["ocf_to_profit"])
    elif features["cf_nm"].isna().all() and not _CF_NM_DEP_WARNED:
        # 缺列可能是 cashflow 未启用，也可能是数据缺失；
        # 按构建会话只提示一次，避免批量构建按交易日刷屏
        logger.warning(
            "cf_nm 依赖 cashflow 因子 ocf_to_profit 回填，当前该列缺失/未启用，"
            "cf_nm 保持全 NaN（训练侧将按常量列剔除）"
        )
        _CF_NM_DEP_WARNED = True

    return features


def _get_lookback_dates_static(
    trade_date: str,
    n: int,
    trading_dates: List[str],
    trading_date_index: Optional[Dict[str, int]],
) -> List[str]:
    if trading_date_index is not None:
        idx = trading_date_index.get(trade_date, -1)
        if idx == -1:
            return []
    else:
        if trade_date not in trading_dates:
            return []
        idx = trading_dates.index(trade_date)
    if idx < n:
        return []
    return trading_dates[idx - n : idx]


def _calculate_window_features_static(
    hist_data: pd.DataFrame, current_data: pd.DataFrame, window: int
) -> pd.DataFrame:
    if len(hist_data) == 0:
        return pd.DataFrame(columns=["ts_code"])
    hist_data = hist_data.sort_values(["ts_code", "trade_date"])
    grouped = hist_data.groupby("ts_code", as_index=False)
    window_features = grouped.agg(
        {"close_adj": ["first", "last", "mean"], "vol": "mean", "amount": "mean"}
    )
    new_columns = []
    for col in window_features.columns:
        if col[0] == "ts_code":
            new_columns.append("ts_code")
        else:
            new_columns.append("_".join(col).strip("_"))
    window_features.columns = new_columns
    window_features = window_features.rename(
        columns={
            "close_adj_first": "first_close",
            "close_adj_last": "last_close",
            "close_adj_mean": "ma_close",
            "vol_mean": "mean_vol",
            "amount_mean": "mean_amount",
        }
    )
    window_features[f"ret_{window}"] = (
        window_features["last_close"] / window_features["first_close"]
    ) - 1
    current_vol_amount = current_data[["ts_code", "vol", "amount", "close_adj"]].copy()
    window_features = window_features.merge(current_vol_amount, on="ts_code", how="left")
    window_features[f"vol_ratio_{window}"] = np.where(
        window_features["mean_vol"] > 1e-6,
        window_features["vol"] / window_features["mean_vol"],
        np.nan,
    )
    window_features[f"ma_deviation_{window}"] = np.where(
        window_features["ma_close"] > 1e-6,
        (window_features["close_adj"] - window_features["ma_close"]) / window_features["ma_close"],
        np.nan,
    )
    keep_cols = [
        "ts_code",
        f"ret_{window}",
        f"vol_ratio_{window}",
        f"ma_deviation_{window}",
        "mean_amount",
    ]
    window_features = window_features[keep_cols]
    window_features = window_features.rename(columns={"mean_amount": f"amount_ma{window}"})
    return window_features