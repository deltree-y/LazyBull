# -*- coding: utf-8 -*-
"""FeatureBuilder 静态附加函数：价值红利/资金流/风控/高级因子/过滤标记（无状态）。"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from ...common.date_utils import normalize_date_columns, to_trade_date_str
from ...factors import (
    calculate_acceleration,
    calculate_amplitude,
    calculate_industry_alpha_windows,
    calculate_shadows,
    calculate_volume_burst,
)
from ...factors.normalization import cross_sectional_zscore
from .static_core import _get_lookback_dates_static


def _add_value_dividend_features_static(
    features: pd.DataFrame,
    daily_basic_data: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    """静态版价值红利特征构建，供串行与并行路径共用。"""
    from ...common.feature_utils import log1p_transform

    daily_basic_today = daily_basic_data[daily_basic_data["trade_date"] == trade_date].copy()
    if len(daily_basic_today) == 0:
        logger.warning(f"{trade_date} 没有 daily_basic 数据，价值红利特征将为空")
        return features

    value_cols = [
        "ts_code",
        "pb",
        "pe_ttm",
        "ps_ttm",
        "dv_ttm",
        "total_mv",
        "circ_mv",
        "turnover_rate",
        "volume_ratio",
    ]
    existing_cols = ["ts_code"] + [c for c in value_cols[1:] if c in daily_basic_today.columns]
    daily_basic_today = daily_basic_today[existing_cols].copy()
    features = features.merge(daily_basic_today, on="ts_code", how="left")

    if "dv_ttm" in features.columns:
        features["dv_ttm"] = features["dv_ttm"].fillna(0)
    if "pe_ttm" in features.columns:
        features["ep_ttm"] = np.where(
            (features["pe_ttm"].notna()) & (features["pe_ttm"] > 0),
            1.0 / features["pe_ttm"],
            np.nan,
        )
        features["is_loss"] = ((features["pe_ttm"].isna()) | (features["pe_ttm"] <= 0)).astype(int)
    if "pb" in features.columns:
        features["bp"] = np.where(
            (features["pb"].notna()) & (features["pb"] > 0),
            1.0 / features["pb"],
            np.nan,
        )
    if "total_mv" in features.columns:
        features["log_total_mv"] = log1p_transform(features["total_mv"])
    if "circ_mv" in features.columns:
        features["log_circ_mv"] = log1p_transform(features["circ_mv"])

    return features


def _add_moneyflow_features_static(
    features: pd.DataFrame,
    moneyflow_data: pd.DataFrame,
    trade_date: str,
    trading_dates: List[str],
    current_idx: int,
    trading_date_index: Optional[Dict[str, int]],
) -> pd.DataFrame:
    """静态版资金流特征构建，供串行与并行路径共用。"""
    from ...common.feature_utils import winsorize_series

    moneyflow_today = moneyflow_data[moneyflow_data["trade_date"] == trade_date].copy()
    if len(moneyflow_today) == 0:
        logger.warning(f"{trade_date} 没有 moneyflow 数据，资金流特征将为空")
        return features

    merge_cols = ["ts_code", "net_mf_amount"]
    merge_cols = [c for c in merge_cols if c in moneyflow_today.columns]
    if len(merge_cols) > 1:
        features = features.merge(moneyflow_today[merge_cols], on="ts_code", how="left")

    if "buy_lg_amount" in moneyflow_today.columns and "sell_lg_amount" in moneyflow_today.columns:
        moneyflow_today["lg_net_amount"] = moneyflow_today["buy_lg_amount"] - moneyflow_today["sell_lg_amount"]
        features = features.merge(moneyflow_today[["ts_code", "lg_net_amount"]], on="ts_code", how="left")

    if "buy_elg_amount" in moneyflow_today.columns and "sell_elg_amount" in moneyflow_today.columns:
        moneyflow_today["elg_net_amount"] = (
            moneyflow_today["buy_elg_amount"] - moneyflow_today["sell_elg_amount"]
        )
        features = features.merge(moneyflow_today[["ts_code", "elg_net_amount"]], on="ts_code", how="left")
        _total = moneyflow_today["buy_elg_amount"] + moneyflow_today["sell_elg_amount"]
        moneyflow_today["order_imbalance"] = np.where(
            _total > 1e-6,
            (moneyflow_today["buy_elg_amount"] - moneyflow_today["sell_elg_amount"]) / _total,
            np.nan,
        )
        features = features.merge(moneyflow_today[["ts_code", "order_imbalance"]], on="ts_code", how="left")

    for window in [5, 20]:
        hist_dates = _get_lookback_dates_static(
            trade_date,
            window,
            trading_dates,
            trading_date_index,
        )
        if not hist_dates:
            features[f"net_mf_amount_sum_{window}"] = np.nan
            features[f"net_mf_amount_mean_{window}"] = np.nan
            if "lg_net_amount" in features.columns:
                features[f"lg_net_amount_sum_{window}"] = np.nan
            if "elg_net_amount" in features.columns:
                features[f"elg_net_amount_sum_{window}"] = np.nan
            continue

        hist_moneyflow = moneyflow_data[moneyflow_data["trade_date"].isin(hist_dates)].copy()
        if len(hist_moneyflow) == 0:
            continue

        if "buy_lg_amount" in hist_moneyflow.columns and "sell_lg_amount" in hist_moneyflow.columns:
            hist_moneyflow["lg_net_amount"] = (
                hist_moneyflow["buy_lg_amount"] - hist_moneyflow["sell_lg_amount"]
            )
        if "buy_elg_amount" in hist_moneyflow.columns and "sell_elg_amount" in hist_moneyflow.columns:
            hist_moneyflow["elg_net_amount"] = (
                hist_moneyflow["buy_elg_amount"] - hist_moneyflow["sell_elg_amount"]
            )
            _total = hist_moneyflow["buy_elg_amount"] + hist_moneyflow["sell_elg_amount"]
            hist_moneyflow["order_imbalance"] = np.where(
                _total > 1e-6,
                (hist_moneyflow["buy_elg_amount"] - hist_moneyflow["sell_elg_amount"]) / _total,
                np.nan,
            )

        agg_dict = {}
        if "net_mf_amount" in hist_moneyflow.columns:
            agg_dict["net_mf_amount"] = ["sum", "mean"]
        if "lg_net_amount" in hist_moneyflow.columns:
            agg_dict["lg_net_amount"] = ["sum"]
        if "elg_net_amount" in hist_moneyflow.columns:
            agg_dict["elg_net_amount"] = ["sum"]
        if "order_imbalance" in hist_moneyflow.columns:
            agg_dict["order_imbalance"] = ["mean"]
        if not agg_dict:
            continue

        rolling_features = hist_moneyflow.groupby("ts_code").agg(agg_dict).reset_index()
        new_columns = ["ts_code"]
        for col in rolling_features.columns[1:]:
            if isinstance(col, tuple):
                new_columns.append(f"{col[0]}_{col[1]}_{window}")
            else:
                new_columns.append(col)
        rolling_features.columns = new_columns
        features = features.merge(rolling_features, on="ts_code", how="left")

    winsorize_cols = [c for c in features.columns if "net_amount" in c or "mf_amount" in c]
    for col in winsorize_cols:
        if col in features.columns:
            features[col] = winsorize_series(features[col], limits=(0.01, 0.01))

    return features


def _attach_risk_factors_static(
    features: pd.DataFrame,
    trade_date: str,
    risk_factor_cache_dict: Optional[Dict[str, pd.DataFrame]],
    risk_factor_names: Optional[List[str]],
) -> pd.DataFrame:
    """静态版风控因子合并：预计算缓存查表 + 公告类截面因子逐日计算。

    供串行（FeatureBuilder）与并行（parallel.py）两条路径共用。
    """
    if risk_factor_cache_dict is None or not risk_factor_names:
        return features
    try:
        from ...risk.factor_registry import compute_all_risk_factors
    except ImportError:
        return features

    # 1. 历史窗口因子：O(1) 查表合并
    day_df = risk_factor_cache_dict.get(trade_date)
    if day_df is not None and len(day_df) > 0:
        merged = features[["ts_code"]].merge(day_df, on="ts_code", how="left")
        block = merged[list(risk_factor_names)]
        block.index = features.index
        features = pd.concat([features, block], axis=1)
    else:
        nan_block = pd.DataFrame(np.nan, index=features.index, columns=list(risk_factor_names))
        features = pd.concat([features, nan_block], axis=1)

    # 2. 公告类截面因子：逐日计算（不依赖 daily_adj 历史窗口，开销极小）
    risk_cols = compute_all_risk_factors(
        df=features,
        daily_adj=None,
        market_state=None,
        trade_date=trade_date,
        exclude=set(risk_factor_names),
    )
    if risk_cols:
        new_df = pd.DataFrame(
            {name: (s.values if isinstance(s, pd.Series) else s) for name, s in risk_cols.items()},
            index=features.index,
        )
        features = pd.concat([features, new_df], axis=1)
    return features


def _add_advanced_factors_static(
    features: pd.DataFrame,
    current_data: pd.DataFrame,
    trade_date: str,
    trading_dates: List[str],
    current_idx: int,
    lookback_windows: List[int],
    tech_factor_cache_dict: Optional[Dict[str, pd.DataFrame]] = None,
    _get_tech_factor_today_fn=None,
) -> pd.DataFrame:
    """模块级高级因子计算（无状态版本）。"""
    result = features.copy()

    if all(
        col in current_data.columns for col in ["high_adj", "low_adj", "pre_close", "adj_factor"]
    ):
        amplitude_df = calculate_amplitude(current_data)
        result = result.merge(amplitude_df, on=["ts_code", "trade_date"], how="left")

    if all(col in current_data.columns for col in ["open_adj", "high_adj", "low_adj", "close_adj"]):
        shadows_df = calculate_shadows(current_data)
        result = result.merge(shadows_df, on=["ts_code", "trade_date"], how="left")

    if "ret_1" in result.columns and current_idx >= max(lookback_windows):
        tech_today = None
        if tech_factor_cache_dict is not None:
            tech_today = tech_factor_cache_dict.get(
                trade_date, pd.DataFrame(columns=["ts_code", "trade_date"])
            )
        elif _get_tech_factor_today_fn is not None:
            tech_today = _get_tech_factor_today_fn(trade_date)
        if tech_today is not None and len(tech_today) > 0:
            vol_cols = [
                f"volatility_{w}"
                for w in lookback_windows
                if f"volatility_{w}" in tech_today.columns
            ]
            if vol_cols:
                result = result.merge(
                    tech_today[["ts_code", "trade_date"] + vol_cols],
                    on=["ts_code", "trade_date"],
                    how="left",
                )

    if all(f"ret_{w}" in result.columns for w in lookback_windows):
        industry_col = "sw_industry" if "sw_industry" in result.columns else None
        if industry_col is not None:
            industry_alpha_df = calculate_industry_alpha_windows(
                result,
                ret_windows=lookback_windows,
                industry_col=industry_col,
            )
            result = result.merge(
                industry_alpha_df,
                on=["ts_code", "trade_date"],
                how="left",
            )
            from ...factors.industry import calculate_industry_momentum_features

            ind_mom_df = calculate_industry_momentum_features(
                result,
                industry_col=industry_col,
                ret_col="ret_20",
            )
            result = result.merge(ind_mom_df, on=["ts_code", "trade_date"], how="left")

    if "ret_5" in result.columns and "ret_10" in result.columns:
        acceleration_df = calculate_acceleration(result)
        result = result.merge(acceleration_df, on=["ts_code", "trade_date"], how="left")

    vol_ratio_cols = [f"vol_ratio_{w}" for w in lookback_windows]
    if all(col in result.columns for col in vol_ratio_cols):
        vol_burst_df = calculate_volume_burst(result, vol_ratio_windows=lookback_windows)
        result = result.merge(vol_burst_df, on=["ts_code", "trade_date"], how="left")

    if current_idx >= 30:
        tech_today = None
        if tech_factor_cache_dict is not None:
            tech_today = tech_factor_cache_dict.get(
                trade_date, pd.DataFrame(columns=["ts_code", "trade_date"])
            )
        elif _get_tech_factor_today_fn is not None:
            tech_today = _get_tech_factor_today_fn(trade_date)
        if tech_today is not None and len(tech_today) > 0:
            tech_indicator_cols = [
                c
                for c in [
                    "rsi_14",
                    "kdj_k",
                    "kdj_d",
                    "kdj_j",
                    "macd_dif",
                    "macd_dea",
                    "macd_hist",
                    "bb_middle",
                    "bb_upper",
                    "bb_lower",
                    "bb_width",
                    "bb_pct",
                    "atr_14",
                    "atr_pct_14",
                ]
                if c in tech_today.columns
            ]
            if tech_indicator_cols:
                result = result.merge(
                    tech_today[["ts_code", "trade_date"] + tech_indicator_cols],
                    on=["ts_code", "trade_date"],
                    how="left",
                )

    return result


def _add_filter_flags_static(
    df: pd.DataFrame,
    stock_basic: pd.DataFrame,
    suspend_info: Optional[pd.DataFrame],
    trade_date: str,
) -> pd.DataFrame:
    result = df.copy()
    has_clean_flags = all(
        col in result.columns for col in ["is_st", "is_suspended", "tradable", "list_days"]
    )
    if has_clean_flags:
        logger.debug("数据已包含 clean 层过滤标记，直接复用")
        return result

    logger.info("clean 层标记不存在，开始计算过滤标记")
    stock_names = stock_basic[["ts_code", "name"]].copy()
    result = result.merge(stock_names, on="ts_code", how="left")
    result["is_st"] = (
        result["name"]
        .fillna("")
        .str.contains(r"^\*?S?\*?ST|退", case=False, regex=True)
        .astype(int)
    )
    stock_list_date = stock_basic[["ts_code", "list_date"]].copy()
    if pd.api.types.is_datetime64_any_dtype(stock_list_date["list_date"]):
        stock_list_date["list_date"] = stock_list_date["list_date"].dt.strftime("%Y%m%d")
    result = result.merge(
        stock_list_date,
        on="ts_code",
        how="left",
        suffixes=("", "_basic"),
    )
    try:
        trade_date_dt = pd.to_datetime(trade_date, format="%Y%m%d")
        result["list_date_dt"] = pd.to_datetime(
            result["list_date"],
            format="%Y%m%d",
            errors="coerce",
        )
        result["list_days"] = (trade_date_dt - result["list_date_dt"]).dt.days
        result.drop(columns=["list_date_dt"], inplace=True)
    except Exception as e:
        logger.warning(f"计算上市天数失败: {e}，使用默认值")
        result["list_days"] = 999

    if "vol" in result.columns:
        result["is_suspended"] = (result["vol"] <= 0).astype(int)
    else:
        result["is_suspended"] = 0

    if suspend_info is not None and len(suspend_info) > 0:
        if "suspend_date" in suspend_info.columns and "resume_date" in suspend_info.columns:
            suspend_info_normalized = normalize_date_columns(
                suspend_info,
                ["suspend_date", "resume_date"],
                to_str=True,
            )
            trade_date_str = to_trade_date_str(trade_date)
            suspend_today = suspend_info_normalized[
                (suspend_info_normalized["suspend_date"] <= trade_date_str)
                & (
                    (suspend_info_normalized["resume_date"] >= trade_date_str)
                    | (suspend_info_normalized["resume_date"].isna())
                )
            ]["ts_code"].unique()
            result.loc[result["ts_code"].isin(suspend_today), "is_suspended"] = 1
        elif "trade_date" in suspend_info.columns and "suspend_type" in suspend_info.columns:
            suspend_info_normalized = normalize_date_columns(
                suspend_info,
                ["trade_date"],
                to_str=True,
            )
            trade_date_str = to_trade_date_str(trade_date)
            suspend_today = suspend_info_normalized[
                (suspend_info_normalized["trade_date"] == trade_date_str)
                & (suspend_info_normalized["suspend_type"] == "S")
            ]["ts_code"].unique()
            result.loc[result["ts_code"].isin(suspend_today), "is_suspended"] = 1

    return result


def _add_limit_flags_static(
    df: pd.DataFrame,
    daily_data: pd.DataFrame,
    limit_info: Optional[pd.DataFrame],
    trade_date: str,
) -> pd.DataFrame:
    result = df.copy()
    has_clean_limit_flags = all(col in result.columns for col in ["is_limit_up", "is_limit_down"])
    if has_clean_limit_flags:
        logger.debug("数据已包含 clean 层涨跌停标记，直接复用")
        return result

    logger.warning(
        "检测到缺失 clean 层涨跌停标记，按设计约束不在 features 层重算，"
        "将回退填充为 0。"
    )
    result["is_limit_up"] = 0
    result["is_limit_down"] = 0
    return result


def _apply_filters_static(
    df: pd.DataFrame,
    require_label: bool = True,
    label_filter_mode: str = "all",
    horizon: int = 20,
    horizons: Optional[List[int]] = None,
    min_list_days: int = 365,
) -> pd.DataFrame:
    horizons = horizons or [5, 10, 20]

    filter_mask = (
        (df["is_st"] == 0) & (df["list_days"] >= min_list_days) & (df["is_suspended"] == 0)
    )

    if require_label:
        if label_filter_mode == "single":
            primary_col = f"y_ret_{horizon}"
            if primary_col in df.columns:
                filter_mask = filter_mask & df[primary_col].notna()
        else:
            for h in horizons:
                label_col = f"y_ret_{h}"
                if label_col in df.columns:
                    filter_mask = filter_mask & df[label_col].notna()

    return df[filter_mask].copy()


def _add_new_individual_features_static(result: pd.DataFrame) -> pd.DataFrame:
    # 去碎片化：上游多次逐列 merge/assign 导致 DataFrame 内部碎片，copy() 消除 PerformanceWarning
    result = result.copy()
    if "list_days" in result.columns:
        result["is_new_stock"] = (result["list_days"] < 365).astype(int)
    else:
        result["is_new_stock"] = 0

    if "circ_mv" in result.columns:
        result["size"] = result["circ_mv"]

    if "size" in result.columns and "sw_industry" in result.columns:
        result["_log1p_size"] = np.log1p(result["size"])
        result = cross_sectional_zscore(
            result,
            columns=["_log1p_size"],
            group_col="sw_industry",
            tradable_col="tradable",
            min_group_size=5,
            suffix="_z",
        )
        if "_log1p_size_z" in result.columns:
            result.rename(columns={"_log1p_size_z": "zscore_size"}, inplace=True)
        if "_log1p_size" in result.columns:
            result.drop(columns=["_log1p_size"], inplace=True)

    if "zscore_volatility_20" in result.columns and "zscore_size" in result.columns:
        result["spec_score"] = result["zscore_volatility_20"] * (-result["zscore_size"])
    else:
        result["spec_score"] = np.nan

    return result