"""行业/市值中性化 —— 从 builder.py 拆出，含向量化性能优化。

优化要点：
- 用 groupby().transform() 批量处理所有列，替代逐列 groupby+merge（46+ 次 → 2 次）
- 市值中性化用 transform 替代双层 for 循环
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def apply_industry_neutralization(
    features: pd.DataFrame,
    horizons: List[int],
    lookback_windows: List[int],
    shenwan_level: str = "l2",
) -> pd.DataFrame:
    """应用行业中性化（去均值 + Z-Score），使用向量化 transform 批量处理。

    优先尝试分层回退中性化（L3→L2→L1→全市场），失败时回退单层 sw_industry。
    """
    if "sw_industry" not in features.columns:
        logger.error("缺少申万行业列 sw_industry，无法进行行业中性化。")
        return features

    result = features.copy()

    # ── 判断分层路径 ──
    hierarchy_cols = None
    hierarchy_label = f"{shenwan_level.upper()}→全市场"
    if shenwan_level == "l3" and all(
        col in result.columns for col in ["sw_industry_code", "sw_l2_code", "sw_l1_code"]
    ):
        hierarchy_cols = ("sw_industry_code", "sw_l2_code", "sw_l1_code")
        hierarchy_label = "L3→L2→L1→全市场"
    elif shenwan_level == "l2" and all(
        col in result.columns for col in ["sw_industry_code", "sw_l1_code"]
    ):
        hierarchy_cols = ("sw_industry_code", "sw_l1_code", "__missing_l1__")
        hierarchy_label = "L2→L1→全市场"

    has_hierarchy = hierarchy_cols is not None

    # ── 1. 去均值（demean）──
    demean_columns = _build_demean_columns(result, horizons, lookback_windows)
    if demean_columns:
        if has_hierarchy:
            result = _hierarchical_demean(result, demean_columns, hierarchy_cols)
        else:
            result = _vectorized_industry_demean(result, demean_columns)

    # ── 2. Z-Score ──
    zscore_columns = _build_zscore_columns(result, lookback_windows)
    if zscore_columns:
        if has_hierarchy:
            result = _hierarchical_zscore(result, zscore_columns, hierarchy_cols)
        else:
            result = _vectorized_industry_zscore(result, zscore_columns)

    return result


def apply_size_neutralization(
    result: pd.DataFrame,
    n_size_groups: int = 10,
) -> pd.DataFrame:
    """市值中性化：对 zscore_* 列按市值分位做组内 Z-Score，生成 zscore_*_sz 列。

    使用 groupby().transform() 向量化处理，消除原双层 for 循环。
    """
    # 去碎片化：上游大量逐列 merge/assign 导致 DataFrame 内部碎片，copy() 消除 PerformanceWarning
    result = result.copy()

    if "log_total_mv" not in result.columns:
        logger.warning("缺少 log_total_mv 列，跳过市值中性化")
        return result

    zscore_cols = [c for c in result.columns if c.startswith("zscore_")]
    if not zscore_cols:
        logger.debug("未找到 zscore_* 列，跳过市值中性化")
        return result

    tradable_mask = result.get("tradable", pd.Series(1, index=result.index)) == 1
    tradable_size = result.loc[tradable_mask, "log_total_mv"].dropna()

    if len(tradable_size) < n_size_groups * 2:
        logger.warning(f"可交易样本过少 ({len(tradable_size)})，跳过市值中性化")
        return result

    try:
        bins = np.unique(np.quantile(tradable_size, np.linspace(0, 1, n_size_groups + 1)))
        if len(bins) <= 1:
            logger.warning("市值分位边界过少，跳过市值中性化")
            return result
        result["_size_group"] = pd.cut(
            result["log_total_mv"], bins=bins, labels=False, include_lowest=True
        )
    except Exception as e:
        logger.warning(f"市值分位计算失败: {e}，跳过市值中性化")
        return result

    # 向量化：使用 groupby().transform() 一次性处理所有列
    sz_added = 0
    for col in zscore_cols:
        sz_col = f"{col}_sz"
        try:
            grp = result.groupby("_size_group")[col]
            mean = grp.transform("mean")
            std = grp.transform("std")
            mask = std > 1e-9
            result[sz_col] = np.where(mask, (result[col] - mean) / std, np.nan)
            if result[sz_col].notna().any():
                sz_added += 1
        except Exception:
            pass

    result.drop(columns=["_size_group"], inplace=True)
    if sz_added > 0:
        logger.debug(f"市值中性化完成: 新增 {sz_added} 个 zscore_*_sz 列")
    return result


# ── 辅助函数 ──────────────────────────────────────────────────


def _build_demean_columns(
    result: pd.DataFrame, horizons: List[int], lookback_windows: List[int]
) -> List[str]:
    """构建需要去均值的列名列表。"""
    demean_columns = []
    for horizon in horizons:
        label_col = f"y_ret_{horizon}"
        if label_col in result.columns:
            demean_columns.append(label_col)
    if "ret_1" in result.columns:
        demean_columns.append("ret_1")
    for window in lookback_windows:
        ret_col = f"ret_{window}"
        if ret_col in result.columns:
            demean_columns.append(ret_col)
    return demean_columns


def _build_zscore_columns(result: pd.DataFrame, lookback_windows: List[int]) -> List[str]:
    """构建需要 Z-Score 的特征列名列表。"""
    base = [
        "pe_ttm",
        "pb",
        "bp",
        "dv_ttm",
        "log_total_mv",
        "amount_ma20",
        "turnover_rate",
        "volatility_5",
        "volatility_10",
        "volatility_20",
        "net_mf_amount",
        "ma_deviation_20",
        "elg_net_amount_sum_20",
        "acceleration",
        "macd_hist",
        "bb_width",
        "roe_waa",
        "roe_dt",
        "roa",
        "or_yoy",
        "netprofit_yoy",
        "profit_dedt",
        "q_gr_yoy",
        "equity_yoy",
        "grossprofit_margin",
        "netprofit_margin",
        "debt_to_assets",
        "current_ratio",
        "quick_ratio",
        "cf_sales",
        "cf_nm",
        "int_to_talcap",
        "assets_turn",
        "inv_turn",
        "opening_strength",
        "intraday_vol_structure",
        "order_imbalance",
        "ocf_to_revenue",
        "ocf_to_profit",
        "fcf_yield",
        "capex_to_ocf",
        "cons_eps_revision_accel",
        "cons_eps_dispersion",
        "cons_eps_dispersion_chg",
        "cons_target_upside",
        "cons_revision_target_upside",
        "cons_target_upside_chg",
        "cons_analyst_count_chg",
        "cons_rating_upgrade_ratio",
    ]
    existing = [col for col in base if col in result.columns]
    for window in lookback_windows:
        vol_col = f"volatility_{window}"
        if vol_col in result.columns and vol_col not in existing:
            existing.append(vol_col)
    return existing


# ── 向量化单层 demean / zscore ────────────────────────────────


def _vectorized_industry_demean(result: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """用 groupby('sw_industry').transform() 一次性去均值。"""
    industry_col = "sw_industry"
    existing = [c for c in columns if c in result.columns]
    if not existing:
        return result

    grp = result.groupby(industry_col)[existing]
    means = grp.transform("mean")
    for col in existing:
        neu_col = f"neu_{col}"
        result[neu_col] = result[col] - means[col]

    logger.debug(f"去均值完成（向量化），新增 {len(existing)} 列")
    return result


def _vectorized_industry_zscore(result: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """用 groupby('sw_industry').transform() 一次性做行业内 Z-Score。"""
    industry_col = "sw_industry"
    existing = [c for c in columns if c in result.columns]
    if not existing:
        return result

    grp = result.groupby(industry_col)[existing]
    means = grp.transform("mean")
    stds = grp.transform("std")
    for col in existing:
        z_col = f"zscore_{col}"
        mask = stds[col] > 1e-9
        result[z_col] = np.where(mask, (result[col] - means[col]) / stds[col], np.nan)

    logger.debug(f"Z-Score 完成（向量化），新增 {len(existing)} 列")
    return result


# ── 分层回退路径（保持与原有分层行为一致）────────────────────


def _hierarchical_demean(
    result: pd.DataFrame,
    columns: List[str],
    hierarchy_cols: tuple,
) -> pd.DataFrame:
    """分层回退去均值（使用原有 hierarchical_demean）。"""
    from ..factors.hierarchical_industry_neutralization import hierarchical_demean

    try:
        result = hierarchical_demean(
            result,
            columns=columns,
            l3_col=hierarchy_cols[0],
            l2_col=hierarchy_cols[1],
            l1_col=hierarchy_cols[2],
            tradable_col="tradable",
            min_group_size=5,
            prefix="neu_",
        )
        actual_new = [f"neu_{c}" for c in columns if f"neu_{c}" in result.columns]
        logger.debug(f"分层去均值完成，新增 {len(actual_new)} 列")
    except Exception as e:
        logger.error(f"分层行业去均值失败：{e}")
    return result


def _hierarchical_zscore(
    result: pd.DataFrame,
    columns: List[str],
    hierarchy_cols: tuple,
) -> pd.DataFrame:
    """分层回退 Z-Score（使用原有 hierarchical_zscore）。"""
    from ..factors.hierarchical_industry_neutralization import hierarchical_zscore

    try:
        result = hierarchical_zscore(
            result,
            columns=columns,
            l3_col=hierarchy_cols[0],
            l2_col=hierarchy_cols[1],
            l1_col=hierarchy_cols[2],
            tradable_col="tradable",
            min_group_size=5,
            prefix="zscore_",
        )
        actual_new = [f"zscore_{c}" for c in columns if f"zscore_{c}" in result.columns]
        logger.debug(f"分层 Z-Score 完成，新增 {len(actual_new)} 列")
    except Exception as e:
        logger.error(f"分层行业内 Z-Score 失败：{e}")
    return result
