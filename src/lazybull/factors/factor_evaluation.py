#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子IC评估模块

提供单因子RankIC/ICIR计算的核心功能，支持行业中性化和市值中性化。

核心函数：
- compute_daily_factor_ic: 单日单因子IC
- compute_factor_ic_series: 多日因子IC序列
- compute_factor_ic_summary: IC汇总统计(均值/标准差/ICIR/胜率)
- compute_ic_decay: IC衰减曲线(多horizon)
- neutralize_factor: 因子中性化(行业/市值/双重)
- evaluate_all_factors: 批量评估所有因子

使用示例:
    >>> from src.lazybull.factors.factor_evaluation import evaluate_all_factors
    >>> summary = evaluate_all_factors(
    ...     data, factor_cols, label_cols=["y_ret_5", "y_ret_10", "y_ret_20"],
    ...     neutralize_mode="both"
    ... )
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr, spearmanr


# ── 非因子列模式（自动排除）────────────────────────────────────────
# 这些前缀/精确匹配的列不被视为因子，不会参与IC计算
NON_FACTOR_PATTERNS = {
    # 元数据
    "trade_date",
    "ts_code",
    "name",
    # 标签列
    "y_ret_",
    "neu_y_ret_",
    # 行业信息
    "sw_l1",
    "sw_l2",
    "sw_l3",
    "sw_l1_code",
    "sw_l2_code",
    "sw_l3_code",
    "sw_l1_id",
    "sw_l2_id",
    "sw_industry",
    "sw_industry_code",
    "sw_industry_id",
    "in_date",
    # 过滤标记
    "is_st",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
    "is_new_stock",
    "tradable",
    "list_days",
    # 市场状态
    "mkt_",
    # 数据新鲜度
    "_freshness_days",
    # 原始量价（非特征，因子已从这些衍生）
    "vol",
    "amount",
}

# 市值中性化时用作分层依据的列
DEFAULT_SIZE_COL = "log_total_mv"

# 行业中性化时默认使用的行业列
DEFAULT_INDUSTRY_COL = "sw_l1_code"


def _is_factor_column(col: str) -> bool:
    """判断某列是否为因子列（排除元数据、标签、过滤标记等）"""
    for pattern in NON_FACTOR_PATTERNS:
        if pattern.startswith("_") or pattern.endswith("_"):
            # 带前后缀的模式：包含匹配
            if pattern in col:
                return False
        else:
            # 精确匹配或前缀匹配
            if col == pattern or col.startswith(pattern):
                return False
    return True


def auto_detect_factor_columns(df: pd.DataFrame) -> List[str]:
    """从DataFrame中自动识别因子列

    Args:
        df: 特征截面DataFrame

    Returns:
        因子列名列表
    """
    factor_cols = [c for c in df.columns if _is_factor_column(c)]
    logger.info(f"自动识别到 {len(factor_cols)} 个因子列（总列数: {len(df.columns)}）")
    return factor_cols


def auto_detect_label_columns(df: pd.DataFrame) -> List[str]:
    """从DataFrame中自动识别标签列（y_ret_* 但不含 neu_ 前缀）"""
    label_cols = [
        c for c in df.columns if c.startswith("y_ret_") and not c.startswith("neu_")
    ]
    label_cols = sorted(label_cols)
    return label_cols


def _align_and_filter(
    factor: pd.Series,
    label: pd.Series,
    tradable: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """对齐因子与标签，移除NaN和不可交易样本

    Args:
        factor: 因子值Series
        label: 标签值Series
        tradable: 可交易标记Series（1=可交易），为None则不筛选

    Returns:
        (factor_array, label_array) numpy数组
    """
    # 对齐index
    common_idx = factor.index.intersection(label.index)
    if len(common_idx) == 0:
        return np.array([]), np.array([])

    f = factor.loc[common_idx]
    l = label.loc[common_idx]

    # 移除NaN
    valid = f.notna() & l.notna()
    if tradable is not None:
        tradable_aligned = tradable.reindex(common_idx, fill_value=0)
        valid = valid & (tradable_aligned == 1)

    if valid.sum() < 2:
        return np.array([]), np.array([])

    return f[valid].values, l[valid].values


def compute_daily_rankic(
    factor: np.ndarray,
    label: np.ndarray,
) -> float:
    """计算单日RankIC（Spearman秩相关系数）

    Args:
        factor: 因子值数组
        label: 标签值数组

    Returns:
        RankIC值，样本不足时返回NaN
    """
    if len(factor) < 3:
        return np.nan
    ic, _ = spearmanr(factor, label)
    return float(ic)


def compute_daily_pearson_ic(
    factor: np.ndarray,
    label: np.ndarray,
) -> float:
    """计算单日Pearson IC（线性相关系数）

    Args:
        factor: 因子值数组
        label: 标签值数组

    Returns:
        Pearson IC值，样本不足时返回NaN
    """
    if len(factor) < 3:
        return np.nan
    ic, _ = pearsonr(factor, label)
    return float(ic)


def neutralize_factor(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: Optional[str] = None,
    size_col: Optional[str] = None,
    n_size_groups: int = 10,
    tradable_col: str = "tradable",
) -> pd.Series:
    """对因子进行行业和/或市值中性化

    行业中性化：因子值减去行业内均值（行业均值仅用可交易样本计算）
    市值中性化：按市值分n_size_groups组，因子值减去组内均值

    Args:
        df: 单日截面DataFrame
        factor_col: 因子列名
        industry_col: 行业列名，None表示不做行业中性化
        size_col: 市值列名，None表示不做市值中性化
        n_size_groups: 市值分组数，默认10（十分位）
        tradable_col: 可交易标记列名

    Returns:
        中性化后的因子值Series，索引与df对齐
    """
    result = df[factor_col].copy()

    if industry_col is not None and industry_col in df.columns:
        # 行业中性化：减去行业内均值
        tradable_mask = df.get(tradable_col, pd.Series(1, index=df.index)) == 1
        industry_means = df.loc[tradable_mask].groupby(industry_col)[factor_col].mean()
        result = result - df[industry_col].map(industry_means)

    if size_col is not None and size_col in df.columns:
        # 市值中性化：按市值分位数组内去均值
        tradable_mask = df.get(tradable_col, pd.Series(1, index=df.index)) == 1
        tradable_df = df[tradable_mask].dropna(subset=[size_col])

        if len(tradable_df) >= n_size_groups * 2:
            # 对可交易样本计算分位数边界
            size_values = tradable_df[size_col]
            quantiles = np.linspace(0, 1, n_size_groups + 1)
            bins = np.quantile(size_values, quantiles)
            # 避免边界重复导致分组失败
            bins = np.unique(bins)
            if len(bins) > 1:
                size_groups = pd.cut(df[size_col], bins=bins, labels=False, include_lowest=True)
                # 对全部样本（含不可交易）使用整体均值做回退
                group_means = df.groupby(size_groups)[factor_col].transform("mean")
                result = result - group_means

    return result


def compute_daily_factor_ic(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    neutralize_industry: bool = False,
    neutralize_size: bool = False,
    industry_col: str = DEFAULT_INDUSTRY_COL,
    size_col: str = DEFAULT_SIZE_COL,
    n_size_groups: int = 10,
    tradable_col: str = "tradable",
) -> Dict[str, Any]:
    """计算单日单因子的RankIC和Pearson IC

    Args:
        df: 单日截面DataFrame，需包含因子列、标签列、tradable列
        factor_col: 因子列名
        label_col: 标签列名（如 y_ret_5）
        neutralize_industry: 是否行业中性化
        neutralize_size: 是否市值中性化
        industry_col: 行业列名
        size_col: 市值列名
        n_size_groups: 市值中性化分组数
        tradable_col: 可交易标记列名

    Returns:
        {
            "trade_date": str,
            "factor": str,
            "label": str,
            "rank_ic": float,
            "pearson_ic": float,
            "n_samples": int,
            "neutralize": str,
        }
    """
    # 提取因子值和标签值
    factor_raw = df[factor_col]

    # 中性化处理
    neutralize_desc = "raw"
    if neutralize_industry or neutralize_size:
        factor_raw = neutralize_factor(
            df,
            factor_col,
            industry_col=industry_col if neutralize_industry else None,
            size_col=size_col if neutralize_size else None,
            n_size_groups=n_size_groups,
            tradable_col=tradable_col,
        )
        parts = []
        if neutralize_industry:
            parts.append("ind")
        if neutralize_size:
            parts.append("size")
        neutralize_desc = "_".join(parts)

    # 对齐并过滤
    label = df[label_col] if label_col in df.columns else pd.Series(dtype=float)
    tradable = df.get(tradable_col)

    f_arr, l_arr = _align_and_filter(factor_raw, label, tradable)

    rank_ic = compute_daily_rankic(f_arr, l_arr) if len(f_arr) >= 3 else np.nan
    pearson_ic = compute_daily_pearson_ic(f_arr, l_arr) if len(f_arr) >= 3 else np.nan

    return {
        "trade_date": str(df.iloc[0].get("trade_date", "")),
        "factor": factor_col,
        "label": label_col,
        "rank_ic": rank_ic,
        "pearson_ic": pearson_ic,
        "n_samples": len(f_arr),
        "neutralize": neutralize_desc,
    }


def compute_factor_ic_series(
    data: pd.DataFrame,
    factor_col: str,
    label_col: str,
    date_col: str = "trade_date",
    neutralize_industry: bool = False,
    neutralize_size: bool = False,
    industry_col: str = DEFAULT_INDUSTRY_COL,
    size_col: str = DEFAULT_SIZE_COL,
    n_size_groups: int = 10,
    tradable_col: str = "tradable",
    verbose: bool = False,
) -> pd.DataFrame:
    """计算单个因子在多日上的IC序列

    Args:
        data: 多日截面DataFrame
        factor_col: 因子列名
        label_col: 标签列名
        date_col: 日期列名
        neutralize_industry: 是否行业中性化
        neutralize_size: 是否市值中性化
        industry_col: 行业列名
        size_col: 市值列名
        n_size_groups: 市值中性化分组数
        tradable_col: 可交易标记列名
        verbose: 是否输出逐日日志

    Returns:
        逐日IC DataFrame，列: trade_date, factor, label, rank_ic, pearson_ic, n_samples, neutralize
    """
    dates = sorted(data[date_col].unique())
    records = []

    for date in dates:
        day_df = data[data[date_col] == date]
        result = compute_daily_factor_ic(
            day_df,
            factor_col,
            label_col,
            neutralize_industry=neutralize_industry,
            neutralize_size=neutralize_size,
            industry_col=industry_col,
            size_col=size_col,
            n_size_groups=n_size_groups,
            tradable_col=tradable_col,
        )
        records.append(result)
        if verbose and len(records) % 50 == 0:
            logger.debug(f"  [{factor_col}] 已处理 {len(records)}/{len(dates)} 日")

    return pd.DataFrame(records)


def compute_factor_ic_summary(
    daily_ic: pd.DataFrame,
    ic_col: str = "rank_ic",
) -> Dict[str, float]:
    """从逐日IC序列计算汇总统计

    Args:
        daily_ic: compute_factor_ic_series 的输出
        ic_col: IC列名，默认 "rank_ic"

    Returns:
        {
            "IC_mean": float,       # IC均值
            "IC_std": float,        # IC标准差
            "ICIR": float,          # IC信息比 = mean/std
            "IC_win_rate": float,   # IC > 0 的交易日占比
            "IC_pos_mean": float,   # 正IC日均值
            "IC_neg_mean": float,   # 负IC日均值
            "IC_abs_mean": float,   # |IC| 均值
            "n_days": int,          # 有效交易日数
            "n_days_valid": int,    # IC非NaN的交易日数
        }
    """
    ic_series = daily_ic[ic_col].dropna()

    if len(ic_series) == 0:
        return {
            "IC_mean": np.nan,
            "IC_std": np.nan,
            "ICIR": np.nan,
            "IC_win_rate": np.nan,
            "IC_pos_mean": np.nan,
            "IC_neg_mean": np.nan,
            "IC_abs_mean": np.nan,
            "n_days": len(daily_ic),
            "n_days_valid": 0,
        }

    ic_mean = float(ic_series.mean())
    ic_std = float(ic_series.std())
    icir = ic_mean / ic_std if ic_std > 0 else np.nan

    positive_mask = ic_series > 0
    ic_win_rate = float(positive_mask.sum() / len(ic_series))

    ic_pos_mean = float(ic_series[positive_mask].mean()) if positive_mask.any() else np.nan
    neg_mask = ic_series < 0
    ic_neg_mean = float(ic_series[neg_mask].mean()) if neg_mask.any() else np.nan
    ic_abs_mean = float(ic_series.abs().mean())

    return {
        "IC_mean": ic_mean,
        "IC_std": ic_std,
        "ICIR": icir,
        "IC_win_rate": ic_win_rate,
        "IC_pos_mean": ic_pos_mean,
        "IC_neg_mean": ic_neg_mean,
        "IC_abs_mean": ic_abs_mean,
        "n_days": len(daily_ic),
        "n_days_valid": len(ic_series),
    }


def compute_ic_decay(
    data: pd.DataFrame,
    factor_col: str,
    label_cols: List[str],
    date_col: str = "trade_date",
    neutralize_industry: bool = False,
    neutralize_size: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """计算因子在不同horizon标签上的IC衰减曲线

    Args:
        data: 多日截面DataFrame
        factor_col: 因子列名
        label_cols: 标签列名列表，按horizon从小到大排序（如 ["y_ret_5","y_ret_10","y_ret_20"]）
        date_col: 日期列名
        neutralize_industry: 是否行业中性化
        neutralize_size: 是否市值中性化
        **kwargs: 传递给 compute_factor_ic_series 的其他参数

    Returns:
        DataFrame，列: horizon, horizon_days, IC_mean, IC_std, ICIR, IC_win_rate
    """
    records = []
    for label_col in label_cols:
        # 从列名提取 horizon 天数
        horizon_days = _extract_horizon_days(label_col)

        daily_ic = compute_factor_ic_series(
            data,
            factor_col,
            label_col,
            date_col=date_col,
            neutralize_industry=neutralize_industry,
            neutralize_size=neutralize_size,
            **kwargs,
        )
        summary = compute_factor_ic_summary(daily_ic)
        summary["horizon"] = label_col
        summary["horizon_days"] = horizon_days
        summary["factor"] = factor_col
        records.append(summary)

    result = pd.DataFrame(records)
    # 按 horizon_days 升序排列，确保衰减曲线从小到大
    if "horizon_days" in result.columns:
        result = result.sort_values("horizon_days").reset_index(drop=True)
    return result


def _extract_horizon_days(label_col: str) -> int:
    """从标签列名提取horizon天数，如 y_ret_5 → 5"""
    try:
        return int(label_col.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def _get_neutralize_label(
    neutralize_industry: bool,
    neutralize_size: bool,
) -> str:
    """生成中性化模式标签"""
    if neutralize_industry and neutralize_size:
        return "ind_size"
    elif neutralize_industry:
        return "industry"
    elif neutralize_size:
        return "size"
    return "raw"


def _neutralize_factor_matrix(
    day_df: pd.DataFrame,
    factor_cols: List[str],
    industry_col: Optional[str] = None,
    size_col: Optional[str] = None,
    n_size_groups: int = 10,
    tradable_col: str = "tradable",
) -> np.ndarray:
    """对单日所有因子批量中性化（向量化实现，无逐因子循环）

    行业中性化：因子值减去行业内可交易样本均值
    市值中性化：按市值十分位去组内均值

    Args:
        day_df: 单日截面 DataFrame
        factor_cols: 因子列名列表
        industry_col: 行业列名，None 则不做行业中性化
        size_col: 市值列名，None 则不做市值中性化
        n_size_groups: 市值分组数
        tradable_col: 可交易标记列名

    Returns:
        (n_samples, n_factors) 中性化后的因子矩阵
    """
    result = day_df[factor_cols].values.astype(np.float64)
    tradable_mask = day_df.get(tradable_col, pd.Series(1, index=day_df.index)) == 1

    # ── 行业中性化（向量化）──
    if industry_col is not None and industry_col in day_df.columns:
        # 仅用可交易样本计算行业均值 (n_industries × n_factors)
        tradable_df = day_df.loc[tradable_mask, [industry_col] + factor_cols]
        ind_means = tradable_df.groupby(industry_col)[factor_cols].mean()
        # 将行业均值映射到每行
        row_means = ind_means.reindex(day_df[industry_col]).values
        # NaN 行业（无对应均值）填 0（不调整）
        row_means = np.nan_to_num(row_means, nan=0.0)
        result = result - row_means

    # ── 市值中性化（向量化）──
    if size_col is not None and size_col in day_df.columns:
        size_values = day_df[size_col].values
        tradable_size = size_values[tradable_mask.values]
        tradable_size = tradable_size[~np.isnan(tradable_size)]

        if len(tradable_size) >= n_size_groups * 2:
            bins = np.unique(np.quantile(tradable_size, np.linspace(0, 1, n_size_groups + 1)))
            if len(bins) > 1:
                # 分配每个样本到市值分组
                size_groups = np.digitize(size_values, bins[1:-1], right=True)
                # 计算每组均值（对所有因子列）
                unique_groups = np.unique(size_groups[~np.isnan(size_groups)])
                for g in unique_groups:
                    g = int(g)
                    mask = size_groups == g
                    if mask.sum() > 1:
                        result[mask] = result[mask] - result[mask].mean(axis=0, keepdims=True)

    return result


def _compute_daily_batch_ic(
    day_df: pd.DataFrame,
    factor_cols: List[str],
    label_col: str,
    tradable_col: str = "tradable",
    min_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """单日批量计算所有因子的 RankIC 和 Pearson IC

    对每个因子独立计算 IC（避免全因子非空要求导致样本量为0），
    但使用 numpy 向量化替代 scipy.spearmanr，单因子耗时约 50μs。

    Args:
        day_df: 单日截面 DataFrame
        factor_cols: 因子列名列表
        label_col: 标签列名
        tradable_col: 可交易标记列名
        min_samples: 最少有效样本数

    Returns:
        (rank_ic_array, pearson_ic_array, n_samples_common)
    """
    from scipy.stats import rankdata

    n_factors = len(factor_cols)
    nan_arr = np.full(n_factors, np.nan)

    # ── 公共过滤：tradable + label 非 NaN ──
    tradable_mask = day_df.get(tradable_col, pd.Series(1, index=day_df.index)) == 1
    label_valid = day_df[label_col].notna()
    common_mask = tradable_mask.values & label_valid.values
    n_common = common_mask.sum()

    if n_common < min_samples:
        return nan_arr, nan_arr, n_common

    # 提取公共有效子集（仅一次）
    common_idx = day_df.index[common_mask]
    label_arr = day_df.loc[common_idx, label_col].values.astype(np.float64)

    rank_ic = np.full(n_factors, np.nan)
    pearson_ic = np.full(n_factors, np.nan)

    # ── 逐因子在自身非 NaN 子集上计算 IC ──
    for j, factor_col in enumerate(factor_cols):
        factor_vals = day_df.loc[common_idx, factor_col].values.astype(np.float64)
        valid = ~np.isnan(factor_vals)
        n_valid = valid.sum()
        if n_valid < min_samples:
            continue

        f = factor_vals[valid]
        l = label_arr[valid]

        # RankIC (Spearman = Pearson on ranks)
        f_rank = rankdata(f)
        l_rank = rankdata(l)
        rank_ic[j] = _pearson_1d(f_rank, l_rank)

        # Pearson IC
        pearson_ic[j] = _pearson_1d(f, l)

    return rank_ic, pearson_ic, n_common


def _pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    """单变量 Pearson 相关系数（numpy 实现，比 scipy 快 3-5 倍）"""
    x_c = x - x.mean()
    y_c = y - y.mean()
    numer = np.dot(x_c, y_c)
    denom = np.sqrt(np.dot(x_c, x_c) * np.dot(y_c, y_c))
    if denom == 0:
        return np.nan
    return float(numer / denom)


def _vectorized_pearson(
    factor_arr: np.ndarray,
    label_arr: np.ndarray,
) -> np.ndarray:
    """向量化计算多列因子与同一标签的 Pearson 相关系数

    Args:
        factor_arr: (n_samples, n_factors) 因子矩阵
        label_arr: (n_samples,) 标签向量

    Returns:
        (n_factors,) 相关系数数组
    """
    # 中心化
    f_centered = factor_arr - factor_arr.mean(axis=0, keepdims=True)
    l_centered = label_arr - label_arr.mean()

    # 分子: (n_factors,) = f_centered^T @ l_centered
    numer = f_centered.T @ l_centered

    # 分母: ||f_centered||_col * ||l_centered||
    f_norm = np.sqrt((f_centered ** 2).sum(axis=0))
    l_norm = np.sqrt((l_centered ** 2).sum())

    denom = f_norm * l_norm
    # 避免除零
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numer / denom
    result[denom == 0] = np.nan
    return result


def _evaluate_factors_for_label(
    data: pd.DataFrame,
    factor_cols: List[str],
    label_col: str,
    date_col: str,
    neutralize_industry: bool,
    neutralize_size: bool,
    industry_col: str,
    size_col: str,
    n_size_groups: int,
    tradable_col: str,
    verbose: bool,
) -> pd.DataFrame:
    """针对单个标签，批量计算所有因子的逐日IC序列（优化版）

    每天一次性算出所有因子的 IC，然后按因子聚合为汇总统计。

    Returns:
        DataFrame，每行一个因子，含 IC_mean/IC_std/ICIR 等汇总列
    """
    dates = sorted(data[date_col].unique())
    n_dates = len(dates)
    n_factors = len(factor_cols)

    # ── 逐日批量计算 IC ──
    # daily_rank_ic[j][i] = 第 i 天第 j 个因子的 RankIC
    daily_rank_ic = np.full((n_factors, n_dates), np.nan)
    daily_pearson_ic = np.full((n_factors, n_dates), np.nan)
    daily_n_samples = np.zeros(n_dates, dtype=int)

    for day_idx, date in enumerate(dates):
        day_df = data[data[date_col] == date]

        # 中性化（如果启用）
        if neutralize_industry or neutralize_size:
            day_factors = _neutralize_factor_matrix(
                day_df,
                factor_cols,
                industry_col=industry_col if neutralize_industry else None,
                size_col=size_col if neutralize_size else None,
                n_size_groups=n_size_groups,
                tradable_col=tradable_col,
            )
            # 用中性化后的值替换原因子列
            day_df = day_df.copy()
            day_df[factor_cols] = day_factors

        rank_ics, pearson_ics, n_valid = _compute_daily_batch_ic(
            day_df, factor_cols, label_col, tradable_col
        )
        daily_rank_ic[:, day_idx] = rank_ics
        daily_pearson_ic[:, day_idx] = pearson_ics
        daily_n_samples[day_idx] = n_valid

        if verbose and (day_idx + 1) % 200 == 0:
            logger.info(
                f"  [{label_col}] 已处理 {day_idx + 1}/{n_dates} 日 "
                f"({(day_idx + 1) / n_dates:.0%})"
            )

    # ── 按因子聚合汇总统计 ──
    horizon_days = _extract_horizon_days(label_col)
    records = []

    for j, factor_col in enumerate(factor_cols):
        rank_ic_series = daily_rank_ic[j]
        # 构造伪 DataFrame 以复用 compute_factor_ic_summary
        daily_df = pd.DataFrame({"rank_ic": rank_ic_series, "pearson_ic": daily_pearson_ic[j]})
        summary = compute_factor_ic_summary(daily_df)
        summary["factor"] = factor_col
        summary["label"] = label_col
        summary["horizon_days"] = horizon_days
        records.append(summary)

    return pd.DataFrame(records)


def evaluate_all_factors(
    data: pd.DataFrame,
    factor_cols: Optional[List[str]] = None,
    label_cols: Optional[List[str]] = None,
    date_col: str = "trade_date",
    neutralize_mode: str = "none",
    industry_col: str = DEFAULT_INDUSTRY_COL,
    size_col: str = DEFAULT_SIZE_COL,
    n_size_groups: int = 10,
    tradable_col: str = "tradable",
    verbose: bool = True,
) -> pd.DataFrame:
    """批量评估所有因子在所有horizon上的IC（向量化优化版）

    对每个标签，每天一次性计算所有因子的 IC，然后按因子汇总。
    相比逐因子逐日循环，速度提升 50-100 倍。

    Args:
        data: 多日截面DataFrame，包含所有因子列和标签列
        factor_cols: 因子列名列表，None则自动检测
        label_cols: 标签列名列表，None则自动检测（y_ret_5, y_ret_10, y_ret_20）
        date_col: 日期列名
        neutralize_mode: 中性化模式
            - "none": 不中性化
            - "industry": 仅行业中性化
            - "size": 仅市值中性化
            - "both": 行业+市值双重中性化
        industry_col: 行业列名
        size_col: 市值列名
        n_size_groups: 市值分组数
        tradable_col: 可交易标记列名
        verbose: 是否输出进度日志

    Returns:
        DataFrame，每行为一个 (因子, 标签) 组合的IC汇总
    """
    # 自动检测列
    if factor_cols is None:
        factor_cols = auto_detect_factor_columns(data)

    if label_cols is None:
        label_cols = auto_detect_label_columns(data)

    if not factor_cols:
        logger.error("未检测到任何因子列，请检查数据")
        return pd.DataFrame()

    if not label_cols:
        logger.error("未检测到任何标签列（y_ret_*），请检查数据")
        return pd.DataFrame()

    # 中性化参数
    neutralize_industry = neutralize_mode in ("industry", "both")
    neutralize_size = neutralize_mode in ("size", "both")
    neutralize_label = _get_neutralize_label(neutralize_industry, neutralize_size)

    logger.info(
        f"因子IC评估开始: {len(factor_cols)} 个因子 × {len(label_cols)} 个标签"
        f" × {neutralize_label} 中性化"
        f" | 数据范围: {data[date_col].min()} ~ {data[date_col].max()}"
        f" | 交易日数: {data[date_col].nunique()}"
    )

    all_parts = []
    for label_col in label_cols:
        logger.info(f"  ▸ 处理标签: {label_col} ...")
        part = _evaluate_factors_for_label(
            data=data,
            factor_cols=factor_cols,
            label_col=label_col,
            date_col=date_col,
            neutralize_industry=neutralize_industry,
            neutralize_size=neutralize_size,
            industry_col=industry_col,
            size_col=size_col,
            n_size_groups=n_size_groups,
            tradable_col=tradable_col,
            verbose=verbose,
        )
        part["neutralize"] = neutralize_label
        all_parts.append(part)

    result = pd.concat(all_parts, ignore_index=True)

    # 按 ICIR 降序排列
    if "ICIR" in result.columns:
        result = result.sort_values("ICIR", ascending=False).reset_index(drop=True)

    logger.info(
        f"因子IC评估完成: {len(result)} 条记录, "
        f"ICIR 范围: [{result['ICIR'].min():.3f}, {result['ICIR'].max():.3f}]"
    )

    return result


def compare_neutralization_modes(
    data: pd.DataFrame,
    factor_cols: Optional[List[str]] = None,
    label_cols: Optional[List[str]] = None,
    date_col: str = "trade_date",
    modes: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """对比不同中性化模式下的因子IC变化

    对同一个因子在不同中性化模式下分别计算IC，用于识别：
    - 因子是否依赖行业暴露（raw IC高但industry中性化后IC大幅下降）
    - 因子是否依赖市值暴露（raw IC高但size中性化后IC大幅下降）
    - 真正的alpha因子（所有中性化后IC仍显著为正）

    Args:
        data: 多日截面DataFrame
        factor_cols: 因子列名列表，None则自动检测
        label_cols: 标签列名列表
        date_col: 日期列名
        modes: 中性化模式列表，默认 ["none", "industry", "size", "both"]
        **kwargs: 传递给 evaluate_all_factors

    Returns:
        DataFrame，与 evaluate_all_factors 输出格式相同，但 neutralize 列有多个值
    """
    if modes is None:
        modes = ["none", "industry", "size", "both"]

    # 提前检测因子列（只做一次）
    if factor_cols is None:
        factor_cols = auto_detect_factor_columns(data)
    if label_cols is None:
        label_cols = auto_detect_label_columns(data)

    n_modes = len(modes)
    all_results = []
    for mode_idx, mode in enumerate(modes):
        logger.info(
            f"── 中性化模式 [{mode_idx + 1}/{n_modes}]: {mode} ──"
        )
        result = evaluate_all_factors(
            data,
            factor_cols=factor_cols,
            label_cols=label_cols,
            date_col=date_col,
            neutralize_mode=mode,
            verbose=True,  # 恢复进度日志
            **kwargs,
        )
        all_results.append(result)

    combined = pd.concat(all_results, ignore_index=True)
    return combined.sort_values("ICIR", ascending=False).reset_index(drop=True)
