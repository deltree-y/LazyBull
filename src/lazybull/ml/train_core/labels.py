# -*- coding: utf-8 -*-
"""labels：train_core 拆分模块。"""

from loguru import logger
from src.lazybull.common.feature_utils import cross_sectional_zscore as _single_col_zscore
from typing import Optional
import numpy as np
import pandas as pd


def add_blended_return_label(
    df: pd.DataFrame,
    neutral_label: str,
    blend_weight: float,
) -> str:
    """按权重混合行业中性标签与原始收益标签，并返回实际训练列名。"""
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError(f"neutral_label_blend_weight 必须在 [0, 1]，实际为 {blend_weight}")
    if blend_weight == 0.0:
        return neutral_label
    if not neutral_label.startswith("neu_y_ret_"):
        raise ValueError(
            "neutral_label_blend_weight > 0 时标签必须为 neu_y_ret_N，"
            f"实际为 {neutral_label}"
        )

    raw_label = neutral_label[len("neu_") :]
    missing_columns = [
        col for col in (neutral_label, raw_label) if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"混合标签缺少源列: {', '.join(missing_columns)}")

    horizon = neutral_label.rsplit("_", 1)[-1]
    blended_label = f"y_blend_ret_{horizon}"
    df[blended_label] = (
        (1.0 - blend_weight) * df[neutral_label] + blend_weight * df[raw_label]
    )
    return blended_label

def transform_labels_cs_zscore(
    df: pd.DataFrame, label_column: str, winsorize_p: float = 0.01
) -> pd.DataFrame:
    """对标签进行截面 winsorize + zscore 变换

    仅在训练阶段生效，对每个 trade_date 的原始回归标签进行：
    1. 截面 winsorize（截断极端值）
    2. 截面 zscore（标准化：均值=0，标准差=1）

    Args:
        df: 训练数据 DataFrame
        label_column: 标签列名
        winsorize_p: winsorize 参数，默认 0.01（截断上下1%极端值）

    Returns:
        变换后的 DataFrame（标签列已替换为标准化后的值）
    """
    # 使用别名以区别于 normalization.cross_sectional_zscore（后者处理多列 DataFrame）
    from src.lazybull.common.feature_utils import cross_sectional_zscore as _single_col_zscore

    cross_sectional_zscore = _single_col_zscore

    logger.info(f"对标签 {label_column} 进行截面 z-score 标准化...")
    logger.info(f"  winsorize 参数: {winsorize_p}")

    nan_count_ori = df[label_column].isna().sum()
    logger.info(f"原始标签 NaN 数量: {nan_count_ori}")

    # 按 trade_date 分组进行截面标准化（仅对标签列计算，不深拷贝整个 DataFrame）
    transformed_label = cross_sectional_zscore(
        df,
        value_col=label_column,
        group_col="trade_date",
        winsorize_limits=(winsorize_p, winsorize_p),
        ddof=0,
    )

    # 统计标准化后的效果
    mean = transformed_label.mean()
    std = transformed_label.std()
    logger.info(f"标准化后: 均值={mean:.6f}, 标准差={std:.6f}")

    # 检查是否有 NaN（可能由于某天标准差为0）
    nan_mask = transformed_label.isna()
    nan_count = nan_mask.sum()
    if nan_count > 0:
        logger.warning(f"标准化后产生 {nan_count} 个 NaN（可能某天标准差为0），将被移除")
        df_transformed = df.loc[~nan_mask].assign(
            **{label_column: transformed_label[~nan_mask].clip(-5.0, 5.0)}
        )
    else:
        # 硬截断，防止标准化后依然存在离群值干扰 MSE
        transformed_label = transformed_label.clip(-5.0, 5.0)
        # 使用 assign 创建新 DataFrame，共享未修改列的内存（避免 ~2 GiB 深拷贝）
        df_transformed = df.assign(**{label_column: transformed_label})

    return df_transformed

def generate_classification_labels(
    df: pd.DataFrame,
    label_column: str,
    pos_quantile: Optional[float] = None,
    pos_topk: Optional[int] = None,
) -> pd.DataFrame:
    """生成分类标签（TopN 正类）

    按每个交易日截面，将原始标签按分位阈值或数量阈值转为 0/1 标签。

    Args:
        df: 训练数据 DataFrame
        label_column: 原始标签列名
        pos_quantile: 百分比阈值（例如 0.2 表示 Top20% 为正类）
        pos_topk: 数量阈值（例如 300 表示每个交易日收益最高的 300 只为正类）

    Returns:
        添加了二分类标签的 DataFrame（新增列 {label_column}_binary）

    Note:
        pos_quantile 和 pos_topk 二选一，pos_topk 优先级更高
        使用 rank(method='first') 确保 topk 数量严格等于 k（打散并列）
    """
    logger.info(f"生成分类标签（基于 {label_column}）...")

    if pos_quantile is None and pos_topk is None:
        raise ValueError("必须指定 pos_quantile 或 pos_topk 之一")

    if pos_topk is not None and pos_quantile is not None:
        logger.warning("同时指定了 pos_topk 和 pos_quantile，使用 pos_topk（优先级更高）")

    df_labeled = df.copy()
    binary_label_col = f"{label_column}_binary"

    # 初始化标签列为 NaN
    df_labeled[binary_label_col] = np.nan

    # 按 trade_date 分组，对每组的标签进行排名
    df_labeled["_rank"] = df_labeled.groupby("trade_date")[label_column].rank(
        method="first", ascending=False, na_option="keep"
    )

    if pos_topk is not None:
        # 数量模式：Top K（排名 <= K 为正类）
        df_labeled[binary_label_col] = (df_labeled["_rank"] <= pos_topk).astype(float)
        df_labeled.loc[df_labeled["_rank"].isna(), binary_label_col] = np.nan
    else:
        # 百分比模式：Top X%
        valid_counts = df_labeled.groupby("trade_date")["_rank"].transform("count")
        threshold_ranks = (valid_counts * pos_quantile).clip(lower=1).astype(int)
        df_labeled[binary_label_col] = (df_labeled["_rank"] <= threshold_ranks).astype(float)
        df_labeled.loc[df_labeled["_rank"].isna(), binary_label_col] = np.nan

    # 删除临时排名列
    df_labeled = df_labeled.drop(columns=["_rank"])

    # 统计正类比例
    total_valid = df_labeled[binary_label_col].notna().sum()
    pos_count = df_labeled[binary_label_col].sum()
    pos_ratio = pos_count / total_valid if total_valid > 0 else 0

    logger.info(f"分类标签生成完成:")
    logger.info(
        f"  模式: {'pos_topk=' + str(pos_topk) if pos_topk else 'pos_quantile=' + str(pos_quantile)}"
    )
    logger.info(f"  正类样本数: {pos_count:.0f} / {total_valid:.0f} ({pos_ratio:.2%})")

    if pos_topk is not None:
        pos_counts_per_day = df_labeled.groupby("trade_date")[binary_label_col].sum()
        logger.debug(
            f"  各交易日正类数量统计: min={pos_counts_per_day.min():.0f}, max={pos_counts_per_day.max():.0f}, mean={pos_counts_per_day.mean():.1f}"
        )

    return df_labeled
