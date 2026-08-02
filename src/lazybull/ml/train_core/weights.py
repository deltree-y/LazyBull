# -*- coding: utf-8 -*-
"""weights：train_core 拆分模块。"""

from loguru import logger
import numpy as np
import pandas as pd


def build_time_decay_weights(
    df_train: pd.DataFrame, half_life_years: float = 1.0, date_col: str = "trade_date"
) -> np.ndarray:
    """按时间衰减构造训练样本权重

    越近的样本权重越高，使模型更重视近期市场模式。
    使用指数衰减：weight = 0.5 ^ (distance_years / half_life_years)

    - half_life_years=1.0 → 1年前的样本权重为0.5，2年前为0.25
    - half_life_years=2.0 → 2年前的样本权重为0.5，4年前为0.25（衰减更慢）

    Args:
        df_train: 训练集 DataFrame，需包含 trade_date 列
        half_life_years: 半衰期（年），即权重衰减到 0.5 所需的年数
        date_col: 日期列名

    Returns:
        与 df_train 行数相同的 numpy 数组，包含每个样本的权重（最新样本=1.0）
    """
    if date_col not in df_train.columns:
        logger.warning(f"日期列 {date_col} 不存在，返回全为1的权重")
        return np.ones(len(df_train), dtype=float)

    dates = pd.to_datetime(df_train[date_col].astype(str))
    max_date = dates.max()

    # 距最新日期的交易日数（用实际行数近似，避免需要交易日历）
    # 按唯一日期排序，给每个日期赋予序号距离
    unique_dates = sorted(dates.unique())
    date_to_rank = {d: i for i, d in enumerate(unique_dates)}
    total_days = len(unique_dates)

    ranks = dates.map(date_to_rank).values.astype(float)
    # distance: 距最新日期的交易日距离（最新=0，最旧=total_days-1）
    distance = (total_days - 1) - ranks
    distance_years = distance / 252.0

    weights = np.power(0.5, distance_years / half_life_years)

    logger.info(
        f"时间衰减权重构造完成: half_life={half_life_years}y, "
        f"训练跨度={total_days}交易日(≈{total_days/252:.1f}y), "
        f"最旧样本权重={weights.min():.4f}, 最新样本权重={weights.max():.4f}"
    )
    return weights

def build_rank_sample_weights(
    df_train: pd.DataFrame,
    label_column: str,
    topk: int = 30,
    top_weight: float = 5.0,
    topk_weight_mode: str = "linear_decay",
    date_col: str = "trade_date",
) -> np.ndarray:
    """按日截面排名构造训练样本权重

    对训练集按每个交易日截面排序：
    - Top K 使用可配置模式赋权（默认 linear_decay）：第1名=top_weight，递减到第K名=2.0。
    - Bottom K 使用同样规则赋权：最差样本=top_weight，递减到第K名=2.0。

    处理规则：
    - 若某日样本数 <= topk，则该日样本全部设为 top_weight（避免退化时完全无区分）。
    - 若某日样本数 > topk，按标签列升序排序后同时取 Top K（最大值）与 Bottom K（最小值）赋权。
    - 排名依据：标签列在当日截面内的值（升序，头部为 Bottom，尾部为 Top）。

    Args:
        df_train: 训练集 DataFrame，需包含 trade_date 列和标签列
        label_column: 排名所用标签列名（如 neu_y_ret_20）
        topk: 每日 Top/Bottom 各取前 K 个样本，默认 30
        top_weight: Top1/Bottom1 样本权重上限，默认 5.0
        topk_weight_mode: Top/Bottom 赋权模式，支持 linear_decay|flat，默认 linear_decay
        date_col: 日期列名，默认 trade_date

    Returns:
        与 df_train 行数相同的 numpy 数组，包含每个样本的权重
    """
    weights = np.ones(len(df_train), dtype=float)

    if label_column not in df_train.columns:
        logger.warning(f"标签列 {label_column} 不存在，返回全为1的权重")
        return weights

    if date_col not in df_train.columns:
        logger.warning(f"日期列 {date_col} 不存在，返回全为1的权重")
        return weights

    effective_mode = (topk_weight_mode or "linear_decay").strip().lower()
    if effective_mode not in {"linear_decay", "flat"}:
        logger.warning(
            f"未知 topk_weight_mode={topk_weight_mode}，回退到 linear_decay"
        )
        effective_mode = "linear_decay"

    linear_floor_weight = 2.0

    # 按日截面处理
    for date, grp_idx in df_train.groupby(date_col).groups.items():
        grp = df_train.loc[grp_idx, label_column].dropna()
        n = len(grp)
        if n == 0:
            continue

        if n <= topk:
            # 样本数不足时，整组都赋予 top_weight（退化处理）
            positions = df_train.index.get_indexer_for(grp_idx)
            valid_positions = positions[positions >= 0]
            weights[valid_positions] = top_weight
            continue

        # 排序取 Top/Bottom K
        sorted_vals = grp.sort_values()
        bottom_k_idx = sorted_vals.iloc[:topk].index
        top_k_idx = sorted_vals.iloc[-topk:].index

        if effective_mode == "flat":
            top_positions = df_train.index.get_indexer_for(top_k_idx)
            bottom_positions = df_train.index.get_indexer_for(bottom_k_idx)
            weights[top_positions[top_positions >= 0]] = top_weight
            weights[bottom_positions[bottom_positions >= 0]] = top_weight
        else:
            # linear_decay：Top/Bottom 第1名=top_weight，逐步衰减到第K名=2
            top_k_ranked_desc = sorted_vals.iloc[-topk:].sort_values(ascending=False)
            bottom_k_ranked_asc = sorted_vals.iloc[:topk].sort_values(ascending=True)
            denom = max(topk - 1, 1)

            def _assign_decay_weight(sample_idx, rank_idx: int) -> None:
                position = df_train.index.get_indexer_for([sample_idx])
                valid = position[position >= 0]
                if len(valid) == 0:
                    return
                if topk == 1:
                    decay_weight = top_weight
                else:
                    decay_weight = linear_floor_weight + (top_weight - linear_floor_weight) * (
                        (topk - 1 - rank_idx) / denom
                    )
                weights[valid] = np.maximum(weights[valid], decay_weight)

            for rank_idx, sample_idx in enumerate(top_k_ranked_desc.index):
                _assign_decay_weight(sample_idx, rank_idx)

            for rank_idx, sample_idx in enumerate(bottom_k_ranked_asc.index):
                _assign_decay_weight(sample_idx, rank_idx)

    top_weighted_count = int((weights > 1.0).sum())
    logger.info(
        f"样本权重构造完成: Top/Bottom {topk} 增强，"
        f"模式={effective_mode}，"
        f"线性下限={linear_floor_weight}，"
        f"加权样本数={top_weighted_count}，权重上限={top_weight}，"
        f"普通样本数={len(weights) - top_weighted_count}"
    )
    return weights
