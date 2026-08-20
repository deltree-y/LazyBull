# -*- coding: utf-8 -*-
"""事件型 freshness 衰减（训练侧与推理侧共享）。

训练侧：prepare_training_data 在 state_keep_event_decay 策略下，
对事件型因子按 freshness 做指数衰减并从特征列移除 freshness 列。

推理侧（MLSignal / OOS 评估）必须按模型训练参数复现同一衰减，
否则出现 train/serve skew：模型在压缩分布上学习，推理却用未衰减原值。
"""

from typing import Dict, List, Tuple

import math

import numpy as np
import pandas as pd

from .constants import (
    EVENT_FRESHNESS_TO_VALUE_COLUMNS,
    FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY,
)


def apply_event_freshness_decay(
    df: pd.DataFrame,
    event_freshness_cols: List[str],
    half_life_days: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """按 freshness 对事件型因子做指数衰减（半衰期）。

    权重: w = exp(-ln(2) * days / half_life)
    - freshness 缺失时权重按 1.0 处理（避免无谓引入缺失）
    - freshness < 0 按 0 处理

    Args:
        df: 特征 DataFrame（原地修改值列并返回）。
        event_freshness_cols: 事件型 freshness 列名列表。
        half_life_days: 衰减半衰期（天）。

    Returns:
        (df, decay_stats)：df 为衰减后的 DataFrame，decay_stats 为
        {freshness_col: 受影响样本数}。
    """
    if half_life_days <= 0:
        raise ValueError("event_freshness_half_life_days 必须 > 0")

    decay_stats: Dict[str, int] = {}
    if len(df) == 0 or not event_freshness_cols:
        return df, decay_stats

    ln2 = math.log(2.0)
    for freshness_col in event_freshness_cols:
        if freshness_col not in df.columns:
            continue
        value_cols = [
            col
            for col in EVENT_FRESHNESS_TO_VALUE_COLUMNS.get(freshness_col, [])
            if col in df.columns
        ]
        if not value_cols:
            continue

        freshness = pd.to_numeric(df[freshness_col], errors="coerce")
        decay_weight = np.exp(-ln2 * freshness.clip(lower=0) / float(half_life_days))
        decay_weight = pd.Series(decay_weight, index=df.index).fillna(1.0)

        touched = 0
        for value_col in value_cols:
            raw = pd.to_numeric(df[value_col], errors="coerce")
            before_non_na = int(raw.notna().sum())
            df[value_col] = raw * decay_weight
            touched += before_non_na

        decay_stats[freshness_col] = touched

    return df, decay_stats


def apply_serving_event_decay(
    df: pd.DataFrame,
    freshness_strategy: str,
    event_freshness_half_life_days: float,
) -> pd.DataFrame:
    """推理侧复现训练时的事件型 freshness 衰减（消除 train/serve skew）。

    仅 state_keep_event_decay 策略衰减；其他策略（no_decay/drop_all）原样返回，
    与训练侧行为一致（no_decay/drop_all 训练时同样不衰减）。

    Args:
        df: 特征 DataFrame（原地修改值列并返回）。
        freshness_strategy: 模型训练时的 freshness 策略。
        event_freshness_half_life_days: 模型训练时的衰减半衰期（天）。

    Returns:
        衰减后的 DataFrame。
    """
    if freshness_strategy != FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY:
        return df
    event_freshness_cols = [c for c in EVENT_FRESHNESS_TO_VALUE_COLUMNS if c in df.columns]
    if not event_freshness_cols:
        return df
    decayed, _ = apply_event_freshness_decay(
        df,
        event_freshness_cols=event_freshness_cols,
        half_life_days=float(event_freshness_half_life_days),
    )
    return decayed
