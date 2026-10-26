# -*- coding: utf-8 -*-
"""事件型 freshness 衰减与共享常量（训练侧与推理侧共享；P3-T2 上移，Q4-D / §16.1-A）。

训练侧：prepare_training_data 在 state_keep_event_decay 策略下，
对事件型因子按 freshness 做指数衰减并从特征列移除 freshness 列。

推理侧（MLSignal / OOS 评估）必须按模型训练参数复现同一衰减，
否则出现 train/serve skew：模型在压缩分布上学习，推理却用未衰减原值。

事件衰减契约（CLAUDE.md §5）：新增衰减逻辑只允许落在公共模块——本模块即
`state_keep_event_decay` 策略训练/推理同一实现的唯一落点（原
`ml/train_core/freshness.py` + `ml/train_core/constants.py:224-275` 共享子集上移；
旧模块改为 re-export 保持旧导入路径）。

common 层禁令（F14）：本模块不得 import 任何业务包（store/core/train/sleeves/factors/
universe/hosts/evidence/report/quality）；零项目内依赖。
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── freshness 策略名（推理共享常量）────────────────────────────
FRESHNESS_STRATEGY_DROP_ALL = "drop_all"

FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY = "state_keep_event_decay"

FRESHNESS_STRATEGY_STATE_KEEP_EVENT_NO_DECAY = "state_keep_event_no_decay"

DEFAULT_EVENT_FRESHNESS_HALF_LIFE_DAYS = 45.0

EVENT_FRESHNESS_TO_VALUE_COLUMNS = {
    "forecast_freshness_days": [
        "forecast_type_score",
        "forecast_chg_mid",
    ],
    "express_freshness_days": [
        "express_revenue_yoy",
        "express_profit_yoy",
        "express_roe",
        "express_surprise",
    ],
    "consensus_freshness_days": [
        "cons_analyst_count_30d",
        # 旧绝对值列仅供存量模型推理，新模型使用下方经济归一化列。
        "cons_eps_mean_fym1",
        "cons_eps_mean_fy0",
        "cons_eps_mean_fy1",
        "cons_eps_mean_fy2",
        "cons_eps_yield_fym1",
        "cons_eps_yield_fy0",
        "cons_eps_yield_fy1",
        "cons_eps_yield_fy2",
        "cons_eps_revision_30d",
        "cons_target_price_mid",
        "cons_target_upside",
        "cons_rating_score",
    ],
    "cons_revision_freshness_days": [
        "zscore_cons_eps_revision_accel",
        "zscore_cons_eps_dispersion",
        "zscore_cons_eps_dispersion_chg",
        "zscore_cons_target_upside_chg",
        "zscore_cons_analyst_count_chg",
        "zscore_cons_rating_upgrade_ratio",
    ],
}


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
