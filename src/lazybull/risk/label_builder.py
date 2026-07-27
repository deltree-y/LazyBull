"""风控模型标签构造

根据前向波动率调整收益 (RAR) 的截面三分位数，为每只股票生成三分类标签。

标签定义：
  class 0 (REDUCE)   — RAR 在下三分之一 → 建议减仓 (系数 0.5x)
  class 1 (HOLD)     — RAR 在中三分之一 → 维持不变 (系数 1.0x)
  class 2 (INCREASE) — RAR 在上三分之一 → 可加仓 (系数 1.5x)

时间对齐：标签基于 T+1 到 T+1+N 的前向窗口计算，与现有的 y_ret_N 约定一致。
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

_EPS = 1e-6

# 类别常量
CLASS_REDUCE = 0
CLASS_HOLD = 1
CLASS_INCREASE = 2

CLASS_LABELS = {
    CLASS_REDUCE: "REDUCE",
    CLASS_HOLD: "HOLD",
    CLASS_INCREASE: "INCREASE",
}

COEFFICIENT_MAP = {
    CLASS_REDUCE: 0.5,
    CLASS_HOLD: 1.0,
    CLASS_INCREASE: 1.5,
}


def compute_rar(
    forward_ret: pd.Series,
    forward_vol: pd.Series,
) -> pd.Series:
    """计算波动率调整收益 RAR。

    Args:
        forward_ret: 前向 N 日收益率
        forward_vol: 前向 N 日年化波动率

    Returns:
        RAR = forward_ret / forward_vol
    """
    return forward_ret / (forward_vol + _EPS)


def compute_labels_from_rar(
    rar: pd.Series,
) -> pd.Series:
    """根据 RAR 的截面三分位数生成标签。

    Args:
        rar: 同交易日所有股票的 RAR 序列

    Returns:
        标签序列 (0/1/2)，长度与 rar 一致
    """
    valid = rar.dropna()
    if len(valid) < 30:
        return pd.Series(np.nan, index=rar.index)

    q_low = valid.quantile(1 / 3)
    q_high = valid.quantile(2 / 3)

    labels = pd.Series(CLASS_HOLD, index=rar.index)
    labels[rar <= q_low] = CLASS_REDUCE
    labels[rar >= q_high] = CLASS_INCREASE
    labels[rar.isna()] = np.nan

    return labels.astype('Int64')  # nullable integer


def build_position_risk_labels(
    features_df: pd.DataFrame,
    forward_ret_col: str = 'y_ret_10',
    forward_vol_col: Optional[str] = None,
    holding_period: int = 10,
) -> pd.DataFrame:
    """为特征 DataFrame 构造风控标签。

    如果 forward_vol_col 未指定，则尝试从 daily_adj 计算前向波动率。
    如果不可用，则用 |forward_ret| 作为简化的波动率代理（不推荐，仅做 fallback）。

    Args:
        features_df: 含 ts_code, trade_date 和 forward_ret_col 的 DataFrame
        forward_ret_col: 前向收益率列名（如 y_ret_10）
        forward_vol_col: 前向波动率列名（可选）
        holding_period: 持有期天数（用于日志）

    Returns:
        新增了 rar 和 label 两列的 DataFrame
    """
    df = features_df.copy()

    if forward_ret_col not in df.columns:
        logger.error(f"标签构造失败：缺少 {forward_ret_col} 列")
        return df

    forward_ret = df[forward_ret_col].astype(float)

    # 前向波动率
    if forward_vol_col and forward_vol_col in df.columns:
        forward_vol = df[forward_vol_col].astype(float)
    else:
        # Fallback：用 |ret| 作为简化代理
        logger.warning(
            f"未找到前向波动率列 '{forward_vol_col}'，"
            f"使用 |{forward_ret_col}| 作为简化代理（建议提供实际波动率）"
        )
        forward_vol = forward_ret.abs()

    # 过滤无效样本
    valid_mask = forward_ret.notna() & forward_vol.notna() & (forward_vol > _EPS)
    n_total = len(df)
    n_valid = valid_mask.sum()
    if n_valid < 30:
        logger.warning(f"有效标签样本过少 ({n_valid}/{n_total})，跳过标签构造")
        df['rar'] = np.nan
        df['label'] = np.nan
        return df

    logger.info(f"标签构造：{n_valid}/{n_total} 有效样本，持有期={holding_period}天")

    # 按日期分组计算截面 RAR 和标签
    df['rar'] = np.nan
    df['label'] = np.nan

    for trade_date, group in df[valid_mask].groupby('trade_date'):
        rar = compute_rar(
            group[forward_ret_col],
            group[forward_vol_col] if forward_vol_col and forward_vol_col in df.columns
            else group[forward_ret_col].abs(),
        )
        labels = compute_labels_from_rar(rar)
        df.loc[group.index, 'rar'] = rar.values
        df.loc[group.index, 'label'] = labels.values

    # 统计
    label_counts = df['label'].value_counts().to_dict()
    logger.info(
        f"标签分布: REDUCE={label_counts.get(0, 0)}, "
        f"HOLD={label_counts.get(1, 0)}, "
        f"INCREASE={label_counts.get(2, 0)}"
    )

    return df


def validate_label_quality(df: pd.DataFrame, forward_ret_col: str = 'y_ret_10') -> dict:
    """验证标签质量：检查三类 forward return 是否单调递增。

    Returns:
        {'monotonic': bool, 'reduce_mean': float, 'hold_mean': float, 'increase_mean': float}
    """
    if 'label' not in df.columns:
        return {'monotonic': False, 'error': 'no label column'}

    metrics = {}
    for cls, name in CLASS_LABELS.items():
        mask = df['label'] == cls
        if mask.sum() > 0:
            metrics[f'{name.lower()}_mean'] = df.loc[mask, forward_ret_col].mean()
            metrics[f'{name.lower()}_count'] = int(mask.sum())
        else:
            metrics[f'{name.lower()}_mean'] = np.nan
            metrics[f'{name.lower()}_count'] = 0

    reduce_mean = metrics.get('reduce_mean', np.nan)
    hold_mean = metrics.get('hold_mean', np.nan)
    increase_mean = metrics.get('increase_mean', np.nan)

    monotonic = False
    if not np.isnan(reduce_mean) and not np.isnan(hold_mean) and not np.isnan(increase_mean):
        monotonic = (reduce_mean < hold_mean < increase_mean)

    metrics['monotonic'] = monotonic
    return metrics
