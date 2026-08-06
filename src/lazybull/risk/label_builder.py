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


def compute_forward_volatility(
    df: pd.DataFrame,
    horizon: int,
    ret_col: str = 'ret_1',
) -> pd.Series:
    """自动计算每只股票每个交易日的前向 N 日年化已实现波动率。

    算法（对齐 y_ret_N 的 T+1 买入约定）：
      - 先计算每只股票的滚动 N 日 std（窗口 [t-N+1, t]）
      - 再整体前移 N 位：T 日的前向波动率 = 窗口 [T+1, T+N] 的 std
      - 年化：× sqrt(252)

    与 forward_ret（close(T+1+N)/close(T+1)-1）使用同一窗口，保证对齐。

    Args:
        df: 含 ts_code, trade_date, ret_col 的 DataFrame（需按日期排序）
        horizon: 持有期天数 N
        ret_col: 日收益率列名

    Returns:
        Series（索引与 df 对齐），T 日前向 N 日年化波动率；窗口不足返回 NaN
    """
    if ret_col not in df.columns:
        logger.warning(f"缺少 {ret_col} 列，无法计算前向波动率")
        return pd.Series(np.nan, index=df.index)

    if horizon <= 0:
        logger.warning(f"持有期 horizon={horizon} 无效，返回 NaN")
        return pd.Series(np.nan, index=df.index)

    sorted_df = df.sort_values(['ts_code', 'trade_date'])
    min_periods = max(int(horizon * 0.8), 3)

    # 逐股滚动 N 日 std（向后窗口）
    roll_std = (
        sorted_df.groupby('ts_code')[ret_col]
        .rolling(horizon, min_periods=min_periods)
        .std()
        .reset_index(level=0, drop=True)
    )
    # 前移 N 位 → T 日取到窗口 [T+1, T+N]
    forward_std = roll_std.groupby(sorted_df['ts_code']).shift(-horizon)

    # 对齐回原 df 的索引顺序
    forward_vol = (forward_std * np.sqrt(252)).reindex(df.index)
    return forward_vol


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
    ret_1_col: str = 'ret_1',
) -> pd.DataFrame:
    """为特征 DataFrame 构造风控标签。

    前向波动率的获取优先级：
      1. forward_vol_col 指定的列（如果存在）
      2. 从 ret_1 自动计算前向已实现波动率（推荐，无需额外数据）
      3. 用 |forward_ret| 作为简化代理（仅当前两者都不可用）

    Args:
        features_df: 含 ts_code, trade_date, forward_ret_col 的 DataFrame
        forward_ret_col: 前向收益率列名（如 y_ret_10）
        forward_vol_col: 前向波动率列名（可选）
        holding_period: 持有期天数（标签窗口）
        ret_1_col: 日收益率列名（用于自动计算前向波动率）

    Returns:
        新增了 rar 和 label 两列的 DataFrame
    """
    df = features_df.copy()

    if forward_ret_col not in df.columns:
        logger.error(f"标签构造失败：缺少 {forward_ret_col} 列")
        return df

    forward_ret = df[forward_ret_col].astype(float)

    # 前向波动率：优先级 1 → 2 → 3
    if forward_vol_col and forward_vol_col in df.columns:
        logger.info(f"使用前向波动率列: {forward_vol_col}")
        forward_vol = df[forward_vol_col].astype(float)
    elif ret_1_col in df.columns:
        logger.info(
            f"未提供前向波动率列，从 {ret_1_col} 自动计算前向 {holding_period} 日"
            f"已实现波动率..."
        )
        forward_vol = compute_forward_volatility(
            df, horizon=holding_period, ret_col=ret_1_col
        )
    else:
        logger.warning(
            f"未找到前向波动率列 '{forward_vol_col}'，且缺少 {ret_1_col} 列，"
            f"使用 |{forward_ret_col}| 作为简化代理"
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
            forward_vol.loc[group.index],
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
