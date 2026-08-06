"""衍生风控因子（由其他特征列构造）

这三个因子是纯函数衍生，需要在训练和推理时都存在（cs_train/cs_infer 一致），
因此作为注册因子由 builder 在特征构建阶段统一生成。

  - momentum_decay      : 动量衰减率 = ret_5 / |ret_20|，接近 1=稳定，趋向 0=衰竭
  - earnings_yield      : 盈利收益率 = 1 / pe_ttm（PE<=0 时置 NaN）
  - ret_volatility_ratio: 收益波动比 = ret_20 / volatility_20（类 Sharpe）
"""

import numpy as np
import pandas as pd

from .factor_registry import register_risk_factor

_EPS = 1e-8


@register_risk_factor("momentum_decay")
def compute_momentum_decay(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """动量衰减率：ret_5 / (|ret_20| + eps)。

    接近 1 表示 5 日动量与 20 日动量一致（趋势稳定）；
    趋向 0 表示短期动量远弱于中期（动量在衰竭）。
    """
    if 'ret_5' not in df.columns or 'ret_20' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df['ret_5'] / (df['ret_20'].abs() + _EPS)


@register_risk_factor("earnings_yield")
def compute_earnings_yield(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """盈利收益率：1 / pe_ttm。PE<=0 或缺失时置 NaN（无意义的负/零估值）。"""
    if 'pe_ttm' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    pe = df['pe_ttm'].astype(float)
    result = np.where((pe > 0) & pe.notna(), 1.0 / pe, np.nan)
    return pd.Series(result, index=df.index)


@register_risk_factor("ret_volatility_ratio")
def compute_ret_volatility_ratio(
    df: pd.DataFrame,
    daily_adj: pd.DataFrame = None,
    market_state: dict = None,
    **kwargs,
) -> pd.Series:
    """收益波动比：ret_20 / (volatility_20 + eps)。

    衡量单位波动换来的中期收益（类 Sharpe），高值表示收益质量好。
    """
    vol_col = 'volatility_20'
    if 'ret_20' not in df.columns or vol_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df['ret_20'] / (df[vol_col] + _EPS)
