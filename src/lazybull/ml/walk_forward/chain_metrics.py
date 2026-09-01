"""Walk-forward 串联净值指标计算。"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def calculate_chain_metrics(
    chain_df: pd.DataFrame,
    annual_trading_days: int = 252,
    annual_risk_free_rate: float = 0.03,
) -> Dict[str, Optional[float]]:
    """按有效收益区间计算串联净值的全周期指标。"""
    empty: Dict[str, Optional[float]] = {
        "total_return": None,
        "cagr": None,
        "max_drawdown": None,
        "volatility": None,
        "sharpe": None,
        "trading_days": None,
    }
    if chain_df is None or chain_df.empty or "nav" not in chain_df.columns:
        return empty
    if annual_trading_days <= 0:
        raise ValueError("annual_trading_days 必须大于 0")
    if annual_risk_free_rate <= -1:
        raise ValueError("annual_risk_free_rate 必须大于 -1")

    columns = ["nav"]
    if "split_index" in chain_df.columns:
        columns.append("split_index")
    metrics_df = chain_df[columns].copy()
    metrics_df["nav"] = pd.to_numeric(metrics_df["nav"], errors="coerce")
    metrics_df = metrics_df[metrics_df["nav"].notna()].reset_index(drop=True)
    if metrics_df.empty or metrics_df["nav"].iloc[0] == 0:
        return empty

    nav = metrics_df["nav"]
    nav_ratio = nav.iloc[-1] / nav.iloc[0]
    total_return = nav_ratio - 1
    cumulative_max = nav.cummax()
    max_drawdown = ((nav - cumulative_max) / cumulative_max).min()

    daily_returns = nav.pct_change(fill_method=None)
    if "split_index" in metrics_df.columns:
        same_split = metrics_df["split_index"].eq(metrics_df["split_index"].shift())
        daily_returns = daily_returns.where(same_split)
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    trading_days = len(daily_returns)

    cagr = None
    if trading_days > 0 and nav_ratio > 0:
        cagr = nav_ratio ** (annual_trading_days / trading_days) - 1

    volatility = None
    sharpe = None
    if len(daily_returns) > 1:
        daily_std = daily_returns.std()
        if pd.notna(daily_std) and daily_std > 0:
            volatility = daily_std * np.sqrt(annual_trading_days)
            daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / annual_trading_days) - 1
            sharpe = (
                (daily_returns.mean() - daily_risk_free_rate)
                / daily_std
                * np.sqrt(annual_trading_days)
            )

    return {
        "total_return": float(total_return),
        "cagr": float(cagr) if cagr is not None else None,
        "max_drawdown": float(max_drawdown),
        "volatility": float(volatility) if volatility is not None else None,
        "sharpe": float(sharpe) if sharpe is not None else None,
        "trading_days": trading_days,
    }
