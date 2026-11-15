# -*- coding: utf-8 -*-
"""P4 冻结口径的收益、回撤与相关区间实现。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def nav_metrics(levels: pd.Series) -> dict:
    values = pd.to_numeric(levels, errors="coerce").dropna().astype(float)
    values = values.loc[values > 0]
    if len(values) < 2:
        raise ValueError("NAV 序列不足以计算指标")
    dates = pd.to_datetime(values.index, format="%Y%m%d")
    days = (dates[-1] - dates[0]).days
    total = float(values.iloc[-1] / values.iloc[0] - 1.0)
    cagr = float((values.iloc[-1] / values.iloc[0]) ** (365.25 / days) - 1.0) if days > 0 else None
    drawdown = values / values.cummax() - 1.0
    returns = values.pct_change().dropna()
    vol = float(returns.std(ddof=1) * math.sqrt(252)) if len(returns) > 1 else None
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else None
    )
    return {
        "start": str(values.index[0]),
        "end": str(values.index[-1]),
        "start_nav": float(values.iloc[0]),
        "end_nav": float(values.iloc[-1]),
        "total_return": total,
        "cagr": cagr,
        "max_drawdown": float(drawdown.min()),
        "annual_vol": vol,
        "sharpe_rf0": sharpe,
        "calendar_days": int(days),
        "observations": int(len(values)),
    }


def monthly_returns(levels: pd.Series, start_date: str) -> pd.Series:
    series = pd.to_numeric(levels, errors="coerce").dropna().sort_index()
    if start_date not in series.index:
        raise ValueError(f"月度起点 {start_date} 不在 NAV 序列")
    monthly = series.groupby(series.index.str.slice(0, 6)).last()
    result = monthly.pct_change()
    result.iloc[0] = float(monthly.iloc[0] / series.loc[start_date] - 1.0)
    result.index.name = "month"
    return result.dropna()


def yearly_returns(levels: pd.Series) -> dict[str, float]:
    series = pd.to_numeric(levels, errors="coerce").dropna().sort_index()
    year_end = series.groupby(series.index.str.slice(0, 4)).last()
    returns = year_end.pct_change()
    returns.iloc[0] = float(year_end.iloc[0] / series.iloc[0] - 1.0)
    return {str(key): float(value) for key, value in returns.items()}


def pearson(left: np.ndarray, right: np.ndarray) -> tuple[float | None, int]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None, int(mask.sum())
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None, int(mask.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(mask.sum())


def compound_blocks(values: np.ndarray, block: int) -> np.ndarray:
    result = []
    for start in range(0, len(values) - block + 1, block):
        chunk = values[start : start + block]  # noqa: E203
        if np.all(np.isfinite(chunk)):
            result.append(float(np.prod(1.0 + chunk) - 1.0))
    return np.asarray(result, dtype=float)


def block_bootstrap_upper(
    left: np.ndarray,
    right: np.ndarray,
    *,
    block: int,
    resamples: int = 10000,
    seed: int = 42,
    chunk: int = 500,
) -> dict:
    """Circular moving-block bootstrap，返回 Pearson 的 97.5% 分位。

    P3 冻结了 block 长度、重采样次数、seed 与分位；这里用确定性循环块
    （每次随机起点、连续 block 个观测、越界环绕），两臂共用索引。
    """
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) != len(y):
        raise ValueError("相关两臂长度不一致")
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return {
            "n": n,
            "point": None,
            "upper_975": None,
            "lower_025": None,
            "resamples": 0,
        }
    point, _ = pearson(x, y)
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(n / block))
    correlations: list[float] = []
    for _ in range(math.ceil(resamples / chunk)):
        size = min(chunk, resamples - len(correlations))
        starts = rng.integers(0, n, size=(size, blocks))
        offsets = np.arange(block)
        indices = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(size, -1)[:, :n]
        sx = x[indices]
        sy = y[indices]
        xm = sx - sx.mean(axis=1, keepdims=True)
        ym = sy - sy.mean(axis=1, keepdims=True)
        cov = (xm * ym).mean(axis=1)
        denom = np.sqrt((xm * xm).mean(axis=1) * (ym * ym).mean(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / denom
        correlations.extend(corr[np.isfinite(corr)].tolist())
    values = np.asarray(correlations, dtype=float)
    if len(values) == 0:
        return {"n": n, "point": point, "upper_975": None, "lower_025": None, "resamples": 0}
    return {
        "n": n,
        "point": point,
        "upper_975": float(np.quantile(values, 0.975)),
        "lower_025": float(np.quantile(values, 0.025)),
        "resamples": int(len(values)),
    }


def corr_line(
    left: pd.Series,
    right: pd.Series,
    *,
    block: int,
    min_observations: int,
) -> dict:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner")
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < min_observations:
        return {
            "n": int(len(joined)),
            "point": None,
            "upper_975": None,
            "lower_025": None,
            "resamples": 0,
            "estimable": False,
        }
    line = block_bootstrap_upper(
        joined["left"].to_numpy(),
        joined["right"].to_numpy(),
        block=block,
    )
    line["estimable"] = line["point"] is not None
    return line


def tail_attribution(
    a_only: pd.Series, combined: pd.Series, a_leg: pd.Series, candidate: pd.Series
) -> dict:
    """A-only 最差 5% 日的组合差及其按滞后权重的腿贡献。

    组合日收益恒等式：``combined_ret = w_a*a_leg_ret + w_c*candidate_ret``，
    权重取前一交易日实际市值占比；两腿贡献相加减去 A-only 即组合差。
    """
    levels = (
        pd.concat(
            [
                a_only.rename("a_only"),
                combined.rename("combined"),
                a_leg.rename("a_leg"),
                candidate.rename("candidate"),
            ],
            axis=1,
            join="inner",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(levels) < 2:
        return {"days": 0}
    returns = levels.pct_change()
    a_weight = (levels["a_leg"] / levels["combined"]).shift(1)
    c_weight = (levels["candidate"] / levels["combined"]).shift(1)
    a_contrib = a_weight * returns["a_leg"] - returns["a_only"]
    c_contrib = c_weight * returns["candidate"]
    diff = returns["combined"] - returns["a_only"]
    joined = (
        pd.concat(
            [
                returns["a_only"].rename("a_only_ret"),
                diff.rename("diff"),
                a_contrib.rename("a_contrib"),
                c_contrib.rename("c_contrib"),
            ],
            axis=1,
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    threshold = joined["a_only_ret"].quantile(0.05)
    tail = joined.loc[joined["a_only_ret"] <= threshold]
    return {
        "days": int(len(tail)),
        "worst_5pct_threshold": float(threshold),
        "sum_combined_minus_a_only": float(tail["diff"].sum()),
        "sum_a_leg_contribution": float(tail["a_contrib"].sum()),
        "sum_candidate_contribution": float(tail["c_contrib"].sum()),
        "sum_contribution_residual": float(
            tail["diff"].sum() - tail["a_contrib"].sum() - tail["c_contrib"].sum()
        ),
    }
