# -*- coding: utf-8 -*-
"""市场调整收益、分块相关矩阵与冻结同伴网络（协议 §3）。

数值口径：``rho`` 与分数排序键均按十进制 12 位五成双；共同样本不足、
方差 ``<=1e-12``、``rho<=0`` 一律不可建边，缺失绝不填 0 当完整样本。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import spec
from .common import ResearchError, round12


def build_excess_returns(
    close_adj: pd.DataFrame,
    eligible: pd.DataFrame,
    suspended: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """日对数收益、市场调整收益 ``e`` 与市场参照 ``m``。

    ``close_adj`` / ``eligible`` / ``suspended`` 必须同索引、同列、按开市日排序。
    """
    if not close_adj.index.equals(eligible.index) or not close_adj.columns.equals(
        eligible.columns
    ):
        raise ResearchError("复权价与资格帧轴不一致")
    positive = close_adj.where(close_adj > 0)
    prev = positive.shift(1)
    valid = positive.notna() & prev.notna() & eligible.fillna(False)
    valid &= ~suspended.fillna(True)
    valid &= ~suspended.fillna(True).shift(1).fillna(True)
    returns = np.log(positive / prev).where(valid)
    counts = returns.notna().sum(axis=1)
    median = returns.median(axis=1, skipna=True)
    median = median.where(counts >= spec.BREADTH_MIN)
    excess = returns.sub(median, axis=0)
    meta = {
        "days": int(len(close_adj)),
        "cells": int(valid.size),
        "return_cells": int(returns.notna().sum().sum()),
        "median_missing_days": int(median.isna().sum()),
        "min_daily_breadth": int(counts.min()) if len(counts) else 0,
        "breadth_below_100": int((counts < spec.BREADTH_MIN).sum()),
    }
    return excess, median, meta


def _pairwise_blocked(excess: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """矩阵乘版本：只在共同有限样本上算均值、方差与协方差。

    ``m*filled`` 把两侧掩码相乘，使 ``sum_i[i,j]`` 恰为 i 在 (i,j) 共同样本上的和；
    缺失一律以掩码排除，绝不填 0 后当完整样本。
    """
    values = excess.to_numpy(dtype=float)
    mask = np.isfinite(values)
    filled = np.where(mask, values, 0.0)
    m = mask.astype(float)
    count = m.T @ m
    mask_value = m * filled
    sum_i = mask_value.T @ m
    sum_j = sum_i.T
    sumsq_i = (mask_value * filled).T @ m
    cross = filled.T @ filled
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_i = sum_i / count
        mean_j = sum_j / count
        var_i = sumsq_i / count - mean_i * mean_i
        var_j = var_i.T
        cov = cross / count - mean_i * mean_j
        rho = cov / np.sqrt(var_i * var_j)
        degenerate = (var_i <= spec.VARIANCE_FLOOR) | (var_j <= spec.VARIANCE_FLOOR)
        rho = np.where(degenerate, np.nan, rho)
    codes = list(excess.columns)
    return (
        pd.DataFrame(count, index=codes, columns=codes),
        pd.DataFrame(rho, index=codes, columns=codes),
    )


def _pairwise_bruteforce(excess: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐对金标准（测试用）：只在共同有限样本上算 Pearson。"""
    codes = list(excess.columns)
    values = excess.to_numpy(dtype=float)
    size = len(codes)
    count = np.zeros((size, size), dtype=float)
    rho = np.full((size, size), np.nan, dtype=float)
    for i in range(size):
        left = values[:, i]
        for j in range(i, size):
            right = values[:, j]
            pair = np.isfinite(left) & np.isfinite(right)
            n = int(pair.sum())
            count[i, j] = count[j, i] = float(n)
            if n < 2:
                continue
            a = left[pair]
            b = right[pair]
            va = float(np.var(a, ddof=0))
            vb = float(np.var(b, ddof=0))
            if va <= spec.VARIANCE_FLOOR or vb <= spec.VARIANCE_FLOOR:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            rho[i, j] = rho[j, i] = corr
    return (
        pd.DataFrame(count, index=codes, columns=codes),
        pd.DataFrame(rho, index=codes, columns=codes),
    )


def pairwise_correlation(
    excess: pd.DataFrame, *, brute_force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if brute_force:
        return _pairwise_bruteforce(excess)
    return _pairwise_blocked(excess)


def select_peers(
    rho_row: pd.Series, count_row: pd.Series, center: str
) -> list[dict[str, Any]]:
    """前 10 个正相关邻居；严格并列按代码升序。"""
    candidates = []
    for code, value in rho_row.items():
        if code == center:
            continue
        if not np.isfinite(value) or float(value) <= 0.0:
            continue
        if float(count_row.get(code, 0.0)) < spec.CORR_MIN_PAIRS:
            continue
        candidates.append((round12(float(value)), str(code)))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [
        {
            "center": center,
            "peer": code,
            "rank": rank,
            "rho": value,
            "pair_n": int(count_row.get(code, 0)),
        }
        for rank, (value, code) in enumerate(candidates[: spec.TOP_PEERS], start=1)
    ]


def fit_beta(
    excess: pd.DataFrame, center: str, peers: Sequence[str]
) -> dict[str, Any]:
    """带截距 OLS 的斜率；x 为至少 8 个有限同伴的等权均值。"""
    peer_frame = excess[list(peers)]
    valid_peers = peer_frame.notna().sum(axis=1)
    peer_mean = peer_frame.mean(axis=1, skipna=True).where(
        valid_peers >= spec.PEER_MIN_VALID
    )
    pair = pd.concat(
        [excess[center].rename("target"), peer_mean.rename("peer")], axis=1
    ).dropna()
    n = int(len(pair))
    record: dict[str, Any] = {
        "center": center,
        "pair_n": n,
        "beta": float("nan"),
        "beta_n": float("nan"),
        "peer_valid": True,
        "reason": "",
    }
    if n < spec.BETA_MIN_PAIRS:
        record["reason"] = f"pair_n={n}<{spec.BETA_MIN_PAIRS}"
        record["peer_valid"] = False
        return record
    x = pair["peer"].to_numpy(dtype=float)
    y = pair["target"].to_numpy(dtype=float)
    var_x = float(np.var(x, ddof=0))
    if var_x <= spec.VARIANCE_FLOOR:
        record["reason"] = "peer_variance_floor"
        record["peer_valid"] = False
        return record
    beta = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    if not np.isfinite(beta) or beta <= 0.0:
        record["reason"] = "beta_non_positive"
        record["peer_valid"] = False
        return record
    record["beta"] = round12(beta)
    record["beta_n"] = round12(beta)
    return record


def formation(
    excess: pd.DataFrame, codes: Sequence[str], *, brute_force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """一个月末形成日：建图 + beta。返回边表、beta 表与诊断。"""
    subset = excess[list(codes)]
    count, rho = pairwise_correlation(subset, brute_force=brute_force)
    edges: list[dict[str, Any]] = []
    betas: list[dict[str, Any]] = []
    no_network = 0
    for center in subset.columns:
        peers = select_peers(rho[center], count[center], center)
        if len(peers) < spec.TOP_PEERS:
            no_network += 1
            betas.append(
                {
                    "center": center,
                    "pair_n": 0,
                    "beta": float("nan"),
                    "beta_n": float("nan"),
                    "peer_valid": False,
                    "reason": f"peers={len(peers)}<{spec.TOP_PEERS}",
                }
            )
            continue
        edges.extend(peers)
        betas.append(fit_beta(subset, center, [item["peer"] for item in peers]))
    edge_frame = pd.DataFrame(edges)
    beta_frame = pd.DataFrame(betas)
    meta = {
        "centers": int(len(subset.columns)),
        "edges": int(len(edge_frame)),
        "no_network": int(no_network),
        "valid_beta": int(beta_frame["beta"].notna().sum()) if len(beta_frame) else 0,
        "cross_industry_ratio": None,
    }
    return edge_frame, beta_frame, meta


def market_daily_frame(median: pd.Series) -> pd.DataFrame:
    frame = median.rename("m").reset_index()
    frame.columns = ["trade_date", "m"]
    frame["m_valid"] = frame["m"].notna()
    return frame
