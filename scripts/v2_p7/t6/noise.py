# -*- coding: utf-8 -*-
"""P7 T6 噪声带：配对分块 bootstrap + 分年符号检验。

预登记口径（``calibration.BAND``，**运行前冻结、运行后不改**）
=========================================================
- **配对**：两臂用**同一组**重采样索引（配对差的自相关结构得以保留）；
- **块长** ``block=20`` = 冻结持期（配对差的自相关尺度）；敏感性 ``block ∈ {1, 250}`` 只报不判；
- ``B=4,000``、``seed=42``（重跑逐位一致，G5）；区间 = 2.5%/97.5% 分位。

与唯一入口的**同源口径**（本轮评审整改，R1-01 / R2-01）
==================================================
``calculate_chain_metrics`` 对**跨折日收益做了 ``same_split`` 掩蔽**（本窗口 1,745 个净值点 →
1,744 个日收益，其中 13 个折界日被丢弃 ⇒ ``trading_days=1,731``）。若带用「全部 1,744 个日收益」
重算，``cagr``/``volatility``/``sharpe`` 与点估计**不是同一估计量**（cagr 水平差约 0.13pp，
已超 δ_cagr）。故本模块按指标分两个重采样宇宙：

- **全序列宇宙**（1,744 个日收益）：``total_return`` / ``max_drawdown``（与入口一致）；
  ``cagr`` 的指数分子取**入口的交易日数**（``252/1,731``），与入口同式；
- **同折宇宙**（剔除 13 个折界日后 1,731 个）：``volatility`` / ``sharpe``（入口只在这些日收益上算）。

抽样机制（C6-02 整改）
==================
``calibration.BAND.method = paired_stationary_block_bootstrap``（预登记原文，**不得事后改**），
实现与之一致：逐位置以 ``1/block`` 概率重抽起点、否则沿当前块前进一日（越界环回），
块长服从均值为 ``block`` 的几何分布（``block=1`` 退化为 i.i.d.）。
全部向量化：``restart`` 判定 → 段起点位置 → 起点抽取 → ``(base + offset) % n``；
同一索引矩阵作用于两臂 ⇒ 配对。抽样机制的单测见
``tests/test_v2_p7_t6_frames_metrics.py::test_stationary_sampler_*``。

带宽对块长**非单调**（例如 volatility 带 block=1 < block=20），故不把任何单点带宽
当作方法无关常数；敏感性 ``block ∈ {1, 250}`` **只报不判**。

解释边界
========
本 run 的两臂是确定性模拟；带只描述「同一机制在行情路径变动下的抽样波动」，
是**可复算的裁定尺度**，不是概率论断，也不外推为前向/日历验证。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

NOISE_SCHEMA = "p7_t6_noise/v1"

#: 指标 → 重采样宇宙（``full`` = 全序列；``same_split`` = 剔除折界日）
METRIC_UNIVERSE: Mapping[str, str] = {
    "total_return": "full",
    "max_drawdown": "full",
    "cagr": "full",
    "volatility": "same_split",
    "sharpe": "same_split",
}

#: 参与噪声带的指标（与链式指标同名）
BAND_METRICS: Tuple[str, ...] = ("total_return", "max_drawdown", "cagr", "volatility", "sharpe")

#: 单块复制数（内存与速度折中；结果与分块大小无关）
CHUNK = 500

#: 与 ``calculate_chain_metrics`` 默认值一致（口径不另立）
ANNUAL_TRADING_DAYS = 252
ANNUAL_RISK_FREE_RATE = 0.03

#: 实际实现的方法（**不进入预登记摘要**；必须与预登记 ``band.method`` 同名）
METHOD_IMPLEMENTED = "paired_stationary_block_bootstrap"


def returns_from_nav(nav: Sequence[float]) -> np.ndarray:
    """日收益序列（``nav_t / nav_{t-1} − 1``；与链式指标同一差分方式）。"""
    values = np.asarray([float(item) for item in nav], dtype=float)
    if len(values) < 2:
        return np.zeros(0, dtype=float)
    return values[1:] / values[:-1] - 1.0


def same_split_mask(split_index: Sequence[int]) -> np.ndarray:
    """``calculate_chain_metrics`` 的 ``same_split`` 掩码（长度 = 日收益数）。

    ``True`` 表示该日收益与前一日同折（被入口计入 ``trading_days`` 与波动/夏普）。
    """
    values = np.asarray([int(item) for item in split_index], dtype=int)
    if len(values) < 2:
        return np.zeros(0, dtype=bool)
    return values[1:] == values[:-1]


def build_noise(
    *,
    nav: Mapping[str, Sequence[float]],
    split_index: Sequence[int],
    point_deltas: Mapping[str, float],
    calibration: Mapping[str, Any],
    deltas_by_year: Sequence[float],
    deltas_by_month: Sequence[float],
) -> Dict[str, Any]:
    """主带 + 敏感带 + 分年符号检验（全部可复算，且与唯一入口同源）。"""
    band = dict(calibration["band"])
    full_control = returns_from_nav(nav["vvh"])
    full_treatment = returns_from_nav(nav["vvs"])
    if len(full_control) != len(full_treatment) or len(full_control) < 2:
        raise ValueError("两臂日收益长度不一致或过短，无法配对 bootstrap")
    mask = same_split_mask(split_index)
    if len(mask) != len(full_control):
        raise ValueError("split_index 与日收益长度不一致，无法复刻折界掩蔽")
    same_control = full_control[mask]
    same_treatment = full_treatment[mask]
    if len(same_control) < 2:
        raise ValueError("同折日收益过少，无法估计波动/夏普带")
    universes = {
        "full": (full_control, full_treatment),
        "same_split": (same_control, same_treatment),
    }
    trading_days = int(len(same_control))
    deltas = dict(calibration.get("delta") or {})
    primary = _band_payload(
        universes=universes,
        block=int(band["block"]),
        replicates=int(band["replicates"]),
        seed=int(band["seed"]),
        interval=float(band["interval"]),
        points=point_deltas,
        trading_days=trading_days,
        deltas=deltas,
    )
    sensitivity: Dict[str, Any] = {}
    for block in band.get("sensitivity_blocks") or ():
        sensitivity[str(int(block))] = _band_payload(
            universes=universes,
            block=int(block),
            replicates=int(band["replicates"]),
            seed=int(band["seed"]),
            interval=float(band["interval"]),
            points=point_deltas,
            trading_days=trading_days,
            deltas=deltas,
        )
    return {
        "schema": NOISE_SCHEMA,
        "method": str(band["method"]),
        "method_implemented": METHOD_IMPLEMENTED,
        "method_matches_preregistration": bool(str(band["method"]) == METHOD_IMPLEMENTED),
        "sampler": {
            "kind": "stationary（几何块长，均值 = block，越界环回，两臂共用同一索引）",
            "restart_probability": 1.0 / float(band["block"]),
            "degenerates_to_iid_at_block_1": True,
            "vectorized": True,
        },
        "block": int(band["block"]),
        "replicates": int(band["replicates"]),
        "seed": int(band["seed"]),
        "interval": float(band["interval"]),
        "days": {
            "full": int(len(full_control)),
            "same_split": trading_days,
            "dropped_at_split_boundary": int(len(full_control) - trading_days),
        },
        "primary": primary,
        "sensitivity": sensitivity,
        "year_sign_test": sign_test(deltas_by_year, unit="year"),
        "month_sign_test": sign_test(deltas_by_month, unit="month"),
        "note": (
            "带只覆盖行情路径变动下的抽样波动；两臂为确定性模拟，越带判定还需同时满足"
            "经济实质阈值 δ（见 calibration.over_band_rule）。"
            "``cagr`` 的指数分子取入口交易日数（与 calculate_chain_metrics 同式）；"
            "``volatility``/``sharpe`` 只在同折日收益上算（复刻 same_split 掩蔽）。"
            "**区间以 boot_mean 为中心，不是点估计 point 的置信区间**；"
            "``bias = boot_mean − point``、``mde = 1.96·se``（正态近似的等价可判下限）；"
            "越带只看 95% 区间是否含 0，与 point 是否贴近 boot_mean 无关。"
            "敏感性 block∈{1,250} **只报不判**。"
        ),
        "definitions": {
            "point": "点估计（来自 metrics.paired，经唯一入口计算）",
            "boot_mean": "重采样分布均值（区间中心）",
            "bias": "boot_mean − point（重采样偏差，可正可负）",
            "mde": "1.96·se：正态近似的等价可判下限（与 δ 对比可见判据由哪条腿主导）",
            "mde_over_delta": "mde / δ（>1 表示统计腿主导，δ 事实上被架空）",
            "excludes_zero": "95% 区间不含 0（越带的统计腿）",
        },
    }


def _band_payload(
    *,
    universes: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    block: int,
    replicates: int,
    seed: int,
    interval: float,
    points: Mapping[str, float],
    trading_days: int,
    deltas: Mapping[str, float],
) -> Dict[str, Any]:
    """按宇宙分别重采样，再按指标合并（同一 ``block``/``seed``，复算逐位一致）。"""
    alpha = (1.0 - interval) / 2.0
    per_metric: Dict[str, Dict[str, np.ndarray]] = {}
    invalid: Dict[str, int] = {}
    for universe, metrics in _group_by_universe().items():
        control, treatment = universes[universe]
        draws, dropped = _boot_deltas(
            control=control,
            treatment=treatment,
            block=block,
            replicates=replicates,
            seed=seed,
            metrics=metrics,
            cagr_days=trading_days if "cagr" in metrics else None,
        )
        per_metric.update(draws)
        invalid.update(dropped)
    payload: Dict[str, Any] = {}
    for name in BAND_METRICS:
        sample = per_metric[name]
        low = float(np.percentile(sample, alpha * 100.0))
        high = float(np.percentile(sample, (1.0 - alpha) * 100.0))
        point = points.get(name)
        se = float(sample.std(ddof=1))
        delta = deltas.get(name)
        payload[name] = {
            "point": float(point) if point is not None else None,
            "boot_mean": float(sample.mean()),
            "bias": (float(sample.mean()) - float(point)) if point is not None else None,
            "low": low,
            "high": high,
            "se": se,
            "mde": float(1.96 * se),
            "delta": float(delta) if delta is not None else None,
            "mde_over_delta": (float(1.96 * se) / float(delta)) if delta else None,
            "excludes_zero": bool(low > 0.0 or high < 0.0),
            "band_width": float(high - low),
            "universe": METRIC_UNIVERSE[name],
            "invalid_replicates": int(invalid.get(name, 0)),
            "replicates": int(len(sample)),
        }
    return {
        "block": int(block),
        "replicates": int(replicates),
        "seed": int(seed),
        "trading_days": int(trading_days),
        "metrics": payload,
    }


def _group_by_universe() -> Dict[str, Tuple[str, ...]]:
    grouped: Dict[str, List[str]] = {}
    for name, universe in METRIC_UNIVERSE.items():
        grouped.setdefault(universe, []).append(name)
    return {key: tuple(values) for key, values in grouped.items()}


def _boot_deltas(
    *,
    control: np.ndarray,
    treatment: np.ndarray,
    block: int,
    replicates: int,
    seed: int,
    metrics: Sequence[str],
    cagr_days: int | None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """配对重采样：同一索引矩阵分别作用于两臂，返回 Δ 的复制分布（剔除非有限值）。"""
    if block < 1:
        raise ValueError(f"块长必须 >= 1：{block}")
    n = len(control)
    rng = np.random.default_rng(seed)
    chunks = max(1, math.ceil(replicates / CHUNK))
    collected: Dict[str, List[np.ndarray]] = {name: [] for name in metrics}
    invalid: Dict[str, int] = {name: 0 for name in metrics}
    remaining = replicates
    for _ in range(chunks):
        size = min(CHUNK, remaining)
        remaining -= size
        index = _stationary_index(rng=rng, size=size, days=n, block=block)
        left = _metrics_path(control[index], cagr_days=cagr_days)
        right = _metrics_path(treatment[index], cagr_days=cagr_days)
        for name in metrics:
            delta = np.asarray(right[name]) - np.asarray(left[name])
            finite = np.isfinite(delta)
            invalid[name] += int((~finite).sum())
            collected[name].append(delta[finite])
    return {
        name: np.concatenate(parts) if parts else np.zeros(0) for name, parts in collected.items()
    }, invalid


def _stationary_index(*, rng: np.random.Generator, size: int, days: int, block: int) -> np.ndarray:
    """配对 **stationary bootstrap** 索引（``block`` = 平均块长，块长服从几何分布）。

    逐位置以 ``1/block`` 概率重抽起点、否则沿当前块前进一日（越界环回），
    与预登记方法 ``paired_stationary_block_bootstrap`` 一致；``block=1`` 退化为 i.i.d.。
    同一索引矩阵作用于两臂 ⇒ 配对；全部向量化（不做逐日循环）。
    """
    if block < 1:
        raise ValueError(f"块长必须 >= 1：{block}")
    n = int(days)
    if n < 1:
        raise ValueError("重采样长度必须 >= 1")
    restart = rng.random((size, n)) < (1.0 / float(block))
    restart[:, 0] = True
    draws = rng.integers(0, n, size=(size, n))
    run_id_positions = np.where(restart, np.arange(n)[None, :], 0)
    np.maximum.accumulate(run_id_positions, axis=1, out=run_id_positions)
    offsets = np.arange(n)[None, :] - run_id_positions
    bases = np.take_along_axis(draws, run_id_positions, axis=1)
    return (bases + offsets) % n


def _metrics_path(returns: np.ndarray, *, cagr_days: int | None = None) -> Dict[str, np.ndarray]:
    """按路径复算五项指标（**与 ``calculate_chain_metrics`` 同式**）。

    ``total_return``/``max_drawdown`` 用整条路径；``cagr`` 的指数分子用 ``cagr_days``
    （缺省 = 该宇宙的日收益数）；``volatility``/``sharpe`` 用**简单收益**的样本标准差
    （ddof=1），由调用方在同折宇宙上调用以复刻 ``same_split`` 掩蔽。

    注意：重采样路径**沿用观测到的入口交易日数**（本窗口 1,731），不对重排后的路径
    重新判定折界（重排会打散折界、使掩码失去意义）。该近似对 Δcagr 的影响约 1e-6，
    远小于带宽 1e-3 量级，已在 ``docs/reports/p7_t6_closure_20261113.md`` 登记。
    """
    nav = np.cumprod(1.0 + returns, axis=1)
    initial = np.ones((nav.shape[0], 1))
    path = np.concatenate([initial, nav], axis=1)
    peak = np.maximum.accumulate(path, axis=1)
    drawdown = path / peak - 1.0
    total_return = nav[:, -1] - 1.0
    days = int(cagr_days) if cagr_days else returns.shape[1]
    cagr = np.power(1.0 + total_return, ANNUAL_TRADING_DAYS / days) - 1.0
    std = returns.std(axis=1, ddof=1)
    mean = returns.mean(axis=1)
    daily_rf = (1.0 + ANNUAL_RISK_FREE_RATE) ** (1.0 / ANNUAL_TRADING_DAYS) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        volatility = std * math.sqrt(ANNUAL_TRADING_DAYS)
        sharpe = (mean - daily_rf) / std * math.sqrt(ANNUAL_TRADING_DAYS)
    return {
        "total_return": total_return,
        "max_drawdown": drawdown.min(axis=1),
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
    }


def sign_test(deltas: Sequence[float], *, unit: str) -> Dict[str, Any]:
    """分单位符号检验（exact binomial，双侧；**精确为 0 的差值**剔除）。"""
    values = [float(item) for item in deltas]
    positive = [value for value in values if value > 0.0]
    negative = [value for value in values if value < 0.0]
    non_zero = len(positive) + len(negative)
    share = (max(len(positive), len(negative)) / non_zero) if non_zero else None
    return {
        "unit": unit,
        "observations": len(values),
        "positive": len(positive),
        "negative": len(negative),
        "non_zero": non_zero,
        "same_sign_share": share,
        "p_value": _binomial_two_sided(len(positive), non_zero),
        "note": "精确二项检验（p=0.5 双侧）；同一符号占比即稳健性判据的输入",
    }


def _binomial_two_sided(successes: int, trials: int) -> float:
    if trials <= 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(0, min(successes, trials - successes) + 1))
    return float(min(1.0, 2.0 * tail / (2**trials)))
