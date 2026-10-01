# -*- coding: utf-8 -*-
"""功效标定（P5a-1：证据机器的「最小刻度表」；方案 §3.5）。

**构造规则（v2 修订，P0 评审 A1）**：合成臂必须携带**真实量级的 Δ 波动**——
裁决的困难从来不在点估计偏移而在 Δ 分布宽度，纯等值平移会把宽度构造没了
（实测自证矛盾：+1pp 等值平移检出概率 ≈100%，而真实 B0 vs B1 的 +1.78pp 不可区分 P=0.78）。
- **首选**：`arm = B0 + Δ_noise + δ`，其中 Δ_noise = 真实换种子批的逐日差
  （B0′(seeds 43,62,83) − B0）；检出功效 = 真实噪声宽度下 δ 的检出概率，
  这才是「尺子的真实刻度」。
- **备选**（换种子批不可得时）：block bootstrap 从 B0 自身构造 Δ_t 噪声，
  登记为代理口径（报告标注 `noise_source="block_bootstrap_proxy"`）。

产出：合成臂 +δ（年化 pp）→ 配对制度重排口径下的**检出概率**曲线。
检出判据覆盖**双指标**（P0 评审 A1 修订：北极星主判据是 ΔMaxDD，标定不得只标 ΔCAGR）：
- ΔCAGR 95% 区间下界 > 0；
- ΔMaxDD 95% 区间下界 > 0（MaxDD 改善 = Δ > 0 方向）。
所有分布结论标注有效样本量（重排 N 次 ≠ N 个独立制度，深尾分位有效自由度 ≈ 折数级）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts.compare.fold_subset import (
    RunArtifacts,
    chain_metrics_from_fold_returns,
    per_fold_returns,
)

# 折内日收益的年度化因子（与 chain_metrics 口径一致：252 交易日）
_ANNUAL_DAYS = 252


def shift_fold_returns_annual_pp(
    fold_returns: Dict[int, np.ndarray], annual_pp: float
) -> Dict[int, np.ndarray]:
    """折内日收益等值平移：每折每交易日收益加同一日度偏移。

    注意（P0 评审 A1）：单独使用此平移构造的合成臂 **Δ 波动为 0**，系统性高估检出
    功效；仅作为「加噪声后叠加水平偏移」的一部分，不得单独用于功效标定。

    Args:
        fold_returns: 折号 → 折内日收益序列
        annual_pp: 目标抬升的年化百分点（如 +2.0 表示 +2pp 年化）

    Returns:
        平移后的折收益字典（新数组，不改原值）。
    """
    daily_delta = (1.0 + annual_pp / 100.0) ** (1.0 / _ANNUAL_DAYS) - 1.0
    return {
        fold: (rets + daily_delta) if rets.size > 0 else rets
        for fold, rets in fold_returns.items()
    }


def _fold_diff_pool(
    base_folds: Dict[int, np.ndarray], alt_folds: Dict[int, np.ndarray]
) -> Dict[int, np.ndarray]:
    """真实换种子噪声池：折内逐日差（alt − base）。逐折对齐，按短者截断。"""
    pool: Dict[int, np.ndarray] = {}
    for fold, base_rets in base_folds.items():
        noise = alt_folds.get(fold)
        if noise is None or noise.size == 0:
            continue
        n = min(base_rets.size, noise.size)
        pool[fold] = noise[:n] - base_rets[:n]
    return pool


def _sample_noise_realization(
    base_folds: Dict[int, np.ndarray],
    noise_pool: Optional[Dict[int, np.ndarray]],
    rng: np.random.Generator,
) -> Dict[int, np.ndarray]:
    """采样一个噪声实现（与配对重排同源的随机性）。

    首选（noise_pool 非空）：对「真实换种子逐日差」按**逐折**做有放回重采样——
    每折从噪声池的折集合里抽一个折的差序列贴到本折（保方差与量级，避免块自举的
    右偏伪实现；R-02 补充的保真度教训）。
    兜底（noise_pool 为空）：折内块自举（保自相关），标注 noise_source=block_bootstrap_proxy。
    """
    out: Dict[int, np.ndarray] = {}
    if noise_pool:
        pool_keys = sorted(noise_pool.keys())
        for fold, base_rets in base_folds.items():
            n = base_rets.size
            if n == 0:
                out[fold] = base_rets
                continue
            # 抽一个噪声折的差序列，重采样/截断到本折长度
            src = noise_pool[rng.choice(pool_keys)]
            if src.size == 0:
                out[fold] = np.zeros(n)
                continue
            idx = rng.integers(0, src.size, size=n)  # 逐日有放回（噪声量级代理）
            out[fold] = src[idx]
    else:
        for fold, base_rets in base_folds.items():
            n = base_rets.size
            if n == 0:
                out[fold] = base_rets
                continue
            block = 20
            n_blocks = max(1, (n + block - 1) // block)
            picks = rng.integers(0, n_blocks, size=n_blocks)
            noise = np.concatenate([base_rets[i * block:(i + 1) * block] for i in picks])[:n]
            if noise.size < n:
                noise = np.pad(noise, (0, n - noise.size), mode="edge")
            out[fold] = noise - base_rets  # 噪声 = 块重排 − 原（零均值化）
    return out


@dataclass
class PowerCurvePoint:
    """功效曲线单点：某平移档位下的检出概率（双指标）。"""

    shift_annual_pp: float
    detection_prob_cagr: float  # P(ΔCAGR 95% 区间下界 > 0)
    detection_prob_maxdd: float  # P(ΔMaxDD 95% 区间下界 > 0)（北极星主判据）
    n_boot: int
    noise_source: str  # "real_seed_swap" / "block_bootstrap_proxy"
    effective_dof_note: str  # 有效样本量声明


@dataclass
class PowerCurvePoint:
    """功效曲线单点：某平移档位下的检出概率（双指标，频率语义）。

    检出概率 = M 个独立噪声实现中「配对重排 Δ 95% 区间下界 > 0」的**频率**
    （R-01 修订：不再是单次实现的布尔 {0,1}，而是对噪声实现分布求期望）。
    """

    shift_annual_pp: float
    detection_prob_cagr: float  # 检出频率（M 个噪声实现）
    detection_prob_maxdd: float  # 检出频率（北极星主判据）
    n_noise: int  # 噪声实现数 M
    n_boot: int  # 每实现内的配对重排次数
    noise_source: str  # "real_seed_swap_resample" / "block_bootstrap_proxy"
    effective_dof_note: str  # 有效样本量声明


def _detect_once(
    base_folds: Dict[int, np.ndarray],
    arm_folds: Dict[int, np.ndarray],
    splits: Sequence[int],
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> bool:
    """单次噪声实现的检出判定：配对重排 Δ 的 95% 区间下界 > 0。"""
    deltas: List[float] = []
    for _ in range(int(n_boot)):
        picks = rng.integers(0, len(splits), size=len(splits))
        base_m = chain_metrics_from_fold_returns([base_folds[splits[i]] for i in picks])
        arm_m = chain_metrics_from_fold_returns([arm_folds[splits[i]] for i in picks])
        if base_m[metric] is not None and arm_m[metric] is not None:
            deltas.append(float(arm_m[metric] - base_m[metric]))
    arr = np.asarray(deltas, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return False
    return bool(np.quantile(arr, 0.025) > 0)


def power_calibration_curve(
    baseline: RunArtifacts,
    splits: Sequence[int],
    shift_grid_pp: Sequence[float] = (1.0, 2.0, 5.0),
    n_boot: int = 200,
    n_noise: int = 20,
    seed: int = 42,
    alt_seed: Optional[RunArtifacts] = None,
) -> List[PowerCurvePoint]:
    """功效标定曲线（配对口径，频率语义）：对基线做 +δ 平移 + M 个独立噪声实现，
    逐档统计「配对重排 Δ 95% 区间下界 > 0」的**检出频率**（双指标）。

    Args:
        baseline: 基线臂
        splits: 折子集
        shift_grid_pp: 平移档位（年化 pp）
        n_boot: 每个噪声实现内的配对重排次数
        n_noise: 独立噪声实现数 M（检出概率 = M 个实现里的检出频率；R-01）
        seed: 随机种子
        alt_seed: 真实换种子批（首选噪声源——逐日差池重采样）；为 None 时降级
            block bootstrap 代理并在报告中标注 ``noise_source="block_bootstrap_proxy"``。
    """
    splits = list(splits)
    if not splits:
        raise ValueError("折子集为空，无法做功效标定")

    base_folds = per_fold_returns(baseline, splits)
    alt_folds = per_fold_returns(alt_seed, splits) if alt_seed is not None else None
    noise_pool = _fold_diff_pool(base_folds, alt_folds) if alt_folds is not None else None
    noise_source = "real_seed_swap_resample" if noise_pool else "block_bootstrap_proxy"
    dof_note = (
        f"检出概率 = {n_noise} 个独立噪声实现的检出频率；每实现内 {n_boot} 次配对重排；"
        f"有效自由度 ≈ 折数级（{len(splits)} 折）"
    )
    points: List[PowerCurvePoint] = []

    for shift_pp in shift_grid_pp:
        n_detect_cagr = 0
        n_detect_maxdd = 0
        for k in range(int(n_noise)):
            rng = np.random.default_rng(seed + 1000 * k)  # 每实现独立种子
            noise = _sample_noise_realization(base_folds, noise_pool, rng)
            # 合成臂 = 基线 + 噪声 + 平移（shift 恒叠加，修 R-02）
            shifted = shift_fold_returns_annual_pp(base_folds, shift_pp)
            arm_folds = {
                f: (shifted[f][: noise[f].size] + noise[f]) if noise[f].size else shifted[f]
                for f in base_folds
            }
            if _detect_once(base_folds, arm_folds, splits, "cagr", n_boot, rng):
                n_detect_cagr += 1
            if _detect_once(base_folds, arm_folds, splits, "max_drawdown", n_boot, rng):
                n_detect_maxdd += 1
        points.append(
            PowerCurvePoint(
                shift_annual_pp=float(shift_pp),
                detection_prob_cagr=n_detect_cagr / n_noise,
                detection_prob_maxdd=n_detect_maxdd / n_noise,
                n_noise=int(n_noise),
                n_boot=int(n_boot),
                noise_source=noise_source,
                effective_dof_note=dof_note,
            )
        )
    return points
