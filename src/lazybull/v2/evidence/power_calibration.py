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


def _real_noise_arm(
    base_folds: Dict[int, np.ndarray],
    alt_seed_folds: Dict[int, np.ndarray],
    shift_pp: float,
) -> Dict[int, np.ndarray]:
    """首选构造（P0 评审 A1）：arm = base + (alt_seed − base) + shift。

    Δ_noise = 真实换种子批的逐日差——携带真实噪声宽度，检出功效才是「尺子的真实刻度」。
    折内日数不一致时按短者截断（保守）。
    """
    shifted = shift_fold_returns_annual_pp(base_folds, shift_pp)
    arm: Dict[int, np.ndarray] = {}
    for fold, base_rets in shifted.items():
        noise = alt_seed_folds.get(fold)
        if noise is None or noise.size == 0:
            arm[fold] = base_rets
            continue
        n = min(base_rets.size, noise.size)
        # arm_t = base_t + (alt_t − base_t) + shift_t = alt_t + shift_t（按短者截断对齐）
        arm[fold] = noise[:n] + (base_rets[:n] - base_rets[:n]) + (shifted[fold][:n] - base_folds[fold][:n])
    return arm


def _block_bootstrap_noise_arm(
    base_folds: Dict[int, np.ndarray],
    shift_pp: float,
    seed: int,
    block: int = 20,
) -> Dict[int, np.ndarray]:
    """备选构造（换种子批不可得时代理）：对基线自身做块自举扰动构造 Δ 噪声。"""
    rng = np.random.default_rng(seed)
    shifted = shift_fold_returns_annual_pp(base_folds, shift_pp)
    arm: Dict[int, np.ndarray] = {}
    for fold, rets in shifted.items():
        n = rets.size
        if n == 0:
            arm[fold] = rets
            continue
        # 块自举重排基线收益（保留自相关的噪声代理）
        n_blocks = max(1, n // block)
        picks = rng.integers(0, n_blocks, size=n_blocks)
        noise = np.concatenate([
            base_folds[fold][i * block:(i + 1) * block] for i in picks
        ])[:n]
        if noise.size < n:
            noise = np.pad(noise, (0, n - noise.size), mode="edge")
        arm[fold] = noise
    return arm


@dataclass
class PowerCurvePoint:
    """功效曲线单点：某平移档位下的检出概率（双指标）。"""

    shift_annual_pp: float
    detection_prob_cagr: float  # P(ΔCAGR 95% 区间下界 > 0)
    detection_prob_maxdd: float  # P(ΔMaxDD 95% 区间下界 > 0)（北极星主判据）
    n_boot: int
    noise_source: str  # "real_seed_swap" / "block_bootstrap_proxy"
    effective_dof_note: str  # 有效样本量声明


def power_calibration_curve(
    baseline: RunArtifacts,
    splits: Sequence[int],
    shift_grid_pp: Sequence[float] = (1.0, 2.0, 5.0),
    n_boot: int = 1000,
    seed: int = 42,
    alt_seed: Optional[RunArtifacts] = None,
) -> List[PowerCurvePoint]:
    """功效标定曲线（配对口径）：对基线做 +δ 平移 + 真实噪声合成臂，
    逐档做配对制度重排（基线 vs 合成臂），读 ΔCAGR 与 ΔMaxDD 双指标检出概率。

    Args:
        baseline: 基线臂
        splits: 折子集
        shift_grid_pp: 平移档位（年化 pp）
        n_boot / seed: 配对重排参数
        alt_seed: 真实换种子批（首选噪声源，P0 评审 A1）；为 None 时降级 block bootstrap
            代理并在报告中标注 ``noise_source="block_bootstrap_proxy"``。

    检出定义（裁决唯一口径，双指标）：配对差分 Δ 的 95% 区间下界 > 0。
    """
    splits = list(splits)
    if not splits:
        raise ValueError("折子集为空，无法做功效标定")

    base_folds = per_fold_returns(baseline, splits)
    noise_source = "real_seed_swap" if alt_seed is not None else "block_bootstrap_proxy"
    alt_folds = per_fold_returns(alt_seed, splits) if alt_seed is not None else None
    dof_note = (
        f"重排 {n_boot} 次 ≠ {n_boot} 个独立制度；有效自由度 ≈ 折数级（{len(splits)} 折）"
    )
    points: List[PowerCurvePoint] = []

    for gi, shift_pp in enumerate(shift_grid_pp):
        # 构造合成臂：真实换种子噪声（首选）或 block bootstrap 代理
        if alt_folds is not None:
            arm_folds = _real_noise_arm(base_folds, alt_folds, shift_pp)
        else:
            arm_folds = _block_bootstrap_noise_arm(base_folds, shift_pp, seed=seed + gi)

        rng = np.random.default_rng(seed)
        d_cagr: List[float] = []
        d_maxdd: List[float] = []
        for _ in range(int(n_boot)):
            picks = rng.integers(0, len(splits), size=len(splits))
            base_m = chain_metrics_from_fold_returns([base_folds[splits[i]] for i in picks])
            arm_m = chain_metrics_from_fold_returns([arm_folds[splits[i]] for i in picks])
            if base_m["cagr"] is not None and arm_m["cagr"] is not None:
                d_cagr.append(float(arm_m["cagr"] - base_m["cagr"]))
            if base_m["max_drawdown"] is not None and arm_m["max_drawdown"] is not None:
                d_maxdd.append(float(arm_m["max_drawdown"] - base_m["max_drawdown"]))

        def _detect(samples: List[float]) -> float:
            arr = np.asarray(samples, dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return 0.0
            return float(np.quantile(arr, 0.025) > 0)

        points.append(
            PowerCurvePoint(
                shift_annual_pp=float(shift_pp),
                detection_prob_cagr=_detect(d_cagr),
                detection_prob_maxdd=_detect(d_maxdd),
                n_boot=int(n_boot),
                noise_source=noise_source,
                effective_dof_note=dof_note,
            )
        )
    return points
