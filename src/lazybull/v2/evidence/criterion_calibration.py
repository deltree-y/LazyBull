# -*- coding: utf-8 -*-
"""判据自洽性标定（P5a-1：基准袖子过不了的判据不得用于裁决；方案 §3.2 / R1-3）。

两类自洽性测试（P0 评审 A2 修订：必须跑**真实换种子批**，恒等对照只作流程自检）：
1. **A vs A 换种子（真实批）⇒ 判据应判「不可区分」**（否则判据把噪声当信号）；
2. **A vs A+2pp 平移（携带真实噪声）⇒ 判据应判「可检出」**（否则在真实改进量级上无功效）。

判定载体 = 配对制度重排的 Δ 分布：「不可区分」= Δ 95% 区间跨 0；「可检出」= 下界 > 0。
**双指标**（P0 评审 A1）：主判据 ΔMaxDD 与 ΔCAGR 都须过自洽性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts.compare.fold_subset import RunArtifacts, chain_metrics_from_fold_returns, per_fold_returns
from src.lazybull.v2.evidence.power_calibration import (
    _fold_diff_pool,
    _sample_noise_realization,
    shift_fold_returns_annual_pp,
)
from src.lazybull.v2.evidence.regime_resample import (
    paired_regime_bootstrap,
    PairedDeltaResult,
)


@dataclass
class SelfConsistencyResult:
    """判据自洽性测试结果（双指标；测试 2 升级为频率语义）。"""

    metric: str
    # 测试 1：A vs A 换种子（应不可区分）——真实批 + 配对重排（有效）
    seed_swap: PairedDeltaResult
    seed_swap_indistinguishable: bool  # Δ 95% 区间跨 0
    seed_swap_is_real: bool  # 是否真实换种子批（False = 恒等自检，不作数）
    # 测试 2：A vs A+shift（携带 M 个噪声实现，应可检出）——检出频率
    shift_pp: float
    shift_detection_freq: float  # M 个噪声实现中检出的频率（R-01 修订）
    n_noise: int  # 噪声实现数 M
    shift_detect_threshold: float  # 检出频率阈值（如 0.8 = 80% 实现可检出才算有功效）
    shifted_detectable: bool  # shift_detection_freq >= shift_detect_threshold
    # 综合：判据是否通过自洽性（可用于裁决）；恒等自检恒为 False（不作数）
    criterion_passes: bool
    note: str


def _interval_crosses_zero(result: PairedDeltaResult, metric: str) -> bool:
    s = result.summary(metric)
    if not s:
        return True  # 无样本视为不可区分（保守）
    return bool(s["ci95_low"] <= 0 <= s["ci95_high"])


def _interval_above_zero(result: PairedDeltaResult, metric: str) -> bool:
    s = result.summary(metric)
    if not s:
        return False
    return bool(s["ci95_low"] > 0)


def criterion_self_consistency(
    arm_a: RunArtifacts,
    arm_a_alt_seed: Optional[RunArtifacts],
    splits: Sequence[int],
    metric: str = "cagr",
    shift_pp: float = 2.0,
    n_boot: int = 100,
    n_noise: int = 20,
    seed: int = 42,
    shift_detect_threshold: float = 0.8,
) -> SelfConsistencyResult:
    """判据自洽性标定。

    Args:
        arm_a: 基准臂（如 B0 / 基准袖子 A）
        arm_a_alt_seed: A 的换种子臂；为 None 时用「A 自身」做恒等对照
            （配对骨架下同臂 Δ 恒为 0，恒不可区分——仅作流程自检，
            真实换种子对照需独立的换种子批次）。
        splits: 折子集
        metric: 判据指标（cagr / max_drawdown / sharpe）
        shift_pp: 平移档位（默认 +2pp，判据可检出性测试）
        n_boot: 每个噪声实现内的配对重排次数
        n_noise: 独立噪声实现数 M（测试 2 检出概率 = M 个实现里的检出频率；R-01）
        seed: 随机种子
        shift_detect_threshold: 测试 2 的检出频率阈值（默认 0.8 = 80% 实现可检出才算有功效）
    """
    splits = list(splits)
    if not splits:
        raise ValueError("折子集为空，无法做判据自洽性标定")

    # --- 测试 1：A vs A 换种子（应不可区分；真实批优先，恒等对照只作流程自检） ---
    is_real = arm_a_alt_seed is not None
    seed_ref = arm_a_alt_seed if is_real else arm_a
    seed_swap = paired_regime_bootstrap(arm_a, seed_ref, splits, n_boot=n_boot, seed=seed)
    seed_swap_indistinguishable = _interval_crosses_zero(seed_swap, metric)

    # --- 测试 2：A vs A+shift 平移（应可检出；M 个噪声实现的检出频率，R-01/R-03 修订） ---
    base_folds = per_fold_returns(arm_a, splits)
    alt_folds = per_fold_returns(arm_a_alt_seed, splits) if is_real else None
    noise_pool = _fold_diff_pool(base_folds, alt_folds) if alt_folds is not None else None

    n_detect = 0
    for k in range(int(n_noise)):
        rng = np.random.default_rng(seed + 1000 * k)
        noise = _sample_noise_realization(base_folds, noise_pool, rng)
        shifted_base = shift_fold_returns_annual_pp(base_folds, shift_pp)
        arm_folds = {
            f: (shifted_base[f][: noise[f].size] + noise[f]) if noise[f].size else shifted_base[f]
            for f in base_folds
        }
        # 单实现检出判定：配对重排 Δ 95% 区间下界 > 0
        deltas: List[float] = []
        for _ in range(int(n_boot)):
            picks = rng.integers(0, len(splits), size=len(splits))
            base_m = chain_metrics_from_fold_returns([base_folds[splits[i]] for i in picks])
            arm_m = chain_metrics_from_fold_returns([arm_folds[splits[i]] for i in picks])
            if base_m[metric] is not None and arm_m[metric] is not None:
                deltas.append(float(arm_m[metric] - base_m[metric]))
        arr = np.asarray(deltas, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size and np.quantile(arr, 0.025) > 0:
            n_detect += 1
    shift_detection_freq = n_detect / n_noise
    shifted_detectable = bool(shift_detection_freq >= shift_detect_threshold)

    # 恒等自检（无真实换种子批）不作数：criterion_passes 恒 False 并标注
    if not is_real:
        criterion_passes = False
        note = (
            f"[恒等自检，不作数] {metric}：未提供真实换种子批（arm_a_alt_seed=None），"
            "Δ 恒 0 ⇒「不可区分」平凡为真；真实标定需接入换种子批（P0 评审 A2）"
        )
    else:
        criterion_passes = bool(seed_swap_indistinguishable and shifted_detectable)
        if criterion_passes:
            note = (
                f"判据自洽（{metric}）：换种子不可区分 ✓ +{shift_pp}pp 检出频率 "
                f"{shift_detection_freq:.0%}（≥{shift_detect_threshold:.0%}）✓ ⇒ 可用于裁决"
            )
        elif not seed_swap_indistinguishable:
            note = (
                f"判据失效（{metric}）：A vs A 换种子被判为「可区分」——"
                "判据把噪声当信号，不得用于裁决（沿 R1-3）"
            )
        else:
            note = (
                f"判据功效不足（{metric}）：+{shift_pp}pp 检出频率 {shift_detection_freq:.0%} "
                f"< {shift_detect_threshold:.0%}——该改进量级上无分辨力，裁决需降级（沿 P1.5 fallback）"
            )

    return SelfConsistencyResult(
        metric=metric,
        seed_swap=seed_swap,
        seed_swap_indistinguishable=seed_swap_indistinguishable,
        seed_swap_is_real=is_real,
        shift_pp=float(shift_pp),
        shift_detection_freq=shift_detection_freq,
        n_noise=int(n_noise),
        shift_detect_threshold=shift_detect_threshold,
        shifted_detectable=shifted_detectable,
        criterion_passes=criterion_passes,
        note=note,
    )


def criterion_self_consistency_dual(
    arm_a: RunArtifacts,
    arm_a_alt_seed: Optional[RunArtifacts],
    splits: Sequence[int],
    metrics: Sequence[str] = ("cagr", "max_drawdown"),
    shift_pp: float = 2.0,
    n_boot: int = 100,
    n_noise: int = 20,
    seed: int = 42,
    shift_detect_threshold: float = 0.8,
) -> Dict[str, SelfConsistencyResult]:
    """双指标判据自洽性（P0 评审 A1：主判据 ΔMaxDD 与 ΔCAGR 都须过）。

    Returns:
        指标名 → SelfConsistencyResult；整体通过 = 所有指标 criterion_passes。
    """
    return {
        m: criterion_self_consistency(
            arm_a,
            arm_a_alt_seed,
            splits,
            metric=m,
            shift_pp=shift_pp,
            n_boot=n_boot,
            n_noise=n_noise,
            seed=seed,
            shift_detect_threshold=shift_detect_threshold,
        )
        for m in metrics
    }
