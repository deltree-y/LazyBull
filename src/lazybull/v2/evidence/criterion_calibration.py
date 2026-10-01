# -*- coding: utf-8 -*-
"""判据自洽性标定（P5a-1：基准袖子过不了的判据不得用于裁决；方案 §3.2 / R1-3）。

两类自洽性测试：
1. **A vs A 换种子 ⇒ 判据应判「不可区分」**（否则判据把噪声当信号）；
2. **A vs A+2pp 平移 ⇒ 判据应判「可检出」**（否则判据在真实改进量级上无功效）。

判定载体 = 配对制度重排（regime_resample.paired_regime_bootstrap）的 Δ 分布：
「不可区分」= Δ 95% 区间跨 0；「可检出」= Δ 95% 区间下界 > 0。

用途：P1.5 政策层裁决 / P7 袖子准入判据的适用性判定（基准袖子过不了的判据，
不得用于裁决——判据自洽性标定是 metrics_v1 一次性修订窗口的触发输入）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts.compare.fold_subset import RunArtifacts, per_fold_returns
from src.lazybull.v2.evidence.power_calibration import shift_fold_returns_annual_pp
from src.lazybull.v2.evidence.regime_resample import (
    paired_regime_bootstrap,
    PairedDeltaResult,
)


@dataclass
class SelfConsistencyResult:
    """判据自洽性测试结果。"""

    # A vs A 换种子（应不可区分）
    seed_swap: PairedDeltaResult
    seed_swap_indistinguishable: bool  # Δ 95% 区间跨 0
    # A vs A+shift 平移（应可检出）
    shifted: PairedDeltaResult
    shifted_detectable: bool  # Δ 95% 区间下界 > 0
    shift_pp: float
    # 综合：判据是否通过自洽性（可用于裁决）
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
    n_boot: int = 1000,
    seed: int = 42,
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
        n_boot / seed: 配对重排参数
    """
    splits = list(splits)
    if not splits:
        raise ValueError("折子集为空，无法做判据自洽性标定")

    # --- 测试 1：A vs A 换种子（应不可区分） ---
    seed_ref = arm_a_alt_seed if arm_a_alt_seed is not None else arm_a
    seed_swap = paired_regime_bootstrap(arm_a, seed_ref, splits, n_boot=n_boot, seed=seed)
    seed_swap_indistinguishable = _interval_crosses_zero(seed_swap, metric)

    # --- 测试 2：A vs A+shift 平移（应可检出） ---
    shifted_folds = shift_fold_returns_annual_pp(per_fold_returns(arm_a, splits), shift_pp)
    # 构造一个「平移臂」RunArtifacts 视图（只改 chain 折收益，其余沿用）
    # 注：paired_regime_bootstrap 只消费 per_fold_returns 结果，故此处直接用
    # 平移后的折收益字典做配对，绕过 RunArtifacts 包装。
    from scripts.compare.fold_subset import chain_metrics_from_fold_returns

    base_folds = per_fold_returns(arm_a, splits)
    rng = np.random.default_rng(seed)
    delta_samples: Dict[str, List[float]] = {metric: []}
    for _ in range(int(n_boot)):
        picks = rng.integers(0, len(splits), size=len(splits))
        base_m = chain_metrics_from_fold_returns([base_folds[splits[i]] for i in picks])
        arm_m = chain_metrics_from_fold_returns([shifted_folds[splits[i]] for i in picks])
        if base_m[metric] is not None and arm_m[metric] is not None:
            delta_samples[metric].append(float(arm_m[metric] - base_m[metric]))
    shifted = PairedDeltaResult(
        n_boot=int(n_boot),
        n_folds=len(splits),
        seed=int(seed),
        delta_samples=delta_samples,
        point_delta={},
    )
    shifted_detectable = _interval_above_zero(shifted, metric)

    criterion_passes = bool(seed_swap_indistinguishable and shifted_detectable)
    if criterion_passes:
        note = f"判据自洽（{metric}）：换种子不可区分 ✓ +{shift_pp}pp 可检出 ✓ ⇒ 可用于裁决"
    elif not seed_swap_indistinguishable:
        note = (
            f"判据失效（{metric}）：A vs A 换种子被判为「可区分」——"
            "判据把噪声当信号，不得用于裁决（沿 R1-3：基准袖过不了的判据不得用于裁决）"
        )
    else:
        note = (
            f"判据功效不足（{metric}）：+{shift_pp}pp 平移未被检出——"
            "判据在该改进量级上无分辨力，裁决需降级（沿 P1.5 fallback）"
        )

    return SelfConsistencyResult(
        seed_swap=seed_swap,
        seed_swap_indistinguishable=seed_swap_indistinguishable,
        shifted=shifted,
        shifted_detectable=shifted_detectable,
        shift_pp=float(shift_pp),
        criterion_passes=criterion_passes,
        note=note,
    )
