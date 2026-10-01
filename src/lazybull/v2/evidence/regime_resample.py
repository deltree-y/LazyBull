# -*- coding: utf-8 -*-
"""配对制度重排（P5a-1 裁决唯一口径；方案 §3.5 v1.6 定死）。

裁决口径：**两臂共用同一折排列集**（同一组随机抽折序列），各自重建链式净值后
比较 Δ 指标分布——非配对单臂分布（如 MaxDD p90）只反映排列不确定性，禁止用于
两臂裁决（沿 block_paired_delta / 「ΔMaxDD 仅限同一 λ 序列内比较」契约在重排域
的自然推广）。

实现复用生产在用的配对骨架 ``scripts/compare/fold_subset.py::bootstrap_delta``
（per_fold_returns + chain_metrics_from_fold_returns），不复制公式；本模块只做
「指定折集合 + 指定两臂」的薄封装与结果汇总。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from scripts.compare.fold_subset import (
    RunArtifacts,
    chain_metrics_from_fold_returns,
    per_fold_returns,
)


@dataclass
class PairedDeltaResult:
    """配对重排差分结果（两臂共用同一折排列集）。"""

    n_boot: int
    n_folds: int
    seed: int
    # 逐指标的 Δ 分布采样（arm - baseline，长度 = n_boot）
    delta_samples: Dict[str, List[float]] = field(default_factory=dict)
    # 点估计（原始链）
    point_delta: Dict[str, Optional[float]] = field(default_factory=dict)

    def summary(self, metric_key: str) -> Dict[str, float]:
        """单指标分布摘要：P(Δ>0) / 中位 / 95% 区间 / 最差排列 Δ。"""
        samples = np.asarray(self.delta_samples.get(metric_key, []), dtype=float)
        samples = samples[~np.isnan(samples)]
        if samples.size == 0:
            return {}
        return {
            "n": int(samples.size),
            "p_gt_0": float((samples > 0).mean()),
            "median": float(np.median(samples)),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "worst": float(samples.min()),
        }


def paired_regime_bootstrap(
    baseline: RunArtifacts,
    arm: RunArtifacts,
    splits: Sequence[int],
    n_boot: int = 1000,
    seed: int = 42,
) -> PairedDeltaResult:
    """配对制度重排：两臂共用同一折排列集，比较 ΔMaxDD / ΔCAGR / Δ夏普 分布。

    Args:
        baseline: 基线臂产物
        arm: 对照臂产物
        splits: 折子集（两臂必须同集合，调用方先校验）
        n_boot: 重排次数
        seed: 随机种子（同 seed 保证两臂共用同一排列集）

    Returns:
        PairedDeltaResult（delta_samples 的键 = cagr / max_drawdown / sharpe）
    """
    splits = list(splits)
    if not splits:
        raise ValueError("折子集为空，无法做配对制度重排")

    base_folds = per_fold_returns(baseline, splits)
    arm_folds = per_fold_returns(arm, splits)
    rng = np.random.default_rng(seed)

    deltas: Dict[str, List[float]] = {"cagr": [], "max_drawdown": [], "sharpe": []}
    for _ in range(int(n_boot)):
        # 同一抽折序列（配对核心：两臂 picks 完全相同）
        picks = rng.integers(0, len(splits), size=len(splits))
        base_m = chain_metrics_from_fold_returns([base_folds[splits[i]] for i in picks])
        arm_m = chain_metrics_from_fold_returns([arm_folds[splits[i]] for i in picks])
        for key in deltas:
            left, right = arm_m[key], base_m[key]
            if left is None or right is None:
                continue
            deltas[key].append(float(left - right))

    # 点估计（原始全链）
    base_point = chain_metrics_from_fold_returns([base_folds[s] for s in splits])
    arm_point = chain_metrics_from_fold_returns([arm_folds[s] for s in splits])
    point = {
        key: (
            None
            if arm_point[key] is None or base_point[key] is None
            else float(arm_point[key] - base_point[key])
        )
        for key in ("cagr", "max_drawdown", "sharpe")
    }

    return PairedDeltaResult(
        n_boot=int(n_boot),
        n_folds=len(splits),
        seed=int(seed),
        delta_samples=deltas,
        point_delta=point,
    )
