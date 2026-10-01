# -*- coding: utf-8 -*-
"""功效标定（P5a-1：证据机器的「最小刻度表」；方案 §3.5）。

构造规则（方案 §3.5 预登记，禁止事后挑选）：**折内日收益等值平移**——
保持折内路径形状、只抬水平，最贴近「同策略更好运气」的语义；
构造规则本身是消融位，写死才允许产出曲线。

产出：合成臂 +1 / +2 / +5pp 年化 → 配对制度重排口径下的**检出概率**曲线
（检出 = 配对差分 ΔCAGR 95% 区间下界 > 0）。所有分布结论标注有效样本量
（重排 N 次 ≠ N 个独立制度，深尾分位有效自由度 ≈ 折数级，方案 §3.5）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

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
    """折内日收益等值平移（构造规则写死）：每折每交易日收益加同一日度偏移。

    Args:
        fold_returns: 折号 → 折内日收益序列
        annual_pp: 目标抬升的年化百分点（如 +2.0 表示 +2pp 年化）

    Returns:
        平移后的折收益字典（新数组，不改原值）。
    """
    # 年化 pp → 日度复利偏移：日收益加 delta，使 (1+delta)^252 ≈ 1 + annual_pp/100
    daily_delta = (1.0 + annual_pp / 100.0) ** (1.0 / _ANNUAL_DAYS) - 1.0
    return {
        fold: (rets + daily_delta) if rets.size > 0 else rets
        for fold, rets in fold_returns.items()
    }


@dataclass
class PowerCurvePoint:
    """功效曲线单点：某平移档位下的检出概率。"""

    shift_annual_pp: float
    detection_prob: float  # P(ΔCAGR 95% 区间下界 > 0)
    n_boot: int
    effective_dof_note: str  # 有效样本量声明


def power_calibration_curve(
    baseline: RunArtifacts,
    splits: Sequence[int],
    shift_grid_pp: Sequence[float] = (1.0, 2.0, 5.0),
    n_boot: int = 1000,
    seed: int = 42,
) -> List[PowerCurvePoint]:
    """功效标定曲线（配对口径）：对基线做 +1/+2/+5pp 平移合成臂，
    逐档做配对制度重排（基线 vs 平移臂），读 ΔCAGR 的检出概率。

    检出定义（裁决唯一口径）：配对差分 ΔCAGR 的 95% 区间下界 > 0。
    """
    splits = list(splits)
    if not splits:
        raise ValueError("折子集为空，无法做功效标定")

    base_folds = per_fold_returns(baseline, splits)
    dof_note = (
        f"重排 {n_boot} 次 ≠ {n_boot} 个独立制度；有效自由度 ≈ 折数级（{len(splits)} 折）"
    )
    points: List[PowerCurvePoint] = []

    for shift_pp in shift_grid_pp:
        shifted = shift_fold_returns_annual_pp(base_folds, shift_pp)
        rng = np.random.default_rng(seed)
        delta_cagr: List[float] = []
        for _ in range(int(n_boot)):
            picks = rng.integers(0, len(splits), size=len(splits))
            base_m = chain_metrics_from_fold_returns([base_folds[splits[i]] for i in picks])
            arm_m = chain_metrics_from_fold_returns([shifted[splits[i]] for i in picks])
            if base_m["cagr"] is None or arm_m["cagr"] is None:
                continue
            delta_cagr.append(float(arm_m["cagr"] - base_m["cagr"]))
        arr = np.asarray(delta_cagr, dtype=float)
        arr = arr[~np.isnan(arr)]
        # 检出 = 95% 区间下界 > 0
        ci_low = float(np.quantile(arr, 0.025)) if arr.size else float("nan")
        detection = float(ci_low > 0) if arr.size else 0.0
        points.append(
            PowerCurvePoint(
                shift_annual_pp=float(shift_pp),
                detection_prob=detection,
                n_boot=int(n_boot),
                effective_dof_note=dof_note,
            )
        )
    return points
