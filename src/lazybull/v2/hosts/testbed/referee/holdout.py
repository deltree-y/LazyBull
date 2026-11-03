# -*- coding: utf-8 -*-
"""P5b T4 收益裁判 holdout 验收（设计 v4 §5.3「逐单元 Wilson 门」）。

验收口径
========
- 验收单元 = **依赖场景 × 零假设 × H × 裁判**；每个单元单独计算「任一注册 look 触发」的
  family-wise 频率，**禁止跨场景/跨单元加权平均或混池**；
- ``mu=0`` 与 ``mu=-delta`` 是**两个独立单元**（不得合并）；
- 判定：每单元 Wilson 95% **上界** ≤ 6%；任一单元不过 ⇒ 裁判不可用（不得删困难场景、
  不得扩样本反复调 ``c_k``）；
- 无效 look **不触发**否决，但实现仍留在 Wilson 分母，并逐 look 报告无效率（>10% ⇒ 该单元
  直接不通过）；
- 本模块**只读**临界值产物：入参是已冻结的阈值序列，不返回任何可写回 ``c_k`` 的路径。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.sampling import IMPLEMENTATION_ID
from src.lazybull.v2.hosts.testbed.referee.scenes import SceneRegistry
from src.lazybull.v2.hosts.testbed.referee.statistics import (
    DEFAULT_CHUNK,
    DELTA_LOG_DAILY,
    STATISTIC_PSD,
    LookPlan,
    veto_flags,
)
from src.lazybull.v2.hosts.testbed.referee.training import (
    NULL_SPECS,
    ThresholdConfig,
    generate_scores,
)

__all__ = [
    "HOLDOUT_SCHEMA",
    "WILSON_CONFIDENCE",
    "WILSON_Z",
    "HoldoutError",
    "evaluate_holdout",
    "holdout_unit_count",
    "wilson_interval",
]

#: 产物 schema
HOLDOUT_SCHEMA = "p5b_testbed_income_holdout/v1"
#: Wilson 95% 双侧正态分位
WILSON_Z = 1.959963984540054
#: Wilson 置信水平（冻结）
WILSON_CONFIDENCE = 0.95
#: 报告口径（禁止聚合的唯一声明点）
REPORTING_RULE = "per_unit_only_no_pooling_no_weighting"


class HoldoutError(TestbedError):
    """holdout 输入非法（阈值个数不符、阈值非有限、注册表角色错误）。"""

    __test__ = False


def wilson_interval(successes: int, total: int, *, z: float = WILSON_Z) -> Tuple[float, float]:
    """Wilson score 区间（``total<=0`` 即报错，避免用空单元冒充通过）。"""
    count = int(total)
    if count <= 0:
        raise HoldoutError(f"Wilson 区间要求 total>0，实际 {total!r}")
    hits = int(successes)
    if hits < 0 or hits > count:
        raise HoldoutError(f"successes 必须在 [0,total] 内: {successes}/{total}")
    p = hits / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def holdout_unit_count(scene_count: int) -> int:
    """验收单元数 = 场景 × 零假设 × H(1) × 裁判(1)。"""
    return int(scene_count) * len(NULL_SPECS)


def _validate_holdout_inputs(
    *,
    plan: LookPlan,
    thresholds: Sequence[float],
    config: ThresholdConfig,
    wilson_upper_max: float,
    z: float,
) -> List[float]:
    """阈值与门的冻结口径校验（任何偏离即拒绝；返回规范化后的 ``c_k``）。

    Raises:
        HoldoutError: 阈值个数/取值非法、Wilson 门或分位被放宽/收紧、无效率门被改动。
    """
    frozen = [float(value) for value in thresholds]
    if len(frozen) != plan.k:
        raise HoldoutError(f"阈值个数({len(frozen)}) 必须等于 K({plan.k})")
    if not np.isfinite(frozen).all():
        raise HoldoutError(f"阈值必须有限（未训练出临界值不得验收）: {frozen}")
    if float(wilson_upper_max) > 0.06:
        raise HoldoutError(f"Wilson 上界门不得放宽到 0.06 以上，实际 {wilson_upper_max}")
    # z 也是门的一部分：偏离冻结值（例如 z=0 会让区间退化为 [p,p]）必须拒绝
    if not math.isclose(float(z), WILSON_Z, rel_tol=0.0, abs_tol=1e-9):
        raise HoldoutError(f"Wilson 分位 z 必须等于冻结值 {WILSON_Z}，实际 {z}")
    if not math.isclose(float(wilson_upper_max), 0.06, rel_tol=1e-12, abs_tol=1e-12):
        raise HoldoutError(
            f"Wilson 上界门必须等于冻结值 0.06（不得收紧后充当其它口径），实际 {wilson_upper_max}"
        )
    # 无效率门同样是冻结口径（可被放大成"永不过门"从而静默废掉该门）
    if not math.isclose(float(config.invalid_rate_per_look_max), 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise HoldoutError(f"无效率门必须等于冻结值 0.10，实际 {config.invalid_rate_per_look_max}")
    return frozen


def evaluate_holdout(
    *,
    plan: LookPlan,
    registry: SceneRegistry,
    seed: int,
    count: int,
    thresholds: Sequence[float],
    config: ThresholdConfig,
    statistic: str = STATISTIC_PSD,
    delta: float = DELTA_LOG_DAILY,
    prefix: str = "",
    wilson_upper_max: float = 0.06,
    z: float = WILSON_Z,
    chunk: Optional[int] = None,
) -> Dict[str, Any]:
    """逐单元 holdout 验收（只读阈值，不产生任何阈值回写）。

    Args:
        plan: look 映射。
        registry: **holdout 池**场景注册表（``pool_role`` 必须是 ``holdout``）。
        seed: 命名种子根（与训练同根、子流互斥）。
        count: 每单元路径数（manifest 冻结值 10000）。
        thresholds: 已冻结 ``c_k``（长度必须 = K）。
        config: 阈值参数（用于无效率门）。
        statistic: 注册统计量。
        delta: 日度 log 基准。
        prefix: realization 前缀。
        wilson_upper_max: Wilson 上界门（冻结 0.06）。
        z: Wilson 分位。
        chunk: 分块路径数。

    Returns:
        ``{schema, units, failed_units, referee_usable, ...}``（逐单元明细，无聚合字段）。

    Raises:
        HoldoutError: 注册表角色非 holdout、阈值个数/取值非法、门槛被放宽。
    """
    if registry.pool_role != "holdout":
        raise HoldoutError(
            f"holdout 必须使用独立池（pool_role='holdout'），实际 {registry.pool_role!r}"
        )
    frozen = _validate_holdout_inputs(
        plan=plan, thresholds=thresholds, config=config, wilson_upper_max=wilson_upper_max, z=z
    )
    units: Dict[str, Any] = {}
    failed: List[str] = []
    for scene in registry.names():
        for null, scale in NULL_SPECS:
            mu = 0.0 if null == "mu=0" else float(scale) * float(delta)
            scores, realization_id = generate_scores(
                registry,
                scene,
                seed=seed,
                stage="holdout",
                null=null,
                mu=mu,
                count=int(count),
                plan=plan,
                statistic=statistic,
                delta=delta,
                prefix=prefix,
                chunk=chunk,
            )
            flags = veto_flags(scores.values, frozen)
            successes = int(flags.sum())
            total = int(flags.size)
            low, high = wilson_interval(successes, total, z=z)
            invalid_by_look = [scores.invalid_rate(index) for index in range(plan.k)]
            reasons: List[str] = []
            if high > float(wilson_upper_max):
                reasons.append("wilson_upper_exceeds_max")
            if max(invalid_by_look) > config.invalid_rate_per_look_max:
                reasons.append("invalid_rate_exceeds_max")
            key = f"{scene}|{null}"
            units[key] = {
                "unit": key,
                "scene": scene,
                "null": null,
                "mu": float(mu),
                "H": int(plan.h),
                "referee": "income_psd_boundary_centered",
                "realization_id": realization_id,
                "successes": successes,
                "total": total,
                "rate": successes / total,
                "wilson_low": low,
                "wilson_high": high,
                "invalid_rate_by_look": invalid_by_look,
                "invalid_reasons_by_look": [scores.reason_counts(index) for index in range(plan.k)],
                "pass": not reasons,
                "failure_reasons": reasons,
            }
            if reasons:
                failed.append(key)
    return {
        "schema": HOLDOUT_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "statistic": statistic,
        "delta": float(delta),
        "plan": plan.as_dict(),
        "unit_definition": "scene × null × H × referee",
        "unit_count": len(units),
        "counts": {
            "holdout_per_unit": int(count),
            "seed": int(seed),
            "realization_prefix": prefix,
            # 保真：None = 真正单批，不得伪记为默认分块值
            "chunk": None if chunk is None else int(chunk),
            "effective_chunk": DEFAULT_CHUNK if chunk is None else int(chunk),
        },
        "reporting_rule": REPORTING_RULE,
        "wilson_confidence": WILSON_CONFIDENCE,
        "wilson_z": float(z),
        "wilson_upper_max": float(wilson_upper_max),
        "invalid_rate_per_look_max": float(config.invalid_rate_per_look_max),
        "thresholds": frozen,
        "units": units,
        "failed_units": failed,
        "referee_usable": not failed,
    }
