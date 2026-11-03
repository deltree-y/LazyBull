# -*- coding: utf-8 -*-
"""P5b T4 收益裁判临界值训练（设计 v4 §5.2 五步规则 + §5.5 冻结）。

五步（写死在代码里、逐组输出中间量可审计）
==========================================
1. 训练集按 **场景 × 零假设** 单元分组（``ma20``/``ar1_06``/``regime_hetero``/``real_seed_block``
   × ``mu=0``/``mu=-delta``）；
2. 每组用命名子流生成固定数量路径（默认 2000/单元），逐 look 计算注册统计量 ``T_k``；
3. 对每组、每个 look 取**下尾**分位 ``q_{alpha_train/K}``，令
   ``c_{k,unit} = -q_{alpha_train/K}(T_valid[unit,k])``（等价于取 ``-T_k`` 的上分位）；
4. ``c_k = max_unit(c_{k,unit})`` —— 最不利边界场景参与阈值选择；
5. 该规则**不读取 holdout**、不因 holdout 结果回调。保证强度如实说明：这是
   **角色断言**（``registry.pool_role`` 必须为 ``train``）+ **池由互斥源折独立构造**
   （``pool_role='train'`` 的池只含训练源折）+ **产物只写一次**（已存在即拒绝覆盖）三者叠加的
   契约保证，不是编译期不可表达（同一进程内手工把 holdout 池塞进 ``SceneRegistry`` 仍可绕过；
   该风险由 manifest 冻结的源折清单与 ``ScriptedRegistry`` 用例显式暴露）。

硬失败（不得"排除后继续"）
==========================
- 任一单元任一已登记 look 的无效率 > ``invalid_rate_per_look_max``（0.10）⇒ 配置失败；
- 某 look 的有效样本不足以估计分位（< ``min_valid_quantile_sample``）⇒ 配置失败；
- 某 look 无任何可用单元（阈值不可估计）⇒ 配置失败（禁止用极值替代）。
无效样本本身**不触发否决**，但保留在 family 分母与无效率统计中（不得删除实现）。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.sampling import IMPLEMENTATION_ID, substream
from src.lazybull.v2.hosts.testbed.referee.scenes import SceneRegistry
from src.lazybull.v2.hosts.testbed.referee.statistics import (
    DELTA_LOG_DAILY,
    STATISTIC_PSD,
    LookPlan,
    ScoreBlock,
    score_paths,
)

__all__ = [
    "MIN_VALID_QUANTILE_SAMPLE",
    "NULL_SPECS",
    "ThresholdConfig",
    "ThresholdError",
    "TrainingResult",
    "TRAINING_RULE_STEPS",
    "generate_scores",
    "threshold_rule_steps",
    "train_income_thresholds",
]

#: 有效样本不足以估计分位的下限（与 T0.5b harness 同口径）
MIN_VALID_QUANTILE_SAMPLE = 100
#: 零假设单元（名称, 日度 log 位移）
NULL_SPECS: Tuple[Tuple[str, float], ...] = (("mu=0", 0.0), ("mu=-delta", -1.0))
#: §5.2 五步规则的机器可读描述（写进 manifest/thresholds 产物，供审计比对）
TRAINING_RULE_STEPS: Tuple[str, ...] = (
    "group_by(scene × null)",
    "per_look_lower_quantile(T_valid, alpha_train/K)",
    "per_unit c_k = -lower_quantile",
    "c_k = max over eligible units (worst-case scene participates)",
    "c_k must be finite and > 0 (non-positive ⇒ config failure; no clamping)",
    "rule_does_not_read_holdout_and_no_callback_from_holdout",
)


class ThresholdError(TestbedError):
    """临界值不可估计 / 配置硬失败（无效率超门、有效样本不足）。"""

    __test__ = False


@dataclass(frozen=True)
class ThresholdConfig:
    """阈值生成参数（唯一来源 = 冻结 manifest）。"""

    alpha_train: float = 0.045
    alpha_per_look: float = 0.01125
    invalid_rate_per_look_max: float = 0.10
    min_valid_quantile_sample: int = MIN_VALID_QUANTILE_SAMPLE

    @classmethod
    def from_manifest(cls, doc: Mapping[str, Any]) -> "ThresholdConfig":
        threshold = dict((doc.get("income_referee") or {}).get("threshold") or {})
        return cls(
            alpha_train=float(threshold["alpha_train"]),
            alpha_per_look=float(threshold["alpha_per_look"]),
            invalid_rate_per_look_max=float(threshold["invalid_rate_per_look_max"]),
            min_valid_quantile_sample=int(
                threshold.get("min_valid_quantile_sample", MIN_VALID_QUANTILE_SAMPLE)
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alpha_train": float(self.alpha_train),
            "alpha_per_look": float(self.alpha_per_look),
            "invalid_rate_per_look_max": float(self.invalid_rate_per_look_max),
            "min_valid_quantile_sample": int(self.min_valid_quantile_sample),
        }


def threshold_rule_steps() -> List[str]:
    """五步规则的冻结文本（manifest 只允许与实现一致）。"""
    return list(TRAINING_RULE_STEPS)


@dataclass(frozen=True)
class TrainingResult:
    """训练结果（临界值 + 逐单元中间量 + 硬失败清单）。"""

    thresholds: Tuple[float, ...]
    units: Tuple[Dict[str, Any], ...]
    determining_units: Tuple[str, ...]
    failures: Tuple[Dict[str, Any], ...]
    plan: LookPlan
    statistic: str
    delta: float
    config: ThresholdConfig
    realization_ids: Mapping[str, str]

    @property
    def passed(self) -> bool:
        """是否可作为临界值产物（无硬失败且每个 look 有阈值）。"""
        return not self.failures and all(np.isfinite(self.thresholds))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": "p5b_testbed_calibration_thresholds/v1",
            "implementation_id": IMPLEMENTATION_ID,
            "statistic": self.statistic,
            "delta": self.delta,
            "plan": self.plan.as_dict(),
            "config": self.config.as_dict(),
            "rule_steps": threshold_rule_steps(),
            "c_by_look": [float(value) for value in self.thresholds],
            "determining_units": list(self.determining_units),
            "units": [dict(unit) for unit in self.units],
            "failures": [dict(failure) for failure in self.failures],
            "realization_ids": dict(sorted(self.realization_ids.items())),
        }


def generate_scores(
    registry: SceneRegistry,
    scene: str,
    *,
    seed: int,
    stage: str,
    null: str,
    mu: float,
    count: int,
    plan: LookPlan,
    statistic: str = STATISTIC_PSD,
    delta: float = DELTA_LOG_DAILY,
    prefix: str = "",
    level: float = 0.0,
    chunk: Optional[int] = None,
) -> Tuple[ScoreBlock, str]:
    """生成一个单元的路径并逐 look 打分（分块生成，结果与块大小无关）。

    Args:
        registry: 场景注册表（其 ``pool_role`` 决定用哪个池）。
        scene: 场景名。
        seed: 命名种子根。
        stage: ``train`` / ``holdout`` / ``effect``。
        null: 零假设名（``mu=0`` / ``mu=-delta``；效应档传空串）。
        mu: 日度 log 位移（零假设用 ``0`` / ``-delta``；效应档用 ``-level*delta``）。
        count: 路径数。
        plan: look 映射。
        statistic: 注册统计量。
        delta: 日度 log 基准位移。
        prefix: realization 前缀（manifest 冻结值）。
        level: 效应档（δ 倍数；进入子流编码，**必须**传入才能让各档独立）。
        chunk: 每批路径数（``None`` = 单批）。

    Returns:
        ``(ScoreBlock, realization_id)``。
    """
    item = substream(seed, stage, scene, null=null, level=float(level), prefix=prefix)
    total = int(count)
    if total <= 0:
        raise ThresholdError(f"路径数必须为正: {count!r}")
    step = int(chunk) if chunk else total
    blocks: List[np.ndarray] = []
    reasons: List[Tuple[Optional[str], ...]] = []
    produced = 0
    while produced < total:
        size = min(step, total - produced)
        paths = registry.build(scene, item.generator, size) + float(mu)
        scores = score_paths(paths, plan, statistic=statistic, delta=delta)
        blocks.append(scores.values)
        reasons.extend(scores.invalid_reason)
        produced += size
    merged = ScoreBlock(
        values=np.vstack(blocks),
        invalid_reason=tuple(reasons),
        statistic=statistic,
        delta=float(delta),
    )
    return merged, item.realization_id


def _quantile_unit(scores: ScoreBlock, look_index: int, config: ThresholdConfig) -> Dict[str, Any]:
    """单个单元单 look 的下尾分位中间量（无效样本不进入分位、但计入无效率）。"""
    values = scores.values[:, look_index]
    valid = values[np.isfinite(values)]
    invalid_rate = float(1.0 - valid.size / values.size)
    detail: Dict[str, Any] = {
        "valid_count": int(valid.size),
        "total": int(values.size),
        "invalid_rate": invalid_rate,
        "invalid_reasons": scores.reason_counts(look_index),
        "quantile": None,
        "c": None,
        "eligible": False,
        "rejection": None,
        # 分位的有效超越点数量级审计（2000 × 1.125% ≈ 22.5 个）：c_k 由第 ~23 个次序统计量决定，
        # 属经验分位而非解析尾分位；登记该量以便读者判断 c_k 的抽样不确定性
        "expected_exceedances": float(config.alpha_per_look) * float(values.size),
        "order_statistic_index": max(1, int(math.ceil(config.alpha_per_look * float(values.size)))),
    }
    if invalid_rate > config.invalid_rate_per_look_max:
        detail["rejection"] = "invalid_rate_exceeds_max"
        return detail
    if valid.size < config.min_valid_quantile_sample:
        detail["rejection"] = "insufficient_valid_sample"
        return detail
    quantile = float(np.quantile(valid, config.alpha_per_look))
    detail["quantile"] = quantile
    detail["c"] = -quantile
    detail["eligible"] = True
    return detail


def _guard_training_inputs(
    plan: LookPlan, registry: SceneRegistry, config: ThresholdConfig
) -> None:
    """训练前的不变式（角色/alpha 分配/冻结门；抽取自 ``train_income_thresholds``，行为不变）。"""
    if registry.pool_role != "train":
        raise ThresholdError(
            f"训练必须使用训练池（pool_role='train'），实际 {registry.pool_role!r}"
            "（设计 §5.3：训练只用于选 c_k，不得使用 holdout）"
        )
    if config.alpha_per_look > 0 and not np.isclose(
        config.alpha_per_look, config.alpha_train / plan.k, rtol=1e-12, atol=0.0
    ):
        raise ThresholdError(
            f"alpha_per_look({config.alpha_per_look}) 必须等于 alpha_train/K"
            f"({config.alpha_train}/{plan.k})"
        )
    # 与 power/holdout 同口径：两个门都是冻结值，禁止新调用方自行放宽（否则 10% 无效率门
    # 与 ≥100 有效样本门可被静默废掉，训练侧会"通过"一份不可信临界值）
    if not math.isclose(float(config.invalid_rate_per_look_max), 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise ThresholdError(
            f"无效率门必须等于冻结值 0.10，实际 {config.invalid_rate_per_look_max}"
        )
    if int(config.min_valid_quantile_sample) != 100:
        raise ThresholdError(
            f"最小有效分位样本必须等于冻结值 100，实际 {config.min_valid_quantile_sample}"
        )


def train_income_thresholds(
    *,
    plan: LookPlan,
    registry: SceneRegistry,
    seed: int,
    counts: Mapping[str, int],
    config: ThresholdConfig,
    prefix: str = "",
    statistic: str = STATISTIC_PSD,
    delta: float = DELTA_LOG_DAILY,
    chunk: Optional[int] = None,
) -> TrainingResult:
    """执行 §5.2 五步训练，返回临界值与逐组中间量。

    Args:
        plan: look 映射。
        registry: **训练池**场景注册表（``pool_role`` 必须是 ``train``）。
        seed: 命名种子根。
        counts: ``{"train_per_unit": int}``。
        config: 阈值参数。
        prefix: realization 前缀。
        statistic: 注册统计量。
        delta: 日度 log 基准。
        chunk: 分块路径数。

    Returns:
        :class:`TrainingResult`（``passed=False`` 时仍完整给出中间量与失败原因，供审计）。

    Raises:
        ThresholdError: 注册表角色非 ``train``（角色断言；配合互斥源折构造与产物只写一次，
            共同保证训练不消费 holdout —— 这是契约保证，不是编译期不可表达）。
    """
    _guard_training_inputs(plan, registry, config)
    count = int(counts["train_per_unit"])
    units: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    realization_ids: Dict[str, str] = {}
    look_details: Dict[int, List[Dict[str, Any]]] = {index: [] for index in range(plan.k)}
    for scene in registry.names():
        for null, scale in NULL_SPECS:
            mu = 0.0 if null == "mu=0" else float(scale) * float(delta)
            scores, realization_id = generate_scores(
                registry,
                scene,
                seed=seed,
                stage="train",
                null=null,
                mu=mu,
                count=count,
                plan=plan,
                statistic=statistic,
                delta=delta,
                prefix=prefix,
                chunk=chunk,
            )
            unit_key = f"{scene}|{null}"
            realization_ids[unit_key] = realization_id
            per_look: List[Dict[str, Any]] = []
            for index, j in enumerate(plan.j_looks):
                detail = _quantile_unit(scores, index, config)
                detail = {"J": int(j), "look_days": int(plan.look_days[index]), **detail}
                per_look.append(detail)
                look_details[index].append({"unit": unit_key, **detail})
                if detail["rejection"] is not None:
                    failures.append(
                        {
                            "stage": "train",
                            "unit": unit_key,
                            "look": index,
                            "J": int(j),
                            "reason": detail["rejection"],
                            "invalid_rate": detail["invalid_rate"],
                            "valid_count": detail["valid_count"],
                        }
                    )
            units.append(
                {
                    "unit": unit_key,
                    "scene": scene,
                    "null": null,
                    "mu": float(mu),
                    "realization_id": realization_id,
                    "looks": per_look,
                }
            )
    thresholds: List[float] = []
    determining: List[str] = []
    for index in range(plan.k):
        eligible = [item for item in look_details[index] if item["eligible"]]
        if not eligible:
            thresholds.append(float("nan"))
            determining.append("")
            failures.append(
                {
                    "stage": "train",
                    "unit": None,
                    "look": index,
                    "reason": "threshold_not_estimable",
                    "detail": "该 look 无任何可用单元（禁止用极值替代）",
                }
            )
            continue
        best = max(eligible, key=lambda item: float(item["c"]))
        value = float(best["c"])
        if not np.isfinite(value) or value <= 0:
            # 非正临界值 = 该 look 的下尾已越过边界零点 ⇒ 无法作为否决边界（且成因通常是
            # 退化路径的浮点残余方差）。这里**判配置失败**，绝不截断/取绝对值（禁止 clamp）。
            thresholds.append(float("nan"))
            determining.append("")
            failures.append(
                {
                    "stage": "train",
                    "unit": str(best["unit"]),
                    "look": index,
                    "reason": "threshold_not_positive",
                    "detail": (
                        f"该 look 的最大单元 c={value!r} 非正（退化路径或不可分边界），"
                        "按配置失败处理，不得用作否决边界"
                    ),
                }
            )
            continue
        thresholds.append(value)
        determining.append(str(best["unit"]))
    return TrainingResult(
        thresholds=tuple(thresholds),
        units=tuple(units),
        determining_units=tuple(determining),
        failures=tuple(failures),
        plan=plan,
        statistic=statistic,
        delta=float(delta),
        config=config,
        realization_ids=realization_ids,
    )
