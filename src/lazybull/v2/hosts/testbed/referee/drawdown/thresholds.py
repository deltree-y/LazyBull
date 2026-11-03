# -*- coding: utf-8 -*-
"""P5b T5 两个回撤裁判的临界值（设计 v4 §5.4.1/§5.4.2）。

两个裁判、两组阈值、两个事件（**分别命名，不得互相抵扣**）
=======================================================
- ``DD-W`` 恶化否决：训练集只用 **A/A 路径**估 ``D_k`` 下尾
  ``c^W_{k,scene}=q_{alpha_train/K}(D_k)``，取最保守 ``c^W_k = min_scene c^W_{k,scene}``；
  拒绝事件 ``D_k < c^W_k``，记名 :data:`WORSEN_EVENT`；
- ``DD-I`` 改善检出：``I_k = D_k``（**不得再对 D 取负**），训练集只用 A/A 路径估上尾
  ``c^I_{k,scene}=q_{1-alpha_train/K}(I_k)``，取最保守 ``c^I_k = max_scene c^I_{k,scene}``；
  检出事件 ``I_k > c^I_k``，记名 :data:`IMPROVE_EVENT`（只报告检出能力，不产生转正建议）。

方向自检（§5.4.2 末）
=====================
同一组 A/A 读数下必须满足 ``c^I_k > c^W_k``；这是**必要非充分**条件：两个事件都是 family-wise
（任一 look 触发），"look0 恶化否决 + look3 改善检出"在数值上仍可能同真。因此生产路径由
:func:`event_overlap_count` 在 holdout 读数上**实际计数**并落报告字段（披露、不阻断，阻断会
改变已冻结的单元口径），方向正确性另由两个确定性 fixture 证明。训练侧补一条位置检查：A/A 的
逐 look 中位数必须落在 ``(c^W_k, c^I_k)`` 内部（阈值退化或方向写反时该检查先失败）。

"A/A 不得用同一路径零注入"的落点
================================
A/A 读数由调用方传入（:class:`AaReadings`），其构造入口
:func:`aa_readings_from_independent_paths` 只接受**两条不同路径矩阵**（arm/base）；
同一数组传入即报错，结构上排除「同路径零注入」冒充 A/A。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    baseline_percent_by_look,
    cumulative_log_path,
    log_to_percent,
    max_drawdown_log,
)
from src.lazybull.v2.hosts.testbed.referee.statistics import LookPlan

__all__ = [
    "ALPHA_TRAIN",
    "AaReadings",
    "DrawdownThresholdError",
    "DrawdownThresholds",
    "IMPROVE_EVENT",
    "MIN_VALID_QUANTILE_SAMPLE",
    "QUANTILE_INTERPOLATION",
    "WORSEN_EVENT",
    "aa_readings_from_independent_paths",
    "aa_summary",
    "assert_events_disjoint",
    "direction_self_check",
    "event_overlap_count",
    "improve_event",
    "merge_aa_readings",
    "reading_matrix",
    "threshold_section",
    "train_drawdown_thresholds",
    "worsen_event",
]

#: 训练侧总误判率（family-wise；与收益裁判同口径）
ALPHA_TRAIN = 0.045
#: 有效分位样本下限（< 该值即配置失败，禁止用极值替代）
MIN_VALID_QUANTILE_SAMPLE = 100
#: 分位插值口径（与收益裁判一致；冻结写入 manifest）
QUANTILE_INTERPOLATION = "linear"
#: DD-W 拒绝事件（机器可读记名）
WORSEN_EVENT = "vetoed_by_calibrated_drawdown_worsening"
#: DD-I 检出事件（机器可读记名）
IMPROVE_EVENT = "detected_by_calibrated_drawdown_improvement"


class DrawdownThresholdError(TestbedError):
    """临界值不可估计或方向自检失败（一律 fail-closed）。"""

    __test__ = False


@dataclass(frozen=True)
class AaReadings:
    """一个场景的 A/A 读数（``(M, K)`` 的 ``D_k``，pp；arm/base 为独立路径）。"""

    scene: str
    d_matrix: np.ndarray
    realization_id: str

    def __post_init__(self) -> None:
        values = np.asarray(self.d_matrix, dtype=float)
        if values.ndim != 2:
            raise DrawdownThresholdError(f"A/A 读数必须是二维 (M, K)，实际 shape={values.shape}")
        if values.shape[0] < 1:
            raise DrawdownThresholdError("A/A 读数至少需要一条实现")

    @property
    def samples(self) -> int:
        return int(np.asarray(self.d_matrix).shape[0])

    @property
    def looks(self) -> int:
        return int(np.asarray(self.d_matrix).shape[1])


def reading_matrix(arm: np.ndarray, baseline_pct: np.ndarray, plan: LookPlan) -> np.ndarray:
    """``D_k = MDD_arm(T_k) - MDD_base(T_k)`` 的 ``(M, K)`` 读数矩阵（pp）。

    ``baseline_pct`` 是**基准臂**（B0 冻结窗口）在各 look 上的 MDD（pp，见
    ``mdd.baseline_percent_by_look``）：A/A 与效应单元共用同一组基准读数，因此读数差里
    **保留噪声**（设计 §5.4.4「观察量保留噪声」），纯注入效应另由求解器给出。

    Raises:
        DrawdownThresholdError: 形状/look 数不符。
    """
    values = np.asarray(arm, dtype=float)
    base = np.asarray(baseline_pct, dtype=float).reshape(-1)
    if values.ndim != 2:
        raise DrawdownThresholdError(f"arm 必须是二维 (M, N)，实际 shape={values.shape}")
    if base.size != int(plan.k):
        raise DrawdownThresholdError(f"基准读数个数({base.size}) 必须等于 K({plan.k})")
    cumulative = cumulative_log_path(values)
    columns = []
    for index, day in enumerate(plan.look_days):
        stop = int(day)
        if stop >= cumulative.shape[1]:
            raise DrawdownThresholdError(
                f"look 终点 {stop} 超出路径长度 {cumulative.shape[1] - 1}（禁止截断）"
            )
        raw = max_drawdown_log(cumulative[:, : stop + 1])
        columns.append(log_to_percent(raw) - base[index])
    return np.column_stack(columns)


def aa_readings_from_independent_paths(
    arm: np.ndarray,
    base: np.ndarray,
    plan: LookPlan,
    *,
    scene: str,
    realization_id: str = "",
) -> AaReadings:
    """构造 A/A 读数：``arm`` 为**独立**噪声路径，``base`` 为冻结基准路径。

    - ``base`` 允许 ``(N,)``（单条基准，按行广播）或 ``(M, N)``（逐实现基准）；
    - **禁止** ``arm`` 与 ``base`` 相同（同一路径零注入会让 ``D_k ≡ 0``，A/A 退化）；
    - ``arm`` 必须来自独立抽取（调用方保证；本函数结构上拒绝恒等情形）。

    Raises:
        DrawdownThresholdError: 形状不符或 ``arm`` 与 ``base`` 恒等（禁止零注入冒充 A/A）。
    """
    first = np.asarray(arm, dtype=float)
    second = np.asarray(base, dtype=float)
    if first.ndim != 2:
        raise DrawdownThresholdError(f"arm 必须是二维 (M, N)，实际 shape={first.shape}")
    if first is second:
        raise DrawdownThresholdError("A/A 必须使用独立路径（禁止同一路径零注入冒充 A/A）")
    if second.ndim == 1:
        if second.size != first.shape[1]:
            raise DrawdownThresholdError(
                f"基准路径长度 {second.size} 与 arm 天数 {first.shape[1]} 不符"
            )
        if np.allclose(first, second[None, :]):
            raise DrawdownThresholdError(
                "A/A 的 arm 与基准完全相同（零注入）⇒ D_k ≡ 0，禁止作为 A/A"
            )
        baseline = baseline_percent_by_look(second, plan.look_days)
    elif second.shape == first.shape:
        if np.array_equal(first, second):
            raise DrawdownThresholdError(
                "A/A 必须使用同分布**独立**路径（禁止同一路径零注入冒充 A/A）"
            )
        # 逐 look 的 horizon 守卫：短路径不得被越界切片静默当作足长 look（设计 §5.4.4）
        longest = int(max(plan.look_days))
        if first.shape[1] < longest:
            raise DrawdownThresholdError(
                f"路径收益数 {first.shape[1]} 小于最长 look {longest}（禁止把短路径当作足长 horizon）"
            )
        # 逐实现配对（设计 v4 §5.4.1/§5.4.2 的真 A/A）：D[i, k] = MDD(a_i,k) − MDD(b_i,k)
        arm_mdd = np.column_stack(
            [
                log_to_percent(max_drawdown_log(cumulative_log_path(first)[:, : int(day) + 1]))
                for day in plan.look_days
            ]
        )
        base_mdd = np.column_stack(
            [
                log_to_percent(max_drawdown_log(cumulative_log_path(second)[:, : int(day) + 1]))
                for day in plan.look_days
            ]
        )
        return AaReadings(
            scene=str(scene),
            d_matrix=arm_mdd - base_mdd,
            realization_id=str(realization_id),
        )
    else:
        raise DrawdownThresholdError(
            f"base 必须是 (N,) 或与 arm 同形，实际 {second.shape} vs {first.shape}"
        )
    return AaReadings(
        scene=str(scene),
        d_matrix=reading_matrix(first, baseline, plan),
        realization_id=str(realization_id),
    )


@dataclass(frozen=True)
class DrawdownThresholds:
    """两个裁判的临界值（``c^W_k`` / ``c^I_k``）+ 逐场景中间量（可审计）。"""

    plan: LookPlan
    alpha_train: float
    alpha_per_look: float
    worsen: Tuple[float, ...]
    improve: Tuple[float, ...]
    worsen_by_scene: Mapping[str, Tuple[float, ...]]
    improve_by_scene: Mapping[str, Tuple[float, ...]]
    worsen_units: Tuple[Dict[str, Any], ...]
    improve_units: Tuple[Dict[str, Any], ...]
    determining_worsen: Tuple[str, ...]
    determining_improve: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": "p5b_testbed_drawdown_thresholds/v1",
            "plan": self.plan.as_dict(),
            "alpha_train": float(self.alpha_train),
            "alpha_per_look": float(self.alpha_per_look),
            "quantile_interpolation": QUANTILE_INTERPOLATION,
            "c_w_by_look": [float(value) for value in self.worsen],
            "c_i_by_look": [float(value) for value in self.improve],
            "c_w_by_scene": {
                key: [float(v) for v in value] for key, value in self.worsen_by_scene.items()
            },
            "c_i_by_scene": {
                key: [float(v) for v in value] for key, value in self.improve_by_scene.items()
            },
            "determining_units_worsen": list(self.determining_worsen),
            "determining_units_improve": list(self.determining_improve),
            "units_worsen": [dict(item) for item in self.worsen_units],
            "units_improve": [dict(item) for item in self.improve_units],
            "events": {"worsen": WORSEN_EVENT, "improve": IMPROVE_EVENT},
            "direction_check": {
                "rule": "c^I_k > c^W_k（同一组 A/A 读数）",
                "passed": not direction_self_check(self.worsen, self.improve),
            },
        }


def merge_aa_readings(items: Sequence[AaReadings]) -> AaReadings:
    """把同一场景的多个分块读数纵向合并（每块各自已过独立路径断言）。"""
    blocks = list(items)
    if not blocks:
        raise DrawdownThresholdError("A/A 分块为空（不得用空读数估分位）")
    scenes = {item.scene for item in blocks}
    if len(scenes) != 1:
        raise DrawdownThresholdError(f"分块必须属于同一场景: {sorted(scenes)}")
    looks = {item.looks for item in blocks}
    if len(looks) != 1:
        raise DrawdownThresholdError(f"分块 look 数必须一致: {sorted(looks)}")
    realization_ids = {item.realization_id for item in blocks}
    if len(realization_ids) != 1:
        raise DrawdownThresholdError(
            f"分块 realization_id 必须一致（禁止混用不同子流）: {sorted(realization_ids)}"
        )
    return AaReadings(
        scene=blocks[0].scene,
        d_matrix=np.vstack([np.asarray(item.d_matrix, dtype=float) for item in blocks]),
        realization_id=blocks[0].realization_id,
    )


def aa_summary(readings: Sequence[AaReadings]) -> List[Dict[str, Any]]:
    """逐场景 A/A 读数的位置量（零假设偏移的披露材料，§5.4.4「保留噪声」）。

    ``MDD`` 是路径的非线性泛函 ⇒ 固定基准臂 + 单侧加噪会让 ``D_k`` 的中位数系统性偏负；
    该偏移直接决定 ``c^W_k`` 的位置与 ``DD-W`` 的功效，必须与阈值一起报告。
    """
    rows: List[Dict[str, Any]] = []
    for item in readings:
        matrix = np.asarray(item.d_matrix, dtype=float)
        alpha = ALPHA_TRAIN / max(item.looks, 1)
        rows.append(
            {
                "scene": item.scene,
                "samples": int(matrix.shape[0]),
                "median_by_look": [float(np.median(matrix[:, i])) for i in range(item.looks)],
                "mean_by_look": [float(np.mean(matrix[:, i])) for i in range(item.looks)],
                "std_by_look": [float(np.std(matrix[:, i], ddof=1)) for i in range(item.looks)],
                "q_lower_by_look": [
                    float(np.quantile(matrix[:, i], alpha)) for i in range(item.looks)
                ],
                "q_upper_by_look": [
                    float(np.quantile(matrix[:, i], 1.0 - alpha)) for i in range(item.looks)
                ],
                "realization_id": item.realization_id,
                "pool_note": "训练池与 holdout 池源折互斥，噪声尺度可能不同（逐场景对照见本表）",
            }
        )
    return rows


def event_overlap_count(
    d_matrix: np.ndarray, worsen: Sequence[float], improve: Sequence[float]
) -> int:
    """同一读数上"恶化否决"与"改善检出"**同时成立**的实现数（family-wise 事件）。

    单 look 互斥（``c^W_k < c^I_k``），但任一 look 触发的 family-wise 事件跨 look 可能同真
    （look0 触发否决、look3 触发检出）。此计数是**披露量**，不用于阻断单元判定。
    """
    low = worsen_event(d_matrix, worsen)
    high = improve_event(d_matrix, improve)
    return int(np.sum(low & high))


def _quantile(values: np.ndarray, probability: float) -> float:
    ordered = np.asarray(values, dtype=float)
    valid = ordered[np.isfinite(ordered)]
    if valid.size < MIN_VALID_QUANTILE_SAMPLE:
        raise DrawdownThresholdError(
            f"有效 A/A 样本 {valid.size} 少于分位下限 {MIN_VALID_QUANTILE_SAMPLE}"
            "（分位不可估计，禁止用极值替代）"
        )
    return float(np.quantile(valid, float(probability), method=QUANTILE_INTERPOLATION))


def _check_median_inside(
    medians_by_scene: Mapping[str, Sequence[float]],
    worsen: Mapping[str, Sequence[float]],
    improve: Mapping[str, Sequence[float]],
) -> None:
    """必要非充分条件的位置检查：A/A 中位数必须落在 ``(c^W_k, c^I_k)`` 内部。

    阈值退化（分位塌到同一点）或把 ``I_k`` 写成 ``−D_k`` 时该项先失败；跨 look 的事件重合
    另由 :func:`event_overlap_count` 在 holdout 读数上披露。

    Raises:
        DrawdownThresholdError: 任一场景任一 look 的中位数不在区间内部。
    """
    for scene, medians in medians_by_scene.items():
        lower = worsen[scene]
        upper = improve[scene]
        for index, median in enumerate(medians):
            if not (float(lower[index]) < float(median) < float(upper[index])):
                raise DrawdownThresholdError(
                    f"方向自检失败：场景 {scene!r} look {index} 的 A/A 中位数 {median} 不在其自身 "
                    f"(c^W={lower[index]}, c^I={upper[index]}) 内部（阈值退化或方向写反）"
                )


def train_drawdown_thresholds(
    readings: Sequence[AaReadings],
    plan: LookPlan,
    *,
    alpha_train: float = ALPHA_TRAIN,
) -> DrawdownThresholds:
    """按 §5.4.1/§5.4.2 生成两组临界值（只读 A/A 读数，不读 holdout/效应）。

    Args:
        readings: 逐场景 A/A 读数（场景集合必须与调用方冻结值一致）。
        plan: look 映射（K 决定 ``alpha_per_look = alpha_train / K``）。
        alpha_train: 训练侧总误判率（冻结 0.045）。

    Raises:
        DrawdownThresholdError: 场景为空/重复、look 数不符、样本不足、方向自检失败。
    """
    items = list(readings)
    if not items:
        raise DrawdownThresholdError("A/A 读数为空（阈值不可估计）")
    scenes = [item.scene for item in items]
    if len(set(scenes)) != len(scenes):
        raise DrawdownThresholdError(f"场景重复（禁止重复计权）: {scenes}")
    alpha = float(alpha_train)
    if not (0.0 < alpha < 1.0):
        raise DrawdownThresholdError(f"alpha_train 必须落在 (0,1)，实际 {alpha}")
    alpha_look = alpha / int(plan.k)
    root_worsen: Dict[str, List[float]] = {}
    root_improve: Dict[str, List[float]] = {}
    root_median: Dict[str, List[float]] = {}
    worsen_units: List[Dict[str, Any]] = []
    improve_units: List[Dict[str, Any]] = []
    for item in items:
        if item.looks != int(plan.k):
            raise DrawdownThresholdError(
                f"场景 {item.scene!r} 的 A/A 读数 look 数 {item.looks} 与 K={plan.k} 不符"
            )
        lower: List[float] = []
        upper: List[float] = []
        medians: List[float] = []
        for index in range(int(plan.k)):
            column = np.asarray(item.d_matrix, dtype=float)[:, index]
            low = _quantile(column, alpha_look)
            high = _quantile(column, 1.0 - alpha_look)
            lower.append(low)
            upper.append(high)
            medians.append(float(np.median(column)))
            worsen_units.append(
                {
                    "unit": f"{item.scene}|dd_worsen",
                    "scene": item.scene,
                    "look": index,
                    "look_days": int(plan.look_days[index]),
                    "quantile_probability": alpha_look,
                    "quantile": low,
                    "c": low,
                    "samples": int(column.size),
                    "realization_id": item.realization_id,
                }
            )
            improve_units.append(
                {
                    "unit": f"{item.scene}|dd_improve",
                    "scene": item.scene,
                    "look": index,
                    "look_days": int(plan.look_days[index]),
                    "quantile_probability": 1.0 - alpha_look,
                    "quantile": high,
                    "c": high,
                    "samples": int(column.size),
                    "realization_id": item.realization_id,
                }
            )
        root_worsen[item.scene] = lower
        root_improve[item.scene] = upper
        root_median[item.scene] = medians
    worsen: List[float] = []
    improve: List[float] = []
    determining_worsen: List[str] = []
    determining_improve: List[str] = []
    for index in range(int(plan.k)):
        worst = min(scenes, key=lambda scene: root_worsen[scene][index])
        best = max(scenes, key=lambda scene: root_improve[scene][index])
        worsen.append(float(root_worsen[worst][index]))
        improve.append(float(root_improve[best][index]))
        determining_worsen.append(f"{worst}|dd_worsen")
        determining_improve.append(f"{best}|dd_improve")
    if not np.isfinite(worsen).all() or not np.isfinite(improve).all():
        raise DrawdownThresholdError("临界值含非有限值（分位不可用，禁止 clamp）")
    errors = direction_self_check(worsen, improve)
    if errors:
        raise DrawdownThresholdError("方向自检失败（c^I_k > c^W_k 必须成立）: " + "; ".join(errors))
    _check_median_inside(root_median, root_worsen, root_improve)
    return DrawdownThresholds(
        plan=plan,
        alpha_train=alpha,
        alpha_per_look=alpha_look,
        worsen=tuple(worsen),
        improve=tuple(improve),
        worsen_by_scene={scene: tuple(values) for scene, values in root_worsen.items()},
        improve_by_scene={scene: tuple(values) for scene, values in root_improve.items()},
        worsen_units=tuple(worsen_units),
        improve_units=tuple(improve_units),
        determining_worsen=tuple(determining_worsen),
        determining_improve=tuple(determining_improve),
    )


def _family_flags(d_matrix: np.ndarray, limits: Sequence[float], *, below: bool) -> np.ndarray:
    values = np.asarray(d_matrix, dtype=float)
    bounds = np.asarray([float(item) for item in limits], dtype=float)
    if values.ndim != 2:
        raise DrawdownThresholdError(f"D 读数必须是二维 (M, K)，实际 shape={values.shape}")
    if values.shape[1] != bounds.size:
        raise DrawdownThresholdError(
            f"临界值个数({bounds.size}) 与 look 数({values.shape[1]}) 不符"
        )
    if not np.isfinite(bounds).all():
        raise DrawdownThresholdError(f"临界值必须有限: {bounds.tolist()}")
    if below:
        hits = np.isfinite(values) & (values < bounds[None, :])
    else:
        hits = np.isfinite(values) & (values > bounds[None, :])
    return np.any(hits, axis=1)


def worsen_event(d_matrix: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """``DD-W`` family-wise 拒绝事件：``any_k(D_k < c^W_k)``。"""
    return _family_flags(d_matrix, thresholds, below=True)


def improve_event(d_matrix: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """``DD-I`` family-wise 检出事件：``any_k(I_k = D_k > c^I_k)``（不取负）。"""
    return _family_flags(d_matrix, thresholds, below=False)


def assert_events_disjoint(
    d_matrix: np.ndarray, worsen: Sequence[float], improve: Sequence[float]
) -> None:
    """两个事件在同一组读数上不得重合（``c^W < c^I`` 的直接后果）。

    Raises:
        DrawdownThresholdError: 存在同时"恶化否决"与"改善检出"的实现。
    """
    low = worsen_event(d_matrix, worsen)
    high = improve_event(d_matrix, improve)
    overlap = int(np.sum(low & high))
    if overlap:
        raise DrawdownThresholdError(
            f"恶化否决与改善检出事件重合 {overlap} 条（两裁判互相抵扣，禁止）"
        )


def direction_self_check(worsen: Sequence[float], improve: Sequence[float]) -> List[str]:
    """方向自检：逐 look 必须 ``c^I_k > c^W_k``（同一组 A/A 读数）。

    Returns:
        错误列表（``[]`` = 通过）。
    """
    lower = [float(item) for item in worsen]
    upper = [float(item) for item in improve]
    if len(lower) != len(upper):
        return [f"两组临界值个数不符: {len(lower)} vs {len(upper)}"]
    return [
        f"look {index}: c^I={upper[index]!r} 未严格大于 c^W={lower[index]!r}"
        for index in range(len(lower))
        if not upper[index] > lower[index]
    ]


def threshold_section(plan: LookPlan) -> Dict[str, Any]:
    """manifest 的阈值/事件冻结段（§5.4.1 + §5.4.2 + §5.5）。"""
    return {
        "alpha_train": float(ALPHA_TRAIN),
        "alpha_per_look": float(ALPHA_TRAIN) / int(plan.k),
        "quantile_interpolation": QUANTILE_INTERPOLATION,
        "min_valid_quantile_sample": int(MIN_VALID_QUANTILE_SAMPLE),
        "worsen_rule": (
            "c^W_{k,scene}=q_{alpha_train/K}(D_k)（A/A 下尾），c^W_k = min_scene c^W_{k,scene}"
        ),
        "improve_rule": (
            "c^I_{k,scene}=q_{1-alpha_train/K}(I_k)（A/A 上尾），c^I_k = max_scene c^I_{k,scene}"
        ),
        "worsen_conservative": "min over scene（最保守）",
        "improve_conservative": "max over scene（最保守）",
        "worsen_event": WORSEN_EVENT,
        "improve_event": IMPROVE_EVENT,
        "worsen_condition": "D_k < c^W_k（family-wise：任一 look 触发）",
        "improve_condition": "I_k = D_k > c^I_k（family-wise：任一 look 触发）",
        "improve_is_d_k": True,
        "improve_negation_forbidden": True,
        "no_cross_offsetting": True,
        "direction_check": "同一组 A/A 读数下 c^I_k > c^W_k，且两事件不重合",
        "no_holdout_callback": True,
        "aa_rule": "训练集只用 A/A 路径（同分布独立路径；禁止同一路径零注入）",
    }
