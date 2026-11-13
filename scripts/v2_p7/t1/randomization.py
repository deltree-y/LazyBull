# -*- coding: utf-8 -*-
"""P7 T1 同域内层随机化、单侧临界值、误放行率与功效（设计 §6.5）。

零假设构造（冻结）
==================
1. 对每个匹配组，六股在共同入离场后的**净残差向量**固定；组内事件标签随机化即从六股中
   均匀抽一股为伪事件、其余五股为伪对照，保留原协变量、摩擦、价源与公司行为；
2. 日期按 h 日 **circular moving block** 重采样（块长 = 持有期 h，命名
   ``randomization_block_h``），同一被抽日期的全部组一起进入；每个组独立置换标签；
3. 单元统计量 = 全部**意图**上净残差的加权均值（未成交组按设计取真实 0 残差参与，
   因此与 `intent_total` 同一分母）。

三个互斥种子流
==============
- 校准流（prereg 冻结的 42）：只用于生成组件临界值 ``Q_{1-0.05/3}(D^0)``；
- 评估流（固定偏移种子）：只用于估计分量门误放行率与其 Clopper–Pearson 上界；
- 功效流（另一固定偏移种子）：只用于读出 ``δ_R·σ̄_{D^0}`` 注入下的检出率与 MDE。

功效注入等价性：给每个事件侧组净残差同加 ``δ_R·σ̄`` 使统计量恰好平移 ``δ_R·σ̄``，
因此 ``P(D + δσ > c) = P(D > c − δσ)``；实现仍按注入后的样本直接比较，避免解析近似。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # 正常路径使用精确二项上界
    from scipy.stats import beta as _beta
except ImportError:  # pragma: no cover - scipy 为项目依赖，仅作降级保护
    _beta = None

MEMBER_COUNT = 6
DEFAULT_CHUNK = 32


@dataclass
class NullDistribution:
    """一次零假设随机化的结果。"""

    values: np.ndarray
    dates: int
    groups: int
    implementations: int
    block_h: int
    seed: int
    chunk: int


@dataclass
class CellStatistics:
    """单元统计量（观察值）与两个口径的样本量。"""

    statistic_intent: Optional[float]
    statistic_filled_only: Optional[float]
    intents: int
    resolved: int
    unfilled: int


def observed_statistic(
    intent_residual: np.ndarray,
    resolved_mask: np.ndarray,
    unfilled_mask: Optional[np.ndarray] = None,
) -> CellStatistics:
    """全部意图口径与仅成交口径的观察统计量。

    ``intent_residual`` 对不可解析组为 NaN、未成交组为真实 0；只要存在 NaN，
    意图口径统计量不成立（返回 None，由 verdict 判为冻结）。
    """
    values = np.asarray(intent_residual, dtype=np.float64)
    resolved = np.asarray(resolved_mask, dtype=bool)
    unfilled = (
        np.asarray(unfilled_mask, dtype=bool)
        if unfilled_mask is not None
        else (np.isfinite(values) & ~resolved)
    )
    intents = int(values.size)
    n_resolved = int(resolved.sum())
    n_unfilled = int(unfilled.sum())
    statistic_intent = float(values.mean()) if intents and np.isfinite(values).all() else None
    statistic_filled = float(values[resolved].mean()) if n_resolved else None
    return CellStatistics(statistic_intent, statistic_filled, intents, n_resolved, n_unfilled)


def null_distribution(
    member_residuals: np.ndarray,
    group_date: np.ndarray,
    *,
    block_h: int,
    implementations: int,
    seed: int,
    day_span: Optional[Tuple[int, int]] = None,
    chunk: int = DEFAULT_CHUNK,
) -> NullDistribution:
    """circular moving block 重采样 + 组内标签独立随机化的零假设分布。

    - **块轴 = 冻结评价窗的完整交易日轴**（`day_span` 闭区间），不是「有事件的日子」：
      无事件日同样占位，避免把相隔多日的两块当成相邻 h 日块（T1 评审 R6）；
    - **每个被抽日期实例独立置换标签**：同一日期被抽中 c 次时，该日全部组各抽 c 组
      独立标签，等价于对 c 个多重伯努利计数取 `Multinomial(c, 1/6×6)`（T1 评审 R7）。
    """
    matrix = np.asarray(member_residuals, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != MEMBER_COUNT:
        raise ValueError(f"净残差矩阵必须为 (n, 6)：{matrix.shape}")
    group_date = np.asarray(group_date, dtype=np.int64)
    n_groups = int(matrix.shape[0])
    if day_span is not None:
        start, end = int(day_span[0]), int(day_span[1])
    elif group_date.size:
        start, end = int(group_date.min()), int(group_date.max())
    else:
        start, end = 0, -1
    n_dates = max(end - start + 1, 0)
    if n_dates <= 0 or n_groups == 0 or implementations <= 0:
        return NullDistribution(np.array([]), n_dates, n_groups, 0, block_h, seed, chunk)
    if group_date.min() < start or group_date.max() > end:
        raise ValueError("组日期超出冻结评价窗")
    date_index = group_date - start
    total = matrix.sum(axis=1, dtype=np.float64)
    n_blocks = max(int(np.ceil(n_dates / block_h)), 1)
    offsets = np.arange(block_h, dtype=np.int64)
    generator = np.random.default_rng(int(seed))
    values = np.empty(int(implementations), dtype=np.float64)
    filled = 0
    while filled < int(implementations):
        size = min(int(chunk), int(implementations) - filled)
        starts = generator.integers(0, n_dates, size=(size, n_blocks), dtype=np.int32)
        index = (starts[:, :, None].astype(np.int64) + offsets[None, None, :]) % n_dates
        resampled = index.reshape(size, n_blocks * block_h)[:, :n_dates]
        counts = np.zeros((size, n_dates), dtype=np.int16)
        for row in range(size):
            counts[row] = np.bincount(resampled[row], minlength=n_dates).astype(np.int16)
        multiplicity = counts[:, date_index].astype(np.int16)
        labels = generator.integers(0, MEMBER_COUNT, size=(size, n_groups), dtype=np.int8)
        selected = np.take_along_axis(
            np.broadcast_to(matrix[None, :, :], (size, n_groups, MEMBER_COUNT)),
            labels.astype(np.int64)[:, :, None],
            axis=2,
        )[:, :, 0].astype(np.float64)
        selected[multiplicity == 0] = 0.0
        max_mult = int(multiplicity.max()) if multiplicity.size else 0
        for occurrence in range(2, max_mult + 1):
            rows, columns = np.nonzero(multiplicity >= occurrence)
            if rows.size == 0:
                continue
            extra = generator.integers(0, MEMBER_COUNT, size=rows.size, dtype=np.int64)
            selected[rows, columns] += matrix[columns, extra]
        numerator = (
            MEMBER_COUNT * selected.sum(axis=1)
            - (multiplicity.astype(np.float64) * total[None, :]).sum(axis=1)
        ) / (MEMBER_COUNT - 1)
        denominator = multiplicity.sum(axis=1, dtype=np.float64)
        values[filled : filled + size] = numerator / np.maximum(denominator, 1.0)
        filled += size
    return NullDistribution(values, n_dates, n_groups, int(implementations), block_h, seed, chunk)


def clopper_pearson_upper(failures: int, trials: int, confidence: float = 0.95) -> Optional[float]:
    """二项比例的单侧 Clopper–Pearson 上界。"""
    if trials <= 0:
        return None
    if failures >= trials:
        return 1.0
    if _beta is None:  # pragma: no cover
        return min(1.0, (failures + 1) / trials)
    return float(_beta.ppf(confidence, failures + 1, trials - failures))


def critical_value(values: np.ndarray, quantile: float) -> Optional[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, quantile, method="linear"))


def power_table(
    power_values: np.ndarray,
    critical: Optional[float],
    sigma: Optional[float],
    delta_grid: Sequence[float],
    min_power: float,
) -> Dict[str, Any]:
    """以 `δ_R·σ` 注入的检出率与 MDE（网格耗尽即不可判定）。"""
    values = np.asarray(power_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    payload: Dict[str, Any] = {
        "delta_R_grid": [float(item) for item in delta_grid],
        "injection": "r_event += delta_R * sigma_D0（等价于 D_shifted = D + delta_R*sigma）",
        "min_power": float(min_power),
        "power": {},
        "mde": None,
        "grid_exhausted": False,
    }
    if (
        critical is None
        or sigma is None
        or not np.isfinite(sigma)
        or sigma <= 0
        or values.size == 0
    ):
        payload["grid_exhausted"] = True
        payload["note"] = "临界值或零假设标准差不可用，功效不可估计"
        return payload
    for delta in delta_grid:
        shifted = values + float(delta) * float(sigma)
        payload["power"][f"{float(delta):.2f}"] = float((shifted > float(critical)).mean())
    for delta in delta_grid:
        if payload["power"][f"{float(delta):.2f}"] >= float(min_power):
            payload["mde"] = float(delta)
            break
    else:
        payload["grid_exhausted"] = True
    return payload


def analyze_cell(
    member_residuals: np.ndarray,
    group_date: np.ndarray,
    intent_residual: np.ndarray,
    resolved_mask: np.ndarray,
    unfilled_mask: np.ndarray,
    *,
    block_h: int,
    implementations: int,
    seeds: Dict[str, int],
    quantile: float,
    delta_grid: Sequence[float],
    min_power: float,
    day_span: Optional[Tuple[int, int]] = None,
    chunk: int = DEFAULT_CHUNK,
) -> Dict[str, Any]:
    """单格：观察统计量 + 校准流临界值 + 独立流误放行 + 独立流功效。"""
    statistics = observed_statistic(intent_residual, resolved_mask, unfilled_mask)
    calibration = null_distribution(
        member_residuals,
        group_date,
        block_h=block_h,
        implementations=implementations,
        seed=int(seeds["calibration"]),
        day_span=day_span,
        chunk=chunk,
    )
    evaluation = null_distribution(
        member_residuals,
        group_date,
        block_h=block_h,
        implementations=implementations,
        seed=int(seeds["evaluation"]),
        day_span=day_span,
        chunk=chunk,
    )
    power_stream = null_distribution(
        member_residuals,
        group_date,
        block_h=block_h,
        implementations=implementations,
        seed=int(seeds["power"]),
        day_span=day_span,
        chunk=chunk,
    )
    critical = critical_value(calibration.values, quantile)
    sigma = float(np.std(calibration.values, ddof=1)) if calibration.values.size > 1 else None
    failures = int((evaluation.values > critical).sum()) if critical is not None else 0
    return {
        "intents": statistics.intents,
        "resolved": statistics.resolved,
        "unfilled": statistics.unfilled,
        "statistic_intent": statistics.statistic_intent,
        "statistic_filled_only": statistics.statistic_filled_only,
        "critical_value": critical,
        "quantile": float(quantile),
        "quantile_method": "linear",
        "sigma_D0": sigma,
        "null_calibration": {
            "implementations": calibration.implementations,
            "seed": calibration.seed,
            "dates": calibration.dates,
            "groups": calibration.groups,
            "block_h": block_h,
            "mean": float(np.mean(calibration.values)) if calibration.values.size else None,
            "std": sigma,
            "percentiles": _percentiles(calibration.values),
            "in_sample_reference_rate": (
                float((calibration.values > critical).mean()) if critical is not None else None
            ),
        },
        "null_evaluation": {
            "implementations": evaluation.implementations,
            "seed": evaluation.seed,
            "dates": evaluation.dates,
            "percentiles": _percentiles(evaluation.values),
        },
        "false_pass": {
            "failures": failures,
            "trials": int(evaluation.values.size),
            "rate": (failures / evaluation.values.size) if evaluation.values.size else None,
            "cp_upper_95": clopper_pearson_upper(failures, int(evaluation.values.size)),
        },
        "power": power_table(power_stream.values, critical, sigma, delta_grid, min_power),
        "power_stream": {
            "implementations": power_stream.implementations,
            "seed": power_stream.seed,
            "percentiles": _percentiles(power_stream.values),
        },
        "chunk": int(chunk),
        "randomization_block_h": int(block_h),
        "randomization_date_key": "signal_day",
    }


def _percentiles(values: np.ndarray) -> Dict[str, Optional[float]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"p01": None, "p05": None, "p50": None, "p95": None, "p99": None}
    return {
        "p01": float(np.quantile(finite, 0.01)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
    }
