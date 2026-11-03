# -*- coding: utf-8 -*-
"""P5b T4 收益裁判的 look 映射与 block 统计量（设计 v4 §5.1/§5.2 + PSD 预登记）。

契约来源
========
- 设计 v4 §5.1：``n_max=503``、``J_max=floor(n_max/H)``、
  ``J_candidates={ceil(120/H),ceil(240/H),ceil(360/H),J_max}``、
  ``J_looks = sorted(j for j in J_candidates if 6 <= j <= J_max)``、``K=len(J_looks)``、
  ``look_days=[H*j for j in J_looks]``；K<2 或任一 look 有效 block<6 ⇒ 拒绝该统计配置。
- 设计 v4 §5.2：block 均值、``gamma0``/``gamma1`` 用 ``J-1`` 分母、``Z_k = B_k/sqrt(V_k/J)``；
  ``V_k<=0``／非有限／block 不可构造 ⇒ 该 look **fail-closed**（不否决、也不产生有利结论）。
- PSD 预登记（`docs/plans/v2/p5b_income_referee_psd_prereg_draft_20261101.md`，
  sha256 首 8 位 ``1b92b25c``）：现行注册统计量
  ``T_k=(B_k+delta)/sqrt(V_PSD,k/J)``、``V_PSD=gamma0+gamma1``（等价平方和，恒 >= 0）。

两个统计量都实现（可审计、可对照），但**只有 PSD 版可注册**：
``gamma0+2*gamma1`` 在 T0.5b 已被实测判定首 look 无效率 >10%（AR1/real），它留在库里只用于
fail-closed 演示与旧证据复核，不得作为 H4 注册统计量（见 :data:`REGISTERED_STATISTICS`）。

单位：``B`` 与 ``delta`` 为 log 收益单位，``V`` 为其平方。禁止 ``max(V, eps)`` 截断与
任何带宽/平滑替换（设计 §5.2「不使用固定带宽 Bartlett」条款已由 PSD 预登记明确收紧为
lag-1 闭式，见 :func:`variance_psd_lag1`）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError

__all__ = [
    "DEFAULT_CHUNK",
    "ANNUAL_TRADING_DAYS",
    "DELTA_ANNUAL_TARGET",
    "DELTA_LOG_DAILY",
    "LOOK_N_MAX",
    "LookPlan",
    "LookPlanError",
    "MIN_BLOCKS_PER_LOOK",
    "MIN_LOOKS",
    "REGISTERED_STATISTICS",
    "STATISTIC_DESIGN_V4",
    "STATISTIC_PSD",
    "ScoreBlock",
    "StatisticError",
    "annualized_log_return",
    "block_means",
    "make_look_plan",
    "score_paths",
    "veto_flags",
    "variance_gamma0_plus_2gamma1",
    "variance_psd_lag1",
]

#: 收益样本上限（设计 §5.1）
LOOK_N_MAX = 503
#: 每个 look 的最少有效 block 数（设计 §5.1；不是只检查终点 J_max）
MIN_BLOCKS_PER_LOOK = 6
#: 可注册 look 的最少个数（K<2 ⇒ 拒绝该统计配置）
MIN_LOOKS = 2
#: 分块路径数默认值（放在最上游模块，避免 training/holdout/power 之间循环导入）
DEFAULT_CHUNK = 2000
#: look 目标交易日（**只有前三个**进入 ``ceil(target/H)``；第四个 look 由 ``J_max=floor(503/H)``
#: 决定，不写成 500 以免与 §5.1 口径混写）
LOOK_TARGET_DAYS: Tuple[int, ...] = (120, 240, 360)
#: 年化换算的共享常量（log 尺度线性年化；账户自然日复合年化唯一实现在 ``paper/performance.py``）
ANNUAL_TRADING_DAYS = 252
#: 年化收益目标（5%）
DELTA_ANNUAL_TARGET = 0.05
#: 日度 log 位移基准（= ln(1.05)/252，冻结字面值见 PSD manifest ``statistic.delta_log_daily``）
DELTA_LOG_DAILY = 0.0001936117626

#: 现行注册统计量（PSD + 边界中心化；预登记 sha ``1b92b25c``）
STATISTIC_PSD = "psd_bartlett_lag1_boundary_centered"
#: 设计 v4 §5.2 原式（``gamma0+2*gamma1``）：T0.5b 已判定首 look 无效率超门，**不可注册**
STATISTIC_DESIGN_V4 = "bartlett_lag1_gamma0_plus_2gamma1"
#: 可注册统计量白名单（只含 PSD 版）
REGISTERED_STATISTICS: Tuple[str, ...] = (STATISTIC_PSD,)
#: 已知但**不可注册**的统计量（保留用于 fail-closed 演示与旧证据复核）
KNOWN_STATISTICS: Tuple[str, ...] = (STATISTIC_PSD, STATISTIC_DESIGN_V4)

#: 单 look 无效原因
REASON_INSUFFICIENT_BLOCKS = "insufficient_blocks"
REASON_NON_FINITE = "non_finite"
REASON_NON_POSITIVE_VARIANCE = "non_positive_variance"


class LookPlanError(TestbedError):
    """look 构造不可用（K<2、无有效 look、H 非法）。"""

    __test__ = False


class StatisticError(TestbedError):
    """统计量输入非法（形状/维度/参数）。"""

    __test__ = False


def delta_annualized_ok(delta: float) -> bool:
    """冻结 ``delta`` 是否与 ``ln(1+5%)/252`` 一致（1e-9 相对容差）。

    冻结字面值 ``0.0001936117626`` 是 ``ln(1.05)/252 = 0.00019361176257711127`` 的
    10 位小数舍入（相对差 1.2e-10）；容差按此设定，既拦住换 δ，又不与舍入冲突。
    """
    expected = math.log(1.0 + DELTA_ANNUAL_TARGET) / ANNUAL_TRADING_DAYS
    return math.isclose(float(delta), expected, rel_tol=1e-9, abs_tol=0.0)


def annualized_log_return(daily_log: float) -> float:
    """log 尺度年化（线性）：``daily * 252``。

    注意：这是**收益裁判的 log 单位换算**，不是账户年化。账户起始日解析与自然日复合年化
    只允许在 ``paper/performance.py`` 实现，两者不得互换。
    """
    return float(daily_log) * ANNUAL_TRADING_DAYS


@dataclass(frozen=True)
class LookPlan:
    """H 的 look/block 映射（§5.1 冻结规则的唯一实现点）。"""

    h: int
    n_max: int
    j_max: int
    j_candidates: Tuple[int, ...]
    j_looks: Tuple[int, ...]
    look_days: Tuple[int, ...]

    @property
    def k(self) -> int:
        """注册 look 个数。"""
        return len(self.j_looks)

    def horizon(self, index: int) -> int:
        """第 ``index`` 个 look 的样本终点日数（``H*J``）。"""
        return int(self.look_days[index])

    def max_horizon(self) -> int:
        """最大 look 终点（``H*J_max``，与样本上限对齐）。"""
        return int(self.look_days[-1])

    def as_dict(self) -> Dict[str, Any]:
        """可序列化形态（manifest/report 共用口径）。"""
        return {
            "n_max": int(self.n_max),
            "H": int(self.h),
            "j_max": int(self.j_max),
            "j_candidates": [int(j) for j in self.j_candidates],
            "look_blocks": [int(j) for j in self.j_looks],
            "look_days": [int(day) for day in self.look_days],
            "K": self.k,
            "min_blocks_per_look": int(MIN_BLOCKS_PER_LOOK),
        }


def make_look_plan(h: int, n_max: int = LOOK_N_MAX) -> LookPlan:
    """按设计 §5.1 构造 look 映射；不可用即抛 :class:`LookPlanError`。

    规则（写死在代码里、可审计）::

        J_max       = floor(n_max / H)
        J_candidates= sorted({ceil(120/H), ceil(240/H), ceil(360/H), J_max})
        J_looks     = [j for j in J_candidates if 6 <= j <= J_max]
        look_days   = [H*j for j in J_looks]

    ``K = len(J_looks) < 2`` ⇒ 拒绝（不能把不可构造 look 计入 K 后宣称已校准）。

    Args:
        h: 持有期（交易日）。
        n_max: 收益样本上限（默认 503）。

    Returns:
        :class:`LookPlan`。

    Raises:
        LookPlanError: ``H <= 0``、``n_max < H`` 或 K<2（含无 look）。
    """
    height = int(h)
    cap = int(n_max)
    if height <= 0:
        raise LookPlanError(f"H 必须为正整数，实际 {h!r}")
    if cap < height:
        raise LookPlanError(f"收益样本上限 {cap} 不足一个持有期 H={height}（无法构造 block）")
    j_max = cap // height
    candidates = sorted({math.ceil(target / height) for target in LOOK_TARGET_DAYS} | {j_max})
    j_looks = tuple(j for j in candidates if MIN_BLOCKS_PER_LOOK <= j <= j_max)
    if len(j_looks) < MIN_LOOKS:
        raise LookPlanError(
            f"H={height} 的 look 配置不可用：J_candidates={candidates} → J_looks={list(j_looks)} "
            f"(K={len(j_looks)} < {MIN_LOOKS})，J_max={j_max}；"
            "设计 §5.1 要求每个 look 至少 6 个有效 block 且 K>=2"
        )
    return LookPlan(
        h=height,
        n_max=cap,
        j_max=j_max,
        j_candidates=tuple(candidates),
        j_looks=j_looks,
        look_days=tuple(height * j for j in j_looks),
    )


def block_means(paths: np.ndarray, j: int, h: int) -> np.ndarray:
    """把每条路径的前 ``j*h`` 个日收益切成 ``j`` 个不重叠 block 并取均值。

    Args:
        paths: ``(n_paths, n_days)`` 日收益（log 差或合成路径）。
        j: block 数。
        h: block 长度（持有期）。

    Returns:
        ``(n_paths, j)`` block 均值。

    Raises:
        StatisticError: 形状非法或路径长度不足 ``j*h``（不得用较短长度截断）。
    """
    matrix = np.asarray(paths, dtype=float)
    if matrix.ndim != 2:
        raise StatisticError(f"paths 必须是二维 (n_paths, n_days)，实际 shape={matrix.shape}")
    blocks = int(j)
    length = int(h)
    if blocks < 1 or length < 1:
        raise StatisticError(f"block 参数非法: J={j!r} H={h!r}")
    need = blocks * length
    if matrix.shape[1] < need:
        raise StatisticError(
            f"路径长度不足一个 look 的 block 构造：需要 {need} 日（J={blocks}×H={length}），"
            f"实际 {matrix.shape[1]} 日（禁止按较短长度截断）"
        )
    return matrix[:, :need].reshape(matrix.shape[0], blocks, length).mean(axis=2)


def variance_psd_lag1(centered: np.ndarray) -> np.ndarray:
    """PSD lag-1 方差 ``V_PSD = gamma0 + gamma1``（等价平方和，恒 >= 0）。

    ``V = (c1^2 + cJ^2 + sum_{j=2..J}(cj + c_{j-1})^2) / (2*(J-1))``；
    与共同 ``J-1`` 分母的 ``gamma0+gamma1`` 是同一统计量。

    Args:
        centered: ``(n_paths, J)`` **已中心化**的 block 均值。

    Returns:
        ``(n_paths,)`` 非负方差（J<2 或非有限输入返回 ``nan``，由调用方判无效）。
    """
    matrix = np.asarray(centered, dtype=float)
    if matrix.ndim != 2:
        raise StatisticError(f"centered 必须是二维 (n_paths, J)，实际 shape={matrix.shape}")
    blocks = matrix.shape[1]
    out = np.full(matrix.shape[0], np.nan, dtype=float)
    if blocks < 2:
        return out
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.any():
        return out
    rows = matrix[finite]
    numerator = (
        rows[:, 0] ** 2 + rows[:, -1] ** 2 + np.sum((rows[:, 1:] + rows[:, :-1]) ** 2, axis=1)
    )
    out[finite] = numerator / (2.0 * (blocks - 1))
    return out


def variance_gamma0_plus_2gamma1(centered: np.ndarray) -> np.ndarray:
    """设计 v4 §5.2 原式 ``gamma0 + 2*gamma1``（共同 ``J-1`` 分母；**可为负**）。

    仅用于旧证据复核与 fail-closed 演示：T0.5b 实测该式在首 look 有 7%~13% 路径出现
    非正方差，已被判定不可注册（见 :data:`REGISTERED_STATISTICS`）。
    """
    matrix = np.asarray(centered, dtype=float)
    if matrix.ndim != 2:
        raise StatisticError(f"centered 必须是二维 (n_paths, J)，实际 shape={matrix.shape}")
    blocks = matrix.shape[1]
    out = np.full(matrix.shape[0], np.nan, dtype=float)
    if blocks < 2:
        return out
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.any():
        return out
    rows = matrix[finite]
    gamma0 = np.sum(rows * rows, axis=1) / (blocks - 1)
    gamma1 = np.sum(rows[:, 1:] * rows[:, :-1], axis=1) / (blocks - 1)
    out[finite] = gamma0 + 2.0 * gamma1
    return out


@dataclass(frozen=True)
class ScoreBlock:
    """逐 look 的评分块（值 + 有效性 + 无效原因；fail-closed 的唯一载体）。"""

    values: np.ndarray
    invalid_reason: Tuple[Optional[str], ...]
    statistic: str
    delta: float

    @property
    def look_count(self) -> int:
        return int(self.values.shape[1])

    def valid(self, look_index: int) -> np.ndarray:
        """第 ``look_index`` 的有效掩码（``False`` = fail-closed，不否决也不计有利）。"""
        return np.isfinite(self.values[:, int(look_index)])

    def invalid_rate(self, look_index: int) -> float:
        """第 ``look_index`` 的无效比例（计入 family 分母，不删除实现）。"""
        return float(1.0 - self.valid(look_index).mean())

    def reason_counts(self, look_index: int) -> Dict[str, int]:
        """第 ``look_index`` 的无效原因计数（审计用）。"""
        counts: Dict[str, int] = {}
        for path_reasons in self.invalid_reason:
            reason = path_reasons[int(look_index)]
            if reason is None:
                continue
            counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items()))


def score_paths(
    paths: np.ndarray,
    plan: LookPlan,
    *,
    statistic: str = STATISTIC_PSD,
    delta: float = DELTA_LOG_DAILY,
) -> ScoreBlock:
    """逐 look 计算注册统计量（PSD 版默认）。

    ``PSD``：``T_k = (B_k + delta) / sqrt(V_PSD/J)``（边界中心化把 ``mu=-delta`` 移到零点）。
    ``DESIGN_V4``：``T_k = B_k / sqrt((gamma0+2gamma1)/J)``（无 ``delta``；仅复核用）。

    无效样本（``V<=0``、非有限、block 不可构造）在结果中为 ``nan`` 并记录原因；
    调用方**不得**把无效值当作否决，也不得删除该实现（计入 family 分母与无效率）。

    Args:
        paths: ``(n_paths, n_days)`` 日收益。
        plan: look 映射。
        statistic: :data:`KNOWN_STATISTICS` 之一。
        delta: 日度 log 位移（PSD 版分子偏移）。

    Returns:
        :class:`ScoreBlock`。

    Raises:
        StatisticError: 统计量未知或 ``paths`` 非二维。
    """
    if statistic not in KNOWN_STATISTICS:
        raise StatisticError(
            f"未知收益统计量: {statistic!r}；允许 {list(KNOWN_STATISTICS)}"
            f"（可注册仅 {list(REGISTERED_STATISTICS)}）"
        )
    matrix = np.asarray(paths, dtype=float)
    if matrix.ndim != 2:
        raise StatisticError(f"paths 必须是二维 (n_paths, n_days)，实际 shape={matrix.shape}")
    n_paths = matrix.shape[0]
    values = np.full((n_paths, plan.k), np.nan, dtype=float)
    reasons: List[Tuple[Optional[str], ...]] = [tuple([None] * plan.k) for _ in range(n_paths)]
    for index, j in enumerate(plan.j_looks):
        if matrix.shape[1] < j * plan.h:
            reasons = _set_reason(reasons, index, REASON_INSUFFICIENT_BLOCKS, n_paths)
            continue
        blocks = block_means(matrix, j, plan.h)
        non_finite = ~np.isfinite(blocks).all(axis=1)
        bar = np.where(np.isfinite(blocks).all(axis=1), blocks.mean(axis=1), np.nan)
        centered = np.where(non_finite[:, None], np.nan, blocks - bar[:, None])
        variance = (
            variance_psd_lag1(centered)
            if statistic == STATISTIC_PSD
            else variance_gamma0_plus_2gamma1(centered)
        )
        usable = np.isfinite(variance) & (variance > 0) & ~non_finite
        numerator = bar + float(delta) if statistic == STATISTIC_PSD else bar
        values[usable, index] = numerator[usable] / np.sqrt(variance[usable] / j)
        reasons = _set_reason(reasons, index, REASON_NON_FINITE, n_paths, mask=non_finite)
        reasons = _set_reason(
            reasons,
            index,
            REASON_NON_POSITIVE_VARIANCE,
            n_paths,
            mask=np.isfinite(variance) & (variance <= 0),
        )
    return ScoreBlock(
        values=values,
        invalid_reason=tuple(reasons),
        statistic=statistic,
        delta=float(delta),
    )


def _set_reason(
    reasons: List[Tuple[Optional[str], ...]],
    index: int,
    reason: str,
    n_paths: int,
    *,
    mask: Optional[np.ndarray] = None,
) -> List[Tuple[Optional[str], ...]]:
    """把某 look 的无效原因写入逐路径原因表（未指定掩码 = 全部路径）。"""
    flags = np.ones(n_paths, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    updated: List[Tuple[Optional[str], ...]] = []
    for row, current in enumerate(reasons):
        entry = list(current)
        if flags[row] and entry[index] is None:
            entry[index] = reason
        updated.append(tuple(entry))
    return updated


def veto_flags(values: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """family 否决事件：``any_k(valid_k and T_k < -c_k)``（无效 look 不触发）。"""
    matrix = np.asarray(values, dtype=float)
    limits = np.asarray([float(value) for value in thresholds], dtype=float)
    if matrix.ndim != 2:
        raise StatisticError(f"values 必须是二维 (n_paths, K)，实际 shape={matrix.shape}")
    if matrix.shape[1] != limits.size:
        raise StatisticError(
            f"阈值个数与 look 数不一致: thresholds={limits.size} values={matrix.shape[1]}"
        )
    if not np.isfinite(limits).all():
        raise StatisticError(f"阈值必须有限（不可用极值/None 代替）: {limits.tolist()}")
    return np.any(np.isfinite(matrix) & (matrix < -limits[None, :]), axis=1)
