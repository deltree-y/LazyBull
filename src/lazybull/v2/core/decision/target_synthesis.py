"""P6.5 T4 目标合成：**先合并再整手**、插入序保持、单次归一化。

契约依据：
- 设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md`` §8.4
  「区分错误的补充构造」表（先合并再整手 / 多股票保序 / 不重复归一化 /
  目标与配额校验）；
- ``docs/contracts/netting_freeze.md`` F25 §5（Q18 意图上界与余数分摊、
  物理市场订单 100 股整手、虚拟分摊单位 1 股）；
- ``docs/contracts/protocols.md`` F13 §3.4 C65-02（配额 ``[0,1]`` 无量纲、
  ``math.fsum`` 校验、通过后**不重新归一化**）。

本模块只解决三件事（均为纯函数，不持账本、不读配置、不做 IO）：

1. **母配额合并**：各子袖配额按袖 ID 字典序做显式 float64 左到右求和，得到母配额
   （唯一合并点；禁止用 3.12 起带补偿的 ``sum``）；
2. **整手只发生在母层**：``母目标股数 = floor(母配额 / 价 / 整手) × 整手``，随后按
   Q18 同一内核把母目标股数分摊回各子袖（1 股单位）。**逐袖先整手再相加**是必须被
   捕获的错误构造（300 → 100+100=200），由 ``assert_merged_rounding`` 可失败守卫拒；
3. **单次归一化**：正分数按插入序做**一次** ``score_i / Σscore``，保留插入序与
   位模式；不排序、不二次归一化。浮点和以 ``n × 2^-52`` 容差判定（精确 ``== 1``
   会误拒合法归一化结果）。

边界（如实登记）：
- 本模块不实现 S1/S4/S5 判定，不产生任何物理或虚拟副作用；
- 整手除数与容量口径只覆盖买入方向（S4/建仓）；卖出方向归 P7；
- 归一化输出只是目标权重，不进入指纹，也不构成 runs 配置键。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from src.lazybull.v2.core.accounting.g4_nav import float64_bits
from src.lazybull.v2.core.accounting.settlement_kernel import allocate_shares
from src.lazybull.v2.core.decision.sizing import compute_lot_shares

#: 本模块唯一阶段标识（进入全部错误上下文）
STAGE = "P6.5"
#: 物理市场订单整手（A 股；虚拟分摊单位仍为 1 股）
LOT_SIZE = 100
#: float64 单位舍入误差（多袖总和容差 = n × 2^-52）
_FLOAT64_EPS = 2.0**-52


class TargetSynthesisError(ValueError):
    """目标合成输入非法（**零副作用**拒绝；错误含阶段与左/右值）。"""


def _error(message: str) -> Exception:
    """带阶段上下文的目标合成拒绝。"""
    return TargetSynthesisError(f"阶段={STAGE} 目标合成 {message}")


def _tolerance(n_sleeves: int) -> float:
    """多袖总和容差 ``n × 2^-52``（float64 单位舍入的保守上界）。"""
    return max(1, int(n_sleeves)) * _FLOAT64_EPS


def sum_left_to_right(values: Sequence[float]) -> float:
    """显式 float64 左到右求和（禁止用 3.12 起带补偿的 ``sum``；顺序即证据）。"""
    total = 0.0
    for value in values:
        total += value
    return total


def _require_finite_non_negative(value: Any, field: str) -> float:
    """有限非负数解析（NaN/Inf/负值/非数一律拒绝）。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise _error(f"{field} 必须为有限非负数，实际 {value!r}（{error}）") from error
    if not math.isfinite(number) or number < 0.0:
        raise _error(f"{field} 必须为有限非负数，实际 {value!r}")
    return number


# ────────────────────────── 母配额合并 ──────────────────────────


def merge_sleeve_values(
    values: Mapping[str, float], *, label: str = ""
) -> Tuple[float, Tuple[Tuple[str, float], ...]]:
    """各子袖配额合并为母配额（袖 ID 字典序 + float64 左到右，唯一合并点）。

    Returns:
        ``(母配额, 按合并顺序排列的 (袖, 配额) 元组)``。

    Raises:
        TargetSynthesisError: 空映射/袖 ID 为空/重复/配额非有限非负。
    """
    if not isinstance(values, Mapping) or not values:
        raise _error(f"{label} 子袖配额必须为非空 Mapping，实际 {type(values).__name__}")
    ordered: list[Tuple[str, float]] = []
    seen: set = set()
    for sleeve_id in sorted(str(key) for key in values):
        if not sleeve_id.strip():
            raise _error(f"{label} 子袖 ID 不得为空")
        if sleeve_id in seen:
            raise _error(f"{label} 子袖 ID 重复：{sleeve_id!r}")
        seen.add(sleeve_id)
        ordered.append(
            (
                sleeve_id,
                _require_finite_non_negative(values[sleeve_id], f"{label} 袖 {sleeve_id} 配额"),
            )
        )
    mother = sum_left_to_right([value for _sleeve, value in ordered])
    return mother, tuple(ordered)


@dataclass(frozen=True, slots=True)
class LotTargetSynthesis:
    """母配额 → 母整手目标 → 逐袖子目标（1 股单位）的冻结证据。"""

    price: float
    lot_size: int
    mother_value: float
    mother_target_shares: int
    values: Tuple[Tuple[str, float], ...]
    capacity_by_sleeve: Tuple[Tuple[str, int], ...]
    shares_by_sleeve: Tuple[Tuple[str, int], ...]

    def shares_of(self, sleeve_id: str) -> int:
        """单袖目标股数（未参与袖返回 0）。"""
        for key, value in self.shares_by_sleeve:
            if key == sleeve_id:
                return value
        return 0

    def as_mapping(self) -> Dict[str, int]:
        """逐袖目标股数的可变投影（dict 插入序 = 袖 ID 字典序）。"""
        return {key: value for key, value in self.shares_by_sleeve}

    @property
    def allocated_total(self) -> int:
        """逐袖目标股数合计（必须等于母目标股数，0 股容差）。"""
        return sum(value for _key, value in self.shares_by_sleeve)


def synthesize_lot_targets(
    values: Mapping[str, float],
    *,
    price: float,
    lot_size: int = LOT_SIZE,
    label: str = "",
) -> LotTargetSynthesis:
    """**先合并再整手**：母配额整手一次，再按 Q18 内核把股数分摊回子袖。

    规则（冻结）：
    1. 母配额 = Σ子袖配额（``merge_sleeve_values``，袖 ID 字典序、float64 左到右）；
    2. 母目标股数 = ``compute_lot_shares(母配额, 价, 整手)``——**整手只在这里发生一次**；
    3. 单袖容量 = ``compute_lot_shares(袖配额, 价, 1)``（1 股单位；不逐袖整手），
       母目标股数超过容量和即拒绝（fail-closed，不得强行塞给某一袖）；
    4. 分摊复用 Q18 唯一内核 ``allocate_shares``（余数按 ``(容量降序, 袖 ID 升序)``），
       因此不存在第二套余数规则。

    Raises:
        TargetSynthesisError: 配额非法/价格非法/整手非法/容量不足/自洽复核失败。
        VirtualLedgerError: ``allocate_shares`` 内核拒绝（超意图/余数无法分配）。
    """
    synthesis = synthesize_lot_targets_unchecked(
        values, price=price, lot_size=lot_size, label=label
    )
    if synthesis.allocated_total != synthesis.mother_target_shares:
        raise _error(
            f"{label} 逐袖目标合计 {synthesis.allocated_total} 股 != 母目标 "
            f"{synthesis.mother_target_shares} 股（容差=0 股 单位=股）"
        )
    return synthesis


def assert_merged_rounding(
    values: Mapping[str, float],
    shares_by_sleeve: Mapping[str, int] | Sequence[Tuple[str, int]],
    *,
    price: float,
    lot_size: int = LOT_SIZE,
    label: str = "",
) -> int:
    """可失败守卫：逐袖股数必须等于「先合并母配额、再整手、再分摊」的唯一结果。

    这是「子袖先各自整手」构造的判据：母配额 3,000、价 10 的正确母目标是 300 股
    （子袖各 150），若实现先按子配额 1,500 各自整手得到 100+100=200，则本守卫以
    左/右值拒绝。

    Returns:
        母目标股数（守卫通过时）。

    Raises:
        TargetSynthesisError: 逐袖股数与唯一结果不一致/合计不符母目标。
    """
    synthesized = synthesize_lot_targets_unchecked(
        values, price=price, lot_size=lot_size, label=label
    )
    if isinstance(shares_by_sleeve, Mapping):
        given = {str(key): int(value) for key, value in shares_by_sleeve.items()}
    else:
        given = {str(key): int(value) for key, value in shares_by_sleeve}
    expected = {key: value for key, value in synthesized.shares_by_sleeve}
    if given != expected:
        raise _error(
            f"{label} 逐袖目标股数 {sorted(given.items())} != 先合并再整手的唯一结果 "
            f"{sorted(expected.items())}（母配额={synthesized.mother_value!r} "
            f"母目标={synthesized.mother_target_shares} 股 价={price!r} 整手={lot_size}）；"
            "子袖先各自整手再相加必须失败（容差=0 股 单位=股）"
        )
    return synthesized.mother_target_shares


def synthesize_lot_targets_unchecked(
    values: Mapping[str, float],
    *,
    price: float,
    lot_size: int = LOT_SIZE,
    label: str = "",
) -> LotTargetSynthesis:
    """``synthesize_lot_targets`` 的守卫前置版本（同一实现，仅不做自洽复核）。"""
    price_value = _require_finite_non_negative(price, f"{label} 目标价")
    if price_value <= 0.0:
        raise _error(f"{label} 目标价必须为正数，实际 {price!r}")
    if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size <= 0:
        raise _error(f"{label} 整手必须为正整数，实际 {lot_size!r}")
    mother, ordered = merge_sleeve_values(values, label=label)
    mother_shares = compute_lot_shares(mother, price_value, lot_size)
    capacity: Dict[str, int] = {
        sleeve_id: compute_lot_shares(value, price_value, 1) for sleeve_id, value in ordered
    }
    total_capacity = sum(capacity.values())
    if mother_shares > total_capacity:
        raise _error(
            f"{label} 母目标股数 {mother_shares} 股 > 逐袖容量和 {total_capacity} 股"
            f"（袖数={len(ordered)} 母配额={mother!r} 价={price_value!r}）；"
            "禁止把超出容量的股数强行塞给某一袖"
        )
    allocations = allocate_shares(capacity, mother_shares)
    return LotTargetSynthesis(
        price=price_value,
        lot_size=lot_size,
        mother_value=mother,
        mother_target_shares=mother_shares,
        values=tuple((sleeve_id, value) for sleeve_id, value in ordered),
        capacity_by_sleeve=tuple(
            (sleeve_id, capacity[sleeve_id]) for sleeve_id in sorted(capacity)
        ),
        shares_by_sleeve=tuple((entry.sleeve_id, entry.allocated_shares) for entry in allocations),
    )


# ────────────────────────── 单次归一化 ──────────────────────────


def normalize_positive_scores(
    scores: Sequence[Tuple[str, float]], *, label: str = ""
) -> Dict[str, float]:
    """正分数**单次**归一化：``w_i = score_i / Σscore``，保留插入序与位模式。

    - 求和使用 float64 左到右（插入序），不做第二次除法，不排序改写顺序；
    - 归一化后以 ``math.fsum`` 复核总和为 1（容差 ``n × 2^-52``）；**禁止**用精确
      ``== 1`` 判定——``8/35+9/35+9/35+9/35`` 的 float64 和是 ``0.9999999999999999``，
      精确判定会误拒合法输入。

    Args:
        scores: ``[(股票代码, 原始正分数)]``，顺序即插入序（禁止内部排序）。

    Returns:
        与输入同序的 ``{股票代码: 权重}``（不排序、不二次归一化）。
    """
    if isinstance(scores, (str, bytes)) or not isinstance(scores, Sequence) or not scores:
        raise _error(f"{label} 归一化输入必须为非空正分数序列，实际 {type(scores).__name__}")
    ordered: list[Tuple[str, float]] = []
    seen: set = set()
    for item in scores:
        try:
            code, raw = item
        except (TypeError, ValueError) as error:
            raise _error(f"{label} 归一化元素必须为 (代码, 分数)，实际 {item!r}") from error
        code = str(code)
        if not code.strip():
            raise _error(f"{label} 归一化股票代码不得为空")
        if code in seen:
            raise _error(f"{label} 归一化股票代码重复：{code!r}")
        seen.add(code)
        score = _require_finite_non_negative(raw, f"{label} 股票 {code} 原始分数")
        if score <= 0.0:
            raise _error(f"{label} 股票 {code} 原始分数必须为正数，实际 {raw!r}")
        ordered.append((code, score))
    total = sum_left_to_right([score for _code, score in ordered])
    if total <= 0.0:  # pragma: no cover - 全正分数时不可达
        raise _error(f"{label} 原始分数合计必须为正数，实际 {total!r}")
    weights = {code: score / total for code, score in ordered}
    tolerance = _tolerance(len(weights))
    fsum = math.fsum(weights.values())
    if abs(fsum - 1.0) > tolerance:
        raise _error(
            f"{label} 归一化后权重 fsum={fsum!r} 偏离 1 超过容差 {tolerance}"
            f"（n={len(weights)} × 2^-52）；拒绝，且不得通过二次归一化掩盖"
        )
    return weights


def assert_single_normalized(
    scores: Sequence[Tuple[str, float]],
    weights: Mapping[str, float],
    *,
    label: str = "",
) -> Dict[str, float]:
    """可失败守卫：权重必须等于**一次**归一化的位模式结果（二次归一化即改位）。

    二次归一化 ``w_i / Σw`` 会把 ``8/35``（0.22857142857142856）改成
    0.2285714285714286；本守卫逐股按 float64 位模式比较（容差 0）。
    """
    expected = normalize_positive_scores(scores, label=label)
    if set(weights) != set(expected):
        raise _error(f"{label} 归一化键集不一致：给定={sorted(weights)} 期望={sorted(expected)}")
    # 键序也属契约（插入序 = 分数输入序）：外部评审 T4-Codex-06
    if tuple(weights) != tuple(expected):
        raise _error(
            f"{label} 归一化键序被改写：给定={list(weights)} 期望={list(expected)}；"
            "重新排序均必须失败（插入序即契约序）"
        )
    drifted: Dict[str, Tuple[str, str]] = {}
    for code, value in expected.items():
        given_bits = float64_bits(float(weights[code]))
        expected_bits = float64_bits(value)
        if given_bits != expected_bits:
            drifted[code] = (given_bits, expected_bits)
    if drifted:
        raise _error(
            f"{label} 权重位模式被改写（给定位 vs 单次归一化位）：{drifted}；"
            "重复归一化/重新排序均必须失败（容差=0）"
        )
    return expected


__all__ = (
    "LOT_SIZE",
    "LotTargetSynthesis",
    "STAGE",
    "TargetSynthesisError",
    "assert_merged_rounding",
    "assert_single_normalized",
    "merge_sleeve_values",
    "normalize_positive_scores",
    "sum_left_to_right",
    "synthesize_lot_targets",
    "synthesize_lot_targets_unchecked",
)
