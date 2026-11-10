"""P6.5 T4 G4 双 NAV 门：先合并货币总值、再单次派生 NAV、与单袖**位模式**比较。

契约依据：
- ``docs/contracts/netting_freeze.md`` F25 §4（I3 直接贡献 NAV 和的 ``1e-6`` 容差门）；
- 设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md`` §4.3（G4 定义）、
  §8.4 与 K22（逐袖分别除法再求和与先合并总值再除法可差 1 ULP，**两个门不得共用同一
  计算顺序**）、§10.1 G4。

与 I3 的运算顺序分离（冻结，不得混用）：

| 门 | 计算顺序 | 判定 | 失败语义 |
|---|---|---|---|
| I3 | ``Σ_i calculate_nav(total_value_i, C0)`` | 与物理 NAV 差 ≤ ``1e-6`` | ``ReconciliationError`` |
| G4 | ``calculate_nav(Σ_i total_value_i, C0)`` | 与单袖 NAV **位模式同**（0 容差） | ``G4NavMismatchError`` |

D2 见证（设计 §8.4）：两袖总值 50,200 / 50,195、初始资金 100,000 时
``Σ NAV_i_contrib = 1.0039500000000001``，而物理 ``calculate_nav(100395, 100000)``
= ``1.00395``，两者相差 **1 ULP**（``2.220446049250313e-16``）：I3 容差门通过、
G4 位模式门必须一致。**禁止**把逐袖已舍入贡献 NAV 的和当作 G4 字段，也禁止用
``approx``/容差兜底 G4。

边界：本模块只做派生与比较，不写账本、不读配置；缺价/跨日价源仍由
``reconciliation`` 的 I3 与 ``VirtualLedger.Error`` 体系负责拒绝（本模块只复核
``price_source.as_of == context.trade_date`` 这一条同源前置）。
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.core.accounting.virtual_ledger import (
    PriceSource,
    VirtualLedgerError,
    VirtualSleeveState,
)

#: 本模块唯一阶段标识（进入全部错误上下文）
STAGE = "P6.5"


class G4NavMismatchError(RuntimeError):
    """G4 破缺：合并后单次派生的 NAV 与单袖 NAV 位模式不同（终态语义，禁止降级）。"""


def _error(message: str) -> Exception:
    """带阶段上下文的 G4 输入拒绝。"""
    return ValueError(f"阶段={STAGE} G4 {message}")


def float64_bits(value: float) -> str:
    """float64 位模式（16 位小写十六进制；NaN/Inf 也各得其位模式，不参与近似比较）。"""
    return struct.pack(">d", float(value)).hex()


def ulp_distance(left: float, right: float) -> int:
    """两个 float64 之间的 ULP 距离（负值按单调映射处理；NaN 返回 -1）。"""
    left_value = float(left)
    right_value = float(right)
    if math.isnan(left_value) or math.isnan(right_value):
        return -1
    return abs(_monotonic_int(left_value) - _monotonic_int(right_value))


def _monotonic_int(value: float) -> int:
    """float64 → 单调递增的整数映射（同号可比、跨零不跳变）。"""
    bits = struct.unpack(">q", struct.pack(">d", value))[0]
    if bits < 0:
        return -(bits & 0x7FFFFFFFFFFFFFFF)
    return bits


@dataclass(frozen=True, slots=True)
class G4NavEvidence:
    """G4 门证据（合并总值、单次派生 NAV、单袖 NAV、位模式与 ULP 距离）。"""

    price_source_id: str
    physical_initial_capital: float
    sleeve_totals: Tuple[Tuple[str, float], ...]
    merged_total_value: float
    merged_nav: float
    single_arm_nav: float
    merged_total_bits: str
    merged_nav_bits: str
    single_arm_bits: str
    ulp_distance: int


def merged_sleeve_total_value(
    virtual_states: Mapping[str, VirtualSleeveState],
    price_source: PriceSource,
    *,
    context: Any,
) -> Tuple[float, Tuple[Tuple[str, float], ...]]:
    """把各袖货币总值按**固定顺序**（袖 ID 字典序）左到右合并（G4 第一步）。

    求和使用显式 float64 左到右累加（禁止 3.12 起带补偿的 ``sum``），并拒绝非有限
    总值：NaN/Inf 会让位比较失去意义。
    """
    label = _context_label(context)
    if not isinstance(virtual_states, Mapping) or not virtual_states:
        raise _error(f"{label} G4 需要非空袖快照，实际 {type(virtual_states).__name__}")
    totals: list[Tuple[str, float]] = []
    for sleeve_id in sorted(str(key) for key in virtual_states):
        state = virtual_states[sleeve_id]
        try:
            raw = state.total_value(price_source)
        except VirtualLedgerError as error:
            raise _error(f"{label} 袖 {sleeve_id} 同日同价源估值失败：{error}") from error
        value = float(raw)
        if not math.isfinite(value):
            raise _error(f"{label} 袖 {sleeve_id} 货币总值非有限（{raw!r}），拒绝参与 G4")
        totals.append((sleeve_id, value))
    merged = 0.0
    for _sleeve_id, value in totals:
        merged += value
    return merged, tuple(totals)


def g4_merged_nav(
    virtual_states: Mapping[str, VirtualSleeveState],
    price_source: PriceSource,
    physical_initial_capital: float,
    *,
    context: Any,
) -> G4NavEvidence:
    """G4 派生：合并各袖货币总值后**只调用一次** ``calculate_nav``。

    与 I3 的 ``Σ calculate_nav(total_value_i)`` 顺序不同，二者不得互相替代：
    本函数返回的 ``merged_nav`` 是 G4 位模式门左值；I3 左值由
    ``reconciliation.check_nav_identity`` 独立产生。
    """
    label = _context_label(context)
    if not isinstance(price_source, PriceSource):
        raise _error(f"{label} G4 需要 PriceSource，实际 {type(price_source).__name__}")
    trade_date = getattr(context, "trade_date", None)
    if trade_date is not None and price_source.as_of != trade_date:
        raise _error(
            f"{label} 价格源日期 {price_source.as_of} != 校验日期 {trade_date}；"
            "所有袖必须共用同日同价源"
        )
    initial = float(physical_initial_capital)
    if not math.isfinite(initial) or initial <= 0.0:
        raise _error(f"{label} 物理初始资金必须为有限正数，实际 {physical_initial_capital!r}")
    merged, totals = merged_sleeve_total_value(virtual_states, price_source, context=context)
    merged_nav = float(calculate_nav(merged, initial))
    return G4NavEvidence(
        price_source_id=price_source.source_id,
        physical_initial_capital=initial,
        sleeve_totals=totals,
        merged_total_value=merged,
        merged_nav=merged_nav,
        single_arm_nav=float("nan"),
        merged_total_bits=float64_bits(merged),
        merged_nav_bits=float64_bits(merged_nav),
        single_arm_bits="",
        ulp_distance=-1,
    )


def assert_g4_nav_matches(
    *,
    virtual_states: Mapping[str, VirtualSleeveState],
    price_source: PriceSource,
    physical_initial_capital: float,
    single_arm_nav: float,
    context: Any,
) -> G4NavEvidence:
    """G4 门：合并后单次派生 NAV 与单袖 NAV **位模式逐位一致**（容差 0）。

    Raises:
        G4NavMismatchError: 位模式不同（含 NaN 位置差、1 ULP 差）；消息给出左右值、
            两侧位模式、ULP 距离与袖级总值明细。
    """
    from dataclasses import replace

    evidence = g4_merged_nav(
        virtual_states, price_source, physical_initial_capital, context=context
    )
    single = float(single_arm_nav)
    if not math.isfinite(single):
        raise G4NavMismatchError(
            f"[{STAGE}][G4] 单袖 NAV 非有限（{single_arm_nav!r}），无法作为位模式基准｜"
            f"{_context_label(context)}"
        )
    evidence = replace(
        evidence,
        single_arm_nav=single,
        single_arm_bits=float64_bits(single),
        ulp_distance=ulp_distance(evidence.merged_nav, single),
    )
    if evidence.merged_nav_bits != evidence.single_arm_bits:
        raise G4NavMismatchError(
            f"[{STAGE}][G4] 合并后单次派生 NAV {evidence.merged_nav!r} 与单袖 NAV "
            f"{single!r} 位模式不同（左(合并)={evidence.merged_nav_bits} "
            f"右(单袖)={evidence.single_arm_bits} 差={evidence.merged_nav!r} - "
            f"{single!r} ULP={evidence.ulp_distance}）；容差=0（G4 禁止近似比较）"
            f"｜袖总值={[(sid, value) for sid, value in evidence.sleeve_totals]} "
            f"合并总值位模式={evidence.merged_total_bits} "
            f"初始资金={evidence.physical_initial_capital!r} "
            f"｜{_context_label(context)}"
        )
    return evidence


def assert_g4_nav_bitwise_equal(
    left_nav: float, right_nav: float, *, context: Any, label: str = ""
) -> Tuple[float, float]:
    """两臂 G4 NAV **位模式**相等（单袖 vs 合并派生；容差 0）。

    与 ``assert_g4_nav_matches`` 的区别：后者在**同一袖快照**上派生并与给定基准比较；
    本函数在两个**已派生**的 NAV 之间做位比较，供装配/证据链路复用同一判定口径
    （T4 复审 A-R2 口径：G4 只保留一个位模式判定入口）。
    """
    left = float(left_nav)
    right = float(right_nav)
    left_bits = float64_bits(left)
    right_bits = float64_bits(right)
    if left_bits != right_bits:
        raise G4NavMismatchError(
            f"[{STAGE}][G4] {label} 两臂 NAV 位模式不同：左={left!r}（位={left_bits}） "
            f"右={right!r}（位={right_bits}） ULP={ulp_distance(left, right)}；"
            "容差=0（G4 禁止近似比较）"
            f"｜{_context_label(context)}"
        )
    return left, right


def _context_label(context: Any) -> str:
    """G4 错误上下文（阶段/日期/宿主/运行 ID/袖集合/价格源）。"""
    describe = getattr(context, "describe", None)
    if callable(describe):
        return f"上下文={describe()}"
    return f"上下文={context!r}"


__all__ = (
    "G4NavEvidence",
    "G4NavMismatchError",
    "STAGE",
    "assert_g4_nav_bitwise_equal",
    "assert_g4_nav_matches",
    "float64_bits",
    "g4_merged_nav",
    "merged_sleeve_total_value",
    "ulp_distance",
)
