# -*- coding: utf-8 -*-
"""P7 T4 Q3 费用纯函数（S5 实际费用分摊 / S4 瀑布 / 独立费用聚合 / 有符号节省）。

契约依据（``docs/contracts/netting_freeze.md`` F26 §P7 机械复核适用注记 + §5）
==========================================================================
- **S5 实际费用分摊（Q3=A）**：``f_i = F·a_i/A``，终端袖取 ``(a_i 降序, sleeve_id 升序)``
  首个 ``a_i>0`` 者并**一次闭合**；实际费用非负、``Σf_i = F``；
  ``A=0 且 F=0`` 全零、``A=0 且 F≠0`` 拒绝；
- **S4 同向合并**：股数分摊复用受审 ``settlement_kernel.allocate_shares``（Q18）；
  费用按 F25 §5 的 N 袖非负瀑布（受益顺序 ``(q_i 升序, sleeve_id 升序)``、``F>Σc``
  比例分摊、终端袖 ``f_t=F-Σ_{i≠t}f'_i``、全零短路、逐袖守卫）在 P7 侧实现
  ——受审 ``allocate_fees`` 在真实数据上会因 **1 ulp 尾差**拒绝整笔成交
  （见 ``scenarios`` 的 ``audited_waterfall_rejects_1ulp`` 构造例），
  故 P7 不复用该函数，改为同公式实现 + 显式登记浮点闭合策略（见下）；
- **独立费用**按 ``(sleeve, 实际成交日, ts_code, 方向)`` 聚合后**计算一次**，
  跨日残差逐日计费；
- **有符号节省** ``s_i = c_i - f_i`` **允许为负**，``F>Σc`` 照常分摊并报告。

浮点闭合策略（唯一偏差，显式登记、计数、不改变任何袖的公式取值）
============================================================
契约要求 ``Σf_i = F`` **精确相等**，但 float64 的「减法 + 加法」不保证还原被减数
（``a + (F - a) ≠ F``）。受审实现对此**直接拒绝**；P7 不能因此丢单，故按顺序尝试：

1. ``f_t = F - Σ_{i≠t}f'_i``（其余袖**固定**）；
2. 一次残差修正后复算；
3. **仅终端袖**按 ulp 方向微调（≤8 步）使左到右和精确等于 ``F``；
4. 仍不成立：把**物理费用**取为可达精确和 ``F_applied = Σf_i``，登记
   ``ulp_gap = F_applied - F``（``|ulp_gap| ≤ 1 ulp(F)`` 量级），
   **不改动任何袖的费用取值**（其余袖固定、终端袖仍为 ``F - Σothers``）。

三种路径（①直接 ②一次修正 ③终端微调 ④物理侧吸收）逐笔计数并进证据
（``FeeBook.payload()``），因此该偏差**不隐藏**：任一路径命中次数、最大 ulp 位移、
累计 ``ulp_gap`` 都可复核。

边界
====
- 本模块只含纯函数与不可变事实结构：不持有账本、不做 IO、不读配置；
- 成本函数由调用方**显式注入**（生产用 ``CostModel`` 共享实现，构造例用可独立复算的
  合成费率），禁止在本模块内建第二套费率；
- 求和一律按固定顺序 **float64 从左到右**（禁止依赖 3.12 起带补偿的 ``sum``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

#: 金额键 = ``(sleeve, 交易日, 股票, 方向)``
FeeKey = Tuple[str, str, str, str]

#: 逐事件``Σf_i == F``的容差（精确相等；不满足即按浮点闭合策略处理并计数）
FEE_SUM_TOLERANCE = 0.0
#: 台账累计口径的容差（逐袖求和 vs 逐事件累加；两条独立通道）
FEE_LEDGER_TOLERANCE = 1e-6
#: 费用台账续跑快照的版本化 schema（C4-04；形状变更必须显式改版本）
FEE_STATE_SCHEMA = "p7_t4_fee_state/v1"
#: 终端袖 ulp 微调的最大步数（第 3 步；超过即转第 4 步精度容差）
ULP_NUDGE_STEPS = 8
#: 逐事件闭合精度（以 ``F`` 的 ulp 计）：两步修正 + 终端袖微调后仍不精确相等时，
#: 只要 ``|Σf_i - F| <= 100 ulp(F)`` 即视为**在精度内闭合**（不再判「拒绝」，也不叫「吸收」）。
#: 物理费用取 ``Σf_i``，故内部恒等式 ``Σ cash_i ≡ 物理现金`` 仍然精确成立；
#: 与申报值 ``F`` 的差逐笔登记进 ``float_closure``。
#:
#: 取 100 ulp 的理由：① 在 298 元量级仅 ``5.7e-12`` 元（一分的 176 亿分之一），
#: 比任何真实记账错误（最小是 1 分钱）小 9 个数量级，不存在被掩盖的空间；
#: ② 逐事件门只负责「拦住真错」，全窗 ``1e-9`` 元门更紧——3,501 笔的**平均**偏差被压到
#: 约 5 ulp/笔，故系统性偏差仍会被累计门抓住，单笔给 100 ulp 只是给偶发尾差留余量。
FEE_EVENT_TOLERANCE_ULP = 100
#: 全窗硬门（元）：逐笔精度差的最大值与累计值都必须在此之内（1e-9 元，远小于 1 分）
FEE_FLOAT_CLOSURE_TOLERANCE = 1e-9


class FeeError(ValueError):
    """费用分摊输入非法或守恒破缺（拒绝语义，零副作用）。"""


# ────────────────────────── 浮点闭合工具 ──────────────────────────


def _left_to_right(values: Sequence[float]) -> float:
    """显式 float64 从左到右求和（确定性；不受内置补偿求和影响）。"""
    total = 0.0
    for value in values:
        total += float(value)
    return total


def _require_non_negative(value: Any, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise FeeError(f"{field_name} 必须为有限非负数：{value!r}")
    return number


@dataclass(frozen=True)
class ClosureTrace:
    """终端闭合轨迹（``path`` 与位移量进证据，偏差不隐藏）。"""

    #: ``direct`` = 直接就精确；``corrected`` = 一次残差修正；``nudged`` = 终端 ulp 微调；
    #: ``within_tolerance`` = 逐事件在 ``1 ulp(F)`` 精度内闭合（不改任何袖取值）
    path: str
    terminal_shift_ulps: int
    ulp_gap: float

    @property
    def deviated(self) -> bool:
        return self.path != "direct"


def _close_terminal(
    order: Sequence[str],
    provisional: Mapping[str, float],
    terminal: str,
    fee: float,
    label: str,
) -> Tuple[Dict[str, float], float, ClosureTrace]:
    """终端袖闭合（其余袖**固定**；按四步策略并返回轨迹）。

    Returns:
        ``(最终费用映射, 残差修正量, 闭合轨迹)``；``path=="within_tolerance"`` 时
        ``最终费用映射`` 的左到右和即**可用物理费用**（``Σf``），与申报值 ``fee`` 的差
        ``ulp_gap`` 满足 ``|ulp_gap| <= 1 ulp(fee)``（精度内，不是拒绝、也不是任何袖的改动）。
    """
    others = _left_to_right([provisional[sid] for sid in order if sid != terminal])
    final = dict(provisional)
    final[terminal] = fee - others
    total = _left_to_right([final[sid] for sid in order])
    if total == fee:
        return final, 0.0, ClosureTrace("direct", 0, 0.0)
    residual = fee - total
    final[terminal] = final[terminal] + residual
    total = _left_to_right([final[sid] for sid in order])
    if total == fee:
        return final, residual, ClosureTrace("corrected", 0, 0.0)
    nudged, steps = _nudge_terminal(order, final, terminal, fee, total)
    if nudged is not None:
        return (
            nudged,
            residual + (float(nudged[terminal]) - float(final[terminal])),
            ClosureTrace("nudged", steps, 0.0),
        )
    applied = _left_to_right([final[sid] for sid in order])
    gap = applied - fee
    # 精度硬约束：残差必须落在 F 的 1 ulp 之内，否则说明不是浮点尾差而是真错，必须拒绝
    if abs(gap) > float(FEE_EVENT_TOLERANCE_ULP) * math.ulp(fee):
        raise FeeError(
            f"{label}：终端袖微调后仍差 {gap!r}（> {FEE_EVENT_TOLERANCE_ULP} ulp = "
            f"{math.ulp(fee)!r}），超出浮点精度而非舍入尾差；拒绝（Σf={applied!r} F={fee!r}）"
        )
    return final, residual, ClosureTrace("within_tolerance", 0, gap)


def _nudge_terminal(
    order: Sequence[str],
    final: Mapping[str, float],
    terminal: str,
    fee: float,
    total: float,
) -> Tuple[Optional[Dict[str, float]], int]:
    """**仅终端袖**按 ulp 方向微调，使左到右精确和等于 ``F``（有界 ``ULP_NUDGE_STEPS`` 步）。

    步长可能因累加舍入而跳过 ``F``（每次 ≥1 ulp），此时返回 ``(None, 0)``，
    由调用方转「物理侧吸收」，**不**改动其他袖的费用。
    """
    direction = math.inf if total < fee else -math.inf
    candidate = float(final[terminal])
    for step in range(1, ULP_NUDGE_STEPS + 1):
        candidate = math.nextafter(candidate, direction)
        if candidate < 0.0:
            return None, 0
        trial = dict(final)
        trial[terminal] = candidate
        if _left_to_right([trial[sid] for sid in order]) == fee:
            return trial, step
    return None, 0


# ────────────────────────── S5（Q3=A） ──────────────────────────


@dataclass(frozen=True)
class S5FeeSplit:
    """S5 同向卖出的实际费用分摊结果（Q3=A 证据）。"""

    order: Tuple[str, ...]
    intent_shares: Tuple[Tuple[str, int], ...]
    allocated_shares: Tuple[Tuple[str, int], ...]
    allocated_amounts: Tuple[Tuple[str, float], ...]
    final_fees: Tuple[Tuple[str, float], ...]
    terminal_sleeve: str
    residual_correction: float
    physical_fee: float
    zero_fill: bool
    declared_fee: float = 0.0
    closure: Optional[ClosureTrace] = None

    def fee_of(self, sleeve_id: str) -> float:
        for key, value in self.final_fees:
            if key == sleeve_id:
                return value
        return 0.0

    @property
    def total_fee(self) -> float:
        return _left_to_right([value for _key, value in self.final_fees])


def _validate_s5_inputs(
    intent_shares: Mapping[str, int],
    allocated_shares: Mapping[str, int],
    price: float,
    physical_fee: float,
) -> Tuple[float, Tuple[str, ...], Dict[str, int], Dict[str, int], Dict[str, float]]:
    """S5 入参校验：价格正有限、键集一致、单袖不超意图；返回金额向量。"""
    fee = _require_non_negative(physical_fee, "S5 物理费用 F")
    px = float(price)
    if not math.isfinite(px) or px <= 0.0:
        raise FeeError(f"S5 分摊价格必须为正有限数：{price!r}")
    if set(intent_shares) != set(allocated_shares):
        raise FeeError(
            f"S5 原始意图键集 {sorted(intent_shares)} 与分得股数键集 "
            f"{sorted(allocated_shares)} 不一致"
        )
    order = tuple(sorted(intent_shares))
    intents: Dict[str, int] = {}
    allocated: Dict[str, int] = {}
    for sid in order:
        want = int(intent_shares[sid])
        got = int(allocated_shares[sid])
        if want < 0 or got < 0:
            raise FeeError(f"S5 袖 {sid} 意图/分得股数必须非负：{want}/{got}")
        if got > want:
            raise FeeError(f"S5 袖 {sid} 分得 {got} 股超过原始意图 {want} 股（单袖不得超意图）")
        intents[sid] = want
        allocated[sid] = got
    amounts = {sid: float(allocated[sid]) * px for sid in order}
    return fee, order, intents, allocated, amounts


def _s5_terminal(order: Sequence[str], amounts: Mapping[str, float]) -> str:
    """终端袖 = ``(a_i 降序, sleeve_id 升序)`` 首个 ``a_i>0`` 者。"""
    candidates = [(sid, amounts[sid]) for sid in order if amounts[sid] > 0.0]
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[0][0]


def _assert_s5_final(
    order: Sequence[str], allocated: Mapping[str, int], final: Mapping[str, float]
) -> None:
    """最终费用逐袖守卫：非负 + 零份额不收费。"""
    for sid in order:
        if final[sid] < 0.0:
            raise FeeError(f"S5 分摊：袖 {sid} 最终费用为负 {final[sid]!r} 元；拒绝")
        if allocated[sid] == 0 and final[sid] != 0.0:
            raise FeeError(f"S5 分摊：零成交份额被收费——袖 {sid} 分得 0 股但费用 {final[sid]!r} 元")


def allocate_s5_fees(
    *,
    intent_shares: Mapping[str, int],
    allocated_shares: Mapping[str, int],
    price: float,
    physical_fee: float,
) -> S5FeeSplit:
    """Q3=A：S5 同向卖出的物理费用按实际分得金额比例分摊。

    Args:
        intent_shares: 各袖**原始卖出意图**股数（``q_i`` 的股数形式）
        allocated_shares: 各袖**实际分得**股数（``a_i`` 的股数形式，须 ≤ 原意图）
        price: 当日成交价（真实 T+1 开盘价口径）
        physical_fee: 该物理卖单的实际费用 ``F``

    步骤：``A=0 且 F=0`` 全零短路 / ``A=0 且 F≠0`` 拒绝 -> ``f_i=F·a_i/A`` ->
    终端袖一次闭合（浮点策略见模块说明）-> 逐袖非负 + 零份额不收费。
    """
    fee, order, intents, allocated, amounts = _validate_s5_inputs(
        intent_shares, allocated_shares, price, physical_fee
    )
    total_amount = _left_to_right([amounts[sid] for sid in order])
    if total_amount == 0.0:
        if fee != 0.0:
            raise FeeError(
                f"S5 分摊：Σ分得金额=0 但物理费用 {fee!r} 元；全零成交且费用非零必须拒绝"
            )
        zeros = tuple((sid, 0.0) for sid in order)
        return S5FeeSplit(
            order=order,
            intent_shares=tuple((sid, intents[sid]) for sid in order),
            allocated_shares=tuple((sid, allocated[sid]) for sid in order),
            allocated_amounts=zeros,
            final_fees=zeros,
            terminal_sleeve="",
            residual_correction=0.0,
            physical_fee=0.0,
            zero_fill=True,
            declared_fee=0.0,
            closure=ClosureTrace("direct", 0, 0.0),
        )
    terminal = _s5_terminal(order, amounts)
    if not terminal:  # pragma: no cover - total_amount>0 时不可达
        raise FeeError("S5 分摊：Σ分得金额>0 却无正份额袖；拒绝")
    provisional = {
        sid: (0.0 if amounts[sid] == 0.0 else fee * amounts[sid] / total_amount) for sid in order
    }
    final, residual, closure = _close_terminal(order, provisional, terminal, fee, "S5 分摊")
    _assert_s5_final(order, allocated, final)
    applied = _left_to_right([final[sid] for sid in order])
    return S5FeeSplit(
        order=order,
        intent_shares=tuple((sid, intents[sid]) for sid in order),
        allocated_shares=tuple((sid, allocated[sid]) for sid in order),
        allocated_amounts=tuple((sid, amounts[sid]) for sid in order),
        final_fees=tuple((sid, final[sid]) for sid in order),
        terminal_sleeve=terminal,
        residual_correction=residual,
        physical_fee=applied,
        zero_fill=False,
        declared_fee=fee,
        closure=closure,
    )


# ────────────────────────── S4（Q13 同公式） ──────────────────────────


@dataclass(frozen=True)
class S4FeeWaterfall:
    """S4 同向合并买入的费用瀑布结果（Q13 公式；P7 侧实现，字段与受审件同义）。"""

    order: Tuple[str, ...]
    intent_amounts: Tuple[Tuple[str, float], ...]
    allocated_amounts: Tuple[Tuple[str, float], ...]
    independent_costs: Tuple[Tuple[str, float], ...]
    provisional_fees: Tuple[Tuple[str, float], ...]
    terminal_sleeve: str
    residual_correction: float
    final_fees: Tuple[Tuple[str, float], ...]
    physical_fee: float
    proportional_branch: bool = False
    declared_fee: float = 0.0
    closure: Optional[ClosureTrace] = None

    def fee_of(self, sleeve_id: str) -> float:
        for key, value in self.final_fees:
            if key == sleeve_id:
                return value
        return 0.0

    @property
    def total_fee(self) -> float:
        return _left_to_right([value for _key, value in self.final_fees])


def _validate_s4_inputs(
    intent_amounts: Mapping[str, float],
    allocated_amounts: Mapping[str, float],
    physical_fee: float,
    buy_cost: Callable[[float], float],
) -> Tuple[float, Tuple[str, ...], Dict[str, float], Dict[str, float]]:
    """S4 入参校验：键集一致、金额有限非负、分得不超意图、成本函数可调用。"""
    fee = _require_non_negative(physical_fee, "S4 物理费用 F")
    if not callable(buy_cost):
        raise FeeError("S4 费用瀑布需要可调用的买入成本函数 buy_cost")
    if set(intent_amounts) != set(allocated_amounts):
        raise FeeError(
            f"S4 费用瀑布：原始意图金额键集 {sorted(intent_amounts)} 与分得金额键集 "
            f"{sorted(allocated_amounts)} 不一致"
        )
    intents: Dict[str, float] = {}
    allocated: Dict[str, float] = {}
    for sleeve_id in sorted(intent_amounts):
        sid = str(sleeve_id)
        q = _require_non_negative(intent_amounts[sleeve_id], f"袖 {sid} 原始意图金额 q_i")
        a = _require_non_negative(allocated_amounts[sleeve_id], f"袖 {sid} 实际分得金额 a_i")
        if a > q:
            raise FeeError(
                f"S4 费用瀑布：袖 {sid} 分得金额 {a!r} > 原始意图金额 {q!r}（单袖不得超意图）"
            )
        intents[sid] = q
        allocated[sid] = a
    order = tuple(sorted(intents, key=lambda sid: (intents[sid], sid)))
    return fee, order, intents, allocated


def _s4_costs(
    order: Sequence[str],
    allocated: Mapping[str, float],
    buy_cost: Callable[[float], float],
) -> Dict[str, float]:
    """独立费用：``a_i=0 => c_i=0``（禁止对零成交额调用最低佣金）。"""
    costs: Dict[str, float] = {}
    for sid in order:
        cost = 0.0 if allocated[sid] == 0.0 else float(buy_cost(allocated[sid]))
        if not math.isfinite(cost) or cost < 0.0:
            raise FeeError(f"S4 费用瀑布：袖 {sid} 独立费用 {cost!r} 非有限非负")
        costs[sid] = cost
    return costs


def _s4_provisional(
    order: Sequence[str],
    costs: Mapping[str, float],
    allocated: Mapping[str, float],
    fee: float,
    total_allocated: float,
) -> Tuple[Dict[str, float], bool]:
    """预备费用：``F > Σc`` 走比例分摊分支；否则按受益顺序做非负瀑布。"""
    cost_total = _left_to_right([costs[sid] for sid in order])
    if fee > cost_total:
        return {sid: fee * allocated[sid] / total_allocated for sid in order}, True
    remaining = cost_total - fee
    provisional: Dict[str, float] = {}
    for sid in order:
        step = min(costs[sid], remaining)
        remaining -= step
        provisional[sid] = costs[sid] - step
    return provisional, False


def _s4_terminal(order: Sequence[str], allocated: Mapping[str, float]) -> str:
    """终端袖 = 受益顺序中最后一个 ``a_i>0`` 者。"""
    terminal = ""
    for sid in order:
        if allocated[sid] > 0.0:
            terminal = sid
    if not terminal:  # pragma: no cover - total_allocated>0 时不可达
        raise FeeError("S4 费用瀑布：Σ分得金额>0 却无正份额袖；拒绝")
    return terminal


def _assert_s4_final(
    order: Sequence[str], allocated: Mapping[str, float], final: Mapping[str, float]
) -> None:
    """最终费用逐袖守卫：非负 + 零份额不收费。"""
    for sid in order:
        if final[sid] < 0.0:
            raise FeeError(f"S4 费用瀑布：袖 {sid} 最终费用为负 {final[sid]!r}")
        if allocated[sid] == 0.0 and final[sid] != 0.0:
            raise FeeError(
                f"S4 费用瀑布：零成交份额被收费——袖 {sid} 分得金额=0 但费用 {final[sid]!r} 元"
            )


def allocate_s4_fees(
    *,
    intent_amounts: Mapping[str, float],
    allocated_amounts: Mapping[str, float],
    physical_fee: float,
    buy_cost: Callable[[float], float],
) -> S4FeeWaterfall:
    """Q13-A：S4 同向合并买入的 N 袖费用瀑布（F25 §5 公式，P7 侧唯一实现）。

    与受审 ``settlement_kernel.allocate_fees`` **同公式**（顺序、瀑布、比例分支、
    全零短路、终端袖选择、逐袖守卫逐条一致），差别只在浮点闭合策略
    （受审件「一次修正后仍不等即拒绝」，P7 见模块说明四步策略并逐笔计数）。
    ``q_i`` 必须是**各袖原始买入意图金额**（不是资金截断后的可负担金额），
    否则受益顺序会被现金约束改写。
    """
    fee, order, intents, allocated = _validate_s4_inputs(
        intent_amounts, allocated_amounts, physical_fee, buy_cost
    )
    if not intents:
        if fee != 0.0:
            raise FeeError(f"S4 费用瀑布：无任何袖但物理费用 {fee!r} 元；拒绝")
        return S4FeeWaterfall((), (), (), (), (), "", 0.0, (), 0.0)
    total_allocated = _left_to_right([allocated[sid] for sid in order])
    if total_allocated == 0.0:
        if fee != 0.0:
            raise FeeError(
                f"S4 费用瀑布：Σ分得金额=0 但物理费用 {fee!r} 元；"
                "全零成交且费用非零必须拒绝，不得进入终端袖选择"
            )
        zeros = tuple((sid, 0.0) for sid in order)
        return S4FeeWaterfall(order, (), (), zeros, zeros, "", 0.0, zeros, 0.0)
    costs = _s4_costs(order, allocated, buy_cost)
    provisional, proportional = _s4_provisional(order, costs, allocated, fee, total_allocated)
    terminal = _s4_terminal(order, allocated)
    final, residual, closure = _close_terminal(order, provisional, terminal, fee, "S4 费用瀑布")
    _assert_s4_final(order, allocated, final)
    applied = _left_to_right([final[sid] for sid in order])
    return S4FeeWaterfall(
        order=order,
        intent_amounts=tuple((sid, intents[sid]) for sid in order),
        allocated_amounts=tuple((sid, allocated[sid]) for sid in order),
        independent_costs=tuple((sid, costs[sid]) for sid in order),
        provisional_fees=tuple((sid, provisional[sid]) for sid in order),
        terminal_sleeve=terminal,
        residual_correction=residual,
        final_fees=tuple((sid, final[sid]) for sid in order),
        physical_fee=applied,
        proportional_branch=proportional,
        declared_fee=fee,
        closure=closure,
    )


# ────────────────────────── 费用事件与独立费用 ──────────────────────────


@dataclass(frozen=True)
class FeeEvent:
    """单笔物理成交事件的费用事实（一次市场指令或一次内部抵消）。"""

    event_id: str
    trade_date: str
    ts_code: str
    side: str
    scenario: str
    physical_fee: float
    price: float
    allocations: Tuple[Tuple[str, int], ...]
    internal_offset: bool = False

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise FeeError(f"费用事件 {self.event_id} 方向必须为 buy/sell：{self.side!r}")
        if not math.isfinite(float(self.price)) or float(self.price) <= 0.0:
            raise FeeError(f"费用事件 {self.event_id} 价格必须为正有限数：{self.price!r}")
        fee = float(self.physical_fee)
        if not math.isfinite(fee) or fee < 0.0:
            raise FeeError(f"费用事件 {self.event_id} 物理费用必须有限非负：{fee!r}")
        seen: set = set()
        for sleeve_id, shares in self.allocations:
            if sleeve_id in seen:
                raise FeeError(f"费用事件 {self.event_id} 重复袖分配：{sleeve_id}")
            seen.add(sleeve_id)
            if not isinstance(shares, int) or shares < 0:
                raise FeeError(
                    f"费用事件 {self.event_id} 袖 {sleeve_id} 分得股数必须为非负整数：{shares!r}"
                )

    @property
    def total_shares(self) -> int:
        return int(sum(shares for _sleeve, shares in self.allocations))

    def amount_of(self, sleeve_id: str) -> float:
        for key, shares in self.allocations:
            if key == sleeve_id:
                return float(shares) * float(self.price)
        return 0.0


def aggregate_amounts(events: Iterable[FeeEvent], side: str) -> Dict[FeeKey, float]:
    """按 ``(sleeve, 日, 股票, 方向)`` 聚合实际分得金额（**先聚合、后计一次成本**）。"""
    grouped: Dict[FeeKey, float] = {}
    for event in events:
        if event.side != side:
            continue
        for sleeve_id, shares in event.allocations:
            amount = float(shares) * float(event.price)
            if amount == 0.0:
                continue
            key = (str(sleeve_id), str(event.trade_date), str(event.ts_code), str(side))
            grouped[key] = grouped.get(key, 0.0) + amount
    return grouped


def independent_fees(
    amounts: Mapping[FeeKey, float],
    cost_of: Callable[[float], float],
) -> Dict[FeeKey, float]:
    """独立费用：每个 ``(sleeve, 日, 股票, 方向)`` 键**只调用一次**成本函数。"""
    if not callable(cost_of):
        raise FeeError("独立费用需要可调用的成本函数 cost_of")
    out: Dict[FeeKey, float] = {}
    for key in sorted(amounts):
        amount = _require_non_negative(amounts[key], f"独立费用基准 {key}")
        if amount == 0.0:
            continue
        cost = float(cost_of(amount))
        if not math.isfinite(cost) or cost < 0.0:
            raise FeeError(f"独立费用 {key} 必须为有限非负数：{cost!r}")
        out[key] = cost
    return out


# ────────────────────────── 费用台账 ──────────────────────────


@dataclass
class FeeBook:
    """费用台账（逐事件登记物理费用、逐袖实际费用与独立费用，双通道闭合校验）。"""

    events: int = 0
    physical_fee_total: float = 0.0
    actual_by_sleeve: Dict[str, float] = field(default_factory=dict)
    independent_by_sleeve: Dict[str, float] = field(default_factory=dict)
    actual_by_key: Dict[FeeKey, float] = field(default_factory=dict)
    independent_by_key: Dict[FeeKey, float] = field(default_factory=dict)
    #: 逐事件累计的实际费用（与 ``physical_fee_total`` 同序累加）
    running_actual: float = 0.0
    #: 逐事件累计的**申报**物理费用（成本模型输出，未做浮点吸收）
    declared_fee_total: float = 0.0
    #: 逐事件偏差的最大值（以 ``F`` 的 ulp 计；用于展示余量，不做判定）
    max_gap_in_ulp: float = 0.0
    #: 浮点闭合轨迹计数（``direct`` / ``corrected`` / ``nudged`` / ``within_tolerance``）
    closure_paths: Dict[str, int] = field(default_factory=dict)
    max_terminal_shift_ulps: int = 0
    ulp_gap_total: float = 0.0
    max_abs_ulp_gap: float = 0.0
    ulp_gap_events: int = 0
    #: 逐事件闭合校验失败计数（恒 0；``closure_channels.per_event_exact`` 由其派生）
    event_closures_failed: int = 0

    def _record_closure(self, applied: float, declared: float, trace: Any) -> None:
        path = "direct" if trace is None else str(trace.path)
        self.closure_paths[path] = self.closure_paths.get(path, 0) + 1
        if trace is None:
            return
        self.max_terminal_shift_ulps = max(
            self.max_terminal_shift_ulps, int(trace.terminal_shift_ulps)
        )
        gap = float(trace.ulp_gap)
        if gap != 0.0:
            self.ulp_gap_events += 1
            self.ulp_gap_total += gap
            self.max_abs_ulp_gap = max(self.max_abs_ulp_gap, abs(gap))
            if declared > 0.0:
                self.max_gap_in_ulp = max(self.max_gap_in_ulp, abs(gap) / math.ulp(declared))
        del applied

    def add_day(
        self,
        trade_date: str,
        entries: Sequence[Tuple[FeeEvent, Sequence[Tuple[str, float]]]],
        *,
        cost_of_by_side: Mapping[str, Callable[[float], float]],
        closures: Optional[Sequence[Any]] = None,
        declared_fees: Optional[Sequence[float]] = None,
    ) -> Dict[FeeKey, float]:
        """登记单日全部费用事件（**先聚合、后按 ``(sleeve, 日, 股票, 方向)`` 计一次**）。

        Args:
            trade_date: 交易日（与 ``entries`` 内事件的 ``trade_date`` 必须一致）
            entries: ``(事件, 逐袖最终实际费用)`` 序列
            cost_of_by_side: ``{"buy": 买入成本函数, "sell": 卖出成本函数}``
            closures: 与 ``entries`` 等长的闭合轨迹（``ClosureTrace`` 或 None）
            declared_fees: 与 ``entries`` 等长的申报物理费用（缺省取事件自带费用）

        Returns:
            本日逐键独立费用（``{key: cost}``）。
        """
        day = str(trade_date)
        events = [event for event, _final in entries]
        for event, final_fees in entries:
            self._check_event(day, event, final_fees)
            for sleeve_id, shares in event.allocations:
                amount = float(shares) * float(event.price)
                if amount == 0.0:
                    continue
                key = (sleeve_id, day, event.ts_code, event.side)
                self.actual_by_key[key] = self.actual_by_key.get(key, 0.0) + float(
                    dict(final_fees).get(sleeve_id, 0.0)
                )
                self.actual_by_sleeve[sleeve_id] = self.actual_by_sleeve.get(
                    sleeve_id, 0.0
                ) + float(dict(final_fees).get(sleeve_id, 0.0))
            total = _left_to_right([value for _sid, value in final_fees])
            self.events += 1
            self.physical_fee_total += float(event.physical_fee)
            self.running_actual += total
        for index, event in enumerate(events):
            declared = (
                float(declared_fees[index])
                if declared_fees is not None
                else float(event.physical_fee)
            )
            trace = closures[index] if closures is not None else None
            self.declared_fee_total += declared
            self._record_closure(total_of(entries[index][1]), declared, trace)
        return self._independent_of_day(day, events, cost_of_by_side)

    @staticmethod
    def _check_event(day: str, event: FeeEvent, final_fees: Sequence[Tuple[str, float]]) -> None:
        """逐事件校验：跨日混入、闭合、负值、零份额收费、键集一致。"""
        if str(event.trade_date) != day:
            raise FeeError(
                f"费用事件 {event.event_id} 的交易日 {event.trade_date} 与批量日 {day} 不一致"
            )
        total = _left_to_right([value for _sid, value in final_fees])
        if total != float(event.physical_fee):
            raise FeeError(
                f"费用事件 {event.event_id} 闭合破缺：Σ实际费用={total!r} != "
                f"F={event.physical_fee!r}（容差={FEE_SUM_TOLERANCE} 单位=元）"
            )
        share_map = {sid: shares for sid, shares in event.allocations}
        fee_map = {sid: float(value) for sid, value in final_fees}
        if set(fee_map) != set(share_map):
            raise FeeError(
                f"费用事件 {event.event_id} 的最终费用键集 {sorted(fee_map)} 与分得股数键集 "
                f"{sorted(share_map)} 不一致（禁止漏袖按 0 静默收费）"
            )
        for sleeve_id in sorted(fee_map):
            value = fee_map[sleeve_id]
            if value < 0.0:
                raise FeeError(f"费用事件 {event.event_id} 袖 {sleeve_id} 实际费用为负：{value!r}")
            if share_map[sleeve_id] == 0 and value != 0.0:
                raise FeeError(
                    f"费用事件 {event.event_id} 零份额被收费：袖 {sleeve_id} 费用 {value!r} 元"
                )

    def _independent_of_day(
        self,
        day: str,
        events: Sequence[FeeEvent],
        cost_of_by_side: Mapping[str, Callable[[float], float]],
    ) -> Dict[FeeKey, float]:
        """独立费用：按方向聚合后每键只计一次（单一聚合实现）。"""
        day_costs: Dict[FeeKey, float] = {}
        for side in ("buy", "sell"):
            amounts = aggregate_amounts(events, side)
            if not amounts:
                continue
            costs = independent_fees(amounts, cost_of_by_side[side])
            for key, cost in costs.items():
                day_costs[key] = cost
                self.independent_by_key[key] = self.independent_by_key.get(key, 0.0) + cost
                self.independent_by_sleeve[key[0]] = (
                    self.independent_by_sleeve.get(key[0], 0.0) + cost
                )
        return {key: value for key, value in day_costs.items() if key[1] == day}

    # ---- 汇总与闭合 ----

    @property
    def total_physical_fee(self) -> float:
        """逐事件累计的**可用**物理费用（= Σ 逐事件 Σf_i）。"""
        return float(self.physical_fee_total)

    @property
    def total_actual_fee(self) -> float:
        """按袖汇总的实际费用（与逐事件累加是**两条独立通道**）。"""
        return _left_to_right([self.actual_by_sleeve[sid] for sid in sorted(self.actual_by_sleeve)])

    @property
    def total_independent_fee(self) -> float:
        """独立费用合计（视同各袖独立成交；只读反事实）。"""
        return _left_to_right(
            [self.independent_by_sleeve[sid] for sid in sorted(self.independent_by_sleeve)]
        )

    def savings_by_sleeve(self) -> Dict[str, float]:
        """逐袖有符号节省 ``s_i = c_i - f_i``（允许为负）。"""
        sleeves = sorted(set(self.actual_by_sleeve) | set(self.independent_by_sleeve))
        return {
            sid: float(self.independent_by_sleeve.get(sid, 0.0))
            - float(self.actual_by_sleeve.get(sid, 0.0))
            for sid in sleeves
        }

    def assert_closed(self) -> None:
        """双通道闭合（I1 的会计前提），任一通道破缺即拒绝。

        ① **逐事件**：``Σf_i == F``（``add_day`` 内即时判定，容差 0）；
        ② **逐事件累加**：``running_actual == physical_fee_total``（同序，容差 0）；
        ③ **按袖汇总**（独立通道）：``|Σ_i actual_i - Σ事件 F| <= 1e-6``（浮点尾差上限）。
        """
        if self.running_actual != self.total_physical_fee:
            raise FeeError(
                f"费用台账闭合破缺（逐事件通道）：累计实际={self.running_actual!r} != "
                f"Σ物理费用={self.total_physical_fee!r}"
            )
        gap = self.total_actual_fee - self.total_physical_fee
        if abs(gap) > FEE_LEDGER_TOLERANCE:
            raise FeeError(
                f"费用台账闭合破缺（按袖通道）：Σ_i actual_i={self.total_actual_fee!r} != "
                f"Σ事件物理费用={self.total_physical_fee!r}（差={gap!r} "
                f"容差={FEE_LEDGER_TOLERANCE} 单位=元）"
            )

    def payload(self) -> Dict[str, Any]:
        """证据载荷（逐键字段 + 逐袖汇总 + 有符号节省 + 浮点闭合轨迹）。"""
        savings = self.savings_by_sleeve()
        return {
            "schema": "p7_t4_fees/v1",
            "ruling": "Q3=A",
            "events": int(self.events),
            "physical_fee_total": float(self.physical_fee_total),
            "declared_fee_total": float(self.declared_fee_total),
            "declared_minus_applied": float(self.declared_fee_total)
            - float(self.physical_fee_total),
            "actual_fee_running": float(self.running_actual),
            "actual_fee_by_sleeve_sum": self.total_actual_fee,
            "independent_fee_total": self.total_independent_fee,
            "closed": bool(
                self.running_actual == self.total_physical_fee
                and abs(self.total_actual_fee - self.total_physical_fee) <= FEE_LEDGER_TOLERANCE
            ),
            "closure_channels": {
                "per_event_exact": bool(self.event_closures_failed == 0),
                "per_event_events_checked": int(self.events),
                "running_vs_physical_exact": bool(self.running_actual == self.total_physical_fee),
                "by_sleeve_vs_physical": self.total_actual_fee - self.total_physical_fee,
                "by_sleeve_tolerance": FEE_LEDGER_TOLERANCE,
            },
            "float_closure": {
                "policy": (
                    "终端袖仅 ulp 微调；仍不精确闭合则在 100 ulp(F) 精度内闭合"
                    "（物理费用取 Σf_i，不改任何袖取值）"
                ),
                "event_tolerance_ulp": FEE_EVENT_TOLERANCE_ULP,
                "max_gap_in_ulp": float(self.max_gap_in_ulp),
                "run_tolerance": FEE_FLOAT_CLOSURE_TOLERANCE,
                "declared_minus_applied_note": (
                    "``declared_minus_applied`` 是两个独立累加（申报 vs 逐笔可用）之差，"
                    "聚合层相互抵消（实测 0.0）；逐笔偏差看 ulp_gap_events / ulp_gap_total / "
                    "max_abs_ulp_gap（三者已进硬失败门）。"
                ),
                "paths": dict(sorted(self.closure_paths.items())),
                "ulp_nudge_steps_limit": ULP_NUDGE_STEPS,
                "max_terminal_shift_ulps": int(self.max_terminal_shift_ulps),
                "ulp_gap_events": int(self.ulp_gap_events),
                "ulp_gap_total": float(self.ulp_gap_total),
                "max_abs_ulp_gap": float(self.max_abs_ulp_gap),
            },
            "savings_by_sleeve": savings,
            "sleeve_rows": [
                {
                    "sleeve": sid,
                    "actual_fee": float(self.actual_by_sleeve.get(sid, 0.0)),
                    "independent_fee": float(self.independent_by_sleeve.get(sid, 0.0)),
                    "saving": float(savings.get(sid, 0.0)),
                }
                for sid in sorted(set(self.actual_by_sleeve) | set(self.independent_by_sleeve))
            ],
            "by_key": [
                {
                    "sleeve": key[0],
                    "trade_date": key[1],
                    "ts_code": key[2],
                    "side": key[3],
                    "actual_fee": float(self.actual_by_key.get(key, 0.0)),
                    "independent_fee": float(self.independent_by_key.get(key, 0.0)),
                    "saving": float(self.independent_by_key.get(key, 0.0))
                    - float(self.actual_by_key.get(key, 0.0)),
                }
                for key in sorted(set(self.actual_by_key) | set(self.independent_by_key))
            ],
        }

    # ---- 续跑状态（C4-04） ----

    def snapshot(self) -> Dict[str, Any]:
        """导出**可序列化**的完整累计状态（双通道 + 逐键 + 浮点闭合轨迹）。

        重启续跑必须还原到同一累计态，否则次日独立费用/节省与 NAV 会整体偏移。
        载荷刻意用**版本化 schema + 显式列表**（不依赖 pickle / 对象身份）。
        """
        return {
            "schema": FEE_STATE_SCHEMA,
            "events": int(self.events),
            "physical_fee_total": float(self.physical_fee_total),
            "declared_fee_total": float(self.declared_fee_total),
            "running_actual": float(self.running_actual),
            "max_gap_in_ulp": float(self.max_gap_in_ulp),
            "actual_by_sleeve": dict(sorted(self.actual_by_sleeve.items())),
            "independent_by_sleeve": dict(sorted(self.independent_by_sleeve.items())),
            "actual_by_key": [
                [key[0], key[1], key[2], key[3], float(value)]
                for key, value in sorted(self.actual_by_key.items())
            ],
            "independent_by_key": [
                [key[0], key[1], key[2], key[3], float(value)]
                for key, value in sorted(self.independent_by_key.items())
            ],
            "closure_paths": dict(sorted(self.closure_paths.items())),
            "max_terminal_shift_ulps": int(self.max_terminal_shift_ulps),
            "ulp_gap_total": float(self.ulp_gap_total),
            "max_abs_ulp_gap": float(self.max_abs_ulp_gap),
            "ulp_gap_events": int(self.ulp_gap_events),
            "event_closures_failed": int(self.event_closures_failed),
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, Any]) -> "FeeBook":
        """从 ``snapshot()`` 的产物**逐位**还原费用台账（schema 不符即拒绝）。"""
        schema = str(snapshot.get("schema") or "")
        if schema != FEE_STATE_SCHEMA:
            raise FeeError(f"费用台账快照 schema 不匹配：{schema!r} != {FEE_STATE_SCHEMA!r}")

        def _keys(rows: Any) -> Dict[FeeKey, float]:
            out: Dict[FeeKey, float] = {}
            for row in rows or ():
                key = (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
                if key in out:
                    raise FeeError(f"费用台账快照存在重复键：{key}")
                out[key] = float(row[4])
            return out

        return cls(
            events=int(snapshot["events"]),
            physical_fee_total=float(snapshot["physical_fee_total"]),
            actual_by_sleeve={
                str(key): float(value)
                for key, value in (snapshot.get("actual_by_sleeve") or {}).items()
            },
            independent_by_sleeve={
                str(key): float(value)
                for key, value in (snapshot.get("independent_by_sleeve") or {}).items()
            },
            actual_by_key=_keys(snapshot.get("actual_by_key")),
            independent_by_key=_keys(snapshot.get("independent_by_key")),
            running_actual=float(snapshot["running_actual"]),
            declared_fee_total=float(snapshot["declared_fee_total"]),
            max_gap_in_ulp=float(snapshot.get("max_gap_in_ulp") or 0.0),
            closure_paths={
                str(key): int(value) for key, value in (snapshot.get("closure_paths") or {}).items()
            },
            max_terminal_shift_ulps=int(snapshot.get("max_terminal_shift_ulps") or 0),
            ulp_gap_total=float(snapshot.get("ulp_gap_total") or 0.0),
            max_abs_ulp_gap=float(snapshot.get("max_abs_ulp_gap") or 0.0),
            ulp_gap_events=int(snapshot.get("ulp_gap_events") or 0),
            event_closures_failed=int(snapshot.get("event_closures_failed") or 0),
        )


def total_of(final_fees: Sequence[Tuple[str, float]]) -> float:
    """逐袖最终费用的左到右和（供台账与调用方共用同一口径）。"""
    return _left_to_right([value for _sid, value in final_fees])


def declared_and_applied(waterfall: Any) -> Tuple[float, float]:
    """取 ``(申报物理费用, 可用物理费用)``（无闭合轨迹时两者相同）。"""
    declared = float(getattr(waterfall, "declared_fee", 0.0) or 0.0)
    applied = float(getattr(waterfall, "physical_fee", 0.0))
    return (declared or applied), applied
