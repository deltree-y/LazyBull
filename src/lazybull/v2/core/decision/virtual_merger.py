"""多袖组合合并器（P6.5 T2：S1/S4/S6 + 公开净额接口的物理全量目标）。

契约依据：
- ``docs/contracts/netting_freeze.md`` F25 §2/§3/§5（S1~S7 分区、S4 同向分摊、
  S6 仅真正单袖、Q18 意图上界）；
- ``docs/contracts/protocols.md`` F13 §3.4（``PortfolioMerger``、``net_across_sleeves``
  的多袖虚拟账户上下文与 C65-01 物理全量目标）；
- 设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md``
  §4.2/§4.4/§6.1/§6.2/§8.2/§8.3 与 §10.1 G1。

能力边界（T2）：
- ``merge``：多袖目标 → 单一物理目标。构造期配额（``[0,1]`` 无量纲比例）在这里
  校验，**只约束目标合成，绝不作为费用分摊比例**；每个袖的 ``target_weights``
  同样按 ``math.fsum`` 校验为 1，通过后**保持权重对象、顺序与位模式不变**，
  不重新归一化、不排序改写。
- ``net_across_sleeves``：按 F13 恢复**物理全量目标**——对每只股票
  ``物理目标 = Σ_所有配置袖当前虚拟持仓 + Σ_本批各袖解析增量``；本批不发单的袖
  仍以当前虚拟持仓计入基数。净增量为 0 时只做虚拟内部转账（``NettingPlan.offsets``），
  **不产生市场指令**。物理订单的袖归属/分摊事实只由内部结算批携带，禁止从聚合后的
  ``source_sleeve`` 反推（本实现用显式伪袖 ``<physical>`` 标注）。
- 状态机：S1 等量反向、S4 同向买入、S6 仅真正单袖直通；S2/S3（反向不等量）、
  S5（同向卖出）、S7（净残差）一律**零副作用拒绝**。**S1 本期严格一对等量反向**
  （一卖一买）；"多卖一买/一卖多买且净零"属未登记推广，显式拒绝并归 P7——账本层
  （``settlement_batch``）按同一口径拒绝，两层判定不得相反。
- 保留伪袖：``MERGED_SLEEVE_ID``（合并目标）与 ``PHYSICAL_SLEEVE_ID``（物理指令）
  只是内部标记，**不得**回喂为 ``source_sleeve`` 或写进袖配置/信号映射；袖级归属
  只由内部结算批次携带。

不实现：订单→执行→物理入账全链（T3）、跨袖借用（FundScheduler）、公司行为、
S2/S3/S5/S7 生命周期。本类不进入生产默认路径（`hosts/backtest/runtime.py` 与
`PaperHost` 不变），只由合成测试显式构造。

公开净额契约的三条拒绝面（F13）：物理全量目标不得退化为发单袖目标、不得遗漏
不动作袖基数、不得在多袖账户上静默走 S6。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.common.types import (
    Account,
    Order,
    OrderReason,
    OrderSide,
    OrderType,
    Shares,
    TargetPortfolio,
    TSCode,
    VirtualAccount,
    Weight,
)
from src.lazybull.v2.core.accounting.virtual_ledger import SleeveIntent, TransferFact

#: 本模块唯一阶段标识（进入全部错误上下文）
_STAGE = "P6.5"
#: 合并后物理目标的袖标识（显式伪袖：物理订单不代表任一子袖）
MERGED_SLEEVE_ID = "<merged>"
#: 物理订单来源标识（分摊事实只在内部结算批次内，禁止据此反推归属）
PHYSICAL_SLEEVE_ID = "<physical>"
#: float64 单位舍入误差（多袖总和容差 = n × 2^-52）
_FLOAT64_EPS = 2.0**-52


class VirtualMergerError(ValueError):
    """多袖合并输入非法（**零副作用**拒绝，错误含阶段与状态机编号）。"""


def _tolerance(n_sleeves: int) -> float:
    """多袖总和容差 ``n × 2^-52``（n = 袖数；float64 单位舍入的保守上界）。"""
    return max(1, int(n_sleeves)) * _FLOAT64_EPS


def _as_int(value: Any, field: str) -> int:
    """严格整数解析（拒绝 bool/float；``numpy.integer`` 经 ``__index__`` 接受）。"""
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise VirtualMergerError(
            f"阶段={_STAGE} {field} 必须为整数股，实际 {type(value).__name__}={value!r}"
        )
    number = int(value.__index__())
    if number < 0:
        raise VirtualMergerError(f"阶段={_STAGE} {field} 必须非负，实际 {number}")
    return number


def _as_finite_weight(value: Any, field: str) -> float:
    """有限非负权重解析（NaN/Inf/负值一律拒绝）。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise VirtualMergerError(
            f"阶段={_STAGE} {field} 必须为有限非负数，实际 {value!r}（{error}）"
        ) from error
    if not math.isfinite(number) or number < 0:
        raise VirtualMergerError(
            f"阶段={_STAGE} {field} 必须为有限非负数，实际 {value!r}"
        )
    return number


def _assert_merged_total(merged_values: Mapping[str, float], tolerance: float) -> None:
    """合成后权重总和守卫：``TargetPortfolio.target_weights`` 的上游不变量为 1。

    配额参与合成后（``Σ quota_i × w_i``）总和理论上仍为 1；本守卫拒绝任何因实现
    漂移而返回 Σ≠1 目标的路径（B-R1-01：不得静默返回 Σ=N 的合并目标）。

    **口径登记（B-R3-02）**：逐袖门与配额门各允许 ``n×2^-52`` 的输入残差，两项叠加
    后合并结果的理论上界可达 ``2×n×2^-52``。本守卫仍按 ``n×2^-52`` 判定，是**比
    逐袖门更严的保守界**：极端贴边输入（两袖各自恰好落在容差边界）可能被本门拒绝。
    该保守行为是 fail-closed 的设计选择（宁可拒绝也不返回 Σ≠1 的目标），不是漏判；
    若未来契约给出复合容差口径，须先改契约定值再改本实现。
    """
    total = math.fsum(merged_values[key] for key in sorted(merged_values))
    if abs(total - 1.0) > tolerance:
        raise VirtualMergerError(
            f"阶段={_STAGE} 多袖 merge 合成后权重 fsum={total!r} != 1，差="
            f"{total - 1.0!r} 容差={tolerance}；配额参与合成后仍须满足 "
            "TargetPortfolio 上游不变量，不得静默返回 Σ≠1 的目标"
        )


@dataclass(frozen=True, slots=True)
class InternalOffset:
    """S1 内部抵消事实（等量反向：零物理指令、双方零费用）。

    该事实由 ``net_across_sleeves`` 的净额计划给出，经内部结算批次落账；
    合并器自身不写虚拟账（写点唯一在 ``VirtualLedger``）。
    """

    ts_code: str
    from_sleeve: str
    to_sleeve: str
    shares: int

    def __post_init__(self) -> None:
        if not str(self.ts_code).strip():
            raise VirtualMergerError(f"阶段={_STAGE} S1 内部抵消的 ts_code 不得为空")
        if self.from_sleeve == self.to_sleeve:
            raise VirtualMergerError(
                f"阶段={_STAGE} S1 内部抵消的转出/转入袖相同（{self.from_sleeve}）"
            )
        if isinstance(self.shares, bool) or not hasattr(self.shares, "__index__"):
            raise VirtualMergerError(
                f"阶段={_STAGE} S1 内部抵消股数必须为整数，实际 {self.shares!r}"
            )
        object.__setattr__(self, "shares", _as_int(self.shares, "S1 内部抵消股数"))
        if self.shares <= 0:
            raise VirtualMergerError(
                f"阶段={_STAGE} S1 内部抵消股数必须为正，实际 {self.shares}"
            )


@dataclass(frozen=True, slots=True)
class NettingPlan:
    """净额计划（原始意图 + 物理全量目标 + 内部抵消事实）。

    ``intents`` 直接复用内部结算批次的 ``SleeveIntent`` 结构，避免同一原始意图
    在合并器与账本之间出现第二套形态。
    """

    intents: Tuple[SleeveIntent, ...]
    physical_current: Mapping[str, int]
    physical_targets: Mapping[str, int]
    net_by_code: Mapping[str, int]
    offsets: Tuple[InternalOffset, ...]
    orders: Tuple[Order, ...]

    def physical_orders(self) -> Tuple[Order, ...]:
        """物理市场指令（``target_shares`` = 物理账户全量目标）。"""
        return self.orders

    def transfer_facts(self, prices: Mapping[str, float]) -> Tuple[TransferFact, ...]:
        """把内部抵消事实转成结算批次的转账事实（价格 = 当日同一价源）。"""
        facts: List[TransferFact] = []
        for offset in self.offsets:
            price = prices.get(offset.ts_code)
            if price is None:
                raise VirtualMergerError(
                    f"阶段={_STAGE} 股票={offset.ts_code}：内部抵消缺当日价源价格；"
                    "P6.5 无价即拒绝，不顺延"
                )
            facts.append(
                TransferFact(
                    ts_code=offset.ts_code,
                    from_sleeve=offset.from_sleeve,
                    to_sleeve=offset.to_sleeve,
                    shares=offset.shares,
                    price=float(price),
                )
            )
        return tuple(facts)


class MultiSleevePortfolioMerger:
    """多袖合并器（P6.5 T2；``PortfolioMerger`` 协议的结构兼容实现）。

    构造期注入（不可变）：``sleeve_ids`` 与 ``quotas``（配额是目标合成约束，
    不进入费用分摊）。计数只记录通过全部校验并真实执行的 merge/net 调用。
    """

    def __init__(self, sleeve_ids: Sequence[str], quotas: Mapping[str, float]) -> None:
        ids = tuple(str(sleeve_id) for sleeve_id in sleeve_ids)
        if not ids:
            raise VirtualMergerError(f"阶段={_STAGE} 多袖合并器至少需要一个袖 ID")
        if any(not sleeve_id.strip() for sleeve_id in ids):
            raise VirtualMergerError(f"阶段={_STAGE} 袖 ID 不得为空：{list(ids)}")
        if len(set(ids)) != len(ids):
            raise VirtualMergerError(f"阶段={_STAGE} 袖 ID 重复：{list(ids)}")
        self._sleeve_ids: Tuple[str, ...] = tuple(sorted(ids))
        self._id_set = frozenset(self._sleeve_ids)
        quota_values: Dict[str, float] = {}
        if set(quotas) != set(self._sleeve_ids):
            raise VirtualMergerError(
                f"阶段={_STAGE} 配额键集={sorted(quotas)} 与袖 ID 集="
                f"{list(self._sleeve_ids)} 不一致（缺失/多余均拒绝）"
            )
        for sleeve_id in self._sleeve_ids:
            value = _as_finite_weight(quotas[sleeve_id], f"袖 {sleeve_id} 构造期配额")
            if value > 1.0:
                raise VirtualMergerError(
                    f"阶段={_STAGE} 袖 {sleeve_id} 构造期配额 {value!r} 超出 [0,1]；"
                    "配额是无量纲比例，不进入费用分摊"
                )
            quota_values[sleeve_id] = value
        total = math.fsum(quota_values.values())
        tolerance = _tolerance(len(self._sleeve_ids))
        if abs(total - 1.0) > tolerance:
            raise VirtualMergerError(
                f"阶段={_STAGE} 构造期配额总和 fsum={total!r} != 1，差={total - 1.0!r} "
                f"容差={tolerance}（n×2^-52，n={len(self._sleeve_ids)}）"
            )
        self._quotas: Mapping[str, float] = MappingProxyType(quota_values)
        self.merge_call_count = 0
        self.net_call_count = 0

    # ────────────────────────── 只读属性 ──────────────────────────

    @property
    def sleeve_ids(self) -> Tuple[str, ...]:
        """构造期袖 ID（字典序，确定性）。"""
        return self._sleeve_ids

    @property
    def quotas(self) -> Mapping[str, float]:
        """构造期配额（只读投影）。"""
        return self._quotas

    @property
    def is_single_sleeve(self) -> bool:
        """是否"真正单袖"（S6 直通的唯一前提）。"""
        return len(self._sleeve_ids) == 1

    # ────────────────────────── 目标合并 ──────────────────────────

    def merge(
        self,
        targets: Mapping[str, TargetPortfolio],
        current: Account,
    ) -> TargetPortfolio:
        """多袖目标 → 单一物理目标（S6 仅真正单袖直通）。

        多袖校验（F13 / C65-02）：键集必须与构造期袖 ID 完全一致；每个
        ``target_weights`` 有限、非负，``math.fsum`` 之和以 ``n×2^-52`` 容差校验为
        1；``as_of_date`` 必须一致。

        合成口径（配额参与的**唯一**位置）：``w_merged(x) = Σ_i quota_i × w_i(x)``，
        按构造期袖顺序（字典序）左到右累加。配额只在此处参与目标合成，**不得**
        作为费用分摊比例传到账本；各袖原始权重对象/顺序/位模式不改写、不重新
        归一化（乘配额是合成，不是归一化）。输出键序 = 各袖插入序的首次出现
        次序；合并结果之和必须以同一 ``n×2^-52`` 容差为 1（``TargetPortfolio``
        的上游不变量），越界即拒绝而不是静默返回 Σ≠1 的目标。

        强等价（S6）：真正单袖时返回 ``targets[sleeve_id]`` 对象本身，权重映射
        对象与顺序不变，不读取 ``current``、不新增权重判定分支。

        零副作用：多袖分支只读输入（targets/权重/配额/账户均不改写），计数仅在
        全部校验通过后 +1；任何拒绝路径都不产生副作用（B-R4-07）。``current`` 在
        多袖分支本阶段**不读取**：集中度约束仍归 ``DecisionMaker.apply_constraints``，
        本层不重复实现。

        Raises:
            VirtualMergerError（ValueError 语义）：空/多目标映射、键袖不一致、
                权重非有限/负值/总和越界、``as_of_date`` 不一致、合并后总和越界；
                另含构造期配额被替换时的防御性复核失败。
        """
        self._assert_quotas()
        if not isinstance(targets, Mapping):
            raise VirtualMergerError(
                f"阶段={_STAGE} merge 需要 Mapping[str, TargetPortfolio]，实际 "
                f"{type(targets).__name__}"
            )
        if self.is_single_sleeve:
            return self._merge_single(targets)
        if set(targets) != set(self._sleeve_ids):
            raise VirtualMergerError(
                f"阶段={_STAGE} 多袖 merge 拒绝键集不一致：targets 键={sorted(targets)}，"
                f"构造期袖={list(self._sleeve_ids)}（缺失/多余均拒绝）"
            )
        tolerance = _tolerance(len(self._sleeve_ids))
        dates = set()
        for sleeve_id in self._sleeve_ids:
            target = targets[sleeve_id]
            if not isinstance(target, TargetPortfolio):
                raise VirtualMergerError(
                    f"阶段={_STAGE} 袖 {sleeve_id} 的目标必须为 TargetPortfolio，实际 "
                    f"{type(target).__name__}"
                )
            if target.sleeve_id != sleeve_id:
                raise VirtualMergerError(
                    f"阶段={_STAGE} 映射键 {sleeve_id!r} != target.sleeve_id "
                    f"{target.sleeve_id!r}"
                )
            self._assert_weights(sleeve_id, target.target_weights, tolerance)
            dates.add(target.as_of_date)
        if len(dates) != 1:
            raise VirtualMergerError(
                f"阶段={_STAGE} 多袖目标 as_of_date 不一致：{sorted(str(item) for item in dates)}；"
                "禁止合并不同决策日的目标"
            )
        merged_values: Dict[str, float] = {}
        merged_order: List[str] = []
        for sleeve_id in self._sleeve_ids:
            quota = self._quotas[sleeve_id]
            for code, weight in targets[sleeve_id].target_weights.items():
                key = str(code.value)
                value = quota * float(weight.value)
                if key in merged_values:
                    merged_values[key] += value
                else:
                    merged_values[key] = value
                    merged_order.append(key)
        _assert_merged_total(merged_values, tolerance)
        weights: Dict[TSCode, Weight] = {
            TSCode(code): Weight(merged_values[code]) for code in merged_order
        }
        self.merge_call_count += 1
        return TargetPortfolio(
            sleeve_id=MERGED_SLEEVE_ID,
            target_weights=MappingProxyType(weights),
            as_of_date=next(iter(dates)),
        )

    def _merge_single(self, targets: Mapping[str, TargetPortfolio]) -> TargetPortfolio:
        """S6：真正单袖直通（返回对象本身，权重与顺序不变）。"""
        sleeve_id = self._sleeve_ids[0]
        if set(targets) != {sleeve_id}:
            raise VirtualMergerError(
                f"阶段={_STAGE} S6 单袖直通拒绝键集不一致：targets 键={sorted(targets)}，"
                f"构造期袖={[sleeve_id]}"
            )
        target = targets[sleeve_id]
        if not isinstance(target, TargetPortfolio):
            raise VirtualMergerError(
                f"阶段={_STAGE} 袖 {sleeve_id} 的目标必须为 TargetPortfolio，实际 "
                f"{type(target).__name__}"
            )
        if target.sleeve_id != sleeve_id:
            raise VirtualMergerError(
                f"阶段={_STAGE} 映射键 {sleeve_id!r} != target.sleeve_id "
                f"{target.sleeve_id!r}"
            )
        self.merge_call_count += 1
        return target

    @staticmethod
    def _assert_weights(
        sleeve_id: str, weights: Mapping[TSCode, Weight], tolerance: float
    ) -> None:
        """单袖目标权重校验：键/值类型、有限、非负、``fsum`` 以 n×2^-52 容差为 1。

        T2-P3-6：非 ``TSCode`` 键与非法权重值必须在这里以 ``VirtualMergerError`` 拒绝，
        不得放行到合成循环再抛 ``AttributeError``（拒绝面错误类型漂移）。
        """
        if not isinstance(weights, Mapping):
            raise VirtualMergerError(
                f"阶段={_STAGE} 袖 {sleeve_id} 的 target_weights 必须为 Mapping，实际 "
                f"{type(weights).__name__}"
            )
        values: Dict[str, float] = {}
        for code, weight in weights.items():
            if not isinstance(code, TSCode):
                raise VirtualMergerError(
                    f"阶段={_STAGE} 袖 {sleeve_id} 的目标权重键必须为 TSCode，实际 "
                    f"{type(code).__name__}={code!r}（禁止用字符串键冒充）"
                )
            if not isinstance(weight, Weight):
                raise VirtualMergerError(
                    f"阶段={_STAGE} 袖 {sleeve_id} 股票 {code.value} 的权重必须为 Weight，实际 "
                    f"{type(weight).__name__}={weight!r}"
                )
            key = str(code.value)
            values[key] = _as_finite_weight(weight.value, f"袖 {sleeve_id} 股票 {key} 目标权重")
        total = math.fsum(values[key] for key in sorted(values))
        if abs(total - 1.0) > tolerance:
            raise VirtualMergerError(
                f"阶段={_STAGE} 袖 {sleeve_id} 目标权重 fsum={total!r} != 1，"
                f"差={total - 1.0!r} 容差={tolerance}；通过后不重新归一化"
            )

    def _assert_quotas(self) -> None:
        """构造期配额不变式复核（防御性，正常路径不可达）。

        ``_quotas`` 是构造期局部 dict 上的 ``MappingProxyType``，外部无引用可改写，
        因此"配额漂移"分支只在属性被整体替换时才可能触发（B-R4-04 登记）；
        保留它是为了在子类/反射改写后仍 fail-closed，不构成对输入的常规校验。
        """
        total = math.fsum(self._quotas.values())
        tolerance = _tolerance(len(self._sleeve_ids))
        if abs(total - 1.0) > tolerance:
            raise VirtualMergerError(
                f"阶段={_STAGE} 构造期配额总和漂移：fsum={total!r} 容差={tolerance}"
            )

    # ────────────────────────── 公开净额接口 ──────────────────────────

    def net_across_sleeves(
        self,
        orders: Sequence[Order],
        virtual_accounts: Optional[Mapping[str, VirtualAccount]] = None,
    ) -> Sequence[Order]:
        """跨袖净额化（F13 / Q11=A）：返回**物理账户全量目标**指令。

        真正单袖（构造期只有一袖）时按 S6 原样透传同一序列对象；多袖时必须
        提供 ``virtual_accounts``（ID 集与构造期配置一致），否则拒绝——不得
        在多袖账户上静默走 S6。

        Raises:
            VirtualMergerError（ValueError 语义）：缺上下文、ID 集不符、重复袖/
                股票订单、非有限/负目标、S2/S3/S5/S7 未支持组合。全部零副作用。
        """
        if self.is_single_sleeve:
            self._assert_s6(orders, virtual_accounts)
            self.net_call_count += 1
            return orders
        plan = self.plan_netting(orders, virtual_accounts)
        self.net_call_count += 1
        return plan.physical_orders()

    def plan_netting(
        self,
        orders: Sequence[Order],
        virtual_accounts: Optional[Mapping[str, VirtualAccount]],
    ) -> NettingPlan:
        """构造净额计划（结构化事实：原始意图 + 物理全量目标 + S1 抵消）。"""
        accounts = self._require_multi_context(virtual_accounts)
        intents = self._build_intents(orders, accounts)
        current = self._current_shares(accounts)
        codes = {intent.ts_code for intent in intents}
        net_by_code: Dict[str, int] = {}
        targets: Dict[str, int] = {}
        offsets: List[InternalOffset] = []
        reason_by_code: Dict[str, OrderReason] = {}
        for code in sorted(codes):
            base = 0
            for sleeve_id in self._sleeve_ids:
                base += current.get((sleeve_id, code), 0)
            code_intents = [intent for intent in intents if intent.ts_code == code]
            net = sum(intent.delta_shares for intent in code_intents)
            self._assert_supported_state(code, code_intents, net)
            net_by_code[code] = net
            targets[code] = base + net
            if net == 0:
                offsets.extend(self._build_offsets(code, code_intents))
            reason_by_code[code] = self._resolve_reason(code_intents)
        physical_current = {
            code: sum(current.get((sleeve_id, code), 0) for sleeve_id in self._sleeve_ids)
            for code in sorted(codes)
        }
        physical_orders = tuple(
            self._physical_order(code, targets[code], net_by_code[code], reason_by_code[code])
            for code in sorted(codes)
            if net_by_code[code] != 0
        )
        return NettingPlan(
            intents=tuple(intents),
            physical_current=MappingProxyType(physical_current),
            physical_targets=MappingProxyType(dict(targets)),
            net_by_code=MappingProxyType(dict(net_by_code)),
            offsets=tuple(offsets),
            orders=physical_orders,
        )

    @staticmethod
    def _physical_order(code: str, target: int, net: int, reason: OrderReason) -> Order:
        """物理市场指令（``target_shares`` 是账户全量目标；方向由净增量决定）。"""
        return Order(
            ts_code=TSCode(code),
            side=OrderSide.BUY if net > 0 else OrderSide.SELL,
            target_shares=Shares(target),
            order_type=OrderType.MARKET_OPEN,
            reason=reason,
            source_sleeve=PHYSICAL_SLEEVE_ID,
        )

    def _require_multi_context(
        self, virtual_accounts: Optional[Mapping[str, VirtualAccount]]
    ) -> Mapping[str, VirtualAccount]:
        """多袖上下文校验：缺上下文/ID 集不符一律拒绝（不静默套 S6）。"""
        if virtual_accounts is None:
            raise VirtualMergerError(
                f"阶段={_STAGE} 状态机=S6 net_across_sleeves：多袖账户缺虚拟账户上下文"
                f"（构造期袖={list(self._sleeve_ids)}）；多袖实现禁止缺上下文直通"
            )
        if not isinstance(virtual_accounts, Mapping):
            raise VirtualMergerError(
                f"阶段={_STAGE} virtual_accounts 必须为 Mapping，实际 "
                f"{type(virtual_accounts).__name__}"
            )
        if set(virtual_accounts) != set(self._sleeve_ids):
            raise VirtualMergerError(
                f"阶段={_STAGE} 虚拟账户 ID 集={sorted(virtual_accounts)} 与构造期袖="
                f"{list(self._sleeve_ids)} 不一致（缺失/多余均拒绝）"
            )
        for sleeve_id in self._sleeve_ids:
            account = virtual_accounts[sleeve_id]
            if not isinstance(account, VirtualAccount):
                raise VirtualMergerError(
                    f"阶段={_STAGE} 袖 {sleeve_id} 的虚拟账户必须为 VirtualAccount，实际 "
                    f"{type(account).__name__}"
                )
            if account.sleeve_id != sleeve_id:
                raise VirtualMergerError(
                    f"阶段={_STAGE} 虚拟账户键 {sleeve_id!r} != account.sleeve_id "
                    f"{account.sleeve_id!r}"
                )
        return virtual_accounts

    @staticmethod
    def _current_shares(
        accounts: Mapping[str, VirtualAccount]
    ) -> Dict[Tuple[str, str], int]:
        """各袖当前虚拟持仓（按股票汇总整数股；lot 级来源不影响目标解析）。"""
        current: Dict[Tuple[str, str], int] = {}
        for sleeve_id in sorted(accounts):
            account = accounts[sleeve_id]
            if not isinstance(account.positions, Mapping):
                raise VirtualMergerError(
                    f"阶段={_STAGE} 袖 {sleeve_id} 的 positions 必须为 Mapping"
                )
            for code, position in account.positions.items():
                key = str(getattr(code, "value", code))
                total = 0
                for lot in position.lots:
                    shares = getattr(lot, "shares", None)
                    total += _as_int(getattr(shares, "value", shares), f"袖 {sleeve_id} {key} 虚拟股数")
                current[(sleeve_id, key)] = total
        return current

    def _build_intents(
        self, orders: Sequence[Order], accounts: Mapping[str, VirtualAccount]
    ) -> List[SleeveIntent]:
        """原始意图集合：目标股数 − 该袖当前虚拟持仓（禁止把目标当增量）。"""
        if isinstance(orders, (str, bytes)) or not isinstance(orders, Sequence):
            raise VirtualMergerError(
                f"阶段={_STAGE} net_across_sleeves 需要指令序列，实际 "
                f"{type(orders).__name__}"
            )
        current = self._current_shares(accounts)
        seen = set()
        intents: List[SleeveIntent] = []
        for order in orders:
            if not isinstance(order, Order):
                raise VirtualMergerError(
                    f"阶段={_STAGE} 指令必须为 Order，实际 {type(order).__name__}"
                )
            sleeve_id = order.source_sleeve
            if sleeve_id not in self._id_set:
                raise VirtualMergerError(
                    f"阶段={_STAGE} 指令袖 {sleeve_id!r} 不在构造期袖集合 "
                    f"{list(self._sleeve_ids)} 内"
                )
            code = order.ts_code.value
            key = (sleeve_id, code)
            if key in seen:
                raise VirtualMergerError(
                    f"阶段={_STAGE} 重复袖/股票指令：袖={sleeve_id} 股票={code}；"
                    "同一袖同一股票只允许一条目标指令"
                )
            seen.add(key)
            target = _as_int(order.target_shares.value, f"袖 {sleeve_id} {code} 目标股数")
            current_shares = current.get(key, 0)
            delta = target - current_shares
            side = order.side.value
            if (side == "buy" and delta < 0) or (side == "sell" and delta > 0):
                raise VirtualMergerError(
                    f"阶段={_STAGE} 股票={code} 袖={sleeve_id}：指令方向 {side!r} 与解析增量 "
                    f"{delta} 股符号不一致（目标={target} 当前={current_shares}）；"
                    "方向只由净增量决定，输入自相矛盾必须零副作用拒绝"
                )
            intents.append(
                SleeveIntent(
                    sleeve_id=sleeve_id,
                    ts_code=code,
                    side=side,
                    target_shares=target,
                    current_shares=current_shares,
                    reason=order.reason.value,
                )
            )
        return intents

    def _assert_supported_state(
        self, code: str, intents: Sequence[SleeveIntent], net: int
    ) -> None:
        """状态机分区：仅 S1（一对等量反向）、S4（同向买入）与单袖发单放行。

        单袖发单（含卖出）不属于 S5：S5 定义是**两个及以上袖同向卖出合并**；
        多袖账户里只有一个袖发单时按 F13 恢复物理全量目标（本批其余袖以当前
        虚拟持仓计入基数），其虚拟侧卖出结算归 T3。

        S1 严格按契约的一卖一买等量反向；"多卖一买/一卖多买且净零"属未登记的
        S1 推广，P6.5 显式拒绝（不静默按 S1 处理）。
        """
        buys = [item for item in intents if item.delta_shares > 0]
        sells = [item for item in intents if item.delta_shares < 0]
        if buys and sells:
            if net != 0:
                state = "S2" if net < 0 else "S3"
                raise VirtualMergerError(
                    f"阶段={_STAGE} 状态机={state} 股票={code}：同股反向但净残差 "
                    f"{net} 股（买={sum(i.delta_shares for i in buys)} "
                    f"卖={sum(-i.delta_shares for i in sells)}）；P6.5 只覆盖 S1 等量反向，"
                    "净残差顺延归 P7，不做部分抵消"
                )
            if len(sells) > 1 or len(buys) > 1:
                raise VirtualMergerError(
                    f"阶段={_STAGE} 状态机=S1 股票={code}：净残差为 0 但涉及 "
                    f"{len(sells)} 卖 / {len(buys)} 买；S1 契约只定义一对等量反向，"
                    "多卖/多买零残差属未登记推广，P6.5 零副作用拒绝"
                )
            return
        if len(sells) > 1:
            sell_view = {item.sleeve_id: item.delta_shares for item in sells}
            raise VirtualMergerError(
                f"阶段={_STAGE} 状态机=S5 股票={code}：同向卖出合并（{sell_view}）不在 T2 "
                "范围；S5 独立费用口径归 P7"
            )

    def _build_offsets(
        self, code: str, intents: Sequence[SleeveIntent]
    ) -> List[InternalOffset]:
        """S1 抵消：按袖 ID 字典序把卖方股数配给买方（确定性、零物理指令）。"""
        sellers = sorted(
            ((item.sleeve_id, -item.delta_shares) for item in intents if item.delta_shares < 0),
            key=lambda pair: pair[0],
        )
        buyers = sorted(
            ((item.sleeve_id, item.delta_shares) for item in intents if item.delta_shares > 0),
            key=lambda pair: pair[0],
        )
        seller_remaining = {sid: shares for sid, shares in sellers}
        buyer_remaining = {sid: shares for sid, shares in buyers}
        offsets: List[InternalOffset] = []
        for seller_id, _seller_shares in sellers:
            for buyer_id, _buyer_shares in buyers:
                available = seller_remaining[seller_id]
                wanted = buyer_remaining[buyer_id]
                if available == 0 or wanted == 0:
                    continue
                take = min(available, wanted)
                offsets.append(InternalOffset(code, seller_id, buyer_id, take))
                seller_remaining[seller_id] = available - take
                buyer_remaining[buyer_id] = wanted - take
        leftover = sum(seller_remaining.values()) + sum(buyer_remaining.values())
        if leftover:  # pragma: no cover - 等量反向时不可达
            raise VirtualMergerError(
                f"阶段={_STAGE} 状态机=S1 股票={code}：抵消配对后仍余 {leftover} 股；"
                "等量反向应完全配对"
            )
        return offsets

    @staticmethod
    def _resolve_reason(intents: Sequence[SleeveIntent]) -> OrderReason:
        """物理指令原因：全部贡献袖一致则沿用，否则标 NETTING。"""
        reasons = {item.reason for item in intents if item.delta_shares != 0}
        if len(reasons) == 1:
            value = next(iter(reasons))
            try:
                return OrderReason(value)
            except ValueError:
                return OrderReason.NETTING
        return OrderReason.NETTING

    def _assert_s6(
        self, orders: Sequence[Order], virtual_accounts: Optional[Mapping[str, VirtualAccount]]
    ) -> None:
        """S6 前置：真正单袖账户；上下文出现其他袖/非账户值即拒绝（不得静默绕过）。"""
        sleeve_id = self._sleeve_ids[0]
        if virtual_accounts is not None:
            if not isinstance(virtual_accounts, Mapping) or set(virtual_accounts) != {sleeve_id}:
                keys = sorted(virtual_accounts) if isinstance(virtual_accounts, Mapping) else None
                raise VirtualMergerError(
                    f"阶段={_STAGE} 状态机=S6：单袖直通收到多袖/异 ID 虚拟账户上下文 "
                    f"{keys}；构造期袖={sleeve_id!r}"
                )
            account = virtual_accounts[sleeve_id]
            if not isinstance(account, VirtualAccount) or account.sleeve_id != sleeve_id:
                raise VirtualMergerError(
                    f"阶段={_STAGE} 状态机=S6：虚拟账户值必须为 sleeve_id={sleeve_id!r} 的 "
                    f"VirtualAccount，实际 {type(account).__name__}"
                )
        for order in orders:
            if not isinstance(order, Order):
                raise VirtualMergerError(
                    f"阶段={_STAGE} 指令必须为 Order，实际 {type(order).__name__}"
                )
            if order.source_sleeve != sleeve_id:
                raise VirtualMergerError(
                    f"阶段={_STAGE} 状态机=S6 单袖直通拒绝异袖指令：构造期袖={sleeve_id!r}，"
                    f"指令袖={order.source_sleeve!r}"
                )
