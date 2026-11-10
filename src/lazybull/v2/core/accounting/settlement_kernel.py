"""P6.5 T2 结算内核：Q13 费用瀑布 + Q18 股数分摊 + 结算事实结构。

契约依据：``docs/contracts/netting_freeze.md`` F25 §5（Q13/Q18、终端残差、
全零成交短路）与设计 §8.3 的逐步算例。

职责边界：
- 本模块只含**纯函数内核**与不可变事实结构：不持有账本、不读配置、不做 IO；
- 唯一错误类型 ``VirtualLedgerError``（延迟取用，避免与 ``virtual_ledger`` 循环导入）；
- 结算编排在 ``settlement_batch``；``settlement_batch``/``virtual_ledger`` 对本模块
  公开名做重导出，T1/T2 测试导入路径不变。

数值口径（冻结）：
- 受益顺序 ``(q_i 升序, sleeve_id 字典序升序)``；余数顺序
  ``(原始增量降序, sleeve_id 字典序升序)``；
- 所有求和按契约固定顺序 **float64 左到右**（禁止用 3.12 起带补偿的 ``sum``）；
- 费用精确和必须等于物理费用，残差只允许一次终端修正，失败即拒绝。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from src.lazybull.v2.common.types import TradeDate

#: 本模块唯一阶段标识（进入全部错误上下文）
_STAGE = "P6.5"


def _error(message: str) -> Exception:
    """延迟取用 ``VirtualLedgerError``（避免模块级循环导入）。"""
    from src.lazybull.v2.core.accounting.virtual_ledger import VirtualLedgerError

    return VirtualLedgerError(message)


def _require_positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    """整数校验（复用账本侧唯一实现，保持错误口径一致）。"""
    from src.lazybull.v2.core.accounting.virtual_ledger import _require_positive_int as impl

    return impl(value, field, allow_zero=allow_zero)


def _require_finite_non_negative(value: Any, field: str) -> float:
    """有限非负数校验（复用账本侧唯一实现）。"""
    from src.lazybull.v2.core.accounting.virtual_ledger import (
        _require_finite_non_negative as impl,
    )

    return impl(value, field)


def _require_finite_positive(value: Any, field: str) -> float:
    """有限正数校验（复用账本侧唯一实现）。"""
    from src.lazybull.v2.core.accounting.virtual_ledger import (
        _require_finite_positive as impl,
    )

    return impl(value, field)


# ────────────────────────── 结算批次事实 ──────────────────────────


@dataclass(frozen=True, slots=True)
class SleeveIntent:
    """单袖原始结算意图（``target_shares`` 是**目标股数**，不是增量）。"""

    sleeve_id: str
    ts_code: str
    side: str  # "buy" / "sell"
    target_shares: int
    current_shares: int
    reason: str = ""

    def __post_init__(self) -> None:
        if not str(self.sleeve_id).strip():
            raise _error("结算意图 sleeve_id 不得为空")
        if not str(self.ts_code).strip():
            raise _error(f"结算意图（袖 {self.sleeve_id}）的 ts_code 不得为空")
        if self.side not in ("buy", "sell"):
            raise _error(
                f"结算意图（袖 {self.sleeve_id}）side 必须为 buy/sell，实际 {self.side!r}"
            )
        object.__setattr__(
            self,
            "target_shares",
            _require_positive_int(
                self.target_shares, f"袖 {self.sleeve_id} 目标股数", allow_zero=True
            ),
        )
        object.__setattr__(
            self,
            "current_shares",
            _require_positive_int(
                self.current_shares, f"袖 {self.sleeve_id} 当前虚拟股数", allow_zero=True
            ),
        )

    @property
    def delta_shares(self) -> int:
        """解析增量 = 目标 − 当前（禁止把目标当增量）。"""
        return self.target_shares - self.current_shares


@dataclass(frozen=True, slots=True)
class PhysicalFillFact:
    """物理成交事实（显式注入；T2 只覆盖 S4 买入，卖出归 S5 显式拒绝）。"""

    ts_code: str
    side: str
    shares: int
    price: float
    fee: float
    source_lot_id: str
    fill_id: str

    def __post_init__(self) -> None:
        if not str(self.ts_code).strip() or not str(self.source_lot_id).strip():
            raise _error("物理成交的 ts_code/source_lot_id 不得为空")
        if not str(self.fill_id).strip():
            raise _error("物理成交 fill_id 不得为空（稳定成交身份）")
        if self.side != "buy":
            raise _error(
                f"阶段={_STAGE} 状态机=S5／卖出结算未实现 股票={self.ts_code} 方向={self.side}："
                "T2 只结算 S4 合并买入；S5（两袖同向卖出）与单袖卖出的虚拟侧结算归 T3/P7"
            )
        object.__setattr__(self, "shares", _require_positive_int(self.shares, "物理成交股数"))
        object.__setattr__(
            self,
            "price",
            _require_finite_positive(self.price, f"物理成交价 {self.ts_code}"),
        )
        object.__setattr__(
            self, "fee", _require_finite_non_negative(self.fee, f"物理成交费用 {self.ts_code}")
        )


@dataclass(frozen=True, slots=True)
class TransferFact:
    """S1 内部转账事实（等量反向；零物理指令、双方零费用）。"""

    ts_code: str
    from_sleeve: str
    to_sleeve: str
    shares: int
    price: float

    def __post_init__(self) -> None:
        if not str(self.ts_code).strip():
            raise _error("内部转账事实的 ts_code 不得为空")
        if not str(self.from_sleeve).strip() or not str(self.to_sleeve).strip():
            raise _error("内部转账事实的转出/转入袖不得为空")
        if self.from_sleeve == self.to_sleeve:
            raise _error(f"内部转账事实的转出/转入袖相同（{self.from_sleeve}）")
        object.__setattr__(self, "shares", _require_positive_int(self.shares, "内部转账股数"))
        object.__setattr__(self, "price", _require_finite_positive(self.price, "内部转账价"))


@dataclass(frozen=True, slots=True)
class ShareAllocation:
    """Q18 股数分摊结果（单袖）。"""

    sleeve_id: str
    intent_shares: int
    allocated_shares: int


@dataclass(frozen=True, slots=True)
class FeeWaterfall:
    """Q13 费用瀑布证据（受益顺序、预备费用、终端残差、最终费用与精确和校验）。"""

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
    # 精确和由 `_close_terminal_fee` 的"不成立即抛"强制，因此不再单列恒真布尔字段
    # （A-R3-04/A-R4-03 处置）；可复算证据 = ``final_fees`` 的左到右和 == ``physical_fee``。

    def fee_of(self, sleeve_id: str) -> float:
        """单袖最终费用（未登记袖返回 0.0）。"""
        for key, value in self.final_fees:
            if key == sleeve_id:
                return value
        return 0.0


def _check_intents(intents: Sequence[SleeveIntent], batch_id: str) -> None:
    """意图结构校验：类型与 ``(sleeve, stock)`` 唯一。"""
    seen = set()
    for intent in intents:
        if not isinstance(intent, SleeveIntent):
            raise _error(
                f"批次 {batch_id} 的 intents 含非 SleeveIntent 元素：{type(intent).__name__}"
            )
        key = (intent.sleeve_id, intent.ts_code)
        if key in seen:
            raise _error(
                f"批次 {batch_id} 重复袖/股票意图：袖={key[0]} 股票={key[1]}；"
                "同一袖同一股票只允许一条目标指令"
            )
        seen.add(key)


def _check_fills(fills: Sequence[PhysicalFillFact], batch_id: str) -> None:
    """成交结构校验：类型、每股一笔、``fill_id`` 唯一。"""
    seen_codes = set()
    seen_ids = set()
    for fill in fills:
        if not isinstance(fill, PhysicalFillFact):
            raise _error(
                f"批次 {batch_id} 的 physical_fills 含非 PhysicalFillFact 元素："
                f"{type(fill).__name__}"
            )
        if fill.ts_code in seen_codes:
            raise _error(
                f"批次 {batch_id} 同一股票多笔物理成交未支持：股票={fill.ts_code}；"
                "T2 结算批次边界为每股一笔成交（拆分归 T3 订单级全链）"
            )
        seen_codes.add(fill.ts_code)
        if fill.fill_id in seen_ids:
            raise _error(f"批次 {batch_id} 物理成交身份重复：fill_id={fill.fill_id}")
        seen_ids.add(fill.fill_id)


def _check_transfers(transfers: Sequence[TransferFact], batch_id: str) -> None:
    """转账结构校验：类型与 ``(stock, from, to)`` 唯一。"""
    seen = set()
    for fact in transfers:
        if not isinstance(fact, TransferFact):
            raise _error(
                f"批次 {batch_id} 的 transfers 含非 TransferFact 元素：{type(fact).__name__}"
            )
        key = (fact.ts_code, fact.from_sleeve, fact.to_sleeve)
        if key in seen:
            raise _error(
                f"批次 {batch_id} 重复内部转账事实：股票={key[0]} {key[1]}→{key[2]}"
            )
        seen.add(key)


@dataclass(frozen=True, slots=True)
class InternalSettlementBatch:
    """内部结算批次（原始意图 + 物理净额事实 + 内部转账 + 稳定幂等身份）。

    ``physical_cash_before`` / ``physical_cash_after`` 是**显式注入的物理事实**：
    结算后 Σ 虚拟现金必须等于 ``physical_cash_after``（差 = 物理费用），该等式
    取代 T1「Σ 现金不漂移」守卫（费用分摊会合法改变合计）。
    """

    batch_id: str
    trade_date: TradeDate
    intents: Tuple[SleeveIntent, ...] = ()
    physical_fills: Tuple[PhysicalFillFact, ...] = ()
    transfers: Tuple[TransferFact, ...] = ()
    physical_cash_before: float = 0.0
    physical_cash_after: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.batch_id).strip():
            raise _error("内部结算批次 batch_id 不得为空（幂等身份）")
        if not isinstance(self.trade_date, TradeDate):
            raise _error(
                f"批次 {self.batch_id} 的 trade_date 必须为 TradeDate，实际 "
                f"{type(self.trade_date).__name__}"
            )
        object.__setattr__(
            self,
            "physical_cash_before",
            _require_finite_non_negative(
                self.physical_cash_before, f"批次 {self.batch_id} 物理现金（前）"
            ),
        )
        object.__setattr__(
            self,
            "physical_cash_after",
            _require_finite_non_negative(
                self.physical_cash_after, f"批次 {self.batch_id} 物理现金（后）"
            ),
        )
        _check_intents(self.intents, self.batch_id)
        _check_fills(self.physical_fills, self.batch_id)
        _check_transfers(self.transfers, self.batch_id)

    def fingerprint(self) -> Tuple[Any, ...]:
        """稳定内容指纹（幂等键冲突判定；浮点用 repr 保留位模式）。"""
        intents = tuple(
            (
                item.sleeve_id,
                item.ts_code,
                item.side,
                item.target_shares,
                item.current_shares,
                item.reason,
            )
            for item in self.intents
        )
        fills = tuple(
            (
                item.ts_code,
                item.side,
                item.shares,
                repr(item.price),
                repr(item.fee),
                item.source_lot_id,
                item.fill_id,
            )
            for item in self.physical_fills
        )
        transfers = tuple(
            (item.ts_code, item.from_sleeve, item.to_sleeve, item.shares, repr(item.price))
            for item in self.transfers
        )
        return (
            str(self.trade_date),
            intents,
            fills,
            transfers,
            repr(self.physical_cash_before),
            repr(self.physical_cash_after),
        )


@dataclass(frozen=True, slots=True)
class SettlementResult:
    """结算应用结果（成功证据；``applied=False`` 表示同身份重复提交）。"""

    batch_id: str
    applied: bool
    share_allocations: Tuple[ShareAllocation, ...]
    fee_waterfalls: Tuple[FeeWaterfall, ...]
    transfers: Tuple[Any, ...]
    consumed_lots: Tuple[Tuple[str, str, int], ...]
    cash_total_after: float
    source_shares_after: Tuple[Tuple[str, str, int], ...]
    zero_fill: bool
    #: Q22 剩余意图日终作废证据（``(sleeve_id, ts_code, 未成交增量股)``，已排序）
    voided_intents: Tuple[Tuple[str, str, int], ...] = ()
    #: 结算后 I1/I2 的成功证据（``reconciliation.IdentityResult`` 元组；A-R4-04）
    identity_results: Tuple[Any, ...] = ()
    #: 本次是否已与引擎聚合持仓对账（``physical_positions`` 是否显式注入）
    positions_reconciled: bool = False


# ────────────────────────── Q18 股数分摊 ──────────────────────────


def _distribute_remainder(
    capacity: Mapping[str, int],
    allocated: Dict[str, int],
    remainder: int,
) -> None:
    """余数逐股分配（``(增量降序, sleeve_id 升序)``；容量满即跳过）。

    完整一轮后仍有余数说明输入或计算非法：拒绝，禁止强行塞给最大袖，也禁止
    第二轮分配（会让同额袖的结果依赖实现细节）。
    """
    if remainder == 0:
        return
    order = sorted(
        (sid for sid in capacity if capacity[sid] > 0), key=lambda sid: (-capacity[sid], sid)
    )
    for sid in order:
        if remainder == 0:
            break
        if allocated[sid] < capacity[sid]:
            allocated[sid] += 1
            remainder -= 1
    if remainder != 0:
        changed = {sid: (capacity[sid], allocated[sid]) for sid in sorted(capacity)}
        raise _error(
            f"阶段={_STAGE} 状态机=S4 Q18：余数 {remainder} 股无法在容量内完成一轮分配"
            f"（增量/已分得={changed}）；容差=0 股 单位=股"
        )


def allocate_shares(
    increments: Mapping[str, int],
    filled_shares: int,
) -> Tuple[ShareAllocation, ...]:
    """Q18：把物理成交股数按原始买入增量分摊到各袖（虚拟分摊单位 = 1 股）。

    规则（F25 §5，冻结）：
    1. ``Q ≤ Σ 原始买入增量``，否则拒绝（零副作用）；
    2. ``r_i = Q × 增量_i / Σ增量``，先 ``floor``（整数整除，避免浮点二次舍入）；
    3. 余数按 ``(增量降序, sleeve_id 字典序升序)`` 逐股分配；候选必须
       ``已分得 < 增量``，容量满则确定性跳过，**不得超单袖增量**；零意图袖容量 0；
    4. 一轮后仍有余数即拒绝（不得强行塞给最大袖）。

    Returns:
        按 ``sleeve_id`` 字典序排列的分配结果（确定性证据形态）。
    """
    filled = _require_positive_int(filled_shares, "物理成交股数", allow_zero=True)
    capacity: Dict[str, int] = {}
    for sleeve_id, increment in increments.items():
        sid = str(sleeve_id)
        if not sid.strip():
            raise _error("Q18 股数分摊的袖 ID 不得为空")
        capacity[sid] = _require_positive_int(
            increment, f"袖 {sid} 原始买入增量", allow_zero=True
        )
    total_capacity = 0
    for sid in sorted(capacity):
        total_capacity += capacity[sid]
    if filled > total_capacity:
        increments_view = {sid: capacity[sid] for sid in sorted(capacity)}
        raise _error(
            f"阶段={_STAGE} 状态机=S4 Q18：物理成交股数 {filled} > Σ原始买入增量 "
            f"{total_capacity}（各袖增量={increments_view}）；超意图成交拒绝，"
            "容差=0 股 单位=股"
        )
    if filled == 0:
        return tuple(ShareAllocation(sid, capacity[sid], 0) for sid in sorted(capacity))
    # filled > 0 时 total_capacity > 0（上方超意图检查已拦下 filled > capacity 总和）
    allocated = {sid: (filled * capacity[sid]) // total_capacity for sid in capacity}
    remainder = filled
    for sid in sorted(allocated):
        remainder -= allocated[sid]
    _distribute_remainder(capacity, allocated, remainder)
    for sid in sorted(capacity):
        if not 0 <= allocated[sid] <= capacity[sid]:
            raise _error(
                f"阶段={_STAGE} 状态机=S4 Q18 实现缺陷：袖 {sid} 分得 {allocated[sid]} 股 "
                f"越过原始增量 {capacity[sid]} 股"
            )
    return tuple(ShareAllocation(sid, capacity[sid], allocated[sid]) for sid in sorted(capacity))


# ────────────────────────── Q13 费用瀑布 ──────────────────────────


def _fee_sorted_order(intent_amounts: Mapping[str, float]) -> Tuple[str, ...]:
    """受益顺序 ``(q_i 升序, sleeve_id 字典序升序)``（不使用输入顺序）。"""
    return tuple(sorted(intent_amounts, key=lambda sid: (intent_amounts[sid], sid)))


def _left_to_right(values: Sequence[float]) -> float:
    """显式 float64 从左到右求和（禁止用 3.12 起带补偿的 ``sum``）。"""
    total = 0.0
    for value in values:
        total += value
    return total


def _validate_fee_inputs(
    intent_amounts: Mapping[str, float],
    allocated_amounts: Mapping[str, float],
    physical_fee: float,
    buy_cost: Callable[[float], float],
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """费用瀑布输入校验：键集一致、金额有限非负、分得不超意图、成本函数可调用。"""
    fee = _require_finite_non_negative(physical_fee, "物理费用 F")
    if not callable(buy_cost):
        raise _error("费用瀑布需要可调用的买入成本函数 buy_cost")
    if set(intent_amounts) != set(allocated_amounts):
        raise _error(
            f"阶段={_STAGE} 状态机=S4 费用瀑布：原始意图金额键集 "
            f"{sorted(intent_amounts)} 与分得金额键集 {sorted(allocated_amounts)} 不一致"
        )
    intents: Dict[str, float] = {}
    allocated: Dict[str, float] = {}
    for sleeve_id in sorted(intent_amounts):
        sid = str(sleeve_id)
        q = _require_finite_non_negative(
            intent_amounts[sleeve_id], f"袖 {sid} 原始意图金额 q_i"
        )
        a = _require_finite_non_negative(
            allocated_amounts[sleeve_id], f"袖 {sid} 实际分得金额 a_i"
        )
        if a > q:
            raise _error(
                f"阶段={_STAGE} 状态机=S4 费用瀑布：袖 {sid} 分得金额 {a!r} > 原始意图"
                f"金额 {q!r}；单袖不得超意图（差={a - q!r} 容差=0 单位=元）"
            )
        intents[sid] = q
        allocated[sid] = a
    return fee, intents, allocated


def _independent_costs(
    order: Tuple[str, ...],
    allocated: Mapping[str, float],
    buy_cost: Callable[[float], float],
) -> Dict[str, float]:
    """独立费用基准：``a_i=0 ⇒ c_i=0``（禁止对零成交额调用最低佣金）。"""
    costs: Dict[str, float] = {}
    for sid in order:
        cost = 0.0 if allocated[sid] == 0.0 else float(buy_cost(allocated[sid]))
        if not math.isfinite(cost) or cost < 0.0:
            raise _error(
                f"阶段={_STAGE} 状态机=S4 费用瀑布：袖 {sid} 独立费用 {cost!r} 非有限非负"
            )
        costs[sid] = cost
    return costs


def _provisional_fees(
    order: Tuple[str, ...],
    costs: Mapping[str, float],
    allocated: Mapping[str, float],
    fee: float,
    total_allocated: float,
) -> Tuple[Dict[str, float], bool]:
    """预备费用：正常分支走非负瀑布；``F>Σc`` 走比例分摊分支。"""
    # Σc 也必须按契约的固定顺序 float64 左到右（与终端残差同一口径）
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


def _close_terminal_fee(
    order: Tuple[str, ...],
    provisional: Mapping[str, float],
    physical_fee: float,
    terminal: str,
) -> Tuple[Dict[str, float], float, float]:
    """终端残差闭合（F25 §5 第 5 步）：一次修正 + 精确和校验 + 非负守卫。

    Returns:
        ``(最终费用映射, 残差修正量, 最终精确和)``；精确和不成立或终端为负即拒绝。
    """
    others = _left_to_right([provisional[sid] for sid in order if sid != terminal])
    terminal_fee = physical_fee - others
    final = dict(provisional)
    final[terminal] = terminal_fee
    total = _left_to_right([final[sid] for sid in order])
    residual = 0.0
    if total != physical_fee:
        residual = physical_fee - total
        final[terminal] = terminal_fee + residual
        total = _left_to_right([final[sid] for sid in order])
    if total != physical_fee:
        raise _error(
            f"阶段={_STAGE} 状态机=S4 费用瀑布：最终费用精确和不成立——Σf={total!r} != "
            f"F={physical_fee!r}（差={total - physical_fee!r} 容差=0 单位=元）；"
            f"终端袖={terminal} 修正={residual!r}；禁止把余数只登记为游离项"
        )
    if final[terminal] < 0.0:
        raise _error(
            f"阶段={_STAGE} 状态机=S4 费用瀑布：终端袖 {terminal} 费用为负 "
            f"{final[terminal]!r}（F={physical_fee!r} 其他袖合计={others!r}）；拒绝"
        )
    return final, residual, total


def _assert_final_fees(
    order: Tuple[str, ...], allocated: Mapping[str, float], final: Mapping[str, float]
) -> None:
    """最终费用逐袖守卫：不得为负，零成交份额不得被收费。"""
    for sid in order:
        if final[sid] < 0.0:
            raise _error(
                f"阶段={_STAGE} 状态机=S4 费用瀑布：袖 {sid} 最终费用为负 {final[sid]!r}"
            )
        if allocated[sid] == 0.0 and final[sid] != 0.0:
            raise _error(
                f"阶段={_STAGE} 状态机=S4 费用瀑布：零成交份额被收费——袖 {sid} "
                f"分得金额=0 但费用={final[sid]!r} 元；容差=0 单位=元"
            )


def allocate_fees(
    *,
    intent_amounts: Mapping[str, float],
    allocated_amounts: Mapping[str, float],
    physical_fee: float,
    buy_cost: Callable[[float], float],
) -> FeeWaterfall:
    """Q13-A：N 袖费用瀑布 + 终端残差闭合（零成交份额不收费）。

    步骤（F25 §5，冻结）：
    0. ``Σa=0`` 且 ``F=0`` ⇒ 全零费用短路（不调用成本模型）；``Σa=0`` 且 ``F≠0`` 拒绝；
    1. ``a_i=0 ⇒ c_i=0``（禁止对零成交额调用最低佣金）；
    2. ``c_i = buy_cost(a_i)``（按实际分得金额，不按原始全额意图）；
    3. ``S = max(Σc_i − F, 0)``；受益顺序 ``(q_i 升序, sleeve_id 升序)``；
    4. 非负瀑布 ``s_i=min(c_i, remaining)``、``f'_i=c_i−s_i``；
    5. 终端袖 = 该顺序中最后一个 ``a_i>0`` 的袖，``f_t=F−Σ_{i≠t}f'_i``（float64
       左到右）；``Σf_i≠F`` 时允许**一次**残差修正并复算，仍不等即拒绝；
    6. 终端为负 / 零份额被收费 ⇒ 拒绝；``F>Σc_i`` 时先按 ``F×a_i/Σa`` 比例分摊
       再执行第 5~6 步。
    """
    fee, intents, allocated = _validate_fee_inputs(
        intent_amounts, allocated_amounts, physical_fee, buy_cost
    )
    if not intents:
        if fee != 0.0:
            raise _error(
                f"阶段={_STAGE} 状态机=S4 费用瀑布：无任何袖但物理费用 {fee!r} 元；拒绝"
            )
        return FeeWaterfall((), (), (), (), (), "", 0.0, (), 0.0, True)
    order = _fee_sorted_order(intents)
    total_allocated = _left_to_right([allocated[sid] for sid in order])
    if total_allocated == 0.0:
        # 全零成交短路（Q13 第 0 步）：不调用成本模型、不创建 lot、不改变现金
        if fee != 0.0:
            raise _error(
                f"阶段={_STAGE} 状态机=S4 费用瀑布：Σ分得金额=0 但物理费用 {fee!r} 元；"
                "全零成交且费用非零必须拒绝，不得进入终端袖选择"
            )
        zeros = tuple((sid, 0.0) for sid in order)
        return FeeWaterfall(
            order=order,
            intent_amounts=tuple((sid, intents[sid]) for sid in order),
            allocated_amounts=tuple((sid, allocated[sid]) for sid in order),
            independent_costs=zeros,
            provisional_fees=zeros,
            terminal_sleeve="",
            residual_correction=0.0,
            final_fees=zeros,
            physical_fee=0.0,
        )
    costs = _independent_costs(order, allocated, buy_cost)
    provisional, proportional = _provisional_fees(order, costs, allocated, fee, total_allocated)
    terminal = ""
    for sid in order:
        if allocated[sid] > 0.0:
            terminal = sid
    if not terminal:  # pragma: no cover - total_allocated>0 时不可达
        raise _error(f"阶段={_STAGE} 状态机=S4 费用瀑布：Σ分得金额>0 却无正份额袖；拒绝")
    final, residual, total_fee = _close_terminal_fee(order, provisional, fee, terminal)
    _assert_final_fees(order, allocated, final)
    return FeeWaterfall(
        order=order,
        intent_amounts=tuple((sid, intents[sid]) for sid in order),
        allocated_amounts=tuple((sid, allocated[sid]) for sid in order),
        independent_costs=tuple((sid, costs[sid]) for sid in order),
        provisional_fees=tuple((sid, provisional[sid]) for sid in order),
        terminal_sleeve=terminal,
        residual_correction=residual,
        final_fees=tuple((sid, final[sid]) for sid in order),
        physical_fee=fee,
        proportional_branch=proportional,
    )
