"""P6.5 T3 订单级全链：真实 Order → 既有执行器 → 物理入账 → 内部结算 → 虚拟投影。

契约依据：设计 ``docs/plans/v2/p65_merger_reconciliation_design_20261108.md``
§4.4-1/§4.4-3（Q11=A 物理全量目标、Q21=A 订单级真实接线）、§8.2/§8.3（黄金值）、
``docs/contracts/netting_freeze.md`` F25 §4/§5、``docs/contracts/protocols.md``
F13 §3.3/§3.4。

链路（每一步都有可区分证据，禁止用"成交后证据投影"替代）：

1. 真实 ``Order``：每袖目标股数（**不是增量**）→ ``MultiSleevePortfolioMerger.plan_netting``；
2. 物理全量目标：``Σ所有袖当前虚拟持仓 + Σ本批增量``；净增量为 0 只走内部转账；
3. **预演门**（执行前、零副作用）：预测成交 → 在账本/视图副本上跑一次真实
   ``apply_settlement``；任何拒绝都在物理执行前发生；
4. 既有执行器：``_buy_stock_direct``（含加仓路径）→ ``Ledger.apply_fills``；
   物理成交事实由执行链的受控注入点（``physical_fill_observer``）产出；
5. ``InternalSettlementBatch``：股数/价/费用/来源 lot/幂等键与物理现金前后值全部
   取自执行链，不手工注入；
6. ``VirtualLedger.apply_settlement`` → 虚拟投影；成功后替换物理 lot 伴随视图。

``fill_capacity``（宿主注入）：合成流动性上限，用于模拟部分成交。物理订单目标仍是
**合并后的物理全量目标**（订单证据），执行器只在上限内成交；禁止按子袖各自整手
拆单（100+100=200 与合并目标 400 可区分，且同股两笔成交会被批次结构拒绝）。

接口选择登记：本链消费合并器的 ``plan_netting``（净额**计划**扩展面：原始意图 +
物理全量目标 + 内部抵消），而不是 F13 冻结的 ``net_across_sleeves``（只返回物理指令
序列，缺少意图与抵消事实，无法构造结算批次）。缺 ``plan_netting`` 的 F13 合规实现
在本阶段显式拒绝，不做"指令反推计划"的猜测；该扩展面归 T2 交付并已由本链复用。

边界（如实登记，不得读成更强结论）：
- 卖出：多袖账户里**单袖物理卖出**的虚拟侧结算未实现（T2 只结算 S4 买入；
  S5 归 P7），本模块在物理执行前显式拒绝，**不产生任何市场指令**；
- 本模块不实现 FundScheduler / 借用 / 公司行为 / S2/S3/S5/S7；不进入生产默认
  路径（``hosts/backtest/runtime.py`` 与 ``PaperHost`` 不变），只由 T3 合成宿主
  显式调用；
- 结算后 I1/I2 的右值仍是**合成物理 lot 伴随视图**，不等于生产物理账逐 lot 对账；
- 预演门失败是**输入/计划拒绝**（``OrderChainError``，零副作用）；执行后结算或
  观测不一致是链路实现缺陷（``OrderChainInvariantError``），不得降级为 warning。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.lazybull.v2.common.types import (
    Order,
    OrderReason,
    OrderSide,
    OrderType,
    Shares,
    TradeDate,
    TSCode,
)
from src.lazybull.v2.core.accounting.ledger import make_fill_key
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLotView
from src.lazybull.v2.core.accounting.reconciliation import (
    IdentityContext,
    ReconciliationError,
)
from src.lazybull.v2.core.accounting.settlement_kernel import (
    InternalSettlementBatch,
    PhysicalFillFact,
    SettlementResult,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    VirtualLedger,
    VirtualLedgerConservationError,
    VirtualLedgerError,
)
from src.lazybull.v2.core.decision.sizing import size_add_shares
from src.lazybull.v2.core.decision.virtual_merger import NettingPlan
from src.lazybull.v2.core.execution.buy_execution import _size_buy_shares

#: 本模块唯一阶段标识（进入全部错误上下文）
STAGE = "P6.5"
#: 物理订单单位（整手；虚拟分摊单位 1 股，二者不得混用）
LOT_SIZE = 100
#: 物理来源 lot 身份前缀（由物理成交幂等键派生，禁止手写来源批次）
SOURCE_LOT_PREFIX = "PHY:"


class OrderChainError(ValueError):
    """全链输入/计划拒绝（**零副作用**：全部在物理执行前完成）。"""


class OrderChainInvariantError(RuntimeError):
    """全链实现缺陷（观测与预测不一致 / 成交未入账 / 结算预演通过但实际拒绝）。"""


def _error(message: str) -> Exception:
    """带阶段上下文的链输入拒绝。"""
    return OrderChainError(f"阶段={STAGE} {message}")


def _as_int(value: Any, field: str) -> int:
    """严格整数解析（拒绝 bool/float/NaN；非负；单位=股）。"""
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise _error(f"{field} 必须为整数股，实际 {type(value).__name__}={value!r}（单位=股）")
    number = int(value.__index__())
    if number < 0:
        raise _error(f"{field} 必须非负，实际 {number}（单位=股）")
    return number


def _position_shares(positions: Mapping[str, Mapping[str, Any]], code: str) -> int:
    """引擎聚合持仓股数（缺失 = 0；拒绝非整数，禁止静默截断）。"""
    info = positions.get(code)
    if info is None:
        return 0
    return _as_int(info.get("shares", 0), f"物理持仓 {code} 股数")


# ────────────────────────── 事实结构 ──────────────────────────


@dataclass(frozen=True, slots=True)
class ChainFill:
    """链上单笔成交事实（预测与实测同形；``shares=0`` = 当日该目标未成交）。

    ``fill_key`` / ``source_lot_id`` 由物理入账幂等键派生：既证明事实来自
    ``Ledger.apply_fills``，又为 I2 提供稳定的物理来源键。
    """

    ts_code: str
    order_target: int
    requested_shares: int
    shares: int
    price: float
    fee: float
    cash_before: float
    cash_after: float
    fill_key: str
    source_lot_id: str


@dataclass(frozen=True, slots=True)
class OrderChainResult:
    """订单级全链结果（证据：计划 + 批次 + 结算 + 预测/实测成交 + 物理现金）。

    ``request_fingerprint`` = 输入请求（日期/各袖目标/合成上限/原因/显式价源标识）
    的稳定指纹：同一 ``batch_id`` 只允许承载同一请求（同输入重放返回
    ``applied=False``，异输入直接拒绝，禁止一个身份承载两套事实）。
    """

    trade_date: TradeDate
    batch_id: str
    request_fingerprint: str
    plan: NettingPlan
    batch: InternalSettlementBatch
    settlement: SettlementResult
    predicted: Tuple[ChainFill, ...]
    observed: Tuple[ChainFill, ...]
    physical_cash_before: float
    physical_cash_after: float
    applied: bool


def validate_sleeve_targets(
    *, targets: Mapping[str, Mapping[str, int]], batch_id: str, date: pd.Timestamp
) -> Dict[str, Dict[str, int]]:
    """目标股数严格校验并归一（拒绝 bool/float/None/负值；单位=股）。

    指纹与订单构造共用本函数：任何非法目标在**指纹计算前**就以
    ``OrderChainError``（含批次/日期/袖/股票）拒绝，不得先 ``int()`` 再漂移。
    """
    label = f"批次={batch_id} 日期={pd.Timestamp(date).date()}"
    if not isinstance(targets, Mapping) or not targets:
        raise _error(f"{label}：各袖目标必须为非空 Mapping，实际 {type(targets).__name__}")
    normalized: Dict[str, Dict[str, int]] = {}
    for sleeve_id, sleeve_map in targets.items():
        if not isinstance(sleeve_map, Mapping):
            raise _error(
                f"{label} 袖={sleeve_id}：目标必须为 Mapping，实际 " f"{type(sleeve_map).__name__}"
            )
        normalized[str(sleeve_id)] = {
            str(code): _as_int(value, f"{label} 袖={sleeve_id} 股票={code} 目标股数")
            for code, value in sleeve_map.items()
        }
    return normalized


def chain_request_fingerprint(
    *,
    date: pd.Timestamp,
    targets: Mapping[str, Mapping[str, int]],
    capacity: Optional[Mapping[str, int]],
    reason: OrderReason,
    price_source_id: str = "",
    price_source_as_of: str = "",
    price_map: Optional[Mapping[str, float]] = None,
) -> str:
    """全链输入请求指纹（稳定序列化；浮点按 repr 保留位模式，禁止随机盐）。

    ``targets`` 必须已由 ``validate_sleeve_targets`` 归一为 int；``price_map`` /
    ``price_source_as_of`` 让"同 source_id 不同价目/不同日期"成为不同请求。
    """
    payload = {
        "date": str(pd.Timestamp(date).date()),
        "targets": {
            str(sleeve): {
                str(code): int(shares) for code, shares in sorted(targets[sleeve].items())
            }
            for sleeve in sorted(targets)
        },
        "capacity": (
            None
            if capacity is None
            else {str(code): int(shares) for code, shares in sorted(capacity.items())}
        ),
        "reason": str(getattr(reason, "value", reason)),
        "price_source_id": str(price_source_id),
        "price_source_as_of": str(price_source_as_of),
        "price_map": (
            None
            if price_map is None
            else {str(code): repr(float(price)) for code, price in sorted(price_map.items())}
        ),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ────────────────────────── 1. Order 构造 ──────────────────────────


def build_sleeve_orders(
    *,
    targets: Mapping[str, Mapping[str, int]],
    ledger: VirtualLedger,
    date: pd.Timestamp,
    batch_id: str = "",
    reason: OrderReason = OrderReason.REBALANCE,
) -> Tuple[Order, ...]:
    """由各袖目标股数构造真实 ``Order``（目标量，不是增量；方向由增量符号决定）。

    零增量意图单项保留（``side=buy``）：结算层按 F25 忽略零增量参与方向解析，
    合并器也不会为净 0 产生物理指令。
    """
    label = f"批次={batch_id} 日期={pd.Timestamp(date).date()}"
    if not isinstance(targets, Mapping) or not targets:
        raise _error(f"{label}：各袖目标必须为非空 Mapping，实际 {type(targets).__name__}")
    opened = set(ledger.sleeve_ids())
    if set(targets) != opened:
        raise _error(
            f"{label}：目标袖集合={sorted(targets)} 与虚拟账已开户袖={sorted(opened)} 不一致"
            "（缺失/多余均拒绝，差集见两集合）"
        )
    orders: List[Order] = []
    for sleeve_id in sorted(targets):
        sleeve_targets = targets[sleeve_id]
        if not isinstance(sleeve_targets, Mapping):
            raise _error(
                f"{label} 袖={sleeve_id}：目标必须为 Mapping，实际 "
                f"{type(sleeve_targets).__name__}"
            )
        state = ledger.state_of(sleeve_id)
        for code in sorted(sleeve_targets):
            if not str(code).strip():
                raise _error(f"{label} 袖={sleeve_id}：目标股票代码不得为空")
            target = _as_int(sleeve_targets[code], f"{label} 袖={sleeve_id} 股票={code} 目标股数")
            current = state.shares_of(str(code))
            side = OrderSide.BUY if target - current >= 0 else OrderSide.SELL
            orders.append(
                Order(
                    ts_code=TSCode(str(code)),
                    side=side,
                    target_shares=Shares(target),
                    order_type=OrderType.MARKET_OPEN,
                    reason=reason,
                    source_sleeve=sleeve_id,
                )
            )
    return tuple(orders)


# ────────────────────────── 2. 净额计划与物理基数 ──────────────────────────


def plan_order_chain(
    *,
    merger: Any,
    ledger: VirtualLedger,
    price_source: Any,
    orders: Sequence[Order],
    batch_id: str,
    date: pd.Timestamp,
) -> NettingPlan:
    """调用多袖合并器生成净额计划（缺多袖能力即拒绝，不得退回 S6 假通过）。"""
    if not hasattr(merger, "plan_netting"):
        raise _error(
            f"批次={batch_id} 日期={pd.Timestamp(date).date()} 合并器 "
            f"{type(merger).__name__} 不提供 plan_netting；订单级全链需要多袖净额实现"
            "（SingleSleevePortfolioMerger 只覆盖 S6 直通）"
        )
    accounts = ledger.as_virtual_accounts(price_source)
    plan = merger.plan_netting(orders, accounts)
    if not isinstance(plan, NettingPlan):  # pragma: no cover - 实现缺陷防御
        raise OrderChainInvariantError(
            f"批次={batch_id} 日期={pd.Timestamp(date).date()}：plan_netting 返回类型非法："
            f"{type(plan).__name__}（应为 NettingPlan）"
        )
    return plan


def assert_plan_integrity(
    plan: NettingPlan,
    *,
    batch_id: str,
    date: pd.Timestamp,
    source_orders: Optional[Sequence[Order]] = None,
) -> None:
    """计划内部自洽（**意图驱动**）：目标覆盖、``net = Σ意图增量``、``目标 = 基数 + net``。

    - 覆盖门：每条非零意图的股票必须出现在 ``net_by_code`` 与 ``physical_targets``
      中（防止合并器丢股后被当成"合法零成交"放过）；
    - 净增量为正的股票必须有对应物理指令；
    - **计划源交叉校验**（T3-A-R3-01）：传入 ``source_orders`` 时，计划 ``intents``
      必须与计划源一一对应（袖/股票/目标股数），因此"计划源被整源清空、intents 也
      清空"不再能被当成平凡自洽；
    - 这是"子袖先各自整手再合并"的可失败守卫：该变异会让物理目标（100+100=200）
      与原始意图之和（400）不一致；不依赖合并器实现细节也能打红。
    """
    label = f"批次={batch_id} 日期={pd.Timestamp(date).date()}"
    if source_orders is not None:
        assert_plan_source_covered(plan, source_orders, label=label)
    intent_net_by_code = _collect_intent_net(plan, label)
    _assert_net_and_orders(plan, intent_net_by_code, label)


def _collect_intent_net(plan: NettingPlan, label: str) -> Dict[str, int]:
    """按股票汇总意图净增量，并校验每条非零意图都进入了计划目标/净额（覆盖门）。"""
    intent_net_by_code: Dict[str, int] = {}
    for intent in plan.intents:
        if intent.delta_shares == 0:
            continue
        code = intent.ts_code
        intent_net_by_code[code] = intent_net_by_code.get(code, 0) + intent.delta_shares
        if code not in plan.physical_targets or code not in plan.net_by_code:
            raise OrderChainInvariantError(
                f"{label} 股票={code} 袖={intent.sleeve_id}：意图增量 "
                f"{intent.delta_shares} 股未进入计划目标/净额（计划漏股）；"
                "禁止把丢失的意图当成合法零成交"
            )
    return intent_net_by_code


def _assert_net_and_orders(
    plan: NettingPlan, intent_net_by_code: Mapping[str, int], label: str
) -> None:
    """``net = Σ意图增量``、``目标 = 基数 + net``、正净增量必须有指令且目标一致。"""
    for code in sorted(intent_net_by_code):
        intent_net = intent_net_by_code[code]
        net = int(plan.net_by_code[code])
        if net != intent_net:
            raise OrderChainInvariantError(
                f"{label} 股票={code}：计划净增量 {net} 股 != Σ意图解析增量 {intent_net} 股"
                f"（差={net - intent_net} 股 容差=0 股 单位=股）；"
                "计划自相矛盾（子袖先整手/目标当增量均可由此打红）"
            )
        base = int(plan.physical_current.get(code, 0))
        target = int(plan.physical_targets[code])
        if target != base + net:
            raise OrderChainInvariantError(
                f"{label} 股票={code}：物理目标 {target} 股 != 基数 {base} 股 + 净增量 "
                f"{net} 股（差={target - base - net} 股 容差=0 股 单位=股）"
            )
    order_targets = {
        order.ts_code.value: int(order.target_shares.value) for order in plan.physical_orders()
    }
    for code in sorted(intent_net_by_code):
        if intent_net_by_code[code] <= 0:
            continue
        if code not in order_targets:
            raise OrderChainInvariantError(
                f"{label} 股票={code}：净增量 {intent_net_by_code[code]} 股为正但缺物理指令"
                "（单位=股）；禁止静默丢单"
            )
    for code, order_target in order_targets.items():
        target = int(plan.physical_targets.get(code, -1))
        if order_target != target:
            raise OrderChainInvariantError(
                f"{label} 股票={code}：物理指令目标 {order_target} 股 != 计划目标 "
                f"{target} 股（单位=股）"
            )


def assert_plan_source_covered(
    plan: NettingPlan, source_orders: Sequence[Order], *, label: str = ""
) -> None:
    """计划源交叉校验（T3-A-R3-01 处置）：计划 ``intents`` 必须与输入订单一一对应。

    此前自洽门**只由 intents 驱动**：若合并器把计划源与 intents 一起清空（例如丢了
    整批订单），门会平凡为真，把"漏整源"读成"合法零成交"。本门以**计划源订单**为
    基准反向校验袖/股票/目标三元组，任一缺、多或目标股数不同即拒绝。

    Args:
        plan: 待校验的净额计划
        source_orders: 计划源（``build_sleeve_orders`` 的产物；每袖每股一条目标指令）
        label: 错误上下文（批次/日期）

    Raises:
        OrderChainInvariantError: 计划源与计划 intents 不是一一对应。
    """
    source: Dict[Tuple[str, str], int] = {}
    for order in source_orders:
        key = (str(order.source_sleeve), str(order.ts_code.value))
        if key in source:
            raise OrderChainInvariantError(
                f"{label} 计划源含重复袖/股票指令：袖={key[0]} 股票={key[1]}；"
                "同一袖同一股票只允许一条目标指令"
            )
        source[key] = int(order.target_shares.value)
    planned = {
        (str(intent.sleeve_id), str(intent.ts_code)): int(intent.target_shares)
        for intent in plan.intents
    }
    only_source = sorted(set(source) - set(planned))
    only_plan = sorted(set(planned) - set(source))
    if only_source or only_plan:
        raise OrderChainInvariantError(
            f"{label} 计划源与计划 intents 不是一一对应：仅计划源有={only_source} "
            f"仅 intents 有={only_plan}（计划源条数={len(source)} intents 条数={len(planned)}）；"
            "整源清空不得被当成平凡自洽"
        )
    drifted = {
        key: (source[key], planned[key]) for key in sorted(source) if source[key] != planned[key]
    }
    if drifted:
        raise OrderChainInvariantError(
            f"{label} 计划源目标股数与 intents 不一致（左=计划源 右=intents）：{drifted}；"
            "容差=0 股 单位=股"
        )


def assert_physical_base(
    plan: NettingPlan,
    positions: Mapping[str, Mapping[str, Any]],
    date: pd.Timestamp,
    batch_id: str,
) -> None:
    """虚拟基数与物理聚合持仓必须逐股一致（陈旧快照会放大增量，T2-P1 同口径）。"""
    for code in sorted(plan.physical_current):
        virtual_total = int(plan.physical_current[code])
        physical_total = _position_shares(positions, code)
        if virtual_total != physical_total:
            raise _error(
                f"批次={batch_id} 日期={pd.Timestamp(date).date()} 股票={code}："
                f"虚拟袖合计当前持仓 {virtual_total} 股 != 物理聚合持仓 {physical_total} 股"
                f"（差={virtual_total - physical_total} 股 容差=0 股 单位=股）；"
                "物理全量目标必须以同一基数解析"
            )


def assert_no_physical_sells(plan: NettingPlan, date: pd.Timestamp, batch_id: str) -> None:
    """物理卖出单在 T3 显式拒绝（虚拟侧卖出结算未实现；S5 归 P7）。

    拒绝必须发生在**任何市场指令之前**：本函数由宿主在预演与执行之前调用。
    """
    for order in plan.physical_orders():
        if order.side is OrderSide.SELL:
            code = order.ts_code.value
            target = int(order.target_shares.value)
            current = int(plan.physical_current.get(code, 0))
            raise _error(
                f"批次={batch_id} 日期={pd.Timestamp(date).date()} 状态机=S5/单袖卖出 "
                f"股票={code} 袖={order.source_sleeve}：物理目标 {target} 股 当前 "
                f"{current} 股 增量 {target - current} 股；T3 只结算 S4 合并买入与 S1 "
                "内部转账，物理卖出的虚拟侧结算未实现（归 P7），零副作用拒绝"
            )


# ────────────────────────── 3. 预演（执行前、零副作用） ──────────────────────────


def _resolve_capacity(code: str, increment: int, capacity: Optional[Mapping[str, int]]) -> int:
    """合成流动性上限（缺省不限；显式上限按整数股解析）。"""
    if capacity is None:
        return increment
    raw = capacity.get(code)
    if raw is None:
        return increment
    return min(increment, _as_int(raw, f"合成成交上限 {code}"))


def _predict_one(
    *,
    engine: Any,
    order: Order,
    price_source: Any,
    date: pd.Timestamp,
    capacity: Optional[Mapping[str, int]],
    cash: float,
    seq: int,
    batch_id: str,
) -> Tuple[ChainFill, float]:
    """单笔物理订单的预演（复用执行器同一 sizing 实现；不触碰账户）。"""
    code = order.ts_code.value
    label = f"批次={batch_id} 日期={pd.Timestamp(date).date()} 股票={code}"
    target = int(order.target_shares.value)
    before = _position_shares(engine.positions, code)
    increment = target - before
    if increment <= 0:  # pragma: no cover - plan 只为净增量为正的股票产生指令
        raise OrderChainInvariantError(
            f"{label}：物理订单目标 {target} 股 <= 当前 {before} 股（单位=股）；"
            "买入单必须为正增量"
        )
    request = _resolve_capacity(code, increment, capacity)
    try:
        price = price_source.price_of(code)
    except VirtualLedgerError as error:
        raise _error(f"{label}：同日价源缺价——{error}（无价即拒绝，不顺延）") from error
    engine_price = engine._get_trade_price(date, code)
    if engine_price is None or pd.isna(engine_price) or float(engine_price) != float(price):
        raise _error(
            f"{label}：价源价格 {price!r} 元 != 引擎成交价 {engine_price!r} 元；"
            "物理成交与虚拟结算必须同日同价源（差=0 容差=0 单位=元）"
        )
    if code in engine.positions:
        sized = size_add_shares(
            target_value=request * price,
            trade_price=price,
            capital=cash,
            cost_model=engine.cost_model,
        )
    else:
        sized = _size_buy_shares(
            target_value=request * price,
            requested_shares=request,
            trade_price=price,
            capital=cash,
            cost_model=engine.cost_model,
        )
        if sized is not None:
            threshold = engine._get_min_buy_value_threshold(date)
            if threshold > 0 and sized[1] < threshold:
                sized = None
    if sized is None:
        fill = ChainFill(code, target, request, 0, price, 0.0, cash, cash, "", "")
        return fill, cash
    shares, _amount, fee, total_cost = sized
    if shares > increment:  # pragma: no cover - sizing 只会向下取整
        raise OrderChainInvariantError(
            f"{label}：预演成交 {shares} 股 > 物理增量 {increment} 股"
            "（差>0 容差=0 单位=股，超意图成交）"
        )
    fill_key = make_fill_key(date, "buy", code, seq)
    cash_after = cash - float(total_cost)
    fill = ChainFill(
        ts_code=code,
        order_target=target,
        requested_shares=request,
        shares=int(shares),
        price=float(price),
        fee=float(fee),
        cash_before=cash,
        cash_after=cash_after,
        fill_key=fill_key,
        source_lot_id=f"{SOURCE_LOT_PREFIX}{fill_key}",
    )
    return fill, cash_after


def predict_fills(
    *,
    engine: Any,
    plan: NettingPlan,
    price_source: Any,
    date: pd.Timestamp,
    batch_id: str,
    capacity: Optional[Mapping[str, int]] = None,
) -> Tuple[ChainFill, ...]:
    """按物理指令顺序预演成交（现金逐笔推进；零成交订单也保留预测事实）。"""
    cash = float(engine.current_capital)
    seq = len(engine.trades)
    fills: List[ChainFill] = []
    for order in plan.physical_orders():
        fill, cash = _predict_one(
            engine=engine,
            order=order,
            price_source=price_source,
            date=date,
            capacity=capacity,
            cash=cash,
            seq=seq + sum(1 for item in fills if item.shares > 0),
            batch_id=batch_id,
        )
        fills.append(fill)
    return tuple(fills)


def register_fills(view: PhysicalLotView, fills: Sequence[ChainFill]) -> PhysicalLotView:
    """把成交登记到**副本**伴随视图（成功后才替换宿主视图；拒绝不污染）。

    成本取自同一条执行链观测（``shares × price + fee``），不引入第二套计价：
    T5 状态信封即使恢复来源批次成本，也不改变 I2（I2 只比整数股数）。
    """
    candidate = view.copy()
    for fill in fills:
        if fill.shares == 0:
            continue
        candidate.apply_buy_fill(
            ts_code=fill.ts_code,
            source_lot_id=fill.source_lot_id,
            shares=fill.shares,
            event_id=fill.fill_key,
            cost=float(fill.shares) * float(fill.price) + float(fill.fee),
        )
    return candidate


def predicted_positions(engine: Any, fills: Sequence[ChainFill]) -> Dict[str, Mapping[str, int]]:
    """预演后的物理聚合持仓（I2 右值前置对账用；只含股数字段）。"""
    positions: Dict[str, int] = {
        code: _position_shares(engine.positions, code) for code in engine.positions
    }
    for fill in fills:
        if fill.shares:
            positions[fill.ts_code] = positions.get(fill.ts_code, 0) + fill.shares
    return {code: {"shares": shares} for code, shares in positions.items() if shares}


def build_chain_batch(
    *,
    plan: NettingPlan,
    fills: Sequence[ChainFill],
    batch_id: str,
    date: pd.Timestamp,
    price_source: Any,
    engine: Any,
) -> InternalSettlementBatch:
    """由执行事实构造内部结算批次（价格/费用/现金全部取自链，不手工注入）。"""
    executed = [fill for fill in fills if fill.shares > 0]
    for previous, current in zip(executed, executed[1:]):
        if previous.cash_after != current.cash_before:
            raise OrderChainInvariantError(
                f"批次={batch_id} 股票={current.ts_code}：现金链断裂（上笔后 "
                f"{previous.cash_after!r} != 本笔前 {current.cash_before!r} 元）"
            )
    cash_before = executed[0].cash_before if executed else float(engine.current_capital)
    cash_after = executed[-1].cash_after if executed else float(engine.current_capital)
    transfers = plan.transfer_facts(
        {offset.ts_code: price_source.price_of(offset.ts_code) for offset in plan.offsets}
    )
    physical_fills = tuple(
        PhysicalFillFact(
            ts_code=fill.ts_code,
            side="buy",
            shares=fill.shares,
            price=fill.price,
            fee=fill.fee,
            source_lot_id=fill.source_lot_id,
            fill_id=fill.fill_key,
        )
        for fill in executed
    )
    return InternalSettlementBatch(
        batch_id=batch_id,
        trade_date=TradeDate(pd.Timestamp(date).date()),
        intents=plan.intents,
        physical_fills=physical_fills,
        transfers=transfers,
        physical_cash_before=cash_before,
        physical_cash_after=cash_after,
    )


def _clone_ledger(ledger: VirtualLedger) -> VirtualLedger:
    """账本副本（预演用）：内容 + 幂等身份 + 终态全部镜像（见 ``preflight_clone``）。"""
    return ledger.preflight_clone()


def assert_settleable(
    *,
    ledger: VirtualLedger,
    batch: InternalSettlementBatch,
    price_source: Any,
    view: PhysicalLotView,
    context: IdentityContext,
    positions: Optional[Mapping[str, Mapping[str, Any]]],
    batch_id: str,
) -> SettlementResult:
    """在副本账本/副本视图上预演一次真实结算；拒绝即零副作用终止。

    预演使用**同一实现**（``VirtualLedger.apply_settlement``），不复制费用瀑布/
    FIFO/覆盖/现金校验规则；成功后真实结算在相同输入上不可能因输入原因拒绝。
    """
    clone = _clone_ledger(ledger)
    try:
        return clone.apply_settlement(
            batch,
            price_source=price_source,
            physical_view=view,
            identity_context=context,
            physical_positions=positions,
        )
    except (VirtualLedgerError, VirtualLedgerConservationError, ReconciliationError) as error:
        raise _error(
            f"批次={batch_id} 日期={batch.trade_date}：预演结算拒绝（物理执行前，零副作用）——"
            f"{error}"
        ) from error


# ────────────────────────── 4. 真实执行（受控注入点） ──────────────────────────


class _FillSink:
    """执行链接口：记录执行器落账后的事实（只读；不含任何数值计算）。"""

    def __init__(self) -> None:
        self.events: List[Tuple[Any, float, float]] = []

    def __call__(self, entry: Any, cash_before: float, cash_after: float) -> None:
        self.events.append((entry, cash_before, cash_after))


def _restore_observer(engine: Any, previous: Any) -> None:
    """恢复（或删除）受控注入点，保证生产默认路径不残留观察者。"""
    if previous is _ABSENT:
        if hasattr(engine, "physical_fill_observer"):
            del engine.physical_fill_observer
        return
    engine.physical_fill_observer = previous


_ABSENT = object()


def execute_fills(
    *,
    engine: Any,
    plan: NettingPlan,
    predictions: Sequence[ChainFill],
    date: pd.Timestamp,
    batch_id: str,
) -> Tuple[ChainFill, ...]:
    """按预演顺序调用既有执行器；物理成交事实由观察点产出并与预演逐值比对。"""
    sink = _FillSink()
    previous = getattr(engine, "physical_fill_observer", _ABSENT)
    engine.physical_fill_observer = sink
    try:
        for fill in predictions:
            if fill.shares == 0:
                continue
            engine._buy_stock_direct(
                date,
                fill.ts_code,
                float(fill.requested_shares) * fill.price,
                signal_date=date,
                requested_shares=fill.requested_shares,
            )
    finally:
        _restore_observer(engine, previous)
    return _observations_from_sink(sink, predictions=predictions, date=date, batch_id=batch_id)


def _observations_from_sink(
    sink: _FillSink, *, predictions: Sequence[ChainFill], date: pd.Timestamp, batch_id: str
) -> Tuple[ChainFill, ...]:
    """把执行链观测转换为成交事实，并逐值核对预演（不一致即实现缺陷）。"""
    label = f"批次={batch_id} 日期={pd.Timestamp(date).date()}"
    executed = [fill for fill in predictions if fill.shares > 0]
    if len(sink.events) != len(executed):
        raise OrderChainInvariantError(
            f"{label}：执行链注入点记录 {len(sink.events)} 笔成交，预演 {len(executed)} 笔；"
            "全链证据必须由执行路径产出，禁止用成交后投影补齐"
        )
    observations: List[ChainFill] = []
    for (entry, cash_before, cash_after), predicted in zip(sink.events, executed):
        observation = _observation_from_entry(
            entry, cash_before, cash_after, predicted, date=date, batch_id=batch_id
        )
        _assert_observation_matches(observation, predicted, date=date, batch_id=batch_id)
        observations.append(observation)
    return tuple(observations)


def _observation_from_entry(
    entry: Any,
    cash_before: float,
    cash_after: float,
    predicted: ChainFill,
    *,
    date: pd.Timestamp,
    batch_id: str,
) -> ChainFill:
    """``BookingEntry`` + 现金前后值 → 成交事实（字段全部取自入账指令）。

    自洽校验（B-R1 同款）：成交记录的股数/价/费用必须与入账现金增量及观察到的
    现金前后差值一致——防止"成交日志写一套、现金按另一套落账"的伪事实通过。
    """
    label = f"批次={batch_id} 日期={pd.Timestamp(date).date()} 股票={entry.stock}"
    record = entry.trade_record
    if entry.action != "buy" or entry.stock != predicted.ts_code:
        raise OrderChainInvariantError(
            f"{label}：执行链观测与预演目标不符（入账 action={entry.action!r} "
            f"stock={entry.stock!r}，预演 {predicted.ts_code!r}）"
        )
    if str(record.get("action")) != "buy" or str(record.get("stock")) != entry.stock:
        raise OrderChainInvariantError(
            f"{label}：成交日志 action/stock={record.get('action')!r}/"
            f"{record.get('stock')!r} 与入账指令不符"
        )
    record_date = pd.Timestamp(record.get("date"))
    if record_date != pd.Timestamp(date):
        raise OrderChainInvariantError(
            f"{label}：成交日志日期 {record_date} != 全链日期 {pd.Timestamp(date)}"
        )
    try:
        shares = _as_int(record.get("shares"), f"{label} 成交记录股数")
        price = float(record.get("price"))
        fee = float(record.get("cost"))
    except (OrderChainError, TypeError, ValueError) as error:
        raise OrderChainInvariantError(
            f"{label}：成交记录股数/价/费用非法（执行后必为缺陷，不得复用零副作用拒绝类）——"
            f"{error}"
        ) from error
    expected_delta = -(shares * price + fee)
    if float(entry.cash_delta) != expected_delta:
        raise OrderChainInvariantError(
            f"{label}：入账现金增量 {float(entry.cash_delta)!r} 元 与成交记录 "
            f"(股数={shares} 价={price!r} 元 费用={fee!r} 元) 推算 {expected_delta!r} 元 "
            "不一致；成交事实必须同源自洽"
        )
    # 余额关系按入账的同一运算方向核对：cash_after = cash_before + cash_delta。
    # 禁止用 cash_after - cash_before 反推（float64 减法不可逆：正常两位小数价格
    # 会得到与 cash_delta 相差 1 ULP 的"观测差"，把合法成交误判为链路破缺）。
    if float(cash_after) != float(cash_before) + float(entry.cash_delta):
        raise OrderChainInvariantError(
            f"{label}：现金余额 {float(cash_after)!r} 元 != 入账前 {float(cash_before)!r} 元 + "
            f"入账增量 {float(entry.cash_delta)!r} 元（同向核对，非反向相减）；"
            "成交事实必须同源自洽"
        )
    return ChainFill(
        ts_code=str(entry.stock),
        order_target=predicted.order_target,
        requested_shares=predicted.requested_shares,
        shares=shares,
        price=price,
        fee=fee,
        cash_before=float(cash_before),
        cash_after=float(cash_after),
        fill_key=str(entry.fill_key),
        source_lot_id=f"{SOURCE_LOT_PREFIX}{entry.fill_key}",
    )


def _assert_observation_matches(
    observation: ChainFill, predicted: ChainFill, *, date: pd.Timestamp, batch_id: str
) -> None:
    """观测必须与预演逐值一致（股数/价/费用/现金/幂等键）。"""
    mismatches = [
        field
        for field, left, right in (
            ("shares", observation.shares, predicted.shares),
            ("price", observation.price, predicted.price),
            ("fee", observation.fee, predicted.fee),
            ("cash_before", observation.cash_before, predicted.cash_before),
            ("cash_after", observation.cash_after, predicted.cash_after),
            ("fill_key", observation.fill_key, predicted.fill_key),
        )
        if left != right
    ]
    if mismatches:
        detail = {
            field: (getattr(observation, field), getattr(predicted, field)) for field in mismatches
        }
        raise OrderChainInvariantError(
            f"批次={batch_id} 日期={pd.Timestamp(date).date()} 股票={observation.ts_code}："
            f"执行链观测与预演不一致（观测 vs 预演）：{detail}；"
            "链路实现缺陷必须停止，不得用投影掩盖"
        )


def assert_fills_booked(
    engine: Any, observations: Sequence[ChainFill], *, batch_id: str, date: pd.Timestamp
) -> None:
    """成交事实必须已在物理账本幂等键与成交日志中（证明来自 ``apply_fills``）。"""
    label = f"批次={batch_id} 日期={pd.Timestamp(date).date()}"
    applied = set(engine.ledger.export_state()["applied_keys"])
    for observation in observations:
        if observation.fill_key not in applied:
            raise OrderChainInvariantError(
                f"{label} 股票={observation.ts_code} 成交键 {observation.fill_key} 不在"
                "物理账本 applied_keys 中：拒绝用成交后证据投影冒充真实执行"
            )
    tail: List[Mapping[str, Any]] = []
    if observations:
        start = len(engine.trades) - len(observations)
        tail = list(engine.trades)[start:]
    if len(tail) != len(observations):
        raise OrderChainInvariantError(
            f"{label}：成交日志不足——期望 {len(observations)} 条，实际 {len(tail)} 条"
        )
    for record, observation in zip(tail, observations):
        actual = (
            str(record.get("stock")),
            _as_int(record.get("shares"), f"{label} 成交日志股数"),
            float(record.get("price")),
            float(record.get("cost")),
        )
        expected = (
            observation.ts_code,
            observation.shares,
            observation.price,
            observation.fee,
        )
        if actual != expected:
            raise OrderChainInvariantError(
                f"{label}：成交日志与观测不一致——日志={actual} 观测={expected}"
                "（股票/股数/价/费用）"
            )
