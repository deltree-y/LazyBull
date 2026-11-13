# -*- coding: utf-8 -*-
"""P7 T4 执行与双口径账本（净额计划落账 + I1/I2/I3 + 费用归属）。

契约依据
========
- ``docs/contracts/netting_freeze.md`` §4（I1/I2/I3 容差与口径）、§5（lot 与费用归属）、
  P7 注记（有符号 I3、`Σ credit_i = 0`、借用态守卫的 P7 差异）；
- F26 §P7：S5 实际费用走 Q3=A（``fees.allocate_s5_fees``）；S4 的**股数分摊**复用受审
  纯函数 ``settlement_kernel.allocate_shares``（Q18，不重写），**费用瀑布**按 F25 §5 同一
  公式在 P7 侧实现（``fees.allocate_s4_fees``）——受审 ``allocate_fees`` 在真实数据上会因
  1 ulp 尾差拒绝整笔成交（构造例 ``audited_waterfall_rejects_1ulp`` 已落盘样例与异常文本），
  故不复用该函数；浮点闭合策略逐笔计数见 ``fees.json.float_closure``。

执行纪律
========
1. **成交价只认真实 T+1 开盘价**（``PriceBook.execution_price``）；不可用即不成交并登记，
   禁止回退当日收盘；
2. **内部转账只认当日真实开盘价**（``PriceBook.market_open``）；估值链兜底价一概不得
   完成 B1 转账（缓存估值不完成转账）；
3. **先卖后买**：同日先执行净市场卖出与内部转账，再执行净市场买入，最后**一次结算**；
4. **零副作用拒绝**：所有校验先于写入；任何拒绝抛异常，调用方保持原状态；
5. 物理账本**独立**维护（不派生自虚拟侧），I2 逐 ``(ts_code, 物理来源 lot_id)`` 比对。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts.v2_p7.t4 import fees as fee_mod
from scripts.v2_p7.t4.netting import (
    VOID_REASON_NO_POSITION,
    Intent,
    NettingPlan,
    PlanGroup,
    _held_by_code,
    _rebuild_market_legs,
    _void_reason,
    assert_group_consistency,
)
from scripts.v2_p7.t4.price import PriceBook
from scripts.v2_p7.t4.state import (
    ORIGIN_CARRY,
    P7Lot,
    PhysicalBook,
    PhysicalLot,
    ResidualEntry,
    SleeveState,
    fifo_lots,
)

#: 现金约束的整手回退轮数上限（超过即拒绝，不静默放宽）
CASH_ROUNDS = 64
#: 汇总层货币容差（Σ 分得金额与整单金额的浮点尾差；逐事件闭合仍为 0 容差）
AMOUNT_TOLERANCE = 1e-6

#: 日终作废原因（不再顺延）


class LedgerError(RuntimeError):
    """落账前置条件不满足或恒等式破缺（拒绝语义，不降级为告警）。"""


@dataclass
class GroupResult:
    """单股单日执行结果（证据形态）。"""

    ts_code: str
    scenario: str
    intent_shares: Dict[str, int] = field(default_factory=dict)
    offset_shares: Dict[str, int] = field(default_factory=dict)
    filled_shares: Dict[str, int] = field(default_factory=dict)
    voided_shares: Dict[str, int] = field(default_factory=dict)
    carry_shares: Dict[str, int] = field(default_factory=dict)
    blocked_reason: str = ""


@dataclass
class ExecutionResult:
    """单日执行汇总（新状态 + 证据；调用方负责 generation 提交）。"""

    trade_date: str
    sleeves: Dict[str, SleeveState]
    physical: PhysicalBook
    scenario_counts: Dict[str, int]
    offset_legs: int
    offset_shares: int
    market_filled: Dict[str, int]
    blocks: Dict[str, int]
    voids: Dict[str, int]
    carries: Dict[str, int]
    fee_entries: List[Tuple[fee_mod.FeeEvent, Tuple[Tuple[str, float], ...]]]
    trades: List[Dict[str, Any]]
    conservation_rows: List[Dict[str, Any]]
    next_residuals: Dict[str, List[ResidualEntry]]
    #: 与 ``fee_entries`` 等长的浮点闭合轨迹（内部抵消为 ``None``）
    closures: List[Any] = field(default_factory=list)
    #: 与 ``fee_entries`` 等长的申报物理费用（成本模型输出）
    declared_fees: List[float] = field(default_factory=list)
    #: 因买方袖现金不足而未执行的内部抵消股数（C4-01 / 口径 a）
    offset_capped_shares: int = 0


def _lot_round(shares: int, lot_size: int) -> int:
    """向下取整到整手（``lot_size`` ≤ 1 时原样返回）。"""
    size = int(lot_size)
    if size <= 1:
        return int(shares)
    return (int(shares) // size) * size


def _as_amount(shares: int, price: float) -> float:
    return float(shares) * float(price)


def _zero_cost(_amount: float) -> float:
    """内部转账无物理费用（费用按独立口径留存），故可负担性判断不计费。"""
    return 0.0


def _affordable_shares(
    cash: float,
    price: float,
    lot_size: int,
    buy_cost: Any,
    wanted: int,
) -> int:
    """现金可负担的最大整手股数（含买入费用预留）；不超过 ``wanted``。"""
    if cash <= 0.0 or price <= 0.0 or wanted <= 0:
        return 0
    shares = min(_lot_round(int(cash // price), lot_size), _lot_round(int(wanted), lot_size))
    step = max(1, int(lot_size))
    while shares > 0:
        amount = shares * price
        if amount + float(buy_cost(amount)) <= cash:
            return shares
        shares -= step
    return 0


def execute_plan(
    *,
    plan: NettingPlan,
    sleeves: Mapping[str, SleeveState],
    physical: PhysicalBook,
    prices: PriceBook,
    cost_model: Any,
    lot_size: int,
    seq_base: int = 0,
) -> ExecutionResult:
    """执行单日预净额计划（先卖后买、内部转账、一次结算；不修改入参）。"""
    day = str(plan.trade_date)
    book = _WorkingBook(sleeves=sleeves, physical=physical)
    results = _prepare_results(plan)

    offsets = _run_offsets(plan, results, prices, book, day, int(seq_base), int(lot_size))
    # 只替换市场腿，**不能**重建 results——抵消记账在里面（C4-R1-01）
    plan = _rebuild_market_legs(plan, offsets.get("capped_sell") or {})
    sells = _run_sells(plan, results, prices, cost_model, book, day)
    buys = _run_buys(plan, results, prices, cost_model, book, int(lot_size), day)

    classified = _classify_residuals(
        plan,
        results,
        day,
        holdings={sid: _held_by_code(book.lots[sid]) for sid in book.lots},
    )
    book.residual_next = classified["next_residuals"]
    blocks: Dict[str, int] = {}
    for group in plan.groups:
        reason = results[group.ts_code].blocked_reason
        if reason:
            blocks[reason] = blocks.get(reason, 0) + 1
    for reason, value in offset_block_counts(offsets).items():
        blocks[reason] = blocks.get(reason, 0) + value
    voids = dict(classified["voids"])
    for voided_row in plan.voided_rows:
        voids[VOID_REASON_NO_POSITION] = voids.get(VOID_REASON_NO_POSITION, 0) + int(
            voided_row["voided"]
        )
    conservation = list(classified["conservation"]) + [dict(row) for row in plan.voided_rows]
    return ExecutionResult(
        trade_date=day,
        sleeves=book.freeze_sleeves(),
        physical=book.freeze_physical(),
        scenario_counts=_scenario_counts(plan),
        offset_legs=int(offsets["legs"]),
        offset_shares=int(offsets["shares"]),
        offset_capped_shares=int(offsets.get("capped_shares", 0)),
        market_filled={
            "buy": int(buys["filled"]),
            "sell": int(sells["filled"]),
        },
        blocks=dict(sorted(blocks.items())),
        voids=dict(sorted(voids.items())),
        carries=dict(sorted(classified["carries"].items())),
        fee_entries=list(offsets["fee_entries"])
        + list(sells["fee_entries"])
        + list(buys["fee_entries"]),
        closures=list(offsets["closures"]) + list(sells["closures"]) + list(buys["closures"]),
        declared_fees=list(offsets["declared_fees"])
        + list(sells["declared_fees"])
        + list(buys["declared_fees"]),
        trades=list(offsets["trades"]) + list(sells["trades"]) + list(buys["trades"]),
        conservation_rows=conservation,
        next_residuals=classified["next_residuals"],
    )


class _WorkingBook:
    """单日可变工作账（现金 / 批次 / 物理侧 / 累计金额）；结算后冻结为不可变快照。"""

    def __init__(self, *, sleeves: Mapping[str, SleeveState], physical: PhysicalBook) -> None:
        self.inputs = sleeves
        self.cash = {sid: float(state.cash) for sid, state in sleeves.items()}
        self.credit = {sid: float(state.credit) for sid, state in sleeves.items()}
        self.lots: Dict[str, List[P7Lot]] = {
            sid: list(state.lots) for sid, state in sleeves.items()
        }
        self.phys_cash = float(physical.cash)
        self.phys_lots: Dict[str, PhysicalLot] = dict(physical.lots)
        self.buy_amount = 0.0
        self.sell_amount = 0.0
        self.fee_total = 0.0
        self.base = physical
        #: 次日待处理残差（由 ``_classify_residuals`` 填充；冻结时写入袖快照）
        self.residual_next: Dict[str, List[ResidualEntry]] = {sid: [] for sid in sleeves}

    def freeze_sleeves(self) -> Dict[str, SleeveState]:
        return {
            sid: SleeveState(
                sleeve_id=sid,
                cash=self.cash[sid],
                credit=self.credit[sid],
                lots=tuple(sorted(self.lots[sid], key=lambda item: item.virtual_lot_id)),
                residuals=tuple(
                    sorted(self.residual_next.get(sid, ()), key=lambda item: item.intent_key)
                ),
                borrow_lots=self.inputs[sid].borrow_lots,
                recalled_amount=self.inputs[sid].recalled_amount,
                deadlocked_amount=self.inputs[sid].deadlocked_amount,
            )
            for sid in sorted(self.inputs)
        }

    def freeze_physical(self) -> PhysicalBook:
        return PhysicalBook(
            cash=self.phys_cash,
            lots=dict(self.phys_lots),
            total_fee=self.base.total_fee + self.fee_total,
            total_buy_amount=self.base.total_buy_amount + self.buy_amount,
            total_sell_amount=self.base.total_sell_amount + self.sell_amount,
        )


def _prepare_results(plan: NettingPlan) -> Dict[str, GroupResult]:
    """建立逐组结果对象并校验计划自洽（不改任何账本）。"""
    results: Dict[str, GroupResult] = {}
    for group in plan.groups:
        assert_group_consistency(group)
        result = GroupResult(ts_code=group.ts_code, scenario=group.scenario)
        for intent in group.intents:
            result.intent_shares[intent.intent_key] = intent.shares
        results[group.ts_code] = result
    return results


def _scenario_counts(plan: NettingPlan) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for group in plan.groups:
        counts[group.scenario] = counts.get(group.scenario, 0) + 1
    return dict(sorted(counts.items()))


def offset_block_counts(offsets: Mapping[str, Any]) -> Dict[str, int]:
    """内部抵消被阻断的原因计数（无价顺延）。"""
    return dict(offsets.get("blocks") or {})


def _run_offsets(
    plan: NettingPlan,
    results: Mapping[str, GroupResult],
    prices: PriceBook,
    book: _WorkingBook,
    day: str,
    seq_base: int,
    lot_size: int = 1,
) -> Dict[str, Any]:
    """① 内部抵消（S1~S3）：只认当日真实开盘价，无价即顺延（不完成转账）。

    **转账限额（C4-01 / 口径 a）**：内部转账必须由**买方袖自有现金**支撑——否则等于用
    买方袖的负现金给卖方袖充值出「可花费的虚拟购买力」，卖方拿它去市场买入就会把物理账户
    扣成负数（账户透支）。撑不住的部分**不打内部抵消**，残差留在意图上，按既有资金约束
    语义顺延（``transfer_cash_capped``）。
    """
    fee_entries: List[Tuple[fee_mod.FeeEvent, Tuple[Tuple[str, float], ...]]] = []
    trades: List[Dict[str, Any]] = []
    blocks: Dict[str, int] = {}
    seq = int(seq_base)
    legs = 0
    shares_total = 0
    capped_shares = 0
    capped_sell: Dict[str, Dict[str, int]] = {}
    for group in plan.groups:
        result = results[group.ts_code]
        transfer = prices.market_open(group.ts_code)
        for leg in group.offsets:
            if not transfer.ok or transfer.price is None:
                prices.record_block(transfer.reason)
                blocks[transfer.reason] = blocks.get(transfer.reason, 0) + 1
                result.blocked_reason = result.blocked_reason or transfer.reason
                continue
            price = float(transfer.price)
            affordable = _affordable_shares(
                float(book.cash.get(leg.to_sleeve, 0.0)),
                price,
                lot_size,
                _zero_cost,
                int(leg.shares),
            )
            if affordable < int(leg.shares):
                blocks["transfer_cash_capped"] = blocks.get("transfer_cash_capped", 0) + 1
                result.blocked_reason = result.blocked_reason or "transfer_cash_capped"
                shortfall = int(leg.shares) - affordable
                capped_shares += shortfall
                # 卖方仍有底仓：这部分必须改走市场卖出（C4-R1-01），不能只顺延
                per_code = capped_sell.setdefault(group.ts_code, {})
                per_code[leg.from_sleeve] = per_code.get(leg.from_sleeve, 0) + shortfall
            if affordable <= 0:
                continue
            if affordable < int(leg.shares):
                leg = replace(leg, shares=int(affordable))
            _apply_transfer(leg=leg, price=price, day=day, lots=book.lots, cash=book.cash, seq=seq)
            seq += 1
            legs += 1
            shares_total += int(leg.shares)
            result.offset_shares[leg.from_intent_key] = result.offset_shares.get(
                leg.from_intent_key, 0
            ) + int(leg.shares)
            result.offset_shares[leg.to_intent_key] = result.offset_shares.get(
                leg.to_intent_key, 0
            ) + int(leg.shares)
            fee_entries.extend(_offset_fee_entries(day, group.ts_code, leg, price))
            trades.append(_offset_trade_row(day, group, leg, price))
    return {
        "legs": legs,
        "shares": shares_total,
        "capped_shares": capped_shares,
        "capped_sell": {code: dict(owners) for code, owners in capped_sell.items()},
        "fee_entries": fee_entries,
        "closures": [None] * len(fee_entries),
        "declared_fees": [float(event.physical_fee) for event, _final in fee_entries],
        "trades": trades,
        "blocks": blocks,
    }


def _offset_trade_row(day: str, group: PlanGroup, leg: Any, price: float) -> Dict[str, Any]:
    return {
        "trade_date": day,
        "ts_code": group.ts_code,
        "kind": "internal_offset",
        "scenario": group.scenario,
        "from_sleeve": leg.from_sleeve,
        "to_sleeve": leg.to_sleeve,
        "shares": int(leg.shares),
        "price": price,
        "fee": 0.0,
    }


def _run_sells(
    plan: NettingPlan,
    results: Mapping[str, GroupResult],
    prices: PriceBook,
    cost_model: Any,
    book: _WorkingBook,
    day: str,
) -> Dict[str, Any]:
    """② 净市场卖出（S2/S3/S5/S6）。"""
    fee_entries: List[Tuple[fee_mod.FeeEvent, Tuple[Tuple[str, float], ...]]] = []
    closures: List[Any] = []
    declared: List[float] = []
    trades: List[Dict[str, Any]] = []
    filled = 0
    for group in plan.groups:
        market = group.market
        if market is None or market.side != "sell":
            continue
        result = results[group.ts_code]
        outcome = _execute_sell(
            group=group,
            result=result,
            prices=prices,
            cost_model=cost_model,
            book=book,
            day=day,
        )
        fee_entries.extend(outcome["fee_entries"])
        closures.extend(outcome["closures"])
        declared.extend(outcome["declared_fees"])
        trades.extend(outcome["trades"])
        filled += int(outcome["filled"])
    return {
        "filled": filled,
        "fee_entries": fee_entries,
        "closures": closures,
        "declared_fees": declared,
        "trades": trades,
    }


def _run_buys(
    plan: NettingPlan,
    results: Mapping[str, GroupResult],
    prices: PriceBook,
    cost_model: Any,
    book: _WorkingBook,
    lot_size: int,
    day: str,
) -> Dict[str, Any]:
    """③ 净市场买入（S2/S3/S4/S6）。"""
    fee_entries: List[Tuple[fee_mod.FeeEvent, Tuple[Tuple[str, float], ...]]] = []
    closures: List[Any] = []
    declared: List[float] = []
    trades: List[Dict[str, Any]] = []
    filled = 0
    seq = 0
    for group in plan.groups:
        market = group.market
        if market is None or market.side != "buy":
            continue
        result = results[group.ts_code]
        outcome = _execute_buy(
            group=group,
            result=result,
            prices=prices,
            cost_model=cost_model,
            book=book,
            lot_size=lot_size,
            seq=seq,
            day=day,
        )
        seq += int(outcome["seq_used"])
        fee_entries.extend(outcome["fee_entries"])
        closures.extend(outcome["closures"])
        declared.extend(outcome["declared_fees"])
        trades.extend(outcome["trades"])
        filled += int(outcome["filled"])
    return {
        "filled": filled,
        "fee_entries": fee_entries,
        "closures": closures,
        "declared_fees": declared,
        "trades": trades,
    }


def _classify_residuals(
    plan: NettingPlan,
    results: Mapping[str, GroupResult],
    day: str,
    holdings: Optional[Mapping[str, Mapping[str, int]]] = None,
) -> Dict[str, Any]:
    """④ 残差 / 日终作废归类（``original = offset + filled + voided + carry``）。"""
    voids: Dict[str, int] = {}
    carries: Dict[str, int] = {}
    conservation: List[Dict[str, Any]] = []
    next_residuals: Dict[str, List[ResidualEntry]] = {}
    for group in plan.groups:
        result = results[group.ts_code]
        for intent in group.intents:
            remaining = _classify_one(
                group=group,
                intent=intent,
                result=result,
                day=day,
                voids=voids,
                carries=carries,
                next_residuals=next_residuals,
                held=int(((holdings or {}).get(intent.sleeve_id) or {}).get(intent.ts_code, 0)),
            )
            conservation.append(remaining)
    return {
        "voids": voids,
        "carries": carries,
        "conservation": conservation,
        "next_residuals": next_residuals,
    }


def _classify_one(
    *,
    group: PlanGroup,
    intent: Intent,
    result: GroupResult,
    day: str,
    voids: Dict[str, int],
    carries: Dict[str, int],
    next_residuals: Dict[str, List[ResidualEntry]],
    held: int = 0,
) -> Dict[str, Any]:
    """单意图残差归类（顺延或日终作废），返回守恒登记行。

    ``held`` 是该袖在该股的当前持仓：卖出义务只要还有底仓，就不能记成「无持仓作废」
    （C4-R1-01），必须留空转顺延。
    """
    offset = int(result.offset_shares.get(intent.intent_key, 0))
    filled = int(result.filled_shares.get(intent.intent_key, 0))
    remaining = int(intent.shares) - offset - filled
    if remaining < 0:
        raise LedgerError(
            f"意图 {intent.intent_key} 出现负残差：原 {intent.shares} - 抵消 {offset} "
            f"- 成交 {filled} = {remaining}"
        )
    if remaining > 0:
        void_reason = _void_reason(group.scenario, result, intent, remaining, held)
        if void_reason:
            result.voided_shares[intent.intent_key] = remaining
            voids[void_reason] = voids.get(void_reason, 0) + remaining
        else:
            reason = result.blocked_reason or "carried"
            result.carry_shares[intent.intent_key] = remaining
            carries[reason] = carries.get(reason, 0) + remaining
            next_residuals.setdefault(intent.sleeve_id, []).append(
                ResidualEntry(
                    intent_key=intent.intent_key,
                    sleeve_id=intent.sleeve_id,
                    ts_code=intent.ts_code,
                    side=intent.side,
                    shares=remaining,
                    origin=ORIGIN_CARRY,
                    reason=reason,
                    first_blocked_date=_first_blocked(intent, day),
                    days_carried=_carried_days(intent) + 1,
                    signal_date=intent.signal_date,
                    due_date=intent.due_date,
                    model_version=int(intent.model_version),
                )
            )
    return {
        "trade_date": day,
        "intent_key": intent.intent_key,
        "sleeve": intent.sleeve_id,
        "ts_code": intent.ts_code,
        "side": intent.side,
        "scenario": group.scenario,
        "original": int(intent.shares),
        "offset": offset,
        "filled": filled,
        "voided": int(result.voided_shares.get(intent.intent_key, 0)),
        "carry": int(result.carry_shares.get(intent.intent_key, 0)),
        "carried_in": bool(intent.carried),
    }


# ────────────────────────── 内部抵消 ──────────────────────────


def _offset_fee_entries(
    day: str, ts_code: str, leg: Any, price: float
) -> List[Tuple[fee_mod.FeeEvent, Tuple[Tuple[str, float], ...]]]:
    """内部抵消的双方费用事件（物理费用 0；双方各自留存本应支付的费用）。"""
    sell_event = fee_mod.FeeEvent(
        event_id=f"offset:{day}:{ts_code}:{leg.from_sleeve}->{leg.to_sleeve}:sell",
        trade_date=day,
        ts_code=ts_code,
        side="sell",
        scenario="offset",
        physical_fee=0.0,
        price=price,
        allocations=((leg.from_sleeve, int(leg.shares)),),
        internal_offset=True,
    )
    buy_event = fee_mod.FeeEvent(
        event_id=f"offset:{day}:{ts_code}:{leg.from_sleeve}->{leg.to_sleeve}:buy",
        trade_date=day,
        ts_code=ts_code,
        side="buy",
        scenario="offset",
        physical_fee=0.0,
        price=price,
        allocations=((leg.to_sleeve, int(leg.shares)),),
        internal_offset=True,
    )
    return [
        (sell_event, ((leg.from_sleeve, 0.0),)),
        (buy_event, ((leg.to_sleeve, 0.0),)),
    ]


def _apply_transfer(
    *,
    leg: Any,
    price: float,
    day: str,
    lots: Dict[str, List[P7Lot]],
    cash: Dict[str, float],
    seq: int,
) -> None:
    """S1~S3 抵消：卖方 FIFO 扣减、买方按同一物理来源新建批次、现金对转（零费用）。"""
    shares = int(leg.shares)
    seller = lots[leg.from_sleeve]
    consumed: List[Tuple[P7Lot, int]] = []
    remaining = shares
    for lot in fifo_lots(seller, leg.ts_code):
        if remaining == 0:
            break
        take = min(lot.shares, remaining)
        consumed.append((lot, take))
        remaining -= take
    if remaining != 0:
        raise LedgerError(
            f"内部抵消股数不足：袖 {leg.from_sleeve} 在 {leg.ts_code} 缺 {remaining} 股"
        )
    taken = {lot.virtual_lot_id: take for lot, take in consumed}
    kept: List[P7Lot] = []
    for lot in seller:
        take = taken.get(lot.virtual_lot_id)
        if take is None:
            kept.append(lot)
        elif lot.shares > take:
            kept.append(
                replace(
                    lot,
                    shares=lot.shares - take,
                    acquired_fee=lot.acquired_fee * (lot.shares - take) / lot.shares,
                )
            )
    lots[leg.from_sleeve] = kept
    amount = _as_amount(shares, price)
    available = float(cash[leg.to_sleeve])
    if available + 1e-9 < amount:
        raise LedgerError(
            f"内部转账超买方现金：袖 {leg.to_sleeve} 现金 {available!r} < 转账额 {amount!r}"
            f"（{leg.ts_code} {shares} 股 @ {price!r}）；转账限额逻辑失效"
        )
    cash[leg.from_sleeve] += amount
    cash[leg.to_sleeve] -= amount
    buyer = lots[leg.to_sleeve]
    for index, (lot, take) in enumerate(consumed):
        buyer.append(
            P7Lot(
                virtual_lot_id=f"{day}@{seq}#{index}",
                ts_code=lot.ts_code,
                source_lot_id=lot.source_lot_id,
                shares=take,
                acquired_date=day,
                acquired_price=price,
                acquired_fee=0.0,
                buy_date=lot.buy_date,
                due_date=lot.due_date,
                signal_date=lot.signal_date,
                origin_model_version=int(lot.origin_model_version),
            )
        )


# ────────────────────────── 市场卖出 ──────────────────────────


def _execute_sell(
    *,
    group: PlanGroup,
    result: GroupResult,
    prices: PriceBook,
    cost_model: Any,
    book: _WorkingBook,
    day: str,
) -> Dict[str, Any]:
    """卖出市场腿（净卖 S2/S6 或合并卖 S5）。"""
    market = group.market
    assert market is not None
    lookup = prices.execution_price(group.ts_code, "sell")
    if not lookup.ok:
        prices.record_block(lookup.reason)
        result.blocked_reason = result.blocked_reason or lookup.reason
        return _empty_market()
    price = float(lookup.price or 0.0)
    owners = [(sid, int(wanted)) for sid, wanted in market.owners]
    effective = _sellable_shares(group, owners, book.lots)
    if any(effective[sid] < wanted for sid, wanted in owners):
        result.blocked_reason = result.blocked_reason or "no_position"
    filled_total = sum(effective.values())
    if filled_total <= 0:
        return _empty_market()
    _consume_for_sell(group=group, effective=effective, lots=book.lots, phys_lots=book.phys_lots)
    amount = _as_amount(filled_total, price)
    declared_fee = float(cost_model.calculate_sell_cost(amount))
    final_fees, applied_fee, closure = _sell_fee_split(owners, effective, price, declared_fee)
    for sleeve_id, shares in effective.items():
        if shares <= 0:
            continue
        book.cash[sleeve_id] += _as_amount(shares, price)
        _distribute_filled(group, result, sleeve_id, "sell", shares)
    for sleeve_id, value in final_fees:
        book.cash[sleeve_id] -= float(value)
    book.phys_cash += amount - applied_fee
    book.sell_amount += amount
    book.fee_total += applied_fee
    event = fee_mod.FeeEvent(
        event_id=f"market:{day}:{group.ts_code}:sell",
        trade_date=day,
        ts_code=group.ts_code,
        side="sell",
        scenario=group.scenario,
        physical_fee=applied_fee,
        price=price,
        allocations=tuple((sid, int(effective[sid])) for sid in sorted(effective)),
    )
    trade = {
        "trade_date": day,
        "ts_code": group.ts_code,
        "kind": "market_sell",
        "scenario": group.scenario,
        "shares": int(filled_total),
        "price": price,
        "fee": applied_fee,
        "declared_fee": declared_fee,
        "sleeves": {sid: int(effective[sid]) for sid in sorted(effective) if effective[sid] > 0},
        "fee_split": {sid: float(value) for sid, value in final_fees},
        "float_closure": (
            None
            if closure is None
            else {
                "path": closure.path,
                "terminal_shift_ulps": int(closure.terminal_shift_ulps),
                "ulp_gap": float(closure.ulp_gap),
            }
        ),
    }
    return {
        "filled": int(filled_total),
        "fee_entries": [(event, final_fees)],
        "closures": [closure],
        "declared_fees": [declared_fee],
        "trades": [trade],
    }


def _empty_market(*, seq_used: int = 0) -> Dict[str, Any]:
    """市场腿未成交的统一空结果（闭合轨迹与申报费用同为空，保持等长约定）。"""
    return {
        "filled": 0,
        "fee_entries": [],
        "closures": [],
        "declared_fees": [],
        "trades": [],
        "seq_used": int(seq_used),
    }


def _sellable_shares(
    group: PlanGroup,
    owners: Sequence[Tuple[str, int]],
    lots: Mapping[str, Sequence[P7Lot]],
) -> Dict[str, int]:
    """可卖股数 = 该袖当前实际持有（计划账可能因前期未成交而不存在对应持仓）。"""
    effective: Dict[str, int] = {}
    for sleeve_id, wanted in owners:
        held = sum(lot.shares for lot in lots[sleeve_id] if lot.ts_code == group.ts_code)
        effective[sleeve_id] = min(int(wanted), int(held))
    return effective


def _sell_fee_split(
    owners: Sequence[Tuple[str, int]],
    effective: Mapping[str, int],
    price: float,
    declared_fee: float,
) -> Tuple[Tuple[Tuple[str, float], ...], float, Any]:
    """卖出费用归属：多袖同向走 Q3=A；单袖全额归属该袖。

    ``intent_shares`` 用 ``owners`` 的**原始卖出意图**（不是持仓截断后的有效股数），
    ``allocated_shares`` 用实际可卖股数 —— 这样 ``allocate_s5_fees`` 的
    「分得不得超意图」与「零份额不收费」两道守卫在主路径上**可失败**。

    Returns:
        ``(逐袖最终费用, 可用物理费用, 闭合轨迹)``。
    """
    if len(owners) > 1:
        split = fee_mod.allocate_s5_fees(
            intent_shares={sid: int(wanted) for sid, wanted in owners},
            allocated_shares={sid: int(effective[sid]) for sid, wanted in owners},
            price=price,
            physical_fee=declared_fee,
        )
        return tuple(split.final_fees), float(split.physical_fee), split.closure
    return ((owners[0][0], declared_fee),), declared_fee, None


def _consume_for_sell(
    *,
    group: PlanGroup,
    effective: Mapping[str, int],
    lots: Dict[str, List[P7Lot]],
    phys_lots: Dict[str, PhysicalLot],
) -> None:
    """逐袖 FIFO 扣减虚拟批次，并镜像到物理来源批次。"""
    consumed_by_source: Dict[str, int] = {}
    for sleeve_id in sorted(effective):
        shares = int(effective[sleeve_id])
        if shares <= 0:
            continue
        lots[sleeve_id] = _consume_sleeve_lots(
            lots=lots[sleeve_id],
            ts_code=group.ts_code,
            shares=shares,
            sleeve_id=sleeve_id,
            consumed_by_source=consumed_by_source,
        )
    for source_lot_id, shares in consumed_by_source.items():
        _decrement_physical(phys_lots, source_lot_id, shares)


def _consume_sleeve_lots(
    *,
    lots: Sequence[P7Lot],
    ts_code: str,
    shares: int,
    sleeve_id: str,
    consumed_by_source: Dict[str, int],
) -> List[P7Lot]:
    """单袖 FIFO 扣减（返回剩余批次；来源批次消耗量累加到 ``consumed_by_source``）。"""
    remaining = int(shares)
    taken: Dict[str, int] = {}
    for lot in fifo_lots(list(lots), ts_code):
        if remaining == 0:
            break
        take = min(lot.shares, remaining)
        taken[lot.virtual_lot_id] = take
        remaining -= take
    if remaining != 0:
        raise LedgerError(f"袖 {sleeve_id} 在 {ts_code} 的虚拟批次不足：仍缺 {remaining} 股")
    kept: List[P7Lot] = []
    for lot in lots:
        take = taken.get(lot.virtual_lot_id)
        if take is None:
            kept.append(lot)
            continue
        consumed_by_source[lot.source_lot_id] = consumed_by_source.get(lot.source_lot_id, 0) + take
        if lot.shares > take:
            kept.append(
                replace(
                    lot,
                    shares=lot.shares - take,
                    acquired_fee=lot.acquired_fee * (lot.shares - take) / lot.shares,
                )
            )
    return kept


def _decrement_physical(phys_lots: Dict[str, PhysicalLot], source_lot_id: str, shares: int) -> None:
    """物理来源批次扣减（不足即拒绝；与虚拟侧逐一镜像）。"""
    lot = phys_lots.get(source_lot_id)
    if lot is None:
        raise LedgerError(f"虚拟批次引用了物理不存在的来源批次：{source_lot_id}")
    if shares > lot.shares:
        raise LedgerError(
            f"物理批次 {source_lot_id} 股数不足：持有 {lot.shares}，申请卖出 {shares}"
        )
    if shares == lot.shares:
        phys_lots.pop(source_lot_id)
    else:
        phys_lots[source_lot_id] = replace(lot, shares=lot.shares - shares)


# ────────────────────────── 市场买入 ──────────────────────────


def _execute_buy(
    *,
    group: PlanGroup,
    result: GroupResult,
    prices: PriceBook,
    cost_model: Any,
    book: _WorkingBook,
    lot_size: int,
    seq: int,
    day: str,
) -> Dict[str, Any]:
    """买入市场腿（净买 S3/S6 或合并买 S4；Q18 股数分摊 + Q13 费用瀑布）。"""
    from src.lazybull.v2.core.accounting.settlement_kernel import allocate_shares

    market = group.market
    assert market is not None
    lookup = prices.execution_price(group.ts_code, "buy")
    if not lookup.ok:
        prices.record_block(lookup.reason)
        result.blocked_reason = result.blocked_reason or lookup.reason
        return _empty_market(seq_used=0)
    price = float(lookup.price or 0.0)
    effective, capped = _affordable_by_sleeve(
        group=group, price=price, lot_size=lot_size, cost_model=cost_model, book=book
    )
    if capped:
        result.blocked_reason = result.blocked_reason or "insufficient_cash"
    if sum(effective.values()) <= 0:
        return _empty_market(seq_used=0)
    resolved = _resolve_buy_allocation(
        group=group,
        effective=effective,
        price=price,
        cash=book.cash,
        cost_model=cost_model,
        lot_size=lot_size,
        allocate_shares=allocate_shares,
    )
    if resolved is None:
        result.blocked_reason = result.blocked_reason or "insufficient_cash"
        return _empty_market(seq_used=0)
    total, share_map, waterfall = resolved
    amount = _as_amount(total, price)
    declared_fee, applied_fee = fee_mod.declared_and_applied(waterfall)
    source_lot_id = f"P{day}#{group.ts_code}#{int(seq)}"
    if source_lot_id in book.phys_lots:
        raise LedgerError(f"物理来源批次身份冲突：{source_lot_id}")
    book.phys_lots[source_lot_id] = PhysicalLot(
        source_lot_id=source_lot_id,
        ts_code=group.ts_code,
        shares=int(total),
        buy_date=day,
        buy_price=price,
        buy_fee=applied_fee,
    )
    fee_map = {sid: float(value) for sid, value in waterfall.final_fees}
    for sleeve_id in sorted(share_map):
        allocated = int(share_map[sleeve_id])
        if allocated <= 0:
            continue
        value = float(fee_map.get(sleeve_id, 0.0))
        book.cash[sleeve_id] -= _as_amount(allocated, price) + value
        _append_buy_lots(
            group=group,
            result=result,
            sleeve_id=sleeve_id,
            allocated=allocated,
            fee_value=value,
            price=price,
            day=day,
            source_lot_id=source_lot_id,
            lots=book.lots,
        )
    if float(book.phys_cash) + 1e-9 < amount + applied_fee:
        raise LedgerError(
            f"物理现金不足：可用 {float(book.phys_cash)!r} < 应付 {amount + applied_fee!r}"
            f"（{day} {group.ts_code} 市场买入）；非杠杆前提破缺"
        )
    book.phys_cash -= amount + applied_fee
    book.buy_amount += amount
    book.fee_total += applied_fee
    event = fee_mod.FeeEvent(
        event_id=f"market:{day}:{group.ts_code}:buy",
        trade_date=day,
        ts_code=group.ts_code,
        side="buy",
        scenario=group.scenario,
        physical_fee=applied_fee,
        price=price,
        allocations=tuple((sid, int(share_map[sid])) for sid in sorted(share_map)),
    )
    trade = {
        "trade_date": day,
        "ts_code": group.ts_code,
        "kind": "market_buy",
        "scenario": group.scenario,
        "shares": int(total),
        "price": price,
        "fee": applied_fee,
        "declared_fee": declared_fee,
        "source_lot_id": source_lot_id,
        "sleeves": {
            sid: int(share_map[sid]) for sid in sorted(share_map) if int(share_map[sid]) > 0
        },
        # C4-02：冻结 Q18 的两项可复核输入（原始增量 + 物理 Q）。
        # 审阅方可用 ``allocate_shares(intent_shares, shares)`` 逐股复算，无需信任实现。
        "intent_shares": {str(sid): int(shares) for sid, shares in group.market.owners},
        "allocated_fees": {sid: float(value) for sid, value in waterfall.final_fees},
        "terminal_sleeve": str(waterfall.terminal_sleeve),
        "residual_correction": float(waterfall.residual_correction),
        "proportional_branch": bool(waterfall.proportional_branch),
        "float_closure": (
            None
            if waterfall.closure is None
            else {
                "path": waterfall.closure.path,
                "terminal_shift_ulps": int(waterfall.closure.terminal_shift_ulps),
                "ulp_gap": float(waterfall.closure.ulp_gap),
            }
        ),
    }
    return {
        "filled": int(total),
        "fee_entries": [(event, tuple(waterfall.final_fees))],
        "closures": [waterfall.closure],
        "declared_fees": [declared_fee],
        "trades": [trade],
        "seq_used": 1,
    }


def _affordable_by_sleeve(
    *,
    group: PlanGroup,
    price: float,
    lot_size: int,
    cost_model: Any,
    book: _WorkingBook,
) -> Tuple[Dict[str, int], bool]:
    """逐袖现金约束（无借用）：返回 ``(可负担股数, 是否存在截断)``。"""
    market = group.market
    assert market is not None
    effective: Dict[str, int] = {}
    capped = False
    for sleeve_id, wanted in market.owners:
        shares = _affordable_shares(
            cash=book.cash[sleeve_id],
            price=price,
            lot_size=lot_size,
            buy_cost=cost_model.calculate_buy_cost,
            wanted=int(wanted),
        )
        effective[sleeve_id] = shares
        if shares < int(wanted):
            capped = True
    return effective, capped


def _append_buy_lots(
    *,
    group: PlanGroup,
    result: GroupResult,
    sleeve_id: str,
    allocated: int,
    fee_value: float,
    price: float,
    day: str,
    source_lot_id: str,
    lots: Dict[str, List[P7Lot]],
) -> None:
    """按意图分摊成交股数并逐意图建虚拟批次（继承到期义务与建仓模型版本）。"""
    per_intent = _distribute_filled(group, result, sleeve_id, "buy", allocated)
    for seq_no, (intent_key, shares_of_intent) in enumerate(per_intent):
        intent = _intent_by_key(group, intent_key)
        lots[sleeve_id].append(
            P7Lot(
                virtual_lot_id=f"{day}${source_lot_id}#{sleeve_id}#{seq_no}",
                ts_code=group.ts_code,
                source_lot_id=source_lot_id,
                shares=int(shares_of_intent),
                acquired_date=day,
                acquired_price=price,
                acquired_fee=(
                    fee_value if len(per_intent) == 1 else fee_value * shares_of_intent / allocated
                ),
                buy_date=day,
                due_date=intent.due_date if intent else "",
                signal_date=intent.signal_date if intent else "",
                origin_model_version=int(intent.model_version) if intent else 0,
            )
        )


def _resolve_buy_allocation(
    *,
    group: PlanGroup,
    effective: Mapping[str, int],
    price: float,
    cash: Mapping[str, float],
    cost_model: Any,
    lot_size: int,
    allocate_shares: Any,
) -> Optional[Tuple[int, Dict[str, int], Any]]:
    """物理买入股数 + Q18 股数分摊 + Q13 费用瀑布（含现金二次校验的整手回退）。

    ``q_i``（受益顺序基准）取 ``group.market.owners`` 的**原始买入意图金额**，
    ``a_i`` 取 Q18 分得金额 —— 契约要求 ``q_i`` 是原始意图，不能是现金截断后的
    可负担金额（否则受益顺序被资金约束改写）。
    """
    owner_shares = {str(sid): int(shares) for sid, shares in group.market.owners}
    caps = {sid: max(0, int(effective.get(sid, 0))) for sid in owner_shares}
    if not any(caps.values()):
        return None
    original = dict(owner_shares)
    # C4-02：Q18 分摊基数必须是**冻结原始意图**（契约 §5 `Q × 原始增量_i / Σ原始增量`）。
    # 现金可执行量（``caps``）只用于**上限裁剪**，不得改写分摊基数——否则资金约束会
    # 悄悄改变股数归属、批次与后续费用归属。
    # 基数 = 冻结原始意图（**含零资金袖**，不得因资金为 0 就把袖从分母里删掉）；
    # 零意图袖的容量 0 由受审 ``allocate_shares`` 自己处理，不得用 1 股兜底污染分母。
    weights = {sid: int(original.get(sid, 0)) for sid in sorted(owner_shares)}
    # C4-02：**不做落地后重分配**。冻结公式给出唯一向量，若该向量超出某个袖的费用后现金，
    # 就**降低物理 Q**（整手步长）再算一次，直到向量本身可行；Q 降到 0 即不成交。
    # 这样最终向量始终是 `Q x 原始增量_i / Σ原始增量` 的直接结果，不是另一套分摊规则。
    total = _lot_round(sum(caps.values()), lot_size)
    step = max(1, int(lot_size))
    while total > 0:
        shares_alloc = allocate_shares(weights, total)
        share_map = {item.sleeve_id: int(item.allocated_shares) for item in shares_alloc}
        if sum(share_map.values()) != total:
            total -= step
            continue
        amount = _as_amount(total, price)
        physical_fee = float(cost_model.calculate_buy_cost(amount))
        waterfall = fee_mod.allocate_s4_fees(
            intent_amounts={sid: _as_amount(weights[sid], price) for sid in share_map},
            allocated_amounts={sid: _as_amount(share_map[sid], price) for sid in share_map},
            physical_fee=physical_fee,
            buy_cost=cost_model.calculate_buy_cost,
        )
        fee_map = {sid: float(value) for sid, value in waterfall.final_fees}
        overspend = [
            sid
            for sid in share_map
            if _as_amount(share_map[sid], price) + float(fee_map.get(sid, 0.0)) > cash[sid]
        ]
        if not overspend:
            return total, share_map, waterfall
        total -= step
    return None


# ────────────────────────── 辅助 ──────────────────────────


def _intent_caps(
    group: PlanGroup, sleeve_id: str, side: str, result: GroupResult
) -> Dict[str, int]:
    """该袖在该方向的逐意图可成交上限（原意图 - 已抵消）。"""
    caps: Dict[str, int] = {}
    for intent in group.intents:
        if intent.sleeve_id != sleeve_id or intent.side != side:
            continue
        caps[intent.intent_key] = int(intent.shares) - int(
            result.offset_shares.get(intent.intent_key, 0)
        )
    return caps


def _distribute_filled(
    group: PlanGroup,
    result: GroupResult,
    sleeve_id: str,
    side: str,
    total: int,
) -> List[Tuple[str, int]]:
    """把袖级成交股数按意图身份确定性分摊（字典序；不得超过各意图剩余量）。"""
    caps = _intent_caps(group, sleeve_id, side, result)
    remaining = int(total)
    allocated: List[Tuple[str, int]] = []
    for key in sorted(caps):
        if remaining <= 0:
            break
        take = min(remaining, caps[key])
        if take <= 0:
            continue
        result.filled_shares[key] = result.filled_shares.get(key, 0) + take
        allocated.append((key, take))
        remaining -= take
    if remaining != 0:
        raise LedgerError(
            f"{group.ts_code} 袖 {sleeve_id} 的成交股数 {total} 超过意图剩余量之和"
            f"（{sum(caps.values())}）；分摊无法闭合"
        )
    return allocated


def _intent_by_key(group: PlanGroup, key: str) -> Optional[Intent]:
    for intent in group.intents:
        if intent.intent_key == key:
            return intent
    return None


def _first_blocked(intent: Intent, day: str) -> str:
    """首次阻断日：新意图取当日；顺延意图沿用残差里记着的首阻断日（C4-05）。"""
    if not intent.carried:
        return day
    return str(intent.first_blocked_date or intent.signal_date or intent.due_date or day)


def _carried_days(intent: Intent) -> int:
    """已顺延天数：新意图 0；顺延意图沿用残差的累计值（C4-05）。"""
    return int(intent.carried_days) if intent.carried else 0
