# -*- coding: utf-8 -*-
"""P7 T4 净额化生命周期（S1~S7 的**零副作用**预净额计划）。

契约依据
========
``docs/contracts/netting_freeze.md`` §3 状态机表与 P7 注记（F26）：

===========  ==========================================  ==================================
场景          物理市场指令                                 虚拟账本动作
===========  ==========================================  ==================================
S1           无（严格等量抵消）                           抵消股数按当日真实开盘价内部转账
S2           净卖 ``s-b``（涨跌停/停牌顺延）               抵消 ``b``；**净额与顺延资格只归卖方袖**
S3           净买 ``b-s``（涨跌停/停牌顺延）               抵消 ``s``；**净额与顺延资格只归买方袖**
S4           合并买 ``Σb``                                 剩余意图日终作废（Q22 保留于 F26）
S5           合并卖 ``Σs``                                 各袖按原意图减实际分得（Q3=A 费用）
S6           单袖指令原样                                  多袖账户单袖发单 = 该袖自身结算
S7           持有到期（不新增场景）                         各袖 lot 独立到期后进入同一净额流程
===========  ==========================================  ==================================

关键不变式
==========
- 预净额计划**不读价格、不改状态**：只按股数与方向生成抵消腿/市场腿；
- **抵消腿只有在卖方袖确实持有对应股数时才能生成**（否则计划非法，拒绝而非静默降级）；
- 残差只来自**未成交**部分；已抵消/已成交流程永不回流 => 次日重新净额天然**不重复抵消**；
- ``original = offset + filled + voided + carry`` 逐意图恒等（`:func:`assert_intent_conservation``）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.v2_p7.t4.state import (
    ORIGIN_CARRY,
    ORIGIN_EXPIRY_SELL,
    ORIGIN_SIGNAL_BUY,
    SIDES,
)

#: 场景标签
S1, S2, S3, S4, S5, S6 = "S1", "S2", "S3", "S4", "S5", "S6"


class NettingError(RuntimeError):
    """预净额计划非法（意图冲突/超持仓/身份重复），拒绝且零副作用。"""


@dataclass(frozen=True)
class Intent:
    """单袖单股单方向意图（``intent_key`` 跨日稳定，用于不重复抵消核对）。"""

    intent_key: str
    sleeve_id: str
    ts_code: str
    side: str
    shares: int
    origin: str
    signal_date: str
    due_date: str
    #: 建仓信号所属模型版本（lot 的 ``origin_model_version`` 来源；0 = 未登记）
    model_version: int = 0
    #: 首次阻断日（顺延意图沿用残差值，不回落信号日；C4-05）
    first_blocked_date: str = ""
    #: 已顺延交易日数（顺延意图沿用残差累计值；C4-05）
    carried_days: int = 0

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise NettingError(f"意图 {self.intent_key} 方向必须为 buy/sell：{self.side!r}")
        if not isinstance(self.shares, int) or isinstance(self.shares, bool) or self.shares <= 0:
            raise NettingError(f"意图 {self.intent_key} 股数必须为正整数：{self.shares!r}")

    @property
    def carried(self) -> bool:
        return self.origin == ORIGIN_CARRY


@dataclass(frozen=True)
class OffsetLeg:
    """内部抵消腿（S1~S3；撮合价在成交步按当日真实开盘价落定）。"""

    ts_code: str
    from_sleeve: str
    to_sleeve: str
    shares: int
    from_intent_key: str
    to_intent_key: str


@dataclass(frozen=True)
class MarketLeg:
    """物理市场腿（净额或合并；``owners`` 为参与该腿的袖与其原始意图股数）。"""

    ts_code: str
    side: str
    shares: int
    owners: Tuple[Tuple[str, int], ...]

    @property
    def single_owner(self) -> Optional[str]:
        return self.owners[0][0] if len(self.owners) == 1 else None


@dataclass(frozen=True)
class PlanGroup:
    """单股单日的净额计划。"""

    ts_code: str
    scenario: str
    intents: Tuple[Intent, ...]
    offsets: Tuple[OffsetLeg, ...]
    market: Optional[MarketLeg]

    @property
    def offset_shares(self) -> int:
        return sum(leg.shares for leg in self.offsets)

    @property
    def market_shares(self) -> int:
        return 0 if self.market is None else int(self.market.shares)


@dataclass(frozen=True)
class NettingPlan:
    """单日预净额计划（零副作用）。"""

    trade_date: str
    groups: Tuple[PlanGroup, ...] = ()
    #: 计划阶段即作废的意图（卖出意图超过实际持仓 => 「原信号资格」已失效，不顺延）
    voided_rows: Tuple[Dict[str, Any], ...] = ()

    def scenario_counts(self) -> Dict[str, int]:
        counts = {label: 0 for label in (S1, S2, S3, S4, S5, S6)}
        for group in self.groups:
            counts[group.scenario] = counts.get(group.scenario, 0) + 1
        return counts

    def offset_shares(self) -> int:
        return sum(group.offset_shares for group in self.groups)

    def market_shares(self) -> int:
        return sum(group.market_shares for group in self.groups)


def classify_sides(per_sleeve: Mapping[str, Tuple[int, int]]) -> str:
    """按状态机表归类（与 T2.5 ``intent_precheck._classify`` **同义**，泛化到 N 袖）。

    S1/S2/S3 属**跨袖反向**（一袖卖、另一袖买）；S4/S5 属**跨袖同向**；
    只有一袖有指令即 S6（单袖）。同袖同时有买卖不计反向（本引擎不会出现，
    若出现即视为非法输入，由 :func:`plan_group` 拒绝）。

    Args:
        per_sleeve: ``{sleeve_id: (buy_shares, sell_shares)}``
    """
    buy_sleeves = [sid for sid, (buy, _sell) in per_sleeve.items() if buy > 0]
    sell_sleeves = [sid for sid, (_buy, sell) in per_sleeve.items() if sell > 0]
    reverse = any(
        left in sell_sleeves and right in buy_sleeves
        for left in sell_sleeves
        for right in buy_sleeves
        if left != right
    )
    if reverse:
        sell_total = sum(sell for _buy, sell in per_sleeve.values())
        buy_total = sum(buy for buy, _sell in per_sleeve.values())
        if sell_total == buy_total:
            return S1
        return S2 if sell_total > buy_total else S3
    if len(buy_sleeves) >= 2:
        return S4
    if len(sell_sleeves) >= 2:
        return S5
    return S6


def _sum_by_sleeve(intents: Iterable[Intent]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for intent in intents:
        buy, sell = out.get(intent.sleeve_id, (0, 0))
        if intent.side == "buy":
            buy += intent.shares
        else:
            sell += intent.shares
        out[intent.sleeve_id] = (buy, sell)
    return out


def _pair_offsets(
    intents: Sequence[Intent],
    available: Mapping[str, Mapping[str, int]],
) -> List[OffsetLeg]:
    """按 ``sleeve_id`` 字典序做确定性配对，总量 = ``min(Σ卖, Σ买)``。"""
    sell_keys = [
        intent for intent in sorted(intents, key=_intent_sort_key) if intent.side == "sell"
    ]
    buy_keys = [intent for intent in sorted(intents, key=_intent_sort_key) if intent.side == "buy"]
    remaining_sell = {intent.intent_key: intent.shares for intent in sell_keys}
    remaining_buy = {intent.intent_key: intent.shares for intent in buy_keys}
    legs: List[OffsetLeg] = []
    for seller in sell_keys:
        for buyer in buy_keys:
            if seller.sleeve_id == buyer.sleeve_id:
                continue
            shares = min(remaining_sell[seller.intent_key], remaining_buy[buyer.intent_key])
            if shares <= 0:
                continue
            holding = int(available.get(seller.sleeve_id, {}).get(seller.ts_code, 0))
            if holding < shares:
                raise NettingError(
                    f"抵消腿超持仓：袖 {seller.sleeve_id} 在 {seller.ts_code} 持有 {holding} 股，"
                    f"意图转出 {shares} 股（意图={seller.intent_key}）"
                )
            legs.append(
                OffsetLeg(
                    ts_code=seller.ts_code,
                    from_sleeve=seller.sleeve_id,
                    to_sleeve=buyer.sleeve_id,
                    shares=shares,
                    from_intent_key=seller.intent_key,
                    to_intent_key=buyer.intent_key,
                )
            )
            remaining_sell[seller.intent_key] -= shares
            remaining_buy[buyer.intent_key] -= shares
    return legs


def _intent_sort_key(intent: Intent) -> Tuple[str, str, str]:
    return (intent.sleeve_id, intent.intent_key, intent.ts_code)


def plan_group(
    ts_code: str,
    intents: Sequence[Intent],
    available: Mapping[str, Mapping[str, int]],
) -> PlanGroup:
    """单股预净额（不做价格/资金判定，只决定抵消腿与市场腿）。"""
    per_sleeve = _sum_by_sleeve(intents)
    scenario = classify_sides(per_sleeve)
    for sleeve_id, (buy, sell) in per_sleeve.items():
        if buy > 0 and sell > 0:
            raise NettingError(
                f"{ts_code} 的袖 {sleeve_id} 同日同时有买 {buy} 与卖 {sell}；同袖双向不在状态机表内"
            )
    offsets: List[OffsetLeg] = []
    market: Optional[MarketLeg] = None
    if scenario in (S1, S2, S3):
        offsets = _pair_offsets(intents, available)
        sell_total = sum(sell for _buy, sell in per_sleeve.values())
        buy_total = sum(buy for buy, _sell in per_sleeve.values())
        net = sell_total - buy_total
        if scenario == S2:
            owner = _net_owner(intents, "sell")
            market = MarketLeg(ts_code, "sell", net, ((owner, net),))
        elif scenario == S3:
            owner = _net_owner(intents, "buy")
            market = MarketLeg(ts_code, "buy", -net, ((owner, -net),))
    elif scenario in (S4, S5):
        side = "buy" if scenario == S4 else "sell"
        if any(intent.side != side for intent in intents):
            raise NettingError(
                f"{ts_code} 的 {scenario} 计划混入反向意图："
                f"{[(item.sleeve_id, item.side) for item in intents]}"
            )
        # 同一袖可能同时有顺延意图与新意图：市场腿按**袖合计**登记，成交后再按意图分摊
        sums: Dict[str, int] = {}
        for intent in sorted(intents, key=_intent_sort_key):
            sums[intent.sleeve_id] = sums.get(intent.sleeve_id, 0) + intent.shares
        owners = tuple(sorted(sums.items()))
        market = MarketLeg(ts_code, side, sum(shares for _sid, shares in owners), owners)
    else:
        if len(intents) != 1:
            raise NettingError(
                f"{ts_code} 的 {scenario} 单袖计划必须只有一条意图：{[i.intent_key for i in intents]}"
            )
        only = intents[0]
        market = MarketLeg(ts_code, only.side, only.shares, ((only.sleeve_id, only.shares),))
    return PlanGroup(
        ts_code=str(ts_code),
        scenario=scenario,
        intents=tuple(sorted(intents, key=_intent_sort_key)),
        offsets=tuple(offsets),
        market=market,
    )


def _net_owner(intents: Sequence[Intent], side: str) -> str:
    """净额方向的袖子（S2 归卖方、S3 归买方；多袖同侧时取字典序首个并显式登记）。"""
    owners = sorted({intent.sleeve_id for intent in intents if intent.side == side})
    if not owners:
        raise NettingError(f"净额方向 {side} 无对应袖子")
    return owners[0]


def cap_sell_intents(
    intents: Sequence[Intent],
    available: Mapping[str, Mapping[str, int]],
    trade_date: str,
) -> Tuple[List[Intent], List[Dict[str, Any]]]:
    """卖出意图按袖实际持仓截断（超出部分即「原信号资格失效」，日终作废不顺延）。

    截断必须发生在预净额**之前**：否则抵消腿会引用不存在的持仓，把计划做废。
    """
    remaining: Dict[str, Dict[str, int]] = {}
    capped: List[Intent] = []
    voided: List[Dict[str, Any]] = []
    for intent in sorted(intents, key=_intent_sort_key):
        if intent.side != "sell":
            capped.append(intent)
            continue
        held = remaining.setdefault(intent.sleeve_id, dict(available.get(intent.sleeve_id, {})))
        free = int(held.get(intent.ts_code, 0))
        take = min(intent.shares, free)
        if take > 0:
            held[intent.ts_code] = free - take
            capped.append(replace_intent(intent, take))
        lost = intent.shares - take
        if lost > 0:
            # 该行的 ``original`` 只记**被作废的差额**：截断后的意图另出一行，
            # 两行按 ``intent_key`` 相加才等于原始意图（逐行也各自守恒）。
            voided.append(
                {
                    "trade_date": str(trade_date),
                    "intent_key": intent.intent_key,
                    "sleeve": intent.sleeve_id,
                    "ts_code": intent.ts_code,
                    "side": "sell",
                    "scenario": "n/a",
                    "original": int(lost),
                    "offset": 0,
                    "filled": 0,
                    "voided": int(lost),
                    "carry": 0,
                    "carried_in": bool(intent.carried),
                    "void_reason": "no_position_voided",
                    # 截断前的原始意图股数：跨日聚合时用于复原「首次原始意图」
                    "truncated_from": int(intent.shares),
                }
            )
    return capped, voided


def replace_intent(intent: Intent, shares: int) -> Intent:
    """按新股数复制意图（身份不变 => 守恒核对仍按原 ``intent_key`` 累加）。"""
    return Intent(
        intent_key=intent.intent_key,
        sleeve_id=intent.sleeve_id,
        ts_code=intent.ts_code,
        side=intent.side,
        shares=int(shares),
        origin=intent.origin,
        signal_date=intent.signal_date,
        due_date=intent.due_date,
        model_version=intent.model_version,
        # C4-05：截断股数不得顺手把顺延年龄与首阻断日重置
        first_blocked_date=intent.first_blocked_date,
        carried_days=intent.carried_days,
    )


def assert_single_direction_precondition(intents: Sequence[Intent]) -> None:
    """显式前置断言：同一袖同一股票**当日只有一个方向**（T4 两袖形态的前提）。

    多袖（N≥3）下的净额归并与同袖双向属 T5/T6 范围；此处 fail-fast 并给出原因，
    不静默降级（复审 B-15）。
    """
    pairs: Dict[Tuple[str, str], str] = {}
    for intent in intents:
        key = (intent.sleeve_id, intent.ts_code)
        seen = pairs.get(key)
        if seen is not None and seen != intent.side:
            raise NettingError(
                f"袖 {key[0]} 在 {key[1]} 同日出现双向意图（{seen} / {intent.side}）；"
                "T4 只覆盖「每袖单日单股单向」的冻结形态，同袖双向归 T5/T6"
            )
        pairs[key] = intent.side


def plan_day(
    trade_date: str,
    intents: Sequence[Intent],
    available: Mapping[str, Mapping[str, int]],
) -> NettingPlan:
    """单日预净额计划（逐股分组；``available`` = 各袖当前可卖股数）。

    先按实际持仓截断卖出意图（超出部分登记为 ``no_position_voided`` 作废），
    再做跨袖抵消与市场腿判定。
    """
    if not intents:
        return NettingPlan(trade_date=str(trade_date), groups=())
    assert_single_direction_precondition(intents)
    capped, voided = cap_sell_intents(intents, available, trade_date)
    keys: Dict[str, int] = {}
    for intent in capped:
        key = f"{intent.sleeve_id}|{intent.intent_key}"
        if key in keys:
            raise NettingError(f"同一袖同一意图身份重复：{key}")
        keys[key] = 1
    grouped: Dict[str, List[Intent]] = {}
    for intent in capped:
        grouped.setdefault(str(intent.ts_code), []).append(intent)
    groups = tuple(plan_group(code, grouped[code], available) for code in sorted(grouped))
    return NettingPlan(
        trade_date=str(trade_date),
        groups=groups,
        voided_rows=tuple(voided),
    )


def assert_intent_conservation(
    rows: Sequence[Mapping[str, Any]],
    tolerance_shares: int = 0,
) -> Dict[str, int]:
    """``original = offset + filled + voided + carry`` 逐意图恒等校验。

    Args:
        rows: 每个意图一行，含 ``intent_key``/``original``/``offset``/``filled``/
            ``voided``/``carry`` 六个整数字段

    Returns:
        ``{"rows": n, "checked": n, "violations": m}``；``violations > 0`` 即拒绝。
    """
    violations = 0
    for row in rows:
        total = int(row["offset"]) + int(row["filled"]) + int(row["voided"]) + int(row["carry"])
        if abs(total - int(row["original"])) > int(tolerance_shares):
            violations += 1
    return {"rows": len(rows), "checked": len(rows), "violations": violations}


def cross_day_conservation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """按 ``intent_key`` 跨日聚合，检验「次日重净额**不重复抵消**」。

    逐日逐行的 ``original = offset + filled + voided + carry`` 只保证单日自平衡
    （顺延后次日会以缩小后的 ``original`` 重新出现，任何二次抵消都能通过）。
    本函数改为按意图身份跨日复原：

    - ``first_original`` = 该 ``intent_key`` **最早一行**的原始意图股数；
    - ``Σoffset + Σfilled + Σvoided + 末次 carry`` 必须等于 ``first_original``；
    - ``Σoffset <= first_original``（抵消总额不得超过首次原始意图）。

    Returns:
        ``{"intents": n, "violations": m, "examples": [...]}``；``violations > 0`` 即拒绝。
    """
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["intent_key"]), []).append(row)
    violations: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        series = sorted(grouped[key], key=lambda item: (str(item["trade_date"]), item["original"]))
        # 首次原始意图 = **最早交易日的全部行之和**：部分截断时该日会有
        # 「意图行（original=截断后股数）」+「cap 行（original=作废差额）」两行，
        # 全额截断时只有 cap 行；两种情形相加都等于首日真实原意图。
        earliest = str(series[0]["trade_date"])
        first_original = sum(
            int(item["original"]) for item in series if str(item["trade_date"]) == earliest
        )
        total_offset = sum(int(item["offset"]) for item in series)
        total_filled = sum(int(item["filled"]) for item in series)
        total_voided = sum(int(item["voided"]) for item in series)
        last_carry = int(series[-1]["carry"])
        closed = total_offset + total_filled + total_voided + last_carry
        if closed != first_original or total_offset > first_original:
            violations.append(
                {
                    "intent_key": key,
                    "first_original": first_original,
                    "offset": total_offset,
                    "filled": total_filled,
                    "voided": total_voided,
                    "last_carry": last_carry,
                    "closed_total": closed,
                    "days": len(series),
                }
            )
    return {
        "intents": len(grouped),
        "rows": len(rows),
        "violations": len(violations),
        "examples": violations[:5],
        "rule": (
            "first_original = Σoffset + Σfilled + Σvoided + 末次 carry；"
            "且 Σoffset <= first_original（容差 0 股）"
        ),
    }


def intent_from_plan_row(
    *,
    sleeve_id: str,
    ts_code: str,
    side: str,
    shares: int,
    signal_date: str,
    due_date: str,
    model_version: int = 0,
) -> Intent:
    """把 T2.5 计划行转成意图（``intent_key`` 跨日稳定且**跨袖唯一**）。

    ``intent_key = "<袖>|<方向>|<参考日>|<股票>"``：跨袖同股同向是**不同身份**
    （各自独立守恒核对），但共用同一条物理市场腿。
    """
    origin = ORIGIN_SIGNAL_BUY if side == "buy" else ORIGIN_EXPIRY_SELL
    reference = str(signal_date) if side == "buy" else str(due_date)
    key = f"{sleeve_id}|{side}|{reference}|{ts_code}"
    return Intent(
        intent_key=key,
        sleeve_id=str(sleeve_id),
        ts_code=str(ts_code),
        side=str(side),
        shares=int(shares),
        origin=origin,
        signal_date=str(signal_date),
        due_date=str(due_date),
        model_version=int(model_version),
    )


def carried_intent(entry: Any) -> Intent:
    """把袖级残差转成次日意图（保留原 ``intent_key`` => 不重复抵消可核对）。"""
    return Intent(
        intent_key=str(entry.intent_key),
        sleeve_id=str(entry.sleeve_id),
        ts_code=str(entry.ts_code),
        side=str(entry.side),
        shares=int(entry.shares),
        origin=ORIGIN_CARRY,
        signal_date=str(entry.signal_date),
        due_date=str(entry.due_date),
        model_version=int(getattr(entry, "model_version", 0) or 0),
        first_blocked_date=str(getattr(entry, "first_blocked_date", "") or ""),
        carried_days=int(getattr(entry, "days_carried", 0) or 0),
    )


# -------------------------- 残差作废原因（自 ledger.py 迁入） ----------
VOID_REASON_S4_REMAINDER = "s4_remainder_voided_eod"
VOID_REASON_FUNDING = "insufficient_cash_voided_eod"
VOID_REASON_NO_POSITION = "no_position_voided"

#: 市场可达性阻断（跨日顺延）
MARKET_BLOCK_REASONS: Tuple[str, ...] = (
    "no_valid_open",
    "row_absent",
    "suspended",
    "limit_up_blocked",
    "limit_down_blocked",
)


def _held_by_code(lots: Sequence[Any]) -> Dict[str, int]:
    """该袖当前逐股持仓（用于判断「卖出义务是否还有底仓」）。"""
    held: Dict[str, int] = {}
    for lot in lots:
        held[lot.ts_code] = held.get(lot.ts_code, 0) + int(lot.shares)
    return held


def _void_reason(
    scenario: str,
    result: Any,
    intent: Intent,
    remaining: int,
    held: int = 0,
) -> str:
    """日终作废判定（S4 剩余 / 资金不足 / 无持仓；市场可达性阻断则顺延）。"""
    if remaining <= 0:
        return ""
    if scenario == S4:
        return VOID_REASON_S4_REMAINDER
    reason = result.blocked_reason
    if reason in MARKET_BLOCK_REASONS:
        return ""
    if reason == "no_position":
        return VOID_REASON_NO_POSITION
    if reason == "insufficient_cash":
        return VOID_REASON_FUNDING
    if intent.side == "buy":
        return VOID_REASON_FUNDING
    if held >= remaining:
        # 仍有底仓的卖出义务（C4-R1-01）：不得因对手方缺钱/无市场腿被记成
        # 「无持仓作废」——那会凭空取消到期退出。留空即转顺延，义务跨日保留。
        return ""
    return VOID_REASON_NO_POSITION


def _rebuild_market_legs(
    plan: NettingPlan, capped_sell: Mapping[str, Mapping[str, int]]
) -> NettingPlan:
    """转账被资金截断后**重建市场腿**（C4-R1-01）。

    计划期的 ``group.market`` 是按原始双边意图算出来的；抵消量一旦被截断，净市场量随之改变。
    未成交的卖方股数必须并回市场卖出腿（存在则合并，不存在则新建），否则这笔卖出会被
    误判为「无持仓」而丢掉退出义务。
    """
    if not capped_sell:
        return plan
    groups: List[PlanGroup] = []
    for group in plan.groups:
        extra = capped_sell.get(group.ts_code)
        if not extra:
            groups.append(group)
            continue
        market = group.market
        owners: Dict[str, int] = {}
        if market is not None and market.side == "sell":
            owners.update({sid: int(shares) for sid, shares in market.owners})
        for sleeve_id, shares in extra.items():
            owners[sleeve_id] = owners.get(sleeve_id, 0) + int(shares)
        owners = {sid: value for sid, value in owners.items() if value > 0}
        if not owners:
            groups.append(group)
            continue
        total = sum(owners.values())
        groups.append(
            replace(
                group,
                market=MarketLeg(
                    ts_code=group.ts_code,
                    side="sell",
                    shares=int(total),
                    owners=tuple((sid, int(owners[sid])) for sid in sorted(owners)),
                ),
            )
        )
    return replace(plan, groups=tuple(groups))


def assert_group_consistency(group: PlanGroup) -> None:
    """计划自检：抵消腿股数与市场腿股数必须能解释全部意图股数。"""
    offset_by_intent: Dict[str, int] = {}
    for leg in group.offsets:
        offset_by_intent[leg.from_intent_key] = (
            offset_by_intent.get(leg.from_intent_key, 0) + leg.shares
        )
        offset_by_intent[leg.to_intent_key] = (
            offset_by_intent.get(leg.to_intent_key, 0) + leg.shares
        )
    for intent in group.intents:
        offset = offset_by_intent.get(intent.intent_key, 0)
        if offset > intent.shares:
            raise NettingError(
                f"意图 {intent.intent_key} 抵消 {offset} 股超过原意图 {intent.shares} 股"
            )
    if group.market is not None:
        if group.market.shares <= 0:
            raise NettingError(f"{group.ts_code} 的市场腿股数必须为正：{group.market.shares}")
        if group.scenario == S1:
            raise NettingError(f"{group.ts_code} 的 S1 场景不得有市场腿")
    elif group.scenario not in (S1,):
        raise NettingError(f"{group.ts_code} 的 {group.scenario} 场景缺少市场腿")
