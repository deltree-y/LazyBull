"""P6.5 T5 五项状态所有者的投影/还原（独立同代状态信封的内容层）。

契约依据：设计 §4.5 五项所有权表与 §7.2 失败模式；本模块只做**纯编码/解码**与
**统一写入**，不做文件 IO、不读配置、不读真实行情。

分工：
- ``build_*_owner``：从运行时对象（``EngineState`` / ``PhysicalLotView`` /
  ``VirtualLedger`` / 宿主链结果）投影出所有者字节，**只读、零写入**；
- ``decode_owners``：把信封五项解码为 ``RestoredOwners``，并完成**跨项同代一致性**
  校验（物理幂等键必须等于 P4 blob 的账本键、内部结算键必须等于虚拟账批次与转账键、
  价源日期必须等于信封交易日）——任何不符都在写入前拒绝；
- ``apply_owners``：把 ``RestoredOwners`` 一次性写入**新建**宿主（先全量校验已完成，
  此处只做赋值）。

为什么"不得从虚拟侧反推"：物理来源 lot 的成本与批次顺序只存在于伴随视图，虚拟侧
只有取得批次（``acquired_price``/``acquired_fee``），两者语义不同；由虚拟批次反推
物理成本会把"取得价"冒充"原始成本"，因此编码/解码都不做任何此类推导。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.core.accounting.engine_state import (
    EngineState,
    dumps_blob,
    loads_blob,
    restore_state,
    state_from_blob,
    state_to_blob,
)
from src.lazybull.v2.core.accounting.ledger import LEDGER_STATE_KEYS
from src.lazybull.v2.core.accounting.physical_lot_view import (
    PhysicalLotView,
    lot_order_fingerprint,
)
from src.lazybull.v2.core.accounting.reconciliation import IdentityResult
from src.lazybull.v2.core.accounting.state_consistency import assert_economic_consistency
from src.lazybull.v2.core.accounting.settlement_kernel import (
    FeeWaterfall,
    SettlementResult,
    ShareAllocation,
)
from src.lazybull.v2.core.accounting.state_envelope import (
    ENVELOPE_SCHEMA_VERSION,
    MEDIA_TYPE_CANONICAL_JSON,
    MEDIA_TYPE_PICKLE5,
    OWNER_P4_ENGINE_STATE,
    OWNER_PHYSICAL_LOT_VIEW,
    OWNER_SETTLEMENT_IDENTITY,
    OWNER_VIRTUAL_LEDGER,
    EnvelopeOwner,
    StateEnvelope,
    StateEnvelopeError,
    canonical_json_bytes,
    f64,
    make_owner,
    require_key_set,
    sha256_bytes,
    unf64,
    validate_marker,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    InternalTransfer,
    LotAllocation,
    VirtualLedger,
    VirtualLot,
    _SleeveBook,
)

#: 本模块唯一阶段标识（进入全部错误上下文）
STAGE = "P6.5"


class StateOwnerError(StateEnvelopeError):
    """所有者投影/还原输入非法（**零副作用**拒绝）。"""


def _error(message: str) -> Exception:
    """带阶段上下文的所有者拒绝。"""
    return StateOwnerError(f"阶段={STAGE} 状态所有者 {message}")


# ────────────────────────── 数值编码 ──────────────────────────


def _num(value: Any) -> Dict[str, Any]:
    """int/float 的位精确编码（int 走 ``$int``，float 走 ``$f64``）。"""
    if isinstance(value, bool):
        return {"$bool": bool(value)}
    if isinstance(value, int):
        return {"$int": int(value)}
    return f64(value)


def _unum(node: Any, field: str) -> Any:
    """``_num`` 的逆（形态非法即拒绝）。"""
    if not isinstance(node, Mapping) or len(node) != 1:
        raise _error(f"数值解码失败（{field}）：{node!r}")
    if "$int" in node:
        value = node["$int"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise _error(f"数值解码失败（{field}）：$int 必须为 int，实际 {value!r}")
        return value
    if "$bool" in node:
        if not isinstance(node["$bool"], bool):
            raise _error(f"数值解码失败（{field}）：$bool 必须为 bool")
        return node["$bool"]
    return unf64(node, field=field)


def _float_list(pairs: Sequence[Tuple[str, float]], field: str) -> List[List[Any]]:
    """``(键, 浮点)`` 序列 → 可规范化列表（浮点逐项位精确编码）。"""
    return [[str(key), f64(float(value))] for key, value in pairs]


def _pair_floats(node: Any, field: str) -> Tuple[Tuple[str, float], ...]:
    """可规范化列表 → ``((键, 浮点), ...)``（形态/顺序原样保留）。"""
    if not isinstance(node, list):
        raise _error(f"{field} 必须为列表，实际 {type(node).__name__}")
    items: List[Tuple[str, float]] = []
    for entry in node:
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
            raise _error(f"{field} 的条目必须为 [键, 浮点]，实际 {entry!r}")
        items.append((entry[0], unf64(entry[1], field=f"{field}[{entry[0]}]")))
    return tuple(items)


# ────────────────────────── ① P4 基础 blob ──────────────────────────


def build_p4_owner(engine: Any) -> EnvelopeOwner:
    """P4 演化状态 → blob 字节所有者（**只读**；不改键集/字段/校验语义）。"""
    blob = state_to_blob(engine_state_export(engine))
    return make_owner(OWNER_P4_ENGINE_STATE, dumps_blob(blob), MEDIA_TYPE_PICKLE5)


def engine_state_export(engine: Any) -> EngineState:
    """``engine_state.export_state`` 的薄包装（唯一导出入口，便于测试替换）。"""
    from src.lazybull.v2.core.accounting.engine_state import export_state

    return export_state(engine)


def decode_p4_owner(owner: EnvelopeOwner) -> EngineState:
    """P4 所有者 → ``EngineState``（版本/字段/子对象/类型全量校验，零写入）。"""
    if owner.media_type != MEDIA_TYPE_PICKLE5:
        raise _error(
            f"P4 所有者媒介类型必须为 {MEDIA_TYPE_PICKLE5!r}，实际 {owner.media_type!r}"
        )
    blob = loads_blob(owner.payload_bytes())
    return state_from_blob(blob)


# ────────────────────────── ② 物理 lot 伴随视图 ──────────────────────────


@dataclass(frozen=True, slots=True)
class PhysicalViewState:
    """物理来源 lot 伴随视图的完整状态（批次顺序 + 成本 + 事件水位 + 初始身份）。"""

    order: Tuple[Tuple[str, str], ...]
    lots: Tuple[Tuple[str, str, int, float], ...]
    events: Tuple[Tuple[str, str, str, int, str], ...]
    origin_fingerprint: str
    order_fingerprint: str


def build_physical_owner(view: PhysicalLotView) -> EnvelopeOwner:
    """伴随视图 → 所有者（成本/顺序/事件水位/初始身份全部显式落盘）。"""
    order = tuple(view.batch_order())
    contents = tuple(view.contents())
    if tuple((code, lot_id) for code, lot_id, _shares, _cost in contents) != order:
        raise _error("物理伴随视图的批次顺序与内容顺序不一致（内部状态破缺）")
    events = tuple(
        (label, str(fingerprint[0]), str(fingerprint[1]), int(fingerprint[2]), str(fingerprint[3]))
        for label, fingerprint in view.event_watermark()
    )
    payload = {
        "batch_order": [[code, lot_id] for code, lot_id in order],
        "lots": [
            {
                "ts_code": code,
                "source_lot_id": lot_id,
                "shares": int(shares),
                "cost": f64(cost),
            }
            for code, lot_id, shares, cost in contents
        ],
        "events": [
            {
                "event_id": label,
                "ts_code": code,
                "source_lot_id": lot_id,
                "shares": shares,
                "cost_repr": cost_repr,
            }
            for label, code, lot_id, shares, cost_repr in events
        ],
        "origin_fingerprint": view.origin_fingerprint(),
        "order_fingerprint": lot_order_fingerprint(order),
        "shares_by_stock": {
            code: int(shares) for code, shares in sorted(view.shares_by_stock().items())
        },
        "note": "I2 只按 (ts_code, 来源 lot_id) 汇总整数股数；成本与顺序不参与 I2",
    }
    return make_owner(
        OWNER_PHYSICAL_LOT_VIEW, canonical_json_bytes(payload), MEDIA_TYPE_CANONICAL_JSON
    )


def _decode_physical_order(payload: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """``batch_order`` 段解码（来源键唯一、顺序原样）。"""
    order: List[Tuple[str, str]] = []
    for entry in payload["batch_order"]:
        if not isinstance(entry, list) or len(entry) != 2:
            raise _error(f"物理所有者 batch_order 条目非法：{entry!r}")
        order.append((str(entry[0]), str(entry[1])))
    if len(set(order)) != len(order):
        raise _error(f"物理所有者 batch_order 含重复来源键：{order}")
    return order


def _decode_physical_lots(
    payload: Mapping[str, Any], order: Sequence[Tuple[str, str]]
) -> List[Tuple[str, str, int, float]]:
    """``lots`` 段解码（正整数股、位精确成本、顺序必须与 ``batch_order`` 一致）。"""
    lots: List[Tuple[str, str, int, float]] = []
    for entry in payload["lots"]:
        if not isinstance(entry, Mapping):
            raise _error(f"物理所有者 lots 条目非法：{entry!r}")
        require_key_set(
            entry, ("ts_code", "source_lot_id", "shares", "cost"), "物理所有者 lots 条目"
        )
        shares = entry["shares"]
        if isinstance(shares, bool) or not isinstance(shares, int) or shares <= 0:
            raise _error(
                f"物理所有者 lot 股数必须为正整数，实际 {shares!r}"
                f"（股票 {entry['ts_code']} 来源 {entry['source_lot_id']}）"
            )
        lots.append(
            (
                str(entry["ts_code"]),
                str(entry["source_lot_id"]),
                int(shares),
                unf64(entry["cost"], field=f"cost[{entry['source_lot_id']}]"),
            )
        )
    if [(code, lot_id) for code, lot_id, _shares, _cost in lots] != list(order):
        raise _error("物理所有者 batch_order 与 lots 顺序不一致（批次顺序不可重排）")
    return lots


def _decode_physical_events(payload: Mapping[str, Any]) -> List[Tuple[str, str, str, int, str]]:
    """``events`` 段解码（合成买入事件水位；事件身份唯一）。"""
    events: List[Tuple[str, str, str, int, str]] = []
    for entry in payload["events"]:
        if not isinstance(entry, Mapping):
            raise _error(f"物理所有者 events 条目非法：{entry!r}")
        require_key_set(
            entry,
            ("event_id", "ts_code", "source_lot_id", "shares", "cost_repr"),
            "物理所有者 events 条目",
        )
        shares = entry["shares"]
        if isinstance(shares, bool) or not isinstance(shares, int) or shares <= 0:
            raise _error(
                f"物理所有者事件 {entry['event_id']} 的 shares 必须为正整数，实际 {shares!r}"
                "（与 lots 股数同一口径：禁止 bool/float 静默截断）"
            )
        events.append(
            (
                str(entry["event_id"]),
                str(entry["ts_code"]),
                str(entry["source_lot_id"]),
                int(shares),
                str(entry["cost_repr"]),
            )
        )
    if len({item[0] for item in events}) != len(events):
        raise _error("物理所有者 events 含重复事件身份")
    return events


def _check_physical_totals(
    payload: Mapping[str, Any], lots: Sequence[Tuple[str, str, int, float]]
) -> None:
    """``shares_by_stock`` 与 lots 汇总必须一致（防止只改汇总不改批次）。"""
    totals: Dict[str, int] = {}
    for code, _lot_id, shares, _cost in lots:
        totals[code] = totals.get(code, 0) + shares
    expected = {code: totals[code] for code in sorted(totals)}
    if dict(payload["shares_by_stock"]) != expected:
        raise _error(
            f"物理所有者 shares_by_stock 与 lots 汇总不一致：登记 "
            f"{payload['shares_by_stock']} 实算 {expected}"
        )


def decode_physical_owner(owner: EnvelopeOwner) -> PhysicalViewState:
    """物理所有者 → 状态（键集/顺序/成本/事件/指纹逐项校验，零写入）。"""
    if owner.media_type != MEDIA_TYPE_CANONICAL_JSON:
        raise _error(f"物理所有者媒介类型非法：{owner.media_type!r}")
    import json

    try:
        payload = json.loads(owner.payload_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:  # pragma: no cover - 哈希已先校验
        raise _error(f"物理所有者文档不可解析：{error}") from error
    if not isinstance(payload, Mapping):
        raise _error("物理所有者文档必须为对象")
    require_key_set(
        payload,
        (
            "batch_order",
            "lots",
            "events",
            "origin_fingerprint",
            "order_fingerprint",
            "shares_by_stock",
            "note",
        ),
        "物理所有者",
    )
    order = _decode_physical_order(payload)
    lots = _decode_physical_lots(payload, order)
    events = _decode_physical_events(payload)
    if lot_order_fingerprint(tuple(order)) != payload["order_fingerprint"]:
        raise _error("物理所有者 order_fingerprint 与批次顺序不符（顺序被改写）")
    origin = payload["origin_fingerprint"]
    if not isinstance(origin, str) or len(origin) != 64:
        raise _error(f"物理所有者 origin_fingerprint 必须为 64 位十六进制，实际 {origin!r}")
    _check_physical_totals(payload, lots)
    return PhysicalViewState(
        order=tuple(order),
        lots=tuple(lots),
        events=tuple(events),
        origin_fingerprint=origin,
        order_fingerprint=str(payload["order_fingerprint"]),
    )


def restore_physical_view(state: PhysicalViewState) -> PhysicalLotView:
    """由解码状态重建伴随视图（**新建对象**；批次顺序/成本/事件/身份原样）。"""
    view = PhysicalLotView()
    view._lots = {(code, lot_id): shares for code, lot_id, shares, _cost in state.lots}
    view._costs = {(code, lot_id): cost for code, lot_id, _shares, cost in state.lots}
    view._order = [(code, lot_id) for code, lot_id in state.order]
    view._events = {
        label: (code, lot_id, shares, cost_repr)
        for label, code, lot_id, shares, cost_repr in state.events
    }
    view._origin = state.origin_fingerprint
    if view.contents() != state.lots:  # pragma: no cover - 由解码校验保证
        raise _error("物理伴随视图重建后内容与解码状态不一致")
    if view.event_watermark() != tuple(
        (label, (code, lot_id, shares, cost_repr))
        for label, code, lot_id, shares, cost_repr in state.events
    ):
        raise _error("物理伴随视图重建后事件水位与解码状态不一致")
    return view


# ────────────────────────── ③ 虚拟账 ──────────────────────────


@dataclass(frozen=True, slots=True)
class VirtualLedgerState:
    """虚拟账完整状态（各袖现金/双层 lot/费用/占用 + 结算身份与批次结果）。"""

    sleeve_order: Tuple[str, ...]
    books: Tuple[Tuple[str, float, float, Tuple[VirtualLot, ...]], ...]
    settlements: Tuple[Tuple[str, Tuple[Any, ...]], ...]
    batches: Tuple[Tuple[str, Tuple[Any, ...], SettlementResult], ...]
    last_settlement_id: Optional[str]
    last_settlement_source_id: Optional[str]
    poisoned: Optional[str]

    def settlement_keys(self) -> Tuple[str, ...]:
        """结算身份键序（直接转账 + 派生转账子键）。"""
        return tuple(key for key, _fingerprint in self.settlements)

    def batch_ids(self) -> Tuple[str, ...]:
        """结算批次身份键序。"""
        return tuple(batch_id for batch_id, _fingerprint, _result in self.batches)


def _encode_virtual_lot(lot: VirtualLot) -> Dict[str, Any]:
    """虚拟取得批次 → 可规范化对象（含费用与取得语义）。"""
    return {
        "virtual_lot_id": lot.virtual_lot_id,
        "ts_code": lot.ts_code,
        "source_lot_id": lot.source_lot_id,
        "shares": int(lot.shares),
        "acquired_date": str(lot.acquired_date),
        "acquired_price": f64(lot.acquired_price),
        "acquired_fee": f64(lot.acquired_fee),
    }


def _decode_virtual_lot(payload: Mapping[str, Any], sleeve_id: str) -> VirtualLot:
    """可规范化对象 → 虚拟取得批次（键集封闭）。"""
    require_key_set(
        payload,
        (
            "virtual_lot_id",
            "ts_code",
            "source_lot_id",
            "shares",
            "acquired_date",
            "acquired_price",
            "acquired_fee",
        ),
        f"袖 {sleeve_id} 虚拟批次",
    )
    acquired = str(payload["acquired_date"])
    if len(acquired) != 8 or not acquired.isdigit():
        raise _error(f"袖 {sleeve_id} 虚拟批次取得日必须为 YYYYMMDD，实际 {acquired!r}")
    return VirtualLot(
        virtual_lot_id=str(payload["virtual_lot_id"]),
        ts_code=str(payload["ts_code"]),
        source_lot_id=str(payload["source_lot_id"]),
        shares=int(payload["shares"]),
        acquired_date=TradeDate.from_str(acquired),
        acquired_price=unf64(payload["acquired_price"], field="acquired_price"),
        acquired_fee=unf64(payload["acquired_fee"], field="acquired_fee"),
    )


def _encode_fingerprint(fingerprint: Sequence[Any]) -> List[Any]:
    """幂等指纹 → JSON（只允许 str/int 与嵌套元组，其余即拒绝）。"""
    encoded: List[Any] = []
    for item in fingerprint:
        if isinstance(item, (list, tuple)):
            encoded.append(_encode_fingerprint(item))
        elif isinstance(item, bool) or not isinstance(item, (str, int)):
            raise _error(
                f"幂等指纹含非法元素 {item!r}（{type(item).__name__}）；"
                "指纹必须是 str/int/嵌套序列（浮点须由产出方用 repr 转字符串）"
            )
        else:
            encoded.append(item)
    return encoded


def _decode_fingerprint(payload: Any, field: str) -> Tuple[Any, ...]:
    """JSON → 幂等指纹（嵌套列表转回元组，保持与账本内指纹逐项相等）。"""
    if not isinstance(payload, list):
        raise _error(f"{field} 必须为列表，实际 {type(payload).__name__}")
    decoded: List[Any] = []
    for item in payload:
        if isinstance(item, list):
            decoded.append(_decode_fingerprint(item, field))
        elif isinstance(item, str) or (isinstance(item, int) and not isinstance(item, bool)):
            decoded.append(item)
        else:
            raise _error(f"{field} 含非法元素 {item!r}")
    return tuple(decoded)


def _encode_waterfall(waterfall: FeeWaterfall) -> Dict[str, Any]:
    """费用瀑布 → 可规范化对象（逐袖费用与终端残差位精确）。"""
    return {
        "order": list(waterfall.order),
        "intent_amounts": _float_list(waterfall.intent_amounts, "intent_amounts"),
        "allocated_amounts": _float_list(waterfall.allocated_amounts, "allocated_amounts"),
        "independent_costs": _float_list(waterfall.independent_costs, "independent_costs"),
        "provisional_fees": _float_list(waterfall.provisional_fees, "provisional_fees"),
        "terminal_sleeve": waterfall.terminal_sleeve,
        "residual_correction": f64(waterfall.residual_correction),
        "final_fees": _float_list(waterfall.final_fees, "final_fees"),
        "physical_fee": f64(waterfall.physical_fee),
        "proportional_branch": bool(waterfall.proportional_branch),
    }


def _decode_waterfall(payload: Mapping[str, Any]) -> FeeWaterfall:
    """可规范化对象 → 费用瀑布。"""
    require_key_set(
        payload,
        (
            "order",
            "intent_amounts",
            "allocated_amounts",
            "independent_costs",
            "provisional_fees",
            "terminal_sleeve",
            "residual_correction",
            "final_fees",
            "physical_fee",
            "proportional_branch",
        ),
        "费用瀑布",
    )
    return FeeWaterfall(
        order=tuple(str(item) for item in payload["order"]),
        intent_amounts=_pair_floats(payload["intent_amounts"], "intent_amounts"),
        allocated_amounts=_pair_floats(payload["allocated_amounts"], "allocated_amounts"),
        independent_costs=_pair_floats(payload["independent_costs"], "independent_costs"),
        provisional_fees=_pair_floats(payload["provisional_fees"], "provisional_fees"),
        terminal_sleeve=str(payload["terminal_sleeve"]),
        residual_correction=unf64(payload["residual_correction"], field="residual_correction"),
        final_fees=_pair_floats(payload["final_fees"], "final_fees"),
        physical_fee=unf64(payload["physical_fee"], field="physical_fee"),
        proportional_branch=bool(payload["proportional_branch"]),
    )


def _encode_transfer(transfer: InternalTransfer) -> Dict[str, Any]:
    """内部转账 → 可规范化对象（分配批次顺序原样）。"""
    return {
        "settlement_id": transfer.settlement_id,
        "trade_date": str(transfer.trade_date),
        "ts_code": transfer.ts_code,
        "from_sleeve": transfer.from_sleeve,
        "to_sleeve": transfer.to_sleeve,
        "price": f64(transfer.price),
        "allocations": [[item.virtual_lot_id, int(item.shares)] for item in transfer.allocations],
    }


def _decode_transfer(payload: Mapping[str, Any]) -> InternalTransfer:
    """可规范化对象 → 内部转账。"""
    require_key_set(
        payload,
        (
            "settlement_id",
            "trade_date",
            "ts_code",
            "from_sleeve",
            "to_sleeve",
            "price",
            "allocations",
        ),
        "内部转账",
    )
    trade_date = str(payload["trade_date"])
    if len(trade_date) != 8 or not trade_date.isdigit():
        raise _error(f"内部转账交易日必须为 YYYYMMDD，实际 {trade_date!r}")
    allocations = tuple(
        LotAllocation(str(item[0]), int(item[1])) for item in payload["allocations"]
    )
    return InternalTransfer(
        settlement_id=str(payload["settlement_id"]),
        trade_date=TradeDate.from_str(trade_date),
        ts_code=str(payload["ts_code"]),
        from_sleeve=str(payload["from_sleeve"]),
        to_sleeve=str(payload["to_sleeve"]),
        price=unf64(payload["price"], field="transfer.price"),
        allocations=allocations,
    )


def _encode_identity_result(result: IdentityResult) -> Dict[str, Any]:
    """恒等式成功证据 → 可规范化对象（I2 逐来源明细原样）。"""
    return {
        "identity": result.identity,
        "left": _num(result.left),
        "right": _num(result.right),
        "diff": _num(result.diff),
        "tolerance": _num(result.tolerance),
        "unit": result.unit,
        "sources": [[code, lot_id, int(shares)] for code, lot_id, shares in result.sources],
    }


def _decode_identity_result(payload: Mapping[str, Any]) -> IdentityResult:
    """可规范化对象 → 恒等式成功证据。"""
    require_key_set(
        payload,
        ("identity", "left", "right", "diff", "tolerance", "unit", "sources"),
        "恒等式结果",
    )
    return IdentityResult(
        identity=str(payload["identity"]),
        left=_unum(payload["left"], "identity.left"),
        right=_unum(payload["right"], "identity.right"),
        diff=_unum(payload["diff"], "identity.diff"),
        tolerance=_unum(payload["tolerance"], "identity.tolerance"),
        unit=str(payload["unit"]),
        sources=tuple(
            (str(item[0]), str(item[1]), int(item[2])) for item in payload["sources"]
        ),
    )


def _encode_settlement_result(result: SettlementResult) -> Dict[str, Any]:
    """结算结果 → 可规范化对象（分摊/瀑布/转账/消耗批次/作废意图全量）。"""
    return {
        "batch_id": result.batch_id,
        "applied": bool(result.applied),
        "share_allocations": [
            [item.sleeve_id, int(item.intent_shares), int(item.allocated_shares)]
            for item in result.share_allocations
        ],
        "fee_waterfalls": [_encode_waterfall(item) for item in result.fee_waterfalls],
        "transfers": [_encode_transfer(item) for item in result.transfers],
        "consumed_lots": [
            [str(sid), str(lot_id), int(shares)] for sid, lot_id, shares in result.consumed_lots
        ],
        "cash_total_after": f64(result.cash_total_after),
        "source_shares_after": [
            [str(code), str(lot_id), int(shares)]
            for code, lot_id, shares in result.source_shares_after
        ],
        "zero_fill": bool(result.zero_fill),
        "voided_intents": [
            [str(sid), str(code), int(shares)] for sid, code, shares in result.voided_intents
        ],
        "identity_results": [_encode_identity_result(item) for item in result.identity_results],
        "positions_reconciled": bool(result.positions_reconciled),
    }


def _decode_settlement_result(payload: Mapping[str, Any]) -> SettlementResult:
    """可规范化对象 → 结算结果（用于恢复后同键同内容重放返回 applied=False）。"""
    require_key_set(
        payload,
        (
            "batch_id",
            "applied",
            "share_allocations",
            "fee_waterfalls",
            "transfers",
            "consumed_lots",
            "cash_total_after",
            "source_shares_after",
            "zero_fill",
            "voided_intents",
            "identity_results",
            "positions_reconciled",
        ),
        "结算结果",
    )
    return SettlementResult(
        batch_id=str(payload["batch_id"]),
        applied=bool(payload["applied"]),
        share_allocations=tuple(
            ShareAllocation(str(item[0]), int(item[1]), int(item[2]))
            for item in payload["share_allocations"]
        ),
        fee_waterfalls=tuple(_decode_waterfall(item) for item in payload["fee_waterfalls"]),
        transfers=tuple(_decode_transfer(item) for item in payload["transfers"]),
        consumed_lots=tuple(
            (str(item[0]), str(item[1]), int(item[2])) for item in payload["consumed_lots"]
        ),
        cash_total_after=unf64(payload["cash_total_after"], field="cash_total_after"),
        source_shares_after=tuple(
            (str(item[0]), str(item[1]), int(item[2])) for item in payload["source_shares_after"]
        ),
        zero_fill=bool(payload["zero_fill"]),
        voided_intents=tuple(
            (str(item[0]), str(item[1]), int(item[2])) for item in payload["voided_intents"]
        ),
        identity_results=tuple(
            _decode_identity_result(item) for item in payload["identity_results"]
        ),
        positions_reconciled=bool(payload["positions_reconciled"]),
    )


def build_virtual_owner(ledger: VirtualLedger) -> EnvelopeOwner:
    """虚拟账 → 所有者（各袖现金/双层 lot/费用/占用 + 结算身份与批次结果）。"""
    books = []
    sleeve_order: List[str] = []
    for sleeve_id in ledger.sleeve_ids():
        sleeve_order.append(str(sleeve_id))
        book = ledger._books[sleeve_id]
        books.append(
            {
                "sleeve_id": str(sleeve_id),
                "cash": f64(book.cash),
                "borrowed_credit": f64(book.borrowed_credit),
                "lots": [_encode_virtual_lot(lot) for lot in book.lots],
            }
        )
    settlements = [
        [str(key), _encode_fingerprint(fingerprint)]
        for key, fingerprint in ledger._settlements.items()
    ]
    batches = [
        [
            str(batch_id),
            _encode_fingerprint(fingerprint),
            _encode_settlement_result(result),
        ]
        for batch_id, (fingerprint, result) in ledger._batches.items()
    ]
    payload = {
        "sleeve_order": sleeve_order,
        "books": books,
        "settlements": settlements,
        "batches": batches,
        "last_settlement_id": ledger._last_settlement_id,
        "last_settlement_source_id": ledger._last_settlement_source_id,
        "poisoned": ledger._poisoned,
        "note": "虚拟账为运行时概念；I2/I3 只消费 VirtualSleeveState，本项不写生产账本",
    }
    return make_owner(
        OWNER_VIRTUAL_LEDGER, canonical_json_bytes(payload), MEDIA_TYPE_CANONICAL_JSON
    )


def _decode_books(payload: Mapping[str, Any]) -> Tuple[Tuple[str, ...], Tuple[Any, ...]]:
    """虚拟账 books 段解码（袖序 + 各袖账）。"""
    if not isinstance(payload["sleeve_order"], list) or not isinstance(payload["books"], list):
        raise _error("虚拟账 sleeve_order/books 必须为列表")
    order = tuple(str(item) for item in payload["sleeve_order"])
    if list(order) != [str(item.get("sleeve_id")) for item in payload["books"]]:
        raise _error("虚拟账 sleeve_order 与 books 顺序不一致（袖序不可重排）")
    books: List[Tuple[str, float, float, Tuple[VirtualLot, ...]]] = []
    for entry in payload["books"]:
        if not isinstance(entry, Mapping):
            raise _error(f"虚拟账 books 条目非法：{entry!r}")
        require_key_set(entry, ("sleeve_id", "cash", "borrowed_credit", "lots"), "虚拟账 books 条目")
        lots = tuple(_decode_virtual_lot(item, str(entry["sleeve_id"])) for item in entry["lots"])
        if len({lot.virtual_lot_id for lot in lots}) != len(lots):
            raise _error(
                f"袖 {entry['sleeve_id']} 的虚拟取得批次身份重复（(sleeve_id, lot_id) 必须唯一）"
            )
        books.append(
            (
                str(entry["sleeve_id"]),
                unf64(entry["cash"], field="cash"),
                unf64(entry["borrowed_credit"], field="borrowed_credit"),
                lots,
            )
        )
    return order, tuple(books)


def _decode_settlements(payload: Mapping[str, Any]) -> List[Tuple[str, Tuple[Any, ...]]]:
    """``settlements`` 段解码（结算身份唯一、指纹逐项还原）。"""
    settlements: List[Tuple[str, Tuple[Any, ...]]] = []
    for entry in payload["settlements"]:
        if not isinstance(entry, list) or len(entry) != 2:
            raise _error(f"虚拟账 settlements 条目非法：{entry!r}")
        settlements.append(
            (str(entry[0]), _decode_fingerprint(entry[1], "settlement.fingerprint"))
        )
    if len({key for key, _fingerprint in settlements}) != len(settlements):
        raise _error("虚拟账 settlements 含重复结算身份")
    return settlements


def _decode_batches(
    payload: Mapping[str, Any],
) -> List[Tuple[str, Tuple[Any, ...], SettlementResult]]:
    """``batches`` 段解码（批次键与结果 batch_id 一致、身份唯一）。"""
    batches: List[Tuple[str, Tuple[Any, ...], SettlementResult]] = []
    for entry in payload["batches"]:
        if not isinstance(entry, list) or len(entry) != 3:
            raise _error(f"虚拟账 batches 条目非法：{entry!r}")
        result = _decode_settlement_result(entry[2])
        if result.batch_id != str(entry[0]):
            raise _error(
                f"虚拟账批次键 {entry[0]!r} 与结果 batch_id {result.batch_id!r} 不一致"
            )
        batches.append((str(entry[0]), _decode_fingerprint(entry[1], "batch.fingerprint"), result))
    if len({batch_id for batch_id, _fingerprint, _result in batches}) != len(batches):
        raise _error("虚拟账 batches 含重复批次身份")
    return batches


def decode_virtual_owner(owner: EnvelopeOwner) -> VirtualLedgerState:
    """虚拟所有者 → 状态（键集/袖序/批次身份/指纹逐项校验，零写入）。"""
    if owner.media_type != MEDIA_TYPE_CANONICAL_JSON:
        raise _error(f"虚拟账所有者媒介类型非法：{owner.media_type!r}")
    import json

    try:
        payload = json.loads(owner.payload_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:  # pragma: no cover - 哈希已先校验
        raise _error(f"虚拟账所有者文档不可解析：{error}") from error
    if not isinstance(payload, Mapping):
        raise _error("虚拟账所有者文档必须为对象")
    require_key_set(
        payload,
        (
            "sleeve_order",
            "books",
            "settlements",
            "batches",
            "last_settlement_id",
            "last_settlement_source_id",
            "poisoned",
            "note",
        ),
        "虚拟账所有者",
    )
    order, books = _decode_books(payload)
    settlements = _decode_settlements(payload)
    batches = _decode_batches(payload)
    overlap = {key for key, _fingerprint in settlements} & {
        batch_id for batch_id, _fingerprint, _result in batches
    }
    if overlap:
        raise _error(f"虚拟账结算身份与批次身份重叠：{sorted(overlap)}")
    return VirtualLedgerState(
        sleeve_order=order,
        books=books,
        settlements=tuple(settlements),
        batches=tuple(batches),
        last_settlement_id=payload["last_settlement_id"],
        last_settlement_source_id=payload["last_settlement_source_id"],
        poisoned=payload["poisoned"],
    )


def restore_virtual_ledger(ledger: VirtualLedger, state: VirtualLedgerState) -> None:
    """把解码状态写入**新建**账本（**全量校验已在前置完成**；此处只赋值）。"""
    books: Dict[str, _SleeveBook] = {}
    for sleeve_id, cash, borrowed_credit, lots in state.books:
        sleeve = _SleeveBook(cash=cash, lots=lots, borrowed_credit=borrowed_credit)
        if sleeve.cash != cash:  # pragma: no cover - 防御性：float 归一后必须等值
            raise _error(f"袖 {sleeve_id} 现金重建后与解码值不一致")
        books[sleeve_id] = sleeve
    ledger._books = books
    ledger._settlements = {key: fingerprint for key, fingerprint in state.settlements}
    ledger._batches = {
        batch_id: (fingerprint, result) for batch_id, fingerprint, result in state.batches
    }
    ledger._last_settlement_id = state.last_settlement_id
    ledger._last_settlement_source_id = state.last_settlement_source_id
    ledger._poisoned = state.poisoned


# ────────────────────────── ④ 结算与估值身份 ──────────────────────────


@dataclass(frozen=True, slots=True)
class SettlementIdentityState:
    """结算与估值身份（幂等键、事件序号、价源标识与日期、链级批次指纹）。"""

    physical_fill_keys: Tuple[str, ...]
    physical_last_fill_date: Optional[str]
    internal_settlement_keys: Tuple[str, ...]
    batch_ids: Tuple[str, ...]
    last_event_sequence: int
    last_settlement_id: Optional[str]
    last_settlement_source_id: Optional[str]
    price_source_id: str
    price_source_as_of: str
    chain_batch_fingerprints: Tuple[Tuple[str, str], ...]

    def watermark(self) -> Dict[str, Any]:
        """水位投影（跨代单调性校验与证据登记共用）。"""
        return {
            "applied_keys": self.physical_fill_keys,
            "settlement_keys": self.internal_settlement_keys,
            "batch_ids": self.batch_ids,
            "last_fill_date": self.physical_last_fill_date,
            "last_event_sequence": self.last_event_sequence,
        }


def _chain_fingerprint_pairs(host: Any) -> List[List[str]]:
    """链级批次指纹（恢复代已登记 ∪ 本代 ``_chain_results``；键序稳定、跨代只增不减）。

    T5-A-R1-02：信封恢复不重建 ``OrderChainResult``（计划/成交/结算事实未持久化），
    但**必须**保留其请求指纹——否则恢复代的链级身份会逐代收缩，且"同键同内容 vs
    同键异内容"在恢复路径上不可区分。
    """
    restored = dict(getattr(host, "restored_chain_fingerprints", {}) or {})
    current = {
        str(batch_id): str(result.request_fingerprint)
        for batch_id, result in getattr(host, "_chain_results", {}).items()
    }
    for batch_id, fingerprint in current.items():
        recorded = restored.get(batch_id)
        if recorded is not None and recorded != fingerprint:
            raise _error(
                f"链级批次指纹冲突：恢复代登记 {batch_id} 的指纹 {recorded}，本代链结果为 "
                f"{fingerprint}；同一批次键不得承载两套请求"
            )
    merged = dict(restored)
    merged.update(current)
    return [[str(batch_id), str(fingerprint)] for batch_id, fingerprint in merged.items()]


def build_identity_owner(
    *,
    host: Any,
    ledger: VirtualLedger,
    price_source_id: str,
    price_source_as_of: str,
) -> EnvelopeOwner:
    """结算/估值身份 → 所有者（物理/内部幂等键 + 事件序号 + 价源 + 链级指纹）。"""
    physical = host.ledger.export_state()
    last_fill = physical["last_fill_date"]
    payload = {
        "physical_fill_keys": [str(key) for key in physical["applied_keys"]],
        "physical_last_fill_date": None if last_fill is None else str(last_fill),
        "internal_settlement_keys": [str(key) for key in ledger.settlement_ids()],
        "batch_ids": [str(key) for key in ledger.batch_ids()],
        "last_event_sequence": len(ledger.settlement_ids()) + len(ledger.batch_ids()),
        "last_settlement_id": ledger.last_settlement_id,
        "last_settlement_source_id": ledger.last_settlement_source_id,
        "price_source_id": str(price_source_id),
        "price_source_as_of": str(price_source_as_of),
        "chain_batch_fingerprints": _chain_fingerprint_pairs(host),
        "note": (
            "物理幂等键必须等于 P4 blob 的账本键；内部结算键必须等于虚拟账批次与转账键；"
            "价源日期必须等于信封交易日（同代一致性在解码时逐项校验）；"
            "链级批次指纹 = 恢复代已登记指纹 ∪ 本代链结果（跨代只增不减）"
        ),
    }
    return make_owner(
        OWNER_SETTLEMENT_IDENTITY, canonical_json_bytes(payload), MEDIA_TYPE_CANONICAL_JSON
    )


def decode_identity_owner(owner: EnvelopeOwner) -> SettlementIdentityState:
    """结算/估值身份 → 状态（键集/类型/序号逐项校验，零写入）。"""
    if owner.media_type != MEDIA_TYPE_CANONICAL_JSON:
        raise _error(f"结算身份所有者媒介类型非法：{owner.media_type!r}")
    import json

    try:
        payload = json.loads(owner.payload_bytes().decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:  # pragma: no cover - 哈希已先校验
        raise _error(f"结算身份所有者文档不可解析：{error}") from error
    if not isinstance(payload, Mapping):
        raise _error("结算身份所有者文档必须为对象")
    require_key_set(
        payload,
        (
            "physical_fill_keys",
            "physical_last_fill_date",
            "internal_settlement_keys",
            "batch_ids",
            "last_event_sequence",
            "last_settlement_id",
            "last_settlement_source_id",
            "price_source_id",
            "price_source_as_of",
            "chain_batch_fingerprints",
            "note",
        ),
        "结算身份所有者",
    )
    sequence = payload["last_event_sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise _error(f"结算事件序号必须为非负 int，实际 {sequence!r}")
    as_of = str(payload["price_source_as_of"])
    if len(as_of) != 8 or not as_of.isdigit():
        raise _error(f"价源日期必须为 YYYYMMDD，实际 {as_of!r}")
    if not str(payload["price_source_id"]).strip():
        raise _error("价源标识不得为空（禁止在未绑定价源时落同代信封）")
    chain = payload["chain_batch_fingerprints"]
    if not isinstance(chain, list):
        raise _error("链级批次指纹必须为列表")
    pairs: List[Tuple[str, str]] = []
    for entry in chain:
        if not isinstance(entry, list) or len(entry) != 2:
            raise _error(f"链级批次指纹条目非法：{entry!r}")
        pairs.append((str(entry[0]), str(entry[1])))
    if len({batch_id for batch_id, _fingerprint in pairs}) != len(pairs):
        raise _error("链级批次指纹含重复批次身份")
    return SettlementIdentityState(
        physical_fill_keys=tuple(str(item) for item in payload["physical_fill_keys"]),
        physical_last_fill_date=payload["physical_last_fill_date"],
        internal_settlement_keys=tuple(str(item) for item in payload["internal_settlement_keys"]),
        batch_ids=tuple(str(item) for item in payload["batch_ids"]),
        last_event_sequence=int(sequence),
        last_settlement_id=payload["last_settlement_id"],
        last_settlement_source_id=payload["last_settlement_source_id"],
        price_source_id=str(payload["price_source_id"]),
        price_source_as_of=as_of,
        chain_batch_fingerprints=tuple(pairs),
    )


# ────────────────────────── 五项联合解码/写入 ──────────────────────────


@dataclass(frozen=True, slots=True)
class RestoredOwners:
    """信封五项的**已解码**投影（全部校验通过后才会被写入宿主）。"""

    p4_state: EngineState
    physical: PhysicalViewState
    virtual: VirtualLedgerState
    identity: SettlementIdentityState
    marker: Mapping[str, Any]

    def watermark(self) -> Dict[str, Any]:
        """水位投影（同代起点；恢复后必须与信封登记值逐项相等）。"""
        return self.identity.watermark()


def _assert_identity_consistency(
    *,
    envelope: StateEnvelope,
    p4_state: EngineState,
    physical: PhysicalViewState,
    virtual: VirtualLedgerState,
    identity: SettlementIdentityState,
) -> None:
    """跨项同代一致性（五项必须来自同一提交点；任一不符即拒绝）。"""
    label = f"信封 {envelope.generation_token} 日期={envelope.trade_date}"
    ledger_state = p4_state.ledger
    if set(ledger_state.keys()) != set(LEDGER_STATE_KEYS):
        raise _error(f"{label}：P4 blob 的账本投影键集不符 {sorted(ledger_state.keys())}")
    if tuple(ledger_state["applied_keys"]) != identity.physical_fill_keys:
        raise _error(
            f"{label}：结算身份登记 {len(identity.physical_fill_keys)} 个物理幂等键 != "
            f"P4 blob 账本 {len(ledger_state['applied_keys'])} 个（同代一致性破缺）"
        )
    p4_last_fill = ledger_state["last_fill_date"]
    p4_last_fill_text = None if p4_last_fill is None else str(p4_last_fill)
    if p4_last_fill_text != identity.physical_last_fill_date:
        raise _error(
            f"{label}：结算身份的最后成交日 {identity.physical_last_fill_date!r} != "
            f"P4 blob 账本 {p4_last_fill_text!r}"
        )
    if tuple(identity.internal_settlement_keys) != virtual.settlement_keys():
        raise _error(
            f"{label}：结算身份的内部结算键 {len(identity.internal_settlement_keys)} 个 != "
            f"虚拟账 {len(virtual.settlement_keys())} 个（同代一致性破缺）"
        )
    if tuple(identity.batch_ids) != virtual.batch_ids():
        raise _error(
            f"{label}：结算身份的批次身份 {len(identity.batch_ids)} 个 != "
            f"虚拟账 {len(virtual.batch_ids())} 个"
        )
    if identity.last_settlement_id != virtual.last_settlement_id:
        raise _error(
            f"{label}：最后结算身份 {identity.last_settlement_id!r} != "
            f"虚拟账 {virtual.last_settlement_id!r}"
        )
    if identity.last_settlement_source_id != virtual.last_settlement_source_id:
        raise _error(
            f"{label}：最后结算价源 {identity.last_settlement_source_id!r} != "
            f"虚拟账 {virtual.last_settlement_source_id!r}"
        )
    expected_sequence = len(virtual.settlement_keys()) + len(virtual.batch_ids())
    if identity.last_event_sequence != expected_sequence:
        raise _error(
            f"{label}：结算事件序号 {identity.last_event_sequence} != "
            f"结算身份总数 {expected_sequence}（序号必须由身份集合唯一确定）"
        )
    if identity.price_source_as_of != envelope.trade_date:
        raise _error(
            f"{label}：价源日期 {identity.price_source_as_of!r} != 信封交易日 "
            f"{envelope.trade_date!r}（禁止跨日错价）"
        )
    if virtual.poisoned is not None:
        raise _error(
            f"{label}：信封内的虚拟账处于终态（{virtual.poisoned}）；"
            "破缺态不得写入新信封，也不能作为恢复起点"
        )
    assert_economic_consistency(
        label=label,
        trade_date=envelope.trade_date,
        positions=p4_state.positions,
        physical_lots=physical.lots,
        virtual_books=virtual.books,
        physical_cash=p4_state.current_capital,
        portfolio_values=p4_state.portfolio_values,
    )


def decode_owners(envelope: StateEnvelope) -> RestoredOwners:
    """五项联合解码 + 全量校验（**纯读**：不改宿主、不落盘）。"""
    p4_state = decode_p4_owner(envelope.owner(OWNER_P4_ENGINE_STATE))
    physical = decode_physical_owner(envelope.owner(OWNER_PHYSICAL_LOT_VIEW))
    virtual = decode_virtual_owner(envelope.owner(OWNER_VIRTUAL_LEDGER))
    identity = decode_identity_owner(envelope.owner(OWNER_SETTLEMENT_IDENTITY))
    marker = validate_marker(envelope)
    _assert_identity_consistency(
        envelope=envelope,
        p4_state=p4_state,
        physical=physical,
        virtual=virtual,
        identity=identity,
    )
    return RestoredOwners(
        p4_state=p4_state,
        physical=physical,
        virtual=virtual,
        identity=identity,
        marker=marker,
    )


def assert_fresh_instance(host: Any) -> None:
    """恢复目标必须是**新建实例**（禁止同进程续跑旧实例；**零写入**）。

    判据：无成交、无持仓、P4 账本无幂等键、虚拟账未开户/无结算身份、物理伴随视图
    为空。任一不满足即拒绝——恢复只能"由信封新建引擎"，不得把信封状态叠加到已经
    演化的实例上（否则两份水位混用，幂等与守恒同时失去意义）。
    """
    problems: List[str] = []
    if list(getattr(host, "trades", []) or []):
        problems.append(f"trades 非空（{len(host.trades)} 条）")
    if dict(getattr(host, "positions", {}) or {}):
        problems.append(f"positions 非空（{sorted(host.positions)}）")
    ledger_state = host.ledger.export_state()
    if list(ledger_state["applied_keys"]):
        problems.append(f"物理账本已有幂等键（{len(ledger_state['applied_keys'])} 个）")
    if ledger_state["last_fill_date"] is not None:
        problems.append(f"物理账本最后成交日非空（{ledger_state['last_fill_date']}）")
    ledger = getattr(host, "virtual_ledger", None)
    if ledger is None:
        problems.append("缺少虚拟账本")
    else:
        if ledger.sleeve_ids():
            problems.append(f"虚拟账已开户（{list(ledger.sleeve_ids())}）")
        if ledger.settlement_ids() or ledger.batch_ids():
            problems.append("虚拟账已有结算身份")
    view = getattr(host, "physical_lot_view", None)
    if view is None:
        problems.append("缺少物理 lot 伴随视图")
    elif view.batch_order():
        problems.append(f"物理伴随视图非空（{len(view.batch_order())} 个来源批次）")
    if problems:
        raise _error(
            "恢复目标不是新建实例（禁止同进程续跑旧实例）："
            + "；".join(problems)
            + "；恢复必须由信封新建引擎后重放"
        )


def apply_owners(host: Any, restored: RestoredOwners) -> None:
    """把五项写入**新建**宿主（前置校验已全部完成；本函数只做统一赋值）。

    顺序固定：P4 状态（唯一经 ``restore_state`` 的校验-写入入口）→ 物理伴随视图
    → 虚拟账 → 链级身份投影。任一环节抛异常都意味着前置校验被绕过（实现缺陷），
    此时宿主已被部分写入，调用方必须丢弃该实例并从同一信封重新新建。

    链级身份（``OrderChainResult`` 含计划/成交/结算事实）**不重建**：恢复后只登记
    其指纹投影（``restored_chain_fingerprints``）；同一 ``batch_id`` 的再次提交由
    虚拟账的批次身份拒绝（"链级身份不得复用"），不会因缺少内存结果而被静默重放。
    """
    assert_fresh_instance(host)
    restore_state(host, restored.p4_state)
    host.physical_lot_view = restore_physical_view(restored.physical)
    restore_virtual_ledger(host.virtual_ledger, restored.virtual)
    chain_results = getattr(host, "_chain_results", None)
    if chain_results is None:
        raise _error("合成宿主缺少链结果容器（_chain_results）；无法恢复链级身份")
    chain_results.clear()
    host.restored_chain_fingerprints = MappingProxyType(
        {
            batch_id: fingerprint
            for batch_id, fingerprint in restored.identity.chain_batch_fingerprints
        }
    )


def envelope_schema_version() -> int:
    """信封 schema 版本（证据层登记用；与常量同源）。"""
    return ENVELOPE_SCHEMA_VERSION


def owner_sha256(owner: EnvelopeOwner) -> str:
    """所有者字节摘要（等于登记值；供证据层重算复核）。"""
    return sha256_bytes(owner.payload_bytes())


__all__ = (
    "PhysicalViewState",
    "RestoredOwners",
    "SettlementIdentityState",
    "StateOwnerError",
    "VirtualLedgerState",
    "apply_owners",
    "build_identity_owner",
    "build_p4_owner",
    "build_physical_owner",
    "build_virtual_owner",
    "decode_identity_owner",
    "decode_owners",
    "decode_p4_owner",
    "decode_physical_owner",
    "decode_virtual_owner",
    "envelope_schema_version",
    "owner_sha256",
    "restore_physical_view",
    "restore_virtual_ledger",
)
