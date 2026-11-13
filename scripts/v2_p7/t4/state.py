# -*- coding: utf-8 -*-
"""P7 T4 专用状态信封（物理/虚拟 lot、cash/credit、残差、借款、召回/死局、幂等）。

契约依据
========
- ``docs/contracts/netting_freeze.md``§3（lot 批次模型）/§4（I1~I3）/
  P7 注记（借用态恒等式、有符号 I3）；
- 任务书 §4.4：**P7 专用**信封，**不改** P4 ``EngineState``；幂等键与 generation
  **同代提交**，重放/重启幂等。

设计要点
========
1. **物理侧不是虚拟侧的派生视图**：``PhysicalBook`` 由物理成交/卖出独立落账，
   恒等式校验时再与虚拟侧逐 ``(ts_code, 物理来源 lot_id)`` 汇总比对（I2 有真实内容）；
2. **内部转账保留物理来源谱系**：买方新建虚拟批次时复用卖方的 ``source_lot_id``，
   物理账本不变（F25 / Q14=A）；
3. **残差是袖级**（``ResidualEntry``）：只承载**未成交**部分，已抵消/已成交流程
   永不回流，因此"次日重新净额"天然**不重复抵消**；
4. **借款 lot / 召回 / 死局字段在本 task 恒为空**，但结构、校验与证据位置已就位，
   T5 只做填充，不改信封形状（``reserved_for_t5``）。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: 侧别
SIDES: Tuple[str, ...] = ("buy", "sell")

#: 残差来源标签（意图出处）
ORIGIN_SIGNAL_BUY = "signal_buy"
ORIGIN_EXPIRY_SELL = "expiry_sell"
ORIGIN_CARRY = "carry"
ORIGINS: Tuple[str, ...] = (ORIGIN_SIGNAL_BUY, ORIGIN_EXPIRY_SELL, ORIGIN_CARRY)

#: 借款 lot 状态（T4 恒为空；T5 填充）
BORROW_STATES: Tuple[str, ...] = ("outstanding", "recalled", "returned", "deadlocked")

#: 完整结算后信封的**版本化** schema（C4-04：续跑只认这一版载荷）
ENVELOPE_SCHEMA = "p7_t4_state/v2"


class StateError(RuntimeError):
    """状态信封前置条件不满足（身份/代际/守恒），一律 fail-fast。"""


def _require_finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise StateError(f"{label} 必须为有限数：{value!r}")
    return number


def _require_shares(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StateError(f"{label} 必须为整数股数：{value!r}")
    if value <= 0:
        raise StateError(f"{label} 必须为正整数：{value!r}")
    return value


@dataclass(frozen=True)
class P7Lot:
    """虚拟取得批次（物理来源谱系 + 虚拟取得信息 + 到期义务）。"""

    virtual_lot_id: str
    ts_code: str
    source_lot_id: str
    shares: int
    acquired_date: str
    acquired_price: float
    acquired_fee: float
    buy_date: str
    due_date: str
    signal_date: str
    origin_model_version: int

    def __post_init__(self) -> None:
        if not str(self.virtual_lot_id).strip():
            raise StateError("虚拟取得 lot_id 不得为空")
        if not str(self.ts_code).strip() or not str(self.source_lot_id).strip():
            raise StateError(f"lot {self.virtual_lot_id} 的 ts_code/source_lot_id 不得为空")
        _require_shares(self.shares, f"lot {self.virtual_lot_id} shares")
        price = _require_finite(self.acquired_price, f"lot {self.virtual_lot_id} acquired_price")
        if price <= 0.0:
            raise StateError(f"lot {self.virtual_lot_id} 取得价必须为正：{price!r}")
        fee = _require_finite(self.acquired_fee, f"lot {self.virtual_lot_id} acquired_fee")
        if fee < 0.0:
            raise StateError(f"lot {self.virtual_lot_id} 取得费用必须非负：{fee!r}")

    @property
    def acquired_cost(self) -> float:
        return self.shares * self.acquired_price + self.acquired_fee


@dataclass(frozen=True)
class PhysicalLot:
    """物理来源批次（I2 右值；由物理成交独立产生，不由虚拟侧派生）。"""

    source_lot_id: str
    ts_code: str
    shares: int
    buy_date: str
    buy_price: float
    buy_fee: float

    def __post_init__(self) -> None:
        if not str(self.source_lot_id).strip():
            raise StateError("物理来源 lot_id 不得为空")
        _require_shares(self.shares, f"物理 lot {self.source_lot_id} shares")
        price = _require_finite(self.buy_price, f"物理 lot {self.source_lot_id} buy_price")
        if price <= 0.0:
            raise StateError(f"物理 lot {self.source_lot_id} 买入价必须为正：{price!r}")


@dataclass(frozen=True)
class ResidualEntry:
    """袖级残差（只承载未成交部分；携带原始意图身份用于不重复抵消核对）。"""

    intent_key: str
    sleeve_id: str
    ts_code: str
    side: str
    shares: int
    origin: str
    reason: str
    first_blocked_date: str
    days_carried: int
    signal_date: str
    due_date: str
    #: 原始信号所属模型版本（C4-05：顺延不得丢模型血缘）
    model_version: int = 0

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise StateError(f"残差 {self.intent_key} 方向必须为 buy/sell：{self.side!r}")
        if self.origin not in ORIGINS:
            raise StateError(f"残差 {self.intent_key} 来源标签未知：{self.origin!r}")
        _require_shares(self.shares, f"残差 {self.intent_key} shares")

    def next_day(self, reason: str) -> "ResidualEntry":
        """顺延到下一交易日（``days_carried`` 递增，``reason`` 更新为最近阻断原因）。"""
        return replace(self, reason=str(reason), days_carried=self.days_carried + 1)


@dataclass(frozen=True)
class BorrowLot:
    """借款 lot（T4 恒为空；字段与校验先就位，T5 填充）。"""

    borrow_id: str
    lender: str
    borrower: str
    amount: float
    grant_date: str
    due_date: str
    returned_amount: float = 0.0
    state: str = "outstanding"

    def __post_init__(self) -> None:
        if self.state not in BORROW_STATES:
            raise StateError(f"借款 lot {self.borrow_id} 状态未知：{self.state!r}")
        if self.lender == self.borrower:
            raise StateError(f"借款 lot {self.borrow_id} 出借/借入袖相同：{self.lender}")
        amount = _require_finite(self.amount, f"借款 lot {self.borrow_id} amount")
        if amount <= 0.0:
            raise StateError(f"借款 lot {self.borrow_id} 金额必须为正：{amount!r}")


@dataclass(frozen=True)
class SleeveState:
    """单袖只读快照（含残差/借款/召回/死局）。"""

    sleeve_id: str
    cash: float
    credit: float
    lots: Tuple[P7Lot, ...] = ()
    residuals: Tuple[ResidualEntry, ...] = ()
    borrow_lots: Tuple[BorrowLot, ...] = ()
    recalled_amount: float = 0.0
    deadlocked_amount: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.sleeve_id).strip():
            raise StateError("袖快照 sleeve_id 不得为空")
        for name in ("cash", "credit", "recalled_amount", "deadlocked_amount"):
            object.__setattr__(self, name, _require_finite(getattr(self, name), f"袖 {name}"))
        seen: set = set()
        for lot in self.lots:
            key = (self.sleeve_id, lot.virtual_lot_id)
            if key in seen:
                raise StateError(
                    f"虚拟取得批次身份重复：袖={self.sleeve_id} lot={lot.virtual_lot_id}"
                )
            seen.add(key)

    def shares_of(self, ts_code: str) -> int:
        return sum(lot.shares for lot in self.lots if lot.ts_code == str(ts_code))

    def shares_by_source(self) -> Dict[Tuple[str, str], int]:
        """按 ``(ts_code, 物理来源 lot_id)`` 汇总的整数股数（I2 左值）。"""
        aggregated: Dict[Tuple[str, str], int] = {}
        for lot in self.lots:
            key = (lot.ts_code, lot.source_lot_id)
            aggregated[key] = aggregated.get(key, 0) + lot.shares
        return aggregated

    @property
    def equity(self) -> float:
        """``E_i = cash_i + MV_i - credit_i`` 的现金/债权部分（MV 由调用方加总）。"""
        return self.cash - self.credit


@dataclass
class PhysicalBook:
    """物理账本（真实成交产生的来源批次 + 现金；**独立**于虚拟侧维护）。"""

    cash: float
    lots: Dict[str, PhysicalLot] = field(default_factory=dict)
    total_fee: float = 0.0
    total_buy_amount: float = 0.0
    total_sell_amount: float = 0.0

    def __post_init__(self) -> None:
        self.cash = _require_finite(self.cash, "物理现金")

    def add_lot(self, lot: PhysicalLot) -> None:
        if lot.source_lot_id in self.lots:
            raise StateError(f"物理来源 lot_id 重复：{lot.source_lot_id}")
        self.lots[lot.source_lot_id] = lot

    def consume(self, source_lot_id: str, shares: int) -> None:
        lot = self.lots.get(str(source_lot_id))
        if lot is None:
            raise StateError(f"物理来源批次不存在：{source_lot_id}")
        if shares > lot.shares:
            raise StateError(
                f"物理批次 {source_lot_id} 股数不足：持有 {lot.shares}，申请减少 {shares}"
            )
        if shares == lot.shares:
            self.lots.pop(str(source_lot_id))
        else:
            self.lots[str(source_lot_id)] = replace(lot, shares=lot.shares - shares)

    def shares_by_source(self) -> Dict[Tuple[str, str], int]:
        """I2 右值：``(ts_code, 物理来源 lot_id) -> 整数股数``。"""
        return {(lot.ts_code, lot.source_lot_id): lot.shares for lot in self.lots.values()}

    def shares_of(self, ts_code: str) -> int:
        return sum(lot.shares for lot in self.lots.values() if lot.ts_code == str(ts_code))


@dataclass
class DualCaliber:
    """双口径累计（实际含节省 / 只读反事实不含节省）。"""

    actual_fee: Dict[str, float] = field(default_factory=dict)
    independent_fee: Dict[str, float] = field(default_factory=dict)
    total_value: Dict[str, float] = field(default_factory=dict)
    contribution_nav: Dict[str, float] = field(default_factory=dict)
    independent_nav: Dict[str, Optional[float]] = field(default_factory=dict)
    invalid_sleeves: Dict[str, str] = field(default_factory=dict)

    def savings(self, sleeve_id: str) -> float:
        return float(self.independent_fee.get(sleeve_id, 0.0)) - float(
            self.actual_fee.get(sleeve_id, 0.0)
        )


def _stable(value: Any) -> Any:
    """稳定序列化（浮点用 ``repr`` 保留位模式；映射按键排序）。"""
    if isinstance(value, float):
        return f"f:{value!r}"
    if isinstance(value, Mapping):
        return {str(key): _stable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    return value


def stable_digest(payload: Any) -> str:
    """状态内容摘要（重放逐位一致性的唯一比较口径）。"""
    text = json.dumps(_stable(payload), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class StateEnvelope:
    """P7 专用状态信封（``generation`` 同代提交 + 幂等键 + 双口径累计）。"""

    run_id: str
    generation: int = 0
    as_of: str = ""
    sleeves: Dict[str, SleeveState] = field(default_factory=dict)
    physical: Optional[PhysicalBook] = None
    dual_caliber: DualCaliber = field(default_factory=DualCaliber)
    applied: Dict[str, Tuple[int, str]] = field(default_factory=dict)
    reserved_for_t5: Tuple[str, ...] = ("borrow_lots", "recall", "deadlock", "credit_granted")
    #: 续跑态（C4-04）：价源 last_known 缓存 / 禁买集合 / 交易序号 ——
    #: 这些不在袖与物理账本里，却直接决定下一日的估值与成交，必须随信封一起落盘与比对。
    resume_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.run_id).strip():
            raise StateError("状态信封 run_id 不得为空")

    # ────────────────────────── 幂等与代际 ──────────────────────────

    def commit(
        self,
        *,
        key: str,
        generation: int,
        fingerprint: str,
        sleeves: Mapping[str, SleeveState],
        physical: PhysicalBook,
        as_of: str,
        resume_state: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """同代提交一次状态迁移。

        Returns:
            True = 首次提交；False = 同键同代同内容重复提交（**状态不变**，重放幂等）。

        Raises:
            StateError: 同键不同内容、跨代提交（``generation`` 不连续）或键已被
                更高代占用。
        """
        if not str(key).strip():
            raise StateError("幂等键不得为空")
        if not str(fingerprint).strip():
            raise StateError(f"幂等键 {key} 的指纹不得为空")
        recorded = self.applied.get(key)
        if recorded is not None:
            recorded_generation, recorded_fingerprint = recorded
            if recorded_generation != int(generation):
                raise StateError(
                    f"幂等键 {key} 已登记代际 {recorded_generation}，本次 {generation}；"
                    "跨 generation 提交一律拒绝"
                )
            if recorded_fingerprint != str(fingerprint):
                raise StateError(f"幂等键 {key} 同代但内容不同；禁止同一身份承载不同状态迁移")
            # 同键同代**还要**比对状态本体：指纹只是摘要，不能让「状态不同但计数相同」
            # 的重放被静默当成无操作（T4 复审 B-05）。
            # C4-04：续跑态（价源缓存/禁买集合/序号）与账本本体**一起**校验 —— 它决定
            # 下一日的估值与成交，不是可以排除的旁路；序号推进必须走下一代提交。
            incoming_resume = self.resume_state if resume_state is None else dict(resume_state)
            incoming = stable_digest(
                {
                    "as_of": str(as_of),
                    "sleeves": _sleeve_payload(sleeves),
                    "physical": _physical_payload(physical),
                    "resume_state": _resume_payload(incoming_resume),
                }
            )
            if incoming != stable_digest(self.settlement_payload()):
                raise StateError(
                    f"幂等键 {key} 同代同指纹但**结算后状态不同**（含续跑态）；"
                    "禁止把不同状态当作同一次提交的重放"
                )
            return False
        expected = self.generation + 1
        if int(generation) != expected:
            raise StateError(
                f"幂等键 {key} 的 generation={generation} 与期望 {expected} 不符；"
                "禁止跳代或回退提交"
            )
        self.sleeves = {str(sid): sleeves[sid] for sid in sorted(sleeves)}
        self.physical = physical
        self.generation = int(generation)
        self.as_of = str(as_of)
        self.applied[key] = (int(generation), str(fingerprint))
        if resume_state is not None:
            self.resume_state = dict(resume_state)
        return True

    def restore_from(self, payload: Mapping[str, Any]) -> None:
        """从**完整版本化信封**还原：代际 + 幂等历史 + 双口径 + 续跑态 + 账本本体。

        续跑必须**接着原代际**（不能新开 1 代），否则幂等键与 generation 整体错位，
        「同代重放 = 无操作」的语义失效（C4-04）。
        """
        schema = str(payload.get("schema") or "")
        if schema != ENVELOPE_SCHEMA:
            raise StateError(f"状态信封 schema 不匹配：{schema!r} != {ENVELOPE_SCHEMA!r}")
        generation = int(payload.get("generation") or 0)
        applied = {
            str(key): (int(value["generation"]), str(value["fingerprint"]))
            for key, value in (payload.get("idempotency") or {}).items()
        }
        if generation < 0 or len(applied) != generation:
            raise StateError(
                "状态信封代际与幂等历史不闭合："
                f"generation={generation}，幂等键={len(applied)}（必须逐日一次提交）"
            )
        self.generation = generation
        self.as_of = str(payload.get("as_of") or "")
        self.sleeves = sleeves_from_payload(payload.get("sleeves") or {})
        self.physical = physical_from_payload(payload.get("physical"))
        self.dual_caliber = dual_caliber_from_payload(payload.get("dual_caliber") or {})
        self.applied = applied
        self.resume_state = _resume_payload(payload.get("resume_state") or {})
        self.reserved_for_t5 = tuple(payload.get("reserved_for_t5") or ())

    # ────────────────────────── 一致性 ──────────────────────────

    def assert_credit_balanced(self) -> None:
        """借用态恒等式之一：``Σ credit_i = 0``（本金逐笔对消）。"""
        total = 0.0
        for sid in sorted(self.sleeves):
            total += float(self.sleeves[sid].credit)
        if total != 0.0:
            raise StateError(f"Σ credit_i={total!r} != 0（借入/借出本金未对消）")

    def digest(self) -> str:
        """信封全量内容摘要（含幂等历史；仅用于收尾整体身份）。"""
        return stable_digest(self.to_payload())

    def state_payload(self) -> Dict[str, Any]:
        """**状态本体**载荷（时点 + 袖 + 物理；不含 run_id / 幂等历史 / 双口径累计）。

        逐日重放与重启核对只比对本载荷：幂等历史随运行天数增长、双口径是**派生**
        指标（累计口径依赖起始日），混入比较会把「状态一致」误判为「不同」。
        """
        payload = self.to_payload()
        return {
            "as_of": payload["as_of"],
            "sleeves": payload["sleeves"],
            "physical": payload["physical"],
        }

    def state_digest(self) -> str:
        """状态本体摘要（同 generation 重放必须逐位一致）。"""
        return stable_digest(self.state_payload())

    def settlement_payload(self) -> Dict[str, Any]:
        """**结算后完整**状态载荷 = 状态本体 + 续跑态（同一次提交一起校验/提交）。

        与 ``state_payload()`` 的区别：后者保持三键契约（逐日重放与证据比较只认它），
        本载荷只在幂等判定与续跑还原时使用（C4-04）。
        """
        payload = self.state_payload()
        payload["resume_state"] = _resume_payload(self.resume_state)
        return payload

    def to_payload(self) -> Dict[str, Any]:
        """证据载荷（``state_matrix.json`` 的状态部分；**版本化**完整信封）。"""
        physical = self.physical
        return {
            "schema": ENVELOPE_SCHEMA,
            "run_id": self.run_id,
            "generation": int(self.generation),
            "as_of": self.as_of,
            "sleeves": _sleeve_payload(self.sleeves),
            "physical": _physical_payload(physical),
            "dual_caliber": {
                "actual_fee": dict(sorted(self.dual_caliber.actual_fee.items())),
                "independent_fee": dict(sorted(self.dual_caliber.independent_fee.items())),
                "savings": {
                    sid: self.dual_caliber.savings(sid)
                    for sid in sorted(self.dual_caliber.actual_fee)
                },
                "total_value": dict(sorted(self.dual_caliber.total_value.items())),
                "contribution_nav": dict(sorted(self.dual_caliber.contribution_nav.items())),
                "independent_nav": dict(sorted(self.dual_caliber.independent_nav.items())),
                "invalid_sleeves": dict(sorted(self.dual_caliber.invalid_sleeves.items())),
            },
            "idempotency": {
                key: {"generation": value[0], "fingerprint": value[1]}
                for key, value in sorted(self.applied.items())
            },
            "resume_state": _resume_payload(self.resume_state),
            "reserved_for_t5": list(self.reserved_for_t5),
        }


def _sleeve_payload(sleeves: Mapping[str, SleeveState]) -> Dict[str, Any]:
    """袖快照的证据形态（``to_payload`` 与幂等比对共用同一实现）。"""
    return {
        sid: {
            "cash": float(state.cash),
            "credit": float(state.credit),
            "cash_minus_credit": float(state.equity),
            "lots": [
                {
                    "virtual_lot_id": lot.virtual_lot_id,
                    "ts_code": lot.ts_code,
                    "source_lot_id": lot.source_lot_id,
                    "shares": int(lot.shares),
                    "acquired_date": lot.acquired_date,
                    "acquired_price": float(lot.acquired_price),
                    "acquired_fee": float(lot.acquired_fee),
                    "buy_date": lot.buy_date,
                    "due_date": lot.due_date,
                    "signal_date": lot.signal_date,
                    "origin_model_version": int(lot.origin_model_version),
                }
                for lot in sorted(state.lots, key=lambda item: item.virtual_lot_id)
            ],
            "residuals": [
                {
                    "intent_key": item.intent_key,
                    "ts_code": item.ts_code,
                    "side": item.side,
                    "shares": int(item.shares),
                    "origin": item.origin,
                    "reason": item.reason,
                    "origin_signal_date": item.first_blocked_date,
                    "signal_date": item.signal_date,
                    "due_date": item.due_date,
                    "days_carried": int(item.days_carried),
                    "model_version": int(item.model_version),
                }
                for item in sorted(state.residuals, key=lambda item: item.intent_key)
            ],
            "borrow_lots": [item.borrow_id for item in state.borrow_lots],
            "recalled_amount": float(state.recalled_amount),
            "deadlocked_amount": float(state.deadlocked_amount),
        }
        for sid, state in sorted(sleeves.items())
    }


def _physical_payload(physical: Optional[PhysicalBook]) -> Optional[Dict[str, Any]]:
    """物理账本的证据形态（``to_payload`` 与幂等比对共用同一实现）。"""
    if physical is None:
        return None
    return {
        "cash": float(physical.cash),
        "total_fee": float(physical.total_fee),
        "total_buy_amount": float(physical.total_buy_amount),
        "total_sell_amount": float(physical.total_sell_amount),
        "lots": [
            {
                "source_lot_id": lot.source_lot_id,
                "ts_code": lot.ts_code,
                "shares": int(lot.shares),
                "buy_date": lot.buy_date,
                "buy_price": float(lot.buy_price),
                "buy_fee": float(lot.buy_fee),
            }
            for lot in sorted(physical.lots.values(), key=lambda item: item.source_lot_id)
        ],
    }


def _resume_payload(resume: Mapping[str, Any]) -> Dict[str, Any]:
    """续跑态的稳定形态（键排序 + 集合排序 + 序号整数化），供幂等比对与信封落盘。

    只白名单化**已知键**的规范形态，其余键原样保留（T5 扩展不需要改本函数）。
    """
    payload: Dict[str, Any] = {}
    for key in sorted(resume or {}):
        value = resume[key]
        if key == "no_new_buy":
            payload[key] = sorted({str(item) for item in (value or ())})
        elif key == "seq":
            payload[key] = int(value or 0)
        elif key == "price_last_known":
            payload[key] = {
                str(code): {
                    "close": float((item or {}).get("close") or 0.0),
                    "day": str((item or {}).get("day") or ""),
                }
                for code, item in sorted((value or {}).items())
            }
        else:
            payload[key] = value
    return payload


def dual_caliber_from_payload(payload: Mapping[str, Any]) -> DualCaliber:
    """把 ``to_payload().dual_caliber`` 还原成 ``DualCaliber``（C4-04 累计口径可还原）。"""
    if not payload:
        return DualCaliber()
    return DualCaliber(
        actual_fee={
            str(key): float(value) for key, value in (payload.get("actual_fee") or {}).items()
        },
        independent_fee={
            str(key): float(value) for key, value in (payload.get("independent_fee") or {}).items()
        },
        total_value={
            str(key): float(value) for key, value in (payload.get("total_value") or {}).items()
        },
        contribution_nav={
            str(key): float(value) for key, value in (payload.get("contribution_nav") or {}).items()
        },
        independent_nav={
            str(key): (None if value is None else float(value))
            for key, value in (payload.get("independent_nav") or {}).items()
        },
        invalid_sleeves={
            str(key): str(value) for key, value in (payload.get("invalid_sleeves") or {}).items()
        },
    )


def sleeves_from_payload(payload: Mapping[str, Any]) -> Dict[str, SleeveState]:
    """把 ``_sleeve_payload`` 的产物**还原**成 ``SleeveState``（C4-04 重启可还原性）。"""
    restored: Dict[str, SleeveState] = {}
    for sleeve_id in sorted(payload):
        item = payload[sleeve_id]
        lots = tuple(
            P7Lot(
                virtual_lot_id=str(row["virtual_lot_id"]),
                ts_code=str(row["ts_code"]),
                source_lot_id=str(row["source_lot_id"]),
                shares=int(row["shares"]),
                acquired_date=str(row["acquired_date"]),
                acquired_price=float(row["acquired_price"]),
                acquired_fee=float(row["acquired_fee"]),
                buy_date=str(row["buy_date"]),
                due_date=str(row.get("due_date") or ""),
                signal_date=str(row.get("signal_date") or ""),
                origin_model_version=int(row["origin_model_version"]),
            )
            for row in item["lots"]
        )
        residuals = tuple(
            ResidualEntry(
                intent_key=str(row["intent_key"]),
                sleeve_id=str(sleeve_id),
                ts_code=str(row["ts_code"]),
                side=str(row["side"]),
                shares=int(row["shares"]),
                origin=str(row["origin"]),
                reason=str(row["reason"]),
                first_blocked_date=str(row["origin_signal_date"]),
                days_carried=int(row["days_carried"]),
                signal_date=str(row.get("signal_date") or ""),
                due_date=str(row.get("due_date") or ""),
                model_version=int(row.get("model_version") or 0),
            )
            for row in item["residuals"]
        )
        restored[sleeve_id] = SleeveState(
            sleeve_id=sleeve_id,
            cash=float(item["cash"]),
            credit=float(item["credit"]),
            lots=lots,
            residuals=residuals,
            recalled_amount=float(item.get("recalled_amount") or 0.0),
            deadlocked_amount=float(item.get("deadlocked_amount") or 0.0),
        )
    return restored


def physical_from_payload(payload: Optional[Mapping[str, Any]]) -> Optional[PhysicalBook]:
    """把 ``_physical_payload`` 的产物还原成 ``PhysicalBook``（C4-04）。"""
    if payload is None:
        return None
    return PhysicalBook(
        cash=float(payload["cash"]),
        lots={
            str(row["source_lot_id"]): PhysicalLot(
                source_lot_id=str(row["source_lot_id"]),
                ts_code=str(row["ts_code"]),
                shares=int(row["shares"]),
                buy_date=str(row["buy_date"]),
                buy_price=float(row["buy_price"]),
                buy_fee=float(row.get("buy_fee") or 0.0),
            )
            for row in payload["lots"]
        },
        total_fee=float(payload.get("total_fee") or 0.0),
        total_buy_amount=float(payload.get("total_buy_amount") or 0.0),
        total_sell_amount=float(payload.get("total_sell_amount") or 0.0),
    )


def fifo_lots(lots: Sequence[P7Lot], ts_code: str) -> List[P7Lot]:
    """单袖某股票的 FIFO 批次序：``(buy_date 升序, 登记序升序)``（F25 §5）。"""
    indexed = [(index, lot) for index, lot in enumerate(lots) if lot.ts_code == str(ts_code)]
    indexed.sort(key=lambda item: (item[1].buy_date, item[0]))
    return [lot for _index, lot in indexed]


def plan_fifo_consumption(
    lots: Sequence[P7Lot], ts_code: str, shares: int
) -> List[Tuple[str, int]]:
    """FIFO 扣减计划 ``[(virtual_lot_id, shares), ...]``（不足即拒绝）。"""
    wanted = _require_shares(int(shares), f"FIFO 扣减股数（{ts_code}）")
    plan: List[Tuple[str, int]] = []
    remaining = wanted
    for lot in fifo_lots(lots, ts_code):
        if remaining == 0:
            break
        take = min(lot.shares, remaining)
        plan.append((lot.virtual_lot_id, take))
        remaining -= take
    if remaining != 0:
        held = sum(lot.shares for lot in lots if lot.ts_code == str(ts_code))
        raise StateError(
            f"股票 {ts_code} 批次不足：申请 {wanted} 股，可用 {held} 股，仍缺 {remaining} 股"
        )
    return plan


def merge_sleeve_lots(lots: Iterable[P7Lot]) -> Dict[Tuple[str, str], int]:
    """按 ``(ts_code, 物理来源 lot_id)`` 汇总虚拟股数（I2 左值）。"""
    out: Dict[Tuple[str, str], int] = {}
    for lot in lots:
        key = (lot.ts_code, lot.source_lot_id)
        out[key] = out.get(key, 0) + lot.shares
    return out
