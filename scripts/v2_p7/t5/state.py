# -*- coding: utf-8 -*-
"""P7 T5 状态信封扩展：``p7_t5_state/v1``（``credit`` / 借款 lot / 召回 / 死局 / 迟回收）。

设计要点
========
1. **不与 ``p7_t4_state/v2`` 混用**：本信封是独立 schema，``restore_from`` 只认 T5 版本；
2. **兼容视图不变**：``state_payload()`` 仍是 T4 的三键（``as_of/sleeves/physical``），
   逐日摘要与 T4 完成根**逐位可比**（T4 兼容门），T5 的 credit 扩展不进入该视图；
3. **同代提交一次校验**：credit 台账（借款 lot/归还/死局）放在 ``resume_state["credit_state"]``，
   由 T4 既有 ``commit`` 与账本本体**一起**校验（同键同代异状态拒绝、跨代拒绝）；
4. **逐袖资产负债**：``cash / credit / MV / E`` 逐日登记，``E_i = cash_i + MV_i − credit_i``，
   ``cash_i + credit_i >= 0`` 为**逐日硬门**（CD-2 的 T5 归属）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts.v2_p7.t4.state import (
    ENVELOPE_SCHEMA,
    SleeveState,
    StateEnvelope,
    StateError,
    dual_caliber_from_payload,
    physical_from_payload,
    sleeves_from_payload,
    stable_digest,
)

#: T5 版本化信封 schema（**不得**与 ``p7_t4_state/v2`` 混用）
T5_ENVELOPE_SCHEMA = "p7_t5_state/v1"

#: T5 新增/填充的信封字段（证据口径；不改变 T4 三键兼容视图）
T5_ENVELOPE_FIELDS: Tuple[str, ...] = (
    "credit",
    "borrow_lots",
    "recall_queue",
    "deadlocked",
    "late_recovery",
)

#: ``cash_i + credit_i >= 0`` 的硬门容差（元；契约要求 0 违例）
GATE_TOLERANCE = 1e-6


class T5StateError(StateError):
    """T5 信封前置条件不满足（schema/代际/credit 失衡），一律 fail-fast。"""


@dataclass
class T5StateEnvelope(StateEnvelope):
    """T5 信封：T4 语义 + credit 扩展（版本化 ``p7_t5_state/v1``）。"""

    #: credit 扩展视图（借款 lot / 召回 / 死局 / 迟回收；由宿主在提交前注入）
    t5_view: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = super().to_payload()
        payload["schema"] = T5_ENVELOPE_SCHEMA
        payload["t4_compatible_schema"] = ENVELOPE_SCHEMA
        payload["reserved_for_t5"] = []
        payload["t5_view"] = _t5_view_payload(self.t5_view)
        return payload

    def restore_from(self, payload: Mapping[str, Any]) -> None:
        """从 **T5 版本化信封**还原（代际 + 幂等历史 + 双口径 + 续跑态 + 账本本体）。"""
        schema = str((payload or {}).get("schema") or "")
        if schema != T5_ENVELOPE_SCHEMA:
            raise T5StateError(
                f"T5 状态信封 schema 不匹配：{schema!r} != {T5_ENVELOPE_SCHEMA!r}"
                "（禁止与 p7_t4_state/v2 混用）"
            )
        recorded_run = str((payload or {}).get("run_id") or "")
        if recorded_run and recorded_run != str(self.run_id):
            raise T5StateError(
                f"信封 run_id 不匹配：{recorded_run!r} != {self.run_id!r}（禁止跨 run 还原）"
            )
        generation = int(payload.get("generation") or 0)
        applied = {
            str(key): (int(value["generation"]), str(value["fingerprint"]))
            for key, value in (payload.get("idempotency") or {}).items()
        }
        if generation < 0 or len(applied) != generation:
            raise T5StateError(
                "状态信封代际与幂等历史不闭合："
                f"generation={generation}，幂等键={len(applied)}（必须逐日一次提交）"
            )
        self.generation = generation
        self.as_of = str(payload.get("as_of") or "")
        self.sleeves = sleeves_from_payload(payload.get("sleeves") or {})
        self.physical = physical_from_payload(payload.get("physical"))
        self.dual_caliber = dual_caliber_from_payload(payload.get("dual_caliber") or {})
        self.applied = applied
        self.resume_state = dict(payload.get("resume_state") or {})
        self.t5_view = dict(payload.get("t5_view") or {})
        self.reserved_for_t5 = ()

    def credit_state(self) -> Dict[str, Any]:
        """信封内的借用台账快照（``resume_state["credit_state"]`` 为唯一来源）。"""
        return dict((self.resume_state or {}).get("credit_state") or {})


def _t5_view_payload(view: Mapping[str, Any]) -> Dict[str, Any]:
    """credit 扩展视图的稳定形态（键排序；嵌套容器原样保留）。"""
    out: Dict[str, Any] = {}
    for key in sorted(view or {}):
        value = view[key]
        if isinstance(value, Mapping):
            out[str(key)] = {str(k): value[k] for k in sorted(value, key=str)}
        elif isinstance(value, (list, tuple)):
            out[str(key)] = list(value)
        else:
            out[str(key)] = value
    return out


@dataclass(frozen=True)
class SleeveBalance:
    """单袖单日资产负债（``E_i = cash + MV − credit``；``cash_minus_credit`` 按字段落地）。"""

    trade_date: str
    sleeve: str
    cash: float
    credit: float
    market_value: float
    equity: float

    @property
    def cash_plus_credit(self) -> float:
        return float(self.cash) + float(self.credit)

    @property
    def cash_minus_credit(self) -> float:
        return float(self.cash) - float(self.credit)

    def payload(self) -> Dict[str, Any]:
        return {
            "trade_date": self.trade_date,
            "sleeve": self.sleeve,
            "cash": float(self.cash),
            "credit": float(self.credit),
            "market_value": float(self.market_value),
            "equity": float(self.equity),
            "cash_minus_credit": float(self.cash_minus_credit),
            "cash_plus_credit": float(self.cash_plus_credit),
        }


def sleeve_balances(
    *,
    trade_date: str,
    sleeves: Mapping[str, SleeveState],
    prices: Any,
) -> List[SleeveBalance]:
    """逐袖资产负债（MV 走同一估值链：当日价 → 最后已知价 → 买入价）。"""
    rows: List[SleeveBalance] = []
    for sleeve_id in sorted(sleeves):
        state = sleeves[sleeve_id]
        value = 0.0
        for lot in state.lots:
            price, _day, source = prices.valuation(lot.ts_code, float(lot.acquired_price))
            if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
                raise T5StateError(
                    f"{trade_date} 袖 {sleeve_id} 的 {lot.ts_code} 估值不可用（source={source}）"
                )
            value += float(lot.shares) * float(price)
        rows.append(
            SleeveBalance(
                trade_date=str(trade_date),
                sleeve=sleeve_id,
                cash=float(state.cash),
                credit=float(state.credit),
                market_value=float(value),
                equity=float(state.cash) + float(value) - float(state.credit),
            )
        )
    return rows


def assert_nonnegative_gate(rows: Sequence[SleeveBalance]) -> Dict[str, Any]:
    """``cash_i + credit_i >= 0`` **逐日硬门**（本 task 归属；违约即拒绝，不降级为告警）。"""
    violations = [row for row in rows if row.cash_plus_credit < -GATE_TOLERANCE]
    if violations:
        detail = "；".join(
            f"{row.trade_date} {row.sleeve}: cash={row.cash!r} credit={row.credit!r} "
            f"cash+credit={row.cash_plus_credit!r}"
            for row in violations[:5]
        )
        raise T5StateError(f"逐袖非负门破缺（cash_i + credit_i >= 0）：{detail}")
    return {
        "ok": True,
        "tolerance": GATE_TOLERANCE,
        "rows": len(rows),
        "violations": 0,
        "min_cash_plus_credit": (min(row.cash_plus_credit for row in rows) if rows else None),
        "per_sleeve_min": {
            sleeve: min(
                (row.cash_plus_credit for row in rows if row.sleeve == sleeve), default=None
            )
            for sleeve in sorted({row.sleeve for row in rows})
        },
        "rule": "现金可以为负（净额化内部转账的欠款），但必须由 credit 背书",
    }


def zero_pnl_check(
    *, before: Sequence[SleeveBalance], after: Sequence[SleeveBalance]
) -> Dict[str, Any]:
    """本金转移**零 P&L**：无价变时 ``E_i`` 不因借入/归还而改变。

    只比较 ``cash + MV`` 与 ``E`` 的**变化量**：借入/归还只搬 ``cash`` 与 ``credit``，
    两者对 ``E`` 的贡献互相抵消，故 ``ΔE`` 必须为 0（容差 1e-6 元）。
    """
    pairs: List[Dict[str, Any]] = []
    before_by_sleeve = {row.sleeve: row for row in before}
    ok = True
    for row in after:
        previous = before_by_sleeve.get(row.sleeve)
        if previous is None:
            continue
        delta = float(row.equity) - float(previous.equity)
        consistent = abs(delta) <= GATE_TOLERANCE
        ok = ok and consistent
        pairs.append(
            {
                "sleeve": row.sleeve,
                "equity_before": float(previous.equity),
                "equity_after": float(row.equity),
                "delta": delta,
                "ok": bool(consistent),
            }
        )
    return {"ok": bool(ok), "rows": pairs, "tolerance": GATE_TOLERANCE}


def balance_digest(rows: Sequence[SleeveBalance]) -> str:
    return stable_digest([row.payload() for row in rows])


def summary_by_sleeve(rows: Sequence[SleeveBalance]) -> Dict[str, Any]:
    """逐袖逐日序列的机械汇总（不产出任何经济结论）。"""
    out: Dict[str, Any] = {}
    for sleeve in sorted({row.sleeve for row in rows}):
        series = [row for row in rows if row.sleeve == sleeve]
        out[sleeve] = {
            "days": len(series),
            "first": series[0].payload() if series else None,
            "last": series[-1].payload() if series else None,
            "min_cash": min(row.cash for row in series) if series else None,
            "min_cash_plus_credit": (
                min(row.cash_plus_credit for row in series) if series else None
            ),
            "max_credit": max(row.credit for row in series) if series else None,
            "min_credit": min(row.credit for row in series) if series else None,
            "negative_cash_days": sum(1 for row in series if row.cash < 0.0),
            "negative_cash_plus_credit_days": sum(
                1 for row in series if row.cash_plus_credit < -GATE_TOLERANCE
            ),
            "nonpositive_equity_days": sum(1 for row in series if row.equity <= 0.0),
        }
    return out
