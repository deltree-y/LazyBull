# -*- coding: utf-8 -*-
"""P7 T5 借用会计：出借方分摊、借入方决胜、本金对消（``Σ credit_i = 0``）。

契约依据（任务书 §4.2 / §5 硬契约 1、3）
======================================
- **多出借方按闲置比例分摊**；**多借方按缺口降序 + ``sleeve_id`` 升序**决胜；
- ``L`` 只按**本日转账一次**入账：转账后 ``cash_i = cash_open_i + L_i``、
  ``credit_i = credit_outstanding_i + L_i``（出借方为负）；
- ``credit_outstanding`` **不参与**授权（授权只看 ``cash_open + L``）；
- **本金转移零 P&L**：借入/归还只改 ``cash`` / ``credit``，不改 ``MV`` 与 ``E``；
- ``Σ_i credit_i = 0`` 且**本金逐笔对消**（``borrow.principal_offsets`` 逐 lot 核对）；
- ``E_i <= 0`` 的袖**禁新增买入与 credit**（``nonpositive_equity``）。

浮点残差纪律
============
逐袖余额按 FIFO 序累加；若累加残差非 0（|δ| ≤ 1e-9 元），由**绝对余额最大**的袖吸收，
并把 ``residual_adjustment`` 逐日登记。该调整**不改**任何 lot 的本金、归还与状态，
只保证 ``Σ_i credit_i == 0.0`` **精确成立**（契约硬门）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts.v2_p7.t5.borrow import (
    GRANT_EVENT,
    OUTSTANDING,
    BorrowLot,
    fifo_order,
    principal_offsets,
)

#: 借用台账 schema（``credit_ledger.json``）
CREDIT_SCHEMA = "p7_t5_credit/v1"
#: 浮点残差吸收上限（元）
RESIDUAL_TOLERANCE = 1e-9
#: 单笔借款的最小金额（元；低于此值不立 lot，登记为 ``below_minimum``）
MIN_GRANT = 1e-6


class CreditError(RuntimeError):
    """借用会计前置条件不满足（余额不平、越界分摊、重复 lot），一律 fail-fast。"""


@dataclass(frozen=True)
class CreditGrant:
    """单笔授信（出借 → 借入，本日转账一次入账）。"""

    lender: str
    borrower: str
    amount: float
    due_date: str

    def __post_init__(self) -> None:
        if str(self.lender) == str(self.borrower):
            raise CreditError(f"授信出借/借入袖相同：{self.lender}")
        amount = float(self.amount)
        if not math.isfinite(amount) or amount <= 0.0:
            raise CreditError(f"授信金额必须为正有限数：{self.amount!r}")


def allocate_credit(
    *,
    requests: Mapping[str, float],
    idle: Mapping[str, float],
    due_date: str,
) -> Tuple[List[CreditGrant], Dict[str, Any]]:
    """把借入方缺口按「闲置比例」从出借方额度里分摊。

    Args:
        requests: ``{borrower: 资金缺口}``（缺口降序 + ID 升序决胜）；
        idle: ``{lender: 可出借闲置现金}``（正数；<= 0 自动剔除）；
        due_date: 召回日（T+1）。

    Returns:
        ``(授信列表, 诊断)``；诊断含逐借入方未满足缺口与逐出借方剩余闲置。
    """
    pool = {str(key): max(0.0, float(value)) for key, value in idle.items()}
    grants: List[CreditGrant] = []
    unmet: Dict[str, float] = {}
    ordered = sorted(
        ((str(key), max(0.0, float(value))) for key, value in requests.items()),
        key=lambda item: (-item[1], item[0]),
    )
    for borrower, gap in ordered:
        need = float(gap)
        if need <= MIN_GRANT:
            continue
        while need > MIN_GRANT:
            lenders = {
                key: value for key, value in pool.items() if key != borrower and value > MIN_GRANT
            }
            total = float(sum(lenders.values()))
            if total <= MIN_GRANT:
                break
            take_total = min(need, total)
            for lender in sorted(lenders):
                share = take_total * lenders[lender] / total
                share = min(share, lenders[lender], need)
                if share <= MIN_GRANT:
                    continue
                grants.append(
                    CreditGrant(
                        lender=lender,
                        borrower=borrower,
                        amount=float(share),
                        due_date=str(due_date),
                    )
                )
                pool[lender] = lenders[lender] - share
                need -= share
            if take_total <= MIN_GRANT:
                break
        unmet[borrower] = float(max(0.0, need))
    return grants, {
        "borrower_order": [key for key, _gap in ordered],
        "rule": "多出借方按闲置比例分摊；多借方按缺口降序 + sleeve_id 升序",
        "remaining_idle": {key: float(value) for key, value in sorted(pool.items())},
        "unmet": {key: float(value) for key, value in sorted(unmet.items())},
        "grant_count": len(grants),
        "granted_total": float(sum(item.amount for item in grants)),
    }


class CreditBook:
    """借用台账（lot 账本 + 事件流 + 申请记录；host 持有并负责现金变动）。"""

    def __init__(self) -> None:
        self.lots: List[BorrowLot] = []
        self.events: List[Dict[str, Any]] = []
        self.requests: List[Dict[str, Any]] = []
        self.seq = 0
        self.granted_total = 0.0
        self.returned_total = 0.0
        self.deadlocked_total = 0.0
        self.late_recovery_total = 0.0
        #: 逐日浮点残差吸收记录
        self.residual_adjustments: List[Dict[str, Any]] = []
        #: 逐日 credit 态摘要（续跑比对用）
        self.digest_series: List[Tuple[str, str]] = []

    # ────────────────────────── 授信 ──────────────────────────

    def apply_grants(
        self, *, day: str, grants: Sequence[CreditGrant]
    ) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
        """登记授信 lot，返回逐袖**现金变动**与**credit 变动**（宿主负责落到账本）。

        Returns:
            ``(cash_delta, credit_delta, events)``：

            - ``cash_delta[borrower] += L``、``cash_delta[lender] -= L``；
            - ``credit_delta[borrower] += L``、``credit_delta[lender] -= L``；
            - 两侧逐笔对消，``Σ credit_delta == 0``（逐 lot 精确，2 袖形态下位模式相等）。
        """
        cash_delta: Dict[str, float] = {}
        credit_delta: Dict[str, float] = {}
        events: List[Dict[str, Any]] = []
        for grant in grants:
            amount = float(grant.amount)
            if amount <= MIN_GRANT:
                continue
            # 末日授信无下一交易日 => due_date 为空，登记右删失（不臆造到期日、不判死局）
            self.seq += 1
            lot = BorrowLot(
                borrow_id=f"L{day}#{self.seq:04d}",
                lender=str(grant.lender),
                borrower=str(grant.borrower),
                amount=amount,
                grant_date=str(day),
                due_date=str(grant.due_date),
                state=OUTSTANDING,
            )
            self.lots.append(lot)
            cash_delta[lot.borrower] = cash_delta.get(lot.borrower, 0.0) + amount
            cash_delta[lot.lender] = cash_delta.get(lot.lender, 0.0) - amount
            credit_delta[lot.borrower] = credit_delta.get(lot.borrower, 0.0) + amount
            credit_delta[lot.lender] = credit_delta.get(lot.lender, 0.0) - amount
            self.granted_total += amount
            event = {
                "kind": GRANT_EVENT,
                "borrow_id": lot.borrow_id,
                "lender": lot.lender,
                "borrower": lot.borrower,
                "trade_date": str(day),
                "amount": amount,
                "due_date": lot.due_date,
                "right_censored": bool(lot.right_censored),
            }
            self.events.append(event)
            events.append(event)
        return cash_delta, credit_delta, events

    def record_request(self, *, day: str, sleeve: str, **fields: Any) -> None:
        """登记借用申请（VV-H 也登记：借用申请日占比与缺口分布是可复核机械量）。"""
        self.requests.append({"trade_date": str(day), "sleeve": str(sleeve), **fields})

    # ────────────────────────── 余额 ──────────────────────────

    def credit_by_sleeve(
        self, sleeves: Sequence[str], *, day: str = "", record: bool = False
    ) -> Dict[str, Any]:
        """逐袖 ``credit`` 余额（借入为正、出借为负），并保证 ``Σ == 0.0`` 精确成立。

        ``record=True``（每日结算一次）时把浮点残差吸收登记为带 ``trade_date/sleeve``
        的条目；``record=False``（查询/复核）只计算不登记，避免同日重复落两条。
        """
        balances: Dict[str, float] = {str(sleeve): 0.0 for sleeve in sleeves}
        for lot in fifo_order(self.lots):
            if not lot.open:
                continue
            balances[lot.borrower] = balances.get(lot.borrower, 0.0) + lot.outstanding
            balances[lot.lender] = balances.get(lot.lender, 0.0) - lot.outstanding
        residue = float(sum(balances.values()))
        adjustment = 0.0
        if residue != 0.0:
            if abs(residue) > RESIDUAL_TOLERANCE:
                raise CreditError(
                    f"Σ credit 残差 {residue!r} 超过 {RESIDUAL_TOLERANCE}（拒绝吸收）"
                )
            host = max(sorted(balances), key=lambda key: abs(balances[key]))
            balances[host] = balances[host] - residue
            adjustment = -residue
            if record:
                self.residual_adjustments.append(
                    {
                        "trade_date": str(day),
                        "sleeve": host,
                        "residue": float(residue),
                        "adjustment": float(adjustment),
                    }
                )
        return {"balances": balances, "residue": residue, "adjustment": adjustment}

    def exposures(self) -> Dict[str, Any]:
        return principal_offsets(self.lots)

    def assert_balanced(
        self, sleeves: Sequence[str], *, day: str = "", record: bool = False
    ) -> Dict[str, Any]:
        """``Σ credit = 0`` + 本金逐笔对消（两道独立判定）。"""
        payload = self.credit_by_sleeve(sleeves, day=day, record=record)
        offsets = self.exposures()
        if not offsets["balanced"]:
            raise CreditError(
                "本金未逐笔对消："
                f"借入合计={offsets['borrowed_total']!r} 出借合计={offsets['lent_total']!r}"
            )
        total = float(sum(payload["balances"].values()))
        if total != 0.0:
            raise CreditError(f"Σ credit_i={total!r} != 0（借入/出借本金未对消）")
        return {
            "credit_by_sleeve": {
                key: float(value) for key, value in sorted(payload["balances"].items())
            },
            "residue_adjustment": float(payload["adjustment"]),
            "principal": offsets,
            "balanced": True,
        }

    # ────────────────────────── lot 更新 ──────────────────────────

    def replace_lot(self, lot: BorrowLot) -> None:
        for pos, item in enumerate(self.lots):
            if item.borrow_id == lot.borrow_id:
                self.lots[pos] = lot
                return
        raise CreditError(f"借款 lot 不存在：{lot.borrow_id}")

    def open_lots(self) -> List[BorrowLot]:
        return [lot for lot in fifo_order(self.lots) if lot.open]

    def totals(self) -> Dict[str, Any]:
        open_total = float(sum(lot.outstanding for lot in self.lots if lot.open))
        return {
            "lots": len(self.lots),
            "open_lots": sum(1 for lot in self.lots if lot.open),
            "granted_total": float(self.granted_total),
            "returned_total": float(self.returned_total),
            "outstanding_total": open_total,
            "deadlocked_total": float(self.deadlocked_total),
            "late_recovery_total": float(self.late_recovery_total),
        }

    # ────────────────────────── 证据与续跑 ──────────────────────────

    def payload(self) -> Dict[str, Any]:
        events_by_kind: Dict[str, int] = {}
        for event in self.events:
            kind = str(event.get("kind") or "")
            events_by_kind[kind] = events_by_kind.get(kind, 0) + 1
        return {
            "schema": CREDIT_SCHEMA,
            "totals": self.totals(),
            "events_by_kind": dict(sorted(events_by_kind.items())),
            "lots": [lot.payload() for lot in fifo_order(self.lots)],
            "events": [dict(event) for event in self.events],
            "requests": [dict(item) for item in self.requests],
            "residual_adjustments": [dict(item) for item in self.residual_adjustments],
            "residual_adjustment_total": float(
                sum(abs(float(item.get("adjustment") or 0.0)) for item in self.residual_adjustments)
            ),
            "request_days": len({item["trade_date"] for item in self.requests}),
            "note": (
                "借期与 cap-binding 只作机械量（alpha_consumed=false）；"
                "本金转移零 P&L；late_recovery 只来自自然卖出产生的现金，不设被动卖出路径。"
            ),
        }

    def snapshot(self) -> Dict[str, Any]:
        """可持久化快照（与信封同一次提交一起校验）。"""
        return {
            "schema": CREDIT_SCHEMA,
            "seq": int(self.seq),
            "granted_total": float(self.granted_total),
            "returned_total": float(self.returned_total),
            "deadlocked_total": float(self.deadlocked_total),
            "late_recovery_total": float(self.late_recovery_total),
            "lots": [lot.payload() for lot in fifo_order(self.lots)],
            "events": [dict(event) for event in self.events],
            "requests": [dict(item) for item in self.requests],
            "residual_adjustments": [dict(item) for item in self.residual_adjustments],
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "CreditBook":
        """从快照还原（schema 不符即拒绝，不静默降级）。"""
        schema = str((payload or {}).get("schema") or "")
        if schema and schema != CREDIT_SCHEMA:
            raise CreditError(f"借用台账 schema 不匹配：{schema!r} != {CREDIT_SCHEMA!r}")
        book = cls()
        book.seq = int((payload or {}).get("seq") or 0)
        book.granted_total = float((payload or {}).get("granted_total") or 0.0)
        book.returned_total = float((payload or {}).get("returned_total") or 0.0)
        book.deadlocked_total = float((payload or {}).get("deadlocked_total") or 0.0)
        book.late_recovery_total = float((payload or {}).get("late_recovery_total") or 0.0)
        book.events = [dict(item) for item in (payload or {}).get("events") or ()]
        book.requests = [dict(item) for item in (payload or {}).get("requests") or ()]
        book.residual_adjustments = [
            dict(item) for item in (payload or {}).get("residual_adjustments") or ()
        ]
        for row in (payload or {}).get("lots") or ():
            book.lots.append(
                BorrowLot(
                    borrow_id=str(row["borrow_id"]),
                    lender=str(row["lender"]),
                    borrower=str(row["borrower"]),
                    amount=float(row["amount"]),
                    grant_date=str(row["grant_date"]),
                    due_date=str(row["due_date"]),
                    returned_amount=float(row.get("returned_amount") or 0.0),
                    deadlocked=bool(row.get("deadlocked")),
                    late_recovery=bool(row.get("late_recovery")),
                    recall_date=str(row.get("recall_date") or ""),
                    state=str(row.get("state") or OUTSTANDING),
                )
            )
        return book

    def digest(self) -> str:
        from scripts.v2_p7.t4.state import stable_digest

        return stable_digest(self.snapshot())


def apply_credit_delta(state: Any, credit_delta: Mapping[str, float]) -> Any:
    """把 ``credit_delta`` 落到 ``SleeveState``（借入为正、出借为负；零 P&L）。"""
    from scripts.v2_p7.t4.state import SleeveState

    return SleeveState(
        sleeve_id=state.sleeve_id,
        cash=float(state.cash),
        credit=float(state.credit) + float(credit_delta.get(state.sleeve_id, 0.0)),
        lots=state.lots,
        residuals=state.residuals,
        borrow_lots=state.borrow_lots,
        recalled_amount=float(state.recalled_amount),
        deadlocked_amount=float(state.deadlocked_amount),
    )
