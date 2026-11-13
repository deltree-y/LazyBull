# -*- coding: utf-8 -*-
"""P7 T5 借款 lot：FIFO、部分归还、逐 lot 状态（本 task 计息为零）。

契约依据（任务书 §4.2 / §5.1）
=============================
- **FIFO 借款 lot**：部分归还、逐 lot 计息为零（本 task 无利息）；
- **T+1 召回**：未按期归还即 ``deadlocked``；自然卖出只记 ``late_recovery``，
  **不得**用被动卖出归还；
- 本金转移**零 P&L**：借入/归还只改 ``cash`` / ``credit``，不改 ``MV`` 与 ``E``（无价变时）；
- ``Σ credit_i = 0`` 且**本金逐笔对消**：借入方 ``credit > 0``、出借方 ``credit < 0``，
  两侧按 lot 逐笔对应（``outstanding`` 为唯一计量口径）。

状态机
======
``outstanding`` --(到期日 EOD 仍未还清)--> ``deadlocked``（永久登记，不清除）；
``deadlocked`` --(此后自然现金归还)--> ``state=returned`` 且 ``late_recovery=True``；
``outstanding`` --(到期日前还清)--> ``state=returned``（``late_recovery=False``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: lot 状态
OUTSTANDING = "outstanding"
DEADLOCKED = "deadlocked"
RETURNED = "returned"
BORROW_STATES: Tuple[str, ...] = (OUTSTANDING, DEADLOCKED, RETURNED)

#: 归还事件类型
REPAY_ON_TIME = "repay_on_time"
REPAY_PARTIAL = "repay_partial"
REPAY_LATE = "repay_late_recovery"
DEADLOCK_EVENT = "deadlock"
GRANT_EVENT = "grant"
RECALL_EVENT = "recall_due"

#: 金额容差（元）：归还金额与账面 outstanding 的比较下限
AMOUNT_EPS = 1e-9


class BorrowError(RuntimeError):
    """借款 lot 前置条件不满足（金额非法、身份冲突、超额归还），一律 fail-fast。"""


@dataclass(frozen=True)
class BorrowLot:
    """单笔借款（本金 + 逐笔归还进度 + 召回/死局/迟回收登记）。"""

    borrow_id: str
    lender: str
    borrower: str
    amount: float
    grant_date: str
    due_date: str
    returned_amount: float = 0.0
    deadlocked: bool = False
    late_recovery: bool = False
    recall_date: str = ""
    state: str = OUTSTANDING

    def __post_init__(self) -> None:
        # ``due_date`` 允许为空：仅在**末日授信无下一交易日**时出现，登记为右删失
        # （``right_censored=True``），不得臆造到期日。
        for label in ("borrow_id", "lender", "borrower", "grant_date"):
            if not str(getattr(self, label)).strip():
                raise BorrowError(f"借款 lot 的 {label} 不得为空：{getattr(self, label)!r}")
        if self.lender == self.borrower:
            raise BorrowError(f"借款 lot {self.borrow_id} 出借/借入袖相同：{self.lender}")
        amount = float(self.amount)
        if not math.isfinite(amount) or amount <= 0.0:
            raise BorrowError(f"借款 lot {self.borrow_id} 本金必须为正有限数：{self.amount!r}")
        returned = float(self.returned_amount)
        if not math.isfinite(returned) or returned < -AMOUNT_EPS:
            raise BorrowError(f"借款 lot {self.borrow_id} 已还金额非法：{self.returned_amount!r}")
        if returned > amount + AMOUNT_EPS:
            raise BorrowError(f"借款 lot {self.borrow_id} 已还 {returned!r} 超过本金 {amount!r}")
        if self.state not in BORROW_STATES:
            raise BorrowError(f"借款 lot {self.borrow_id} 状态未知：{self.state!r}")
        if self.state == RETURNED and self.outstanding > AMOUNT_EPS:
            raise BorrowError(
                f"借款 lot {self.borrow_id} 标为已归还但仍有 outstanding={self.outstanding!r}"
            )
        if self.late_recovery and float(self.returned_amount) <= AMOUNT_EPS:
            # 语义：late_recovery = 发生过**逾期回收**（允许部分归还、lot 仍未结清），
            # 但必须确有归还金额；与「lot 是否结清」解耦（C-T5-06）。
            raise BorrowError(
                f"借款 lot {self.borrow_id} 标 late_recovery 但无归还金额（state={self.state!r}）"
            )

    @property
    def outstanding(self) -> float:
        return max(0.0, float(self.amount) - float(self.returned_amount))

    @property
    def right_censored(self) -> bool:
        """末日授信（无下一交易日）——不判召回、不判死局，逐条登记。"""
        return not str(self.due_date).strip()

    @property
    def open(self) -> bool:
        """仍有未还本金（无论是否已登记死局）。"""
        return self.outstanding > AMOUNT_EPS

    def repay(self, amount: float, day: str) -> Tuple["BorrowLot", Dict[str, Any]]:
        """按金额归还（部分归还合法；超额即拒绝）。返回新 lot 与事件。"""
        value = float(amount)
        if not math.isfinite(value) or value <= 0.0:
            raise BorrowError(f"归还金额必须为正有限数：{amount!r}")
        if value > self.outstanding + AMOUNT_EPS:
            raise BorrowError(
                f"借款 lot {self.borrow_id} 归还 {value!r} 超过未还本金 {self.outstanding!r}"
            )
        applied = min(value, self.outstanding)
        remaining = max(0.0, self.outstanding - applied)
        closed = remaining <= AMOUNT_EPS
        late = bool(self.deadlocked) or (
            bool(str(self.due_date).strip()) and str(day) > str(self.due_date)
        )
        updated = replace(
            self,
            returned_amount=float(self.returned_amount) + applied,
            state=RETURNED if closed else self.state,
            # 「本笔是否逾期回收」与「lot 是否结清」分开表示：逾期部分归还同样计入迟回收
            late_recovery=bool(self.late_recovery or late),
        )
        event_kind = REPAY_LATE if late else (REPAY_ON_TIME if closed else REPAY_PARTIAL)
        event = {
            "kind": event_kind,
            "borrow_id": self.borrow_id,
            "lender": self.lender,
            "borrower": self.borrower,
            "trade_date": str(day),
            "amount": float(applied),
            "outstanding_after": float(remaining),
            "due_date": self.due_date,
            "late_recovery": bool(updated.late_recovery),
        }
        return updated, event

    def deadlock(self, day: str) -> Tuple["BorrowLot", Dict[str, Any]]:
        """到期日 EOD 未还清 => 永久 ``deadlocked``（重复登记是幂等的）。"""
        if not self.open:
            raise BorrowError(f"借款 lot {self.borrow_id} 已还清，不得登记死局")
        updated = replace(self, deadlocked=True, state=DEADLOCKED)
        event = {
            "kind": DEADLOCK_EVENT,
            "borrow_id": self.borrow_id,
            "lender": self.lender,
            "borrower": self.borrower,
            "trade_date": str(day),
            "outstanding": float(self.outstanding),
            "due_date": self.due_date,
            "amount": float(self.amount),
        }
        return updated, event

    def recalled(self, day: str) -> "BorrowLot":
        """召回队列命中（到期日当日）。"""
        return replace(self, recall_date=str(day))

    def payload(self) -> Dict[str, Any]:
        return {
            "borrow_id": self.borrow_id,
            "lender": self.lender,
            "borrower": self.borrower,
            "amount": float(self.amount),
            "returned_amount": float(self.returned_amount),
            "outstanding": float(self.outstanding),
            "grant_date": self.grant_date,
            "due_date": self.due_date,
            "recall_date": self.recall_date,
            "deadlocked": bool(self.deadlocked),
            "late_recovery": bool(self.late_recovery),
            "state": self.state,
            "interest_rate": 0.0,
        }


def fifo_order(lots: Iterable[BorrowLot]) -> List[BorrowLot]:
    """FIFO 序：``(grant_date 升序, borrow_id 升序)``（部分归还逐 lot 推进）。"""
    return sorted(lots, key=lambda lot: (str(lot.grant_date), str(lot.borrow_id)))


def next_due_date(days: Sequence[str], today: str) -> str:
    """T+1 召回日 = 日历上的下一个交易日；末日无下一日时返回空串（登记为右删失）。"""
    window = [str(day) for day in days]
    key = str(today)
    for pos, day in enumerate(window):
        if day == key:
            return str(window[pos + 1]) if pos + 1 < len(window) else ""
    raise BorrowError(f"召回日解析失败：{today!r} 不在运行日历内")


def principal_offsets(lots: Sequence[BorrowLot]) -> Dict[str, Any]:
    """本金逐笔对消核对：``Σ outstanding(借入) == Σ outstanding(出借)``。"""
    borrowed: Dict[str, float] = {}
    lent: Dict[str, float] = {}
    for lot in lots:
        if not lot.open:
            continue
        borrowed[lot.borrower] = borrowed.get(lot.borrower, 0.0) + lot.outstanding
        lent[lot.lender] = lent.get(lot.lender, 0.0) + lot.outstanding
    borrowed_total = float(sum(borrowed.values()))
    lent_total = float(sum(lent.values()))
    return {
        "borrowed_by_sleeve": {key: float(value) for key, value in sorted(borrowed.items())},
        "lent_by_sleeve": {key: float(value) for key, value in sorted(lent.items())},
        "borrowed_total": borrowed_total,
        "lent_total": lent_total,
        "balanced": bool(abs(borrowed_total - lent_total) <= AMOUNT_EPS),
        "open_lots": sum(1 for lot in lots if lot.open),
    }


def find_lot(lots: Sequence[BorrowLot], borrow_id: str) -> Optional[BorrowLot]:
    for lot in lots:
        if lot.borrow_id == str(borrow_id):
            return lot
    return None
