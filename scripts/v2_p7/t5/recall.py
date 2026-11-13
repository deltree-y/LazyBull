# -*- coding: utf-8 -*-
"""P7 T5 召回与死局：T+1 召回队列、``deadlocked`` 永久登记、``late_recovery``。

契约依据（任务书 §4.2 / §6 验收门）
=================================
- **T+1 召回**：到期日（下一个交易日）EOD 仍未还清即 ``deadlocked``，且**永久**登记；
- ``late_recovery`` **只来自自然卖出产生的现金**（归还只发生在本模块的逐日服务里），
  **不设被动卖出路径**（``OrderReason`` 无对应枚举，禁止以卖持仓归还借款）；
- 归还顺序 **FIFO**（``(grant_date, borrow_id)`` 升序），允许部分归还；
- 归还只改 ``cash`` / ``credit``（本金逐笔对消，零 P&L）；
- 末日授信无下一交易日 => ``due_date=""``，登记 ``right_censored``，不判死局（不得臆造到期日）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from scripts.v2_p7.t5.borrow import AMOUNT_EPS, RECALL_EVENT, BorrowLot, fifo_order

#: 服务结果的空骨架键
SERVICE_KEYS: Tuple[str, ...] = (
    "recall_queue",
    "repayments",
    "deadlocks",
    "late_recoveries",
    "right_censored",
)


def _recall_queue(book: Any, day: str) -> List[Dict[str, Any]]:
    """当日召回队列：``due_date == day`` 的未还清 lot（逐日登记，首次命中即记 ``recall_date``）。"""
    queue: List[Dict[str, Any]] = []
    for lot in fifo_order(book.lots):
        if not lot.open or str(lot.due_date) != str(day):
            continue
        recalled = lot if lot.recall_date else lot.recalled(day)
        if recalled is not lot:
            book.replace_lot(recalled)
            book.events.append(
                {
                    "kind": RECALL_EVENT,
                    "borrow_id": recalled.borrow_id,
                    "lender": recalled.lender,
                    "borrower": recalled.borrower,
                    "trade_date": str(day),
                    "amount": float(recalled.amount),
                    "outstanding": float(recalled.outstanding),
                    "due_date": recalled.due_date,
                }
            )
        queue.append(
            {
                "borrow_id": recalled.borrow_id,
                "lender": recalled.lender,
                "borrower": recalled.borrower,
                "outstanding": float(recalled.outstanding),
                "due_date": recalled.due_date,
                "deadlocked": bool(recalled.deadlocked),
            }
        )
    return queue


def service_credit_day(
    *,
    book: Any,
    day: str,
    cash: Mapping[str, float],
) -> Dict[str, Any]:
    """单日借用服务（在**结算后**调用一次）：召回 → 归还 → 死局登记。

    Args:
        book: :class:`scripts.v2_p7.t5.credit.CreditBook`（就地更新 lot 与事件）；
        day: 当前交易日（YYYYMMDD）；
        cash: 逐袖**结算后**现金（归还只能动用已有现金，不得把现金扣成负数）。

    Returns:
        ``{"cash": 新现金映射, "recall_queue": [...], "repayments": [...],
        "deadlocks": [...], "late_recoveries": [...], "right_censored": [...]}``。
    """
    updated_cash = {str(key): float(value) for key, value in cash.items()}
    queue = _recall_queue(book, day)
    repayments: List[Dict[str, Any]] = []
    deadlocks: List[Dict[str, Any]] = []
    late: List[Dict[str, Any]] = []
    right_censored: List[Dict[str, Any]] = []

    for lot in fifo_order(book.lots):
        if not lot.open:
            continue
        if not str(lot.due_date).strip():
            right_censored.append({"borrow_id": lot.borrow_id, "outstanding": lot.outstanding})
            continue
        if str(lot.due_date) > str(day):
            continue
        borrower_cash = float(updated_cash.get(lot.borrower, 0.0))
        available = max(0.0, borrower_cash)
        if available > AMOUNT_EPS:
            amount = min(available, lot.outstanding)
            repaid, event = lot.repay(amount, day)
            book.replace_lot(repaid)
            book.events.append(event)
            book.returned_total += float(amount)
            updated_cash[lot.borrower] = borrower_cash - float(amount)
            updated_cash[lot.lender] = float(updated_cash.get(lot.lender, 0.0)) + float(amount)
            repayments.append(event)
            if event.get("late_recovery"):
                # 逾期归还的**每一笔**（含部分归还）都计入迟回收本金
                book.late_recovery_total += float(amount)
                late.append(dict(event))
            lot = repaid
        if lot.open and not lot.deadlocked:
            deadlocked, event = lot.deadlock(day)
            book.replace_lot(deadlocked)
            book.events.append(event)
            book.deadlocked_total += float(deadlocked.outstanding)
            deadlocks.append(event)

    return {
        "cash": updated_cash,
        "recall_queue": queue,
        "repayments": repayments,
        "deadlocks": deadlocks,
        "late_recoveries": late,
        "right_censored": right_censored,
    }


def due_today(book: Any, day: str) -> List[BorrowLot]:
    """当日应召回的 lot（只读查询，供宿主在**授权前**登记缺口）。"""
    return [lot for lot in fifo_order(book.lots) if lot.open and str(lot.due_date) == str(day)]


def deadlocked_snapshot(book: Any) -> Dict[str, Any]:
    """死局位快照（逐袖金额 + lot 清单；进入状态信封与证据）。"""
    by_sleeve: Dict[str, float] = {}
    lots: List[Dict[str, Any]] = []
    for lot in fifo_order(book.lots):
        if not lot.deadlocked:
            continue
        by_sleeve[lot.borrower] = by_sleeve.get(lot.borrower, 0.0) + float(lot.outstanding)
        lots.append(
            {
                "borrow_id": lot.borrow_id,
                "borrower": lot.borrower,
                "lender": lot.lender,
                "outstanding": float(lot.outstanding),
                "due_date": lot.due_date,
                "late_recovery": bool(lot.late_recovery),
                "state": lot.state,
            }
        )
    ever: Dict[str, float] = {}
    for lot in fifo_order(book.lots):
        if not lot.deadlocked:
            continue
        ever[lot.borrower] = ever.get(lot.borrower, 0.0) + float(lot.amount)
    return {
        "by_sleeve": {key: float(value) for key, value in sorted(by_sleeve.items())},
        "open_deadlocked_by_sleeve": {
            key: float(value) for key, value in sorted(by_sleeve.items())
        },
        "ever_deadlocked_total": float(
            sum(float(lot.amount) for lot in book.lots if lot.deadlocked)
        ),
        "ever_deadlocked_by_sleeve": {key: float(value) for key, value in sorted(ever.items())},
        "lots": lots,
        "total": float(sum(by_sleeve.values())),
        "note": (
            "``by_sleeve``/``total`` 只统计**仍未清偿**的死局本金（迟回收后为 0）；"
            "``ever_deadlocked_*`` 统计**累计曾死局**本金（永久保留，不清除）。"
        ),
    }


def late_recovery_snapshot(book: Any) -> Dict[str, Any]:
    """``late_recovery`` 明细（逾期归还的每一笔，含部分归还）。"""
    events = [
        dict(item)
        for item in book.events
        if item.get("late_recovery") and str(item.get("kind", "")).startswith("repay")
    ]
    return {
        "events": events,
        "count": len(events),
        "amount": float(sum(float(item.get("amount") or 0.0) for item in events)),
    }


__all__ = [
    "SERVICE_KEYS",
    "service_credit_day",
    "due_today",
    "deadlocked_snapshot",
    "late_recovery_snapshot",
]
