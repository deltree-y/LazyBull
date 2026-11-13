# -*- coding: utf-8 -*-
"""P7 T5 配额/现金拒绝登记（同一意图当日一行；两类拒绝分字段计数）。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from scripts.v2_p7.t5.quota import SleeveQuotaDecision

#: 额度截断（本 task 新增）
QUOTA_VOID_KEY = "quota_capped_voided_eod"
#: 额度内但费用后现金不足（T4 既有语义）
INSUFFICIENT_CASH_KEY = "insufficient_cash_voided_eod"


def _denied_split(
    quota_only: Mapping[str, SleeveQuotaDecision],
    decisions: Mapping[str, SleeveQuotaDecision],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """把逐意图拒绝拆成「配额拒绝」与「现金拒绝」。

    **上限一律取最终拒绝量**（``decisions[*].denied_by_intent``）：授信完全解除某键的拒绝时，
    该键不再出现在最终拒绝里，授权前的额度拒绝必须随之清零（只用于分类，不留下已解除的量）。
    """
    quota_only_denied: Dict[str, int] = {}
    for _sid, decision in sorted(quota_only.items()):
        for key, shares in decision.denied_by_intent.items():
            quota_only_denied[str(key)] = quota_only_denied.get(str(key), 0) + int(shares)
    quota_denied: Dict[str, int] = {}
    cash_denied: Dict[str, int] = {}
    for _sid, decision in sorted(decisions.items()):
        for key, shares in decision.denied_by_intent.items():
            total = int(shares)
            quota_part = min(total, int(quota_only_denied.get(str(key), 0)))
            if quota_part > 0:
                quota_denied[str(key)] = quota_denied.get(str(key), 0) + quota_part
            leftover = total - quota_part
            if leftover > 0:
                cash_denied[str(key)] = cash_denied.get(str(key), 0) + leftover
    return quota_denied, cash_denied


def _void_reason_of(quota_part: int, cash_part: int) -> str:
    return QUOTA_VOID_KEY if quota_part >= cash_part else INSUFFICIENT_CASH_KEY


def _apply_denials_to_rows(
    rows: List[Dict[str, Any]],
    quota_denied: Mapping[str, int],
    cash_denied: Mapping[str, int],
    voids: Dict[str, int],
) -> set:
    """把拒绝并入已存在的守恒行（返回已处理的意图键）。"""
    seen: set = set()
    for row in rows:
        key = str(row["intent_key"])
        if key not in quota_denied and key not in cash_denied:
            continue
        if key in seen:
            # 上游若出现同键多行，只并入首行（避免对 original/voided 重复累加）
            continue
        quota_part = int(quota_denied.get(key, 0))
        cash_part = int(cash_denied.get(key, 0))
        extra = quota_part + cash_part
        if extra <= 0:
            continue
        row["original"] = int(row["original"]) + extra
        row["voided"] = int(row["voided"]) + extra
        row["quota_denied"] = quota_part
        row["cash_denied"] = cash_part
        row["void_reason"] = row.get("void_reason") or _void_reason_of(quota_part, cash_part)
        seen.add(key)
        _add_voids(voids, quota_part, cash_part)
    return seen


def _append_denial_rows(
    day: str,
    outcome: Any,
    quota_denied: Mapping[str, int],
    cash_denied: Mapping[str, int],
    seen: set,
) -> None:
    """全额拒绝的意图不会进入计划 ⇒ 补一行守恒登记（保持逐意图闭合）。"""
    for key in sorted(set(quota_denied) | set(cash_denied)):
        if key in seen:
            continue
        quota_part = int(quota_denied.get(key, 0))
        cash_part = int(cash_denied.get(key, 0))
        extra = quota_part + cash_part
        if extra <= 0:
            continue
        outcome.conservation_rows.append(
            {
                "trade_date": str(day),
                "intent_key": key,
                "sleeve": key.split("|")[0],
                "ts_code": key.split("|")[-1],
                "side": "buy",
                "scenario": "n/a",
                "original": extra,
                "offset": 0,
                "filled": 0,
                "voided": extra,
                "carry": 0,
                "carried_in": False,
                "void_reason": _void_reason_of(quota_part, cash_part),
                "quota_denied": quota_part,
                "cash_denied": cash_part,
            }
        )
        _add_voids(outcome.voids, quota_part, cash_part)


def _add_voids(voids: Dict[str, int], quota_part: int, cash_part: int) -> None:
    voids[QUOTA_VOID_KEY] = voids.get(QUOTA_VOID_KEY, 0) + int(quota_part)
    voids[INSUFFICIENT_CASH_KEY] = voids.get(INSUFFICIENT_CASH_KEY, 0) + int(cash_part)
