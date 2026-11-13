# -*- coding: utf-8 -*-
"""P7 T5 重启/续跑审计：从 JSON 往返的**完整 T5 信封 + 费用快照 + credit 台账**续跑。

职责
====
1. 从检查点「该日**开始**」的完整 T5 信封（代际 + 幂等历史 + 双口径 + 续跑态 +
   ``credit_state``）与费用台账快照还原（先做 JSON 往返，证明对象真正可持久化）；
2. 用**新宿主**连续推进若干交易日，逐日比对：完整状态本体（lot 级账本）、费用累计、
   实际/独立 NAV、账本投影行、**credit 台账摘要**；
3. 覆盖 T5 指定的三个 credit 检查点场景：「借款未还中」「召回当日」「``deadlocked`` 中」，
   另保留 T4 的「顺延残差 / 停牌中 / 禁买状态」覆盖；
4. 在续跑信封上复核幂等语义（同键同代无操作、跨代拒绝）。

模块边界
========
只依赖 ``state``/``credit`` 的类型与宿主读取面；宿主在 ``_finalize`` 内**延迟导入**本模块，
避免 ``host <-> restart`` 循环导入。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

from scripts.v2_p7.t4.state import StateError
from scripts.v2_p7.t5.state import T5StateEnvelope

#: 重启检查点续跑的交易日数（>=2 才是「多日推进」；末日检查点自动截断）
RESTART_CONTINUATION_DAYS = 3

#: T5 必须覆盖的检查点场景（真实命中才算覆盖）
T5_COVERAGE_KEYS = (
    "borrow_outstanding",
    "recall_due",
    "deadlocked",
    "carry_residual",
    "suspension",
    "no_new_buy",
)


def _first_mismatch(
    window: Sequence[str], flags: Mapping[str, Sequence[bool]]
) -> Dict[str, Any] | None:
    for pos, day in enumerate(window):
        failed = [name for name in sorted(flags) if not flags[name][pos]]
        if failed:
            return {"trade_date": str(day), "failed": failed}
    return None


def _restart_child(
    host: Any, *, envelope: Mapping[str, Any], fee_state: Mapping[str, Any], window: Sequence[str]
) -> Any:
    """用**新宿主**从给定信封 + 费用快照续跑一个窗口（不递归自检）。"""
    from scripts.v2_p7.t5.host import QuotaCreditHost

    child = QuotaCreditHost(
        params=host.params,
        run_id=host.run_id,
        prices=host._fresh_prices(),  # noqa: SLF001 - 同一价源隔离策略
        cost_model=host.cost_model,
        buy_index=host.buy_index,
        sell_index=host.sell_index,
        days=window,
        price_factory=host._price_factory,  # noqa: SLF001 - 同上
        mode=host.mode,
        restore={"envelope": dict(envelope), "fee_book": dict(fee_state)},
        self_check=False,
    )
    return child.run()


def _idempotency_probe(envelope: T5StateEnvelope) -> Dict[str, Any]:
    """同键同代重复提交 => 状态不变；跨代提交 => 拒绝。"""
    key = next(iter(envelope.applied), "") if envelope.applied else ""
    before = envelope.digest()
    same_generation_noop = False
    cross_generation_rejected = False
    if key:
        generation, fingerprint = envelope.applied[key]
        try:
            applied = envelope.commit(
                key=key,
                generation=generation,
                fingerprint=fingerprint,
                sleeves=envelope.sleeves,
                physical=envelope.physical,  # type: ignore[arg-type]
                as_of=envelope.as_of,
            )
            same_generation_noop = applied is False
        except StateError:
            same_generation_noop = False
        try:
            envelope.commit(
                key=f"{key}:cross",
                generation=generation,
                fingerprint="x",
                sleeves=envelope.sleeves,
                physical=envelope.physical,  # type: ignore[arg-type]
                as_of=envelope.as_of,
            )
        except StateError:
            cross_generation_rejected = True
    # 同键同代 + 篡改 credit 台账 => 必须拒绝（credit 是同一提交的一部分）
    credit_state_tamper_rejected: Any = None
    if key:
        generation, fingerprint = envelope.applied[key]
        resume = dict(envelope.resume_state or {})
        credit = dict(resume.get("credit_state") or {})
        credit["lots"] = list(credit.get("lots") or ()) + [
            {"borrow_id": "TAMPER", "amount": 1.0, "lender": "?", "borrower": "?"}
        ]
        resume["credit_state"] = credit
        try:
            envelope.commit(
                key=key,
                generation=generation,
                fingerprint=fingerprint,
                sleeves=envelope.sleeves,
                physical=envelope.physical,  # type: ignore[arg-type]
                as_of=envelope.as_of,
                resume_state=resume,
            )
        except StateError:
            credit_state_tamper_rejected = True
        else:
            credit_state_tamper_rejected = False
    after = envelope.digest()
    return {
        "probe_key": key,
        "same_key_same_generation_is_noop": bool(same_generation_noop),
        "state_unchanged_after_replay": bool(before == after),
        "cross_generation_rejected": bool(cross_generation_rejected),
        "credit_state_tamper_rejected": credit_state_tamper_rejected,
    }


def _credit_digest(envelope_payload: Mapping[str, Any]) -> str:
    """信封内 credit 台账摘要（续跑必须逐位复现）。"""
    from scripts.v2_p7.t4.state import stable_digest

    credit = (envelope_payload.get("resume_state") or {}).get("credit_state") or {}
    return stable_digest(credit)


def _credit_nav_day_equal(replay: Any, reference: Mapping[Any, Any], day: str) -> bool:
    """同一交易日的权威序列（credit 调整后）逐位比较。"""
    left = [
        dict(item)
        for item in getattr(replay, "credit_nav_rows", ())
        if str(item["trade_date"]) == str(day)
    ]
    right = [row for (key_day, _sleeve), row in reference.items() if str(key_day) == str(day)]
    if len(left) != len(right):
        return False
    keyed = {(str(item["trade_date"]), str(item["sleeve"])): item for item in left}
    return all(keyed.get((str(day), str(row["sleeve"]))) == row for row in right)


def audit_restarts(host: Any, state: Any) -> Dict[str, Any]:
    """从检查点重启：还原原代际/幂等历史/价源/**credit 台账**，用新宿主续跑多日。"""
    if not state.checkpoints:
        return {"checked": False, "note": "无检查点"}
    digest_by_day = dict(state.digest_series)
    fee_by_day = dict(state.fee_series)
    credit_by_day = dict(state.credit_series)
    rows_by_day = {row["trade_date"]: row for row in state.day_rows}
    nav_by_day = {
        arm: {row["trade_date"]: row for row in series} for arm, series in state.nav_series.items()
    }
    credit_nav_by_day = {
        (str(row["trade_date"]), str(row["sleeve"])): row for row in state.credit_nav_rows
    }
    rows: List[Dict[str, Any]] = []
    probes: List[Dict[str, Any]] = []
    for item in state.checkpoints:
        index = int(item["index"])
        span = min(RESTART_CONTINUATION_DAYS, len(host._days) - index)  # noqa: SLF001
        window = host._days[index : index + span]  # noqa: SLF001
        envelope_payload = json.loads(
            json.dumps(item.get("envelope") or {}, ensure_ascii=False, sort_keys=True)
        )
        fee_payload = json.loads(
            json.dumps(item.get("fee_state") or {}, ensure_ascii=False, sort_keys=True)
        )
        replay = _restart_child(
            host, envelope=envelope_payload, fee_state=fee_payload, window=window
        )
        flags = {
            "ledger": [
                replay.digest_series[pos][1] == digest_by_day[day] for pos, day in enumerate(window)
            ],
            "fee": [replay.fee_series[pos][1] == fee_by_day[day] for pos, day in enumerate(window)],
            "credit": [
                replay.credit_series[pos][1] == credit_by_day[day] for pos, day in enumerate(window)
            ],
            # 权威序列（credit 调整后）同样逐日逐位比较（R1-02）
            "credit_nav": [_credit_nav_day_equal(replay, credit_nav_by_day, day) for day in window],
            "nav": [
                all(
                    replay.nav_series[arm][pos] == nav_by_day[arm][day]
                    for arm in sorted(nav_by_day)
                )
                for pos, day in enumerate(window)
            ],
            "daily_row": [
                replay.daily_rows[pos] == rows_by_day[day] for pos, day in enumerate(window)
            ],
        }
        end_at_last = index + span == len(host._days)  # noqa: SLF001
        checks = {
            "generation_continued": int(replay.envelope.generation) == index + span,
            "ledger_identical": all(flags["ledger"]),
            "fee_identical": all(flags["fee"]),
            "credit_identical": all(flags["credit"]),
            "nav_identical": all(flags["nav"]),
            "credit_nav_identical": all(flags["credit_nav"]),
            "daily_row_identical": all(flags["daily_row"]),
            "full_fee_payload_identical": (
                replay.fee_book.payload() == host.fee_book.payload() if end_at_last else None
            ),
            "full_credit_payload_identical": (
                replay.credit_payload == host.credit.payload() if end_at_last else None
            ),
        }
        required = {key: value for key, value in checks.items() if value is not None}
        probes.append(_idempotency_probe(replay.envelope))
        rows.append(
            {
                "trade_date": str(item["trade_date"]),
                "index": index,
                "window": [window[0], window[-1]],
                "days_continued": span,
                "restored_generation": int(envelope_payload.get("generation") or 0),
                "restored_idempotency_keys": len(envelope_payload.get("idempotency") or {}),
                "restored_credit_lots": len(
                    ((envelope_payload.get("resume_state") or {}).get("credit_state") or {}).get(
                        "lots"
                    )
                    or ()
                ),
                "price_cache_restored": bool(
                    (envelope_payload.get("resume_state") or {}).get("price_last_known")
                ),
                "coverage": dict(item.get("coverage") or {}),
                **checks,
                "all_bitwise_identical": all(required.values()),
                "first_mismatch": _first_mismatch(window, flags),
            }
        )
    coverage = {
        key: any(bool(row["coverage"].get(key)) for row in rows) for key in T5_COVERAGE_KEYS
    }
    return {
        "checked": True,
        "policy": (
            "检查点 = 该日**开始**的完整 T5 信封（代际 + 幂等历史 + 双口径 + 续跑态 + credit 台账）"
            f"+ 费用台账快照；新宿主连续推进 {RESTART_CONTINUATION_DAYS} 日，"
            "逐日比对账本/费用/credit/NAV/账本投影行。"
        ),
        "checkpoints": rows,
        "checkpoint_count": len(rows),
        "continuation_days": sorted({row["days_continued"] for row in rows}),
        "all_bitwise_identical": all(row["all_bitwise_identical"] for row in rows),
        "generation_all_continued": all(row["generation_continued"] for row in rows),
        "ledger_all_identical": all(row["ledger_identical"] for row in rows),
        "fee_all_identical": all(row["fee_identical"] for row in rows),
        "credit_all_identical": all(row["credit_identical"] for row in rows),
        "nav_all_identical": all(row["nav_identical"] for row in rows),
        "credit_nav_all_identical": all(row["credit_nav_identical"] for row in rows),
        "daily_row_all_identical": all(row["daily_row_identical"] for row in rows),
        "coverage": coverage,
        "coverage_days_continued": {
            key: [
                int(row["days_continued"])
                for row in rows
                if bool((row.get("coverage") or {}).get(key))
            ]
            for key in T5_COVERAGE_KEYS
        },
        "coverage_continuation_ok": {
            key: (
                all(
                    int(row["days_continued"]) >= 3
                    for row in rows
                    if bool((row.get("coverage") or {}).get(key))
                    and row["trade_date"] != rows[-1]["trade_date"]
                )
                if any(bool((row.get("coverage") or {}).get(key)) for row in rows)
                else False
            )
            for key in T5_COVERAGE_KEYS
        },
        "coverage_note": (
            "T5 必达覆盖 = 借款未还中 / 召回当日 / deadlocked 中；"
            "未真实命中的场景由构造例（scenarios）覆盖并在证据中分列。"
        ),
        "idempotency": probes[0] if probes else {},
        "price_book_isolated": host._price_factory is not None,  # noqa: SLF001
    }
