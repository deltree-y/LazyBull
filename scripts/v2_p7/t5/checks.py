# -*- coding: utf-8 -*-
"""P7 T5 验收门判定（配额/授权/借用/非负/恒等式/续跑/旧面）。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts.v2_p7.t5 import scenarios as scenario_mod
from scripts.v2_p7.t5.errors import RunnerError
from scripts.v2_p7.t5.host import T5HostResult
from scripts.v2_p7.t5.params import T5Params, quota_ratio

#: 只需 ``ok``/``failures`` 的简单门（价源门在此，任一臂失败即禁止发布）
_SIMPLE_GATES: Tuple[Tuple[str, str], ...] = (
    ("quota", "配额门未过"),
    ("authorization", "授权门未过"),
    ("credit_accounting", "借用会计门未过"),
    ("price_source", "价源门未过"),
    ("identity_restart", "身份/续跑门未过"),
    ("denial_double_count", "拒绝重复并入"),
    ("credit_state_substitutes", "历史缺席 credit 态缺合成续跑替代证据"),
)


def _conservation_unique(results: Mapping[str, Any]) -> Dict[str, Any]:
    """``(trade_date, intent_key)`` 的行重数（**诊断量，不作硬门**）。

    设计上同一意图当日可按腿（内部抵消 / 市场腿 / 拒绝补行）拆成多行，键不唯一是正常的；
    真正的不变量是「每行代数式成立」（``_conservation_check``）与「拒绝不得重复并入」
    （``_denial_double_count``）。
    """
    arms: Dict[str, Any] = {}
    failures: List[str] = []
    for name, result in sorted(results.items()):
        rows = list(getattr(result, "conservation_rows", ()) or ())
        keys = [(str(row["trade_date"]), str(row["intent_key"])) for row in rows]
        duplicates = len(keys) - len(set(keys))
        arms[name] = {
            "rows": len(keys),
            "unique_keys": len(set(keys)),
            "duplicate_rows": duplicates,
        }
        if duplicates:
            failures.append(f"{name}: {duplicates} 行重复 (trade_date,intent_key)")
    return {
        "ok": True,
        "note": "诊断量：行重数不唯一属设计内（按腿/阶段拆行），不作为发布阻断条件",
        "arms": arms,
        "duplicate_rows_total": sum(int(item["duplicate_rows"]) for item in arms.values()),
        "observations": failures[:10],
    }


def _denial_double_count(results: Mapping[str, Any]) -> Dict[str, Any]:
    """拒绝不得重复并入：同一 ``(trade_date, intent_key)`` 至多一行带 ``voided > 0``。"""
    arms: Dict[str, Any] = {}
    failures: List[str] = []
    for name, result in sorted(results.items()):
        rows = list(getattr(result, "conservation_rows", ()) or ())
        counter: Dict[Tuple[str, str], int] = {}
        for row in rows:
            if int(row.get("voided", 0) or 0) <= 0:
                continue
            key = (str(row["trade_date"]), str(row["intent_key"]))
            counter[key] = counter.get(key, 0) + 1
        bad = {f"{d}|{k}": v for (d, k), v in sorted(counter.items()) if v > 1}
        arms[name] = {"keys_with_voided": len(counter), "multi_row_keys": len(bad)}
        if bad:
            failures.append(f"{name}: {len(bad)} 个键出现多行 voided>0")
    return {"ok": not failures, "arms": arms, "failures": failures[:10]}


def _stop_on_hard_failures(checks: Mapping[str, Any]) -> None:
    failures: List[str] = []
    if not checks["t4_parity"].get("skipped") and not checks["t4_parity"]["ok"]:
        failures.append("T4 兼容门未过（credit=off & quota=off 与 r31 不逐位一致）")
    if not checks["vvh_reference"].get("skipped") and not checks["vvh_reference"]["ok"]:
        failures.append("VV-H 与「同配额禁 credit」参照不逐位一致")
    for key, label in _SIMPLE_GATES:
        row = checks[key]
        if not row["ok"]:
            failures.append(f"{label}：{row.get('failures')}")
    if not checks["nonnegative_gate"]["ok"]:
        failures.append(
            "逐袖非负门破缺："
            f"{checks['nonnegative_gate']['violation_days']} 日，"
            f"min={checks['nonnegative_gate']['min_cash_plus_credit']!r}"
        )
    if not checks["identities"]["ok"]:
        failures.append(f"恒等式破缺日数={checks['identities']['failure_days']}")
    if not checks["constructed_cases"]["ok"]:
        failures.append(f"构造例未通过：{checks['constructed_cases']['failed']}")
    if not checks["legacy_sha"]["ok"]:
        failures.append(f"旧受审面变动：{checks['legacy_sha']['modified_in_worktree']}")
    if failures:
        raise RunnerError(f"T5 硬失败，禁止发布：{'；'.join(failures)}")


# ────────────────────────── 验收门 ──────────────────────────


def _acceptance_checks(
    *,
    params: T5Params,
    results: Mapping[str, T5HostResult],
    cases: Sequence[Any],
    parity: Mapping[str, Any],
    reference: Mapping[str, Any],
    segment: bool = False,
) -> Dict[str, Any]:
    summary = scenario_mod.summarize_cases(cases)
    return {
        "quota": _quota_check(results),
        "conservation_unique": _conservation_unique(results),
        "denial_double_count": _denial_double_count(results),
        "authorization": _authorization_check(results),
        "credit_accounting": _credit_check(results),
        "nonnegative_gate": _gate_check(results),
        "identities": _identity_check(results),
        "constructed_cases": {
            "ok": bool(summary["failed"] == []),
            "total": summary["total"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "synthetic": summary["synthetic"],
            "constructed_on_history": summary["constructed_on_history"],
        },
        "identity_restart": _restart_check(results, segment=segment),
        "t4_parity": dict(parity),
        "vvh_reference": dict(reference),
        "legacy_sha": _legacy_check(params, results),
        "alpha_consumed": False,
        "scope_note": (
            "只交付机械量与逐日序列：两臂绩效、三连指标经济裁定与阶段收口归 T6；"
            "alpha_consumed=false，verdict=null。"
        ),
        "price_source": _price_source_all(results),
        "credit_state_substitutes": _credit_state_substitutes(
            cases, _restart_credit_absent(results)
        ),
    }


def _quota_check(results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    failures: List[str] = []
    rows: List[Dict[str, Any]] = []
    for name in ("vvh", "vvs"):
        result = results.get(name)
        if result is None:
            continue
        for row in result.quota_rows:
            rows.append({**row, "arm": name})
            if float(row["quota"]) < -1e-9:
                failures.append(f"{name} {row['trade_date']} {row['sleeve']} R<0")
            if float(row["quota"]) > float(row["demand"]) + 1e-6:
                failures.append(f"{name} {row['trade_date']} {row['sleeve']} R>D（配额未进入 R）")
            if (
                float(row["authorized_notional"])
                > min(float(row["demand"]), float(row["quota_with_credit"])) + 1e-6
            ):
                failures.append(f"{name} {row['trade_date']} {row['sleeve']} A>min(D,R+L)")
    binding = [row for row in rows if row["cap_binding"]]
    days = sorted({row["trade_date"] for row in rows})
    binding_days = sorted({row["trade_date"] for row in binding})
    return {
        "ok": not failures,
        "failures": failures[:10],
        "sleeve_days": len(rows),
        "days": len(days),
        "cap_binding_sleeve_days": len(binding),
        "cap_binding_days": len(binding_days),
        "not_binding_days": len(days) - len(binding_days),
        "denied_shares_total": int(sum(int(row["denied_shares"]) for row in rows)),
        "no_lot_days": sorted(
            {row["trade_date"] for row in rows if row["reason"] == "quota_no_lot_after_rounding"}
        ),
        "cap_binding_examples": binding[:5],
        "ratio": dict(sorted(quota_ratio().items())),
        "rule": (
            "R=max(0,ρG−MV)∈[0,D]；A≤min(D,R+L)；配额拒绝与现金拒绝分字段计数；"
            "cap-binding = 当日该袖买入负债被额度或费用后现金截断"
        ),
    }


def params_ratio() -> Mapping[str, float]:
    from scripts.v2_p7.t5.params import quota_ratio

    return quota_ratio()


def _authorization_check(results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    failures: List[str] = []
    rows: List[Dict[str, Any]] = []
    for name in ("vvh", "vvs"):
        result = results.get(name)
        if result is None:
            continue
        for row in result.quota_rows:
            rows.append({**row, "arm": name})
            available = float(row["cash_open"]) + float(row["credit_granted"])
            if float(row["authorized_notional"]) + float(row["authorized_fee"]) > available + 1e-6:
                failures.append(f"{name} {row['trade_date']} {row['sleeve']} A+Fee>cash_open+L")
            # 零份额不收费（``A=0 ⇒ F=0``）：``buy_cost(0)`` 会返回最小佣金
            if float(row["authorized_notional"]) <= 0.0 and float(row["authorized_fee"]) > 0.0:
                failures.append(
                    f"{name} {row['trade_date']} {row['sleeve']} A=0 但 F={row['authorized_fee']!r}"
                )
    return {
        "ok": not failures,
        "failures": failures[:10],
        "sleeve_days": len(rows),
        "credit_outstanding_excluded": True,
        "rule": (
            "A+Fee(A) ≤ cash_open + L；credit_outstanding 不参与授权；"
            "L 只按本日转账一次入账（同一次提交）"
        ),
        "with_credit": sum(1 for row in rows if float(row["credit_granted"]) > 0.0),
        "as_lender": sum(1 for row in rows if float(row["credit_granted"]) < 0.0),
        "zero_share_no_fee": True,
        "execution_leg_order": (
            "内部抵消/转账 → 净市场卖出 → 净市场买入（T4 已收口语义，本 task 复用不改）："
            "转账须由买方袖**自有现金**支撑，故同日卖出回款不参与该转账；该偏差逐日登记 "
            "``transfer_cash_capped``"
        ),
    }


def _credit_check(results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    failures: List[str] = []
    payload: Dict[str, Any] = {}
    for name in ("vvs", "vvh"):
        result = results.get(name)
        if result is None:
            continue
        balance = result.credit_balance
        payload[name] = {
            "totals": result.credit_payload["totals"],
            "credit_by_sleeve": balance["credit_by_sleeve"],
            "principal": balance["principal"],
            "residue_adjustment": balance["residue_adjustment"],
            "requests": len(result.credit_payload["requests"]),
            "zero_pnl_violations": len(result.zero_pnl["violations"]),
        }
        if not balance["balanced"]:
            failures.append(f"{name} Σcredit≠0 或本金未逐笔对消")
        if float(sum(balance["credit_by_sleeve"].values())) != 0.0:
            failures.append(f"{name} Σcredit 不精确为 0")
        if abs(float(balance["residue_adjustment"])) > 1e-9:
            failures.append(f"{name} credit 浮点残差吸收越界")
        if result.zero_pnl["violations"]:
            failures.append(f"{name} 本金转移存在 P&L 变化")
    vvh = results.get("vvh")
    if vvh is not None and int(vvh.credit_payload["totals"]["lots"]) != 0:
        failures.append("VV-H（禁 credit）出现借款 lot")
    return {
        "ok": not failures,
        "failures": failures[:10],
        "per_arm": payload,
        "rule": (
            "Σ credit_i = 0 逐日精确；本金逐笔对消；本金转移零 P&L；"
            "late_recovery 只来自自然卖出（无被动卖出路径）"
        ),
    }


def _gate_check(results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    failures: List[str] = []
    per_arm: Dict[str, Any] = {}
    for name, result in sorted(results.items()):
        per_arm[name] = {
            "ok": bool(result.gate_payload["ok"]),
            "days": int(result.gate_payload["days"]),
            "violation_days": int(result.gate_payload["violation_days"]),
            "min_cash_plus_credit": result.gate_payload["min_cash_plus_credit"],
            "per_sleeve_min": result.gate_payload["per_sleeve_min"],
        }
        if not result.gate_payload["ok"]:
            failures.append(f"{name} 逐袖非负门破缺")
    violations = sum(item["violation_days"] for item in per_arm.values())
    minimum = min(
        (
            item["min_cash_plus_credit"]
            for item in per_arm.values()
            if item["min_cash_plus_credit"] is not None
        ),
        default=None,
    )
    return {
        "ok": not failures,
        "failures": failures,
        "violation_days": int(violations),
        "min_cash_plus_credit": minimum,
        "per_arm": per_arm,
        "rule": "cash_i + credit_i ≥ 0 逐日硬门（CD-2 的 T5 归属），容差 1e-6 元",
    }


def _identity_check(results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    failure_days = {name: len(result.identity_failures) for name, result in sorted(results.items())}
    invalid = {name: dict(result.invalid_days) for name, result in sorted(results.items())}
    return {
        "ok": not any(failure_days.values()),
        "failure_days": sum(failure_days.values()),
        "per_arm": failure_days,
        "invalid_equity_days": invalid,
        "tolerance": {"I1": 1e-6, "I1b": 0.0, "I2": 0.0, "I3": 1e-6, "I3b": 1e-6},
        "note": (
            "I1b（Σcredit=0）与 I2（来源批次逐股）为 0 容差；I1/I3 为 1e-6；"
            "非正 E 不删日（只标 invalid_nonpositive_equity）。"
        ),
    }


def _credit_state_occurrences(result: Any) -> Dict[str, int]:
    """该臂全窗内各 credit 态的**出现日数**（0 表示该态在本窗口从未发生）。"""
    counts = {"borrow_outstanding": 0, "recall_due": 0, "deadlocked": 0}
    for day_row in getattr(result, "credit_daily", ()) or ():
        balances = day_row.get("credit_by_sleeve") or {}
        if any(abs(float(value)) > 0.0 for value in balances.values()):
            counts["borrow_outstanding"] += 1
        if day_row.get("recall_queue"):
            counts["recall_due"] += 1
        if day_row.get("deadlocks"):
            counts["deadlocked"] += 1
    return {key: int(value) for key, value in counts.items()}


def _restart_check(results: Mapping[str, T5HostResult], *, segment: bool = False) -> Dict[str, Any]:
    failures: List[str] = []
    per_arm: Dict[str, Any] = {}
    for name, result in sorted(results.items()):
        restart = result.restart or {}
        replay = result.replay or {}
        per_arm[name] = {
            "replay_checked": bool(replay.get("checked")),
            "replay_identical": replay.get("all_bitwise_identical"),
            "restart_checked": bool(restart.get("checked")),
            "restart_identical": restart.get("all_bitwise_identical"),
            "credit_all_identical": restart.get("credit_all_identical"),
            "checkpoints": restart.get("checkpoint_count"),
            "coverage": restart.get("coverage"),
            "idempotency": restart.get("idempotency"),
        }
        if replay.get("checked") and not replay.get("all_bitwise_identical"):
            failures.append(f"{name} 重放不一致")
        if restart.get("checked"):
            if not restart.get("all_bitwise_identical"):
                failures.append(f"{name} 重启/续跑不一致")
            if not restart.get("credit_all_identical"):
                failures.append(f"{name} credit 台账续跑不一致")
    vvs = per_arm.get("vvs") or {}
    coverage = vvs.get("coverage") or {}
    occurrences = _credit_state_occurrences(results.get("vvs"))
    missing = [
        key
        for key in ("borrow_outstanding", "recall_due", "deadlocked")
        if not coverage.get(key) and int(occurrences.get(key, 0)) > 0
    ]
    if not segment and "vvs" in results and missing:
        failures.append(
            f"VV-S 未覆盖 credit 检查点：{missing}（出现日数={occurrences}；"
            "出现日数为 0 的态按 not_applicable 处理）"
        )
    absent_states = [key for key, value in occurrences.items() if int(value) == 0]
    return {
        "ok": not failures,
        "failures": failures[:10],
        "credit_state_occurrences": occurrences,
        "credit_states_absent": absent_states,
        "per_arm": per_arm,
        "rule": (
            "重放 + 3 个 credit 检查点（借款未还中 / 召回当日 / deadlocked 中）续跑 ≥3 日，"
            "账本/费用/credit/NAV/账本投影行逐位一致"
        ),
    }


def _legacy_check(params: T5Params, results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    modules = params.t4_module_sha
    legacy = params.legacy_sha
    files = legacy.get("files") or {}
    modified = sorted(
        key for key, item in files.items() if item.get("modified_in_worktree") is True
    )
    crosscheck = legacy.get("old_evidence_crosscheck") or {}
    crosscheck_ok = all(item["match"] for item in crosscheck.values()) if crosscheck else None
    return {
        "ok": bool(not modified and crosscheck_ok is not False),
        "modified_in_worktree": modified,
        "src_files": len(files),
        "t4_modules": len(modules),
        "t4_module_sha": {key: str(value["sha256"]) for key, value in sorted(modules.items())},
        "old_evidence_crosscheck_all_match": crosscheck_ok,
        "note": (
            "``src/lazybull/v2/**`` 和 ``scripts/v2_p7/t25/**`` 逐件 SHA 由 T4 参数体系登记；"
            "本 task 追加 ``scripts/v2_p7/t4/**`` 逐件 SHA（证明未改 T4 语义）。"
        ),
    }


def _price_source_all(results: Mapping[str, Any]) -> Dict[str, Any]:
    """逐臂价源结果汇总：**任一臂失败即整体失败**（R1-05）。"""
    arms: Dict[str, Any] = {}
    failures: List[str] = []
    for name, result in sorted(results.items()):
        payload = _price_source(result)
        arms[name] = payload
        if not payload.get("ok"):
            failures.append(f"{name}: {payload.get('failures')}")
    return {
        "ok": not failures,
        "arms": arms,
        "failures": failures[:10],
        "checked_arms": sorted(arms),
    }


def _restart_credit_absent(results: Mapping[str, Any]) -> List[str]:
    """历史窗口内**从未出现**的 credit 态（由续跑门的出现日数判定）。"""
    restart = _restart_check(results)
    return list(restart.get("credit_states_absent") or [])


def _credit_state_substitutes(cases: Sequence[Any], absent: Sequence[str]) -> Dict[str, Any]:
    """历史缺席态必须由构造例提供完整续跑替代证据（R1-06）。

    要求：命中该态的检查点存在、连续推进 >= 3 日、账本/费用/credit/权威序列/投影行
    逐位一致；仅 ``coverage=true`` 不算证据。
    """
    substitutes: Dict[str, Any] = {}
    for case in cases:
        payload = getattr(case, "observed", None) or {}
        substitute = payload.get("deadlock_restart_substitute")
        if substitute:
            substitutes[str(substitute.get("state"))] = substitute
    failures: List[str] = []
    for state in absent:
        substitute = substitutes.get(str(state))
        if not substitute:
            failures.append(f"缺席态 {state} 无合成续跑替代证据")
            continue
        if int(substitute.get("days_continued", 0)) < 3:
            failures.append(f"缺席态 {state} 合成续跑不足 3 日：{substitute.get('days_continued')}")
        if not substitute.get("ok"):
            failures.append(f"缺席态 {state} 合成续跑存在不一致：{substitute.get('failures')}")
    return {
        "ok": not failures,
        "absent_states": list(absent),
        "substitutes": substitutes,
        "failures": failures[:10],
    }


def _price_source(result: T5HostResult) -> Dict[str, Any]:
    payload = result.price_payload or {}
    return {
        "ok": bool(
            payload.get("distribution", {}).get("unavailable", 0) == 0
            and payload.get("suspension_valuation", {}).get("drift", 0) == 0
            and payload.get("valuation_dates_missing", 0) == 0
            and not payload.get("resume_consistency", {}).get("mismatch")
        ),
        "distribution": payload.get("distribution"),
        "valuation_dates_missing": payload.get("valuation_dates_missing"),
        "suspension_drift": payload.get("suspension_valuation", {}).get("drift"),
        "resume_mismatch": len(payload.get("resume_consistency", {}).get("mismatch") or ()),
    }


# ────────────────────────── 落盘 ──────────────────────────
