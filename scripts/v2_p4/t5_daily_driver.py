# -*- coding: utf-8 -*-
"""v2 P4 T5 日级单一驱动：D4 → 旧链恢复或三臂影子 → 门 5/6 证据。

缺省仅打印计划。真实执行须显式 ``--execute --authorized-minutes N``；本工具不替代
用户授权，也不支持真实切换。每次调用只处理一个真实交易日，长窗口由外层按窗口级
预授权逐日调用并登记实耗。

三种模式：

- ``restart``：Q11-A 现役旧链恢复日；有效日进入累计台账，失败日保留现场但不计数；
- ``shadow``：三臂影子日；正式运行必须 ``--active-production``（Q11-d 同频推进现役
  账户），未绑定的开发演练写入 ``sla/drill`` 并标注演练，不能冒充真实证据；
- ``preflight``：首跑前逐源缺口扫描（只读，不写数据、不驱动任何臂）。

D4 任一步/旧链/任一影子臂异常时，先落当日记录（已执行步骤、失败步骤、上下文与
containment），再向上抛出原异常；不因抛异常丢失当天证据，也不统一误标原因。

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_daily_evidence as evidence_io  # noqa: E402
from scripts.v2_p4.t4_gate_runner import capture_data_state  # noqa: E402
from scripts.v2_p4.t5_daily_evidence import (  # noqa: E402
    D4_STAGES,
    MachineBudget,
    atomic_dump,
    build_day_record,
    extract_stage_metrics,
    load_json,
    load_window_records,
    scan_data_gaps,
    write_day_record,
)
from scripts.v2_p4.t5_gate1_identity import load_calendar  # noqa: E402
from scripts.v2_p4.t5_shadow_compare import compare_shadow_day  # noqa: E402
from scripts.v2_p4.t5_sla import evaluate_window_dir  # noqa: E402
from scripts.v2_p4.t5_three_arm import (  # noqa: E402
    ARM_ACTIVE,
    ARM_NEUTRAL,
    ARM_V2,
    ArmSpec,
    _effective_identity,
    prepare_three_arms,
    run_legacy_arm,
    run_v2_arm,
)
from scripts.v2_p4.t05b_live_anchor import snapshot_protected_root  # noqa: E402
from src.lazybull.v2.hosts.paper.config_adapter import (  # noqa: E402
    load_paper_config,
    resolve_engine,
)
from src.lazybull.v2.hosts.paper.factory import create_paper_host  # noqa: E402
from src.lazybull.v2.hosts.paper.shadow import (  # noqa: E402
    PaperShadowAccumulator,
    write_paper_shadow_batch,
)

#: 失败步骤 → 既有系统原因词表（不统一误标；实际阶段与错误另存于 failure_context）
SYSTEM_REASON_BY_STEP: Dict[str, str] = {
    "snapshot_prerequisite": "readiness_false_judgment",
    "ensure_raw": "host_process_crash",
    "ensure_clean": "host_process_crash",
    "cs_infer": "host_process_crash",
    "panel_label_refresh": "host_process_crash",
    "readiness": "readiness_false_judgment",
    "arms": "v2_implementation_defect",
}


BUSINESS_TIMEZONE = "Asia/Shanghai"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def business_today() -> str:
    """业务日期：Asia/Shanghai 的当前交易日文本（tzdata 缺失时退回本地日期）。"""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(BUSINESS_TIMEZONE)).strftime("%Y%m%d")
    except Exception:  # noqa: BLE001 - 缺 tzdata 时退回本地日期并保持可运行
        return datetime.now().strftime("%Y%m%d")


def _dump(path: Path, payload: Any) -> None:
    atomic_dump(path, payload)


def _load(path: Path) -> Any:
    return load_json(path)


def build_daily_plan(mode: str, day: str, run_id: str) -> List[str]:
    """dry-run 计划：数据只经 D4，臂执行严格位于就绪门之后。"""
    if mode not in {"restart", "shadow", "preflight"}:
        raise ValueError(f"未知日跑模式: {mode}")
    if mode == "preflight":
        return [
            f"python scripts/v2_p4/t5_daily_driver.py --mode preflight "
            f"--start-date <扫描起点> --target-date {day} --evidence-dir <证据根>"
        ]
    commands = [
        f"python scripts/v2_p4/daily_data_coordinator.py --stages {stage} "
        f"--target-date {day} --start-date {day} --cs-infer-start {day} "
        f"--run-id {run_id}_daily_{day} --execute"
        for stage in D4_STAGES
    ]
    if mode == "restart":
        commands.append(f"python scripts/paper_trade.py run --trade-date {day}")
    else:
        commands.extend(
            [
                f"legacy_neutral.run({day}) [isolated, exposure=null, network/write fail-fast]",
                f"v2_shadow.run_t1({day}) [isolated, host_mode=paper_shadow]",
                "compare(v2_shadow, legacy_neutral) + evaluate_real_calendar_sla()",
            ]
        )
    return commands


def _run_command(command: Sequence[str], log_path: Path) -> None:
    import subprocess

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as sink:
        sink.write(f"\n===== {_now()} {' '.join(command)}\n")
        sink.flush()
        completed = subprocess.run(
            list(command), cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"子进程 exit={completed.returncode}: {' '.join(command)}；日志={log_path}"
        )


def _d4_command(stage: str, day: str, run_id: str, authorized_minutes: float) -> List[str]:
    return [
        sys.executable,
        "scripts/v2_p4/daily_data_coordinator.py",
        "--stages",
        stage,
        "--target-date",
        day,
        "--start-date",
        day,
        "--cs-infer-start",
        day,
        "--run-id",
        f"{run_id}_daily_{day}",
        "--panel-authorized-minutes",
        str(authorized_minutes),
        "--execute",
    ]


def run_d4(
    *,
    day: str,
    run_id: str,
    authorized_minutes: float,
    log_path: Path,
    data_root: Optional[Path] = None,
    steps: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """逐阶段调用同一 D4 入口；指标从协调者真实产物提取，失败先记录再抛出。"""
    data_root = Path(data_root or ROOT / "data")
    steps = {} if steps is None else steps
    coordinator_run = f"{run_id}_daily_{day}"
    for stage in D4_STAGES:
        started = time.monotonic()
        try:
            _run_command(_d4_command(stage, day, run_id, authorized_minutes), log_path)
        except Exception as exc:
            steps[stage] = {
                "elapsed_seconds": time.monotonic() - started,
                "retries": 0,
                "gaps": [],
                "degraded": True,
                "observed": True,
                "reasons": [f"{stage} 失败: {exc}"],
                "failed": True,
            }
            raise
        metrics = extract_stage_metrics(data_root, coordinator_run, stage)
        steps[stage] = {"elapsed_seconds": time.monotonic() - started, **metrics}
    return steps


def _failed_step(steps: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    for name in reversed(list(steps)):
        if steps[name].get("failed"):
            return evidence_io.D4_STAGE_TO_SLA_STEP.get(name, name)
    return None


def _read_run_status(paper_root: Path, day: str) -> str:
    for kind in ("t1", "t0"):
        path = paper_root / "runs" / f"{kind}_{day}.json"
        if path.is_file():
            return str((_load(path)).get("t0_status") or "")
    return ""


def _next_day(calendar: Sequence[str], day: str) -> str:
    following = [value for value in calendar if value > day]
    if not following:
        raise RuntimeError(f"交易日历无 {day} 的次日，无法读取 T+1 指令")
    return following[0]


def read_orders(paper_root: Path, signal_day: str, calendar: Sequence[str]) -> List[Dict[str, Any]]:
    """读取实际 T+1 指令投影，统一为门 5 八字段比较口径。"""
    execution_day = _next_day(calendar, signal_day)
    path = paper_root / "instructions" / f"{execution_day}.parquet"
    if not path.is_file():
        return []
    frame = pd.read_parquet(path)
    orders: List[Dict[str, Any]] = []
    for record in frame.to_dict("records"):
        action = str(record["action"])
        orders.append(
            {
                "date": signal_day,
                "phase": "T0",
                "action": action,
                "ts_code": str(record["ts_code"]),
                "shares": int(record["shares"]),
                "price_basis": record.get(
                    "buy_price_type" if action == "buy" else "sell_price_type"
                ),
                "reason": record.get("reason"),
                "execution_date": execution_day,
            }
        )
    return orders


def _budget(evidence: Path, authorized_minutes: float, total_minutes: float) -> MachineBudget:
    return MachineBudget(
        evidence / "machine_budget.json",
        total_minutes=float(total_minutes),
        daily_cap_minutes=float(authorized_minutes),
    )


def _finish_budget(budget: MachineBudget, day: str, wall_started: float) -> Dict[str, Any]:
    return budget.finish(day, time.monotonic() - wall_started)


def _classify_failure(record: Dict[str, Any], failed_step: Optional[str], exc: Exception) -> None:
    """失败上下文如实落档：既有词表分类 + 实际步骤/异常，不统一误标。"""
    record["failed_step"] = failed_step
    record["error"] = f"{type(exc).__name__}: {exc}"
    record["failure_context"] = {
        "failed_step": failed_step,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "executed_steps": sorted(record.get("steps") or {}),
    }
    if record.get("system_reason") in (None, ""):
        record["system_reason"] = SYSTEM_REASON_BY_STEP.get(
            str(failed_step), "v2_implementation_defect"
        )


def _persist_restart(evidence: Path, record: Dict[str, Any]) -> Dict[str, Any]:
    """恢复日证据：逐日文件 + 尝试记录总是落盘；仅有效日进入累计台账。"""
    day = str(record["trade_date"])
    write_day_record(evidence / "sla" / "restart" / f"{day}.json", record)
    attempts_path = evidence / "q11a_restart_attempts.json"
    attempts = _load(attempts_path) if attempts_path.is_file() else []
    if not isinstance(attempts, list):
        raise ValueError(f"恢复尝试台账必须为 list: {attempts_path}")
    attempts.append(dict(record))
    _dump(attempts_path, attempts)
    ledger_path = evidence / "q11a_restart_ledger.json"
    records = _load(ledger_path) if ledger_path.is_file() else []
    if not isinstance(records, list):
        raise ValueError(f"累计恢复台账必须为 list: {ledger_path}")
    if record.get("counts_as_recovery_day", False):
        remaining = [item for item in records if str(item.get("trade_date")) != day]
        remaining.append(dict(record))
        records = sorted(remaining, key=lambda item: str(item.get("trade_date")))
        _dump(ledger_path, records)
    return {
        "ledger_count": len(records),
        "freeze_due": len(records) >= 10,
        "freeze_command": (
            f"python scripts/v2_p4/t5_gate1_identity.py --evidence-dir {evidence} "
            f"--model-version <现役模型> --restart-ledger {ledger_path} --freeze"
        ),
    }


def run_restart_day(
    *,
    day: str,
    run_id: str,
    evidence: Path,
    authorized_minutes: float,
    total_minutes: float,
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Q11-A 现役 legacy 恢复日；D4 与旧链实耗分开登记。"""
    data_root = Path(data_root or ROOT / "data")
    paper_root = ROOT / "data" / "paper"
    started_at = _now()
    wall_started = time.monotonic()
    d4_stages: Dict[str, Any] = {}
    arms_step: Optional[Dict[str, Any]] = None
    failure: Optional[Exception] = None
    budget = _budget(evidence, authorized_minutes, total_minutes)
    budget.check(day, allow_reentry=True)
    try:
        run_d4(
            day=day,
            run_id=run_id,
            authorized_minutes=authorized_minutes,
            log_path=ROOT / "logs" / f"{run_id}_{day}_d4.log",
            data_root=data_root,
            steps=d4_stages,
        )
        arm_started = time.monotonic()
        _run_command(
            [sys.executable, "scripts/paper_trade.py", "run", "--trade-date", day],
            ROOT / "logs" / f"{run_id}_{day}_legacy.log",
        )
        arms_step = {
            "elapsed_seconds": time.monotonic() - arm_started,
            "retries": 0,
            "gaps": [],
            "degraded": False,
            "reasons": ["legacy 日跑完成"],
        }
    except Exception as exc:  # noqa: BLE001 - 失败现场必须落档后原样抛出
        failure = exc
    account = _load(paper_root / "state" / "account.json")
    status = _read_run_status(paper_root, day)
    record = build_day_record(
        day=day,
        started_at=started_at,
        completed_at=_now(),
        d4_stages=d4_stages,
        arms_step=arms_step,
        extra={
            "corrected_date": day,
            "account_last_update": account.get("last_update"),
            "account_advanced": account.get("last_update") == day,
            "t0_status": status,
            "actual_calendar_day": day == business_today(),
            "drill": False,
        },
    )
    # 有效日以「账户已推进」为准：证据/日志写入失败不回滚已发生推进，也不重复入账。
    record["counts_as_recovery_day"] = bool(
        record["account_advanced"]
        and record["actual_calendar_day"]
        and not record["ensure_degraded"]
        and not record["missing_steps"]
        and status != ""
        and not status.startswith("error")
    )
    record["run_completed"] = bool(record["counts_as_recovery_day"] and failure is None)
    if failure is not None and record["counts_as_recovery_day"]:
        # 账户已推进、当日恢复有效；失败仅属证据/日志侧，单独留痕不改判定、不重复入账。
        record["evidence_warnings"] = [
            f"{type(failure).__name__}: {failure}",
            f"failed_step={_failed_step(d4_stages) or evidence_io.SLA_ARM_STEP}",
        ]
    elif failure is not None:
        _classify_failure(record, _failed_step(d4_stages) or evidence_io.SLA_ARM_STEP, failure)
    elif not record["counts_as_recovery_day"]:
        record["failure_context"] = {
            "failed_step": "verification",
            "executed_steps": sorted(record["steps"]),
            "gaps": record["gaps"],
        }
        record["system_reason"] = "readiness_false_judgment"
    persisted = _persist_restart(evidence, record)
    record.update(persisted)
    record["budget"] = _finish_budget(budget, day, wall_started)
    _dump(evidence / "sla" / "restart" / f"{day}.json", record)
    if failure is not None:
        raise failure
    if not record["counts_as_recovery_day"]:
        raise RuntimeError(
            f"Q11-A 恢复日未有效完成: {day} status={status!r} "
            f"account={account.get('last_update')} degraded={record['ensure_degraded']} "
            f"gaps={record['gaps']}"
        )
    return record


def _accumulator(evidence: Path) -> PaperShadowAccumulator:
    path = evidence / "shadow_accumulator.json"
    return (
        PaperShadowAccumulator.from_dict(_load(path))
        if path.is_file()
        else PaperShadowAccumulator()
    )


def _write_completed_shadow_batch(
    *,
    run_id: str,
    evidence: Path,
    spec: ArmSpec,
    gate1: Mapping[str, Any],
    accumulator: PaperShadowAccumulator,
) -> str:
    """仅在真实门 5/6 窗口全过后写新 runs 批；不回改历史批次。"""
    flat = load_paper_config(spec.root)
    host = create_paper_host(store=spec.root, config_flat=flat, data_root=str(ROOT / "data"))
    days = accumulator.days
    result = accumulator.result(host=host, start_date=days[0], end_date=days[-1])
    batch_id = f"p4_t5_{run_id}_paper_shadow"
    out = write_paper_shadow_batch(
        result,
        batch_id=batch_id,
        runs_root=ROOT / "data" / "runs",
        config=_effective_identity(flat),
        code_state=gate1["code_state"],
        data_state=gate1["data_state"],
        model_version=gate1["model"]["version"],
        train_start=gate1["model"].get("train_start_date"),
        train_end=gate1["model"].get("train_end_date"),
        baseline_ref=None,
        arm_label=ARM_V2,
    )
    return str(out)


def _assert_production_ready(
    *, paper_root: Path, day: str, gate1: Mapping[str, Any], evidence: Path, calendar: Sequence[str]
) -> Dict[str, Any]:
    """现役臂驱动前核对：engine=legacy、账户末日、完整 run 标记、同日只驱动一次。"""
    if day not in calendar:
        raise ValueError(f"现役臂只接受真实开市日: {day}")
    engine = resolve_engine(load_paper_config(paper_root))
    if engine != "legacy":
        raise RuntimeError(f"影子期生产现役臂必须 engine=legacy，实际={engine}")
    account = _load(paper_root / "state" / "account.json")
    last = account.get("last_update")
    reported = [
        str(item.get("trade_date")) for item in load_window_records(evidence / "sla" / "daily")
    ]
    expected = reported[-1] if reported else (gate1.get("account") or {}).get("last_update")
    if last != expected:
        raise RuntimeError(f"生产账户末日 {last} 与期望 {expected} 不一致，拒绝驱动现役臂")
    if day in reported:
        raise RuntimeError(
            f"交易日 {day} 的现役臂已驱动过：同一交易日只驱动一次；"
            "现役已成功的失败影子日按系统原因重计窗口，不重放也不回滚已发生成交"
        )
    status = _read_run_status(paper_root, str(last))
    if (
        not (paper_root / "runs").joinpath(f"t1_{last}.json").is_file()
        and not (paper_root / "runs").joinpath(f"t0_{last}.json").is_file()
    ):
        raise RuntimeError(f"生产账户末日 {last} 缺完整 run 标记")
    if status.startswith("error"):
        raise RuntimeError(f"生产账户末日 {last} run 状态异常: {status}")
    return {"engine": engine, "account_last_update": last, "run_status": status}


def _shadow_containment(
    *,
    before_prod: Mapping[str, str],
    after_d4: Mapping[str, str],
    after_active: Mapping[str, str],
    after_neutral: Mapping[str, str],
    after_all: Mapping[str, str],
    data_before: Mapping[str, Any],
    data_after: Mapping[str, Any],
    active_production: bool,
    drill: bool,
) -> Dict[str, Any]:
    """逐相位 containment：现役合法写与每个非现役臂的零污染分别判定。"""
    d4_wrote = after_d4 != before_prod
    active_wrote = after_active != after_d4
    neutral_wrote = after_neutral != after_active
    v2_wrote = after_all != after_neutral
    shadow_wrote = neutral_wrote or v2_wrote
    if active_production:
        no_shadow_pollution = not shadow_wrote and not d4_wrote
    else:
        # 未绑定现役账户的演练：生产纸面根任何一侧都不许变化
        no_shadow_pollution = not (shadow_wrote or active_wrote or d4_wrote)
    return {
        "d4_wrote_production": d4_wrote,
        "active_arm_wrote_production": active_wrote,
        "neutral_arm_wrote_production": neutral_wrote,
        "v2_arm_wrote_production": v2_wrote,
        "shadow_arms_wrote_production": shadow_wrote,
        "production_changed_by_active_arm": active_wrote,
        "production_unchanged_outside_active_arm": no_shadow_pollution,
        "production_writes_attributed_to_active_only": no_shadow_pollution,
        "data_state_decision_phase_unchanged": dict(data_before) == dict(data_after),
        "active_production_authorized": active_production,
        "drill": drill,
    }


def run_shadow_day(
    *,
    day: str,
    run_id: str,
    evidence: Path,
    work_root: Path,
    gate1_path: Path,
    authorized_minutes: float,
    total_minutes: float,
    active_production: bool,
    data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """真实三臂日；验收边只比较 neutral/v2，现役臂只留信息性产物。"""
    data_root = Path(data_root or ROOT / "data")
    paper_root = ROOT / "data" / "paper"
    gate1 = _load(gate1_path)
    if not gate1.get("gate1_frozen") or gate1.get("verdict") != "PASS":
        raise RuntimeError("门 1 未冻结（Q11-A 未完成），禁止影子日执行")
    calendar = load_calendar(data_root)
    if day not in calendar:
        raise ValueError(f"影子日只接受真实开市日: {day}")
    drill = not active_production
    budget = _budget(evidence, authorized_minutes, total_minutes)
    specs = prepare_three_arms(
        frozen_paper_root=paper_root,
        work_root=work_root,
        gate1=gate1,
        data_root=data_root,
        evidence=evidence / "initial_migration",
    )
    production_check: Optional[Dict[str, Any]] = None
    if active_production:
        budget.check(day)
        production_check = _assert_production_ready(
            paper_root=paper_root, day=day, gate1=gate1, evidence=evidence, calendar=calendar
        )
        specs[ARM_ACTIVE] = ArmSpec(ARM_ACTIVE, paper_root, "legacy", True)
    started_at = _now()
    wall_started = time.monotonic()
    before_prod = snapshot_protected_root(paper_root)
    d4_stages: Dict[str, Any] = {}
    arms_step: Optional[Dict[str, Any]] = None
    arm_payloads: Dict[str, Any] = {}
    after_d4: Dict[str, str] = dict(before_prod)
    after_active: Dict[str, str] = dict(before_prod)
    after_neutral: Dict[str, str] = dict(before_prod)
    after_all: Dict[str, str] = dict(before_prod)
    data_before: Dict[str, Any] = capture_data_state(data_root)
    data_after: Dict[str, Any] = dict(data_before)
    failure: Optional[Exception] = None
    accumulator = _accumulator(evidence)
    day_evidence = evidence / "gate5" / "daily_inputs"
    try:
        run_d4(
            day=day,
            run_id=run_id,
            authorized_minutes=authorized_minutes,
            log_path=ROOT / "logs" / f"{run_id}_{day}_d4.log",
            data_root=data_root,
            steps=d4_stages,
        )
        after_d4 = snapshot_protected_root(paper_root)
        data_before = capture_data_state(data_root)
        arm_started = time.monotonic()
        arm_payloads["active"] = run_legacy_arm(
            specs[ARM_ACTIVE], data_root=data_root, day=day, evidence_dir=day_evidence
        )
        after_active = snapshot_protected_root(paper_root)
        arm_payloads["neutral"] = run_legacy_arm(
            specs[ARM_NEUTRAL], data_root=data_root, day=day, evidence_dir=day_evidence
        )
        after_neutral = snapshot_protected_root(paper_root)
        arm_payloads["v2"] = run_v2_arm(
            specs[ARM_V2],
            data_root=data_root,
            day=day,
            evidence_dir=day_evidence,
            accumulator=accumulator,
        )
        arms_step = {
            "elapsed_seconds": time.monotonic() - arm_started,
            "retries": 0,
            "gaps": [],
            "degraded": False,
            "reasons": ["三臂执行完成"],
        }
    except Exception as exc:  # noqa: BLE001 - 失败现场必须落档后原样抛出
        failure = exc
    after_all = snapshot_protected_root(paper_root)
    data_after = capture_data_state(data_root)
    containment = _shadow_containment(
        before_prod=before_prod,
        after_d4=after_d4,
        after_active=after_active,
        after_neutral=after_neutral,
        after_all=after_all,
        data_before=data_before,
        data_after=data_after,
        active_production=active_production,
        drill=drill,
    )
    try:
        gate5 = _shadow_gate5(
            evidence=evidence,
            day=day,
            calendar=calendar,
            specs=specs,
            arm_payloads=arm_payloads,
            failure=failure,
        )
    except Exception as exc:  # noqa: BLE001 - 比较异常也必须先落档
        failure = failure or exc
        gate5 = {
            "date": day,
            "verdict": "FAIL",
            "errors": [f"门 5 比较异常: {type(exc).__name__}: {exc}"],
            "unexplained": 0,
            "differences": [],
            "active_informational_only": True,
        }
    record = build_day_record(
        day=day,
        started_at=started_at,
        completed_at=_now(),
        d4_stages=d4_stages,
        arms_step=arms_step,
        extra={
            "drill": drill,
            "active_production": active_production,
            "production_check": production_check,
            "containment": containment,
            "gate5_verdict": gate5.get("verdict"),
            "actual_calendar_day": day == business_today(),
            "corrected_days": sorted(
                str(item.get("date")) for item in arm_payloads.values() if item.get("date")
            ),
        },
    )
    record["run_completed"] = bool(
        failure is None and gate5.get("verdict") == "PASS" and not record["missing_steps"]
    )
    record["account_advanced"] = bool(containment["production_changed_by_active_arm"])
    if failure is not None:
        _classify_failure(record, _failed_step(d4_stages) or evidence_io.SLA_ARM_STEP, failure)
    elif record["run_completed"] is False:
        record["system_reason"] = "v2_implementation_defect"
        record["failure_context"] = {
            "failed_step": "gate5",
            "gate5_verdict": gate5.get("verdict"),
            "unexplained": gate5.get("unexplained"),
            "errors": gate5.get("errors"),
        }
    target = (evidence / "sla" / ("drill" if drill else "daily")) / f"{day}.json"
    write_day_record(target, record)
    window = evaluate_window_dir(evidence / "sla" / "daily", calendar)
    _dump(evidence / "sla" / "window.json", window)
    _dump(evidence / "containment" / f"{day}.json", containment)
    _dump(evidence / "shadow_accumulator.json", accumulator.to_dict())
    record["budget"] = _finish_budget(budget, day, wall_started)
    write_day_record(target, record)
    if failure is not None:
        raise failure
    if not containment["production_unchanged_outside_active_arm"]:
        raise RuntimeError("影子臂写入生产 data/paper，containment FAIL")
    if not containment["data_state_decision_phase_unchanged"]:
        raise RuntimeError("三臂决策段改变数据态，违反 D4 单一协调者约束")
    runs_batch = None
    if not drill and window["verdict"] == "PASS":
        runs_batch = _write_completed_shadow_batch(
            run_id=run_id,
            evidence=evidence,
            spec=specs[ARM_V2],
            gate1=gate1,
            accumulator=accumulator,
        )
    if gate5.get("verdict") != "PASS":
        raise RuntimeError(
            f"门 5 当日 FAIL: {day} unexplained={gate5.get('unexplained')} errors={gate5.get('errors')}"
        )
    return {
        "gate5": gate5,
        "sla": window,
        "containment": containment,
        "runs_batch": runs_batch,
        "drill": drill,
    }


def _shadow_gate5(
    *,
    evidence: Path,
    day: str,
    calendar: Sequence[str],
    specs: Mapping[str, ArmSpec],
    arm_payloads: Mapping[str, Any],
    failure: Optional[Exception],
) -> Dict[str, Any]:
    """门 5 逐日比较；任一侧缺失即 FAIL（不以部分结果冒充一致）。"""
    if failure is not None or not {ARM_ACTIVE, ARM_NEUTRAL, ARM_V2} <= set(arm_payloads):
        return {
            "date": day,
            "verdict": "FAIL",
            "errors": [f"三臂未全部产出: 已执行={sorted(arm_payloads)}"] if failure else [],
            "unexplained": 0,
            "differences": [],
            "active_informational_only": True,
        }
    corrected = {
        str(arm_payloads[name].get("date"))
        for name in (ARM_ACTIVE, ARM_NEUTRAL, ARM_V2)
    }
    if corrected != {day}:
        return {
            "date": day,
            "verdict": "FAIL",
            "errors": [
                f"影子日发生日期校正，拒绝计入真实窗口: requested={day} actual={sorted(corrected)}"
            ],
            "unexplained": 0,
            "differences": [],
            "active_informational_only": True,
        }
    causal_path = evidence / "gate5" / "causal" / f"{day}.json"
    gate5 = compare_shadow_day(
        left_input=arm_payloads[ARM_V2]["decision_input"],
        right_input=arm_payloads[ARM_NEUTRAL]["decision_input"],
        left_orders=read_orders(specs[ARM_V2].root, day, calendar),
        right_orders=read_orders(specs[ARM_NEUTRAL].root, day, calendar),
        evidence=_load(causal_path) if causal_path.is_file() else None,
        acceptance_edge=True,
    )
    gate5["date"] = day
    gate5["active_informational_only"] = True
    gate5["active_t0_status"] = arm_payloads[ARM_ACTIVE].get("t0_status")
    _dump(evidence / "gate5" / f"daily_{day}.json", gate5)
    return gate5


def run_preflight(
    *, evidence: Path, data_root: Path, start_date: str, target_date: str
) -> Dict[str, Any]:
    """首跑前只读扫描各层所需交易日缺口；不写数据、不驱动任何臂。"""
    calendar = load_calendar(data_root)
    result = scan_data_gaps(data_root, start_date, target_date, calendar)
    _dump(evidence / "preflight" / "data_gaps.json", result)
    return result


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("restart", "shadow", "preflight"), required=True)
    parser.add_argument(
        "--trade-date",
        default="",
        help="restart/shadow 必填（真实开市日）；preflight 使用 --target-date",
    )
    parser.add_argument("--run-id", required=False, default="t5")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--work-root")
    parser.add_argument("--gate1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--authorized-minutes", type=float, default=0.0, help="本次（单日）机器时间授权"
    )
    parser.add_argument(
        "--authorized-total-minutes",
        type=float,
        default=0.0,
        help="窗口级机器时间授权总额（真实执行必填）",
    )
    parser.add_argument("--active-production", action="store_true")
    parser.add_argument("--start-date", help="preflight 扫描起点（YYYYMMDD）")
    parser.add_argument("--target-date", help="preflight 扫描截止日（YYYYMMDD）")
    parser.add_argument("--data-root", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    evidence = Path(args.evidence_dir)
    data_root = Path(args.data_root) if args.data_root else ROOT / "data"
    if not args.execute:
        print("\n".join(build_daily_plan(args.mode, args.trade_date, args.run_id)))
        return 0
    if args.mode == "preflight":
        result = run_preflight(
            evidence=evidence,
            data_root=data_root,
            start_date=args.start_date or args.trade_date,
            target_date=args.target_date or args.trade_date,
        )
        print(json.dumps(result, ensure_ascii=False, default=str))
        return 0
    if not args.trade_date:
        raise ValueError("restart/shadow 真实执行必须给出 --trade-date（真实开市日）")
    if args.trade_date != business_today():
        raise ValueError("SLA/Q11-A 只接受当前真实日历交易日；历史回放请使用 T4 工具")
    if args.trade_date not in load_calendar(data_root):
        raise ValueError("SLA/Q11-A 只接受真实开市交易日，非交易日不得校正后计数")
    if args.authorized_minutes <= 0:
        raise ValueError("真实执行必须显式给出本次 --authorized-minutes > 0")
    if args.authorized_total_minutes <= 0:
        raise ValueError("真实执行必须显式给出窗口级 --authorized-total-minutes > 0")
    if args.mode == "restart":
        result = run_restart_day(
            day=args.trade_date,
            run_id=args.run_id,
            evidence=evidence,
            authorized_minutes=args.authorized_minutes,
            total_minutes=args.authorized_total_minutes,
            data_root=data_root,
        )
    else:
        if not args.work_root or not args.gate1:
            raise ValueError("shadow 真实执行必须给 --work-root 与 --gate1")
        result = run_shadow_day(
            day=args.trade_date,
            run_id=args.run_id,
            evidence=evidence,
            work_root=Path(args.work_root),
            gate1_path=Path(args.gate1),
            authorized_minutes=args.authorized_minutes,
            total_minutes=args.authorized_total_minutes,
            active_production=args.active_production,
            data_root=data_root,
        )
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
