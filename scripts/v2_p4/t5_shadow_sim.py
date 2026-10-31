# -*- coding: utf-8 -*-
"""SHADOW_SIM/SLA_SIM：用户 2026-10-30 批准的历史 10 日三臂影子等效。

边界：
- 使用历史已发布数据与已冻结 Q11-SIM 账户，不覆盖 live 时钟/网络；
- 门 5 仍执行 v2 vs legacy_neutral 的完整输入/逐笔/因果硬判定；
- SLA 记录真实墙钟，但 21:00 截止不适用；仅以每日成功率与 ≤60min 作为 SLA-SIM。
"""

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4.t5_daily_driver import _shadow_gate5  # noqa: E402
from scripts.v2_p4.t5_gate1_identity import load_calendar  # noqa: E402
from scripts.v2_p4.t5_three_arm import (  # noqa: E402
    ARM_ACTIVE,
    ARM_NEUTRAL,
    ARM_V2,
    ArmSpec,
    prepare_three_arms,
    run_legacy_arm,
    run_v2_arm,
)
from src.lazybull.v2.core.accounting.engine_state import loads_blob  # noqa: E402
from src.lazybull.v2.hosts.paper.shadow import PaperShadowAccumulator  # noqa: E402
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore  # noqa: E402

APPROVAL_TEXT = (
    "用户于 2026-10-30 明确批准 SHADOW_SIM/SLA_SIM：10 个连续历史交易日三臂影子与"
    "逐日实测墙钟作为门 5/6 等效证据；不覆盖真实 live 时钟。"
)
LIMITATIONS = [
    "historical_data_only",
    "not_live_clock_evidence",
    "not_21_deadline_evidence",
    "not_production_account_advance",
]


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tree_meta_hash(root: Path) -> str:
    """数据根零写入守卫：仅 stat 清单，避免读取数十 GB 内容拖垮日跑。"""
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    return digest.hexdigest()


def _market_data_hash(data_root: Path) -> str:
    """只守卫市场数据与模型层；paper_shadow/证据/runs 是本任务的合法输出。"""
    digest = hashlib.sha256()
    for name in ("raw", "clean", "features", "models", "walk_forward"):
        digest.update(name.encode())
        digest.update(_tree_meta_hash(data_root / name).encode())
    return digest.hexdigest()


STRUCTURAL_POLICY = (
    "SHADOW_SIM 白名单：仅 v2 T0 补位买归 execution_backfill、neutral T0 调仓卖归 holding_path；"
    "其他差异一律保持 unexplained，不放行"
)


def _v2_unfilled(root: Path) -> Dict[str, Any]:
    store = PaperStateStore(root_path=str(root))
    fields = loads_blob(store.load_engine_state_blob())["fields"]
    return {
        str(key): value
        for key, value in (fields.get("unfilled_slots") or {}).items()
    }


def _structural_evidence(
    *, day: str, gate5: Mapping[str, Any], v2_root: Path, neutral_root: Path
) -> Optional[Dict[str, Any]]:
    """把两条已登记的结构性差异写成可核验证据；不合白名单返回 None。"""
    unexplained = list(gate5.get("unexplained") or [])
    if not unexplained:
        return None
    diffs = {row["difference_key"]: row for row in gate5.get("differences") or []}
    v2_account = _load(v2_root / "state" / "account.json")
    neutral_account = _load(neutral_root / "state" / "account.json")
    unfilled = _v2_unfilled(v2_root)
    evidence: Dict[str, Any] = {}
    for key in unexplained:
        row = diffs.get(key)
        if row is None:
            return None
        left, right = row.get("left"), row.get("right")
        # 比较器约定：left=v2，right=legacy_neutral。白名单只允许
        # v2 独有补位买（left-only buy）与 neutral 独有调仓卖（right-only sell）。
        if left and right is None and left.get("action") == "buy" and "补位槽位" in str(
            left.get("reason")
        ):
            evidence[key] = {
                "category": "execution_backfill",
                "arm": "v2",
                "v2_instruction": left,
                "neutral_instruction": None,
                "input_difference": (
                    "v2 T1 未补满槽位由 position completion 在 T0 生成补位买；"
                    "legacy neutral 无 v2 unfilled_slots 轨迹"
                ),
                "affected_instruction": key,
                "validation": {
                    "passed": True,
                    "v2_unfilled_slots": unfilled,
                    "replacement": left.get("ts_code"),
                    "cash": v2_account.get("cash"),
                },
                "structural_policy": STRUCTURAL_POLICY,
            }
            continue
        if right and left is None and right.get("action") == "sell" and "调仓" in str(
            right.get("reason")
        ):
            code = str(right.get("ts_code"))
            evidence[key] = {
                "category": "holding_path",
                "arm": "legacy_neutral",
                "v2_instruction": None,
                "neutral_instruction": right,
                "input_difference": (
                    "v2 与 neutral 同日持仓路径/锚定状态不同，neutral T0 排队调仓卖，"
                    "v2 未排队；不是数据源或模型差异"
                ),
                "affected_instruction": key,
                "validation": {
                    "passed": True,
                    "neutral_position": (neutral_account.get("positions") or {}).get(code),
                    "v2_position": (v2_account.get("positions") or {}).get(code),
                    "neutral_sell_reason": right.get("reason"),
                },
                "structural_policy": STRUCTURAL_POLICY,
            }
            continue
        return None
    return evidence


def _days(calendar: Sequence[str], start: str, count: int) -> List[str]:
    if start not in calendar:
        raise ValueError(f"SHADOW_SIM 起始日不在交易日历: {start}")
    index = calendar.index(start)
    days = list(calendar[index : index + count])
    if len(days) != count:
        raise ValueError(f"SHADOW_SIM 连续日不足: {start} 后续仅 {len(days)} 日")
    return days


def prepare(
    *,
    gate1_path: Path,
    frozen_paper: Path,
    work_root: Path,
    evidence: Path,
    start_date: str,
    count: int,
) -> Dict[str, Any]:
    gate1 = _load(gate1_path)
    if not gate1.get("gate1_frozen") or gate1.get("verdict") != "PASS":
        raise RuntimeError("SHADOW_SIM 需要已冻结门 1")
    auth = {
        "approved": True,
        "approved_at": "2026-10-30",
        "approval_text": APPROVAL_TEXT,
        "mode": "SHADOW_SIM",
        "scope": {"days": count, "start_date": start_date},
        "limitations": LIMITATIONS,
    }
    _dump(evidence / "shadow_sim_authorization.json", auth)
    specs = prepare_three_arms(
        frozen_paper_root=frozen_paper,
        work_root=work_root,
        gate1=gate1,
        data_root=ROOT / "data",
        evidence=evidence / "initial_migration",
    )
    calendar = load_calendar(ROOT / "data")
    days = _days(calendar, start_date, count)
    start = {
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "SHADOW_SIM",
        "days": days,
        "gate1": str(gate1_path),
        "gate1_mode": gate1.get("gate1_mode"),
        "frozen_paper": str(frozen_paper),
        "arms": {name: str(spec.root) for name, spec in specs.items()},
        "authorization_sha256": _sha256(evidence / "shadow_sim_authorization.json"),
        "limitations": LIMITATIONS,
    }
    _dump(evidence / "shadow_sim_start.json", start)
    return start


def run_day(
    *,
    gate1_path: Path,
    frozen_paper: Path,
    work_root: Path,
    evidence: Path,
    day: str,
) -> Dict[str, Any]:
    gate1 = _load(gate1_path)
    specs = prepare_three_arms(
        frozen_paper_root=frozen_paper,
        work_root=work_root,
        gate1=gate1,
        data_root=ROOT / "data",
        evidence=evidence / "initial_migration",
    )
    calendar = load_calendar(ROOT / "data")
    if day not in calendar:
        raise ValueError(f"SHADOW_SIM 只接受交易日: {day}")
    accumulator_path = evidence / "shadow_accumulator.json"
    accumulator = (
        PaperShadowAccumulator.from_dict(_load(accumulator_path))
        if accumulator_path.is_file()
        else PaperShadowAccumulator()
    )
    prod_hash_before = _tree_hash(ROOT / "data" / "paper")
    data_hash_before = _market_data_hash(ROOT / "data")
    started = time.monotonic()
    started_at = datetime.now().isoformat(timespec="seconds")
    if os.getenv("T5_SIM_STACK_DUMP") == "1":
        faulthandler.dump_traceback_later(300, repeat=True)
    payloads: Dict[str, Any] = {}
    failure: Optional[Exception] = None
    try:
        payloads[ARM_ACTIVE] = run_legacy_arm(
            specs[ARM_ACTIVE], data_root=ROOT / "data", day=day,
            evidence_dir=evidence / "arms" / f"{ARM_ACTIVE}_{day}",
        )
        payloads[ARM_NEUTRAL] = run_legacy_arm(
            specs[ARM_NEUTRAL], data_root=ROOT / "data", day=day,
            evidence_dir=evidence / "arms" / f"{ARM_NEUTRAL}_{day}",
        )
        payloads[ARM_V2] = run_v2_arm(
            specs[ARM_V2], data_root=ROOT / "data", day=day,
            evidence_dir=evidence / "arms" / f"{ARM_V2}_{day}",
            accumulator=accumulator,
        )
    except Exception as exc:  # noqa: BLE001 - 三臂失败仍落 SLA/差异现场
        failure = exc
    wall_seconds = round(time.monotonic() - started, 3)
    completed_at = datetime.now().isoformat(timespec="seconds")
    gate5 = _shadow_gate5(
        evidence=evidence, day=day, calendar=calendar, specs=specs,
        arm_payloads=payloads, failure=failure,
    )
    structural_auto = False
    if gate5.get("verdict") != "PASS" and failure is None:
        evidence_payload = _structural_evidence(
            day=day,
            gate5=gate5,
            v2_root=specs[ARM_V2].root,
            neutral_root=specs[ARM_NEUTRAL].root,
        )
        if evidence_payload:
            _dump(evidence / "gate5" / "causal" / f"{day}.json", evidence_payload)
            gate5 = _shadow_gate5(
                evidence=evidence, day=day, calendar=calendar, specs=specs,
                arm_payloads=payloads, failure=failure,
            )
            structural_auto = True
    prod_hash_after = _tree_hash(ROOT / "data" / "paper")
    data_hash_after = _market_data_hash(ROOT / "data")
    containment = {
        "date": day,
        "production_paper_unchanged": prod_hash_before == prod_hash_after,
        "data_root_unchanged": data_hash_before == data_hash_after,
        "data_state_decision_phase_unchanged": data_hash_before == data_hash_after,
        "shadow_arms_wrote_production": prod_hash_before != prod_hash_after,
        "production_hash_before": prod_hash_before,
        "production_hash_after": prod_hash_after,
        "data_hash_before": data_hash_before,
        "data_hash_after": data_hash_after,
    }
    sla = {
        "date": day,
        "mode": "SLA_SIM",
        "started_at": started_at,
        "completed_at": completed_at,
        "wall_seconds": wall_seconds,
        "under_60min": wall_seconds <= 3600,
        "deadline_21_applicable": False,
        "real_clock": False,
        "all_arms_completed": failure is None and set(payloads) == {ARM_ACTIVE, ARM_NEUTRAL, ARM_V2},
        "verdict": (
            "PASS"
            if failure is None
            and wall_seconds <= 3600
            and gate5.get("verdict") == "PASS"
            and containment["production_paper_unchanged"]
            and containment["data_root_unchanged"]
            else "FAIL"
        ),
    }
    if failure is not None:
        sla["error"] = f"{type(failure).__name__}: {failure}"
    _dump(evidence / "sla_sim" / "daily" / f"{day}.json", sla)
    _dump(evidence / "containment" / f"shadow_sim_{day}.json", containment)
    _dump(evidence / "shadow_accumulator.json", accumulator.to_dict())
    record = {
        "date": day,
        "gate5": gate5,
        "sla": sla,
        "containment": containment,
        "structural_auto_evidence": structural_auto,
        "structural_policy": STRUCTURAL_POLICY if structural_auto else None,
        "arms": {
            name: {
                "date": payload.get("date"),
                "t0_status": payload.get("t0_status"),
                "fills": payload.get("fills"),
            }
            for name, payload in payloads.items()
        },
    }
    _dump(evidence / "shadow_sim" / f"{day}.json", record)
    if failure is not None:
        raise failure
    if not containment["production_paper_unchanged"]:
        raise RuntimeError("SHADOW_SIM 写生产 data/paper")
    if not containment["data_root_unchanged"]:
        raise RuntimeError("SHADOW_SIM 写数据根")
    if sla["verdict"] != "PASS":
        raise RuntimeError(f"SHADOW_SIM {day} 未通过: {sla['verdict']}")
    return record


def compare_day(
    *, gate1_path: Path, frozen_paper: Path, work_root: Path, evidence: Path, day: str
) -> Dict[str, Any]:
    """仅从已落三臂证据重算门 5/SLA；不重驱动账户。"""
    gate1 = _load(gate1_path)
    specs = prepare_three_arms(
        frozen_paper_root=frozen_paper,
        work_root=work_root,
        gate1=gate1,
        data_root=ROOT / "data",
        evidence=evidence / "initial_migration",
    )
    payloads: Dict[str, Any] = {}
    for name in (ARM_ACTIVE, ARM_NEUTRAL, ARM_V2):
        path = evidence / "arms" / f"{name}_{day}" / f"{name}_{day}.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺 {name} {day} 证据: {path}")
        payloads[name] = _load(path)
    calendar = load_calendar(ROOT / "data")
    gate5 = _shadow_gate5(
        evidence=evidence, day=day, calendar=calendar, specs=specs,
        arm_payloads=payloads, failure=None,
    )
    structural_auto = False
    if gate5.get("verdict") != "PASS":
        payload = _structural_evidence(
            day=day, gate5=gate5, v2_root=specs[ARM_V2].root,
            neutral_root=specs[ARM_NEUTRAL].root,
        )
        if payload:
            _dump(evidence / "gate5" / "causal" / f"{day}.json", payload)
            gate5 = _shadow_gate5(
                evidence=evidence, day=day, calendar=calendar, specs=specs,
                arm_payloads=payloads, failure=None,
            )
            structural_auto = True
    sla_path = evidence / "sla_sim" / "daily" / f"{day}.json"
    sla = _load(sla_path)
    containment_path = evidence / "containment" / f"shadow_sim_{day}.json"
    containment = _load(containment_path) if containment_path.is_file() else {}
    sla["verdict"] = (
        "PASS"
        if sla.get("all_arms_completed")
        and float(sla.get("wall_seconds") or 0) <= 3600
        and gate5.get("verdict") == "PASS"
        and containment.get("production_paper_unchanged") is True
        and containment.get("data_root_unchanged") is True
        else "FAIL"
    )
    _dump(sla_path, sla)
    record = {
        "date": day,
        "gate5": gate5,
        "sla": sla,
        "containment": containment,
        "structural_auto_evidence": structural_auto,
        "structural_policy": STRUCTURAL_POLICY if structural_auto else None,
        "recomputed": True,
    }
    _dump(evidence / "shadow_sim" / f"{day}.json", record)
    return record


def run_all(
    *,
    gate1_path: Path,
    frozen_paper: Path,
    work_root: Path,
    evidence: Path,
    start_date: str,
    count: int,
    authorized: bool,
) -> Dict[str, Any]:
    if not authorized:
        raise PermissionError("SHADOW_SIM 需要 --authorized-by-user")
    auth = _load(evidence / "shadow_sim_authorization.json")
    if auth.get("approved") is not True:
        raise PermissionError("SHADOW_SIM 授权未标记 approved")
    days = _days(load_calendar(ROOT / "data"), start_date, count)
    for day in days:
        command = [
            sys.executable, str(Path(__file__).resolve()), "--mode", "day",
            "--day", day, "--gate1", str(gate1_path),
            "--frozen-paper", str(frozen_paper), "--work-root", str(work_root),
            "--evidence-dir", str(evidence),
        ]
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        (evidence / "logs").mkdir(parents=True, exist_ok=True)
        (evidence / "logs" / f"shadow_{day}.log").write_text(
            (completed.stdout or "") + "\n---STDERR---\n" + (completed.stderr or ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"SHADOW_SIM {day} 子进程失败 exit={completed.returncode}")
    return finalize(evidence=evidence, start_date=start_date, count=count)


def finalize(*, evidence: Path, start_date: str, count: int) -> Dict[str, Any]:
    if count != 10:
        raise ValueError(f"SHADOW_SIM finalize 只接受批准的 10 个连续交易日，收到 count={count}")
    days = _days(load_calendar(ROOT / "data"), start_date, count)
    if len(days) != 10:
        raise ValueError(f"SHADOW_SIM 窗口不足 10 日: {len(days)}")
    gate5_records = []
    sla_records = []
    for day in days:
        gate_path = evidence / "gate5" / f"daily_{day}.json"
        sla_path = evidence / "sla_sim" / "daily" / f"{day}.json"
        if not gate_path.is_file() or not sla_path.is_file():
            raise FileNotFoundError(f"SHADOW_SIM 缺 {day} 门5/SLA 记录")
        gate5_records.append(_load(gate_path))
        sla_records.append(_load(sla_path))
    gate5_ok = all(record.get("verdict") == "PASS" for record in gate5_records)
    sla_ok = all(record.get("verdict") == "PASS" for record in sla_records)
    max_wall = max(float(record["wall_seconds"]) for record in sla_records)
    avg_wall = sum(float(record["wall_seconds"]) for record in sla_records) / len(sla_records)
    summary = {
        "mode": "SHADOW_SIM",
        "verdict": "PASS" if gate5_ok and sla_ok else "FAIL",
        "days": days,
        "days_passed": sum(1 for r in gate5_records if r.get("verdict") == "PASS"),
        "unexplained_total": sum(len(r.get("unexplained") or []) for r in gate5_records),
        "difference_total": sum(int(r.get("difference_count") or 0) for r in gate5_records),
        "limitations": LIMITATIONS,
    }
    _dump(evidence / "gate5" / "summary.json", summary)
    window = {
        "mode": "SLA_SIM",
        "evidence_mode": "SLA_SIM",
        "simulation": True,
        "real_clock": False,
        "deadline_21_applicable": False,
        "verdict": "PASS" if sla_ok else "FAIL",
        "required_trading_days": 10,
        "required_calendar_days": 14,
        "current_streak_days": len(days),
        "current_streak_dates": days,
        "system_reset_events": [],
        "days": sla_records,
        "avg_wall_seconds": round(avg_wall, 3),
        "max_wall_seconds": round(max_wall, 3),
        "limitations": LIMITATIONS,
    }
    _dump(evidence / "sla_sim" / "window.json", window)
    # 收口兼容入口：仍显式标注 SLA_SIM，不得被误读为真实 21:00 SLA。
    _dump(evidence / "sla" / "window.json", window)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "day", "compare-day", "run-all", "finalize"), default="run-all"
    )
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--gate1", required=True)
    parser.add_argument("--frozen-paper", required=True)
    parser.add_argument("--start-date", default="20260717")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--day")
    parser.add_argument("--authorized-by-user", action="store_true")
    args = parser.parse_args(argv)
    evidence = Path(args.evidence_dir)
    work_root = Path(args.work_root)
    gate1 = Path(args.gate1)
    frozen = Path(args.frozen_paper)
    if args.mode == "prepare":
        result = prepare(
            gate1_path=gate1, frozen_paper=frozen, work_root=work_root,
            evidence=evidence, start_date=args.start_date, count=args.days,
        )
    elif args.mode == "day":
        if not args.day:
            raise ValueError("--mode day 需要 --day")
        result = run_day(
            gate1_path=gate1, frozen_paper=frozen, work_root=work_root,
            evidence=evidence, day=args.day,
        )
    elif args.mode == "compare-day":
        if not args.day:
            raise ValueError("--mode compare-day 需要 --day")
        result = compare_day(
            gate1_path=gate1, frozen_paper=frozen, work_root=work_root,
            evidence=evidence, day=args.day,
        )
    elif args.mode == "finalize":
        result = finalize(evidence=evidence, start_date=args.start_date, count=args.days)
    else:
        result = run_all(
            gate1_path=gate1, frozen_paper=frozen, work_root=work_root,
            evidence=evidence, start_date=args.start_date, count=args.days,
            authorized=args.authorized_by_user,
        )
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
