# -*- coding: utf-8 -*-
"""Q11-SIM：连续历史交易日的旧链逐日独立进程重启（用户 2026-10-30 批准的等效证据）。

限制（写入每份证据，不可省略）：
- 使用历史已发布数据，不覆盖真实 live 网络、实时钟与当日调度证据；
- 仅作为用户批准的 Q11-A 等效口径，不得在其他场景冒充真实日跑；
- 所有纸面写入限于隔离根，数据根禁写/禁网；失败日保留现场但不计入连续段。
"""

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4.t5_gate1_identity import load_calendar  # noqa: E402
from scripts.v2_p4.t5_three_arm import (  # noqa: E402
    ARM_NEUTRAL,
    ArmSpec,
    run_legacy_arm,
)

APPROVAL_TEXT = (
    "用户于 2026-10-30 明确批准 Q11-SIM 等效：10 个连续历史交易日的旧链逐日重启"
    "可作为 Q11-A 等效证据；不覆盖真实 live 网络与时钟证据。"
)


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _days(calendar: Sequence[str], start: str, count: int) -> List[str]:
    if start not in calendar:
        raise ValueError(f"SIM 起始日不在交易日历: {start}")
    index = calendar.index(start)
    days = list(calendar[index : index + count])
    if len(days) != count:
        raise ValueError(f"SIM 连续日不足: {start} 后续仅 {len(days)} 日")
    return days


def prepare(
    *,
    source_paper: Path,
    target_paper: Path,
    evidence: Path,
    start_date: str,
    count: int,
) -> Dict[str, Any]:
    """复制生产纸面根到持久隔离根，写审批与起始身份；禁止覆盖。"""
    if target_paper.exists():
        raise FileExistsError(f"Q11-SIM 目标根已存在，禁止覆盖: {target_paper}")
    if not (source_paper / "state" / "account.json").is_file():
        raise FileNotFoundError(f"源账户缺失: {source_paper}")
    account = _load(source_paper / "state" / "account.json")
    days = _days(load_calendar(ROOT / "data"), start_date, count)
    target_paper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_paper, target_paper)
    authorization = {
        "approved": True,
        "approved_at": "2026-10-30",
        "approval_text": APPROVAL_TEXT,
        "mode": "Q11_SIM",
        "scope": {"days": count, "start_date": start_date, "end_date": days[-1]},
        "limitations": [
            "historical_data_only",
            "not_live_network_evidence",
            "not_real_clock_evidence",
            "not_production_account_advance",
        ],
    }
    _dump(evidence / "q11a_sim_authorization.json", authorization)
    start = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "source_paper": str(source_paper),
        "target_paper": str(target_paper),
        "source_account_sha256": _sha256(source_paper / "state" / "account.json"),
        "target_account_sha256": _sha256(target_paper / "state" / "account.json"),
        "account_last_update": account.get("last_update"),
        "cash": account.get("cash"),
        "position_count": len(account.get("positions") or {}),
        "days": days,
        "authorization_sha256": _sha256(evidence / "q11a_sim_authorization.json"),
    }
    _dump(evidence / "q11a_sim_start.json", start)
    return start


def run_day(*, root: Path, day: str, evidence: Path) -> Dict[str, Any]:
    """单日独立进程：调用隔离旧链一次，返回可累计台账记录。"""
    before_path = root / "state" / "account.json"
    before_hash = _sha256(before_path)
    before = _load(before_path)
    arm = ArmSpec("legacy_q11sim", root, "legacy", False)
    payload = run_legacy_arm(
        arm, data_root=ROOT / "data", day=day, evidence_dir=evidence / "daily_inputs"
    )
    after_path = root / "state" / "account.json"
    after = _load(after_path)
    status = str(payload.get("t0_status") or "")
    run_record = root / "runs" / f"t1_{day}.json"
    if not run_record.is_file():
        run_record = root / "runs" / f"t0_{day}.json"
    run_payload = _load(run_record) if run_record.is_file() else {}
    raw_evidence = evidence / "daily_inputs" / f"legacy_q11sim_{day}.json"
    daily_evidence = run_record if run_record.is_file() else raw_evidence
    fills = payload.get("fills") or []
    record = {
        "trade_date": day,
        "corrected_date": payload.get("date"),
        "simulation": True,
        "source_kind": "historical_sim",
        "actual_calendar_day": False,
        "restart_verified": True,
        "evidence_mode": "Q11_SIM",
        "authorization": APPROVAL_TEXT,
        "before_account_last_update": before.get("last_update"),
        "account_last_update": after.get("last_update"),
        "before_account_sha256": before_hash,
        "after_account_sha256": _sha256(after_path),
        "run_completed": daily_evidence.is_file()
        and bool(status)
        and not status.startswith("error")
        and status != "not_ready",
        "t0_status": status,
        "daily_run_record": str(daily_evidence).replace("\\", "/")
        if daily_evidence.is_file()
        else None,
        "t0_instructions_count": int(run_payload.get("instructions_count") or 0),
        "account_advanced": after.get("last_update") != before.get("last_update"),
        "account_unchanged_reason": (
            None if after.get("last_update") != before.get("last_update") else "no_t1_fills"
        ),
        "fills": fills,
        "fill_count": len(fills),
        "network_calls": payload.get("network_calls"),
        "data_write_calls": payload.get("data_write_calls"),
        "ensure_degraded": bool(payload.get("network_calls") or payload.get("data_write_calls")),
        "raw_evidence": str(raw_evidence).replace("\\", "/"),
    }
    _dump(evidence / "q11a_sim" / f"{day}.json", record)
    return record


def run_all(
    *,
    root: Path,
    evidence: Path,
    start_date: str,
    count: int,
    authorized: bool,
) -> Dict[str, Any]:
    """逐日启动独立子进程；任何失败立即停止，不删除失败现场。"""
    if not authorized:
        raise PermissionError("Q11-SIM 需要 --authorized-by-user（用户已批准等效口径）")
    auth = _load(evidence / "q11a_sim_authorization.json")
    if auth.get("approved") is not True:
        raise PermissionError("Q11-SIM 审批证据未标记 approved")
    calendar = load_calendar(ROOT / "data")
    days = _days(calendar, start_date, count)
    ledger: List[Dict[str, Any]] = []
    for day in days:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            "day",
            "--day",
            day,
            "--root",
            str(root),
            "--evidence-dir",
            str(evidence),
        ]
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        (evidence / "logs").mkdir(parents=True, exist_ok=True)
        (evidence / "logs" / f"day_{day}.log").write_text(
            (completed.stdout or "") + "\n---STDERR---\n" + (completed.stderr or ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Q11-SIM {day} 子进程失败 exit={completed.returncode}")
        lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
        if not lines:
            raise RuntimeError(f"Q11-SIM {day} 未输出台账记录")
        record = json.loads(lines[-1])
        ledger.append(record)
        _dump(evidence / "q11a_sim_ledger.json", ledger)
        if not record.get("run_completed"):
            raise RuntimeError(f"Q11-SIM {day} 未完成账户推进: {record}")
    summary = {
        "verdict": "Q11_SIM_PASS",
        "mode": "Q11_SIM",
        "days": days,
        "records": len(ledger),
        "start_account_last_update": ledger[0].get("before_account_last_update"),
        "end_account_last_update": ledger[-1].get("account_last_update"),
        "limitations": auth.get("limitations"),
        "ledger_sha256": _sha256(evidence / "q11a_sim_ledger.json"),
    }
    _dump(evidence / "q11a_sim_summary.json", summary)
    return summary


def finalize(*, evidence: Path, start_date: str, count: int) -> Dict[str, Any]:
    """从逐日原始记录重建完整台账；缺日、重复或不足即失败。"""
    days = _days(load_calendar(ROOT / "data"), start_date, count)
    records = []
    for day in days:
        path = evidence / "q11a_sim" / f"{day}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Q11-SIM 缺 {day} 原始记录: {path}")
        records.append(_load(path))
    if any(not record.get("run_completed") for record in records):
        raise RuntimeError("Q11-SIM 存在未完成日，拒绝生成通过台账")
    auth = _load(evidence / "q11a_sim_authorization.json")
    _dump(evidence / "q11a_sim_ledger.json", records)
    summary = {
        "verdict": "Q11_SIM_PASS",
        "mode": "Q11_SIM",
        "days": days,
        "records": len(records),
        "start_account_last_update": records[0].get("before_account_last_update"),
        "end_account_last_update": records[-1].get("account_last_update"),
        "total_fills": sum(int(record.get("fill_count") or 0) for record in records),
        "limitations": auth.get("limitations"),
        "ledger_sha256": _sha256(evidence / "q11a_sim_ledger.json"),
    }
    _dump(evidence / "q11a_sim_summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run-all", "day", "finalize"), default="run-all")
    parser.add_argument("--source-paper", default=str(ROOT / "data" / "paper"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--start-date", default="20260703")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--day")
    parser.add_argument("--authorized-by-user", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    evidence = Path(args.evidence_dir)
    if args.mode == "prepare":
        result = prepare(
            source_paper=Path(args.source_paper),
            target_paper=root,
            evidence=evidence,
            start_date=args.start_date,
            count=args.days,
        )
    elif args.mode == "day":
        if not args.day:
            raise ValueError("--mode day 需要 --day")
        record = run_day(root=root, day=args.day, evidence=evidence)
        print(json.dumps(record, ensure_ascii=False, default=str))
        return 0 if record["run_completed"] else 2
    elif args.mode == "finalize":
        result = finalize(evidence=evidence, start_date=args.start_date, count=args.days)
    else:
        result = run_all(
            root=root,
            evidence=evidence,
            start_date=args.start_date,
            count=args.days,
            authorized=args.authorized_by_user,
        )
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
