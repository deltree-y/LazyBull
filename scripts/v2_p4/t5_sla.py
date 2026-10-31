# -*- coding: utf-8 -*-
"""v2 P4 T5 门 6 真实日历 SLA 逐日登记与窗口判定（阶段专用）。"""

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4.t5_daily_evidence import (  # noqa: E402
    SLA_REQUIRED_STEPS,
    load_window_records,
)

NON_SYSTEM_REASONS = frozenset(
    {
        "data_source_failure",
        "network_instability",
        "tushare_rate_limit",
        "power_outage",
        "authorization_wait",
    }
)
SYSTEM_REASONS = frozenset(
    {"host_process_crash", "v2_implementation_defect", "readiness_false_judgment"}
)
MAX_ELAPSED_SECONDS = 3600
#: D4→SLA 步骤映射的唯一来源（t5_daily_evidence），此处只做别名保持既有调用方。
REQUIRED_STEPS = SLA_REQUIRED_STEPS


def evaluate_day(record: Mapping[str, Any]) -> Dict[str, Any]:
    """校验单日完整台账并计算 21:00/60min/原因分类，不信任调用方预算 verdict。"""
    day = str(record.get("trade_date") or "")
    errors: List[str] = []
    try:
        day_start = datetime.strptime(day, "%Y%m%d")
    except ValueError:
        return {"verdict": "FAIL", "errors": ["trade_date 非法"], "system_failure": True}
    try:
        started = datetime.fromisoformat(str(record["started_at"]))
        completed = datetime.fromisoformat(str(record["completed_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        return {"verdict": "FAIL", "errors": [f"起止时间非法: {exc}"], "system_failure": True}
    elapsed = (completed - started).total_seconds()
    if elapsed < 0:
        errors.append("completed_at 早于 started_at")
    on_time = completed <= day_start + timedelta(hours=21)
    within_budget = 0 <= elapsed <= MAX_ELAPSED_SECONDS
    steps = record.get("steps")
    if not isinstance(steps, Mapping):
        errors.append("steps 必须为逐步耗时对象")
        steps = {}
    for step in REQUIRED_STEPS:
        raw = steps.get(step)
        if not isinstance(raw, Mapping):
            errors.append(f"缺步骤台账: {step}")
            continue
        duration = raw.get("elapsed_seconds")
        if not isinstance(duration, (int, float)) or duration < 0:
            errors.append(f"步骤耗时非法: {step}")
        retries = raw.get("retries")
        if not isinstance(retries, int) or retries < 0:
            errors.append(f"步骤重试计数非法: {step}")
    system_reason = record.get("system_reason")
    non_system_reason = record.get("non_system_reason")
    if system_reason not in (None, "") and system_reason not in SYSTEM_REASONS:
        errors.append(f"未知系统原因: {system_reason}")
    if non_system_reason not in (None, "") and non_system_reason not in NON_SYSTEM_REASONS:
        errors.append(f"未知非系统原因: {non_system_reason}")
    if system_reason not in (None, "") and non_system_reason not in (None, ""):
        errors.append("系统/非系统原因不得同时登记")
    if not record.get("actual_calendar_day", False):
        errors.append("历史回放/补跑不得计入真实 SLA")
    if record.get("drill", False):
        errors.append("演练记录不得计入真实 SLA/门 5/6 窗口")
    if (
        not record.get("run_completed", False)
        and system_reason in (None, "")
        and non_system_reason in (None, "")
    ):
        errors.append("未完成日必须登记系统或非系统原因")
    gaps = record.get("gaps")
    if not isinstance(gaps, list):
        errors.append("gaps 必须为列表（无缺口=[]）")
    has_exemption_reason = non_system_reason in NON_SYSTEM_REASONS
    system_failure = (
        bool(errors)
        or system_reason not in (None, "")
        or ((not on_time or not within_budget) and not has_exemption_reason)
    )
    exempt = not system_failure and has_exemption_reason
    return {
        "trade_date": day,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": elapsed,
        "on_time": on_time,
        "within_60min": within_budget,
        "system_failure": system_failure,
        "exempt": exempt,
        "authorization_wait": non_system_reason == "authorization_wait",
        "errors": errors,
        "verdict": "FAIL" if system_failure else ("EXEMPT" if exempt else "PASS"),
    }


def evaluate_window(
    records: Sequence[Mapping[str, Any]], calendar: Sequence[str]
) -> Dict[str, Any]:
    """窗口=max(10 交易日,14 自然日)；系统原因清零重计，非系统豁免留痕。"""
    days = evaluate_many(records)
    streak = 0
    resets: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    for item in days:
        if item["system_failure"]:
            if current:
                resets.append({"date": item["trade_date"], "discarded_days": len(current)})
            streak, current = 0, []
            continue
        streak += 1
        current.append(item)
    calendar_days = 0
    if days:
        start = datetime.strptime(days[0]["trade_date"], "%Y%m%d")
        end = datetime.strptime(days[-1]["trade_date"], "%Y%m%d")
        calendar_days = (end - start).days + 1
    current_calendar_days = 0
    if current:
        start = datetime.strptime(current[0]["trade_date"], "%Y%m%d")
        end = datetime.strptime(current[-1]["trade_date"], "%Y%m%d")
        current_calendar_days = (end - start).days + 1
    observed_days = [item["trade_date"] for item in current]
    calendar_positions = [calendar.index(day) for day in observed_days if day in calendar]
    consecutive = len(calendar_positions) == len(observed_days) and (
        not calendar_positions
        or calendar_positions
        == list(range(calendar_positions[0], calendar_positions[0] + len(calendar_positions)))
    )
    on_time_rate = (
        sum(
            bool(item["on_time"] and item["within_60min"]) or bool(item["exempt"])
            for item in current
        )
        / len(current)
        if current
        else 0.0
    )
    passed = bool(
        len(current) >= 10
        and current_calendar_days >= 14
        and consecutive
        and on_time_rate == 1.0
        and not any(item["authorization_wait"] for item in current)
    )
    return {
        "verdict": "PASS" if passed else "IN_PROGRESS",
        "required": {
            "trading_days": 10,
            "calendar_days": 14,
            "on_time_rate": 1.0,
            "max_elapsed_seconds": 3600,
        },
        "observed_trading_days_total": len(days),
        "current_streak_days": len(current),
        # 门 5 与门 6 必须使用同一窗口日期集合（收口据此对齐逐日比较产物）。
        "current_streak_dates": observed_days,
        "calendar_days_total": calendar_days,
        "current_streak_calendar_days": current_calendar_days,
        "current_streak_consecutive": consecutive,
        "current_on_time_rate": on_time_rate,
        "system_reset_events": resets,
        "authorization_wait_days": [
            item["trade_date"] for item in days if item["authorization_wait"]
        ],
        "days": days,
        "historical_replay_accepted": False,
    }


def evaluate_many(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    evaluated = [evaluate_day(record) for record in records]
    days = [item["trade_date"] for item in evaluated]
    if len(days) != len(set(days)):
        raise ValueError("SLA 台账存在重复交易日")
    return sorted(evaluated, key=lambda item: item["trade_date"])


def evaluate_window_dir(directory: Path, calendar: Sequence[str]) -> Dict[str, Any]:
    """逐日目录（每文件单条 record）→ 平坦窗口判定；形态非法即报错。"""
    return evaluate_window(load_window_records(Path(directory)), calendar)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="逐日 SLA JSON 目录，或累计 list 文件")
    parser.add_argument("--calendar", required=True, help="交易日 JSON list")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    ledger = Path(args.ledger)
    if ledger.is_dir():
        records = load_window_records(ledger)
    else:
        payload = json.loads(ledger.read_text(encoding="utf-8"))
        records = payload.get("records") if isinstance(payload, dict) else payload
    calendar = json.loads(Path(args.calendar).read_text(encoding="utf-8"))
    result = evaluate_window(records, calendar)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "streak": result["current_streak_days"]}))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
