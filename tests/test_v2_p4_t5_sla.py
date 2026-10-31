# -*- coding: utf-8 -*-
"""门 6 SLA 合成测试；不发起真实日跑。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_sla as sla  # noqa: E402

DAYS = [
    "20261001", "20261002", "20261005", "20261006", "20261007",
    "20261008", "20261009", "20261012", "20261013", "20261014",
]


def _record(day: str, **overrides) -> dict:
    payload = {
        "trade_date": day,
        "started_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T16:00:00",
        "completed_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T16:20:00",
        "steps": {
            step: {"elapsed_seconds": 10.0, "retries": 0}
            for step in sla.REQUIRED_STEPS
        },
        "gaps": [],
        "run_completed": True,
        "actual_calendar_day": True,
        "system_reason": None,
        "non_system_reason": None,
    }
    payload.update(overrides)
    return payload


def test_day_passes_with_full_step_ledger() -> None:
    result = sla.evaluate_day(_record("20261030"))
    assert result["verdict"] == "PASS"
    assert result["on_time"] is True
    assert result["within_60min"] is True


def test_day_rejects_missing_step_and_retry_shape() -> None:
    record = _record("20261030")
    del record["steps"]["ensure_raw"]
    record["steps"]["arms"]["retries"] = -1
    result = sla.evaluate_day(record)
    assert result["system_failure"] is True
    assert any("ensure_raw" in error for error in result["errors"])
    assert any("重试" in error for error in result["errors"])


def test_late_without_exemption_is_system_failure() -> None:
    result = sla.evaluate_day(_record("20261030", completed_at="2026-10-30T21:00:01"))
    assert result["verdict"] == "FAIL"
    assert result["system_failure"] is True


def test_network_failure_is_exempt_but_registered() -> None:
    result = sla.evaluate_day(
        _record(
            "20261030",
            completed_at="2026-10-30T21:30:00",
            run_completed=False,
            non_system_reason="network_instability",
            gaps=["raw/daily"],
        )
    )
    assert result["verdict"] == "EXEMPT"
    assert result["system_failure"] is False


def test_historical_replay_never_counts() -> None:
    result = sla.evaluate_day(_record("20261030", actual_calendar_day=False))
    assert result["system_failure"] is True
    assert any("历史回放" in error for error in result["errors"])


def test_window_passes_ten_days_spanning_fourteen_calendar_days() -> None:
    result = sla.evaluate_window([_record(day) for day in DAYS], DAYS)
    assert result["verdict"] == "PASS"
    assert result["current_streak_days"] == 10
    assert result["current_streak_calendar_days"] == 14
    assert result["current_on_time_rate"] == 1.0


def test_system_reason_resets_streak() -> None:
    records = [_record(day) for day in DAYS]
    records[4]["system_reason"] = "host_process_crash"
    records[4]["run_completed"] = False
    result = sla.evaluate_window(records, DAYS)
    assert result["verdict"] == "IN_PROGRESS"
    assert result["current_streak_days"] == 5
    assert result["system_reset_events"][0]["discarded_days"] == 4


def test_reset_window_cannot_borrow_calendar_days_before_reset() -> None:
    records = [_record(day) for day in DAYS]
    records[1]["system_reason"] = "v2_implementation_defect"
    result = sla.evaluate_window(records, DAYS)
    assert result["calendar_days_total"] == 14
    assert result["current_streak_calendar_days"] == 10
    assert result["verdict"] == "IN_PROGRESS"


def test_authorization_wait_is_zero_occurrence_goal() -> None:
    records = [_record(day) for day in DAYS]
    records[0]["non_system_reason"] = "authorization_wait"
    result = sla.evaluate_window(records, DAYS)
    assert result["verdict"] == "IN_PROGRESS"
    assert result["authorization_wait_days"] == [DAYS[0]]


def test_duplicate_days_are_rejected() -> None:
    with pytest.raises(ValueError, match="重复交易日"):
        sla.evaluate_many([_record("20261030"), _record("20261030")])
