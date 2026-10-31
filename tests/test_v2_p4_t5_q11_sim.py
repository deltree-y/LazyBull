# -*- coding: utf-8 -*-
"""Q11-SIM 用户批准等效口径与门 1 SIM 判据测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_gate1_identity as gate1  # noqa: E402
from scripts.v2_p4 import t5_q11_sim as sim  # noqa: E402

DAYS = [
    "20260703", "20260706", "20260707", "20260708", "20260709",
    "20260710", "20260713", "20260714", "20260715", "20260716",
]


def _sim_records(evidence_rel: str = "ev") -> list:
    records = []
    for day in DAYS:
        records.append(
            {
                "trade_date": day,
                "corrected_date": day,
                "simulation": True,
                "source_kind": "historical_sim",
                "actual_calendar_day": False,
                "restart_verified": True,
                "run_completed": True,
                "daily_run_record": f"{evidence_rel}/{day}.json",
                "before_account_last_update": "20260603",
                "account_last_update": "20260706",
                "account_advanced": day == "20260706",
                "fill_count": 30 if day == "20260706" else 0,
                "t0_status": "success" if day == "20260706" else "not_rebalance_day",
                "ensure_degraded": False,
                "authorization": "user approved Q11-SIM",
                "raw_evidence": f"{evidence_rel}/{day}.json",
            }
        )
    return records


def _write_evidence(root: Path, rel: str = "ev") -> None:
    path = root / rel
    path.mkdir(parents=True, exist_ok=True)
    for day in DAYS:
        (path / f"{day}.json").write_text("{}", encoding="utf-8")


def test_sim_validator_accepts_ten_consecutive_historical_days(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gate1, "ROOT", tmp_path)
    _write_evidence(tmp_path)
    result = gate1.validate_restart_records(
        _sim_records(), DAYS, account_last_update="20260706", mode="sim"
    )
    assert result["passed"] is True
    assert result["evidence_mode"] == "Q11_SIM"
    assert result["simulation_not_live_evidence"] is True
    assert result["limitations"]


def test_sim_rejects_unmarked_history(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gate1, "ROOT", tmp_path)
    _write_evidence(tmp_path)
    records = _sim_records()
    records[0]["simulation"] = False
    result = gate1.validate_restart_records(
        records, DAYS, account_last_update="20260706", mode="sim"
    )
    assert result["passed"] is False
    assert any("simulation=true" in error for error in result["errors"])


def test_live_mode_still_rejects_historical_sim(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gate1, "ROOT", tmp_path)
    _write_evidence(tmp_path)
    result = gate1.validate_restart_records(
        _sim_records(), DAYS, account_last_update="20260706", mode="live"
    )
    assert result["passed"] is False
    assert any("非真实日历日跑" in error for error in result["errors"])


def test_sim_days_selects_exact_consecutive_window() -> None:
    calendar = ["20260702"] + DAYS + ["20260717"]
    assert sim._days(calendar, "20260703", 10) == DAYS
    with pytest.raises(ValueError, match="不足"):
        sim._days(calendar, "20260703", 12)


def test_prepare_copies_account_and_writes_authorization(tmp_path: Path) -> None:
    source = tmp_path / "paper"
    (source / "state").mkdir(parents=True)
    (source / "state" / "account.json").write_text(
        json.dumps({"cash": 1000.0, "last_update": "20260603", "positions": {}}),
        encoding="utf-8",
    )
    target = tmp_path / "sim" / "paper"
    evidence = tmp_path / "evidence"
    monkeypatch_calendar = ["20260702"] + DAYS
    original = sim.load_calendar
    sim.load_calendar = lambda root: monkeypatch_calendar
    try:
        start = sim.prepare(
            source_paper=source,
            target_paper=target,
            evidence=evidence,
            start_date="20260703",
            count=10,
        )
    finally:
        sim.load_calendar = original
    assert (target / "state" / "account.json").is_file()
    assert start["days"] == DAYS
    auth = json.loads((evidence / "q11a_sim_authorization.json").read_text(encoding="utf-8"))
    assert auth["approved"] is True
    assert "not_live_network_evidence" in auth["limitations"]


def test_finalize_rejects_missing_day(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "q11a_sim_authorization.json").write_text(
        json.dumps({"approved": True, "limitations": []}), encoding="utf-8"
    )
    original = sim.load_calendar
    sim.load_calendar = lambda root: ["20260702"] + DAYS
    try:
        with pytest.raises(FileNotFoundError, match="缺 .* 原始记录"):
            sim.finalize(evidence=evidence, start_date="20260703", count=10)
    finally:
        sim.load_calendar = original
