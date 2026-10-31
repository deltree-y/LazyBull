# -*- coding: utf-8 -*-
"""T5 收口总账的状态升级边界测试。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_closeout as closeout  # noqa: E402


def _setup(monkeypatch, tmp_path: Path, *, gate1_pass: bool = False) -> Path:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    account = tmp_path / "data" / "paper" / "state" / "account.json"
    account.parent.mkdir(parents=True)
    account.write_bytes(b"account")
    gate1 = {
        "verdict": "PASS" if gate1_pass else "BLOCKED_Q11A",
        "gate1_frozen": gate1_pass,
        "data_state": {
            "data_state_id": "state",
            "sources": {"cs_train": "20260702"},
            "labels": {},
        },
        "account": {"sha256": hashlib.sha256(b"account").hexdigest()},
        "q11a_restart": {"observed_days": 10 if gate1_pass else 0},
    }
    (evidence / "verification_summary.json").write_text(json.dumps(gate1), encoding="utf-8")
    monkeypatch.setattr(closeout, "ROOT", tmp_path)
    monkeypatch.setattr(
        closeout,
        "capture_data_state",
        lambda root: {"data_state_id": "state", "cs_train_latest": "20260702"},
    )
    monkeypatch.setattr(closeout, "compute_worktree_digest", lambda root: {"sha256": "worktree"})
    monkeypatch.setattr(closeout, "build_file_inventory", lambda root: [])
    monkeypatch.setattr(
        closeout,
        "PaperStorage",
        lambda: type("Storage", (), {"load_config": lambda self: {"engine": "legacy"}})(),
    )
    return evidence


def test_report_parser_vetoes_failed_count_and_incomplete_complexity(tmp_path: Path) -> None:
    test_report = tmp_path / "test.json"
    test_report.write_text(
        json.dumps({"verdict": "PASS", "count": 5, "failed": 1}), encoding="utf-8"
    )
    parsed = closeout._test_report(test_report, label="focused")
    assert parsed["status"] == "FAIL"
    assert parsed["failed"] == 1

    complexity = tmp_path / "complexity.json"
    complexity.write_text(json.dumps({"paths": {}}), encoding="utf-8")
    assert closeout._complexity_report(complexity)["status"] == "FAIL"


def test_data_state_view_accepts_gate1_string_and_d4_dict_sources() -> None:
    gate1 = {
        "data_state": {
            "data_state_id": "frozen",
            "sources": {
                "raw_daily": "20261026",
                "clean_daily": "20261026",
                "cs_train": "20260702",
                "cs_infer": "20261026",
                "panel": "20261026",
            },
            "labels": {"y_ret_5": {"sealed_through": "20260917"}},
        }
    }
    current = {
        "data_state_id": "current",
        "sources": {
            "raw_daily": {"max": "20261030"},
            "clean_daily": {"max": "20261030"},
            "cs_train": {"max": "20260702"},
            "cs_infer": {"max": "20261030"},
            "panel": {"max": "20261030"},
        },
        "labels": {"y_ret_5": {"sealed_through": "20260917"}},
    }
    view = closeout._data_state_view(gate1, current)
    assert view["frozen_start_data_state_id"] == "frozen"
    assert view["current_data_state_id"] == "current"
    assert view["cs_train_frozen_unchanged"] is True
    assert set(view["daily_legit_updates"]) == {
        "raw_daily", "clean_daily", "cs_infer", "panel"
    }
    assert view["frozen_labels_unchanged"] is True


def test_closeout_sim_account_baseline_is_explicit_na(monkeypatch, tmp_path: Path) -> None:
    evidence = _setup(monkeypatch, tmp_path, gate1_pass=True)
    payload = json.loads((evidence / "verification_summary.json").read_text(encoding="utf-8"))
    payload["gate1_mode"] = "Q11_SIM"
    (evidence / "verification_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    closeout.closeout(evidence)
    containment = json.loads(
        (evidence / "containment" / "summary.json").read_text(encoding="utf-8")
    )
    assert containment["account_still_baseline"] is None
    assert "隔离账户" in containment["account_baseline_reason"]


def test_closeout_empty_real_window_is_blocked(monkeypatch, tmp_path: Path) -> None:
    evidence = _setup(monkeypatch, tmp_path)
    result = closeout.closeout(evidence)
    assert result["verdict"] == "BLOCKED"
    assert result["gates"]["gate1"] == "BLOCKED_Q11A"
    assert result["gates"]["gate5"] == "NOT_STARTED_OR_INCOMPLETE"
    assert result["real_shadow_days"] == 0
    assert result["actual_cost"]["full_pytest"]["status"] == "MISSING"
    assert result["statuses"]["technical_conditions_ready"] is False


def _runs_batch(tmp_path: Path, run_id: str) -> Path:
    batch = tmp_path / "data" / "runs" / f"p4_t5_{run_id}_paper_shadow"
    batch.mkdir(parents=True)
    (batch / "batch_meta.json").write_text(
        json.dumps(
            {
                "batch_id": f"p4_t5_{run_id}_paper_shadow",
                "host_mode": "paper_shadow",
                "meta_notes": {
                    "shadow_days": "10",
                    "shadow_window": "20260717~20260730",
                    "evidence_mode": "SHADOW_SIM",
                    "sla_mode": "SLA_SIM",
                },
            }
        ),
        encoding="utf-8",
    )
    return batch


def test_closeout_all_real_gates_ready_but_not_committed(monkeypatch, tmp_path: Path) -> None:
    evidence = _setup(monkeypatch, tmp_path, gate1_pass=True)
    (evidence / "gate5").mkdir()
    for index in range(10):
        (evidence / "gate5" / f"daily_202610{index:02d}.json").write_text(
            json.dumps({"verdict": "PASS", "differences": []}), encoding="utf-8"
        )
    (evidence / "sla").mkdir()
    (evidence / "sla" / "window.json").write_text(json.dumps({"verdict": "PASS"}))
    (evidence / "gate7").mkdir()
    (evidence / "gate7" / "rehearsal_result.json").write_text(
        json.dumps({"verdict": "PASS", "synthetic_only": False})
    )
    batch = _runs_batch(tmp_path, evidence.name)
    monkeypatch.setattr(closeout, "load_runs_batch", lambda path: object())
    result = closeout.closeout(evidence, runs_batches=[batch])
    # R2-04：缺全量/复杂度/containment 时门 8 必须否决，不得 READY。
    assert result["verdict"] == "BLOCKED"
    assert result["gates"]["gate8"] == "BLOCKED"
    assert result["production_switch_changed"] is False
    assert "NOT_COMMITTED" in result["commit_status"]
    assert result["runs_batches"][0]["loaded"] is True
    assert any("门 8" in reason for reason in result["blocked_reasons"])


def test_closeout_gate8_all_pass_and_containment_clean_is_ready(
    monkeypatch, tmp_path: Path
) -> None:
    evidence = _setup(monkeypatch, tmp_path, gate1_pass=True)
    (evidence / "gate5").mkdir()
    for index in range(10):
        (evidence / "gate5" / f"daily_202610{index:02d}.json").write_text(
            json.dumps({"verdict": "PASS", "differences": []}), encoding="utf-8"
        )
    (evidence / "sla").mkdir()
    (evidence / "sla" / "window.json").write_text(json.dumps({"verdict": "PASS"}))
    (evidence / "gate7").mkdir()
    (evidence / "gate7" / "real_rehearsal_result.json").write_text(
        json.dumps({"verdict": "PASS", "synthetic_only": False, "items_incomplete": []})
    )
    (evidence / "containment").mkdir()
    (evidence / "containment" / "d1.json").write_text(
        json.dumps({"date": "20261001", "data_state_decision_phase_unchanged": True})
    )
    monkeypatch.setattr(
        closeout,
        "_test_report",
        lambda path, label: {"status": "PASS", "label": label, "count": 1, "failed": 0},
    )
    monkeypatch.setattr(
        closeout, "_complexity_report", lambda path: {"status": "PASS", "paths": {}}
    )
    batch = _runs_batch(tmp_path, evidence.name)
    monkeypatch.setattr(closeout, "load_runs_batch", lambda path: object())
    result = closeout.closeout(evidence, runs_batches=[batch])
    assert result["gates"]["gate8"] == "PASS"
    assert result["verdict"] == "READY_FOR_FINAL_AUTHORIZATION"
    assert result["blocked_reasons"] == []


def test_closeout_containment_violation_blocks_gate8(monkeypatch, tmp_path: Path) -> None:
    evidence = _setup(monkeypatch, tmp_path, gate1_pass=True)
    (evidence / "gate5").mkdir()
    for index in range(10):
        (evidence / "gate5" / f"daily_202610{index:02d}.json").write_text(
            json.dumps({"verdict": "PASS", "differences": []}), encoding="utf-8"
        )
    (evidence / "sla").mkdir()
    (evidence / "sla" / "window.json").write_text(json.dumps({"verdict": "PASS"}))
    (evidence / "gate7").mkdir()
    (evidence / "gate7" / "real_rehearsal_result.json").write_text(
        json.dumps({"verdict": "PASS", "synthetic_only": False, "items_incomplete": []})
    )
    (evidence / "containment").mkdir()
    (evidence / "containment" / "bad.json").write_text(
        json.dumps(
            {
                "date": "20261020",
                "data_state_decision_phase_unchanged": False,
                "shadow_arms_wrote_production": True,
            }
        )
    )
    monkeypatch.setattr(
        closeout,
        "_test_report",
        lambda path, label: {"status": "PASS", "label": label, "count": 1, "failed": 0},
    )
    monkeypatch.setattr(
        closeout, "_complexity_report", lambda path: {"status": "PASS", "paths": {}}
    )
    batch = _runs_batch(tmp_path, evidence.name)
    monkeypatch.setattr(closeout, "load_runs_batch", lambda path: object())
    result = closeout.closeout(evidence, runs_batches=[batch])
    assert result["gates"]["gate8"] == "BLOCKED"
    assert result["verdict"] == "BLOCKED"
    assert any("20261020" in reason for reason in result["blocked_reasons"])


def test_closeout_blocks_when_runs_batch_missing(monkeypatch, tmp_path: Path) -> None:
    evidence = _setup(monkeypatch, tmp_path, gate1_pass=True)
    (evidence / "gate5").mkdir()
    for index in range(10):
        (evidence / "gate5" / f"daily_202610{index:02d}.json").write_text(
            json.dumps({"verdict": "PASS", "differences": []}), encoding="utf-8"
        )
    (evidence / "sla").mkdir()
    (evidence / "sla" / "window.json").write_text(json.dumps({"verdict": "PASS"}))
    (evidence / "gate7").mkdir()
    (evidence / "gate7" / "rehearsal_result.json").write_text(
        json.dumps({"verdict": "PASS", "synthetic_only": False})
    )
    result = closeout.closeout(evidence)
    assert result["verdict"] == "BLOCKED"
    assert any("runs 批次" in reason for reason in result["blocked_reasons"])


def test_closeout_never_upgrades_synthetic_gate7(monkeypatch, tmp_path: Path) -> None:
    evidence = _setup(monkeypatch, tmp_path, gate1_pass=True)
    (evidence / "gate7").mkdir()
    (evidence / "gate7" / "rehearsal_result.json").write_text(
        json.dumps({"verdict": "SYNTHETIC_PASS", "synthetic_only": True})
    )
    result = closeout.closeout(evidence)
    assert result["gates"]["gate7"] == "REAL_REHEARSAL_INCOMPLETE"


def test_closeout_gate5_uses_sla_window_dates(monkeypatch, tmp_path: Path) -> None:
    evidence = _setup(monkeypatch, tmp_path, gate1_pass=True)
    monkeypatch.setattr(closeout, "_valid_window_dates", lambda days: True)
    (evidence / "gate5").mkdir()
    window_dates = ["20260717", "20260720"]
    for day in window_dates + ["20260716"]:
        (evidence / "gate5" / f"daily_{day}.json").write_text(
            json.dumps({"verdict": "PASS", "date": day, "differences": []}), encoding="utf-8"
        )
    (evidence / "sla").mkdir()
    (evidence / "sla" / "window.json").write_text(
        json.dumps({"verdict": "IN_PROGRESS", "current_streak_dates": window_dates})
    )
    result = closeout.closeout(evidence)
    assert result["gate5_window"]["window_dates_source"] == "sla_window"
    assert result["gate5_window"]["days_outside_window"] == ["20260716"]
    assert result["gate5_window"]["verdict"] == "NOT_STARTED_OR_INCOMPLETE"


def test_closeout_rejects_nine_day_sla_window_even_all_pass(monkeypatch, tmp_path: Path) -> None:
    evidence = _setup(monkeypatch, tmp_path, gate1_pass=True)
    monkeypatch.setattr(closeout, "_valid_window_dates", lambda days: True)
    (evidence / "gate5").mkdir()
    days = [
        "20260717", "20260720", "20260721", "20260722", "20260723",
        "20260724", "20260727", "20260728", "20260729",
    ]
    for day in days:
        (evidence / "gate5" / f"daily_{day}.json").write_text(
            json.dumps({"verdict": "PASS", "date": day}), encoding="utf-8"
        )
    (evidence / "sla").mkdir()
    (evidence / "sla" / "window.json").write_text(
        json.dumps({"verdict": "PASS", "current_streak_dates": days})
    )
    result = closeout.closeout(evidence)
    assert result["gate5_window"]["covered_days"] == 9
    assert result["gate5_window"]["verdict"] == "NOT_STARTED_OR_INCOMPLETE"
    assert result["gates"]["gate5"] == "NOT_STARTED_OR_INCOMPLETE"


def test_closeout_rejects_duplicate_window_dates(monkeypatch, tmp_path: Path) -> None:
    evidence = _setup(monkeypatch, tmp_path, gate1_pass=True)
    (evidence / "gate5").mkdir()
    (evidence / "gate5" / "daily_20260717.json").write_text(
        json.dumps({"verdict": "PASS", "date": "20260717"}), encoding="utf-8"
    )
    (evidence / "sla").mkdir()
    (evidence / "sla" / "window.json").write_text(
        json.dumps(
            {"verdict": "PASS", "current_streak_dates": ["20260717"] * 10}
        ),
        encoding="utf-8",
    )
    result = closeout.closeout(evidence)
    assert result["gate5_window"]["covered_days"] == 0
    assert result["gate5_window"]["verdict"] == "NOT_STARTED_OR_INCOMPLETE"
    assert "唯一" in str(result["gate5_window"].get("reason"))


def test_closeout_difference_ledger_preserves_unexplained(monkeypatch, tmp_path: Path) -> None:
    evidence = _setup(monkeypatch, tmp_path)
    (evidence / "gate5").mkdir()
    (evidence / "gate5" / "daily_20261030.json").write_text(
        json.dumps(
            {
                "verdict": "FAIL",
                "date": "20261030",
                "differences": [{"category": "unexplained", "path": "x"}],
            }
        ),
        encoding="utf-8",
    )
    closeout.closeout(evidence)
    ledger = json.loads((evidence / "difference_ledger.json").read_text(encoding="utf-8"))
    assert ledger["unexplained"] == 1
    assert ledger["entries"][0]["date"] == "20261030"
