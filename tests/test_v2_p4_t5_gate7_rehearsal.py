# -*- coding: utf-8 -*-
"""门 7 合成/真实演练回归；合成结果不得据此判正式门 7 PASS。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4.t5_gate7_rehearsal import GATE7_ITEMS, run_real, run_rehearsal  # noqa: E402


def test_gate7_synthetic_rehearsal_full_sequence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    result = run_rehearsal(tmp_path / "work", evidence)
    assert result["verdict"] == "SYNTHETIC_PASS"
    assert result["synthetic_only"] is True
    assert result["formal_gate7_pass"] is False
    assert result["fills_count"] >= 1
    assert result["delayed_order_present"] is True
    assert result["adjust_cash"] == 90000.0
    assert result["second_writer_fail_fast"] is True
    assert result["rollback_same_day_guard"] is True
    assert result["rollback_next_day_guard"] is True
    persisted = json.loads(
        (evidence / "gate7" / "rehearsal_result.json").read_text(encoding="utf-8")
    )
    assert persisted == result


def test_gate7_rehearsal_refuses_non_empty_work_root(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="工作根必须为空"):
        run_rehearsal(work, tmp_path / "evidence")
    assert (work / "keep.txt").read_text(encoding="utf-8") == "keep"


def _item(tmp_path: Path, key: str, **overrides) -> dict:
    raw_output = tmp_path / f"{key}_output.txt"
    raw_output.write_text("ok", encoding="utf-8")
    input_root = tmp_path / "isolated" / key
    input_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "item": key,
        "executed_at": "2026-10-30T20:00:00",
        "entry": f"python scripts/paper_trade.py {key}",
        "input_root": str(input_root),
        "trade_date": "20261030",
        "result": "PASS",
        "raw_output": str(raw_output),
        "account_summary": {"cash": 1000.0, "positions": 2},
    }
    payload.update(overrides)
    return payload


def _checklist(tmp_path: Path, items: list[dict]) -> Path:
    path = tmp_path / "checklist.json"
    path.write_text(json.dumps({"items": items}), encoding="utf-8")
    return path


def test_gate7_real_marks_missing_items_incomplete(tmp_path: Path) -> None:
    checklist = _checklist(tmp_path, [_item(tmp_path, "run")])
    result = run_real(
        checklist=checklist,
        evidence=tmp_path / "evidence",
        production_paper_root=tmp_path / "production",
    )
    assert result["verdict"] == "INCOMPLETE"
    assert result["synthetic_only"] is False
    assert result["formal_gate7_pass"] is False
    assert any("positions" in item for item in result["items_incomplete"])
    persisted = json.loads(
        (tmp_path / "evidence" / "gate7" / "real_rehearsal_result.json").read_text(encoding="utf-8")
    )
    assert persisted["verdict"] == "INCOMPLETE"


def test_gate7_real_passes_with_all_items_covered(tmp_path: Path) -> None:
    items = [_item(tmp_path, key) for key in GATE7_ITEMS]
    checklist = _checklist(tmp_path, items)
    result = run_real(
        checklist=checklist,
        evidence=tmp_path / "evidence",
        production_paper_root=tmp_path / "production",
    )
    assert result["verdict"] == "PASS"
    assert result["formal_gate7_pass"] is True
    assert result["items_incomplete"] == []
    assert result["production_touched"] is False


def test_gate7_real_rejects_production_root_for_write_items(tmp_path: Path) -> None:
    production = tmp_path / "production"
    production.mkdir()
    items = [_item(tmp_path, key) for key in GATE7_ITEMS if key != "migration_copy"]
    items.append(_item(tmp_path, "migration_copy", input_root=str(production)))
    checklist = _checklist(tmp_path, items)
    result = run_real(
        checklist=checklist, evidence=tmp_path / "evidence", production_paper_root=production
    )
    assert result["verdict"] == "INCOMPLETE"
    assert result["production_write_violations"] == ["migration_copy"]
    assert any("生产纸面根" in item for item in result["items_incomplete"])


def test_gate7_real_requires_raw_output_file(tmp_path: Path) -> None:
    items = [_item(tmp_path, key) for key in GATE7_ITEMS]
    items[0]["raw_output"] = str(tmp_path / "missing.txt")
    checklist = _checklist(tmp_path, items)
    result = run_real(
        checklist=checklist,
        evidence=tmp_path / "evidence",
        production_paper_root=tmp_path / "production",
    )
    assert result["verdict"] == "INCOMPLETE"
    assert any("原始输出不存在" in item for item in result["items_incomplete"])


def test_gate7_real_rejects_malformed_checklist(tmp_path: Path) -> None:
    path = tmp_path / "checklist.json"
    path.write_text(json.dumps({"items": [{"item": "run"}]}), encoding="utf-8")
    result = run_real(
        checklist=path,
        evidence=tmp_path / "evidence",
        production_paper_root=tmp_path / "production",
    )
    assert result["verdict"] == "INCOMPLETE"
    assert result["items_incomplete"]


def test_gate7_real_allows_user_waiver_only_for_device_smb(tmp_path: Path) -> None:
    items = [_item(tmp_path, key) for key in GATE7_ITEMS]
    for item in items:
        if item["item"] == "device_smb_read":
            item["result"] = "WAIVED"
            item["waiver"] = {
                "approved_by_user": True,
                "approved_at": "2026-10-30",
                "approved_text": "门7的SMB项可以忽略",
            }
    result = run_real(
        checklist=_checklist(tmp_path, items),
        evidence=tmp_path / "evidence",
        production_paper_root=tmp_path / "production",
    )
    assert result["verdict"] == "PASS"
    assert result["items_waived"] == ["device_smb_read"]
    assert result["items_incomplete"] == []


def test_gate7_real_rejects_waiver_without_user_approval_or_wrong_item(tmp_path: Path) -> None:
    items = [_item(tmp_path, key) for key in GATE7_ITEMS]
    for item in items:
        if item["item"] == "device_smb_read":
            item["result"] = "WAIVED"
    result = run_real(
        checklist=_checklist(tmp_path, items),
        evidence=tmp_path / "evidence",
        production_paper_root=tmp_path / "production",
    )
    assert result["verdict"] == "INCOMPLETE"
    assert any("用户批准" in item for item in result["items_incomplete"])

    items = [_item(tmp_path, key) for key in GATE7_ITEMS]
    for item in items:
        if item["item"] == "run":
            item["result"] = "WAIVED"
            item["waiver"] = {"approved_by_user": True, "approved_at": "2026-10-30"}
    result = run_real(
        checklist=_checklist(tmp_path, items),
        evidence=tmp_path / "evidence2",
        production_paper_root=tmp_path / "production",
    )
    assert result["verdict"] == "INCOMPLETE"
    assert any("仅 device_smb_read" in item for item in result["items_incomplete"])
