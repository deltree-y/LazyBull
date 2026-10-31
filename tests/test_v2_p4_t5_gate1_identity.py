# -*- coding: utf-8 -*-
"""T5 门 1 身份与 Q11-A 连续性校验；全部使用合成数据。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_gate1_identity as gate1  # noqa: E402


def _records(days: list[str]) -> list[dict[str, object]]:
    return [
        {
            "corrected_date": day,
            "run_completed": True,
            "account_last_update": day,
            "t0_status": "success",
            "ensure_degraded": False,
            "actual_calendar_day": True,
        }
        for day in days
    ]


def test_restart_records_accepts_ten_consecutive_days() -> None:
    days = [
        "20261001",
        "20261002",
        "20261005",
        "20261006",
        "20261007",
        "20261008",
        "20261009",
        "20261012",
        "20261013",
        "20261014",
    ]
    result = gate1.validate_restart_records(_records(days), days)
    assert result["passed"] is True
    assert result["observed_days"] == 10


def test_restart_records_rejects_gap_duplicate_and_stale_account() -> None:
    days = [f"202610{day:02d}" for day in range(1, 11)]
    records = _records(days)
    records[3]["account_last_update"] = "20261003"
    records.append(dict(records[-1]))
    result = gate1.validate_restart_records(records, days)
    assert result["passed"] is False
    assert any("同日重复" in error for error in result["errors"])
    assert any("账户未推进" in error for error in result["errors"])


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ({"run_completed": False}, "run_completed"),
        ({"t0_status": "error: boom"}, "t0_status"),
        ({"t0_status": "not_ready"}, "t0_status"),
        ({"ensure_degraded": True}, "ensure"),
        ({"actual_calendar_day": False}, "历史回放"),
    ],
)
def test_restart_records_rejects_false_success(mutation: dict, expected: str) -> None:
    days = [f"202610{day:02d}" for day in range(1, 11)]
    records = _records(days)
    records[0].update(mutation)
    result = gate1.validate_restart_records(records, days)
    assert result["passed"] is False
    assert any(expected in error for error in result["errors"])


def test_restart_records_requires_exactly_ten_days() -> None:
    days = [f"202610{day:02d}" for day in range(1, 10)]
    result = gate1.validate_restart_records(_records(days), days)
    assert result["passed"] is False
    assert "要求=10" in result["errors"][0]


def test_more_than_ten_records_require_explicit_segment() -> None:
    days = [f"202610{day:02d}" for day in range(1, 13)]
    result = gate1.validate_restart_records(_records(days), days)
    assert result["passed"] is False
    assert any("--restart-segment-end" in error for error in result["errors"])
    assert result["selected_days"] == []


def test_explicit_segment_selects_trailing_ten_consecutive_days() -> None:
    days = [f"202610{day:02d}" for day in range(1, 13)]
    result = gate1.validate_restart_records(_records(days), days, segment_end="20261012")
    assert result["passed"] is True
    assert result["selected_days"] == days[2:]
    assert result["ledger_days"] == 12
    assert result["discarded_days"]["before"] == 2


def test_segment_end_must_exist_in_ledger() -> None:
    days = [f"202610{day:02d}" for day in range(1, 13)]
    result = gate1.validate_restart_records(_records(days), days, segment_end="20261030")
    assert result["passed"] is False
    assert any("不在台账" in error for error in result["errors"])


def test_frozen_account_end_day_must_align_with_segment() -> None:
    days = [f"202610{day:02d}" for day in range(1, 11)]
    result = gate1.validate_restart_records(_records(days), days, account_last_update="20261030")
    assert result["passed"] is False
    assert result["account_aligned"] is False
    assert any("不对齐" in error for error in result["errors"])


def test_empty_status_is_not_a_success_form() -> None:
    days = [f"202610{day:02d}" for day in range(1, 11)]
    records = _records(days)
    records[0]["t0_status"] = ""
    result = gate1.validate_restart_records(records, days)
    assert result["passed"] is False
    assert any("缺 t0_status" in error for error in result["errors"])


def _frozen_result(identity: str = "a") -> dict:
    return {
        "captured_at": "2026-10-30T16:00:00",
        "verdict": "PASS",
        "gate1_frozen": True,
        "frozen_at": "2026-10-30T16:00:00",
        "account": {"sha256": identity, "last_update": "20261030"},
        "data_state": {"data_state_id": "state"},
        "model": {"content_fingerprint": "model"},
        "execution": {"fingerprint": "exec"},
        "q11a_restart": {"observed_days": 10},
        "worktree": {},
        "holdings_validity": {"positions": []},
    }


def test_main_requires_existing_restart_ledger_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="恢复台账不存在"):
        gate1.main(
            [
                "--evidence-dir",
                str(tmp_path / "evidence"),
                "--restart-ledger",
                str(tmp_path / "missing.json"),
            ]
        )


def test_main_freeze_requires_restart_ledger_argument(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="必须显式提供 --restart-ledger"):
        gate1.main(["--evidence-dir", str(tmp_path / "evidence"), "--freeze"])


def _seed_ledger(evidence: Path) -> Path:
    ledger = evidence / "q11a_restart_ledger.json"
    ledger.write_text("[]", encoding="utf-8")
    return ledger


def test_main_keeps_frozen_identity_immutable(monkeypatch, tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    frozen = _frozen_result("a")
    (evidence / "verification_summary.json").write_text(json.dumps(frozen), encoding="utf-8")
    ledger = _seed_ledger(evidence)
    monkeypatch.setattr(gate1, "collect_gate1", lambda **kwargs: _frozen_result("b"))
    with pytest.raises(RuntimeError, match="冻结身份不可变"):
        gate1.main(["--evidence-dir", str(evidence), "--freeze", "--restart-ledger", str(ledger)])
    assert (
        json.loads((evidence / "verification_summary.json").read_text(encoding="utf-8")) == frozen
    )


def test_main_writes_progress_precheck_when_identity_unchanged(monkeypatch, tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    frozen = _frozen_result("a")
    (evidence / "verification_summary.json").write_text(json.dumps(frozen), encoding="utf-8")
    ledger = _seed_ledger(evidence)
    monkeypatch.setattr(gate1, "collect_gate1", lambda **kwargs: _frozen_result("a"))
    code = gate1.main(
        ["--evidence-dir", str(evidence), "--freeze", "--restart-ledger", str(ledger)]
    )
    assert code == 0
    assert (
        json.loads((evidence / "verification_summary.json").read_text(encoding="utf-8")) == frozen
    )
    precheck = json.loads((evidence / "gate1_progress_precheck.json").read_text(encoding="utf-8"))
    assert precheck["reason"].startswith("门 1 已冻结")


def test_restart_segment_flag_is_parsed(tmp_path: Path) -> None:
    args = gate1._parse_args(
        ["--evidence-dir", str(tmp_path), "--restart-segment-end", "20261030", "--freeze"]
    )
    assert args.restart_segment_end == "20261030"
    assert args.freeze is True


def test_assess_holdings_uses_clean_prices_and_marks_validity(tmp_path: Path) -> None:
    day_dir = tmp_path / "clean" / "daily"
    day_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts_code": ["A", "B"],
            "close": [10.0, None],
            "pre_close": [9.0, 8.0],
            "tradable": [1, 1],
            "is_st": [0, 1],
            "is_suspended": [0, 0],
        }
    ).to_parquet(day_dir / "2026-10-26.parquet")
    account = {
        "cash": 100.0,
        "positions": {
            "A": {"shares": 100, "buy_date": "20261001", "buy_pnl_price": 7.0},
            "B": {"shares": 200, "buy_date": "20261001", "buy_pnl_price": 7.0},
        },
    }
    result = gate1.assess_holdings(account, tmp_path)
    assert result["position_count"] == 2
    assert result["valid_count"] == 1
    assert result["positions"][0]["mark_source"] == "clean.daily.close"
    assert result["positions"][1]["mark_source"] == "clean.daily.pre_close"
    assert result["total_value"] == pytest.approx(2700.0)


def test_assess_holdings_keeps_d10_buy_pnl_fallback(tmp_path: Path) -> None:
    day_dir = tmp_path / "clean" / "daily"
    day_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts_code": ["A"],
            "close": [None],
            "pre_close": [None],
            "tradable": [0],
            "is_st": [0],
            "is_suspended": [1],
        }
    ).to_parquet(day_dir / "2026-10-26.parquet")
    account = {"cash": 0.0, "positions": {"A": {"shares": 100, "buy_pnl_price": 6.0}}}
    result = gate1.assess_holdings(account, tmp_path)
    assert result["positions"][0]["mark_source"] == "account.buy_pnl_price"
    assert result["positions"][0]["valid"] is False
    assert result["total_value"] == 600.0


def test_execution_identity_contains_prices_and_neutral_exposure() -> None:
    result = gate1.collect_execution_identity(
        {"buy_price": "close", "sell_price": "open", "exposure_policy": None}
    )
    assert result["config"]["buy_price"] == "close"
    assert result["config"]["sell_price"] == "open"
    assert result["config"]["engine"] == "legacy"
    assert result["config"]["exposure_policy"] is None
    assert len(result["fingerprint"]) == 16


def test_model_identity_hashes_all_three_artifacts(tmp_path: Path) -> None:
    (tmp_path / "v7_model.joblib").write_bytes(b"model")
    (tmp_path / "v7_metadata.json").write_text(
        json.dumps({"model_type": "xgboost", "train_start_date": "20200101"}),
        encoding="utf-8",
    )
    (tmp_path / "v7_features.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    result = gate1.collect_model_identity(tmp_path, 7)
    assert result["version"] == 7
    assert result["feature_count"] == 2
    assert set(result["artifacts"]) == {"model.joblib", "metadata.json", "features.json"}
