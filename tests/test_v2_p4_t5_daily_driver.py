# -*- coding: utf-8 -*-
"""T5 日级驱动的 dry-run、台账形态与失败留痕测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_daily_driver as driver  # noqa: E402
from scripts.v2_p4 import t5_daily_evidence as evidence_io  # noqa: E402


def _record(day: str, **overrides) -> dict:
    payload = {
        "trade_date": day,
        "started_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T16:00:00",
        "completed_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T16:20:00",
        "steps": {
            step: {"elapsed_seconds": 1.0, "retries": 0} for step in evidence_io.SLA_REQUIRED_STEPS
        },
        "gaps": [],
        "run_completed": True,
        "actual_calendar_day": True,
    }
    payload.update(overrides)
    return payload


def test_restart_plan_puts_d4_before_legacy_run() -> None:
    plan = driver.build_daily_plan("restart", "20261030", "t5_test")
    assert len(plan) == 7
    assert "daily_data_coordinator" in plan[0]
    assert plan[-1].endswith("paper_trade.py run --trade-date 20261030")


def test_shadow_plan_has_two_validation_arms_and_compare() -> None:
    plan = driver.build_daily_plan("shadow", "20261030", "t5_test")
    assert "legacy_neutral.run" in plan[-3]
    assert "v2_shadow.run_t1" in plan[-2]
    assert "compare(v2_shadow, legacy_neutral)" in plan[-1]
    assert not any("paper_trade.py run" in step for step in plan)


def test_preflight_plan_is_read_only_scan() -> None:
    plan = driver.build_daily_plan("preflight", "20261030", "t5_test")
    assert len(plan) == 1
    assert "--mode preflight" in plan[0]
    assert "daily_data_coordinator" not in plan[0]


def test_build_plan_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="未知日跑模式"):
        driver.build_daily_plan("replay", "20261030", "t5_test")


def test_cumulative_ledger_rejects_duplicate_day(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    evidence_io.append_cumulative_record(path, _record("20261030"))
    with pytest.raises(evidence_io.EvidenceFormatError, match="拒绝重复累计"):
        evidence_io.append_cumulative_record(path, _record("20261030", run_completed=False))
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(persisted, list) and len(persisted) == 1
    assert persisted[0]["run_completed"] is True


def test_day_file_is_single_record_while_ledger_is_list(tmp_path: Path) -> None:
    day_path = tmp_path / "sla" / "daily" / "20261030.json"
    evidence_io.write_day_record(day_path, _record("20261030"))
    assert isinstance(json.loads(day_path.read_text(encoding="utf-8")), dict)
    flat = evidence_io.load_window_records(day_path.parent)
    assert [item["trade_date"] for item in flat] == ["20261030"]
    assert all(not isinstance(item, list) for item in flat)


def test_window_loader_rejects_nested_list(tmp_path: Path) -> None:
    path = tmp_path / "sla" / "daily" / "20261030.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([[_record("20261030")]]), encoding="utf-8")
    with pytest.raises(evidence_io.EvidenceFormatError, match="单日台账"):
        evidence_io.load_window_records(path.parent)


def test_window_loader_converts_legacy_single_list(tmp_path: Path) -> None:
    path = tmp_path / "sla" / "daily" / "20261030.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([_record("20261030")]), encoding="utf-8")
    flat = evidence_io.load_window_records(path.parent)
    assert len(flat) == 1
    assert flat[0]["_source_format"] == evidence_io.SOURCE_LEGACY_WRAPPED


def test_read_orders_maps_all_frozen_fields(tmp_path: Path) -> None:
    root = tmp_path / "paper"
    (root / "instructions").mkdir(parents=True)
    pd.DataFrame(
        {
            "action": ["buy", "sell"],
            "ts_code": ["A", "B"],
            "shares": [100, 200],
            "buy_price_type": ["close", "close"],
            "sell_price_type": ["open", "open"],
            "reason": ["调仓买入", "持有期到期"],
        }
    ).to_parquet(root / "instructions" / "20261102.parquet")
    orders = driver.read_orders(root, "20261030", ["20261030", "20261102"])
    assert orders[0] == {
        "date": "20261030",
        "phase": "T0",
        "action": "buy",
        "ts_code": "A",
        "shares": 100,
        "price_basis": "close",
        "reason": "调仓买入",
        "execution_date": "20261102",
    }
    assert orders[1]["price_basis"] == "open"


def test_read_orders_missing_instruction_file_is_empty(tmp_path: Path) -> None:
    assert driver.read_orders(tmp_path, "20261030", ["20261030", "20261102"]) == []


def test_next_day_requires_future_calendar_day() -> None:
    with pytest.raises(RuntimeError, match="次日"):
        driver._next_day(["20261030"], "20261030")


def test_main_dry_run_does_not_require_authorization(capsys, tmp_path: Path) -> None:
    code = driver.main(
        [
            "--mode",
            "shadow",
            "--trade-date",
            "20261030",
            "--run-id",
            "t5_test",
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ]
    )
    assert code == 0
    assert "v2_shadow.run_t1" in capsys.readouterr().out
    assert not (tmp_path / "evidence").exists()


def test_main_execute_rejects_non_current_date_before_any_work(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="当前真实日历交易日"):
        driver.main(
            [
                "--mode",
                "restart",
                "--trade-date",
                "20260101",
                "--run-id",
                "t5_test",
                "--evidence-dir",
                str(tmp_path),
                "--execute",
                "--authorized-minutes",
                "5",
            ]
        )


def test_main_execute_rejects_non_trading_day_before_d4(monkeypatch, tmp_path: Path) -> None:
    real_datetime = driver.datetime

    class _DateTime:
        @staticmethod
        def now():  # noqa: ANN202
            return real_datetime(2026, 10, 31)

    monkeypatch.setattr(driver, "datetime", _DateTime)
    monkeypatch.setattr(driver, "load_calendar", lambda root: ["20261030"])
    with pytest.raises(ValueError, match="非交易日"):
        driver.main(
            [
                "--mode",
                "restart",
                "--trade-date",
                "20261031",
                "--run-id",
                "t5_test",
                "--evidence-dir",
                str(tmp_path),
                "--execute",
                "--authorized-minutes",
                "5",
            ]
        )


def test_main_execute_requires_window_level_total_budget(monkeypatch, tmp_path: Path) -> None:
    real_datetime = driver.datetime

    class _DateTime:
        @staticmethod
        def now():  # noqa: ANN202
            return real_datetime(2026, 10, 30)

    monkeypatch.setattr(driver, "datetime", _DateTime)
    monkeypatch.setattr(driver, "load_calendar", lambda root: ["20261030"])
    with pytest.raises(ValueError, match="authorized-total-minutes"):
        driver.main(
            [
                "--mode",
                "restart",
                "--trade-date",
                "20261030",
                "--run-id",
                "t5_test",
                "--evidence-dir",
                str(tmp_path),
                "--execute",
                "--authorized-minutes",
                "30",
            ]
        )
