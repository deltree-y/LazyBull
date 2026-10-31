# -*- coding: utf-8 -*-
"""T5 修复项连贯验证：D4→SLA、台账→冻结、迁移→首日/次日、containment 与预算。

全部使用临时根与合成数据；不读真实生产纸面根、不写数据根、不启动真实日跑。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_closeout as closeout  # noqa: E402
from scripts.v2_p4 import t5_daily_driver as driver  # noqa: E402
from scripts.v2_p4 import t5_daily_evidence as evidence_io  # noqa: E402
from scripts.v2_p4 import t5_gate1_identity as gate1  # noqa: E402
from scripts.v2_p4 import t5_three_arm as arms  # noqa: E402
from scripts.v2_p4.t5_gate7_rehearsal import (  # noqa: E402
    _DAYS,
    _make_data,
    _make_host,
    _make_legacy_source,
)
from scripts.v2_p4.t5_three_arm import ARM_V2  # noqa: E402
from src.lazybull.paper import PaperStorage  # noqa: E402
from src.lazybull.paper.models import AccountState  # noqa: E402
from src.lazybull.v2.hosts.paper.config_adapter import load_paper_config  # noqa: E402

CALENDAR = [
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
        "account_last_update": day,
        "t0_status": "success",
        "ensure_degraded": False,
        "actual_calendar_day": True,
        "corrected_date": day,
        "drill": False,
    }
    payload.update(overrides)
    return payload


def _write_artifacts(data_root: Path, run_id: str, day: str, *, fail_raw: bool = False) -> None:
    base = evidence_io.coordinator_evidence_dir(data_root, f"{run_id}_daily_{day}")
    base.mkdir(parents=True, exist_ok=True)
    (base / "raw_after.json").write_text(
        json.dumps(
            {
                "exit_codes": [0, 1] if fail_raw else [0, 0, 0],
                "unexpected_gaps": {"daily": ["20261030"]} if fail_raw else {},
                "source_side_gaps": {"moneyflow_hsgt": ["20261019"]},
            }
        ),
        encoding="utf-8",
    )
    (base / "clean_after.json").write_text(
        json.dumps({"datasets": {"daily": {"present": 1, "absent": [], "duplicate_key_days": []}}}),
        encoding="utf-8",
    )
    (base / "cs_infer_after.json").write_text(
        json.dumps({"built": 1, "skipped": 0, "failures": []}), encoding="utf-8"
    )
    (base / "panel_after.json").write_text(
        json.dumps({"exit_code": 0, "readiness_not_ready": []}), encoding="utf-8"
    )
    (base / "verification_summary.json").write_text(
        json.dumps({"verdict": "PASS", "readiness_exit_code": 0}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# R2：D4 真实产物 → SLA 步骤
# ---------------------------------------------------------------------------
def test_run_d4_maps_real_artifacts_into_sla_steps(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_artifacts(data_root, "t5run", "20261030")
    monkeypatch.setattr(driver, "_run_command", lambda command, log: None)
    steps = driver.run_d4(
        day="20261030",
        run_id="t5run",
        authorized_minutes=10,
        log_path=tmp_path / "d4.log",
        data_root=data_root,
    )
    record = evidence_io.build_day_record(
        day="20261030",
        started_at="2026-10-30T16:00:00",
        completed_at="2026-10-30T16:20:00",
        d4_stages=steps,
        arms_step={"elapsed_seconds": 1.0, "retries": 0},
    )
    assert set(evidence_io.SLA_REQUIRED_STEPS) <= set(record["steps"])
    assert record["steps"]["ensure_raw"]["origin_stage"] == "raw"
    assert record["steps"]["panel_label_refresh"]["origin_stage"] == "panel"
    assert record["steps"][evidence_io.PREREQUISITE_STEP_KEY]["origin_stage"] == "snapshot"
    assert record["missing_steps"] == []
    assert record["ensure_degraded"] is False
    # 源侧登记缺口不当作降级，但保留在原因里可追溯
    assert any("源侧缺口" in reason for reason in record["steps"]["ensure_raw"]["reasons"])


def test_run_d4_failure_is_recorded_before_raising(monkeypatch, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_artifacts(data_root, "t5run", "20261030")
    calls = {"n": 0}

    def fake_run(command, log):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("clean 子进程 exit=1")

    monkeypatch.setattr(driver, "_run_command", fake_run)
    steps: dict = {}
    with pytest.raises(RuntimeError, match="clean 子进程"):
        driver.run_d4(
            day="20261030",
            run_id="t5run",
            authorized_minutes=10,
            log_path=tmp_path / "d4.log",
            data_root=data_root,
            steps=steps,
        )
    assert steps["clean"]["failed"] is True
    assert driver._failed_step(steps) == "ensure_clean"


def test_degraded_artifact_marks_ensure_degraded(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_artifacts(data_root, "t5run", "20261030", fail_raw=True)
    steps = {}
    for stage in ("snapshot", "raw", "clean", "cs_infer", "panel", "verify"):
        steps[stage] = {
            "elapsed_seconds": 1.0,
            **evidence_io.extract_stage_metrics(data_root, "t5run_daily_20261030", stage),
        }
    record = evidence_io.build_day_record(
        day="20261030",
        started_at="2026-10-30T16:00:00",
        completed_at="2026-10-30T16:20:00",
        d4_stages=steps,
        arms_step={"elapsed_seconds": 1.0, "retries": 0},
    )
    assert record["ensure_degraded"] is True
    assert "20261030" in record["gaps"]


def test_missing_coordinator_artifact_is_fail_closed(tmp_path: Path) -> None:
    metrics = evidence_io.extract_stage_metrics(tmp_path / "data", "empty_run", "raw")
    assert metrics["observed"] is False
    assert metrics["degraded"] is True
    assert any("缺协调者产物" in reason for reason in metrics["reasons"])


# ---------------------------------------------------------------------------
# R1：台账 → 门 1 冻结
# ---------------------------------------------------------------------------
def test_nine_days_blocked_then_ten_days_frozen(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    for day in CALENDAR[:9]:
        evidence_io.write_day_record(evidence / "sla" / "restart" / f"{day}.json", _record(day))
    nine = gate1.validate_restart_records(
        evidence_io.load_window_records(evidence / "sla" / "restart"),
        CALENDAR,
        account_last_update=CALENDAR[8],
    )
    assert nine["passed"] is False
    assert nine["observed_days"] == 9

    evidence_io.write_day_record(
        evidence / "sla" / "restart" / f"{CALENDAR[9]}.json", _record(CALENDAR[9])
    )
    ten = gate1.validate_restart_records(
        evidence_io.load_window_records(evidence / "sla" / "restart"),
        CALENDAR,
        account_last_update=CALENDAR[9],
    )
    assert ten["passed"] is True
    assert ten["observed_days"] == 10
    assert ten["account_aligned"] is True


def test_restart_ledger_skips_failed_day_and_keeps_attempt(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    good = _record("20261001")
    good["counts_as_recovery_day"] = True
    bad = _record("20261002", run_completed=False, system_reason="host_process_crash")
    bad["counts_as_recovery_day"] = False
    driver._persist_restart(evidence, good)
    driver._persist_restart(evidence, bad)
    ledger = json.loads((evidence / "q11a_restart_ledger.json").read_text(encoding="utf-8"))
    attempts = json.loads((evidence / "q11a_restart_attempts.json").read_text(encoding="utf-8"))
    assert [item["trade_date"] for item in ledger] == ["20261001"]
    assert [item["trade_date"] for item in attempts] == ["20261001", "20261002"]
    assert (evidence / "sla" / "restart" / "20261002.json").is_file()


def test_restart_day_failure_persists_evidence_and_reraises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    paper = tmp_path / "data" / "paper" / "state"
    paper.mkdir(parents=True)
    (paper / "account.json").write_text(
        json.dumps({"cash": 1000.0, "last_update": "20261001", "positions": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        driver,
        "run_d4",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("D4 raw 失败")),
    )
    real_datetime = driver.datetime

    class _DateTime:
        @staticmethod
        def now():  # noqa: ANN202
            return real_datetime(2026, 10, 2)

    monkeypatch.setattr(driver, "datetime", _DateTime)
    evidence = tmp_path / "evidence"
    with pytest.raises(RuntimeError, match="D4 raw 失败"):
        driver.run_restart_day(
            day="20261002",
            run_id="t5run",
            evidence=evidence,
            authorized_minutes=10,
            total_minutes=600,
            data_root=tmp_path / "data",
        )
    record = json.loads(
        (evidence / "sla" / "restart" / "20261002.json").read_text(encoding="utf-8")
    )
    assert record["counts_as_recovery_day"] is False
    assert record["run_completed"] is False
    assert record["error"]
    assert record["failure_context"]["failed_step"]
    assert not (evidence / "q11a_restart_ledger.json").is_file()


def test_advance_without_complete_steps_is_not_counted(monkeypatch, tmp_path: Path) -> None:
    """账户已推进但当日步骤不全（run 未完成）⇒ 记录推进事实但不计入恢复日。"""
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    paper = tmp_path / "data" / "paper"
    (paper / "state").mkdir(parents=True)
    (paper / "runs").mkdir(parents=True)
    (paper / "state" / "account.json").write_text(
        json.dumps({"cash": 1000.0, "last_update": "20261002", "positions": {}}), encoding="utf-8"
    )
    (paper / "runs" / "t1_20261002.json").write_text(
        json.dumps({"t0_status": "success"}), encoding="utf-8"
    )
    monkeypatch.setattr(driver, "run_d4", lambda **kwargs: {})
    monkeypatch.setattr(
        driver, "_run_command", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("run 崩溃"))
    )
    real_datetime = driver.datetime

    class _DateTime:
        @staticmethod
        def now():  # noqa: ANN202
            return real_datetime(2026, 10, 2)

    monkeypatch.setattr(driver, "datetime", _DateTime)
    evidence = tmp_path / "evidence"
    with pytest.raises(RuntimeError, match="run 崩溃"):
        driver.run_restart_day(
            day="20261002",
            run_id="t5run",
            evidence=evidence,
            authorized_minutes=10,
            total_minutes=600,
            data_root=tmp_path / "data",
        )
    record = json.loads(
        (evidence / "sla" / "restart" / "20261002.json").read_text(encoding="utf-8")
    )
    assert record["account_advanced"] is True
    assert record["counts_as_recovery_day"] is False
    assert not (evidence / "q11a_restart_ledger.json").is_file()


def test_same_day_repair_keeps_one_ledger_entry(tmp_path: Path) -> None:
    """同日补救只补齐证据、不增加有效日数（累计台账同日条目保持 1 条）。"""
    evidence = tmp_path / "evidence"
    record = _record("20261002")
    record["counts_as_recovery_day"] = True
    driver._persist_restart(evidence, record)
    repaired = {**record, "evidence_warnings": ["日志写入失败已补记"]}
    driver._persist_restart(evidence, repaired)
    ledger = json.loads((evidence / "q11a_restart_ledger.json").read_text(encoding="utf-8"))
    attempts = json.loads((evidence / "q11a_restart_attempts.json").read_text(encoding="utf-8"))
    assert [item["trade_date"] for item in ledger] == ["20261002"]
    assert ledger[0]["evidence_warnings"] == ["日志写入失败已补记"]
    assert len(attempts) == 2


# ---------------------------------------------------------------------------
# R3/R4：迁移 → 首日/次日恢复；containment；预算
# ---------------------------------------------------------------------------
@pytest.fixture()
def arm_env(tmp_path: Path) -> dict:
    _make_data(tmp_path / "data")
    frozen = tmp_path / "frozen"
    _make_legacy_source(frozen)
    storage = PaperStorage(root_path=str(frozen))
    storage.save_account_state(AccountState(cash=100000.0, positions={}, last_update=_DAYS[0]))
    account_bytes = (frozen / "state" / "account.json").read_bytes()
    import hashlib

    gate1_payload = {
        "gate1_frozen": True,
        "verdict": "PASS",
        "account": {"sha256": hashlib.sha256(account_bytes).hexdigest(), "last_update": _DAYS[0]},
        "data_state": {"data_state_id": "state-1"},
        "model": {"content_fingerprint": "model-1"},
        "execution": {"fingerprint": "exec-1"},
        "q11a_restart": {"selected_days": [_DAYS[0]]},
    }
    return {
        "frozen": frozen,
        "data_root": tmp_path / "data",
        "gate1": gate1_payload,
        "work_root": tmp_path / "work",
        "evidence": tmp_path / "evidence",
    }


def test_migrated_v2_arm_recovers_and_runs_next_day(arm_env) -> None:
    specs = arms.prepare_three_arms(
        frozen_paper_root=arm_env["frozen"],
        work_root=arm_env["work_root"],
        gate1=arm_env["gate1"],
        data_root=arm_env["data_root"],
        evidence=arm_env["evidence"] / "initial_migration",
        host_builder=_make_host,
    )
    flat = load_paper_config(specs[ARM_V2].root)
    host = _make_host(specs[ARM_V2].root, flat, arm_env["data_root"])
    before = host.store.load_engine_state_blob()
    assert before is not None
    host.run_t1(_DAYS[1])
    result = host.last_day_result
    assert result is not None and result.driven is True
    after = host.store.load_engine_state_blob()
    assert after is not None and after != before
    # 次日不再重新迁移/重锚：隔离根仍只有一份初始化记录，且账户已推进
    marker = arms.load_init_marker(arm_env["work_root"])
    assert marker is not None
    assert (
        json.loads((specs[ARM_V2].root / "state" / "account.json").read_text(encoding="utf-8"))[
            "last_update"
        ]
        == result.corrected_date
    )


def test_shadow_containment_flags_production_write() -> None:
    before = {"state/account.json": "a"}
    after_all = {"state/account.json": "b"}
    containment = driver._shadow_containment(
        before_prod=before,
        after_d4=before,
        after_active=before,
        after_neutral=before,
        after_all=after_all,
        data_before={"data_state_id": "x"},
        data_after={"data_state_id": "x"},
        active_production=True,
        drill=False,
    )
    assert containment["v2_arm_wrote_production"] is True
    assert containment["shadow_arms_wrote_production"] is True
    assert containment["production_unchanged_outside_active_arm"] is False


def test_shadow_containment_allows_active_only_writes() -> None:
    before = {"state/account.json": "a"}
    after_active = {"state/account.json": "b"}
    containment = driver._shadow_containment(
        before_prod=before,
        after_d4=before,
        after_active=after_active,
        after_neutral=after_active,
        after_all=after_active,
        data_before={"data_state_id": "x"},
        data_after={"data_state_id": "x"},
        active_production=True,
        drill=False,
    )
    assert containment["active_arm_wrote_production"] is True
    assert containment["shadow_arms_wrote_production"] is False
    assert containment["production_unchanged_outside_active_arm"] is True
    assert containment["production_writes_attributed_to_active_only"] is True


def test_shadow_containment_rejects_d4_write_to_production_paper() -> None:
    before = {"state/account.json": "a"}
    after_d4 = {"state/account.json": "b"}
    containment = driver._shadow_containment(
        before_prod=before,
        after_d4=after_d4,
        after_active=after_d4,
        after_neutral=after_d4,
        after_all=after_d4,
        data_before={"data_state_id": "x"},
        data_after={"data_state_id": "x"},
        active_production=True,
        drill=False,
    )
    assert containment["d4_wrote_production"] is True
    assert containment["production_unchanged_outside_active_arm"] is False


def test_machine_budget_registers_totals_and_blocks_reentry(tmp_path: Path) -> None:
    path = tmp_path / "machine_budget.json"
    budget = evidence_io.MachineBudget(path, total_minutes=100, daily_cap_minutes=30)
    budget.check("20261030")
    budget.finish("20261030", 600.0)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["authorized_minutes_total"] == 100
    assert saved["consumed_minutes"] == pytest.approx(10.0)
    assert saved["remaining_minutes"] == pytest.approx(90.0)
    with pytest.raises(RuntimeError, match="不得重复驱动"):
        budget.check("20261030")
    with pytest.raises(RuntimeError, match="授权总额与已登记台账不一致"):
        evidence_io.MachineBudget(path, total_minutes=200, daily_cap_minutes=30)


def test_scan_data_gaps_reports_first_missing_source(tmp_path: Path) -> None:
    day_dir = tmp_path / "clean" / "daily"
    day_dir.mkdir(parents=True)
    pd.DataFrame({"ts_code": ["A"], "close": [1.0]}).to_parquet(day_dir / "2026-10-30.parquet")
    result = evidence_io.scan_data_gaps(tmp_path, "20261029", "20261030", ["20261029", "20261030"])
    assert result["target_in_calendar"] is True
    assert result["first_missing_day"] == "20261029"
    assert result["sources"]["clean/daily"]["missing_count"] == 1
    assert result["all_clear"] is False


# ---------------------------------------------------------------------------
# 跨模块连贯：十日台账 → 门 1 冻结 → 门 5 比较 → 收口汇总
# ---------------------------------------------------------------------------
def _gate1_fixture(tmp_path: Path) -> dict:
    data_root = tmp_path / "data"
    (data_root / "clean" / "daily").mkdir(parents=True)
    pd.DataFrame(
        {
            "cal_date": [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in CALENDAR],
            "is_open": [1] * len(CALENDAR),
        }
    ).to_parquet(data_root / "clean" / "trade_cal.parquet", index=False)
    pd.DataFrame(
        {
            "ts_code": ["A"],
            "close": [10.0],
            "pre_close": [10.0],
            "tradable": [1],
            "is_st": [0],
            "is_suspended": [0],
        }
    ).to_parquet(data_root / "clean" / "daily" / "2026-10-14.parquet", index=False)
    models = data_root / "models" / "stock_selection"
    models.mkdir(parents=True)
    (models / "v7_model.joblib").write_bytes(b"model")
    (models / "v7_metadata.json").write_text(
        json.dumps({"model_type": "xgboost", "train_start_date": "20200101"}), encoding="utf-8"
    )
    (models / "v7_features.json").write_text(json.dumps(["ret_1"]), encoding="utf-8")
    paper_root = tmp_path / "paper"
    storage = PaperStorage(root_path=str(paper_root))
    storage.save_config({"model_version": 7, "initial_capital": 100000.0, "rebalance_freq": 20})
    (paper_root / "state" / "account.json").write_text(
        json.dumps({"cash": 100000.0, "last_update": CALENDAR[-1], "positions": {}}),
        encoding="utf-8",
    )
    return {"data_root": data_root, "paper_root": paper_root}


def test_ten_day_ledger_freezes_gate1_then_gate5_and_closeout(monkeypatch, tmp_path: Path) -> None:
    env = _gate1_fixture(tmp_path)
    evidence = tmp_path / "evidence"
    for day in CALENDAR:
        evidence_io.write_day_record(evidence / "sla" / "restart" / f"{day}.json", _record(day))
    ledger = [
        json.loads((evidence / "sla" / "restart" / f"{d}.json").read_text(encoding="utf-8"))
        for d in CALENDAR
    ]
    (evidence / "q11a_restart_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(
        gate1,
        "capture_data_state",
        lambda root: {
            "data_state_id": "state-1",
            "sources": {
                k: {"max": "20261014"}
                for k in ("raw_daily", "clean_daily", "cs_train", "cs_infer", "panel")
            },
            "labels": {"y_ret_5": {"sealed_through": "20260917"}},
        },
    )
    monkeypatch.setattr(gate1, "compute_worktree_digest", lambda root: {"sha256": "worktree"})
    monkeypatch.setattr(gate1, "_code_state", lambda: {"commit": "abc"})
    result = gate1.collect_gate1(
        repo_root=tmp_path,
        data_root=env["data_root"],
        paper_root=env["paper_root"],
        model_version=7,
        restart_records=gate1.load_cumulative_ledger(evidence / "q11a_restart_ledger.json"),
        require_restart=True,
    )
    assert result["gate1_frozen"] is True
    assert result["q11a_restart"]["account_aligned"] is True
    assert result["q11a_restart"]["observed_days"] == 10
    (evidence / "verification_summary.json").write_text(json.dumps(result), encoding="utf-8")

    from scripts.v2_p4.t5_shadow_compare import compare_shadow_day

    identical = {
        "date_phase": {"trade_date": CALENDAR[-1], "phase": "T0_T1"},
        "merged_target": [],
        "account": {"cash": 100000.0, "positions": {}},
        "pending": {"pending_buys": {}, "pending_sells": {}, "instructions": {}},
        "schedule_anchor": {},
        "ranked_candidates": [],
        "stop_loss_state": {},
        "market_data": {"source": "cs_infer+clean.daily", "features": [], "prices": []},
        "effective_identity": {},
    }
    gate5 = compare_shadow_day(
        left_input=identical,
        right_input=dict(identical),
        left_orders=[],
        right_orders=[],
        evidence=None,
        acceptance_edge=True,
    )
    (evidence / "gate5").mkdir(parents=True, exist_ok=True)
    for day in CALENDAR:
        (evidence / "gate5" / f"daily_{day}.json").write_text(
            json.dumps({**gate5, "date": day}), encoding="utf-8"
        )
    (evidence / "sla").mkdir(exist_ok=True)
    (evidence / "sla" / "window.json").write_text(
        json.dumps(
            {"verdict": "PASS", "current_streak_dates": CALENDAR, "current_streak_days": 10}
        ),
        encoding="utf-8",
    )
    (evidence / "gate7").mkdir(parents=True, exist_ok=True)
    (evidence / "gate7" / "real_rehearsal_result.json").write_text(
        json.dumps({"verdict": "PASS", "synthetic_only": False, "items_incomplete": []}),
        encoding="utf-8",
    )
    batch = tmp_path / "runs" / "p4_t5_x_paper_shadow"
    batch.mkdir(parents=True)
    (batch / "batch_meta.json").write_text(
        json.dumps({"batch_id": "p4_t5_x_paper_shadow"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        closeout,
        "capture_data_state",
        lambda root: {
            "data_state_id": "state-1",
            "sources": {
                k: {"max": "20261014"}
                for k in ("raw_daily", "clean_daily", "cs_train", "cs_infer", "panel")
            },
            "labels": {"y_ret_5": {"sealed_through": "20260917"}},
        },
    )
    monkeypatch.setattr(closeout, "compute_worktree_digest", lambda root: {"sha256": "worktree"})
    monkeypatch.setattr(closeout, "build_file_inventory", lambda root: [])
    monkeypatch.setattr(
        closeout,
        "PaperStorage",
        lambda: type("Storage", (), {"load_config": lambda self: {"engine": "legacy"}})(),
    )
    monkeypatch.setattr(closeout, "ROOT", tmp_path)
    monkeypatch.setattr(closeout, "load_runs_batch", lambda path: object())
    status = closeout.closeout(evidence, runs_batches=[batch])
    assert status["gates"]["gate1"] == "PASS"
    assert status["gates"]["gate5"] == "PASS"
    assert status["gates"]["gate6"] == "PASS"
    assert status["gates"]["gate7"] == "PASS"
    # R2-04：缺全量/复杂度/containment 时门 8 必须否决，不能 READY。
    assert status["gates"]["gate8"] == "BLOCKED"
    assert status["verdict"] == "BLOCKED"
    assert any("门 8" in reason for reason in status["blocked_reasons"])
    assert status["statuses"]["user_switch_authorized"] is False


# ---------------------------------------------------------------------------
# R4/R5：命令串真的能被参数解析器吃掉
# ---------------------------------------------------------------------------
def _concretize(command: str) -> str:
    command = command.replace("<真实交易日>", "20261030")
    command = command.replace("<本次授权>", "45")
    command = command.replace("<窗口总授权>", "900")
    command = command.replace("<扫描出的首缺日>", "20261001")
    command = command.replace("<首跑日>", "20261030")
    command = command.replace("<现役模型>", "24022")
    return command


def test_recovery_commands_parse_with_real_argparse(tmp_path: Path) -> None:
    commands = closeout._recovery_commands(tmp_path / "evidence", {"model": {"version": 24022}})
    assert len(commands) == 4
    for command in commands:
        argv = re.split(r"\s+", _concretize(command))[2:]  # 去掉 python scripts/v2_p4/xxx.py
        if "t5_daily_driver.py" in command:
            args = driver._parse_args(argv)
            assert args.mode in {"preflight", "restart", "shadow"}
            if args.mode != "preflight":
                assert args.authorized_minutes == 45
                assert args.authorized_total_minutes == 900
            if args.mode == "shadow":
                assert args.active_production is True
        else:
            args = gate1._parse_args(argv)
            assert args.freeze is True
            assert args.restart_ledger.endswith("q11a_restart_ledger.json")


def test_shadow_plan_and_gate1_freeze_command_carry_inputs(tmp_path: Path) -> None:
    commands = closeout._recovery_commands(tmp_path / "evidence", {"model": {"version": 24022}})
    assert any("--restart-ledger" in item for item in commands)
    assert any("--active-production" in item for item in commands)
    assert any("paper_shadow" in item and "arms" in item for item in commands)


def test_preflight_mode_does_not_call_d4(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(driver, "ROOT", tmp_path)
    (tmp_path / "data" / "clean").mkdir(parents=True)
    pd.DataFrame({"cal_date": ["2026-10-30"], "is_open": [1]}).to_parquet(
        tmp_path / "data" / "clean" / "trade_cal.parquet"
    )
    monkeypatch.setattr(driver, "load_calendar", lambda root: ["20261030"])
    captured = {}

    def fake_run_d4(**kwargs):  # noqa: ANN003, ANN202
        captured["called"] = True
        raise AssertionError("preflight 不得驱动 D4")

    monkeypatch.setattr(driver, "run_d4", fake_run_d4)
    code = driver.main(
        [
            "--mode",
            "preflight",
            "--trade-date",
            "20261030",
            "--start-date",
            "20261030",
            "--target-date",
            "20261030",
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--execute",
        ]
    )
    assert code == 0
    assert "called" not in captured
    assert (tmp_path / "evidence" / "preflight" / "data_gaps.json").is_file()
