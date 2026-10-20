# -*- coding: utf-8 -*-
"""P3 pre T3 日更 CLI 计划、授权、备份与 refresh-only 编排测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p3_pre import update_daily_store as daily_update
from src.lazybull.data.storage import Storage
from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.store.column_groups import LABEL_TABLES, PANEL_GROUPS
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.panel_builder import bootstrap_manifest
from src.lazybull.v2.store.store_identity import collect_file_identities

_DAY = "20260703"
_CALENDAR = pd.bdate_range("2026-06-01", "2026-08-31").strftime("%Y%m%d").tolist()
_CODES = [f"00000{i}.SZ" for i in range(6)]
_PRODUCER = {"producer_digest": "producer-test"}
_WORKSPACE = {"source_digest": "workspace-test"}


class _Builder:
    def __init__(self, loader):
        self._loader = loader
        self._builder = SimpleNamespace(shenwan_level="l2")
        self.build_calls = []

    def _full_calendar(self):
        return list(_CALENDAR)

    def build_daily(self, date, groups, store):
        self.build_calls.append((str(date), tuple(groups)))
        panel = _panel_frame(str(date))
        _write_complete_panel(store, panel)
        _write_forming(store, panel)


def _input_loader(root: Path, first: str, last: str, *, empty_tail: bool = False):
    loader = SimpleNamespace(storage=Storage(root_path=str(root)))
    (loader.storage.clean_path / "daily").mkdir(parents=True, exist_ok=True)
    dates = [date for date in _CALENDAR if first <= date <= last]
    for date in dates:
        if date == last and empty_tail:
            frame = pd.DataFrame(columns=["trade_date", "ts_code", "open_adj", "close_adj"])
        else:
            frame = pd.DataFrame(
                {
                    "trade_date": [date] * len(_CODES),
                    "ts_code": _CODES,
                    "open_adj": np.arange(len(_CODES)) + 11.0,
                    "close_adj": np.arange(len(_CODES)) + 10.0,
                }
            )
        filename = f"{date[:4]}-{date[4:6]}-{date[6:]}.parquet"
        frame.to_parquet(loader.storage.clean_path / "daily" / filename, index=False)
    pd.DataFrame({"cal_date": _CALENDAR, "is_open": [1] * len(_CALENDAR)}).to_parquet(
        loader.storage.clean_path / "trade_cal.parquet", index=False
    )
    pd.DataFrame({"ts_code": _CODES}).to_parquet(
        loader.storage.raw_path / "stock_basic.parquet", index=False
    )
    return loader


def _panel_frame(date: str = _DAY) -> pd.DataFrame:
    n = len(_CODES)
    return pd.DataFrame(
        {
            "trade_date": [date] * n,
            "ts_code": _CODES,
            "sw_industry": ["行业A"] * 3 + ["行业B"] * 3,
            "sw_industry_code": ["A"] * 3 + ["B"] * 3,
            "sw_l1_code": ["L1"] * n,
            "sw_l2_code": ["A"] * 3 + ["B"] * 3,
            "tradable": [1] * n,
        }
    )


def _write_complete_panel(store: PanelDataStore, panel: pd.DataFrame) -> None:
    for group in PANEL_GROUPS:
        if group == "core":
            frame = panel.copy()
        else:
            column = next(c for c in PANEL_GROUPS[group] if c not in {"trade_date", "ts_code"})
            frame = panel[["trade_date", "ts_code"]].assign(**{column: 1.0})
        store.append_features(TradeDate.from_str(panel["trade_date"].iloc[0]), group, frame)


def _write_forming(store: PanelDataStore, panel: pd.DataFrame) -> None:
    for name in LABEL_TABLES:
        table = panel[["ts_code", "trade_date"]].assign(
            label_value=np.nan, neu_label_value=np.nan, maturity_status="forming"
        )
        store.append_labels(name, table)


def _complete_store(data_root: Path, end_date: str) -> PanelDataStore:
    store = PanelDataStore(data_root)
    bootstrap_manifest(store)
    for date in (d for d in _CALENDAR if _DAY <= d <= end_date):
        panel = _panel_frame(date)
        _write_complete_panel(store, panel)
        _write_forming(store, panel)
    return store


def _run(data_root, backup_root, loader, builder, **kwargs):
    options = {
        "store": PanelDataStore(data_root),
        "loader": loader,
        "builder": builder,
        "producer": _PRODUCER,
        "workspace": _WORKSPACE,
    }
    options.update(kwargs)
    return daily_update.update_daily_store(data_root, backup_root, **options)


def test_realistic_forming_dry_run_and_time_budget_reject(tmp_path):
    data_root = tmp_path / "store"
    loader = _input_loader(tmp_path / "input", "20260706", "20260731")
    builder = _Builder(loader)
    backup = tmp_path / "backup" / "run-forming"
    report = _run(
        data_root,
        backup,
        loader,
        builder,
        start_date=_DAY,
        authorized_minutes=200,
    )
    assert report["status"] == "planned" and report["mode"] == "dry_run"
    assert report["end_date"] == "20260731"
    assert len(report["build_plan"]) == 21
    assert report["refresh_plans"] == []
    assert report["estimated_seconds"] == 21 * daily_update._BUILD_SECONDS_PER_DAY
    assert report["build_daily_calls"] == 21 and report["refresh_calls"] == 0
    assert report["estimated_build_seconds"] == 21 * daily_update._BUILD_SECONDS_PER_DAY
    assert report["estimated_refresh_seconds"] == 0
    assert len(report["daily_plan"]) == 21
    assert report["daily_plan"][0]["buy_date"] == "20260706"
    assert report["daily_plan"][0]["endpoint_date"] == "20260803"
    assert report["daily_plan"][0]["expected_status"] == "forming"
    assert (
        report["daily_plan"][-1]["cumulative_estimated_seconds"]
        == 21 * daily_update._BUILD_SECONDS_PER_DAY
    )
    assert not backup.exists() and not Path(report["report_path"]).exists()
    assert builder.build_calls == []

    with pytest.raises(RuntimeError, match="超过本次授权"):
        _run(
            data_root,
            backup,
            loader,
            builder,
            start_date=_DAY,
            authorized_minutes=20,
            execute=True,
        )
    assert not backup.exists() and builder.build_calls == []


def test_default_dependencies_use_explicit_data_root(tmp_path, monkeypatch):
    data_root = tmp_path / "explicit-store-root"
    _input_loader(data_root, "20260706", "20260731")
    seen = []

    def fake_builder(loader):
        seen.append(loader)
        return _Builder(loader)

    monkeypatch.setattr(daily_update, "V2PanelBuilder", fake_builder)
    report = daily_update.update_daily_store(
        data_root,
        tmp_path / "backup" / "run-default-root",
        start_date=_DAY,
        authorized_minutes=200,
        producer=_PRODUCER,
        workspace=_WORKSPACE,
    )
    assert report["status"] == "planned"
    assert len(report["build_plan"]) == 21
    assert len(seen) == 1
    assert Path(seen[0].storage.root_path).resolve() == data_root.resolve()


def test_execute_calls_build_daily_per_day_after_snapshot(tmp_path, monkeypatch):
    import json

    data_root = tmp_path / "store"
    loader = _input_loader(tmp_path / "input", "20260706", "20260731")
    builder = _Builder(loader)
    backup = tmp_path / "backup" / "run-build"
    real_snapshot = daily_update._save_run_snapshot

    def assert_locked_snapshot(store, *args, **kwargs):
        assert store._lock._depth > 0
        return real_snapshot(store, *args, **kwargs)

    monkeypatch.setattr(daily_update, "_save_run_snapshot", assert_locked_snapshot)
    report = _run(
        data_root,
        backup,
        loader,
        builder,
        start_date=_DAY,
        execute=True,
        authorized_minutes=200,
    )
    assert report["status"] == "executed"
    assert len(builder.build_calls) == 21
    assert all(groups == tuple(PANEL_GROUPS) for _, groups in builder.build_calls)
    assert Path(report["report_path"]).exists()
    assert len(report["results"]) == 21  # reviewer1 S1 处置：results 含逐日 build_daily 结果
    assert {item["action"] for item in report["results"]} == {"build_daily"}
    assert {item["status"] for item in report["results"]} == {"built"}
    assert all(item["elapsed_seconds"] >= 0 for item in report["results"])
    assert len(report["readiness"]) == 21
    assert {item["status"] for item in report["readiness"]} == {"forming"}
    checkpoint = json.loads((backup / "checkpoint.json").read_text(encoding="utf-8"))
    planned = checkpoint["planned_new_files"]
    assert sum(path.startswith("features/panel/") for path in planned) == 21 * 8
    assert sum(path.startswith("labels/") for path in planned) == 21 * 3


def test_empty_build_cannot_report_executed(tmp_path):
    import json

    data_root = tmp_path / "store"
    loader = _input_loader(tmp_path / "input", "20260706", "20260731")
    builder = _Builder(loader)
    builder.build_daily = lambda *args: None
    backup = tmp_path / "backup" / "run-empty-build"
    with pytest.raises(RuntimeError, match="八族"):
        _run(
            data_root,
            backup,
            loader,
            builder,
            start_date=_DAY,
            execute=True,
            authorized_minutes=200,
        )
    report = json.loads(
        (data_root / "reports" / "v2_p3_pre_label_refresh_run-empty-build.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["input_identity_unchanged"] is None


def test_execute_failure_writes_reason_and_preserves_checkpoint(tmp_path):
    import json

    data_root = tmp_path / "store"
    loader = _input_loader(tmp_path / "input", "20260706", "20260731")
    builder = _Builder(loader)
    backup = tmp_path / "backup" / "run-fail"

    def fail_build(*args):
        raise RuntimeError("模拟逐日日更失败")

    builder.build_daily = fail_build
    with pytest.raises(RuntimeError, match="模拟逐日日更失败"):
        _run(
            data_root,
            backup,
            loader,
            builder,
            start_date=_DAY,
            execute=True,
            authorized_minutes=200,
        )
    report_path = data_root / "reports" / "v2_p3_pre_label_refresh_run-fail.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["failure_reason"] == "模拟逐日日更失败"
    assert report["results"] == []
    assert (backup / "checkpoint.json").exists()


def test_refresh_only_execute_then_noop_preserves_report_and_audits(tmp_path):
    data_root = tmp_path / "store"
    store = PanelDataStore(data_root)
    bootstrap_manifest(store)
    for date in (d for d in _CALENDAR if _DAY <= d <= "20260803"):
        panel = _panel_frame(date)
        _write_complete_panel(store, panel)
        _write_forming(store, panel)
    loader = _input_loader(tmp_path / "input", "20260706", "20260803")
    builder = _Builder(loader)
    backup = tmp_path / "backup" / "run-refresh"
    report = daily_update.update_daily_store(
        data_root,
        backup,
        start_date=_DAY,
        execute=True,
        refresh_only=True,
        store=store,
        loader=loader,
        builder=builder,
        producer=_PRODUCER,
        workspace=_WORKSPACE,
    )
    assert report["status"] == "executed"
    assert report["estimated_seconds"] == 6
    assert report["estimated_preflight_seconds"] == 3
    assert report["daily_plan"][0]["estimated_preflight_seconds"] == 3
    assert [item["status"] for item in report["results"]] == ["refreshed"]
    assert builder.build_calls == []
    report_bytes = Path(report["report_path"]).read_bytes()
    assert len(store.manifest.label_refreshes()) == 1

    second = daily_update.update_daily_store(
        data_root,
        backup,
        start_date=_DAY,
        execute=True,
        refresh_only=True,
        store=store,
        loader=loader,
        builder=builder,
        producer=_PRODUCER,
        workspace=_WORKSPACE,
    )
    assert second["status"] == "no_op"
    assert [item["status"] for item in second["results"]] == ["no_op"]
    assert Path(report["report_path"]).read_bytes() == report_bytes
    assert len(store.manifest.label_refresh_intents()) == 1
    assert len(store.manifest.label_refreshes()) == 1


def test_recover_event_cli_dry_run_then_roll_forward(tmp_path, monkeypatch):
    import json

    from src.lazybull.v2.store.manifest import Manifest

    data_root = tmp_path / "store"
    store = PanelDataStore(data_root)
    bootstrap_manifest(store)
    panel = _panel_frame()
    _write_complete_panel(store, panel)
    _write_forming(store, panel)
    loader = _input_loader(tmp_path / "input", "20260706", "20260803")
    builder = _Builder(loader)
    frames, watermark, identities = daily_update._load_clean_inputs(loader, _CALENDAR, _DAY)
    kwargs = {
        "trading_dates": _CALENDAR,
        "daily_adj_dict": frames,
        "input_watermark": watermark,
        "shenwan_level": "l2",
        "producer": _PRODUCER,
        "input_identities": identities,
        "input_base_dir": loader.storage.root_path,
        "run_id": "run-recover",
    }

    def crash_partition(*args):
        raise RuntimeError("模拟 replace 后 WAL 前崩溃")

    def crash_save(self):
        raise RuntimeError("模拟 manifest.save 崩溃")

    monkeypatch.setattr(store._wal, "register_label_partition", crash_partition)
    monkeypatch.setattr(Manifest, "save", crash_save)
    with pytest.raises(RuntimeError, match="manifest.save 崩溃"):
        daily_update.refresh_labels_for_day(store, _DAY, **kwargs)
    monkeypatch.undo()
    manifest_path = data_root / "features" / "manifest.json"
    wal_path = data_root / "features" / "manifest_wal.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    wal_bytes = wal_path.read_bytes()
    wal_events = [json.loads(line) for line in wal_bytes.decode("utf-8").splitlines()]
    event_key = next(
        event["record"]["event_key"]
        for event in wal_events
        if event["op"] == "append_label_refresh_intent"
    )
    backup = tmp_path / "backup" / "run-recover"
    reopened = PanelDataStore(data_root)
    original_source_fp = Manifest(manifest_path).content_fingerprint()

    with pytest.raises(RuntimeError, match="超过本次授权"):
        daily_update.update_daily_store(
            data_root,
            backup,
            recover_event=event_key,
            execute=True,
            authorized_minutes=0.001,
            store=reopened,
            loader=loader,
            builder=builder,
            producer=_PRODUCER,
            workspace=_WORKSPACE,
        )
    assert manifest_path.read_bytes() == manifest_bytes
    assert wal_path.read_bytes() == wal_bytes
    assert not backup.exists()

    plan = daily_update.update_daily_store(
        data_root,
        backup,
        recover_event=event_key,
        store=reopened,
        loader=loader,
        builder=builder,
        producer=_PRODUCER,
        workspace=_WORKSPACE,
    )
    assert plan["status"] == "planned"
    assert plan["preflight"][0]["status"] == "recovery_planned"
    assert not backup.exists()
    assert manifest_path.read_bytes() == manifest_bytes
    assert wal_path.read_bytes() == wal_bytes
    report = daily_update.update_daily_store(
        data_root,
        backup,
        recover_event=event_key,
        execute=True,
        store=reopened,
        loader=loader,
        builder=builder,
        producer=_PRODUCER,
        workspace=_WORKSPACE,
    )
    assert report["results"][0]["status"] == "recovered"
    assert len(report["readiness"]) == 1 and report["readiness"][0]["status"] == "sealed"
    assert not wal_path.exists()
    checkpoint = json.loads((backup / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["planned_new_files"] == []
    assert len(checkpoint["affected_existing_files"]) == 3
    assert checkpoint["empty_checkpoint"] is False
    assert (backup / "manifest_wal.jsonl").read_bytes() == wal_bytes
    snapshot = json.loads((backup / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["source_content_fingerprint"] == original_source_fp


def test_late_start_date_cannot_skip_earliest_deferred_day(tmp_path):
    data_root = tmp_path / "store"
    store = PanelDataStore(data_root)
    bootstrap_manifest(store)
    for date in (_DAY, "20260706"):
        panel = _panel_frame(date)
        _write_complete_panel(store, panel)
        _write_forming(store, panel)
    loader = _input_loader(tmp_path / "input", "20260707", "20260804")
    builder = _Builder(loader)
    backup = tmp_path / "backup" / "run-late-start"
    with pytest.raises(RuntimeError, match="缺早日跳晚日"):
        daily_update.update_daily_store(
            data_root,
            backup,
            start_date="20260706",
            store=store,
            loader=loader,
            builder=builder,
            producer=_PRODUCER,
            workspace=_WORKSPACE,
        )
    assert not backup.exists()


def test_refresh_only_missing_panel_and_empty_tail_fail_before_writes(tmp_path):
    data_root = tmp_path / "store"
    loader = _input_loader(tmp_path / "input", "20260706", "20260803")
    builder = _Builder(loader)
    backup = tmp_path / "backup" / "run-missing"
    with pytest.raises(RuntimeError, match="八族"):
        _run(
            data_root,
            backup,
            loader,
            builder,
            start_date=_DAY,
            refresh_only=True,
        )
    assert not backup.exists()

    empty_loader = _input_loader(tmp_path / "empty-input", "20260706", "20260731", empty_tail=True)
    with pytest.raises(RuntimeError, match="尾部候选为空"):
        _run(data_root, backup, empty_loader, _Builder(empty_loader), start_date=_DAY)
    assert not backup.exists()


def test_clean_revision_during_read_is_rejected_before_any_store_write(tmp_path, monkeypatch):
    data_root = tmp_path / "store"
    store = _complete_store(data_root, "20260803")
    loader = _input_loader(tmp_path / "input", "20260706", "20260803")
    builder = _Builder(loader)
    backup = tmp_path / "backup" / "run-read-race"
    real_read = daily_update._read_clean_partition

    def revise_after_read(path, partition_date):
        frame = real_read(path, partition_date)
        if partition_date == "20260706":
            revised = pd.read_parquet(path)
            revised.loc[0, "close_adj"] = 20.0
            revised.to_parquet(path, index=False)
        return frame

    monkeypatch.setattr(daily_update, "_read_clean_partition", revise_after_read)
    with pytest.raises(daily_update.InputIdentityDriftError, match="读取期间"):
        daily_update.update_daily_store(
            data_root,
            backup,
            start_date=_DAY,
            execute=True,
            refresh_only=True,
            store=store,
            loader=loader,
            builder=builder,
            producer=_PRODUCER,
            workspace=_WORKSPACE,
        )
    assert not backup.exists()
    labels = pd.read_parquet(data_root / "labels" / "y_ret_20" / f"{_DAY}.parquet")
    assert labels["maturity_status"].eq("forming").all()


def test_input_drift_during_build_writes_failed_report_with_false_identity(tmp_path):
    import json

    data_root = tmp_path / "store"
    loader = _input_loader(tmp_path / "input", "20260706", "20260731")
    builder = _Builder(loader)
    original_build = builder.build_daily
    revised = False

    def build_then_revise(date, groups, store):
        nonlocal revised
        original_build(date, groups, store)
        if not revised:
            path = loader.storage.clean_path / "daily" / "2026-07-06.parquet"
            frame = pd.read_parquet(path)
            frame.loc[0, "close_adj"] = 20.0
            frame.to_parquet(path, index=False)
            revised = True

    builder.build_daily = build_then_revise
    backup = tmp_path / "backup" / "run-build-drift"
    with pytest.raises(daily_update.InputIdentityDriftError, match="输入身份变化"):
        _run(
            data_root,
            backup,
            loader,
            builder,
            start_date=_DAY,
            execute=True,
            authorized_minutes=200,
        )
    report = json.loads(
        (data_root / "reports" / "v2_p3_pre_label_refresh_run-build-drift.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["input_identity_unchanged"] is False


@pytest.mark.parametrize("mode", ["missing", "bad_keys", "bad_fp"])
def test_noop_forming_incomplete_or_drifted_assets_are_rejected(tmp_path, mode):
    import json

    data_root = tmp_path / "store"
    store = _complete_store(data_root, "20260731")
    path = data_root / "labels" / "y_ret_5" / "20260731.parquet"
    if mode == "missing":
        path.unlink()
    else:
        frame = pd.read_parquet(path)
        if mode == "bad_keys":
            frame = pd.concat([frame, frame.iloc[[0]].assign(ts_code="extra.SZ")])
        else:
            frame.loc[0, "label_value"] = 0.123
        frame.to_parquet(path, index=False)
    loader = _input_loader(tmp_path / "input", "20260706", "20260731")
    builder = _Builder(loader)
    with pytest.raises(RuntimeError, match="缺标签|键集|登记不一致"):
        daily_update.update_daily_store(
            data_root,
            tmp_path / "backup" / "run-noop",
            start_date=_DAY,
            execute=True,
            store=store,
            loader=loader,
            builder=builder,
            producer=_PRODUCER,
            workspace=_WORKSPACE,
        )
    report = json.loads(
        (data_root / "reports" / "v2_p3_pre_label_refresh_run-noop.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["input_identity_unchanged"] is None


def test_noop_input_drift_reports_false_identity(tmp_path, monkeypatch):
    import json

    data_root = tmp_path / "store"
    store = _complete_store(data_root, "20260731")
    loader = _input_loader(tmp_path / "input", "20260706", "20260731")
    builder = _Builder(loader)
    real_readiness = daily_update._readiness_for_run

    def readiness_then_revise(context):
        readiness = real_readiness(context)
        path = loader.storage.clean_path / "daily" / "2026-07-06.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "close_adj"] = 20.0
        frame.to_parquet(path, index=False)
        return readiness

    monkeypatch.setattr(daily_update, "_readiness_for_run", readiness_then_revise)
    with pytest.raises(daily_update.InputIdentityDriftError, match="输入身份变化"):
        daily_update.update_daily_store(
            data_root,
            tmp_path / "backup" / "run-noop-drift",
            start_date=_DAY,
            execute=True,
            store=store,
            loader=loader,
            builder=builder,
            producer=_PRODUCER,
            workspace=_WORKSPACE,
        )
    report = json.loads(
        (data_root / "reports" / "v2_p3_pre_label_refresh_run-noop-drift.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["input_identity_unchanged"] is False


def test_refresh_write_stage_drift_reports_false_identity(tmp_path, monkeypatch):
    import json

    from src.lazybull.v2.store import label_refresh as label_refresh_module

    data_root = tmp_path / "store"
    store = _complete_store(data_root, "20260803")
    loader = _input_loader(tmp_path / "input", "20260706", "20260803")
    builder = _Builder(loader)
    real_append = label_refresh_module._append_completion_once

    def append_then_revise(store, record):
        result = real_append(store, record)
        path = loader.storage.clean_path / "daily" / "2026-07-06.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "close_adj"] = 20.0
        frame.to_parquet(path, index=False)
        return result

    monkeypatch.setattr(label_refresh_module, "_append_completion_once", append_then_revise)
    backup = tmp_path / "backup" / "run-refresh-drift"
    with pytest.raises(daily_update.InputIdentityDriftError, match="运行前后输入身份变化"):
        daily_update.update_daily_store(
            data_root,
            backup,
            start_date=_DAY,
            execute=True,
            refresh_only=True,
            store=store,
            loader=loader,
            builder=builder,
            producer=_PRODUCER,
            workspace=_WORKSPACE,
        )
    report = json.loads(
        (data_root / "reports" / "v2_p3_pre_label_refresh_run-refresh-drift.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert report["input_identity_unchanged"] is False


def test_failed_report_from_one_shot_io_error_is_also_recovered(tmp_path, monkeypatch):
    import json

    data_root = tmp_path / "store"
    loader = _input_loader(tmp_path / "input", "20260706", "20260731")
    builder = _Builder(loader)
    backup = tmp_path / "backup" / "run-one-shot-report"
    report_path = data_root / "reports" / "v2_p3_pre_label_refresh_run-one-shot-report.json"
    real_write = daily_update._write_json_atomic
    attempts = {"failed": False}

    def fail_once(path, payload):
        if Path(path) == report_path and not attempts["failed"]:
            attempts["failed"] = True
            raise OSError("普通报告 I/O 失败")
        return real_write(path, payload)

    monkeypatch.setattr(daily_update, "_write_json_atomic", fail_once)
    with pytest.raises(OSError, match="普通报告 I/O 失败"):
        _run(
            data_root,
            backup,
            loader,
            builder,
            start_date=_DAY,
            execute=True,
            authorized_minutes=200,
        )
    monkeypatch.setattr(daily_update, "_write_json_atomic", real_write)
    failed = json.loads(report_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["input_identity_unchanged"] is None
    before = PanelDataStore(data_root)
    manifest_fp = before.manifest.content_fingerprint()
    label_bytes = {path: path.read_bytes() for path in (data_root / "labels").glob("*/*.parquet")}

    recovered = _run(
        data_root, backup, loader, builder, start_date=_DAY, execute=True, authorized_minutes=200
    )
    assert recovered["status"] == "report_recovered"
    assert recovered["previous_failure"]["failure_reason"] == "普通报告 I/O 失败"
    recovered_bytes = report_path.read_bytes()
    after = PanelDataStore(data_root)
    assert after.manifest.content_fingerprint() == manifest_fp
    for path, payload in label_bytes.items():
        assert path.read_bytes() == payload
    third = _run(
        data_root, backup, loader, builder, start_date=_DAY, execute=True, authorized_minutes=200
    )
    assert third["status"] == "no_op"
    assert report_path.read_bytes() == recovered_bytes


def test_missing_final_report_is_recovered_once_without_rewriting_assets(tmp_path, monkeypatch):
    data_root = tmp_path / "store"
    loader = _input_loader(tmp_path / "input", "20260706", "20260731")
    builder = _Builder(loader)
    backup = tmp_path / "backup" / "run-report-crash"
    report_path = data_root / "reports" / "v2_p3_pre_label_refresh_run-report-crash.json"
    real_write = daily_update._write_json_atomic

    def fail_final_report(path, payload):
        if Path(path) == report_path:
            raise OSError("模拟最终报告写入失败")
        return real_write(path, payload)

    monkeypatch.setattr(daily_update, "_write_json_atomic", fail_final_report)
    with pytest.raises(OSError, match="模拟最终报告写入失败"):
        _run(
            data_root,
            backup,
            loader,
            builder,
            start_date=_DAY,
            execute=True,
            authorized_minutes=200,
        )
    monkeypatch.setattr(daily_update, "_write_json_atomic", real_write)
    assert not report_path.exists()
    before = PanelDataStore(data_root)
    manifest_fp = before.manifest.content_fingerprint()
    audits = (len(before.manifest.label_refresh_intents()), len(before.manifest.label_refreshes()))
    label_bytes = {path: path.read_bytes() for path in (data_root / "labels").glob("*/*.parquet")}

    recovered = _run(
        data_root, backup, loader, builder, start_date=_DAY, execute=True, authorized_minutes=200
    )
    assert recovered["status"] == "report_recovered" and recovered["recovered"] is True
    assert report_path.exists()
    report_bytes = report_path.read_bytes()
    after = PanelDataStore(data_root)
    assert after.manifest.content_fingerprint() == manifest_fp
    assert (
        len(after.manifest.label_refresh_intents()),
        len(after.manifest.label_refreshes()),
    ) == audits
    for path, payload in label_bytes.items():
        assert path.read_bytes() == payload

    third = _run(
        data_root, backup, loader, builder, start_date=_DAY, execute=True, authorized_minutes=200
    )
    assert third["status"] == "no_op"
    assert report_path.read_bytes() == report_bytes


def test_cli_guards_and_input_identity_collection(tmp_path):
    with pytest.raises(ValueError, match="互斥"):
        daily_update.update_daily_store(
            tmp_path / "store", tmp_path / "backup", refresh_only=True, recover_event="event"
        )
    with pytest.raises(ValueError, match="不得早于"):
        daily_update.update_daily_store(
            tmp_path / "store", tmp_path / "backup", start_date="20260702"
        )
    loader = _input_loader(tmp_path / "input", "20260706", "20260731")
    builder = _Builder(loader)
    report = _run(
        tmp_path / "store",
        tmp_path / "backup" / "run",
        loader,
        builder,
        start_date=_DAY,
        authorized_minutes=200,
    )
    paths = {item["path"] for item in report["input_files"]}
    assert "clean/trade_cal.parquet" in paths and "raw/stock_basic.parquet" in paths
    records = collect_file_identities(
        [loader.storage.root_path / path for path in paths], base_dir=loader.storage.root_path
    )
    assert records == sorted(records, key=lambda item: item["path"])
