# -*- coding: utf-8 -*-
"""P3 pre 脚本入口测试；T2 覆盖 manifest 迁移 dry-run、快照与恰一次审计；T4 覆盖验收工具预检/守卫/负例。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "v2_p3_pre"))

import migrate_store_manifest as migration  # noqa: E402
import verify_iteration_parts as parts  # noqa: E402
import verify_store_iteration as vsi  # noqa: E402

from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.column_groups import LABEL_TABLES, PANEL_GROUPS  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.manifest import Manifest, sha256_16_of_file  # noqa: E402
from src.lazybull.v2.store.panel_builder import bootstrap_manifest  # noqa: E402
from src.lazybull.v2.store.row_domain import migrate_manifest_row_domains  # noqa: E402


def _legacy_store(root: Path) -> dict[str, Path]:
    panel = root / "features" / "panel" / "20260702" / "core.parquet"
    label = root / "labels" / "y_ret_20" / "20260702.parquet"
    panel.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    panel.write_bytes(b"panel-bytes")
    label.write_bytes(b"label-bytes")
    data = {
        "manifest_version": "1",
        "updated_at": "2026-10-06T00:00:00",
        "columns": {},
        "partitions": {
            "panel/20260702/core.parquet": {
                "sha256_16": sha256_16_of_file(panel),
                "rows": 1,
                "written_at": "2026-10-06T00:00:00",
                "zone": "hot",
            }
        },
        "dependencies": {"raw_datasets": [], "factor_functions": {}},
        "labels": {
            "y_ret_20": {
                "partitions": {
                    "20260702.parquet": {
                        "sha256_16": sha256_16_of_file(label),
                        "rows": 1,
                        "written_at": "2026-10-06T00:00:00",
                        "zone": "hot",
                    }
                },
                "sealed_through": "20260702",
            }
        },
        "repairs": [{"rel_path": "panel/20260702/core.parquet", "reason": "历史修复"}],
    }
    manifest_path = root / "features" / "manifest.json"
    manifest_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"manifest": manifest_path, "panel": panel, "label": label}


def _crash_after_migration(tmp_path, monkeypatch, data_root=None, backup_root=None):
    data_root = data_root or tmp_path / "data"
    backup_root = backup_root or tmp_path / "backup" / "report-crash"
    paths = _legacy_store(data_root)
    report_path = migration._report_path(data_root, backup_root)
    real_write = migration._write_json_atomic

    def fail_final_report(path, payload):
        if path == report_path:
            raise OSError("模拟报告写入失败")
        return real_write(path, payload)

    monkeypatch.setattr(migration, "_write_json_atomic", fail_final_report)
    with pytest.raises(OSError, match="模拟报告写入失败"):
        migration.migrate_manifest(data_root, backup_root, execute=True)
    monkeypatch.setattr(migration, "_write_json_atomic", real_write)
    assert len(Manifest(paths["manifest"]).schema_migrations()) == 1
    assert not report_path.exists()
    return data_root, backup_root, paths, report_path


def test_migration_dry_run_execute_and_idempotent_rerun(tmp_path):
    data_root = tmp_path / "data"
    paths = _legacy_store(data_root)
    backup_root = tmp_path / "backup" / "run-1"
    old_main = paths["manifest"].read_bytes()
    old_panel = paths["panel"].read_bytes()
    old_label = paths["label"].read_bytes()
    old_repairs = Manifest(paths["manifest"]).repairs()

    plan = migration.migrate_manifest(data_root, backup_root)
    assert plan["mode"] == "dry_run" and plan["status"] == "planned"
    assert plan["migrate_counts"] == {"panel": 1, "labels": 1}
    assert plan["registered_partition_count"] == 2
    assert not backup_root.exists()
    assert paths["manifest"].read_bytes() == old_main
    assert not (data_root / "reports").exists()

    report = migration.migrate_manifest(data_root, backup_root, execute=True)
    assert report["status"] == "executed"
    assert report["registered_fingerprints_unchanged"] is True
    assert report["repairs_unchanged"] is True
    assert report["readiness_counts"] == {"panel": 1, "labels": 1}
    assert paths["panel"].read_bytes() == old_panel
    assert paths["label"].read_bytes() == old_label
    migrated = Manifest(paths["manifest"])
    assert migrated.repairs() == old_repairs
    assert len(migrated.schema_migrations()) == 1
    audit = migrated.schema_migrations()[0]
    assert audit["migration_id"] == migration.MIGRATION_ID
    assert audit["input_fingerprint"] == plan["input_fingerprint"]
    assert audit["output_fingerprint"] == plan["expected_output_fingerprint"]
    assert report["final_manifest_fingerprint"] != audit["output_fingerprint"]
    assert (backup_root / "manifest.json").exists()
    assert (backup_root / "checkpoint.json").exists()
    assert (backup_root / "snapshot.json").exists()
    assert json.loads((backup_root / "checkpoint.json").read_text(encoding="utf-8"))[
        "empty_checkpoint"
    ]
    assert Path(report["report_path"]).exists()

    before_noop = paths["manifest"].read_bytes()
    noop = migration.migrate_manifest(data_root, backup_root, execute=True)
    assert noop["status"] == "no_op"
    assert paths["manifest"].read_bytes() == before_noop
    assert len(Manifest(paths["manifest"]).schema_migrations()) == 1


def test_migration_rejects_pending_wal_before_any_write(tmp_path):
    data_root = tmp_path / "data"
    _legacy_store(data_root)
    wal = data_root / "features" / "manifest_wal.jsonl"
    wal.write_text('{"op":"old_unfinished"}\n', encoding="utf-8")
    backup_root = tmp_path / "backup" / "run-wal"
    with pytest.raises(RuntimeError, match="未处理 WAL"):
        migration.migrate_manifest(data_root, backup_root)
    assert not backup_root.exists()
    assert wal.exists()


def test_migration_rejects_conflicting_existing_domain(tmp_path):
    data_root = tmp_path / "data"
    paths = _legacy_store(data_root)
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    data["partitions"]["panel/20260702/core.parquet"]["row_domain"] = "structural"
    paths["manifest"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="行域冲突"):
        migration.migrate_manifest(data_root, tmp_path / "backup" / "conflict")


def test_migration_rejects_nonempty_backup_before_mutation(tmp_path):
    data_root = tmp_path / "data"
    paths = _legacy_store(data_root)
    old_main = paths["manifest"].read_bytes()
    backup_root = tmp_path / "backup" / "occupied"
    backup_root.mkdir(parents=True)
    (backup_root / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="备份目录非空"):
        migration.migrate_manifest(data_root, backup_root, execute=True)
    assert paths["manifest"].read_bytes() == old_main
    assert (backup_root / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_migration_report_crash_recovers_from_original_run_snapshot(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    paths = _legacy_store(data_root)
    backup_root = tmp_path / "backup" / "report-crash"
    report_path = migration._report_path(data_root, backup_root)
    real_write = migration._write_json_atomic

    def fail_final_report(path, payload):
        if path == report_path:
            raise OSError("模拟报告写入失败")
        return real_write(path, payload)

    monkeypatch.setattr(migration, "_write_json_atomic", fail_final_report)
    with pytest.raises(OSError, match="模拟报告写入失败"):
        migration.migrate_manifest(data_root, backup_root, execute=True)
    monkeypatch.undo()
    assert len(Manifest(paths["manifest"]).schema_migrations()) == 1
    assert not report_path.exists()

    plan = migration.migrate_manifest(data_root, backup_root)
    assert plan["status"] == "report_recovery_planned"
    assert not report_path.exists()
    report = migration.migrate_manifest(data_root, backup_root, execute=True)
    assert report["status"] == "report_recovered" and report["recovered"] is True
    assert report_path.exists()
    assert len(Manifest(paths["manifest"]).schema_migrations()) == 1
    assert migration.migrate_manifest(data_root, backup_root)["status"] == "no_op"


def test_report_recovery_rejects_mismatched_original_input(tmp_path, monkeypatch):
    data_root, backup_root, paths, report_path = _crash_after_migration(tmp_path, monkeypatch)
    original_path = backup_root / "manifest.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    migrate_manifest_row_domains(original)
    original_path.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(RuntimeError, match="原始 manifest 输入指纹"):
        migration.migrate_manifest(data_root, backup_root, execute=True)
    assert not report_path.exists()
    assert len(Manifest(paths["manifest"]).schema_migrations()) == 1


def test_report_recovery_rejects_mismatched_output_fingerprint(tmp_path, monkeypatch):
    data_root, backup_root, paths, report_path = _crash_after_migration(tmp_path, monkeypatch)
    current = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    current["schema_migrations"][0]["output_fingerprint"] = "mismatched-output"
    paths["manifest"].write_text(json.dumps(current), encoding="utf-8")

    with pytest.raises(RuntimeError, match="迁移后输出指纹"):
        migration.migrate_manifest(data_root, backup_root, execute=True)
    assert not report_path.exists()


def test_report_recovery_resolves_relative_evidence_outside_original_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_root = Path("data")
    backup_root = Path("backup") / "relative-cwd"
    _, _, paths, report_path = _crash_after_migration(
        tmp_path, monkeypatch, data_root=data_root, backup_root=backup_root
    )
    assert paths["manifest"].resolve().is_relative_to(tmp_path)
    current = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    current["schema_migrations"][0]["evidence_path"] = str(report_path.relative_to(tmp_path))
    paths["manifest"].write_text(json.dumps(current), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    report = migration.migrate_manifest(
        tmp_path / "data", tmp_path / "backup" / "relative-cwd", execute=True
    )
    assert report["status"] == "report_recovered"
    assert (tmp_path / report_path).exists()
    assert len(Manifest(tmp_path / paths["manifest"]).schema_migrations()) == 1


@pytest.mark.parametrize("legacy_relative", [False, True])
def test_nested_relative_data_root_noop_and_original_path_recovery(
    tmp_path, monkeypatch, legacy_relative
):
    monkeypatch.chdir(tmp_path)
    data_root = Path("nested") / "case" / "data"
    backup_root = Path("nested") / "case" / "backup" / "run-1"
    paths = _legacy_store(data_root)
    first = migration.migrate_manifest(data_root, backup_root, execute=True)
    assert first["status"] == "executed"
    original_report = Path(first["report_path"])
    assert original_report.is_absolute() and original_report.exists()
    if legacy_relative:
        current = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        current["schema_migrations"][0]["evidence_path"] = str(
            data_root / "reports" / original_report.name
        )
        paths["manifest"].write_text(json.dumps(current), encoding="utf-8")
    original_report_bytes = original_report.read_bytes()
    original_manifest_bytes = paths["manifest"].read_bytes()
    files_before_noop = {path for path in tmp_path.rglob("*") if path.is_file()}

    assert migration.migrate_manifest(data_root, backup_root)["status"] == "no_op"
    assert migration.migrate_manifest(data_root, backup_root, execute=True)["status"] == "no_op"
    assert original_report.read_bytes() == original_report_bytes
    assert paths["manifest"].read_bytes() == original_manifest_bytes
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == files_before_noop
    assert len(Manifest(paths["manifest"]).schema_migrations()) == 1

    original_report.unlink()
    report = migration.migrate_manifest(data_root, backup_root, execute=True)
    assert report["status"] == "report_recovered"
    assert Path(report["report_path"]) == original_report
    assert original_report.exists()
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == files_before_noop
    assert len(Manifest(paths["manifest"]).schema_migrations()) == 1


def test_ambiguous_relative_evidence_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_root = Path("nested") / "ambiguous" / "data"
    backup_root = Path("nested") / "ambiguous" / "backup" / "run-1"
    paths = _legacy_store(data_root)
    migration.migrate_manifest(data_root, backup_root, execute=True)
    current = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    current["schema_migrations"][0]["evidence_path"] = "unknown/reports/missing.json"
    paths["manifest"].write_text(json.dumps(current), encoding="utf-8")

    with pytest.raises(RuntimeError, match="无法确定原数据根"):
        migration.migrate_manifest(data_root, backup_root)


def test_daily_update_cli_defaults_to_dry_run_and_forwards_execute(monkeypatch, capsys):
    import update_daily_store as daily

    seen = []

    def fake_update(data_root, backup_root, **kwargs):
        seen.append((data_root, backup_root, kwargs))
        return {"status": "planned"}

    monkeypatch.setattr(daily, "update_daily_store", fake_update)
    assert daily.main(["--data-root", "data", "--backup-root", "backup/run"]) == 0
    assert seen[-1][2]["execute"] is False
    assert json.loads(capsys.readouterr().out) == {"status": "planned"}
    assert daily.main(["--data-root", "data", "--backup-root", "backup/run", "--execute"]) == 0
    assert seen[-1][2]["execute"] is True


def test_daily_update_cli_rejects_conflicting_or_frozen_arguments():
    import update_daily_store as daily

    with pytest.raises(SystemExit):
        daily._parse_args(["--backup-root", "backup/run", "--refresh-only", "--recover-event", "e"])
    with pytest.raises(SystemExit):
        daily._parse_args(["--backup-root", "backup/run", "--start-date", "20260702"])
    with pytest.raises(SystemExit):
        daily._parse_args(["--backup-root", "backup/run", "--authorized-minutes", "0"])


# ======================================================================
# T4：verify_store_iteration 合成预检 / 授权守卫 / 验收负例
# ======================================================================

_VCAL = pd.bdate_range("2026-06-01", "2026-09-30").strftime("%Y%m%d").tolist()
_VDAY = "20260702"
_VNEXT = "20260703"
_VCODES = [f"00000{i}.SZ" for i in range(4)]
_LABEL_OUT = ["ts_code", "trade_date", "label_value", "neu_label_value", "maturity_status"]


def _v_panel(date: str) -> pd.DataFrame:
    n = len(_VCODES)
    return pd.DataFrame(
        {
            "trade_date": [date] * n,
            "ts_code": _VCODES,
            "sw_industry": ["行业A"] * n,
            "sw_industry_code": ["A"] * n,
            "sw_l1_code": ["L1"] * n,
            "sw_l2_code": ["A"] * n,
            "tradable": [1] * n,
        }
    )


def _v_write_panel(store: PanelDataStore, panel: pd.DataFrame) -> None:
    date = panel["trade_date"].iloc[0]
    for group in PANEL_GROUPS:
        if group == "core":
            frame = panel.copy()
        else:
            column = next(c for c in PANEL_GROUPS[group] if c not in {"trade_date", "ts_code"})
            frame = panel[["trade_date", "ts_code"]].assign(**{column: 1.0})
        store.append_features(TradeDate.from_str(date), group, frame)


def _v_label_table(panel: pd.DataFrame, status: str) -> pd.DataFrame:
    return panel[["ts_code", "trade_date"]].assign(
        label_value=0.01, neu_label_value=0.005, maturity_status=status
    )[_LABEL_OUT]


def _v_prod_store(root: Path, date: str = _VDAY, status: str = "sealed"):
    store = PanelDataStore(root)
    bootstrap_manifest(store)
    panel = _v_panel(date)
    _v_write_panel(store, panel)
    for name in LABEL_TABLES:
        store.append_labels(name, _v_label_table(panel, status))
    return store, panel


def _v_ctx(store, watermark: str = "20260731") -> vsi._Context:
    return vsi._Context(
        store=store,
        loader=None,
        builder=None,
        calendar=list(_VCAL),
        watermark=watermark,
        producer={},
        workspace={},
    )


def _v_daily_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for index, date in enumerate(_VCAL):
        codes = [*_VCODES, "999999.SZ"]
        base = np.arange(len(codes), dtype=float) + 10.0 + index / 1000
        frames[date] = pd.DataFrame(
            {
                "trade_date": [date] * len(codes),
                "ts_code": codes,
                "close_adj": base,
                "open_adj": base * 1.02,
            }
        )
    return frames


def _v_refresh_kwargs(daily: dict[str, pd.DataFrame]) -> dict:
    return {
        "trading_dates": list(_VCAL),
        "daily_adj_dict": daily,
        "input_watermark": "20260731",
        "shenwan_level": "l2",
        "producer": {"producer_digest": "producer-a"},
        "run_id": "run-a",
    }


def test_verify_cli_defaults_and_argument_guards():
    args = vsi._parse_args(
        ["--dates", "20260731,20260703", "--scratch-root", "s", "--out", "o.json"]
    )
    assert args.dates == ["20260703", "20260731"]
    assert args.historical_shadow_count == 20
    assert args.shadow_budget_seconds == 60.0
    assert args.authorized_minutes == 60.0
    with pytest.raises(SystemExit):
        vsi._parse_args(["--dates", "20260702", "--scratch-root", "s", "--out", "o"])
    with pytest.raises(SystemExit):
        vsi._parse_args(["--dates", "2026070", "--scratch-root", "s", "--out", "o"])
    with pytest.raises(SystemExit):
        vsi._parse_args(["--dates", "20260703,20260703", "--scratch-root", "s", "--out", "o"])
    with pytest.raises(SystemExit):
        vsi._parse_args(
            [
                "--dates",
                "20260703",
                "--scratch-root",
                "s",
                "--out",
                "o",
                "--historical-shadow-count",
                "0",
            ]
        )
    with pytest.raises(SystemExit):
        vsi._parse_args(
            [
                "--dates",
                "20260703",
                "--scratch-root",
                "s",
                "--out",
                "o",
                "--shadow-budget-seconds",
                "0",
            ]
        )
    with pytest.raises(SystemExit):
        vsi._parse_args(
            [
                "--dates",
                "20260703",
                "--scratch-root",
                "s",
                "--out",
                "o",
                "--authorized-minutes",
                "0",
            ]
        )


def test_verify_cli_main_return_codes(monkeypatch):
    seen = []

    def fake_verify(data_root, scratch_root, out_path, **kwargs):
        seen.append((data_root, scratch_root, out_path, kwargs))
        return {"status": "passed"}

    monkeypatch.setattr(vsi, "verify_store_iteration", fake_verify)
    rc = vsi.main(["--dates", "20260703", "--scratch-root", "s", "--out", "o.json"])
    assert rc == 0
    assert seen[-1][3]["shadow_count"] == 20
    monkeypatch.setattr(
        vsi, "verify_store_iteration", lambda *a, **k: {"status": "precheck_failed"}
    )
    assert vsi.main(["--dates", "20260703", "--scratch-root", "s", "--out", "o.json"]) == 1


def test_endpoints_and_sealed_candidates_are_mechanical():
    buy, endpoint = vsi._endpoints_of(_VCAL, _VDAY)
    assert buy == _VNEXT and endpoint == _VCAL[_VCAL.index(_VDAY) + 21]
    assert vsi._endpoints_of(_VCAL, _VCAL[-1]) == (None, None)
    assert vsi._sealed_candidates(_VCAL, "20260731") == []
    candidates = vsi._sealed_candidates(_VCAL, "20260803")
    assert candidates == [_VNEXT]  # 反证：水位覆盖 E20 时候选非空，检查不是恒空摆设


def test_select_shadow_days_picks_most_recent_mature_frozen_days():
    days = vsi._select_shadow_days(_VCAL, "20260731", 3)
    assert days == ["20260630", "20260701", "20260702"]
    assert len(vsi._select_shadow_days(_VCAL, "20260731", 20)) == 20
    earlier = vsi._select_shadow_days(_VCAL, "20260701", 20)
    assert earlier[-1] < days[-1]  # 水位前移时影子尾日前移，不凑数


def test_mechanical_precheck_passes_with_valid_samples(tmp_path):
    store, _ = _v_prod_store(tmp_path / "prod")
    precheck = vsi._mechanical_precheck(_v_ctx(store), [_VNEXT], 1)
    assert precheck["shadow_days"] == [_VDAY]
    entry = precheck["daily_entries"][0]
    assert entry["expected_status"] == "forming" and entry["row_domain"] == "structural"
    assert precheck["sealed_candidates"] == []


def test_mechanical_precheck_rejects_sealable_or_frozen_date(tmp_path):
    store, _ = _v_prod_store(tmp_path / "prod")
    with pytest.raises(vsi.AcceptanceError, match="状态与预检不符"):
        vsi._mechanical_precheck(_v_ctx(store, watermark="20260803"), [_VNEXT], 1)
    with pytest.raises(vsi.AcceptanceError, match="structural"):
        vsi._mechanical_precheck(_v_ctx(store), [_VDAY], 1)


def test_mechanical_precheck_rejects_short_shadow_and_existing_segment(tmp_path):
    store, _ = _v_prod_store(tmp_path / "prod")
    with pytest.raises(vsi.AcceptanceError, match="历史影子不足"):
        vsi._mechanical_precheck(_v_ctx(store), [_VNEXT], 25)  # 合成日历可选不足 25 日
    with pytest.raises(vsi.AcceptanceError, match="缺 panel core 登记"):
        vsi._mechanical_precheck(_v_ctx(store), [_VNEXT], 2)  # 20260701 未登记不得凑数
    panel = _v_panel(_VNEXT)
    _v_write_panel(store, panel)
    for name in LABEL_TABLES:
        store.append_labels(name, _v_label_table(panel, "forming"))
    with pytest.raises(vsi.AcceptanceError, match="禁止重复初建"):
        vsi._mechanical_precheck(_v_ctx(store), [_VNEXT], 1)


def test_shadow_integrity_rejects_forming_or_missing_tables(tmp_path):
    store, _ = _v_prod_store(tmp_path / "prod", status="forming")
    with pytest.raises(vsi.AcceptanceError, match="不是 sealed|封存水位"):
        vsi._assert_shadow_sample_integrity(store, [_VDAY])


def test_shadow_fixture_sets_forming_and_clears_watermark(tmp_path):
    prod, _ = _v_prod_store(tmp_path / "prod")
    scratch = PanelDataStore(tmp_path / "scratch")
    bootstrap_manifest(scratch)
    parts.copy_shadow_day_to_scratch(prod, scratch, _VDAY)
    parts.assert_forming_fixture(scratch, [_VDAY])
    parts.assert_fixture_panel_fidelity(prod, scratch, [_VDAY])
    for name in LABEL_TABLES:
        assert scratch.manifest.label_sealed_through(name) is None


def test_shadow_fixture_fidelity_detects_tampering(tmp_path):
    prod, _ = _v_prod_store(tmp_path / "prod")
    scratch = PanelDataStore(tmp_path / "scratch")
    bootstrap_manifest(scratch)
    parts.copy_shadow_day_to_scratch(prod, scratch, _VDAY)
    core_path = scratch.panel_dir / _VDAY / "core.parquet"
    frame = pd.read_parquet(core_path)
    frame.loc[0, "tradable"] = 0
    frame.to_parquet(core_path, index=False)
    with pytest.raises(vsi.AcceptanceError, match="core 与生产不一致"):
        parts.assert_fixture_panel_fidelity(prod, scratch, [_VDAY])


def test_forming_fixture_assertion_rejects_sealed_leftover(tmp_path):
    prod, _ = _v_prod_store(tmp_path / "prod")
    scratch = PanelDataStore(tmp_path / "scratch")
    bootstrap_manifest(scratch)
    parts.copy_shadow_day_to_scratch(prod, scratch, _VDAY)
    path = scratch.labels_dir / "y_ret_5" / f"{_VDAY}.parquet"
    frame = pd.read_parquet(path)
    sealed = frame.assign(maturity_status="sealed")  # 原 sealed 副本漏置 forming 的反变异
    sealed.to_parquet(path, index=False)
    with pytest.raises(vsi.AcceptanceError, match="不是 forming|水位未清空"):
        parts.assert_forming_fixture(scratch, [_VDAY])


def test_reconcile_shadow_day_exact_and_detects_value_or_row_drift(tmp_path):
    prod = PanelDataStore(tmp_path / "prod")
    scratch = PanelDataStore(tmp_path / "scratch")
    panel = _v_panel(_VDAY)
    for root in (prod.labels_dir, scratch.labels_dir):
        for name in LABEL_TABLES:
            path = root / name / f"{_VDAY}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            _v_label_table(panel, "sealed").to_parquet(path, index=False)
    assert vsi._reconcile_shadow_day(prod, scratch, _VDAY) == []

    drifted = _v_label_table(panel, "sealed")
    drifted.loc[0, "label_value"] = 0.5
    drifted.to_parquet(scratch.labels_dir / "y_ret_5" / f"{_VDAY}.parquet", index=False)
    problems = vsi._reconcile_shadow_day(prod, scratch, _VDAY)
    assert len(problems) == 1 and "y_ret_5" in problems[0]

    short = _v_label_table(panel, "sealed").iloc[:-1]
    short.to_parquet(scratch.labels_dir / "y_ret_10" / f"{_VDAY}.parquet", index=False)
    problems = vsi._reconcile_shadow_day(prod, scratch, _VDAY)
    assert any("y_ret_10" in problem for problem in problems)

    schema_drift = _v_label_table(panel, "sealed").drop(columns=["neu_label_value"])
    schema_drift.to_parquet(scratch.labels_dir / "y_ret_20" / f"{_VDAY}.parquet", index=False)
    problems = vsi._reconcile_shadow_day(prod, scratch, _VDAY)
    assert any("y_ret_20" in problem and "列集" in problem for problem in problems)


def test_shadow_transitions_reject_noop_or_wrong_domain():
    good = {
        "trade_date": _VDAY,
        "status": "refreshed",
        "actual_status": "sealed",
        "row_domain": "label_filtered",
    }
    vsi._assert_shadow_transitions([good], 1)
    with pytest.raises(vsi.AcceptanceError, match="未实际转换"):
        vsi._assert_shadow_transitions([{**good, "status": "no_op"}], 1)  # 零转换不得充数
    with pytest.raises(vsi.AcceptanceError, match="行域异常"):
        vsi._assert_shadow_transitions([{**good, "row_domain": "structural"}], 1)
    with pytest.raises(vsi.AcceptanceError, match="转换日数"):
        vsi._assert_shadow_transitions([good], 2)


def test_noop_rerun_check_on_refreshed_store(tmp_path):
    store, _ = _v_prod_store(tmp_path / "scratch", status="forming")
    daily = _v_daily_frames()
    kwargs = _v_refresh_kwargs(daily)
    results, _ = vsi._refresh_shadow_days(store, [_VDAY], kwargs)
    vsi._assert_shadow_transitions(results, 1)
    summary = vsi._noop_rerun_check(store, [_VDAY], kwargs)
    assert summary["all_no_op"] is True and summary["audit_counts"] == [1, 1]


def test_appendix_b_report_marks_mismatch_and_missing(tmp_path):
    data_root = tmp_path / "data"
    target = data_root / "raw" / "x.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"payload")
    good = sha256_16_of_file(target)
    table = [("raw/x.parquet", "x", good), ("raw/missing.parquet", "y", "0" * 16)]
    report = parts.appendix_b_report(data_root, table)
    assert report["all_match"] is False and report["matched"] == 1
    missing = report["items"][1]
    assert missing["actual"] is None and missing["match"] is False
    ok = parts.appendix_b_report(data_root, [("raw/x.parquet", "x", good)])
    assert ok["all_match"] is True
    target.write_bytes(b"tampered")
    assert parts.appendix_b_report(data_root, [("raw/x.parquet", "x", good)])["all_match"] is False


def test_frozen_registered_mapping_excludes_structural_segment(tmp_path):
    store, _ = _v_prod_store(tmp_path / "prod")
    panel = _v_panel(_VNEXT)
    _v_write_panel(store, panel)
    for name in LABEL_TABLES:
        store.append_labels(name, _v_label_table(panel, "forming"))
    manifest = parts.Manifest(store.features_dir / "manifest.json")
    mapping = parts.frozen_registered_mapping(manifest)
    assert any(key.startswith(f"panel/{_VDAY}/") for key in mapping)
    assert any(key.endswith(f"/{_VDAY}.parquet") for key in mapping)
    assert not any(_VNEXT in key for key in mapping)


def test_check_file_against_registration(tmp_path):
    path = tmp_path / "f.parquet"
    path.write_bytes(b"bytes")
    fingerprint = sha256_16_of_file(path)
    assert parts.check_file_against_registration(fingerprint, path, "t")["match"] is True
    assert parts.check_file_against_registration("0" * 16, path, "t")["match"] is False
    assert parts.check_file_against_registration(None, path, "t")["match"] is False
    assert (
        parts.check_file_against_registration(fingerprint, tmp_path / "none", "t")["match"] is False
    )


def test_validate_roots_guards(tmp_path):
    data_root = tmp_path / "data"
    with pytest.raises(vsi.AcceptanceError, match="manifest 缺失"):
        vsi._validate_roots(data_root, tmp_path / "scratch")
    manifest = data_root / "features" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(vsi.AcceptanceError, match="不得落在生产数据根内"):
        vsi._validate_roots(data_root, data_root / "scratch")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(vsi.AcceptanceError, match="scratch 根非空"):
        vsi._validate_roots(data_root, scratch)


def test_run_id_of_report_path():
    assert vsi._run_id_of(Path("data/reports/v2_p3_pre_store_acceptance_run1.json")) == "run1"
    assert vsi._run_id_of(Path("other.json")) == "other"


def test_production_source_identities_cover_panel_and_labels(tmp_path):
    store, _ = _v_prod_store(tmp_path / "prod")
    records = parts.production_source_identities(store, [_VDAY])
    assert len(records) == len(PANEL_GROUPS) + len(LABEL_TABLES)
    assert all(record["sha256"] and record["size"] > 0 for record in records)
    (store.labels_dir / "y_ret_5" / f"{_VDAY}.parquet").unlink()
    with pytest.raises(vsi.AcceptanceError, match="来源文件缺失"):
        parts.production_source_identities(store, [_VDAY])


def test_refresh_shadow_days_uses_same_primitive_with_frozen_domain(tmp_path):
    store, _ = _v_prod_store(tmp_path / "scratch", status="forming")
    kwargs = _v_refresh_kwargs(_v_daily_frames())
    results, elapsed = vsi._refresh_shadow_days(store, [_VDAY], kwargs)
    assert elapsed >= 0
    assert results[0]["row_domain"] == "label_filtered"
    assert results[0]["endpoint_date"] == _VCAL[_VCAL.index(_VDAY) + 21]
    # 生产 planner 不选择影子日
    assert vsi.plan_label_refreshes(store, [_VDAY], _VCAL, "20260731") == []


# ======================================================================
# T4 评审整改反例（reviewer1 M1~M5 / reviewer2 R2-T4-01~06）
# ======================================================================


def test_report_path_rejected_preserves_existing_evidence(tmp_path):
    """M3/R2-T4-01：报告已存在/命中生产资产/scratch/tmp 残留均拒绝且原字节不变。"""
    data_root = tmp_path / "data"
    manifest = data_root / "features" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    out = tmp_path / "reports" / "existing.json"
    out.parent.mkdir()
    original = b'{"previous": "passed"}'
    out.write_bytes(original)
    report = vsi.verify_store_iteration(
        data_root, tmp_path / "scratch", out, dates=["20260703"], shadow_count=1
    )
    assert report["status"] == "rejected"
    assert out.read_bytes() == original
    report = vsi.verify_store_iteration(
        data_root,
        tmp_path / "scratch2",
        data_root / "features" / "manifest.json",
        dates=["20260703"],
        shadow_count=1,
    )
    assert report["status"] == "rejected"
    assert json.loads(manifest.read_text(encoding="utf-8")) == {}
    report = vsi.verify_store_iteration(
        data_root,
        tmp_path / "s3",
        tmp_path / "s3" / "r.json",
        dates=["20260703"],
        shadow_count=1,
    )
    assert report["status"] == "rejected"
    fresh = tmp_path / "reports" / "fresh.json"
    fresh.with_name("fresh.json.tmp").write_text("x", encoding="utf-8")
    report = vsi.verify_store_iteration(
        data_root, tmp_path / "s4", fresh, dates=["20260703"], shadow_count=1
    )
    assert report["status"] == "rejected"
    assert not fresh.exists()


def test_underlying_exception_lands_failed_report(tmp_path, monkeypatch):
    """M5/R2-T4-05：下层具体异常同样落 failed 报告并记录类别与阶段。"""
    data_root = tmp_path / "data"
    manifest = data_root / "features" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    out = tmp_path / "reports" / "fail.json"

    def _boom(*args, **kwargs):
        raise ValueError("synthetic missing trade calendar")

    monkeypatch.setattr(vsi, "_open_context", _boom)
    report = vsi.verify_store_iteration(
        data_root, tmp_path / "scratch", out, dates=["20260703"], shadow_count=1
    )
    assert report["status"] == "precheck_failed"
    assert report["exception_type"] == "ValueError"
    assert report["failed_stage"] == "open_context"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["failure_reason"].startswith("ValueError")


def test_estimate_guard_rejects_before_any_write(tmp_path):
    """S6：预计超授权在 open_context 与 scratch 写入前拒绝，仍落 failed 报告。"""
    data_root = tmp_path / "data"
    manifest = data_root / "features" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    scratch = tmp_path / "scratch"
    out = tmp_path / "reports" / "guard.json"
    report = vsi.verify_store_iteration(
        data_root, scratch, out, dates=["20260703"], shadow_count=1, authorized_minutes=0.001
    )
    assert report["status"] == "precheck_failed"
    assert report["failed_stage"] == "estimate"
    assert "超过本次授权" in report["failure_reason"]
    assert not scratch.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["estimated_seconds"] > 0


def test_estimate_acceptance_seconds_scales_with_scope():
    base = vsi._estimate_acceptance_seconds(1, 1)
    assert vsi._estimate_acceptance_seconds(2, 1) > base
    assert vsi._estimate_acceptance_seconds(1, 20) > base
    # §19.9：构建/参照按 T4 实测 500s/日估算，默认授权 60 分钟覆盖冻结命令形态
    assert vsi._estimate_acceptance_seconds(2, 20) <= 60 * 60


def test_daily_day_state_rejects_label_keyset_mismatch(tmp_path):
    """R2-T4-02：共享 readiness 拒绝缺键标签，不得仅凭行域/status 放行。"""
    store = PanelDataStore(tmp_path / "daily")
    bootstrap_manifest(store)
    panel = _v_panel(_VNEXT)
    _v_write_panel(store, panel)
    for name in LABEL_TABLES:
        store.append_labels(name, _v_label_table(panel.iloc[:1], "forming"))  # 缺键标签
    with pytest.raises(RuntimeError, match="键集"):
        vsi._assert_daily_day_state(store, _VNEXT, panel, _VCAL, "20260731")


class _FakeStorage:
    def __init__(self, root: Path):
        self.root_path = str(root)
        self.clean_path = root / "clean"
        self.raw_path = root / "raw"


class _FakeLoader:
    def __init__(self, root: Path):
        self.storage = _FakeStorage(root)


def _fake_pipeline_root(root: Path) -> Path:
    for dataset in ("daily", "daily_basic", "moneyflow"):
        directory = root / "clean" / dataset
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "2026-07-31.parquet").write_bytes(b"x")
    (root / "clean" / "stock_basic.parquet").write_bytes(b"x")
    (root / "clean" / "trade_cal.parquet").write_bytes(b"x")
    (root / "raw" / "stock_basic.parquet").parent.mkdir(parents=True, exist_ok=True)
    (root / "raw" / "stock_basic.parquet").write_bytes(b"x")
    (root / "raw" / "trade_cal.parquet").write_bytes(b"x")
    (root / "raw" / "shenwan_industry.parquet").write_bytes(b"x")
    top_inst = root / "raw" / "top_inst"
    top_inst.mkdir(parents=True, exist_ok=True)
    (top_inst / "2024-12-31.parquet").write_bytes(b"x")  # 早于窗口：全年份读取必须纳入
    (top_inst / "2025-12-31.parquet").write_bytes(b"x")
    (top_inst / "2026-12-31.parquet").write_bytes(b"x")
    for relative, _, _ in parts.APPENDIX_B:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (root / "features" / "cs_infer").mkdir(parents=True, exist_ok=True)
    (root / "features" / "cs_infer" / "20260702.parquet").write_bytes(b"x")
    (root / "features" / "cs_train").mkdir(parents=True, exist_ok=True)
    (root / "features" / "cs_train" / "20260702.parquet").write_bytes(b"x")
    return root


def test_pipeline_identities_cover_pipeline_inputs_and_detect_drift(tmp_path):
    """M1/R2-T4-03：身份清单覆盖二次预热窗口、raw 日历与 top_inst 全年；漂移必须被发现。"""
    root = _fake_pipeline_root(tmp_path / "data")
    loader = _FakeLoader(root)
    identities, left = parts.pipeline_input_identities(
        loader,
        ["20260701", "20260702", "20260703"],
        ["20260702"],
        ["20260703"],
        "20260731",
        lambda date: "20251203",
    )
    assert left == "20250503"  # 捕获起点 20251203 再减 7 个自然月（pipeline.py:384）
    paths = {record["path"] for record in identities}
    assert "clean/stock_basic.parquet" in paths
    assert "clean/daily_basic/2026-07-31.parquet" in paths
    assert "raw/trade_cal.parquet" in paths
    assert "raw/top_inst/2024-12-31.parquet" in paths
    assert "features/cs_infer/20260702.parquet" in paths
    assert parts.verify_file_identities(identities, base_dir=root) == []
    (root / "clean" / "stock_basic.parquet").write_bytes(b"tampered")
    assert parts.verify_file_identities(identities, base_dir=root) == ["clean/stock_basic.parquet"]


def test_recording_parquet_reads_captures_actual_reads(tmp_path):
    """R2-T4-03：读取层记录首次读取即哈希；清单外文件被读即入清单，篡改后必被拒。"""
    root = _fake_pipeline_root(tmp_path / "data")
    preloaded = parts.collect_file_identities([root / "raw" / "stock_basic.parquet"], base_dir=root)
    warmup = root / "clean" / "daily" / "2025-06-02.parquet"
    frame = pd.DataFrame({"trade_date": ["20250602"], "ts_code": ["000001.SZ"], "v": [1.0]})
    frame.to_parquet(warmup, index=False)  # 基线清单之外的分区
    with parts.recording_parquet_reads(root, preloaded) as seen:
        pd.read_parquet(warmup)
        pd.read_parquet(warmup)  # 重复读取不重复哈希
    identities = list(seen.values())
    paths = {record["path"] for record in identities}
    assert paths == {"raw/stock_basic.parquet", "clean/daily/2025-06-02.parquet"}
    assert parts.verify_file_identities(identities, base_dir=root) == []
    frame.assign(v=[2.0]).to_parquet(warmup, index=False)
    assert parts.verify_file_identities(identities, base_dir=root) == [
        "clean/daily/2025-06-02.parquet"
    ]


def test_directory_listing_evidence_detects_add_and_delete(tmp_path):
    """R2-T4-03：raw/clean 全数据集与 store 资产的文件集合新增/删除必须可检出。"""
    root = _fake_pipeline_root(tmp_path / "data")
    before = parts.directory_listing_evidence(root)
    assert before["clean/daily"] == ["2026-07-31.parquet"]
    assert before["raw/top_inst"] == [
        "2024-12-31.parquet",
        "2025-12-31.parquet",
        "2026-12-31.parquet",
    ]
    assert before["features/panel"] is None  # 假根无该目录；缺失也是可比对状态
    assert parts.diff_directory_listing(before, before) == {}
    # 其他 raw 家族新增分区不得以"首次读取"名义蒙混
    fina = root / "raw" / "fina_indicator"
    fina.mkdir(parents=True, exist_ok=True)
    (fina / "2026-06-30.parquet").write_bytes(b"new")
    diff = parts.diff_directory_listing(before, parts.directory_listing_evidence(root))
    assert diff == {"raw/fina_indicator": ["+2026-06-30.parquet"]}
    (root / "features" / "cs_infer" / "20260702.parquet").unlink()
    diff = parts.diff_directory_listing(before, parts.directory_listing_evidence(root))
    assert "features/cs_infer" in diff and "raw/fina_indicator" in diff


def test_recording_detects_replace_between_read_and_verify(tmp_path, monkeypatch):
    """R2-T4-03-A：读取取得旧内容 A、磁盘被替换为 B 时，读取当次即拒绝。"""
    root = _fake_pipeline_root(tmp_path / "data")
    target = root / "raw" / "fina_indicator" / "2026-03-31.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"v": [9.0]}).to_parquet(target, index=False)
    real_read = pd.read_parquet

    def _swap_after_read(path, *args, **kwargs):
        frame = real_read(path, *args, **kwargs)
        pd.DataFrame({"v": [19.0]}).to_parquet(path, index=False)  # 读 A 后磁盘换成 B
        return frame

    monkeypatch.setattr(pd, "read_parquet", _swap_after_read)
    with pytest.raises(vsi.AcceptanceError, match="读取期间输入漂移"):
        with parts.recording_parquet_reads(root):
            pd.read_parquet(target)


def test_frozen_after_check_detects_disk_manifest_edit(tmp_path, monkeypatch):
    """R2-T4-04：仅磁盘 manifest/repairs 改变（缓存实例仍存活）必须被后验发现。"""
    monkeypatch.setattr(parts, "physical_spot_checks", lambda store, manifest: [])
    store, _ = _v_prod_store(tmp_path / "prod")
    data_root = tmp_path / "prod"
    before = parts.capture_frozen_before(store, data_root)
    summary = parts.frozen_segment_after_check(store, data_root, before)
    assert summary["manifest_file_identity_unchanged"] is True
    manifest_path = store.features_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.setdefault("repairs", []).append(
        {"rel_path": "panel/20260702/core.parquet", "reason": "外部篡改"}
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(vsi.AcceptanceError, match="manifest 文件在验收前后变化"):
        parts.frozen_segment_after_check(store, data_root, before)


def test_fault_injection_check_six_scenarios_on_synthetic(tmp_path):
    """R2-T4-06：六场景故障门；活动 WAL 现场保留、全新实例重放、终态可独立重建。"""
    prod, _ = _v_prod_store(tmp_path / "prod")
    kwargs = _v_refresh_kwargs(_v_daily_frames())
    summary = parts.fault_injection_check(prod, tmp_path / "scratch", _VDAY, kwargs)
    names = [scenario["scenario"] for scenario in summary["scenarios"]]
    assert names == [
        "intent_crash_clean_rerun",
        "partial_tables_live_wal",
        "completion_crash_live_wal",
        "manifest_save_crash_wal_replay",
        "replace_wal_window_controlled_rollforward",
        "same_generation_rollback",
    ]
    live = {s["scenario"]: s.get("wal_live_at_crash") for s in summary["scenarios"]}
    assert live["partial_tables_live_wal"] is True
    assert live["completion_crash_live_wal"] is True
    assert live["manifest_save_crash_wal_replay"] is True
    evidence = summary["scenarios"][3]["replay_required_evidence"]
    assert evidence == {"disk_manifest_completions": 0, "replayed_audit_counts": [1, 1]}
    assert summary["scenarios"][5]["fault_wal_replayed"] is False


def test_manifest_save_scenario_fails_when_replay_disabled(tmp_path, monkeypatch):
    """R2-T4-06 反变异：禁用 WAL 重放时故障门必须失败。"""
    from src.lazybull.v2.store.manifest_wal import ManifestWal

    prod, _ = _v_prod_store(tmp_path / "prod")
    kwargs = _v_refresh_kwargs(_v_daily_frames())
    monkeypatch.setattr(ManifestWal, "replay", lambda self, **kwargs: None)
    with pytest.raises(vsi.AcceptanceError, match="重放后审计计数异常"):
        parts.manifest_save_crash_scenario(prod, tmp_path / "scratch", _VDAY, kwargs)


def test_shadow_phase_fragment_survives_budget_failure(tmp_path):
    """M2：超预算失败时 per_day/总耗时/build 调用数已写入 fragment（随 failed 报告落盘）。"""
    prod, _ = _v_prod_store(tmp_path / "prod")
    fragment: dict = {}
    with pytest.raises(vsi.AcceptanceError, match="超预算"):
        vsi._historical_shadow_phase(
            _v_ctx(prod),
            tmp_path / "scratch",
            [_VDAY],
            _v_refresh_kwargs(_v_daily_frames()),
            1e-9,
            fragment,
        )
    assert fragment["per_day"] and fragment["total_elapsed_seconds"] >= 0
    assert fragment["build_features_data_calls"] == 0
    assert fragment["transitions"] == 1


def _ti_frame(
    sum_values: list[float], ratio_values: list[float] | None = None, days_20: float = 2.0
) -> pd.DataFrame:
    n = len(sum_values)
    ratios = ratio_values if ratio_values is not None else list(sum_values)
    return pd.DataFrame(
        {
            "trade_date": [_VDAY] * n,
            "ts_code": [f"00000{i}.SZ" for i in range(n)],
            "ti_inst_net_ratio": ratios,
            "ti_inst_buy_ratio": [0.1] * n,
            "ti_inst_sell_ratio": [0.05] * n,
            "ti_inst_net_sum_20": sum_values,
            "ti_inst_days_20": [days_20] * n,
            "ti_schema_v1": [1.0] * n,
        }
    )


def _write_announcement(root: Path, date: str, frame: pd.DataFrame) -> Path:
    path = root / "features" / "panel" / date / "announcement.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def test_ti_layered_criterion_exact_and_tail_tolerance(tmp_path):
    """§19.9 分层判据：全等通过；累计列 1e-17 尾差通过并记录尾差量级。"""
    prod = _write_announcement(tmp_path / "prod", _VDAY, _ti_frame([0.001, -0.002]))
    scratch = _write_announcement(tmp_path / "scratch", _VDAY, _ti_frame([0.001, -0.002]))
    problems, stats = vsi._compare_ti_columns(prod, scratch, _VDAY)
    assert problems == [] and stats["ti_inst_net_sum_20_max_abs"] == 0.0

    prod2 = _write_announcement(tmp_path / "prod2", _VDAY, _ti_frame([0.001, -0.002]))
    # 仅累计列带 cumsum 跨上下文实测尾差量级；精确列保持逐位一致
    tailed = _ti_frame([0.001 + 1.4e-17, -0.002], ratio_values=[0.001, -0.002])
    scratch2 = _write_announcement(tmp_path / "scratch2", _VDAY, tailed)
    problems, stats = vsi._compare_ti_columns(prod2, scratch2, _VDAY)
    assert problems == []
    assert 0.0 < stats["ti_inst_net_sum_20_max_abs"] < 1e-15


def test_ti_layered_criterion_rejects_real_drift(tmp_path):
    """§19.9 分层判据：累计列超容差、精确列差异、窗口计数差异均拒绝。"""
    prod = _write_announcement(tmp_path / "prod", _VDAY, _ti_frame([0.001, -0.002]))
    # 仅累计列超容差；精确列保持逐位一致
    big = _write_announcement(
        tmp_path / "s1", _VDAY, _ti_frame([0.0011, -0.002], ratio_values=[0.001, -0.002])
    )
    problems, _ = vsi._compare_ti_columns(prod, big, _VDAY)
    assert len(problems) == 1 and "超容差" in problems[0]

    exact_drift = _ti_frame([0.001, -0.002])
    exact_drift["ti_inst_net_ratio"] = [0.001, -0.0022]  # 当日值列属精确判据
    exact_path = _write_announcement(tmp_path / "s2", _VDAY, exact_drift)
    problems, _ = vsi._compare_ti_columns(prod, exact_path, _VDAY)
    assert len(problems) == 1 and "精确列不一致" in problems[0]

    days_drift = _ti_frame([0.001, -0.002], days_20=3.0)  # 窗口事件计数属整数精确判据
    days_path = _write_announcement(tmp_path / "s3", _VDAY, days_drift)
    problems, _ = vsi._compare_ti_columns(prod, days_path, _VDAY)
    assert len(problems) == 1 and "精确列不一致" in problems[0]

    problems, _ = vsi._compare_ti_columns(prod, tmp_path / "none.parquet", _VDAY)
    assert problems and "缺失" in problems[0]


def test_ti_layered_criterion_rejects_nan_pattern_mismatch(tmp_path):
    """reviewer2 S1：容差列 NaN 模式不一致或全 NaN 均拒绝。"""
    prod = _write_announcement(tmp_path / "prod", _VDAY, _ti_frame([0.001, -0.002]))
    nan_frame = _ti_frame([0.001, -0.002], ratio_values=[0.001, -0.002])
    nan_frame.loc[0, "ti_inst_net_sum_20"] = float("nan")
    nan_path = _write_announcement(tmp_path / "s1", _VDAY, nan_frame)
    problems, stats = vsi._compare_ti_columns(prod, nan_path, _VDAY)
    assert len(problems) == 1 and "有限性" in problems[0]
    assert stats["ti_inst_net_sum_20_nonfinite_rows"] == 1

    all_nan_prod = _ti_frame([0.001, -0.002], ratio_values=[0.001, -0.002])
    all_nan_prod["ti_inst_net_sum_20"] = float("nan")
    prod_nan = _write_announcement(tmp_path / "prod_nan", _VDAY, all_nan_prod)
    all_nan_scratch = _ti_frame([0.001, -0.002], ratio_values=[0.001, -0.002])
    all_nan_scratch["ti_inst_net_sum_20"] = float("nan")
    nan2_path = _write_announcement(tmp_path / "s2", _VDAY, all_nan_scratch)
    problems, _ = vsi._compare_ti_columns(prod_nan, nan2_path, _VDAY)
    assert len(problems) == 1 and "全部为 NaN" in problems[0]


def test_cross_seam_impact_for_day_structural_nonzero(tmp_path):
    """R2-T4-CLOSE-01：structural 日（y20 未到期为空、y5/y10 有值）产生非零计数；
    label_filtered 日（y20 有值）计数为零；两域不得混用。"""
    store = PanelDataStore(tmp_path / "store")
    bootstrap_manifest(store)
    panel = _v_panel(_VNEXT)
    _v_write_panel(store, panel)
    y20_all_nan = _v_label_table(panel, "forming").assign(
        label_value=float("nan"), neu_label_value=float("nan")
    )
    store.append_labels("y_ret_20", y20_all_nan)
    store.append_labels("y_ret_5", _v_label_table(panel, "forming"))
    store.append_labels("y_ret_10", _v_label_table(panel, "forming"))
    impact = vsi.cross_seam_impact_for_day(store, _VNEXT)
    assert impact["y_ret_5"]["affected_rows"] == len(_VCODES)
    assert impact["y_ret_10"]["affected_rows"] == len(_VCODES)
    assert impact["y_ret_5"]["by_industry"] == {"行业A": len(_VCODES)}

    frozen = PanelDataStore(tmp_path / "frozen")
    bootstrap_manifest(frozen)
    frozen_panel = _v_panel(_VDAY)
    _v_write_panel(frozen, frozen_panel)
    for name in LABEL_TABLES:
        frozen.append_labels(name, _v_label_table(frozen_panel, "sealed"))
    zero = vsi.cross_seam_impact_for_day(frozen, _VDAY)
    assert zero["y_ret_5"]["affected_rows"] == 0
    assert zero["y_ret_10"]["affected_rows"] == 0


def _supplement_original_report(tmp_path: Path, prod_root: Path) -> Path:
    original = {
        "run_id": "orig-run",
        "status": "passed",
        "data_root": str(prod_root),
        "scratch_root": str(tmp_path / "no-such-scratch"),
        "precheck": {"shadow_count": 1, "shadow_range": "20260702~20260702"},
        "historical_shadow": {
            "per_day": [
                {
                    "trade_date": "20260702",
                    "cross_seam_impact": {
                        "y_ret_5": {"affected_rows": 0, "by_industry": {}},
                        "y_ret_10": {"affected_rows": 0, "by_industry": {}},
                    },
                }
            ]
        },
        "real_daily": {"entries": [{"trade_date": _VNEXT}]},
        "identity": {"loaded_watermark": "20260731"},
    }
    original_path = tmp_path / "original.json"
    original_path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    return original_path


def test_supplement_cross_seam_separates_scopes(tmp_path):
    """R2-T4-CLOSE-01：补充报告三范围分列；影子零不代表 structural 非零；非 passed 拒绝。"""
    prod_root = tmp_path / "prod"
    prod_store = PanelDataStore(prod_root)
    bootstrap_manifest(prod_store)
    panel = _v_panel(_VNEXT)
    _v_write_panel(prod_store, panel)
    y20_all_nan = _v_label_table(panel, "forming").assign(
        label_value=float("nan"), neu_label_value=float("nan")
    )
    prod_store.append_labels("y_ret_20", y20_all_nan)
    prod_store.append_labels("y_ret_5", _v_label_table(panel, "forming"))
    prod_store.append_labels("y_ret_10", _v_label_table(panel, "forming"))
    (prod_root / "clean" / "daily").mkdir(parents=True)
    (prod_root / "clean" / "daily" / "2026-07-31.parquet").write_bytes(b"x")
    original_path = _supplement_original_report(tmp_path, prod_root)

    out = tmp_path / "supplement.json"
    report = vsi.supplement_cross_seam(original_path, out, data_root=prod_root)
    scopes = report["scopes"]
    assert scopes["historical_shadow"]["y_ret_5"]["affected_rows"] == 0
    assert scopes["real_daily_scratch"]["available"] is False
    prod_scope = scopes["production_append"]
    assert prod_scope["days"] == 1 and prod_scope["range"] == f"{_VNEXT}~{_VNEXT}"
    assert prod_scope["total"]["y_ret_5"]["affected_rows"] == len(_VCODES)
    assert prod_scope["total"]["y_ret_10"]["affected_rows"] == len(_VCODES)
    assert prod_scope["per_day"][0]["trade_date"] == _VNEXT
    assert (
        report["original_report_sha256"] == hashlib.sha256(original_path.read_bytes()).hexdigest()
    )
    assert report["data_identity"]["clean_dir_watermark"] == "20260731"
    assert report["restriction"].startswith("B0 14 折不跨缝")

    with pytest.raises(vsi.AcceptanceError, match="拒绝覆盖"):
        vsi.supplement_cross_seam(original_path, out, data_root=prod_root)
    failed_original = json.loads(original_path.read_text(encoding="utf-8"))
    failed_original["status"] = "failed"
    failed_path = tmp_path / "failed-original.json"
    failed_path.write_text(json.dumps(failed_original, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(vsi.AcceptanceError, match="非 passed"):
        vsi.supplement_cross_seam(failed_path, tmp_path / "s2.json", data_root=prod_root)


def test_supplement_cli_dispatch(monkeypatch):
    seen = []

    def fake_supplement(original, out, **kwargs):
        seen.append((original, out, kwargs))
        return {"status": "passed"}

    monkeypatch.setattr(vsi, "supplement_cross_seam", fake_supplement)
    rc = vsi.main(["--supplement-cross-seam", "orig.json", "--out", "supplement.json"])
    assert rc == 0 and seen[-1][0] == "orig.json"
    with pytest.raises(SystemExit):
        vsi._parse_args(
            ["--supplement-cross-seam", "orig.json", "--dates", "20260703", "--out", "o.json"]
        )
    with pytest.raises(SystemExit):
        vsi._parse_args(["--out", "o.json"])  # 既无 --dates 也无补充模式


def _supplement_fixture_root(root: Path) -> Path:
    store = PanelDataStore(root)
    bootstrap_manifest(store)
    panel = _v_panel(_VNEXT)
    _v_write_panel(store, panel)
    y20_all_nan = _v_label_table(panel, "forming").assign(
        label_value=float("nan"), neu_label_value=float("nan")
    )
    store.append_labels("y_ret_20", y20_all_nan)
    store.append_labels("y_ret_5", _v_label_table(panel, "forming"))
    store.append_labels("y_ret_10", _v_label_table(panel, "forming"))
    (root / "clean" / "daily").mkdir(parents=True)
    (root / "clean" / "daily" / "2026-07-31.parquet").write_bytes(b"x")
    return root


def test_supplement_rejects_pending_wal_without_side_effects(tmp_path):
    """R2-T4-CLOSE-02：活动 WAL 拒绝；manifest/.bak/WAL 字节与存在性不变、零副作用。"""
    prod_root = _supplement_fixture_root(tmp_path / "prod")
    store = PanelDataStore(prod_root)
    original_path = _supplement_original_report(tmp_path, prod_root)
    # 合法待处理事件：经低层 append 只写 WAL（盘上 manifest 未变、WAL 非空）
    store._wal.append(
        {
            "op": "register_repair",
            "rel_path": "panel/20260702/core.parquet",
            "reason": "合成待处理修复",
            "added_dates": ["20260702"],
            "old_fingerprint": "0" * 16,
            "new_fingerprint": "1" * 16,
        }
    )
    manifest_path = prod_root / "features" / "manifest.json"
    bak_path = prod_root / "features" / "manifest.json.bak"
    wal_path = prod_root / "features" / "manifest_wal.jsonl"
    before = {p: p.read_bytes() for p in (manifest_path, bak_path, wal_path) if p.exists()}
    assert wal_path.exists() and wal_path.stat().st_size > 0
    with pytest.raises(vsi.AcceptanceError, match="活动 WAL"):
        vsi.supplement_cross_seam(original_path, tmp_path / "supplement.json", data_root=prod_root)
    assert not (tmp_path / "supplement.json").exists()
    for path, payload in before.items():
        assert path.read_bytes() == payload
    # WAL 处置（清空）后正常路径仍可通过
    wal_path.unlink()
    report = vsi.supplement_cross_seam(
        original_path, tmp_path / "supplement.json", data_root=prod_root
    )
    assert report["status"] == "passed"
    assert report["scopes"]["production_append"]["total"]["y_ret_5"]["affected_rows"] > 0


def test_supplement_output_path_isolation(tmp_path):
    """R2-T4-CLOSE-03：生产资产/非规范 .json/原子临时目标已存在均拒绝且现场保全。"""
    prod_root = _supplement_fixture_root(tmp_path / "prod")
    original_path = _supplement_original_report(tmp_path, prod_root)
    with pytest.raises(vsi.AcceptanceError, match="规范 .json"):
        vsi.supplement_cross_seam(
            original_path, prod_root / "features" / "manifest.json.tmp", data_root=prod_root
        )
    assert not (prod_root / "features" / "manifest.json.tmp").exists()
    with pytest.raises(vsi.AcceptanceError, match="不在 reports/ 下"):
        vsi.supplement_cross_seam(
            original_path, prod_root / "features" / "x.json", data_root=prod_root
        )
    assert not (prod_root / "features" / "x.json").exists()
    out = tmp_path / "reports" / "supplement.json"
    out.parent.mkdir()
    sentinel = b"sentinel-tmp-bytes"
    out.with_suffix(".json.tmp").write_bytes(sentinel)
    with pytest.raises(vsi.AcceptanceError, match="原子临时目标已存在"):
        vsi.supplement_cross_seam(original_path, out, data_root=prod_root)
    assert not out.exists()
    assert out.with_suffix(".json.tmp").read_bytes() == sentinel
    with pytest.raises(vsi.AcceptanceError, match="规范 .json"):
        vsi.supplement_cross_seam(original_path, tmp_path / "x.txt", data_root=prod_root)


def _wal_line() -> str:
    return (
        '{"op": "register_repair", "rel_path": "panel/20260702/core.parquet", '
        '"reason": "late-injected", "added_dates": ["20260702"], '
        f'"old_fingerprint": "{"0" * 16}", "new_fingerprint": "{"1" * 16}"}}\n'
    )


def test_supplement_rejects_wal_appearing_during_late_reads(tmp_path, monkeypatch):
    """CLOSE-02 残留：后段读取（scratch 范围）期间新增活动 WAL 必须拒绝并保留现场。"""
    prod_root = _supplement_fixture_root(tmp_path / "prod")
    original_path = _supplement_original_report(tmp_path, prod_root)
    wal_path = prod_root / "features" / "manifest_wal.jsonl"
    manifest_before = (prod_root / "features" / "manifest.json").read_bytes()
    real_scope = parts._scratch_daily_scope

    def inject_wal_during_scope(original):
        wal_path.write_text(_wal_line(), encoding="utf-8")
        return real_scope(original)

    monkeypatch.setattr(parts, "_scratch_daily_scope", inject_wal_during_scope)
    with pytest.raises(vsi.AcceptanceError, match="活动 WAL"):
        vsi.supplement_cross_seam(original_path, tmp_path / "out.json", data_root=prod_root)
    assert not (tmp_path / "out.json").exists()
    assert (prod_root / "features" / "manifest.json").read_bytes() == manifest_before
    assert wal_path.exists()  # 现场保留：不删除、不压实、不恢复


def test_supplement_rejects_manifest_change_during_late_reads(tmp_path, monkeypatch):
    """CLOSE-02 残留：后段读取（目录水位）期间 manifest 变化必须拒绝，不得谎报 unchanged。"""
    prod_root = _supplement_fixture_root(tmp_path / "prod")
    original_path = _supplement_original_report(tmp_path, prod_root)
    manifest_path = prod_root / "features" / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    real_watermark = parts._dir_watermark

    def modify_manifest_during_watermark(root):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["dependencies"]["raw_datasets"] = ["tampered-during-read"]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return real_watermark(root)

    monkeypatch.setattr(parts, "_dir_watermark", modify_manifest_during_watermark)
    out = tmp_path / "out.json"
    with pytest.raises(vsi.AcceptanceError, match="状态变化"):
        vsi.supplement_cross_seam(original_path, out, data_root=prod_root)
    assert not out.exists()
    manifest_path.write_bytes(manifest_before)  # tmp 根内复位，便于后续用例


def test_supplement_rejects_output_inside_original_scratch(tmp_path):
    """CLOSE-03 残留：输出落在原报告 scratch_root 内，函数与 CLI 均拒绝且零越界新文件。"""
    prod_root = _supplement_fixture_root(tmp_path / "prod")
    original_path = _supplement_original_report(tmp_path, prod_root)
    scratch_root = Path(json.loads(original_path.read_text(encoding="utf-8"))["scratch_root"])
    out = scratch_root / "inside.json"
    with pytest.raises(vsi.AcceptanceError, match="scratch"):
        vsi.supplement_cross_seam(original_path, out, data_root=prod_root)
    assert not out.exists()
    rc = vsi.main(
        [
            "--supplement-cross-seam",
            str(original_path),
            "--out",
            str(out),
            "--data-root",
            str(prod_root),
        ]
    )
    assert rc == 1
    assert not out.exists()
    assert not scratch_root.exists()  # 原 scratch 现场未被创建或改写
