# -*- coding: utf-8 -*-
"""P3 pre T2 行域派生、迁移 readiness 与写路径保持测试（全部使用合成数据）。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.manifest import Manifest
from src.lazybull.v2.store.panel_builder import bootstrap_manifest
from src.lazybull.v2.store.row_domain import (
    ROW_DOMAIN_SEGMENTS,
    migrate_manifest_row_domains,
    row_domain_for_date,
    row_domain_for_partition,
    validate_manifest_row_domains,
)


def _legacy_snapshot() -> dict:
    return {
        "manifest_version": "1",
        "updated_at": "2026-10-18T00:00:00",
        "columns": {},
        "partitions": {
            "panel/20260702/core.parquet": {
                "sha256_16": "panel_fp",
                "rows": 2,
                "written_at": "2026-10-01T00:00:00",
                "zone": "hot",
            },
            "panel_archive/2025-05/core.parquet": {
                "sha256_16": "archive_fp",
                "rows": 4,
                "written_at": "2026-10-01T00:00:00",
                "zone": "archive",
            },
        },
        "dependencies": {"raw_datasets": [], "factor_functions": {}},
        "labels": {
            "y_ret_20": {
                "partitions": {
                    "20260702.parquet": {
                        "sha256_16": "label_fp",
                        "rows": 2,
                        "written_at": "2026-10-01T00:00:00",
                        "zone": "hot",
                    }
                },
                "sealed_through": "20260702",
            }
        },
        "repairs": [{"rel_path": "panel/20260702/core.parquet", "reason": "保留"}],
    }


def _make_store(tmp_path: Path) -> PanelDataStore:
    store = PanelDataStore(tmp_path)
    store.manifest.register_column("ret_1", "core", "test")
    store.manifest.save()
    return store


def _feature_df(date: str, values=(0.01, -0.02)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["600000.SH", "000001.SZ"],
            "ret_1": list(values),
        }
    )


def test_date_and_partition_boundaries():
    assert row_domain_for_date("20260702") == "label_filtered"
    assert row_domain_for_date("20260703") == "structural"
    assert row_domain_for_partition("panel/20260703/core.parquet") == "structural"
    assert row_domain_for_partition("panel_archive/2025-05/core.parquet") == "label_filtered"
    assert row_domain_for_partition("panel_archive/2026-08/core.parquet") == "structural"


@pytest.mark.parametrize("value", ["2026-07-02", "20260230", "abc"])
def test_invalid_date_rejected(value):
    with pytest.raises(ValueError, match="行域日期"):
        row_domain_for_date(value)


def test_cross_domain_archive_month_rejected_before_write(tmp_path):
    with pytest.raises(ValueError, match="跨行域边界月"):
        row_domain_for_partition("panel_archive/2026-07/core.parquet")
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="跨行域边界月"):
        store.append_archive_features("2026-07", "core", _feature_df("20260702"))
    assert not (tmp_path / "features" / "panel_archive" / "2026-07" / "core.parquet").exists()


def test_migration_is_metadata_only_and_idempotent():
    original = _legacy_snapshot()
    migrated = copy.deepcopy(original)
    counts = migrate_manifest_row_domains(migrated)
    assert counts == {"panel": 2, "labels": 1}
    assert migrated["row_domain_segments"] == ROW_DOMAIN_SEGMENTS
    assert migrated["repairs"] == original["repairs"]
    assert migrated["partitions"]["panel/20260702/core.parquet"]["sha256_16"] == "panel_fp"
    assert migrated["partitions"]["panel/20260702/core.parquet"]["row_domain"] == "label_filtered"
    assert migrated["labels"]["y_ret_20"]["partitions"]["20260702.parquet"]["row_domain"] == (
        "label_filtered"
    )
    assert validate_manifest_row_domains(migrated) == {"panel": 2, "labels": 1}
    assert migrate_manifest_row_domains(migrated) == {"panel": 0, "labels": 0}


def test_conflicting_or_missing_domain_readiness_rejected():
    migrated = copy.deepcopy(_legacy_snapshot())
    migrate_manifest_row_domains(migrated)
    migrated["partitions"]["panel/20260702/core.parquet"]["row_domain"] = "structural"
    with pytest.raises(ValueError, match="行域冲突"):
        migrate_manifest_row_domains(migrated)
    with pytest.raises(ValueError, match="缺失或非法"):
        validate_manifest_row_domains(migrated)

    old_writer = copy.deepcopy(_legacy_snapshot())
    migrate_manifest_row_domains(old_writer)
    old_writer["partitions"]["panel/20260702/core.parquet"] = {
        "sha256_16": "new_fp",
        "rows": 2,
        "written_at": "2026-10-18T01:00:00",
        "zone": "hot",
    }
    with pytest.raises(ValueError, match="缺失或非法"):
        validate_manifest_row_domains(old_writer)


def test_legacy_manifest_requires_explicit_migration_before_register(tmp_path):
    path = tmp_path / "features" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_legacy_snapshot()), encoding="utf-8")
    manifest = Manifest(path)
    with pytest.raises(RuntimeError, match="显式迁移"):
        manifest.register_partition("panel/20260703/core.parquet", "fp", rows=1)
    with pytest.raises(ValueError, match="行域段缺失"):
        manifest.validate_row_domain_readiness()


def test_legacy_manifest_append_rejected_before_physical_write(tmp_path):
    data = _legacy_snapshot()
    data["columns"]["ret_1"] = {
        "group": "core",
        "source": "test",
        "definition_version": "v1",
        "available_from": None,
        "backfilled_at": None,
        "status": "active",
    }
    path = tmp_path / "features" / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    store = PanelDataStore(tmp_path)
    with pytest.raises(RuntimeError, match="行域 schema 未迁移"):
        store.append_features(TradeDate.from_str("20260703"), "core", _feature_df("20260703"))
    assert not (tmp_path / "features" / "panel" / "20260703" / "core.parquet").exists()
    assert not store.wal_path.exists()


def test_dropped_domain_rewrite_rejected_before_replace(tmp_path):
    store = _make_store(tmp_path)
    store.append_features(TradeDate.from_str("20260703"), "core", _feature_df("20260703"))
    rel = "panel/20260703/core.parquet"
    final = tmp_path / "features" / rel
    original = final.read_bytes()
    data = store.manifest.snapshot()
    del data["partitions"][rel]["row_domain"]
    store.manifest.path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="行域缺失或冲突"):
        PanelDataStore(tmp_path).rewrite_partition(
            rel, _feature_df("20260703", values=(9.9, 8.8)), "模拟旧写入者损坏"
        )
    assert final.read_bytes() == original


def test_dropped_label_domain_append_rejected_before_replace(tmp_path):
    store = _make_store(tmp_path)
    labels = pd.DataFrame(
        {
            "trade_date": ["20260703", "20260703"],
            "ts_code": ["600000.SH", "000001.SZ"],
            "label_value": [0.01, 0.02],
            "maturity_status": ["forming", "forming"],
        }
    )
    store.append_labels("y_ret_20", labels)
    final = tmp_path / "labels" / "y_ret_20" / "20260703.parquet"
    original = final.read_bytes()
    data = store.manifest.snapshot()
    del data["labels"]["y_ret_20"]["partitions"]["20260703.parquet"]["row_domain"]
    store.manifest.path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="行域缺失或冲突"):
        PanelDataStore(tmp_path).append_labels("y_ret_20", labels.assign(label_value=[9.9, 8.8]))
    assert final.read_bytes() == original


def test_append_rewrite_resync_and_wal_replay_preserve_domain(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    store.append_features(TradeDate.from_str("20260703"), "core", _feature_df("20260703"))
    rel = "panel/20260703/core.parquet"
    assert store.manifest.partition_meta(rel)["row_domain"] == "structural"
    store.rewrite_partition(rel, _feature_df("20260703", values=(0.1, 0.2)), "测试重写")
    assert store.manifest.partition_meta(rel)["row_domain"] == "structural"

    final = tmp_path / "features" / rel
    _feature_df("20260703", values=(0.3, 0.4)).to_parquet(final, index=False)
    store.resync_partition_fingerprints("测试 resync")
    assert store.manifest.partition_meta(rel)["row_domain"] == "structural"

    def crash_save(self):
        raise RuntimeError("模拟崩溃")

    monkeypatch.setattr(Manifest, "save", crash_save)
    with pytest.raises(RuntimeError, match="模拟崩溃"):
        store.append_features(TradeDate.from_str("20260704"), "core", _feature_df("20260704"))
    monkeypatch.undo()
    replayed = PanelDataStore(tmp_path)
    meta = replayed.manifest.partition_meta("panel/20260704/core.parquet")
    assert meta is not None and meta["row_domain"] == "structural"


def test_archive_repairs_and_label_resync_preserve_domain(tmp_path):
    store = _make_store(tmp_path)
    store.append_archive_features("2025-05", "core", _feature_df("20250506"))
    rel = "panel_archive/2025-05/core.parquet"
    assert store.manifest.partition_meta(rel)["row_domain"] == "label_filtered"
    store.repair_archive_partition("2025-05", "core", _feature_df("20250507"), "补缺日")
    assert store.manifest.partition_meta(rel)["row_domain"] == "label_filtered"
    store.replace_archive_days("2025-05", "core", _feature_df("20250506"), "替换缺陷日")
    assert store.manifest.partition_meta(rel)["row_domain"] == "label_filtered"

    labels = pd.DataFrame(
        {
            "trade_date": ["20260703", "20260703"],
            "ts_code": ["600000.SH", "000001.SZ"],
            "label_value": [0.01, 0.02],
            "maturity_status": ["forming", "forming"],
        }
    )
    store.append_labels("y_ret_20", labels)
    assert store.manifest.label_partition_meta("y_ret_20", "20260703")["row_domain"] == (
        "structural"
    )
    labels.assign(label_value=[0.03, 0.04]).to_parquet(
        tmp_path / "labels" / "y_ret_20" / "20260703.parquet", index=False
    )
    store.resync_partition_fingerprints("标签重同步")
    assert store.manifest.label_partition_meta("y_ret_20", "20260703")["row_domain"] == (
        "structural"
    )


def test_unknown_wal_event_fails_before_apply_or_compaction(tmp_path):
    store = _make_store(tmp_path)
    wal = store.wal_path
    valid = {
        "op": "register_partition",
        "rel_path": "panel/20260106/core.parquet",
        "sha256_16": "new_fp",
        "rows": 2,
        "zone": "hot",
        "row_domain": "label_filtered",
    }
    wal.write_text(
        json.dumps(valid) + "\n" + json.dumps({"op": "future_unknown_event"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="未知 WAL 事件类型"):
        _ = PanelDataStore(tmp_path).manifest
    assert wal.exists()
    on_disk = Manifest(tmp_path / "features" / "manifest.json")
    assert on_disk.partition_fingerprint("panel/20260106/core.parquet") is None


def test_unknown_wal_schema_fails_and_preserves_file(tmp_path):
    store = _make_store(tmp_path)
    event = {
        "op": "register_partition",
        "rel_path": "panel/20260106/core.parquet",
        "sha256_16": "new_fp",
        "zone": "hot",
        "row_domain": "label_filtered",
    }
    store.wal_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺字段"):
        _ = PanelDataStore(tmp_path).manifest
    assert store.wal_path.exists()


def test_failed_wal_open_cannot_be_bypassed_by_same_instance(tmp_path):
    _make_store(tmp_path)
    main_path = tmp_path / "features" / "manifest.json"
    main_bytes = main_path.read_bytes()
    wal = tmp_path / "features" / "manifest_wal.jsonl"
    wal.write_text('{"op":"future_event"}\n', encoding="utf-8")
    wal_bytes = wal.read_bytes()
    store = PanelDataStore(tmp_path)
    for _ in range(2):
        with pytest.raises(ValueError, match="未知 WAL 事件类型"):
            _ = store.manifest
        assert store._manifest is None

    labels = pd.DataFrame(
        {
            "trade_date": ["20260105"],
            "ts_code": ["600000.SH"],
            "label_value": [0.01],
            "maturity_status": ["forming"],
        }
    )
    with pytest.raises(ValueError, match="未知 WAL 事件类型"):
        store.append_labels("y_ret_20", labels)
    with pytest.raises(ValueError, match="未知 WAL 事件类型"):
        bootstrap_manifest(store)
    with pytest.raises(ValueError, match="未知 WAL 事件类型"):
        with store.defer_manifest_save():
            pass
    assert wal.read_bytes() == wal_bytes
    assert main_path.read_bytes() == main_bytes
    assert not (tmp_path / "labels" / "y_ret_20" / "20260105.parquet").exists()


@pytest.mark.parametrize("broken_index", [0, 1, 2])
def test_corrupt_wal_json_any_position_aborts_whole_batch(tmp_path, broken_index):
    _make_store(tmp_path)
    main_path = tmp_path / "features" / "manifest.json"
    main_bytes = main_path.read_bytes()
    events = []
    for date, fp in (("20260106", "fp-1"), ("20260107", "fp-2")):
        events.append(
            json.dumps(
                {
                    "op": "register_partition",
                    "rel_path": f"panel/{date}/core.parquet",
                    "sha256_16": fp,
                    "rows": 2,
                    "zone": "hot",
                    "row_domain": "label_filtered",
                }
            )
        )
    events.insert(broken_index, '{"op":"register_partition","rel_path":"truncated')
    wal = tmp_path / "features" / "manifest_wal.jsonl"
    wal.write_text("\n".join(events) + "\n", encoding="utf-8")
    wal_bytes = wal.read_bytes()
    with pytest.raises(ValueError, match="JSON 损坏"):
        _ = PanelDataStore(tmp_path).manifest
    assert wal.read_bytes() == wal_bytes
    assert main_path.read_bytes() == main_bytes
    on_disk = Manifest(main_path)
    assert on_disk.partition_fingerprint("panel/20260106/core.parquet") is None
    assert on_disk.partition_fingerprint("panel/20260107/core.parquet") is None


def test_defer_excludes_other_instance_until_commit_then_allows_resume(tmp_path):
    store_a = _make_store(tmp_path)
    store_b = PanelDataStore(tmp_path)
    _ = store_b.manifest
    with store_a.defer_manifest_save():
        store_a.append_features(TradeDate.from_str("20260105"), "core", _feature_df("20260105"))
        with pytest.raises(RuntimeError, match="写锁已被持有"):
            store_b.append_features(TradeDate.from_str("20260106"), "core", _feature_df("20260106"))
        assert not (tmp_path / "features" / "panel" / "20260106" / "core.parquet").exists()
    assert not store_a.wal_path.exists()

    store_b.append_features(TradeDate.from_str("20260106"), "core", _feature_df("20260106"))
    partitions = PanelDataStore(tmp_path).get_manifest()["partitions"]
    assert "panel/20260105/core.parquet" in partitions
    assert "panel/20260106/core.parquet" in partitions


def test_invalid_sealed_watermark_date_rejected(tmp_path):
    store = _make_store(tmp_path)
    store.wal_path.write_text(
        json.dumps(
            {"op": "set_label_sealed_through", "label_name": "y_ret_20", "date": "2026-01-05"}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="行域日期"):
        _ = PanelDataStore(tmp_path).manifest
    assert store.wal_path.exists()
    assert (
        Manifest(tmp_path / "features" / "manifest.json").label_sealed_through("y_ret_20") is None
    )


@pytest.mark.parametrize("wal_line", ['{"op":"future_event"}', '{"op":"truncated'])
def test_reload_replay_failure_cannot_be_bypassed_by_cached_instance(tmp_path, wal_line):
    store_a = _make_store(tmp_path)
    store_b = PanelDataStore(tmp_path)
    store_b.manifest.register_column("extra", "core", "probe")
    store_b.manifest.save()
    main_path = store_b.manifest.path
    main_bytes = main_path.read_bytes()
    store_b.wal_path.write_text(wal_line + "\n", encoding="utf-8")
    wal_bytes = store_b.wal_path.read_bytes()
    labels = pd.DataFrame(
        {
            "trade_date": ["20260105"],
            "ts_code": ["600000.SH"],
            "label_value": [0.01],
            "maturity_status": ["forming"],
        }
    )
    for _ in range(2):
        with pytest.raises(ValueError):
            store_a.append_labels("y_ret_20", labels)
        assert store_a._manifest is None
        assert store_b.wal_path.read_bytes() == wal_bytes
        assert main_path.read_bytes() == main_bytes
    assert not (tmp_path / "labels" / "y_ret_20" / "20260105.parquet").exists()


def test_failed_defer_wal_is_absorbed_by_old_cached_instance(tmp_path, monkeypatch):
    store_a = _make_store(tmp_path)
    store_b = PanelDataStore(tmp_path)
    _ = store_b.manifest

    def fail_save():
        raise OSError("模拟 defer 主提交失败")

    monkeypatch.setattr(store_b._wal, "save", fail_save)
    with pytest.raises(OSError, match="模拟 defer 主提交失败"):
        with store_b.defer_manifest_save():
            store_b.append_features(TradeDate.from_str("20260105"), "core", _feature_df("20260105"))
    assert store_b.wal_path.exists()
    assert not store_a.manifest.changed_on_disk()
    assert not store_b._lock._path.exists()

    store_a.append_features(TradeDate.from_str("20260106"), "core", _feature_df("20260106"))
    partitions = PanelDataStore(tmp_path).get_manifest()["partitions"]
    assert "panel/20260105/core.parquet" in partitions
    assert "panel/20260106/core.parquet" in partitions
    assert not store_a.wal_path.exists()
    assert (tmp_path / "features" / "panel" / "20260105" / "core.parquet").exists()
