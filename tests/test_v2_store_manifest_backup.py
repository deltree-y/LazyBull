# -*- coding: utf-8 -*-
"""P3 pre T2 manifest 备份、刷新审计、WAL 与显式恢复故障注入测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "v2_p1"))

import recover_manifest as recovery  # noqa: E402

from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.manifest import Manifest  # noqa: E402


def _new_manifest(path: Path) -> Manifest:
    return Manifest(path, allow_create=True)


def _prepare_record() -> dict:
    return {
        "record_type": "prepare",
        "event_key": "event-1",
        "trade_date": "20260703",
        "row_domain": "structural",
        "input_digest": "input-1",
        "producer_digest": "producer-1",
        "target_status": "sealed",
    }


_VALID_OUTPUT_FINGERPRINTS = {
    "y_ret_5": "0123456789abcdef",
    "y_ret_10": "1111111111111111",
    "y_ret_20": "abcdef0123456789",
}


def _completion_record() -> dict:
    record = _prepare_record()
    del record["record_type"]
    record["output_fingerprints"] = dict(_VALID_OUTPUT_FINGERPRINTS)
    return record


def _feature_df(values=(0.01, -0.02), date="20260105") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["600000.SH", "000001.SZ"],
            "ret_1": list(values),
        }
    )


def _store_with_partition(tmp_path: Path) -> PanelDataStore:
    store = PanelDataStore(tmp_path)
    store.manifest.register_column("ret_1", "core", "test")
    store.manifest.save()
    store.append_features(TradeDate.from_str("20260105"), "core", _feature_df())
    store.manifest.save()  # 让 bak 与当前分区文件同代
    return store


def test_first_and_second_save_create_valid_previous_generation_backup(tmp_path):
    path = tmp_path / "features" / "manifest.json"
    manifest = _new_manifest(path)
    manifest.register_column("ret_1", "core", "test")
    manifest.save()
    assert manifest.backup_path.exists()
    assert Manifest(manifest.backup_path).content_fingerprint() == manifest.content_fingerprint()
    assert list(path.parent.glob("*.tmp")) == []

    manifest.register_column("ret_5", "core", "test")
    manifest.save()
    backup = Manifest(manifest.backup_path)
    assert backup.has_column("ret_1") and not backup.has_column("ret_5")
    assert Manifest(path).has_column("ret_5")


def test_backup_failure_aborts_main_commit(tmp_path, monkeypatch):
    path = tmp_path / "features" / "manifest.json"
    manifest = _new_manifest(path)
    manifest.save()
    original_main = path.read_bytes()
    original_bak = manifest.backup_path.read_bytes()
    manifest.register_column("ret_1", "core", "test")

    def fail_backup(self, payload: bytes):
        raise RuntimeError("模拟备份失败")

    monkeypatch.setattr(Manifest, "_write_backup", fail_backup)
    with pytest.raises(RuntimeError, match="模拟备份失败"):
        manifest.save()
    assert path.read_bytes() == original_main
    assert manifest.backup_path.read_bytes() == original_bak
    assert not Manifest(path).has_column("ret_1")


def test_backup_replace_failure_cleans_bak_tmp(tmp_path, monkeypatch):
    path = tmp_path / "features" / "manifest.json"
    manifest = _new_manifest(path)
    manifest.save()
    original_main = path.read_bytes()
    manifest.register_column("ret_1", "core", "test")

    def fail_backup_replace(source, target, *, action):
        raise RuntimeError("模拟备份替换失败")

    monkeypatch.setattr(Manifest, "_replace_with_retry", staticmethod(fail_backup_replace))
    with pytest.raises(RuntimeError, match="模拟备份替换失败"):
        manifest.save()
    assert path.read_bytes() == original_main
    assert not manifest.backup_path.with_suffix(".json.bak.tmp").exists()


def test_main_replace_failure_keeps_original_exception_chain(tmp_path, monkeypatch):
    path = tmp_path / "features" / "manifest.json"
    manifest = _new_manifest(path)
    manifest.save()
    original_main = path.read_bytes()
    manifest.register_column("ret_1", "core", "test")

    import src.lazybull.v2.store.manifest as manifest_module

    real_replace = manifest_module.os.replace

    def deny_main(source, target):
        if Path(target) == path:
            raise PermissionError("模拟主文件拒绝")
        return real_replace(source, target)

    monkeypatch.setattr(manifest_module.os, "replace", deny_main)
    monkeypatch.setattr(manifest_module.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="manifest 落盘") as exc_info:
        manifest.save()
    assert isinstance(exc_info.value.__cause__, PermissionError)
    assert path.read_bytes() == original_main


def test_corrupt_current_main_does_not_overwrite_valid_backup(tmp_path):
    path = tmp_path / "features" / "manifest.json"
    manifest = _new_manifest(path)
    manifest.save()
    good_bak = manifest.backup_path.read_bytes()
    path.write_text("{broken", encoding="utf-8")
    manifest.register_column("ret_1", "core", "test")
    with pytest.raises(ValueError, match="解析失败"):
        manifest.save()
    assert manifest.backup_path.read_bytes() == good_bak


def test_missing_manifest_requires_explicit_create_or_recovery(tmp_path):
    path = tmp_path / "features" / "manifest.json"
    with pytest.raises(ValueError, match="recover_manifest"):
        Manifest(path)
    manifest = _new_manifest(path)
    manifest.save()
    path.unlink()
    with pytest.raises(ValueError, match="禁止从空骨架"):
        manifest.reload()

    store_root = tmp_path / "store"
    (store_root / "labels" / "y_ret_20").mkdir(parents=True)
    with pytest.raises(ValueError, match="recover_manifest"):
        _ = PanelDataStore(store_root).manifest


@pytest.mark.parametrize("tmp_name", ["manifest.json.tmp", "manifest.json.bak.tmp"])
def test_orphan_manifest_tmp_prevents_empty_skeleton_initialization(tmp_path, tmp_name):
    features = tmp_path / "store" / "features"
    features.mkdir(parents=True)
    (features / tmp_name).write_text("orphan", encoding="utf-8")
    with pytest.raises(ValueError, match="recover_manifest"):
        _ = PanelDataStore(tmp_path / "store").manifest


def test_main_good_backup_bad_is_repaired_by_next_normal_save(tmp_path):
    path = tmp_path / "features" / "manifest.json"
    manifest = _new_manifest(path)
    manifest.save()
    manifest.backup_path.write_text("{bad bak", encoding="utf-8")
    manifest.register_column("ret_1", "core", "test")
    manifest.save()
    backup = Manifest(manifest.backup_path)
    assert backup.manifest_version == "1"


def test_save_rejects_old_writer_dropped_row_domain(tmp_path):
    path = tmp_path / "features" / "manifest.json"
    manifest = _new_manifest(path)
    manifest.register_partition("panel/20260105/core.parquet", "fp", rows=2)
    manifest.save()
    data = manifest.snapshot()
    del data["partitions"]["panel/20260105/core.parquet"]["row_domain"]
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = Manifest(path)
    loaded.register_column("ret_1", "core", "test")
    with pytest.raises(ValueError, match="缺失或非法"):
        loaded.save()


def test_refresh_audits_are_append_only_idempotent_and_separate_from_repairs(tmp_path):
    manifest = _new_manifest(tmp_path / "features" / "manifest.json")
    prepare = _prepare_record()
    assert manifest.append_label_refresh_intent(prepare) is True
    assert (
        manifest.append_label_refresh_intent(
            {**prepare, "recorded_at": "另一个时间", "run_id": "另一个 run"}
        )
        is False
    )
    with pytest.raises(ValueError, match="审计身份冲突"):
        manifest.append_label_refresh_intent({**prepare, "input_digest": "changed"})

    orphan = {
        "record_type": "orphan",
        "event_key": "event-1",
        "related_event_key": "event-2",
        "superseded_by": "event-2",
        "reason": "输入身份变化",
    }
    assert manifest.append_label_refresh_intent(orphan) is True
    intents = manifest.label_refresh_intents()
    assert len(intents) == 2 and intents[0]["record_type"] == "prepare"

    completion = _completion_record()
    assert manifest.append_label_refresh_completion(completion) is True
    assert manifest.append_label_refresh_completion({**completion, "recovered": True}) is False
    with pytest.raises(ValueError, match="审计身份冲突"):
        manifest.append_label_refresh_completion({**completion, "target_status": "forming"})
    assert len(manifest.label_refreshes()) == 1
    assert manifest.repairs() == []


def test_refresh_audit_wal_replay_exactly_once(tmp_path, monkeypatch):
    store = PanelDataStore(tmp_path)
    store.manifest.save()
    prepare = _prepare_record()
    completion = _completion_record()

    def crash_save(self):
        raise RuntimeError("模拟崩溃")

    monkeypatch.setattr(Manifest, "save", crash_save)
    store._wal.append_label_refresh_intent(prepare)
    store._wal.append_label_refresh_completion(completion)
    with pytest.raises(RuntimeError, match="模拟崩溃"):
        store._wal.maybe_save()
    monkeypatch.undo()
    assert store.wal_path.exists()
    on_disk = Manifest(tmp_path / "features" / "manifest.json")
    assert on_disk.label_refresh_intents() == [] and on_disk.label_refreshes() == []

    replayed = PanelDataStore(tmp_path)
    assert len(replayed.manifest.label_refresh_intents()) == 1
    assert len(replayed.manifest.label_refreshes()) == 1
    assert not store.wal_path.exists()
    reopened = PanelDataStore(tmp_path)
    assert len(reopened.manifest.label_refresh_intents()) == 1
    assert len(reopened.manifest.label_refreshes()) == 1


def test_invalid_prepare_domain_rejected(tmp_path):
    manifest = _new_manifest(tmp_path / "features" / "manifest.json")
    with pytest.raises(ValueError, match="行域与日期不一致"):
        manifest.append_label_refresh_intent({**_prepare_record(), "row_domain": "label_filtered"})


@pytest.mark.parametrize(
    "bad_outputs",
    [
        pytest.param("__missing__", id="missing-whole-mapping"),
        pytest.param([], id="not-a-mapping"),
        pytest.param(None, id="null-mapping"),
        pytest.param({}, id="empty-mapping"),
        pytest.param(
            {
                name: fingerprint
                for name, fingerprint in _VALID_OUTPUT_FINGERPRINTS.items()
                if name != "y_ret_10"
            },
            id="missing-one-table",
        ),
        pytest.param({**_VALID_OUTPUT_FINGERPRINTS, "other": "0" * 16}, id="extra-table"),
        pytest.param({**_VALID_OUTPUT_FINGERPRINTS, "y_ret_5": ""}, id="empty-value"),
        pytest.param({**_VALID_OUTPUT_FINGERPRINTS, "y_ret_5": "0" * 15}, id="too-short"),
        pytest.param({**_VALID_OUTPUT_FINGERPRINTS, "y_ret_5": "0" * 17}, id="too-long"),
        pytest.param(
            {**_VALID_OUTPUT_FINGERPRINTS, "y_ret_5": "ABCDEF0123456789"},
            id="uppercase-hex",
        ),
        pytest.param({**_VALID_OUTPUT_FINGERPRINTS, "y_ret_5": "g" * 16}, id="non-hex"),
        pytest.param({**_VALID_OUTPUT_FINGERPRINTS, "y_ret_5": 0}, id="non-string"),
    ],
)
def test_completion_output_fingerprints_rejected_before_wal(tmp_path, bad_outputs):
    store = PanelDataStore(tmp_path)
    store.manifest.save()
    record = _completion_record()
    if bad_outputs == "__missing__":
        del record["output_fingerprints"]
    else:
        record["output_fingerprints"] = bad_outputs
    event = {"op": "append_label_refresh_completion", "record": record}
    wal_before = (
        json.dumps(
            {
                "op": "set_label_sealed_through",
                "label_name": "y_ret_20",
                "date": "20260703",
            }
        )
        + "\n"
    ).encode("utf-8")
    store.wal_path.write_bytes(wal_before)

    with pytest.raises((TypeError, ValueError), match="output_fingerprints"):
        store._wal._validate_event(event)
    with pytest.raises((TypeError, ValueError), match="output_fingerprints"):
        store._wal.append_label_refresh_completion(record)

    assert store.wal_path.read_bytes() == wal_before
    assert store.manifest.label_refreshes() == []


def test_recovery_dry_run_then_execute_restores_matching_backup(tmp_path):
    data_root = tmp_path / "data"
    store = _store_with_partition(data_root)
    expected_fp = store.manifest.partition_fingerprint("panel/20260105/core.parquet")
    manifest_path = store.features_dir / "manifest.json"
    manifest_path.write_text("{broken main", encoding="utf-8")
    backup_root = tmp_path / "backup" / "run-1"

    plan = recovery.recover_manifest(data_root, backup_root)
    assert plan["status"] == "recovery_planned" and plan["execute_allowed"] is True
    assert not backup_root.exists()
    report = recovery.recover_manifest(data_root, backup_root, execute=True)
    assert report["status"] == "recovered"
    assert (backup_root / "quarantine" / "manifest.json").read_text(encoding="utf-8") == (
        "{broken main"
    )
    restored = PanelDataStore(data_root)
    assert restored.manifest.partition_fingerprint("panel/20260105/core.parquet") == expected_fp
    assert (backup_root / "recovery_report.json").exists()


def test_recovery_stale_backup_rejected_without_writes(tmp_path):
    data_root = tmp_path / "data"
    store = _store_with_partition(data_root)
    store.features_dir.joinpath("manifest.json").write_text("{broken", encoding="utf-8")
    final = data_root / "features" / "panel" / "20260105" / "core.parquet"
    _feature_df(values=(9.9, 8.8)).to_parquet(final, index=False)
    backup_root = tmp_path / "backup" / "stale"
    plan = recovery.recover_manifest(data_root, backup_root)
    assert plan["status"] == "manual_required_stale_backup"
    with pytest.raises(RuntimeError, match="恢复预检未通过"):
        recovery.recover_manifest(data_root, backup_root, execute=True)
    assert not backup_root.exists()


def test_recovery_pending_wal_rejected(tmp_path):
    data_root = tmp_path / "data"
    store = _store_with_partition(data_root)
    store.features_dir.joinpath("manifest.json").write_text("{broken", encoding="utf-8")
    store.wal_path.write_text('{"op":"future_event"}\n', encoding="utf-8")
    plan = recovery.recover_manifest(data_root, tmp_path / "backup" / "wal")
    assert plan["status"] == "manual_required_wal"
    assert store.wal_path.exists()


def test_recovery_double_corruption_has_no_candidate(tmp_path):
    data_root = tmp_path / "data"
    store = _store_with_partition(data_root)
    store.features_dir.joinpath("manifest.json").write_text("{broken", encoding="utf-8")
    store.manifest.backup_path.write_text("{also broken", encoding="utf-8")
    plan = recovery.recover_manifest(data_root, tmp_path / "backup" / "double")
    assert plan["status"] == "no_valid_backup"
    assert plan["execute_allowed"] is False


def test_recovery_valid_main_with_inconsistent_files_never_falls_back(tmp_path):
    data_root = tmp_path / "data"
    store = PanelDataStore(data_root)
    store.manifest.register_column("ret_1", "core", "test")
    store.manifest.save()
    store.append_features(TradeDate.from_str("20260105"), "core", _feature_df())
    store.manifest.save()
    store.append_features(TradeDate.from_str("20260106"), "core", _feature_df(date="20260106"))
    manifest_path = store.features_dir / "manifest.json"
    original_main = manifest_path.read_bytes()
    (data_root / "features" / "panel" / "20260106" / "core.parquet").unlink()
    backup_root = tmp_path / "backup" / "inconsistent-main"

    plan = recovery.recover_manifest(data_root, backup_root)
    assert plan["status"] == "manual_required_inconsistent_main"
    assert plan["main"]["missing_files"] == ["panel/20260106/core.parquet"]
    assert plan["bak"]["files_consistent"] is True
    with pytest.raises(RuntimeError, match="恢复预检未通过"):
        recovery.recover_manifest(data_root, backup_root, execute=True)
    assert manifest_path.read_bytes() == original_main
    assert not backup_root.exists()


@pytest.mark.parametrize("kind", ["hot", "archive", "labels"])
def test_recovery_untracked_final_files_rejected(tmp_path, kind):
    data_root = tmp_path / "data"
    store = PanelDataStore(data_root)
    store.manifest.register_column("ret_1", "core", "test")
    store.manifest.save()
    if kind == "hot":
        store.append_features(TradeDate.from_str("20260105"), "core", _feature_df())
        final = data_root / "features" / "panel" / "20260105" / "core.parquet"
    elif kind == "archive":
        store.append_archive_features("2025-05", "core", _feature_df(date="20250506"))
        final = data_root / "features" / "panel_archive" / "2025-05" / "core.parquet"
    else:
        labels = pd.DataFrame(
            {
                "trade_date": ["20260105", "20260105"],
                "ts_code": ["600000.SH", "000001.SZ"],
                "label_value": [0.01, 0.02],
                "maturity_status": ["forming", "forming"],
            }
        )
        store.append_labels("y_ret_20", labels)
        final = data_root / "labels" / "y_ret_20" / "20260105.parquet"
    manifest_path = store.features_dir / "manifest.json"
    manifest_path.write_text("{broken", encoding="utf-8")
    backup_root = tmp_path / "backup" / f"untracked-{kind}"

    plan = recovery.recover_manifest(data_root, backup_root)
    assert plan["status"] == "manual_required_untracked_files"
    assert len(plan["bak"]["untracked_files"]) == 1
    with pytest.raises(RuntimeError, match="恢复预检未通过"):
        recovery.recover_manifest(data_root, backup_root, execute=True)
    assert final.exists()
    assert not backup_root.exists()


def test_recovery_lock_conflict_happens_before_any_mutation(tmp_path):
    data_root = tmp_path / "data"
    store = _store_with_partition(data_root)
    manifest_path = store.features_dir / "manifest.json"
    manifest_path.write_text("{broken under lock", encoding="utf-8")
    locked_bytes = manifest_path.read_bytes()
    backup_root = tmp_path / "backup" / "locked"
    with store.write_lock():
        with pytest.raises(RuntimeError, match="写锁已被持有"):
            recovery.recover_manifest(data_root, backup_root, execute=True)
        assert manifest_path.read_bytes() == locked_bytes
        assert not backup_root.exists()
