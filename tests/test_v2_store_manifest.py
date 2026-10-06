# -*- coding: utf-8 -*-
"""v2 store manifest 测试（tmp_path 合成数据，不触碰真实 data/）。

覆盖：创建骨架 / 列登记与冲突 / 原子写（tmp 不残留）/ 快照深拷贝防篡改 /
unknown 列报错 / 分区登记与指纹查询 / labels 封存水位单调性 / 加载 schema 校验。
"""

import hashlib
import json

import pytest

from src.lazybull.v2.store.manifest import (
    MANIFEST_VERSION,
    Manifest,
    sha256_16_of_bytes,
    sha256_16_of_file,
)


@pytest.fixture
def manifest_path(tmp_path):
    return tmp_path / "features" / "manifest.json"


class TestManifestCreateAndSave:
    def test_create_skeleton_when_missing(self, manifest_path):
        m = Manifest(manifest_path)
        assert m.manifest_version == MANIFEST_VERSION
        assert not manifest_path.exists()  # 未 save 不落盘
        m.save()
        assert manifest_path.exists()

    def test_save_and_reload_roundtrip(self, manifest_path):
        m = Manifest(manifest_path)
        m.register_column("ret_1", "core", "clean/daily")
        m.register_partition("panel/20260105/core.parquet", "abc123", rows=10, zone="hot")
        m.save()
        m2 = Manifest(manifest_path)
        assert m2.column_group("ret_1") == "core"
        assert m2.partition_fingerprint("panel/20260105/core.parquet") == "abc123"

    def test_atomic_write_leaves_no_tmp(self, manifest_path):
        m = Manifest(manifest_path)
        m.save()
        leftovers = list(manifest_path.parent.glob("*.tmp"))
        assert leftovers == []

    def test_updated_at_refreshed_on_save(self, manifest_path):
        m = Manifest(manifest_path)
        before = m.snapshot()["updated_at"]
        m.save()
        assert m.snapshot()["updated_at"] >= before


class TestColumnRegister:
    def test_register_columns_batch(self, manifest_path):
        m = Manifest(manifest_path)
        m.register_columns(
            {
                "ret_1": {"group": "core", "source": "clean/daily"},
                "net_mf_amount": {
                    "group": "moneyflow",
                    "source": "raw/moneyflow",
                    "available_from": "20200102",
                },
            }
        )
        assert m.column_group("net_mf_amount") == "moneyflow"
        assert m.column_meta("net_mf_amount")["available_from"] == "20200102"

    def test_unknown_column_raises_key_error(self, manifest_path):
        m = Manifest(manifest_path)
        with pytest.raises(KeyError, match="未登记"):
            m.column_group("no_such_col")
        with pytest.raises(KeyError, match="未登记"):
            m.column_meta("no_such_col")
        assert not m.has_column("no_such_col")

    def test_invalid_group_rejected(self, manifest_path):
        m = Manifest(manifest_path)
        with pytest.raises(ValueError, match="非法列族"):
            m.register_column("x", "not_a_group", "src")

    def test_invalid_available_from_rejected(self, manifest_path):
        m = Manifest(manifest_path)
        with pytest.raises(ValueError, match="available_from"):
            m.register_column("x", "core", "src", available_from="2026-01-05")

    def test_reregister_conflict_group_rejected(self, manifest_path):
        m = Manifest(manifest_path)
        m.register_column("ret_1", "core", "clean/daily")
        with pytest.raises(ValueError, match="重复登记冲突"):
            m.register_column("ret_1", "moneyflow", "clean/daily")

    def test_reregister_identical_ok(self, manifest_path):
        m = Manifest(manifest_path)
        m.register_column("ret_1", "core", "clean/daily")
        m.register_column("ret_1", "core", "clean/daily", status="deprecated")
        assert m.column_meta("ret_1")["status"] == "deprecated"


class TestPartitionRegister:
    def test_register_and_query(self, manifest_path):
        m = Manifest(manifest_path)
        m.register_partition("panel_archive/2025-12/core.parquet", "fp16", rows=100, zone="archive")
        meta = m.partition_meta("panel_archive/2025-12/core.parquet")
        assert meta["rows"] == 100
        assert meta["zone"] == "archive"
        assert m.partition_fingerprint("panel_archive/2025-12/core.parquet") == "fp16"

    def test_unknown_partition_returns_none(self, manifest_path):
        m = Manifest(manifest_path)
        assert m.partition_fingerprint("panel/20990101/core.parquet") is None
        assert m.partition_meta("panel/20990101/core.parquet") is None

    def test_invalid_zone_rejected(self, manifest_path):
        m = Manifest(manifest_path)
        with pytest.raises(ValueError, match="zone"):
            m.register_partition("p", "fp", rows=1, zone="warm")


class TestSnapshotDeepCopy:
    def test_snapshot_mutation_does_not_leak(self, manifest_path):
        m = Manifest(manifest_path)
        m.register_column("ret_1", "core", "clean/daily")
        snap = m.snapshot()
        snap["columns"]["ret_1"]["group"] = "hacked"
        snap["partitions"]["evil"] = {}
        assert m.column_group("ret_1") == "core"
        assert m.partition_meta("evil") is None


class TestLabelsEntry:
    def test_label_partition_and_fingerprint(self, manifest_path):
        m = Manifest(manifest_path)
        m.register_label_partition("y_ret_20", "20260105", "fp", rows=5)
        assert m.label_partition_fingerprint("y_ret_20", "20260105") == "fp"
        assert m.label_partition_fingerprint("y_ret_20", "20260106") is None

    def test_sealed_through_monotonic(self, manifest_path):
        m = Manifest(manifest_path)
        assert m.label_sealed_through("y_ret_20") is None
        m.set_label_sealed_through("y_ret_20", "20260105")
        m.set_label_sealed_through("y_ret_20", "20260101")  # 只进不退
        assert m.label_sealed_through("y_ret_20") == "20260105"
        m.set_label_sealed_through("y_ret_20", "20260110")
        assert m.label_sealed_through("y_ret_20") == "20260110"


class TestLoadValidation:
    def test_missing_top_keys_rejected(self, manifest_path):
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({"manifest_version": "1"}), encoding="utf-8")
        with pytest.raises(ValueError, match="缺顶层键"):
            Manifest(manifest_path)

    def test_unsupported_version_rejected(self, manifest_path):
        manifest_path.parent.mkdir(parents=True)
        data = {
            "manifest_version": "2",
            "updated_at": "",
            "columns": {},
            "partitions": {},
            "dependencies": {},
            "labels": {},
        }
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest_version"):
            Manifest(manifest_path)

    def test_broken_json_rejected(self, manifest_path):
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="解析失败"):
            Manifest(manifest_path)


class TestFingerprint:
    def test_sha256_16_of_bytes(self):
        assert sha256_16_of_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()[:16]

    def test_sha256_16_of_file(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"lazybull")
        assert sha256_16_of_file(p) == hashlib.sha256(b"lazybull").hexdigest()[:16]


class TestContentFingerprint:
    """manifest 内容指纹（R3-10）：内容演进变化、updated_at 剔除、同内容稳定。"""

    def test_stable_for_same_content(self, manifest_path):
        m = Manifest(manifest_path)
        m.register_column("ret_1", "core", "clean/daily")
        assert m.content_fingerprint() == m.content_fingerprint()

    def test_changes_on_register_column_and_partition(self, manifest_path):
        m = Manifest(manifest_path)
        fp0 = m.content_fingerprint()
        m.register_column("ret_1", "core", "clean/daily")
        fp1 = m.content_fingerprint()
        assert fp1 != fp0
        m.register_partition("panel/20260105/core.parquet", "abc123", rows=10, zone="hot")
        assert m.content_fingerprint() != fp1

    def test_updated_at_excluded(self, manifest_path):
        m = Manifest(manifest_path)
        m.register_column("ret_1", "core", "clean/daily")
        fp_before = m.content_fingerprint()
        m.save()  # updated_at 落盘刷新不影响内容指纹
        assert m.content_fingerprint() == fp_before


class TestDirtyAndReload:
    """dirty 标记生命周期 + 盘上变化检测（WAL/sync 的判定基元）。"""

    def test_dirty_set_on_register_cleared_on_save(self, manifest_path):
        m = Manifest(manifest_path)
        assert not m.dirty
        m.register_column("ret_1", "core", "src")
        assert m.dirty
        m.save()
        assert not m.dirty

    def test_changed_on_disk_and_reload_if_changed(self, manifest_path):
        m = Manifest(manifest_path)
        m.save()
        assert not m.changed_on_disk()
        m2 = Manifest(manifest_path)  # 另一实例登记并落盘
        m2.register_column("x", "core", "s")
        m2.save()
        assert m.changed_on_disk()
        assert m.reload_if_changed() is True
        assert m.has_column("x")
        assert m.reload_if_changed() is False  # 无变化不重读


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
