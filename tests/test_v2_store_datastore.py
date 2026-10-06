# -*- coding: utf-8 -*-
"""v2 PanelDataStore 测试（tmp_path 合成数据，不触碰真实 data/）。

覆盖：两阶段提交（成功 / 冲突 raise / 指纹一致放行 / .tmp 清理）、写锁互斥与 stale 抢占、
孤儿 GC、load_features 跨热/冷区合并 + 三类 ValueError + universe 过滤、
append_labels forming 可重写 / sealed 拒改写 / 指纹一致 no-op、append_ledger_entry 序列化与校验。
"""

import json
import types
from pathlib import Path

import pandas as pd
import psutil
import pytest

from src.lazybull.v2.common.types import (
    DataState,
    FeatureQuery,
    LabelQuery,
    LedgerEntry,
    TradeDate,
    TSCode,
)
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.manifest import Manifest

_D1 = TradeDate.from_str("20260105")
_D2 = TradeDate.from_str("20260106")
_CODES = ["600000.SH", "000001.SZ"]


def _make_store(tmp_path, extra_columns=None) -> PanelDataStore:
    """建 store 并登记测试列：ret_1(core)、net_mf_amount(moneyflow, available_from=20200102)。"""
    store = PanelDataStore(tmp_path)
    columns = {
        "ret_1": {"group": "core", "source": "clean/daily"},
        "net_mf_amount": {
            "group": "moneyflow",
            "source": "raw/moneyflow",
            "available_from": "20200102",
        },
    }
    if extra_columns:
        columns.update(extra_columns)
    store.manifest.register_columns(columns)
    store.manifest.save()
    return store


def _feature_df(date: str, group_col: str = "ret_1", values=(0.01, -0.02)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date] * len(_CODES),
            "ts_code": list(_CODES),
            group_col: list(values),
        }
    )


def _label_df(date: str, values=(0.03, -0.01), maturity: str = "forming") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date] * len(_CODES),
            "ts_code": list(_CODES),
            "label_value": list(values),
            "maturity_status": [maturity] * len(_CODES),
        }
    )


def _ledger_entry(hypothesis_id: str = "H-test-1", kind: str = "prereg") -> LedgerEntry:
    return LedgerEntry(
        hypothesis_id=hypothesis_id,
        kind=kind,
        payload={"title": "测试假设", "vehicle": "rule"},
        data_state=DataState(
            raw_partitions={"daily": "20260731"},
            features_partition_max=TradeDate.from_str("20260702"),
            config_digest="cfg",
            code_digest="abc1234",
        ),
        registered_at="2026-10-03T10:00:00",
    )


class TestAppendFeaturesTwoPhase:
    def test_success_path_registers_partition(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        final = tmp_path / "features" / "panel" / "20260105" / "core.parquet"
        assert final.exists()
        meta = store.manifest.partition_meta("panel/20260105/core.parquet")
        assert meta["rows"] == 2
        assert meta["zone"] == "hot"
        assert list(final.parent.glob("*.tmp")) == []  # 提交后无临时文件残留

    def test_identical_rewrite_passes(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        store.append_features(_D1, "core", _feature_df("20260105"))  # 指纹一致放行
        loaded = pd.read_parquet(tmp_path / "features" / "panel" / "20260105" / "core.parquet")
        assert len(loaded) == 2

    def test_conflicting_rewrite_raises_and_cleans_tmp(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        with pytest.raises(RuntimeError, match="指纹冲突"):
            store.append_features(_D1, "core", _feature_df("20260105", values=(9.9, 9.9)))
        part_dir = tmp_path / "features" / "panel" / "20260105"
        assert list(part_dir.glob("*.tmp")) == []
        loaded = pd.read_parquet(part_dir / "core.parquet")  # 原分区内容未被污染
        assert loaded["ret_1"].tolist() == [0.01, -0.02]

    def test_unregistered_column_raises(self, tmp_path):
        store = _make_store(tmp_path)
        df = _feature_df("20260105").assign(mystery=1.0)
        with pytest.raises(RuntimeError, match="未登记"):
            store.append_features(_D1, "core", df)

    def test_group_mismatch_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(RuntimeError, match="登记在族"):
            store.append_features(_D1, "moneyflow", _feature_df("20260105"))

    def test_multi_date_df_raises(self, tmp_path):
        store = _make_store(tmp_path)
        df = pd.concat([_feature_df("20260105"), _feature_df("20260106")])
        with pytest.raises(RuntimeError, match="单日"):
            store.append_features(_D1, "core", df)

    def test_missing_key_column_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(RuntimeError, match="缺键列"):
            store.append_features(_D1, "core", _feature_df("20260105").drop(columns=["ts_code"]))

    def test_sealed_archive_month_raises(self, tmp_path):
        store = _make_store(tmp_path)
        archive_dir = tmp_path / "features" / "panel_archive" / "2026-01"
        archive_dir.mkdir(parents=True)
        _feature_df("20260105").to_parquet(archive_dir / "core.parquet", index=False)
        with pytest.raises(RuntimeError, match="已封存"):
            store.append_features(_D1, "core", _feature_df("20260105"))

    def test_orphan_tmp_gc(self, tmp_path):
        store = _make_store(tmp_path)
        part_dir = tmp_path / "features" / "panel" / "20260105"
        part_dir.mkdir(parents=True)
        orphan = part_dir / "orphan.parquet.tmp"
        orphan.write_bytes(b"broken")
        store.append_features(_D1, "core", _feature_df("20260105"))
        assert not orphan.exists()


class TestWriteLock:
    def test_second_instance_blocked(self, tmp_path):
        store1 = _make_store(tmp_path)
        store2 = PanelDataStore(tmp_path)
        with store1.write_lock():
            with pytest.raises(RuntimeError, match="写锁已被持有"):
                store2.append_features(_D1, "core", _feature_df("20260105"))
        store2.append_features(_D1, "core", _feature_df("20260105"))  # 锁释放后可写

    def test_reentrant_same_instance(self, tmp_path):
        store = _make_store(tmp_path)
        with store.write_lock():
            with store.write_lock():  # 同实例可重入
                store.append_features(_D1, "core", _feature_df("20260105"))
        assert not (tmp_path / "features" / ".write.lock").exists()

    def test_stale_lock_preempted(self, tmp_path):
        store = _make_store(tmp_path)
        dead_pid = 4194303
        if psutil.pid_exists(dead_pid):  # 防御：极端巧合下换一个
            dead_pid += 1
        lock_path = tmp_path / "features" / ".write.lock"
        lock_path.write_text(
            json.dumps({"pid": dead_pid, "host": "ghost", "acquired_at": "2026-01-01T00:00:00"}),
            encoding="utf-8",
        )
        store.append_features(_D1, "core", _feature_df("20260105"))  # 抢占 stale 锁后正常提交
        assert not lock_path.exists()


class TestLoadFeatures:
    def _build_hot_cold(self, tmp_path) -> PanelDataStore:
        """冷区 2025-12（core+moneyflow）+ 热区 20260105（双族）/20260106（仅 core）。"""
        store = _make_store(tmp_path)
        for group, col in (("core", "ret_1"), ("moneyflow", "net_mf_amount")):
            archive_dir = tmp_path / "features" / "panel_archive" / "2025-12"
            archive_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {
                    "trade_date": ["20251230", "20251231", "20251201"],  # 区间外日期验证裁剪
                    "ts_code": ["600000.SH"] * 3,
                    col: [1.0, 2.0, 999.0],
                }
            )
            df.to_parquet(archive_dir / f"{group}.parquet", index=False)
        store.append_features(_D1, "core", _feature_df("20260105"))
        store.append_features(
            _D1, "moneyflow", _feature_df("20260105", "net_mf_amount", (100.0, 200.0))
        )
        store.append_features(_D2, "core", _feature_df("20260106", values=(0.05, 0.06)))
        return store

    def test_hot_cold_merge_and_crop(self, tmp_path):
        store = self._build_hot_cold(tmp_path)
        frame = store.load_features(
            FeatureQuery(
                columns=["ret_1", "net_mf_amount"],
                start_date=TradeDate.from_str("20251230"),
                end_date=TradeDate.from_str("20260106"),
            )
        )
        df = frame.df
        assert list(df.columns) == ["ret_1", "net_mf_amount"]
        assert df.index.names == ["trade_date", "ts_code"]
        assert "20251201" not in df.index.get_level_values("trade_date")  # 冷区按区间裁剪
        cold = df.loc[("20251231", "600000.SH")]
        assert cold["ret_1"] == 2.0 and cold["net_mf_amount"] == 2.0
        hot = df.loc[("20260105", "000001.SZ")]
        assert hot["ret_1"] == -0.02 and hot["net_mf_amount"] == 200.0
        missing = df.loc[("20260106", "600000.SH")]  # moneyflow 缺日整族 NaN
        assert missing["ret_1"] == 0.05 and pd.isna(missing["net_mf_amount"])
        assert frame.manifest_version == "1"
        assert frame.available_from["net_mf_amount"] == TradeDate.from_str("20200102")
        assert "ret_1" not in frame.available_from  # 未登记起点 ⇒ 恒可用（不进映射）

    def test_hot_file_in_archived_month_ignored(self, tmp_path):
        store = self._build_hot_cold(tmp_path)
        stale_dir = tmp_path / "features" / "panel" / "20251230"
        stale_dir.mkdir(parents=True)
        _feature_df("20251230", values=(777.0, 777.0)).to_parquet(
            stale_dir / "core.parquet", index=False
        )
        frame = store.load_features(
            FeatureQuery(
                columns=["ret_1"],
                start_date=TradeDate.from_str("20251230"),
                end_date=TradeDate.from_str("20251230"),
            )
        )
        assert frame.df.loc[("20251230", "600000.SH"), "ret_1"] == 1.0  # 冷区为准

    def test_unknown_column_raises_value_error(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="未登记"):
            store.load_features(FeatureQuery(columns=["no_such_col"], start_date=_D1, end_date=_D2))

    def test_available_from_violation_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="available_from 越界"):
            store.load_features(
                FeatureQuery(
                    columns=["net_mf_amount"],
                    start_date=TradeDate.from_str("20191231"),  # < 20200102
                    end_date=_D1,
                )
            )

    def test_universe_filter(self, tmp_path):
        store = self._build_hot_cold(tmp_path)
        frame = store.load_features(
            FeatureQuery(
                columns=["ret_1"],
                start_date=_D1,
                end_date=_D2,
                universe=[TSCode("600000.SH")],
            )
        )
        assert set(frame.df.index.get_level_values("ts_code")) == {"600000.SH"}

    def test_empty_result(self, tmp_path):
        store = _make_store(tmp_path)
        frame = store.load_features(
            FeatureQuery(
                columns=["ret_1"],
                start_date=TradeDate.from_str("20300101"),
                end_date=TradeDate.from_str("20300131"),
            )
        )
        assert frame.df.empty
        assert frame.df.index.names == ["trade_date", "ts_code"]


class TestLabels:
    def test_forming_rewrite_allowed(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_labels("y_ret_20", _label_df("20260105"))
        store.append_labels("y_ret_20", _label_df("20260105", values=(0.5, 0.6)))  # forming 可重写
        df = pd.read_parquet(tmp_path / "labels" / "y_ret_20" / "20260105.parquet")
        assert df["label_value"].tolist() == [0.5, 0.6]
        assert store.manifest.label_sealed_through("y_ret_20") is None

    def test_identical_rewrite_noop(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_labels("y_ret_20", _label_df("20260105"))
        store.append_labels("y_ret_20", _label_df("20260105"))  # 指纹一致 no-op，不报错

    def test_sealed_reject_rewrite(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_labels("y_ret_20", _label_df("20260105", maturity="sealed"))
        assert store.manifest.label_sealed_through("y_ret_20") == "20260105"
        with pytest.raises(RuntimeError, match="已封存"):
            store.append_labels(
                "y_ret_20", _label_df("20260105", values=(9.9, 9.9), maturity="sealed")
            )
        store.append_labels("y_ret_20", _label_df("20260105", maturity="sealed"))  # 一致 no-op

    def test_earlier_date_blocked_by_watermark(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_labels("y_ret_20", _label_df("20260105"))
        store.append_labels("y_ret_20", _label_df("20260106", maturity="sealed"))
        with pytest.raises(RuntimeError, match="已封存"):
            store.append_labels("y_ret_20", _label_df("20260105", values=(9.9, 9.9)))

    def test_missing_maturity_status_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="缺列"):
            store.append_labels("y_ret_20", _label_df("20260105").drop(columns=["maturity_status"]))

    def test_mixed_maturity_one_day_raises(self, tmp_path):
        store = _make_store(tmp_path)
        df = _label_df("20260105")
        df.loc[1, "maturity_status"] = "sealed"
        with pytest.raises(ValueError, match="单日成熟度不一致"):
            store.append_labels("y_ret_20", df)

    def test_load_labels_shape_and_universe(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_labels("y_ret_20", _label_df("20260105"))
        store.append_labels("y_ret_20", _label_df("20260106", maturity="sealed"))
        df = store.load_labels(
            LabelQuery(
                label_name="y_ret_20",
                start_date=_D1,
                end_date=_D2,
                universe=[TSCode("000001.SZ")],
            )
        )
        assert list(df.columns) == ["label_value", "maturity_status"]
        assert df.index.names == ["trade_date", "ts_code"]
        assert set(df.index.get_level_values("ts_code")) == {"000001.SZ"}
        assert df.loc[("20260106", "000001.SZ"), "maturity_status"] == "sealed"

    def test_load_labels_empty(self, tmp_path):
        store = _make_store(tmp_path)
        df = store.load_labels(LabelQuery(label_name="y_ret_20", start_date=_D1, end_date=_D2))
        assert df.empty
        assert list(df.columns) == ["label_value", "maturity_status"]


class TestLedgerAppend:
    def test_serialization_roundtrip(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_ledger_entry(_ledger_entry())
        with open(store.ledger_path, encoding="utf-8") as f:
            record = json.loads(f.readline())
        assert record["schema_version"] == 1
        assert record["hypothesis_id"] == "H-test-1"
        assert record["kind"] == "prereg"
        assert record["payload"]["title"] == "测试假设"
        assert record["data_state"]["features_partition_max"] == "20260702"
        assert record["data_state"]["raw_partitions"] == {"daily": "20260731"}
        assert record["registered_at"] == "2026-10-03T10:00:00"

    def test_bad_kind_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="kind 非法"):
            store.append_ledger_entry(_ledger_entry(kind="draft"))

    def test_empty_hypothesis_id_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="hypothesis_id 不能为空"):
            store.append_ledger_entry(_ledger_entry(hypothesis_id="  "))

    def test_duplicate_id_conflict(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_ledger_entry(_ledger_entry())
        with pytest.raises(ValueError, match="重复"):
            store.append_ledger_entry(_ledger_entry())
        store.append_ledger_entry(_ledger_entry(hypothesis_id="H-test-2", kind="conclusion"))
        with open(store.ledger_path, encoding="utf-8") as f:
            assert len([line for line in f if line.strip()]) == 2


class TestGetManifest:
    def test_mapping_proxy_and_deepcopy(self, tmp_path):
        store = _make_store(tmp_path)
        snapshot = store.get_manifest()
        assert isinstance(snapshot, types.MappingProxyType)
        with pytest.raises(TypeError):
            snapshot["columns"] = {}
        snapshot["columns"]["ret_1"]["group"] = "hacked"  # 嵌套篡改不渗回内部
        assert store.manifest.column_group("ret_1") == "core"


class TestWalCrashRecovery:
    """WAL 崩溃恢复（R2-6 回归）：崩溃丢登记 ⇒ 打开重放补登记 ⇒ 不可变保护恢复生效。"""

    @staticmethod
    def _crash_save(monkeypatch) -> None:
        """模拟 manifest 落盘时进程崩溃：save 抛异常 ⇒ WAL 保留、manifest 未更新。"""

        def _crash(self):
            raise RuntimeError("模拟崩溃")

        monkeypatch.setattr(Manifest, "save", _crash)

    def test_sealed_label_crash_replay(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        self._crash_save(monkeypatch)
        with pytest.raises(RuntimeError, match="模拟崩溃"):
            store.append_labels("y_ret_20", _label_df("20260105", maturity="sealed"))
        monkeypatch.undo()
        wal = tmp_path / "features" / "manifest_wal.jsonl"
        assert wal.exists() and wal.read_text(encoding="utf-8").strip()  # 事件已累积
        on_disk = Manifest(tmp_path / "features" / "manifest.json")
        assert on_disk.label_partition_fingerprint("y_ret_20", "20260105") is None  # 登记未落盘
        store2 = PanelDataStore(tmp_path)  # 打开即 WAL 重放补登记 + 压实
        assert store2.manifest.label_partition_fingerprint("y_ret_20", "20260105") is not None
        assert store2.manifest.label_sealed_through("y_ret_20") == "20260105"
        assert not wal.exists()  # 重放后压实
        with pytest.raises(RuntimeError, match="已封存"):  # 封存保护恢复生效
            store2.append_labels(
                "y_ret_20", _label_df("20260105", values=(9.9, 9.9), maturity="sealed")
            )
        store2.append_labels("y_ret_20", _label_df("20260105", maturity="sealed"))  # 同值 no-op

    def test_archive_crash_replay(self, tmp_path, monkeypatch):
        """冷区同构用例：崩溃丢冷区登记 ⇒ 重放补登记 ⇒ 原地修改仍被拒。"""
        store = _make_store(tmp_path)
        self._crash_save(monkeypatch)
        df = pd.DataFrame(
            {"trade_date": ["20250506"] * 2, "ts_code": _CODES, "ret_1": [0.01, -0.02]}
        )
        with pytest.raises(RuntimeError, match="模拟崩溃"):
            store.append_archive_features("2025-05", "core", df)
        monkeypatch.undo()
        on_disk = Manifest(tmp_path / "features" / "manifest.json")
        assert on_disk.partition_fingerprint("panel_archive/2025-05/core.parquet") is None
        store2 = PanelDataStore(tmp_path)
        assert store2.manifest.partition_fingerprint("panel_archive/2025-05/core.parquet")
        with pytest.raises(RuntimeError, match="禁止原地修改"):
            store2.append_archive_features("2025-05", "core", df.assign(ret_1=[9.9, 9.9]))
        store2.append_archive_features("2025-05", "core", df)  # 指纹一致 no-op

    def test_two_instances_sequential_writes(self, tmp_path):
        """A 落盘后 B（持旧快照）写：B 锁内 sync 重载，A/B 登记俱在。"""
        store_a = _make_store(tmp_path)
        store_b = PanelDataStore(tmp_path)
        _ = store_b.manifest  # B 触达 manifest（持旧快照）
        store_a.append_features(_D1, "core", _feature_df("20260105"))
        store_b.append_features(_D2, "core", _feature_df("20260106"))
        partitions = store_b.get_manifest()["partitions"]
        assert "panel/20260105/core.parquet" in partitions
        assert "panel/20260106/core.parquet" in partitions

    def test_dirty_instance_conflict_fail_closed(self, tmp_path):
        """B 持未落盘登记（脏）时 A 落盘 ⇒ B 写时 RuntimeError（并发冲突 fail-closed）。"""
        store_a = _make_store(tmp_path)
        store_b = PanelDataStore(tmp_path)
        store_b.manifest.register_column("dirty_col", "core", "test")  # 脏（未落盘）
        store_a.append_features(_D1, "core", _feature_df("20260105"))
        with pytest.raises(RuntimeError, match="并发冲突"):
            store_b.append_features(_D2, "core", _feature_df("20260106"))

    def test_defer_wal_accumulates_and_compacts(self, tmp_path):
        """defer 期内 WAL 累积多日事件，出口统一落盘 + 压实。"""
        store = _make_store(tmp_path)
        wal = tmp_path / "features" / "manifest_wal.jsonl"
        with store.defer_manifest_save():
            store.append_features(_D1, "core", _feature_df("20260105"))
            store.append_features(_D2, "core", _feature_df("20260106"))
            assert wal.exists()
            assert len(wal.read_text(encoding="utf-8").strip().splitlines()) == 2
        assert not wal.exists()  # 出口压实
        assert store.manifest.partition_fingerprint("panel/20260106/core.parquet") is not None


class TestWriteLockRelease:
    """写锁释放的 unlink 失败处理（有限退避重试 + 不掩盖锁内原异常）。"""

    @staticmethod
    def _patch_unlink(monkeypatch, lock_path: Path, fail_times: int) -> list[int]:
        calls: list[int] = []
        real_unlink = Path.unlink

        def flaky(self, *args, **kwargs):
            if self == lock_path and len(calls) < fail_times:
                calls.append(1)
                raise PermissionError("模拟瞬时占用")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", flaky)
        return calls

    def test_unlink_failure_does_not_mask_inner_exception(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        calls = self._patch_unlink(monkeypatch, tmp_path / "features" / ".write.lock", 99)
        with pytest.raises(ValueError, match="内部异常"):
            with store.write_lock():
                raise ValueError("内部异常")
        assert len(calls) == 3  # 重试 3 次后降级 warning，原异常不被掩盖

    def test_unlink_persistent_failure_raises(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        self._patch_unlink(monkeypatch, tmp_path / "features" / ".write.lock", 99)
        with pytest.raises(RuntimeError, match="写锁文件删除连续失败"):
            with store.write_lock():
                pass

    def test_unlink_transient_failure_recovers(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        calls = self._patch_unlink(monkeypatch, tmp_path / "features" / ".write.lock", 1)
        with store.write_lock():  # 第 2 次重试成功，不报错
            pass
        assert len(calls) == 1


class TestCoreAnchorLoad:
    """load_features 的 core 左表锚定（R3-02）：core 键域为左表，非 core 多出键被裁。"""

    def _anchored_store(self, tmp_path) -> PanelDataStore:
        """core 两日 A/B + moneyflow 仅 D1 的 A。"""
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        store.append_features(_D2, "core", _feature_df("20260106"))
        mf = pd.DataFrame(
            {"trade_date": ["20260105"], "ts_code": ["600000.SH"], "net_mf_amount": [100.0]}
        )
        store.append_features(_D1, "moneyflow", mf)
        return store

    def test_non_core_query_anchored_to_core_domain(self, tmp_path):
        store = self._anchored_store(tmp_path)
        frame = store.load_features(
            FeatureQuery(columns=["net_mf_amount"], start_date=_D1, end_date=_D2)
        )
        df = frame.df
        assert len(df) == 4  # 行集 = core 键域（D1/D2 × A/B）
        assert df.loc[("20260105", "600000.SH"), "net_mf_amount"] == 100.0
        assert pd.isna(df.loc[("20260105", "000001.SZ"), "net_mf_amount"])  # 缺键整族 NaN
        assert pd.isna(df.loc[("20260106", "600000.SH"), "net_mf_amount"])  # 缺日整族 NaN
        assert frame.manifest_fingerprint == store.manifest.content_fingerprint()
        assert frame.manifest_version == "1"  # schema 版本（与内容指纹区分）

    def test_keys_only_query_returns_core_domain(self, tmp_path):
        store = self._anchored_store(tmp_path)
        frame = store.load_features(
            FeatureQuery(columns=["trade_date", "ts_code"], start_date=_D1, end_date=_D2)
        )
        assert frame.df.shape == (4, 0)  # core 键域空列帧
        assert frame.df.index.names == ["trade_date", "ts_code"]

    def test_non_core_extra_keys_trimmed(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        mf = pd.DataFrame(
            {
                "trade_date": ["20260105"] * 3,
                "ts_code": [*_CODES, "300999.SZ"],  # 多出 core 键域的 C
                "net_mf_amount": [1.0, 2.0, 3.0],
            }
        )
        store.append_features(_D1, "moneyflow", mf)
        frame = store.load_features(
            FeatureQuery(columns=["net_mf_amount"], start_date=_D1, end_date=_D1)
        )
        assert set(frame.df.index.get_level_values("ts_code")) == set(_CODES)

    def test_core_missing_entirely_empty_result(self, tmp_path):
        """core 分区整体缺失（冷/热均无）⇒ 空结果（现状语义保持）。"""
        store = _make_store(tmp_path)
        mf = pd.DataFrame(
            {"trade_date": ["20260105"], "ts_code": ["600000.SH"], "net_mf_amount": [100.0]}
        )
        store.append_features(_D1, "moneyflow", mf)
        frame = store.load_features(
            FeatureQuery(columns=["net_mf_amount"], start_date=_D1, end_date=_D1)
        )
        assert frame.df.empty

    def test_duplicate_keys_in_partition_raise(self, tmp_path):
        """合并后键唯一断言（防御：盘上分区被旁路污染出重复键）。"""
        store = _make_store(tmp_path)
        part_dir = tmp_path / "features" / "panel" / "20260105"
        part_dir.mkdir(parents=True)
        pd.concat([_feature_df("20260105"), _feature_df("20260105")]).to_parquet(
            part_dir / "core.parquet", index=False
        )
        with pytest.raises(RuntimeError, match="不唯一"):
            store.load_features(FeatureQuery(columns=["ret_1"], start_date=_D1, end_date=_D1))

    def test_hot_partition_wrong_date_rows_filtered(self, tmp_path):
        """热区读路径日期过滤（防御：分区文件混入分区名外日期行）。"""
        store = _make_store(tmp_path)
        part_dir = tmp_path / "features" / "panel" / "20260105"
        part_dir.mkdir(parents=True)
        pd.concat([_feature_df("20260105"), _feature_df("20260106")]).to_parquet(
            part_dir / "core.parquet", index=False
        )
        frame = store.load_features(FeatureQuery(columns=["ret_1"], start_date=_D1, end_date=_D1))
        assert set(frame.df.index.get_level_values("trade_date")) == {"20260105"}


class TestLabelVariant:
    """load_labels 变体路由（R3-05）：raw/neu 取值、未知标签名报错。"""

    def _store_with_neu(self, tmp_path) -> PanelDataStore:
        store = _make_store(tmp_path)
        df = _label_df("20260105")
        df["neu_label_value"] = [0.005, -0.005]
        store.append_labels("y_ret_20", df)
        return store

    def test_raw_and_neu_variants(self, tmp_path):
        store = self._store_with_neu(tmp_path)
        raw = store.load_labels(LabelQuery(label_name="y_ret_20", start_date=_D1, end_date=_D1))
        neu = store.load_labels(
            LabelQuery(label_name="y_ret_20", start_date=_D1, end_date=_D1, variant="neu")
        )
        assert raw["label_value"].tolist() == [-0.01, 0.03]  # index 按 ts_code 排序
        assert neu["label_value"].tolist() == [-0.005, 0.005]
        assert list(neu.columns) == ["label_value", "maturity_status"]  # 输出 shape 不变

    def test_unknown_label_name_raises(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(ValueError, match="未知标签名"):
            store.load_labels(LabelQuery(label_name="no_such_label", start_date=_D1, end_date=_D1))


class TestResyncFingerprints:
    """resync_partition_fingerprints（panel 热区 + labels 分区覆盖）。"""

    def test_hot_partition_resync(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        rel = "panel/20260105/core.parquet"
        old_fp = store.manifest.partition_fingerprint(rel)
        # 旁路篡改盘上文件（模拟修复写入在 manifest 落盘阶段中断后的不一致）
        _feature_df("20260105", values=(7.7, 8.8)).to_parquet(
            tmp_path / "features" / "panel" / "20260105" / "core.parquet", index=False
        )
        stats = store.resync_partition_fingerprints("测试 resync")
        assert stats["resynced"] == 1 and stats["missing_file"] == 0
        assert store.manifest.partition_fingerprint(rel) != old_fp
        assert any(r["reason"].startswith("[resync]") for r in store.manifest.repairs())

    def test_consistent_partition_noop(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        stats = store.resync_partition_fingerprints("无需校正")
        assert stats == {"checked": 1, "resynced": 0, "missing_file": 0}
        assert store.manifest.repairs() == []

    def test_labels_partition_resync(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_labels("y_ret_20", _label_df("20260105"))
        old_fp = store.manifest.label_partition_fingerprint("y_ret_20", "20260105")
        _label_df("20260105", values=(7.7, 8.8)).to_parquet(
            tmp_path / "labels" / "y_ret_20" / "20260105.parquet", index=False
        )
        stats = store.resync_partition_fingerprints("labels 校正")
        assert stats["resynced"] == 1
        assert store.manifest.label_partition_fingerprint("y_ret_20", "20260105") != old_fp
        assert any("labels/y_ret_20" in r["rel_path"] for r in store.manifest.repairs())

    def test_missing_file_counted(self, tmp_path):
        store = _make_store(tmp_path)
        store.append_features(_D1, "core", _feature_df("20260105"))
        (tmp_path / "features" / "panel" / "20260105" / "core.parquet").unlink()
        stats = store.resync_partition_fingerprints("缺文件")
        assert stats["missing_file"] == 1 and stats["resynced"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
