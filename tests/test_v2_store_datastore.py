# -*- coding: utf-8 -*-
"""v2 PanelDataStore 测试（tmp_path 合成数据，不触碰真实 data/）。

覆盖：两阶段提交（成功 / 冲突 raise / 指纹一致放行 / .tmp 清理）、写锁互斥与 stale 抢占、
孤儿 GC、load_features 跨热/冷区合并 + 三类 ValueError + universe 过滤、
append_labels forming 可重写 / sealed 拒改写 / 指纹一致 no-op、append_ledger_entry 序列化与校验。
"""

import json
import types

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
