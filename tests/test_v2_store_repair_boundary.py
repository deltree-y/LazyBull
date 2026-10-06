# -*- coding: utf-8 -*-
"""单元 4 修复通道测试：repair_archive_partition + 冷区日级已写判定（D-09）。"""

import pandas as pd
import pytest

from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.panel_builder import bootstrap_manifest


def _day_df(date: str, codes: list[str], base: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date] * len(codes),
            "ts_code": codes,
            "vol": [base + i for i in range(len(codes))],
        }
    )


@pytest.fixture()
def store(tmp_path):
    s = PanelDataStore(tmp_path)
    bootstrap_manifest(s)
    return s


class TestRepairArchivePartition:
    def test_merge_appends_missing_days(self, store):
        """合并 = 存量不变 + 新日追加 + 指纹更新 + 修复登记。"""
        store.append_archive_features("2024-01", "core", _day_df("20240102", ["A", "B"], 1.0))
        old_fp = store.manifest.partition_fingerprint("panel_archive/2024-01/core.parquet")
        result = store.repair_archive_partition(
            "2024-01", "core", _day_df("20240103", ["A", "B"], 5.0), reason="缺口回补测试"
        )
        merged = pd.read_parquet(store.archive_dir / "2024-01" / "core.parquet")
        assert sorted(merged["trade_date"].astype(str).unique()) == ["20240102", "20240103"]
        assert result["added_dates"] == ["20240103"]
        new_fp = store.manifest.partition_fingerprint("panel_archive/2024-01/core.parquet")
        assert new_fp != old_fp
        repairs = store.manifest.repairs()
        assert len(repairs) == 1 and repairs[0]["reason"] == "缺口回补测试"
        assert repairs[0]["old_fingerprint"] == old_fp

    def test_reject_overlap_day(self, store):
        store.append_archive_features("2024-01", "core", _day_df("20240102", ["A"], 1.0))
        with pytest.raises(RuntimeError, match="相交"):
            store.repair_archive_partition(
                "2024-01", "core", _day_df("20240102", ["A"], 9.0), reason="相交应拒绝"
            )

    def test_reject_empty_reason(self, store):
        store.append_archive_features("2024-01", "core", _day_df("20240102", ["A"], 1.0))
        with pytest.raises(RuntimeError, match="理由"):
            store.repair_archive_partition(
                "2024-01", "core", _day_df("20240103", ["A"], 2.0), reason="  "
            )

    def test_reject_missing_target(self, store):
        with pytest.raises(RuntimeError, match="不存在"):
            store.repair_archive_partition(
                "2024-01", "core", _day_df("20240103", ["A"], 2.0), reason="目标不存在"
            )

    def test_reject_month_outside_days(self, store):
        store.append_archive_features("2024-01", "core", _day_df("20240102", ["A"], 1.0))
        with pytest.raises(RuntimeError, match="月外日期"):
            store.repair_archive_partition(
                "2024-01", "core", _day_df("20240201", ["A"], 2.0), reason="跨月应拒绝"
            )


class TestReplaceArchiveDays:
    """replace_archive_days（按日替换修复通道，D-09/D-10 收口）。"""

    def _two_days(self) -> pd.DataFrame:
        return pd.concat(
            [_day_df("20240102", ["A", "B"], 1.0), _day_df("20240103", ["A", "B"], 2.0)],
            ignore_index=True,
        )

    def test_replace_swaps_only_target_day(self, store):
        store.append_archive_features("2024-01", "core", self._two_days())
        old_fp = store.manifest.partition_fingerprint("panel_archive/2024-01/core.parquet")
        result = store.replace_archive_days(
            "2024-01", "core", _day_df("20240103", ["A", "B"], 99.0), reason="替换测试"
        )
        merged = pd.read_parquet(store.archive_dir / "2024-01" / "core.parquet")
        assert result["replaced_dates"] == ["20240103"]
        # 存量其余日逐位保留
        d2 = merged[merged["trade_date"].astype(str) == "20240102"]["vol"].tolist()
        assert d2 == [1.0, 2.0]
        # 目标日已替换
        d3 = merged[merged["trade_date"].astype(str) == "20240103"]["vol"].tolist()
        assert d3 == [99.0, 100.0]
        assert store.manifest.partition_fingerprint("panel_archive/2024-01/core.parquet") != old_fp
        repairs = store.manifest.repairs()
        assert repairs[0]["reason"].startswith("[replace_days]")

    def test_reject_day_not_in_existing(self, store):
        store.append_archive_features("2024-01", "core", self._two_days())
        with pytest.raises(RuntimeError, match="不在存量分区"):
            store.replace_archive_days(
                "2024-01", "core", _day_df("20240104", ["A"], 1.0), reason="缺日应拒绝"
            )

    def test_reject_empty_reason(self, store):
        store.append_archive_features("2024-01", "core", self._two_days())
        with pytest.raises(RuntimeError, match="理由"):
            store.replace_archive_days(
                "2024-01", "core", _day_df("20240103", ["A"], 3.0), reason=""
            )

    def test_rewrite_partition_changes_content_with_audit(self, store):
        """rewrite_partition：内容变化 + 理由登记 + 指纹更新 + 键校验。"""
        store.append_archive_features("2024-01", "core", _day_df("20240102", ["A", "B"], 1.0))
        old_fp = store.manifest.partition_fingerprint("panel_archive/2024-01/core.parquet")
        store.rewrite_partition(
            "panel_archive/2024-01/core.parquet",
            _day_df("20240102", ["A", "B"], 9.0),
            reason="列值修正测试",
        )
        df = pd.read_parquet(store.archive_dir / "2024-01" / "core.parquet")
        assert df["vol"].tolist() == [9.0, 10.0]
        new_fp = store.manifest.partition_fingerprint("panel_archive/2024-01/core.parquet")
        assert new_fp != old_fp
        repairs = store.manifest.repairs()
        assert repairs[-1]["reason"] == "[rewrite_partition] 列值修正测试"

    def test_rewrite_partition_reject_unregistered(self, store):
        with pytest.raises(RuntimeError, match="不存在"):
            store.rewrite_partition(
                "panel_archive/2024-01/core.parquet",
                _day_df("20240102", ["A"], 1.0),
                reason="目标不存在",
            )


class TestColdDayLevelWritten:
    """backfill_panel._written_days 的冷区日级精度（月登记但缺日 ⇒ 未写）。"""

    def test_month_registered_but_day_absent_is_unwritten(self, tmp_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "backfill_panel", "scripts/v2_p1/backfill_panel.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        # 月内只写 0102、0103 两日（缺 0104）
        df = pd.concat(
            [_day_df("20240102", ["A"], 1.0), _day_df("20240103", ["A"], 2.0)],
            ignore_index=True,
        )
        store.append_archive_features("2024-01", "core", df)
        for g in [g for g in __import__("src.lazybull.v2.store.column_groups", fromlist=["PANEL_GROUPS"]).PANEL_GROUPS if g != "core"]:
            store.append_archive_features(
                "2024-01", g, pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["A"]})
            )
        written = mod._written_days(store, ["20240102", "20240103", "20240104"])
        assert written == {"20240102", "20240103"}  # 0104 月登记但缺日 ⇒ 未写
