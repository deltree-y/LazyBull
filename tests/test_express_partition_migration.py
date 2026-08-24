# -*- coding: utf-8 -*-
"""express 迁移季度分区存储回归测试（v0.95.6）。

背景: express 由超大单文件改为按季度 end_date 分区存储，与 forecast/fina_indicator
对齐；旧单文件在首次加载/下载时一次性自动迁移。覆盖:
- Storage.migrate_raw_single_file_to_partitions: 分组写分区、删除旧文件、返回全量
- DataLoader.load_express: 分区优先、旧单文件自动迁移回退、日期标准化
- _try_download_express: 迁移后走分区增量、无数据时全量分区下载
"""

import tempfile
import warnings

import pandas as pd
import pytest

from src.lazybull.data import DataLoader, Storage
from src.lazybull.features.ensure import downloads as ensure_downloads


@pytest.fixture
def temp_storage():
    """临时存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Storage(tmpdir)


def _legacy_express_df() -> pd.DataFrame:
    """旧单文件形式的业绩快报数据（2 只股票、2 个报告期）。"""
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240120",
                "end_date": "20231231",
                "revenue": 100.0,
                "yoy_net_profit": 20.0,
                "diluted_roe": 10.0,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20231028",
                "end_date": "20230930",
                "revenue": 90.0,
                "yoy_net_profit": 18.0,
                "diluted_roe": 9.0,
            },
            {
                "ts_code": "600000.SH",
                "ann_date": "20240125",
                "end_date": "20231231",
                "revenue": 200.0,
                "yoy_net_profit": -5.0,
                "diluted_roe": 6.0,
            },
        ]
    )


class TestMigrateSingleFile:
    def test_migrate_writes_partitions_and_removes_legacy(self, temp_storage):
        temp_storage.save_raw(_legacy_express_df(), "express", is_force=True)

        result = temp_storage.migrate_raw_single_file_to_partitions(
            "express",
            partition_date_col="end_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
        )

        assert result is not None and len(result) == 3
        assert temp_storage.list_partitions("raw", "express") == ["2023-09-30", "2023-12-31"]
        assert temp_storage.load_raw("express") is None  # 旧单文件已删除

    def test_migrate_without_legacy_returns_none(self, temp_storage):
        result = temp_storage.migrate_raw_single_file_to_partitions(
            "express",
            partition_date_col="end_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
        )
        assert result is None

    def test_migrate_missing_partition_col_keeps_legacy(self, temp_storage):
        temp_storage.save_raw(_legacy_express_df(), "express", is_force=True)

        result = temp_storage.migrate_raw_single_file_to_partitions(
            "express",
            partition_date_col="not_exist_col",
            dedup_cols=["ts_code", "end_date", "ann_date"],
        )

        # 缺少分区列时跳过迁移：返回 None（由调用方保留已有分区数据）、不写分区、旧文件保留
        assert result is None
        assert temp_storage.list_partitions("raw", "express") == []
        assert temp_storage.load_raw("express") is not None

    def test_existing_partition_wins_on_key_conflict(self, temp_storage):
        """同主键冲突时已有分区优先，旧单文件不得覆盖新分区数据。"""
        # 新分区 revenue=999（新下载入口写入的更新值）
        temp_storage.save_raw_by_date(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240120",
                        "end_date": "20231231",
                        "revenue": 999.0,
                        "yoy_net_profit": 99.0,
                        "diluted_roe": 9.9,
                    }
                ]
            ),
            "express",
            "20231231",
        )
        # 旧单文件同 key 的旧值 revenue=100
        legacy = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240120",
                    "end_date": "20231231",
                    "revenue": 100.0,
                    "yoy_net_profit": 20.0,
                    "diluted_roe": 10.0,
                }
            ]
        )
        temp_storage.save_raw(legacy, "express", is_force=True)

        result = temp_storage.migrate_raw_single_file_to_partitions(
            "express",
            partition_date_col="end_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
        )

        # 新分区值胜出，旧文件值不得回写覆盖
        assert result is not None and len(result) == 1
        assert result.iloc[0]["revenue"] == 999.0
        assert result.iloc[0]["yoy_net_profit"] == 99.0
        assert temp_storage.load_raw("express") is None

    def test_migrate_suppresses_all_na_concat_warning(self, temp_storage, monkeypatch):
        """混合态迁移保留全 NA 列，且不泄露 pandas concat FutureWarning。"""
        existing = pd.DataFrame(
            [
                {
                    "ts_code": "600000.SH",
                    "ann_date": "20240125",
                    "end_date": "20231231",
                    "optional_metric": None,
                }
            ]
        )
        legacy = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240120",
                    "end_date": "20231231",
                }
            ]
        )
        temp_storage.save_raw_by_date(existing, "express", "20231231")
        temp_storage.save_raw(legacy, "express", is_force=True)

        original_concat = pd.concat

        def concat_with_warning(*args, **kwargs):
            warnings.warn(
                "The behavior of DataFrame concatenation with empty or all-NA entries "
                "is deprecated.",
                FutureWarning,
                stacklevel=2,
            )
            return original_concat(*args, **kwargs)

        monkeypatch.setattr(pd, "concat", concat_with_warning)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = temp_storage.migrate_raw_single_file_to_partitions(
                "express",
                partition_date_col="end_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
            )

        assert result is not None and len(result) == 2
        assert "optional_metric" in result.columns
        assert result["optional_metric"].isna().all()
        assert not any(
            issubclass(item.category, FutureWarning)
            and "DataFrame concatenation with empty or all-NA entries" in str(item.message)
            for item in caught
        )

    def test_invalid_calendar_date_counted_as_skipped(self, temp_storage):
        """八位但非真实日历日期（如 20230230）计入无效跳过，不中断迁移。"""
        legacy = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240120",
                    "end_date": "20231231",
                    "revenue": 100.0,
                    "yoy_net_profit": 20.0,
                    "diluted_roe": 10.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": "20230230",
                    "end_date": "20230230",
                    "revenue": 200.0,
                    "yoy_net_profit": 5.0,
                    "diluted_roe": 8.0,
                },
            ]
        )
        temp_storage.save_raw(legacy, "express", is_force=True)

        result = temp_storage.migrate_raw_single_file_to_partitions(
            "express",
            partition_date_col="end_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
        )

        # 有效记录迁移，非法日历日期计入 skipped：不抛异常
        assert result is not None and len(result) == 1
        assert temp_storage.list_partitions("raw", "express") == ["2023-12-31"]
        # 存在跳过记录 → 旧文件保留
        assert temp_storage.load_raw("express") is not None

    def test_migrate_merges_with_existing_partitions(self, temp_storage):
        """混合态：已有部分分区 + 旧单文件，迁移合并后不丢任何一侧数据。"""
        # 已存在 2024Q4 分区（新版下载入口写入的部分数据）
        temp_storage.save_raw_by_date(
            pd.DataFrame(
                [
                    {
                        "ts_code": "600519.SH",
                        "ann_date": "20240301",
                        "end_date": "20231231",
                        "revenue": 300.0,
                        "yoy_net_profit": 15.0,
                        "diluted_roe": 20.0,
                    }
                ]
            ),
            "express",
            "20231231",
        )
        # 旧单文件含 2023Q3 + 2024Q4 记录
        legacy = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20231028",
                    "end_date": "20230930",
                    "revenue": 90.0,
                    "yoy_net_profit": 18.0,
                    "diluted_roe": 9.0,
                },
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240120",
                    "end_date": "20231231",
                    "revenue": 100.0,
                    "yoy_net_profit": 20.0,
                    "diluted_roe": 10.0,
                },
            ]
        )
        temp_storage.save_raw(legacy, "express", is_force=True)

        result = temp_storage.migrate_raw_single_file_to_partitions(
            "express",
            partition_date_col="end_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
        )

        # 分区全量 = 已有分区 + 旧单文件，不丢任何一侧
        assert result is not None and len(result) == 3
        assert set(result["ts_code"]) == {"000001.SZ", "600519.SH"}
        assert temp_storage.load_raw("express") is None
        assert temp_storage.list_partitions("raw", "express") == [
            "2023-09-30",
            "2023-12-31",
        ]

    def test_invalid_partition_key_keeps_legacy_file(self, temp_storage):
        """存在无效分区键时：有效行迁移、无效行保留旧文件（不静默丢数）。"""
        legacy = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240120",
                    "end_date": "20231231",
                    "revenue": 100.0,
                    "yoy_net_profit": 20.0,
                    "diluted_roe": 10.0,
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": "20240120",
                    "end_date": "无效日期",
                    "revenue": 200.0,
                    "yoy_net_profit": 5.0,
                    "diluted_roe": 8.0,
                },
            ]
        )
        temp_storage.save_raw(legacy, "express", is_force=True)

        result = temp_storage.migrate_raw_single_file_to_partitions(
            "express",
            partition_date_col="end_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
        )

        # 有效记录已迁移
        assert result is not None and len(result) == 1
        assert temp_storage.list_partitions("raw", "express") == ["2023-12-31"]
        # 旧单文件因存在无效记录而保留，不静默删除
        assert temp_storage.load_raw("express") is not None


class TestLoadExpressMigration:
    def test_load_express_migrates_legacy_single_file(self, temp_storage):
        temp_storage.save_raw(_legacy_express_df(), "express", is_force=True)
        loader = DataLoader(temp_storage)

        df = loader.load_express()

        assert df is not None and len(df) == 3
        assert set(df["end_date"]) == {"20230930", "20231231"}
        assert set(df["ann_date"]) == {"20231028", "20240120", "20240125"}
        # 迁移完成后旧单文件删除、分区保留
        assert temp_storage.load_raw("express") is None
        assert len(temp_storage.list_partitions("raw", "express")) == 2

    def test_load_express_prefers_partitions(self, temp_storage):
        temp_storage.save_raw_by_date(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240120",
                        "end_date": "20231231",
                        "revenue": 100.0,
                        "yoy_net_profit": 20.0,
                        "diluted_roe": 10.0,
                    }
                ]
            ),
            "express",
            "20231231",
        )
        loader = DataLoader(temp_storage)

        df = loader.load_express()

        assert df is not None and len(df) == 1
        assert df.iloc[0]["end_date"] == "20231231"

    def test_load_express_empty_returns_none(self, temp_storage):
        loader = DataLoader(temp_storage)
        assert loader.load_express() is None

    def test_load_express_migrates_mixed_state(self, temp_storage):
        """混合态（部分分区 + 旧单文件）加载时迁移合并，不丢旧数据。"""
        temp_storage.save_raw_by_date(
            pd.DataFrame(
                [
                    {
                        "ts_code": "600519.SH",
                        "ann_date": "20240301",
                        "end_date": "20231231",
                        "revenue": 300.0,
                        "yoy_net_profit": 15.0,
                        "diluted_roe": 20.0,
                    }
                ]
            ),
            "express",
            "20231231",
        )
        temp_storage.save_raw(_legacy_express_df(), "express", is_force=True)

        df = DataLoader(temp_storage).load_express()

        # 旧单文件与已有分区合并，全部可见
        assert df is not None and len(df) == 4
        assert set(df["ts_code"]) == {"000001.SZ", "600000.SH", "600519.SH"}
        assert temp_storage.load_raw("express") is None

    def test_load_express_empty_legacy_does_not_shadow_partitions(self, temp_storage):
        """空旧单文件不遮蔽已有分区数据。"""
        temp_storage.save_raw_by_date(
            pd.DataFrame(
                [
                    {
                        "ts_code": "600519.SH",
                        "ann_date": "20240301",
                        "end_date": "20231231",
                        "revenue": 300.0,
                        "yoy_net_profit": 15.0,
                        "diluted_roe": 20.0,
                    }
                ]
            ),
            "express",
            "20231231",
        )
        temp_storage.save_raw(pd.DataFrame(), "express", is_force=True)

        df = DataLoader(temp_storage).load_express()

        # 分区数据保留，空旧文件作为垃圾被清理
        assert df is not None and len(df) == 1
        assert df.iloc[0]["ts_code"] == "600519.SH"
        assert temp_storage.load_raw("express") is None

    def test_load_express_legacy_missing_col_does_not_shadow_partitions(self, temp_storage):
        """旧单文件缺少分区列时保留已有分区数据，不遮蔽。"""
        temp_storage.save_raw_by_date(
            pd.DataFrame(
                [
                    {
                        "ts_code": "600519.SH",
                        "ann_date": "20240301",
                        "end_date": "20231231",
                        "revenue": 300.0,
                        "yoy_net_profit": 15.0,
                        "diluted_roe": 20.0,
                    }
                ]
            ),
            "express",
            "20231231",
        )
        # 破损旧单文件：无 end_date 列
        temp_storage.save_raw(
            pd.DataFrame([{"ts_code": "000001.SZ", "ann_date": "20240120", "revenue": 100.0}]),
            "express",
            is_force=True,
        )

        df = DataLoader(temp_storage).load_express()

        # 分区数据保留，破损旧文件不被采用且保留待处理
        assert df is not None and len(df) == 1
        assert df.iloc[0]["ts_code"] == "600519.SH"
        assert temp_storage.load_raw("express") is not None


class TestTryDownloadExpressPartition:
    def test_incremental_uses_partition_params_after_migration(self, temp_storage, monkeypatch):
        """旧单文件存在时先迁移分区，再按分区参数走增量补齐。"""
        temp_storage.save_raw(_legacy_express_df(), "express", is_force=True)
        monkeypatch.setattr(ensure_downloads, "_MIN_EXPRESS_RECORDS", 1)
        captured = {}

        def _fake_catchup(**kwargs):
            captured.update(kwargs)
            return kwargs["existing_df"]

        monkeypatch.setattr(
            ensure_downloads, "_incremental_catchup_by_calendar_date", _fake_catchup
        )
        client = object()

        result = ensure_downloads._try_download_express(client, temp_storage, "20260201")

        assert result is not None and len(result) == 3
        assert captured["dataset_name"] == "express"
        assert captured["partition_date_col"] == "end_date"
        assert captured["partition_mode"] == "quarter"
        assert temp_storage.load_raw("express") is None

    def test_download_empty_legacy_does_not_shadow_partitions(self, temp_storage, monkeypatch):
        """空旧单文件不遮蔽已有分区：仍走增量而非全量重下。"""
        temp_storage.save_raw_by_date(
            pd.DataFrame(
                [
                    {
                        "ts_code": "600519.SH",
                        "ann_date": "20240301",
                        "end_date": "20231231",
                        "revenue": 300.0,
                        "yoy_net_profit": 15.0,
                        "diluted_roe": 20.0,
                    }
                ]
            ),
            "express",
            "20231231",
        )
        temp_storage.save_raw(pd.DataFrame(), "express", is_force=True)
        monkeypatch.setattr(ensure_downloads, "_MIN_EXPRESS_RECORDS", 1)
        captured = {}

        def _fake_catchup(**kwargs):
            captured.update(kwargs)
            return kwargs["existing_df"]

        monkeypatch.setattr(
            ensure_downloads, "_incremental_catchup_by_calendar_date", _fake_catchup
        )

        result = ensure_downloads._try_download_express(object(), temp_storage, "20260201")

        # 已有分区数据保留并进入增量补齐，空旧文件不触发全量重下
        assert result is not None and len(result) == 1
        assert result.iloc[0]["ts_code"] == "600519.SH"
        assert captured["existing_df"] is not None and len(captured["existing_df"]) == 1

    def test_full_download_uses_partition_mode(self, temp_storage, monkeypatch):
        """无本地数据时全量下载按季度分区落盘。"""
        captured = {}
        rebuilt = pd.DataFrame([{"ts_code": "000001.SZ"}])
        loaded = iter([None, rebuilt])

        def _fake_bulk(client, storage, **kwargs):
            captured.update(kwargs)
            return None

        monkeypatch.setattr(ensure_downloads, "_bulk_download_by_period", _fake_bulk)
        monkeypatch.setattr(ensure_downloads, "_load_all_partitions", lambda s, n: next(loaded))
        monkeypatch.setattr(ensure_downloads, "_MIN_EXPRESS_RECORDS", 1)
        client = object()

        result = ensure_downloads._try_download_express(client, temp_storage, "20260201")

        assert result is rebuilt
        assert captured["dataset_name"] == "express"
        assert captured["api_name"] == "express_vip"
        assert captured["partition_by_period"] is True
        # 数据不足为异常态：全量重下不得跳过已有残缺季度
        assert captured["force"] is True

    def test_full_download_below_threshold_raises(self, temp_storage, monkeypatch):
        """强制重建后仍低于完整性门槛时明确失败，不返回残缺数据。"""
        rebuilt = pd.DataFrame([{"ts_code": "000001.SZ"}])
        loaded = iter([None, rebuilt])

        monkeypatch.setattr(ensure_downloads, "_bulk_download_by_period", lambda *a, **k: None)
        monkeypatch.setattr(ensure_downloads, "_load_all_partitions", lambda s, n: next(loaded))
        monkeypatch.setattr(ensure_downloads, "_MIN_EXPRESS_RECORDS", 2)

        with pytest.raises(RuntimeError, match="强制全量重建后数据仍不足"):
            ensure_downloads._try_download_express(object(), temp_storage, "20260201")

    def test_force_download_failure_raises(self, temp_storage, monkeypatch):
        """force=True 有季度下载异常时明确失败，不把残缺分区当作重建成功。"""
        from src.lazybull.features.ensure import bulk as ensure_bulk

        monkeypatch.setattr(
            ensure_bulk,
            "_generate_quarter_periods",
            lambda start_year, end_year: ["20231231"],
        )

        class _FailingClient:
            @property
            def pro(self):
                return self

            def query(self, api_name, fields=None, **kwargs):
                raise RuntimeError("模拟接口失败")

        with pytest.raises(RuntimeError, match="强制全量下载失败.*20231231"):
            ensure_bulk._bulk_download_by_period(
                _FailingClient(),
                temp_storage,
                dataset_name="express",
                api_name="express_vip",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                partition_by_period=True,
                force=True,
            )

    def test_force_redownloads_existing_periods(self, temp_storage):
        """force=True 时断点续传不跳过已有季度（残缺恢复）。"""
        from src.lazybull.features.ensure.bulk import _bulk_download_by_period

        temp_storage.save_raw_by_date(
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "ann_date": "20240120",
                        "end_date": "20231231",
                    }
                ]
            ),
            "express",
            "20231231",
        )

        class _FakeClient:
            def __init__(self):
                self.queried = []

            @property
            def pro(self):
                return self

            def query(self, api_name, fields=None, **kwargs):
                self.queried.append(kwargs.get("period"))
                return pd.DataFrame()

        client = _FakeClient()
        _bulk_download_by_period(
            client,
            temp_storage,
            dataset_name="express",
            api_name="express_vip",
            dedup_cols=["ts_code", "end_date", "ann_date"],
            partition_by_period=True,
            force=True,
        )
        # force 模式已有季度也重查
        assert "20231231" in client.queried

        client_keep = _FakeClient()
        _bulk_download_by_period(
            client_keep,
            temp_storage,
            dataset_name="express",
            api_name="express_vip",
            dedup_cols=["ts_code", "end_date", "ann_date"],
            partition_by_period=True,
            force=False,
        )
        # 默认断点续传跳过已有季度
        assert "20231231" not in client_keep.queried
