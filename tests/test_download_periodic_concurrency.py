# -*- coding: utf-8 -*-
"""download_by_period 并发化回归测试。

背景: fund_portfolio 等按季度下载的数据集原为纯串行逐季度下载,
每个季度内部还要逐页 (page_limit=8000) 翻页, 每请求服务端响应约 5s,
导致大季度 (上百万条) 单季度就要十几分钟。改造后复用 _run_concurrent
(受 TushareClient 令牌桶限频约束), 网络等待并行化。

覆盖:
- _run_concurrent 的 collect 模式: 串行/并发路径均按 work_items 顺序返回 worker 返回值
- download_by_period 分区模式 (fund_portfolio/fina_indicator/forecast): 各季度独立落盘, 计数正确
- download_by_period 非分区模式 (express 等): 全部下载后统一合并去重落盘
"""

import pandas as pd
import pytest

import scripts.raw_download.core as raw_core
from scripts.raw_download.periodic import _dedup_rows, download_by_period


@pytest.fixture(autouse=True)
def _reset_error_collector():
    """每个测试前清空全局错误收集器，避免跨测试污染。"""
    with raw_core.ERROR_COLLECTOR._lock:
        raw_core.ERROR_COLLECTOR._errors.clear()
    yield


class TestRunConcurrentCollect:
    """_run_concurrent 的 collect 模式应支持收集返回值且保持输入顺序。"""

    def test_serial_path_returns_ordered_results(self, monkeypatch):
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 1)
        results = raw_core._run_concurrent([1, 2, 3], lambda x: x * 10, label="t", collect=True)
        assert results == [10, 20, 30]

    def test_parallel_path_returns_ordered_results(self, monkeypatch):
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 4)
        results = raw_core._run_concurrent(
            list(range(10)), lambda x: x * 10, label="t", collect=True
        )
        assert results == [x * 10 for x in range(10)]

    def test_collect_false_returns_none(self, monkeypatch):
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 2)
        results = raw_core._run_concurrent([1, 2, 3], lambda x: x * 10, label="t", collect=False)
        assert results is None

    def test_max_workers_overrides_global(self, monkeypatch):
        """max_workers 覆盖全局并发; 仍按序收集结果。"""
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 16)
        results = raw_core._run_concurrent(
            list(range(6)), lambda x: x, label="t", collect=True, max_workers=2
        )
        assert results == list(range(6))

    def test_max_workers_one_serial(self, monkeypatch):
        """max_workers=1 走串行路径, 行为与 _DOWNLOAD_CONCURRENCY=1 一致。"""
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 16)
        results = raw_core._run_concurrent(
            [1, 2, 3], lambda x: x * 10, label="t", collect=True, max_workers=1
        )
        assert results == [10, 20, 30]


class TestDownloadByPeriodConcurrency:
    """download_by_period 并发化: 分区/非分区模式行为与串行一致。"""

    @staticmethod
    def _make_fake_storage():
        class _FakeStorage:
            def __init__(self):
                self.saved_partitions = {}
                self.saved_merged = None

            def is_data_exists(self, layer, name, period):
                return False

            def save_raw_by_date(self, df, name, period):
                self.saved_partitions[period] = df.copy()

            def load_raw(self, name):
                return None

            def save_raw(self, df, name, is_force=False):
                self.saved_merged = df.copy()

        return _FakeStorage()

    @staticmethod
    def _make_fake_client(rows_by_period):
        class _FakeClient:
            def query(self, api_name, fields=None, **kwargs):
                period = kwargs.get("period")
                n = rows_by_period.get(period, 0)
                prefix = str(period)
                return pd.DataFrame({"ts_code": [f"{prefix}_{i}" for i in range(n)]})

        return _FakeClient()

    def test_partition_mode_writes_each_period(self, monkeypatch):
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 4)
        storage = self._make_fake_storage()
        client = self._make_fake_client({"20241231": 3, "20250331": 0, "20250630": 5})
        download_by_period(
            client,
            storage,
            dataset_name="fund_portfolio",
            api_name="fund_portfolio",
            start_date="20240101",
            end_date="20251231",
            dedup_cols=["ts_code"],
            page_limit=10,
            partition_by_period=True,
        )
        # 有数据的季度各自独立落盘; 空季度不写盘; 非分区合并不触发
        assert set(storage.saved_partitions.keys()) == {"20241231", "20250630"}
        assert len(storage.saved_partitions["20241231"]) == 3
        assert len(storage.saved_partitions["20250630"]) == 5
        assert storage.saved_merged is None

    def test_partition_mode_single_period_serial(self, monkeypatch):
        """单季度 + 串行降级路径仍正确落盘。"""
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 1)
        storage = self._make_fake_storage()
        client = self._make_fake_client({"20241231": 3})
        download_by_period(
            client,
            storage,
            dataset_name="fund_portfolio",
            api_name="fund_portfolio",
            start_date="20240101",
            end_date="20241231",
            dedup_cols=["ts_code"],
            page_limit=10,
            partition_by_period=True,
        )
        assert set(storage.saved_partitions.keys()) == {"20241231"}

    def test_merged_mode_concats_all_periods(self, monkeypatch):
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 4)
        storage = self._make_fake_storage()
        client = self._make_fake_client({"20241231": 3, "20250630": 5})
        download_by_period(
            client,
            storage,
            dataset_name="express",
            api_name="express_vip",
            start_date="20240101",
            end_date="20251231",
            dedup_cols=["ts_code"],
            page_limit=10,
        )
        # 非分区模式: 各季度 df 汇总后统一合并落盘, 不逐季度写分区
        assert storage.saved_merged is not None
        assert len(storage.saved_merged) == 8
        assert storage.saved_partitions == {}

    def test_merged_mode_dedups(self, monkeypatch):
        """合并模式按 dedup_cols 去重。"""
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 2)

        class _ClientWithDup:
            def query(self, api_name, fields=None, **kwargs):
                period = kwargs.get("period")
                if period == "20241231":
                    return pd.DataFrame({"ts_code": ["a", "b"]})
                # 20250630 与上一季度重复一条
                return pd.DataFrame({"ts_code": ["b", "c"]})

        storage = self._make_fake_storage()
        download_by_period(
            _ClientWithDup(),
            storage,
            dataset_name="express",
            api_name="express_vip",
            start_date="20240101",
            end_date="20251231",
            dedup_cols=["ts_code"],
            page_limit=10,
        )
        assert storage.saved_merged is not None
        assert set(storage.saved_merged["ts_code"]) == {"a", "b", "c"}

    def test_partition_mode_dedups_before_save(self, monkeypatch):
        """分区模式落盘前按 dedup_cols 去重（此前去重仅作用于非分区合并路径）。

        同一报告期"季报前十大 + 半年报/年报全量"两批公告同 (ts_code, end_date)，
        未去重会导致下游聚合 sum 双重计数。
        """
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 2)

        class _ClientWithDup:
            def query(self, api_name, fields=None, **kwargs):
                period = kwargs.get("period")
                if period == "20241231":
                    # 同 ts_code 两条（不同 ann_date 两批公告）
                    return pd.DataFrame(
                        {
                            "ts_code": ["a", "b", "a"],
                            "ann_date": ["20241020", "20241020", "20241101"],
                        }
                    )
                return pd.DataFrame()

        storage = self._make_fake_storage()
        download_by_period(
            _ClientWithDup(),
            storage,
            dataset_name="fund_portfolio",
            api_name="fund_portfolio",
            start_date="20240101",
            end_date="20241231",
            dedup_cols=["ts_code"],
            page_limit=10,
            partition_by_period=True,
            sort_cols=["ann_date"],
        )
        saved = storage.saved_partitions["20241231"]
        assert set(saved["ts_code"]) == {"a", "b"}
        # keep="last" 按 ann_date 升序后保留最晚公告记录
        assert saved[saved["ts_code"] == "a"]["ann_date"].iloc[0] == "20241101"

    def test_dedup_rows_prefers_official_latest_update_flag(self):
        """同键财报行必须选择 update_flag=1，且不受接口返回顺序影响。"""
        rows = pd.DataFrame(
            {
                "ts_code": ["a", "a"],
                "end_date": ["20231231", "20231231"],
                "f_ann_date": ["20240430", "20240430"],
                "update_flag": [1, 0],
                "free_cashflow": [1.0, 2.0],
            }
        )
        key = ["ts_code", "end_date", "f_ann_date"]

        forward = _dedup_rows(rows, key, sort_cols=["f_ann_date"])
        reverse = _dedup_rows(rows.iloc[::-1], key, sort_cols=["f_ann_date"])

        assert forward.loc[0, "free_cashflow"] == 1.0
        assert reverse.loc[0, "free_cashflow"] == 1.0
