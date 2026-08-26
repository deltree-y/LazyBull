# -*- coding: utf-8 -*-
"""forecast / report_rc 按时间分区存储回归测试。

背景: forecast (按季度 end_date) 与 report_rc (按年 report_date) 由超大独立单文件
改为按时间分区存储, 与 fina_indicator/cashflow/fund_portfolio 对齐; 增量补齐
不再整文件读-合并-重写, 而是按分区键路由写入对应分区。

覆盖:
- _partition_date_str: quarter/year 分区键映射
- _append_and_save_partitioned: 增量按分区路由、分区内去重、返回全量
- _incremental_catchup_by_calendar_date: 分区模式增量补齐
- DataLoader.load_forecast / load_report_rc: 纯分区加载 (无单文件兜底)
"""

import tempfile
import warnings

import pandas as pd
import pytest

from src.lazybull.data import DataLoader, Storage
from src.lazybull.features.ensure.incremental import (
    _append_and_save_partitioned,
    _incremental_catchup_by_calendar_date,
    _load_all_partitions,
    _partition_date_str,
)


@pytest.fixture
def temp_storage():
    """临时存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        yield storage


class TestPartitionDateStr:
    def test_quarter_mode_maps_as_is(self):
        # 报告期 end_date 本身即季度末, 原样映射为 YYYY-MM-DD
        assert _partition_date_str("20260331", "quarter") == "2026-03-31"
        assert _partition_date_str("2025-12-31", "quarter") == "2025-12-31"

    def test_year_mode_maps_to_year_end(self):
        # report_date 按年聚合到该年 12-31
        assert _partition_date_str("20260410", "year") == "2026-12-31"
        assert _partition_date_str("2025-01-02", "year") == "2025-12-31"

    def test_invalid_date_returns_none(self):
        assert _partition_date_str(None, "quarter") is None
        assert _partition_date_str("", "year") is None
        assert _partition_date_str("not-a-date", "year") is None

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError):
            _partition_date_str("20260101", "month")


class TestAppendAndSavePartitioned:
    def test_routes_forecast_to_quarter_partitions(self, temp_storage):
        """forecast 增量按 end_date 路由到对应季度分区。"""
        new_df = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "ann_date": "20260411", "end_date": "20260331", "v": 1},
                {"ts_code": "000002.SZ", "ann_date": "20260412", "end_date": "20260331", "v": 2},
                {"ts_code": "000003.SZ", "ann_date": "20260413", "end_date": "20260630", "v": 3},
            ]
        )
        result = _append_and_save_partitioned(
            temp_storage,
            "forecast",
            new_df,
            dedup_cols=["ts_code", "end_date", "ann_date"],
            partition_date_col="end_date",
            partition_mode="quarter",
        )
        assert temp_storage.list_partitions("raw", "forecast") == ["2026-03-31", "2026-06-30"]
        assert result is not None and len(result) == 3

    def test_dedup_within_partition(self, temp_storage):
        """同一分区内按 dedup_cols 去重 (与已有分区数据合并)。"""
        existing = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "ann_date": "20260411", "end_date": "20260331", "v": 1},
            ]
        )
        temp_storage.save_raw_by_date(existing, "forecast", "20260331")

        new_df = pd.DataFrame(
            [
                # 与已有分区重复 (同 ts_code/end_date/ann_date) -> 去重
                {"ts_code": "000001.SZ", "ann_date": "20260411", "end_date": "20260331", "v": 1},
                # 新记录
                {"ts_code": "000002.SZ", "ann_date": "20260412", "end_date": "20260331", "v": 2},
            ]
        )
        result = _append_and_save_partitioned(
            temp_storage,
            "forecast",
            new_df,
            dedup_cols=["ts_code", "end_date", "ann_date"],
            partition_date_col="end_date",
            partition_mode="quarter",
        )
        assert result is not None and len(result) == 2
        # 分区内仅 2 条 (重复已去)
        part = temp_storage.load_raw_by_date("forecast", "2026-03-31")
        assert part is not None and len(part) == 2

    def test_routes_report_rc_to_year_partitions(self, temp_storage):
        """report_rc 增量按 report_date 路由到对应年份分区。"""
        new_df = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20260410",
                    "org_name": "a",
                    "author_name": "分析师甲",
                    "report_title": "2026年研报",
                    "quarter": "2026Q1",
                },
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20251230",
                    "org_name": "b",
                    "author_name": "分析师乙",
                    "report_title": "2025年研报",
                    "quarter": "2025Q4",
                },
            ]
        )
        _append_and_save_partitioned(
            temp_storage,
            "report_rc",
            new_df,
            dedup_cols=["ts_code", "report_date", "org_name", "quarter"],
            partition_date_col="report_date",
            partition_mode="year",
        )
        assert temp_storage.list_partitions("raw", "report_rc") == ["2025-12-31", "2026-12-31"]

    def test_report_rc_dedup_preserves_distinct_reports_with_same_old_four_keys(self, temp_storage):
        """标题或作者不同即为不同研报，不能被旧四键误删。"""
        common = {
            "ts_code": "000001.SZ",
            "report_date": "20260410",
            "org_name": "机构甲",
            "quarter": "2026Q4",
        }
        new_df = pd.DataFrame(
            [
                {**common, "author_name": "分析师甲", "report_title": "年度报告"},
                {**common, "author_name": "分析师乙", "report_title": "业绩点评"},
                {**common, "author_name": "分析师甲", "report_title": "年度报告"},
            ]
        )

        _append_and_save_partitioned(
            temp_storage,
            "report_rc",
            new_df,
            dedup_cols=["ts_code", "report_date", "org_name", "quarter"],
            partition_date_col="report_date",
            partition_mode="year",
        )

        saved = temp_storage.load_raw_by_date("report_rc", "2026-12-31")
        assert saved is not None
        assert len(saved) == 2
        assert set(saved["report_title"]) == {"年度报告", "业绩点评"}

    def test_empty_new_df_returns_existing_partitions(self, temp_storage):
        existing = pd.DataFrame(
            [{"ts_code": "000001.SZ", "ann_date": "20260411", "end_date": "20260331"}]
        )
        temp_storage.save_raw_by_date(existing, "forecast", "20260331")
        result = _append_and_save_partitioned(
            temp_storage,
            "forecast",
            pd.DataFrame(),
            dedup_cols=["ts_code", "end_date", "ann_date"],
            partition_date_col="end_date",
            partition_mode="quarter",
        )
        assert result is not None and len(result) == 1


class TestIncrementalCatchupPartition:
    def test_writes_incremental_to_partitions(self, temp_storage):
        """分区模式增量补齐: 新公告路由写入对应分区, 并返回全量。"""
        existing = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "ann_date": "20260410", "end_date": "20260331"},
            ]
        )
        temp_storage.save_raw_by_date(existing, "forecast", "20260331")

        queried = []

        def _fetch(ann_date):
            queried.append(ann_date)
            if ann_date == "20260412":
                return pd.DataFrame(
                    [
                        {"ts_code": "000002.SZ", "ann_date": "20260412", "end_date": "20260630"},
                    ]
                )
            return pd.DataFrame()

        result = _incremental_catchup_by_calendar_date(
            storage=temp_storage,
            dataset_name="forecast",
            existing_df=_load_all_partitions(temp_storage, "forecast"),
            trade_date="20260413",
            date_col="ann_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
            fetch_by_date=_fetch,
            partition_date_col="end_date",
            partition_mode="quarter",
        )

        # 2026-04-11 ~ 04-13 逐日补齐
        assert queried == ["20260411", "20260412", "20260413"]
        # 新数据按 end_date=20260630 路由到新季度分区
        assert temp_storage.list_partitions("raw", "forecast") == ["2026-03-31", "2026-06-30"]
        assert result is not None and len(result) == 2

    def test_without_partition_params_keeps_single_file_append(self, temp_storage):
        """未传分区参数时沿用整文件追加 (不影响 stk_holdernumber 等单文件数据集)。"""
        existing = pd.DataFrame(
            [{"ts_code": "000001.SZ", "ann_date": "20260410", "end_date": "20260331"}]
        )
        temp_storage.save_raw(existing, "stk_holdernumber", is_force=True)

        def _fetch(ann_date):
            if ann_date == "20260412":
                return pd.DataFrame(
                    [{"ts_code": "000002.SZ", "ann_date": "20260412", "end_date": "20260630"}]
                )
            return pd.DataFrame()

        result = _incremental_catchup_by_calendar_date(
            storage=temp_storage,
            dataset_name="stk_holdernumber",
            existing_df=temp_storage.load_raw("stk_holdernumber"),
            trade_date="20260413",
            date_col="ann_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
            fetch_by_date=_fetch,
        )

        # 仍写单文件 (非分区), 返回合并全量
        assert temp_storage.list_partitions("raw", "stk_holdernumber") == []
        single = temp_storage.load_raw("stk_holdernumber")
        assert single is not None and len(single) == 2
        assert result is not None and len(result) == 2


class TestLoaderPartitionLoading:
    def test_load_forecast_all_partitions(self, temp_storage):
        """load_forecast 无日期参数: 合并全部季度分区, 日期列标准化为 YYYYMMDD。"""
        df1 = pd.DataFrame(
            [{"ts_code": "000001.SZ", "ann_date": "20260411", "end_date": "20260331"}]
        )
        df2 = pd.DataFrame(
            [{"ts_code": "000002.SZ", "ann_date": "20260412", "end_date": "20260630"}]
        )
        temp_storage.save_raw_by_date(df1, "forecast", "20260331")
        temp_storage.save_raw_by_date(df2, "forecast", "20260630")

        loader = DataLoader(temp_storage)
        result = loader.load_forecast()
        assert result is not None and len(result) == 2
        assert result["ann_date"].iloc[0] == "20260411"
        assert result["end_date"].iloc[0] == "20260331"

    def test_load_forecast_no_partitions_returns_none(self, temp_storage):
        """无分区时返回 None (纯分区, 无单文件兜底)。"""
        loader = DataLoader(temp_storage)
        assert loader.load_forecast() is None

    def test_load_report_rc_all_year_partitions(self, temp_storage):
        """load_report_rc: 合并全部年分区, report_date 标准化为 YYYYMMDD。"""
        df1 = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20251230",
                    "org_name": "a",
                    "author_name": "分析师甲",
                    "report_title": "2025年研报",
                    "quarter": "2025Q4",
                }
            ]
        )
        df2 = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20260410",
                    "org_name": "b",
                    "author_name": "分析师乙",
                    "report_title": "2026年研报",
                    "quarter": "2026Q1",
                }
            ]
        )
        temp_storage.save_raw_by_date(df1, "report_rc", "2025-12-31")
        temp_storage.save_raw_by_date(df2, "report_rc", "2026-12-31")

        loader = DataLoader(temp_storage)
        result = loader.load_report_rc()
        assert result is not None and len(result) == 2
        assert result["report_date"].iloc[0] == "20251230"
        assert result["report_date"].iloc[1] == "20260410"

    def test_load_report_rc_no_partitions_returns_none(self, temp_storage):
        """无分区时返回 None (纯分区, 无单文件兜底)。"""
        loader = DataLoader(temp_storage)
        assert loader.load_report_rc() is None

    def test_load_report_rc_rejects_incomplete_identity_schema(self, temp_storage):
        """旧弱身份 schema 必须明确失败，提示强制重下。"""
        old_schema = pd.DataFrame(
            [{"ts_code": "000001.SZ", "report_date": "20251230", "org_name": "a"}]
        )
        temp_storage.save_raw_by_date(old_schema, "report_rc", "2025-12-31")

        with pytest.raises(ValueError, match="report_rc 身份 schema 不完整"):
            DataLoader(temp_storage).load_report_rc()

    def test_load_report_rc_all_na_column_no_future_warning(self, temp_storage):
        """含全 NA 列的分区合并时不触发 concat FutureWarning（告警已屏蔽）。

        背景: report_rc 按年分区存储, 部分分区存在整列全 NA (如 org_name 某年
        全部缺失), 裸 pd.concat 会触发 pandas FutureWarning (DataFrame
        concatenation with empty or all-NA entries is deprecated), 纸面交易
        run 时被黄色告警刷屏; load_report_rc 现与 storage.py 统一模式屏蔽该告警。
        """
        df1 = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20251230",
                    "org_name": "a",
                    "author_name": "分析师甲",
                    "report_title": "2025年研报",
                    "quarter": "2025Q4",
                }
            ]
        )
        # 模拟真实场景: 部分分区整列全 NA (org_name 字段某年全部缺失)
        df2 = pd.DataFrame(
            [
                {
                    "ts_code": "000002.SZ",
                    "report_date": "20260410",
                    "org_name": None,
                    "author_name": "分析师乙",
                    "report_title": "2026年研报",
                    "quarter": "2026Q1",
                }
            ]
        )
        temp_storage.save_raw_by_date(df1, "report_rc", "2025-12-31")
        temp_storage.save_raw_by_date(df2, "report_rc", "2026-12-31")

        loader = DataLoader(temp_storage)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = loader.load_report_rc()

        assert result is not None and len(result) == 2
        concat_warnings = [
            w
            for w in caught
            if issubclass(w.category, FutureWarning)
            and "DataFrame concatenation with empty or all-NA entries" in str(w.message)
        ]
        assert concat_warnings == []
