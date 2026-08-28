# -*- coding: utf-8 -*-
"""现金流质量因子全链路隐患修复专项测试（v0.96.3 / schema v3）。

覆盖：
- 季度列表推导（修订刷新窗口）
- 批量下载分页粒度按接口映射解析（cashflow_vip=6400）
- 修订刷新：版本化合并 + 水位控制
- 版本化去重键保留多版本
- 汇总层现金流实际入模列提取
"""

import pandas as pd
import pytest

from src.lazybull.features.ensure.bulk import _API_PAGE_LIMITS, _bulk_download_by_period
from src.lazybull.features.ensure.downloads import (
    _CASHFLOW_REVISION_FULL_REFRESH_WATERMARK,
    _CASHFLOW_VERSION_DEDUP_COLS,
    _recent_quarter_periods,
    _refresh_cashflow_revisions_if_due,
    _try_download_cashflow,
)
from src.lazybull.features.ensure.incremental import _drop_duplicates_keep_updated
from src.lazybull.ml.walk_forward.summary import _live_cashflow_quality_cols


def test_recent_quarter_periods_basic():
    """最近已结束季度推导（含跨年与季度边界）。"""
    # 2026-08-28：最近已结束 = 2026Q2
    assert _recent_quarter_periods("20260828", 4) == [
        "20260630",
        "20260331",
        "20251231",
        "20250930",
    ]
    # 1 月属于上一年 Q4
    assert _recent_quarter_periods("20260115", 2) == ["20251231", "20250930"]
    # 4 月初：最近已结束 = 本年 Q1
    assert _recent_quarter_periods("20260401", 1) == ["20260331"]


def test_api_page_limits_map_covers_cashflow_cap():
    """cashflow_vip 单次上限 6400 必须登记，避免首屏误判截断。"""
    assert _API_PAGE_LIMITS["cashflow_vip"] == 6400
    assert _API_PAGE_LIMITS["fina_indicator_vip"] == 12000
    assert _API_PAGE_LIMITS["fund_portfolio"] == 8000


def test_cashflow_ann_date_incremental_query_uses_6400_pagination(monkeypatch):
    """公告日增量同样必须分页，避免峰值公告日超过 6400 行时静默截断。"""
    existing = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20260827"],
            "f_ann_date": ["20260827"],
            "end_date": ["20260630"],
        }
    )
    client = _FakeQueryClient()
    storage = _RefreshStorage(partitions={"20260630": existing})

    monkeypatch.setattr("src.lazybull.features.ensure.downloads._MIN_CASHFLOW_RECORDS", 1)
    monkeypatch.setattr(
        "src.lazybull.features.ensure.downloads._load_all_partitions",
        lambda *_args, **_kwargs: existing,
    )

    def _run_incremental(**kwargs):
        kwargs["fetch_by_date"]("20260828")
        return existing

    monkeypatch.setattr(
        "src.lazybull.features.ensure.downloads._incremental_catchup_by_calendar_date",
        _run_incremental,
    )
    monkeypatch.setattr(
        "src.lazybull.features.ensure.downloads.DataLoader.load_cashflow",
        lambda *_args, **_kwargs: existing,
    )

    _try_download_cashflow(client, storage, "20260828")

    assert client.calls
    assert {call["limit"] for call in client.calls} == {6400}


class _FakeQueryClient:
    """捕获 _query_with_pagination 调用参数的最小客户端。"""

    def __init__(self, rows_per_page=0):
        self.calls = []
        self.rows_per_page = rows_per_page

    def query(self, api_name, fields="", limit=0, offset=0, **kwargs):
        self.calls.append({"api": api_name, "limit": limit, "offset": offset, **kwargs})
        # 返回空页即可终止翻页
        return pd.DataFrame()


class _RefreshStorage:
    """修订刷新测试用最小存储桩。"""

    def __init__(self, watermarks=None, partitions=None):
        self.watermarks = dict(watermarks or {})
        self.partitions = dict(partitions or {})

    def load_sync_watermark(self, name):
        return self.watermarks.get(name)

    def save_sync_watermark(self, name, value):
        self.watermarks[name] = value

    def load_raw_by_date(self, dataset_name, period):
        return self.partitions.get(period)

    def save_raw_by_date(self, df, dataset_name, period):
        self.partitions[period] = df

    def list_partitions(self, kind, dataset_name):
        return sorted(self.partitions.keys())


def test_bulk_download_resolves_cashflow_page_limit_from_map(monkeypatch):
    """cashflow_vip 批量下载必须按 6400 分页（默认 50000 会静默截断）。"""
    client = _FakeQueryClient()
    storage = _RefreshStorage()

    monkeypatch.setattr(
        "src.lazybull.features.ensure.bulk._query_with_pagination",
        lambda c, api, page_limit=50000, fields=None, max_pages=1000, **kw: (
            client.query(api, fields=fields, limit=page_limit, **kw) or pd.DataFrame()
        ),
    )

    _bulk_download_by_period(
        client,
        storage,
        dataset_name="cashflow",
        api_name="cashflow_vip",
        dedup_cols=_CASHFLOW_VERSION_DEDUP_COLS,
        partition_by_period=True,
        start_year=2025,
    )

    limits = {c["limit"] for c in client.calls}
    assert limits == {6400}


def test_refresh_cashflow_revisions_merges_versions_and_gates_by_watermark():
    """修订刷新：同报告期多版本共存，水位当日只跑一次。"""
    client = _FakeQueryClient()
    # 预置一个近期季度分区（含旧版本行；刷新窗口为最近 8 个季度）
    existing = pd.DataFrame(
        {
            "ts_code": ["002111.SZ"],
            "ann_date": ["20260428"],
            "f_ann_date": ["20260428"],
            "end_date": ["20260630"],
            "n_cashflow_act": [804189271.09],
            "update_flag": [0],
        }
    )
    storage = _RefreshStorage(partitions={"20260630": existing})

    revision = pd.DataFrame(
        {
            "ts_code": ["002111.SZ"],
            "ann_date": ["20260428"],
            "f_ann_date": ["20260705"],
            "end_date": ["20260630"],
            "n_cashflow_act": [784189271.09],
            "update_flag": [1],
        }
    )

    def _fake_query(api_name, fields="", limit=0, offset=0, **kwargs):
        if kwargs.get("period") == "20260630":
            return revision
        return pd.DataFrame()

    client.query = _fake_query  # type: ignore[method-assign]

    _refresh_cashflow_revisions_if_due(client, storage, "20260828")

    merged = storage.partitions["20260630"]
    # 版本化去重键下两版本共存（旧版 + 修订版）
    assert len(merged) == 2
    assert set(merged["f_ann_date"]) == {"20260428", "20260705"}
    # 水位推进到当日
    assert storage.load_sync_watermark("cashflow_revision_refresh") == "20260828"

    # 同一天第二次调用：水位命中，不再请求
    client2 = _FakeQueryClient()
    _refresh_cashflow_revisions_if_due(client2, storage, "20260828")
    assert client2.calls == []


def test_drop_duplicates_keep_updated_preserves_versions_with_f_ann_date_key():
    """版本化键 (ts_code, end_date, f_ann_date) 下多版本共存。"""
    df = pd.DataFrame(
        {
            "ts_code": ["002111.SZ", "002111.SZ"],
            "ann_date": ["20210420", "20210420"],
            "f_ann_date": ["20210420", "20230705"],
            "end_date": ["20201231", "20201231"],
            "n_cashflow_act": [804189271.09, 784189271.09],
            "update_flag": [1, 0],
        }
    )
    result = _drop_duplicates_keep_updated(df, _CASHFLOW_VERSION_DEDUP_COLS)
    assert len(result) == 2


def test_drop_duplicates_same_version_prefers_official_latest_flag():
    """同一版本键冲突时，必须稳定选择 TuShare 官方 update_flag=1 最新行。"""
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231"],
            "f_ann_date": ["20240430", "20240430"],
            "update_flag": [1, 0],
            "free_cashflow": [1.0e7, 2.0e7],
        }
    )

    forward = _drop_duplicates_keep_updated(df, _CASHFLOW_VERSION_DEDUP_COLS)
    reverse = _drop_duplicates_keep_updated(
        df.iloc[::-1].reset_index(drop=True), _CASHFLOW_VERSION_DEDUP_COLS
    )

    assert forward.loc[0, "free_cashflow"] == pytest.approx(1.0e7)
    assert reverse.loc[0, "free_cashflow"] == pytest.approx(1.0e7)


def test_drop_duplicates_same_version_and_flag_is_order_independent():
    """官方标志无法区分冲突行时，结果也不得依赖接口或 parquet 行序。"""
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231"],
            "f_ann_date": ["20240430", "20240430"],
            "update_flag": [1, 1],
            "free_cashflow": [1.0e7, 2.0e7],
        }
    )

    forward = _drop_duplicates_keep_updated(df, _CASHFLOW_VERSION_DEDUP_COLS)
    reverse = _drop_duplicates_keep_updated(
        df.iloc[::-1].reset_index(drop=True), _CASHFLOW_VERSION_DEDUP_COLS
    )

    pd.testing.assert_frame_equal(forward, reverse)


def test_drop_duplicates_cashflow_without_update_flag_is_order_independent():
    """现金流旧数据缺少官方标志时，版本键冲突仍需稳定决胜。"""
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231"],
            "f_ann_date": ["20240430", "20240430"],
            "free_cashflow": [1.0e7, 2.0e7],
        }
    )

    forward = _drop_duplicates_keep_updated(df, _CASHFLOW_VERSION_DEDUP_COLS)
    reverse = _drop_duplicates_keep_updated(
        df.iloc[::-1].reset_index(drop=True), _CASHFLOW_VERSION_DEDUP_COLS
    )

    pd.testing.assert_frame_equal(forward, reverse)


def test_refresh_cashflow_revisions_first_run_covers_old_partitions():
    """首次版本化迁移必须重查旧分区，不能只刷新最近 8 个季度。"""
    old = pd.DataFrame(
        {
            "ts_code": ["002111.SZ"],
            "ann_date": ["20210420"],
            "f_ann_date": ["20210420"],
            "end_date": ["20201231"],
            "update_flag": [1],
        }
    )
    storage = _RefreshStorage(partitions={"20201231": old})
    client = _FakeQueryClient()

    def _query(api_name, fields="", limit=0, offset=0, **kwargs):
        client.calls.append({"api": api_name, "limit": limit, "offset": offset, **kwargs})
        if kwargs.get("period") == "20201231":
            return old
        return pd.DataFrame()

    client.query = _query  # type: ignore[method-assign]
    _refresh_cashflow_revisions_if_due(client, storage, "20260828")

    queried_periods = {call.get("period") for call in client.calls}
    assert "20201231" in queried_periods
    assert storage.load_sync_watermark(_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK) == "20260828"


def test_refresh_cashflow_revisions_skips_old_partitions_before_full_interval():
    """全历史水位未到 90 天时，只执行近期每日刷新。"""
    old = pd.DataFrame(
        {
            "ts_code": ["002111.SZ"],
            "ann_date": ["20210420"],
            "f_ann_date": ["20210420"],
            "end_date": ["20201231"],
        }
    )
    storage = _RefreshStorage(
        watermarks={_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK: "20260801"},
        partitions={"20201231": old},
    )
    client = _FakeQueryClient()

    _refresh_cashflow_revisions_if_due(client, storage, "20260828")

    queried_periods = {call.get("period") for call in client.calls}
    assert "20201231" not in queried_periods


def test_refresh_cashflow_revisions_empty_existing_partition_does_not_advance_full_watermark():
    """历史分区已有数据但接口返回空时必须重试，不能把空响应提交为完整迁移。"""
    old = pd.DataFrame(
        {
            "ts_code": ["002111.SZ"],
            "ann_date": ["20210420"],
            "f_ann_date": ["20210420"],
            "end_date": ["20201231"],
        }
    )
    storage = _RefreshStorage(partitions={"20201231": old})

    _refresh_cashflow_revisions_if_due(_FakeQueryClient(), storage, "20260828")

    assert storage.load_sync_watermark(_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK) is None


@pytest.mark.parametrize("error_type", [AttributeError, NotImplementedError])
def test_refresh_cashflow_revisions_tolerates_unsupported_partition_listing(error_type):
    """兼容存储无法枚举分区时继续近期刷新，但不得提交全历史水位。"""
    storage = _RefreshStorage()

    def _raise_unsupported(*_args, **_kwargs):
        raise error_type("不支持分区枚举")

    storage.list_partitions = _raise_unsupported  # type: ignore[method-assign]

    _refresh_cashflow_revisions_if_due(_FakeQueryClient(), storage, "20260828")

    assert storage.load_sync_watermark("cashflow_revision_refresh") == "20260828"
    assert storage.load_sync_watermark(_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK) is None


def test_download_cashflow_skip_does_not_fake_full_refresh_watermark(monkeypatch):
    """已有迁移水位且本次未强制查询时，不得仅因分区被跳过就刷新水位。"""
    from scripts.raw_download.alt import download_cashflow

    storage = _RefreshStorage(
        watermarks={_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK: "20260101"},
        partitions={"20201231": pd.DataFrame({"ts_code": ["000001.SZ"]})},
    )
    captured = {}

    def _skip_download(*_args, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr("scripts.raw_download.alt.download_by_period", _skip_download)

    download_cashflow(
        _FakeQueryClient(),
        storage,
        start_date="20200101",
        end_date="20261231",
    )

    assert captured["force"] is False
    assert storage.load_sync_watermark(_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK) == "20260101"


def test_live_cashflow_quality_cols_extracts_post_gate_columns():
    """汇总层按门禁后 feature_columns 提取实际入模列（含 _sz 变体）。"""
    result = {
        "feature_columns": [
            "zscore_ocf_to_revenue",
            "zscore_ocf_to_revenue_sz",
            "zscore_fcf_yield",
            "zscore_bp",  # 无关列
            "cashflow_freshness_days",
        ]
    }
    live = _live_cashflow_quality_cols(result)
    assert "zscore_ocf_to_revenue" in live
    assert "zscore_ocf_to_revenue_sz" in live
    assert "zscore_fcf_yield" in live
    assert "cashflow_freshness_days" in live
    assert "zscore_bp" not in live
    assert "zscore_capex_to_ocf" not in live  # 被门禁移除的列不出现


def test_live_cashflow_quality_cols_empty_when_disabled():
    """未启用现金流因子时实际入模列为空。"""
    assert _live_cashflow_quality_cols({"feature_columns": ["zscore_bp"]}) == ""


def test_recent_quarter_periods_count_exact():
    assert len(_recent_quarter_periods("20260828", 8)) == 8
    assert _recent_quarter_periods("20260828", 8)[-1] == "20240930"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
