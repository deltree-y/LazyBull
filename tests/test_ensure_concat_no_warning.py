# -*- coding: utf-8 -*-
"""ensure 子包 concat 告警屏蔽与现金流修订刷新并发测试（v0.96.4）。

覆盖：
- _concat_no_warning 屏蔽 empty/all-NA FutureWarning 且数据原样保留（不做剔除）
- 修订刷新并发路径：多季度成功时结果与串行一致、两个水位均推进
- 并发下个别季度失败：对应水位不推进，其余分区正常落盘
"""

import warnings

import pandas as pd

from src.lazybull.features.ensure.concat_utils import _concat_no_warning
from src.lazybull.features.ensure.downloads import (
    _CASHFLOW_REVISION_FULL_REFRESH_WATERMARK,
    _refresh_cashflow_revisions_if_due,
)


def test_concat_no_warning_suppresses_future_warning_and_keeps_data():
    """empty/all-NA 片段合并不触发 FutureWarning，行/列原样保留。"""
    empty_df = pd.DataFrame()
    all_na_df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    normal_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _concat_no_warning([empty_df, all_na_df, normal_df])

    leaked = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert not leaked, f"FutureWarning 未被屏蔽: {[str(w.message) for w in leaked]}"
    # 数据原样保留：不做任何全 NA 行/列剔除
    assert len(result) == 4
    assert list(result.columns) == ["a", "b"]
    assert result["a"].isna().sum() == 2


def test_concat_no_warning_result_matches_plain_concat():
    """屏蔽版与裸 pd.concat 结果完全一致（仅告警行为不同）。"""
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [3], "b": ["z"]})
    expected = pd.concat([df1, df2], ignore_index=True)
    result = _concat_no_warning([df1, df2])
    pd.testing.assert_frame_equal(result, expected)


class _FakeClient:
    """按 period 返回单行现金流数据的最小客户端桩。"""

    def __init__(self, fail_periods=()):
        self.fail_periods = set(fail_periods)
        self.calls = []

    def query(self, api_name, fields="", limit=0, offset=0, **kwargs):
        period = kwargs.get("period")
        self.calls.append(period)
        if period in self.fail_periods:
            raise RuntimeError("模拟 TuShare 限流")
        # 单行（远小于分页上限 6400）即终止翻页
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20260428"],
                "f_ann_date": ["20260705"],
                "end_date": [period],
                "update_flag": [1],
            }
        )


class _Storage:
    """修订刷新测试用最小存储桩（dict 操作 GIL 下并发安全）。"""

    def __init__(self, partitions=None, watermarks=None):
        self.partitions = dict(partitions or {})
        self.watermarks = dict(watermarks or {})

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


def _make_partitions(periods):
    """为每个 period 预置一行旧版本数据。"""
    return {
        period: pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20260428"],
                "f_ann_date": ["20260428"],
                "end_date": [period],
                "update_flag": [0],
            }
        )
        for period in periods
    }


def test_refresh_cashflow_revisions_concurrent_success(monkeypatch):
    """并发刷新：近期 + 历史季度全部成功，两个水位均推进，版本化合并保留双版本。"""
    monkeypatch.setattr(
        "src.lazybull.features.ensure.downloads.get_tushare_settings",
        lambda: {"download_concurrency": 4},
    )
    # 1 个历史分区 + 8 个近期季度（2026-08-28 窗口）共 9 个 period
    partitions = _make_partitions(["20231231"])
    storage = _Storage(partitions=partitions)
    client = _FakeClient()

    _refresh_cashflow_revisions_if_due(client, storage, "20260828")

    # 8 个近期季度 + 1 个历史分区，全部查询成功
    assert len(client.calls) == 9
    # 历史分区有预置行：旧版本行 + 修订行按版本键共存
    assert len(storage.partitions) == 9
    assert len(storage.partitions["20231231"]) == 2
    # 近期季度无预置分区：仅修订行
    assert len(storage.partitions["20260630"]) == 1
    assert storage.load_sync_watermark("cashflow_revision_refresh") == "20260828"
    assert storage.load_sync_watermark(_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK) == "20260828"


def test_refresh_cashflow_revisions_concurrent_partial_failure(monkeypatch):
    """并发下个别近期季度失败：日水位不推进，历史水位正常推进，失败季度不落盘。"""
    monkeypatch.setattr(
        "src.lazybull.features.ensure.downloads.get_tushare_settings",
        lambda: {"download_concurrency": 4},
    )
    # 20260331 是近期季度之一但无预置分区，其失败不会拖住历史水位
    partitions = _make_partitions(["20231231", "20260630"])
    storage = _Storage(partitions=partitions)
    client = _FakeClient(fail_periods={"20260331"})

    _refresh_cashflow_revisions_if_due(client, storage, "20260828")

    # 近期季度有失败 -> 日水位不推进
    assert storage.load_sync_watermark("cashflow_revision_refresh") is None
    # 历史分区全部成功 -> 全历史水位推进
    assert storage.load_sync_watermark(_CASHFLOW_REVISION_FULL_REFRESH_WATERMARK) == "20260828"
    # 失败季度不落盘；成功分区正常合并
    assert "20260331" not in storage.partitions
    assert len(storage.partitions["20231231"]) == 2
    assert len(storage.partitions["20260630"]) == 2
