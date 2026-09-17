#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stk_holdertrade raw 层（分页下载 / 去重 / 年分区 / 水位）专项测试。"""

from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.data.holdertrade_raw import (
    HOLDERTRADE_RAW_NAME,
    HOLDERTRADE_RESUME_OVERLAP_DAYS,
    deduplicate_holdertrade,
    download_holdertrade,
    holdertrade_latest_ann_date,
    load_holdertrade,
    month_windows,
)
from src.lazybull.data.storage import Storage


class FakePagingClient:
    """模拟 TuShare：单页上限 3000 行、按 offset 翻页、跨页重复、按时间降序。"""

    def __init__(self, truth: pd.DataFrame, page_limit: int = 3000, overlap_rows: int = 1):
        self.truth = truth
        self.page_limit = page_limit
        self.overlap_rows = overlap_rows  # 模拟"跨页边界重复行"
        self.calls = []

    def _query_with_pagination(
        self, api_name: str, page_limit: int = 3000, **kwargs
    ) -> pd.DataFrame:
        self.calls.append((api_name, kwargs))
        start, end = str(kwargs["start_date"]), str(kwargs["end_date"])
        window = self.truth[self.truth["ann_date"].between(start, end)]
        # 接口默认按 ann_date 降序返回（Phase 0 实测）
        ordered = window.sort_values("ann_date", ascending=False).reset_index(drop=True)
        pages = []
        offset = 0
        while True:
            page = ordered.iloc[offset : offset + page_limit]
            if len(page) == 0:
                break
            pages.append(page)
            if len(page) < page_limit:
                break
            offset += page_limit - self.overlap_rows  # 下一页与上一页重叠 overlap_rows 行
        if not pages:
            return pd.DataFrame(columns=self.truth.columns)
        return pd.concat(pages, ignore_index=True)


def _truth_df() -> pd.DataFrame:
    rows = []
    for idx, year in enumerate(("2023", "2024")):
        for month in ("03", "04"):
            for day in range(1, 6):
                for seq in range(3):
                    rows.append(
                        {
                            "ts_code": f"00000{seq}.SZ",
                            "ann_date": f"{year}{month}{day:02d}",
                            "holder_name": f"H{seq}",
                            "holder_type": "C",
                            "in_de": "IN" if seq % 2 else "DE",
                            "change_vol": float(idx * 100 + day + seq),
                            "total_share": 1_000_000.0,
                        }
                    )
    # 源内整行重复（Phase 0 实测 24.7%）
    dup = rows[0].copy()
    rows.append(dup)
    return pd.DataFrame(rows)


def test_month_windows_split_and_clip():
    assert month_windows("20240115", "20240305") == [
        ("20240115", "20240131"),
        ("20240201", "20240229"),
        ("20240301", "20240305"),
    ]
    with pytest.raises(ValueError, match="晚于结束日期"):
        month_windows("20240305", "20240101")
    with pytest.raises(ValueError, match="日期格式非法"):
        month_windows("2024-1-1", "20240305")


def test_deduplicate_drops_invalid_ann_date_and_full_row_dups():
    df = pd.DataFrame(
        {
            "ts_code": ["a", "a", "a", "b"],
            "ann_date": ["2024-03-01", "20240301", "20240302", "20240230"],
            "change_vol": [1.0, 1.0, 2.0, 3.0],
        }
    )
    out = deduplicate_holdertrade(df)
    # 20240230 非法（二月无 30 日）→ 剔除；两条同值 20240301 → 去重
    assert out["ann_date"].tolist() == ["20240301", "20240302"]
    assert len(out) == 2


def test_download_pagination_recovers_all_rows_and_partitions_by_year(tmp_path):
    truth = _truth_df()
    client = FakePagingClient(truth, page_limit=5, overlap_rows=1)
    storage = Storage(tmp_path)

    # 逐月独立拉取（无截断）作为真值参考
    summary = download_holdertrade(client, storage, "20230301", "20240405")

    expected = deduplicate_holdertrade(truth)
    loaded = load_holdertrade(storage)
    assert loaded is not None
    assert len(loaded) == len(expected)
    assert summary["rows_total"] == len(expected)

    partitions = sorted(storage.list_partitions("raw", HOLDERTRADE_RAW_NAME))
    assert partitions == ["2023-12-31", "2024-12-31"]
    year_2023 = storage.load_raw_by_date(HOLDERTRADE_RAW_NAME, "2023-12-31")
    assert set(year_2023["ann_date"].astype(str).str[:4]) == {"2023"}
    assert holdertrade_latest_ann_date(storage) == "20240405"


def test_download_is_incremental_with_overlap(tmp_path):
    truth = _truth_df()
    client = FakePagingClient(truth, page_limit=5, overlap_rows=1)
    storage = Storage(tmp_path)

    download_holdertrade(client, storage, "20230301", "20240405")
    first_calls = len(client.calls)

    # 第二次同区间：水位已覆盖 → 不再发请求
    summary = download_holdertrade(client, storage, "20230301", "20240405")
    assert len(client.calls) == first_calls
    assert summary["rows_new"] == 0
    assert summary["resumed_from"] is None

    # 追加新月份：只拉水位回拉窗口之后的窗口
    extra = pd.DataFrame(
        {
            "ts_code": ["000009.SZ"],
            "ann_date": ["20240510"],
            "holder_name": ["H9"],
            "holder_type": "P",
            "in_de": "IN",
            "change_vol": [123.0],
            "total_share": [1_000_000.0],
        }
    )
    client.truth = pd.concat([truth, extra], ignore_index=True)
    summary = download_holdertrade(client, storage, "20230301", "20240531")
    assert summary["rows_new"] >= 1
    assert summary["resumed_from"] == "20240402"  # 20240405 往前回拉 3 天
    windows_requested = [call[1] for call in client.calls[first_calls:]]
    assert all(window["start_date"] >= "20240401" for window in windows_requested)

    merged = load_holdertrade(storage)
    assert merged is not None
    assert "20240510" in set(merged["ann_date"].astype(str))
    # 回拉窗口内的旧数据不得丢失
    assert sum(merged["ann_date"].astype(str) == "20240403") == 3


def test_download_force_refetches_from_requested_start(tmp_path):
    truth = _truth_df()
    client = FakePagingClient(truth, page_limit=5, overlap_rows=1)
    storage = Storage(tmp_path)
    download_holdertrade(client, storage, "20230301", "20240405")
    calls_before = len(client.calls)

    summary = download_holdertrade(client, storage, "20230301", "20240405", force=True)
    assert summary["resumed_from"] == "20230301"
    assert len(client.calls) > calls_before
    assert all(call[1]["start_date"] >= "20230301" for call in client.calls[calls_before:])


def test_download_without_data_returns_zero_and_does_not_create_partitions(tmp_path):
    client = FakePagingClient(pd.DataFrame(columns=["ts_code", "ann_date", "change_vol"]))
    storage = Storage(tmp_path)
    summary = download_holdertrade(client, storage, "20240101", "20240131")
    assert summary["rows_total"] == 0
    assert storage.list_partitions("raw", HOLDERTRADE_RAW_NAME) == []
    assert holdertrade_latest_ann_date(storage) is None


def test_save_removes_stale_partitions(tmp_path):
    truth = _truth_df()
    client = FakePagingClient(truth)
    storage = Storage(tmp_path)
    download_holdertrade(client, storage, "20230301", "20240405")
    assert len(storage.list_partitions("raw", HOLDERTRADE_RAW_NAME)) == 2

    # force + 仅 2024 数据 → 2023 分区应被移除
    only_2024 = truth[truth["ann_date"].astype(str).str.startswith("2024")]
    client2 = FakePagingClient(only_2024)
    download_holdertrade(client2, storage, "20240101", "20240405", force=True)
    assert storage.list_partitions("raw", HOLDERTRADE_RAW_NAME) == ["2024-12-31"]


def test_resume_overlap_constant_documented():
    # 回拉天数必须为正（防同日补录/更正），且不应大到让每日增量成本失控
    assert 1 <= HOLDERTRADE_RESUME_OVERLAP_DAYS <= 7


def test_load_holdertrade_missing_returns_none(tmp_path):
    storage = Storage(tmp_path)
    assert load_holdertrade(storage) is None
    assert holdertrade_latest_ann_date(storage) is None
    assert not Path(tmp_path / "raw" / HOLDERTRADE_RAW_NAME).exists()
