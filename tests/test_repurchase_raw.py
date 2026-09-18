#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""repurchase（股票回购）raw 层（分页下载 / 去重 / 年分区 / 水位）专项测试。

对照 Phase 0 审计（`docs/repurchase_pit_audit.md`）：单页上限 2000、offset 可翻页、
跨页重复、无自然唯一键（`ts_code+ann_date` 可重复）、`ann_date` 为唯一 PIT 锚点。
"""

from pathlib import Path

import pandas as pd
import pytest

from src.lazybull.data.repurchase_raw import (
    REPURCHASE_PAGE_LIMIT,
    REPURCHASE_RAW_NAME,
    REPURCHASE_RESUME_OVERLAP_DAYS,
    deduplicate_repurchase,
    download_repurchase,
    load_repurchase,
    month_windows,
    repurchase_latest_ann_date,
)
from src.lazybull.data.storage import Storage


class FakePagingClient:
    """模拟 TuShare repurchase：单页上限 2000、按 ann_date 降序、跨页重叠重复行。"""

    def __init__(self, truth: pd.DataFrame, page_limit: int = 2000, overlap_rows: int = 1):
        self.truth = truth
        self.page_limit = page_limit
        self.overlap_rows = overlap_rows
        self.calls = []

    def _query_with_pagination(
        self, api_name: str, page_limit: int = 2000, **kwargs
    ) -> pd.DataFrame:
        self.calls.append((api_name, kwargs))
        start, end = str(kwargs["start_date"]), str(kwargs["end_date"])
        window = self.truth[self.truth["ann_date"].between(start, end)]
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
            offset += page_limit - self.overlap_rows
        if not pages:
            return pd.DataFrame(columns=self.truth.columns)
        return pd.concat(pages, ignore_index=True)


def _truth_df() -> pd.DataFrame:
    """两年 × 两个月 × 每天若干条（多阶段同公告日多行，模拟真实结构）。"""
    rows = []
    for year in ("2023", "2024"):
        for month in ("03", "04"):
            for day in range(1, 6):
                for seq, proc in enumerate(("预案", "股东大会通过", "完成")):
                    rows.append(
                        {
                            "ts_code": f"00000{seq}.SZ",
                            "ann_date": f"{year}{month}{day:02d}",
                            "end_date": f"{year}{month}{day:02d}",
                            "proc": proc,
                            "exp_date": None,
                            "vol": 1_000_000 * (seq + 1),
                            "amount": 1e7 * (seq + 1),
                            "high_limit": 10.0 + seq,
                            "low_limit": None,
                        }
                    )
    return pd.DataFrame(rows)


def test_month_windows_boundaries():
    windows = month_windows("20230115", "20230305")
    assert windows == [("20230115", "20230131"), ("20230201", "20230228"), ("20230301", "20230305")]


def test_deduplicate_drops_invalid_ann_date_and_full_row_duplicates():
    frame = pd.DataFrame(
        [
            {"ts_code": "A", "ann_date": "20240102", "proc": "预案"},
            {"ts_code": "A", "ann_date": "2024-01-02", "proc": "预案"},  # 与上一行归一化后重复
            {"ts_code": "B", "ann_date": None, "proc": "预案"},  # 非法：剔除并告警
            {"ts_code": "B", "ann_date": "20240103", "proc": "预案"},
            {"ts_code": "B", "ann_date": "20240103", "proc": "完成"},  # 同日不同阶段：保留
        ]
    )
    out = deduplicate_repurchase(frame)
    assert len(out) == 3
    assert set(out["ann_date"]) == {"20240102", "20240103"}
    assert not out["ann_date"].isna().any()
    assert out["ann_date"].str.match(r"^\d{8}$").all()


def test_deduplicate_requires_ann_date():
    with pytest.raises(ValueError, match="ann_date"):
        deduplicate_repurchase(pd.DataFrame([{"ts_code": "A"}]))


def test_pagination_fetches_full_dataset_and_dedupes_overlap(tmp_path):
    truth = _truth_df()
    client = FakePagingClient(truth, page_limit=10, overlap_rows=3)  # 强制多页 + 跨页重复
    storage = Storage(str(tmp_path))
    summary = download_repurchase(client, storage, "20230101", "20241231")

    assert summary["rows_new"] == len(truth)  # 分页取全（跨页重复已去重）
    assert summary["rows_total"] == len(truth)
    stored = load_repurchase(storage)
    assert stored is not None and len(stored) == len(truth)
    assert set(stored["ts_code"]) == set(truth["ts_code"])
    # 多阶段同公告日多行必须保留
    same_day = stored[(stored["ts_code"] == "000000.SZ") & (stored["ann_date"] == "20240301")]
    assert len(same_day) == 1  # 同一 (code, ann_date) 下 seq 不同 → ts_code 不同，这里只查一只
    assert stored["proc"].nunique() == 3


def test_year_partitions_and_load_subset(tmp_path):
    truth = _truth_df()
    client = FakePagingClient(truth)
    storage = Storage(str(tmp_path))
    download_repurchase(client, storage, "20230101", "20241231")

    partitions = sorted(storage.list_partitions("raw", REPURCHASE_RAW_NAME))
    assert partitions == ["2023-12-31", "2024-12-31"]
    part_2023 = Path(storage.raw_path) / REPURCHASE_RAW_NAME / "2023-12-31.parquet"
    assert part_2023.exists()
    subset = load_repurchase(storage, years=["2023"])
    assert subset is not None and set(subset["ann_date"].str[:4]) == {"2023"}


def test_incremental_resume_uses_overlap_and_skips_when_covered(tmp_path):
    truth = _truth_df()
    storage = Storage(str(tmp_path))
    client = FakePagingClient(truth)
    download_repurchase(client, storage, "20230101", "20241231")
    assert repurchase_latest_ann_date(storage) == "20240405"

    # 水位已覆盖（end 不晚于水位）⇒ 无需下载
    client.calls.clear()
    summary = download_repurchase(client, storage, "20230101", "20240405")
    assert summary["rows_new"] == 0 and client.calls == []

    # 增量：起点应回拉 overlap 天
    client.calls.clear()
    download_repurchase(client, storage, "20230101", "20250131")
    first_start = client.calls[0][1]["start_date"]
    expected = (
        pd.to_datetime("20240405") - pd.Timedelta(days=REPURCHASE_RESUME_OVERLAP_DAYS)
    ).strftime("%Y%m%d")
    assert first_start == expected


def test_force_full_rebuild_removes_stale_partitions(tmp_path):
    truth = _truth_df()
    storage = Storage(str(tmp_path))
    download_repurchase(FakePagingClient(truth), storage, "20230101", "20241231")
    # 制造一个不存在的旧分区，force 全量重下后应被清理
    stale = Path(storage.raw_path) / REPURCHASE_RAW_NAME / "1999-12-31.parquet"
    pd.DataFrame([{"ts_code": "X", "ann_date": "19990101", "proc": "预案"}]).to_parquet(
        stale, index=False
    )
    download_repurchase(FakePagingClient(truth), storage, "20230101", "20241231", force=True)
    assert not stale.exists()
    assert sorted(storage.list_partitions("raw", REPURCHASE_RAW_NAME)) == [
        "2023-12-31",
        "2024-12-31",
    ]


def test_empty_window_does_not_write_partitions(tmp_path):
    storage = Storage(str(tmp_path))
    empty = _truth_df().iloc[0:0]
    summary = download_repurchase(FakePagingClient(empty), storage, "20250101", "20250131")
    assert summary["rows_new"] == 0 and summary["rows_total"] == 0
    assert storage.list_partitions("raw", REPURCHASE_RAW_NAME) == []


def test_page_limit_constant_matches_audit():
    """单页上限必须是 Phase 0 实测的 2000（写死常量，防退回默认值）。"""
    assert REPURCHASE_PAGE_LIMIT == 2000
