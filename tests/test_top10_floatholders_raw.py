#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""top10_floatholders（十大流通股东）raw 层（分页下载 / 去重 / 年分区 / 水位）专项测试。

对照 Phase 0 审计（`docs/top10_floatholders_pit_audit.md`）：单页上限 6000（`limit=10000` 仍回 6000）、
offset 可翻页、跨页重复 0、唯一键 `(ts_code, end_date, ann_date, holder_name)`、
`ann_date` 为 PIT 锚点（缺失 0%）、按 `period`（报告期）批量拉取、增量回拉最近 2 个已存报告期。
"""

import pandas as pd
import pytest

from src.lazybull.data.storage import Storage
from src.lazybull.data.top10_floatholders_raw import (
    TOP10FH_PAGE_LIMIT,
    TOP10FH_RAW_NAME,
    TOP10FH_RESUME_OVERLAP_PERIODS,
    deduplicate_top10fh,
    download_top10fh,
    load_top10fh,
    quarter_ends,
    resolve_incremental_periods,
    save_top10fh_by_year,
    top10fh_latest_end_date,
)


class FakePeriodClient:
    """模拟 TuShare top10_floatholders：按报告期批量、**单页静默上限**、触顶页继续翻。

    真实接口对 `limit` 有硬上限（实测 `limit=10000` 仍回 6000），因此这里取
    `min(请求 page_limit, 接口上限)` —— 用来锁定"请求再大也按上限分页"的语义。
    """

    def __init__(self, truth: pd.DataFrame, page_limit: int = 6000, overlap_rows: int = 0):
        self.truth = truth
        self.api_page_limit = page_limit
        self.overlap_rows = overlap_rows
        self.calls = []
        #: 每次调用的实际页数（period, 页数）——用于验证“触顶页继续翻”
        self.page_log = []

    def _query_with_pagination(
        self, api_name: str, page_limit: int = 6000, **kwargs
    ) -> pd.DataFrame:
        self.calls.append((api_name, dict(kwargs)))
        effective = min(int(page_limit), self.api_page_limit)
        period = str(kwargs["period"])
        window = self.truth[self.truth["end_date"] == period].reset_index(drop=True)
        pages = []
        offset = 0
        while True:
            page = window.iloc[offset : offset + effective]
            if len(page) == 0:
                break
            pages.append(page)
            if len(page) < effective:
                break
            offset += effective - self.overlap_rows
        if not pages:
            self.page_log.append((period, 0))
            return pd.DataFrame(columns=self.truth.columns)
        self.page_log.append((period, len(pages)))
        return pd.concat(pages, ignore_index=True)


def _truth_df(n_stocks_per_period: int = 6) -> pd.DataFrame:
    """三期（两个年终 + 一个中报）× 每股 10 行股东明细。"""
    rows = []
    for period, ann in (
        ("20231231", "20240426"),
        ("20240630", "20240830"),
        ("20241231", "20250426"),
    ):
        for idx in range(n_stocks_per_period):
            code = f"0000{idx:02d}.SZ"
            for rank in range(10):
                rows.append(
                    {
                        "ts_code": code,
                        "ann_date": ann,
                        "end_date": period,
                        "holder_name": f"股东{rank}",
                        "hold_amount": float(10_000_000 - rank * 100_000),
                        "hold_ratio": 10.0 - rank,
                        "hold_float_ratio": 12.0 - rank,
                        "hold_change": None if rank == 9 else float(rank * 1000),
                        "holder_type": "自然人" if rank % 2 else "社保基金、社保机构",
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 报告期枚举


def test_quarter_ends_enumerates_period_dates():
    assert quarter_ends("20230101", "20231231") == ["20230331", "20230630", "20230930", "20231231"]
    assert quarter_ends("20231231", "20231231") == ["20231231"]
    assert quarter_ends("20231201", "20240331") == ["20231231", "20240331"]
    assert quarter_ends("20230102", "20230331") == ["20230331"]
    assert quarter_ends("20230102", "20230330") == []


def test_quarter_ends_rejects_inverted_range():
    with pytest.raises(ValueError, match="开始日期晚于结束日期"):
        quarter_ends("20240101", "20230101")


# ---------------------------------------------------------------- 去重与规范化


def test_deduplicate_normalizes_dates_and_drops_invalid():
    raw = _truth_df(n_stocks_per_period=1)
    extra = pd.DataFrame(
        [
            {
                "ts_code": "000099.SZ",
                "ann_date": None,  # 非法：剔除
                "end_date": "20231231",
                "holder_name": "X",
                "hold_amount": 1.0,
                "hold_ratio": 1.0,
                "hold_float_ratio": 1.0,
                "hold_change": 0.0,
                "holder_type": "自然人",
            },
            {
                "ts_code": "000098.SZ",
                "ann_date": "2024-04-26",  # 带横线：归一化
                "end_date": "2024-06-30",
                "holder_name": "Y",
                "hold_amount": 1.0,
                "hold_ratio": 1.0,
                "hold_float_ratio": 1.0,
                "hold_change": 0.0,
                "holder_type": "自然人",
            },
        ]
    )
    work = deduplicate_top10fh(pd.concat([raw, extra], ignore_index=True))
    assert "000099.SZ" not in set(work["ts_code"])
    norm = work[work["ts_code"] == "000098.SZ"].iloc[0]
    assert norm["ann_date"] == "20240426" and norm["end_date"] == "20240630"


def test_deduplicate_removes_only_full_row_duplicates():
    raw = _truth_df(n_stocks_per_period=1)
    doubled = pd.concat([raw, raw], ignore_index=True)
    assert len(deduplicate_top10fh(doubled)) == len(raw)
    # 同一 (ts_code, end_date) 的多行股东明细必须保留
    work = deduplicate_top10fh(raw)
    assert (work.groupby(["ts_code", "end_date"]).size() == 10).all()


def test_deduplicate_requires_date_columns():
    with pytest.raises(ValueError, match="缺少 end_date"):
        deduplicate_top10fh(pd.DataFrame({"ts_code": ["000001.SZ"], "ann_date": ["20240426"]}))


# ---------------------------------------------------------------- 分页读满


def test_download_reads_pages_until_empty(tmp_path):
    """单页 6000 上限：触顶页必须继续翻（12 只 × 10 行 = 120 行，用 page_limit=50 模拟触顶）。"""
    truth = _truth_df(n_stocks_per_period=12)  # 每期 120 行
    storage = Storage(root_path=str(tmp_path))
    client = FakePeriodClient(truth, page_limit=50)

    # 区间含 3 个报告期（20231231 / 20240331 / 20240630），其中两期有数据
    summary = download_top10fh(client, storage, "20231201", "20240630")
    assert summary["periods"] == 3
    assert summary["rows_new"] == len(truth[truth["end_date"] <= "20240630"])
    assert summary["latest_end_date"] == "20240630"
    # 每期 120 行 / 50 行一页 ⇒ 3 页（50/50/20）
    assert [n for p, n in client.page_log if p == "20231231"] == [3]
    # 无数据报告期也要被请求到（翻页读到空）
    assert any(c[1]["period"] == "20240331" for c in client.calls)


def test_download_merges_cross_page_duplicates(tmp_path):
    """跨页重叠行必须被全字段去重（真实数据集实测 0 重复，此处锁定去重语义）。"""
    truth = _truth_df(n_stocks_per_period=12)
    storage = Storage(root_path=str(tmp_path))
    client = FakePeriodClient(truth, page_limit=50, overlap_rows=5)

    summary = download_top10fh(client, storage, "20231201", "20231231")
    assert summary["rows_new"] == len(truth[truth["end_date"] == "20231231"])


# ---------------------------------------------------------------- 年分区与加载


def test_save_by_year_and_load_subset(tmp_path):
    storage = Storage(root_path=str(tmp_path))
    truth = _truth_df()
    save_top10fh_by_year(storage, deduplicate_top10fh(truth))

    partitions = sorted(storage.list_partitions("raw", TOP10FH_RAW_NAME))
    assert partitions == ["2023-12-31", "2024-12-31"]

    only_2023 = load_top10fh(storage, years=["2023"])
    assert only_2023 is not None
    assert set(only_2023["end_date"].unique()) == {"20231231"}
    assert top10fh_latest_end_date(storage) == "20241231"


def test_save_by_year_removes_stale_partition(tmp_path):
    storage = Storage(root_path=str(tmp_path))
    save_top10fh_by_year(storage, deduplicate_top10fh(_truth_df()))
    # 只保留 2023 年数据 ⇒ 2024 分区应被移除
    only_2023 = deduplicate_top10fh(_truth_df())
    only_2023 = only_2023[only_2023["end_date"].str.startswith("2023")]
    save_top10fh_by_year(storage, only_2023)
    assert sorted(storage.list_partitions("raw", TOP10FH_RAW_NAME)) == ["2023-12-31"]


def test_save_by_year_requires_normalized_end_date(tmp_path):
    storage = Storage(root_path=str(tmp_path))
    with pytest.raises(ValueError, match="年份非法"):
        save_top10fh_by_year(storage, pd.DataFrame({"end_date": ["abcd1234"]}))


# ---------------------------------------------------------------- 水位与增量


def test_resolve_incremental_periods_pulls_overlap_and_new(tmp_path):
    """存量报告期 {20231231,20240630,20241231} ⇒ 回拉最近 2 期 + 补齐区间内全部缺口期。"""
    storage = Storage(root_path=str(tmp_path))
    save_top10fh_by_year(storage, deduplicate_top10fh(_truth_df()))

    periods, floor = resolve_incremental_periods(storage, "20230101", "20241231")
    assert floor == "20240630"
    # 已存 3 期（20231231/20240630/20241231）⇒ 回拉最近 2 期 + 补齐 5 个缺口期，共 7 期
    assert periods == [
        "20230331",
        "20230630",
        "20230930",
        "20240331",
        "20240630",
        "20240930",
        "20241231",
    ]


def test_resolve_incremental_periods_fills_history_gap(tmp_path):
    """区间内无存量数据 ⇒ 全部按缺口拉取（本数据集不许留下永久空洞）。"""
    storage = Storage(root_path=str(tmp_path))
    save_top10fh_by_year(storage, deduplicate_top10fh(_truth_df()))
    periods, floor = resolve_incremental_periods(storage, "20200101", "20200630")
    assert periods == ["20200331", "20200630"] and floor is None


def test_resolve_incremental_periods_force_pulls_all(tmp_path):
    storage = Storage(root_path=str(tmp_path))
    save_top10fh_by_year(storage, deduplicate_top10fh(_truth_df()))
    periods, floor = resolve_incremental_periods(storage, "20231231", "20240630", force=True)
    assert periods == ["20231231", "20240331", "20240630"] and floor is None


def test_download_is_idempotent_on_second_call(tmp_path):
    truth = _truth_df()
    storage = Storage(root_path=str(tmp_path))
    client = FakePeriodClient(truth)

    first = download_top10fh(client, storage, "20231201", "20241231")
    calls_after_first = len(client.calls)
    second = download_top10fh(client, storage, "20231201", "20241231")

    assert second["latest_end_date"] == first["latest_end_date"]
    assert second["rows_total"] == first["rows_total"]
    # 第二次仍会回拉最近 2 期（披露窗口内补充），但总行数不因重复拉取而膨胀
    assert len(client.calls) >= calls_after_first
    assert load_top10fh(storage) is not None
    assert len(load_top10fh(storage)) == first["rows_total"]


def test_resume_overlap_constant_is_registered():
    """审计 §5.1 登记：回拉最近 2 个已存报告期；单页 6000。"""
    assert TOP10FH_RESUME_OVERLAP_PERIODS == 2
    assert TOP10FH_PAGE_LIMIT == 6000
