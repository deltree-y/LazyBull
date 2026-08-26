# -*- coding: utf-8 -*-
"""report_rc 单次查询超限 (100000 条) 自适应二分分片回归测试。

背景: TuShare report_rc 接口对"一次查询 (start_date/end_date + offset 翻页)"
的总行数上限为 100000 条 (offset 上限 100000)。超过后继续翻页返回
"查询数据失败, 请确认参数！" (实测 2009 年在 offset=102000 失败, 2020/2023
等年份约 20~30 万条同样触发)。download_report_rc 现通过
_query_report_rc_adaptive 在整段查询失败时自动二分日期范围重试。

覆盖:
- _query_report_rc_adaptive 整段成功: 直接返回, 不二分
- _query_report_rc_adaptive 整段失败: 二分左右段并合并返回
- _query_report_rc_adaptive 二分到 max_depth 仍失败: 抛 RuntimeError
- download_report_rc 超限年份自动二分下载成功, 无错误记录
- 日期辅助函数 _mid_date_str / _next_date_str
"""

import warnings
from datetime import datetime

import pandas as pd
import pytest

import scripts.raw_download.core as raw_core
from scripts.raw_download import alt as raw_download_alt
from scripts.raw_download.alt import (
    _mid_date_str,
    _next_date_str,
    _query_report_rc_adaptive,
    download_report_rc,
)
from src.lazybull.data.report_rc import query_report_rc_adaptive


@pytest.fixture(autouse=True)
def _reset_error_collector():
    """每个测试前清空全局错误收集器，避免跨测试污染。"""
    with raw_core.ERROR_COLLECTOR._lock:
        raw_core.ERROR_COLLECTOR._errors.clear()
    yield


def _make_overlimit_pagination(limit_days=180):
    """构造 fake 分页: 日期跨度超过 limit_days 天视为"超限"抛错, 否则返回 1 行。

    模拟 report_rc 单次查询 100000 条上限: 数据量越大越容易失败。
    同时记录调用参数供断言。
    """
    calls = []

    def _fake_pagination(client, api_name, page_limit=50000, fields=None, **kwargs):
        # 用真实日期差计算跨度 (YYYYMMDD 字符串转 int 相减不是真实天数)
        start = datetime.strptime(kwargs["start_date"], "%Y%m%d")
        end = datetime.strptime(kwargs["end_date"], "%Y%m%d")
        days = (end - start).days + 1
        calls.append((kwargs["start_date"], kwargs["end_date"], days))
        if days > limit_days:
            raise RuntimeError("查询数据失败，请确认参数！")
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "report_date": [kwargs["end_date"]],
                "org_name": ["x"],
                "author_name": ["y"],
                "report_title": ["测试研报"],
                "quarter": [f"{kwargs['end_date'][:4]}Q4"],
            }
        )

    return _fake_pagination, calls


def _make_overlimit_pagination_with_na_col(limit_days=180):
    """同 _make_overlimit_pagination, 但返回页含全 NaN 列 (触发 concat FutureWarning)。"""
    calls = []

    def _fake_pagination(client, api_name, page_limit=50000, fields=None, **kwargs):
        start = datetime.strptime(kwargs["start_date"], "%Y%m%d")
        end = datetime.strptime(kwargs["end_date"], "%Y%m%d")
        days = (end - start).days + 1
        calls.append((kwargs["start_date"], kwargs["end_date"], days))
        if days > limit_days:
            raise RuntimeError("查询数据失败，请确认参数！")
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "report_date": [kwargs["end_date"]],
                "org_name": ["x"],
                "author_name": ["y"],
                "report_title": ["测试研报"],
                "quarter": [f"{kwargs['end_date'][:4]}Q4"],
                "max_price": [None],  # 全 NaN 列, 触发 concat FutureWarning
            }
        )

    return _fake_pagination, calls


class TestDateHelpers:
    def test_mid_date_str(self):
        # 2024 闰年: 365 天中点 = 182.5 天 -> 7 月 1 日 (取整)
        assert _mid_date_str("20240101", "20241231") == "20240701"
        # 2009 非闰年: 364 天中点 = 182 天 -> 7 月 2 日
        assert _mid_date_str("20090101", "20091231") == "20090702"

    def test_next_date_str(self):
        assert _next_date_str("20240702") == "20240703"
        assert _next_date_str("20241231") == "20250101"


class TestQueryReportRcAdaptive:
    def test_whole_range_success_no_bisect(self, monkeypatch):
        fake_pagination, calls = _make_overlimit_pagination()
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        # 短区间 (30 天) 不超限 -> 直接返回, 只调用一次
        df = _query_report_rc_adaptive(object(), "20240101", "20240131")
        assert len(df) == 1
        assert len(calls) == 1

    def test_overlimit_bisects_and_merges(self, monkeypatch):
        fake_pagination, calls = _make_overlimit_pagination(limit_days=180)
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        # 整年 365 天 > 180 -> 失败, 二分到 ~91 天一段后全部成功
        df = _query_report_rc_adaptive(object(), "20240101", "20241231")
        assert len(df) == 4  # 每段 1 行, 共 4 段
        assert len(calls) > 1  # 发生了二分 (含失败的整段/半段尝试)

    def test_exhausted_depth_raises(self, monkeypatch):
        fake_pagination, _ = _make_overlimit_pagination(limit_days=180)
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        with pytest.raises(RuntimeError, match="二分 1 层后仍失败"):
            _query_report_rc_adaptive(object(), "20240101", "20241231", max_depth=1)

    def test_empty_range_returns_empty(self, monkeypatch):
        fake_pagination, calls = _make_overlimit_pagination()
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        df = _query_report_rc_adaptive(object(), "20241231", "20240101")
        assert len(df) == 0
        assert len(calls) == 0

    def test_non_overlimit_error_not_bisected(self, monkeypatch):
        """网络超时等非超限错误不应触发二分, 直接上抛 (避免对全局性问题无意义递归)。"""
        calls = []

        def _fake_pagination(client, api_name, page_limit=50000, fields=None, **kwargs):
            calls.append(kwargs["start_date"])
            raise TimeoutError("HTTPConnectionPool(...): Read timed out. (read timeout=30)")

        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", _fake_pagination)
        with pytest.raises(TimeoutError):
            _query_report_rc_adaptive(object(), "20240101", "20241231")
        assert len(calls) == 1  # 只尝试整段一次, 未发生二分

    def test_bisect_concat_suppresses_warning(self, monkeypatch):
        """二分合并 (pd.concat parts) 也应屏蔽 empty/all-NA FutureWarning。"""
        fake_pagination, _ = _make_overlimit_pagination_with_na_col(limit_days=180)
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = _query_report_rc_adaptive(object(), "20240101", "20241231")

        assert len(df) == 4  # 二分成功取全
        assert not any(
            "FutureWarning" in str(x.category) and "empty or all-NA" in str(x.message) for x in w
        )


class TestCommonQueryReportRcAdaptive:
    """直接覆盖 ensure 与离线下载共用的生产二分函数。"""

    def test_bisects_and_merges_contiguous_ranges(self):
        successful_ranges = []

        def _query_range(start_date, end_date):
            days = (
                datetime.strptime(end_date, "%Y%m%d") - datetime.strptime(start_date, "%Y%m%d")
            ).days
            if days > 1:
                raise RuntimeError("查询数据失败，请确认参数！")
            successful_ranges.append((start_date, end_date))
            return pd.DataFrame({"start_date": [start_date], "end_date": [end_date]})

        result = query_report_rc_adaptive(_query_range, "20240101", "20240108")

        assert len(result) == 4
        assert successful_ranges == [
            ("20240101", "20240102"),
            ("20240103", "20240104"),
            ("20240105", "20240106"),
            ("20240107", "20240108"),
        ]

    @pytest.mark.parametrize(
        "message",
        [
            "查询数据失败: 网络连接中断",
            "请求失败，请确认参数后重试",
            "HTTPConnectionPool: Read timed out",
        ],
    )
    def test_non_overlimit_errors_propagate_without_bisect(self, message):
        calls = []

        def _query_range(start_date, end_date):
            calls.append((start_date, end_date))
            raise RuntimeError(message)

        with pytest.raises(RuntimeError, match=message):
            query_report_rc_adaptive(_query_range, "20240101", "20241231")
        assert calls == [("20240101", "20241231")]

    def test_single_day_overlimit_fails_without_repeating_same_range(self):
        calls = []

        def _query_range(start_date, end_date):
            calls.append((start_date, end_date))
            raise RuntimeError("查询数据失败，请确认参数！")

        with pytest.raises(RuntimeError, match="二分 6 层后仍失败"):
            query_report_rc_adaptive(_query_range, "20240101", "20240101")
        assert calls == [("20240101", "20240101")]


class TestDownloadReportRcAdaptive:
    def test_current_year_partition_resumes_from_latest_report_date(self, monkeypatch):
        """当前年已有分区仍应从最新研报日次日续传，并与旧分区合并。"""
        current_year = datetime.now().year
        year = str(current_year)
        partition = f"{year}-12-31"
        existing = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": f"{year}0110",
                    "org_name": "机构甲",
                    "author_name": "分析师甲",
                    "report_title": "旧研报",
                    "quarter": f"{year}Q4",
                }
            ]
        )
        captured = []

        def _fake_pagination(client, api_name, page_limit=50000, fields=None, **kwargs):
            captured.append(kwargs)
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "report_date": f"{year}0111",
                        "org_name": "机构甲",
                        "author_name": "分析师甲",
                        "report_title": "新研报",
                        "quarter": f"{year}Q4",
                    }
                ]
            )

        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", _fake_pagination)

        class _FakeStorage:
            def __init__(self):
                self.saved_partitions = {}

            def list_partitions(self, layer, name):
                return [partition]

            def load_raw_by_date(self, name, period):
                assert period == partition
                return existing.copy()

            def save_raw_by_date(self, df, name, period):
                self.saved_partitions[period] = df.copy()

        storage = _FakeStorage()
        download_report_rc(
            client=object(),
            storage=storage,
            start_date=f"{year}0101",
            end_date=f"{year}0112",
        )

        assert len(captured) == 1
        assert captured[0]["start_date"] == f"{year}0111"
        assert captured[0]["end_date"] == f"{year}0112"
        saved = storage.saved_partitions[partition]
        assert saved["report_title"].tolist() == ["旧研报", "新研报"]

    def test_overlimit_year_downloaded_via_bisect(self, monkeypatch):
        fake_pagination, _ = _make_overlimit_pagination(limit_days=180)
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        class _FakeStorage:
            def __init__(self):
                self.saved_partitions = {}

            def list_partitions(self, layer, name):
                return []

            def save_raw_by_date(self, df, name, period):
                self.saved_partitions[period] = df.copy()

        storage = _FakeStorage()
        download_report_rc(
            client=object(),
            storage=storage,
            start_date="20240101",
            end_date="20241231",
            force=True,
        )

        # 超限年份通过二分成功下载, 按年独立分区落盘
        assert set(storage.saved_partitions.keys()) == {"2024-12-31"}
        assert len(storage.saved_partitions["2024-12-31"]) == 4
        assert "report_rc" not in raw_core.ERROR_COLLECTOR._errors

    def test_multi_year_concurrent_merge(self, monkeypatch):
        """并发下载多年份: 各年份 (含超限二分) 合并落盘, 计数正确。"""
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 4)
        fake_pagination, _ = _make_overlimit_pagination(limit_days=180)
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        class _FakeStorage:
            def __init__(self):
                self.saved_partitions = {}

            def list_partitions(self, layer, name):
                return []

            def save_raw_by_date(self, df, name, period):
                self.saved_partitions[period] = df.copy()

        storage = _FakeStorage()
        download_report_rc(
            client=object(),
            storage=storage,
            start_date="20230101",
            end_date="20241231",
            force=True,
        )

        # 2023 + 2024 两年, 每年超限二分后 4 段 -> 两个年分区各 4 行
        assert set(storage.saved_partitions.keys()) == {"2023-12-31", "2024-12-31"}
        assert sum(len(v) for v in storage.saved_partitions.values()) == 8
        assert "report_rc" not in raw_core.ERROR_COLLECTOR._errors

    def test_uses_conservative_concurrency(self, monkeypatch):
        """download_report_rc 应使用保守并发 (_REPORT_RC_CONCURRENCY), 避免打爆本地代理。"""
        from scripts.raw_download.alt import _REPORT_RC_CONCURRENCY

        captured = {}

        def _fake_pagination(client, api_name, page_limit=50000, fields=None, **kwargs):
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "report_date": [kwargs["end_date"]],
                    "org_name": ["x"],
                    "author_name": ["y"],
                    "report_title": ["测试研报"],
                    "quarter": [f"{kwargs['end_date'][:4]}Q4"],
                }
            )

        def _fake_run_concurrent(work_items, worker, label, collect=False, max_workers=None):
            captured["max_workers"] = max_workers
            return [worker(y) for y in work_items]

        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", _fake_pagination)
        monkeypatch.setattr(raw_download_alt, "_run_concurrent", _fake_run_concurrent)

        class _FakeStorage:
            def __init__(self):
                self.saved_partitions = {}

            def list_partitions(self, layer, name):
                return []

            def save_raw_by_date(self, df, name, period):
                self.saved_partitions[period] = df.copy()

        storage = _FakeStorage()
        download_report_rc(
            client=object(),
            storage=storage,
            start_date="20240101",
            end_date="20241231",
            force=True,
        )

        assert captured["max_workers"] == _REPORT_RC_CONCURRENCY
        assert set(storage.saved_partitions) == {"2024-12-31"}
        assert "report_rc" not in raw_core.ERROR_COLLECTOR._errors

    def test_serial_degrade_consistent(self, monkeypatch):
        """串行降级 (_DOWNLOAD_CONCURRENCY=1) 与并发结果一致。"""
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 1)
        fake_pagination, _ = _make_overlimit_pagination(limit_days=180)
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        class _FakeStorage:
            def __init__(self):
                self.saved_partitions = {}

            def list_partitions(self, layer, name):
                return []

            def save_raw_by_date(self, df, name, period):
                self.saved_partitions[period] = df.copy()

        storage = _FakeStorage()
        download_report_rc(
            client=object(),
            storage=storage,
            start_date="20230101",
            end_date="20241231",
            force=True,
        )

        # 串行降级: 两个年分区各 4 行, 与并发结果一致
        assert set(storage.saved_partitions.keys()) == {"2023-12-31", "2024-12-31"}
        assert sum(len(v) for v in storage.saved_partitions.values()) == 8
        assert "report_rc" not in raw_core.ERROR_COLLECTOR._errors
