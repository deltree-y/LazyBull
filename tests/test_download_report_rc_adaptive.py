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


class TestDownloadReportRcAdaptive:
    def test_overlimit_year_downloaded_via_bisect(self, monkeypatch):
        fake_pagination, _ = _make_overlimit_pagination(limit_days=180)
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        class _FakeStorage:
            def __init__(self):
                self.saved = None

            def load_raw(self, name):
                return None

            def save_raw(self, df, name, is_force=False):
                self.saved = df.copy()

        storage = _FakeStorage()
        download_report_rc(
            client=object(),
            storage=storage,
            start_date="20240101",
            end_date="20241231",
            force=True,
        )

        # 超限年份通过二分成功下载并合并保存
        assert storage.saved is not None
        assert len(storage.saved) == 4
        assert "report_rc" not in raw_core.ERROR_COLLECTOR._errors

    def test_multi_year_concurrent_merge(self, monkeypatch):
        """并发下载多年份: 各年份 (含超限二分) 合并落盘, 计数正确。"""
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 4)
        fake_pagination, _ = _make_overlimit_pagination(limit_days=180)
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        class _FakeStorage:
            def __init__(self):
                self.saved = None

            def load_raw(self, name):
                return None

            def save_raw(self, df, name, is_force=False):
                self.saved = df.copy()

        storage = _FakeStorage()
        download_report_rc(
            client=object(),
            storage=storage,
            start_date="20230101",
            end_date="20241231",
            force=True,
        )

        # 2023 + 2024 两年, 每年超限二分后 4 段 -> 合并 8 行
        assert storage.saved is not None
        assert len(storage.saved) == 8
        assert "report_rc" not in raw_core.ERROR_COLLECTOR._errors

    def test_serial_degrade_consistent(self, monkeypatch):
        """串行降级 (_DOWNLOAD_CONCURRENCY=1) 与并发结果一致。"""
        monkeypatch.setattr(raw_core, "_DOWNLOAD_CONCURRENCY", 1)
        fake_pagination, _ = _make_overlimit_pagination(limit_days=180)
        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", fake_pagination)

        class _FakeStorage:
            def __init__(self):
                self.saved = None

            def load_raw(self, name):
                return None

            def save_raw(self, df, name, is_force=False):
                self.saved = df.copy()

        storage = _FakeStorage()
        download_report_rc(
            client=object(),
            storage=storage,
            start_date="20230101",
            end_date="20241231",
            force=True,
        )

        assert storage.saved is not None
        assert len(storage.saved) == 8
        assert "report_rc" not in raw_core.ERROR_COLLECTOR._errors
