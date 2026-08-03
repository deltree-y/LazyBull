# -*- coding: utf-8 -*-
"""download_raw 分页与断点续传修复回归测试。

覆盖：
- _query_with_pagination 走 client.query 限频（不再绕过）+ 分页累积 + max_pages 兜底
- download_stk_holdernumber 断点续传（已有 ann_date 后不再全量重下）
- download_report_rc 按年分页（规避单次 2000 条上限截断）
- download_basic_data 每次全量下载 trade_cal（不按 start/end 裁剪）
"""

import sys

import pandas as pd
import pytest

from scripts import download_raw as download_raw_module
from scripts.raw_download import alt as raw_download_alt
import scripts.raw_download.core as raw_core
from scripts.raw_download.cli import main
from scripts.raw_download.basic import download_basic_data


@pytest.fixture(autouse=True)
def _reset_error_collector():
    """每个测试前清空全局错误收集器，避免跨测试污染。"""
    with download_raw_module.ERROR_COLLECTOR._lock:
        download_raw_module.ERROR_COLLECTOR._errors.clear()
    yield


class TestQueryWithPagination:
    """修复: _query_with_pagination 应走 client.query 限频、分页累积、max_pages 兜底。"""

    def test_short_page_returns_single_page(self):
        calls = []

        class _FakeClient:
            def query(self, api_name, fields=None, **kwargs):
                calls.append((api_name, kwargs))
                return pd.DataFrame({"v": [1, 2, 3]})

        result = download_raw_module._query_with_pagination(
            _FakeClient(), "test_api", page_limit=100
        )

        assert len(result) == 3
        assert len(calls) == 1
        assert calls[0][0] == "test_api"
        assert calls[0][1]["limit"] == 100
        assert calls[0][1]["offset"] == 0

    def test_rolls_offset_when_exact_page_limit(self):
        calls = []
        pages = [
            pd.DataFrame({"v": list(range(100))}),  # 恰好整页
            pd.DataFrame({"v": [1, 2]}),            # 第二页不足 → 结束
        ]

        class _FakeClient:
            def query(self, api_name, fields=None, **kwargs):
                calls.append(kwargs)
                return pages.pop(0) if pages else pd.DataFrame()

        result = download_raw_module._query_with_pagination(
            _FakeClient(), "test_api", page_limit=100
        )

        assert len(result) == 102
        assert len(calls) == 2
        assert calls[0]["offset"] == 0
        assert calls[1]["offset"] == 100

    def test_max_pages_guards_against_infinite_loop(self):
        class _FakeClient:
            def query(self, api_name, fields=None, **kwargs):
                # 永远返回恰好整页（模拟接口不支持 offset 的情形）→ 触发 max_pages 兜底
                return pd.DataFrame({"v": list(range(50))})

        result = download_raw_module._query_with_pagination(
            _FakeClient(), "test_api", page_limit=50, max_pages=3
        )

        assert len(result) == 150  # 恰 3 页后停止，不再无限循环


class TestStkHoldernumberResume:
    """修复: download_stk_holdernumber 断点续传 + 单月分页。"""

    def test_only_downloads_months_after_latest_ann_date(self, monkeypatch):
        existing = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20240331"],
                "end_date": ["20231231"],
            }
        )

        class _FakeStorage:
            def load_raw(self, name):
                assert name == "stk_holdernumber"
                return existing

            def save_raw(self, df, name, is_force=False):
                assert name == "stk_holdernumber"
                assert len(df) > 0

        captured = []

        def _fake_pagination(client, api_name, page_limit=50000, fields=None, **kwargs):
            captured.append((page_limit, kwargs))
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "ann_date": [kwargs["end_date"]],
                    "end_date": ["20240331"],
                }
            )

        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", _fake_pagination)

        download_raw_module.download_stk_holdernumber(
            client=object(),
            storage=_FakeStorage(),
            start_date="20240101",
            end_date="20240630",
        )

        # 20240101~20240630 共 6 个月；已有最新 ann_date=20240331
        # 只应下载 month_end > 20240331 的月份：2024-04/05/06 → 3 次
        assert len(captured) == 3
        assert captured[0][0] == 3000  # 单月分页，规避 3000 条上限
        assert captured[0][1]["start_date"] == "20240401"

    def test_skips_entirely_when_existing_covers_range(self, monkeypatch):
        existing = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20240630"],  # 已覆盖到月末 → 全部月份无需下载
                "end_date": ["20240331"],
            }
        )

        class _FakeStorage:
            def load_raw(self, name):
                return existing

            def save_raw(self, df, name, is_force=False):
                raise AssertionError("不应触发保存")

        captured = []

        def _fake_pagination(client, api_name, page_limit=50000, fields=None, **kwargs):
            captured.append(kwargs)
            return pd.DataFrame()

        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", _fake_pagination)

        download_raw_module.download_stk_holdernumber(
            client=object(),
            storage=_FakeStorage(),
            start_date="20240101",
            end_date="20240630",
        )

        assert captured == []  # 已有数据覆盖整个区间，一次都不下载


class TestTradeCalFullDownload:
    """download_basic_data 每次全量下载 trade_cal（不按 start/end 裁剪）。"""

    def test_fetches_full_trade_cal_ignoring_date_range(self):
        existing = pd.DataFrame(
            {"cal_date": ["20240101"], "exchange": ["SSE"], "is_open": [1]}
        )
        full = pd.DataFrame(
            {
                "exchange": ["SSE"] * 3,
                "cal_date": ["20230101", "20240101", "20250101"],
                "is_open": [1, 1, 1],
            }
        )

        class _FakeStorage:
            def __init__(self):
                self.saved = None

            def load_raw(self, name):
                if name == "trade_cal":
                    return existing
                return None

            def save_raw(self, df, name, is_force=False):
                self.saved = (name, df.copy())

            def check_basic_data_freshness(self, name, end_date):
                return True  # 跳过 stock_basic，聚焦 trade_cal

        calls = []

        class _FakeClient:
            def get_trade_cal(self, start_date=None, end_date=None, exchange="SSE"):
                calls.append(
                    {"start_date": start_date, "end_date": end_date, "exchange": exchange}
                )
                return full

            def get_stock_basic(self, list_status="L", fields=None):
                return pd.DataFrame()

        storage = _FakeStorage()
        client = _FakeClient()
        download_basic_data(client, storage, "20240101", "20240131")

        # 关键断言：无论请求区间如何，trade_cal 都以全量拉取（不传 start/end）
        assert calls[0]["start_date"] is None
        assert calls[0]["end_date"] is None
        assert calls[0]["exchange"] == "SSE"
        # 合并去重排序后保存
        assert storage.saved is not None
        assert storage.saved[0] == "trade_cal"
        assert storage.saved[1]["cal_date"].tolist() == [
            "20230101",
            "20240101",
            "20250101",
        ]

    def test_keeps_existing_when_full_fetch_returns_empty(self):
        existing = pd.DataFrame(
            {"cal_date": ["20240101"], "exchange": ["SSE"], "is_open": [1]}
        )

        class _FakeStorage:
            def __init__(self):
                self.saved = None

            def load_raw(self, name):
                if name == "trade_cal":
                    return existing
                return None

            def save_raw(self, df, name, is_force=False):
                self.saved = (name, df.copy())

            def check_basic_data_freshness(self, name, end_date):
                return True

        class _FakeClient:
            def get_trade_cal(self, start_date=None, end_date=None, exchange="SSE"):
                return None  # 全量拉取失败

            def get_stock_basic(self, list_status="L", fields=None):
                return pd.DataFrame()

        storage = _FakeStorage()
        result = download_basic_data(
            _FakeClient(), storage, "20240101", "20240131"
        )

        # 拉取失败时保留已有数据，不抛异常
        assert storage.saved is None
        assert len(result) == 1


class TestCliMainRuntime:
    """cli.main 运行时：并发数应写入 core 模块（供 _run_concurrent 读取），且无 NameError。"""

    def test_main_writes_concurrency_to_core(self, monkeypatch):
        monkeypatch.setattr("scripts.raw_download.cli.setup_logger", lambda log_level="INFO": None)
        monkeypatch.setattr("scripts.raw_download.cli.get_config", lambda: None)
        monkeypatch.setattr(
            "scripts.raw_download.cli.get_tushare_settings",
            lambda: {
                "rate_limit": 500,
                "retry_rate_limit_sleep": 15.0,
                "download_concurrency": 4,
            },
        )
        monkeypatch.setattr("scripts.raw_download.cli.TushareClient", lambda verbose=True: object())
        monkeypatch.setattr("scripts.raw_download.cli.Storage", lambda: object())
        monkeypatch.setattr(
            "scripts.raw_download.cli.download_basic_data",
            lambda client, storage, start_date, end_date, force=False: pd.DataFrame(
                {"cal_date": ["20240101"], "is_open": [1]}
            ),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "download_raw.py",
                "--only-basic",
                "--concurrency",
                "8",
                "--start-date",
                "20240101",
                "--end-date",
                "20240131",
            ],
        )

        with pytest.raises(SystemExit):
            main()

        # 关键断言：并发数写入 core 模块，而非 cli 模块的孤立变量
        assert raw_core._DOWNLOAD_CONCURRENCY == 8


class TestReportRcPagination:
    """修复: download_report_rc 按年分页，规避单次 2000 条上限截断。"""

    def test_uses_pagination_with_2000_page_limit(self, monkeypatch):
        class _FakeStorage:
            def load_raw(self, name):
                assert name == "report_rc"
                return None

            def save_raw(self, df, name, is_force=False):
                assert name == "report_rc"

        captured = []

        def _fake_pagination(client, api_name, page_limit=50000, fields=None, **kwargs):
            captured.append((api_name, page_limit, kwargs))
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "report_date": [kwargs["end_date"]],
                    "org_name": ["x"],
                    "author_name": ["y"],
                }
            )

        monkeypatch.setattr(raw_download_alt, "_query_with_pagination", _fake_pagination)

        download_raw_module.download_report_rc(
            client=object(),
            storage=_FakeStorage(),
            start_date="20240101",
            end_date="20241231",
            force=True,
        )

        assert len(captured) == 1
        assert captured[0][0] == "report_rc"
        assert captured[0][1] == 2000
        assert captured[0][2]["start_date"] == "20240101"
        assert captured[0][2]["end_date"] == "20241231"
