# -*- coding: utf-8 -*-
"""TushareClient 接口级限频自适应回归测试。

覆盖：
- 已知低限频接口 (cyq_perf=100/分钟) 使用独立 interval，不拖累全局 500/分钟
- 每个接口独立令牌桶分桶
- 限流错误自动解析"频率超限(X次/分钟)"并更新接口限频
"""

import time

import pandas as pd
import pytest

from src.lazybull.data import tushare_client as tc_module
from src.lazybull.data.tushare_client import TushareClient


def _make_client(rate_limit: int = 500) -> TushareClient:
    """绕过 __init__（需真实 token），手工构造最小实例。"""
    client = object.__new__(TushareClient)
    client.rate_limit = rate_limit
    client._request_interval = 60.0 / rate_limit
    client._api_rate_limits = dict(tc_module._API_RATE_LIMITS_DEFAULT)
    client._rate_limit_locks = {}
    client._last_request_time_by_api = {}
    client.max_retries = 3
    client.retry_delay = 0.0
    client._retry_rate_limit_sleep = 0.0
    client._rate_limit_keywords = [
        "每分钟",
        "访问",
        "频次",
        "rate",
        "limit",
        "频率",
        "429",
        "超过",
    ]
    client.verbose = False
    return client


class TestInterfaceRateLimit:
    def test_known_low_freq_interface_uses_smaller_interval(self):
        client = _make_client(rate_limit=500)
        # cyq_perf 限频 100/分钟 → interval 0.6s；未知接口回退全局 500/分钟 → 0.12s
        assert client._request_interval_for("cyq_perf") == pytest.approx(60.0 / 100)
        assert client._request_interval_for("daily") == pytest.approx(60.0 / 500)

    def test_rate_limit_wait_uses_per_api_bucket(self):
        client = _make_client(rate_limit=500)
        # 每个接口首次调用各自分桶，不因其他接口已调用而 sleep
        t0 = time.perf_counter()
        client._rate_limit_wait("cyq_perf")
        client._rate_limit_wait("daily")
        client._rate_limit_wait("cyq_perf")
        elapsed = time.perf_counter() - t0
        # 两次 cyq_perf 间隔应 >= 0.6s（60/100），daily 与 cyq_perf 独立不额外等待
        assert elapsed >= 0.58

    def test_rate_limit_error_updates_api_rate_limit(self):
        client = _make_client(rate_limit=500)
        calls = {"n": 0}

        class _Pro:
            def query(self, api_name, fields=None, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise Exception("抱歉，您访问接口(cyq_perf)频率超限(200次/分钟)，具体频次详情")
                return pd.DataFrame({"x": [1]})

        client.pro = _Pro()
        client.query("cyq_perf", skip_rate_limit=True)

        assert calls["n"] == 2  # 第一次限流 → 重试成功
        assert client._api_rate_limits["cyq_perf"] == 200  # 自适应更新
        # 更新后该接口 interval 变为 60/200
        assert client._request_interval_for("cyq_perf") == pytest.approx(60.0 / 200)

    def test_non_rate_limit_error_does_not_update(self):
        client = _make_client(rate_limit=500)
        calls = {"n": 0}

        class _Pro:
            def query(self, api_name, fields=None, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise Exception("接口不存在")
                return pd.DataFrame({"x": [1]})

        client.pro = _Pro()
        client.query("daily", skip_rate_limit=True)

        assert client._api_rate_limits.get("daily") is None  # 非限流错误不更新


class TestTopListRateLimitOverride:
    """get_top_list 应局部放宽限频到 1000 次/分钟 (注释意图), 加速历史批量下载。"""

    def test_get_top_list_uses_1000_override(self, monkeypatch):
        client = _make_client(rate_limit=500)
        captured = {}

        def _fake_query(
            api_name,
            fields=None,
            skip_rate_limit=False,
            rate_limit_override=None,
            **kwargs,
        ):
            captured["api_name"] = api_name
            captured["rate_limit_override"] = rate_limit_override
            captured["kwargs"] = kwargs
            return pd.DataFrame({"ts_code": ["000001.SZ"]})

        monkeypatch.setattr(client, "query", _fake_query)
        df = client.get_top_list(trade_date="20240101")

        assert captured["api_name"] == "top_list"
        assert captured["rate_limit_override"] == 1000  # 60 次/分钟是笔误
        assert captured["kwargs"]["trade_date"] == "20240101"
        assert len(df) == 1
