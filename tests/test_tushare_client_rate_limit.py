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
        # 已知低限频接口使用独立 interval；未知接口回退全局 500/分钟
        cyq_limit = client._api_rate_limits["cyq_perf"]
        assert client._request_interval_for("cyq_perf") == pytest.approx(60.0 / cyq_limit)
        assert client._request_interval_for("daily") == pytest.approx(60.0 / 500)

    def test_rate_limit_wait_uses_per_api_bucket(self):
        client = _make_client(rate_limit=500)
        cyq_interval = client._request_interval_for("cyq_perf")
        # 每个接口首次调用各自分桶，不因其他接口已调用而 sleep
        t0 = time.perf_counter()
        client._rate_limit_wait("cyq_perf")
        client._rate_limit_wait("daily")
        client._rate_limit_wait("cyq_perf")
        elapsed = time.perf_counter() - t0
        # 两次 cyq_perf 间隔应 >= 接口级 interval；daily 与 cyq_perf 独立不额外等待
        assert elapsed >= cyq_interval - 0.02

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

    def test_deterministic_error_not_retried(self):
        """确定性业务错误 (查询数据失败/参数错误) 重试必失败, 应直接抛以节省请求量。"""
        client = _make_client(rate_limit=500)
        calls = {"n": 0}

        class _Pro:
            def query(self, api_name, fields=None, **kwargs):
                calls["n"] += 1
                raise Exception("查询数据失败，请确认参数！可以反馈管理员协助您排查问题")

        client.pro = _Pro()
        with pytest.raises(Exception, match="查询数据失败"):
            client.query("report_rc", skip_rate_limit=True)

        assert calls["n"] == 1  # 不重试

    def test_report_rc_has_interface_limit(self):
        """report_rc 应有接口级限频 (避免长期高并发被 TuShare 拒绝)。"""
        client = _make_client(rate_limit=500)
        limit = client._api_rate_limits["report_rc"]
        assert limit > 0
        assert client._request_interval_for("report_rc") == pytest.approx(60.0 / limit)


class TestTopListRateLimitOverride:
    """get_top_list 应局部放宽限频到 1000 次/分钟 (注释意图), 加速历史批量下载。"""

    def test_get_top_list_uses_api_level_limit(self, monkeypatch):
        """get_top_list 不应传 rate_limit_override (会绕过接口级限频);
        限频由 _API_RATE_LIMITS_DEFAULT["top_list"] 控制 (低于官方 500 次/分钟)。"""
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
        assert captured["rate_limit_override"] is None  # 不绕过接口级限频
        assert captured["kwargs"]["trade_date"] == "20240101"
        assert len(df) == 1

    def test_top_list_interval_uses_configured_limit(self):
        """top_list 令牌桶间隔应取 _API_RATE_LIMITS_DEFAULT["top_list"] (如 400)。"""
        client = _make_client(rate_limit=500)
        top_list_limit = client._api_rate_limits["top_list"]
        assert top_list_limit > 0
        # 接口级间隔 = 60 / 配置限频
        assert client._request_interval_for("top_list") == pytest.approx(60.0 / top_list_limit)
        # 配置限频应不高于官方 500 次/分钟 (间隔 >= 60/500 = 0.12s), 避免被限流
        assert client._request_interval_for("top_list") >= 60.0 / 500
