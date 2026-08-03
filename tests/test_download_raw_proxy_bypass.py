# -*- coding: utf-8 -*-
"""download_raw 启动时自动绕过代理的回归测试。

覆盖：
- 默认启动时清除进程内代理环境变量 (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY 及小写)
- LAZYBULL_DOWNLOAD_BYPASS_PROXY=0 可关闭绕过
- 未注入代理时为空操作
"""

import os

import pytest

from scripts.raw_download.cli import (
    _DOWNLOAD_PROXY_ENV_KEYS,
    _bypass_proxy_for_download,
    _should_bypass_proxy_for_download,
)


@pytest.fixture
def _clear_flag(monkeypatch):
    """确保测试从无开关状态开始（默认启用）。"""
    monkeypatch.delenv("LAZYBULL_DOWNLOAD_BYPASS_PROXY", raising=False)


def _set_all_proxy_envs(monkeypatch):
    """注入全部代理环境变量。"""
    for key in _DOWNLOAD_PROXY_ENV_KEYS:
        monkeypatch.setenv(key, f"http://proxy-{key.lower()}")


class TestShouldBypassProxyForDownload:
    """开关读取：单变量直读、单默认值。"""

    def test_default_true(self, _clear_flag):
        assert _should_bypass_proxy_for_download() is True

    def test_true_when_one(self, monkeypatch):
        monkeypatch.setenv("LAZYBULL_DOWNLOAD_BYPASS_PROXY", "1")
        assert _should_bypass_proxy_for_download() is True

    def test_false_when_zero(self, monkeypatch):
        monkeypatch.setenv("LAZYBULL_DOWNLOAD_BYPASS_PROXY", "0")
        assert _should_bypass_proxy_for_download() is False

    def test_false_when_off_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LAZYBULL_DOWNLOAD_BYPASS_PROXY", "OFF")
        assert _should_bypass_proxy_for_download() is False

    def test_true_when_other_value(self, monkeypatch):
        monkeypatch.setenv("LAZYBULL_DOWNLOAD_BYPASS_PROXY", "yes")
        assert _should_bypass_proxy_for_download() is True


class TestBypassProxyForDownload:
    """进程内清除代理环境变量。"""

    def test_clears_proxy_env_vars_by_default(self, _clear_flag, monkeypatch):
        _set_all_proxy_envs(monkeypatch)
        _bypass_proxy_for_download()
        for key in _DOWNLOAD_PROXY_ENV_KEYS:
            assert os.environ.get(key) is None

    def test_keeps_proxy_env_vars_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LAZYBULL_DOWNLOAD_BYPASS_PROXY", "0")
        _set_all_proxy_envs(monkeypatch)
        _bypass_proxy_for_download()
        for key in _DOWNLOAD_PROXY_ENV_KEYS:
            assert os.environ.get(key) is not None

    def test_noop_when_no_proxy_env(self, _clear_flag, monkeypatch):
        # 未注入代理时调用不应报错, 也不应产生副作用
        _bypass_proxy_for_download()
        for key in _DOWNLOAD_PROXY_ENV_KEYS:
            assert os.environ.get(key) is None
