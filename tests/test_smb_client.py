#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SMB 客户端模块测试。

mock _read_file_raw 测试解析层；mock subprocess.run 测试 SMB 错误路径。
"""

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.lazybull.common.smb_client import SMBFileReader, parse_smb_url


# ============================================================
# parse_smb_url
# ============================================================

class TestParseSmbUrl:
    def test_double_slash(self):
        r = parse_smb_url("//192.168.1.21/docker/a/b")
        assert r == {"host": "192.168.1.21", "share": "docker", "path": "a/b"}

    def test_smb_protocol(self):
        r = parse_smb_url("smb://h/s/p")
        assert r == {"host": "h", "share": "s", "path": "p"}

    def test_backslash(self):
        r = parse_smb_url("\\\\h\\s\\a\\b")
        assert r == {"host": "h", "share": "s", "path": "a/b"}

    def test_no_path(self):
        assert parse_smb_url("//h/s") == {"host": "h", "share": "s", "path": ""}

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_smb_url("//host")


# ============================================================
# SMBFileReader — mock _read_file_raw
# ============================================================

class TestSMBFileReader:
    def _reader(self, raw_bytes=b"", **kw):
        reader = SMBFileReader("1.2.3.4", "share", "pref", **kw)
        reader._read_file_raw = MagicMock(return_value=raw_bytes)
        return reader

    def test_path_prefix(self):
        assert self._reader()._build_remote_path("a/b.json") == "pref/a/b.json"

    def test_path_no_prefix(self):
        r = SMBFileReader("h", "s")
        assert r._build_remote_path("a.json") == "a.json"

    def test_smb_url(self):
        assert SMBFileReader("1.2.3.4", "docker")._smb_url == "//1.2.3.4/docker"

    def test_auth_user(self):
        assert SMBFileReader("h", "s", username="u", password="p")._auth_arg == "u%p"

    def test_auth_guest(self):
        assert SMBFileReader("h", "s")._auth_arg == "guest%"

    # ---- JSON ----

    def test_read_json(self):
        assert self._reader(b'{"a":1}').read_json("f") == {"a": 1}

    def test_read_json_empty(self):
        with pytest.raises(ValueError, match="文件为空"):
            self._reader(b"").read_json("f")

    def test_read_json_file_not_found(self):
        r = SMBFileReader("h", "s")
        r._read_file_raw = MagicMock(side_effect=FileNotFoundError("gone"))
        with pytest.raises(FileNotFoundError):
            r.read_json("f")

    # ---- Parquet ----

    def test_read_parquet(self):
        df = pd.DataFrame({"x": [1]})
        buf = io.BytesIO()
        df.to_parquet(buf)
        result = self._reader(buf.getvalue()).read_parquet("f.pq")
        assert len(result) == 1

    def test_read_parquet_not_found(self):
        r = SMBFileReader("h", "s")
        r._read_file_raw = MagicMock(side_effect=FileNotFoundError("gone"))
        assert r.read_parquet("f.pq") is None

    # ---- file_exists ----

    def test_file_exists_true(self):
        assert self._reader(b"x").file_exists("f") is True

    def test_file_exists_false(self):
        r = SMBFileReader("h", "s")
        r._read_file_raw = MagicMock(side_effect=FileNotFoundError("gone"))
        assert r.file_exists("f") is False

    # ---- cache ----

    def test_cache(self):
        r = SMBFileReader("h", "s")
        r.set_cache("k", "v")
        assert r.get_cached("k", 3600) == "v"
        r.clear_cache()
        assert r.get_cached("k", 3600) is None

    def test_cache_expiry(self):
        r = SMBFileReader("h", "s")
        r.set_cache("k", "v")
        assert r.get_cached("k", -1) is None


# ============================================================
# _read_file_raw — mock subprocess.run
# ============================================================

def _mk_result(returncode, stderr, stdout=""):
    """构造模拟 subprocess.CompletedProcess。"""
    r = MagicMock()
    r.returncode = returncode
    r.stderr = stderr
    r.stdout = stdout
    return r


class TestReadFileRaw:
    def _mock_all(self, returncode, stderr):
        """一次性 patch 所有 subprocess/tempfile/os 依赖。"""
        return patch.multiple(
            "src.lazybull.common.smb_client",
            subprocess=MagicMock(run=MagicMock(return_value=_mk_result(returncode, stderr))),
            tempfile=MagicMock(mkstemp=MagicMock(return_value=(999, "/tmp/f"))),
            os=MagicMock(path=MagicMock(exists=MagicMock(return_value=True))),
        )

    def test_file_not_found(self):
        result = _mk_result(1, "NT_STATUS_OBJECT_NAME_NOT_FOUND foo")
        with patch("subprocess.run", return_value=result):
            with patch("tempfile.mkstemp", return_value=(999, "/tmp/f")):
                with patch("os.close"):
                    with patch("os.path.exists", return_value=True):
                        with patch("os.unlink"):
                            reader = SMBFileReader("1.2.3.4", "share")
                            with pytest.raises(FileNotFoundError):
                                reader._read_file_raw("test.json")

    def test_auth_failed(self):
        result = _mk_result(1, "NT_STATUS_LOGON_FAILURE session setup")
        with patch("subprocess.run", return_value=result):
            with patch("tempfile.mkstemp", return_value=(999, "/tmp/f")):
                with patch("os.close"):
                    with patch("os.path.exists", return_value=True):
                        with patch("os.unlink"):
                            reader = SMBFileReader("1.2.3.4", "share")
                            with pytest.raises(ConnectionError, match="认证失败"):
                                reader._read_file_raw("test.json")

    def test_other_error(self):
        result = _mk_result(1, "some other smb error")
        with patch("subprocess.run", return_value=result):
            with patch("tempfile.mkstemp", return_value=(999, "/tmp/f")):
                with patch("os.close"):
                    with patch("os.path.exists", return_value=True):
                        with patch("os.unlink"):
                            reader = SMBFileReader("1.2.3.4", "share")
                            with pytest.raises(ConnectionError):
                                reader._read_file_raw("test.json")
