#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SMB 客户端模块测试。

Mock _read_file / _connect 内部方法，不依赖 pysmb 安装。
"""

import io
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.lazybull.common.smb_client import SMBFileReader, parse_smb_url


class TestParseSmbUrl:
    """parse_smb_url 函数测试。"""

    def test_double_slash_format(self):
        result = parse_smb_url("//192.168.1.21/docker/lazybull/data/paper")
        assert result == {
            "host": "192.168.1.21",
            "share": "docker",
            "path": "lazybull/data/paper",
        }

    def test_smb_protocol_format(self):
        result = parse_smb_url("smb://192.168.1.21/docker/lazybull/data/paper")
        assert result == {
            "host": "192.168.1.21",
            "share": "docker",
            "path": "lazybull/data/paper",
        }

    def test_backslash_format(self):
        result = parse_smb_url("\\\\192.168.1.21\\docker\\lazybull\\data\\paper")
        assert result == {
            "host": "192.168.1.21",
            "share": "docker",
            "path": "lazybull/data/paper",
        }

    def test_no_path(self):
        result = parse_smb_url("//192.168.1.21/docker")
        assert result == {"host": "192.168.1.21", "share": "docker", "path": ""}

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError):
            parse_smb_url("//192.168.1.21")

    def test_trailing_slash(self):
        result = parse_smb_url("//192.168.1.21/docker/path/")
        assert result["path"] == "path"


class TestSMBFileReader:
    """SMBFileReader 单元测试（mock _read_file / _connect）。"""

    def test_build_remote_path_with_prefix(self):
        reader = SMBFileReader("1.2.3.4", "share", "prefix/sub")
        assert reader._build_remote_path("state/account.json") == "prefix/sub/state/account.json"

    def test_build_remote_path_no_prefix(self):
        reader = SMBFileReader("1.2.3.4", "share")
        assert reader._build_remote_path("state/account.json") == "state/account.json"

    def test_build_remote_path_backslash(self):
        reader = SMBFileReader("1.2.3.4", "share", "pref\\ix")
        assert reader._build_remote_path("state\\account.json") == "pref/ix/state/account.json"

    def test_read_json_success(self):
        reader = SMBFileReader("1.2.3.4", "share", "prefix")
        reader._read_file = MagicMock(return_value=b'{"key": "value"}')
        result = reader.read_json("test.json")
        assert result == {"key": "value"}

    def test_read_json_empty_raises(self):
        """空文件应抛出 ValueError。"""
        reader = SMBFileReader("1.2.3.4", "share")
        reader._read_file = MagicMock(return_value=b"")
        with pytest.raises(ValueError, match="文件为空"):
            reader.read_json("test.json")

    def test_read_json_exception_propagates(self):
        """SMB 读取异常应向上传播（不再静默吞掉）。"""
        reader = SMBFileReader("1.2.3.4", "share")
        reader._read_file = MagicMock(side_effect=ConnectionError("SMB error"))
        with pytest.raises(ConnectionError, match="SMB error"):
            reader.read_json("test.json")

    def test_read_parquet_success(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        buf = io.BytesIO()
        df.to_parquet(buf)
        parquet_bytes = buf.getvalue()

        reader = SMBFileReader("1.2.3.4", "share")
        reader._read_file = MagicMock(return_value=parquet_bytes)
        result = reader.read_parquet("test.parquet")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_read_parquet_exception(self):
        reader = SMBFileReader("1.2.3.4", "share")
        reader._read_file = MagicMock(side_effect=Exception("SMB error"))
        result = reader.read_parquet("test.parquet")
        assert result is None

    def test_file_exists_true(self):
        reader = SMBFileReader("1.2.3.4", "share", "prefix")
        mock_conn = MagicMock()
        reader._connect = MagicMock(return_value=mock_conn)
        assert reader.file_exists("state/account.json") is True

    def test_file_exists_false(self):
        reader = SMBFileReader("1.2.3.4", "share")
        mock_conn = MagicMock()
        mock_conn.getAttributes.side_effect = Exception("not found")
        reader._connect = MagicMock(return_value=mock_conn)
        assert reader.file_exists("nonexistent.json") is False

    def test_cache_set_and_get(self):
        reader = SMBFileReader("1.2.3.4", "share")
        reader.set_cache("test_key", "cached_value")
        assert reader.get_cached("test_key", max_age_seconds=3600) == "cached_value"

    def test_cache_expiry(self):
        """使用负值确保必定过期。"""
        reader = SMBFileReader("1.2.3.4", "share")
        reader.set_cache("test_key", "cached_value")
        assert reader.get_cached("test_key", max_age_seconds=-1.0) is None

    def test_cache_clear(self):
        reader = SMBFileReader("1.2.3.4", "share")
        reader.set_cache("test_key", "value")
        reader.clear_cache()
        assert reader.get_cached("test_key", max_age_seconds=3600) is None
