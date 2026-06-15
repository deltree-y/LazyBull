#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SMB 远端文件读取客户端。

为 LCD35 从群晖 NAS 读取 paper 数据提供轻量 SMB2/3 协议访问，
基于 smbprotocol 库（纯 Python SMB2/3 实现）。
设计原则：每交易日仅读取一次，每次独立建立连接。
"""

import io
import json
import uuid
from typing import Optional

import pandas as pd
from loguru import logger


class SMBFileReader:
    """SMB 远端文件只读客户端（SMB2/3 协议）。

    每次调用独立建立连接→读取→断开。
    适用于低频读取场景（每交易日一次）。

    Usage:
        reader = SMBFileReader("192.168.1.21", "docker", "lazybull/data/paper")
        account = reader.read_json("state/account.json")
    """

    def __init__(
        self,
        host: str,
        share: str,
        path_prefix: str = "",
        username: str = "guest",
        password: str = "",
        port: int = 445,
        timeout: int = 15,
    ):
        self.host = host
        self.share = share
        self.path_prefix = str(path_prefix).replace("\\", "/").strip("/")
        self.username = username
        self.password = password
        self.port = port
        self.timeout = timeout
        self._cache: dict[str, tuple[float, object]] = {}

    def _build_remote_path(self, relative_path: str) -> str:
        """构建共享内的相对路径（使用正斜杠）。"""
        clean = str(relative_path).replace("\\", "/").strip("/")
        if self.path_prefix:
            return f"{self.path_prefix}/{clean}"
        return clean

    def _connect_all(self):
        """建立完整 SMB2/3 连接链: Connection -> Session -> TreeConnect。

        Returns:
            (connection, tree_connect) 元组
        """
        from smbprotocol.connection import Connection
        from smbprotocol.session import Session
        from smbprotocol.tree import TreeConnect

        try:
            connection = Connection(uuid.uuid4(), self.host, self.port)
            connection.connect(timeout=self.timeout)
            session = Session(connection, username=self.username, password=self.password)
            session.connect()
            tree = TreeConnect(session, self.share)
            tree.connect()
            return connection, tree
        except Exception as exc:
            raise ConnectionError(
                f"SMB 连接失败: host={self.host}, port={self.port}, "
                f"share={self.share}, err={type(exc).__name__}: {exc}"
            ) from exc

    def _disconnect(self, connection, tree) -> None:
        """安全断开 SMB 连接链。"""
        for obj in (tree, connection):
            if obj is not None:
                try:
                    obj.disconnect()
                except Exception:
                    pass

    def _open_and_read(self, relative_path: str) -> bytes:
        """建立连接并读取文件全部内容。

        Returns:
            文件字节内容

        Raises:
            FileNotFoundError: 文件不存在
            ConnectionError: SMB 连接失败
        """
        from smbprotocol.open import (
            Open, CreateDisposition, FileAttributes,
            ImpersonationLevel, ShareAccess, AccessMask,
        )

        remote_path = self._build_remote_path(relative_path)
        connection, tree = self._connect_all()
        try:
            open_file = Open(tree, remote_path)
            open_file.create(
                ImpersonationLevel.Impersonation,
                AccessMask.GENERIC_READ,
                ShareAccess.FILE_SHARE_READ,
                CreateDisposition.FILE_OPEN,
                FileAttributes.FILE_ATTRIBUTE_NORMAL,
            )
            try:
                file_size = open_file.end_of_file
                data = open_file.read(0, file_size) if file_size > 0 else b""
                return data
            finally:
                open_file.close()
        except Exception as exc:
            msg = str(exc).lower()
            if any(kw in msg for kw in (
                "not found", "no such file", "status_object_name_not_found",
                "status_no_such_file",
            )):
                raise FileNotFoundError(
                    f"SMB 文件不存在: share={self.share}, path={remote_path}"
                ) from exc
            raise FileNotFoundError(
                f"SMB 文件读取失败: share={self.share}, path={remote_path}, "
                f"err={type(exc).__name__}: {exc}"
            ) from exc
        finally:
            self._disconnect(connection, tree)

    def _read_file(self, relative_path: str) -> bytes:
        """读取文件原始字节（便于测试时 mock）。"""
        return self._open_and_read(relative_path)

    def file_exists(self, relative_path: str) -> bool:
        """检查远端文件是否存在。"""
        try:
            self._open_and_read(relative_path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    # ---- 公共读取方法 ----

    def read_text(self, relative_path: str, encoding: str = "utf-8") -> str:
        return self._read_file(relative_path).decode(encoding)

    def read_json(self, relative_path: str) -> dict:
        raw = self.read_text(relative_path)
        if not raw.strip():
            raise ValueError(f"SMB 远端文件为空: {self._build_remote_path(relative_path)}")
        return json.loads(raw)

    def read_yaml(self, relative_path: str) -> dict:
        import yaml
        raw = self.read_text(relative_path)
        result = yaml.safe_load(raw)
        if result is None:
            raise ValueError(f"SMB 远端 YAML 文件为空: {self._build_remote_path(relative_path)}")
        return result

    def read_parquet(self, relative_path: str) -> Optional[pd.DataFrame]:
        try:
            raw_bytes = self._read_file(relative_path)
            return pd.read_parquet(io.BytesIO(raw_bytes))
        except FileNotFoundError:
            logger.warning(f"SMB 远端 Parquet 文件不存在: {relative_path}")
            return None
        except Exception as exc:
            logger.warning(
                f"SMB 读取远端 Parquet 失败: path={relative_path}, "
                f"err={type(exc).__name__}: {exc}"
            )
            return None

    # ---- 内存缓存 ----

    def get_cached(self, relative_path: str, max_age_seconds: float) -> Optional[object]:
        import time
        entry = self._cache.get(relative_path)
        if entry is None:
            return None
        cached_at, data = entry
        if time.monotonic() - cached_at > max_age_seconds:
            return None
        return data

    def set_cache(self, relative_path: str, data: object) -> None:
        import time
        self._cache[relative_path] = (time.monotonic(), data)

    def clear_cache(self) -> None:
        self._cache.clear()


def parse_smb_url(url: str) -> dict:
    """解析 SMB URL 为连接参数字典。

    支持格式: //host/share/path | smb://host/share/path | \\\\host\\share\\path
    """
    url = str(url).strip()
    for prefix in ("smb://", "//", "\\\\"):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break
    parts = url.replace("\\", "/").split("/")
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"SMB URL 格式无效，需要至少 host/share: {url}")
    host = parts[0]
    share = parts[1]
    path = "/".join(parts[2:]) if len(parts) > 2 else ""
    return {"host": host, "share": share, "path": path}
