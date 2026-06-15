#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SMB 远端文件读取客户端。

为 LCD35 从群晖 NAS 读取 paper 数据提供轻量 SMB 协议访问。
设计原则：每交易日仅读取一次，无需长连接复用。
"""

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class SMBFileReader:
    """SMB 远端文件只读客户端。

    每次调用独立建立连接→读取→断开，不使用连接池。
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
        """初始化 SMB 读取器。

        Args:
            host: SMB 服务器 IP 或主机名
            share: SMB 共享名
            path_prefix: 共享内的子路径（如 lazybull/data/paper）
            username: SMB 用户名，默认 guest
            password: SMB 密码
            port: SMB 端口，默认 445
            timeout: 连接超时秒数
        """
        self.host = host
        self.share = share
        self.path_prefix = str(path_prefix).replace("\\", "/").strip("/")
        self.username = username
        self.password = password
        self.port = port
        self.timeout = timeout

        # 内存缓存：{relative_path: (timestamp, data)}
        self._cache: dict[str, tuple[float, object]] = {}

    def _build_remote_path(self, relative_path: str) -> str:
        """构建 SMB 服务名内的完整路径。"""
        clean = str(relative_path).replace("\\", "/").strip("/")
        if self.path_prefix:
            return f"{self.path_prefix}/{clean}"
        return clean

    def _connect(self) -> object:
        """建立 SMB 连接。

        Returns:
            SMBConnection 实例
        """
        from smb.SMBConnection import SMBConnection

        conn = SMBConnection(
            self.username,
            self.password,
            "lazybull-respi",
            self.host,
            use_ntlm_v2=True,
            is_direct_tcp=True,
        )
        connected = conn.connect(self.host, self.port, timeout=self.timeout)
        if not connected:
            raise ConnectionError(
                f"SMB 连接失败: host={self.host}, port={self.port}, share={self.share}"
            )
        return conn

    def _read_file(self, relative_path: str) -> bytes:
        """从 SMB 读取文件原始字节。

        Args:
            relative_path: 相对于 path_prefix 的文件路径

        Returns:
            文件字节内容

        Raises:
            FileNotFoundError: 文件不存在
            ConnectionError: SMB 连接或读取失败
        """
        remote_path = self._build_remote_path(relative_path)
        try:
            conn = self._connect()
        except Exception as exc:
            raise ConnectionError(
                f"SMB 连接失败: host={self.host}, share={self.share}, "
                f"err={type(exc).__name__}: {exc}"
            ) from exc
        try:
            buf = io.BytesIO()
            conn.retrieveFile(self.share, remote_path, buf)
            return buf.getvalue()
        except Exception as exc:
            raise FileNotFoundError(
                f"SMB 文件不存在或读取失败: share={self.share}, "
                f"path={remote_path}, err={type(exc).__name__}: {exc}"
            ) from exc
        finally:
            conn.close()

    def file_exists(self, relative_path: str) -> bool:
        """检查远端文件是否存在。"""
        remote_path = self._build_remote_path(relative_path)
        conn = None
        try:
            conn = self._connect()
            # 尝试获取文件属性来判断存在性
            conn.getAttributes(self.share, remote_path)
            return True
        except Exception:
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def read_text(self, relative_path: str, encoding: str = "utf-8") -> str:
        """读取文本文件内容。

        Raises:
            FileNotFoundError: 文件不存在
            ConnectionError: SMB 连接失败
        """
        return self._read_file(relative_path).decode(encoding)

    def read_json(self, relative_path: str) -> dict:
        """读取 JSON 文件并解析为字典。

        Returns:
            解析后的字典

        Raises:
            FileNotFoundError: 文件不存在
            ConnectionError: SMB 连接失败
            ValueError: JSON 解析失败
        """
        raw = self.read_text(relative_path)
        if not raw.strip():
            raise ValueError(f"SMB 远端文件为空: {self._build_remote_path(relative_path)}")
        return json.loads(raw)

    def read_yaml(self, relative_path: str) -> dict:
        """读取 YAML 文件并解析为字典。

        Returns:
            解析后的字典

        Raises:
            FileNotFoundError: 文件不存在
            ConnectionError: SMB 连接失败
        """
        import yaml

        raw = self.read_text(relative_path)
        result = yaml.safe_load(raw)
        if result is None:
            raise ValueError(f"SMB 远端 YAML 文件为空: {self._build_remote_path(relative_path)}")
        return result

    def read_parquet(self, relative_path: str) -> Optional[pd.DataFrame]:
        """读取 Parquet 文件为 DataFrame。

        注意：SMB 不支持流式读取 Parquet，需先下载到临时文件。

        Returns:
            DataFrame，文件不存在或读取失败时返回 None
        """
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

    def get_cached(self, relative_path: str, max_age_seconds: float) -> Optional[object]:
        """读取内存缓存（按过期时间过滤）。

        Args:
            relative_path: 缓存键
            max_age_seconds: 最大缓存有效期（秒）

        Returns:
            缓存数据，过期或不存在返回 None
        """
        import time

        entry = self._cache.get(relative_path)
        if entry is None:
            return None
        cached_at, data = entry
        if time.monotonic() - cached_at > max_age_seconds:
            return None
        return data

    def set_cache(self, relative_path: str, data: object) -> None:
        """写入内存缓存。"""
        import time

        self._cache[relative_path] = (time.monotonic(), data)

    def clear_cache(self) -> None:
        """清空全部缓存。"""
        self._cache.clear()


def parse_smb_url(url: str) -> dict:
    """解析 SMB URL 为连接参数字典。

    支持格式:
        - //host/share/path
        - smb://host/share/path
        - \\\\host\\share\\path

    Returns:
        {"host": str, "share": str, "path": str}

    Raises:
        ValueError: URL 格式无效
    """
    url = str(url).strip()

    # 去掉协议前缀
    for prefix in ("smb://", "//", "\\\\"):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break

    # 按 / 或 \\ 分割
    parts = url.replace("\\", "/").split("/")
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) < 2:
        raise ValueError(
            f"SMB URL 格式无效，需要至少 host/share: {url}"
        )

    host = parts[0]
    share = parts[1]
    path = "/".join(parts[2:]) if len(parts) > 2 else ""

    return {"host": host, "share": share, "path": path}
