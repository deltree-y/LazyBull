#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SMB 远端文件读取客户端。

通过系统 smbclient 命令（SMB2/3 协议）从群晖 NAS 读取 paper 数据。
可靠性优于纯 Python SMB 库（自动处理 SPNEGO/Kerberos/NTLM 认证协商）。

依赖：树莓派需安装 smbclient（sudo apt install smbclient）。
"""

import io
import json
import os
import subprocess
import tempfile
from typing import Optional

import pandas as pd
from loguru import logger


class SMBFileReader:
    """SMB 远端文件只读客户端（基于 smbclient 命令行）。

    每次调用独立执行 smbclient 命令。

    Usage:
        reader = SMBFileReader("192.168.1.21", "docker", "lazybull/data/paper")
        account = reader.read_json("state/account.json")
    """

    def __init__(
        self,
        host: str,
        share: str,
        path_prefix: str = "",
        username: str = "",
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

    @property
    def _smb_url(self) -> str:
        """构造 SMB URL。"""
        return f"//{self.host}/{self.share}"

    @property
    def _auth_arg(self) -> str:
        """构造 smbclient 认证参数。"""
        if self.username:
            return f"{self.username}%{self.password}"
        return "guest%"

    def _build_remote_path(self, relative_path: str) -> str:
        """构建共享内的相对路径。"""
        clean = str(relative_path).replace("\\", "/").strip("/")
        if self.path_prefix:
            return f"{self.path_prefix}/{clean}"
        return clean

    def _read_file_raw(self, relative_path: str) -> bytes:
        """通过 smbclient 读取远端文件，写入临时文件后读回。

        Raises:
            FileNotFoundError: 文件不存在
            ConnectionError: SMB 连接或认证失败
        """
        remote_path = self._build_remote_path(relative_path)
        tmp_path = None

        try:
            # 创建临时文件
            tmp_fd, tmp_path = tempfile.mkstemp(prefix="lazybull_smb_")
            os.close(tmp_fd)

            # 构造 smbclient 命令
            # smbclient -U 'user%pass' //host/share -c 'get "remote/path" "local/path"'
            cmd = [
                "smbclient",
                self._smb_url,
                "-U", self._auth_arg,
                "-p", str(self.port),
                "-c", f'get "{remote_path}" "{tmp_path}"',
            ]

            # 执行
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "LANG": "C"},  # 英文输出便于解析
            )

            stderr_lower = result.stderr.lower() if result.stderr else ""

            if result.returncode != 0:
                # 区分错误类型
                if any(kw in stderr_lower for kw in (
                    "nt_status_object_name_not_found",
                    "does not exist",
                    "no such file",
                    "errno 2",
                )):
                    raise FileNotFoundError(
                        f"SMB 文件不存在: share={self.share}, path={remote_path}"
                    )
                elif any(kw in stderr_lower for kw in (
                    "nt_status_access_denied",
                    "nt_status_logon_failure",
                    "nt_status_account",
                    "session setup failed",
                    "nt_status_password_expired",
                )):
                    raise ConnectionError(
                        f"SMB 认证失败: host={self.host}, share={self.share}, "
                        f"user={self.username or 'guest'}, "
                        f"stderr={result.stderr.strip()[:200]}"
                    )
                else:
                    raise ConnectionError(
                        f"SMB 读取失败: share={self.share}, path={remote_path}, "
                        f"rc={result.returncode}, "
                        f"stderr={result.stderr.strip()[:300]}"
                    )

            # 读取临时文件
            with open(tmp_path, "rb") as f:
                return f.read()

        except subprocess.TimeoutExpired:
            raise ConnectionError(
                f"SMB 连接超时: host={self.host}, path={remote_path}, "
                f"timeout={self.timeout}s"
            )
        except (FileNotFoundError, ConnectionError):
            raise
        except Exception as exc:
            raise ConnectionError(
                f"SMB 读取异常: share={self.share}, path={remote_path}, "
                f"err={type(exc).__name__}: {exc}"
            ) from exc
        finally:
            # 清理临时文件
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ---- 公共读取方法 ----

    def _read_file(self, relative_path: str) -> bytes:
        """读取原始字节（便于测试 mock）。"""
        return self._read_file_raw(relative_path)

    def file_exists(self, relative_path: str) -> bool:
        """检查远端文件是否存在。"""
        try:
            self._read_file_raw(relative_path)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def read_text(self, relative_path: str, encoding: str = "utf-8") -> str:
        return self._read_file(relative_path).decode(encoding)

    def read_json(self, relative_path: str) -> dict:
        raw = self.read_text(relative_path)
        if not raw.strip():
            raise ValueError(
                f"SMB 远端文件为空: {self._build_remote_path(relative_path)}"
            )
        return json.loads(raw)

    def read_yaml(self, relative_path: str) -> dict:
        import yaml
        raw = self.read_text(relative_path)
        result = yaml.safe_load(raw)
        if result is None:
            raise ValueError(
                f"SMB 远端 YAML 文件为空: {self._build_remote_path(relative_path)}"
            )
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
    return {"host": parts[0], "share": parts[1], "path": "/".join(parts[2:]) if len(parts) > 2 else ""}
