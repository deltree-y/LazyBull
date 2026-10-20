# -*- coding: utf-8 -*-
"""v2 store 单机写锁（带 holder 信息、stale 抢占与同实例重入）。"""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from loguru import logger

__all__ = ["_WriteLock"]


class _WriteLock:
    """单机写锁（``features/.write.lock``，带 holder 信息 + 获取时间）。

    - 持锁期间第二个实例操作 ⇒ RuntimeError；
    - stale 检测：holder 进程不存在则抢占并告警（loguru warning）；
    - 对同一 PanelDataStore 实例可重入（深度计数）。
    """

    def __init__(self, lock_path: Path) -> None:
        self._path = lock_path
        self._depth = 0

    def acquire(self) -> None:
        if self._depth > 0:
            self._depth += 1
            return
        self._create_lock_file()
        self._depth = 1

    def release(self, during_exception: bool = False) -> None:
        if self._depth == 0:
            raise RuntimeError(f"写锁未持有却 release: {self._path}")
        self._depth -= 1
        if self._depth == 0:
            self._unlink_lock_file(during_exception)

    def _unlink_lock_file(self, during_exception: bool) -> None:
        """删除锁文件（有限退避重试 3 次）。

        Windows 瞬时占用（PermissionError）不掩盖锁内原始异常：锁内已有异常在飞
        ⇒ 降级 warning；无异常在飞且重试仍失败 ⇒ 抛 RuntimeError。
        """
        last_exc: OSError | None = None
        for attempt in range(3):
            try:
                self._path.unlink(missing_ok=True)
                return
            except OSError as exc:
                last_exc = exc
                time.sleep(0.1 * (attempt + 1))
        if during_exception:
            logger.warning(
                f"写锁文件删除失败（锁内已有异常在飞，降级不掩盖）: {self._path}（{last_exc}）"
            )
            return
        raise RuntimeError(f"写锁文件删除连续失败（3 次）: {self._path}") from last_exc

    def __enter__(self) -> "_WriteLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release(during_exception=exc_type is not None)

    def _create_lock_file(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        holder = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": datetime.now().isoformat(timespec="seconds"),
        }
        for attempt in range(2):
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                self._handle_existing_lock(attempt)
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(holder, f, ensure_ascii=False)
                return
        raise RuntimeError(f"写锁抢占后仍无法创建: {self._path}")

    def _handle_existing_lock(self, attempt: int) -> None:
        holder = self._read_holder()
        pid = holder.get("pid") if holder else None
        if isinstance(pid, int) and psutil.pid_exists(pid):
            raise RuntimeError(
                f"写锁已被持有: {self._path}（holder={holder}）；持锁期间禁止第二个实例写入"
            )
        logger.warning(
            f"检测到 stale 写锁（holder={holder}），原 holder 进程不存在，抢占: {self._path}"
        )
        self._path.unlink(missing_ok=True)
        if attempt >= 1:
            raise RuntimeError(f"写锁 stale 清理后重试仍失败: {self._path}")

    def _read_holder(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"写锁文件读取失败（按 stale 处理）: {self._path}（{exc}）")
            return {}
