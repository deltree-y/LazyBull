# -*- coding: utf-8 -*-
"""PaperStateStore 单写入者写锁 mixin（P4 T2 复审处置拆分）。

进程内 RLock + 持锁线程所有权负责同实例互斥，O_EXCL 文件锁负责跨进程互斥；
活跃心跳刷新文件 mtime，避免整日任务被 stale 超时误抢占。异常类型由宿主类经
``_lock_error_type`` / ``_state_error_type`` 注入，保持既有公开异常身份不变。
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from loguru import logger

WRITE_LOCK_FILENAME = ".write.lock"


class PaperWriteLockMixin:
    """写锁职责族；self 为 PaperStateStore，路径/远端标记与异常类型由宿主提供。"""

    _lock_error_type: type[RuntimeError]
    _state_error_type: type[RuntimeError]

    def _init_write_lock(self, lock_timeout_seconds: float) -> None:
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._lock_depth = 0
        self._process_lock = threading.RLock()
        self._lock_owner_thread: Optional[int] = None
        self._lock_heartbeat_stop: Optional[threading.Event] = None
        self._lock_heartbeat_thread: Optional[threading.Thread] = None

    @property
    def _lock_path(self) -> Path:
        return self.state_path / WRITE_LOCK_FILENAME

    def acquire_write_lock(self) -> None:
        """获取写锁（同线程可重入；其他线程/进程冲突 ⇒ PaperLockError）。"""
        if self._is_remote:
            return
        if not self.state_path.exists():
            raise self._state_error_type(
                f"状态目录不存在（create_layout=False 只读形态禁止写入）: {self.state_path}"
            )
        if not self._process_lock.acquire(blocking=False):
            raise self._lock_error_type(
                f"写锁已被同一状态库的其他线程持有: {self._lock_path}"
                f"（owner_thread={self._lock_owner_thread}）；持锁期间禁止第二个写入者"
            )
        if self._lock_depth > 0:
            self._lock_depth += 1
            return
        try:
            self._lock_owner_thread = threading.get_ident()
            self._create_lock_file()
            self._lock_depth = 1
            self._start_lock_heartbeat()
        except Exception:
            self._lock_owner_thread = None
            self._lock_depth = 0
            self._process_lock.release()
            raise

    def release_write_lock(self) -> None:
        """释放写锁（仅持锁线程；深度归零时停心跳并删除锁文件）。"""
        if self._is_remote:
            return
        if self._lock_depth == 0 or self._lock_owner_thread != threading.get_ident():
            raise self._state_error_type(
                f"写锁未由当前线程持有却 release: {self._lock_path}"
                f"（owner_thread={self._lock_owner_thread}）"
            )
        if self._lock_depth > 1:
            self._lock_depth -= 1
            self._process_lock.release()
            return
        self._stop_lock_heartbeat()
        try:
            self._lock_path.unlink(missing_ok=True)
        finally:
            self._lock_depth = 0
            self._lock_owner_thread = None
            self._process_lock.release()

    @contextmanager
    def write_lock(self) -> Iterator[None]:
        """写锁上下文管理器（同实例仅持锁线程可重入）。"""
        self.acquire_write_lock()
        try:
            yield
        finally:
            self.release_write_lock()

    def _start_lock_heartbeat(self) -> None:
        """启动最外层文件锁心跳（活跃持锁期间持续刷新 mtime，防 stale 误抢占）。"""
        stop = threading.Event()
        interval = min(30.0, max(0.01, self._lock_timeout_seconds / 3.0))
        thread = threading.Thread(
            target=self._heartbeat_write_lock,
            args=(stop, interval),
            name=f"PaperStateStoreLockHeartbeat-{id(self):x}",
            daemon=True,
        )
        self._lock_heartbeat_stop = stop
        self._lock_heartbeat_thread = thread
        thread.start()

    def _heartbeat_write_lock(self, stop: threading.Event, interval: float) -> None:
        while not stop.wait(interval):
            try:
                os.utime(self._lock_path, None)
            except FileNotFoundError:
                logger.warning(f"活跃写锁文件被外部删除，心跳停止: {self._lock_path}")
                return
            except OSError as exc:
                logger.warning(f"活跃写锁心跳刷新失败，心跳停止: {self._lock_path}（{exc}）")
                return

    def _stop_lock_heartbeat(self) -> None:
        stop = self._lock_heartbeat_stop
        thread = self._lock_heartbeat_thread
        self._lock_heartbeat_stop = None
        self._lock_heartbeat_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _create_lock_file(self) -> None:
        holder = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": datetime.now().isoformat(timespec="seconds"),
        }
        for attempt in range(2):
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                self._handle_existing_lock(attempt)
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(holder, fh, ensure_ascii=False)
                return
        raise self._lock_error_type(f"写锁抢占后仍无法创建: {self._lock_path}")

    def _handle_existing_lock(self, attempt: int) -> None:
        holder = self._read_lock_holder()
        try:
            lock_mtime = self._lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        age_seconds = time.time() - lock_mtime
        if age_seconds < self._lock_timeout_seconds:
            raise self._lock_error_type(
                f"写锁已被持有: {self._lock_path}（holder={holder}，"
                f"锁龄 {age_seconds:.1f}s < 超时 {self._lock_timeout_seconds}s）；"
                "持锁期间禁止第二个写入者"
            )
        logger.warning(
            f"检测到 stale 写锁（holder={holder}，锁龄 {age_seconds:.1f}s ≥ "
            f"超时 {self._lock_timeout_seconds}s），抢占删除重建: {self._lock_path}"
        )
        self._lock_path.unlink(missing_ok=True)
        if attempt >= 1:
            raise self._lock_error_type(f"写锁 stale 清理后重试仍失败: {self._lock_path}")

    def _read_lock_holder(self) -> dict[str, Any]:
        try:
            data = json.loads(self._lock_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"写锁文件读取失败（按无 holder 信息处理）: {self._lock_path}（{exc}）")
            return {}
