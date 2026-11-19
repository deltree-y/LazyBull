"""BaoStock 单连接守卫（服务方明确禁止并发连接访问）。

背景：2026-11-18 22:33 与 23:03 各做过一次"2 进程同时查询"的吞吐对比，此后匿名查询
长时间无返回（1 天窗口 >60 秒），符合"并发连接被判异常/限流"的表现；当日请求量仅约 438 次，
远低于 5 万/日上限，因此**不是**请求量触发的。

本模块提供一个**文件锁**：同一 campaign 下同一时刻只允许一个会话；锁文件含 PID 与时间，
超过 `stale_seconds` 视为残留可覆盖，避免进程被杀后永久卡死。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

LOCK_NAME = "baostock.lock"


class ConcurrentSessionError(RuntimeError):
    """检测到另一个 BaoStock 会话仍持有锁（禁止并发连接）。"""


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _alive(pid: int | None) -> bool:
    """尽力判断 PID 是否存活（POSIX 用 signal 0；Windows 用 os.kill 会抛错）。"""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, PermissionError):
        return False


def acquire(campaign: Path, *, owner: str = "unknown", stale_seconds: float = 3600.0) -> Path:
    """获取单连接锁；已被占用且未过期则抛 `ConcurrentSessionError`。返回锁文件路径。"""
    root = Path(campaign)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_NAME
    if path.exists():
        payload = _read(path)
        started = float(payload.get("started", 0.0) or 0.0)
        fresh = (time.time() - started) < float(stale_seconds)
        if fresh:
            raise ConcurrentSessionError(
                f"另一个 BaoStock 会话仍持有锁（pid={payload.get('pid')}，"
                f"started={payload.get('started_utc')}）；服务方禁止并发连接"
            )
    record = {
        "pid": os.getpid(),
        "owner": owner,
        "started": time.time(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    staging = path.with_suffix(".partial")
    staging.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    staging.replace(path)
    return path


def release(path: Path | None) -> None:
    """释放锁（仅当仍是本进程持有）。"""
    if path is None:
        return
    current = _read(Path(path))
    if int(current.get("pid") or -1) == os.getpid():
        try:
            Path(path).unlink()
        except OSError:
            pass


@contextmanager
def single_session(
    campaign: Path, *, stale_seconds: float = 3600.0, owner: str = "unknown"
) -> Iterator[Path]:
    """上下文版本：`with single_session(campaign): ...`。"""
    path = acquire(campaign, owner=owner, stale_seconds=stale_seconds)
    try:
        yield path
    finally:
        release(path)
