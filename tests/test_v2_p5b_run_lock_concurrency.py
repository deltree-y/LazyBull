# -*- coding: utf-8 -*-
"""R2-P5B-02 反例：互斥在并发下不得出现两个持有者；前任退出不得破坏后继锁。

设计（本轮整改后）：互斥由 **OS 文件锁**（跨进程）+ **进程内互斥注册表**（同进程多 fd）
共同保证；因此不存在 stale 接管，也没有「按路径删除他人锁」的窗口。
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t2_env as t2env  # noqa: E402

from src.lazybull.v2.hosts.testbed.layout import layout_for  # noqa: E402
from src.lazybull.v2.hosts.testbed.runner import (  # noqa: E402
    RUN_LOCK_FILENAME,
    CampaignBusyError,
    CampaignRunLock,
)


def _layout(tmp_path: Path):
    env = t2env.build_env(tmp_path)
    prepared = t2env.prepare_campaign(
        env,
        campaign_id="lock_race",
        arms=("base",),
        roles={"base": "aa_clone"},
    )
    return layout_for(prepared.root)


def test_no_two_holders_at_same_time(tmp_path):
    """并发争抢：任何时刻至多一个持有者（允许先后串行获取）。"""
    layout = _layout(tmp_path)
    active = {"now": 0, "max": 0}
    guard = threading.Lock()
    errors: list[str] = []
    barrier = threading.Barrier(4)

    def worker() -> None:
        barrier.wait()
        try:
            with CampaignRunLock(layout, timeout_seconds=20.0):
                with guard:
                    active["now"] += 1
                    active["max"] = max(active["max"], active["now"])
                time.sleep(0.1)
                with guard:
                    active["now"] -= 1
        except CampaignBusyError as exc:  # 超时未取到也算正常（未进入临界区）
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert active["max"] == 1, f"出现并发持有者：max={active['max']}"


def test_predecessor_exit_keeps_lock_file(tmp_path):
    """前任退出不得删除锁文件（锁文件承载 OS 锁位，删除会把互斥交给别人）。"""
    layout = _layout(tmp_path)
    lock_path = layout.journal_root / RUN_LOCK_FILENAME
    predecessor = CampaignRunLock(layout, timeout_seconds=5.0)
    predecessor.__enter__()
    assert lock_path.is_file()
    predecessor.__exit__(None, None, None)
    assert lock_path.is_file(), "前任退出删除了锁文件"
    holder = lock_path.with_name(f"{lock_path.name}.holder")
    if holder.is_file():
        payload = json.loads(holder.read_text(encoding="utf-8"))
        assert payload["token"] == predecessor.token


def test_live_holder_blocks_and_releases(tmp_path):
    """存活持有者必须阻塞他人（超时报错）；释放后可再取。"""
    layout = _layout(tmp_path)
    first = CampaignRunLock(layout, timeout_seconds=5.0)
    first.__enter__()
    try:
        blocked = False
        try:
            CampaignRunLock(layout, timeout_seconds=0.3).__enter__()
        except CampaignBusyError:
            blocked = True
        assert blocked is True
    finally:
        first.__exit__(None, None, None)
    with CampaignRunLock(layout, timeout_seconds=5.0):
        assert (layout.journal_root / RUN_LOCK_FILENAME).is_file()


def test_dead_process_lock_is_released_by_os(tmp_path):
    """持有进程死亡 ⇒ OS 自动释放锁，无需任何 stale 接管。"""
    layout = _layout(tmp_path)
    lock_path = layout.journal_root / RUN_LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": 999999999, "host": socket.gethostname(), "token": "dead"}),
        encoding="utf-8",
    )
    with CampaignRunLock(layout, timeout_seconds=5.0):
        assert lock_path.is_file()
