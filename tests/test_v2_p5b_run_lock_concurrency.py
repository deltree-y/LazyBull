# -*- coding: utf-8 -*-
"""R2-P5B-02 反例：锁接管必须串行化，且前任退出不得删除后继锁。"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t2_env as t2env  # noqa: E402

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
    from src.lazybull.v2.hosts.testbed.layout import layout_for

    return layout_for(prepared.root)


def _write_stale(path: Path) -> None:
    """写入一个“已死持有者”的锁（pid 取一个必然不存在的值）。"""
    payload = {"pid": 999999999, "host": socket.gethostname(), "token": "dead", "acquired_at": "x"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_two_recoverers_only_one_acquires(tmp_path):
    """两个恢复者同时接管死锁 ⇒ 至多一个进入临界区。"""
    layout = _layout(tmp_path)
    lock_path = layout.journal_root / RUN_LOCK_FILENAME
    _write_stale(lock_path)
    outcome: list[str] = []
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        barrier.wait()
        try:
            with CampaignRunLock(layout, timeout_seconds=10.0):
                outcome.append(name)
                import time as _time

                _time.sleep(0.2)
        except CampaignBusyError:
            return

    threads = [threading.Thread(target=worker, args=(name,)) for name in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(outcome) == 1, f"期望恰好一个持有者，实际 {outcome}"


def test_predecessor_exit_does_not_delete_successor_lock(tmp_path):
    """前任退出不得删除后继锁（释放必须绑定所有权）。"""
    layout = _layout(tmp_path)
    lock_path = layout.journal_root / RUN_LOCK_FILENAME
    predecessor = CampaignRunLock(layout)
    predecessor.__enter__()
    # 模拟“后继接管”：直接替换锁内容为另一个令牌
    successor_token = "successor-token"
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "token": successor_token,
                "acquired_at": "y",
            }
        ),
        encoding="utf-8",
    )
    predecessor.__exit__(None, None, None)
    assert lock_path.is_file(), "前任退出删除了后继锁"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["token"] == successor_token


def test_live_holder_blocks_and_releases(tmp_path):
    """存活持有者必须阻塞他人；正常释放后他人可取。"""
    layout = _layout(tmp_path)
    first = CampaignRunLock(layout)
    first.__enter__()
    try:
        try:
            CampaignRunLock(layout).__enter__()
        except CampaignBusyError:
            blocked = True
        else:  # pragma: no cover
            blocked = False
        assert blocked is True
    finally:
        first.__exit__(None, None, None)
    with CampaignRunLock(layout):
        assert (layout.journal_root / RUN_LOCK_FILENAME).is_file()
