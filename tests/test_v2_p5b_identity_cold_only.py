# -*- coding: utf-8 -*-
"""第三轮复审 P1 反例：纯冷区（archive-only）布局的身份覆盖；R4-01：锁获取失败可恢复。"""

from __future__ import annotations

import errno
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t2_env as t2env  # noqa: E402

from src.lazybull.v2.hosts.testbed.identity import (  # noqa: E402
    collect_data_state_identity,
)
from src.lazybull.v2.hosts.testbed.layout import layout_for  # noqa: E402
from src.lazybull.v2.hosts.testbed.runner import CampaignRunLock  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.panel_builder import bootstrap_manifest  # noqa: E402


def _cold_only_root(tmp_path: Path) -> tuple[Path, list[str]]:
    """构造只有 ``features/panel_archive`` 的合法布局（无热区根）。"""
    days = list(pd.bdate_range("2026-08-03", periods=3).strftime("%Y%m%d"))
    root = tmp_path / "data"
    t2env.write_market_data(root, days)
    t2env.write_models(root)
    store = PanelDataStore(root)
    bootstrap_manifest(store)
    core = pd.concat([t2env._feature_frame(day) for day in days], ignore_index=True)
    fundamental = core[["trade_date", "ts_code"]].copy()
    fundamental["assets_turn"] = 1.0
    store.append_archive_features("2026-08", "core", core)
    store.append_archive_features("2026-08", "fundamental", fundamental)
    assert not (root / "features" / "panel").exists()
    return root, days


def test_archive_only_layout_is_covered_by_identity(tmp_path):
    """纯冷区布局也必须登记身份（panel_covered=True 且含 archive 分区）。"""
    root, days = _cold_only_root(tmp_path)
    assert not (root / "features" / "panel").is_dir()
    identity = collect_data_state_identity(data_root=str(root), virtual_days=days)
    assert identity["panel_covered"] is True
    paths = {item["path"] for item in identity["window_files"]}
    assert any("panel_archive" in path for path in paths)


def test_archive_only_rewrite_changes_window_identity(tmp_path):
    """纯冷区期间改动 archive 分区 ⇒ window_state_id 必变（禁止沿用旧身份发布）。"""
    root, days = _cold_only_root(tmp_path)
    before = collect_data_state_identity(data_root=str(root), virtual_days=days)
    target = root / "features" / "panel_archive" / f"{days[0][:4]}-{days[0][4:6]}" / "core.parquet"
    target.write_bytes(b"core.parquet:rewritten\n")
    after = collect_data_state_identity(data_root=str(root), virtual_days=days)
    assert after["window_state_id"] != before["window_state_id"]


def test_lock_enter_recovers_after_holder_write_failure(tmp_path, monkeypatch):
    """R4-01：诊断写入失败必须回滚 mutex/fd，故障解除后同进程重试成功。"""
    env = t2env.build_env(tmp_path)
    prepared = t2env.prepare_campaign(
        env, campaign_id="lock_retry", arms=("base",), roles={"base": "aa_clone"}
    )
    layout = layout_for(prepared.root)
    original = Path.write_bytes
    state = {"raised": False}

    def flaky(self, payload):  # noqa: ANN001 - 与 Path.write_bytes 同签名
        if self.name.endswith(".holder") and not state["raised"]:
            state["raised"] = True
            raise OSError(errno.ENOSPC, "simulated")
        return original(self, payload)

    monkeypatch.setattr(Path, "write_bytes", flaky)
    with pytest.raises(OSError):
        CampaignRunLock(layout, timeout_seconds=1.0).__enter__()
    # 故障已解除：同一进程内必须能重新取得锁
    with CampaignRunLock(layout, timeout_seconds=2.0) as lock:
        assert lock.acquired is True
