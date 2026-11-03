# -*- coding: utf-8 -*-
"""R2-P5B-01 反例：panel 合法重写 与 冻结配置变更 必须在发布前被拒。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t2_env as t2env  # noqa: E402

from src.lazybull.v2.hosts.testbed.arm_runtime import load_arm_specs  # noqa: E402
from src.lazybull.v2.hosts.testbed.identity import (  # noqa: E402
    collect_data_state_identity,
)
from src.lazybull.v2.hosts.testbed.paths import TestbedError  # noqa: E402


def test_panel_partition_rewrite_changes_window_identity(tmp_path):
    """panel 分区字节变化（正常重写入口）必须改变 window_state_id。"""
    env = t2env.build_env(tmp_path)
    days = list(env.days[:3])
    before = collect_data_state_identity(data_root=str(env.data_root), virtual_days=days)
    assert before["panel_covered"] is True
    partition = Path(env.data_root) / "features" / "panel" / days[0] / "core.parquet"
    assert partition.is_file()
    partition.write_bytes(partition.read_bytes() + b"\x00")
    after = collect_data_state_identity(data_root=str(env.data_root), virtual_days=days)
    assert after["window_state_id"] != before["window_state_id"]


def test_missing_panel_partition_in_window_is_rejected(tmp_path):
    """窗口内缺 panel 分区 ⇒ 拒绝（宿主实际消费该输入）。"""
    env = t2env.build_env(tmp_path)
    days = list(env.days[:3])
    (Path(env.data_root) / "features" / "panel" / days[1] / "core.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        collect_data_state_identity(data_root=str(env.data_root), virtual_days=days)


def test_frozen_config_change_is_rejected_before_run(tmp_path):
    """注册后改动 config.json（如 top_n）⇒ 运行前复算指纹不符，拒绝加载臂规格。"""
    env = t2env.build_env(tmp_path)
    prepared = t2env.prepare_campaign(
        env,
        campaign_id="drift_cfg",
        arms=("base",),
        roles={"base": "aa_clone"},
    )
    config_path = prepared.root / "registered" / "arms" / "base" / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["top_n"] = int(payload.get("top_n") or 5) + 1
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(TestbedError) as excinfo:
        load_arm_specs(prepared.root)
    assert "冻结配置已变更" in str(excinfo.value)


def test_untouched_config_passes(tmp_path):
    """未改动配置时正常通过（证明断言非真空）。"""
    env = t2env.build_env(tmp_path)
    prepared = t2env.prepare_campaign(
        env,
        campaign_id="drift_ok",
        arms=("base",),
        roles={"base": "aa_clone"},
    )
    specs = load_arm_specs(prepared.root)
    assert "base" in specs and specs["base"].config_flat


def test_other_feature_family_change_changes_window_identity(tmp_path):
    """其他特征族（如 fundamental）变化必须改变 window_state_id（R2-P5B-01）。"""
    env = t2env.build_env(tmp_path)
    days = list(env.days[:3])
    panel_dir = Path(env.data_root) / "features" / "panel" / days[0]
    fundamental = panel_dir / "fundamental.parquet"
    fundamental.write_bytes(b"assets_turn,1\n")
    before = collect_data_state_identity(data_root=str(env.data_root), virtual_days=days)
    fundamental.write_bytes(b"assets_turn,100\n")
    after = collect_data_state_identity(data_root=str(env.data_root), virtual_days=days)
    assert after["window_state_id"] != before["window_state_id"]


def test_cold_archive_partition_change_changes_window_identity(tmp_path):
    """冷区 archive 分区（月份目录）变化必须改变 window_state_id（R2-P5B-01）。"""
    env = t2env.build_env(tmp_path)
    days = list(env.days[:3])
    month = f"{days[0][:4]}-{days[0][4:6]}"
    archive = Path(env.data_root) / "features" / "panel_archive" / month
    archive.mkdir(parents=True, exist_ok=True)
    core = archive / "core.parquet"
    core.write_bytes(b"bp,0.01\n")
    before = collect_data_state_identity(data_root=str(env.data_root), virtual_days=days)
    core.write_bytes(b"bp,1.0\n")
    after = collect_data_state_identity(data_root=str(env.data_root), virtual_days=days)
    assert after["window_state_id"] != before["window_state_id"]
