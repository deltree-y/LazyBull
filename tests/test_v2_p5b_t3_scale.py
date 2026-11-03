# -*- coding: utf-8 -*-
"""P5b T3 规模与长期行为测试（tests/ 全量豁免复杂度）。

覆盖：≥40 个连续虚拟日（跨多代）下分片 / 链 / manifest / index 的增量、审计与修复耗时、
**线性**读取结构（审计分片读取与「幂等重跑」的 journal 全量加载次数均随日数线性；首次逐日发布
路径仍按日重建派生投影，属已登记的 T4/T6 性能义务）、staging 残留与回收（T2 遗留项 1 的处置
证据）、单批墙钟登记。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t3_env as envmod  # noqa: E402

from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    audit_ledger,
    layout_for,
    prune_staging,
    repair_ledger,
)
from src.lazybull.v2.hosts.testbed import journal as journal_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import ledger_audit as ledger_audit_module  # noqa: E402
from src.lazybull.v2.hosts.testbed import runner as runner_module  # noqa: E402

#: 规模窗口（设计 §4.3「长期分片账本」；≥40 连续虚拟日、跨多代）
SCALE_DAYS = 40
#: 小窗口（线性读取结构的对照）
SMALL_DAYS = 10


def _publish_all(env, prepared, days):
    started = time.perf_counter()
    report = envmod.run(env, prepared, list(days))
    return report, time.perf_counter() - started


def _measure_audit(root, campaign_id, days_count, monkeypatch):
    """审计一次并统计分片读取次数（只读，不改现场）。"""
    reads = {"count": 0}
    original_read = ledger_audit_module._read_shard

    def counting_read(path):
        reads["count"] += 1
        return original_read(path)

    monkeypatch.setattr(ledger_audit_module, "_read_shard", counting_read)
    started = time.perf_counter()
    audit = audit_ledger(layout_for(root), campaign_id)
    seconds = time.perf_counter() - started
    monkeypatch.undo()
    return {
        "seconds": seconds,
        "reads": reads["count"],
        "days": days_count,
        "links": audit["chain"]["arms"]["aa_1"]["link_count"],
    }


@pytest.fixture(scope="module")
def scale_campaign(tmp_path_factory):
    """40 日 2 臂现场只构建一次：两个规模用例都只读消费它。

    D0 fresh 与逐日回放是主要固定成本（40 日 × 2 臂 ≈ 55s），重复构建不增加证据。
    """
    window = envmod.scale_days(SCALE_DAYS)
    tmp = tmp_path_factory.mktemp("scale")
    env = envmod.build_env(tmp, days=window[:SCALE_DAYS], calendar=window)
    prepared = envmod.prepare(env, arms=("aa_1", "aa_2"))
    report, run_seconds = _publish_all(env, prepared, window[:SCALE_DAYS])
    return SimpleNamespace(
        env=env,
        prepared=prepared,
        layout=layout_for(prepared.root),
        window=window,
        report=report,
        run_seconds=run_seconds,
    )


def test_40_virtual_day_ledger_is_linear_and_auditable(scale_campaign, monkeypatch):
    window = scale_campaign.window
    layout = scale_campaign.layout
    report, run_seconds = scale_campaign.report, scale_campaign.run_seconds

    assert report["published"]["published_generation"] == SCALE_DAYS
    assert len(report["transactions"]) == SCALE_DAYS
    assert run_seconds < runner_module.LONG_TASK_THRESHOLD_SECONDS

    # 结构增量：每臂每日恰一片、链节与 manifest/index 条目数一致
    shard_counts = {
        arm: len(list((layout.ledger_root / arm).glob("*.json"))) - 1  # 去掉 manifest.json
        for arm in ("aa_1", "aa_2")
    }
    assert shard_counts == {"aa_1": SCALE_DAYS, "aa_2": SCALE_DAYS}

    audit = audit_ledger(layout, envmod.CAMPAIGN_ID, clock_days=list(window[:SCALE_DAYS]))
    assert audit["three_way"]["consistent"] is True
    assert audit["status"] == "PASS"
    for arm in ("aa_1", "aa_2"):
        assert audit["arms"][arm]["journal_days"] == SCALE_DAYS
        assert audit["chain"]["arms"][arm]["link_count"] == SCALE_DAYS
    assert audit["published"]["published_generation"] == SCALE_DAYS

    # 线性读取结构：审计的分片读取次数 <= 2 × 日数 × 臂数（无重复扫描/无 O(n²)）
    reads = {"count": 0}
    original_read = ledger_audit_module._read_shard

    def counting_read(path):
        reads["count"] += 1
        return original_read(path)

    monkeypatch.setattr(ledger_audit_module, "_read_shard", counting_read)
    started = time.perf_counter()
    audit = audit_ledger(layout, envmod.CAMPAIGN_ID, clock_days=list(window[:SCALE_DAYS]))
    audit_seconds = time.perf_counter() - started
    assert reads["count"] <= 2 * SCALE_DAYS * 2
    assert audit_seconds < 20.0

    dry = repair_ledger(
        layout, envmod.CAMPAIGN_ID, apply=False, clock_days=list(window[:SCALE_DAYS])
    )
    assert dry["planned"] == []
    assert dry["blocked"] is False
    assert dry["seconds"] < 20.0

    # T2 遗留项 1：整臂复制残留由运行收尾回收（收尾实测残量 + 回收后为 0）
    staging = (report.get("machine_time") or {}).get("staging_pruned") or {}
    assert staging["bytes_before"] > 0
    assert staging["bytes_after"] == 0
    assert staging["attempts_removed"] == SCALE_DAYS
    prune_dry = prune_staging(layout, apply=False)
    assert prune_dry["files_before"] == 0
    assert not any(path.is_dir() for path in layout.staging_root.glob("*/*"))

    # 规模与机器墙钟（单批远低于 20 分钟授权阈值）
    assert run_seconds / SCALE_DAYS < 5.0
    assert (report.get("machine_time") or {}).get("long_task_threshold_exceeded") is False


def test_recovery_scan_does_not_reload_all_journals(tmp_path, monkeypatch):
    """幂等重跑时不得对每个已发布 attempt 重做全量 journal 加载（旧的 O(A·N²)）。"""
    days = envmod.scale_days(SMALL_DAYS)
    env = envmod.build_env(tmp_path, days=days[:SMALL_DAYS], calendar=days)
    prepared = envmod.prepare(env, arms=("aa_1", "aa_2"))
    envmod.run(env, prepared, days[:SMALL_DAYS])

    calls = {"count": 0}
    original = journal_module.load_all_attempts

    def counting(layout):
        calls["count"] += 1
        return original(layout)

    monkeypatch.setattr(journal_module, "load_all_attempts", counting)
    envmod.run(env, prepared, days[:SMALL_DAYS])
    # 一次幂等重跑只需常数次全量加载（预检审计 + 收尾审计），与日数/臂数无关
    assert calls["count"] <= 8, f"幂等重跑触发了 {calls['count']} 次全量 journal 加载"


def test_audit_cost_grows_sub_quadratically(tmp_path, scale_campaign, monkeypatch):
    """10 日 vs 40 日：分片读取次数严格线性增长（4 倍日数 ⇒ ≤4 倍读取）。"""
    small_window = envmod.scale_days(SMALL_DAYS)
    small_env = envmod.build_env(
        tmp_path / "small", days=small_window[:SMALL_DAYS], calendar=small_window
    )
    small_prepared = envmod.prepare(small_env, campaign_id="t3_small", arms=("aa_1", "aa_2"))
    envmod.run(small_env, small_prepared, small_window[:SMALL_DAYS])

    measurements = {
        "small": _measure_audit(small_prepared.root, "t3_small", SMALL_DAYS, monkeypatch),
        # 40 日对照复用 scale_campaign 现场（只读审计，重复构建不增加证据）
        "large": _measure_audit(
            scale_campaign.prepared.root, envmod.CAMPAIGN_ID, SCALE_DAYS, monkeypatch
        ),
    }

    ratio_days = measurements["large"]["days"] / measurements["small"]["days"]
    ratio_reads = measurements["large"]["reads"] / max(1, measurements["small"]["reads"])
    assert measurements["large"]["links"] == SCALE_DAYS
    assert measurements["small"]["links"] == SMALL_DAYS
    assert ratio_reads <= ratio_days + 0.5
    assert ratio_reads < ratio_days**2
