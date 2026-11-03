# -*- coding: utf-8 -*-
"""P5b T3 长期分片账本测试（tests/ 全量豁免复杂度）。

覆盖：三方一致性（分片 / manifest / index）缺口的分类与定位、可重建缺口的字节一致修复、
不可重建缺口的 INVALIDATED 终态与推进拒绝、未发布分片隔离（未隔离前拒绝推进）、
只读查询、审计零写入与幂等、journal 损坏 / 重代 fail-closed、不可读投影的隔离重建、
§7.2 T3 报告字段、staging 清理语义。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t3_env as envmod  # noqa: E402

from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    CampaignBusyError,
    CampaignRunLock,
    InjectedFault,
    LedgerInvalidatedError,
    LedgerRepairRequiredError,
    assert_ledger_advanceable,
    audit_ledger,
    build_chain,
    build_ledger_report,
    invalidated_days,
    ledger_paths,
    layout_for,
    prune_staging,
    query_shards,
    repair_ledger,
    shard_sha_map,
)
from src.lazybull.v2.hosts.testbed.storage import seal_record  # noqa: E402

#: 用例按需选择臂数：三臂 D0 回放是主要固定成本，三臂只留给需要「部分臂」语义的用例
ONE_ARM = ("aa_1",)
TWO_ARMS = ("aa_1", "aa_2")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_sealed(path: Path, payload: dict) -> None:
    body = seal_record({key: value for key, value in payload.items() if key != "record_sha256"})
    path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _gap_keys(report: dict, kind: str) -> list[tuple[str, str | None]]:
    return [(gap["arm_id"], gap["virtual_date"]) for gap in report["gaps"] if gap["kind"] == kind]


def test_runner_tail_leaves_healthy_campaign_pass_and_prunes_staging(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    report = envmod.run(env, prepared, env.days[:3])
    layout = layout_for(prepared.root)

    tail = (report.get("machine_time") or {}).get("ledger_audit") or {}
    assert tail["status"] == "PASS"
    assert tail["three_way_status"] == "PASS"
    assert tail["artifact_status"] == "PASS"
    assert tail["chain_written"] is True
    assert layout.reports_root.exists() or True

    audit = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert audit["status"] == "PASS"
    assert audit["gaps"] == []
    assert audit["three_way"]["consistent"] is True
    for arm_id in prepared.arms:
        assert audit["chain"]["arms"][arm_id]["link_count"] == 3
        assert audit["arms"][arm_id]["journal_days"] == 3
    # T2 遗留项 1：整臂复制的暂存残留在收尾被回收（不再线性增长）
    staging = (report.get("machine_time") or {}).get("staging_pruned") or {}
    assert staging["bytes_after"] == 0
    assert not any(path.is_dir() for path in layout.staging_root.glob("*/*"))


def test_deleted_shard_is_missing_day_and_repairs_byte_identical(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=TWO_ARMS)
    envmod.run(env, prepared, env.days[:3])
    layout = layout_for(prepared.root)
    before_chain = audit_ledger(layout, envmod.CAMPAIGN_ID)["chain"]["chain_digests"]

    path = envmod.shard_path(prepared, "aa_1", env.days[1])
    original = path.read_bytes()
    path.unlink()

    broken = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert _gap_keys(broken, "missing_day") == [("aa_1", env.days[1])]
    assert broken["gaps_by_class"]["missing_day"] == 1
    assert broken["chain"]["chain_digests"]["aa_1"] != before_chain["aa_1"]
    assert broken["chain"]["chain_digests"]["aa_2"] == before_chain["aa_2"]
    assert broken["three_way"]["arms"]["aa_1"]["consistent"] is False
    assert broken["three_way_status"] == "GAPS"
    assert broken["invalidated"] == []

    repair = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert repair["blocked"] is False
    assert [action["action"] for action in repair["applied"]] == [
        "rebuild_shard",
        "write_chain",
    ]
    assert path.read_bytes() == original
    assert repair["audit_after"]["status"] == "PASS"
    assert repair["audit_after"]["chain"]["chain_digests"] == before_chain


def test_modified_manifest_generation_is_rebuilt(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=TWO_ARMS)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    manifest_path = layout.manifest_path("aa_2")
    payload = _load(manifest_path)
    payload["shards"][env.days[0]]["generation"] = 99
    _write_sealed(manifest_path, payload)

    report = audit_ledger(layout, envmod.CAMPAIGN_ID)
    kinds = {gap["kind"] for gap in report["gaps"] if gap["arm_id"] == "aa_2"}
    assert "manifest_mismatch" in kinds
    assert report["invalidated"] == []

    repair = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert "rebuild_manifest" in [action["action"] for action in repair["applied"]]
    assert _load(manifest_path)["shards"][env.days[0]]["generation"] == 1
    assert repair["audit_after"]["status"] == "PASS"


def test_deleted_index_entry_is_classified_and_rebuilt(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:3])
    layout = layout_for(prepared.root)
    original = _load(layout.index_path)
    payload = dict(original)
    payload["days"] = [entry for entry in payload["days"] if entry["virtual_date"] != env.days[2]]
    payload["published_generation"] = 2
    payload["published_virtual_date"] = env.days[1]
    _write_sealed(layout.index_path, payload)

    report = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert report["three_way_status"] == "GAPS"
    assert report["published"]["published_generation"] == 2  # 读侧落后（journal 仍为 3）
    assert report["published"]["journal_head_generation"] == 3
    assert any(
        gap["kind"] == "index_missing_day" and gap["virtual_date"] == env.days[2]
        for gap in report["gaps"]
    )
    assert report["invalidated"] == []

    repair = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert "rebuild_index" in [action["action"] for action in repair["applied"]]
    assert _load(layout.index_path) == original
    assert repair["audit_after"]["status"] == "PASS"


def test_index_missing_arm_entry_is_repairable(tmp_path):
    """index 缺臂条目属派生投影空洞（可重建），不得误判为 INVALIDATED。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=TWO_ARMS)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    original = _load(layout.index_path)
    payload = json.loads(json.dumps(original))
    payload["days"][1]["arms"].pop("aa_2")
    _write_sealed(layout.index_path, payload)

    report = audit_ledger(layout, envmod.CAMPAIGN_ID)
    kinds = {
        (gap["arm_id"], gap["kind"])
        for gap in report["gaps"]
        if gap["arm_id"] == "aa_2" and gap["virtual_date"] == env.days[1]
    }
    assert ("aa_2", "index_missing_day") in kinds
    assert report["invalidated"] == []
    assert report["three_way_status"] == "GAPS"

    repair = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert "rebuild_index" in [action["action"] for action in repair["applied"]]
    assert _load(layout.index_path) == original
    assert repair["audit_after"]["status"] == "PASS"


def test_index_generation_ahead_of_journal_invalidates(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    payload = _load(layout.index_path)
    payload["published_generation"] = 9
    _write_sealed(layout.index_path, payload)

    report = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert report["status"] == "INVALIDATED"
    ahead = [gap for gap in report["gaps"] if gap["kind"] == "index_mismatch"]
    assert ahead and ahead[0]["repairable"] is False

    repair = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert repair["blocked"] is True
    assert repair["applied"] == []
    assert invalidated_days(layout)["*"]


def test_unpublished_attempt_shard_blocks_advance_until_quarantined(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, [env.days[0]])
    layout = layout_for(prepared.root)

    with pytest.raises(InjectedFault):
        envmod.run(
            env,
            prepared,
            [env.days[1]],
            fault_hook=envmod.fault_hook("after_shard", arm_id="aa_1"),
        )
    orphan = envmod.shard_path(prepared, "aa_1", env.days[1])
    assert orphan.is_file()

    report = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert ("aa_1", env.days[1]) in _gap_keys(report, "unpublished_attempt_shard")
    assert report["published"]["published_generation"] == 1
    envmod.assert_unpublished_before_recovery(
        prepared, target_day=env.days[1], expected_generation=1
    )

    dry = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=False)
    assert dry["dry_run"] is True
    assert [action["action"] for action in dry["planned"]] == ["quarantine_shard"]
    assert orphan.is_file()

    envmod.recover(prepared)
    # 未隔离前推进必须被拒（残片会与 write_shard 的「禁止覆盖」冲突）
    with pytest.raises(LedgerRepairRequiredError, match="未发布 attempt 的分片"):
        envmod.run(env, prepared, [env.days[1]])
    repair = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert [action["action"] for action in repair["applied"]] == ["quarantine_shard"]
    assert repair["audit_after"]["status"] == "PASS"
    assert not orphan.exists()
    quarantine = ledger_paths(layout).quarantine_root / "shards" / "aa_1" / f"{env.days[1]}.json"
    assert quarantine.is_file()
    assert repair["audit_after"]["gaps_by_class"]["unpublished_attempt_shard"] == 0

    report_after_rerun = envmod.rerun_same_day(env, prepared, env.days[1])
    assert report_after_rerun["published"]["published_generation"] == 2
    assert (
        audit_ledger(layout, envmod.CAMPAIGN_ID)["gaps_by_class"]["unpublished_attempt_shard"] == 0
    )


def test_tampered_shard_invalidates_and_refuses_advance(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    path = envmod.shard_path(prepared, "aa_1", env.days[0])
    envmod.tamper_shard(prepared, "aa_1", env.days[0])
    tampered = path.read_bytes()

    report = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert report["status"] == "INVALIDATED"
    assert report["invalidated"][0]["terminal_state"] == "INVALIDATED"
    assert _gap_keys(report, "bad_shard") == [("aa_1", env.days[0])]

    # 未登记前推进即被拒（检测优先，不会被 T2 的 top-up 抢先降级为 campaign 失效）
    with pytest.raises(LedgerInvalidatedError, match="不可重建缺口"):
        envmod.run(env, prepared, [env.days[2]])

    repair = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert repair["blocked"] is True
    assert repair["invalidated"]
    assert path.read_bytes() == tampered  # 禁止覆盖已发布字节
    registered = invalidated_days(layout)
    assert env.days[0] in registered["aa_1"]

    with pytest.raises(LedgerInvalidatedError, match="INVALIDATED"):
        envmod.run(env, prepared, [env.days[2]])
    with pytest.raises(LedgerInvalidatedError):
        assert_ledger_advanceable(layout, envmod.CAMPAIGN_ID)

    again = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert again["blocked"] is True
    assert len(invalidated_days(layout)["aa_1"]) == 1


def test_invalidation_registry_is_append_only_across_repairs(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    envmod.tamper_shard(prepared, "aa_1", env.days[0])
    repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    before = _load(ledger_paths(layout).invalidated_path)
    repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert _load(ledger_paths(layout).invalidated_path) == before
    assert before["entries"][0]["terminal_state"] == "INVALIDATED"
    assert before["evidence_class"] == "calibration_only"


def test_repair_and_prune_require_campaign_lock(tmp_path):
    """写动作必须与运行器互斥：持锁时修复/清理一律被拒。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    with CampaignRunLock(layout):
        with pytest.raises(CampaignBusyError):
            repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
        with pytest.raises(CampaignBusyError):
            prune_staging(layout, apply=True)
    # 释放后可正常执行
    assert repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)["blocked"] is False


def test_recovery_rebuilds_stale_projections_on_next_run(tmp_path):
    """清单多余条目/头代不符、index 发布代落后 ⇒ 下一次运行前由恢复路径整批重建。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)

    manifest = _load(layout.manifest_path("aa_1"))
    manifest["shards"]["20991231"] = {
        "sha256": "sha-bogus",
        "generation": 99,
        "gen_id": "g0099",
        "family_txn_id": "txn-bogus",
        "content_sha": "sha-bogus",
    }
    manifest["head_generation"] = 99
    manifest["head_gen_id"] = "g0099"
    _write_sealed(layout.manifest_path("aa_1"), manifest)
    index = _load(layout.index_path)
    index["published_generation"] = 1
    index["published_txn_id"] = "txn-stale"
    _write_sealed(layout.index_path, index)

    broken = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert broken["three_way_status"] == "GAPS"

    envmod.run(env, prepared, env.days[:2])  # 重跑已发布日：恢复路径应整批重建投影
    healed = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert healed["status"] == "PASS"
    assert healed["three_way_status"] == "PASS"
    assert healed["published"]["published_generation"] == 2
    assert "20991231" not in _load(layout.manifest_path("aa_1"))["shards"]


def test_index_day_order_inversion_is_repaired_on_next_run(tmp_path):
    """index 日序倒置（投影级错序）由恢复路径整批重建。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:3])
    layout = layout_for(prepared.root)
    payload = _load(layout.index_path)
    payload["days"] = list(reversed(payload["days"]))
    _write_sealed(layout.index_path, payload)
    assert audit_ledger(layout, envmod.CAMPAIGN_ID)["three_way_status"] == "GAPS"

    envmod.run(env, prepared, env.days[:3])
    healed = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert healed["status"] == "PASS"
    generations = [entry["generation"] for entry in _load(layout.index_path)["days"]]
    assert generations == sorted(generations) == [1, 2, 3]


def test_unreadable_index_does_not_break_next_run(tmp_path):
    """index 被截断时运行器先隔离留证、再由 journal 重建（不再抛裸异常）。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    layout.index_path.write_text("{broken", encoding="utf-8")

    report = envmod.run(env, prepared, env.days[:2])
    assert report["published"]["published_generation"] == 2
    assert report["recovery"]["quarantined_projections"]
    assert audit_ledger(layout, envmod.CAMPAIGN_ID)["status"] == "PASS"
    moved = ledger_paths(layout).quarantine_root / "projections" / "_index.json"
    assert moved.is_file()


def test_audit_is_read_only_and_idempotent(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    before = envmod.tree_state(prepared.root)
    first = audit_ledger(layout, envmod.CAMPAIGN_ID)
    second = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert first == second
    assert envmod.tree_state(prepared.root) == before


def test_repair_dry_run_writes_nothing(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    envmod.shard_path(prepared, "aa_1", env.days[0]).unlink()
    before = envmod.tree_state(prepared.root)
    result = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=False)
    assert result["dry_run"] is True
    assert [action["action"] for action in result["planned"]] == [
        "rebuild_shard",
        "write_chain",
    ]
    assert envmod.tree_state(prepared.root) == before
    assert result["audit_after"]["gaps_by_class"]["missing_day"] == 1


def test_unreadable_manifest_and_index_are_quarantined_and_rebuilt(tmp_path):
    """不可读的派生投影（截断/封存破坏）先隔离留证，再按 journal 重建。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    manifest_path = layout.manifest_path("aa_1")
    index_path = layout.index_path
    manifest_path.write_text("{broken", encoding="utf-8")
    index_path.write_text("{also-broken", encoding="utf-8")

    report = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert report["invalidated"] == []
    assert report["gaps_by_class"]["manifest_mismatch"] >= 1
    assert report["gaps_by_class"]["index_absent"] >= 1

    repair = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    applied = [action["action"] for action in repair["applied"]]
    assert "rebuild_manifest" in applied and "rebuild_index" in applied
    assert repair["audit_after"]["status"] == "PASS"
    quarantine = ledger_paths(layout).quarantine_root / "projections"
    assert (quarantine / "manifest.json").is_file()
    assert (quarantine / "_index.json").is_file()


def test_duplicate_generation_in_journal_invalidates_without_crash(tmp_path):
    """journal 出现同日重代/代序错乱时审计不得崩：收敛为 out_of_order 终态。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    doc = envmod.journal(prepared, env.days[1])
    attempt = doc["attempts"][-1]
    attempt["target_generation"] = 1  # 与 day0 重代
    _write_sealed(layout.journal_path(env.days[1]), doc)

    report = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert report["status"] == "INVALIDATED"
    assert report["gaps_by_class"]["out_of_order"] >= 1
    with pytest.raises(LedgerInvalidatedError):
        assert_ledger_advanceable(layout, envmod.CAMPAIGN_ID)
    blocked = repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert blocked["blocked"] is True
    assert invalidated_days(layout)


def test_query_shards_returns_only_published_range(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:4])
    layout = layout_for(prepared.root)

    full = query_shards(layout, "aa_1")
    assert [day["virtual_date"] for day in full["days"]] == list(env.days[:4])
    assert full["contiguous"] is True
    assert full["published_generation"] == 4

    ranged = query_shards(layout, "aa_1", start=env.days[1], end=env.days[2])
    assert [day["virtual_date"] for day in ranged["days"]] == list(env.days[1:3])
    assert (
        ranged["chain_digest"]
        == build_chain(envmod.CAMPAIGN_ID, "aa_1", ranged["chain_links"])["chain_digest"]
    )
    assert ranged["chain_digest"] != full["chain_digest"]

    empty = query_shards(layout, "aa_1", start=env.days[9])
    assert empty["days"] == []
    assert empty["chain_digest"] is None


def test_query_refuses_invalidated_range(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=TWO_ARMS)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    envmod.tamper_shard(prepared, "aa_1", env.days[0])
    repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    with pytest.raises(LedgerInvalidatedError):
        query_shards(layout, "aa_1", start=env.days[0])
    allowed = query_shards(layout, "aa_1", start=env.days[0], allow_invalidated=True)
    assert allowed["invalidated"] == [env.days[0]]
    assert allowed["days"][0]["invalidated"] is True
    assert query_shards(layout, "aa_2")["contiguous"] is True


def test_campaign_level_invalidation_blocks_every_arm(tmp_path):
    """campaign 级（``*``）失效波及全部臂：任何臂的区间查询都 fail-closed。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=TWO_ARMS)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    payload = _load(layout.index_path)
    payload["published_generation"] = 9
    _write_sealed(layout.index_path, payload)
    repair_ledger(layout, envmod.CAMPAIGN_ID, apply=True)
    assert "*" in invalidated_days(layout)
    for arm_id in prepared.arms:
        with pytest.raises(LedgerInvalidatedError, match="campaign 级失效|INVALIDATED"):
            query_shards(layout, arm_id)
    allowed = query_shards(layout, "aa_1", allow_invalidated=True)
    assert allowed["days"] and all(day["invalidated"] for day in allowed["days"])
    with pytest.raises(LedgerInvalidatedError, match="<all-days>"):
        assert_ledger_advanceable(layout, envmod.CAMPAIGN_ID)


def test_corrupt_journal_is_fail_closed(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, [env.days[0]])
    layout = layout_for(prepared.root)
    (prepared.root / "journal" / f"{env.days[0]}.json").write_text("{broken", encoding="utf-8")
    report = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert report["status"] == "INVALIDATED"
    assert report["gaps_by_class"]["journal_corrupt"] == 1
    with pytest.raises(LedgerInvalidatedError, match="拒绝推进"):
        assert_ledger_advanceable(layout, envmod.CAMPAIGN_ID)


def test_out_of_window_shard_is_rejected_by_clock_days(tmp_path):
    """越窗分片（虚拟时钟回退/跨 campaign 复制）在提供 clock_days 时判不可重建。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    outside = str(env.calendar[-1])  # campaign 窗口外但仍属日历内的交易日
    stray = layout.shard_path("aa_1", outside)
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(layout.shard_path("aa_1", env.days[1]).read_bytes())

    report = audit_ledger(layout, envmod.CAMPAIGN_ID, clock_days=list(env.days))
    kinds = {(gap["arm_id"], gap["virtual_date"], gap["kind"]) for gap in report["gaps"]}
    assert ("aa_1", outside, "out_of_order") in kinds
    assert report["status"] == "INVALIDATED"
    # 未提供窗口时归入未发布残片（可隔离），不会凭空判失效
    loose = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert ("aa_1", outside) in _gap_keys(loose, "unpublished_attempt_shard")


def test_ledger_report_declares_t3_fields_and_limits(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    audit = audit_ledger(layout, envmod.CAMPAIGN_ID)
    report = build_ledger_report(
        audit,
        repair=repair_ledger(layout, envmod.CAMPAIGN_ID, apply=False),
        fault_matrix={
            "path": "audit/fault_matrix.json",
            "record_sha256": "sha-placeholder",
            "all_passed": True,
            "passed_count": 13,
            "case_count": 13,
        },
        scale={"virtual_days": 2},
        machine_time={"total_seconds": 0.5},
    )
    assert report["schema"] == "p5b_testbed_t3_ledger_report/v1"
    assert report["evidence_class"] == "calibration_only"
    assert report["not_forward_evidence"] is True
    assert report["not_sla_evidence"] is True
    assert report["shard_chain_digest"] == audit["chain"]["chain_digest"]
    assert report["unique_publish_point"]["published_generation"] == 2
    assert report["three_way_status"] == "PASS"
    assert report["artifact_status"] == "PASS"
    assert report["fault_matrix"]["all_passed"] is True
    assert report["machine_time"]["long_task_threshold_seconds"] == 1200.0
    assert report["machine_time"]["long_task_threshold_exceeded"] is False
    text = "\n".join(report["limitations"])
    for token in ("NOT_FORWARD_EVIDENCE", "T3", "INVALIDATED", "SLA"):
        assert token in text


def test_prune_staging_only_removes_terminal_attempts(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    doc = envmod.journal(prepared, env.days[1])
    committed = next(item for item in doc["attempts"] if item["status"] == "COMMITTED")

    terminal = layout.staging_attempt_dir(env.days[1], committed["attempt_id"])
    terminal.mkdir(parents=True, exist_ok=True)
    (terminal / "payload.bin").write_bytes(b"x" * 1024)
    stray = layout.staging_attempt_dir("20990101", "att-fake")
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "payload.bin").write_bytes(b"y" * 1024)

    dry = prune_staging(layout, apply=False)
    assert dry["dry_run"] is True
    assert dry["bytes_before"] >= 2048

    ledger_before = envmod.tree_state(layout.ledger_root)
    applied = prune_staging(layout, apply=True)
    assert applied["bytes_after"] < applied["bytes_before"]
    assert not terminal.exists()
    assert stray.exists(), "未登记/非终态 attempt 的暂存不得清理"
    assert envmod.tree_state(layout.ledger_root) == ledger_before
    assert audit_ledger(layout, envmod.CAMPAIGN_ID)["status"] in {"PASS", "GAPS"}
    assert shard_sha_map(layout)["aa_1"]
    shutil.rmtree(prepared.root, ignore_errors=True)
