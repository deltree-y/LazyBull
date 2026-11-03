# -*- coding: utf-8 -*-
"""P5b T2 族级事务与恢复测试（tests/ 全量豁免复杂度）。

覆盖 §4.3 状态表：0 / 部分 / 全部 roots 提升 × 未发布（ABORTING→ABORTED、新 root 移入
失败现场、从可验证父代备份还原、未发布分片移出）+ 已发布（只做分片/manifest/index 幂等
补齐、不重跑不重复入账）+ 已发布但分片缺失/内容不符；恢复中再崩溃继续同一事务；父代备份
不可验证 ⇒ campaign 失效；未发布 attempt 的 already_run 不直接接纳。
"""

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

import p5b_t2_env as envmod  # noqa: E402

from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    CampaignInvalidError,
    FamilyTxnError,
    InjectedFault,
    TestbedError,
    abort_attempt,
    read_index,
    rebuild_index,
    recover_campaign,
    root_identity,
    run_virtual_campaign,
)
from src.lazybull.v2.hosts.testbed.layout import layout_for  # noqa: E402
from src.lazybull.v2.hosts.testbed.storage import seal_record  # noqa: E402


def _fault_hook(*, point: str, arm_id: str | None = None):
    """故障钩子：命中注入点即模拟进程崩溃（不回滚、不清理）。"""

    def hook(name: str, payload):
        if name != point:
            return
        if arm_id is not None and payload.get("arm_id") != arm_id:
            return
        raise InjectedFault(f"injected fault at {name}（arm={payload.get('arm_id')}）")

    return hook


def _arm_sha(prepared, arm_id: str) -> str:
    return root_identity(prepared.root / "arms" / arm_id)["content_sha"]


def _run_day0(env, prepared):
    return envmod.run_campaign(env, prepared, days=[env.days[0]])


def test_zero_roots_promoted_abort_keeps_parent_and_index(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    parents = {arm: _arm_sha(prepared, arm) for arm in ("aa_1", "aa_2")}
    layout = layout_for(prepared.root)

    with pytest.raises(InjectedFault):
        envmod.run_campaign(
            env,
            prepared,
            days=[env.days[0]],
            fault_hook=_fault_hook(point="after_drive", arm_id="aa_1"),
        )
    attempt = envmod.journal(prepared, env.days[0])["attempts"][-1]
    assert attempt["status"] == "DRIVING"
    assert attempt["promoted_count"] == 0
    assert attempt["published_generation"] is None

    result = recover_campaign(layout, "t2_campaign")
    assert result["applied"][0]["status"] == "ABORTED"
    aborted = envmod.journal(prepared, env.days[0])["attempts"][-1]
    assert aborted["status"] == "ABORTED"
    assert aborted["attempt_id"] == attempt["attempt_id"]
    assert aborted["abort"]["from_status"] == "DRIVING"
    assert aborted["abort"]["arms"] == {"aa_1": "unchanged", "aa_2": "unchanged"}
    assert {arm: _arm_sha(prepared, arm) for arm in parents} == parents
    assert read_index(layout)["published_generation"] == 0
    assert not layout.staging_attempt_dir(env.days[0], attempt["attempt_id"]).exists()
    assert not (prepared.root / "ledger" / "aa_1" / f"{env.days[0]}.json").exists()


def test_partial_roots_promoted_moves_new_root_to_failed_and_restores_parent(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    parents = {arm: _arm_sha(prepared, arm) for arm in ("aa_1", "aa_2")}
    layout = layout_for(prepared.root)

    with pytest.raises(InjectedFault):
        envmod.run_campaign(
            env,
            prepared,
            days=[env.days[0]],
            fault_hook=_fault_hook(point="after_promote", arm_id="aa_1"),
        )
    attempt = envmod.journal(prepared, env.days[0])["attempts"][-1]
    assert attempt["status"] == "PROMOTING"
    assert attempt["promoted_count"] == 1
    assert attempt["arms"]["aa_1"]["promote_state"] == "promoted"
    assert attempt["arms"]["aa_2"]["promote_state"] == "pending"
    new_sha = attempt["arms"]["aa_1"]["staged"]["content_sha"]
    assert _arm_sha(prepared, "aa_1") == new_sha

    recover_campaign(layout, "t2_campaign")
    aborted = envmod.journal(prepared, env.days[0])["attempts"][-1]
    assert aborted["status"] == "ABORTED"
    assert aborted["abort"]["arms"] == {"aa_1": "restored", "aa_2": "unchanged"}
    assert _arm_sha(prepared, "aa_1") == parents["aa_1"]
    assert _arm_sha(prepared, "aa_2") == parents["aa_2"]
    failed_root = prepared.root / "failed" / env.days[0] / attempt["attempt_id"] / "roots" / "aa_1"
    assert failed_root.is_dir()
    assert root_identity(failed_root)["content_sha"] == new_sha
    assert read_index(layout)["published_generation"] == 0


def test_crash_between_two_renames_of_single_promotion_is_recoverable(tmp_path):
    """单臂提升是两个 rename：父代已移走、新根未就位时崩溃 ⇒ 从父代备份还原。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    parents = {arm: _arm_sha(prepared, arm) for arm in ("aa_1", "aa_2")}
    layout = layout_for(prepared.root)

    with pytest.raises(InjectedFault):
        envmod.run_campaign(
            env,
            prepared,
            days=[env.days[0]],
            fault_hook=_fault_hook(point="mid_promote", arm_id="aa_1"),
        )
    attempt = envmod.journal(prepared, env.days[0])["attempts"][-1]
    assert attempt["status"] == "PROMOTING"
    assert attempt["promoted_count"] == 0
    assert not (prepared.root / "arms" / "aa_1").exists()

    recover_campaign(layout, "t2_campaign")
    aborted = envmod.journal(prepared, env.days[0])["attempts"][-1]
    assert aborted["status"] == "ABORTED"
    assert aborted["abort"]["arms"]["aa_1"] == "restored"
    assert {arm: _arm_sha(prepared, arm) for arm in parents} == parents
    assert read_index(layout)["published_generation"] == 0


def test_all_roots_promoted_and_shards_written_are_moved_out_on_abort(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    parents = {arm: _arm_sha(prepared, arm) for arm in ("aa_1", "aa_2")}
    layout = layout_for(prepared.root)

    with pytest.raises(InjectedFault):
        envmod.run_campaign(
            env,
            prepared,
            days=[env.days[0]],
            fault_hook=_fault_hook(point="after_shards"),
        )
    attempt = envmod.journal(prepared, env.days[0])["attempts"][-1]
    assert attempt["status"] == "ARTIFACTS"
    assert attempt["promoted_count"] == 2
    assert set(attempt["shards"]) == {"aa_1", "aa_2"}
    shard_path = layout.shard_path("aa_1", env.days[0])
    assert shard_path.is_file()

    recover_campaign(layout, "t2_campaign")
    aborted = envmod.journal(prepared, env.days[0])["attempts"][-1]
    assert aborted["status"] == "ABORTED"
    assert {arm: _arm_sha(prepared, arm) for arm in parents} == parents
    assert not shard_path.exists()
    moved = prepared.root / "failed" / env.days[0] / attempt["attempt_id"] / "shards"
    assert (moved / "aa_1" / f"{env.days[0]}.json").is_file()
    assert read_index(layout)["published_generation"] == 0
    assert not layout.manifest_path("aa_1").exists()


def test_published_generation_tops_up_without_rerun_or_rebooking(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    layout = layout_for(prepared.root)
    first = _run_day0(env, prepared)
    assert first["published"]["published_generation"] == 1
    index_before = envmod.index(prepared)
    trades_before = envmod.shard(prepared, "aa_1", env.days[0])["evidence"]["fills"]
    root_before = _arm_sha(prepared, "aa_1")

    # 抹掉派生投影：分片 + manifest + index（模拟"已发布但分片/index 未写"）
    layout.shard_path("aa_1", env.days[0]).unlink()
    layout.manifest_path("aa_2").unlink()
    layout.index_path.unlink()

    result = recover_campaign(layout, "t2_campaign")
    actions = [item["action"] for item in result["applied"]]
    assert "top_up" in actions
    assert envmod.index(prepared) == index_before
    rebuilt = envmod.shard(prepared, "aa_1", env.days[0])
    assert rebuilt["evidence"]["fills"] == trades_before
    assert rebuilt["family_txn_id"] == index_before["published_txn_id"]
    assert envmod.manifest(prepared, "aa_1")["head_generation"] == 1
    assert _arm_sha(prepared, "aa_1") == root_before
    # 幂等：再恢复一次零写盘
    again = recover_campaign(layout, "t2_campaign")
    assert again["applied"] == []


def test_published_shard_content_mismatch_invalidates(tmp_path):
    """重封存后仍能被内容身份比对识别（不是只靠封存哈希）。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    layout = layout_for(prepared.root)
    _run_day0(env, prepared)
    path = layout.shard_path("aa_1", env.days[0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evidence"]["valuation"]["nav"] = 0.123456
    path.write_text(
        json.dumps(seal_record(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    with pytest.raises(CampaignInvalidError, match="内容身份不符"):
        recover_campaign(layout, "t2_campaign")
    # 未重封存的篡改由封存哈希先拒（双重防线）
    payload["evidence"]["valuation"]["nav"] = 0.5
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(TestbedError, match="封存哈希不一致"):
        recover_campaign(layout, "t2_campaign")


def test_crash_during_abort_resumes_same_transaction(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    parents = {arm: _arm_sha(prepared, arm) for arm in ("aa_1", "aa_2")}
    layout = layout_for(prepared.root)

    with pytest.raises(InjectedFault):
        envmod.run_campaign(
            env,
            prepared,
            days=[env.days[0]],
            fault_hook=_fault_hook(point="after_promote", arm_id="aa_1"),
        )
    attempt_id = envmod.journal(prepared, env.days[0])["attempts"][-1]["attempt_id"]

    def crash_hook(name: str, payload):
        if name == "after_abort_restore":
            raise InjectedFault("crash during abort")

    with pytest.raises(InjectedFault):
        recover_campaign(layout, "t2_campaign", fault_hook=crash_hook)
    mid = envmod.journal(prepared, env.days[0])["attempts"][-1]
    assert mid["status"] == "ABORTING"
    assert mid["attempt_id"] == attempt_id

    recover_campaign(layout, "t2_campaign")
    done = envmod.journal(prepared, env.days[0])["attempts"][-1]
    assert done["status"] == "ABORTED"
    assert done["attempt_id"] == attempt_id
    assert len(envmod.journal(prepared, env.days[0])["attempts"]) == 1
    assert {arm: _arm_sha(prepared, arm) for arm in parents} == parents


def test_unverifiable_parent_backup_invalidates_campaign(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    layout = layout_for(prepared.root)
    with pytest.raises(InjectedFault):
        envmod.run_campaign(
            env,
            prepared,
            days=[env.days[0]],
            fault_hook=_fault_hook(point="after_promote", arm_id="aa_1"),
        )
    attempt = envmod.journal(prepared, env.days[0])["attempts"][-1]
    backup = (
        prepared.root / "backups" / env.days[0] / attempt["attempt_id"] / "aa_1" / "config.yaml"
    )
    backup.write_text("tampered: true\n", encoding="utf-8")
    with pytest.raises(CampaignInvalidError, match="父代备份不可验证"):
        recover_campaign(layout, "t2_campaign")


def test_abort_allows_new_attempt_same_virtual_date(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    layout = layout_for(prepared.root)
    with pytest.raises(InjectedFault):
        envmod.run_campaign(
            env,
            prepared,
            days=[env.days[0]],
            fault_hook=_fault_hook(point="after_stage", arm_id="aa_2"),
        )
    recover_campaign(layout, "t2_campaign")
    day = env.days[0]
    attempts = envmod.journal(prepared, day)["attempts"]
    assert len(attempts) == 1 and attempts[0]["status"] == "ABORTED"

    report = envmod.run_campaign(env, prepared, days=[day])
    assert report["published"]["published_generation"] == 1
    attempts = envmod.journal(prepared, day)["attempts"]
    assert len(attempts) == 2
    aborted, committed = attempts
    assert aborted["status"] == "ABORTED" and committed["status"] == "COMMITTED"
    assert aborted["family_txn_id"] != committed["family_txn_id"]
    assert aborted["attempt_id"] != committed["attempt_id"]
    # 只入账一次：分片成交与臂成交行一一对应
    shard = envmod.shard(prepared, "aa_1", day)
    assert shard["evidence"]["fills"] == []
    from src.lazybull.v2.hosts.paper.state_store import PaperStateStore

    store = PaperStateStore(root_path=str(prepared.root / "arms" / "aa_1"), create_layout=False)
    assert store.load_all_trades() is None


def test_unpublished_already_run_is_not_accepted(tmp_path):
    """已驱动但未发布的暂存根返回 already_run ⇒ 终止 attempt、不写 index、不重复入账。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    layout = layout_for(prepared.root)
    parents = {arm: _arm_sha(prepared, arm) for arm in ("aa_1", "aa_2")}
    day = env.days[0]
    inner = envmod.make_host_factory(env)

    def pre_driven_factory(spec, store_root):
        warm = inner(spec, store_root)
        warm.run_t1(day)
        return inner(spec, store_root)  # 新宿主：权威末日 == day ⇒ already_run

    with pytest.raises(CampaignInvalidError, match="already_run"):
        run_virtual_campaign(
            prepared.root,
            days=[day],
            host_factory=pre_driven_factory,
            allow_external_root=True,
        )
    aborted = envmod.journal(prepared, day)["attempts"][-1]
    assert aborted["status"] == "ABORTED"
    assert aborted["abort"]["reason"] == "already_run_unpublished"
    assert read_index(layout)["published_generation"] == 0
    assert {arm: _arm_sha(prepared, arm) for arm in parents} == parents
    assert not layout.shard_path("aa_1", day).exists()


def test_stale_index_is_repaired_from_journal_before_next_day(tmp_path):
    """index 落后于 journal（发布点已写、投影未写）⇒ 下一次运行前必须自愈，不得重跑已发布日。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    layout = layout_for(prepared.root)
    envmod.run_campaign(env, prepared, days=env.days[:2])
    full = envmod.index(prepared)
    layout.index_path.unlink()

    report = envmod.run_campaign(env, prepared, days=env.days[:2])
    assert envmod.index(prepared) == full
    assert report["published"]["published_generation"] == 2
    assert all(record["mode"] == "already_published" for record in report["transactions"])
    assert report["recovery"]["published_generation"] >= 2


def test_index_rebuild_from_journal_is_deterministic(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    layout = layout_for(prepared.root)
    envmod.run_campaign(env, prepared, days=env.days[:2])
    before = envmod.index(prepared)
    layout.index_path.unlink()
    rebuilt = rebuild_index(layout, "t2_campaign")
    assert rebuilt["written"] is True
    assert envmod.index(prepared) == before
    # 无已发布 attempt 时不得凭空写出 index（genesis 由运行器在首次运行时独占建立）
    env2 = envmod.build_env(tmp_path / "genesis_only")
    prepared2 = envmod.prepare_campaign(env2, campaign_id="t2_genesis")
    layout2 = layout_for(prepared2.root)
    assert read_index(layout2) is None
    empty = rebuild_index(layout2, "t2_genesis")
    assert empty["written"] is False
    assert read_index(layout2) is None


def test_abort_refuses_published_attempt(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare_campaign(env)
    layout = layout_for(prepared.root)
    _run_day0(env, prepared)
    attempt = envmod.journal(prepared, env.days[0])["attempts"][-1]
    with pytest.raises(FamilyTxnError, match="不得回滚"):
        abort_attempt(layout, attempt, reason="illegal")
