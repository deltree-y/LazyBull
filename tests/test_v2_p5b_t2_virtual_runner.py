# -*- coding: utf-8 -*-
"""P5b T2 虚拟时钟多臂运行器测试（tests/ 全量豁免复杂度）。

覆盖：注册 → D0 fresh → 连续 10 个 structural 历史虚拟日推进；A/A 两臂日期/成交/现金/
持仓/NAV 逐值一致；T+1 语义；跌停阻断卖出的 pending 与次日补卖；无成交日完整权威估值；
行域回执；整族 `not_ready` 暂停不推进；已发布代幂等重跑不重复入账；生产目录零写入。
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

from src.lazybull.v2.hosts.paper.state_store import PaperStateStore  # noqa: E402
from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    FamilyTxnError,
    initialize_d0_fresh,
    register_arm,
    register_campaign,
    run_virtual_campaign,
    verify_d0_fresh,
)
from src.lazybull.v2.store.store_identity import canonical_json_sha256  # noqa: E402


def _recursive_entries(path: Path) -> set:
    if not path.is_dir():
        return set()
    return {entry.relative_to(path).as_posix() for entry in path.rglob("*")}


def _fresh_campaign(tmp_path, **env_kwargs):
    env = envmod.build_env(tmp_path, **env_kwargs)
    prepared = envmod.prepare_campaign(env)
    return env, prepared


def test_ten_structural_virtual_days_promote_and_publish(tmp_path):
    env, prepared = _fresh_campaign(tmp_path)
    report = envmod.run_campaign(env, prepared, write_report=True)

    assert report["scope"] == "historical_testbed"
    assert report["evidence_class"] == "calibration_only"
    assert report["virtual_clock"] is True and report["not_forward_evidence"] is True
    assert report["schedule_mode"] == "replay"
    window = report["window"]
    assert window["day_count"] == 10 and window["d0_date"] == env.days[0]
    assert window["d0_role"] == "valuation_only"
    assert window["first_return_date"] == env.days[1]
    assert len(window["adjacent_return_pairs"]) == 9

    committed = [r for r in report["transactions"] if r["mode"] == "committed"]
    assert len(committed) == 10
    assert [r["published_generation"] for r in committed] == list(range(1, 11))
    assert report["published"]["published_generation"] == 10
    assert report["published"]["head_gen_id"] == "g0010"
    # 唯一发布点：每次发布的 published_txn_id == 该 attempt 的 family_txn_id
    for record in committed:
        assert record["unique_publish_point"]["value"] == record["family_txn_id"]
        assert record["parent_generation"] + 1 == record["published_generation"]
        assert record["promoted_count"] == 2
    # 虚拟日不是事务唯一键：同日不同 attempt 由 family_txn_id 区分
    txn_ids = [record["family_txn_id"] for record in committed]
    assert len(set(txn_ids)) == len(txn_ids)

    assert report["paused"] is None
    assert report["aa_consistency"]["identical"] is True
    assert report["aa_consistency"]["compared_days"] == list(env.days)
    for arm_id in ("aa_1", "aa_2"):
        summary = report["arms"][arm_id]
        assert summary["driven_days"] == 10
        assert summary["driven_dates"] == list(env.days)
        assert summary["complete_generations"] == 10
        assert summary["authoritative_values"] == 10
        assert summary["identity"]["train_end_date"] < env.days[0]
    assert report["row_domain"]["verified"] is True
    assert set(report["row_domain"]["per_day"]) == set(env.days)
    for per_arm in report["row_domain"]["per_day"].values():
        for receipt in per_arm.values():
            assert receipt["actual_row_domains"] == ["structural"]
            assert receipt["passed"] is True
    assert Path(report["report_path"]).is_file()
    assert report["machine_time"]["total_seconds"] > 0
    assert set(report["machine_time"]["per_day_seconds"]) == set(env.days)


def test_aa_arms_match_on_date_fills_cash_positions_and_nav(tmp_path):
    env, prepared = _fresh_campaign(tmp_path)
    envmod.run_campaign(env, prepared)
    for day in env.days:
        left = envmod.shard(prepared, "aa_1", day)["evidence"]
        right = envmod.shard(prepared, "aa_2", day)["evidence"]
        assert left["driven_date"] == right["driven_date"] == day
        assert left["valuation"]["nav"] == right["valuation"]["nav"]
        assert left["valuation"]["portfolio_value"] == right["valuation"]["portfolio_value"]
        assert left["cash"] == right["cash"]
        assert left["positions"] == right["positions"]
        assert left["fills"] == right["fills"]
        assert left["day_digest"] == right["day_digest"]
    # 各臂根内容身份不同（run 记录含墙钟时间戳），但逐值证据相同——A/A 不等于逐字节相同
    index = envmod.index(prepared)
    assert index["arms"]["aa_1"]["content_sha"] != index["arms"]["aa_2"]["content_sha"]


def test_t1_semantics_signal_day_then_next_day_fills(tmp_path):
    env, prepared = _fresh_campaign(tmp_path)
    envmod.run_campaign(env, prepared)
    first, second = env.days[0], env.days[1]
    # D0 只生成次日指令、不成交（D0 = 估值点）
    assert envmod.shard(prepared, "aa_1", first)["evidence"]["fills"] == []
    frame_path = prepared.root / "arms" / "aa_1" / "instructions" / f"{second}.parquet"
    assert frame_path.is_file()
    fills = envmod.shard(prepared, "aa_1", second)["evidence"]["fills"]
    assert [row["action"] for row in fills] == ["buy", "buy"]
    # 每臂当日恰一条权威估值行（来自完整 state 字段，非稀疏 nav.parquet）
    for day in env.days:
        for arm_id in ("aa_1", "aa_2"):
            valuation = envmod.shard(prepared, arm_id, day)["evidence"]["valuation"]
            assert valuation["source"] == "engine_state.portfolio_values"
            assert valuation["row_count"] == 1
            assert (
                abs(valuation["cash"] + valuation["position_value"] - valuation["portfolio_value"])
                < 1e-6
            )


def test_pending_sell_on_limit_down_day_and_fill_next_day(tmp_path):
    """跌停阻断到期卖出 ⇒ pending_sells 队列；次日补卖；阻断日仍有完整估值行。"""
    blocked_day = envmod.DAYS[6]
    env = envmod.build_env(tmp_path, limit_down={blocked_day: [envmod.CODES[0], envmod.CODES[1]]})
    prepared = envmod.prepare_campaign(env)
    first = envmod.run_campaign(env, prepared, days=env.days[:7])
    assert first["published"]["published_generation"] == 7

    blocked = envmod.shard(prepared, "aa_1", blocked_day)["evidence"]
    assert blocked["fills"] == []
    assert blocked["valuation"]["row_count"] == 1
    assert blocked["t0_status"] in {"success", "not_rebalance_day", "no_targets"}
    queue = envmod.pending(prepared, "aa_1", "sells")
    assert {row["ts_code"] for row in queue} == {envmod.CODES[0], envmod.CODES[1]}
    assert all(int(row["attempts"]) >= 1 for row in queue)

    envmod.run_campaign(env, prepared)
    next_day = env.days[7]
    fills = envmod.shard(prepared, "aa_1", next_day)["evidence"]["fills"]
    sold = {(row["stock"], row["action"]) for row in fills}
    assert (envmod.CODES[0], "sell") in sold and (envmod.CODES[1], "sell") in sold
    assert envmod.pending(prepared, "aa_1", "sells") == []


def test_not_ready_pauses_whole_family_without_advancing(tmp_path):
    """单臂就绪门不过 ⇒ 整族停留同一虚拟日：attempt 终止、index 不写、后续日不推进。"""
    broken_day = envmod.DAYS[3]
    # 该日 panel 行数掉出行数带（[0.9,1.1] × 近 20 交易日中位数）⇒ 就绪门 ok=False
    env = envmod.build_env(tmp_path, row_subset={broken_day: 3})
    prepared = envmod.prepare_campaign(env)
    report = envmod.run_campaign(env, prepared)

    assert report["paused"] is not None
    assert report["paused"]["virtual_date"] == broken_day
    assert report["paused"]["reason"] == "not_ready"
    assert report["paused"]["index_written"] is False
    assert report["paused"]["advanced"] is False
    index = envmod.index(prepared)
    assert index["published_generation"] == 3
    assert index["published_virtual_date"] == env.days[2]
    committed = [r["virtual_date"] for r in report["transactions"] if r["mode"] == "committed"]
    assert committed == list(env.days[:3])

    last = envmod.journal(prepared, broken_day)["attempts"][-1]
    assert last["status"] == "ABORTED"
    assert last["abort"]["reason"] == "not_ready"
    # 父代未被推进（臂根仍为已发布代）
    for arm_id in ("aa_1", "aa_2"):
        root = prepared.root / "arms" / arm_id
        assert (root / "runs" / f"t1_{env.days[2]}.json").is_file()
        assert not (root / "runs" / f"t1_{broken_day}.json").exists()
    assert not (prepared.root / "ledger" / "aa_1" / f"{broken_day}.json").exists()


def test_published_generation_is_idempotent_and_never_rebooked(tmp_path):
    env, prepared = _fresh_campaign(tmp_path)
    envmod.run_campaign(env, prepared)
    before = envmod.index(prepared)
    shard_before = envmod.shard(prepared, "aa_1", env.days[5])
    ledger_before = _recursive_entries(prepared.root / "ledger")
    journal_before = envmod.journal(prepared, env.days[5])

    again = envmod.run_campaign(env, prepared)
    assert again["published"]["published_generation"] == 10
    assert all(record["mode"] == "already_published" for record in again["transactions"])
    assert envmod.index(prepared) == before
    assert envmod.shard(prepared, "aa_1", env.days[5]) == shard_before
    assert _recursive_entries(prepared.root / "ledger") == ledger_before
    # journal 不因幂等重跑改写（同一 attempt 记录逐字节一致）
    assert envmod.journal(prepared, env.days[5]) == journal_before
    # 入账只发生一次：trades.parquet 行数不翻倍
    store = PaperStateStore(root_path=str(prepared.root / "arms" / "aa_1"), create_layout=False)
    trades = store.load_all_trades()
    total = 0 if trades is None else len(trades)
    assert total == sum(
        len(envmod.shard(prepared, "aa_1", day)["evidence"]["fills"]) for day in env.days
    )


def test_day_requests_cannot_skip_or_reorder(tmp_path):
    env, prepared = _fresh_campaign(tmp_path)
    with pytest.raises(FamilyTxnError, match="连续推进"):
        envmod.run_campaign(env, prepared, days=[env.days[1], env.days[2]])
    with pytest.raises(FamilyTxnError, match="升序"):
        envmod.run_campaign(env, prepared, days=[env.days[1], env.days[0]])
    with pytest.raises(FamilyTxnError, match="窗口"):
        envmod.run_campaign(env, prepared, days=["20991231"])


def test_no_production_writes_and_no_ledger_entry(tmp_path):
    project_data = ROOT / "data"
    watched = {
        name: _recursive_entries(project_data / name) for name in ("paper", "runs", "models")
    }
    env, prepared = _fresh_campaign(tmp_path)
    envmod.run_campaign(env, prepared, write_report=True)
    for name, before in watched.items():
        assert _recursive_entries(project_data / name) == before, name
    # 产物只落 campaign 根
    assert prepared.root.is_relative_to(env.base_root)
    for name in ("arms", "ledger", "journal", "reports", "_index.json"):
        assert (prepared.root / name).exists(), name


def test_d0_fresh_record_survives_run_and_identity_is_registered(tmp_path):
    env, prepared = _fresh_campaign(tmp_path)
    record_sha = prepared.d0["record_sha256"]
    envmod.run_campaign(env, prepared)
    from src.lazybull.v2.hosts.testbed import load_d0_record

    assert load_d0_record(prepared.root)["record_sha256"] == record_sha
    # 驱动后 D0 空账户复核必然失效（不是空账户了）——登记语义，不复用 verify_d0_fresh
    with pytest.raises(Exception):
        verify_d0_fresh(prepared.root)
    report = envmod.run_campaign(env, prepared)
    assert report["identity"]["d0_fresh_record_sha256"] == record_sha


def test_report_identity_covers_five_segments(tmp_path):
    env, prepared = _fresh_campaign(tmp_path)
    report = envmod.run_campaign(env, prepared)
    identity = report["identity"]
    assert identity["window_state_id"]
    assert identity["execution_digest"]
    assert identity["d0_fresh_record_sha256"]
    summary = identity["campaign_identity_summary"]
    assert set(summary) >= {"input", "config", "model", "code", "window_state"}
    for arms in report["model_availability"].values():
        assert arms["trained_before_d0"] is True
    assert canonical_json_sha256({"k": 1}) == canonical_json_sha256({"k": 1})


def test_two_arms_registered_before_d0_only(tmp_path):
    env = envmod.build_env(tmp_path)
    campaign = register_campaign(
        campaign_id="t2_scope",
        d0_date=env.days[0],
        virtual_days=env.days,
        calendar=env.calendar,
        data_root=str(env.data_root),
        base_root=str(env.base_root),
        repo_root=str(env.repo_root),
        allow_external_root=True,
    )
    root = Path(campaign["campaign_root"])
    envmod_register = register_arm(
        campaign_root=root,
        arm_id="aa_1",
        arm_role="aa_clone",
        config_flat=dict(env.config),
        models_dir=str(env.models_dir),
        repo_root=str(env.repo_root),
        allow_external_root=True,
    )
    assert envmod_register["arm_id"] == "aa_1"
    initialize_d0_fresh(root, allow_external_root=True)
    # 只有单臂 ⇒ 运行可行但 A/A 不可判定（不冒称一致）
    report = run_virtual_campaign(
        root,
        days=[env.days[0]],
        host_factory=envmod.make_host_factory(env),
        allow_external_root=True,
    )
    assert report["aa_consistency"]["comparable"] is False
    assert report["aa_consistency"]["identical"] is True
    assert "note" in report["aa_consistency"]
    assert (
        json.loads((root / "_index.json").read_text(encoding="utf-8"))["published_generation"] == 1
    )
