# -*- coding: utf-8 -*-
"""P5b T3 分片链测试（tests/ 全量豁免复杂度）。

覆盖：链 digest 的确定性、缺片哨兵、篡改定位（臂 + 日期）、链产物校验与改写拒绝、
campaign/臂绑定（禁止跨 campaign 复用链）。
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

import p5b_t3_env as envmod  # noqa: E402

#: 用例按需选择臂数（三臂 D0 回放是主要固定成本）
ONE_ARM = ("aa_1",)
TWO_ARMS = ("aa_1", "aa_2")

from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    CHAIN_LINK_MISSING,
    FamilyTxnError,
    build_chain,
    campaign_chain_digest,
    chain_seed,
    ledger_paths,
    layout_for,
    verify_chain,
    write_ledger_artifacts,
    audit_ledger,
)


def _links(count: int = 3, *, missing: int | None = None) -> list[dict]:
    return [
        {
            "virtual_date": f"202607{6 + index:02d}",
            "generation": index + 1,
            "family_txn_id": f"txn-{index + 1}",
            "shard_sha256": None if missing == index else f"sha-{index}",
        }
        for index in range(count)
    ]


def test_chain_digest_is_deterministic_and_content_sensitive():
    first = build_chain("camp", "aa_1", _links())
    second = build_chain("camp", "aa_1", _links())
    assert first == second
    assert first["link_count"] == 3
    assert [link["gen_id"] for link in first["links"]] == ["g0001", "g0002", "g0003"]

    altered = _links()
    altered[1]["shard_sha256"] = "sha-tampered"
    changed = build_chain("camp", "aa_1", altered)
    assert changed["chain_digest"] != first["chain_digest"]
    # 首节不受影响、次节起全部变化（链式传播）
    assert changed["links"][0]["chain_digest"] == first["links"][0]["chain_digest"]
    assert changed["links"][1]["chain_digest"] != first["links"][1]["chain_digest"]
    assert changed["links"][2]["chain_digest"] != first["links"][2]["chain_digest"]


def test_missing_shard_uses_sentinel_and_returns_after_fill():
    intact = build_chain("camp", "aa_1", _links())
    broken = build_chain("camp", "aa_1", _links(missing=1))
    assert broken["links"][1]["shard_sha256"] is None
    assert broken["chain_digest"] != intact["chain_digest"]
    # 缺片按 journal 补齐后链 digest 回到原值（可验证性）
    assert build_chain("camp", "aa_1", _links())["chain_digest"] == intact["chain_digest"]
    assert CHAIN_LINK_MISSING == "MISSING"


def test_chain_rejects_out_of_order_links():
    links = _links()
    with pytest.raises(FamilyTxnError, match="代序"):
        build_chain("camp", "aa_1", [links[1], links[0], links[2]])
    with pytest.raises(FamilyTxnError, match="日期非严格升序|代序"):
        build_chain("camp", "aa_1", [links[0], links[0]])


def test_chain_seed_binds_campaign_and_arm():
    assert chain_seed("camp_a", "aa_1") != chain_seed("camp_b", "aa_1")
    assert chain_seed("camp_a", "aa_1") != chain_seed("camp_a", "aa_2")
    per_arm = {
        "aa_1": build_chain("camp", "aa_1", _links()),
        "aa_2": build_chain("camp", "aa_2", _links()),
    }
    assert campaign_chain_digest(per_arm) == campaign_chain_digest(per_arm)
    per_arm["aa_2"] = build_chain("camp", "aa_2", _links(missing=0))
    assert campaign_chain_digest(per_arm) != campaign_chain_digest(
        {"aa_1": per_arm["aa_1"], "aa_2": build_chain("camp", "aa_2", _links())}
    )


def test_tampering_any_published_shard_changes_chain_and_locates_arm_and_date(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=TWO_ARMS)
    envmod.run(env, prepared, env.days[:3])
    layout = layout_for(prepared.root)
    first = audit_ledger(layout, envmod.CAMPAIGN_ID)
    write_ledger_artifacts(layout, envmod.CAMPAIGN_ID, first)
    before_chain = dict(first["chain"]["chain_digests"])
    assert before_chain["aa_1"] != before_chain["aa_2"]

    envmod.tamper_shard(prepared, "aa_2", env.days[1])
    after = audit_ledger(layout, envmod.CAMPAIGN_ID)
    assert after["chain"]["chain_digests"]["aa_1"] == before_chain["aa_1"]
    assert after["chain"]["chain_digests"]["aa_2"] != before_chain["aa_2"]
    assert after["chain"]["chain_digest"] != first["chain"]["chain_digest"]
    bad = [gap for gap in after["gaps"] if gap["kind"] == "bad_shard"]
    assert [(gap["arm_id"], gap["virtual_date"]) for gap in bad] == [("aa_2", env.days[1])]
    assert bad[0]["repairable"] is False
    assert after["status"] == "INVALIDATED"


def test_genesis_shape_chain_artifact_verifies(tmp_path):
    """0 已发布日的链产物必须可 verify（campaign 级 digest 为 None，与重算同口径）。"""
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, [])  # 只建 genesis，不发布任何虚拟日
    layout = layout_for(prepared.root)
    write_ledger_artifacts(layout, envmod.CAMPAIGN_ID, audit_ledger(layout, envmod.CAMPAIGN_ID))
    document = json.loads(ledger_paths(layout).chain_path.read_text(encoding="utf-8"))
    assert document["chain_digest"] is None
    rebuilt = verify_chain(document, envmod.CAMPAIGN_ID)
    assert sorted(rebuilt) == sorted(prepared.arms)
    assert all(payload["link_count"] == 0 for payload in rebuilt.values())


def test_chain_artifact_verifies_and_rejects_rewrite(tmp_path):
    env = envmod.build_env(tmp_path)
    prepared = envmod.prepare(env, arms=ONE_ARM)
    envmod.run(env, prepared, env.days[:2])
    layout = layout_for(prepared.root)
    write_ledger_artifacts(layout, envmod.CAMPAIGN_ID, audit_ledger(layout, envmod.CAMPAIGN_ID))
    document = json.loads(ledger_paths(layout).chain_path.read_text(encoding="utf-8"))
    rebuilt = verify_chain(document, envmod.CAMPAIGN_ID)
    assert sorted(rebuilt) == sorted(prepared.arms)

    document["arms"]["aa_1"]["links"][0]["shard_sha256"] = "sha-forged"
    with pytest.raises(FamilyTxnError, match="无法重算一致|与重算不一致"):
        verify_chain(document, envmod.CAMPAIGN_ID)
