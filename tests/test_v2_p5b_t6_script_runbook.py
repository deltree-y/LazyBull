# -*- coding: utf-8 -*-
"""P5b T6 薄入口与运行手册测试（tests/ 豁免复杂度）。

覆盖：脚本可执行（``--help``）、单批外推授权门（>20 分钟必须要求授权而非缩量）、
授权记录字段校验、运行手册章节/产物名/哈希块一致性、真实产物-文档一致性（存在时）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t2_env as t2env  # noqa: E402
import p5b_t6_env as t6env  # noqa: E402

from src.lazybull.v2.hosts.testbed.rehearsal import artifacts  # noqa: E402
from src.lazybull.v2.hosts.testbed.rehearsal.assembly import (  # noqa: E402
    doc_consistency_hashes,
    existing_artifact_hashes,
    verify_runbook,
)
from src.lazybull.v2.hosts.testbed.rehearsal.runner import (  # noqa: E402
    LongTaskAuthorizationRequired,
    load_authorization,
    project_batch_seconds,
)
from src.lazybull.v2.hosts.testbed.storage import seal_record  # noqa: E402

SCRIPT = ROOT / "scripts" / "v2_p5b" / "rehearsal.py"
RUNBOOK = ROOT / "docs" / "guide" / "p5b_testbed_runbook.md"
REAL_CAMPAIGN = ROOT / "data" / "paper_shadow" / "p5b_testbed" / "p5b_t6_rehearsal_20261103_r15"


def test_script_help_is_executable():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert completed.returncode == 0
    assert "--freeze" in completed.stdout and "--check" in completed.stdout


def test_projection_gate_uses_measured_single_day():
    projection = project_batch_seconds(23.0, 79)
    assert projection["exceeds_threshold"] is True
    assert projection["projected_seconds"] == pytest.approx(23.0 * 79)
    short = project_batch_seconds(23.0, 10)
    assert short["exceeds_threshold"] is False
    assert "五臂族级" in short["basis"]


def test_authorization_record_requires_all_fields(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"authorized_at": "2026-11-02"}), encoding="utf-8")
    with pytest.raises(Exception):
        load_authorization(str(path))
    no_scope = {
        "authorized_at": "2026-11-02T21:00:00",
        "authorized_by": "user",
        "quote": "同意在约 30 分钟内跑完 79 日",
        "basis": "单日实测 23s × 79",
    }
    path.write_text(json.dumps(no_scope, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Exception):
        load_authorization(str(path))  # 缺 scope ⇒ 拒绝（授权必须绑定范围）
    complete = {
        **no_scope,
        "scope": {"campaign_id": "t6_gate", "max_virtual_days": 79},
    }
    path.write_text(json.dumps(complete, ensure_ascii=False), encoding="utf-8")
    loaded = load_authorization(str(path))
    assert loaded["authorized_by"] == "user" and len(loaded["sha256"]) == 64
    assert loaded["scope"]["max_virtual_days"] == 79
    assert load_authorization(None) is None


def _gate_campaign(tmp_path) -> tuple:
    """准备一个已注册 campaign + 冻结件副本（供授权门用例）。"""
    env = t2env.build_env(tmp_path)
    repo = t6env.build_repo(tmp_path)
    prepared = t2env.prepare_campaign(
        env,
        campaign_id="t6_gate",
        arms=("base", "mech_aa"),
        roles={"base": "aa_clone", "mech_aa": "aa_clone"},
    )
    document = t6env.manifest_document(env, repo, campaign_id="t6_gate")
    target = prepared.root / "reports" / "rehearsal_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(seal_record(document), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return env, repo, prepared


def _seed_baseline(prepared, env, seconds: float = 300.0) -> None:
    """写入一条单日基线（模拟已跑过的墙钟探针）。"""
    path = prepared.root / "reports" / "machine_time_run.json"
    path.write_text(
        json.dumps(
            {
                "stage": "probe",
                "days": [env.days[0]],
                "per_day_seconds": {env.days[0]: seconds},
                "batch_seconds": seconds,
                "history": [
                    {
                        "stage": "probe",
                        "days": [env.days[0]],
                        "per_day_seconds": {env.days[0]: seconds},
                        "batch_seconds": seconds,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_run_refuses_without_authorization_when_projection_exceeds(tmp_path, monkeypatch):
    from src.lazybull.v2.hosts.testbed.rehearsal import runner as runner_mod

    env, repo, prepared = _gate_campaign(tmp_path)
    _seed_baseline(prepared, env)
    fake_report = {
        "published": {"published_generation": 1},
        "machine_time": {"per_day_seconds": {env.days[0]: 300.0}},
        "aa_consistency": {},
    }
    monkeypatch.setattr(runner_mod, "campaign_root_for", lambda campaign_id: prepared.root)
    monkeypatch.setattr(runner_mod, "run_virtual_campaign", lambda *a, **k: fake_report)
    with pytest.raises(LongTaskAuthorizationRequired) as excinfo:
        runner_mod.run_rehearsal(repo_root=repo, campaign_id="t6_gate", allow_external_root=True)
    assert excinfo.value.projection["exceeds_threshold"] is True
    assert excinfo.value.projection["projected_seconds"] > 1200
    grant = {
        "sha256": "a" * 64,
        "authorized_by": "user",
        "scope": {"campaign_id": "t6_gate", "max_virtual_days": 10},
    }
    # 范围不符（授权只覆盖 5 日）⇒ 同样拒绝
    with pytest.raises(LongTaskAuthorizationRequired):
        runner_mod.run_rehearsal(
            repo_root=repo,
            campaign_id="t6_gate",
            allow_external_root=True,
            authorization={**grant, "scope": {"campaign_id": "t6_gate", "max_virtual_days": 5}},
        )
    # 别的 campaign 的授权 ⇒ 直接报错
    with pytest.raises(Exception):
        runner_mod.run_rehearsal(
            repo_root=repo,
            campaign_id="t6_gate",
            allow_external_root=True,
            authorization={
                **grant,
                "scope": {"campaign_id": "other_campaign", "max_virtual_days": 79},
            },
        )
    outcome = runner_mod.run_rehearsal(
        repo_root=repo,
        campaign_id="t6_gate",
        allow_external_root=True,
        authorization=grant,
    )
    assert outcome["long_task_authorization"]["authorized_by"] == "user"


def test_run_refuses_without_baseline_for_multi_day_batch(tmp_path, monkeypatch):
    """无基线且待跑 > 1 日 ⇒ 必须先跑探针或提供授权（门在跑前判定）。"""
    from src.lazybull.v2.hosts.testbed.rehearsal import runner as runner_mod

    env, repo, prepared = _gate_campaign(tmp_path)
    monkeypatch.setattr(runner_mod, "campaign_root_for", lambda campaign_id: prepared.root)
    called = {"n": 0}

    def _fake(*args, **kwargs):
        called["n"] += 1
        return {
            "published": {"published_generation": 1},
            "machine_time": {"per_day_seconds": {env.days[0]: 1.0}},
            "aa_consistency": {},
        }

    monkeypatch.setattr(runner_mod, "run_virtual_campaign", _fake)
    with pytest.raises(LongTaskAuthorizationRequired):
        runner_mod.run_rehearsal(repo_root=repo, campaign_id="t6_gate", allow_external_root=True)
    assert called["n"] == 0


def test_runbook_required_sections_and_hashes(tmp_path):
    repo = t6env.build_repo(tmp_path)
    hashes = {
        artifacts.ARTIFACT_REPORT: "a" * 64,
        artifacts.ARTIFACT_VERDICT: "b" * 64,
    }
    (repo / "docs" / "guide" / "p5b_testbed_runbook.md").write_text(
        t6env.runbook_text(hashes), encoding="utf-8"
    )
    verified = verify_runbook(repo, artifact_hashes=hashes)
    assert verified["verified"] is True, verified["mismatches"]
    tampered = verify_runbook(repo, artifact_hashes={**hashes, artifacts.ARTIFACT_REPORT: "c" * 64})
    assert tampered["verified"] is False
    assert any("hash_mismatch" in item for item in tampered["mismatches"])


def test_runbook_missing_is_not_verified(tmp_path):
    repo = tmp_path / "empty_repo"
    repo.mkdir()
    assert verify_runbook(repo, artifact_hashes={})["verified"] is False


def test_runbook_matches_real_artifacts_when_present():
    verdict = REAL_CAMPAIGN / "reports" / artifacts.ARTIFACT_VERDICT
    if not (RUNBOOK.is_file() and verdict.is_file()):
        pytest.skip("真实演练产物或运行手册尚未生成")
    document = json.loads(
        (REAL_CAMPAIGN / "reports" / artifacts.ARTIFACT_MANIFEST).read_text(encoding="utf-8")
    )
    hashes = {
        **existing_artifact_hashes(REAL_CAMPAIGN),
        **doc_consistency_hashes(REAL_CAMPAIGN, document),
    }
    manifest = json.loads(
        (REAL_CAMPAIGN / "reports" / artifacts.ARTIFACT_MANIFEST).read_text(encoding="utf-8")
    )
    result = verify_runbook(
        ROOT,
        artifact_hashes=hashes,
        campaign_id=str(REAL_CAMPAIGN.name),
        lineage_sequence=int(manifest["lineage"]["sequence"]),
    )
    assert result["verified"] is True, result["mismatches"]


def test_real_verdict_has_no_gate_pass_field_when_present():
    from src.lazybull.v2.hosts.testbed.rehearsal.assembly import forbidden_gate_fields

    verdict_path = REAL_CAMPAIGN / "reports" / artifacts.ARTIFACT_VERDICT
    if not verdict_path.is_file():
        pytest.skip("真实判定表尚未生成")
    payload = json.loads(verdict_path.read_text(encoding="utf-8"))
    assert forbidden_gate_fields(payload) == []
    assert payload["h6_claim"] == "evaluated_not_declared_pass"
    assert payload["h7_claim"] == "evaluated_not_declared_pass"
    assert payload["gate_owner"] == "user_at_merge_acceptance"
