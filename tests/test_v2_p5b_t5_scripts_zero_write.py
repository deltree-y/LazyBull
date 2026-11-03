# -*- coding: utf-8 -*-
"""P5b T5：零写入、产物落位、重复运行拒绝、同 seed 逐字节一致、脚本端到端。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import p5b_t5_env
from src.lazybull.v2.hosts.testbed.referee.drawdown import (
    ARTIFACT_HOLDOUT,
    ARTIFACT_MANIFEST,
    ARTIFACT_POWER,
    ARTIFACT_REPORT,
    ARTIFACT_REPORT_MD,
    ARTIFACT_THRESHOLDS,
    ARTIFACT_VERDICT,
    ArtifactError,
    ManifestError,
    freeze_drawdown_manifest,
    run_drawdown_holdout,
    write_json_once,
)

ROOT = p5b_t5_env.ROOT
SCRIPTS = ROOT / "scripts" / "v2_p5b"
#: 确定性产物（不含 machine_time）
DETERMINISTIC_ARTIFACTS = (
    ARTIFACT_MANIFEST,
    ARTIFACT_THRESHOLDS,
    ARTIFACT_HOLDOUT,
    ARTIFACT_POWER,
    ARTIFACT_VERDICT,
    ARTIFACT_REPORT,
    ARTIFACT_REPORT_MD,
)


def _run(script: str, args, *, env_root: Path) -> subprocess.CompletedProcess:
    import os

    command = [sys.executable, str(SCRIPTS / script), *[str(item) for item in args]]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYTHONIOENCODING": "utf-8",
            "TEMP": str(env_root),
            "TMP": str(env_root),
        }
    )
    return subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )


def test_no_production_writes_and_artifacts_only_in_reports(tmp_path):
    before = p5b_t5_env.production_snapshot()
    t5_before = p5b_t5_env.t4_campaign_state()
    repo_before = p5b_t5_env.t4_repo_files_state()
    built = p5b_t5_env.full_t5(tmp_path)
    assert p5b_t5_env.production_snapshot() == before
    assert p5b_t5_env.t4_campaign_state() == t5_before
    assert p5b_t5_env.t4_repo_files_state() == repo_before
    reports = built.campaign.root / "reports"
    for name in DETERMINISTIC_ARTIFACTS:
        assert (reports / name).is_file()
    outside = [
        path
        for path in built.campaign.root.rglob("*")
        if path.is_file() and reports not in path.parents
    ]
    assert outside == []


def test_source_has_no_ledger_entry_and_no_production_writes():
    """源码级守卫：回撤子包不写 LedgerEntry、不联网、不引用生产写入口与交易链路。"""
    forbidden = (
        "LedgerEntry(",
        "tushare",
        "akshare",
        "pending_buys",
        "pending_sells",
        "from src.lazybull.paper",
        "from src.lazybull.trading",
        "from src.lazybull.ml",
        "write_json_atomic(",
    )
    for where, line in p5b_t5_env.iter_source_lines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for token in forbidden:
            assert token not in line, f"{where} 出现禁用写入标记 {token}: {line}"


def test_artifacts_are_exclusive(tmp_path):
    built = p5b_t5_env.full_t5(tmp_path)
    reports = built.campaign.root / "reports"
    with pytest.raises(ArtifactError):
        write_json_once(reports / ARTIFACT_VERDICT, {"schema": "x"}, what="重复判定表")
    with pytest.raises(ManifestError):
        freeze_drawdown_manifest(
            built.campaign.root,
            built.campaign.doc,
            apply=True,
            allow_external_root=True,
        )
    with pytest.raises(Exception):
        run_drawdown_holdout(
            built.campaign.root,
            repo_root=built.parent.env.repo_root,
            apply=True,
            allow_external_root=True,
            require_scale_floor=False,
        )


def test_holdout_dry_run_writes_nothing(tmp_path):
    campaign = p5b_t5_env.t5_campaign(p5b_t5_env.parent_campaign(tmp_path))
    reports = campaign.root / "reports"
    before = p5b_t5_env.tree_state(reports)
    outcome = p5b_t5_env.run_holdout(campaign, apply=False)
    assert outcome["dry_run"] is True
    assert p5b_t5_env.tree_state(reports) == before


def test_same_seed_rerun_is_byte_identical(tmp_path):
    first = p5b_t5_env.full_t5(tmp_path / "a")
    second = p5b_t5_env.full_t5(tmp_path / "b")
    first_reports = first.campaign.root / "reports"
    second_reports = second.campaign.root / "reports"
    for name in DETERMINISTIC_ARTIFACTS:
        assert (
            p5b_t5_env.tree_state(first_reports)[name]
            == p5b_t5_env.tree_state(second_reports)[name]
        ), f"{name} 不是逐字节一致"
    # 不同 seed 必须改变产物（证明种子真的进入读数）
    third = p5b_t5_env.full_t5(tmp_path / "c", seed=20261104)
    assert p5b_t5_env.tree_state(third.campaign.root / "reports")[ARTIFACT_THRESHOLDS] != (
        p5b_t5_env.tree_state(first_reports)[ARTIFACT_THRESHOLDS]
    )


def test_report_carries_scope_and_h5_boundary(tmp_path):
    built = p5b_t5_env.full_t5(tmp_path)
    report = json.loads(
        (built.campaign.root / "reports" / ARTIFACT_REPORT).read_text(encoding="utf-8")
    )
    assert report["scope"] == "historical_testbed"
    assert report["evidence_class"] == "calibration_only"
    assert report["row_domain"] == "structural"
    assert report["schedule_mode"] == "replay"
    assert report["virtual_clock"] is True
    assert report["not_forward_evidence"] is True
    assert report["not_sla_evidence"] is True
    assert report["h5_boundary"]["h5_claim"] == "evaluated_not_declared_pass"
    assert report["h5_boundary"]["gate_owner"] == "user_at_merge_acceptance"
    verdict = json.loads(
        (built.campaign.root / "reports" / ARTIFACT_VERDICT).read_text(encoding="utf-8")
    )
    assert verdict["h5_boundary"]["h5_claim"] == "evaluated_not_declared_pass"
    assert "h5_pass" not in json.dumps(verdict).lower()


def test_cli_freeze_and_train_dry_run_write_nothing(tmp_path):
    # 脚本入口要求父仪器自身也过 §5.3 下限（生产的父仪器即 2000/10000/2000）
    parent = p5b_t5_env.parent_campaign(tmp_path, counts=p5b_t5_env.PARENT_FLOOR_COUNTS)
    campaign_id = "t5_cli_dry"
    campaign_root = parent.env.repo_root / "campaigns" / campaign_id
    campaign_root.mkdir(parents=True, exist_ok=True)
    window = p5b_t5_env.source_window(parent.env)
    outcome = _run(
        "drawdown_calibration.py",
        [
            "--campaign-root",
            campaign_root,
            "--freeze",
            "--parent-manifest",
            parent.manifest_path,
            "--parent-manifest-sha256",
            parent.manifest_sha256,
            "--parent-thresholds",
            parent.thresholds_path,
            "--parent-thresholds-sha256",
            p5b_t5_env._sha256(parent.thresholds_path),
            "--repo-root",
            parent.env.repo_root,
            "--repo-manifest",
            campaign_root / "repo_copy.json",
            "--expected-window-mdd",
            window.mdd_percent,
            "--allow-external-root",
        ],
        env_root=tmp_path,
    )
    assert outcome.returncode == 0, outcome.stderr
    assert not (campaign_root / "reports").exists()


def test_export_table_is_read_only(tmp_path):
    built = p5b_t5_env.full_t5(tmp_path)
    before = p5b_t5_env.tree_state(built.campaign.root)
    outcome = _run(
        "drawdown_export_table.py",
        ["--campaign-root", built.campaign.root, "--allow-external-root"],
        env_root=tmp_path,
    )
    assert outcome.returncode == 0, outcome.stderr
    assert "h5_claim=evaluated_not_declared_pass" in outcome.stdout
    assert p5b_t5_env.tree_state(built.campaign.root) == before


def test_repo_manifest_does_not_bypass_write_scope(tmp_path):
    """`--repo-manifest` 同样过写域守卫（生产根禁令不得被该参数绕过）。"""
    parent = p5b_t5_env.parent_campaign(tmp_path, counts=p5b_t5_env.PARENT_FLOOR_COUNTS)
    campaign_root = parent.env.repo_root / "campaigns" / "t5_scope"
    campaign_root.mkdir(parents=True, exist_ok=True)
    forbidden = p5b_t5_env.ROOT / "data" / "paper" / "t5_forbidden.json"
    window = p5b_t5_env.source_window(parent.env)
    outcome = _run(
        "drawdown_calibration.py",
        [
            "--campaign-root",
            campaign_root,
            "--freeze",
            "--apply",
            "--parent-manifest",
            parent.manifest_path,
            "--parent-manifest-sha256",
            parent.manifest_sha256,
            "--parent-thresholds",
            parent.thresholds_path,
            "--parent-thresholds-sha256",
            p5b_t5_env._sha256(parent.thresholds_path),
            "--repo-root",
            parent.env.repo_root,
            "--repo-manifest",
            forbidden,
            "--expected-window-mdd",
            window.mdd_percent,
            "--allow-external-root",
        ],
        env_root=tmp_path,
    )
    assert outcome.returncode != 0
    assert "禁止写生产根" in (outcome.stderr + outcome.stdout)
    assert not forbidden.exists()


def test_chunked_and_unchunked_runs_agree(tmp_path):
    """分块与不分块必须逐字节一致（只影响峰值内存，不影响读数）。"""
    first = p5b_t5_env.full_t5(tmp_path / "a")
    parent = first.parent
    campaign = p5b_t5_env.t5_campaign(parent, campaign_id="t5_unchunked")
    unchunked = p5b_t5_env.run_holdout(campaign, chunk=None)
    chunked = first.holdout
    assert unchunked["usable_by_direction"] == chunked["usable_by_direction"]
    assert unchunked["mde_worsen"]["by_scene"] == chunked["mde_worsen"]["by_scene"]
    assert unchunked["mde_improve"]["by_scene"] == chunked["mde_improve"]["by_scene"]
    thresholds = p5b_t5_env.tree_state(campaign.root / "reports")
    reference = p5b_t5_env.tree_state(first.campaign.root / "reports")
    for name in (ARTIFACT_THRESHOLDS,):
        assert thresholds[name] == reference[name]


def test_report_and_verdict_carry_review_hardening_fields(tmp_path):
    """终审整改字段的回归保护（可比性标记 / 分方向 injection / MDE 场景明细 / 不可达标注）。"""
    built = p5b_t5_env.full_t5(tmp_path)
    report = json.loads(
        (built.campaign.root / "reports" / ARTIFACT_REPORT).read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (built.campaign.root / "reports" / ARTIFACT_VERDICT).read_text(encoding="utf-8")
    )
    injection = verdict["h5_conditions"]["injection"]
    assert injection["target_convention"].startswith("DD-I")
    assert set(injection["ok_by_direction"]) == {"dd_worsen", "dd_improve"}
    assert injection["cells_comparable"] <= injection["cells_total"]
    assert "boundary_degeneracy_note" in injection
    mde_condition = verdict["h5_conditions"]["mde"]
    assert "scenes_registered" in mde_condition and "gate_scope" in mde_condition
    cell = next(
        entry
        for levels in report["effects"]["by_direction_level_scene"].values()
        for scenes in levels.values()
        for entry in scenes.values()
    )
    assert "rate_not_comparable" in cell and "constructibility_floor" in cell
    referee = verdict["drawdown_referee"]
    assert "worsen_unreachable_in_window" in referee
    assert "worsen_unreachable_reason" in referee
    assert "power_curve_dips" in referee["MDE_drawdown_improve"]
