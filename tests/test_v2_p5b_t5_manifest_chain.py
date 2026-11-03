# -*- coding: utf-8 -*-
"""P5b T5：回撤 manifest 冻结、字段/门槛校验、三环互链与上游链接。"""

from __future__ import annotations

import json
import shutil

import pytest

import p5b_t5_env
from src.lazybull.v2.hosts.testbed.referee.drawdown import (
    CALIBRATOR_MODULES,
    MANIFEST_SCHEMA,
    SCALE_FLOOR,
    ManifestError,
    calibrator_identity,
    freeze_manifest,
    inspect_parent,
    load_manifest,
    manifest_sha256,
    scale_floor_errors,
    validate_manifest,
    verify_calibration_chain,
    verify_upstream,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.artifacts import ARTIFACT_THRESHOLDS


def _doc(tmp_path):
    parent = p5b_t5_env.parent_campaign(tmp_path)
    doc = p5b_t5_env.t5_document(parent)
    return parent, doc


def test_document_validates_and_is_deterministic(tmp_path):
    parent, doc = _doc(tmp_path)
    assert validate_manifest(doc) == []
    assert doc["schema"] == MANIFEST_SCHEMA
    assert doc["lineage"]["sequence"] == 1
    again = p5b_t5_env.t5_document(parent)
    assert json.dumps(doc, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert doc["files"]["parent_manifest"]["sha256"] == parent.manifest_sha256


def test_missing_field_is_rejected(tmp_path):
    _, doc = _doc(tmp_path)
    broken = json.loads(json.dumps(doc))
    del broken["drawdown_referee"]["effect"]["grid_pp"]
    errors = validate_manifest(broken)
    assert any("grid_pp" in item for item in errors)


def test_gates_cannot_be_loosened(tmp_path):
    _, doc = _doc(tmp_path)
    loosened = json.loads(json.dumps(doc))
    loosened["drawdown_referee"]["solver"]["tolerance_pp"] = 0.5
    loosened["drawdown_referee"]["holdout"]["wilson_upper_max"] = 0.5
    loosened["drawdown_referee"]["windows"]["max_windows"] = 5
    errors = validate_manifest(loosened)
    assert len(errors) >= 3


def test_direction_and_event_fields_are_enforced(tmp_path):
    _, doc = _doc(tmp_path)
    tampered = json.loads(json.dumps(doc))
    tampered["drawdown_referee"]["threshold"]["improve_is_d_k"] = False
    tampered["drawdown_referee"]["threshold"]["improve_event"] = tampered["drawdown_referee"][
        "threshold"
    ]["worsen_event"]
    tampered["drawdown_referee"]["reading"]["t_anchor"] = 1714
    errors = validate_manifest(tampered)
    assert any("improve_is_d_k" in item for item in errors)
    assert any("事件名不得相同" in item for item in errors)
    assert any("t_anchor" in item for item in errors)


def test_aa_and_injection_fields_are_enforced(tmp_path):
    _, doc = _doc(tmp_path)
    tampered = json.loads(json.dumps(doc))
    tampered["drawdown_referee"]["aa"]["same_path_zero_injection_forbidden"] = False
    tampered["drawdown_referee"]["injection"]["dd_worsen"]["range"] = [0.0, 1.0]
    tampered["drawdown_referee"]["injection"]["dd_improve"]["parameter"] = "zeta"
    errors = validate_manifest(tampered)
    assert any("same_path_zero_injection_forbidden" in item for item in errors)
    assert any("dd_worsen.range" in item for item in errors)
    assert any("dd_improve.parameter" in item for item in errors)


def test_h6_placeholder_and_drawdown_window_list_required(tmp_path):
    _, doc = _doc(tmp_path)
    tampered = json.loads(json.dumps(doc))
    tampered["h6_numeric"]["status"] = "executed"
    tampered["drawdown_referee"]["windows"]["frozen_list"] = []
    errors = validate_manifest(tampered)
    assert any("h6_numeric.status" in item for item in errors)
    assert any("frozen_list" in item for item in errors)


def test_scale_floor_only_goes_up():
    assert scale_floor_errors(dict(SCALE_FLOOR)) == []
    below = {key: value - 1 for key, value in SCALE_FLOOR.items()}
    assert len(scale_floor_errors(below)) == len(SCALE_FLOOR)


def test_freeze_is_exclusive_and_byte_stable(tmp_path):
    parent, doc = _doc(tmp_path)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    assert freeze_manifest(doc, first)["sha256"] == freeze_manifest(doc, second)["sha256"]
    with pytest.raises(ManifestError):
        freeze_manifest(doc, first)


def test_load_detects_source_file_tamper(tmp_path):
    parent, doc = _doc(tmp_path)
    target = tmp_path / "manifest.json"
    written = freeze_manifest(doc, target)
    assert load_manifest(target, expected_sha256=written["sha256"], require_scale_floor=False)
    with pytest.raises(ManifestError):
        load_manifest(target, expected_sha256="0" * 64, require_scale_floor=False)


def test_load_detects_calibrator_code_change(tmp_path):
    parent, doc = _doc(tmp_path)
    target = tmp_path / "manifest.json"
    written = freeze_manifest(doc, target)
    copy_dir = p5b_t5_env.copy_modules(tmp_path / "modules", names=CALIBRATOR_MODULES)
    path = copy_dir / "mdd.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(
            target,
            expected_sha256=written["sha256"],
            require_scale_floor=False,
            module_dir=copy_dir,
        )


def test_verify_upstream_ok_and_hash_tamper(tmp_path):
    parent, doc = _doc(tmp_path)
    info = verify_upstream(doc, repo_root=parent.env.repo_root, require_scale_floor=False)
    assert info["parent_manifest"]["sha256"] == parent.manifest_sha256
    assert info["parent_drawdown_status"] == "not_implemented_in_t4"
    broken = json.loads(json.dumps(doc))
    broken["files"]["parent_thresholds"]["sha256"] = "f" * 64
    with pytest.raises(ManifestError):
        verify_upstream(broken, repo_root=parent.env.repo_root, require_scale_floor=False)


def test_verify_upstream_rejects_source_pair_mismatch(tmp_path):
    parent, doc = _doc(tmp_path)
    tampered = json.loads(json.dumps(doc))
    tampered["source_pair"]["base_sha256"] = "a" * 64
    with pytest.raises(ManifestError):
        verify_upstream(tampered, repo_root=parent.env.repo_root, require_scale_floor=False)


def test_verify_upstream_rejects_parent_byte_change(tmp_path):
    parent, doc = _doc(tmp_path)
    copy_dir = tmp_path / "parent_copy"
    shutil.copytree(parent.root, copy_dir)
    path = copy_dir / "reports/calibration_manifest.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = json.loads(json.dumps(doc))
    tampered["files"]["parent_manifest"]["path"] = path.as_posix()
    with pytest.raises(ManifestError):
        verify_upstream(tampered, repo_root=parent.env.repo_root, require_scale_floor=False)


def test_inspect_parent_exposes_hashes_and_calibration(tmp_path):
    parent = p5b_t5_env.parent_campaign(tmp_path)
    info = inspect_parent(parent.manifest_path, parent.thresholds_path, require_scale_floor=False)
    assert info["parent_thresholds"]["sha256"]
    assert "dgp" in info["parent_calibration"]
    assert info["parent_counts"]["seed"] == p5b_t5_env.PARENT_COUNTS["seed"]


def test_three_ring_chain_detects_tamper(tmp_path):
    campaign = p5b_t5_env.t5_campaign(p5b_t5_env.parent_campaign(tmp_path))
    thresholds_doc = json.loads(
        (campaign.root / "reports" / ARTIFACT_THRESHOLDS).read_text(encoding="utf-8")
    )
    thresholds_sha = manifest_sha256(campaign.root / "reports" / ARTIFACT_THRESHOLDS)
    identity = calibrator_identity()
    manifest_doc = load_manifest(
        campaign.root / "reports/drawdown_manifest.json", require_scale_floor=False
    )
    chain = verify_calibration_chain(
        manifest_doc=manifest_doc,
        manifest_file_sha256=campaign.frozen["sha256"],
        calibrator=identity,
        thresholds=thresholds_doc,
        thresholds_sha256=thresholds_sha,
    )
    assert chain["ok"] is True
    tampered = dict(thresholds_doc)
    tampered["manifest_sha256"] = "b" * 64
    with pytest.raises(ManifestError):
        verify_calibration_chain(
            manifest_doc=manifest_doc,
            manifest_file_sha256=campaign.frozen["sha256"],
            calibrator=identity,
            thresholds=tampered,
            thresholds_sha256=thresholds_sha,
        )


def test_lineage_requires_supersede_record(tmp_path):
    from src.lazybull.v2.hosts.testbed.referee.drawdown import lineage_section

    _, doc = _doc(tmp_path)
    base = json.loads(json.dumps(doc))
    base["lineage"] = lineage_section(sequence=1, supersede_reason="x" * 20)
    assert any("sequence=1" in item for item in validate_manifest(base))
    base["lineage"] = lineage_section(sequence=2)
    assert any("superseded_manifest_sha256" in item for item in validate_manifest(base))
    base["lineage"] = lineage_section(
        sequence=2,
        superseded_manifest_sha256="c" * 64,
        superseded_calibrator_sha256="d" * 64,
        supersede_reason="重冻：补字段并收紧阈值口径说明",
    )
    assert validate_manifest(base) == []


def test_calibrator_exclusions_are_frozen(tmp_path):
    """未纳入身份的模块与理由必须写进 manifest（否则改 report.py 不会被身份拦住）。"""
    from src.lazybull.v2.hosts.testbed.referee.drawdown import CALIBRATOR_EXCLUSIONS

    _, doc = _doc(tmp_path)
    exclusions = doc["files"]["calibrator"]["exclusions"]
    assert exclusions == dict(CALIBRATOR_EXCLUSIONS)
    assert "report.py" in exclusions
    broken = json.loads(json.dumps(doc))
    broken["files"]["calibrator"]["exclusions"] = {}
    assert any("exclusions" in item for item in validate_manifest(broken))


def test_named_seed_deviation_is_registered(tmp_path):
    _, doc = _doc(tmp_path)
    items = {entry["item"] for entry in doc["deviation_registry"]}
    assert "命名种子" in items
    broken = json.loads(json.dumps(doc))
    broken["deviation_registry"] = [
        entry for entry in broken["deviation_registry"] if entry["item"] != "命名种子"
    ]
    assert any("命名种子" in item for item in validate_manifest(broken))
