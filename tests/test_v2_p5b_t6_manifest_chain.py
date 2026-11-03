# -*- coding: utf-8 -*-
"""P5b T6 manifest 链测试（tests/ 豁免复杂度）。

覆盖：字段/窗口/H1 下限/五臂角色/δ 倍数/λ 界/限制声明/谱系/冻结独占/校准器身份/
上游哈希/冻结窗哈希/仓库副本逐字节/三环互链。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t6_env as t6env  # noqa: E402

from src.lazybull.v2.hosts.testbed.rehearsal.manifest import (  # noqa: E402
    ManifestError,
    build_manifest_document,
    calibrator_identity,
    freeze_manifest,
    load_manifest,
    validate_manifest,
    verify_chain,
    verify_upstream,
)

DAYS = tuple(
    f"2026{month:02d}{day:02d}"
    for month, day in [
        (7, 6),
        (7, 7),
        (7, 8),
        (7, 9),
        (7, 10),
        (7, 13),
        (7, 14),
        (7, 15),
        (7, 16),
        (7, 17),
    ]
)


def _env(tmp_path: Path) -> SimpleNamespace:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(data_root=data_root, base_root=tmp_path / "testbed", days=DAYS)


def _document(tmp_path: Path, **kwargs) -> dict:
    env = _env(tmp_path)
    repo = t6env.build_repo(tmp_path)
    return t6env.manifest_document(env, repo, days=kwargs.pop("days", None), **kwargs)


def test_document_validates_and_is_deterministic(tmp_path):
    first = _document(tmp_path)
    second = _document(tmp_path)
    assert validate_manifest(first) == []
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["window"]["day_count"] == len(DAYS)
    assert first["window"]["return_days"] == len(DAYS) - 1
    assert first["counts"]["arms"] == 5
    assert first["lineage"]["sequence"] == 1


def test_missing_required_field_is_rejected(tmp_path):
    document = _document(tmp_path)
    broken = copy.deepcopy(document)
    broken.pop("upstream")
    errors = validate_manifest(broken)
    assert any("upstream" in error for error in errors)


def test_window_below_h1_floor_is_rejected(tmp_path):
    document = _document(tmp_path, days=DAYS[:9])
    errors = validate_manifest(document)
    assert any("至少 10 日" in error for error in errors)


def test_non_structural_day_is_rejected(tmp_path):
    document = _document(tmp_path)
    broken = copy.deepcopy(document)
    broken["window"]["virtual_days"][0] = "20260702"
    broken["window"]["d0_date"] = "20260702"
    errors = validate_manifest(broken)
    assert any("行域非 structural" in error for error in errors)


def test_arm_renaming_and_delta_multiple_are_rejected(tmp_path):
    document = _document(tmp_path)
    renamed = copy.deepcopy(document)
    renamed["arms"]["base_clone"] = renamed["arms"].pop("mech_aa")
    assert any("五臂" in error for error in validate_manifest(renamed))
    scaled = copy.deepcopy(document)
    scaled["arms"]["income_minus_5d"]["injection"]["delta_multiple"] = 3.0
    assert any("5.0" in error for error in validate_manifest(scaled))


def test_lambda_and_anchor_out_of_contract_are_rejected(tmp_path):
    document = _document(tmp_path, parameters={"dd_improve": 1.5, "dd_worsen": 0.2})
    assert any("参数必须落在" in error for error in validate_manifest(document))
    broken = copy.deepcopy(_document(tmp_path))
    broken["arms"]["dd_worsen_12pp"]["injection"]["anchor"] = 240
    assert any("锚必须是 500" in error for error in validate_manifest(broken))


def test_limitations_must_cover_all_5_6_markers(tmp_path):
    document = _document(tmp_path)
    broken = copy.deepcopy(document)
    broken["limitations"] = ["只有一条限制"]
    errors = validate_manifest(broken)
    assert any("至少 4 条" in error for error in errors)
    partial = copy.deepcopy(document)
    partial["limitations"] = [text for text in document["limitations"] if "MA(H−1)" not in text]
    assert any("ma_overlap" in error for error in validate_manifest(partial))


def test_lineage_first_freeze_and_refreeze_rules(tmp_path):
    document = _document(tmp_path)
    assert document["lineage"]["sequence"] == 1
    broken = copy.deepcopy(document)
    broken["lineage"]["sequence"] = 2
    # 重冻必须登记前身哈希与原因，否则拒绝
    assert any("superseded_manifest_sha256" in error for error in validate_manifest(broken))
    refrozen = copy.deepcopy(document)
    refrozen["lineage"] = {
        "sequence": 2,
        "policy": "重冻策略",
        "superseded_manifest_sha256": "a" * 64,
        "superseded_calibrator_sha256": "b" * 64,
        "supersede_reason": "评审整改后重冻",
    }
    assert validate_manifest(refrozen) == []
    zero = copy.deepcopy(document)
    zero["lineage"]["sequence"] = 0
    assert any(">= 1" in error for error in validate_manifest(zero))


def test_freeze_is_exclusive_and_sealed(tmp_path):
    document = _document(tmp_path)
    target = tmp_path / "docs" / "plans" / "v2" / "p5b_t6_rehearsal_manifest_2026-11-02.json"
    written = freeze_manifest(document, target)
    assert target.is_file() and len(written["sha256"]) == 64
    with pytest.raises(ManifestError):
        freeze_manifest(document, target)


def test_load_rejects_calibrator_identity_mismatch(tmp_path):
    document = _document(tmp_path)
    corrupted = copy.deepcopy(document)
    corrupted["files"]["calibrator"]["sha256"] = "0" * 64
    target = tmp_path / "manifest.json"
    from src.lazybull.v2.hosts.testbed.storage import seal_record, write_json_exclusive

    write_json_exclusive(target, seal_record(corrupted))
    with pytest.raises(ManifestError):
        load_manifest(target, verify_calibrator=True)


def test_load_requires_repo_copy_byte_identical(tmp_path):
    document = _document(tmp_path)
    sealed_path = tmp_path / "sealed.json"
    freeze_manifest(document, sealed_path)
    other = tmp_path / "other.json"
    freeze_manifest({**document, "run_id": "another"}, other)
    with pytest.raises(ManifestError):
        load_manifest(sealed_path, repo_manifest=other, verify_calibrator=False)
    loaded, digest = load_manifest(sealed_path, repo_manifest=sealed_path, verify_calibrator=False)
    assert loaded["run_id"] == document["run_id"] and len(digest) == 64


def test_upstream_hash_mismatch_refuses_run(tmp_path):
    document = _document(tmp_path)
    repo = tmp_path / "repo"
    with pytest.raises(ManifestError):
        verify_upstream(
            {
                **document,
                "upstream": {
                    **document["upstream"],
                    "t4_manifest": {**document["upstream"]["t4_manifest"], "sha256": "a" * 64},
                },
            },
            repo_root=repo,
        )


def test_frozen_window_hash_mismatch_refuses_run(tmp_path):
    document = _document(tmp_path)
    repo = tmp_path / "repo"
    verify_upstream(document, repo_root=repo)
    tampered = copy.deepcopy(document)
    tampered["injection_window"]["frozen_list"][0]["window_start_ordinal"] = 5
    with pytest.raises(ManifestError):
        verify_upstream(tampered, repo_root=repo)


def test_verify_chain_detects_three_ring_tamper(tmp_path):
    document = _document(tmp_path)
    identity = calibrator_identity()
    chain = verify_chain(
        manifest_doc=document,
        manifest_file_sha256="b" * 64,
        calibrator=identity,
        artifact_hashes={"rehearsal_report.json": "c" * 64},
    )
    assert chain["manifest_sha256"] == "b" * 64
    assert chain["calibrator_sha256"] == identity["sha256"]
    with pytest.raises(ManifestError):
        verify_chain(
            manifest_doc=document,
            manifest_file_sha256="b" * 64,
            calibrator={**identity, "sha256": "d" * 64},
            artifact_hashes={},
        )
    with pytest.raises(ManifestError):
        verify_chain(
            manifest_doc=document,
            manifest_file_sha256="b" * 64,
            calibrator=identity,
            artifact_hashes={"rehearsal_report.json": "short"},
        )


def test_manifest_records_upstream_t4_t5_hashes_verbatim(tmp_path):
    document = _document(tmp_path)
    repo = tmp_path / "repo"
    upstream = t6env.upstream_payload(repo)
    for key, payload in upstream.items():
        assert document["upstream"][key]["sha256"] == payload["sha256"]
    assert document["injection_window"]["mapping_mode"] == "index_aligned_frozen_sample_index"


def test_build_document_rejects_unknown_frozen_window_anchor(tmp_path):
    env = _env(tmp_path)
    repo = t6env.build_repo(tmp_path)
    document = build_manifest_document(
        run_id="r",
        created_at="2026-11-02",
        campaign_id="t6_campaign",
        campaign_root=str(env.base_root / "t6_campaign"),
        base_root=str(env.base_root),
        data_root=str(env.data_root),
        days=DAYS,
        watermark=DAYS[-1],
        selection_rule="测试",
        frozen_windows=[],
        frozen_list_sha256=t6env.sha256_text("[]"),
        parameters={"dd_improve": None, "dd_worsen": None},
        upstream=t6env.upstream_payload(repo),
    )
    errors = validate_manifest(document)
    assert any("frozen_list" in error for error in errors)
