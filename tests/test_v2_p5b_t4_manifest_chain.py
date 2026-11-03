# -*- coding: utf-8 -*-
"""P5b T4 校准 manifest 与三方哈希互链（设计 v4 §5.5）。

覆盖：§5.5 全表字段、证据类别/行域/时钟口径、look 与 α 分配、门槛不得放宽、
实现数下限（只能上调）、冻结件独占创建、封存与文件哈希校验、校准器身份随代码变化、
manifest ↔ 校准器 ↔ 临界值（↔ holdout/power）互链与任一环篡改即失败。
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

import p5b_t4_env as env  # noqa: E402

from src.lazybull.v2.hosts.testbed.referee import (  # noqa: E402
    CALIBRATOR_MODULES,
    MANIFEST_SCHEMA,
    ManifestError,
    SCALE_FLOOR,
    calibrator_identity,
    freeze_manifest,
    load_manifest,
    manifest_sha256,
    scale_floor_errors,
    validate_manifest,
    verify_calibration_chain,
    verify_source_files,
)


def _doc(tmp_path: Path):
    return env.fast_manifest(tmp_path)


def _write_manifest(tmp_path: Path, doc) -> Path:
    target = Path(tmp_path) / "frozen" / "calibration_manifest.json"
    freeze_manifest(doc, target)
    return target


def test_validate_accepts_generated_document(tmp_path):
    built = _doc(tmp_path)
    assert built.doc["schema"] == MANIFEST_SCHEMA
    assert validate_manifest(built.doc) == []


def test_missing_field_rejected(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["income_referee"]["threshold"].pop("min_valid_quantile_sample")
    errors = validate_manifest(doc)
    assert any("min_valid_quantile_sample" in item for item in errors)


def test_evidence_flags_enforced(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["evidence"]["evidence_class"] = "forward_evidence"
    doc["evidence"]["schedule_mode"] = "live"
    errors = validate_manifest(doc)
    assert any("evidence_class" in item for item in errors)
    assert any("schedule_mode" in item for item in errors)


def test_look_mapping_mismatch_rejected(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["income_referee"]["sample"]["look_days"] = [120, 240, 360, 480]
    errors = validate_manifest(doc)
    assert any("look_days" in item for item in errors)


def test_alpha_allocation_rejected(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["income_referee"]["threshold"]["alpha_per_look"] = 0.02
    errors = validate_manifest(doc)
    assert any("alpha_per_look" in item for item in errors)


def test_gates_cannot_be_loosened(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["income_referee"]["threshold"]["holdout_wilson_upper_max"] = 0.10
    doc["income_referee"]["threshold"]["invalid_rate_per_look_max"] = 0.20
    errors = validate_manifest(doc)
    assert any("holdout_wilson_upper_max" in item for item in errors)
    assert any("invalid_rate_per_look_max" in item for item in errors)


def test_scale_floor_counts_only_go_up(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["counts"]["train_per_unit"] = SCALE_FLOOR["train_per_unit"] - 1
    assert scale_floor_errors(doc["counts"])
    assert validate_manifest(doc) == []  # 结构合法，但下限校验单独失败


def test_drawdown_placeholder_required(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["drawdown_referee"]["status"] = "implemented"
    errors = validate_manifest(doc)
    assert any("drawdown_referee.status" in item for item in errors)


def test_source_folds_must_be_disjoint(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["source_pair"]["holdout_folds"] = list(doc["source_pair"]["train_folds"])
    errors = validate_manifest(doc)
    assert any("重叠" in item for item in errors)


def test_scene_set_must_be_complete(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["scenes"].pop("ar1_06")
    errors = validate_manifest(doc)
    assert any("scenes" in item for item in errors)


def test_freeze_refuses_to_overwrite(tmp_path):
    built = _doc(tmp_path)
    target = _write_manifest(tmp_path, built.doc)
    with pytest.raises(ManifestError):
        freeze_manifest(built.doc, target)


def test_load_rejects_manifest_below_scale_floor(tmp_path):
    built = _doc(tmp_path)
    target = _write_manifest(tmp_path, built.doc)
    with pytest.raises(ManifestError):
        load_manifest(target)  # 小规模 manifest 一律被下限门拒绝
    doc = load_manifest(target, require_scale_floor=False)
    assert doc["counts"]["train_per_unit"] < SCALE_FLOOR["train_per_unit"]


def test_load_rejects_tampered_seal(tmp_path):
    built = _doc(tmp_path)
    target = _write_manifest(tmp_path, built.doc)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["counts"]["seed"] = 1
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ManifestError):
        load_manifest(target, require_scale_floor=False)


def test_load_rejects_manifest_file_sha_mismatch(tmp_path):
    built = _doc(tmp_path)
    target = _write_manifest(tmp_path, built.doc)
    with pytest.raises(ManifestError):
        load_manifest(target, expected_sha256="0" * 64, require_scale_floor=False)


def test_load_detects_calibrator_code_change(tmp_path):
    built = _doc(tmp_path)
    target = _write_manifest(tmp_path, built.doc)
    copy_dir = Path(tmp_path) / "referee_copy"
    copy_dir.mkdir()
    for name in CALIBRATOR_MODULES:
        shutil.copyfile(env.REFEREE_DIR / name, copy_dir / name)
    with (copy_dir / "statistics.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# 代码改动（模拟）\n")
    with pytest.raises(ManifestError):
        load_manifest(target, require_scale_floor=False, module_dir=copy_dir)


def test_calibrator_identity_is_stable_and_covers_modules():
    first = calibrator_identity()
    second = calibrator_identity()
    assert first["sha256"] == second["sha256"]
    assert set(first["modules"]) == set(CALIBRATOR_MODULES)


def test_source_file_hash_mismatch_rejected(tmp_path):
    built = _doc(tmp_path)
    doc = json.loads(json.dumps(built.doc))
    doc["source_pair"]["base_chain_nav"] = built.env.base_path.relative_to(
        built.env.repo_root
    ).as_posix()
    doc["source_pair"]["noise_chain_nav"] = built.env.noise_path.relative_to(
        built.env.repo_root
    ).as_posix()
    verify_source_files(doc, root=built.env.repo_root)
    doc["source_pair"]["base_sha256"] = "0" * 64
    with pytest.raises(ManifestError):
        verify_source_files(doc, root=built.env.repo_root)


def test_hash_chain_ok_and_tamper_detection(tmp_path):
    built = _doc(tmp_path)
    target = _write_manifest(tmp_path, built.doc)
    sha = manifest_sha256(target)
    identity = calibrator_identity()
    thresholds = {"manifest_sha256": sha, "calibrator_sha256": identity["sha256"]}
    holdout = {
        "manifest_sha256": sha,
        "calibrator_sha256": identity["sha256"],
        "thresholds_sha256": "t" * 64,
    }
    power = dict(holdout)
    result = verify_calibration_chain(
        manifest_doc=built.doc,
        manifest_file_sha256=sha,
        calibrator=identity,
        thresholds=thresholds,
        thresholds_sha256="t" * 64,
        holdout=holdout,
        power=power,
    )
    assert result["ok"] is True
    assert len(result["checks"]) >= 6

    # 篡改 holdout 记录的 manifest 哈希 ⇒ 链断
    broken = dict(holdout, manifest_sha256="x" * 64)
    with pytest.raises(ManifestError):
        verify_calibration_chain(
            manifest_doc=built.doc,
            manifest_file_sha256=sha,
            calibrator=identity,
            thresholds=thresholds,
            thresholds_sha256="t" * 64,
            holdout=broken,
        )

    # 篡改临界值记录的 manifest 哈希 ⇒ 链断
    with pytest.raises(ManifestError):
        verify_calibration_chain(
            manifest_doc=built.doc,
            manifest_file_sha256=sha,
            calibrator=identity,
            thresholds={"manifest_sha256": "y" * 64, "calibrator_sha256": identity["sha256"]},
            thresholds_sha256="t" * 64,
        )

    # 校准器身份不符 ⇒ 链断
    with pytest.raises(ManifestError):
        verify_calibration_chain(
            manifest_doc=built.doc,
            manifest_file_sha256=sha,
            calibrator={"sha256": "z" * 64, "implementation_id": "other", "modules": {}},
            thresholds=thresholds,
            thresholds_sha256="t" * 64,
        )

    # holdout 指向别的临界值产物 ⇒ 链断
    with pytest.raises(ManifestError):
        verify_calibration_chain(
            manifest_doc=built.doc,
            manifest_file_sha256=sha,
            calibrator=identity,
            thresholds=thresholds,
            thresholds_sha256="u" * 64,
            holdout=holdout,
        )


def test_chain_requires_thresholds_sha_when_holdout_given(tmp_path):
    built = _doc(tmp_path)
    target = _write_manifest(tmp_path, built.doc)
    sha = manifest_sha256(target)
    identity = calibrator_identity()
    with pytest.raises(ManifestError):
        verify_calibration_chain(
            manifest_doc=built.doc,
            manifest_file_sha256=sha,
            calibrator=identity,
            thresholds={"manifest_sha256": sha, "calibrator_sha256": identity["sha256"]},
            holdout={"manifest_sha256": sha, "calibrator_sha256": identity["sha256"]},
        )


# --------------------------------------------------------------------------
# 首轮评审接纳项（reviewer1 R3/R7、reviewer2 R2-T4-03/04/11）
# --------------------------------------------------------------------------


def test_deviation_registry_entries_need_full_schema(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["deviation_registry"][0].pop("reason")
    errors = validate_manifest(doc)
    assert any("deviation_registry[0].reason" in item for item in errors)


def test_deviation_registry_requires_mandatory_items(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["deviation_registry"] = [
        entry for entry in doc["deviation_registry"] if entry["item"] != "B0' 臂口径"
    ]
    errors = validate_manifest(doc)
    assert any("缺少必登记项" in item and "B0' 臂口径" in item for item in errors)


def test_manifest_requires_noise_arm_note(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    doc["source_pair"].pop("noise_arm_note")
    assert any("noise_arm_note" in item for item in validate_manifest(doc))
    doc["source_pair"]["noise_arm_note"] = "B0' 是同策略换种子臂"
    assert any("500 树" in item for item in validate_manifest(doc))


def test_calibrator_identity_covers_orchestration_module():
    """calibration.py 决定 counts/seed/prefix 取用，必须纳入校准器身份。"""
    assert "calibration.py" in CALIBRATOR_MODULES
    modules = calibrator_identity()["modules"]
    assert set(modules) == set(CALIBRATOR_MODULES)
    assert len(modules["calibration.py"]) == 64


def test_chain_records_thresholds_manifest_link_exactly_once(tmp_path):
    built = _doc(tmp_path)
    target = _write_manifest(tmp_path, built.doc)
    sha = manifest_sha256(target)
    identity = calibrator_identity()
    thresholds = {"manifest_sha256": sha, "calibrator_sha256": identity["sha256"]}
    checks = verify_calibration_chain(
        manifest_doc=built.doc,
        manifest_file_sha256=sha,
        calibrator=identity,
        thresholds=thresholds,
        thresholds_sha256="0" * 64,
    )["checks"]
    links = [item["link"] for item in checks]
    assert links.count("thresholds:manifest") == 1
    assert links.count("thresholds:calibrator") == 1


def test_lineage_first_freeze_has_no_predecessor(tmp_path):
    doc = json.loads(json.dumps(_doc(tmp_path).doc))
    assert doc["lineage"]["sequence"] == 1
    assert doc["lineage"]["superseded_manifest_sha256"] is None
    assert validate_manifest(doc) == []
    doc["lineage"]["superseded_manifest_sha256"] = "a" * 64
    assert any("首次冻结" in item for item in validate_manifest(doc))


def test_lineage_refreeze_requires_supersede_record(tmp_path):
    """重冻必须留废止记录，否则无法区分"先看读数再改代码再重冻"。"""
    built = _doc(tmp_path)
    doc = json.loads(json.dumps(built.doc))
    doc["lineage"] = {
        "sequence": 2,
        "superseded_manifest_sha256": None,
        "superseded_calibrator_sha256": None,
        "supersede_reason": None,
    }
    errors = validate_manifest(doc)
    assert any("superseded_manifest_sha256" in item for item in errors)
    assert any("supersede_reason" in item for item in errors)
    doc["lineage"].update(
        {
            "superseded_manifest_sha256": "b" * 64,
            "superseded_calibrator_sha256": "c" * 64,
            "supersede_reason": "复审整改后重新冻结（新增谱系段与身份覆盖）",
        }
    )
    assert validate_manifest(doc) == []


def test_calibrator_identity_covers_manifest_and_serialization_modules():
    """冻结文字与序列化口径必须进身份；呈现层排除必须显式登记理由。"""
    from src.lazybull.v2.hosts.testbed.referee import CALIBRATOR_EXCLUSIONS

    for name in ("manifest.py", "manifest_sections.py", "artifacts.py"):
        assert name in CALIBRATOR_MODULES, name
    assert "report.py" in CALIBRATOR_EXCLUSIONS
    assert CALIBRATOR_EXCLUSIONS["report.py"].strip()


def test_referee_package_all_exports_exist():
    """__all__ 不得悬空（清理死代码后导出面必须同步）。"""
    import src.lazybull.v2.hosts.testbed.referee as package

    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert missing == []


def test_load_rejects_repo_manifest_byte_mismatch(tmp_path):
    from src.lazybull.v2.hosts.testbed.referee import load_calibration_manifest

    built = env.fast_manifest(tmp_path)
    campaign = Path(tmp_path) / "campaigns" / "p5b_t4_repo_check"
    (campaign / "reports").mkdir(parents=True, exist_ok=True)
    target = campaign / "reports" / "calibration_manifest.json"
    freeze_manifest(built.doc, target)
    repo_copy = Path(tmp_path) / "repo" / "docs" / "plans" / "v2" / "frozen.json"
    repo_copy.parent.mkdir(parents=True, exist_ok=True)
    repo_copy.write_bytes(target.read_bytes())
    kwargs = {"repo_manifest": repo_copy, "require_scale_floor": False}
    doc, sha = load_calibration_manifest(campaign, **kwargs)
    assert sha == manifest_sha256(target)
    repo_copy.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ManifestError):
        load_calibration_manifest(campaign, **kwargs)
    kwargs["repo_manifest"] = Path(tmp_path) / "missing.json"
    with pytest.raises(ManifestError):
        load_calibration_manifest(campaign, **kwargs)
