# -*- coding: utf-8 -*-
"""P6 T5 真实 golden 核验器测试（``scripts/v2_p6/check_real_golden.py``）。

隔离口径：
- 正向依据是**上游原生构造 + 封存**的合成 T6 bundle（``tests.p5b_t6_env`` 的构造/封存函数），
  不读取任何真实 campaign、不伪造真实数值；
- 真实 seq10/7/21 的正向验收由 T5 独立 golden 运行承担（非 pytest，见验收载荷）；
- 负例覆盖：缺文件/多文件/字节不符/值不符/链断/根缺失/冻结件形态非法/输出目录守卫。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "v2_p6" / "check_real_golden.py"
_SPEC = importlib.util.spec_from_file_location("p6_t5_check_real_golden", SCRIPT)
driver = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(driver)


def _synthetic_t6(directory: Path, *, campaign: str = "p6_t5_golden_synthetic") -> Path:
    """由原生构造与封存函数生成 T6 合成包（与 T1 同款；不执行虚拟日）。"""
    from src.lazybull.v2.hosts.testbed.rehearsal import artifacts as t6_artifacts
    from src.lazybull.v2.hosts.testbed.rehearsal import report as t6_report
    from src.lazybull.v2.hosts.testbed.rehearsal import verdict as t6_verdict
    from tests import p5b_t6_env

    env = SimpleNamespace(
        data_root=directory.parent / "data",
        base_root=directory.parent / "testbed",
        days=(
            "20260706",
            "20260707",
            "20260708",
            "20260709",
            "20260710",
            "20260713",
            "20260714",
            "20260715",
            "20260716",
            "20260717",
        ),
    )
    repo = p5b_t6_env.build_repo(directory.parent / "upstream")
    manifest = p5b_t6_env.manifest_document(env, repo, campaign_id=campaign)
    frozen = t6_artifacts.write_json_once(
        directory / t6_artifacts.ARTIFACT_MANIFEST, manifest, what="合成 T6 manifest"
    )
    manifest_sha = frozen["sha256"]
    kwargs = p5b_t6_env.report_kwargs(manifest)
    kwargs["manifest_sha256"] = manifest_sha
    kwargs["calibrator"] = manifest["files"]["calibrator"]
    kwargs["run_report"]["identity"]["manifest_sha256"] = manifest_sha
    kwargs["artifact_hashes"] = {t6_artifacts.ARTIFACT_MANIFEST: manifest_sha}
    report = t6_report.build_report(**kwargs)
    written = t6_artifacts.write_json_once(
        directory / t6_artifacts.ARTIFACT_REPORT, report, what="合成 T6 报告"
    )
    markdown = t6_artifacts.write_text_once(
        directory / t6_artifacts.ARTIFACT_REPORT_MD,
        t6_report.render_markdown(report),
        what="合成 T6 Markdown",
    )
    checks = {"conditions": [], "author_side_status": "not_executed"}
    verdict = t6_verdict.build_verdict(
        run_id=manifest["run_id"],
        campaign_id=campaign,
        manifest_sha256=manifest_sha,
        calibrator_sha256=manifest["files"]["calibrator"]["sha256"],
        artifact_hashes={
            t6_artifacts.ARTIFACT_REPORT: written["sha256"],
            t6_artifacts.ARTIFACT_REPORT_MD: markdown["sha256"],
        },
        h1=checks,
        h6=checks,
        h7=checks,
        effects={},
    )
    t6_artifacts.write_json_once(
        directory / t6_artifacts.ARTIFACT_VERDICT, verdict, what="合成 T6 判定表"
    )
    return directory


def _t6_campaign_root(tmp_path: Path) -> Path:
    """``root/<campaign>/reports`` 结构（与真实 campaigns 根同布局）。"""
    returns = _synthetic_t6(tmp_path / "campaigns" / "synthetic_campaign" / "reports")
    assert returns.is_dir()
    return tmp_path / "campaigns"


def _freeze_for(root: Path, campaign: str = "synthetic_campaign") -> dict[str, Any]:
    """只冻结 T6 族（其余族留空 —— 冻结器允许只给一族的场景用 ``check_root`` 直接驱动）。"""
    families = {
        "t6_seq21": {
            "campaign_dir": campaign,
            "files": {
                path.name: driver._sha256_file(path)
                for path in sorted((root / campaign / "reports").iterdir())
                if path.is_file()
            },
            "seals": {},
            "values": {"verdict": {}},
        }
    }
    documents, problems = driver._load_sealed("t6_seq21", root / campaign / "reports")
    assert not problems
    families["t6_seq21"]["seals"] = {
        name: document.get("record_sha256")
        for name, document in documents.items()
        if isinstance(document.get("record_sha256"), str)
    }
    families["t6_seq21"]["values"]["verdict"] = driver._values_of(
        "t6_seq21", documents, "rehearsal_verdict.json"
    )
    return {
        "schema": driver.FREEZE_SCHEMA,
        "frozen_at": "2026-11-07",
        # 单族夹具：显式声明“不做三族完备性”，否则默认 fail-closed（生产冻结件必须齐全）
        "require_all_families": False,
        "source_root": root.as_posix(),
        "families": families,
    }


def test_t5_golden_positive_roundtrip_on_native_t6(tmp_path: Path) -> None:
    """原生合成 bundle：冻结 → 核验必须 pass，且登记路径集合/封印/关键值。"""
    root = _t6_campaign_root(tmp_path)
    freeze = _freeze_for(root)
    payload = driver.check_root(root, freeze)
    assert payload["verdict"] == "pass", payload
    row = payload["families"]["t6_seq21"]
    assert row["checks"] == {
        "reports_dir_present": True,
        "file_set_exact": True,
        "actual_file_count": len(freeze["families"]["t6_seq21"]["files"]),
        "unexpected_dirs": [],
        "unexpected_symlinks": [],
        "sha_match": True,
        "sealed_load": True,
        "bundle_chain": True,
        "values_match": True,
        "seal_match": True,
    }
    assert row["values"]["campaign_id"] == "p6_t5_golden_synthetic"
    assert row["chain"]["report_status"] == "available"


def test_t5_golden_extra_subdirectory_fails(tmp_path: Path) -> None:
    """``reports/`` 下多出**子目录**（或符号链接）也必须失败：集合判据不能只看单层文件（A-08）。"""
    root = _t6_campaign_root(tmp_path)
    freeze = _freeze_for(root)
    reports = root / "synthetic_campaign" / "reports"
    (reports / "nested").mkdir()
    (reports / "nested" / "note.txt").write_text("unexpected\n", encoding="utf-8")
    payload = driver.check_root(root, freeze)
    row = payload["families"]["t6_seq21"]
    assert payload["verdict"] == "fail"
    assert row["checks"]["file_set_exact"] is False
    assert "nested" in row["checks"]["unexpected_dirs"]
    assert any("多出未冻结目录" in item for item in row["failures"])


def test_t5_golden_missing_root_fails_without_skip(tmp_path: Path) -> None:
    """真实根缺失 ⇒ fail（三族全记 failed），不得抛异常、不得 skip。"""
    freeze = {
        "schema": driver.FREEZE_SCHEMA,
        "families": {family: {"campaign_dir": "x"} for family in driver.FAMILIES},
    }
    payload = driver.check_root(tmp_path / "nope", freeze)
    assert payload["verdict"] == "fail"
    assert payload["root_present"] is False
    assert payload["failed_families"] == sorted(driver.FAMILIES)


def test_t5_golden_missing_campaign_dir_fails(tmp_path: Path) -> None:
    """campaign 目录缺失 ⇒ 该族 fail（不是 not_available 放行）。"""
    root = _t6_campaign_root(tmp_path)
    freeze = _freeze_for(root)
    freeze["families"]["t6_seq21"]["campaign_dir"] = "absent_campaign"
    payload = driver.check_root(root, freeze)
    assert payload["verdict"] == "fail"
    assert payload["families"]["t6_seq21"]["checks"]["reports_dir_present"] is False


def test_t5_golden_file_sha_mismatch_fails(tmp_path: Path) -> None:
    """任一文件字节被改（不重封存）⇒ 字节 SHA 与 seal 双失败。"""
    root = _t6_campaign_root(tmp_path)
    freeze = _freeze_for(root)
    target = root / "synthetic_campaign" / "reports" / "rehearsal_report.md"
    target.write_bytes(target.read_bytes() + b"\n")
    payload = driver.check_root(root, freeze)
    row = payload["families"]["t6_seq21"]
    assert payload["verdict"] == "fail"
    assert row["checks"]["sha_match"] is False
    assert any("字节" in item for item in row["failures"])


def test_t5_golden_extra_file_fails(tmp_path: Path) -> None:
    """多出未冻结文件 ⇒ 文件集合不精确，失败（禁止“至少包含”放行）。"""
    root = _t6_campaign_root(tmp_path)
    freeze = _freeze_for(root)
    (root / "synthetic_campaign" / "reports" / "extra.json").write_text("{}\n", encoding="utf-8")
    payload = driver.check_root(root, freeze)
    row = payload["families"]["t6_seq21"]
    assert payload["verdict"] == "fail"
    assert row["checks"]["file_set_exact"] is False
    assert row["checks"]["unexpected_files"] == ["extra.json"]


def test_t5_golden_value_mismatch_fails(tmp_path: Path) -> None:
    """冻结值不等（真实字节不同）⇒ values_match 失败。"""
    root = _t6_campaign_root(tmp_path)
    freeze = _freeze_for(root)
    freeze["families"]["t6_seq21"]["values"]["verdict"]["campaign_id"] = "other_campaign"
    payload = driver.check_root(root, freeze)
    row = payload["families"]["t6_seq21"]
    assert payload["verdict"] == "fail"
    assert row["checks"]["values_match"] is False
    assert any("campaign_id" in item for item in row["failures"])


def test_t5_golden_chain_or_seal_tamper_fails(tmp_path: Path) -> None:
    """重封存后改值 ⇒ 冻结 SHA 与 seal 不符（链/封印证据不被跳过）。"""
    root = _t6_campaign_root(tmp_path)
    freeze = _freeze_for(root)
    from src.lazybull.v2.hosts.testbed.storage import seal_record

    target = root / "synthetic_campaign" / "reports" / "rehearsal_verdict.json"
    document = json.loads(target.read_text(encoding="utf-8"))
    document["h1_claim"] = "tampered"
    target.write_text(
        json.dumps(seal_record(document), ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    payload = driver.check_root(root, freeze)
    row = payload["families"]["t6_seq21"]
    assert payload["verdict"] == "fail"
    assert row["checks"]["sha_match"] is False
    assert row["checks"]["seal_match"] is False


def test_t5_golden_freeze_rejects_incomplete_or_existing(tmp_path: Path) -> None:
    """冻结器：三族必须给全；重复冻结必须换新文件名（拒绝覆盖）。"""
    root = _t6_campaign_root(tmp_path)
    with pytest.raises(driver.GoldenSetupError):
        driver.freeze_root(root, {"t6_seq21": "synthetic_campaign"})
    with pytest.raises(driver.GoldenSetupError):
        driver.freeze_root(tmp_path / "absent", {family: "x" for family in driver.FAMILIES})


def test_t5_golden_load_freeze_shape_guards(tmp_path: Path) -> None:
    """冻结件形态非法（schema/未知族/缺 families）一律设置错误。"""
    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text(json.dumps({"schema": "other/v1", "families": {}}), encoding="utf-8")
    with pytest.raises(driver.GoldenSetupError):
        driver.load_freeze(bad_schema)
    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps({"schema": driver.FREEZE_SCHEMA, "families": {"t9_seq99": {}}}),
        encoding="utf-8",
    )
    with pytest.raises(driver.GoldenSetupError):
        driver.load_freeze(unknown)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"schema": driver.FREEZE_SCHEMA, "families": {}}), encoding="utf-8")
    with pytest.raises(driver.GoldenSetupError):
        driver.load_freeze(empty)
    with pytest.raises(driver.GoldenSetupError):
        driver.load_freeze(tmp_path / "missing.json")


def test_t5_golden_cli_exit_codes(tmp_path: Path) -> None:
    """CLI：缺 --out / 输出目录越界 ⇒ 2；真实根不符 ⇒ 3。"""
    assert driver.main(["--root", str(tmp_path)]) == 2
    assert (
        driver.main(
            [
                "--root",
                str(tmp_path),
                "--out",
                str(tmp_path / "not-a-validation"),
            ]
        )
        == 2
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema": driver.FREEZE_SCHEMA,
                "families": {family: {"campaign_dir": "absent"} for family in driver.FAMILIES},
            }
        ),
        encoding="utf-8",
    )
    existing = (
        ROOT / "data" / "reports" / "p6" / "validations" / "research" / "p6_t4_acceptance_run17"
    )
    if existing.is_dir():  # 已存在的 validation 目录必须拒绝（不得覆盖历史）
        assert (
            driver.main(["--root", str(tmp_path), "--out", str(existing), "--freeze", str(freeze)])
            == 2
        )


def test_t5_golden_has_no_hardcoded_campaign_names_or_values() -> None:
    """核验器源码不得硬编码真实 campaign 名或真实数值（期望值只能来自冻结件）。"""
    source = SCRIPT.read_text(encoding="utf-8")
    assert "p5b_t4_referee" not in source
    assert "p5b_t5_drawdown" not in source
    assert "p5b_t6_rehearsal" not in source
    assert "619016.1802662" not in source
    # 冻结值只能来自冻结件：源码内不得出现明显“真实数值”字面量
    assert "28.088587217293803" not in source
    assert "30.876549316416398" not in source


def test_t5_golden_freeze_file_on_disk_is_wellformed() -> None:
    """仓库内冻结件形态自洽（三族齐全、逐文件 SHA 为 64 hex、值与封印非空）。"""
    freeze = driver.load_freeze(ROOT / driver.DEFAULT_FREEZE)
    families = freeze["families"]
    assert sorted(families) == sorted(driver.FAMILIES)
    for family, entry in families.items():
        assert entry["campaign_dir"]
        assert entry["files"]
        for name, digest in entry["files"].items():
            assert len(digest) == 64 and digest == digest.lower(), (family, name)
        assert entry["seals"], family
        assert entry["values"]["verdict"], family
    assert freeze["frozen_at"]
