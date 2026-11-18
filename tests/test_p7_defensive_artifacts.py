"""DEF-ML-60-v1 工件层测试：拒绝覆盖、写域守卫、封存（临时隔离根）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.v2_p7.common import REPO_ROOT, WriteScopeError
from scripts.v2_p7.defensive_ml import artifacts


def test_campaign_root_rejects_illegal_ids(tmp_path: Path) -> None:
    for bad in ("", "..", "a/b", "a\\b"):
        with pytest.raises(artifacts.ArtifactError):
            artifacts.campaign_root(bad, tmp_path)


def test_create_run_dir_refuses_existing(tmp_path: Path) -> None:
    root = tmp_path / "camp"
    first = artifacts.create_run_dir(root, "readiness_20260101T000000", tmp_path)
    assert first.exists()
    with pytest.raises(artifacts.ArtifactError):
        artifacts.create_run_dir(root, "readiness_20260101T000000", tmp_path)


def test_write_json_checked_is_idempotent_and_refuses_conflict(tmp_path: Path) -> None:
    target = tmp_path / "spec.json"
    first = artifacts.write_json_checked(target, {"a": 1}, tmp_path)
    second = artifacts.write_json_checked(target, {"a": 1}, tmp_path)
    assert first["written"] == "created"
    assert second["written"] == "unchanged"
    assert first["sha256"] == second["sha256"]
    with pytest.raises(artifacts.ArtifactError):
        artifacts.write_json_checked(target, {"a": 2}, tmp_path)


def test_write_outside_allowed_roots_is_rejected() -> None:
    target = REPO_ROOT / "data" / "models" / "def_ml_should_not_exist.json"
    with pytest.raises(WriteScopeError):
        artifacts.write_json_checked(target, {"a": 1})
    assert not target.exists()


def test_copy_verbatim_refuses_different_bytes(tmp_path: Path) -> None:
    source = tmp_path / "protocol.md"
    source.write_text("v1\n", encoding="utf-8")
    copied = artifacts.copy_verbatim(source, tmp_path / "camp" / "protocol.md", tmp_path)
    assert copied["written"] == "created"
    again = artifacts.copy_verbatim(source, tmp_path / "camp" / "protocol.md", tmp_path)
    assert again["written"] == "unchanged"
    source.write_text("v2\n", encoding="utf-8")
    with pytest.raises(artifacts.ArtifactError):
        artifacts.copy_verbatim(source, tmp_path / "camp" / "protocol.md", tmp_path)


def test_seal_stage_lists_files_but_not_manifest(tmp_path: Path) -> None:
    stage = tmp_path / "runs" / "readiness_x"
    stage.mkdir(parents=True)
    artifacts.write_json_checked(
        stage / "readiness.json", {"status": "ready_with_limits"}, tmp_path
    )
    seal = artifacts.seal_stage(
        stage, "readiness", {"readiness_sha256": "abc"}, artifact_base=tmp_path
    )
    assert set(seal["files"]) == {"readiness.json"}
    assert seal["payload"]["readiness_sha256"] == "abc"
    assert (stage / "manifest.json").exists()


def test_assert_under_rejects_root_itself(tmp_path: Path) -> None:
    with pytest.raises(artifacts.ArtifactError):
        artifacts.assert_under(tmp_path, tmp_path)
    artifacts.assert_under(tmp_path / "child", tmp_path)


def test_ensure_layout_creates_fixed_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "camp"
    layout = artifacts.ensure_layout(root, tmp_path)
    assert set(layout) == set(artifacts.SUBDIRS)
    assert all(Path(path).is_dir() for path in layout.values())
