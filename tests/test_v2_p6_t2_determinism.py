# -*- coding: utf-8 -*-
"""P6 T2 确定性、历史保留与原子失败（确定性/历史/原子失败 3 项，设计 §8.2 / §9 / §10）。

- 同输入且同 ``generated_at`` 的产出**逐字节一致**；不同 ``generated_at`` 即新 run；
- 旧 run 目录**不被改动、不被覆盖**（同 report_id 再发布必须失败）；
- 生成失败**不留半成品**：目标 run 目录不存在、本次 staging 已清理。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import publish, render, validation  # noqa: E402
from src.lazybull.v2.reporting.generate import (  # noqa: E402
    DashboardRequest,
    generate_dashboard,
)
from tests.p6_t2_env import case_paper_root  # noqa: E402

GENERATED_AT = "20261106T040000Z"


def _request(
    paper: Path,
    quality: Path,
    reports_root: Path,
    *,
    attempt: str,
    generated_at: str = GENERATED_AT,
) -> DashboardRequest:
    return DashboardRequest(
        paper_root=paper,
        quality_root=quality,
        reports_root=reports_root,
        evidence_class="research",
        generated_at=generated_at,
        attempt_id=attempt,
        input_evidence_classes=("production",),
        git_record={"git_commit": "<synthetic>", "git_dirty": False},
    )


def _tree_bytes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): validation.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_p6_t2_11_determinism_same_input_same_generated_at(tmp_path: Path) -> None:
    """同输入且同 generated_at 逐字节一致；换 generated_at 即新 run（字节不同）。"""
    paper, quality = case_paper_root(tmp_path, "standard")
    first = generate_dashboard(_request(paper, quality, tmp_path / "a", attempt="det_a"))
    second = generate_dashboard(_request(paper, quality, tmp_path / "b", attempt="det_b"))
    assert first["files"] == second["files"]
    assert first["report_id"] == second["report_id"]
    assert first["manifest"]["record_sha256"] == second["manifest"]["record_sha256"]
    assert first["outputs"] == second["outputs"]

    later = generate_dashboard(
        _request(
            paper,
            quality,
            tmp_path / "c",
            attempt="det_c",
            generated_at="20261106T050000Z",
        )
    )
    assert later["report_id"] != first["report_id"]
    assert later["files"]["report.html"] != first["files"]["report.html"]
    # projection 里 generated_at 是显式输入 ⇒ 新 run 的 projection 字节随之改变
    assert later["files"]["projection.json"] != first["files"]["projection.json"]

    # 视图层同样确定性：同视图两次渲染字节一致（不依赖进程内缓存）
    html_a = render.render_dashboard_html(first["view"])
    html_b = render.render_dashboard_html(first["view"])
    assert html_a == html_b


def test_p6_t2_12_history_not_overwritten(tmp_path: Path) -> None:
    """同 report_id 再发布必须拒绝；旧 run 字节与清单完全不变。"""
    paper, quality = case_paper_root(tmp_path, "standard")
    reports = tmp_path / "reports"
    first = generate_dashboard(_request(paper, quality, reports, attempt="hist_1"))
    target = first["report_path"]
    assert target is not None and target.is_dir()
    before = _tree_bytes(target)
    assert "report.html" in before

    with pytest.raises(publish.PublishError):
        generate_dashboard(_request(paper, quality, reports, attempt="hist_2"))
    assert _tree_bytes(target) == before

    runs = sorted(path.name for path in (reports / "research" / "dashboard").iterdir())
    assert runs == [first["report_id"]]
    staging = reports / ".staging"
    assert not staging.exists() or not any(staging.iterdir())

    # 新 generated_at ⇒ 新目录；旧 run 仍逐字节不变
    newer = generate_dashboard(
        _request(paper, quality, reports, attempt="hist_3", generated_at="20261106T060000Z")
    )
    assert newer["report_path"] != target
    assert _tree_bytes(target) == before
    assert len(list((reports / "research" / "dashboard").iterdir())) == 2


def test_p6_t2_13_atomic_failure_leaves_no_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """渲染失败 ⇒ 目标 run 目录不存在、staging 已清理（不留半成品）。"""
    paper, quality = case_paper_root(tmp_path, "standard")
    reports = tmp_path / "reports"

    def _boom(*args: Any, **kwargs: Any) -> bytes:
        raise RuntimeError("合成渲染失败")

    monkeypatch.setattr(render, "render_dashboard_bytes", _boom)
    with pytest.raises(RuntimeError, match="合成渲染失败"):
        generate_dashboard(_request(paper, quality, reports, attempt="atomic_fail"))
    assert not (reports / "research").exists()
    assert not (reports / ".staging" / "atomic_fail").exists()
    monkeypatch.undo()

    # 发布阶段冲突：目标 run 目录已存在 ⇒ 拒绝覆盖且不改写既有字节
    published = generate_dashboard(_request(paper, quality, reports, attempt="atomic_ok"))
    target = published["report_path"]
    assert target is not None
    before = _tree_bytes(target)
    with pytest.raises(publish.PublishError):
        publish.publish_new_run(
            reports_root=reports,
            evidence_class="research",
            report_kind="dashboard",
            report_id=published["report_id"],
            files=published["files"],
            attempt_id="atomic_conflict",
        )
    assert _tree_bytes(target) == before
    staging = reports / ".staging"
    assert not staging.exists() or not any(staging.iterdir())

    # 文件集校验：未登记文件名/缺 report.html 一律拒绝（不写盘）
    with pytest.raises(publish.PublishError):
        publish.publish_new_run(
            reports_root=reports,
            evidence_class="research",
            report_kind="dashboard",
            report_id="p6_dashboard_20261106T040000Z_0123456789ab",
            files={"report.html": b"<html></html>", "extra.csv": b"x"},
            attempt_id="atomic_files",
        )
    with pytest.raises(publish.PublishError):
        publish.publish_new_run(
            reports_root=reports,
            evidence_class="research",
            report_kind="dashboard",
            report_id="p6_dashboard_20261106T040000Z_0123456789ab",
            files={"projection.json": b"{}"},
            attempt_id="atomic_files2",
        )


def test_p6_t2_23_snapshot_retry_assembly_and_staging_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2 生成链的重试装配：首次漂移 → 换 attempt 重试成功；失败路径不留 staging。"""
    from src.lazybull.v2.reporting import snapshot as snapshot_module

    paper, quality = case_paper_root(tmp_path, "standard")
    reports = tmp_path / "reports"
    calls = {"count": 0}
    original = snapshot_module.StableSnapshot.verify_live

    def _flaky(self: Any) -> list[str]:
        calls["count"] += 1
        if calls["count"] == 1:
            return ["合成：首次读取期漂移"]
        return original(self)

    monkeypatch.setattr(snapshot_module.StableSnapshot, "verify_live", _flaky)
    result = generate_dashboard(_request(paper, quality, reports, attempt="retry_probe"))
    assert calls["count"] >= 2
    attempts = [item["attempt_id"] for item in result["snapshot_audit"]["attempts"]]
    assert attempts == ["r1", "r2"]
    assert result["snapshot_audit"]["retries"] == 1
    assert (reports / "research" / "dashboard" / result["report_id"]).is_dir()
    assert not (reports / ".staging" / "retry_probe").exists()

    # 漂移耗尽：重试用完 → 抛错且本次 staging 父目录被清理（不留空壳）
    monkeypatch.setattr(
        snapshot_module.StableSnapshot, "verify_live", lambda self: ["合成：持续漂移"]
    )
    exhausted = tmp_path / "reports_exhausted"
    with pytest.raises(snapshot_module.UnstableSnapshotError):
        generate_dashboard(_request(paper, quality, exhausted, attempt="retry_fail"))
    assert not (exhausted / ".staging" / "retry_fail").exists()


def test_p6_t2_28_cleanup_field_records_discard_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cleanup.staging_discard_ok`` 必须能反映清理失败（不是恒 True 的自述字段）。"""
    paper, quality = case_paper_root(tmp_path, "standard")

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("合成：清理失败")

    monkeypatch.setattr(publish, "discard_attempt", _boom)
    result = generate_dashboard(_request(paper, quality, tmp_path / "r1", attempt="cleanup_fail"))
    assert result["cleanup"]["staging_discard_ok"] is False
    assert result["cleanup"]["attempt_id"] == "cleanup_fail"

    monkeypatch.undo()
    ok = generate_dashboard(_request(paper, quality, tmp_path / "r2", attempt="cleanup_ok"))
    assert ok["cleanup"]["staging_discard_ok"] is True
