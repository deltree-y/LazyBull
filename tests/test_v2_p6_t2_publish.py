# -*- coding: utf-8 -*-
"""P6 T2 原子发布与写审计组合的负例（T2-R1-10 / R2-03 / R2-04 / R2-06 / R2-08）。

覆盖 T2 新引入的组合（有界子树快照 + 仓库顶层文件集合 + ``extra_roots`` + 重解析点 +
扫描域盲区），全部用合成 repo 与 ``tmp_path``，不触碰真实数据根：

1. 白名单内**多写一个**文件 ⇒ ``unexpected_inside`` 非空、``verdict=fail``；
2. **改写既存** run 文件 ⇒ ``overwrote_existing`` 非空、``verdict=fail``；
3. 绑定前缀目录**在 before 已存在**（含空目录）⇒ “新目录”硬判据失败；
4. ``extra_roots``（repo 之外的输入根）任何变化 ⇒ ``foreign_root_paths`` 非空；
5. 快照携带**重解析点** ⇒ ``reparse_paths`` 非空、``verdict=fail``；
6. 顶层/``data/*`` 出现**未登记**目录 ⇒ ``uncovered`` 非空、``verdict=fail``；
7. ``attempt_id`` 必须是 T1 同一口径的字面量（拒 ``..``/大写），``discard_attempt`` 只删自己的目录。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import publish  # noqa: E402

REPORT_ID = "p6_dashboard_20261106T090000Z_0123456789ab"
EVIDENCE = "research"
KIND = "dashboard"
ATTEMPT = "audit_probe"
RELATIVE = f"data/reports/p6/{EVIDENCE}/{KIND}/{REPORT_ID}"


def _fake_repo(tmp_path: Path) -> Path:
    """最小合成 repo：只保留被登记覆盖的子树，``uncovered_paths`` 必须为空。"""
    repo = tmp_path / "repo"
    for relative in ("data/reports/p6", "src", "scripts"):
        (repo / relative).mkdir(parents=True, exist_ok=True)
    (repo / "README.md").write_text("# synthetic\n", encoding="utf-8")
    return repo


def _run_files(payload: bytes = b"<html>ok</html>\n") -> dict[str, bytes]:
    return {
        "report.html": payload,
        "projection.json": b"{}\n",
        "report_manifest.json": b'{"schema_version": "x"}\n',
    }


def _publish(repo: Path) -> Path:
    return publish.publish_new_run(
        reports_root=repo / "data/reports/p6",
        evidence_class=EVIDENCE,
        report_kind=KIND,
        report_id=REPORT_ID,
        files=_run_files(),
        attempt_id=ATTEMPT,
    )


def _expected() -> list[str]:
    return [f"{RELATIVE}/{name}" for name in publish.RUN_FILENAMES]


def test_p6_t2_17_audit_rejects_extra_file_and_rewrite(tmp_path: Path) -> None:
    """多写一个文件 / 改写既存文件 都必须 fail（含精确诊断字段）。"""
    repo = _fake_repo(tmp_path)
    assert publish.uncovered_paths(repo) == []
    before = publish.scan_audit_roots(repo_root=repo)
    target = _publish(repo)
    after = publish.scan_audit_roots(repo_root=repo)
    report = publish.audit_generate(
        before=before,
        after=after,
        expected_paths=_expected(),
        bindings={
            "attempt": ATTEMPT,
            "evidence_class": EVIDENCE,
            "kind": KIND,
            "report_id": REPORT_ID,
        },
        output_root="worktree",
        repo_root=repo,
    )
    assert report["verdict"] == "pass"
    assert report["scan_scope"]["uncovered"] == []
    assert report["reparse_paths"] == []

    (target / "extra.txt").write_text("unexpected\n", encoding="utf-8")
    extra = publish.audit_generate(
        before=before,
        after=publish.scan_audit_roots(repo_root=repo),
        expected_paths=_expected(),
        bindings={
            "attempt": ATTEMPT,
            "evidence_class": EVIDENCE,
            "kind": KIND,
            "report_id": REPORT_ID,
        },
        output_root="worktree",
        repo_root=repo,
    )
    assert extra["verdict"] == "fail"
    assert extra["unexpected_inside"] == [f"{RELATIVE}/extra.txt"]

    # 改写既存 run 文件：篡改前先删掉多余文件，回到“仅改写”形态
    (target / "extra.txt").unlink()
    rewritten_before = publish.scan_audit_roots(repo_root=repo)
    (target / "report.html").write_bytes(b"<html>tampered</html>\n")
    rewritten = publish.audit_generate(
        before=rewritten_before,
        after=publish.scan_audit_roots(repo_root=repo),
        expected_paths=_expected(),
        bindings={
            "attempt": ATTEMPT,
            "evidence_class": EVIDENCE,
            "kind": KIND,
            "report_id": REPORT_ID,
        },
        output_root="worktree",
        repo_root=repo,
    )
    assert rewritten["verdict"] == "fail"
    assert rewritten["overwrote_existing"]


def test_p6_t2_18_audit_rejects_pre_existing_dir_foreign_root_and_reparse(
    tmp_path: Path,
) -> None:
    """已存在目录 / 外来根变化 / 重解析点 三类都必须 fail。"""
    repo = _fake_repo(tmp_path)
    existing = repo / RELATIVE
    existing.mkdir(parents=True)  # 空目录也算“已存在”（禁止复用旧目录追加）
    before = publish.scan_audit_roots(repo_root=repo)
    for name, payload in _run_files().items():
        (existing / name).write_bytes(payload)
    pre_existing = publish.audit_generate(
        before=before,
        after=publish.scan_audit_roots(repo_root=repo),
        expected_paths=_expected(),
        bindings={
            "attempt": ATTEMPT,
            "evidence_class": EVIDENCE,
            "kind": KIND,
            "report_id": REPORT_ID,
        },
        output_root="worktree",
        repo_root=repo,
    )
    assert pre_existing["verdict"] == "fail"
    assert pre_existing["overwrote_existing"]

    # 外来根（repo 之外的输入根）发生任何变化都不允许
    extra_root = tmp_path / "paper_input"
    extra_root.mkdir()
    (extra_root / "state.json").write_text("{}", encoding="utf-8")
    foreign_before = publish.scan_audit_roots(
        repo_root=repo, extra_roots={"paper_input": extra_root}
    )
    (extra_root / "state.json").write_text('{"changed": true}', encoding="utf-8")
    foreign = publish.audit_generate(
        before=foreign_before,
        after=publish.scan_audit_roots(repo_root=repo, extra_roots={"paper_input": extra_root}),
        expected_paths=[],
        bindings={
            "attempt": ATTEMPT,
            "evidence_class": EVIDENCE,
            "kind": KIND,
            "report_id": REPORT_ID,
        },
        output_root="worktree",
        repo_root=repo,
    )
    assert foreign["verdict"] == "fail"
    assert foreign["foreign_root_paths"]

    # 重解析点：由快照携带（合成属性），必须进 reparse_paths 并直接 fail
    tree = publish.scan_audit_roots(repo_root=repo)
    tree.reparse = frozenset({"worktree:src/link-to-outside"})
    reparse = publish.audit_generate(
        before=tree,
        after=tree,
        expected_paths=[],
        bindings={
            "attempt": ATTEMPT,
            "evidence_class": EVIDENCE,
            "kind": KIND,
            "report_id": REPORT_ID,
        },
        output_root="worktree",
        repo_root=repo,
    )
    assert reparse["verdict"] == "fail"
    assert reparse["reparse_paths"] == ["worktree:src/link-to-outside"]
    assert reparse["scan_scope"]["reparse_points"] == ["worktree:src/link-to-outside"]


def test_p6_t2_19_audit_rejects_unregistered_scan_scope(tmp_path: Path) -> None:
    """顶层/``data/*``（目录**与文件**）出现未登记项 ⇒ 审计盲区判据 fail。"""
    repo = _fake_repo(tmp_path)
    (repo / "data" / "mystery_box").mkdir()
    (repo / "mystery_dir").mkdir()
    (repo / "data" / "scratch.json").write_text("{}", encoding="utf-8")
    uncovered = publish.uncovered_paths(repo)
    assert uncovered == ["data/mystery_box", "data/scratch.json", "mystery_dir"]
    report = publish.audit_generate(
        before=publish.scan_audit_roots(repo_root=repo),
        after=publish.scan_audit_roots(repo_root=repo),
        expected_paths=[],
        bindings={
            "attempt": ATTEMPT,
            "evidence_class": EVIDENCE,
            "kind": KIND,
            "report_id": REPORT_ID,
        },
        output_root="worktree",
        repo_root=repo,
    )
    assert report["verdict"] == "fail"
    assert report["scan_scope"]["uncovered"] == [
        "data/mystery_box",
        "data/scratch.json",
        "mystery_dir",
    ]
    assert report["runtime_violations"]


def test_p6_t2_25_audit_requires_reparse_evidence(tmp_path: Path) -> None:
    """快照未携带 ``reparse`` 集合时，“未检查”不得读作“已检查且为空”。"""
    repo = _fake_repo(tmp_path)
    tree = publish.scan_audit_roots(repo_root=repo)
    unchecked = publish.scan_audit_roots(repo_root=repo)
    del unchecked.reparse  # 保留 dirs 证据，但缺 reparse 集合：语义上等于“未检查”
    report = publish.audit_generate(
        before=unchecked,
        after=unchecked,
        expected_paths=[],
        bindings={
            "attempt": ATTEMPT,
            "evidence_class": EVIDENCE,
            "kind": KIND,
            "report_id": REPORT_ID,
        },
        output_root="worktree",
        repo_root=repo,
    )
    assert report["verdict"] == "fail"
    assert report["scan_scope"]["reparse_checked"] is False
    assert report["runtime_violations"]

    ok = publish.audit_generate(
        before=tree,
        after=tree,
        expected_paths=[],
        bindings={
            "attempt": ATTEMPT,
            "evidence_class": EVIDENCE,
            "kind": KIND,
            "report_id": REPORT_ID,
        },
        output_root="worktree",
        repo_root=repo,
    )
    assert ok["verdict"] == "pass"
    assert ok["scan_scope"]["reparse_checked"] is True

    # ``reparse=None`` 形态同样算“未检查”（属性存在不等于已检查集合）
    none_tree = publish.scan_audit_roots(repo_root=repo)
    none_tree.reparse = None
    none_report = publish.audit_generate(
        before=none_tree,
        after=none_tree,
        expected_paths=[],
        bindings={
            "attempt": ATTEMPT,
            "evidence_class": EVIDENCE,
            "kind": KIND,
            "report_id": REPORT_ID,
        },
        output_root="worktree",
        repo_root=repo,
    )
    assert none_report["verdict"] == "fail"
    assert none_report["scan_scope"]["reparse_checked"] is False


def test_p6_t2_20_attempt_id_literal_and_discard_scope(tmp_path: Path) -> None:
    """``attempt_id`` 走 T1 同一口径；``discard_attempt`` 只删自己的目录。"""
    repo = _fake_repo(tmp_path)
    reports_root = repo / "data/reports/p6"
    for bad in (
        "..",
        ".",
        "Gen_Upper",
        "with-dash",
        "",
        "a",
        "x" * 40,
        "con",
        "aux",
        "prn",
        "nul",
        "com1",
        "lpt9",
    ):
        with pytest.raises(publish.PublishError):
            publish.discard_attempt(reports_root, bad)
        with pytest.raises(publish.PublishError):
            publish.publish_new_run(
                reports_root=reports_root,
                evidence_class=EVIDENCE,
                report_kind=KIND,
                report_id=REPORT_ID,
                files=_run_files(),
                attempt_id=bad,
            )
    keep = reports_root / "research" / "keep.txt"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("keep\n", encoding="utf-8")
    _publish(repo)
    publish.discard_attempt(reports_root, ATTEMPT)
    assert not (reports_root / ".staging" / ATTEMPT).exists()
    assert keep.is_file()
