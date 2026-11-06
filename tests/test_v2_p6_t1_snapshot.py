# -*- coding: utf-8 -*-
"""P6 T1 受信根、并发稳定快照与精确写审计（P6-T1-11 / -15 / -16）。

覆盖：symlink/junction/硬链接/逃逸/SMB 负例与 POSIX 目录多链接正例；读取期间代推进导致
staging 丢弃与重试；generate/verify 的“白名单外全等、白名单内恰为允许集合”判定与
``readonly_audit.json`` 落盘。
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import status  # noqa: E402
from src.lazybull.v2.reporting.bundle import CALIBRATION_EVIDENCE_CLASS  # noqa: E402,F401
from src.lazybull.v2.reporting.cockpit import capture_paper_inputs  # noqa: E402
from src.lazybull.v2.reporting.contracts import WRITE_WHITELIST  # noqa: E402
from src.lazybull.v2.reporting.readonly_audit import (  # noqa: E402
    FORBIDDEN_PREFIXES,
    ReadonlyGuard,
    ReadonlyViolationError,
    audit_command,
    compile_whitelist,
    is_forbidden,
    match_allowed,
    snapshot_tree,
    write_audit,
)
from src.lazybull.v2.reporting.snapshot import (  # noqa: E402
    UnstableSnapshotError,
    UntrustedRootError,
    assert_trusted_dir,
    assert_trusted_file,
    assert_trusted_root,
    capture_stable_snapshot,
)
from tests.p6_t1_env import build_paper_root  # noqa: E402

#: ``generate`` 审计必须绑定全部占位符（R2-N6 硬判据）
GENERATE_BINDINGS = {
    "attempt": "p6_t1_attempt_a",
    "evidence_class": "production",
    "kind": "dashboard",
    "report_id": "p6_dashboard_20261105T000000Z_0123456789ab",
}


def _stub_stat(real: os.stat_result, *, nlink: int) -> SimpleNamespace:
    """替换 ``os.lstat`` 返回值（只覆盖 ``st_nlink``，其余字段原样保留）。"""
    values = {name: getattr(real, name) for name in dir(real) if name.startswith("st_")}
    values["st_nlink"] = nlink
    return SimpleNamespace(**values)


def test_p6_t1_15_trusted_root_rejects_unsafe_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """symlink/junction/硬链接/逃逸/SMB 全部 fail-closed（文件硬链接尤其拒绝）。"""
    root = tmp_path / "paper"
    (root / "state").mkdir(parents=True)
    (root / "state" / "account.json").write_text("{}", encoding="utf-8")

    assert assert_trusted_root(root, root_id="paper")["root_id"] == "paper"
    with pytest.raises(UntrustedRootError):
        assert_trusted_root(root, root_id="paper", base=tmp_path / "elsewhere")
    with pytest.raises(UntrustedRootError):
        assert_trusted_root("\\\\server\\share\\paper", root_id="paper")
    with pytest.raises(UntrustedRootError):
        assert_trusted_root(root / "missing", root_id="paper")
    with pytest.raises(UntrustedRootError):
        assert_trusted_root(root, root_id="Bad-Id")

    link = tmp_path / "paper_link"
    try:
        os.symlink(root, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        link = None
    if link is not None:
        with pytest.raises(UntrustedRootError):
            assert_trusted_root(link, root_id="paper")
        with pytest.raises(UntrustedRootError):
            assert_trusted_dir(link, root=root)

    target = root / "state" / "account.json"
    hardlink = root / "state" / "account_hard.json"
    try:
        os.link(target, hardlink)
        real_lstat = os.lstat
        monkeypatch.setattr(os, "lstat", lambda path, *a, **k: real_lstat(path, *a, **k))
        with pytest.raises(UntrustedRootError):
            assert_trusted_file(hardlink, root=root)
    except (OSError, NotImplementedError):
        real_lstat = os.lstat
        monkeypatch.setattr(
            os, "lstat", lambda path, *a, **k: _stub_stat(real_lstat(path, *a, **k), nlink=2)
        )
        with pytest.raises(UntrustedRootError):
            assert_trusted_file(target, root=root)


def test_p6_t1_15b_dir_multi_links_allowed_file_hardlink_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSIX 正常目录 nlink≥2 必须通过（不得用目录 st_nlink 判受信）。"""
    root = tmp_path / "paper"
    (root / "sub" / "deeper").mkdir(parents=True)
    payload = root / "state.json"
    payload.write_text("{}", encoding="utf-8")
    real_lstat = os.lstat

    def fake_dir_lstat(path: object, *args: object, **kwargs: object) -> object:
        real = real_lstat(path, *args, **kwargs)
        return _stub_stat(real, nlink=7)

    monkeypatch.setattr(os, "lstat", fake_dir_lstat)
    assert assert_trusted_root(root, root_id="paper")["root_id"] == "paper"
    assert_trusted_dir(root / "sub" / "deeper", root=root)
    with pytest.raises(UntrustedRootError):
        assert_trusted_file(payload, root=root)


def test_p6_t1_11_generation_advance_across_reads_discards_staging(tmp_path: Path) -> None:
    """读取期间推进代（account 被改写）⇒ 丢弃 staging 并以新 attempt 重试。"""
    built = build_paper_root(tmp_path / "paper")
    paper = built["root"]
    staging = tmp_path / "staging"
    account = paper / "state" / "account.json"
    counter = {"n": 0}

    def reader(snapshot: object) -> None:
        capture_paper_inputs(snapshot)  # type: ignore[arg-type]
        counter["n"] += 1
        if counter["n"] == 1:
            account.write_text(account.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    snapshot, audit = capture_stable_snapshot(
        roots={"paper": paper},
        staging_root=staging,
        attempt_ids=["p6_t1_attempt_a", "p6_t1_attempt_b", "p6_t1_attempt_c"],
        reader=reader,
    )
    assert audit["retries"] == 1
    assert audit["attempts"][0]["stable"] is False
    assert audit["attempts"][1]["stable"] is True
    assert not (staging / "p6_t1_attempt_a").exists()
    assert snapshot.snapshot_root.is_dir()


def test_p6_t1_11b_persistent_drift_fails_closed(tmp_path: Path) -> None:
    """每次读取后都推进代 ⇒ 重试 2 次后 ``invalid/unstable_snapshot`` 且不留 staging。"""
    built = build_paper_root(tmp_path / "paper")
    paper = built["root"]
    staging = tmp_path / "staging"
    account = paper / "state" / "account.json"

    def reader(snapshot: object) -> None:
        capture_paper_inputs(snapshot)  # type: ignore[arg-type]
        account.write_text(account.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(UnstableSnapshotError) as excinfo:
        capture_stable_snapshot(
            roots={"paper": paper},
            staging_root=staging,
            attempt_ids=["p6_t1_attempt_a", "p6_t1_attempt_b", "p6_t1_attempt_c"],
            reader=reader,
        )
    row = status.make_status("invalid", reason_code="unstable_snapshot", detail="x")
    assert row["degradation_reason_code"] in str(excinfo.value)
    assert not any(staging.iterdir())


def test_p6_t1_16_generate_audit_whitelist_exact(tmp_path: Path) -> None:
    """generate：白名单内恰为本命令允许写；logs/temp/cache 额外写直接失败。"""
    worktree = tmp_path / "worktree"
    reports = worktree / "data" / "reports" / "p6"
    (reports / "production" / "dashboard").mkdir(parents=True)
    (worktree / ".gitkeep").write_text("", encoding="utf-8")
    roots = {"worktree": worktree}
    before = snapshot_tree(roots)
    report_dir = (
        reports / "production" / "dashboard" / "p6_dashboard_20261105T000000Z_0123456789ab"
    )
    allowed = WRITE_WHITELIST["generate"]

    with ReadonlyGuard(allowed_patterns=allowed, worktree_root=worktree) as guard:
        report_dir.mkdir(parents=True)
        (report_dir / "projection.json").write_text("{}", encoding="utf-8")
        (report_dir / "report_manifest.json").write_text("{}", encoding="utf-8")
        for forbidden_dir in ("logs", "temp", "__pycache__", ".pytest_cache"):
            with pytest.raises(ReadonlyViolationError):
                (worktree / forbidden_dir).mkdir(parents=True)
    assert len(guard.violations) == 4
    assert {item["path"] for item in guard.violations} == {
        "logs",
        "temp",
        "__pycache__",
        ".pytest_cache",
    }
    assert all(is_forbidden(item["path"]) for item in guard.violations)
    assert all(prefix in "".join(FORBIDDEN_PREFIXES) for prefix in ("logs/", "temp/"))

    after = snapshot_tree(roots)
    relative_dir = report_dir.relative_to(worktree).as_posix()
    expected = [f"{relative_dir}/projection.json", f"{relative_dir}/report_manifest.json"]
    report = audit_command(
        command="generate",
        output_root="worktree",
        before=before,
        after=after,
        expected_paths=expected,
        bindings=GENERATE_BINDINGS,
    )
    assert report["verdict"] == "pass"
    assert report["outside_whitelist"] == []
    assert report["forbidden_paths"] == []
    assert sorted(report["inside_whitelist"]) == sorted(expected)
    # P2-08：运行期违规必须使审计失败（前后字节全等不能把已发生的禁写尝试改判成功）
    with_violations = audit_command(
        command="generate",
        output_root="worktree",
        before=before,
        after=after,
        expected_paths=expected,
        bindings=GENERATE_BINDINGS,
        extra_violations=guard.violations,
    )
    assert with_violations["verdict"] == "fail"
    assert with_violations["runtime_violations"] and all(
        item["action"] in ("os.mkdir", "mkdir") for item in with_violations["runtime_violations"]
    )

    audit_path = tmp_path / "validations" / "readonly_audit.json"
    write_audit(audit_path, report)
    persisted = json.loads(audit_path.read_text(encoding="utf-8"))
    assert persisted["verdict"] == "pass"
    assert persisted["command"] == "generate"


def test_p6_t1_16b_verify_zero_write_and_extra_write_fails(tmp_path: Path) -> None:
    """verify：零写入才 pass；白名单内多写/少写或白名单外写入均 fail（R1-06/R2-02）。"""
    worktree = tmp_path / "worktree"
    run_dir = worktree / "data" / "reports" / "p6" / "research" / "experiment" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    roots = {"worktree": worktree}
    before = snapshot_tree(roots)

    zero_write = audit_command(
        command="verify",
        output_root="worktree",
        before=before,
        after=before,
        expected_paths=[],
    )
    assert zero_write["verdict"] == "pass"
    assert zero_write["allowed_patterns"] == []

    touched = dict(before)
    touched["worktree:data/reports/p6/research/experiment/run_a/report.html"] = {
        "sha256": "0" * 64,
        "size": 1,
    }
    modified = audit_command(
        command="verify",
        output_root="worktree",
        before=before,
        after=touched,
        expected_paths=[],
    )
    assert modified["verdict"] == "fail"
    assert modified["diff"]["modified"] == [
        "worktree:data/reports/p6/research/experiment/run_a/report.html"
    ]

    generate_scope = WRITE_WHITELIST["generate"]
    run_relative = run_dir.relative_to(worktree).as_posix()
    extra_inside = dict(before)
    extra_inside[f"worktree:{run_relative}/extra.json"] = {"sha256": "1" * 64, "size": 2}
    extra_inside[f"worktree:{run_relative}/report.html"] = before[
        f"worktree:{run_relative}/report.html"
    ]
    run_bindings = {
        "attempt": "p6_t1_attempt_a",
        "evidence_class": "research",
        "kind": "experiment",
        "report_id": "run_a",
    }
    unexpected = audit_command(
        command="generate",
        output_root="worktree",
        before=before,
        after=extra_inside,
        expected_paths=[f"{run_relative}/report.html"],
        bindings=run_bindings,
    )
    assert unexpected["verdict"] == "fail"
    assert unexpected["unexpected_inside"] == [f"{run_relative}/extra.json"]
    assert generate_scope  # 白名单非空，失败来自“恰为”条件而非匹配缺失

    missing = audit_command(
        command="generate",
        output_root="worktree",
        before=before,
        after=extra_inside,
        expected_paths=[f"{run_relative}/extra.json", f"{run_relative}/projection.json"],
        bindings=run_bindings,
    )
    assert missing["verdict"] == "fail"
    assert missing["missing_expected"] == [f"{run_relative}/projection.json"]

    outside = dict(before)
    outside["worktree:scripts/probe.py"] = {"sha256": "2" * 64, "size": 3}
    escaped = audit_command(
        command="generate",
        output_root="worktree",
        before=before,
        after=outside,
        expected_paths=["scripts/probe.py"],
        bindings=run_bindings,
    )
    assert escaped["verdict"] == "fail"
    assert escaped["outside_whitelist"] == ["scripts/probe.py"]


def test_p6_t1_16c_expected_paths_is_required(tmp_path: Path) -> None:
    """R1-06/R2-02：``expected_paths`` 缺参即 TypeError（禁止默认单向判定）。"""
    with pytest.raises(TypeError):
        audit_command(command="generate", before={}, after={})  # type: ignore[call-arg]


def test_p6_t1_16d_overwriting_existing_report_fails(tmp_path: Path) -> None:
    """R2-03：``generate`` 下改写已存在的历史报告 ⇒ fail（“新建”语义）。"""
    worktree = tmp_path / "worktree"
    relative = "data/reports/p6/production/dashboard/p6_dashboard_20261105T000000Z_0123456789ab"
    report_dir = worktree / relative
    report_dir.mkdir(parents=True)
    (report_dir / "projection.json").write_text('{"v": 1}', encoding="utf-8")
    before = snapshot_tree({"worktree": worktree})
    (report_dir / "projection.json").write_text('{"v": 2}', encoding="utf-8")
    after = snapshot_tree({"worktree": worktree})
    row = audit_command(
        command="generate",
        output_root="worktree",
        before=before,
        after=after,
        expected_paths=[f"{relative}/projection.json"],
        bindings={
            "attempt": "p6_t1_attempt_a",
            "evidence_class": "production",
            "kind": "dashboard",
            "report_id": "p6_dashboard_20261105T000000Z_0123456789ab",
        },
    )
    assert row["verdict"] == "fail"
    assert row["overwrote_existing"] == [
        f"{relative}（目录已存在，非新建目录）",
        f"{relative}/projection.json（已有报告被就地覆盖）",
    ]
    assert "data/reports/p6/.staging/p6_t1_attempt_a" in row["new_prefixes"]
    assert {relative} <= set(row["new_prefixes"])

    fresh_dir = (
        worktree
        / "data/reports/p6/production/dashboard"
        / "p6_dashboard_20261106T000000Z_aaaaaaaaaaaa"
    )
    fresh_relative = fresh_dir.relative_to(worktree).as_posix()
    before_new = snapshot_tree({"worktree": worktree})
    fresh_dir.mkdir(parents=True)
    (fresh_dir / "projection.json").write_text("{}", encoding="utf-8")
    after_new = snapshot_tree({"worktree": worktree})
    ok = audit_command(
        command="generate",
        output_root="worktree",
        before=before_new,
        after=after_new,
        expected_paths=[f"{fresh_relative}/projection.json"],
        bindings={
            "attempt": "p6_t1_attempt_b",
            "evidence_class": "production",
            "kind": "dashboard",
            "report_id": "p6_dashboard_20261106T000000Z_aaaaaaaaaaaa",
        },
    )
    assert ok["verdict"] == "pass"


def test_p6_t1_16h_generate_without_bindings_is_rejected(tmp_path: Path) -> None:
    """R2-N6：``generate`` 未绑定全部占位符 ⇒ ValueError（禁止静默放弃“新目录”判据）。"""
    worktree = tmp_path / "worktree"
    relative = "data/reports/p6/production/dashboard/p6_dashboard_20261105T000000Z_0123456789ab"
    report_dir = worktree / relative
    report_dir.mkdir(parents=True)
    (report_dir / "projection.json").write_text("{}", encoding="utf-8")
    before = snapshot_tree({"worktree": worktree})
    (report_dir / "projection.json").write_text('{"v": 2}', encoding="utf-8")
    after = snapshot_tree({"worktree": worktree})
    with pytest.raises(ValueError):
        audit_command(
            command="generate",
            output_root="worktree",
            before=before,
            after=after,
            expected_paths=[f"{relative}/projection.json"],
        )
    with pytest.raises(ValueError):  # 只绑一部分也不允许
        audit_command(
            command="generate",
            output_root="worktree",
            before=before,
            after=after,
            expected_paths=[f"{relative}/projection.json"],
            bindings={"report_id": "p6_dashboard_20261105T000000Z_0123456789ab"},
        )


def test_p6_t1_16i_measure_and_reproduce_require_bindings(tmp_path: Path) -> None:
    """R2-N7：``measure``/``reproduce`` 的“新 ``<new_id>``”判据同样强制绑定（禁止静默关闭）。"""
    worktree = tmp_path / "worktree"
    relative = "data/reports/p6/validations/research/p6_t1_acceptance_run9"
    target = worktree / relative
    target.mkdir(parents=True)
    (target / "t1_acceptance.json").write_text('{"v": 1}', encoding="utf-8")
    before = snapshot_tree({"worktree": worktree})
    (target / "t1_acceptance.json").write_text('{"v": 2}', encoding="utf-8")
    after = snapshot_tree({"worktree": worktree})
    for command in ("measure", "real_golden", "reproduce"):
        with pytest.raises(ValueError):
            audit_command(
                command=command,
                output_root="worktree",
                before=before,
                after=after,
                expected_paths=[f"{relative}/t1_acceptance.json"],
            )
    bound = audit_command(
        command="measure",
        output_root="worktree",
        before=before,
        after=after,
        expected_paths=[f"{relative}/t1_acceptance.json"],
        bindings={"evidence_class": "research", "validation_id": "p6_t1_acceptance_run9"},
    )
    assert bound["verdict"] == "fail"  # 绑定后“覆盖已存在 validation 目录”必须 fail
    assert bound["overwrote_existing"] == [
        f"{relative}（目录已存在，非新建目录）",
        f"{relative}/t1_acceptance.json（已有报告被就地覆盖）"
    ]


def test_p6_t1_16k_bindings_must_be_literal(tmp_path: Path) -> None:
    """R2-N9：``bindings`` 值含通配符必须拒绝（否则 ``*`` 会放宽白名单，fail-open）。"""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    before = snapshot_tree({"worktree": worktree})
    after = {
        "worktree:data/reports/p6/validations/research/OTHER_RUN/f.json": {
            "sha256": "1" * 64,
            "size": 1,
        }
    }
    for bad in ("*", "**", "run?", "a/b"):
        with pytest.raises(ValueError):
            audit_command(
                command="measure",
                output_root="worktree",
                before=before,
                after=after,
                expected_paths=["data/reports/p6/validations/research/OTHER_RUN/f.json"],
                bindings={"evidence_class": "research", "validation_id": bad},
            )
    strict = audit_command(
        command="measure",
        output_root="worktree",
        before=before,
        after=after,
        expected_paths=["data/reports/p6/validations/research/OTHER_RUN/f.json"],
        bindings={"evidence_class": "research", "validation_id": "p6_t1_acceptance_run9"},
    )
    assert strict["verdict"] == "fail"
    assert strict["outside_whitelist"] == [
        "data/reports/p6/validations/research/OTHER_RUN/f.json"
    ]


def test_p6_t1_16l_binding_edge_cases(tmp_path: Path) -> None:
    """R5-01/R2-N10：未引用的多余绑定忽略；点开头/结尾的绑定值拒绝（Windows 尾点）。"""
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    before = snapshot_tree({"worktree": worktree})
    after = {
        "worktree:data/reports/p6/validations/research/run9/f.json": {"sha256": "1" * 64, "size": 1}
    }
    tolerant = audit_command(
        command="measure",
        output_root="worktree",
        before=before,
        after=after,
        expected_paths=["data/reports/p6/validations/research/run9/f.json"],
        bindings={
            "evidence_class": "research",
            "validation_id": "run9",
            "unused": "a*b",
        },
    )
    assert tolerant["verdict"] == "pass"  # 共享绑定表可整体透传（未引用键不校验）
    for bad in ("run9.", ".run9"):
        with pytest.raises(ValueError):
            audit_command(
                command="measure",
                output_root="worktree",
                before=before,
                after=after,
                expected_paths=["data/reports/p6/validations/research/run9/f.json"],
                bindings={"evidence_class": "research", "validation_id": bad},
            )


def test_p6_t1_16m_input_root_write_cannot_be_whitelisted(tmp_path: Path) -> None:
    """P1-03：受审树含多根时，输入根写入不得借同名相对路径进白名单。"""
    worktree = tmp_path / "worktree"
    input_root = tmp_path / "input"
    relative = "data/reports/p6/validations/research/probe_old"
    (worktree / relative).mkdir(parents=True)
    (input_root / relative).mkdir(parents=True)
    roots = {"worktree": worktree, "input": input_root}
    before = snapshot_tree(roots)
    (input_root / relative / "new.json").write_text("{}", encoding="utf-8")
    after = snapshot_tree(roots)
    bindings = {"evidence_class": "research", "validation_id": "probe_old"}
    with pytest.raises(TypeError):  # output_root 必填：不传无法判定输出根
        audit_command(  # type: ignore[call-arg]
            command="measure",
            before=before,
            after=after,
            expected_paths=[f"{relative}/new.json"],
            bindings=bindings,
        )
    row = audit_command(
        command="measure",
        output_root="worktree",
        before=before,
        after=after,
        expected_paths=[f"{relative}/new.json"],
        bindings=bindings,
    )
    assert row["verdict"] == "fail"
    assert row["foreign_root_paths"] == [f"input:{relative}/new.json"]
    assert row["inside_whitelist"] == []


def test_p6_t1_16n_appending_to_existing_dir_fails(tmp_path: Path) -> None:
    """P2-07：保留旧文件再追加新文件也算“非新建目录”，必须 fail。"""
    worktree = tmp_path / "worktree"
    relative = "data/reports/p6/validations/research/probe_old"
    target = worktree / relative
    target.mkdir(parents=True)
    (target / "old.json").write_text("{}", encoding="utf-8")
    before = snapshot_tree({"worktree": worktree})
    (target / "new.json").write_text("{}", encoding="utf-8")
    after = snapshot_tree({"worktree": worktree})
    row = audit_command(
        command="measure",
        output_root="worktree",
        before=before,
        after=after,
        expected_paths=[f"{relative}/new.json"],
        bindings={"evidence_class": "research", "validation_id": "probe_old"},
    )
    assert row["verdict"] == "fail"
    assert any("非新建目录" in item for item in row["overwrote_existing"])


def test_p6_t1_16o_reusing_existing_empty_dir_fails(tmp_path: Path) -> None:
    """R2-20261106-07（复审）：已有**空目录**被复用也必须 fail（需目录集合证据）。"""
    worktree = tmp_path / "worktree"
    relative = "data/reports/p6/validations/research/probe_empty"
    (worktree / relative).mkdir(parents=True)
    before = snapshot_tree({"worktree": worktree})  # FileTree 带 dirs
    (worktree / relative / "new.json").write_text("{}", encoding="utf-8")
    after = snapshot_tree({"worktree": worktree})
    bindings = {"evidence_class": "research", "validation_id": "probe_empty"}
    row = audit_command(
        command="measure",
        output_root="worktree",
        before=before,
        after=after,
        expected_paths=[f"{relative}/new.json"],
        bindings=bindings,
    )
    assert row["verdict"] == "fail"
    assert any("目录已存在" in item for item in row["overwrote_existing"])
    with pytest.raises(ValueError):  # 无目录证据即拒绝判定
        audit_command(
            command="measure",
            output_root="worktree",
            before=dict(before),
            after=dict(after),
            expected_paths=[f"{relative}/new.json"],
            bindings=bindings,
        )


def test_p6_t1_16j_guard_blocks_all_write_entry_points(tmp_path: Path) -> None:
    """R1-05/R2-01：``Path.open``/``io.open``/``os.open``/链接/时间戳/临时文件/删除均被拦。"""
    worktree = tmp_path / "worktree"
    (worktree / "scripts").mkdir(parents=True)
    (worktree / "scripts" / "keep.py").write_text("", encoding="utf-8")
    target = worktree / "scripts" / "a.py"
    attempts = {
        "path_open": lambda: target.open("w", encoding="utf-8"),
        "io_open": lambda: io.open(target, "w", encoding="utf-8"),
        "os_open": lambda: os.open(str(target), os.O_CREAT | os.O_WRONLY),
        "path_write_text": lambda: target.write_text("x", encoding="utf-8"),
        "os_link": lambda: os.link(worktree / "scripts" / "keep.py", target),
        "os_utime": lambda: os.utime(worktree / "scripts" / "keep.py", (1, 1)),
        "os_truncate": lambda: os.truncate(worktree / "scripts" / "keep.py", 0),
        "tempfile": lambda: tempfile.NamedTemporaryFile(dir=worktree / "scripts"),
        "os_unlink": lambda: os.unlink(worktree / "scripts" / "keep.py"),
        "shutil_rmtree": lambda: shutil.rmtree(worktree / "scripts", ignore_errors=False),
    }
    with ReadonlyGuard(allowed_patterns=(), worktree_root=worktree) as guard:
        for name, action in attempts.items():
            with pytest.raises(ReadonlyViolationError):
                action()
    # CPython 审计事件名：`open()`/`os.open()` 均为 ``open``，``os.unlink`` 为 ``os.remove``
    assert {item["action"] for item in guard.violations} >= {
        "open",
        "os.link",
        "os.utime",
        "os.truncate",
        "os.remove",
        "shutil.rmtree",
    }
    assert all(item["path"] != "<socket>" for item in guard.violations)
    assert (worktree / "scripts" / "keep.py").is_file()  # 未发生破坏


def test_p6_t1_16f_junction_escape_fails(tmp_path: Path) -> None:
    """R2-05：经 junction 把写入洗到工作树外 ⇒ 运行期拦截（不返回 None 放行）。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    worktree = tmp_path / "worktree"
    (worktree / "data").mkdir(parents=True)
    junction = worktree / "data" / "link"
    created = False
    try:
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
            capture_output=True,
            text=True,
            check=False,
        )
        created = result.returncode == 0
    except OSError:
        created = False
    if not created:
        return  # 无建 junction 权限的环境跳过该攻击面（其余用例仍覆盖重解析点判定）
    with ReadonlyGuard(allowed_patterns=(), worktree_root=worktree) as guard:
        with pytest.raises(ReadonlyViolationError):
            (junction / "leak.json").write_text("{}", encoding="utf-8")
    assert guard.violations[0]["kind"] == "reparse_escape"


def test_p6_t1_16g_whitelist_rejects_dot_segments_and_forbidden_case(
    tmp_path: Path,
) -> None:
    """R2-06/R2-07：``..`` 路径不匹配白名单；``Logs``/``logs.``/``" logs"`` 同判禁止段。"""
    compiled = compile_whitelist(["data/reports/p6/validations/<id>/**"])
    assert match_allowed("data/reports/p6/validations/../../data/paper/x.json", compiled) is False
    assert compile_whitelist(["data/reports/p6/production/dashboard/<report_id>/**"])
    with pytest.raises(ValueError):
        compile_whitelist(["data/../data/reports/**"])
    for variant in ("logs/a.json", "Logs/a.json", "LOGS/a.json", " logs/a.json", "logs./a.json"):
        assert is_forbidden(variant) is True, variant
    assert is_forbidden("logs/x/y.json") is True
    assert is_forbidden("data/reports/p6/logs_meta.json") is False
