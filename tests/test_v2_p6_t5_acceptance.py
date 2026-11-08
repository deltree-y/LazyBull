# -*- coding: utf-8 -*-
"""P6 T5 验收驱动的纯函数与判定语义测试。

覆盖口径：判定聚合（写域/零写入/历史/双平台）、探针解析、包内重建（真实生成器正面
依据）、参数形态；**不**在 pytest 内运行真实 smoke / 真实 golden / Chromium 测量
（那是 T5 独立验收，见验收载荷）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "v2_p6" / "run_t5_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("p6_t5_run_acceptance", SCRIPT)
driver = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(driver)


def _good_smoke(**overrides: Any) -> dict[str, Any]:
    """构造一条“写域全绿”的 smoke 记录（只替换被测算的字段）。"""
    record: dict[str, Any] = {
        "label": "unit_smoke",
        "exit_code": 0,
        "verdict": "pass",
        "audit": {
            "verdict": "pass",
            "uncovered": [],
            "reparse_checked": True,
            "reparse_points": [],
            "missing_expected": [],
            "unexpected_inside": [],
            "outside_whitelist": [],
            "foreign_root_paths": [],
            "overwrote_existing": [],
            "forbidden_paths": [],
        },
        "driver_audit": {
            "verdict": "pass",
            "outside_whitelist": [],
            "unexpected_inside": [],
            "missing_expected": [],
            "foreign_root_paths": [],
            "overwrote_existing": [],
            "changed": [],
            "uncovered": [],
        },
        "input_evidence": {"roots": ["paper"], "unchanged": True},
    }
    record.update(overrides)
    return record


def test_t5_acceptance_check_helpers() -> None:
    """``_check``/``_all_passed`` 保守语义：空集合不得算通过。"""
    assert driver._all_passed([driver._check("a", True)]) is True
    assert driver._all_passed([driver._check("a", True), driver._check("b", False)]) is False
    assert driver._all_passed([]) is False
    assert driver._check("a", 1)["passed"] is True


def test_t5_acceptance_audit_checks_all_green() -> None:
    """三层写域证据全绿 ⇒ 全部判据通过。"""
    checks = driver._audit_checks(_good_smoke())
    assert [item["name"] for item in checks] == [
        "exit_code_zero",
        "payload_verdict",
        "product_audit_pass",
        "product_audit_clean",
        "driver_audit_pass",
        "driver_audit_clean",
        "inputs_unchanged",
    ]
    assert driver._all_passed(checks)


@pytest.mark.parametrize(
    ("mutate", "failed_name"),
    [
        (lambda record: record.__setitem__("exit_code", 3), "exit_code_zero"),
        (lambda record: record.__setitem__("verdict", "fail"), "payload_verdict"),
        (lambda record: record["audit"].__setitem__("verdict", "fail"), "product_audit_pass"),
        (lambda record: record["audit"].__setitem__("uncovered", ["x"]), "product_audit_clean"),
        (
            lambda record: record["audit"].__setitem__("reparse_checked", None),
            "product_audit_clean",
        ),
        (
            lambda record: record["audit"].__setitem__("foreign_root_paths", ["paper:x"]),
            "product_audit_clean",
        ),
        (lambda record: record["driver_audit"].__setitem__("verdict", "fail"), "driver_audit_pass"),
        (
            lambda record: record["driver_audit"].__setitem__("missing_expected", ["y"]),
            "driver_audit_clean",
        ),
        (
            lambda record: record["driver_audit"].__setitem__("uncovered", ["data/raw"]),
            "driver_audit_clean",
        ),
        (
            lambda record: record["input_evidence"].__setitem__("unchanged", False),
            "inputs_unchanged",
        ),
    ],
)
def test_t5_acceptance_audit_checks_single_fault_fails(mutate: Any, failed_name: str) -> None:
    """任一写域证据被破坏都必须判失败（不得“其它项通过就放行”）。"""
    record = _good_smoke()
    mutate(record)
    checks = {item["name"]: item["passed"] for item in driver._audit_checks(record)}
    assert checks[failed_name] is False
    assert not driver._all_passed(driver._audit_checks(record))


def test_t5_acceptance_zero_write_aggregation() -> None:
    """零写入汇总：任一条 smoke 失败即整项失败。"""
    smokes = {"p4": _good_smoke(), "runs": _good_smoke(), "p5b": _good_smoke()}
    payload = driver.zero_write(smokes)
    assert payload["passed"] is True
    assert set(payload["smokes"]) == {"p4", "runs", "p5b"}
    assert payload["runtime_guard"]["deny_network"] is True
    broken = {"p4": _good_smoke(), "runs": {"label": "no-audit"}, "p5b": _good_smoke()}
    assert driver.zero_write(broken)["passed"] is False


def test_t5_acceptance_history_detects_overwrite_and_foreign_add() -> None:
    """历史检查：既有文件改动/删除、以及允许集合外新增都必须失败。"""
    before = {
        "worktree:data/reports/p6/research/trend/old/report.html": {"sha256": "a" * 64},
        "worktree:data/reports/p6/validations/research/old/t3.json": {"sha256": "b" * 64},
    }
    record = {"label": "unit", "_before_tree": before, "_after_tree": dict(before)}
    ok = driver._history_of(record, ["data/reports/p6/validations/research/newid"])
    assert ok["passed"] is True

    after = dict(before)
    after["worktree:data/reports/p6/research/trend/old/report.html"] = {"sha256": "c" * 64}
    tampered = {"label": "unit", "_before_tree": before, "_after_tree": after}
    row = driver._history_of(tampered, ["data/reports/p6/validations/research/newid"])
    assert row["passed"] is False
    assert row["modified"]

    added_foreign = dict(before)
    added_foreign["worktree:src/lazybull/v2/reporting/render.py"] = {"sha256": "d" * 64}
    row = driver._history_of(
        {"label": "unit", "_before_tree": before, "_after_tree": added_foreign},
        ["data/reports/p6/validations/research/newid"],
    )
    assert row["passed"] is False
    assert row["added_outside_allowed"] == ["worktree:src/lazybull/v2/reporting/render.py"]

    removed = {key: value for key, value in before.items() if "old/t3.json" not in key}
    row = driver._history_of(
        {"label": "unit", "_before_tree": before, "_after_tree": removed},
        ["data/reports/p6/validations/research/newid"],
    )
    assert row["passed"] is False
    assert row["removed"]


def test_t5_acceptance_unmet_excludes_waived_and_design_items() -> None:
    """用户豁免（waived_by_user）与设计约束（by_design）不计入 unmet；其余未达成仍会阻断 pass。"""
    unmet = [item["id"] for item in driver.KNOWN_LIMITATIONS if item["status"] == "not_achieved"]
    assert unmet == [], unmet
    waived = {item["id"] for item in driver.USER_WAIVERS}
    assert {"L1", "L2"} <= waived
    statuses = {item["id"]: item["status"] for item in driver.KNOWN_LIMITATIONS}
    assert statuses["L1"] == "waived_by_user" and statuses["L2"] == "waived_by_user"
    assert statuses["L10"] == "by_design"
    assert driver._final_verdict([], []) == "pass"


def test_t5_acceptance_t5_test_files_cover_disk() -> None:
    """R3-T5-R1-03：磁盘上的 `tests/test_v2_p6_t5_*.py` 必须全部纳入验收清单。"""
    assert driver._missing_t5_test_files() == []
    assert "tests/test_v2_p6_t5_selftest_guard.py" in driver.T5_TEST_FILES


def test_t5_acceptance_final_verdict_blocks_pass_on_unmet() -> None:
    """R3-T5-02：有未达成必需项时总体验收不得为 pass。"""
    assert driver._final_verdict([], []) == "pass"
    assert driver._final_verdict([], ["L1"]) == "incomplete"
    assert driver._final_verdict(["reproduce"], []) == "fail"
    assert driver._final_verdict(["reproduce"], ["L1"]) == "fail"


def _identity(platform_name: str, python: str = "3.12.10", modules=None) -> dict:
    """固定身份夹具（不得依赖执行主机，评审 R3-T5-R1-02）。"""
    return {
        "exit_code": 0,
        "platform": platform_name,
        "python": python,
        "modules": (
            modules
            if modules is not None
            else {"jinja2": "3.1.6", "pandas": "3.0.0", "numpy": "2.0.0"}
        ),
        "stderr_tail": [],
    }


def test_t5_acceptance_dual_platform_rejects_non_linux_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-T5-03/R1-02：Windows 身份被拒（mock 身份，不依赖执行主机）。"""
    monkeypatch.setattr(driver, "_linux_probe", lambda: {"available": False, "reason": "unit"})
    monkeypatch.setattr(driver, "_linux_identity", lambda python, cwd: _identity("win32"))
    result = driver.dual_platform(
        "unit_dual",
        {},
        {"rows": {}, "passed": True},
        work_root=None,
        linux_python="python",
        linux_cwd=str(ROOT),
    )
    assert result["linux"]["status"] == "rejected_not_linux"
    assert result["achieved"] is False
    assert result["unmet"] == ["linux_dual_platform"]
    assert result["passed"] is False


def test_t5_acceptance_dual_platform_rejects_old_or_deprived_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-T5-R1-02：Linux 身份但 Python<3.12 或依赖缺失 ⇒ rejected_identity（不执行对照）。"""
    monkeypatch.setattr(driver, "_linux_probe", lambda: {"available": True, "reason": "unit"})
    monkeypatch.setattr(driver, "_run", lambda *a, **k: pytest.fail("不得进入生成分支"))
    monkeypatch.setattr(
        driver, "_linux_identity", lambda python, cwd: _identity("linux", python="3.10.12")
    )
    old = driver.dual_platform(
        "unit_old",
        {},
        {"rows": {}, "passed": True},
        work_root=None,
        linux_python="python3",
        linux_cwd=str(ROOT),
    )
    assert old["linux"]["status"] == "rejected_identity"
    assert old["achieved"] is False

    monkeypatch.setattr(
        driver,
        "_linux_identity",
        lambda python, cwd: _identity(
            "linux", modules={"jinja2": "3.1.6", "pandas": None, "numpy": "2.0.0"}
        ),
    )
    deprived = driver.dual_platform(
        "unit_deps",
        {},
        {"rows": {}, "passed": True},
        work_root=None,
        linux_python="python3",
        linux_cwd=str(ROOT),
    )
    assert deprived["linux"]["status"] == "rejected_identity"
    assert deprived["achieved"] is False


def test_t5_acceptance_dual_platform_accepts_verified_linux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R3-T5-R1-02 正面分支：身份合格且三面哈希全等 ⇒ achieved=True（生成用替身）。"""
    run_dir = tmp_path / "linux_run"
    run_dir.mkdir()
    (run_dir / "report.html").write_text("x", encoding="utf-8")
    digest = hashlib.sha256(b"x").hexdigest()
    rows = {
        kind: {"files": {"report.html": digest}, "canonical_sha256": "c" * 64}
        for kind in driver.KINDS
    }
    monkeypatch.setattr(driver, "_linux_probe", lambda: {"available": True})
    monkeypatch.setattr(driver, "_linux_identity", lambda python, cwd: _identity("linux"))
    monkeypatch.setattr(
        driver,
        "_run",
        lambda *a, **k: {
            "exit_code": 0,
            "stdout": json.dumps({"report_path": str(run_dir)}),
            "stderr_tail": [],
            "command": list(a[0]) if a else [],
        },
    )
    result = driver.dual_platform(
        "unit_ok",
        {},
        {"rows": rows, "passed": True},
        work_root=None,
        linux_python="python3",
        linux_cwd=str(ROOT),
    )
    assert result["linux"]["status"] == "executed"
    assert result["achieved"] is True
    assert result["unmet"] == []
    assert result["passed"] is True


def test_t5_acceptance_face_rejects_failed_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R3-T5-04：5 次生成全部失败时，面判定必须失败（不得靠 byte_identical 蒙过）。"""
    import shutil

    work = ROOT / "temp" / "p6_t5_unit_face"
    if work.exists():
        shutil.rmtree(work)
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    failing = {
        "gate": {"passed": True, "max_ms": 1.0},
        "max_ms": 1.0,
        "threshold_ms": 10000,
        "byte_identical": True,
        "all_succeeded": False,
        "passed": False,
        "records": [
            {"exit_code": 3, "report_path": str(run_dir), "report_id": f"r{i}"} for i in range(5)
        ],
    }
    monkeypatch.setattr(driver, "_generation_evidence", lambda *a, **k: dict(failing))
    monkeypatch.setattr(
        driver.run_t2_acceptance,
        "_timed_generation",
        lambda *a, **k: {"records": [{"report_path": str(run_dir), "report_id": "r0"}]},
    )
    monkeypatch.setattr(
        driver.build_t2_corpus,
        "build_case",
        lambda root, profile: {"paper": str(tmp_path / "p"), "quality_root": str(tmp_path / "q")},
    )
    monkeypatch.setattr(
        driver,
        "_open_gate",
        lambda *a, **k: {"passed": True, "open_max_ms": 1.0, "report_bytes": 10},
    )
    try:
        row = driver._face_evidence("dashboard", "standard", "unit_face", work / "out", work / "w")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    assert row["generation"]["all_succeeded"] is False
    names = {item["name"]: item["passed"] for item in row["checks"]}
    assert names["generation_all_succeeded"] is False
    assert names["generation_ok"] is False
    assert row["passed"] is False


def test_t5_acceptance_tree_digest_fails_closed(tmp_path: Path) -> None:
    """输入根缺失/为空不得真空通过（`entries=[]` 会让 `unchanged=True` 假成立，评审 B2-1）。"""
    with pytest.raises(driver.AcceptanceSetupError):
        driver._tree_digest({"missing": tmp_path / "nope"})
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(driver.AcceptanceSetupError):
        driver._tree_digest({"empty": empty})


def test_t5_acceptance_native_batch_rejects_paper_replay(tmp_path: Path) -> None:
    """原生提交批次判据：`paper_state`/`checkpoints` 目录不是批次（即使带 summary）。"""
    good = tmp_path / "run_ok"
    good.mkdir()
    for name in driver.NATIVE_BATCH_FILES:
        (good / name).write_text("{}", encoding="utf-8")
    (good / "batch_meta.json").write_text('{"created_at": "2026-10-31T00:00:00"}', encoding="utf-8")
    (good / "folds").mkdir()
    assert driver._native_batch(good)[0] is True

    replay = tmp_path / "run_replay"
    replay.mkdir()
    for name in driver.NATIVE_BATCH_FILES:
        (replay / name).write_text("{}", encoding="utf-8")
    (replay / "batch_meta.json").write_text(
        '{"created_at": "2026-10-31T00:00:00"}', encoding="utf-8"
    )
    (replay / "folds").mkdir()
    (replay / "paper_state").mkdir()
    native, reason, _ = driver._native_batch(replay)
    assert native is False
    assert "回放标记" in reason


def test_t5_acceptance_generation_all_succeeded_without_default() -> None:
    """`all_succeeded` 不取默认 True：缺键时按逐条 `exit_code` 判定（评审 B3-2）。"""
    gate = {"passed": True, "max_ms": 1.0}
    ok = driver._generation_evidence(
        "dashboard",
        "standard",
        {"gate": gate, "byte_identical": True, "records": [{"exit_code": 0}, {"exit_code": 0}]},
    )
    bad = driver._generation_evidence(
        "dashboard",
        "standard",
        {"gate": gate, "byte_identical": True, "records": [{"exit_code": 0}, {"exit_code": 2}]},
    )
    empty = driver._generation_evidence(
        "dashboard", "standard", {"gate": gate, "byte_identical": True, "records": []}
    )
    assert ok["all_succeeded"] is True and ok["passed"] is True
    assert bad["all_succeeded"] is False and bad["passed"] is False
    assert empty["all_succeeded"] is False and empty["passed"] is False


def test_t5_acceptance_top_buckets_aggregates_by_prefix() -> None:
    """新增文件按目录前缀聚合（供第三方独立复核“全在允许前缀”，评审 A-04）。"""
    keys = [
        "worktree:data/reports/p6/validations/research/r1/a.json",
        "worktree:data/reports/p6/validations/research/r1/b.json",
        "worktree:data/reports/p6/production/dashboard/x/report.html",
    ]
    assert driver._top_buckets(keys, depth=4) == {
        "data/reports/p6/validations": 2,
        "data/reports/p6/production": 1,
    }


def test_t5_acceptance_report_prefixes_cover_run_and_audit() -> None:
    """允许新增集合 = 新报告 run 目录 + 本次审计目录（缺一即会被历史/整轮判失败）。"""
    record = {
        "report_path": str(ROOT / "data/reports/p6/production/dashboard/run_x"),
        "audit": {"path": "data/reports/p6/validations/production/run_x_audit/readonly_audit.json"},
    }
    assert driver._report_prefixes(record) == [
        "data/reports/p6/production/dashboard/run_x",
        "data/reports/p6/validations/production/run_x_audit",
    ]
    assert driver._report_prefixes({}) == []


def test_t5_acceptance_produced_prefixes_are_exact_dirs() -> None:
    """整轮允许集合按**精确目录**枚举：兄弟目录（`_golden`/`_open_*`/`_audit_*`）必须各自登记。"""
    checks = {
        "real_golden": {"out": "data/reports/p6/validations/research/r5_golden"},
        "real_smoke": {
            "smokes": {
                "p4": {
                    "report_path": str(ROOT / "data/reports/p6/production/dashboard/run_x"),
                    "audit": {
                        "path": "data/reports/p6/validations/production/r5_audit_p4/"
                        "readonly_audit.json"
                    },
                }
            }
        },
        "three_faces": {
            "faces": {
                "dashboard": {
                    "standard": {
                        "open": {
                            "performance_json": "data/reports/p6/validations/research/"
                            "r5_open_dashboard_standard/performance.json"
                        }
                    }
                }
            }
        },
    }
    prefixes = driver._produced_prefixes(checks, "r5")
    expected = {
        "data/reports/p6/validations/research/r5",
        "data/reports/p6/validations/research/r5_golden",
        "data/reports/p6/validations/research/r5_open_dashboard_standard",
        "data/reports/p6/validations/production/r5_audit_p4",
        "data/reports/p6/production/dashboard/run_x",
    }
    assert expected <= set(prefixes)
    inside = "data/reports/p6/validations/research/r5_golden/real_golden.json"
    assert any(driver._within(inside, prefix) for prefix in prefixes)
    sibling = "data/reports/p6/validations/research/r5_other/report.html"
    assert not any(driver._within(sibling, prefix) for prefix in prefixes)


def test_t5_acceptance_global_scope() -> None:
    """整轮写域：允许前缀内的新增放行，其它根/前缀一律判失败。"""
    before = {"worktree:data/paper/account.json": {"sha256": "a" * 64}}
    after = dict(before)
    after["worktree:data/reports/p6/validations/research/run1/t5_acceptance.json"] = {
        "sha256": "b" * 64
    }
    ok = driver._global_scope(before, after, ["data/reports/p6/validations/research/run1"])
    assert ok["passed"] is True
    assert ok["added"] == 1

    foreign = dict(after)
    foreign["campaign:reports/drawdown_manifest.json"] = {"sha256": "c" * 64}
    row = driver._global_scope(before, foreign, ["data/reports/p6/validations/research/run1"])
    assert row["passed"] is False
    assert row["added_outside_allowed"] == ["campaign:reports/drawdown_manifest.json"]


def test_t5_acceptance_linux_probe_parser() -> None:
    """探测输出解析：只登记真实出现的键，不伪造缺失项。"""
    text = "PY=Python 3.10.12\nDEPSLINE=ModuleNotFoundError: No module named 'pandas'\nPIP=no\n"
    parsed = driver._parse_probe(text)
    assert parsed["PY"] == "Python 3.10.12"
    assert parsed["PIP"] == "no"
    assert "pandas" in parsed["deps_line"]
    assert "KERNEL" not in parsed


def test_t5_acceptance_dual_platform_registers_unmet_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 Linux 环境时显式 ``not_achieved``（不得用跳过代替），且给出可执行准备包。"""
    monkeypatch.setattr(
        driver,
        "_linux_probe",
        lambda: {"available": False, "reason": "unit-probe", "probe": {}},
    )
    faces = {
        kind: {"standard": {"passed": True, "reproduce_args": ["--kind", kind]}}
        for kind in driver.KINDS
    }
    reproduce = {
        "passed": True,
        "rows": {
            kind: {"canonical_sha256": "f" * 64, "files": {"report.html": "e" * 64}}
            for kind in driver.KINDS
        },
    }
    payload = driver.dual_platform("unit", faces, reproduce)
    assert payload["linux"]["status"] == "not_achieved"
    assert payload["linux"]["prepared"] is True
    assert payload["linux"]["linux_commands"]
    assert payload["unmet"] == ["linux_dual_platform"]
    assert payload["passed"] is True
    assert all(item["passed"] for item in payload["checks"])


def test_t5_acceptance_dual_platform_requires_windows_canonical() -> None:
    """Windows 侧缺 canonical ⇒ 该判据失败（不允许“只登记 Linux 未达成”就通过）。"""
    faces = {kind: {"standard": {"passed": True, "reproduce_args": []}} for kind in driver.KINDS}
    payload = driver.dual_platform("unit", faces, {"passed": False, "rows": {}})
    assert payload["passed"] is False


def test_t5_acceptance_driver_audit_fails_closed_on_unbound_placeholders() -> None:
    """驱动侧审计缺占位符绑定 ⇒ 直接 ValueError（“新目录”判据不得退化为通配）。"""
    from src.lazybull.v2.reporting import readonly_audit

    empty = readonly_audit.FileTree({}, [])
    empty.reparse = frozenset()
    with pytest.raises(ValueError):
        driver._driver_audit(
            before=empty,
            after=empty,
            expected_paths=[],
            allowed_patterns=driver._driver_patterns(
                "data/reports/p6/validations/research/unit_audit"
            ),
            bindings={"evidence_class": "research"},
        )


def test_t5_acceptance_driver_patterns_cover_generate_plus_audit() -> None:
    """驱动侧白名单 = generate 冻结白名单 + 本次 validation 目录（不是整体放宽）。"""
    patterns = driver._driver_patterns("data/reports/p6/validations/research/unit_audit")
    assert list(driver.contracts.WRITE_WHITELIST["generate"]) == patterns[:2]
    assert patterns[-1] == "data/reports/p6/validations/research/unit_audit/**"


def test_t5_acceptance_face_args_are_kind_specific() -> None:
    """三面 generate 参数形态：必须含各自的输入白名单开关与固定 generated_at。"""
    dashboard = driver._face_publish_args(
        "dashboard", "", {"paper": Path("temp/p"), "quality": Path("temp/q")}
    )
    assert "--paper-root" in dashboard and "--quality-root" in dashboard
    assert driver.FACE_GENERATED_AT["dashboard"] in dashboard
    experiment = driver._face_publish_args(
        "experiment", "", {"base": Path("temp/b"), "arm": Path("temp/a")}
    )
    assert experiment.count("--arm") == 2 and "--baseline" in experiment
    trend = driver._face_publish_args(
        "trend", "", {"new_run": Path("temp/n"), "previous": Path("temp/prev")}
    )
    assert "--previous" in trend and "runs:temp/n" in trend


def test_t5_acceptance_pass_required_matches_collect_keys() -> None:
    """``_PASS_REQUIRED`` 必须覆盖 ``_collect`` 产出的全部判定键（改名即失败）。"""
    assert set(driver._PASS_REQUIRED) == {
        "pytest",
        "complexity",
        "real_golden",
        "real_smoke",
        "zero_write",
        "historical",
        "three_faces",
        "reproduce",
        "dual_platform",
        "documentation",
        "global_scope",
    }


def test_t5_acceptance_package_internal_rebuild_is_diagnostic_only(tmp_path: Path) -> None:
    """包内重建是**诊断项**：差异必须精确定位到 footer 的 generator 行（不得冒充通过）。

    实测口径（设计 §8.3 层 2）：footer 以 ``dict`` repr 渲染 ``generator_files``，
    canonical JSON（``sort_keys``）往返丢失插入序 ⇒ 包内重建字节不等；复算判据是层 3。
    """
    from scripts.v2_p6 import build_t2_corpus
    from src.lazybull.v2.reporting import generate

    case = build_t2_corpus.build_case(tmp_path / "corpus", "standard")
    request = generate.DashboardRequest(
        paper_root=Path(str(case["paper"])),
        quality_root=Path(str(case["quality_root"])),
        reports_root=tmp_path / "reports",
        evidence_class="research",
        generated_at="20261107T000000Z",
        attempt_id="unit_package_internal",
        input_evidence_classes=("research",),
        report_kind="dashboard",
        git_record={"git_commit": "unit", "git_dirty": False},
        publish_run=True,
    )
    result = generate.generate_dashboard(request)
    run_dir = Path(result["report_path"])
    probe = driver._package_internal_rebuild(run_dir)
    assert probe["manifest_seal"] == "available"
    assert probe["line_count_equal"] is True
    assert probe["bytes_equal"] is False
    assert probe["rebuilt_sha256"] != probe["published_sha256"]
    assert probe["differing_lines"] == 1
    assert probe["differing_lines_all_generator_footer"] is True


def test_t5_acceptance_package_internal_missing_files(tmp_path: Path) -> None:
    """缺包内文件 ⇒ 诊断项显式失败（不静默当成相等）。"""
    row = driver._package_internal_rebuild(tmp_path)
    assert row["passed"] is False
    assert "缺" in row["detail"]


def test_t5_acceptance_documentation_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """文档检查：登记版本与七条判据；缺文件必须判失败（不回退为通过）。"""
    payload = driver.documentation("")
    assert payload["project_version"] == driver.generate._project_version()
    assert [item["name"] for item in payload["checks"]] == [
        "files_present",
        "version_read",
        "changelog_has_version",
        "plan_has_t5_record",
        "readme_mentions_p6",
        "claude_has_p6_contract",
        "closure_report_present",
        "plan_and_closure_name_this_run",
    ]
    assert set(payload["files"]) == set(driver.DOC_FILES)
    assert json.dumps(payload, ensure_ascii=False, allow_nan=False)
    rows = {item["name"]: item["passed"] for item in payload["checks"]}
    # 未传轮次 ⇒ 该判据必须失败，其余结构判据在真实文档上通过
    assert rows["plan_and_closure_name_this_run"] is False
    assert all(value for name, value in rows.items() if name != "plan_and_closure_name_this_run")
    monkeypatch.setattr(driver, "DOC_FILES", ("missing_for_unit_test.md",))
    broken = driver.documentation("")
    assert broken["files"]["missing_for_unit_test.md"]["exists"] is False
    assert broken["files"]["missing_for_unit_test.md"]["sha256_at_run_time"] is None
    assert broken["passed"] is False
    assert "files_present" in [item["name"] for item in broken["checks"] if not item["passed"]]


def test_t5_acceptance_documentation_requires_run_id_in_docs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """计划与闭合报告必须写明本次证据轮次（防“文档引用别的轮次/别的数字”）。"""
    version = driver.generate._project_version()
    (tmp_path / "docs" / "plans" / "v2").mkdir(parents=True)
    (tmp_path / "docs" / "reports").mkdir(parents=True)
    (tmp_path / "CHANGELOG.md").write_text(f"## [{version}]\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("P6 使用文档\n", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("P6 报告系统 T5 契约\n", encoding="utf-8")
    (tmp_path / driver.PLAN_DOC).write_text("## 24. T5 记录：run_abc\n", encoding="utf-8")
    (tmp_path / driver.CLOSURE_DOC).write_text("最终证据：run_abc\n", encoding="utf-8")
    monkeypatch.setattr(driver, "ROOT", tmp_path)

    ok = driver.documentation("run_abc")
    assert ok["passed"] is True, [item for item in ok["checks"] if not item["passed"]]
    other = driver.documentation("run_other")
    assert other["passed"] is False
    assert [item["name"] for item in other["checks"] if not item["passed"]] == [
        "plan_and_closure_name_this_run"
    ]


def test_t5_acceptance_clean_checks_require_explicit_pass() -> None:
    """空审计不得因为“列表全空”被当成干净（缺证据 ≠ 无差异）。"""
    record = {"exit_code": 0, "verdict": "pass", "audit": {}, "driver_audit": {}}
    rows = {item["name"]: item["passed"] for item in driver._audit_checks(record)}
    assert rows["product_audit_clean"] is False
    assert rows["driver_audit_clean"] is False
    assert rows["inputs_unchanged"] is False


def test_t5_acceptance_complexity_uses_single_paths_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--paths`` 是 ``nargs='+'``：多次出现只保留最后一次，双口径必须一次给全。"""
    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> dict[str, Any]:
        calls.append(list(command))
        return {
            "command": command,
            "exit_code": 0,
            "stdout": "[hard] 存量基线 1 条\n新增硬超限 0 条，存量基线 1 条（登记不阻塞），豁免 0 条\n",
            "stderr": "",
            "stderr_tail": [],
        }

    monkeypatch.setattr(driver, "_run", fake_run)
    payload = driver.run_complexity()
    assert payload["passed"] is True
    assert len(calls) == 2
    assert all(command.count("--paths") == 1 for command in calls)
    tails = []
    for command in calls:
        index = command.index("--paths") + 1
        tails.append(command[index:])
    assert tails == [["src/lazybull"], ["src/lazybull", "scripts"]]


def test_t5_acceptance_within_rejects_prefix_confusion() -> None:
    """允许前缀必须是“精确等值或后跟分隔符”（``run4`` 不得放行 ``run40``）。"""
    assert driver._within("data/reports/p6/x/run4", "data/reports/p6/x/run4/")
    assert driver._within("data/reports/p6/x/run4/a.json", "data/reports/p6/x/run4")
    assert not driver._within("data/reports/p6/x/run40/a.json", "data/reports/p6/x/run4")
    assert not driver._within("data/reports/p6/x/run4_audit/a.json", "data/reports/p6/x/run4")


def test_t5_acceptance_static_network_scan_is_clean() -> None:
    """报告包内不联网、不起子进程（守卫窗口外的静态证据）。"""
    scan = driver._static_network_scan()
    assert scan["files"] > 0
    assert scan["hits"] == []
    assert scan["passed"] is True


def test_t5_acceptance_golden_source_has_no_skip_marks() -> None:
    """真实 golden 核验器源码不得出现 pytest/skipif/xfail（判定不靠命令行自证）。"""
    assert driver._source_skips(driver.GOLDEN_SCRIPT) == []


def test_t5_acceptance_known_limitations_shape() -> None:
    """已知限制：编号唯一、L1 为 Linux（2026-11-08 起为 waived_by_user，与计划 §24.4 同编号）。"""
    ids = [item["id"] for item in driver.KNOWN_LIMITATIONS]
    assert ids == [f"L{index}" for index in range(1, len(ids) + 1)]
    first = driver.KNOWN_LIMITATIONS[0]
    assert first["id"] == "L1" and first["status"] == "waived_by_user"
    assert "Linux" in first["item"]
    assert all(
        item["status"]
        in {"not_achieved", "waived_by_user", "by_design"}
        | {
            "not_measured",
            "not_covered",
            "not_triggered",
            "not_implemented",
        }
        for item in driver.KNOWN_LIMITATIONS
    )


def test_t5_acceptance_utc_stamp_shape() -> None:
    """真实 smoke 的墙钟 stamp 必须是 ``YYYYMMDDTHHMMSSZ``（报告 id 依赖它分轮次）。"""
    assert re.fullmatch(r"\d{8}T\d{6}Z", driver._utc_stamp())
