# -*- coding: utf-8 -*-
"""P6 T1 验收驱动（P6-M1 可复跑证据包）。

一次运行产出 ``data/reports/p6/validations/research/<validation_id>/t1_acceptance.json``：

1. T1 六组 pytest 合成夹具（P6-T1-01~17 全场景；禁用字节码与 pytest 缓存）；
2. 复杂度 hook 双口径（``src/lazybull`` 与 ``src/lazybull`` + ``scripts``）；
3. 报告包内 Playwright 零 import 守卫（生产包硬边界）；
4. 汇总 verdict（任一检查失败即 fail，保留原始子证据路径）。

本驱动只读源码与测试，不写生产根（``data/paper``/``data/runs``/``data/models``），
输出只落本次新 validation 目录；已存在的 validation 目录一律拒绝（无覆盖）。
"""

# 复杂度豁免：阶段专用验收驱动（scripts/v2_p6，不入生产链路，见 CLAUDE.md §7.5）

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）
    sys.dont_write_bytecode = True
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import validation  # noqa: E402

#: T1 七组测试文件（P6-T1-01~17 + 整改负例；合成夹具，不读真实数据）
T1_TEST_FILES: tuple[str, ...] = (
    "tests/test_v2_p6_t1_snapshot.py",
    "tests/test_v2_p6_t1_authority.py",
    "tests/test_v2_p6_t1_instructions.py",
    "tests/test_v2_p6_t1_config_model.py",
    "tests/test_v2_p6_t1_sources.py",
    "tests/test_v2_p6_t1_manifest.py",
    "tests/test_v2_p6_t1_negatives.py",
)
#: 治理面定向集（与计划 §10 口径一致：T1 + T0.5 + 两个棘轮）
GOVERNANCE_TEST_FILES: tuple[str, ...] = (
    "tests/test_v2_p6_t05_corpus.py",
    "tests/test_v2_p6_t05_deps.py",
    "tests/test_v2_p6_t05_isolation.py",
    "tests/test_v2_p6_t05_perf_gate.py",
    "tests/test_check_inuse_closure.py",
    "tests/test_check_complexity.py",
)
#: 最少通过数（低于此值即视为“测试未真的跑”）
EXPECTED_MIN_PASSED = 70
#: 报告包根（禁止 import Playwright 的硬边界）
REPORTING_PACKAGE = Path("src/lazybull/v2/reporting")
BASELINE = Path("scripts/check/complexity_baseline.json")
EXIT_CODE_CONVENTION = {"pass": 0, "fail": 3, "setup_error": 2}


class AcceptanceSetupError(RuntimeError):
    """验收驱动前置条件不满足（输出目录形态非法/已存在）。"""


def _default_validation_id() -> str:
    """默认 validation_id（全小写数字，生成即过冻结 pattern）。"""
    return "p6_t1_acceptance_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _assert_new_validation_out(path: Path) -> Path:
    """只允许 ``data/reports/p6/validations/<evidence_class>/<id>`` 且必须不存在。"""
    target = Path(path).resolve()
    expected_root = (ROOT / "data/reports/p6/validations").resolve()
    try:
        relative = target.relative_to(expected_root)
    except ValueError as exc:
        raise AcceptanceSetupError(f"输出目录不在 validation 根下：{target}") from exc
    parts = relative.parts
    if len(parts) != 2 or parts[0] not in ("production", "research", "calibration_only"):
        raise AcceptanceSetupError(f"输出目录形态非法（须 <evidence_class>/<validation_id>）：{relative}")
    validation.validate_validation_id(parts[1])
    if target.exists():
        raise AcceptanceSetupError(f"validation 目录已存在，拒绝覆盖：{target}")
    return target


def _run(command: Sequence[str], *, env_extra: dict[str, str] | None = None) -> dict[str, Any]:
    """执行子进程并记录实际命令/退出码/首尾输出（不吞错）。"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.update(env_extra or {})
    completed = subprocess.run(
        list(command),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        check=False,
    )
    tail = [line for line in completed.stdout.splitlines() if line.strip()][-3:]
    return {
        "command": list(command),
        "exit_code": completed.returncode,
        "stdout_tail": tail,
        "stderr_tail": completed.stderr.splitlines()[-3:] if completed.stderr else [],
    }


def run_tests(with_governance: bool = True) -> dict[str, Any]:
    """运行 T1 七组 + 治理面定向集（禁用字节码与 pytest 缓存，避免污染工作树）。"""
    files = list(T1_TEST_FILES) + (list(GOVERNANCE_TEST_FILES) if with_governance else [])
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *files,
    ]
    result = _run(command)
    counts = _parse_pytest_counts(result["stdout_tail"])
    minimum = EXPECTED_MIN_PASSED if with_governance else 0
    result.update(
        {
            "test_files": files,
            "with_governance": with_governance,
            "counts": counts,
            "expected_min_passed": minimum,
            "passed": (
                result["exit_code"] == 0
                and counts["failed"] == 0
                and counts["skipped"] == 0
                and counts["passed"] >= minimum
            ),
        }
    )
    return result


def _parse_pytest_counts(lines: Sequence[str]) -> dict[str, int]:
    """解析 pytest 汇总行 ``N passed, M skipped``（缺失字段记 0）。"""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for line in lines:
        for key in counts:
            match = re.search(rf"(\d+)\s+{key}", line)
            if match:
                counts[key] = int(match.group(1))
    return counts


def run_complexity() -> dict[str, Any]:
    """复杂度 hook 双口径（均须 exit=0、新增硬超限 0、且 mccabe 确实运行过）。"""
    src_only = _run(
        [sys.executable, "scripts/check/check_complexity.py", "--paths", "src/lazybull",
         "--baseline", str(BASELINE)]
    )
    with_scripts = _run(
        [sys.executable, "scripts/check/check_complexity.py", "--paths", "src/lazybull",
         "scripts", "--baseline", str(BASELINE)]
    )
    summary = {}
    for name, result in (("src_only", src_only), ("with_scripts", with_scripts)):
        line = _find_summary_line(result["stdout_tail"])
        soft = _extract_int(line, "软超限")
        summary[name] = {
            **result,
            "summary_line": line,
            "new_hard": _extract_int(line, "新增硬超限"),
            "soft": soft,
            "baseline_kept": _extract_int(line, "存量基线"),
            # 正控制：软超限 > 0 说明 flake8/mccabe 真的跑过（否则“0 硬超限”不可区分）
            "mccabe_ran": bool(soft),
            "degraded": any("[warn]" in item for item in result["stderr_tail"]),
        }
    passed = all(
        item["exit_code"] == 0
        and item["new_hard"] == 0
        and item["mccabe_ran"]
        and not item["degraded"]
        for item in summary.values()
    )
    return {"suites": summary, "passed": passed}


def _find_summary_line(lines: Sequence[str]) -> str:
    """取含“检查文件”的汇总行（不依赖“必须是最后一行”）。"""
    for line in reversed(list(lines)):
        if "检查文件" in line:
            return line
    return lines[-1] if lines else ""


def _extract_int(line: str, label: str) -> int | None:
    """从中文汇总行提取 ``<label>[：: ]N`` 的 N（分隔符容忍全/半角，缺失返回 None）。"""
    match = re.search(rf"{re.escape(label)}\s*[:：]?\s*(\d+)", str(line))
    return int(match.group(1)) if match else None


def worktree_fingerprint() -> dict[str, Any]:
    """证据绑定代码状态：HEAD + **完整**工作树摘要 + 驱动自身字节 SHA（可回溯）。"""
    head = _git(["rev-parse", "HEAD"]).strip() or "<unavailable>"
    status = _git(["status", "--porcelain"])
    lines = [line for line in status.splitlines() if line.strip()]
    driver = Path(__file__)
    return {
        "head": head,
        "status_lines": len(lines),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty": bool(lines),
        "driver_sha256": validation.sha256_file(driver) if driver.is_file() else "<missing>",
        "driver_path": driver.relative_to(ROOT).as_posix(),
        "python": sys.version.split()[0],
    }


def _git(arguments: Sequence[str]) -> str:
    """执行 git 并返回**完整** stdout（不经 ``_run`` 的 tail 截断）。"""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return ""
    return completed.stdout or ""


def check_reporting_package_purity() -> dict[str, Any]:
    """生产包内不得 **import** Playwright（AST 判定；文档提及不算违规）。"""
    offenders: list[str] = []
    for path in sorted((ROOT / REPORTING_PACKAGE).glob("**/*.py")):
        imported = _imported_modules(path)
        if any(name.split(".")[0] == "playwright" for name in imported):
            offenders.append(path.relative_to(ROOT).as_posix())
    return {"passed": not offenders, "offenders": offenders}


def _imported_modules(path: Path) -> set[str]:
    """AST 收集模块的全部 import 目标（含函数体内惰性 import）。"""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def run(validation_id: str | None = None) -> dict[str, Any]:
    """执行全部检查并落证据包；返回汇总载荷。"""
    identifier = validation_id or _default_validation_id()
    out_dir = _assert_new_validation_out(
        ROOT / "data/reports/p6/validations/research" / identifier
    )
    validation.ensure_new_dir(out_dir)
    started = datetime.now(timezone.utc)
    tests = run_tests(with_governance=True)
    complexity = run_complexity()
    purity = check_reporting_package_purity()
    worktree = worktree_fingerprint()
    checks = {
        "pytest": tests["passed"],
        "complexity": complexity["passed"],
        "reporting_package_purity": purity["passed"],
    }
    payload = {
        "schema": "p6_t1_acceptance/v2",
        "validation_id": identifier,
        "generated_at": validation.utc_stamp(started),
        "verdict": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "pytest": tests,
        "complexity": complexity,
        "reporting_package_purity": purity,
        "worktree": worktree,
        "exit_code_convention": EXIT_CODE_CONVENTION,
    }
    validation.write_json_new(out_dir / "t1_acceptance.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口；退出码 = 0 pass / 3 fail / 2 setup_error。"""
    parser = argparse.ArgumentParser(description="P6 T1 验收驱动（合成夹具 + 复杂度 + 包纯度）")
    parser.add_argument("--validation-id", default=None, help="新 validation 目录名")
    parser.add_argument("--json", action="store_true", help="stdout 输出汇总 JSON")
    args = parser.parse_args(argv)
    try:
        payload = run(args.validation_id)
    except AcceptanceSetupError as exc:
        print(f"设置错误：{exc}", file=sys.stderr)
        return EXIT_CODE_CONVENTION["setup_error"]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"verdict={payload['verdict']} checks={payload['checks']}")
    return EXIT_CODE_CONVENTION["pass" if payload["verdict"] == "pass" else "fail"]


if __name__ == "__main__":
    raise SystemExit(main())
