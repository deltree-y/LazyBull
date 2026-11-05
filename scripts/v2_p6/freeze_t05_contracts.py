# -*- coding: utf-8 -*-
"""P6 T0.5 契约冻结与开工基线记录。

冻结内容（设计 §2.5/§5.3.1/§7.3/§8.1/§10）：compare 映射样例、canonical 字段、精确写白名单、
validation 根、依赖上下限/验收精确版本、语料规模。冻结件与 ``contracts.py`` 同源，测试逐项比对。

基线内容：git HEAD/工作树状态、``src+scripts`` 稳定内容摘要、复杂度 hook 双口径 exit code、
依赖安装版本，以及 **T0.5 开工前已存在的 P4 NAV 修复未提交文件清单**（不得混入 P6 收口提交）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）；pytest 导入时不改全局
    sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import contracts, deps, validation  # noqa: E402
from src.lazybull.v2.store.store_identity import (  # noqa: E402
    canonical_json_sha256,
    workspace_identity,
)

__all__ = ["build_contract_record", "collect_baseline", "main"]

FREEZE_RECORD_SCHEMA = "p6_t05_freeze_record/v1"
BASELINE_SCHEMA = "p6_t05_baseline/v1"

#: T0.5 开工前工作树中已存在的 P4 NAV 修复代码/测试（不含 P4 与 P6 共用修改面的三份元数据）
P4_UNCOMMITTED_CODE_PATHS: tuple[str, ...] = (
    "scripts/v2_p4/t5_gate1_identity.py",
    "scripts/v2_p4/t5_migration.py",
    "scripts/v2_p4/t5_nav_row_repair.py",
    "scripts/v2_p4/t5_three_arm.py",
    "src/lazybull/v2/hosts/paper/nav_valuation.py",
    "src/lazybull/v2/hosts/paper/state_store.py",
    "src/lazybull/v2/hosts/testbed/paths.py",
    "tests/fixtures/inuse_closure_baseline_20261006.json",
    "tests/test_check_inuse_closure.py",
    "tests/test_v2_p4_t2_state_store.py",
    "tests/test_v2_p4_t5_migration.py",
    "tests/test_v2_p4_t5_nav_row_repair.py",
    "tests/test_v2_p5b_t4_scripts_zero_write.py",
)
#: P4 与 P6 T0.5 共用的元数据文件（提交边界须人工拆分，不得整文件归入单侧）
SHARED_METADATA_PATHS: tuple[str, ...] = ("CHANGELOG.md", "README.md", "pyproject.toml")


def build_contract_record() -> dict[str, Any]:
    """构造冻结记录（与 ``contracts.as_frozen_payload()`` 同源）。"""
    payload = contracts.as_frozen_payload()
    return {
        "schema": FREEZE_RECORD_SCHEMA,
        "generated_at": validation.utc_stamp(),
        "contract": payload,
        "contract_sha256": canonical_json_sha256(payload),
        "source": {
            "architecture_reference": contracts.ARCHITECTURE_REFERENCE,
            "design": "docs/plans/v2/p6_report_design_20261104.md v6",
            "plan": "docs/plans/v2/p6_report_plan_20261104.md v6",
        },
    }


def _git_status(repo_root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"采集 git status 失败：{completed.stderr.strip()}")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _run_complexity(repo_root: Path, paths: Sequence[str]) -> dict[str, Any]:
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(repo_root / "scripts" / "check" / "check_complexity.py"),
        "--paths",
        *paths,
        "--baseline",
        "scripts/check/complexity_baseline.json",
    ]
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    return {
        "paths": list(paths),
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout.strip().splitlines()[-25:],
        "stderr_tail": completed.stderr.strip().splitlines()[-15:],
    }


def collect_baseline(repo_root: Path | str = ROOT) -> dict[str, Any]:
    """采集开工基线：工作树、稳定源码摘要、复杂度双口径与依赖版本。"""
    root = Path(repo_root).resolve()
    status = _git_status(root)
    status_paths = {line[3:].strip() for line in status if len(line) > 3}
    installed = {
        "jinja2": getattr(deps.load_jinja2(), "__version__", ""),
        "playwright": _installed_playwright(),
    }
    return {
        "schema": BASELINE_SCHEMA,
        "generated_at": validation.utc_stamp(),
        "workspace": workspace_identity(root),
        "git_status": status,
        "worktree_boundary": {
            "p4_uncommitted_code_paths": list(P4_UNCOMMITTED_CODE_PATHS),
            "p4_uncommitted_present": sorted(
                path for path in P4_UNCOMMITTED_CODE_PATHS if path in status_paths
            ),
            "shared_metadata_paths": list(SHARED_METADATA_PATHS),
            "note": "P4 NAV 修复工作树改动由用户决定提交边界；P6 T0.5 不修改其代码与测试",
        },
        "complexity": {
            "src_only": _run_complexity(root, ("src/lazybull",)),
            "src_and_scripts": _run_complexity(root, ("src/lazybull", "scripts")),
        },
        "dependency": {
            "declared_bounds": {
                name: dict(bounds) for name, bounds in contracts.DEPENDENCY_BOUNDS.items()
            },
            "acceptance_versions": dict(contracts.ACCEPTANCE_VERSIONS),
            "installed": installed,
        },
    }


def _installed_playwright() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("playwright")
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _assert_new_out(out: Path) -> Path:
    reports_root = (ROOT / contracts.REPORTS_ROOT).resolve()
    target = out.resolve()
    if target.exists():
        raise validation.ValidationError(f"输出目录已存在，拒绝覆盖：{target}")
    try:
        target.relative_to(reports_root / "validations")
    except ValueError as exc:
        raise validation.ValidationError(
            f"基线记录只允许写 {reports_root / 'validations'} 之下：{target}"
        ) from exc
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结 P6 T0.5 契约并记录开工基线")
    parser.add_argument(
        "--out", required=True, help="新的 validation 目录（位于 data/reports/p6/validations/）"
    )
    parser.add_argument("--docs-out", default=None, help="可选：写 docs 冻结快照（必须不存在）")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        target = _assert_new_out(Path(args.out))
        validation.ensure_new_dir(target)
        contract_record = build_contract_record()
        baseline = collect_baseline(ROOT)
        validation.write_json_new(target / "contract_freeze.json", contract_record)
        validation.write_json_new(target / "baseline.json", baseline)
        if args.docs_out:
            validation.write_json_new(Path(args.docs_out), contract_record)
    except (validation.ValidationError, RuntimeError) as exc:
        print(f"[P6-T0.5] 冻结/基线失败：{exc}", file=sys.stderr)
        return 1
    summary = {
        "contract_sha256": contract_record["contract_sha256"],
        "complexity_exit": {
            name: record["exit_code"] for name, record in baseline["complexity"].items()
        },
        "out": str(target),
    }
    print(json.dumps(summary, ensure_ascii=False) if args.json else f"冻结与基线已写入：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
