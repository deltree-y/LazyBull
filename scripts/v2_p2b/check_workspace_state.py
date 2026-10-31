# -*- coding: utf-8 -*-
"""P2b 门 2：实际工作树修改面、稳定代码摘要与依赖方向检查。

覆盖已提交、已暂存、未暂存和未跟踪文件；代码摘要固定覆盖
``src/lazybull/**/*.py + scripts/**/*.py``，环境版本单独采集。
测试使用合成临时 Git 仓库，不修改真实工作区。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_PRODUCTION_BASE = "13ae1bf"
P2B_MODIFICATION_BASE = "eaebc30"
DIGEST_SCOPES = ("src/lazybull", "scripts")
OLD_PRODUCTION_PATHS = (
    "src/lazybull/backtest",
    "src/lazybull/trading",
    "src/lazybull/signals",
    "src/lazybull/portfolio",
    "src/lazybull/common",
    "src/lazybull/execution",
    "src/lazybull/universe",
    "src/lazybull/data",
    "src/lazybull/ml/model_registry.py",
    "src/lazybull/ml/train_core",
)
ALLOWED_CHANGED_PATHS = frozenset(
    {
        ".github/copilot-instructions.md",
        "AGENTS.md",
        "CHANGELOG.md",
        "CLAUDE.md",
        "README.md",
        "pyproject.toml",
        "docs/contracts/README.md",
        "docs/contracts/protocols.md",
        "docs/contracts/v2_architecture_plan.md",
        "docs/glossary/terms.yaml",
        "docs/plans/v2/mvp_ledger_t05_20261006.md",
        "docs/plans/v2/p2b_implementation_plan_20261016.md",
        "docs/plans/v2/backup/p2b_implementation_plan_v3_reviewed_20261016.md",
        "src/lazybull/v2/common/protocols/core.py",
        "src/lazybull/v2/common/types.py",
        "src/lazybull/v2/core/accounting/nav.py",
        "src/lazybull/v2/core/decision/portfolio_merger.py",
        "src/lazybull/v2/core/execution/engine.py",
        "src/lazybull/v2/core/execution/signal_execution.py",
        "src/lazybull/v2/hosts/native_engine.py",
        "src/lazybull/v2/hosts/backtest/replay.py",
        "src/lazybull/v2/hosts/backtest/runs_writer.py",
        "src/lazybull/v2/hosts/backtest/runtime.py",
        "scripts/v2_p2a/compare_replay_vs_b0.py",
        "scripts/v2_p2a/replay_b0.py",
        "tests/fixtures/inuse_closure_baseline_20261006.json",
        "tests/test_check_inuse_closure.py",
        "tests/test_ensure_and_t0_printing.py",
        "tests/test_replenishment_no_sell.py",
        "tests/test_v2_p2a_t5_equivalence.py",
    }
)
ALLOWED_CHANGED_PREFIXES = (
    "docs/review/v2_p2b/",
    "scripts/v2_p2b/",
    "tests/test_v2_p2b_",
)
ENVIRONMENT_PACKAGES = (
    "pandas",
    "numpy",
    "pyarrow",
    "xgboost",
    "scikit-learn",
    "scipy",
    "joblib",
)
VALUE_OBJECT_CONSTRUCTORS = frozenset(
    {
        "Account",
        "Lot",
        "Money",
        "Position",
        "Price",
        "Shares",
        "TargetPortfolio",
        "TradeDate",
        "TSCode",
        "Weight",
    }
)


def canonical_json(value: object) -> str:
    """生成跨臂可比的紧凑 canonical JSON。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _git_lines(repo_root: Path, args: Sequence[str]) -> List[str]:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git 命令失败（{list(args)}）: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _changes_since(repo_root: Path, base: str) -> Dict[str, List[str]]:
    commands = {
        "committed": ["diff", "--name-only", "--no-renames", f"{base}..HEAD", "--"],
        "staged": ["diff", "--name-only", "--no-renames", "--cached", "--"],
        "unstaged": ["diff", "--name-only", "--no-renames", "--"],
        "untracked": ["ls-files", "--others", "--exclude-standard", "--"],
    }
    changes: Dict[str, List[str]] = {}
    for state, args in commands.items():
        for path in _git_lines(repo_root, args):
            changes.setdefault(path.replace("\\", "/"), []).append(state)
    return changes


def _under_any(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def _allowed_change(path: str, extra_allowed: Sequence[str]) -> bool:
    return (
        path in ALLOWED_CHANGED_PATHS
        or path in extra_allowed
        or path.startswith(ALLOWED_CHANGED_PREFIXES)
    )


def _module_name(repo_root: Path, path: Path) -> str:
    relative = path.relative_to(repo_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _from_import_base(package: str, node: ast.ImportFrom) -> str:
    parts = [] if node.level == 0 else package.split(".")
    if node.level:
        parts = parts[: len(parts) - node.level + 1]
    if node.module:
        parts += node.module.split(".")
    return ".".join(parts)


def _from_import_names(package: str, node: ast.ImportFrom) -> List[str]:
    base = _from_import_base(package, node)
    names = [base] if base else []
    names += [f"{base}.{alias.name}" if base else alias.name for alias in node.names]
    return names


def _is_hosts_import(module: str) -> bool:
    return (
        module == "hosts"
        or module.startswith("hosts.")
        or module == "src.lazybull.v2.hosts"
        or module.startswith("src.lazybull.v2.hosts.")
    )


def _symbol_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _dynamic_import_names(package: str, nodes: List[ast.AST]) -> set:
    """收集 __import__ 与 importlib.import_module 的直接导入名/别名。"""
    names = {"__import__"}
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and _from_import_base(package, node) == "importlib":
            names |= {
                alias.asname or alias.name for alias in node.names if alias.name == "import_module"
            }
    return names


def _import_violations(repo_root: Path, path: Path, tree: ast.AST) -> List[str]:
    module = _module_name(repo_root, path)
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    nodes = list(ast.walk(tree))
    dynamic_names = _dynamic_import_names(package, nodes)
    imported: List[str] = []
    for node in nodes:
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += _from_import_names(package, node)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            is_dynamic = (
                isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
            ) or (isinstance(node.func, ast.Name) and node.func.id in dynamic_names)
            if is_dynamic and isinstance(node.args[0].value, str):
                imported.append(node.args[0].value)
    relative = path.relative_to(repo_root).as_posix()
    return [
        f"{relative}: core 禁止 import hosts（{name}）"
        for name in imported
        if _is_hosts_import(name)
    ]


def _assignment_parts(node: ast.AST) -> tuple[Optional[ast.AST], List[ast.AST]]:
    """统一提取 Assign/AnnAssign 的值和目标；非赋值或声明无值返回空。"""
    if isinstance(node, ast.Assign):
        return node.value, list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return node.value, [node.target]
    return None, []


def _constructor_violations(repo_root: Path, path: Path, tree: ast.AST) -> List[str]:
    module = _module_name(repo_root, path)
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    local_names = set(VALUE_OBJECT_CONSTRUCTORS)
    nodes = list(ast.walk(tree))
    for node in nodes:
        if (
            isinstance(node, ast.ImportFrom)
            and _from_import_base(package, node) == "src.lazybull.v2.common.types"
        ):
            local_names |= {
                alias.asname or alias.name
                for alias in node.names
                if alias.name in VALUE_OBJECT_CONSTRUCTORS
            }
    for node in nodes:
        value, targets = _assignment_parts(node)
        if value is not None and _symbol_name(value) in local_names:
            local_names |= {name for target in targets if (name := _symbol_name(target))}
    relative = path.relative_to(repo_root).as_posix()
    violations = []
    for node in nodes:
        if isinstance(node, ast.Call) and _symbol_name(node.func) in local_names:
            name = _symbol_name(node.func)
            violations.append(
                f"{relative}:{node.lineno}: 值对象构造只能位于宿主 native_engine（{name}）"
            )
    return violations


def _contains_any_name(node: ast.AST, names: set) -> bool:
    return any(_symbol_name(item) in names for item in ast.walk(node))


def _nav_division_violations(
    repo_root: Path,
    path: Path,
    tree: ast.AST,
    exempt_functions: frozenset[str] = frozenset(),
) -> List[str]:
    capital_names = {"initial_capital"}
    nodes = list(ast.walk(tree))
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}

    def _in_exempt_function(node: ast.AST) -> bool:
        current = node
        while current in parents:
            current = parents[current]
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if current.name in exempt_functions:
                    return True
        return False

    for node in nodes:
        value, targets = _assignment_parts(node)
        if value is not None and _contains_any_name(value, capital_names):
            capital_names |= {name for target in targets if (name := _symbol_name(target))}
    relative = path.relative_to(repo_root).as_posix()
    violations = []
    for node in nodes:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if not _in_exempt_function(node) and _contains_any_name(node.right, capital_names):
                violations.append(
                    f"{relative}:{node.lineno}: 宿主不得自行除以 initial_capital 派生 NAV"
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                not _in_exempt_function(node)
                and node.func.attr == "__truediv__"
                and any(_contains_any_name(arg, capital_names) for arg in node.args)
            ):
                violations.append(f"{relative}:{node.lineno}: 宿主不得通过 __truediv__ 私算 NAV")
    return violations


def check_ast_boundaries(repo_root: Path = PROJECT_ROOT) -> List[str]:
    """检查 core→hosts 依赖、值对象转换归属与宿主 NAV 私算。"""
    violations: List[str] = []
    v2_root = repo_root / "src" / "lazybull" / "v2"
    if not v2_root.exists():
        return violations
    value_object_boundaries = {
        v2_root / "hosts" / "native_engine.py",
        v2_root / "hosts" / "paper" / "host.py",
        v2_root / "train" / "data_adapter.py",
        v2_root / "common" / "types.py",
    }
    # D3 年化唯一实现 / F15 展示快照两个既有公式；旧壳删除时复审收缩。
    nav_division_exemptions = {
        v2_root / "hosts" / "paper" / "performance.py": frozenset({"calculate_annualized_return"}),
        v2_root / "hosts" / "paper" / "reporting.py": frozenset({"load_position_snapshot"}),
    }
    for path in sorted(v2_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if (v2_root / "core") in path.parents:
            violations += _import_violations(repo_root, path, tree)
        if path not in value_object_boundaries:
            violations += _constructor_violations(repo_root, path, tree)
        if (v2_root / "hosts") in path.parents:
            violations += _nav_division_violations(
                repo_root,
                path,
                tree,
                nav_division_exemptions.get(path, frozenset()),
            )
    return violations


def collect_workspace_digest(repo_root: Path = PROJECT_ROOT) -> Dict[str, object]:
    """采集稳定 src+scripts 文件 SHA256、删除清单与总 digest。

    ``deleted_files`` 记录当前索引/工作树删除；已提交删除不重复列名，而是由
    当前文件集合缺项体现，因此同样改变总 digest（S3-T3-03 语义边界）。
    """
    repo_root = Path(repo_root)
    files: Dict[str, str] = {}
    for scope in DIGEST_SCOPES:
        root = repo_root / scope
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.is_file():
                relative = path.relative_to(repo_root).as_posix()
                files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    deleted = set(_git_lines(repo_root, ["ls-files", "--deleted", "--"]))
    deleted |= set(
        _git_lines(repo_root, ["diff", "--cached", "--name-only", "--diff-filter=D", "--"])
    )
    deleted_files = sorted(
        path.replace("\\", "/")
        for path in deleted
        if _under_any(path.replace("\\", "/"), DIGEST_SCOPES)
    )
    payload = {"scope": list(DIGEST_SCOPES), "files": files, "deleted_files": deleted_files}
    return {
        **payload,
        "digest": hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
    }


def validate_workspace_digest(document: Dict[str, object]) -> bool:
    """复核固定 scope、逐文件哈希、删除清单及总 digest 的结构与自洽性。"""
    if not isinstance(document, dict) or document.get("scope") != list(DIGEST_SCOPES):
        return False
    files = document.get("files")
    deleted = document.get("deleted_files")
    if not isinstance(files, dict) or not isinstance(deleted, list):
        return False
    paths = list(files) + deleted
    if any(not isinstance(path, str) or not _under_any(path, DIGEST_SCOPES) for path in paths):
        return False
    if set(files) & set(deleted):
        return False
    for value in files.values():
        if not isinstance(value, str) or len(value) != 64:
            return False
        if any(char not in "0123456789abcdef" for char in value):
            return False
    payload = {
        "scope": document.get("scope"),
        "files": files,
        "deleted_files": deleted,
    }
    expected = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return document.get("digest") == expected


def collect_environment_versions() -> Dict[str, object]:
    """单列 Python 与关键第三方包版本，不混入 workspace.digest。"""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": {name: metadata.version(name) for name in ENVIRONMENT_PACKAGES},
    }


def check_workspace_state(
    repo_root: Path = PROJECT_ROOT,
    *,
    old_production_base: str = OLD_PRODUCTION_BASE,
    modification_base: str = P2B_MODIFICATION_BASE,
    extra_allowed: Sequence[str] = (),
) -> Dict[str, object]:
    """执行修改面、依赖方向、代码摘要与环境证据采集。"""
    repo_root = Path(repo_root)
    production_changes = _changes_since(repo_root, old_production_base)
    surface_changes = _changes_since(repo_root, modification_base)
    production_violations = sorted(
        f"{path}（{','.join(states)}）"
        for path, states in production_changes.items()
        if _under_any(path, OLD_PRODUCTION_PATHS)
    )
    surface_violations = sorted(
        f"{path}（{','.join(states)}）"
        for path, states in surface_changes.items()
        if not _allowed_change(path, extra_allowed)
    )
    ast_violations = check_ast_boundaries(repo_root)
    violations = production_violations + surface_violations + ast_violations
    return {
        "ok": not violations,
        "violations": violations,
        "old_production_violations": production_violations,
        "modification_surface_violations": surface_violations,
        "ast_violations": ast_violations,
        "changes": surface_changes,
        "workspace": collect_workspace_digest(repo_root),
        "environment": collect_environment_versions(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = check_workspace_state(args.repo_root)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for violation in report["violations"]:
        logger.error(violation)
    logger.info(
        f"workspace.digest={report['workspace']['digest']}；"
        f"文件数={len(report['workspace']['files'])}；ok={report['ok']}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
