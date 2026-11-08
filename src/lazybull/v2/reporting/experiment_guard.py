# -*- coding: utf-8 -*-
"""实验生成链的 bootstrap 拦截（AST 扫描 + 运行期哨兵，契约 §T3-⑪）。

两层防线：

1. **静态扫描**：``scan_reporting_package`` 对报告包逐文件做 AST 检查，覆盖
   ``ImportFrom`` 别名（``import bootstrap_delta as bd``）、``Name``/``Attribute`` 字面引用、
   ``getattr/import_module/__import__`` 的常量名与 ``__dict__[...]`` 下标常量；
2. **运行期哨兵**：``forbid_bootstrap_calls`` 在生成链执行期间把
   ``scripts.compare.fold_subset`` 的两个 bootstrap 函数替换为抛错桩（进入即失败），
   结束后恢复原函数——即使静态扫描被别名绕过，实际调用仍 fail-closed。

本模块不 import Playwright，也不写盘；只做拦截与证据。
"""

from __future__ import annotations

import ast
import contextlib
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Sequence

__all__ = [
    "BootstrapForbiddenError",
    "FORBIDDEN_NAMES",
    "assert_generation_chain",
    "forbid_bootstrap_calls",
    "scan_paths",
    "scan_source",
]

#: 生成链禁止调用的既有函数（唯一闭集）
FORBIDDEN_NAMES: tuple[str, ...] = ("bootstrap_delta", "bootstrap_table")
#: 动态取用/导入函数名（其常量参数命中禁用名同样视为绕过）
_DYNAMIC_ACCESSORS: frozenset[str] = frozenset(
    {"getattr", "setattr", "import_module", "__import__", "get"}
)


class BootstrapForbiddenError(RuntimeError):
    """生成链尝试访问/调用被禁的 bootstrap 函数（fail-closed，不降级）。"""


def _static_value(node: ast.AST | None) -> Any:
    """常量折叠：``Constant`` 与**简单字符串拼接**（``BinOp(Add)``）取最终值。"""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_value(node.left)
        right = _static_value(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    return None


def _forbidden_hit(text: Any) -> str | None:
    """字符串（含简单拼接后的最终值）是否包含禁用名。"""
    if not isinstance(text, str):
        return None
    return next((name for name in FORBIDDEN_NAMES if name in text), None)


def _import_hits(node: ast.AST, label: str) -> list[str]:
    """``import`` / ``from ... import ...``（含别名）命中。"""
    hits: list[str] = []
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name in FORBIDDEN_NAMES:
                hits.append(f"{label}:{node.lineno}:from-import {alias.name}")
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name in FORBIDDEN_NAMES:
                hits.append(f"{label}:{node.lineno}:import {alias.name}")
    return hits


def _reference_hits(node: ast.AST, label: str) -> list[str]:
    """字面 ``Name`` / ``Attribute`` 命中。"""
    if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
        return [f"{label}:{node.lineno}:name {node.id}"]
    if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
        return [f"{label}:{node.lineno}:attr {node.attr}"]
    return []


def _accessor_name(func: ast.AST) -> str | None:
    """调用目标名（``getattr`` / ``importlib.import_module`` 一类）。"""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _dynamic_hits(node: ast.AST, label: str) -> list[str]:
    """``getattr(x, 'bootstrap_delta')`` / ``__dict__['...']`` 一类动态取用命中。"""
    if isinstance(node, ast.Subscript):
        matched = _forbidden_hit(_static_value(node.slice))
        return [f"{label}:{node.lineno}:subscript {matched}"] if matched else []
    if not isinstance(node, ast.Call):
        return []
    name = _accessor_name(node.func)
    if name not in _DYNAMIC_ACCESSORS:
        return []
    hits: list[str] = []
    for argument in node.args:
        matched = _forbidden_hit(_static_value(argument))
        if matched:
            hits.append(f"{label}:{node.lineno}:{name}({matched})")
    return hits


def scan_source(source: str, *, label: str = "<memory>") -> list[str]:
    """扫描单个源码文本；返回命中列表（空 = 未发现被禁访问）。"""
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        hits.extend(_import_hits(node, label))
        hits.extend(_reference_hits(node, label))
        hits.extend(_dynamic_hits(node, label))
    return hits


def scan_paths(paths: Sequence[Path]) -> dict[str, Any]:
    """扫描多个源码文件；返回 ``{scanned, hits}`` 证据。"""
    hits: list[str] = []
    scanned = 0
    for path in paths:
        scanned += 1
        try:
            source = Path(path).read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - 目录内文件读取失败极罕见
            hits.append(f"{Path(path).name}:unreadable {type(exc).__name__}")
            continue
        hits.extend(scan_source(source, label=Path(path).name))
    return {"scanned": scanned, "hits": hits}


def assert_generation_chain(
    reporting_root: Path | str | None = None, extra_paths: Sequence[Path] = ()
) -> dict[str, Any]:
    """静态扫描报告包（可选加扫生产入口）；任何命中即抛 ``BootstrapForbiddenError``。

    扫描域默认 = ``src/lazybull/v2/reporting/*.py``；``extra_paths`` 用于把生产入口
    （``scripts/p6_report.py``）纳入，避免“入口不在扫描域”的缺口（B-R2-01）。
    """
    root = Path(reporting_root or Path(__file__).resolve().parent)
    paths = sorted(root.glob("*.py")) + [Path(item) for item in extra_paths if Path(item).is_file()]
    report = scan_paths(paths)
    report["scope"] = {
        "reporting_root": str(root),
        "extra_paths": [str(Path(item)) for item in extra_paths],
    }
    if report["hits"]:
        raise BootstrapForbiddenError(
            "报告包出现被禁的 bootstrap 访问（生成链拦截）：" + "; ".join(report["hits"][:5])
        )
    return report


@contextlib.contextmanager
def forbid_bootstrap_calls(module: ModuleType | None = None) -> Iterator[dict[str, Any]]:
    """生成链执行期哨兵：被禁函数一旦被调用即抛错，退出后恢复原函数。"""
    if module is None:
        try:
            from scripts.compare import fold_subset as module  # 局部导入，避免包级循环
        except ImportError as exc:
            # 无 scripts/ 的安装环境：运行期层降级为“不可安装”，静态扫描仍是硬门
            yield {
                "guarded": [],
                "module": None,
                "reason": "not_importable",
                "detail": f"{type(exc).__name__}: {exc}",
            }
            return
    originals: dict[str, Any] = {}
    evidence: dict[str, Any] = {"guarded": [], "module": getattr(module, "__name__", "<unknown>")}

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise BootstrapForbiddenError("生成链禁止调用 bootstrap_delta/bootstrap_table")

    for name in FORBIDDEN_NAMES:
        if hasattr(module, name):
            originals[name] = getattr(module, name)
            setattr(module, name, _boom)
            evidence["guarded"].append(name)
    try:
        yield evidence
    finally:
        for name, value in originals.items():
            setattr(module, name, value)
