# -*- coding: utf-8 -*-
"""T1 引擎属性写点 AST 枚举工具（P4 规划 §5-T1 / D2：字段清单 = AST 枚举 + 人工复核）。

扫描 v2 内核全部模块，枚举引擎（self / state.py 的 engine 形参）与子对象
（PendingOrderManager / StopLossMonitor / Ledger）的属性写点：
- 属性重写绑定（``self.x = ...`` / ``self.x += ...`` / 注解赋值 / ``del self.x``）
- 属性项级改写（``self.x[...] = ...`` / ``del self.x[...]`` / ``self.x.append/update/pop`` 等方法调用）

输出：``{module:relpath, class, function, attr, kind, line}`` 逐条 + 按属性聚合的
汇总表（写点计数、所在函数），供人工复核逐字段标注 配置型/演化型/输入接线/日界暂态。

用法：``python scripts/v2_p4/t1_ast_enum.py``（只读扫描，标准输出出报告）。

复杂度豁免依据：CLAUDE.md §7.5（阶段专用验收工具按临时开发文件处理）。
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]

# 扫描范围：v2 内核 + common/rules（子对象实现）+ hosts/backtest（宿主写点）
_SCAN_ROOTS = [
    "src/lazybull/v2/core",
    "src/lazybull/v2/common/rules",
    "src/lazybull/v2/hosts",
]
# 模块级函数以 ``engine`` / ``self`` 形参指代引擎实例（state.py 的 init 族用
# ``engine``；buy_execution.py 等 D8 提取的模块级函数同样用 ``engine``）
_ENGINE_PARAM_MODULES = {"state.py"}  # 保留作语义注记；实际两名字全范围生效

_WRITE_METHODS = {
    "append",
    "extend",
    "insert",
    "update",
    "pop",
    "popitem",
    "clear",
    "setdefault",
    "add",
    "discard",
    "remove",
    "sort",
    "reverse",
}


def _attr_name(node: ast.AST) -> Optional[str]:
    """提取 ``self.xxx`` / ``engine.xxx`` 的属性名；非属性目标返回 None。"""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr
    return None


def _is_engine_base(base: ast.AST, engine_names: set) -> bool:
    return isinstance(base, ast.Name) and base.id in engine_names


def _scan_file(path: Path, rel: str) -> List[Dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    engine_names = {"self", "engine"}
    rows: List[Dict] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.scope: List[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def _record(self, target: ast.AST, kind: str, lineno: int) -> None:
            # 属性级：self.x / engine.x
            if isinstance(target, ast.Attribute) and _is_engine_base(target.value, engine_names):
                rows.append(
                    {
                        "module": rel,
                        "class": self.scope[-2] if len(self.scope) >= 2 else "",
                        "function": self.scope[-1] if self.scope else "",
                        "attr": target.attr,
                        "kind": kind,
                        "line": lineno,
                    }
                )
                return
            # 项级/方法级：self.x[...] / self.x.method(...)
            inner = target
            while isinstance(inner, (ast.Subscript,)):
                inner = inner.value
            if isinstance(inner, ast.Attribute) and _is_engine_base(inner.value, engine_names):
                rows.append(
                    {
                        "module": rel,
                        "class": self.scope[-2] if len(self.scope) >= 2 else "",
                        "function": self.scope[-1] if self.scope else "",
                        "attr": inner.attr,
                        "kind": kind,
                        "line": lineno,
                    }
                )

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Attribute):
                        self._record(sub, "assign", node.lineno)
                    elif isinstance(sub, ast.Subscript):
                        self._record(sub, "item_assign", node.lineno)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            target = node.target
            if isinstance(target, ast.Attribute):
                self._record(target, "aug_assign", node.lineno)
            elif isinstance(target, ast.Subscript):
                self._record(target, "item_aug_assign", node.lineno)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is not None:
                target = node.target
                if isinstance(target, ast.Attribute):
                    self._record(target, "ann_assign", node.lineno)
                elif isinstance(target, ast.Subscript):
                    self._record(target, "item_assign", node.lineno)
            self.generic_visit(node)

        def visit_Delete(self, node: ast.Delete) -> None:
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    self._record(target, "delete", node.lineno)
                elif isinstance(target, ast.Subscript):
                    self._record(target, "item_delete", node.lineno)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _WRITE_METHODS
                and isinstance(func.value, ast.Attribute)
                and _is_engine_base(func.value.value, engine_names)
            ):
                rows.append(
                    {
                        "module": rel,
                        "class": self.scope[-2] if len(self.scope) >= 2 else "",
                        "function": self.scope[-1] if self.scope else "",
                        "attr": func.value.attr,
                        "kind": f"method_call:{func.attr}",
                        "line": node.lineno,
                    }
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    return rows


def scan(repo_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for root in _SCAN_ROOTS:
        for path in sorted((repo_root / root).rglob("*.py")):
            rel = path.relative_to(repo_root).as_posix()
            rows.extend(_scan_file(path, rel))
    return rows


def summarize(rows: List[Dict]) -> Dict[str, Dict]:
    """按属性聚合：写点计数 + 所在模块/函数清单（人工复核底稿）。"""
    summary: Dict[str, Dict] = {}
    for row in rows:
        entry = summary.setdefault(
            row["attr"],
            {"count": 0, "kinds": set(), "sites": []},
        )
        entry["count"] += 1
        entry["kinds"].add(row["kind"])
        entry["sites"].append(f"{row['module']}:{row['line']}:{row['class']}.{row['function']}")
    return {
        attr: {
            "count": e["count"],
            "kinds": sorted(e["kinds"]),
            "sites": sorted(e["sites"]),
        }
        for attr, e in sorted(summary.items())
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v2 P4 T1 引擎属性写点 AST 枚举")
    parser.add_argument("--json-out", default=None, help="可选：JSON 明细输出路径")
    args = parser.parse_args(argv)
    rows = scan(_REPO_ROOT)
    summary = summarize(rows)
    print(f"写点总数: {len(rows)}；属性数: {len(summary)}")
    for attr, entry in summary.items():
        print(f"\n== {attr}（{entry['count']} 处，{','.join(entry['kinds'])}）")
        for site in entry["sites"]:
            print(f"   {site}")
    if args.json_out:
        payload = {"scan_roots": _SCAN_ROOTS, "rows": rows, "summary": summary}
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nJSON 明细已写: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
