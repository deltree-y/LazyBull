# -*- coding: utf-8 -*-
"""在用全域闭包检查脚本（v2 方案 §4.3「在用全域」可验证机制的工具化承载，F10R-4）。

以两条生产主链路入口为起点，ast 递归解析全部 import（含函数体 / 条件块内惰性
import），得到静态可达的 ``src.lazybull.*`` 模块闭包；再并入 ``factors/risk/
factor_registry.py`` 经 importlib 动态加载的模块（静态分析不可见的运行时调用点），
输出**在用全域**清单与数量守恒校验。

入口（方案 §4.3 v1.11-1B 口径）：
- 链路 1（WF 产线）：scripts/walk_forward.py + scripts/compare_walk_forward.py
- 链路 2（纸面）：scripts/paper_trade.py

用途：P2a 迁移期间的重跑监控——模块搬家 / 删除后重跑本脚本，比对在用清单漂移。

用法：
    python scripts/check/check_inuse_closure.py               # 摘要 + 守恒校验
    python scripts/check/check_inuse_closure.py --json-out out.json
    python scripts/check/check_inuse_closure.py --entry scripts/walk_forward.py [--entry ...]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 默认入口 = 方案两条主链路
DEFAULT_ENTRIES = [
    "scripts/walk_forward.py",
    "scripts/compare_walk_forward.py",
    "scripts/paper_trade.py",
]

# factor_registry._ensure_all_modules_imported 的 importlib 动态加载清单
# （factors/risk/factor_registry.py；静态闭包已含 volatility_factors，此处全列、自动去重）
DYNAMIC_RISK_MODULES = [
    "src.lazybull.factors.risk.downside_factors",
    "src.lazybull.factors.risk.volatility_factors",
    "src.lazybull.factors.risk.liquidity_factors",
    "src.lazybull.factors.risk.announcement_factors",
    "src.lazybull.factors.risk.derived_factors",
]


def _module_to_path(name: str) -> Optional[Path]:
    """点分模块名 → 仓库内文件路径；仓库外返回 None。

    大小写精确匹配（Windows 文件系统大小写不敏感，防止 ``data.Storage`` 误匹配
    ``storage.py``）。
    """
    if name.startswith("src.lazybull."):
        base = PROJECT_ROOT / "src" / "lazybull" / Path(*name[len("src.lazybull."):].split("."))
    elif name.startswith("scripts."):
        base = PROJECT_ROOT / Path(*name.split("."))
    else:
        return None
    p = base.with_suffix(".py")
    if p.exists() and p.name in (f.name for f in p.parent.iterdir()):
        return p
    if (base / "__init__.py").exists():
        return base / "__init__.py"
    return None


def _try_submodule(pkg: str, name: str) -> Optional[str]:
    candidate = f"{pkg}.{name}"
    return candidate if _module_to_path(candidate) is not None else None


def _collect_imports(tree: ast.AST, module_name: str, is_init: bool) -> List[str]:
    """收集一棵语法树里全部 import 的目标模块名（仓库内模块）。"""
    targets: List[str] = []
    for node in ast.walk(tree):
        sites: List[str] = []
        if isinstance(node, ast.Import):
            sites = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 相对导入：普通模块砍 level 层、__init__ 砍 level-1 层
                parts = module_name.split(".")
                cut = node.level - 1 if is_init else node.level
                pkg = ".".join(parts[: max(len(parts) - cut, 0)])
                if node.module:
                    pkg = f"{pkg}.{node.module}" if pkg else node.module
                sites = [_try_submodule(pkg, a.name) or pkg for a in node.names]
            else:
                pkg = node.module or ""
                sites = [_try_submodule(pkg, a.name) or pkg for a in node.names]
        for mod in sites:
            if mod.startswith("src.lazybull") or mod.startswith("scripts."):
                targets.append(mod)
    return targets


def compute_static_closure(entry_paths: List[Path]) -> Dict[str, str]:
    """从入口文件出发计算静态 import 闭包，返回 {模块名: 首个引用者}。"""
    discovered: Dict[str, str] = {}
    queue: List[Tuple[str, Path]] = []
    seen: Set[str] = set()
    for entry in entry_paths:
        mod = entry.relative_to(PROJECT_ROOT).as_posix()[:-3].replace("/", ".")
        discovered.setdefault(mod, "<entry>")
        queue.append((mod, entry))
        seen.add(mod)
    while queue:
        mod, path = queue.pop(0)
        if not path.exists():
            print(f"[WARN] 入口不存在: {path}", file=sys.stderr)
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for target in _collect_imports(tree, mod, path.name == "__init__.py"):
            discovered.setdefault(target, mod)
            p = _module_to_path(target)
            if p is not None and target not in seen:
                seen.add(target)
                queue.append((target, p))
    return discovered


def list_all_lazybull_modules() -> Set[str]:
    """全仓 src/lazybull 模块全集（``src.`` 前缀口径）。"""
    all_mods: Set[str] = set()
    for p in (PROJECT_ROOT / "src" / "lazybull").rglob("*.py"):
        rel = p.relative_to(PROJECT_ROOT / "src").with_suffix("").as_posix().replace("/", ".")
        if p.name == "__init__.py":
            rel = rel[: -len(".__init__")]
        all_mods.add(f"src.{rel}")
    return all_mods


def build_inuse_report(entries: Optional[List[str]] = None) -> Dict:
    """计算在用全域并做守恒校验，返回结构化报告。"""
    entry_paths = [PROJECT_ROOT / e for e in (entries or DEFAULT_ENTRIES)]
    static = compute_static_closure(entry_paths)
    static_lb = {m for m in static if m.startswith("src.lazybull")}
    dynamic_extra = sorted(set(DYNAMIC_RISK_MODULES) - static_lb)
    inuse = static_lb | set(dynamic_extra)
    all_mods = list_all_lazybull_modules()
    notin = sorted(all_mods - inuse)

    # 守恒断言：在用 + 不在用 = 全仓；在用必须全部真实存在
    missing_files = sorted(m for m in inuse if _module_to_path(m) is None and m != "src.lazybull")
    conservation = len(inuse) + len(notin) == len(all_mods)

    pkg_count: Dict[str, int] = {}
    for m in inuse:
        parts = m.split(".")
        key = parts[2] if len(parts) > 3 else "(root)"
        pkg_count[key] = pkg_count.get(key, 0) + 1

    return {
        "entries": [e.as_posix() for e in entry_paths],
        "static_closure_count": len(static_lb),
        "dynamic_extra": dynamic_extra,
        "inuse_count": len(inuse),
        "all_modules_count": len(all_mods),
        "notin_count": len(notin),
        "conservation_ok": conservation and not missing_files,
        "missing_files": missing_files,
        "pkg_count": dict(sorted(pkg_count.items())),
        "inuse_modules": sorted(inuse),
        "notin_modules": notin,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="在用全域 import 闭包检查（v2 §4.3 可验证机制）")
    parser.add_argument("--entry", action="append", default=None, help="入口 .py（可多次；默认两条主链路三入口）")
    parser.add_argument("--json-out", default=None, help="结构化结果输出路径（可选）")
    args = parser.parse_args(argv)

    report = build_inuse_report(args.entry)

    print(f"静态闭包: {report['static_closure_count']}  动态加载补充: {len(report['dynamic_extra'])}")
    print(f"在用全域: {report['inuse_count']}  全仓: {report['all_modules_count']}  不在用: {report['notin_count']}")
    print(f"守恒校验: {'PASS' if report['conservation_ok'] else 'FAIL'}")
    print("包级计数:", ", ".join(f"{k}={v}" for k, v in report["pkg_count"].items()))
    if not report["conservation_ok"]:
        for m in report["missing_files"]:
            print(f"[FAIL] 在用模块文件缺失: {m}", file=sys.stderr)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"JSON 底稿: {out}")
    return 0 if report["conservation_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
