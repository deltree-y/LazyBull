# -*- coding: utf-8 -*-
"""代码复杂度检查脚本（v2 代码复杂度契约 §4.4 的工具化承载）

检查项（软上限触发拆分义务 / 硬上限未经豁免禁止提交）：
- 单文件行数：软 800 / 硬 1200
- 单函数行数：软 80 / 硬 150
- 嵌套深度：软 4 / 硬 5（if/for/while/with/try/match 计入）
- 圈复杂度（mccabe）：硬 12（经 flake8 --max-complexity=12，环境中存在 flake8 时启用）

豁免机制（契约 §4.4）：文件头 30 行内含 ``# 复杂度豁免：`` 注释（须附理由与拆分计划）
的文件降级为告警登记，不触发硬失败。

用法：
    python scripts/check/check_complexity.py [--paths src/lazybull scripts] [--hard-only]
    python scripts/check/check_complexity.py --write-baseline temp/complexity_baseline.json
    python scripts/check/check_complexity.py --baseline temp/complexity_baseline.json

退出码：存在未豁免硬超限 = 1，否则 = 0（软超限只报告）。

存量棘轮（契约 §4.4"迁移即重构"的工具化）：--write-baseline 把当前硬超限登记为基线清单；
--baseline 模式下，基线内的存量超限降级为登记（不阻塞），**任何新增硬超限**仍失败——
存量在 P1~P4 迁移时顺手拆分，基线只准缩小不准扩大（条目消失即胜利，新增即失败）。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

# ---- 契约 §4.4 阈值（唯一来源，禁止调用方覆盖） ----
FILE_LINES_SOFT = 800
FILE_LINES_HARD = 1200
FUNC_LINES_SOFT = 80
FUNC_LINES_HARD = 150
NESTING_SOFT = 4
NESTING_HARD = 5
MCCABE_HARD = 12
WAIVER_MARKER = "复杂度豁免"
WAIVER_HEADER_LINES = 30

_NESTING_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)
try:  # Python 3.10+ match 语句
    _NESTING_NODES = _NESTING_NODES + (ast.Match,)  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - 旧版本解释器
    pass


@dataclass
class Violation:
    """单条超限记录"""

    path: str
    kind: str  # file_lines / func_lines / nesting / mccabe
    level: str  # soft / hard
    detail: str
    waived: bool = False


@dataclass
class FileReport:
    """单文件检查报告"""

    path: str
    violations: List[Violation] = field(default_factory=list)
    waived: bool = False


def _has_waiver(path: Path) -> bool:
    """检查文件头是否登记了复杂度豁免"""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = [next(fh, "") for _ in range(WAIVER_HEADER_LINES)]
    except OSError:
        return False
    return any(WAIVER_MARKER in line for line in head)


def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    """计算函数体内的最大控制结构嵌套深度"""
    best = depth
    for child in ast.iter_child_nodes(node):
        child_depth = depth + 1 if isinstance(child, _NESTING_NODES) else depth
        best = max(best, _max_nesting(child, child_depth))
    return best


def _repo_rel_key(path: Path, root: Path) -> str:
    """统一违规路径键：仓库相对 + 正斜杠 + Windows 大小写归一（跨入口 / 跨机器稳定）。

    P2a-T1 复审 R2-T1-R2-01：flake8 回传的绝对路径直接入键，会在盘符大小写
    （d: vs D:，Windows pathlib 比较大小写不敏感但字符串比较敏感）或仓库目录
    变化时把已登记存量误判为新增；相对化 + posix 化 + ``os.path.normcase``
    大小写归一消除该依赖（POSIX 下 normcase 为恒等，不影响大小写敏感系统）。
    """
    for p in (path, path.resolve()):
        for r in (root, root.resolve()):
            try:
                rel = p.relative_to(r)
                return os.path.normcase(str(rel)).replace("\\", "/")
            except ValueError:
                continue
    return os.path.normcase(str(path)).replace("\\", "/")


def check_file(path: Path, root: Path) -> FileReport:
    """对单文件执行行数 / 函数行数 / 嵌套深度检查"""
    rel = _repo_rel_key(path, root)
    report = FileReport(path=rel)
    waived = _has_waiver(path)
    report.waived = waived

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        report.violations.append(
            Violation(rel, "read_error", "hard", f"读取失败: {exc}", waived=False)
        )
        return report

    n_lines = len(source.splitlines())
    if n_lines > FILE_LINES_HARD:
        report.violations.append(
            Violation(rel, "file_lines", "hard", f"文件 {n_lines} 行 > 硬上限 {FILE_LINES_HARD}", waived)
        )
    elif n_lines > FILE_LINES_SOFT:
        report.violations.append(
            Violation(rel, "file_lines", "soft", f"文件 {n_lines} 行 > 软上限 {FILE_LINES_SOFT}", waived)
        )

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        report.violations.append(
            Violation(rel, "syntax", "hard", f"语法错误: {exc}", waived=False)
        )
        return report

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None) or node.lineno
        func_lines = end - node.lineno + 1
        if func_lines > FUNC_LINES_HARD:
            report.violations.append(
                Violation(
                    rel, "func_lines", "hard",
                    f"函数 {node.name}（L{node.lineno}）{func_lines} 行 > 硬上限 {FUNC_LINES_HARD}",
                    waived,
                )
            )
        elif func_lines > FUNC_LINES_SOFT:
            report.violations.append(
                Violation(
                    rel, "func_lines", "soft",
                    f"函数 {node.name}（L{node.lineno}）{func_lines} 行 > 软上限 {FUNC_LINES_SOFT}",
                    waived,
                )
            )
        depth = _max_nesting(node)
        if depth > NESTING_HARD:
            report.violations.append(
                Violation(
                    rel, "nesting", "hard",
                    f"函数 {node.name}（L{node.lineno}）嵌套深度 {depth} > 硬上限 {NESTING_HARD}",
                    waived,
                )
            )
        elif depth > NESTING_SOFT:
            report.violations.append(
                Violation(
                    rel, "nesting", "soft",
                    f"函数 {node.name}（L{node.lineno}）嵌套深度 {depth} > 软上限 {NESTING_SOFT}",
                    waived,
                )
            )
    return report


def check_mccabe(paths: List[Path], root: Path) -> List[Violation]:
    """经 flake8 跑 mccabe 圈复杂度硬检查；flake8 不可用时降级为告警并跳过

    格式串必须含 ``%(code)s``（P2a-T1 评审 R2-T1-01 修复）：旧格式只有
    ``%(path)s:%(row)d: %(text)s``，输出行不含 "C901" 字样，下游按 "C901"
    过滤会把全部真实超限静默丢弃（漏报缺陷）。
    路径键统一经 ``_repo_rel_key`` 相对化（R2-T1-R2-01 修复）。
    """
    cmd = [sys.executable, "-m", "flake8", f"--max-complexity={MCCABE_HARD}",
           "--select=C901", "--format=%(path)s:%(row)d: %(code)s %(text)s"]
    cmd += [str(p) for p in paths]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[warn] flake8 不可用，圈复杂度检查跳过：{exc}", file=sys.stderr)
        return []
    if proc.returncode not in (0, 1):  # 0=干净 1=有违规；其他为 flake8 自身错误
        print(f"[warn] flake8 执行异常（退出码 {proc.returncode}）：{proc.stderr.strip()}", file=sys.stderr)
        return []
    violations: List[Violation] = []
    # 贪婪 path 段兼容 Windows 盘符冒号（D:\...）；仅采集 C901 行
    line_re = re.compile(r"^(?P<path>.*):(?P<row>\d+): (?P<code>C901) (?P<text>.*)$")
    for line in proc.stdout.splitlines():
        m = line_re.match(line.strip())
        if not m:
            continue
        p = Path(m.group("path").strip())
        violations.append(
            Violation(_repo_rel_key(p, root), "mccabe", "hard", m.group("text").strip(),
                      _has_waiver(p))
        )
    return violations


def collect_py_files(paths: List[Path]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
    return [f for f in files if "__pycache__" not in f.parts]


def _violation_key(v: Violation) -> str:
    """基线比对的稳定键（路径 + 类别 + 摘要；不含行号以免无关编辑冲掉基线）"""
    return f"{v.path}|{v.kind}|{v.detail.split('（')[0]}"


def write_baseline(path: Path, violations: List[Violation]) -> None:
    """把当前硬超限登记为基线清单（JSON，升序稳定）"""
    keys = sorted({_violation_key(v) for v in violations if v.level == "hard"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "keys": keys}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_baseline(path: Path) -> Set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("keys"), list):
        raise ValueError(f"基线文件结构非法：{path}")
    return set(data["keys"])


def main(argv: Optional[List[str]] = None) -> int:
    # 输出编码防御（P0 评审 B3）：管道 / 重定向到文件时强制 UTF-8
    #（默认走 locale 的 cp936，UTF-8 消费方打开乱码）；真控制台不干预
    #（Windows 控制台走 WriteConsoleW，Unicode 安全）
    for stream in (sys.stdout, sys.stderr):
        if stream.isatty():
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    parser = argparse.ArgumentParser(description="LazyBull 代码复杂度检查（契约 §4.4）")
    parser.add_argument("--paths", nargs="+", default=["src/lazybull"], help="检查路径（默认 src/lazybull）")
    parser.add_argument("--hard-only", action="store_true", help="只输出硬超限")
    parser.add_argument("--skip-mccabe", action="store_true", help="跳过 flake8 圈复杂度（离线兜底）")
    parser.add_argument("--write-baseline", default=None, help="把当前硬超限登记为基线 JSON 后退出")
    parser.add_argument("--baseline", default=None, help="存量棘轮：基线外的新增硬超限才失败")
    args = parser.parse_args(argv)

    root = Path.cwd()
    paths = [root / p for p in args.paths]
    files = collect_py_files(paths)
    reports = [check_file(f, root) for f in files]
    violations: List[Violation] = [v for r in reports for v in r.violations]
    if not args.skip_mccabe:
        violations.extend(check_mccabe(paths, root))

    if args.write_baseline:
        write_baseline(root / args.write_baseline, violations)
        hard_n = len([v for v in violations if v.level == "hard"])
        print(f"基线已写出：{args.write_baseline}（硬超限 {hard_n} 条）")
        return 0

    baseline_keys: Set[str] = set()
    if args.baseline:
        baseline_keys = load_baseline(root / args.baseline)

    shown = [v for v in violations if v.level == "hard"] if args.hard_only else violations
    for v in sorted(shown, key=lambda x: (x.path, x.kind)):
        marks = []
        if v.waived:
            marks.append("已豁免")
        if v.level == "hard" and _violation_key(v) in baseline_keys:
            marks.append("存量基线")
        mark = f"（{'，'.join(marks)}）" if marks else ""
        print(f"[{v.level}] {v.path} {v.kind}: {v.detail}{mark}")

    hard_active = [
        v for v in violations
        if v.level == "hard" and not v.waived and _violation_key(v) not in baseline_keys
    ]
    soft_active = [v for v in violations if v.level == "soft" and not v.waived]
    waived_n = len([v for v in violations if v.waived])
    baseline_n = len([v for v in violations if v.level == "hard" and _violation_key(v) in baseline_keys])
    print(
        f"\n检查文件 {len(files)} 个；软超限 {len(soft_active)} 条（拆分义务），"
        f"新增硬超限 {len(hard_active)} 条，存量基线 {baseline_n} 条（登记不阻塞），豁免 {waived_n} 条"
    )
    return 1 if hard_active else 0


if __name__ == "__main__":
    sys.exit(main())
