# -*- coding: utf-8 -*-
"""术语引用统计脚本（v2 术语契约 §4.5 的工具化承载）

读取术语库 ``docs/glossary/terms.yaml``（唯一数据源），统计每个术语（含别名）
在工作区文档 / 代码注释中的被引用次数，按引用热度排序输出。

两档口径：
- 全量引用：扫描 docs/、src/、scripts/、CLAUDE.md、CHANGELOG.md、README.md 中的出现次数；
- 近 90 天引用：仅统计 ``git log --since='90 days ago'`` 触及过的文件中的出现次数
  （无 git 或命令失败时降级为 "-"，不报错）。

用途：① 术语库 HTML 生成器的排序输入；② 提交验收清单"术语同步"项的检查工具
（新术语必须已入库，否则 --check-missing 模式下可与文档新词人工对照）。

用法：
    python scripts/check/glossary_refs.py [--terms docs/glossary/terms.yaml] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

SCAN_SUFFIXES = {".md", ".py", ".yaml", ".yml"}
DEFAULT_SCAN_PATHS = ["docs", "src", "scripts", "configs"]
EXTRA_FILES = ["CLAUDE.md", "CHANGELOG.md", "README.md"]


@dataclass
class TermStat:
    term: str
    total_refs: int
    recent_refs: Optional[int]  # 近 90 天；git 不可用时 None


def load_terms(terms_path: Path) -> List[dict]:
    """加载术语库"""
    with terms_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    terms = (data or {}).get("terms", [])
    if not isinstance(terms, list):
        raise ValueError(f"terms.yaml 结构非法：terms 必须是列表（{terms_path}）")
    return terms


def collect_files(root: Path, scan_paths: List[str], extra_files: List[str]) -> List[Path]:
    """收集待扫描文件"""
    files: List[Path] = []
    for rel in scan_paths:
        base = root / rel
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in SCAN_SUFFIXES and "__pycache__" not in p.parts:
                files.append(p)
    for rel in extra_files:
        p = root / rel
        if p.is_file():
            files.append(p)
    return files


def git_recent_files(root: Path, days: int = 90) -> Optional[Set[Path]]:
    """近 N 天 git 触及过的文件集合；git 不可用返回 None"""
    try:
        proc = subprocess.run(
            ["git", "log", f"--since={days} days ago", "--name-only", "--format="],
            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    result: Set[Path] = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            result.add((root / line).resolve())
    return result


def _patterns_for(term_entry: dict) -> List[re.Pattern]:
    """为术语及其别名构建匹配模式（ASCII 别名按词边界，中文别名按子串）"""
    names = [term_entry.get("term", "")] + list(term_entry.get("aliases", []) or [])
    patterns: List[re.Pattern] = []
    for name in names:
        name = str(name).strip()
        if not name:
            continue
        if name.isascii():
            patterns.append(re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])"))
        else:
            patterns.append(re.compile(re.escape(name)))
    return patterns


def count_references(files: List[Path], patterns_map: Dict[str, List[re.Pattern]]) -> Dict[str, int]:
    """统计各术语在给定文件集合中的出现次数（按文件计：一个文件命中多次只记 1）"""
    counts: Dict[str, int] = {term: 0 for term in patterns_map}
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for term, patterns in patterns_map.items():
            if any(p.search(text) for p in patterns):
                counts[term] += 1
    return counts


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
    parser = argparse.ArgumentParser(description="LazyBull 术语引用统计（契约 §4.5）")
    parser.add_argument("--terms", default="docs/glossary/terms.yaml", help="术语库路径")
    parser.add_argument("--csv", default=None, help="可选：结果写出 CSV")
    parser.add_argument("--days", type=int, default=90, help="近引用窗口天数（默认 90）")
    args = parser.parse_args(argv)

    root = Path.cwd()
    terms = load_terms(root / args.terms)
    patterns_map: Dict[str, List[re.Pattern]] = {t["term"]: _patterns_for(t) for t in terms if t.get("term")}

    files = collect_files(root, DEFAULT_SCAN_PATHS, EXTRA_FILES)
    total_counts = count_references(files, patterns_map)

    recent_set = git_recent_files(root, days=args.days)
    if recent_set is None:
        recent_counts: Optional[Dict[str, int]] = None
    else:
        recent_files = [p for p in files if p.resolve() in recent_set]
        recent_counts = count_references(recent_files, patterns_map)

    stats = [
        TermStat(
            term=term,
            total_refs=total_counts[term],
            recent_refs=None if recent_counts is None else recent_counts[term],
        )
        for term in patterns_map
    ]
    stats.sort(key=lambda s: (-(s.recent_refs if s.recent_refs is not None else -1), -s.total_refs, s.term))

    print(f"{'术语':<20} {'近%d天引用' % args.days:<12} {'全量引用':<8}")
    print("-" * 44)
    for s in stats:
        recent = "-" if s.recent_refs is None else str(s.recent_refs)
        print(f"{s.term:<20} {recent:<12} {s.total_refs:<8}")
    print(f"\n共 {len(stats)} 个术语；扫描文件 {len(files)} 个")

    if args.csv:
        out = root / args.csv
        with out.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["术语", f"近{args.days}天引用", "全量引用"])
            for s in stats:
                writer.writerow([s.term, "-" if s.recent_refs is None else s.recent_refs, s.total_refs])
        print(f"CSV 已写出：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
