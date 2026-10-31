# -*- coding: utf-8 -*-
"""T1 门 2b 证据增强：工作树内容身份摘要（r1-P3-3 处置；阶段专用临时开发文件）。

口径 = 门 1 工作树摘要稳定超集：``src/lazybull/**/*.py + scripts/**/*.py``，
按仓库相对路径排序，逐项拼接 UTF-8 路径 + NUL + 文件原始字节 + NUL 后
SHA-256。解决「git_dirty=true 但无法还原工作树内容」的证据缺口：T3 后门 2b
对照与复盘可凭摘要确认代码态。

用法：``python scripts/v2_p4/t1_worktree_digest.py --out <证据目录>/worktree_digest.json``
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_worktree_digest(repo_root: Path) -> dict:
    """计算工作树内容摘要（稳定超集口径）。"""
    files = sorted(
        list(repo_root.glob("src/lazybull/**/*.py")) + list(repo_root.glob("scripts/**/*.py"))
    )
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "scope": "src/lazybull/**/*.py + scripts/**/*.py",
        "algorithm": "sha256 over sorted(relpath, NUL, bytes, NUL)",
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="P4 T1 工作树内容身份摘要")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args(argv)
    result = compute_worktree_digest(_REPO_ROOT)
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"工作树摘要: {result['file_count']} 文件, sha256={result['sha256'][:16]}... -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
