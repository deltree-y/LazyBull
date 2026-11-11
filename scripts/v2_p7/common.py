# -*- coding: utf-8 -*-
"""P7 T0.5 公共件：仓库根解析、规范化 JSON 写出、写域守卫。

写域守卫（F13 / 运行手册 §1）
=============================
- 允许根默认只有 `data/paper_shadow/p7_testbed` 与 `temp`（仓库内）；
- 目标路径 `resolve()` 后必须落在某个允许根之内，且不得落在冻结禁写面；
- 拒绝即抛 :class:`WriteScopeError`，调用方须在 **任何 mkdir/写字节之前** 校验，
  保证失败路径零副作用；
- 测试可用 ``--allow-root`` 显式追加隔离临时根（生产运行不得追加生产根）。
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 默认允许写入的仓库内根（相对仓库根）
DEFAULT_WRITE_ROOTS: tuple[str, ...] = ("data/paper_shadow/p7_testbed", "temp")

#: 冻结禁写面（任何情况下不得写入；即使被 allow-root 覆盖也拒绝）
FORBIDDEN_WRITE_PREFIXES: tuple[str, ...] = (
    "data/paper",
    "data/runs",
    "data/models",
    "data/raw",
    "data/clean",
    "data/features",
    "data/labels",
    "data/ledger",
    "data/frozen_reference/v2_p1",
    "data/frozen_reference/v2_p3",
    "data/frozen_reference/v2_p4",
    "data/frozen_reference/v2_p5b",
    "data/frozen_reference/v2_p65",
    "src/lazybull",
    "configs",
    "scripts",
    "docs",
)


class WriteScopeError(RuntimeError):
    """目标路径不在冻结写域内（一律 fail-closed，且调用前不得产生任何副作用）。"""


def sha256_bytes(data: bytes) -> str:
    """字节级 SHA256。"""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """文件字节级 SHA256（分块读取，避免大文件一次性载入）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    """规范化 JSON 字节（sort_keys + UTF-8 + LF 结尾，禁止 NaN）。"""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> str:
    """写规范化 JSON（二进制 LF）并返回磁盘字节 SHA256。

    必须二进制写入：文本模式会把 ``\\n`` 翻译成 ``\\r\\n``，使登记 SHA 与磁盘字节不一致。
    """
    data = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def _relative_to_repo(path: Path) -> str | None:
    """仓库内相对路径（POSIX 分隔符）；仓库外返回 None。"""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def resolve_write_roots(extra_roots: Sequence[str] = ()) -> tuple[Path, ...]:
    """解析允许写入根（仓库内相对或绝对路径均支持）。"""
    roots: list[Path] = []
    for item in DEFAULT_WRITE_ROOTS:
        roots.append((REPO_ROOT / item).resolve())
    for item in extra_roots:
        roots.append(Path(item).expanduser().resolve())
    return tuple(roots)


def _is_reparse_point(path: Path) -> bool:
    """符号链接 / Windows junction 等重解析点（真正的逃逸载体）。"""
    if path.is_symlink():
        return True
    try:
        metadata = os.lstat(path)
    except OSError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _assert_no_reparse_between(target: Path, roots: Sequence[Path]) -> None:
    """目标到允许根之间的任何既存分量都不得是重解析点（防子目录 junction 逃逸）。"""
    resolved_roots = [Path(root).resolve() for root in roots]
    cursor = Path(target).expanduser()
    if not cursor.is_absolute():
        cursor = (REPO_ROOT / cursor).resolve()
    for _ in range(64):
        if cursor.exists() and _is_reparse_point(cursor):
            raise WriteScopeError(f"写域包含重解析点（链接/junction）：{cursor}")
        if any(cursor.resolve() == root for root in resolved_roots):
            return
        parent = cursor.parent
        if parent == cursor:
            return
        cursor = parent


def assert_within_write_roots(
    target: Path,
    roots: Sequence[Path],
    forbidden_prefixes: Iterable[str] = FORBIDDEN_WRITE_PREFIXES,
) -> Path:
    """校验目标路径落在允许根内且不在冻结禁写面；返回解析后的路径。

    除根目录外，还会沿目标到根逐级检查既存分量是否为链接/junction，
    防止「根合法、子目录指向禁写面」的逃逸。

    Raises:
        WriteScopeError: 越界、命中禁写面、根目录本身或含重解析点。
    """
    resolved = Path(target).expanduser().resolve()
    relative = _relative_to_repo(resolved)
    if relative is None:
        if not any(_is_within(resolved, root) for root in roots):
            raise WriteScopeError(f"写域越界（仓库外且未授权）：{resolved}")
        _assert_no_reparse_between(target, roots)
        return resolved
    for prefix in forbidden_prefixes:
        if relative == prefix or relative.startswith(f"{prefix}/"):
            raise WriteScopeError(f"命中冻结禁写面 `{prefix}`：{relative}")
    for root in roots:
        if resolved == Path(root).resolve() or _is_within(resolved, root):
            _assert_no_reparse_between(target, roots)
            return resolved
    raise WriteScopeError(f"写域越界（不在允许根内）：{relative}")


def assert_targets_within_write_roots(
    targets: Sequence[Path],
    roots: Sequence[Path],
    forbidden_prefixes: Iterable[str] = FORBIDDEN_WRITE_PREFIXES,
) -> List[Path]:
    """**写前**整体预检一组目标（全部通过才返回），保证拒绝路径零副作用。"""
    resolved: List[Path] = []
    for target in targets:
        resolved.append(
            assert_within_write_roots(target, roots, forbidden_prefixes=forbidden_prefixes)
        )
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    """``path`` 是否严格位于 ``root`` 之下（含重解析点逃逸防护）。"""
    try:
        path.relative_to(Path(root).resolve())
    except ValueError:
        return False
    return path != Path(root).resolve()
