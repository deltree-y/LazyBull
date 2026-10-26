# -*- coding: utf-8 -*-
"""v2 store 运行证据、精确输入摘要与生产者身份。

事件键只使用调用方切好的当日精确输入；本模块不把 run 级整文件哈希混入事件键。
文件清单、工作树和环境版本用于运行前后核验，DataFrame 摘要用于稳定事件身份。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow

__all__ = [
    "InputIdentityDriftError",
    "canonical_json_sha256",
    "collect_file_identities",
    "environment_identity",
    "file_identity",
    "frame_content_digest",
    "producer_identity",
    "verify_file_identities",
    "workspace_identity",
]


class InputIdentityDriftError(RuntimeError):
    """实际输入文件在读取或执行期间发生物理内容漂移。"""


_PRODUCER_FILES = (
    "src/lazybull/common/config.py",
    "src/lazybull/common/date_utils.py",
    "src/lazybull/factors/hierarchical_industry_neutralization.py",
    "src/lazybull/features/labels.py",
    "src/lazybull/features/neutralization.py",
    "src/lazybull/v2/common/types.py",
    "src/lazybull/v2/store/column_groups.py",
    "src/lazybull/v2/store/data_store.py",
    "src/lazybull/v2/store/label_refresh.py",
    "src/lazybull/v2/store/labels_builder.py",
    "src/lazybull/v2/store/manifest.py",
    "src/lazybull/v2/store/manifest_wal.py",
    "src/lazybull/v2/store/row_domain.py",
    "src/lazybull/v2/store/store_identity.py",
    "src/lazybull/v2/store/write_lock.py",
)


def canonical_json_sha256(payload: Any) -> str:
    """canonical JSON 的完整 SHA256；作为稳定事件键和内容摘要的统一编码。"""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _display_path(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return str(path.resolve()).replace("\\", "/")
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def file_identity(path: Path | str, *, base_dir: Path | str | None = None) -> dict[str, Any]:
    """采集一个实际读取文件的物理身份，不记录易变 mtime。"""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"输入身份文件不存在或不是普通文件: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    base = Path(base_dir) if base_dir is not None else None
    return {
        "path": _display_path(source, base),
        "sha256": digest.hexdigest(),
        "size": int(source.stat().st_size),
    }


def collect_file_identities(
    paths: Iterable[Path | str], *, base_dir: Path | str | None = None
) -> list[dict[str, Any]]:
    """按显示路径排序采集文件清单，重复路径必须显式拒绝。"""
    sources = [Path(path) for path in paths]
    base = Path(base_dir) if base_dir is not None else None
    display_paths = [_display_path(path, base) for path in sources]
    if len(set(display_paths)) != len(display_paths):
        raise ValueError("输入身份清单含重复路径，禁止产生歧义证据")
    records = [file_identity(path, base_dir=base) for path in sources]
    return sorted(records, key=lambda record: record["path"])


def verify_file_identities(
    expected: Sequence[Mapping[str, Any]], *, base_dir: Path | str | None = None
) -> list[str]:
    """重算清单并返回漂移路径；空列表表示运行前后输入完全一致。"""
    base = Path(base_dir) if base_dir is not None else None
    mismatches: list[str] = []
    for record in expected:
        raw_path = Path(str(record["path"]))
        path = raw_path if raw_path.is_absolute() or base is None else base / raw_path
        try:
            actual = file_identity(path, base_dir=base)
        except ValueError:
            mismatches.append(str(record["path"]))
            continue
        if any(actual.get(key) != record.get(key) for key in ("sha256", "size")):
            mismatches.append(str(record["path"]))
    return mismatches


def frame_content_digest(frame: pd.DataFrame) -> str:
    """按列 schema、行序和值计算 DataFrame 内容摘要，不依赖其原索引。"""
    if frame.columns.has_duplicates:
        raise ValueError("DataFrame 含重复列名，无法形成无歧义内容摘要")
    work = frame.reset_index(drop=True)
    columns = [{"name": str(column), "dtype": str(work[column].dtype)} for column in work.columns]
    row_hashes = pd.util.hash_pandas_object(work, index=False, categorize=True).to_numpy(
        dtype=np.uint64
    )
    digest = hashlib.sha256()
    header = canonical_json_sha256({"columns": columns, "rows": len(work)}).encode("ascii")
    digest.update(header)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def environment_identity() -> dict[str, str]:
    """标签计算与物理写入相关的运行时版本。"""
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
    }


def producer_identity(repo_root: Path | str) -> dict[str, Any]:
    """标签刷新实现闭包及环境身份；仓库内无关文件不进入该摘要。"""
    root = Path(repo_root)
    files = collect_file_identities([root / path for path in _PRODUCER_FILES], base_dir=root)
    environment = environment_identity()
    payload = {"files": files, "environment": environment}
    return {
        "producer_digest": canonical_json_sha256(payload),
        "files": files,
        "environment": environment,
    }


def _git_output(repo_root: Path, args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError) as exc:  # 含 TimeoutExpired(高负载下 git 偶发超时)
        raise RuntimeError(f"采集 git 工作树身份失败: {repo_root}（{exc}）") from exc
    return completed.stdout.strip()


def workspace_identity(repo_root: Path | str) -> dict[str, Any]:
    """run 级代码证据：git 状态与稳定 src/scripts 内容摘要分列。"""
    root = Path(repo_root)
    paths = sorted((root / "src" / "lazybull").rglob("*.py"))
    paths.extend(sorted((root / "scripts").rglob("*.py")))
    files = collect_file_identities(paths, base_dir=root)
    status = _git_output(root, ["status", "--porcelain", "--untracked-files=all"])
    return {
        "git_commit": _git_output(root, ["rev-parse", "HEAD"]),
        "git_dirty": bool(status),
        "source_digest": canonical_json_sha256(files),
        "source_file_count": len(files),
        "environment": environment_identity(),
    }
