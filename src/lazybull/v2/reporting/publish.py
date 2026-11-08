# -*- coding: utf-8 -*-
"""原子新 run 发布与 generate 的精确写审计入口（设计 §2.5 / §9 / §10）。

硬口径（P6 T2）：

- **新 run**：目标目录已存在即拒绝（无 latest、不覆盖、不追加）；
- **原子发布**：先在 ``.staging/<attempt>/package`` 写全量字节并逐文件回验 SHA，
  再 ``os.rename`` 到 ``<evidence_class>/<kind>/<report_id>``；任何一步失败都清理本次
  package，**不留下半成品 run 目录**；
- **精确写审计**：``audit_generate`` 是 ``readonly_audit.audit_command`` 的薄封装，
  **判定性判据与阈值全部来自 T1**，本模块只补充自证字段（扫描域、重解析点）而不放宽；
  ``readonly_audit.json`` 由调用方写入本次新 validation 目录（不属 generate 允许写集）。

诚实边界：

- ``os.rename`` 在 Windows 上对已存在目标抛错（POSIX 会替换同名**目录**），故换名前
  **二次检查**目标不存在；本地单用户场景无并发覆盖窗口。
- 审计哈希扫描域是**有界子树**（``AUDIT_SCAN_INCLUDES``）：数十 GB 级历史/临时树不参与
  哈希（枚举本身即分钟级），其写保护由运行期 ``ReadonlyGuard`` 承担；扫描域、排除清单与
  重解析点均写入审计载荷，并由 ``uncovered_paths`` 强制“顶层与 ``data/*`` 无未登记盲区”。
- 崩溃残留 staging 按设计 §2.5 保留为失败现场，但永不发布、不参与索引；失败路径清理失败
  会被审计以 ``unexpected_inside`` 捕获（残留必在白名单内但不在 ``expected_paths``）。
"""

from __future__ import annotations

import os
import re
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import contracts, experiment_csv, readonly_audit, trend_csv, validation
from src.lazybull.v2.reporting.snapshot import ROOT_ID_PATTERN

__all__ = [
    "AUDIT_SCAN_EXCLUDES",
    "AUDIT_SCAN_INCLUDES",
    "DASHBOARD_FILENAMES",
    "EXPERIMENT_FILENAMES",
    "RUN_FILENAMES",
    "RUN_FILENAMES_BY_KIND",
    "TREND_FILENAMES",
    "PublishError",
    "assert_attempt_id",
    "audit_generate",
    "discard_attempt",
    "publish_new_run",
    "scan_audit_roots",
    "uncovered_paths",
]

#: 驾驶舱 run 的固定文件集（``RUN_FILENAMES`` 保留为 T2 兼容别名）
DASHBOARD_FILENAMES: tuple[str, ...] = (
    "report.html",
    "projection.json",
    "report_manifest.json",
)
RUN_FILENAMES: tuple[str, ...] = DASHBOARD_FILENAMES
#: 实验 run 的固定文件集（三份只读件 + **五份完整 CSV**；缺一即失败）
EXPERIMENT_FILENAMES: tuple[str, ...] = DASHBOARD_FILENAMES + experiment_csv.CSV_FILENAMES
#: 趋势 run 的固定文件集（三份只读件 + **两份完整 CSV**；缺一即失败）
TREND_FILENAMES: tuple[str, ...] = DASHBOARD_FILENAMES + trend_csv.TREND_FILENAMES
#: 报告类型 → 允许的固定文件名闭集（未登记类型 fail-closed）
RUN_FILENAMES_BY_KIND: dict[str, tuple[str, ...]] = {
    "dashboard": DASHBOARD_FILENAMES,
    "experiment": EXPERIMENT_FILENAMES,
    "trend": TREND_FILENAMES,
}
#: 审计扫描的**有界**子树（repo 相对路径）。``data/models``/``data/frozen_reference``/
#: ``data/walk_forward``/``data/paper_shadow``/``temp`` 等数十 GB 级历史与临时产物不在
#: 哈希扫描域内；其写保护由运行期 ``ReadonlyGuard`` 承担。
AUDIT_SCAN_INCLUDES: tuple[str, ...] = (
    ".github",
    "configs",
    "data/ledger",
    "data/paper",
    "data/reports",
    "data/runs",
    "docs",
    "examples",
    "logs",
    "scripts",
    "src",
    "tests",
)
#: 未纳入哈希扫描域的目录（理由见上；登记进审计载荷，并由 ``uncovered_paths`` 强制无盲区）
AUDIT_SCAN_EXCLUDES: tuple[str, ...] = (
    ".git/**",
    ".claude/**",
    ".codebuddy/**",
    ".idea/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".venv/**",
    ".vscode/**",
    "__pycache__/**",
    "data/backup/**",
    "data/clean/**",
    "data/empty/**",
    "data/features/**",
    "data/frozen_reference/**",
    "data/labels/**",
    "data/models/**",
    "data/paper_backup_pre_v2_20261031/**",
    "data/paper_legacy_pre_switch_20261031/**",
    "data/paper_q11sim/**",
    "data/paper_shadow/**",
    "data/raw/**",
    "data/walk_forward/**",
    "temp/**",
    "venv/**",
)


#: Windows 保留设备名（正则允许但无法建目录；提前拦成“参数形态非法”）
_RESERVED_DEVICE_NAMES: frozenset[str] = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


class PublishError(RuntimeError):
    """run 目录不合法、已存在或字节回验失败（stop-the-line，不发布）。"""


def _sha256_bytes(payload: bytes) -> str:
    """字节 SHA-256（outputs 登记口径）。"""
    return sha256(payload).hexdigest()


def assert_attempt_id(attempt_id: str) -> str:
    """attempt 标识与 T1 快照同一口径（小写字母开头、仅小写/数字/下划线、<=32）。

    复用 ``snapshot.ROOT_ID_PATTERN`` 并**全串匹配**（``fullmatch``，避免 ``$`` 放过
    ``"ab\\n"`` 这类尾随换行）：``.``/``..``/首尾点会直接改变 ``.staging/<attempt>`` 的
    解析目标（``discard_attempt`` 会整树删除），必须 fail-fast；生成入口也复用本函数，
    防止把 staging 写到白名单之外（T2-R2-N1）。

    另拒 Windows 保留设备名（``con``/``prn``/``aux``/``nul``/``com1..9``/``lpt1..9``）：
    正则允许它们，但建目录会抛 ``OSError`` 并逃逸为裸 traceback（T2-R2-M1）。
    """
    text = str(attempt_id)
    if not ROOT_ID_PATTERN.fullmatch(text):
        raise PublishError(
            f"attempt_id 形态非法（要求 {ROOT_ID_PATTERN.pattern}，全串匹配）：{attempt_id!r}"
        )
    if text in _RESERVED_DEVICE_NAMES:
        raise PublishError(f"attempt_id 不得使用 Windows 保留设备名：{attempt_id!r}")
    return attempt_id


def _relative_run_path(evidence_class: str, report_kind: str, report_id: str) -> str:
    """run 目录的 repo 相对 POSIX 路径（与精确写白名单同源拼接）。"""
    if evidence_class not in contracts.EVIDENCE_CLASSES:
        raise PublishError(f"未知 evidence_class：{evidence_class!r}")
    if report_kind not in contracts.REPORT_KINDS:
        raise PublishError(f"未知 report_kind：{report_kind!r}")
    if not re.match(contracts.REPORT_ID_PATTERN, report_id):
        raise PublishError(f"report_id 形态不符冻结 pattern：{report_id!r}")
    return f"{contracts.REPORTS_ROOT}/{evidence_class}/{report_kind}/{report_id}"


def _validate_files(files: Mapping[str, bytes], report_kind: str) -> dict[str, bytes]:
    """文件集校验：按报告类型只允许固定名、非空字节、无路径分隔符。"""
    allowed = RUN_FILENAMES_BY_KIND.get(str(report_kind))
    if allowed is None:
        raise PublishError(f"未知 report_kind：{report_kind!r}（合法 {sorted(RUN_FILENAMES_BY_KIND)}）")
    unknown = sorted(set(files) - set(allowed))
    if unknown:
        raise PublishError(f"run 文件集含未登记文件：{unknown}（{report_kind} 只允许 {allowed}）")
    missing = [name for name in allowed if name not in files]
    if missing:
        raise PublishError(f"{report_kind} run 文件集缺 {missing}（固定文件集必须齐备）")
    material: dict[str, bytes] = {}
    for name, payload in files.items():
        if "/" in name or "\\" in name or name in ("", ".", ".."):
            raise PublishError(f"文件名必须是单段固定名：{name!r}")
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            raise PublishError(f"{name} 必须是非空 bytes")
        material[name] = bytes(payload)
    return material


def _write_package(package: Path, files: Mapping[str, bytes]) -> dict[str, str]:
    """写 staging 包并逐文件回验 SHA（写坏/被截断即失败）。"""
    digests: dict[str, str] = {}
    for name in sorted(files):
        target = package / name
        target.write_bytes(files[name])
        actual = validation.sha256_file(target)
        expected = _sha256_bytes(files[name])
        if actual != expected:
            raise PublishError(f"{name} 落盘字节与输入不一致：{actual} ≠ {expected}")
        digests[name] = actual
    return digests


def publish_new_run(
    *,
    reports_root: Path | str,
    evidence_class: str,
    report_kind: str,
    report_id: str,
    files: Mapping[str, bytes],
    attempt_id: str,
) -> Path:
    """原子发布新 run；返回 run 目录绝对路径（失败时目标目录不存在）。

    Raises:
        PublishError: 路径不合法、目标已存在、package 已存在或字节回验失败。
    """
    material = _validate_files(files, report_kind)
    relative = _relative_run_path(evidence_class, report_kind, report_id)
    root = Path(reports_root)
    target = root / Path(relative).relative_to(contracts.REPORTS_ROOT)
    if target.exists():
        raise PublishError(f"run 目录已存在，拒绝覆盖：{target}")
    assert_attempt_id(attempt_id)
    package = root / ".staging" / attempt_id / "package"
    if package.exists():
        raise PublishError(f"staging package 已存在，拒绝复用：{package}")
    try:
        package.mkdir(parents=True)
        _write_package(package, material)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():  # 换名前二次检查（POSIX rename 会替换同名目录）
            raise PublishError(f"run 目录在发布过程中出现，拒绝覆盖：{target}")
        os.rename(package, target)
    except BaseException:
        shutil.rmtree(package, ignore_errors=True)
        raise
    return target


def discard_attempt(reports_root: Path | str, attempt_id: str) -> None:
    """清理本次 staging attempt（成功收尾与失败回滚共用；只删自己的 attempt 目录）。"""
    assert_attempt_id(attempt_id)
    staging_root = Path(reports_root) / ".staging"
    target = staging_root / attempt_id
    if target.parent != staging_root:  # 双保险：目标必须恰是 .staging 的直接子目录
        raise PublishError(f"拒绝清理非本次 staging 目录：{target}")
    shutil.rmtree(target, ignore_errors=True)


def _covered(relative: str, patterns: Sequence[str]) -> bool:
    """路径是否被 includes/excludes 的某个前缀覆盖（``dir`` 或 ``dir/**`` 形态）。"""
    for pattern in patterns:
        base = str(pattern).split("/**")[0]
        if relative == base or relative.startswith(f"{base}/"):
            return True
    return False


def uncovered_paths(repo_root: Path | str) -> list[str]:
    """未登记的顶层目录/文件与 ``data/*`` 目录/文件（非空即审计盲区，必须 fail）。

    ``data/`` 根下的**文件**也在候选内（哈希侧只覆盖 repo 顶层文件，若放任
    ``data/<file>`` 既不进哈希也不进闭包，会形成结构性盲区——T2-R2-N2）。
    """
    base = Path(repo_root)
    patterns = list(AUDIT_SCAN_INCLUDES) + list(AUDIT_SCAN_EXCLUDES)
    candidates: list[str] = []
    for path in sorted(base.iterdir()):
        if path.is_dir() and path.name != "data":
            # ``data`` 自身由 data/* 条目逐项登记（见下），不单独计为候选
            candidates.append(path.name)
    data_root = base / "data"
    if data_root.is_dir():
        for path in sorted(data_root.iterdir()):
            candidates.append(f"data/{path.name}")
    return sorted(item for item in candidates if not _covered(item, patterns))


def scan_audit_roots(
    *,
    repo_root: Path | str,
    extra_roots: Mapping[str, Path | str] | None = None,
    includes: tuple[str, ...] = AUDIT_SCAN_INCLUDES,
) -> readonly_audit.FileTree:
    """按**有界子树**做 repo 相对整树快照（含目录集合与重解析点，供“新目录”判据）。

    键形如 ``worktree:<repo 相对路径>``；``extra_roots`` 用于 repo 之外的输入根
    （其任何变化都会被 ``audit_command`` 判为 ``foreign_root_paths``）。
    重解析点集合挂在返回的 ``FileTree.reparse`` 上（``scan_tree`` 不穿透、只登记）。
    """
    base = Path(repo_root)
    files: dict[str, dict[str, Any]] = {}
    dirs: set[str] = set()
    reparse: set[str] = set()
    for relative in includes:
        root = base / relative
        if not root.is_dir():
            continue
        scan = readonly_audit.scan_tree({"worktree": root})
        for key, value in scan.files.items():
            files[f"worktree:{relative}/{key.split(':', 1)[1]}"] = dict(value)
        for item in scan.dirs:
            dirs.add(f"worktree:{relative}/{item.split(':', 1)[1]}")
        for item in scan.reparse:
            reparse.add(f"worktree:{relative}/{item.split(':', 1)[1]}")
    for path in sorted(base.iterdir()):
        if path.is_file():
            files[f"worktree:{path.name}"] = {
                "sha256": validation.sha256_file(path),
                "size": path.stat().st_size,
            }
    for root_id, root in sorted((extra_roots or {}).items()):
        directory = Path(root)
        if not directory.is_dir():
            continue
        scan = readonly_audit.scan_tree({root_id: directory})
        files.update({key: dict(value) for key, value in scan.files.items()})
        dirs.update(scan.dirs)
        reparse.update(scan.reparse)
    tree = readonly_audit.FileTree(files, dirs)
    tree.reparse = frozenset(reparse)  # type: ignore[attr-defined]
    return tree


def _reparse_of(tree: Any) -> list[str]:
    """取快照携带的重解析点（缺失视为无；不静默默认成功由调用方另行登记）。"""
    return sorted(getattr(tree, "reparse", ()) or ())


def audit_generate(
    *,
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    expected_paths: Any,
    bindings: Mapping[str, str],
    output_root: str,
    repo_root: Path | str,
) -> dict[str, Any]:
    """generate 命令的精确写判定（薄封装 T1 ``audit_command``，只补自证字段）。

    ``output_root`` 必填（与 T1 同口径，缺参即 TypeError）；``repo_root`` 用于登记
    “扫描域未覆盖的路径”这一盲区判据——非空即 fail（``extra_violations``）。
    """
    uncovered = uncovered_paths(repo_root)
    # “已检查”要求快照**真的携带**一个可迭代的重解析点集合（``reparse=None`` 不算检查过）
    checked = all(isinstance(getattr(tree, "reparse", None), (frozenset, set, tuple, list))
                  for tree in (before, after))
    reparse = sorted(set(_reparse_of(before)) | set(_reparse_of(after)))
    violations = [
        {"action": "scan_scope", "path": item, "kind": "unregistered_scan_scope"}
        for item in uncovered
    ]
    if not checked:
        # 快照未携带 reparse 集合 ⇒ “未检查”不能读作“已检查且为空”（T2-R2-N3）
        violations.append(
            {"action": "reparse_scan", "path": "<missing>", "kind": "reparse_not_checked"}
        )
    report = readonly_audit.audit_command(
        command="generate",
        before=before,
        after=after,
        expected_paths=list(expected_paths),
        bindings=dict(bindings),
        reparse_paths=reparse,
        extra_violations=violations,
        output_root=output_root,
    )
    report["scan_scope"] = {
        "includes": list(AUDIT_SCAN_INCLUDES),
        "excludes": list(AUDIT_SCAN_EXCLUDES),
        "uncovered": uncovered,
        "reparse_checked": checked,
        "reparse_points": reparse,
        "note": "有界子树内哈希零差异；排除树由运行期只读守卫承担",
    }
    return report
