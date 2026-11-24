# -*- coding: utf-8 -*-
"""campaign / run 布局、写域守卫与封存发布（协议 §2.2、§2.3、§9）。

三条硬纪律
==========
1. **写域封闭**：只允许 ``data/reports/p7_cashflow_split_20261120`` 与 ``temp``；
   命中 ``data/models`` / ``data/features`` / ``data/paper`` 等冻结面一律 fail-closed
   （复用 :func:`scripts.v2_p7.common.assert_within_write_roots`）。
2. **不覆盖旧证据**：``create_run_dir`` 用 ``exist_ok=False``；同一 run 重跑必须新 run。
3. **中断不算完成**：所有工件先落 ``<run>/work`` 暂存，校验后逐件发布到 ``<run>`` 根，
   最后才写 ``manifest.json`` 与 ``PUBLISHED.json``。缺 ``PUBLISHED.json`` 的目录
   一律视为未完成（:func:`require_published` 直接拒绝）。
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from scripts.v2_p7.cashflow_sleeve_split import spec
from scripts.v2_p7.common import (
    REPO_ROOT,
    WriteScopeError,
    assert_targets_within_write_roots,
    sha256_bytes,
    sha256_file,
    write_json,
)

#: run 状态文件（缺失即未完成）
PUBLISHED_NAME = "PUBLISHED.json"
MANIFEST_NAME = "manifest.json"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$")
_CAMPAIGN_RE = re.compile(r"^[a-z0-9][a-z0-9_]{1,63}$")


class ArtifactError(RuntimeError):
    """工件缺失、身份不符或写入被拒（fail-closed）。"""


def default_artifact_base() -> Path:
    """正式研究产物根（协议 §2.2）。"""
    return REPO_ROOT / spec.ARTIFACT_ROOT_RELATIVE


def write_roots(extra: Sequence[str] = ()) -> Tuple[Path, ...]:
    """允许写入的根：研究产物根 + ``temp``（可显式追加隔离测试根）。"""
    roots = [default_artifact_base().resolve(), (REPO_ROOT / "temp").resolve()]
    roots.extend(Path(item).expanduser().resolve() for item in extra)
    return tuple(roots)


def guard(target: Path, extra_roots: Sequence[str] = ()) -> Path:
    """写前校验单个目标路径（任何 mkdir/写字节之前调用）。"""
    return assert_targets_within_write_roots([Path(target)], write_roots(extra_roots))[0]


def campaign_root(campaign: str, artifact_base: Path | None = None) -> Path:
    """``<base>/<campaign>``；不建目录、不改身份。"""
    if not _CAMPAIGN_RE.match(str(campaign)):
        raise ArtifactError(f"campaign 名称非法（须小写字母/数字/下划线）：{campaign!r}")
    base = Path(artifact_base) if artifact_base is not None else default_artifact_base()
    return base / str(campaign)


def ensure_layout(camp: Path) -> Dict[str, Path]:
    """建立 campaign 固定子目录（幂等；只建空目录）。"""
    layout = {
        "specification": camp / "specification",
        "inputs": camp / "inputs",
        "runs": camp / "runs",
        "models": camp / "models",
        "cache": camp / "cache",
    }
    for path in layout.values():
        guard(path)
        path.mkdir(parents=True, exist_ok=True)
    return layout


def utc_now() -> str:
    """UTC 时间戳（秒级，登记用）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(stage: str, now: datetime | None = None) -> str:
    """``<stage>_<UTC yyyymmddTHHMMSS>``。"""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    return f"{stage}_{stamp}"


def create_run_dir(camp: Path, run_id: str, extra_roots: Sequence[str] = ()) -> Path:
    """新建 run 目录与 ``work`` 暂存区；已存在即拒绝（禁止同 run 重写）。"""
    if not _RUN_ID_RE.match(str(run_id)):
        raise ArtifactError(f"run_id 非法：{run_id!r}")
    run_dir = Path(camp) / "runs" / str(run_id)
    guard(run_dir, extra_roots)
    if run_dir.exists():
        raise ArtifactError(
            f"run 目录已存在，禁止覆盖旧证据：{run_dir}（请用新 run_id，协议 §2.3）"
        )
    run_dir.mkdir(parents=True)
    (run_dir / "work").mkdir()
    return run_dir


def publish_file(
    work_path: Path,
    run_dir: Path,
    name: str,
    extra_roots: Sequence[str] = (),
) -> Dict[str, Any]:
    """把暂存件发布到 run 根（同名字节相同则幂等，不同即拒绝）。"""
    source = Path(work_path)
    if not source.exists():
        raise ArtifactError(f"待发布工件缺件：{source}")
    target = Path(run_dir) / name
    guard(target, extra_roots)
    digest = sha256_file(source)
    if target.exists():
        if sha256_file(target) != digest:
            raise ArtifactError(f"同名工件字节不同，拒绝覆盖：{target}")
        return {"name": name, "sha256": digest, "size": target.stat().st_size, "reused": True}
    shutil.copy2(source, target)
    if sha256_file(target) != digest:
        raise ArtifactError(f"发布后字节不符（复制损坏）：{target}")
    return {"name": name, "sha256": digest, "size": target.stat().st_size, "reused": False}


def file_digests(
    directory: Path, skip: Iterable[str] = (MANIFEST_NAME, PUBLISHED_NAME)
) -> Dict[str, str]:
    """目录下全部文件的相对路径 → SHA256（跳过封存件本身）。"""
    skip_set = set(skip)
    out: Dict[str, str] = {}
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in skip_set or relative.startswith("work/"):
            continue
        out[relative] = sha256_file(path)
    return out


#: 附件子目录（补丁、被改源码、测量件；随 run 封存但不算七件工件）
ATTACHMENTS_DIR = "attachments"


def publish_attachments(
    run_dir: Path,
    sources: Mapping[str, Path],
    extra_roots: Sequence[str] = (),
) -> Dict[str, str]:
    """把附件（补丁/源码/测量件）复制进 ``attachments/`` 并返回逐件 SHA。"""
    import shutil

    root = Path(run_dir) / ATTACHMENTS_DIR
    digests: Dict[str, str] = {}
    for relative, source in sources.items():
        source = Path(source)
        if not source.is_file():
            raise ArtifactError(f"附件缺件：{source}")
        target = root / relative
        guard(target, extra_roots)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digests[f"{ATTACHMENTS_DIR}/{relative}"] = sha256_file(target)
    return digests


def seal_run(
    run_dir: Path,
    stage: str,
    payload: Mapping[str, Any],
    artifact_names: Sequence[str],
    extra_roots: Sequence[str] = (),
    attachments: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """校验全部必需工件齐全后写 ``manifest.json`` 与 ``PUBLISHED.json``。

    缺件、字节不符、状态异常 ⇒ 抛 :class:`ArtifactError`，且**不写**发布标记。
    """
    if stage not in spec.STAGES:
        raise ArtifactError(f"未知阶段：{stage!r}")
    missing = [name for name in artifact_names if not (Path(run_dir) / name).exists()]
    if missing:
        raise ArtifactError(f"阶段 {stage} 缺工件，拒绝封存：{missing}")
    digests = file_digests(Path(run_dir))
    prefix = f"{ATTACHMENTS_DIR}/"
    undeclared = sorted(
        name
        for name in set(digests) - set(artifact_names)
        if not name.startswith(prefix) and name not in (MANIFEST_NAME, PUBLISHED_NAME)
    )
    if undeclared:
        raise ArtifactError(f"阶段 {stage} 存在未声明工件：{undeclared}")
    declared_attachments = dict(attachments or {})
    actual_attachments = {
        name: digest for name, digest in digests.items() if name.startswith(prefix)
    }
    if set(actual_attachments) != set(declared_attachments):
        raise ArtifactError(
            f"附件清单与磁盘不符：缺 {sorted(set(declared_attachments) - set(actual_attachments))}；"
            f"多 {sorted(set(actual_attachments) - set(declared_attachments))}"
        )
    for name, digest in declared_attachments.items():
        if actual_attachments[name] != digest:
            raise ArtifactError(f"附件字节与登记不符：{name}")

    manifest = {
        "schema": "p7_cashflow_split_manifest/v1",
        "candidate_id": spec.CANDIDATE_ID,
        "stage": stage,
        "status": "sealed",
        "sealed_at": utc_now(),
        "protocol_sha256": spec.protocol_sha256(),
        "spec_sha256": spec.spec_sha256(),
        "artifacts": {name: digests[name] for name in artifact_names},
        "attachments": dict(attachments or {}),
        "attachments_n": len(attachments or {}),
        "payload": dict(payload),
    }
    manifest_path = Path(run_dir) / MANIFEST_NAME
    guard(manifest_path, extra_roots)
    manifest_sha = write_json(manifest_path, manifest)
    marker = {
        "schema": "p7_cashflow_split_published/v1",
        "stage": stage,
        "manifest_sha256": manifest_sha,
        "artifact_count": len(artifact_names),
        "published_at": utc_now(),
    }
    marker_sha = write_json(Path(run_dir) / PUBLISHED_NAME, marker)
    return {
        "run_dir": str(run_dir),
        "stage": stage,
        "manifest_sha256": manifest_sha,
        "published_sha256": marker_sha,
        "artifacts": dict(manifest["artifacts"]),
    }


def require_published(camp: Path, run_id: str | None, stage: str) -> Dict[str, Any]:
    """读取**已完成**上游 run 的 manifest；缺件/未发布/阶段不符即拒绝。"""
    if not run_id:
        raise ArtifactError(f"缺少上游 run 身份：--parent（阶段 {stage}）")
    run_dir = Path(camp) / "runs" / str(run_id)
    marker = run_dir / PUBLISHED_NAME
    manifest_path = run_dir / MANIFEST_NAME
    if not marker.exists() or not manifest_path.exists():
        raise ArtifactError(
            f"上游 run 未完成或不存在（缺 {PUBLISHED_NAME}）：{run_dir}；"
            "中断分片不得作为已完成输入（协议 §2.3）"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker_body = json.loads(marker.read_text(encoding="utf-8"))
    if sha256_file(manifest_path) != str(marker_body.get("manifest_sha256")):
        raise ArtifactError(f"manifest 字节与发布标记不符（疑似被改写）：{manifest_path}")
    if manifest.get("stage") != stage:
        raise ArtifactError(f"上游 run 阶段不符：期望 {stage}，实际 {manifest.get('stage')}")
    for name, digest in (manifest.get("artifacts") or {}).items():
        path = run_dir / name
        if not path.exists():
            raise ArtifactError(f"上游工件缺件：{path}")
        if sha256_file(path) != str(digest):
            raise ArtifactError(f"上游工件字节不符：{path}")
    return manifest


def require_write_scope_probe() -> Dict[str, Any]:
    """自检：生产根必须不可写（返回被拒样例，供 readiness 引用）。"""
    probe = REPO_ROOT / "data" / "models" / "stock_selection" / "__cf_split_probe__"
    try:
        guard(probe)
    except WriteScopeError as error:
        return {"rejected": True, "target": str(probe), "reason": str(error)}
    return {"rejected": False, "target": str(probe), "reason": "未被拒绝（写域守卫失效）"}


__all__ = [
    "ArtifactError",
    "MANIFEST_NAME",
    "PUBLISHED_NAME",
    "campaign_root",
    "create_run_dir",
    "default_artifact_base",
    "ensure_layout",
    "file_digests",
    "guard",
    "new_run_id",
    "publish_file",
    "require_published",
    "require_write_scope_probe",
    "seal_run",
    "sha256_bytes",
    "utc_now",
    "write_roots",
]
