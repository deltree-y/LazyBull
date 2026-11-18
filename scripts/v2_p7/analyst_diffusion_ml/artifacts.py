"""研究工件落盘：隔离根、原子写、拒绝覆盖、阶段封存。

写域 = `<仓库>/data/reports/p7_analyst_diffusion_ml_20261118/<campaign_id>`，
复用 `scripts.v2_p7.common` 的写域守卫（硬拒 `data/raw|clean|features|models|paper|runs|ledger`）。
测试可显式传入隔离根（`artifact_base`），默认根之外的写入一律被拒。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.v2_p7.analyst_diffusion_ml import spec
from scripts.v2_p7.common import (
    REPO_ROOT,
    assert_targets_within_write_roots,
    assert_within_write_roots,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_json,
)

SUBDIRS: tuple[str, ...] = (
    "specification",
    "inputs",
    "events",
    "dataset",
    "models",
    "predictions",
    "accounts",
    "diagnostics",
    "runs",
    "reports",
)


class ArtifactError(RuntimeError):
    """工件层错误（覆盖、越界、内容不符）。"""


def default_artifact_base() -> Path:
    """默认研究工件根（仓库内）。"""
    return REPO_ROOT / spec.ARTIFACT_ROOT_RELATIVE


def write_roots(artifact_base: Path | None = None) -> tuple[Path, ...]:
    """允许写入的根：研究工件根 + temp（仓库内）。"""
    base = Path(artifact_base) if artifact_base is not None else default_artifact_base()
    return (base.resolve(), (REPO_ROOT / "temp").resolve())


def campaign_root(campaign_id: str, artifact_base: Path | None = None) -> Path:
    """campaign 根（校验写域，不创建目录）。"""
    if not campaign_id or "/" in campaign_id or "\\" in campaign_id or campaign_id in (".", ".."):
        raise ArtifactError(f"非法 campaign_id：{campaign_id!r}")
    base = Path(artifact_base) if artifact_base is not None else default_artifact_base()
    target = base / campaign_id
    return assert_within_write_roots(target, write_roots(artifact_base))


def ensure_layout(root: Path, artifact_base: Path | None = None) -> dict[str, str]:
    """创建 campaign 固定子目录，返回子目录 -> 绝对路径。"""
    roots = write_roots(artifact_base)
    resolved = assert_within_write_roots(root, roots)
    paths = [resolved / name for name in SUBDIRS]
    assert_targets_within_write_roots(paths, roots)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return {name: str(resolved / name) for name in SUBDIRS}


def utc_now() -> str:
    """UTC 时间戳（只作运行元数据，不进数值身份）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(stage: str, now: datetime | None = None) -> str:
    """run id = `<stage>_<UTC yyyymmddTHHMMSS>`。"""
    moment = now or datetime.now(timezone.utc)
    return f"{stage}_{moment.strftime('%Y%m%dT%H%M%S')}"


def create_run_dir(root: Path, run_id: str, artifact_base: Path | None = None) -> Path:
    """新建 run 目录（已存在即拒绝，不覆盖旧证据）。"""
    if not run_id:
        raise ArtifactError("run_id 不得为空")
    roots = write_roots(artifact_base)
    target = assert_within_write_roots(root / "runs" / run_id, roots)
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ArtifactError(f"run 目录已存在，拒绝覆盖：{target}") from exc
    return target


def write_json_checked(
    path: Path, payload: Mapping[str, Any], artifact_base: Path | None = None
) -> dict[str, str]:
    """写规范化 JSON；已存在时只允许与本次字节完全一致（幂等），否则拒绝。"""
    resolved = assert_within_write_roots(path, write_roots(artifact_base))
    data = canonical_json_bytes(dict(payload))
    digest = sha256_bytes(data)
    if resolved.exists():
        existing = resolved.read_bytes()
        if existing == data:
            return {"path": str(resolved), "sha256": digest, "written": "unchanged"}
        raise ArtifactError(
            f"目标已存在且内容不同，拒绝覆盖：{resolved}"
            f"（existing={sha256_bytes(existing)[:12]} new={digest[:12]}）"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    recorded = write_json(resolved, dict(payload))
    if recorded != digest:
        raise ArtifactError(f"落盘 SHA 与内存字节不一致：{resolved}")
    return {"path": str(resolved), "sha256": digest, "written": "created"}


def copy_verbatim(source: Path, target: Path, artifact_base: Path | None = None) -> dict[str, str]:
    """按字节复制并核验（协议快照等原件）。"""
    resolved = assert_within_write_roots(target, write_roots(artifact_base))
    data = Path(source).read_bytes()
    digest = sha256_bytes(data)
    if resolved.exists():
        if resolved.read_bytes() == data:
            return {"path": str(resolved), "sha256": digest, "written": "unchanged"}
        raise ArtifactError(f"快照目标已存在且字节不同，拒绝覆盖：{resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(data)
    return {"path": str(resolved), "sha256": digest, "written": "created"}


def file_digests(directory: Path, relative: bool = True) -> dict[str, str]:
    """目录下全部文件的相对路径 -> 字节 SHA256（稳定排序）。"""
    if not directory.exists():
        return {}
    items: list[tuple[str, Path]] = []
    for path in directory.rglob("*"):
        if path.is_file():
            key = path.relative_to(directory).as_posix() if relative else path.as_posix()
            items.append((key, path))
    return {key: sha256_file(path) for key, path in sorted(items)}


def seal_stage(
    stage_dir: Path,
    stage: str,
    payload: Mapping[str, Any],
    *,
    exclude: Sequence[str] = ("manifest.json",),
    artifact_base: Path | None = None,
) -> dict[str, Any]:
    """为阶段目录写自封存 manifest（逐文件 SHA + 账载 payload）。"""
    resolved = assert_within_write_roots(stage_dir, write_roots(artifact_base))
    files = file_digests(resolved)
    for name in exclude:
        files.pop(name, None)
    files = {key: value for key, value in files.items() if not key.startswith("artifacts/")}
    manifest = {
        "stage": stage,
        "generated_at": utc_now(),
        "files": files,
        "payload": dict(payload),
    }
    written = write_json_checked(resolved / "manifest.json", manifest, artifact_base)
    manifest["manifest_sha256"] = written["sha256"]
    return manifest


def list_run_files(run_dir: Path) -> dict[str, str]:
    """run 目录逐文件 SHA（用于交接与复跑核对）。"""
    return file_digests(run_dir)


#: 阶段 -> 该阶段产出并须封存的工件子目录（发布后不可覆盖）
STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "dataset": ("dataset", "events"),
    # 模型本体在 campaign/models（隔离 ModelRegistry 根），run 内只封存预测与折清单
    "train_predict": ("predictions",),
    "replay": ("accounts",),
    "diagnostics": ("diagnostics",),
    "report": ("reports",),
}


def publish_artifacts(
    source_root: Path,
    run_dir: Path,
    stage: str,
    names: Sequence[str],
    upstream: Mapping[str, Any],
    artifact_base: Path | None = None,
) -> dict[str, Any]:
    """把阶段工作目录原子发布到 `<run>/artifacts/<name>` 并登记逐文件 SHA。"""
    roots = write_roots(artifact_base)
    resolved_run = assert_within_write_roots(run_dir, roots)
    target = resolved_run / "artifacts"
    if target.exists():
        raise ArtifactError(f"该 run 已发布过工件，拒绝覆盖：{target}")
    staging = resolved_run / ".staging" / "artifacts"
    if staging.exists():
        raise ArtifactError(f"暂存目录已存在（上次发布未清理）：{staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name in names:
        source = Path(source_root) / name
        if not source.exists():
            raise ArtifactError(f"阶段 {stage} 缺少工作目录：{source}")
        destination = staging / name
        shutil.copytree(source, destination)
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                key = f"artifacts/{name}/{path.relative_to(destination).as_posix()}"
                files[key] = sha256_file(path)
    staging.rename(target)
    shutil.rmtree(resolved_run / ".staging", ignore_errors=True)
    return {"files": files, "upstream": dict(upstream), "published_at": utc_now()}


def require_published(camp: Path, run_id: str, name: str) -> Path:
    """校验上游 run 的发布件存在且逐文件 SHA 相符，返回只读路径。"""
    run_dir = Path(camp) / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise ArtifactError(f"上游 run 缺 manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = manifest.get("payload", {})
    files = payload.get("artifacts", {}).get("files", {})
    prefix = f"artifacts/{name}/"
    entries = {key: value for key, value in files.items() if key.startswith(prefix)}
    if not entries:
        raise ArtifactError(f"上游 run {run_id} 未发布 {name}")
    for relative, digest in entries.items():
        path = run_dir / relative
        if not path.exists():
            raise ArtifactError(f"上游发布件缺失：{path}")
        if sha256_file(path) != digest:
            raise ArtifactError(f"上游发布件内容漂移：{path}")
    return run_dir / "artifacts" / name


def assert_under(path: Path, root: Path) -> None:
    """路径必须严格位于 root 之下（用于暂存/清理范围自检）。"""
    resolved = Path(path).resolve()
    base = Path(root).resolve()
    if resolved == base or not str(resolved).startswith(str(base) + os.sep):
        raise ArtifactError(f"路径不在允许范围内：{resolved} 不属于 {base}")


def summarize_inputs(files: Iterable[str]) -> dict[str, Any]:
    """文件清单摘要（数量 + 路径列表）。"""
    items = sorted(files)
    return {"count": len(items), "items": items}
