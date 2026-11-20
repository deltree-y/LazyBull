# -*- coding: utf-8 -*-
"""隔离研究件的公共设施：写域守卫、SHA、原子发布与 manifest。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import spec


class ResearchError(RuntimeError):
    """研究件拒绝继续（身份、写域、上游或输入不合规）。"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha(rel: str) -> str:
    path = spec.REPO_ROOT / rel
    if not path.exists():
        raise ResearchError(f"代码件缺失：{rel}")
    return sha256_file(path)


def protocol_sha() -> str:
    return code_sha(spec.PROTOCOL_REL)


def research_root() -> Path:
    return spec.REPO_ROOT / spec.OUTPUT_REL


def run_root(campaign: str, run_id: str) -> Path:
    return research_root() / campaign / "runs" / run_id


def _guard(path: Path) -> Path:
    """只允许写研究根与 temp/logs；绝不触碰生产数据区。"""
    resolved = path.resolve()
    allowed = [
        research_root().resolve(),
        (spec.REPO_ROOT / spec.TEMP_REL).resolve(),
        (spec.REPO_ROOT / spec.LOG_REL).resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise ResearchError(f"写域拒绝：{resolved} 不在研究根/temp/logs 内")
    return resolved


def write_json(path: Path, payload: Any) -> str:
    target = _guard(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    target.write_text(text + "\n", encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_parquet(frame: Any, path: Path) -> str:
    target = _guard(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return sha256_file(target)


@dataclass
class StageContext:
    """一次 stage 运行的写域、上游与计时。"""

    campaign: str
    stage: str
    run_id: str
    upstream: Sequence[str] = ()
    started: float = field(default_factory=time.perf_counter)
    notes: list[str] = field(default_factory=list)

    @property
    def run_dir(self) -> Path:
        return run_root(self.campaign, self.run_id)

    @property
    def staging_dir(self) -> Path:
        return _guard(self.run_dir.parent / f".staging_{self.run_id}_{int(self.started)}")

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def require_upstream(self) -> list[dict[str, Any]]:
        """按阶段校验直接依赖，并读取父 manifest 的 SHA。"""
        need = spec.REQUIRED_UPSTREAM[self.stage]
        got: dict[str, str] = {}
        for run_id in self.upstream:
            stage = str(run_id).split("_", 1)[0]
            if stage in got:
                raise ResearchError(f"上游阶段重复：{stage}")
            got[stage] = str(run_id)
        missing = [stage for stage in need if stage not in got]
        if missing:
            raise ResearchError(f"{self.stage} 缺少直接上游：{missing}")
        extra = [stage for stage in got if stage not in need]
        if extra:
            raise ResearchError(f"{self.stage} 不接受这些上游：{extra}")
        rows = []
        for stage in need:
            run_id = got[stage]
            manifest = run_root(self.campaign, run_id) / "manifest.json"
            if not manifest.exists():
                raise ResearchError(f"上游 manifest 缺失：{manifest}")
            digest = sha256_file(manifest)
            payload = read_json(manifest)
            if payload.get("stage") != stage:
                raise ResearchError(f"上游 {run_id} 的 stage 字段不是 {stage}")
            rows.append(
                {"stage": stage, "run_id": run_id, "manifest_sha256": digest}
            )
        return rows

    def finish(self, extras: Mapping[str, Any]) -> dict[str, Any]:
        """原子发布 staging 目录并写 manifest（已存在即拒绝覆盖）。

        调用方须先把全部产物写入 ``staging_dir``；本方法只补 manifest 再整体改名。
        """
        staged = self.staging_dir
        if not staged.exists():
            raise ResearchError(f"staging 目录不存在：{staged}")
        if self.run_dir.exists():
            raise ResearchError(f"run 已存在，拒绝覆盖：{self.run_dir}")
        payload = {
            "schema": "p7_peer_gap_manifest/v1",
            "study": spec.STUDY,
            "campaign": self.campaign,
            "stage": self.stage,
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "wall_clock_seconds": round(self.elapsed, 3),
            "protocol": spec.PROTOCOL_REL,
            "protocol_sha256": protocol_sha(),
            "spec_sha256": code_sha("scripts/v2_p7/peer_return_gap/spec.py"),
            "package_sha256": package_sha(),
            "upstream": self.require_upstream(),
            "notes": list(self.notes),
        }
        payload.update(dict(extras))
        digest = write_json(staged / "manifest.json", payload)
        staged.rename(self.run_dir)
        published = read_json(self.run_dir / "manifest.json")
        published["manifest_sha256"] = digest
        return published

    def cleanup(self) -> None:
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir, ignore_errors=True)

    def preserve_failure(self) -> str | None:
        """失败现场整体保留：staging 改名为 failed_<run_id>，不删除。"""
        if not self.staging_dir.exists():
            return None
        target = self.run_dir.parent / f"failed_{self.run_id}"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        self.staging_dir.rename(target)
        return str(target)


def package_sha() -> dict[str, str]:
    """研究包内每个 Python 件的 SHA（不含 __pycache__）。"""
    root = Path(__file__).resolve().parent
    return {
        path.name: sha256_file(path)
        for path in sorted(root.glob("*.py"))
        if path.name != "__pycache__"
    }


def round12(value: float) -> float:
    """十进制 12 位四舍六入五成双（协议 §3/§4 的排序键口径）。"""
    from decimal import Decimal, ROUND_HALF_EVEN

    if value is None:
        raise ResearchError("round12 收到空值")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ResearchError(f"round12 收到非有限值：{value!r}")
    return float(Decimal(repr(number)).quantize(Decimal("1e-12"), rounding=ROUND_HALF_EVEN))


def sorted_by_score(rows: Iterable[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    """分数降序 + 代码升序；排序键按 12 位五成双。"""
    material = [dict(row) for row in rows]
    return sorted(material, key=lambda row: (-round12(float(row[key])), str(row["ts_code"])))


def log_stage(stage: str, message: str) -> None:
    log_dir = spec.REPO_ROOT / spec.LOG_REL
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with (log_dir / f"p7_peer_gap_{stage}.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def ensure_clean_run(ctx: StageContext) -> None:
    if ctx.run_dir.exists():
        raise ResearchError(f"run 已存在：{ctx.run_dir}（实现修改请换新 run_id）")
    if os.environ.get("PYTHONHASHSEED") not in (None, "0"):
        ctx.notes.append(f"PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}")
