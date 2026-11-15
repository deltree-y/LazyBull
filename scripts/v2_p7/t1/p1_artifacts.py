# -*- coding: utf-8 -*-
"""P-1 重开证据发布：原子发布、manifest、逐文件 SHA 与复算。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """写 JSON（UTF-8、缩进 2、末尾换行），返回字节 SHA。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_bytes(text.encode("utf-8"))


def write_text(path: Path, text: str) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256_bytes(text.encode("utf-8"))


def collect_files(root: Path) -> Dict[str, str]:
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def publish(partial: Path, target: Path) -> None:
    """原子发布：目标已存在则拒绝覆盖（旧证据根只读保留）。"""
    if Path(target).exists():
        raise FileExistsError(f"证据根已存在（禁止覆盖）：{target}")
    Path(partial).rename(Path(target))


def cleanup_partial(partial: Path) -> None:
    shutil.rmtree(Path(partial), ignore_errors=True)


def manifest_payload(
    *,
    run_id: str,
    params: Any,
    code_state: Mapping[str, Any],
    data_state: Mapping[str, Any],
    written: Mapping[str, str],
    elapsed_seconds: float,
    smoke: Mapping[str, Any],
    prereg: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": "p7_p1r_manifest/v1",
        "run_id": run_id,
        "task": "P7 P-1 重开（只许否决的筛查门）",
        "evidence_level": "H",
        "prereg_id": params.prereg_id,
        "prereg_sha256": params.prereg_sha256,
        "criteria_digest": params.criteria_digest,
        "prereg_status": prereg.get("status"),
        "params": params.params_payload(),
        "code_state": dict(code_state),
        "data_state": dict(data_state),
        "smoke_limits": dict(smoke),
        "diagnostic_only": bool(smoke.get("diagnostic_only", False)),
        "written_files_sha256": dict(written),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "not_forward_evidence": True,
        "not_real_clock_evidence": True,
        "not_production_authorization": True,
    }


def verify_payload(target: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """复算：逐文件 SHA 与 manifest 对照，返回差异清单（空 = 全等）。"""
    recorded = dict(manifest.get("written_files_sha256") or {})
    actual = collect_files(Path(target))
    actual.pop("manifest.json", None)
    actual.pop("verify.json", None)
    mismatches = []
    for name, digest in recorded.items():
        if name in ("manifest.json", "verify.json"):
            continue
        if actual.get(name) != digest:
            mismatches.append({"file": name, "recorded": digest, "actual": actual.get(name)})
    missing = sorted(set(recorded) - set(actual) - {"manifest.json", "verify.json"})
    extra = sorted(set(actual) - set(recorded))
    return {
        "schema": "p7_p1r_verify/v1",
        "checked_files": len(recorded),
        "mismatches": mismatches,
        "missing": missing,
        "extra": extra,
        "all_equal": not mismatches and not missing and not extra,
    }


def iter_written(written: Mapping[str, str]) -> Iterable[tuple[str, str]]:
    return sorted(written.items())


def replace_dir(tmp: Path, final: Path) -> None:
    """内部用：仅在调用方已确认 final 不存在时使用。"""
    if final.exists():
        raise FileExistsError(f"目标已存在：{final}")
    os.replace(tmp, final)
