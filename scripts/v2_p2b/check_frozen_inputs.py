# -*- coding: utf-8 -*-
"""P2b 门 1：生产模型实际选择路径与两类冻结参照身份核验。

模型文件选择复用生产 ``ModelRegistry.load_model``，仅以替身替换反序列化；
不写 metadata、不另写路径回退规则、不静默重采清单。旧 B0 raw 的每个
glob 必须恰好命中一个文件，缺失或多候选均 fail-fast。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence
from unittest.mock import patch

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.lazybull.ml.model_registry as model_registry_module
from scripts.v2_p2a.compare_replay_vs_b0 import RawReferenceFiles
from scripts.v2_p2a.input_manifest import MODEL_VERSION_RANGE, _sha256_file

DEFAULT_MANIFEST = "data/frozen_reference/p2a_b0_input_manifest_20261008_withconfig.json"
DEFAULT_P2A_RUNS = "data/runs/p2a_b0replay_20261008_144307_d79f3f"
DEFAULT_B0_RAW = "data/walk_forward/batches/wf_batch_20260930_171221/raw"


def probe_model_paths(
    models_dir: Path,
    versions: Sequence[int] = tuple(MODEL_VERSION_RANGE),
) -> Dict[int, Dict[str, Path]]:
    """经生产加载逻辑记录每版实际模型与 features 路径，反序列化替身不落盘。"""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {models_dir}")
    selected: Dict[int, Dict[str, Path]] = {}
    active: List[int] = []

    def _joblib_stub(path):
        version = active[-1]
        selected[version]["model_path"] = Path(path).resolve()
        return object()

    def _native_stub(path, _model_type):
        version = active[-1]
        selected[version]["model_path"] = Path(path).resolve()
        return object()

    with (
        patch.object(model_registry_module.joblib, "load", new=_joblib_stub),
        patch.object(model_registry_module, "_load_xgboost_native", new=_native_stub),
        patch.object(
            model_registry_module.ModelRegistry,
            "_save_metadata_sidecar",
            new=lambda self, metadata: None,
        ),
        patch.object(
            model_registry_module.ModelRegistry,
            "_save_latest_version_file",
            new=lambda self, version: None,
        ),
    ):
        registry = model_registry_module.ModelRegistry(str(models_dir))
        for version in versions:
            selected[version] = {}
            active.append(version)
            try:
                _, metadata = registry.load_model(int(version))
            finally:
                active.pop()
            if "model_path" not in selected[version]:
                raise RuntimeError(f"v{version} 生产加载未记录实际模型路径")
            selected[version]["features_path"] = (models_dir / metadata["features_file"]).resolve()
    return selected


def _manifest_key(path: Path, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"实际选择路径越出 data_root: {path}") from exc


def check_model_paths_in_manifest(
    data_root: Path,
    manifest_path: Path,
    versions: Sequence[int] = tuple(MODEL_VERSION_RANGE),
) -> Dict[str, object]:
    """核验实际模型/features 均在冻结清单且逐字节身份一致。"""
    data_root = Path(data_root)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_files = manifest.get("files") or {}
    models_dir = data_root / "models" / "stock_selection"
    selected = probe_model_paths(models_dir, versions)
    errors: List[str] = []
    selected_keys: Dict[str, Dict[str, str]] = {}
    for version, paths in selected.items():
        selected_keys[str(version)] = {}
        for kind, path in paths.items():
            key = _manifest_key(path, data_root)
            selected_keys[str(version)][kind] = key
            if key not in manifest_files:
                errors.append(f"v{version} 实际 {kind} 清单外: {key}")
                continue
            actual_hash = _sha256_file(path)
            if actual_hash != manifest_files[key].get("sha256"):
                errors.append(f"v{version} 实际 {kind} 内容哈希不符: {key}")
    return {
        "ok": not errors,
        "errors": errors,
        "selected": selected_keys,
        "manifest": str(manifest_path),
    }


def resolve_unique(directory: Path, pattern: str, label: str) -> Path:
    """参照 glob 恰好一个命中；缺失与歧义使用同一显式失败边界。"""
    hits = sorted(path for path in Path(directory).glob(pattern) if path.is_file())
    if len(hits) != 1:
        raise ValueError(f"冻结参照 {label} 命中数={len(hits)}（要求=1）: {directory / pattern}")
    return hits[0]


def resolve_b0_raw_reference(
    raw_dir: Path,
    splits: Sequence[int] = tuple(range(14)),
) -> RawReferenceFiles:
    """显式解析旧 B0 raw 比较所需的固定文件集。"""
    raw_dir = Path(raw_dir)
    summary = resolve_unique(raw_dir, "walk_forward_summary_0101_0001.csv", "summary")
    chain = resolve_unique(raw_dir, "chain_nav_*.csv", "chain_nav")
    trades = {
        int(split): resolve_unique(
            raw_dir, f"walk_forward_trades_*_split{int(split):02d}.csv", f"split{split:02d}.trades"
        )
        for split in splits
    }
    attribution = {
        int(split): resolve_unique(
            raw_dir,
            f"walk_forward_execution_attribution_*_split{int(split):02d}.csv",
            f"split{split:02d}.attribution",
        )
        for split in splits
    }
    return RawReferenceFiles(
        summary=summary, chain_nav=chain, trades=trades, attribution=attribution
    )


def resolve_p2a_runs_reference(
    batch_dir: Path,
    splits: Sequence[int] = tuple(range(14)),
) -> Dict[str, Path]:
    """固定 P2a runs 参照文件集并核对目录/meta 批次身份。"""
    batch_dir = Path(batch_dir)
    meta_path = batch_dir / "batch_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"P2a runs 参照缺 batch_meta.json: {batch_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("batch_id") != batch_dir.name:
        raise ValueError(
            f"P2a runs 参照身份歧义: 目录={batch_dir.name} vs batch_meta={meta.get('batch_id')}"
        )
    files = {
        "batch_meta.json": meta_path,
        "summary.parquet": batch_dir / "summary.parquet",
        "chain_nav.parquet": batch_dir / "chain_nav.parquet",
    }
    for split in splits:
        fold_dir = batch_dir / "folds" / f"split{int(split):02d}"
        for name in ("trades", "attribution", "holdings_snapshot", "daily"):
            files[f"folds/split{int(split):02d}/{name}.parquet"] = fold_dir / f"{name}.parquet"
        files[f"folds/split{int(split):02d}/_meta.json"] = fold_dir / "_meta.json"
    missing = [key for key, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P2a runs 参照缺固定文件: {missing}")
    return files


def digest_reference_files(files: Mapping[str, Path], root: Path) -> Dict[str, object]:
    """按稳定逻辑键保存参照相对路径、逐文件 SHA256 与总 digest。"""
    root = Path(root)
    entries: Dict[str, Dict[str, str]] = {}
    for key, path in sorted(files.items()):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"参照文件缺失: {key} -> {path}")
        entries[key] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"files": entries, "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def digest_p2a_runs_reference(
    batch_dir: Path,
    splits: Sequence[int] = tuple(range(14)),
) -> Dict[str, object]:
    """采集 P2a 冻结 runs 参照摘要；默认终验锚定完整 14 折。"""
    batch_dir = Path(batch_dir)
    return digest_reference_files(resolve_p2a_runs_reference(batch_dir, splits), batch_dir)


def digest_b0_raw_reference(
    raw_dir: Path,
    reference_files: Optional[RawReferenceFiles] = None,
) -> Dict[str, object]:
    """采集旧 B0 raw 参照摘要；可复用已完成歧义检查的解析结果。"""
    raw_dir = Path(raw_dir)
    refs = reference_files or resolve_b0_raw_reference(raw_dir)
    files: Dict[str, Path] = {"summary": refs.summary, "chain_nav": refs.chain_nav}
    for split, path in refs.trades.items():
        files[f"split{split:02d}/trades"] = path
    for split, path in refs.attribution.items():
        if path is not None:
            files[f"split{split:02d}/attribution"] = path
    return digest_reference_files(files, raw_dir)


def canonical_digest_note(document: Mapping[str, object]) -> str:
    """将参照摘要转为 runs notes 所需的 canonical JSON 字符串。"""
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def check_all_frozen_inputs(
    *,
    data_root: Path,
    manifest_path: Path,
    p2a_runs_dir: Path,
    b0_raw_dir: Path,
) -> Dict[str, object]:
    """聚合模型路径与两类参照证据（不运行真实重放）。"""
    model_report = check_model_paths_in_manifest(data_root, manifest_path)
    raw_refs = resolve_b0_raw_reference(b0_raw_dir)
    return {
        "ok": model_report["ok"],
        "model_report": model_report,
        "p2a_runs": digest_p2a_runs_reference(p2a_runs_dir),
        "b0_raw": digest_b0_raw_reference(b0_raw_dir, raw_refs),
        "raw_reference_files": raw_refs,
    }


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--p2a-runs", default=DEFAULT_P2A_RUNS)
    parser.add_argument("--b0-raw", default=DEFAULT_B0_RAW)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = check_all_frozen_inputs(
        data_root=_repo_path(args.data_root),
        manifest_path=_repo_path(args.manifest),
        p2a_runs_dir=_repo_path(args.p2a_runs),
        b0_raw_dir=_repo_path(args.b0_raw),
    )
    serializable = {key: value for key, value in report.items() if key != "raw_reference_files"}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for error in report["model_report"]["errors"]:
        logger.error(error)
    logger.info(
        f"冻结输入核验 ok={report['ok']}；P2a digest={report['p2a_runs']['digest']}；"
        f"B0 raw digest={report['b0_raw']['digest']}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
