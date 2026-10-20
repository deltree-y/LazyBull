# -*- coding: utf-8 -*-
"""v2 manifest 显式恢复工具（默认 dry-run；真实执行须单独授权）。

只在主 manifest 缺失/无效、上一代 `.bak` 自身有效、活动 WAL 为空、bak 登记文件
指纹一致且盘上无候选未登记最终 parquet 时，才允许 `--execute`。主有效但文件不一致、
陈旧 bak、WAL 交错、主/bak 双坏或任何登记↔文件集合不一致均 fail-closed；不自动空骨架
重建、不自动 resync，也不以 parquet 内容猜测登记。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.manifest import Manifest, sha256_16_of_file  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="v2 store 数据根")
    parser.add_argument(
        "--backup-root",
        required=True,
        help="故障现场与恢复报告目录（建议使用 data/backup/v2_p3_pre/<RUN_ID>/）",
    )
    parser.add_argument("--execute", action="store_true", help="显式执行恢复；缺省为只读 dry-run")
    return parser.parse_args(argv)


def _partition_files(snapshot: dict[str, Any], data_root: Path):
    for rel_path, meta in snapshot["partitions"].items():
        yield rel_path, data_root / "features" / rel_path, str(meta["sha256_16"])
    for label_name, entry in snapshot["labels"].items():
        for filename, meta in entry["partitions"].items():
            rel_path = f"labels/{label_name}/{filename}"
            yield rel_path, data_root / "labels" / label_name / filename, str(meta["sha256_16"])


def _actual_partition_rel_paths(data_root: Path) -> set[str]:
    """枚举活动最终 parquet；只用于发现未登记文件，不据此猜测或补登记。"""
    actual: set[str] = set()
    features = data_root / "features"
    for directory in (features / "panel", features / "panel_archive"):
        if directory.exists():
            actual.update(
                path.relative_to(features).as_posix()
                for path in directory.rglob("*.parquet")
                if path.is_file()
            )
    labels = data_root / "labels"
    if labels.exists():
        actual.update(
            path.relative_to(data_root).as_posix()
            for path in labels.glob("*/*.parquet")
            if path.is_file()
        )
    return actual


def _inspect_candidate(path: Path, data_root: Path) -> dict[str, Any]:
    """独立校验主/bak JSON、schema、指纹及登记↔活动文件双向集合。"""
    report: dict[str, Any] = {"path": str(path), "exists": path.exists(), "valid": False}
    if not path.exists():
        report["error"] = "文件缺失"
        return report
    try:
        manifest = Manifest(path)
    except (OSError, ValueError) as exc:
        report["error"] = str(exc)
        return report
    snapshot = manifest.snapshot()
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    checked = 0
    registered: set[str] = set()
    for rel_path, file_path, expected in _partition_files(snapshot, data_root):
        registered.add(rel_path)
        checked += 1
        if not file_path.exists():
            missing.append(rel_path)
            continue
        actual = sha256_16_of_file(file_path)
        if actual != expected:
            mismatched.append({"rel_path": rel_path, "registered": expected, "actual": actual})
    untracked = sorted(_actual_partition_rel_paths(data_root) - registered)
    report.update(
        {
            "valid": True,
            "content_fingerprint": manifest.content_fingerprint(),
            "partition_files_checked": checked,
            "missing_files": missing,
            "fingerprint_mismatches": mismatched,
            "untracked_files": untracked,
            "files_consistent": not missing and not mismatched and not untracked,
        }
    )
    return report


def build_recovery_plan(data_root: Path | str, backup_root: Path | str) -> dict[str, Any]:
    """只读生成恢复计划；不创建备份目录、报告或锁文件。"""
    root = Path(data_root)
    backup = Path(backup_root)
    manifest_path = root / "features" / "manifest.json"
    bak_path = manifest_path.with_suffix(".json.bak")
    wal_path = root / "features" / "manifest_wal.jsonl"
    main = _inspect_candidate(manifest_path, root)
    bak = _inspect_candidate(bak_path, root)
    wal_pending = wal_path.exists() and wal_path.stat().st_size > 0
    plan: dict[str, Any] = {
        "mode": "dry_run",
        "data_root": str(root),
        "backup_root": str(backup),
        "main": main,
        "bak": bak,
        "wal": {
            "path": str(wal_path),
            "present": wal_path.exists(),
            "size": wal_path.stat().st_size if wal_path.exists() else 0,
            "pending": wal_pending,
        },
        "execute_allowed": False,
    }
    if main.get("valid"):
        if main.get("files_consistent"):
            plan["status"] = "main_valid_no_recovery"
            plan["reason"] = (
                "主 manifest 与活动文件一致，不执行回退；坏 bak 可由下一次正常主提交重建"
            )
        else:
            plan["status"] = "manual_required_inconsistent_main"
            plan["reason"] = "主 manifest 有效但活动文件不一致，禁止以 bak 覆盖并掩盖差异"
        return plan
    if not bak.get("valid"):
        plan["status"] = "no_valid_backup"
        plan["reason"] = "主文件不可用且 bak 缺失或损坏，禁止自动恢复"
        return plan
    if wal_pending:
        plan["status"] = "manual_required_wal"
        plan["reason"] = "活动 WAL 非空，需人工归因同代关系，禁止自动 replay 或压实"
        return plan
    if bak.get("untracked_files"):
        plan["status"] = "manual_required_untracked_files"
        plan["reason"] = "盘上存在候选 bak 未登记的最终 parquet，禁止恢复成无保护孤儿文件"
        return plan
    if not bak.get("files_consistent"):
        plan["status"] = "manual_required_stale_backup"
        plan["reason"] = "bak 登记与当前活动文件不一致，禁止自动 resync 或覆盖"
        return plan
    plan.update(
        {
            "status": "recovery_planned",
            "execute_allowed": True,
            "reason": "主文件不可用；bak 有效、空 WAL/checkpoint 且全部登记文件指纹一致",
            "restore": {"source": str(bak_path), "target": str(manifest_path)},
        }
    )
    return plan


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _quarantine(plan: dict[str, Any]) -> Path:
    backup = Path(plan["backup_root"])
    if backup.exists() and any(backup.iterdir()):
        raise RuntimeError(f"恢复备份目录非空，拒绝覆盖既有现场: {backup}")
    quarantine = backup / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=False)
    for key, path_str in (
        ("main", plan["main"]["path"]),
        ("bak", plan["bak"]["path"]),
        ("wal", plan["wal"]["path"]),
    ):
        source = Path(path_str)
        if source.exists():
            shutil.copy2(source, quarantine / source.name)
    _write_json_atomic(
        backup / "checkpoint.json",
        {
            "main": plan["main"],
            "bak": plan["bak"],
            "wal": plan["wal"],
            "quarantine": str(quarantine),
        },
    )
    _write_json_atomic(backup / "recovery_plan.json", plan)
    return quarantine


def recover_manifest(
    data_root: Path | str, backup_root: Path | str, *, execute: bool = False
) -> dict:
    """返回 dry-run 计划；execute 在同一写锁内核验、保存现场、恢复与复验。"""
    if not execute:
        return build_recovery_plan(data_root, backup_root)
    store = PanelDataStore(data_root)
    with store.write_lock():
        plan = build_recovery_plan(data_root, backup_root)
        if not plan["execute_allowed"]:
            raise RuntimeError(f"恢复预检未通过（{plan['status']}）: {plan['reason']}")
        quarantine = _quarantine(plan)
        source = Path(plan["restore"]["source"])
        target = Path(plan["restore"]["target"])
        tmp = target.with_suffix(".json.tmp")
        tmp.write_bytes(source.read_bytes())
        Manifest(tmp)  # 恢复内容再次独立做 JSON/schema 校验
        os.replace(tmp, target)
        restored = Manifest(target)
        restored_fp = restored.content_fingerprint()
        if restored_fp != plan["bak"]["content_fingerprint"]:
            raise RuntimeError(
                f"恢复后内容指纹不符: {restored_fp} ≠ {plan['bak']['content_fingerprint']}"
            )
        report = {
            **plan,
            "mode": "execute",
            "status": "recovered",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "quarantine": str(quarantine),
            "restored_content_fingerprint": restored_fp,
            "wal_replay": "空 WAL/checkpoint 仅复制留证；无故障事件可重放",
        }
        _write_json_atomic(Path(backup_root) / "recovery_report.json", report)
        return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = recover_manifest(args.data_root, args.backup_root, execute=args.execute)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
