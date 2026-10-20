# -*- coding: utf-8 -*-
"""P3 pre T2：显式迁移 v2 manifest 行域与审计字段（默认 dry-run）。

规范用法（备份根唯一口径）：
    python -X utf8 scripts/v2_p3_pre/migrate_store_manifest.py \
      --data-root data --backup-root data/backup/v2_p3_pre/<RUN_ID>
    python -X utf8 scripts/v2_p3_pre/migrate_store_manifest.py \
      --data-root data --backup-root data/backup/v2_p3_pre/<RUN_ID> --execute

dry-run 只读并打印计划；execute 先保存同代快照，再仅修改 manifest。旧代码禁止写新
manifest/WAL；代码回滚前必须恢复匹配的 manifest、WAL/checkpoint 和活动文件状态。
"""

from __future__ import annotations

import argparse
import copy
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
from src.lazybull.v2.store.manifest import (  # noqa: E402
    Manifest,
    content_fingerprint_of,
)
from src.lazybull.v2.store.row_domain import migrate_manifest_row_domains  # noqa: E402

MIGRATION_ID = "p3_pre_row_domain_v1"
_EXTENSION_LISTS = ("label_refresh_intents", "label_refreshes", "schema_migrations")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "data"),
        help="v2 store 数据根；跨 cwd 使用相对 evidence_path 时须传绝对路径",
    )
    parser.add_argument(
        "--backup-root",
        required=True,
        help="同代快照根，必须为 data/backup/v2_p3_pre/<RUN_ID>/ 形式的目标目录",
    )
    parser.add_argument("--execute", action="store_true", help="显式执行迁移；缺省为只读 dry-run")
    return parser.parse_args(argv)


def _reject_pending_wal(wal_path: Path) -> None:
    if wal_path.exists() and wal_path.stat().st_size > 0:
        raise RuntimeError(f"开工前存在未处理 WAL，禁止混入新迁移 run: {wal_path}")


def _report_path(data_root: Path, backup_root: Path) -> Path:
    """新迁移证据统一使用绝对路径，避免后续 cwd 与相对根歧义。"""
    return (
        data_root / "reports" / f"v2_p3_pre_manifest_migration_{backup_root.name}.json"
    ).resolve()


def _resolve_evidence_path(value: str, data_root: Path, report_path: Path) -> Path:
    """兼容可证明同源的旧相对证据；歧义路径不得猜测拼接。"""
    path = Path(value)
    if path.is_absolute():
        return path
    legacy_candidate = data_root / "reports" / report_path.name
    if path == legacy_candidate:
        return legacy_candidate.resolve()
    if path.parts and path.parts[0] == data_root.name:
        return data_root.resolve().parent / path
    raise RuntimeError(
        f"相对 evidence_path 无法确定原数据根，拒绝解析: {value!r}（data_root={data_root}）"
    )


def _registered_fingerprints(snapshot: dict[str, Any]) -> dict[str, str]:
    result = {rel_path: str(meta["sha256_16"]) for rel_path, meta in snapshot["partitions"].items()}
    for label_name, entry in snapshot["labels"].items():
        for filename, meta in entry["partitions"].items():
            result[f"labels/{label_name}/{filename}"] = str(meta["sha256_16"])
    return result


def _migration_record(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    records = [
        r for r in snapshot.get("schema_migrations", []) if r.get("migration_id") == MIGRATION_ID
    ]
    if len(records) > 1:
        raise ValueError(f"迁移审计 {MIGRATION_ID} 重复 {len(records)} 条")
    return records[0] if records else None


def _plan_for_manifest(
    manifest: Manifest, *, backup_root: Path, report_path: Path
) -> dict[str, Any]:
    """基于当前内存 manifest 生成确定性的迁移计划，不写盘。"""
    snapshot = manifest.snapshot()
    input_fp = manifest.content_fingerprint()
    existing = _migration_record(snapshot)
    base: dict[str, Any] = {
        "migration_id": MIGRATION_ID,
        "data_root": str(manifest.path.parent.parent),
        "backup_root": str(backup_root),
        "report_path": str(report_path),
        "input_fingerprint": input_fp,
        "existing_migration_audit": existing is not None,
        "snapshot_files": [
            str(manifest.path),
            str(manifest.path.parent / "manifest_wal.jsonl"),
            str(manifest.backup_path),
        ],
        "old_writer_warning": "旧代码禁止写新 manifest/WAL；回滚前须恢复同代 manifest/WAL/文件",
    }
    if existing is not None:
        counts = manifest.validate_row_domain_readiness()
        missing_lists = [key for key in _EXTENSION_LISTS if key not in snapshot]
        if missing_lists:
            raise ValueError(f"迁移审计已存在但扩展字段缺失 {missing_lists}")
        evidence_path = _resolve_evidence_path(
            str(existing["evidence_path"]), manifest.path.parent.parent, report_path
        )
        report_missing = not evidence_path.exists()
        return {
            **base,
            "status": "report_recovery_planned" if report_missing else "no_op",
            "report_path": str(evidence_path),
            "migrate_counts": {"panel": 0, "labels": 0},
            "readiness_counts": counts,
            "expected_output_fingerprint": existing["output_fingerprint"],
        }

    simulated = copy.deepcopy(snapshot)
    had_segments = "row_domain_segments" in simulated
    had_lists = tuple(key in simulated for key in _EXTENSION_LISTS)
    counts = migrate_manifest_row_domains(simulated)
    for key in _EXTENSION_LISTS:
        simulated.setdefault(key, [])
    changed = any(counts.values()) or not had_segments or not all(had_lists)
    if not changed:
        readiness = manifest.validate_row_domain_readiness()
        return {
            **base,
            "status": "no_op_current_without_migration",
            "migrate_counts": counts,
            "readiness_counts": readiness,
            "expected_output_fingerprint": input_fp,
        }
    return {
        **base,
        "status": "planned",
        "migrate_counts": counts,
        "expected_output_fingerprint": content_fingerprint_of(simulated),
        "registered_partition_count": len(_registered_fingerprints(snapshot)),
        "repairs_count": len(snapshot.get("repairs", [])),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _save_run_snapshot(
    store: PanelDataStore, backup_root: Path, *, source_fingerprint: str
) -> dict[str, Any]:
    """在写锁内、任何迁移 mutation 前保存 manifest/WAL/checkpoint 同代现场。"""
    if backup_root.exists() and any(backup_root.iterdir()):
        raise RuntimeError(f"备份目录非空，拒绝覆盖既有 run 快照: {backup_root}")
    backup_root.mkdir(parents=True, exist_ok=True)
    manifest_path = store.features_dir / "manifest.json"
    copied: list[str] = []
    for source in (manifest_path, store.manifest.backup_path, store.wal_path):
        if source.exists():
            shutil.copy2(source, backup_root / source.name)
            copied.append(source.name)
    checkpoint = {
        "wal_present": store.wal_path.exists(),
        "wal_size": store.wal_path.stat().st_size if store.wal_path.exists() else 0,
        "empty_checkpoint": not store.wal_path.exists() or store.wal_path.stat().st_size == 0,
        "copied_files": copied,
        "affected_existing_files": ["features/manifest.json", "features/manifest.json.bak"],
        "planned_new_files": ["features/manifest.json.bak"],
        "affected_parquet_files": [],
    }
    _write_json_atomic(backup_root / "checkpoint.json", checkpoint)
    snapshot = store.manifest.snapshot()
    _write_json_atomic(
        backup_root / "snapshot.json",
        {
            "source_content_fingerprint": source_fingerprint,
            "registered_fingerprints": _registered_fingerprints(snapshot),
            "repairs": snapshot.get("repairs", []),
        },
    )
    return checkpoint


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"迁移报告恢复证据读取失败: {path}（{exc}）") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"迁移报告恢复证据不是 JSON 对象: {path}")
    return data


def _recover_missing_report(
    manifest: Manifest, backup_root: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    """凭原 run 同代证据补写缺失报告；终态有任何后续漂移都拒绝自动补证。"""
    snapshot = manifest.snapshot()
    audit = _migration_record(snapshot)
    evidence_path = (
        _resolve_evidence_path(
            audit["evidence_path"], manifest.path.parent.parent, Path(plan["report_path"])
        )
        if audit is not None
        else None
    )
    if evidence_path is None or evidence_path != Path(plan["report_path"]):
        raise RuntimeError("迁移报告恢复必须使用原审计 evidence_path 与同代 run 快照")
    original_path = backup_root / "manifest.json"
    snapshot_path = backup_root / "snapshot.json"
    checkpoint_path = backup_root / "checkpoint.json"
    original = _read_json_object(original_path)
    run_snapshot = _read_json_object(snapshot_path)
    checkpoint = _read_json_object(checkpoint_path)
    actual_source_fp = content_fingerprint_of(original)
    if actual_source_fp != audit["input_fingerprint"]:
        raise RuntimeError("原始 manifest 输入指纹与迁移审计不一致，拒绝补写报告")
    if run_snapshot.get("source_content_fingerprint") != actual_source_fp:
        raise RuntimeError("原 run 快照声明与原始 manifest 实际输入指纹不一致，拒绝补写报告")
    if _migration_record(original) is not None:
        raise RuntimeError("原 run manifest 已含迁移审计，无法作为迁移前证据")

    current_fps = _registered_fingerprints(snapshot)
    if current_fps != run_snapshot.get("registered_fingerprints"):
        raise RuntimeError("当前登记指纹已偏离原迁移终态，拒绝自动补写报告")
    if snapshot.get("repairs", []) != run_snapshot.get("repairs"):
        raise RuntimeError("当前 repairs 已偏离原迁移终态，拒绝自动补写报告")
    migrated_counts = migrate_manifest_row_domains(original)
    for key in _EXTENSION_LISTS:
        original.setdefault(key, [])
    if migrated_counts != audit["migrate_counts"]:
        raise RuntimeError("原始 manifest 迁移计数与审计不一致，拒绝补写报告")
    actual_output_fp = content_fingerprint_of(original)
    if actual_output_fp != audit["output_fingerprint"]:
        raise RuntimeError("迁移后输出指纹与审计不一致，拒绝补写报告")
    original["schema_migrations"].append(copy.deepcopy(audit))
    expected_final = content_fingerprint_of(original)
    if manifest.content_fingerprint() != expected_final:
        raise RuntimeError("当前 manifest 已偏离原迁移终态，拒绝自动补写报告")

    report = {
        **plan,
        "mode": "execute",
        "status": "report_recovered",
        "recovered": True,
        "input_fingerprint": audit["input_fingerprint"],
        "migrate_counts": audit["migrate_counts"],
        "registered_partition_count": len(current_fps),
        "repairs_count": len(snapshot.get("repairs", [])),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_fingerprint": audit["output_fingerprint"],
        "final_manifest_fingerprint": expected_final,
        "registered_fingerprints_unchanged": True,
        "repairs_unchanged": True,
        "checkpoint": checkpoint,
    }
    _write_json_atomic(Path(plan["report_path"]), report)
    return report


def migrate_manifest(
    data_root: Path | str, backup_root: Path | str, *, execute: bool = False
) -> dict:
    """执行 dry-run 或显式迁移；返回可 JSON 序列化的计划/报告。"""
    root = Path(data_root)
    backup = Path(backup_root)
    manifest_path = root / "features" / "manifest.json"
    wal_path = root / "features" / "manifest_wal.jsonl"
    report_path = _report_path(root, backup)
    if not execute:
        _reject_pending_wal(wal_path)
        plan = _plan_for_manifest(
            Manifest(manifest_path), backup_root=backup, report_path=report_path
        )
        plan["mode"] = "dry_run"
        return plan

    store = PanelDataStore(root)
    with store.write_lock():
        _reject_pending_wal(wal_path)
        manifest = store.manifest
        plan = _plan_for_manifest(manifest, backup_root=backup, report_path=report_path)
        if plan["status"] == "report_recovery_planned":
            return _recover_missing_report(manifest, backup, plan)
        if plan["status"].startswith("no_op"):
            plan["mode"] = "execute"
            return plan
        before_snapshot = manifest.snapshot()
        before_fps = _registered_fingerprints(before_snapshot)
        before_repairs = copy.deepcopy(before_snapshot.get("repairs", []))
        checkpoint = _save_run_snapshot(store, backup, source_fingerprint=plan["input_fingerprint"])
        counts = manifest.migrate_row_domain_metadata()
        if counts != plan["migrate_counts"]:
            raise RuntimeError(f"迁移计划与实际计数漂移: {counts} ≠ {plan['migrate_counts']}")
        output_fp = manifest.content_fingerprint()
        if output_fp != plan["expected_output_fingerprint"]:
            raise RuntimeError("迁移输出指纹与 dry-run 口径不一致")
        manifest.register_schema_migration(
            {
                "migration_id": MIGRATION_ID,
                "run_id": backup.name,
                "input_fingerprint": plan["input_fingerprint"],
                "output_fingerprint": output_fp,
                "evidence_path": str(report_path),
                "migrate_counts": counts,
            }
        )
        manifest.save()
        readiness = manifest.validate_row_domain_readiness()
        after_snapshot = manifest.snapshot()
        fingerprints_unchanged = before_fps == _registered_fingerprints(after_snapshot)
        repairs_unchanged = before_repairs == after_snapshot.get("repairs", [])
        if not fingerprints_unchanged or not repairs_unchanged:
            raise RuntimeError("迁移后 parquet 登记指纹或 repairs 发生变化")
        report = {
            **plan,
            "mode": "execute",
            "status": "executed",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "output_fingerprint": output_fp,
            "final_manifest_fingerprint": manifest.content_fingerprint(),
            "readiness_counts": readiness,
            "registered_fingerprints_unchanged": fingerprints_unchanged,
            "repairs_unchanged": repairs_unchanged,
            "checkpoint": checkpoint,
        }
        _write_json_atomic(report_path, report)
        return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = migrate_manifest(args.data_root, args.backup_root, execute=args.execute)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
