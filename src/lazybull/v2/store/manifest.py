# -*- coding: utf-8 -*-
"""v2 panel manifest（``data/features/manifest.json``）的读写、校验与登记 API。

schema（冻结物 `docs/data/v2_p1_build_freeze.md` §4）::

    {
      "manifest_version": "1",
      "updated_at": "<ISO>",
      "columns": { "<列名>": {"group": ..., "source": ..., "definition_version": "v1",
                              "available_from": "YYYYMMDD"|null, "backfilled_at": "<ISO>"|null,
                              "status": "active"|"deprecated"} },
      "partitions": { "<相对路径 panel/20260104/core.parquet>":
                      {"sha256_16": "...", "rows": N, "written_at": "<ISO>", "zone": ...,
                       "row_domain": "label_filtered"|"structural"} },
      "dependencies": {"raw_datasets": [...], "factor_functions": {...}},
      "labels": { "<label_name>": {"partitions": {...同 partitions 结构（含 row_domain）...},
                                   "sealed_through": "YYYYMMDD"|null} },
      "row_domain_segments": {"label_filtered_through": "20260702",
                              "structural_from": "20260703"},
      "label_refresh_intents": [...], "label_refreshes": [...], "schema_migrations": [...]
    }

口径要点：
- 主文件 tmp + replace 前先形成上一代有效 `.bak`；备份失败中止主提交；
- 快照返回深拷贝，防外部篡改内部状态；
- 分区内容指纹 = 文件内容 sha256 前 16 位 hex（与步骤 0 快照口径一致）；
- labels 分区键用相对标签目录的文件名（``YYYYMMDD.parquet``），panel 分区键相对 features/ 目录。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from src.lazybull.v2.store.column_groups import LABEL_TABLES
from src.lazybull.v2.store.row_domain import (
    ROW_DOMAIN_SEGMENTS,
    migrate_manifest_row_domains,
    row_domain_for_date,
    row_domain_for_partition,
    validate_manifest_row_domains,
)

__all__ = [
    "MANIFEST_VERSION",
    "VALID_GROUPS",
    "Manifest",
    "sha256_16_of_file",
    "sha256_16_of_bytes",
    "content_fingerprint_of",
    "validate_label_output_fingerprints",
    "validate_label_refresh_completion",
    "validate_label_refresh_intent",
]

MANIFEST_VERSION = "1"

#: 合法列族（冻结物 §2：panel 8 族 + candidate 候选族）
VALID_GROUPS = frozenset(
    {
        "core",
        "fundamental",
        "moneyflow",
        "technical",
        "announcement",
        "risk",
        "market_state",
        "neutralized",
        "candidate",
    }
)

_VALID_STATUS = frozenset({"active", "deprecated"})
_VALID_ZONE = frozenset({"hot", "archive"})
_INTENT_TYPES = frozenset({"prepare", "superseded", "orphan"})
_AUDIT_VOLATILE_KEYS = frozenset({"recorded_at", "run_id", "refreshed_at", "recovered"})

#: 顶层必需键（加载校验用）
_REQUIRED_TOP_KEYS = (
    "manifest_version",
    "updated_at",
    "columns",
    "partitions",
    "dependencies",
    "labels",
)
_LIST_EXTENSION_KEYS = ("label_refresh_intents", "label_refreshes", "schema_migrations")


def sha256_16_of_bytes(data: bytes) -> str:
    """内容指纹：sha256 前 16 位 hex（与步骤 0 快照口径一致）。"""
    return hashlib.sha256(data).hexdigest()[:16]


def content_fingerprint_of(data: Mapping[str, Any]) -> str:
    """对任意 manifest 快照复用唯一内容指纹口径（剔除 updated_at）。"""
    stable = {k: v for k, v in data.items() if k != "updated_at"}
    canonical = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_16_of_bytes(canonical.encode("utf-8"))


def sha256_16_of_file(path: Path) -> str:
    """文件物理字节指纹（分块读）；不作逻辑表等价放行。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_data() -> dict[str, Any]:
    """空 manifest 骨架（首次创建时用）。"""
    return {
        "manifest_version": MANIFEST_VERSION,
        "updated_at": _now_iso(),
        "columns": {},
        "partitions": {},
        "dependencies": {"raw_datasets": [], "factor_functions": {}},
        "labels": {},
        "row_domain_segments": dict(ROW_DOMAIN_SEGMENTS),
        "label_refresh_intents": [],
        "label_refreshes": [],
        "schema_migrations": [],
    }


def _validate_event_key(record: Mapping[str, Any], *, context: str) -> str:
    event_key = record.get("event_key")
    if not isinstance(event_key, str) or not event_key.strip():
        raise ValueError(f"{context} 缺非空 event_key: {record!r}")
    return event_key


def validate_label_refresh_intent(record: Mapping[str, Any]) -> None:
    """校验刷新意图事件的最小 schema；扩展证据字段由 T3 按同一事件键承载。"""
    if not isinstance(record, Mapping):
        raise TypeError(f"刷新意图必须是对象，实得 {type(record).__name__}")
    _validate_event_key(record, context="刷新意图")
    record_type = record.get("record_type")
    if record_type not in _INTENT_TYPES:
        raise ValueError(f"刷新意图 record_type 非法: {record_type!r}")
    if record_type == "prepare":
        required = ("trade_date", "row_domain", "input_digest", "producer_digest", "target_status")
        missing = [key for key in required if not record.get(key)]
        if missing:
            raise ValueError(f"prepare 刷新意图缺字段 {missing}: {record!r}")
        row_domain_for_date(str(record["trade_date"]))
        if record["row_domain"] != row_domain_for_date(str(record["trade_date"])):
            raise ValueError(f"prepare 刷新意图行域与日期不一致: {record!r}")
    elif not str(record.get("reason", "")).strip():
        raise ValueError(f"{record_type} 刷新意图必须登记 reason: {record!r}")


def validate_label_output_fingerprints(fingerprints: Any, *, context: str) -> None:
    """校验三表输出指纹完整且均为非空 16 位小写 hex，禁止缺省降级为不核验。"""
    if not isinstance(fingerprints, Mapping):
        raise TypeError(
            f"{context} output_fingerprints 必须是 Mapping，实得 {type(fingerprints).__name__}"
        )
    expected = set(LABEL_TABLES)
    actual = set(fingerprints)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected, key=str)
        raise ValueError(
            f"{context} output_fingerprints 键须恰等于 {sorted(expected)}: "
            f"缺失 {missing}，额外 {extra}"
        )
    for name in LABEL_TABLES:
        value = fingerprints[name]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None:
            raise ValueError(
                f"{context} output_fingerprints[{name!r}] 须为非空 sha256_16"
                f"（16 位小写 hex），实得 {value!r}"
            )


def validate_label_refresh_completion(record: Mapping[str, Any]) -> None:
    """校验刷新完成事件的最小 schema 与完整三表输出指纹。"""
    if not isinstance(record, Mapping):
        raise TypeError(f"刷新完成事件必须是对象，实得 {type(record).__name__}")
    _validate_event_key(record, context="刷新完成事件")
    required = ("trade_date", "row_domain", "input_digest", "producer_digest", "target_status")
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise ValueError(f"刷新完成事件缺字段 {missing}: {record!r}")
    if record["row_domain"] != row_domain_for_date(str(record["trade_date"])):
        raise ValueError(f"刷新完成事件行域与日期不一致: {record!r}")
    if "output_fingerprints" not in record:
        raise ValueError("刷新完成事件缺 output_fingerprints，禁止缺省为不核验")
    validate_label_output_fingerprints(record["output_fingerprints"], context="刷新完成事件")


def _semantic_record(record: Mapping[str, Any]) -> str:
    stable = {k: v for k, v in record.items() if k not in _AUDIT_VOLATILE_KEYS}
    return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_available_from(value: Any, col_name: str) -> None:
    """available_from 仅允许 None 或合法 YYYYMMDD 字符串。"""
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(
            f"列 {col_name}: available_from 须为 YYYYMMDD 字符串或 null，实得 {value!r}"
        )
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"列 {col_name}: available_from 非法日期 {value!r}") from exc


class Manifest:
    """manifest 文件的内存视图 + 登记 API。

    仅显式 ``allow_create=True`` 的全新空数据根允许从空骨架起步；已有资产缺主文件时
    必须走显式恢复，禁止自动以空 manifest、bak 或盘上 parquet 重建。
    """

    def __init__(self, path: Path | str, *, allow_create: bool = False) -> None:
        self._path = Path(path)
        self._allow_create = allow_create
        self._dirty = False  # 未落盘登记标记（任何 register_*/set_* 置脏，save 成功清脏）
        self._disk_stat: tuple[int, int] | None = None  # 加载/保存时的 (mtime_ns, size) 快照
        self.reload()

    # ---------- 加载 / 保存 ----------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def backup_path(self) -> Path:
        return self._path.with_suffix(".json.bak")

    @property
    def dirty(self) -> bool:
        """是否有未落盘的登记（并发冲突 fail-closed 判定用）。"""
        return self._dirty

    def _file_stat(self) -> tuple[int, int] | None:
        """盘上文件 (mtime_ns, size)；不存在返回 None。"""
        try:
            st = self._path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def reload(self) -> None:
        """从盘上重读；仅首次显式初始化允许空骨架，意外缺失 fail-closed。"""
        if self._path.exists():
            self._data = self._load_and_validate()
        elif self._allow_create and self._disk_stat is None:
            self._data = _default_data()
        else:
            raise ValueError(
                f"manifest 主文件缺失，禁止从空骨架继续: {self._path}；"
                "请使用 scripts/v2_p1/recover_manifest.py 先 dry-run 核验恢复"
            )
        self._disk_stat = self._file_stat()
        self._dirty = False

    def changed_on_disk(self) -> bool:
        """盘上文件相对加载/保存时快照是否变化（mtime_ns+size 比对）。"""
        return self._file_stat() != self._disk_stat

    def reload_if_changed(self) -> bool:
        """盘上变化则重读（返回是否重读）；脏检查与并发裁决归调用方（store 写锁内）。"""
        if not self.changed_on_disk():
            return False
        self.reload()
        return True

    def content_fingerprint(self) -> str:
        """manifest 内容指纹：剔除 updated_at 的 canonical JSON（sort_keys+紧凑分隔符）取 sha256_16。

        manifest_version 只是 schema 常量；内容演进（列/分区/labels 登记变化）由本指纹承载。
        """
        return content_fingerprint_of(self._data)

    def _load_and_validate(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"manifest 读取/解析失败: {self._path}（{exc}）") from exc
        if not isinstance(data, dict):
            raise ValueError(f"manifest 顶层必须是 JSON 对象: {self._path}")
        missing = [k for k in _REQUIRED_TOP_KEYS if k not in data]
        if missing:
            raise ValueError(f"manifest 缺顶层键 {missing}: {self._path}")
        if str(data["manifest_version"]) != MANIFEST_VERSION:
            raise ValueError(
                f"manifest_version 不支持: {data['manifest_version']!r}（当前实现仅认 {MANIFEST_VERSION!r}）"
            )
        for key in ("columns", "partitions", "dependencies", "labels"):
            if not isinstance(data[key], dict):
                raise ValueError(f"manifest[{key}] 必须是对象: {self._path}")
        for key in _LIST_EXTENSION_KEYS:
            if key in data and not isinstance(data[key], list):
                raise ValueError(f"manifest[{key}] 必须是数组: {self._path}")
        if "row_domain_segments" in data and not isinstance(data["row_domain_segments"], dict):
            raise ValueError(f"manifest[row_domain_segments] 必须是对象: {self._path}")
        return data

    @staticmethod
    def _replace_with_retry(source: Path, target: Path, *, action: str) -> None:
        """Windows 瞬时占用下有限退避；目标自身异常保留为异常链根因。"""
        last_exc: OSError | None = None
        for attempt in range(5):
            try:
                os.replace(source, target)
                return
            except PermissionError as exc:
                last_exc = exc
                time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"{action}连续被拒（5 次）: {target}") from last_exc

    def _write_backup(self, payload: bytes) -> None:
        """先形成经 JSON/schema 校验的备份，再允许主提交；失败即中止并清理 tmp。"""
        bak_tmp = self.backup_path.with_suffix(".json.bak.tmp")
        try:
            bak_tmp.write_bytes(payload)
            try:
                Manifest(bak_tmp)
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"manifest 备份校验失败，主提交中止: {bak_tmp}（{exc}）"
                ) from exc
            self._replace_with_retry(bak_tmp, self.backup_path, action="manifest 备份落盘")
        except (OSError, RuntimeError, ValueError) as exc:
            try:
                bak_tmp.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                exc.add_note(f"manifest 备份临时文件清理也失败: {bak_tmp}（{cleanup_exc}）")
            raise

    def save(self) -> None:
        """先备份上一代有效 manifest，再以 tmp→replace 原子提交主文件。"""
        if "row_domain_segments" in self._data:
            self.validate_row_domain_readiness()
        self._data["updated_at"] = _now_iso()
        new_payload = (json.dumps(self._data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            Manifest(self._path)  # 当前主文件损坏时不得覆盖既有有效 bak
            backup_payload = self._path.read_bytes()
        elif self._disk_stat is not None or not self._allow_create:
            raise ValueError(f"manifest 主文件意外缺失，禁止无备份提交: {self._path}")
        else:
            backup_payload = new_payload
        self._write_backup(backup_payload)
        tmp_path = self._path.with_suffix(".json.tmp")
        tmp_path.write_bytes(new_payload)
        Manifest(tmp_path)
        self._replace_with_retry(tmp_path, self._path, action="manifest 落盘")
        self._disk_stat = self._file_stat()
        self._dirty = False
        self._allow_create = False

    def snapshot(self) -> dict[str, Any]:
        """全量深拷贝快照（防外部篡改内部状态）。"""
        return copy.deepcopy(self._data)

    @property
    def manifest_version(self) -> str:
        return str(self._data["manifest_version"])

    # ---------- 列登记 / 查询 ----------

    def register_column(
        self,
        name: str,
        group: str,
        source: str,
        definition_version: str = "v1",
        available_from: str | None = None,
        backfilled_at: str | None = None,
        status: str = "active",
    ) -> None:
        """登记单列。重复登记同列：族/定义版本冲突 ⇒ ValueError，否则按新值更新其余字段。"""
        if group not in VALID_GROUPS:
            raise ValueError(f"列 {name}: 非法列族 {group!r}（合法值 {sorted(VALID_GROUPS)}）")
        if status not in _VALID_STATUS:
            raise ValueError(f"列 {name}: 非法 status {status!r}")
        _validate_available_from(available_from, name)
        existing = self._data["columns"].get(name)
        if existing is not None:
            for key, new_val in (("group", group), ("definition_version", definition_version)):
                if existing.get(key) != new_val:
                    raise ValueError(
                        f"列 {name} 重复登记冲突: {key} 已登记 {existing.get(key)!r} ≠ 新值 {new_val!r}"
                    )
        self._data["columns"][name] = {
            "group": group,
            "source": source,
            "definition_version": definition_version,
            "available_from": available_from,
            "backfilled_at": backfilled_at,
            "status": status,
        }
        self._dirty = True

    def register_columns(self, columns: Mapping[str, Mapping[str, Any]]) -> None:
        """批量登记列（每项字段同 register_column，缺省字段走默认值）。"""
        for name, meta in columns.items():
            self.register_column(
                name=name,
                group=str(meta["group"]),
                source=str(meta.get("source", "")),
                definition_version=str(meta.get("definition_version", "v1")),
                available_from=meta.get("available_from"),
                backfilled_at=meta.get("backfilled_at"),
                status=str(meta.get("status", "active")),
            )

    def has_column(self, name: str) -> bool:
        return name in self._data["columns"]

    def column_meta(self, name: str) -> dict[str, Any]:
        """列元数据（深拷贝）。未登记 ⇒ KeyError。"""
        try:
            return copy.deepcopy(self._data["columns"][name])
        except KeyError:
            raise KeyError(f"列未登记 manifest: {name!r}") from None

    def column_group(self, name: str) -> str:
        """列所属族。未登记 ⇒ KeyError。"""
        return str(self.column_meta(name)["group"])

    # ---------- panel 分区登记 / 指纹查询 ----------

    def _require_row_domain_schema(self) -> None:
        if self._data.get("row_domain_segments") != ROW_DOMAIN_SEGMENTS:
            raise RuntimeError("manifest 行域 schema 未迁移，禁止普通登记；请先运行显式迁移工具")

    @staticmethod
    def _check_existing_row_domain(
        existing: dict[str, Any] | None, expected: str, *, context: str
    ) -> None:
        if existing is not None and existing.get("row_domain") != expected:
            raise ValueError(
                f"{context} 行域缺失或冲突: {existing.get('row_domain')!r} ≠ {expected!r}；"
                "禁止由普通登记静默修补"
            )

    def migrate_row_domain_metadata(self) -> dict[str, int]:
        """显式迁移入口专用：补行域与扩展容器，不改变任何 parquet 登记指纹。"""
        had_segments = "row_domain_segments" in self._data
        before = tuple(key in self._data for key in _LIST_EXTENSION_KEYS)
        counts = migrate_manifest_row_domains(self._data)
        for key in _LIST_EXTENSION_KEYS:
            self._data.setdefault(key, [])
        if any(counts.values()) or not all(before) or not had_segments:
            self._dirty = True
        return counts

    def validate_row_domain_readiness(self) -> dict[str, int]:
        """P3 前的行域 readiness 子门；完整日级 readiness 由后续任务组合。"""
        return validate_manifest_row_domains(self._data)

    def register_partition(
        self, rel_path: str, sha256_16: str, rows: int, zone: str = "hot"
    ) -> None:
        """登记 panel 分区（rel_path 相对 features/ 目录，如 panel/20260104/core.parquet）。"""
        if zone not in _VALID_ZONE:
            raise ValueError(f"分区 {rel_path}: 非法 zone {zone!r}")
        self._require_row_domain_schema()
        expected = row_domain_for_partition(rel_path)
        self._check_existing_row_domain(
            self._data["partitions"].get(rel_path), expected, context=f"分区 {rel_path}"
        )
        self._data["partitions"][rel_path] = {
            "sha256_16": sha256_16,
            "rows": int(rows),
            "written_at": _now_iso(),
            "zone": zone,
            "row_domain": expected,
        }
        self._dirty = True

    def partition_meta(self, rel_path: str) -> dict[str, Any] | None:
        """分区元数据（深拷贝）；未登记返回 None。"""
        meta = self._data["partitions"].get(rel_path)
        return copy.deepcopy(meta) if meta is not None else None

    def partition_fingerprint(self, rel_path: str) -> str | None:
        """分区内容指纹；既有登记行域损坏时在物理重写前 fail-closed。"""
        meta = self._data["partitions"].get(rel_path)
        if meta is not None:
            self._check_existing_row_domain(
                meta, row_domain_for_partition(rel_path), context=f"分区 {rel_path}"
            )
        return str(meta["sha256_16"]) if meta is not None else None

    def register_repair(
        self, rel_path: str, reason: str, added_dates: list[str], old_fp: str, new_fp: str
    ) -> None:
        """登记一次修复性合并（append-only 审计轨迹；不改变分区注册表语义）。

        修复 ≠ 口径修正：既有单元格逐值不变、仅追加缺失日。调用方（store 层）
        负责校验行超集语义，此处只留痕。
        """
        repairs = self._data.setdefault("repairs", [])
        repairs.append(
            {
                "rel_path": rel_path,
                "reason": reason,
                "added_dates": sorted(added_dates),
                "old_fingerprint": old_fp,
                "new_fingerprint": new_fp,
                "repaired_at": _now_iso(),
            }
        )
        self._dirty = True

    def repairs(self) -> list[dict[str, Any]]:
        """全部修复登记（深拷贝）。"""
        return copy.deepcopy(self._data.get("repairs", []))

    # ---------- labels 分区与封存水位 ----------

    def _label_entry(self, label_name: str) -> dict[str, Any]:
        """取（必要时建）labels.<name> 条目。"""
        return self._data["labels"].setdefault(
            label_name, {"partitions": {}, "sealed_through": None}
        )

    def register_label_partition(
        self, label_name: str, date_str: str, sha256_16: str, rows: int, zone: str = "hot"
    ) -> None:
        """登记标签分区（键 = 相对标签目录文件名 YYYYMMDD.parquet）。"""
        if zone not in _VALID_ZONE:
            raise ValueError(f"标签 {label_name} 分区 {date_str}: 非法 zone {zone!r}")
        self._require_row_domain_schema()
        expected = row_domain_for_date(date_str)
        filename = f"{date_str}.parquet"
        partitions = self._label_entry(label_name)["partitions"]
        self._check_existing_row_domain(
            partitions.get(filename), expected, context=f"标签 {label_name}/{filename}"
        )
        partitions[filename] = {
            "sha256_16": sha256_16,
            "rows": int(rows),
            "written_at": _now_iso(),
            "zone": zone,
            "row_domain": expected,
        }
        self._dirty = True

    def label_partition_meta(self, label_name: str, date_str: str) -> dict[str, Any] | None:
        """标签分区元数据（深拷贝）；未登记返回 None。"""
        meta = self._label_entry(label_name)["partitions"].get(f"{date_str}.parquet")
        return copy.deepcopy(meta) if meta is not None else None

    def label_partition_fingerprint(self, label_name: str, date_str: str) -> str | None:
        """标签内容指纹；既有登记行域损坏时在物理重写前 fail-closed。"""
        meta = self._label_entry(label_name)["partitions"].get(f"{date_str}.parquet")
        if meta is not None:
            self._check_existing_row_domain(
                meta,
                row_domain_for_date(date_str),
                context=f"标签 {label_name}/{date_str}.parquet",
            )
        return str(meta["sha256_16"]) if meta is not None else None

    def label_sealed_through(self, label_name: str) -> str | None:
        """标签封存水位（YYYYMMDD）；未封存返回 None。"""
        value = self._label_entry(label_name).get("sealed_through")
        return str(value) if value is not None else None

    def set_label_sealed_through(self, label_name: str, date_str: str) -> None:
        """推进封存水位（只进不退）。"""
        current = self.label_sealed_through(label_name)
        if current is None or date_str > current:
            self._label_entry(label_name)["sealed_through"] = date_str
            self._dirty = True

    # ---------- 刷新与迁移审计（append-only） ----------

    def _append_audit(
        self, container_key: str, record: Mapping[str, Any], identity_keys: tuple[str, ...]
    ) -> bool:
        """按稳定身份追加审计；同语义 no-op，同身份不同语义 fail-closed。"""
        self._require_row_domain_schema()
        identity = tuple(record.get(key) for key in identity_keys)
        semantic = _semantic_record(record)
        for existing in self._data[container_key]:
            existing_identity = tuple(existing.get(key) for key in identity_keys)
            if existing_identity == identity:
                if _semantic_record(existing) == semantic:
                    return False
                raise ValueError(f"{container_key} 审计身份冲突: {identity!r}")
        payload = copy.deepcopy(dict(record))
        payload.setdefault("recorded_at", _now_iso())
        self._data[container_key].append(payload)
        self._dirty = True
        return True

    def append_label_refresh_intent(self, record: Mapping[str, Any]) -> bool:
        """追加 prepare/superseded/orphan 意图事件，不原地改写历史。"""
        validate_label_refresh_intent(record)
        keys = ("record_type", "event_key", "related_event_key", "superseded_by")
        return self._append_audit("label_refresh_intents", record, keys)

    def append_label_refresh_completion(self, record: Mapping[str, Any]) -> bool:
        """按 event_key 恰一次登记刷新完成审计。"""
        validate_label_refresh_completion(record)
        return self._append_audit("label_refreshes", record, ("event_key",))

    def register_schema_migration(self, record: Mapping[str, Any]) -> bool:
        """登记一次显式 schema 迁移；migration_id 唯一且内容不可冲突。"""
        required = ("migration_id", "input_fingerprint", "output_fingerprint", "evidence_path")
        missing = [key for key in required if not record.get(key)]
        if missing:
            raise ValueError(f"schema 迁移审计缺字段 {missing}: {record!r}")
        return self._append_audit("schema_migrations", record, ("migration_id",))

    def label_refresh_intents(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._data.get("label_refresh_intents", []))

    def label_refreshes(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._data.get("label_refreshes", []))

    def schema_migrations(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._data.get("schema_migrations", []))

    # ---------- 依赖声明 ----------

    def register_dependencies(
        self,
        raw_datasets: list[str] | None = None,
        factor_functions: Mapping[str, str] | None = None,
    ) -> None:
        """登记依赖声明（raw 清单 + factors 派生函数标识；幂等合并，去重保序）。"""
        deps = self._data["dependencies"]
        if raw_datasets is not None:
            merged = list(dict.fromkeys([*deps.get("raw_datasets", []), *raw_datasets]))
            deps["raw_datasets"] = merged
        if factor_functions is not None:
            deps["factor_functions"] = {
                **deps.get("factor_functions", {}),
                **dict(factor_functions),
            }
        self._dirty = True
