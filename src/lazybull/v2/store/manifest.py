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
                      {"sha256_16": "...", "rows": N, "written_at": "<ISO>", "zone": ...} },
      "dependencies": {"raw_datasets": [...], "factor_functions": {...}},
      "labels": { "<label_name>": {"partitions": {...同 partitions 结构...},
                                   "sealed_through": "YYYYMMDD"|null} }
    }

口径要点：
- 原子写（tmp + replace，沿 `data/storage.py::_save_data` 先例）；
- 快照返回深拷贝，防外部篡改内部状态；
- 分区内容指纹 = 文件内容 sha256 前 16 位 hex（与步骤 0 快照口径一致）；
- labels 分区键用相对标签目录的文件名（``YYYYMMDD.parquet``），panel 分区键相对 features/ 目录。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "MANIFEST_VERSION",
    "VALID_GROUPS",
    "Manifest",
    "sha256_16_of_file",
    "sha256_16_of_bytes",
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

#: 顶层必需键（加载校验用）
_REQUIRED_TOP_KEYS = (
    "manifest_version",
    "updated_at",
    "columns",
    "partitions",
    "dependencies",
    "labels",
)


def sha256_16_of_bytes(data: bytes) -> str:
    """内容指纹：sha256 前 16 位 hex（与步骤 0 快照口径一致）。"""
    return hashlib.sha256(data).hexdigest()[:16]


def sha256_16_of_file(path: Path) -> str:
    """文件内容指纹（分块读，防大文件撑爆内存）。"""
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
    }


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

    文件不存在时按空骨架起步（首次 ``save()`` 落盘）；存在则加载并做 schema 校验。
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        if self._path.exists():
            self._data = self._load_and_validate()
        else:
            self._data = _default_data()

    # ---------- 加载 / 保存 ----------

    @property
    def path(self) -> Path:
        return self._path

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
        return data

    def save(self) -> None:
        """原子写（tmp + replace）；``updated_at`` 在落盘时刷新。"""
        self._data["updated_at"] = _now_iso()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp_path, self._path)

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

    def register_partition(
        self, rel_path: str, sha256_16: str, rows: int, zone: str = "hot"
    ) -> None:
        """登记 panel 分区（rel_path 相对 features/ 目录，如 panel/20260104/core.parquet）。"""
        if zone not in _VALID_ZONE:
            raise ValueError(f"分区 {rel_path}: 非法 zone {zone!r}")
        self._data["partitions"][rel_path] = {
            "sha256_16": sha256_16,
            "rows": int(rows),
            "written_at": _now_iso(),
            "zone": zone,
        }

    def partition_meta(self, rel_path: str) -> dict[str, Any] | None:
        """分区元数据（深拷贝）；未登记返回 None。"""
        meta = self._data["partitions"].get(rel_path)
        return copy.deepcopy(meta) if meta is not None else None

    def partition_fingerprint(self, rel_path: str) -> str | None:
        """分区内容指纹；未登记返回 None。"""
        meta = self._data["partitions"].get(rel_path)
        return str(meta["sha256_16"]) if meta is not None else None

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
        self._label_entry(label_name)["partitions"][f"{date_str}.parquet"] = {
            "sha256_16": sha256_16,
            "rows": int(rows),
            "written_at": _now_iso(),
            "zone": zone,
        }

    def label_partition_fingerprint(self, label_name: str, date_str: str) -> str | None:
        """标签分区内容指纹；未登记返回 None。"""
        meta = self._label_entry(label_name)["partitions"].get(f"{date_str}.parquet")
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
