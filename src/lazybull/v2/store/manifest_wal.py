# -*- coding: utf-8 -*-
"""v2 store manifest WAL、延迟落盘与锁内同步。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.lazybull.v2.store.manifest import (
    Manifest,
    validate_label_refresh_completion,
    validate_label_refresh_intent,
)
from src.lazybull.v2.store.row_domain import row_domain_for_date, row_domain_for_partition

if TYPE_CHECKING:
    from src.lazybull.v2.store.data_store import PanelDataStore

__all__ = ["ManifestDefer", "ManifestWal"]


class ManifestDefer:
    """整段持写锁的延迟落盘上下文（计数可重入，禁止跨实例交错压实）。"""

    def __init__(self, store: "PanelDataStore") -> None:
        self._store = store

    def __enter__(self) -> None:
        self._store.write_lock().acquire()
        self._store._manifest_defer += 1
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        had_error = exc_type is not None
        try:
            self._store._manifest_defer -= 1
            if self._store._manifest_defer == 0:
                self._store._wal.sync_manifest_under_lock()
                self._store._wal.save()
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            had_error = True
            raise
        finally:
            self._store.write_lock().release(during_exception=had_error)


class ManifestWal:
    """manifest 登记事件 WAL（登记事件先写 WAL 再改内存）。"""

    def __init__(self, store: "PanelDataStore") -> None:
        self._store = store

    def _current_wal_stat(self) -> tuple[int, int] | None:
        try:
            stat = self._store.wal_path.stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _mark_wal_absorbed(self) -> None:
        """记录已被当前 manifest 内存快照吸收的 WAL 代际。"""
        self._store._wal_stat = self._current_wal_stat()

    def _wal_changed(self) -> bool:
        return self._current_wal_stat() != self._store._wal_stat

    def load_manifest(self, *, persist: bool = True) -> Manifest:
        """加载并重放；persist=False 仅供恢复 dry-run 的内存预演，禁止后续绕过预检。"""
        store = self._store
        path = store.features_dir / "manifest.json"
        try:
            allow_create = not path.exists() and store._can_initialize_manifest(path)
            store._manifest = Manifest(path, allow_create=allow_create)
            self.replay(persist=persist)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            store._manifest = None
            raise
        return store._manifest

    def save(self) -> None:
        """manifest 落盘后压实 WAL，并标记当前快照已吸收该代际。"""
        self._store.manifest.save()
        self.truncate()
        self._mark_wal_absorbed()

    def maybe_save(self) -> None:
        """延迟模式外立即落盘；延迟模式内由出口统一落盘。"""
        if self._store._manifest_defer == 0:
            self.save()

    def truncate(self) -> None:
        """压实 WAL（删除；manifest 落盘是其事件已持久化的前提）。"""
        wal = self._store.wal_path
        try:
            wal.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(f"WAL 压实删除失败（下次打开重放兜底，幂等无害）: {wal}（{exc}）")

    def append(self, event: dict[str, Any]) -> None:
        """追加一行 WAL 事件（flush；调用方须在写锁内、物理写成功后、内存登记前）。"""
        wal = self._store.wal_path
        wal.parent.mkdir(parents=True, exist_ok=True)
        with open(wal, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()

    def register_partition(self, rel_path: str, fp: str, rows: int, zone: str) -> None:
        self.append(
            {
                "op": "register_partition",
                "rel_path": rel_path,
                "sha256_16": fp,
                "rows": rows,
                "zone": zone,
                "row_domain": row_domain_for_partition(rel_path),
            }
        )
        self._store.manifest.register_partition(rel_path, fp, rows=rows, zone=zone)
        self._mark_wal_absorbed()

    def register_label_partition(self, label_name: str, date_str: str, fp: str, rows: int) -> None:
        self.append(
            {
                "op": "register_label_partition",
                "label_name": label_name,
                "date": date_str,
                "sha256_16": fp,
                "rows": rows,
                "row_domain": row_domain_for_date(date_str),
            }
        )
        self._store.manifest.register_label_partition(label_name, date_str, fp, rows)
        self._mark_wal_absorbed()

    def append_label_refresh_intent(self, record: dict[str, Any]) -> bool:
        """先持久化刷新意图 WAL，再改内存登记。"""
        validate_label_refresh_intent(record)
        self.append({"op": "append_label_refresh_intent", "record": record})
        result = self._store.manifest.append_label_refresh_intent(record)
        self._mark_wal_absorbed()
        return result

    def append_label_refresh_completion(self, record: dict[str, Any]) -> bool:
        """先持久化刷新完成 WAL，再改内存登记。"""
        validate_label_refresh_completion(record)
        self.append({"op": "append_label_refresh_completion", "record": record})
        result = self._store.manifest.append_label_refresh_completion(record)
        self._mark_wal_absorbed()
        return result

    def set_label_sealed_through(self, label_name: str, date_str: str) -> None:
        self.append({"op": "set_label_sealed_through", "label_name": label_name, "date": date_str})
        self._store.manifest.set_label_sealed_through(label_name, date_str)
        self._mark_wal_absorbed()

    def register_repair(
        self, rel_path: str, reason: str, added_dates: list[str], old_fp: str, new_fp: str
    ) -> None:
        self.append(
            {
                "op": "register_repair",
                "rel_path": rel_path,
                "reason": reason,
                "added_dates": sorted(added_dates),
                "old_fingerprint": old_fp,
                "new_fingerprint": new_fp,
            }
        )
        self._store.manifest.register_repair(rel_path, reason, added_dates, old_fp, new_fp)
        self._mark_wal_absorbed()

    def replay(self, *, persist: bool = True) -> None:
        """写锁内重放 WAL；未知事件/schema 在任何应用与压实前 fail-closed。"""
        with self._store.write_lock():
            wal = self._store.wal_path
            try:
                if not wal.exists() or wal.stat().st_size == 0:
                    self._mark_wal_absorbed()
                    return
            except OSError:
                return
            events: list[dict[str, Any]] = []
            with open(wal, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"WAL 第 {line_no} 行 JSON 损坏，禁止部分重放或压实: {wal}（{exc}）"
                        ) from exc
                    try:
                        self._validate_event(event)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(
                            f"WAL 第 {line_no} 行 schema 拒绝: {wal}（{exc}）"
                        ) from exc
                    events.append(event)
            applied = sum(self._apply_event(event) for event in events)
            if applied and persist:
                logger.warning(f"WAL 重放补登记 {applied} 条事件（崩溃恢复），落盘并压实: {wal}")
                self.save()
            else:
                if applied:
                    logger.warning(f"WAL 内存预演 {applied} 条事件（dry-run，保留现场）: {wal}")
                self._mark_wal_absorbed()

    @staticmethod
    def _validate_event(event: dict[str, Any]) -> None:
        """预检单行事件，保证后续批次不会因未知 schema 部分应用后压实。"""
        if not isinstance(event, dict):
            raise TypeError(f"WAL 事件必须是对象，实得 {type(event).__name__}")
        op = event.get("op")
        required = {
            "register_partition": ("rel_path", "sha256_16", "rows", "zone"),
            "register_label_partition": ("label_name", "date", "sha256_16", "rows"),
            "set_label_sealed_through": ("label_name", "date"),
            "register_repair": (
                "rel_path",
                "reason",
                "added_dates",
                "old_fingerprint",
                "new_fingerprint",
            ),
            "append_label_refresh_intent": ("record",),
            "append_label_refresh_completion": ("record",),
        }
        if op not in required:
            raise ValueError(f"未知 WAL 事件类型: {op!r}")
        missing = [key for key in required[op] if key not in event]
        if missing:
            raise ValueError(f"WAL 事件 {op} 缺字段 {missing}")
        if op == "register_partition":
            int(event["rows"])
            expected = row_domain_for_partition(event["rel_path"])
            if event.get("row_domain", expected) != expected:
                raise ValueError(f"WAL 分区行域冲突: {event!r}")
        elif op == "register_label_partition":
            int(event["rows"])
            expected = row_domain_for_date(event["date"])
            if event.get("row_domain", expected) != expected:
                raise ValueError(f"WAL 标签分区行域冲突: {event!r}")
        elif op == "set_label_sealed_through":
            row_domain_for_date(event["date"])
        elif op == "append_label_refresh_intent":
            validate_label_refresh_intent(event["record"])
        elif op == "append_label_refresh_completion":
            validate_label_refresh_completion(event["record"])

    def _apply_register_partition(self, event: dict[str, Any]) -> int:
        manifest = self._store.manifest
        expected = row_domain_for_partition(event["rel_path"])
        if manifest.partition_fingerprint(event["rel_path"]) == event["sha256_16"]:
            meta = manifest.partition_meta(event["rel_path"])
            if meta is None or meta.get("row_domain") != expected:
                raise ValueError(f"WAL 已登记分区行域缺失或冲突: {event['rel_path']}")
            return 0
        logger.warning(f"WAL 补登记分区: {event['rel_path']}")
        manifest.register_partition(
            event["rel_path"],
            event["sha256_16"],
            rows=int(event["rows"]),
            zone=str(event["zone"]),
        )
        return 1

    def _apply_register_label_partition(self, event: dict[str, Any]) -> int:
        manifest = self._store.manifest
        expected = row_domain_for_date(event["date"])
        if (
            manifest.label_partition_fingerprint(event["label_name"], event["date"])
            == event["sha256_16"]
        ):
            meta = manifest.label_partition_meta(event["label_name"], event["date"])
            if meta is None or meta.get("row_domain") != expected:
                raise ValueError(f"WAL 已登记标签分区行域缺失或冲突: {event!r}")
            return 0
        logger.warning(f"WAL 补登记标签分区: {event['label_name']}/{event['date']}")
        manifest.register_label_partition(
            event["label_name"], event["date"], event["sha256_16"], int(event["rows"])
        )
        return 1

    def _apply_set_label_sealed_through(self, event: dict[str, Any]) -> int:
        manifest = self._store.manifest
        current = manifest.label_sealed_through(event["label_name"])
        if current is not None and current >= event["date"]:
            return 0
        logger.warning(f"WAL 补登记封存水位: {event['label_name']} -> {event['date']}")
        manifest.set_label_sealed_through(event["label_name"], event["date"])
        return 1

    def _apply_register_repair(self, event: dict[str, Any]) -> int:
        manifest = self._store.manifest
        dup = any(
            r["rel_path"] == event["rel_path"]
            and r["reason"] == event["reason"]
            and r["old_fingerprint"] == event["old_fingerprint"]
            and r["new_fingerprint"] == event["new_fingerprint"]
            for r in manifest.repairs()
        )
        if dup:
            return 0
        logger.warning(f"WAL 补登记修复审计: {event['rel_path']}")
        manifest.register_repair(
            event["rel_path"],
            event["reason"],
            list(event["added_dates"]),
            event["old_fingerprint"],
            event["new_fingerprint"],
        )
        return 1

    def _apply_event(self, event: dict[str, Any]) -> int:
        """应用单行 WAL 事件（manifest 缺失或指纹不同才落），返回应用数（0/1）。"""
        op = event["op"]
        if op == "append_label_refresh_intent":
            return int(self._store.manifest.append_label_refresh_intent(event["record"]))
        if op == "append_label_refresh_completion":
            return int(self._store.manifest.append_label_refresh_completion(event["record"]))
        handlers = {
            "register_partition": self._apply_register_partition,
            "register_label_partition": self._apply_register_label_partition,
            "set_label_sealed_through": self._apply_set_label_sealed_through,
            "register_repair": self._apply_register_repair,
        }
        if op not in handlers:
            raise ValueError(f"未知 WAL 事件类型: {op!r}")
        return handlers[op](event)

    def sync_manifest_under_lock(self) -> None:
        """写锁内同步 manifest 与 WAL 代际；任何加载/重放失败都使缓存失效。"""
        store = self._store
        manifest = store.manifest
        changed = manifest.changed_on_disk()
        wal_changed = self._wal_changed()
        if changed and manifest.dirty:
            raise RuntimeError(
                f"manifest 并发冲突（fail-closed）: {manifest.path} 盘上已被外部改动，"
                "而本实例持有未落盘登记；请重开实例后重跑"
            )
        try:
            if changed:
                manifest.reload()
            manifest._require_row_domain_schema()
            if changed or wal_changed:
                self.replay()
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            store._manifest = None
            raise
