# -*- coding: utf-8 -*-
"""公告正文/附件 append-only 版本库（P2 共用工具）。

边界：
- 只管理本机已取得的公告事实，不抓网、不判断候选；
- 同一 ``(source, notice_id)`` 内容变化必须新增版本，禁止原地覆盖；
- 正文按 canonical JSON + SHA256 内容寻址；附件按字节 SHA 内容寻址；
- 区分 ``source_publish_date``（公告发布日期）与 ``captured_at``（实际观测时间）；
- 单进程单写者；索引逐行追加，历史行不修改。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from scripts.v2_p7.p2.types import InstrumentError, normalize_date

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class AnnouncementStoreError(InstrumentError):
    """版本库结构错误或 append-only 被破坏。"""


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    name: str
    sha256: str
    bytes: int
    blob_path: str


@dataclass(frozen=True, slots=True)
class VersionRecord:
    source: str
    notice_id: str
    version: int
    publish_date: str
    captured_at: str
    payload_sha256: str
    payload_path: str
    attachments: tuple[AttachmentRef, ...]

    @property
    def captured_date(self) -> str:
        return self.captured_at[:10].replace("-", "")

    def load_payload(self, root: Path) -> dict:
        path = Path(root) / self.payload_path
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnnouncementStoreError(f"公告内容不可读: {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CaptureResult:
    record: VersionRecord
    duplicate: bool
    verification: dict


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalize_timestamp(value: object) -> str:
    """归一到带 UTC 的 ISO 时间；仅日期按当日 00:00:00 UTC。"""
    text = str(value).strip()
    if not text:
        raise AnnouncementStoreError("captured_at 不得为空")
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}T00:00:00+00:00"
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise AnnouncementStoreError(f"captured_at 不是合法 ISO 时间: {value!r}") from exc
    if parsed.tzinfo is None:
        raise AnnouncementStoreError("captured_at 必须携带时区，禁止本地时间歧义")
    return parsed.astimezone(timezone.utc).isoformat()


def _validate_id(value: object, field: str) -> str:
    text = str(value).strip()
    if not _ID_RE.fullmatch(text):
        raise AnnouncementStoreError(f"{field} 非法（仅 ASCII 字母数字 ._-）: {value!r}")
    return text


class AnnouncementStore:
    """本地 append-only 公告版本库。"""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._index_path = self._root / "index.jsonl"

    @property
    def root(self) -> Path:
        return self._root

    def capture(
        self,
        *,
        source: str,
        notice_id: str,
        publish_date: object,
        captured_at: object,
        payload: Mapping[str, object],
        attachments: Sequence[Path] = (),
    ) -> CaptureResult:
        """登记一次观测；相同内容不新增版本，变化内容新增下一版本。"""
        source_key = _validate_id(source, "source")
        notice_key = _validate_id(notice_id, "notice_id")
        publish = normalize_date(publish_date)
        captured = _normalize_timestamp(captured_at)
        content_bytes, content_sha, attachment_refs = self._build_content(
            publish, payload, attachments
        )
        rows = self._read_index()
        same_key = [
            row for row in rows if row["source"] == source_key and row["notice_id"] == notice_key
        ]
        latest = max((row["version"] for row in same_key), default=0)
        same_version = next((row for row in reversed(same_key) if row["version"] == latest), None)
        duplicate = bool(same_version and same_version["payload_sha256"] == content_sha)
        version = latest if duplicate else latest + 1
        payload_relative = (
            Path("versions") / source_key / notice_key / f"v{version:06d}_{content_sha}.json"
        )
        self._persist_payload(payload_relative, content_bytes, content_sha)
        record = VersionRecord(
            source=source_key,
            notice_id=notice_key,
            version=version,
            publish_date=publish,
            captured_at=captured,
            payload_sha256=content_sha,
            payload_path=payload_relative.as_posix(),
            attachments=tuple(attachment_refs),
        )
        self._append_index(self._index_row(record, duplicate))
        verification = self.verify()
        if not verification["ok"]:
            raise AnnouncementStoreError(f"版本库自检失败: {verification['errors']}")
        return CaptureResult(record=record, duplicate=duplicate, verification=verification)

    def _build_content(
        self,
        publish: str,
        payload: Mapping[str, object],
        attachments: Sequence[Path],
    ) -> tuple[bytes, str, list[AttachmentRef]]:
        if not isinstance(payload, Mapping):
            raise AnnouncementStoreError("payload 必须为 mapping")
        refs = [self._store_attachment(Path(path)) for path in attachments]
        refs.sort(key=lambda item: (item.name, item.sha256))
        content = {
            "publish_date": publish,
            "payload": dict(payload),
            "attachments": [
                {"name": item.name, "sha256": item.sha256, "bytes": item.bytes} for item in refs
            ],
        }
        content_bytes = _canonical_bytes(content)
        return content_bytes, hashlib.sha256(content_bytes).hexdigest(), refs

    def _persist_payload(self, relative: Path, content_bytes: bytes, content_sha: str) -> None:
        payload_path = self._root / relative
        if payload_path.exists():
            existing = payload_path.read_bytes()
            if hashlib.sha256(existing).hexdigest() != content_sha:
                raise AnnouncementStoreError(f"版本文件哈希冲突: {payload_path}")
            return
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(content_bytes)

    @staticmethod
    def _index_row(record: VersionRecord, duplicate: bool) -> dict:
        return {
            "source": record.source,
            "notice_id": record.notice_id,
            "version": record.version,
            "publish_date": record.publish_date,
            "captured_at": record.captured_at,
            "payload_sha256": record.payload_sha256,
            "payload_path": record.payload_path,
            "attachments": [
                {
                    "name": item.name,
                    "sha256": item.sha256,
                    "bytes": item.bytes,
                    "blob_path": item.blob_path,
                }
                for item in record.attachments
            ],
            "duplicate": duplicate,
        }

    def versions(self, *, source: str, notice_id: str) -> tuple[VersionRecord, ...]:
        """该公告的全部观测行（可能同一版本有多次重复观测）。"""
        source_key = _validate_id(source, "source")
        notice_key = _validate_id(notice_id, "notice_id")
        result: list[VersionRecord] = []
        for row in self._read_index():
            if row["source"] != source_key or row["notice_id"] != notice_key:
                continue
            result.append(self._row_to_record(row))
        return tuple(sorted(result, key=lambda item: (item.version, item.captured_at)))

    def latest_as_published(
        self, *, source: str, notice_id: str, as_of: object
    ) -> VersionRecord | None:
        """按公告发布日期选择当时已发布的最新版本（当前档案回放口径）。

        这不能证明当时确实拿到了该版本；严格"只见过什么"请用
        :meth:`latest_as_observed`。
        """
        cutoff = normalize_date(as_of)
        rows = [
            item
            for item in self.versions(source=source, notice_id=notice_id)
            if item.publish_date <= cutoff
        ]
        if not rows:
            return None
        highest = max(item.version for item in rows)
        return max((item for item in rows if item.version == highest), key=lambda x: x.captured_at)

    def latest_as_observed(
        self, *, source: str, notice_id: str, as_of: object
    ) -> VersionRecord | None:
        """只返回 ``captured_at <= as_of`` 的版本（严格前向证据）。"""
        cutoff = normalize_date(as_of)
        rows = [
            item
            for item in self.versions(source=source, notice_id=notice_id)
            if item.publish_date <= cutoff and item.captured_date <= cutoff
        ]
        if not rows:
            return None
        highest = max(item.version for item in rows)
        return max((item for item in rows if item.version == highest), key=lambda x: x.captured_at)

    def verify(self) -> dict:
        """重算索引、正文与附件身份；任何缺件/哈希不符即失败。"""
        errors: list[str] = []
        rows = self._read_index()
        seen_versions: set[tuple[str, str, int, str]] = set()
        for row in rows:
            payload_path = self._root / row["payload_path"]
            if not payload_path.is_file():
                errors.append(f"缺正文: {row['payload_path']}")
                continue
            actual = hashlib.sha256(payload_path.read_bytes()).hexdigest()
            if actual != row["payload_sha256"]:
                errors.append(f"正文哈希不符: {row['payload_path']}")
            identity = (row["source"], row["notice_id"], int(row["version"]), row["payload_sha256"])
            if identity in seen_versions and row.get("duplicate") is not True:
                errors.append(f"同版本非重复观测: {identity}")
            seen_versions.add(identity)
            for attachment in row.get("attachments", []):
                blob = self._root / attachment["blob_path"]
                if not blob.is_file():
                    errors.append(f"缺附件: {attachment['blob_path']}")
                    continue
                blob_sha = hashlib.sha256(blob.read_bytes()).hexdigest()
                if blob_sha != attachment["sha256"]:
                    errors.append(f"附件哈希不符: {attachment['blob_path']}")
        return {
            "ok": not errors,
            "rows": len(rows),
            "versions": len({(x[0], x[1], x[2]) for x in seen_versions}),
            "errors": errors,
        }

    def _store_attachment(self, path: Path) -> AttachmentRef:
        if not path.is_file():
            raise AnnouncementStoreError(f"附件不存在: {path}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        relative = Path("blobs") / digest[:2] / digest
        destination = self._root / relative
        if destination.exists():
            if destination.read_bytes() != data:
                raise AnnouncementStoreError(f"附件内容寻址冲突: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with destination.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError:
                if destination.read_bytes() != data:
                    raise AnnouncementStoreError(f"附件并发写入冲突: {destination}")
        return AttachmentRef(
            name=path.name,
            sha256=digest,
            bytes=len(data),
            blob_path=relative.as_posix(),
        )

    def _append_index(self, row: Mapping[str, object]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        line = _canonical_bytes(row) + b"\n"
        with self._index_path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _read_index(self) -> list[dict]:
        if not self._index_path.exists():
            return []
        rows: list[dict] = []
        try:
            text = self._index_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AnnouncementStoreError(f"索引不可读: {exc}") from exc
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnouncementStoreError(f"索引第 {number} 行非法 JSON: {exc}") from exc
            required = {
                "source",
                "notice_id",
                "version",
                "publish_date",
                "captured_at",
                "payload_sha256",
                "payload_path",
            }
            missing = required - set(row)
            if missing:
                raise AnnouncementStoreError(f"索引第 {number} 行缺字段: {sorted(missing)}")
            rows.append(row)
        return rows

    @staticmethod
    def _row_to_record(row: Mapping[str, object]) -> VersionRecord:
        attachments = tuple(
            AttachmentRef(
                name=str(item["name"]),
                sha256=str(item["sha256"]),
                bytes=int(item["bytes"]),
                blob_path=str(item["blob_path"]),
            )
            for item in row.get("attachments", [])
        )
        return VersionRecord(
            source=str(row["source"]),
            notice_id=str(row["notice_id"]),
            version=int(row["version"]),
            publish_date=str(row["publish_date"]),
            captured_at=str(row["captured_at"]),
            payload_sha256=str(row["payload_sha256"]),
            payload_path=str(row["payload_path"]),
            attachments=attachments,
        )
