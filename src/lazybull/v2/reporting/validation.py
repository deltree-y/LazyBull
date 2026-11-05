# -*- coding: utf-8 -*-
"""P6 validation 根、新目录守卫与隔离写盘（设计 §2.5 / §10）。

T0.5 只实现验收侧需要的最小面：
- validation 根形态 ``data/reports/p6/validations/<evidence_class>/<validation_id>/``；
- 已存在的目录/文件一律拒绝（无 latest、不覆盖、不追加）；
- JSON 写入固定为 UTF-8/LF、``sort_keys``、``allow_nan=False``，并拒绝覆盖既有文件。

完整“精确写白名单审计”（全树哈希、写/锁/socket 拦截）归 T1；本模块不承担审计器职责。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from src.lazybull.v2.reporting import contracts

__all__ = [
    "VALIDATION_ID_PATTERN",
    "ValidationError",
    "ensure_new_dir",
    "relative_posix",
    "resolve_validation_dir",
    "sha256_file",
    "utc_stamp",
    "validate_validation_id",
    "write_json_new",
]

#: validation_id 形态（小写字母/数字/下划线/连字符，3~64 位）
VALIDATION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_\-]{2,63}$")


class ValidationError(RuntimeError):
    """validation 路径/写入不满足契约（已存在、越界、非有限数值）。"""


def utc_stamp(moment: datetime | None = None) -> str:
    """UTC ``YYYYMMDDTHHMMSSZ``；validation_id 的时间部分唯一来源。"""
    current = moment or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validate_validation_id(validation_id: str) -> str:
    """校验 validation_id；不合法直接报错，禁止清洗或猜测。"""
    if not VALIDATION_ID_PATTERN.match(validation_id):
        raise ValidationError(
            f"validation_id 不合法：{validation_id!r}（要求 ^[a-z0-9][a-z0-9_-]{{2,63}}$）"
        )
    return validation_id


def resolve_validation_dir(
    reports_root: Path | str,
    evidence_class: str,
    validation_id: str,
) -> Path:
    """返回**尚不存在**的 validation 目录；已存在即拒绝（不覆盖历史）。"""
    if evidence_class not in contracts.EVIDENCE_CLASSES:
        raise ValidationError(f"未知 evidence_class：{evidence_class!r}")
    validate_validation_id(validation_id)
    target = Path(reports_root) / "validations" / evidence_class / validation_id
    if target.exists():
        raise ValidationError(f"validation 目录已存在，拒绝覆盖：{target}")
    return target


def ensure_new_dir(path: Path | str) -> Path:
    """创建新目录（父目录可新建）；目录已存在即拒绝。"""
    target = Path(path)
    if target.exists():
        raise ValidationError(f"输出目录已存在，拒绝覆盖：{target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def sha256_file(path: Path | str) -> str:
    """文件字节 SHA-256；调用方负责路径存在性。"""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix(path: Path | str, base: Path | str) -> str:
    """root-relative POSIX 路径；越界路径直接报错（禁止绝对路径进入 canonical 输出）。"""
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError as exc:
        raise ValidationError(f"路径不在 base 之下：{path}（base={base}）") from exc


def write_json_new(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """写新 JSON 文件（不允许覆盖）；UTF-8/LF、sort_keys、allow_nan=False。"""
    target = Path(path)
    if target.exists():
        raise ValidationError(f"JSON 目标已存在，拒绝覆盖：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    target.write_bytes((text + "\n").encode("utf-8"))
    return target
