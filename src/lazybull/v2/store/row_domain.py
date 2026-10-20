# -*- coding: utf-8 -*-
"""v2 store 行域边界、派生与 readiness 校验的唯一来源。"""

from __future__ import annotations

import calendar
import re
from datetime import datetime
from typing import Any, Mapping

__all__ = [
    "LABEL_FILTERED_THROUGH",
    "STRUCTURAL_FROM",
    "ROW_DOMAIN_SEGMENTS",
    "VALID_ROW_DOMAINS",
    "row_domain_for_date",
    "row_domain_for_partition",
    "migrate_manifest_row_domains",
    "validate_manifest_row_domains",
]

LABEL_FILTERED_THROUGH = "20260702"
STRUCTURAL_FROM = "20260703"
LABEL_FILTERED = "label_filtered"
STRUCTURAL = "structural"
VALID_ROW_DOMAINS = frozenset({LABEL_FILTERED, STRUCTURAL})
ROW_DOMAIN_SEGMENTS = {
    "label_filtered_through": LABEL_FILTERED_THROUGH,
    "structural_from": STRUCTURAL_FROM,
}

_PANEL_RE = re.compile(r"panel/(\d{8})/[^/]+\.parquet\Z")
_ARCHIVE_RE = re.compile(r"panel_archive/(\d{4}-(?:0[1-9]|1[0-2]))/[^/]+\.parquet\Z")


def _validate_date(date: str, *, context: str) -> str:
    value = str(date)
    if not re.fullmatch(r"\d{8}", value):
        raise ValueError(f"{context} 须为恰 8 位数字日（YYYYMMDD），实得 {value!r}")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{context} 非法日期 {value!r}") from exc
    return value


def row_domain_for_date(date: str) -> str:
    """按信号日派生写侧行域，调用方不能覆盖该边界。"""
    value = _validate_date(date, context="行域日期")
    if value <= LABEL_FILTERED_THROUGH:
        return LABEL_FILTERED
    return STRUCTURAL


def _archive_month_domain(month: str, *, context: str) -> str:
    year, mon = (int(part) for part in month.split("-"))
    first = f"{year:04d}{mon:02d}01"
    last = f"{year:04d}{mon:02d}{calendar.monthrange(year, mon)[1]:02d}"
    if last <= LABEL_FILTERED_THROUGH:
        return LABEL_FILTERED
    if first >= STRUCTURAL_FROM:
        return STRUCTURAL
    raise ValueError(f"{context} 跨行域边界月 {month}，禁止按单月分区武断派生行域")


def row_domain_for_partition(rel_path: str) -> str:
    """从 panel 热区日分区或冷区月分区路径派生行域。"""
    value = str(rel_path).replace("\\", "/")
    match = _PANEL_RE.fullmatch(value)
    if match:
        return row_domain_for_date(match.group(1))
    match = _ARCHIVE_RE.fullmatch(value)
    if match:
        return _archive_month_domain(match.group(1), context=f"分区 {value}")
    raise ValueError(
        f"无法派生行域的分区路径 {rel_path!r}；须为 panel/YYYYMMDD/<group>.parquet "
        "或 panel_archive/YYYY-MM/<group>.parquet"
    )


def _validate_existing_domain(meta: Mapping[str, Any], expected: str, *, context: str) -> None:
    existing = meta.get("row_domain")
    if existing is not None and existing != expected:
        raise ValueError(f"{context} 行域冲突: 已登记 {existing!r} ≠ 派生值 {expected!r}")


def migrate_manifest_row_domains(data: dict[str, Any]) -> dict[str, int]:
    """显式迁移一份 manifest 内存数据，只补行域字段，不改 parquet 或 repairs。"""
    segments = data.get("row_domain_segments")
    if segments is not None and segments != ROW_DOMAIN_SEGMENTS:
        raise ValueError(f"row_domain_segments 冲突: {segments!r} ≠ {ROW_DOMAIN_SEGMENTS!r}")
    counts = {"panel": 0, "labels": 0}
    for rel_path, meta in data["partitions"].items():
        expected = row_domain_for_partition(rel_path)
        _validate_existing_domain(meta, expected, context=f"分区 {rel_path}")
        if meta.get("row_domain") is None:
            meta["row_domain"] = expected
            counts["panel"] += 1
    for label_name, entry in data["labels"].items():
        for filename, meta in entry["partitions"].items():
            if not re.fullmatch(r"\d{8}\.parquet", filename):
                raise ValueError(f"标签 {label_name} 分区文件名非法: {filename!r}")
            expected = row_domain_for_date(filename.removesuffix(".parquet"))
            context = f"标签 {label_name}/{filename}"
            _validate_existing_domain(meta, expected, context=context)
            if meta.get("row_domain") is None:
                meta["row_domain"] = expected
                counts["labels"] += 1
    data["row_domain_segments"] = dict(ROW_DOMAIN_SEGMENTS)
    return counts


def validate_manifest_row_domains(data: Mapping[str, Any]) -> dict[str, int]:
    """校验迁移后的行域 schema；缺字段、未知值或路径冲突均 fail-closed。"""
    segments = data.get("row_domain_segments")
    if segments != ROW_DOMAIN_SEGMENTS:
        raise ValueError(f"manifest 行域段缺失或冲突: {segments!r}；请先显式迁移")
    counts = {"panel": 0, "labels": 0}
    for rel_path, meta in data["partitions"].items():
        expected = row_domain_for_partition(rel_path)
        if meta.get("row_domain") != expected:
            raise ValueError(
                f"分区 {rel_path} 行域缺失或非法: {meta.get('row_domain')!r}，期望 {expected!r}"
            )
        counts["panel"] += 1
    for label_name, entry in data["labels"].items():
        for filename, meta in entry["partitions"].items():
            if not re.fullmatch(r"\d{8}\.parquet", filename):
                raise ValueError(f"标签 {label_name} 分区文件名非法: {filename!r}")
            expected = row_domain_for_date(filename.removesuffix(".parquet"))
            if meta.get("row_domain") != expected:
                raise ValueError(
                    f"标签 {label_name}/{filename} 行域缺失或非法: "
                    f"{meta.get('row_domain')!r}，期望 {expected!r}"
                )
            counts["labels"] += 1
    return counts
