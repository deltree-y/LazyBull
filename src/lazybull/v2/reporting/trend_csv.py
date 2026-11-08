# -*- coding: utf-8 -*-
"""趋势面两份完整 CSV 的 schema、排序与键集守恒（设计 §5.7）。

硬口径：

- ``run_index.csv``：每个发现 run **恰一行**（含 ``invalid``），行键 ``run_key``，
  按 ``(window_end, run_id, run_key)`` 稳定升序，``index_mode`` 恒为 ``full``；
- ``trend.csv``：行键 ``(run_key, metric_name)``；每个 run 必须**恰有**五类指标行
  （缺失也留状态行），不插值、不抽样、不改行键；
- 两份文件 UTF-8 无 BOM / LF / 固定列序；守恒校验在本模块唯一实现（`assert_key_conservation`）；
- 本模块只做序列化、排序与守恒，不计算指标。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import trend_contract
from src.lazybull.v2.reporting.csv_core import (
    CsvContractError,
    parse_table,
    render_table,
    verify_table,
)

__all__ = [
    "FILE_RUN_INDEX",
    "FILE_TREND",
    "TREND_FILENAMES",
    "assert_key_conservation",
    "build_trend_bundle",
    "parse_run_index",
    "parse_trend",
    "render_run_index",
    "render_trend",
    "sort_index_rows",
    "sort_trend_rows",
    "verify_run_index",
    "verify_trend",
]

#: run 索引文件名（companion 链接与 manifest outputs 同源）
FILE_RUN_INDEX = "run_index.csv"
#: 趋势指标文件名
FILE_TREND = "trend.csv"
#: 趋势 run 的固定 CSV 文件集
TREND_FILENAMES: tuple[str, ...] = (FILE_RUN_INDEX, FILE_TREND)


def sort_index_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按 ``(window_end, run_id, run_key)`` 稳定升序（缺失排序键显式写空串）。"""
    material = [dict(row) for row in rows]
    return sorted(
        material,
        key=lambda row: (
            str(row.get("window_end") or ""),
            str(row.get("run_id") or ""),
            str(row.get("run_key") or ""),
        ),
    )


def sort_trend_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按 ``(run_key, 指标位次)`` 稳定升序（指标位次来自契约闭集）。"""
    material = [dict(row) for row in rows]
    return sorted(
        material,
        key=lambda row: (
            str(row.get("run_key") or ""),
            trend_contract.metric_order(str(row.get("metric_name") or "")),
        ),
    )


def render_run_index(rows: Sequence[Mapping[str, Any]]) -> str:
    """渲染 ``run_index.csv``（调用方应先 ``sort_index_rows``）。"""
    return render_table(
        FILE_RUN_INDEX,
        trend_contract.RUN_INDEX_COLUMNS,
        trend_contract.RUN_INDEX_KEYS,
        rows,
    )


def render_trend(rows: Sequence[Mapping[str, Any]]) -> str:
    """渲染 ``trend.csv``（调用方应先 ``sort_trend_rows``）。"""
    return render_table(
        FILE_TREND,
        trend_contract.TREND_COLUMNS,
        trend_contract.TREND_KEYS,
        rows,
    )


def verify_run_index(text: str, *, expected_rows: int | None = None) -> dict[str, Any]:
    """回读校验 ``run_index.csv``（列序/行数/键唯一/LF/无 BOM）。"""
    return verify_table(
        FILE_RUN_INDEX,
        trend_contract.RUN_INDEX_COLUMNS,
        trend_contract.RUN_INDEX_KEYS,
        text,
        expected_rows=expected_rows,
    )


def verify_trend(text: str, *, expected_rows: int | None = None) -> dict[str, Any]:
    """回读校验 ``trend.csv``。"""
    return verify_table(
        FILE_TREND,
        trend_contract.TREND_COLUMNS,
        trend_contract.TREND_KEYS,
        text,
        expected_rows=expected_rows,
    )


def parse_run_index(text: str) -> list[dict[str, str]]:
    """把已发布 ``run_index.csv`` 解析回文本行（继承上一快照时使用）。"""
    return parse_table(
        FILE_RUN_INDEX,
        trend_contract.RUN_INDEX_COLUMNS,
        trend_contract.RUN_INDEX_KEYS,
        text,
    )


def parse_trend(text: str) -> list[dict[str, str]]:
    """把已发布 ``trend.csv`` 解析回文本行（继承时逐字节保留原值文本）。"""
    return parse_table(
        FILE_TREND,
        trend_contract.TREND_COLUMNS,
        trend_contract.TREND_KEYS,
        text,
    )


def assert_key_conservation(
    index_rows: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
) -> None:
    """键集守恒：run 行键唯一；每个 run 恰有五类指标行；无孤儿趋势行。

    Raises:
        CsvContractError: 重复 run_key / 缺指标 / 多指标 / 孤儿 trend 行 / 指纹缺失。
    """
    run_keys: list[str] = []
    for row in index_rows:
        key = str(row.get("run_key") or "")
        if not key:
            raise CsvContractError("run_index 行缺 run_key（身份未登记）")
        run_keys.append(key)
    duplicated = sorted({key for key in run_keys if run_keys.count(key) > 1})
    if duplicated:
        raise CsvContractError(f"run_index 行键重复：{duplicated[:5]}")
    expected = set(trend_contract.METRIC_NAMES)
    seen: dict[str, set[str]] = {key: set() for key in run_keys}
    for row in trend_rows:
        key = str(row.get("run_key") or "")
        metric = str(row.get("metric_name") or "")
        if key not in seen:
            raise CsvContractError(f"trend 行引用未登记 run_key：{key!r}（孤儿趋势行）")
        if metric not in expected:
            raise CsvContractError(f"trend 行含未登记指标：{metric!r}")
        if metric in seen[key]:
            raise CsvContractError(f"trend 行键重复：({key}, {metric})")
        seen[key].add(metric)
    for key, metrics in seen.items():
        missing = sorted(expected - metrics)
        if missing:
            raise CsvContractError(f"run_key {key} 缺指标行 {missing}（缺失也必须留状态行）")


def build_trend_bundle(
    index_rows: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
    *,
    expected_index_rows: int | None = None,
) -> dict[str, Any]:
    """生成两份 CSV 的字节与 manifest 登记项（含键集守恒与行数门）。

    Returns:
        ``{"files": {name: bytes}, "schemas": {name: {...}}, "digests": {name: sha}}``
    """
    assert_key_conservation(index_rows, trend_rows)
    index_text = render_run_index(sort_index_rows(index_rows))
    trend_text = render_trend(sort_trend_rows(trend_rows))
    index_schema = verify_run_index(index_text, expected_rows=expected_index_rows)
    expected_metric_rows = len(index_rows) * len(trend_contract.METRICS)
    trend_schema = verify_trend(trend_text, expected_rows=expected_metric_rows)
    files = {
        FILE_RUN_INDEX: index_text.encode("utf-8"),
        FILE_TREND: trend_text.encode("utf-8"),
    }
    schemas = {FILE_RUN_INDEX: index_schema, FILE_TREND: trend_schema}
    digests = {name: schemas[name]["sha256"] for name in TREND_FILENAMES}
    return {"files": files, "schemas": schemas, "digests": digests}
