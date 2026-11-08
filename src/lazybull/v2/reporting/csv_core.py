# -*- coding: utf-8 -*-
"""完整 CSV 的通用序列化与守恒原语（实验面五份 CSV 与趋势面两份 CSV 共用）。

硬口径（设计 §5.4 / §5.7，P6 T3/T4 同一口径）：

- UTF-8 无 BOM、**LF**、固定列序、非抽样；空值写空串（禁止 ``-``/0 掩盖缺失）；
- 行键唯一（重复即失败）；每行必须给出**全部声明列**（缺列/多列均失败）；
- 任何 NaN/Inf 失败；单元格文本不得含换行（破坏一物理行一记录）；
- 回读校验覆盖列序/行数/键唯一/无 CR/无 BOM/以单 LF 结尾。

本模块只做序列化与守恒校验，**不计算任何指标**；具体文件的列集/行键由各面注册表给出
（实验面 ``experiment_csv``、趋势面 ``trend_csv``）。唯一实现只有这一份，禁止各面复制。
"""

from __future__ import annotations

import csv
import io
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "CsvContractError",
    "assert_unique_columns",
    "cell_text",
    "key_text",
    "parse_table",
    "render_table",
    "verify_table",
]


class CsvContractError(ValueError):
    """CSV 守恒契约违反（缺列/多列/重复键/NaN/Inf/行数不符/换行/BOM/CR）。"""


def cell_text(value: Any, *, column: str, row_index: int) -> str:
    """单元格文本：``None`` → 空；NaN/Inf/缺失一律失败（不静默写空）。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise CsvContractError(
                f"第 {row_index} 行列 {column} 含非有限值 {value!r}（NaN/Inf 必须为 None）"
            )
        return repr(number)
    text = str(value)
    if "\r" in text or "\n" in text:
        raise CsvContractError(f"第 {row_index} 行列 {column} 含换行（破坏一物理行一记录）")
    return text


def key_text(value: Any) -> str:
    """行键文本（与 CSV 单元格同一口径，保证键唯一性与文件一致）。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def assert_unique_columns(name: str, columns: Sequence[str]) -> None:
    """声明列集自身不得重名（表头重名会同时污染 manifest schema）。"""
    duplicated = sorted({column for column in columns if list(columns).count(column) > 1})
    if duplicated:
        raise CsvContractError(f"{name} 声明列集存在重复列名：{duplicated}")


def _check_keys(name: str, columns: Sequence[str], keys: Sequence[str]) -> None:
    """行键列必须落在声明列集内（否则行键无法从文件回读）。"""
    missing_keys = [column for column in keys if column not in columns]
    if missing_keys:
        raise CsvContractError(f"{name} 行键列不在声明列集内：{missing_keys}")


def _row_values(
    name: str,
    columns: Sequence[str],
    row: Mapping[str, Any],
    *,
    row_index: int,
) -> list[Any]:
    """行值校验（缺列/多列即失败；列序由声明列集决定）。"""
    missing = [column for column in columns if column not in row]
    if missing:
        raise CsvContractError(f"{name} 第 {row_index} 行缺列 {missing}（缺列即失败）")
    extra = [column for column in row if column not in columns]
    if extra:
        raise CsvContractError(f"{name} 第 {row_index} 行含未声明列 {extra}（列序不可守恒）")
    return [row[column] for column in columns]


def render_table(
    name: str,
    columns: Sequence[str],
    keys: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """渲染单份 CSV 文本（固定列序、LF、重复键失败）。"""
    assert_unique_columns(name, columns)
    _check_keys(name, columns, keys)
    key_indexes = [list(columns).index(column) for column in keys]
    seen: set[tuple[str, ...]] = set()
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(list(columns))
    for index, row in enumerate(rows):
        values = _row_values(name, columns, row, row_index=index)
        key = tuple(key_text(values[position]) for position in key_indexes)
        if key in seen:
            raise CsvContractError(f"{name} 行键重复：{key}")
        seen.add(key)
        writer.writerow(
            [
                cell_text(value, column=columns[position], row_index=index)
                for position, value in enumerate(values)
            ]
        )
    return buffer.getvalue()


def _read_rows(name: str, text: str) -> tuple[list[str], list[list[str]]]:
    """回读基础校验：无 CR/BOM、以单 LF 结尾、有表头、行列数与表头一致。"""
    if "\r" in text:
        raise CsvContractError(f"{name} 含 CR（必须 LF 换行）")
    if text.startswith("\ufeff"):
        raise CsvContractError(f"{name} 含 BOM（必须 UTF-8 无 BOM）")
    if not text.endswith("\n"):
        raise CsvContractError(f"{name} 未以单个 LF 结尾")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise CsvContractError(f"{name} 无表头")
    header = rows[0]
    body = rows[1:]
    for index, row in enumerate(body):
        if len(row) != len(header):
            raise CsvContractError(f"{name} 第 {index} 行列数 {len(row)} 与表头不符")
    return header, body


def verify_table(
    name: str,
    columns: Sequence[str],
    keys: Sequence[str],
    text: str,
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """回读校验：列序/行数/键唯一/无 CR/无 BOM；返回登记项（不重算指标）。"""
    assert_unique_columns(name, columns)
    _check_keys(name, columns, keys)
    header, body = _read_rows(name, text)
    if tuple(header) != tuple(columns):
        raise CsvContractError(f"{name} 列序与声明不一致：{header}")
    if expected_rows is not None and len(body) != expected_rows:
        raise CsvContractError(f"{name} 行数 {len(body)} 与期望 {expected_rows} 不符")
    positions = [list(columns).index(column) for column in keys]
    seen: set[tuple[str, ...]] = set()
    for index, row in enumerate(body):
        key = tuple(row[position] for position in positions)
        if key in seen:
            raise CsvContractError(f"{name} 行键重复：{key}")
        seen.add(key)
    return {
        "columns": list(columns),
        "row_count": len(body),
        "key_columns": list(keys),
        "sha256": sha256(text.encode("utf-8")).hexdigest(),
    }


def parse_table(
    name: str,
    columns: Sequence[str],
    keys: Sequence[str],
    text: str,
) -> list[dict[str, str]]:
    """把已发布 CSV 解析回行字典（继承上一快照时使用；同样执行守恒校验）。

    返回的行值全部是**文件里的原文本**（不转 float），保证“继承行字节不变”的可核验性。
    """
    verify_table(name, columns, keys, text)
    header, body = _read_rows(name, text)
    return [dict(zip(header, row)) for row in body]
