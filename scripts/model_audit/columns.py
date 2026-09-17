# -*- coding: utf-8 -*-
"""模型列集审计：读取各模型版本的特征清单，生成列集漂移台账 / 出现频次 / 跨来源差异。"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from .constants import (
    DEFAULT_LAST_VERSIONS,
    FEATURE_FILE_GLOB,
    STATUS_ALWAYS,
    STATUS_INTERMITTENT,
    STATUS_NEW,
    STATUS_REMOVED,
)

_VERSION_PATTERN = re.compile(r"v(\d+)")


@dataclass(frozen=True)
class SourceSpec:
    """一个待审计来源（目录 + 文件名模式）。"""

    label: str
    root: Path
    pattern: str = FEATURE_FILE_GLOB


@dataclass(frozen=True)
class SourceColumns:
    """单个模型版本（或折目录）的特征清单。"""

    label: str
    name: str
    path: Path
    columns: Tuple[str, ...]
    mtime: float


def parse_source_arg(text: str) -> SourceSpec:
    """解析 ``LABEL=PATH`` 或 ``PATH``（缺省 label 用目录名）。"""
    raw = str(text).strip()
    if not raw:
        raise ValueError("来源参数不能为空")
    if "=" in raw:
        label, _, path_text = raw.partition("=")
        label = label.strip()
        path_text = path_text.strip()
        if not label or not path_text:
            raise ValueError(f"来源参数格式非法（应为 LABEL=PATH）: {text}")
        return SourceSpec(label=label, root=Path(path_text))
    path = Path(raw)
    return SourceSpec(label=path.name or str(path), root=path)


def _version_sort_key(name: str) -> Tuple[int, float, str]:
    match = _VERSION_PATTERN.search(name)
    version = int(match.group(1)) if match else -1
    return (version, 0.0, name)


def load_feature_columns(path: Path) -> List[str]:
    """读取特征清单 JSON（应为字符串数组），返回去重后的列名列表。"""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"特征清单解析失败: {path}: {exc}") from exc
    if isinstance(payload, dict):  # 兼容 {"feature_columns": [...]} 的旧格式
        payload = payload.get("feature_columns", [])
    if not isinstance(payload, list):
        raise ValueError(f"特征清单格式非法（应为字符串数组）: {path}")
    columns: List[str] = []
    seen = set()
    for item in payload:
        name = str(item)
        if name not in seen:
            seen.add(name)
            columns.append(name)
    if not columns:
        raise ValueError(f"特征清单为空: {path}")
    return columns


def scan_source(spec: SourceSpec, last: Optional[int] = None) -> List[SourceColumns]:
    """扫描一个来源下的全部特征清单，按版本号升序返回（``last`` 限制尾部条数）。"""
    root = Path(spec.root)
    if not root.is_dir():
        raise FileNotFoundError(f"来源目录不存在: {root}")
    paths = sorted(
        (p for p in root.glob(spec.pattern) if p.is_file()),
        key=lambda p: _version_sort_key(p.name),
    )
    if not paths:
        raise FileNotFoundError(f"来源目录下未找到匹配 {spec.pattern} 的文件: {root}")
    if last is not None and last > 0:
        paths = paths[-last:]
    items: List[SourceColumns] = []
    for path in paths:
        items.append(
            SourceColumns(
                label=spec.label,
                name=(
                    path.name[: -len("_features.json")]
                    if path.name.endswith("_features.json")
                    else path.stem
                ),
                path=path,
                columns=tuple(load_feature_columns(path)),
                mtime=path.stat().st_mtime,
            )
        )
    logger.info(f"来源 {spec.label}: 扫描到 {len(items)} 个特征清单（{root}）")
    return items


def presence_matrix(items: Sequence[SourceColumns]) -> pd.DataFrame:
    """列集存在矩阵：行 = 来源/版本，列 = 全部出现过的特征（布尔）。"""
    if not items:
        raise ValueError("存在矩阵构建失败: 输入为空")
    names = sorted({name for item in items for name in item.columns})
    index = pd.MultiIndex.from_tuples(
        [(item.label, item.name) for item in items], names=["来源", "版本"]
    )
    data = [[name in item.columns for name in names] for item in items]
    matrix = pd.DataFrame(data, index=index, columns=names)
    return matrix.sort_index(level=0, sort_remaining=False)


def drift_ledger(items: Sequence[SourceColumns]) -> pd.DataFrame:
    """漂移台账：每个版本相对**同来源上一版本**的列集变化（版本号升序比较）。"""
    rows: List[Dict[str, object]] = []
    by_label: Dict[str, List[SourceColumns]] = {}
    for item in items:
        by_label.setdefault(item.label, []).append(item)
    for label, label_items in by_label.items():
        ordered = sorted(label_items, key=lambda i: _version_sort_key(i.name))
        previous: Optional[SourceColumns] = None
        for item in ordered:
            current = set(item.columns)
            if previous is None:
                added, removed = [], []
            else:
                prev_set = set(previous.columns)
                added = sorted(current - prev_set)
                removed = sorted(prev_set - current)
            rows.append(
                {
                    "来源": label,
                    "版本": item.name,
                    "列数": len(item.columns),
                    "相对上一版本新增数": len(added),
                    "相对上一版本移除数": len(removed),
                    "新增列": ", ".join(added),
                    "移除列": ", ".join(removed),
                }
            )
            previous = item
    return pd.DataFrame(rows)


def frequency_table(items: Sequence[SourceColumns]) -> pd.DataFrame:
    """列集出现频次：按来源统计每列出现的版本数、首末版本与状态。"""
    rows: List[Dict[str, object]] = []
    by_label: Dict[str, List[SourceColumns]] = {}
    for item in items:
        by_label.setdefault(item.label, []).append(item)
    for label, label_items in by_label.items():
        ordered = sorted(label_items, key=lambda i: _version_sort_key(i.name))
        total = len(ordered)
        latest_name = ordered[-1].name
        occurrence: Dict[str, List[str]] = {}
        for item in ordered:
            for name in item.columns:
                occurrence.setdefault(name, []).append(item.name)
        for name, versions in occurrence.items():
            count = len(versions)
            if count == total:
                status = STATUS_ALWAYS
            elif count == 1 and versions[0] == latest_name:
                status = STATUS_NEW
            elif versions[-1] != latest_name:
                status = STATUS_REMOVED
            else:
                status = STATUS_INTERMITTENT
            rows.append(
                {
                    "来源": label,
                    "特征": name,
                    "出现版本数": count,
                    "版本总数": total,
                    "出现占比": round(count / total, 4) if total else 0.0,
                    "首次出现版本": versions[0],
                    "末次出现版本": versions[-1],
                    "最新版本保留": bool(versions[-1] == latest_name),
                    "状态": status,
                }
            )
    return pd.DataFrame(rows)


def configuration_groups(items: Sequence[SourceColumns]) -> pd.DataFrame:
    """列集配置分组：同一来源内按**列集完全一致**聚类。

    同一模型目录会累积多套实验配置（不同开关/不同特征集），逐版本 diff 会把
    「换配置」误读成「掉列」；先分组才能判断真正的漂移。
    """
    by_label: Dict[str, List[SourceColumns]] = {}
    for item in items:
        by_label.setdefault(item.label, []).append(item)
    rows: List[Dict[str, object]] = []
    for label, label_items in by_label.items():
        ordered = sorted(label_items, key=lambda i: _version_sort_key(i.name))
        groups: Dict[frozenset, List[SourceColumns]] = {}
        for item in ordered:
            groups.setdefault(frozenset(item.columns), []).append(item)
        group_items = sorted(groups.values(), key=lambda members: -len(members))
        for index, members in enumerate(group_items, start=1):
            rows.append(
                {
                    "来源": label,
                    "配置ID": f"{label}-C{index}",
                    "列数": len(next(iter(members)).columns),
                    "版本数": len(members),
                    "版本列表": ", ".join(item.name for item in members),
                }
            )
    return pd.DataFrame(rows)


def config_diff_vs_latest(items: Sequence[SourceColumns]) -> pd.DataFrame:
    """各配置相对**同来源最新版本所属配置**的列差异（区分「开关差异」与「掉列」）。"""
    by_label: Dict[str, List[SourceColumns]] = {}
    for item in items:
        by_label.setdefault(item.label, []).append(item)
    rows: List[Dict[str, object]] = []
    for label, label_items in by_label.items():
        ordered = sorted(label_items, key=lambda i: _version_sort_key(i.name))
        reference = set(ordered[-1].columns)
        groups: Dict[frozenset, List[SourceColumns]] = {}
        for item in ordered:
            groups.setdefault(frozenset(item.columns), []).append(item)
        group_items = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        for index, (columns, members) in enumerate(group_items, start=1):
            missing = sorted(reference - set(columns))
            extra = sorted(set(columns) - reference)
            rows.append(
                {
                    "来源": label,
                    "配置ID": f"{label}-C{index}",
                    "列数": len(columns),
                    "版本数": len(members),
                    "相对最新配置缺失列数": len(missing),
                    "相对最新配置多出列数": len(extra),
                    "缺失列": ", ".join(missing),
                    "多出列": ", ".join(extra),
                }
            )
    return pd.DataFrame(rows)


def with_family(table: pd.DataFrame, column: str = "特征") -> pd.DataFrame:
    """为频次表补充家族列（复用因子体检的家族口径，``has_*`` 归可用性标记）。"""
    from scripts.factor_health import family_of

    if table.empty:
        table["家族"] = []
        return table
    result = table.copy()
    result["家族"] = [family_of(str(name)) for name in result[column]]
    return result


def cross_source_diff(items: Sequence[SourceColumns]) -> pd.DataFrame:
    """跨来源列集差异：两两比较各来源**最新版本**（与并集口径）的列集差别。"""
    latest: Dict[str, SourceColumns] = {}
    union: Dict[str, set] = {}
    for item in items:
        union.setdefault(item.label, set()).update(item.columns)
        current = latest.get(item.label)
        if current is None or _version_sort_key(item.name) > _version_sort_key(current.name):
            latest[item.label] = item
    labels = sorted(latest.keys())
    rows: List[Dict[str, object]] = []
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            left_cols = set(latest[left].columns)
            right_cols = set(latest[right].columns)
            union_cols = union[left] | union[right]
            rows.append(
                {
                    "来源A": left,
                    "版本A": latest[left].name,
                    "来源B": right,
                    "版本B": latest[right].name,
                    "A独有列数": len(left_cols - right_cols),
                    "B独有列数": len(right_cols - left_cols),
                    "共有列数": len(left_cols & right_cols),
                    "并集列数": len(union_cols),
                    "A独有列": ", ".join(sorted(left_cols - right_cols)),
                    "B独有列": ", ".join(sorted(right_cols - left_cols)),
                }
            )
    return pd.DataFrame(rows)


def removed_column_summary(frequency: pd.DataFrame) -> pd.DataFrame:
    """按家族汇总「已移除 / 间断出现」列（回答「哪个家族的列在漂移」）。"""
    if frequency.empty:
        return pd.DataFrame()
    mask = frequency["状态"].isin([STATUS_REMOVED, STATUS_INTERMITTENT])
    if not mask.any():
        return pd.DataFrame()
    grouped = (
        frequency[mask]
        .groupby(["来源", "家族", "状态"])
        .agg(列数=("特征", "size"), 列=("特征", lambda s: ", ".join(sorted(s))))
        .reset_index()
        .sort_values(["来源", "列数"], ascending=[True, False])
    )
    return grouped


def build_audit_tables(
    items: Sequence[SourceColumns],
    include_matrix: bool = True,
) -> Dict[str, pd.DataFrame]:
    """组装全部审计表（键为中文产物名）。"""
    tables: Dict[str, pd.DataFrame] = {}
    tables["列集配置分组.csv"] = configuration_groups(items)
    tables["配置差异对比.csv"] = config_diff_vs_latest(items)
    tables["列集漂移台账.csv"] = drift_ledger(items)
    tables["列集出现频次.csv"] = with_family(frequency_table(items))
    diff = cross_source_diff(items)
    if not diff.empty:
        tables["跨来源列集差异.csv"] = diff
    tables["漂移家族汇总.csv"] = removed_column_summary(tables["列集出现频次.csv"])
    if include_matrix:
        tables["列集存在矩阵.csv"] = presence_matrix(items)
    return {name: table for name, table in tables.items() if not table.empty}


def iter_source_items(
    specs: Iterable[SourceSpec], last: Optional[int] = DEFAULT_LAST_VERSIONS
) -> List[SourceColumns]:
    """扫描全部来源。"""
    items: List[SourceColumns] = []
    for spec in specs:
        items.extend(scan_source(spec, last=last))
    if not items:
        raise ValueError("未扫描到任何特征清单")
    return items
