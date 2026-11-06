# -*- coding: utf-8 -*-
"""质量长表聚合投影（设计 §5.8）。

口径（逐条对应设计）：
- ``metric_count/error_count/warning_count`` = 长表**指标行数**（不按 column 去重）；
- 数据集摘要按 ``(layer, dataset)`` 分组，组内错误/警告仍按指标行计数；
- ``partition_count/latest_partition/sync_watermark`` 各取该组 ``partition`` 为空且
  metric 匹配的**唯一** dataset 级行；重复/多值即 ``invalid``；
  ``partition_count`` 字符串必须严格解析为非负整数（``quality_value_not_numeric``），
  其余两项保留原字符串，``None`` 记 ``not_available``；
- 扫描时间、扫描窗口与“与上次变化”**当前没有来源**，固定 ``not_available/no_source``
  （不得用 mtime 冒充）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import status

__all__ = ["DATASET_SUMMARY_METRICS", "project_quality"]

#: dataset 级摘要行使用的 metric 名（设计 §5.8 点名的三项）
DATASET_SUMMARY_METRICS: tuple[str, ...] = (
    "partition_count",
    "latest_partition",
    "sync_watermark",
)
_ABNORMAL_STATUSES = ("error", "warning")


def _is_summary_row(row: Mapping[str, Any]) -> bool:
    """dataset 级摘要行：``partition`` 为空且 metric 属于三项摘要指标。"""
    partition = row.get("partition")
    empty_partition = partition is None or str(partition).strip() == ""
    return empty_partition and str(row.get("metric") or "") in DATASET_SUMMARY_METRICS


def _unique_summary_value(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> tuple[Any, dict[str, Any] | None]:
    """取组内某摘要指标的**唯一**原值；重复行（含同值）或多值一律冲突。

    R2-20261106-09：设计 §5.8 要求“唯一 dataset 级行，重复/多值 invalid”——
    同值重复也是重复行，不得当 available。
    """
    values = [row.get("value") for row in rows if str(row.get("metric") or "") == metric]
    if not values:
        return None, None
    distinct = {str(value) for value in values}
    if len(values) > 1:
        return None, status.make_status(
            "invalid",
            reason_code="multiple_quality_value",
            detail=(
                f"{metric} 出现 {len(values)} 个 dataset 级行"
                f"（不同值 {sorted(distinct)[:5]}）；契约要求唯一行"
            ),
        )
    return values[0], None


def _parse_partition_count(value: Any) -> tuple[int | None, dict[str, Any] | None]:
    """``partition_count`` 严格解析为非负整数（拒布尔/浮点/负数/非数字）。"""
    if value is None:
        return None, status.make_status(
            "not_available", reason_code="no_source", detail="partition_count 缺值"
        )
    if isinstance(value, bool):
        return None, status.make_status(
            "invalid", reason_code="quality_value_not_numeric", detail=f"partition_count={value!r}"
        )
    text = str(value).strip()
    if not text.isdigit():
        return None, status.make_status(
            "invalid", reason_code="quality_value_not_numeric", detail=f"partition_count={value!r}"
        )
    return int(text), None


def _dataset_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """单个 ``(layer, dataset)`` 分组的指标计数与三项摘要。"""
    summary_rows = [row for row in rows if _is_summary_row(row)]
    statuses = [str(row.get("status") or "") for row in rows]
    count_value, count_conflict = _unique_summary_value(summary_rows, "partition_count")
    partition_count, count_row = _parse_partition_count(count_value)
    if count_conflict is not None:  # 冲突/重复优先于“缺值”与解析结果
        count_row = count_conflict
    latest_value, latest_row = _unique_summary_value(summary_rows, "latest_partition")
    watermark_value, watermark_row = _unique_summary_value(summary_rows, "sync_watermark")
    return {
        "metric_count": len(rows),
        "error_count": statuses.count("error"),
        "warning_count": statuses.count("warning"),
        "partition_count": partition_count,
        "partition_count_status": count_row
        or status.make_status("available", detail="partition_count 严格解析为整数"),
        "latest_partition": None if latest_value is None else str(latest_value),
        "latest_partition_status": latest_row
        or (
            status.make_status(
                "not_available", reason_code="no_source", detail="latest_partition 缺行"
            )
            if latest_value is None
            else status.make_status("available", detail="latest_partition 原字符串")
        ),
        "sync_watermark": None if watermark_value is None else str(watermark_value),
        "sync_watermark_status": watermark_row
        or (
            status.make_status("not_available", reason_code="no_source", detail="sync_watermark 缺行")
            if watermark_value is None
            else status.make_status("available", detail="sync_watermark 原字符串")
        ),
    }


def project_quality(
    *, rows: Sequence[Mapping[str, Any]] | None, present: bool
) -> dict[str, Any]:
    """质量区投影；缺来源固定 ``not_available/no_source``（扫描时间等三项无来源）。"""
    no_source = status.make_status(
        "not_available", reason_code="no_source", detail="当前没有该质量字段的合法来源"
    )
    if not present or rows is None:
        row = status.make_status(
            "not_available", reason_code="no_source", detail="缺质量长表 latest_metrics"
        )
        row.update({"scan_time": dict(no_source), "scan_window": dict(no_source),
                    "change_vs_last": dict(no_source)})
        return row
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for entry in rows:
        if not isinstance(entry, Mapping):
            return status.make_status(
                "invalid", reason_code="invalid_json", detail="质量长表存在非对象行"
            )
        key = (str(entry.get("layer") or ""), str(entry.get("dataset") or ""))
        grouped.setdefault(key, []).append(entry)
    datasets: list[dict[str, Any]] = []
    for (layer, dataset), group in sorted(grouped.items()):
        summary = _dataset_summary(group)
        summary.update({"layer": layer, "dataset": dataset})
        datasets.append(summary)
    statuses = [str(entry.get("status") or "") for entry in rows]
    row = status.make_status(
        "available",
        detail=f"{len(rows)} 条质量指标行 / {len(datasets)} 个数据集",
        extra={
            "metric_count": len(rows),
            "error_count": statuses.count(_ABNORMAL_STATUSES[0]),
            "warning_count": statuses.count(_ABNORMAL_STATUSES[1]),
            "datasets": datasets,
        },
    )
    row.update({"scan_time": dict(no_source), "scan_window": dict(no_source),
                "change_vs_last": dict(no_source)})
    return row
