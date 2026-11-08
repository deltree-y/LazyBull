# -*- coding: utf-8 -*-
"""趋势投影装配（P6 T4：上一快照 + 新 run → 规范索引/趋势行 → HTML 视图数据）。

硬口径（设计 §4.3 / §5.7）：

- **历史不覆盖**：上一快照的行按 ``run_key`` 原样继承（不重算、不改值、不删除）；
  输入根中消失的 run 仍保留历史行（不视为撤回）；
- **键集守恒**：每个 run 恰一行索引 + 恰五条指标行（缺失也留状态行）；继承行与新行
  在同一守恒门内校验，任何孤儿/缺口即失败；
- **可见窗口**：HTML 只取按 ``(window_end, run_id, run_key)`` 升序的最后
  ``VISIBLE_ROWS_MAX`` 行；完整索引与完整趋势值只在 CSV（不抽样）；
- 本模块只读已解析行与观测，不做 I/O、不计算指标。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import status, trend_contract, trend_csv

__all__ = [
    "assert_no_identity_conflict",
    "build_trend_projection",
    "merge_rows",
    "visible_rows",
]

_CALIBRATION_MDD_NOTE = (
    "B0 冻结源窗口基准 MDD（500 收益样本，log→pp；不是策略回撤读数，"
    "也不是自然日序列回撤）"
)


def _text(value: Any) -> str | None:
    """文本归一：``None``/空串 → ``None``（不写空串冒充值）。"""
    if value is None:
        return None
    text = str(value)
    return text or None


def _number(value: Any) -> float | None:
    """严格数值归一（只接受可解析文本/有限数；解析失败返回 ``None``，不猜值）。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def assert_no_identity_conflict(
    previous_rows: Sequence[Mapping[str, Any]], fresh_rows: Sequence[Mapping[str, Any]]
) -> None:
    """同一来源身份不得以不同 ``run_key`` 再现（root_id 由来源根路径派生，换根即换身份）。"""
    known = {
        (str(row.get("run_id")), str(row.get("source_identity_sha256")))
        for row in previous_rows
        if row.get("source_identity_sha256")
    }
    conflicts = sorted(
        {
            (str(row.get("run_id")), str(row.get("source_identity_sha256")))
            for row in fresh_rows
            if row.get("source_identity_sha256")
            and (str(row.get("run_id")), str(row.get("source_identity_sha256"))) in known
        }
    )
    if conflicts:
        raise ValueError(
            "同一来源身份出现在不同 root_id（来源根路径变化视为不同输入）："
            + "、".join(f"{run_id}/{digest[:12]}" for run_id, digest in conflicts[:5])
        )


def _metric_catalog(evidence_class: str) -> list[dict[str, Any]]:
    """该类别的五类指标目录（单位/日期语义按类分列，禁止跨类互转）。"""
    catalog: list[dict[str, Any]] = []
    for spec in trend_contract.metrics_for(evidence_class):
        unit = trend_contract.unit_for(str(spec["name"]), evidence_class)
        note = str(spec["note"])
        if evidence_class == "calibration_only" and str(spec["name"]) == "max_drawdown":
            note = _CALIBRATION_MDD_NOTE  # 防误读为策略回撤（口径不改、数值不改）
        catalog.append(
            {
                "name": str(spec["name"]),
                "label": str(spec["label"]),
                "note": note,
                "unit": unit,
                "unit_label": trend_contract.unit_label(unit),
                "generated": unit is not None,
            }
        )
    return catalog


def merge_rows(
    previous_rows: Sequence[Mapping[str, Any]],
    fresh_rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """按行键合并（**上一快照优先**：同键不覆盖，保证历史值字节不变）。"""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in list(previous_rows) + list(fresh_rows):
        key = tuple(str(row.get(field) or "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(row))
    return merged


def visible_rows(
    index_rows: Sequence[Mapping[str, Any]], *, limit: int = trend_contract.VISIBLE_ROWS_MAX
) -> list[dict[str, Any]]:
    """HTML 可见窗口：升序排序后的最后 ``limit`` 行（总数不足则全量）。"""
    ordered = trend_csv.sort_index_rows(index_rows)
    if limit <= 0:
        raise ValueError(f"可见行上限必须为正：{limit}")
    return ordered[-limit:]


def _series(
    metric_name: str,
    visible: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """单指标的图序列（按可见行顺序；无值点保留位置以形成断线，不插值）。"""
    by_key = {
        str(row.get("run_key")): row
        for row in trend_rows
        if str(row.get("metric_name")) == metric_name
    }
    points: list[dict[str, Any]] = []
    for row in visible:
        key = str(row.get("run_key"))
        metric = dict(by_key.get(key) or {})
        # CSV 继承行保留原始文本（字节不变）；投影/图/表只接受**严格可解析**的数值，
        # 解析失败仍是 None（不插值、不补 0、不做猜测性转换）。
        value = _number(metric.get("value"))
        points.append(
            {
                "run_key": key,
                "run_id": _text(row.get("run_id")),
                "window_end": _text(row.get("window_end")),
                "value": value,
                "unit": _text(metric.get("unit")),
                "report_status": _text(metric.get("report_status")),
                "degradation_reason_code": _text(metric.get("degradation_reason_code")),
                "source_ref": _text(metric.get("source_ref")),
            }
        )
    return points


def _metric_summaries(
    catalog: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """每指标的全量状态分布（可用行数/各状态行数；用于页面事实与降级面）。"""
    summaries: list[dict[str, Any]] = []
    for metric in catalog:
        rows = [
            dict(row) for row in trend_rows if str(row.get("metric_name")) == metric["name"]
        ]
        by_status: dict[str, int] = {}
        reasons: dict[str, int] = {}
        for row in rows:
            word = str(row.get("report_status") or "not_available")
            by_status[word] = by_status.get(word, 0) + 1
            code = str(row.get("degradation_reason_code") or "")
            if code and not status.is_usable(row):
                reasons[code] = reasons.get(code, 0) + 1
        usable = sum(1 for row in rows if status.is_usable(row))
        # 有值行：包含“区级 degraded 但当前点已核对”的数值（R3-N1-R3 汇总与渲染同口径）
        with_value = sum(1 for row in rows if _number(row.get("value")) is not None)
        summaries.append(
            {
                **dict(metric),
                "row_count": len(rows),
                "usable_rows": usable,
                "with_value_rows": with_value,
                "status_rows": len(rows) - with_value,
                "by_status": dict(sorted(by_status.items())),
                "reason_codes": dict(sorted(reasons.items())),
            }
        )
    return summaries


def _index_status(
    total: int,
    index_rows: Sequence[Mapping[str, Any]],
    authority_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """索引级状态：空 index 默认合法空；权威门未过时不得报合法空（R3-01-R3）。"""
    if total == 0:
        if authority_gate is not None and not status.is_usable(authority_gate):
            return dict(authority_gate)
        return status.make_status(
            "empty_valid", detail="未发现任何 run；完整索引为合法空（visible_rows=0）"
        )
    severity = {"available": 0, "empty_valid": 0, "stale": 1, "degraded": 2,
                "not_available": 3, "invalid": 4}
    worst = max(
        (str(row.get("report_status") or "not_available") for row in index_rows),
        key=lambda word: severity.get(word, 3),
    )
    if authority_gate is not None and not status.is_usable(authority_gate):
        gate_word = str(authority_gate.get("report_status") or "degraded")
        if severity.get(gate_word, 3) >= severity.get(worst, 3):
            return dict(authority_gate)
    reason = next(
        (
            str(row.get("degradation_reason_code"))
            for row in index_rows
            if str(row.get("report_status")) == worst and row.get("degradation_reason_code")
        ),
        None,
    )
    detail = f"{total} 个 run；最严重状态 {worst}"
    if worst == "invalid":
        return status.make_status("invalid", reason_code=reason or "source_contract_violation",
                                  detail=detail)
    if worst in ("degraded", "not_available", "stale"):
        return status.make_status(
            worst, reason_code=reason or "missing_precondition", detail=detail
        )
    return status.make_status("available", detail=detail)


def _manifest_absence(index_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """无 manifest 的 run 统计（``source_manifest_sha256=null`` 且必须标 manifest_absent）。"""
    missing = sorted(
        str(row.get("run_key"))
        for row in index_rows
        if not str(row.get("source_manifest_sha256") or "").strip()
    )
    return {
        "flag": "manifest_absent",
        "count": len(missing),
        "sample": missing[:10],
        "rule": "无 manifest 时身份 = 实际消费白名单文件 path+sha 的 canonical 哈希",
    }


def _data_window(index_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """数据窗口（各 run 窗口的 min/max；缺失不推算）。"""
    starts = sorted(str(row.get("window_start")) for row in index_rows if row.get("window_start"))
    ends = sorted(str(row.get("window_end")) for row in index_rows if row.get("window_end"))
    return {"start": starts[0] if starts else None, "end": ends[-1] if ends else None}


def build_trend_projection(
    *,
    evidence_class: str,
    generated_at: str,
    root_map: Mapping[str, str],
    index_rows: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
    observations: Sequence[Any],
    previous: Mapping[str, Any] | None,
    inherited_roots: Sequence[str] = (),
    inherited_source: Mapping[str, Any] | None = None,
    authority_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """装配趋势投影（可见行 + 五指标序列 + 统计 + 继承与降级登记）。"""
    trend_csv.assert_key_conservation(index_rows, trend_rows)
    catalog = _metric_catalog(evidence_class)
    ordered = trend_csv.sort_index_rows(index_rows)
    visible = visible_rows(ordered)
    summaries = _metric_summaries(catalog, trend_rows)
    inheritance = {
        "mode": "incremental" if previous is not None else "full_build",
        "previous_report_id": None if previous is None else previous.get("report_id"),
        "previous_manifest_sha256": (
            None if previous is None else previous.get("manifest", {}).get("record_sha256")
        ),
        "previous_manifest_bytes_sha256": (
            None if previous is None else previous.get("manifest_sha256")
        ),
        "inherited_runs": 0 if previous is None else len(previous.get("index_rows") or []),
        "new_runs": len(observations),
        "inherited_roots": sorted(str(item) for item in inherited_roots),
        "inherited_source": None if inherited_source is None else dict(inherited_source),
    }
    degradations: list[dict[str, Any]] = []
    for observation in observations:
        degradations.extend(dict(item) for item in observation.degradations)
    if authority_gate is not None:
        degradations.append(
            status.degradation_entry(authority_gate, field_path="authority")
        )
    return {
        "schema": trend_contract.TREND_SCHEMA,
        "generated_at": str(generated_at),
        "evidence_class": str(evidence_class),
        "index_mode": trend_contract.INDEX_MODE_FULL,
        "visible": {
            "index_mode": trend_contract.INDEX_MODE_VISIBLE,
            "visible_rows": len(visible),
            "total_rows": len(ordered),
            "limit": trend_contract.VISIBLE_ROWS_MAX,
        },
        "metrics": summaries,
        "metric_catalog": catalog,
        "series": {
            str(metric["name"]): _series(str(metric["name"]), visible, trend_rows)
            for metric in catalog
        },
        "runs": visible,
        "counts": {
            "run_total": len(ordered),
            "metric_row_total": len(trend_rows),
            "expected_metric_row_total": len(ordered) * len(trend_contract.METRICS),
            "invalid_runs": sum(
                1 for row in ordered if str(row.get("report_status")) == "invalid"
            ),
        },
        "inheritance": inheritance,
        "source_identity": _manifest_absence(ordered),
        "inherited_roots": sorted(str(item) for item in inherited_roots),
        "inherited_source": None if inherited_source is None else dict(inherited_source),
        "status": _index_status(len(ordered), ordered, authority_gate),
        "authority": None if authority_gate is None else dict(authority_gate),
        "degradations": degradations,
        "root_map": {str(key): str(value) for key, value in sorted(root_map.items())},
        "data_window": _data_window(ordered),
        "source_watermarks": {
            "run_total": len(ordered),
            "metric_row_total": len(trend_rows),
            "latest_window_end": _data_window(ordered)["end"],
            "previous_report_id": inheritance["previous_report_id"],
        },
    }
