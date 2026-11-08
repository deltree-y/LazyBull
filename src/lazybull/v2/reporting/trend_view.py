# -*- coding: utf-8 -*-
"""趋势页视图模型（P6 T4：只读投影 → 区段/图表/表格，防御读取）。

硬口径：

- 页面为 ``index_mode=visible``（最近 ``VISIBLE_ROWS_MAX`` 行）；完整索引与完整趋势值只
  在 CSV，页面**不从 CSV 抽样**、也不暗示页面是全量；
- 五类指标按证据类分列（单位/日期语义不同轴），不提供跨指标/跨类合计；
- 缺值显式写状态与原因码（不写 0、不留空白冒充）；
- 本模块不做 HTML 转义（模板过滤器 ``p6text`` 统一负责），也不做指标计算。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import (
    evidence,
    status,
    text_safety,
    trend_contract,
    trend_svg,
    view_primitives,
)

__all__ = ["VIEW_SCHEMA", "build_trend_view"]

#: 视图模型 schema（不进入 canonical 报告，仅渲染入口校验）
VIEW_SCHEMA = "p6_trend_view/v1"
#: 页面内每指标“最近点”表行数（完整值在 CSV）
_RECENT_POINTS = 10
#: 页面内降级登记行上限
_DEGRADATION_ROWS = 100
#: 页面内 run 索引表列集（完整列集在 run_index.csv）
_RUN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "window_end",
    "implementation",
    "host_mode",
    "data_state_id",
    "report_status",
    "degradation_reason_code",
    "source_ref",
)


def _metric_status(summary: Mapping[str, Any]) -> dict[str, Any]:
    """指标区状态（按该指标全量行的可用性判定，不按页面可见子集）。"""
    if not bool(summary.get("generated")):
        return status.make_status(
            "not_available",
            reason_code="no_source",
            detail="该类证据域不生成该指标（设计 §5.5；不跨类复用单位）",
        )
    total = int(summary.get("row_count") or 0)
    usable = int(summary.get("usable_rows") or 0)
    with_value = int(summary.get("with_value_rows") or usable)
    if total == 0:
        return status.make_status("empty_valid", detail="该指标无任何 run 行")
    if with_value == total and usable == total:
        return status.make_status("available", detail=f"{total} 行全部可用")
    reasons = dict(summary.get("reason_codes") or {})
    reason = next(iter(sorted(reasons)), None)
    if with_value == 0:
        return status.make_status(
            "not_available",
            reason_code=reason or "missing_precondition",
            detail=f"{total} 行全部为状态行（明细见各 run 的 trend 行与 CSV）",
        )
    # 有值但区级降级：不得宣称“全部无值/全部状态行”（R3-N1-R3）
    return status.make_status(
        "degraded",
        reason_code=reason or "missing_precondition",
        detail=f"{total} 行中 {with_value} 行有值、{total - with_value} 行为状态行"
        "（不插值、不补 0）",
    )


def _index_facts(projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """索引区事实（页面可见性与完整 CSV 的差异必须显式）。"""
    visible = dict(projection.get("visible") or {})
    counts = dict(projection.get("counts") or {})
    inheritance = dict(projection.get("inheritance") or {})
    identity = dict(projection.get("source_identity") or {})
    window = dict(projection.get("data_window") or {})
    facts = [
        view_primitives.fact_view("CSV 索引模式", str(projection.get("index_mode") or "")),
        view_primitives.fact_view("页面索引模式", str(visible.get("index_mode") or "")),
        view_primitives.fact_view(
            "run 总数 / 页面可见行",
            f"{counts.get('run_total', 0)} / {visible.get('visible_rows', 0)}"
            f"（上限 {visible.get('limit')}）",
        ),
        view_primitives.fact_view(
            "指标行数 / 期望",
            f"{counts.get('metric_row_total', 0)} / {counts.get('expected_metric_row_total', 0)}"
            f"（= run 数 × {len(trend_contract.METRICS)}）",
        ),
        view_primitives.fact_view("无效 run 行数", str(counts.get("invalid_runs", 0))),
        view_primitives.fact_view(
            "数据窗口", f"{window.get('start') or '—'} ~ {window.get('end') or '—'}"
        ),
        view_primitives.fact_view(
            "生成模式",
            f"{inheritance.get('mode')}（继承 {inheritance.get('inherited_runs', 0)} / "
            f"新增 {inheritance.get('new_runs', 0)}）",
        ),
        view_primitives.fact_view(
            "上一快照 report_id", str(inheritance.get("previous_report_id") or "—")
        ),
        view_primitives.fact_view(
            "无 manifest 的 run",
            f"{identity.get('count', 0)} 行（标 {identity.get('flag')}；身份由消费文件 canonical 哈希给出）",
        ),
    ]
    return facts


def _run_rows(projection: Mapping[str, Any]) -> list[list[str]]:
    """可见 run 索引表（列集固定；值原样展示）。"""
    rows = view_primitives.rows_of(projection.get("runs"))
    material: list[list[str]] = []
    for row in rows:
        material.append([view_primitives.text_of(row.get(column)) for column in _RUN_COLUMNS])
    return material


def _run_table(projection: Mapping[str, Any]) -> dict[str, Any]:
    """可见索引表（``total`` 为**可见行数**；完整索引行数在事实与 CSV）。"""
    rows = _run_rows(projection)
    total = int((projection.get("visible") or {}).get("total_rows") or 0)
    note = (
        f"run 总数 {total} 行；页面只显示最近 {len(rows)} 行（完整索引见 run_index.csv，"
        "永不抽样）"
        if total > len(rows)
        else f"run 总数 {total} 行；页面显示全部可见行（完整索引见 run_index.csv）"
    )
    return view_primitives.table_view(
        "最近 run 索引（visible）",
        list(_RUN_COLUMNS),
        rows,
        total=len(rows),
        note=note,
        row=projection.get("status"),
    )


def _metric_section(
    summary: Mapping[str, Any], projection: Mapping[str, Any]
) -> dict[str, Any]:
    """单指标区段：事实 + 图 + 最近点表（完整值在 trend.csv）。"""
    name = str(summary.get("name"))
    points = view_primitives.rows_of((projection.get("series") or {}).get(name))
    row = _metric_status(summary)
    unit_label = str(summary.get("unit_label") or "不适用")
    facts = [
        view_primitives.fact_view("单位", unit_label),
        view_primitives.fact_view("该指标行数", str(summary.get("row_count", 0))),
        view_primitives.fact_view(
            "有值行（含降级） / 可用行 / 状态行",
            f"{summary.get('with_value_rows', 0)} / {summary.get('usable_rows', 0)}"
            f" / {summary.get('status_rows', 0)}",
        ),
        view_primitives.fact_view(
            "状态分布",
            "、".join(
                f"{key}={value}"
                for key, value in dict(summary.get("by_status") or {}).items()
            )
            or "无行",
        ),
    ]
    charts = [
        {
            "caption": f"{summary.get('label')}（{unit_label}）",
            "svg": trend_svg.metric_chart(
                points, caption=str(summary.get("label") or name), unit_label=unit_label
            ),
            "note": (
                f"图最多 min({trend_contract.MAX_CHART_POINTS}, 可见行) 点"
                "（首尾与极值必留）；缺值断线、不插值；完整值见 trend.csv"
            ),
        }
    ]
    table_rows = [
        [
            view_primitives.text_of(point.get("run_id")),
            view_primitives.text_of(point.get("window_end")),
            (
                view_primitives.number_of(point.get("value"), digits=6)
                if point.get("value") is not None
                else view_primitives.text_of("不可用（见状态列）")
            ),
            view_primitives.text_of(point.get("report_status")),
            view_primitives.text_of(point.get("degradation_reason_code")),
        ]
        for point in points[-_RECENT_POINTS:]
    ]
    tables = [
        view_primitives.table_view(
            f"最近 {len(table_rows)} 个可见点（完整值见 trend.csv）",
            ["run_id", "window_end", "value", "report_status", "degradation_reason_code"],
            table_rows,
            total=len(points),
            note="页面为最近可见行；CSV 为全量且不抽样",
            row=row,
        )
    ]
    return view_primitives.section_view(
        f"metric-{name}",
        f"{summary.get('label')}（{name}）",
        row,
        facts=facts,
        tables=tables,
        charts=charts,
        note=str(summary.get("note") or ""),
    )


def _degradation_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """降级登记（按 field_path 稳定排序；只显示前 N 行）。"""
    rows = view_primitives.rows_of(projection.get("degradations"))
    rows = sorted(rows, key=lambda row: str(row.get("field_path") or ""))
    material = [
        [
            view_primitives.text_of(row.get("field_path")),
            view_primitives.text_of(row.get("report_status")),
            view_primitives.text_of(row.get("reason_code")),
            view_primitives.text_of(row.get("source_ref")),
            view_primitives.text_of(row.get("detail")),
        ]
        for row in rows[:_DEGRADATION_ROWS]
    ]
    count = len(rows)
    row = status.make_status(
        "degraded" if count else "empty_valid",
        reason_code="missing_precondition" if count else None,
        detail=f"本次新增 run 的降级登记 {count} 条（继承行见上一快照 manifest）",
    )
    return view_primitives.section_view(
        "trend-degradations",
        "降级清单（本次新增 run）",
        row,
        tables=[
            view_primitives.table_view(
                "降级登记",
                ["字段路径", "状态", "原因码", "来源", "说明"],
                material,
                total=count,
                note=view_primitives.cap_note(count, len(material), "降级登记"),
                row=row,
            )
        ],
        note="继承行不重复登记；其历史降级以上一快照 manifest 为准",
    )


def build_trend_view(
    projection: Mapping[str, Any],
    *,
    report_id: str,
    generated_at: str,
    input_evidence_classes: Sequence[str],
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    """装配趋势页视图（横幅 + 身份 + 索引区 + 每指标区 + 降级区 + companion）。"""
    material = dict(projection)
    evidence_class = str(material.get("evidence_class") or "")
    banner = evidence.banner_for(evidence_class, material.get("status"))
    metrics = view_primitives.rows_of(material.get("metrics"))
    sections = [
        view_primitives.section_view(
            "trend-index",
            "run 索引与可见性",
            material.get("status"),
            facts=_index_facts(material),
            tables=[_run_table(material)],
            note="完整索引永为 full（run_index.csv）；页面只显示最近可见行，不抽样",
        )
    ]
    sections.extend(_metric_section(summary, material) for summary in metrics)
    sections.append(_degradation_section(material))
    return {
        "schema": VIEW_SCHEMA,
        "title": f"P6 趋势库（{evidence_class}）",
        "banner": banner,
        "report": {
            "report_id": str(report_id),
            "report_kind": "trend",
            "evidence_class": evidence_class,
            "input_evidence_classes": "、".join(str(item) for item in input_evidence_classes),
            "generated_at": str(generated_at),
            "projection_schema": str(material.get("schema") or ""),
            "projection_generated_at": str(material.get("generated_at") or ""),
            "index_mode": str((material.get("visible") or {}).get("index_mode") or ""),
        },
        "sections": sections,
        "footer": {
            "note": (
                "离线只读趋势库：不提供 latest、不做跨证据类总览、不插值、不补 0；"
                "图不参与任何指标计算。"
            ),
            "generator": (
                f"lazybull {generator.get('project_version')} / python {generator.get('python')} / "
                f"jinja2 {generator.get('jinja2')} / git {generator.get('git_commit')}"
                f"（dirty={generator.get('git_dirty')}）"
            ),
        },
        "companions": text_safety.companion_links(text_safety.TREND_COMPANION_FILES),
    }
