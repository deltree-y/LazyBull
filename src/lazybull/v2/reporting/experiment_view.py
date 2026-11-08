# -*- coding: utf-8 -*-
"""实验报告视图模型（设计 §4.2：基线/臂、链式 NAV、Δ、分折榜、稳定性、归因、成本、制度、四卡）。

职责边界（P6 T3）：

- 只**读**实验投影，不做新指标、不补值、不重算（数值全部来自唯一实现）；
- 可见行上限只裁剪 HTML 展示，**完整值在五份 CSV**（页面显式说明）；
- 页面上的一切来源文本原样交给模板，由 ``p6text`` 统一 NFC + 转义；
- 未覆盖的 F23 图（回撤水面、制度分布）在页面上固定写“未达成/降级”，不画伪图。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import evidence, svg, text_safety
from src.lazybull.v2.reporting.view_primitives import (
    MISSING_VALUE,
    cap_note,
    fact_view,
    join_text,
    number_of,
    row_of,
    rows_of,
    section_view,
    table_view,
    text_of,
)

__all__ = [
    "EXPERIMENT_VIEW_SCHEMA",
    "MAX_ARM_ROWS",
    "MAX_ATTRIBUTION_ROWS",
    "MAX_FOLD_ROWS",
    "MAX_STABILITY_ROWS",
    "build_experiment_view",
]

#: 视图模型 schema（只用于渲染入口校验，不进入 canonical 报告）
EXPERIMENT_VIEW_SCHEMA = "p6_experiment_view/v1"
#: 可见行上限（完整值在 CSV；仅减可见行，不删数据）
MAX_FOLD_ROWS = 80
MAX_STABILITY_ROWS = 40
MAX_ATTRIBUTION_ROWS = 30
MAX_ARM_ROWS = 40


def _available() -> dict[str, Any]:
    """区段可用的最小状态行（渲染侧统一展示文案）。"""
    return {"report_status": "available", "degradation_reason_code": None}


def _metrics_row(arm: Mapping[str, Any]) -> list[str]:
    """臂指标表一行（原值格式化；缺失写固定文案）。"""
    metrics = dict(arm.get("chain", {}).get("metrics") or {})
    compare = dict(arm.get("compare", {}).get("summary") or {})
    identity = dict(arm.get("identity") or {})
    status_row = dict(arm.get("status") or {})
    return [
        text_of(identity.get("arm_id")),
        text_of(identity.get("implementation")),
        text_of(status_row.get("report_status")),
        number_of(metrics.get("cagr"), digits=6),
        number_of(metrics.get("max_drawdown"), digits=6),
        number_of(metrics.get("sharpe"), digits=4),
        number_of(compare.get("delta_cagr"), digits=6),
        number_of(compare.get("delta_max_drawdown"), digits=6),
        number_of(compare.get("delta_sharpe"), digits=4),
        text_of(arm.get("chain", {}).get("split_count")),
    ]


def _window_text(value: Any) -> str:
    """数据窗口文本（缺失写固定文案）。"""
    window = row_of(value)
    if not window:
        return MISSING_VALUE
    return f"{text_of(window.get('start'))} ~ {text_of(window.get('end'))}"


def _overview_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """基线/臂总览（显式基线、臂集合、Δ 方向约定）。"""
    arms = rows_of(projection.get("arms"))
    usable = [arm for arm in arms if str(arm.get("status", {}).get("report_status")) != "invalid"]
    facts = [
        fact_view("显式基线臂", text_of(projection.get("baseline_arm"))),
        fact_view("臂数", str(len(arms))),
        fact_view("可用臂数", str(len(usable))),
        fact_view("折子集", join_text([str(item) for item in projection.get("splits") or []])),
        fact_view("数据窗口", _window_text(projection.get("data_window"))),
        fact_view("ΔMaxDD 方向约定", "Δ = 臂 − 基线；ΔMaxDD > 0 表示回撤改善（非收益结论）"),
    ]
    columns = ["臂", "实现", "状态词", "CAGR", "MaxDD", "Sharpe", "ΔCAGR", "ΔMaxDD", "ΔSharpe", "折数"]
    rows = [_metrics_row(arm) for arm in arms[:MAX_ARM_ROWS]]
    table = table_view(
        "臂指标（链式，唯一实现 calculate_chain_metrics；Δ 由既有 compare_runs 给出）",
        columns,
        rows,
        total=len(arms),
        note=cap_note(len(arms), len(rows), "臂"),
    )
    return section_view(
        "sec-overview",
        "基线与臂",
        _available(),
        facts=facts,
        tables=[table],
        note="跨实现（runs vs 旧 WF）只做对账陈列，不产生差值结论。",
    )


def _chain_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """链式 NAV（每臂一张静态图；抽样只影响图，完整值在 CSV）。"""
    arms = rows_of(projection.get("arms"))
    columns = ["臂", "源行数", "使用行数", "折数", "交易日数", "总收益", "CAGR", "MaxDD"]
    rows: list[list[str]] = []
    charts: list[dict[str, Any]] = []
    for arm in arms:
        identity = dict(arm.get("identity") or {})
        chain = dict(arm.get("chain") or {})
        metrics = dict(chain.get("metrics") or {})
        label = text_of(identity.get("arm_id"))
        rows.append(
            [
                label,
                text_of(chain.get("rows_total")),
                text_of(chain.get("rows_used")),
                text_of(chain.get("split_count")),
                text_of(chain.get("trading_days")),
                number_of(metrics.get("total_return"), digits=6),
                number_of(metrics.get("cagr"), digits=6),
                number_of(metrics.get("max_drawdown"), digits=6),
            ]
        )
        chart = row_of(chain.get("chart"))
        series = rows_of(chart.get("series"))
        if len(series) < 2:
            continue
        charts.append(
            {
                "caption": f"{label} 链式 NAV（抽样仅影响图）",
                "svg": svg.nav_chart(series, caption=f"{label} 链式 NAV"),
                "note": (
                    f"可见点 {len(series)} 个"
                    + ("（已确定性抽样）" if bool(chart.get("sampled")) else "")
                    + "；完整值与行序见 chain_nav.csv"
                ),
            }
        )
    table = table_view(
        "链式净值行数守恒（合法输入须 使用行数 = 源行数，交易日数 = N − S）",
        columns,
        rows[:MAX_ARM_ROWS],
        total=len(rows),
    )
    return section_view(
        "sec-chain",
        "链式 NAV",
        _available(),
        tables=[table],
        charts=charts,
        note="链式指标只在全部行可解析、有限且 NAV>0 时计算；坏行使该臂 invalid。",
    )


def _fold_section(projection: Mapping[str, Any], csv_rows: Mapping[str, Any]) -> dict[str, Any]:
    """分折榜（可见行裁剪；完整值在 fold_metrics.csv）。"""
    rows_source = rows_of(csv_rows.get("fold_metrics.csv"))
    columns = [
        "臂",
        "折",
        "测试起点",
        "测试终点",
        "模型版本",
        "折总收益",
        "折回撤",
        "折收益",
        "TopK20",
        "TopK30",
    ]
    rows = [
        [
            text_of(item.get("arm_id")),
            text_of(item.get("split_index")),
            text_of(item.get("test_start")),
            text_of(item.get("test_end")),
            text_of(item.get("model_version")),
            number_of(item.get("bt_total_return"), digits=6),
            number_of(item.get("bt_max_drawdown"), digits=6),
            number_of(item.get("fold_return"), digits=6),
            number_of(item.get("topk20_positive_day_ratio"), digits=4),
            number_of(item.get("topk30_positive_day_ratio"), digits=4),
        ]
        for item in rows_source[:MAX_FOLD_ROWS]
    ]
    table = table_view(
        "分折榜（源 summary 原值 + 既有 fold_table 折收益/回撤）",
        columns,
        rows,
        total=len(rows_source),
        note=cap_note(len(rows_source), len(rows), "逐折行") + "；完整值见 fold_metrics.csv",
    )
    return section_view(
        "sec-folds",
        "分折榜",
        (_available() if rows_source else
         {"report_status": "not_available", "degradation_reason_code": "no_source"}),
        tables=[table],
        note="TopK 正日占比只对旧 WF 的 KEY_* 有合法来源；规范化 runs 固定 not_available。",
    )


def _stability_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """跨期稳定性（完整等价类 + 显式状态行）。"""
    block = row_of(projection.get("stability"))
    rows_source = rows_of(block.get("rows"))
    columns = [
        "等价类键",
        "期数",
        "时间段",
        "运行ID",
        "得分",
        "CAGR均值",
        "CAGR标准差",
        "回撤均值",
        "回撤最差",
        "Sharpe均值",
        "状态",
        "原因码",
    ]
    rows = [
        [
            text_of(item.get("group_key"))[:16],
            text_of(item.get("period_count")),
            text_of(item.get("period_labels")),
            text_of(item.get("run_id_list")),
            number_of(item.get("stability_score"), digits=1),
            number_of(item.get("chain_cagr_mean"), digits=6),
            number_of(item.get("chain_cagr_std"), digits=6),
            number_of(item.get("chain_max_drawdown_mean"), digits=6),
            number_of(item.get("chain_max_drawdown_worst"), digits=6),
            number_of(item.get("chain_sharpe_mean"), digits=4),
            text_of(item.get("report_status")),
            text_of(item.get("degradation_reason_code")),
        ]
        for item in rows_source[:MAX_STABILITY_ROWS]
    ]
    table = table_view(
        "稳定性（DYNAMIC_EXACT 单一分类 + stable_identity 外层键；无前置/不足两期为状态行）",
        columns,
        rows,
        total=len(rows_source),
        note=cap_note(len(rows_source), len(rows), "稳定性行") + "；完整值见 stability.csv",
    )
    return section_view(
        "sec-stability",
        "跨期稳定性",
        block.get("status") or _available(),
        facts=[
            fact_view("稳定等价类数", text_of(block.get("partitions"))),
            fact_view(
                "被投影的动态分组列",
                join_text([str(item) for item in block.get("dynamic_columns") or []]),
            ),
        ],
        tables=[table],
        note="batch_run_id 不得二次拆组；原始 run/折/模型版本在 CSV 与 source_ref 中保留。",
    )


def _attribution_section(
    projection: Mapping[str, Any], csv_rows: Mapping[str, Any]
) -> dict[str, Any]:
    """执行归因（19 列原值；页面只给计数与预览）。"""
    arms = rows_of(projection.get("arms"))
    rows_source = rows_of(csv_rows.get("execution_attribution.csv"))
    facts = [
        fact_view(
            f"{text_of(dict(arm.get('identity') or {}).get('arm_id'))} 归因行数",
            text_of(dict(arm.get("attribution") or {}).get("row_count")),
        )
        for arm in arms[:MAX_ARM_ROWS]
    ]
    columns = ["臂", "折", "源行号", "计划代码", "实际代码", "阶段", "状态", "原因", "信号到买入"]
    rows = [
        [
            text_of(item.get("arm_id")),
            text_of(item.get("split_index")),
            text_of(item.get("source_row_index")),
            text_of(item.get("planned_ts_code")),
            text_of(item.get("actual_ts_code")),
            text_of(item.get("execution_stage")),
            text_of(item.get("status")),
            text_of(item.get("reason")),
            number_of(item.get("signal_to_buy_return"), digits=8),
        ]
        for item in rows_source[:MAX_ATTRIBUTION_ROWS]
    ]
    table = table_view(
        "执行归因预览（契约 19 列全值 + 原 status/reason 在 execution_attribution.csv）",
        columns,
        rows,
        total=len(rows_source),
        note=cap_note(len(rows_source), len(rows), "归因行"),
    )
    return section_view(
        "sec-attribution",
        "执行归因",
        (_available() if rows_source else
         {"report_status": "not_available", "degradation_reason_code": "fold_file_missing"}),
        facts=facts,
        tables=[table],
        note="非成功执行状态原值保留（不映射成报告状态词）。",
    )


def _cost_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """成本/双边成交金额（不是换手率，也不是 NAV 贡献）。"""
    arms = rows_of(projection.get("arms"))
    columns = ["臂", "双边成交金额", "成本金额", "状态", "原因码"]
    rows = []
    for arm in arms:
        identity = dict(arm.get("identity") or {})
        cost = dict(arm.get("cost") or {})
        cost_status = dict(cost.get("status") or {})
        rows.append(
            [
                text_of(identity.get("arm_id")),
                number_of(cost.get("turnover_total"), digits=2),
                number_of(cost.get("cost_total"), digits=2),
                text_of(cost_status.get("report_status")),
                text_of(cost_status.get("degradation_reason_code")),
            ]
        )
    table = table_view(
        "成本与双边成交金额（fold+date 聚合，折界重复日不归并）",
        columns,
        rows,
        total=len(rows),
    )
    return section_view(
        "sec-cost",
        "成本与双边成交金额",
        (_available() if rows else
         {"report_status": "not_available", "degradation_reason_code": "no_source"}),
        tables=[table],
        note="金额差不是 NAV 贡献；当前没有成本率/换手率唯一实现，不补造。",
    )


def _topk_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """TopK 正日占比（仅旧 WF 有来源）。"""
    arms = rows_of(projection.get("arms"))
    columns = ["臂", "实现", "状态", "原因码", "Top20 均值", "Top30 均值"]
    rows = []
    for arm in arms:
        identity = dict(arm.get("identity") or {})
        topk = dict(arm.get("topk") or {})
        status_row = dict(topk.get("status") or {})
        means = dict(topk.get("means") or {})
        rows.append(
            [
                text_of(identity.get("arm_id")),
                text_of(identity.get("implementation")),
                text_of(status_row.get("report_status")),
                text_of(status_row.get("degradation_reason_code")),
                number_of(means.get("topk20_positive_day_ratio_mean"), digits=4),
                number_of(means.get("topk30_positive_day_ratio_mean"), digits=4),
            ]
        )
    table = table_view(
        "TopK 正日占比（旧 WF 仅字段改名；跨折均值取自 aggregate_run）",
        columns,
        rows,
        total=len(rows),
    )
    return section_view(
        "sec-topk",
        "TopK 正日占比",
        _available(),
        tables=[table],
        note="规范化 runs 固定 no_unique_implementation：禁止从 topk_detail 私算。",
    )


def _regime_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """制度重排摘要（有 bins 才陈列计数；无 bins 不画分布）。"""
    regime = row_of(projection.get("regime"))
    results = rows_of(regime.get("results"))
    columns = sorted({key for item in results for key in item})
    rows = [
        [text_of(item.get(column)) for column in columns] for item in results[:MAX_STABILITY_ROWS]
    ]
    bins = regime.get("bins")
    tables = [
        table_view(
            "制度重排摘要（显式 P5a JSON 原值）",
            columns or ["说明"],
            rows,
            total=len(results),
            note="" if results else "无摘要行：来源未提供",
        )
    ]
    if isinstance(bins, Mapping) and bins:
        bin_rows: list[list[str]] = []
        for scene, counts in sorted(bins.items()):
            for key, value in sorted(dict(counts).items()):
                bin_rows.append([str(scene), str(key), text_of(value)])
        tables.append(
            table_view(
                "制度分箱计数（只读既有产物，不重采样）",
                ["场景", "分箱", "计数"],
                bin_rows[:MAX_STABILITY_ROWS],
                total=len(bin_rows),
            )
        )
    return section_view(
        "sec-regime",
        "制度重排摘要",
        regime.get("status") or {"report_status": "not_available",
                                 "degradation_reason_code": "no_source"},
        facts=[
            fact_view("point_delta", text_of(regime.get("point_delta"))),
            fact_view("可得 bins", "有" if bins else "无（不画伪分布）"),
        ],
        tables=tables,
        note="无 bins 时固定 not_available/no_regime_bins：不画分布、不重采样。",
    )


def _cards_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """四段证据卡（信号/执行/β暴露/成本 + 稳定性）。"""
    cards = rows_of(projection.get("cards"))
    columns = ["证据卡", "状态", "原因码", "事实", "说明"]
    rows = []
    for card in cards:
        facts = "；".join(
            f"{text_of(item.get('label'))}={text_of(item.get('value'))}"
            for item in rows_of(card.get("facts"))
        )
        status_row = dict(card.get("status") or {})
        rows.append(
            [
                text_of(card.get("card")),
                text_of(status_row.get("report_status")),
                text_of(status_row.get("degradation_reason_code")),
                facts,
                text_of(card.get("note")),
            ]
        )
    table = table_view("四段证据卡（独立列证据，不可相加抵账）", columns, rows, total=len(rows))
    return section_view(
        "sec-cards",
        "证据卡",
        _available(),
        tables=[table],
        note="不是守恒瀑布：成本金额差不等于 NAV 贡献，β 暴露无来源即降级。",
    )


def _degradation_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """降级清单（manifest degradations 的页面视图）。"""
    entries = rows_of(projection.get("degradations"))
    columns = ["字段路径", "状态", "原因码", "来源", "明细"]
    rows = [
        [
            text_of(item.get("field_path")),
            text_of(item.get("report_status")),
            text_of(item.get("reason_code")),
            text_of(item.get("source_ref")),
            text_of(item.get("detail")),
        ]
        for item in entries
    ]
    table = table_view("降级与不可用项（按字段路径稳定排序）", columns, rows, total=len(rows))
    return section_view(
        "sec-degradations",
        "降级清单",
        (_available() if entries else
         {"report_status": "empty_valid", "degradation_reason_code": None}),
        tables=[table],
        note="空表示没有登记降级项（已核实为空）。",
    )


def _notes_section(projection: Mapping[str, Any]) -> dict[str, Any]:
    """口径与未达成项（固定可见，不隐藏边界）。"""
    notes = [str(item) for item in projection.get("notes") or []]
    facts = [fact_view(f"口径 {index + 1}", note) for index, note in enumerate(notes)]
    return section_view(
        "sec-notes",
        "口径与未达成项",
        _available(),
        facts=facts,
        note="F23 图未覆盖项在此明示；不得把降级项当作已达成。",
    )


def build_experiment_view(
    projection: Mapping[str, Any],
    *,
    report_id: str,
    generated_at: str,
    input_evidence_classes: Sequence[str],
    generator: Mapping[str, Any] | None = None,
    csv_rows: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """装配实验报告视图（十一区 + 固定 companion 链接 + 生成器页脚）。"""
    material = row_of(projection)
    evidence_class = str(material.get("evidence_class") or "")
    banner = evidence.banner_for(evidence_class, None)
    rows = dict(csv_rows or {})
    sections = [
        _overview_section(material),
        _chain_section(material),
        _fold_section(material, rows),
        _stability_section(material),
        _attribution_section(material, rows),
        _cost_section(material),
        _topk_section(material),
        _regime_section(material),
        _cards_section(material),
        _degradation_section(material),
        _notes_section(material),
    ]
    return {
        "schema": EXPERIMENT_VIEW_SCHEMA,
        "lang": "zh-CN",
        "title": f"LazyBull 实验报告（{evidence_class}）",
        "report": {
            "report_id": report_id,
            "report_kind": "experiment",
            "generated_at": generated_at,
            "evidence_class": evidence_class,
            "input_evidence_classes": "、".join(
                sorted(str(item) for item in input_evidence_classes)
            ),
            "projection_schema": text_of(material.get("schema")),
            "projection_generated_at": text_of(material.get("generated_at")),
            "baseline_arm": text_of(material.get("baseline_arm")),
        },
        "banner": banner,
        "sections": sections,
        "three_ends": {"visible": False, "title": "", "rows": [], "note": ""},
        "companions": text_safety.companion_links(text_safety.EXPERIMENT_COMPANION_FILES),
        "footer": {
            "generator": "、".join(
                f"{key}={value}" for key, value in sorted(dict(generator or {}).items())
            )
            or "—",
            "note": (
                "本页为只读静态投影：不重训、不重跑 WF、不产生采纳或闸门结论；"
                "研究读数不代表未来收益。"
            ),
        },
    }
