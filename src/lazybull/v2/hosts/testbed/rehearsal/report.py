# -*- coding: utf-8 -*-
"""P5b T6 演练报告（§7.2 字段全覆盖 + markdown），不含任何闸门结论。

契约来源：设计 v4 §7.2（报告字段逐项）、§7.1（产物根）、§6.3（验收口径）、§5.6（限制）、
任务书「判定表必须固定 claim 字段并逐条列出作者侧检查」。

字段纪律
========
- ``h6.items`` 三条各自给出「目标值 / 实际值 / 判定 / 证据字段路径」；
- 频率类字段一律标注「引用 T4/T5」，本演练不自产；
- ``scope=historical_testbed``、``virtual_clock=true``、``not_forward_evidence=true``、
  ``not_sla_evidence=true`` 必须逐字出现在报告顶层。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.lazybull.v2.hosts.testbed.paths import EVIDENCE_CLASS, SCOPE

__all__ = ["REPORT_SCHEMA", "build_report", "render_markdown"]

REPORT_SCHEMA = "p5b_testbed_rehearsal_report/v1"


def build_report(
    *,
    campaign_id: str,
    run_id: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    calibrator: Mapping[str, Any],
    upstream: Mapping[str, Any],
    run_report: Mapping[str, Any],
    aa: Mapping[str, Any],
    h6_items: Sequence[Mapping[str, Any]],
    income_reading: Mapping[str, Any],
    drawdown_readings: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    effects: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    machine_time: Mapping[str, Any],
    authorization: Optional[Mapping[str, Any]],
    production: Mapping[str, Any],
    referenced: Mapping[str, Any],
    t4_reference: Mapping[str, Any],
    t5_reference: Mapping[str, Any],
) -> Dict[str, Any]:
    """组装 §7.2 报告（字段命名与 T1~T5 报告保持可对齐）。"""
    window = dict(manifest["window"])
    return {
        "schema": REPORT_SCHEMA,
        "scope": SCOPE,
        "evidence_class": EVIDENCE_CLASS,
        "virtual_clock": True,
        "not_forward_evidence": True,
        "not_sla_evidence": True,
        "not_real_candidate": True,
        "campaign_id": str(campaign_id),
        "run_id": str(run_id),
        "schedule_mode": "replay",
        "row_domain": dict(run_report.get("row_domain") or {}),
        "history": {
            "start": window.get("d0_date"),
            "end": (window.get("virtual_days") or [None])[-1],
            "d0_date": window.get("d0_date"),
            "first_return_date": window.get("first_return_date"),
            "virtual_days": int(window.get("day_count", 0)),
            "return_days": int(window.get("return_days", 0)),
            "selection_rule": window.get("selection_rule"),
            "deletion_forbidden": bool(window.get("deletion_forbidden")),
        },
        "label_cutoff": {
            "label_filtered_through": window.get("label_filtered_through"),
            "status": "not_consumed_by_mechanical_layer",
            "note": "机械层按 row_domain=structural 过滤，不消费语料标签；该值来自冻结窗口声明",
            **dict(run_report.get("label_cutoff") or {}),
        },
        "model_availability": dict(run_report.get("model_availability") or {}),
        "identity": {
            "campaign": dict(run_report.get("identity") or {}),
            "calibrator": {
                "implementation_id": calibrator["implementation_id"],
                "sha256": calibrator["sha256"],
                "covered_modules": list(calibrator["modules"]),
            },
            "manifest_sha256": str(manifest_sha256),
            "upstream": dict(upstream),
        },
        "arms": dict(run_report.get("arms") or {}),
        "mechanical_aa": dict(aa),
        "transactions": list(run_report.get("transactions") or []),
        "published": dict(run_report.get("published") or {}),
        "recovery": dict(run_report.get("recovery") or {}),
        "fault_injections": list(run_report.get("fault_injections") or []),
        "effects": {arm: dict(payload) for arm, payload in sorted(effects.items())},
        "income": {
            "reading": dict(income_reading),
            "thresholds_reference": dict(t4_reference),
        },
        "drawdown": {
            "readings": dict(drawdown_readings),
            "thresholds_reference": dict(t5_reference),
            "frozen_windows": list(manifest["injection_window"]["frozen_list"]),
            "c_w_by_look": list(thresholds["c_w"]),
            "c_i_by_look": list(thresholds["c_i"]),
            "anchor": int(thresholds["anchor"]),
            "directions_judged_separately": True,
        },
        "h6": {
            "claim": "evaluated_not_declared_pass",
            "items": [dict(item) for item in h6_items],
            "frequency_source": "T4/T5 holdout 产物（本演练不产生频率/MDE）",
            "mutual_offset_forbidden": True,
            "executed_effects": sorted(
                {
                    str((effect or {}).get("kind"))
                    for effect in effects.values()
                    if bool((effect or {}).get("active"))
                }
            ),
            "not_executed_effects": sorted(
                {
                    str((effect or {}).get("kind"))
                    for effect in effects.values()
                    if str((effect or {}).get("status")) == "not_constructible"
                }
            ),
            "anchor_solver": dict(manifest.get("anchor_solver") or {}),
        },
        "referenced_readings": dict(referenced),
        "artifact_hashes": {str(key): str(value) for key, value in sorted(artifact_hashes.items())},
        "machine_time": dict(machine_time),
        "long_task_authorization": None if authorization is None else dict(authorization),
        "production_zero_write": dict(production),
        "limitations": list(manifest["limitations"]),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """报告可读版（产物哈希与判定边界逐条带出）。"""
    lines: List[str] = [
        "# P5b T6 历史端到端演练报告",
        "",
        f"> scope=`{report['scope']}` / evidence_class=`{report['evidence_class']}` / "
        "virtual_clock=true / not_forward_evidence=true / not_sla_evidence=true",
        f"> campaign=`{report['campaign_id']}` run=`{report['run_id']}`",
        "> **闸门判定归用户**：本报告只给作者侧检查结果。",
        "",
        "## 1. 窗口与链路",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 历史起止 | {_fmt(report['history']['start'])} ~ {_fmt(report['history']['end'])} |",
        f"| 虚拟日 / 收益样本 | {report['history']['virtual_days']} / "
        f"{report['history']['return_days']} |",
        f"| 行域回执 | structural verified={_fmt(report['row_domain'].get('verified'))} |",
        f"| 已发布代 | {_fmt(report['published'].get('published_generation'))} |",
        f"| 机械 A/A | identical={_fmt(report['mechanical_aa'].get('identical'))} "
        f"天={report['mechanical_aa'].get('compared_days')} "
        f"字段={report['mechanical_aa'].get('compared_fields')} |",
        "",
        "## 2. 五臂效应",
        "",
        "| 臂 | 种类 | 状态 | 参数 | 目标 |",
        "|---|---|---|---:|---:|",
    ]
    for arm, payload in sorted(dict(report["effects"]).items()):
        lines.append(
            f"| `{arm}` | {payload.get('kind')} | {payload.get('status')} | "
            f"{_fmt(payload.get('parameter'))} | {_fmt(payload.get('target_pp'))} |"
        )
    lines.extend(
        [
            "",
            "## 3. H6 三条单条数值",
            "",
            "| 条目 | 状态 | 目标 | 实际 | 判定 |",
            "|---|---|---|---:|---|",
        ]
    )
    for item in report["h6"]["items"]:
        actual = item.get("d_anchor_pp")
        if item.get("name") == "income_minus_5d":
            actual = item.get("paired_mean")
        lines.append(
            f"| `{item['name']}` | {item['status']} | {_fmt(item.get('target_pp'))} | "
            f"{_fmt(actual, 8)} | {item.get('judgment')} |"
        )
    lines.extend(
        [
            "",
            "> 三条分别判定：任一失败或未执行不得用另一条抵扣；"
            "频率/MDE 只引用 T4/T5 holdout 读数。",
            "",
            "## 4. 读数",
            "",
            f"- 收益臂配对样本 {report['income']['reading'].get('sample_days')} 日；"
            f"可执行 look={report['income']['reading'].get('executable_looks')}；"
            f"可识别 look={report['income']['reading'].get('identifiable_looks')}",
            f"- 回撤锚 T_anchor={report['drawdown'].get('anchor')}；"
            f"c^W={report['drawdown'].get('c_w_by_look')}；"
            f"c^I={report['drawdown'].get('c_i_by_look')}",
            f"- 引用（不重算）：{report['referenced_readings'].get('note')}",
            "",
            "## 5. 产物与墙钟",
            "",
            "| 产物 | sha256 |",
            "|---|---|",
        ]
    )
    for name, digest in sorted(dict(report["artifact_hashes"]).items()):
        lines.append(f"| `{name}` | `{digest}` |")
    machine = dict(report["machine_time"])
    batches = list(machine.get("batches") or [])
    batch_text = ",".join(str(int(batch.get("days", 0))) for batch in batches) or "—"
    production = dict(report["production_zero_write"] or {})
    dirs = sorted(
        {name for section in ("before", "after") for name in dict(production.get(section) or {})}
    )
    production_lines = [
        f"- 生产零写入：unchanged={production.get('unchanged')} "
        f"window={production.get('window')} method={production.get('method')}"
    ]
    for name in dirs:
        before = dict((production.get("before") or {}).get(name) or {})
        after = dict((production.get("after") or {}).get(name) or {})
        same_digest = before.get("path_digest") == after.get("path_digest")
        production_lines.append(
            f"  - `{name}`: before files={before.get('files')} bytes={before.get('bytes')} "
            f"/ after files={after.get('files')} bytes={after.get('bytes')}"
            f" / path_digest {'相同' if same_digest else '不同'}"
        )
    lines.extend(
        [
            "",
            f"- 机器墙钟：virtual_days_total={machine.get('virtual_days_total')} "
            f"batches=[{batch_text}]（逐日数值见 `machine_time_run.json`，不入产物身份）",
            f"- 长任务授权：{report['long_task_authorization']}",
            *production_lines,
            "",
            "## 6. 限制声明",
            "",
        ]
    )
    for index, text in enumerate(report["limitations"], start=1):
        lines.append(f"{index}. {text}")
    lines.append("")
    return "\n".join(lines)
