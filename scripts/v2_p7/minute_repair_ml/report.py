"""资源决定映射与中文结果报告（协议 §11.5/§17）。

状态语汇固定：`access_not_ready / data_not_ready / engineering_invalid /
economic_target_not_met / continue_limited / economic_candidate`；
`formal_admission` 恒为 `not_evaluated`（研究完成不等于 P7 出口通过）。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.v2_p7.minute_repair_ml import analysis, spec


def decide(
    *,
    engineering_status: str,
    data_status: str,
    targets: Mapping[str, Any] | None,
    pressure: Mapping[str, Any] | None,
    recent_flag: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """把工程/数据/经济读数映射成资源建议（不混同正式准入）。"""
    if data_status in ("access_not_ready", "data_not_ready"):
        return {
            "research_decision": data_status,
            "resource_action": "pause_data_dependency_deliver_evidence",
            "reason": "权限、广度、时间含义或覆盖不足，不判经济失败",
        }
    if engineering_status == "invalid":
        return {
            "research_decision": "engineering_invalid",
            "resource_action": "fix_mechanics_and_rerun_within_budget",
            "reason": "账务/PIT/标签/复现失效：invalid",
        }
    if engineering_status == "partial":
        # 局部合成测试通过、数据依赖项未验：只能按数据未就绪处理，不得写成工程通过
        return {
            "research_decision": "data_not_ready",
            "resource_action": "pause_data_dependency_deliver_evidence",
            "reason": "仅有局部合成测试通过；数据访问/语义/吞吐/后续链路未验证",
        }
    if targets is None:
        return {
            "research_decision": "data_not_ready",
            "resource_action": "pause_data_dependency_deliver_evidence",
            "reason": "缺主表读数，无法判定经济目标",
        }
    if not targets.get("all_met", False):
        return {
            "research_decision": "economic_target_not_met",
            "resource_action": "stop_new_investment",
            "reason": "工程有效但主表四项目标未全部满足",
        }
    limited: list[str] = []
    if pressure is not None and not pressure.get("all_met", True):
        limited.append("固定压力未全部满足")
    if recent_flag is not None and recent_flag.get("triggered", False):
        limited.append("最后固定子段对现金为负且对 A100 回撤更差")
    if limited:
        return {
            "research_decision": "continue_limited",
            "resource_action": "keep_candidate_pause_extension",
            "reason": "；".join(limited),
        }
    return {
        "research_decision": "economic_candidate",
        "resource_action": "propose_formal_g_package",
        "reason": "主表与固定压力均满足，仍未进入正式 P7 准入",
    }


def summary_payload(
    *,
    identity: Mapping[str, Any],
    engineering_status: str,
    data_status: str,
    targets_by_capital: Mapping[str, Any],
    pressure_status: Mapping[str, Any],
    decision: Mapping[str, Any],
    limitations: Sequence[str],
    unmet_items: Sequence[str],
    machine_hours: Mapping[str, Any],
    requests: int,
    bytes_written: int,
    new_data_cost: int = 0,
    verification_scope: Mapping[str, Any] | None = None,
    budget: Mapping[str, Any] | None = None,
    storage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """`summary.json`（字段缺失不得默认 false/pass）。"""
    return {
        **dict(identity),
        "engineering_status": engineering_status,
        "data_status": data_status,
        "verification_scope": dict(verification_scope or {}),
        "budget": dict(budget or {}),
        "storage": dict(storage or {}),
        "economic_targets_by_capital": dict(targets_by_capital),
        "pressure_status": dict(pressure_status),
        "research_decision": decision.get("research_decision"),
        "resource_action": decision.get("resource_action"),
        "decision_reason": decision.get("reason"),
        "formal_admission": spec.FORMAL_ADMISSION,
        "limitations": list(limitations),
        "unmet_items": list(unmet_items),
        "machine_hours": dict(machine_hours),
        "requests": int(requests),
        "bytes": int(bytes_written),
        "new_data_cost": int(new_data_cost),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> str:
    """中文报告（结论身份 / 主表 / 新增信息 / 压力 / 账户 / 投入账 / 下一步）。"""
    payload = dict(extra or {})
    lines: list[str] = []
    lines.append("# LR5M-ML-10-v1 研究包结果报告")
    lines.append("")
    lines.append("## 0. 结论摘要")
    lines.append("")
    lines.append(
        f"- 研究结论：**{summary.get('research_decision')}**（{summary.get('decision_reason')}）"
    )
    lines.append(f"- 资源动作：`{summary.get('resource_action')}`")
    lines.append(
        f"- 工程状态：`{summary.get('engineering_status')}`；数据状态：`{summary.get('data_status')}`"
    )
    scope = summary.get("verification_scope") or {}
    if scope:
        lines.append("- 验证范围：" + "；".join(f"{key}=`{value}`" for key, value in scope.items()))
    lines.append(f"- 正式准入：`{summary.get('formal_admission')}`（研究完成不等于 P7 出口通过）")
    lines.append("")
    lines.append("## 1. 结论身份")
    lines.append("")
    for key in (
        "candidate_id",
        "protocol_sha256",
        "spec_sha256",
        "code_commit",
        "dirty_patch_sha",
        "data_status",
        "baseline_sha",
        "run_id",
        "campaign_id",
    ):
        if key in summary:
            lines.append(f"- `{key}` = `{summary.get(key)}`")
    lines.append("")
    lines.append("## 2. 两资本主表")
    lines.append("")
    targets = summary.get("economic_targets_by_capital") or {}
    if not targets:
        lines.append("主表未运行（数据依赖未就绪），本节无经济读数。")
    else:
        for capital, block in targets.items():
            lines.append(f"### {capital}")
            metrics = block.get("metrics") or {}
            lines.append("")
            lines.append("| 臂 | CAGR | MDD_loss | Sharpe | 累计收益 |")
            lines.append("|---|---:|---:|---:|---:|")
            for arm in analysis.ARMS:
                row = metrics.get(arm, {})
                lines.append(
                    f"| {arm} | {_fmt(row.get('cagr'))} | {_fmt(row.get('mdd_loss'))} | "
                    f"{_fmt(row.get('sharpe'))} | {_fmt(row.get('total_return'))} |"
                )
            values = block.get("targets", {}).get("values", {})
            checks = block.get("targets", {}).get("checks", {})
            lines.append("")
            lines.append("| 目标 | 读数 | 达标 |")
            lines.append("|---|---:|---|")
            for name, value in values.items():
                flag = checks.get(name.replace("cagr_delta", "cagr_delta_ok"), "")
                lines.append(f"| {name} | {_fmt(value)} | {flag} |")
            lines.append("")
            lines.append(f"全部达标：`{block.get('targets', {}).get('all_met')}`")
            lines.append("")
    lines.append("## 3. 新增信息证据")
    lines.append("")
    for item in payload.get("increment", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 4. 压力与分段")
    lines.append("")
    lines.append(
        f"- 压力状态：`{json.dumps(summary.get('pressure_status'), ensure_ascii=False, default=str)}`"
    )
    lines.append("")
    lines.append("## 5. 账户可信度")
    lines.append("")
    for item in payload.get("account", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 6. 投入账")
    lines.append("")
    lines.append(f"- 机器小时：`{json.dumps(summary.get('machine_hours'), ensure_ascii=False)}`")
    budget = summary.get("budget") or {}
    if budget:
        lines.append(
            f"- 已登记累计：{budget.get('registered_machine_hours')} 小时；"
            f"剩余 {budget.get('remaining_hours')} 小时 / 预算 {budget.get('total_budget_hours')} 小时"
        )
        lines.append(f"- 未登记开发耗时：{budget.get('unregistered_development_hours')}")
    lines.append(f"- 请求次数：{summary.get('requests')}")
    lines.append(f"- 落盘字节（合计）：{summary.get('bytes')}")
    storage = summary.get("storage") or {}
    if storage:
        lines.append(f"- 落盘分类：`{json.dumps(storage, ensure_ascii=False)}`")
    lines.append(f"- 新增采购（元）：{summary.get('new_data_cost')}")
    lines.append("")
    lines.append("## 7. 限制与未完成项")
    lines.append("")
    for item in summary.get("limitations", []):
        lines.append(f"- {item}")
    for item in summary.get("unmet_items", []):
        lines.append(f"- 未完成：{item}")
    lines.append("")
    lines.append("## 8. 下一步")
    lines.append("")
    lines.append(payload.get("next_step", "见研究结论与未完成项。"))
    lines.append("")
    lines.append(
        "> 本报告为历史证据等级 H 的研究记录；不称“前向验证通过”或“真实日历验证通过”，"
        "也不构成投资建议。"
    )
    return "\n".join(lines) + "\n"


def write_report(
    directory: Path, summary: Mapping[str, Any], extra: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """写 `summary.json` 与中文报告，返回路径。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    summary_path.write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    report_path = directory / "report.md"
    report_path.write_text(render_markdown(summary, extra), encoding="utf-8")
    return {"summary": str(summary_path), "report": str(report_path)}
