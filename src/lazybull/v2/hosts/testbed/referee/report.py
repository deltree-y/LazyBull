# -*- coding: utf-8 -*-
"""P5b T4 收益裁判判定表与 §7.2 报告（calibration_only，不含回撤段）。

报告硬口径
==========
- ``scope=historical_testbed``、``virtual_clock=true``、``evidence_class=calibration_only``、
  ``not_forward_evidence=true``、``not_sla_evidence=true``、``row_domain=structural``、
  ``schedule_mode=replay``；
- 逐单元明细（场景 × 零假设）**必须完整出现**，且**不得**出现跨场景聚合/加权字段；
- MDE 只在 A/A 过门后登记，并逐场景给出（含「5δ 未达 80%」的如实状态）；
- 报告**不含墙钟**（墙钟落 ``machine_time.json``，避免破坏重跑逐字节一致）；
- H4 闸门判定不在此声明：T4 只给证据，闸门由用户在合并验收时确认。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "REPORT_SCHEMA",
    "VERDICT_SCHEMA",
    "build_income_report",
    "build_referee_verdict",
    "render_markdown",
]

#: 判定表 schema
VERDICT_SCHEMA = "p5b_testbed_referee_verdict/v1"
#: §7.2 收益部分报告 schema
REPORT_SCHEMA = "p5b_testbed_t4_income_referee_report/v1"


def _scope() -> Dict[str, Any]:
    return {
        "scope": "historical_testbed",
        "evidence_class": "calibration_only",
        "virtual_clock": True,
        "not_forward_evidence": True,
        "not_sla_evidence": True,
        "not_real_candidate": True,
        "row_domain": "structural",
        "schedule_mode": "replay",
    }


def build_referee_verdict(
    *,
    run_id: str,
    campaign_id: str,
    manifest_sha256: str,
    calibrator_sha256: str,
    thresholds_sha256: str,
    holdout_sha256: str,
    power_sha256: str,
    holdout: Mapping[str, Any],
    power: Mapping[str, Any],
    implementation_id: str,
) -> Dict[str, Any]:
    """判定表：三环哈希 + 逐单元门结果 + MDE 登记状态（不含闸门结论）。"""
    units = dict(holdout["units"])
    return {
        "schema": VERDICT_SCHEMA,
        **_scope(),
        "run_id": str(run_id),
        "campaign_id": str(campaign_id),
        "implementation_id": str(implementation_id),
        "referee": "income_psd_boundary_centered",
        "hash_chain": {
            "manifest_sha256": str(manifest_sha256),
            "calibrator_sha256": str(calibrator_sha256),
            "thresholds_sha256": str(thresholds_sha256),
            "holdout_sha256": str(holdout_sha256),
            "power_sha256": str(power_sha256),
        },
        "income_referee": {
            "unit_definition": holdout["unit_definition"],
            "unit_count": int(holdout["unit_count"]),
            "units_failed": list(holdout["failed_units"]),
            "wilson_upper_max": float(holdout["wilson_upper_max"]),
            "wilson_confidence": float(holdout["wilson_confidence"]),
            "invalid_rate_per_look_max": float(holdout["invalid_rate_per_look_max"]),
            # R2-P5B-04：资格由**实际子流阶段**推导，且在 holdout/verdict/report/CLI/MDE 间一致。
            "aa_passed": bool(holdout["referee_usable"]),
            "holdout_stage": str(holdout.get("stage") or ""),
            "conditional_reproduction": bool(
                holdout["referee_usable"] and holdout.get("stage") == "holdout"
            ),
            "independent_acceptance": bool(
                holdout["referee_usable"] and holdout.get("stage") == "holdout_independent_v1"
            ),
            "acceptance_basis": (
                "independent_acceptance：holdout 使用一次性独立子流 holdout_independent_v1 且全部单元过门"
                if holdout.get("stage") == "holdout_independent_v1"
                else "conditional_reproduction：holdout 与 T0.5b PSD r3 同源，读数已观察"
            ),
            "referee_usable": bool(
                holdout["referee_usable"] and holdout.get("stage") == "holdout_independent_v1"
            ),
            "rows": [
                {
                    "unit": key,
                    "successes": int(item["successes"]),
                    "total": int(item["total"]),
                    "rate": float(item["rate"]),
                    "wilson_low": float(item["wilson_low"]),
                    "wilson_high": float(item["wilson_high"]),
                    "pass": bool(item["pass"]),
                    "failure_reasons": list(item["failure_reasons"]),
                }
                for key, item in sorted(units.items())
            ],
            "mde_registered": bool(power["mde"]["registered"]),
            "mde_by_scene": dict(power["mde"]["by_scene"]),
            "mde_status_by_scene": dict(power["mde"]["status_by_scene"]),
            # 机器可读的"命中网格上界"标记：下游只读判定表时也不会把上界档当精确 MDE
            "mde_at_grid_upper_bound": dict(power["mde"].get("at_grid_upper_bound") or {}),
            "mde_grid_upper_level": power["mde"].get("grid_upper_level"),
            "detection_event": str(power["detection_event"]),
        },
        "drawdown_referee": {
            "status": "not_implemented_in_t4",
            "note": "DD-W / DD-I 归 P5b T5；本判定表不产生任何回撤读数",
        },
        "h4_boundary": {
            "gate": "H4",
            "t4_claim": "evaluated_not_declared_pass",
            "gate_owner": "user_at_merge_acceptance",
            "note": "T4 只提供 holdout/功效证据；闸门通过与生产裁决权由用户确认",
        },
        "limitations": [
            "HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE",
            "holdout 与训练共享同一对真实源，仅以互斥源折与独立子流隔离",
            "任一单元不过即裁判不可用；不得删除困难场景或扩样本回调阈值",
            (
                "holdout realization 与 T0.5b PSD r3 同一子流编码与 seed 根（已在 r3 观察）："
                "本次为条件复现，不构成独立 holdout 验收"
            ),
        ],
    }


def build_income_report(
    *,
    campaign_id: str,
    run_id: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    calibrator: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    thresholds_sha256: str,
    holdout: Mapping[str, Any],
    holdout_sha256: str,
    power: Mapping[str, Any],
    power_sha256: str,
    verdict: Mapping[str, Any],
    verdict_sha256: str,
    source: Any,
    ledger_sample: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """组装 §7.2 的收益裁判部分（逐单元明细完整，无跨场景聚合）。"""
    plan = dict(holdout["plan"])
    units = {
        key: {
            "scene": item["scene"],
            "null": item["null"],
            "mu": float(item["mu"]),
            "H": int(item["H"]),
            "referee": item["referee"],
            "realization_id": item["realization_id"],
            "successes": int(item["successes"]),
            "total": int(item["total"]),
            "rate": float(item["rate"]),
            "wilson_low": float(item["wilson_low"]),
            "wilson_high": float(item["wilson_high"]),
            "invalid_rate_by_look": [float(x) for x in item["invalid_rate_by_look"]],
            "pass": bool(item["pass"]),
            "failure_reasons": list(item["failure_reasons"]),
        }
        for key, item in sorted(dict(holdout["units"]).items())
    }
    return {
        "schema": REPORT_SCHEMA,
        **_scope(),
        "campaign_id": str(campaign_id),
        "run_id": str(run_id),
        "referee": "income_psd_boundary_centered",
        "identity": {
            "manifest_sha256": str(manifest_sha256),
            "calibrator_sha256": str(calibrator["sha256"]),
            "calibrator_implementation_id": str(calibrator["implementation_id"]),
            "calibrator_modules": dict(calibrator["modules"]),
            "thresholds_sha256": str(thresholds_sha256),
            "holdout_sha256": str(holdout_sha256),
            "power_sha256": str(power_sha256),
            "verdict_sha256": str(verdict_sha256),
        },
        "hash_chain": dict(verdict["hash_chain"]),
        "sample": {
            **plan,
            "block_length": int(manifest["block_resample"]["block_length"]),
            "block_method": str(manifest["block_resample"]["method"]),
            "block_center": str(manifest["block_resample"]["center"]),
            "statistic": str(manifest["income_referee"]["statistic"]["name"]),
            "delta_log_daily": float(manifest["income_referee"]["statistic"]["delta_log_daily"]),
            "delta_annualization": dict(
                manifest["income_referee"]["statistic"]["delta_annualization"]
            ),
        },
        "sample_sources": {
            "source_pair": dict(manifest["source_pair"]["rows"]),
            "source_run_id": str(manifest["source_pair"]["source_run_id"]),
            "train_folds": list(manifest["source_pair"]["train_folds"]),
            "holdout_folds": list(manifest["source_pair"]["holdout_folds"]),
            "train_pool_rows": int(source.train_pool["rows"]),
            "holdout_pool_rows": int(source.holdout_pool["rows"]),
            "date_semantics": str(manifest["source_pair"]["date_blocks"]["date_semantics"]),
            "ledger_sample": None if ledger_sample is None else dict(ledger_sample),
        },
        "label_cutoff": {
            "status": "not_applicable_statistical_layer",
            "reason": "统计层只消费已发布链式净值与冻结源对，不读取标签列",
        },
        "model_availability": {
            "status": "not_applicable_statistical_layer",
            "reason": "收益裁判校准不使用模型；模型截止证据属机械层（T1/T2）",
        },
        "thresholds": {
            "c_by_look": [float(value) for value in thresholds["c_by_look"]],
            "alpha_train": float(thresholds["config"]["alpha_train"]),
            "alpha_per_look": float(thresholds["config"]["alpha_per_look"]),
            "invalid_rate_per_look_max": float(thresholds["config"]["invalid_rate_per_look_max"]),
            "min_valid_quantile_sample": int(thresholds["config"]["min_valid_quantile_sample"]),
            "determining_units": list(thresholds["determining_units"]),
            "rule_steps": list(thresholds["rule_steps"]),
            "units": list(thresholds["units"]),
            "failures": list(thresholds["failures"]),
            "passed": bool(thresholds["passed"]),
        },
        "holdout": {
            "unit_definition": str(holdout["unit_definition"]),
            "unit_count": int(holdout["unit_count"]),
            "reporting_rule": str(holdout["reporting_rule"]),
            "wilson_confidence": float(holdout["wilson_confidence"]),
            "wilson_upper_max": float(holdout["wilson_upper_max"]),
            "units": units,
            "failed_units": list(holdout["failed_units"]),
            "referee_usable": bool(holdout["referee_usable"]),
        },
        "effects": {
            "grid_delta": [float(level) for level in power["grid_delta"]],
            "detection_event": str(power["detection_event"]),
            "target_power": float(power["target_power"]),
            "by_level_scene": {
                str(level): {
                    scene: {
                        "successes": int(item["successes"]),
                        "total": int(item["total"]),
                        "rate": float(item["rate"]),
                        "wilson_low": float(item["wilson_low"]),
                        "wilson_high": float(item["wilson_high"]),
                        "realization_id": item["realization_id"],
                    }
                    for scene, item in sorted(scenes.items())
                }
                for level, scenes in dict(power["by_level_scene"]).items()
            },
        },
        "mde": dict(power["mde"]),
        "counts": dict(manifest["counts"]),
        "implementation": {
            "implementation_id": str(calibrator["implementation_id"]),
            "covered_modules": list(manifest["files"]["calibrator"]["covered_modules"]),
            "substreams": dict(manifest["substreams"]),
        },
        "drawdown_referee": dict(manifest["drawdown_referee"]),
        "h6_numeric": dict(manifest["h6_numeric"]),
        "h4_boundary": dict(verdict["h4_boundary"]),
        "machine_time": {
            "artifact": "machine_time.json",
            "excluded_from_determinism": True,
            "reason": "墙钟不属于产物身份；单独落盘以保证同 seed 重跑逐字节一致",
        },
        "deviation_registry": list(manifest["deviation_registry"]),
        "limitations": list(manifest["limitations"]),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """渲染可读报告（判定表 + 逐单元 holdout + 功效/MDE）。"""
    sample = report["sample"]
    lines: List[str] = [
        f"# P5b T4 收益裁判校准报告（{report['run_id']}）",
        "",
        f"- campaign：`{report['campaign_id']}`",
        f"- 范围：`{report['scope']}` / `{report['evidence_class']}` / virtual_clock=true",
        f"- 裁判：`{report['referee']}`；K={sample['K']}，look_days={sample['look_days']}，"
        f"H={sample['H']}，block_length={sample['block_length']}",
        f"- 统计量：`{sample['statistic']}`；δ={sample['delta_log_daily']}（log/日）",
        f"- manifest sha256：`{report['identity']['manifest_sha256']}`",
        f"- 校准器 sha256：`{report['identity']['calibrator_sha256']}`",
        f"- 临界值产物 sha256：`{report['identity']['thresholds_sha256']}`",
        "",
        "## 1. 低尾临界值 c_k",
        "",
        "| look | J | look_days | c_k | 决定单元 |",
        "|---:|---:|---:|---:|---|",
    ]
    for index, j in enumerate(sample["look_blocks"]):
        lines.append(
            f"| {index} | {j} | {sample['look_days'][index]} | "
            f"{report['thresholds']['c_by_look'][index]:.10f} | "
            f"{report['thresholds']['determining_units'][index]} |"
        )
    lines.extend(
        [
            "",
            "## 2. holdout 逐单元（scene × null；Wilson 95% 上界 ≤ "
            f"{report['holdout']['wilson_upper_max']:.2%}）",
            "",
            "| 单元 | 触发 | 实现数 | 频率 | Wilson 低 | Wilson 高 | 结论 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for key, item in report["holdout"]["units"].items():
        verdict = "通过" if item["pass"] else "不通过：" + ",".join(item["failure_reasons"])
        lines.append(
            f"| {key} | {item['successes']} | {item['total']} | {item['rate']:.4%} | "
            f"{item['wilson_low']:.4%} | {item['wilson_high']:.4%} | {verdict} |"
        )
    lines.extend(
        [
            "",
            f"裁判可用性（A/A）：**{'通过' if report['holdout']['referee_usable'] else '不通过'}**"
            f"（单元 {report['holdout']['unit_count']} 个，"
            f"未通过 {len(report['holdout']['failed_units'])} 个）",
            "",
            "## 3. 功效与 MDE（逐场景，禁止聚合）",
            "",
            "| 效应档(δ) | 场景 | 检出 | 实现数 | 检出率 | Wilson 高 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for level, scenes in report["effects"]["by_level_scene"].items():
        for scene, item in scenes.items():
            lines.append(
                f"| {level} | {scene} | {item['successes']} | {item['total']} | "
                f"{item['rate']:.2%} | {item['wilson_high']:.2%} |"
            )
    lines.extend(["", "| 场景 | MDE | 状态 | 网格上界？ |", "|---|---:|---|---|"])
    at_upper = dict(report["mde"].get("at_grid_upper_bound") or {})
    for scene, value in report["mde"]["by_scene"].items():
        status = report["mde"]["status_by_scene"][scene]
        text = "—" if value is None else f"{value:g}δ"
        upper = "是（只能宣称 MDE ≤ 上界档）" if at_upper.get(scene) else "否"
        lines.append(f"| {scene} | {text} | {status} | {upper} |")
    lines.extend(
        [
            "",
            f"- MDE 登记：**{'是' if report['mde']['registered'] else '否（A/A 未过门）'}**；"
            "不得宣称 5δ 通用灵敏",
            f"- 实现数：训练 {report['counts']['train_per_unit']}/单元，"
            f"holdout {report['counts']['holdout_per_unit']}/单元，"
            f"效应 {report['counts']['effect_per_scene_per_level']}/档·场景；"
            f"seed={report['counts']['seed']}，prefix=`{report['counts']['realization_prefix']}`，"
            f"chunk={report['counts'].get('chunk')}",
            "",
            "## 4. 边界与限制",
            "",
            "- H4 闸门判定不在本报告内：T4 只给证据，闸门由用户在合并验收时确认；",
            "- 回撤裁判（DD-W/DD-I）未实现（T5），本报告不含任何回撤读数；",
        ]
    )
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 5. 已知差异登记（deviation_registry）",
            "",
            "| 项 | 设计值 | 冻结/实际值 | 原因 |",
            "|---|---|---|---|",
        ]
    )
    for entry in report.get("deviation_registry") or []:
        lines.append(
            f"| {entry.get('item')} | {entry.get('design_value')} | "
            f"{entry.get('frozen_value')} | {entry.get('reason')} |"
        )
    return "\n".join(lines) + "\n"
