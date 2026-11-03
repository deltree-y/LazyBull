# -*- coding: utf-8 -*-
"""P5b T5 回撤裁判判定表与 §7.2 报告（calibration_only）。

报告硬口径
==========
- ``scope=historical_testbed``、``virtual_clock=true``、``evidence_class=calibration_only``、
  ``not_forward_evidence=true``、``not_sla_evidence=true``、``row_domain=structural``、
  ``schedule_mode=replay``；
- DD-W 与 DD-I 的注入（λ/ζ）、A/A、阈值 ``c^W_k``/``c^I_k``、MDE、horizon **分别列出**，
  逐单元（场景 × 裁判）明细完整，**不得**出现跨场景聚合/加权字段；
- 逐档必须同时给：纯注入效应、含噪观察效应、各 look 实际效应、可构造率 C/M；
- 报告**不含墙钟**（墙钟落 ``machine_time.json``，保证同 seed 重跑逐字节一致）；
- H5 判定不在此宣布：判定表只写 ``h5_claim=evaluated_not_declared_pass``、
  ``gate_owner=user_at_merge_acceptance``。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "REPORT_SCHEMA",
    "VERDICT_SCHEMA",
    "build_drawdown_report",
    "build_drawdown_verdict",
    "render_drawdown_markdown",
]

#: 判定表 schema
VERDICT_SCHEMA = "p5b_testbed_drawdown_verdict/v1"
#: §7.2 回撤部分报告 schema
REPORT_SCHEMA = "p5b_testbed_t5_drawdown_referee_report/v1"


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


def h5_conditions(holdout: Mapping[str, Any], power: Mapping[str, Any]) -> Dict[str, Any]:
    """H5 三条件（全部由数据推导；闸门判定仍归用户）。

    - ``aa``：两个方向各自的 Wilson 上界门；
    - ``injection``：纯注入效应落在 0.05pp 容差内的格数（构造可行性的实测值）；
    - ``mde``：两个方向的 MDE 是否登记（≠ A/A 是否过门）。
    """
    cells = [
        entry
        for levels in dict(power["by_direction_level_scene"]).values()
        for scenes in levels.values()
        for entry in scenes.values()
    ]
    comparable = [entry for entry in cells if not entry.get("rate_not_comparable", False)]
    within_by_direction = {"dd_worsen": 0, "dd_improve": 0}
    comparable_by_direction = {"dd_worsen": 0, "dd_improve": 0}
    for entry in comparable:
        direction = entry["direction"]
        comparable_by_direction[direction] += 1
        sign = 1.0 if direction == "dd_improve" else -1.0
        goal = sign * float(entry["level_pp"])
        pure = entry["pure_effect_pp"]["median"]
        if pure is not None and abs(float(pure) - goal) <= 0.05 + 1e-9:
            within_by_direction[direction] += 1
    within = int(sum(within_by_direction.values()))
    aa = {
        "worsen": bool(holdout["usable_by_direction"].get("dd_worsen")),
        "improve": bool(holdout["usable_by_direction"].get("dd_improve")),
    }
    worsen_mde = dict(power["mde"]["MDE_drawdown_worsen"])
    improve_mde = dict(power["mde"]["MDE_drawdown_improve"])
    mde = {
        "worsen": bool(worsen_mde.get("registered")),
        "improve": bool(improve_mde.get("registered")),
        "scenes_registered": {
            "worsen": list(worsen_mde.get("scenes_registered") or []),
            "improve": list(improve_mde.get("scenes_registered") or []),
        },
        "scenes_total": {
            "worsen": list(worsen_mde.get("scenes_total") or []),
            "improve": list(improve_mde.get("scenes_total") or []),
        },
        "gate_scope": "方向级 registered = 至少一个场景登记 MDE；逐场景状态见 MDE_drawdown_* 对象",
    }
    injection_ok_by_direction = {
        direction: comparable_by_direction[direction] > 0
        and within_by_direction[direction] == comparable_by_direction[direction]
        for direction in ("dd_worsen", "dd_improve")
    }
    failures = []
    if not aa["worsen"]:
        failures.append("dd_worsen.aa_gate")
    if not aa["improve"]:
        failures.append("dd_improve.aa_gate")
    for direction, ok in injection_ok_by_direction.items():
        if not ok:
            failures.append(f"{direction}.injection_out_of_tolerance")
    if not mde["worsen"]:
        failures.append("dd_worsen.mde_not_registered")
    if not mde["improve"]:
        failures.append("dd_improve.mde_not_registered")
    return {
        "aa": aa,
        "injection": {
            "tolerance_pp": 0.05,
            "cells_within_tolerance": int(within),
            "cells_comparable": int(len(comparable)),
            "cells_total": len(cells),
            "comparable_rule": "C/M >= constructibility_floor(0.80) 的格参与容差判定",
            "boundary_degeneracy_note": (
                "求解器取最小干预左沿 ⇒ 可构造格的纯注入恒为 |目标|-0.05pp（容差边界），"
                "故本条件的实测信息主要是「存在可比格」；不得据此宣称注入精度有富余"
            ),
            "within_by_direction": within_by_direction,
            "comparable_by_direction": comparable_by_direction,
            "ok_by_direction": injection_ok_by_direction,
            "ok": all(injection_ok_by_direction.values()),
            "target_convention": "DD-I 目标 +level、DD-W 目标 -level（按方向签名）",
        },
        "mde": mde,
        "holdout_gate_only": True,
        "all_conditions_met": not failures,
        "why_not_met": failures,
    }


def _unit_rows(holdout: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "unit": key,
            "scene": item["scene"],
            "null": item["null"],
            "referee": item["referee"],
            "successes": int(item["successes"]),
            "total": int(item["total"]),
            "rate": float(item["rate"]),
            "wilson_low": float(item["wilson_low"]),
            "wilson_high": float(item["wilson_high"]),
            "pass": bool(item["pass"]),
            "failure_reasons": list(item["failure_reasons"]),
        }
        for key, item in sorted(dict(holdout["units"]).items())
    ]


def build_drawdown_verdict(
    *,
    run_id: str,
    campaign_id: str,
    manifest_sha256: str,
    calibrator_sha256: str,
    thresholds_sha256: str,
    holdout_sha256: str,
    power_sha256: str,
    thresholds: Mapping[str, Any],
    holdout: Mapping[str, Any],
    power: Mapping[str, Any],
    implementation_id: str,
    windows: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """判定表：三环哈希 + 逐单元门结果 + 两个 MDE（不含闸门结论）。"""
    mde = dict(power["mde"])
    return {
        "schema": VERDICT_SCHEMA,
        **_scope(),
        "run_id": str(run_id),
        "campaign_id": str(campaign_id),
        "implementation_id": str(implementation_id),
        "referee": "drawdown_worsen_and_improve",
        "hash_chain": {
            "manifest_sha256": str(manifest_sha256),
            "calibrator_sha256": str(calibrator_sha256),
            "thresholds_sha256": str(thresholds_sha256),
            "holdout_sha256": str(holdout_sha256),
            "power_sha256": str(power_sha256),
        },
        "drawdown_referee": {
            "crash_windows": [dict(item) for item in windows],
            "t_anchor": int(power["plan"]["look_days"][-1]),
            "horizons": [int(day) for day in power["plan"]["look_days"]],
            "c_w_by_look": [float(item) for item in thresholds["c_w_by_look"]],
            "c_i_by_look": [float(item) for item in thresholds["c_i_by_look"]],
            "direction_check": dict(thresholds["direction_check"]),
            "unit_definition": str(holdout["unit_definition"]),
            "unit_count": int(holdout["unit_count"]),
            "units_failed": list(holdout["failed_units"]),
            "wilson_upper_max": float(holdout["wilson_upper_max"]),
            "wilson_confidence": float(holdout["wilson_confidence"]),
            "usable_by_direction": dict(holdout["usable_by_direction"]),
            "referee_usable": bool(holdout["referee_usable"]),
            "rows": _unit_rows(holdout),
            "MDE_drawdown_worsen": dict(mde["MDE_drawdown_worsen"]),
            "MDE_drawdown_improve": dict(mde["MDE_drawdown_improve"]),
            "worsen_unreachable_in_window": bool(
                bool(holdout["usable_by_direction"].get("dd_worsen"))
                and not list(dict(mde["MDE_drawdown_worsen"]).get("scenes_registered") or [])
            ),
            "worsen_unreachable_reason": (
                "c^W_k 取 min-over-scene（最保守），而 DD-W 的锚上读数被噪声左尾主导，"
                "12pp 网格内所有场景检出率均未达 80%（A/A 未过门时本字段为 false，归因不混同）"
            ),
        },
        "h5_conditions": h5_conditions(holdout, power),
        "h5_boundary": {
            "gate": "H5",
            "h5_claim": "evaluated_not_declared_pass",
            "gate_owner": "user_at_merge_acceptance",
            "note": "T5 只提供 A/A、注入与 MDE 证据；闸门通过与生产裁决权由用户确认",
        },
        "limitations": [
            "HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE",
            "DD-W 与 DD-I 分别判定，不得互相抵扣",
            "不可构造的实现按未检出计入主分母（M），C/M 单独披露",
        ],
    }


def _cell_rows(power: Mapping[str, Any]) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for direction, levels in sorted(dict(power["by_direction_level_scene"]).items()):
        rows[direction] = {}
        for level, scenes in sorted(levels.items(), key=lambda item: float(item[0])):
            rows[direction][level] = {
                scene: {
                    "successes": int(item["successes"]),
                    "total": int(item["total"]),
                    "rate": float(item["rate"]),
                    "wilson_low": float(item["wilson_low"]),
                    "wilson_high": float(item["wilson_high"]),
                    "constructible_count": int(item["constructible_count"]),
                    "constructibility_rate": float(item["constructibility_rate"]),
                    "constructibility_floor": float(item.get("constructibility_floor", 0.80)),
                    "rate_not_comparable": bool(item.get("rate_not_comparable", False)),
                    "rate_not_comparable_note": str(item.get("rate_not_comparable_note", "")),
                    "not_constructible_count": int(item["not_constructible_count"]),
                    "parameter_stats": dict(item["parameter_stats"]),
                    "pure_effect_pp": dict(item["pure_effect_pp"]),
                    "observed_effect_pp": dict(item["observed_effect_pp"]),
                    "per_look_effect_pp": [dict(entry) for entry in item["per_look_effect_pp"]],
                    "per_look_triggers": [int(value) for value in item["per_look_triggers"]],
                    "realization_id": item["realization_id"],
                    "event": item["event"],
                }
                for scene, item in sorted(scenes.items())
            }
    return rows


def build_drawdown_report(
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
    upstream: Mapping[str, Any],
    source: Any,
    ledger_sample: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """组装 §7.2 的回撤裁判部分（逐单元 + 逐档明细，无跨场景聚合）。"""
    windows = list(manifest["drawdown_referee"]["windows"]["frozen_list"])
    return {
        "schema": REPORT_SCHEMA,
        **_scope(),
        "campaign_id": str(campaign_id),
        "run_id": str(run_id),
        "referee": "drawdown_worsen_and_improve",
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
        "upstream": {
            "parent_manifest": dict(manifest["files"]["parent_manifest"]),
            "parent_thresholds": dict(manifest["files"]["parent_thresholds"]),
            "verified": dict(upstream),
        },
        "sample": {
            "H": int(power["plan"]["H"]),
            "K": int(power["plan"]["K"]),
            "look_days": [int(day) for day in power["plan"]["look_days"]],
            "look_blocks": [int(j) for j in power["plan"]["look_blocks"]],
            "t_anchor": int(manifest["drawdown_referee"]["reading"]["t_anchor"]),
            "window_samples": int(manifest["drawdown_referee"]["windows"]["window_samples"]),
            "block_length": int(manifest["block_resample"]["block_length"]),
            "block_method": str(manifest["block_resample"]["method"]),
            "block_center": str(manifest["block_resample"]["center"]),
            "mdd_definition": str(manifest["drawdown_referee"]["mdd"]["definition"]),
            "mdd_percent_conversion": str(
                manifest["drawdown_referee"]["mdd"]["percent_conversion"]
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
            "b0_window": dict(manifest["source_window"]["window"]),
            "ledger_sample": None if ledger_sample is None else dict(ledger_sample),
        },
        "crash_windows": {
            "frozen_list": [dict(item) for item in windows],
            "window_samples": int(manifest["drawdown_referee"]["windows"]["window_samples"]),
            "selection_rule": str(manifest["drawdown_referee"]["windows"]["selection_rule"]),
            "tie_break": str(manifest["drawdown_referee"]["windows"]["tie_break"]),
            "boundary_rule": str(manifest["drawdown_referee"]["windows"]["boundary_rule"]),
            "fixed_for_all_realizations": True,
        },
        "injection": {
            "dd_improve": dict(manifest["drawdown_referee"]["injection"]["dd_improve"]),
            "dd_worsen": dict(manifest["drawdown_referee"]["injection"]["dd_worsen"]),
            "solver": dict(manifest["drawdown_referee"]["solver"]),
            "counterfactual": dict(manifest["drawdown_referee"]["counterfactual"]),
        },
        "thresholds": {
            "c_w_by_look": [float(item) for item in thresholds["c_w_by_look"]],
            "c_i_by_look": [float(item) for item in thresholds["c_i_by_look"]],
            "c_w_by_scene": {
                key: [float(value) for value in values]
                for key, values in dict(thresholds["c_w_by_scene"]).items()
            },
            "c_i_by_scene": {
                key: [float(value) for value in values]
                for key, values in dict(thresholds["c_i_by_scene"]).items()
            },
            "alpha_train": float(thresholds["alpha_train"]),
            "alpha_per_look": float(thresholds["alpha_per_look"]),
            "determining_units_worsen": list(thresholds["determining_units_worsen"]),
            "determining_units_improve": list(thresholds["determining_units_improve"]),
            "direction_check": dict(thresholds["direction_check"]),
            "events": dict(thresholds["events"]),
            "units_worsen": [dict(item) for item in thresholds["units_worsen"]],
            "units_improve": [dict(item) for item in thresholds["units_improve"]],
        },
        "aa": dict(manifest["drawdown_referee"]["aa"]),
        "holdout": {
            "unit_definition": str(holdout["unit_definition"]),
            "unit_count": int(holdout["unit_count"]),
            "reporting_rule": str(holdout["reporting_rule"]),
            "wilson_confidence": float(holdout["wilson_confidence"]),
            "wilson_upper_max": float(holdout["wilson_upper_max"]),
            "units": {key: dict(item) for key, item in sorted(dict(holdout["units"]).items())},
            "failed_units": list(holdout["failed_units"]),
            "usable_by_direction": dict(holdout["usable_by_direction"]),
            "referee_usable": bool(holdout["referee_usable"]),
            "counts": dict(holdout["counts"]),
        },
        "effects": {
            "grid_pp": [float(item) for item in power["grid_pp"]],
            "target_power": float(power["target_power"]),
            "by_direction_level_scene": _cell_rows(power),
            "mde": dict(power["mde"]),
            "reference_on_b0": dict(power.get("reference_on_b0") or {}),
        },
        "counts": dict(manifest["counts"]),
        "implementation": {
            "implementation_id": str(calibrator["implementation_id"]),
            "covered_modules": list(manifest["files"]["calibrator"]["modules"]),
            "exclusions": dict(manifest["files"]["calibrator"].get("exclusions") or {}),
            "substreams": dict(manifest["substreams"]),
        },
        "h6_numeric": dict(manifest["h6_numeric"]),
        "h5_boundary": dict(verdict["h5_boundary"]),
        "machine_time": {
            "artifact": "machine_time.json",
            "excluded_from_determinism": True,
            "reason": "墙钟不属于产物身份；单独落盘以保证同 seed 重跑逐字节一致",
        },
        "deviation_registry": list(manifest["deviation_registry"]),
        "limitations": list(manifest["limitations"]),
    }


def _reference_text(report: Mapping[str, Any]) -> str:
    """B0 单路径 λ/ζ 复算的可读行（与 T0.5b §5 预检对照）。"""
    reference = dict(report["effects"].get("reference_on_b0") or {})
    parts = []
    for direction, levels in sorted(reference.items()):
        pairs = [
            f"{level}pp={value.get('parameter')}"
            for level, value in sorted(levels.items(), key=lambda item: float(item[0]))
        ]
        parts.append(f"{direction}: " + ", ".join(pairs))
    return " | ".join(parts) if parts else "—"


def _effect_rows(report: Mapping[str, Any]) -> List[str]:
    """逐方向 × 逐档 × 逐场景的功效行（md 表体）。"""
    rows: List[str] = []
    for direction, levels in report["effects"]["by_direction_level_scene"].items():
        for level, scenes in levels.items():
            for scene, item in scenes.items():
                median = item["parameter_stats"].get("median")
                pure = item["pure_effect_pp"].get("median")
                observed = item["observed_effect_pp"].get("median")
                flag = "（不可比）" if item.get("rate_not_comparable") else ""
                rows.append(
                    f"| {direction} | {level} | {scene} | {item['rate']:.2%}{flag} | "
                    f"{item['wilson_high']:.2%} | {item['constructibility_rate']:.2%} | "
                    f"{'—' if median is None else format(median, '.4f')} | "
                    f"{'—' if pure is None else format(pure, '.4f')} | "
                    f"{'—' if observed is None else format(observed, '.4f')} |"
                )
    return rows


def render_drawdown_markdown(report: Mapping[str, Any]) -> str:
    """渲染可读报告（崩盘窗 + 方向阈值 + 逐单元 holdout + 逐档功效）。"""
    sample = report["sample"]
    lines: List[str] = [
        f"# P5b T5 回撤裁判校准报告（{report['run_id']}）",
        "",
        f"- campaign：`{report['campaign_id']}`",
        f"- 范围：`{report['scope']}` / `{report['evidence_class']}` / virtual_clock=true",
        f"- 裁判：`{report['referee']}`；H={sample['H']}，look_days={sample['look_days']}，"
        f"T_anchor={sample['t_anchor']}",
        f"- 冻结窗口：B0 最后 {sample['window_samples']} 个收益样本；"
        f"block_length={sample['block_length']}（{sample['block_method']}）",
        f"- manifest sha256：`{report['identity']['manifest_sha256']}`",
        f"- 校准器 sha256：`{report['identity']['calibrator_sha256']}`",
        f"- 临界值产物 sha256：`{report['identity']['thresholds_sha256']}`",
        f"- 上游收益 manifest：`{report['upstream']['parent_manifest']['sha256']}`",
        f"- 上游收益临界值：`{report['upstream']['parent_thresholds']['sha256']}`",
        "",
        "## 1. 崩盘窗（冻结，全实现共享）",
        "",
        "| # | peak | trough | 深度(pp) | 窗口样本 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in report["crash_windows"]["frozen_list"]:
        lines.append(
            f"| {item['rank']} | {item['peak']} | {item['trough']} | "
            f"{item['depth_pp']:.4f} | {item['window_samples']} |"
        )
    lines.extend(
        [
            "",
            "## 2. 两个方向的临界值（同一组 A/A 读数）",
            "",
            "| look | look_days | c^W_k（min over scene） | c^I_k（max over scene） |",
            "|---:|---:|---:|---:|",
        ]
    )
    for index, day in enumerate(sample["look_days"]):
        lines.append(
            f"| {index} | {day} | {report['thresholds']['c_w_by_look'][index]:.6f} | "
            f"{report['thresholds']['c_i_by_look'][index]:.6f} |"
        )
    lines.extend(
        [
            "",
            "- 本表只表示 **A/A（零假设）过门**，不含注入与 MDE；H5 三条件见判定表 `h5_conditions`",
            "",
            "## 3. holdout 逐单元（Wilson 95% 上界 ≤ "
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
    usable = report["holdout"]["usable_by_direction"]
    lines.extend(
        [
            "",
            f"- `dd_worsen` 可用性：**{'通过' if usable.get('dd_worsen') else '不通过'}**；"
            f"`dd_improve` 可用性：**{'通过' if usable.get('dd_improve') else '不通过'}**"
            "（分别判定，不得互相抵扣）",
            "",
            "## 4. 功效、可构造率与 MDE（逐方向 × 逐档 × 逐场景）",
            "",
            "| 方向 | 档(pp) | 场景 | 检出率 | Wilson 高 | C/M | λ/ζ 中位 | 纯注入(pp) | 含噪观察(pp) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(_effect_rows(report))
    mde = report["effects"]["mde"]
    determining_w = ",".join(report["thresholds"]["determining_units_worsen"])
    determining_i = ",".join(report["thresholds"]["determining_units_improve"])
    lines.extend(["", "| 场景 | MDE_drawdown_worsen | MDE_drawdown_improve |", "|---|---|---|"])
    worsen = mde["MDE_drawdown_worsen"]
    improve = mde["MDE_drawdown_improve"]
    for scene in sorted(worsen["by_scene"]):
        left = worsen["by_scene"][scene]
        right = improve["by_scene"][scene]
        lines.append(
            f"| {scene} | {'—' if left is None else format(left, 'g') + 'pp'}"
            f"（{worsen['status_by_scene'][scene]}） | "
            f"{'—' if right is None else format(right, 'g') + 'pp'}"
            f"（{improve['status_by_scene'][scene]}） |"
        )
    upper = [
        f"{scene}({direction})"
        for direction, entry in (("worsen", worsen), ("improve", improve))
        for scene, flag in entry["at_grid_upper_bound"].items()
        if flag
    ]
    non_const = dict(mde.get("not_constructible_levels") or {})
    lines.extend(
        [
            "",
            "- 档名 = 目标值；锚上**实际**纯注入 = |目标| − 0.05pp（最小干预左沿），见上表「纯注入」列",
            "- 各 look 的实际效应（mean/min/max）与逐 look 触发数见 "
            "`drawdown_report.json → effects.by_direction_level_scene[*].per_look_effect_pp`"
            " 与 `drawdown_power.json`；本表只列锚上读数",
            "- 校验单元的有效独立 look ≈ 2（looks 嵌套：同一路径、base 恒定、MDD 随 horizon 单调）",
            f"- 决定单元（determining units）：worsen={determining_w} / improve={determining_i}"
            "（min/max-over-scene ⇒ 单一场景同时决定两侧阈值；8/8 通过不等于 8 次独立验收）",
            f"- 命中网格上界的单元：{', '.join(upper) if upper else '无'}"
            "（命中上界只能宣称 MDE ≤ 上界档）",
            f"- 存在不可构造档（按未检出计入 M，C/M 逐格披露）：{non_const}",
            f"- B0 单路径 λ/ζ 复算（与 T0.5b §5 预检对照）：{_reference_text(report)}",
            f"- 实现数：训练 {report['counts']['train_per_unit']}/单元，"
            f"holdout {report['counts']['holdout_per_unit']}/单元，"
            f"效应 {report['counts']['effect_per_scene_per_level']}/档·场景；"
            f"seed={report['counts']['seed']}，prefix=`{report['counts']['realization_prefix']}`",
            "",
            "## 5. 边界与限制",
            "",
            "- H5 闸门判定不在本报告内：T5 只给证据，闸门由用户在合并验收时确认；",
            "- 不可构造的实现按未检出计入主分母（M），C/M 单独披露；",
        ]
    )
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 6. 已知差异登记（deviation_registry）",
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
