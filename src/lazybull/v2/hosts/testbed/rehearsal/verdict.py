# -*- coding: utf-8 -*-
"""P5b T6 判定表：H1/H6/H7 的**作者侧检查**（闸门判定归用户）。

契约来源：设计 v4 §8（H1/H6/H7 条件）、§6.3（不得用另一条抵扣）、§7.2（报告字段）、
任务书「判定表必须固定 ``h6_claim=evaluated_not_declared_pass``、
``h7_claim=evaluated_not_declared_pass``、``gate_owner=user_at_merge_acceptance``，
并逐条列出作者侧检查结果，**不得**自行写闸门通过」。

状态语义（三值，由证据推导）
==========================
- ``pass``：该条件全部证据齐备且成立；
- ``fail``：证据显示不成立（如实记录，不用其它条目抵扣）；
- ``not_executed``：证据不可得（数据窗不足/臂不可构造/未运行），**不得**记为 pass。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.lazybull.v2.hosts.testbed.rehearsal.injection import (
    RehearsalEffect,
)
from src.lazybull.v2.hosts.testbed.rehearsal.readings import (
    ITEM_FAIL,
    ITEM_NOT_EXECUTED,
    ITEM_PARTIAL,
    ITEM_PASS,
)

__all__ = [
    "CONDITION_FAIL",
    "CONDITION_NOT_EXECUTED",
    "CONDITION_PASS",
    "VERDICT_SCHEMA",
    "aggregate_status",
    "build_verdict",
    "h1_conditions",
    "h6_conditions",
    "h7_conditions",
]

VERDICT_SCHEMA = "p5b_testbed_rehearsal_verdict/v1"
#: 条件状态码
CONDITION_PASS = "pass"
CONDITION_FAIL = "fail"
CONDITION_NOT_EXECUTED = "not_executed"


def _condition(identifier: str, requirement: str, status: str, evidence: Any) -> Dict[str, Any]:
    if status not in (CONDITION_PASS, CONDITION_FAIL, CONDITION_NOT_EXECUTED):
        raise ValueError(f"未知条件状态: {status!r}")
    return {
        "id": identifier,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
    }


def aggregate_status(conditions: Sequence[Mapping[str, Any]]) -> str:
    """条件集合 → 总体状态（任一 fail ⇒ fail；否则任一 not_executed ⇒ not_executed）。"""
    statuses = [item["status"] for item in conditions]
    if not statuses:
        return CONDITION_NOT_EXECUTED
    if CONDITION_FAIL in statuses:
        return CONDITION_FAIL
    if CONDITION_NOT_EXECUTED in statuses:
        return CONDITION_NOT_EXECUTED
    return CONDITION_PASS


def h1_conditions(
    *,
    window: Mapping[str, Any],
    row_domain: Mapping[str, Any],
    aa: Mapping[str, Any],
    arm_summaries: Mapping[str, Any],
    timings: Mapping[str, Any],
) -> Dict[str, Any]:
    """H1 机械门：≥10 个 structural 虚拟日 + 多臂 A/A 逐值一致 + 每臂每日恰一条估值。"""
    days = list(window.get("virtual_days") or [])
    conditions = [
        _condition(
            "H1-1",
            "至少 10 个连续 structural 历史虚拟日",
            CONDITION_PASS if len(days) >= 10 else CONDITION_FAIL,
            {
                "virtual_days": len(days),
                "d0": window.get("d0_date"),
                "last": days[-1] if days else None,
            },
        ),
        _condition(
            "H1-2",
            "逐日逐臂行域回执全为 structural",
            CONDITION_PASS if row_domain.get("verified") else CONDITION_FAIL,
            {
                "declared": row_domain.get("declared"),
                "verified": bool(row_domain.get("verified")),
                "days_with_receipts": len(row_domain.get("per_day") or {}),
            },
        ),
        _condition(
            "H1-3",
            "机械 A/A 日期/成交/现金/持仓/NAV 逐值一致（从分片复算）",
            CONDITION_PASS if aa.get("identical") else CONDITION_FAIL,
            {
                "compared_days": int(aa.get("compared_days", 0)),
                "compared_fields": int(aa.get("compared_fields", 0)),
                "mismatches": list(aa.get("mismatches") or [])[:5],
                "source": aa.get("source"),
            },
        ),
        _condition(
            "H1-4",
            "A/A 比对非真空（比对行数与字段数均 > 0）",
            (
                CONDITION_PASS
                if int(aa.get("compared_days", 0)) > 0 and int(aa.get("compared_fields", 0)) > 0
                else CONDITION_FAIL
            ),
            {
                "compared_days": int(aa.get("compared_days", 0)),
                "compared_fields": int(aa.get("compared_fields", 0)),
            },
        ),
        _condition(
            "H1-5",
            "每臂每日**恰一条**权威估值行（逐日 row_count==1，取合计之外的单日判据）",
            CONDITION_PASS if _valuations_complete(arm_summaries, len(days)) else CONDITION_FAIL,
            {
                arm: {
                    "driven_days": int(payload.get("driven_days", 0)),
                    "authoritative_values": int(payload.get("authoritative_values", 0)),
                    "complete_generations": int(payload.get("complete_generations", 0)),
                    "row_counts_all_one": bool(payload.get("row_counts_all_one", False)),
                    "row_counts_not_one": list(payload.get("row_counts_not_one") or []),
                }
                for arm, payload in sorted((arm_summaries or {}).items())
            },
        ),
        _condition(
            "H1-6",
            "生产零写入（data/paper、data/runs、data/models 递归快照不变）",
            CONDITION_PASS if timings.get("production_zero_write") else CONDITION_NOT_EXECUTED,
            {
                "production_zero_write": bool(timings.get("production_zero_write")),
                "fingerprints": dict(timings.get("production_fingerprints") or {}),
            },
        ),
    ]
    return {
        "gate": "H1",
        "conditions": conditions,
        "author_side_status": aggregate_status(conditions),
    }


def _valuations_complete(arm_summaries: Mapping[str, Any], day_count: int) -> bool:
    if not arm_summaries:
        return False
    for payload in arm_summaries.values():
        if int(payload.get("driven_days", 0)) != day_count:
            return False
        if int(payload.get("authoritative_values", 0)) != day_count:
            return False
        if int(payload.get("complete_generations", 0)) != day_count:
            return False
        if not bool(payload.get("row_counts_all_one", False)):
            return False
    return True


def h6_conditions(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """H6 端到端门：三条分别判定（不得用另一条抵扣）。"""
    names = [str(item.get("name")) for item in items]
    conditions: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        status = _item_status(str(item.get("status")))
        conditions.append(
            _condition(
                f"H6-{index + 1}",
                f"{names[index]}：{item.get('target')}",
                status,
                {
                    "arm_id": item.get("arm_id"),
                    "judgment": item.get("judgment"),
                    "target_pp": item.get("target_pp"),
                    "pure_effect_pp": item.get("pure_effect_pp"),
                    "d_anchor_pp": item.get("d_anchor_pp"),
                    "constructibility": item.get("constructibility"),
                    "constructibility_reason": item.get("constructibility_reason"),
                    "available_return_days": item.get("available_return_days"),
                    "required_return_days": item.get("required_return_days"),
                    "event": item.get("event"),
                    "paired_mean": item.get("paired_mean"),
                    "z_point_estimate": item.get("z_point_estimate"),
                    "proof_level": "点估计方向与链路（不含检出率/MDE）",
                },
            )
        )
    return {
        "gate": "H6",
        "conditions": conditions,
        "author_side_status": aggregate_status(conditions),
        "mutual_offset_forbidden": True,
        "frequency_source": "T4/T5 holdout（本演练不产生频率/MDE）",
    }


def _item_status(value: str) -> str:
    mapping = {
        ITEM_PASS: CONDITION_PASS,
        ITEM_PARTIAL: CONDITION_FAIL,
        ITEM_FAIL: CONDITION_FAIL,
        ITEM_NOT_EXECUTED: CONDITION_NOT_EXECUTED,
    }
    if value not in mapping:
        raise ValueError(f"未知 H6 条目状态: {value!r}")
    return mapping[value]


def h7_conditions(
    *,
    manifest: Mapping[str, Any],
    upstream: Mapping[str, Any],
    runbook: Mapping[str, Any],
    version: Mapping[str, Any],
) -> Dict[str, Any]:
    """H7 文档门：manifest 哈希/上游链接、运行手册、版本同步、限制声明、判定表边界。"""
    conditions = [
        _condition(
            "H7-1",
            "manifest 哈希链与上游链接完整（运行期逐字节校验）",
            CONDITION_PASS if upstream.get("verified") else CONDITION_FAIL,
            {
                "verified": bool(upstream.get("verified")),
                "links": dict(upstream.get("links") or {}),
            },
        ),
        _condition(
            "H7-2",
            "运行手册存在且命令/产物哈希与实际一致",
            CONDITION_PASS if runbook.get("verified") else CONDITION_NOT_EXECUTED,
            {
                "path": runbook.get("path"),
                "sha256": runbook.get("sha256"),
                "verified": bool(runbook.get("verified")),
                "mismatches": list(runbook.get("mismatches") or [])[:5],
            },
        ),
        _condition(
            "H7-3",
            "版本号与 CHANGELOG/README/pyproject 同步",
            CONDITION_PASS if version.get("synchronised") else CONDITION_FAIL,
            dict(version.get("evidence") or {}),
        ),
        _condition(
            "H7-4",
            "限制声明覆盖 §5.6 全部要点",
            CONDITION_PASS if int(manifest.get("limitation_count", 0)) >= 4 else CONDITION_FAIL,
            {"limitation_count": int(manifest.get("limitation_count", 0))},
        ),
        _condition(
            "H7-5",
            "判定表不含任何「闸门通过」字段",
            CONDITION_PASS if manifest.get("no_gate_pass_field") else CONDITION_FAIL,
            {
                "claim_fields": ["h1_claim", "h6_claim", "h7_claim"],
                "gate_owner": "user_at_merge_acceptance",
            },
        ),
    ]
    return {
        "gate": "H7",
        "conditions": conditions,
        "author_side_status": aggregate_status(conditions),
    }


def build_verdict(
    *,
    run_id: str,
    campaign_id: str,
    manifest_sha256: str,
    calibrator_sha256: str,
    artifact_hashes: Mapping[str, str],
    h1: Mapping[str, Any],
    h6: Mapping[str, Any],
    h7: Mapping[str, Any],
    effects: Mapping[str, RehearsalEffect],
    machine_time: Optional[Mapping[str, Any]] = None,
    authorization: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """组装判定表（固定 claim 字段 + 逐条作者侧检查；不含任何闸门结论）。"""
    return {
        "schema": VERDICT_SCHEMA,
        "scope": "historical_testbed",
        "evidence_class": "calibration_only",
        "virtual_clock": True,
        "not_forward_evidence": True,
        "not_sla_evidence": True,
        "run_id": str(run_id),
        "campaign_id": str(campaign_id),
        "hash_chain": {
            "manifest_sha256": str(manifest_sha256),
            "calibrator_sha256": str(calibrator_sha256),
            **{f"{name}_sha256": str(digest) for name, digest in sorted(artifact_hashes.items())},
        },
        "author_side_checks": {"h1": dict(h1), "h6": dict(h6), "h7": dict(h7)},
        "effects": {arm_id: effect.to_dict() for arm_id, effect in sorted(effects.items())},
        "h1_boundary": _boundary("H1"),
        "h6_boundary": _boundary("H6"),
        "h7_boundary": _boundary("H7"),
        "h1_claim": "evaluated_not_declared_pass",
        "h6_claim": "evaluated_not_declared_pass",
        "h7_claim": "evaluated_not_declared_pass",
        "gate_owner": "user_at_merge_acceptance",
        "machine_time": None if machine_time is None else dict(machine_time),
        "long_task_authorization": None if authorization is None else dict(authorization),
        "limitations": [
            "HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE：历史试车场演练，不是前向影子",
            "机械 A/A（同路径克隆）不得充当统计 A/A；统计 A/A 只来自 T4/T5 独立路径",
            "三条 H6 分别判定，任一失败或未执行不得用另一条抵扣",
            "频率/MDE 只引用 T4/T5 holdout 产物；本演练不产生任何频率读数",
            "闸门通过与生产裁决权由用户在合并验收时确认（gate_owner）",
        ],
    }


def _boundary(gate: str) -> Dict[str, Any]:
    return {
        "gate": gate,
        "claim": "evaluated_not_declared_pass",
        "gate_owner": "user_at_merge_acceptance",
        "note": "T6 只提供作者侧检查结果；闸门是否通过由用户判定",
    }
