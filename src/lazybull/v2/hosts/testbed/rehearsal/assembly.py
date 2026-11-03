# -*- coding: utf-8 -*-
"""P5b T6 事实汇集与报告落盘（**只读账本 + 只读上游**；产物独占创建）。

契约来源：设计 v4 §7.2（报告字段逐项）、§6.3（机械 A/A 与读数）、§8-H1/H6/H7。

事实口径
========
- 逐臂/逐日事实一律从**分片 + index + journal** 复算（T3 只读查询门），不用 runner 运行期字段
  替代；``day_digest`` 与 ``family_txn_id``/``parent_generation`` 也取自分片；
- 读数（配对收益、Z、D_k）见 :mod:`readings`；本模块只做组装与落盘；
- 频率类字段**只引用** T4/T5 冻结产物（不重算、不聚合）。

零写入：报告与判定表只落 ``campaign/reports/``；本模块不写任何生产根。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.layout import CampaignLayout
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import DIRECTIONS
from src.lazybull.v2.hosts.testbed.rehearsal import artifacts
from src.lazybull.v2.hosts.testbed.rehearsal.injection import (
    RehearsalEffect,
    arm_effect_plan,
)
from src.lazybull.v2.hosts.testbed.rehearsal.manifest import (
    MANIFEST_FILENAME,
    ManifestError,
    calibrator_identity,
    file_sha256,
    manifest_sha256,
    verify_chain,
)
from src.lazybull.v2.hosts.testbed.rehearsal.readings import (
    ITEM_NOT_EXECUTED,
    drawdown_look_reading,
    expected_income_shift,
    frozen_thresholds,
    h6_drawdown_item,
    h6_income_item,
    income_look_reading,
    paired_paths,
    pure_effect_at_anchor,
)
from src.lazybull.v2.hosts.testbed.rehearsal.report import build_report, render_markdown
from src.lazybull.v2.hosts.testbed.rehearsal.verdict import (
    build_verdict,
    h1_conditions,
    h6_conditions,
    h7_conditions,
)

__all__ = [
    "ArmFactsError",
    "build_readings",
    "collect_facts",
    "existing_artifact_hashes",
    "forbidden_gate_fields",
    "machine_time_summary",
    "model_availability",
    "production_state",
    "published_generation",
    "referenced_readings",
    "version_sync",
    "verify_runbook",
    "write_outputs",
]

#: 报告与判定表的必填产物（哈希链成员）
#: 运行手册必需章节关键词（H7 要件）
RUNBOOK_SECTIONS: Tuple[str, ...] = (
    "## 1",
    "## 2",
    "## 3",
    "## 4",
    "## 5",
    "## 6",
    "## 7",
    "## 8",
)
#: 运行手册「正式演练根」声明（正文必须与冻结件的 campaign 精确一致）
_FORMAL_RUNBOOK_DECLARATION = re.compile(r"正式演练根[^\n]*?(p5b_t6_rehearsal_[0-9]+(?:_r[0-9]+)?)")
#: 人工维护的产物哈希块（文档-产物一致性用例的锚）
_HASH_BLOCK = ("<!-- rehearsal-artifacts -->", "<!-- /rehearsal-artifacts -->")


class ArmFactsError(RuntimeError):
    """账本事实不可得（缺分片/缺 index/封存不符）。"""


def _shard(layout: CampaignLayout, arm_id: str, date: str) -> Dict[str, Any]:
    from src.lazybull.v2.hosts.testbed.journal import load_shard

    shard = load_shard(layout, arm_id, date)
    if shard is None:
        raise ArmFactsError(f"臂 {arm_id!r} 在 {date} 缺分片（报告拒绝在缺片状态下生成）")
    return shard


def collect_facts(
    layout: CampaignLayout, arms: Sequence[str]
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """从分片复算逐臂事实、行域回执与逐日事务清单。

    Returns:
        ``(arm_summaries, row_domain, transactions)``。
    """
    from src.lazybull.v2.hosts.testbed.journal import read_index

    index = read_index(layout)
    if index is None:
        raise ArmFactsError(f"campaign {layout.root.name!r} 缺已发布 index")
    days = [
        str(entry["virtual_date"])
        for entry in sorted(index.get("days") or [], key=lambda item: int(item["generation"]))
    ]
    summaries: Dict[str, Any] = {}
    row_domain: Dict[str, Any] = {}
    transactions: List[Dict[str, Any]] = []
    for date in days:
        entry = next(item for item in index["days"] if str(item["virtual_date"]) == date)
        arm_payload: Dict[str, Any] = {}
        family: Optional[str] = None
        for arm_id in sorted(arms):
            shard = _shard(layout, arm_id, date)
            evidence = dict(shard.get("evidence") or {})
            family = family or str(shard["family_txn_id"])
            arm_payload[arm_id] = {
                "driven": bool(evidence.get("driven")),
                "driven_date": evidence.get("driven_date"),
                "authoritative_values": int((evidence.get("valuation") or {}).get("row_count", 0)),
                "complete_generation": bool(evidence.get("complete_generation")),
                "valuation_source": (evidence.get("valuation") or {}).get("source"),
                "nav": float((evidence.get("valuation") or {}).get("nav", float("nan"))),
                "day_digest": evidence.get("day_digest"),
                "receipt_passed": bool((evidence.get("row_domain_receipt") or {}).get("passed")),
                "actual_row_domains": list(
                    (evidence.get("row_domain_receipt") or {}).get("actual_row_domains") or []
                ),
                "fills_count": len(evidence.get("fills") or []),
                "positions_count": len(evidence.get("positions") or {}),
                "root_identity": dict(shard.get("root_identity") or {}),
            }
            _append_arm(summaries, arm_id, date, arm_payload[arm_id])
        row_domain[date] = arm_payload
        transactions.append(
            {
                "virtual_date": date,
                "generation": int(entry["generation"]),
                "family_txn_id": family,
                "published_txn_id": entry.get("family_txn_id"),
                "parent_generation": int(entry["generation"]) - 1,
                "arms": arm_payload,
            }
        )
    verified = all(
        payload["receipt_passed"] and set(payload["actual_row_domains"]) == {"structural"}
        for day in row_domain.values()
        for payload in day.values()
    )
    return (
        summaries,
        {
            "declared": "structural",
            "verified": bool(verified and row_domain),
            "per_day": row_domain,
            "day_count": len(row_domain),
        },
        transactions,
    )


def _append_arm(
    summaries: Dict[str, Any], arm_id: str, date: str, payload: Mapping[str, Any]
) -> None:
    entry = summaries.setdefault(
        arm_id,
        {
            "driven_days": 0,
            "dates": [],
            "complete_generations": 0,
            "authoritative_values": 0,
            "row_counts_all_one": True,
            "row_counts_not_one": [],
            "last_nav": None,
            "fills_total": 0,
            "day_digests": [],
            "root_identities": [],
        },
    )
    entry["driven_days"] = int(entry["driven_days"]) + (1 if payload.get("driven") else 0)
    entry["dates"].append(date)
    entry["complete_generations"] = int(entry["complete_generations"]) + (
        1 if payload.get("complete_generation") else 0
    )
    entry["authoritative_values"] = int(entry["authoritative_values"]) + int(
        payload.get("authoritative_values", 0)
    )
    if int(payload.get("authoritative_values", 0)) != 1:
        entry["row_counts_all_one"] = False
        entry["row_counts_not_one"].append(
            {"virtual_date": date, "row_count": int(payload.get("authoritative_values", 0))}
        )
    entry["last_nav"] = payload.get("nav")
    entry["fills_total"] = int(entry["fills_total"]) + int(payload.get("fills_count", 0))
    entry["day_digests"].append(payload.get("day_digest"))
    entry["root_identities"].append(dict(payload.get("root_identity") or {}))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_readings(
    layout: CampaignLayout,
    document: Mapping[str, Any],
    *,
    repo_root: Path,
    start: Optional[str] = None,
) -> Dict[str, Any]:
    """从账本构造读数与 H6 三条（只读；阈值只来自 T4/T5 冻结产物）。

    样本不足（已发布估值点 < 2）或区间失效时，**不伪造读数**：该条以 ``not_executed``
    如实结束并记录失败原因（fail-closed；不得降级为 0 或 pass）。
    """
    t4_thresholds = _read_json(Path(repo_root) / str(document["upstream"]["t4_thresholds"]["path"]))
    t5_thresholds = _read_json(Path(repo_root) / str(document["upstream"]["t5_thresholds"]["path"]))
    thresholds = frozen_thresholds(t4_thresholds, t5_thresholds)
    effects = arm_effect_plan(document, int(document["window"]["return_days"]))
    base_paths, base_error = _safe_pair(layout, "base", "mech_aa", start)
    income = _income_block(layout, effects, thresholds, start)
    drawdown_readings: Dict[str, Any] = {}
    items: List[Dict[str, Any]] = [income["item"]]
    for arm_id, direction in zip(
        ("dd_improve_12pp", "dd_worsen_12pp"), (DIRECTIONS[1], DIRECTIONS[0])
    ):
        drawdown_readings[arm_id], item = _drawdown_block(
            layout, document, effects, thresholds, arm_id, direction, start
        )
        items.append(item)
    return {
        "thresholds": thresholds,
        "effects": effects,
        "income": income["reading"],
        "income_item": income["item"],
        "drawdown": drawdown_readings,
        "h6_items": items,
        "aa_paths_identity": {
            "base_vs_clone_sample_days": (
                0 if base_paths is None else int(len(base_paths["return_dates"]))
            ),
            "base_vs_clone_paired_mean": (
                0.0 if base_paths is None else _mean(base_paths["pair"]["returns"])
            ),
            "base_vs_clone_read_error": base_error,
        },
    }


def _safe_pair(
    layout: CampaignLayout, candidate: str, base: str, start: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """配对读取（失败即返回原因，不吞异常类别）。"""
    from src.lazybull.v2.hosts.testbed.paths import TestbedError

    try:
        return paired_paths(layout, candidate, base, start=start), None
    except TestbedError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _income_block(
    layout: CampaignLayout,
    effects: Mapping[str, RehearsalEffect],
    thresholds: Mapping[str, Any],
    start: Optional[str],
) -> Dict[str, Any]:
    """收益臂读数块（样本不足 ⇒ ``not_executed``）。"""
    paths, error = _safe_pair(layout, "income_minus_5d", "base", start)
    effect = effects["income_minus_5d"]
    if paths is None:
        return {
            "reading": {
                "sample_days": 0,
                "paired_mean": None,
                "rows": [],
                "executable_looks": [],
                "identifiable_looks": [],
                "read_error": error,
            },
            "item": _unavailable_income_item(effect, error),
        }
    reading = income_look_reading(np.asarray(paths["pair"]["returns"], dtype=float), thresholds)
    return {
        "reading": reading,
        "item": h6_income_item(effect, reading, expected_income_shift(effect)),
    }


def _unavailable_income_item(effect: RehearsalEffect, error: Optional[str]) -> Dict[str, Any]:
    """配对样本不可得时的收益臂条目（不得写成 pass）。"""
    return {
        "name": "income_minus_5d",
        "arm_id": effect.arm_id,
        "target": "配对日度 log 差均值 = −5δ",
        "target_value": None,
        "paired_mean": None,
        "max_abs_residual": None,
        "direction_ok": False,
        "paired_sample_days": 0,
        "looks": [],
        "executable_looks": [],
        "identifiable_looks": [],
        "z_point_estimate": None,
        "z_identifiable": False,
        "status": ITEM_NOT_EXECUTED,
        "judgment": f"配对账本样本不可得（{error}）；未执行，不得记为通过",
        "evidence_fields": ["h6.items[0].paired_mean", "h6.items[0].z_point_estimate"],
    }


def _drawdown_block(
    layout: CampaignLayout,
    document: Mapping[str, Any],
    effects: Mapping[str, RehearsalEffect],
    thresholds: Mapping[str, Any],
    arm_id: str,
    direction: str,
    start: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """回撤臂读数块（不可构造或样本不足 ⇒ ``not_executed``）。"""
    effect = effects[arm_id]
    paths, error = _safe_pair(layout, arm_id, "base", start)
    if paths is None:
        return (
            {"rows": [], "executable_looks": [], "read_error": error},
            h6_drawdown_item(effect, None, direction=direction, pure_effect_pp=None),
        )
    reading = drawdown_look_reading(
        np.asarray(paths["returns_candidate"], dtype=float),
        np.asarray(paths["returns_base"], dtype=float),
        thresholds,
    )
    pure = pure_effect_at_anchor(
        np.asarray(paths["returns_base"], dtype=float),
        effect,
        list(document["injection_window"]["frozen_list"]),
    )
    block = {
        **reading,
        "pair_identity": {
            "published_generation": paths["published_generation"],
            "chain_digests": paths["chain_digests"],
            "nav_days": int(len(paths["dates"])),
            "return_samples": int(len(paths["return_dates"])),
        },
    }
    return block, h6_drawdown_item(effect, reading, direction=direction, pure_effect_pp=pure)


def _mean(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()) if array.size else 0.0


def version_sync(repo_root: Path) -> Dict[str, Any]:
    """版本号同步（pyproject ↔ CHANGELOG 首条 ↔ README 手册入口）。"""
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    head = re.search(r"(?m)^##\s*\[?([0-9]+\.[0-9]+\.[0-9]+)", changelog)
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    payload = {
        "pyproject": None if version is None else version.group(1),
        "changelog": None if head is None else head.group(1),
        "readme_runbook_entry": "p5b_testbed_runbook" in readme,
    }
    payload["synchronised"] = bool(
        payload["pyproject"]
        and payload["pyproject"] == payload["changelog"]
        and payload["readme_runbook_entry"]
    )
    payload["evidence"] = {
        "pyproject_version": payload["pyproject"],
        "changelog_head": payload["changelog"],
        "readme_runbook_entry": payload["readme_runbook_entry"],
        "readme_path": "README.md",
        "runbook_path": "docs/guide/p5b_testbed_runbook.md",
    }
    return payload


def _runbook_identity_mismatches(
    text: str, campaign_id: Optional[str], lineage_sequence: Optional[int]
) -> List[str]:
    """手册正文与冻结件身份的机器校验（campaign / 被取代 run / sequence）。"""
    mismatches: List[str] = []
    if campaign_id:
        declared = _FORMAL_RUNBOOK_DECLARATION.findall(text)
        if declared and set(declared) != {campaign_id}:
            mismatches.append(f"formal_campaign_declared={sorted(set(declared))}!={campaign_id}")
        elif not declared and campaign_id not in text:
            mismatches.append(f"campaign_id_not_in_runbook={campaign_id}")
    if lineage_sequence is not None:
        declared = {int(value) for value in re.findall(r"lineage\.sequence=(\d+)", text)}
        if declared != {int(lineage_sequence)}:
            mismatches.append(
                f"lineage_sequence_mismatch=runbook{sorted(declared)}!=manifest{lineage_sequence}"
            )
    return mismatches


def verify_runbook(
    repo_root: Path,
    *,
    artifact_hashes: Mapping[str, str],
    campaign_id: Optional[str] = None,
    lineage_sequence: Optional[int] = None,
) -> Dict[str, Any]:
    """运行手册要件与产物一致性（文档-产物一致性用例的唯一实现）。

    除章节/产物名/哈希块外，还校验**正文**与冻结件的 campaign/sequence 一致
    （防「哈希块对、正文指向被取代 run」）。
    """
    path = repo_root / "docs" / "guide" / "p5b_testbed_runbook.md"
    if not path.is_file():
        return {"path": path.as_posix(), "verified": False, "mismatches": ["runbook_missing"]}
    text = path.read_text(encoding="utf-8")
    mismatches: List[str] = _runbook_identity_mismatches(text, campaign_id, lineage_sequence)
    missing = [marker for marker in RUNBOOK_SECTIONS if marker not in text]
    if missing:
        mismatches.append(f"missing_sections={missing}")
    for name in artifacts.ARTIFACT_NAMES:
        if name not in text:
            mismatches.append(f"artifact_not_named={name}")
    recorded = _hash_block(text)
    if recorded is None:
        mismatches.append("hash_block_missing")
    else:
        for name, digest in sorted(recorded.items()):
            if name not in artifact_hashes:
                mismatches.append(f"unknown_artifact={name}")
            elif artifact_hashes[name] != digest:
                mismatches.append(f"hash_mismatch={name}")
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "verified": not mismatches,
        "mismatches": mismatches,
    }


def _hash_block(text: str) -> Optional[Dict[str, str]]:
    start, stop = _HASH_BLOCK
    if start not in text or stop not in text:
        return None
    body = text.split(start, 1)[1].split(stop, 1)[0]
    recorded: Dict[str, str] = {}
    for line in body.splitlines():
        match = re.match(r"\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|", line.strip())
        if match:
            recorded[match.group(1)] = match.group(2)
    return recorded or None


def write_outputs(
    *,
    layout: CampaignLayout,
    document: Mapping[str, Any],
    repo_root: Path,
    facts: Mapping[str, Any],
    aa: Mapping[str, Any],
    readings: Mapping[str, Any],
    machine_time: Mapping[str, Any],
    authorization: Optional[Mapping[str, Any]],
    production: Mapping[str, Any],
    referenced: Mapping[str, Any],
    runbook: Mapping[str, Any],
    upstream_check: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """写报告 + 判定表（先报告后判定表；判定表记录报告的哈希 ⇒ 三环闭合）。

    ``upstream_check`` 必须是**本次调用**真实核验上游五链的结果（缺省 ⇒ H7-1 判未执行），
    禁止再用常量 ``{"verified": True}`` 冒充。
    """
    manifest_path = layout.root / "reports" / MANIFEST_FILENAME
    manifest_sha = manifest_sha256(manifest_path)
    identity = calibrator_identity()
    report_doc = build_report(
        campaign_id=str(layout.root.name),
        run_id=str(document["run_id"]),
        manifest=document,
        manifest_sha256=manifest_sha,
        calibrator=identity,
        upstream=dict(document["upstream"]),
        run_report=facts,
        aa=aa,
        h6_items=readings["h6_items"],
        income_reading=readings["income"],
        drawdown_readings=readings["drawdown"],
        thresholds=readings["thresholds"],
        effects={arm: effect.to_dict() for arm, effect in readings["effects"].items()},
        artifact_hashes={artifacts.ARTIFACT_MANIFEST: manifest_sha},
        machine_time=machine_time,
        authorization=authorization,
        production=production,
        referenced=referenced,
        t4_reference={
            "path": document["upstream"]["t4_thresholds"]["path"],
            "sha256": document["upstream"]["t4_thresholds"]["sha256"],
            "c_k": list(readings["thresholds"]["c_k"]),
        },
        t5_reference={
            "path": document["upstream"]["t5_thresholds"]["path"],
            "sha256": document["upstream"]["t5_thresholds"]["sha256"],
            "c_w": list(readings["thresholds"]["c_w"]),
            "c_i": list(readings["thresholds"]["c_i"]),
        },
    )
    report_written = artifacts.write_json_once(
        artifacts.artifact_path(layout.root, artifacts.ARTIFACT_REPORT),
        report_doc,
        what="T6 演练报告",
    )
    markdown_written = artifacts.write_text_once(
        artifacts.artifact_path(layout.root, artifacts.ARTIFACT_REPORT_MD),
        render_markdown(
            {
                **report_doc,
                "artifact_hashes": {
                    artifacts.ARTIFACT_MANIFEST: manifest_sha,
                    artifacts.ARTIFACT_REPORT: report_written["sha256"],
                },
            }
        ),
        what="T6 演练报告（markdown）",
    )
    hashes = {
        artifacts.ARTIFACT_REPORT: report_written["sha256"],
        artifacts.ARTIFACT_REPORT_MD: markdown_written["sha256"],
    }
    chain = verify_chain(
        manifest_doc=document,
        manifest_file_sha256=manifest_sha,
        calibrator=identity,
        artifact_hashes=hashes,
    )
    verdict = build_verdict(
        run_id=str(document["run_id"]),
        campaign_id=str(layout.root.name),
        manifest_sha256=manifest_sha,
        calibrator_sha256=identity["sha256"],
        artifact_hashes=hashes,
        h1=h1_conditions(
            window=document["window"],
            row_domain=facts["row_domain"],
            aa=aa,
            arm_summaries=facts["arms"],
            timings={
                "production_zero_write": bool(production.get("unchanged")),
                "production_fingerprints": dict(production.get("after") or {}),
                "production_window": str(production.get("window") or ""),
            },
        ),
        h6=h6_conditions(readings["h6_items"]),
        h7=h7_conditions(
            manifest={
                "limitation_count": len(document["limitations"]),
                "no_gate_pass_field": not forbidden_gate_fields(
                    {"manifest": document, "facts": facts, "readings": readings}
                ),
                "gate_scan_scope": "inputs(manifest/facts/readings)；产物侧由写盘前扫描强制",
            },
            upstream=dict(upstream_check or {"verified": False, "reason": "report 阶段未核验"}),
            runbook=dict(runbook),
            version=version_sync(Path(repo_root)),
        ),
        effects=readings["effects"],
        machine_time=machine_time,
        authorization=authorization,
    )
    blocked = forbidden_gate_fields(verdict)
    if blocked:
        raise ValueError(f"判定表出现禁止的闸门字段（拒绝写盘）: {blocked}")
    verdict["hash_chain"].update(chain)
    verdict_written = artifacts.write_json_once(
        artifacts.artifact_path(layout.root, artifacts.ARTIFACT_VERDICT),
        verdict,
        what="T6 判定表",
    )
    return {
        "report": report_written,
        "report_md": markdown_written,
        "verdict": verdict_written,
        "author_side_status": {
            "h1": verdict["author_side_checks"]["h1"]["author_side_status"],
            "h6": verdict["author_side_checks"]["h6"]["author_side_status"],
            "h7": verdict["author_side_checks"]["h7"]["author_side_status"],
        },
        "h6_items": [
            {
                "name": item["name"],
                "status": item["status"],
                "judgment": item["judgment"],
            }
            for item in readings["h6_items"]
        ],
    }


def machine_time_summary(campaign_root: Path) -> Dict[str, Any]:
    """汇总运行阶段墙钟（**逐批**；非身份产物；缺文件即返空摘要）。"""
    path = artifacts.artifact_path(Path(campaign_root), artifacts.ARTIFACT_MACHINE_TIME_RUN)
    if not path.is_file():
        return {
            "virtual_days_total": 0,
            "batch_seconds_total": None,
            "batches": [],
            "per_day_seconds": None,
        }
    payload = _read_json(path)
    history = payload.get("history") or [payload]
    batches: List[Dict[str, Any]] = []
    total_seconds = 0.0
    total_days = 0
    for entry in history:
        days = [str(day) for day in (entry.get("days") or [])]
        per_day = dict(entry.get("per_day_seconds") or {})
        count = len(days) or len(per_day)
        seconds = float(entry.get("batch_seconds") or 0.0)
        batches.append(
            {
                "days": count,
                "batch_seconds": round(seconds, 6),
                "per_day_mean": round(seconds / count, 6) if count else None,
                "stage": entry.get("stage"),
                "authorization_sha256": entry.get("authorization_sha256"),
            }
        )
        total_seconds += seconds
        total_days += count
    return {
        "virtual_days_total": total_days,
        "batch_seconds_total": round(total_seconds, 6),
        "per_day_seconds_total": round(total_seconds / total_days, 6) if total_days else None,
        "batches": batches,
        "per_day_seconds": {
            key: value
            for entry in history
            for key, value in (entry.get("per_day_seconds") or {}).items()
        },
        "note": "墙钟不属于产物身份；同 seed 重跑确定性产物逐字节一致",
    }


def machine_time_reference(
    campaign_root: Path, authorization: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """报告/判定表内嵌的**确定性**墙钟摘要（不含任何耗时数值，避免破坏逐字节可复现）。"""
    summary = machine_time_summary(campaign_root)
    return {
        "artifact": artifacts.ARTIFACT_MACHINE_TIME_RUN,
        "virtual_days_total": int(summary["virtual_days_total"]),
        "batches": [
            {"days": int(batch["days"]), "stage": batch["stage"]} for batch in summary["batches"]
        ],
        "note": "逐日/逐批墙钟只写 machine_time*.json（不入产物身份）；本字段只保留批次数与日数",
        "authorization": dict(authorization) if authorization else None,
    }


def production_state(
    repo_root: Path, before: Optional[Mapping[str, Any]], *, window: str = ""
) -> Dict[str, Any]:
    """生产根零写入比对（``before`` 取自 register 阶段快照，覆盖 register → report）。

    内嵌快照**剔除绝对时刻**（``max_mtime_ns``）⇒ 封存产物不含墙钟类数值；比对仍是
    ``files + bytes + path_digest``（digest 含 mtime，但在演练期间生产树不动 ⇒ 可复现）。
    ``before`` 缺失时 ``unchanged=None``（⇒ H1-6 判未执行，而不是零宽度窗口假通过）。
    """
    from src.lazybull.v2.hosts.testbed.rehearsal.sources import collect_production_fingerprints

    after = collect_production_fingerprints(repo_root)

    def _clean(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            name: {
                "files": int(payload.get("files", 0)),
                "bytes": int(payload.get("bytes", 0)),
                "path_digest": str(payload.get("path_digest") or ""),
            }
            for name, payload in dict(snapshot or {}).items()
        }

    if before is None:
        return {
            "before": None,
            "after": _clean(after),
            "unchanged": None,
            "window": window or "report-only（缺 register 阶段快照）",
            "method": "files + bytes + path_digest（data/paper、data/runs、data/models）",
        }
    return {
        "before": _clean(before),
        "after": _clean(after),
        "unchanged": bool(_clean(before) == _clean(after)),
        "window": window or "register → report（跨阶段）",
        "method": "files + bytes + path_digest（data/paper、data/runs、data/models）",
    }


def referenced_readings(repo_root: Path, document: Mapping[str, Any]) -> Dict[str, Any]:
    """T4/T5 冻结读数引用（**只引用不重算**；本演练不产生频率/MDE）。

    引用文件必须来自本 manifest 的**冻结上游**，并按期望哈希校验（R2-P5B-06）。
    """
    root = Path(repo_root)
    upstream = dict(document.get("upstream") or {})
    t4_path = root / str(upstream["t4_holdout"]["path"])
    t5_path = root / str(upstream["t5_power"]["path"])
    for label, path, meta in (
        ("income_holdout", t4_path, upstream["t4_holdout"]),
        ("drawdown_power", t5_path, upstream["t5_power"]),
    ):
        if not path.is_file():
            raise ManifestError(f"引用读数缺文件（{label}）: {path}")
        if file_sha256(path) != str(meta["sha256"]):
            raise ManifestError(f"引用读数与冻结上游哈希不符（{label}）: {path}")
    t4 = _read_json(t4_path)
    t5 = _read_json(t5_path)
    return {
        "income_holdout": {
            "unit_count": int(t4.get("unit_count", 0)),
            "referee_usable": bool(t4.get("referee_usable")),
            "wilson_upper_max": t4.get("wilson_upper_max"),
        },
        "drawdown_mde": {
            "MDE_drawdown_improve": _numeric_only(
                (t5.get("mde") or {}).get("MDE_drawdown_improve")
            ),
            "MDE_drawdown_worsen": _numeric_only((t5.get("mde") or {}).get("MDE_drawdown_worsen")),
        },
        "provenance": {
            "t4_holdout": t4_path.relative_to(root).as_posix(),
            "t5_power": t5_path.relative_to(root).as_posix(),
            "t4_holdout_sha256": str(upstream["t4_holdout"]["sha256"]),
            "t5_power_sha256": str(upstream["t5_power"]["sha256"]),
            "source": "resolved_from_manifest_upstream",
            "note": "只引用数值字段；T5 的闸门类字段（如 aa_gate_passed/gate_scope）不进入本报告",
        },
        "note": "频率/MDE 只来自本 manifest 冻结上游的 T4/T5 产物；本演练只给点估计方向与链路",
        "no_frequency_produced_here": True,
    }


def _numeric_only(payload: Any) -> Dict[str, Any]:
    """只保留数值/布尔读数（丢弃闸门语义字段，避免被误读为本演练的闸门结论）。"""
    if not isinstance(payload, Mapping):
        return {}
    kept: Dict[str, Any] = {}
    for key, value in payload.items():
        lowered = str(key).strip().lower()
        if lowered in FORBIDDEN_GATE_EXACT or lowered.endswith(FORBIDDEN_GATE_SUFFIX):
            continue
        if lowered in {"gate_scope", "gate_status", "gate_result"}:
            continue
        if isinstance(value, (int, float, bool)) or value is None:
            kept[str(key)] = value
    return kept


def model_availability(campaign_root: Path, arms: Sequence[str]) -> Dict[str, Any]:
    """模型可得性（训练截止严格早于 D0；取自已注册臂身份）。"""
    from src.lazybull.v2.hosts.testbed.registration import load_arm

    payload: Dict[str, Any] = {}
    for arm_id in arms:
        record = load_arm(campaign_root, arm_id)
        model = (record.get("identity") or {}).get("model") or {}
        payload[arm_id] = {
            "model_version": model.get("version"),
            "train_end_date": model.get("train_end_date"),
            "model_content_fingerprint": model.get("content_fingerprint"),
            "trained_before_d0": bool(model.get("trained_before_d0")),
        }
    return payload


def published_generation(layout: CampaignLayout) -> int:
    """已发布代（index 唯一权威）。"""
    from src.lazybull.v2.hosts.testbed.journal import read_index

    index = read_index(layout)
    if index is None:
        raise ArmFactsError(f"campaign {layout.root.name!r} 缺已发布 index")
    return int(index["published_generation"])


def existing_artifact_hashes(campaign_root: Path) -> Dict[str, str]:
    """已有确定性产物的哈希（供运行手册一致性用例）。"""
    root = Path(campaign_root)
    hashes: Dict[str, str] = {}
    for name in artifacts.ARTIFACT_NAMES:
        path = artifacts.artifact_path(root, name)
        if path.is_file():
            hashes[name] = file_sha256(path)
    return hashes


def doc_consistency_hashes(campaign_root: Path, document: Mapping[str, Any]) -> Dict[str, str]:
    """运行手册哈希块的**稳定成员**（冻结件 + 上游链接；报告产物不进入该块）。

    这样手册可以在 ``--report`` 之前写成最终形态：报告/判定表的哈希记录在报告本体与
    判定表的 ``hash_chain`` 里（三环互链），不依赖「手册先知道报告哈希」的循环。
    """
    hashes = {
        artifacts.ARTIFACT_MANIFEST: file_sha256(
            Path(campaign_root) / "reports" / MANIFEST_FILENAME
        )
    }
    for key, payload in sorted(dict(document["upstream"]).items()):
        if isinstance(payload, Mapping) and payload.get("sha256"):
            hashes[key] = str(payload["sha256"])
    return hashes


FORBIDDEN_GATE_EXACT = frozenset(
    {
        "gate_passed",
        "gate_pass",
        "gates_passed",
        "all_gates_passed",
        "h1_passed",
        "h6_passed",
        "h7_passed",
        "gate_result",
        "gate_verdict",
    }
)
FORBIDDEN_GATE_SUFFIX = ("_gate_passed", "_gates_passed", "_gate_pass")


def forbidden_gate_fields(payload: Any, *, prefix: str = "") -> List[str]:
    """递归扫描「闸门通过」类字段（判定表与报告均不得含任何此类键）。"""
    forbidden: List[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).strip().lower()
            if lowered in FORBIDDEN_GATE_EXACT or lowered.endswith(FORBIDDEN_GATE_SUFFIX):
                forbidden.append(name)
            forbidden.extend(forbidden_gate_fields(value, prefix=name))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            forbidden.extend(forbidden_gate_fields(item, prefix=f"{prefix}[{index}]"))
    return forbidden
