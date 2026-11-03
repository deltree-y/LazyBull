# -*- coding: utf-8 -*-
"""P5b T6 历史端到端演练薄入口（--freeze / --register / --run / --report / --check）。

用法（完整序列见 ``docs/guide/p5b_testbed_runbook.md``）::

    python scripts/v2_p5b/rehearsal.py --freeze --campaign-id <id> --apply
    python scripts/v2_p5b/rehearsal.py --register --campaign-id <id>
    python scripts/v2_p5b/rehearsal.py --run --campaign-id <id> --days 1 --json
    python scripts/v2_p5b/rehearsal.py --run --campaign-id <id> --authorization <file>
    python scripts/v2_p5b/rehearsal.py --report --campaign-id <id> --apply
    python scripts/v2_p5b/rehearsal.py --check --campaign-id <id>

默认 dry-run / 只读：``--freeze`` 与 ``--report`` 需显式 ``--apply`` 才写盘；``--register`` 与
``--run`` 属运行期动作（会写 campaign 根）；``--check`` 只读。

退出码：0 = 执行成功（``--report``/``--check`` 下指作者侧检查全部 pass）；3 = 作者侧检查存在
fail 或 not_executed；4 = 单批预计超过 20 分钟且缺少用户授权（设计 §9）；2 = 用法错误；
1 = 校验/写域/前置错误。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.hosts.testbed.journal import published_days, read_index  # noqa: E402
from src.lazybull.v2.hosts.testbed.paths import assert_write_scope  # noqa: E402
from src.lazybull.v2.hosts.testbed.registration import list_arms, load_campaign  # noqa: E402
from src.lazybull.v2.hosts.testbed.rehearsal import artifacts  # noqa: E402
from src.lazybull.v2.hosts.testbed.rehearsal.aa import mechanical_aa  # noqa: E402
from src.lazybull.v2.hosts.testbed.rehearsal.assembly import (  # noqa: E402
    build_readings,
    collect_facts,
    doc_consistency_hashes,
    forbidden_gate_fields,
    machine_time_reference,
    model_availability,
    production_state,
    published_generation,
    referenced_readings,
    verify_runbook,
    write_outputs,
)
from src.lazybull.v2.hosts.testbed.rehearsal.manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    load_manifest,
    manifest_sha256,
    verify_upstream,
)
from src.lazybull.v2.hosts.testbed.rehearsal.runner import (  # noqa: E402
    LongTaskAuthorizationRequired,
    campaign_root_for,
    freeze_rehearsal,
    load_authorization,
    register_rehearsal,
    rehearsal_layout,
    run_rehearsal,
    verify_rehearsal_window,
)
from src.lazybull.v2.hosts.testbed.rehearsal.sections import lineage_section  # noqa: E402
from src.lazybull.v2.hosts.testbed.rehearsal.sources import (  # noqa: E402
    require_repo_manifest,
)

#: 默认演练 campaign 标识（新 campaign；不得复用 T4/T5 目录）
DEFAULT_CAMPAIGN_ID = "p5b_t6_rehearsal_20261102"
#: 默认演练 run_id
DEFAULT_RUN_ID = "p5b_t6_rehearsal_20261102"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--freeze", action="store_true", help="冻结 T6 manifest（默认 dry-run）")
    stage.add_argument("--register", action="store_true", help="登记 campaign + 五臂 + D0 fresh")
    stage.add_argument("--run", action="store_true", help="逐日推进虚拟时钟（宿主层注入）")
    stage.add_argument("--report", action="store_true", help="从账本出读数、报告与判定表")
    stage.add_argument("--check", action="store_true", help="只读校验（三环/上游/A-A/边界）")
    parser.add_argument("--campaign-id", default=DEFAULT_CAMPAIGN_ID, help="演练 campaign 标识")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="演练 run_id（写入 manifest）")
    parser.add_argument("--data-root", default="", help="隔离数据根（缺省仓库 data/）")
    parser.add_argument("--repo-root", default="", help="仓库根（仅测试用隔离假仓库）")
    parser.add_argument("--days", type=int, default=0, help="本次推进的虚拟日数（0=全部剩余）")
    parser.add_argument("--apply", action="store_true", help="实际写盘（--freeze/--report）")
    parser.add_argument("--authorization", default="", help=">20 分钟长任务授权记录 JSON")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出摘要")
    parser.add_argument("--manifest-suffix", default="", help="仅重冻：冻结件文件名后缀（如 _r2）")
    parser.add_argument(
        "--lineage-sequence", type=int, default=1, help="仅重冻：谱系序号（首次=1）"
    )
    parser.add_argument("--supersede-manifest-sha", default="", help="重冻：前身 manifest sha256")
    parser.add_argument("--supersede-calibrator-sha", default="", help="重冻：前身校准器 sha256")
    parser.add_argument("--supersede-reason", default="", help="重冻：重冻原因")
    parser.add_argument(
        "--allow-external-root",
        action="store_true",
        help="仅测试/临时脚本：放开写域白名单（不解除生产根禁令）",
    )
    return parser.parse_args(argv)


def _print(payload: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    for key in sorted(payload):
        print(f"{key}={payload[key]}")


def _selection(campaign_id: str, count: int) -> Optional[List[str]]:
    """``--days N`` ⇒ 从最早未发布日起取 N 日（1 日即墙钟探针）。"""
    if count <= 0:
        return None
    root = campaign_root_for(campaign_id)
    manifest = json.loads((root / "reports" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    published = published_days(read_index(rehearsal_layout(_REPO_ROOT, campaign_id)))
    pending = [str(day) for day in manifest["window"]["virtual_days"] if str(day) not in published]
    return pending[:count]


def _lineage(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """重冻谱系段（``--lineage-sequence 1`` 时返回 None，交默认段生成）。"""
    sequence = int(getattr(args, "lineage_sequence", 1) or 1)
    if sequence <= 1:
        return None
    return lineage_section(
        sequence=sequence,
        superseded_manifest_sha256=args.supersede_manifest_sha or None,
        superseded_calibrator_sha256=args.supersede_calibrator_sha or None,
        reason=args.supersede_reason or None,
    )


def _stage_freeze(args: argparse.Namespace) -> Dict[str, Any]:
    outcome = freeze_rehearsal(
        repo_root=_REPO_ROOT,
        campaign_id=args.campaign_id,
        run_id=args.run_id,
        data_root=args.data_root or None,
        apply=args.apply,
        suffix=args.manifest_suffix,
        lineage=_lineage(args),
    )
    outcome.pop("doc", None)
    return outcome


def _stage_run(args: argparse.Namespace) -> Dict[str, Any]:
    document, _ = load_manifest(
        campaign_root_for(args.campaign_id) / "reports" / MANIFEST_FILENAME,
        repo_manifest=require_repo_manifest(_REPO_ROOT),
    )
    verify_upstream(document, repo_root=_REPO_ROOT)
    return run_rehearsal(
        repo_root=_REPO_ROOT,
        campaign_id=args.campaign_id,
        days=_selection(args.campaign_id, args.days),
        authorization=load_authorization(args.authorization or None),
        allow_external_root=args.allow_external_root,
    )


def _report_facts(campaign_root: Path, document: Dict[str, Any]) -> Dict[str, Any]:
    layout = rehearsal_layout(_REPO_ROOT, campaign_root.name)
    arms = sorted(list_arms(campaign_root))
    arm_facts, row_domain, transactions = collect_facts(layout, arms)
    campaign = load_campaign(campaign_root)
    identity = campaign.get("identity") or {}
    return {
        "arms": arm_facts,
        "row_domain": row_domain,
        "transactions": transactions,
        "identity": {
            "manifest_sha256": manifest_sha256(campaign_root / "reports" / MANIFEST_FILENAME),
            "campaign_identity_summary": dict(campaign.get("identity_summary") or {}),
            "window_state_id": (identity.get("window_state") or {}).get("window_state_id"),
            "execution_digest": (identity.get("code") or {}).get("execution_digest"),
            "global_watermark": (identity.get("window_state") or {}).get("global_watermark"),
        },
        "label_cutoff": {
            "value": None,
            "status": "not_consumed_by_mechanical_layer",
            "reason": "机械层不消费标签；标签截止属统计层登记（设计 §7.2）",
        },
        "model_availability": model_availability(campaign_root, arms),
        "published": {"published_generation": published_generation(layout)},
        "recovery": {"status": "no_recovery_needed_in_report_phase"},
        "fault_injections": [],
    }


def _stage_report(args: argparse.Namespace) -> Dict[str, Any]:
    campaign_root = campaign_root_for(args.campaign_id)
    assert_write_scope(campaign_root, allow_external_root=args.allow_external_root)
    document, manifest_sha = load_manifest(
        campaign_root / "reports" / MANIFEST_FILENAME,
        repo_manifest=require_repo_manifest(_REPO_ROOT),
    )
    layout = rehearsal_layout(_REPO_ROOT, args.campaign_id)
    verify_rehearsal_window(_REPO_ROOT, args.campaign_id)
    upstream_check = verify_upstream(document, repo_root=_REPO_ROOT)
    started = time.perf_counter()
    facts = _report_facts(campaign_root, document)
    aa = mechanical_aa(layout, arm_base="base", arm_clone="mech_aa")
    readings = build_readings(layout, document, repo_root=_REPO_ROOT)
    authorization = _run_authorization(campaign_root)
    measurements = machine_time_reference(campaign_root, authorization)
    runbook = verify_runbook(
        _REPO_ROOT,
        artifact_hashes=doc_consistency_hashes(campaign_root, document),
        campaign_id=str(document["campaign"]["campaign_id"]),
        lineage_sequence=int(document["lineage"]["sequence"]),
    )
    outcome: Dict[str, Any] = {
        "campaign_root": campaign_root.as_posix(),
        "manifest_sha256": manifest_sha,
        "upstream_links_verified": len(upstream_check),
        "aa_identical": bool(aa["identical"]),
        "aa_compared_days": int(aa["compared_days"]),
        "h6_items": [
            {"name": item["name"], "status": item["status"], "judgment": item["judgment"]}
            for item in readings["h6_items"]
        ],
        "machine_time": measurements,
        "report_seconds": time.perf_counter() - started,
        "dry_run": not args.apply,
    }
    if not args.apply:
        return outcome
    written = write_outputs(
        layout=layout,
        document=document,
        repo_root=_REPO_ROOT,
        facts={**facts, "manifest_sha256": manifest_sha},
        aa=aa,
        readings=readings,
        machine_time=measurements,
        authorization=authorization,
        production=production_state(
            _REPO_ROOT, _production_before(campaign_root), window="register → report（跨阶段）"
        ),
        referenced=referenced_readings(_REPO_ROOT, document),
        runbook=runbook,
        upstream_check={"verified": True, "links": upstream_check, "checked_in": "report"},
    )
    outcome.update(written)
    artifacts.write_json_once(
        artifacts.artifact_path(campaign_root, artifacts.ARTIFACT_MACHINE_TIME),
        {
            "stage": "report",
            "report_seconds": outcome["report_seconds"],
            "note": "墙钟不属于产物身份",
        },
        what="墙钟登记（report）",
        sealed=False,
    )
    return outcome


def _production_before(campaign_root: Path) -> Optional[Dict[str, Any]]:
    """取 register 阶段的生产根快照；**缺失即返回 None**（fail-closed ⇒ H1-6 判未执行）。"""
    path = artifacts.artifact_path(campaign_root, artifacts.ARTIFACT_PRODUCTION_BEFORE)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload.get("snapshot") or {})
    return None


def _run_authorization(campaign_root: Path) -> Optional[Dict[str, Any]]:
    """取运行阶段登记的授权记录（无授权 ⇒ None）。"""
    path = artifacts.artifact_path(campaign_root, artifacts.ARTIFACT_MACHINE_TIME_RUN)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in reversed(list(payload.get("history") or [])):
        if entry.get("authorization_sha256"):
            return {
                "stage": entry.get("stage"),
                "authorization_sha256": entry.get("authorization_sha256"),
                "note": (
                    "授权记录哈希由运行阶段登记；原文落 docs/reports/"
                    "p5b_t6_long_task_authorization_20261102.json（本字段不含时刻，保持产物确定性）"
                ),
            }
    return None


def _stage_check(args: argparse.Namespace) -> Dict[str, Any]:
    campaign_root = campaign_root_for(args.campaign_id)
    document, manifest_sha = load_manifest(
        campaign_root / "reports" / MANIFEST_FILENAME,
        repo_manifest=require_repo_manifest(_REPO_ROOT),
    )
    upstream = verify_upstream(document, repo_root=_REPO_ROOT)
    layout = rehearsal_layout(_REPO_ROOT, args.campaign_id)
    aa = mechanical_aa(layout, arm_base="base", arm_clone="mech_aa")
    verdict_path = artifacts.artifact_path(campaign_root, artifacts.ARTIFACT_VERDICT)
    verdict = (
        artifacts.load_artifact(verdict_path, what="T6 判定表") if verdict_path.is_file() else None
    )
    status = (
        None
        if verdict is None
        else {
            key: verdict["author_side_checks"][key]["author_side_status"]
            for key in ("h1", "h6", "h7")
        }
    )
    chain_ok, chain_mismatches = _verify_disk_chain(campaign_root, verdict)
    report_path = artifacts.artifact_path(campaign_root, artifacts.ARTIFACT_REPORT)
    report_payload = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    )
    repo_sha = manifest_sha256(require_repo_manifest(_REPO_ROOT))
    return {
        "campaign_root": campaign_root.as_posix(),
        "manifest_sha256": manifest_sha,
        "repo_manifest_sha256": repo_sha,
        "repo_copy_identical": manifest_sha == repo_sha,
        "upstream_links_verified": len(upstream),
        "aa_identical": bool(aa["identical"]),
        "aa_compared_days": int(aa["compared_days"]),
        "verdict_present": verdict is not None,
        "artifact_chain_verified": chain_ok,
        "artifact_chain_mismatches": chain_mismatches,
        "author_side_status": status,
        "forbidden_gate_fields": ([] if verdict is None else forbidden_gate_fields(verdict))
        + forbidden_gate_fields(report_payload),
        "h6_claim": None if verdict is None else verdict.get("h6_claim"),
        "gate_owner": None if verdict is None else verdict.get("gate_owner"),
    }


def _verify_disk_chain(
    campaign_root: Path, verdict: Optional[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """三环第三环落地：重算磁盘报告哈希并与判定表 ``hash_chain`` 比对。"""
    if verdict is None:
        return False, ["verdict_missing"]
    chain = dict(verdict.get("hash_chain") or {})
    mismatches: List[str] = []
    for name, key in (
        (artifacts.ARTIFACT_REPORT, "rehearsal_report.json_sha256"),
        (artifacts.ARTIFACT_REPORT_MD, "rehearsal_report.md_sha256"),
    ):
        path = artifacts.artifact_path(campaign_root, name)
        expected = str(chain.get(key) or "")
        if not path.is_file():
            mismatches.append(f"{name}:missing")
            continue
        actual = artifacts.sha256_file(path)
        if actual != expected:
            mismatches.append(f"{name}:{actual}!={expected}")
    return (not mismatches), mismatches


def main(argv: Optional[List[str]] = None) -> int:
    global _REPO_ROOT
    args = _parse_args(argv)
    _REPO_ROOT = Path(args.repo_root) if args.repo_root else ROOT
    try:
        if args.freeze:
            outcome = _stage_freeze(args)
        elif args.register:
            outcome = register_rehearsal(
                repo_root=_REPO_ROOT,
                campaign_id=args.campaign_id,
                allow_external_root=args.allow_external_root,
            )
        elif args.run:
            outcome = _stage_run(args)
        elif args.report:
            outcome = _stage_report(args)
        else:
            outcome = _stage_check(args)
    except LongTaskAuthorizationRequired as exc:
        _print({"status": "authorization_required", **exc.projection}, args.json)
        return 4
    except Exception as exc:  # noqa: BLE001 - 薄入口统一转退出码，异常信息完整打印
        print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print(outcome, args.json)
    if outcome.get("artifact_chain_verified") is False:
        return 1
    if outcome.get("forbidden_gate_fields"):
        return 1
    status = outcome.get("author_side_status")
    if isinstance(status, dict) and any(value != "pass" for value in status.values()):
        return 3
    return 0


_REPO_ROOT = ROOT


if __name__ == "__main__":
    raise SystemExit(main())
