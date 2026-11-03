# -*- coding: utf-8 -*-
"""P5b T3 账本审计薄入口（默认只读；``--repair`` 才写 ``audit/`` 与补缺片）。

用法::

    # 只读审计（不写盘）
    python scripts/v2_p5b/ledger_audit.py --campaign-root data/paper_shadow/p5b_testbed/<id>

    # 审计 + 缺口修复（可重建缺口；INVALIDATED 只登记终态、拒绝推进）
    python scripts/v2_p5b/ledger_audit.py --campaign-root <id> --repair --prune-staging

    # 只读查询某臂日期区间的连续分片与链 digest
    python scripts/v2_p5b/ledger_audit.py --campaign-root <id> \\
        --arm aa_1 --start 20260706 --end 20260710

边界：只写 ``<campaign_root>/audit/`` 与 ``<campaign_root>/reports/``（``--report``），
不写生产根、不写持仓、不改已发布分片字节。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.hosts.testbed import (  # noqa: E402
    assert_write_scope,
    audit_ledger,
    build_ledger_report,
    layout_for,
    ledger_paths,
    load_campaign,
    matrix_fingerprint,
    prune_staging,
    query_shards,
    repair_ledger,
)
from src.lazybull.v2.hosts.testbed.storage import write_json_atomic  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--campaign-root", required=True, help="campaign 根目录")
    parser.add_argument("--arm", default="", help="只查询该臂的已发布分片区间（只读）")
    parser.add_argument("--start", default="", help="查询起始虚拟日 YYYYMMDD（含）")
    parser.add_argument("--end", default="", help="查询结束虚拟日 YYYYMMDD（含）")
    parser.add_argument("--repair", action="store_true", help="执行缺口修复（默认 dry-run）")
    parser.add_argument(
        "--prune-staging", action="store_true", help="清理已终态 attempt 的暂存目录"
    )
    parser.add_argument("--report", action="store_true", help="把 §7.2-T3 报告落 reports/")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出摘要")
    parser.add_argument(
        "--allow-external-root",
        action="store_true",
        help="仅测试/临时脚本：放开写域白名单（不解除生产根禁令）",
    )
    return parser.parse_args(argv)


def _campaign_clock_days(root: Path) -> list[str]:
    """读 campaign 虚拟窗口（可选；读不到则越窗检查跳过，不影响审计主体）。"""
    try:
        campaign = load_campaign(root)
    except Exception:  # noqa: BLE001 - 工具入口：注册记录缺失不阻断只读审计
        return []
    return [str(day) for day in (campaign.get("clock") or {}).get("days") or []]


def _existing_verdict_table(layout) -> dict | None:
    """引用既有故障矩阵判定表（无则 None；只做指纹引用与**指纹核对**，不重跑矩阵）。

    指纹与当前 `fault_matrix.MATRIX` 不一致 ⇒ 标 `stale=True` 且不认 `all_passed`，
    避免把陈旧/异版判定表当成本次证据。
    """
    path = ledger_paths(layout).matrix_path
    if not path.is_file():
        return None
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_fingerprint = matrix_fingerprint()
    matches = table.get("matrix_fingerprint") == expected_fingerprint
    cases = list(table.get("cases") or [])
    return {
        "path": path.as_posix(),
        "record_sha256": table.get("record_sha256"),
        "matrix_fingerprint": table.get("matrix_fingerprint"),
        "expected_matrix_fingerprint": expected_fingerprint,
        "stale": not matches,
        "all_passed": bool(table.get("all_passed")) and matches,
        "passed_count": table.get("passed_count"),
        "case_count": table.get("case_count"),
        "rows": [
            {
                "case_id": row.get("case_id"),
                "fault_point": row.get("fault_point"),
                "terminal": row.get("terminal"),
                "expected_terminal": row.get("expected_terminal"),
                "invariants": row.get("invariants"),
                "passed": row.get("passed"),
            }
            for row in cases
        ],
    }


def _summary(audit: dict) -> dict:
    return {
        "campaign_id": audit.get("campaign_id"),
        "status": audit.get("status"),
        "three_way_status": audit.get("three_way_status"),
        "artifact_status": audit.get("artifact_status"),
        "published_generation": (audit.get("published") or {}).get("published_generation"),
        "published_txn_id": (audit.get("published") or {}).get("published_txn_id"),
        "shard_chain_digest": (audit.get("chain") or {}).get("chain_digest"),
        "three_way_consistent": (audit.get("three_way") or {}).get("consistent"),
        "gaps_by_class": {k: v for k, v in (audit.get("gaps_by_class") or {}).items() if v},
        "invalidated": audit.get("invalidated"),
        "repair_required": audit.get("repair_required"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.campaign_root)
    layout = layout_for(root)
    campaign_id = root.resolve().name

    if args.arm:
        result = query_shards(
            layout,
            args.arm,
            start=args.start or None,
            end=args.end or None,
            allow_invalidated=False,
        )
        payload = {
            "arm_id": result["arm_id"],
            "contiguous": result["contiguous"],
            "chain_digest": result["chain_digest"],
            "published_generation": result["published_generation"],
            "days": [
                {"virtual_date": day["virtual_date"], "generation": day["generation"]}
                for day in result["days"]
            ],
            "gaps": result["gaps"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    clock_days = _campaign_clock_days(root)
    started = time.perf_counter()
    audit = audit_ledger(layout, campaign_id, clock_days=clock_days)
    audit_seconds = time.perf_counter() - started
    repair: dict | None = None
    pruning: dict | None = None
    if args.repair or args.prune_staging or args.report:
        assert_write_scope(root, allow_external_root=args.allow_external_root)
    if args.repair:
        repair = repair_ledger(
            layout,
            campaign_id,
            apply=True,
            prune_staging_after=args.prune_staging,
            clock_days=clock_days,
        )
        audit = repair["audit_after"]
    elif args.prune_staging:
        pruning = prune_staging(layout, apply=True)
        audit = audit_ledger(layout, campaign_id, clock_days=clock_days)
    # 注：repair_ledger(apply=True) 已在持锁上下文内写过 audit 产物，此处不重复写，
    # 避免锁外覆盖并发运行器刚写的新鲜产物。

    report_path = ""
    if args.report:
        frozen = _existing_verdict_table(layout)
        report = build_ledger_report(
            audit,
            repair=repair,
            fault_matrix=frozen,
            scale={"virtual_days": len(clock_days) if clock_days else None},
            machine_time={
                "total_seconds": time.perf_counter() - started,
                "audit_seconds": audit_seconds,
                "prune_staging": pruning,
                "captured_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = layout.reports_root / f"t3_ledger_{stamp}.json"
        write_json_atomic(target, report)
        report_path = target.as_posix()

    payload = _summary(audit)
    payload.update(
        {
            "audit_seconds": audit_seconds,
            "repair": (
                None
                if repair is None
                else {
                    "blocked": repair["blocked"],
                    "planned": [item["action"] for item in repair["planned"]],
                    "applied": [item["action"] for item in repair["applied"]],
                }
            ),
            "prune_staging": pruning,
            "report_path": report_path or None,
        }
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"campaign={payload['campaign_id']} status={payload['status']}")
        print(
            f"published_generation={payload['published_generation']} "
            f"txn={payload['published_txn_id']}"
        )
        print(f"shard_chain_digest={payload['shard_chain_digest']}")
        print(
            f"three_way_consistent={payload['three_way_consistent']} "
            f"gaps={payload['gaps_by_class']}"
        )
        if payload["repair"]:
            print(f"repair={payload['repair']}")
        if report_path:
            print(f"report={report_path}")
    # INVALIDATED 属于终态：退出码 2 便于脚本/人工识别「需人工介入」
    return 2 if audit.get("invalidated") else 0


if __name__ == "__main__":
    raise SystemExit(main())
