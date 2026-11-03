# -*- coding: utf-8 -*-
"""P5b 试车场 T3 账本修复与运营面：缺口修复、只读查询、推进守卫、暂存清理、T3 报告。

契约来源：设计 v4 §4.3（缺口修复、INVALIDATED 终态、唯一发布点）、§7.1（产物根）、
§7.2（报告字段）、§8-H2/H3。

- 修复默认 **dry-run**；``INVALIDATED`` 缺口时**不做任何账本修复**（只登记终态）；
- 修复只处理「可重建」缺口：缺片按 journal 重写（逐字节一致）、清单/index 重建、链重写、
  未发布分片移入 ``audit/quarantine/``；不可读的派生投影先隔离留证再重建；
- 写动作（``apply=True``）取 campaign 运行锁（与 runner 互斥）、并受生产根硬禁；
- 查询与守卫**只读**：不写持仓、不写账本。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.hosts.testbed.journal import (
    load_day_journal,
    read_index,
    rebuild_index,
    rebuild_manifest,
    write_shard,
)
from src.lazybull.v2.hosts.testbed.layout import (
    CampaignLayout,
    FamilyTxnError,
    gen_id,
    remove_root,
)
from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.storage import read_json, seal_record, verify_seal

from src.lazybull.v2.hosts.testbed.shard_chain import build_chain
from src.lazybull.v2.hosts.testbed.ledger_audit import (
    audit_ledger,
    write_ledger_artifacts,
    _shard_gaps,
)
from src.lazybull.v2.hosts.testbed.ledger_common import (
    LedgerGap,
    LedgerInvalidatedError,
    guard_ledger_write,
    invalidated_days,
    ledger_paths,
    record_invalidations,
    _gap,
    _scope_fields,
)

__all__ = [
    "REPORT_SCHEMA",
    "LedgerRepairRequiredError",
    "assert_ledger_advanceable",
    "build_ledger_report",
    "prune_staging",
    "query_shards",
    "repair_ledger",
]


class LedgerRepairRequiredError(FamilyTxnError):
    """存在**可修复但未修复**的账本缺口（未发布分片残留）⇒ 推进前必须先隔离。"""

    __test__ = False


#: T3 账本报告 schema
REPORT_SCHEMA = "p5b_testbed_t3_ledger_report/v1"
#: 未发布分片等「缺口类」修复动作
_ACTION_REBUILD_SHARD = "rebuild_shard"
_ACTION_REBUILD_MANIFEST = "rebuild_manifest"
_ACTION_REBUILD_INDEX = "rebuild_index"
_ACTION_QUARANTINE_SHARD = "quarantine_shard"
_ACTION_WRITE_CHAIN = "write_chain"
_ACTION_PRUNE_STAGING = "prune_staging"

# ---------------------------------------------------------------------------
# 缺口修复（默认 dry-run）
# ---------------------------------------------------------------------------


def _plan_actions(
    layout: CampaignLayout, audit: Mapping[str, Any], *, prune: bool
) -> List[Dict[str, Any]]:
    """由缺口推导修复动作（顺序：隔离 → 补片 → 清单 → index → 链 → 暂存清理）。"""
    actions: List[Dict[str, Any]] = []
    for gap in audit.get("gaps") or []:
        kind = gap["kind"]
        if kind == "unpublished_attempt_shard":
            arm_id, date = str(gap["arm_id"]), str(gap["virtual_date"])
            actions.append(
                {
                    "action": _ACTION_QUARANTINE_SHARD,
                    "arm_id": arm_id,
                    "virtual_date": date,
                    "target": ledger_paths(layout).quarantine_shard_path(arm_id, date).as_posix(),
                }
            )
        elif kind == "missing_day":
            actions.append(
                {
                    "action": _ACTION_REBUILD_SHARD,
                    "arm_id": str(gap["arm_id"]),
                    "virtual_date": str(gap["virtual_date"]),
                    "sha256": gap.get("expected_sha256"),
                }
            )
    if any(gap["kind"] in {"manifest_missing", "manifest_mismatch"} for gap in audit["gaps"]):
        arms = sorted(
            {
                str(gap["arm_id"])
                for gap in audit["gaps"]
                if gap["kind"] in {"manifest_missing", "manifest_mismatch"}
            }
        )
        actions.extend({"action": _ACTION_REBUILD_MANIFEST, "arm_id": arm} for arm in arms)
    if any(gap["kind"] in {"index_absent", "index_missing_day"} for gap in audit["gaps"]):
        actions.append({"action": _ACTION_REBUILD_INDEX})
    if any(gap["kind"] == "chain_stale" for gap in audit["gaps"]):
        actions.append({"action": _ACTION_WRITE_CHAIN})
    if prune:
        actions.append({"action": _ACTION_PRUNE_STAGING})
    return actions


def _apply_action(
    layout: CampaignLayout, campaign_id: str, action: Mapping[str, Any]
) -> Dict[str, Any]:
    """执行单个修复动作（幂等；返回动作 + 输出摘要）。"""
    kind = str(action["action"])
    if kind == _ACTION_QUARANTINE_SHARD:
        return _apply_quarantine(layout, action)
    if kind == _ACTION_REBUILD_SHARD:
        return _apply_rebuild_shard(layout, campaign_id, action)
    if kind == _ACTION_REBUILD_MANIFEST:
        arm_id = str(action["arm_id"])
        quarantined = _quarantine_unreadable(
            layout, layout.manifest_path(arm_id), what=f"分片清单 {arm_id}"
        )
        result = rebuild_manifest(layout, campaign_id, arm_id)
        return {**dict(action), "written": bool(result["written"]), "quarantined": quarantined}
    if kind == _ACTION_REBUILD_INDEX:
        quarantined = _quarantine_unreadable(layout, layout.index_path, what="试车场 index")
        result = rebuild_index(layout, campaign_id)
        return {**dict(action), "written": bool(result["written"]), "quarantined": quarantined}
    if kind == _ACTION_WRITE_CHAIN:
        audit = audit_ledger(layout, campaign_id)
        written = write_ledger_artifacts(layout, campaign_id, audit)
        return {**dict(action), **written}
    if kind == _ACTION_PRUNE_STAGING:
        return {
            **dict(action),
            "removed": prune_staging(layout, apply=True, take_lock=False)["removed"],
        }
    raise FamilyTxnError(f"未知修复动作: {kind!r}")


def _unique_path(path: Path) -> Path:
    """目标名去重（递增后缀，绝不覆盖既有隔离证据）。"""
    candidate = Path(path)
    index = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        index += 1
    return candidate


def _quarantine_unreadable(
    layout: CampaignLayout,
    path: Path,
    *,
    what: str,
    verify: Optional[Callable[[Mapping[str, Any]], None]] = None,
) -> Optional[str]:
    """不可读（非法 JSON / 封存不符 / schema 不符）的派生投影先移入隔离区留证，再重建。"""
    if not path.is_file():
        return None
    try:
        doc = read_json(path, what=what)
        if verify is not None:
            verify(doc)
    except TestbedError:
        target = _unique_path(ledger_paths(layout).quarantine_root / "projections" / path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        return target.as_posix()
    return None


def quarantine_unreadable_projections(layout: CampaignLayout) -> List[str]:
    """把不可读的读侧派生投影（``_index.json`` / 各臂 ``manifest.json``）隔离留证。

    运行器在取锁后、启动恢复前调用：这样恢复与推进看到的是「投影缺失」而非「投影损坏」，
    再按 journal 幂等重建，避免以未分类异常中断（审计本就把这两类判为可重建）。

    Returns:
        被隔离的文件路径列表（可读投影原样保留）。
    """
    guard_ledger_write(layout)
    moved: List[str] = []
    index_quarantined = _quarantine_unreadable(
        layout,
        layout.index_path,
        what="试车场 index",
        verify=lambda doc: verify_seal(doc, what="试车场 index"),
    )
    if index_quarantined:
        moved.append(index_quarantined)
    if layout.ledger_root.is_dir():
        for arm_dir in sorted(path for path in layout.ledger_root.iterdir() if path.is_dir()):
            quarantined = _quarantine_unreadable(
                layout,
                arm_dir / "manifest.json",
                what=f"分片清单 {arm_dir.name}",
                verify=lambda doc, arm=arm_dir.name: verify_seal(doc, what=f"分片清单 {arm}"),
            )
            if quarantined:
                moved.append(quarantined)
    return moved


def _apply_quarantine(layout: CampaignLayout, action: Mapping[str, Any]) -> Dict[str, Any]:
    """未发布分片移入隔离区（不动已发布内容；同名已存在则递增后缀，绝不覆盖既有证据）。"""
    source = layout.shard_path(str(action["arm_id"]), str(action["virtual_date"]))
    target = Path(str(action["target"]))
    if not source.is_file():
        return {**dict(action), "moved": False}
    target = _unique_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    return {**dict(action), "moved": True, "quarantined": target.as_posix()}


def _apply_rebuild_shard(
    layout: CampaignLayout, campaign_id: str, action: Mapping[str, Any]
) -> Dict[str, Any]:
    """已发布日缺分片 ⇒ 由 journal 重建（``write_shard`` 载荷无墙钟，逐字节一致）。"""
    arm_id = str(action["arm_id"])
    date = str(action["virtual_date"])
    doc = load_day_journal(layout, date)
    attempt = next(
        (
            dict(item)
            for item in doc.get("attempts") or []
            if item.get("status") == "COMMITTED" and item.get("virtual_date") == date
        ),
        None,
    )
    if attempt is None:
        raise FamilyTxnError(f"修复缺分片失败：{date} 无 COMMITTED attempt（禁止凭空补片）")
    result = write_shard(layout, campaign_id, attempt, arm_id)
    if str(result["sha256"]) != str(action.get("sha256")):
        raise FamilyTxnError(
            f"修复缺分片后内容身份与审计期望不符（campaign 失效）: {arm_id}/{date} "
            f"重建={result['sha256']} 期望={action.get('sha256')}"
        )
    return {**dict(action), "written": bool(result["written"])}


def repair_ledger(
    layout: CampaignLayout,
    campaign_id: str,
    *,
    apply: bool = False,
    prune_staging_after: bool = False,
    clock_days: Optional[Sequence[str]] = None,
    take_lock: bool = True,
) -> Dict[str, Any]:
    """缺口修复（默认 dry-run；幂等、留证、绝不覆盖已发布分片）。

    规则：**存在 INVALIDATED 缺口时不做任何账本修复**（只登记终态），因为此时「哪些字节是
    已发布内容」不可唯一确定；修复只处理可重建缺口。

    Args:
        layout: campaign 布局。
        campaign_id: campaign 标识。
        apply: ``False``（默认）= 只算计划，零写盘。
        prune_staging_after: 修复后是否顺带清理已终态 attempt 的暂存目录。
        clock_days: campaign 虚拟窗口（可选；提供时额外校验分片越窗）。
        take_lock: ``apply=True`` 时是否取 campaign 运行锁（与 runner 互斥；默认取）。

    Returns:
        ``{dry_run, blocked, audit_before, audit_after, planned, applied, invalidated}``。

    Raises:
        CampaignBusyError: ``apply=True`` 且另一运行器持有 campaign 锁。
        TestbedScopeError: campaign 根落在生产面。
    """
    started = time.perf_counter()
    if not apply:
        return _repair_once(
            layout,
            campaign_id,
            apply=False,
            prune_staging_after=prune_staging_after,
            clock_days=clock_days,
            started=started,
        )
    guard_ledger_write(layout)
    if not take_lock:
        return _repair_once(
            layout,
            campaign_id,
            apply=True,
            prune_staging_after=prune_staging_after,
            clock_days=clock_days,
            started=started,
        )
    from src.lazybull.v2.hosts.testbed.runner import CampaignRunLock

    with CampaignRunLock(layout):
        return _repair_once(
            layout,
            campaign_id,
            apply=True,
            prune_staging_after=prune_staging_after,
            clock_days=clock_days,
            started=started,
        )


def _repair_once(
    layout: CampaignLayout,
    campaign_id: str,
    *,
    apply: bool,
    prune_staging_after: bool,
    clock_days: Optional[Sequence[str]],
    started: float,
) -> Dict[str, Any]:
    """修复主体（不取锁；由 :func:`repair_ledger` 负责加锁与守卫）。"""
    before = audit_ledger(layout, campaign_id, clock_days=clock_days)
    invalidated = list(before["invalidated"])
    planned = [] if invalidated else _plan_actions(layout, before, prune=prune_staging_after)
    applied: List[Dict[str, Any]] = []
    after = before
    if apply and invalidated:
        record_invalidations(layout, campaign_id, _invalidated_gaps(before))
        write_ledger_artifacts(layout, campaign_id, before)
        after = audit_ledger(layout, campaign_id, clock_days=clock_days)
    elif apply:
        for action in planned:
            applied.append(_apply_action(layout, campaign_id, action))
        interim = audit_ledger(layout, campaign_id, clock_days=clock_days)
        write_ledger_artifacts(layout, campaign_id, interim)
        after = audit_ledger(layout, campaign_id, clock_days=clock_days)
        if after != interim:
            write_ledger_artifacts(layout, campaign_id, after)
    return {
        "dry_run": not apply,
        "blocked": bool(invalidated),
        "audit_before": before,
        "audit_after": after,
        "planned": planned,
        "applied": applied,
        "invalidated": invalidated,
        "seconds": time.perf_counter() - started,
    }


def _invalidated_gaps(audit: Mapping[str, Any]) -> List[LedgerGap]:
    """把审计的 INVALIDATED 清单还原成缺口对象（登记用）。"""
    gaps: List[LedgerGap] = []
    for entry in audit.get("invalidated") or []:
        gaps.append(
            _gap(
                str(entry["arm_id"]),
                None if entry.get("virtual_date") is None else str(entry["virtual_date"]),
                str(entry["kind"]),
                str(entry.get("detail") or ""),
                repairable=False,
                generation=entry.get("generation"),
            )
        )
    return gaps


# ---------------------------------------------------------------------------
# 只读查询（供 T4/T5 消费；旁路只读、禁止写回持仓或账本）
# ---------------------------------------------------------------------------


def query_shards(
    layout: CampaignLayout,
    arm_id: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    allow_invalidated: bool = False,
) -> Dict[str, Any]:
    """按臂 + 日期区间返回**已发布**连续分片与链 digest（只读，零写盘）。

    Args:
        layout: campaign 布局。
        arm_id: 臂标识。
        start: 起始虚拟日（含；None = 不设下界）。
        end: 结束虚拟日（含；None = 不设上界）。
        allow_invalidated: 区间含 INVALIDATED 日时是否仍返回（默认 fail-closed 拒绝）。

    Returns:
        ``{arm_id, start, end, days, contiguous, chain_digest, published_generation, gaps}``。

    **链 digest 口径**：``chain_digest`` 是**本区间**的链值（与全链共用 seed，故区间链值不等于
    全链末值）；全链末值见 :func:`audit_ledger` 的 ``chain.chain_digests`` 或
    ``audit/shard_chain.json``。

    Raises:
        FamilyTxnError: campaign 缺已发布 index。
        LedgerInvalidatedError: 区间含 INVALIDATED 日（含 campaign 级 ``*`` 登记）且未显式放开。
    """
    index = read_index(layout)
    if index is None:
        raise FamilyTxnError(f"campaign {layout.root.name!r} 缺已发布 index，无法查询分片")
    published = sorted(
        (dict(entry) for entry in index.get("days") or []),
        key=lambda entry: int(entry["generation"]),
    )
    window = [entry for entry in published if _in_window(str(entry["virtual_date"]), start, end)]
    taint_all, tainted = _invalidated_scope(layout, arm_id)
    blocked = sorted(date for date in tainted if _in_window(date, start, end))
    if taint_all:
        blocked = sorted({*blocked, *[str(entry["virtual_date"]) for entry in window]})
    if blocked and not allow_invalidated:
        raise LedgerInvalidatedError(
            f"臂 {arm_id!r} 的查询区间含 INVALIDATED 日（需人工介入）: {blocked}"
            + ("；campaign 级失效波及全部虚拟日" if taint_all else "")
        )
    days: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    for entry in window:
        date = str(entry["virtual_date"])
        generation = int(entry["generation"])
        arm_entry = (entry.get("arms") or {}).get(arm_id) or {}
        expected_sha = arm_entry.get("shard_sha256")
        shard_path = layout.shard_path(arm_id, date)
        actual_sha, day_gaps = _query_day(arm_id, date, generation, expected_sha, shard_path)
        gaps.extend(gap.to_dict() for gap in day_gaps)
        links.append(
            {
                "virtual_date": date,
                "generation": generation,
                "family_txn_id": entry.get("family_txn_id"),
                "shard_sha256": actual_sha,
            }
        )
        days.append(
            {
                "virtual_date": date,
                "generation": generation,
                "gen_id": gen_id(generation),
                "family_txn_id": entry.get("family_txn_id"),
                "shard_sha256": actual_sha,
                "invalidated": bool(taint_all or date in tainted),
            }
        )
    campaign_id = str(index.get("campaign_id") or layout.root.name)
    chain = build_chain(campaign_id, arm_id, links) if links else None
    contiguous = _is_contiguous([int(entry["generation"]) for entry in window])
    return {
        "arm_id": arm_id,
        "start": start,
        "end": end,
        "days": days,
        "contiguous": bool(contiguous and not gaps),
        "chain_digest": None if chain is None else chain["chain_digest"],
        "chain_links": [] if chain is None else chain["links"],
        "published_generation": int(index.get("published_generation", 0)),
        "gaps": gaps,
        "invalidated": blocked,
    }


def _invalidated_scope(layout: CampaignLayout, arm_id: str) -> Tuple[bool, set]:
    """INVALIDATED 波及范围：``(是否全 campaign 波及, 具体日期集合)``。

    - 本臂条目 + campaign 级（``arm_id == "*"``）条目都参与；
    - campaign 级且日期为空 ⇒ 波及全部虚拟日（journal 损坏 / index 超前类）。
    """
    registry = invalidated_days(layout)
    own = set(registry.get(arm_id, {}))
    campaign = set(registry.get("*", {}))
    taint_all = "" in campaign
    tainted = (own | campaign) - {""}
    return taint_all, tainted


def _query_day(
    arm_id: str,
    date: str,
    generation: int,
    expected_sha: Optional[str],
    shard_path: Path,
) -> Tuple[Optional[str], List[LedgerGap]]:
    """查询单日分片（只读；缺片/坏片返回缺口而非抛异常）。"""
    if expected_sha is None:
        # 与审计侧同口径：index 缺臂条目属派生投影空洞（可重建），不是失效终态
        return (
            None,
            [
                _gap(
                    arm_id,
                    date,
                    "index_missing_day",
                    f"index 该日缺此臂条目（可从 journal 重建）: {arm_id}/{date}",
                    repairable=True,
                    generation=generation,
                )
            ],
        )
    gap_list, actual_sha = _shard_gaps(
        arm_id,
        date,
        {"generation": generation, "sha256": expected_sha},
        shard_path if shard_path.is_file() else None,
    )
    return actual_sha, gap_list


def _in_window(date: str, start: Optional[str], end: Optional[str]) -> bool:
    if start is not None and date < str(start):
        return False
    if end is not None and date > str(end):
        return False
    return True


def _is_contiguous(generations: Sequence[int]) -> bool:
    return all(current == previous + 1 for previous, current in zip(generations, generations[1:]))


# ---------------------------------------------------------------------------
# 推进守卫（runner 前置；只读，不修复）
# ---------------------------------------------------------------------------


def assert_ledger_advanceable(
    layout: CampaignLayout,
    campaign_id: str,
    *,
    clock_days: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """虚拟时钟推进前守卫（只读，不修复）。

    拒绝推进的三类条件：

    1. 已登记 INVALIDATED（append-only 终态，需人工介入）；
    2. 本次审计检测到**不可重建**缺口（含 journal 不可读、坏片、混代、越窗）；
    3. 账本内残留**未发布 attempt 分片**（``unpublished_attempt_shard``）：该残片会让同日重算
       撞上 ``write_shard`` 的「禁止覆盖已发布分片」，必须先隔离再推进。

    「可重建」缺口（缺片 / 清单 / index 落后）不在此拒绝：``recover_campaign`` 会在推进前按
    journal 幂等补齐（§4.3 状态表「修复后再推进」由**顺序**保证，不是靠拒绝）。

    Raises:
        LedgerInvalidatedError: 条件 1/2 成立。
        LedgerRepairRequiredError: 条件 3 成立（须先 ``repair_ledger(apply=True)``）。
    """
    registered = invalidated_days(layout)
    if registered:
        detail = sorted(
            f"{arm}/{date or '<all-days>'}" for arm, days in registered.items() for date in days
        )
        raise LedgerInvalidatedError(
            f"campaign {campaign_id!r} 存在 INVALIDATED 终态（后续虚拟日拒绝推进，需人工介入）: "
            f"{detail}"
        )
    audit = audit_ledger(layout, campaign_id, clock_days=clock_days)
    if audit["invalidated"]:
        detail = [
            f"{entry['arm_id']}/{entry['virtual_date']}/{entry['kind']}"
            for entry in audit["invalidated"]
        ]
        raise LedgerInvalidatedError(
            f"campaign {campaign_id!r} 检测到不可重建缺口（后续虚拟日拒绝推进，需人工介入）: "
            f"{detail}；先运行 scripts/v2_p5b/ledger_audit.py --repair 登记终态"
        )
    orphans = sorted(
        f"{gap['arm_id']}/{gap['virtual_date']}"
        for gap in audit["gaps"]
        if gap["kind"] == "unpublished_attempt_shard"
    )
    if orphans:
        raise LedgerRepairRequiredError(
            f"campaign {campaign_id!r} 账本内残留未发布 attempt 的分片（同日重算会被 "
            f"write_shard 拒绝覆盖），须先隔离再推进: {orphans}；"
            "运行 scripts/v2_p5b/ledger_audit.py --repair"
        )
    return {
        "status": audit["status"],
        "three_way_status": audit.get("three_way_status"),
        "artifact_status": audit.get("artifact_status"),
        "published_generation": audit["published"].get("published_generation"),
        "chain_digest": audit["chain"].get("chain_digest"),
        "gaps": len(audit["gaps"]),
        "repairable_gaps": sorted({gap["kind"] for gap in audit["gaps"] if gap["repairable"]}),
    }


# ---------------------------------------------------------------------------
# staging 清理（T2 遗留项：整臂复制导致磁盘线性增长）
# ---------------------------------------------------------------------------


def _staging_bytes(layout: CampaignLayout) -> Tuple[int, int]:
    """``staging/`` 的文件数与总字节（规模登记用）。"""
    root = layout.staging_root
    if not root.is_dir():
        return 0, 0
    files = 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            files += 1
            total += path.stat().st_size
    return files, total


def prune_staging(
    layout: CampaignLayout, *, apply: bool = False, take_lock: bool = True
) -> Dict[str, Any]:
    """清理**已终态** attempt 的暂存目录（默认 dry-run；不动未终态 attempt）。

    口径：``staging/<date>/<attempt>`` 只在 journal 该 attempt 为 ``COMMITTED`` 或 ``ABORTED``
    时清理；``COMMITTED`` 代里被提升时留下的 ``_rolled/<arm>``（父代旧根）随之删除——父代仍由
    ``backups/<date>/<attempt>/<arm>`` 保留，可验证性不受影响。

    Args:
        layout: campaign 布局。
        apply: ``False``（默认）= dry-run，零写盘。
        take_lock: ``apply=True`` 时是否取 campaign 运行锁（runner 内部调用须传 False）。

    Raises:
        CampaignBusyError: ``apply=True`` 且另一运行器持有 campaign 锁。
        TestbedScopeError: campaign 根落在生产面。
    """
    if not apply:
        return _prune_once(layout, apply=False)
    guard_ledger_write(layout)
    if not take_lock:
        return _prune_once(layout, apply=True)
    from src.lazybull.v2.hosts.testbed.runner import CampaignRunLock

    with CampaignRunLock(layout):
        return _prune_once(layout, apply=True)


def _prune_once(layout: CampaignLayout, *, apply: bool) -> Dict[str, Any]:
    files_before, bytes_before = _staging_bytes(layout)
    removed: List[str] = []
    if layout.staging_root.is_dir():
        for day_dir in sorted(path for path in layout.staging_root.iterdir() if path.is_dir()):
            removed.extend(_prune_day(layout, day_dir, apply=apply))
    files_after, bytes_after = _staging_bytes(layout)
    return {
        "dry_run": not apply,
        "removed": removed,
        "files_before": files_before,
        "bytes_before": bytes_before,
        "files_after": files_after,
        "bytes_after": bytes_after,
    }


def _prune_day(layout: CampaignLayout, day_dir: Path, *, apply: bool) -> List[str]:
    try:
        doc = load_day_journal(layout, day_dir.name)
    except (TestbedError, FamilyTxnError, OSError):
        return []
    terminal = {
        str(attempt["attempt_id"])
        for attempt in doc.get("attempts") or []
        if str(attempt.get("status")) in {"COMMITTED", "ABORTED"}
    }
    removed: List[str] = []
    for attempt_dir in sorted(path for path in day_dir.iterdir() if path.is_dir()):
        if attempt_dir.name not in terminal:
            continue
        removed.append(attempt_dir.as_posix())
        if apply:
            remove_root(attempt_dir)
    return removed


# ---------------------------------------------------------------------------
# §7.2 T3 报告
# ---------------------------------------------------------------------------


def build_ledger_report(
    audit: Mapping[str, Any],
    *,
    repair: Optional[Mapping[str, Any]] = None,
    fault_matrix: Optional[Mapping[str, Any]] = None,
    scale: Optional[Mapping[str, Any]] = None,
    machine_time: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """组装 §7.2 的 T3 覆盖字段。

    覆盖：链 digest / 三方一致 / 缺口与 INVALIDATED / 故障矩阵判定表引用 / 墙钟与长任务阈值 /
    唯一发布点与代际身份。``fault_matrix`` 段为**引用 + 指纹**（判定表本体落
    ``audit/fault_matrix.json``，两处通过 ``matrix_fingerprint`` 与 ``record_sha256`` 互链）。
    """
    long_task_threshold = float((machine_time or {}).get("long_task_threshold_seconds") or 1200.0)
    total_seconds = float((machine_time or {}).get("total_seconds") or 0.0)
    return seal_record(
        {
            "schema": REPORT_SCHEMA,
            **_scope_fields(str(audit.get("campaign_id"))),
            "ledger_status": audit.get("status"),
            "three_way_status": audit.get("three_way_status"),
            "artifact_status": audit.get("artifact_status"),
            "published": dict(audit.get("published") or {}),
            "unique_publish_point": dict(audit.get("unique_publish_point") or {}),
            "shard_chain_digest": (audit.get("chain") or {}).get("chain_digest"),
            "shard_chain": {
                "arms": dict((audit.get("chain") or {}).get("chain_digests") or {}),
                "link_counts": {
                    arm: int(payload["link_count"])
                    for arm, payload in ((audit.get("chain") or {}).get("arms") or {}).items()
                },
            },
            "three_way": dict(audit.get("three_way") or {}),
            "gaps": list(audit.get("gaps") or []),
            "gaps_by_class": dict(audit.get("gaps_by_class") or {}),
            "invalidated": list(audit.get("invalidated") or []),
            "repair": (
                None
                if repair is None
                else {
                    "dry_run": repair.get("dry_run"),
                    "blocked": repair.get("blocked"),
                    "planned": list(repair.get("planned") or []),
                    "applied": list(repair.get("applied") or []),
                }
            ),
            "fault_matrix": None if fault_matrix is None else dict(fault_matrix),
            "scale": None if scale is None else dict(scale),
            "machine_time": {
                **dict(machine_time or {}),
                "long_task_threshold_seconds": long_task_threshold,
                "long_task_threshold_exceeded": bool(total_seconds > long_task_threshold),
            },
            "limitations": [
                "HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE：历史虚拟时钟试车场，不是前向影子",
                "T3 只覆盖长期分片账本、正式故障注入矩阵与 containment 结论",
                "收益/回撤裁判（A/A、MDE、临界值、holdout）归 T4/T5，本报告不产生裁判结论",
                "INVALIDATED 不自动清除；人工介入前该臂该日不得作为任何统计输入",
                "本报告不含 SLA、恢复时限或可用性承诺",
            ],
        }
    )
