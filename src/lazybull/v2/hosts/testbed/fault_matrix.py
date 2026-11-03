# -*- coding: utf-8 -*-
"""P5b 试车场 T3 正式故障注入矩阵（表驱动）+ containment 分级 + 五不变量校验。

契约来源：设计 v4 §4.3（必须注入的故障点、每个故障点验证的五条不变量、状态表）、
§4.4（机械验收窗口）、§7.2（报告字段「故障注入点与恢复结果」）、§8-H2。

三条硬口径
==========
1. **终态与恢复结果都由证据推导**：执行侧只允许上报「测到了什么」（错误类型、生产根快照是否
   变化等原始测量值），**不得**上报期望值。``terminal`` 由 :func:`observed_terminal` 从 journal
   与 index 推出；``recovery`` 由 :func:`derive_postconditions` 从落盘证据推出（父代是否还原、
   暂存是否清理、残片落在 ``failed/`` 还是 ``audit/quarantine/``、投影是否修复、是否重算、
   INVALIDATED 是否登记、推进是否被拒、生产根是否零写入）。判定 = 观测集合 **精确等于**矩阵
   期望集合，集合来源不同，不构成自证。
2. **五条不变量**（§4.3）：``no_double_booking``（已发布账本不重复入账）、
   ``unpublished_not_in_generation``（未发布 attempt 不混入逻辑代）、``deterministic_recompute``
   （新 attempt 可确定重算）、``read_side_published_only``（读侧只见 ``published_generation``）、
   ``parent_backup_verifiable``（父代备份可验证）。每项都有**非真空下限**（无已发布日 / 无
   attempt / 无备份被检 ⇒ 判 False，而不是空集合恒真）。
3. **containment**：只给确定性动作与终态，不含 SLA / 恢复时限 / 可用性承诺。

本模块只读 campaign 现场，不写账本、不写持仓；判定表由调用方落 ``audit/fault_matrix.json``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.hosts.testbed.journal import (
    build_shard,
    load_all_attempts,
    read_index,
    shard_sha256,
)
from src.lazybull.v2.hosts.testbed.layout import (
    CampaignLayout,
    FamilyTxnError,
    optional_identity,
    root_identity,
)
from src.lazybull.v2.hosts.testbed.ledger_audit import audit_ledger, shard_sha_map
from src.lazybull.v2.hosts.testbed.ledger_common import invalidated_days
from src.lazybull.v2.hosts.testbed.ledger_repair import query_shards
from src.lazybull.v2.hosts.testbed.paths import EVIDENCE_CLASS, SCOPE, TestbedError
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "CONTAINMENT",
    "INVARIANTS",
    "MATRIX",
    "MATRIX_SCHEMA",
    "POSTCONDITIONS",
    "ContainmentLevel",
    "FaultCase",
    "build_verdict_table",
    "check_invariants",
    "derive_postconditions",
    "evaluate_case",
    "first_attempt_id",
    "matrix_fingerprint",
    "observed_terminal",
    "snapshot_campaign",
    "table_markdown",
]

#: 判定表 schema
MATRIX_SCHEMA = "p5b_testbed_fault_matrix/v1"
#: 五条不变量（§4.3；顺序即报告顺序）
INVARIANTS: Tuple[str, ...] = (
    "no_double_booking",
    "unpublished_not_in_generation",
    "deterministic_recompute",
    "read_side_published_only",
    "parent_backup_verifiable",
)
#: 恢复后置条件（由落盘证据推导；``recovery`` 列 = 成立项排序拼接）
POSTCONDITIONS: Tuple[str, ...] = (
    "parent_restored",
    "staging_cleaned",
    "unpublished_shard_isolated",
    "unpublished_shard_failed",
    "unpublished_shard_quarantined",
    "projections_repaired",
    "no_redrive_of_published",
    "same_day_recomputed",
    "invalidated_registered",
    "advance_refused",
    "zero_write_scope",
)
#: 视为「派生投影缺口」的分类（修复后应清零；``bad_shard`` 属终态，不在此列）
PROJECTION_GAP_KINDS = frozenset(
    {
        "missing_day",
        "manifest_missing",
        "manifest_mismatch",
        "index_absent",
        "index_missing_day",
        "index_mismatch",
        "mixed_generation",
        "out_of_order",
        "unpublished_attempt_shard",
    }
)
#: 终态枚举（判定表列）
TERMINALS = ("ABORTED", "COMMITTED", "INVALIDATED", "REJECTED", "UNCHANGED", "REGRESSED")


@dataclass(frozen=True)
class FaultCase:
    """一行故障注入矩阵（§4.3「必须注入的故障点」逐项）。

    ``expected_postconditions`` 与 ``expected_terminal`` 是**期望**；执行侧不得回填，
    只能提供原始测量（``error_type`` / 生产根快照比对结果等）。
    """

    case_id: str
    group: str
    fault_point: str
    arm_id: str
    expected_terminal: str
    expected_postconditions: Tuple[str, ...]
    expected_generation_delta: int
    isolation: str
    description: str
    min_promoted: Optional[int] = None
    expected_gap_kinds: Tuple[str, ...] = ()
    expected_error_types: Tuple[str, ...] = ()
    requires_repair: bool = False

    @property
    def expected_recovery(self) -> str:
        """恢复动作的期望形态（供报告展示；判定用集合相等，不用字符串）。"""
        return "+".join(sorted(self.expected_postconditions))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "group": self.group,
            "fault_point": self.fault_point,
            "arm_id": self.arm_id,
            "min_promoted": self.min_promoted,
            "expected_terminal": self.expected_terminal,
            "expected_postconditions": list(self.expected_postconditions),
            "expected_recovery": self.expected_recovery,
            "expected_generation_delta": int(self.expected_generation_delta),
            "isolation": self.isolation,
            "expected_gap_kinds": list(self.expected_gap_kinds),
            "expected_error_types": list(self.expected_error_types),
            "requires_repair": bool(self.requires_repair),
            "description": self.description,
        }


_ABORT_POST = (
    "parent_restored",
    "staging_cleaned",
    "unpublished_shard_isolated",
    "projections_repaired",
    "no_redrive_of_published",
    "same_day_recomputed",
)
_COMMIT_POST = (
    "unpublished_shard_isolated",
    "projections_repaired",
    "no_redrive_of_published",
)
_REFUSE_POST = (
    "unpublished_shard_isolated",
    "projections_repaired",
    "no_redrive_of_published",
    "advance_refused",
)

#: §4.3 必须注入的故障点 + T3 追加项（顺序 = 判定表顺序）
MATRIX: Tuple[FaultCase, ...] = (
    FaultCase(
        case_id="T3-FM-00",
        group="提升阶段崩溃",
        fault_point="after_stage",
        arm_id="aa_2",
        expected_terminal="ABORTED",
        expected_postconditions=_ABORT_POST,
        expected_generation_delta=1,
        isolation="attempt",
        description="0 个 root 提升即中断（§4.3 状态表第 1 行：清理暂存、保留父代）",
    ),
    FaultCase(
        case_id="T3-FM-01",
        group="提升阶段崩溃",
        fault_point="after_promote",
        arm_id="aa_1",
        expected_terminal="ABORTED",
        expected_postconditions=_ABORT_POST,
        expected_generation_delta=1,
        isolation="attempt",
        description="第 1 个 root 提升后崩溃（部分 roots 已提升、未发布）",
        min_promoted=1,
    ),
    FaultCase(
        case_id="T3-FM-02",
        group="提升阶段崩溃",
        fault_point="after_promote",
        arm_id="aa_2",
        expected_terminal="ABORTED",
        expected_postconditions=_ABORT_POST,
        expected_generation_delta=1,
        isolation="attempt",
        description="第 2 个 root 提升后崩溃（三臂场景：2 已提升、1 仍为父代，统一回滚）",
        min_promoted=2,
    ),
    FaultCase(
        case_id="T3-FM-03",
        group="提升阶段崩溃",
        fault_point="after_promote_all",
        arm_id="",
        expected_terminal="ABORTED",
        expected_postconditions=_ABORT_POST,
        expected_generation_delta=1,
        isolation="attempt",
        description="全部 roots 提升但分片未写",
    ),
    FaultCase(
        case_id="T3-FM-04",
        group="分片阶段崩溃",
        fault_point="after_shard",
        arm_id="aa_1",
        expected_terminal="ABORTED",
        expected_postconditions=_ABORT_POST + ("unpublished_shard_quarantined",),
        expected_generation_delta=1,
        isolation="arm",
        description="只写部分臂分片（journal 尚未登记分片 ⇒ 未发布分片混入，须隔离后再重算）",
        expected_gap_kinds=("unpublished_attempt_shard",),
        requires_repair=True,
    ),
    FaultCase(
        case_id="T3-FM-05",
        group="分片阶段崩溃",
        fault_point="after_shards",
        arm_id="",
        expected_terminal="ABORTED",
        expected_postconditions=_ABORT_POST + ("unpublished_shard_failed",),
        expected_generation_delta=1,
        isolation="attempt",
        description="分片已写、manifest 未发布（发布点之前 ⇒ 未发布分片移入 failed/）",
    ),
    FaultCase(
        case_id="T3-FM-06",
        group="发布阶段崩溃",
        fault_point="after_publish",
        arm_id="",
        expected_terminal="COMMITTED",
        expected_postconditions=_COMMIT_POST,
        expected_generation_delta=1,
        isolation="attempt",
        description="唯一发布点已写但 manifest/index 未写（只补齐、不重跑）",
        expected_gap_kinds=("index_missing_day", "manifest_mismatch"),
    ),
    FaultCase(
        case_id="T3-FM-07",
        group="发布阶段崩溃",
        fault_point="after_manifests",
        arm_id="",
        expected_terminal="COMMITTED",
        expected_postconditions=_COMMIT_POST,
        expected_generation_delta=1,
        isolation="attempt",
        description="manifest 已发布、index 未写",
        expected_gap_kinds=("index_missing_day",),
    ),
    FaultCase(
        case_id="T3-FM-08",
        group="恢复中崩溃",
        fault_point="after_abort_restore",
        arm_id="",
        expected_terminal="ABORTED",
        expected_postconditions=_ABORT_POST,
        expected_generation_delta=1,
        isolation="attempt",
        description="回滚/恢复过程中再次崩溃（继续同一事务，不新建 attempt）",
    ),
    FaultCase(
        case_id="T3-FM-09",
        group="时钟与数据态",
        fault_point="virtual_date_regression",
        arm_id="",
        expected_terminal="REJECTED",
        expected_postconditions=_REFUSE_POST,
        expected_generation_delta=0,
        isolation="campaign",
        description="虚拟日期回退（请求日序倒置）",
        expected_error_types=("FamilyTxnError",),
    ),
    FaultCase(
        case_id="T3-FM-10",
        group="时钟与数据态",
        fault_point="data_fingerprint_drift",
        arm_id="",
        expected_terminal="REJECTED",
        expected_postconditions=_REFUSE_POST,
        expected_generation_delta=0,
        isolation="data_state",
        description="窗口数据指纹漂移（历史分区内容被改写）",
        expected_error_types=("CampaignInvalidError",),
    ),
    FaultCase(
        case_id="T3-FM-11",
        group="账本失效",
        fault_point="published_shard_mismatch",
        arm_id="aa_1",
        expected_terminal="INVALIDATED",
        expected_postconditions=(
            # 坏片属内容损坏：三方一致为假 ⇒ 不要求 projections_repaired
            "unpublished_shard_isolated",
            "no_redrive_of_published",
            "invalidated_registered",
            "advance_refused",
        ),
        expected_generation_delta=0,
        isolation="campaign",
        description="已发布分片内容身份不符（不可重建）⇒ INVALIDATED 终态 + 拒绝推进",
        expected_gap_kinds=("bad_shard",),
        requires_repair=True,
    ),
    FaultCase(
        case_id="T3-FM-12",
        group="边界",
        fault_point="production_zero_write",
        arm_id="",
        expected_terminal="COMMITTED",
        expected_postconditions=(
            "unpublished_shard_isolated",
            "projections_repaired",
            "no_redrive_of_published",
            "zero_write_scope",
        ),
        expected_generation_delta=1,
        isolation="campaign",
        description="生产根/数据资产零写入（含一次完整提交）",
    ),
)


@dataclass(frozen=True)
class ContainmentLevel:
    """故障影响面分级与升级判据（确定性动作，不含 SLA）。"""

    level: str
    triggers: Tuple[str, ...]
    terminal_state: str
    deterministic_action: str
    escalation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "triggers": list(self.triggers),
            "terminal_state": self.terminal_state,
            "deterministic_action": self.deterministic_action,
            "escalation": self.escalation,
            "no_sla_claim": True,
        }


#: containment 分级（§4.3 状态表 + T3 缺口分类的映射；升级阈值与代码实现一致）
CONTAINMENT: Tuple[ContainmentLevel, ...] = (
    ContainmentLevel(
        level="attempt",
        triggers=(
            "STAGED/DRIVING/DRIVEN/PROMOTING/PROMOTED/ARTIFACTS 阶段中断",
            "唯一发布点之后崩溃（已 COMMITTED）",
            "部分 roots 已提升（统一回滚，不部分前滚）",
        ),
        terminal_state="ABORTED / COMMITTED",
        deterministic_action="未发布 ⇒ 终止并从父代重算；已发布 ⇒ 只幂等补齐分片/manifest/index",
        escalation="父代备份不可验证 ⇒ 立即升级为 campaign 失效",
    ),
    ContainmentLevel(
        level="arm",
        triggers=(
            "单臂 root 缺位（两次 rename 之间中断）",
            "未发布 attempt 的单臂分片残留（journal 未登记该片）",
        ),
        terminal_state="ABORTED（该臂回到父代）",
        deterministic_action="该臂从可验证父代备份还原；残留分片移入 audit/quarantine/ 后才允许重算",
        escalation="**任一臂**父代备份不可验证或还原后内容身份不符 ⇒ 立即升级为 campaign 失效",
    ),
    ContainmentLevel(
        level="campaign",
        triggers=(
            "父代备份内容身份不符",
            "现场根与已发布代不一致",
            "已发布分片内容不可由 journal 唯一重建（bad_shard）",
            "journal 不可读 / index 声明了 journal 未确认的已发布代",
        ),
        terminal_state="CampaignInvalid / INVALIDATED",
        deterministic_action="停止推进虚拟时钟，登记终态，人工核对后重开 campaign",
        escalation="同一数据态下多个 campaign 同时失效 ⇒ 升级为数据态级",
    ),
    ContainmentLevel(
        level="data_state",
        triggers=(
            "窗口历史分区内容漂移（window_state_id 变化）",
            "试车场/复用件源码漂移（execution_digest 变化）",
        ),
        terminal_state="CampaignInvalid（拒绝推进）",
        deterministic_action="拒绝推进；重新登记 campaign 并重建 D0（不得就地修补）",
        escalation="无更高层级；按数据态整体隔离",
    ),
)


def matrix_fingerprint() -> str:
    """矩阵定义指纹（判定表与报告互链用；MATRIX 改动即变化）。"""
    return canonical_json_sha256([case.to_dict() for case in MATRIX])


# ---------------------------------------------------------------------------
# 现场快照（只读）
# ---------------------------------------------------------------------------


def _attempts(layout: CampaignLayout) -> List[Dict[str, Any]]:
    """读取全部 attempt（journal 不可读 ⇒ 空表，交由审计报 ``journal_corrupt``）。"""
    try:
        return [dict(item) for item in load_all_attempts(layout)]
    except (TestbedError, FamilyTxnError, OSError):
        return []


def _index_days(layout: CampaignLayout) -> List[Dict[str, Any]]:
    try:
        index = read_index(layout)
    except (FamilyTxnError, OSError):
        return []
    if index is None:
        return []
    return [
        {
            "virtual_date": str(entry.get("virtual_date")),
            "generation": int(entry.get("generation", -1)),
            "family_txn_id": entry.get("family_txn_id"),
        }
        for entry in index.get("days") or []
    ]


def _arm_roots(layout: CampaignLayout, arms: Sequence[str]) -> Dict[str, Optional[str]]:
    roots: Dict[str, Optional[str]] = {}
    for arm_id in arms:
        identity = optional_identity(layout.arm_root(arm_id))
        roots[arm_id] = None if identity is None else identity["content_sha"]
    return roots


def snapshot_campaign(layout: CampaignLayout, campaign_id: str) -> Dict[str, Any]:
    """现场快照（只读）：已发布头、index 日序、臂根身份、分片哈希、attempt 全量、审计摘要。"""
    audit = audit_ledger(layout, campaign_id)
    attempts = _attempts(layout)
    arms = sorted(
        {str(arm) for attempt in attempts for arm in (attempt.get("arms") or {})}
        | set(audit.get("arms") or {})
    )
    return {
        "campaign_id": campaign_id,
        "audit_status": audit["status"],
        "published_generation": audit["published"].get("published_generation"),
        "published_txn_id": audit["published"].get("published_txn_id"),
        "chain_digest": audit["chain"].get("chain_digest"),
        "index_days": _index_days(layout),
        "arms": arms,
        "arm_roots": _arm_roots(layout, arms),
        "shards": shard_sha_map(layout),
        "attempts": attempts,
        "gaps_by_class": dict(audit.get("gaps_by_class") or {}),
        "three_way_consistent": bool((audit.get("three_way") or {}).get("consistent")),
        "invalidated": list(audit.get("invalidated") or []),
    }


def _committed_by_day(snapshot: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(attempt["virtual_date"]): attempt
        for attempt in snapshot.get("attempts") or []
        if attempt.get("status") == "COMMITTED"
    }


def _published_dates(snapshot: Mapping[str, Any]) -> set:
    return {str(entry["virtual_date"]) for entry in snapshot.get("index_days") or []}


def _attempt_by_id(snapshot: Mapping[str, Any], attempt_id: str) -> Optional[Dict[str, Any]]:
    for attempt in snapshot.get("attempts") or []:
        if str(attempt.get("attempt_id")) == str(attempt_id):
            return attempt
    return None


def first_attempt_id(snapshot: Mapping[str, Any], target_date: str) -> str:
    """本次案例在该虚拟日创建的**首个** attempt（被注入的那一个）。"""
    for attempt in snapshot.get("attempts") or []:
        if str(attempt.get("virtual_date")) == str(target_date):
            return str(attempt.get("attempt_id") or "")
    return ""


# ---------------------------------------------------------------------------
# 终态（由证据推导）
# ---------------------------------------------------------------------------


def observed_terminal(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    target_date: str,
    error: str = "",
    attempt_id: str = "",
) -> str:
    """由证据推导**本次注入 attempt** 的终态（不信任执行侧自报）。

    终态口径 = 被注入 attempt 的状态（``ABORTED``/``COMMITTED``），而不是「该日最终是否发布」：
    未发布 attempt 终止后同日重算并发布，是 §4.3 规定的正常收敛路径。
    """
    if after.get("audit_status") == "INVALIDATED" or after.get("invalidated"):
        return "INVALIDATED"
    if attempt_id:
        attempt = _attempt_by_id(after, attempt_id)
        if attempt is not None:
            return "COMMITTED" if attempt.get("status") == "COMMITTED" else "ABORTED"
    before_generation = int(before.get("published_generation") or 0)
    after_generation = int(after.get("published_generation") or 0)
    delta = after_generation - before_generation
    if delta == 1:
        return "COMMITTED"
    if delta != 0:
        return "REGRESSED"
    if any(
        str(attempt.get("virtual_date")) == str(target_date)
        for attempt in after.get("attempts") or []
    ):
        return "ABORTED"
    if error:
        return "REJECTED"
    return "UNCHANGED"


# ---------------------------------------------------------------------------
# 恢复后置条件（由落盘证据推导）
# ---------------------------------------------------------------------------


def _post_parent_restored(
    layout: CampaignLayout, after: Mapping[str, Any], attempt_id: str
) -> bool:
    """父代是否已还原：同日后续 attempt 必须从**同一父代**重算；无后续则现场根须等于父代。"""
    injected = _attempt_by_id(after, attempt_id)
    if injected is None or str(injected.get("status")) != "ABORTED":
        return False
    date = str(injected["virtual_date"])
    later = [
        attempt
        for attempt in after.get("attempts") or []
        if str(attempt.get("virtual_date")) == date
        and str(attempt.get("attempt_id")) != str(attempt_id)
    ]
    arms = sorted(injected.get("arms") or {})
    if not arms:
        return False
    if later:
        return all(
            str((later[-1].get("arms") or {}).get(arm, {}).get("parent", {}).get("content_sha"))
            == str(injected["arms"][arm]["parent"]["content_sha"])
            for arm in arms
        )
    for arm_id in arms:
        identity = optional_identity(layout.arm_root(arm_id))
        if identity is None:
            return False
        if identity["content_sha"] != str(injected["arms"][arm_id]["parent"]["content_sha"]):
            return False
    return True


def _post_staging_cleaned(layout: CampaignLayout, after: Mapping[str, Any]) -> bool:
    """已终止 attempt 的暂存目录必须清理（无已终止 attempt ⇒ 无证据，判 False）。"""
    aborted = [
        attempt
        for attempt in after.get("attempts") or []
        if str(attempt.get("status")) == "ABORTED"
    ]
    if not aborted:
        return False
    for attempt in aborted:
        if layout.staging_attempt_dir(
            str(attempt["virtual_date"]), str(attempt["attempt_id"])
        ).exists():
            return False
    return True


def _post_shard_isolated(after: Mapping[str, Any]) -> bool:
    """账本内不得残留未发布 attempt 的分片（已发布日集合非空才算有证据）。"""
    published = _published_dates(after)
    if not published:
        return False
    for shards in (after.get("shards") or {}).values():
        if set(shards) - published:
            return False
    return True


def _post_shard_failed(layout: CampaignLayout, attempt_id: str) -> bool:
    """未发布分片是否落入失败现场 ``failed/<date>/<attempt>/shards/``。"""
    return bool(attempt_id) and any(
        path.is_file() for path in layout.failed_root.glob(f"*/{attempt_id}/shards/*/*.json")
    )


def _post_shard_quarantined(
    layout: CampaignLayout, attempt_id: str, arm_id: str, target_date: str
) -> bool:
    """未发布分片是否**精确**落入隔离区（臂 + 日），且该 attempt 未记入 failed 分片。"""
    from src.lazybull.v2.hosts.testbed.ledger_common import ledger_paths

    if not attempt_id or not arm_id:
        return False
    quarantined = ledger_paths(layout).quarantine_shard_path(arm_id, target_date).is_file()
    return quarantined and not _post_shard_failed(layout, attempt_id)


def _post_projections_repaired(after: Mapping[str, Any]) -> bool:
    """派生投影缺口是否清零（坏片属终态，不计入本项）。"""
    published = _published_dates(after)
    if not published:
        return False
    if not after.get("three_way_consistent"):
        return False
    counted = {
        kind
        for kind, count in (after.get("gaps_by_class") or {}).items()
        if count and kind in PROJECTION_GAP_KINDS
    }
    return not counted


def _post_no_redrive(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    """历史已发布日不得被重新驱动（事务 ID 与分片字节都不变；无历史日 ⇒ 无证据，判 False）。"""
    before_days = {str(entry["virtual_date"]): entry for entry in before.get("index_days") or []}
    if not before_days:
        return False
    after_days = {str(entry["virtual_date"]): entry for entry in after.get("index_days") or []}
    for date, entry in before_days.items():
        current = after_days.get(date)
        if current is None or current.get("family_txn_id") != entry.get("family_txn_id"):
            return False
        for arm_id, shards in (before.get("shards") or {}).items():
            if shards.get(date) != (after.get("shards") or {}).get(arm_id, {}).get(date):
                return False
    return True


def _post_same_day_recomputed(after: Mapping[str, Any], attempt_id: str, target_date: str) -> bool:
    """目标日是否由**新的** attempt 重算并发布。"""
    if not attempt_id:
        return False
    committed = _committed_by_day(after).get(str(target_date))
    return committed is not None and str(committed["attempt_id"]) != str(attempt_id)


def _post_invalidated_registered(layout: CampaignLayout, after: Mapping[str, Any]) -> bool:
    """检测到的 INVALIDATED 是否已登记（含 campaign 级 ``*`` 条目）。"""
    if not after.get("invalidated"):
        return False
    return not _invalidated_registration_problems(layout, after)


def _post_advance_refused(
    before: Mapping[str, Any], after: Mapping[str, Any], target_date: str
) -> bool:
    """推进是否被拒：代不变、目标日未发布、无该日 COMMITTED attempt。"""
    if int(after.get("published_generation") or 0) != int(before.get("published_generation") or 0):
        return False
    if str(target_date) in _published_dates(after):
        return False
    return not any(
        str(attempt.get("virtual_date")) == str(target_date)
        and attempt.get("status") == "COMMITTED"
        for attempt in after.get("attempts") or []
    )


def derive_postconditions(
    layout: CampaignLayout,
    after: Mapping[str, Any],
    case: FaultCase,
    before: Mapping[str, Any],
    *,
    target_date: str,
    attempt_id: str,
    measurements: Mapping[str, Any],
) -> Dict[str, bool]:
    """从落盘证据与原始测量推导全部后置条件（不读取任何期望值）。"""
    return {
        "parent_restored": _post_parent_restored(layout, after, attempt_id),
        "staging_cleaned": _post_staging_cleaned(layout, after),
        "unpublished_shard_isolated": _post_shard_isolated(after),
        "unpublished_shard_failed": _post_shard_failed(layout, attempt_id),
        "unpublished_shard_quarantined": _post_shard_quarantined(
            layout, attempt_id, case.arm_id, target_date
        ),
        "projections_repaired": _post_projections_repaired(after),
        "no_redrive_of_published": _post_no_redrive(before, after),
        "same_day_recomputed": _post_same_day_recomputed(after, attempt_id, target_date),
        "invalidated_registered": _post_invalidated_registered(layout, after),
        "advance_refused": _post_advance_refused(before, after, target_date),
        "zero_write_scope": measurements.get("production_roots_unchanged") is True,
    }


# ---------------------------------------------------------------------------
# 五条不变量（非真空下限 + 可由测试构造反例）
# ---------------------------------------------------------------------------


def _check_no_double_booking(
    before: Mapping[str, Any], after: Mapping[str, Any], *_unused: Any
) -> Tuple[bool, Dict[str, Any]]:
    """① 已发布账本不重复入账：代唯一连续、历史分片字节不变（**纯证据，不含期望值**）。

    代增量与矩阵期望的比对属于 ``passed`` 的独立条件（:func:`evaluate_case`），不混入本项，
    否则「期望写错」会同时污染不变量结论。
    """
    problems: List[str] = []
    days = after.get("index_days") or []
    generations = [int(entry["generation"]) for entry in days]
    dates = [str(entry["virtual_date"]) for entry in days]
    if not days:
        problems.append("无已发布日（本项不可判定 ⇒ 判失败，不允许空集合恒真）")
    if generations != list(range(1, len(generations) + 1)):
        problems.append(f"index 代序不连续或重复: {generations}")
    if len(set(dates)) != len(dates):
        problems.append(f"index 存在重复虚拟日: {dates}")
    committed = _committed_by_day(after)
    if len(committed) != len(days):
        problems.append(f"已发布日数 {len(days)} != COMMITTED attempt 数 {len(committed)}")
    compared = 0
    for arm_id, shards in (before.get("shards") or {}).items():
        for date, sha in shards.items():
            current = (after.get("shards") or {}).get(arm_id, {}).get(date)
            if current != sha:
                problems.append(f"历史分片被改写: {arm_id}/{date}")
            compared += 1
    if compared == 0 and (before.get("index_days") or []):
        problems.append("历史分片比对为空（本项不可判定 ⇒ 判失败）")
    delta = int(after.get("published_generation") or 0) - int(
        before.get("published_generation") or 0
    )
    if delta < 0:
        problems.append(f"已发布代回退（禁止）: {delta}")
    return not problems, {"problems": problems, "generation_delta": delta, "compared": compared}


def _check_unpublished(
    layout: CampaignLayout, after: Mapping[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """② 未发布 attempt 不混入逻辑代：读侧只含 COMMITTED 日，账本无未发布分片。"""
    problems: List[str] = []
    published = _published_dates(after)
    if not published:
        problems.append("无已发布日（本项不可判定 ⇒ 判失败）")
    committed = _committed_by_day(after)
    for entry in after.get("index_days") or []:
        date = str(entry["virtual_date"])
        if date not in committed:
            problems.append(f"index 含非 COMMITTED 日: {date}")
            continue
        if entry.get("family_txn_id") != committed[date]["family_txn_id"]:
            problems.append(f"index 事务 ID 与 journal 不符: {date}")
    strays = 0
    for arm_id, shards in (after.get("shards") or {}).items():
        leftovers = sorted(set(shards) - published)
        if leftovers:
            problems.append(f"账本残留未发布分片: {arm_id} {leftovers}")
            strays += len(leftovers)
    return not problems, {"problems": problems, "strays": strays}


def _shard_rebuild_problems(campaign_id: str, after: Mapping[str, Any]) -> Tuple[List[str], int]:
    """盘上分片与 journal 重建内容的逐片比对（跳过已登记 INVALIDATED 的片）。"""
    problems: List[str] = []
    invalidated = {
        (str(entry.get("arm_id")), str(entry.get("virtual_date")))
        for entry in after.get("invalidated") or []
    }
    committed = _committed_by_day(after)
    checked = 0
    for arm_id, shards in (after.get("shards") or {}).items():
        for date, sha in shards.items():
            if (str(arm_id), str(date)) in invalidated:
                continue
            attempt = committed.get(date)
            if attempt is None:
                continue
            try:
                rebuilt = shard_sha256(build_shard(campaign_id, attempt, arm_id))
            except FamilyTxnError as exc:
                problems.append(f"无法由 journal 重建分片 {arm_id}/{date}: {exc}")
                continue
            checked += 1
            if rebuilt != sha:
                problems.append(f"盘上分片与 journal 重建不一致: {arm_id}/{date}")
    return problems, checked


def _digest_instability(after: Mapping[str, Any]) -> List[str]:
    """同日同输入多次 attempt 的驱动摘要必须逐值一致。"""
    digests: Dict[Tuple[str, str], List[str]] = {}
    for attempt in after.get("attempts") or []:
        date = str(attempt.get("virtual_date"))
        for arm_id, payload in (attempt.get("arms") or {}).items():
            digest = (payload.get("evidence") or {}).get("day_digest")
            if digest:
                digests.setdefault((arm_id, date), []).append(str(digest))
    return sorted(
        f"{arm}/{date}" for (arm, date), values in digests.items() if len(set(values)) > 1
    )


def _check_determinism(campaign_id: str, after: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """③ 新 attempt 可确定重算：盘上分片 == journal 重建；同日多 attempt 摘要逐值一致。

    已登记 INVALIDATED 的（臂 + 日）不计入本项：其内容**已判定不可重建**，
    由「④ 读侧只见已发布代 + INVALIDATED 登记」负责隔离，不得用它反向否定链的确定性。
    """
    problems, checked = _shard_rebuild_problems(campaign_id, after)
    if checked == 0:
        problems.append("无可比对分片（本项不可判定 ⇒ 判失败）")
    unstable = _digest_instability(after)
    if unstable:
        problems.append(f"同日同输入重算结果不一致: {unstable}")
    return not problems, {"problems": problems, "checked": checked, "unstable": unstable}


def _check_read_side(
    layout: CampaignLayout, campaign_id: str, after: Mapping[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """④ 读侧只见 ``published_generation``。"""
    problems: List[str] = []
    committed = _committed_by_day(after)
    max_generation = max(
        (int(attempt["target_generation"]) for attempt in committed.values()), default=0
    )
    published = int(after.get("published_generation") or 0)
    if published < 1:
        problems.append("无已发布代（本项不可判定 ⇒ 判失败）")
    if published != max_generation:
        problems.append(f"published_generation {published} != 最大 COMMITTED 代 {max_generation}")
    for entry in after.get("index_days") or []:
        if int(entry["generation"]) > published:
            problems.append(f"index 含超出发布代的日: {entry['virtual_date']}")
    published_dates = _published_dates(after)
    queried = 0
    for arm_id in after.get("arms") or []:
        try:
            result = query_shards(layout, str(arm_id), allow_invalidated=True)
        except FamilyTxnError as exc:
            problems.append(f"只读查询失败 {arm_id}: {exc}")
            continue
        queried += 1
        returned = {str(day["virtual_date"]) for day in result["days"]}
        if not returned <= published_dates:
            problems.append(f"只读查询返回未发布日: {arm_id} {sorted(returned - published_dates)}")
    if queried == 0:
        problems.append("未执行任何只读查询（本项不可判定 ⇒ 判失败）")
    problems.extend(_invalidated_registration_problems(layout, after))
    return not problems, {
        "problems": problems,
        "published_generation": published,
        "queried_arms": queried,
    }


def _invalidated_registration_problems(
    layout: CampaignLayout, after: Mapping[str, Any]
) -> List[str]:
    """检测到的 INVALIDATED 必须已登记（含 campaign 级 ``*`` 条目）。"""
    detected = list(after.get("invalidated") or [])
    if not detected:
        return []
    registry = invalidated_days(layout)
    problems: List[str] = []
    for entry in detected:
        arm_id = str(entry.get("arm_id"))
        registered = registry.get(arm_id) or {}
        if not registered:
            problems.append(f"INVALIDATED 未登记: {arm_id}")
            continue
        date = entry.get("virtual_date")
        if date is not None and str(date) not in registered:
            problems.append(f"INVALIDATED 未登记: {arm_id}/{date}")
    return problems


def _check_backups(layout: CampaignLayout, after: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """⑤ 父代备份可验证：备份内容身份 == 登记父代；最后一次 attempt 被终止时臂已回父代。"""
    problems: List[str] = []
    checked = 0
    attempts = [dict(item) for item in after.get("attempts") or []]
    latest_by_day: Dict[str, str] = {
        str(attempt.get("virtual_date")): str(attempt.get("attempt_id")) for attempt in attempts
    }
    for attempt in attempts:
        date = str(attempt.get("virtual_date"))
        attempt_id = str(attempt.get("attempt_id"))
        status = str(attempt.get("status"))
        for arm_id, payload in (attempt.get("arms") or {}).items():
            parent = dict(payload.get("parent") or {})
            recorded = str(parent.get("content_sha"))
            backup_root = layout.backup_arm_root(date, attempt_id, arm_id)
            if backup_root.is_dir():
                identity = root_identity(backup_root)
                checked += 1
                if identity["content_sha"] != recorded:
                    problems.append(f"父代备份不可验证: {arm_id}@{date}/{attempt_id}")
            elif status == "COMMITTED":
                problems.append(f"已发布 attempt 缺父代备份: {arm_id}@{date}/{attempt_id}")
            if status == "ABORTED" and latest_by_day.get(date) == attempt_id:
                actual = optional_identity(layout.arm_root(arm_id))
                if actual is None or actual["content_sha"] != recorded:
                    problems.append(f"终止后臂未回到父代: {arm_id}@{date}/{attempt_id}")
    if checked == 0:
        problems.append("无父代备份被检（本项不可判定 ⇒ 判失败）")
    return not problems, {"problems": problems, "backups_checked": checked}


def check_invariants(
    layout: CampaignLayout,
    campaign_id: str,
    case: FaultCase,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Dict[str, Tuple[bool, Dict[str, Any]]]:
    """校验五条不变量（公开入口：测试可用手工构造的快照证明每项都会失败）。"""
    return {
        "no_double_booking": _check_no_double_booking(before, after, case),
        "unpublished_not_in_generation": _check_unpublished(layout, after),
        "deterministic_recompute": _check_determinism(campaign_id, after),
        "read_side_published_only": _check_read_side(layout, campaign_id, after),
        "parent_backup_verifiable": _check_backups(layout, after),
    }


def evaluate_case(
    layout: CampaignLayout,
    campaign_id: str,
    case: FaultCase,
    before: Mapping[str, Any],
    *,
    target_date: str,
    error_type: str = "",
    detected_gaps: Sequence[str] = (),
    injected_attempt_id: str = "",
    measurements: Optional[Mapping[str, Any]] = None,
    after: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """按一份现场与矩阵行生成判定表记录。

    终态、恢复后置条件、五不变量均由证据独立计算；执行侧只能提供原始测量
    （``error_type`` 与 ``measurements``），**不得**提供期望值。

    Args:
        layout: campaign 布局。
        campaign_id: campaign 标识。
        case: 矩阵行。
        before: 注入前快照（:func:`snapshot_campaign`）。
        target_date: 本次故障注入所在虚拟日（拒绝类为「被拒推进的那一日」）。
        error_type: 实际抛出的异常类名（白名单比对用；未抛异常为空串）。
        detected_gaps: 修复**之前**审计到的缺口分类（可修复缺口在修复后消失，需单独留证）。
        injected_attempt_id: 被注入的 attempt（执行侧从 journal 读取，不是期望值）。
        measurements: 原始测量值（如 ``production_roots_unchanged``）。
        after: 结束时快照（缺省 = 现场重算）。

    Returns:
        判定表记录：``terminal``/``recovery``/``postconditions``/``invariants``/``passed``。
    """
    after = snapshot_campaign(layout, campaign_id) if after is None else after
    measurements = dict(measurements or {})
    terminal = observed_terminal(
        before, after, target_date=target_date, error=error_type, attempt_id=injected_attempt_id
    )
    post = derive_postconditions(
        layout,
        after,
        case,
        before,
        target_date=target_date,
        attempt_id=injected_attempt_id,
        measurements=measurements,
    )
    checks = check_invariants(layout, campaign_id, case, before, after)
    observed_gaps = sorted(
        set(kind for kind, count in (after.get("gaps_by_class") or {}).items() if count)
        | {str(kind) for kind in detected_gaps}
    )
    observed_set = {name for name, value in post.items() if value}
    expected_set = set(case.expected_postconditions)
    error_ok = (not case.expected_error_types) or (
        str(error_type) in set(case.expected_error_types)
    )
    record = {
        "case_id": case.case_id,
        "group": case.group,
        "fault_point": case.fault_point,
        "arm_id": case.arm_id,
        "expected_terminal": case.expected_terminal,
        "terminal": terminal,
        "expected_recovery": "+".join(sorted(expected_set)),
        "recovery": "+".join(sorted(observed_set)),
        "expected_postconditions": sorted(expected_set),
        "observed_postconditions": sorted(observed_set),
        "postconditions": dict(post),
        "missing_postconditions": sorted(expected_set - observed_set),
        "unexpected_postconditions": sorted(observed_set - expected_set),
        "error_type": str(error_type),
        "expected_error_types": list(case.expected_error_types),
        "error_type_matched": bool(error_ok),
        "expected_generation_delta": case.expected_generation_delta,
        "generation_delta": int(after.get("published_generation") or 0)
        - int(before.get("published_generation") or 0),
        "isolation": case.isolation,
        "invariants": {name: bool(checks[name][0]) for name in INVARIANTS},
        "details": {name: dict(checks[name][1]) for name in INVARIANTS},
        "observed_gaps": observed_gaps,
        "missing_expected_gaps": sorted(set(case.expected_gap_kinds) - set(observed_gaps)),
    }
    record["passed"] = bool(
        record["terminal"] == case.expected_terminal
        and observed_set == expected_set
        and record["generation_delta"] == case.expected_generation_delta
        and record["error_type_matched"]
        and all(record["invariants"].values())
        and not record["missing_expected_gaps"]
    )
    return record


def build_verdict_table(
    results: Sequence[Mapping[str, Any]], *, campaign_id: str = ""
) -> Dict[str, Any]:
    """汇总判定表（含 containment 表与矩阵指纹；确定性、无墙钟）。"""
    rows = [dict(result) for result in results]
    from src.lazybull.v2.hosts.testbed.storage import seal_record

    return seal_record(
        {
            "schema": MATRIX_SCHEMA,
            "campaign_id": str(campaign_id),
            "scope": SCOPE,
            "evidence_class": EVIDENCE_CLASS,
            "virtual_clock": True,
            "not_forward_evidence": True,
            "not_sla_evidence": True,
            "not_real_candidate": True,
            "matrix_fingerprint": matrix_fingerprint(),
            "matrix_case_ids": [case.case_id for case in MATRIX],
            "invariants": list(INVARIANTS),
            "postconditions": list(POSTCONDITIONS),
            "cases": rows,
            "all_passed": all(bool(row.get("passed")) for row in rows) if rows else False,
            "passed_count": sum(1 for row in rows if row.get("passed")),
            "case_count": len(rows),
            "containment": [level.to_dict() for level in CONTAINMENT],
            "limitations": [
                "HISTORICAL_TESTBED / NOT_FORWARD_EVIDENCE：历史虚拟时钟试车场，不是前向影子",
                "本表只证明已注入故障点的恢复确定性，不构成 SLA、恢复时限或可用性承诺",
                "收益/回撤裁判与端到端演练归 T4/T5/T6，本表不含任何裁判结论",
            ],
        }
    )


def table_markdown(results: Sequence[Mapping[str, Any]]) -> str:
    """判定表 Markdown 渲染（验收工具导出用）。"""
    header = (
        "| 用例 | 注入点 | 终态（期望/实际） | 恢复后置条件（实测） | "
        + " | ".join(INVARIANTS)
        + " | 判定 |"
    )
    divider = "|" + "---|" * (4 + len(INVARIANTS) + 1)
    lines = [header, divider]
    for row in results:
        marks = ["PASS" if row["invariants"].get(name) else "FAIL" for name in INVARIANTS]
        lines.append(
            "| {case} | {point} | {expected}/{actual} | {recovery} | {marks} | {verdict} |".format(
                case=row["case_id"],
                point=row["fault_point"],
                expected=row["expected_terminal"],
                actual=row["terminal"],
                recovery=row["recovery"],
                marks=" | ".join(marks),
                verdict="通过" if row.get("passed") else "失败",
            )
        )
    return "\n".join(lines)
