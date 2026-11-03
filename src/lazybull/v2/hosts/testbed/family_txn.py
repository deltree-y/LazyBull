# -*- coding: utf-8 -*-
"""P5b 试车场 T2 族级事务：暂存 / 提升 / 唯一发布点 / 拒绝部分前滚。

契约来源：设计 v4 §4.3（族级事务、0/部分/全部 roots 提升状态表、恢复中再崩溃）。

状态机（journal attempt ``status``）::

    STAGED → DRIVING → DRIVEN → PROMOTING → PROMOTED → ARTIFACTS → COMMITTED
                        └────────── 任一未发布状态中断 ──────────┐
                                                               ▼
                                                    ABORTING → ABORTED

三条不可让渡的口径
==================
1. **唯一发布点** = journal 记录转 ``COMMITTED`` 的那一次原子写（写入
   ``published_generation`` 与 ``published_txn_id``）。index / manifest / 分片都是派生投影，
   可反复重建，重建与首写逐字节一致（确定性载荷）。
2. **未发布 attempt 不部分前滚**：一旦中断即 ``ABORTING→ABORTED``；已提升的新 root 移入
   失败现场 ``failed/<date>/<attempt>/roots/``，需要恢复的臂从**可验证父代备份**还原
   （备份内容身份复算必须等于登记父代 ``content_sha``）。父代不可验证 ⇒
   :class:`CampaignInvalidError`（campaign 失效，禁止拼接或猜测）。
3. **已发布只做幂等补齐**：``COMMITTED`` 之后只重建分片/manifest/index，任何情况下不重新
   驱动交易、不重复入账。

``ABORTING`` 中再次崩溃：重启后 :func:`recover_campaign` 继续同一 attempt 的回滚（不新建
attempt、不回滚已发布代）。``ABORTED`` 之后允许新的 ``family_txn_id/attempt_id`` 在同一虚拟
日从父代重算（虚拟日期不是事务唯一键）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.hosts.testbed.journal import (
    TERMINAL_STATUSES,
    add_attempt,
    build_shard,
    find_attempt,
    load_day_journal,
    load_manifest,
    load_shard,
    read_index,
    rebuild_index,
    rebuild_manifest,
    save_day_journal,
    shard_sha256,
    write_shard,
)
from src.lazybull.v2.hosts.testbed.layout import (
    CampaignInvalidError,
    CampaignLayout,
    FamilyTxnError,
    copy_root,
    gen_id,
    move_root,
    optional_identity,
    remove_root,
    root_identity,
)
from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.registration import list_arms, load_campaign
from src.lazybull.v2.hosts.testbed.storage import read_json

__all__ = [
    "FAULT_POINTS",
    "FamilyTransaction",
    "InjectedFault",
    "abort_attempt",
    "cache_manifests",
    "open_family_transaction",
    "persist_attempt",
    "recover_campaign",
    "top_up_if_needed",
    "top_up_published",
]

#: 故障注入点（测试注入用；生产 None，不插桩不写盘）
#:
#: T3 追加（向后兼容，不改变任何既有注入点的位置与语义）：
#: - ``after_promote_all``：全部臂提升完成、状态仍为 ``PROMOTING``（分片未写）；
#: - ``after_shard``：单臂分片已写但 ``shards`` 尚未登记进 journal（**未发布分片混入**的构造点）；
#: - ``after_manifests``：已发布且清单已写、index 尚未写（``manifest 已发布 index 未写``）。
FAULT_POINTS: Tuple[str, ...] = (
    "after_backup",
    "after_stage",
    "after_drive",
    "before_promote",
    "mid_promote",
    "after_promote",
    "after_promote_all",
    "after_shard",
    "after_shards",
    "after_publish",
    "after_manifests",
    "abort_begin",
    "after_abort_restore",
    "abort_done",
)
#: 故障钩子：``(注入点, 现场载荷) -> None``；抛异常即模拟进程崩溃
FaultHook = Callable[[str, Mapping[str, Any]], None]


class InjectedFault(RuntimeError):
    """故障注入（模拟进程崩溃）：调用方**不得**把它当作可清理的业务异常自动 abort。"""

    __test__ = False


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _unique_path(path: Path) -> Path:
    """失败现场目标名去重（同一臂多次搬运不覆盖既有现场）。"""
    candidate = Path(path)
    index = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}-{index}")
        index += 1
    return candidate


def persist_attempt(layout: CampaignLayout, attempt: Mapping[str, Any]) -> Dict[str, Any]:
    """把 attempt 当前状态整体回写当日 journal（唯一写入原语；终态清 active）。"""
    doc = load_day_journal(layout, str(attempt["virtual_date"]))
    attempt_id = str(attempt["attempt_id"])
    attempts = [
        dict(attempt) if item.get("attempt_id") == attempt_id else item
        for item in doc.get("attempts") or []
    ]
    if not any(item.get("attempt_id") == attempt_id for item in attempts):
        attempts.append(dict(attempt))
    terminal = str(attempt.get("status")) in TERMINAL_STATUSES
    doc = {
        **doc,
        "attempt_seq": len(attempts),
        "attempts": attempts,
        "active_attempt_id": None if terminal else attempt_id,
    }
    save_day_journal(layout, doc)
    return dict(attempt)


def _verified_backup_identity(
    layout: CampaignLayout, attempt: Mapping[str, Any], arm_id: str
) -> Dict[str, Any]:
    """复算父代备份内容身份，必须等于登记父代 ``content_sha``（否则 campaign 失效）。"""
    parent = dict(attempt["arms"][arm_id]["parent"])
    backup_root = layout.backup_arm_root(
        str(attempt["virtual_date"]), str(attempt["attempt_id"]), arm_id
    )
    if not backup_root.is_dir():
        raise CampaignInvalidError(
            f"父代备份缺失，campaign 失效（禁止拼接或猜测）: 臂 {arm_id!r} "
            f"attempt={attempt['attempt_id']!r} 备份={backup_root}"
        )
    identity = root_identity(backup_root)
    if identity["content_sha"] != parent["content_sha"]:
        raise CampaignInvalidError(
            f"父代备份不可验证（内容身份不符），campaign 失效: 臂 {arm_id!r} "
            f"备份={identity['content_sha']} 登记父代={parent['content_sha']}"
        )
    if identity["file_count"] != parent.get("file_count"):
        raise CampaignInvalidError(
            f"父代备份文件数与登记不符，campaign 失效: 臂 {arm_id!r} "
            f"备份={identity['file_count']} 登记父代={parent.get('file_count')}"
        )
    return identity


def _restore_arm(
    layout: CampaignLayout,
    attempt: Mapping[str, Any],
    arm_id: str,
    *,
    fault_hook: Optional[FaultHook],
) -> str:
    """单臂回滚：新 root 移入失败现场 → 从可验证父代备份还原（幂等可重入）。"""
    virtual_date = str(attempt["virtual_date"])
    attempt_id = str(attempt["attempt_id"])
    parent_sha = str(attempt["arms"][arm_id]["parent"]["content_sha"])
    _verified_backup_identity(layout, attempt, arm_id)
    arm_root = layout.arm_root(arm_id)
    layout.clean_restore_leftovers(arm_id)
    current = optional_identity(arm_root)
    if current is not None and current["content_sha"] == parent_sha:
        return "unchanged"
    if current is not None:
        target = _unique_path(
            layout.failed_arm_root(virtual_date, attempt_id, arm_id, slot="roots")
        )
        move_root(arm_root, target)
    backup_root = layout.backup_arm_root(virtual_date, attempt_id, arm_id)
    restore_tmp = arm_root.with_name(f"{arm_id}.restore-tmp")
    remove_root(restore_tmp)
    copy_root(backup_root, restore_tmp)
    if root_identity(restore_tmp)["content_sha"] != parent_sha:
        remove_root(restore_tmp)
        raise CampaignInvalidError(
            f"父代备份还原后内容身份不符，campaign 失效: 臂 {arm_id!r}（{restore_tmp}）"
        )
    move_root(restore_tmp, arm_root)
    if fault_hook is not None:
        fault_hook("after_abort_restore", {"arm_id": arm_id, "virtual_date": virtual_date})
    return "restored"


def abort_attempt(
    layout: CampaignLayout,
    attempt: Mapping[str, Any],
    *,
    reason: str,
    fault_hook: Optional[FaultHook] = None,
) -> Dict[str, Any]:
    """未发布 attempt 终止：``ABORTING→ABORTED``，含部分 roots 提升回滚（幂等可重入）。"""
    if str(attempt.get("status")) == "COMMITTED":
        raise FamilyTxnError(
            f"已发布 attempt 不得回滚（{attempt.get('family_txn_id')!r}）；"
            "已发布事务只做分片/manifest/index 幂等补齐"
        )
    working = dict(attempt)
    working["arms"] = {
        arm_id: dict(payload) for arm_id, payload in (attempt.get("arms") or {}).items()
    }
    if working.get("status") != "ABORTING":
        working = {
            **working,
            "status": "ABORTING",
            "abort": {
                "reason": str(reason),
                "from_status": working.get("status"),
                "begun_at": _now(),
            },
        }
        persist_attempt(layout, working)
        if fault_hook is not None:
            fault_hook("abort_begin", {"attempt_id": working["attempt_id"], "reason": reason})
    promoted: Dict[str, str] = {}
    for arm_id in sorted(working["arms"]):
        state = _restore_arm(layout, working, arm_id, fault_hook=fault_hook)
        working["arms"][arm_id]["promote_state"] = (
            "restored" if state == "restored" else "unchanged"
        )
        promoted[arm_id] = state
        persist_attempt(layout, working)
    _forget_unpublished_shards(layout, working)
    remove_root(
        layout.staging_attempt_dir(str(working["virtual_date"]), str(working["attempt_id"]))
    )
    for arm_id in sorted(working["arms"]):
        parent_sha = str(working["arms"][arm_id]["parent"]["content_sha"])
        actual = optional_identity(layout.arm_root(arm_id))
        if actual is None or actual["content_sha"] != parent_sha:
            raise CampaignInvalidError(
                f"回滚后臂 {arm_id!r} 未回到父代（campaign 失效，禁止拼接）: "
                f"{None if actual is None else actual['content_sha']} vs {parent_sha}"
            )
    working = {
        **working,
        "status": "ABORTED",
        "finished_at": _now(),
        "promoted_count": 0,
        "abort": {
            **dict(working.get("abort") or {}),
            "finished_at": _now(),
            "arms": promoted,
            "staging_cleaned": True,
        },
    }
    persist_attempt(layout, working)
    if fault_hook is not None:
        fault_hook("abort_done", {"attempt_id": working["attempt_id"]})
    return {
        "virtual_date": working["virtual_date"],
        "family_txn_id": working["family_txn_id"],
        "attempt_id": working["attempt_id"],
        "action": "abort",
        "status": "ABORTED",
        "reason": str(reason),
        "arms": promoted,
    }


def _forget_unpublished_shards(layout: CampaignLayout, attempt: Mapping[str, Any]) -> List[str]:
    """把本 attempt 已写但未发布的分片移入失败现场（读侧永不把它当正式输入）。"""
    moved: List[str] = []
    virtual_date = str(attempt["virtual_date"])
    attempt_id = str(attempt["attempt_id"])
    for arm_id in sorted(attempt.get("shards") or {}):
        shard_path = layout.shard_path(arm_id, virtual_date)
        if not shard_path.is_file():
            continue
        target = _unique_path(
            layout.attempt_dir(layout.failed_root, virtual_date, attempt_id)
            / "shards"
            / arm_id
            / shard_path.name
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        move_root(shard_path, target)
        moved.append(target.as_posix())
    return moved


def top_up_published(
    layout: CampaignLayout,
    campaign_id: str,
    attempt: Mapping[str, Any],
    *,
    fault_hook: Optional[FaultHook] = None,
) -> Dict[str, Any]:
    """已发布事务的幂等补齐：分片/manifest/index（**不**重新驱动交易、不重复入账）。

    Args:
        fault_hook: 故障注入钩子（测试用；``after_manifests`` 用于构造「清单已写、index 未写」）。
    """
    virtual_date = str(attempt["virtual_date"])
    restored: List[str] = []
    for arm_id in sorted(attempt.get("arms") or {}):
        expected = build_shard(campaign_id, attempt, arm_id)
        existing = load_shard(layout, arm_id, virtual_date)
        if existing is None:
            write_shard(layout, campaign_id, attempt, arm_id)
            restored.append(arm_id)
            continue
        if shard_sha256(existing) != shard_sha256(expected):
            raise CampaignInvalidError(
                f"已发布分片内容身份不符（失效，禁止重跑覆盖）: 臂 {arm_id!r} 日 {virtual_date}"
            )
    manifests = {
        arm_id: rebuild_manifest(layout, campaign_id, arm_id)["written"]
        for arm_id in sorted(attempt.get("arms") or {})
    }
    if fault_hook is not None:
        fault_hook(
            "after_manifests",
            {"virtual_date": virtual_date, "family_txn_id": attempt["family_txn_id"]},
        )
    index_result = rebuild_index(layout, campaign_id)
    return {
        "virtual_date": virtual_date,
        "family_txn_id": attempt["family_txn_id"],
        "action": "top_up",
        "status": "COMMITTED",
        "shards_restored": restored,
        "manifests_written": [arm_id for arm_id, written in manifests.items() if written],
        "index_written": bool(index_result["written"]),
    }


def _index_complete(
    index: Optional[Mapping[str, Any]], committed: Sequence[Mapping[str, Any]]
) -> bool:
    """index 是否与 journal 已发布 attempt 一一对应（日序 / 代 / 事务 / 发布头）。"""
    if index is None:
        return False
    ordered = sorted(committed, key=lambda item: int(item["target_generation"]))
    days = list(index.get("days") or [])
    if len(days) != len(ordered):
        return False
    for day, attempt in zip(days, ordered):
        if str(day.get("virtual_date")) != str(attempt["virtual_date"]):
            return False
        if int(day.get("generation", -1)) != int(attempt["target_generation"]):
            return False
        if str(day.get("family_txn_id")) != str(attempt["family_txn_id"]):
            return False
    if not ordered:
        return int(index.get("published_generation", 0)) == 0
    head = ordered[-1]
    if int(index.get("published_generation", 0)) != int(head["target_generation"]):
        return False
    # 顶层发布点字段也必须与末条日条目一致（否则报告的唯一发布点会被顶字段污染）
    if str(index.get("published_txn_id")) != str(head["family_txn_id"]):
        return False
    return str(index.get("published_virtual_date")) == str(head["virtual_date"])


def _manifest_complete(
    expected: Mapping[str, Mapping[str, Any]], manifest: Optional[Mapping[str, Any]]
) -> bool:
    """单臂清单是否与 journal 期望逐日一致（条目集 / 分片 sha / 代 / 头代）。"""
    if manifest is None:
        return False
    shards = dict(manifest.get("shards") or {})
    if set(shards) != set(expected):
        return False
    expected_head = max((int(item["generation"]) for item in expected.values()), default=0)
    if int(manifest.get("head_generation", -1)) != expected_head:
        return False
    for date, item in expected.items():
        entry = shards[date]
        if str(entry.get("sha256")) != str(item["sha256"]):
            return False
        if int(entry.get("generation", -1)) != int(item["generation"]):
            return False
    return True


def _shard_files_complete(
    layout: CampaignLayout, expected: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> bool:
    """分片文件是否在场且内容身份与 journal 重建一致（逐片只读）。"""
    for arm_id, dates in expected.items():
        for date, item in dates.items():
            path = layout.shard_path(arm_id, date)
            if not path.is_file():
                return False
            try:
                shard = read_json(path, what=f"分片 {arm_id}/{date}")
            except TestbedError:
                return False
            if shard_sha256(shard) != str(item["sha256"]):
                return False
    return True


def _attempt_complete(
    layout: CampaignLayout,
    attempt: Mapping[str, Any],
    index: Mapping[str, Any],
    manifests: Mapping[str, Optional[Mapping[str, Any]]],
) -> bool:
    """单条已发布 attempt 的三个派生投影是否都已就位（逐日探测，不读 journal）。"""
    date = str(attempt["virtual_date"])
    day = next(
        (entry for entry in index.get("days") or [] if str(entry.get("virtual_date")) == date),
        None,
    )
    if day is None or str(day.get("family_txn_id")) != str(attempt["family_txn_id"]):
        return False
    for arm_id in sorted(attempt.get("arms") or {}):
        entry = (day.get("arms") or {}).get(arm_id) or {}
        path = layout.shard_path(arm_id, date)
        if not path.is_file():
            return False
        try:
            shard = read_json(path, what=f"分片 {arm_id}/{date}")
        except TestbedError:
            return False
        if shard_sha256(shard) != str(entry.get("shard_sha256")):
            return False
        manifest = manifests.get(arm_id)
        if manifest is None or date not in (manifest.get("shards") or {}):
            return False
    return True


def cache_manifests(
    layout: CampaignLayout,
    attempt: Mapping[str, Any],
    cache: Dict[str, Optional[Dict[str, Any]]],
) -> None:
    """按臂缓存分片清单（避免逐 attempt 重读大文件）。"""
    for arm_id in sorted(attempt.get("arms") or {}):
        if arm_id in cache:
            continue
        try:
            cache[arm_id] = load_manifest(layout, arm_id)
        except (TestbedError, FamilyTxnError):
            cache[arm_id] = None


def top_up_if_needed(
    layout: CampaignLayout,
    campaign_id: str,
    attempt: Mapping[str, Any],
    *,
    index: Optional[Mapping[str, Any]],
    manifests: Dict[str, Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """已发布 attempt：投影完整则零动作（幂等）；否则幂等补齐并返回动作摘要。

    ``index``/``manifests`` 由调用方缓存并按需失效；命中完整时**不读 journal**，
    因此长窗口下幂等重跑是线性成本（旧实现逐日 ``top_up_published`` ⇒ O(A·N²)）。
    """
    if index is not None and _attempt_complete(layout, attempt, index, manifests):
        return None
    # 恢复期的幂等补齐不挂故障钩子：``after_manifests`` 只属于发布路径，
    # 否则「已发布日的例行补齐」会误触发注入点（跨日误伤）。
    result = top_up_published(layout, campaign_id, attempt)
    if not (result["shards_restored"] or result["manifests_written"] or result["index_written"]):
        return None
    for arm_id in sorted(attempt.get("arms") or {}):
        manifests.pop(arm_id, None)  # 清单已重写 ⇒ 缓存失效
    return result


def _expected_shards(
    campaign_id: str, committed: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """journal 唯一决定的「每臂每日应有什么分片」（与审计侧同口径）。"""
    expected: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for attempt in sorted(committed, key=lambda item: int(item["target_generation"])):
        for arm_id in sorted(attempt.get("arms") or {}):
            shard = build_shard(campaign_id, attempt, arm_id)
            expected.setdefault(arm_id, {})[str(attempt["virtual_date"])] = {
                "generation": int(attempt["target_generation"]),
                "sha256": shard_sha256(shard),
            }
    return expected


def _all_projections_complete(
    layout: CampaignLayout,
    campaign_id: str,
    committed: Sequence[Mapping[str, Any]],
) -> bool:
    """已发布代的三个派生投影是否已**逐日逐臂**完整（含清单头代/多余条目、index 日序/发布头）。

    任一项不成立即整批走幂等补齐路径（宁可多修，不可漏修）。
    """
    try:
        index = read_index(layout)
    except TestbedError:
        return False
    if not _index_complete(index, committed):
        return False
    expected = _expected_shards(campaign_id, committed)
    for arm_id, dates in expected.items():
        try:
            manifest = load_manifest(layout, arm_id)
        except (TestbedError, FamilyTxnError):
            return False
        if not _manifest_complete(dates, manifest):
            return False
    return _shard_files_complete(layout, expected)


def _safe_index(layout: CampaignLayout) -> Optional[Dict[str, Any]]:
    """读取读侧 index；不可读（截断/封存破坏）⇒ None（交由补齐路径重建）。"""
    try:
        return read_index(layout)
    except (TestbedError, FamilyTxnError):
        return None


def _collect_attempts(
    layout: CampaignLayout,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """一次遍历 journal，分离已发布 attempt 与待终止 attempt。"""
    committed: List[Mapping[str, Any]] = []
    pending: List[Mapping[str, Any]] = []
    for path in sorted(layout.journal_root.glob("*.json")):
        doc = load_day_journal(layout, path.stem)
        for attempt in doc.get("attempts") or []:
            status = str(attempt.get("status"))
            if status == "COMMITTED":
                committed.append(attempt)
            elif status not in TERMINAL_STATUSES:
                pending.append(attempt)
    return committed, pending


def _top_up_committed(
    layout: CampaignLayout,
    campaign_id: str,
    committed: Sequence[Mapping[str, Any]],
    applied: List[Dict[str, Any]],
) -> None:
    """逐 attempt **无条件**幂等补齐（仅在整批完备性判定失败时进入）。

    刻意不复用 ``top_up_if_needed`` 的逐日探测：整批判定失败意味着至少有一处投影不一致
    （例如 index 发布代落后、清单多余条目），逐日窄探测会误判「该日完整」而漏修。
    此路径的代价是一次全量重建，仅在损坏现场发生。
    """
    for attempt in sorted(committed, key=lambda item: int(item["target_generation"])):
        result = top_up_published(layout, campaign_id, attempt)
        if result["shards_restored"] or result["manifests_written"] or result["index_written"]:
            applied.append(result)


def recover_campaign(
    layout: CampaignLayout, campaign_id: str, *, fault_hook: Optional[FaultHook] = None
) -> Dict[str, Any]:
    """启动恢复：处理全部非终态 attempt（已发布 ⇒ 幂等补齐；未发布 ⇒ 终止回滚）。

    幂等：重复调用在干净现场上不产生任何写盘（``applied`` 为空）。
    性能：先做一次**逐日逐臂投影完备性判定**（:func:`_all_projections_complete`，只读分片/
    清单/index），全量完整即整批零动作——长窗口下不再逐 attempt 全量重建（旧实现 O(A·N²) 读盘）。
    """
    if not layout.journal_root.is_dir():
        return {"campaign_id": campaign_id, "applied": [], "published_generation": 0}
    committed, pending = _collect_attempts(layout)
    applied: List[Dict[str, Any]] = [
        abort_attempt(layout, attempt, reason="startup_recovery", fault_hook=fault_hook)
        for attempt in pending
    ]
    if committed and not _all_projections_complete(layout, campaign_id, committed):
        _top_up_committed(layout, campaign_id, committed, applied)
    index = _safe_index(layout)
    return {
        "campaign_id": campaign_id,
        "applied": applied,
        "published_generation": 0 if index is None else int(index["published_generation"]),
    }


class FamilyTransaction:
    """一次虚拟日的族级事务（一个 attempt 一个 ``family_txn_id``）。

    Args:
        layout: campaign 布局。
        campaign_id: campaign 标识（与注册记录一致）。
        arms: 臂规格（``arm_id → spec``，来自 :func:`load_arm_specs`）。
        virtual_date: 目标虚拟日（**不是**事务唯一键）。
        fault_hook: 故障注入钩子（测试用；生产 None）。
    """

    def __init__(
        self,
        layout: CampaignLayout,
        campaign_id: str,
        arms: Mapping[str, Any],
        virtual_date: str,
        *,
        fault_hook: Optional[FaultHook] = None,
    ) -> None:
        self.layout = layout
        self.campaign_id = campaign_id
        self.arms = dict(arms)
        self.virtual_date = str(virtual_date)
        self.fault_hook = fault_hook
        self.attempt: Dict[str, Any] = {}
        self.parent_identities: Dict[str, Dict[str, Any]] = {}

    # ---- 阶段 0：父代核验 + 备份 + 暂存 ----
    def open(self) -> Dict[str, Any]:
        """建立 attempt：父代身份核验 → 父代备份 → 暂存根复制（journal ``STAGED``）。"""
        index = read_index(self.layout)
        if index is None:
            raise FamilyTxnError(
                f"campaign {self.campaign_id!r} 缺已发布 index（先建立 genesis：D0 fresh 起点）"
            )
        published_generation = int(index["published_generation"])
        published_arms = index.get("arms") or {}
        doc = load_day_journal(self.layout, self.virtual_date)
        seq = int(doc.get("attempt_seq") or 0) + 1
        attempt_id = f"att-{self.virtual_date}-{seq:04d}"
        family_txn_id = f"txn-{self.virtual_date}-{seq:04d}-{gen_id(published_generation)}"
        arms_payload: Dict[str, Any] = {}
        for arm_id in sorted(self.arms):
            spec = self.arms[arm_id]
            identity = optional_identity(self.layout.arm_root(arm_id))
            if identity is None:
                raise CampaignInvalidError(
                    f"臂 {arm_id!r} 的根不存在（无法确定父代）: {self.layout.arm_root(arm_id)}"
                )
            recorded = published_arms.get(arm_id) or {}
            if identity["content_sha"] != recorded.get("content_sha"):
                raise CampaignInvalidError(
                    f"臂 {arm_id!r} 的现场根与已发布代不一致（禁止在未恢复现场上开新 attempt）: "
                    f"现场={identity['content_sha']} 已发布={recorded.get('content_sha')}"
                )
            self.parent_identities[arm_id] = identity
            arms_payload[arm_id] = {
                "arm_role": str(spec.arm_role),
                "parent": {
                    "gen_id": gen_id(published_generation),
                    "content_sha": identity["content_sha"],
                    "file_count": identity["file_count"],
                },
                "staged": None,
                "promote_state": "pending",
                "evidence": None,
            }
        self.attempt = {
            "family_txn_id": family_txn_id,
            "attempt_id": attempt_id,
            "virtual_date": self.virtual_date,
            "parent_generation": published_generation,
            "target_generation": published_generation + 1,
            "status": "STAGED",
            "started_at": _now(),
            "finished_at": None,
            "arms": arms_payload,
            "promoted_count": 0,
            "published_generation": None,
            "published_txn_id": None,
            "shards": {},
            "artifacts": {},
            "faults": [],
            "abort": None,
        }
        add_attempt(self.layout, self.virtual_date, self.attempt)
        # 两段式：**先把全部臂的父代备份做完**，再逐臂建暂存根。
        # 保证「暂存阶段任意位置崩溃」时全部臂的父代备份都可验证（否则 abort 会因后续臂
        # 备份缺失而升级为 campaign 失效），即 §4.3 的「0 roots 提升」状态可确定回滚。
        for arm_id in sorted(self.arms):
            self._make_backup(arm_id)
        for arm_id in sorted(self.arms):
            self._make_staged_root(arm_id)
        return self.attempt

    def _make_backup(self, arm_id: str) -> None:
        backup_root = self.layout.backup_arm_root(
            self.virtual_date, self.attempt["attempt_id"], arm_id
        )
        copy_root(self.layout.arm_root(arm_id), backup_root)
        _verified_backup_identity(self.layout, self.attempt, arm_id)
        self._fire("after_backup", {"arm_id": arm_id})

    def _make_staged_root(self, arm_id: str) -> None:
        staged_root = self.staged_arm_root(arm_id)
        copy_root(self.layout.arm_root(arm_id), staged_root)
        identity = root_identity(staged_root)
        parent_sha = self.attempt["arms"][arm_id]["parent"]["content_sha"]
        if identity["content_sha"] != parent_sha:
            raise FamilyTxnError(
                f"暂存根复制后内容身份与父代不一致: 臂 {arm_id!r} "
                f"{identity['content_sha']} vs {parent_sha}"
            )
        self._fire("after_stage", {"arm_id": arm_id})

    def staged_arm_root(self, arm_id: str) -> Path:
        return self.layout.staged_arm_root(
            self.virtual_date, str(self.attempt["attempt_id"]), arm_id
        )

    def _fire(self, point: str, payload: Mapping[str, Any]) -> None:
        """故障注入点：先落 journal（留痕），再由钩子抛异常模拟崩溃。"""
        if self.fault_hook is None:
            return
        entry = {"point": point, "at": _now(), **dict(payload)}
        self.attempt = {**self.attempt, "faults": [*self.attempt["faults"], entry]}
        persist_attempt(self.layout, self.attempt)
        self.fault_hook(point, dict(payload))

    # ---- 阶段 1：逐臂驱动证据 ----
    def record_driven(self, arm_id: str, evidence: Mapping[str, Any]) -> Dict[str, Any]:
        """登记单臂驱动证据与新根内容身份（父代未被触碰，仍在暂存根上）。"""
        staged_root = self.staged_arm_root(arm_id)
        identity = root_identity(staged_root)
        arms = {key: dict(value) for key, value in self.attempt["arms"].items()}
        arms[arm_id] = {
            **arms[arm_id],
            "staged": {
                "gen_id": gen_id(int(self.attempt["target_generation"])),
                "content_sha": identity["content_sha"],
                "file_count": identity["file_count"],
            },
            "evidence": dict(evidence),
        }
        self.attempt = {**self.attempt, "status": "DRIVING", "arms": arms}
        persist_attempt(self.layout, self.attempt)
        self._fire("after_drive", {"arm_id": arm_id})
        return self.attempt

    def finish_driving(self) -> Dict[str, Any]:
        """全部臂驱动完成：缺证据 / 内容身份缺失 / 代际不一致 ⇒ 拒绝进入提升阶段。"""
        missing = [
            arm_id
            for arm_id, payload in self.attempt["arms"].items()
            if not payload.get("evidence") or not payload.get("staged")
        ]
        if missing:
            raise FamilyTxnError(
                f"attempt {self.attempt['attempt_id']!r} 缺臂驱动证据，禁止提升: {missing}"
            )
        target = gen_id(int(self.attempt["target_generation"]))
        drifted = {
            arm_id: (payload.get("staged") or {}).get("gen_id")
            for arm_id, payload in self.attempt["arms"].items()
            if (payload.get("staged") or {}).get("gen_id") != target
        }
        if drifted:
            raise FamilyTxnError(f"全部臂完整代必须一致（期望 {target}），实际不一致: {drifted}")
        self.attempt = {**self.attempt, "status": "DRIVEN"}
        persist_attempt(self.layout, self.attempt)
        return self.attempt

    # ---- 阶段 2：提升（每臂一次目录替换） ----
    def promote(self) -> Dict[str, Any]:
        """逐臂提升暂存根（``PROMOTING`` 先落盘，再动第一个 root）。"""
        self.attempt = {**self.attempt, "status": "PROMOTING"}
        persist_attempt(self.layout, self.attempt)
        self._fire("before_promote", {"attempt": self.attempt["attempt_id"]})
        for arm_id in sorted(self.attempt["arms"]):
            self._promote_arm(arm_id)
        # 全部 roots 已提升、分片未写（状态仍为 PROMOTING）：T3 矩阵的「全部提升但未写分片」
        self._fire("after_promote_all", {"promoted_count": int(self.attempt["promoted_count"])})
        self.attempt = {**self.attempt, "status": "PROMOTED"}
        persist_attempt(self.layout, self.attempt)
        return self.attempt

    def _promote_arm(self, arm_id: str) -> None:
        payload = dict(self.attempt["arms"][arm_id])
        staged_root = self.staged_arm_root(arm_id)
        recorded_sha = str((payload.get("staged") or {})["content_sha"])
        if root_identity(staged_root)["content_sha"] != recorded_sha:
            raise FamilyTxnError(f"暂存根内容身份在提升前被改动: 臂 {arm_id!r}（{staged_root}）")
        arm_root = self.layout.arm_root(arm_id)
        # 比较交换守卫：父代必须仍是本 attempt 登记的那一代（并发写入者/外部推进 ⇒ 拒绝）
        current = optional_identity(arm_root)
        parent_sha = str(payload["parent"]["content_sha"])
        if current is None or current["content_sha"] != parent_sha:
            raise FamilyTxnError(
                f"提升前父代已被推进（拒绝覆盖父代）: 臂 {arm_id!r} "
                f"现场={None if current is None else current['content_sha']} 登记父代={parent_sha}"
            )
        rolled = _unique_path(
            self.layout.staging_attempt_dir(self.virtual_date, str(self.attempt["attempt_id"]))
            / "_rolled"
            / arm_id
        )
        move_root(arm_root, rolled)
        # 单臂提升是两个 rename：父代已移走、新根尚未就位的中断点必须可恢复
        self._fire("mid_promote", {"arm_id": arm_id})
        move_root(staged_root, arm_root)
        if root_identity(arm_root)["content_sha"] != recorded_sha:
            raise FamilyTxnError(f"提升后臂根内容身份不符: 臂 {arm_id!r}（{arm_root}）")
        arms = {key: dict(value) for key, value in self.attempt["arms"].items()}
        arms[arm_id] = {**arms[arm_id], "promote_state": "promoted"}
        self.attempt = {
            **self.attempt,
            "arms": arms,
            "promoted_count": int(self.attempt["promoted_count"]) + 1,
        }
        persist_attempt(self.layout, self.attempt)
        self._fire(
            "after_promote", {"arm_id": arm_id, "promoted_count": self.attempt["promoted_count"]}
        )

    # ---- 阶段 3：分片 + 唯一发布点 ----
    def write_artifacts(self) -> Dict[str, Any]:
        """写全部臂的单日分片（尚未发布；abort 时会移入失败现场）。"""
        shards: Dict[str, Any] = {}
        for arm_id in sorted(self.attempt["arms"]):
            result = write_shard(self.layout, self.campaign_id, self.attempt, arm_id)
            shards[arm_id] = {"state": "written", "sha256": result["sha256"]}
            # 单臂分片已写、journal 尚未登记 ⇒ T3 矩阵的「只写部分臂分片」构造点
            self._fire("after_shard", {"arm_id": arm_id, "shards": sorted(shards)})
        self.attempt = {**self.attempt, "status": "ARTIFACTS", "shards": shards}
        persist_attempt(self.layout, self.attempt)
        self._fire("after_shards", {"shards": sorted(shards)})
        return self.attempt

    def publish(self) -> Dict[str, Any]:
        """唯一发布点：journal 单次原子写 ``COMMITTED``；随后幂等补齐派生投影。"""
        self.attempt = {
            **self.attempt,
            "status": "COMMITTED",
            "published_generation": int(self.attempt["target_generation"]),
            "published_txn_id": str(self.attempt["family_txn_id"]),
            "finished_at": _now(),
        }
        persist_attempt(self.layout, self.attempt)
        self._fire("after_publish", {"family_txn_id": self.attempt["family_txn_id"]})
        top_up = top_up_published(
            self.layout, self.campaign_id, self.attempt, fault_hook=self.fault_hook
        )
        self.attempt = {**self.attempt, "artifacts": {"top_up": top_up}}
        persist_attempt(self.layout, self.attempt)
        return self.attempt

    # ---- 终止 ----
    def abort(self, reason: str) -> Dict[str, Any]:
        """终止未发布 attempt（已发布一律拒绝）。"""
        result = abort_attempt(self.layout, self.attempt, reason=reason, fault_hook=self.fault_hook)
        doc = load_day_journal(self.layout, self.virtual_date)
        reloaded = find_attempt(doc, str(self.attempt["attempt_id"]))
        if reloaded is None:
            raise FamilyTxnError(f"终止后 journal 丢失 attempt: {self.attempt['attempt_id']!r}")
        self.attempt = reloaded
        return result


def open_family_transaction(
    layout: CampaignLayout,
    campaign_id: str,
    arms: Mapping[str, Any],
    virtual_date: str,
    *,
    fault_hook: Optional[FaultHook] = None,
) -> FamilyTransaction:
    """构造并打开一个族级事务（父代核验 + 备份 + 暂存）。"""
    load_campaign(layout.root)
    unknown = sorted(set(arms) - set(list_arms(layout.root)))
    if unknown:
        raise FamilyTxnError(f"存在未注册臂，禁止进入事务（先按 T1 注册）: {unknown}")
    txn = FamilyTransaction(layout, campaign_id, arms, virtual_date, fault_hook=fault_hook)
    txn.open()
    return txn
