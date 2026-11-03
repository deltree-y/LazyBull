# -*- coding: utf-8 -*-
"""P5b 试车场 T2 journal / index / 分片 / manifest 载体（语义与读取口径唯一来源）。

契约来源：设计 v4 §4.3（journal 必记字段、唯一发布点、幂等补齐）与 §7.1（产物根）。

三条硬口径
==========
1. **唯一发布点 = journal attempt 状态转 ``COMMITTED`` 的那一次原子写**，该记录同时写入
   ``published_generation`` 与 ``published_txn_id``。``_index.json``、``manifest.json``、
   ``ledger/<arm>/<date>.json`` 都是**由 journal 与根系身份确定性派生**的投影，可反复重建。
2. **派生投影不含墙钟时间**：分片/index/manifest 只由 journal 证据与根内容身份构成，
   因此「重建」与「首次写入」逐字节一致，哈希比对才有意义（墙钟只落 journal 与报告）。
3. **读侧只见已发布代**：读侧入口 = :func:`read_index` / :func:`published_days`；
   未发布 attempt 的暂存根与分片不得成为正式输入（abort 时移入失败现场）。

**复核边界**：``roots`` 内容身份只对**发布头**（最新已发布代）与现场根比对——历史代的臂根早已
被后续代替换，不可复得。历史日的可信性由"分片内容身份 == 由 journal 确定性重建的分片"保证，
不得读成"历史分片无根复核即不可信"。

封存：全部记录经 ``seal_record`` 落 ``record_sha256``，读取即复算（与 T1 注册台账同口径）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.layout import (
    CampaignLayout,
    FamilyTxnError,
    gen_id,
)
from src.lazybull.v2.hosts.testbed.paths import EVIDENCE_CLASS, SCOPE
from src.lazybull.v2.hosts.testbed.storage import (
    read_json,
    seal_record,
    verify_seal,
    write_json_atomic,
    write_json_exclusive,
)
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "GENESIS_TXN",
    "INDEX_SCHEMA",
    "JOURNAL_SCHEMA",
    "MANIFEST_SCHEMA",
    "SHARD_SCHEMA",
    "active_attempt",
    "add_attempt",
    "assert_json_safe",
    "build_index",
    "build_manifest",
    "build_shard",
    "empty_journal",
    "find_attempt",
    "head_publication",
    "load_all_attempts",
    "load_day_journal",
    "load_manifest",
    "load_shard",
    "published_days",
    "read_index",
    "rebuild_index",
    "rebuild_manifest",
    "save_day_journal",
    "shard_sha256",
    "write_index",
    "write_shard",
]

#: journal 文档 schema
JOURNAL_SCHEMA = "p5b_testbed_family_journal/v1"
#: 读侧已发布代投影 schema
INDEX_SCHEMA = "p5b_testbed_index/v1"
#: 逐日分片 schema
SHARD_SCHEMA = "p5b_testbed_ledger_shard/v1"
#: 臂分片清单 schema
MANIFEST_SCHEMA = "p5b_testbed_ledger_manifest/v1"
#: D0 fresh 起点的事务标识（generation 0 的发布点）
GENESIS_TXN = "genesis"
#: attempt 终态集合
TERMINAL_STATUSES = frozenset({"COMMITTED", "ABORTED"})


def assert_json_safe(payload: Mapping[str, Any], *, what: str) -> None:
    """记录必须可被严格 JSON 序列化（禁止 NaN/Inf 与非原生类型被 ``default=str`` 静默转串）。

    派生投影的可重建性依赖这一约束：一旦某个值被静默字符串化，重建与首写就不再逐字节
    一致，分片/manifest/index 的内容身份比对随即失去意义。
    """
    try:
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FamilyTxnError(
            f"{what} 含非 JSON 安全值（禁止 NaN/Inf 或非原生类型），拒绝落盘: {exc}"
        ) from exc


def _common_fields(campaign_id: str) -> Dict[str, Any]:
    """全部记录共有的范围与证据类别声明（calibration_only 边界）。"""
    return {
        "campaign_id": campaign_id,
        "scope": SCOPE,
        "evidence_class": EVIDENCE_CLASS,
        "virtual_clock": True,
        "not_forward_evidence": True,
        "not_sla_evidence": True,
        "not_real_candidate": True,
    }


# ---------------------------------------------------------------------------
# journal
# ---------------------------------------------------------------------------


def empty_journal(campaign_id: str, virtual_date: str) -> Dict[str, Any]:
    """空 journal 文档（``journal/<date>.json`` 的首个形态）。"""
    return seal_record(
        {
            "schema": JOURNAL_SCHEMA,
            **_common_fields(campaign_id),
            "virtual_date": virtual_date,
            "attempt_seq": 0,
            "attempts": [],
            "active_attempt_id": None,
        }
    )


def load_day_journal(layout: CampaignLayout, virtual_date: str) -> Dict[str, Any]:
    """读取当日 journal；缺失 ⇒ 返回空文档（首次运行形态），不落盘。"""
    path = layout.journal_path(virtual_date)
    if not path.is_file():
        return empty_journal(_campaign_id_of(layout), virtual_date)
    doc = read_json(path, what=f"族级事务 journal（{virtual_date}）")
    verify_seal(doc, what=f"族级事务 journal（{path}）")
    if doc.get("schema") != JOURNAL_SCHEMA:
        raise FamilyTxnError(f"journal schema 非法: {doc.get('schema')!r}（{path}）")
    if doc.get("virtual_date") != virtual_date:
        raise FamilyTxnError(
            f"journal 日期与文件不一致: {doc.get('virtual_date')!r} vs {virtual_date!r}（{path}）"
        )
    return doc


def _campaign_id_of(layout: CampaignLayout) -> str:
    """campaign_id 的唯一来源 = 目录名（与 T1 注册记录同口径）。"""
    return layout.root.name


def save_day_journal(layout: CampaignLayout, doc: Mapping[str, Any]) -> Dict[str, Any]:
    """封存并原子落盘当日 journal（每次状态迁移都整体重写，杜绝半写状态）。"""
    sealed = seal_record({key: value for key, value in dict(doc).items() if key != "record_sha256"})
    assert_json_safe(sealed, what=f"journal（{sealed.get('virtual_date')}）")
    write_json_atomic(layout.journal_path(str(sealed["virtual_date"])), sealed)
    return sealed


def add_attempt(
    layout: CampaignLayout, virtual_date: str, attempt: Mapping[str, Any]
) -> Dict[str, Any]:
    """把新 attempt 追加进当日 journal 并设为 active。"""
    doc = load_day_journal(layout, virtual_date)
    attempts = list(doc.get("attempts") or [])
    attempts.append(dict(attempt))
    doc = {
        **doc,
        "attempt_seq": len(attempts),
        "attempts": attempts,
        "active_attempt_id": attempt["attempt_id"],
    }
    return save_day_journal(layout, doc)


def find_attempt(doc: Mapping[str, Any], attempt_id: str) -> Optional[Dict[str, Any]]:
    """按 id 取 attempt（浅拷贝语义：调用方改动需回写 save_day_journal）。"""
    for attempt in doc.get("attempts") or []:
        if attempt.get("attempt_id") == attempt_id:
            return dict(attempt)
    return None


def active_attempt(doc: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """取当前 active attempt（无则 None）。"""
    active_id = doc.get("active_attempt_id")
    if not active_id:
        return None
    return find_attempt(doc, str(active_id))


def load_all_attempts(layout: CampaignLayout) -> List[Dict[str, Any]]:
    """按虚拟日升序收集全部 attempt（读侧投影重建的唯一原料）。"""
    if not layout.journal_root.is_dir():
        return []
    collected: List[Dict[str, Any]] = []
    for path in sorted(layout.journal_root.glob("*.json")):
        doc = load_day_journal(layout, path.stem)
        collected.extend(dict(attempt) for attempt in doc.get("attempts") or [])
    return collected


def _committed_attempts(layout: CampaignLayout) -> List[Dict[str, Any]]:
    """已发布 attempt（按 target_generation 升序）。"""
    committed = [
        attempt for attempt in load_all_attempts(layout) if attempt.get("status") == "COMMITTED"
    ]
    return sorted(committed, key=lambda attempt: int(attempt["target_generation"]))


# ---------------------------------------------------------------------------
# 分片 / manifest / index（确定性派生）
# ---------------------------------------------------------------------------


def build_shard(campaign_id: str, attempt: Mapping[str, Any], arm_id: str) -> Dict[str, Any]:
    """由 journal attempt 证据确定性构造单臂单日分片（不含墙钟时间）。"""
    arm = (attempt.get("arms") or {}).get(arm_id)
    if not arm or not arm.get("evidence"):
        raise FamilyTxnError(
            f"attempt {attempt.get('attempt_id')!r} 缺臂 {arm_id!r} 的驱动证据，禁止构造分片"
        )
    generation = int(attempt["target_generation"])
    return {
        "schema": SHARD_SCHEMA,
        **_common_fields(campaign_id),
        "arm_id": arm_id,
        "arm_role": arm.get("arm_role"),
        "virtual_date": attempt["virtual_date"],
        "generation": generation,
        "gen_id": gen_id(generation),
        "family_txn_id": attempt["family_txn_id"],
        "parent_generation": int(attempt["parent_generation"]),
        "evidence": dict(arm["evidence"]),
        "root_identity": dict(arm["staged"]),
    }


def shard_sha256(shard: Mapping[str, Any]) -> str:
    """分片内容哈希（对除封存字段外的全字段 canonical 哈希）。"""
    payload = {key: value for key, value in dict(shard).items() if key != "record_sha256"}
    return canonical_json_sha256(payload)


def write_shard(
    layout: CampaignLayout, campaign_id: str, attempt: Mapping[str, Any], arm_id: str
) -> Dict[str, Any]:
    """写单臂单日分片（封存 + 原子替换；重复写同内容为幂等 no-op）。"""
    shard = build_shard(campaign_id, attempt, arm_id)
    sealed = seal_record(shard)
    assert_json_safe(sealed, what=f"分片 {arm_id}/{attempt['virtual_date']}")
    path = layout.shard_path(arm_id, str(attempt["virtual_date"]))
    if path.is_file():
        existing = read_json(path, what=f"分片 {arm_id}/{attempt['virtual_date']}")
        if existing == sealed:
            return {"path": path, "sha256": shard_sha256(shard), "written": False}
        raise FamilyTxnError(
            f"分片已存在且内容不同（禁止覆盖已发布分片）: {path}；"
            f"已登记={existing.get('record_sha256')} 本次={sealed.get('record_sha256')}"
        )
    write_json_atomic(path, sealed)
    return {"path": path, "sha256": shard_sha256(shard), "written": True}


def load_shard(layout: CampaignLayout, arm_id: str, virtual_date: str) -> Optional[Dict[str, Any]]:
    """读取并校验单臂单日分片（缺失 ⇒ None；封存不符 ⇒ 失败）。"""
    path = layout.shard_path(arm_id, virtual_date)
    if not path.is_file():
        return None
    shard = read_json(path, what=f"分片 {arm_id}/{virtual_date}")
    verify_seal(shard, what=f"分片 {path}")
    if shard.get("schema") != SHARD_SCHEMA:
        raise FamilyTxnError(f"分片 schema 非法: {shard.get('schema')!r}（{path}）")
    return shard


def build_manifest(
    campaign_id: str, arm_id: str, attempts: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    """由已发布 attempt 构造单臂分片清单（确定性；无时间戳）。"""
    shards: Dict[str, Any] = {}
    head_generation = 0
    for attempt in sorted(attempts, key=lambda item: int(item["target_generation"])):
        shard = build_shard(campaign_id, attempt, arm_id)
        date = str(attempt["virtual_date"])
        shards[date] = {
            "sha256": shard_sha256(shard),
            "generation": int(attempt["target_generation"]),
            "gen_id": gen_id(int(attempt["target_generation"])),
            "family_txn_id": attempt["family_txn_id"],
            "content_sha": (attempt["arms"][arm_id].get("staged") or {}).get("content_sha"),
        }
        head_generation = int(attempt["target_generation"])
    return seal_record(
        {
            "schema": MANIFEST_SCHEMA,
            **_common_fields(campaign_id),
            "arm_id": arm_id,
            "shards": shards,
            "head_generation": head_generation,
            "head_gen_id": gen_id(head_generation),
            "head_content_sha": (shards[max(shards)]["content_sha"] if shards else None),
        }
    )


def load_manifest(layout: CampaignLayout, arm_id: str) -> Optional[Dict[str, Any]]:
    """读取臂分片清单（缺失 ⇒ None；封存不符 ⇒ 失败）。"""
    path = layout.manifest_path(arm_id)
    if not path.is_file():
        return None
    manifest = read_json(path, what=f"分片清单 {arm_id}")
    verify_seal(manifest, what=f"分片清单 {path}")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise FamilyTxnError(f"分片清单 schema 非法: {manifest.get('schema')!r}（{path}）")
    return manifest


def _write_if_changed(path: Path, payload: Dict[str, Any]) -> bool:
    """内容相同 ⇒ 零写盘；不同 ⇒ 原子替换（幂等补齐的通用原语）。"""
    assert_json_safe(payload, what=f"投影 {path.name}")
    if path.is_file():
        try:
            existing = read_json(path, what=f"投影 {path.name}")
        except TestbedError:
            # 派生投影不可读（截断/封存破坏）⇒ 允许覆盖重建（调用方负责先隔离留证）
            existing = None
        if existing == payload:
            return False
    write_json_atomic(path, payload)
    return True


def rebuild_manifest(layout: CampaignLayout, campaign_id: str, arm_id: str) -> Dict[str, Any]:
    """从 journal 重建单臂分片清单（幂等；返回 ``{"written": bool, "manifest": ...}``）。"""
    committed = [
        attempt for attempt in _committed_attempts(layout) if arm_id in (attempt.get("arms") or {})
    ]
    manifest = build_manifest(campaign_id, arm_id, committed)
    written = _write_if_changed(layout.manifest_path(arm_id), manifest)
    return {"written": written, "manifest": manifest}


def build_index(campaign_id: str, attempts: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """由已发布 attempt 构造读侧 index（确定性；``published_generation`` 唯一权威）。"""
    ordered = sorted(attempts, key=lambda item: int(item["target_generation"]))
    days: List[Dict[str, Any]] = []
    arms: Dict[str, Any] = {}
    for attempt in ordered:
        arm_payload = {
            arm_id: {
                "gen_id": gen_id(int(attempt["target_generation"])),
                "content_sha": (arm.get("staged") or {}).get("content_sha"),
                "shard_sha256": shard_sha256(build_shard(campaign_id, attempt, arm_id)),
            }
            for arm_id, arm in (attempt.get("arms") or {}).items()
        }
        days.append(
            {
                "generation": int(attempt["target_generation"]),
                "virtual_date": attempt["virtual_date"],
                "family_txn_id": attempt["family_txn_id"],
                "arms": arm_payload,
            }
        )
        arms = {
            arm_id: {"gen_id": entry["gen_id"], "content_sha": entry["content_sha"]}
            for arm_id, entry in arm_payload.items()
        }
    head_generation = int(ordered[-1]["target_generation"]) if ordered else 0
    head = ordered[-1] if ordered else None
    return seal_record(
        {
            "schema": INDEX_SCHEMA,
            **_common_fields(campaign_id),
            "published_generation": head_generation,
            "published_txn_id": str(head["family_txn_id"]) if head else GENESIS_TXN,
            "published_virtual_date": str(head["virtual_date"]) if head else None,
            "days": days,
            "arms": arms,
        }
    )


def rebuild_index(layout: CampaignLayout, campaign_id: str) -> Dict[str, Any]:
    """从 journal 重建 index（幂等补齐；返回 ``{"written": bool, "index": ...}``）。

    尚无已发布 attempt（仅 genesis 形态）⇒ 不重建：genesis 记录带 D0 链接字段，重算会丢掉它。
    """
    committed = _committed_attempts(layout)
    generations = [int(attempt["target_generation"]) for attempt in committed]
    if len(set(generations)) != len(generations):
        raise FamilyTxnError(f"同一代存在多个已发布 attempt（禁止拼接已发布代）: {generations}")
    if not committed:
        existing = read_index(layout)
        return {
            "written": False,
            "index": existing if existing is not None else build_index(campaign_id, []),
        }
    index = build_index(campaign_id, committed)
    written = _write_if_changed(layout.index_path, index)
    return {"written": written, "index": index}


def write_index(layout: CampaignLayout, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """直接落盘 index（仅 genesis 建立与重建路径使用；封存后原子替换）。"""
    sealed = seal_record(
        {key: value for key, value in dict(payload).items() if key != "record_sha256"}
    )
    assert_json_safe(sealed, what="index")
    write_json_atomic(layout.index_path, sealed)
    return sealed


def read_index(layout: CampaignLayout) -> Optional[Dict[str, Any]]:
    """读取读侧 index（缺失 ⇒ None；封存/schema 不符 ⇒ 失败）。"""
    path = layout.index_path
    if not path.is_file():
        return None
    index = read_json(path, what="试车场 index")
    verify_seal(index, what=f"试车场 index（{path}）")
    if index.get("schema") != INDEX_SCHEMA:
        raise FamilyTxnError(f"index schema 非法: {index.get('schema')!r}（{path}）")
    if index.get("campaign_id") != layout.root.name:
        raise FamilyTxnError(
            f"index campaign 与目录不一致: {index.get('campaign_id')!r} vs {layout.root.name!r}"
        )
    return index


def published_days(index: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """已发布虚拟日 → 发布记录（``{}`` = 尚无已发布日）。"""
    if not index:
        return {}
    return {str(entry["virtual_date"]): dict(entry) for entry in index.get("days") or []}


def head_publication(index: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """当前发布头（``published_generation`` 对应的日记录；无已发布日 ⇒ None）。"""
    days = published_days(index)
    if not days:
        return None
    latest = max(days.values(), key=lambda entry: int(entry["generation"]))
    return dict(latest)


def genesis_index(
    campaign_id: str, arms_identity: Mapping[str, Any], d0_record_sha: str
) -> Dict[str, Any]:
    """D0 fresh 起点的 index（generation 0；内容身份 = 各臂 fresh 根观测）。"""
    return seal_record(
        {
            "schema": INDEX_SCHEMA,
            **_common_fields(campaign_id),
            "published_generation": 0,
            "published_txn_id": GENESIS_TXN,
            "published_virtual_date": None,
            "days": [],
            "arms": {
                arm_id: {
                    "gen_id": gen_id(0),
                    "content_sha": str(payload["content_sha"]),
                }
                for arm_id, payload in arms_identity.items()
            },
            "genesis": {"d0_fresh_record_sha256": str(d0_record_sha), "trained_before_d0": True},
        }
    )


def create_genesis_index(
    layout: CampaignLayout, campaign_id: str, arms_identity: Mapping[str, Any], d0_record_sha: str
) -> Dict[str, Any]:
    """建立 D0 fresh 起点 index（独占创建：并发建立由内核裁决）。"""
    payload = genesis_index(campaign_id, arms_identity, d0_record_sha)
    assert_json_safe(payload, what="genesis index")
    try:
        write_json_exclusive(layout.index_path, payload)
    except FileExistsError as exc:
        raise FamilyTxnError(
            f"index 已存在（禁止覆盖 genesis 或已发布代）: {layout.index_path}（{exc}）"
        ) from exc
    return payload
