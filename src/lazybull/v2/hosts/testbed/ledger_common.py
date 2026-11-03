# -*- coding: utf-8 -*-
"""P5b 试车场 T3 账本旁路公共定义：产物路径、缺口分类、INVALIDATED 终态登记。

契约来源：设计 v4 §4.3（族级事务 / 部分 roots 提升 / 分片·manifest·index 状态表）、
§7.1（产物根）、§8-H2/H3。

三条硬口径（全 T3 账本模块共用）
================================
1. **不改已发布分片字节**：``ledger/<arm>/<date>.json`` 是已发布代的正式证据；只有「该日分片
   **缺失**且 journal 能唯一重建」才补写（载荷不含墙钟，与首写逐字节一致）；**存在但内容身份
   不符**时一律不覆盖，只登记 INVALIDATED。
2. **旁路只读**：链摘要、审计结果、INVALIDATED 登记全部落在 ``audit/`` 新增平行文件
   （``shard_chain.json`` / ``ledger_audit.json`` / ``invalidated.json`` / ``quarantine/``），
   不写 ``arms/``、不写持仓、不改 journal、不改已发布分片。
3. **读侧只见已发布代**：一切查询以 ``_index.json`` 的 ``published_generation`` 为准。

缺口分类（``GAP_CLASSES``）
==========================
| 类 | 含义 | 处置 |
|---|---|---|
| ``missing_day`` | 已发布日缺分片（缺日） | 可修复：按 journal 重建（逐字节一致） |
| ``bad_shard`` | 分片内容身份不符 / 不可读（坏片） | **INVALIDATED**（禁止覆盖已发布字节） |
| ``out_of_order`` | 代序或日期序错乱 | journal 级错乱 ⇒ INVALIDATED；投影级 ⇒ 可重建 |
| ``mixed_generation`` | 同一日的代/事务在 journal 与 index 间矛盾 | **INVALIDATED** |
| ``unpublished_attempt_shard`` | 未发布 attempt 的分片混入 ``ledger/`` | 可修复：移入 ``audit/quarantine/`` |
| ``manifest_missing`` / ``manifest_mismatch`` | 分片清单缺失或与 journal 不符 | 可修复：从 journal 重建 |
| ``index_absent`` / ``index_missing_day`` | 读侧 index 缺失或落后 | 可修复：从 journal 重建 |
| ``index_mismatch`` | index 记录的已发布臂内容与 journal 不符 | **INVALIDATED**（读侧代不可回退） |
| ``chain_stale`` | 链文件缺失或与重算不一致 | 可修复：重写链文件 |
| ``journal_corrupt`` | journal 不可读（证据源损坏） | **INVALIDATED**（无证据可重建） |

INVALIDATED 与 :class:`layout.CampaignInvalidError` 的关系
=========================================================
- ``CampaignInvalidError``（T2）= **父代备份不可验证 / 现场根与已发布代不一致** ⇒ 整个
  campaign 停止推进，禁止拼接或猜测；
- :class:`LedgerInvalidatedError`（T3）= **某个已发布日的内容无法由 journal 唯一重建**（或已
  被登记为 INVALIDATED）⇒ 该「臂 + 日」进入终态 ``INVALIDATED``，虚拟时钟同样拒绝推进，
  需人工介入；campaign 其余日仍可只读审计。两者都是 fail-closed，粒度为「campaign」 vs
  「臂 + 日」，且 INVALIDATED **不自动清除**（append-only 登记）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.lazybull.v2.hosts.testbed.journal import assert_json_safe
from src.lazybull.v2.hosts.testbed.layout import CampaignLayout, FamilyTxnError
from src.lazybull.v2.hosts.testbed.paths import (
    EVIDENCE_CLASS,
    SCOPE,
    TestbedError,
    assert_write_scope,
)
from src.lazybull.v2.hosts.testbed.storage import (
    read_json,
    seal_record,
    verify_seal,
    write_json_atomic,
)

__all__ = [
    "AUDIT_DIR",
    "AUDIT_FILENAME",
    "AUDIT_SCHEMA",
    "CHAIN_FILENAME",
    "GAP_CLASSES",
    "INVALIDATED_FILENAME",
    "INVALIDATED_SCHEMA",
    "LedgerGap",
    "LedgerInvalidatedError",
    "LedgerPaths",
    "MATRIX_FILENAME",
    "QUARANTINE_DIR",
    "empty_invalidations",
    "invalidated_days",
    "ledger_paths",
    "load_invalidations",
    "record_invalidations",
    "write_if_changed",
]

#: 旁路审计目录（§7.1 的 T3 新增子树，campaign 根内）
AUDIT_DIR = "audit"
#: 分片链产物（逐臂链摘要；**新增平行文件**，不改写 ``ledger/<arm>/<date>.json``）
CHAIN_FILENAME = "shard_chain.json"
#: 三方一致性审计产物
AUDIT_FILENAME = "ledger_audit.json"
#: INVALIDATED 终态登记（append-only，不自动清除）
INVALIDATED_FILENAME = "invalidated.json"
#: 隔离区（坏片 / 未发布 attempt 分片移出后的证据落点）
QUARANTINE_DIR = "quarantine"
#: 故障矩阵判定表产物
MATRIX_FILENAME = "fault_matrix.json"
#: 审计文档 schema
AUDIT_SCHEMA = "p5b_testbed_ledger_audit/v1"
#: INVALIDATED 登记 schema
INVALIDATED_SCHEMA = "p5b_testbed_ledger_invalidated/v1"
#: 缺口分类全集（报告与判定表按此排序）
GAP_CLASSES: Tuple[str, ...] = (
    "missing_day",
    "bad_shard",
    "out_of_order",
    "mixed_generation",
    "unpublished_attempt_shard",
    "manifest_missing",
    "manifest_mismatch",
    "index_absent",
    "index_missing_day",
    "index_mismatch",
    "chain_stale",
    "journal_corrupt",
)


class LedgerInvalidatedError(FamilyTxnError):
    """已发布分片无法由 journal 唯一重建（或已登记 INVALIDATED）⇒ 虚拟时钟拒绝推进。"""

    __test__ = False


@dataclass(frozen=True)
class LedgerPaths:
    """T3 旁路产物路径（``audit/`` 子树的唯一定义点）。"""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def audit_root(self) -> Path:
        return self.root / AUDIT_DIR

    @property
    def chain_path(self) -> Path:
        return self.audit_root / CHAIN_FILENAME

    @property
    def audit_path(self) -> Path:
        return self.audit_root / AUDIT_FILENAME

    @property
    def invalidated_path(self) -> Path:
        return self.audit_root / INVALIDATED_FILENAME

    @property
    def matrix_path(self) -> Path:
        return self.audit_root / MATRIX_FILENAME

    @property
    def quarantine_root(self) -> Path:
        return self.audit_root / QUARANTINE_DIR

    def quarantine_shard_path(self, arm_id: str, virtual_date: str) -> Path:
        return self.quarantine_root / "shards" / arm_id / f"{virtual_date}.json"


def ledger_paths(layout: CampaignLayout) -> LedgerPaths:
    """构造 T3 旁路路径（不建目录）。"""
    return LedgerPaths(layout.root)


def guard_ledger_write(layout: CampaignLayout) -> None:
    """库层写入口的生产根硬禁（**永远**生效；写域白名单由工具/运行器入口负责）。

    写域白名单只在工具/运行器入口校验（那里才知道是否测试/临时根）；库层至少保证
    ``data/paper`` / ``data/runs`` / ``data/models`` 永不被写入——任何新调用方（含后续阶段
    直接 import）都无法绕过。
    """
    assert_write_scope(layout.root, allow_external_root=True)


@dataclass(frozen=True)
class LedgerGap:
    """单个缺口（臂 + 日 + 分类 + 可修复性）。"""

    arm_id: str
    virtual_date: Optional[str]
    kind: str
    detail: str
    repairable: bool = True
    generation: Optional[int] = None
    expected_sha256: Optional[str] = None
    actual_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in GAP_CLASSES:
            raise FamilyTxnError(f"未知缺口分类: {self.kind!r}（须在 GAP_CLASSES 内）")

    @property
    def invalidated(self) -> bool:
        """不可重建 ⇒ 该「臂 + 日」进入 INVALIDATED 终态。"""
        return not self.repairable

    def key(self) -> str:
        return f"{self.arm_id}|{self.virtual_date}|{self.kind}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "virtual_date": self.virtual_date,
            "kind": self.kind,
            "detail": self.detail,
            "repairable": bool(self.repairable),
            "generation": self.generation,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
        }


def _gap(
    arm_id: str,
    virtual_date: Optional[str],
    kind: str,
    detail: str,
    *,
    repairable: bool,
    generation: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    actual_sha256: Optional[str] = None,
) -> LedgerGap:
    return LedgerGap(
        arm_id=arm_id,
        virtual_date=virtual_date,
        kind=kind,
        detail=detail,
        repairable=repairable,
        generation=generation,
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
    )


def _scope_fields(campaign_id: str) -> Dict[str, Any]:
    """旁路产物共有的范围声明（``calibration_only`` 边界）。"""
    return {
        "campaign_id": str(campaign_id),
        "scope": SCOPE,
        "evidence_class": EVIDENCE_CLASS,
        "virtual_clock": True,
        "not_forward_evidence": True,
        "not_sla_evidence": True,
        "not_real_candidate": True,
    }


def write_if_changed(path: Path, payload: Mapping[str, Any], *, what: str) -> bool:
    """内容相同 ⇒ 零写盘；不同 ⇒ 原子替换（旁路产物的幂等落盘原语）。"""
    assert_json_safe(payload, what=what)
    if path.is_file():
        try:
            existing = read_json(path, what=what)
        except TestbedError:
            existing = None
        if existing == dict(payload):
            return False
    write_json_atomic(path, dict(payload))
    return True


# ---------------------------------------------------------------------------
# INVALIDATED 终态登记（append-only，不自动清除）
# ---------------------------------------------------------------------------


def empty_invalidations(campaign_id: str) -> Dict[str, Any]:
    """空登记文档（无 INVALIDATED 条目时的形态）。"""
    return seal_record(
        {
            "schema": INVALIDATED_SCHEMA,
            **_scope_fields(campaign_id),
            "entries": [],
        }
    )


def load_invalidations(layout: CampaignLayout) -> Dict[str, Any]:
    """读取 INVALIDATED 登记；缺失 ⇒ 空文档（不落盘）。

    Raises:
        TestbedError: 登记损坏或封存不符（fail-closed，禁止当成「无失效」）。
    """
    path = ledger_paths(layout).invalidated_path
    if not path.is_file():
        return empty_invalidations(layout.root.name)
    doc = read_json(path, what="INVALIDATED 登记")
    verify_seal(doc, what=f"INVALIDATED 登记（{path}）")
    if doc.get("schema") != INVALIDATED_SCHEMA:
        raise FamilyTxnError(f"INVALIDATED 登记 schema 非法: {doc.get('schema')!r}（{path}）")
    return doc


def invalidated_days(layout: CampaignLayout) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """INVALIDATED 索引：``{arm_id: {virtual_date: entry}}``。

    约定：``arm_id == "*"`` 表示 **campaign 级**登记；``virtual_date`` 为空串（原值 None）
    表示**波及全部虚拟日**（如 journal 损坏 / index 超前）。日期键一律为 ``YYYYMMDD`` 字符串
    或空串，便于脚本与查询按字典序比较。
    """
    doc = load_invalidations(layout)
    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for entry in doc.get("entries") or []:
        arm_id = str(entry.get("arm_id"))
        raw_date = entry.get("virtual_date")
        date = "" if raw_date is None else str(raw_date)
        index.setdefault(arm_id, {})[date] = dict(entry)
    return index


def record_invalidations(
    layout: CampaignLayout, campaign_id: str, gaps: Iterable[LedgerGap]
) -> Dict[str, Any]:
    """登记 INVALIDATED 终态（幂等 append-only；已存在同 key 条目不改写）。

    Returns:
        ``{"written": bool, "added": [...], "total": int}``。
    """
    guard_ledger_write(layout)
    current = load_invalidations(layout)
    entries = [dict(item) for item in current.get("entries") or []]
    seen = {(item["arm_id"], item["virtual_date"], item.get("kind")) for item in entries}
    added: List[Dict[str, Any]] = []
    for gap in gaps:
        key = (gap.arm_id, gap.virtual_date, gap.kind)
        if key in seen:
            continue
        entries.append(
            {
                "arm_id": gap.arm_id,
                "virtual_date": gap.virtual_date,
                "kind": gap.kind,
                "detail": gap.detail,
                "generation": gap.generation,
                "expected_sha256": gap.expected_sha256,
                "actual_sha256": gap.actual_sha256,
                "terminal_state": "INVALIDATED",
            }
        )
        seen.add(key)
        added.append(dict(entries[-1]))
    entries.sort(key=lambda item: (item["arm_id"], str(item["virtual_date"]), item["kind"]))
    doc = seal_record(
        {
            "schema": INVALIDATED_SCHEMA,
            **_scope_fields(campaign_id),
            "entries": entries,
        }
    )
    written = write_if_changed(ledger_paths(layout).invalidated_path, doc, what="INVALIDATED 登记")
    return {"written": bool(written and added), "added": added, "total": len(entries)}
