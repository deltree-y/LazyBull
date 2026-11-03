# -*- coding: utf-8 -*-
"""P5b 试车场 T3 长期分片账本审计：链式审计、三方一致性、缺口分类与旁路产物。

契约来源：设计 v4 §4.3（族级事务 / 分片·manifest·index 状态表）、§7.1（产物根）、
§7.2（报告字段「分片链 digest / 三方一致性 / 缺口清单」）、§8-H3（账本一致）。

- 审计**只读**：读错误一律归入缺口分类（``bad_shard`` / ``journal_corrupt`` 等），不抛裸异常；
- 链摘要与审计结果落 ``audit/shard_chain.json`` / ``audit/ledger_audit.json``（新增平行文件）；
- 缺口分类、路径与 INVALIDATED 语义见 :mod:`ledger_common`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.lazybull.v2.hosts.testbed.journal import (
    SHARD_SCHEMA,
    build_shard,
    load_all_attempts,
    load_manifest,
    read_index,
    shard_sha256,
)
from src.lazybull.v2.hosts.testbed.layout import CampaignLayout, FamilyTxnError, gen_id
from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.shard_chain import (
    CHAIN_SCHEMA,
    build_chain,
    campaign_chain_digest,
)
from src.lazybull.v2.hosts.testbed.storage import read_json, seal_record, verify_seal

from src.lazybull.v2.hosts.testbed.ledger_common import (
    AUDIT_SCHEMA,
    GAP_CLASSES,
    LedgerGap,
    guard_ledger_write,
    ledger_paths,
    write_if_changed,
    _gap,
    _scope_fields,
)

__all__ = [
    "audit_ledger",
    "build_chain_document",
    "shard_sha_map",
    "write_ledger_artifacts",
]

#: 分片文件名形态（``YYYYMMDD.json``；排除 manifest/链等平行文件）
_SHARD_NAME = re.compile(r"^(\d{8})\.json$")
#: 设计口径（journal / 分片 / manifest / index）相关缺口；链产物新鲜度不计入
_DESIGN_GAP_KINDS = frozenset(kind for kind in GAP_CLASSES if kind != "chain_stale")
#: 内容类缺口（命中即该臂「三方一致」为假：计数相等也可能是内容损坏）
_CONTENT_GAP_KINDS = frozenset(
    {
        "missing_day",
        "bad_shard",
        "manifest_mismatch",
        "index_mismatch",
        "mixed_generation",
    }
)

# ---------------------------------------------------------------------------
# 读盘（审计必须分类而不是崩掉：读错误进入缺口，不抛裸异常）
# ---------------------------------------------------------------------------


@dataclass
class _JournalState:
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    committed_by_day: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None


def _journal_state(layout: CampaignLayout) -> _JournalState:
    try:
        attempts = load_all_attempts(layout)
    except (TestbedError, FamilyTxnError, OSError) as exc:
        return _JournalState(error=f"{type(exc).__name__}: {exc}")
    committed = {
        str(attempt["virtual_date"]): dict(attempt)
        for attempt in attempts
        if attempt.get("status") == "COMMITTED"
    }
    return _JournalState(attempts=attempts, committed_by_day=committed)


def _expected_shards(
    campaign_id: str, committed: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """journal 唯一决定的「每臂每日应有什么分片」（内容身份由 journal 重建）。"""
    expected: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for date, attempt in sorted(committed.items(), key=lambda kv: int(kv[1]["target_generation"])):
        for arm_id in sorted(attempt.get("arms") or {}):
            shard = build_shard(campaign_id, attempt, arm_id)
            expected.setdefault(arm_id, {})[date] = {
                "generation": int(attempt["target_generation"]),
                "gen_id": shard["gen_id"],
                "family_txn_id": str(attempt["family_txn_id"]),
                "attempt_id": str(attempt["attempt_id"]),
                "sha256": shard_sha256(shard),
            }
    return expected


def _index_state(layout: CampaignLayout) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return read_index(layout), None
    except (TestbedError, FamilyTxnError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _manifest_state(
    layout: CampaignLayout, arms: Sequence[str]
) -> Tuple[Dict[str, Optional[Dict[str, Any]]], Dict[str, str]]:
    manifests: Dict[str, Optional[Dict[str, Any]]] = {}
    errors: Dict[str, str] = {}
    for arm_id in arms:
        try:
            manifests[arm_id] = load_manifest(layout, arm_id)
        except (TestbedError, FamilyTxnError, OSError) as exc:
            manifests[arm_id] = None
            errors[arm_id] = f"{type(exc).__name__}: {exc}"
    return manifests, errors


def _scan_shards(layout: CampaignLayout) -> Dict[str, Dict[str, Path]]:
    """扫描 ``ledger/<arm>/*.json``（只认 ``YYYYMMDD.json``；平行文件不计）。"""
    scanned: Dict[str, Dict[str, Path]] = {}
    if not layout.ledger_root.is_dir():
        return scanned
    for arm_dir in sorted(layout.ledger_root.iterdir()):
        if not arm_dir.is_dir():
            continue
        for path in sorted(arm_dir.glob("*.json")):
            matched = _SHARD_NAME.match(path.name)
            if not matched:
                continue
            scanned.setdefault(arm_dir.name, {})[matched.group(1)] = path
    return scanned


def _read_shard(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """读分片内容（缺失/损坏/封存不符 ⇒ ``(None, None, error)``，不抛异常）。"""
    try:
        shard = read_json(path, what=f"分片 {path.name}")
        verify_seal(shard, what=f"分片 {path}")
        if shard.get("schema") != SHARD_SCHEMA:
            raise FamilyTxnError(f"分片 schema 非法: {shard.get('schema')!r}")
        return shard, shard_sha256(shard), None
    except (TestbedError, FamilyTxnError, OSError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def shard_sha_map(layout: CampaignLayout) -> Dict[str, Dict[str, Optional[str]]]:
    """已落盘分片的内容哈希（只读；坏片记 None，供旁路取证与不变量比对）。"""
    result: Dict[str, Dict[str, Optional[str]]] = {}
    for arm_id, dates in _scan_shards(layout).items():
        result[arm_id] = {}
        for date, path in dates.items():
            _shard, sha, _error = _read_shard(path)
            result[arm_id][date] = sha
    return result


# ---------------------------------------------------------------------------
# 缺口分类（逐臂 / 逐日）
# ---------------------------------------------------------------------------


def _shard_gaps(
    arm_id: str,
    date: str,
    expected: Mapping[str, Any],
    path: Optional[Path],
) -> Tuple[List[LedgerGap], Optional[str]]:
    """已发布日的分片存在性 / 内容身份（返回缺口与实际 sha）。"""
    generation = int(expected["generation"])
    if path is None:
        return (
            [
                _gap(
                    arm_id,
                    date,
                    "missing_day",
                    f"已发布日缺分片（journal 可重建）: {arm_id}/{date}",
                    repairable=True,
                    generation=generation,
                    expected_sha256=expected["sha256"],
                )
            ],
            None,
        )
    shard, actual_sha, error = _read_shard(path)
    if error is not None:
        return (
            [
                _gap(
                    arm_id,
                    date,
                    "bad_shard",
                    f"分片不可读或封存不符（禁止覆盖已发布字节）: {arm_id}/{date}（{error}）",
                    repairable=False,
                    generation=generation,
                    expected_sha256=expected["sha256"],
                )
            ],
            None,
        )
    if actual_sha != expected["sha256"]:
        return (
            [
                _gap(
                    arm_id,
                    date,
                    "bad_shard",
                    f"分片内容身份与 journal 重建不一致: {arm_id}/{date}",
                    repairable=False,
                    generation=generation,
                    expected_sha256=expected["sha256"],
                    actual_sha256=actual_sha,
                )
            ],
            actual_sha,
        )
    return [], actual_sha


def _manifest_gaps(
    arm_id: str,
    expected: Mapping[str, Mapping[str, Any]],
    manifest: Optional[Dict[str, Any]],
    manifest_error: Optional[str],
) -> List[LedgerGap]:
    """分片清单：缺失 / 与 journal 不符（清单是派生投影，可整份重建）。"""
    if manifest_error is not None:
        return [
            _gap(
                arm_id,
                None,
                "manifest_mismatch",
                f"分片清单不可读（可从 journal 重建）: {arm_id}（{manifest_error}）",
                repairable=True,
            )
        ]
    if manifest is None:
        if not expected:
            return []
        return [
            _gap(
                arm_id,
                None,
                "manifest_missing",
                f"分片清单缺失（可从 journal 重建）: {arm_id}",
                repairable=True,
            )
        ]
    shards = dict(manifest.get("shards") or {})
    gaps: List[LedgerGap] = []
    for date, item in sorted(expected.items()):
        entry = shards.get(date)
        if entry is None:
            gaps.append(
                _gap(
                    arm_id,
                    date,
                    "manifest_mismatch",
                    f"清单缺该日条目: {arm_id}/{date}",
                    repairable=True,
                    generation=int(item["generation"]),
                    expected_sha256=item["sha256"],
                )
            )
            continue
        if str(entry.get("sha256")) != str(item["sha256"]):
            gaps.append(
                _gap(
                    arm_id,
                    date,
                    "manifest_mismatch",
                    f"清单分片 sha 与 journal 不符: {arm_id}/{date}",
                    repairable=True,
                    generation=int(item["generation"]),
                    expected_sha256=item["sha256"],
                    actual_sha256=entry.get("sha256"),
                )
            )
        elif int(entry.get("generation", -1)) != int(item["generation"]):
            gaps.append(
                _gap(
                    arm_id,
                    date,
                    "manifest_mismatch",
                    f"清单代际与 journal 不符: {arm_id}/{date}",
                    repairable=True,
                    generation=int(item["generation"]),
                )
            )
    extras = sorted(set(shards) - set(expected))
    for date in extras:
        gaps.append(
            _gap(
                arm_id,
                date,
                "manifest_mismatch",
                f"清单含未发布日条目（重建后消失）: {arm_id}/{date}",
                repairable=True,
            )
        )
    head = int(manifest.get("head_generation", 0))
    expected_head = max((int(item["generation"]) for item in expected.values()), default=0)
    if head != expected_head:
        gaps.append(
            _gap(
                arm_id,
                None,
                "manifest_mismatch",
                f"清单 head_generation 与 journal 不符: {arm_id} {head} vs {expected_head}",
                repairable=True,
            )
        )
    return gaps


def _index_arm_gaps(
    arm_id: str,
    expected: Mapping[str, Mapping[str, Any]],
    index_days: Mapping[str, Mapping[str, Any]],
) -> List[LedgerGap]:
    """index 中该臂条目的内容身份（index 落后可重建；代/事务矛盾即失效）。"""
    gaps: List[LedgerGap] = []
    for date, item in sorted(expected.items()):
        day = index_days.get(date)
        if day is None:
            gaps.append(
                _gap(
                    arm_id,
                    date,
                    "index_missing_day",
                    f"index 缺已发布日（可从 journal 重建）: {date}",
                    repairable=True,
                    generation=int(item["generation"]),
                    expected_sha256=item["sha256"],
                )
            )
            continue
        entry = (day.get("arms") or {}).get(arm_id)
        if entry is None:
            # 缺臂条目与「整天缺失」同属派生投影空洞：index 由 journal 完全可重建 ⇒ 可修复。
            # 只有「已发布代不可回退」类（published_generation 超前 / 事务 ID 混代）才失效。
            gaps.append(
                _gap(
                    arm_id,
                    date,
                    "index_missing_day",
                    f"index 该日缺此臂条目（可从 journal 重建）: {arm_id}/{date}",
                    repairable=True,
                    generation=int(item["generation"]),
                    expected_sha256=item["sha256"],
                )
            )
            continue
        if str(entry.get("shard_sha256")) != str(item["sha256"]):
            gaps.append(
                _gap(
                    arm_id,
                    date,
                    "index_mismatch",
                    f"index 记录的分片身份与 journal 不符: {arm_id}/{date}",
                    repairable=False,
                    generation=int(item["generation"]),
                    expected_sha256=item["sha256"],
                    actual_sha256=entry.get("shard_sha256"),
                )
            )
        elif str(entry.get("gen_id")) != str(item["gen_id"]):
            gaps.append(
                _gap(
                    arm_id,
                    date,
                    "mixed_generation",
                    f"index 代际标签与 journal 不符: {arm_id}/{date}",
                    repairable=False,
                    generation=int(item["generation"]),
                )
            )
        elif str(day.get("family_txn_id")) != str(item["family_txn_id"]):
            gaps.append(
                _gap(
                    arm_id,
                    date,
                    "mixed_generation",
                    f"index 事务 ID 与 journal 不符（混代）: {arm_id}/{date}",
                    repairable=False,
                    generation=int(item["generation"]),
                )
            )
    return gaps


def _stray_gaps(
    arm_id: str,
    expected: Mapping[str, Mapping[str, Any]],
    on_disk: Mapping[str, Path],
) -> List[LedgerGap]:
    """未发布 attempt 的分片混入（journal 无该日 COMMITTED，却在 ledger 里）。"""
    gaps: List[LedgerGap] = []
    for date, path in sorted(on_disk.items()):
        if date in expected:
            continue
        gaps.append(
            _gap(
                arm_id,
                date,
                "unpublished_attempt_shard",
                f"未发布 attempt 的分片混入账本（移入隔离区）: {path}",
                repairable=True,
            )
        )
    return gaps


def _ordering_gaps(
    arm_id: str,
    expected: Mapping[str, Mapping[str, Any]],
    manifest: Optional[Dict[str, Any]],
) -> List[LedgerGap]:
    """清单内的代序/日期序（投影级错序可整份重建）。"""
    if manifest is None:
        return []
    shards = dict(manifest.get("shards") or {})
    pairs = [
        (str(date), int(entry.get("generation", -1)))
        for date, entry in shards.items()
        if date in expected
    ]
    ordered = sorted(pairs, key=lambda item: item[1])
    if [date for date, _ in ordered] != sorted(date for date, _ in pairs):
        return [
            _gap(
                arm_id,
                None,
                "out_of_order",
                f"清单内代序与日期序不一致（重建后归一）: {arm_id}",
                repairable=True,
            )
        ]
    return []


def _journal_order_gaps(committed: Mapping[str, Mapping[str, Any]]) -> List[LedgerGap]:
    """journal 级代序/日期序（journal 是唯一证据源，错乱即失效）。"""
    if not committed:
        return []
    ordered = sorted(
        ((date, int(attempt["target_generation"])) for date, attempt in committed.items()),
        key=lambda item: item[1],
    )
    gaps: List[LedgerGap] = []
    generations = [generation for _, generation in ordered]
    expected = list(range(1, len(generations) + 1))
    if generations != expected:
        gaps.append(
            _gap(
                "*",
                None,
                "out_of_order",
                f"journal 已发布代序不连续（期望 1..{len(expected)}，实际 {generations}）",
                repairable=False,
            )
        )
    if [date for date, _ in ordered] != sorted(date for date, _ in ordered):
        gaps.append(
            _gap(
                "*",
                None,
                "out_of_order",
                f"journal 代序与虚拟日期序不一致: {[date for date, _ in ordered]}",
                repairable=False,
            )
        )
    return gaps


def _index_campaign_gaps(
    index: Optional[Dict[str, Any]],
    index_error: Optional[str],
    committed: Mapping[str, Mapping[str, Any]],
) -> List[LedgerGap]:
    """campaign 级 index 一致性：缺失/落后可重建，超前或混代即失效。"""
    if index is None:
        if not committed and index_error is None:
            return []  # 尚无已发布代（genesis 由运行器独占建立）
        if not committed:
            return [
                _gap(
                    "*",
                    None,
                    "index_absent",
                    f"index 损坏且 journal 无已发布 attempt（无法重建）: {index_error}",
                    repairable=False,
                )
            ]
        return [
            _gap(
                "*",
                None,
                "index_absent",
                f"index 缺失或损坏（可从 journal 重建）: {index_error or '缺失'}",
                repairable=True,
            )
        ]
    days = list(index.get("days") or [])
    generations = [int(entry.get("generation", -1)) for entry in days]
    if generations != sorted(generations) or len(set(generations)) != len(generations):
        return [
            _gap(
                "*",
                None,
                "out_of_order",
                f"index 日序非严格升序或代重复: {generations}",
                repairable=True,
            )
        ]
    expected_head = max(
        (int(attempt["target_generation"]) for attempt in committed.values()), default=0
    )
    published = int(index.get("published_generation", 0))
    if published != expected_head:
        ahead = published > expected_head
        return [
            _gap(
                "*",
                None,
                "index_mismatch" if ahead else "index_missing_day",
                f"index published_generation 与 journal 不符（{'超前=混代风险' if ahead else '落后可重建'}）: "
                f"index={published} journal={expected_head}",
                repairable=not ahead,
            )
        ]
    index_days = {str(entry.get("virtual_date")): entry for entry in days}
    return _index_day_gaps(index_days, committed)


def _index_day_gaps(
    index_days: Mapping[str, Mapping[str, Any]],
    committed: Mapping[str, Mapping[str, Any]],
) -> List[LedgerGap]:
    """index 日 → journal attempt 的事务/代际一致性（混代即失效）。"""
    gaps: List[LedgerGap] = []
    for date, entry in sorted(index_days.items()):
        attempt = committed.get(date)
        if attempt is None:
            gaps.append(
                _gap(
                    "*",
                    date,
                    "mixed_generation",
                    f"index 声明已发布但 journal 无 COMMITTED attempt（证据缺失）: {date}",
                    repairable=False,
                )
            )
            continue
        if str(entry.get("family_txn_id")) != str(attempt["family_txn_id"]):
            gaps.append(
                _gap(
                    "*",
                    date,
                    "mixed_generation",
                    f"index 事务 ID 与 journal 不符: {date}",
                    repairable=False,
                    generation=int(attempt["target_generation"]),
                )
            )
    return gaps


def _chain_state(
    layout: CampaignLayout,
    campaign_id: str,
    links_by_arm: Mapping[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, str], List[LedgerGap]]:
    """重算分片链并与落盘链文件比对（链是纯投影，可重写）。

    链节错序（同日重代 / 代序×日期序交叉）不得让审计崩掉：收敛为 campaign 级
    ``out_of_order``（不可重建）并让该臂链缺省，交由 INVALIDATED 路径处置。
    """
    chains, gaps = _build_chains(campaign_id, links_by_arm)
    stored, stored_error = _stored_chain(layout)
    gaps.extend(_stored_chain_gaps(stored, stored_error, chains))
    return chains, {arm: chain["chain_digest"] for arm, chain in chains.items()}, gaps


def _build_chains(
    campaign_id: str, links_by_arm: Mapping[str, List[Dict[str, Any]]]
) -> Tuple[Dict[str, Any], List[LedgerGap]]:
    """逐臂建链；链节错序收敛为不可重建缺口（不让审计崩掉）。"""
    chains: Dict[str, Any] = {}
    gaps: List[LedgerGap] = []
    for arm_id, links in sorted(links_by_arm.items()):
        try:
            chains[arm_id] = build_chain(campaign_id, arm_id, links)
        except FamilyTxnError as exc:
            gaps.append(
                _gap(
                    arm_id,
                    None,
                    "out_of_order",
                    f"分片链节无法建链（代序/日期序错乱，证据不可重建）: {arm_id}（{exc}）",
                    repairable=False,
                )
            )
    return chains, gaps


def _stored_chain(layout: CampaignLayout) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """读取落盘链产物（缺失/损坏均不抛：链是纯投影，可重写）。"""
    path = ledger_paths(layout).chain_path
    if not path.is_file():
        return None, None
    try:
        stored = read_json(path, what="分片链")
        verify_seal(stored, what=f"分片链（{path}）")
    except (TestbedError, FamilyTxnError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return stored, None


def _stored_chain_gaps(
    stored: Optional[Mapping[str, Any]],
    stored_error: Optional[str],
    chains: Mapping[str, Mapping[str, Any]],
) -> List[LedgerGap]:
    """落盘链与重算链的差异（缺臂 / digest 不符 / 未知臂）。"""
    stored_arms = dict((stored or {}).get("arms") or {})
    gaps: List[LedgerGap] = []
    for arm_id, chain in sorted(chains.items()):
        recorded = stored_arms.get(arm_id)
        if recorded is None:
            if not chain["links"]:
                continue  # 尚无已发布日（genesis 形态）：链产物无意义，不报缺口
            gaps.append(
                _gap(
                    arm_id,
                    None,
                    "chain_stale",
                    f"分片链文件缺该臂条目（可重写）: {arm_id}"
                    + (f"（{stored_error}）" if stored_error else ""),
                    repairable=True,
                )
            )
        elif str(recorded.get("chain_digest")) != str(chain["chain_digest"]):
            gaps.append(
                _gap(
                    arm_id,
                    None,
                    "chain_stale",
                    f"分片链 digest 与重算不一致（可重写）: {arm_id}",
                    repairable=True,
                    actual_sha256=recorded.get("chain_digest"),
                    expected_sha256=chain["chain_digest"],
                )
            )
    if stored is not None:
        for arm_id in sorted(set(stored_arms) - set(chains)):
            gaps.append(
                _gap(
                    arm_id,
                    None,
                    "chain_stale",
                    f"分片链文件含未知臂（可重写）: {arm_id}",
                    repairable=True,
                )
            )
    return gaps


# ---------------------------------------------------------------------------
# 审计主入口
# ---------------------------------------------------------------------------


def _arms_scope(
    layout: CampaignLayout,
    expected: Mapping[str, Mapping[str, Any]],
    index: Optional[Mapping[str, Any]],
) -> List[str]:
    arms = set(expected)
    if layout.ledger_root.is_dir():
        arms |= {path.name for path in layout.ledger_root.iterdir() if path.is_dir()}
    if index:
        arms |= set(index.get("arms") or {})
    return sorted(arms)


def _audit_arm(
    arm_id: str,
    *,
    expected: Mapping[str, Mapping[str, Any]],
    on_disk: Mapping[str, Path],
    manifest: Optional[Dict[str, Any]],
    manifest_error: Optional[str],
    index_days: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[LedgerGap], List[Dict[str, Any]]]:
    """单臂审计：逐日缺口 + 链节序列（按 journal 代序）。"""
    gaps: List[LedgerGap] = []
    links: List[Dict[str, Any]] = []
    for date, item in sorted(expected.items(), key=lambda kv: (int(kv[1]["generation"]), kv[0])):
        day_gaps, actual_sha = _shard_gaps(arm_id, date, item, on_disk.get(date))
        gaps.extend(day_gaps)
        links.append(
            {
                "virtual_date": date,
                "generation": int(item["generation"]),
                "family_txn_id": item["family_txn_id"],
                "shard_sha256": actual_sha,
            }
        )
    gaps.extend(_manifest_gaps(arm_id, expected, manifest, manifest_error))
    gaps.extend(_ordering_gaps(arm_id, expected, manifest))
    gaps.extend(_index_arm_gaps(arm_id, expected, index_days))
    gaps.extend(_stray_gaps(arm_id, expected, on_disk))
    return gaps, links


def audit_ledger(
    layout: CampaignLayout, campaign_id: str, *, clock_days: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """三方一致性审计（journal / 分片 / manifest / index），只读、不落盘。

    Args:
        layout: campaign 布局。
        campaign_id: campaign 标识（与注册记录一致）。
        clock_days: campaign 虚拟窗口（可选；提供时额外校验分片日期不越窗）。

    Returns:
        审计文档：``status`` ∈ {PASS, GAPS, INVALIDATED, JOURNAL_CORRUPT}、逐臂三方计数、
        链摘要、缺口分类计数、INVALIDATED 清单、已发布头与唯一发布点。
    """
    journal = _journal_state(layout)
    if journal.error is not None:
        gap = _gap(
            "*",
            None,
            "journal_corrupt",
            f"journal 不可读（无证据可重建）: {journal.error}",
            repairable=False,
        )
        return _audit_document(
            campaign_id=campaign_id,
            published={},
            arms={},
            chains={},
            chain_digests={},
            gaps=[gap],
            index=None,
        )
    try:
        expected = _expected_shards(campaign_id, journal.committed_by_day)
    except (FamilyTxnError, TestbedError) as exc:
        gap = _gap(
            "*",
            None,
            "journal_corrupt",
            f"journal attempt 缺驱动证据，无法重建分片: {exc}",
            repairable=False,
        )
        return _audit_document(
            campaign_id=campaign_id,
            published={},
            arms={},
            chains={},
            chain_digests={},
            gaps=[gap],
            index=None,
        )
    index, index_error = _index_state(layout)
    arms = _arms_scope(layout, expected, index)
    manifests, manifest_errors = _manifest_state(layout, arms)
    on_disk = _scan_shards(layout)
    index_days = {
        str(entry.get("virtual_date")): dict(entry) for entry in (index or {}).get("days") or []
    }
    gaps: List[LedgerGap] = _journal_order_gaps(journal.committed_by_day)
    gaps.extend(_index_campaign_gaps(index, index_error, journal.committed_by_day))
    links_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    arm_stats: Dict[str, Dict[str, Any]] = {}
    for arm_id in arms:
        arm_expected = expected.get(arm_id, {})
        arm_manifest = manifests.get(arm_id)
        arm_disk = on_disk.get(arm_id, {})
        arm_gaps, links = _audit_arm(
            arm_id,
            expected=arm_expected,
            on_disk=arm_disk,
            manifest=arm_manifest,
            manifest_error=manifest_errors.get(arm_id),
            index_days=index_days,
        )
        gaps.extend(arm_gaps)
        links_by_arm[arm_id] = links
        arm_stats[arm_id] = _arm_stats(arm_id, arm_expected, arm_disk, arm_manifest, index_days)
    gaps.extend(_window_gaps(on_disk, clock_days))
    chains, chain_digests, chain_gaps = _chain_state(layout, campaign_id, links_by_arm)
    gaps.extend(chain_gaps)
    published = {
        "published_generation": (
            None if index is None else int(index.get("published_generation", 0))
        ),
        "published_txn_id": None if index is None else index.get("published_txn_id"),
        "published_virtual_date": None if index is None else index.get("published_virtual_date"),
        # 读侧（index）与证据侧（journal）双值：不等即为 index 缺口，便于判定「落后 / 超前」
        "journal_head_generation": max(
            (int(attempt["target_generation"]) for attempt in journal.committed_by_day.values()),
            default=0,
        ),
        "journal_head_txn_id": (
            max(
                journal.committed_by_day.values(),
                key=lambda attempt: int(attempt["target_generation"]),
            )["family_txn_id"]
            if journal.committed_by_day
            else None
        ),
    }
    return _audit_document(
        campaign_id=campaign_id,
        published=published,
        arms=arm_stats,
        chains=chains,
        chain_digests=chain_digests,
        gaps=gaps,
        index=index,
    )


def _window_gaps(
    on_disk: Mapping[str, Mapping[str, Path]], clock_days: Optional[Sequence[str]]
) -> List[LedgerGap]:
    """分片日期越窗（虚拟时钟回退/越窗的物理证据）。"""
    if not clock_days:
        return []
    window = {str(day) for day in clock_days}
    gaps: List[LedgerGap] = []
    for arm_id, shards in sorted(on_disk.items()):
        for date in sorted(shards):
            if date not in window:
                gaps.append(
                    _gap(
                        arm_id,
                        date,
                        "out_of_order",
                        f"分片日期不在 campaign 虚拟窗口内（越窗/回退）: {arm_id}/{date}",
                        repairable=False,
                    )
                )
    return gaps


def _arm_stats(
    arm_id: str,
    expected: Mapping[str, Mapping[str, Any]],
    on_disk: Mapping[str, Path],
    manifest: Optional[Dict[str, Any]],
    index_days: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """单臂三方计数（分片 / manifest / index 各有多少已发布日）。"""
    manifest_days = set((manifest or {}).get("shards") or {})
    index_arm_days = {
        date for date, entry in index_days.items() if arm_id in ((entry.get("arms") or {}))
    }
    return {
        "journal_days": len(expected),
        "shard_days": len(on_disk),
        "manifest_days": len(manifest_days & set(expected)),
        "index_days": len(index_arm_days & set(expected)),
        "head_generation": max((int(item["generation"]) for item in expected.values()), default=0),
    }


def _audit_document(
    *,
    campaign_id: str,
    published: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    chains: Mapping[str, Mapping[str, Any]],
    chain_digests: Mapping[str, str],
    gaps: Sequence[LedgerGap],
    index: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """组装审计文档（确定性：无墙钟、按类与臂排序）。"""
    ordered_gaps = sorted(gaps, key=lambda gap: (gap.kind, gap.arm_id, str(gap.virtual_date)))
    counts = {kind: 0 for kind in GAP_CLASSES}
    for gap in ordered_gaps:
        counts[gap.kind] += 1
    invalidated = [gap for gap in ordered_gaps if gap.invalidated]
    # 「三方一致」= 计数相等 **且** 无内容类缺口（缺片/坏片/清单或 index 内容不符/混代），
    # 避免坏片这类内容损坏在计数相等时被误读成「一致」。
    content_by_arm: Dict[str, List[str]] = {}
    for gap in ordered_gaps:
        if gap.kind in _CONTENT_GAP_KINDS:
            content_by_arm.setdefault(gap.arm_id, []).append(gap.kind)
    consistent = all(
        item["journal_days"] == item["shard_days"] == item["manifest_days"] == item["index_days"]
        and arm_id not in content_by_arm
        for arm_id, item in arms.items()
    )
    # 设计口径（journal/分片/manifest/index 三方一致）与 T3 旁路产物新鲜度分开计状态，
    # 避免「链文件尚未写」把健康 campaign 稀释成 GAPS。
    design_gaps = [
        gap for gap in ordered_gaps if gap.kind != "chain_stale" and gap.kind in _DESIGN_GAP_KINDS
    ]
    artifact_gaps = [gap for gap in ordered_gaps if gap.kind == "chain_stale"]
    invalidated_design = [gap for gap in design_gaps if gap.invalidated]
    if invalidated_design:
        three_way_status = "INVALIDATED"
    elif design_gaps:
        three_way_status = "GAPS"
    else:
        three_way_status = "PASS"
    artifact_status = "GAPS" if artifact_gaps else "PASS"
    if invalidated:
        status = "INVALIDATED"
    elif ordered_gaps:
        status = "GAPS"
    else:
        status = "PASS"
    return {
        "schema": AUDIT_SCHEMA,
        **_scope_fields(campaign_id),
        "status": status,
        "three_way_status": three_way_status,
        "artifact_status": artifact_status,
        "published": dict(published),
        "unique_publish_point": {
            "field": "journal.attempt.published_txn_id",
            "value": published.get("published_txn_id"),
            "published_generation": published.get("published_generation"),
            "head_gen_id": (
                None
                if published.get("published_generation") is None
                else gen_id(int(published["published_generation"]))
            ),
        },
        "arms": {arm_id: dict(payload) for arm_id, payload in sorted(arms.items())},
        "chain": {
            "arms": {
                arm_id: {
                    "link_count": int(chain["link_count"]),
                    "seed": chain["seed"],
                    "chain_digest": chain["chain_digest"],
                    "links": list(chain["links"]),
                }
                for arm_id, chain in sorted(chains.items())
            },
            "chain_digests": {arm: str(digest) for arm, digest in sorted(chain_digests.items())},
            "chain_digest": (
                campaign_chain_digest(chains)
                if any(chain["links"] for chain in chains.values())
                else None
            ),
        },
        "three_way": {
            "consistent": bool(consistent),
            "arms": {
                arm_id: {
                    "consistent": (
                        item["journal_days"]
                        == item["shard_days"]
                        == item["manifest_days"]
                        == item["index_days"]
                        and arm_id not in content_by_arm
                    ),
                    "content_gaps": sorted(set(content_by_arm.get(arm_id, []))),
                    **dict(item),
                }
                for arm_id, item in sorted(arms.items())
            },
        },
        "gaps": [gap.to_dict() for gap in ordered_gaps],
        "gaps_by_class": counts,
        "invalidated": [
            {
                "arm_id": gap.arm_id,
                "virtual_date": gap.virtual_date,
                "kind": gap.kind,
                "detail": gap.detail,
                "generation": gap.generation,
                "terminal_state": "INVALIDATED",
            }
            for gap in invalidated
        ],
        "repair_required": bool(ordered_gaps),
        "index_present": index is not None,
        "index_published_generation": (
            None if index is None else int(index.get("published_generation", 0))
        ),
    }


# ---------------------------------------------------------------------------
# 旁路产物落盘（链 / 审计）
# ---------------------------------------------------------------------------


def build_chain_document(campaign_id: str, audit: Mapping[str, Any]) -> Dict[str, Any]:
    """由审计结果构造 campaign 级链文档（确定性、封存）。"""
    arms = dict((audit.get("chain") or {}).get("arms") or {})
    return seal_record(
        {
            "schema": CHAIN_SCHEMA,
            **_scope_fields(campaign_id),
            "arms": {
                arm_id: {
                    "seed": payload["seed"],
                    "link_count": payload["link_count"],
                    "links": payload["links"],
                    "chain_digest": payload["chain_digest"],
                }
                for arm_id, payload in sorted(arms.items())
            },
            # genesis 形态（无任何链节）与 audit 侧同口径：campaign 级 digest 为 None
            "chain_digest": (
                (audit.get("chain") or {}).get("chain_digest")
                if any(payload["links"] for payload in arms.values())
                else None
            ),
            "published_generation": (audit.get("published") or {}).get("published_generation"),
            "published_txn_id": (audit.get("published") or {}).get("published_txn_id"),
        }
    )


def write_ledger_artifacts(
    layout: CampaignLayout, campaign_id: str, audit: Mapping[str, Any]
) -> Dict[str, Any]:
    """把链与审计结果落 ``audit/``（内容相同即零写盘；生产根硬禁）。"""
    guard_ledger_write(layout)
    paths = ledger_paths(layout)
    chain_written = write_if_changed(
        paths.chain_path, build_chain_document(campaign_id, audit), what="分片链"
    )
    audit_written = write_if_changed(paths.audit_path, seal_record(audit), what="账本审计")
    return {"chain_written": bool(chain_written), "audit_written": bool(audit_written)}
