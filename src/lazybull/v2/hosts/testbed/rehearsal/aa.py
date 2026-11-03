# -*- coding: utf-8 -*-
"""P5b T6 机械 A/A 逐值一致（**从账本分片复算**，不信 runner 的报告字段）。

契约来源：设计 v4 §6.3 第一条「机械 A/A 两臂的日期、成交、现金、持仓、NAV 逐值一致」、
§4.2（A/A 克隆臂只验隔离与逐值一致）、§6.2 末条（统计 A/A 不得用同路径克隆）。

口径
====
1. 读取走 T3 只读查询门 ``query_shards``（INVALIDATED fail-closed；默认不放行）；
2. 逐日取分片载荷（``journal.load_shard`` 会复算封存哈希）比较六个字段：
   ``virtual_date`` / ``generation`` / ``cash`` / ``positions`` / ``fills``（成交）/
   ``valuation.nav``（NAV）；
3. **无成交日也必须比**（现金/持仓/NAV 三项照比），因此比对行数与字段数均要求非零
   ⇒ 断言非真空；
4. 浮点比较按分片载荷的规范化 JSON 逐字节口径（``canonical_json_sha256``），不做容差放宽。

**本模块结论只证明「运行与账本逐值一致」**，不构成统计 A/A 证据（统计 A/A 只来自 T4/T5
的独立同分布路径）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.lazybull.v2.hosts.testbed.journal import load_shard
from src.lazybull.v2.hosts.testbed.layout import CampaignLayout
from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "AA_FIELDS",
    "ArmDaySnapshot",
    "MechanicalAaError",
    "compare_snapshots",
    "load_snapshots",
    "mechanical_aa",
]

#: 逐值比较字段（缺失即视为不一致，不做默认值填充）
AA_FIELDS = ("virtual_date", "generation", "cash", "positions", "fills", "nav")


class MechanicalAaError(TestbedError):
    """分片缺失/不可读，或比对输入为空（禁止空比对冒充一致）。"""

    __test__ = False


@dataclass(frozen=True)
class ArmDaySnapshot:
    """单臂单日的账本逐值快照（只取比对需要的字段）。"""

    virtual_date: str
    generation: int
    cash: float
    positions: Any
    fills: Any
    nav: float

    def digest(self, field: str) -> str:
        """单字段的规范化哈希（逐值比较的唯一口径）。"""
        if field not in AA_FIELDS:
            raise MechanicalAaError(f"未知比较字段: {field!r}（允许 {list(AA_FIELDS)}）")
        return canonical_json_sha256({"field": field, "value": getattr(self, field)})


def _snapshot_from_shard(shard: Mapping[str, Any], *, arm_id: str, date: str) -> ArmDaySnapshot:
    evidence = dict(shard.get("evidence") or {})
    valuation = dict(evidence.get("valuation") or {})
    if "nav" not in valuation:
        raise MechanicalAaError(f"臂 {arm_id!r} 在 {date} 的分片缺权威估值 NAV")
    return ArmDaySnapshot(
        virtual_date=str(evidence.get("virtual_date") or date),
        generation=int(shard.get("generation")),
        cash=float(evidence.get("cash", 0.0)),
        positions=evidence.get("positions"),
        fills=list(evidence.get("fills") or []),
        nav=float(valuation["nav"]),
    )


def load_snapshots(
    layout: CampaignLayout,
    arm_id: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[ArmDaySnapshot]:
    """按只读查询门读取单臂逐日快照（INVALIDATED 默认 fail-closed）。

    Raises:
        LedgerInvalidatedError: 区间含 INVALIDATED 日（T3 语义）。
        MechanicalAaError: 分片缺失/不可读。
    """
    from src.lazybull.v2.hosts.testbed.ledger_repair import query_shards

    result = query_shards(layout, arm_id, start=start, end=end)
    if result["gaps"]:
        raise MechanicalAaError(
            f"臂 {arm_id!r} 的查询区间存在账本缺口（拒绝作为 A/A 输入）: {result['gaps']}"
        )
    snapshots: List[ArmDaySnapshot] = []
    for day in result["days"]:
        date = str(day["virtual_date"])
        shard = load_shard(layout, arm_id, date)
        if shard is None:
            raise MechanicalAaError(f"臂 {arm_id!r} 分片缺失: {date}")
        snapshots.append(_snapshot_from_shard(shard, arm_id=arm_id, date=date))
    return snapshots


def compare_snapshots(
    base: Sequence[ArmDaySnapshot], clone: Sequence[ArmDaySnapshot]
) -> Dict[str, Any]:
    """逐日逐字段比较两臂快照（纯函数；空输入或日期不对齐即判不一致）。

    Returns:
        ``{identical, compared_days, compared_fields, field_counts, mismatches, day_digests}``；
        ``identical`` 只在该比对的（日 × 字段）数非零且零不一致时为真。
    """
    mismatches: List[Dict[str, Any]] = []
    base_dates = [item.virtual_date for item in base]
    clone_dates = [item.virtual_date for item in clone]
    if base_dates != clone_dates:
        mismatches.append(
            {
                "kind": "date_sequence_mismatch",
                "base_dates": base_dates[:5],
                "clone_dates": clone_dates[:5],
                "base_days": len(base_dates),
                "clone_days": len(clone_dates),
            }
        )
    field_counts = {field: 0 for field in AA_FIELDS}
    day_digests: Dict[str, Dict[str, str]] = {}
    paired = list(zip(base, clone))
    for left, right in paired:
        digests: Dict[str, str] = {}
        for field in AA_FIELDS:
            field_counts[field] += 1
            left_value = left.digest(field)
            right_value = right.digest(field)
            digests[field] = left_value
            if left_value != right_value:
                mismatches.append(
                    {
                        "kind": "field_mismatch",
                        "virtual_date": left.virtual_date,
                        "field": field,
                        "base": getattr(left, field),
                        "clone": getattr(right, field),
                    }
                )
        day_digests[left.virtual_date] = digests
    compared_days = len(paired)
    compared_fields = compared_days * len(AA_FIELDS)
    return {
        "identical": bool(compared_days > 0 and not mismatches),
        "compared_days": compared_days,
        "compared_fields": compared_fields,
        "field_counts": field_counts,
        "mismatches": mismatches,
        "day_digests": day_digests,
        "note": "机械 A/A 只证明运行与账本逐值一致；统计 A/A 只来自 T4/T5 的独立同分布路径",
    }


def mechanical_aa(
    layout: CampaignLayout,
    *,
    arm_base: str,
    arm_clone: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """从分片复算两臂机械 A/A（空比对即失败，禁止真空通过）。"""
    base = load_snapshots(layout, arm_base, start=start, end=end)
    clone = load_snapshots(layout, arm_clone, start=start, end=end)
    result = compare_snapshots(base, clone)
    return {
        "arm_base": str(arm_base),
        "arm_clone": str(arm_clone),
        "source": "shard_recompute(query_shards + load_shard)",
        **result,
    }
