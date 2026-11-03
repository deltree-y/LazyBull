# -*- coding: utf-8 -*-
"""P5b T4 配对账本样本：从试车场已发布账本构造 ``d_t``（设计 v4 §5.1）。

读侧口径
========
1. **入口走 T3 只读查询** ``query_shards(layout, arm_id, start, end)``：它返回已发布区间、
   连续性、链 digest 与 ``published_generation``，并对 INVALIDATED 区间 fail-closed
   （``allow_invalidated=True`` 才放行，属审计旁路）；
2. 通过查询门后，逐日 **NAV 只从校验过的分片读取**（``journal.load_shard`` 会复算封存哈希），
   取值口径 = 分片 ``evidence.valuation.nav``（T2 的**唯一权威估值**；不用展示 NAV，也不从
   ``nav.parquet`` 推断）。直接读分片的原因：``query_shards`` 只返回分片 sha256 与区间元数据，
   不含 NAV 载荷，而重算 NAV 需要完整 state 根（已随代际替换）；因此「查询门 + 校验分片读」是
   既保持 fail-closed、又能取到权威估值的最小口径。
3. 两臂日期序列必须**逐日对齐**且连续；``D0`` 只作估值日，``D1`` 起产生第一个相邻日收益；
4. ``d_t = ln(V_a,t/V_a,t-1) - ln(V_b,t/V_b,t-1)``；**禁止**用逐日独立方差或单臂序列冒充配对。

本模块只读：不写账本、不写持仓、不推进时钟。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from src.lazybull.v2.hosts.testbed.journal import load_shard
from src.lazybull.v2.hosts.testbed.layout import CampaignLayout
from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.referee.scenes import paired_log_diff

__all__ = [
    "LedgerSampleError",
    "MIN_PAIRED_RETURNS",
    "ledger_sample_identity",
    "nav_series_from_ledger",
    "paired_returns_from_ledger",
]

#: 配对收益最少条数（一条只够估一个收益；统计至少要能形成 block）
MIN_PAIRED_RETURNS = 1


class LedgerSampleError(TestbedError):
    """配对账本样本不可构造（缺片/不连续/NAV 非法/日期不对齐）。"""

    __test__ = False


def nav_series_from_ledger(
    layout: CampaignLayout,
    arm_id: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    allow_invalidated: bool = False,
) -> Dict[str, Any]:
    """按只读查询门取单臂已发布 NAV 序列（日期升序，含 D0 估值点）。

    Raises:
        LedgerInvalidatedError: 区间含 INVALIDATED 日且未显式放开（T3 fail-closed）。
        LedgerSampleError: 区间不连续、缺口、分片缺 NAV 或非有限。
    """
    from src.lazybull.v2.hosts.testbed.ledger_repair import query_shards

    result = query_shards(layout, arm_id, start=start, end=end, allow_invalidated=allow_invalidated)
    if result["gaps"]:
        raise LedgerSampleError(
            f"臂 {arm_id!r} 的查询区间存在账本缺口（拒绝作为统计输入）: {result['gaps']}"
        )
    if not result["contiguous"]:
        raise LedgerSampleError(
            f"臂 {arm_id!r} 的查询区间不连续（禁止跳日拼样本）: "
            f"{[day['virtual_date'] for day in result['days']]}"
        )
    dates: List[str] = []
    nav: List[float] = []
    for day in result["days"]:
        date = str(day["virtual_date"])
        shard = load_shard(layout, arm_id, date)
        if shard is None:
            raise LedgerSampleError(f"臂 {arm_id!r} 分片缺失: {date}")
        valuation = dict((shard.get("evidence") or {}).get("valuation") or {})
        value = valuation.get("nav")
        if value is None or not np.isfinite(float(value)) or float(value) <= 0:
            raise LedgerSampleError(
                f"臂 {arm_id!r} 在 {date} 的权威估值不可用（必须为有限正值 NAV）: {value!r}"
            )
        dates.append(date)
        nav.append(float(value))
    return {
        "arm_id": arm_id,
        "dates": dates,
        "nav": nav,
        "generations": [int(day["generation"]) for day in result["days"]],
        "shard_sha256": [day["shard_sha256"] for day in result["days"]],
        "chain_digest": result["chain_digest"],
        "published_generation": result["published_generation"],
    }


def paired_returns_from_ledger(
    layout: CampaignLayout,
    arm_a: str,
    arm_b: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    allow_invalidated: bool = False,
) -> Dict[str, Any]:
    """从试车场配对账本构造 ``d_t``（a = 候选臂，b = 基线臂）。

    Args:
        layout: campaign 布局。
        arm_a: 候选臂 id。
        arm_b: 基线臂 id。
        start/end: 查询区间（含端点；None = 全窗）。
        allow_invalidated: 审计旁路（默认 False = INVALIDATED 区间 fail-closed）。

    Returns:
        ``{dates, returns, d0_date, arm_order, published_generation, chain_digests,
        generation_span, sample_days, read_path}``。

    Raises:
        LedgerSampleError: 区间不可用/日期不对齐/样本不足。
        LedgerInvalidatedError: 区间含 INVALIDATED 日（fail-closed）。
    """
    series_a = nav_series_from_ledger(
        layout, arm_a, start=start, end=end, allow_invalidated=allow_invalidated
    )
    series_b = nav_series_from_ledger(
        layout, arm_b, start=start, end=end, allow_invalidated=allow_invalidated
    )
    if series_a["dates"] != series_b["dates"]:
        raise LedgerSampleError(
            "两臂已发布日期序列不对齐（禁止用不同日期窗口拼配对差）: "
            f"{len(series_a['dates'])} vs {len(series_b['dates'])}"
        )
    dates = list(series_a["dates"])
    if len(dates) < MIN_PAIRED_RETURNS + 1:
        raise LedgerSampleError(
            f"配对收益至少需要 {MIN_PAIRED_RETURNS + 1} 个已发布估值点（D0 只作估值日），"
            f"实际 {len(dates)} 个: {dates}"
        )
    returns = paired_log_diff(series_a["nav"], series_b["nav"])
    return {
        "dates": dates[1:],
        "d0_date": dates[0],
        "returns": returns,
        "sample_days": int(returns.size),
        "arm_order": [arm_a, arm_b],
        "published_generation": {
            arm_a: series_a["published_generation"],
            arm_b: series_b["published_generation"],
        },
        "chain_digests": {arm_a: series_a["chain_digest"], arm_b: series_b["chain_digest"]},
        "generation_span": {
            arm_a: [min(series_a["generations"]), max(series_a["generations"])],
            arm_b: [min(series_b["generations"]), max(series_b["generations"])],
        },
        "read_path": "query_shards(只读查询门) + load_shard(校验分片) → evidence.valuation.nav",
        "evidence_class": "calibration_only",
    }


def ledger_sample_identity(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """配对样本身份摘要（报告引用；不含 NAV 载荷）。"""
    return {
        "arm_order": list(sample["arm_order"]),
        "d0_date": sample["d0_date"],
        "first_return_date": (sample["dates"][0] if sample["dates"] else None),
        "last_return_date": (sample["dates"][-1] if sample["dates"] else None),
        "sample_days": int(sample["sample_days"]),
        "published_generation": dict(sample["published_generation"]),
        "chain_digests": dict(sample["chain_digests"]),
        "read_path": sample["read_path"],
    }
