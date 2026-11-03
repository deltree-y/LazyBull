# -*- coding: utf-8 -*-
"""P5b 试车场 T3 分片链（确定性链摘要；纯函数、无 I/O、无墙钟）。

契约来源：设计 v4 §4.3（分片 / manifest / index 状态与内容身份）、§7.1（产物根）、
§7.2（报告字段「分片链 digest」）、§8-H3（分片 + manifest + index 一致）。

链定义（逐片 sha → 链 digest）::

    seed      = sha256_canonical({schema, campaign_id, arm_id})
    chain[i]  = sha256_canonical({prev_chain, virtual_date, generation, shard_sha256})
    digest    = chain[-1]

口径：

- 逐片 ``shard_sha256`` = :func:`journal.shard_sha256`（分片内容 canonical 哈希，排除封存字段），
  与 ``manifest.json`` / ``_index.json`` 记录的 ``sha256`` 同口径；
- 缺片以哨兵 :data:`CHAIN_LINK_MISSING` 参与链推进 ⇒ **缺片也改变链 digest**，按 journal 补齐后
  回到原值（修复可验证性）；
- 链只由「已发布日序 + 分片内容」决定，不含墙钟、不含路径 ⇒ 可反复重算且逐字节一致；
- 篡改任意一片 ⇒ 该片起链值全变 ⇒ 链 digest 变化，配合审计可精确定位到「臂 + 日期」。

本模块**不读盘不写盘**：全部输入由调用方（``ledger_audit``）提供，便于单测链数学本身。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.lazybull.v2.hosts.testbed.layout import FamilyTxnError, gen_id
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "CHAIN_LINK_MISSING",
    "CHAIN_SCHEMA",
    "advance_chain",
    "build_chain",
    "campaign_chain_digest",
    "chain_seed",
    "verify_chain",
]

#: 链文档 schema
CHAIN_SCHEMA = "p5b_testbed_shard_chain/v1"
#: 缺片哨兵（参与链推进，保证「缺片 → 链变化」可定位）
CHAIN_LINK_MISSING = "MISSING"


def chain_seed(campaign_id: str, arm_id: str) -> str:
    """链起点：把（campaign, 臂）绑定到链上，禁止跨臂/跨 campaign 拼接复用。"""
    return canonical_json_sha256(
        {"schema": CHAIN_SCHEMA, "campaign_id": str(campaign_id), "arm_id": str(arm_id)}
    )


def advance_chain(
    prev_chain: str,
    *,
    virtual_date: str,
    generation: int,
    shard_sha256: Optional[str],
) -> str:
    """推进一节链（缺片用哨兵，仍推进）。"""
    return canonical_json_sha256(
        {
            "prev_chain": str(prev_chain),
            "virtual_date": str(virtual_date),
            "generation": int(generation),
            "shard_sha256": CHAIN_LINK_MISSING if not shard_sha256 else str(shard_sha256),
        }
    )


def _require_ordered(links: Sequence[Mapping[str, Any]]) -> None:
    """链只接受「代升序 + 日期升序」的链节序列（错序必须由审计先判定）。"""
    previous_generation: Optional[int] = None
    previous_date = ""
    for index, link in enumerate(links):
        generation = int(link["generation"])
        date = str(link["virtual_date"])
        if previous_generation is not None and generation <= previous_generation:
            raise FamilyTxnError(
                f"链节代序非严格升序（第 {index} 节 {generation} <= {previous_generation}），"
                "错序须先由审计判定，禁止直接建链"
            )
        if previous_date and date <= previous_date:
            raise FamilyTxnError(
                f"链节日期非严格升序（第 {index} 节 {date!r} <= {previous_date!r}）"
            )
        previous_generation, previous_date = generation, date


def build_chain(
    campaign_id: str, arm_id: str, links: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    """按「已发布日序」构造单臂分片链。

    Args:
        campaign_id: campaign 标识。
        arm_id: 臂标识。
        links: 链节序列（须按代升序、日期升序）；每节至少含
            ``virtual_date`` / ``generation`` / ``family_txn_id`` / ``shard_sha256``。
            ``shard_sha256`` 为 None 表示该日缺片。

    Returns:
        ``{schema, arm_id, seed, link_count, links[], chain_digest}``；``links[i]`` 含该节
        累计 ``chain_digest``（审计定位用）。

    Raises:
        FamilyTxnError: 链节代序或日期序非严格升序。
    """
    ordered: List[Mapping[str, Any]] = [dict(link) for link in links]
    _require_ordered(ordered)
    seed = chain_seed(campaign_id, arm_id)
    chain = seed
    out: List[Dict[str, Any]] = []
    for index, link in enumerate(ordered):
        date = str(link["virtual_date"])
        generation = int(link["generation"])
        sha = link.get("shard_sha256")
        chain = advance_chain(chain, virtual_date=date, generation=generation, shard_sha256=sha)
        out.append(
            {
                "index": index,
                "virtual_date": date,
                "generation": generation,
                "gen_id": gen_id(generation),
                "family_txn_id": link.get("family_txn_id"),
                "shard_sha256": None if not sha else str(sha),
                "chain_digest": chain,
            }
        )
    return {
        "schema": CHAIN_SCHEMA,
        "arm_id": str(arm_id),
        "seed": seed,
        "link_count": len(out),
        "links": out,
        "chain_digest": chain,
    }


def campaign_chain_digest(per_arm: Mapping[str, Mapping[str, Any]]) -> str:
    """campaign 级链 digest = 各臂链 digest 的 canonical 哈希（按臂名排序）。"""
    payload = {arm_id: str(chain["chain_digest"]) for arm_id, chain in sorted(per_arm.items())}
    return canonical_json_sha256({**payload, "schema": CHAIN_SCHEMA})


def verify_chain(document: Mapping[str, Any], campaign_id: str) -> Dict[str, Dict[str, Any]]:
    """重算并校验整份 **campaign 级** 链文档（链文件被改写 / 与分片不一致即失败）。

    Returns:
        ``{arm_id: 重算后的单臂链}``。

    Raises:
        FamilyTxnError: schema 不符、逐节/末节链值无法重算一致，或 campaign 级 digest 不符。
    """
    if document.get("schema") != CHAIN_SCHEMA:
        raise FamilyTxnError(f"分片链 schema 非法: {document.get('schema')!r}")
    arms = dict(document.get("arms") or {})
    rebuilt: Dict[str, Dict[str, Any]] = {}
    for arm_id, payload in sorted(arms.items()):
        links = [dict(link) for link in payload.get("links") or []]
        chain = build_chain(campaign_id, arm_id, links)
        if chain["chain_digest"] != payload.get("chain_digest"):
            raise FamilyTxnError(
                f"分片链 digest 无法重算一致（臂 {arm_id!r}：记录="
                f"{payload.get('chain_digest')} 实际={chain['chain_digest']}）"
            )
        recorded_links = payload.get("links") or []
        for position, link in enumerate(chain["links"]):
            if dict(recorded_links[position]) != link:
                raise FamilyTxnError(
                    f"分片链第 {position} 节与重算不一致（{arm_id}@{link['virtual_date']}）"
                )
        rebuilt[arm_id] = chain
    # genesis 形态（逐臂 0 链节）与审计/产物侧同口径：campaign 级 digest 为 None
    expected = (
        campaign_chain_digest(rebuilt)
        if any(payload["links"] for payload in rebuilt.values())
        else None
    )
    if document.get("chain_digest") != expected:
        raise FamilyTxnError(
            f"campaign 级链 digest 无法重算一致（记录={document.get('chain_digest')} "
            f"实际={expected}）"
        )
    return rebuilt
