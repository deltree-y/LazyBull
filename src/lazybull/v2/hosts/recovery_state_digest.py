"""P6.5 T5 逐位状态投影与比较（G5 的判据实现）。

口径（冻结，禁止放宽）：
- 所有 float64 经 ``f64()``（``float.hex``）编码，整数股归一为 ``int``，键序/批次序
  原样保留——因此"规范化字节相等"等价于"值与位模式逐位一致"，不需要任何容差；
- P4 演化状态用 ``normalize_p4_value`` 做**类型归一投影**（np 标量归一、Timestamp →
  ISO、映射保序），而不是比较 pickle blob 字节：``pickle`` 协议 5 用 memo 表示重复
  对象引用，同一 ``Timestamp`` 在参照链内被多条目共享、跨恢复边界后是等价但不同的
  实例，字节因此不同而值完全相同（P4 自身的 export→restore→export 字节一致由
  ``engine_state`` 保证，不在本模块判据内）。

边界：只做投影与比较，不写任何状态；不读配置、不读真实行情。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from src.lazybull.v2.core.accounting.g4_nav import float64_bits, g4_merged_nav
from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.core.accounting.state_envelope import (
    canonical_json_bytes,
    f64,
    sha256_bytes,
)


def _trade_repr(host: Any) -> str:
    """成交日志逐条 ``repr``（float 用 ``repr`` 位精确；顺序即落账顺序）。"""
    return repr([dict(item) for item in host.trades])


def normalize_p4_value(value: Any) -> Any:
    """P4 演化状态值 → 类型归一的规范化投影（**值/位模式/键序**判据）。

    为什么不直接比 P4 blob 字节：``pickle`` 协议 5 用 memo 表示**重复对象引用**，
    同一 ``Timestamp`` 在参照链内可能被多条目共享、跨恢复边界后是等价但不同的
    实例——字节因此不同而值完全相同。故跨恢复边界的逐位比较以本投影为判据：
    float64 → ``float.hex``（位模式）、整数 → ``int``（np 整型归一）、
    ``Timestamp`` → ISO 字符串、映射 → **保序**键值对序列、序列 → 保序列表。
    """
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, np.floating):
        return f64(float(value))
    if isinstance(value, float):
        return f64(float(value))
    if isinstance(value, pd.Timestamp):
        return {"$ts": value.isoformat()}
    if isinstance(value, (datetime, date)):
        return {"$date": value.isoformat()}
    if isinstance(value, Mapping):
        return {
            "$map": [
                [normalize_p4_value(key), normalize_p4_value(item)]
                for key, item in value.items()
            ]
        }
    if isinstance(value, pd.Series):
        return {
            "$series": [normalize_p4_value(item) for item in value.tolist()],
            "index": [normalize_p4_value(item) for item in value.index.tolist()],
        }
    if isinstance(value, (list, tuple)):
        return {
            "$seq": [normalize_p4_value(item) for item in value],
            "kind": type(value).__name__,
        }
    return {"$other": type(value).__name__, "$repr": repr(value)}


def p4_state_projection(host: Any) -> Dict[str, Any]:
    """P4 演化状态 → 字段名有序的规范化投影（21 字段全覆盖，含子对象）。"""
    from src.lazybull.v2.core.accounting.engine_state import (
        SNAPSHOT_FIELDS,
        SNAPSHOT_SUB_OBJECTS,
        export_state,
    )

    state = export_state(host)
    fields: List[List[Any]] = []
    for name in list(SNAPSHOT_FIELDS) + list(SNAPSHOT_SUB_OBJECTS):
        fields.append([name, normalize_p4_value(getattr(state, name))])
    return {"schema_version": int(_engine_state_schema_version()), "fields": fields}


def _engine_state_schema_version() -> int:
    """P4 快照 schema 版本（只读引用；禁止在本模块重定义）。"""
    from src.lazybull.v2.core.accounting.engine_state import ENGINE_STATE_SCHEMA_VERSION

    return ENGINE_STATE_SCHEMA_VERSION


class RecoveryDigestError(ValueError):
    """逐位投影输入拒绝（缺当日快照等；**输入拒绝**语义，与完整性破缺分离）。

    T5-A-R2-02：本模块不依赖 ``recovery_runtime``（避免循环导入），因此自带一个
    **分层明确**的输入拒绝类型；两种拒绝语义仍严格分开——本类为 ``ValueError``
    （可修正的输入问题），``RecoveryIntegrityError`` 为 ``RuntimeError``（结论破缺）。
    """


def _portfolio_value_of(host: Any, date: Any) -> float:
    """按**该交易日**取物理组合总值（不是"最后一个快照"）。

    T5-A-R1-04：原实现取 ``portfolio_values[-1]``，在"跑完多日后再逐日取投影"时会把
    最后一日总值贴到每一天的标签上，使"逐日投影"名不副实。这里按日期精确匹配；
    缺失即拒绝（不允许用邻日估值冒充当日）。
    """
    target = pd.Timestamp(date)
    for item in host.portfolio_values:
        if pd.Timestamp(item["date"]) == target:
            return float(item["portfolio_value"])
    raise RecoveryDigestError(
        f"阶段=P6.5 恢复驱动 逐位投影：物理组合总值缺少 {target.date()} 的快照；"
        "按日投影必须使用该交易日自己的 portfolio_values（禁止用邻日估值冒充）"
    )


def state_digest(
    host: Any,
    *,
    scenario: Any,
    day_index: int,
) -> Dict[str, Any]:
    """单日状态的**逐位**投影（物理/虚拟/身份/NAV 全覆盖；float64 位编码）。

    形参类型注记：``host`` 为 ``VirtualNativeMergeBacktestEngineML``、``scenario`` 为
    ``RecoveryScenario``；本模块被 ``recovery_runtime`` 导入，故按 ``Any`` 标注以避免
    循环导入（运行时行为不依赖该标注）。

    所有 float 一律经 ``f64()``（``float.hex``）编码，因此两次 digest 的规范化字节
    相等即等价于"逐位一致"，无需任何容差；``digest_sha256`` 是该字节的摘要。
    """
    date = scenario.date_list()[day_index]
    ledger = host.virtual_ledger
    states = ledger.snapshot()
    source = host._build_price_source(date, states=states)
    physical_total_value = _portfolio_value_of(host, date)
    initial = float(scenario.physical_initial_capital)
    g4 = g4_merged_nav(states, source, initial, context=host._identity_context(
        date, states, source.source_id
    ))
    sleeves = []
    for sleeve_id in sorted(states):
        state = states[sleeve_id]
        total_value = float(state.total_value(source))
        sleeves.append(
            {
                "sleeve_id": sleeve_id,
                "cash": f64(state.cash),
                "borrowed_credit": f64(state.borrowed_credit),
                "total_value": f64(total_value),
                "nav_contrib": f64(float(calculate_nav(total_value, initial))),
                "lots": [
                    {
                        "virtual_lot_id": lot.virtual_lot_id,
                        "source_lot_id": lot.source_lot_id,
                        "shares": int(lot.shares),
                        "acquired_date": str(lot.acquired_date),
                        "acquired_price": f64(lot.acquired_price),
                        "acquired_fee": f64(lot.acquired_fee),
                    }
                    for lot in state.lots
                ],
            }
        )
    view = host.physical_lot_view
    ledger_state = host.ledger.export_state()
    digest = {
        "day_index": int(day_index),
        "trade_date": str(source.as_of),
        "physical": {
            "cash": f64(host.current_capital),
            "positions": [
                [str(code), int(info["shares"])]
                for code, info in sorted(host.positions.items())
            ],
            "trades_repr": _trade_repr(host),
            "trade_count": len(host.trades),
            "total_value": f64(physical_total_value),
            "nav": f64(float(calculate_nav(physical_total_value, initial))),
        },
        "physical_lot_view": {
            "order": [[code, lot_id] for code, lot_id in view.batch_order()],
            "lots": [
                [code, lot_id, int(shares), f64(cost)]
                for code, lot_id, shares, cost in view.contents()
            ],
            "events": [
                [label, list(fingerprint)] for label, fingerprint in view.event_watermark()
            ],
            "origin_fingerprint": view.origin_fingerprint(),
        },
        "virtual": sleeves,
        "identity": {
            "physical_applied_keys": [str(key) for key in ledger_state["applied_keys"]],
            "physical_last_fill_date": (
                None
                if ledger_state["last_fill_date"] is None
                else str(ledger_state["last_fill_date"])
            ),
            "settlement_keys": [str(key) for key in ledger.settlement_ids()],
            "batch_ids": [str(key) for key in ledger.batch_ids()],
            "last_settlement_id": ledger.last_settlement_id,
            "last_settlement_source_id": ledger.last_settlement_source_id,
            "poisoned": ledger.poisoned_reason,
            "price_source_id": source.source_id,
            "price_source_as_of": str(source.as_of),
        },
        "nav": {
            "merged_total_value": f64(g4.merged_total_value),
            "merged_nav": f64(g4.merged_nav),
            "merged_nav_bits": float64_bits(g4.merged_nav),
            "physical_nav_bits": float64_bits(float(calculate_nav(physical_total_value, initial))),
        },
        "p4_state": p4_state_projection(host),
    }
    return digest


def digest_sha256(digest: Mapping[str, Any]) -> str:
    """投影 → 规范化字节摘要（逐位一致的**唯一判据**）。"""
    return sha256_bytes(canonical_json_bytes(dict(digest)))


def _flatten(node: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """把嵌套投影压平为 ``(路径, repr)`` 列表（逐字段差异定位）。"""
    flat: List[Tuple[str, str]] = []
    if isinstance(node, Mapping):
        for key in sorted(node, key=str):
            flat.extend(_flatten(node[key], f"{prefix}.{key}" if prefix else str(key)))
        return flat
    if isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            flat.extend(_flatten(value, f"{prefix}[{index}]"))
        return flat
    flat.append((prefix, repr(node)))
    return flat


def compare_digests(
    reference: Mapping[str, Any], recovered: Mapping[str, Any]
) -> Tuple[str, ...]:
    """逐字段比较两份投影（**逐位**：差异即返回路径与左右值）。"""
    left = dict(_flatten(reference))
    right = dict(_flatten(recovered))
    differences: List[str] = []
    for key in sorted(set(left) | set(right)):
        left_value = left.get(key, "<缺失>")
        right_value = right.get(key, "<缺失>")
        if left_value != right_value:
            differences.append(f"{key}: 参照={left_value} 恢复={right_value}")
    return tuple(differences)
