# -*- coding: utf-8 -*-
"""v2 权威纸面状态的手工调整/重置服务；投影只由快照确定性重建。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.lazybull.v2.core.accounting.engine_state import (
    dumps_blob,
    loads_blob,
    state_from_blob,
    state_to_blob,
)
from src.lazybull.v2.hosts.paper.adjustment_rules import apply_adjustment
from src.lazybull.v2.hosts.paper.config_adapter import PaperHostError
from src.lazybull.v2.hosts.paper.factory import create_paper_host
from src.lazybull.v2.hosts.paper.state_store import (
    GENERATION_INFO_KEY,
    PaperStateStore,
    build_account_projection,
)


def _load_state(store: PaperStateStore):  # noqa: ANN202
    raw = store.load_engine_state_blob()
    if raw is None:
        raise PaperHostError("v2 adjust 需要权威 engine_state.bin；请先完成账户迁移")
    blob = loads_blob(raw)
    return blob, state_from_blob(blob)


def _commit_state(
    store: PaperStateStore,
    blob: Mapping[str, Any],
    state: Any,
    *,
    trade_date: str,
    data_root: Optional[str],
    host: Any = None,
) -> Dict[str, Any]:
    generation = dict(blob.get(GENERATION_INFO_KEY) or {})
    if not generation.get("trade_date"):
        raise PaperHostError("权威快照缺 paper_generation，拒绝手工调整后失去恢复身份")
    new_blob = state_to_blob(state)
    new_blob[GENERATION_INFO_KEY] = generation
    store._atomic_write_bytes(store.state_path / "engine_state.bin", dumps_blob(new_blob))
    # 人工调整就是经授权的显式修复：先由权威快照重投影，再执行完整代检查/队列重建。
    store.repair_projections()
    if host is None:
        host = create_paper_host(store=store, data_root=data_root)
    host._ensure_generation_complete(str(generation["trade_date"]))
    account = build_account_projection(state.current_capital, state.positions, trade_date)
    store._atomic_write_json(store.state_path / "account.json", account)
    return account


def adjust_v2_account(
    store: PaperStateStore,
    *,
    command: str,
    trade_date: str,
    values: Mapping[str, Any],
    data_root: Optional[str] = None,
    host: Any = None,
) -> Dict[str, Any]:
    """持整日写锁修改权威状态并重建全部投影；第二写入者 fail-fast。"""
    with store.write_lock():
        blob, state = _load_state(store)
        apply_adjustment(state, command=command, trade_date=trade_date, values=values)
        return _commit_state(
            store, blob, state, trade_date=trade_date, data_root=data_root, host=host
        )


def reset_v2_t0(
    store: PaperStateStore, *, trade_date: str, data_root: Optional[str] = None
) -> Dict[str, Any]:
    """清空最近 T0 及延迟队列，不重置账户现金/持仓；随后由权威快照重投影。"""
    with store.write_lock():
        blob, state = _load_state(store)
        if state.pending_signals:
            latest = max(state.pending_signals)
            del state.pending_signals[latest]
        state.pending_condition_sells = {}
        state.pending_stop_loss_sells = {}
        if state.pending_order_manager is not None:
            state.pending_order_manager["pending_orders"] = {}
        return _commit_state(store, blob, state, trade_date=trade_date, data_root=data_root)
