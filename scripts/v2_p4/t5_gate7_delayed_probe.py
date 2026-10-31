# -*- coding: utf-8 -*-
# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
"""门 7 同账户延迟队列探针：停牌买单 → adjust → 回滚 → 次日 legacy 续跑。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4.t5_migration import rollback_account
from scripts.v2_p4.t5_three_arm import ArmSpec, run_legacy_arm
from src.lazybull.paper.storage import PaperStorage
from src.lazybull.v2.core.accounting.engine_state import (
    dumps_blob,
    loads_blob,
    state_from_blob,
    state_to_blob,
)
from src.lazybull.v2.hosts.paper.config_adapter import load_paper_config
from src.lazybull.v2.hosts.paper.entrypoints import (
    assert_legacy_rollback_allowed,
    run_v2_day,
)
from src.lazybull.v2.hosts.paper.state_service import adjust_v2_account
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore

SOURCE = ROOT / "data/paper_shadow/t5_gate7_recheck_20261030/v2_work"
EVIDENCE = ROOT / "data/frozen_reference/v2_p4/t5_gate7_delayed_20261030"
WORK = ROOT / "data/paper_shadow/t5_gate7_delayed_20261030"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_state(store: PaperStateStore, state: Any) -> bytes:
    raw = dumps_blob(state_to_blob(state))
    with store.write_lock():
        store._atomic_write_bytes(store.state_path / "engine_state.bin", raw)
        store.repair_projections()
    return raw


def _pending(state: Any) -> list[Dict[str, Any]]:
    manager = state.pending_order_manager
    orders = (
        manager.get_all_orders()
        if hasattr(manager, "get_all_orders")
        else list((manager.get("pending_orders") or {}).values())
    )
    records = [
        {
            "stock": order.stock,
            "action": order.action,
            "target_value": order.target_value,
            "signal_date": order.signal_date,
            "create_date": order.create_date,
            "retry_count": order.retry_count,
            "last_reason": order.last_reason,
        }
        for order in orders
    ]
    # v2 的停牌补位买落在 unfilled_slots；回滚时必须一并落入 legacy pending_buys。
    for signal_date, slot in (getattr(state, "unfilled_slots", {}) or {}).items():
        for item in slot.get("unfilled_slot_weights") or []:
            records.append(
                {
                    "stock": item["stock"],
                    "action": "buy",
                    "target_value": None,
                    "signal_date": signal_date,
                    "create_date": slot.get("first_attempt_date") or signal_date,
                    "retry_count": int(slot.get("attempts") or 0),
                    "last_reason": "停牌/延迟补位",
                    "target_weight": float(item.get("weight") or 0.0),
                }
            )
    return records


def _write_legacy_pending(legacy: Path, orders: list[Dict[str, Any]]) -> None:
    storage = PaperStorage(root_path=str(legacy))
    for action, directory in (("buy", storage.pending_buys_path), ("sell", storage.pending_sells_path)):
        rows = []
        for order in orders:
            if order["action"] != action:
                continue
            weight = (
                float(order.get("target_weight") or 0.0)
                if order.get("target_weight") is not None
                else (
                    float(order.get("target_value") or 0.0) / 100000.0
                    if action == "buy"
                    else 0.0
                )
            )
            rows.append(
                {
                    "ts_code": order["stock"],
                    "original_signal_date": str(pd.Timestamp(order["signal_date"]).strftime("%Y%m%d")),
                    "create_date": str(pd.Timestamp(order["create_date"]).strftime("%Y%m%d")),
                    "target_weight": weight,
                    "attempts": int(order["retry_count"] or 1),
                    "last_attempt_date": str(pd.Timestamp(order["create_date"]).strftime("%Y%m%d")),
                    "reason": order["last_reason"],
                }
            )
        (directory / ("pending_buys.json" if action == "buy" else "pending_sells.json")).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", default=str(EVIDENCE))
    parser.add_argument("--source", default=str(SOURCE))
    args = parser.parse_args(argv)
    evidence = Path(args.evidence_dir)
    source = Path(args.source)
    evidence.mkdir(parents=True, exist_ok=True)
    work = WORK
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    v2 = work / "v2_work"
    shutil.copytree(source, v2)

    store = PaperStateStore(root_path=str(v2))
    state = state_from_blob(loads_blob(store.load_engine_state_blob()))
    suspended = "000037.SZ"  # 20260803 真实涨停，买入会被延迟
    state.pending_signals[pd.Timestamp("20260731")] = {
        "signals": {suspended: 0.02},
        "ranked_candidates": [(suspended, 1.0)],
        "priority_candidates": [(suspended, 1.0)],
        "slot_weights": [{"stock": suspended, "weight": 0.02}],
        "target_n": 20,
        "desired_position_count": 20,
        "tranche_idx": 0,
    }
    _write_state(store, state)

    result = run_v2_day("20260803", paper_root=v2, data_root=str(ROOT / "data"))
    after_run = state_from_blob(loads_blob(store.load_engine_state_blob()))
    pending_after_run = _pending(after_run)
    delayed_present = any(
        order["stock"] == suspended and order["action"] == "buy"
        for order in pending_after_run
    )

    adjust_v2_account(
        store, command="cash", trade_date="20260803",
        values={"set": float(after_run.current_capital) + 1000.0},
        data_root=str(ROOT / "data"),
    )
    adjusted = state_from_blob(loads_blob(store.load_engine_state_blob()))

    legacy = work / "legacy_resume"
    shutil.copytree(v2, legacy)
    raw = (legacy / "state" / "engine_state.bin").read_bytes()
    projection = rollback_account(adjusted, "20260803")
    (legacy / "state" / "account.json").write_text(
        json.dumps(projection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flat = load_paper_config(legacy)
    flat["engine"] = "legacy"
    production_flat = PaperStorage(root_path=str(ROOT / "data" / "paper")).load_config() or {}
    for key, value in production_flat.items():
        if "exposure" in key.lower() or "policy" in key.lower() or "replenish" in key.lower():
            flat[key] = value
    PaperStorage(root_path=str(legacy)).save_config(dict(flat))
    manifest = {
        "validated": True,
        "engine_state_sha256": hashlib.sha256(raw).hexdigest(),
        "account_sha256": hashlib.sha256((legacy / "state" / "account.json").read_bytes()).hexdigest(),
        "rollback_date": "20260803",
    }
    (legacy / "state" / "rollback_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    assert_legacy_rollback_allowed(PaperStateStore(root_path=str(legacy)))
    _write_legacy_pending(legacy, pending_after_run)

    resumed = run_legacy_arm(
        ArmSpec("legacy_delayed", legacy, "legacy", False),
        data_root=ROOT / "data",
        day="20260804",
        evidence_dir=evidence / "gate7" / "delayed_resume",
    )
    legacy_account = json.loads((legacy / "state" / "account.json").read_text(encoding="utf-8"))
    payload = {
        "date": "20260803",
        "suspended_stock": suspended,
        "v2_t0_status": result.t0_status,
        "pending_after_v2": pending_after_run,
        "delayed_present": delayed_present,
        "adjusted_cash": projection.get("cash"),
        "rollback_projection": projection,
        "legacy_next_date": resumed.get("date"),
        "legacy_t0_status": resumed.get("t0_status"),
        "legacy_fills": resumed.get("fills"),
        "legacy_last_update": legacy_account.get("last_update"),
        "legacy_pending_files": {
            "buys": str(legacy / "pending_buys" / "buy_pending.json"),
            "sells": str(legacy / "pending_sells" / "sell_pending.json"),
        },
        "verdict": "PASS" if delayed_present else "FAIL",
        "captured_at": _now(),
    }
    log = evidence / "gate7" / "logs" / "gate7_20260803_trade_delayed_order.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary = evidence / "gate7" / "delayed_probe.json"
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "delayed_present": delayed_present,
                      "legacy": payload["legacy_t0_status"], "fills": len(resumed.get("fills") or [])},
                     ensure_ascii=False))
    return 0 if delayed_present else 2


if __name__ == "__main__":
    raise SystemExit(main())
