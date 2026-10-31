# -*- coding: utf-8 -*-
"""T5 门 7 演练：``synthetic`` 全合成连贯 rehearsal，``real`` 真实演练逐项取证。

``synthetic``：覆盖迁移→真实 PaperHost 成交与延迟订单→权威 adjust→回滚→legacy 次日
续跑防呆，以及并发第二写入者 fail-fast；结果仅为 ``SYNTHETIC_PASS``，不是正式门 7。

``real``：读取操作清单（每项含执行入口、输入根、日期、结果、原始输出、账户摘要），
逐项校验后才可能给出 ``verdict=PASS``，且 ``synthetic_only=false``。合成结果无法借此
升级；历史行情演练不计入真实日历天数；操作手册见
``docs/guide/t5_gate7_real_rehearsal_manual.md``。全部写类项必须在隔离/副本根执行。

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4.t5_migration import migrate_isolated, rollback_account  # noqa: E402
from scripts.v2_p4.t05b_spike_runner import (  # noqa: E402
    _COST_SETTINGS,
    _DeterministicSignal,
    _StaticUniverse,
)
from src.lazybull.data import DataLoader, Storage  # noqa: E402
from src.lazybull.paper import PaperStorage  # noqa: E402
from src.lazybull.paper.models import AccountState  # noqa: E402
from src.lazybull.v2.common.cost import CostModel  # noqa: E402
from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.hosts.paper.config_adapter import (  # noqa: E402
    to_engine_kwargs,
    to_trading_config,
)
from src.lazybull.v2.hosts.paper.entrypoints import assert_legacy_rollback_allowed  # noqa: E402
from src.lazybull.v2.hosts.paper.host import PaperHost  # noqa: E402
from src.lazybull.v2.hosts.paper.state_service import adjust_v2_account  # noqa: E402
from src.lazybull.v2.hosts.paper.state_store import (  # noqa: E402
    PaperLockError,
    PaperStateStore,
)
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.panel_builder import bootstrap_manifest  # noqa: E402

_CODES = ["A", "B", "C"]
_DAYS = pd.bdate_range("2026-10-29", periods=4).strftime("%Y%m%d").tolist()
_FLAT = {
    "model_version": 7,
    "top_n": 2,
    "rebalance_freq": 1,
    "stagger_tranches": 1,
    "exclude_st": True,
    "min_list_days": 365,
    "position_sizing": "equal",
    "buy_price": "close",
    "sell_price": "open",
    "initial_capital": 100000.0,
    "min_buy_value_ratio": 0.0,
    "horizon": 20,
    "universe": "all",
    "engine": "v2",
}


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _make_data(data_root: Path) -> PanelDataStore:
    panel = PanelDataStore(data_root)
    bootstrap_manifest(panel)
    storage = Storage(root_path=str(data_root))
    daily_dir = storage.clean_path / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    for day in _DAYS:
        panel.append_features(
            TradeDate.from_str(day),
            "core",
            pd.DataFrame(
                {
                    "trade_date": [day] * 3,
                    "ts_code": _CODES,
                    "ret_1": [0.1, 0.2, 0.3],
                    "ret_5": [0.2, 0.3, 0.4],
                    "bp": [0.01, 0.02, 0.03],
                    "amount_ma20": [60000.0] * 3,
                    "total_mv": [800000.0] * 3,
                    "sw_l1_code": ["801780"] * 3,
                }
            ),
        )
        frame = pd.DataFrame(
            {
                "trade_date": [day] * 3,
                "ts_code": _CODES,
                "close": [10.0, 20.0, 30.0],
                "close_adj": [10.0, 20.0, 30.0],
                "pre_close": [10.0, 20.0, 30.0],
                "open": [10.0, 20.0, 30.0],
                "open_adj": [10.0, 20.0, 30.0],
                "is_suspended": [0] * 3,
                "is_limit_up": [0, int(day == _DAYS[1]), 0],
                "is_limit_down": [0] * 3,
                "vol": [1000.0] * 3,
                "pct_chg": [0.0] * 3,
                "is_st": [0] * 3,
                "list_days": [1000] * 3,
                "tradable": [1] * 3,
            }
        )
        frame.to_parquet(daily_dir / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False)
    return panel


def _make_legacy_source(source: Path) -> None:
    storage = PaperStorage(root_path=str(source))
    storage.save_config({**_FLAT, "engine": "legacy"})
    storage.save_account_state(AccountState(cash=100000.0, positions={}, last_update=_DAYS[0]))
    storage.save_last_trade_date(_DAYS[0])
    storage.save_rebalance_state(
        {
            "last_rebalance_date": _DAYS[0],
            "last_scheduled_rebalance_date": _DAYS[0],
            "tranche_anchor_date": _DAYS[0],
            "rebalance_freq": 1,
            "stagger_tranches": 1,
        }
    )
    pd.DataFrame(
        {
            "trade_date": [_DAYS[0]],
            "cash": [100000.0],
            "position_value": [0.0],
            "total_value": [100000.0],
            "nav": [1.0],
        }
    ).to_parquet(storage.nav_path / "nav.parquet", index=False)
    pd.DataFrame().to_parquet(storage.trades_path / "trades.parquet", index=False)
    (storage.pending_buys_path / "B.json").write_text(
        json.dumps(
            {
                "ts_code": "B",
                "target_weight": 0.1,
                "reason": "涨停",
                "create_date": _DAYS[0],
                "attempts": 1,
                "last_attempt_date": _DAYS[0],
                "original_signal_date": _DAYS[0],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "action": ["buy"],
            "ts_code": ["A"],
            "shares": [4000],
            "target_weight": [0.4],
            "buy_price_type": ["close"],
            "sell_price_type": ["open"],
            "reason": ["调仓买入"],
        }
    ).to_parquet(storage.instructions_path / f"{_DAYS[1]}.parquet", index=False)
    _dump(storage.runs_path / f"t1_{_DAYS[0]}.json", {"status": "ok"})


def _make_host(target: Path, flat: Dict[str, Any], data_root: Path) -> PaperHost:
    storage = Storage(root_path=str(data_root))
    config = to_trading_config(flat)
    plan = {day: [("A", 0.9), ("B", 0.8)] for day in _DAYS}
    return PaperHost(
        store=PaperStateStore(root_path=str(target)),
        trading_config=config,
        universe=_StaticUniverse(_CODES),
        signal=_DeterministicSignal(2, plan, plan[_DAYS[0]]),
        cost_model=CostModel(**_COST_SETTINGS),
        stock_basic=None,
        data_storage=storage,
        panel_store=PanelDataStore(data_root),
        trading_calendar=[pd.Timestamp(day) for day in _DAYS],
        engine_kwargs=to_engine_kwargs(flat, config),
        feature_columns=["ret_1", "ret_5", "bp"],
        train_params={},
        data_loader=DataLoader(storage=storage, verbose=False),
        schedule_mode="live",
        verbose=False,
    )


def run_rehearsal(work_root: Path, evidence: Path) -> Dict[str, Any]:
    if work_root.exists() and any(work_root.iterdir()):
        raise FileExistsError(f"rehearsal 工作根必须为空: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    data_root = work_root / "data"
    _make_data(data_root)
    source = work_root / "legacy_source"
    _make_legacy_source(source)
    target = work_root / "migrated_v2"
    migration = migrate_isolated(
        source=source,
        target=target,
        data_root=data_root,
        evidence=evidence,
        execute=True,
        host_builder=_make_host,
    )
    host = _make_host(target, {**_FLAT, "engine": "v2"}, data_root)
    host.run_t1(_DAYS[1])
    result = host.last_day_result
    if result is None or not result.fills:
        raise RuntimeError("rehearsal 未产生真实 PaperHost 成交")
    raw = host.store.load_engine_state_blob()
    if raw is None or b"pending_orders" not in raw:
        raise RuntimeError("rehearsal 快照缺延迟订单管理器")
    nav_before = host.store.load_all_nav()
    adjusted = adjust_v2_account(
        host.store,
        command="cash",
        trade_date=_DAYS[2],
        values={"set": 90000.0},
        data_root=str(data_root),
        host=host,
    )
    nav_after = host.store.load_all_nav()
    if len(nav_before) != len(nav_after):
        raise RuntimeError("adjust 不得追加无成交日 NAV 行")
    with host.store.write_lock():
        second = PaperStateStore(root_path=str(target), lock_timeout_seconds=0.05)
        try:
            with second.write_lock():
                raise RuntimeError("第二写入者未被拒绝")
        except PaperLockError:
            second_writer_fail_fast = True
    # 回滚日使用 adjust 后最新权威快照，而不是成交前 raw。
    latest_raw = host.store.load_engine_state_blob()
    from src.lazybull.v2.core.accounting.engine_state import loads_blob, state_from_blob

    rollback = rollback_account(state_from_blob(loads_blob(latest_raw)), _DAYS[2])
    host.store._atomic_write_json(host.store.state_path / "account.json", rollback)
    manifest = {
        "validated": True,
        "rollback_date": _DAYS[2],
        "engine_state_sha256": hashlib.sha256(latest_raw).hexdigest(),
        "account_sha256": hashlib.sha256(
            (host.store.state_path / "account.json").read_bytes()
        ).hexdigest(),
    }
    _dump(host.store.state_path / "rollback_manifest.json", manifest)
    assert_legacy_rollback_allowed(host.store)
    rollback["cash"] -= 100.0
    rollback["last_update"] = _DAYS[3]
    host.store._atomic_write_json(host.store.state_path / "account.json", rollback)
    _dump(host.store.runs_path / f"t1_{_DAYS[3]}.json", {"status": "legacy_rehearsal"})
    assert_legacy_rollback_allowed(host.store)
    output = {
        "verdict": "SYNTHETIC_PASS",
        "synthetic_only": True,
        "formal_gate7_pass": False,
        "migration_verdict": migration["verdict"],
        "fills_count": len(result.fills),
        "delayed_order_present": True,
        "adjust_cash": adjusted["cash"],
        "nav_gap_semantics_preserved": True,
        "second_writer_fail_fast": second_writer_fail_fast,
        "rollback_same_day_guard": True,
        "rollback_next_day_guard": True,
        "production_touched": False,
    }
    _dump(evidence / "gate7" / "rehearsal_result.json", output)
    return output


#: 真实门 7 必采项（顺序即执行建议；缺任一项即未完成）
GATE7_ITEMS: tuple[str, ...] = (
    "run",
    "positions",
    "real",
    "model_info",
    "config",
    "adjust",
    "bot_commands",
    "migration_copy",
    "trade_delayed_order",
    "adjust_rollback_resume",
    "concurrency_mutex",
    "registry_old_model",
    "device_smb_read",
)

#: 这些项涉及状态写入，input_root 不得直接是生产纸面根（必须用副本/隔离根）
NO_PRODUCTION_WRITE_ITEMS = frozenset(
    {"migration_copy", "trade_delayed_order", "adjust_rollback_resume", "adjust", "run"}
)

REQUIRED_ITEM_FIELDS = (
    "item",
    "executed_at",
    "entry",
    "input_root",
    "trade_date",
    "result",
    "raw_output",
    "account_summary",
)


def _resolve(raw: str, base: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path)


def _checklist_item(
    raw: Mapping[str, Any], base: Path, production_paper_root: Path
) -> Dict[str, Any]:
    """单项校验：字段齐全、原始输出存在、写类项不得直接落在生产纸面根。"""
    missing_fields = [key for key in REQUIRED_ITEM_FIELDS if not raw.get(key)]
    problems: List[str] = []
    if missing_fields:
        problems.append(f"缺字段={missing_fields}")
    input_root = None
    raw_output = None
    if raw.get("input_root"):
        input_root = _resolve(str(raw["input_root"]), base)
        if not input_root.exists():
            problems.append(f"输入根不存在={input_root}")
    if raw.get("raw_output"):
        raw_output = _resolve(str(raw["raw_output"]), base)
        if not raw_output.is_file():
            problems.append(f"原始输出不存在={raw_output}")
    production_write_suspected = False
    if input_root is not None and str(raw.get("item")) in NO_PRODUCTION_WRITE_ITEMS:
        try:
            inside = input_root.resolve() == production_paper_root.resolve() or (
                production_paper_root.resolve() in input_root.resolve().parents
            )
        except OSError:
            inside = False
        if inside:
            production_write_suspected = True
            problems.append("写类项不得直接以生产纸面根为输入（须先复制再演练）")
    result = str(raw.get("result") or "").upper()
    waiver = raw.get("waiver") if isinstance(raw.get("waiver"), Mapping) else {}
    if result == "WAIVED":
        if str(raw.get("item")) != "device_smb_read":
            problems.append("仅 device_smb_read 允许用户显式豁免")
        if waiver.get("approved_by_user") is not True or not waiver.get("approved_at"):
            problems.append("WAIVED 项缺用户批准证据")
    elif result and result not in {"PASS", "FAIL"}:
        problems.append(f"result 非法={raw.get('result')!r}")
    return {
        "item": str(raw.get("item")),
        "executed_at": raw.get("executed_at"),
        "entry": raw.get("entry"),
        "input_root": str(input_root) if input_root else None,
        "trade_date": raw.get("trade_date"),
        "result": result or "UNKNOWN",
        "raw_output": str(raw_output) if raw_output else None,
        "account_summary": raw.get("account_summary"),
        "problems": problems,
        "production_write_suspected": production_write_suspected,
    }


def run_real(
    *,
    checklist: Path,
    evidence: Path,
    production_paper_root: Path = ROOT / "data" / "paper",
) -> Dict[str, Any]:
    """真实门 7 取证：读操作清单并逐项校验，合成结果无法借此升级。"""
    checklist = Path(checklist)
    payload = json.loads(checklist.read_text(encoding="utf-8"))
    raw_items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(raw_items, list):
        raise ValueError(f"门 7 操作清单必须含 items list: {checklist}")
    base = checklist.parent
    collected: Dict[str, Dict[str, Any]] = {}
    extra: List[str] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise ValueError(f"门 7 清单项必须是对象: {raw!r}")
        item = _checklist_item(raw, base, Path(production_paper_root))
        key = item["item"]
        if key not in GATE7_ITEMS:
            extra.append(key)
            collected.setdefault(key, item)
            continue
        if key in collected:
            raise ValueError(f"门 7 清单项重复: {key}")
        collected[key] = item
    incomplete: List[str] = []
    waived: List[str] = []
    for key in GATE7_ITEMS:
        item = collected.get(key)
        if item is None:
            incomplete.append(f"{key}: 未采集")
        elif item["problems"]:
            incomplete.append(f"{key}: {'; '.join(item['problems'])}")
        elif item["result"] == "WAIVED":
            waived.append(key)
        elif item["result"] != "PASS":
            incomplete.append(f"{key}: result={item['result']}")
    violations = [key for key, item in collected.items() if item["production_write_suspected"]]
    verdict = "PASS" if not incomplete and not violations else "INCOMPLETE"
    result = {
        "verdict": verdict,
        "synthetic_only": False,
        "formal_gate7_pass": verdict == "PASS",
        "items": collected,
        "items_incomplete": incomplete,
        "items_waived": waived,
        "items_extra": extra,
        "production_write_violations": violations,
        "production_touched": False,
        "checklist": str(checklist),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "note": "真实演练逐项取证；历史行情演练不累计真实日历天数，未覆盖项继续未完成。",
    }
    _dump(Path(evidence) / "gate7" / "real_rehearsal_result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--work-root")
    parser.add_argument("--checklist", help="真实门 7 操作清单 JSON（--mode real 必填）")
    parser.add_argument("--paper-root", default=str(ROOT / "data" / "paper"))
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args(argv)
    evidence = Path(args.evidence_dir)
    if args.mode == "real":
        if not args.checklist:
            raise ValueError("--mode real 必须提供 --checklist（逐项操作记录）")
        result = run_real(
            checklist=Path(args.checklist),
            evidence=evidence,
            production_paper_root=Path(args.paper_root),
        )
    else:
        if not args.work_root:
            raise ValueError("--mode synthetic 必须提供 --work-root（隔离空目录）")
        result = run_rehearsal(Path(args.work_root), evidence)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("verdict") == "PASS" or result.get("verdict") == "SYNTHETIC_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
