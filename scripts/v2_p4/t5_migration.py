# -*- coding: utf-8 -*-
"""v2 P4 T5 账户迁移、调整与回滚投影（默认 dry-run，阶段专用）。

首代迁移只写全新隔离根，不伪造 t0/t1 成交运行。源账户、锚、pending、候选、
止损、历史成交与 NAV 逐项映射并校验；死文件不迁，exposure/模型预热只留回滚。
"""

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4.t5_gate1_identity import assess_holdings  # noqa: E402
from src.lazybull.paper import PaperStorage  # noqa: E402
from src.lazybull.v2.core.accounting.engine_state import (  # noqa: E402
    EngineState,
    dumps_blob,
    export_state,
    state_from_blob,
    state_to_blob,
)
from src.lazybull.v2.core.accounting.ledger import make_fill_key  # noqa: E402
from src.lazybull.v2.core.execution.live_schedule import LiveAnchorState  # noqa: E402
from src.lazybull.v2.hosts.paper.adjustment_rules import apply_adjustment  # noqa: E402
from src.lazybull.v2.hosts.paper.config_adapter import load_paper_config  # noqa: E402
from src.lazybull.v2.hosts.paper.factory import create_paper_host  # noqa: E402
from src.lazybull.v2.hosts.paper.state_store import (  # noqa: E402
    GENERATION_INFO_KEY,
    build_account_projection,
)

STATIC_INVENTORY: Tuple[Tuple[str, str, str, str], ...] = (
    ("state/account.json", "json", "migrate", "现金/股数/买入日/买价/费用逐项"),
    ("state/last_trade_date.json", "json", "migrate", "与账户 last_update 一致性"),
    ("state/ranked_candidates.json", "json", "migrate", "signal_date + 有序候选"),
    ("state/strategy_state.json", "json", "migrate", "last_rebalance_nav"),
    ("state/stop_loss_state.json", "json", "migrate", "连续跌停计数字典"),
    ("runs/rebalance_state.json", "json", "migrate", "LiveAnchorState 五字段"),
    ("pending_buys/*.json", "json", "migrate", "信号日/创建日/权重/尝试次数"),
    ("pending_sells/*.json", "json", "migrate", "信号日/创建日/尝试次数"),
    ("instructions/*.parquet", "parquet", "migrate", "执行日/信号日/方向/股票/权重"),
    ("trades/trades.parquet", "parquet", "migrate", "逐笔字段 + ledger 幂等键"),
    ("nav/nav.parquet", "parquet", "migrate", "原行保留；禁止补无成交日"),
    ("state/exposure_policy.json", "json", "rollback_only", "不迁 v2；回滚原值校验"),
    ("terminal_risk_model_root", "directory", "rollback_only", "路径/内容可用性"),
    ("paper_warmup/state.json", "json", "rollback_only", "路径/内容可用性"),
    ("account.json", "json", "dead", "死文件显式不迁移"),
    ("state/weakness_exit_state.json", "json", "dead", "死文件显式不迁移"),
)


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _last_completed_run_day(storage: PaperStorage, fallback: str) -> str:
    """以真实 t0/t1 运行记录判定最后完整处理日；无记录才退回账户末日。"""
    days: List[str] = []
    for pattern in ("t0_*.json", "t1_*.json"):
        for path in storage.runs_path.glob(pattern):
            day = path.stem.split("_", 1)[-1]
            if len(day) == 8 and day.isdigit():
                days.append(day)
    return max(days) if days else fallback


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_file_inventory(paper_root: Path) -> List[Dict[str, Any]]:
    """「文件 × 载体 × 校验」二分清单，动态 pending/instructions 单独展开。"""
    rows: List[Dict[str, Any]] = []
    for pattern, carrier, action, validation in STATIC_INVENTORY:
        matches = sorted(paper_root.glob(pattern)) if "*" in pattern else [paper_root / pattern]
        if not matches:
            matches = [paper_root / pattern]
        for path in matches:
            rows.append(
                {
                    "path": str(path.relative_to(paper_root)).replace("\\", "/"),
                    "carrier": carrier,
                    "action": action,
                    "validation": validation,
                    "exists": path.exists(),
                    "sha256": _sha256(path) if path.is_file() else None,
                }
            )
    return rows


def map_account(account: Any) -> Dict[str, Dict[str, Any]]:
    """旧 Position → v2 持仓；buy_cost 是费用额，不是本金+费用。"""
    positions: Dict[str, Dict[str, Any]] = {}
    for code, position in sorted(account.positions.items()):
        shares = int(position.shares)
        buy_price = float(position.buy_price)
        positions[code] = {
            "shares": shares,
            "buy_date": pd.Timestamp(position.buy_date),
            "signal_date": pd.Timestamp(position.buy_date),
            "buy_trade_price": buy_price,
            "buy_pnl_price": float(position.buy_pnl_price),
            "buy_cost_cash": shares * buy_price + float(position.buy_cost),
            "last_known_price": None,
        }
    return positions


def map_anchor(
    raw: Mapping[str, Any], *, default_freq: int, default_tranches: int
) -> LiveAnchorState:
    """旧版两字段与新版五字段均支持；freq/K 只从配置缺省补齐。"""

    def timestamp(key: str) -> Optional[pd.Timestamp]:
        value = raw.get(key)
        return None if value in (None, "") else pd.Timestamp(str(value))

    return LiveAnchorState(
        anchor_date=timestamp("tranche_anchor_date"),
        rebalance_freq=int(raw.get("rebalance_freq") or default_freq),
        stagger_tranches=int(raw.get("stagger_tranches") or default_tranches),
        last_scheduled_rebalance_date=timestamp("last_scheduled_rebalance_date"),
        last_rebalance_date=timestamp("last_rebalance_date"),
    )


def map_pending_orders(storage: PaperStorage, total_value: float) -> Dict[str, Any]:
    """旧 pending → v2 PendingOrderManager 投影；权重按迁移日组合总值换预算。"""
    pending: Dict[Tuple[str, str], Any] = {}
    from src.lazybull.v2.core.execution.pending_order import PendingOrder

    for directory, action in (
        (storage.pending_buys_path, "buy"),
        (storage.pending_sells_path, "sell"),
    ):
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload if isinstance(payload, list) else [payload]
            for raw in items:
                if not isinstance(raw, Mapping):
                    raise RuntimeError(f"pending 条目非对象: {path}")
                signal = str(raw.get("original_signal_date") or raw.get("create_date"))
                code = str(raw["ts_code"])
                pending[(code, action)] = PendingOrder(
                    stock=code,
                    action=action,
                    target_value=(
                        float(raw.get("target_weight") or 0.0) * total_value
                        if action == "buy"
                        else None
                    ),
                    signal_date=pd.Timestamp(signal),
                    create_date=pd.Timestamp(str(raw["create_date"])),
                    retry_count=int(raw.get("attempts") or 0),
                    last_reason=raw.get("reason"),
                )
    return {
        "pending_orders": pending,
        "total_added": len(pending),
        "total_expired": 0,
        "total_succeeded": 0,
    }


def map_trades_and_ledger(frame: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """旧成交投影为引擎记录，并按全流程序列重建稳定幂等键。"""
    records: List[Dict[str, Any]] = []
    keys: List[str] = []
    last_date: Optional[pd.Timestamp] = None
    for index, raw in enumerate(frame.to_dict("records")):
        day = pd.Timestamp(str(raw.get("trade_date") or raw.get("date")))
        code = str(raw.get("ts_code") or raw.get("stock"))
        action = str(raw["action"])
        record = dict(raw)
        record.update(
            {
                "date": day,
                "stock": code,
                "signal_date": pd.Timestamp(str(raw.get("signal_date") or day)),
                "cost": float(raw.get("cost", raw.get("total_cost", 0.0)) or 0.0),
            }
        )
        records.append(record)
        keys.append(make_fill_key(day, action, code, index))
        last_date = day
    return records, {"applied_keys": keys, "last_fill_date": last_date}


def migrate_state(
    state: EngineState,
    *,
    storage: PaperStorage,
    trading_config: Any,
    calendar: Sequence[str],
    mark_prices: Mapping[str, float],
) -> EngineState:
    """把完整旧状态映射到基线 EngineState；缺任何必需语义即 fail-fast。"""
    account = storage.load_account_state()
    if account is None or not account.last_update:
        raise RuntimeError("迁移源账户缺失或 last_update 为空")
    state.current_capital = float(account.cash)
    state.positions = map_account(account)
    anchor = map_anchor(
        storage.load_rebalance_state() or {},
        default_freq=int(trading_config.rebalance_freq),
        default_tranches=int(trading_config.stagger_tranches),
    )
    state.live_anchor_state = anchor.to_dict()
    market_value = sum(
        int(position["shares"]) * float(mark_prices.get(code, position["buy_trade_price"]))
        for code, position in state.positions.items()
    )
    total_value = state.current_capital + market_value
    state.pending_order_manager = map_pending_orders(storage, total_value)
    nav = storage.load_all_nav()
    portfolio_values: List[Dict[str, Any]] = []
    if nav is not None and not nav.empty:
        for raw in nav.to_dict("records"):
            portfolio_values.append(
                {
                    "date": pd.Timestamp(str(raw.get("trade_date") or raw.get("date"))),
                    "portfolio_value": float(raw.get("total_value")),
                    "capital": float(raw.get("cash")),
                    "market_value": float(raw.get("position_value", raw.get("market_value", 0.0))),
                }
            )
    # account.json 是当前权威账户快照：同日 NAV 行可能停留在当日成交前，
    # 迁移只能用账户重算值覆盖该行；历史 NAV 行与跳空保持原语义。
    authoritative_day = pd.Timestamp(account.last_update).strftime("%Y%m%d")
    replaced = False
    for item in portfolio_values:
        if pd.Timestamp(item["date"]).strftime("%Y%m%d") == authoritative_day:
            item.update(
                {
                    "portfolio_value": total_value,
                    "capital": state.current_capital,
                    "market_value": market_value,
                }
            )
            replaced = True
    if not replaced:
        portfolio_values.append(
            {
                "date": pd.Timestamp(account.last_update),
                "portfolio_value": total_value,
                "capital": state.current_capital,
                "market_value": market_value,
            }
        )
    state.portfolio_values = sorted(portfolio_values, key=lambda item: pd.Timestamp(item["date"]))
    trades = storage.load_all_trades()
    state.trades, state.ledger = map_trades_and_ledger(
        trades if trades is not None else pd.DataFrame()
    )
    candidates = storage.load_ranked_candidates()
    if isinstance(candidates, tuple) and len(candidates) == 2:
        state._last_ranked_candidates = list(candidates[0])
        state._last_signal_date = pd.Timestamp(candidates[1])
    stop_loss = storage.load_stop_loss_state() or {}
    state.stop_loss_monitor = (
        {"consecutive_limit_down_days": dict(stop_loss.get("consecutive_limit_down_days") or {})}
        if bool(trading_config.stop_loss_enabled)
        else None
    )
    strategy = storage.load_strategy_state() or {}
    state._last_rebalance_nav = strategy.get("last_rebalance_nav", total_value)
    state.pending_signals = {}
    state.pending_condition_sells = {}
    state.pending_stop_loss_sells = {}
    last_completed_day = _last_completed_run_day(storage, account.last_update)
    pending_manager = state.pending_order_manager
    pending_orders = (
        list((pending_manager.get("pending_orders") or {}).values())
        if isinstance(pending_manager, Mapping)
        else list(pending_manager.get_all_orders())
    )
    pending_sell_codes = {
        order.stock for order in pending_orders if getattr(order, "action", None) == "sell"
    }
    for path in sorted(storage.instructions_path.glob("*.parquet")):
        execution_day = path.stem
        previous = [day for day in calendar if day < execution_day]
        if not previous:
            raise RuntimeError(f"指令 {path} 无法解析信号日")
        signal_day = previous[-1]
        frame = pd.read_parquet(path)
        for column, default in (
            ("price_type", None),
            ("reason", ""),
            ("source_date", signal_day),
            ("target_weight", 0.0),
            ("original_signal_date", signal_day),
            ("desired_position_count", 0),
            ("retry_attempt", 0),
            ("replacement_slot_code", ""),
            ("keep_buy_date", False),
        ):
            if column not in frame:
                if column == "price_type":
                    frame[column] = frame["action"].map(
                        {"buy": "close", "sell": "open"}
                    ).fillna(default)
                else:
                    frame[column] = default
        buys = frame[frame["action"] == "buy"]
        sells = frame[frame["action"] == "sell"]
        if not buys.empty:
            signals = {
                str(raw["ts_code"]): float(raw.get("target_weight") or 0.0)
                for raw in buys.to_dict("records")
            }
            desired = int(
                buys.iloc[0].get("desired_position_count")
                or len(buys) + len(state.positions)
            )
            instruction_shares = {
                str(raw["ts_code"]): int(raw["shares"])
                for raw in buys.to_dict("records")
                if raw.get("shares") is not None
            }
            priority = list(state._last_ranked_candidates) or list(signals.items())
            state.pending_signals[pd.Timestamp(signal_day)] = {
                "signals": signals,
                "ranked_candidates": list(state._last_ranked_candidates),
                "priority_candidates": priority,
                "slot_weights": [
                    {"stock": code, "weight": weight} for code, weight in signals.items()
                ],
                "instruction_shares": instruction_shares,
                "instruction_rows": json.loads(
                    frame.to_json(orient="records", date_format="iso")
                ),
                "target_n": desired,
                "desired_position_count": desired,
                "tranche_idx": int(buys.iloc[0].get("tranche_idx") or 0),
            }
        elif not sells.empty:
            # 纯卖指令也保存完整行集，恢复时不能按当前配置重算 sell price_type。
            state.pending_signals[pd.Timestamp(signal_day)] = {
                "signals": {},
                "ranked_candidates": list(state._last_ranked_candidates),
                "priority_candidates": [],
                "slot_weights": [],
                "instruction_rows": json.loads(
                    frame.to_json(orient="records", date_format="iso")
                ),
                "target_n": 0,
                "desired_position_count": len(state.positions),
                "tranche_idx": 0,
            }
        # 仅未来未执行指令中的卖单进入 condition sells；已消费历史卖单不重激活，
        # 已失败 pending sell 由 PendingOrderManager 承载，避免同一卖单双写。
        if execution_day > last_completed_day and not sells.empty:
            for raw in sells.to_dict("records"):
                code = str(raw["ts_code"])
                if code in pending_sell_codes:
                    continue
                state.pending_condition_sells[code] = {
                    "trigger_date": pd.Timestamp(signal_day),
                    "sell_type": (
                        "holding_period"
                        if "到期" in str(raw.get("reason"))
                        else "rebalance"
                    ),
                }
    blob = state_to_blob(state)
    blob[GENERATION_INFO_KEY] = {"trade_date": account.last_update, "run_kind": "t1"}
    # 完整 schema/子对象键集往返，禁止以部分字典冒充 EngineState。
    return state_from_blob(blob)


def rollback_account(state: EngineState, trade_date: str) -> Dict[str, Any]:
    """v2 → legacy 账户投影；NAV/成交由独立追加清单处理，不在此补跳空。"""
    return build_account_projection(state.current_capital, state.positions, trade_date)


def _frozen_valuation(
    *,
    assessment: Mapping[str, Any],
    valuation: Optional[Mapping[str, Any]],
    account: Mapping[str, Any],
) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    """确定迁移估值：显式冻结输入优先，否则用最新截面（须显式登记来源）。"""
    if valuation is None:
        marks = {
            str(row["ts_code"]): float(row["mark_price"])
            for row in assessment["positions"]
            if row.get("mark_price") is not None
        }
        return (
            marks,
            float(assessment["total_value"]),
            {
                "source": "latest_clean_section",
                "market_date": assessment["market_date"],
            },
        )
    market_date = str(valuation.get("market_date") or "")
    if market_date != str(assessment["market_date"]):
        raise RuntimeError(
            f"迁移估值日期与冻结时点不一致: 冻结={market_date} 最新截面={assessment['market_date']}；"
            "禁止用新价格回填旧日状态"
        )
    marks = {
        str(code): float(price)
        for code, price in (valuation.get("mark_prices") or {}).items()
        if price is not None
    }
    missing = [code for code in (account.get("positions") or {}) if code not in marks]
    if missing:
        raise RuntimeError(f"冻结估值缺持仓现价，拒绝迁移: {missing}")
    try:
        total = float(valuation["total_value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"冻结估值缺 total_value，拒绝迁移: {exc}") from exc
    if abs(total - float(assessment["total_value"])) > 1e-6:
        raise RuntimeError(
            f"冻结估值总值与 D10 复算不一致: 冻结={total} 复算={assessment['total_value']}"
        )
    return (
        marks,
        total,
        {
            "source": "frozen_gate1_holdings",
            "market_date": market_date,
            "content_fingerprint": valuation.get("content_fingerprint"),
        },
    )


def _account_projection_matches(source: Mapping[str, Any], projection: Mapping[str, Any]) -> bool:
    """迁移后账户投影等价（现金 + 股数/买价/费用，1e-6 容差）。"""
    if abs(float(source.get("cash") or 0.0) - float(projection.get("cash") or 0.0)) > 1e-6:
        return False
    source_positions = source.get("positions") or {}
    target_positions = projection.get("positions") or {}
    if set(source_positions) != set(target_positions):
        return False
    for code, raw in source_positions.items():
        migrated = target_positions[code]
        for source_key, target_key in (
            ("shares", "shares"),
            ("buy_price", "buy_price"),
            ("buy_cost", "buy_cost"),
        ):
            if (
                source_key in raw
                and abs(float(raw[source_key]) - float(migrated.get(target_key) or 0.0)) > 1e-6
            ):
                return False
    return True


def migrate_isolated(
    *,
    source: Path,
    target: Path,
    data_root: Path,
    evidence: Path,
    execute: bool,
    host_builder: Optional[Callable[..., Any]] = None,
    valuation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """默认 dry-run；execute 也只允许全新隔离 target，禁止直接写生产根。

    ``valuation`` 给出时（``{market_date, mark_prices, total_value, content_fingerprint}``）
    迁移估值绑定冻结时点；缺省沿用最新 clean 截面并在证据中登记来源。
    """
    if target == data_root / "paper" or data_root / "paper" in target.parents:
        raise RuntimeError("迁移工具只允许全新隔离 target；生产迁移须另行走授权切换程序")
    inventory = build_file_inventory(source)
    if not execute:
        return {"verdict": "DRY_RUN", "inventory": inventory}
    if target.exists():
        raise FileExistsError(f"迁移 target 已存在，禁止覆盖: {target}")
    shutil.copytree(source, target)
    flat = load_paper_config(target)
    flat["engine"] = "v2"
    flat["exposure_policy"] = None
    # 迁移 API 只在内存改 flat config；这里把目标侧生效值真正持久化并回读校验。
    storage = PaperStorage(root_path=str(target))
    storage.save_config(dict(flat))
    persisted = load_paper_config(target)
    if persisted.get("engine") != "v2" or persisted.get("exposure_policy") not in (None, ""):
        raise RuntimeError(
            f"迁移后目标配置未生效: engine={persisted.get('engine')!r} "
            f"exposure_policy={persisted.get('exposure_policy')!r}"
        )
    if host_builder is None:
        host = create_paper_host(store=target, config_flat=persisted, data_root=str(data_root))
    else:
        host = host_builder(target, persisted, data_root)
    engine = host._new_engine({})
    calendar = [day.strftime("%Y%m%d") for day in host.trading_calendar]
    source_account = json.loads((source / "state" / "account.json").read_text(encoding="utf-8"))
    assessment = assess_holdings(source_account, data_root)
    if assessment["valid_count"] != assessment["position_count"]:
        raise RuntimeError(f"迁移日 D10 持仓现价验证未过: {assessment}")
    marks, frozen_total, valuation_meta = _frozen_valuation(
        assessment=assessment, valuation=valuation, account=source_account
    )
    state = migrate_state(
        export_state(engine),
        storage=storage,
        trading_config=host.trading_config,
        calendar=calendar,
        mark_prices=marks,
    )
    migrated_total = float(state.portfolio_values[-1]["portfolio_value"])
    if abs(migrated_total - frozen_total) > 1e-6:
        raise RuntimeError(
            "迁移组合总值与 D10 现价验证不一致: "
            f"migrated={migrated_total} frozen={frozen_total} meta={valuation_meta}"
        )
    blob = state_to_blob(state)
    blob[GENERATION_INFO_KEY] = {
        "trade_date": storage.load_account_state().last_update,
        "run_kind": "t1",
    }
    raw = dumps_blob(blob)
    with host.store.write_lock():
        host.store._atomic_write_bytes(host.store.state_path / "engine_state.bin", raw)
        host.store.repair_projections()
    projection = rollback_account(state, storage.load_account_state().last_update)
    if not _account_projection_matches(source_account, projection):
        raise RuntimeError("迁移后账户投影与冻结源不等价（现金/股数/买价/费用）")
    result = {
        "verdict": "PASS",
        "target": str(target),
        "engine_state_sha256": hashlib.sha256(raw).hexdigest(),
        "inventory": inventory,
        "account_projection": projection,
        "account_projection_equivalent": True,
        "valuation": {**valuation_meta, "total_value": frozen_total},
        "pending_orders": len(state.pending_order_manager.get("pending_orders") or {}),
        "ledger_keys": len((state.ledger or {}).get("applied_keys") or []),
        "generation": blob.get(GENERATION_INFO_KEY),
        "config": {
            "engine": persisted.get("engine"),
            "exposure_policy": persisted.get("exposure_policy"),
        },
        "nav_gap_semantics": "nav.parquet 原行保留；portfolio_values 仅用于权威恢复，不补 NAV 投影",
    }
    _dump(evidence / "migration_result.json", result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = migrate_isolated(
        source=Path(args.source),
        target=Path(args.target),
        data_root=Path(args.data_root),
        evidence=Path(args.evidence_dir),
        execute=args.execute,
    )
    print(
        json.dumps(
            {"verdict": result["verdict"], "target": result.get("target")}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
