# -*- coding: utf-8 -*-
"""P6 T1 合成夹具：构造 ``data/paper`` 形态的隔离根（**不引用任何真实生产数据**）。

夹具形态与生产同构（评审 R1-01/R1-02/R1-10 整改）：

- **NAV 跳空语义**：``nav.parquet`` 只在**有成交日**追加行（``state_store``/``nav_valuation``
  的生产口径），默认根因此只有当前日一行（当前日有成交）；需要多日历史曲线的用例显式传
  ``nav=nav_rows()``（dense）或 ``nav_rows(only_days=...)``；
- **权威成交形态**：``EngineState.trades`` 存**引擎记录**（``date/stock/action/price/shares/
  amount/cost``），``trades.parquet`` 存 11 列投影；``build_paper_root`` 默认把传入的行转成引擎
  形态再入权威 blob（``authority_trades_shape="engine"``），投影侧由 ``trade_row()`` 提供。

所有日期与数值均为合成常量。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from src.lazybull.v2.core.accounting.engine_state import EngineState, dumps_blob, state_to_blob
from src.lazybull.v2.core.accounting.ledger import Ledger
from src.lazybull.v2.hosts.paper.instruction_projection import INSTRUCTION_COLUMNS
from src.lazybull.v2.hosts.paper.state_store import (
    NAV_COLUMNS,
    TRADES_COLUMNS,
    build_account_projection,
    build_rebalance_state_projection,
)

#: 合成交易日（与真实生产日期无关）
TRADE_DATE = "20260603"
PREV_DATE = "20260602"
FUTURE_DATE = "20260604"
EARLIER_DATE = "20260601"
#: 权威业务日序列中交易日的前一日（用于“当前日缺 NAV 行”负例）
LAST_NAV = EARLIER_DATE
#: 合成初始资金与总资产
INITIAL_CAPITAL = 500000.0
CASH = 18526.180266199968
MARKET_VALUE = 600490.0
TOTAL_VALUE = CASH + MARKET_VALUE

POSITIONS: dict[str, dict[str, Any]] = {
    "600000.SH": {
        "shares": 10000,
        "buy_date": "20260506",
        "signal_date": "20260505",
        "buy_trade_price": 10.0,
        "buy_pnl_price": 10.0,
        "buy_cost_cash": 100050.0,
    },
    "000001.SZ": {
        "shares": 20000,
        "buy_date": "20260519",
        "signal_date": "20260518",
        "buy_trade_price": 12.0,
        "buy_pnl_price": 12.1,
        "buy_cost_cash": 240060.0,
    },
}

#: 六个业务日（合成序列；末日 = TRADE_DATE）
BUSINESS_DAYS = ("20260506", "20260519", "20260522", "20260527", "20260601", TRADE_DATE)


def portfolio_values(
    *, values: Mapping[str, tuple[float, float]] | None = None
) -> list[dict[str, Any]]:
    """权威 ``portfolio_values`` 逐业务日行（``portfolio_value`` / ``capital`` / ``market_value``）。"""
    rows: list[dict[str, Any]] = []
    for index, day in enumerate(BUSINESS_DAYS):
        capital = CASH if day == TRADE_DATE else 20000.0 + index
        market = TOTAL_VALUE - capital if day == TRADE_DATE else 500000.0 + index * 1000
        total, cash = (values or {}).get(day, (capital + market, capital))
        rows.append(
            {
                "date": pd.Timestamp(day),
                "portfolio_value": total,
                "capital": cash,
                "market_value": total - cash,
            }
        )
    return rows


def nav_row(day: str = TRADE_DATE) -> dict[str, Any]:
    """单个业务日的持久 NAV 行（与权威三原值同值）。"""
    entry = next(item for item in portfolio_values() if item["date"].strftime("%Y%m%d") == day)
    return {
        "trade_date": day,
        "cash": entry["capital"],
        "position_value": entry["market_value"],
        "total_value": entry["portfolio_value"],
        "nav": entry["portfolio_value"] / INITIAL_CAPITAL,
    }


def nav_rows(
    *,
    overrides: Mapping[str, Mapping[str, float]] | None = None,
    drop: Sequence[str] = (),
    only_days: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """持久 NAV 行（``overrides`` 用于合成篡改；``only_days`` 模拟真实跳空）。"""
    selected = None if only_days is None else {str(day) for day in only_days}
    rows: list[dict[str, Any]] = []
    for entry in portfolio_values():
        day = entry["date"].strftime("%Y%m%d")
        if day in drop or (selected is not None and day not in selected):
            continue
        row = nav_row(day)
        row.update(dict((overrides or {}).get(day, {})))
        rows.append(row)
    return rows


def trade_row(
    *,
    trade_date: str = TRADE_DATE,
    ts_code: str = "600000.SH",
    action: str = "buy",
    shares: int = 100,
    price: float = 10.0,
    reason: str = "信号生成",
) -> dict[str, Any]:
    """单笔成交的 **11 列投影形态**（``trades.parquet`` 行；成本三列不参与身份判断）。"""
    amount = round(shares * price, 6)
    return {
        "trade_date": trade_date,
        "ts_code": ts_code,
        "action": action,
        "shares": shares,
        "price": price,
        "amount": amount,
        "commission": 5.0,
        "stamp_tax": 0.0,
        "slippage": 0.5,
        "total_cost": 5.5,
        "reason": reason,
    }


def authority_trade_record(
    *,
    trade_date: str = TRADE_DATE,
    ts_code: str = "600000.SH",
    action: str = "buy",
    shares: int = 100,
    price: float = 10.0,
    reason: str = "信号生成",
) -> dict[str, Any]:
    """单笔成交的**引擎记录形态**（``EngineState.trades`` 行，生产真实形态）。"""
    amount = round(shares * price, 6)
    return {
        "date": pd.Timestamp(trade_date),
        "stock": ts_code,
        "action": action,
        "shares": shares,
        "price": price,
        "amount": amount,
        "cost": 5.5,
        "reason": reason,
    }


def to_engine_trade_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """把 11 列投影行转成引擎记录形态（夹具内部使用，保持权威侧形态真实）。"""
    return {
        "date": row.get("date") or row.get("trade_date"),
        "stock": row.get("stock") or row.get("ts_code"),
        "action": row.get("action"),
        "shares": int(row["shares"]),
        "price": float(row["price"]),
        "amount": float(row["amount"]),
        "cost": float(row.get("cost", row.get("total_cost", 0.0))),
        **({"reason": row["reason"]} if row.get("reason") else {}),
    }


def instruction_rows(
    *, source_date: str = TRADE_DATE, count: int = 3, execution_date: str = FUTURE_DATE
) -> list[dict[str, Any]]:
    """未来指令行（12 列契约）；``source_date`` 默认 = 当前权威日。"""
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(
            {
                "ts_code": f"6{index:05d}.SH",
                "action": "buy" if index % 2 == 0 else "sell",
                "shares": 100 * (index + 1),
                "price_type": "close",
                "reason": "调仓计划",
                "source_date": source_date,
                "target_weight": 0.05,
                "original_signal_date": source_date,
                "desired_position_count": 20,
                "retry_attempt": 0,
                "replacement_slot_code": "",
                "keep_buy_date": False,
            }
        )
    return rows


def build_engine_blob(
    *,
    trade_date: str = TRADE_DATE,
    run_kind: str = "t1",
    cash: float = CASH,
    positions: Mapping[str, dict[str, Any]] | None = None,
    portfolio: Sequence[Mapping[str, Any]] | None = None,
    authority_trades: Sequence[Mapping[str, Any]] | None = None,
    generation: bool = True,
) -> bytes:
    """构造 v2 权威 blob 字节（``EngineState`` 正规往返 + 顶层 ``paper_generation``）。"""
    state = EngineState(
        current_capital=float(cash),
        positions={key: dict(value) for key, value in (positions or POSITIONS).items()},
        pending_signals={},
        pending_stop_loss_sells={},
        pending_condition_sells={},
        _cycle_anchor_idx=0,
        portfolio_values=[
            dict(row) for row in (portfolio_values() if portfolio is None else portfolio)
        ],
        trades=[dict(row) for row in (authority_trades or [])],
        execution_attribution_records=[],
        holdings_snapshots=[],
        unfilled_slots={},
        completion_stats={},
        _normalize_log_count=0,
        _last_ranked_candidates=[],
        _last_signal_date=None,
        _last_rebalance_nav=None,
        _signal_dates={},
        pending_order_manager=None,
        stop_loss_monitor=None,
        ledger=Ledger().export_state(),
        live_anchor_state=None,
    )
    blob = state_to_blob(state)
    if generation:
        blob["paper_generation"] = {"trade_date": trade_date, "run_kind": run_kind}
    return dumps_blob(blob)


def build_paper_root(
    root: Path,
    *,
    trade_date: str = TRADE_DATE,
    run_kind: str = "t1",
    engine: str | None = "v2",
    record_extra: Mapping[str, Any] | None = None,
    drop_record_keys: Sequence[str] = (),
    account_override: Mapping[str, Any] | None = None,
    authority_trades: Sequence[Mapping[str, Any]] | None = None,
    authority_trades_shape: str = "engine",
    trades_rows: Sequence[Mapping[str, Any]] | None = None,
    nav: Sequence[Mapping[str, Any]] | None = None,
    nav_file: bool = True,
    instructions: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    pending_buys: Any = (),
    pending_sells: Any = (),
    pending_buys_present: bool = True,
    pending_sells_present: bool = True,
    rebalance: Any = None,
    config: Mapping[str, Any] | None = None,
    blob: bytes | None = None,
) -> dict[str, Any]:
    """构造完整的合成纸面根；返回关键期望值（供断言与投影对照）。

    ``authority_trades=None`` ⇒ 默认一笔当前日成交（正常根形态）；显式传 ``[]`` 表示零成交。
    ``nav=None`` ⇒ 按**跳空语义**只写有成交日的 NAV 行（与生产同构）。
    """
    root = Path(root)
    for relative in (
        "state",
        "nav",
        "trades",
        "runs",
        "pending_buys",
        "pending_sells",
        "instructions",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    rows_in = [trade_row()] if authority_trades is None else list(authority_trades)
    projected_default = [trade_row()] if authority_trades is None else list(authority_trades)
    if authority_trades_shape == "engine":
        authoritative = [to_engine_trade_record(row) for row in rows_in]
    else:
        authoritative = [dict(row) for row in rows_in]
    projection_rows = list(trades_rows if trades_rows is not None else projected_default)
    (root / "state" / "engine_state.bin").write_bytes(
        blob
        if blob is not None
        else build_engine_blob(
            trade_date=trade_date,
            run_kind=run_kind,
            authority_trades=authoritative,
        )
    )
    config_payload = dict(config) if config is not None else {
        "engine": "v2",
        "initial_capital": INITIAL_CAPITAL,
        "model_version": 24001,
        "model_version_b": None,
        "ensemble_weight_a": 1.0,
    }
    (root / "config.yaml").write_text(
        yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    positions = {key: dict(value) for key, value in POSITIONS.items()}
    account = build_account_projection(CASH, positions, trade_date)
    if account_override:
        account.update(dict(account_override))
    (root / "state" / "account.json").write_text(
        json.dumps(account, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    if nav_file:
        traded_days = sorted({str(row["trade_date"]) for row in projection_rows})
        nav_selected = (
            list(nav) if nav is not None else nav_rows(only_days=traded_days)
        )
        pd.DataFrame(nav_selected, columns=list(NAV_COLUMNS)).to_parquet(
            root / "nav" / "nav.parquet", index=False
        )
    pd.DataFrame(list(projection_rows), columns=list(TRADES_COLUMNS)).to_parquet(
        root / "trades" / "trades.parquet", index=False
    )
    record: dict[str, Any] = {"trade_date": trade_date}
    if engine is not None:
        record["engine"] = engine
    if run_kind:
        record["run_kind"] = run_kind
    record.update(
        {
            "t0_status": "success",
            "buy_price_type": "close",
            "sell_price_type": "open",
            "instructions_count": 0,
            "orders_count": 0,
            "fills_count": len(projection_rows),
            "pending_buys_count": len(pending_buys) if isinstance(pending_buys, list) else 0,
            "pending_sells_count": len(pending_sells) if isinstance(pending_sells, list) else 0,
            "nav": TOTAL_VALUE,
            "timestamp": "2026-06-03T15:30:00",
        }
    )
    record.update(dict(record_extra or {}))
    for key in drop_record_keys:
        record.pop(key, None)
    (root / "runs" / f"{run_kind}_{trade_date}.json").write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    if rebalance is None:
        rebalance_payload: Any = build_rebalance_state_projection(_anchor(trade_date))
    else:
        rebalance_payload = dict(rebalance) if isinstance(rebalance, Mapping) else rebalance
    (root / "runs" / "rebalance_state.json").write_text(
        json.dumps(rebalance_payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    if pending_buys_present:
        (root / "pending_buys" / "pending_buys.json").write_text(
            json.dumps(pending_buys, ensure_ascii=False), encoding="utf-8"
        )
    if pending_sells_present:
        (root / "pending_sells" / "pending_sells.json").write_text(
            json.dumps(pending_sells, ensure_ascii=False), encoding="utf-8"
        )
    plans = dict(instructions or {})
    for execution_day, rows in plans.items():
        pd.DataFrame(list(rows), columns=list(INSTRUCTION_COLUMNS)).to_parquet(
            root / "instructions" / f"{execution_day}.parquet", index=False
        )
    return {
        "root": root,
        "trade_date": trade_date,
        "run_kind": run_kind,
        "account": account,
        "record": record,
        "instructions": plans,
        "authority_trades": authoritative,
        "trades_rows": projection_rows,
    }


def _anchor(trade_date: str) -> Any:
    """合成 ``LiveAnchorState``（五字段投影源）。"""
    from src.lazybull.v2.core.execution.live_schedule import LiveAnchorState

    return LiveAnchorState(
        last_rebalance_date=None,
        last_scheduled_rebalance_date=None,
        anchor_date=trade_date,
        rebalance_freq=20,
        stagger_tranches=2,
    )


def build_projection(
    paper_root: Path,
    staging_root: Path,
    *,
    attempt_id: str = "p6_t1_attempt_a",
    generated_at: str = "20261105T000000Z",
    **kwargs: Any,
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    """采集稳定快照并装配驾驶舱 projection（返回 projection/快照/审计/采集信息）。"""
    from src.lazybull.v2.reporting.cockpit import build_dashboard_projection, capture_paper_inputs
    from src.lazybull.v2.reporting.snapshot import capture_stable_snapshot

    info: dict[str, Any] = {}

    def reader(snapshot: Any) -> None:
        info.clear()
        info.update(capture_paper_inputs(snapshot))

    snapshot, audit = capture_stable_snapshot(
        roots={"paper": Path(paper_root)},
        staging_root=Path(staging_root),
        attempt_ids=[attempt_id],
        reader=reader,
        max_retries=0,
    )
    projection = build_dashboard_projection(
        snapshot=snapshot, generated_at=generated_at, capture_info=info, **kwargs
    )
    return projection, snapshot, audit, info


def write_config_root(root: Path, config: Mapping[str, Any]) -> Path:
    """仅写 ``config.yaml``（配置顺序/冲突用例）。"""
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.yaml").write_text(
        yaml.safe_dump(dict(config), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return target / "config.yaml"
