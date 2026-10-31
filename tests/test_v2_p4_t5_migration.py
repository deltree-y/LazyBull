# -*- coding: utf-8 -*-
"""T5 账户迁移、adjust 与回滚投影的合成测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_migration as migration  # noqa: E402
from src.lazybull.paper import PaperStorage  # noqa: E402
from src.lazybull.paper.models import AccountState, Position  # noqa: E402
from src.lazybull.v2.core.accounting.engine_state import EngineState  # noqa: E402


def _state() -> EngineState:
    return EngineState(
        current_capital=1000.0,
        positions={},
        pending_signals={},
        pending_stop_loss_sells={},
        pending_condition_sells={},
        _cycle_anchor_idx=0,
        portfolio_values=[],
        trades=[],
        execution_attribution_records=[],
        holdings_snapshots=[],
        unfilled_slots={},
        completion_stats={},
        _normalize_log_count=0,
        _last_ranked_candidates=[],
        _last_signal_date=None,
        _last_rebalance_nav=None,
        _signal_dates={},
    )


def _position() -> Position:
    return Position(
        ts_code="A",
        shares=100,
        buy_price=10.0,
        buy_cost=5.0,
        buy_date="20261001",
        buy_pnl_price=12.0,
    )


def test_inventory_separates_migrate_rollback_and_dead(tmp_path: Path) -> None:
    rows = migration.build_file_inventory(tmp_path)
    actions = {row["path"]: row["action"] for row in rows}
    assert actions["state/account.json"] == "migrate"
    assert actions["state/exposure_policy.json"] == "rollback_only"
    assert actions["account.json"] == "dead"
    assert actions["state/weakness_exit_state.json"] == "dead"


def test_map_account_preserves_fee_and_principal_semantics() -> None:
    account = AccountState(cash=100.0, positions={"A": _position()}, last_update="20261030")
    position = migration.map_account(account)["A"]
    assert position["buy_cost_cash"] == 1005.0
    assert position["buy_trade_price"] == 10.0
    assert position["buy_pnl_price"] == 12.0
    assert position["buy_date"] == pd.Timestamp("20261001")


def test_map_anchor_accepts_legacy_two_field_shape() -> None:
    anchor = migration.map_anchor(
        {"last_rebalance_date": "20260602", "rebalance_freq": 20},
        default_freq=20,
        default_tranches=2,
    )
    assert anchor.last_rebalance_date == pd.Timestamp("20260602")
    assert anchor.anchor_date is None
    assert anchor.last_scheduled_rebalance_date is None
    assert anchor.stagger_tranches == 2


def test_map_pending_orders_preserves_attempts_and_target_value(tmp_path: Path) -> None:
    storage = PaperStorage(root_path=str(tmp_path / "paper"))
    (storage.pending_buys_path / "A.json").write_text(
        json.dumps(
            {
                "ts_code": "A",
                "target_weight": 0.1,
                "create_date": "20261029",
                "original_signal_date": "20261028",
                "attempts": 3,
                "reason": "涨停",
            }
        ),
        encoding="utf-8",
    )
    result = migration.map_pending_orders(storage, total_value=5000.0)
    order = result["pending_orders"][("A", "buy")]
    assert order.target_value == 500.0
    assert order.retry_count == 3
    assert order.signal_date == pd.Timestamp("20261028")


def test_map_trades_and_ledger_rebuilds_stable_keys() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20261029", "20261030"],
            "ts_code": ["A", "B"],
            "action": ["buy", "sell"],
            "total_cost": [5.0, 6.0],
        }
    )
    records, ledger = migration.map_trades_and_ledger(frame)
    assert len(records) == 2
    assert list(ledger["applied_keys"]) == ["20261029|buy|A|0", "20261030|sell|B|1"]
    assert ledger["last_fill_date"] == pd.Timestamp("20261030")


def test_adjust_cash_and_delete_position() -> None:
    state = _state()
    state.positions = migration.map_account(AccountState(cash=1000.0, positions={"A": _position()}))
    migration.apply_adjustment(state, command="cash", trade_date="20261030", values={"set": 500.0})
    assert state.current_capital == 500.0
    migration.apply_adjustment(
        state, command="delete-position", trade_date="20261030", values={"ts_code": "A"}
    )
    assert state.current_capital == 1500.0
    assert state.positions == {}


def test_adjust_update_position_cash_delta() -> None:
    state = _state()
    state.positions = migration.map_account(AccountState(cash=0.0, positions={"A": _position()}))
    migration.apply_adjustment(
        state,
        command="update-position",
        trade_date="20261030",
        values={"ts_code": "A", "shares": 200, "buy_price": 8.0},
    )
    assert state.current_capital == 400.0
    assert state.positions["A"]["buy_cost_cash"] == 1600.0


def test_adjust_add_shares_weighted_price_and_cash() -> None:
    state = _state()
    state.current_capital = 2000.0
    state.positions = migration.map_account(AccountState(cash=0.0, positions={"A": _position()}))
    migration.apply_adjustment(
        state,
        command="add-shares",
        trade_date="20261030",
        values={"ts_code": "A", "shares": 100, "price": 20.0},
    )
    assert state.current_capital == 0.0
    assert state.positions["A"]["shares"] == 200
    assert state.positions["A"]["buy_trade_price"] == 15.0
    assert state.positions["A"]["buy_cost_cash"] == 3000.0
    assert state.positions["A"]["buy_date"] == pd.Timestamp("20261001")


def test_last_completed_run_day_uses_runs_not_account(tmp_path: Path) -> None:
    storage = migration.PaperStorage(root_path=str(tmp_path / "paper"))
    storage.runs_path.mkdir(parents=True, exist_ok=True)
    (storage.runs_path / "t0_20260716.json").write_text("{}", encoding="utf-8")
    (storage.runs_path / "t1_20260715.json").write_text("{}", encoding="utf-8")
    assert migration._last_completed_run_day(storage, "20260706") == "20260716"


def test_adjust_add_shares_rejects_insufficient_cash() -> None:
    state = _state()
    state.current_capital = 0.0
    state.positions = migration.map_account(AccountState(cash=0.0, positions={"A": _position()}))
    with pytest.raises(RuntimeError, match="现金不足"):
        migration.apply_adjustment(
            state,
            command="add-shares",
            trade_date="20261030",
            values={"ts_code": "A", "shares": 100, "price": 20.0},
        )


def test_rollback_account_uses_legacy_fee_projection() -> None:
    state = _state()
    state.positions = migration.map_account(AccountState(cash=0.0, positions={"A": _position()}))
    account = migration.rollback_account(state, "20261030")
    assert account["last_update"] == "20261030"
    assert account["positions"]["A"]["buy_cost"] == 5.0
    assert account["positions"]["A"]["buy_price"] == 10.0


def test_frozen_valuation_binds_market_date_and_total(tmp_path: Path) -> None:
    day_dir = tmp_path / "clean" / "daily"
    day_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts_code": ["A", "B"],
            "close": [10.0, 20.0],
            "pre_close": [9.0, 19.0],
            "tradable": [1, 1],
            "is_st": [0, 0],
            "is_suspended": [0, 0],
        }
    ).to_parquet(day_dir / "2026-10-30.parquet")
    account = {
        "cash": 0.0,
        "positions": {"A": {"shares": 100}, "B": {"shares": 50}},
    }
    assessment = migration.assess_holdings(account, tmp_path)
    frozen = {
        "market_date": "20261030",
        "mark_prices": {"A": 10.0, "B": 20.0},
        "total_value": assessment["total_value"],
        "content_fingerprint": "fp",
    }
    marks, total, meta = migration._frozen_valuation(
        assessment=assessment, valuation=frozen, account=account
    )
    assert marks == {"A": 10.0, "B": 20.0}
    assert total == pytest.approx(2000.0)
    assert meta["source"] == "frozen_gate1_holdings"


def test_frozen_valuation_rejects_stale_market_date(tmp_path: Path) -> None:
    day_dir = tmp_path / "clean" / "daily"
    day_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts_code": ["A"],
            "close": [10.0],
            "pre_close": [10.0],
            "tradable": [1],
            "is_st": [0],
            "is_suspended": [0],
        }
    ).to_parquet(day_dir / "2026-10-30.parquet")
    account = {"cash": 0.0, "positions": {"A": {"shares": 100}}}
    assessment = migration.assess_holdings(account, tmp_path)
    with pytest.raises(RuntimeError, match="冻结时点不一致"):
        migration._frozen_valuation(
            assessment=assessment,
            valuation={
                "market_date": "20261029",
                "mark_prices": {"A": 10.0},
                "total_value": 1000.0,
            },
            account=account,
        )


def test_frozen_valuation_rejects_total_mismatch(tmp_path: Path) -> None:
    day_dir = tmp_path / "clean" / "daily"
    day_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts_code": ["A"],
            "close": [10.0],
            "pre_close": [10.0],
            "tradable": [1],
            "is_st": [0],
            "is_suspended": [0],
        }
    ).to_parquet(day_dir / "2026-10-30.parquet")
    account = {"cash": 0.0, "positions": {"A": {"shares": 100}}}
    assessment = migration.assess_holdings(account, tmp_path)
    with pytest.raises(RuntimeError, match="总值与 D10 复算不一致"):
        migration._frozen_valuation(
            assessment=assessment,
            valuation={"market_date": "20261030", "mark_prices": {"A": 10.0}, "total_value": 999.0},
            account=account,
        )


def test_account_projection_equivalence_checks_cash_and_positions() -> None:
    source = {
        "cash": 100.0,
        "positions": {"A": {"shares": 100, "buy_price": 10.0, "buy_cost": 5.0}},
    }
    same = {
        "cash": 100.0,
        "positions": {"A": {"shares": 100, "buy_price": 10.0, "buy_cost": 5.0}},
    }
    assert migration._account_projection_matches(source, same) is True
    assert migration._account_projection_matches(source, {**same, "cash": 100.5}) is False
    assert migration._account_projection_matches(source, {"cash": 100.0, "positions": {}}) is False


def test_migrate_isolated_rejects_production_target(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    with pytest.raises(RuntimeError, match="只允许全新隔离 target"):
        migration.migrate_isolated(
            source=tmp_path / "source",
            target=data_root / "paper",
            data_root=data_root,
            evidence=tmp_path / "evidence",
            execute=True,
        )


def test_migrate_isolated_dry_run_does_not_create_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    PaperStorage(root_path=str(source))
    target = tmp_path / "target"
    result = migration.migrate_isolated(
        source=source,
        target=target,
        data_root=tmp_path / "data",
        evidence=tmp_path / "evidence",
        execute=False,
    )
    assert result["verdict"] == "DRY_RUN"
    assert not target.exists()
