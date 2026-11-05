# -*- coding: utf-8 -*-
"""T5 账户迁移、adjust 与回滚投影的合成测试。

估值口径（v0.209.9）：迁移**不得**用最新截面价格回填旧业务日；业务日同日 nav 行逐位
沿用，缺失才按业务日 clean 截面复算；「当前市值」只落 ``state/valuation_preview.json``。
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4 import t5_migration as migration  # noqa: E402
from src.lazybull.paper import PaperStorage  # noqa: E402
from src.lazybull.paper.models import AccountState, NAVRecord, Position  # noqa: E402
from src.lazybull.v2.core.accounting.engine_state import EngineState  # noqa: E402
from src.lazybull.v2.hosts.paper.state_store import PaperStateStore  # noqa: E402

#: migrate_state 只需 trading_config 的三个属性（不构造完整宿主）
_CFG = types.SimpleNamespace(rebalance_freq=20, stagger_tranches=2, stop_loss_enabled=False)


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


# ---------------------------------------------------------------------------
# 估值口径（v0.209.9）：业务日绑定、旧行沿用、只读旁路
# ---------------------------------------------------------------------------


def _write_section(data_root: Path, day: str, closes: dict) -> None:
    """合成 clean/daily 分区（day = YYYYMMDD）。"""
    directory = data_root / "clean" / "daily"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ts_code": list(closes),
            "close": list(closes.values()),
            "pre_close": list(closes.values()),
            "tradable": [1] * len(closes),
            "is_st": [0] * len(closes),
            "is_suspended": [0] * len(closes),
        }
    ).to_parquet(directory / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False)


def _legacy_storage(
    tmp_path: Path, *, cash: float = 100.0, last_update: str = "20260603", nav_rows=()
) -> PaperStorage:
    """旧链布局（state/account.json + nav/nav.parquet）。"""
    storage = PaperStorage(root_path=str(tmp_path / "paper"))
    storage.save_account_state(
        AccountState(cash=cash, positions={"A": _position()}, last_update=last_update)
    )
    for row_cash, position_value in nav_rows:
        total = row_cash + position_value
        storage.append_nav(
            NAVRecord(
                trade_date=last_update,
                cash=row_cash,
                position_value=position_value,
                total_value=total,
                nav=total / 1000.0,
            )
        )
    return storage


def test_assess_holdings_honors_explicit_market_date(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_section(data_root, "20260603", {"A": 10.0})
    _write_section(data_root, "20261029", {"A": 8.0})
    account = {"cash": 0.0, "positions": {"A": {"shares": 100}}}

    latest = migration.assess_holdings(account, data_root)
    assert latest["market_date"] == "20261029"
    assert latest["total_value"] == pytest.approx(800.0)

    # 显式业务日必须取该日分区（接受 2026-06-03 形态）
    frozen = migration.assess_holdings(account, data_root, market_date="2026-06-03")
    assert frozen["market_date"] == "20260603"
    assert frozen["total_value"] == pytest.approx(1000.0)


def test_assess_holdings_rejects_missing_explicit_section(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_section(data_root, "20261029", {"A": 8.0})
    account = {"cash": 0.0, "positions": {"A": {"shares": 100}}}
    with pytest.raises(FileNotFoundError, match="clean/daily 分区缺失"):
        migration.assess_holdings(account, data_root, market_date="20260603")


def test_valuation_prefers_legacy_nav_row_without_pricing(tmp_path: Path) -> None:
    """旧 NAV 同日行 = 已发布事实：逐位沿用，且**不需要任何行情分区**。"""
    valuation = migration.resolve_business_valuation(
        account={"cash": 100.0, "positions": {"A": {"shares": 100}}},
        business_day="20260603",
        legacy_row={"cash": 100.0, "position_value": 900.0, "total_value": 1000.0},
        frozen=None,
        data_root=tmp_path / "data",  # 无 clean/daily：若尝试定价会直接报错
    )
    assert valuation["source"] == "legacy_nav_row"
    assert valuation["market_date"] == "20260603"
    assert valuation["total_value"] == pytest.approx(1000.0)
    assert valuation["market_value"] == pytest.approx(900.0)


def test_valuation_rejects_legacy_row_cash_mismatch() -> None:
    with pytest.raises(RuntimeError, match="现金与账户不一致"):
        migration.resolve_business_valuation(
            account={"cash": 100.0, "positions": {"A": {"shares": 100}}},
            business_day="20260603",
            legacy_row={"cash": 90.0, "position_value": 900.0, "total_value": 990.0},
            frozen=None,
            data_root=Path("unused"),
        )


def test_valuation_falls_back_to_business_day_section(tmp_path: Path) -> None:
    """无同日 nav 行 ⇒ 用**业务日**收盘复算，不得用最新截面价格。"""
    data_root = tmp_path / "data"
    _write_section(data_root, "20260603", {"A": 10.0})
    _write_section(data_root, "20261029", {"A": 8.0})
    valuation = migration.resolve_business_valuation(
        account={"cash": 0.0, "positions": {"A": {"shares": 100}}},
        business_day="20260603",
        legacy_row=None,
        frozen=None,
        data_root=data_root,
    )
    assert valuation["source"] == "business_day_clean_section"
    assert valuation["market_date"] == "20260603"
    assert valuation["total_value"] == pytest.approx(1000.0)  # 10.0×100，不是 8.0×100


def test_valuation_fallback_rejects_missing_business_day(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_section(data_root, "20261029", {"A": 8.0})
    with pytest.raises(FileNotFoundError, match="clean/daily 分区缺失"):
        migration.resolve_business_valuation(
            account={"cash": 0.0, "positions": {"A": {"shares": 100}}},
            business_day="20260603",
            legacy_row=None,
            frozen=None,
            data_root=data_root,
        )


def test_migrate_state_keeps_business_day_nav_row_verbatim(tmp_path: Path) -> None:
    """权威末日行 = 旧 NAV 同日行（1000.0），不得按行情重估为 1100.0。"""
    storage = _legacy_storage(tmp_path, cash=100.0, nav_rows=[(100.0, 900.0)])
    valuation = {
        "cash": 100.0,
        "market_value": 900.0,
        "total_value": 1000.0,
        "market_date": "20260603",
        "source": "legacy_nav_row",
    }
    state = migration.migrate_state(
        _state(),
        storage=storage,
        trading_config=_CFG,
        calendar=[],
        valuation=valuation,
    )
    assert len(state.portfolio_values) == 1
    row = state.portfolio_values[-1]
    assert row["date"] == pd.Timestamp("20260603")
    assert row["portfolio_value"] == pytest.approx(1000.0)
    assert row["capital"] == pytest.approx(100.0)
    assert row["market_value"] == pytest.approx(900.0)


def test_migrate_state_appends_row_when_business_day_missing(tmp_path: Path) -> None:
    """无同日行（无成交日不写 nav）⇒ 追加一条按业务日复算的行。"""
    storage = _legacy_storage(tmp_path, cash=100.0, nav_rows=[])
    valuation = {
        "cash": 100.0,
        "market_value": 950.0,
        "total_value": 1050.0,
        "market_date": "20260603",
        "source": "business_day_clean_section",
    }
    state = migration.migrate_state(
        _state(),
        storage=storage,
        trading_config=_CFG,
        calendar=[],
        valuation=valuation,
    )
    assert len(state.portfolio_values) == 1
    assert state.portfolio_values[-1]["portfolio_value"] == pytest.approx(1050.0)


def test_migrate_state_rejects_valuation_from_other_day(tmp_path: Path) -> None:
    storage = _legacy_storage(tmp_path, cash=100.0, nav_rows=[(100.0, 900.0)])
    with pytest.raises(RuntimeError, match="其他时点的价格"):
        migration.migrate_state(
            _state(),
            storage=storage,
            trading_config=_CFG,
            calendar=[],
            valuation={
                "cash": 100.0,
                "market_value": 800.0,
                "total_value": 900.0,
                "market_date": "20261029",
                "source": "latest_clean_section",
            },
        )


def test_migrate_state_rejects_conflict_with_published_nav_row(tmp_path: Path) -> None:
    storage = _legacy_storage(tmp_path, cash=100.0, nav_rows=[(100.0, 900.0)])
    with pytest.raises(RuntimeError, match="禁止择一覆盖"):
        migration.migrate_state(
            _state(),
            storage=storage,
            trading_config=_CFG,
            calendar=[],
            valuation={
                "cash": 100.0,
                "market_value": 700.0,
                "total_value": 800.0,
                "market_date": "20260603",
                "source": "business_day_clean_section",
            },
        )


def test_valuation_preview_is_sidecar_only(tmp_path: Path) -> None:
    """当前市值只落 valuation_preview.json（带 market_date），不碰权威/NAV 投影。"""
    store = PaperStateStore(root_path=str(tmp_path / "paper"))
    assessment = {
        "market_date": "20261029",
        "cash": 100.0,
        "total_value": 900.0,
        "positions": [{"ts_code": "A", "mark_price": 8.0}],
    }
    preview = migration._write_valuation_preview(
        store, assessment=assessment, authority_last="20260603"
    )
    assert preview["market_date"] == "20261029"
    assert preview["authority_last_date"] == "20260603"
    assert preview["market_value"] == pytest.approx(800.0)
    assert preview["total_value"] == pytest.approx(900.0)
    assert preview["mark_prices"] == {"A": 8.0}
    assert store.load_valuation_preview() == preview
    # 只读旁路：既不写权威快照，也不写 nav 投影
    assert store.load_engine_state_blob() is None
    assert store.load_all_nav() is None
