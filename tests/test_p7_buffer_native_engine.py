"""固定续持研究子类的合成执行验证；不读取真实模型、特征或行情。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import pytest

from scripts.v2_p7.buffer_native_engine import BufferNativeEngine
from src.lazybull.universe.base import Universe
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.core.accounting.engine_state import export_state
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger
from src.lazybull.v2.core.signal.base import Signal
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML


class _Universe(Universe):
    def get_stocks(self, date: pd.Timestamp, quote_data: Any = None) -> list[str]:
        return [f"S{i:02}" for i in range(40)]


class _Signal(Signal):
    def __init__(self, ranks: dict[pd.Timestamp, list[tuple[str, float]]]) -> None:
        super().__init__("synthetic_buffer")
        self.top_n = 20
        self.ranks = ranks

    def generate(self, date: pd.Timestamp, universe: list[str], data: dict) -> dict[str, float]:
        return dict(self.generate_ranked(date, universe, data)[:20])

    def generate_ranked(self, date: pd.Timestamp, universe: list[str], data: dict) -> list:
        return list(self.ranks[date])


def _data(n: int = 90) -> tuple:
    days = list(pd.bdate_range("20200101", periods=n))
    ranked = {day: [(f"S{i:02}", float(40 - i)) for i in range(40)] for day in days}
    prices = pd.DataFrame(
        [
            {
                "trade_date": day.strftime("%Y%m%d"),
                "ts_code": code,
                "open": 9.0,
                "close": 10.0,
                "open_adj": 9.0,
                "close_adj": 10.0,
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
                "tradable": True,
                "is_st": False,
                "list_days": 1000,
                "pct_chg": 0.0,
                "vol": 1000000.0,
            }
            for day in days
            for code, _score in ranked[day]
        ]
    )
    return days, ranked, prices


def _engine(days: list, ranks: dict, *, enabled: bool | None, **overrides: Any) -> Any:
    kwargs = dict(
        universe=_Universe("test"),
        signal=_Signal(ranks),
        features_by_date={
            day.strftime("%Y%m%d"): pd.DataFrame({"ts_code": ["S00"]}) for day in days
        },
        initial_capital=1_000_000.0,
        cost_model=CostModel(
            commission_rate=0.0001954, min_commission=5, stamp_tax=0.0005, slippage=0.0005
        ),
        rebalance_freq=20,
        stagger_tranches=2,
        position_sizing="equal",
        max_weight_per_stock=0.15,
        min_buy_value_ratio=0,
        enable_position_completion=True,
        completion_window_days=5,
        enable_pending_order=True,
        enable_early_rebalance_on_empty=False,
        sell_timing="open",
        verbose=False,
        portfolio_merger=SingleSleevePortfolioMerger(),
        sleeve_id="value",
    )
    kwargs.update(overrides)
    if enabled is None:
        engine = NativeMergeBacktestEngineML(**kwargs)
    else:
        engine = BufferNativeEngine(
            **kwargs, buffer_enabled=enabled, rank_provider=lambda d: ranks[d]
        )
    engine.record_holdings_snapshot = True
    return engine


def _run(engine: Any, days: list, prices: pd.DataFrame) -> pd.DataFrame:
    return engine.run(days[0], days[-1], days, prices)


def _wire(engine: Any, days: list, prices: pd.DataFrame) -> None:
    engine._prepare_price_index(prices)
    engine.price_data_cache = prices
    engine._trade_date_index = {day: i for i, day in enumerate(days)}
    engine._signal_dates = {days[0]: 0, days[10]: 1}


def _seed(engine: Any, days: list, code: str, batch: int, shares: int = 5000) -> None:
    signal_date = days[batch]
    engine.positions[code] = {
        "shares": shares,
        "buy_date": days[1],
        "signal_date": signal_date,
        "buy_trade_price": 10.0,
        "buy_pnl_price": 10.0,
        "buy_cost_cash": shares * 10.0,
    }
    engine.current_capital -= shares * 10.0
    engine._buffer_meta[code] = {
        "life": (days[1], signal_date),
        "tranche_idx": batch,
        "renewal_count": 0,
    }
    engine._buffer_owner[(code, signal_date)] = batch


def test_disabled_exact_native_trades_nav_positions_and_state() -> None:
    """关闭覆写不改变真实候补、补齐、成交钟或账户状态。"""
    days, ranks, prices = _data(65)
    prices.loc[prices.trade_date == days[1].strftime("%Y%m%d"), "is_limit_up"] = True
    native = _engine(days, ranks, enabled=None)
    disabled = _engine(days, ranks, enabled=False)
    left, right = _run(native, days, prices), _run(disabled, days, prices)
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    pd.testing.assert_frame_equal(native.get_trades(), disabled.get_trades(), check_exact=True)
    pd.testing.assert_frame_equal(
        native.get_holdings_snapshot(), disabled.get_holdings_snapshot(), check_exact=True
    )
    assert native.positions == disabled.positions
    assert native.current_capital == disabled.current_capital
    assert export_state(native).positions == export_state(disabled).positions
    assert native.unfilled_slots == disabled.unfilled_slots
    assert disabled.events == []
    assert set(native.get_trades().loc[lambda f: f.action.eq("buy"), "price"]) == {10.0}


def test_repeated_renewal_keeps_actual_buy_date_and_never_adds_shares() -> None:
    days, ranks, prices = _data(90)
    for day in days[70:]:
        ranks[day] = []
    engine = _engine(days, ranks, enabled=True)
    _run(engine, days, prices)
    rows = [row for row in engine.events if row["kind"] == "expiry" and row["ts_code"] == "S00"]
    assert [row["holding_days"] for row in rows] == [19, 39, 59, 79]
    assert [row["renew"] for row in rows] == [True, True, True, False]
    assert {row["buy_date"] for row in rows} == {days[1].strftime("%Y%m%d")}
    buys = [row for row in engine.trades if row["stock"] == "S00" and row["action"] == "buy"]
    sells = [row for row in engine.trades if row["stock"] == "S00" and row["action"] == "sell"]
    assert len(buys) == len(sells) == 1
    assert buys[0]["shares"] == sells[0]["shares"]
    assert sells[0]["date"] == days[81]


def test_falling_out_between_due_dates_does_not_exit_early() -> None:
    days, ranks, prices = _data(45)
    for day in days[25:]:
        ranks[day] = ranks[day][1:] + ranks[day][:1]
    engine = _engine(days, ranks, enabled=True)
    _run(engine, days, prices)
    sell = next(row for row in engine.trades if row["stock"] == "S00" and row["action"] == "sell")
    assert sell["date"] == days[41]
    renewals = [row for row in engine.events if row["kind"] == "expiry" and row["ts_code"] == "S00"]
    assert [(row["holding_days"], row["renew"]) for row in renewals] == [(19, True), (39, False)]


@pytest.mark.parametrize("queue", ["condition", "stop", "pending"])
def test_existing_sell_obligation_never_cancelled(queue: str) -> None:
    days, ranks, prices = _data(23)
    engine = _engine(days, ranks, enabled=True)
    _wire(engine, days, prices)
    _seed(engine, days, "S00", 0)
    if queue == "condition":
        engine.pending_condition_sells["S00"] = {
            "trigger_date": days[19],
            "sell_type": "holding_period",
        }
    elif queue == "stop":
        engine.pending_stop_loss_sells["S00"] = {"trigger_date": days[19]}
    else:
        engine.pending_order_manager.add_order("S00", "sell", days[19], days[19], reason="跌停")
    engine._check_and_sell(days[20], days, engine._trade_date_index)
    assert engine._has_sell_obligation("S00")
    assert engine._buffer_meta["S00"]["renewal_count"] == 0
    assert not any(row["kind"] == "expiry" for row in engine.events)


def test_same_day_fallback_uses_native_candidate_order_and_original_batch() -> None:
    days, ranks, prices = _data(5)
    prices.loc[
        (prices.trade_date == days[1].strftime("%Y%m%d")) & prices.ts_code.eq("S00"), "is_limit_up"
    ] = True
    engine = _engine(days, ranks, enabled=True)
    _run(engine, days, prices)
    bought = [row["stock"] for row in engine.trades if row["action"] == "buy"]
    assert bought == [f"S{i:02}" for i in range(1, 11)]
    assert all(meta["tranche_idx"] == 0 for meta in engine._buffer_meta.values())


def test_cross_day_completion_reuses_native_window_and_actual_buy_date() -> None:
    days, ranks, prices = _data(24)
    prices.loc[prices.trade_date == days[1].strftime("%Y%m%d"), "is_limit_up"] = True
    engine = _engine(days, ranks, enabled=True)
    _run(engine, days, prices)
    first = next(row for row in engine.trades if row["stock"] == "S00" and row["action"] == "buy")
    assert first["date"] == days[2] and first["signal_date"] == days[0]
    expiry = next(
        row for row in engine.events if row["kind"] == "expiry" and row["ts_code"] == "S00"
    )
    assert expiry["trade_date"] == days[21].strftime("%Y%m%d")
    assert expiry["holding_days"] == 19
    assert engine._buffer_meta["S00"]["tranche_idx"] == 0
    assert engine.completion_stats["total_completed"] > 0


def test_retained_budget_and_fixed_shares_cannot_buy_extra_or_add_to_retained() -> None:
    days, ranks, prices = _data(23)
    engine = _engine(days, ranks, enabled=True)
    _wire(engine, days, prices)
    _seed(engine, days, "S00", 0, shares=40_000)
    # 尚未卖出的待卖股与正常保留股一样占额，不能把未兑现卖款作为新批预算。
    engine.pending_condition_sells["S00"] = {
        "trigger_date": days[19],
        "sell_type": "holding_period",
    }
    engine._buffer_exec_batch = 0
    engine._buy_stock_direct(days[20], "S01", 900_000, days[19], requested_shares=90_000)
    assert engine.positions["S01"]["shares"] <= 10_000
    before = deepcopy(engine.positions["S00"])
    engine._buy_stock_direct(days[20], "S00", 100_000, days[19], requested_shares=10_000)
    assert engine.positions["S00"] == before
    assert engine.current_capital >= 0
    assert any(row["kind"] == "tranche_budget_cap" for row in engine.events)


@pytest.mark.parametrize("delayed", [False, True])
def test_mixed_expiry_refill_fallback_keeps_real_original_batch(delayed: bool) -> None:
    days, ranks, prices = _data(23)
    engine = _engine(days, ranks, enabled=True)
    _wire(engine, days, prices)
    _seed(engine, days, "S00", 0)
    _seed(engine, days, "S01", 1)
    ranks[days[20]] = ranks[days[20]][2:] + ranks[days[20]][:2]
    engine._check_and_sell(days[20], days, engine._trade_date_index)
    assert [g["tranche_idx"] for g in engine.pending_signals[days[20]]["buffer_groups"]] == [0, 1]
    if delayed:
        prices.loc[prices.trade_date == days[21].strftime("%Y%m%d"), "is_limit_up"] = True
    engine._execute_pending_condition_sells(days[21], days, engine._trade_date_index)
    engine._execute_pending_buys(days[21], days, engine._trade_date_index)
    if delayed:
        assert set(engine.unfilled_slots) == {(days[20], 0), (days[20], 1)}
        engine._process_position_completion(days[22], days, prices, engine._trade_date_index)
    # D+2 必须消费 D+1 的最新排名；已售旧股重新排前时可成为新槽候补。
    first_code, second_code = ("S00", "S01") if delayed else ("S02", "S03")
    assert engine._buffer_meta[first_code]["tranche_idx"] == 0
    assert engine._buffer_meta[second_code]["tranche_idx"] == 1
    execution_day = days[22] if delayed else days[21]
    assert engine.positions[first_code]["buy_date"] == execution_day
    assert engine.positions[second_code]["buy_date"] == execution_day


def test_completion_window_stops_after_original_five_days() -> None:
    days, ranks, prices = _data(8)
    prices.loc[
        prices.trade_date.isin([day.strftime("%Y%m%d") for day in days[1:6]]), "is_limit_up"
    ] = True
    engine = _engine(days, ranks, enabled=True)
    _run(engine, days, prices)
    assert engine.trades == []
    assert engine.unfilled_slots == {}
    assert engine.completion_stats["total_abandoned"] == 1


def test_full_original_tranche_blocks_extra_slot() -> None:
    days, ranks, prices = _data(23)
    engine = _engine(days, ranks, enabled=True)
    _wire(engine, days, prices)
    for index in range(10):
        _seed(engine, days, f"S{index:02}", 0, shares=100)
    engine._buffer_exec_batch = 0
    engine._buy_stock_direct(days[20], "S10", 10_000, days[19])
    assert "S10" not in engine.positions
    assert engine.events[-1]["kind"] == "tranche_slot_block"
