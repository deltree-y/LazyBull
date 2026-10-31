# -*- coding: utf-8 -*-
"""P4-T3 E3：原生 TargetPortfolio/Order 双宿主链与过渡件删除验收。"""

from __future__ import annotations

import copy
import inspect
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.protocols.core import PortfolioMerger
from src.lazybull.v2.common.trading_config import TradingConfig
from src.lazybull.v2.common.types import (
    Money,
    Order,
    OrderReason,
    OrderSide,
    OrderType,
    Shares,
    TargetPortfolio,
    TradeDate,
    TSCode,
    Weight,
)
from src.lazybull.v2.core.accounting.nav import calculate_nav, generate_nav_curve
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger
from src.lazybull.v2.core.execution import buy_execution
from src.lazybull.v2.core.execution.ml_signal_feed import BacktestEngineML
from src.lazybull.v2.core.execution.signal_execution import BacktestSignalExecutionMixin
from src.lazybull.v2.core.signal.base import Signal
from src.lazybull.v2.hosts import native_engine as native_module
from src.lazybull.v2.hosts.backtest.replay import _run_replay_engine_and_collect
from src.lazybull.v2.hosts.backtest.runtime import LEGACY_BYPASS
from src.lazybull.v2.hosts.native_engine import (
    NativeMergeBacktestEngineML,
    build_native_account,
    project_t3_evidence_orders,
)
from src.lazybull.v2.hosts.backtest.runtime import create_backtest_engine_from_config
from src.lazybull.v2.hosts.paper.host import PaperDayResult, PaperHost
from src.lazybull.universe.base import Universe

_D0 = pd.Timestamp("2026-01-05")
_D1 = pd.Timestamp("2026-01-06")
_COST = {
    "commission_rate": 0.0003,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.001,
}


def _positions():
    return {
        "A": {
            "shares": 1000,
            "buy_date": _D0,
            "buy_trade_price": 10.0,
            "buy_pnl_price": 88.8,
            "buy_cost_cash": 10013.0,
            "last_known_price": 10.5,
        },
        "B": {
            "shares": 2000,
            "buy_date": _D0,
            "buy_trade_price": 20.0,
            "buy_pnl_price": 77.7,
            "buy_cost_cash": 40026.0,
        },
        "C": {"shares": 0, "buy_date": _D0, "buy_trade_price": 30.0, "buy_cost_cash": 0.0},
    }


def _native(positions=None):
    engine = object.__new__(NativeMergeBacktestEngineML)
    engine.current_capital = 500000.0
    engine.positions = _positions() if positions is None else positions
    engine.initial_capital = 777777.0
    engine._last_rebalance_nav = 1234567.89
    engine.portfolio_merger = SingleSleevePortfolioMerger()
    engine.merger_sleeve_id = "value"
    engine.max_weight_per_stock = None
    engine.verbose = False
    return engine


def _order(stock="A", shares=100, sleeve="value"):
    return Order(
        TSCode(stock),
        OrderSide.BUY,
        Shares(shares),
        OrderType.MARKET_OPEN,
        OrderReason.REBALANCE,
        sleeve,
    )


def test_calculate_nav_scalar_exact():
    assert calculate_nav(1234567.89, 777777.0) == 1234567.89 / 777777.0


def test_calculate_nav_series_semantics():
    index = pd.to_datetime(["2026-01-05", "2026-01-06"])
    values = pd.Series([100.0, np.nan], index=index, dtype="float64")
    result = calculate_nav(values, 100.0)
    assert result.index is index
    pd.testing.assert_series_equal(result, values / 100.0, check_exact=True)


def test_generate_nav_curve_delegates_and_exact():
    records = [
        {"date": _D0, "portfolio_value": 1000000.0},
        {"date": _D1, "portfolio_value": 1010000.5},
    ]
    result = generate_nav_curve(records, 1000000.0)
    expected = pd.DataFrame(records)
    expected["nav"] = expected["portfolio_value"] / 1000000.0
    expected["return"] = expected["nav"] - 1
    pd.testing.assert_frame_equal(result, expected, check_exact=True)


def test_native_account_fields_and_units():
    engine = _native()
    account = build_native_account(engine, engine._last_rebalance_nav)
    assert account.cash.value == 500000.0
    assert account.total_value.value == 1234567.89
    assert account.nav == 1234567.89 / 777777.0
    assert list(account.positions) == [TSCode("A"), TSCode("B")]
    assert account.positions[TSCode("A")].shares.value == 1000
    assert account.positions[TSCode("A")].cost_basis.value == 10013.0


def test_native_account_last_price_fallback_never_pnl_price():
    account = build_native_account(_native(), 1234567.89)
    assert account.positions[TSCode("A")].last_price.value == 10.5
    assert account.positions[TSCode("B")].last_price.value == 20.0


def test_native_account_immutable_outer_container():
    account = build_native_account(_native(), 1234567.89)
    assert isinstance(account.positions, MappingProxyType)
    with pytest.raises(TypeError):
        account.positions[TSCode("X")] = None


def test_native_account_fresh_snapshot_per_call():
    engine = _native()
    first = engine._build_current_account(1000000.0)
    engine.current_capital = 1.0
    second = engine._build_current_account(1000001.0)
    assert first is not second
    assert first.cash.value == 500000.0 and second.cash.value == 1.0


def test_native_weight_roundtrip_lossless_and_ordered():
    engine = _native(positions={})
    weights = {"A": np.float64(0.6), "B": np.float64(0.4)}
    result = engine._merge_target_weights(weights, _D1)
    assert list(result) == ["A", "B"]
    assert result["A"] is weights["A"] and result["B"] is weights["B"]


@pytest.mark.parametrize("missing", ["shares", "buy_date", "buy_cost_cash", "buy_trade_price"])
def test_native_account_missing_critical_fields_fail_fast(missing):
    positions = _positions()
    del positions["A"][missing]
    with pytest.raises(ValueError, match=missing):
        build_native_account(_native(positions), 1234567.89)


@pytest.mark.parametrize("marker", [None, np.nan, pd.NaT, pd.NA])
def test_native_account_missing_markers_fail_fast(marker):
    positions = _positions()
    positions["A"]["buy_cost_cash"] = marker
    with pytest.raises(ValueError, match="buy_cost_cash"):
        build_native_account(_native(positions), 1234567.89)


def test_native_total_value_none_fails_before_merge():
    engine = _native(positions={})
    engine._last_rebalance_nav = None
    with pytest.raises(RuntimeError, match="_last_rebalance_nav"):
        engine._merge_target_weights({"A": 1.0}, _D1)
    assert engine.portfolio_merger.merge_call_count == 0


def test_native_merge_count_real_only():
    engine = _native(positions={})
    engine._merge_target_weights({"A": 1.0}, _D1)
    assert engine.portfolio_merger.merge_call_count == 1
    with pytest.raises(ValueError):
        engine.portfolio_merger.merge({}, None)
    assert engine.portfolio_merger.merge_call_count == 1


def test_native_merge_does_not_mutate_inputs():
    engine = _native(positions={})
    weights = {"A": 0.6, "B": 0.4}
    before = copy.deepcopy(weights)
    positions_before = copy.deepcopy(engine.positions)
    engine._merge_target_weights(weights, _D1)
    assert weights == before and engine.positions == positions_before


def test_native_engine_uses_public_merger_protocol():
    assert isinstance(SingleSleevePortfolioMerger(), PortfolioMerger)
    assert isinstance(_native(), NativeMergeBacktestEngineML)


def test_native_account_never_calls_portfolio_valuation():
    engine = _native()
    engine._calculate_portfolio_value = lambda date: (_ for _ in ()).throw(
        AssertionError("不得再估值")
    )
    before = copy.deepcopy(engine.positions)
    build_native_account(engine, 1234567.89)
    assert engine.positions == before


def test_legacy_merge_slot_and_transition_module_removed():
    engine = _native()
    assert "merge_port" not in vars(engine)
    adapter = Path(native_module.__file__).parent / "backtest" / "merger_adapter.py"
    assert not adapter.exists()


def test_native_seam_order_normalize_merge_cap(monkeypatch):
    engine = _native(positions={})
    engine._normalize_signals = lambda signals, date: {"A": 0.8, "B": 0.2}
    observed = []

    def _non_identity(targets, current):
        target = targets["value"]
        return replace(
            target,
            target_weights=MappingProxyType({TSCode("A"): Weight(0.6), TSCode("B"): Weight(0.4)}),
        )

    engine.portfolio_merger.merge = _non_identity
    monkeypatch.setattr(
        native_module,
        "_cap_normalized_signals",
        lambda self, signals, tranche_idx: observed.append((tranche_idx, dict(signals))) or signals,
    )
    result = engine._normalize_and_cap_signals({"A": 0.9, "B": 0.1}, _D1, 1)
    assert result == {"A": 0.6, "B": 0.4}
    assert observed == [(1, {"A": 0.6, "B": 0.4})]


def test_native_tranches_share_same_native_seam():
    engine = _native(positions={})
    engine._normalize_signals = lambda signals, date: dict(signals)
    for tranche_idx in (0, 1):
        engine._normalize_and_cap_signals({"A": 1.0}, _D1, tranche_idx)
    assert engine.portfolio_merger.merge_call_count == 2


def test_local_refill_and_slot_paths_do_not_enter_native_merge():
    refill = inspect.getsource(BacktestSignalExecutionMixin._queue_condition_sell_refill_signal)
    slots = Path(inspect.getfile(buy_execution)).read_text(encoding="utf-8")
    assert "_merge_target_weights" not in refill
    assert "_merge_target_weights" not in slots


def test_project_trade_orders_preserves_real_sequence_and_targets():
    trades = [
        {"stock": "A", "action": "buy", "shares": 100},
        {"stock": "B", "action": "buy", "shares": 200},
        {"stock": "A", "action": "sell", "shares": 100, "sell_type": "holding_period"},
    ]
    orders = project_t3_evidence_orders(trades)
    assert [order.target_shares.value for order in orders] == [100, 200, 0]
    assert [order.side for order in orders] == [OrderSide.BUY, OrderSide.BUY, OrderSide.SELL]
    assert orders[-1].reason == OrderReason.EXPIRY
    assert {order.source_sleeve for order in orders} == {"value"}


def test_evidence_projection_uses_initial_positions_and_trim_reason():
    trades = [
        {
            "stock": "A",
            "action": "sell",
            "shares": 500,
            "sell_type": "risk_trim",
            "sell_timing": "open",
        }
    ]
    order = project_t3_evidence_orders(trades, initial_positions={"A": 1000})[0]
    assert order.target_shares.value == 500
    assert order.reason == OrderReason.TRIM


def test_evidence_projection_rejects_close_instead_of_faking_open():
    trades = [{"stock": "A", "action": "sell", "shares": 100, "sell_timing": "close"}]
    with pytest.raises(ValueError, match="OrderType 无 close 枚举"):
        project_t3_evidence_orders(trades, initial_positions={"A": 1000})


def test_backtest_replay_calls_real_s6_on_native_orders():
    engine = _native(positions={})
    engine.trades = [{"stock": "A", "action": "buy", "shares": 100, "date": _D1, "price": 10.0}]
    engine.execution_attribution_records = []
    engine.holdings_snapshots = []
    nav = pd.DataFrame({"date": [_D1], "nav": [1.0], "return": [0.0]})
    engine.run = lambda **kwargs: nav
    metrics = _run_replay_engine_and_collect(
        engine, "20260105", "20260106", ["20260105", "20260106"], pd.DataFrame(), 1, 0
    )
    assert metrics["_merger_net_call_count"] == 1
    assert metrics["_native_evidence_order_count"] == 1
    assert metrics["_merger_mode"] == "single_sleeve_passthrough"
    assert "_native_orders" not in metrics


def test_paper_run_t0_returns_committed_orders_without_second_s6():
    host = object.__new__(PaperHost)
    host.portfolio_merger = SingleSleevePortfolioMerger()
    orders = [_order("A"), _order("B", 200)]
    host._correct_trade_date = lambda date: "20260106"
    host._run_daily = lambda requested, corrected, run_kind: PaperDayResult(
        requested, corrected, run_kind, True, "success", [], orders, 1.0
    )
    result = host.run_t0("20260106")
    assert result is orders
    assert host.portfolio_merger.net_call_count == 0


def test_paper_run_t0_empty_orders_does_not_fake_s6_count():
    host = object.__new__(PaperHost)
    host.portfolio_merger = SingleSleevePortfolioMerger()
    host._correct_trade_date = lambda date: "20260106"
    host._run_daily = lambda requested, corrected, run_kind: PaperDayResult(
        requested, corrected, run_kind, True, "no_targets", [], [], 1.0
    )
    assert host.run_t0("20260106") == []
    assert host.portfolio_merger.net_call_count == 0


class _Universe(Universe):
    def get_stocks(self, date, quote_data=None):
        return ["A"]


class _Signal(Signal):
    def __init__(self):
        super().__init__("native_test")
        self.top_n = 1

    def generate(self, date, universe, data):
        return {"A": 1.0}


class _Loader:
    def __init__(self):
        self.calls = 0

    def load_shenwan_industry(self):
        self.calls += 1
        return pd.DataFrame({"ts_code": ["A"], "sw_l2": ["行业甲"]})


def test_paper_engine_wires_industry_reader_and_shared_merger(monkeypatch):
    monkeypatch.setattr(
        "src.lazybull.v2.core.decision.industry_constraint.get_shenwan_level", lambda: "l2"
    )
    host = object.__new__(PaperHost)
    host.universe = _Universe("native")
    host.signal = _Signal()
    host.cost_model = CostModel(**_COST)
    host.stock_basic = pd.DataFrame({"ts_code": ["A"]})
    host.data_storage = object()
    host.loader = _Loader()
    host.verbose = False
    host.schedule_mode = "live"
    host.engine_kwargs = {"max_per_industry": 1}
    host.portfolio_merger = SingleSleevePortfolioMerger()
    engine = host._new_engine({})
    assert isinstance(engine, NativeMergeBacktestEngineML)
    assert engine.portfolio_merger is host.portfolio_merger
    assert engine.industry_mapping == {"A": "行业甲"}
    assert host.loader.calls == 1


def test_backtest_factory_defaults_to_native_and_legacy_is_explicit():
    config = TradingConfig(model_version=1, position_sizing="equal")
    kwargs = dict(
        trading_config=config,
        universe=_Universe("native"),
        signal=_Signal(),
        features_by_date={},
        stock_basic=pd.DataFrame({"ts_code": ["A"]}),
        data_storage=None,
        cost_model=CostModel(**_COST),
    )
    native = create_backtest_engine_from_config(**kwargs)
    legacy = create_backtest_engine_from_config(**kwargs, merger_mode="legacy_bypass")
    assert isinstance(native, NativeMergeBacktestEngineML)
    assert type(legacy) is BacktestEngineML
    assert "portfolio_merger" not in vars(legacy)


@pytest.mark.parametrize("mutation", ["reorder", "renormalize"])
def test_native_weight_mutation_is_observable(mutation):
    engine = _native(positions={})
    original = engine.portfolio_merger.merge

    def _mutating(targets, current):
        target = original(targets, current)
        items = [(code.value, weight.value) for code, weight in target.target_weights.items()]
        if mutation == "reorder":
            items.reverse()
        else:
            total = sum(weight for _, weight in items)
            items = [(code, weight / total) for code, weight in items]
        return replace(
            target,
            target_weights=MappingProxyType(
                {TSCode(code): Weight(weight) for code, weight in items}
            ),
        )

    engine.portfolio_merger.merge = _mutating
    weights = {"A": 0.9, "B": 0.4}
    result = engine._merge_target_weights(weights, _D1)
    if mutation == "reorder":
        assert list(result) != list(weights)
    else:
        assert result != weights


def test_mixed_sleeve_s6_rejection_does_not_increment_count():
    merger = SingleSleevePortfolioMerger()
    with pytest.raises(ValueError, match="混合袖"):
        merger.net_across_sleeves([_order("A", sleeve="value"), _order("B", sleeve="other")])
    assert merger.net_call_count == 0


def test_regression_edge_entrypoints_keep_legacy_bypass():
    from scripts.v2_p2b import replay_merger
    from scripts.v2_p3 import regression_edge_runner

    assert replay_merger.LEGACY_BYPASS == LEGACY_BYPASS == "legacy_bypass"
    assert regression_edge_runner.LEGACY_BYPASS == LEGACY_BYPASS
    assert regression_edge_runner.FROZEN_BATCH == "p2b_t5_closed_20261017_203508"


def test_native_host_uses_nav_single_implementation():
    source = Path(native_module.__file__).read_text(encoding="utf-8")
    assert "calculate_nav(total_value, engine.initial_capital)" in source
    assert "total_value / engine.initial_capital" not in source
