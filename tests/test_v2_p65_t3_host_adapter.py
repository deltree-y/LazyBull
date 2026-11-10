# -*- coding: utf-8 -*-
"""P6.5 T3 宿主承载与执行适配：父类/run_loop 零改动、默认路径零影响、终态语义。

覆盖：宿主同时承载全链与逐日挂载；父类与 ``run_loop.py`` 无链方法且
``_finalize_day`` 归属不变；执行侧受控注入点在没有观察者时与直调
``Ledger.apply_fills`` 逐位等价（现金/持仓/成交日志/调用次数）；观察者只在
执行期间存在、异常路径也会复原；终态实例拒绝新链；链证据不进入
``EngineState`` 键集；无成交日仍执行 I1/I2/I3。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pandas as pd
import pytest

from src.lazybull.universe.base import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.core.accounting.engine_state import export_state, state_to_blob
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLot, PhysicalLotView
from src.lazybull.v2.core.accounting.reconciliation import ReconciliationError
from src.lazybull.v2.core.accounting.virtual_ledger import (
    PriceSource,
    VirtualLedger,
    VirtualLot,
)
from src.lazybull.v2.core.decision.virtual_merger import MultiSleevePortfolioMerger
from src.lazybull.v2.core.execution.engine import BacktestEngine
from src.lazybull.v2.core.execution.fill_observer import apply_booking_entries
from src.lazybull.v2.core.execution.run_loop import BacktestRunLoopMixin
from src.lazybull.v2.core.signal.base import EqualWeightSignal
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML
from src.lazybull.v2.hosts.order_chain import OrderChainError, OrderChainInvariantError
from src.lazybull.v2.hosts.virtual_native_engine import (
    ChainFailedTerminatedError,
    IdentityTerminatedError,
    VirtualNativeMergeBacktestEngineML,
)

_D0 = TradeDate(pd.Timestamp("2026-01-05").date())
_D1 = pd.Timestamp("2026-01-06")
_D2 = pd.Timestamp("2026-01-07")
_X = "600000.SH"
_Y = "000001.SZ"


def _cost_model() -> CostModel:
    return CostModel(commission_rate=0.0003, min_commission=5.0, stamp_tax=0.0005, slippage=0.0)


def _stock_basic_df() -> pd.DataFrame:
    return pd.DataFrame({"ts_code": [_X, _Y], "name": ["甲", "乙"], "industry": ["I1", "I2"]})


def _price_data(prices: Mapping[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": list(prices),
            "trade_date": [_D1] * len(prices),
            "close": [float(value) for value in prices.values()],
        }
    )


def _position(shares: int, price: float = 10.0) -> Dict[str, Any]:
    return {
        "shares": shares,
        "buy_date": pd.Timestamp("2025-12-31"),
        "buy_cost_cash": shares * price,
        "buy_trade_price": price,
    }


def _base_kwargs(*, merger: Optional[Any] = None, initial: float = 100000.0) -> Dict[str, Any]:
    return {
        "features_by_date": {},
        "universe": BasicUniverse(_stock_basic_df(), verbose=False),
        "signal": EqualWeightSignal(top_n=5),
        "initial_capital": initial,
        "cost_model": _cost_model(),
        "verbose": False,
        "portfolio_merger": merger or MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5}),
    }


def _new_virtual_host(
    *,
    prices: Mapping[str, float],
    capital: float = 10000.0,
    positions: Optional[Mapping[str, Any]] = None,
    ledger: Optional[VirtualLedger] = None,
    view: Optional[PhysicalLotView] = None,
    provider: Optional[Any] = None,
    initial: float = 10000.0,
    min_buy_ratio: float = 0.0,
) -> VirtualNativeMergeBacktestEngineML:
    host = VirtualNativeMergeBacktestEngineML(
        **_base_kwargs(initial=initial),
        virtual_ledger=ledger or _empty_ledger(),
        physical_lot_view=view or PhysicalLotView(),
        price_source_provider=provider,
        min_buy_value_ratio=min_buy_ratio,
        run_id="T3H",
    )
    host.current_capital = capital
    host.positions = dict(positions or {})
    host._emit_immediate_log = lambda *args, **kwargs: None
    host._prepare_price_index(_price_data(prices))
    return host


def _empty_ledger() -> VirtualLedger:
    """两袖空仓账（现金合计 = 物理 10,000；S4 合成费率口径）。"""
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 2500.0)
    ledger.open_account("B", 7500.0)
    return ledger


def _finalize(host: Any, *, date: pd.Timestamp = _D1, value: float) -> None:
    host._finalize_day(0, date, 0, 1, 1, value, [date], {date: 0})


def _plain_engine(prices: Mapping[str, float], capital: float) -> NativeMergeBacktestEngineML:
    from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger

    engine = NativeMergeBacktestEngineML(**_base_kwargs(merger=SingleSleevePortfolioMerger()))
    engine.current_capital = capital
    engine.positions = {}
    engine._emit_immediate_log = lambda *args, **kwargs: None
    engine._prepare_price_index(_price_data(prices))
    return engine


def test_host_carries_chain_and_gate_for_the_same_day() -> None:
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    result = host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H1", fill_capacity={_X: 200}
    )
    _finalize(host, value=9995.0)

    assert result.applied is True
    assert host.identity_check_count == 1
    report = host.identity_reports[-1]
    assert report.identities == ("I1", "I2", "I3")
    assert report.last_settlement_id == "H1"
    assert len(host.portfolio_values) == 1


def test_chain_evidence_never_enters_engine_state_key_set() -> None:
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    before = set(state_to_blob(export_state(host))["fields"])
    host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H2", fill_capacity={_X: 200}
    )
    after = set(state_to_blob(export_state(host))["fields"])
    assert after == before
    assert not any("chain" in name or "identity" in name for name in after)


def test_terminated_host_rejects_new_chain_without_side_effects() -> None:
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 51000.0)  # 首日 I1 已破缺
    ledger.open_account("B", 50000.0)
    host = _new_virtual_host(prices={_X: 10.0}, capital=100000.0, ledger=ledger)
    with pytest.raises(ReconciliationError):
        _finalize(host, value=100000.0)
    assert host.is_terminated

    cash = host.current_capital
    with pytest.raises(IdentityTerminatedError):
        host.execute_order_chain(_D1, {"A": {_X: 100}, "B": {_X: 0}}, batch_id="H3")
    assert host.current_capital == cash
    assert host.trades == []


def test_no_trade_day_chain_still_runs_identities() -> None:
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 50000.0, [VirtualLot("A:L1", _X, "L1", 400, _D0, 10.0)])
    ledger.open_account("B", 50000.0)
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    host = _new_virtual_host(
        prices={_X: 10.0},
        capital=100000.0,
        positions={_X: _position(400)},
        ledger=ledger,
        view=view,
    )
    result = host.execute_order_chain(_D1, {"A": {_X: 400}, "B": {_X: 0}}, batch_id="H4")
    assert result.plan.physical_orders() == ()
    assert result.settlement.zero_fill is False
    _finalize(host, value=104000.0)
    assert host.identity_reports[-1].identities == ("I1", "I2", "I3")
    assert host.trades == []


def test_parent_and_run_loop_have_no_chain_api() -> None:
    assert not hasattr(NativeMergeBacktestEngineML, "execute_order_chain")
    assert not hasattr(BacktestRunLoopMixin, "execute_order_chain")
    assert not hasattr(BacktestEngine, "execute_order_chain")
    assert NativeMergeBacktestEngineML._finalize_day is BacktestRunLoopMixin._finalize_day
    assert VirtualNativeMergeBacktestEngineML._finalize_day is not (
        BacktestRunLoopMixin._finalize_day
    )
    assert issubclass(VirtualNativeMergeBacktestEngineML, NativeMergeBacktestEngineML)


def test_default_buy_path_is_identical_without_observer() -> None:
    """无观察者的买入与带观察者的买入必须逐位一致（现金/持仓/成交日志）。"""
    plain = _plain_engine({_X: 10.0}, capital=10000.0)
    watched = _plain_engine({_X: 10.0}, capital=10000.0)
    events = []

    def observer(entry, cash_before, cash_after):
        events.append((entry.fill_key, cash_before, cash_after))

    watched.physical_fill_observer = observer
    for engine in (plain, watched):
        engine._buy_stock_direct(_D1, _X, 2000.0, signal_date=_D1, requested_shares=200)

    assert plain.current_capital == watched.current_capital == 7995.0
    assert plain.positions == watched.positions
    assert [dict(item) for item in plain.trades] == [dict(item) for item in watched.trades]
    assert len(events) == 1
    assert events[0][1] == 10000.0 and events[0][2] == 7995.0


def test_booking_entries_equals_direct_ledger_call() -> None:
    """无观察者时封装与 ``ledger.apply_fills`` 直调在账户上逐位等价。"""
    from src.lazybull.v2.core.accounting.ledger import BookingEntry

    wrapped = _plain_engine({_X: 10.0}, capital=10000.0)
    direct = _plain_engine({_X: 10.0}, capital=10000.0)
    entry = BookingEntry(
        fill_key="20260106|buy|600000.SH|0",
        date=_D1,
        stock=_X,
        action="buy",
        cash_delta=-2005.0,
        trade_record={
            "date": _D1,
            "stock": _X,
            "action": "buy",
            "price": 10.0,
            "shares": 200,
            "amount": 2000.0,
            "cost": 5.0,
        },
        new_position={
            "shares": 200,
            "buy_date": _D1,
            "signal_date": _D1,
            "buy_trade_price": 10.0,
            "buy_pnl_price": 10.0,
            "buy_cost_cash": 2005.0,
        },
    )
    apply_booking_entries(wrapped, [entry])
    direct.ledger.apply_fills(direct, [entry])
    assert wrapped.current_capital == direct.current_capital == 7995.0
    assert wrapped.positions == direct.positions
    assert wrapped.trades == direct.trades


def test_observer_rejects_multi_entry_batch() -> None:
    engine = _plain_engine({_X: 10.0}, capital=10000.0)
    engine.physical_fill_observer = lambda *args: None
    entry = object()
    with pytest.raises(ValueError, match="只支持单笔入账批次"):
        apply_booking_entries(engine, [entry, entry])


def test_chain_restores_previous_observer_attribute() -> None:
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    sentinel = object()
    host.physical_fill_observer = sentinel
    real_buy = host._buy_stock_direct

    def boom(*args, **kwargs):
        real_buy(*args, **kwargs)
        raise RuntimeError("执行器中途抛出：注入点必须在异常路径复原")

    host._buy_stock_direct = boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="注入点必须在异常路径复原"):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H5", fill_capacity={_X: 200}
        )
    assert host.physical_fill_observer is sentinel

    fresh = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, initial=10000.0)
    fresh.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H6", fill_capacity={_X: 200}
    )
    assert not hasattr(fresh, "physical_fill_observer")


def test_one_apply_fills_call_per_physical_fill() -> None:
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    calls = []
    real_apply = host.ledger.apply_fills

    def spy(engine, fills):
        calls.append(list(fills))
        return real_apply(engine, fills)

    host.ledger.apply_fills = spy  # type: ignore[assignment]
    host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H7", fill_capacity={_X: 200}
    )
    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert calls[0][0].action == "buy"


def test_explicit_price_source_must_match_physical_and_start_no_trade() -> None:
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    wrong = PriceSource(as_of=TradeDate(_D1.date()), source_id="syn:wrong", prices={_X: 11.0})
    with pytest.raises(ReconciliationError, match="不同源"):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 0}}, batch_id="H8", price_source=wrong
        )
    assert host.trades == []
    assert host.current_capital == 10000.0


def test_chain_rejects_non_multi_sleeve_merger() -> None:
    from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger

    ledger = _empty_ledger()
    host = VirtualNativeMergeBacktestEngineML(
        **_base_kwargs(merger=SingleSleevePortfolioMerger(), initial=10000.0),
        virtual_ledger=ledger,
        physical_lot_view=PhysicalLotView(),
        run_id="T3H",
    )
    host.current_capital = 10000.0
    host.positions = {}
    host._emit_immediate_log = lambda *args, **kwargs: None
    host._prepare_price_index(_price_data({_X: 10.0}))
    with pytest.raises(OrderChainError, match="plan_netting"):
        host.execute_order_chain(_D1, {"A": {_X: 100}, "B": {_X: 0}}, batch_id="H9")
    assert host.trades == []


def test_same_batch_id_with_different_request_is_rejected() -> None:
    """同一批次键只能承载同一请求：异目标/异上限必须拒绝（不得静默 applied=False）。"""
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H11", fill_capacity={_X: 200}
    )
    cash = host.current_capital
    with pytest.raises(OrderChainError, match="另一套请求"):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H11", fill_capacity={_X: 100}
        )
    assert host.current_capital == cash


def test_chain_batch_id_cannot_reuse_existing_settlement() -> None:
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H10", fill_capacity={_X: 200}
    )
    del host._chain_results["H10"]
    with pytest.raises(OrderChainError, match="链级身份不得复用"):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H10", fill_capacity={_X: 200}
        )


def _s1_state_host() -> tuple[VirtualNativeMergeBacktestEngineML, VirtualLedger]:
    """A 持 400（L1）、B 空；物理 400 @12。"""
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 50000.0, [VirtualLot("A:L1", _X, "L1", 400, _D0, 10.0)])
    ledger.open_account("B", 50000.0)
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    host = _new_virtual_host(
        prices={_X: 12.0},
        capital=100000.0,
        positions={_X: _position(400)},
        ledger=ledger,
        view=view,
        initial=100000.0,
    )
    return host, ledger


def _transfer(ledger: VirtualLedger, settlement_id: str, shares: int) -> None:
    from src.lazybull.v2.core.accounting.virtual_ledger import InternalTransfer, LotAllocation

    ledger.transfer(
        InternalTransfer(
            settlement_id=settlement_id,
            trade_date=TradeDate(_D1.date()),
            ts_code=_X,
            from_sleeve="A",
            to_sleeve="B",
            price=12.0,
            allocations=(LotAllocation("A:L1", shares),),
        ),
        price_source=PriceSource(
            as_of=TradeDate(_D1.date()), source_id="syn:t3", prices={_X: 12.0}
        ),
    )


def test_poisoned_virtual_ledger_rejects_chain_before_execution() -> None:
    """账本终态必须前置拒绝：绝不允许"物理已成交、随后才拒绝"。"""
    from src.lazybull.v2.core.accounting.virtual_ledger import PhysicalFillFact, SleeveIntent
    from src.lazybull.v2.core.accounting.settlement_kernel import InternalSettlementBatch

    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    bad = InternalSettlementBatch(
        batch_id="POISON",
        trade_date=TradeDate(_D1.date()),
        intents=(SleeveIntent("A", _X, "buy", 100, 0),),
        physical_fills=(PhysicalFillFact(_X, "buy", 100, 10.0, 5.0, "L9", "F9"),),
        physical_cash_before=10000.0,
        physical_cash_after=8995.0,
    )
    with pytest.raises(ReconciliationError):
        ledger.apply_settlement(
            bad,
            price_source=PriceSource(
                as_of=TradeDate(_D1.date()), source_id="syn:t3", prices={_X: 10.0}
            ),
            physical_view=PhysicalLotView(),  # 缺物理 lot ⇒ I2 破缺 ⇒ 终态
            identity_context=host._identity_context(_D1, ledger.snapshot(), "syn:t3"),
        )
    assert ledger.poisoned_reason is not None

    cash = host.current_capital
    with pytest.raises(OrderChainError, match="进入终态"):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="POISON2", fill_capacity={_X: 200}
        )
    assert host.current_capital == cash
    assert host.trades == []


def test_chain_batch_id_conflicting_with_internal_transfer_is_rejected() -> None:
    host, ledger = _s1_state_host()
    _transfer(ledger, "S1", 100)  # 虚拟 A 300 / B 100，物理不变
    cash = host.current_capital
    with pytest.raises(OrderChainError, match="内部转账/派生键占用"):
        host.execute_order_chain(_D1, {"A": {_X: 300}, "B": {_X: 200}}, batch_id="S1")
    assert host.current_capital == cash
    assert host.positions[_X]["shares"] == 400
    assert host.trades == []


def test_derived_transfer_subkey_conflict_is_caught_by_preflight() -> None:
    """派生键占用必须由预演（镜像幂等身份）发现，而不是成交后才拒绝。"""
    host, ledger = _s1_state_host()
    _transfer(ledger, "S1#T0", 10)  # 占用 S1 的派生键；虚拟 A 390 / B 10
    cash = host.current_capital
    with pytest.raises(OrderChainError, match="派生转账子键"):
        host.execute_order_chain(_D1, {"A": {_X: 290}, "B": {_X: 110}}, batch_id="S1")
    assert host.current_capital == cash
    assert host.positions[_X]["shares"] == 400
    assert host.trades == []


def test_min_buy_value_ratio_must_be_zero_for_order_chain() -> None:
    ledger = _empty_ledger()
    host = _new_virtual_host(
        prices={_X: 10.0},
        capital=10000.0,
        ledger=ledger,
        initial=10000.0,
        min_buy_ratio=0.2,
    )
    with pytest.raises(OrderChainError, match="min_buy_value_ratio=0"):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="RATIO", fill_capacity={_X: 200}
        )
    assert host.trades == []
    assert host.current_capital == 10000.0


def test_same_batch_id_with_different_price_map_is_rejected() -> None:
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    source = PriceSource(as_of=TradeDate(_D1.date()), source_id="syn:t3", prices={_X: 10.0})
    host.execute_order_chain(
        _D1,
        {"A": {_X: 100}, "B": {_X: 300}},
        batch_id="PRICE",
        price_source=source,
        fill_capacity={_X: 200},
    )
    other = PriceSource(as_of=TradeDate(_D1.date()), source_id="syn:t3", prices={_X: 11.0})
    with pytest.raises(OrderChainError, match="另一套请求"):
        host.execute_order_chain(
            _D1,
            {"A": {_X: 100}, "B": {_X: 300}},
            batch_id="PRICE",
            price_source=other,
            fill_capacity={_X: 200},
        )

    # 同 source_id / 同价目但不同 as_of：同样是异请求（不得被当成重放）
    shifted = PriceSource(as_of=TradeDate(_D2.date()), source_id="syn:t3", prices={_X: 10.0})
    with pytest.raises(OrderChainError, match="另一套请求"):
        host.execute_order_chain(
            _D1,
            {"A": {_X: 100}, "B": {_X: 300}},
            batch_id="PRICE",
            price_source=shifted,
            fill_capacity={_X: 200},
        )


def test_real_settlement_failure_is_invariant_and_blocks_retry() -> None:
    """预演通过/真实拒绝属实现缺陷：必须抛 Invariant 并登记，禁止同键重试。"""
    from src.lazybull.v2.core.accounting.virtual_ledger import VirtualLedgerError

    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)

    def boom(*args, **kwargs):
        raise VirtualLedgerError("模拟真实结算拒绝")

    host.virtual_ledger.apply_settlement = boom  # type: ignore[assignment]
    with pytest.raises(OrderChainInvariantError, match="预演通过但真实结算拒绝"):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H12", fill_capacity={_X: 200}
        )
    assert "H12" in host._chain_failures
    assert host.is_terminated is True
    cash = host.current_capital
    with pytest.raises(ChainFailedTerminatedError):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H12", fill_capacity={_X: 200}
        )
    assert host.current_capital == cash


def test_mid_execution_exception_blocks_same_key_retry() -> None:
    """执行链中途抛任意异常：必须转 Invariant、保持占用并禁止同键重试。"""
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    real_buy = host._buy_stock_direct

    def boom(*args, **kwargs):
        real_buy(*args, **kwargs)
        raise RuntimeError("执行器中途抛出（非枚举异常类型）")

    host._buy_stock_direct = boom  # type: ignore[assignment]
    with pytest.raises(OrderChainInvariantError, match="执行链中途异常"):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H13", fill_capacity={_X: 200}
        )
    assert "H13" in host._chain_failures
    assert host.is_terminated is True
    cash = host.current_capital
    trades = list(host.trades)
    with pytest.raises(ChainFailedTerminatedError):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="H13", fill_capacity={_X: 200}
        )
    assert host.current_capital == cash
    assert host.trades == trades


def test_non_integer_targets_are_rejected_with_batch_context() -> None:
    """目标股数在指纹/订单构造前严格校验：None/float 不得先 int() 再漂移。"""
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    with pytest.raises(OrderChainError, match="批次=BAD 日期=2026-01-06"):
        host.execute_order_chain(_D1, {"A": {_X: None}, "B": {_X: 0}}, batch_id="BAD")
    with pytest.raises(OrderChainError, match="必须为整数股"):
        host.execute_order_chain(_D1, {"A": {_X: 100.0}, "B": {_X: 0}}, batch_id="BAD2")
    assert host.current_capital == 10000.0
    assert host.trades == []


def test_target_sleeve_set_mismatch_is_rejected_before_execution() -> None:
    ledger = _empty_ledger()
    host = _new_virtual_host(prices={_X: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0)
    with pytest.raises(OrderChainError, match="目标袖集合"):
        host.execute_order_chain(_D1, {"A": {_X: 100}, "C": {_X: 100}}, batch_id="IDSET")
    assert host.current_capital == 10000.0
    assert host.trades == []
    assert ledger.batch_ids() == ()


def test_real_settlement_breach_locks_terminal_state() -> None:
    """外部评审 P2：真实结算破缺后宿主锁定终态，后续日步进不得新增成交。"""
    ledger = _empty_ledger()
    host = _new_virtual_host(
        prices={_X: 10.0, _Y: 10.0}, capital=10000.0, ledger=ledger, initial=10000.0
    )
    real_settlement = host.virtual_ledger.apply_settlement

    def broken(*args: Any, **kwargs: Any) -> Any:
        kwargs["physical_view"] = PhysicalLotView()
        return real_settlement(*args, **kwargs)

    host.virtual_ledger.apply_settlement = broken  # type: ignore[method-assign]
    with pytest.raises(OrderChainInvariantError):
        host.execute_order_chain(
            _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="TERM", fill_capacity={_X: 200}
        )
    assert host.is_terminated is True
    cash = host.current_capital
    trades = list(host.trades)
    prices = pd.DataFrame(
        {
            "ts_code": [_X, _Y, _X, _Y],
            "trade_date": ["20260106", "20260106", "20260107", "20260107"],
            "close": [10.0, 10.0, 10.0, 10.0],
            "vol": [10000, 10000, 10000, 10000],
            "is_suspended": [False, False, False, False],
            "is_limit_up": [False, False, False, False],
        }
    )
    host._prepare_price_index(prices)
    host.price_data_cache = prices
    host._trade_date_index = {_D1: 0, _D2: 1}
    host._signal_dates = {}
    host._run_total_days = 2
    host._cycle_anchor_idx = 0
    host.pending_signals[_D1] = {"signals": {_Y: 0.5}, "desired_position_count": 2}
    with pytest.raises(ChainFailedTerminatedError):
        host.run_step(_D2, [_D1, _D2], prices)
    assert host.current_capital == cash
    assert host.trades == trades
    assert _Y not in host.positions
    with pytest.raises(ChainFailedTerminatedError):
        host.execute_order_chain(_D1, {"A": {_X: 0}, "B": {_X: 0}}, batch_id="TERM2")
