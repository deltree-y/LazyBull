# -*- coding: utf-8 -*-
"""P6.5 T3 订单级全链：真实 Order → 执行 → 物理入账 → 结算 → 虚拟投影。

覆盖：S4 合并买入黄金值（2,500/7,500 → 50/150、费用 (0,5)、2,000/5,995、
7,995/9,995、0.25/0.7495/0.9995）；S1 零物理成交（51,200/48,800、
300/100、0.548/0.500/1.048）；物理事实由执行链产出（成交键 ∈ 账本
applied_keys、成交日志逐值一致）；目标不是增量；重放不重复入账；预演门把
拒绝全部挡在物理执行前（零副作用）；反变异（成交后投影 / 子袖先整手 /
目标当增量 / 单袖超意图）均可打红。

全部为合成输入：价源、费率、现金、持仓与成交上限均显式注入，不读真实配置/行情。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pandas as pd
import pytest

from src.lazybull.universe.base import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.types import OrderReason, OrderSide, OrderType, TradeDate
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLot, PhysicalLotView
from src.lazybull.v2.core.accounting.reconciliation import ReconciliationError
from src.lazybull.v2.core.accounting.settlement_kernel import (
    InternalSettlementBatch,
    PhysicalFillFact,
    SleeveIntent,
    allocate_shares,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    PriceSource,
    VirtualLedger,
    VirtualLedgerError,
    VirtualLot,
)
from src.lazybull.v2.core.decision.virtual_merger import MultiSleevePortfolioMerger
from src.lazybull.v2.core.signal.base import EqualWeightSignal
from src.lazybull.v2.hosts.order_chain import (
    ChainFill,
    OrderChainError,
    OrderChainInvariantError,
    assert_fills_booked,
    assert_plan_integrity,
    assert_settleable,
    build_sleeve_orders,
)
from src.lazybull.v2.hosts.virtual_native_engine import (
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


def _merger() -> MultiSleevePortfolioMerger:
    return MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5})


def _host(
    *,
    ledger: VirtualLedger,
    view: PhysicalLotView,
    capital: float,
    positions: Mapping[str, Any],
    prices: Mapping[str, float],
    initial: float,
    provider: Optional[Any] = None,
) -> VirtualNativeMergeBacktestEngineML:
    host = VirtualNativeMergeBacktestEngineML(
        features_by_date={},
        universe=BasicUniverse(_stock_basic_df(), verbose=False),
        signal=EqualWeightSignal(top_n=5),
        initial_capital=initial,
        cost_model=_cost_model(),
        verbose=False,
        portfolio_merger=_merger(),
        virtual_ledger=ledger,
        physical_lot_view=view,
        price_source_provider=provider,
        run_id="T3R1",
    )
    host.current_capital = capital
    host.positions = dict(positions)
    host._emit_immediate_log = lambda *args, **kwargs: None
    host._prepare_price_index(_price_data(prices))
    return host


def _finalize(host: Any, *, date: pd.Timestamp = _D1, value: float) -> None:
    host._finalize_day(0, date, 0, 1, 1, value, [date], {date: 0})


def _source(price: float, code: str = _X) -> PriceSource:
    return PriceSource(as_of=TradeDate(_D1.date()), source_id="syn:t3", prices={code: price})


# ────────────────────────── S4 全链黄金值 ──────────────────────────


def _s4_host() -> tuple[VirtualNativeMergeBacktestEngineML, VirtualLedger]:
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 2500.0)
    ledger.open_account("B", 7500.0)
    host = _host(
        ledger=ledger,
        view=PhysicalLotView(),
        capital=10000.0,
        positions={},
        prices={_X: 10.0},
        initial=10000.0,
    )
    return host, ledger


def test_s4_full_chain_golden_values_from_execution_path() -> None:
    host, ledger = _s4_host()
    result = host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="S4", fill_capacity={_X: 200}
    )

    # 1) 物理侧：一笔合买卖 200 股 @10、费用 5，现金 10,000 → 7,995
    assert result.applied is True
    orders = result.plan.physical_orders()
    assert len(orders) == 1
    assert orders[0].target_shares.value == 400  # 物理目标是合并目标，不是子袖整手
    assert orders[0].side is OrderSide.BUY
    assert len(result.observed) == 1
    observation = result.observed[0]
    assert (observation.shares, observation.price, observation.fee) == (200, 10.0, 5.0)
    assert (observation.cash_before, observation.cash_after) == (10000.0, 7995.0)
    assert observation.order_target == 400
    assert observation.requested_shares == 200
    assert host.positions[_X]["shares"] == 200
    assert host.current_capital == 7995.0
    assert [record["shares"] for record in host.trades] == [200]

    # 2) 事实来自执行链：账本幂等键 / 成交日志 / 伴随视图三处独立读数交叉一致
    applied = host.ledger.export_state()["applied_keys"]
    assert observation.fill_key in applied
    assert result.batch.physical_fills[0].fill_id in applied
    trade = host.trades[-1]
    assert (trade["stock"], trade["shares"], trade["price"], trade["cost"]) == (
        _X,
        observation.shares,
        observation.price,
        observation.fee,
    )
    assert host.physical_lot_view.shares_of(_X, observation.source_lot_id) == observation.shares
    assert result.batch.physical_cash_before == 10000.0
    assert result.batch.physical_cash_after == 7995.0

    # 3) 虚拟侧：Q18 50/150、Q13 费用 (0,5)、现金 2,000/5,995
    allocations = [
        (item.sleeve_id, item.allocated_shares) for item in result.settlement.share_allocations
    ]
    assert allocations == [("A", 50), ("B", 150)]
    waterfall = result.settlement.fee_waterfalls[0]
    assert waterfall.final_fees == (("A", 0.0), ("B", 5.0))
    assert ledger.state_of("A").cash == 2000.0
    assert ledger.state_of("B").cash == 5995.0
    assert ledger.state_of("A").shares_of(_X) == 50
    assert ledger.state_of("B").shares_of(_X) == 150
    assert ledger.state_of("B").lots[0].acquired_fee == 5.0
    assert result.settlement.cash_total_after == 7995.0
    assert result.settlement.source_shares_after == ((_X, observation.source_lot_id, 200),)
    # 4) 贡献 NAV：0.25 + 0.7495 = 0.9995（分母 = 物理初始资金 10,000）
    source = _source(10.0)
    assert ledger.state_of("A").total_value(source) == 2500.0
    assert ledger.state_of("B").total_value(source) == 7495.0
    assert ledger.state_of("A").total_value(source) / 10000.0 == 0.25
    assert ledger.state_of("B").total_value(source) / 10000.0 == 0.7495
    held = float(host.positions[_X]["shares"])
    assert host.current_capital + held * 10.0 == 9995.0
    assert (host.current_capital + held * 10.0) / 10000.0 == 0.9995

    # 5) 日终挂载：无成交日后置门仍执行三项恒等式
    _finalize(host, value=9995.0)
    report = host.identity_reports[-1]
    assert report.identities == ("I1", "I2", "I3")
    assert report.results[1].right == 200


def test_s4_voided_intents_record_partial_fill() -> None:
    """Q22：合并目标 400 只成交 200 时，未成交增量由结算层登记作废。"""
    host, _ledger = _s4_host()
    result = host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="S4v", fill_capacity={_X: 200}
    )
    assert result.settlement.voided_intents == (("A", _X, 50), ("B", _X, 150))


def test_s4_without_capacity_fills_the_whole_merged_target() -> None:
    host, ledger = _s4_host()
    result = host.execute_order_chain(_D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="S4full")
    assert result.observed[0].shares == 400
    assert host.positions[_X]["shares"] == 400
    assert result.settlement.voided_intents == ()
    assert ledger.state_of("A").shares_of(_X) == 100
    assert ledger.state_of("B").shares_of(_X) == 300


# ────────────────────────── S1 全链黄金值 ──────────────────────────


def _s1_host() -> tuple[VirtualNativeMergeBacktestEngineML, VirtualLedger, PhysicalLotView]:
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 50000.0, [VirtualLot("A:L1", _X, "L1", 400, _D0, 10.0)])
    ledger.open_account("B", 50000.0)
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    host = _host(
        ledger=ledger,
        view=view,
        capital=100000.0,
        positions={_X: _position(400)},
        prices={_X: 12.0},
        initial=100000.0,
    )
    return host, ledger, view


def test_s1_full_chain_golden_values_zero_physical_fill() -> None:
    host, ledger, _view = _s1_host()
    result = host.execute_order_chain(_D1, {"A": {_X: 300}, "B": {_X: 100}}, batch_id="S1")

    # 1) 零物理成交：无市场指令、无成交、现金/持仓逐位不变
    assert result.applied is True
    assert result.plan.physical_orders() == ()
    assert result.observed == ()
    assert result.batch.physical_fills == ()
    assert len(result.batch.transfers) == 1
    assert result.batch.physical_cash_before == 100000.0
    assert result.batch.physical_cash_after == 100000.0
    assert host.current_capital == 100000.0
    assert host.positions[_X]["shares"] == 400
    assert host.trades == []

    # 2) 内部转账完成虚拟侧：A 51,200/300、B 48,800/100（来源谱系 L1）
    assert result.settlement.transfers[0].total_shares == 100
    assert result.settlement.transfers[0].price == 12.0
    assert ledger.state_of("A").cash == 51200.0
    assert ledger.state_of("A").shares_of(_X) == 300
    assert ledger.state_of("B").cash == 48800.0
    assert ledger.state_of("B").shares_of(_X) == 100
    assert ledger.state_of("B").lots[0].source_lot_id == "L1"
    assert ledger.state_of("B").lots[0].acquired_price == 12.0
    assert ledger.state_of("B").lots[0].acquired_fee == 0.0
    assert result.settlement.cash_total_after == 100000.0

    # 3) 贡献 NAV：0.548 + 0.500 = 1.048（分母 = 物理初始资金 100,000）
    source = _source(12.0)
    assert ledger.state_of("A").total_value(source) == 54800.0
    assert ledger.state_of("B").total_value(source) == 50000.0
    assert ledger.state_of("A").total_value(source) / 100000.0 == 0.548
    assert ledger.state_of("B").total_value(source) / 100000.0 == 0.500
    physical_value = host.current_capital + float(host.positions[_X]["shares"]) * 12.0
    assert physical_value == 104800.0
    assert physical_value / 100000.0 == 1.048
    _finalize(host, value=physical_value)
    assert host.identity_reports[-1].identities == ("I1", "I2", "I3")


def test_s1_requires_same_day_price_and_keeps_zero_side_effect() -> None:
    """缺价/跨日价源必须在物理执行与虚拟落账之前拒绝。"""
    host, ledger, _view = _s1_host()

    def broken_provider(date: pd.Timestamp) -> PriceSource:
        return PriceSource(as_of=TradeDate(date.date()), source_id="syn:no-price", prices={})

    host.price_source_provider = broken_provider
    before = dict(ledger.snapshot())
    with pytest.raises(ReconciliationError, match="缺少"):
        host.execute_order_chain(_D1, {"A": {_X: 300}, "B": {_X: 100}}, batch_id="S1np")
    assert dict(ledger.snapshot()) == before
    assert host.current_capital == 100000.0
    assert host.positions[_X]["shares"] == 400
    assert host.trades == []


# ────────────────────────── 目标不是增量 / 全账户基数 ──────────────────────────


def test_single_sleeve_order_uses_full_account_base_and_real_executor() -> None:
    """A 持 400 不发单、B 持 100 目标 300 ⇒ 物理目标 700，真实执行买入 200。"""
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 50000.0, [VirtualLot("A:L1", _X, "L1", 400, _D0, 10.0)])
    ledger.open_account("B", 50000.0, [VirtualLot("B:L2", _X, "L2", 100, _D0, 10.0)])
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400), PhysicalLot(_X, "L2", 100)])
    host = _host(
        ledger=ledger,
        view=view,
        capital=100000.0,
        positions={_X: _position(500)},
        prices={_X: 10.0},
        initial=100000.0,
    )

    result = host.execute_order_chain(_D1, {"A": {_X: 400}, "B": {_X: 300}}, batch_id="S6buy")

    assert result.plan.physical_targets[_X] == 700
    assert result.plan.net_by_code[_X] == 200
    assert len(result.observed) == 1
    # 目标 300 不是增量：实际买入 = 300 − 100 = 200
    assert result.observed[0].requested_shares == 200
    assert result.observed[0].shares == 200
    assert host.positions[_X]["shares"] == 700
    assert host.current_capital == 100000.0 - 2000.0 - 5.0
    assert ledger.state_of("A").shares_of(_X) == 400
    assert ledger.state_of("B").shares_of(_X) == 300
    _finalize(host, value=host.current_capital + 700 * 10.0)
    assert host.identity_reports[-1].identities == ("I1", "I2", "I3")


def test_physical_sell_is_rejected_before_any_market_instruction() -> None:
    """多袖账户单袖卖出的虚拟侧结算未实现（S5 归 P7）：零副作用拒绝。"""
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 50000.0, [VirtualLot("A:L1", _X, "L1", 400, _D0, 10.0)])
    ledger.open_account("B", 50000.0, [VirtualLot("B:L2", _X, "L2", 100, _D0, 10.0)])
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400), PhysicalLot(_X, "L2", 100)])
    host = _host(
        ledger=ledger,
        view=view,
        capital=100000.0,
        positions={_X: _position(500)},
        prices={_X: 10.0},
        initial=100000.0,
    )
    before_states = dict(ledger.snapshot())
    with pytest.raises(OrderChainError, match="未实现"):
        host.execute_order_chain(_D1, {"A": {_X: 400}, "B": {_X: 0}}, batch_id="S5sell")
    assert dict(ledger.snapshot()) == before_states
    assert host.current_capital == 100000.0
    assert host.positions[_X]["shares"] == 500
    assert host.trades == []
    assert host.ledger.export_state()["applied_keys"] == []


def test_stale_virtual_base_is_rejected_before_execution() -> None:
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 50000.0, [VirtualLot("A:L1", _X, "L1", 300, _D0, 10.0)])
    ledger.open_account("B", 50000.0)
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    host = _host(
        ledger=ledger,
        view=view,
        capital=100000.0,
        positions={_X: _position(400)},
        prices={_X: 10.0},
        initial=100000.0,
    )
    with pytest.raises(OrderChainError, match="物理聚合持仓"):
        host.execute_order_chain(_D1, {"A": {_X: 300}, "B": {_X: 100}}, batch_id="STALE")
    assert host.current_capital == 100000.0
    assert host.positions[_X]["shares"] == 400
    assert host.trades == []


def test_unsupported_state_s2_is_rejected_before_execution() -> None:
    """同股反向净残差（S2）不在 T3 范围：合并器零副作用拒绝。"""
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 50000.0, [VirtualLot("A:L1", _X, "L1", 400, _D0, 10.0)])
    ledger.open_account("B", 50000.0)
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    host = _host(
        ledger=ledger,
        view=view,
        capital=100000.0,
        positions={_X: _position(400)},
        prices={_X: 10.0},
        initial=100000.0,
    )
    before = dict(ledger.snapshot())
    with pytest.raises(OrderChainError, match="状态机=S2"):
        host.execute_order_chain(_D1, {"A": {_X: 300}, "B": {_X: 50}}, batch_id="S2")
    assert dict(ledger.snapshot()) == before
    assert host.current_capital == 100000.0
    assert host.trades == []


# ────────────────────────── 幂等与重放 ──────────────────────────


def test_same_batch_id_replay_does_not_double_book() -> None:
    host, ledger = _s4_host()
    first = host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="S4idem", fill_capacity={_X: 200}
    )
    assert first.applied is True
    cash = host.current_capital
    shares = host.positions[_X]["shares"]
    trades = list(host.trades)

    second = host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="S4idem", fill_capacity={_X: 200}
    )
    assert second.applied is False
    assert host.current_capital == cash
    assert host.positions[_X]["shares"] == shares
    assert host.trades == trades
    assert ledger.batch_ids() == ("S4idem",)


def test_settlement_identity_replay_returns_applied_false() -> None:
    host, ledger = _s4_host()
    result = host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="S4rep", fill_capacity={_X: 200}
    )
    again = ledger.apply_settlement(
        result.batch,
        price_source=_source(10.0),
        physical_view=host.physical_lot_view,
        identity_context=host._identity_context(_D1, ledger.snapshot(), "syn:t3"),
    )
    assert again.applied is False
    assert ledger.state_of("A").cash == 2000.0


# ────────────────────────── 预演门（零副作用） ──────────────────────────


def test_over_intent_fill_is_rejected_by_preflight_gate() -> None:
    """超意图成交必须在物理执行前被预演门拒绝（不得先成交再结算失败）。"""
    host, ledger = _s4_host()
    batch = InternalSettlementBatch(
        batch_id="OVER",
        trade_date=TradeDate(_D1.date()),
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _X, "buy", 300, 0),
        ),
        physical_fills=(PhysicalFillFact(_X, "buy", 500, 10.0, 5.0, "L9", "F9"),),
        physical_cash_before=10000.0,
        physical_cash_after=4995.0,
    )
    with pytest.raises(OrderChainError, match="预演结算拒绝"):
        assert_settleable(
            ledger=ledger,
            batch=batch,
            price_source=_source(10.0),
            view=PhysicalLotView(),
            context=host._identity_context(_D1, ledger.snapshot(), "syn:t3"),
            positions={_X: {"shares": 500}},
            batch_id="OVER",
        )
    assert ledger.batch_ids() == ()


def test_sleeve_cash_shortfall_is_rejected_before_execution() -> None:
    """物理现金充足但单袖费用后自有现金不足：预演门拒绝且不产生物理成交。"""
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 100.0)
    ledger.open_account("B", 9900.0)
    host = _host(
        ledger=ledger,
        view=PhysicalLotView(),
        capital=10000.0,
        positions={},
        prices={_X: 10.0},
        initial=10000.0,
    )
    with pytest.raises(OrderChainError, match="预演结算拒绝"):
        host.execute_order_chain(_D1, {"A": {_X: 300}, "B": {_X: 0}}, batch_id="NOCASH")
    assert host.current_capital == 10000.0
    assert host.positions == {}
    assert host.trades == []
    assert ledger.batch_ids() == ()


def test_zero_affordable_fill_is_legal_voided_intent() -> None:
    """物理侧完全买不起时是合法零成交（Q22 日终作废），不得伪造成交。"""
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 100.0)
    ledger.open_account("B", 100.0)
    host = _host(
        ledger=ledger,
        view=PhysicalLotView(),
        capital=200.0,
        positions={},
        prices={_X: 10.0},
        initial=200.0,
    )
    result = host.execute_order_chain(_D1, {"A": {_X: 100}, "B": {_X: 0}}, batch_id="NOFILL")
    assert result.observed == ()
    assert result.settlement.zero_fill is True
    assert result.settlement.voided_intents == (("A", _X, 100),)
    assert host.current_capital == 200.0
    assert host.trades == []


# ────────────────────────── 反变异 ──────────────────────────


def test_fabricated_evidence_projection_is_rejected() -> None:
    """成交后证据投影（成交键不在账本）必须被拒绝。"""
    host, _ledger = _s4_host()
    host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="S4f", fill_capacity={_X: 200}
    )
    fake = ChainFill(
        ts_code=_X,
        order_target=400,
        requested_shares=200,
        shares=200,
        price=10.0,
        fee=5.0,
        cash_before=10000.0,
        cash_after=7995.0,
        fill_key="20260106|buy|600000.SH|999",
        source_lot_id="PHY:fake",
    )
    with pytest.raises(OrderChainInvariantError, match="成交后证据投影"):
        assert_fills_booked(host, [fake], batch_id="S4f", date=_D1)


def test_observation_cash_and_date_consistency_are_enforced() -> None:
    """成交日志/现金增量/日期必须同源自洽（防止"日志一套、现金另一套"）。"""
    from src.lazybull.v2.core.accounting.ledger import BookingEntry
    from src.lazybull.v2.hosts.order_chain import _observation_from_entry

    predicted = ChainFill(_X, 400, 200, 200, 10.0, 5.0, 10000.0, 7995.0, "K1", "PHY:K1")
    base = {
        "date": _D1,
        "stock": _X,
        "action": "buy",
        "price": 10.0,
        "shares": 200,
        "amount": 2000.0,
        "cost": 5.0,
    }
    entry = BookingEntry(
        fill_key="K1",
        date=_D1,
        stock=_X,
        action="buy",
        cash_delta=-2005.0,
        trade_record=base,
    )
    observation = _observation_from_entry(
        entry, 10000.0, 7995.0, predicted, date=_D1, batch_id="OBS"
    )
    assert observation.shares == 200 and observation.fee == 5.0

    broken_cash = BookingEntry(
        fill_key="K1",
        date=_D1,
        stock=_X,
        action="buy",
        cash_delta=-1000.0,
        trade_record=base,
    )
    with pytest.raises(OrderChainInvariantError, match="同源自洽"):
        _observation_from_entry(broken_cash, 10000.0, 9000.0, predicted, date=_D1, batch_id="OBS")

    broken_date = BookingEntry(
        fill_key="K1",
        date=_D1,
        stock=_X,
        action="buy",
        cash_delta=-2005.0,
        trade_record={**base, "date": _D2},
    )
    with pytest.raises(OrderChainInvariantError, match="成交日志日期"):
        _observation_from_entry(broken_date, 10000.0, 7995.0, predicted, date=_D1, batch_id="OBS")


def test_two_physical_fills_for_same_code_cannot_settle() -> None:
    """子袖先各自整手（100+100）会产生同股两笔成交：批次结构直接拒绝。"""
    with pytest.raises(VirtualLedgerError, match="同一股票多笔物理成交"):
        InternalSettlementBatch(
            batch_id="SPLITLOT",
            trade_date=TradeDate(_D1.date()),
            intents=(
                SleeveIntent("A", _X, "buy", 100, 0),
                SleeveIntent("B", _X, "buy", 300, 0),
            ),
            physical_fills=(
                PhysicalFillFact(_X, "buy", 100, 10.0, 5.0, "L9", "F1"),
                PhysicalFillFact(_X, "buy", 100, 10.0, 5.0, "L10", "F2"),
            ),
        )


def test_allocations_never_exceed_each_sleeve_intent() -> None:
    items = allocate_shares({"A": 100, "B": 300}, 200)
    assert [(item.sleeve_id, item.allocated_shares) for item in items] == [("A", 50), ("B", 150)]
    for item in items:
        assert 0 <= item.allocated_shares <= item.intent_shares
    with pytest.raises(VirtualLedgerError, match="超意图成交"):
        allocate_shares({"A": 100, "B": 300}, 401)


def test_per_sleeve_lot_rounding_plan_is_rejected() -> None:
    """子袖先各自整手（100+100=200）与意图之和不符：计划自洽门必须打红。"""
    from dataclasses import replace as dataclass_replace

    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 2500.0)
    ledger.open_account("B", 7500.0)
    orders = build_sleeve_orders(targets={"A": {_X: 100}, "B": {_X: 300}}, ledger=ledger, date=_D1)
    plan = _merger().plan_netting(orders, ledger.as_virtual_accounts(_source(10.0)))
    assert plan.physical_targets[_X] == 400
    assert plan.net_by_code[_X] == 400

    doctored = dataclass_replace(plan, net_by_code={_X: 200}, physical_targets={_X: 200})
    with pytest.raises(OrderChainInvariantError, match="计划自相矛盾"):
        assert_plan_integrity(doctored, batch_id="MUT", date=_D1)

    # 目标当增量的变异：目标 200 但基数为 0、净增 200 ⇒ 自洽；但目标写成 500 则红
    drift = dataclass_replace(plan, physical_targets={_X: 500})
    with pytest.raises(OrderChainInvariantError, match="物理目标"):
        assert_plan_integrity(drift, batch_id="MUT2", date=_D1)

    # 丢股变异：目标/净额/指令全部清空，意图仍在 ⇒ 不得被当成"合法零成交"
    dropped = dataclass_replace(plan, net_by_code={}, physical_targets={}, orders=())
    with pytest.raises(OrderChainInvariantError, match="计划漏股"):
        assert_plan_integrity(dropped, batch_id="MUT3", date=_D1)

    # 丢单变异：目标/净额保留但物理指令缺失 ⇒ 净增量为正必须有指令
    no_order = dataclass_replace(plan, orders=())
    with pytest.raises(OrderChainInvariantError, match="缺物理指令"):
        assert_plan_integrity(no_order, batch_id="MUT4", date=_D1)


def test_build_sleeve_orders_keeps_target_semantics() -> None:
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 5000.0, [VirtualLot("A:L1", _X, "L1", 400, _D0, 10.0)])
    ledger.open_account("B", 5000.0)
    orders = build_sleeve_orders(targets={"A": {_X: 300}, "B": {_X: 100}}, ledger=ledger, date=_D1)
    by_sleeve = {order.source_sleeve: order for order in orders}
    assert by_sleeve["A"].side is OrderSide.SELL
    assert by_sleeve["A"].target_shares.value == 300
    assert by_sleeve["B"].side is OrderSide.BUY
    assert by_sleeve["B"].target_shares.value == 100
    assert by_sleeve["A"].order_type is OrderType.MARKET_OPEN
    assert by_sleeve["A"].reason is OrderReason.REBALANCE


def test_order_chain_result_records_plan_and_batch_identity() -> None:
    host, _ledger = _s4_host()
    result = host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="S4ev", fill_capacity={_X: 200}
    )
    assert result.batch_id == "S4ev"
    assert result.trade_date == TradeDate(_D1.date())
    assert result.plan.intents[0].current_shares == 0
    assert result.plan.intents[1].target_shares == 300
    assert result.physical_cash_before == 10000.0
    assert result.physical_cash_after == 7995.0
    assert host.order_chain_result("S4ev") is result


def test_two_decimal_prices_settle_without_false_cash_breach() -> None:
    """外部评审 P1：正常两位小数价格不得因现金反向相减误判为链路破缺。"""
    host, ledger = _s4_host()
    host._prepare_price_index(_price_data({_X: 20.01}))
    result = host.execute_order_chain(
        _D1, {"A": {_X: 100}, "B": {_X: 300}}, batch_id="FLOAT1", fill_capacity={_X: 200}
    )
    assert result.applied is True
    expected = 10000.0 - (200 * 20.01 + result.observed[0].fee)
    assert host.current_capital == expected
    assert host.positions[_X]["shares"] == 200
    states = (ledger.state_of("A"), ledger.state_of("B"))
    assert abs(sum(state.cash for state in states) - expected) <= 1e-6
    assert sum(lot.shares for state in states for lot in state.lots) == 200

    ledger2 = VirtualLedger(cost_model=_cost_model())
    ledger2.open_account("A", 50000.0)
    ledger2.open_account("B", 50000.0)
    host2 = _host(
        ledger=ledger2,
        view=PhysicalLotView(),
        capital=100000.0,
        positions={},
        prices={_X: 12.34},
        initial=100000.0,
    )
    result2 = host2.execute_order_chain(
        _D1, {"A": {_X: 1000}, "B": {_X: 1000}}, batch_id="FLOAT2", fill_capacity={_X: 2000}
    )
    assert result2.applied is True
    expected2 = 100000.0 - (2000 * 12.34 + result2.observed[0].fee)
    assert host2.current_capital == expected2
    assert host2.positions[_X]["shares"] == 2000
    states2 = (ledger2.state_of("A"), ledger2.state_of("B"))
    assert abs(sum(state.cash for state in states2) - expected2) <= 1e-6
    assert sum(lot.shares for state in states2 for lot in state.lots) == 2000
