# -*- coding: utf-8 -*-
"""P6.5 T2 内部结算批次：S1/S4 落账、FIFO、零成交短路、幂等与结算后 I1/I2 回验。

契约依据：``docs/contracts/netting_freeze.md`` F25 §4/§5、设计 §4.3/§7.2/§8.2/§8.3。
全部输入为合成值：价源、成交、费用与物理现金都是显式注入的事实，不读真实
行情/配置；唯一成交入账口仍是 T3 的生产 ``Ledger.apply_fills``。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.core.accounting.physical_lot_view import (
    PhysicalLot,
    PhysicalLotView,
    PhysicalLotViewError,
)
from src.lazybull.v2.core.accounting.reconciliation import (
    IdentityContext,
    ReconciliationError,
    check_settlement_identities,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    InternalSettlementBatch,
    InternalTransfer,
    LotAllocation,
    PhysicalFillFact,
    PriceSource,
    SleeveIntent,
    TransferFact,
    VirtualLedger,
    VirtualLedgerConservationError,
    VirtualLedgerError,
    VirtualLot,
)

_X = "600000.SH"
_Y = "000001.SZ"
_T0 = TradeDate(pd.Timestamp("2026-01-05").date())
_T1 = TradeDate(pd.Timestamp("2026-01-06").date())
_T2 = TradeDate(pd.Timestamp("2026-01-07").date())


def _lot(
    lot_id: str = "A:L1",
    *,
    shares: int = 400,
    source: str = "L1",
    date=_T0,
    price: float = 10.0,
    fee: float = 0.0,
) -> VirtualLot:
    return VirtualLot(lot_id, _X, source, shares, date, price, fee)


def _context(date=_T1, sleeves=("A", "B")) -> IdentityContext:
    return IdentityContext(
        trade_date=date,
        host="pytest_settlement",
        run_id="R1",
        sleeve_ids=tuple(sleeves),
        price_source_id="syn:p65",
    )


def _source(price: float = 12.0, date=_T1) -> PriceSource:
    return PriceSource(as_of=date, source_id="syn:p65", prices={_X: price})


def _s1_ledger() -> VirtualLedger:
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [_lot()])
    ledger.open_account("B", 50000.0)
    return ledger


def _s1_batch(shares: int = 100, cash_after: float = 100000.0) -> InternalSettlementBatch:
    return InternalSettlementBatch(
        batch_id="S1",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 400 - shares, 400),
            SleeveIntent("B", _X, "buy", shares, 0),
        ),
        transfers=(TransferFact(_X, "A", "B", shares, 12.0),),
        physical_cash_before=100000.0,
        physical_cash_after=cash_after,
    )


def _s1_view() -> PhysicalLotView:
    return PhysicalLotView([PhysicalLot(_X, "L1", 400)])


# ────────────────────────── S1 结算黄金值 ──────────────────────────


def test_s1_settlement_moves_cash_and_lots_internally():
    ledger = _s1_ledger()
    result = ledger.apply_settlement(
        _s1_batch(),
        price_source=_source(),
        physical_view=_s1_view(),
        identity_context=_context(),
    )
    assert result.applied is True
    assert result.zero_fill is False
    assert result.consumed_lots == (("S1#T0", "A:L1", 100),)
    assert len(result.transfers) == 1
    transport = result.transfers[0]
    assert transport.from_sleeve == "A" and transport.to_sleeve == "B"
    assert transport.total_shares == 100 and transport.price == 12.0

    seller, buyer = ledger.state_of("A"), ledger.state_of("B")
    assert seller.cash == 51200.0
    assert seller.shares_of(_X) == 300
    assert buyer.cash == 48800.0
    assert buyer.shares_of(_X) == 100
    assert buyer.lots[0].source_lot_id == "L1"
    assert buyer.lots[0].acquired_price == 12.0
    assert buyer.lots[0].acquired_fee == 0.0
    assert result.cash_total_after == 100000.0
    assert result.source_shares_after == ((_X, "L1", 400),)


def test_s1_settlement_prerequisite_total_value_and_nav():
    """第 8.2 件数值：A 54800/0.548、B 50000/0.500、物理 104800/1.048。"""
    ledger = _s1_ledger()
    ledger.apply_settlement(
        _s1_batch(),
        price_source=_source(),
        physical_view=_s1_view(),
        identity_context=_context(),
    )
    source = _source()
    assert ledger.state_of("A").total_value(source) == 54800.0
    assert ledger.state_of("B").total_value(source) == 50000.0
    assert ledger.state_of("A").total_value(source) / 100000.0 == 0.548
    assert ledger.state_of("B").total_value(source) / 100000.0 == 0.500
    # 物理侧总值必须由注入的物理现金事实 + 股数×价派生（不得写恒真算式）
    batch = _s1_batch()
    physical_total = batch.physical_cash_after + 400 * 12.0
    assert physical_total == 104800.0
    assert physical_total / 100000.0 == 1.048


def test_settlement_is_idempotent_and_rejects_key_reuse():
    ledger = _s1_ledger()
    batch = _s1_batch()
    ledger.apply_settlement(
        batch, price_source=_source(), physical_view=_s1_view(), identity_context=_context()
    )
    before = dict(ledger.snapshot())
    again = ledger.apply_settlement(
        batch, price_source=_source(), physical_view=_s1_view(), identity_context=_context()
    )
    assert again.applied is False
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ("S1",)

    with pytest.raises(VirtualLedgerError, match="幂等键冲突"):
        ledger.apply_settlement(
            _s1_batch(shares=50),
            price_source=_source(),
            physical_view=_s1_view(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before


def test_fifo_consumes_earliest_lot_first():
    ledger = VirtualLedger()
    ledger.open_account(
        "A",
        50000.0,
        [
            _lot("A:L1", shares=100, source="L1", date=_T0),
            _lot("A:L2", shares=300, source="L2", date=_T1),
        ],
    )
    ledger.open_account("B", 50000.0)
    plan = ledger.fifo_allocations("A", _X, 200)
    assert [(item.virtual_lot_id, item.shares) for item in plan] == [("A:L1", 100), ("A:L2", 100)]
    with pytest.raises(VirtualLedgerError, match="FIFO 批次不足"):
        ledger.fifo_allocations("A", _X, 401)


def test_transfer_rejects_non_fifo_consumption():
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [_lot("A:L1", shares=100, source="L1", date=_T0)])
    ledger.open_account("B", 50000.0)
    ledger._books["A"].lots.append(_lot("A:L2", shares=300, source="L2", date=_T1))
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="不符合 FIFO"):
        ledger.transfer(
            InternalTransfer(
                settlement_id="S1",
                trade_date=_T1,
                ts_code=_X,
                from_sleeve="A",
                to_sleeve="B",
                price=12.0,
                allocations=(LotAllocation("A:L2", 200),),
            ),
            price_source=_source(),
        )
    assert dict(ledger.snapshot()) == before


def test_settlement_s1_consumes_multiple_lots_in_fifo_order():
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [_lot("A:L1", shares=100, source="L1", date=_T0)])
    ledger._books["A"].lots.append(_lot("A:L2", shares=300, source="L2", date=_T1))
    ledger.open_account("B", 50000.0)
    view = PhysicalLotView([PhysicalLot(_X, "L1", 100), PhysicalLot(_X, "L2", 300)])
    batch = InternalSettlementBatch(
        batch_id="S1b",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 200, 400),
            SleeveIntent("B", _X, "buy", 200, 0),
        ),
        transfers=(TransferFact(_X, "A", "B", 200, 12.0),),
        physical_cash_before=100000.0,
        physical_cash_after=100000.0,
    )
    result = ledger.apply_settlement(
        batch, price_source=_source(), physical_view=view, identity_context=_context()
    )
    assert result.consumed_lots == (("S1b#T0", "A:L1", 100), ("S1b#T0", "A:L2", 100))
    assert ledger.state_of("A").shares_by_source() == {(_X, "L2"): 200}
    assert ledger.state_of("B").shares_by_source() == {(_X, "L1"): 100, (_X, "L2"): 100}


def test_s1_transfer_must_match_intents():
    ledger = _s1_ledger()
    before = dict(ledger.snapshot())
    short = InternalSettlementBatch(
        batch_id="S1short",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 300, 400),
            SleeveIntent("B", _X, "buy", 50, 0),
        ),
        transfers=(TransferFact(_X, "A", "B", 50, 12.0),),
        physical_cash_before=100000.0,
        physical_cash_after=100000.0,
    )
    with pytest.raises(VirtualLedgerError, match="S2/S3"):
        ledger.apply_settlement(
            short, price_source=_source(), physical_view=_s1_view(), identity_context=_context()
        )
    assert dict(ledger.snapshot()) == before

    no_intent = InternalSettlementBatch(
        batch_id="S1none",
        trade_date=_T1,
        transfers=(TransferFact(_X, "A", "B", 50, 12.0),),
        physical_cash_before=100000.0,
        physical_cash_after=100000.0,
    )
    with pytest.raises(VirtualLedgerError, match="缺原始意图事实"):
        ledger.apply_settlement(
            no_intent, price_source=_source(), physical_view=_s1_view(), identity_context=_context()
        )
    assert dict(ledger.snapshot()) == before


# ────────────────────────── S4 结算黄金值 ──────────────────────────


def _s4_ledger(cost_model=None) -> VirtualLedger:
    ledger = VirtualLedger(cost_model=cost_model)
    ledger.open_account("A", 2500.0)
    ledger.open_account("B", 7500.0)
    return ledger


def _s4_batch(cash_after: float = 7995.0, fee: float = 5.0) -> InternalSettlementBatch:
    return InternalSettlementBatch(
        batch_id="S4",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _X, "buy", 300, 0),
        ),
        physical_fills=(PhysicalFillFact(_X, "buy", 200, 10.0, fee, "L2", "F1"),),
        physical_cash_before=10000.0,
        physical_cash_after=cash_after,
    )


class _SyntheticCost:
    """合成买入成本：``max(0.0003×amount, 5)``，记录调用金额。"""

    def __init__(self) -> None:
        self.calls: list = []

    def calculate_buy_cost(self, amount: float) -> float:
        self.calls.append(amount)
        return max(amount * 0.0003, 5.0)


def _s4_view(shares: int = 200) -> PhysicalLotView:
    view = PhysicalLotView()
    view.apply_buy_fill(ts_code=_X, source_lot_id="L2", shares=shares, event_id="F1")
    return view


def test_s4_settlement_allocates_shares_fees_and_lots():
    cost = _SyntheticCost()
    ledger = _s4_ledger(cost)
    result = ledger.apply_settlement(
        _s4_batch(),
        price_source=_source(price=10.0),
        physical_view=_s4_view(),
        identity_context=_context(),
        physical_positions={_X: {"shares": 200}},
    )
    assert [(item.sleeve_id, item.allocated_shares) for item in result.share_allocations] == [
        ("A", 50),
        ("B", 150),
    ]
    waterfall = result.fee_waterfalls[0]
    assert waterfall.order == ("A", "B")
    assert waterfall.final_fees == (("A", 0.0), ("B", 5.0))
    assert cost.calls == [500.0, 1500.0]

    assert ledger.state_of("A").cash == 2000.0
    assert ledger.state_of("B").cash == 5995.0
    assert ledger.state_of("A").lots[0].acquired_fee == 0.0
    assert ledger.state_of("B").lots[0].acquired_fee == 5.0
    assert ledger.state_of("B").lots[0].acquired_cost == 1505.0
    assert result.cash_total_after == 7995.0
    assert result.source_shares_after == ((_X, "L2", 200),)


def test_s4_settlement_totals_and_nav_golden():
    ledger = _s4_ledger(_SyntheticCost())
    ledger.apply_settlement(
        _s4_batch(),
        price_source=_source(price=10.0),
        physical_view=_s4_view(),
        identity_context=_context(),
    )
    source = _source(price=10.0)
    assert ledger.state_of("A").total_value(source) == 2500.0
    assert ledger.state_of("B").total_value(source) == 7495.0
    assert ledger.state_of("A").total_value(source) / 10000.0 == 0.25
    assert ledger.state_of("B").total_value(source) / 10000.0 == 0.7495
    physical_total = _s4_batch().physical_cash_after + 200 * 10.0
    assert physical_total == 9995.0
    assert physical_total / 10000.0 == 0.9995


def test_zero_fill_short_circuit_creates_nothing_and_skips_cost_model():
    class _Forbidden:
        def calculate_buy_cost(self, amount: float) -> float:  # pragma: no cover - 必须不调用
            raise AssertionError("零成交短路不得调用成本模型")

    ledger = _s4_ledger(_Forbidden())
    batch = InternalSettlementBatch(
        batch_id="S4zero",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _X, "buy", 300, 0),
        ),
        physical_cash_before=10000.0,
        physical_cash_after=10000.0,
    )
    result = ledger.apply_settlement(
        batch,
        price_source=_source(price=10.0),
        physical_view=PhysicalLotView(),
        identity_context=_context(),
    )
    assert result.zero_fill is True
    assert result.share_allocations == ()
    assert result.fee_waterfalls == ()
    assert ledger.state_of("A").lots == () and ledger.state_of("B").lots == ()
    assert ledger.state_of("A").cash == 2500.0 and ledger.state_of("B").cash == 7500.0


def test_zero_fill_with_fee_is_rejected_without_commit():
    ledger = _s4_ledger(_SyntheticCost())
    before = dict(ledger.snapshot())
    batch = InternalSettlementBatch(
        batch_id="S4zeroFee",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _X, "buy", 300, 0),
        ),
        physical_cash_before=10000.0,
        physical_cash_after=9995.0,
    )
    with pytest.raises(VirtualLedgerError):
        ledger.apply_settlement(
            batch,
            price_source=_source(price=10.0),
            physical_view=PhysicalLotView(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()


# ────────────────────────── 结算后 I1/I2 回验与零提交 ──────────────────────────


def test_missing_physical_lot_breaks_i2_and_keeps_zero_commit():
    ledger = _s4_ledger(_SyntheticCost())
    before = dict(ledger.snapshot())
    with pytest.raises(ReconciliationError) as exc:
        ledger.apply_settlement(
            _s4_batch(),
            price_source=_source(price=10.0),
            physical_view=PhysicalLotView(),  # 物理成交未注入伴随视图
            identity_context=_context(),
        )
    assert "I2" in exc.value.identities
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()
    assert ledger.poisoned_reason is not None
    with pytest.raises(VirtualLedgerError, match="终态"):
        ledger.apply_settlement(
            _s4_batch(),
            price_source=_source(price=10.0),
            physical_view=_s4_view(),
            identity_context=_context(),
        )


def test_cash_conservation_break_is_terminal_and_zero_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.lazybull.v2.core.accounting.settlement_batch as batch_mod

    ledger = _s4_ledger(_SyntheticCost())
    original = batch_mod._apply_batch_fills

    def broken(ledger_ref, candidate, batch):
        result = original(ledger_ref, candidate, batch)
        candidate["A"].cash += 1.0  # 故意破坏 Σ 现金守恒
        return result

    monkeypatch.setattr(batch_mod, "_apply_batch_fills", broken)
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerConservationError, match="与物理现金不一致"):
        ledger.apply_settlement(
            _s4_batch(),
            price_source=_source(price=10.0),
            physical_view=_s4_view(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before
    monkeypatch.undo()
    with pytest.raises(VirtualLedgerError, match="终态"):
        ledger.apply_settlement(
            _s4_batch(),
            price_source=_source(price=10.0),
            physical_view=_s4_view(),
            identity_context=_context(),
        )


def test_source_share_drift_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.lazybull.v2.core.accounting.settlement_batch as batch_mod

    ledger = _s4_ledger(_SyntheticCost())
    original = batch_mod._apply_batch_fills

    def broken(ledger_ref, candidate, batch):
        result = original(ledger_ref, candidate, batch)
        candidate["A"].lots.pop()  # 虚拟取得批次丢失 → 来源股数漂移
        return result

    monkeypatch.setattr(batch_mod, "_apply_batch_fills", broken)
    with pytest.raises(VirtualLedgerConservationError, match="来源股数漂移"):
        ledger.apply_settlement(
            _s4_batch(),
            price_source=_source(price=10.0),
            physical_view=_s4_view(),
            identity_context=_context(),
        )
    assert ledger.batch_ids() == ()
    assert ledger.poisoned_reason is not None


def test_settlement_identities_helper_reports_i1_and_i2():
    ledger = _s1_ledger()
    states = dict(ledger.snapshot())
    with pytest.raises(ReconciliationError) as exc:
        check_settlement_identities(
            physical_cash=100000.5,
            physical_view=_s1_view(),
            virtual_states=states,
            context=_context(),
        )
    assert exc.value.identities == ("I1",)

    wrong_view = PhysicalLotView([PhysicalLot(_X, "L1", 399)])
    with pytest.raises(ReconciliationError) as exc2:
        check_settlement_identities(
            physical_cash=100000.0,
            physical_view=wrong_view,
            virtual_states=states,
            context=_context(),
        )
    assert "I2" in exc2.value.identities


# ────────────────────────── 输入拒绝（零副作用） ──────────────────────────


def test_fill_price_must_equal_same_day_source():
    ledger = _s4_ledger(_SyntheticCost())
    before = dict(ledger.snapshot())
    batch = InternalSettlementBatch(
        batch_id="S4price",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _X, "buy", 300, 0),
        ),
        physical_fills=(PhysicalFillFact(_X, "buy", 200, 10.5, 5.0, "L2", "F1"),),
        physical_cash_before=10000.0,
        physical_cash_after=7895.0,
    )
    with pytest.raises(VirtualLedgerError, match="同日同价源定价契约"):
        ledger.apply_settlement(
            batch,
            price_source=_source(price=10.0),
            physical_view=_s4_view(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before


def test_cross_day_price_source_is_rejected():
    ledger = _s1_ledger()
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="必须同日同价源"):
        ledger.apply_settlement(
            _s1_batch(),
            price_source=_source(date=_T0),
            physical_view=_s1_view(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before


def test_physical_cash_delta_must_match_fill_amount_and_fee():
    ledger = _s4_ledger(_SyntheticCost())
    with pytest.raises(VirtualLedgerError, match="物理现金减少额"):
        ledger.apply_settlement(
            _s4_batch(cash_after=8000.0),
            price_source=_source(price=10.0),
            physical_view=_s4_view(),
            identity_context=_context(),
        )
    assert ledger.batch_ids() == ()


def test_context_date_and_sleeve_set_must_match():
    ledger = _s1_ledger()
    with pytest.raises(VirtualLedgerError, match="上下文日期"):
        ledger.apply_settlement(
            _s1_batch(),
            price_source=_source(),
            physical_view=_s1_view(),
            identity_context=_context(date=_T0),
        )
    with pytest.raises(VirtualLedgerError, match="袖集合"):
        ledger.apply_settlement(
            _s1_batch(),
            price_source=_source(),
            physical_view=_s1_view(),
            identity_context=_context(sleeves=("A", "C")),
        )


def test_batch_structure_validation():
    with pytest.raises(VirtualLedgerError, match="重复袖/股票意图"):
        InternalSettlementBatch(
            batch_id="dup",
            trade_date=_T1,
            intents=(SleeveIntent("A", _X, "buy", 1, 0), SleeveIntent("A", _X, "buy", 2, 0)),
        )
    with pytest.raises(VirtualLedgerError, match="同一股票多笔物理成交"):
        InternalSettlementBatch(
            batch_id="dupfill",
            trade_date=_T1,
            physical_fills=(
                PhysicalFillFact(_X, "buy", 100, 10.0, 5.0, "L2", "F1"),
                PhysicalFillFact(_X, "buy", 100, 10.0, 5.0, "L3", "F2"),
            ),
        )
    with pytest.raises(VirtualLedgerError, match="状态机=S5"):
        PhysicalFillFact(_X, "sell", 100, 10.0, 5.0, "L2", "F1")
    with pytest.raises(VirtualLedgerError, match="fill_id"):
        InternalSettlementBatch(
            batch_id="dupId",
            trade_date=_T1,
            physical_fills=(
                PhysicalFillFact("600000.SH", "buy", 100, 10.0, 5.0, "L2", "F1"),
                PhysicalFillFact("000001.SZ", "buy", 100, 10.0, 5.0, "L3", "F1"),
            ),
        )
    with pytest.raises(VirtualLedgerError, match="物理现金"):
        InternalSettlementBatch("neg", _T1, physical_cash_after=-1.0)


def test_transfer_and_fill_cannot_share_a_stock():
    ledger = _s1_ledger()
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    batch = InternalSettlementBatch(
        batch_id="mix",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 300, 400),
            SleeveIntent("B", _X, "buy", 100, 0),
        ),
        physical_fills=(PhysicalFillFact(_X, "buy", 100, 12.0, 0.0, "L3", "F1"),),
        transfers=(TransferFact(_X, "A", "B", 100, 12.0),),
        physical_cash_before=100000.0,
        physical_cash_after=98800.0,
    )
    with pytest.raises(VirtualLedgerError, match="既登记内部转账又登记物理成交"):
        ledger.apply_settlement(
            batch, price_source=_source(), physical_view=view, identity_context=_context()
        )


# ────────────────────────── 物理伴随视图同事件维护 ──────────────────────────


def test_physical_view_buy_fill_is_idempotent_and_conflict_safe():
    view = PhysicalLotView()
    assert view.apply_buy_fill(ts_code=_X, source_lot_id="L2", shares=200, event_id="F1") is True
    assert view.aggregate() == {(_X, "L2"): 200}
    assert view.apply_buy_fill(ts_code=_X, source_lot_id="L2", shares=200, event_id="F1") is False
    assert view.aggregate() == {(_X, "L2"): 200}
    with pytest.raises(PhysicalLotViewError, match="幂等键冲突"):
        view.apply_buy_fill(ts_code=_X, source_lot_id="L2", shares=100, event_id="F1")
    with pytest.raises(PhysicalLotViewError, match="正整数"):
        view.apply_buy_fill(ts_code=_X, source_lot_id="L9", shares=0, event_id="F2")
    assert view.applied_event_ids() == ("F1",)


def test_intents_without_any_settlement_fact_are_rejected():
    """A-R1-01：声明了卖出/反向意图却没有转账事实时不得静默通过。"""
    ledger = _s1_ledger()
    before = dict(ledger.snapshot())
    batch = InternalSettlementBatch(
        batch_id="NOFACT",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 300, 400),
            SleeveIntent("B", _X, "buy", 100, 0),
        ),
        physical_cash_before=100000.0,
        physical_cash_after=100000.0,
    )
    with pytest.raises(VirtualLedgerError, match="意图缺结算事实"):
        ledger.apply_settlement(
            batch, price_source=_source(), physical_view=_s1_view(), identity_context=_context()
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()
    assert ledger.poisoned_reason is None


def test_ledger_rejects_unregistered_s1_extension_like_merger():
    """A-R2-02：账本层 S1 必须与合并器同口径——只接受一对等量反向。"""
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [_lot()])
    ledger.open_account("B", 50000.0)
    ledger.open_account("C", 50000.0)
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    batch = InternalSettlementBatch(
        batch_id="ATTRIB",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 300, 400),
            SleeveIntent("B", _X, "buy", 50, 0),
            SleeveIntent("C", _X, "buy", 50, 0),
        ),
        transfers=(TransferFact(_X, "A", "B", 100, 12.0),),
        physical_cash_before=150000.0,
        physical_cash_after=150000.0,
    )
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="只定义一对等量反向"):
        ledger.apply_settlement(
            batch,
            price_source=_source(),
            physical_view=view,
            identity_context=_context(sleeves=("A", "B", "C")),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()


def test_s1_batch_records_no_voided_intents():
    """A-R2-01：内部转账已完成的买入增量不得登记为 Q22 未成交作废。"""
    ledger = _s1_ledger()
    result = ledger.apply_settlement(
        _s1_batch(),
        price_source=_source(),
        physical_view=_s1_view(),
        identity_context=_context(),
    )
    assert result.voided_intents == ()
    assert result.zero_fill is False


def test_settlement_key_cannot_be_reused_across_registries():
    """A-R2-09：转账与结算批次共享同一稳定键空间（同一身份不得承载两套事实）。"""
    ledger = _s1_ledger()
    ledger.transfer(
        InternalTransfer(
            settlement_id="S1",
            trade_date=_T1,
            ts_code=_X,
            from_sleeve="A",
            to_sleeve="B",
            price=12.0,
            allocations=(LotAllocation("A:L1", 50),),
        ),
        price_source=_source(),
    )
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="同一结算键不得承载两套事实"):
        ledger.apply_settlement(
            _s1_batch(shares=100),
            price_source=_source(),
            physical_view=_s1_view(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()


def test_fee_settlement_requires_injected_cost_model():
    """A-R2-06：缺成本模型必须 fail-closed，禁止静默读取生产费率。"""
    ledger = VirtualLedger()  # 未注入 cost_model
    ledger.open_account("A", 2500.0)
    ledger.open_account("B", 7500.0)
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="显式注入买入成本模型"):
        ledger.apply_settlement(
            _s4_batch(),
            price_source=_source(price=10.0),
            physical_view=_s4_view(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before


def test_borrowed_credit_is_preserved_and_blocks_settlement():
    """A-R1-03：候选副本不得静默清零借用占用；逐袖守卫必须可打红。"""
    ledger = _s4_ledger(_SyntheticCost())
    ledger._books["A"].borrowed_credit = 100.0
    before = dict(ledger.snapshot())
    with pytest.raises(ReconciliationError) as exc:
        ledger.apply_settlement(
            _s4_batch(),
            price_source=_source(price=10.0),
            physical_view=_s4_view(),
            identity_context=_context(),
        )
    assert "I1" in exc.value.identities
    assert "借用占用" in str(exc.value)
    assert dict(ledger.snapshot()) == before
    assert ledger.poisoned_reason is not None


def test_voided_intents_record_partial_and_zero_fill_q22():
    """Q22 日终作废证据：部分成交与全零成交都必须留下未成交增量。"""
    ledger = _s4_ledger(_SyntheticCost())
    partial = ledger.apply_settlement(
        _s4_batch(),
        price_source=_source(price=10.0),
        physical_view=_s4_view(),
        identity_context=_context(),
    )
    # 意图 100/300、成交 200 ⇒ 50/150 已分得，未成交 50/150 作废
    assert partial.voided_intents == (("A", _X, 50), ("B", _X, 150))
    # S1 批次不得出现任何作废（转账已全额完成买入增量）
    assert _s1_ledger().apply_settlement(
        _s1_batch(), price_source=_source(), physical_view=_s1_view(), identity_context=_context()
    ).voided_intents == ()
    zero = _s4_ledger(_SyntheticCost())
    zero_result = zero.apply_settlement(
        InternalSettlementBatch(
            batch_id="S4zero2",
            trade_date=_T1,
            intents=(
                SleeveIntent("A", _X, "buy", 100, 0),
                SleeveIntent("B", _X, "buy", 300, 0),
            ),
            physical_cash_before=10000.0,
            physical_cash_after=10000.0,
        ),
        price_source=_source(price=10.0),
        physical_view=PhysicalLotView(),
        identity_context=_context(),
    )
    assert zero_result.zero_fill is True
    assert zero_result.voided_intents == (("A", _X, 100), ("B", _X, 300))


def test_mixed_batch_zero_fill_stock_still_records_voided_intents():
    """A-R3-01：混合批次中"某股零成交"必须逐股票登记 Q22 作废，不得静默丢弃。"""
    ledger = _s4_ledger(_SyntheticCost())
    view = PhysicalLotView()
    view.apply_buy_fill(ts_code=_Y, source_lot_id="L9", shares=100, event_id="F9")
    batch = InternalSettlementBatch(
        batch_id="MIX2",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _Y, "buy", 100, 0),
        ),
        physical_fills=(PhysicalFillFact(_Y, "buy", 100, 10.0, 5.0, "L9", "F9"),),
        physical_cash_before=10000.0,
        physical_cash_after=8995.0,
    )
    result = ledger.apply_settlement(
        batch,
        price_source=PriceSource(as_of=_T1, source_id="syn:p65", prices={_X: 10.0, _Y: 10.0}),
        physical_view=view,
        identity_context=_context(),
        physical_positions={_Y: {"shares": 100}},
    )
    assert result.voided_intents == (("A", _X, 100),)


def test_batch_transfer_sub_ids_enter_the_shared_id_space():
    """A-R3-02/B-R3-03：批次内转账子键不得被外部转账复用（同键异内容拒绝）。"""
    ledger = _s1_ledger()
    ledger.apply_settlement(
        _s1_batch(), price_source=_source(), physical_view=_s1_view(), identity_context=_context()
    )
    assert "S1#T0" in ledger.settlement_ids()
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="幂等键冲突"):
        ledger.transfer(
            InternalTransfer(
                settlement_id="S1#T0",
                trade_date=_T1,
                ts_code=_X,
                from_sleeve="A",
                to_sleeve="B",
                price=12.0,
                allocations=(LotAllocation("A:L1", 50),),
            ),
            price_source=_source(),
        )
    assert dict(ledger.snapshot()) == before


def test_external_transfer_occupying_derived_key_blocks_batch():
    """A-R4-01/B-R4-01：外部先占用派生键时，批次不得静默覆盖既有幂等指纹。"""
    ledger = _s1_ledger()
    ledger.transfer(
        InternalTransfer(
            settlement_id="S1#T0",
            trade_date=_T1,
            ts_code=_X,
            from_sleeve="A",
            to_sleeve="B",
            price=12.0,
            allocations=(LotAllocation("A:L1", 50),),
        ),
        price_source=_source(),
    )
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="派生转账子键"):
        ledger.apply_settlement(
            _s1_batch(),
            price_source=_source(),
            physical_view=_s1_view(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()
    assert ledger.settlement_ids() == ("S1#T0",)


def test_zero_delta_intent_does_not_break_s1_direction_parsing():
    """A-R4-02：零增量意图不参与方向解析（不得因持有袖的目标=当前而拒绝）。"""
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [_lot()])
    ledger.open_account("B", 50000.0)
    ledger.open_account("C", 50000.0, [_lot("C:L2", shares=50, source="L2")])
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400), PhysicalLot(_X, "L2", 50)])
    batch = InternalSettlementBatch(
        batch_id="S1zero",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 300, 400),
            SleeveIntent("B", _X, "buy", 100, 0),
            SleeveIntent("C", _X, "buy", 50, 50),
        ),
        transfers=(TransferFact(_X, "A", "B", 100, 12.0),),
        physical_cash_before=150000.0,
        physical_cash_after=150000.0,
    )
    result = ledger.apply_settlement(
        batch,
        price_source=_source(),
        physical_view=view,
        identity_context=_context(sleeves=("A", "B", "C")),
    )
    assert result.voided_intents == ()
    assert ledger.state_of("C").shares_of(_X) == 50


def test_settlement_records_identity_evidence():
    """A-R4-04：结算结果必须携带 I1/I2 成功证据与是否已与聚合持仓对账。"""
    ledger = _s4_ledger(_SyntheticCost())
    result = ledger.apply_settlement(
        _s4_batch(),
        price_source=_source(price=10.0),
        physical_view=_s4_view(),
        identity_context=_context(),
        physical_positions={_X: {"shares": 200}},
    )
    assert [item.identity for item in result.identity_results] == ["I1", "I2"]
    assert result.positions_reconciled is True
    ledger2 = _s4_ledger(_SyntheticCost())
    result2 = ledger2.apply_settlement(
        _s4_batch(),
        price_source=_source(price=10.0),
        physical_view=_s4_view(),
        identity_context=_context(),
    )
    assert [item.identity for item in result2.identity_results] == ["I1", "I2"]
    assert result2.positions_reconciled is False


def test_settlement_gate_aggregates_i1_and_i2_breaches():
    """A-R3-03：结算门必须汇总全部破缺编号（借用占用 I1 + 缺物理 lot I2）。"""
    ledger = _s4_ledger(_SyntheticCost())
    ledger._books["A"].borrowed_credit = 100.0
    with pytest.raises(ReconciliationError) as exc:
        ledger.apply_settlement(
            _s4_batch(),
            price_source=_source(price=10.0),
            physical_view=PhysicalLotView(),  # 缺物理 lot → I2 也破缺
            identity_context=_context(),
        )
    assert set(exc.value.identities) == {"I1", "I2"}


def test_settlement_internal_api_surface_exists():
    """B-R1-10：结算编排依赖的账本内部面必须有显式存在性锁。"""
    required = (
        "_copy_books",
        "_cash_total",
        "_source_share_totals",
        "_states_of",
        "_assert_cash_total",
        "_assert_source_shares",
        "_assert_no_negative_cash",
        "_assert_acquisition_ids_free",
        "_assert_lot_dates",
        "_apply_transfer",
        "_resolve_allocations",
        "_fifo_lots",
        "_plan_fifo_allocations",
        "_poison",
        "_commit_settlement",
        "_buy_cost",
        "_assert_writable",
    )
    missing = [name for name in required if not hasattr(VirtualLedger, name)]
    assert missing == []
    assert hasattr(VirtualLedger, "apply_settlement")
    instance = VirtualLedger()
    assert hasattr(instance, "_books") and hasattr(instance, "_batches")


def test_settlement_never_reaches_production_ledger():
    """结构锁（B-R4-06）：结算路径不持有/不 import 生产 `Ledger`，无类级共享状态。"""
    import inspect

    import src.lazybull.v2.core.accounting.settlement_batch as batch_mod
    import src.lazybull.v2.core.accounting.settlement_kernel as kernel_mod
    import src.lazybull.v2.core.accounting.virtual_ledger as ledger_mod

    for module in (batch_mod, kernel_mod, ledger_mod):
        source = inspect.getsource(module)
        assert "core.accounting.ledger import" not in source
        assert "from src.lazybull.v2.core.accounting.ledger" not in source
    params = inspect.signature(VirtualLedger.apply_settlement).parameters
    assert "ledger" not in params and "production_ledger" not in params
    assert not hasattr(VirtualLedger, "ledger")
    ledger = _s4_ledger(_SyntheticCost())
    ledger.apply_settlement(
        _s4_batch(),
        price_source=_source(price=10.0),
        physical_view=_s4_view(),
        identity_context=_context(),
    )
    assert ledger.state_of("A").cash == 2000.0


# ──────────────────── Codex 复审（C65-T2-01/02/03）────────────────────


def test_intent_current_shares_must_match_ledger_holdings():
    """C65-T2-01（P1）：目标不是增量——陈旧当前股数必须拒绝，禁止超目标分配。"""
    ledger = VirtualLedger(cost_model=_SyntheticCost())
    ledger.open_account("A", 5000.0, [_lot("A:OLD", shares=100, source="OLD")])
    ledger.open_account("B", 5000.0)
    view = PhysicalLotView([PhysicalLot(_X, "OLD", 100), PhysicalLot(_X, "NEW", 200)])
    batch = InternalSettlementBatch(
        batch_id="STALE",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),  # 实持 100，却声明 0
            SleeveIntent("B", _X, "buy", 100, 0),
        ),
        physical_fills=(PhysicalFillFact(_X, "buy", 200, 10.0, 5.0, "NEW", "F1"),),
        physical_cash_before=10000.0,
        physical_cash_after=7995.0,
    )
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="意图声明当前股数 0 != 账本当前虚拟持仓 100"):
        ledger.apply_settlement(
            batch,
            price_source=_source(price=10.0),
            physical_view=view,
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()
    assert ledger.state_of("A").shares_of(_X) == 100

    # S1 同口径：A 实持 400，却声明 current=100（若放行会残留 300 股且目标不兑现）
    s1 = _s1_ledger()
    stale_s1 = InternalSettlementBatch(
        batch_id="S1stale",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 0, 100),
            SleeveIntent("B", _X, "buy", 100, 0),
        ),
        transfers=(TransferFact(_X, "A", "B", 100, 12.0),),
        physical_cash_before=100000.0,
        physical_cash_after=100000.0,
    )
    before_s1 = dict(s1.snapshot())
    with pytest.raises(
        VirtualLedgerError, match="意图声明当前股数 100 != 账本当前虚拟持仓 400"
    ):
        s1.apply_settlement(
            stale_s1,
            price_source=_source(),
            physical_view=_s1_view(),
            identity_context=_context(),
        )
    assert dict(s1.snapshot()) == before_s1
    assert s1.batch_ids() == ()


def test_zero_fill_still_requires_price_for_incremental_intents():
    """C65-T2-02（P2）：零成交短路不豁免缺价拒绝面（F25 §4 停牌/无价 fail-fast）。"""
    ledger = _s4_ledger(_SyntheticCost())
    batch = InternalSettlementBatch(
        batch_id="S4noPrice",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _X, "buy", 300, 0),
        ),
        physical_cash_before=10000.0,
        physical_cash_after=10000.0,
    )
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="零成交短路不豁免缺价拒绝"):
        ledger.apply_settlement(
            batch,
            # 价源标识必须与上下文一致（T2-P3-3 交叉校验）；本用例只测缺价
            price_source=PriceSource(as_of=_T1, source_id="syn:p65", prices={}),
            physical_view=PhysicalLotView(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()

    # 混合批次：X 有成交有价，Y 零成交且缺价 ⇒ 同样拒绝（不得因其他股有成交而放过）
    mixed = InternalSettlementBatch(
        batch_id="S4mixedNoPrice",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _Y, "buy", 100, 0),
        ),
        physical_fills=(PhysicalFillFact(_X, "buy", 100, 10.0, 5.0, "L3", "F1"),),
        physical_cash_before=10000.0,
        physical_cash_after=8995.0,
    )
    with pytest.raises(VirtualLedgerError, match="零成交短路不豁免缺价拒绝"):
        ledger.apply_settlement(
            mixed,
            price_source=PriceSource(as_of=_T1, source_id="syn:p65", prices={_X: 10.0}),
            physical_view=PhysicalLotView(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()


def test_partial_transfer_splits_acquired_fee_proportionally():
    """C65-T2-03（P2）：带费用批次部分转出后，剩余成本按股数比例拆分。"""
    ledger = VirtualLedger(cost_model=_SyntheticCost())
    ledger.open_account("A", 5000.0)
    ledger.open_account("B", 5000.0)
    view = PhysicalLotView()
    view.apply_buy_fill(ts_code=_X, source_lot_id="NEW", shares=200, event_id="F1")
    ledger.apply_settlement(
        InternalSettlementBatch(
            batch_id="S4",
            trade_date=_T1,
            intents=(
                SleeveIntent("A", _X, "buy", 100, 0),
                SleeveIntent("B", _X, "buy", 100, 0),
            ),
            physical_fills=(PhysicalFillFact(_X, "buy", 200, 10.0, 5.0, "NEW", "F1"),),
            physical_cash_before=10000.0,
            physical_cash_after=7995.0,
        ),
        price_source=_source(price=10.0),
        physical_view=view,
        identity_context=_context(),
    )
    # q 相同 ⇒ 受益顺序 A→B、终端 B：A 承担 0、B 承担 5（B 的 lot 成本 1005）
    assert ledger.state_of("A").lots[0].acquired_fee == 0.0
    assert ledger.state_of("B").lots[0].acquired_fee == 5.0
    assert ledger.state_of("B").lots[0].acquired_cost == 1005.0

    def _transfer(batch_id: str, shares: int, seller_hold: int, buyer_hold: int):
        """付费袖 B 部分转出：剩余批次成本必须按股数比例拆分。"""
        ledger.apply_settlement(
            InternalSettlementBatch(
                batch_id=batch_id,
                trade_date=_T2,
                intents=(
                    SleeveIntent("B", _X, "sell", seller_hold - shares, seller_hold),
                    SleeveIntent("A", _X, "buy", buyer_hold + shares, buyer_hold),
                ),
                transfers=(TransferFact(_X, "B", "A", shares, 12.0),),
                physical_cash_before=7995.0,
                physical_cash_after=7995.0,
            ),
            price_source=_source(price=12.0, date=_T2),
            physical_view=view,
            identity_context=_context(date=_T2),
        )

    _transfer("S1a", 50, 100, 100)
    remaining = ledger.state_of("B").lots[0]
    assert remaining.shares == 50
    assert remaining.acquired_fee == 2.5
    assert remaining.acquired_cost == 502.5
    assert all(lot.acquired_fee == 0.0 for lot in ledger.state_of("A").lots)

    _transfer("S1b", 30, 50, 150)
    remaining = ledger.state_of("B").lots[0]
    assert remaining.shares == 20
    assert remaining.acquired_fee == 1.0
    assert remaining.acquired_cost == 201.0

    _transfer("S1c", 20, 20, 180)
    assert ledger.state_of("B").lots == ()
    assert ledger.state_of("B").shares_of(_X) == 0
    assert ledger.state_of("A").shares_of(_X) == 200
