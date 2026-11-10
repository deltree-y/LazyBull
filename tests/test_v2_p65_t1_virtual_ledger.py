# -*- coding: utf-8 -*-
"""P6.5 T1：最小虚拟账本与物理 lot 伴随视图（全部显式合成输入）。

覆盖：期初开户、最小内部转账（来源谱系 + 虚拟取得）、**同日同价源定价绑定**、
幂等写入身份、零副作用拒绝、落账守恒自检（含反变异）、逐袖现金守卫的写入侧前置、
生产账本/引擎零触碰、伴随视图与物理聚合对账。
"""

from __future__ import annotations

import copy
import inspect
from types import SimpleNamespace

import pandas as pd
import pytest

from src.lazybull.v2.common.types import TSCode, TradeDate
from src.lazybull.v2.core.accounting.ledger import Ledger
from src.lazybull.v2.core.accounting.physical_lot_view import (
    PhysicalLot,
    PhysicalLotMismatchError,
    PhysicalLotView,
    PhysicalLotViewError,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    InternalTransfer,
    LotAllocation,
    PriceSource,
    VirtualLedger,
    VirtualLedgerConservationError,
    VirtualLedgerError,
    VirtualLot,
    VirtualSleeveState,
)

_D0 = TradeDate(pd.Timestamp("2026-01-05").date())
_D1 = TradeDate(pd.Timestamp("2026-01-06").date())
_X = "600000.SH"


def _lot(
    virtual_lot_id: str = "A:L1",
    *,
    ts_code: str = _X,
    source_lot_id: str = "L1",
    shares: int = 400,
    price: float = 10.0,
    date: TradeDate = _D0,
) -> VirtualLot:
    return VirtualLot(
        virtual_lot_id=virtual_lot_id,
        ts_code=ts_code,
        source_lot_id=source_lot_id,
        shares=shares,
        acquired_date=date,
        acquired_price=price,
    )


def _transfer(
    settlement_id: str = "S1",
    *,
    shares: int = 100,
    price: float = 12.0,
    allocations: tuple[LotAllocation, ...] | None = None,
    from_sleeve: str = "A",
    to_sleeve: str = "B",
    trade_date: TradeDate = _D1,
) -> InternalTransfer:
    return InternalTransfer(
        settlement_id=settlement_id,
        trade_date=trade_date,
        ts_code=_X,
        from_sleeve=from_sleeve,
        to_sleeve=to_sleeve,
        price=price,
        allocations=allocations or (LotAllocation("A:L1", shares),),
    )


def _source(
    price: float = 12.0,
    *,
    date: TradeDate = _D1,
    prices: dict[str, float] | None = None,
) -> PriceSource:
    table = {_X: price} if prices is None else prices
    return PriceSource(as_of=date, source_id="syn:p65", prices=table)


def _ledger() -> VirtualLedger:
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [_lot()])
    ledger.open_account("B", 50000.0)
    return ledger


def test_open_account_records_cash_and_lots() -> None:
    ledger = _ledger()
    assert ledger.sleeve_ids() == ("A", "B")
    state = ledger.state_of("A")
    assert state.cash == 50000.0
    assert state.borrowed_credit == 0.0
    assert state.shares_of(_X) == 400
    assert ledger.state_of("B").lots == ()


def test_transfer_moves_cash_and_preserves_source_lineage() -> None:
    ledger = _ledger()
    assert ledger.transfer(_transfer(), price_source=_source()) is True

    seller = ledger.state_of("A")
    buyer = ledger.state_of("B")
    assert seller.cash == 51200.0
    assert sum(lot.shares for lot in seller.lots) == 300
    assert seller.lots[0].source_lot_id == "L1"

    assert buyer.cash == 48800.0
    assert len(buyer.lots) == 1
    acquired = buyer.lots[0]
    assert acquired.source_lot_id == "L1"
    assert acquired.shares == 100
    assert acquired.acquired_date == _D1
    assert acquired.acquired_price == 12.0
    assert acquired.acquired_cost == 1200.0
    assert acquired.virtual_lot_id == "S1#0"
    assert ledger.last_settlement_id == "S1"
    assert ledger.last_settlement_source_id == "syn:p65"
    assert ledger.settlement_ids() == ("S1",)


def test_transfer_preserves_cash_and_source_share_totals() -> None:
    ledger = _ledger()
    cash_before = sum(state.cash for state in ledger.snapshot().values())
    shares_before: dict[tuple[str, str], int] = {}
    for state in ledger.snapshot().values():
        for key, shares in state.shares_by_source().items():
            shares_before[key] = shares_before.get(key, 0) + shares

    ledger.transfer(_transfer(), price_source=_source())

    cash_after = sum(state.cash for state in ledger.snapshot().values())
    shares_after: dict[tuple[str, str], int] = {}
    for state in ledger.snapshot().values():
        for key, shares in state.shares_by_source().items():
            shares_after[key] = shares_after.get(key, 0) + shares
    assert cash_after == cash_before
    assert shares_after == shares_before == {(_X, "L1"): 400}


def test_post_transfer_conservation_guard_catches_broken_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _ledger()
    original = VirtualLedger.__dict__["_apply_transfer"].__func__

    def broken(transfer, seller, buyer, consumed, amount):
        original(transfer, seller, buyer, consumed, amount)
        buyer.cash += 1.0  # 故意破坏 Σ 现金守恒（模拟落账实现被改坏）

    monkeypatch.setattr(VirtualLedger, "_apply_transfer", staticmethod(broken))
    with pytest.raises(VirtualLedgerConservationError, match="Σ 虚拟现金漂移"):
        ledger.transfer(_transfer(), price_source=_source())


def test_post_transfer_guard_catches_source_share_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1 复审 P3（T1-A-R2-02）：来源股数漂移分支必须有专用负例。"""
    ledger = _ledger()
    original = VirtualLedger.__dict__["_apply_transfer"].__func__

    def broken(transfer, seller, buyer, consumed, amount):
        original(transfer, seller, buyer, consumed, amount)
        seller.lots.pop(0)  # 虚拟批次丢失：来源股数漂移

    monkeypatch.setattr(VirtualLedger, "_apply_transfer", staticmethod(broken))
    with pytest.raises(VirtualLedgerConservationError, match="来源股数漂移"):
        ledger.transfer(_transfer(), price_source=_source())


def test_post_transfer_guard_catches_negative_sleeve_cash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1 复审 P3：逐袖现金转负分支必须有专用负例（组合合计仍可为正）。"""
    ledger = _ledger()
    original = VirtualLedger.__dict__["_apply_transfer"].__func__

    def broken(transfer, seller, buyer, consumed, amount):
        original(transfer, seller, buyer, consumed, amount)
        # 组合合计仍为 100000（不触发 Σ 现金守卫），但买入袖透支
        seller.cash = 100001.0
        buyer.cash = -1.0

    monkeypatch.setattr(VirtualLedger, "_apply_transfer", staticmethod(broken))
    with pytest.raises(VirtualLedgerConservationError, match="现金为负"):
        ledger.transfer(_transfer(), price_source=_source())


def test_conservation_tolerance_capped_at_cash_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T1 复审 P3（T1-B-R2-04）：1e9 量级相对容差不得放宽到 1 元。"""
    ledger = VirtualLedger()
    ledger.open_account("A", 1e9, [])
    ledger.open_account("B", 100.0)
    original = VirtualLedger.__dict__["_apply_transfer"].__func__

    def broken(transfer, seller, buyer, consumed, amount):
        original(transfer, seller, buyer, consumed, amount)
        buyer.cash += 1.0  # 相对容差 1e-9×1e9 = 1.0 会放过；封顶 1e-6 必须捕获

    monkeypatch.setattr(VirtualLedger, "_apply_transfer", staticmethod(broken))
    transfer = InternalTransfer(
        settlement_id="S1",
        trade_date=_D1,
        ts_code=_X,
        from_sleeve="A",
        to_sleeve="B",
        price=1.0,
        allocations=(LotAllocation("A:L1", 1),),
    )
    ledger._books["A"].lots = [_lot(shares=1, price=1.0)]
    with pytest.raises(VirtualLedgerConservationError, match="漂移"):
        ledger.transfer(transfer, price_source=_source(price=1.0))


def test_conservation_break_poisons_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """T1 复审 P3（T1-A-R2-03）：破缺后实例置终态，捕获后重试必须被拒。"""
    ledger = _ledger()
    original = VirtualLedger.__dict__["_apply_transfer"].__func__

    def broken(transfer, seller, buyer, consumed, amount):
        original(transfer, seller, buyer, consumed, amount)
        buyer.cash += 1.0

    monkeypatch.setattr(VirtualLedger, "_apply_transfer", staticmethod(broken))
    with pytest.raises(VirtualLedgerConservationError):
        ledger.transfer(_transfer(), price_source=_source())
    assert ledger.poisoned_reason is not None
    monkeypatch.undo()
    with pytest.raises(VirtualLedgerError, match="终态"):
        ledger.transfer(_transfer(), price_source=_source())


def test_repeated_settlement_is_idempotent() -> None:
    ledger = _ledger()
    assert ledger.transfer(_transfer(), price_source=_source()) is True
    before = dict(ledger.snapshot())
    assert ledger.transfer(_transfer(), price_source=_source()) is False
    assert dict(ledger.snapshot()) == before
    assert ledger.settlement_ids() == ("S1",)


def test_same_settlement_id_with_different_content_is_rejected() -> None:
    ledger = _ledger()
    ledger.transfer(_transfer(), price_source=_source())
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="幂等键冲突"):
        ledger.transfer(_transfer(shares=50), price_source=_source())
    assert dict(ledger.snapshot()) == before


def test_transfer_rejects_price_not_bound_to_same_day_source() -> None:
    ledger = _ledger()
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="F25 §5 定价契约"):
        ledger.transfer(_transfer(price=10.0), price_source=_source(12.0))
    assert dict(ledger.snapshot()) == before
    assert ledger.settlement_ids() == ()
    assert ledger.last_settlement_source_id is None


def test_transfer_rejects_cross_day_price_source() -> None:
    ledger = _ledger()
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="必须同日同价源"):
        ledger.transfer(_transfer(), price_source=_source(date=_D0))
    assert dict(ledger.snapshot()) == before


def test_transfer_rejects_missing_price() -> None:
    ledger = _ledger()
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="缺少"):
        ledger.transfer(_transfer(), price_source=_source(prices={}))
    assert dict(ledger.snapshot()) == before


def test_transfer_rejects_backdated_lot() -> None:
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [_lot(date=_D1)])
    ledger.open_account("B", 50000.0)
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="晚于结算日"):
        ledger.transfer(
            _transfer(trade_date=_D0, price=10.0),
            price_source=_source(date=_D0, price=10.0),
        )
    assert dict(ledger.snapshot()) == before


def test_transfer_rejects_insufficient_shares_without_side_effects() -> None:
    ledger = _ledger()
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="超出批次持仓"):
        ledger.transfer(_transfer(shares=500), price_source=_source())
    assert dict(ledger.snapshot()) == before
    assert ledger.settlement_ids() == ()


def test_transfer_rejects_insufficient_buyer_cash_without_side_effects() -> None:
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [_lot()])
    ledger.open_account("B", 50.0)
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="现金不足"):
        ledger.transfer(_transfer(), price_source=_source())
    assert dict(ledger.snapshot()) == before


def test_transfer_rejects_acquisition_id_collision() -> None:
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [_lot()])
    ledger.open_account("B", 50000.0, [_lot("S1#0", source_lot_id="L9", shares=1)])
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="取得批次身份冲突"):
        ledger.transfer(_transfer(), price_source=_source())
    assert dict(ledger.snapshot()) == before


def test_transfer_rejects_unknown_sleeve() -> None:
    ledger = _ledger()
    with pytest.raises(VirtualLedgerError, match="未在虚拟账本开户"):
        ledger.transfer(_transfer(from_sleeve="C"), price_source=_source())


def test_transfer_construction_rejects_self_transfer_and_bad_price() -> None:
    with pytest.raises(VirtualLedgerError, match="转出/转入袖相同"):
        _transfer(to_sleeve="A")
    with pytest.raises(VirtualLedgerError, match="price"):
        _transfer(price=0.0)
    with pytest.raises(VirtualLedgerError, match="shares"):
        LotAllocation("A:L1", 0)


def test_open_account_rejects_duplicate_sleeve_and_negative_cash() -> None:
    ledger = _ledger()
    with pytest.raises(VirtualLedgerError, match="重复开户"):
        ledger.open_account("A", 1.0)
    with pytest.raises(VirtualLedgerError, match="期初虚拟现金"):
        ledger.open_account("C", -1.0)


def test_sleeve_state_rejects_non_finite_cash_and_float_shares() -> None:
    with pytest.raises(VirtualLedgerError, match="必须为有限数"):
        VirtualSleeveState("A", float("nan"), 0.0, ())
    with pytest.raises(VirtualLedgerError, match="shares"):
        _lot(shares=100.0)  # type: ignore[arg-type]


def test_split_source_across_sleeves_keeps_one_source_key() -> None:
    ledger = _ledger()
    ledger.transfer(_transfer(shares=100), price_source=_source())
    ledger.transfer(
        _transfer(
            "S2",
            shares=40,
            allocations=(LotAllocation("S1#0", 40),),
            from_sleeve="B",
            to_sleeve="A",
        ),
        price_source=_source(),
    )

    seller = ledger.state_of("A")
    buyer = ledger.state_of("B")
    assert seller.shares_by_source() == {(_X, "L1"): 340}
    assert buyer.shares_by_source() == {(_X, "L1"): 60}
    # 同一来源被合法拆成两个取得批次：来源键仍唯一，条目数不比
    assert len(seller.lots) == 2
    assert seller.shares_of(_X) + buyer.shares_of(_X) == 400


def test_physical_view_guards_transfer_source_key() -> None:
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    ledger = VirtualLedger(physical_lot_view=view)
    ledger.open_account("A", 50000.0, [_lot(source_lot_id="L9")])
    ledger.open_account("B", 50000.0)
    before = dict(ledger.snapshot())
    with pytest.raises(PhysicalLotViewError, match="物理不存在"):
        ledger.transfer(_transfer(), price_source=_source())
    assert dict(ledger.snapshot()) == before


def test_physical_view_reconciles_against_engine_positions() -> None:
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400), PhysicalLot(_X, "L2", 100)])
    view.validate_against_positions({_X: {"shares": 500}})
    assert view.shares_by_stock() == {_X: 500}
    assert view.shares_of(_X, "L2") == 100

    with pytest.raises(PhysicalLotMismatchError, match="物理聚合持仓"):
        view.validate_against_positions({_X: {"shares": 400}})
    with pytest.raises(PhysicalLotMismatchError, match="股票集不一致"):
        view.validate_against_positions({"000001.SZ": {"shares": 500}})


def test_physical_view_rejects_float_shares_instead_of_truncating() -> None:
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    with pytest.raises(PhysicalLotMismatchError, match="非整数股"):
        view.validate_against_positions({_X: {"shares": 400.7}})
    with pytest.raises(PhysicalLotMismatchError, match="非整数股"):
        view.validate_against_positions({_X: {"shares": float("inf")}})


def test_virtual_ledger_never_touches_engine_or_production_ledger() -> None:
    engine = SimpleNamespace(
        current_capital=100000.0,
        positions={_X: {"shares": 400}},
        trades=[],
    )
    engine_before = copy.deepcopy(engine.__dict__)
    production_ledger = Ledger()
    ledger_before = production_ledger.export_state()

    ledger = _ledger()
    ledger.transfer(_transfer(), price_source=_source())

    assert engine.__dict__ == engine_before
    assert production_ledger.export_state() == ledger_before
    assert "engine" not in inspect.signature(VirtualLedger.transfer).parameters


def test_as_virtual_accounts_projects_runtime_state() -> None:
    ledger = _ledger()
    ledger.transfer(_transfer(), price_source=_source())
    source = PriceSource(as_of=_D1, source_id="syn:p65", prices={_X: 12.0})
    accounts = ledger.as_virtual_accounts(source)

    seller = accounts["A"]
    assert seller.cash.value == 51200.0
    assert seller.total_value.value == 54800.0
    assert seller.borrowed_credit.value == 0.0
    assert sum(lot.shares.value for lot in seller.positions[TSCode(_X)].lots) == 300

    buyer = accounts["B"]
    assert buyer.total_value.value == 50000.0


def test_price_source_rejects_missing_and_non_positive_prices() -> None:
    source = PriceSource(as_of=_D1, source_id="syn:p65", prices={_X: 12.0})
    assert source.price_of(_X) == 12.0
    with pytest.raises(VirtualLedgerError, match="缺少"):
        source.price_of("000001.SZ")
    with pytest.raises(VirtualLedgerError, match="有限正数"):
        PriceSource(as_of=_D1, source_id="syn:p65", prices={_X: 0.0})
