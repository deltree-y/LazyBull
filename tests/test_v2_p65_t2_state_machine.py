# -*- coding: utf-8 -*-
"""P6.5 T2 状态机边界：S2/S3/S5/S7 拒绝、S1+S4 混合输入与端到端结算。

契约依据：``docs/contracts/netting_freeze.md`` F25 §3 状态机表与 P6.5 边界条款、
设计 §6.1（S2/S3/S5/S7 不实现、显式拒绝）。

覆盖：
- 反向不等量（S2/S3）与同向卖出（S5）零副作用拒绝；
- 到期原因（S7）进入已支持的 S1/S4 时不被拒绝，且不得重算到期阈值；
- 同一批次内 S1（一只股票）+ S4（另一只股票）混合结算；
- 多袖账户单袖发单端到端：物理全量目标 + 虚拟分摊 + I1/I2 同时成立；
- 拒绝路径不产生任何账本/视图/批次残留。
"""

from __future__ import annotations

from types import MappingProxyType

import pandas as pd
import pytest

from src.lazybull.v2.common.types import (
    Lot,
    Money,
    Order,
    OrderReason,
    OrderSide,
    OrderType,
    Shares,
    TSCode,
    TradeDate,
    VirtualAccount,
    VirtualPosition,
)
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLot, PhysicalLotView
from src.lazybull.v2.core.accounting.reconciliation import IdentityContext
from src.lazybull.v2.core.accounting.virtual_ledger import (
    InternalSettlementBatch,
    PhysicalFillFact,
    PriceSource,
    SleeveIntent,
    TransferFact,
    VirtualLedger,
    VirtualLedgerError,
    VirtualLot,
)
from src.lazybull.v2.core.decision.virtual_merger import (
    MultiSleevePortfolioMerger,
    VirtualMergerError,
)

_X = "600000.SH"
_Y = "000001.SZ"
_T0 = TradeDate(pd.Timestamp("2026-01-05").date())
_T1 = TradeDate(pd.Timestamp("2026-01-06").date())


class _Cost:
    """合成买入成本 ``max(0.0003×amount, 5)``。"""

    def calculate_buy_cost(self, amount: float) -> float:
        return max(amount * 0.0003, 5.0)


def _source(prices: dict[str, float], date: TradeDate = _T1) -> PriceSource:
    return PriceSource(as_of=date, source_id="syn:p65", prices=prices)


def _context(sleeves=("A", "B")) -> IdentityContext:
    return IdentityContext(
        trade_date=_T1,
        host="pytest_state_machine",
        run_id="R1",
        sleeve_ids=tuple(sleeves),
        price_source_id="syn:p65",
    )


def _order(sleeve: str, side: OrderSide, target: int, code: str = _X, reason=OrderReason.REBALANCE):
    return Order(
        ts_code=TSCode(code),
        side=side,
        target_shares=Shares(target),
        order_type=OrderType.MARKET_OPEN,
        reason=reason,
        source_sleeve=sleeve,
    )


def _vaccount(sleeve: str, cash: float, holdings: dict[str, int], price: float = 10.0):
    positions = {
        TSCode(code): VirtualPosition(
            ts_code=TSCode(code),
            lots=(Lot(Shares(shares), _T0, Money(shares * price)),),
        )
        for code, shares in holdings.items()
        if shares
    }
    total = cash + sum(shares * price for shares in holdings.values())
    return VirtualAccount(
        sleeve_id=sleeve,
        cash=Money(cash),
        positions=MappingProxyType(positions),
        total_value=Money(total),
    )


def _merger(ids=("A", "B")) -> MultiSleevePortfolioMerger:
    return MultiSleevePortfolioMerger(list(ids), {sleeve: 1.0 / len(ids) for sleeve in ids})


def _accounts_signature(accounts: dict) -> tuple:
    """账户不可变签名（拒绝路径零副作用的可比对快照）。"""
    return tuple(
        (
            sid,
            account.cash.value,
            account.total_value.value,
            tuple(sorted(str(code.value) for code in account.positions)),
        )
        for sid, account in sorted(accounts.items())
    )


# ────────────────────────── S2/S3/S5 拒绝 ──────────────────────────


@pytest.mark.parametrize(
    "orders, state",
    [
        # A 400→200（Δ=-200）、B 100→200（Δ=+100）⇒ 净 -100 ⇒ S2
        (
            [_order("A", OrderSide.SELL, 200), _order("B", OrderSide.BUY, 200)],
            "S2",
        ),
        (
            [_order("A", OrderSide.SELL, 300), _order("B", OrderSide.BUY, 300)],
            "S3",
        ),
        (
            [_order("A", OrderSide.SELL, 300), _order("B", OrderSide.SELL, 50)],
            "S5",
        ),
    ],
)
def test_unsupported_states_are_rejected_with_zero_side_effects(orders, state):
    merger = _merger()
    accounts = {
        "A": _vaccount("A", 10000.0, {_X: 400}),
        "B": _vaccount("B", 10000.0, {_X: 100}),
    }
    before = _accounts_signature(accounts)
    with pytest.raises(VirtualMergerError, match=f"状态机={state}"):
        merger.plan_netting(orders, accounts)
    assert _accounts_signature(accounts) == before


def test_s5_sell_fill_is_rejected_at_construction():
    """merger 可以产出 SELL 物理单（单袖发单），但 T2 不结算卖出成交。"""
    merger = _merger()
    accounts = {
        "A": _vaccount("A", 10000.0, {_X: 400}),
        "B": _vaccount("B", 10000.0, {_X: 100}),
    }
    plan = merger.plan_netting([_order("B", OrderSide.SELL, 0)], accounts)
    assert plan.physical_orders()[0].side is OrderSide.SELL
    with pytest.raises(VirtualLedgerError, match="状态机=S5／卖出结算未实现"):
        PhysicalFillFact(_X, "sell", 100, 10.0, 5.0, "L2", "F1")


# ────────────────────────── S7 到期原因 ──────────────────────────


def test_expiry_reason_is_allowed_on_supported_states():
    merger = _merger()
    accounts = {
        "A": _vaccount("A", 10000.0, {_X: 400}),
        "B": _vaccount("B", 10000.0, {_X: 100}),
    }
    plan = merger.plan_netting(
        [
            _order("A", OrderSide.SELL, 300, reason=OrderReason.EXPIRY),
            _order("B", OrderSide.BUY, 200, reason=OrderReason.EXPIRY),
        ],
        accounts,
    )
    assert dict(plan.net_by_code) == {_X: 0}
    assert [(item.from_sleeve, item.to_sleeve, item.shares) for item in plan.offsets] == [
        ("A", "B", 100)
    ]
    # 到期阈值仍由共享规则决定：本层只看目标/增量，不重算任何期限
    assert [item.reason for item in plan.intents] == ["expiry", "expiry"]


def test_s4_reason_is_preserved_and_mixed_reasons_become_netting():
    merger = _merger()
    accounts = {"A": _vaccount("A", 10000.0, {}), "B": _vaccount("B", 10000.0, {})}
    same = merger.plan_netting(
        [_order("A", OrderSide.BUY, 100), _order("B", OrderSide.BUY, 200)], accounts
    )
    assert same.physical_orders()[0].reason is OrderReason.REBALANCE
    mixed = merger.plan_netting(
        [
            _order("A", OrderSide.BUY, 100, reason=OrderReason.REBALANCE),
            _order("B", OrderSide.BUY, 200, reason=OrderReason.EXPIRY),
        ],
        accounts,
    )
    assert mixed.physical_orders()[0].reason is OrderReason.NETTING


# ────────────────────────── 混合批次（S1 + S4） ──────────────────────────


def test_mixed_batch_settles_s1_and_s4_together():
    cost = _Cost()
    ledger = VirtualLedger(cost_model=cost)
    ledger.open_account("A", 20000.0, [VirtualLot("A:L1", _X, "L1", 400, _T0, 10.0)])
    ledger.open_account("B", 20000.0, [VirtualLot("B:L2", _Y, "L2", 100, _T0, 10.0)])
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400), PhysicalLot(_Y, "L2", 100)])
    view.apply_buy_fill(ts_code=_Y, source_lot_id="L3", shares=200, event_id="F1")
    batch = InternalSettlementBatch(
        batch_id="MIX",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 300, 400),
            SleeveIntent("B", _X, "buy", 100, 0),
            SleeveIntent("B", _Y, "buy", 300, 100),
        ),
        transfers=(TransferFact(_X, "A", "B", 100, 12.0),),
        physical_fills=(PhysicalFillFact(_Y, "buy", 200, 10.0, 5.0, "L3", "F1"),),
        physical_cash_before=40000.0,
        physical_cash_after=37995.0,
    )
    result = ledger.apply_settlement(
        batch,
        price_source=_source({_X: 12.0, _Y: 10.0}),
        physical_view=view,
        identity_context=_context(),
        physical_positions={_X: {"shares": 400}, _Y: {"shares": 300}},
    )
    # S1：A→B 100 股，零费用
    assert result.consumed_lots == (("MIX#T0", "A:L1", 100),)
    # S4：B 分得 200 股，费用 5（A 未在 Y 上发单，不参与该股票分摊）
    assert [(item.sleeve_id, item.allocated_shares) for item in result.share_allocations] == [
        ("B", 200)
    ]
    assert dict(result.fee_waterfalls[0].final_fees) == {"B": 5.0}
    assert ledger.state_of("A").cash == 20000.0 + 1200.0
    assert ledger.state_of("B").cash == 20000.0 - 1200.0 - 2000.0 - 5.0
    assert result.cash_total_after == 37995.0
    assert ledger.state_of("A").shares_of(_X) == 300
    assert ledger.state_of("B").shares_of(_X) == 100
    assert ledger.state_of("B").shares_of(_Y) == 300


def test_mixed_batch_rejection_leaves_no_residue():
    ledger = VirtualLedger(cost_model=_Cost())
    ledger.open_account("A", 20000.0, [VirtualLot("A:L1", _X, "L1", 400, _T0, 10.0)])
    ledger.open_account("B", 20000.0, [VirtualLot("B:L2", _Y, "L2", 100, _T0, 10.0)])
    before = dict(ledger.snapshot())
    # S4 成交股数超过原始增量（100+200=300 < 400）
    batch = InternalSettlementBatch(
        batch_id="MIXBAD",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 300, 400),
            SleeveIntent("B", _X, "buy", 100, 0),
            SleeveIntent("B", _Y, "buy", 300, 100),
        ),
        transfers=(TransferFact(_X, "A", "B", 100, 12.0),),
        physical_fills=(PhysicalFillFact(_Y, "buy", 400, 10.0, 5.0, "L3", "F1"),),
        physical_cash_before=40000.0,
        physical_cash_after=35995.0,
    )
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400), PhysicalLot(_Y, "L3", 400)])
    with pytest.raises(VirtualLedgerError, match="超意图成交"):
        ledger.apply_settlement(
            batch,
            price_source=_source({_X: 12.0, _Y: 10.0}),
            physical_view=view,
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()
    assert ledger.poisoned_reason is None


# ────────────────────────── 多袖账户单袖发单端到端 ──────────────────────────


def test_single_issuing_sleeve_end_to_end_uses_full_physical_base():
    merger = _merger()
    ledger = VirtualLedger(cost_model=_Cost())
    ledger.open_account("A", 10000.0, [VirtualLot("A:L1", _X, "L1", 400, _T0, 10.0)])
    ledger.open_account("B", 10000.0, [VirtualLot("B:L2", _X, "L2", 100, _T0, 10.0)])
    accounts = {
        "A": _vaccount("A", 10000.0, {_X: 400}),
        "B": _vaccount("B", 10000.0, {_X: 100}),
    }
    plan = merger.plan_netting([_order("B", OrderSide.BUY, 300)], accounts)
    assert dict(plan.physical_targets) == {_X: 700}
    order = plan.physical_orders()[0]
    assert order.side is OrderSide.BUY and order.target_shares.value == 700

    view = PhysicalLotView([PhysicalLot(_X, "L1", 400), PhysicalLot(_X, "L2", 100)])
    view.apply_buy_fill(ts_code=_X, source_lot_id="L3", shares=200, event_id="F1")
    batch = InternalSettlementBatch(
        batch_id="ONEISSUE",
        trade_date=_T1,
        intents=plan.intents,
        physical_fills=(PhysicalFillFact(_X, "buy", 200, 10.0, 5.0, "L3", "F1"),),
        physical_cash_before=20000.0,
        physical_cash_after=17995.0,
    )
    result = ledger.apply_settlement(
        batch,
        price_source=_source({_X: 10.0}),
        physical_view=view,
        identity_context=_context(),
        physical_positions={_X: {"shares": 700}},
    )
    assert result.source_shares_after == ((_X, "L1", 400), (_X, "L2", 100), (_X, "L3", 200))
    assert ledger.state_of("A").shares_of(_X) == 400
    assert ledger.state_of("B").shares_of(_X) == 300
    assert ledger.state_of("A").cash == 10000.0
    assert ledger.state_of("B").cash == 7995.0
    assert result.cash_total_after == 17995.0


# ────────────────────────── 拒绝后仍可继续（未破缺实例） ──────────────────────────


def test_input_rejection_does_not_poison_ledger():
    ledger = VirtualLedger(cost_model=_Cost())
    ledger.open_account("A", 2500.0)
    ledger.open_account("B", 7500.0)
    bad = InternalSettlementBatch(
        batch_id="BAD",
        trade_date=_T1,
        intents=(SleeveIntent("A", _X, "buy", 100, 0), SleeveIntent("B", _X, "buy", 100, 0)),
        physical_fills=(PhysicalFillFact(_X, "buy", 600, 10.0, 5.0, "L2", "F1"),),
        physical_cash_before=10000.0,
        physical_cash_after=3995.0,
    )
    with pytest.raises(VirtualLedgerError, match="超意图成交"):
        ledger.apply_settlement(
            bad,
            price_source=_source({_X: 10.0}),
            physical_view=PhysicalLotView([PhysicalLot(_X, "L2", 600)]),
            identity_context=_context(),
        )
    assert ledger.poisoned_reason is None
    # 修正后的批次可以正常提交（输入拒绝不是终态）
    good = InternalSettlementBatch(
        batch_id="GOOD",
        trade_date=_T1,
        intents=(SleeveIntent("A", _X, "buy", 100, 0), SleeveIntent("B", _X, "buy", 100, 0)),
        physical_fills=(PhysicalFillFact(_X, "buy", 200, 10.0, 5.0, "L2", "F1"),),
        physical_cash_before=10000.0,
        physical_cash_after=7995.0,
    )
    result = ledger.apply_settlement(
        good,
        price_source=_source({_X: 10.0}),
        physical_view=PhysicalLotView([PhysicalLot(_X, "L2", 200)]),
        identity_context=_context(),
    )
    assert result.applied is True
    assert ledger.state_of("A").shares_of(_X) == 100
    assert ledger.state_of("B").shares_of(_X) == 100
