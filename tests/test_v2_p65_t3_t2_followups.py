# -*- coding: utf-8 -*-
"""P6.5 T3 顺带处置的 T2 收口 P3 清单（实现细节部分）。

逐条对应 ``docs/review/v2_p65_t2/t2_closure_20261109.md`` §6：

1. ``_commit_settlement`` 派生键防御必须先于任何写入（拒绝不得半提交）；
2. ``_fill_increments`` 零增量卖出意图不得误拒合法买入批次（与 S1 路径同口径）；
3. 上下文价源标识与实际结算价源必须交叉校验；
4. 买方现金不足（转账/成交两条路径）专用负例；
6. 非 ``TSCode`` 键/非 ``Weight`` 值必须以 ``VirtualMergerError`` 拒绝（不得 AttributeError）；
7. 非法批次类型必须先于 ``batch.batch_id`` 属性访问拒绝；
8. ``plan_netting → InternalSettlementBatch → apply_settlement`` 端到端接缝。

第 5 条已就地修正 T2 用例的 ``pytest.approx``；第 9/10 条属契约待补（上报，不改契约）。
"""

from __future__ import annotations

from types import MappingProxyType

import pandas as pd
import pytest

from src.lazybull.v2.common.types import (
    Account,
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
_T0 = TradeDate(pd.Timestamp("2026-01-05").date())
_T1 = TradeDate(pd.Timestamp("2026-01-06").date())


class _Cost:
    def calculate_buy_cost(self, amount: float) -> float:
        return max(amount * 0.0003, 5.0)


def _context(sleeves=("A", "B"), date=_T1, source_id="syn:p65") -> IdentityContext:
    return IdentityContext(
        trade_date=date,
        host="pytest_t3_followups",
        run_id="R1",
        sleeve_ids=tuple(sleeves),
        price_source_id=source_id,
    )


def _source(price: float = 12.0, date=_T1, source_id="syn:p65") -> PriceSource:
    return PriceSource(as_of=date, source_id=source_id, prices={_X: price})


def _s1_ledger() -> VirtualLedger:
    ledger = VirtualLedger(cost_model=_Cost())
    ledger.open_account("A", 50000.0, [VirtualLot("A:L1", _X, "L1", 400, _T0, 10.0)])
    ledger.open_account("B", 50000.0)
    return ledger


def _s1_batch(shares: int = 100) -> InternalSettlementBatch:
    return InternalSettlementBatch(
        batch_id="S1",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "sell", 400 - shares, 400),
            SleeveIntent("B", _X, "buy", shares, 0),
        ),
        transfers=(TransferFact(_X, "A", "B", shares, 12.0),),
        physical_cash_before=100000.0,
        physical_cash_after=100000.0,
    )


def _s1_view() -> PhysicalLotView:
    return PhysicalLotView([PhysicalLot(_X, "L1", 400)])


# ── 1. 派生键防御零半提交 ──


def test_commit_settlement_derived_key_defense_has_no_partial_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.lazybull.v2.core.accounting.settlement_batch as batch_mod

    ledger = _s1_ledger()
    original = batch_mod._apply_batch_transfers

    def poisoned(ledger_ref, candidate, batch, view):
        result = original(ledger_ref, candidate, batch, view)
        # 防御分支触发：在候选账已改、提交点未执行之间插入外部占用
        ledger_ref._settlements["S1#T0"] = ("__batch_transfer__", "other")
        return result

    monkeypatch.setattr(batch_mod, "_apply_batch_transfers", poisoned)
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
    assert ledger.last_settlement_id is None


# ── 2. 零增量卖出意图不阻断买入成交 ──


def test_zero_delta_sell_intent_does_not_block_buy_fill() -> None:
    ledger = VirtualLedger(cost_model=_Cost())
    ledger.open_account("A", 5000.0)
    ledger.open_account("B", 5000.0, [VirtualLot("B:L1", _X, "L1", 50, _T0, 10.0)])
    view = PhysicalLotView([PhysicalLot(_X, "L1", 50)])
    view.apply_buy_fill(ts_code=_X, source_lot_id="L2", shares=100, event_id="F1")
    batch = InternalSettlementBatch(
        batch_id="ZD",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _X, "sell", 50, 50),  # 目标=当前 ⇒ 零增量（side=sell）
        ),
        physical_fills=(PhysicalFillFact(_X, "buy", 100, 10.0, 5.0, "L2", "F1"),),
        physical_cash_before=10000.0,
        physical_cash_after=8995.0,
    )
    result = ledger.apply_settlement(
        batch,
        price_source=_source(price=10.0),
        physical_view=view,
        identity_context=_context(),
    )
    assert [(item.sleeve_id, item.allocated_shares) for item in result.share_allocations] == [
        ("A", 100)
    ]
    assert result.voided_intents == ()
    assert ledger.state_of("A").shares_of(_X) == 100
    assert ledger.state_of("B").shares_of(_X) == 50
    assert result.cash_total_after == 8995.0


# ── 3. 上下文价源标识交叉校验 ──


def test_context_price_source_id_must_match_price_source() -> None:
    ledger = _s1_ledger()
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="上下文价源标识"):
        ledger.apply_settlement(
            _s1_batch(),
            price_source=_source(source_id="syn:other"),
            physical_view=_s1_view(),
            identity_context=_context(source_id="syn:p65"),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()


# ── 4. 买方现金不足（转账 / 成交两条路径） ──


def test_transfer_buyer_cash_shortfall_is_rejected() -> None:
    ledger = VirtualLedger(cost_model=_Cost())
    ledger.open_account("A", 50000.0, [VirtualLot("A:L1", _X, "L1", 400, _T0, 10.0)])
    ledger.open_account("B", 0.0)
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="虚拟买入现金不足"):
        ledger.apply_settlement(
            _s1_batch(),
            price_source=_source(price=12.0),
            physical_view=_s1_view(),
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()


def test_fill_sleeve_cash_shortfall_is_rejected() -> None:
    ledger = VirtualLedger(cost_model=_Cost())
    ledger.open_account("A", 100.0)
    ledger.open_account("B", 9900.0)
    view = PhysicalLotView()
    view.apply_buy_fill(ts_code=_X, source_lot_id="L2", shares=200, event_id="F1")
    batch = InternalSettlementBatch(
        batch_id="SHORT",
        trade_date=_T1,
        intents=(
            SleeveIntent("A", _X, "buy", 100, 0),
            SleeveIntent("B", _X, "buy", 300, 0),
        ),
        physical_fills=(PhysicalFillFact(_X, "buy", 200, 10.0, 5.0, "L2", "F1"),),
        physical_cash_before=10000.0,
        physical_cash_after=7995.0,
    )
    before = dict(ledger.snapshot())
    with pytest.raises(VirtualLedgerError, match="费用后虚拟现金不足"):
        ledger.apply_settlement(
            batch,
            price_source=_source(price=10.0),
            physical_view=view,
            identity_context=_context(),
        )
    assert dict(ledger.snapshot()) == before
    assert ledger.batch_ids() == ()


# ── 6. 非 TSCode 键 / 非 Weight 值 ──


def _account() -> Account:
    return Account(
        cash=Money(10000.0),
        positions=MappingProxyType({}),
        total_value=Money(10000.0),
        nav=1.0,
    )


def test_non_tscode_key_is_value_error_not_attribute_error() -> None:
    merger = MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5})
    bad_key = {
        "A": TargetPortfolio(
            sleeve_id="A",
            target_weights=MappingProxyType({_X: Weight(1.0)}),
            as_of_date=_T1,
        ),
        "B": TargetPortfolio(
            sleeve_id="B",
            target_weights=MappingProxyType({_X: Weight(1.0)}),
            as_of_date=_T1,
        ),
    }
    with pytest.raises(VirtualMergerError, match="TSCode"):
        merger.merge(bad_key, _account())

    bad_value = {
        "A": TargetPortfolio(
            sleeve_id="A",
            target_weights=MappingProxyType({TSCode(_X): 1.0}),
            as_of_date=_T1,
        ),
        "B": TargetPortfolio(
            sleeve_id="B",
            target_weights=MappingProxyType({TSCode(_X): Weight(1.0)}),
            as_of_date=_T1,
        ),
    }
    with pytest.raises(VirtualMergerError, match="Weight"):
        merger.merge(bad_value, _account())


# ── 7. 非法批次类型的拒绝顺序 ──


def test_apply_settlement_rejects_non_batch_type_with_context() -> None:
    ledger = _s1_ledger()
    with pytest.raises(VirtualLedgerError, match="InternalSettlementBatch"):
        ledger.apply_settlement(
            object(),
            price_source=_source(),
            physical_view=_s1_view(),
            identity_context=_context(),
        )
    assert ledger.batch_ids() == ()


# ── 8. plan_netting → batch → apply_settlement 端到端接缝 ──


def test_plan_netting_batch_seam_end_to_end() -> None:
    merger = MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5})
    ledger = VirtualLedger(cost_model=_Cost())
    ledger.open_account("A", 2500.0)
    ledger.open_account("B", 7500.0)
    orders = (
        Order(
            TSCode(_X),
            OrderSide.BUY,
            Shares(100),
            OrderType.MARKET_OPEN,
            OrderReason.REBALANCE,
            "A",
        ),
        Order(
            TSCode(_X),
            OrderSide.BUY,
            Shares(300),
            OrderType.MARKET_OPEN,
            OrderReason.REBALANCE,
            "B",
        ),
    )
    plan = merger.plan_netting(orders, ledger.as_virtual_accounts(_source(price=10.0)))
    assert plan.physical_targets[_X] == 400
    assert [item.sleeve_id for item in plan.intents] == ["A", "B"]

    view = PhysicalLotView()
    view.apply_buy_fill(ts_code=_X, source_lot_id="L2", shares=200, event_id="F1")
    batch = InternalSettlementBatch(
        batch_id="SEAM",
        trade_date=_T1,
        intents=plan.intents,
        physical_fills=(PhysicalFillFact(_X, "buy", 200, 10.0, 5.0, "L2", "F1"),),
        transfers=plan.transfer_facts({}),
        physical_cash_before=10000.0,
        physical_cash_after=7995.0,
    )
    result = ledger.apply_settlement(
        batch,
        price_source=_source(price=10.0),
        physical_view=view,
        identity_context=_context(),
    )
    assert [(item.sleeve_id, item.allocated_shares) for item in result.share_allocations] == [
        ("A", 50),
        ("B", 150),
    ]
    assert result.cash_total_after == 7995.0
    assert ledger.state_of("A").cash == 2000.0
    assert ledger.state_of("B").cash == 5995.0
