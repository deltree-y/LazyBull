# -*- coding: utf-8 -*-
"""P6.5 T1：I1/I2/I3 三恒等式与贡献 NAV（全部显式合成输入）。

口径依据 ``docs/contracts/netting_freeze.md`` F25 §4：
- I1 货币绝对差 ≤1e-6，逐袖无借用 + 费用后自有现金非负；
- I2 按 ``(ts_code, 物理来源 lot_id)`` 汇总的**整数股**精确相等，不比条目数；
- I3 直接贡献 NAV（分母 = 物理初始资金）绝对差 ≤1e-6，1 ULP 不要求逐位。
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.core.accounting.physical_lot_view import (
    PhysicalLot,
    PhysicalLotView,
)
from src.lazybull.v2.core.accounting.reconciliation import (
    CASH_TOLERANCE,
    NAV_TOLERANCE,
    DailyIdentityReport,
    IdentityContext,
    ReconciliationError,
    check_cash_identity,
    check_holdings_identity,
    check_nav_identity,
    run_daily_identities,
    sleeve_total_value,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    InternalTransfer,
    LotAllocation,
    PriceSource,
    VirtualLedger,
    VirtualLedgerError,
    VirtualLot,
    VirtualSleeveState,
)

_D0 = TradeDate(pd.Timestamp("2026-01-05").date())
_D1 = TradeDate(pd.Timestamp("2026-01-06").date())
_X = "600000.SH"
_Y = "000001.SZ"


def _lot(
    virtual_lot_id: str,
    *,
    source_lot_id: str = "L1",
    shares: int = 100,
    ts_code: str = _X,
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


def _state(
    sleeve_id: str,
    cash: float,
    lots: tuple[VirtualLot, ...] = (),
    credit: float = 0.0,
) -> VirtualSleeveState:
    return VirtualSleeveState(
        sleeve_id=sleeve_id,
        cash=cash,
        borrowed_credit=credit,
        lots=lots,
    )


def _context(
    sleeve_ids: tuple[str, ...] = ("A", "B"),
    *,
    date: TradeDate = _D1,
    price_source_id: str = "syn:p65",
) -> IdentityContext:
    return IdentityContext(
        trade_date=date,
        host="pytest_synthetic",
        run_id="R1",
        sleeve_ids=sleeve_ids,
        price_source_id=price_source_id,
        last_settlement_id="S1",
    )


def _source(
    prices: dict[str, float] | None = None,
    *,
    date: TradeDate = _D1,
    source_id: str = "syn:p65",
) -> PriceSource:
    resolved = {_X: 12.0} if prices is None else prices
    return PriceSource(as_of=date, source_id=source_id, prices=resolved)


# ────────────────────────── I1 ──────────────────────────


def test_i1_accepts_exact_and_boundary_difference() -> None:
    context = _context(("A",))
    exact = check_cash_identity(
        physical_cash=50000.0,
        virtual_states={"A": _state("A", 50000.0)},
        context=context,
    )
    assert exact.diff == 0.0
    assert exact.tolerance == CASH_TOLERANCE

    boundary = check_cash_identity(
        physical_cash=0.0,
        virtual_states={"A": _state("A", CASH_TOLERANCE)},
        context=context,
    )
    assert boundary.diff == CASH_TOLERANCE


def test_i1_rejects_difference_beyond_boundary() -> None:
    with pytest.raises(ReconciliationError, match=r"\[I1\]") as caught:
        check_cash_identity(
            physical_cash=0.0,
            virtual_states={"A": _state("A", 1.0000001e-6)},
            context=_context(("A",)),
        )
    assert caught.value.identity == "I1"
    assert "容差=1e-06 单位=元" in str(caught.value)


def test_i1_rejects_single_sleeve_overdraft_even_if_portfolio_non_negative() -> None:
    states = {"A": _state("A", -100.0), "B": _state("B", 200.0)}
    with pytest.raises(ReconciliationError, match="费用后自有虚拟现金为负") as caught:
        check_cash_identity(physical_cash=100.0, virtual_states=states, context=_context())
    assert caught.value.identity == "I1"
    assert "袖 A" in str(caught.value)


def test_i1_rejects_borrowed_credit_pair_that_offsets_to_zero() -> None:
    states = {
        "A": _state("A", 100.0, credit=100.0),
        "B": _state("B", 0.0, credit=-100.0),
    }
    with pytest.raises(ReconciliationError, match="borrowed_credit") as caught:
        check_cash_identity(physical_cash=200.0, virtual_states=states, context=_context())
    assert caught.value.identity == "I1"


def test_i1_rejects_non_finite_cash() -> None:
    # 快照构造期即拒绝 NaN（防「恒等式被浮点化」），校验器侧另有有限性守卫
    with pytest.raises(VirtualLedgerError, match="必须为有限数"):
        _state("A", float("nan"))
    with pytest.raises(ReconciliationError, match="非有限"):
        check_cash_identity(
            physical_cash=float("nan"),
            virtual_states={"A": _state("A", 0.0)},
            context=_context(("A",)),
        )


# ────────────────────────── I2 ──────────────────────────


def test_i2_accepts_same_source_split_across_two_sleeves() -> None:
    states = {
        "A": _state("A", 0.0, (_lot("A:L1", shares=300),)),
        "B": _state("B", 0.0, (_lot("B:L1", shares=100),)),
    }
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    result = check_holdings_identity(
        physical_view=view, virtual_states=states, context=_context()
    )
    assert result.diff == 0
    assert int(result.left) == 400
    assert int(result.right) == 400
    # 逐来源键明细是 I2 证据（不是条目数证据）
    assert result.sources == ((_X, "L1", 400),)


def test_i2_accepts_multiple_sources_bought_same_day() -> None:
    states = {
        "A": _state("A", 0.0, (_lot("A:L1", shares=100), _lot("A:L2", source_lot_id="L2"))),
        "B": _state("B", 0.0, (_lot("B:L3", source_lot_id="L3", shares=50),)),
    }
    view = PhysicalLotView(
        [PhysicalLot(_X, "L1", 100), PhysicalLot(_X, "L2", 100), PhysicalLot(_X, "L3", 50)]
    )
    result = check_holdings_identity(
        physical_view=view, virtual_states=states, context=_context()
    )
    assert int(result.left) == 250
    assert int(result.right) == 250


def test_i2_rejects_same_total_with_wrong_source_attribution() -> None:
    states = {"A": _state("A", 0.0, (_lot("A:L2", source_lot_id="L2", shares=200),))}
    view = PhysicalLotView([PhysicalLot(_X, "L1", 100), PhysicalLot(_X, "L2", 100)])
    with pytest.raises(ReconciliationError, match="来源键集不一致") as caught:
        check_holdings_identity(
            physical_view=view, virtual_states=states, context=_context(("A",))
        )
    assert caught.value.identity == "I2"
    assert "L1" in str(caught.value)


def test_i2_rejects_virtual_source_key_absent_from_physical() -> None:
    states = {
        "A": _state("A", 0.0, (_lot("A:L1", shares=100), _lot("A:L9", source_lot_id="L9"))),
    }
    view = PhysicalLotView([PhysicalLot(_X, "L1", 100), PhysicalLot(_X, "L2", 100)])
    with pytest.raises(ReconciliationError, match=r"虚拟有而物理无=\[\('600000.SH', 'L9'\)\]"):
        check_holdings_identity(
            physical_view=view, virtual_states=states, context=_context(("A",))
        )


def test_i2_rejects_single_source_plus_one_share() -> None:
    states = {"A": _state("A", 0.0, (_lot("A:L1", shares=401),))}
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    with pytest.raises(ReconciliationError, match="来源批次股数不精确相等") as caught:
        check_holdings_identity(
            physical_view=view, virtual_states=states, context=_context(("A",))
        )
    assert caught.value.identity == "I2"
    message = str(caught.value)
    assert "左(虚拟)=401 股" in message
    assert "右(物理伴随视图)=400 股" in message
    assert "差=1 股" in message
    assert "容差=0 股" in message


def test_i2_rejects_same_totals_but_swapped_source_shares() -> None:
    lots = (_lot("A:L1", shares=150), _lot("A:L2", source_lot_id="L2", shares=50))
    states = {"A": _state("A", 0.0, lots)}
    view = PhysicalLotView([PhysicalLot(_X, "L1", 50), PhysicalLot(_X, "L2", 150)])
    with pytest.raises(ReconciliationError, match="来源批次股数不精确相等") as caught:
        check_holdings_identity(
            physical_view=view, virtual_states=states, context=_context(("A",))
        )
    assert caught.value.identity == "I2"


def test_i2_does_not_compare_lot_entry_count() -> None:
    states = {
        "A": _state("A", 0.0, (_lot("A:L1:a", shares=100), _lot("A:L1:b", shares=100))),
    }
    view = PhysicalLotView([PhysicalLot(_X, "L1", 200)])
    result = check_holdings_identity(
        physical_view=view, virtual_states=states, context=_context(("A",))
    )
    assert result.diff == 0
    assert len(states["A"].lots) == 2


def test_i2_rejects_duplicate_virtual_acquisition_identity() -> None:
    states = {"A": _state("A", 0.0, (_lot("A:L1"), _lot("A:L1", shares=100)))}
    view = PhysicalLotView([PhysicalLot(_X, "L1", 200)])
    with pytest.raises(ReconciliationError, match="虚拟取得批次身份重复") as caught:
        check_holdings_identity(
            physical_view=view, virtual_states=states, context=_context(("A",))
        )
    assert caught.value.identity == "I2"


def test_i2_rejects_view_drift_from_engine_positions() -> None:
    states = {"A": _state("A", 0.0, (_lot("A:L1", shares=400),))}
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400)])
    with pytest.raises(ReconciliationError, match="右值前置破缺") as caught:
        check_holdings_identity(
            physical_view=view,
            virtual_states=states,
            context=_context(("A",)),
            physical_positions={_X: {"shares": 300}},
        )
    assert caught.value.identity == "I2"


# ────────────────────────── I3 ──────────────────────────


def _s1_states() -> dict[str, VirtualSleeveState]:
    return {
        "A": _state("A", 51200.0, (_lot("A:L1", shares=300, price=12.0),)),
        "B": _state("B", 48800.0, (_lot("S1#0", shares=100, price=12.0),)),
    }


def test_i3_s1_contribution_nav_is_direct_sum() -> None:
    result = check_nav_identity(
        virtual_states=_s1_states(),
        physical_total_value=104800.0,
        physical_initial_capital=100000.0,
        price_source=_source({_X: 12.0}),
        context=_context(),
    )
    assert result.left == 0.548 + 0.500
    assert result.right == 104800.0 / 100000.0
    assert result.diff == 0.0
    assert result.unit == "无量纲"
    assert sleeve_total_value(_s1_states()["A"], _source({_X: 12.0})) == 54800.0


def test_i3_accepts_one_ulp_difference_without_bitwise_requirement() -> None:
    states = {
        "A": _state("A", 48000.0, (_lot("A:L1", shares=200, price=11.0),)),
        "B": _state("B", 47995.0, (_lot("B:L1", shares=200, price=11.0),)),
    }
    per_sleeve = 50200.0 / 100000.0 + 50195.0 / 100000.0
    physical_nav = 100395.0 / 100000.0
    assert per_sleeve != physical_nav  # 1 ULP 差异是设计已知事实
    result = check_nav_identity(
        virtual_states=states,
        physical_total_value=100395.0,
        physical_initial_capital=100000.0,
        price_source=_source({_X: 11.0}),
        context=_context(),
    )
    assert result.diff != 0.0
    assert abs(result.diff) <= math.ulp(physical_nav)
    assert abs(result.diff) <= NAV_TOLERANCE


def test_i3_rejects_contribution_drift_beyond_tolerance() -> None:
    with pytest.raises(ReconciliationError, match="Σ NAV_i_contrib") as caught:
        check_nav_identity(
            virtual_states=_s1_states(),
            physical_total_value=104900.0,
            physical_initial_capital=100000.0,
            price_source=_source({_X: 12.0}),
            context=_context(),
        )
    assert caught.value.identity == "I3"


def test_i3_rejects_invalid_initial_capital() -> None:
    with pytest.raises(ReconciliationError, match="初始资金"):
        check_nav_identity(
            virtual_states=_s1_states(),
            physical_total_value=104800.0,
            physical_initial_capital=0.0,
            price_source=_source({_X: 12.0}),
            context=_context(),
        )


def test_i3_rejects_cross_day_price_source() -> None:
    with pytest.raises(ReconciliationError, match="同日同价源"):
        check_nav_identity(
            virtual_states=_s1_states(),
            physical_total_value=104800.0,
            physical_initial_capital=100000.0,
            price_source=_source({_X: 12.0}, date=_D0),
            context=_context(),
        )


def test_i3_rejects_missing_price() -> None:
    with pytest.raises(ReconciliationError, match="缺少") as caught:
        check_nav_identity(
            virtual_states=_s1_states(),
            physical_total_value=104800.0,
            physical_initial_capital=100000.0,
            price_source=_source({}),
            context=_context(),
        )
    assert caught.value.identity == "I3"


def test_i3_value_sum_is_secondary_evidence() -> None:
    states = {"A": _state("A", 600000000.0), "B": _state("B", 400000000.5)}
    with pytest.raises(ReconciliationError, match="辅证") as caught:
        check_nav_identity(
            virtual_states=states,
            physical_total_value=1000000000.0,
            physical_initial_capital=1000000000.0,
            price_source=_source({_X: 12.0}),
            context=_context(),
        )
    assert caught.value.identity == "I3"


# ────────────────────────── 逐日编排 ──────────────────────────


def _run_all(states: dict[str, VirtualSleeveState]) -> DailyIdentityReport:
    return run_daily_identities(
        physical_cash=100000.0,
        physical_total_value=104800.0,
        physical_initial_capital=100000.0,
        physical_positions={_X: {"shares": 400}},
        physical_view=PhysicalLotView([PhysicalLot(_X, "L1", 400)]),
        virtual_states=states,
        price_source=_source({_X: 12.0}),
        context=_context(),
    )


def test_daily_identities_run_in_fixed_order() -> None:
    report = _run_all(_s1_states())
    assert report.identities == ("I1", "I2", "I3")
    assert report.price_source_id == "syn:p65"
    assert report.sleeve_ids == ("A", "B")
    assert report.last_settlement_id == "S1"


def test_end_to_end_transfer_then_identities() -> None:
    """首个虚拟账写点的真实产物必须能直接喂给三恒等式（§7.2-1「同批」口径）。"""
    ledger = VirtualLedger()
    ledger.open_account("A", 50000.0, [_lot("A:L1", shares=400, price=10.0)])
    ledger.open_account("B", 50000.0)
    applied = ledger.transfer(
        InternalTransfer(
            settlement_id="S1",
            trade_date=_D1,
            ts_code=_X,
            from_sleeve="A",
            to_sleeve="B",
            price=12.0,
            allocations=(LotAllocation("A:L1", 100),),
        ),
        price_source=_source({_X: 12.0}),
    )
    assert applied is True

    report = run_daily_identities(
        physical_cash=100000.0,
        physical_total_value=104800.0,
        physical_initial_capital=100000.0,
        physical_positions={_X: {"shares": 400}},
        physical_view=PhysicalLotView([PhysicalLot(_X, "L1", 400)]),
        virtual_states=ledger.snapshot(),
        price_source=_source({_X: 12.0}),
        context=_context(),
    )
    assert report.identities == ("I1", "I2", "I3")
    assert report.results[0].left == 100000.0
    assert report.results[1].sources == ((_X, "L1", 400),)
    assert report.results[2].left == 0.548 + 0.5
    assert report.last_settlement_id == "S1"


def test_daily_identities_report_all_breakages() -> None:
    """联动故障必须报出全部破缺编号（设计 §7.2-5），且仍在同一次调用内终止。"""
    broken = {
        "A": _state("A", 1.0, (_lot("A:L9", source_lot_id="L9", shares=300),)),
        "B": _state("B", 48800.0, (_lot("S1#0", shares=100, price=12.0),)),
    }
    with pytest.raises(ReconciliationError) as caught:
        _run_all(broken)
    error = caught.value
    assert error.identities == ("I1", "I2", "I3")
    assert error.identity == "I1"
    message = str(error)
    for identity in ("I1", "I2", "I3"):
        assert f"[{identity}]" in message
    assert "联动破缺" in message


def test_daily_identities_report_single_breakage() -> None:
    with pytest.raises(ReconciliationError) as caught:
        run_daily_identities(
            physical_cash=100001.0,  # 仅物理现金差 1 元 → 只有 I1 破缺
            physical_total_value=104800.0,
            physical_initial_capital=100000.0,
            physical_positions={_X: {"shares": 400}},
            physical_view=PhysicalLotView([PhysicalLot(_X, "L1", 400)]),
            virtual_states=_s1_states(),
            price_source=_source({_X: 12.0}),
            context=_context(),
        )
    assert caught.value.identities == ("I1",)
    assert "联动破缺" not in str(caught.value)
