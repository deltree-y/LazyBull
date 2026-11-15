# -*- coding: utf-8 -*-
"""P4 执行层合成探针：账户动作、候选循环与毛基准。"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from scripts.v2_p7.p2.account import ResearchAccount
from scripts.v2_p7.p2.types import MarketSnapshot, PriceObservation, PriceStatus
from scripts.v2_p7.p4.engine import (
    BatchPlan,
    max_lot_shares,
    run_benchmark,
    run_candidate,
)
from scripts.v2_p7.p4.metrics import block_bootstrap_upper, monthly_returns


class FakeData:
    dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
    dividends = pd.DataFrame(
        columns=[
            "ts_code",
            "end_date",
            "ex_date",
            "pay_date",
            "cash_div",
            "cash_div_tax",
            "stk_div",
        ]
    )

    def trade_days(self, start: str, end: str) -> list[str]:
        return [day for day in self.dates if start <= day <= end]

    def prev_trade_day(self, day: str) -> str:
        return self.dates[self.dates.index(day) - 1]

    def effective_trade_day(self, day: str) -> str:
        return next(item for item in self.dates if item >= day)


def _frame(code: str, values: dict) -> pd.DataFrame:
    row = {
        "code": code,
        "trade_date": values["day"],
        "open": values["open"],
        "close": values["close"],
        "pre_close": values["pre_close"],
        "amount": 1_000_000.0,
        "open_adj": values["open"],
        "close_adj": values["close"],
        "is_st": 0,
        "is_suspended": 0,
        "is_limit_down": 0,
    }
    return pd.DataFrame([row])


def _plan(selected=("AAA",), members=("AAA", "BBB")) -> BatchPlan:
    return BatchPlan(
        notice_id="1",
        signal_date="20240102",
        entry_date="20240103",
        flow_date="20240105",
        selected=tuple(
            {
                "notice_id": "1",
                "ts_code": code,
                "name": code,
                "rank": rank,
                "rank_proxy": 1.0,
                "circ_mv": 1.0,
                "adv20": 1_000_000.0,
            }
            for rank, code in enumerate(selected, start=1)
        ),
        selection_audit=(),
        membership=tuple(members),
        adv20={code: 1_000_000.0 for code in selected},
        capacity_1pct=6_000_000.0,
        capacity_5pct=30_000_000.0,
    )


def test_max_lot_shares_includes_fee_and_lot():
    assert max_lot_shares(Decimal("1005"), Decimal("10")) == 0
    assert max_lot_shares(Decimal("1006"), Decimal("10")) == 100
    assert max_lot_shares(Decimal("50000"), Decimal("10")) == 4900


def test_stock_dividend_adjusts_shares_not_cost():
    account = ResearchAccount(100_000)
    account.mark(MarketSnapshot("20240102", {}))
    account.submit_order("B", "AAA", "buy", 100, "20240102", "20240103")
    account.execute_fill("B", trade_date="20240103", quantity=100, trade_price=10, fee=0)
    account.mark(MarketSnapshot("20240103", {"AAA": PriceObservation(PriceStatus.OK, 10)}))
    item = account.apply_stock_dividend("STK", "AAA", Decimal("0.5"), "20240104")
    assert item.shares_before == 100
    assert item.shares_after == 150
    assert account.position("AAA").shares == 150
    assert account.position("AAA").cost_basis == Decimal("1000.00")


def test_candidate_buy_dividend_and_close_sell():
    data = FakeData()
    data.dividends = pd.DataFrame(
        [
            {
                "ts_code": "AAA",
                "end_date": "20231231",
                "ex_date": "20240104",
                "pay_date": "20240105",
                "cash_div": 0.5,
                "cash_div_tax": 0.625,
                "stk_div": 0.0,
            }
        ]
    )
    cache = {
        "20240102": _frame("AAA", {"day": "20240102", "open": 10, "close": 10, "pre_close": 10}),
        "20240103": _frame("AAA", {"day": "20240103", "open": 10, "close": 10.5, "pre_close": 10}),
        "20240104": _frame(
            "AAA", {"day": "20240104", "open": 10.4, "close": 10.8, "pre_close": 10.5}
        ),
        "20240105": _frame(
            "AAA", {"day": "20240105", "open": 10.8, "close": 11.0, "pre_close": 10.8}
        ),
    }
    run = run_candidate(
        data,
        {"1": _plan()},
        cache,
        principal=300_000,
        start_date="20240102",
        label="synthetic",
    )
    assert len(run.fills) == 2
    assert run.final_positions == ()
    assert run.corporate_actions[0]["cash_amount"] > 0
    assert run.levels.iloc[0] == 300_000
    assert run.levels.iloc[-1] > 300_000
    assert run.position_days >= 2


def test_candidate_defers_limit_down_sell_to_next_open():
    data = FakeData()
    cache = {
        "20240102": _frame("AAA", {"day": "20240102", "open": 10, "close": 10, "pre_close": 10}),
        "20240103": _frame("AAA", {"day": "20240103", "open": 10, "close": 10.5, "pre_close": 10}),
        "20240104": _frame(
            "AAA", {"day": "20240104", "open": 10.4, "close": 10.8, "pre_close": 10.5}
        ),
        "20240105": _frame(
            "AAA", {"day": "20240105", "open": 10.8, "close": 9.9, "pre_close": 10.8}
        ),
        "20240108": _frame("AAA", {"day": "20240108", "open": 9.8, "close": 9.8, "pre_close": 9.9}),
    }
    cache["20240105"].loc[0, "is_limit_down"] = 1
    run = run_candidate(
        data,
        {"1": _plan()},
        cache,
        principal=300_000,
        start_date="20240102",
        label="deferred",
    )
    assert len(run.fills) == 2
    assert run.final_positions == ()
    assert any(item["side"] == "sell" and item["status"] == "deferred" for item in run.exceptions)


def test_benchmark_keeps_missing_entry_weight_in_cash():
    data = FakeData()
    cache = {
        "20240102": pd.concat(
            [
                _frame("AAA", {"day": "20240102", "open": 10, "close": 10, "pre_close": 10}),
                _frame("BBB", {"day": "20240102", "open": 20, "close": 20, "pre_close": 20}),
            ]
        ),
        "20240103": pd.concat(
            [
                _frame("AAA", {"day": "20240103", "open": 10, "close": 10, "pre_close": 10}),
                _frame("BBB", {"day": "20240103", "open": 20, "close": 20, "pre_close": 20}),
            ]
        ),
        "20240104": pd.concat(
            [
                _frame("AAA", {"day": "20240104", "open": 10, "close": 10, "pre_close": 10}),
                _frame("BBB", {"day": "20240104", "open": 20, "close": 21, "pre_close": 20}),
            ]
        ),
        "20240105": pd.concat(
            [
                _frame("AAA", {"day": "20240105", "open": 10, "close": 10, "pre_close": 10}),
                _frame("BBB", {"day": "20240105", "open": 20, "close": 20, "pre_close": 21}),
            ]
        ),
    }
    cache["20240103"].loc[cache["20240103"]["code"].eq("AAA"), "open_adj"] = float("nan")
    benchmark = run_benchmark(
        data,
        {"1": _plan(members=("AAA", "BBB"))},
        cache,
        principal=300_000,
        start_date="20240102",
    )
    assert benchmark.event_rows[0]["uninvested_weight"] == pytest.approx(0.5)
    assert benchmark.levels.loc["20240103"] == pytest.approx(300_000)
    assert benchmark.levels.loc["20240104"] == pytest.approx(307_500)


def test_capacity_is_informational_not_a_gate():
    from scripts.v2_p7.p4.run import _decide

    table1 = {
        "es1": {"annualized_pp": 3.0},
        "cost_drag_pp_per_year": 0.3,
        "metrics": {"cagr": 0.04},
        "capacity": {"1pct_cny": 1_000.0, "5pct_cny": 5_000.0},
        "missing": {"max_unresolved_weight": 0.0, "missing_position_day_ratio": 0.0},
    }
    result = _decide(
        table1,
        {"estimable": True, "point_max": 0.1},
        {"d_maxdd_pp": 2.0, "p_cagr_pp": 1.0, "pass": True},
    )
    assert result["verdict"] == "continue_to_formal_review"
    assert result["es"]["capacity"]["is_informational"] is True
    assert result["reasons"] == ["all_es_pass_capacity_informational"]


def test_bootstrap_and_monthly_baseline_are_deterministic():
    left = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0, -0.02, 0.01, 0.04])
    right = pd.Series([0.00, 0.01, -0.02, 0.02, 0.01, -0.01, 0.02, 0.03])
    first = block_bootstrap_upper(left.to_numpy(), right.to_numpy(), block=2)
    second = block_bootstrap_upper(left.to_numpy(), right.to_numpy(), block=2)
    assert first == second
    levels = pd.Series(
        [100.0, 110.0, 120.0],
        index=["20240115", "20240131", "20240229"],
    )
    monthly = monthly_returns(levels, "20240115")
    assert monthly.iloc[0] == pytest.approx(0.1)
    assert monthly.iloc[1] == pytest.approx(120 / 110 - 1)
