"""只读时点归因的合成核验，不读取真实经济包。"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.polyester_timing_diagnostic import (
    CODES,
    aggregate_difference,
    collapse_signals,
    compare_signals,
    planned_execution_comparison,
    source_legs,
    transaction_context,
    wealth_difference,
)


def records() -> list[dict]:
    return [
        {
            "source_id": str(index),
            "observation_date": month + "15",
            "publication_date": month + "16",
            "available_date": month + "17",
            "series_id": "same",
            "source_kind": "synthetic",
            "quality_usable": True,
            "strict_usable": True,
            "poy": poy,
            "pta": 100,
            "meg": 100,
        }
        for index, (month, poy) in enumerate(
            zip(["202103", "202102", "202101", "202012"], [140, 130, 120, 150])
        )
    ]


def signal(**updates: object) -> dict:
    return {
        "signal_date": "20210401",
        "source_ids": ["0", "1", "2", "3"],
        "months": ["202103", "202102", "202101", "202012"],
        "availability_cutoff": "20210401",
        "input_known": True,
        "eligible": True,
        "reason": "spread_improving",
        "spread": 21.0,
        "prior_three_mean": 43 / 3,
        "unknown_on": True,
        "delay_trading_days": 0,
        **updates,
    }


def test_shared_signal_collapses_arrays_and_rejects_disagreement() -> None:
    frame = pd.DataFrame([{**signal(), "ts_code": code} for code in CODES])
    frame["source_ids"] = frame.source_ids.map(np.array)
    assert len(collapse_signals(frame)) == 1
    frame.loc[1, "eligible"] = False
    with pytest.raises(ValueError, match="不一致"):
        collapse_signals(frame)


def test_source_legs_verifies_saved_dates_and_arithmetic() -> None:
    frame = pd.DataFrame([signal()])
    result = source_legs(frame, records(), "main")
    assert result.spread.tolist() == [21, 11, 1, 31]
    frame.loc[0, "prior_three_mean"] = 14
    with pytest.raises(ValueError, match="前三月均值"):
        source_legs(frame, records(), "main")


def test_source_equal_to_cutoff_is_unavailable() -> None:
    data = records()
    data[0]["available_date"] = "20210401"
    with pytest.raises(ValueError, match="时点"):
        source_legs(pd.DataFrame([signal()]), data, "main")


def test_source_unknown_does_not_imply_known_negative() -> None:
    main = pd.DataFrame([signal()])
    delayed = pd.DataFrame(
        [
            signal(
                source_ids=["0", None, "2", "3"],
                input_known=False,
                eligible=False,
                prior_three_mean=None,
                reason="missing_prior_month",
            )
        ]
    )
    legs = pd.concat(
        [
            source_legs(main, records(), "main"),
            source_legs(delayed, records(), "delay5"),
            source_legs(main, records(), "source_strict"),
        ]
    )
    result = compare_signals(main, delayed, main, legs)
    assert result.category.iloc[0] == "main_known_delay_unknown"
    assert result.changed_lags.iloc[0] == [1]
    assert result.decision_changed.iloc[0]


def test_known_reversal_is_distinct_from_unknown_transition() -> None:
    data = records() + [
        {
            **records()[0],
            "source_id": "older",
            "poy": 125,
            "observation_date": "20210310",
            "publication_date": "20210311",
            "available_date": "20210312",
        }
    ]
    main = pd.DataFrame([signal()])
    delay = pd.DataFrame(
        [
            signal(
                source_ids=["older", "1", "2", "3"],
                eligible=False,
                spread=6,
                reason="spread_not_improving",
                availability_cutoff="20210325",
            )
        ]
    )
    legs = pd.concat(
        [
            source_legs(main, data, "main"),
            source_legs(delay, data, "delay5"),
            source_legs(main, data, "source_strict"),
        ]
    )
    result = compare_signals(main, delay, main, legs)
    assert result.category.iloc[0] == "both_known_main_on_delay_off"
    assert result.main_observation_date.iloc[0] == "20210315"
    assert result.delay5_observation_date.iloc[0] == "20210310"


def daily(values: list[float], fees: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20201231", "20210129", "20210201"],
            "total_value": values,
            "cash": values,
            "market_value": 0.0,
            "receivable": 0.0,
            "dividend_haircut_liability": 0.0,
            "fees": fees,
            "buy_amount": 0.0,
            "sell_amount": 0.0,
            "holding_count": 0,
        }
    )


def test_pnl_difference_telescopes_and_fee_split_is_additive() -> None:
    frame, summary = wealth_difference(
        daily([200, 210, 208], [0, 1, 2]), daily([200, 205, 210], [0, 2, 0]), 1000
    )
    assert frame.net_pnl_difference.tolist() == [0, 5, -7]
    assert frame.fee_contribution.tolist() == [0, 1, -2]
    assert summary["ending_wealth_difference_yuan"] == -2
    assert summary["gross_pnl_difference_yuan"] == -1
    assert summary["fee_contribution_yuan"] == -1
    monthly = aggregate_difference(frame, 6)
    assert monthly.net_pnl_difference.tolist() == [0, 5, -7]
    assert monthly.total_capital_return_contribution_pp.sum() == pytest.approx(-0.2)
    assert aggregate_difference(frame, 4).net_pnl_difference.tolist() == [0, -2]


def test_pnl_rejects_date_intersection_or_capital_scaling() -> None:
    main, delay = daily([200, 210, 208], [0, 1, 2]), daily([200, 205, 210], [0, 2, 0])
    with pytest.raises(ValueError, match="日轴"):
        wealth_difference(main, delay.iloc[:-1], 1000)
    with pytest.raises(ValueError, match="实际 B 初始资本"):
        wealth_difference(main, delay, 2000)


def test_small_buy_after_existing_holding_is_topup_not_new_position() -> None:
    days = daily([200, 210, 208], [0, 1, 2])
    holdings = pd.DataFrame([{"trade_date": "20201231", "ts_code": "603225.SH", "shares": 1000}])
    trades = pd.DataFrame(
        [
            {
                "trade_date": "20210129",
                "signal_date": "20210128",
                "ts_code": "603225.SH",
                "side": "buy",
                "shares": 100,
                "amount": 1028,
            },
            {
                "trade_date": "20210129",
                "signal_date": "20210128",
                "ts_code": "601233.SH",
                "side": "buy",
                "shares": 1000,
                "amount": 10000,
            },
        ]
    )
    result = transaction_context(trades, holdings, days, "main")
    assert result.transaction_role.tolist() == ["existing_position_topup", "new_position"]
    assert result.prior_close_shares.tolist() == [1000, 0]


def test_five_day_information_cutoff_keeps_saved_execution_schedule() -> None:
    main = pd.DataFrame(
        [
            {
                "kind": "planned",
                "trade_date": "20210401",
                "ts_code": code,
                "execution_date": "20210402",
                "eligible": True,
                "notional": 10000,
            }
            for code in CODES
        ]
    )
    delay = copy.deepcopy(main)
    delay["eligible"] = False
    delay["notional"] = 0
    result = planned_execution_comparison(main, delay)
    assert result.same_execution_date.all()
