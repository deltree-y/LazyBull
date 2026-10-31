# -*- coding: utf-8 -*-
"""v2 P4 T4 双宿主比较器与执行配置前置测试。"""

import pandas as pd
import pytest

from scripts.v2_p4.t4_compare_hosts import (
    TRADE_COLUMNS,
    canonical_nav,
    canonical_positions,
    canonical_trades,
    compare_nav,
    compare_positions,
    compare_trades,
)
from scripts.v2_p4.t4_gate_runner import _parse_excluded_dates
from src.lazybull.v2.hosts.paper.replay import (
    ExecutionConfigMismatchError,
    assert_execution_configs_match,
)


def _trades(values=("A", "B")):
    rows = []
    for index, code in enumerate(values):
        rows.append(
            {
                "trade_date": "20260706",
                "signal_date": "20260703",
                "ts_code": code,
                "action": "buy",
                "price": 10.0 + index,
                "shares": 100,
                "amount": 1000.0,
                "cost": 5.0,
                "sell_type": None,
            }
        )
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def test_canonical_trades_buy_only_adds_sell_columns():
    raw = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("20260706"),
                "signal_date": pd.Timestamp("20260703"),
                "stock": "A",
                "action": "buy",
                "price": 10.0,
                "shares": 100,
                "amount": 1000.0,
                "cost": 5.0,
            }
        ]
    )
    frame = canonical_trades(raw)
    assert list(frame.columns) == TRADE_COLUMNS
    assert frame.loc[0, "sell_type"] is pd.NA


def test_trade_exact_and_multiset_difference():
    passed, _ = compare_trades(_trades(), _trades(), selected_count=20, allow_tie_swaps=True)
    assert passed["passed"] and passed["ordered_identical"]
    changed = _trades()
    changed.loc[0, "shares"] = 200
    failed, detail = compare_trades(_trades(), changed, selected_count=20, allow_tie_swaps=True)
    assert not failed["passed"] and not failed["multiset_equal"]
    assert not detail["ordered_equal"].all()


def test_tie_swap_threshold_is_strictly_above_five_percent():
    left = _trades(("A", "B"))
    right = left.iloc[::-1].reset_index(drop=True)
    # 两行位置不同；threshold=2 时不超阈（严格 > 才触发）。
    accepted, _ = compare_trades(left, right, selected_count=40, allow_tie_swaps=True)
    assert accepted["multiset_equal"] and not accepted["ordered_identical"]
    assert accepted["tie_swap_count"] == 2
    assert accepted["passed"] and not accepted["resolution_insufficient"]
    rejected, _ = compare_trades(left, right, selected_count=20, allow_tie_swaps=True)
    assert rejected["tie_swap_count"] == 2
    assert rejected["resolution_insufficient"] and not rejected["passed"]


def test_gate2_does_not_allow_tie_swap():
    left = _trades(("A", "B"))
    result, _ = compare_trades(
        left, left.iloc[::-1].reset_index(drop=True), selected_count=100, allow_tie_swaps=False
    )
    assert not result["passed"]


def test_nav_tolerance_and_date_mismatch():
    left = canonical_nav(pd.DataFrame({"date": ["20260703", "20260706"], "nav": [1.0, 1.01]}))
    right = left.copy()
    right.loc[1, "nav"] += 1e-6
    passed, _ = compare_nav(left, right)
    assert passed["passed"] and passed["max_abs_delta"] == pytest.approx(1e-6)
    right.loc[1, "nav"] += 1e-8
    failed, _ = compare_nav(left, right)
    assert not failed["passed"]
    missing, _ = compare_nav(left.iloc[:1], right)
    assert not missing["passed"] and missing["mismatch_dates"] == ["20260706"]


def test_positions_compare_exact():
    frame = pd.DataFrame(
        {
            "date": ["20260706"],
            "ts_code": ["A"],
            "shares": [100],
            "market_value": [1000.0],
            "weight": [0.1],
            "portfolio_value": [10000.0],
            "buy_date": ["20260703"],
            "signal_date": ["20260702"],
            "holding_days": [1],
            "planned_exit_date": ["20260731"],
            "remaining_intervals": [19],
        }
    )
    left = canonical_positions(frame)
    assert compare_positions(left, left.copy())["passed"]
    changed = left.copy()
    changed.loc[0, "shares"] = 200
    assert not compare_positions(left, changed)["passed"]


def test_execution_config_fingerprint_precedes_comparison():
    left = {"buy_price": "close", "sell_price": "open", "initial_capital": 650000.0}
    result = assert_execution_configs_match(left, dict(left))
    assert result["passed"] and result["left_fingerprint"] == result["right_fingerprint"]
    with pytest.raises(ExecutionConfigMismatchError, match="sell_price"):
        assert_execution_configs_match(left, {**left, "sell_price": "close"})


def test_excluded_dates_default_empty_and_sorted():
    assert _parse_excluded_dates("", False) == ()
    assert _parse_excluded_dates("20261020,20261019", True) == ("20261019", "20261020")


def test_excluded_dates_require_explicit_auxiliary_mode():
    with pytest.raises(ValueError, match="auxiliary-exclusion"):
        _parse_excluded_dates("20261019", False)


def test_excluded_dates_reject_invalid_and_duplicate():
    for raw in ("2026-10-19", "20261019,20261019"):
        with pytest.raises(ValueError):
            _parse_excluded_dates(raw, True)
