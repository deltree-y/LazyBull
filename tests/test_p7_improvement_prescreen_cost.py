"""成本首关合成探针：时钟、FIFO、最低佣金与来源隔离。"""

import pandas as pd
import pytest

from scripts.v2_p7.improvement_prescreen_cost import (
    cost_summary,
    deletion_costs,
    fee_parts,
    match_sell_rebuy,
    plan_continuation,
    validate_trades,
)

CALENDAR = ["20260102", "20260105", "20260106", "20260107", "20260108", "20260109", "20260112"]


def trade(key, action, day, shares=100, price=10.0, fold=0, code="X", timing="open"):
    amount = shares * price
    return {
        "trade_id": key,
        "split_index": fold,
        "trade_date": day,
        "signal_date": "20260102",
        "ts_code": code,
        "action": action,
        "price": price,
        "shares": shares,
        "amount": amount,
        "cost": sum(fee_parts(amount, action).values()),
        "sell_type": "holding_period",
        "sell_timing": timing,
        "previous_nav": 1_000_000.0,
    }


def test_fifo_never_double_counts_quantities_or_crosses_folds():
    frame = pd.DataFrame(
        [
            trade("s1", "sell", "20260105", 200),
            trade("s2", "sell", "20260106", 100),
            trade("b1", "buy", "20260106", 100),
            trade("b2", "buy", "20260107", 200),
            trade("b3", "buy", "20260107", 100, fold=1),
        ]
    )
    pairs = match_sell_rebuy(frame, CALENDAR)
    assert pairs.matched_shares.sum() == 300
    assert set(pairs.buy_id) == {"b1", "b2"}
    assert pairs.groupby("sell_id").matched_shares.sum().to_dict() == {"s1": 200, "s2": 100}


def test_calendar_sessions_and_unknown_same_day_order():
    frame = pd.DataFrame(
        [
            trade("s", "sell", "20260109", timing="close"),
            trade("b0", "buy", "20260109"),
            trade("b1", "buy", "20260112"),
        ]
    )
    pairs = match_sell_rebuy(frame, CALENDAR)
    assert pairs.buy_id.tolist() == ["b1"]
    assert pairs.gap_sessions.tolist() == [1]


def test_partial_deletion_cannot_proportionally_save_minimum_commission():
    frame = pd.DataFrame([trade("s", "sell", "20260105", 200), trade("b", "buy", "20260105", 100)])
    pairs = match_sell_rebuy(frame, CALENDAR)
    rows = deletion_costs(frame, pairs).set_index("trade_id")
    assert rows.loc["s", "saved_commission"] == 0
    assert rows.loc["b", "saved_commission"] == 5
    assert rows.loc["s", "saved_stamp_tax"] == pytest.approx(0.5)
    assert rows.loc["b", "saved_stamp_tax"] == 0


def test_two_million_reprices_fee_floor_instead_of_scaling_fees():
    frame = pd.DataFrame([trade("s", "sell", "20260105"), trade("b", "buy", "20260105")])
    pairs = match_sell_rebuy(frame, CALENDAR)
    one, two = deletion_costs(frame, pairs, 1), deletion_costs(frame, pairs, 2)
    assert one.saved_commission.sum() == two.saved_commission.sum() == 10
    assert two.saved_slippage_budget.sum() == 2 * one.saved_slippage_budget.sum()


def test_buy_signal_known_flag_does_not_use_sell_original_signal():
    sell = trade("s", "sell", "20260106")
    sell["signal_date"] = "20250102"
    buy = trade("b", "buy", "20260107")
    buy["signal_date"] = "20260106"
    pairs = match_sell_rebuy(pd.DataFrame([sell, buy]), CALENDAR)
    assert not pairs.iloc[0].buy_signal_known_before_sell


def test_unmatched_and_zero_amount_have_no_spurious_fee_saving():
    frame = pd.DataFrame([trade("b", "buy", "20260105")])
    pairs = match_sell_rebuy(frame, CALENDAR)
    rows = deletion_costs(frame, pairs)
    assert rows.filter(like="saved_").to_numpy().sum() == 0
    assert sum(fee_parts(0, "buy").values()) == 0
    with pytest.raises(ValueError):
        fee_parts(-1, "sell")


def test_summary_fee_intensity_uses_each_previous_nav():
    frame = pd.DataFrame([trade("s", "sell", "20260105"), trade("b", "buy", "20260105")])
    frame.loc[1, "previous_nav"] = 2_000_000.0
    pairs = match_sell_rebuy(frame, CALENDAR)
    summary = cost_summary(deletion_costs(frame, pairs), 252, 1_000_000)
    assert summary["saved"]["commission"]["previous_nav_pp_per_252_sessions"] == pytest.approx(
        0.00075
    )
    assert summary["saved"]["commission"]["initial_capital_pp_per_252_sessions"] == pytest.approx(
        0.001
    )


def test_validation_rejects_bad_fee_and_does_not_modify_input():
    frame = pd.DataFrame([trade("b", "buy", "20260105")])
    before = frame.copy(deep=True)
    validate_trades(frame)
    pd.testing.assert_frame_equal(frame, before)
    frame.loc[0, "cost"] += 1
    with pytest.raises(ValueError, match="费用"):
        validate_trades(frame)


def test_plan_continuation_is_separate_and_labels_cross_fold():
    plans = pd.DataFrame(
        [
            {
                "ts_code": "X",
                "signal_date": "20260102",
                "execution_date": "20260105",
                "due_date": "20260107",
                "tranche_idx": 0,
                "split_index": 0,
                "shares": 200,
            },
            {
                "ts_code": "X",
                "signal_date": "20260106",
                "execution_date": "20260107",
                "due_date": "20260112",
                "tranche_idx": 1,
                "split_index": 1,
                "shares": 100,
            },
        ]
    )
    rows = plan_continuation(plans, CALENDAR)
    assert rows.iloc[1].relationship == "due_same_day"
    assert rows.iloc[1].cross_fold
    assert rows.iloc[1].planned_overlap_shares == 100
    assert rows.not_actual_trade.all()


def test_pair_overallocation_is_rejected():
    frame = pd.DataFrame([trade("s", "sell", "20260105"), trade("b", "buy", "20260105")])
    pairs = match_sell_rebuy(frame, CALENDAR)
    pairs.loc[0, "matched_shares"] = 101
    with pytest.raises(ValueError, match="超出"):
        deletion_costs(frame, pairs)
