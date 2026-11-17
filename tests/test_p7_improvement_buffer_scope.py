"""缓冲作用面补充的合成测试，不读取真实历史产物。"""

import pandas as pd
import pytest

from scripts.v2_p7.improvement_buffer_scope import classify_plans, summarize_scope, validate_topk

CALENDAR = ["20260102", "20260105", "20260106", "20260107", "20260108", "20260109", "20260112"]
FOLDS = [
    {"split_index": 0, "model_version": 100, "signal_span": ["20260102", "20260105"]},
    {"split_index": 1, "model_version": 101, "signal_span": ["20260106", "20260109"]},
]


def plan(code="X", due="20260107", budget=100.0):
    return {
        "ts_code": code,
        "signal_date": "20260102",
        "execution_date": "20260105",
        "due_date": due,
        "split_index": 0,
        "model_version": 100,
        "tranche_idx": 0,
        "budget": budget,
        "weight": 0.01,
    }


def rankings(rows):
    return pd.DataFrame(rows, columns=["split_index", "trade_date", "ts_code", "rank"])


def test_cross_fold_uses_decision_model_and_previous_session():
    plans = pd.DataFrame([plan()])
    before = plans.copy(deep=True)
    topk = rankings([(0, "20260105", "X", 1), (1, "20260106", "Y", 1)])
    result = classify_plans(plans, CALENDAR, FOLDS, topk, set(), set())
    row = result.iloc[0]
    assert row.decision_date == "20260106"
    assert row.current_model_version == 101
    assert row.cross_fold and row.status == "miss"
    pd.testing.assert_frame_equal(plans, before)


def test_weekend_uses_friday_and_top20_membership_only():
    topk = rankings([(1, "20260109", "X", 20)])
    result = classify_plans(
        pd.DataFrame([plan(due="20260112")]), CALENDAR, FOLDS, topk, set(), set()
    )
    assert result.iloc[0].decision_date == "20260109"
    assert result.iloc[0].status == "hit"
    assert result.iloc[0].rank_within_top20 == 20


def test_unknown_does_not_become_miss_and_budget_is_not_weighted_twice():
    topk = rankings([(1, "20260106", "X", 1)])
    plans = pd.DataFrame([plan(budget=100), plan("Y", "20260108", 300)])
    result = classify_plans(plans, CALENDAR, FOLDS, topk, set(), set())
    summary = summarize_scope(result)
    assert summary["hit_plans"] == 1 and summary["miss_plans"] == 0
    assert summary["unknown_plans"] == 1
    assert summary["hit_rate_known_plans"] == 1
    assert summary["hit_rate_all_plans_lower"] == 0.5
    assert summary["hit_rate_all_plans_upper"] == 1
    assert summary["total_initial_plan_budget"] == 400
    assert summary["hit_budget_share_all_lower"] == 0.25
    assert summary["hit_budget_share_all_upper"] == 1
    assert summary["unknown_reason_counts"] == {"missing_ranking": 1}


@pytest.mark.parametrize(
    "closure,empty,reason",
    [
        ({"20260106"}, set(), "stock_connect_closure"),
        (set(), {"20260106"}, "empty_ranking"),
    ],
)
def test_declared_unavailable_days_are_unknown_even_if_rankings_exist(closure, empty, reason):
    result = classify_plans(
        pd.DataFrame([plan()]), CALENDAR, FOLDS, rankings([(1, "20260106", "X", 1)]), closure, empty
    )
    assert result.iloc[0].status == "unknown"
    assert result.iloc[0].unknown_reason == reason
    assert summarize_scope(result)["hit_rate_known_plans"] is None


def test_runoff_calendar_gap_and_missing_due_are_separate():
    calendar = CALENDAR + ["20260113"]
    plans = pd.DataFrame([plan(due="20260113"), plan(due="20260101"), plan(due="")])
    result = classify_plans(plans, calendar, FOLDS, rankings([]), set(), set())
    assert result.unknown_reason.tolist() == ["runoff", "calendar_uncovered", "missing_due_date"]
    assert summarize_scope(result)["hit_rate_all_plans_upper"] == 1


def test_topk_duplicate_and_rank_gaps_are_rejected():
    frame = rankings([(1, "20260106", "X", 1), (1, "20260106", "X", 2)])
    with pytest.raises(ValueError, match="重复"):
        validate_topk(frame, 1, FOLDS[1]["signal_span"])
    frame = rankings([(1, "20260106", "X", 2)])
    with pytest.raises(ValueError, match="名次"):
        validate_topk(frame, 1, FOLDS[1]["signal_span"])


def test_overlapping_windows_and_invalid_budget_are_rejected():
    with pytest.raises(ValueError, match="重叠"):
        classify_plans(
            pd.DataFrame([plan()]), CALENDAR, [FOLDS[0], FOLDS[0]], rankings([]), set(), set()
        )
    with pytest.raises(ValueError, match="预算"):
        classify_plans(
            pd.DataFrame([plan(budget=float("nan"))]), CALENDAR, FOLDS, rankings([]), set(), set()
        )
