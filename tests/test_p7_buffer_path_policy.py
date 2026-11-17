"""固定续持规则的合成验证：到期时钟、优先级、名额和预算。"""

from copy import deepcopy

import pytest

from scripts.v2_p7.buffer_path_policy import due_decision, plan_tranche
from scripts.v2_p7.t25.target_layer import PlannerKernel, select_targets, sleeve_weights

DAY = "20260106"
RANKED = [(f"S{index:02d}", float(100 - index)) for index in range(30)]


class SyntheticVariance:
    """不读磁盘，只记录函数索取的当日方差。"""

    def __init__(self):
        self.calls = []

    def variance(self, code, day):
        self.calls.append((code, day))
        return 0.001


def test_due_clock_uses_actual_age_across_unlimited_renewals():
    assert not due_decision("S00", 18, 0, RANKED, True)["due"]
    first = due_decision("S00", 19, 0, RANKED, True)
    assert first == {"due": True, "renew": True, "next_renewal_count": 1, "reason": "renew_top20"}
    assert not due_decision("S00", 20, 1, RANKED, True)["due"]
    assert not due_decision("S00", 38, 1, RANKED, True)["due"]
    assert due_decision("S00", 39, 1, RANKED, True)["next_renewal_count"] == 2
    assert due_decision("S00", 219, 10, RANKED, True)["next_renewal_count"] == 11


@pytest.mark.parametrize(
    "enabled,active,forced,reason",
    [
        (False, True, False, "buffer_disabled"),
        (True, False, False, "inactive_signal"),
        (True, True, True, "forced_exit"),
        (False, False, True, "forced_exit"),
    ],
)
def test_existing_exit_and_inactive_signal_never_renew(enabled, active, forced, reason):
    result = due_decision("S00", 19, 0, RANKED, enabled, active, forced)
    assert result["due"] and not result["renew"]
    assert result["next_renewal_count"] == 0 and result["reason"] == reason


def test_only_current_top20_can_renew_and_empty_rank_is_not_a_signal():
    assert due_decision("S19", 19, 0, RANKED, True)["renew"]
    assert not due_decision("S20", 19, 0, RANKED, True)["renew"]
    assert due_decision("S00", 19, 0, [], True)["reason"] == "missing_ranking"
    switched = RANKED[1:] + RANKED[:1]
    assert not due_decision("S00", 19, 0, switched, True)["renew"]


def test_no_retained_reproduces_kernel_selection_and_half_kelly_budget():
    variance = SyntheticVariance()
    held = {"S00", "S01"}
    result = plan_tranche(DAY, 0, RANKED, held, set(), 0, 1_000_000, variance)
    picks, _ = select_targets(PlannerKernel(), RANKED, DAY, 0, sorted(held))
    expected = sleeve_weights(dict(picks), DAY, SyntheticVariance())
    assert result["new_weights"] == expected
    assert [row["ts_code"] for row in result["new_buys"]] == [code for code, _ in picks]
    assert result["new_budgets"] == {code: 500_000 * weight for code, weight in expected.items()}
    assert result["budget_scale"] == 1
    assert {day for _, day in variance.calls} == {DAY}


def test_retained_consume_slots_and_market_value_without_new_buy_in_same_stock():
    held, retained = {"S00", "S01", "S02"}, {"S00", "S01"}
    inputs = deepcopy((RANKED, held, retained))
    result = plan_tranche(DAY, 1, RANKED, held, retained, 450_000, 1_000_000, None)
    assert result["retained_count"] == 2 and result["new_slot_count"] == 8
    assert set(result["new_budgets"]).isdisjoint(held)
    assert result["new_total_budget"] == pytest.approx(50_000)
    assert result["budget_scale"] < 1
    assert all(value <= 150_000 for value in result["new_budgets"].values())
    assert (RANKED, held, retained) == inputs
    assert result["cashguard_applied"] is False


def test_missing_retained_score_is_zero_and_budget_never_expands_to_spend_residual():
    result = plan_tranche(DAY, 0, RANKED, {"MISSING"}, {"MISSING"}, 1_000, 1_000_000, None)
    assert result["retained_missing_score"] == ["MISSING"]
    assert result["composite_scores"]["MISSING"] == 0
    assert result["new_slot_count"] == 9
    assert result["budget_scale"] == 1
    assert result["new_total_budget"] < result["new_budget_cap"]


def test_sparse_candidates_leave_slots_and_cash_and_single_stock_cap():
    result = plan_tranche(DAY, 0, RANKED[:1], set(), set(), 0, 1_000_000, None)
    assert result["new_slot_count"] == 1 and result["unfilled_slot_count"] == 9
    assert result["new_budgets"] == {"S00": 150_000}
    assert result["new_total_budget"] < result["batch_budget"]


def test_retained_over_batch_budget_and_full_batch_cannot_create_new_exposure():
    result = plan_tranche(DAY, 0, RANKED, {"S00"}, {"S00"}, 600_000, 1_000_000, None)
    assert result["new_total_budget"] == 0
    kept = {code for code, _ in RANKED[:10]}
    result = plan_tranche(DAY, 0, RANKED, kept, kept, 300_000, 1_000_000, None)
    assert result["new_buys"] == [] and result["new_slot_count"] == 0


def test_invalid_policy_state_and_duplicate_rank_fail_closed():
    with pytest.raises(ValueError, match="续持次数"):
        due_decision("S00", 19, -1, RANKED, True)
    with pytest.raises(ValueError, match="重复"):
        due_decision("S00", 19, 0, RANKED + RANKED[:1], True)
    with pytest.raises(ValueError, match="实际持仓"):
        plan_tranche(DAY, 0, RANKED, set(), {"S00"}, 1, 100, None)
    kept = {code for code, _ in RANKED[:11]}
    with pytest.raises(ValueError, match="10 个槽位"):
        plan_tranche(DAY, 0, RANKED, kept, kept, 1, 100, None)
