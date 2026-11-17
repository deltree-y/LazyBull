"""冻结加工价差规则：仅合成数据，验证时点、跨源和缺失语义。"""

import copy

import pandas as pd
import pytest

from scripts.v2_p7.polyester_policy import build_signals, signal_dates, signal_summary


def point(month, spread, *, available=None, series="same"):
    observed = f"{month}20"
    return {
        "source_id": observed,
        "observation_date": observed,
        "publication_date": observed,
        "available_date": available or f"{month}21",
        "poy": 0.855 * 5000 + 0.335 * 4000 + spread,
        "pta": 5000,
        "meg": 4000,
        "unit": "CNY/t",
        "series_id": series,
        "quality_usable": True,
    }


def samples():
    return [point("202001", 100), point("202002", 200), point("202003", 300), point("202004", 250)]


def test_improvement_is_vs_three_prior_months_not_last_month():
    result = build_signals(samples(), ["20200506", "20200507"])
    assert result.eligible.tolist() == [True, True]
    assert result.prior_three_mean.tolist() == [200.0, 200.0]
    assert signal_summary(result)["known_months"] == 1


def test_publication_available_on_signal_day_is_not_known():
    records = samples()
    records[-1]["available_date"] = "20200506"
    row = build_signals(records, ["20200506"]).iloc[0]
    assert not row.eligible and not row.input_known and row.unknown_on


def test_later_observation_and_revision_cannot_rewrite_earlier_signal():
    records = samples()
    extra = point("202004", 10, available="20200507")
    extra.update(source_id="revision", observation_date="20200430", publication_date="20200506")
    before = build_signals(records, ["20200506"])
    after = build_signals([*records, extra], ["20200506"])
    pd.testing.assert_frame_equal(before, after)


def test_missing_calendar_month_is_not_compressed_to_previous_observation():
    records = [point("201912", 100), *samples()[1:]]
    row = build_signals(records, ["20200506"]).iloc[0]
    assert not row.input_known and row.reason == "missing_prior_month"


def test_source_break_cannot_create_spurious_improvement():
    records = samples()
    records[-1]["series_id"] = "different_source"
    row = build_signals(records, ["20200506"]).iloc[0]
    assert not row.input_known and row.reason == "series_break"


@pytest.mark.parametrize("spread", [200, 0, -100])
def test_equality_and_nonpositive_spread_do_not_buy(spread):
    records = samples()[:-1] + [point("202004", spread)]
    row = build_signals(records, ["20200506"]).iloc[0]
    assert row.input_known and not row.eligible and not row.unknown_on


def test_known_negative_current_blocks_unknown_on_even_if_history_missing():
    row = build_signals([point("202004", -100)], ["20200506"]).iloc[0]
    assert not row.input_known and not row.unknown_on


def test_fixed_delay_uses_trading_days():
    records = samples()
    records[-1]["available_date"] = "20200430"
    calendar = ["20200427", "20200428", "20200429", "20200430", "20200506", "20200507"]
    normal = build_signals(records, calendar)
    delayed = build_signals(records, calendar, delay_trading_days=2)
    assert normal.loc[normal.signal_date.eq("20200506"), "eligible"].all()
    assert not delayed.loc[delayed.signal_date.eq("20200506"), "input_known"].any()


def test_window_start_delay_can_use_known_prior_trading_calendar():
    records = samples()
    records[-1]["available_date"] = "20200421"
    clock = ["20200427", "20200428", "20200429", "20200430", "20200506", "20200507"]
    result = build_signals(
        records, ["20200506", "20200507"], delay_trading_days=4, availability_calendar=clock
    )
    assert result.eligible.all()
    assert result.availability_cutoff.tolist() == ["20200427", "20200427"]


@pytest.mark.parametrize(
    "field,value", [("poy", float("nan")), ("unit", "USD/t"), ("quality_usable", "yes")]
)
def test_bad_source_contract_is_rejected(field, value):
    records = copy.deepcopy(samples())
    records[0][field] = value
    with pytest.raises(ValueError):
        build_signals(records, ["20200506"])


def test_month_start_dates_and_duplicate_calendar_rejection():
    assert signal_dates(["20200506", "20200507", "20200601"]) == ["20200506", "20200601"]
    with pytest.raises(ValueError):
        signal_dates(["20200506", "20200506"])
