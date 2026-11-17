"""生猪比值规则的合成检验：严格时点、月历、口径及未知压力。"""

import copy

import pandas as pd
import pytest

from scripts.v2_p7.hog_policy import (
    CODES,
    build_signals,
    signal_dates,
    signal_summary,
    validate_records,
)


def point(month: str, ratio: float, **overrides) -> dict:
    observed = f"{month}20"
    return {
        "source_id": observed,
        "observation_date": observed,
        "publication_date": observed,
        "available_date": f"{month}21",
        "hog": ratio * 2,
        "feed": 2.0,
        "unit": "CNY/kg",
        "series_id": "moa_national_hog_fattening_feed",
        "quality_usable": True,
        **overrides,
    }


def samples() -> list[dict]:
    return [point("202001", 2), point("202002", 3), point("202003", 4), point("202004", 3.5)]


def test_ratio_improvement_uses_three_month_mean_not_previous_month() -> None:
    result = build_signals(samples(), ["20200506", "20200507"])
    assert result.ts_code.tolist() == list(CODES)
    assert result.eligible.tolist() == [True, True]
    assert result.prior_three_mean.tolist() == [3.0, 3.0]
    assert result.ratio.tolist() == [3.5, 3.5]
    assert result.iloc[0].months == ["202004", "202003", "202002", "202001"]
    assert result.iloc[0].source_ids == ["20200420", "20200320", "20200220", "20200120"]
    assert signal_summary(result) == {
        "months": 1,
        "known_months": 1,
        "eligible_months": 1,
        "unknown_months": 0,
        "reason_counts": {"ratio_improving": 1},
        "independent_stock_observations_claimed": False,
    }


def test_no_absolute_break_even_threshold() -> None:
    records = [point("202001", 0.25), point("202002", 0.5), point("202003", 0.75)]
    row = build_signals([*records, point("202004", 0.625)], ["20200506"]).iloc[0]
    assert row.eligible and row.ratio < 1


@pytest.mark.parametrize("ratio", [3.0, 2.5])
def test_equality_or_decline_is_known_but_not_bought(ratio: float) -> None:
    row = build_signals(samples()[:-1] + [point("202004", ratio)], ["20200506"]).iloc[0]
    assert row.input_known and not row.eligible and not row.unknown_on
    assert row.reason == "ratio_not_improving"


def test_future_publication_cannot_rewrite_earlier_signal() -> None:
    later = point(
        "202004",
        1,
        source_id="later_report",
        observation_date="20200430",
        publication_date="20200507",
        available_date="20200508",
    )
    expected = build_signals(samples(), ["20200506"])
    actual = build_signals([*samples(), later], ["20200506"])
    pd.testing.assert_frame_equal(expected, actual)


def test_available_on_signal_day_is_not_consumed() -> None:
    records = samples()
    records[-1]["available_date"] = "20200506"
    row = build_signals(records, ["20200506"]).iloc[0]
    assert not row.input_known and not row.eligible and row.unknown_on
    assert row.reason == "missing_current"


def test_five_day_delay_selects_older_week_across_month_boundary() -> None:
    records = samples()[:-1] + [point("202004", 2)]
    records.append(
        point(
            "202004",
            4,
            source_id="late_april",
            observation_date="20200428",
            publication_date="20200428",
            available_date="20200429",
        )
    )
    clock = [
        "20200422",
        "20200423",
        "20200424",
        "20200427",
        "20200428",
        "20200429",
        "20200430",
        "20200506",
        "20200507",
    ]
    normal = build_signals(records, ["20200506", "20200507"], availability_calendar=clock)
    delayed = build_signals(
        records,
        ["20200506", "20200507"],
        delay_trading_days=5,
        availability_calendar=clock,
    )
    assert normal.eligible.all() and not delayed.eligible.any()
    assert delayed.input_known.all()
    assert delayed.availability_cutoff.tolist() == ["20200424", "20200424"]
    assert delayed.iloc[0].source_ids[0] == "20200420"
    assert normal.iloc[0].source_ids[0] == "late_april"
    assert normal.signal_date.tolist() == delayed.signal_date.tolist()
    assert normal.prior_three_mean.tolist() == delayed.prior_three_mean.tolist()


def test_delay_without_warmup_calendar_is_unknown() -> None:
    result = build_signals(samples(), ["20200506"], delay_trading_days=5)
    assert not result.input_known.any() and result.unknown_on.all()
    assert result.availability_cutoff.tolist() == ["00000000", "00000000"]


def test_missing_month_does_not_compress_to_older_observation() -> None:
    records = [point("201912", 1), *samples()[1:]]
    row = build_signals(records, ["20200506"]).iloc[0]
    assert not row.input_known and row.unknown_on
    assert row.reason == "missing_prior_month" and row.source_ids[-1] is None


def test_series_break_is_unknown_without_searching_an_older_matching_week() -> None:
    changed = point(
        "202004",
        4,
        source_id="new_series",
        observation_date="20200428",
        publication_date="20200428",
        available_date="20200429",
        series_id="changed_sampling_scope",
    )
    row = build_signals([*samples(), changed], ["20200506"]).iloc[0]
    assert not row.input_known and not row.eligible and row.unknown_on
    assert row.reason == "series_break" and row.source_ids[0] == "new_series"


def test_unknown_on_has_no_current_ratio_gate() -> None:
    row = build_signals([point("202004", 0.01)], ["20200506"]).iloc[0]
    assert not row.input_known and not row.eligible and row.unknown_on


def test_unusable_raw_row_is_preserved_but_never_consumed() -> None:
    unusable = point(
        "202004",
        1,
        source_id="unparsed",
        observation_date="20200430",
        publication_date="20200430",
        available_date="20200501",
        quality_usable=False,
        hog=None,
        feed="未解析",
        unit=None,
        series_id=None,
    )
    records = [*samples(), unusable]
    original = copy.deepcopy(records)
    assert validate_records(records)[-1] == unusable
    pd.testing.assert_frame_equal(
        build_signals(records, ["20200506"]), build_signals(samples(), ["20200506"])
    )
    assert records == original


def test_input_order_and_same_date_tie_break_are_deterministic() -> None:
    tied = point("202004", 4, source_id="z_same_date")
    records = [*samples(), tied]
    forward = build_signals(records, ["20200506"])
    backward = build_signals(list(reversed(records)), ["20200506"])
    pd.testing.assert_frame_equal(forward, backward)
    assert forward.iloc[0].source_ids[0] == "z_same_date"


@pytest.mark.parametrize(
    "field,value",
    [
        ("hog", float("nan")),
        ("hog", float("inf")),
        ("hog", -1),
        ("hog", True),
        ("feed", 0),
        ("feed", "2.0"),
        ("feed", None),
        ("unit", "CNY/t"),
        ("series_id", ""),
        ("series_id", []),
        ("quality_usable", "yes"),
        ("observation_date", "20200230"),
        ("publication_date", "20200119"),
        ("available_date", "20200119"),
    ],
)
def test_bad_source_contract_is_rejected(field: str, value: object) -> None:
    records = samples()
    records[0][field] = value
    with pytest.raises(ValueError):
        build_signals(records, ["20200506"])


def test_duplicate_source_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="source_id"):
        validate_records([*samples(), samples()[0]])


@pytest.mark.parametrize("delay", [-1, 1.5, True])
def test_illegal_delay_is_rejected(delay: object) -> None:
    with pytest.raises(ValueError, match="非负整数"):
        build_signals(samples(), ["20200506"], delay_trading_days=delay)


@pytest.mark.parametrize("calendar", [[], ["20200506", "20200506"], ["20200507", "20200506"]])
def test_empty_duplicate_or_unsorted_calendar_is_rejected(calendar: list[str]) -> None:
    with pytest.raises(ValueError, match="唯一升序"):
        build_signals(samples(), calendar)


def test_month_starts_and_availability_calendar_domain() -> None:
    assert signal_dates(["20200506", "20200507", "20200601"]) == ["20200506", "20200601"]
    with pytest.raises(ValueError, match="全部研究交易日"):
        build_signals(samples(), ["20200506", "20200507"], availability_calendar=["20200506"])


def test_no_quotes_retains_unknown_months_and_both_stock_slots() -> None:
    result = build_signals([], ["20200506", "20200507", "20200601"])
    assert len(result) == 4 and not result.eligible.any() and result.unknown_on.all()
    summary = signal_summary(result)
    assert summary["months"] == 2 and summary["unknown_months"] == 2
