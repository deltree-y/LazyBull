"""固定经营规则的信息截止、未知语义和合并月份回归。"""

from scripts.v2_p7.operating_monthly_policy import build_signals, signal_dates


def record(month=2, date="20260320", volume=5, price=1, **extra):
    return {
        "ts_code": "002352.SZ",
        "operating_year": 2026,
        "month": month,
        "publication_date": date,
        "volume_yoy": volume,
        "canonical_price_yoy": price,
        "announcement_id": "a",
        "sha256": "hash",
        "title": "经营简报",
        **extra,
    }


def test_strict_after_twentieth_and_same_day_excluded():
    assert signal_dates(["20260320", "20260323", "20260324"]) == ["20260323"]
    frame = build_signals([record(date="20260323")], ["20260320", "20260323"])
    assert not frame.iloc[0].eligible
    assert frame.iloc[0].availability == "late_or_same_day"


def test_combined_month_not_split_and_no_stale_month():
    frame = build_signals([record(period_months=[1, 2])], ["20260224", "20260323"])
    assert not frame.iloc[0].eligible
    assert frame.iloc[4].eligible


def test_unknown_on_does_not_override_known_negative_or_zero():
    for value in [-3, 0]:
        frame = build_signals([record(volume=value, price=None)], ["20260323"])
        assert not frame.iloc[0].unknown_on
    frame = build_signals([record(price=None)], ["20260323"])
    assert frame.iloc[0].unknown_on and not frame.iloc[0].eligible


def test_later_correction_unknown_does_not_fall_back():
    frame = build_signals(
        [
            record(date="20260318"),
            record(date="20260319", price=None, announcement_id="b", title="经营简报更正"),
        ],
        ["20260323"],
    )
    assert frame.iloc[0].announcement_id == "b"
    assert not frame.iloc[0].eligible


def test_partial_profit_note_does_not_replace_operating_monthly():
    frame = build_signals(
        [
            record(date="20260318"),
            record(date="20260319", price=None, announcement_id="b", title="经营业绩初步核算"),
        ],
        ["20260323"],
    )
    assert frame.iloc[0].eligible
    assert frame.iloc[0].announcement_id == "a"
