"""固定分段来源归一化；不依赖生产或真实原件。"""

import pytest

from scripts.v2_p7.polyester_inputs import dated_tbs_rows, normalize_htfc, normalize_tbs


def chart(rows):
    return {"data": [["日期", "价格"], *rows]}


def test_month_day_rollover_and_no_row_alignment():
    payloads = {
        "poy": chart([["12-31", 7000], ["01-02", 7100], ["01-05", 7200]]),
        "pta": chart([["12-31", 5000], ["01-05", 5100]]),
        "meg": chart([["12-31", 4000], ["01-02", 4050], ["01-05", 4100]]),
    }
    records, omissions = normalize_tbs(payloads, {})
    assert [row["observation_date"] for row in records] == ["20251231", "20260105"]
    assert records[1]["pta"] == 5100
    assert records[0]["available_date"] == "20260101"
    assert all(row["publication_date_assumed"] and not row["strict_usable"] for row in records)
    assert omissions[0]["observation_date"] == "20260102"
    assert omissions[0]["missing_legs"] == ["pta"]


def test_duplicate_or_outside_range_rejected():
    with pytest.raises(ValueError):
        dated_tbs_rows(chart([["12-01", 1], ["12-01", 2]]), "20251201", "20261031")
    with pytest.raises(ValueError):
        dated_tbs_rows(chart([["11-30", 1]]), "20251201", "20261031")


def test_conflicting_day_excluded_and_identical_duplicate_audited():
    ledger = []
    result = dated_tbs_rows(
        chart([["12-01", 10], ["12-01", 11], ["12-02", 12], ["12-02", 12]]),
        "20251201",
        "20261031",
        duplicate_ledger=ledger,
    )
    assert result == {"20251202": 12}
    assert [row["reason"] for row in ledger] == [
        "conflicting_prices_excluded",
        "duplicate_same_value_deduplicated",
    ]


def test_unsorted_rows_do_not_create_a_false_new_year():
    result = dated_tbs_rows(chart([["01-05", 1], ["01-02", 2]]), "20251201", "20261031")
    assert result == {"20260105": 1, "20260102": 2}


def test_htfc_does_not_silently_accept_conflict_or_invent_missing_date():
    records, omissions = normalize_htfc(
        [
            {
                "article_id": "a",
                "quality_usable": False,
                "observation_date": "20230329",
                "parse_status": "poy_conflict_between_same_date_reports",
            },
            {"article_id": "b", "quality_usable": False, "observation_date": None},
        ]
    )
    assert not records
    assert len(omissions) == 2
    assert omissions[0]["reason"] == "poy_conflict_between_same_date_reports"
