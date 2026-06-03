"""一致预期修正因子双口径兼容测试。"""

import warnings

import pandas as pd

from src.lazybull.factors.consensus_revision import (
    CONSENSUS_REVISION_FRESHNESS_COL,
    build_consensus_revision_lookup_by_date,
)


def _make_daily_lookup(trade_date: str) -> dict:
    return {
        trade_date: pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "close_adj": [10.0],
            }
        )
    }


def test_consensus_revision_supports_rec_schema():
    trade_date = "20240401"
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "report_date": ["20240201", "20240301", "20240320"],
            "rec_fore_Netprofit": [100.0, 120.0, 140.0],
            "rec_target": [12.0, 13.0, 14.0],
        }
    )

    result = build_consensus_revision_lookup_by_date(
        report_rc,
        [trade_date],
        daily_data_lookup=_make_daily_lookup(trade_date),
    )

    assert trade_date in result
    row = result[trade_date].loc[result[trade_date]["ts_code"] == "000001.SZ"].iloc[0]
    assert row[CONSENSUS_REVISION_FRESHNESS_COL] >= 0
    assert pd.notna(row["cons_target_upside"])


def test_consensus_revision_supports_np_tp_and_target_price_proxy():
    trade_date = "20240401"
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "report_date": ["20240201", "20240301", "20240320"],
            "np": [90.0, 110.0, 130.0],
            "max_price": [12.0, 13.0, 14.0],
            "min_price": [10.0, 11.0, 12.0],
        }
    )

    result = build_consensus_revision_lookup_by_date(
        report_rc,
        [trade_date],
        daily_data_lookup=_make_daily_lookup(trade_date),
    )

    assert trade_date in result
    row = result[trade_date].loc[result[trade_date]["ts_code"] == "000001.SZ"].iloc[0]
    assert row[CONSENSUS_REVISION_FRESHNESS_COL] >= 0
    assert pd.notna(row["cons_target_upside"])
    assert pd.notna(row["cons_analyst_count_chg"])


def test_consensus_revision_target_all_nan_does_not_warn_or_crash():
    trade_date = "20240401"
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "report_date": ["20240201", "20240301", "20240320"],
            "np": [90.0, 110.0, 130.0],
            "max_price": [float("nan"), float("nan"), float("nan")],
            "min_price": [float("nan"), float("nan"), float("nan")],
        }
    )

    result = build_consensus_revision_lookup_by_date(
        report_rc,
        [trade_date],
        daily_data_lookup=_make_daily_lookup(trade_date),
    )

    assert trade_date in result
    row = result[trade_date].loc[result[trade_date]["ts_code"] == "000001.SZ"].iloc[0]
    assert pd.isna(row["cons_target_upside"])
    assert pd.isna(row["cons_target_upside_chg"])


def test_consensus_revision_eps_with_single_valid_value_does_not_warn():
    trade_date = "20240401"
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
            "report_date": ["20240201", "20240215", "20240301", "20240310", "20240320"],
            "rec_fore_Netprofit": [100.0, float("nan"), float("nan"), float("nan"), float("nan")],
            "rec_target": [12.0, 12.5, 13.0, 13.5, 14.0],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = build_consensus_revision_lookup_by_date(
            report_rc,
            [trade_date],
            daily_data_lookup=_make_daily_lookup(trade_date),
        )

    assert trade_date in result
    row = result[trade_date].loc[result[trade_date]["ts_code"] == "000001.SZ"].iloc[0]
    assert pd.isna(row["cons_eps_dispersion"])
    assert pd.isna(row["cons_eps_dispersion_chg"])
