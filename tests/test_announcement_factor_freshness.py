import pandas as pd

from src.lazybull.factors.earnings import build_earnings_lookup_by_date
from src.lazybull.factors.express import build_express_lookup_by_date
from src.lazybull.factors.fund_portfolio import build_fund_portfolio_lookup_by_date
from src.lazybull.factors.fundamental import build_fundamental_lookup_by_date
from src.lazybull.factors.holder import build_holder_lookup_by_date


def _days_between(start: str, end: str) -> int:
    start_ts = pd.to_datetime(start, format="%Y%m%d")
    end_ts = pd.to_datetime(end, format="%Y%m%d")
    return int((end_ts - start_ts).days)


def test_express_announcement_keeps_value_and_exposes_freshness():
    df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240120",
                "end_date": "20231231",
                "revenue": 100.0,
                "yoy_net_profit": 20.0,
                "diluted_roe": 10.0,
            }
        ]
    )

    lookup = build_express_lookup_by_date(df, ["20240201", "20250301"])

    assert "20240201" in lookup
    assert "20250301" in lookup
    recent_row = lookup["20240201"].iloc[0]
    stale_row = lookup["20250301"].iloc[0]
    assert recent_row["express_profit_yoy"] == 20.0
    assert stale_row["express_profit_yoy"] == 20.0
    assert recent_row["express_freshness_days"] == _days_between("20240120", "20240201")
    assert stale_row["express_freshness_days"] == _days_between("20240120", "20250301")


def test_fundamental_announcement_keeps_value_and_exposes_freshness():
    df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240120",
                "end_date": "20231231",
                "roe_waa": 12.0,
                "or_yoy": 8.0,
                "netprofit_yoy": 6.0,
                "debt_to_assets": 40.0,
                "q_gr_yoy": 3.0,
            }
        ]
    )

    lookup = build_fundamental_lookup_by_date(df, ["20240201", "20250301"])

    assert "20240201" in lookup
    assert "20250301" in lookup
    recent_row = lookup["20240201"].iloc[0]
    stale_row = lookup["20250301"].iloc[0]
    assert recent_row["fundamental_freshness_days"] == _days_between("20240120", "20240201")
    assert stale_row["fundamental_freshness_days"] == _days_between("20240120", "20250301")


def test_holder_announcement_keeps_value_and_exposes_freshness():
    df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240110",
                "end_date": "20231231",
                "holder_num": 100,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20231010",
                "end_date": "20230930",
                "holder_num": 120,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20230710",
                "end_date": "20230630",
                "holder_num": 130,
            },
        ]
    )

    lookup = build_holder_lookup_by_date(df, ["20240201", "20250301"])

    assert "20240201" in lookup
    assert "20250301" in lookup
    recent_row = lookup["20240201"].iloc[0]
    stale_row = lookup["20250301"].iloc[0]
    assert recent_row["holder_num_chg"] == stale_row["holder_num_chg"]
    assert recent_row["holder_freshness_days"] == _days_between("20240110", "20240201")
    assert stale_row["holder_freshness_days"] == _days_between("20240110", "20250301")


def test_earnings_announcement_keeps_value_and_exposes_freshness():
    df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240120",
                "end_date": "20231231",
                "type": "预增",
                "p_change_min": 10.0,
                "p_change_max": 20.0,
            }
        ]
    )

    lookup = build_earnings_lookup_by_date(df, ["20240201", "20250301"])

    assert "20240201" in lookup
    assert "20250301" in lookup
    recent_row = lookup["20240201"].iloc[0]
    stale_row = lookup["20250301"].iloc[0]
    assert recent_row["forecast_type_score"] == stale_row["forecast_type_score"]
    assert recent_row["forecast_freshness_days"] == _days_between("20240120", "20240201")
    assert stale_row["forecast_freshness_days"] == _days_between("20240120", "20250301")


def test_fund_portfolio_announcement_keeps_value_and_exposes_freshness():
    df = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "ann_date": "20240120",
                "end_date": "20231231",
                "fund_hold_ratio": 1.2,
                "fund_hold_ratio_chg": 0.2,
                "fund_count": 10,
                "fund_count_chg": 1,
            }
        ]
    )

    lookup = build_fund_portfolio_lookup_by_date(
        df,
        ["20240201", "20250301"],
        pre_aggregated=True,
    )

    assert "20240201" in lookup
    assert "20250301" in lookup
    recent_row = lookup["20240201"].iloc[0]
    stale_row = lookup["20250301"].iloc[0]
    assert recent_row["fund_hold_ratio"] == stale_row["fund_hold_ratio"]
    assert recent_row["fund_portfolio_freshness_days"] == _days_between("20240120", "20240201")
    assert stale_row["fund_portfolio_freshness_days"] == _days_between("20240120", "20250301")