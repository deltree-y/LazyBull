"""测试单日 point-in-time 因子快照与多日查询结果一致。"""

import pandas as pd
from pandas.testing import assert_frame_equal

from src.lazybull.factors.earnings import build_earnings_lookup_by_date
from src.lazybull.factors.express import build_express_lookup_by_date
from src.lazybull.factors.fund_portfolio import build_fund_portfolio_lookup_by_date
from src.lazybull.factors.fundamental import build_fundamental_lookup_by_date
from src.lazybull.factors.holder import build_holder_lookup_by_date


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("ts_code").reset_index(drop=True)


def test_fundamental_single_day_matches_multi_day():
    df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240110",
                "end_date": "20231231",
                "roe_waa": 10,
                "or_yoy": 5,
                "netprofit_yoy": 6,
                "debt_to_assets": 40,
                "q_gr_yoy": 7,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240420",
                "end_date": "20240331",
                "roe_waa": 11,
                "or_yoy": 8,
                "netprofit_yoy": 9,
                "debt_to_assets": 41,
                "q_gr_yoy": 10,
            },
            {
                "ts_code": "000002.SZ",
                "ann_date": "20240315",
                "end_date": "20231231",
                "roe_waa": 20,
                "or_yoy": 15,
                "netprofit_yoy": 16,
                "debt_to_assets": 50,
                "q_gr_yoy": 17,
            },
        ]
    )

    single = build_fundamental_lookup_by_date(df, ["20240422"])["20240422"]
    multi = build_fundamental_lookup_by_date(df, ["20240201", "20240422"])["20240422"]

    assert_frame_equal(_sorted(single), _sorted(multi), check_dtype=False)


def test_holder_single_day_matches_multi_day():
    df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20240110", "end_date": "20231231", "holder_num": 100},
            {"ts_code": "000001.SZ", "ann_date": "20240420", "end_date": "20240331", "holder_num": 80},
            {"ts_code": "000002.SZ", "ann_date": "20240315", "end_date": "20231231", "holder_num": 50},
        ]
    )

    single = build_holder_lookup_by_date(df, ["20240422"])["20240422"]
    multi = build_holder_lookup_by_date(df, ["20240201", "20240422"])["20240422"]

    assert_frame_equal(_sorted(single), _sorted(multi), check_dtype=False)


def test_earnings_single_day_matches_multi_day():
    df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20240105", "end_date": "20231231", "type": "略增", "p_change_min": 10, "p_change_max": 20},
            {"ts_code": "000001.SZ", "ann_date": "20240418", "end_date": "20240331", "type": "预增", "p_change_min": 30, "p_change_max": 40},
            {"ts_code": "000002.SZ", "ann_date": "20240301", "end_date": "20231231", "type": "预减", "p_change_min": -20, "p_change_max": -10},
        ]
    )

    single = build_earnings_lookup_by_date(df, ["20240422"])["20240422"]
    multi = build_earnings_lookup_by_date(df, ["20240201", "20240422"])["20240422"]

    assert_frame_equal(_sorted(single), _sorted(multi), check_dtype=False)


def test_express_single_day_matches_multi_day():
    express_df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20230120", "end_date": "20221231", "revenue": 2000000, "yoy_net_profit": 12.0, "diluted_roe": 11.5},
            {"ts_code": "000001.SZ", "ann_date": "20240120", "end_date": "20231231", "revenue": 2360000, "yoy_net_profit": 22.5, "diluted_roe": 13.0},
            {"ts_code": "000002.SZ", "ann_date": "20240315", "end_date": "20231231", "revenue": 800000, "yoy_net_profit": -3.0, "diluted_roe": 6.0},
        ]
    )
    forecast_df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20240110", "end_date": "20231231", "p_change_min": 10.0, "p_change_max": 20.0},
        ]
    )

    single = build_express_lookup_by_date(express_df, ["20240422"], forecast_df=forecast_df)["20240422"]
    multi = build_express_lookup_by_date(
        express_df,
        ["20240201", "20240422"],
        forecast_df=forecast_df,
    )["20240422"]

    assert_frame_equal(_sorted(single), _sorted(multi), check_dtype=False)


def test_fund_portfolio_single_day_matches_multi_day():
    df = pd.DataFrame(
        [
            {"symbol": "000001", "ann_date": "20240120", "end_date": "20231231", "fund_hold_ratio": 1.2, "fund_count": 10},
            {"symbol": "000001", "ann_date": "20240420", "end_date": "20240331", "fund_hold_ratio": 1.5, "fund_count": 12},
            {"symbol": "600000", "ann_date": "20240315", "end_date": "20231231", "fund_hold_ratio": 2.0, "fund_count": 8},
        ]
    )

    single = build_fund_portfolio_lookup_by_date(df, ["20240422"], pre_aggregated=True)["20240422"]
    multi = build_fund_portfolio_lookup_by_date(
        df,
        ["20240201", "20240422"],
        pre_aggregated=True,
    )["20240422"]

    assert_frame_equal(_sorted(single), _sorted(multi), check_dtype=False)