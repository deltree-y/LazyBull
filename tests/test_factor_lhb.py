"""龙虎榜因子单元测试"""

import pandas as pd

from src.lazybull.factors.lhb import LHB_COLS, build_lhb_lookup_by_date


def _make_top_list_df() -> pd.DataFrame:
    # 3 天, 2 只股票, 其中 000001 第 1 天有 2 条上榜理由
    return pd.DataFrame(
        [
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 1_000_000.0, "net_rate": 0.02,
                "amount_rate": 0.15, "reason": "日涨幅偏离值达 7%",
            },
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 500_000.0, "net_rate": 0.01,
                "amount_rate": 0.10, "reason": "换手率达 20%",
            },
            {
                "trade_date": "20240102", "ts_code": "000002.SZ",
                "net_amount": -800_000.0, "net_rate": -0.03,
                "amount_rate": 0.12, "reason": "日跌幅偏离值",
            },
            {
                "trade_date": "20240103", "ts_code": "000001.SZ",
                "net_amount": 200_000.0, "net_rate": 0.005,
                "amount_rate": 0.08, "reason": "日涨幅偏离值",
            },
            {
                "trade_date": "20240104", "ts_code": "000002.SZ",
                "net_amount": 300_000.0, "net_rate": 0.01,
                "amount_rate": 0.09, "reason": "日涨幅偏离值",
            },
        ]
    )


def test_lhb_lookup_basic():
    df = _make_top_list_df()
    trading_dates = ["20240102", "20240103", "20240104"]
    result = build_lhb_lookup_by_date(df, trading_dates)

    assert set(result.keys()) == set(trading_dates)
    for td, frame in result.items():
        assert "ts_code" in frame.columns
        for col in LHB_COLS:
            assert col in frame.columns, f"{td} 缺列 {col}"


def test_lhb_same_day_multiple_reasons_aggregated():
    df = _make_top_list_df()
    result = build_lhb_lookup_by_date(df, ["20240102", "20240103", "20240104"])
    day0 = result["20240102"]
    row = day0[day0["ts_code"] == "000001.SZ"].iloc[0]
    # 两条上榜理由的净买入额应相加
    assert row["lhb_net_amount"] == 1_500_000.0
    assert row["lhb_reason_count"] == 2
    assert row["lhb_on_list"] == 1.0


def test_lhb_rolling_up_days():
    df = _make_top_list_df()
    result = build_lhb_lookup_by_date(df, ["20240102", "20240103", "20240104"])
    # 000001 连续两日上榜, 第 2 日 up_days_20 应为 2
    row = result["20240103"][result["20240103"]["ts_code"] == "000001.SZ"].iloc[0]
    assert row["lhb_up_days_20"] == 2.0


def test_lhb_net_sum_20_uses_trading_day_window_and_net_rate_intensity():
    df = _make_top_list_df()
    result = build_lhb_lookup_by_date(df, ["20240102", "20240103", "20240104"])
    row = result["20240103"][result["20240103"]["ts_code"] == "000001.SZ"].iloc[0]
    # 20240102 聚合后 net_rate = (0.02 + 0.01) / 2 = 0.015，20240103 net_rate = 0.005
    assert abs(row["lhb_net_sum_20"] - 0.02) < 1e-12


def test_lhb_empty():
    assert build_lhb_lookup_by_date(pd.DataFrame(), ["20240102"]) == {}
    assert build_lhb_lookup_by_date(None, ["20240102"]) == {}
