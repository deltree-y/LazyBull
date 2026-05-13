"""一致预期因子单元测试"""

import pandas as pd

from src.lazybull.features import FeatureBuilder
from src.lazybull.factors.consensus import (
    CONS_COLS,
    CONSENSUS_FRESHNESS_COL,
    build_consensus_lookup_by_date,
)


def _make_report_rc_df() -> pd.DataFrame:
    # 两只股票, 覆盖近 90 日内多条研报
    return pd.DataFrame(
        [
            # 000001: 老研报 (60 天前) + 近 30 日研报
            {"ts_code": "000001.SZ", "report_date": "20240110",
             "eps": 1.00, "max_price": 15.0, "min_price": 13.0, "rating": "买入"},
            {"ts_code": "000001.SZ", "report_date": "20240210",
             "eps": 1.10, "max_price": 16.0, "min_price": 14.0, "rating": "增持"},
            {"ts_code": "000001.SZ", "report_date": "20240310",
             "eps": 1.25, "max_price": 18.0, "min_price": 16.0, "rating": "买入"},
            # 000002: 评级文本中性 + 仅一条
            {"ts_code": "000002.SZ", "report_date": "20240220",
             "eps": 0.50, "max_price": 8.0, "min_price": 7.0, "rating": "中性"},
        ]
    )


def _make_builder_inputs():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    trade_cal = pd.DataFrame(
        {
            "cal_date": dates.strftime("%Y%m%d"),
            "is_open": [1] * len(dates),
        }
    )
    daily_rows = []
    adj_rows = []
    for idx, date in enumerate(dates):
        trade_date = date.strftime("%Y%m%d")
        close = 10.0 + idx * 0.1
        pre_close = 10.0 + (idx - 1) * 0.1 if idx > 0 else 10.0
        daily_rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "open": pre_close * 1.001,
                "close": close,
                "pre_close": pre_close,
                "pct_chg": (close / pre_close - 1.0) * 100.0 if pre_close > 0 else 0.0,
                "vol": 1000000,
                "amount": 1000000 * close,
            }
        )
        adj_rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": trade_date,
                "adj_factor": 1.0,
            }
        )

    stock_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "name": ["平安银行"],
            "list_date": ["20100101"],
        }
    )
    return trade_cal, pd.DataFrame(daily_rows), pd.DataFrame(adj_rows), stock_basic


def test_consensus_lookup_basic():
    df = _make_report_rc_df()
    # 评估日 20240315 (3/15)
    result = build_consensus_lookup_by_date(df, ["20240315"])
    assert "20240315" in result
    frame = result["20240315"]
    for col in CONS_COLS:
        assert col in frame.columns


def test_consensus_eps_revision():
    df = _make_report_rc_df()
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    row = frame[frame["ts_code"] == "000001.SZ"].iloc[0]
    # 近 30 日 (2/13 - 3/15): 有 20240310 (eps=1.25)
    # 前 30 日 (1/14 - 2/13): 有 20240210 (eps=1.10)
    # revision = (1.25 - 1.10) / 1.10
    assert abs(row["cons_eps_revision_30d"] - (1.25 - 1.10) / 1.10) < 1e-6


def test_consensus_rating_score_mapping():
    df = _make_report_rc_df()
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    r2 = frame[frame["ts_code"] == "000002.SZ"].iloc[0]
    # 中性 = 3.0
    assert r2["cons_rating_score"] == 3.0


def test_consensus_analyst_count_window():
    df = _make_report_rc_df()
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    r1 = frame[frame["ts_code"] == "000001.SZ"].iloc[0]
    # 近 30 日 (2/14 - 3/15) 000001 只有 20240310 一条
    assert r1["cons_analyst_count_30d"] == 1.0


def test_consensus_freshness_days():
    df = _make_report_rc_df()
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    row = frame[frame["ts_code"] == "000001.SZ"].iloc[0]

    assert CONSENSUS_FRESHNESS_COL in frame.columns
    assert row[CONSENSUS_FRESHNESS_COL] == 5


def test_builder_adds_consensus_freshness_placeholder_when_enabled_but_empty():
    trade_cal, daily_data, adj_factor, stock_basic = _make_builder_inputs()
    builder = FeatureBuilder(min_list_days=0, require_label=False)
    trade_date = trade_cal["cal_date"].iloc[25]

    result = builder.build_features_for_day(
        trade_date=trade_date,
        trade_cal=trade_cal,
        daily_data=daily_data,
        adj_factor=adj_factor,
        stock_basic=stock_basic,
        consensus_data=pd.DataFrame(),
    )

    assert CONSENSUS_FRESHNESS_COL in result.columns
    assert result[CONSENSUS_FRESHNESS_COL].isna().all()


def test_consensus_empty():
    assert build_consensus_lookup_by_date(pd.DataFrame(), ["20240315"]) == {}
    assert build_consensus_lookup_by_date(None, ["20240315"]) == {}
