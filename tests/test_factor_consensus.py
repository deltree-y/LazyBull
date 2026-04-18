"""一致预期因子单元测试"""

import pandas as pd

from src.lazybull.factors.consensus import (
    CONS_COLS,
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


def test_consensus_empty():
    assert build_consensus_lookup_by_date(pd.DataFrame(), ["20240315"]) == {}
    assert build_consensus_lookup_by_date(None, ["20240315"]) == {}
