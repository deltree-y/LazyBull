"""一致预期因子单元测试"""

import pandas as pd

from src.lazybull.features import FeatureBuilder
from src.lazybull.factors.consensus import (
    CONS_COLS,
    CONSENSUS_FRESHNESS_COL,
    build_consensus_lookup_by_date,
)


def _make_report_rc_df() -> pd.DataFrame:
    # 两只股票, 覆盖近 90 日内多条研报; quarter 为预测季度 (Q4 表示年度预测),
    # FY 相对 report_date 发布年份定位 (FY0=当年, FY1=次年, FY2=后年)
    return pd.DataFrame(
        [
            # 000001: 老研报 (60 天前) + 近 30 日研报
            {"ts_code": "000001.SZ", "report_date": "20240110",
             "quarter": "2024Q4", "eps": 1.00, "max_price": 15.0, "min_price": 13.0, "rating": "买入"},
            {"ts_code": "000001.SZ", "report_date": "20240210",
             "quarter": "2025Q4", "eps": 1.10, "max_price": 16.0, "min_price": 14.0, "rating": "增持"},
            {"ts_code": "000001.SZ", "report_date": "20240310",
             "quarter": "2025Q4", "eps": 1.25, "max_price": 18.0, "min_price": 16.0, "rating": "买入"},
            # 000002: 评级文本中性 + 仅一条
            {"ts_code": "000002.SZ", "report_date": "20240220",
             "quarter": "2025Q4", "eps": 0.50, "max_price": 8.0, "min_price": 7.0, "rating": "中性"},
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
    # 修正率为全预测期口径 (与 FY 分组无关):
    # 近 30 日 (2/14 - 3/15): 20240310 (eps=1.25)
    # 前 30 日 (1/14 - 2/13): 20240210 (eps=1.10)
    # revision = (1.25 - 1.10) / 1.10
    assert abs(row["cons_eps_revision_30d"] - (1.25 - 1.10) / 1.10) < 1e-6


def test_consensus_revision_all_periods_mix():
    # 近 30 日同时含 FY0 与 FY1 研报时, revision 使用全预测期 eps 中值 (非 FY1 专用)
    df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "report_date": "20240210",
             "quarter": "2025Q4", "eps": 1.10, "max_price": 16.0, "min_price": 14.0, "rating": "增持"},
            {"ts_code": "000001.SZ", "report_date": "20240305",
             "quarter": "2024Q4", "eps": 1.15, "max_price": 17.0, "min_price": 15.0, "rating": "买入"},
            {"ts_code": "000001.SZ", "report_date": "20240310",
             "quarter": "2025Q4", "eps": 1.25, "max_price": 18.0, "min_price": 16.0, "rating": "买入"},
        ]
    )
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    r1 = frame[frame["ts_code"] == "000001.SZ"].iloc[0]
    # 近 30 日全预测期 eps: [1.15(fy0), 1.25(fy1)] -> 中值 1.20
    # 前 30 日全预测期 eps: [1.10(fy1)] -> 中值 1.10
    # revision = (1.20 - 1.10) / 1.10
    assert abs(r1["cons_eps_revision_30d"] - (1.20 - 1.10) / 1.10) < 1e-6
    # 若误用 FY1 专用中值 (1.25-1.10)/1.10, 应不相等
    assert abs(r1["cons_eps_revision_30d"] - (1.25 - 1.10) / 1.10) > 1e-6


def test_consensus_eps_mean_by_fy():
    df = _make_report_rc_df()
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    r1 = frame[frame["ts_code"] == "000001.SZ"].iloc[0]
    # FY0 (2024Q4, 当年): 仅 20240110 (eps=1.00)
    assert abs(r1["cons_eps_mean_fy0"] - 1.00) < 1e-6
    # FY1 (2025Q4, 次年): 20240210 (1.10) + 20240310 (1.25)
    assert abs(r1["cons_eps_mean_fy1"] - 1.175) < 1e-6
    # FY2 (2026Q4, 后年): 无数据
    assert pd.isna(r1["cons_eps_mean_fy2"])


def test_consensus_without_quarter_column_degrades():
    df = _make_report_rc_df().drop(columns=["quarter"])
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    r1 = frame[frame["ts_code"] == "000001.SZ"].iloc[0]
    # 无 quarter 时 EPS 财年分组列优雅降级为 NaN, 不报错
    assert pd.isna(r1["cons_eps_mean_fy0"])
    assert pd.isna(r1["cons_eps_mean_fy1"])
    assert pd.isna(r1["cons_eps_mean_fy2"])
    # revision 为全预测期口径, 与 quarter 无关, 仍有值
    assert abs(r1["cons_eps_revision_30d"] - (1.25 - 1.10) / 1.10) < 1e-6
    # 预测期无关因子仍正常
    assert r1["cons_analyst_count_30d"] == 1.0


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
