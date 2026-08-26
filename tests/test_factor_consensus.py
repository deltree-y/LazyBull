"""一致预期因子单元测试"""

from typing import Dict, List, Optional

import pandas as pd
import pytest

from src.lazybull.factors.consensus import (
    CONS_COLS,
    CONSENSUS_FRESHNESS_COL,
    _rating_to_score,
)
from src.lazybull.factors.consensus import (
    build_consensus_lookup_by_date as _build_consensus_lookup_by_date,
)
from src.lazybull.features import FeatureBuilder


def _with_report_identity(report_rc_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """为因子数值测试补齐真实 report_rc 身份 schema。"""
    if report_rc_df is None or len(report_rc_df) == 0:
        return report_rc_df
    result = report_rc_df.copy()
    report_dates = result["report_date"].astype("string").str.replace("-", "", regex=False)
    if "org_name" not in result.columns:
        result["org_name"] = "测试机构"
    if "author_name" not in result.columns:
        result["author_name"] = "测试分析师"
    if "report_title" not in result.columns:
        result["report_title"] = (
            "测试研报-" + result["ts_code"].astype("string") + "-" + report_dates
        )
    if "quarter" not in result.columns:
        result["quarter"] = report_dates.str[:4] + "Q4"
    return result


def build_consensus_lookup_by_date(
    report_rc_df: Optional[pd.DataFrame],
    trading_dates: List[str],
    daily_data_lookup: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """使用完整身份测试数据调用生产因子入口。"""
    return _build_consensus_lookup_by_date(
        _with_report_identity(report_rc_df),
        trading_dates,
        daily_data_lookup=daily_data_lookup,
    )


def _make_report_rc_df() -> pd.DataFrame:
    # 两只股票, 覆盖近 90 日内多条研报; quarter 为预测季度 (Q4 表示年度预测),
    # FY 相对 report_date 发布年份定位 (FY0=当年, FY1=次年, FY2=后年)
    return pd.DataFrame(
        [
            # 000001: 老研报 (60 天前) + 近 30 日研报
            {
                "ts_code": "000001.SZ",
                "report_date": "20240110",
                "quarter": "2024Q4",
                "eps": 1.00,
                "max_price": 15.0,
                "min_price": 13.0,
                "rating": "买入",
            },
            {
                "ts_code": "000001.SZ",
                "report_date": "20240210",
                "quarter": "2025Q4",
                "eps": 1.10,
                "max_price": 16.0,
                "min_price": 14.0,
                "rating": "增持",
            },
            {
                "ts_code": "000001.SZ",
                "report_date": "20240310",
                "quarter": "2025Q4",
                "eps": 1.25,
                "max_price": 18.0,
                "min_price": 16.0,
                "rating": "买入",
            },
            # 000002: 评级文本中性 + 仅一条
            {
                "ts_code": "000002.SZ",
                "report_date": "20240220",
                "quarter": "2025Q4",
                "eps": 0.50,
                "max_price": 8.0,
                "min_price": 7.0,
                "rating": "中性",
            },
        ]
    )


def _make_revision_report_rc_df() -> pd.DataFrame:
    rows = [
        {
            "ts_code": "000001.SZ",
            "report_date": date,
            "quarter": "2025Q4",
            "eps": 1.10,
        }
        for date in ("20240110", "20240120", "20240210")
    ]
    rows.extend(
        {
            "ts_code": "000001.SZ",
            "report_date": date,
            "quarter": "2025Q4",
            "eps": 1.25,
        }
        for date in ("20240301", "20240310")
    )
    return pd.DataFrame(rows)


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


def test_consensus_target_price_uses_available_single_bound():
    df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "report_date": "20240310",
                "max_price": 18.0,
                "min_price": 16.0,
                "eps": 1.0,
            },
            {
                "ts_code": "000002.SZ",
                "report_date": "20240310",
                "max_price": None,
                "min_price": 12.0,
                "eps": 1.0,
            },
            {
                "ts_code": "000003.SZ",
                "report_date": "20240310",
                "max_price": 20.0,
                "min_price": None,
                "eps": 1.0,
            },
        ]
    )

    frame = build_consensus_lookup_by_date(df, ["20240315"])["20240315"].set_index("ts_code")

    assert frame.loc["000001.SZ", "cons_target_price_mid"] == 17.0
    assert frame.loc["000002.SZ", "cons_target_price_mid"] == 12.0
    assert frame.loc["000003.SZ", "cons_target_price_mid"] == 20.0


def test_consensus_eps_revision():
    df = _make_revision_report_rc_df()
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    row = frame[frame["ts_code"] == "000001.SZ"].iloc[0]
    # 近 30 日两个报告日中值 1.25，此前 90 日三个报告日中值 1.10。
    # 使用有界对称变化率，避免基准接近 0 时爆炸。
    expected = 2.0 * (1.25 - 1.10) / (1.25 + 1.10)
    assert row["cons_eps_revision_30d"] == pytest.approx(expected)


def test_consensus_revision_all_periods_mix():
    # 同一报告日先跨 FY 取中值，再对报告日取中值；revision 不是 FY1 专用。
    baseline = [
        {"ts_code": "000001.SZ", "report_date": date, "quarter": "2025Q4", "eps": 1.10}
        for date in ("20240110", "20240120", "20240210")
    ]
    recent = [
        {"ts_code": "000001.SZ", "report_date": "20240305", "quarter": "2024Q4", "eps": 1.15},
        {"ts_code": "000001.SZ", "report_date": "20240305", "quarter": "2025Q4", "eps": 1.25},
        {"ts_code": "000001.SZ", "report_date": "20240310", "quarter": "2024Q4", "eps": 1.17},
        {"ts_code": "000001.SZ", "report_date": "20240310", "quarter": "2025Q4", "eps": 1.27},
    ]
    df = pd.DataFrame(baseline + recent)
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    r1 = frame[frame["ts_code"] == "000001.SZ"].iloc[0]
    # 两个近日报告日的全预测期日度中值为 1.20、1.22，窗口中值为 1.21。
    expected = 2.0 * (1.21 - 1.10) / (1.21 + 1.10)
    fy1_only = 2.0 * (1.26 - 1.10) / (1.26 + 1.10)
    assert r1["cons_eps_revision_30d"] == pytest.approx(expected)
    assert r1["cons_eps_revision_30d"] != pytest.approx(fy1_only)


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


def test_consensus_keeps_fym1_and_builds_price_normalized_values():
    df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "report_date": "20260110",
                "quarter": "2025Q4",
                "eps": 0.8,
                "max_price": 15.0,
                "min_price": 13.0,
            },
            {
                "ts_code": "000001.SZ",
                "report_date": "20260112",
                "quarter": "2027Q4",
                "eps": 1.2,
                "max_price": 16.0,
                "min_price": 14.0,
            },
        ]
    )
    price_lookup = {"20260115": pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]})}

    row = build_consensus_lookup_by_date(
        df,
        ["20260115"],
        daily_data_lookup=price_lookup,
    )[
        "20260115"
    ].iloc[0]

    assert row["cons_eps_mean_fym1"] == pytest.approx(0.8)
    assert row["cons_eps_yield_fym1"] == pytest.approx(0.08)
    assert row["cons_eps_yield_fy1"] == pytest.approx(0.12)
    assert row["cons_target_upside"] == pytest.approx(0.45)


def test_consensus_state_does_not_disappear_at_90_day_boundary():
    df = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "report_date": "20260101",
                "quarter": "2026Q4",
                "eps": 1.0,
                "max_price": 12.0,
                "min_price": 10.0,
            }
        ]
    )
    trading_dates = ["20260331", "20260401"]
    price_lookup = {
        date: pd.DataFrame({"ts_code": ["000001.SZ"], "close": [10.0]}) for date in trading_dates
    }

    result = build_consensus_lookup_by_date(
        df,
        trading_dates,
        daily_data_lookup=price_lookup,
    )

    assert result["20260331"].iloc[0]["cons_eps_yield_fy0"] == pytest.approx(0.1)
    assert result["20260401"].iloc[0]["cons_eps_yield_fy0"] == pytest.approx(0.1)
    assert result["20260401"].iloc[0][CONSENSUS_FRESHNESS_COL] == 90


def test_consensus_revision_uses_minimum_report_days_and_bounded_change():
    base_rows = [
        {
            "ts_code": "000001.SZ",
            "report_date": date,
            "quarter": "2026Q4",
            "eps": 1e-9,
        }
        for date in ("20251120", "20251210", "20260120")
    ]
    recent_rows = [
        {
            "ts_code": "000001.SZ",
            "report_date": date,
            "quarter": "2026Q4",
            "eps": 1.0,
        }
        for date in ("20260301", "20260310")
    ]

    row = build_consensus_lookup_by_date(
        pd.DataFrame(base_rows + recent_rows),
        ["20260315"],
    )[
        "20260315"
    ].iloc[0]
    revision = row["cons_eps_revision_30d"]

    assert 1.9 < revision <= 2.0

    insufficient = build_consensus_lookup_by_date(
        pd.DataFrame(base_rows + recent_rows[:1]),
        ["20260315"],
    )["20260315"].iloc[0]
    assert pd.isna(insufficient["cons_eps_revision_30d"])


def test_consensus_rejects_incomplete_identity_schema():
    df = _make_revision_report_rc_df().drop(columns=["quarter"])

    with pytest.raises(ValueError, match="report_rc 身份 schema 不完整"):
        _build_consensus_lookup_by_date(
            _with_report_identity(df).drop(columns=["quarter"]), ["20240315"]
        )


def test_consensus_rating_score_mapping():
    df = _make_report_rc_df()
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    r2 = frame[frame["ts_code"] == "000002.SZ"].iloc[0]
    # 中性 = 3.0
    assert r2["cons_rating_score"] == 3.0


@pytest.mark.parametrize(
    "rating,expected",
    [
        ("买进", 5.0),
        ("BUY", 5.0),
        ("跑赢行业", 4.0),
        ("优于大市", 4.0),
        ("Overweight", 4.0),
        ("overweight", 4.0),
    ],
)
def test_consensus_rating_extended_mapping(rating, expected):
    assert _rating_to_score(rating) == expected


@pytest.mark.parametrize("rating", ["无", "", None, "观察"])
def test_consensus_unknown_or_missing_rating_stays_nan(rating):
    assert pd.isna(_rating_to_score(rating))


def test_consensus_analyst_count_window():
    df = _make_report_rc_df()
    result = build_consensus_lookup_by_date(df, ["20240315"])
    frame = result["20240315"]
    r1 = frame[frame["ts_code"] == "000001.SZ"].iloc[0]
    # 近 30 日 (2/14 - 3/15) 000001 只有 20240310 一条
    assert r1["cons_analyst_count_30d"] == 1.0


def test_consensus_coverage_count_uses_unique_reports_not_forecast_rows():
    rows = []
    for quarter in ("2026Q4", "2027Q4", "2028Q4"):
        rows.append(
            {
                "ts_code": "000001.SZ",
                "report_date": "20260310",
                "org_name": "机构甲",
                "author_name": "分析师甲",
                "report_title": "年度深度报告",
                "quarter": quarter,
                "eps": 1.0,
            }
        )
    rows.append(
        {
            "ts_code": "000001.SZ",
            "report_date": "20260310",
            "org_name": "机构甲",
            "author_name": "分析师甲",
            "report_title": "业绩点评",
            "quarter": "2026Q4",
            "eps": 1.1,
        }
    )

    frame = build_consensus_lookup_by_date(pd.DataFrame(rows), ["20260315"])["20260315"]

    assert frame.iloc[0]["cons_analyst_count_30d"] == 2.0


def test_consensus_aggregations_deduplicate_updated_forecast_rows():
    """同一预测行的后到修正版覆盖旧值，不能重复参与聚合。"""
    common = {
        "ts_code": "000001.SZ",
        "report_date": "20260310",
        "org_name": "机构甲",
        "author_name": "分析师甲",
        "report_title": "年度深度报告",
        "quarter": "2026Q4",
    }
    rows = [
        {**common, "eps": 1.0},
        {**common, "eps": 7.0},
        {
            **common,
            "org_name": "机构乙",
            "author_name": "分析师乙",
            "report_title": "业绩点评",
            "eps": 5.0,
        },
    ]

    frame = build_consensus_lookup_by_date(pd.DataFrame(rows), ["20260315"])["20260315"]

    row = frame.iloc[0]
    assert row["cons_eps_mean_fy0"] == 6.0
    assert row["cons_analyst_count_30d"] == 2.0


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
