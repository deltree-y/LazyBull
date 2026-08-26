"""一致预期修正因子双口径兼容测试。"""

import warnings
from typing import Dict, List, Optional

import pandas as pd
import pytest

from src.lazybull.factors.consensus_revision import (
    CONSENSUS_REVISION_FRESHNESS_COL,
)
from src.lazybull.factors.consensus_revision import (
    build_consensus_revision_lookup_by_date as _build_consensus_revision_lookup_by_date,
)


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


def build_consensus_revision_lookup_by_date(
    report_rc_df: Optional[pd.DataFrame],
    trading_dates: List[str],
    daily_data_lookup: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """使用完整身份测试数据调用生产修正因子入口。"""
    return _build_consensus_revision_lookup_by_date(
        _with_report_identity(report_rc_df),
        trading_dates,
        daily_data_lookup=daily_data_lookup,
    )


def _make_daily_lookup(trade_date: str) -> dict:
    return {
        trade_date: pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "close": [10.0],
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
    assert pd.notna(row["cons_revision_target_upside"])


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
    assert pd.notna(row["cons_revision_target_upside"])
    assert pd.notna(row["cons_analyst_count_chg"])


def test_consensus_revision_target_proxy_uses_available_single_bound():
    trade_date = "20240401"
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "report_date": ["20240201", "20240301", "20240320"],
            "np": [90.0, 110.0, 130.0],
            "max_price": [float("nan"), float("nan"), float("nan")],
            "min_price": [10.0, 11.0, 12.0],
        }
    )

    result = build_consensus_revision_lookup_by_date(
        report_rc,
        [trade_date],
        daily_data_lookup=_make_daily_lookup(trade_date),
    )

    row = result[trade_date].loc[result[trade_date]["ts_code"] == "000001.SZ"].iloc[0]
    assert abs(row["cons_revision_target_upside"] - 0.1) < 1e-12


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
    assert pd.isna(row["cons_revision_target_upside"])
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


def test_consensus_revision_counts_unique_reports_not_forecast_rows():
    """覆盖变化按唯一研报计数，同研报多预测期不能放大。"""
    reports = [
        ("20231215", "机构甲", "分析师甲", "基准研报"),
        ("20240115", "机构乙", "分析师乙", "重叠研报"),
        ("20240315", "机构丙", "分析师丙", "近期研报一"),
        ("20240315", "机构丁", "分析师丁", "近期研报二"),
        ("20240401", "机构戊", "分析师戊", "最新研报"),
    ]
    rows = []
    for report_date, org_name, author_name, report_title in reports:
        for quarter in ("2024Q4", "2025Q4"):
            rows.append(
                {
                    "ts_code": "000001.SZ",
                    "report_date": report_date,
                    "org_name": org_name,
                    "author_name": author_name,
                    "report_title": report_title,
                    "quarter": quarter,
                    "np": 100.0,
                    "max_price": 12.0,
                    "min_price": 10.0,
                }
            )

    result = build_consensus_revision_lookup_by_date(
        pd.DataFrame(rows),
        ["20240401"],
        daily_data_lookup=_make_daily_lookup("20240401"),
    )

    row = result["20240401"].iloc[0]
    assert row["cons_analyst_count_chg"] == 1.0


def test_consensus_revision_state_kept_for_365_days_from_latest_report():
    """修正状态锚定最新研报保留 365 日，不在第 90 日硬消失。"""
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 3,
            "report_date": ["20240102", "20240115", "20240201"],
            "np": [90.0, 100.0, 110.0],
            "max_price": [11.0, 12.0, 13.0],
            "min_price": [9.0, 10.0, 11.0],
        }
    )
    kept_date = "20240515"
    expired_date = "20250202"

    result = build_consensus_revision_lookup_by_date(
        report_rc,
        [kept_date, expired_date],
        daily_data_lookup=_make_daily_lookup(kept_date),
    )

    assert kept_date in result
    assert result[kept_date].iloc[0][CONSENSUS_REVISION_FRESHNESS_COL] == 104
    assert expired_date not in result


def test_consensus_revision_target_column_does_not_shadow_base_factor():
    """修正目标价使用独立列名，不能覆盖基础 cons_target_upside。"""
    trade_date = "20240401"
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 3,
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

    assert "cons_revision_target_upside" in result[trade_date].columns
    assert "cons_target_upside" not in result[trade_date].columns


def test_consensus_revision_rejects_incomplete_identity_schema():
    report_rc = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "report_date": ["20240320"],
            "np": [130.0],
        }
    )

    with pytest.raises(ValueError, match="report_rc 身份 schema 不完整"):
        _build_consensus_revision_lookup_by_date(report_rc, ["20240401"])
