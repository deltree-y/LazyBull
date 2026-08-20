# -*- coding: utf-8 -*-
"""业绩预告因子测试：未知类型 NaN、同报告期修正版本、报告期优先查询。"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.announcement_utils import build_latest_announcement_lookup_by_date
from src.lazybull.factors.earnings import build_earnings_lookup_by_date


def _forecast_frame(records) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": r[0],
                "ann_date": r[1],
                "end_date": r[2],
                "type": r[3],
                "p_change_min": r[4],
                "p_change_max": r[5],
            }
            for r in records
        ]
    )


# ═══════════════════════════════════════════════════════════════
# 类型评分：未知/缺失与"不确定"区分
# ═══════════════════════════════════════════════════════════════


def test_unknown_type_is_nan_not_zero():
    df = _forecast_frame(
        [
            ("000001.SZ", "20240120", "20231231", "不确定", 0.0, 0.0),
            ("000002.SZ", "20240120", "20231231", "预增", 10.0, 20.0),
            ("000003.SZ", "20240120", "20231231", None, 10.0, 20.0),
            ("000004.SZ", "20240120", "20231231", "未知类型", 10.0, 20.0),
        ]
    )
    lookup = build_earnings_lookup_by_date(df, ["20240201"])
    rows = lookup["20240201"].set_index("ts_code")

    # "不确定"是已知的 0 分语义
    assert rows.loc["000001.SZ", "forecast_type_score"] == pytest.approx(0.0)
    # 未知/缺失类型保留 NaN，与"不确定"区分，交由模型 NaN 处理
    assert np.isnan(rows.loc["000003.SZ", "forecast_type_score"])
    assert np.isnan(rows.loc["000004.SZ", "forecast_type_score"])


def test_type_column_missing_scores_nan():
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["20240120"],
            "end_date": ["20231231"],
            "p_change_min": [10.0],
            "p_change_max": [20.0],
        }
    )
    lookup = build_earnings_lookup_by_date(df, ["20240201"])
    assert np.isnan(lookup["20240201"]["forecast_type_score"].iloc[0])


# ═══════════════════════════════════════════════════════════════
# 同报告期修正版本：修正公告后取修正版，修正前取首发版
# ═══════════════════════════════════════════════════════════════


def test_correction_version_wins_after_announcement():
    df = _forecast_frame(
        [
            ("000001.SZ", "20240410", "20240331", "预增", 10.0, 20.0),   # Q1 首发
            ("000001.SZ", "20240510", "20240331", "预减", -20.0, -10.0),  # Q1 修正
        ]
    )
    lookup = build_earnings_lookup_by_date(df, ["20240415", "20240515"])

    row = lookup["20240415"].iloc[0]
    assert row["forecast_type_score"] == pytest.approx(1.0)  # 修正未公告，取首发
    assert row["forecast_chg_mid"] == pytest.approx(15.0)

    row = lookup["20240515"].iloc[0]
    assert row["forecast_type_score"] == pytest.approx(-1.0)  # 修正版替代
    assert row["forecast_chg_mid"] == pytest.approx(-15.0)


def test_late_old_period_correction_does_not_override_new_period():
    """晚发的旧报告期修正公告不覆盖已公告的新报告期预告。"""
    df = _forecast_frame(
        [
            ("000001.SZ", "20251010", "20250930", "预增", 10.0, 20.0),   # Q3 首发
            ("000001.SZ", "20260110", "20251231", "预减", -20.0, -10.0),  # Q4 首发
            ("000001.SZ", "20260120", "20250930", "首亏", -100.0, -50.0),  # Q3 修正（晚发）
        ]
    )
    lookup = build_earnings_lookup_by_date(df, ["20251101", "20260115", "20260125"])

    # 修正前：取 Q3 首发
    row = lookup["20251101"].iloc[0]
    assert row["forecast_type_score"] == pytest.approx(1.0)

    # Q4 首发后：报告期优先取 Q4
    row = lookup["20260115"].iloc[0]
    assert row["forecast_type_score"] == pytest.approx(-1.0)
    assert row["forecast_freshness_days"] == 5  # 20260110 → 20260115

    # Q3 修正公告后：仍取报告期最新的 Q4，不被旧期修正覆盖
    row = lookup["20260125"].iloc[0]
    assert row["forecast_type_score"] == pytest.approx(-1.0)
    assert row["forecast_freshness_days"] == 15  # 20260110 → 20260125


# ═══════════════════════════════════════════════════════════════
# 通用查询函数 end_col 行为
# ═══════════════════════════════════════════════════════════════


def test_lookup_end_col_prefers_latest_report_date():
    factor_df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20240410", "end_date": "20240331", "val": 1.0},
            {"ts_code": "000001.SZ", "ann_date": "20240710", "end_date": "20240630", "val": 2.0},
            {"ts_code": "000001.SZ", "ann_date": "20240810", "end_date": "20240331", "val": 1.5},
        ]
    )
    lookup = build_latest_announcement_lookup_by_date(
        factor_df,
        ["20240715", "20240815"],
        value_cols=["val"],
        end_col="end_date",
    )
    # 旧期修正公告前：取新期
    assert lookup["20240715"]["val"].iloc[0] == pytest.approx(2.0)
    # 旧期修正公告后：仍取新期（end_date 20240630 > 20240331）
    assert lookup["20240815"]["val"].iloc[0] == pytest.approx(2.0)


def test_lookup_without_end_col_keeps_legacy_behavior():
    factor_df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20240710", "end_date": "20240630", "val": 2.0},
            {"ts_code": "000001.SZ", "ann_date": "20240810", "end_date": "20240331", "val": 1.5},
        ]
    )
    lookup = build_latest_announcement_lookup_by_date(
        factor_df,
        ["20240815"],
        value_cols=["val"],
    )
    # 未启用 end_col：按公告日取最新记录（旧行为）
    assert lookup["20240815"]["val"].iloc[0] == pytest.approx(1.5)


def test_lookup_end_col_with_missing_end_dates_drops_those_records():
    factor_df = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "ann_date": "20240410", "end_date": "20240331", "val": 1.0},
            {"ts_code": "000001.SZ", "ann_date": "20240420", "end_date": None, "val": 9.0},
        ]
    )
    lookup = build_latest_announcement_lookup_by_date(
        factor_df,
        ["20240425"],
        value_cols=["val"],
        end_col="end_date",
    )
    assert lookup["20240425"]["val"].iloc[0] == pytest.approx(1.0)
