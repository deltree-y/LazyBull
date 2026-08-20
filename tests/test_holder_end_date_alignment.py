# -*- coding: utf-8 -*-
"""股东人数因子跨报告期环比对齐测试（同报告期修正版本不稀释信号）。"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.holder import build_holder_lookup_by_date


def _holder_frame(records) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": r[0],
                "ann_date": r[1],
                "end_date": r[2],
                "holder_num": r[3],
            }
            for r in records
        ]
    )


def test_cross_period_chg_uses_latest_announced_value():
    """同报告期首发+修正版本：环比基准取修正版（跨期），而非同期上一版。"""
    df = _holder_frame(
        [
            ("000001.SZ", "20240110", "20231231", 120),  # Q0
            ("000001.SZ", "20240410", "20240331", 100),  # Q1 首发
            ("000001.SZ", "20240510", "20240331", 90),   # Q1 修正
            ("000001.SZ", "20240710", "20240630", 80),   # Q2
        ]
    )
    lookup = build_holder_lookup_by_date(
        df, ["20240411", "20240511", "20240711"]
    )

    # 首发公告后：取 Q1 首发版（修正版尚未公告）
    row = lookup["20240411"].iloc[0]
    assert row["holder_num_chg"] == pytest.approx((100 - 120) / 120)
    assert row["holder_freshness_days"] == 1

    # 修正公告后：同报告期取修正版，chg 仍为跨期（90 对 Q0 的 120）
    row = lookup["20240511"].iloc[0]
    assert row["holder_num_chg"] == pytest.approx((90 - 120) / 120)
    assert row["holder_freshness_days"] == 1

    # Q2：基准为 Q1 修正版 90（而非 Q1 首发版 100）
    row = lookup["20240711"].iloc[0]
    assert row["holder_num_chg"] == pytest.approx((80 - 90) / 90)
    assert row["holder_num_chg_2q"] == pytest.approx((80 - 120) / 120)


def test_late_old_period_correction_does_not_override_new_period():
    """晚发的旧报告期修正公告不覆盖已公告的新报告期。"""
    df = _holder_frame(
        [
            ("000001.SZ", "20240110", "20231231", 120),  # Q0
            ("000001.SZ", "20240410", "20240331", 100),  # Q1 首发
            ("000001.SZ", "20240710", "20240630", 80),   # Q2 首发（早于 Q1 修正）
            ("000001.SZ", "20240810", "20240331", 90),   # Q1 修正（晚发）
        ]
    )
    lookup = build_holder_lookup_by_date(df, ["20240715", "20240815"])

    # 修正公告前：Q2 的基准是当时可见的 Q1 首发版 100
    row = lookup["20240715"].iloc[0]
    assert row["holder_num_chg"] == pytest.approx((80 - 100) / 100)

    # 修正公告后：仍选报告期最新的 Q2（end_date 0630 > 0331），值不变
    row = lookup["20240815"].iloc[0]
    assert row["holder_num_chg"] == pytest.approx((80 - 100) / 100)
    assert row["holder_freshness_days"] == 36  # 20240710 → 20240815


def test_first_period_chg_is_nan():
    """首期无基准：环比为 NaN。"""
    df = _holder_frame([("000001.SZ", "20240110", "20231231", 120)])
    lookup = build_holder_lookup_by_date(df, ["20240201"])
    row = lookup["20240201"].iloc[0]
    assert np.isnan(row["holder_num_chg"])
    assert np.isnan(row["holder_num_chg_2q"])


def test_cross_stock_isolation():
    """不同股票的公告不互相影响基准。"""
    df = _holder_frame(
        [
            ("000001.SZ", "20240110", "20231231", 120),
            ("000001.SZ", "20240410", "20240331", 100),
            ("000002.SZ", "20240410", "20240331", 50),  # 另一只股票首期
        ]
    )
    lookup = build_holder_lookup_by_date(df, ["20240411"])
    rows = lookup["20240411"].set_index("ts_code")
    assert rows.loc["000001.SZ", "holder_num_chg"] == pytest.approx((100 - 120) / 120)
    assert np.isnan(rows.loc["000002.SZ", "holder_num_chg"])


def test_empty_input_returns_empty_lookup_without_error():
    """空输入边界：不应因基准列缺失而 KeyError（回归保护）。"""
    df = _holder_frame([])
    lookup = build_holder_lookup_by_date(
        df.reindex(columns=["ts_code", "ann_date", "end_date", "holder_num"]),
        ["20260101"],
    )
    assert lookup == {}
