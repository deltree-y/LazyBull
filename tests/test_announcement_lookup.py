# -*- coding: utf-8 -*-
"""风控公告类接线测试：lookup builder（PIT 前向填充）、因子处理器、端到端因子。"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.risk.announcement_lookup import (
    build_block_trade_lookup_by_date,
    build_pledge_lookup_by_date,
    build_share_float_lookup_by_date,
)
from src.lazybull.factors.risk.factor_registry import compute_all_risk_factors
from src.lazybull.features.handlers_announcement import (
    BlockTradeFactorHandler,
    PledgeFactorHandler,
    ShareFloatFactorHandler,
)

_TRADING_DATES = [
    "20240102",
    "20240103",
    "20240104",
    "20240105",
    "20240108",
    "20240109",
    "20240110",
    "20240111",
    "20240112",
    "20240115",
    "20240116",
    "20240117",
]


def _features_frame(ts_codes):
    return pd.DataFrame({"ts_code": ts_codes})


# ═══════════════════════════════════════════════════════════════
# 质押 lookup
# ═══════════════════════════════════════════════════════════════


def test_pledge_lookup_pit_forward_fill():
    raw = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240110",
                "end_date": "20231231",
                "pledge_ratio": 0.30,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240210",
                "end_date": "20240131",
                "pledge_ratio": 0.50,
            },
        ]
    )
    dates = ["20240101", "20240115", "20240215", "20240301"]
    lookup = build_pledge_lookup_by_date(raw, dates)

    # 公告前不可见
    assert "20240101" not in lookup
    # 公告后前向填充
    row = lookup["20240115"].iloc[0]
    assert row["pledge_ratio"] == pytest.approx(0.30)
    assert row["pledge_freshness_days"] == 5  # 20240110 → 20240115
    assert np.isnan(row["pledge_ratio_prev"])
    # 新公告覆盖 + prev 为上一条
    row2 = lookup["20240215"].iloc[0]
    assert row2["pledge_ratio"] == pytest.approx(0.50)
    assert row2["pledge_freshness_days"] == 5
    assert row2["pledge_ratio_prev"] == pytest.approx(0.30)
    # 更晚日期仍保持最新值
    row3 = lookup["20240301"].iloc[0]
    assert row3["pledge_ratio"] == pytest.approx(0.50)
    assert row3["pledge_freshness_days"] == 20  # 20240210 → 20240301（2024 闰年 2 月 29 天）


def test_pledge_lookup_fallback_to_end_date():
    raw = pd.DataFrame([{"ts_code": "000001.SZ", "end_date": "20231231", "pledge_ratio": 0.25}])
    lookup = build_pledge_lookup_by_date(raw, ["20240105", "20240120"])
    assert "20240105" in lookup
    row = lookup["20240120"].iloc[0]
    assert row["pledge_freshness_days"] == 20  # 20231231 → 20240120
    assert row["pledge_ratio"] == pytest.approx(0.25)


def test_pledge_lookup_empty_input():
    assert build_pledge_lookup_by_date(None, ["20240101"]) == {}
    assert build_pledge_lookup_by_date(pd.DataFrame(), ["20240101"]) == {}


# ═══════════════════════════════════════════════════════════════
# 限售解禁 lookup
# ═══════════════════════════════════════════════════════════════


def test_share_float_lookup_pit_and_nearest_unlock():
    raw = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240105",
                "float_date": "20240120",
                "float_ratio": 0.10,
            },
            {
                "ts_code": "000001.SZ",
                "ann_date": "20240115",
                "float_date": "20240210",
                "float_ratio": 0.20,
            },
        ]
    )
    dates = ["20240101", "20240110", "20240118", "20240220"]
    lookup = build_share_float_lookup_by_date(raw, dates)

    # 首条公告前不可见
    assert "20240101" not in lookup
    # 仅首条公告可见
    row = lookup["20240110"].iloc[0]
    assert row["days_to_unlock"] == 10  # 20240120 - 20240110
    assert row["unlock_ratio"] == pytest.approx(0.10)
    # 两条公告可见，取最近解禁日（20240120）
    row2 = lookup["20240118"].iloc[0]
    assert row2["days_to_unlock"] == 2
    assert row2["unlock_ratio"] == pytest.approx(0.10)
    # 所有解禁日已过 → 不再出现
    assert "20240220" not in lookup


def test_share_float_lookup_empty_input():
    assert build_share_float_lookup_by_date(None, ["20240101"]) == {}


# ═══════════════════════════════════════════════════════════════
# 大宗交易 lookup
# ═══════════════════════════════════════════════════════════════


def test_block_trade_lookup_discount_aggregation():
    raw = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20240102", "price": 9.5},  # 折价 -5%
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "price": 10.5,
            },  # 溢价 +5%（同日均值 0）
            {"ts_code": "000001.SZ", "trade_date": "20240110", "price": 9.0},  # 折价 -10%
        ]
    )
    close_lookup = {d: {"000001.SZ": 10.0} for d in _TRADING_DATES}
    lookup = build_block_trade_lookup_by_date(raw, _TRADING_DATES, close_lookup=close_lookup)

    # 首笔交易日
    row = lookup["20240102"].iloc[0]
    assert row["block_discount_avg_10d"] == pytest.approx(0.0)  # (-0.05+0.05)/2
    assert row["block_discount_days_10d"] == 1  # 有折价
    # 10 日窗口聚合到第二笔：20240102 同日两笔均值 0.0 + 20240110 一笔 -0.10 → 平均 -0.05
    row2 = lookup["20240110"].iloc[0]
    assert row2["block_discount_avg_10d"] == pytest.approx((0.0 + (-0.10)) / 2)
    assert row2["block_discount_days_10d"] == 2
    # 更晚日期（20240117，窗口 20240108..20240117）：仅剩 20240110 一笔
    row3 = lookup["20240117"].iloc[0]
    assert row3["block_discount_avg_10d"] == pytest.approx(-0.10)
    assert row3["block_discount_days_10d"] == 1


def test_block_trade_lookup_no_close_yields_empty():
    raw = pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20240102", "price": 9.5}])
    lookup = build_block_trade_lookup_by_date(raw, _TRADING_DATES, close_lookup=None)
    assert lookup == {}


# ═══════════════════════════════════════════════════════════════
# 因子处理器（接入 features）
# ═══════════════════════════════════════════════════════════════


def test_handlers_placeholder_on_empty_data():
    features = _features_frame(["000001.SZ", "600000.SH"])

    out = PledgeFactorHandler().apply(features.copy(), pd.DataFrame(), "20240101", None)
    for col in ("pledge_ratio", "pledge_freshness_days", "pledge_ratio_prev"):
        assert col in out and np.isnan(out[col])

    out2 = ShareFloatFactorHandler().apply(features.copy(), pd.DataFrame(), "20240101", None)
    for col in ("days_to_unlock", "unlock_ratio"):
        assert col in out2 and np.isnan(out2[col])

    out3 = BlockTradeFactorHandler().apply(features.copy(), pd.DataFrame(), "20240101", None)
    assert np.isnan(out3["block_discount_avg_10d"])
    assert out3["block_discount_days_10d"] == 0


def test_handlers_merge_wired_columns():
    features = _features_frame(["000001.SZ", "600000.SH"])
    data = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "pledge_ratio": 0.3,
                "pledge_freshness_days": 5,
                "pledge_ratio_prev": 0.2,
            },
        ]
    )
    out = PledgeFactorHandler().apply(features.copy(), data, "20240101", None)
    assert out["pledge_ratio"].iloc[0] == pytest.approx(0.3)
    assert np.isnan(out["pledge_ratio"].iloc[1])


# ═══════════════════════════════════════════════════════════════
# 端到端：原始列进入 features 后公告因子自动生效
# ═══════════════════════════════════════════════════════════════


def test_announcement_factors_from_wired_columns():
    features = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "pledge_ratio": [0.6, 0.1],
            "pledge_freshness_days": [10, 100],
            "pledge_ratio_prev": [0.4, 0.1],
            "days_to_unlock": [15.0, 200.0],
            "unlock_ratio": [0.05, 0.02],
            "block_discount_avg_10d": [-0.03, 0.0],
            "block_discount_days_10d": [2, 0],
        }
    )
    results = compute_all_risk_factors(df=features, daily_adj=None, market_state=None)

    # 质押：0.6 × exp(-10/30)，且 >50% 触发高危
    assert "pledge_ratio_decayed" in results
    assert results["pledge_ratio_decayed"].iloc[0] == pytest.approx(0.6 * np.exp(-10 / 30))
    assert results["pledge_high_flag"].iloc[0] == 1
    assert results["pledge_high_flag"].iloc[1] == -1  # 0.1 < 0.30
    # delta：0.6-0.4=0.2 保留；0.1-0.1=0 清零
    assert results["pledge_delta"].iloc[0] == pytest.approx(0.2)
    assert results["pledge_delta"].iloc[1] == 0.0
    # 解禁：15 天 → 危险档 2；200 天 → 安全 0
    assert results["unlock_risk_flag"].iloc[0] == 2
    assert results["unlock_risk_flag"].iloc[1] == 0
    assert results["unlock_ratio"].iloc[0] == pytest.approx(0.05)
    # 大宗：折价列直通
    assert results["block_discount_avg_10d"].iloc[0] == pytest.approx(-0.03)
    assert results["block_discount_days_10d"].iloc[0] == 2
