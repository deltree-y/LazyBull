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
            "pledge_ratio": [60.0, 10.0],
            "pledge_freshness_days": [10, 100],
            "pledge_ratio_prev": [40.0, 10.0],
            "days_to_unlock": [15.0, 200.0],
            "unlock_ratio": [0.05, 0.02],
            "block_discount_avg_10d": [-0.03, 0.0],
            "block_discount_days_10d": [2, 0],
        }
    )
    results = compute_all_risk_factors(df=features, daily_adj=None, market_state=None)

    # 质押（百分比口径）：60 × exp(-10/30)，且 >50% 触发高危
    assert "pledge_ratio_decayed" in results
    assert results["pledge_ratio_decayed"].iloc[0] == pytest.approx(60.0 * np.exp(-10 / 30))
    assert results["pledge_high_flag"].iloc[0] == 1  # 60% > 50% 高危
    assert results["pledge_high_flag"].iloc[1] == -1  # 10% < 30% 安全
    # delta（百分点）：60-40=20 保留；10-10=0 清零
    assert results["pledge_delta"].iloc[0] == pytest.approx(20.0)
    assert results["pledge_delta"].iloc[1] == 0.0
    # 解禁：15 天 → 危险档 2；200 天 → 安全 0
    assert results["unlock_risk_flag"].iloc[0] == 2
    assert results["unlock_risk_flag"].iloc[1] == 0
    # 透传列（unlock_ratio/block_discount_*）已由 handler 提供，因子层不再重复输出
    assert "unlock_ratio" not in results
    assert "block_discount_avg_10d" not in results
    assert "block_discount_days_10d" not in results


def test_pledge_high_flag_percentage_threshold():
    """回归：pledge_ratio 为百分比口径（0-100），分档阈值必须是 50/30。

    旧 bug：阈值按小数 0.50 写，导致 0.6%（0.6）的股票被判高危 1，
    实测 90% 有质押股票全部误判高危（pledge_high_flag median=1.0）。
    """
    features = pd.DataFrame(
        {
            "ts_code": ["a", "b", "c", "d", "e", "f", "g"],
            "pledge_ratio": [0.6, 29.0, 30.0, 50.0, 51.0, 75.09, np.nan],
        }
    )
    results = compute_all_risk_factors(df=features, daily_adj=None, market_state=None)
    flag = results["pledge_high_flag"]
    assert flag.iloc[0] == -1  # 0.6% < 30% → 安全（旧代码误判为 1）
    assert flag.iloc[1] == -1  # 29% < 30% → 安全
    assert flag.iloc[2] == 0  # 30% 边界 → 中性
    assert flag.iloc[3] == 0  # 50% 边界 → 中性（>50 才高危）
    assert flag.iloc[4] == 1  # 51% > 50% → 高危
    assert flag.iloc[5] == 1  # 75.09% → 高危
    assert flag.iloc[6] == 0  # 缺数据 → 0


def test_pledge_delta_threshold_percentage_points():
    """回归：pledge_delta 的实质变化阈值应为 0.5 个百分点（百分比口径）。

    旧 bug：0.005 阈值（0.005 个百分点）过严，几乎任何变化都保留为噪声。
    """
    features = pd.DataFrame(
        {
            "ts_code": ["a", "b"],
            "pledge_ratio": [50.4, 51.0],
            "pledge_freshness_days": [1, 1],
            "pledge_ratio_prev": [50.0, 50.0],
        }
    )
    results = compute_all_risk_factors(df=features, daily_adj=None, market_state=None)
    delta = results["pledge_delta"]
    assert delta.iloc[0] == 0.0  # +0.4 个百分点 < 0.5 → 清零
    assert delta.iloc[1] == pytest.approx(1.0)  # +1.0 个百分点 → 保留


def test_attach_risk_factors_no_duplicate_columns():
    """回归：handler 合并原始列后，_attach_risk_factors_static 不得产生重复列。

    真实场景暴露的 bug：features 已含 unlock_ratio/block_discount_*/short_balance_change_5
    原始列，透传型因子重复输出同名列导致 pd.concat 报 Duplicate column names。
    """
    from src.lazybull.features.builder import _attach_risk_factors_static
    from src.lazybull.risk.precompute import PRECOMPUTED_RISK_FACTOR_NAMES

    features = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "pledge_ratio": [60.0, 10.0],
            "pledge_freshness_days": [10, 100],
            "pledge_ratio_prev": [40.0, 10.0],
            "days_to_unlock": [15.0, 200.0],
            "unlock_ratio": [0.05, 0.02],
            "block_discount_avg_10d": [-0.03, 0.0],
            "block_discount_days_10d": [2, 0],
            "short_balance_change_5": [0.1, -0.1],
        }
    )
    result = _attach_risk_factors_static(features, "20240115", {}, PRECOMPUTED_RISK_FACTOR_NAMES)
    # 无重复列：所有列名唯一
    dupes = [c for c in result.columns if (result.columns == c).sum() > 1]
    assert dupes == [], f"存在重复列: {dupes}"
    # 原始列保留原值，加工因子（unlock_risk_flag）仍生成
    assert (result["unlock_ratio"] == features["unlock_ratio"]).all()
    assert "unlock_risk_flag" in result.columns
    assert "pledge_high_flag" in result.columns
