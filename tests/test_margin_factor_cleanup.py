# -*- coding: utf-8 -*-
"""融资融券因子清理回归测试（v0.95.2）。

覆盖：
1. lookup 输出列集合：主模型 3 列 + 风控 3 列，不再输出幽灵列 margin_net_buy_ratio；
2. MARGIN_COLS / MARGIN_RISK_COLS 清单划分与训练列清单一致性；
3. 变化率 / 比率 / 净买入中间列数值正确性；
4. 融券源列缺失时风控列仍存在（全 NaN），保证合并链路列稳定；
5. 缓存完整性补检包含风控融券列。
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.margin import MARGIN_COLS, MARGIN_RISK_COLS, build_margin_lookup_by_date
from src.lazybull.features.ensure.schema import _REQUIRED_FACTOR_COLS
from src.lazybull.ml.train_core.constants import MARGIN_FEATURE_COLUMNS


def _make_margin_detail(n_stocks: int = 3, n_days: int = 25, drop_cols=None) -> pd.DataFrame:
    """构造线性增长的 margin_detail 样例数据（不依赖真实数据）。"""
    rows = []
    for si in range(n_stocks):
        ts_code = f"{si:06d}.SZ"
        for di in range(n_days):
            day = di + 1
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": f"202601{day:02d}",
                    "rzye": 1_000_000.0 + si * 100_000.0 + day * 1000.0,
                    "rqye": 50_000.0 + day * 100.0,
                    "rzmre": 100_000.0 + day * 10.0,
                    "rzche": 60_000.0 + day * 5.0,
                    "rqmcl": 1_000.0 + day,
                }
            )
    df = pd.DataFrame(rows)
    if drop_cols:
        df = df.drop(columns=list(drop_cols))
    return df


def test_margin_lookup_output_columns():
    """lookup 输出列 = 主模型 3 列 + 风控 3 列 + ts_code，不含幽灵列。"""
    n_days = 25
    df = _make_margin_detail(n_days=n_days)
    trading_dates = [f"202601{day:02d}" for day in range(1, n_days + 1)]

    lookup = build_margin_lookup_by_date(df, trading_dates)
    last_day = trading_dates[-1]
    today = lookup[last_day]

    expected = {"ts_code"} | set(MARGIN_COLS) | set(MARGIN_RISK_COLS)
    assert set(today.columns) == expected
    assert "margin_net_buy_ratio" not in today.columns
    # 每日期截面均具备相同列集合，保证合并链路 schema 稳定
    for trade_date in trading_dates:
        assert set(lookup[trade_date].columns) == expected


def test_margin_col_lists_contract():
    """主模型/风控列清单划分与训练列清单一致，且均不含幽灵列。"""
    assert MARGIN_COLS == ["rzye_chg_5", "rzye_chg_20", "rqye_rzye_ratio"]
    assert MARGIN_RISK_COLS == [
        "margin_net_buy",
        "short_balance_change_5",
        "short_sell_vol_change_5",
    ]
    assert "margin_net_buy_ratio" not in MARGIN_COLS
    assert "margin_net_buy_ratio" not in MARGIN_RISK_COLS
    # 训练列清单与主模型列清单保持一致
    assert MARGIN_FEATURE_COLUMNS == MARGIN_COLS


def test_margin_change_rate_and_ratio_values():
    """变化率 / 多空比 / 净买入中间列数值正确性。"""
    n_days = 25
    df = _make_margin_detail(n_stocks=1, n_days=n_days)
    trading_dates = [f"202601{day:02d}" for day in range(1, n_days + 1)]
    lookup = build_margin_lookup_by_date(df, trading_dates)

    # day=21 时 shift(5) 对应 day=16，rzye 线性增长 step=1000
    today = lookup["20260121"].iloc[0]
    rzye_16 = 1_000_000.0 + 16 * 1000.0
    rzye_21 = 1_000_000.0 + 21 * 1000.0
    expected_chg_5 = (rzye_21 - rzye_16) / rzye_16
    assert today["rzye_chg_5"] == pytest.approx(expected_chg_5)

    # day=21 时 shift(20) 对应 day=1
    rzye_1 = 1_000_000.0 + 1000.0
    expected_chg_20 = (rzye_21 - rzye_1) / rzye_1
    assert today["rzye_chg_20"] == pytest.approx(expected_chg_20)

    # 多空比与净买入中间列
    assert today["rqye_rzye_ratio"] == pytest.approx(
        (50_000.0 + 21 * 100.0) / rzye_21
    )
    assert today["margin_net_buy"] == pytest.approx(
        (100_000.0 + 21 * 10.0) - (60_000.0 + 21 * 5.0)
    )

    # 融券变化率：day=21 与 day=16 的 rqye
    rqye_16 = 50_000.0 + 16 * 100.0
    rqye_21 = 50_000.0 + 21 * 100.0
    assert today["short_balance_change_5"] == pytest.approx((rqye_21 - rqye_16) / rqye_16)


def test_margin_lookup_missing_short_columns_still_present():
    """融券源列缺失时风控列仍输出（全 NaN），保证全列合并 schema 稳定。"""
    df = _make_margin_detail(drop_cols=["rqye", "rqmcl"])
    trading_dates = [f"202601{day:02d}" for day in range(1, 26)]
    lookup = build_margin_lookup_by_date(df, trading_dates)
    today = lookup["20260121"]

    assert "short_balance_change_5" in today.columns
    assert "short_sell_vol_change_5" in today.columns
    assert today["short_balance_change_5"].isna().all()
    assert today["short_sell_vol_change_5"].isna().all()


def test_required_factor_cols_include_risk_margin_col():
    """缓存完整性补检包含风控融券列，旧 cs_infer 缓存缺列会触发重建。"""
    assert "short_balance_change_5" in _REQUIRED_FACTOR_COLS
