"""北向资金因子单元测试"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.north_flow import (
    NORTH_COLS,
    NORTH_NET_BUY_COLS,
    NORTH_TURNOVER_COLS,
    NORTH_TURNOVER_SWITCH_DATE,
    build_north_flow_lookup_by_date,
)
from src.lazybull.ml.train_core.constants import NORTH_FEATURE_COLUMNS


def _make_hsgt_df(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="B").strftime("%Y%m%d")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "hgt": rng.normal(0, 10, size=n),
            "sgt": rng.normal(0, 8, size=n),
            "north_money": rng.normal(5, 15, size=n),
        }
    )


def test_north_flow_lookup_basic():
    df = _make_hsgt_df(30)
    trading_dates = df["trade_date"].tolist()
    result = build_north_flow_lookup_by_date(df, trading_dates)

    assert len(result) == 30, "应该覆盖全部交易日"
    for td, rec in result.items():
        assert isinstance(rec, dict)
        for col in NORTH_COLS:
            assert col in rec, f"{td} 缺少列 {col}"


def test_north_output_schema_matches_training_columns():
    """因子生产列与主模型 north 训练列必须严格一致。"""
    assert NORTH_FEATURE_COLUMNS == NORTH_COLS


def test_north_flow_empty_input():
    assert build_north_flow_lookup_by_date(pd.DataFrame(), ["20240102"]) == {}
    assert build_north_flow_lookup_by_date(None, ["20240102"]) == {}


def test_north_flow_fallback_from_hgt_sgt():
    df = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103", "20240104"],
            "hgt": [10.0, -5.0, 3.0],
            "sgt": [2.0, 1.0, -4.0],
        }
    )
    result = build_north_flow_lookup_by_date(df, df["trade_date"].tolist())
    # 原始单位为百万元, 统一 ÷100 换算亿元
    assert np.isclose(result["20240102"]["north_net_buy"], 0.12)
    assert np.isclose(result["20240103"]["north_net_buy"], -0.04)
    assert result["20240102"]["north_turnover"] == 0.0


def test_sign_streak_logic():
    df = pd.DataFrame(
        {
            "trade_date": [f"2024010{i}" for i in range(2, 7)],
            "north_money": [10.0, 20.0, -5.0, -8.0, 3.0],
        }
    )
    result = build_north_flow_lookup_by_date(df, df["trade_date"].tolist())
    # 前两日流入 -> streak=1,2; 第三四日流出 -> -1,-2; 第五日转正 -> 1
    assert result["20240102"]["north_net_buy_sign_streak"] == 1.0
    assert result["20240103"]["north_net_buy_sign_streak"] == 2.0
    assert result["20240104"]["north_net_buy_sign_streak"] == -1.0
    assert result["20240105"]["north_net_buy_sign_streak"] == -2.0
    assert result["20240106"]["north_net_buy_sign_streak"] == 1.0


def test_north_flow_unit_scale_and_streak_by_era():
    """全程百万元 ÷100 统一亿元; 成交额口径 streak 按环比方向计算（全期有值）。"""
    df = pd.DataFrame(
        {
            "trade_date": ["20240816", "20240819", "20240820", "20240821"],
            "north_money": [-6774.99, 88110.55, 89201.95, 84789.27],
        }
    )
    result = build_north_flow_lookup_by_date(df, df["trade_date"].tolist())
    # 单位: 切换前净买入同为百万元, 全期 ÷100
    assert np.isclose(result["20240816"]["north_net_buy"], -67.7499)
    assert result["20240816"]["north_turnover"] == 0.0
    assert result["20240819"]["north_net_buy"] == 0.0
    assert np.isclose(result["20240819"]["north_turnover"], 881.1055)
    # 口径指示列: 切换前 0, 切换后 1
    assert result["20240816"]["north_turnover_flag"] == 0.0
    assert result["20240819"]["north_turnover_flag"] == 1.0
    # 切换前 streak = 净流入符号
    assert result["20240816"]["north_net_buy_sign_streak"] == -1.0
    # 切换后 streak = 成交额环比方向: 首日 diff 填 0 -> 0, 升 -> +1, 降 -> -1
    assert result["20240819"]["north_turnover_change_streak"] == 0.0
    assert result["20240820"]["north_turnover_change_streak"] == 1.0
    assert result["20240821"]["north_turnover_change_streak"] == -1.0


def test_north_regime_columns_are_mutually_exclusive():
    """两套口径列互斥，跨制度 OOS 不会把成交额送入净买入列。"""
    df = pd.DataFrame(
        {
            "trade_date": ["20240816", "20240819", "20240820"],
            "north_money": [-1000.0, 80000.0, 90000.0],
        }
    )
    result = build_north_flow_lookup_by_date(df, df["trade_date"].tolist())

    assert all(result["20240816"][col] == 0.0 for col in NORTH_TURNOVER_COLS)
    assert all(result["20240819"][col] == 0.0 for col in NORTH_NET_BUY_COLS)
    assert all(result["20240820"][col] == 0.0 for col in NORTH_NET_BUY_COLS)
    assert result["20240816"]["north_net_buy"] == -10.0
    assert result["20240819"]["north_turnover"] == 800.0


def test_north_flow_neutralizes_internal_market_holiday_without_hiding_trailing_gap():
    """源数据区间内的港股休市日取中性值，末尾下载缺口仍保持缺失。"""
    df = pd.DataFrame(
        {
            "trade_date": ["20260630", "20260702"],
            "north_money": [364452.41, 424018.05],
        }
    )

    result = build_north_flow_lookup_by_date(
        df,
        ["20260630", "20260701", "20260702", "20260703"],
    )

    assert all(result["20260701"][column] == 0.0 for column in NORTH_NET_BUY_COLS)
    assert all(result["20260701"][column] == 0.0 for column in NORTH_TURNOVER_COLS)
    assert result["20260701"]["north_turnover_flag"] == 1.0
    assert "20260703" not in result


def test_north_flow_internal_holiday_participates_in_following_rolling_window():
    """内部休市的零成交额应进入后续滚动窗口，不能跨日取更早记录。"""
    df = pd.DataFrame(
        {
            "trade_date": ["20260701", "20260702", "20260706"],
            "north_money": [10_000.0, 10_000.0, 10_000.0],
        }
    )
    calendar_dates = ["20260701", "20260702", "20260703", "20260706"]

    result = build_north_flow_lookup_by_date(df, calendar_dates)

    assert result["20260703"]["north_turnover_ma5"] == 0.0
    assert result["20260706"]["north_turnover_ma5"] == pytest.approx(75.0)


def test_north_flow_rolling_window_no_cross_era():
    """滚动窗口不跨口径切换日: 断点后首日窗口仅含成交额口径自身。"""
    assert NORTH_TURNOVER_SWITCH_DATE == "20240819"
    df = pd.DataFrame(
        {
            "trade_date": ["20240815", "20240816", "20240819", "20240820"],
            # 前两日=净买入（百万元），后两日=成交额（百万元），全期 ÷100 -> 亿元
            "north_money": [-10000.0, -20000.0, 10000.0, 20000.0],
        }
    )
    result = build_north_flow_lookup_by_date(df, df["trade_date"].tolist())
    # 断点后首日: 组内仅 1 天, 窗口不吞断点前负值
    assert np.isclose(result["20240819"]["north_turnover_ma20"], 100.0)
    assert np.isclose(result["20240819"]["north_turnover_sum5"], 100.0)
    assert np.isclose(result["20240819"]["north_turnover_ma5"], 100.0)
    # 断点后 z20 预热不足（std min_periods=5）置 0 中性（避免全空拒绝预测）
    assert result["20240819"]["north_turnover_z20"] == 0.0
    # 断点前最后一日: 窗口不含断点后数据（-100, -200 亿元均值/累计）
    assert np.isclose(result["20240816"]["north_net_buy_ma20"], -150.0)
    assert np.isclose(result["20240816"]["north_net_buy_sum5"], -300.0)
    assert result["20240816"]["north_turnover_sum5"] == 0.0
    assert result["20240819"]["north_net_buy_sum5"] == 0.0


def test_sign_streak_window_cap():
    """streak 窗口化为近 20 日: 连续同方向超过 20 日封顶 20, 不随加载起点漂移。"""
    dates = pd.date_range("2024-01-01", periods=25, freq="B").strftime("%Y%m%d")
    df = pd.DataFrame({"trade_date": dates, "north_money": [10.0] * 25})
    result = build_north_flow_lookup_by_date(df, dates.tolist())
    # 第 6 天 -> streak=6; 最后一天连续 25 日但窗口封顶 20
    assert result[dates[5]]["north_net_buy_sign_streak"] == 6.0
    assert result[dates[-1]]["north_net_buy_sign_streak"] == 20.0
