"""北向资金因子单元测试"""

import numpy as np
import pandas as pd

from src.lazybull.factors.north_flow import (
    NORTH_COLS,
    build_north_flow_lookup_by_date,
)


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
    assert result["20240102"]["north_flow"] == 12.0
    assert result["20240103"]["north_flow"] == -4.0


def test_sign_streak_logic():
    df = pd.DataFrame(
        {
            "trade_date": [f"2024010{i}" for i in range(2, 7)],
            "north_money": [10.0, 20.0, -5.0, -8.0, 3.0],
        }
    )
    result = build_north_flow_lookup_by_date(df, df["trade_date"].tolist())
    # 前两日流入 -> streak=1,2; 第三四日流出 -> -1,-2; 第五日转正 -> 1
    assert result["20240102"]["north_flow_sign_streak"] == 1.0
    assert result["20240103"]["north_flow_sign_streak"] == 2.0
    assert result["20240104"]["north_flow_sign_streak"] == -1.0
    assert result["20240105"]["north_flow_sign_streak"] == -2.0
    assert result["20240106"]["north_flow_sign_streak"] == 1.0
