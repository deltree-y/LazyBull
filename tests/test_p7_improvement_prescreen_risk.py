"""共同风险首关的经济语义合成测试；不依赖生产配置或数据。"""

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.improvement_prescreen_risk import (
    bounded_minimum_variance,
    common_factor_metrics,
    correlation_graph,
    diagnose,
    industry_at,
    past_window,
    simple_returns,
)


def test_future_prices_do_not_enter_observation_window():
    dates = pd.bdate_range("2020-01-01", periods=150).strftime("%Y%m%d").tolist()
    prices = pd.DataFrame({"a": np.exp(np.arange(150) * 0.001)}, index=dates)
    before = past_window(simple_returns(prices, dates), dates[125])
    future_start = dates[126]
    prices.loc[future_start:, "a"] *= 100
    after = past_window(simple_returns(prices, dates), dates[125])
    pd.testing.assert_frame_equal(before, after)
    assert len(after) == 120
    assert after.index.max() == dates[125]


def test_missing_calendar_day_does_not_bridge_returns():
    prices = pd.DataFrame({"a": [100.0, 121.0, 133.1]}, index=["20200102", "20200106", "20200107"])
    calendar = ["20200102", "20200103", "20200106", "20200107"]
    result = simple_returns(prices, calendar)
    assert pd.isna(result.loc["20200103", "a"])
    assert pd.isna(result.loc["20200106", "a"])
    assert result.loc["20200107", "a"] == pytest.approx(0.1)


def test_ols_common_factor_and_residual_partition_variance():
    market = np.array([-2, -1, 0, 1, 2], dtype=float) / 100
    residual = np.array([1, -2, 2, -2, 1], dtype=float) / 100
    assert market @ residual == pytest.approx(0)
    stock = 2 * market + residual + 0.003
    result = common_factor_metrics(stock[:, None], np.array([1.0]), market)
    expected = 4 * np.var(market, ddof=1) / np.var(stock, ddof=1)
    assert result["market_beta"] == pytest.approx(2)
    assert result["common_factor_variance_share"] == pytest.approx(expected)
    assert result["common_factor_variance_share"] + result[
        "residual_variance_share"
    ] == pytest.approx(1)


def test_minimum_variance_preserves_budget_and_individual_bounds():
    covariance = np.diag([0.00001, 0.0004, 0.0009])
    weights = np.array([0.2, 0.3, 0.5])
    fitted = bounded_minimum_variance(covariance, weights)
    assert fitted.sum() == pytest.approx(1)
    assert np.all(fitted >= weights * 0.5 - 1e-8)
    assert np.all(fitted <= weights * 1.5 + 1e-8)
    assert fitted[0] == pytest.approx(0.3, abs=1e-6)
    assert fitted @ covariance @ fitted < weights @ covariance @ weights


def test_unknown_industry_is_neither_same_nor_cross_industry():
    correlation = np.array([[1, 0.8, 0.7], [0.8, 1, 0.9], [0.7, 0.9, 1]])
    result, edges, clusters = correlation_graph(
        correlation, np.array([0.2, 0.3, 0.5]), ["a", "b", "c"], ["L1", None, "L2"]
    )
    assert result["high_correlation_edges"] == 3
    assert result["unknown_industry_edges"] == 2
    assert result["known_industry_edges"] == 1
    assert result["cross_l1_edge_share"] == 1
    assert len(clusters) == 1
    assert clusters[0]["weight"] == pytest.approx(1)
    assert sum(edge["cross_l1"] is None for edge in edges) == 2


def test_industry_intervals_are_left_closed_right_open_and_conflicts_unknown():
    pit = pd.DataFrame(
        {
            "ts_code": ["a", "a", "b", "b"],
            "in_date": ["20100101", "20200103", "20100101", "20100101"],
            "out_date": ["20200103", "99991231", "99991231", "99991231"],
            "l1_code": ["old", "new", "x", "y"],
        }
    )
    assert industry_at(pit, ["a", "b", "c"], "20200103") == ["new", None, None]


def test_insufficient_common_days_excludes_entire_observation_not_stock():
    dates = pd.bdate_range("2020-01-01", periods=120).strftime("%Y%m%d")
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(0, 0.01, size=(120, 2)), index=dates, columns=["a", "b"])
    frame.iloc[:31, 1] = np.nan
    observation = {
        "group": "synthetic",
        "fold": "",
        "trade_date": dates[-1],
        "observation_id": "test",
        "codes": ["a", "b"],
        "weights": [0.5, 0.5],
    }
    pit = pd.DataFrame(columns=["ts_code", "in_date", "out_date", "l1_code"])
    result, stocks, edges, clusters = diagnose(observation, frame, frame.mean(axis=1), pit)
    assert result["stock_count"] == 2
    assert result["stock_common_days"] == 89
    assert result["status"] == "not_estimable"
    assert stocks == edges == clusters == []
