# -*- coding: utf-8 -*-
"""P4-T3 E2：新宿主统一 normalize/Kelly 语义与保活旧纸面 Oracle 对账。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.lazybull.paper.runner.pricing import PaperPricingMixin
from src.lazybull.paper.runner.signals import PaperSignalMixin
from src.lazybull.v2.core.decision import weighting
from src.lazybull.v2.core.execution.engine import BacktestEngine


class _LegacyOracle(PaperSignalMixin, PaperPricingMixin):
    """仅替换行情加载，直接执行保活旧纸面实现；不读取真实配置或数据。"""

    def __init__(self, sizing: str, price_data: pd.DataFrame, horizon: int = 20) -> None:
        self.position_sizing = sizing
        self.price_data = price_data
        self.horizon = horizon
        self.kelly_max_leverage = 1.0

    def _load_kelly_window_data(self, trade_date: str):
        return self.price_data[self.price_data["trade_date"] <= trade_date]


def _price_data(rows: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=rows)
    records = []
    for stock, base, phase in (("A", 10.0, 0), ("B", 20.0, 1), ("C", 30.0, 2)):
        for index, date in enumerate(dates):
            close = base * (1 + 0.002 * index + 0.01 * np.sin(index / 3 + phase))
            records.append(
                {
                    "ts_code": stock,
                    "trade_date": date.strftime("%Y%m%d"),
                    "close": close,
                    "close_adj": close,
                }
            )
    return pd.DataFrame(records)


def _new(signals, sizing, prices=None, window=60):
    prices = _price_data() if prices is None else prices
    result, _ = weighting.normalize_signals(
        dict(signals),
        pd.Timestamp("2025-04-15"),
        position_sizing=sizing,
        normalize_log_count=5,
        verbose=False,
        kelly_max_leverage=1.0,
        price_data_cache=prices,
        kelly_vol_window=window,
    )
    return result


def _old(signals, sizing, prices=None):
    prices = _price_data() if prices is None else prices
    oracle = _LegacyOracle(sizing, prices)
    return oracle._normalize_signals(dict(signals), "20250415")


@pytest.mark.parametrize("mode", ["equal", "score", "kelly", "half_kelly"])
def test_non_divergent_positive_scores_exact_for_four_modes(mode):
    signals = {"A": 0.9, "B": 0.5, "C": 0.25}
    assert _new(signals, mode) == _old(signals, mode)


def test_score_negative_scores_use_backtest_full_sum_adjudication():
    signals = {"A": 0.8, "B": -0.2, "C": 0.1}
    expected = {
        "A": pytest.approx(0.8 / 0.7),
        "B": pytest.approx(-0.2 / 0.7),
        "C": pytest.approx(0.1 / 0.7),
    }
    assert _new(signals, "score") == expected
    assert _old(signals, "score") == {"A": pytest.approx(0.8 / 0.9), "C": pytest.approx(0.1 / 0.9)}


def test_score_positive_subset_but_nonpositive_total_falls_back_to_equal():
    signals = {"A": 1.0, "B": -3.0, "C": 0.5}
    assert _new(signals, "score") == {key: 1 / 3 for key in signals}
    assert _old(signals, "score") == {"A": 2 / 3, "C": 1 / 3}


def test_score_tiny_positive_total_adjudicates_full_sum_not_positive_subset():
    signals = {"A": 1e-13, "B": -1e-14}
    result = _new(signals, "score")
    assert result["A"] == pytest.approx(10 / 9)
    assert result["B"] == pytest.approx(-1 / 9)
    assert _old(signals, "score") == {"A": 1.0}


def test_all_nonpositive_scores_remain_exact_equal_weights():
    signals = {"A": -0.2, "B": 0.0, "C": -0.4}
    assert _new(signals, "score") == _old(signals, "score") == {key: 1 / 3 for key in signals}


def test_unknown_sizing_new_host_is_silent_equal_instead_of_legacy_score():
    signals = {"A": 0.9, "B": 0.1}
    assert _new(signals, "unknown") == {"A": 0.5, "B": 0.5}
    assert _old(signals, "unknown") == {"A": 0.9, "B": 0.1}


@pytest.mark.parametrize("mode", ["kelly", "half_kelly"])
def test_kelly_vol_window_uses_configured_window(mode):
    signals = {"A": 0.9, "B": 0.5, "C": 0.25}
    prices = _price_data()
    assert _new(signals, mode, prices, window=60) == _old(signals, mode, prices)
    assert _new(signals, mode, prices, window=20) != _old(signals, mode, prices)


def test_engine_wrapper_delegates_to_unique_weighting_implementation(monkeypatch):
    expected = {"A": 1.0}
    calls = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return expected, 7

    monkeypatch.setattr("src.lazybull.v2.core.execution.engine.normalize_signals", _spy)
    engine = BacktestEngine.__new__(BacktestEngine)
    engine.position_sizing = "equal"
    engine._normalize_log_count = 6
    engine.verbose = False
    engine.kelly_max_leverage = 0.2
    engine.price_data_cache = None
    engine.kelly_vol_window = 60
    assert engine._normalize_signals({"A": 0.8}, pd.Timestamp("2025-04-15")) is expected
    assert engine._normalize_log_count == 7
    assert len(calls) == 1
