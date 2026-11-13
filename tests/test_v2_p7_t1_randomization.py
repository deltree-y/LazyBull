# -*- coding: utf-8 -*-
"""P7 T1 内层随机化、临界值、误放行与功效的定点回归（设计 §6.5）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v2_p7.t1 import randomization  # noqa: E402


def test_null_distribution_is_deterministic_per_seed() -> None:
    members = np.array(
        [
            [0.10, 0.00, -0.02, 0.01, 0.03, -0.01],
            [0.05, 0.02, 0.01, -0.01, 0.00, 0.02],
            [0.20, 0.10, 0.00, -0.05, 0.03, 0.01],
        ]
    )
    dates = np.array([0, 0, 1])
    first = randomization.null_distribution(members, dates, block_h=2, implementations=200, seed=42)
    second = randomization.null_distribution(
        members, dates, block_h=2, implementations=200, seed=42
    )
    third = randomization.null_distribution(
        members, dates, block_h=2, implementations=200, seed=1042
    )
    assert np.array_equal(first.values, second.values)
    assert not np.array_equal(first.values, third.values)
    assert first.values.size == 200


def test_null_is_centered_when_one_group_only() -> None:
    members = np.array([[0.10, 0.0, 0.0, 0.0, 0.0, 0.0]])
    result = randomization.null_distribution(
        members, np.array([0]), block_h=1, implementations=6000, seed=7
    )
    assert set(np.round(result.values, 6)) <= {0.1, -0.02}
    assert result.values.mean() == pytest.approx(0.0, abs=0.01)


def test_unfilled_groups_enter_denominator_as_true_zero() -> None:
    """未成交组按真实 0 残差参与意图均值（分母不缩水）。"""
    members = np.array(
        [
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    dates = np.array([0, 0])
    intent = np.array([0.10, 0.0])
    statistics = randomization.observed_statistic(
        intent, np.array([True, False]), np.array([False, True])
    )
    assert statistics.intents == 2
    assert statistics.statistic_intent == pytest.approx(0.05)
    assert statistics.statistic_filled_only == pytest.approx(0.10)
    result = randomization.null_distribution(
        members, dates, block_h=1, implementations=2000, seed=42
    )
    assert result.values.mean() == pytest.approx(0.0, abs=0.02)


def test_missing_outcome_makes_intent_statistic_undefined() -> None:
    values = np.array([0.10, np.nan])
    statistics = randomization.observed_statistic(
        values, np.array([True, False]), np.array([False, False])
    )
    assert statistics.statistic_intent is None
    assert statistics.statistic_filled_only == pytest.approx(0.10)


def test_critical_value_and_clopper_pearson() -> None:
    values = np.arange(1000, dtype=float)
    assert randomization.critical_value(values, 0.9) == pytest.approx(899.1, abs=1e-6)
    assert randomization.critical_value(np.array([]), 0.9) is None
    assert randomization.clopper_pearson_upper(0, 1000) == pytest.approx(0.002995, abs=1e-5)
    assert randomization.clopper_pearson_upper(1000, 1000) == 1.0
    assert randomization.clopper_pearson_upper(0, 0) is None


def test_power_table_detects_shift_and_reports_mde() -> None:
    rng = np.random.default_rng(0)
    null = rng.normal(0.0, 0.01, size=4000)
    critical = float(np.quantile(null, 1 - 0.05 / 3))
    sigma = float(np.std(null, ddof=1))
    payload = randomization.power_table(null, critical, sigma, (0.5, 1.0, 2.0, 4.0), 0.8)
    assert payload["power"]["0.50"] < payload["power"]["2.00"]
    assert payload["mde"] is not None
    assert payload["power"][f"{payload['mde']:.2f}"] >= 0.8
    assert payload["grid_exhausted"] is False


def test_power_table_marks_grid_exhausted_and_zero_sigma() -> None:
    payload = randomization.power_table(np.array([]), None, None, (1.0,), 0.8)
    assert payload["grid_exhausted"] is True
    assert payload["mde"] is None
    flat = np.zeros(100)
    payload = randomization.power_table(flat, 0.0, 0.0, (1.0, 2.0), 0.8)
    assert payload["grid_exhausted"] is True


def test_analyze_cell_reports_all_required_fields() -> None:
    members = np.array(
        [
            [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.20, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.00, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    dates = np.array([0, 0, 1])
    intent = np.array([0.10, 0.20, 0.0])
    payload = randomization.analyze_cell(
        members,
        dates,
        intent,
        np.array([True, True, False]),
        np.array([False, False, True]),
        block_h=10,
        implementations=300,
        seeds={"calibration": 42, "evaluation": 1042, "power": 2042},
        quantile=1 - 0.05 / 3,
        delta_grid=(0.5, 1.0),
        min_power=0.8,
    )
    for key in (
        "statistic_intent",
        "statistic_filled_only",
        "critical_value",
        "sigma_D0",
        "false_pass",
        "power",
        "randomization_block_h",
        "randomization_date_key",
    ):
        assert key in payload
    assert payload["statistic_intent"] == pytest.approx(0.1)
    assert payload["randomization_block_h"] == 10
    assert payload["randomization_date_key"] == "signal_day"
    assert 0.0 <= payload["false_pass"]["rate"] <= 1.0
    assert payload["false_pass"]["trials"] == 300
