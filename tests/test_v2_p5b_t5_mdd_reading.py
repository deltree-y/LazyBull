# -*- coding: utf-8 -*-
"""P5b T5：MDD 与 D_k 读数（golden 手算 + 逐 look horizon + 基准减数）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

import p5b_t5_env  # noqa: F401  （sys.path 注入）
from src.lazybull.v2.hosts.testbed.referee.statistics import LookPlan
from src.lazybull.v2.hosts.testbed.referee.drawdown import (
    ANCHOR_HORIZON,
    LOOK_DAYS,
    DrawdownReadingError,
    baseline_percent_by_look,
    cumulative_log_path,
    drawdown_look_plan,
    drawdown_reading,
    horizon_mdd_log,
    horizon_mdd_percent,
    log_to_percent,
    max_drawdown_log,
    percent_to_log,
    reading_from_cum,
    validate_horizons,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.thresholds import reading_matrix


def test_golden_max_drawdown_hand_computed():
    """小序列已知峰谷：cum=0,0.05,-0.15,-0.13,-0.23,-0.08 ⇒ MDD=-0.28（log）/−24.4216%（pp）。"""
    log_returns = np.array([0.05, -0.20, 0.02, -0.10, 0.15])
    cumulative = cumulative_log_path(log_returns)
    assert np.allclose(cumulative, [0.0, 0.05, -0.15, -0.13, -0.23, -0.08])
    assert math.isclose(float(max_drawdown_log(cumulative)), -0.28, rel_tol=0, abs_tol=1e-12)
    percent = float(log_to_percent(np.asarray([-0.28]))[0])
    assert math.isclose(percent, (math.exp(-0.28) - 1.0) * 100.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(float(horizon_mdd_percent(log_returns, 5)), percent, abs_tol=1e-12)


def test_start_point_is_included_in_drawdown():
    """第一天就下跌时不能漏掉起点：首点 0 必须计入 running max。"""
    path = np.array([-0.10, -0.05, 0.02])
    assert math.isclose(float(max_drawdown_log(cumulative_log_path(path))), -0.15, abs_tol=1e-12)


def test_monotone_decline_mdd_equals_total_log_return():
    path = np.full(50, -0.004)
    assert math.isclose(
        float(max_drawdown_log(cumulative_log_path(path))), -0.2, rel_tol=0, abs_tol=1e-12
    )


def test_percent_and_log_roundtrip():
    values = np.array([-0.2231, -0.1053, 0.0])
    back = percent_to_log(log_to_percent(values))
    assert np.allclose(back, values, atol=1e-12)
    with pytest.raises(DrawdownReadingError):
        percent_to_log(np.array([-100.0]))


def test_horizon_is_independent_per_look():
    """每 look 用自己的 T_k 重算 MDD（不得用 1714 全链档位套用）。"""
    path = np.zeros(int(max(LOOK_DAYS)))
    path[:100] = -0.002  # 前 100 日下跌 20%
    path[150:250] = -0.001  # 240 日窗口内再跌 10%
    short = float(horizon_mdd_percent(path, 120))
    long = float(horizon_mdd_percent(path, 500))
    assert short < 0 and long < short
    assert math.isclose(short, (math.exp(-0.2) - 1.0) * 100.0, abs_tol=1e-9)
    assert math.isclose(long, (math.exp(-0.3) - 1.0) * 100.0, abs_tol=1e-9)


def test_horizon_rejects_out_of_range():
    with pytest.raises(DrawdownReadingError):
        horizon_mdd_log(np.zeros(10), 11)
    with pytest.raises(DrawdownReadingError):
        horizon_mdd_log(np.zeros(10), 0)


def test_non_finite_path_is_rejected():
    with pytest.raises(DrawdownReadingError):
        cumulative_log_path(np.array([0.01, np.nan]))
    with pytest.raises(DrawdownReadingError):
        cumulative_log_path(np.zeros((2, 2, 2)))


def test_reading_sign_convention():
    """D_k = MDD_arm − MDD_base：改善为正、恶化为负。"""
    base = np.zeros(50)
    base[:20] = math.log(0.8) / 20.0  # MDD = -20%
    better = np.zeros(50)
    better[:20] = math.log(0.9) / 20.0  # MDD = -10%
    worse = np.zeros(50)
    worse[:20] = math.log(0.7) / 20.0  # MDD = -30%
    assert math.isclose(
        float(drawdown_reading(better[None, :], base[None, :], 50)[0]), 10.0, abs_tol=1e-9
    )
    assert math.isclose(
        float(drawdown_reading(worse[None, :], base[None, :], 50)[0]), -10.0, abs_tol=1e-9
    )


def test_baseline_percent_by_look_matches_manual():
    base = np.zeros(500)
    base[:100] = -0.002  # 120 日窗口 −20%，500 日窗口仍 −20%
    base[200:300] = -0.001  # 240/360/500 窗口再跌 10%
    values = baseline_percent_by_look(base, LOOK_DAYS)
    assert values.size == 4
    assert math.isclose(values[0], (math.exp(-0.2) - 1.0) * 100.0, abs_tol=1e-9)
    # 240 日窗口只含第二段下跌的前 40 日（−0.04），500 日窗口含完整 −0.1
    assert math.isclose(values[1], (math.exp(-0.24) - 1.0) * 100.0, abs_tol=1e-9)
    assert math.isclose(values[3], (math.exp(-0.3) - 1.0) * 100.0, abs_tol=1e-9)
    with pytest.raises(DrawdownReadingError):
        baseline_percent_by_look(base, (10**6,))


def test_reading_matrix_uses_frozen_baseline():
    """A/A 与效应单元的 D_k 同一减数：arm 与基准相同 ⇒ 全部为 0。"""
    plan = drawdown_look_plan()
    base = np.zeros(500)
    base[:100] = -0.002
    baseline = baseline_percent_by_look(base, plan.look_days)
    matrix = reading_matrix(base[None, :], baseline, plan)
    assert matrix.shape == (1, 4)
    assert np.allclose(matrix, 0.0, atol=1e-12)


def test_reading_from_cum_matches_reading():
    rng = np.random.default_rng(5)
    arm = rng.normal(0, 0.004, size=(4, 500))
    base = rng.normal(0, 0.004, size=(4, 500))
    direct = drawdown_reading(arm, base, 240)
    via_cum = reading_from_cum(cumulative_log_path(arm), cumulative_log_path(base), 240)
    assert np.allclose(direct, via_cum, atol=1e-12)


def test_look_plan_is_frozen_set():
    plan = drawdown_look_plan()
    assert plan.h == 20
    assert tuple(plan.look_days) == (120, 240, 360, 500)
    assert plan.k == 4
    assert plan.max_horizon() == ANCHOR_HORIZON


def test_validate_horizons_rejects_1714_style_plan():
    wrong = LookPlan(
        h=20,
        n_max=503,
        j_max=25,
        j_candidates=(6, 12, 18, 25),
        j_looks=(6, 12, 18, 25),
        look_days=(120, 240, 360, 1714),
    )
    with pytest.raises(DrawdownReadingError):
        validate_horizons(wrong)
    wrong_h = LookPlan(
        h=60, n_max=503, j_max=8, j_candidates=(6, 8), j_looks=(6, 8), look_days=(360, 480)
    )
    with pytest.raises(DrawdownReadingError):
        validate_horizons(wrong_h)
