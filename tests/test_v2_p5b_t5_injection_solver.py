# -*- coding: utf-8 -*-
"""P5b T5：崩盘窗注入与 λ/ζ 求解器（§5.4.3 唯一锚定 + §5.5 回撤求解行）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

import p5b_t5_env  # noqa: F401
from src.lazybull.v2.hosts.testbed.referee.drawdown import (
    ANCHOR_HORIZON,
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
    InjectionError,
    LAMBDA_RANGE,
    MAX_REFINE_STEPS,
    SCAN_POINTS,
    STATUS_UNREACHABLE,
    TOLERANCE_PP,
    ZETA_RANGE,
    apply_injection,
    injection_support,
    parameter_range,
    pure_injection_effect,
    solve_injection,
    solver_section,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown import calibration as calibration_module
from src.lazybull.v2.hosts.testbed.referee.drawdown import drawdown_look_plan


def _paths(seed: int = 3, rows: int = 6, days: int = 500) -> np.ndarray:
    """带崩盘段的噪声实现（MDD 约 −25%，保证注入有可用空间）。"""
    rng = np.random.default_rng(seed)
    paths = rng.normal(0.0, 0.004, size=(rows, days))
    paths[:, 100:180] -= 0.004
    return paths


def _mask(days: int = 500) -> np.ndarray:
    mask = np.zeros(days, dtype=bool)
    mask[100:180] = True
    return mask


def test_support_is_window_and_negative_only():
    paths = np.array([[0.01, -0.02, -0.03, 0.04, -0.05, 0.06]])
    mask = np.array([True, True, False, True, True, False])
    support = injection_support(paths, mask)
    assert support.tolist() == [[False, True, False, False, True, False]]


def test_apply_leaves_positives_and_outside_window_untouched():
    paths = np.array([[0.01, -0.02, -0.03, 0.04, -0.05, 0.06]])
    mask = np.array([True, True, False, True, True, False])
    support = injection_support(paths, mask)
    improved = apply_injection(
        paths, support, direction=DIRECTION_IMPROVE, parameter=np.array([0.5])
    )
    expected = paths.copy()
    expected[0, 1] = -0.02 * 0.5
    expected[0, 4] = -0.05 * 0.5
    assert np.allclose(improved, expected)
    # 正收益日与窗口外（索引 4 在窗口内但为负，索引 1/2/5 检验窗口边界）
    worsened = apply_injection(
        paths, support, direction=DIRECTION_WORSEN, parameter=np.array([1.0])
    )
    assert worsened[0, 0] == paths[0, 0]
    assert worsened[0, 2] == paths[0, 2]
    assert worsened[0, 5] == paths[0, 5]


def test_reconstruction_identity_matches_formula():
    paths = np.array([[-0.02, 0.01, -0.03]])
    support = np.ones_like(paths, dtype=bool)
    arm = apply_injection(paths, support, direction=DIRECTION_IMPROVE, parameter=np.array([0.25]))
    rebuilt = np.expm1(arm)
    assert np.allclose(np.log1p(rebuilt), arm)
    assert arm[0, 0] > paths[0, 0]  # 负收益被衰减 ⇒ log 收益变大（更浅）


def test_domain_is_enforced():
    assert parameter_range(DIRECTION_IMPROVE) == LAMBDA_RANGE == (0.0, 1.0)
    assert parameter_range(DIRECTION_WORSEN) == ZETA_RANGE == (0.0, 2.0)
    paths = np.array([[-0.02, -0.03]])
    support = np.ones_like(paths, dtype=bool)
    with pytest.raises(InjectionError):
        apply_injection(paths, support, direction=DIRECTION_IMPROVE, parameter=np.array([1.5]))
    with pytest.raises(InjectionError):
        apply_injection(paths, support, direction=DIRECTION_WORSEN, parameter=np.array([2.5]))
    with pytest.raises(InjectionError):
        apply_injection(paths, support, direction="nope", parameter=np.array([0.1]))


def test_effect_is_monotone_in_parameter():
    paths = _paths()
    mask = _mask()
    base = pure_injection_effect(
        paths, mask, direction=DIRECTION_IMPROVE, parameter=np.zeros(paths.shape[0])
    )
    assert np.allclose(base, 0.0, atol=1e-12)
    levels = np.linspace(0.0, 1.0, 11)
    values = [
        pure_injection_effect(
            paths, mask, direction=DIRECTION_IMPROVE, parameter=np.full(paths.shape[0], level)
        ).mean()
        for level in levels
    ]
    assert all(right >= left - 1e-12 for left, right in zip(values, values[1:]))
    worsen = [
        pure_injection_effect(
            paths, mask, direction=DIRECTION_WORSEN, parameter=np.full(paths.shape[0], level)
        ).mean()
        for level in np.linspace(0.0, 2.0, 11)
    ]
    assert all(right <= left + 1e-12 for left, right in zip(worsen, worsen[1:]))


def test_solver_hits_target_within_tolerance():
    paths = _paths(seed=7, rows=20)
    mask = _mask()
    for target, direction in (
        (1.0, DIRECTION_IMPROVE),
        (5.0, DIRECTION_IMPROVE),
        (-2.0, DIRECTION_WORSEN),
        (-8.0, DIRECTION_WORSEN),
    ):
        solution = solve_injection(paths, mask, direction=direction, target_pp=target)
        assert solution.scan_points == SCAN_POINTS
        assert solution.refine_steps <= MAX_REFINE_STEPS
        assert solution.domain == (0.0, 1.0 if direction == DIRECTION_IMPROVE else 2.0)
        ok = solution.constructible
        assert ok.any()
        achieved = solution.achieved_pp[ok]
        assert np.all(np.abs(achieved - target) <= TOLERANCE_PP + 1e-9)
        assert np.all(solution.parameter[ok] >= 0.0)


def test_solver_returns_minimal_intervention():
    """多解取最小 λ/ζ：返回值必须是可行区间左沿（同一目标下不会取更大的参数）。"""
    paths = _paths(seed=11, rows=5)
    mask = _mask()
    solution = solve_injection(paths, mask, direction=DIRECTION_IMPROVE, target_pp=8.0)
    ok = solution.constructible
    assert ok.any()
    for row in np.flatnonzero(ok):
        parameter = float(solution.parameter[row])
        # 更小的参数不达标，更大的参数效应更强（不会比当前更接近目标的下沿）
        left = pure_injection_effect(
            paths[row : row + 1],
            mask,
            direction=DIRECTION_IMPROVE,
            parameter=np.array([max(parameter - 0.02, 0.0)]),
        )[0]
        assert left < 8.0 - TOLERANCE_PP
        assert abs(float(solution.achieved_pp[row]) - 8.0) <= TOLERANCE_PP + 1e-9


def test_unreachable_target_is_not_constructible_and_not_rewritten():
    paths = _paths(seed=13, rows=5)
    mask = _mask()
    solution = solve_injection(paths, mask, direction=DIRECTION_IMPROVE, target_pp=60.0)
    assert not solution.constructible.any()
    assert set(solution.status) == {STATUS_UNREACHABLE}
    # 实际效应不得被改报成目标值
    assert np.all(np.abs(solution.achieved_pp - 60.0) > TOLERANCE_PP)
    assert solution.constructibility_rate() == 0.0


def test_tolerance_cannot_be_loosened():
    paths = _paths(rows=3)
    with pytest.raises(InjectionError):
        solve_injection(paths, _mask(), direction=DIRECTION_IMPROVE, target_pp=5.0, tolerance=0.5)


def test_target_sign_must_match_direction():
    paths = _paths(rows=3)
    with pytest.raises(InjectionError):
        solve_injection(paths, _mask(), direction=DIRECTION_IMPROVE, target_pp=-5.0)
    with pytest.raises(InjectionError):
        solve_injection(paths, _mask(), direction=DIRECTION_WORSEN, target_pp=5.0)


def test_lambda_and_zeta_are_solved_independently():
    paths = _paths(seed=17, rows=5)
    mask = _mask()
    improve = solve_injection(paths, mask, direction=DIRECTION_IMPROVE, target_pp=5.0)
    worsen = solve_injection(paths, mask, direction=DIRECTION_WORSEN, target_pp=-5.0)
    assert improve.direction != worsen.direction
    assert not np.allclose(improve.parameter, worsen.parameter)
    assert np.all(improve.achieved_pp[improve.constructible] > 0)
    assert np.all(worsen.achieved_pp[worsen.constructible] < 0)


def test_vectorized_result_matches_row_by_row():
    """跨实现向量化求根 == 逐实现单独求根（禁用逐路径循环，但结果必须一致）。"""
    paths = _paths(seed=19, rows=8)
    mask = _mask()
    batched = solve_injection(paths, mask, direction=DIRECTION_IMPROVE, target_pp=4.0)
    for row in range(paths.shape[0]):
        single = solve_injection(
            paths[row : row + 1], mask, direction=DIRECTION_IMPROVE, target_pp=4.0
        )
        assert bool(single.constructible[0]) == bool(batched.constructible[row])
        if single.constructible[0]:
            assert math.isclose(
                float(single.parameter[0]), float(batched.parameter[row]), abs_tol=1e-9
            )
            assert math.isclose(
                float(single.achieved_pp[0]), float(batched.achieved_pp[row]), abs_tol=1e-9
            )


def test_anchor_must_be_within_path():
    with pytest.raises(InjectionError):
        solve_injection(
            _paths(days=100),
            _mask(days=100),
            direction=DIRECTION_IMPROVE,
            target_pp=1.0,
            anchor=ANCHOR_HORIZON,
        )


def test_single_injection_per_path_shared_across_looks(monkeypatch):
    """同一路径所有 look 共用一次注入（结构断言：注入只调用一次）。"""
    paths = _paths(seed=23, rows=4)
    mask = _mask()
    solution = solve_injection(paths, mask, direction=DIRECTION_IMPROVE, target_pp=6.0)
    plan = drawdown_look_plan()
    baseline = p5b_t5_env  # 仅为避免未使用导入告警
    assert baseline is not None
    calls = {"count": 0}
    original = calibration_module.apply_injection

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(calibration_module, "apply_injection", counting)
    from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import baseline_percent_by_look

    base = np.zeros(500)
    base[:100] = -0.002
    d_matrix, ok, parameter, pure, observed = calibration_module._fill_block(
        paths,
        solution,
        mask,
        plan,
        direction=DIRECTION_IMPROVE,
        baseline_pct=baseline_percent_by_look(base, plan.look_days),
    )
    assert calls["count"] == 1  # 一次注入服务全部 look
    assert d_matrix.shape == (4, plan.k)
    assert np.allclose(observed[ok], d_matrix[ok, -1])
    # 不可构造行仍是 nan，但行数不变（分母不剔除）
    assert d_matrix.shape[0] == paths.shape[0]


def test_solver_section_is_frozen_text():
    section = solver_section()
    assert section["scan_points"] == 101
    assert section["max_refine_steps"] == 80
    assert section["tolerance_pp"] == 0.05
    assert "最小" in section["multi_solution_rule"]
    assert "not_constructible" in section["not_constructible_semantics"]
    assert section["truncation_forbidden"] is True
    assert section["report_target_as_achieved_forbidden"] is True
    assert section["independent_parameters"] is True
