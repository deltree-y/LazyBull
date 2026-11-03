# -*- coding: utf-8 -*-
"""P5b T6 效应注入测试（tests/ 豁免复杂度）。

覆盖：δ 倍数与冻结口径、纯函数性、只改窗口内负收益、路径级与逐日级逐值等价、
λ/ζ 越界拒绝、非有限 fail-closed、窗口越界不得截断、``not_constructible`` 分支。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.lazybull.v2.hosts.testbed.rehearsal.injection import (  # noqa: E402
    EFFECT_NONE,
    InjectionError,
    build_arm_effect,
    income_daily_shift,
    log_returns_from_nav,
    nav_from_log_returns,
    transform_log_returns,
    window_mask_from_frozen,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (  # noqa: E402
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
)
from src.lazybull.v2.hosts.testbed.referee.statistics import DELTA_LOG_DAILY  # noqa: E402

WINDOWS = [
    {"window_start_ordinal": 2, "window_stop_ordinal": 4},
    {"window_start_ordinal": 6, "window_stop_ordinal": 7},
]


def _dd_spec(direction: str, parameter: float | None, target: float = 12.0) -> dict:
    return {
        "kind": "crash_window_transform",
        "direction": direction,
        "target_pp": target if direction == DIRECTION_IMPROVE else -target,
        "anchor": 500,
        "parameter": parameter,
        "parameter_name": "lambda" if direction == DIRECTION_IMPROVE else "zeta",
    }


def _effect(direction: str, parameter: float, *, return_days: int = 500):
    """构造可构造的 DD 效应（锚 500 ⇒ 声明 500 个收益样本；短数组只用于变换本身）。"""
    return build_arm_effect(
        "dd_improve_12pp" if direction == DIRECTION_IMPROVE else "dd_worsen_12pp",
        _dd_spec(direction, parameter),
        return_days=return_days,
        frozen_list=WINDOWS,
    )


def test_income_shift_matches_frozen_delta_and_multiple():
    shift = income_daily_shift()
    assert math.isclose(shift, -5.0 * DELTA_LOG_DAILY, rel_tol=0.0, abs_tol=0.0)
    assert math.isclose(shift, -5.0 * math.log(1.05) / 252, rel_tol=1e-9)
    with pytest.raises(InjectionError):
        income_daily_shift(3.0)


def test_nav_and_log_returns_are_inverse():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.004, size=50)
    nav = nav_from_log_returns(1.0, returns)
    assert nav[0] == 1.0
    recovered = log_returns_from_nav(nav)
    assert np.allclose(recovered, returns, rtol=1e-12, atol=1e-15)


def test_window_mask_is_index_aligned_and_rejects_truncation():
    mask, stops = window_mask_from_frozen(WINDOWS, 8)
    assert stops == [4, 7]
    assert mask.tolist() == [False, False, True, True, False, False, True, False]
    with pytest.raises(InjectionError):
        window_mask_from_frozen(WINDOWS, 6)


def test_only_window_negatives_are_transformed():
    returns = np.array([0.01, -0.01, -0.02, 0.02, -0.03, -0.01, -0.04, 0.03])
    effect = _effect(DIRECTION_IMPROVE, 0.5)
    out = transform_log_returns(returns, effect, frozen_list=WINDOWS)
    # 窗口 [2,4) 与 [6,7)：仅负收益被衰减，位置 1/4 不在窗口内保持原值
    assert out[1] == returns[1]
    assert out[4] == returns[4]
    assert math.isclose(out[2], -0.02 * 0.5, rel_tol=1e-12)
    assert out[3] == returns[3]
    assert math.isclose(out[6], -0.04 * 0.5, rel_tol=1e-12)
    assert out[7] == returns[7]


def test_worsen_amplifies_only_window_negatives():
    returns = np.array([0.01, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, 0.03])
    effect = _effect(DIRECTION_WORSEN, 2.0)
    out = transform_log_returns(returns, effect, frozen_list=WINDOWS)
    assert math.isclose(out[2], -0.02 * 3.0, rel_tol=1e-12)
    assert math.isclose(out[6], -0.06 * 3.0, rel_tol=1e-12)
    assert out[1] == returns[1] and out[4] == returns[4]


def test_path_and_per_day_transforms_are_identical():
    from src.lazybull.v2.hosts.testbed.rehearsal.injection import apply_single_return

    rng = np.random.default_rng(11)
    returns = rng.normal(-0.001, 0.005, size=8)
    effect = _effect(DIRECTION_IMPROVE, 0.42)
    mask, _ = window_mask_from_frozen(WINDOWS, 8)
    vector = transform_log_returns(returns, effect, frozen_list=WINDOWS)
    manual = np.array(
        [
            apply_single_return(value, index, effect, mask=mask)
            for index, value in enumerate(returns)
        ]
    )
    assert np.array_equal(vector, manual)


def test_injection_is_pure(monkeypatch):
    returns = np.array([-0.01, -0.02, 0.01, -0.03, 0.0, -0.04, -0.05, 0.02])
    effect = _effect(DIRECTION_IMPROVE, 0.3)
    first = transform_log_returns(returns, effect, frozen_list=WINDOWS)
    second = transform_log_returns(returns, effect, frozen_list=WINDOWS)
    assert first.tobytes() == second.tobytes()
    assert np.array_equal(returns, np.array([-0.01, -0.02, 0.01, -0.03, 0.0, -0.04, -0.05, 0.02]))


def test_parameter_out_of_range_is_rejected():
    with pytest.raises(InjectionError):
        transform_log_returns(
            np.array([-0.01] * 8), _effect(DIRECTION_IMPROVE, 1.5), frozen_list=WINDOWS
        )
    with pytest.raises(InjectionError):
        transform_log_returns(
            np.array([-0.01] * 8), _effect(DIRECTION_WORSEN, 2.5), frozen_list=WINDOWS
        )


def test_non_finite_inputs_fail_closed():
    effect = _effect(DIRECTION_IMPROVE, 0.3)
    with pytest.raises(InjectionError):
        transform_log_returns(np.array([np.nan] * 8), effect, frozen_list=WINDOWS)
    with pytest.raises(InjectionError):
        nav_from_log_returns(0.0, np.array([0.0]))
    with pytest.raises(InjectionError):
        nav_from_log_returns(1.0, np.array([np.inf]))


def test_not_constructible_when_anchor_or_window_exceeds_path():
    effect = build_arm_effect(
        "dd_improve_12pp", _dd_spec(DIRECTION_IMPROVE, 0.4), return_days=78, frozen_list=WINDOWS
    )
    assert effect.status == "not_constructible"
    assert effect.reason == "anchor_horizon_exceeds_rehearsal_window"
    assert effect.required_return_days == 500
    # 不可构造臂不施加注入（未激活）
    assert effect.active is False


def test_not_constructible_when_parameter_missing():
    effect = build_arm_effect(
        "dd_worsen_12pp", _dd_spec(DIRECTION_WORSEN, None), return_days=600, frozen_list=WINDOWS
    )
    assert effect.status == "not_constructible"
    assert effect.reason == "anchor_parameter_not_constructible"


def test_no_injection_arm_is_transparent():
    effect = build_arm_effect("base", {"kind": EFFECT_NONE}, return_days=8, frozen_list=WINDOWS)
    returns = np.array([-0.01, 0.02] * 4)
    assert np.array_equal(transform_log_returns(returns, effect, frozen_list=WINDOWS), returns)


def test_host_nav_mapping_equals_path_level_transform():
    """宿主逐日映射 ≡ 路径级 ``transform_log_returns`` + 重建（纯函数的双向证据）。"""
    from src.lazybull.v2.hosts.testbed.rehearsal.hosts import injected_nav_series

    rng = np.random.default_rng(21)
    returns = rng.normal(-0.0005, 0.006, size=12)
    base_nav = nav_from_log_returns(1.0, returns)
    effect = _effect(DIRECTION_IMPROVE, 0.35)
    mask, _ = window_mask_from_frozen(WINDOWS, returns.size)
    mapped = np.asarray(injected_nav_series(list(base_nav), effect, mask=mask), dtype=float)
    expected = nav_from_log_returns(
        base_nav[0],
        transform_log_returns(returns, effect, frozen_list=WINDOWS),
    )
    assert np.allclose(mapped, expected, rtol=1e-12, atol=0.0)


def test_income_arm_applies_constant_shift_every_day():
    effect = build_arm_effect(
        "income_minus_5d",
        {"kind": "income_log_shift", "delta_multiple": 5.0},
        return_days=8,
        frozen_list=WINDOWS,
    )
    returns = np.array([0.001, -0.002, 0.003, -0.004, 0.0, -0.005, 0.006, 0.007])
    out = transform_log_returns(returns, effect, frozen_list=WINDOWS)
    assert np.allclose(out - returns, income_daily_shift(), rtol=0.0, atol=1e-18)
