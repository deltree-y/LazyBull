# -*- coding: utf-8 -*-
"""P5b T6 读数与 H6 判定测试（tests/ 豁免复杂度）。

覆盖：样本不足 fail-closed、退化方差不得伪造 Z、golden 手算 Z、两个方向的事件与
互斥、``not_constructible`` ⇒ ``not_executed``、方向不符 ⇒ ``fail``。
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
    build_arm_effect,
    income_daily_shift,
)
from src.lazybull.v2.hosts.testbed.rehearsal.readings import (  # noqa: E402
    LOOK_DEGENERATE,
    LOOK_INSUFFICIENT,
    LOOK_NON_POSITIVE_VARIANCE,
    drawdown_look_reading,
    h6_drawdown_item,
    h6_income_item,
    income_look_reading,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (  # noqa: E402
    DIRECTION_IMPROVE,
    DIRECTION_WORSEN,
)
from src.lazybull.v2.hosts.testbed.referee.statistics import DELTA_LOG_DAILY  # noqa: E402

WINDOWS = [
    {"window_start_ordinal": 100, "window_stop_ordinal": 200},
    {"window_start_ordinal": 300, "window_stop_ordinal": 400},
]


def _thresholds(**overrides) -> dict:
    base = {
        "look_days": [120, 240, 360, 500],
        "look_blocks": [6, 12, 18, 25],
        "h": 20,
        "c_k": [4.72, 3.46, 3.05, 2.93],
        "c_w": [-6.0, -6.0, -6.0, -6.0],
        "c_i": [5.0, 5.0, 5.0, 5.0],
        "delta_log_daily": float(DELTA_LOG_DAILY),
        "anchor": 500,
    }
    base.update(overrides)
    return base


def _income_effect():
    return build_arm_effect(
        "income_minus_5d",
        {"kind": "income_log_shift", "delta_multiple": 5.0},
        return_days=500,
        frozen_list=WINDOWS,
    )


def _dd_effect(direction: str, parameter: float):
    return build_arm_effect(
        "dd_improve_12pp" if direction == DIRECTION_IMPROVE else "dd_worsen_12pp",
        {
            "kind": "crash_window_transform",
            "direction": direction,
            "target_pp": 12.0 if direction == DIRECTION_IMPROVE else -12.0,
            "anchor": 500,
            "parameter": parameter,
        },
        return_days=500,
        frozen_list=WINDOWS,
    )


def test_income_reading_marks_insufficient_blocks():
    returns = np.full(78, -1e-4)
    reading = income_look_reading(returns, _thresholds())
    assert reading["executable_looks"] == []
    assert all(row["status"] == LOOK_INSUFFICIENT for row in reading["rows"])
    assert all(row["z_point_estimate"] is None for row in reading["rows"])
    assert reading["identifiable_looks"] == []


def test_income_reading_golden_hand_computed_z():
    returns = np.array([0.01, 0.02, -0.01, -0.02])
    thresholds = _thresholds(look_days=[4], look_blocks=[2], h=2, c_k=[1.0])
    reading = income_look_reading(returns, thresholds)
    row = reading["rows"][0]
    # blocks=[0.015,-0.015] ⇒ bar=0；V_psd=(0.015^2+0.015^2)/(2*1)=0.000225；
    # z = (0 + delta)/sqrt(V/J) = delta/sqrt(0.0001125)
    expected = DELTA_LOG_DAILY / math.sqrt(0.000225 / 2)
    assert row["status"] == "ok"
    assert math.isclose(row["z_point_estimate"], expected, rel_tol=1e-12)
    assert row["veto"] is False


def test_income_reading_detects_zero_and_degenerate_variance():
    constant = np.full(120, -5.0 * DELTA_LOG_DAILY)
    thresholds = _thresholds(look_days=[120], look_blocks=[6], h=20, c_k=[2.0])
    zero_variance = income_look_reading(constant, thresholds)["rows"][0]
    assert zero_variance["status"] == LOOK_NON_POSITIVE_VARIANCE
    assert zero_variance["z_point_estimate"] is None
    rng = np.random.default_rng(3)
    jitter = constant + rng.normal(0.0, 1e-16, size=120)
    degenerate = income_look_reading(jitter, thresholds)["rows"][0]
    assert degenerate["status"] == LOOK_DEGENERATE
    assert degenerate["identifiable"] is False
    assert degenerate["z_point_estimate"] is None


def test_income_item_passes_on_frozen_criterion_despite_degenerate_z():
    """冻结判据 = 方向 + 幅度 + 链路；Z 不可识别只披露，不改变判定。"""
    shift = income_daily_shift()
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 0.004, size=120)
    arm = base + shift
    thresholds = _thresholds(look_days=[120], look_blocks=[6], h=20, c_k=[2.0])
    reading = income_look_reading(arm - base, thresholds)
    item = h6_income_item(_income_effect(), reading, shift)
    assert item["status"] == "pass"
    assert item["direction_ok"] is True and item["magnitude_ok"] is True
    assert item["z_identifiable"] is False
    assert item["z_point_estimate"] is None


def test_income_item_pass_when_noise_small_and_magnitude_matches():
    shift = income_daily_shift()
    rng = np.random.default_rng(9)
    base = rng.normal(0.0, 0.004, size=120)
    noise = rng.normal(0.0, 1e-9, size=120)
    thresholds = _thresholds(look_days=[120], look_blocks=[6], h=20, c_k=[2.0])
    reading = income_look_reading((base + noise + shift) - base, thresholds)
    item = h6_income_item(_income_effect(), reading, shift)
    assert item["status"] == "pass"
    assert item["direction_ok"] is True and item["magnitude_ok"] is True


def test_income_item_fail_when_magnitude_drifts():
    """幅度偏离预登记 ⇒ `fail`（不是「未执行」：这是已测到的失败证据）。"""
    shift = income_daily_shift()
    rng = np.random.default_rng(10)
    base = rng.normal(0.0, 0.004, size=120)
    noise = rng.normal(0.0, 0.001, size=120)
    thresholds = _thresholds(look_days=[120], look_blocks=[6], h=20, c_k=[2.0])
    reading = income_look_reading((base + noise + shift) - base, thresholds)
    item = h6_income_item(_income_effect(), reading, shift)
    assert item["status"] == "fail"
    assert item["direction_ok"] is True and item["magnitude_ok"] is False


def test_income_item_fails_when_direction_wrong():
    shift = income_daily_shift()
    rng = np.random.default_rng(13)
    base = rng.normal(0.0, 0.004, size=120)
    thresholds = _thresholds(look_days=[120], look_blocks=[6], h=20, c_k=[2.0])
    reading = income_look_reading((base - shift + rng.normal(0, 1e-5, 120)) - base, thresholds)
    item = h6_income_item(_income_effect(), reading, shift)
    assert item["status"] == "fail"
    assert item["direction_ok"] is False


def _crash_path(factor: float) -> np.ndarray:
    """收益路径：两段亏损全部落在冻结窗内（便于构造可判定的 D_k）。"""
    base = np.zeros(500)
    base[100:200] = -0.002 * factor
    base[300:400] = -0.001 * factor
    return base


def test_drawdown_events_are_directional_and_mutually_exclusive():
    base = _crash_path(1.0)
    improve = _crash_path(0.5)
    worsen = _crash_path(1.5)
    thresholds = _thresholds()
    improve_reading = drawdown_look_reading(improve, base, thresholds)
    worsen_reading = drawdown_look_reading(worsen, base, thresholds)
    anchor_improve = improve_reading["rows"][-1]
    anchor_worsen = worsen_reading["rows"][-1]
    assert anchor_improve["d_pp"] > 0 and anchor_improve["improve_event"] is True
    assert anchor_improve["worsen_event"] is False
    assert anchor_worsen["d_pp"] < 0 and anchor_worsen["worsen_event"] is True
    assert anchor_worsen["improve_event"] is False


def test_drawdown_reading_marks_short_path_as_insufficient():
    base = np.zeros(100)
    reading = drawdown_look_reading(base, base, _thresholds())
    assert all(row["status"] == LOOK_INSUFFICIENT for row in reading["rows"])
    assert reading["executable_looks"] == []


def test_h6_drawdown_item_not_executed_when_not_constructible():
    effect = build_arm_effect(
        "dd_improve_12pp",
        {
            "kind": "crash_window_transform",
            "direction": DIRECTION_IMPROVE,
            "target_pp": 12.0,
            "anchor": 500,
            "parameter": 0.4,
        },
        return_days=78,
        frozen_list=WINDOWS,
    )
    item = h6_drawdown_item(effect, None, direction=DIRECTION_IMPROVE, pure_effect_pp=None)
    assert item["status"] == "not_executed"
    assert item["constructibility"] == "not_constructible"
    assert item["constructibility_reason"] == "anchor_horizon_exceeds_rehearsal_window"
    assert item["d_anchor_pp"] is None and item["event"] == "not_executed"


def test_h6_drawdown_item_pass_and_fail():
    base = _crash_path(1.0)
    thresholds = _thresholds()
    improve_reader = drawdown_look_reading(_crash_path(0.5), base, thresholds)
    item = h6_drawdown_item(
        _dd_effect(DIRECTION_IMPROVE, 0.5), improve_reader, direction=DIRECTION_IMPROVE
    )
    assert item["status"] == "pass" and item["event"].endswith("improvement")
    flat_reader = drawdown_look_reading(base, base, thresholds)
    not_triggered = h6_drawdown_item(
        _dd_effect(DIRECTION_IMPROVE, 0.5), flat_reader, direction=DIRECTION_IMPROVE
    )
    assert not_triggered["status"] == "fail"
    assert not_triggered["event"] == "no_event"


def test_drawdown_reading_rejects_mismatched_lengths():
    from src.lazybull.v2.hosts.testbed.rehearsal.readings import ReadingsError

    with pytest.raises(ReadingsError):
        drawdown_look_reading(np.zeros(10), np.zeros(11), _thresholds())
