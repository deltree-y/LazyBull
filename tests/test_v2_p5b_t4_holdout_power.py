# -*- coding: utf-8 -*-
"""P5b T4 holdout 逐单元 Wilson 门与功效/MDE（设计 v4 §5.3）。

覆盖：Wilson 区间 golden、单元 = 场景×零假设（8 个，无跨场景聚合字段）、
任一单元不过即裁判不可用、无效 look 不触发但计入无效率门、
MDE 仅在 A/A 过门后登记且取最小达标档、5δ 未达 80% 的如实状态、禁止聚合 MDE。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t4_env as env  # noqa: E402

from src.lazybull.v2.hosts.testbed.referee import (  # noqa: E402
    EFFECT_GRID_DELTA,
    MDE_NOT_REACHED,
    MDE_NOT_REGISTERED,
    TARGET_POWER,
    HoldoutError,
    PowerError,
    ThresholdConfig,
    build_scene_registry,
    evaluate_holdout,
    evaluate_power,
    holdout_unit_count,
    make_look_plan,
    register_mde,
    wilson_interval,
)

PLAN = make_look_plan(20)
CONFIG = ThresholdConfig()
POSITIVE = [4.0, 3.0, 2.5, 2.4]
COUNT = 200


def _registry(*, role: str = "holdout", sd_daily: float = 0.0037, pool=None):
    values = np.linspace(-0.02, 0.02, 260) if pool is None else np.asarray(pool, dtype=float)
    return build_scene_registry(values, params=env.dgp_params(sd_daily=sd_daily), pool_role=role)


def _holdout(thresholds=POSITIVE, **kwargs):
    return evaluate_holdout(
        plan=PLAN,
        registry=kwargs.pop("registry", _registry()),
        seed=kwargs.pop("seed", 5),
        count=kwargs.pop("count", COUNT),
        thresholds=thresholds,
        config=kwargs.pop("config", CONFIG),
        prefix="p5b_t4_test",
        **kwargs,
    )


def _rate_map(payload):
    return {key: item["rate"] for key, item in payload["units"].items()}


def test_wilson_zero_success_golden():
    low, high = wilson_interval(0, 10000)
    assert low == 0.0
    assert high == pytest.approx(0.0003841, rel=2e-3)


def test_wilson_golden_5_of_100():
    # 手算：den=1+3.8416/100；center=(0.05+0.019208)/den；half=1.96*sqrt(0.05*0.95/100+3.8416/4e4)/den
    low, high = wilson_interval(5, 100)
    assert low == pytest.approx(0.02155, abs=5e-5)
    assert high == pytest.approx(0.11175, abs=5e-5)


def test_wilson_golden_500_of_10000():
    low, high = wilson_interval(500, 10000)
    assert low == pytest.approx(0.0459, abs=1e-3)
    assert high == pytest.approx(0.0545, abs=1e-3)
    assert low < 0.05 < high


def test_wilson_rejects_invalid_inputs():
    with pytest.raises(HoldoutError):
        wilson_interval(0, 0)
    with pytest.raises(HoldoutError):
        wilson_interval(5, 4)


def test_holdout_unit_count_is_scene_times_null():
    assert holdout_unit_count(4) == 8


def test_holdout_units_are_per_scene_null_without_aggregation():
    payload = _holdout()
    assert payload["unit_count"] == 8
    assert set(payload["units"]) == {
        f"{scene}|{null}"
        for scene in ("ar1_06", "ma20", "real_seed_block", "regime_hetero")
        for null in ("mu=0", "mu=-delta")
    }
    assert payload["reporting_rule"] == "per_unit_only_no_pooling_no_weighting"
    forbidden = ("pooled", "average", "overall", "weighted")
    for key in _all_keys(payload):
        assert not any(word in key.lower() for word in forbidden)


def _all_keys(node) -> list:
    keys = []
    if isinstance(node, dict):
        for key, value in node.items():
            keys.append(str(key))
            keys.extend(_all_keys(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            keys.extend(_all_keys(item))
    return keys


def test_holdout_rejects_threshold_length_mismatch():
    with pytest.raises(HoldoutError):
        _holdout(thresholds=[3.0, 2.5])


def test_holdout_rejects_non_finite_thresholds():
    with pytest.raises(HoldoutError):
        _holdout(thresholds=[4.0, float("nan"), 2.5, 2.4])


def test_holdout_rejects_loosened_wilson_gate():
    with pytest.raises(HoldoutError):
        _holdout(wilson_upper_max=0.07)


def test_holdout_requires_holdout_registry():
    with pytest.raises(HoldoutError):
        _holdout(registry=_registry(role="train"))


def test_holdout_uses_all_realizations_in_denominator():
    payload = _holdout()
    for item in payload["units"].values():
        assert item["total"] == COUNT
        assert 0 <= item["successes"] <= COUNT


def test_holdout_unit_failure_makes_referee_unusable():
    payload = _holdout(thresholds=[0.0, 0.0, 0.0, 0.0])
    assert payload["referee_usable"] is False
    assert len(payload["failed_units"]) == 8
    for key in payload["failed_units"]:
        assert "wilson_upper_exceeds_max" in payload["units"][key]["failure_reasons"]


def test_holdout_invalid_look_does_not_trigger_but_blocks_config():
    payload = _holdout(thresholds=POSITIVE, registry=_registry(sd_daily=0.0, pool=np.zeros(260)))
    for item in payload["units"].values():
        assert item["successes"] == 0  # 无效 look 不触发否决
        assert max(item["invalid_rate_by_look"]) == 1.0
        assert "invalid_rate_exceeds_max" in item["failure_reasons"]
    assert payload["referee_usable"] is False


def test_holdout_payload_is_deterministic():
    first = _holdout()
    second = _holdout()
    assert _rate_map(first) == _rate_map(second)
    assert first["failed_units"] == second["failed_units"]


def test_mde_registered_only_when_aa_passes():
    rates = {"5.0": {"ma20": 0.95}}
    result = register_mde(False, rates, [5.0], ["ma20"])
    assert result["registered"] is False
    assert result["by_scene"]["ma20"] is None
    assert result["status_by_scene"]["ma20"] == MDE_NOT_REGISTERED


def test_mde_picks_smallest_level_above_target():
    rates = {
        "0.5": {"ma20": 0.10},
        "1.0": {"ma20": 0.55},
        "1.5": {"ma20": 0.79},
        "3.0": {"ma20": 0.80},
        "5.0": {"ma20": 0.99},
    }
    result = register_mde(True, rates, [0.5, 1.0, 1.5, 3.0, 5.0], ["ma20"])
    assert result["registered"] is True
    assert result["by_scene"]["ma20"] == 3.0
    assert result["status_by_scene"]["ma20"] == "registered"


def test_mde_reports_not_reached_honestly():
    rates = {"5.0": {"ar1_06": 0.42}}
    result = register_mde(True, rates, [5.0], ["ar1_06"])
    assert result["by_scene"]["ar1_06"] is None
    assert result["status_by_scene"]["ar1_06"] == MDE_NOT_REACHED


def test_power_grid_is_frozen():
    assert list(EFFECT_GRID_DELTA) == [0.5, 1.0, 1.5, 3.0, 5.0]
    assert TARGET_POWER == pytest.approx(0.80)


def test_power_reports_per_level_scene_and_forbids_pooling():
    payload = evaluate_power(
        plan=PLAN,
        registry=_registry(),
        seed=5,
        count=60,
        thresholds=POSITIVE,
        applicable=True,
        prefix="p5b_t4_test",
    )
    assert set(payload["by_level_scene"]) == {"0.5", "1.0", "1.5", "3.0", "5.0"}
    for level in payload["by_level_scene"].values():
        assert set(level) == {"ar1_06", "ma20", "real_seed_block", "regime_hetero"}
        for item in level.values():
            assert 0.0 <= item["rate"] <= 1.0
            assert item["total"] == 60
    assert payload["mde"]["pooled_average_forbidden"] is True
    assert payload["mde"]["per_scene_only"] is True


def test_power_monotone_in_effect_level():
    payload = evaluate_power(
        plan=PLAN,
        registry=_registry(),
        seed=5,
        count=200,
        thresholds=POSITIVE,
        applicable=True,
        prefix="p5b_t4_test",
    )
    for scene in payload["by_level_scene"]["0.5"]:
        small = payload["by_level_scene"]["0.5"][scene]["rate"]
        large = payload["by_level_scene"]["5.0"][scene]["rate"]
        assert large >= small


def test_power_mde_not_registered_when_aa_failed():
    payload = evaluate_power(
        plan=PLAN,
        registry=_registry(),
        seed=5,
        count=60,
        thresholds=POSITIVE,
        applicable=False,
        prefix="p5b_t4_test",
    )
    assert payload["mde"]["registered"] is False
    assert set(payload["mde"]["status_by_scene"].values()) == {MDE_NOT_REGISTERED}


def test_power_rejects_train_registry():
    with pytest.raises(PowerError):
        evaluate_power(
            plan=PLAN,
            registry=_registry(role="train"),
            seed=5,
            count=20,
            thresholds=POSITIVE,
        )


def test_power_rejects_negative_effect_levels():
    with pytest.raises(PowerError):
        evaluate_power(
            plan=PLAN,
            registry=_registry(),
            seed=5,
            count=20,
            thresholds=POSITIVE,
            grid=[-1.0, 1.0],
        )


# --------------------------------------------------------------------------
# 首轮评审接纳项（reviewer1 R4/R9、reviewer2 R2-T4-05/06/08/09）
# --------------------------------------------------------------------------


def test_holdout_rejects_non_frozen_wilson_z():
    """z 也是门的一部分：z=0 会把区间压成 [p,p] 使门失效。"""
    with pytest.raises(HoldoutError):
        _holdout(z=0.0)
    with pytest.raises(HoldoutError):
        _holdout(z=1.0)


def test_holdout_rejects_tightened_gate_as_other_caliber():
    """冻结口径唯一：0.05 这类"收紧"会让读数无法与 r3 对齐，同样拒绝。"""
    with pytest.raises(HoldoutError):
        _holdout(wilson_upper_max=0.05)


def test_power_rejects_non_frozen_target_and_z():
    base = dict(
        plan=PLAN,
        registry=_registry(),
        seed=5,
        count=20,
        thresholds=POSITIVE,
        applicable=True,
    )
    with pytest.raises(PowerError):
        evaluate_power(**base, target=0.60)
    with pytest.raises(PowerError):
        evaluate_power(**base, z=0.0)


def test_power_rejects_non_identity_scale_hook():
    """scale_by_scene 只能恒等：任何缩放都会改注册效应档。"""
    base = dict(
        plan=PLAN,
        registry=_registry(),
        seed=5,
        count=20,
        thresholds=POSITIVE,
        applicable=True,
    )
    identity = {scene: 1.0 for scene in ("ma20", "ar1_06", "regime_hetero", "real_seed_block")}
    assert evaluate_power(**base, scale_by_scene=identity)["mde"]["registered"] is True
    with pytest.raises(PowerError):
        evaluate_power(**base, scale_by_scene={**identity, "ma20": 0.5})
    with pytest.raises(PowerError):
        evaluate_power(**base, scale_by_scene={"ma20": 1.0})  # 缺场景同样拒绝


def test_power_degraded_units_block_mde_registration():
    """无效 look 超过门 ⇒ 不得登记 MDE（与 holdout 同一无效率门）。"""
    payload = evaluate_power(
        plan=PLAN,
        registry=_registry(sd_daily=0.0, pool=np.zeros(260)),  # 退化路径：V=0 ⇒ look 无效
        seed=5,
        count=20,
        thresholds=POSITIVE,
        applicable=True,
        prefix="p5b_t4_test",
    )
    assert payload["mde"]["registered"] is False
    assert payload["mde"]["degraded_units"]
    assert "invalid_rate" in payload["mde"]["degraded_reason"]
    assert set(payload["mde"]["status_by_scene"].values()) == {MDE_NOT_REGISTERED}


def test_power_rejects_free_invalid_rate_knob():
    """无效率门也是冻结口径：放大/缩小都拒绝（否则可静默废掉 10% 门）。"""
    base = dict(
        plan=PLAN,
        registry=_registry(),
        seed=5,
        count=20,
        thresholds=POSITIVE,
        applicable=True,
    )
    with pytest.raises(PowerError):
        evaluate_power(**base, invalid_rate_per_look_max=1.0)
    with pytest.raises(PowerError):
        evaluate_power(**base, invalid_rate_per_look_max=0.0)


def test_power_requires_explicit_applicable():
    """A/A 结果必填：无默认值，避免"忘记传参即登记 MDE"。"""
    with pytest.raises(PowerError):
        evaluate_power(
            plan=PLAN,
            registry=_registry(),
            seed=5,
            count=20,
            thresholds=POSITIVE,
        )


def test_mde_marks_grid_upper_bound():
    """命中网格上界只能宣称 MDE <= 上界档，不得当作精确读数。"""
    rates = {"3.0": {"ma20": 0.51}, "5.0": {"ma20": 0.99}}
    result = register_mde(True, rates, [3.0, 5.0], ["ma20"])
    assert result["by_scene"]["ma20"] == 5.0
    assert result["at_grid_upper_bound"]["ma20"] is True
    assert result["grid_upper_level"] == 5.0
    exact = register_mde(True, {"3.0": {"ma20": 0.85}}, [1.0, 3.0, 5.0], ["ma20"])
    assert exact["by_scene"]["ma20"] == 3.0
    assert exact["at_grid_upper_bound"]["ma20"] is False


def test_holdout_and_power_record_chunk():
    holdout = _holdout(chunk=64)
    assert holdout["counts"]["chunk"] == 64
    single = _holdout(chunk=None)
    assert single["counts"]["chunk"] is None  # 保真：单批不得伪记为 2000
    assert single["counts"]["effective_chunk"] == 2000
    power = evaluate_power(
        plan=PLAN,
        registry=_registry(),
        seed=5,
        count=20,
        thresholds=POSITIVE,
        applicable=True,
        chunk=64,
    )
    assert power["counts"]["chunk"] == 64


def test_holdout_uses_all_realizations_in_denominator_strict():
    """分母恒等于配置实现数（无效样本不得被剔除）。"""
    payload = _holdout()
    for item in payload["units"].values():
        assert item["total"] == COUNT
        assert 0 <= item["successes"] <= item["total"]
        assert item["rate"] == pytest.approx(item["successes"] / COUNT)


def test_holdout_single_failing_unit_marks_only_that_unit(monkeypatch):
    """聚合逻辑分支：只有一个单元触发时，只该单元进 failed_units。"""
    from src.lazybull.v2.hosts.testbed.referee import holdout as holdout_module

    calls = {"n": 0}
    original = holdout_module.veto_flags

    def fake_veto(values, limits):
        calls["n"] += 1
        flags = original(values, limits)
        if calls["n"] == 2:  # 第二个单元（ar1_06|mu=-delta）全部触发
            return np.ones_like(flags, dtype=bool)
        return np.zeros_like(flags, dtype=bool)

    monkeypatch.setattr(holdout_module, "veto_flags", fake_veto)
    payload = _holdout()
    assert payload["failed_units"] == ["ar1_06|mu=-delta"]
    assert payload["referee_usable"] is False
    assert payload["units"]["ar1_06|mu=-delta"]["failure_reasons"] == ["wilson_upper_exceeds_max"]
    assert payload["units"]["ma20|mu=0"]["pass"] is True
