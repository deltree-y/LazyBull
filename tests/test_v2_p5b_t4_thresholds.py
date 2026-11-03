# -*- coding: utf-8 -*-
"""P5b T4 临界值训练（§5.2 五步规则）与「holdout 不得写回 c_k」的结构性校验。

覆盖：同 seed 同结果、逐单元中间量可审计、``c_k = max(单元 c)``、无效率/有效样本不足的
硬失败（不得排除后继续、不得用极值替代）、训练只接受训练池、临界值产物只能写一次、
holdout 只读临界值（字节不变）。
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

from src.lazybull.v2.hosts.testbed.paths import TestbedError  # noqa: E402
from src.lazybull.v2.hosts.testbed.referee import (  # noqa: E402
    ARTIFACT_THRESHOLDS,
    ThresholdConfig,
    ThresholdError,
    artifact_path,
    build_scene_registry,
    make_look_plan,
    train_income_thresholds,
)
from src.lazybull.v2.hosts.testbed.referee.training import (  # noqa: E402
    MIN_VALID_QUANTILE_SAMPLE,
    TRAINING_RULE_STEPS,
)

PLAN = make_look_plan(20)
PARAMS = env.dgp_params()
COUNT = 160


def _registry(*, role: str = "train", sd_daily: float = 0.0037, pool=None):
    values = np.linspace(-0.02, 0.02, 260) if pool is None else np.asarray(pool, dtype=float)
    return build_scene_registry(values, params=env.dgp_params(sd_daily=sd_daily), pool_role=role)


def _train(**kwargs):
    config = kwargs.pop("config", ThresholdConfig())
    return train_income_thresholds(
        plan=PLAN,
        registry=kwargs.pop("registry", _registry()),
        seed=kwargs.pop("seed", 7),
        counts=kwargs.pop("counts", {"train_per_unit": COUNT}),
        config=config,
        prefix=kwargs.pop("prefix", "p5b_t4_test"),
        **kwargs,
    )


def test_training_is_deterministic_same_seed():
    first = _train()
    second = _train()
    assert list(first.thresholds) == list(second.thresholds)
    assert first.determining_units == second.determining_units
    assert first.realization_ids == second.realization_ids


def test_training_passes_with_enough_paths():
    result = _train()
    assert result.passed
    assert len(result.thresholds) == PLAN.k
    assert all(np.isfinite(result.thresholds))


def test_threshold_equals_max_over_units():
    result = _train()
    for index in range(PLAN.k):
        per_unit = [
            unit["looks"][index]["c"] for unit in result.units if unit["looks"][index]["eligible"]
        ]
        assert per_unit
        assert result.thresholds[index] == pytest.approx(max(per_unit), rel=0, abs=1e-15)


def test_unit_intermediates_are_auditable():
    result = _train()
    assert len(result.units) == 8  # 4 场景 × 2 零假设
    for unit in result.units:
        assert unit["unit"] == f"{unit['scene']}|{unit['null']}"
        assert unit["realization_id"].startswith("p5b_t4_test:train:")
        assert len(unit["looks"]) == PLAN.k
        for look in unit["looks"]:
            assert set(look) >= {
                "J",
                "look_days",
                "valid_count",
                "total",
                "invalid_rate",
                "quantile",
                "c",
                "eligible",
                "rejection",
            }
            assert look["total"] == COUNT
            if look["eligible"]:
                assert look["c"] == pytest.approx(-look["quantile"], rel=0, abs=1e-15)


def test_rule_steps_are_frozen_five():
    assert len(TRAINING_RULE_STEPS) >= 5
    assert TRAINING_RULE_STEPS[1].startswith("per_look_lower_quantile")
    assert "holdout" in TRAINING_RULE_STEPS[-1]


def test_train_rejects_loosened_gates():
    """两个门都是冻结值：训练侧也拒绝放宽（与 power/holdout 同口径）。"""
    with pytest.raises(ThresholdError):
        _train(config=ThresholdConfig(invalid_rate_per_look_max=1.0))
    with pytest.raises(ThresholdError):
        _train(config=ThresholdConfig(min_valid_quantile_sample=1))


def test_train_rejects_holdout_registry():
    with pytest.raises(ThresholdError):
        _train(registry=_registry(role="holdout"))


def test_train_rejects_alpha_allocation_mismatch():
    with pytest.raises(ThresholdError):
        _train(config=ThresholdConfig(alpha_train=0.045, alpha_per_look=0.02))


def test_train_flags_invalid_rate_over_max():
    """合成场景退化（sd=0）⇒ 该单元无效率 100%，配置整体失败（不得排除后继续）。"""
    result = _train(registry=_registry(sd_daily=0.0))
    assert not result.passed
    kinds = {item["reason"] for item in result.failures}
    assert "invalid_rate_exceeds_max" in kinds
    # 门槛不通过时不得产生"可注册"的结论：仅真实块场景仍可估（且阈值必须为正）
    assert all(unit.startswith("real_seed_block") for unit in result.determining_units)
    assert all(value > 0 for value in result.thresholds if np.isfinite(value))


def test_train_flags_conflict_when_all_units_degenerate():
    """含真实块场景也退化（池全零）⇒ 阈值不可用（NaN）且硬失败，绝不返回非正阈值。"""
    result = _train(registry=_registry(sd_daily=0.0, pool=np.zeros(260)))
    assert not result.passed
    kinds = {item["reason"] for item in result.failures}
    assert "invalid_rate_exceeds_max" in kinds
    assert "threshold_not_estimable" in kinds or "threshold_not_positive" in kinds
    assert all(np.isnan(result.thresholds))


def test_train_flags_insufficient_valid_sample():
    result = _train(counts={"train_per_unit": MIN_VALID_QUANTILE_SAMPLE - 10})
    assert not result.passed
    reasons = {item["reason"] for item in result.failures}
    assert "insufficient_valid_sample" in reasons


def test_training_result_serialises_for_artifact():
    payload = _train().as_dict()
    assert payload["schema"] == "p5b_testbed_calibration_thresholds/v1"
    assert payload["plan"]["look_blocks"] == [6, 12, 18, 25]
    assert payload["config"]["alpha_per_look"] == pytest.approx(0.045 / 4)
    assert len(payload["units"]) == 8
    assert payload["failures"] == []


def test_threshold_config_from_manifest(tmp_path):
    built = env.fast_manifest(tmp_path)
    config = ThresholdConfig.from_manifest(built.doc)
    assert config.alpha_train == pytest.approx(0.045)
    assert config.alpha_per_look == pytest.approx(0.01125)
    assert config.invalid_rate_per_look_max == pytest.approx(0.10)
    assert config.min_valid_quantile_sample == MIN_VALID_QUANTILE_SAMPLE


def test_threshold_artifact_written_once(tmp_path):
    prepared = env.prepare_campaign(tmp_path)
    first = env.train(prepared)
    assert first["passed"]
    target = artifact_path(prepared.root, ARTIFACT_THRESHOLDS)
    assert target.is_file()
    with pytest.raises(TestbedError):
        env.train(prepared)


def test_holdout_does_not_touch_thresholds_bytes(tmp_path):
    prepared = env.prepare_campaign(tmp_path)
    env.train(prepared)
    target = artifact_path(prepared.root, ARTIFACT_THRESHOLDS)
    before = target.read_bytes()
    env.holdout(prepared)
    assert target.read_bytes() == before


def test_holdout_requires_trained_thresholds(tmp_path):
    prepared = env.prepare_campaign(tmp_path)
    with pytest.raises(TestbedError):
        env.holdout(prepared)


def test_config_failure_writes_diagnostic_artifact_with_nulls(tmp_path):
    """配置失败时仍落盘诊断产物（NaN 序列化为 null），且不得进入 holdout。"""
    counts = dict(env.SMALL_COUNTS)
    counts["train_per_unit"] = 60  # < min_valid_quantile_sample(100) ⇒ 阈值不可估计
    prepared = env.prepare_campaign(tmp_path, counts=counts)
    outcome = env.train(prepared)
    assert outcome["passed"] is False
    assert outcome["c_by_look"] == [None, None, None, None]
    from src.lazybull.v2.hosts.testbed.referee import ARTIFACT_THRESHOLDS, load_artifact

    doc = load_artifact(artifact_path(prepared.root, ARTIFACT_THRESHOLDS), what="临界值产物")
    assert doc["passed"] is False
    assert doc["c_by_look"] == [None, None, None, None]
    assert doc["failure_detail"]
    for unit in doc["units"]:
        for look in unit["looks"]:
            assert look["c"] is None
            assert look["rejection"] in ("insufficient_valid_sample", "invalid_rate_exceeds_max")
    with pytest.raises(TestbedError):
        env.holdout(prepared)


def test_unit_details_record_quantile_exceedance_audit(tmp_path):
    """c_k 是经验分位：登记期望超越数与次序统计量序号（不参与阈值选择）。"""
    prepared = env.prepare_campaign(tmp_path)
    env.train(prepared)
    from src.lazybull.v2.hosts.testbed.referee import ARTIFACT_THRESHOLDS, load_artifact

    doc = load_artifact(artifact_path(prepared.root, ARTIFACT_THRESHOLDS), what="临界值产物")
    config = doc["config"]
    for unit in doc["units"]:
        for look in unit["looks"]:
            assert look["expected_exceedances"] == pytest.approx(
                config["alpha_per_look"] * look["total"]
            )
            assert look["order_statistic_index"] >= 1


def test_thresholds_artifact_links_manifest_and_calibrator(tmp_path):
    prepared = env.prepare_campaign(tmp_path)
    outcome = env.train(prepared)
    from src.lazybull.v2.hosts.testbed.referee import (
        ARTIFACT_MANIFEST,
        calibrator_identity,
        load_artifact,
    )

    doc = load_artifact(artifact_path(prepared.root, ARTIFACT_THRESHOLDS), what="临界值产物")
    manifest_sha = env._sha256(artifact_path(prepared.root, ARTIFACT_MANIFEST))
    assert doc["manifest_sha256"] == manifest_sha
    assert doc["calibrator_sha256"] == calibrator_identity()["sha256"]
    assert doc["passed"] is True
    assert outcome["thresholds_sha256"] == env._sha256(
        artifact_path(prepared.root, ARTIFACT_THRESHOLDS)
    )
