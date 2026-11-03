import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.v2_p5b.referee_psd_calibration_v2 import (
    config_consistency_errors,
    evaluation_failures,
    psd_variance,
    register_mde,
    unit_rng,
)

ROOT = Path(__file__).resolve().parents[1]
R2_MANIFEST = ROOT / "docs/plans/v2/p5b_calibration_manifest_psd_20261101_r2.json"


def test_validation_invalid_rate_blocks_pass_even_with_zero_false_alarm():
    validation = {
        "ma20|mu=0": {
            "wilson_high": 0.02,
            "invalid_rate_by_look": [0.0, 0.0, 1.0, 0.0],
        }
    }
    failures = evaluation_failures(validation, max_invalid=0.10, max_wilson=0.06)
    assert any(f["reason"] == "invalid_rate" for f in failures)
    assert any(f["look"] == 2 for f in failures)


def test_validation_wilson_blocks_pass():
    validation = {
        "ma20|mu=0": {
            "wilson_high": 0.061,
            "invalid_rate_by_look": [0.0, 0.0, 0.0, 0.0],
        }
    }
    assert any(f["reason"] == "wilson" for f in evaluation_failures(validation, 0.10, 0.06))


def test_mde_not_registered_on_failure():
    effects = {"5.0": {"ma20": {"rate": 0.99}}}
    assert register_mde(False, effects, [5.0], ["ma20"]) is None


def test_mde_registered_only_on_pass():
    effects = {"5.0": {"ma20": {"rate": 0.80}}}
    assert register_mde(True, effects, [5.0], ["ma20"]) == {"ma20": 5.0}


def test_unit_rng_is_stable_and_unit_separated():
    rng1, rid1 = unit_rng(11, "train", "ma20", "mu=0", prefix="r2")
    rng2, rid2 = unit_rng(11, "train", "ma20", "mu=0", prefix="r2")
    rng3, _ = unit_rng(11, "validation", "ma20", "mu=0", prefix="r2")
    assert rid1 == rid2 == "r2:train:ma20:mu=0:0"
    assert np.array_equal(rng1.normal(size=8), rng2.normal(size=8))
    assert not np.array_equal(rng1.normal(size=8), rng3.normal(size=8))


def test_psd_variance_has_no_negative_artifact():
    assert psd_variance(np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])) == pytest.approx(0.2)
    assert psd_variance(np.ones(6)) == pytest.approx(0.0)


def _r2_manifest():
    return json.loads(R2_MANIFEST.read_text(encoding="utf-8"))


def test_manifest_config_consistency_accepts_r2():
    assert config_consistency_errors(_r2_manifest()) == []


def test_manifest_rejects_block_length_mismatch():
    data = copy.deepcopy(_r2_manifest())
    data["calibration"]["block_resample"]["block_length"] = 21
    errors = config_consistency_errors(data)
    assert any("block_length" in e for e in errors)


def test_manifest_rejects_alpha_allocation_mismatch():
    data = copy.deepcopy(_r2_manifest())
    data["threshold"]["alpha_per_look"] = 0.02
    errors = config_consistency_errors(data)
    assert any("alpha_per_look" in e for e in errors)
