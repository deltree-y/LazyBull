"""DEF-ML-60-v1 冻结规格测试（不读真实行情/账户）。"""

from __future__ import annotations

import re

import pytest

from scripts.v2_p7.defensive_ml import spec


def test_feature_list_is_frozen_19_in_order() -> None:
    assert len(spec.FEATURES) == 19
    assert len(set(spec.FEATURES)) == 19
    assert all(name.startswith("def_") for name in spec.FEATURES)
    assert spec.FEATURES[0] == "def_ret20"
    assert spec.FEATURES[-1] == "def_cashflow_age"
    assert spec.numeric_identity()["feature_count"] == 19


def test_feature_order_change_changes_numeric_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    before = spec.spec_sha256()
    swapped = list(spec.FEATURES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    monkeypatch.setattr(spec, "FEATURES", tuple(swapped))
    assert spec.spec_sha256() != before


def test_numeric_param_change_changes_spec_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    before = spec.spec_sha256()
    monkeypatch.setitem(spec.MODEL, "n_estimators", 301)
    assert spec.spec_sha256() != before


def test_spec_hash_is_stable_and_hex() -> None:
    first = spec.spec_sha256()
    second = spec.spec_sha256()
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_protocol_sha_matches_file_bytes() -> None:
    import hashlib

    expected = hashlib.sha256(spec.protocol_path().read_bytes()).hexdigest()
    assert spec.protocol_sha256() == expected
    assert re.fullmatch(r"[0-9a-f]{64}", expected)


def test_fixed_model_and_cost_parameters() -> None:
    assert spec.MODEL["quantiles"] == [0.10, 0.50, 0.90]
    assert spec.MODEL["objective"] == "reg:quantileerror"
    assert spec.MODEL["n_estimators"] == 300
    assert spec.MODEL["early_stopping"] is False
    assert spec.COSTS == {
        "commission_rate": 0.0001954,
        "min_commission": 5.0,
        "stamp_tax": 0.0005,
        "slippage": 0.0005,
    }
    assert spec.TARGETS["dd_improvement_min"] == 0.03
    assert spec.TARGETS["cagr_delta_min"] == -0.01
    assert spec.PORTFOLIO["tranche_slots"] == [7, 7, 6]
    assert sum(spec.PORTFOLIO["tranche_slots"]) == spec.PORTFOLIO["top_n"]
    assert spec.SPLITS["expected_models"] == (
        spec.SPLITS["expected_folds"] * spec.SPLITS["models_per_fold"]
    )


def test_n_jobs_is_bounded_by_eight() -> None:
    assert spec.n_jobs(20) == 8
    assert spec.n_jobs(4) == 4
    assert spec.n_jobs(1) == 1


def test_machine_spec_document_declares_not_evaluated() -> None:
    doc = spec.machine_spec_document()
    assert doc["formal_admission"] == "not_evaluated"
    assert doc["production_changed"] is False
    assert doc["data_purchase_yuan"] == 0
    assert doc["feature_count"] == 19
    assert doc["score_formula"] == "0.5*q10_fixed+0.5*q50_fixed"
    assert doc["not_virgin_market_holdout"] is True
    assert doc["protocol_sha256"] == spec.protocol_sha256()
    assert doc["spec_sha256"] == spec.spec_sha256()


def test_unit_rules_single_conversion_each() -> None:
    rules = spec.UNIT_RULES
    assert rules["daily_amount"]["multiplier"] == 1000.0
    assert rules["daily_basic_total_mv"]["multiplier"] == 10000.0
    assert rules["daily_basic_turnover_rate"]["multiplier"] == 0.01
    assert set(rules) == {"daily_amount", "daily_basic_total_mv", "daily_basic_turnover_rate"}
