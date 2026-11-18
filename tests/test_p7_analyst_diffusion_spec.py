"""AD-ML-20-v1 冻结规格测试：列序、单笔下限、目标与声明。"""

from __future__ import annotations

import json

import pytest

from scripts.v2_p7.analyst_diffusion_ml import spec


def test_features_frozen_order_and_count() -> None:
    assert len(spec.FEATURES) == 15
    assert spec.FEATURES[:7] == spec.EXISTING_FEATURES
    assert spec.FEATURES[7:] == spec.NEW_FEATURES
    assert spec.NEW_FEATURES == (
        "ad_eps_net_diffusion_90",
        "ad_eps_up_ratio_90",
        "ad_eps_revision_median_90",
        "ad_eps_revision_breadth_90",
        "ad_target_net_diffusion_90",
        "ad_rating_net_diffusion_90",
        "ad_active_analysts_90_log",
        "ad_latest_report_age_days",
    )
    assert len(set(spec.FEATURES)) == 15


def test_min_ticket_value_matches_formula() -> None:
    expected = max(25000.0, 5.0 / 0.0001954)
    assert spec.min_ticket_value() == pytest.approx(expected)
    assert spec.min_ticket_value() == pytest.approx(25588.536335721597)


def test_portfolio_matches_protocol() -> None:
    assert spec.PORTFOLIO["top_n"] == 6
    assert spec.PORTFOLIO["holding_period"] == 20
    assert spec.PORTFOLIO["rebalance_freq"] == 20
    assert spec.PORTFOLIO["stagger_tranches"] == 2
    assert spec.PORTFOLIO["tranche_slots"] == [3, 3]
    assert sum(spec.PORTFOLIO["tranche_slots"]) == 6
    assert spec.PORTFOLIO["tranche_offsets"] == [0, 10]
    assert spec.PORTFOLIO["max_per_industry"] == 2
    assert spec.PORTFOLIO["buy_retry_days"] == 0
    assert spec.PORTFOLIO["fill_window_days"] == 0


def test_label_and_report_windows() -> None:
    assert spec.LABEL["entry_offset"] == 1
    assert spec.LABEL["exit_offset"] == 21
    assert spec.LABEL["min_industry_members"] == 5
    assert spec.REPORT["revision_lookback_days"] == 180
    assert spec.REPORT["diffusion_window_days"] == 90
    assert spec.UNIVERSE["min_active_analysts_90"] == 2
    assert spec.UNIVERSE["min_scored_per_day"] == 100


def test_targets_are_prescreen_not_admission() -> None:
    assert spec.TARGETS["cagr_delta_min"] == pytest.approx(0.005)
    assert spec.TARGETS["mdd_loss_delta_max"] == pytest.approx(0.01)
    assert spec.TARGETS["capital_scenarios"] == [1_000_000, 2_000_000]
    assert spec.TARGETS["initial_b_fraction"] == pytest.approx(0.2)
    document = spec.machine_spec_document()
    assert document["formal_admission"] == "not_evaluated"
    assert document["purpose"] == "economic_prescreen"
    assert document["intended_admission_type"] == "enhancement"
    assert document["production_changed"] is False
    assert document["data_purchase_yuan"] == 0
    assert document["model_count_expected"] == 16
    assert document["not_virgin_market_holdout"] is True


def test_machine_spec_has_no_placeholder_and_real_sha(tmp_path=None) -> None:
    document = json.loads(json.dumps(spec.machine_spec_document()))
    text = json.dumps(document, ensure_ascii=False)
    assert "实施时计算" not in text
    assert len(document["protocol_sha256"]) == 64
    assert len(document["spec_sha256"]) == 64
    assert document["selection_history"]


def test_spec_sha_changes_with_numeric_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    original = spec.spec_sha256()
    monkeypatch.setitem(spec.MODEL, "max_depth", 4)
    changed = spec.spec_sha256()
    monkeypatch.setitem(spec.MODEL, "max_depth", 3)
    assert original != changed
    assert spec.spec_sha256() == original


def test_n_jobs_capped_at_eight() -> None:
    assert spec.n_jobs(4) == 4
    assert spec.n_jobs(64) == 8
    assert spec.n_jobs(1) == 1
