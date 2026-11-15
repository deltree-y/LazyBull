# -*- coding: utf-8 -*-
"""P-1 重开校准测试：层一容差分层（反例 7）与层二 DGP 结构。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.v2_p7.t1 import p1_calibration
from scripts.v2_p7.t1.p1_params import CostRates, P1RParams


def _params() -> P1RParams:
    return P1RParams(
        repo_root=SimpleNamespace(),
        prereg_path=SimpleNamespace(),
        prereg_sha256="a" * 64,
        prereg_id="p7-p1-redesign-20261113",
        criteria_digest="b" * 64,
        evidence_root=SimpleNamespace(),
        run_id="test",
        common_data_cutoff="20261029",
        evaluation_start="20111107",
        evaluation_end="20261029",
        evaluation_trading_days=3637,
        horizon_primary=10,
        horizon_sensitivity=5,
        entry_max_defer=10,
        lot_size=100,
        min_ticket=25588.536335721597,
        cost=CostRates(0.0001954, 5.0, 0.0005, 0.0005),
        initial_capital=300000.0,
        max_concurrent_positions=6,
        maturity_marker_days=60,
        bootstrap_main_block=90,
        bootstrap_aux_block=90,
        bootstrap_replicates=100,
        bootstrap_sensitivity=(45, 180),
        calibration_samples=2,
        calibration_replicates=20,
        seeds={
            "main_stationary": 42,
            "aux_circular": 1042,
            "layer2_calibration_base": 2042,
            "pilot": 3042,
            "sensitivity": 4042,
        },
    )


def test_m1_display_constant_zero_tolerance_must_fail():
    """反例 7：用字面常数 −18.8949bp 做零容差断言必须失败。"""
    params = _params()
    layer1 = p1_calibration.layer1_mechanical(params, {"display_bp": -18.894860513998565})
    literal = -18.8949 * 1e-4
    runtime = layer1["M1"]["expected_leg"]
    assert abs(runtime - literal) > 1e-12
    assert layer1["M1"]["passed"]
    assert layer1["M1"]["gap"] <= 1e-12


def test_layer1_mechanical_structure():
    params = _params()
    layer1 = p1_calibration.layer1_mechanical(params, {"display_bp": -18.894860513998565})
    assert layer1["layer1_passed"]
    assert len(layer1["M2"]["rows"]) == 5
    assert all(row["passed"] for row in layer1["M2"]["rows"])
    assert layer1["M3"]["lot_branch"]["integer_lots"]
    assert layer1["M3"]["min_commission_branch"]["triggered"]


def test_pilot_and_layer2_smoke():
    rng = np.random.default_rng(11)
    n_intents = 240
    day_pos = rng.integers(0, 60, size=n_intents)
    d = rng.normal(0.0, 0.01, size=n_intents)
    pilot = p1_calibration.estimate_pilot(d, day_pos, 60)
    assert -0.95 <= pilot["phi"] <= 0.95
    assert pilot["sigma_epsilon"] > 0
    layer2 = p1_calibration.layer2_statistical(
        pilot,
        day_pos,
        60,
        _params(),
        samples=2,
        replicates=20,
    )
    assert len(layer2["results"]) == 2
    assert layer2["samples_per_theta"] == 2
    assert "coverage" in layer2["results"][0]
    assert "negative_rate" in layer2["results"][0]
