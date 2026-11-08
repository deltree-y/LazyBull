# -*- coding: utf-8 -*-
"""P6 T3 制度摘要（P6-T3-06：无 bins 不画伪分布、不重采样）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import experiment_projection, experiment_view  # noqa: E402


def _view(regime: dict) -> dict:
    projection = {
        "schema": experiment_projection.EXPERIMENT_SCHEMA,
        "generated_at": "20261106T000000Z",
        "evidence_class": "research",
        "baseline_arm": "base",
        "arms": [],
        "stability": {"status": {"report_status": "available"}, "rows": [], "partitions": 0},
        "regime": regime,
        "cards": [],
        "degradations": [],
        "notes": [],
    }
    return experiment_view.build_experiment_view(
        projection,
        report_id="p6_experiment_20261106T000000Z_0123456789ab",
        generated_at="20261106T000000Z",
        input_evidence_classes=["research"],
    )


def test_p6_t3_16_no_bins_is_not_available_without_pseudo_distribution():
    regime = experiment_projection.extract_regime(
        {"results": {"bull": {"cagr": 0.1}, "bear": {"cagr": -0.05}}, "point_delta": 0.02}
    )
    assert regime["status"]["report_status"] == "not_available"
    assert regime["status"]["degradation_reason_code"] == "no_regime_bins"
    assert regime["bins"] is None
    assert regime["point_delta"] == 0.02
    assert {row["scene"] for row in regime["results"]} == {"bull", "bear"}
    section = [item for item in _view(regime)["sections"] if item["id"] == "sec-regime"][0]
    # 只有摘要表，没有分箱计数表（不画伪分布）
    assert len(section["tables"]) == 1
    assert not section["charts"]
    assert section["status"]["word"] == "not_available"
    facts = {fact["label"]: fact["value"] for fact in section["facts"]}
    assert facts["可得 bins"].startswith("无")

    with_bins = experiment_projection.extract_regime(
        {"results": {"bull": {"cagr": 0.1}}, "bins": {"state": {"bull": 3, "bear": 2}}}
    )
    assert with_bins["status"]["report_status"] == "available"
    assert with_bins["bins"] == {"state": {"bear": 2, "bull": 3}}
    section_with_bins = [
        item for item in _view(with_bins)["sections"] if item["id"] == "sec-regime"
    ][0]
    assert len(section_with_bins["tables"]) == 2
