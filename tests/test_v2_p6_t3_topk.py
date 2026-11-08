# -*- coding: utf-8 -*-
"""P6 T3 TopK 正日占比（计划 §3-T3：runs 固定降级；旧 WF 仅字段改名并对账）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare.aggregate import aggregate_run  # noqa: E402
from src.lazybull.v2.reporting import experiment_metrics, experiment_projection  # noqa: E402
from tests.p6_t3_env import load_arms, write_legacy_run, write_runs_batch  # noqa: E402


def test_p6_t3_09_runs_topk_fixed_degraded_and_not_computed(tmp_path):
    """规范化 runs：TopK 固定 ``no_unique_implementation``，禁止从 topk_detail 私算。"""
    batch = write_runs_batch(tmp_path / "runs_a", batch_id="batch_a", with_topk=True)
    arms = load_arms(tmp_path, [("a", "runs", batch)])
    arm = arms[0]
    assert arm.folds and all(fold.topk_detail is not None for fold in arm.folds.values())
    topk = experiment_metrics.topk_values(arm)
    assert topk["status"]["report_status"] == "not_available"
    assert topk["status"]["degradation_reason_code"] == "no_unique_implementation"
    assert topk["per_fold"] == {}
    assert topk["means"] == {}
    built = experiment_projection.build_experiment_projection(
        baseline="a",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "a"},
    )
    fold_rows = built["csv_rows"]["fold_metrics.csv"]
    assert fold_rows
    assert all(row["topk20_positive_day_ratio"] is None for row in fold_rows)
    assert all(row["topk30_positive_day_ratio"] is None for row in fold_rows)


def test_p6_t3_10_legacy_topk_mapped_and_reconciled(tmp_path):
    """旧 WF：逐折原值仅改名，跨折均值必须与 ``aggregate_run`` 完全一致。"""
    legacy = write_legacy_run(tmp_path / "legacy_a", run_id="wf_20260101_000000_a", topk=True)
    arms = load_arms(tmp_path, [("legacy", "legacy_wf", legacy)])
    arm = arms[0]
    topk = experiment_metrics.topk_values(arm)
    assert topk["status"]["report_status"] == "available"
    expected = aggregate_run(arm.summary)
    assert topk["means"]["topk20_positive_day_ratio_mean"] == expected["key_top20_hit_rate_mean"]
    assert topk["means"]["topk30_positive_day_ratio_mean"] == expected["key_top30_hit_rate_mean"]
    per_fold = topk["per_fold"]
    for _, row in arm.summary.iterrows():
        split = int(row["split_index"])
        assert per_fold[split]["topk20"] == float(row["KEY_Top20_hit_rate"])
        assert per_fold[split]["topk30"] == float(row["KEY_Top30_hit_rate"])
    built = experiment_projection.build_experiment_projection(
        baseline="legacy",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "legacy"},
    )
    fold_rows = built["csv_rows"]["fold_metrics.csv"]
    assert fold_rows
    assert all(row["topk20_positive_day_ratio"] is not None for row in fold_rows)
