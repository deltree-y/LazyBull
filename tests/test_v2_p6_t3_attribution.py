# -*- coding: utf-8 -*-
"""P6 T3 执行归因（计划 §3-T3 / P6-T3-13：19 列原值 + 原 status/reason 保留）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from src.lazybull.v2.evidence.runs_schema import ATTR_CONTRACT_COLS  # noqa: E402
from src.lazybull.v2.reporting import experiment_csv, experiment_projection  # noqa: E402
from tests.p6_t3_env import load_arms, write_runs_batch  # noqa: E402


def _project(tmp_path):
    batch = write_runs_batch(tmp_path / "runs_a", batch_id="batch_a", with_attribution=True)
    arms = load_arms(tmp_path, [("a", "runs", batch)])
    built = experiment_projection.build_experiment_projection(
        baseline="a",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "a"},
    )
    return arms[0], built


def test_p6_t3_17_attribution_keeps_19_columns_and_original_status(tmp_path):
    """22 列（19 契约 + 臂/源行序/状态）且列名唯一、行序不变、原 status/reason 保留。"""
    arm, built = _project(tmp_path)
    rows = built["csv_rows"]["execution_attribution.csv"]
    assert rows, "合成归因必须有行"
    # 19 契约列 + arm_id/source_row_index（行键）+ 状态/原因/来源 = 24，且列名唯一
    assert len(experiment_csv.EXECUTION_ATTR_COLUMNS) == 24
    assert len(set(experiment_csv.EXECUTION_ATTR_COLUMNS)) == 24
    text = experiment_csv.render_csv("execution_attribution.csv", rows)
    header = text.splitlines()[0].split(",")
    assert header.count("split_index") == 1
    assert len(header) == 24
    for row in rows:
        for column in ATTR_CONTRACT_COLS:
            assert column in row
        assert "report_status" in row and "degradation_reason_code" in row
    # 行序：逐折按 split 升序、折内 source_row_index 连续递增
    for split in sorted({row["split_index"] for row in rows}):
        indexes = [row["source_row_index"] for row in rows if row["split_index"] == split]
        assert indexes == list(range(len(indexes)))
    source = arm.folds[sorted(arm.folds)[0]].attribution
    first_fold_rows = [row for row in rows if row["split_index"] == sorted(arm.folds)[0]]
    for position, (_, item) in enumerate(source.iterrows()):
        assert first_fold_rows[position]["status"] == item["status"]
        assert first_fold_rows[position]["reason"] == item["reason"]
        assert first_fold_rows[position]["pred_score"] == float(item["pred_score"])
    statuses = {row["status"] for row in rows}
    assert {"filled", "rejected", "skipped"} <= statuses


def test_p6_t3_19_declared_duplicate_column_fails(tmp_path, monkeypatch):
    """声明列集自身重名必须 fail-closed（A-01 的层内负例）。"""
    _arm, built = _project(tmp_path)
    duplicated = experiment_csv.EXECUTION_ATTR_COLUMNS + ("split_index",)
    monkeypatch.setitem(experiment_csv._SCHEMAS, "execution_attribution.csv", duplicated)
    with pytest.raises(experiment_csv.CsvContractError, match="重复列名"):
        experiment_csv.render_csv(
            "execution_attribution.csv", built["csv_rows"]["execution_attribution.csv"]
        )


def test_p6_t3_18_non_success_status_counts_conserved(tmp_path):
    """非成功执行状态原值守恒：投影计数与源行一致，且不改写成报告状态词。"""
    arm, built = _project(tmp_path)
    rows = built["csv_rows"]["execution_attribution.csv"]
    expected: dict[str, int] = {}
    for row in rows:
        value = str(row["status"])
        if value in ("filled", "success"):
            continue
        expected[value] = expected.get(value, 0) + 1
    arm_entry = built["projection"]["arms"][0]
    assert arm_entry["attribution"]["non_success_status"] == dict(sorted(expected.items()))
    assert arm_entry["attribution"]["row_count"] == len(rows)
    # 报告状态词与源 status 不混淆
    assert all(row["report_status"] == "available" for row in rows)
