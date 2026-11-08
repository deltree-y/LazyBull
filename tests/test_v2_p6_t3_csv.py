# -*- coding: utf-8 -*-
"""P6 T3 五份完整 CSV 守恒（P6-T3-07~11：schema/行键/行数/原行序/全值/哈希）。"""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import experiment_csv  # noqa: E402
from tests.p6_t3_env import standard_projection  # noqa: E402


def test_p6_t3_07_five_csv_schema_rows_order_and_hash(tmp_path):
    """五份 CSV 齐备：固定列序、行数守恒、链式原行序、哈希与回读一致。"""
    _arms, built = standard_projection(tmp_path)
    rows = built["csv_rows"]
    bundle = experiment_csv.build_csv_bundle(rows)
    assert set(bundle["files"]) == set(experiment_csv.CSV_FILENAMES)
    for name in experiment_csv.CSV_FILENAMES:
        schema = bundle["schemas"][name]
        assert schema["columns"] == list(experiment_csv._SCHEMAS[name])
        assert schema["row_count"] == len(rows[name])
        assert schema["key_columns"] == list(experiment_csv._KEYS[name])
        payload = bundle["files"][name]
        assert payload == payload.decode("utf-8").encode("utf-8")
        assert payload.startswith(",".join(schema["columns"]).encode("utf-8") + b"\n")
        assert b"\r" not in payload
        assert schema["sha256"] == sha256(payload).hexdigest()
        verified = experiment_csv.verify_csv_text(
            name, payload.decode("utf-8"), expected_rows=len(rows[name])
        )
        assert verified == schema
    # 链式净值：行数与源行数一致，且每折 source_row_index 从 0 连续递增（原行序）
    chain = rows["chain_nav.csv"]
    assert len(chain) == sum(arm.rows_used for arm in _arms)
    for arm_id in {row["arm_id"] for row in chain}:
        for split in {row["split_index"] for row in chain if row["arm_id"] == arm_id}:
            indexes = [
                row["source_row_index"]
                for row in chain
                if row["arm_id"] == arm_id and row["split_index"] == split
            ]
            assert indexes == list(range(len(indexes)))
    # 归因行数 = 源归因行数总和（含非成功状态）
    expected_attr = sum(
        len(fold.attribution) for arm in _arms for fold in arm.folds.values() if fold.attribution is not None
    )
    assert bundle["schemas"]["execution_attribution.csv"]["row_count"] == expected_attr


def test_p6_t3_08_missing_or_extra_rows_fail(tmp_path):
    _arms, built = standard_projection(tmp_path)
    rows = {name: list(items) for name, items in built["csv_rows"].items()}
    dropped = dict(rows)
    dropped["chain_nav.csv"] = rows["chain_nav.csv"][:-1]
    with pytest.raises(experiment_csv.CsvContractError, match="行数"):
        experiment_csv.build_csv_bundle(
            dropped, expected_rows={"chain_nav.csv": len(rows["chain_nav.csv"])}
        )
    duplicated = dict(rows)
    duplicated["arm_metrics.csv"] = rows["arm_metrics.csv"] + [dict(rows["arm_metrics.csv"][0])]
    with pytest.raises(experiment_csv.CsvContractError, match="行键重复"):
        experiment_csv.build_csv_bundle(duplicated)
    # 缺任何一份 CSV 立即失败
    partial = {name: rows[name] for name in experiment_csv.CSV_FILENAMES if name != "stability.csv"}
    with pytest.raises(experiment_csv.CsvContractError, match="缺 CSV"):
        experiment_csv.build_csv_bundle(partial)


def test_p6_t3_28_extra_unique_row_fails_against_expected_rows(tmp_path):
    """多一条**键唯一**的新行也必须失败（期望行数来自源侧，A-07）。"""
    _arms, built = standard_projection(tmp_path)
    rows = {name: list(items) for name, items in built["csv_rows"].items()}
    ghost = dict(rows["arm_metrics.csv"][0])
    ghost["arm_id"] = "ghost"
    rows["arm_metrics.csv"] = rows["arm_metrics.csv"] + [ghost]
    expected = {"arm_metrics.csv": len(built["csv_rows"]["arm_metrics.csv"])}
    with pytest.raises(experiment_csv.CsvContractError, match="行数"):
        experiment_csv.build_csv_bundle(rows, expected_rows=expected)


def test_p6_t3_30_stability_expected_rows_are_source_side(tmp_path):
    """`stability.csv` 期望行数来自源侧分区计数：多一条键唯一行必须失败（A-R2-01）。"""
    _arms, built = standard_projection(tmp_path)
    rows = {name: list(items) for name, items in built["csv_rows"].items()}
    ghost = dict(rows["stability.csv"][0])
    ghost["group_key"] = "f" * 64
    rows["stability.csv"] = rows["stability.csv"] + [ghost]
    expected = dict(built["csv_expected_rows"])
    assert expected["stability.csv"] == len(built["csv_rows"]["stability.csv"])
    with pytest.raises(experiment_csv.CsvContractError, match="行数"):
        experiment_csv.build_csv_bundle(rows, expected_rows=expected)


def test_p6_t3_29_publish_experiment_fixed_file_set():
    """实验 run 固定文件集 = 三件 + 五 CSV；缺一/多一均 PublishError（B-12）。"""
    from src.lazybull.v2.reporting import publish

    files = {name: b"x" for name in publish.EXPERIMENT_FILENAMES}
    assert publish.RUN_FILENAMES_BY_KIND["experiment"] == (
        publish.DASHBOARD_FILENAMES + experiment_csv.CSV_FILENAMES
    )
    assert set(files) == set(publish.RUN_FILENAMES_BY_KIND["experiment"])
    with pytest.raises(publish.PublishError, match="缺"):
        publish._validate_files(
            {name: payload for name, payload in files.items() if name != "stability.csv"},
            report_kind="experiment",
        )
    extra = dict(files)
    extra["extra.csv"] = b"x"
    with pytest.raises(publish.PublishError, match="未登记"):
        publish._validate_files(extra, report_kind="experiment")
    with pytest.raises(publish.PublishError, match="只允许"):
        publish._validate_files(files, report_kind="dashboard")


def test_p6_t3_09_missing_column_fails(tmp_path):
    _arms, built = standard_projection(tmp_path)
    row = dict(built["csv_rows"]["arm_metrics.csv"][0])
    row.pop("delta_cagr")
    with pytest.raises(experiment_csv.CsvContractError, match="缺列"):
        experiment_csv.render_csv("arm_metrics.csv", [row])
    extra = dict(built["csv_rows"]["arm_metrics.csv"][0])
    extra["not_declared"] = 1
    with pytest.raises(experiment_csv.CsvContractError, match="未声明列"):
        experiment_csv.render_csv("arm_metrics.csv", [extra])


def test_p6_t3_10_nan_and_inf_fail(tmp_path):
    _arms, built = standard_projection(tmp_path)
    for bad in (float("nan"), float("inf"), float("-inf")):
        row = dict(built["csv_rows"]["chain_nav.csv"][0])
        row["nav"] = bad
        with pytest.raises(experiment_csv.CsvContractError, match="非有限值"):
            experiment_csv.render_csv("chain_nav.csv", [row])


def test_p6_t3_11_line_endings_bom_and_lf_contract(tmp_path):
    _arms, built = standard_projection(tmp_path)
    text = experiment_csv.render_csv("stability.csv", built["csv_rows"]["stability.csv"])
    assert "\r" not in text and text.endswith("\n")
    with pytest.raises(experiment_csv.CsvContractError, match="CR"):
        experiment_csv.verify_csv_text("stability.csv", text.replace("\n", "\r\n"))
    with pytest.raises(experiment_csv.CsvContractError, match="BOM"):
        experiment_csv.verify_csv_text("stability.csv", "\ufeff" + text)
    with pytest.raises(experiment_csv.CsvContractError, match="列序"):
        experiment_csv.verify_csv_text("stability.csv", "wrong\n")
