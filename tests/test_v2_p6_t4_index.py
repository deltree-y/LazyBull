# -*- coding: utf-8 -*-
"""P6 T4 索引/指标键集守恒与 CSV schema（设计 §5.7）。

覆盖：每个发现 run 恰一行（含 ``invalid``）；每 run 恰五条指标行（缺失也留状态行）；
两份 CSV 的列序/行键/manifest 登记与 ``index_mode=full``。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.lazybull.v2.reporting import generate, trend_contract, trend_csv
from tests import p6_t4_env


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p6_t4_index") / "corpus"
    p6_t4_env.build_corpus(root)
    return root


def _texts(result: dict) -> tuple[str, str]:
    return (
        p6_t4_env.read_csv_text(result, "run_index.csv"),
        p6_t4_env.read_csv_text(result, "trend.csv"),
    )


def test_p6_t4_index_key_conservation_including_invalid(
    corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """一个坏 run 仍保留恰一行索引 + 五条状态行；其余 run 行数/键集全部守恒。"""
    broken = corpus / "runs" / "p6t4_run_0001" / "chain_nav.parquet"
    backup = broken.read_bytes()
    broken.unlink()
    try:
        reports = tmp_path_factory.mktemp("p6_t4_index_invalid")
        result = generate.generate_trend(
            p6_t4_env.research_request(reports, corpus, attempt="index_invalid")
        )
    finally:
        broken.write_bytes(backup)
    index_text, trend_text = _texts(result)
    index = trend_csv.parse_run_index(index_text)
    trend = trend_csv.parse_trend(trend_text)
    trend_csv.assert_key_conservation(index, trend)
    total = len(index)
    assert total == 4  # tiny 档 research = 3 个 runs 批次 + 1 个旧 WF 运行
    assert {row["run_key"] for row in index} == {row["run_key"] for row in trend}
    grouped: dict[str, list[str]] = {}
    for row in trend:
        grouped.setdefault(row["run_key"], []).append(row["metric_name"])
    assert all(sorted(names) == sorted(trend_contract.METRIC_NAMES) for names in grouped.values())
    invalid = [row for row in index if row["report_status"] == "invalid"]
    assert len(invalid) == 1
    assert invalid[0]["degradation_reason_code"] == "source_contract_violation"
    broken_rows = [row for row in trend if row["run_key"] == invalid[0]["run_key"]]
    assert all(row["report_status"] == "invalid" for row in broken_rows)
    assert all(row["value"] == "" for row in broken_rows)


def test_p6_t4_index_schema_and_manifest_registration(
    corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """列序/行键/manifest ``csv_schemas`` 与 `index_mode=full` 逐项一致。"""
    reports = tmp_path_factory.mktemp("p6_t4_index_schema")
    result = generate.generate_trend(
        p6_t4_env.research_request(reports, corpus, attempt="index_schema")
    )
    index_text, trend_text = _texts(result)
    assert index_text.splitlines()[0].split(",") == list(trend_contract.RUN_INDEX_COLUMNS)
    assert trend_text.splitlines()[0].split(",") == list(trend_contract.TREND_COLUMNS)
    index = trend_csv.parse_run_index(index_text)
    trend = trend_csv.parse_trend(trend_text)
    assert {row["metric_name"] for row in trend} <= set(trend_contract.METRIC_NAMES)
    assert all(row["index_mode"] == "full" for row in index)
    assert all(row["source_sha256"] == row["source_identity_sha256"] for row in index)
    assert all(":" in row["source_ref"] for row in index)
    record = json.loads(result["files"]["report_manifest.json"].decode("utf-8"))
    schemas = record["csv_schemas"]
    for name in trend_csv.TREND_FILENAMES:
        assert schemas[name]["columns"] == list(
            trend_contract.RUN_INDEX_COLUMNS
            if name == "run_index.csv"
            else trend_contract.TREND_COLUMNS
        )
        assert schemas[name]["key_columns"] == list(
            trend_contract.RUN_INDEX_KEYS if name == "run_index.csv" else trend_contract.TREND_KEYS
        )
    assert schemas["run_index.csv"]["row_count"] == len(index)
    assert schemas["trend.csv"]["row_count"] == len(index) * 5
    assert record["outputs"]["run_index.csv"] == schemas["run_index.csv"]["sha256"]
    assert record["outputs"]["trend.csv"] == schemas["trend.csv"]["sha256"]
