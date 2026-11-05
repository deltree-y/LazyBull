# -*- coding: utf-8 -*-
"""P6 T0.5 合成语料确定性（P6-T05-01）。

覆盖：标准语料双跑逐文件 SHA 一致、规模与 §7.3 冻结值一致、质量异常行计数口径、
压力语料规模冻结、已存在目录拒绝覆盖。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p6 import build_corpus  # noqa: E402
from src.lazybull.v2.reporting import contracts  # noqa: E402


def _entries(manifest: dict, profile: str) -> dict[str, dict]:
    return {entry["path"]: entry for entry in manifest["profiles"][profile]["files"]}


def _quality_status_counts(path: Path) -> tuple[int, int, int]:
    error = warning = total = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            if row["status"] == "error":
                error += 1
            elif row["status"] == "warning":
                warning += 1
    return total, error, warning


def test_p6_t05_01_corpus_double_run_is_byte_deterministic(tmp_path: Path) -> None:
    first = build_corpus.build(tmp_path / "first", profile="standard")
    second = build_corpus.build(tmp_path / "second", profile="standard")

    assert first["content_sha256"] == second["content_sha256"]
    assert _entries(first, "standard") == _entries(second, "standard")

    entries = _entries(first, "standard")
    expected = contracts.CORPUS_PROFILES["standard"]
    assert (
        entries["standard/dashboard/positions.json"]["rows"] == expected["dashboard"]["positions"]
    )
    assert (
        entries["standard/dashboard/instructions.csv"]["rows"]
        == expected["dashboard"]["instructions"]
    )
    assert entries["standard/dashboard/nav.csv"]["rows"] == expected["dashboard"]["nav_rows"]
    assert (
        entries["standard/experiment/runs/arm_a/summary.csv"]["rows"]
        == expected["experiment"]["folds"]
    )
    assert (
        entries["standard/experiment/runs/arm_a/chain_nav.csv"]["rows"]
        == expected["experiment"]["chain_rows_per_arm"]
    )
    assert (
        entries["standard/experiment/legacy_wf/arm_b/trades.csv"]["rows"]
        == expected["experiment"]["trade_rows_per_arm"]
    )
    assert "standard/experiment/runs/arm_a/topk_detail.csv" not in entries
    assert (
        entries["standard/trend/previous/run_index.csv"]["rows"]
        == expected["trend"]["previous_runs"]
    )
    assert entries["standard/trend/new/run_index.csv"]["rows"] == expected["trend"]["new_runs"]

    total, error, warning = _quality_status_counts(
        tmp_path / "first" / "standard" / "dashboard" / "quality_metrics.csv"
    )
    assert error + warning == expected["dashboard"]["quality_abnormal_rows"]
    assert total == expected["dashboard"]["quality_abnormal_rows"] + 4


def test_p6_t05_01_stress_profile_spec_is_frozen() -> None:
    stress = contracts.CORPUS_PROFILES["stress"]
    assert stress["dashboard"] == {
        "positions": 200,
        "instructions": 500,
        "nav_rows": 5_000,
        "quality_abnormal_rows": 1_000,
    }
    assert stress["experiment"]["chain_rows_per_arm"] == 5_000
    assert stress["experiment"]["topk_detail_rows_per_arm"] == 20_000
    assert stress["trend"] == {
        "previous_runs": 1_999,
        "new_runs": 1,
        "visible_rows": 500,
        "full_rows": 2_000,
    }


def test_p6_t05_01_corpus_refuses_existing_directory(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(build_corpus.CorpusError):
        build_corpus.build(target, profile="standard")
