# -*- coding: utf-8 -*-
"""T1 行业来源开关（snapshot / pit）单测：快照口径逐值不变 + PIT 区间矩阵正确。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.t1 import dataplane, matching


def _write_pit(root: Path, rows) -> Path:
    path = root / "clean" / "sw_industry_pit.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["ts_code", "in_date", "out_date", "l2_code", "sw_version"])
    frame.to_parquet(path, index=False)
    return path


def test_pit_matrix_splits_by_interval(tmp_path):
    _write_pit(
        tmp_path,
        [
            ("000001.SZ", "20200101", "20200103", "480100", "SW2014"),
            ("000001.SZ", "20200103", None, "480300", "SW2021"),
        ],
    )
    days = ["20200101", "20200102", "20200103", "20200104"]
    matrix, info = dataplane._industry_matrix(tmp_path, ["000001.SZ", "600000.SH"], days)

    assert matrix.shape == (4, 2)
    assert matrix[0, 0] == matrix[1, 0]
    assert matrix[2, 0] == matrix[3, 0]
    assert matrix[0, 0] != matrix[2, 0]
    assert np.all(matrix[:, 1] == -1)
    assert info["mode"] == "pit"
    assert info["level"] == "sw_l2_code"
    assert info["matched_rows"] == 2


def test_pit_matrix_counts_unmapped_version_as_known(tmp_path):
    _write_pit(
        tmp_path,
        [("000001.SZ", "19910403", "20140221", "440100", "UNMAPPED")],
    )
    matrix, info = dataplane._industry_matrix(tmp_path, ["000001.SZ"], ["20000104", "20140221"])
    assert matrix[0, 0] >= 0
    assert matrix[1, 0] == -1
    assert info["version_policy"].startswith("UNMAPPED")


def test_pit_matrix_requires_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="缺少行业 PIT 区间表"):
        dataplane._industry_matrix(tmp_path, ["000001.SZ"], ["20200101"])


def test_industry_at_prefers_matrix():
    panel = SimpleNamespace(
        industry_matrix=np.array([[7], [8]], dtype=np.int32),
        industry=np.array([-1]),
        industry_since=np.array([-1]),
    )
    assert int(dataplane.industry_at(panel, 1, np.array([0]))[0]) == 8


def test_unknown_industry_candidate_is_ineligible():
    """v2 方案 A：行业未知候选即使协变量最接近也不得入选。"""
    event_covariates = np.zeros((1, 3), dtype=float)
    candidate_covariates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.02, 0.02, 0.02],
            [0.03, 0.03, 0.03],
            [0.04, 0.04, 0.04],
            [0.05, 0.05, 0.05],
            [0.06, 0.06, 0.06],
        ]
    )
    event_industry = np.array([0], dtype=np.int32)
    candidate_industry = np.array([-1, 0, 0, 0, 0, 0], dtype=np.int32)
    result = matching.match_one_day(
        event_covariates, candidate_covariates, event_industry, candidate_industry
    )
    assert 0 not in result.controls[0].tolist()
    assert result.industry_known[0].all()


def test_event_industry_unknown_yields_no_controls():
    event_covariates = np.zeros((1, 3), dtype=float)
    candidate_covariates = np.tile(np.array([[0.0, 0.0, 0.0]]), (6, 1))
    event_industry = np.array([-1], dtype=np.int32)
    candidate_industry = np.zeros(6, dtype=np.int32)
    result = matching.match_one_day(
        event_covariates, candidate_covariates, event_industry, candidate_industry
    )
    assert result.controls[0].tolist() == [-1] * 5


def test_industry_at_snapshot_fallback_unchanged():
    panel = SimpleNamespace(
        industry_matrix=None,
        industry=np.array([5, 9], dtype=np.int32),
        industry_since=np.array([0, 2], dtype=np.int32),
    )
    stocks = np.array([0, 1])
    assert list(dataplane.industry_at(panel, 0, stocks)) == [5, -1]
    assert list(dataplane.industry_at(panel, 1, stocks)) == [5, -1]
    assert list(dataplane.industry_at(panel, 2, stocks)) == [5, 9]
