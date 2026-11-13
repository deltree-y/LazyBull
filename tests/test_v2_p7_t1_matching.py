# -*- coding: utf-8 -*-
"""P7 T1 匹配组构造的定点回归（设计 §6.4 冻结口径）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v2_p7.t1 import matching  # noqa: E402


def test_standardize_by_mad_uses_median_and_clip() -> None:
    candidates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.array([0.0, 3.0, 10.0])
    result = matching.standardize_by_mad(events, candidates, clip=3.0)
    assert result.median == 3.0
    assert result.mad == 1.0
    assert result.dropped is False
    assert result.z.tolist() == [-3.0, 0.0, 3.0]


def test_zero_mad_variable_is_dropped_not_zero_distance() -> None:
    candidates = np.full(5, 7.0)
    events = np.array([1.0, 7.0])
    result = matching.standardize_by_mad(events, candidates)
    assert result.dropped is True
    assert np.isnan(result.z).all()


def test_match_one_day_prefers_near_and_breaks_ties_by_key() -> None:
    candidate_covariates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ]
    )
    candidate_industry = np.array([0, 0, 0, 1, 1, 1])
    event_covariates = np.array([[0.0, 0.0, 0.0]])
    event_industry = np.array([0])
    result = matching.match_one_day(
        event_covariates,
        candidate_covariates,
        event_industry,
        candidate_industry,
        candidate_keys=np.arange(6),
    )
    assert result.controls.tolist() == [[0, 1, 2, 3, 4]]
    assert result.distances[0, 0] == pytest.approx(0.0, abs=1e-6)
    # 诊断存「同业」而非「异业」：前三个同业、后两个异业，且两方均已知
    assert result.industry_match[0].tolist() == [True, True, True, False, False]
    assert result.industry_known[0].tolist() == [True, True, True, True, True]

    tie = matching.match_one_day(
        event_covariates,
        np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [9.0, 0.0, 0.0]]),
        event_industry,
        np.array([0, 0, 0]),
        config=matching.MatchConfig(control_count=2),
        candidate_keys=np.array([7, 3, 9]),
    )
    assert tie.controls[0, 0] == 1  # 同距离时按决胜键升序（key=3 < key=7）
    assert tie.controls[0, 1] == 0


def test_insufficient_candidates_returns_minus_one() -> None:
    result = matching.match_one_day(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([0]),
        np.array([0, 0]),
    )
    assert result.controls.tolist() == [[-1, -1, -1, -1, -1]]


class _Panel:
    """最小面板：只填 candidate_mask 需要的字段。"""

    def __init__(self, n_days: int = 2, n_stocks: int = 5) -> None:
        shape = (n_days, n_stocks)
        self.n_days = n_days
        self.open_u = np.full(shape, 10.0)
        self.close_adj = np.full(shape, 10.0)
        self.high = np.full(shape, 10.5)
        self.low = np.full(shape, 9.5)
        self.pre_close = np.full(shape, 10.0)
        self.total_mv = np.full(shape, 100000.0)
        self.ret20 = np.full(shape, 0.02)
        self.vol20 = np.full(shape, 0.03)
        self.announce = np.zeros(shape, dtype=bool)
        self.industry = np.array([0, 0, 1, 1, -1], dtype=np.int32)
        self.industry_since = np.zeros(n_stocks, dtype=np.int32)
        self.industry_matrix = None
        self.listed = np.ones(shape, dtype=bool)
        self.share_mult = np.ones(shape)
        self.share_unknown = np.zeros(shape, dtype=np.int32)
        self.days = [
            f"2020{(index // 28) + 1:02d}{(index % 28) + 1:02d}" for index in range(n_days)
        ]
        self.stocks = [f"00000{index}.SZ" for index in range(n_stocks)]
        self.stock_index = {code: index for index, code in enumerate(self.stocks)}


def test_candidate_mask_excludes_announcements_and_incomplete_quotes() -> None:
    panel = _Panel()
    panel.announce[0, 2] = True
    panel.open_u[0, 3] = 0.0
    panel.ret20[0, 4] = np.nan
    mask = matching.candidate_mask(panel, 0)
    assert mask.tolist() == [True, True, False, False, False]


def test_build_groups_marks_missing_covariates_as_unresolved() -> None:
    panel = _Panel(n_days=40, n_stocks=8)
    panel.stocks = [f"00000{index}.SZ" for index in range(8)]
    panel.stock_index = {code: index for index, code in enumerate(panel.stocks)}
    panel.industry = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
    panel.announce[1, 0] = True  # 事件股当日有公告
    panel.announce[1, 1] = True  # 另一只有公告的股票
    panel.ret20[1, 0] = np.nan  # 事件股协变量缺失
    events = pd.DataFrame(
        {
            "event_source": ["forecast", "forecast"],
            "ts_code": [panel.stocks[0], panel.stocks[1]],
            "event_date": ["20200331", "20200331"],
            "knowledge_date": [panel.days[1], panel.days[1]],
            "revision_key": ["a", "b"],
            "predicate_value": [1.0, 1.0],
            "fingerprint": ["f0", "f1"],
        }
    )
    table, diagnostics, domain = matching.build_groups(
        events,
        panel,
        source="forecast",
        evaluation_start_pos=0,
        horizon=1,
    )
    assert table.size == 2
    assert table.status.tolist() == [
        matching.STATUS_EVENT_COVARIATE_MISSING,
        matching.STATUS_OK,
    ]
    assert table.members[0].tolist()[1:] == [-1, -1, -1, -1, -1]
    assert domain["groups"] == 2
    assert diagnostics.events_covariate_missing == 1
