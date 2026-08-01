#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""因子稳定性分析脚本测试。"""

import pandas as pd
import pytest

from scripts.ana.analyze_factor_stability import (
    _iter_leaf_models,
    compute_stability_stats,
    parse_versions,
)


class _LeafModel:
    feature_importances_ = [0.7, 0.3]


class _EnsembleModel:
    def __init__(self, models):
        self.models = models


def test_parse_versions_supports_ranges_prefixes_and_deduplication():
    assert parse_versions("v10-12, 11, v15") == [10, 11, 12, 15]


def test_parse_versions_rejects_reversed_range():
    with pytest.raises(ValueError, match="起点大于终点"):
        parse_versions("12-10")


def test_iter_leaf_models_recursively_expands_ensembles():
    first = _LeafModel()
    second = _LeafModel()
    model = _EnsembleModel([first, _EnsembleModel([second])])

    assert list(_iter_leaf_models(model)) == [first, second]


def test_compute_stability_stats_separates_strict_and_review_candidates():
    records = pd.DataFrame(
        [
            {
                "component": component,
                "feature": feature,
                "normalized_importance": importance,
                "rank": rank,
                "is_zero": is_zero,
                "is_top_half": is_top_half,
            }
            for component in ["v1#00", "v1#01", "v2#00", "v2#01"]
            for feature, importance, rank, is_zero, is_top_half in [
                ("strict_weak", 0.0, 3.0, True, False),
                ("review_weak", 0.1, 2.0, False, False),
                ("strong", 0.9, 1.0, False, True),
            ]
        ]
    )

    stats = compute_stability_stats(
        records,
        bottom_ratio=0.67,
        min_zero_ratio=0.50,
        max_top_half_ratio=0.20,
    ).set_index("feature")

    assert bool(stats.loc["strict_weak", "importance_candidate"])
    assert bool(stats.loc["strict_weak", "review_candidate"])
    assert not bool(stats.loc["review_weak", "importance_candidate"])
    assert bool(stats.loc["review_weak", "review_candidate"])
    assert not bool(stats.loc["strong", "review_candidate"])
