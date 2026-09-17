# -*- coding: utf-8 -*-
"""因子诊断 v2（scripts/factor_health/diagnose.py）测试：全部使用合成数据。"""

import numpy as np
import pandas as pd

from scripts.factor_health.diagnose import (
    DiagnoseThresholds,
    build_candidate_list,
    column_year_profile,
    coverage_return_table,
    partial_ic_table,
    summarize_year_profile,
    usage_stability_table,
)


def _write_day(root, date, frame):
    frame.to_parquet(root / f"{date}.parquet", index=False)


def _synthetic_partitions(root, days=40, n=1000, seed=0):
    """构造：label 与 rep 无关；twin = rep（纯复制）；indep = label 的噪声版本。"""
    rng = np.random.default_rng(seed)
    for day in range(days):
        label = rng.normal(size=n)
        rep = rng.normal(size=n)
        frame = pd.DataFrame(
            {
                "label": label,
                "rep": rep,
                "twin": rep + rng.normal(scale=1e-6, size=n),  # 纯复制（单调变换，秩完全相同）
                "indep": label + rng.normal(scale=0.3, size=n),  # 控制 rep 后仍有强残余信号
                "low_cov": np.where(rng.random(n) < 0.5, 1.0, np.nan),
                "size": rng.normal(size=n),
            }
        )
        # 低覆盖列：有值组规模更大且标签明显更高
        has = frame["low_cov"].notna()
        frame.loc[has, "size"] += 1.0
        frame.loc[has, "label"] += 0.3
        _write_day(root, f"2024{day // 20 + 1:02d}{day % 20 + 1:02d}", frame)
    return sorted(root.glob("*.parquet"))


def test_partial_ic_separates_copy_from_independent(tmp_path):
    files = _synthetic_partitions(tmp_path)
    pairs = {"twin": "rep", "indep": "rep"}
    table = partial_ic_table(files, pairs, "label", min_pairs=50)
    # 纯复制（单调变换）：秩残差≈0 -> 增量信息≈0（个别日浮点扰动可忽略）
    assert abs(table.loc["twin", "partial_ic_mean"]) < 0.01
    assert abs(table.loc["twin", "partial_ic_t"]) < 1.5
    assert int(table.loc["twin", "partial_days"]) == 40
    assert table.loc["indep", "partial_ic_t"] > 5.0  # 独立信息 -> 强偏 IC
    assert table.loc["twin", "control"] == "rep"


def test_partial_ic_skips_when_sample_too_small(tmp_path):
    files = _synthetic_partitions(tmp_path, days=10, n=200)
    table = partial_ic_table(files, {"twin": "rep"}, "label", min_pairs=500)
    assert int(table.loc["twin", "partial_days"]) == 0
    assert np.isnan(table.loc["twin", "partial_ic_t"])


def test_coverage_return_detects_diff_and_size_gap(tmp_path):
    files = _synthetic_partitions(tmp_path, days=30)
    table = coverage_return_table(
        files, ["low_cov"], "label", size_column="size", min_group_rows=30
    )
    info = table.loc["low_cov"]
    assert info["diff_mean"] == pytest_approx(0.3, abs=0.05)
    assert info["diff_pos_ratio"] > 0.9
    assert info["size_diff_mean"] == pytest_approx(1.0, abs=0.2)


def pytest_approx(value, abs=1e-6):
    import pytest

    return pytest.approx(value, abs=abs)


class _FakeBooster:
    def __init__(self, scores):
        self._scores = scores

    def get_score(self, importance_type):
        return dict(self._scores)


class _FakeModel:
    feature_names_in_ = ["alpha", "beta"]

    def __init__(self, scores):
        self._scores = scores

    def get_booster(self):
        return _FakeBooster(self._scores)


def test_usage_stability_computes_cv(tmp_path):
    (tmp_path / "v1_model.joblib").write_bytes(b"")
    (tmp_path / "v2_model.joblib").write_bytes(b"")
    scores = {1: {"f0": 9.0, "f1": 1.0}, 2: {"f0": 1.0, "f1": 9.0}}
    table = usage_stability_table(
        tmp_path,
        [1, 2],
        ["alpha", "beta"],
        loader=lambda path: _FakeModel(scores[int(path.stem[1])]),
    )
    assert table.loc["alpha", "gain_mean"] == pytest_approx(0.5, abs=1e-9)
    # 0.9 / 0.1 两个版本 -> 样本标准差 0.4*sqrt(2)，CV = 0.5657/0.5
    assert table.loc["alpha", "gain_cv"] == pytest_approx(0.4 * np.sqrt(2) / 0.5, abs=1e-6)
    assert int(table.loc["beta", "versions"]) == 2


def _write_year(root, year, frame):
    frame.to_parquet(root / f"{year}0615.parquet", index=False)


def _yearly_dataset(root, years=range(2012, 2027), n=400, seed=1):
    """构造含有“源头起点 / 扩张 / 收缩 / 幅度塌缩 / 稳定”五类形态的逐年分区。"""
    rng = np.random.default_rng(seed)
    for year in years:
        frame = pd.DataFrame(
            {
                "stable": rng.normal(size=n),
                "late_start": rng.normal(size=n) if year >= 2018 else np.full(n, np.nan),
                "expanding": np.where(
                    rng.random(n) < min(0.2 + 0.06 * (year - 2012), 0.9), rng.normal(size=n), np.nan
                ),
                "shrinking": np.where(
                    rng.random(n) < max(0.9 - 0.03 * (year - 2012), 0.3), rng.normal(size=n), np.nan
                ),
                "collapsing": rng.normal(scale=1.0 if year <= 2022 else 0.01, size=n),
            }
        )
        _write_year(root, year, frame)
    return sorted((path.stem[:4], path) for path in root.glob("*.parquet"))


def test_column_year_profile_and_root_causes(tmp_path):
    picks = _yearly_dataset(tmp_path)
    profile = column_year_profile(
        picks, ["stable", "late_start", "expanding", "shrinking", "collapsing"]
    )
    assert len(profile) == 5 * len(picks)
    summary = summarize_year_profile(profile, collapse_ratio=0.1).reset_index().set_index("feature")

    assert summary.loc["stable", "根因"] == "正常"
    assert "源头起点" in summary.loc["late_start", "根因"]
    assert summary.loc["late_start", "first_effective_year"] == 2018
    assert "扩张" in summary.loc["expanding", "根因"]
    assert "覆盖收缩" in summary.loc["shrinking", "根因"]
    assert "口径退化" in summary.loc["collapsing", "根因"]
    assert summary.loc["collapsing", "collapse_ratio"] < 0.1


def test_build_candidate_list_flags_collapse():
    register = pd.DataFrame(
        {
            "coverage": [0.99],
            "coverage_min_year": [0.99],
            "ic_t": [3.0],
            "split_use_frac": [1.0],
            "flag_low_coverage": [False],
            "market_level": [False],
        },
        index=["collapsing"],
    )
    year_summary = pd.DataFrame(
        {
            "first_effective_year": [2012],
            "coverage_first": [0.9],
            "coverage_last": [0.9],
            "hist_std_median": [1.0],
            "hist_std_max": [1.0],
            "recent_std_median": [0.02],
            "collapse_ratio": [0.02],
            "根因": ["口径退化（幅度塌缩 0.020×）"],
        },
        index=["collapsing"],
    )
    table = build_candidate_list(register, year_summary=year_summary)
    rows = table[table["候选类型"] == "F 口径退化"]
    assert len(rows) == 1
    assert rows.iloc[0]["因子"] == "collapsing"
    assert "历史峰值" in rows.iloc[0]["证据"]


def test_build_candidate_list_types_and_rubric():
    register = pd.DataFrame(
        {
            "coverage": [0.5, 0.99, 0.99, 0.99],
            "coverage_min_year": [0.4, 0.99, 0.99, 0.99],
            "ic_t": [3.0, 0.5, 9.0, 1.0],
            "split_use_frac": [1.0, 1.0, 0.2, 1.0],
            "flag_low_coverage": [True, False, False, False],
            "market_level": [False] * 4,
        },
        index=["lowcov", "redundant", "unused_signal", "unstable"],
    )
    partial = pd.DataFrame(
        {
            "control": ["rep"] * 2,
            "partial_ic_mean": [0.001, 0.02],
            "partial_ic_t": [0.5, 4.0],
            "partial_days": [300, 300],
        },
        index=["lowcov", "redundant"],
    )
    coverage = pd.DataFrame(
        {
            "days": [300],
            "has_median_mean": [0.001],
            "missing_median_mean": [-0.001],
            "diff_mean": [0.002],
            "diff_pos_ratio": [0.5],
            "size_diff_mean": [0.8],
        },
        index=["lowcov"],
    )
    stability = pd.DataFrame(
        {
            "versions": [40, 40],
            "gain_mean": [0.01, 0.0001],
            "gain_std": [0.02, 0.0001],
            "gain_min": [0.0, 0.0],
            "gain_max": [0.05, 0.001],
            "gain_present_ratio": [1.0, 1.0],
            "gain_cv": [2.0, 0.5],
        },
        index=["unstable", "lowcov"],
    )
    table = build_candidate_list(register, partial, coverage, stability, DiagnoseThresholds())
    kinds = set(table["候选类型"])
    assert "A 覆盖缺口" in kinds  # lowcov
    assert "B 无增量信息" in kinds  # lowcov（偏 IC t=0.5）+ redundant? t=4.0 不入选
    assert "C 有信号未被使用" in kinds  # unused_signal（|t|=9 且使用率 0.2）
    assert "D 使用不稳定" in kinds  # unstable（CV=2.0）
    assert set(table.columns) >= {
        "候选类型",
        "因子",
        "证据",
        "预期效应量级",
        "可检出性(14折/MDE≈5pp)",
        "建议动作",
    }
    assert "redundant" not in table.loc[table["候选类型"] == "B 无增量信息", "因子"].tolist()
    assert (
        table.loc[table["候选类型"] == "A 覆盖缺口", "可检出性(14折/MDE≈5pp)"].iloc[0] == "可检出"
    )
