# -*- coding: utf-8 -*-
"""P6 T3 链式指标与 Δ 方向（P6-T3-01~05）。

判据（计划 §3-T3 / 设计 §4.2、§5.3.1）：

- MaxDD 为非正小数，``ΔMaxDD = arm - base``，正值表示改善（三例方向必须正确）；
- compare_runs 中文输出按**显式映射**转 P6 列，禁止按英文同名键猜测；
- 跨实现（runs vs 旧 WF）只允许对账陈列，不产生 Δ。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import experiment_metrics, experiment_projection  # noqa: E402
from tests.p6_t3_env import arm_from_folds, arm_from_nav  # noqa: E402


def _pair(tmp_path, base_navs, arm_navs):
    base = arm_from_nav(tmp_path, arm_id="base", navs=base_navs)
    arm = arm_from_nav(tmp_path, arm_id="arm", navs=arm_navs)
    return base, arm


def _delta(tmp_path, base_navs, arm_navs):
    base, arm = _pair(tmp_path, base_navs, arm_navs)
    splits = sorted(int(item) for item in base.windows["split_index"].tolist())
    result = experiment_metrics.compare_pair(base, arm, splits)
    assert result["status"]["report_status"] == "available", result["status"]
    return result["summary"]


def test_p6_t3_01_maxdd_improvement_is_positive(tmp_path):
    """基线 -0.20、臂 -0.10 ⇒ ΔMaxDD = +0.10（改善为正值）。"""
    summary = _delta(tmp_path, [1.0, 0.8, 1.0], [1.0, 0.9, 1.0])
    assert summary["subset_max_drawdown"] == pytest.approx(-0.10)
    assert summary["delta_max_drawdown"] == pytest.approx(0.10)
    assert summary["delta_max_drawdown"] > 0


def test_p6_t3_02_maxdd_worse_is_negative(tmp_path):
    """基线 -0.20、臂 -0.25 ⇒ ΔMaxDD = -0.05（恶化）。"""
    summary = _delta(tmp_path, [1.0, 0.8, 1.0], [1.0, 0.75, 1.0])
    assert summary["subset_max_drawdown"] == pytest.approx(-0.25)
    assert summary["delta_max_drawdown"] == pytest.approx(-0.05)
    assert summary["delta_max_drawdown"] < 0


def test_p6_t3_03_maxdd_equal_is_zero(tmp_path):
    """同回撤 ⇒ ΔMaxDD = 0（既不改善也不恶化）。"""
    summary = _delta(tmp_path, [1.0, 0.8, 1.0], [1.0, 0.8, 1.2])
    assert summary["delta_max_drawdown"] == pytest.approx(0.0, abs=1e-12)


def test_p6_t3_04_compare_mapping_is_explicit_chinese(tmp_path):
    """中文 → 英文映射：源列名必须逐字命中，且实现映射与 T0.5 冻结一致。"""
    assert experiment_metrics.check_compare_mapping() == []
    base, arm = _pair(tmp_path, [1.0, 0.8, 1.0], [1.0, 0.9, 1.0])
    splits = sorted(int(item) for item in base.windows["split_index"].tolist())
    from scripts.compare.fold_subset import compare_runs

    raw = compare_runs(base.artifacts, [arm.artifacts], splits)["summary"]
    # 源表只有中文列；若按英文同名键猜测会全部落空（这正是禁止的路径）
    assert "subset_cagr" not in raw.columns and "delta_cagr" not in raw.columns
    for source in experiment_metrics.COMPARE_SUMMARY_MAP:
        assert source in raw.columns, f"映射源列缺席：{source}"
    mapped = experiment_metrics.compare_pair(base, arm, splits)["summary"]
    for target in experiment_metrics.COMPARE_SUMMARY_MAP.values():
        assert target in mapped
    row = raw[raw["运行"] == "arm"].iloc[0]
    assert mapped["subset_cagr"] == pytest.approx(row["子集CAGR"])
    assert mapped["delta_cagr"] == pytest.approx(row["ΔCAGR(vs基线)"])
    assert mapped["fold_return_aligned_count"] == int(row["折收益同向(vs基线)"])


def test_p6_t3_25_splits_subset_uses_unique_implementation(tmp_path):
    """``--splits`` 子集：基线与对照臂都走 ``compare_runs``，子集交易日数 = N−1（A-02）。"""
    base = arm_from_folds(tmp_path, arm_id="base", folds=3, rows_per_fold=6)
    arm = arm_from_folds(tmp_path, arm_id="arm", folds=3, rows_per_fold=6, nav_dip=0.05)
    built = experiment_projection.build_experiment_projection(
        baseline="base",
        arms=[base, arm],
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "base", "arm_1": "arm"},
        splits=[0, 1],
    )
    rows = {row["arm_id"]: row for row in built["csv_rows"]["arm_metrics.csv"]}
    subset_chain = base.chain[base.chain["split_index"].isin([0, 1])]
    expected_days = len(subset_chain) - 2
    assert rows["base"]["subset_cagr"] is not None
    assert rows["base"]["fold_count"] == 2
    assert rows["arm"]["delta_cagr"] is not None
    assert rows["arm"]["delta_max_drawdown"] == pytest.approx(
        rows["arm"]["subset_max_drawdown"] - rows["base"]["subset_max_drawdown"]
    )
    # 子集指标不得等于全周期指标（3 折链 ≠ 2 折子集）
    base_metrics = experiment_metrics.arm_metrics(base)["metrics"]
    assert rows["base"]["subset_cagr"] != pytest.approx(base_metrics["cagr"], abs=1e-12)
    from scripts.compare.fold_subset import subset_metrics

    reference = subset_metrics(base.artifacts, [0, 1])
    assert rows["base"]["subset_cagr"] == pytest.approx(reference["子集CAGR"])
    assert reference["子集交易日数"] == expected_days
    # 逐折收益（含基线）来自唯一实现
    from scripts.compare.fold_subset import fold_table

    table = fold_table(base.artifacts, [0, 1]).set_index("折序号")["折收益"]
    for row in built["csv_rows"]["fold_metrics.csv"]:
        if row["arm_id"] == "base":
            assert row["fold_return"] == pytest.approx(table[int(row["split_index"])])


def test_p6_t3_29_fold_missing_has_own_reason(tmp_path):
    """逐折收益/回撤缺失必须给本行专用原因码（不借用 TopK 原因，A-03）。"""
    from src.lazybull.v2.reporting import experiment_projection

    base = arm_from_folds(tmp_path, arm_id="base", folds=2, rows_per_fold=4)
    entry = experiment_projection._arm_entry(base, base, [0, 1])
    broken = [dict(item) for item in entry["compare"]["folds"]]
    broken[0]["fold_return"] = None
    entry["compare"]["folds"] = broken
    rows = experiment_projection._fold_rows(base, entry, [0, 1])
    missing = [row for row in rows if row["split_index"] == 0][0]
    assert missing["report_status"] == "degraded"
    assert missing["degradation_reason_code"] == "no_source"


def test_p6_t3_26_compare_mapping_covers_real_columns_exactly(tmp_path):
    """映射键集与 ``compare_runs`` 真实中文列集**完全相等**（A-08 的机器核对）。"""
    base, arm = _pair(tmp_path, [1.0, 0.9, 1.0], [1.0, 0.95, 1.0])
    splits = sorted(int(item) for item in base.windows["split_index"].tolist())
    from scripts.compare.fold_subset import compare_runs

    raw = compare_runs(base.artifacts, [arm.artifacts], splits)["summary"]
    assert set(experiment_metrics.COMPARE_SUMMARY_MAP) == set(raw.columns)


def test_p6_t3_05_cross_implementation_reconciles_only(tmp_path):
    """跨实现只对账：两侧指标并列相等，但不产 Δ、状态固定不可比。"""
    base = arm_from_nav(tmp_path, arm_id="base", navs=[1.0, 0.9, 1.0])
    legacy = arm_from_nav(
        tmp_path, arm_id="legacy", navs=[1.0, 0.9, 1.0], implementation="legacy_wf"
    )
    reconciliation = experiment_metrics.reconcile_implementation(base, legacy)
    assert all(item["equal"] for item in reconciliation["comparisons"])
    assert reconciliation["status"]["report_status"] == "not_available"
    assert reconciliation["status"]["degradation_reason_code"] == "not_comparable"
    splits = sorted(int(item) for item in base.windows["split_index"].tolist())
    paired = experiment_metrics.compare_pair(base, legacy, splits)
    assert paired["status"]["degradation_reason_code"] == "not_comparable"
    assert paired["summary"] == {}
    # 投影层同样不产生差值，并在降级清单可见
    built = experiment_projection.build_experiment_projection(
        baseline="base",
        arms=[base, legacy],
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "base", "arm_1": "legacy"},
    )
    legacy_entry = [arm for arm in built["projection"]["arms"] if arm["identity"]["arm_id"] == "legacy"][0]
    assert legacy_entry["compare"]["summary"] == {}
    paths = {row["field_path"] for row in built["projection"]["degradations"]}
    assert "arms.legacy.compare" in paths
