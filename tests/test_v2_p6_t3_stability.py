# -*- coding: utf-8 -*-
"""P6 T3 稳定性完整等价类（设计 §5.3.2 七条判据；P6-T3-12/-15）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import experiment_projection, experiment_stability  # noqa: E402
from tests.p6_t3_env import DEFAULT_STRATEGY, load_arms, write_legacy_run  # noqa: E402

WATERMARKS = {
    "data_state_id": "ds000001",
    "git_commit": "abc1234",
    "git_dirty": False,
    "data_daily_latest": "2026-07-31",
    "data_cs_train_latest": "20260702",
    "data_dividend_coverage": {"data": 10, "empty": 0, "failed": 0, "pending": 0, "total": 10},
}


def _record(
    arm_id: str,
    *,
    run_id: str | None = None,
    config: dict | None = None,
    host_mode: str = "backtest",
    implementation: str = "runs",
    data_state_id: str = "ds000001",
    git_commit: str = "abc1234",
    batch_run_id: str | None = None,
    batch_period_label: str | None = None,
    score: float = 0.1,
) -> experiment_stability.StabilityRecord:
    watermarks = dict(WATERMARKS)
    watermarks["data_state_id"] = data_state_id
    watermarks["git_commit"] = git_commit
    return experiment_stability.StabilityRecord(
        arm_id=arm_id,
        run_id=run_id or f"wf_{arm_id}",
        config=dict(DEFAULT_STRATEGY if config is None else config),
        host_mode=host_mode,
        implementation=implementation,
        data_state_id=data_state_id,
        watermarks=watermarks,
        metrics={
            "chain_cagr": score,
            "chain_max_drawdown": -0.1,
            "chain_sharpe": 1.0,
            "oos_cross_split_ir": 0.5,
            "bt_win_rate": 0.5,
        },
        batch_id=arm_id,
        batch_run_id=batch_run_id or arm_id,
        batch_period_label=batch_period_label or arm_id,
        split_count=3,
        final_date="20260105",
        model_version="1000",
    )


def test_p6_t3_11_single_classification_and_identity_scope():
    """单一分类：动态/统计互斥，未知键默认 stable；stable_identity 排除动态但保留开关与选股域。"""
    assert experiment_stability.DYNAMIC_EXACT.isdisjoint(experiment_stability.STAT_OUTPUT)
    assert experiment_stability.classify_field("batch_run_id") == "dynamic"
    assert experiment_stability.classify_field("批次ID") == "dynamic"
    assert experiment_stability.classify_field("KEY_Top20_hit_rate") == "stat"
    assert experiment_stability.classify_field("diagnostic_Top30_逐日均值_50分位") == "stat"
    assert experiment_stability.classify_field("bt_total_return") == "stat"
    assert experiment_stability.classify_field("enable_lhb_features") == "stable"
    assert experiment_stability.classify_field("stock_domain") == "stable"
    assert experiment_stability.classify_field("brand_new_switch") == "stable"
    record = _record(
        "a",
        config={**DEFAULT_STRATEGY, "batch_run_id": "b1", "registered_at": "2026-01-01"},
    )
    identity = experiment_stability.stable_identity_of(record)
    config = identity["config_without_dynamic"]
    assert "batch_run_id" not in config and "registered_at" not in config
    assert config["stock_domain"] == "all_a"
    assert config["enable_fundamental"] is True
    for name in experiment_stability.STABLE_IDENTITY_FIELDS:
        assert name in identity


def test_p6_t3_12_legacy_batch_run_id_changes_still_period_count_two(tmp_path):
    """合法跨期正例：batch_id/batch_run_id/折/模型版本/窗口变化仍 period_count≥2。"""
    first = write_legacy_run(
        tmp_path / "legacy_p1",
        run_id="wf_20260101_000000_a",
        batch_run_id="batch_p1",
        batch_period_label="P1",
        model_base=1000,
    )
    second = write_legacy_run(
        tmp_path / "legacy_p2",
        run_id="wf_20260201_000000_b",
        batch_run_id="batch_p2",
        batch_period_label="P2",
        model_base=2000,
    )
    arms = load_arms(tmp_path, [("p1", "legacy_wf", first), ("p2", "legacy_wf", second)])
    built = experiment_projection.build_experiment_projection(
        baseline="p1",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "p1", "arm_1": "p2"},
    )
    stability = built["projection"]["stability"]
    legal = [row for row in stability["rows"] if row["report_status"] == "available"]
    assert len(legal) == 1, stability["rows"]
    assert int(legal[0]["period_count"]) >= 2
    assert "批次ID" in stability["dynamic_columns"]
    assert legal[0]["stability_score"] is not None
    run_ids = str(legal[0]["run_id_list"])
    assert "wf_20260101_000000_a" in run_ids and "wf_20260201_000000_b" in run_ids
    # batch_run_id 变化被投影到统一占位（不二次拆组），原值在臂身份中完整保留
    group_values = str(legal[0]["group_values"])
    assert experiment_stability.DYNAMIC_PLACEHOLDER in group_values
    batch_ids = {entry["identity"]["batch_id"] for entry in built["projection"]["arms"]}
    assert {"batch_p1", "batch_p2"} <= batch_ids


def test_p6_t3_13_precondition_and_insufficient_periods_state_rows():
    """缺前置禁止打分；组内综合得分不足两条有限值同样降级（不产生数值 stability_score）。"""
    records = [_record("a"), _record("b")]
    frame = experiment_stability.build_stability_input(records, with_composite=False)
    missing = experiment_stability.build_stability_rows(records, frame=frame)
    assert missing["rows"]
    assert all(row["report_status"] == "not_available" for row in missing["rows"])
    assert all(row["degradation_reason_code"] == "missing_precondition" for row in missing["rows"])
    assert all(row["stability_score"] is None for row in missing["rows"])
    # 缺指标列：同样 missing_precondition，且不调用旧函数（无合法行）
    without_metric = experiment_stability.build_stability_input(records)
    without_metric = without_metric.drop(columns=["全周期CAGR"])
    blocked = experiment_stability.build_stability_rows(records, frame=without_metric)
    assert all(row["degradation_reason_code"] == "missing_precondition" for row in blocked["rows"])
    # 有限值不足两条：降级为 insufficient_periods，且不保留函数给出的乐观得分
    weak = experiment_stability.build_stability_input(records)
    weak.loc[0, "综合得分"] = np.nan
    degraded = experiment_stability.build_stability_rows(records, frame=weak)
    assert len(degraded["rows"]) == 1
    assert degraded["rows"][0]["degradation_reason_code"] == "insufficient_periods"
    assert degraded["rows"][0]["stability_score"] is None


def test_p6_t3_14_cross_domain_switch_host_code_state_not_grouped():
    """仅 stock_domain / 新增开关 / host / 代码态不同，一律不得并组或跨组去重。"""
    records = [
        _record("a"),
        _record("b"),
        _record("domain", config={**DEFAULT_STRATEGY, "stock_domain": "zz500"}),
        _record("switch", config={**DEFAULT_STRATEGY, "enable_lhb_features": True}),
        _record("host", host_mode="paper_replay"),
        _record("code", git_commit="def5678"),
    ]
    result = experiment_stability.build_stability_rows(records)
    keys = {row["group_key"] for row in result["rows"]}
    assert len(keys) == 5, result["rows"]
    periods = sorted(int(row["period_count"]) for row in result["rows"])
    assert periods == [1, 1, 1, 1, 2]
    merged = [row for row in result["rows"] if int(row["period_count"]) >= 2]
    assert len(merged) == 1
    statuses = {row["degradation_reason_code"] for row in result["rows"]}
    assert "insufficient_periods" in statuses


def test_p6_t3_27_scores_invariant_to_unrelated_arms():
    """等价类得分为**分区内**百分位：增删无关臂不得改变目标 group_key 的数值（A-06）。"""
    base_records = [_record("a", score=0.10), _record("b", score=0.20)]
    baseline = experiment_stability.build_stability_rows(base_records)
    baseline_rows = {
        row["group_key"]: row for row in baseline["rows"] if row["report_status"] == "available"
    }
    assert len(baseline_rows) == 1
    extra_records = base_records + [
        _record("domain", config={**DEFAULT_STRATEGY, "stock_domain": "zz500"}, score=0.9),
        _record("switch", config={**DEFAULT_STRATEGY, "enable_lhb_features": True}, score=0.5),
    ]
    extended = experiment_stability.build_stability_rows(extra_records)
    extended_rows = {
        row["group_key"]: row for row in extended["rows"] if row["report_status"] == "available"
    }
    shared = set(baseline_rows) & set(extended_rows)
    assert shared, "目标等价类在两次运行中必须存在"
    for key in shared:
        for field in (
            "stability_score",
            "score_mean",
            "score_std",
            "score_min",
            "score_max",
            "chain_cagr_mean",
            "chain_cagr_std",
            "chain_max_drawdown_mean",
            "oos_cross_split_ir_mean",
            "bt_win_rate_mean",
        ):
            assert baseline_rows[key][field] == extended_rows[key][field], field


def test_p6_t3_15_no_composite_means_no_numeric_score():
    """删除综合得分后：禁止出现数值 stability_score（P6-T3-15）。"""
    records = [_record("a"), _record("b")]
    frame = experiment_stability.build_stability_input(records, with_composite=False)
    result = experiment_stability.build_stability_rows(records, frame=frame)
    for row in result["rows"]:
        assert row["stability_score"] is None
        assert row["report_status"] != "available"
    assert all(
        value is None
        for row in result["rows"]
        for key, value in row.items()
        if key in ("score_mean", "score_std", "score_min", "score_max")
    )
