# -*- coding: utf-8 -*-
"""P6 T3 reviewer3 复核整改回归（R3-01~R3-12，2026-11-06）。"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lazybull.v2.reporting import (  # noqa: E402
    experiment_csv,
    experiment_metrics,
    experiment_projection,
    experiment_stability,
    generate,
    render,
)
from src.lazybull.v2.store.store_identity import canonical_json_sha256  # noqa: E402
from tests.p6_t3_env import (  # noqa: E402
    DEFAULT_STRATEGY,
    arm_from_nav,
    load_arms,
    standard_projection,
    write_legacy_run,
    write_runs_batch,
)


def _splits(arm):
    return sorted(int(item) for item in arm.windows["split_index"].tolist())


def test_r3_01_unknown_data_state_is_not_comparable(tmp_path):
    """双缺失/单缺失/空串都不得输出可比较 Δ；同态正例仍可用（R3-01）。"""
    confirmed = arm_from_nav(tmp_path, arm_id="base", navs=[1.0, 0.9, 1.0])
    same = arm_from_nav(tmp_path, arm_id="same", navs=[1.0, 0.95, 1.0])
    assert experiment_metrics.compare_pair(confirmed, same, _splits(confirmed))["status"][
        "report_status"
    ] == "available"

    for left_state, right_state in ((None, None), (None, "ds000001"), ("", "")):
        left = arm_from_nav(
            tmp_path, arm_id="left", navs=[1.0, 0.9, 1.0], data_state_id=left_state
        )
        right = arm_from_nav(
            tmp_path, arm_id="right", navs=[1.0, 0.95, 1.0], data_state_id=right_state
        )
        result = experiment_metrics.compare_pair(left, right, _splits(left))
        assert result["status"]["report_status"] == "not_available"
        assert result["status"]["degradation_reason_code"] == "not_comparable"
        assert "数据态不可确认" in result["status"]["detail"]
        assert result["summary"] == {}


def test_r3_02_invalid_baseline_never_computes_subset(tmp_path):
    """坏链基线：子集指标不得泄漏，保留自身原因码（R3-02）。"""
    base = arm_from_nav(tmp_path, arm_id="base", navs=[1.0, 0.9, 1.0])
    base.status = {
        "report_status": "invalid",
        "degradation_reason_code": "duplicate_chain_date",
        "detail": "same date",
    }
    arm = arm_from_nav(tmp_path, arm_id="arm", navs=[1.0, 0.95, 1.0])
    result = experiment_metrics.compare_baseline(base, _splits(arm))
    assert result["status"]["report_status"] == "invalid"
    assert result["status"]["degradation_reason_code"] == "duplicate_chain_date"
    assert result["summary"] == {}
    assert result["folds"] == []


def test_r3_03_legacy_state_invalid_utf8_is_visible(tmp_path):
    """无效 UTF-8 数据态：不得逸出 UnicodeDecodeError（R3-03-R1）。"""
    broken = write_legacy_run(tmp_path / "utf8", run_id="wf_utf8")
    (broken / "data_state_wf_utf8.json").write_bytes(b"\xff\xfe\xfd")
    arms = load_arms(tmp_path, [("utf8", "legacy_wf", broken)])
    assert arms[0].status["report_status"] == "invalid"
    assert arms[0].status["degradation_reason_code"] == "invalid_json"


def test_r3_03_legacy_state_candidates_validated(tmp_path):
    """多候选/坏 JSON/根类型错误 ⇒ visible invalid（R3-03）。"""
    good = write_legacy_run(tmp_path / "good", run_id="wf_good")
    extra_a = good / "data_state_a.json"
    extra_z = good / "data_state_z.json"
    extra_a.write_text(json.dumps({"data_state_id": "dsA"}), encoding="utf-8")
    extra_z.write_text(json.dumps({"data_state_id": "dsZ"}), encoding="utf-8")
    arms = load_arms(tmp_path, [("good", "legacy_wf", good)])
    assert arms[0].status["report_status"] == "invalid"
    assert arms[0].status["degradation_reason_code"] == "multiple_source_files"

    bad = write_legacy_run(tmp_path / "bad", run_id="wf_bad")
    (bad / "data_state_wf_bad.json").write_text("{invalid", encoding="utf-8")
    arms = load_arms(tmp_path, [("bad", "legacy_wf", bad)])
    assert arms[0].status["degradation_reason_code"] == "invalid_json"

    root = write_legacy_run(tmp_path / "root", run_id="wf_root")
    (root / "data_state_wf_root.json").write_text("[]", encoding="utf-8")
    arms = load_arms(tmp_path, [("root", "legacy_wf", root)])
    assert arms[0].status["degradation_reason_code"] == "invalid_json"


def test_r3_04_same_window_duplicates_not_two_periods(tmp_path):
    """同窗口两次重跑不得被 batch_id 补成两个期间（R3-04）。"""
    first = write_legacy_run(tmp_path / "p1", run_id="wf_20260101_000000_a", folds=1)
    second = write_legacy_run(tmp_path / "p2", run_id="wf_20260102_000000_b", folds=1)
    for path in (first, second):
        summary = next(path.glob("walk_forward_summary_*.csv"))
        frame = pd.read_csv(summary)
        frame["batch_period_label"] = ""
        frame.to_csv(summary, index=False)
    arms = load_arms(tmp_path, [("p1", "legacy_wf", first), ("p2", "legacy_wf", second)])
    built = experiment_projection.build_experiment_projection(
        baseline="p1",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "p1", "arm_1": "p2"},
    )
    rows = built["projection"]["stability"]["rows"]
    assert len(rows) == 1
    assert int(rows[0]["period_count"]) == 1
    assert rows[0]["report_status"] == "not_available"
    assert rows[0]["degradation_reason_code"] == "insufficient_periods"


def test_r3_05_known_stat_outputs_stay_in_same_group(tmp_path):
    """仅 val_rankic_ir 变化仍同组；新增真实开关必须拆组（R3-05）。"""
    first = write_legacy_run(
        tmp_path / "p1",
        run_id="wf_20260101_000000_a",
        batch_period_label="P1",
    )
    second = write_legacy_run(
        tmp_path / "p2",
        run_id="wf_20260201_000000_b",
        batch_period_label="P2",
    )
    for path, value in ((first, 0.4), (second, 0.5)):
        summary = next(path.glob("walk_forward_summary_*.csv"))
        frame = pd.read_csv(summary)
        frame["val_rankic_ir"] = value
        frame.to_csv(summary, index=False)
    arms = load_arms(tmp_path, [("p1", "legacy_wf", first), ("p2", "legacy_wf", second)])
    built = experiment_projection.build_experiment_projection(
        baseline="p1",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "p1", "arm_1": "p2"},
    )
    rows = built["projection"]["stability"]["rows"]
    assert len(rows) == 1
    assert int(rows[0]["period_count"]) == 2
    assert rows[0]["report_status"] == "available"
    assert experiment_stability.classify_field("val_rankic_ir") == "stat"
    assert experiment_stability.classify_field("daily_rankic_ir") == "stat"
    # 真实策略参数不得被宽前缀误排（R3-05-R1）
    for parameter in ("bt_top_n", "bt_rebalance_freq", "test_window_months", "val_ratio"):
        assert experiment_stability.classify_field(parameter) == "stable", parameter

    switch = write_legacy_run(
        tmp_path / "p3",
        run_id="wf_20260301_000000_c",
        batch_period_label="P3",
    )
    switch_summary = next(switch.glob("walk_forward_summary_*.csv"))
    switch_frame = pd.read_csv(switch_summary)
    switch_frame["enable_lhb_features"] = True
    switch_frame.to_csv(switch_summary, index=False)
    arms = load_arms(
        tmp_path, [("p1", "legacy_wf", first), ("p3", "legacy_wf", switch)]
    )
    built = experiment_projection.build_experiment_projection(
        baseline="p1",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "p1", "arm_1": "p3"},
    )
    assert built["projection"]["stability"]["partitions"] >= 2


def test_r3_06_nonfinite_money_is_not_available(tmp_path):
    """±Inf 与 NaN/None 同等处理：不得当可用金额，也不得让生成链抛 JSON 错误（R3-06-R1）。"""
    for field, filename in (("turnover_amount", "daily.parquet"), ("cost", "trades.parquet")):
        for value in (float("inf"), float("-inf"), None):
            arms, _built = standard_projection(tmp_path / f"{field}_{len(str(value))}_{value}")
            arm = arms[0]
            split = sorted(arm.folds)[0]
            target = getattr(arm.folds[split], "daily" if field == "turnover_amount" else "trades")
            frame = target.copy()
            frame.loc[0, field] = value
            if field == "turnover_amount":
                arm.folds[split].daily = frame
            else:
                arm.folds[split].trades = frame
            cost = experiment_metrics.cost_turnover(arm)
            assert cost["status"]["report_status"] == "degraded", (field, value)
            assert cost["status"]["degradation_reason_code"] == "missing_precondition"
            if field == "turnover_amount":
                assert cost["turnover_total"] is None
            else:
                assert cost["cost_total"] is None


def test_r3_06_unknown_money_is_not_zero(tmp_path):
    """有源行但金额未知：不得补零，状态降级且合计为 None（R3-06）。"""
    arms, built = standard_projection(tmp_path)
    arm = arms[0]
    frame = next(iter(arm.folds.values())).daily
    assert frame is not None
    broken = frame.copy()
    broken["turnover_amount"] = None
    arm.folds[sorted(arm.folds)[0]].daily = broken
    cost = experiment_metrics.cost_turnover(arm)
    assert cost["turnover_total"] is None
    assert cost["status"]["report_status"] == "degraded"
    assert cost["status"]["degradation_reason_code"] == "missing_precondition"
    assert cost["turnover_by_fold_date"] == []


def test_r3_07_default_fold_csv_keeps_all_arms_folds(tmp_path):
    """默认未指定子集：fold_metrics 保留各臂全部折；显式子集仍过滤（R3-07）。"""
    short = write_legacy_run(tmp_path / "short", run_id="wf_short", folds=3)
    long = write_legacy_run(tmp_path / "long", run_id="wf_long", folds=4)
    arms = load_arms(tmp_path, [("base", "legacy_wf", short), ("arm", "legacy_wf", long)])
    built = experiment_projection.build_experiment_projection(
        baseline="base",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "base", "arm_1": "arm"},
    )
    rows = built["csv_rows"]["fold_metrics.csv"]
    assert {row["split_index"] for row in rows if row["arm_id"] == "arm"} == {0, 1, 2, 3}
    assert len(rows) == 7
    assert built["projection"]["splits"] == [0, 1, 2]
    assert built["projection"]["csv_splits"] == [0, 1, 2, 3]
    assert built["csv_expected_rows"]["fold_metrics.csv"] == 7
    experiment_csv.build_csv_bundle(
        built["csv_rows"], expected_rows=built["csv_expected_rows"]
    )

    subset = experiment_projection.build_experiment_projection(
        baseline="base",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "base", "arm_1": "arm"},
        splits=[0, 1],
    )
    assert len(subset["csv_rows"]["fold_metrics.csv"]) == 4
    assert subset["projection"]["csv_splits"] == [0, 1]


def test_r3_08_all_invalid_still_renders_stability_state_row(tmp_path):
    """全部臂无效：稳定性状态行期望行数为 1，报告不得因行数门误杀（R3-08）。"""
    good = write_legacy_run(tmp_path / "good", run_id="wf_good")
    chain = next(good.glob("chain_nav_*.csv"))
    frame = pd.read_csv(chain)
    frame["date"] = 1
    frame.to_csv(chain, index=False)
    arms = load_arms(tmp_path, [("bad", "legacy_wf", good)])
    built = experiment_projection.build_experiment_projection(
        baseline="bad",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "bad"},
    )
    stability = built["projection"]["stability"]
    assert len(stability["rows"]) == 1
    assert stability["expected_row_count"] == 1
    assert stability["rows"][0]["report_status"] == "not_available"
    experiment_csv.build_csv_bundle(
        built["csv_rows"], expected_rows=built["csv_expected_rows"]
    )


def test_r3_09_csv_source_refs_resolve_to_real_files(tmp_path):
    """四份 CSV 的 source_ref 必须能按 root_id 定位真实文件（R3-09）。"""
    arms, built = standard_projection(tmp_path)
    roots = {arm.spec.root_id: arm.spec.path for arm in arms}
    checked = 0
    for name in (
        "arm_metrics.csv",
        "fold_metrics.csv",
        "chain_nav.csv",
        "execution_attribution.csv",
    ):
        for row in built["csv_rows"][name]:
            root_id, relative = str(row["source_ref"]).split(":", 1)
            assert (roots[root_id] / relative).is_file(), row["source_ref"]
            checked += 1
    assert checked > 0


def test_r3_10_generator_record_covers_shared_implementations():
    """生成器身份必须登记决定指标/CSV 字节的共享唯一实现（R3-10）。"""
    record = generate.build_generator_record({"git_commit": "fixed", "git_dirty": True})
    files = dict(record["generator_files"])
    for relative in render.SHARED_GENERATOR_FILES:
        assert relative in files, relative
        assert len(files[relative]) == 64


def test_r3_11_initial_audit_scan_oserror_is_registered_exit(tmp_path):
    """首次审计扫描 OSError：返回 exit=3/stage=audit_scan，不逸出 traceback（R3-11）。"""
    from scripts import p6_report

    audit_out = (
        ROOT
        / "data/reports/p6/validations/research"
        / f"p6_r3_scan_probe_{uuid.uuid4().hex[:8]}"
    )
    args = p6_report.build_parser().parse_args(
        [
            "generate",
            "--kind",
            "experiment",
            "--arm",
            "base=runs:temp",
            "--baseline",
            "base",
            "--audit-out",
            str(audit_out),
        ]
    )
    with patch.object(
        p6_report.publish, "scan_audit_roots", side_effect=OSError("r3_scan_probe")
    ):
        code, payload = p6_report.run_generate(args)
    assert code == 3
    assert payload["verdict"] == "fail"
    assert payload["stage"] == "audit_scan"
    assert "r3_scan_probe" in payload["detail"]


def test_r3_12_config_json_is_full_canonical_identity():
    """config_json 必须能独立复算出 group_key（R3-12）。"""
    rows = experiment_stability.build_stability_rows(
        [
            experiment_stability.StabilityRecord(
                arm_id="a",
                run_id="wf_a",
                config={"learning_rate": 0.03},
                host_mode="backtest",
                implementation="runs",
                data_state_id="ds000001",
                watermarks={"data_state_id": "ds000001", "git_commit": "abc"},
                metrics={
                    "chain_cagr": 0.1,
                    "chain_max_drawdown": -0.1,
                    "chain_sharpe": 1.0,
                    "oos_cross_split_ir": 0.5,
                    "bt_win_rate": 0.5,
                },
                batch_id="a",
                batch_run_id="a",
                batch_period_label="P1",
                split_count=2,
                final_date="20260105",
                model_version="1",
            ),
            experiment_stability.StabilityRecord(
                arm_id="b",
                run_id="wf_b",
                config={"learning_rate": 0.03},
                host_mode="backtest",
                implementation="runs",
                data_state_id="ds000001",
                watermarks={"data_state_id": "ds000001", "git_commit": "abc"},
                metrics={
                    "chain_cagr": 0.2,
                    "chain_max_drawdown": -0.1,
                    "chain_sharpe": 1.0,
                    "oos_cross_split_ir": 0.5,
                    "bt_win_rate": 0.5,
                },
                batch_id="b",
                batch_run_id="b",
                batch_period_label="P2",
                split_count=2,
                final_date="20260105",
                model_version="1",
            ),
        ]
    )
    for row in rows["rows"]:
        assert canonical_json_sha256(json.loads(row["config_json"])) == row["group_key"]


def test_r3_12b_all_invalid_state_row_hash_matches(tmp_path):
    """全无效臂的空输入状态行同样可重算 group_key（R3-12-R1）。"""
    arms, _built = standard_projection(tmp_path)
    for arm in arms:
        arm.status = {
            "report_status": "invalid",
            "degradation_reason_code": "duplicate_chain_date",
        }
        arm.rows_used = 0
    built = experiment_projection.build_experiment_projection(
        baseline=arms[0].spec.arm_id,
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={arm.spec.root_id: arm.spec.arm_id for arm in arms},
    )
    rows = built["projection"]["stability"]["rows"]
    assert len(rows) == 1
    assert canonical_json_sha256(json.loads(rows[0]["config_json"])) == rows[0]["group_key"]


def test_r3_05_daily_rankic_ir_is_known_output(tmp_path):
    """``daily_rankic_ir`` 变化仍同组（生产 test_daily 原键并入，R3-05-R1）。"""
    paths = []
    for number, metric in ((1, 0.4), (2, 0.5)):
        path = write_legacy_run(
            tmp_path / f"d{number}",
            run_id=f"wf_20260{number}01_000000_a",
            batch_period_label=f"P{number}",
        )
        summary = next(path.glob("walk_forward_summary_*.csv"))
        frame = pd.read_csv(summary)
        frame["daily_rankic_ir"] = metric
        frame.to_csv(summary, index=False)
        paths.append((f"d{number}", "legacy_wf", path))
    arms = load_arms(tmp_path, paths)
    built = experiment_projection.build_experiment_projection(
        baseline="d1",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "d1", "arm_1": "d2"},
    )
    rows = built["projection"]["stability"]["rows"]
    assert len(rows) == 1
    assert int(rows[0]["period_count"]) == 2
    assert rows[0]["report_status"] == "available"


def test_r3_05_top_n_is_strategy_identity(tmp_path):
    """TopN 是策略参数：TopN30 属独立分区，增删它不得改变 TopN20 目标得分（R3-05-R1）。"""
    paths = []
    for name, top_n, step in (("base", 20, 0.004), ("same", 20, 0.005), ("other", 30, 0.02)):
        path = write_runs_batch(
            tmp_path / name,
            batch_id=name,
            config={**DEFAULT_STRATEGY, "bt_top_n": top_n, "batch_period_label": name},
            nav_step=step,
        )
        paths.append((name, "runs", path))
    arms = load_arms(tmp_path, paths)
    before = experiment_projection.build_experiment_projection(
        baseline="base",
        arms=arms[:2],
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={arm.spec.root_id: arm.spec.arm_id for arm in arms[:2]},
    )
    after = experiment_projection.build_experiment_projection(
        baseline="base",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={arm.spec.root_id: arm.spec.arm_id for arm in arms},
    )
    before_rows = {row["group_key"]: row for row in before["projection"]["stability"]["rows"]}
    after_rows = {row["group_key"]: row for row in after["projection"]["stability"]["rows"]}
    shared = set(before_rows) & set(after_rows)
    assert shared, (before_rows, after_rows)
    for key in shared:
        for field in ("stability_score", "score_mean", "partition_arm_count"):
            assert before_rows[key][field] == after_rows[key][field], field


def test_r3_05_top300_output_is_known(tmp_path):
    """生产默认 K=300 的 mean/std 必须归统计输出，不拆组（R3-05-R2）。"""
    for name, value in (("top300_return_mean", 0.4), ("top300_return_std", 0.1)):
        directory = tmp_path / name
        paths = []
        for number, metric in ((1, value), (2, value + 0.1)):
            path = write_legacy_run(
                directory / f"p{number}",
                run_id=f"wf_20260{number}01_000000_a",
                batch_period_label=f"P{number}",
            )
            summary = next(path.glob("walk_forward_summary_*.csv"))
            frame = pd.read_csv(summary)
            frame[name] = metric
            frame.to_csv(summary, index=False)
            paths.append((f"p{number}", "legacy_wf", path))
        assert experiment_stability.classify_field(name) == "stat"
        arms = load_arms(directory, paths)
        built = experiment_projection.build_experiment_projection(
            baseline="p1",
            arms=arms,
            generated_at="20261106T000000Z",
            evidence_class="research",
            root_map={"arm_0": "p1", "arm_1": "p2"},
        )
        rows = built["projection"]["stability"]["rows"]
        assert len(rows) == 1, name
        assert int(rows[0]["period_count"]) == 2, name
        assert rows[0]["report_status"] == "available", name


def test_r3_06b_aggregate_overflow_is_degraded(tmp_path):
    """逐行有限但聚合溢出的成交金额：不得输出 Inf（R3-06-R2）。"""
    arms, _built = standard_projection(tmp_path)
    arm = arms[0]
    for fold in arm.folds.values():
        if fold.daily is None:
            continue
        frame = fold.daily.copy()
        frame["turnover_amount"] = 1e308
        fold.daily = frame
    cost = experiment_metrics.cost_turnover(arm)
    assert cost["status"]["report_status"] == "degraded"
    assert cost["status"]["degradation_reason_code"] == "missing_precondition"
    assert cost["turnover_total"] is None
    assert cost["turnover_by_fold"] == []
    built = experiment_projection.build_experiment_projection(
        baseline=arm.spec.arm_id,
        arms=[arm],
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={arm.spec.root_id: arm.spec.arm_id},
    )
    json.dumps(built["projection"], ensure_ascii=False, allow_nan=False)


def test_r3_12c_missing_precondition_api_identity(tmp_path):
    """显式缺列 frame 的 missing_precondition 状态行同样可重算 group_key（R3-12-R2）。"""
    record = experiment_stability.StabilityRecord(
        arm_id="a",
        run_id="wf_a",
        config={"learning_rate": 0.03},
        host_mode="backtest",
        implementation="runs",
        data_state_id="ds000001",
        watermarks={"data_state_id": "ds000001"},
        metrics={},
        batch_id="a",
        batch_run_id="a",
        batch_period_label="P1",
        split_count=2,
        final_date="20260105",
        model_version="1",
    )
    frame = experiment_stability.build_stability_input([record], with_composite=False)
    result = experiment_stability.build_stability_rows([record], frame=frame)
    assert result["rows"]
    for row in result["rows"]:
        assert row["degradation_reason_code"] == "missing_precondition"
        assert canonical_json_sha256(json.loads(row["config_json"])) == row["group_key"]


def test_r3_09_padded_split_directories_resolve(tmp_path):
    """生产补零 ``folds/split00`` 布局：source_ref 必须指向实际目录（R3-09-R1）。"""
    source = tmp_path / "runs"
    write_runs_batch(source, batch_id="padded")
    old = source / "folds" / "split0"
    new = source / "folds" / "split00"
    old.rename(new)
    arms = load_arms(tmp_path, [("base", "runs", source)])
    built = experiment_projection.build_experiment_projection(
        baseline="base",
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={"arm_0": "base"},
    )
    roots = {arm.spec.root_id: arm.spec.path for arm in arms}
    for row in built["csv_rows"]["execution_attribution.csv"]:
        root_id, relative = str(row["source_ref"]).split(":", 1)
        if relative.startswith("folds/split00/"):
            assert (roots[root_id] / relative).is_file()
    # 第 1 折仍是未补零目录：引用必须与捕获计划一致
    assert any(
        "folds/split1/" in row["source_ref"]
        for row in built["csv_rows"]["execution_attribution.csv"]
    )
