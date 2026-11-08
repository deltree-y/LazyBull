# -*- coding: utf-8 -*-
"""P6 T3 缺失/坏文件（计划 §3-T3：必需缺失 invalid；可选项局部降级；多候选拒绝择一）。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.lazybull.v2.reporting import (  # noqa: E402
    experiment_metrics,
    experiment_projection,
    experiment_sources,
    sources,
)
from tests.p6_t3_env import (  # noqa: E402
    arm_from_nav,
    fold_nav_series,
    load_arms,
    write_legacy_run,
    write_runs_batch,
)


def _project(tmp_path, mapping):
    arms = load_arms(tmp_path, mapping)
    built = experiment_projection.build_experiment_projection(
        baseline=mapping[0][0],
        arms=arms,
        generated_at="20261106T000000Z",
        evidence_class="research",
        root_map={f"arm_{index}": item[0] for index, item in enumerate(mapping)},
    )
    return arms, built


def test_p6_t3_20_missing_sources_are_visible(tmp_path):
    """attribution 缺失 => 局部降级；trades/chain 缺失或多候选 => 该臂 invalid。"""
    no_attr = write_runs_batch(
        tmp_path / "runs_no_attr", batch_id="batch_no_attr", with_attribution=False
    )
    no_trades = write_runs_batch(
        tmp_path / "runs_no_trades", batch_id="batch_no_trades", with_trades=False
    )
    legacy = write_legacy_run(tmp_path / "legacy_a", run_id="wf_20260101_000000_a")
    arms, built = _project(
        tmp_path,
        [
            ("no_attr", "runs", no_attr),
            ("no_trades", "runs", no_trades),
            ("legacy", "legacy_wf", legacy),
        ],
    )
    by_id = {arm.spec.arm_id: arm for arm in arms}
    assert by_id["no_attr"].status["report_status"] == "degraded"
    assert by_id["no_attr"].status["degradation_reason_code"] == "fold_file_missing"
    assert by_id["no_trades"].status["report_status"] == "invalid"
    assert by_id["no_trades"].status["degradation_reason_code"] == "required_source_missing"
    # 归因缺失不发明行：该臂在 CSV 中零行，但降级项在 manifest 可见
    attr_rows = built["csv_rows"]["execution_attribution.csv"]
    assert all(row["arm_id"] != "no_attr" for row in attr_rows)
    paths = {row["field_path"] for row in built["projection"]["degradations"]}
    assert any(path.startswith("arms.no_attr.folds[0].attribution") for path in paths)
    assert "arms.no_trades" in paths


def _chain_frame():
    rows, _windows = fold_nav_series(folds=2, rows_per_fold=4)
    return pd.DataFrame(
        {
            "split_index": [item[0] for item in rows],
            "date": [item[1] for item in rows],
            "nav": [item[2] for item in rows],
        }
    )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda frame: frame.iloc[0:0], "chain_rows_empty"),
        (lambda frame: frame.assign(nav=frame["nav"].where(frame.index != 2, np.nan)),
         "nonfinite_chain_nav"),
        (lambda frame: frame.assign(nav=frame["nav"].where(frame.index != 2, np.inf)),
         "nonfinite_chain_nav"),
        (lambda frame: frame.assign(nav=frame["nav"].where(frame.index != 2, 0.0)),
         "nonfinite_chain_nav"),
        (lambda frame: frame.assign(date=frame["date"].where(frame.index != 2, "not-a-date")),
         "chain_rows_unparsable"),
        (
            lambda frame: pd.concat(
                [frame.iloc[:2], frame.iloc[[1]], frame.iloc[2:]], ignore_index=True
            ),
            "duplicate_chain_date",
        ),
    ],
)
def test_p6_t3_22_t3_and_t1_chain_gate_are_identical(mutate, reason):
    """链行门唯一实现：T1 与 T3 对同一输入给出同一 (状态, 原因码)（A-04）。"""
    frame = mutate(_chain_frame())
    t1 = sources.verify_chain_rows(frame, allow_ordinal_date=True)
    t3 = experiment_sources.validate_chain_rows(frame)
    assert (t1["report_status"], t1["degradation_reason_code"]) == (
        t3["report_status"],
        t3["degradation_reason_code"],
    ), (t1, t3)
    assert t3["degradation_reason_code"] == reason
    if reason != "chain_rows_empty":
        assert t3["report_status"] == "invalid"
        assert "metrics" not in t3 or t3.get("metrics") in (None, {})


def test_p6_t3_23_missing_date_column_and_arm_reason_preserved(tmp_path):
    """缺 date 列 ⇒ missing_precondition；臂级具体原因码不得被改写（A-12/A-05）。"""
    frame = _chain_frame().drop(columns=["date"])
    assert experiment_sources.validate_chain_rows(frame)["degradation_reason_code"] == (
        "missing_precondition"
    )
    arm = arm_from_nav(tmp_path, arm_id="a", navs=[1.0, 1.01, 1.02, 1.03])
    arm.chain.loc[2, "nav"] = float("nan")
    arm.status = experiment_sources.validate_chain_rows(arm.chain)
    metrics = experiment_metrics.arm_metrics(arm)
    assert metrics["status"]["degradation_reason_code"] == "nonfinite_chain_nav"
    assert metrics["metrics"] == {}


def test_p6_t3_24_legacy_topk_all_nan_is_degraded(tmp_path):
    """旧 WF ``KEY_*`` 列存在但全空 ⇒ degraded/no_source（A-11）。"""
    arm = arm_from_nav(tmp_path, arm_id="legacy", navs=[1.0, 1.01, 1.02], topk=True)
    arm.implementation = "legacy_wf"
    arm.summary[["KEY_Top20_hit_rate", "KEY_Top30_hit_rate"]] = np.nan
    topk = experiment_metrics.topk_values(arm)
    assert topk["status"]["report_status"] == "degraded"
    assert topk["status"]["degradation_reason_code"] == "no_source"
    assert all(value is None for value in topk["means"].values())


def test_p6_t3_21_legacy_bad_or_multiple_files_not_silently_picked(tmp_path):
    """旧 WF：缺 chain/chain 损坏/多 chain 候选都必须 visible invalid（不按最新择一）。"""
    missing = tmp_path / "legacy_missing"
    missing.mkdir()
    write_legacy_run(missing, run_id="wf_20260101_000000_x")
    (missing / "chain_nav_wf_20260101_000000_x.csv").unlink()

    broken = tmp_path / "legacy_broken"
    write_legacy_run(broken, run_id="wf_20260101_000000_y")
    (broken / "chain_nav_wf_20260101_000000_y.csv").write_text("foo\n1\n", encoding="utf-8")

    multiple = tmp_path / "legacy_multiple"
    write_legacy_run(multiple, run_id="wf_20260101_000000_z")
    shutil.copy(
        multiple / "chain_nav_wf_20260101_000000_z.csv",
        multiple / "chain_nav_wf_20260102_000000_z2.csv",
    )

    arms, _built = _project(
        tmp_path,
        [
            ("missing", "legacy_wf", missing),
            ("broken", "legacy_wf", broken),
            ("multiple", "legacy_wf", multiple),
        ],
    )
    reasons = {arm.spec.arm_id: arm.status["degradation_reason_code"] for arm in arms}
    assert reasons["missing"] == "required_source_missing"
    assert reasons["broken"] == "source_contract_violation"
    assert reasons["multiple"] == "multiple_source_files"
    assert all(arm.status["report_status"] == "invalid" for arm in arms)
