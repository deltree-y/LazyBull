# -*- coding: utf-8 -*-
"""折子集链式对比（scripts/compare/fold_subset.py）测试。

全部使用合成 chain_nav / summary 产物，不依赖真实 walk-forward 运行。
"""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.compare.fold_subset import (
    RunArtifacts,
    compare_runs,
    fold_table,
    full_metrics,
    load_run,
    parse_split_spec,
    subset_metrics,
    validate_alignment,
)


def _write_run(
    root,
    label,
    splits=(0, 1, 2),
    days_per_split=5,
    daily_returns=None,
    windows=None,
    data_state_id="abc12345",
):
    """构造一个合成的 walk-forward 运行目录。"""
    raw_dir = root / label / "raw"
    raw_dir.mkdir(parents=True)
    rows = []
    nav = 1.0
    for split in splits:
        returns = (
            daily_returns[split]
            if daily_returns is not None
            else [0.01 if split % 2 == 0 else -0.005] * days_per_split
        )
        for index, ret in enumerate(returns):
            if index == 0:
                rows.append({"date": 0, "nav": nav, "split_index": split})
                continue
            nav *= 1 + ret
            rows.append({"date": index, "nav": nav, "split_index": split})
    pd.DataFrame(rows).to_csv(raw_dir / f"chain_nav_wf_{label}.csv", index=False)

    window_rows = [
        {
            "split_index": split,
            "test_start": (windows or {}).get(split, ("20210101", "20210601"))[0],
            "test_end": (windows or {}).get(split, ("20210101", "20210601"))[1],
        }
        for split in splits
    ]
    pd.DataFrame(window_rows).to_csv(raw_dir / f"walk_forward_summary_{label}.csv", index=False)
    if data_state_id is not None:
        (raw_dir / f"data_state_{label}.json").write_text(
            json.dumps({"data_state_id": data_state_id}), encoding="utf-8"
        )
    return raw_dir


def test_parse_split_spec_supports_ranges():
    assert parse_split_spec("8-13") == [8, 9, 10, 11, 12, 13]
    assert parse_split_spec("v0, 2-3") == [0, 2, 3]
    with pytest.raises(ValueError):
        parse_split_spec("5-3")


def test_load_run_reads_chain_windows_and_data_state(tmp_path):
    _write_run(tmp_path, "base")
    run = load_run(tmp_path / "base")
    assert set(run.chain.columns) >= {"nav", "split_index"}
    assert run.windows["split_index"].tolist() == [0, 1, 2]
    assert run.data_state_id == "abc12345"
    assert run.chain["split_index"].tolist() == [0] * 5 + [1] * 5 + [2] * 5


def test_subset_metrics_renormalizes_and_matches_manual(tmp_path):
    _write_run(tmp_path, "base")
    run = load_run(tmp_path / "base")
    metrics = subset_metrics(run, [2])
    # 折 2 每日 +1%：5 行中前 4 次收益 +1%（首行为起点）
    expected = 1.01**4 - 1
    assert metrics["子集总收益"] == pytest.approx(expected, rel=1e-9)
    assert metrics["子集交易日数"] == 4
    assert metrics["子集最大回撤"] == pytest.approx(0.0, abs=1e-12)

    full = full_metrics(run)
    assert full["全周期交易日数"] == 12  # 3 折 × 4 个收益区间
    assert full["全周期总收益"] > metrics["子集总收益"]  # 全周期含多为正收益的折 0


def test_subset_metrics_ignores_cross_split_return(tmp_path):
    """跨折边界不应被当作收益：折 0 末净值 1.01 → 折 1 首行仍为 1.01（不产生跳变）。"""
    _write_run(tmp_path, "base")
    run = load_run(tmp_path / "base")
    chain = run.chain
    boundary = chain[(chain["split_index"] == 1)].index[0]
    assert chain.loc[boundary, "nav"] == pytest.approx(chain.loc[boundary - 1, "nav"], rel=1e-12)
    metrics = subset_metrics(run, [0, 1])
    assert metrics["子集交易日数"] == 8  # 每折 4 个区间，边界不计


def test_fold_table_reports_per_split_returns(tmp_path):
    _write_run(tmp_path, "base", daily_returns={0: [0.02] * 5, 1: [-0.01] * 5, 2: [0.0] * 5})
    run = load_run(tmp_path / "base")
    table = fold_table(run, [0, 1, 2]).set_index("折序号")
    assert table.loc[0, "折收益"] == pytest.approx(1.02**4 - 1, rel=1e-9)
    assert table.loc[1, "折收益"] == pytest.approx(0.99**4 - 1, rel=1e-9)
    assert table.loc[1, "折最大回撤"] < 0


def test_validate_alignment_rejects_window_mismatch(tmp_path):
    _write_run(tmp_path, "base", windows={0: ("20210101", "20210601")})
    _write_run(tmp_path, "arm", windows={0: ("20210101", "20210701")})
    base = load_run(tmp_path / "base")
    arm = load_run(tmp_path / "arm")
    with pytest.raises(ValueError, match="逐折窗口不一致"):
        validate_alignment([base, arm])


def test_validate_alignment_rejects_split_and_state_mismatch(tmp_path):
    _write_run(tmp_path, "base", splits=(0, 1, 2))
    _write_run(tmp_path, "arm", splits=(0, 1))
    base = load_run(tmp_path / "base")
    arm = load_run(tmp_path / "arm")
    with pytest.raises(ValueError, match="折集合不一致"):
        validate_alignment([base, arm])

    _write_run(tmp_path, "arm_state", splits=(0, 1, 2), data_state_id="zzz99999")
    other = load_run(tmp_path / "arm_state")
    with pytest.raises(ValueError, match="数据态不一致"):
        validate_alignment([base, other])


def test_compare_runs_outputs_expected_columns(tmp_path):
    _write_run(tmp_path, "base", daily_returns={0: [0.01] * 5, 1: [0.01] * 5, 2: [-0.02] * 5})
    # 臂 A：折 1 与基线同向（均正），折 2 反向（基线负、臂正）
    _write_run(tmp_path, "arm_a", daily_returns={0: [0.01] * 5, 1: [0.02] * 5, 2: [0.01] * 5})
    base = load_run(tmp_path / "base", label="基线")
    arm = load_run(tmp_path / "arm_a", label="臂A")
    result = compare_runs(base, [arm], [1, 2])

    summary = result["summary"].set_index("运行")
    assert summary.loc["基线", "折数"] == 2
    assert summary.loc["基线", "ΔCAGR(vs基线)"] == pytest.approx(0.0)
    assert summary.loc["臂A", "ΔCAGR(vs基线)"] > 0
    assert summary.loc["基线", "折收益同向(vs基线)"] == 2
    assert summary.loc["臂A", "折收益为正(子集)"] == 2
    assert summary.loc["臂A", "折收益同向(vs基线)"] == 1

    folds = result["folds"]
    assert set(folds.columns) >= {"折序号", "运行", "折收益", "基线折收益", "相对基线折收益差"}
    assert not folds.columns.duplicated().any()
    assert result["fold_returns"].shape == (2, 2)


def test_compare_runs_rejects_unknown_split(tmp_path):
    _write_run(tmp_path, "base")
    _write_run(tmp_path, "arm")
    base = load_run(tmp_path / "base")
    arm = load_run(tmp_path / "arm")
    with pytest.raises(ValueError, match="不在可用折"):
        compare_runs(base, [arm], [7])


def test_load_run_accepts_chain_nav_file_path(tmp_path):
    raw_dir = _write_run(tmp_path, "base")
    chain_file = sorted(raw_dir.glob("chain_nav_*.csv"))[0]
    run = load_run(chain_file)
    assert isinstance(run, RunArtifacts)
    assert run.directory == raw_dir
    assert run.data_state_id == "abc12345"


def test_load_run_warns_when_data_state_missing(tmp_path):
    _write_run(tmp_path, "base", data_state_id=None)
    run = load_run(tmp_path / "base")
    assert run.data_state_id is None
    assert any("数据态" in message for message in run.warnings)


def test_fold_table_handles_empty_segment(tmp_path):
    _write_run(tmp_path, "base")
    run = load_run(tmp_path / "base")
    frame = run.chain.copy()
    frame.loc[frame["split_index"] == 2, "nav"] = np.nan
    run.chain = frame
    table = fold_table(run, [2]).set_index("折序号")
    assert np.isnan(table.loc[2, "折收益"])
    assert table.loc[2, "折交易日数"] == 0
