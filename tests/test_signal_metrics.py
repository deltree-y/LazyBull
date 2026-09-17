# -*- coding: utf-8 -*-
"""日频信号级配对对比（scripts/compare/signal_metrics.py）专项测试。"""

import numpy as np
import pandas as pd
import pytest

from scripts.compare.signal_metrics import (
    align_panels,
    build_day_panels,
    fold_metric_table,
    paired_metric_table,
)
from src.lazybull.risk.terminal_loss.block_stats import BootstrapConfig, paired_day_mean_ci


def _write_batch(root, folds_returns, split_label="wf", run_id="run1"):
    """构造最小可用的 WF 产物：chain_nav + summary + 逐日 Top-K 明细。"""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    chain_rows = []
    summary_rows = []
    for split_index, days in folds_returns.items():
        summary_rows.append(
            {"split_index": split_index, "test_start": days[0], "test_end": days[-1]}
        )
        for day in days:
            chain_rows.append({"date": day, "nav": 1.0, "split_index": split_index})
    detail_rows = []
    for split_index, days in folds_returns.items():
        for day in days:
            for topk in (20, 30):
                for rank in range(1, topk + 1):
                    detail_rows.append(
                        {
                            "split_index": split_index,
                            "trade_date": day,
                            "topk": topk,
                            "rank": rank,
                            "ts_code": f"{rank:06d}.SZ",
                            "pred_score": float(topk - rank),
                            "true_return": 0.01 * (topk - rank) / topk,
                        }
                    )
    pd.DataFrame(chain_rows).to_csv(
        raw / f"chain_nav_{split_label}_{run_id}.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary_rows).to_csv(
        raw / f"walk_forward_summary_x_{run_id}.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(detail_rows).to_csv(
        raw / f"walk_forward_topk_details_{split_label}_{run_id}_split00.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return root


def test_build_day_panels_and_alignment(tmp_path):
    base = _write_batch(tmp_path / "base", {0: ["20240101", "20240102"]})
    arm = _write_batch(tmp_path / "arm", {0: ["20240101", "20240102"]}, run_id="run2")

    panels_a = build_day_panels(arm, topk_list=[20], label="arm")
    panels_b = build_day_panels(base, topk_list=[20], label="base")
    aligned = align_panels(panels_a[20], panels_b[20])

    assert len(aligned) == 2
    assert {"日均收益_A", "日均收益_B", "Δ日均收益", "Δ命中率"} <= set(aligned.columns)
    # 两臂数据相同 → 差值恒为 0
    assert np.allclose(aligned["Δ日均收益"], 0.0)
    assert np.allclose(aligned["Δ命中率"], 0.0)


def test_align_panels_rejects_day_mismatch(tmp_path):
    base = _write_batch(tmp_path / "base", {0: ["20240101", "20240102"]})
    arm = _write_batch(tmp_path / "arm", {0: ["20240101"]}, run_id="run2")
    panels_a = build_day_panels(arm, topk_list=[20], label="arm")
    panels_b = build_day_panels(base, topk_list=[20], label="base")

    with pytest.raises(ValueError, match="未完全对齐"):
        align_panels(panels_a[20], panels_b[20])

    # 显式放行时只告警不报错
    aligned = align_panels(panels_a[20], panels_b[20], allow_day_mismatch=True)
    assert len(aligned) == 1


def test_build_day_panels_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="未找到逐日 Top-K 明细"):
        build_day_panels(tmp_path / "empty", topk_list=[20], label="x")


def test_paired_day_mean_ci_is_deterministic_and_guards_block():
    days = [f"2024{d:04d}" for d in range(101, 201)]
    rng = np.random.default_rng(0)
    deltas = rng.normal(0.0, 0.01, size=len(days))

    first = paired_day_mean_ci(
        deltas, days, BootstrapConfig(block_days=20, n_resamples=200, seed=7)
    )
    second = paired_day_mean_ci(
        deltas, days, BootstrapConfig(block_days=20, n_resamples=200, seed=7)
    )
    assert first == second
    assert first["n_days"] == len(days)

    with pytest.raises(ValueError, match="块长"):
        paired_day_mean_ci(deltas, days, BootstrapConfig(block_days=len(days), n_resamples=10))
    with pytest.raises(ValueError, match="长度不一致"):
        paired_day_mean_ci(deltas, days[:-1])


def test_paired_day_mean_ci_drops_nan_days():
    days = [f"2024{d:04d}" for d in range(101, 141)]
    deltas = np.full(len(days), 0.02)
    deltas[0] = np.nan
    stats = paired_day_mean_ci(deltas, days, BootstrapConfig(block_days=5, n_resamples=50))
    assert stats["n_days_dropped"] == 1
    assert stats["n_days"] == len(days) - 1


def test_paired_metric_table_reports_units_and_fold_sign(tmp_path):
    base = _write_batch(
        tmp_path / "base", {0: ["20240101", "20240102"], 1: ["20240201", "20240202"]}
    )
    arm = _write_batch(
        tmp_path / "arm", {0: ["20240101", "20240102"], 1: ["20240201", "20240202"]}, run_id="run2"
    )
    panels_a = build_day_panels(arm, topk_list=[20], label="arm")
    panels_b = build_day_panels(base, topk_list=[20], label="base")
    aligned = align_panels(panels_a[20], panels_b[20])

    table = paired_metric_table(aligned, block_days_list=[3], n_resamples=50)
    assert set(table["指标"]) == {"Δ平均持有期收益(bps)", "Δ命中率(pp)"}
    row = table[table["指标"] == "Δ平均持有期收益(bps)"].iloc[0]
    assert row["单位"] == "bps"
    assert row["折数"] == 2
    # 两臂相同 → 点估计为 0、折内同向折数为 0
    assert row["点估计"] == pytest.approx(0.0)
    assert row["折内同向折数"] == 0

    folds = fold_metric_table(aligned)
    assert list(folds["split_index"]) == [0, 1]
    assert "Δ平均持有期收益bps" in folds.columns
