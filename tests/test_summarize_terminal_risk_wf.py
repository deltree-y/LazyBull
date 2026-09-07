"""terminal_loss 滚动 WF 汇总工具测试（合成 report 目录）"""

import json
from pathlib import Path

from scripts.summarize_terminal_risk_wf import collect_fold_rows


def _write_fold(root: Path, name: str, pr_auc: float, event_rate: float,
                mean_pred: float, n: int = 1000):
    fold_dir = root / name
    fold_dir.mkdir(parents=True)
    with open(fold_dir / "terminal_loss_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "es": {
                    "n": n, "event_rate": event_rate, "mean_pred": mean_pred,
                    "logloss": 0.31, "brier": 0.08, "pr_auc": pr_auc,
                },
                "train": {"n": n * 3, "event_rate": 0.14, "logloss": 0.37,
                          "brier": 0.12, "pr_auc": 0.33},
                "label_coverage": [],
                "isolation_dropped": {"train": 1},
            },
            f,
        )
    with open(fold_dir / "terminal_loss_model.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": "terminal_vol_scaled_loss",
                "feature_names": ["f1"],
                "metadata": {
                    "stage_dates": {
                        "train": ["20210101", "20241231"],
                        "es": ["20250101", "20250630"],
                    },
                    "best_iteration": 128,
                    "n_train": n * 3,
                    "n_es": n,
                },
            },
            f,
        )


def test_collect_rows_lift_and_bias(tmp_path):
    _write_fold(tmp_path, "2024H1", pr_auc=0.12, event_rate=0.10, mean_pred=0.145)
    _write_fold(tmp_path, "2025H2", pr_auc=0.09, event_rate=0.09, mean_pred=0.13)
    summary = collect_fold_rows(tmp_path)
    assert len(summary) == 2
    row = summary[summary["fold"] == "2024H1"].iloc[0]
    assert row["lift"] == 1.2
    assert abs(row["pred_bias"] - 0.045) < 1e-9
    assert row["es_start"] == "20250101"
    assert row["best_iteration"] == 128


def test_missing_report_skipped(tmp_path):
    _write_fold(tmp_path, "2024H1", pr_auc=0.12, event_rate=0.10, mean_pred=0.14)
    (tmp_path / "empty_fold").mkdir()
    summary = collect_fold_rows(tmp_path)
    assert list(summary["fold"]) == ["2024H1"]
