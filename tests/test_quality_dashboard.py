import sys

import pandas as pd

from scripts.quality_dashboard import main
from src.lazybull.data.storage import Storage
from src.lazybull.quality.report import compare_snapshots, write_html_report


def test_quality_dashboard_writes_html_snapshot_and_returns_error_for_bad_features(
    tmp_path, monkeypatch
):
    storage = Storage(str(tmp_path / "data"))
    storage.save_cs_train_day(pd.DataFrame({"factor": [None]}), "20260828")
    output_dir = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quality_dashboard.py",
            "--data-root",
            str(storage.root_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert main() == 1
    assert (output_dir / "quality_dashboard.html").exists()
    assert (output_dir / "latest_metrics.parquet").exists()
    assert "异常" in (output_dir / "quality_dashboard.html").read_text(encoding="utf-8")


def test_compare_snapshots_marks_new_resolved_and_persistent_errors():
    columns = ["layer", "dataset", "partition", "metric", "column", "status"]
    previous = pd.DataFrame(
        [
            ["raw", "daily", "20260828", "missing_ratio", "close", "error"],
            ["raw", "daily", "20260829", "missing_ratio", "close", "error"],
        ],
        columns=columns,
    )
    current = pd.DataFrame(
        [
            ["raw", "daily", "20260828", "missing_ratio", "close", "error"],
            ["raw", "daily", "20260830", "missing_ratio", "close", "error"],
        ],
        columns=columns,
    )

    changes = compare_snapshots(current, previous)

    assert set(changes["change"]) == {"新增异常", "已修复", "持续异常"}


def test_html_report_limits_high_cardinality_detail_rows(tmp_path):
    metrics = pd.DataFrame(
        [
            {
                "layer": "features",
                "dataset": "cs_train",
                "partition": f"2026{i:04d}",
                "metric": "missing_ratio",
                "column": "factor",
                "value": 1.0,
                "threshold": 0.8,
                "status": "error",
                "detail": "缺失率超过阈值",
            }
            for i in range(150)
        ]
    )
    output_path = tmp_path / "quality_dashboard.html"

    write_html_report(metrics, pd.DataFrame(), output_path, max_detail_rows=100)

    report = output_path.read_text(encoding="utf-8")
    assert "仅展示前 100 条，共 150 条" in report
    assert "20260099" in report
    assert "20260149" not in report
