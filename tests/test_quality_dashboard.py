import sys

import pandas as pd

from scripts.quality_dashboard import main
from src.lazybull.data.storage import Storage
from src.lazybull.quality.report import compare_snapshots


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
