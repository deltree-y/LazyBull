import numpy as np
import pandas as pd

from src.lazybull.quality.metrics import collect_partition_metrics


def test_collect_partition_metrics_reports_missing_and_numeric_anomalies(tmp_path):
    file_path = tmp_path / "2026-08-28.parquet"
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "factor": [1.0, np.nan, np.inf],
            "pe_ttm": [10.0, 2001.0, 20.0],
        }
    ).to_parquet(file_path, index=False)

    metrics = collect_partition_metrics(
        file_path,
        layer="features",
        dataset="cs_train",
        partition="20260828",
        anomaly_limits={"pe_ttm": 1000.0},
    )
    values = {(item["metric"], item["column"]): item["value"] for item in metrics}

    assert values[("rows", None)] == 3
    assert values[("column_count", None)] == 3
    assert values[("missing_ratio", "factor")] == 1 / 3
    assert values[("infinite_count", "factor")] == 1
    assert values[("outlier_count", "pe_ttm")] == 1
