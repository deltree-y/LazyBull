import pandas as pd

from src.lazybull.data.storage import Storage
from src.lazybull.quality.scanner import evaluate_quality, scan_quality


def test_scan_quality_collects_all_layers_watermark_and_schema_status(tmp_path):
    storage = Storage(str(tmp_path))
    daily = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260828"], "close": [10.0]})
    storage.save_raw_by_date(daily, "daily", "20260828")
    storage.save_clean_by_date(daily, "daily", "20260828")
    storage.save_raw_by_date(daily, "daily_basic", "20260828")
    storage.save_sync_watermark("daily", "20260828")
    storage.save_cs_train_day(
        pd.DataFrame({"ts_code": ["000001.SZ"], "factor": [float("inf")]}),
        "20260828",
    )

    metrics = scan_quality(
        storage,
        {
            "datasets": {
                "raw": ["daily", "daily_basic"],
                "clean": ["daily"],
                "features": ["cs_train"],
            }
        },
    )
    evaluated = evaluate_quality(metrics, {"missing_ratio_error": 0.5})

    assert {"raw", "clean", "features"}.issubset(set(metrics["layer"]))
    assert metrics.loc[metrics["metric"] == "sync_watermark", "value"].iloc[0] == "20260828"
    assert (evaluated["status"] == "error").any()


def test_coverage_metric_uses_raw_daily_partitions_as_reference(tmp_path):
    storage = Storage(str(tmp_path))
    daily = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260828"]})
    storage.save_raw_by_date(daily, "daily", "20260828")
    storage.save_raw_by_date(daily, "daily", "20260829")
    storage.save_raw_by_date(daily, "daily_basic", "20260828")

    metrics = scan_quality(
        storage,
        {
            "datasets": {"raw": ["daily", "daily_basic"]},
            "coverage_required_datasets": ["raw/daily_basic"],
            "coverage_ratio_error": 0.85,
        },
    )
    coverage = metrics[metrics["metric"] == "coverage_ratio"].iloc[0]

    assert coverage["value"] == 0.5
    assert (
        evaluate_quality(metrics, {"coverage_ratio_error": 0.85}).loc[coverage.name, "status"]
        == "error"
    )
