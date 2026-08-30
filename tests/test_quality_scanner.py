import pandas as pd
import pytest

from src.lazybull.data.storage import Storage
from src.lazybull.quality import scanner
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


def test_coverage_respects_dataset_start_and_label_tail_lag(tmp_path):
    storage = Storage(str(tmp_path))
    daily = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260828"]})
    for trade_date in ["20260825", "20260826", "20260827", "20260828"]:
        storage.save_raw_by_date(daily.assign(trade_date=trade_date), "daily", trade_date)
    storage.save_cs_train_day(daily.assign(trade_date="20260825"), "20260825")

    metrics = scan_quality(
        storage,
        {
            "datasets": {"raw": ["daily"], "features": ["cs_train"]},
            "coverage_required_datasets": ["features/cs_train"],
            "coverage_start_dates": {"features/cs_train": "20260825"},
            "coverage_tail_lag_trading_days": {"features/cs_train": 3},
        },
    )

    coverage = metrics[metrics["metric"] == "coverage_ratio"].iloc[0]
    assert coverage["value"] == 1.0


def test_coverage_infers_later_dataset_start_without_hiding_explicit_start_gap(tmp_path):
    storage = Storage(str(tmp_path))
    daily = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260825"]})
    for trade_date in ["20260825", "20260826", "20260827"]:
        storage.save_raw_by_date(daily.assign(trade_date=trade_date), "daily", trade_date)
    storage.save_clean_by_date(daily.assign(trade_date="20260826"), "daily", "20260826")
    storage.save_clean_by_date(daily.assign(trade_date="20260827"), "daily", "20260827")
    config = {
        "datasets": {"raw": ["daily"], "clean": ["daily"]},
        "coverage_required_datasets": ["clean/daily"],
    }

    inferred = scan_quality(storage, config)
    explicit = scan_quality(storage, config, start_date="20260825")

    assert inferred.loc[inferred["metric"] == "coverage_ratio", "value"].iloc[0] == 1.0
    assert explicit.loc[explicit["metric"] == "coverage_ratio", "value"].iloc[0] == pytest.approx(
        2 / 3
    )


def test_missing_ratio_is_evaluated_over_full_scan_window(tmp_path):
    storage = Storage(str(tmp_path))
    storage.save_raw_by_date(
        pd.DataFrame({"ts_code": ["000001.SZ"], "factor": [None]}),
        "daily",
        "20260828",
    )
    storage.save_raw_by_date(
        pd.DataFrame({"ts_code": ["000001.SZ"], "factor": [1.0]}),
        "daily",
        "20260829",
    )

    metrics = scan_quality(storage, {"datasets": {"raw": ["daily"]}})
    evaluated = evaluate_quality(metrics, {"missing_ratio_error": 1.0})
    aggregate = evaluated[
        (evaluated["metric"] == "column_missing_ratio") & (evaluated["column"] == "factor")
    ].iloc[0]

    assert aggregate["value"] == 0.5
    assert aggregate["status"] == "ok"
    assert not (evaluated.loc[evaluated["metric"] == "missing_ratio", "status"] == "error").any()


def test_aggregate_missing_ratio_counts_schema_absence_as_missing(tmp_path):
    storage = Storage(str(tmp_path))
    storage.save_cs_train_day(
        pd.DataFrame({"ts_code": ["000001.SZ"], "factor": [1.0]}),
        "20260828",
    )
    storage.save_cs_train_day(
        pd.DataFrame({"ts_code": ["000001.SZ"]}),
        "20260829",
    )
    config = {
        "datasets": {"features": ["cs_train"]},
        "missing_ratio_limits": {"features/cs_train": {"factor": 0.4}},
    }

    evaluated = evaluate_quality(scan_quality(storage, config), config)
    aggregate = evaluated[
        (evaluated["metric"] == "column_missing_ratio") & (evaluated["column"] == "factor")
    ].iloc[0]

    assert aggregate["value"] == 0.5
    assert aggregate["status"] == "error"


def test_column_specific_missing_limit_catches_training_feature_gap(tmp_path):
    storage = Storage(str(tmp_path))
    storage.save_raw_by_date(
        pd.DataFrame({"ts_code": ["000001.SZ"], "cf_nm": [None]}),
        "daily",
        "20260828",
    )
    config = {
        "datasets": {"raw": ["daily"]},
        "missing_ratio_error": 1.0,
        "missing_ratio_limits": {"raw/daily": {"cf_nm": 0.6}},
    }

    evaluated = evaluate_quality(scan_quality(storage, config), config)
    aggregate = evaluated[
        (evaluated["metric"] == "column_missing_ratio") & (evaluated["column"] == "cf_nm")
    ].iloc[0]

    assert aggregate["status"] == "error"


def test_raw_optional_all_missing_column_is_informational(tmp_path):
    storage = Storage(str(tmp_path))
    storage.save_raw_by_date(
        pd.DataFrame({"ts_code": ["000001.SZ"], "suspend_timing": [None]}),
        "suspend",
        "20260828",
    )

    evaluated = evaluate_quality(
        scan_quality(storage, {"datasets": {"raw": ["suspend"]}}),
        {"missing_ratio_error": 1.0},
    )
    aggregate = evaluated[
        (evaluated["metric"] == "column_missing_ratio") & (evaluated["column"] == "suspend_timing")
    ].iloc[0]

    assert aggregate["value"] == 1.0
    assert aggregate["status"] == "ok"


def test_scan_quality_reports_start_progress_and_completion(tmp_path, monkeypatch):
    storage = Storage(str(tmp_path))
    daily = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260828"]})
    storage.save_raw_by_date(daily, "daily", "20260828")
    messages = []
    monkeypatch.setattr(scanner.logger, "info", messages.append)

    scan_quality(
        storage,
        {"datasets": {"raw": ["daily"]}, "progress_interval_seconds": 0},
    )

    assert any("数据质量扫描开始" in message for message in messages)
    assert any("质量扫描进度" in message for message in messages)
    assert any("数据质量扫描完成" in message for message in messages)
