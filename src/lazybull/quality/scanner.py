"""数据质量全历史扫描。"""

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from ..data.storage import Storage
from ..features.ensure.schema import _optional_factor_sentinel_specs
from .metrics import collect_partition_metrics

DEFAULT_DATASETS: Mapping[str, Sequence[str]] = {
    "raw": ("daily", "adj_factor", "daily_basic", "suspend", "stk_limit", "moneyflow"),
    "clean": ("daily",),
    "features": ("cs_train", "cs_infer"),
}


def scan_quality(
    storage: Storage,
    config: Mapping[str, Any],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """扫描指定存储中的全部受管分区，返回规范化质量指标长表。"""
    records: List[Dict[str, Any]] = []
    datasets = config.get("datasets", DEFAULT_DATASETS)
    anomaly_limits = config.get("anomaly_limits", {})
    plans = [
        (layer, dataset, _partition_files(storage, layer, dataset, start_date, end_date))
        for layer, names in datasets.items()
        for dataset in names
    ]
    progress = _ScanProgress(
        total_partitions=sum(len(files) for _, _, files in plans),
        interval_seconds=float(config.get("progress_interval_seconds", 15.0)),
    )
    for layer, names in datasets.items():
        for dataset in names:
            files = next(
                files
                for item_layer, item_dataset, files in plans
                if (item_layer, item_dataset) == (layer, dataset)
            )
            progress.dataset_started(layer, dataset, len(files))
            records.extend(_dataset_metrics(storage, layer, dataset, files))
            for file_path, partition in files:
                progress.before_partition(layer, dataset, partition)
                records.extend(
                    collect_partition_metrics(
                        file_path,
                        layer,
                        dataset,
                        partition,
                        anomaly_limits.get(dataset, {}),
                    )
                )
                if layer == "features":
                    records.extend(_sentinel_metrics(file_path, layer, dataset, partition))
                progress.partition_completed(layer, dataset, partition)
    metrics = pd.DataFrame.from_records(records, columns=_METRIC_COLUMNS)
    metrics = _add_column_missing_metrics(metrics, config)
    metrics = _add_coverage_metrics(metrics, config, scan_start_date=start_date)
    progress.completed(len(metrics))
    return metrics


_METRIC_COLUMNS = [
    "layer",
    "dataset",
    "partition",
    "metric",
    "column",
    "value",
    "threshold",
    "status",
    "detail",
]


class _ScanProgress:
    """按时间间隔汇报分区扫描进度，避免长任务无输出。"""

    def __init__(self, total_partitions: int, interval_seconds: float):
        self.total_partitions = total_partitions
        self.interval_seconds = max(interval_seconds, 0.0)
        self.completed_partitions = 0
        self.started_at = time.monotonic()
        self.last_logged_at = self.started_at
        logger.info(f"数据质量扫描开始：共 {total_partitions} 个分区")

    def dataset_started(self, layer: str, dataset: str, partition_count: int) -> None:
        logger.info(f"扫描数据集：{layer}/{dataset}，共 {partition_count} 个分区")

    def before_partition(self, layer: str, dataset: str, partition: str) -> None:
        logger.info(
            f"质量扫描进行中：即将读取 [{self.completed_partitions + 1}/{self.total_partitions}] "
            f"{layer}/{dataset}/{partition}"
        )

    def partition_completed(self, layer: str, dataset: str, partition: str) -> None:
        self.completed_partitions += 1
        if self.completed_partitions == self.total_partitions or self._should_log():
            elapsed = time.monotonic() - self.started_at
            remaining = self._remaining_seconds(elapsed)
            logger.info(
                f"质量扫描进度：{self.completed_partitions}/{self.total_partitions} "
                f"({self._percentage():.1%})，刚完成 {layer}/{dataset}/{partition}，"
                f"已耗时 {elapsed:.0f} 秒，预计剩余 {remaining:.0f} 秒"
            )

    def completed(self, metric_count: int) -> None:
        elapsed = time.monotonic() - self.started_at
        logger.info(
            "数据质量扫描完成："
            f"{self.completed_partitions} 个分区，{metric_count} 条指标，耗时 {elapsed:.1f} 秒"
        )

    def _should_log(self) -> bool:
        now = time.monotonic()
        if now - self.last_logged_at < self.interval_seconds:
            return False
        self.last_logged_at = now
        return True

    def _remaining_seconds(self, elapsed: float) -> float:
        if self.completed_partitions == 0:
            return 0.0
        return (
            elapsed
            / self.completed_partitions
            * (self.total_partitions - self.completed_partitions)
        )

    def _percentage(self) -> float:
        if self.total_partitions == 0:
            return 1.0
        return self.completed_partitions / self.total_partitions


def evaluate_quality(metrics: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """按配置阈值给指标附加 ok、warning 或 error 状态。"""
    result = metrics.copy()
    result["status"] = "ok"
    result["detail"] = ""
    for index, row in result.iterrows():
        if row["metric"] == "column_missing_ratio":
            missing_limit, enforced = _missing_ratio_rule(
                config,
                str(row["layer"]),
                str(row["dataset"]),
                str(row["column"]),
            )
            result.loc[index, "threshold"] = missing_limit
            value = float(row["value"])
            exceeds = value > missing_limit or (missing_limit >= 1.0 and value >= missing_limit)
            if enforced and exceeds:
                result.loc[index, ["status", "detail"]] = [
                    "error",
                    "扫描区间加权缺失率超过阈值",
                ]
        elif row["metric"] == "coverage_ratio" and row["value"] < row["threshold"]:
            result.loc[index, ["status", "detail"]] = ["error", "分区覆盖率低于阈值"]
        elif row["metric"] == "infinite_count" and row["value"] > 0:
            result.loc[index, ["status", "detail"]] = ["error", "包含无穷值"]
        elif row["metric"] == "outlier_count" and row["value"] > 0:
            result.loc[index, ["status", "detail"]] = ["warning", "存在超过阈值的数值"]
        elif row["metric"] == "sentinel_version" and row["value"] != row["threshold"]:
            result.loc[index, ["status", "detail"]] = ["error", "schema 哨兵版本不符"]
    return result


def _missing_ratio_limit(config: Mapping[str, Any], layer: str, dataset: str, column: str) -> float:
    return _missing_ratio_rule(config, layer, dataset, column)[0]


def _missing_ratio_rule(
    config: Mapping[str, Any], layer: str, dataset: str, column: str
) -> tuple[float, bool]:
    dataset_limits = config.get("missing_ratio_limits", {}).get(f"{layer}/{dataset}", {})
    if column in dataset_limits:
        return float(dataset_limits[column]), True
    required = config.get(
        "missing_ratio_required_datasets",
        ["features/cs_train", "features/cs_infer"],
    )
    return float(config.get("missing_ratio_error", 1.0)), f"{layer}/{dataset}" in required


def save_snapshot(metrics: pd.DataFrame, output_path: Path) -> None:
    """将质量指标保存为可供后续趋势比较的 Parquet 快照。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = metrics.copy()
    snapshot["value"] = snapshot["value"].map(lambda value: None if pd.isna(value) else str(value))
    snapshot.to_parquet(output_path, index=False)


def _partition_files(
    storage: Storage,
    layer: str,
    dataset: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[tuple[Path, str]]:
    if layer in {"raw", "clean"}:
        dates = storage.list_partitions(layer, dataset)
        directory = storage.raw_path / dataset if layer == "raw" else storage.clean_path / dataset
    elif layer == "features":
        directory = storage.features_path / dataset
        dates = [path.stem for path in directory.glob("*.parquet")] if directory.exists() else []
    else:
        raise ValueError(f"不支持的数据层: {layer}")
    files = []
    for date in sorted(dates):
        normalized_date = date.replace("-", "")
        if start_date and normalized_date < start_date.replace("-", ""):
            continue
        if end_date and normalized_date > end_date.replace("-", ""):
            continue
        file_path = directory / f"{date}.parquet"
        if file_path.exists():
            files.append((file_path, normalized_date))
    return files


def _dataset_metrics(
    storage: Storage,
    layer: str,
    dataset: str,
    files: Iterable[tuple[Path, str]],
) -> List[Dict[str, Any]]:
    file_list = list(files)
    records = [
        _record(layer, dataset, None, "partition_count", None, len(file_list)),
        _record(
            layer,
            dataset,
            None,
            "latest_partition",
            None,
            file_list[-1][1] if file_list else None,
        ),
    ]
    if layer == "raw":
        records.append(
            _record(
                layer, dataset, None, "sync_watermark", None, storage.load_sync_watermark(dataset)
            )
        )
    return records


def _add_coverage_metrics(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
    scan_start_date: Optional[str] = None,
) -> pd.DataFrame:
    reference = metrics[
        (metrics["layer"] == "raw")
        & (metrics["dataset"] == "daily")
        & (metrics["metric"] == "latest_partition")
    ]
    daily_partitions = set(
        metrics.loc[
            (metrics["layer"] == "raw")
            & (metrics["dataset"] == "daily")
            & (metrics["metric"] == "rows"),
            "partition",
        ].dropna()
    )
    if reference.empty or not daily_partitions:
        return metrics
    required = config.get("coverage_required_datasets", [])
    start_dates = config.get("coverage_start_dates", {})
    tail_lags = config.get("coverage_tail_lag_trading_days", {})
    records = []
    threshold = float(config.get("coverage_ratio_error", 0.85))
    for item in required:
        layer, dataset = item.split("/", 1)
        partitions = set(
            metrics.loc[
                (metrics["layer"] == layer)
                & (metrics["dataset"] == dataset)
                & (metrics["metric"] == "rows"),
                "partition",
            ].dropna()
        )
        ordered_reference = sorted(daily_partitions)
        configured_start = str(start_dates.get(item, ordered_reference[0])).replace("-", "")
        if scan_start_date is not None:
            start_date = max(
                configured_start,
                ordered_reference[0],
                scan_start_date.replace("-", ""),
            )
        elif partitions:
            start_date = max(configured_start, ordered_reference[0], min(partitions))
        else:
            start_date = max(configured_start, ordered_reference[0])
        tail_lag = max(int(tail_lags.get(item, 0)), 0)
        if tail_lag >= len(ordered_reference):
            expected_partitions = set()
        else:
            end_date = ordered_reference[-(tail_lag + 1)]
            expected_partitions = {
                partition for partition in daily_partitions if start_date <= partition <= end_date
            }
        coverage_ratio = (
            len(partitions & expected_partitions) / len(expected_partitions)
            if expected_partitions
            else 1.0
        )
        records.append(
            _record(
                layer,
                dataset,
                None,
                "coverage_ratio",
                None,
                coverage_ratio,
                threshold=threshold,
            )
        )
    return pd.concat([metrics, pd.DataFrame.from_records(records)], ignore_index=True)


def _add_column_missing_metrics(metrics: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """按扫描区间行数加权汇总列缺失率，避免逐日稀疏状态制造海量误报。"""
    row_counts = metrics.loc[
        metrics["metric"] == "rows", ["layer", "dataset", "partition", "value"]
    ].rename(columns={"value": "row_count"})
    missing = metrics.loc[
        metrics["metric"] == "missing_ratio",
        ["layer", "dataset", "partition", "column", "value"],
    ]
    if missing.empty or row_counts.empty:
        return metrics

    weighted = missing.merge(
        row_counts,
        on=["layer", "dataset", "partition"],
        how="left",
        validate="many_to_one",
    )
    weighted["missing_count"] = weighted["value"].astype(float) * weighted["row_count"].astype(
        float
    )
    dataset_rows = (
        row_counts.groupby(["layer", "dataset"], sort=False)["row_count"].sum().astype(float)
    )
    records = []
    for (layer, dataset, column), group in weighted.groupby(
        ["layer", "dataset", "column"], sort=False
    ):
        total_rows = float(dataset_rows.loc[(layer, dataset)])
        present_rows = float(group["row_count"].sum())
        absent_rows = max(total_rows - present_rows, 0.0)
        missing_rows = float(group["missing_count"].sum()) + absent_rows
        ratio = missing_rows / total_rows if total_rows else 0.0
        records.append(
            _record(
                layer,
                dataset,
                None,
                "column_missing_ratio",
                column,
                ratio,
                threshold=_missing_ratio_limit(config, layer, dataset, column),
            )
        )
    return pd.concat([metrics, pd.DataFrame.from_records(records)], ignore_index=True)


def _sentinel_metrics(
    file_path: Path, layer: str, dataset: str, partition: str
) -> List[Dict[str, Any]]:
    table_schema = pq.read_schema(str(file_path))
    records: List[Dict[str, Any]] = []
    for group, (column, expected) in _optional_factor_sentinel_specs().items():
        if column not in table_schema.names:
            continue
        values = pq.read_table(str(file_path), columns=[column]).column(0).to_pylist()
        actual = values[0] if values and all(value == values[0] for value in values) else None
        records.append(
            _record(
                layer,
                dataset,
                partition,
                "sentinel_version",
                f"{group}:{column}",
                actual,
                threshold=expected,
            )
        )
    return records


def _record(
    layer: str,
    dataset: str,
    partition: Optional[str],
    metric: str,
    column: Optional[str],
    value: Any,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "layer": layer,
        "dataset": dataset,
        "partition": partition,
        "metric": metric,
        "column": column,
        "value": value,
        "threshold": threshold,
        "status": "ok",
        "detail": "",
    }
