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
    progress.completed(len(metrics))
    return _add_coverage_metrics(metrics, config)


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
    missing_limit = float(config.get("missing_ratio_error", 1.0))
    for index, row in result.iterrows():
        if row["metric"] == "missing_ratio" and row["value"] > missing_limit:
            result.loc[index, ["status", "detail"]] = ["error", "缺失率超过阈值"]
        elif row["metric"] == "coverage_ratio" and row["value"] < row["threshold"]:
            result.loc[index, ["status", "detail"]] = ["error", "分区覆盖率低于阈值"]
        elif row["metric"] == "infinite_count" and row["value"] > 0:
            result.loc[index, ["status", "detail"]] = ["error", "包含无穷值"]
        elif row["metric"] == "outlier_count" and row["value"] > 0:
            result.loc[index, ["status", "detail"]] = ["warning", "存在超过阈值的数值"]
        elif row["metric"] == "sentinel_version" and row["value"] != row["threshold"]:
            result.loc[index, ["status", "detail"]] = ["error", "schema 哨兵版本不符"]
    return result


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


def _add_coverage_metrics(metrics: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
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
        records.append(
            _record(
                layer,
                dataset,
                None,
                "coverage_ratio",
                None,
                len(partitions & daily_partitions) / len(daily_partitions),
                threshold=threshold,
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
