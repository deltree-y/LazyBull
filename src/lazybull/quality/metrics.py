"""Parquet 分区质量指标采集。"""

from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def collect_partition_metrics(
    file_path: Path,
    layer: str,
    dataset: str,
    partition: str,
    anomaly_limits: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """采集一个 Parquet 分区的行数、schema、缺失和数值异常指标。"""
    limits = anomaly_limits or {}
    parquet_file = pq.ParquetFile(str(file_path))
    row_count = parquet_file.metadata.num_rows
    metrics: List[Dict[str, Any]] = [
        _metric(layer, dataset, partition, "rows", None, row_count),
        _metric(layer, dataset, partition, "column_count", None, len(parquet_file.schema.names)),
    ]
    table = parquet_file.read()
    for column_name in table.column_names:
        values = table[column_name].to_pandas()
        missing_count = int(values.isna().sum())
        metrics.append(
            _metric(
                layer,
                dataset,
                partition,
                "missing_ratio",
                column_name,
                missing_count / row_count if row_count else 0.0,
            )
        )
        metrics.append(
            _metric(
                layer,
                dataset,
                partition,
                "distinct_count",
                column_name,
                int(values.nunique(dropna=True)),
            )
        )
        # 用 pyarrow 原生类型判断而非 np.issubdtype：pandas 3.0 起字符串列默认为
        # StringDtype，np.issubdtype 无法解释该 dtype 会抛 TypeError。
        col_type = table.schema.field(column_name).type
        if pa.types.is_integer(col_type) or pa.types.is_floating(col_type):
            numeric_values = values.dropna()
            infinite_count = int(np.isinf(numeric_values).sum())
            metrics.append(
                _metric(layer, dataset, partition, "infinite_count", column_name, infinite_count)
            )
            finite_values = numeric_values[np.isfinite(numeric_values)]
            limit = limits.get(column_name)
            if limit is not None:
                abnormal_count = int((finite_values.abs() > limit).sum())
                metrics.append(
                    _metric(
                        layer,
                        dataset,
                        partition,
                        "outlier_count",
                        column_name,
                        abnormal_count,
                        threshold=float(limit),
                    )
                )
    return metrics


def _metric(
    layer: str,
    dataset: str,
    partition: str,
    metric: str,
    column: Optional[str],
    value: Any,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """创建统一指标记录。"""
    if isinstance(value, float) and not isfinite(value):
        value = None
    return {
        "layer": layer,
        "dataset": dataset,
        "partition": partition,
        "metric": metric,
        "column": column,
        "value": value,
        "threshold": threshold,
    }
