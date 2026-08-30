"""数据质量静态 HTML 报告。"""

from html import escape
from pathlib import Path
from typing import Dict, List

import pandas as pd


def compare_snapshots(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """比较两次扫描中的异常项，标记新增、修复和持续问题。"""
    key_columns = ["layer", "dataset", "partition", "metric", "column"]
    current_errors = _error_keys(current, key_columns)
    previous_errors = _error_keys(previous, key_columns)
    records: List[Dict[str, str]] = []
    for key in sorted(current_errors - previous_errors):
        records.append(_change_record(key_columns, key, "新增异常"))
    for key in sorted(previous_errors - current_errors):
        records.append(_change_record(key_columns, key, "已修复"))
    for key in sorted(current_errors & previous_errors):
        records.append(_change_record(key_columns, key, "持续异常"))
    return pd.DataFrame.from_records(records, columns=key_columns + ["change"])


def write_html_report(metrics: pd.DataFrame, changes: pd.DataFrame, output_path: Path) -> None:
    """生成可离线打开的数据质量静态 HTML 报告。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = metrics[metrics["status"] == "error"]
    warnings = metrics[metrics["status"] == "warning"]
    summary = pd.DataFrame(
        [
            {"状态": "错误", "数量": len(errors)},
            {"状态": "警告", "数量": len(warnings)},
            {"状态": "指标总数", "数量": len(metrics)},
        ]
    )
    sections = [
        ("总体状态", summary),
        ("异常清单", metrics[metrics["status"] != "ok"]),
        ("快照变化", changes),
        (
            "数据集摘要",
            metrics[
                metrics["metric"].isin(["partition_count", "latest_partition", "sync_watermark"])
            ],
        ),
        ("特征缺失率 Top 50", _top_missing(metrics)),
    ]
    body = "".join(
        f"<section><h2>{title}</h2>{_table(frame)}</section>" for title, frame in sections
    )
    status = "异常" if len(errors) else "警告" if len(warnings) else "健康"
    document = (
        f"<h1>LazyBull 数据质量看板</h1><p>本次状态：<strong>{status}</strong></p>"
        f"{body}</body></html>"
    )
    header = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>LazyBull 数据质量看板</title>"
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:32px;color:#17212b;}"
        "h1{margin-bottom:4px;}section{margin:28px 0;}table{border-collapse:collapse;width:100%;"
        "font-size:14px;}th,td{border:1px solid #d6dde5;padding:8px;text-align:left;}"
        "th{background:#edf3f7;}"
        ".error{background:#ffe9e7;}.warning{background:#fff6d8;}</style></head><body>"
    )
    output_path.write_text(header + document, encoding="utf-8")


def _error_keys(metrics: pd.DataFrame, key_columns: List[str]) -> set:
    return {
        tuple("" if pd.isna(value) else str(value) for value in row)
        for row in metrics.loc[metrics["status"] == "error", key_columns].itertuples(
            index=False, name=None
        )
    }


def _change_record(columns: List[str], key: tuple, change: str) -> Dict[str, str]:
    return {**dict(zip(columns, key)), "change": change}


def _top_missing(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics[metrics["metric"] == "missing_ratio"].sort_values("value", ascending=False).head(50)
    )


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>无记录</p>"
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for row in frame.itertuples(index=False, name=None):
        class_name = ""
        if "error" in row:
            class_name = ' class="error"'
        elif "warning" in row:
            class_name = ' class="warning"'
        cells = "".join(f"<td>{escape('' if pd.isna(value) else str(value))}</td>" for value in row)
        rows.append(f"<tr{class_name}>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
