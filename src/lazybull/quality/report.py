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


def write_html_report(
    metrics: pd.DataFrame,
    changes: pd.DataFrame,
    output_path: Path,
    max_detail_rows: int = 100,
) -> None:
    """生成可离线打开的数据质量静态 HTML 报告。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = metrics[metrics["status"] == "error"]
    warnings = metrics[metrics["status"] == "warning"]
    issues = metrics[metrics["status"] != "ok"]
    summary = _summary_cards(len(errors), len(warnings), len(metrics))
    sections = [
        _section("数据集摘要", _dataset_summary(metrics)),
        _limited_section("待处理问题", _issue_display(issues), max_detail_rows),
        _limited_section("与上次扫描的变化", _change_display(changes), max_detail_rows),
        _section("缺失率最高的特征", _missing_display(metrics)),
    ]
    body = "".join(f"<section><h2>{title}</h2>{content}</section>" for title, content in sections)
    status = "异常" if len(errors) else "警告" if len(warnings) else "健康"
    document = (
        '<main><header><p class="eyebrow">LAZYBULL / DATA HEALTH</p>'
        f'<h1>数据质量看板 <span class="status {status}">{status}</span></h1>'
        '<p class="subtitle">全历史分区扫描结果。完整指标明细保存在同目录 '
        "<code>latest_metrics.parquet</code>。</p></header>"
        f'<div class="cards">{summary}</div>{body}</main></body></html>'
    )
    styles = "".join(
        [
            ":root{--ink:#192329;--muted:#617078;--line:#dce4e7;--paper:#f7f9f8;}",
            ":root{--teal:#087e7e;--red:#b42318;--amber:#a15c00;}",
            "*{box-sizing:border-box;}",
            "body{margin:0;background:var(--paper);color:var(--ink);"
            "font-family:Georgia,'Microsoft YaHei',serif;}",
            "main{max-width:1440px;margin:auto;padding:38px 32px 60px;}",
            "header{border-bottom:3px solid var(--ink);padding-bottom:20px;}",
            ".eyebrow{font:600 12px 'Segoe UI',sans-serif;letter-spacing:1px;"
            "color:var(--teal);margin:0 0 8px;}",
            "h1{font-size:32px;margin:0;letter-spacing:0;}h2{font-size:18px;margin:0 0 10px;}",
            ".subtitle{color:var(--muted);margin:10px 0 0;}",
            ".status{font:600 14px 'Segoe UI',sans-serif;padding:4px 8px;"
            "vertical-align:middle;}",
            ".status.健康{color:var(--teal);background:#dff4ef;}",
            ".status.警告{color:var(--amber);background:#fff0c7;}",
            ".status.异常{color:var(--red);background:#fde8e7;}",
            ".cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));"
            "gap:12px;margin:24px 0;}",
            ".card{background:#fff;border:1px solid var(--line);padding:16px;}",
            ".card-label{display:block;color:var(--muted);" "font:600 12px 'Segoe UI',sans-serif;}",
            ".card-value{display:block;font:700 28px 'Segoe UI',sans-serif;" "margin-top:4px;}",
            "section{margin-top:32px;}.table-wrap{overflow-x:auto;background:#fff;"
            "border:1px solid var(--line);}",
            "table{border-collapse:collapse;width:100%;"
            "font:13px 'Segoe UI','Microsoft YaHei',sans-serif;}",
            "th,td{border-bottom:1px solid var(--line);padding:9px 10px;"
            "text-align:left;white-space:nowrap;}",
            "th{background:#eaf1ef;color:#314048;font-weight:700;position:sticky;top:0;}",
            "tr.error{background:#fff0ef;}tr.warning{background:#fff9e8;}"
            ".hint{color:var(--muted);font-size:13px;}",
            "code{font-family:Consolas,monospace;}@media(max-width:640px){main{padding:24px 16px;}",
            ".cards{grid-template-columns:1fr;}h1{font-size:27px;}}",
        ]
    )
    header = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>LazyBull 数据质量看板</title>"
        f"<style>{styles}</style></head><body>"
    )
    output_path.write_text(header + document, encoding="utf-8")


def _limited_section(title: str, frame: pd.DataFrame, max_rows: int) -> tuple[str, str]:
    """将高基数明细限制为可读的前若干行，并保留总量提示。"""
    limit = max(int(max_rows), 1)
    displayed = frame.head(limit)
    suffix = ""
    if len(frame) > len(displayed):
        suffix = f'<p class="hint">仅展示前 {len(displayed)} 条，共 {len(frame)} 条；完整明细见 Parquet 快照。</p>'
    return title, suffix + _table(displayed)


def _section(title: str, frame: pd.DataFrame) -> tuple[str, str]:
    """构建普通表格章节，确保 DataFrame 不会退化为连续文本。"""
    return title, _table(frame)


def _summary_cards(error_count: int, warning_count: int, metric_count: int) -> str:
    cards = [("错误", error_count), ("警告", warning_count), ("已采集指标", metric_count)]
    return "".join(
        f'<div class="card"><span class="card-label">{label}</span>'
        f'<span class="card-value">{value:,}</span></div>'
        for label, value in cards
    )


def _dataset_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """将分区级长表压缩为每个数据集一行的运行概览。"""
    rows = []
    for (layer, dataset), group in metrics.groupby(["layer", "dataset"], sort=True):
        rows.append(
            {
                "层级": layer,
                "数据集": dataset,
                "分区数": _metric_value(group, "partition_count"),
                "最新分区": _metric_value(group, "latest_partition"),
                "同步水位": _metric_value(group, "sync_watermark"),
                "错误": int((group["status"] == "error").sum()),
                "警告": int((group["status"] == "warning").sum()),
            }
        )
    return pd.DataFrame(rows)


def _metric_value(group: pd.DataFrame, metric: str) -> object:
    values = group.loc[group["metric"] == metric, "value"]
    return "-" if values.empty or pd.isna(values.iloc[0]) else values.iloc[0]


def _issue_display(issues: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "status",
        "layer",
        "dataset",
        "partition",
        "column",
        "metric",
        "value",
        "threshold",
        "detail",
    ]
    return issues.reindex(columns=columns).sort_values(["status", "layer", "dataset", "partition"])


def _change_display(changes: pd.DataFrame) -> pd.DataFrame:
    return changes.reindex(columns=["change", "layer", "dataset", "partition", "column", "metric"])


def _missing_display(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["layer", "dataset", "partition", "column", "value", "status"]
    return _top_missing(metrics).reindex(columns=columns)


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
    return (
        f'<div class="table-wrap"><table><thead><tr>{headers}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
