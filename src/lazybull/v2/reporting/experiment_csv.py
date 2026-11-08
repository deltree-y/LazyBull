# -*- coding: utf-8 -*-
"""五份实验完整 CSV 的 schema 与守恒写入（设计 §5.4）。

硬口径：

- UTF-8 无 BOM、**LF**、固定列序、非抽样、root-relative ``source_ref``；空值写空并另带
  ``report_status/degradation_reason_code``（禁止用 ``-``/0 掩盖缺失）；
- 行键唯一（重复即失败）；每行必须给出**全部声明列**（缺列即失败）；任何 NaN/Inf 失败；
- 每份文件产出 ``columns,row_count,key_columns,sha256`` 供 manifest ``csv_schemas`` 登记；
- 本模块只做序列化与守恒校验，不计算任何指标（数值全部来自上游适配层）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.lazybull.v2.evidence.runs_schema import ATTR_CONTRACT_COLS
from src.lazybull.v2.reporting.csv_core import (
    CsvContractError,
    render_table,
    verify_table,
)

__all__ = [
    "ARM_METRICS_COLUMNS",
    "CHAIN_NAV_COLUMNS",
    "CsvContractError",
    "CSV_FILENAMES",
    "EXECUTION_ATTR_COLUMNS",
    "FOLD_METRICS_COLUMNS",
    "STABILITY_COLUMNS",
    "build_csv_bundle",
    "render_csv",
    "verify_csv_text",
]

#: 臂指标（最小列集 + 设计 §5.3.1 映射列；``implementation`` 用于可比性陈列）
ARM_METRICS_COLUMNS: tuple[str, ...] = (
    "arm_id",
    "batch_id",
    "host_mode",
    "implementation",
    "data_state_id",
    "config_fingerprint",
    "total_return",
    "cagr",
    "max_drawdown",
    "volatility",
    "sharpe",
    "trading_days",
    "subset_cagr",
    "subset_max_drawdown",
    "subset_sharpe",
    "delta_cagr",
    "delta_max_drawdown",
    "delta_sharpe",
    "fold_return_aligned_count",
    "fold_count",
    "report_status",
    "degradation_reason_code",
    "source_ref",
)
#: 逐折指标（每臂每折）
FOLD_METRICS_COLUMNS: tuple[str, ...] = (
    "arm_id",
    "split_index",
    "test_start",
    "test_end",
    "model_version",
    "wf_run_id",
    "bt_total_return",
    "bt_annual_return",
    "bt_max_drawdown",
    "bt_sharpe",
    "topk20_positive_day_ratio",
    "topk30_positive_day_ratio",
    "fold_return",
    "fold_max_drawdown",
    "report_status",
    "degradation_reason_code",
    "source_ref",
)
#: 链式净值（每臂每折每源行，原行序）
CHAIN_NAV_COLUMNS: tuple[str, ...] = (
    "arm_id",
    "split_index",
    "source_row_index",
    "date",
    "nav",
    "report_status",
    "degradation_reason_code",
    "source_ref",
)
#: 执行归因（契约全 19 列原值 + 臂/源行序 + 报告状态）
#: 注意：``split_index`` **只出现一次**（ATTR 契约列自带），重复列名会让 CSV 表头
#: 与 manifest schema 同时出现重名（A-01），故不再额外前置同名键。
EXECUTION_ATTR_COLUMNS: tuple[str, ...] = (
    "arm_id",
    *ATTR_CONTRACT_COLS,
    "source_row_index",
    "report_status",
    "degradation_reason_code",
    "source_ref",
)
#: 稳定性（每稳定等价类/结果组；含函数分组列与取值的显式陈列）
STABILITY_COLUMNS: tuple[str, ...] = (
    "group_key",
    "config_json",
    "period_count",
    "period_labels",
    "run_id_list",
    "stability_score",
    "score_mean",
    "score_std",
    "score_min",
    "score_max",
    "chain_cagr_mean",
    "chain_cagr_std",
    "chain_cagr_min",
    "chain_max_drawdown_mean",
    "chain_max_drawdown_worst",
    "oos_cross_split_ir_mean",
    "oos_cross_split_ir_std",
    "bt_win_rate_mean",
    "bt_win_rate_min",
    "chain_sharpe_mean",
    "chain_sharpe_std",
    "latest_run_time",
    "group_columns",
    "group_values",
    "partition_arm_count",
    "report_status",
    "degradation_reason_code",
)
#: 五份 CSV 的固定文件名（HTML companion 链接与 manifest outputs 同源）
CSV_FILENAMES: tuple[str, ...] = (
    "arm_metrics.csv",
    "fold_metrics.csv",
    "chain_nav.csv",
    "execution_attribution.csv",
    "stability.csv",
)
_SCHEMAS: dict[str, tuple[str, ...]] = {
    "arm_metrics.csv": ARM_METRICS_COLUMNS,
    "fold_metrics.csv": FOLD_METRICS_COLUMNS,
    "chain_nav.csv": CHAIN_NAV_COLUMNS,
    "execution_attribution.csv": EXECUTION_ATTR_COLUMNS,
    "stability.csv": STABILITY_COLUMNS,
}
_KEYS: dict[str, tuple[str, ...]] = {
    "arm_metrics.csv": ("arm_id",),
    "fold_metrics.csv": ("arm_id", "split_index"),
    "chain_nav.csv": ("arm_id", "split_index", "source_row_index"),
    "execution_attribution.csv": ("arm_id", "split_index", "source_row_index"),
    "stability.csv": ("group_key",),
}


def render_csv(name: str, rows: Sequence[Mapping[str, Any]]) -> str:
    """渲染单份 CSV 文本（固定列序、LF、重复键失败；唯一实现 = ``csv_core.render_table``）。"""
    if name not in _SCHEMAS:
        raise CsvContractError(f"未登记的 CSV：{name!r}（合法 {sorted(_SCHEMAS)}）")
    return render_table(name, _SCHEMAS[name], _KEYS[name], rows)


def verify_csv_text(name: str, text: str, *, expected_rows: int | None = None) -> dict[str, Any]:
    """回读校验（唯一实现 = ``csv_core.verify_table``；返回 manifest 登记项）。"""
    if name not in _SCHEMAS:
        raise CsvContractError(f"未登记的 CSV：{name!r}")
    return verify_table(
        name, _SCHEMAS[name], _KEYS[name], text, expected_rows=expected_rows
    )


def build_csv_bundle(
    files: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_rows: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """生成五份 CSV 的字节与 manifest 登记项（缺任何一份即失败）。

    Returns:
        ``{"files": {name: bytes}, "schemas": {name: {...}}, "digests": {name: sha}}``
    """
    absent = [name for name in CSV_FILENAMES if name not in files]
    if absent:
        raise CsvContractError(f"缺 CSV：{absent}（五份必须齐备）")
    unknown = [name for name in files if name not in CSV_FILENAMES]
    if unknown:
        raise CsvContractError(f"未登记的 CSV：{unknown}")
    outputs: dict[str, bytes] = {}
    schemas: dict[str, Any] = {}
    digests: dict[str, str] = {}
    for name in CSV_FILENAMES:
        rows = list(files[name])
        expected = (expected_rows or {}).get(name)
        text = render_csv(name, rows)
        schema = verify_csv_text(name, text, expected_rows=expected)
        outputs[name] = text.encode("utf-8")
        schemas[name] = schema
        digests[name] = schema["sha256"]
    return {"files": outputs, "schemas": schemas, "digests": digests}
