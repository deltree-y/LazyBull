# -*- coding: utf-8 -*-
"""P6 T4 趋势库结构性常量（机器可读唯一来源）。

契约来源：设计 ``p6_report_design_20261104.md`` §4.3 / §5.5 / §5.7 / §8.1；计划 §3-T4。

硬口径：

- 三类证据域**物理隔离**（根、索引、横幅、输入白名单），不提供跨类总览或混合趋势；
- ``run_index.csv`` 每个发现 run 恰一行（含 ``invalid``），行键 ``run_key``；永为
  ``index_mode=full``（完整索引不抽样）；
- ``trend.csv`` 行键 ``(run_key, metric_name)``，五类指标**每 run 各一行**（缺失也留状态行），
  不插值、不抽样；
- 每行``source_ref + source_sha256`` 必填（指标实际来源文件；无指标时回落到身份来源）；
- HTML 只显示最近 ``VISIBLE_ROWS_MAX`` 行；图上每序列最多 ``MAX_CHART_POINTS`` 点。

本模块只放常量与纯查询，不含 I/O、不含指标计算。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import contracts

__all__ = [
    "DATE_SEMANTICS",
    "INDEX_MODE_FULL",
    "INDEX_MODE_VISIBLE",
    "MAX_CHART_POINTS",
    "METRICS",
    "METRIC_NAMES",
    "RUN_INDEX_COLUMNS",
    "RUN_INDEX_KEYS",
    "TREND_COLUMNS",
    "TREND_KEYS",
    "TREND_SCHEMA",
    "UNIT_LABELS",
    "VISIBLE_ROWS_MAX",
    "metric_order",
    "metric_spec",
    "metrics_for",
    "unit_for",
    "unit_label",
]

#: 趋势投影 schema
TREND_SCHEMA = "p6_trend_projection/v1"
#: 完整索引标记（CSV 恒为 full）
INDEX_MODE_FULL = "full"
#: 页面可见索引标记（仅最近 N 行）
INDEX_MODE_VISIBLE = "visible"
#: HTML 可见 run 行上限（设计 §5.7）
VISIBLE_ROWS_MAX = 500
#: 图每序列最多点数（设计 §3“1,200 点抽样”）
MAX_CHART_POINTS = 1_200
#: 日期语义闭集（不把 fold 序数转自然日）
DATE_SEMANTICS: tuple[str, ...] = ("natural_day", "fold_local_ordinal")

#: 五类指标（顺序即 CSV 内的稳定次级排序；``unit`` 按证据类分列、不得互转）
METRICS: tuple[dict[str, Any], ...] = (
    {
        "name": "annualized_return",
        "label": "年化收益",
        "unit_by_class": {
            "production": "percent_natural_day",
            "research": "decimal_252d",
            "calibration_only": None,
        },
        "note": "production 自然日百分数；research 252 日小数 CAGR；calibration 不生成",
    },
    {
        "name": "max_drawdown",
        "label": "最大回撤",
        "unit_by_class": {
            "production": None,
            "research": "decimal",
            "calibration_only": "pp",
        },
        "note": "research 为链式 MaxDD（非正小数）；calibration 保留 log-MDD/pp；production 无唯一实现",
    },
    {
        "name": "turnover_amount",
        "label": "双边成交金额",
        "unit_by_class": {
            "production": "currency",
            "research": "currency",
            "calibration_only": None,
        },
        "note": "买+卖金额，非换手率；窗口闭区间求和",
    },
    {
        "name": "cost_amount",
        "label": "成本",
        "unit_by_class": {
            "production": "currency",
            "research": "currency",
            "calibration_only": None,
        },
        "note": "金额口径；不等于 NAV 贡献，无成本率",
    },
    {
        "name": "topk_positive_day_ratio",
        "label": "TopK 正日占比",
        "unit_by_class": {
            "production": None,
            "research": "ratio",
            "calibration_only": None,
        },
        "note": "仅旧 WF summary KEY_Top20_hit_rate；规范化 runs 固定 not_available",
    },
)
#: 指标名有序元组（键集守恒的期望值）
METRIC_NAMES: tuple[str, ...] = tuple(str(item["name"]) for item in METRICS)
#: 单位展示文案（固定，不受 locale 影响）
UNIT_LABELS: dict[str, str] = {
    "percent_natural_day": "%（自然日复合年化）",
    "decimal_252d": "小数（252 日 CAGR）",
    "decimal": "小数（非正）",
    "pp": "pp（log-MDD 换算百分点）",
    "currency": "金额",
    "ratio": "比例",
}

#: run_index.csv 列集（设计 §5.7 原文顺序；``index_mode`` 恒为 ``full``）
RUN_INDEX_COLUMNS: tuple[str, ...] = (
    "run_key",
    "run_id",
    "batch_id",
    "campaign_id",
    "evidence_class",
    "host_mode",
    "implementation",
    "window_start",
    "window_end",
    "date_semantics",
    "data_state_id",
    "code_state_id",
    "source_ref",
    "source_identity_sha256",
    "source_manifest_sha256",
    "source_sha256",
    "index_mode",
    "report_status",
    "degradation_reason_code",
)
#: run_index.csv 行键
RUN_INDEX_KEYS: tuple[str, ...] = ("run_key",)
#: trend.csv 列集（设计 §5.7 最小列 + 末尾 ``source_field`` 溯源列）
TREND_COLUMNS: tuple[str, ...] = (
    "run_key",
    "metric_name",
    "value",
    "unit",
    "window_start",
    "window_end",
    "date_semantics",
    "source_ref",
    "source_sha256",
    "report_status",
    "degradation_reason_code",
    "source_field",
)
#: trend.csv 行键
TREND_KEYS: tuple[str, ...] = ("run_key", "metric_name")


def _check_class(evidence_class: str) -> str:
    """证据类闭集校验（跨类即报错，不做隐式回退）。"""
    if evidence_class not in contracts.EVIDENCE_CLASSES:
        raise ValueError(f"未知证据类：{evidence_class!r}（闭集 {contracts.EVIDENCE_CLASSES}）")
    return evidence_class


def metric_spec(metric_name: str) -> Mapping[str, Any]:
    """按指标名取规格（未知指标即报错，禁止自由文本冒充指标名）。"""
    for item in METRICS:
        if item["name"] == metric_name:
            return item
    raise ValueError(f"未知趋势指标：{metric_name!r}（闭集 {METRIC_NAMES}）")


def metric_order(metric_name: str) -> int:
    """指标在稳定排序中的位次（未知即报错）。"""
    return METRIC_NAMES.index(str(metric_spec(metric_name)["name"]))


def unit_for(metric_name: str, evidence_class: str) -> str | None:
    """该证据类下该指标的单位；``None`` 表示该类**不生成**该指标（须留状态行）。"""
    spec = metric_spec(metric_name)
    return dict(spec["unit_by_class"])[_check_class(evidence_class)]


def unit_label(unit: str | None) -> str:
    """单位展示文案（未知单位原样返回，不猜）。"""
    if unit is None:
        return "不适用"
    return UNIT_LABELS.get(str(unit), str(unit))


def metrics_for(evidence_class: str) -> Sequence[Mapping[str, Any]]:
    """该类别的五类指标规格（顺序固定；每类都返回全部五项，不裁剪键集）。"""
    _check_class(evidence_class)
    return METRICS
