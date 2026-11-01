# -*- coding: utf-8 -*-
"""就绪门结构化证据产物写入器（P4 后维护项 B 方案，reviewer2 复审 P1-C-1）。

从 ``host.py`` 拆出（复杂度契约 §4.4：新增功能对应新增文件）：

- ``runs/readiness_<date>.json``：通过日与失败日均持久化的独立证据产物
  （此前成功路径 ``PaperDayResult(readiness=None)`` 直接丢弃报告——通过日
  无可查询证据）；``runs/gap_<date>.json`` 为失败日兼容产物（payload 经
  ``build_readiness_payload`` 统一结构化，补 ``column_missing_rates`` 与
  coverage 证据字段）。
- 旁路纪律：诊断产物——写盘失败告警不阻断主链路，不影响账户状态（同 gap
  先例）；通过 / 失败 / 写盘失败 / 旁路四形态均不改变账户状态语义。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from loguru import logger

__all__ = [
    "append_quality_summary",
    "build_readiness_payload",
    "write_gap_record",
    "write_readiness_evidence",
]


def append_quality_summary(text: str, readiness) -> str:
    """推送文本追加数据质量摘要（决策 B2：钉钉呈现关键信息，内容简化）。

    结构化要素（reviewer3 P2-4）：阻断状态/原因、逐成员阈值、error/warning 数、
    top 异常列；明细列清单留在 runs 证据，避免消息超长。三档均不阻断日跑，
    仅聚合升级会置 `coverage_blocked=True`。
    """
    alerts = list(getattr(readiness, "coverage_alerts", []) or [])
    errors = [a for a in alerts if a.get("severity") == "error"]
    warnings = [a for a in alerts if a.get("severity") == "warning"]
    blocked = bool(getattr(readiness, "coverage_blocked", False))
    if not errors and not warnings and not blocked:
        return text
    top = sorted(
        [a for a in alerts if a.get("column") and a.get("rate") is not None],
        key=lambda item: -float(item["rate"]),
    )[:3]
    detail = ", ".join(f"{a['column']}={float(a['rate']):.1%}" for a in top)
    threshold = getattr(readiness, "coverage_error_threshold", None)
    line = f"数据质量: error {len(errors)} 列 / warning {len(warnings)} 列"
    if threshold:
        line += f"（阈值 {int(threshold)} 列）"
    if detail:
        line += f"（top: {detail}）"
    if blocked:
        reason = getattr(readiness, "coverage_blocked_reason", "") or ""
        line += f"；**聚合阻断**：{reason}" if reason else "；**聚合阻断**"
    return f"{text}\n{line}" if text else line


def build_readiness_payload(corrected: str, readiness) -> dict:
    """结构化就绪证据 payload（readiness / gap 两产物共用 schema 子集）。"""
    return {
        "date": corrected,
        "ok": bool(readiness.ok),
        "gaps": list(readiness.gaps),
        "row_count": int(readiness.row_count),
        "row_count_band": [str(v) for v in readiness.row_count_band],
        "uncovered_columns": list(readiness.uncovered_columns),
        "derived_family_note": readiness.derived_family_note,
        "column_missing_rates": {
            col: float(rate) for col, rate in readiness.column_missing_rates.items()
        },
        "coverage_quantiles": [float(q) for q in readiness.coverage_quantiles],
        "coverage_blocked": bool(readiness.coverage_blocked),
        "coverage_error_threshold": int(getattr(readiness, "coverage_error_threshold", 0)),
        "coverage_blocked_reason": getattr(readiness, "coverage_blocked_reason", ""),
        "rate_denominator": readiness.rate_denominator,
        "n_rows_used": int(readiness.n_rows_used),
        "n_rows_total": int(readiness.n_rows_total),
        "coverage_alerts": list(readiness.coverage_alerts),
        "coverage_members": list(readiness.coverage_members),
        "coverage_verdicts": list(readiness.coverage_verdicts),
        "coverage_extra_keys": list(readiness.coverage_extra_keys),
        "timestamp": pd.Timestamp.now().isoformat(),
    }


def _atomic_write_json(runs_path: Path, filename: str, payload: dict) -> Path:
    path = runs_path / filename
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


def write_readiness_evidence(runs_path: Path, corrected: str, readiness) -> None:
    """逐日结构化就绪证据（通过/失败均落盘；写盘失败告警不阻断）。"""
    payload = {"kind": "readiness_evidence", **build_readiness_payload(corrected, readiness)}
    try:
        path = _atomic_write_json(runs_path, f"readiness_{corrected}.json", payload)
    except OSError as exc:
        logger.error(f"就绪证据写盘失败（旁路，不阻断主链路）: {runs_path}（{exc}）")
        return
    logger.info(f"就绪证据已登记: {path}（ok={readiness.ok}）")


def write_gap_record(runs_path: Path, corrected: str, readiness) -> None:
    """就绪缺口登记（``runs/gap_<date>.json``，失败日兼容产物；写盘失败不阻断）。"""
    payload = {"kind": "readiness_gap", **build_readiness_payload(corrected, readiness)}
    try:
        path = _atomic_write_json(runs_path, f"gap_{corrected}.json", payload)
    except OSError as exc:
        logger.error(f"就绪缺口登记写盘失败（不阻断推迟语义）: {runs_path}（{exc}）")
        return
    logger.warning(f"就绪缺口已登记: {path}")
