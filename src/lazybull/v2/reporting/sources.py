# -*- coding: utf-8 -*-
"""链式净值有限性门与 runs / 旧 WF 白名单读口（设计 §4.2 第 2 条 / §5.3）。

硬口径：
- 链式指标**只调** ``calculate_chain_metrics``（唯一实现）；调用前逐行验证日期可解析、
  NAV 有限且为正，任一坏行 ⇒ 该臂 ``invalid/nonfinite_chain_nav``，
  **禁止**让函数静默过滤坏行后继续出指标；
- 合法输入须 ``rows_used = rows_total = N`` 且 ``trading_days = N - S``（S=折数）；
- 规范化 runs 的唯一读口是 ``load_runs_batch``（不另造 schema/指纹）；
- 旧 WF 只做**白名单文件身份读入**（解析与映射归 T3），缺失显式降级。
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics
from src.lazybull.v2.evidence.runs_loader import load_runs_batch
from src.lazybull.v2.reporting import status
from src.lazybull.v2.reporting.authority import normalize_date

__all__ = [
    "CHAIN_REQUIRED_COLUMNS",
    "LEGACY_WF_FILE_PATTERNS",
    "project_legacy_wf_source",
    "read_runs_batch_source",
    "select_legacy_wf_files",
    "verify_chain_rows",
]

#: 链式净值读取最小列（``split_index`` 可选，存在时用于折内日期单调校验）
CHAIN_REQUIRED_COLUMNS: tuple[str, ...] = ("date", "nav")
#: 旧 WF 白名单文件模式（仅身份读入；解析归 T3）
LEGACY_WF_FILE_PATTERNS: tuple[str, ...] = (
    "chain_nav_*.csv",
    "walk_forward_summary_*.csv",
)
_METRIC_KEYS = ("total_return", "cagr", "max_drawdown", "volatility", "sharpe")


def verify_chain_rows(
    chain_df: pd.DataFrame | None,
    *,
    expected_splits: int | None = None,
    allow_ordinal_date: bool = False,
) -> dict[str, Any]:
    """链式净值全行有效性门 + 唯一实现指标计算（坏行整臂拒绝）。

    ``allow_ordinal_date=True`` 用于旧 WF 产物：其 ``chain_nav_*.csv`` 的 ``date`` 是
    **折内序号**（整数）而非自然日，此时按“序数单调 + 折内唯一”校验；真实日期仍走
    ``normalize_date`` 优先解析。T1 驾驶舱不传该开关（保持严格日期口径）。
    """
    if chain_df is None:
        return status.make_status(
            "not_available", reason_code="chain_source_missing", detail="缺 chain_nav 来源"
        )
    missing = [name for name in CHAIN_REQUIRED_COLUMNS if name not in chain_df.columns]
    if missing:
        return status.make_status(
            "invalid", reason_code="missing_precondition", detail=f"chain_nav 缺必需列 {missing}"
        )
    if chain_df.empty:
        return status.make_status(
            "not_available", reason_code="chain_rows_empty", detail="chain_nav 无数据行"
        )
    rows = chain_df.to_dict("records")
    parsed, failure = _parse_chain_rows(rows, allow_ordinal_date=allow_ordinal_date)
    if failure is not None:
        return failure
    days, navs, splits = parsed
    split_count = len(set(splits))
    metrics = calculate_chain_metrics(
        pd.DataFrame({"date": days, "nav": navs, "split_index": splits})
    )
    details = {
        "rows_total": len(rows),
        "rows_used": len(days),
        "split_count": split_count,
        "metrics": {key: metrics.get(key) for key in _METRIC_KEYS},
        "trading_days": metrics.get("trading_days"),
    }
    return _judge_chain_metrics(
        metrics=metrics,
        details=details,
        rows_used=len(days),
        split_count=split_count,
        expected_splits=expected_splits,
    )


def _parse_chain_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_ordinal_date: bool = False,
) -> tuple[tuple[list[str], list[float], list[int]], dict[str, Any] | None]:
    """逐行校验并解析为 ``(days, navs, splits)``；任一坏行返回 ``(空, 失败状态行)``。"""
    days: list[str] = []
    navs: list[float] = []
    splits: list[int] = []
    last_by_split: dict[int, str] = {}
    for index, row in enumerate(rows):
        raw_date = row.get("date")
        day = normalize_date(raw_date)
        if day is None and allow_ordinal_date:
            day = _ordinal_date(raw_date)
        if day is None:
            return ([], [], []), status.make_status(
                "invalid",
                reason_code="chain_rows_unparsable",
                detail=f"第 {index} 行 date={raw_date!r} 不可解析",
            )
        try:
            nav = float(row.get("nav"))
        except (TypeError, ValueError):
            return ([], [], []), status.make_status(
                "invalid",
                reason_code="nonfinite_chain_nav",
                detail=f"第 {index} 行 nav={row.get('nav')!r} 不可解析",
            )
        if not _finite_positive(nav):
            return ([], [], []), status.make_status(
                "invalid",
                reason_code="nonfinite_chain_nav",
                detail=f"第 {index} 行 nav={nav!r} 非有限或非正（整臂拒绝，不静默过滤）",
            )
        split = _split_index(row, default=1)
        previous = last_by_split.get(split)
        if previous is not None and day == previous:
            return ([], [], []), status.make_status(
                "invalid",
                reason_code="duplicate_chain_date",
                detail=f"折 {split} 内同日重复行：{previous}",
            )
        if previous is not None and day < previous:
            return ([], [], []), status.make_status(
                "invalid",
                reason_code="date_regression",
                detail=f"折 {split} 内日期非严格递增：{previous} → {day}",
            )
        last_by_split[split] = day
        days.append(day)
        navs.append(nav)
        splits.append(split)
    return (days, navs, splits), None


def _ordinal_date(value: Any) -> str | None:
    """折内序号 → 可比较的定长字符串（旧 WF ``date`` 为整数序号时使用）。"""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return f"#{number:09d}"


def _judge_chain_metrics(
    *,
    metrics: Mapping[str, Any],
    details: Mapping[str, Any],
    rows_used: int,
    split_count: int,
    expected_splits: int | None,
) -> dict[str, Any]:
    """指标有限性 + 折数/交易日守恒判定（唯一实现输出只在此处判定）。"""
    bad = [key for key in _METRIC_KEYS if not _finite(metrics.get(key))]
    if bad:
        return status.make_status(
            "invalid",
            reason_code="chain_metric_nonfinite",
            detail=f"链式指标含非有限值：{bad}",
            extra=dict(details),
        )
    if expected_splits is not None and split_count != int(expected_splits):
        return status.make_status(
            "invalid",
            reason_code="count_mismatch",
            detail=f"折数 {split_count} 与声明 {expected_splits} 不一致",
            extra=dict(details),
        )
    trading_days = metrics.get("trading_days")
    if trading_days != rows_used - split_count:
        return status.make_status(
            "invalid",
            reason_code="count_mismatch",
            detail=(
                f"trading_days={trading_days} 与 rows_used-split_count="
                f"{rows_used - split_count} 不一致（折界重复点须剔除）"
            ),
            extra=dict(details),
        )
    return status.make_status(
        "available",
        detail=f"链式净值 {rows_used} 行全有效，指标经唯一实现计算",
        verification_level="authority_verified",
        extra=dict(details),
    )


def _finite(value: Any) -> bool:
    """有限浮点判定（None/NaN/Inf 均 False）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _finite_positive(value: Any) -> bool:
    """有限且为正（NAV 必须 > 0）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return _finite(number) and number > 0


def _split_index(row: Mapping[str, Any], *, default: int) -> int:
    """折序号（缺列/不可解析时全部归入同一折，退化为单折口径）。"""
    value = row.get("split_index")
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_runs_batch_source(batch_dir: Path | str) -> dict[str, Any]:
    """规范化 runs 批次只读读入（唯一读口 ``load_runs_batch``；失败显式 invalid）。"""
    path = Path(batch_dir)
    if not path.is_dir():
        return status.make_status(
            "not_available", reason_code="no_source", detail=f"runs 批次目录不存在：{path}"
        )
    try:
        batch = load_runs_batch(path)
    except (FileNotFoundError, ValueError) as exc:
        return status.make_status(
            "invalid",
            reason_code="missing_precondition",
            detail=f"load_runs_batch 拒绝：{type(exc).__name__}: {exc}",
        )
    chain = verify_chain_rows(batch.chain_nav, expected_splits=len(batch.folds))
    return status.make_status(
        chain.get("report_status", "invalid"),
        reason_code=chain.get("degradation_reason_code"),
        detail=chain.get("detail"),
        verification_level="authority_verified",
        extra={
            "batch_id": str(batch.batch_id),
            "fold_count": len(batch.folds),
            "chain": chain,
        },
    )


def select_legacy_wf_files(names: Iterable[str]) -> dict[str, list[str]]:
    """按白名单模式切分旧 WF 文件名（返回模式 → 命中文件名升序）。"""
    selected: dict[str, list[str]] = {}
    for pattern in LEGACY_WF_FILE_PATTERNS:
        selected[pattern] = sorted(
            str(name) for name in names if fnmatch.fnmatchcase(str(name), pattern)
        )
    return selected


def project_legacy_wf_source(
    *,
    names: Sequence[str],
    receipts: Mapping[str, Any],
    root_id: str,
) -> dict[str, Any]:
    """旧 WF 白名单读口投影：只登记实际命中的文件身份，缺一即显式降级。"""
    selected = select_legacy_wf_files(names)
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for pattern, matched in selected.items():
        if not matched:
            missing.append(pattern)
            continue
        for name in matched:
            receipt = receipts.get(f"{root_id}:{name}")
            if receipt is None:
                missing.append(f"{name}（未纳入稳定快照）")
                continue
            files.append({"path": f"{root_id}:{name}", "sha256": receipt.sha256,
                          "size": receipt.size})
    if not files:
        return status.make_status(
            "not_available",
            reason_code="no_source",
            detail=f"旧 WF 目录未命中白名单文件：{sorted(selected)}",
        )
    return status.make_status(
        "degraded" if missing else "available",
        reason_code="no_source" if missing else None,
        detail=(
            f"白名单命中 {len(files)} 个文件；缺 {missing}"
            if missing
            else f"白名单命中 {len(files)} 个文件"
        ),
        extra={"files": files, "missing": missing},
    )
