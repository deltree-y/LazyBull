# -*- coding: utf-8 -*-
"""趋势库观测装配（P6 T4）：新 run → 索引行 + 五类指标行。

硬口径（设计 §5.5 / §5.7）：

- 每个 run 恰一行索引 + 恰五条指标行（缺失/不可用也留状态行，键集守恒）；
- 指标只调既有唯一实现：``experiment_sources.validate_chain_rows``（research 链式）、
  ``trend_metrics.paper_annualized``（production 年化）、
  ``trend_metrics.legacy_topk_mean``（旧 WF TopK）、bundle 报告的 log-MDD/pp 原值；
- 金额按同窗口求和：缺折**不部分求和**（宁缺不补 0）；
- 每行的 ``source_ref``/``source_sha256`` 指向**实际来源文件**；无指标值时回落到身份来源；
- 本模块不在磁盘上做 I/O（只读快照副本），不写任何文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import math

import pandas as pd
import yaml

from scripts.compare.fold_subset import load_run
from src.lazybull.v2.evidence.runs_loader import load_runs_batch
from src.lazybull.v2.reporting import authority, bundle, cockpit, status, trend_contract
from src.lazybull.v2.reporting import trend_metrics
from src.lazybull.v2.reporting.authority import normalize_date
from src.lazybull.v2.reporting.snapshot import StableSnapshot
from src.lazybull.v2.store.store_identity import canonical_json_sha256
from src.lazybull.v2.reporting.trend_sources import (
    BUNDLE_REPORT,
    RUN_RECORD_PATTERN,
    RunObservation,
    RunSeed,
    _rel,
    _receipt_sha,
)

__all__ = ["build_observations"]

_UNUSABLE_LABELS = ("degraded", "not_available", "invalid", "stale")


def build_observations(
    snapshot: StableSnapshot,
    plan: Mapping[str, Any],
    fresh: Sequence[RunSeed],
) -> list[RunObservation]:
    """为**新 run** 计算身份行与五类指标行（继承行不重算）。"""
    observations: list[RunObservation] = []
    paper = _paper_context(snapshot, plan) if plan.get("evidence_class") == "production" else None
    if paper is not None:
        paper["authority_gate"] = _authority_record_gate(snapshot, plan)
    for seed in fresh:
        if seed.kind == "runs":
            observations.append(_observe_runs(snapshot, seed))
        elif seed.kind == "legacy_wf":
            observations.append(_observe_legacy(snapshot, seed))
        elif seed.kind == "p4_paper":
            observations.append(_observe_paper(snapshot, seed, paper or {}))
        else:
            observations.append(_observe_bundle(snapshot, seed))
    return observations


def _index_row(
    seed: RunSeed,
    evidence_class: str,
    *,
    status_row: Mapping[str, Any],
    **fields: Any,
) -> dict[str, Any]:
    """组装 run_index 行（身份与状态列由本函数统一填）。"""
    return {
        "run_key": seed.run_key,
        "run_id": fields.get("run_id"),
        "batch_id": fields.get("batch_id"),
        "campaign_id": fields.get("campaign_id"),
        "evidence_class": evidence_class,
        "host_mode": fields.get("host_mode"),
        "implementation": fields.get("implementation"),
        "window_start": fields.get("window_start"),
        "window_end": fields.get("window_end"),
        "date_semantics": fields.get("date_semantics") or "natural_day",
        "data_state_id": fields.get("data_state_id"),
        "code_state_id": fields.get("code_state_id"),
        "source_ref": seed.key_ref,
        "source_identity_sha256": seed.identity_sha,
        "source_manifest_sha256": seed.manifest_sha,
        "source_sha256": seed.identity_sha,
        "index_mode": trend_contract.INDEX_MODE_FULL,
        "report_status": status_row.get("report_status"),
        "degradation_reason_code": status_row.get("degradation_reason_code"),
    }


def _metric_row(
    seed: RunSeed,
    metric_name: str,
    *,
    value: float | None,
    unit: str | None,
    status_row: Mapping[str, Any],
    window_start: str | None,
    window_end: str | None,
    date_semantics: str,
    source_ref: str | None,
    source_sha256: str | None,
    source_field: str | None,
) -> dict[str, Any]:
    """组装一条 trend 行（``value`` 为空时仍带状态与可回溯来源）。"""
    return {
        "run_key": seed.run_key,
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "window_start": window_start,
        "window_end": window_end,
        "date_semantics": date_semantics,
        "source_ref": source_ref or seed.identity_ref,
        "source_sha256": source_sha256 or seed.identity_sha,
        "report_status": status_row.get("report_status"),
        "degradation_reason_code": status_row.get("degradation_reason_code"),
        "source_field": source_field,
    }


def _degradation(field_path: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """manifest ``degradations[]`` 的唯一构造口径（复用 ``status``）。"""
    return status.degradation_entry(row, field_path=field_path)


def _default_metric_status(
    evidence_class: str, metric: str, implementation: str
) -> dict[str, Any]:
    """该类/该来源下“不生成该指标”的显式状态行（禁止静默缺行）。"""
    if metric == "max_drawdown" and evidence_class == "production":
        return trend_metrics.status_row(
            "not_available",
            "no_unique_drawdown_series",
            detail="production 无纸面回撤序列唯一实现（不复制 cummax、不补造）",
        )
    if metric == "topk_positive_day_ratio" and implementation == "runs":
        return trend_metrics.status_row(
            "not_available",
            "no_unique_implementation",
            detail="规范化 runs 无 TopK 正日占比唯一实现；禁止从 topk_detail 私算",
        )
    return trend_metrics.status_row(
        "not_available", "no_source", detail=f"该类/该来源不生成 {metric}（设计 §5.5）"
    )


def _metric_rows(
    seed: RunSeed,
    evidence_class: str,
    *,
    implementation: str,
    window_start: str | None,
    window_end: str | None,
    date_semantics: str,
    results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """五条指标行（缺结果时按类默认状态补齐，键集恒完整）。"""
    rows: list[dict[str, Any]] = []
    for metric in trend_contract.METRIC_NAMES:
        unit = trend_contract.unit_for(metric, evidence_class)
        result = dict(results.get(metric) or {})
        has_result = bool(result)
        row_status = dict(result.get("status") or {})
        if not row_status:
            row_status = _default_metric_status(evidence_class, metric, implementation)
        # ``value_usable``：区级 degraded 但当前点已核对（如历史断线），数值仍可发布
        usable = status.is_usable(row_status) or bool(result.get("value_usable"))
        rows.append(
            _metric_row(
                seed,
                metric,
                value=result.get("value") if usable else None,
                unit=unit if usable else None,
                status_row=row_status,
                window_start=window_start,
                window_end=window_end,
                date_semantics=date_semantics,
                source_ref=result.get("ref"),
                source_sha256=result.get("sha"),
                # 无指标来源时 ref/sha 指向 run 身份来源（复合摘要 ⇒ 目录级），
                # 用 ``identity`` 显式标注该 sha 不是单文件 SHA（T4-R1-03）。
                source_field=result.get("field") or (None if has_result else "identity"),
            )
        )
    return rows


def _status_observation(
    seed: RunSeed,
    evidence_class: str,
    reason_row: Mapping[str, Any],
    *,
    metric_status: str | None = None,
) -> RunObservation:
    """不可用 run：仍保留恰好一行索引 + 五条状态行（键集守恒）。"""
    index_row = _index_row(
        seed,
        evidence_class,
        status_row=reason_row,
        run_id=seed.run_dir_name,
        implementation=seed.kind,
        window_start=None,
        window_end=None,
        date_semantics="natural_day",
    )
    metrics = _metric_rows(
        seed,
        evidence_class,
        implementation=seed.kind,
        window_start=None,
        window_end=None,
        date_semantics="natural_day",
        results={},
    )
    for row in metrics:
        row["report_status"] = metric_status or reason_row.get("report_status")
        row["degradation_reason_code"] = reason_row.get("degradation_reason_code")
    return RunObservation(
        index_row=index_row,
        metric_rows=metrics,
        degradations=[_degradation(f"runs.{seed.run_key}", reason_row)],
    )


def _collect_metric_degradations(
    seed: RunSeed, metrics: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """把不可用指标行登记进 manifest ``degradations[]``（每 run 每指标一条）。"""
    entries: list[dict[str, Any]] = []
    for row in metrics:
        if status.is_usable(row):
            continue
        if str(row.get("degradation_reason_code")) == "no_source":
            continue  # 该类本就不生成的指标属契约设计，不登记为降级
        entries.append(_degradation(f"trend.{seed.run_key}.{row.get('metric_name')}", row))
    return entries


def _run_status(
    primary: Mapping[str, Any],
    turnover: Mapping[str, Any] | None,
    cost: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """run 级状态（``turnover``/``cost`` 为 ``None`` 表示该类无该来源，不算降级）。"""
    rows = [dict(primary)]
    for item in (turnover, cost):
        if item is not None:
            rows.append(dict(item["status"]))
    severity = {"available": 0, "empty_valid": 0, "stale": 1, "degraded": 2,
                "not_available": 3, "invalid": 4}
    worst = max(rows, key=lambda item: severity.get(str(item.get("report_status")), 3))
    word = str(worst.get("report_status") or "not_available")
    if word == "invalid":
        return dict(worst)
    if word in _UNUSABLE_LABELS:
        return trend_metrics.status_row(
            "degraded",
            str(worst.get("degradation_reason_code") or "missing_precondition"),
            detail=f"run 可用但存在降级项：{worst.get('detail')}",
            source_ref=worst.get("source_ref"),
        )
    return trend_metrics.status_row("available", None, detail="来源齐备且契约校验通过")


def _fold_money(
    fold_frames: Sequence[tuple[int, pd.DataFrame]],
    column: str,
    windows: Mapping[int, tuple[str, str]],
    *,
    source_ref: str | None,
    source_sha: str | None,
    source_field: str,
) -> dict[str, Any]:
    """逐折按**该折自己的窗口**汇总金额（缺折/未知日期/窗口不全 ⇒ 不求和）。

    R3-04/R3-04-R1：金额必须与行的窗口自洽，且逐折用 ``summary.split_index`` 对应窗口
    （不能把全批外包窗口当成每折 OOS 窗口）。
    R3-N2：筛选前先校验所有消费行日期可解析；存在未知日期即整项降级，不把无法定位的
    金额当成“窗内无成交”。
    R3-03：``ref/sha/field`` 绑定实际消费的逐折文件集合（不再回落到身份 manifest）。
    """
    if not windows:
        return {
            "status": trend_metrics.status_row(
                "not_available",
                "missing_precondition",
                detail="summary 缺可解析的逐折窗口（不求和）",
                source_ref=source_ref,
            )
        }
    by_split = dict(fold_frames)
    missing = sorted(set(windows) - set(by_split))
    if missing:
        return {
            "status": trend_metrics.status_row(
                "not_available",
                "fold_file_missing",
                detail=f"缺折来源或窗口对应：{missing[:5]}（不部分求和）",
                source_ref=source_ref,
            )
        }
    bounded: list[pd.DataFrame] = []
    for split, frame in sorted(by_split.items()):
        if split not in windows:
            return {
                "status": trend_metrics.status_row(
                    "not_available",
                    "missing_precondition",
                    detail=f"折 {split} 缺 summary 窗口（不把全批窗口当本折窗口）",
                    source_ref=source_ref,
                )
            }
        start, end = windows[split]
        if "trade_date" not in frame.columns:
            return {
                "status": trend_metrics.status_row(
                    "not_available",
                    "missing_precondition",
                    detail="逐折来源缺 trade_date，无法证明金额落在声明窗口内",
                    source_ref=source_ref,
                )
            }
        days = frame["trade_date"].map(normalize_date)
        if days.isna().any():  # R3-N2：未知日期不得静默排除
            return {
                "status": trend_metrics.status_row(
                    "not_available",
                    "missing_precondition",
                    detail=f"折 {split} 存在不可解析日期（无法定位是否在窗口内，不求和）",
                    source_ref=source_ref,
                )
            }
        bounded.append(frame[(days >= start) & (days <= end)])
    result = trend_metrics.money_total(bounded, column)
    result.update({"ref": source_ref, "sha": source_sha, "field": source_field})
    return result


def _consumed_binding(
    snapshot: StableSnapshot, seed: RunSeed, suffix: str, field: str
) -> tuple[str, str]:
    """实际消费的逐折文件集合 → (run 目录 ref, 路径+字节 canonical SHA)。"""
    files = [name for name in seed.metric_files if name.endswith(suffix)]
    rows = []
    for name in files:
        receipt = snapshot.receipts.get(f"{seed.root_id}:{name}")
        if receipt is not None:
            rows.append({"path": name, "sha256": receipt.sha256})
    ref = f"{seed.root_id}:{seed.relative_run_path or '.'}"
    sha = canonical_json_sha256(sorted(rows, key=lambda item: item["path"]))
    return ref, sha


def _windows_from_summary(
    summary: pd.DataFrame | None,
) -> dict[int, tuple[str, str]]:
    """逐折窗口 ``{split_index: (start, end)}``（缺 split 或日期不可解析即空，不猜）。"""
    if summary is None or summary.empty:
        return {}
    if "split_index" not in summary.columns:
        return {}
    windows: dict[int, tuple[str, str]] = {}
    for row in summary.to_dict("records"):
        try:
            split = int(row.get("split_index"))
        except (TypeError, ValueError):
            return {}
        start = normalize_date(row.get("test_start"))
        end = normalize_date(row.get("test_end"))
        if start is None or end is None:
            return {}
        windows[split] = (start, end)
    return windows


def _window_from_summary(summary: pd.DataFrame | None) -> tuple[str | None, str | None]:
    """summary 的 ``test_start/test_end`` 最小/最大值（不可解析即 ``None``）。"""
    if summary is None or summary.empty:
        return None, None
    if "test_start" not in summary.columns or "test_end" not in summary.columns:
        return None, None
    starts = [normalize_date(value) for value in summary["test_start"].tolist()]
    ends = [normalize_date(value) for value in summary["test_end"].tolist()]
    if any(item is None for item in starts) or any(item is None for item in ends):
        return None, None
    return min(starts), max(ends)  # type: ignore[type-var]


def _observe_runs(snapshot: StableSnapshot, seed: RunSeed) -> RunObservation:
    """规范化 runs：契约读入 → 链式指标 / 金额（金额要求全折齐备）。"""
    copy_root = snapshot.root_copy(seed.root_id)
    if seed.relative_run_path:
        copy_root = copy_root / seed.relative_run_path
    try:
        batch = load_runs_batch(copy_root)
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        row = trend_metrics.status_row(
            "invalid",
            "source_contract_violation",
            detail=f"runs 契约读入失败：{type(exc).__name__}: {exc}",
            source_ref=seed.key_ref,
        )
        return _status_observation(seed, "research", row)
    window_start, window_end = _window_from_summary(batch.summary)
    chain_rel = _rel(seed.relative_run_path, "chain_nav.parquet")
    chain_ref = f"{seed.root_id}:{chain_rel}"
    chain_sha = _receipt_sha(snapshot, seed.root_id, chain_rel)
    chain = trend_metrics.chain_stats(batch.chain_nav)
    windows = _windows_from_summary(batch.summary)
    daily_pairs = [
        (split, fold.daily) for split, fold in batch.folds.items() if fold.daily is not None
    ]
    trade_pairs = [
        (split, fold.trades) for split, fold in batch.folds.items() if fold.trades is not None
    ]
    turnover_ref, turnover_sha = _consumed_binding(snapshot, seed, "/daily.parquet", "turnover")
    cost_ref, cost_sha = _consumed_binding(snapshot, seed, "/trades.parquet", "cost")
    turnover = _fold_money(
        daily_pairs,
        "turnover_amount",
        windows,
        source_ref=turnover_ref,
        source_sha=turnover_sha,
        source_field="folds/*/daily.parquet:turnover_amount",
    )
    cost = _fold_money(
        trade_pairs,
        "cost",
        windows,
        source_ref=cost_ref,
        source_sha=cost_sha,
        source_field="folds/*/trades.parquet:cost",
    )
    results = {
        "annualized_return": {
            "value": chain.get("cagr"),
            "status": chain["status"],
            "ref": chain_ref,
            "sha": chain_sha,
            "field": "chain_nav.parquet:cagr",
        },
        "max_drawdown": {
            "value": chain.get("max_drawdown"),
            "status": chain["status"],
            "ref": chain_ref,
            "sha": chain_sha,
            "field": "chain_nav.parquet:max_drawdown",
        },
        "turnover_amount": turnover,
        "cost_amount": cost,
    }
    status_row = _run_status(chain["status"], turnover, cost)
    data_state = dict(batch.batch_meta.get("data_state") or {})
    code_state = dict(batch.batch_meta.get("code_state") or {})
    index_row = _index_row(
        seed,
        "research",
        status_row=status_row,
        run_id=str(batch.batch_id),
        batch_id=str(batch.batch_id),
        host_mode=str(batch.batch_meta.get("host_mode") or "") or None,
        implementation="runs",
        window_start=window_start,
        window_end=window_end,
        date_semantics="natural_day",
        data_state_id=data_state.get("data_state_id"),
        code_state_id=code_state.get("git_commit"),
    )
    metrics = _metric_rows(
        seed,
        "research",
        implementation="runs",
        window_start=window_start,
        window_end=window_end,
        date_semantics="natural_day",
        results=results,
    )
    degradations = _collect_metric_degradations(seed, metrics)
    if status_row.get("report_status") == "degraded":
        degradations.append(_degradation(f"runs.{seed.run_key}", status_row))
    return RunObservation(index_row=index_row, metric_rows=metrics, degradations=degradations)


def _observe_legacy(snapshot: StableSnapshot, seed: RunSeed) -> RunObservation:
    """旧 WF：白名单子集读取（chain + summary + 数据态），TopK 只做字段改名。"""
    prefix = seed.relative_run_path
    copy_root = snapshot.root_copy(seed.root_id)
    if prefix:
        copy_root = copy_root / prefix
    chain_files = sorted(
        name for name in snapshot.list_dir(seed.root_id, prefix) if name.startswith("chain_nav_")
    )
    summary_files = sorted(
        name
        for name in snapshot.list_dir(seed.root_id, prefix)
        if name.startswith("walk_forward_summary_")
    )
    if len(chain_files) != 1 or len(summary_files) != 1:
        row = trend_metrics.status_row(
            "invalid",
            "multiple_source_files" if (len(chain_files) > 1 or len(summary_files) > 1)
            else "required_source_missing",
            detail=f"旧 WF 候选不唯一/缺失：chain={chain_files} summary={summary_files}",
            source_ref=seed.key_ref,
        )
        return _status_observation(seed, "research", row)
    chain_name = chain_files[0]
    summary_name = summary_files[0]
    try:
        artifacts = load_run(copy_root, label=seed.run_dir_name)
        summary = pd.read_csv(copy_root / summary_name, encoding="utf-8-sig")
    except (OSError, ValueError, KeyError, AttributeError, UnicodeDecodeError) as exc:
        row = trend_metrics.status_row(
            "invalid",
            "source_contract_violation",
            detail=f"旧 WF 读取失败：{type(exc).__name__}: {exc}",
            source_ref=seed.key_ref,
        )
        return _status_observation(seed, "research", row)
    window_start, window_end = _window_from_summary(summary)
    chain_rel = _rel(prefix, chain_name)
    chain_ref = f"{seed.root_id}:{chain_rel}"
    chain_sha = _receipt_sha(snapshot, seed.root_id, chain_rel)
    chain = trend_metrics.chain_stats(artifacts.chain)
    topk = trend_metrics.legacy_topk_mean(summary)
    results = {
        "annualized_return": {
            "value": chain.get("cagr"),
            "status": chain["status"],
            "ref": chain_ref,
            "sha": chain_sha,
            "field": f"{chain_name}:cagr",
        },
        "max_drawdown": {
            "value": chain.get("max_drawdown"),
            "status": chain["status"],
            "ref": chain_ref,
            "sha": chain_sha,
            "field": f"{chain_name}:max_drawdown",
        },
        "topk_positive_day_ratio": {
            "value": topk.get("value"),
            "status": topk["status"],
            "ref": f"{seed.root_id}:{_rel(prefix, summary_name)}",
            "sha": _receipt_sha(snapshot, seed.root_id, _rel(prefix, summary_name)),
            "field": f"{summary_name}:{trend_metrics.TOP20_COLUMN}",
        },
    }
    # 旧 WF 无 daily/trades 白名单来源：金额按契约缺省（不登记为 run 级降级）
    status_row = _run_status(chain["status"], None, None)
    index_row = _index_row(
        seed,
        "research",
        status_row=status_row,
        run_id=seed.run_dir_name,
        batch_id=seed.run_dir_name,
        host_mode="backtest",
        implementation="legacy_wf",
        window_start=window_start,
        window_end=window_end,
        date_semantics="natural_day",
        data_state_id=_legacy_state_id(snapshot, seed),
    )
    metrics = _metric_rows(
        seed,
        "research",
        implementation="legacy_wf",
        window_start=window_start,
        window_end=window_end,
        date_semantics="natural_day",
        results=results,
    )
    return RunObservation(
        index_row=index_row,
        metric_rows=metrics,
        degradations=_collect_metric_degradations(seed, metrics),
    )


def _legacy_state_id(snapshot: StableSnapshot, seed: RunSeed) -> str | None:
    """旧 WF 数据态（唯一候选才取；多候选留空，不猜）。"""
    names = sorted(
        name
        for name in snapshot.list_dir(seed.root_id, seed.relative_run_path)
        if name.startswith("data_state_") and name.endswith(".json")
    )
    if len(names) != 1:
        return None
    payload, reason = snapshot.read_json(seed.root_id, _rel(seed.relative_run_path, names[0]))
    if reason is not None or not isinstance(payload, Mapping):
        return None
    value = payload.get("data_state_id")
    return None if value is None else str(value)


def _paper_context(snapshot: StableSnapshot, plan: Mapping[str, Any]) -> dict[str, Any]:
    """P4 共享输入（config/nav/trades）：只在计划内捕获一次。"""
    spec = (plan.get("specs") or [None])[0]
    if spec is None:
        return {}
    nav_path = snapshot.copy_path(spec.root_id, "nav/nav.parquet")
    trades_path = snapshot.copy_path(spec.root_id, "trades/trades.parquet")
    nav = pd.read_parquet(nav_path) if nav_path.is_file() else None
    trades = pd.read_parquet(trades_path) if trades_path.is_file() else None
    initial, start_date = _paper_config(snapshot, spec.root_id)
    return {
        "root_id": spec.root_id,
        "nav": nav,
        "trades": trades,
        "initial_capital": initial,
        "account_start_date": start_date,
        "nav_ref": f"{spec.root_id}:nav/nav.parquet" if nav_path.is_file() else None,
        "nav_sha": _receipt_sha(snapshot, spec.root_id, "nav/nav.parquet"),
        "trades_ref": f"{spec.root_id}:trades/trades.parquet" if trades_path.is_file() else None,
        "trades_sha": _receipt_sha(snapshot, spec.root_id, "trades/trades.parquet"),
    }


def _paper_config(
    snapshot: StableSnapshot, root_id: str
) -> tuple[Any, str | None]:
    """P4 配置：只取 initial_capital 与 account_start_date（展平前冲突检测）。"""
    payload = snapshot.read_bytes(root_id, "config.yaml")
    if payload is None:
        return None, None
    try:
        parsed = yaml.safe_load(payload.decode("utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError):
        return None, None
    if not isinstance(parsed, Mapping):
        return None, None
    from src.lazybull.v2.reporting.config_view import resolve_account_start_date

    start_date, conflict = resolve_account_start_date(parsed)
    if conflict is not None:
        return None, None
    return parsed.get("initial_capital"), start_date


def _authority_nav_rows(nav: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    """持久 NAV → 权威核对行（``date/cash/position_value/total_value``；不可解析即 ``None``）。"""
    if nav is None or "trade_date" not in getattr(nav, "columns", []):
        return None
    required = ("cash", "position_value", "total_value")
    if any(name not in nav.columns for name in required):
        return None
    rows: list[dict[str, Any]] = []
    for record in nav.to_dict("records"):
        day = normalize_date(record.get("trade_date"))
        if day is None:
            return None
        rows.append(
            {
                # 持久 NAV 行按权威核对函数期望的列名（trade_date/cash/position_value/total_value）
                "trade_date": day,
                "cash": record.get("cash"),
                "position_value": record.get("position_value"),
                "total_value": record.get("total_value"),
                "nav": record.get("nav"),
            }
        )
    return rows


def _authority_record_gate(
    snapshot: StableSnapshot, plan: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """当前权威代对应的 run 记录必须存在（缺件不得让同日其它相位“顶替”）。

    只返回 gate 状态行；权威本身不可用时返回 ``None``（由逐 run 的权威门处理）。
    """
    row, _state, generation = cockpit.load_authority(snapshot)
    if generation is None or not status.is_usable(row):
        return None
    expected = f"{generation['run_kind']}_{generation['trade_date']}.json"
    names = {seed.run_dir_name for seed in plan.get("seeds") or []}
    if expected in names:
        return None
    return trend_metrics.status_row(
        "degraded",
        "authority_missing",
        detail=f"缺当前权威代记录 runs/{expected}（不允许同日其它相位顶替）",
    )


def _paper_has_trades(trades: pd.DataFrame | None, day: str) -> bool:
    """该业务日是否有成交（权威门用；缺来源视为无成交）。"""
    if trades is None or "trade_date" not in getattr(trades, "columns", []):
        return False
    return bool((trades["trade_date"].map(normalize_date) == day).any())


def _observe_paper(
    snapshot: StableSnapshot, seed: RunSeed, paper: Mapping[str, Any]
) -> RunObservation:
    """P4 纸面 run：run 记录为身份，年化/金额取窗口内共享来源。"""
    payload, reason = snapshot.read_json(seed.root_id, seed.relative_run_path)
    day = RUN_RECORD_PATTERN.match(Path(seed.relative_run_path).name)
    run_day = day.group(1) if day else None
    if reason is not None or not isinstance(payload, Mapping) or run_day is None:
        row = trend_metrics.status_row(
            "invalid",
            "invalid_json",
            detail="run 记录不是对象或文件名日期不可解析",
            source_ref=seed.key_ref,
        )
        return _status_observation(seed, "production", row)
    if str(payload.get("trade_date")) != run_day:
        row = trend_metrics.status_row(
            "invalid",
            "count_mismatch",
            detail=f"run 记录 trade_date={payload.get('trade_date')!r} ≠ 文件名 {run_day}",
            source_ref=seed.key_ref,
        )
        return _status_observation(seed, "production", row)
    gate = paper.get("authority_gate")
    if gate is not None:
        return _status_observation(seed, "production", gate, metric_status="not_available")
    # 权威门（R3-01）：production 数值必须可追溯到 P4 权威代（paper_generation）与
    # NAV 三原值核对；缺权威/记录形态冲突/NAV 不符不得输出 available 年化。
    authority_row, engine_state, generation = cockpit.load_authority(snapshot)
    if engine_state is None or generation is None or not status.is_usable(authority_row):
        row = trend_metrics.status_row(
            "degraded",
            "authority_missing",
            detail=f"production 权威不可用：{authority_row.get('detail')}",
            source_ref=seed.key_ref,
        )
        return _status_observation(seed, "production", row, metric_status="not_available")
    # 历史记录按**自身文件名相位**校验形态；当前代才需与 paper_generation 对齐。
    # 把当前相位套到全部历史会把合法 t0 记录判成冲突（R3-01-R1）。
    record_kind = Path(seed.relative_run_path).name.split("_", 1)[0]
    shape = authority.classify_run_record(payload, run_kind=record_kind, trade_date=run_day)
    if not status.is_usable(shape):
        return _status_observation(seed, "production", shape)
    # 同日另一相位的幂等补写记录是原生宿主合法形态（t1 当前代 + recovery 补写 t0）：
    # 只按记录自身文件名相位校验，不因“与最新头不同”判冲突（R3-01-R2）。
    nav = paper.get("nav")
    nav_rows = _authority_nav_rows(nav)
    has_trades = _paper_has_trades(paper.get("trades"), run_day)
    # 只核验 ≤ run_day 的权威点：末日坏点不得反向抹去此前已核对的历史（R3-N1）。
    values = [
        dict(item)
        for item in engine_state.portfolio_values
        if normalize_date(item.get("date")) is not None
        and normalize_date(item.get("date")) <= run_day
    ]
    nav_rows = [row for row in nav_rows or [] if row["trade_date"] <= run_day]
    nav_verdict = authority.verify_nav_authority(
        portfolio_values=values,
        nav_rows=nav_rows,
        authority_date=run_day,
        current_day_has_trades=has_trades,
    )
    annual_allowed = bool(nav_verdict.get("annualization_allowed"))
    window_start = _paper_window_start(nav, run_day)
    if annual_allowed:
        annual = trend_metrics.paper_annualized(
            nav,
            run_day=run_day,
            initial_capital=paper.get("initial_capital"),
            account_start_date=paper.get("account_start_date"),
        )
        value = annual.get("value")
        finite = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
        if finite and status.is_usable(dict(annual.get("status") or {})):
            if not status.is_usable(nav_verdict):
                # 区级 degraded（历史断线）但当前点已核对：保留数值并标注区级降级（R3-N1-R1）
                annual["value_usable"] = True
                annual["status"] = nav_verdict
        else:
            # 年化函数自身失败（非法初资/缺起始日等）：保留其状态，不得用区级状态覆盖，
            # 也不得放行空值（R3-N1-R2）
            annual.pop("value_usable", None)
    else:
        # 权威未放行：不计算年化（保留权威状态与原因码，禁止以公式替代输入真实性门）
        annual = {"value": None, "status": nav_verdict}
    trades = _paper_trades_window(paper.get("trades"), window_start, run_day)
    absent = {
        "status": trend_metrics.status_row(
            "not_available", "no_source", detail="trades 来源缺失或窗口起点缺失"
        )
    }
    turnover = trend_metrics.money_total([trades], "amount") if trades is not None else absent
    cost = trend_metrics.money_total([trades], "total_cost") if trades is not None else absent
    results = {
        "annualized_return": {
            "value": annual.get("value"),
            "status": annual["status"],
            "value_usable": annual.get("value_usable"),
            "ref": paper.get("nav_ref"),
            "sha": paper.get("nav_sha"),
            "field": "nav.parquet:total_value",
        },
        "turnover_amount": {
            "value": turnover.get("value"),
            "status": turnover["status"],
            "ref": paper.get("trades_ref"),
            "sha": paper.get("trades_sha"),
            "field": "trades.parquet:amount",
        },
        "cost_amount": {
            "value": cost.get("value"),
            "status": cost["status"],
            "ref": paper.get("trades_ref"),
            "sha": paper.get("trades_sha"),
            "field": "trades.parquet:total_cost",
        },
    }
    status_row = _run_status(annual["status"], turnover, cost)
    index_row = _index_row(
        seed,
        "production",
        status_row=status_row,
        run_id=Path(seed.relative_run_path).stem,
        implementation="p4_paper",
        window_start=window_start,
        window_end=run_day,
        date_semantics="natural_day",
    )
    metrics = _metric_rows(
        seed,
        "production",
        implementation="p4_paper",
        window_start=window_start,
        window_end=run_day,
        date_semantics="natural_day",
        results=results,
    )
    return RunObservation(
        index_row=index_row,
        metric_rows=metrics,
        degradations=_collect_metric_degradations(seed, metrics),
    )


def _paper_window_start(nav: pd.DataFrame | None, run_day: str) -> str | None:
    """窗口起点 = 不晚于运行日的 NAV 首行日期（日期归一走 ``normalize_date``）。"""
    if nav is None or "trade_date" not in getattr(nav, "columns", []):
        return None
    days = sorted(
        day for day in nav["trade_date"].map(normalize_date).unique() if day and day <= run_day
    )
    return days[0] if days else None


def _paper_trades_window(
    trades: pd.DataFrame | None, window_start: str | None, run_day: str
) -> pd.DataFrame | None:
    """窗口闭区间内的成交行（缺窗口起点即不可聚合，返回 ``None``）。"""
    if trades is None or window_start is None or "trade_date" not in trades.columns:
        return None
    days = trades["trade_date"].map(normalize_date)
    return trades[(days >= str(window_start)) & (days <= str(run_day))]


def _observe_bundle(snapshot: StableSnapshot, seed: RunSeed) -> RunObservation:
    """P5b calibration bundle：先验封存，再读 log-MDD/pp（不二次换算）。"""
    copy_root = snapshot.root_copy(seed.root_id)
    verified = bundle.verify_p5b_bundle("t5_seq7", copy_root, scope_base=copy_root.parent)
    if not status.is_usable(verified):
        row = trend_metrics.status_row(
            str(verified.get("report_status") or "invalid"),
            str(verified.get("degradation_reason_code") or "bundle_seal_mismatch"),
            detail=f"P5b bundle 校验未通过：{verified.get('detail')}",
            source_ref=seed.identity_ref,
        )
        return _status_observation(seed, "calibration_only", row)
    report, reason = snapshot.read_json(seed.root_id, BUNDLE_REPORT)
    if reason is not None or not isinstance(report, Mapping):
        row = trend_metrics.status_row(
            "invalid", "invalid_json", detail=f"{BUNDLE_REPORT} 缺失或损坏", source_ref=seed.key_ref
        )
        return _status_observation(seed, "calibration_only", row)
    sample = report.get("sample_sources")
    window = dict(sample.get("b0_window") or {}) if isinstance(sample, Mapping) else {}
    mdd = window.get("mdd_percent")
    value = None
    if isinstance(mdd, (int, float)) and not isinstance(mdd, bool):
        value = float(mdd)
    window_start = _ordinal_window(window.get("first_key"))
    window_end = _ordinal_window(window.get("last_key"))
    if value is None:
        row_status = trend_metrics.status_row(
            "not_available",
            "missing_precondition",
            detail="bundle 报告缺 sample_sources.b0_window.mdd_percent（不换算、不猜）",
            source_ref=seed.key_ref,
        )
    else:
        row_status = trend_metrics.status_row(
            "available", None, detail="P5b 回撤裁判 log-MDD 换算的百分点原值（不二次换算）"
        )
    results = {
        "max_drawdown": {
            "value": value,
            "status": row_status,
            "ref": f"{seed.root_id}:{BUNDLE_REPORT}",
            "sha": _receipt_sha(snapshot, seed.root_id, BUNDLE_REPORT),
            "field": "sample_sources.b0_window.mdd_percent",
        }
    }
    index_row = _index_row(
        seed,
        "calibration_only",
        status_row=trend_metrics.status_row(
            "available", None, detail=f"sealed bundle 校验通过（{verified.get('family')}）"
        ),
        run_id=str(report.get("run_id") or Path(seed.run_dir_name).name),
        campaign_id=_campaign_name(seed),
        implementation="p5b_bundle",
        window_start=window_start,
        window_end=window_end,
        date_semantics="fold_local_ordinal",
    )
    metrics = _metric_rows(
        seed,
        "calibration_only",
        implementation="p5b_bundle",
        window_start=window_start,
        window_end=window_end,
        date_semantics="fold_local_ordinal",
        results=results,
    )
    return RunObservation(
        index_row=index_row,
        metric_rows=metrics,
        degradations=_collect_metric_degradations(seed, metrics),
    )


def _ordinal_window(key: Any) -> str | None:
    """折内序数窗口（``[fold, ordinal]`` → ``fold{fold}:{ordinal}``；不转自然日）。"""
    if not isinstance(key, (list, tuple)) or len(key) != 2:
        return None
    try:
        return f"fold{int(key[0])}:{int(key[1])}"
    except (TypeError, ValueError):
        return None


def _campaign_name(seed: RunSeed) -> str:
    """campaign 身份 = bundle 目录名（``reports`` 目录取其父名；不改源字节）。"""
    path = Path(seed.run_dir_name)
    return path.parent.name if path.name == "reports" else path.name
