# -*- coding: utf-8 -*-
"""P2b 三条比较边：关闭 vs P2a runs、关闭 vs 旧 B0 raw、开启 vs 关闭。

先核验批内身份、折集与 NAV 有限性，再复用 P2a 指纹/帧/净值比较原语。
运行身份只按明确列名规范化；模式键按新侧独有期望值或双测前后期望值处理。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v2_p2a.compare_replay_vs_b0 import (
    CompareReport,
    _frame_equal_exact,
    _values_equal,
    compare_fingerprint_gate,
    compare_replay_vs_b0,
)
from scripts.v2_p2b.check_frozen_inputs import (
    DEFAULT_B0_RAW,
    DEFAULT_P2A_RUNS,
    canonical_digest_note,
    digest_b0_raw_reference,
    digest_p2a_runs_reference,
    resolve_b0_raw_reference,
)
from scripts.v2_p2b.check_workspace_state import validate_workspace_digest
from scripts.v2_p2b.replay_merger import P2B_SOURCE
from src.lazybull.v2.evidence.fingerprint_keys import config_fingerprint_keys
from src.lazybull.v2.evidence.runs_loader import RunsBatch, RunsFold, load_runs_batch
from src.lazybull.v2.hosts.backtest.runtime import (
    LEGACY_BYPASS,
    MERGER_SLEEVE_ID,
    SINGLE_SLEEVE_PASSTHROUGH,
)

EDGES = ("edge1_closed_vs_p2a_runs", "edge2_closed_vs_b0_raw", "edge3_enabled_vs_closed")
IDENTITY_COLUMNS = ("batch_id", "wf_run_id", "batch_run_id")
P2B_EVIDENCE_NOTE_KEYS = frozenset(
    {
        "merger.mode",
        "merger.sleeve_id",
        "merger.call_count",
        "merger.call_count_by_fold",
        "workspace.digest",
        "environment.versions",
        "reference.p2a_runs.digest",
        "reference.b0_raw.digest",
    }
)


@dataclass
class MergerCompareReport:
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    edge_results: Dict[str, bool] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors and all(self.edge_results.get(edge) is True for edge in EDGES)

    def fail(self, edge: str, message: str) -> None:
        self.errors.append(f"{edge}: {message}")
        self.edge_results[edge] = False


def _check_identity_values(
    frame: pd.DataFrame, column: str, batch_id: str, label: str
) -> List[str]:
    if column not in frame.columns:
        return []
    bad = [pos for pos, value in enumerate(frame[column].tolist()) if str(value) != batch_id]
    return [f"{label} 列 {column} 与批次 {batch_id} 不一致（行 {bad[:10]}）"] if bad else []


def _validate_meta_config_projection(batch: RunsBatch) -> List[str]:
    """按 writer 口径从 summary 首行重建配置投影并核对 batch_meta.config。"""
    if batch.summary.empty:
        return []
    summary = batch.summary
    row = summary.iloc[0]
    fingerprint_columns = config_fingerprint_keys(summary.columns.tolist())
    expected = {
        key: (row[key].item() if hasattr(row[key], "item") else row[key])
        for key in fingerprint_columns
        if key in summary.columns and pd.notna(row[key])
    }
    actual = batch.batch_meta.get("config")
    if not isinstance(actual, dict):
        return ["batch_meta.config 不是 object，无法与 summary 首行投影核对"]
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    wrong = sorted(
        key for key in set(expected) & set(actual) if not _values_equal(expected[key], actual[key])
    )
    if missing or extra or wrong:
        return [
            "batch_meta.config 与 summary 首行投影不一致——"
            f"缺键={missing}，多键={extra}，错值键={wrong}"
        ]
    return []


def validate_batch_identity(batch: RunsBatch) -> List[str]:
    """目录、meta、summary 与逐折产物先完成批内身份核验。"""
    batch_id = batch.batch_id
    errors: List[str] = []
    if batch.directory.name != batch_id:
        errors.append(f"目录名 {batch.directory.name} != batch_id {batch_id}")
    if batch.batch_meta.get("batch_id") != batch_id:
        errors.append(f"batch_meta.batch_id != {batch_id}")
    summary = batch.summary
    if summary.empty:
        errors.append("summary 为空")
    if summary["split_index"].duplicated().any():
        errors.append("summary 存在重复 split_index 行")
    for column in IDENTITY_COLUMNS:
        errors += _check_identity_values(summary, column, batch_id, "summary")
    data_state = batch.batch_meta.get("data_state") or {}
    for key in ("wf_run_id", "batch_run_id"):
        if key in data_state and str(data_state[key]) != batch_id:
            errors.append(f"batch_meta.data_state.{key} != {batch_id}")
    for split, fold in batch.folds.items():
        errors += _validate_fold_identity(fold, batch_id, split)
    errors += _validate_meta_config_projection(batch)
    return [f"批次 {batch_id}: {message}" for message in errors]


def _validate_fold_identity(fold: RunsFold, batch_id: str, split: int) -> List[str]:
    errors: List[str] = []
    frames = {
        "trades": (fold.trades, "wf_run_id"),
        "attribution": (fold.attribution, "wf_run_id"),
        "holdings_snapshot": (fold.holdings_snapshot, "run_id"),
        "topk_detail": (fold.topk_detail, "wf_run_id"),
    }
    for name, (frame, column) in frames.items():
        if frame is not None:
            errors += _check_identity_values(frame, column, batch_id, f"split{split:02d}/{name}")
    for key in IDENTITY_COLUMNS:
        if key in fold.meta and str(fold.meta[key]) != batch_id:
            errors.append(f"split{split:02d}/_meta.{key} != {batch_id}")
    return errors


def _finite_frame_errors(
    frame: pd.DataFrame,
    *,
    batch_id: str,
    split: int,
    label: str,
    date_column: str,
) -> List[str]:
    errors: List[str] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        value = row["nav"]
        try:
            finite = bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            errors.append(
                f"批次 {batch_id} 折 split{split:02d} {label} 行={position} "
                f"日期/序号={row[date_column]!r} 列=nav 非有限值={value!r}"
            )
    return errors


def validate_nav_finite(batch: RunsBatch) -> List[str]:
    """chain_nav 与每折 daily 的 NAV 逐值有限性检查。"""
    errors: List[str] = []
    for split, segment in batch.chain_nav.groupby("split_index", sort=False):
        errors += _finite_frame_errors(
            segment,
            batch_id=batch.batch_id,
            split=int(split),
            label="chain_nav",
            date_column="date",
        )
    for split, fold in batch.folds.items():
        if fold.daily is not None:
            errors += _finite_frame_errors(
                fold.daily,
                batch_id=batch.batch_id,
                split=split,
                label="daily",
                date_column="trade_date",
            )
    return errors


def _validate_raw_reference(
    summary: pd.DataFrame,
    chain: pd.DataFrame,
    raw_dir: Path,
    expected: Sequence[int],
) -> List[str]:
    errors: List[str] = []
    if summary["split_index"].duplicated().any():
        errors.append("旧 raw summary 存在重复 split_index 行")
    summary_splits = set(summary["split_index"].astype(int))
    chain_splits = set(chain["split_index"].astype(int))
    if summary_splits != chain_splits:
        errors.append(f"旧 raw summary/chain 折集不一致: {summary_splits} vs {chain_splits}")
    expected_splits = set(expected)
    if summary_splits != expected_splits or chain_splits != expected_splits:
        errors.append(
            f"旧 raw 折集必须等于预期 {sorted(expected_splits)}："
            f"summary={sorted(summary_splits)}，chain={sorted(chain_splits)}"
        )
    for split, segment in chain.groupby("split_index", sort=False):
        if not segment["date"].is_unique or not segment["date"].is_monotonic_increasing:
            errors.append(f"旧 raw split{int(split):02d} chain_nav 日期/序号重复或非递增")
        errors += _finite_frame_errors(
            segment,
            batch_id=raw_dir.name,
            split=int(split),
            label="raw chain_nav",
            date_column="date",
        )
    return errors


def _parse_call_counts(batch: RunsBatch, mode: str) -> List[str]:
    notes = batch.batch_meta.get("meta_notes") or {}
    errors: List[str] = []
    try:
        by_fold = json.loads(notes["merger.call_count_by_fold"])
        raw_total = notes["merger.call_count"]
        if not isinstance(raw_total, str) or not raw_total.isdecimal():
            raise ValueError("merger.call_count 必须是非负十进制字符串")
        total = int(raw_total)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [f"批次 {batch.batch_id}: 合并调用计数 notes 缺失或非法: {exc}"]
    if not isinstance(by_fold, dict):
        return [f"批次 {batch.batch_id}: call_count_by_fold 必须是 JSON object"]
    expected_keys = {str(split) for split in batch.folds}
    if set(by_fold) != expected_keys:
        errors.append(f"批次 {batch.batch_id}: call_count_by_fold 折集不符")
    if any(type(value) is not int or value < 0 for value in by_fold.values()):
        errors.append(f"批次 {batch.batch_id}: 逐折调用计数必须为非负整数（布尔亦不合法）")
    if not by_fold or total != sum(by_fold.values()):
        errors.append(f"批次 {batch.batch_id}: call_count != sum(call_count_by_fold)")
    if mode == LEGACY_BYPASS and (total != 0 or any(by_fold.values())):
        errors.append(f"批次 {batch.batch_id}: 关闭批每折与总调用数必须为 0")
    if mode == SINGLE_SLEEVE_PASSTHROUGH and total <= 0:
        errors.append(f"批次 {batch.batch_id}: 开启批真实 merge 总调用数必须 >0")
    return errors


def _validate_mode_and_meta(batch: RunsBatch, mode: str, baseline_ref: str) -> List[str]:
    errors: List[str] = []
    meta = batch.batch_meta
    summary = batch.summary
    for column, expected in (("merger_mode", mode), ("merger_sleeve_id", MERGER_SLEEVE_ID)):
        if column not in summary.columns:
            errors.append(f"summary 缺 {column}")
        elif not all(_values_equal(value, expected) for value in summary[column]):
            errors.append(f"summary.{column} 未逐折等于 {expected!r}")
        if meta.get("config", {}).get(column) != expected:
            errors.append(f"batch_meta.config.{column} != {expected!r}")
    if meta.get("source") != P2B_SOURCE:
        errors.append(f"source={meta.get('source')!r} != {P2B_SOURCE!r}")
    if list(meta.get("arms") or []) != [mode]:
        errors.append(f"arms={meta.get('arms')!r} != {[mode]!r}")
    if meta.get("baseline_ref") != baseline_ref:
        errors.append(f"baseline_ref={meta.get('baseline_ref')!r} != {baseline_ref!r}")
    notes = meta.get("meta_notes") or {}
    if notes.get("merger.mode") != mode or notes.get("merger.sleeve_id") != MERGER_SLEEVE_ID:
        errors.append("meta_notes 的合并模式/袖子 ID 与实际装配不符")
    return [f"批次 {batch.batch_id}: {message}" for message in errors] + _parse_call_counts(
        batch, mode
    )


def _validate_evidence_notes(batch: RunsBatch) -> List[str]:
    notes = batch.batch_meta.get("meta_notes") or {}
    errors: List[str] = []
    for key in P2B_EVIDENCE_NOTE_KEYS:
        if key not in notes:
            errors.append(f"批次 {batch.batch_id}: meta_notes 缺 {key}")
    try:
        workspace = json.loads(notes["workspace.digest"])
        if not validate_workspace_digest(workspace):
            errors.append(f"批次 {batch.batch_id}: workspace.digest 总摘要不自洽")
        environment = json.loads(notes["environment.versions"])
        if (
            not isinstance(environment, dict)
            or "python" not in environment
            or not isinstance(environment.get("packages"), dict)
        ):
            errors.append(f"批次 {batch.batch_id}: environment.versions 结构不完整")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        errors.append(f"批次 {batch.batch_id}: 工作树/环境证据非法: {exc}")
    return errors


def _validate_pair_evidence(
    closed: RunsBatch,
    enabled: RunsBatch,
    p2a_digest: Mapping[str, object],
    raw_digest: Mapping[str, object],
) -> List[str]:
    errors: List[str] = []
    closed_notes = closed.batch_meta.get("meta_notes") or {}
    enabled_notes = enabled.batch_meta.get("meta_notes") or {}
    if set(closed_notes) != set(enabled_notes):
        errors.append("双臂 meta_notes 键集不一致")
    count_keys = {"merger.mode", "merger.call_count", "merger.call_count_by_fold"}
    for key in (set(closed_notes) & set(enabled_notes)) - count_keys:
        if closed_notes[key] != enabled_notes[key]:
            errors.append(f"双臂共享证据 {key} 不一致")
    expected = {
        "reference.p2a_runs.digest": canonical_digest_note(p2a_digest),
        "reference.b0_raw.digest": canonical_digest_note(raw_digest),
    }
    for key, value in expected.items():
        if closed_notes.get(key) != value or enabled_notes.get(key) != value:
            errors.append(f"双臂 {key} 与实际冻结参照摘要不一致")
    return errors


def _validate_reference_meta(reference: RunsBatch) -> List[str]:
    """P2a 参照自身的来源、臂和基线标识也必须是冻结形态。"""
    meta = reference.batch_meta
    errors: List[str] = []
    if meta.get("source") != "v2_p2a_native_replay":
        errors.append(f"P2a 参照 source 非法: {meta.get('source')!r}")
    if list(meta.get("arms") or []):
        errors.append(f"P2a 参照 arms 必须为空: {meta.get('arms')!r}")
    if meta.get("baseline_ref") is not None:
        errors.append(f"P2a 参照 baseline_ref 必须为 None: {meta.get('baseline_ref')!r}")
    return errors


def _validate_reference_notes(
    closed: RunsBatch,
    reference: RunsBatch,
    allow_subset: bool,
) -> List[str]:
    """关闭批只能新增预登记证据；显式子集只豁免折集派生的 topk 折清单。"""
    closed_notes = closed.batch_meta.get("meta_notes") or {}
    reference_notes = reference.batch_meta.get("meta_notes") or {}
    errors: List[str] = []
    unexpected = set(closed_notes) - set(reference_notes) - P2B_EVIDENCE_NOTE_KEYS
    if unexpected:
        errors.append(f"关闭批含未登记 meta_notes 新键: {sorted(unexpected)}")
    subset_dependent = (
        {"topk_detail.missing_folds"}
        if allow_subset and set(closed.folds) < set(reference.folds)
        else set()
    )
    for key, value in reference_notes.items():
        if key not in subset_dependent and closed_notes.get(key) != value:
            errors.append(f"关闭批与 P2a 参照共享 notes {key} 不一致")
    return errors


def _normalized_summary_pair(
    new: RunsBatch,
    reference: RunsBatch,
    *,
    migration_only_values: Mapping[str, object],
    shared_expected_values: Mapping[str, Sequence[object]],
    normalize_data_state: bool,
    allow_subset: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    new_frame = new.summary.copy()
    reference_frame = reference.summary.copy()
    reference_frame = reference_frame[reference_frame["split_index"].isin(new.folds)].reset_index(
        drop=True
    )
    for column in IDENTITY_COLUMNS:
        for frame in (new_frame, reference_frame):
            if column in frame.columns:
                frame[column] = "<normalized>"
    if normalize_data_state:
        new_frame["data_state_id"] = "<normalized-after-gate1>"
        reference_frame["data_state_id"] = "<normalized-after-gate1>"
    if allow_subset and set(new.folds) != set(reference.folds):
        for column in ("wf_start_date", "wf_end_date"):
            if column in new_frame.columns and column in reference_frame.columns:
                new_frame[column] = reference_frame[column] = "<subset>"
    for column in shared_expected_values:
        new_frame[column] = reference_frame[column] = "<expected-mode-difference>"
    new_frame = new_frame.drop(columns=list(migration_only_values), errors="ignore")
    return new_frame, reference_frame


def _compare_summary_exact(
    new: RunsBatch,
    reference: RunsBatch,
    report: MergerCompareReport,
    edge: str,
    **normalization,
) -> None:
    migration = normalization["migration_only_values"]
    shared = normalization["shared_expected_values"]
    for key in migration:
        if key not in new.summary.columns or key in reference.summary.columns:
            report.fail(edge, f"新侧独有键结构不符: {key}")
    for key in shared:
        if key not in new.summary.columns or key not in reference.summary.columns:
            report.fail(edge, f"共享期望键缺失: {key}")
    fp_report = CompareReport()
    compare_fingerprint_gate(
        new.summary,
        reference.summary,
        fp_report,
        allow_subset=normalization["allow_subset"],
        migration_only_values=dict(migration),
        shared_expected_values=dict(shared),
        normalized_shared_keys=("batch_id",),
    )
    for message in fp_report.gate3_diffs:
        report.fail(edge, message)
    report.notes += [f"{edge}: {message}" for message in fp_report.notes]
    new_frame, reference_frame = _normalized_summary_pair(new, reference, **normalization)
    if set(new_frame.columns) != set(reference_frame.columns):
        report.fail(
            edge,
            f"summary 列集不一致: 新多 {sorted(set(new_frame.columns)-set(reference_frame.columns))} "
            f"/ 新缺 {sorted(set(reference_frame.columns)-set(new_frame.columns))}",
        )
        return
    columns = list(reference_frame.columns)
    if not _frame_equal_exact(new_frame[columns], reference_frame[columns]):
        report.fail(edge, "summary 绩效/配置全列精确比较失败")


def _compare_optional_frame(
    new_frame: Optional[pd.DataFrame],
    reference_frame: Optional[pd.DataFrame],
    *,
    normalize_columns: Sequence[str] = (),
) -> bool:
    if new_frame is None or reference_frame is None:
        return new_frame is None and reference_frame is None
    if list(new_frame.columns) != list(reference_frame.columns):
        return False
    left, right = new_frame.copy(), reference_frame.copy()
    for column in normalize_columns:
        left[column] = right[column] = "<normalized>"
    return _frame_equal_exact(left.reset_index(drop=True), right.reset_index(drop=True))


def _compare_fold_artifacts(
    new_fold: RunsFold,
    reference_fold: RunsFold,
    report: MergerCompareReport,
    edge: str,
) -> None:
    split = new_fold.split_index
    specs = (
        ("trades", new_fold.trades, reference_fold.trades, ("wf_run_id",)),
        ("attribution", new_fold.attribution, reference_fold.attribution, ("wf_run_id",)),
        (
            "holdings_snapshot",
            new_fold.holdings_snapshot,
            reference_fold.holdings_snapshot,
            ("run_id",),
        ),
        ("daily", new_fold.daily, reference_fold.daily, ()),
        ("topk_detail", new_fold.topk_detail, reference_fold.topk_detail, ("wf_run_id",)),
    )
    for name, new_frame, reference_frame, normalized in specs:
        if not _compare_optional_frame(new_frame, reference_frame, normalize_columns=normalized):
            report.fail(edge, f"split{split:02d}/{name} 全列行序精确比较失败")
    new_meta = dict(new_fold.meta)
    reference_meta = dict(reference_fold.meta)
    for key in IDENTITY_COLUMNS:
        for document in (new_meta, reference_meta):
            if key in document:
                document[key] = "<normalized>"
    if new_meta != reference_meta:
        report.fail(edge, f"split{split:02d}/_meta.json 不一致")


def _check_fold_relation(
    new: RunsBatch,
    reference: RunsBatch,
    report: MergerCompareReport,
    edge: str,
    allow_subset: bool,
) -> None:
    new_splits, reference_splits = set(new.folds), set(reference.folds)
    if new_splits == reference_splits:
        return
    if not allow_subset or not new_splits < reference_splits:
        report.fail(edge, f"折集不一致: 新 {sorted(new_splits)} vs 参照 {sorted(reference_splits)}")


def compare_runs_exact(
    new: RunsBatch,
    reference: RunsBatch,
    report: MergerCompareReport,
    edge: str,
    *,
    migration_only_values: Mapping[str, object],
    shared_expected_values: Mapping[str, Sequence[object]],
    normalize_data_state: bool,
    allow_subset: bool,
) -> None:
    """runs/runs 全产物精确比较，不排序掩盖行序变化。"""
    before = len(report.errors)
    _check_fold_relation(new, reference, report, edge, allow_subset)
    normalization = {
        "migration_only_values": migration_only_values,
        "shared_expected_values": shared_expected_values,
        "normalize_data_state": normalize_data_state,
        "allow_subset": allow_subset,
    }
    _compare_summary_exact(new, reference, report, edge, **normalization)
    for split, new_fold in new.folds.items():
        reference_fold = reference.folds.get(split)
        if reference_fold is not None:
            _compare_fold_artifacts(new_fold, reference_fold, report, edge)
    reference_chain = reference.chain_nav[reference.chain_nav["split_index"].isin(new.folds)]
    if not _compare_optional_frame(new.chain_nav, reference_chain.reset_index(drop=True)):
        report.fail(edge, "chain_nav 全列行序精确比较失败")
    report.edge_results[edge] = len(report.errors) == before


def _append_raw_report(report: MergerCompareReport, raw_report: CompareReport) -> None:
    edge = "edge2_closed_vs_b0_raw"
    for message in raw_report.gate3_diffs + raw_report.gate4_diffs + raw_report.gate5_diffs:
        report.fail(edge, message)
    report.notes += [f"{edge}: {message}" for message in raw_report.notes]
    report.edge_results[edge] = raw_report.ok


def _run_prechecks(
    *,
    closed: RunsBatch,
    enabled: RunsBatch,
    reference: RunsBatch,
    raw_summary: pd.DataFrame,
    raw_chain: pd.DataFrame,
    b0_raw_dir: Path,
    expected: Sequence[int],
    p2a_digest: Mapping[str, object],
    raw_digest: Mapping[str, object],
    allow_subset: bool,
    report: MergerCompareReport,
) -> None:
    """集中执行三边前置检查；先收集证据再统一登记，避免分支散落。"""
    messages: List[str] = []
    if set(reference.folds) != set(expected):
        messages.append(
            f"P2a 参照折集必须等于预期 {sorted(set(expected))}，实际={sorted(reference.folds)}"
        )
    for batch in (closed, enabled, reference):
        messages += validate_batch_identity(batch) + validate_nav_finite(batch)
    messages += _validate_raw_reference(raw_summary, raw_chain, Path(b0_raw_dir), expected)
    messages += _validate_mode_and_meta(closed, LEGACY_BYPASS, reference.batch_id)
    messages += _validate_mode_and_meta(enabled, SINGLE_SLEEVE_PASSTHROUGH, closed.batch_id)
    for batch in (closed, enabled):
        messages += _validate_evidence_notes(batch)
    messages += _validate_pair_evidence(closed, enabled, p2a_digest, raw_digest)
    messages += _validate_reference_notes(closed, reference, allow_subset)
    messages += _validate_reference_meta(reference)
    for message in messages:
        report.fail("precheck", message)


def compare_merger_batches(
    closed_batch_dir: Path,
    enabled_batch_dir: Path,
    p2a_runs_dir: Path,
    b0_raw_dir: Path,
    *,
    allow_subset: bool = False,
    input_identity_verified: bool = False,
    expected_splits: Optional[Sequence[int]] = None,
) -> MergerCompareReport:
    """执行固定三边验收；默认独立锚定完整 14 折，合成测试可显式注入折集。"""
    report = MergerCompareReport()
    if not input_identity_verified:
        for edge in EDGES:
            report.fail(edge, "缺少门 1 输入身份一致证明，禁止规范化 data_state_id 或放行比较")
        return report
    expected = (
        tuple(range(14)) if expected_splits is None else tuple(int(s) for s in expected_splits)
    )
    if not expected:
        for edge in EDGES:
            report.fail(edge, "预期折集为空")
        return report
    closed = load_runs_batch(Path(closed_batch_dir))
    enabled = load_runs_batch(Path(enabled_batch_dir))
    reference = load_runs_batch(Path(p2a_runs_dir))
    raw_refs = resolve_b0_raw_reference(Path(b0_raw_dir), expected)
    raw_summary = pd.read_csv(raw_refs.summary, encoding="utf-8-sig", float_precision="round_trip")
    raw_chain = pd.read_csv(raw_refs.chain_nav, encoding="utf-8-sig", float_precision="round_trip")
    p2a_digest = digest_p2a_runs_reference(Path(p2a_runs_dir), expected)
    raw_digest = digest_b0_raw_reference(Path(b0_raw_dir), raw_refs)
    _run_prechecks(
        closed=closed,
        enabled=enabled,
        reference=reference,
        raw_summary=raw_summary,
        raw_chain=raw_chain,
        b0_raw_dir=Path(b0_raw_dir),
        expected=expected,
        p2a_digest=p2a_digest,
        raw_digest=raw_digest,
        allow_subset=allow_subset,
        report=report,
    )
    if report.errors:
        return report

    compare_runs_exact(
        closed,
        reference,
        report,
        "edge1_closed_vs_p2a_runs",
        migration_only_values={"merger_mode": LEGACY_BYPASS, "merger_sleeve_id": MERGER_SLEEVE_ID},
        shared_expected_values={},
        normalize_data_state=True,
        allow_subset=allow_subset,
    )
    raw_report = compare_replay_vs_b0(
        Path(closed_batch_dir),
        Path(b0_raw_dir),
        allow_subset=allow_subset,
        reference_files=raw_refs,
        migration_only_values={"merger_mode": LEGACY_BYPASS, "merger_sleeve_id": MERGER_SLEEVE_ID},
    )
    _append_raw_report(report, raw_report)
    compare_runs_exact(
        enabled,
        closed,
        report,
        "edge3_enabled_vs_closed",
        migration_only_values={},
        shared_expected_values={"merger_mode": (SINGLE_SLEEVE_PASSTHROUGH, LEGACY_BYPASS)},
        normalize_data_state=False,
        allow_subset=False,
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-batch", required=True, type=Path)
    parser.add_argument("--enabled-batch", required=True, type=Path)
    parser.add_argument("--p2a-runs", default=DEFAULT_P2A_RUNS)
    parser.add_argument("--b0-raw", default=DEFAULT_B0_RAW)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--input-identity-verified", action="store_true")
    args = parser.parse_args(argv)
    report = compare_merger_batches(
        args.closed_batch,
        args.enabled_batch,
        Path(args.p2a_runs),
        Path(args.b0_raw),
        allow_subset=args.allow_subset,
        input_identity_verified=args.input_identity_verified,
    )
    for message in report.errors:
        print(f"FAIL {message}")
    for message in report.notes:
        print(f"NOTE {message}")
    print(json.dumps({"ok": report.ok, "edges": report.edge_results}, ensure_ascii=False))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
