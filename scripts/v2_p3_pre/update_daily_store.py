# -*- coding: utf-8 -*-
"""P3 pre T3：v2 日更双通路薄入口（默认 dry-run）。

逐日 build_daily 只负责结构段初建；E20 到期标签只经 label_refresh 专用路径 sealed。
普通执行、refresh-only 与 replace→WAL 受控恢复互不借用授权。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "v2_p3_pre"))

from migrate_store_manifest import _reject_pending_wal, _write_json_atomic  # noqa: E402

from src.lazybull.common.date_utils import normalize_series_to_yyyymmdd  # noqa: E402
from src.lazybull.data.loader import DataLoader  # noqa: E402
from src.lazybull.data.storage import Storage  # noqa: E402
from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.column_groups import LABEL_TABLES, PANEL_GROUPS  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402
from src.lazybull.v2.store.label_refresh import (  # noqa: E402
    LabelRefreshPlan,
    plan_label_refreshes,
    refresh_labels_for_day,
    validate_label_day_readiness,
)
from src.lazybull.v2.store.manifest import Manifest, sha256_16_of_file  # noqa: E402
from src.lazybull.v2.store.panel_builder import V2PanelBuilder, bootstrap_manifest  # noqa: E402
from src.lazybull.v2.store.row_domain import STRUCTURAL_FROM  # noqa: E402
from src.lazybull.v2.store.store_identity import (  # noqa: E402
    InputIdentityDriftError,
    collect_file_identities,
    producer_identity,
    verify_file_identities,
    workspace_identity,
)

#: 逐日构建实测估算（P3-pre T4 实测 452~462s/日，2026-10-19；原 75s 估算偏低约 7 倍，
#: 停止规则 #7 处置：按实测 + 余量修正，恢复 dry-run 耗时预检的真实守卫意义）
_BUILD_SECONDS_PER_DAY = 500
_REFRESH_SECONDS_PER_DAY = 3


def _assert_identity_stable(
    expected: Sequence[Mapping[str, Any]], base_dir: Path | str, *, context: str
) -> None:
    mismatches = verify_file_identities(expected, base_dir=base_dir)
    if mismatches:
        raise InputIdentityDriftError(f"{context}: {mismatches}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=STRUCTURAL_FROM, help="结构段日更起点 YYYYMMDD")
    parser.add_argument("--data-root", default=str(ROOT / "data"), help="v2 store 数据根")
    parser.add_argument("--backup-root", required=True, help="同代 run 备份目录")
    parser.add_argument("--execute", action="store_true", help="显式执行；缺省仅 dry-run")
    parser.add_argument("--refresh-only", action="store_true", help="仅追赶已完整 panel 的标签")
    parser.add_argument("--recover-event", help="受控 roll-forward 的既有 prepare event_key")
    parser.add_argument(
        "--authorized-minutes", type=float, default=30.0, help="本次预授权机器分钟数（默认 30）"
    )
    args = parser.parse_args(argv)
    if args.refresh_only and args.recover_event:
        parser.error("--refresh-only 与 --recover-event 互斥")
    if not args.recover_event and args.start_date < STRUCTURAL_FROM:
        parser.error(f"生产日更起点不得早于 {STRUCTURAL_FROM}；历史影子不得由此入口执行")
    if args.authorized_minutes <= 0:
        parser.error("--authorized-minutes 必须为正数")
    return args


def _panel_missing_groups(store: PanelDataStore, trade_date: str) -> list[str]:
    """检查结构段热区缺族；本期禁止月压实，未来若引入须同步扩展冷区判定。"""
    snapshot = store.manifest.snapshot()
    missing: list[str] = []
    for group in PANEL_GROUPS:
        rel_path = f"panel/{trade_date}/{group}.parquet"
        path = store.panel_dir / trade_date / f"{group}.parquet"
        meta = snapshot["partitions"].get(rel_path)
        if meta is None and not path.exists():
            missing.append(group)
        elif meta is None or not path.exists():
            raise RuntimeError(f"panel 文件与登记不一致: {rel_path}")
        elif sha256_16_of_file(path) != meta.get("sha256_16"):
            raise RuntimeError(f"panel 物理指纹与登记不一致: {rel_path}")
    return missing


def _clean_candidate_paths(daily_dir: Path, first_input: str) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for path in daily_dir.glob("*.parquet"):
        date = path.stem.replace("-", "")
        if len(date) == 8 and date.isdigit() and date >= first_input:
            candidates.append((date, path))
    if not candidates:
        raise ValueError(f"clean/daily 无 {first_input} 起候选分区: {daily_dir}")
    return sorted(candidates)


def _read_clean_partition(path: Path, partition_date: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path, columns=["trade_date", "ts_code", "open_adj", "close_adj"])
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError(f"clean/daily 尾部候选读取失败: {path}（{exc}）") from exc
    if frame.empty:
        raise RuntimeError(f"clean/daily 尾部候选为空，禁止回退更早水位: {path}")
    dates = normalize_series_to_yyyymmdd(frame["trade_date"])
    if set(dates) != {partition_date}:
        raise RuntimeError(f"clean/daily 分区名与行日期不一致: {path}")
    return frame.assign(trade_date=dates)


def _static_input_identities(loader: Any) -> list[dict[str, Any]]:
    calendar_path = loader.storage.clean_path / "trade_cal.parquet"
    basic_path = loader.storage.raw_path / "stock_basic.parquet"
    for required in (calendar_path, basic_path):
        if not required.is_file():
            raise ValueError(f"日更输入身份必需文件缺失: {required}")
    return collect_file_identities([calendar_path, basic_path], base_dir=loader.storage.root_path)


def _load_clean_inputs(
    loader: Any,
    calendar: Sequence[str],
    start_date: str,
    *,
    initial_identities: Sequence[Mapping[str, Any]] | None = None,
):
    try:
        start_idx = list(calendar).index(start_date)
        first_input = calendar[start_idx + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"无法确定 {start_date} 的 T+1 输入起点") from exc
    static_identities = list(initial_identities or _static_input_identities(loader))
    daily_dir = loader.storage.clean_path / "daily"
    candidates = _clean_candidate_paths(daily_dir, first_input)
    input_root = Path(loader.storage.root_path)
    daily_identities = [
        collect_file_identities([path], base_dir=input_root)[0] for _, path in candidates
    ]
    identities = sorted([*static_identities, *daily_identities], key=lambda record: record["path"])
    frames: dict[str, pd.DataFrame] = {}
    for (partition_date, path), expected in zip(candidates, daily_identities, strict=True):
        frame = _read_clean_partition(path, partition_date)
        _assert_identity_stable(
            [expected], input_root, context=f"clean/daily {partition_date} 读取期间发生输入漂移"
        )
        frames[partition_date] = frame
    _assert_identity_stable(identities, input_root, context="clean 输入加载完成后发生输入漂移")
    return frames, max(frames), identities


def _recover_date(store: PanelDataStore, event_key: str) -> str:
    records = [
        record
        for record in store.manifest.label_refresh_intents()
        if record.get("record_type") == "prepare" and record.get("event_key") == event_key
    ]
    if len(records) != 1:
        raise ValueError(f"受控恢复事件不是唯一既有 prepare: {event_key}")
    return str(records[0]["trade_date"])


def _estimate_seconds(build_days: int, refresh_days: int) -> int:
    return build_days * _BUILD_SECONDS_PER_DAY + refresh_days * _REFRESH_SECONDS_PER_DAY


def _check_time_budget(
    build_days: int, refresh_days: int, authorized_minutes: float, *, preflight_days: int = 0
) -> int:
    estimated = (
        _estimate_seconds(build_days, refresh_days) + preflight_days * _REFRESH_SECONDS_PER_DAY
    )
    if estimated > authorized_minutes * 60:
        raise RuntimeError(
            f"预计耗时 {estimated / 60:.1f} 分钟超过本次授权 {authorized_minutes:.1f} 分钟；"
            "启动前拒绝，未写文件"
        )
    return estimated


def _refresh_kwargs(
    calendar: Sequence[str],
    daily_frames: Mapping[str, pd.DataFrame],
    watermark: str,
    builder: Any,
    producer: Mapping[str, Any],
    input_identities: Sequence[Mapping[str, Any]],
    input_root: Path,
    run_id: str,
) -> dict[str, Any]:
    return {
        "trading_dates": list(calendar),
        "daily_adj_dict": daily_frames,
        "input_watermark": watermark,
        "shenwan_level": builder._builder.shenwan_level,
        "producer": producer,
        "input_identities": input_identities,
        "input_base_dir": input_root,
        "run_id": run_id,
    }


def _planned_new_files(
    build_plan: Mapping[str, list[str]], refresh_dates: Sequence[str]
) -> list[str]:
    files = [
        f"features/panel/{date}/{group}.parquet"
        for date, groups in build_plan.items()
        for group in groups
    ]
    label_dates = sorted({*build_plan, *refresh_dates})
    files.extend(f"labels/{name}/{date}.parquet" for date in label_dates for name in LABEL_TABLES)
    return sorted(files)


def _save_run_snapshot(
    store: PanelDataStore,
    backup_root: Path,
    *,
    affected_dates: Sequence[str],
    planned_new_files: Sequence[str],
    input_identities: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
    workspace: Mapping[str, Any],
) -> dict[str, Any]:
    if backup_root.exists() and any(backup_root.iterdir()):
        raise RuntimeError(f"备份目录非空，拒绝覆盖既有 run 快照: {backup_root}")
    backup_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in (
        store.features_dir / "manifest.json",
        store.manifest.backup_path,
        store.wal_path,
    ):
        if source.exists():
            shutil.copy2(source, backup_root / source.name)
            copied.append(source.name)
    affected: list[str] = []
    for date in sorted(set(affected_dates)):
        for name in LABEL_TABLES:
            source = store.labels_dir / name / f"{date}.parquet"
            if not source.exists():
                continue
            relative = Path("files") / "labels" / name / source.name
            target = backup_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            affected.append(str(Path("labels") / name / source.name).replace("\\", "/"))
    checkpoint = {
        "empty_checkpoint": not store.wal_path.exists() or store.wal_path.stat().st_size == 0,
        "copied_files": copied,
        "affected_existing_files": affected,
        "planned_new_files": sorted(
            relative for relative in planned_new_files if not (store.root / relative).exists()
        ),
    }
    _write_json_atomic(backup_root / "checkpoint.json", checkpoint)
    snapshot = store.manifest.snapshot()
    manifest_path = store.features_dir / "manifest.json"
    source_fp = (
        Manifest(manifest_path).content_fingerprint()
        if manifest_path.exists()
        else store.manifest.content_fingerprint()
    )
    _write_json_atomic(
        backup_root / "snapshot.json",
        {
            "source_content_fingerprint": source_fp,
            "replayed_content_fingerprint": store.manifest.content_fingerprint(),
            "input_files": list(input_identities),
            "producer": dict(producer),
            "workspace": dict(workspace),
            "label_refresh_intents": snapshot.get("label_refresh_intents", []),
            "label_refreshes": snapshot.get("label_refreshes", []),
            "repairs_count": len(snapshot.get("repairs", [])),
        },
    )
    return checkpoint


def _save_planned_snapshot(
    store: PanelDataStore,
    backup_root: Path,
    build_plan: Mapping[str, list[str]],
    refresh_plans: Sequence[LabelRefreshPlan],
    input_identities: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
    workspace: Mapping[str, Any],
) -> dict[str, Any]:
    refresh_dates = [plan.trade_date for plan in refresh_plans]
    affected_dates = sorted({*build_plan, *refresh_dates})
    with store.write_lock():
        return _save_run_snapshot(
            store,
            backup_root,
            affected_dates=affected_dates,
            planned_new_files=_planned_new_files(build_plan, refresh_dates),
            input_identities=input_identities,
            producer=producer,
            workspace=workspace,
        )


def _report_path(data_root: Path, run_id: str) -> Path:
    return (data_root / "reports" / f"v2_p3_pre_label_refresh_{run_id}.json").resolve()


def _write_failure_report(
    base: Mapping[str, Any],
    error: Exception,
    checkpoint: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    input_identity_unchanged: bool | None = None,
) -> None:
    report = {
        **base,
        "status": "failed",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": dict(checkpoint),
        "results": list(results),
        "failure_reason": str(error),
        "input_identity_unchanged": input_identity_unchanged,
    }
    try:
        _write_json_atomic(Path(str(base["report_path"])), report)
    except OSError as report_error:
        error.add_note(f"失败报告写入也失败: {base['report_path']}（{report_error}）")


def _verify_run_identities(
    input_identities: Sequence[Mapping[str, Any]],
    input_root: Path,
    *,
    producer: Mapping[str, Any] | None,
    workspace: Mapping[str, Any] | None,
    producer_evidence: Mapping[str, Any],
    workspace_evidence: Mapping[str, Any],
) -> None:
    _assert_identity_stable(input_identities, input_root, context="日更运行前后输入身份变化")
    if (
        producer is None
        and producer_identity(ROOT)["producer_digest"] != producer_evidence["producer_digest"]
    ):
        raise RuntimeError("日更运行前后生产者代码或环境身份变化")
    if (
        workspace is None
        and workspace_identity(ROOT)["source_digest"] != workspace_evidence["source_digest"]
    ):
        raise RuntimeError("日更运行前后 src/scripts 内容摘要变化")


def _validate_run_options(
    start_date: str, refresh_only: bool, recover_event: str | None, authorized_minutes: float
) -> None:
    if refresh_only and recover_event:
        raise ValueError("refresh-only 与受控恢复互斥")
    if not recover_event and str(start_date) < STRUCTURAL_FROM:
        raise ValueError(f"生产日更起点不得早于 {STRUCTURAL_FROM}")
    if authorized_minutes <= 0:
        raise ValueError("authorized_minutes 必须为正数")


@dataclass
class _DailyRunContext:
    store: PanelDataStore
    loader: Any
    builder: Any
    backup: Path
    calendar: list[str]
    candidates: list[str]
    effective_start: str
    watermark: str
    recover_event: str | None
    build_plan: dict[str, list[str]]
    refresh_plans: list[LabelRefreshPlan]
    refresh_kwargs: dict[str, Any]
    input_identities: list[dict[str, Any]]
    producer: Mapping[str, Any] | None
    workspace: Mapping[str, Any] | None
    producer_evidence: dict[str, Any]
    workspace_evidence: dict[str, Any]
    base: dict[str, Any]


def _endpoint_or_none(calendar: Sequence[str], trade_date: str, offset: int) -> str | None:
    try:
        index = list(calendar).index(trade_date)
        return calendar[index + offset]
    except (ValueError, IndexError):
        return None


def _daily_plan_entries(
    calendar: Sequence[str],
    candidates: Sequence[str],
    build_plan: Mapping[str, list[str]],
    refresh_plans: Sequence[LabelRefreshPlan],
) -> list[dict[str, Any]]:
    refresh_by_date = {plan.trade_date: plan for plan in refresh_plans}
    entries: list[dict[str, Any]] = []
    cumulative = 0
    for date in candidates:
        refresh = refresh_by_date.get(date)
        build_seconds = _BUILD_SECONDS_PER_DAY if date in build_plan else 0
        refresh_seconds = _REFRESH_SECONDS_PER_DAY if refresh is not None else 0
        preflight_seconds = refresh_seconds if not build_plan else 0
        cumulative += build_seconds + refresh_seconds + preflight_seconds
        entries.append(
            {
                "trade_date": date,
                "buy_date": _endpoint_or_none(calendar, date, 1),
                "endpoint_date": _endpoint_or_none(calendar, date, 21),
                "row_domain": refresh.row_domain if refresh else "structural",
                "expected_status": "sealed" if refresh else "forming",
                "build_daily_call": date in build_plan,
                "refresh_call": refresh is not None,
                "estimated_build_seconds": build_seconds,
                "estimated_refresh_seconds": refresh_seconds,
                "estimated_preflight_seconds": preflight_seconds,
                "estimated_seconds": build_seconds + refresh_seconds + preflight_seconds,
                "cumulative_estimated_seconds": cumulative,
            }
        )
    return entries


def _prepare_update(
    data_root: Path | str,
    backup_root: Path | str,
    *,
    start_date: str,
    execute: bool,
    refresh_only: bool,
    recover_event: str | None,
    authorized_minutes: float,
    store: PanelDataStore | None,
    loader: Any,
    builder: Any,
    producer: Mapping[str, Any] | None,
    workspace: Mapping[str, Any] | None,
) -> _DailyRunContext:
    _validate_run_options(start_date, refresh_only, recover_event, authorized_minutes)
    root = Path(data_root)
    backup = Path(backup_root)
    run_id = backup.name
    store = store or PanelDataStore(root)
    loader = loader or (
        builder._loader if builder is not None else DataLoader(Storage(root_path=str(root)))
    )
    builder = builder or V2PanelBuilder(loader)
    if recover_event:
        with store.write_lock():
            store._wal.load_manifest(persist=False)
    else:
        _reject_pending_wal(store.wal_path)
    static_identities = _static_input_identities(loader)
    calendar = list(builder._full_calendar())
    _assert_identity_stable(
        static_identities,
        Path(loader.storage.root_path),
        context="交易日历/基础信息读取期间发生输入漂移",
    )
    effective_start = _recover_date(store, recover_event) if recover_event else str(start_date)
    daily_frames, watermark, input_identities = _load_clean_inputs(
        loader, calendar, effective_start, initial_identities=static_identities
    )
    producer_evidence = dict(producer or producer_identity(ROOT))
    workspace_evidence = dict(workspace or workspace_identity(ROOT))
    candidates = [date for date in calendar if effective_start <= date <= watermark]
    if not candidates:
        raise ValueError(f"日更区间无交易日: {effective_start}~{watermark}")
    build_plan = (
        {}
        if refresh_only or recover_event
        else {
            date: missing for date in candidates if (missing := _panel_missing_groups(store, date))
        }
    )
    refresh_plans = plan_label_refreshes(
        store,
        [effective_start] if recover_event else candidates,
        calendar,
        watermark,
        start_date=effective_start if recover_event else STRUCTURAL_FROM,
        validate_panel=False,
    )
    preflight_days = len(refresh_plans) if not build_plan else 0
    estimated = _check_time_budget(
        len(build_plan), len(refresh_plans), authorized_minutes, preflight_days=preflight_days
    )
    kwargs = _refresh_kwargs(
        calendar,
        daily_frames,
        watermark,
        builder,
        producer_evidence,
        input_identities,
        Path(loader.storage.root_path),
        run_id,
    )
    plan_dates = [effective_start] if recover_event else candidates
    daily_plan = _daily_plan_entries(calendar, plan_dates, build_plan, refresh_plans)
    base = {
        "run_id": run_id,
        "mode": "execute" if execute else "dry_run",
        "start_date": effective_start,
        "end_date": watermark,
        "input_watermark": watermark,
        "refresh_only": refresh_only,
        "recover_event": recover_event,
        "build_plan": build_plan,
        "refresh_plans": [plan.to_dict() for plan in refresh_plans],
        "daily_plan": daily_plan,
        "build_daily_calls": len(build_plan),
        "refresh_calls": len(refresh_plans),
        "estimated_build_seconds": len(build_plan) * _BUILD_SECONDS_PER_DAY,
        "estimated_refresh_seconds": len(refresh_plans) * _REFRESH_SECONDS_PER_DAY,
        "estimated_preflight_seconds": preflight_days * _REFRESH_SECONDS_PER_DAY,
        "estimated_seconds": estimated,
        "authorized_minutes": authorized_minutes,
        "results_note": "results 依执行顺序含 build_daily 逐日结果与 refresh 结果；"
        "readiness 为逐日消费门",
        "input_files": input_identities,
        "producer": producer_evidence,
        "workspace": workspace_evidence,
        "report_path": str(_report_path(root, run_id)),
    }
    return _DailyRunContext(
        store,
        loader,
        builder,
        backup,
        calendar,
        candidates,
        effective_start,
        watermark,
        recover_event,
        build_plan,
        refresh_plans,
        kwargs,
        input_identities,
        producer,
        workspace,
        producer_evidence,
        workspace_evidence,
        base,
    )


def _preflight_update(context: _DailyRunContext) -> list[dict[str, Any]]:
    if context.build_plan:
        return []
    return [
        refresh_labels_for_day(
            context.store,
            plan.trade_date,
            recover_event_key=context.recover_event,
            dry_run=True,
            **context.refresh_kwargs,
        ).to_dict()
        for plan in context.refresh_plans
    ]


def _wal_pending(store: PanelDataStore) -> bool:
    return store.wal_path.exists() and store.wal_path.stat().st_size > 0


def _has_pending_work(context: _DailyRunContext, preflight: Sequence[Mapping[str, Any]]) -> bool:
    refresh_pending = any(item["status"] != "no_op" for item in preflight)
    wal_pending = bool(context.recover_event) and _wal_pending(context.store)
    return bool(context.build_plan) or refresh_pending or wal_pending


def _commit_replayed_wal(store: PanelDataStore) -> None:
    with store.write_lock():
        store._wal.sync_manifest_under_lock()
        if _wal_pending(store):
            store._wal.save()


def _readiness_for_run(context: _DailyRunContext) -> list[dict[str, Any]]:
    dates = [context.effective_start] if context.recover_event else context.candidates
    return [
        validate_label_day_readiness(
            context.store,
            date,
            consumer=True,
            trading_dates=context.calendar,
            input_watermark=context.watermark,
        )
        for date in dates
    ]


def _write_assets(context: _DailyRunContext, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if context.build_plan and not context.store.manifest.snapshot()["columns"]:
        bootstrap_manifest(context.store)
    for date, groups in sorted(context.build_plan.items()):
        started = time.monotonic()
        context.builder.build_daily(TradeDate.from_str(date), groups, context.store)
        results.append(
            {
                "trade_date": date,
                "action": "build_daily",
                "groups": list(groups),
                "status": "built",
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
        )
    executable_plans = plan_label_refreshes(
        context.store,
        [context.effective_start] if context.recover_event else context.candidates,
        context.calendar,
        context.watermark,
        start_date=context.effective_start if context.recover_event else STRUCTURAL_FROM,
        validate_panel=True,
    )
    for plan in executable_plans:
        result = refresh_labels_for_day(
            context.store,
            plan.trade_date,
            recover_event_key=context.recover_event,
            **context.refresh_kwargs,
        )
        results.append(result.to_dict())
    if context.recover_event:
        _commit_replayed_wal(context.store)
    return _readiness_for_run(context)


def _verify_context_identity(context: _DailyRunContext) -> None:
    _verify_run_identities(
        context.input_identities,
        Path(context.loader.storage.root_path),
        producer=context.producer,
        workspace=context.workspace,
        producer_evidence=context.producer_evidence,
        workspace_evidence=context.workspace_evidence,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"run 恢复证据读取失败: {path}（{exc}）") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"run 恢复证据不是 JSON 对象: {path}")
    return payload


def _validate_snapshot_identity(context: _DailyRunContext, snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("input_files") != context.input_identities:
        raise RuntimeError("当前输入身份与原 run 快照不一致，拒绝补写报告")
    if snapshot.get("producer") != context.producer_evidence:
        raise RuntimeError("当前生产者身份与原 run 快照不一致，拒绝补写报告")
    original_workspace = snapshot.get("workspace", {})
    for key in ("git_commit", "source_digest", "environment"):
        if original_workspace.get(key) != context.workspace_evidence.get(key):
            raise RuntimeError(f"当前工作区身份字段 {key} 与原 run 快照不一致，拒绝补写报告")
    original_manifest = context.backup / "manifest.json"
    if original_manifest.exists():
        actual_source = Manifest(original_manifest).content_fingerprint()
        if actual_source != snapshot.get("source_content_fingerprint"):
            raise RuntimeError("原 run manifest 与快照源指纹不一致，拒绝补写报告")


def _validate_snapshot_events(context: _DailyRunContext, snapshot: Mapping[str, Any]) -> None:
    current = context.store.manifest.snapshot()
    for key in ("label_refresh_intents", "label_refreshes"):
        original_events = snapshot.get(key, [])
        current_events = current.get(key, [])
        if current_events[: len(original_events)] != original_events:
            raise RuntimeError(f"当前 {key} 已偏离原 run append-only 历史，拒绝补写报告")
    if len(current.get("repairs", [])) != snapshot.get("repairs_count"):
        raise RuntimeError("当前 repairs 数量与原 run 快照不一致，拒绝补写报告")


def _validate_snapshot_files(context: _DailyRunContext, checkpoint: Mapping[str, Any]) -> None:
    for relative in checkpoint.get("planned_new_files", []):
        if not (context.store.root / relative).is_file():
            raise RuntimeError(f"原 run 计划产物缺失，拒绝补写报告: {relative}")
    for relative in checkpoint.get("affected_existing_files", []):
        if not (context.backup / "files" / relative).is_file():
            raise RuntimeError(f"原 run 受影响文件备份缺失，拒绝补写报告: {relative}")


def _validate_original_run_snapshot(context: _DailyRunContext) -> dict[str, Any]:
    checkpoint = _read_json_object(context.backup / "checkpoint.json")
    snapshot = _read_json_object(context.backup / "snapshot.json")
    _validate_snapshot_identity(context, snapshot)
    _validate_snapshot_events(context, snapshot)
    _validate_snapshot_files(context, checkpoint)
    return checkpoint


def _handle_noop(
    context: _DailyRunContext, preflight: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    checkpoint_path = context.backup / "checkpoint.json"
    snapshot_path = context.backup / "snapshot.json"
    checkpoint = _read_json_object(checkpoint_path) if checkpoint_path.exists() else {}
    try:
        readiness = _readiness_for_run(context)
        _verify_context_identity(context)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        _write_failure_report(
            context.base,
            exc,
            checkpoint,
            preflight,
            input_identity_unchanged=(False if isinstance(exc, InputIdentityDriftError) else None),
        )
        raise
    report_path = Path(context.base["report_path"])
    existing_report = _read_json_object(report_path) if report_path.exists() else None
    if existing_report is not None and existing_report.get("status") != "failed":
        if existing_report.get("status") not in {"executed", "no_op", "report_recovered"}:
            raise RuntimeError(
                f"既有 run 报告状态未知，拒绝自动补写: {existing_report.get('status')}"
            )
        return {
            **context.base,
            "status": "no_op",
            "results": list(preflight),
            "readiness": readiness,
            "input_identity_unchanged": True,
        }
    if existing_report is None and not checkpoint_path.exists() and not snapshot_path.exists():
        return {
            **context.base,
            "status": "no_op",
            "results": list(preflight),
            "readiness": readiness,
            "input_identity_unchanged": True,
        }
    if not checkpoint_path.exists() or not snapshot_path.exists():
        raise RuntimeError("原 run 报告缺失或失败且快照证据不完整，拒绝自动补写")
    checkpoint = _validate_original_run_snapshot(context)
    previous_failure = None
    if existing_report is not None:
        previous_failure = {
            "status": existing_report.get("status"),
            "failure_reason": existing_report.get("failure_reason"),
            "generated_at": existing_report.get("generated_at"),
        }
    report = {
        **context.base,
        "status": "report_recovered",
        "recovered": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": checkpoint,
        "results": list(preflight),
        "readiness": readiness,
        "previous_failure": previous_failure,
        "input_identity_unchanged": True,
    }
    _write_json_atomic(report_path, report)
    return report


def _execute_update(
    context: _DailyRunContext, preflight: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not _has_pending_work(context, preflight):
        return _handle_noop(context, preflight)
    _verify_context_identity(context)
    checkpoint = _save_planned_snapshot(
        context.store,
        context.backup,
        context.build_plan,
        context.refresh_plans,
        context.input_identities,
        context.producer_evidence,
        context.workspace_evidence,
    )
    results: list[dict[str, Any]] = []
    try:
        readiness = _write_assets(context, results)
        _verify_context_identity(context)
        report = {
            **context.base,
            "status": "executed",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": checkpoint,
            "results": results,
            "readiness": readiness,
            "input_identity_unchanged": True,
        }
        _write_json_atomic(Path(context.base["report_path"]), report)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        identity_state = False if isinstance(exc, InputIdentityDriftError) else None
        _write_failure_report(
            context.base, exc, checkpoint, results, input_identity_unchanged=identity_state
        )
        raise
    return report


def update_daily_store(
    data_root: Path | str,
    backup_root: Path | str,
    *,
    start_date: str = STRUCTURAL_FROM,
    execute: bool = False,
    refresh_only: bool = False,
    recover_event: str | None = None,
    authorized_minutes: float = 30.0,
    store: PanelDataStore | None = None,
    loader: Any = None,
    builder: Any = None,
    producer: Mapping[str, Any] | None = None,
    workspace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """生成日更计划或显式执行；所有写动作都发生在耗时预检与同代快照之后。"""
    context = _prepare_update(
        data_root,
        backup_root,
        start_date=start_date,
        execute=execute,
        refresh_only=refresh_only,
        recover_event=recover_event,
        authorized_minutes=authorized_minutes,
        store=store,
        loader=loader,
        builder=builder,
        producer=producer,
        workspace=workspace,
    )
    preflight = _preflight_update(context)
    if not execute:
        if _has_pending_work(context, preflight):
            return {**context.base, "status": "planned", "preflight": preflight}
        readiness = _readiness_for_run(context)
        _verify_context_identity(context)
        return {
            **context.base,
            "status": "no_op",
            "preflight": preflight,
            "readiness": readiness,
            "input_identity_unchanged": True,
        }
    return _execute_update(context, preflight)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = update_daily_store(
        args.data_root,
        args.backup_root,
        start_date=args.start_date,
        execute=args.execute,
        refresh_only=args.refresh_only,
        recover_event=args.recover_event,
        authorized_minutes=args.authorized_minutes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
