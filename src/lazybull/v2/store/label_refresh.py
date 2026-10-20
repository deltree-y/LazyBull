# -*- coding: utf-8 -*-
"""v2 标签专用 forming→sealed 刷新、恰一次审计与日级 readiness。

本模块只读 panel 与精确端点价格，不调用整日特征构建，也不复制 raw/neu 公式。
单分区 replace 是原子的；三表和 manifest 不构成单日事务，崩溃窗必须显式恢复。
"""

from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.lazybull.common.config import get_shenwan_level, normalize_shenwan_level
from src.lazybull.features.labels import compute_forward_returns
from src.lazybull.features.neutralization import apply_industry_neutralization
from src.lazybull.v2.common.types import FeatureQuery, TradeDate
from src.lazybull.v2.store.column_groups import LABEL_TABLES, PANEL_GROUPS
from src.lazybull.v2.store.labels_builder import extract_labels, maturity_status_for
from src.lazybull.v2.store.manifest import (
    sha256_16_of_bytes,
    sha256_16_of_file,
    validate_label_output_fingerprints,
    validate_label_refresh_completion,
)
from src.lazybull.v2.store.row_domain import STRUCTURAL_FROM, row_domain_for_date
from src.lazybull.v2.store.store_identity import (
    InputIdentityDriftError,
    canonical_json_sha256,
    frame_content_digest,
    producer_identity,
    verify_file_identities,
)

__all__ = [
    "LabelRefreshPlan",
    "LabelRefreshResult",
    "LabelRefreshRecoveryError",
    "plan_label_refreshes",
    "refresh_labels_for_day",
    "validate_label_day_readiness",
]

_HORIZONS = [5, 10, 20]
_LOOKBACK_WINDOWS = [5, 10, 20]
_LABEL_COLUMNS = ["ts_code", "trade_date", "label_value", "neu_label_value", "maturity_status"]
_LABEL_NAMES = tuple(LABEL_TABLES)
_REPO_ROOT = Path(__file__).resolve().parents[4]


class LabelRefreshRecoveryError(RuntimeError):
    """replace→WAL 或盘上/登记不一致；普通入口禁止自动修复。"""


@dataclass(frozen=True)
class LabelRefreshPlan:
    """一个已到期信号日的刷新计划。"""

    trade_date: str
    buy_date: str
    endpoint_date: str
    input_watermark: str
    target_status: str
    row_domain: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelRefreshResult:
    """日级刷新结果；可直接序列化进 run 报告。"""

    trade_date: str
    buy_date: str
    endpoint_date: str
    input_watermark: str
    expected_status: str
    actual_status: str
    event_key: str
    input_digest: str
    producer_digest: str
    row_domain: str
    status: str
    rows: dict[str, int]
    keyset_valid: bool
    output_fingerprints: dict[str, str]
    cross_seam_impact: dict[str, Any]
    orphan_intents: list[dict[str, Any]]
    input_files: list[dict[str, Any]]
    recovered: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _endpoints(
    trade_date: str, trading_dates: Sequence[str], watermark: str
) -> tuple[int, dict[str, str]]:
    dates = [str(value) for value in trading_dates]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError("交易日历必须升序且无重复")
    try:
        index = dates.index(str(trade_date))
    except ValueError as exc:
        raise ValueError(f"日期 {trade_date} 不在交易日历中") from exc
    required_indices = {"B": index + 1, "E5": index + 6, "E10": index + 11, "E20": index + 21}
    if required_indices["E20"] >= len(dates):
        raise ValueError(f"{trade_date} 的 E20 不在完整日历中，禁止专用刷新")
    result = {name: dates[position] for name, position in required_indices.items()}
    if result["E20"] > str(watermark):
        raise ValueError(f"{trade_date} 的 E20={result['E20']} 超实际加载水位 {watermark}")
    if maturity_status_for(str(trade_date), dates, data_end=str(watermark)) != "sealed":
        raise ValueError(f"{trade_date} 在完整日历与水位 {watermark} 下仍不可 sealed")
    return index, result


def _plan_for_date(
    store: Any,
    trade_date: str,
    dates: Sequence[str],
    watermark: str,
    *,
    validate_panel: bool,
) -> LabelRefreshPlan | None:
    try:
        _, endpoints = _endpoints(trade_date, dates, watermark)
    except ValueError as exc:
        if "E20" in str(exc) and ("超实际加载水位" in str(exc) or "不在完整日历" in str(exc)):
            return None
        raise
    if validate_panel:
        _validate_panel_groups(store, trade_date, check_keys=False)
    return LabelRefreshPlan(
        trade_date=str(trade_date),
        buy_date=endpoints["B"],
        endpoint_date=endpoints["E20"],
        input_watermark=str(watermark),
        target_status="sealed",
        row_domain=row_domain_for_date(str(trade_date)),
    )


def plan_label_refreshes(
    store: Any,
    candidate_dates: Sequence[str],
    trading_dates: Sequence[str],
    input_watermark: str,
    *,
    start_date: str = STRUCTURAL_FROM,
    validate_panel: bool = True,
) -> list[LabelRefreshPlan]:
    """扫描起点至最大候选日，禁止省略已到期的未完成日或跨日封存。

    空候选直接返回空计划；范围内已完成和未到期日可从候选中省略。``start_date``
    是受控恢复限定扫描左边界的唯一入口，候选中的历史影子日仍不参与生产计划。
    """
    start = str(start_date)
    candidates = sorted({str(date) for date in candidate_dates if str(date) >= start})
    if not candidates:
        return []
    candidate_set = set(candidates)
    scan_dates = sorted(
        {str(date) for date in trading_dates if start <= str(date) <= candidates[-1]}
        | candidate_set
    )
    plans: list[LabelRefreshPlan] = []
    first_incomplete: str | None = None
    snapshot = store.manifest.snapshot()
    completions = snapshot.get("label_refreshes", [])
    for date in scan_dates:
        is_candidate = date in candidate_set
        plan = _plan_for_date(
            store,
            date,
            trading_dates,
            str(input_watermark),
            validate_panel=validate_panel and is_candidate,
        )
        if plan is None:
            continue
        complete = _day_has_sealed_completion(snapshot, date, completions)
        if first_incomplete is not None and _any_sealed_through(snapshot, date):
            raise RuntimeError(f"封存水位乱序：{first_incomplete} 尚未完成，{date} 已封存")
        if not is_candidate:
            if not complete:
                raise RuntimeError(
                    f"封存水位乱序：{date} 已到 E20 水位但三表封存或完成审计缺失，"
                    "且未出现在 candidate，禁止缺早日跳晚日"
                )
            continue
        plans.append(plan)
        if not complete and first_incomplete is None:
            first_incomplete = date
    return plans


def _day_has_sealed_completion(
    snapshot: Mapping[str, Any], trade_date: str, completions: Sequence[Mapping[str, Any]]
) -> bool:
    labels = snapshot["labels"]
    sealed = all(
        str(labels.get(name, {}).get("sealed_through") or "") >= trade_date for name in _LABEL_NAMES
    )
    return sealed and any(
        record.get("trade_date") == trade_date and record.get("target_status") == "sealed"
        for record in completions
    )


def _any_sealed_through(snapshot: Mapping[str, Any], trade_date: str) -> bool:
    return any(
        str(entry.get("sealed_through") or "") >= trade_date
        for entry in snapshot["labels"].values()
    )


def _required_core_columns(shenwan_level: str) -> list[str]:
    columns = ["trade_date", "ts_code", "sw_industry", "tradable"]
    if shenwan_level == "l2":
        columns.extend(["sw_industry_code", "sw_l1_code"])
    elif shenwan_level == "l3":
        columns.extend(["sw_industry_code", "sw_l2_code", "sw_l1_code"])
    return columns


def _load_panel_core(
    store: Any, trade_date: str, shenwan_level: str | None
) -> tuple[pd.DataFrame, str]:
    level = normalize_shenwan_level(shenwan_level or get_shenwan_level())
    query = FeatureQuery(
        columns=_required_core_columns(level),
        start_date=TradeDate.from_str(trade_date),
        end_date=TradeDate.from_str(trade_date),
    )
    frame = store.load_features(query).df.reset_index()
    if frame.empty:
        raise RuntimeError(f"{trade_date} panel core 为空，refresh-only 不得伪造行集")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise RuntimeError(f"{trade_date} panel core 存在重复键")
    dates = frame["trade_date"].astype(str).unique()
    if len(dates) != 1 or dates[0] != trade_date:
        raise RuntimeError(f"{trade_date} panel core 日期域不一致: {dates}")
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
    return frame, level


def _panel_partition_path(store: Any, trade_date: str, group: str) -> tuple[str, Path]:
    month = f"{trade_date[:4]}-{trade_date[4:6]}"
    archive = store.archive_dir / month / f"{group}.parquet"
    if archive.exists():
        return f"panel_archive/{month}/{group}.parquet", archive
    hot = store.panel_dir / trade_date / f"{group}.parquet"
    return f"panel/{trade_date}/{group}.parquet", hot


def _validate_panel_groups(store: Any, trade_date: str, *, check_keys: bool) -> pd.Index | None:
    snapshot = store.manifest.snapshot()
    core_keys: pd.Index | None = None
    for group in PANEL_GROUPS:
        rel_path, path = _panel_partition_path(store, trade_date, group)
        meta = snapshot["partitions"].get(rel_path)
        if meta is None or not path.is_file():
            raise RuntimeError(f"{trade_date} 缺 panel 八族分区或登记: {group}；先运行 build_daily")
        if meta.get("row_domain") != row_domain_for_date(trade_date):
            raise RuntimeError(f"panel 分区行域不一致: {rel_path}")
        if sha256_16_of_file(path) != meta.get("sha256_16"):
            raise LabelRefreshRecoveryError(f"panel 登记指纹与文件不一致: {rel_path}")
        if not check_keys:
            continue
        frame = pd.read_parquet(path, columns=["trade_date", "ts_code"])
        day = frame[frame["trade_date"].astype(str) == trade_date]
        if day.duplicated(["trade_date", "ts_code"]).any():
            raise RuntimeError(f"panel 分区存在重复键: {rel_path}")
        keys = pd.MultiIndex.from_frame(day[["trade_date", "ts_code"]].astype(str))
        if group == "core":
            core_keys = keys
        elif keys.difference(core_keys).size or core_keys.difference(keys).size:
            raise RuntimeError(f"panel 八族键集与 core 不一致: {rel_path}")
    return core_keys


def _slice_endpoint_frames(
    daily_adj_dict: Mapping[str, pd.DataFrame], endpoints: Mapping[str, str], panel: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    codes = set(panel["ts_code"].astype(str))
    slices: dict[str, pd.DataFrame] = {}
    for name, date in endpoints.items():
        source = daily_adj_dict.get(date)
        price_column = "close_adj" if name == "B" else "open_adj"
        if source is None or source.empty:
            raise ValueError(f"必需端点 {name}={date} 的 clean/daily 帧缺失或为空")
        required = {"trade_date", "ts_code", price_column}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"端点 {name}={date} 缺列 {missing}")
        dates = source["trade_date"].astype(str).unique()
        if len(dates) != 1 or dates[0] != date:
            raise ValueError(f"端点 {name} 的文件日期与 {date} 不一致: {dates}")
        sliced = source.loc[
            source["ts_code"].astype(str).isin(codes),
            ["trade_date", "ts_code", price_column],
        ].copy()
        sliced["trade_date"] = sliced["trade_date"].astype(str)
        sliced["ts_code"] = sliced["ts_code"].astype(str)
        if sliced.duplicated(["ts_code"]).any():
            raise ValueError(f"端点 {name}={date} 在 panel 键域内存在重复 ts_code")
        slices[name] = sliced.sort_values("ts_code", kind="stable").reset_index(drop=True)
    return slices


def _input_digest(
    panel: pd.DataFrame,
    calendar_slice: Sequence[str],
    endpoints: Mapping[str, pd.DataFrame],
    shenwan_level: str,
) -> str:
    payload = {
        "panel_core": frame_content_digest(panel),
        "calendar_slice": list(calendar_slice),
        "endpoint_rows": {name: frame_content_digest(frame) for name, frame in endpoints.items()},
        "config": {
            "shenwan_level": shenwan_level,
            "horizons": _HORIZONS,
            "lookback_windows": _LOOKBACK_WINDOWS,
        },
    }
    return canonical_json_sha256(payload)


def _build_label_tables(
    panel: pd.DataFrame,
    trade_date: str,
    trading_dates: Sequence[str],
    current_idx: int,
    endpoint_frames: Mapping[str, pd.DataFrame],
    watermark: str,
    shenwan_level: str,
) -> dict[str, pd.DataFrame]:
    raw = compute_forward_returns(
        panel[["trade_date", "ts_code"]],
        trade_date,
        list(trading_dates),
        current_idx,
        _HORIZONS,
        daily_adj_dict={
            date: endpoint_frames[name]
            for name, date in {
                "B": trading_dates[current_idx + 1],
                "E5": trading_dates[current_idx + 6],
                "E10": trading_dates[current_idx + 11],
                "E20": trading_dates[current_idx + 21],
            }.items()
        },
    )
    wide = panel.merge(raw, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    wide = apply_industry_neutralization(
        wide, _HORIZONS, _LOOKBACK_WINDOWS, shenwan_level=shenwan_level
    )
    required = {f"{prefix}y_ret_{horizon}" for horizon in _HORIZONS for prefix in ("", "neu_")}
    missing = sorted(required - set(wide.columns))
    if missing:
        raise RuntimeError(f"raw+neu 重算缺列 {missing}，禁止只写 raw 标签")
    if wide["y_ret_20"].notna().sum() == 0:
        raise RuntimeError(f"{trade_date} sealed 日 y_ret_20 非空行数为 0，readiness 拒绝")
    tables = extract_labels(wide, trading_dates, data_end=watermark)
    if set(tables) != set(_LABEL_NAMES):
        raise RuntimeError(f"标签表集合不完整: {sorted(tables)}")
    return tables


def _cross_seam_impact(tables: Mapping[str, pd.DataFrame], panel: pd.DataFrame) -> dict[str, Any]:
    wide = panel[["ts_code", "sw_industry"]].copy()
    for name in _LABEL_NAMES:
        wide[name] = tables[name]["label_value"].to_numpy()
    result: dict[str, Any] = {}
    for shorter in ("y_ret_5", "y_ret_10"):
        mask = wide["y_ret_20"].isna() & wide[shorter].notna()
        result[shorter] = {
            "affected_rows": int(mask.sum()),
            "by_industry": {
                str(key): int(value)
                for key, value in wide.loc[mask].groupby("sw_industry", dropna=False).size().items()
            },
        }
    return result


def _parquet_fingerprint(frame: pd.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return sha256_16_of_bytes(buffer.getvalue())


def _expected_fingerprints(tables: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    return {name: _parquet_fingerprint(frame) for name, frame in tables.items()}


def _label_path(store: Any, label_name: str, trade_date: str) -> Path:
    return store.labels_dir / label_name / f"{trade_date}.parquet"


def _validate_label_registration(
    name: str,
    trade_date: str,
    status: str,
    fingerprint: str,
    snapshot: Mapping[str, Any],
    output_fingerprint: str | None,
) -> None:
    meta = snapshot["labels"].get(name, {}).get("partitions", {}).get(f"{trade_date}.parquet")
    if meta is None or meta.get("sha256_16") != fingerprint:
        raise LabelRefreshRecoveryError(f"标签文件与 manifest 登记不一致: {name}/{trade_date}")
    if meta.get("row_domain") != row_domain_for_date(trade_date):
        raise RuntimeError(f"标签行域不一致: {name}/{trade_date}")
    if output_fingerprint is not None and output_fingerprint != fingerprint:
        raise LabelRefreshRecoveryError(f"标签输出指纹与完成审计不一致: {name}/{trade_date}")
    sealed_through = snapshot["labels"].get(name, {}).get("sealed_through")
    if status == "sealed" and str(sealed_through or "") < trade_date:
        raise LabelRefreshRecoveryError(f"标签 sealed_through 未覆盖 {name}/{trade_date}")
    if status == "forming" and str(sealed_through or "") >= trade_date:
        raise LabelRefreshRecoveryError(
            f"forming 标签落在 sealed_through 水位内: {name}/{trade_date}"
        )


def _read_and_validate_label(
    store: Any,
    name: str,
    trade_date: str,
    panel_keys: pd.MultiIndex,
    snapshot: Mapping[str, Any],
    expected: pd.DataFrame | None,
    output_fingerprint: str | None,
) -> tuple[pd.DataFrame, str, str]:
    path = _label_path(store, name, trade_date)
    if not path.is_file():
        raise RuntimeError(f"缺标签分区: {name}/{trade_date}")
    actual = pd.read_parquet(path)
    if list(actual.columns) != _LABEL_COLUMNS:
        raise RuntimeError(f"标签 schema 不一致: {name}/{trade_date}: {list(actual.columns)}")
    if actual.duplicated(["trade_date", "ts_code"]).any():
        raise RuntimeError(f"标签存在重复键: {name}/{trade_date}")
    actual["trade_date"] = actual["trade_date"].astype(str)
    keys = pd.MultiIndex.from_frame(actual[["trade_date", "ts_code"]].astype(str))
    if keys.difference(panel_keys).size or panel_keys.difference(keys).size:
        raise RuntimeError(f"标签与 panel 键集不一致: {name}/{trade_date}")
    status_values = actual["maturity_status"].astype(str).unique()
    if len(status_values) != 1:
        raise RuntimeError(f"标签单日混合成熟度: {name}/{trade_date}")
    status = str(status_values[0])
    if status not in {"forming", "sealed"}:
        raise RuntimeError(f"标签 maturity_status 非法: {name}/{trade_date}: {status!r}")
    fingerprint = sha256_16_of_file(path)
    _validate_label_registration(
        name, trade_date, status, fingerprint, snapshot, output_fingerprint
    )
    if expected is not None:
        try:
            pd.testing.assert_frame_equal(
                actual.reset_index(drop=True), expected.reset_index(drop=True), check_exact=True
            )
        except AssertionError as exc:
            raise RuntimeError(f"标签读回数值/schema 不一致: {name}/{trade_date}（{exc}）") from exc
    return actual, status, fingerprint


def _read_and_validate_labels(
    store: Any,
    trade_date: str,
    panel_keys: pd.MultiIndex,
    *,
    expected_tables: Mapping[str, pd.DataFrame] | None = None,
    output_fingerprints: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, str], dict[str, int]]:
    if output_fingerprints is not None:
        validate_label_output_fingerprints(output_fingerprints, context=f"{trade_date} 完成审计")
    snapshot = store.manifest.snapshot()
    statuses: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    rows: dict[str, int] = {}
    for name in _LABEL_NAMES:
        expected = expected_tables.get(name) if expected_tables is not None else None
        output_fp = output_fingerprints.get(name) if output_fingerprints is not None else None
        actual, status, fingerprint = _read_and_validate_label(
            store, name, trade_date, panel_keys, snapshot, expected, output_fp
        )
        statuses[name] = status
        fingerprints[name] = fingerprint
        rows[name] = len(actual)
    if len(set(statuses.values())) != 1:
        raise RuntimeError(f"三表成熟度不一致: {statuses}")
    status = statuses[_LABEL_NAMES[0]]
    if status == "sealed":
        y20 = pd.read_parquet(_label_path(store, "y_ret_20", trade_date), columns=["label_value"])
        if y20["label_value"].notna().sum() == 0:
            raise RuntimeError(f"{trade_date} sealed 日 y_ret_20 非空行数为 0")
    return status, fingerprints, rows


def _completion_for_key(store: Any, event_key: str) -> dict[str, Any] | None:
    records = [r for r in store.manifest.label_refreshes() if r.get("event_key") == event_key]
    if len(records) > 1:
        raise RuntimeError(f"完成审计 event_key 重复: {event_key}")
    return records[0] if records else None


def _incomplete_prepares(store: Any, trade_date: str) -> list[dict[str, Any]]:
    """返回当日既未完成、也未被 superseded/orphan 事件终结的 prepare。"""
    intents = store.manifest.label_refresh_intents()
    completed = {r.get("event_key") for r in store.manifest.label_refreshes()}
    terminated = {
        record.get("event_key")
        for record in intents
        if record.get("record_type") in {"superseded", "orphan"}
    }
    excluded = completed | terminated
    return [
        record
        for record in intents
        if record.get("record_type") == "prepare"
        and record.get("trade_date") == trade_date
        and record.get("event_key") not in excluded
    ]


def _intent_semantic(record: Mapping[str, Any]) -> dict[str, Any]:
    """剔除审计易变字段，保留同身份事件必须完全一致的业务语义。"""
    volatile = {"recorded_at", "run_id", "refreshed_at", "recovered"}
    return {key: value for key, value in record.items() if key not in volatile}


def _append_intent_once(store: Any, record: Mapping[str, Any]) -> bool:
    """同语义追加恰一次；同身份语义冲突必须在写 WAL 前 fail-closed。"""
    identity = ("record_type", "event_key", "related_event_key", "superseded_by")
    for existing in store.manifest.label_refresh_intents():
        if all(existing.get(key) == record.get(key) for key in identity):
            if _intent_semantic(existing) != _intent_semantic(record):
                raise ValueError(f"刷新意图审计身份冲突: {record.get('event_key')!r}")
            return False
    return store._wal.append_label_refresh_intent(dict(record))


def _append_completion_once(store: Any, record: Mapping[str, Any]) -> bool:
    if _completion_for_key(store, str(record["event_key"])) is not None:
        return False
    return store._wal.append_label_refresh_completion(dict(record))


def _orphan_intents(store: Any, trade_date: str) -> list[dict[str, Any]]:
    return [
        record
        for record in store.manifest.label_refresh_intents()
        if record.get("trade_date") == trade_date and record.get("record_type") != "prepare"
    ]


def _validate_existing_label_schemas(
    store: Any, trade_date: str, panel_keys: pd.MultiIndex
) -> None:
    for name in _LABEL_NAMES:
        path = _label_path(store, name, trade_date)
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if list(frame.columns) != _LABEL_COLUMNS:
            raise RuntimeError(f"既有标签 schema 冲突: {name}/{trade_date}")
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise RuntimeError(f"既有标签存在重复键: {name}/{trade_date}")
        keys = pd.MultiIndex.from_frame(frame[["trade_date", "ts_code"]].astype(str))
        if keys.difference(panel_keys).size or panel_keys.difference(keys).size:
            raise RuntimeError(f"既有标签与 panel 键集冲突: {name}/{trade_date}")
        statuses = set(frame["maturity_status"].astype(str))
        if len(statuses) != 1 or not statuses <= {"forming", "sealed"}:
            raise RuntimeError(f"既有标签单日成熟度冲突: {name}/{trade_date}: {statuses}")


def _assert_no_stale_final_output(store: Any, trade_date: str) -> None:
    snapshot = store.manifest.snapshot()
    if _any_sealed_through(snapshot, trade_date):
        raise LabelRefreshRecoveryError(f"{trade_date} 已有 sealed 水位，旧意图不得按孤儿略过")
    for name in _LABEL_NAMES:
        path = _label_path(store, name, trade_date)
        if path.exists():
            frame = pd.read_parquet(path, columns=["maturity_status"])
            if (frame["maturity_status"].astype(str) == "sealed").any():
                raise LabelRefreshRecoveryError(f"{trade_date} 已有最终 sealed 输出，须受控恢复")


def _supersede_stale_prepares(
    store: Any, prepares: Sequence[Mapping[str, Any]], event_key: str, run_id: str | None
) -> None:
    for prepare in prepares:
        if prepare.get("event_key") == event_key:
            continue
        _append_intent_once(
            store,
            {
                "record_type": "superseded",
                "event_key": prepare["event_key"],
                "related_event_key": event_key,
                "superseded_by": event_key,
                "trade_date": prepare["trade_date"],
                "reason": "输入或生产者身份变化，旧意图尚未产生 sealed 输出",
                "run_id": run_id,
            },
        )


def _crash_window_labels(
    store: Any, trade_date: str, expected: Mapping[str, str]
) -> tuple[list[str], list[str]]:
    snapshot = store.manifest.snapshot()
    recoverable: list[str] = []
    unrecoverable: list[str] = []
    for name in _LABEL_NAMES:
        path = _label_path(store, name, trade_date)
        meta = snapshot["labels"].get(name, {}).get("partitions", {}).get(f"{trade_date}.parquet")
        registered = meta.get("sha256_16") if meta else None
        if not path.exists():
            if registered is not None:
                unrecoverable.append(name)
            continue
        actual = sha256_16_of_file(path)
        if actual == expected[name] and registered != actual:
            recoverable.append(name)
        elif actual != expected[name] and (registered is None or registered != actual):
            unrecoverable.append(name)
    return recoverable, unrecoverable


def _prepare_record(
    plan: LabelRefreshPlan,
    event_key: str,
    input_digest: str,
    producer_digest: str,
    run_id: str | None,
) -> dict[str, Any]:
    return {
        "record_type": "prepare",
        "event_key": event_key,
        "trade_date": plan.trade_date,
        "row_domain": plan.row_domain,
        "input_digest": input_digest,
        "producer_digest": producer_digest,
        "target_status": plan.target_status,
        "input_watermark": plan.input_watermark,
        "buy_date": plan.buy_date,
        "endpoint_date": plan.endpoint_date,
        "run_id": run_id,
    }


def _completion_record(
    prepare: Mapping[str, Any],
    fingerprints: Mapping[str, str],
    rows: Mapping[str, int],
    impact: Mapping[str, Any],
    *,
    recovered: bool,
) -> dict[str, Any]:
    return {
        **{key: value for key, value in prepare.items() if key != "record_type"},
        "input_event_key": prepare["event_key"],
        "output_fingerprints": dict(fingerprints),
        "rows": dict(rows),
        "cross_seam_impact": dict(impact),
        "refreshed_at": datetime.now().isoformat(timespec="seconds"),
        "recovered": recovered,
    }


def _assert_input_identities(
    expected: Sequence[Mapping[str, Any]], base_dir: Path | str | None
) -> None:
    mismatches = verify_file_identities(expected, base_dir=base_dir)
    if mismatches:
        raise InputIdentityDriftError(f"运行前后输入身份变化: {mismatches}")


def _current_status(store: Any, trade_date: str) -> str:
    """汇总三表当前状态；部分表缺失用 partial_missing，禁止冒充完整状态。"""
    statuses: set[str] = set()
    existing_tables = 0
    for name in _LABEL_NAMES:
        path = _label_path(store, name, trade_date)
        if not path.exists():
            continue
        existing_tables += 1
        frame = pd.read_parquet(path, columns=["maturity_status"])
        statuses.update(frame["maturity_status"].astype(str))
    if existing_tables == 0:
        return "missing"
    if existing_tables != len(_LABEL_NAMES):
        return "partial_missing"
    return statuses.pop() if len(statuses) == 1 else "mixed"


def _result(
    plan: LabelRefreshPlan,
    event_key: str,
    input_digest: str,
    producer_digest: str,
    status: str,
    rows: Mapping[str, int],
    fingerprints: Mapping[str, str],
    impact: Mapping[str, Any],
    orphans: Sequence[Mapping[str, Any]],
    input_files: Sequence[Mapping[str, Any]],
    recovered: bool,
    started: float,
    actual_status: str = "sealed",
) -> LabelRefreshResult:
    plan_values = plan.to_dict()
    plan_values.pop("target_status")
    return LabelRefreshResult(
        **plan_values,
        expected_status=plan.target_status,
        actual_status=actual_status,
        event_key=event_key,
        input_digest=input_digest,
        producer_digest=producer_digest,
        status=status,
        rows=dict(rows),
        keyset_valid=True,
        output_fingerprints=dict(fingerprints),
        cross_seam_impact=dict(impact),
        orphan_intents=[dict(record) for record in orphans],
        input_files=[dict(record) for record in input_files],
        recovered=recovered,
        elapsed_seconds=round(time.monotonic() - started, 6),
    )


def _completed_refresh_result(
    store: Any,
    plan: LabelRefreshPlan,
    completion: Mapping[str, Any],
    event_payload: Mapping[str, Any],
    event_key: str,
    input_digest: str,
    producer_digest: str,
    panel_keys: pd.MultiIndex,
    input_identities: Sequence[Mapping[str, Any]],
    input_base_dir: Path | str | None,
    started: float,
) -> LabelRefreshResult:
    validate_label_refresh_completion(completion)
    if any(completion.get(key) != value for key, value in event_payload.items()):
        raise RuntimeError(f"完成审计与当前事件内容不一致: {event_key}")
    _, fingerprints, rows = _read_and_validate_labels(
        store,
        plan.trade_date,
        panel_keys,
        output_fingerprints=completion["output_fingerprints"],
    )
    _assert_input_identities(input_identities, input_base_dir)
    return _result(
        plan,
        event_key,
        input_digest,
        producer_digest,
        "no_op",
        rows,
        fingerprints,
        completion.get("cross_seam_impact", {}),
        _orphan_intents(store, plan.trade_date),
        input_identities,
        False,
        started,
    )


def refresh_labels_for_day(
    store: Any,
    trade_date: str,
    *,
    trading_dates: Sequence[str],
    daily_adj_dict: Mapping[str, pd.DataFrame],
    input_watermark: str,
    shenwan_level: str | None = None,
    producer: Mapping[str, Any] | None = None,
    input_identities: Sequence[Mapping[str, Any]] = (),
    input_base_dir: Path | str | None = None,
    run_id: str | None = None,
    recover_event_key: str | None = None,
    dry_run: bool = False,
) -> LabelRefreshResult:
    """重新核验精确端点并刷新一日三表；生产计划与历史影子共用此原语。"""
    started = time.monotonic()
    date = str(trade_date)
    producer_evidence = dict(producer or producer_identity(_REPO_ROOT))
    producer_digest = str(producer_evidence["producer_digest"])
    with store.write_lock():
        store._wal.sync_manifest_under_lock()
        current_idx, endpoints = _endpoints(date, trading_dates, str(input_watermark))
        plan = LabelRefreshPlan(
            date,
            endpoints["B"],
            endpoints["E20"],
            str(input_watermark),
            "sealed",
            row_domain_for_date(date),
        )
        _validate_panel_groups(store, date, check_keys=True)
        panel, level = _load_panel_core(store, date, shenwan_level)
        panel_keys = pd.MultiIndex.from_frame(panel[["trade_date", "ts_code"]].astype(str))
        endpoint_frames = _slice_endpoint_frames(daily_adj_dict, endpoints, panel)
        calendar_end_idx = current_idx + 22
        calendar_slice = list(trading_dates)[current_idx:calendar_end_idx]
        input_digest = _input_digest(panel, calendar_slice, endpoint_frames, level)
        event_payload = {
            "trade_date": date,
            "row_domain": plan.row_domain,
            "input_digest": input_digest,
            "producer_digest": producer_digest,
            "target_status": "sealed",
        }
        event_key = canonical_json_sha256(event_payload)
        completion = _completion_for_key(store, event_key)
        if completion is not None:
            return _completed_refresh_result(
                store,
                plan,
                completion,
                event_payload,
                event_key,
                input_digest,
                producer_digest,
                panel_keys,
                input_identities,
                input_base_dir,
                started,
            )
        prepares = _incomplete_prepares(store, date)
        same_prepare = next((r for r in prepares if r.get("event_key") == event_key), None)
        if recover_event_key is not None and (
            recover_event_key != event_key or same_prepare is None
        ):
            raise LabelRefreshRecoveryError(
                f"受控恢复事件必须与当前精确输入和既有 prepare 匹配: {recover_event_key}"
            )
        _validate_existing_label_schemas(store, date, panel_keys)
        if same_prepare is None:
            _assert_no_stale_final_output(store, date)
        tables = _build_label_tables(
            panel,
            date,
            list(trading_dates),
            current_idx,
            endpoint_frames,
            str(input_watermark),
            level,
        )
        expected = _expected_fingerprints(tables)
        impact = _cross_seam_impact(tables, panel)
        recoverable, unrecoverable = _crash_window_labels(store, date, expected)
        if unrecoverable:
            raise LabelRefreshRecoveryError(
                f"{date} 盘上/登记内容偏离精确重算，禁止 roll-forward: {unrecoverable}"
            )
        if recoverable and recover_event_key is None:
            raise LabelRefreshRecoveryError(
                f"{date} 检测到 replace→分区 WAL 崩溃窗 {recoverable}；"
                f"普通入口保留现场，须先以 --recover-event {event_key} dry-run"
            )
        if dry_run:
            _assert_input_identities(input_identities, input_base_dir)
            return _result(
                plan,
                event_key,
                input_digest,
                producer_digest,
                "recovery_planned" if recover_event_key else "planned",
                {name: len(frame) for name, frame in tables.items()},
                expected,
                impact,
                _orphan_intents(store, date),
                input_identities,
                bool(recover_event_key),
                started,
                actual_status=_current_status(store, date),
            )
        recovered = same_prepare is not None or bool(recoverable)
        prepare = same_prepare or _prepare_record(
            plan, event_key, input_digest, producer_digest, run_id
        )
        with store.defer_manifest_save():
            _supersede_stale_prepares(store, prepares, event_key, run_id)
            if same_prepare is None:
                _append_intent_once(store, prepare)
            for name in _LABEL_NAMES:
                store.append_labels(name, tables[name])
            _, fingerprints, rows = _read_and_validate_labels(
                store, date, panel_keys, expected_tables=tables, output_fingerprints=expected
            )
            completion = _completion_record(
                prepare, fingerprints, rows, impact, recovered=recovered
            )
            _append_completion_once(store, completion)
        _assert_input_identities(input_identities, input_base_dir)
        return _result(
            plan,
            event_key,
            input_digest,
            producer_digest,
            "recovered" if recovered else "refreshed",
            rows,
            fingerprints,
            impact,
            _orphan_intents(store, date),
            input_identities,
            recovered,
            started,
        )


def validate_label_day_readiness(
    store: Any,
    trade_date: str,
    *,
    consumer: bool = True,
    trading_dates: Sequence[str] | None = None,
    input_watermark: str | None = None,
) -> dict[str, Any]:
    """日级两阶段 readiness；consumer=False 仅用于崩溃后的恢复检查。"""
    date = str(trade_date)
    core_keys = _validate_panel_groups(store, date, check_keys=True)
    if core_keys is None:
        raise RuntimeError(f"{date} 缺 core 键域")
    status, fingerprints, rows = _read_and_validate_labels(store, date, core_keys)
    completion: dict[str, Any] | None = None
    if status == "sealed" and consumer and row_domain_for_date(date) == "structural":
        matches = []
        for record in store.manifest.label_refreshes():
            if record.get("trade_date") != date or record.get("target_status") != "sealed":
                continue
            try:
                validate_label_refresh_completion(record)
                _read_and_validate_labels(
                    store,
                    date,
                    core_keys,
                    output_fingerprints=record["output_fingerprints"],
                )
            except (RuntimeError, TypeError, ValueError):
                continue
            matches.append(record)
        if len(matches) != 1:
            raise RuntimeError(f"{date} sealed 日缺与当前三表内容匹配的唯一完成审计")
        completion = matches[0]
    if (
        consumer
        and status != "sealed"
        and row_domain_for_date(date) == "structural"
        and (trading_dates is None or input_watermark is None)
    ):
        raise ValueError("consumer readiness 对 forming 必须同时提供完整日历和实际输入水位")
    if (
        consumer
        and status != "sealed"
        and trading_dates is not None
        and input_watermark is not None
        and maturity_status_for(date, trading_dates, data_end=input_watermark) == "sealed"
    ):
        raise RuntimeError(f"{date} E20 已到水位仍为 {status}")
    return {
        "trade_date": date,
        "ready": True,
        "consumer": consumer,
        "status": status,
        "rows": rows,
        "output_fingerprints": fingerprints,
        "completion_event_key": completion.get("event_key") if completion else None,
        "orphan_intents": _orphan_intents(store, date),
    }
