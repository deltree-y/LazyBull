# -*- coding: utf-8 -*-
"""v2 P4 T5 日级证据共享件：D4→SLA 步骤映射、台账读写与真实缺口扫描。

把三件容易各自解释同一份数据的事情收口到唯一实现，避免驱动、门 6 与收口漂移：

1. **步骤映射**：D4 原始阶段（snapshot/raw/clean/cs_infer/panel/verify）→ SLA 字段
   （ensure_raw/ensure_clean/cs_infer/panel_label_refresh/readiness），并保留原始阶段
   记录供追溯；snapshot 单列前置步骤，只计总墙钟，不伪装成数据构建。
2. **台账形态**：累计恢复台账 ``list[record]``；每个 ``sla/daily/<日>.json`` 为单个
   ``record`` 对象；窗口加载器统一输出平坦 ``list[record]``。旧格式只允许单层识别并
   留下来源标记，禁止无限展开任意结构。
3. **真实指标**：重试、缺口、ensure 降级从协调者实际产物提取；产物缺失时 fail-closed
   记为降级（无法证明无降级），不固定写 ``0`` / ``[]`` / ``False``。

本模块只读写调用方给定的证据根与协调者产物目录，不写生产纸面根、数据根或模型目录。

# 复杂度豁免：v2 P4 T5 阶段专用验收共享件，不进入生产链路。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

D4_STAGES: Tuple[str, ...] = ("snapshot", "raw", "clean", "cs_infer", "panel", "verify")
PREREQUISITE_STAGE = "snapshot"
PREREQUISITE_STEP_KEY = "snapshot_prerequisite"
D4_STAGE_TO_SLA_STEP: Dict[str, str] = {
    "raw": "ensure_raw",
    "clean": "ensure_clean",
    "cs_infer": "cs_infer",
    "panel": "panel_label_refresh",
    "verify": "readiness",
}
SLA_ARM_STEP = "arms"
SLA_REQUIRED_STEPS: Tuple[str, ...] = (
    "ensure_raw",
    "ensure_clean",
    "cs_infer",
    "panel_label_refresh",
    "readiness",
    "arms",
)
REQUIRED_RECORD_KEYS: Tuple[str, ...] = (
    "trade_date",
    "started_at",
    "completed_at",
    "steps",
    "gaps",
    "run_completed",
    "actual_calendar_day",
)
#: 窗口/累计台账来源标记（写入时不出现，旧格式读取时补）
SOURCE_FLAT = "flat_record"
SOURCE_LEGACY_WRAPPED = "legacy_single_list"


class EvidenceFormatError(ValueError):
    """台账形态非法（嵌套列表、缺字段、日期非法）。"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_dump(path: Path, payload: Any) -> None:
    """原子写 JSON（同目录 tmp + replace）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 协调者产物 → 真实阶段指标
# ---------------------------------------------------------------------------
def coordinator_evidence_dir(data_root: Path, run_id: str) -> Path:
    """D4 协调者的证据目录（与其 ``_evidence_dir`` 同源口径）。"""
    return Path(data_root) / "frozen_reference" / "v2_p4" / run_id


def _artifact(data_root: Path, run_id: str, name: str) -> Optional[Mapping[str, Any]]:
    path = coordinator_evidence_dir(data_root, run_id) / name
    if not path.is_file():
        return None
    payload = load_json(path)
    return payload if isinstance(payload, Mapping) else None


def _flatten(days: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for value in (days or {}).values():
        if isinstance(value, (list, tuple)):
            out.extend(str(item) for item in value)
        elif value not in (None, ""):
            out.append(str(value))
    return sorted(set(out))


def _unobserved(stage: str, reason: str) -> Dict[str, Any]:
    return {
        "observed": False,
        "retries": 0,
        "gaps": [],
        "degraded": True,
        "reasons": [f"{stage}: {reason}"],
    }


def _observed(
    retries: int, gaps: Sequence[str], degraded: bool, reasons: Sequence[str]
) -> Dict[str, Any]:
    return {
        "observed": True,
        "retries": int(max(retries, 0)),
        "gaps": sorted(set(str(day) for day in gaps)),
        "degraded": bool(degraded),
        "reasons": [str(reason) for reason in reasons],
    }


def _raw_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    codes = [int(code) for code in (payload.get("exit_codes") or [])]
    unexpected = _flatten(payload.get("unexpected_gaps") or {})
    explained = _flatten(payload.get("source_side_gaps") or {})
    retries = sum(1 for code in codes if code != 0)
    reasons = [f"raw exit_codes={codes}"] if codes else ["raw 未记录 exit_codes"]
    if unexpected:
        reasons.append(f"raw 非源侧缺口={unexpected}")
    if explained:
        reasons.append(f"raw 登记源侧缺口={explained}")
    return _observed(retries, unexpected, bool(unexpected) or any(c != 0 for c in codes), reasons)


def _clean_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    gaps: List[str] = []
    reasons: List[str] = []
    for name, entry in (payload.get("datasets") or {}).items():
        absent = [str(day) for day in (entry.get("absent") or [])]
        duplicated = [str(day) for day in (entry.get("duplicate_key_days") or [])]
        gaps.extend(absent + duplicated)
        if absent or duplicated:
            reasons.append(f"clean.{name} absent={absent} duplicate={duplicated}")
    return _observed(len(reasons), gaps, bool(gaps), reasons or ["clean 缺口闭合"])


def _cs_infer_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    failures = [dict(item) for item in (payload.get("failures") or [])]
    gaps = [str(item.get("trade_date")) for item in failures if item.get("trade_date")]
    reasons = [f"cs_infer 失败={item}" for item in failures]
    if not failures:
        reasons.append(f"cs_infer built={payload.get('built')} skipped={payload.get('skipped')}")
    return _observed(len(failures), gaps, bool(failures), reasons)


def _panel_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    exit_code = int(payload.get("exit_code") or 0)
    not_ready = [str(day) for day in (payload.get("readiness_not_ready") or [])]
    reasons: List[str] = []
    if exit_code != 0:
        reasons.append(f"panel exit_code={exit_code}")
    if not_ready:
        reasons.append(f"panel 标签刷新未就绪={not_ready}")
    return _observed(
        0 if exit_code == 0 else 1, not_ready, bool(reasons), reasons or ["panel 完成"]
    )


def _verify_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    verdict = str(payload.get("verdict") or "")
    readiness = payload.get("readiness_exit_code")
    degraded = verdict != "PASS" or (readiness is not None and int(readiness) != 0)
    reasons = [f"verify verdict={verdict or 'missing'} readiness_exit_code={readiness}"]
    return _observed(0, [], degraded, reasons)


_STAGE_EXTRACTORS = {
    "raw": ("raw_after.json", _raw_metrics),
    "clean": ("clean_after.json", _clean_metrics),
    "cs_infer": ("cs_infer_after.json", _cs_infer_metrics),
    "panel": ("panel_after.json", _panel_metrics),
    "verify": ("verification_summary.json", _verify_metrics),
}


def extract_stage_metrics(data_root: Path, run_id: str, stage: str) -> Dict[str, Any]:
    """从协调者真实产物提取某阶段的重试/缺口/降级；缺产物即降级（fail-closed）。"""
    if stage == PREREQUISITE_STAGE:
        return _observed(0, [], False, ["snapshot 为前置身份快照，不作数据构建"])
    entry = _STAGE_EXTRACTORS.get(stage)
    if entry is None:
        raise ValueError(f"未知 D4 阶段，无法映射 SLA 字段: {stage}")
    name, extractor = entry
    payload = _artifact(data_root, run_id, name)
    if payload is None:
        return _unobserved(stage, f"缺协调者产物 {name}")
    return extractor(payload)


# ---------------------------------------------------------------------------
# 记录构造与校验
# ---------------------------------------------------------------------------
def build_steps_record(
    d4_stages: Mapping[str, Mapping[str, Any]],
    arms_step: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """D4 原始阶段 → SLA 步骤字段；保留原始耗时/重试与降级来源。"""
    steps: Dict[str, Dict[str, Any]] = {}
    prerequisite = d4_stages.get(PREREQUISITE_STAGE)
    if prerequisite is not None:
        steps[PREREQUISITE_STEP_KEY] = {
            "elapsed_seconds": float(prerequisite.get("elapsed_seconds") or 0.0),
            "retries": int(prerequisite.get("retries") or 0),
            "gaps": list(prerequisite.get("gaps") or []),
            "degraded": bool(prerequisite.get("degraded", False)),
            "reasons": list(prerequisite.get("reasons") or []),
            "origin_stage": PREREQUISITE_STAGE,
        }
    for stage, sla_name in D4_STAGE_TO_SLA_STEP.items():
        raw = d4_stages.get(stage)
        if raw is None:
            continue
        steps[sla_name] = {
            "elapsed_seconds": float(raw.get("elapsed_seconds") or 0.0),
            "retries": int(raw.get("retries") or 0),
            "gaps": list(raw.get("gaps") or []),
            "degraded": bool(raw.get("degraded", True)),
            "observed": bool(raw.get("observed", False)),
            "reasons": list(raw.get("reasons") or []),
            "origin_stage": stage,
        }
    if arms_step is not None:
        steps[SLA_ARM_STEP] = {
            "elapsed_seconds": float(arms_step.get("elapsed_seconds") or 0.0),
            "retries": int(arms_step.get("retries") or 0),
            "gaps": list(arms_step.get("gaps") or []),
            "degraded": bool(arms_step.get("degraded", False)),
            "reasons": list(arms_step.get("reasons") or []),
            "origin_stage": SLA_ARM_STEP,
        }
    return steps


def aggregate_day_metrics(steps: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """汇总当日缺口/降级/失败步骤；未知字段一律按降级处理。"""
    gaps: List[str] = []
    reasons: List[str] = []
    degraded = False
    for name, step in steps.items():
        if not isinstance(step, Mapping):
            degraded = True
            reasons.append(f"{name}: 步骤不是对象")
            continue
        gaps.extend(str(day) for day in (step.get("gaps") or []))
        if step.get("degraded", True):
            degraded = True
            reasons.extend(f"{name}: {reason}" for reason in (step.get("reasons") or ["降级"]))
    return {
        "gaps": sorted(set(gaps)),
        "ensure_degraded": degraded,
        "degraded_reasons": reasons,
    }


def missing_required_steps(steps: Mapping[str, Any]) -> List[str]:
    return [name for name in SLA_REQUIRED_STEPS if not isinstance(steps.get(name), Mapping)]


def build_day_record(
    *,
    day: str,
    started_at: str,
    completed_at: str,
    d4_stages: Mapping[str, Mapping[str, Any]],
    arms_step: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """构造单日记录：步骤按 SLA 字段命名，原始 D4 阶段原文保留在 ``d4_stages``。"""
    steps = build_steps_record(d4_stages, arms_step)
    metrics = aggregate_day_metrics(steps)
    record: Dict[str, Any] = {
        "trade_date": str(day),
        "started_at": started_at,
        "completed_at": completed_at,
        "steps": steps,
        "d4_stages": {
            stage: {
                "elapsed_seconds": float(raw.get("elapsed_seconds") or 0.0),
                "retries": int(raw.get("retries") or 0),
                "gaps": list(raw.get("gaps") or []),
                "degraded": bool(raw.get("degraded", True)),
                "observed": bool(raw.get("observed", False)),
                "reasons": list(raw.get("reasons") or []),
            }
            for stage, raw in d4_stages.items()
        },
        "gaps": metrics["gaps"],
        "ensure_degraded": metrics["ensure_degraded"],
        "degraded_reasons": metrics["degraded_reasons"],
        "missing_steps": missing_required_steps(steps),
        "run_completed": False,
        "account_advanced": False,
        "actual_calendar_day": False,
        "system_reason": None,
        "non_system_reason": None,
        "failed_step": None,
        "error": None,
        "drill": False,
        "recorded_at": _now(),
    }
    if extra:
        record.update(dict(extra))
    return record


def validate_day_record(record: Mapping[str, Any]) -> None:
    """写前校验：日期、必需字段、步骤形态、单条记录语义。"""
    if not isinstance(record, Mapping):
        raise EvidenceFormatError("日记录必须是对象")
    missing = [key for key in REQUIRED_RECORD_KEYS if key not in record]
    if missing:
        raise EvidenceFormatError(f"日记录缺必需字段: {missing}")
    day = str(record["trade_date"])
    if len(day) != 8 or not day.isdigit():
        raise EvidenceFormatError(f"日记录 trade_date 非法（须 YYYYMMDD）: {day}")
    if not isinstance(record["steps"], Mapping):
        raise EvidenceFormatError("日记录 steps 必须是对象")
    if not isinstance(record["gaps"], list):
        raise EvidenceFormatError("日记录 gaps 必须是列表（无缺口=[]）")
    for key in ("started_at", "completed_at"):
        try:
            datetime.fromisoformat(str(record[key]))
        except ValueError as exc:
            raise EvidenceFormatError(f"日记录 {key} 非法: {exc}") from exc


def _normalize_payload(payload: Any, path: Path) -> Tuple[Dict[str, Any], str]:
    """单条记录归一：dict 原样；旧 ``[record]`` 只做单层转换并留下来源。"""
    if isinstance(payload, Mapping):
        return dict(payload), SOURCE_FLAT
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], Mapping):
            converted = dict(payload[0])
            converted.setdefault("_source_format", SOURCE_LEGACY_WRAPPED)
            converted["_source_path"] = str(path)
            return converted, SOURCE_LEGACY_WRAPPED
        raise EvidenceFormatError(
            f"单日台账不得为多条记录或嵌套结构: {path}（条数={len(payload)}）"
        )
    raise EvidenceFormatError(f"单日台账形态非法（须对象）: {path}")


def write_day_record(path: Path, record: Mapping[str, Any]) -> None:
    """写单日文件：单个 ``record`` 对象，写前校验。"""
    validate_day_record(record)
    atomic_dump(path, dict(record))


def write_cumulative_ledger(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """写累计台账：``list[record]``，按日期升序，重复日 fail-fast。"""
    normalized = [dict(item) for item in records]
    for item in normalized:
        validate_day_record(item)
    days = [str(item["trade_date"]) for item in normalized]
    if len(days) != len(set(days)):
        raise EvidenceFormatError("累计台账存在重复交易日，拒绝写入")
    atomic_dump(path, sorted(normalized, key=lambda item: str(item["trade_date"])))


def read_day_record(path: Path) -> Dict[str, Any]:
    """读单条记录；旧格式单层转换并留下来源标记。"""
    path = Path(path)
    record, _source = _normalize_payload(load_json(path), path)
    return record


def load_cumulative_ledger(path: Path) -> List[Dict[str, Any]]:
    """读累计台账（``list[record]`` 或 ``{records:[...]}``）。"""
    path = Path(path)
    if not path.is_file():
        return []
    payload = load_json(path)
    records = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(records, list):
        raise EvidenceFormatError(f"累计台账必须是 list 或含 records list 的对象: {path}")
    out: List[Dict[str, Any]] = []
    for index, item in enumerate(records):
        if isinstance(item, list):
            raise EvidenceFormatError(
                f"累计台账不得嵌套列表（records[{index}]）: {path}；请用单日文件入口"
            )
        if not isinstance(item, Mapping):
            raise EvidenceFormatError(f"累计台账 records[{index}] 不是对象: {path}")
        out.append(dict(item))
    return sorted(out, key=lambda item: str(item.get("trade_date") or ""))


def append_cumulative_record(path: Path, record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """累计台账只追加：同日重复 fail-fast，不隐式取最后 N 条。"""
    records = load_cumulative_ledger(path)
    day = str(record.get("trade_date") or "")
    if any(str(item.get("trade_date")) == day for item in records):
        raise EvidenceFormatError(f"累计台账已含 {day}，拒绝重复累计: {path}")
    records.append(dict(record))
    write_cumulative_ledger(path, records)
    return records


def load_window_records(directory: Path) -> List[Dict[str, Any]]:
    """窗口加载器：逐日文件 → 平坦 ``list[record]``（拒绝嵌套、重复日）。"""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    records: List[Dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        record, source = _normalize_payload(load_json(path), path)
        record.setdefault("_source_path", str(path))
        record["_source_format"] = record.get("_source_format", source)
        records.append(record)
    days = [str(item.get("trade_date") or "") for item in records]
    duplicated = sorted({day for day in days if days.count(day) > 1})
    if duplicated:
        raise EvidenceFormatError(f"窗口台账存在重复交易日: {duplicated}")
    return sorted(records, key=lambda item: str(item.get("trade_date") or ""))


# ---------------------------------------------------------------------------
# 首跑前逐源缺口扫描（只读）
# ---------------------------------------------------------------------------
#: 逐源扫描清单：层 / 数据集 / 分区载体（file=日分区文件，dir=日目录）
SCAN_SOURCES: Tuple[Tuple[str, str, str], ...] = (
    ("raw", "daily", "file"),
    ("raw", "daily_basic", "file"),
    ("raw", "adj_factor", "file"),
    ("raw", "suspend", "file"),
    ("raw", "stk_limit", "file"),
    ("clean", "daily", "file"),
    ("clean", "daily_basic", "file"),
    ("features", "cs_infer", "file"),
    ("features", "panel", "dir"),
)


def partition_days(root: Path, kind: str) -> List[str]:
    """枚举分区日（文件名 ``YYYY-MM-DD.parquet`` / ``YYYYMMDD.parquet`` 或日目录）。"""
    root = Path(root)
    if not root.is_dir():
        return []
    days: List[str] = []
    for path in root.iterdir():
        name = path.stem if kind == "file" else path.name
        if kind == "file" and path.suffix != ".parquet":
            continue
        stem = name.replace("-", "")
        if len(stem) == 8 and stem.isdigit():
            days.append(stem)
    return sorted(set(days))


def scan_data_gaps(
    data_root: Path, start_date: str, target_date: str, calendar: Sequence[str]
) -> Dict[str, Any]:
    """逐源扫描所需交易日的缺口；不只看最新分区，也不写任何数据。"""
    if target_date not in calendar:
        raise ValueError(f"目标日 {target_date} 不在交易日历，禁止校正后扫描")
    required = [day for day in calendar if start_date <= day <= target_date]
    sources: Dict[str, Any] = {}
    first_missing: Optional[str] = None
    for layer, dataset, kind in SCAN_SOURCES:
        present = set(partition_days(Path(data_root) / layer / dataset, kind))
        missing = [day for day in required if day not in present]
        entry = {
            "layer": layer,
            "dataset": dataset,
            "carrier": kind,
            "required_days": len(required),
            "present_days": sum(1 for day in required if day in present),
            "missing_days": missing,
            "missing_count": len(missing),
            "first_missing": missing[0] if missing else None,
            "last_present": max((day for day in required if day in present), default=None),
        }
        sources[f"{layer}/{dataset}"] = entry
        if missing:
            first_missing = missing[0] if first_missing is None else min(first_missing, missing[0])
    return {
        "scanned_at": _now(),
        "range": {"start": start_date, "target": target_date, "trading_days": len(required)},
        "target_in_calendar": True,
        "sources": sources,
        "first_missing_day": first_missing,
        "all_clear": not any(entry["missing_count"] for entry in sources.values()),
    }


# ---------------------------------------------------------------------------
# 机器时间预算（窗口级授权）
# ---------------------------------------------------------------------------
class MachineBudget:
    """窗口级日更机器时间台账：单日上限、总上限、已耗与剩余。"""

    def __init__(self, path: Path, *, total_minutes: float, daily_cap_minutes: float) -> None:
        if float(total_minutes) <= 0:
            raise ValueError("窗口级机器授权总额必须 > 0（登记总上限/剩余才有意义）")
        self.path = Path(path)
        self.total_minutes = float(total_minutes)
        self.daily_cap_minutes = float(daily_cap_minutes)
        payload = self.load()
        stored = payload.get("authorized_minutes_total")
        if stored is not None and float(stored) != self.total_minutes:
            raise RuntimeError(
                f"窗口级机器授权总额与已登记台账不一致: 台账={stored} 本次={self.total_minutes}"
                "（授权额度变更须新证据根，不得就地改写）"
            )
        payload.update(
            {
                "authorized_minutes_total": self.total_minutes,
                "single_day_cap_minutes": self.daily_cap_minutes,
            }
        )
        self.payload = payload

    def load(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return {"days": {}, "consumed_minutes": 0.0}
        payload = load_json(self.path)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("days"), Mapping):
            raise EvidenceFormatError(f"机器预算台账形态非法: {self.path}")
        return dict(payload)

    @property
    def consumed_minutes(self) -> float:
        return float(self.payload.get("consumed_minutes") or 0.0)

    @property
    def remaining_minutes(self) -> float:
        return self.total_minutes - self.consumed_minutes

    def check(self, day: str, *, allow_reentry: bool = False) -> Dict[str, Any]:
        """执行前检查总上限与同日重入（恢复日允许同日补齐证据，不增加有效日）。"""
        if not allow_reentry and str(day) in (self.payload.get("days") or {}):
            raise RuntimeError(f"机器预算已登记 {day}，同一交易日不得重复驱动")
        if self.remaining_minutes <= 0:
            raise RuntimeError(
                f"机器授权额度已用尽：总 {self.total_minutes} 分钟，"
                f"已耗 {self.consumed_minutes:.2f}，剩余 {self.remaining_minutes:.2f}"
            )
        return {
            "total_minutes": self.total_minutes,
            "single_day_cap_minutes": self.daily_cap_minutes,
            "consumed_minutes_before": self.consumed_minutes,
            "remaining_minutes_before": self.remaining_minutes,
            "same_day_reentry": str(day) in (self.payload.get("days") or {}),
        }

    def finish(self, day: str, elapsed_seconds: float) -> Dict[str, Any]:
        """执行后登记实耗（同日多次累加）；超限只登记超限，不伪造通过。"""
        minutes = round(float(elapsed_seconds) / 60.0, 4)
        days = dict(self.payload.get("days") or {})
        previous = float((days.get(str(day)) or {}).get("elapsed_minutes") or 0.0)
        day_total = round(previous + minutes, 4)
        days[str(day)] = {
            "elapsed_minutes": day_total,
            "last_run_minutes": minutes,
            "over_single_day_cap": day_total > self.daily_cap_minutes,
            "recorded_at": _now(),
        }
        consumed = round(
            sum(float(item.get("elapsed_minutes") or 0.0) for item in days.values()), 4
        )
        self.payload.update(
            {
                "days": days,
                "consumed_minutes": consumed,
                "remaining_minutes": round(self.total_minutes - consumed, 4),
                "over_total_budget": consumed > self.total_minutes,
                "updated_at": _now(),
            }
        )
        atomic_dump(self.path, self.payload)
        return dict(self.payload["days"][str(day)])
