# -*- coding: utf-8 -*-
"""v2 P4 T5 差异/实耗/containment/收口总账（阶段专用）。

本工具只汇总**本次实测证据**，不预填通过数、不把合成测试或历史回放升级为门 5/6/7
PASS。缺失证据一律显示 MISSING/未完成并计入阻塞原因；已完成项不再复述历史的
「未获授权」。

门 5 与门 6 使用同一当前有效窗口日期集合；历史重置前的失败记录保留在
``system_reset_events``，删文件才能通过的做法不被承认。

# 复杂度豁免：v2 P4 T5 阶段专用验收工具，不进入生产链路。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p4.t1_worktree_digest import compute_worktree_digest  # noqa: E402
from scripts.v2_p4.t4_gate_runner import capture_data_state  # noqa: E402
from scripts.v2_p4.t5_migration import build_file_inventory  # noqa: E402
from src.lazybull.paper import PaperStorage  # noqa: E402
from src.lazybull.v2.evidence.runs_loader import load_runs_batch  # noqa: E402

GATE_NAMES = ("gate1", "gate5", "gate6", "gate7", "gate8")
SOURCE_KEYS = ("raw_daily", "clean_daily", "cs_train", "cs_infer", "panel")


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


# ---------------------------------------------------------------------------
# 各证据源读取
# ---------------------------------------------------------------------------
def _test_report(path: Optional[Path], *, label: str) -> Dict[str, Any]:
    """测试报告：缺证据=MISSING，失败数>0=FAIL，不预填通过数。"""
    if path is None:
        return {"status": "MISSING", "label": label, "reason": "未提供本次测试报告"}
    path = Path(path)
    if not path.is_file():
        return {"status": "MISSING", "label": label, "path": str(path), "reason": "报告文件不存在"}
    payload = _load(path, {})
    if not isinstance(payload, Mapping):
        return {"status": "INVALID", "label": label, "path": str(path), "reason": "报告不是对象"}
    failed = int(payload.get("failed") or 0)
    count = int(payload.get("count") or payload.get("passed") or 0)
    verdict = str(payload.get("verdict") or "")
    # 失败数/有效执行数为否决条件，不能被显式 verdict=PASS 覆盖。
    if failed > 0:
        status = "FAIL"
    elif count <= 0:
        status = "INVALID"
    elif verdict:
        status = verdict.upper()
    else:
        status = "UNKNOWN"
    return {
        "status": status,
        "label": label,
        "path": str(path),
        "count": count,
        "failed": failed,
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "run_id": payload.get("run_id"),
        "recorded_at": payload.get("recorded_at"),
    }


def _complexity_report(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"status": "MISSING", "reason": "未提供本次复杂度报告"}
    path = Path(path)
    if not path.is_file():
        return {"status": "MISSING", "path": str(path), "reason": "报告文件不存在"}
    payload = _load(path, {})
    entries = payload.get("paths") if isinstance(payload, Mapping) else None
    if not isinstance(entries, Mapping):
        return {"status": "INVALID", "path": str(path), "reason": "缺 paths 结果"}
    required = {"src/lazybull", "src/lazybull+scripts"}
    all_ok = required.issubset(entries.keys()) and all(
        int((entries.get(key) or {}).get("exit", 1)) == 0
        and int((entries.get(key) or {}).get("new_hard", 1)) == 0
        for key in required
    )
    return {
        "status": "PASS" if all_ok else "FAIL",
        "path": str(path),
        "paths": entries,
        "reason": None if all_ok else "复杂度必须同时提供 src 与 src+scripts 两口径且均 exit=0/new_hard=0",
    }


def _runs_batches(
    *,
    run_id: Optional[str],
    runs_root: Path,
    explicit: Optional[Sequence[Path]],
    expected_days: Optional[Sequence[str]] = None,
    expected_trades: Optional[Mapping[tuple, int]] = None,
    expected_daily: Optional[Mapping[str, tuple]] = None,
) -> List[Dict[str, Any]]:
    """runs 批次必须经既有 loader 真实读回；只填批次名不算证据。"""
    candidates = [Path(item) for item in (explicit or [])]
    if not candidates and run_id:
        candidates = sorted(runs_root.glob(f"*{run_id}*paper_shadow*"))
    results: List[Dict[str, Any]] = []
    for directory in candidates:
        if not directory.is_dir():
            results.append({"path": str(directory), "loaded": False, "error": "批次目录不存在"})
            continue
        try:
            load_runs_batch(directory)
        except Exception as exc:  # noqa: BLE001 - loader 拒绝即登记，不当作已读回
            results.append(
                {"path": str(directory), "loaded": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        meta = _load(directory / "batch_meta.json", {}) or {}
        notes = meta.get("meta_notes") or meta.get("notes") or {}
        batch_id = meta.get("batch_id") or directory.name
        problems = []
        if run_id and run_id not in batch_id:
            problems.append(f"batch_id 不含当前 run_id={run_id}")
        if notes.get("shadow_window") is None:
            problems.append("缺 shadow_window")
        try:
            if int(notes.get("shadow_days") or 0) < 10:
                problems.append("shadow_days<10")
        except (TypeError, ValueError):
            problems.append("shadow_days 非法")
        if notes.get("evidence_mode") not in (None, "SHADOW_SIM"):
            problems.append(f"evidence_mode={notes.get('evidence_mode')!r}")
        if notes.get("sla_mode") not in (None, "SLA_SIM"):
            problems.append(f"sla_mode={notes.get('sla_mode')!r}")
        trades_path = directory / "folds" / "split00" / "trades.parquet"
        if expected_trades is not None:
            if not trades_path.is_file():
                problems.append("缺 folds/split00/trades.parquet")
            else:
                actual_frame = pd.read_parquet(trades_path)
                actual = {}
                for row in actual_frame.to_dict("records"):
                    date = str(row.get("trade_date") or row.get("date") or "").replace("-", "")[:8]
                    stock = str(row.get("ts_code") or row.get("stock") or "")
                    action = str(row.get("action") or "")
                    actual[(date, stock, action)] = int(row.get("shares") or 0)
                if actual != dict(expected_trades):
                    problems.append("批次成交与本次累积器不一致")
                if expected_days:
                    bad_days = sorted(
                        {key[0] for key in actual} - {str(day) for day in expected_days}
                    )
                    if bad_days:
                        problems.append(f"批次含窗口外日期={bad_days}")
        daily_path = directory / "folds" / "split00" / "daily.parquet"
        if expected_daily is not None:
            if not daily_path.is_file():
                problems.append("缺 folds/split00/daily.parquet")
            else:
                daily_frame = pd.read_parquet(daily_path)
                actual_days = {
                    str(value).replace("-", "")[:8]
                    for value in daily_frame.get("trade_date", [])
                }
                if actual_days != {str(day) for day in expected_daily}:
                    problems.append("批次 daily 日期集与本次窗口不一致")
                for row in daily_frame.to_dict("records"):
                    date = str(row.get("trade_date") or "").replace("-", "")[:8]
                    expected = expected_daily.get(date)
                    if expected is None:
                        continue
                    values = (
                        float(row.get("total_value") or 0.0),
                        float(row.get("cash") or 0.0),
                        float(row.get("market_value") or 0.0),
                    )
                    if any(abs(a - b) > 1e-6 for a, b in zip(values, expected)):
                        problems.append(f"批次 daily 数值与累积器不一致={date}")
                        break
        results.append(
            {
                "path": str(directory),
                "batch_id": batch_id,
                "loaded": not problems,
                "error": "; ".join(problems) or None,
                "host_mode": meta.get("host_mode"),
                "shadow_days": notes.get("shadow_days"),
                "shadow_window": notes.get("shadow_window"),
            }
        )
    return results


def _valid_window_dates(days: Sequence[str]) -> bool:
    """窗口必须是唯一、连续的真实交易日；重复/非交易日/缺口一律拒绝。"""
    values = [str(day) for day in days]
    if not values or len(set(values)) != len(values):
        return False
    if any(len(day) != 8 or not day.isdigit() for day in values):
        return False
    try:
        calendar = pd.read_parquet(ROOT / "data" / "clean" / "trade_cal.parquet")
        opens = sorted(
            calendar.loc[calendar["is_open"] == 1, "cal_date"]
            .astype(str)
            .str.replace("-", "", regex=False)
        )
    except Exception:  # noqa: BLE001 - 无日历证据不能升级为 PASS
        return False
    index = {day: idx for idx, day in enumerate(opens)}
    if any(day not in index for day in values):
        return False
    return all(index[b] == index[a] + 1 for a, b in zip(values, values[1:]))


def _gate5_window(evidence: Path, window: Mapping[str, Any]) -> Dict[str, Any]:
    """门 5 与门 6 同窗口：优先用 SLA 当前有效段日期集合对齐逐日比较。"""
    files = {
        path.stem.replace("daily_", ""): path
        for path in sorted((evidence / "gate5").glob("daily_*.json"))
    }
    loaded = {day: (_load(path, {}) or {}) for day, path in files.items()}
    audit = _load(evidence / "gate5" / "reviewer2_direction_audit.json", {}) or {}
    if audit.get("source_gate5_pass_invalidated") is True:
        return {
            "window_dates": list(window.get("current_streak_dates") or []),
            "window_dates_source": "sla_window",
            "covered_days": 0,
            "missing_days": [],
            "failed_days": list(audit.get("affected_days") or []),
            "days_outside_window": [],
            "verdict": "NOT_STARTED_OR_INCOMPLETE",
            "review_blocker": "R2-P4-T5-01: 旧门5方向审计判定 PASS 失效，待用户裁决/新 run 重取",
        }
    window_dates = [str(day) for day in (window.get("current_streak_dates") or [])]
    if window_dates:
        if not _valid_window_dates(window_dates):
            return {
                "window_dates": window_dates,
                "window_dates_source": "sla_window",
                "covered_days": 0,
                "missing_days": [],
                "failed_days": [],
                "days_outside_window": [],
                "verdict": "NOT_STARTED_OR_INCOMPLETE",
                "reason": "窗口必须为唯一、连续的真实交易日集合（重复/非交易日/缺中间日拒绝）",
            }
        missing = [day for day in window_dates if day not in loaded]
        outside = sorted(set(loaded) - set(window_dates))
        failing = [day for day in window_dates if (loaded.get(day) or {}).get("verdict") != "PASS"]
        ok = (
            not missing
            and not failing
            and window.get("verdict") == "PASS"
            and len(window_dates) >= 10
        )
        return {
            "window_dates": window_dates,
            "window_dates_source": "sla_window",
            "covered_days": len(window_dates) - len(missing),
            "missing_days": missing,
            "failed_days": failing,
            "days_outside_window": outside,
            "verdict": "PASS" if ok else "NOT_STARTED_OR_INCOMPLETE",
        }
    passing = sorted(day for day, payload in loaded.items() if payload.get("verdict") == "PASS")
    ok = window.get("verdict") == "PASS" and len(passing) >= 10
    return {
        "window_dates": passing,
        "window_dates_source": "gate5_files_legacy_window",
        "covered_days": len(passing),
        "missing_days": [] if ok else ["SLA 窗口缺当前有效段日期集合"],
        "failed_days": sorted(set(loaded) - set(passing)),
        "days_outside_window": [],
        "verdict": "PASS" if ok else "NOT_STARTED_OR_INCOMPLETE",
    }


def _gate7_state(evidence: Path) -> Dict[str, Any]:
    real = _load(evidence / "gate7" / "real_rehearsal_result.json")
    legacy = _load(evidence / "gate7" / "rehearsal_result.json")
    if isinstance(real, Mapping):
        incomplete = list(real.get("items_incomplete") or [])
        ok = (
            real.get("verdict") == "PASS" and real.get("synthetic_only") is False and not incomplete
        )
        return {
            "source": "real_rehearsal_result",
            "verdict": real.get("verdict"),
            "synthetic_only": real.get("synthetic_only"),
            "items_incomplete": incomplete,
            "ok": ok,
        }
    if isinstance(legacy, Mapping):
        ok = (
            legacy.get("verdict") == "PASS"
            and legacy.get("synthetic_only") is False
            and legacy.get("formal_gate7_pass") is not False
        )
        return {
            "source": "rehearsal_result",
            "verdict": legacy.get("verdict"),
            "synthetic_only": legacy.get("synthetic_only"),
            "items_incomplete": list(legacy.get("items_incomplete") or []),
            "ok": ok,
        }
    return {
        "source": "missing",
        "verdict": "MISSING",
        "synthetic_only": None,
        "items_incomplete": [],
        "ok": False,
    }


def _source_max(source: Any) -> Any:
    """数据源水位归一：门 1 既有 ``str`` 形态，也有 D4 的 ``{"max": ...}`` 形态。"""
    if isinstance(source, Mapping):
        return source.get("max")
    return source if isinstance(source, str) else None


def _data_state_view(gate1: Mapping[str, Any], current: Mapping[str, Any]) -> Dict[str, Any]:
    """区分冻结起点、每日 D4 合法更新、决策段共同快照与冻结段不变。"""
    frozen = gate1.get("data_state") or {}
    frozen_sources = frozen.get("sources") or {}
    current_sources = current.get("sources") or {}
    frozen_max = {name: _source_max(frozen_sources.get(name)) for name in SOURCE_KEYS}
    current_max = {
        name: _source_max(current_sources.get(name)) or current.get(f"{name}_latest")
        for name in SOURCE_KEYS
    }
    advanced = {
        name: {"frozen": frozen_max[name], "current": current_max[name]}
        for name in SOURCE_KEYS
        if frozen_max[name] != current_max[name]
    }
    return {
        "frozen_start_data_state_id": frozen.get("data_state_id"),
        "current_data_state_id": current.get("data_state_id"),
        "daily_legit_updates": advanced,
        "cs_train_frozen_unchanged": frozen_max["cs_train"] == current_max["cs_train"],
        "frozen_labels_unchanged": (frozen.get("labels") or {}) == (current.get("labels") or {}),
    }


def _decision_containment(evidence: Path) -> Dict[str, Any]:
    files = sorted((evidence / "containment").glob("*.json"))
    violations: List[str] = []
    checked = 0
    for path in files:
        payload = _load(path, {}) or {}
        if "data_state_decision_phase_unchanged" not in payload:
            continue
        checked += 1
        ok = bool(payload.get("data_state_decision_phase_unchanged")) and not payload.get(
            "shadow_arms_wrote_production", False
        )
        if not ok:
            violations.append(str(payload.get("date") or path.stem))
    return {"checked_days": checked, "violation_days": violations}


def _production_account(evidence: Path) -> Dict[str, Any]:
    files = (
        sorted((evidence / "sla").glob("daily/*.json"))
        if (evidence / "sla" / "daily").is_dir()
        else []
    )
    advanced = 0
    for path in files:
        payload = _load(path, {}) or {}
        if payload.get("account_advanced"):
            advanced += 1
    return {"real_shadow_days": len(files), "account_advanced_days": advanced}


def _authorization(evidence: Path) -> Dict[str, Any]:
    payload = _load(evidence / "authorization.json")
    if not isinstance(payload, Mapping):
        return {"user_switch_authorized": False, "reason": "未提供用户切换授权证据"}
    return dict(payload)


# ---------------------------------------------------------------------------
# 收口主流程
# ---------------------------------------------------------------------------
def closeout(
    evidence: Path,
    *,
    runs_root: Optional[Path] = None,
    runs_batches: Optional[Sequence[Path]] = None,
    focused_tests_report: Optional[Path] = None,
    full_tests_report: Optional[Path] = None,
    complexity_report: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    evidence = Path(evidence)
    gate1 = _load(evidence / "verification_summary.json")
    if gate1 is None:
        raise FileNotFoundError(f"缺门 1 证据: {evidence / 'verification_summary.json'}")
    window = _load(evidence / "sla" / "window.json", {}) or {}
    gate5 = _gate5_window(evidence, window)
    gate7 = _gate7_state(evidence)
    inventory = build_file_inventory(ROOT / "data" / "paper")
    _dump(evidence / "gate7" / "migration_inventory.json", inventory)
    gate1_ok = gate1.get("verdict") == "PASS" and gate1.get("gate1_frozen") is True
    planned_arms = {
        "status": "BLOCKED_Q11A" if not gate1_ok else "READY_TO_CLONE",
        "same_first_shadow_day": True,
        "same_frozen_account_snapshot": True,
        "acceptance_edge": "v2_shadow vs legacy_neutral",
        "arms": {
            "legacy_active": {
                "engine": "legacy",
                "exposure": "unchanged",
                "informational_only": True,
            },
            "legacy_neutral": {"engine": "legacy", "exposure_policy": None, "isolated_root": True},
            "v2_shadow": {"engine": "v2", "host_mode": "paper_shadow", "isolated_root": True},
        },
    }
    _dump(evidence / "three_arm_plan.json", planned_arms)
    differences: List[Dict[str, Any]] = []
    for path in sorted((evidence / "gate5").glob("daily_*.json")):
        daily = _load(path, {}) or {}
        for row in daily.get("differences", []):
            differences.append(
                {"date": daily.get("date") or path.stem.replace("daily_", ""), **row}
            )
    ledger = {
        "acceptance_edge": "v2_shadow vs legacy_neutral",
        "real_shadow_days": gate5["covered_days"],
        "unexplained": sum(1 for row in differences if row.get("category") == "unexplained"),
        "entries": differences,
        "note": "空台账只表示真实影子未开始/无记录，不等价门 5 PASS",
    }
    _dump(evidence / "difference_ledger.json", ledger)
    account_path = ROOT / "data" / "paper" / "state" / "account.json"
    current_data_state = capture_data_state(ROOT / "data")
    storage = PaperStorage()
    config = storage.load_config() or {}
    data_state = _data_state_view(gate1, current_data_state)
    decision_containment = _decision_containment(evidence)
    containment = {
        **data_state,
        "decision_phase_checks": decision_containment,
        "daily_legit_updates_are_not_drift": True,
        "production_engine": config.get("engine", "legacy"),
        "production_paper_writes_by_shadow": len(decision_containment["violation_days"]),
        "account_still_baseline": (
            None
            if gate1.get("gate1_mode") == "Q11_SIM"
            else (
                hashlib.sha256(account_path.read_bytes()).hexdigest()
                == (gate1.get("account") or {}).get("sha256")
                if account_path.is_file()
                else None
            )
        ),
        "account_baseline_reason": (
            "Q11_SIM 门1账户是隔离账户，不能与生产账户比较；生产账户未写由 containment 判定"
            if gate1.get("gate1_mode") == "Q11_SIM"
            else "对照门1冻结生产账户哈希"
        ),
        "production_account": _production_account(evidence),
    }
    _dump(evidence / "containment" / "summary.json", containment)
    worktree = compute_worktree_digest(ROOT)
    _dump(evidence / "worktree_summary.json", worktree)
    accumulator = _load(evidence / "shadow_accumulator.json", {}) or {}
    expected_days = sorted(
        {
            str(row.get("date") or "").replace("-", "")[:8]
            for row in (accumulator.get("portfolio_values") or [])
            if row.get("date")
        }
    )
    expected_trades = {}
    for row in accumulator.get("trades") or []:
        key = (
            str(row.get("date") or "").replace("-", "")[:8],
            str(row.get("stock") or row.get("ts_code") or ""),
            str(row.get("action") or ""),
        )
        expected_trades[key] = int(row.get("shares") or 0)
    expected_daily = {}
    for row in accumulator.get("portfolio_values") or []:
        date = str(row.get("date") or "").replace("-", "")[:8]
        expected_daily[date] = (
            float(row.get("portfolio_value") or 0.0),
            float(row.get("capital") or 0.0),
            float(row.get("market_value") or 0.0),
        )
    batches = _runs_batches(
        run_id=run_id or evidence.name,
        runs_root=Path(runs_root or (ROOT / "data" / "runs")),
        explicit=runs_batches,
        expected_days=expected_days,
        expected_trades=expected_trades if accumulator else None,
        expected_daily=expected_daily if accumulator else None,
    )
    loaded_batches = [item for item in batches if item.get("loaded")]
    focused = _test_report(focused_tests_report, label="focused_t5_tests")
    full = _test_report(full_tests_report, label="full_pytest")
    complexity = _complexity_report(complexity_report)
    gate8_checks = {
        "focused_tests": focused["status"],
        "full_pytest": full["status"],
        "complexity": complexity["status"],
        "runs_batches": "PASS" if loaded_batches else "MISSING",
    }
    gate8_ok = (
        all(value == "PASS" for value in gate8_checks.values())
        and decision_containment["checked_days"] > 0
        and not decision_containment["violation_days"]
        and containment["cs_train_frozen_unchanged"] is True
        and containment["frozen_labels_unchanged"] is True
    )
    gates = {
        "gate1": "PASS" if gate1_ok else str(gate1.get("verdict") or "BLOCKED"),
        "gate5": gate5["verdict"],
        "gate6": "PASS" if window.get("verdict") == "PASS" else "NOT_STARTED_OR_INCOMPLETE",
        "gate7": "PASS" if gate7["ok"] else "REAL_REHEARSAL_INCOMPLETE",
        "gate8": "PASS" if gate8_ok else "BLOCKED",
    }
    authorization = _authorization(evidence)
    switch_payload = _load(evidence / "switch_result.json")
    switch_completed = bool(
        isinstance(switch_payload, Mapping)
        and switch_payload.get("verdict") == "PASS"
        and containment["production_engine"] == "v2"
    )
    technical_ready = all(gates[name] == "PASS" for name in GATE_NAMES) and bool(loaded_batches)
    blocked_reasons: List[str] = []
    if not gate1_ok:
        blocked_reasons.append("Q11-A 连续 10 个真实交易日恢复尚未完成/门 1 未冻结")
    if gate5["verdict"] != "PASS":
        blocked_reasons.append(
            f"门 5 真实影子窗口未完成: {gate5.get('missing_days') or gate5.get('failed_days')}"
        )
    if gates["gate6"] != "PASS":
        blocked_reasons.append("门 6 真实 SLA 窗口未完成")
    if gates["gate7"] != "PASS":
        blocked_reasons.append(
            f"门 7 真实演练未完成（来源={gate7['source']}，未完成项={gate7['items_incomplete']}）"
        )
    if not loaded_batches:
        blocked_reasons.append("paper_shadow runs 批次缺失或未通过 loader 读回")
    if gates["gate8"] != "PASS":
        failed_checks = [
            f"{name}={status}" for name, status in gate8_checks.items() if status != "PASS"
        ]
        if decision_containment["checked_days"] <= 0:
            failed_checks.append("containment_checked_days=0")
        if decision_containment["violation_days"]:
            failed_checks.append(
                f"containment_violation_days={decision_containment['violation_days']}"
            )
        if containment["cs_train_frozen_unchanged"] is not True:
            failed_checks.append("cs_train_frozen_changed")
        if containment["frozen_labels_unchanged"] is not True:
            failed_checks.append("frozen_labels_changed")
        blocked_reasons.append("门 8 未通过: " + ", ".join(failed_checks))
    status = {
        "stage": "v2_p4_t5_closeout",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_id": evidence.name,
        "run_id": run_id or evidence.name,
        "verdict": "READY_FOR_FINAL_AUTHORIZATION" if technical_ready else "BLOCKED",
        "gates": gates,
        "gate5_window": gate5,
        "gate6_window": {
            "verdict": window.get("verdict"),
            "current_streak_days": window.get("current_streak_days"),
            "current_streak_dates": window.get("current_streak_dates"),
            "system_reset_events": window.get("system_reset_events"),
            "required": window.get("required"),
        },
        "gate7_source": gate7,
        "real_shadow_days": gate5["covered_days"],
        "q11a_observed_days": (gate1.get("q11a_restart") or {}).get("observed_days"),
        "q11a_selected_days": (gate1.get("q11a_restart") or {}).get("selected_days"),
        "statuses": {
            "technical_conditions_ready": technical_ready,
            "user_switch_authorized": bool(authorization.get("user_switch_authorized")),
            "production_switch_completed": switch_completed,
            "final_commit_completed": bool(
                (_load(evidence / "commit_status.json") or {}).get("committed", False)
            ),
        },
        "authorization": authorization,
        "production_switch_changed": switch_completed,
        "production_engine": containment["production_engine"],
        "runs_batches": batches,
        "actual_cost": {
            "real_shadow_days": gate5["covered_days"],
            "real_sla_window": {
                "verdict": window.get("verdict"),
                "current_streak_days": window.get("current_streak_days"),
            },
            "focused_tests": focused,
            "full_pytest": full,
            "machine_budget": _load(evidence / "machine_budget.json", {}),
            "source": "本次证据目录实测；缺证据标 MISSING，不预填通过",
        },
        "complexity": complexity,
        "gate8_checks": gate8_checks,
        "containment": containment,
        "worktree": worktree,
        "commit_status": (
            "COMMITTED"
            if (_load(evidence / "commit_status.json") or {}).get("committed")
            else "NOT_COMMITTED_T1_TO_T5_FINAL_SINGLE_COMMIT_PENDING_USER_INSTRUCTION"
        ),
        "recovery_commands": _recovery_commands(evidence, gate1),
        "blocked_reasons": blocked_reasons,
    }
    _dump(evidence / "closeout_status.json", status)
    return status


def _recovery_commands(evidence: Path, gate1: Mapping[str, Any]) -> List[str]:
    """可直接使用的恢复/影子/冻结命令（含恢复台账、持久 arms 根、现役绑定）。"""
    ledger = evidence / "q11a_restart_ledger.json"
    arms = ROOT / "data" / "paper_shadow" / evidence.name / "arms"
    gate1_path = evidence / "verification_summary.json"
    model_version = (gate1.get("model") or {}).get("version") or "<现役模型>"
    return [
        "python scripts/v2_p4/t5_daily_driver.py --mode preflight --start-date <扫描出的首缺日> "
        f"--target-date <首跑日> --evidence-dir {evidence} --execute",
        "python scripts/v2_p4/t5_daily_driver.py --mode restart --trade-date <真实交易日> "
        f"--run-id {evidence.name} --evidence-dir {evidence} --execute "
        "--authorized-minutes <本次授权> --authorized-total-minutes <窗口总授权>",
        f"python scripts/v2_p4/t5_gate1_identity.py --evidence-dir {evidence} "
        f"--repo-root {ROOT} --data-root {ROOT / 'data'} --paper-root {ROOT / 'data' / 'paper'} "
        f"--model-version {model_version} --restart-ledger {ledger} --freeze",
        "python scripts/v2_p4/t5_daily_driver.py --mode shadow --trade-date <真实交易日> "
        f"--run-id {evidence.name} --evidence-dir {evidence} --work-root {arms} "
        f"--gate1 {gate1_path} --active-production --execute "
        "--authorized-minutes <本次授权> --authorized-total-minutes <窗口总授权>",
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument(
        "--runs-batch", action="append", default=None, help="显式 runs 批次目录（可多次）"
    )
    parser.add_argument("--focused-tests-report", default=None, help="本次聚焦测试 JSON 报告")
    parser.add_argument("--full-tests-report", default=None, help="本次全量 pytest JSON 报告")
    parser.add_argument("--complexity-report", default=None, help="本次复杂度 JSON 报告")
    args = parser.parse_args(argv)
    status = closeout(
        Path(args.evidence_dir),
        runs_root=Path(args.runs_root) if args.runs_root else None,
        runs_batches=[Path(item) for item in (args.runs_batch or [])] or None,
        focused_tests_report=Path(args.focused_tests_report) if args.focused_tests_report else None,
        full_tests_report=Path(args.full_tests_report) if args.full_tests_report else None,
        complexity_report=Path(args.complexity_report) if args.complexity_report else None,
        run_id=args.run_id,
    )
    print(json.dumps({"verdict": status["verdict"], "gates": status["gates"]}, ensure_ascii=False))
    return 0 if status["verdict"] == "READY_FOR_FINAL_AUTHORIZATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
