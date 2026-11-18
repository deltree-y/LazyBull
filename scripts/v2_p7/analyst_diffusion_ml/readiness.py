"""R0 只读可行性与封存：A 身份、report_rc 口径、历史域、100+ 覆盖、单位与水位。

R0 在读取任何候选收益之前完成；覆盖门不通过即 `not_ready`，不得放宽分析师数、
改股票域或改用全历史并集。

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import (
    artifacts,
    dataset,
    panel,
    report_events,
    sources,
    spec,
    universe,
)
from scripts.v2_p7.common import REPO_ROOT, sha256_file

CLOSURE_FILES: tuple[str, ...] = (
    "scripts/v2_p7/analyst_diffusion_ml/spec.py",
    "scripts/v2_p7/analyst_diffusion_ml/report_events.py",
    "scripts/v2_p7/analyst_diffusion_ml/diffusion.py",
    "scripts/v2_p7/analyst_diffusion_ml/universe.py",
    "scripts/v2_p7/analyst_diffusion_ml/dataset.py",
    "scripts/v2_p7/analyst_diffusion_ml/labels.py",
)

STATUS_RANK = {"pass": 0, "warn": 1, "not_estimable": 2, "fail": 3}


def _check(name: str, status: str, detail: Any = None, note: str = "") -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "note": note}


def _git_identity() -> dict[str, Any]:
    """代码身份：commit + dirty 标记（不可得时降级但不伪装干净）。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return {
            "commit": commit or None,
            "dirty": bool(status),
            "dirty_lines": status.splitlines()[:5],
        }
    except OSError as exc:  # pragma: no cover - 环境缺 git
        return {"commit": None, "dirty": None, "error": str(exc)}


def _closure() -> dict[str, str]:
    """执行闭包：本包关键模块的字节 SHA。"""
    out: dict[str, str] = {}
    for relative in CLOSURE_FILES:
        path = REPO_ROOT / relative
        out[relative] = sha256_file(path) if path.exists() else "missing"
    return out


def _environment() -> dict[str, Any]:
    """运行环境（只作登记，不进数值身份）。"""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    try:
        import xgboost

        info["xgboost"] = xgboost.__version__
    except ImportError:  # pragma: no cover
        info["xgboost"] = None
    return info


def _coverage_gate(root: Path, log: Callable[[str], None]) -> dict[str, Any]:
    """主窗每日分析师覆盖（基础域 ∩ N_active>=2），返回覆盖门台账。"""
    calendar = panel.load_calendar(root)
    start = str(spec.WINDOW["start"])
    end = str(spec.WINDOW["end"])
    first = next(i for i, day in enumerate(calendar) if day >= start)
    warmup = int(spec.WINDOW["dataset_load_warmup_trading_days"])
    load_start = calendar[max(0, first - warmup)]
    inputs = dataset.load_inputs(root, load_start, end)
    base = universe.build_universe(
        inputs["prices"],
        inputs["basic"],
        inputs["stock_basic"],
        inputs["industry_pit"],
        min_list_days=int(spec.UNIVERSE["min_list_days"]),
        min_mv_yuan=float(spec.UNIVERSE["min_total_mv_yuan"]),
        max_mv_yuan=float(spec.UNIVERSE["max_total_mv_yuan"]),
        min_amount20_yuan=float(spec.UNIVERSE["min_amount20_yuan"]),
        amount_min_periods=int(spec.UNIVERSE["amount_min_periods"]),
        listing_calendar=calendar,
        l1_names=inputs["l1_names"],
    )
    log(f"覆盖门：基础域完成，条件计数 {base.condition_counts}")
    events = report_events.build_events(
        report_events.load_report_rc(root), trading_days=calendar, log=log
    )
    coverage = report_events.analyst_coverage_matrix(
        events["activity"], events["primary"], inputs["dates"], inputs["codes"]
    )
    domain = universe.daily_domain(
        base,
        coverage["n_active"],
        int(spec.UNIVERSE["min_active_analysts_90"]),
        int(spec.UNIVERSE["min_scored_per_day"]),
    )
    domain = domain[(domain["trade_date"] >= start) & (domain["trade_date"] <= end)]
    gate = dataset.coverage_gate(domain)
    check = _check(
        "analyst_coverage_gate",
        "pass" if gate["status"] == "ready" else "fail",
        gate,
        note="不足 100 只的日期写 universe_below_minimum，当日不产生新买排名",
    )
    return {
        "check": check,
        "domain": domain,
        "gate": gate,
        "events": events,
        "base": base,
        "coverage": coverage,
    }


def run_readiness(
    *,
    repo_root: Path = REPO_ROOT,
    campaign: str = spec.DEFAULT_CAMPAIGN,
    run_id: str | None = None,
    artifact_base: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """R0：写 specification 与 inputs 证据，返回 readiness 台账。"""
    emit = log or (lambda message: None)
    started = time.perf_counter()
    root = Path(repo_root)
    camp = artifacts.campaign_root(campaign, artifact_base)
    artifacts.ensure_layout(camp, artifact_base)
    rid = run_id or artifacts.new_run_id("readiness")
    run_dir = artifacts.create_run_dir(camp, rid, artifact_base)
    emit(f"R0 run={rid} campaign={campaign}")

    protocol_sha = spec.protocol_sha256(root)
    spec_doc = spec.machine_spec_document(protocol_sha)
    specification = camp / "specification"
    spec_written = artifacts.write_json_checked(
        specification / "machine_spec.json", spec_doc, artifact_base
    )
    snapshot = artifacts.copy_verbatim(
        spec.protocol_path(root), specification / "protocol.md", artifact_base
    )
    history = artifacts.write_json_checked(
        specification / "selection_history.json",
        {
            "candidate_id": spec.CANDIDATE_ID,
            "history": [dict(item) for item in spec.SELECTION_HISTORY],
            "note": "正式台账未写；本文件只登记本轮选择史，不改动 data/ledger",
        },
        artifact_base,
    )
    authorization = artifacts.write_json_checked(
        specification / "authorization_scope.json",
        {
            "stages": list(spec.STAGES),
            "ai_wall_clock_hours_max": 28,
            "machine_hours_max": 8,
            "data_purchase_yuan": 0,
            "production_changes": False,
            "formal_ledger_writes": False,
            "review_initiated_by": "user_only",
            "not_virgin_market_holdout": True,
            "note": "G 包（正式 P7 准入）不在 R0—R6 工时内，也不在本次授权内",
        },
        artifact_base,
    )

    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "protocol_frozen",
            "pass" if snapshot["sha256"] == protocol_sha else "fail",
            {"sha256": protocol_sha, "snapshot": snapshot},
        )
    )
    checks.append(
        _check(
            "spec_frozen",
            "pass",
            {"spec_sha256": spec_doc["spec_sha256"], "written": spec_written["sha256"]},
        )
    )

    inventory = sources.inventory(root)
    (camp / "inputs").mkdir(parents=True, exist_ok=True)
    manifest_written = artifacts.write_json_checked(
        camp / "inputs" / "source_manifest.json",
        {
            "candidate_id": spec.CANDIDATE_ID,
            "protocol_sha256": protocol_sha,
            "spec_sha256": spec_doc["spec_sha256"],
            "files": inventory,
        },
        artifact_base,
    )
    missing = [item["name"] for item in inventory if not item["exists"]]
    checks.append(
        _check(
            "inputs_present",
            "pass" if not missing else "fail",
            {"missing": missing, "count": len(inventory)},
        )
    )

    baseline = sources.verify_a_baseline(root)
    checks.append(
        _check(
            "a_baseline_identity",
            baseline["status"],
            baseline,
            note="A 为历史 half-Kelly 研究账户，不等于当前纸面账户",
        )
    )

    calendar = sources.calendar_span(root)
    checks.append(
        _check(
            "trade_calendar",
            "pass" if all(calendar["anchors_present"].values()) else "fail",
            calendar,
        )
    )

    watermark = sources.daily_watermark(root)
    checks.append(
        _check(
            "daily_watermark",
            "pass" if watermark["daily"]["last"] == f"{str(spec.WINDOW['end'])[:4]}-"
            f"{str(spec.WINDOW['end'])[4:6]}-{str(spec.WINDOW['end'])[6:]}" else "warn",
            watermark,
        )
    )

    units = sources.check_units(root)
    checks.append(_check("clean_units", units["status"], units))

    industry = sources.industry_coverage(root)
    checks.append(
        _check(
            "industry_pit",
            "pass" if industry["min_known_ratio"] >= 0.95 else "warn",
            industry,
            note="低于 95% 时未映射股票按 industry_classification_unknown 排除",
        )
    )

    raw = report_events.load_report_rc(root)
    audit = sources.report_rc_audit(raw)
    (camp / "inputs").mkdir(parents=True, exist_ok=True)
    artifacts.write_json_checked(
        camp / "inputs" / "report_rc_audit.json", audit, artifact_base
    )
    schema_ok = (
        not audit["missing_columns"]
        and audit["field_availability"]["quarter_parsed_ratio"] > 0.95
    )
    checks.append(
        _check(
            "report_rc_schema",
            "pass" if schema_ok else "fail",
            {
                "missing_columns": audit["missing_columns"],
                "quarter_parsed_ratio": audit["field_availability"]["quarter_parsed_ratio"],
                "rows": audit["rows"],
            },
        )
    )
    eps_p50 = audit["eps_quantiles"]["p50"]
    target_p50 = audit["target_quantiles"]["p50"]
    unit_ok = (
        eps_p50 is not None
        and -5.0 <= float(eps_p50) <= 20.0
        and target_p50 is not None
        and 1.0 <= float(target_p50) <= 5000.0
    )
    checks.append(
        _check(
            "report_rc_units",
            "pass" if unit_ok else "fail",
            {"eps_p50": eps_p50, "target_p50": target_p50, "unit": audit["unit_conclusion"]},
            note="EPS 元/股、目标价 元/股；不可核即停止，不猜换算",
        )
    )
    checks.append(
        _check(
            "report_rc_conflicts",
            "pass",
            audit["conflict"],
            note="冲突组整组隔离并留原行 SHA；不按行序 keep=last",
        )
    )

    delisted = sources.delisted_traceability(root, str(spec.WINDOW["end"]))
    checks.append(_check("delist_traceability", delisted["status"], delisted))

    gate_result = _coverage_gate(root, emit)
    checks.append(gate_result["check"])
    camp_domain = gate_result["domain"]
    artifacts.assert_under(camp / "inputs" / "analyst_coverage.parquet", camp)
    camp_domain.to_parquet(camp / "inputs" / "analyst_coverage.parquet", index=False)

    code_identity = {"git": _git_identity(), "closure": _closure()}
    checks.append(
        _check(
            "code_identity",
            "pass" if code_identity["git"]["commit"] else "warn",
            code_identity,
        )
    )
    environment = _environment()
    checks.append(
        _check("environment", "pass" if environment.get("xgboost") else "fail", environment)
    )

    statuses = [item["status"] for item in checks]
    worst = max(statuses, key=lambda item: STATUS_RANK[item])
    status = {"pass": "ready", "warn": "ready_with_limits"}.get(worst, "blocked")
    if gate_result["gate"]["status"] != "ready":
        status = "blocked"

    readiness = {
        "stage": "readiness",
        "candidate_id": spec.CANDIDATE_ID,
        "protocol_version": spec.PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha,
        "spec_sha256": spec_doc["spec_sha256"],
        "campaign": campaign,
        "run_id": rid,
        "status": status,
        "checks": checks,
        "coverage_gate": gate_result["gate"],
        "formal_admission": "not_evaluated",
        "production_changed": False,
        "selection_history": history["sha256"],
        "authorization": authorization["sha256"],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "generated_at": artifacts.utc_now(),
    }
    readiness_path = artifacts.write_json_checked(
        run_dir / "readiness.json", readiness, artifact_base
    )
    artifacts.seal_stage(
        run_dir,
        "readiness",
        {
            "readiness_sha256": readiness_path["sha256"],
            "spec_sha256": spec_doc["spec_sha256"],
            "protocol_sha256": protocol_sha,
            "source_manifest_sha256": manifest_written["sha256"],
            "status": status,
        },
        artifact_base=artifact_base,
    )
    emit(f"R0 状态={status}，耗时 {readiness['elapsed_seconds']}s")
    return {**readiness, "run_dir": str(run_dir), "readiness_sha256": readiness_path["sha256"]}
