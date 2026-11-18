"""R0 只读可行性与封存：输入可得性、单位、时点、环境与预算预测。

R0 不产生任何新候选账户收益读数；未决问题一律显式登记，不做隐性假设。
复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from scripts.v2_p7.defensive_ml import artifacts, sources, spec

STATUS_RANK = {"pass": 0, "warn": 1, "not_estimable": 2, "fail": 3}


def _check(name: str, status: str, detail: Any, note: str = "") -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "note": note}


def _environment() -> dict[str, Any]:
    """运行环境与 XGBoost 分位损失可用性冒烟（合成 20 行，不入训练产物）。"""
    import xgboost as xgb

    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, len(spec.FEATURES)))
    y = rng.normal(size=40)
    started = time.perf_counter()
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.1,
        n_estimators=5,
        max_depth=2,
        tree_method="hist",
        device="cpu",
        random_state=42,
        n_jobs=1,
    )
    model.fit(x, y)
    pred = model.predict(x)
    elapsed = time.perf_counter() - started
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "xgboost": xgb.__version__,
        "cpu_logical": os.cpu_count(),
        "n_jobs_selected": spec.n_jobs(os.cpu_count() or 1),
        "quantile_objective_supported": bool(np.isfinite(pred).all()),
        "quantile_smoke_seconds": round(elapsed, 4),
    }


def _git(root: Path, *args: str) -> str:
    """只读 git 查询（失败返回 not_available，不抛异常）。"""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # noqa: BLE001 - 环境不可用即登记
        return f"not_available:{type(exc).__name__}"
    if done.returncode != 0:
        return "not_available"
    return done.stdout.strip()


def _code_identity(root: Path) -> dict[str, Any]:
    """代码态与执行闭包文件 SHA（协议 §8 身份要求；墙钟不进数值身份）。"""
    closure = [
        "scripts/v2_p7/defensive_ml/spec.py",
        "scripts/v2_p7/defensive_ml/artifacts.py",
        "scripts/v2_p7/defensive_ml/sources.py",
        "scripts/v2_p7/defensive_ml/readiness.py",
        "scripts/v2_p7/defensive_ml/run.py",
        "scripts/v2_p7/common.py",
    ]
    digests = {}
    for rel in closure:
        path = Path(root) / rel
        if path.exists():
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            digests[rel] = "missing"
    status = _git(root, "status", "--porcelain")
    known = bool(status) and not status.startswith("not_available")
    dirty_files = [line for line in status.splitlines() if line.strip()] if known else []
    return {
        "git_head": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(dirty_files) if known else "not_available",
        "dirty_entry_count": len(dirty_files) if known else None,
        "closure_sha256": digests,
    }


def _cashflow_availability(root: Path) -> dict[str, Any]:
    """现金流分区跨度与 schema v3 构建器可导入性（全覆盖率留 R1 实测）。"""
    files = sources.partition_files(Path(root) / "data/raw/cashflow")
    years = sorted({p.stem[:4] for p in files})
    builder_ok = True
    error = ""
    try:
        from src.lazybull.factors.cashflow_quality import (  # noqa: F401
            build_cashflow_quality_lookup_by_date,
        )
    except Exception as exc:  # noqa: BLE001 - 只登记不可用原因
        builder_ok = False
        error = f"{type(exc).__name__}: {exc}"
    first_fold = spec.WINDOW["first_train_signal_start"]
    need_from = int(first_fold[:4]) - int(spec.WINDOW["cashflow_warmup_years"])
    return {
        "partitions": len(files),
        "first": files[0].stem if files else None,
        "last": files[-1].stem if files else None,
        "years": years,
        "required_from_year": need_from,
        "warmup_ok": bool(years) and int(years[0]) <= need_from,
        "builder_importable": builder_ok,
        "builder_error": error,
        "schema_version_expected": 3,
        "coverage_measurement": "deferred_to_R1",
    }


def _early_crash_coverage(root: Path) -> dict[str, Any]:
    """2015/2018 窗口输入可得性（B 账户不可回溯，仅登记事实）。"""
    probes = {}
    for date_dash in ("2015-06-12", "2016-01-07", "2018-12-28"):
        path = Path(root) / "data/clean/daily" / f"{date_dash}.parquet"
        probes[date_dash] = {
            "exists": path.exists(),
            "rows": int(len(pd.read_parquet(path, columns=["ts_code"]))) if path.exists() else 0,
        }
    return {
        "probes": probes,
        "b_account_constructible": False,
        "reason": "首折刷新锚点 2019-04-01，2015/2018 无 B 模型可用；不得用当前模型回测早期冒充 OOS",
        "formal_admission_obstacle": True,
    }


def _budget_forecast() -> dict[str, Any]:
    """协议 §15 规划区间（未在本机测速，不作完成时限承诺）。"""
    return {
        "planned_hours_band": {
            "R0": [1, 2],
            "R1_R3": [7, 12],
            "dataset_models_replay": [5, 10],
            "R5_R6": [3, 6],
            "total": [16, 30],
        },
        "hard_cap_hours_user_approved": "not_filled_by_user",
        "data_purchase_yuan": 0,
        "measured_seconds_this_run": None,
        "note": "分项为规划判断，未测速；实际墙钟按自动工作实际经过时间计量",
    }


def _authorization() -> dict[str, Any]:
    """授权范围登记（用户任务词中的占位未填写，一律显式记录，不代填）。"""
    return {
        "authorized_stages": "not_filled_by_user",
        "technical_cap_hours": "not_filled_by_user",
        "data_purchase_budget_yuan": 0,
        "source": "用户任务词（协议 §17 模板）",
        "template_placeholders_present": True,
        "r0_scope_rationale": "R0 为只读可行性与封存，未产生候选账户收益读数",
    }


def _status_of(checks: list[dict[str, Any]]) -> str:
    worst = max(checks, key=lambda c: STATUS_RANK.get(str(c["status"]), 3))
    if worst["status"] == "fail":
        return "blocked"
    if any(c["status"] in ("warn", "not_estimable") for c in checks):
        return "ready_with_limits"
    return "ready"


def _limitations(
    delist: dict[str, Any],
    industry: dict[str, Any],
    cashflow: dict[str, Any],
    early: dict[str, Any],
) -> list[dict[str, str]]:
    items = [
        {
            "code": "no_independent_delist_registry",
            "statement": "本地 stock_basic 无 list_status/delist_date；退市只能由最后报价日推断，回收额不可界定",
            "impact": "label_status=exit_delisted 可判，退市回收额影响标 unbounded",
        },
        {
            "code": "industry_pit_early_gap",
            "statement": (
                f"行业 PIT 一级映射在早段最低仅 {industry.get('min_l1_ratio')!r}，"
                f"全样本均值 {industry.get('mean_l1_ratio')!r}"
            ),
            "impact": "B 选择期（2019-04 起）覆盖需逐日登记；未映射进 UNKNOWN 且不获行业豁免",
        },
        {
            "code": "cashflow_coverage_unmeasured",
            "statement": f"现金流覆盖率留 R1 实测（builder_importable={cashflow['builder_importable']}）",
            "impact": "R0 只确认分区跨度与构建器存在，不宣称 19 列可得",
        },
        {
            "code": "early_crash_b_account_absent",
            "statement": early["reason"],
            "impact": "正式 P7 型 II 的 2015/2018 冻结窗在场资格构成正式准入障碍（G 包）",
        },
    ]
    if not delist.get("has_independent_delist_registry"):
        items.append(
            {
                "code": "delist_registry_absent_confirmed",
                "statement": "stock_basic 列集已实测：无 list_status/delist_date 字段",
                "impact": "退市事实只能由报价终止推断，需在 R1 逐股登记最后报价日",
            }
        )
    return items


def run_readiness(
    *,
    repo_root: Path,
    campaign: str,
    run_id: str | None = None,
    hash_content: bool = True,
    artifact_base: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """执行 R0：只读核验 -> 落盘 specification/inputs/runs。"""
    emit = log or (lambda msg: None)
    started = time.perf_counter()
    root = Path(repo_root)
    base = Path(artifact_base) if artifact_base is not None else artifacts.default_artifact_base()
    camp = artifacts.campaign_root(campaign, base)
    layout = artifacts.ensure_layout(camp, base)
    rid = run_id or artifacts.new_run_id("readiness")
    run_dir = artifacts.create_run_dir(camp, rid, base)
    emit(f"campaign={camp} run={rid}")

    protocol_sha = spec.protocol_sha256(root)
    snapshot = artifacts.copy_verbatim(
        spec.protocol_path(root), Path(layout["specification"]) / "protocol_snapshot.md", base
    )
    spec_doc = spec.machine_spec_document(protocol_sha)
    spec_written = artifacts.write_json_checked(
        Path(layout["specification"]) / "spec.json", spec_doc, base
    )
    artifacts.write_json_checked(Path(layout["specification"]) / "authorization.json",
                                 _authorization(), base)

    inv = sources.inventory(root, hash_content=hash_content)
    units = sources.verify_units(root)
    industry = sources.industry_coverage(root)
    delist = sources.delisted_presence(root)
    calendar = sources.calendar_span(root)
    warmup = sources.warmup_availability(root)
    baseline = sources.verify_a_baseline(root)
    env = _environment()
    code_ident = _code_identity(root)
    cashflow = _cashflow_availability(root)
    early = _early_crash_coverage(root)

    daily_days = inv["clean_daily"]["n_partitions"]
    checks = [
        _check("protocol_frozen", "pass", {"sha256": protocol_sha, "snapshot": snapshot["sha256"]}),
        _check(
            "spec_frozen",
            "pass",
            {"spec_sha256": spec_doc["spec_sha256"], "path": spec_written["path"]},
        ),
        _check("a_baseline_identity", baseline["status"], baseline),
        _check(
            "trade_calendar",
            "pass" if warmup["warmup_ok"] and calendar["anchor_present"]["20190401"] else "fail",
            {"calendar": calendar, "warmup": warmup},
        ),
        _check(
            "daily_partitions",
            "pass" if daily_days >= 3600 else "warn",
            {
                "n": daily_days,
                "first": inv["clean_daily"]["first"],
                "last": inv["clean_daily"]["last"],
            },
        ),
        _check("units", units["status"], units["checks"]),
        _check(
            "industry_pit",
            "pass" if (industry["min_l1_ratio"] or 0) >= 0.95 else "warn",
            {
                "min_l1_ratio": industry["min_l1_ratio"],
                "mean_l1_ratio": industry["mean_l1_ratio"],
                "unmapped_pit_rows": industry["unmapped_pit_rows"],
            },
            note="早段覆盖不足，B 选择期需逐日登记；未映射进 UNKNOWN",
        ),
        _check(
            "delist_traceability",
            "warn" if not delist["has_independent_delist_registry"] else "pass",
            delist,
            note="无独立退市登记表，退市由最后报价日推断",
        ),
        _check(
            "cashflow_inputs",
            "pass" if cashflow["warmup_ok"] and cashflow["builder_importable"] else "warn",
            cashflow,
        ),
        _check(
            "code_identity",
            "pass"
            if all(v != "missing" for v in code_ident["closure_sha256"].values())
            else "warn",
            code_ident,
        ),
        _check(
            "environment",
            "pass" if env["quantile_objective_supported"] else "fail",
            env,
        ),
        _check("early_crash_windows", "not_estimable", early,
               note="输入齐备但 B 账户不可构造；正式准入障碍"),
    ]

    status = _status_of(checks)
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
        "limitations": _limitations(delist, industry, cashflow, early),
        "code_identity": code_ident,
        "authorization": _authorization(),
        "budget_forecast": _budget_forecast(),
        "formal_admission": "not_evaluated",
        "production_changed": False,
        "generated_at": artifacts.utc_now(),
        "elapsed_seconds": None,
    }

    ledger_paths = {
        "inputs/source_manifest.json": {"campaign": campaign, "inventory": inv},
        "inputs/unit_audit.json": units,
        "inputs/industry_coverage.json": industry,
        "inputs/delist_traceability.json": delist,
        "inputs/coverage_ledger.json": {
            "calendar": calendar,
            "warmup": warmup,
            "cashflow": cashflow,
            "early_crash": early,
        },
    }
    written = {}
    for rel, payload in ledger_paths.items():
        written[rel] = artifacts.write_json_checked(camp / rel, payload, base)

    readiness["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    readiness_path = artifacts.write_json_checked(run_dir / "readiness.json", readiness, base)
    seal = artifacts.seal_stage(
        run_dir,
        "readiness",
        {
            "readiness_sha256": readiness_path["sha256"],
            "spec_sha256": spec_doc["spec_sha256"],
            "protocol_sha256": protocol_sha,
            "inputs": written,
        },
        artifact_base=base,
    )
    return {"readiness": readiness, "readiness_path": readiness_path, "seal": seal,
            "layout": layout, "run_dir": str(run_dir)}


def load_json(path: Path) -> Any:
    """读 JSON（UTF-8）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
