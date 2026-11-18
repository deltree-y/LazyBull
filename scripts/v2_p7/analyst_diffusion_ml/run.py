"""AD-ML-20-v1 统一阶段 CLI（协议 §12）。

只开放 `campaign / stage / run-id` 与上游 run 身份；不开放任何规格扫描参数。
规格从封存 machine_spec 加载；改变规格必须新 campaign。

用法：
    python -m scripts.v2_p7.analyst_diffusion_ml.run ^
      --campaign adml20_v1_20261118 --stage readiness --run-id R0

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from scripts.v2_p7.analyst_diffusion_ml import (
    artifacts,
    native_replay,
    panel,
    prediction,
    sources,
    spec,
)
from scripts.v2_p7.common import REPO_ROOT, sha256_file

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 3


class StageError(RuntimeError):
    """阶段无法继续（上游缺件、口径不符、预算超限）。"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.v2_p7.analyst_diffusion_ml.run",
        description="AD-ML-20-v1 研究阶段 CLI（规格冻结，无参数扫描入口）",
    )
    parser.add_argument("--campaign", default=spec.DEFAULT_CAMPAIGN)
    parser.add_argument("--stage", required=True, choices=list(spec.STAGES))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset-run", default=None)
    parser.add_argument("--predictions-run", default=None)
    parser.add_argument("--replay-run", default=None)
    parser.add_argument("--diagnostics-run", default=None)
    parser.add_argument("--train-run", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--machine-hours", type=float, default=8.0)
    parser.add_argument("--no-stress", action="store_true")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--artifact-base", default=None)
    parser.add_argument(
        "--readiness-run",
        default="",
        help="就绪 run 身份（数据/探针/训练/回放阶段必填，必须是 status=ready 的发布件）",
    )
    return parser


def _log(stage: str, run_id: str) -> Callable[[str], None]:
    def emit(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}][{stage}/{run_id}] {message}", flush=True)

    return emit


def _job_count() -> int:
    return spec.n_jobs(os.cpu_count() or 1)


def _train_payload(camp: Path, run_id: str | None) -> dict[str, Any]:
    """读上游训练 run 的封存 payload（模型摘要与折清单的唯一来源）。"""
    if not run_id:
        raise StageError("缺少上游训练 run 身份：--predictions-run")
    path = Path(camp) / "runs" / run_id / "manifest.json"
    if not path.exists():
        raise StageError(f"上游训练 run 未封存：{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")).get("payload") or {})


def _stage_workspace(run_dir: Path, camp: Path, upstream: dict[str, str | None]) -> Path:
    """为上游已发布工件建立只读工作副本（缺失即拒绝）。"""
    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    for name, run_id in upstream.items():
        if not run_id:
            raise StageError(f"阶段缺少上游 run 身份：{name}")
        source = artifacts.require_published(camp, run_id, name)
        target = work / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    return work


def run_readiness_stage(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.v2_p7.analyst_diffusion_ml import readiness

    base = Path(args.artifact_base) if args.artifact_base else None
    result = readiness.run_readiness(
        repo_root=Path(args.repo_root),
        campaign=args.campaign,
        run_id=args.run_id,
        artifact_base=base,
        log=_log("readiness", args.run_id or "auto"),
    )
    return {
        "status": result["status"],
        "run_dir": result["run_dir"],
        "readiness_sha256": result["readiness_sha256"],
        "coverage_gate": result["coverage_gate"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def run_dataset_stage(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.v2_p7.analyst_diffusion_ml import identity, pipeline

    base = Path(args.artifact_base) if args.artifact_base else None
    camp = artifacts.campaign_root(args.campaign, base)
    ready = identity.require_ready(camp, args.readiness_run)
    spec_gate = identity.spec_identity(camp)
    input_gate = identity.verify_inputs(Path(args.repo_root), camp)
    manifest_path = camp / "inputs" / "source_manifest.json"
    source_manifest_sha = sha256_file(manifest_path)
    rid = args.run_id or artifacts.new_run_id("dataset")
    run_dir = artifacts.create_run_dir(camp, rid, base)
    emit = _log("dataset", rid)
    result = pipeline.run_dataset(
        Path(args.repo_root),
        camp,
        source_manifest_sha=source_manifest_sha,
        work_dir=run_dir / "work",
        log=emit,
    )
    published = artifacts.publish_artifacts(
        run_dir / "work",
        run_dir,
        "dataset",
        artifacts.STAGE_ARTIFACTS["dataset"],
        {"source_manifest_sha256": source_manifest_sha, "spec_sha256": spec.spec_sha256()},
        base,
    )
    artifacts.seal_stage(
        run_dir,
        "dataset",
        {
            "spec_sha256": spec.spec_sha256(),
            "source_manifest_sha256": source_manifest_sha,
            "readiness": ready,
            "spec_gate": spec_gate,
            "input_gate": input_gate,
            "artifacts": published,
            "summary": result["summary"],
        },
        artifact_base=base,
    )
    return {"status": "ok", "run_dir": str(run_dir), "summary": result["summary"]}


def run_model_probe_stage(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.v2_p7.analyst_diffusion_ml import identity, pipeline

    base = Path(args.artifact_base) if args.artifact_base else None
    camp = artifacts.campaign_root(args.campaign, base)
    ready = identity.require_ready(camp, args.readiness_run)
    spec_gate = identity.spec_identity(camp)
    rid = args.run_id or artifacts.new_run_id("probe")
    run_dir = artifacts.create_run_dir(camp, rid, base)
    work = _stage_workspace(run_dir, camp, {"dataset": args.dataset_run})
    probe = pipeline.run_model_probe(
        Path(args.repo_root),
        camp,
        work / "dataset",
        work_dir=run_dir / "work" / "probe",
        n_jobs=_job_count(),
        log=_log("probe", rid),
    )
    if not probe.get("finite_predictions"):
        raise StageError(f"首折探针出现非有限预测：{probe}")
    artifacts.seal_stage(
        run_dir,
        "model_probe",
        {
            "spec_sha256": spec.spec_sha256(),
            "dataset_run": args.dataset_run,
            "readiness": ready,
            "spec_gate": spec_gate,
            "probe": probe,
        },
        artifact_base=base,
    )
    return {"status": "ok", "run_dir": str(run_dir), "probe": probe}


def run_train_predict_stage(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.v2_p7.analyst_diffusion_ml import identity, pipeline

    base = Path(args.artifact_base) if args.artifact_base else None
    camp = artifacts.campaign_root(args.campaign, base)
    ready = identity.require_ready(camp, args.readiness_run)
    spec_gate = identity.spec_identity(camp)
    input_gate = identity.verify_inputs(Path(args.repo_root), camp)
    protocol_sha = str(spec_gate["protocol_sha256"])
    rid = args.run_id or artifacts.new_run_id("train")
    run_dir = artifacts.create_run_dir(camp, rid, base)
    work = _stage_workspace(run_dir, camp, {"dataset": args.dataset_run})
    result = pipeline.run_train_predict(
        Path(args.repo_root),
        camp,
        work / "dataset",
        work_dir=run_dir / "work",
        n_jobs=_job_count(),
        protocol_sha=protocol_sha,
        log=_log("train", rid),
    )
    published = artifacts.publish_artifacts(
        run_dir / "work",
        run_dir,
        "train_predict",
        artifacts.STAGE_ARTIFACTS["train_predict"],
        {
            "dataset_run": args.dataset_run,
            "spec_sha256": spec.spec_sha256(),
            "fold_models_sha256": sha256_file(camp / "models" / "fold_models.json"),
        },
        base,
    )
    artifacts.seal_stage(
        run_dir,
        "train_predict",
        {
            "spec_sha256": spec.spec_sha256(),
            "protocol_sha256": protocol_sha,
            "dataset_run": args.dataset_run,
            "readiness": ready,
            "spec_gate": spec_gate,
            "input_gate": input_gate,
            "models_digest": identity.models_digest(camp),
            "artifacts": published,
            "summary": result["summary"],
            "folds": result["records"],
        },
        artifact_base=base,
    )
    return {"status": "ok", "run_dir": str(run_dir), "summary": result["summary"]}


def _write_sleeve(
    campaign_dir: Path, capital: int, label: str, arm: dict[str, Any]
) -> dict[str, Any]:
    base = Path(campaign_dir) / "accounts" / str(capital)
    base.mkdir(parents=True, exist_ok=True)
    daily = arm["daily"]
    daily.to_parquet(base / f"{label}_daily.parquet", index=False)
    if len(arm["trades"]):
        arm["trades"].to_parquet(base / f"{label}_trades.parquet", index=False)
    if len(arm["holdings"]):
        arm["holdings"].to_parquet(base / f"{label}_holdings.parquet", index=False)
    (base / f"{label}_diagnostics.json").write_text(
        json.dumps(arm["diagnostics"], ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    return {
        "capital": int(capital),
        "label": label,
        "rows": int(len(daily)),
        "trades": int(len(arm["trades"])),
        "ending": float(daily["total_value"].iloc[-1]) if len(daily) else None,
        "validation": arm["validation"],
    }


def run_replay_stage(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.v2_p7.analyst_diffusion_ml import identity

    base = Path(args.artifact_base) if args.artifact_base else None
    camp = artifacts.campaign_root(args.campaign, base)
    ready = identity.require_ready(camp, args.readiness_run)
    spec_gate = identity.spec_identity(camp)
    input_gate = identity.verify_inputs(Path(args.repo_root), camp)
    a_gate = sources.verify_a_baseline(Path(args.repo_root))
    if a_gate.get("status") != "pass":
        raise StageError(f"A 基准身份复核不通过：{a_gate}")
    train_payload = _train_payload(camp, args.predictions_run)
    models_gate = identity.require_models(camp, train_payload.get("models_digest"))
    rid = args.run_id or artifacts.new_run_id("replay")
    run_dir = artifacts.create_run_dir(camp, rid, base)
    stage_started = time.time()
    emit = _log("replay", rid)
    work = _stage_workspace(
        run_dir, camp, {"dataset": args.dataset_run, "predictions": args.predictions_run}
    )
    days = native_replay.replay_days(Path(args.repo_root))
    calendar = panel.load_calendar(Path(args.repo_root))
    ml_table = pd.read_parquet(work / "predictions" / "predictions_all.parquet",
                               columns=["trade_date", "ts_code", "rank"])
    simple_table = pd.read_parquet(work / "predictions" / "predictions_simple.parquet",
                                   columns=["trade_date", "ts_code", "rank"])
    ranks = {
        "ML": prediction.rank_table_lookup(ml_table, days),
        "SIMPLE": prediction.rank_table_lookup(simple_table, days),
    }
    pit, names = native_replay.load_industry(Path(args.repo_root))
    price_days = [day for day in calendar if day <= days[-1]][-(len(days) + 260):]
    from scripts.v2_p7.buffer_native_replay import load_prices
    from src.lazybull.v2.core.execution.price_index import prepare_price_index

    prices, _identity = load_prices(Path(args.repo_root), price_days, warmup_cutoff=days[0])
    indexes = prepare_price_index(prices)
    emit(f"价格与排名就绪：{len(days)} 交易日")

    summary: dict[str, Any] = {"capital": {}, "sleeves": []}
    for capital in spec.TARGETS["capital_scenarios"]:
        capital = int(capital)
        b_capital = float(capital) * float(spec.TARGETS["initial_b_fraction"])
        sleeve_series: dict[str, pd.Series] = {}
        for label in ("ML", "SIMPLE"):
            book, audit = native_replay.prepare_corporate(
                Path(args.repo_root), camp, days, tag=f"base_{capital}"
            )
            arm = native_replay.run_arm(
                Path(args.repo_root),
                camp,
                ranked=ranks[label],
                days=days,
                calendar=calendar,
                capital=b_capital,
                corporate_book=book,
                pit=pit,
                l1_names=names,
                anchor=str(spec.WINDOW["cash_anchor"]),
                prices=prices,
                indexes=indexes,
                log=emit,
            )
            arm["diagnostics"]["corporate"] = {
                "unresolved_groups": audit["audit"].get("n_unresolved"),
                "rows_dropped": audit["audit"].get("rows_dropped"),
                "rule": audit["audit"].get("rule"),
                "tag": audit["tag"],
            }
            summary["sleeves"].append(_write_sleeve(camp, capital, label, arm))
            sleeve_series[label] = arm["daily"].set_index("trade_date")["total_value"].astype(float)
        a100 = native_replay.load_a_series(Path(args.repo_root), capital, "A100")
        a80_cash = native_replay.load_a_series(Path(args.repo_root), capital, "A80_CASH20")
        expected = native_replay.load_a_series(Path(args.repo_root), capital, "A80")
        if not (
            (a80_cash.to_numpy() - expected.to_numpy() - b_capital).__abs__() < 1e-6
        ).all():
            raise StageError(f"资本 {capital} 的 A80+现金 与 A80 本体不自洽（不得拼接净值）")
        arms = native_replay.combine_arms(
            capital,
            a100,
            expected,
            {"AD": sleeve_series["ML"], "AD_SIMPLE": sleeve_series["SIMPLE"]},
            b_capital,
        )
        common = a100.index.intersection(arms.index)
        if len(common) != len(a100.index) or len(arms.index) != len(a100.index):
            raise StageError(
                f"资本 {capital} 日期轴不一致：A={len(a100.index)} 合并={len(arms.index)}"
            )
        target = Path(camp) / "accounts" / str(capital) / "arms.parquet"
        arms.to_parquet(target)
        summary["capital"][str(capital)] = {
            "rows": int(len(arms)),
            "first": str(arms.index[0]),
            "last": str(arms.index[-1]),
            "ending": {name: float(arms[name].iloc[-1]) for name in arms.columns},
        }
        emit(f"资本 {capital} 四臂完成：{len(arms)} 行")
    (Path(camp) / "accounts" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    published = artifacts.publish_artifacts(
        camp, run_dir, "replay", artifacts.STAGE_ARTIFACTS["replay"],
        {"dataset_run": args.dataset_run, "predictions_run": args.predictions_run,
         "spec_sha256": spec.spec_sha256()},
        base,
    )
    artifacts.seal_stage(
        run_dir, "replay",
        {"spec_sha256": spec.spec_sha256(), "dataset_run": args.dataset_run,
         "predictions_run": args.predictions_run, "artifacts": published, "summary": summary,
         "readiness": ready, "spec_gate": spec_gate, "input_gate": input_gate,
         "a_baseline_gate": a_gate, "models_gate": models_gate,
         "elapsed_seconds": round(time.time() - stage_started, 1)},
        artifact_base=base,
    )
    return {"status": "ok", "run_dir": str(run_dir), "summary": summary["capital"]}


def run_diagnostics_stage(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.v2_p7.analyst_diffusion_ml import diagnostics, identity

    base = Path(args.artifact_base) if args.artifact_base else None
    camp = artifacts.campaign_root(args.campaign, base)
    rid = args.run_id or artifacts.new_run_id("diagnostics")
    run_dir = artifacts.create_run_dir(camp, rid, base)
    stage_started = time.time()
    emit = _log("diagnostics", rid)
    work = _stage_workspace(
        run_dir,
        camp,
        {"dataset": args.dataset_run, "predictions": args.predictions_run},
    )
    accounts_dir = artifacts.require_published(camp, args.replay_run, "accounts")
    input_gate = identity.verify_inputs(Path(args.repo_root), camp)
    models_gate = identity.require_models(
        camp, _train_payload(camp, args.predictions_run).get("models_digest")
    )
    result = diagnostics.run_diagnostics(
        Path(args.repo_root),
        camp,
        dataset_dir=work / "dataset",
        predictions_dir=work / "predictions",
        accounts_dir=accounts_dir,
        work_dir=run_dir / "work",
        run_stress_flag=not args.no_stress,
        log=emit,
    )
    result["gates"] = {
        "input": input_gate,
        "models": models_gate,
        "accounts_run": args.replay_run,
    }
    out_dir = run_dir / "work" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    published = artifacts.publish_artifacts(
        run_dir / "work", run_dir, "diagnostics", artifacts.STAGE_ARTIFACTS["diagnostics"],
        {"dataset_run": args.dataset_run, "predictions_run": args.predictions_run,
         "replay_run": args.replay_run, "spec_sha256": spec.spec_sha256()},
        base,
    )
    artifacts.seal_stage(
        run_dir, "diagnostics",
        {"spec_sha256": spec.spec_sha256(), "dataset_run": args.dataset_run,
         "predictions_run": args.predictions_run, "replay_run": args.replay_run,
         "artifacts": published, "stress_ran": not args.no_stress,
         "elapsed_seconds": round(time.time() - stage_started, 1)},
        artifact_base=base,
    )
    return {"status": "ok", "run_dir": str(run_dir), "stress_ran": not args.no_stress}


def _payload_seconds(camp: Path, run_id: str | None, where: str) -> float | None:
    """从指定 run 的封存 payload 取耗时（不跨 run 猜）。"""
    if not run_id:
        return None
    path = Path(camp) / "runs" / run_id / "manifest.json"
    if not path.exists():
        return None
    payload = dict(json.loads(path.read_text(encoding="utf-8")).get("payload") or {})
    if where == "summary":
        return (payload.get("summary") or {}).get("elapsed_seconds")
    return payload.get("elapsed_seconds")


def _startup_trimmed(campaign_dir: Path) -> dict[str, Any]:
    """去除最初 20 个交易日后的诊断（协议 §13.1，不替换主窗）。

    直接从同一批已封存四臂账派生；主窗读数仍以诊断阶段为准。
    """
    from scripts.v2_p7.analyst_diffusion_ml import evaluation

    out: dict[str, Any] = {}
    skip = int(spec.WINDOW["startup_exclusion_days"])
    for capital in spec.TARGETS["capital_scenarios"]:
        path = Path(campaign_dir) / "accounts" / str(int(capital)) / "arms.parquet"
        if not path.exists():
            out[str(int(capital))] = {"status": "not_estimable", "reason": "缺四臂账"}
            continue
        arms = pd.read_parquet(path)
        arms.index = arms.index.astype(str)
        if len(arms) <= skip + 30:
            out[str(int(capital))] = {"status": "not_estimable", "reason": "窗口过短"}
            continue
        trimmed = arms.iloc[skip:]
        series = {name: trimmed[name].astype(float) for name in trimmed.columns}
        out[str(int(capital))] = {
            "status": "pass",
            "skip_days": skip,
            "first": str(trimmed.index[0]),
            "target_vs_a100": evaluation.target_check(
                series["A80_AD20"], series["A100"], series["A80_CASH20"]
            ),
        }
    return out


def run_report_stage(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.v2_p7.analyst_diffusion_ml import diagnostics as diagnostics_mod
    from scripts.v2_p7.analyst_diffusion_ml import identity, report, readiness

    base = Path(args.artifact_base) if args.artifact_base else None
    camp = artifacts.campaign_root(args.campaign, base)
    rid = args.run_id or artifacts.new_run_id("report")
    run_dir = artifacts.create_run_dir(camp, rid, base)
    diag_dir = artifacts.require_published(camp, args.diagnostics_run, "diagnostics")
    diagnostics = json.loads((diag_dir / "diagnostics.json").read_text(encoding="utf-8"))
    # 覆盖读数从**发布件**重算：诊断 JSON 可能早于本口径，报告不得沿用旧字段
    coverage = diagnostics_mod.coverage_diagnostics(
        artifacts.require_published(camp, args.dataset_run, "dataset"),
        artifacts.require_published(camp, args.predictions_run, "predictions"),
    )
    diagnostics["coverage"] = {**(diagnostics.get("coverage") or {}), **coverage}
    machine_spec = json.loads(
        (camp / "specification" / "machine_spec.json").read_text(encoding="utf-8")
    )
    train_payload = _train_payload(camp, args.predictions_run)
    models_gate = identity.require_models(camp, train_payload.get("models_digest"))
    fold_records = list(train_payload.get("folds") or [])
    model_summary: dict[str, Any] = {}
    if fold_records:
        model_summary = {
            "folds": len(fold_records),
            "models_registered": len(fold_records),
            "reload_all_identical": all(
                bool(item.get("reload_identical")) for item in fold_records
            ),
            "oos_metrics": (train_payload.get("summary") or {}).get("oos_metrics", {}),
        }
    run_summaries: dict[str, Any] = {
        "dataset_seconds": _payload_seconds(camp, args.dataset_run, "summary"),
        "train_seconds": _payload_seconds(camp, args.predictions_run, "summary"),
        "replay_seconds": _payload_seconds(camp, args.replay_run, "payload"),
        "diagnostics_seconds": _payload_seconds(camp, args.diagnostics_run, "payload"),
    }
    meta = {
        "campaign": args.campaign,
        "protocol_sha256": machine_spec["protocol_sha256"],
        "generated_at": artifacts.utc_now(),
        "identity": {"git": readiness._git_identity(), "closure": readiness._closure()},
        "model_summary": model_summary,
        "budget": run_summaries,
        "startup_trimmed": _startup_trimmed(camp),
        "folds": fold_records,
        "upstream": {
            "dataset_run": args.dataset_run,
            "predictions_run": args.predictions_run,
            "replay_run": args.replay_run,
            "diagnostics_run": args.diagnostics_run,
            "models_gate": models_gate,
            "diagnostics_gates": diagnostics.get("gates") or {},
        },
    }
    written = report.write_report(camp, diagnostics, meta)
    run_dir_work = run_dir / "work"
    (run_dir_work / "reports").mkdir(parents=True, exist_ok=True)
    for name in ("machine_summary.json", "report_zh.md"):
        shutil.copy2(Path(written["summary"]).parent / name, run_dir_work / "reports" / name)
    published = artifacts.publish_artifacts(
        run_dir_work, run_dir, "report", artifacts.STAGE_ARTIFACTS["report"],
        {"diagnostics_run": args.diagnostics_run, "spec_sha256": spec.spec_sha256()},
        base,
    )
    decision = json.loads(Path(written["summary"]).read_text(encoding="utf-8"))["decision"]
    artifacts.seal_stage(
        run_dir, "report",
        {"spec_sha256": spec.spec_sha256(), "diagnostics_run": args.diagnostics_run,
         "artifacts": published, "resource_recommendation": decision["resource_recommendation"]},
        artifact_base=base,
    )
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "resource_recommendation": decision["resource_recommendation"],
        "report": written["report"],
    }


STAGE_HANDLERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "readiness": run_readiness_stage,
    "dataset": run_dataset_stage,
    "model_probe": run_model_probe_stage,
    "train_predict": run_train_predict_stage,
    "replay": run_replay_stage,
    "diagnostics": run_diagnostics_stage,
    "report": run_report_stage,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume:
        print("本版不实现 resume；请用新 run-id 重跑（协议 §12）", file=sys.stderr)
        return EXIT_USAGE
    handler = STAGE_HANDLERS.get(args.stage)
    if handler is None:
        print(f"阶段 {args.stage} 尚未实现", file=sys.stderr)
        return EXIT_USAGE
    try:
        result = handler(args)
    except (StageError, artifacts.ArtifactError) as exc:
        print(f"阶段阻断：{exc}", file=sys.stderr)
        return EXIT_BLOCKED
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    if result.get("status") == "blocked":
        return EXIT_BLOCKED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
