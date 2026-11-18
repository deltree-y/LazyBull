"""DEF-ML-60-v1 统一 CLI（薄入口）。

只开放执行控制项：campaign / stage / run-id / resume / 内容哈希开关。
规格从封存的机器件加载，不提供任何可切换的特征、分位、域、权重或持有期参数。

用法（协议 §10.3）::

    python -m scripts.v2_p7.defensive_ml.run --campaign defml60_v1_20261117 --stage readiness

复杂度豁免：阶段专用研究工具，不入生产链路（CLAUDE §7.5）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from scripts.v2_p7.common import REPO_ROOT, sha256_file
from scripts.v2_p7.defensive_ml import (
    artifacts,
    b_native,
    dataset,
    diagnostics,
    pipeline,
    readiness,
    replay_stage,
    report,
    spec,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 3

#: 已实现阶段
IMPLEMENTED_STAGES: tuple[str, ...] = (
    "readiness",
    "dataset",
    "model_probe",
    "train_predict",
    "replay",
    "diagnostics",
    "report",
)

#: 本 campaign 尚未实现的阶段（一律 fail-closed，不伪造完成态）
UNIMPLEMENTED_STAGES: tuple[str, ...] = tuple(
    stage for stage in spec.STAGES if stage not in IMPLEMENTED_STAGES
)


def build_parser() -> argparse.ArgumentParser:
    """CLI 参数（闭集：不含任何规格参数）。"""
    parser = argparse.ArgumentParser(
        prog="python -m scripts.v2_p7.defensive_ml.run",
        description="DEF-ML-60-v1 研究包执行入口（研究工具，不接生产）",
    )
    parser.add_argument(
        "--campaign", default=spec.DEFAULT_CAMPAIGN, help="campaign id（新规格须新 campaign）"
    )
    parser.add_argument("--stage", required=True, choices=list(spec.STAGES), help="阶段")
    parser.add_argument(
        "--run-id", default=None, help="显式 run id（默认按 UTC 时间生成，已存在即拒绝）"
    )
    parser.add_argument(
        "--resume", action="store_true", help="复用身份一致的完整片段（本阶段暂不适用）"
    )
    parser.add_argument(
        "--machine-hours",
        type=float,
        default=0.0,
        help="报告阶段登记的累计机器墙钟（小时，用于预算核对）",
    )
    parser.add_argument("--dataset-run", default="", help="上游 dataset run id（发布件）")
    parser.add_argument("--predictions-run", default="", help="上游 train_predict run id")
    parser.add_argument("--replay-run", default="", help="上游 replay run id（账户）")
    parser.add_argument("--diagnostics-run", default="", help="上游 diagnostics run id")
    parser.add_argument("--train-run", default="", help="上游 train_predict run id（报告登记用）")
    parser.add_argument(
        "--no-hash-content",
        action="store_true",
        help="跳过整库内容哈希（只登记规模与水位；正式 R0 封存不建议关闭）",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="仓库根（测试用隔离根）")
    return parser


def run_dataset(args: argparse.Namespace) -> int:
    """R1：重建母表 + 19 列特征 + 60 日标签，落分年缓存。"""
    camp = artifacts.campaign_root(args.campaign)
    layout = artifacts.ensure_layout(camp)
    rid = args.run_id or artifacts.new_run_id("dataset")
    run_dir = artifacts.create_run_dir(camp, rid)
    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(layout["inputs"]) / "source_manifest.json"
    if not manifest_path.exists():
        print("refused: run readiness first (inputs/source_manifest.json missing)")
        return EXIT_USAGE
    manifest_sha = sha256_file(manifest_path)
    print(f"campaign={camp} run={rid} source_manifest_sha={manifest_sha[:12]}")
    result = dataset.build_dataset(
        Path(args.repo_root),
        feature_start=str(spec.WINDOW["first_train_signal_start"]),
        end=str(spec.WINDOW["end"]),
        out_dir=work / "dataset",
        source_manifest_sha=manifest_sha,
        log=lambda msg: print(msg, flush=True),
    )
    payload = {
        "stage": "dataset",
        "summary": result["summary"],
        "written": result["written"],
        "source_manifest_sha256": manifest_sha,
        "spec_sha256": spec.spec_sha256(),
    }
    pub = artifacts.publish_artifacts(
        work, run_dir, "dataset", artifacts.STAGE_ARTIFACTS["dataset"],
        {"readiness_source_manifest_sha256": manifest_sha,
         "protocol_sha256": _protocol_sha(args), "spec_sha256": spec.spec_sha256()},
    )
    written = artifacts.write_json_checked(run_dir / "dataset.json", payload)
    seal = artifacts.seal_stage(
        run_dir, "dataset", {"dataset_sha256": written["sha256"], "artifacts": pub}
    )
    print(f"rows={result['summary']['rows']} days={result['summary']['days']}")
    print(f"dataset={written['path']}")
    print(f"seal_manifest_sha256={seal['manifest_sha256']}")
    return EXIT_OK


def _campaign_dirs(args: argparse.Namespace) -> tuple:
    camp = artifacts.campaign_root(args.campaign)
    layout = artifacts.ensure_layout(camp)
    return Path(args.repo_root), Path(camp), layout


def _protocol_sha(args: argparse.Namespace) -> str:
    try:
        return spec.protocol_sha256(Path(args.repo_root))
    except FileNotFoundError:
        return "not_available"


def run_probe(args: argparse.Namespace) -> int:
    """R2 首折性能探针（不写注册表、不产生收益读数）。"""
    repo, camp, run_dir = _stage_workspace(
        args, args.run_id or artifacts.new_run_id("model_probe"),
        {"dataset": args.dataset_run},
    )
    work = run_dir / "work"
    result = pipeline.run_model_probe(repo, work, log=lambda msg: print(msg, flush=True))
    written = artifacts.write_json_checked(run_dir / "probe.json", result)
    seal = artifacts.seal_stage(run_dir, "model_probe", {"probe_sha256": written["sha256"]})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"seal_manifest_sha256={seal['manifest_sha256']}")
    return EXIT_OK


def _stage_workspace(
    args: argparse.Namespace, rid: str, inputs: dict[str, str]
) -> tuple[Path, Path, Path]:
    """建 run 工作区，并把上游**已发布**工件复制进来（逐文件 SHA 校验）。

    返回 (camp, run_dir, work)。下游只读 work，绝不直接读可变共享目录。
    """
    repo, camp, _ = _campaign_dirs(args)
    run_dir = artifacts.create_run_dir(camp, rid)
    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    for name, upstream in inputs.items():
        if not upstream:
            raise SystemExit(f"refused: 需要 --{name}-run 指定上游发布件")
        source = artifacts.require_published(camp, upstream, name)
        shutil.copytree(source, work / name)
    return Path(repo), Path(camp), run_dir


def run_train_predict(args: argparse.Namespace) -> int:
    """R2 全折训练 + 注册 + OOS 全域预测。"""
    repo, camp, run_dir = _stage_workspace(
        args, args.run_id or artifacts.new_run_id("train_predict"),
        {"dataset": args.dataset_run},
    )
    work = run_dir / "work"
    summary = pipeline.run_train_predict(
        repo, work, _protocol_sha(args), log=lambda msg: print(msg, flush=True)
    )
    payload = {
        "stage": "train_predict",
        "spec_sha256": spec.spec_sha256(),
        "protocol_sha256": _protocol_sha(args),
        "summary": summary,
    }
    written = artifacts.write_json_checked(run_dir / "train_predict.json", payload)
    pipeline.fold_manifest(work, payload)
    pub = artifacts.publish_artifacts(
        work, run_dir, "train_predict", artifacts.STAGE_ARTIFACTS["train_predict"],
        {"dataset_run": args.dataset_run, "spec_sha256": spec.spec_sha256()},
    )
    seal = artifacts.seal_stage(
        run_dir, "train_predict",
        {"result_sha256": written["sha256"], "artifacts": pub},
    )
    print(f"folds={summary['folds_run']} models={summary['models_registered']} "
          f"rows={summary['rows']} crossed={summary['crossed_rate']:.4f}")
    print(f"seal_manifest_sha256={seal['manifest_sha256']}")
    return EXIT_OK if summary["folds_run"] else EXIT_BLOCKED


def run_replay_stage(args: argparse.Namespace) -> int:
    """R3/R4：跑 DEF 与 LOWVOL 两袖，四臂合并并落盘真实现金账。"""
    repo, camp, run_dir = _stage_workspace(
        args, args.run_id or artifacts.new_run_id("replay"),
        {"dataset": args.dataset_run, "predictions": args.predictions_run},
    )
    work = run_dir / "work"
    summary: dict[str, Any] = {"sleeves": {}}
    for sleeve in ("def", "lowvol"):
        result = replay_stage.run_replay_stage(
            repo, work, log=lambda msg: print(msg, flush=True), sleeve_kind=sleeve
        )
        for capital in spec.TARGETS["capital_scenarios"]:
            info = replay_stage.write_accounts(work, capital, sleeve.upper(), result)
            summary["sleeves"].setdefault(sleeve, {})[str(capital)] = {
                "rows": info["rows"],
                "trades": info["trades"],
                "ending": info["ending"],
                "stats": info["diagnostics"].get("stats"),
            }
    written = artifacts.write_json_checked(run_dir / "replay.json", summary)
    pub = artifacts.publish_artifacts(
        work, run_dir, "replay", artifacts.STAGE_ARTIFACTS["replay"],
        {"dataset_run": args.dataset_run, "predictions_run": args.predictions_run},
    )
    seal = artifacts.seal_stage(
        run_dir, "replay", {"replay_sha256": written["sha256"], "artifacts": pub}
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"seal_manifest_sha256={seal['manifest_sha256']}")
    return EXIT_OK


def run_diagnostics_stage(args: argparse.Namespace) -> int:
    """R5：四臂读数、目标核对、分段、自举、相关与四个固定压力。"""
    repo, camp, run_dir = _stage_workspace(
        args, args.run_id or artifacts.new_run_id("diagnostics"),
        {
            "dataset": args.dataset_run,
            "predictions": args.predictions_run,
            "accounts": args.replay_run,
        },
    )
    work = run_dir / "work"
    result = diagnostics.run_diagnostics(
        repo, work, log=lambda msg: print(msg, flush=True)
    )
    written = artifacts.write_json_checked(run_dir / "diagnostics.json", result)
    artifacts.write_json_checked(work / "diagnostics" / "summary.json", result)
    pub = artifacts.publish_artifacts(
        work, run_dir, "diagnostics", artifacts.STAGE_ARTIFACTS["diagnostics"],
        {
            "dataset_run": args.dataset_run,
            "predictions_run": args.predictions_run,
            "replay_run": args.replay_run,
        },
    )
    seal = artifacts.seal_stage(
        run_dir, "diagnostics",
        {"diagnostics_sha256": written["sha256"], "artifacts": pub},
    )
    compact = {
        str(capital): {
            "arms": {row["arm"]: round(float(row["cagr"]), 4) for row in data["arms"]},
            "target": data["target_vs_a100"],
        }
        for capital, data in result["main"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, default=float))
    print(f"seal_manifest_sha256={seal['manifest_sha256']}")
    return EXIT_OK


REUSE_NOTES: tuple[str, ...] = (
    "到期判定复用 trading/sell_rules.is_holding_period_exit_due（hp−1 出信号、T+1 开盘执行，"
    "实际持有 60 个交易日），未自建阈值。",
    "未复用 OperatingBaselineEngine：其 engine_kwargs 硬绑 A 的 20 日/两批/half_kelly，"
    "且 buffer 分支断言 top_n==20/stagger==2/holding_period==20；按协议 §10.2 另写薄原生"
    "账户（b_account/b_replay），等权走 position_sizing 语义而非改排名。",
    "未复用 validate_account：该函数签名要求引擎 result payload；本包账户自带资产恒等式、"
    "现金/股数/应收三条守恒与 10 项合成执行测试（tests/test_p7_defensive_r3.py）。",
    "公司行为复用 NativeCorporateBook（现金应收、送转、锁股）；到账按事件 pay_trade_day 结算，"
    "不依赖桥的 pay_events 同日返回。",
)

GAPS: tuple[str, ...] = (
    "F29 型 II 完整门的校准/选择/独立验证未实现（f2 入口仍拒绝未就绪的 risk_adjusted）",
    "研究线、候选池分母与 α 预算未在正式台账登记（本报告 alpha_consumed=false）",
    "2015/2018 冻结危机窗的 B 账户不可构造（R0 已登记为正式准入障碍）",
    "三口径相关（全样本/调仓频/崩盘段）区间上界 <0.3 未做正式实现，仅有日收益点估计",
    "真实多袖订单/净额/费用分摊与三恒等式复核未做（本包为两条独立现金腿相加）",
    "准入裁决、kill criteria 与回滚方案须由用户决定",
)


def _prediction_meta(work: Path, camp: Path, train_run: str | None) -> dict[str, Any]:
    path = work / "predictions" / "predictions_all.parquet"
    if not path.exists():
        return {}
    frame = __import__("pandas").read_parquet(path, columns=["trade_date", "crossed"])
    reload_ok = None
    if train_run:
        train_path = camp / "runs" / train_run / "train_predict.json"
        if train_path.exists():
            payload = json.loads(train_path.read_text(encoding="utf-8"))
            reload_ok = payload["summary"].get("reload_all_identical")
    return {
        "rows": int(len(frame)),
        "days": int(frame["trade_date"].nunique()),
        "crossed_rate": float(frame["crossed"].mean()),
        "reload_all_identical": reload_ok,
    }


def run_report_stage(args: argparse.Namespace) -> int:
    """R6：资源决策表 + 中文报告。"""
    repo, camp, run_dir = _stage_workspace(
        args, args.run_id or artifacts.new_run_id("report"),
        {
            "accounts": args.replay_run,
            "diagnostics": args.diagnostics_run,
            "predictions": args.predictions_run,
        },
    )
    work = run_dir / "work"
    diagnostics = report.load_diagnostics(work)
    meta = {
        "campaign": args.campaign,
        "generated_at": artifacts.utc_now(),
        "protocol_sha256": _protocol_sha(args),
        "spec_sha256": spec.spec_sha256(),
        "prediction": _prediction_meta(work, camp, args.train_run),
        "sleeve_correlation": report.sleeve_correlations(work),
        "machine_hours": args.machine_hours,
        "over_budget": bool(args.machine_hours > 36.0),
        "gaps": list(GAPS),
        "deviations": [dict(item) for item in b_native.DEVIATIONS],
        "reuse_notes": list(REUSE_NOTES),
        "data_purchase_yuan": 0,
        "production_changed": False,
    }
    paths = report.write_report(work, diagnostics, meta)
    decision = report.decide(diagnostics)
    written = artifacts.write_json_checked(run_dir / "report.json", decision)
    pub = artifacts.publish_artifacts(
        work, run_dir, "report", artifacts.STAGE_ARTIFACTS["report"],
        {
            "replay_run": args.replay_run,
            "diagnostics_run": args.diagnostics_run,
            "predictions_run": args.predictions_run,
            "train_run": args.train_run,
        },
    )
    artifacts.seal_stage(
        run_dir, "report", {"decision_sha256": written["sha256"], "artifacts": pub}
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=float))
    print(f"report={paths['report']}")
    print(f"summary={paths['summary']}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """入口：返回进程退出码。"""
    args = build_parser().parse_args(argv)

    if args.stage in UNIMPLEMENTED_STAGES:
        if args.resume:
            print("refused: resume is not available for stage={}".format(args.stage))
            return EXIT_USAGE
        print("refused: stage={} not implemented in this campaign yet".format(args.stage))
        return EXIT_BLOCKED

    if args.resume:
        print("refused: stage=readiness does not accept --resume")
        return EXIT_USAGE

    if args.stage == "dataset":
        return run_dataset(args)
    if args.stage == "model_probe":
        return run_probe(args)
    if args.stage == "train_predict":
        return run_train_predict(args)
    if args.stage == "replay":
        return run_replay_stage(args)
    if args.stage == "diagnostics":
        return run_diagnostics_stage(args)
    if args.stage == "report":
        return run_report_stage(args)

    result = readiness.run_readiness(
        repo_root=Path(args.repo_root),
        campaign=args.campaign,
        run_id=args.run_id,
        hash_content=not args.no_hash_content,
        log=lambda msg: print(msg),
    )
    report = result["readiness"]
    print("stage=readiness status={} campaign={} run={}".format(
        report["status"], report["campaign"], report["run_id"]))
    print("spec_sha256={}".format(report["spec_sha256"]))
    print("protocol_sha256={}".format(report["protocol_sha256"]))
    print("readiness={}".format(result["readiness_path"]["path"]))
    print("seal_manifest_sha256={}".format(result["seal"]["manifest_sha256"]))
    for check in report["checks"]:
        print("check {:28s} {}".format(check["name"], check["status"]))
    return EXIT_OK if report["status"] != "blocked" else EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
