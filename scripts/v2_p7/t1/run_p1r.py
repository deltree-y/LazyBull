# -*- coding: utf-8 -*-
"""P-1 重开入口：构建 L1 → 完整路径包络 → 抽样 → 两层校准 → O1 → 判定 → 原子发布。

用法::

    python -m scripts.v2_p7.t1.run_p1r --run-id p7_p1r_20261113_r1
    python -m scripts.v2_p7.t1.run_p1r --run-id p7_p1r_20261113_smoke \
        --smoke-days 30 --replicates 200 --skip-calibration

全量运行会写新证据根（已存在则拒绝覆盖）；smoke 运行标 ``diagnostic_only``，
不构成任何结论。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.v2_p7 import event_builder  # noqa: E402
from scripts.v2_p7.t1 import dataplane, p1_artifacts, p1_bounds, p1_calibration  # noqa: E402
from scripts.v2_p7.t1.p1_inputs import (  # noqa: E402
    BenchmarkBook,
    build_industry_members,
    build_intents,
    load_extras,
)
from scripts.v2_p7.t1.p1_o1 import constraint_readings  # noqa: E402
from scripts.v2_p7.t1.p1_params import (  # noqa: E402
    DEFAULT_RUN_ID,
    P1RParams,
    build_params,
    load_prereg,
)


def _git_state() -> Dict[str, Any]:
    def run(args: Sequence[str]) -> str:
        completed = subprocess.run(
            ["git", "--no-pager", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout if completed.returncode == 0 else ""

    status = run(["status", "--porcelain"])
    dirty_files = [line[3:] for line in status.splitlines() if line.strip()]
    return {
        "git_head": run(["rev-parse", "HEAD"]).strip() or None,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:200],
        "dirty_file_count": len(dirty_files),
    }


def _script_identity() -> Dict[str, str]:
    files = sorted((ROOT / "scripts" / "v2_p7").rglob("*.py"))
    return {
        path.relative_to(ROOT).as_posix(): p1_artifacts.sha256_file(path)
        for path in files
        if path.is_file()
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P7 P-1 重开（只许否决的筛查门）")
    parser.add_argument("--run-id", default="p7_p1r_20261113_r1")
    parser.add_argument("--smoke-days", type=int, default=0, help="只取前 N 个有意图的信号日")
    parser.add_argument("--smoke-events", type=int, default=0, help="只取前 N 个意图")
    parser.add_argument("--replicates", type=int, default=0, help="覆盖 bootstrap 次数")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-o1", action="store_true")
    parser.add_argument(
        "--benchmark-mode",
        choices=("legacy", "fixed"),
        default="legacy",
        help="基准口径：legacy=r6 原口径；fixed=资格/价源时钟/固定份额修正",
    )
    parser.add_argument(
        "--calendar-mode",
        choices=("legacy", "next_open"),
        default="legacy",
        help="公告日历：legacy=非交易日直接扣除；next_open=顺延到首个交易日",
    )
    parser.add_argument(
        "--fix-prereg",
        default="data/frozen_reference/v2_p7/p7_p1r2_20261114_r1/prereg/p7_p1r2_prereg_v1.json",
        help="修正批次 prereg JSON 路径",
    )
    return parser.parse_args(argv)


def _prepare(args: argparse.Namespace, params: P1RParams) -> Dict[str, Any]:
    print("[1/6] 事件流与面板", flush=True)
    raw_root = ROOT / "data" / "raw"
    events, event_stats = event_builder.build_events(raw_root, cutoff=params.common_data_cutoff)
    days = dataplane.load_trading_days(ROOT / "data", params.common_data_cutoff)
    first_clean = dataplane.first_clean_daily_date(ROOT / "data")
    days = [day for day in days if day >= first_clean]
    panel = dataplane.load_panel(
        ROOT / "data",
        days,
        cutoff=params.common_data_cutoff,
        announcements=None,
        lookback=20,
        industry_source="pit",
    )
    cutoff_pos = panel.n_days - 1
    print(f"    面板 {panel.n_days} 日 × {panel.n_stocks} 股", flush=True)

    print("[2/6] 涨跌停/停牌证据与 L1 意图", flush=True)
    extras = load_extras(raw_root, panel, cutoff=params.common_data_cutoff)
    intents = build_intents(
        events,
        panel,
        evaluation_start=params.evaluation_start,
        horizon=params.horizon_primary,
        entry_max_defer=params.entry_max_defer,
        cutoff_pos=cutoff_pos,
        calendar_mode=str(args.calendar_mode),
    )
    if args.smoke_days > 0 and intents.size:
        keep_days = sorted(set(intents.signal_pos.tolist()))[: args.smoke_days]
        keep = np.isin(intents.signal_pos, keep_days)
        intents = type(intents)(
            stock=intents.stock[keep],
            signal_pos=intents.signal_pos[keep],
            source_code=intents.source_code[keep],
            fingerprint=[fp for fp, flag in zip(intents.fingerprint, keep) if flag],
            knowledge=[kd for kd, flag in zip(intents.knowledge, keep) if flag],
            stats=intents.stats,
        )
    if args.smoke_events > 0 and intents.size:
        keep = np.zeros(intents.size, dtype=bool)
        keep[: args.smoke_events] = True
        intents = type(intents)(
            stock=intents.stock[keep],
            signal_pos=intents.signal_pos[keep],
            source_code=intents.source_code[keep],
            fingerprint=[fp for fp, flag in zip(intents.fingerprint, keep) if flag],
            knowledge=[kd for kd, flag in zip(intents.knowledge, keep) if flag],
            stats=intents.stats,
        )
    print(f"    L1 意图 {intents.size}", flush=True)

    print("[3/6] 行业成员簿与基准", flush=True)
    start_pos = panel.day_position(params.evaluation_start)
    members = build_industry_members(
        panel,
        start_pos=start_pos,
        require_listed=str(args.benchmark_mode) == "fixed",
    )
    benchmark = BenchmarkBook.build(panel, members, mode=str(args.benchmark_mode))
    return {
        "events": events,
        "event_stats": event_stats,
        "panel": panel,
        "extras": extras,
        "intents": intents,
        "benchmark": benchmark,
        "cutoff_pos": cutoff_pos,
    }


def _target_state(target: Path) -> str:
    """证据根状态：absent / prereg_only（预登记阶段）/ occupied（禁止覆盖）。"""
    if not target.exists():
        return "absent"
    entries = {path.name for path in target.iterdir()}
    return "prereg_only" if entries == {"prereg"} else "occupied"


def _publish_target(partial: Path, target: Path, state: str) -> None:
    """发布：空根直接 rename；仅含 prereg 的根先暂存再原子切换。"""
    if state == "absent":
        p1_artifacts.publish(partial, target)
        return
    if state != "prereg_only":
        raise FileExistsError(f"证据根已有产物（禁止覆盖）：{target}")
    spare = target.with_name(f"{target.name}.prereg-spare")
    if spare.exists():
        shutil.rmtree(spare, ignore_errors=True)
    target.rename(spare)
    try:
        partial.rename(target)
    except Exception:
        spare.rename(target)
        raise
    shutil.rmtree(spare, ignore_errors=True)


def _write_runbook(params: P1RParams, run_id: str, verdict: Dict[str, Any]) -> str:
    lines = [
        f"# P-1 重开 runbook（{run_id}）",
        "",
        f"- 预登记：`{params.prereg_path.as_posix()}`（SHA `{params.prereg_sha256}`）",
        f"- 判据摘要：`{params.criteria_digest}`",
        f"- 评价窗口：`{params.evaluation_start}` ~ `{params.evaluation_end}`"
        f"（{params.evaluation_trading_days} 交易日）",
        f"- 成本：{params.cost.as_dict()}；`min_ticket={params.min_ticket}`",
        f"- 判定：`{verdict.get('verdict')}`（{verdict.get('reason_code')}）",
        "- 复算：`verify.json`（逐文件 SHA 全等）",
        "",
        "边界：只回答 B0 是否值得进 T2/T3 验证；不回答能否赚钱；",
        "不构成 B0 复活、B 区转正、生产多袖、前向/日历验证。",
    ]
    return "\n".join(lines) + "\n"


def _reconciliation_payload(
    frame: pd.DataFrame, intent_stats: Dict[str, Any], path_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """事实对账：L1 扣除、成交×持仓交叉、守恒与待复核项。"""
    cross = pd.crosstab(frame["fill"], frame["holding"])
    return {
        "schema": "p7_p1r_reconciliation/v1",
        "deductions": intent_stats.get("deductions"),
        "sources": intent_stats.get("sources"),
        "fill_holding_crosstab": {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in cross.iterrows()
        },
        "conservation": {
            "intent_total": path_stats.get("intent_total"),
            "filled": path_stats.get("filled"),
            "filled_closed": path_stats.get("filled_closed"),
            "filled_open": path_stats.get("filled_open"),
            "filled_na": path_stats.get("filled_na"),
            "proven_not_filled": path_stats.get("proven_not_filled"),
            "unknown_not_filled": path_stats.get("unknown_not_filled"),
            "holding_na": path_stats.get("holding_na"),
            "ok": path_stats.get("conservation_ok"),
        },
        "checks": [
            "未成交（proven/unknown）的 holding 必须为 NA，不得伪作 CLOSED/OPEN",
            "bounded 但路径不可算（unbounded）的已成交记录计入 filled_na",
            "date_not_in_calendar_recoverable 为非交易日公告顺延到下一交易日的诊断计数",
            "code_not_in_panel_samples 为不在面板的代码样例，需核实退市/未上市/代码变更",
        ],
        "note": "本表只对账；不改域、不改判据",
    }


def _attribution_payload(frame: pd.DataFrame) -> Dict[str, Any]:
    """经济归因：逐意图两腿 -> 按源与年份的均值（收益单位，×10000 = bp）。"""
    subset = frame[
        (frame["fill"] == 0) & frame["event_gross"].notna() & frame["bench_gross"].notna()
    ].copy()
    subset["year"] = subset["signal_date"].str.slice(0, 4)

    def summary(group: pd.DataFrame) -> Dict[str, Any]:
        return {
            "count": int(len(group)),
            "mean_residual": float(group["d_point"].mean()),
            "mean_event_gross": float(group["event_gross"].mean()),
            "mean_event_net": float(group["event_net"].mean()),
            "mean_bench_gross": float(group["bench_gross"].mean()),
            "mean_buy_cost": float(group["buy_cost"].mean()),
            "mean_sell_cost": float(group["sell_cost"].mean()),
            "mean_bench_members": float(group["bench_members"].mean()),
            "mean_bench_unvalued": float(group["bench_unvalued"].mean()),
            "mean_bench_fallback": float(group["bench_fallback"].mean()),
        }

    by_source = {
        name: summary(subset[subset["source_code"] == code])
        for code, name in enumerate(("forecast", "express", "fina_report"))
    }
    by_year = {str(year): summary(group) for year, group in subset.groupby("year")}
    return {
        "schema": "p7_p1r_attribution/v1",
        "basis": "FILLED 且两腿可算；收益单位（×10000 = bp）",
        "by_source": by_source,
        "by_year": by_year,
        "note": "只解释来源，不据此删源/选年/搜持期",
    }


def _members_sample(
    prepared: Dict[str, Any], paths: Any, intents: Any, *, limit: int = 500
) -> pd.DataFrame:
    """抽样 confirmed 意图的成员级明细（身份/时点/价源/权重/回退/收益）。"""
    panel = prepared["panel"]
    benchmark = prepared["benchmark"]
    industry_matrix = np.asarray(panel.industry_matrix)
    candidates = [
        index
        for index in range(intents.size)
        if int(paths.fill[index]) == p1_bounds.FILL_CONFIRMED
        and paths.entry_pos[index] >= 0
        and paths.exit_pos[index] >= 0
    ][: int(limit)]
    frames = []
    for index in candidates:
        signal = int(intents.signal_pos[index])
        stock = int(intents.stock[index])
        detail = benchmark.member_detail(
            signal,
            int(industry_matrix[signal, stock]),
            stock,
            int(paths.entry_pos[index]),
            int(paths.exit_pos[index]),
            exit_mode=("close" if int(paths.holding[index]) == p1_bounds.HOLDING_OPEN else "open"),
        )
        if len(detail):
            detail["intent_index"] = index
            detail["source_code"] = int(intents.source_code[index])
            frames.append(detail)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_manifest(
    args: argparse.Namespace,
    params: P1RParams,
    identity: Dict[str, Any],
    elapsed: float,
    smoke: Dict[str, Any],
    written: Dict[str, str],
    partial: Path,
) -> None:
    """写 manifest（含修正批次 prereg 身份与模式）。"""
    manifest = p1_artifacts.manifest_payload(
        run_id=args.run_id,
        params=params,
        code_state=identity["code_state"],
        data_state=identity["data_state"],
        written=written,
        elapsed_seconds=elapsed,
        smoke=smoke,
        prereg=identity["prereg"],
    )
    manifest["benchmark_mode"] = str(args.benchmark_mode)
    manifest["calendar_mode"] = str(args.calendar_mode)
    fix_path = ROOT / str(args.fix_prereg)
    if fix_path.exists():
        fix_payload = json.loads(fix_path.read_text(encoding="utf-8"))
        manifest["fix_prereg"] = {
            "path": fix_path.relative_to(ROOT).as_posix(),
            "sha256": p1_artifacts.sha256_file(fix_path),
            "criteria_digest": fix_payload.get("criteria_digest"),
        }
    written["manifest.json"] = p1_artifacts.write_json(partial / "manifest.json", manifest)


def _write_outputs(
    args: argparse.Namespace,
    params: P1RParams,
    prepared: Dict[str, Any],
    paths: Any,
    sampling: Dict[str, Any],
    layer1: Dict[str, Any],
    pilot: Dict[str, Any],
    layer2: Dict[str, Any],
    o1: Dict[str, Any],
    verdict: Dict[str, Any],
    identity: Dict[str, Any],
    elapsed: float,
    partial: Path,
) -> Dict[str, str]:
    written: Dict[str, str] = {}
    panel = prepared["panel"]
    intents = prepared["intents"]
    prereg_dest = partial / "prereg" / params.prereg_path.name
    prereg_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(params.prereg_path, prereg_dest)
    written[f"prereg/{params.prereg_path.name}"] = p1_artifacts.sha256_file(prereg_dest)
    smoke = {
        "limit_signal_days": int(args.smoke_days),
        "limit_events": int(args.smoke_events),
        "replicates_override": int(args.replicates),
        "skip_calibration": bool(args.skip_calibration),
        "benchmark_mode": str(args.benchmark_mode),
        "calendar_mode": str(args.calendar_mode),
        "diagnostic_only": bool(
            args.smoke_days or args.smoke_events or args.replicates or args.skip_calibration
        ),
    }
    written["params.json"] = p1_artifacts.write_json(
        partial / "params.json", params.params_payload()
    )
    signal_days = [panel.days[int(pos)] for pos in intents.signal_pos]
    frame = pd.DataFrame(
        {
            "stock": intents.stock,
            "signal_pos": intents.signal_pos,
            "signal_date": signal_days,
            "knowledge_date": intents.knowledge,
            "source_code": intents.source_code,
            "d_point": paths.d_point,
            "lo": paths.lo,
            "hi": paths.hi,
            "unbounded": paths.unbounded,
            "holding": paths.holding,
            "fill": paths.fill,
            "value": paths.value,
            "coverage": paths.coverage,
            "entry_pos": paths.entry_pos,
            "exit_pos": paths.exit_pos,
            "entry_defer": paths.entry_defer,
            "exit_defer": paths.exit_defer,
            "paths": paths.paths_per_intent,
            "event_gross": paths.event_gross,
            "event_net": paths.event_net,
            "bench_gross": paths.bench_gross,
            "buy_cost": paths.buy_cost,
            "sell_cost": paths.sell_cost,
            "bench_members": paths.bench_members,
            "bench_unvalued": paths.bench_unvalued,
            "bench_fallback": paths.bench_fallback,
        }
    )
    intents_path = partial / "intents/intents.parquet"
    intents_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(intents_path, index=False, compression="zstd")
    written["intents/intents.parquet"] = p1_artifacts.sha256_file(intents_path)
    written["intents/summary.json"] = p1_artifacts.write_json(
        partial / "intents/summary.json",
        {"stats": intents.stats, "paths": paths.stats},
    )
    written["reconciliation.json"] = p1_artifacts.write_json(
        partial / "reconciliation.json",
        _reconciliation_payload(frame, intents.stats, paths.stats),
    )
    written["attribution/summary.json"] = p1_artifacts.write_json(
        partial / "attribution/summary.json", _attribution_payload(frame)
    )
    sample = _members_sample(prepared, paths, intents, limit=500)
    sample_path = partial / "attribution/members_sample.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(sample_path, index=False, compression="zstd")
    written["attribution/members_sample.parquet"] = p1_artifacts.sha256_file(sample_path)
    noise = (
        {"skipped": True, "note": "smoke 运行不产出噪声保真度表"}
        if pilot.get("skipped")
        else p1_calibration.noise_fidelity(
            pilot,
            intents.signal_pos + 1,
            panel.n_days,
            sampling,
            layer2,
        )
    )
    written["noise_fidelity.json"] = p1_artifacts.write_json(partial / "noise_fidelity.json", noise)
    written["bounds/envelope.json"] = p1_artifacts.write_json(
        partial / "bounds/envelope.json", p1_bounds.aggregate_envelope(paths)
    )
    sampling_payload = {key: value for key, value in sampling.items() if key not in ("records",)}
    written["sampling/envelope.json"] = p1_artifacts.write_json(
        partial / "sampling/envelope.json", sampling_payload
    )
    written["calibration/layer1.json"] = p1_artifacts.write_json(
        partial / "calibration/layer1.json", layer1
    )
    written["calibration/pilot.json"] = p1_artifacts.write_json(
        partial / "calibration/pilot.json", pilot
    )
    layer2_payload = {key: value for key, value in layer2.items() if key != "results"}
    layer2_payload["results"] = [
        {key: value for key, value in item.items() if key != "records"}
        for item in layer2.get("results", [])
    ]
    written["calibration/layer2.json"] = p1_artifacts.write_json(
        partial / "calibration/layer2.json", layer2_payload
    )
    written["o1/readings.json"] = p1_artifacts.write_json(partial / "o1/readings.json", o1)
    written["verdict.json"] = p1_artifacts.write_json(partial / "verdict.json", verdict)
    written["diagnostics.json"] = p1_artifacts.write_json(
        partial / "diagnostics.json",
        {
            "event_stats": prepared["event_stats"],
            "extras": prepared["extras"].stats,
            "panel_load": panel.load_stats,
            "industry_source": panel.industry_source,
            "intents": intents.stats,
            "paths": paths.stats,
            "sampling": {
                "invalid_rate": sampling.get("invalid_rate"),
                "method_sensitive": sampling.get("method_sensitive"),
            },
        },
    )
    written["runbook.md"] = p1_artifacts.write_text(
        partial / "runbook.md", _write_runbook(params, args.run_id, verdict)
    )
    _write_manifest(args, params, identity, elapsed, smoke, written, partial)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    params = build_params(ROOT, run_id=args.run_id)
    prereg_payload, _ = load_prereg(ROOT, params.prereg_path)
    target = (
        params.evidence_root
        if args.run_id == DEFAULT_RUN_ID
        else ROOT / "data/frozen_reference/v2_p7" / args.run_id
    )
    state = _target_state(target)
    if state == "occupied":
        print(f"证据根已存在产物（禁止覆盖）：{target}")
        return 2
    partial = target.with_name(f"{target.name}.partial")
    if partial.exists():
        shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True, exist_ok=False)

    identity = {
        "code_state": _git_state(),
        "scripts": _script_identity(),
        "prereg": prereg_payload,
        "data_state": {
            "common_data_cutoff": params.common_data_cutoff,
            "evaluation_window": [params.evaluation_start, params.evaluation_end],
            "watermarks": prereg_payload["data_identity"]["watermarks"],
        },
    }
    try:
        prepared = _prepare(args, params)
        panel = prepared["panel"]
        intents = prepared["intents"]
        print("[4/6] 完整路径枚举与包络", flush=True)
        paths = p1_bounds.evaluate_intents(
            panel, prepared["extras"], prepared["benchmark"], params, intents
        )
        print(f"    路径完成：{paths.stats}", flush=True)
        replicates = int(args.replicates or params.bootstrap_replicates)
        sampling = p1_bounds.sampling_envelope(
            paths, intents, params, n_days=panel.n_days, replicates=replicates
        )
        print("[5/6] 校准与 O1", flush=True)
        frozen = prereg_payload["calibration"]["layer1_mechanical"]["M1"]["frozen_values"]
        layer1 = p1_calibration.layer1_mechanical(params, frozen)
        bounded = np.isfinite(paths.d_point) & (~paths.unbounded)
        if args.skip_calibration:
            pilot = {"skipped": True}
            layer2 = {
                "skipped": True,
                "passed": False,
                "negative_power": 0.0,
                "failure_consequence": "smoke 运行不产出结论",
            }
        else:
            pilot = p1_calibration.estimate_pilot(
                paths.d_point[bounded],
                intents.signal_pos[bounded] + 1,
                panel.n_days,
            )
            layer2 = p1_calibration.layer2_statistical(
                pilot,
                intents.signal_pos[bounded] + 1,
                panel.n_days,
                params,
                samples=params.calibration_samples,
                replicates=params.calibration_replicates,
            )
        passed = p1_calibration.calibration_passed(layer1, layer2)
        o1 = (
            {"skipped": True}
            if args.skip_o1
            else constraint_readings(intents, paths, params, cutoff_pos=panel.n_days - 1)
        )
        verdict = p1_bounds.decide_verdict(
            calibration_passed=passed,
            method_sensitive=bool(sampling["method_sensitive"]),
            unbounded=bool(paths.unbounded.any()),
            no_valid_interval=bool(sampling["no_valid_interval"]),
            final_lo=sampling["final_lo"],
            final_hi=sampling["final_hi"],
            negative_power=float(layer2.get("negative_power") or 0.0),
            intent_total=int(intents.size),
        )
        verdict.update(
            {
                "schema": "p7_p1r_verdict/v1",
                "run_id": args.run_id,
                "prereg_id": params.prereg_id,
                "prereg_sha256": params.prereg_sha256,
                "criteria_digest": params.criteria_digest,
                "alpha_consumed": False,
                "negative_power": layer2.get("negative_power"),
                "calibration_passed": passed,
                "final_envelope": [sampling["final_lo"], sampling["final_hi"]],
                "counts": paths.stats,
                "diagnostic_only": bool(
                    args.smoke_days or args.smoke_events or args.replicates or args.skip_calibration
                ),
                "boundaries": prereg_payload["boundaries"],
            }
        )
        if verdict["diagnostic_only"]:
            verdict["verdict"] = None
            verdict["reason_code"] = "diagnostic_only"
            verdict["reason"] = "冒烟/降配运行，不构成任何结论"

        elapsed = time.time() - started
        _write_outputs(
            args,
            params,
            prepared,
            paths,
            sampling,
            layer1,
            pilot,
            layer2,
            o1,
            verdict,
            identity,
            elapsed,
            partial,
        )
        _publish_target(partial, target, state)
        manifest_payload = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        verify = p1_artifacts.verify_payload(target, manifest_payload)
        p1_artifacts.write_json(target / "verify.json", verify)
        print(
            f"[6/6] 已发布 {target}（{len(manifest_payload['written_files_sha256'])} 件，"
            f"{elapsed:.1f}s，verify_all_equal={verify['all_equal']}）",
            flush=True,
        )
        print(f"verdict={verdict.get('verdict')} reason={verdict.get('reason_code')}")
        return 0
    except Exception:
        p1_artifacts.cleanup_partial(partial)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
