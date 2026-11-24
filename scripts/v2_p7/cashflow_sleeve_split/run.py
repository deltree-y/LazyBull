# -*- coding: utf-8 -*-
"""CF-SPLIT-20-v1 统一阶段 CLI（协议 §8）。

只开放 ``campaign / stage / run / parent`` 与写域白名单；不开放任何规格扫描参数
（改变规格 = 新 campaign）。**禁止无参数默认全跑**：没有 ``--stage`` 或子命令即
usage 错误。

用法（仓库根执行）::

    python -m scripts.v2_p7.cashflow_sleeve_split.run prepare ^
        --campaign cf_split_v1 --run r0_001

    # run-a 在审点一通过 + 用户命名批次授权后执行（本轮 = 用户 2026-11-20 指令）；
    # 内部按 rebuild → train-f → train-r → score → accounts → deliver 批次推进，
    # 各批可单独恢复（--batch）
    python -m scripts.v2_p7.cashflow_sleeve_split.run run-a ^
        --campaign cf_split_v1 --parent r0_008 --run a_001 [--batch rebuild]

``run-b / verify`` 仍为**显式拒绝桩**：只报告未授权与前置条件，不写任何收益/账户工件。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from scripts.v2_p7.cashflow_sleeve_split import (
    artifacts,
    budget,
    feature_routes,
    fold_clock,
    identity,
    probes,
    readiness,
    run_a,
    spec,
    structbuild,
)
from scripts.v2_p7.cashflow_sleeve_split.artifacts import ArtifactError
from scripts.v2_p7.common import REPO_ROOT, sha256_file, write_json

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 3

#: 定向测试对象（审点一 ``tests.txt`` 的唯一来源）
TEST_TARGETS: Sequence[str] = ("tests/test_p7_cashflow_sleeve_split.py",)

#: 单日验证缓存目录（campaign/cache/<spec_digest>/structural_single_<day>/）


def _single_day_cache_dir(camp: Path) -> Path:
    digest = spec.spec_sha256()[:16]
    return camp / "cache" / digest / f"structural_single_{spec.SINGLE_DAY_VALIDATION}"


def _single_day_validation(
    base: Path, camp: Path, routes_table: Mapping[int, Mapping[str, Any]], emit
) -> Dict[str, Any]:
    """单日端到端验证（r0_003 复核第 1、3 项；评审授权的「研究输入接线与单日验证」）。

    流程：T 日 PIT 行业快照构造 → 注入式结构捕获（写盘研究缓存 + RSS 采样）
    → 三遍标签反事实（原值 / 值改写 / 缺失位置重排）→ 探针消费真实输出。
    只跑 2019-04-01 一个验证日，不启动整窗。
    """
    day = spec.SINGLE_DAY_VALIDATION
    emit(f"    ① 构造 T 日 PIT 行业快照（{day}）")
    snapshot = structbuild.industry_pit_snapshot(day, base)
    identity_probe = snapshot["stats"]["identity_probe"]
    if not identity_probe.get("match"):
        raise StageError(f"行业注入身份探针不符：{identity_probe}")
    emit(
        f"    行业快照 {snapshot['stats']['codes']} 只 / 反查失败 "
        f"l1={snapshot['stats']['l1_unresolved']}；600783.SH → 非银金融 OK"
    )

    emit(f"    ② 注入式单日结构捕获（keep_dates={{{day}}}，含预热；RSS 采样 + 写盘）")
    # 预热窗与反事实同口径（90 自然日）：normal 遍结果复用给反事实，避免重复构建
    start = structbuild.window_start(day, 90)
    cache_dir = _single_day_cache_dir(camp)
    normal = structbuild.capture(
        base,
        start,
        day,
        keep_dates=[day],
        label_mode="normal",
        industry_frame=snapshot["frame"],
        industry_source="industry_pit_injected",
        write_dir=cache_dir,
        sample_rss=True,
    )
    write = normal.get("write") or {}
    emit(
        f"    捕获 {normal['elapsed_seconds']}s；写盘 {write.get('files')} 件 "
        f"{write.get('bytes')} B（{write.get('elapsed_seconds')}s）；峰值 RSS "
        f"{(normal.get('rss_sampled') or {}).get('peak_rss_sampled_mb')} MB"
    )

    emit("    ③ 三遍标签反事实（原值 / 值改写 / 缺失位置重排；normal 遍复用 ②）")
    cf = structbuild.counterfactual(
        base,
        days=[day],
        capture_days=90,
        routes=routes_table,
        version=_route_version_for_day(base),
        industry_frame=snapshot["frame"],
        industry_source="industry_pit_injected",
        write_dir=None,
        prebuilt_normal=normal,
    )
    if cf["status"] != "ok":
        raise StageError(
            f"标签反事实泄漏：value_only={cf['identical_value_only']} "
            f"position={cf['identical_position']}"
        )
    mask = cf.get("source_mask") or {}
    emit(
        f"    反事实 OK：value_only={cf['identical_value_only']} position={cf['identical_position']} "
        f"mask={mask.get('identical')}"
    )

    emit("    ④ 探针消费真实结构输出 + 旧域对照")
    emit("    （单日结构探针并入七日探针结果，见 6/8）")

    return {
        "day": day,
        "capture_window": [start, day],
        "industry_injection": {
            "stats": snapshot["stats"],
            "frame_codes_n": int(snapshot["frame"]["ts_code"].nunique()),
        },
        "capture": {
            "elapsed_seconds": normal["elapsed_seconds"],
            "rows": normal["rows"],
            "columns_n": normal["columns_n"],
            "write": write,
            "rss_sampled": normal.get("rss_sampled"),
            "industry_source": normal["industry_source"],
            "cache_path": str(cache_dir),
        },
        "counterfactual": cf,
        "structural_frame": normal["frames"][0][1] if normal["frames"] else None,
    }


def _refresh_measurements(
    base: Path,
    camp: Path,
    validation: Mapping[str, Any],
    single_probe: Mapping[str, Any],
    emit,
) -> Dict[str, Any]:
    """重算 57 日键数、修正测量附件并写单日端到端证据（r0_003 复核第 3 项）。

    - ``structural_block_hist_2019.json``：用已落盘 57 日文件逐日直读修正
      （逐日相加断言 + 预热天数按交易日历重算），原测量墙钟/字节保留并注明；
    - ``structural_counterfactual_hist_2019.json``：更新为三遍反事实件；
    - ``single_day_<day>.json``：单日端到端证据（行业注入 + 反事实 + 探针消费）。
    """
    measurements = camp / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)
    day_dir = REPO_ROOT / "temp" / "cfsplit_measure_hist_2019"
    summary: Dict[str, Any] = {"keys_recounted": False, "single_day_attachment": None}

    if day_dir.is_dir():
        corrected = structbuild.rebuild_block_measurement(
            measurements / "structural_block_hist_2019.json", day_dir, base
        )
        totals = corrected["key_control"]["totals"]
        if not corrected["key_control"]["sum_matches_total"]:
            raise StageError(
                f"逐日 only_structural 相加 {totals['only_structural']} != 全块总数 "
                f"{corrected['key_control']['previous_summary_only_structural']}"
            )
        write_json(measurements / "structural_block_hist_2019.json", corrected)
        emit(
            f"    键数重算 OK：{totals['days']} 日，only_structural 合计 "
            f"{totals['only_structural']}（与全块总数一致）；only_cs_train={totals['only_cs_train']}"
        )
        summary["keys_recounted"] = True
        summary["key_totals"] = totals
    else:
        emit(f"    [警告] 57 日结构逐日文件不在：{day_dir}（键数沿用 r0_003 封存件）")

    cf = dict(validation["counterfactual"])
    cf_payload = {
        "schema": "p7_cashflow_split_counterfactual/v2",
        "batch": f"{spec.DEFAULT_CAMPAIGN}/measure/single_day_{spec.SINGLE_DAY_VALIDATION}",
        "day": spec.SINGLE_DAY_VALIDATION,
        "row_domain": "label_filtered",
        "status": cf.get("status"),
        "identical_value_only": cf.get("identical_value_only"),
        "identical_position": cf.get("identical_position"),
        "identical": cf.get("identical"),
        "modes": cf.get("modes"),
        "signature_normal": cf.get("signature_normal"),
        "signature_label_value_perturbed": cf.get("signature_label_value_perturbed"),
        "signature_label_position_reshaped": cf.get("signature_label_position_reshaped"),
        "source_mask": cf.get("source_mask"),
        "known_label_targets": cf.get("known_label_targets"),
        "label_removal": cf.get("label_removal"),
        "elapsed_seconds": cf.get("elapsed_seconds"),
        "peak_rss_mb": cf.get("peak_rss_mb"),
        "rows": cf.get("rows"),
        "columns_n": cf.get("columns_n"),
        "industry_source": cf.get("industry_source"),
        "capture_window": cf.get("capture_window"),
        "provenance": {
            "command": (
                f"structbuild.counterfactual(days=['{spec.SINGLE_DAY_VALIDATION}'], "
                "capture_days=90, routes=v24008, industry_pit_injected)"
            ),
            "module": "scripts/v2_p7/cashflow_sleeve_split/structbuild.py",
            "note": "三遍构建（原值/值改写/缺失位置重排）；生成入口随 run 封存",
        },
    }
    cf_payload["content_digest"] = structbuild.measurement_content_digest(cf_payload)
    write_json(measurements / "structural_counterfactual_hist_2019.json", cf_payload)

    single_payload = {
        "schema": "p7_cashflow_split_single_day/v1",
        "batch": f"{spec.DEFAULT_CAMPAIGN}/measure/single_day_{spec.SINGLE_DAY_VALIDATION}",
        "day": spec.SINGLE_DAY_VALIDATION,
        "purpose": (
            "r0_003 复核第 1 项端到端：注入 T 日 PIT 行业 → 三遍标签反事实 → "
            "探针消费真实输出；第 3 项：RSS 采样与写盘单列"
        ),
        "industry_injection": validation["industry_injection"],
        "capture": validation["capture"],
        "counterfactual_summary": {
            key: value
            for key, value in validation["counterfactual"].items()
            if key not in ("frames",)
        },
        "probe_consumption": single_probe,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    single_payload["content_digest"] = structbuild.measurement_content_digest(single_payload)
    write_json(measurements / f"single_day_{spec.SINGLE_DAY_VALIDATION}.json", single_payload)
    summary["single_day_attachment"] = f"measurements/single_day_{spec.SINGLE_DAY_VALIDATION}.json"
    emit(f"    测量附件已修正：single_day={summary['single_day_attachment']}")
    return {"summary": summary, "single_day_payload": single_payload}


def _single_day_seal_payload(
    validation: Mapping[str, Any], corrected: Mapping[str, Any]
) -> Dict[str, Any]:
    """run 封存 manifest 里的单日验证摘要（r0_004 整改证据链入口）。"""
    cf = validation["counterfactual"] or {}
    return {
        "day": spec.SINGLE_DAY_VALIDATION,
        "capture_window": validation["capture_window"],
        "capture": {
            key: validation["capture"].get(key)
            for key in ("elapsed_seconds", "rows", "write", "rss_sampled")
        },
        "counterfactual_status": cf.get("status"),
        "identical_value_only": cf.get("identical_value_only"),
        "identical_position": cf.get("identical_position"),
        "keys_recounted": corrected["summary"]["keys_recounted"],
    }


def _route_version_for_day(base: Path) -> int:
    """验证日生效模型版本（2019-04-01 属折 0 测试域 → v24008）。"""
    clock = fold_clock.build_fold_clock(base)
    for row in clock["rows"]:
        if row.get("kind") != "fold":
            continue
        if str(row.get("test_start")) <= spec.SINGLE_DAY_VALIDATION <= str(row.get("test_end")):
            return int(row["model_version"])
    raise StageError(f"验证日 {spec.SINGLE_DAY_VALIDATION} 不在任何折测试域内")


def _write_single_day_measurement(
    base: Path,
    camp: Path,
    validation: Mapping[str, Any],
    probe_record: Mapping[str, Any],
    key_control: Mapping[str, Any],
) -> Path:
    """组装并写单日端到端证据附件（含 content_digest 自校验）。"""
    payload = {
        "schema": "p7_cashflow_split_single_day/v1",
        "batch": f"{spec.DEFAULT_CAMPAIGN}/measure/single_day_{spec.SINGLE_DAY_VALIDATION}",
        "day": spec.SINGLE_DAY_VALIDATION,
        "purpose": (
            "r0_003 复核第 1 项端到端：注入 T 日 PIT 行业 → 三遍标签反事实 → "
            "探针消费真实输出；第 3 项：RSS 采样与写盘单列"
        ),
        "industry_injection": validation["industry_injection"],
        "capture": validation["capture"],
        "counterfactual": {
            key: value
            for key, value in validation["counterfactual"].items()
            if key not in ("frames",)
        },
        "probe_consumption": probe_record,
        "key_control_totals": key_control.get("totals"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload["content_digest"] = structbuild.measurement_content_digest(payload)
    target = camp / "measurements" / f"single_day_{spec.SINGLE_DAY_VALIDATION}.json"
    write_json(target, payload)
    return target


class StageError(RuntimeError):
    """阶段无法继续（未授权、上游缺件、口径不符）。"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.v2_p7.cashflow_sleeve_split.run",
        description="CF-SPLIT-20-v1 研究阶段 CLI（规格冻结，无参数扫描入口）",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    for name in spec.STAGES:
        sub = subparsers.add_parser(name, help=f"阶段 {name}")
        sub.add_argument("--campaign", default=spec.DEFAULT_CAMPAIGN)
        sub.add_argument("--run", dest="run_id", default=None)
        sub.add_argument("--parent", default=None, help="同 campaign 下的上游 run 身份")
        sub.add_argument("--artifact-base", default=None)
        sub.add_argument("--repo-root", default=str(REPO_ROOT))
        sub.add_argument(
            "--allow-root",
            action="append",
            default=[],
            help="隔离测试用的额外写域根（生产运行不得追加生产根）",
        )
        sub.add_argument(
            "--tests-log",
            default=None,
            help="复用既有 pytest 日志（缺省则在 prepare 内跑定向测试）",
        )
        sub.add_argument(
            "--batch",
            action="append",
            default=None,
            help="run-a 专用：只执行指定批次（可多次；缺省按序执行全部未完成批次）",
        )
        sub.add_argument(
            "--workers",
            type=int,
            default=1,
            help="run-a rebuild 专用：并行捕获进程数（默认 1=串行；只影响 rebuild 批）",
        )
    return parser


def _log(stage: str, run_id: str) -> Callable[[str], None]:
    def emit(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}][{stage}/{run_id}] {message}", flush=True)

    return emit


# ── prepare（审点一唯一实现）──────────────────────────────────────────


def _preflight_stage(base: Path, work: Path, emit) -> Dict[str, Any]:
    """prepare 前置四步：规格预登记、旧 A 锚核验、特征路由、折时钟（写工件）。"""
    emit("1/8 规格与机器预登记")
    machine = spec.machine_spec_document()
    write_json(work / "spec.json", machine)
    emit(
        f"    spec_sha256={machine['spec_sha256'][:16]} protocol={machine['protocol_sha256'][:16]}"
    )

    emit("2/8 旧 A 锚哈希核验 + 输入清单")
    baseline = identity.verify_a_baseline(base)
    probe_dates = [item["date"] for item in spec.PROBE_DATES]
    manifest = identity.build_input_manifest(
        base,
        probe_dates=probe_dates,
        a_baseline={
            "verified": True,
            "relative": baseline["relative"],
            "manifest_sha256": baseline["manifest_sha256"],
            "consumed_files": baseline["consumed_files"],
            "account_checks": baseline["account_checks"],
            "reference_delta": baseline["reference_delta"],
            "status": baseline["status"],
        },
    )
    write_json(work / "input_manifest.json", manifest)
    emit(f"    sources={len(manifest['sources'])} gaps={len(manifest['gaps'])}")

    emit("3/8 逐折特征路由")
    routes = feature_routes.build_routes(base)
    write_json(work / "feature_routes.json", routes)
    emit(f"    versions={routes['summary']['versions']} b_count={routes['summary']['b_count']}")

    emit("4/8 折表与时钟（部署行含早停净化）")
    clock = fold_clock.build_fold_clock(base)
    fold_clock.write_fold_clock_csv(clock, work / "fold_clock.csv")
    deploy = clock["deploy"]
    emit(
        f"    folds={clock['checks']['folds']} seam={clock['checks']['seam_days']} "
        f"deploy={clock['checks']['deploy_days']} deploy_es_purged={deploy['es_purged_samples']} "
        f"(last={deploy['es_last_usable_sample']})"
    )
    return {
        "machine": machine,
        "baseline": baseline,
        "manifest": manifest,
        "routes": routes,
        "clock": clock,
    }


def run_prepare(args: argparse.Namespace) -> Dict[str, Any]:
    """机器预登记 + 输入/特征/日期核查 + 只读小探针 + 成本登记。"""
    base = Path(args.repo_root)
    artifact_base = Path(args.artifact_base) if args.artifact_base else None
    extra = list(args.allow_root or [])
    camp = artifacts.campaign_root(args.campaign, artifact_base)
    artifacts.ensure_layout(camp)
    run_id = args.run_id or artifacts.new_run_id("prepare")
    run_dir = artifacts.create_run_dir(camp, run_id, extra)
    work = run_dir / "work"
    emit = _log("prepare", run_id)
    started = time.time()

    preflight = _preflight_stage(base, work, emit)
    machine = preflight["machine"]
    baseline = preflight["baseline"]
    routes = preflight["routes"]
    clock = preflight["clock"]

    emit(f"5/8 单日端到端验证（{spec.SINGLE_DAY_VALIDATION}：行业注入 + 三遍反事实，无收益）")
    table = feature_routes.route_table(routes)
    version_for_date = _version_map(clock, base)
    validation = _single_day_validation(base, camp, table, emit)
    day = spec.SINGLE_DAY_VALIDATION

    emit("6/8 只读小探针（7 日 + 单日结构帧，不产生收益）")
    probe_result = probes.run_probes(
        base,
        routes=table,
        version_for_date=version_for_date,
        structural_frames={day: validation["structural_frame"]},
    )
    single_probe = next(
        record
        for record in probe_result["records"]
        if record["date"] == day and str(record.get("feature_source", "")).startswith("structural")
    )
    emit(
        f"    ok={probe_result['summary']['ok_days']}/{probe_result['summary']['probe_days']} "
        f"elapsed={probe_result['summary']['total_elapsed_seconds']}s；单日结构帧 "
        f"old_domain={single_probe.get('old_domain_observed_n')} structural="
        f"{single_probe.get('structural_domain_observed_n')} alignment="
        f"{(single_probe.get('industry_pit_alignment') or {}).get('status')}"
    )

    emit("7/8 键数重算 + 测量附件修正（57 日直读；单日证据封存）")
    corrected = _refresh_measurements(base, camp, validation, single_probe, emit)
    write_json(work / "measurements_refresh.json", corrected["summary"])

    emit("8/8 就绪清点 + 成本与预算 + 代码可还原清单 + 定向测试")
    cf_evidence = corrected["single_day_payload"]
    ready = readiness.build_readiness(
        base,
        extra_roots=extra,
        routes=routes,
        clock=clock,
        probe_result=probe_result,
        counterfactual_evidence=cf_evidence,
    )
    write_json(work / "readiness.json", ready)
    blocking = [item for item in ready["blockers"] if item["severity"] == "blocking_for_review"]
    if blocking:
        raise StageError(f"存在阻断项，不能发布审点一材料：{[i['id'] for i in blocking]}")

    io_baseline = budget.measure_io_baseline(base)
    costs = budget.build_pilot_costs(
        probes.probe_costs(probe_result),
        io_baseline,
        ready["costs"]["hash_throughput"],
        extra_measured={"calibration_hint": budget.calibration_hint(probe_result)},
        structural=budget.load_structural_measurement(camp),
    )
    write_json(work / "pilot_costs.json", costs)

    restore = identity.code_restore_payload(base)
    restore_path = camp / "code_restore" / "code_restore.json"
    write_json(restore_path, restore)
    emit(f"    code_restore files={restore['files_n']} patch={restore['tracked_patch_sha256']}")

    # 附件：实际补丁 + 被改源码/测试 + 测量件（随 run 封存并逐件验 SHA）
    patch_path = work / "tracked.patch"
    patch_text = identity._run_git(["diff", "--binary", "--no-color"], base)
    patch_path.write_text(patch_text + "\n", encoding="utf-8", newline="\n")
    attachment_sources: Dict[str, Path] = {"tracked.patch": patch_path}
    for item in restore["files"]:
        attachment_sources[f"files/{item['path']}"] = base / item["path"]
    for measured in sorted((camp / "measurements").glob("*.json")):
        attachment_sources[f"measurements/{measured.name}"] = measured
    attachment_digests = artifacts.publish_attachments(run_dir, attachment_sources, extra)
    emit(f"    attachments={len(attachment_digests)}")
    tests_status = _write_tests(work / "tests.txt", args.tests_log, base, emit)
    if tests_status["status"] != "passed":
        raise StageError(f"定向测试未通过，拒绝发布：{tests_status}")

    emit("发布工件并封存 manifest")
    published = [
        artifacts.publish_file(work / name, run_dir, name, extra)
        for name in spec.STAGE_ARTIFACTS["prepare"]
    ]
    sealed = artifacts.seal_run(
        run_dir,
        "prepare",
        {
            "spec_sha256": machine["spec_sha256"],
            "protocol_sha256": machine["protocol_sha256"],
            "code_identity": ready["code_identity"],
            "a_baseline_manifest_sha256": baseline["manifest_sha256"],
            "a_baseline_reference_status": baseline["reference_delta"],
            "probe_summary": probe_result["summary"],
            "probe_records": ready["probe_records"],
            "probe_version_map": version_for_date,
            "findings": ready["findings"],
            "single_day_validation": _single_day_seal_payload(validation, corrected),
            "code_restore": {
                "path": str(restore_path.relative_to(base)).replace("\\", "/"),
                "files_n": restore["files_n"],
                "tracked_patch_sha256": restore["tracked_patch_sha256"],
            },
            "structural_rebuild": costs["structural_rebuild"],
            "a_baseline_frozen_files": ready["protocol_frozen_preflight"].get(
                "a_baseline_frozen_files"
            ),
            "blockers": ready["blockers"],
            "not_run": ready["not_run"],
            "tests": tests_status,
            "elapsed_seconds": round(time.time() - started, 1),
            "artifact_sha256": {item["name"]: item["sha256"] for item in published},
            "attachments_n": len(attachment_digests),
            "attachment_paths": sorted(attachment_digests),
        },
        spec.STAGE_ARTIFACTS["prepare"],
        extra,
        attachments=attachment_digests,
    )
    emit(f"封存完成：{sealed['manifest_sha256'][:16]}")
    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "manifest_sha256": sealed["manifest_sha256"],
        "spec_sha256": machine["spec_sha256"],
        "probe_summary": probe_result["summary"],
        "blockers": [item["id"] for item in ready["blockers"]],
        "elapsed_seconds": round(time.time() - started, 1),
    }


def _version_map(clock: Mapping[str, Any], base: Path) -> Dict[str, int]:
    """探针日 → 当日生效模型版本（折 / 接缝沿用末折 / 部署）。"""
    folds = [row for row in clock["rows"] if row.get("kind") == "fold"]
    timeline = fold_clock.build_timeline(folds, base)
    mapping: Dict[str, int] = {}
    for item in spec.PROBE_DATES:
        day = item["date"]
        for entry in timeline:
            if str(entry["first"]) <= str(day) <= str(entry["last"]):
                mapping[day] = int(entry["model_version"])
                break
    return mapping


def _write_tests(
    target: Path, tests_log: str | None, repo_root: Path, emit: Callable[[str], None]
) -> Dict[str, Any]:
    """生成 ``tests.txt``：复用既有日志，或就地跑定向测试（非绿不发布）。"""
    if tests_log:
        source = Path(tests_log)
        if not source.exists():
            raise StageError(f"--tests-log 指定的日志不存在：{source}")
        text = source.read_text(encoding="utf-8", errors="replace")
        passed = _tests_passed(text)
        target.write_text(text, encoding="utf-8", newline="\n")
        return {
            "status": "passed" if passed else "failed",
            "source": str(source),
            "command": "复用既有日志",
            "sha256": sha256_file(target),
        }

    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TEST_TARGETS]
    emit(f"    运行定向测试：{' '.join(TEST_TARGETS)}")
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    header = (
        f"# command: {' '.join(command)}\n"
        f"# exit_code: {completed.returncode}\n"
        f"# python: {sys.version.split()[0]}\n\n"
    )
    text = header + (completed.stdout or "") + (completed.stderr or "")
    target.write_text(text, encoding="utf-8", newline="\n")
    passed = completed.returncode == 0 and _tests_passed(text)
    return {
        "status": "passed" if passed else "failed",
        "source": "inline pytest",
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "sha256": sha256_file(target),
    }


def _tests_passed(text: str) -> bool:
    """只认末行汇总：``N passed`` 且无 ``failed``/``error``。"""
    for line in reversed([item.strip() for item in text.splitlines() if item.strip()]):
        lowered = line.lower()
        if " passed" in lowered or " failed" in lowered or " error" in lowered:
            return (
                (" failed" not in lowered) and (" error" not in lowered) and (" passed" in lowered)
            )
    return False


# ── 未授权阶段：显式拒绝（不写任何工件）──────────────────────────────


def _refuse(stage: str, args: argparse.Namespace) -> Dict[str, Any]:
    """审点一后的三批：只报告未授权与前置条件，不产生任何数据。"""
    upstream = {
        "run-a": "prepare",
        "run-b": "run-a",
        "verify": "run-b",
    }[stage]
    message = (
        f"阶段 {stage} 未授权：审点一材料须由用户转交 Codex 审核通过，"
        f"并由用户按命名批次批准后才能启动（协议 §2.4、§8）。"
        f"前置 = 已发布的 {upstream} run（--parent）；本次未写任何工件。"
    )
    return {
        "status": "blocked",
        "stage": stage,
        "reason": "not_authorized",
        "required_upstream": upstream,
        "parent": args.parent,
        "run_id": args.run_id,
        "message": message,
    }


STAGE_HANDLERS: Dict[str, Callable[[argparse.Namespace], Dict[str, Any]]] = {
    "prepare": run_prepare,
    "run-a": run_a.run_run_a,
    "run-b": lambda args: _refuse("run-b", args),
    "verify": lambda args: _refuse("verify", args),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage not in spec.STAGES:
        parser.print_usage()
        return EXIT_USAGE
    handler = STAGE_HANDLERS[args.stage]
    try:
        result = handler(args)
    except (
        StageError,
        ArtifactError,
        identity.IdentityError,
        feature_routes.RouteError,
        fold_clock.ClockError,
        probes.ProbeError,
    ) as error:
        print(f"阶段阻断：{error}", file=sys.stderr)
        return EXIT_BLOCKED
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    if result.get("status") == "blocked":
        return EXIT_BLOCKED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
