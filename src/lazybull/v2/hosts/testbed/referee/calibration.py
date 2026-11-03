# -*- coding: utf-8 -*-
"""P5b T4 收益裁判校准编排：冻结 manifest → 训练阈值 → holdout → 功效/MDE → 判定表。

冻结顺序（顺序即纪律）
======================
1. ``freeze_calibration_manifest``：**先**冻结 §5.5 全表（报告身份、统计量、look、阈值规则、
   效应网格、场景参数、block 规则、实现数/种子、源对与源池隔离、行域/时钟、回撤占位、H6 占位、
   差异登记、限制声明），之后不可改写（独占创建）；
2. ``train_calibration``：只读 manifest + 训练池，产出 ``calibration_thresholds.json``
   （记录 manifest_sha256 与 calibrator_sha256）；
3. ``run_holdout``：校验三环哈希后跑 holdout + 功效，产出 ``income_holdout.json`` /
   ``income_power.json`` / ``referee_verdict.json`` / 报告；**只读临界值**，不得回写；
4. 墙钟单独落 ``machine_time.json``（不参与确定性比对）。

全部写操作都过试车场写域守卫（生产根硬禁）并独占创建（不覆盖既有产物）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from src.lazybull.v2.hosts.testbed.paths import assert_write_scope, project_root
from src.lazybull.v2.hosts.testbed.referee import artifacts
from src.lazybull.v2.hosts.testbed.referee.holdout import evaluate_holdout
from src.lazybull.v2.hosts.testbed.referee.ledger_source import ledger_sample_identity
from src.lazybull.v2.hosts.testbed.referee.manifest import (
    CALIBRATOR_MODULES,
    MANIFEST_SCHEMA,
    ManifestError,
    calibrator_identity,
    file_sha256,
    freeze_manifest,
    load_manifest,
    manifest_sha256,
    verify_calibration_chain,
    verify_source_files,
)
from src.lazybull.v2.hosts.testbed.referee.manifest_sections import (
    DESIGN_DEFAULTS,
    deviation_registry_section,
    drawdown_and_h6_section,
    effect_and_holdout_section,
    finite_safe,
    limitations_section,
    lineage_section,
    source_pair_section,
    statistic_section,
    threshold_section,
)
from src.lazybull.v2.hosts.testbed.referee.power import evaluate_power
from src.lazybull.v2.hosts.testbed.referee.report import (
    build_income_report,
    build_referee_verdict,
    render_markdown,
)
from src.lazybull.v2.hosts.testbed.referee.sampling import IMPLEMENTATION_ID, implementation_ids
from src.lazybull.v2.hosts.testbed.referee.scenes import (
    DgpParams,
    SourcePair,
    build_scene_registry,
    load_source_pair,
    pool_for_folds,
    scene_parameters,
)
from src.lazybull.v2.hosts.testbed.referee.statistics import (
    DEFAULT_CHUNK,
    make_look_plan,
)
from src.lazybull.v2.hosts.testbed.referee.training import (
    ThresholdConfig,
    train_income_thresholds,
)

__all__ = [
    "DEFAULT_CHUNK",
    "DESIGN_DEFAULTS",
    "build_manifest_document",
    "freeze_calibration_manifest",
    "load_calibration_manifest",
    "register_ledger_sample",
    "run_holdout",
    "source_pair_for_manifest",
    "train_calibration",
    "write_machine_time",
]

#: 分块路径数（分块与不分块结果逐值一致，见测试；定义在 ``statistics`` 以避免循环导入）
#: 保留本模块同名别名，供既有调用方与脚本按 ``calibration.DEFAULT_CHUNK`` 读取。
DEFAULT_CHUNK = DEFAULT_CHUNK


@dataclass(frozen=True)
class PreparedSource:
    """冻结源对 + 两池（训练/holdout 各自去中心，禁止跨池）。"""

    pair: SourcePair
    train_pool: Dict[str, Any]
    holdout_pool: Dict[str, Any]
    params: DgpParams


def source_pair_for_manifest(
    doc: Mapping[str, Any], *, repo_root: Optional[Path] = None
) -> PreparedSource:
    """按 manifest 读取源对并构造两池（哈希不符即失败，禁止换源）。"""
    root = Path(repo_root) if repo_root is not None else project_root()
    verify_source_files(doc, root=root)
    pair_doc = doc["source_pair"]
    pair = load_source_pair(
        root / str(pair_doc["base_chain_nav"]),
        root / str(pair_doc["noise_chain_nav"]),
        expected_base_sha256=str(pair_doc["base_sha256"]),
        expected_noise_sha256=str(pair_doc["noise_sha256"]),
    )
    h = int(doc["income_referee"]["sample"]["H"])
    return PreparedSource(
        pair=pair,
        train_pool=pool_for_folds(pair, pair_doc["train_folds"]),
        holdout_pool=pool_for_folds(pair, pair_doc["holdout_folds"]),
        params=DgpParams.from_manifest(doc, h=h),
    )


def _registry(source: PreparedSource, role: str):
    pool = source.train_pool if role == "train" else source.holdout_pool
    return build_scene_registry(pool["centred"], params=source.params, pool_role=role)


def build_manifest_document(
    *,
    run_id: str,
    source: PreparedSource,
    prereg: Mapping[str, str],
    psd_manifest: Mapping[str, str],
    counts: Mapping[str, Any],
    calibrator: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Path] = None,
    h: int = 20,
    lineage: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """生成 §5.5 全表 manifest 文档（不含封存字段；由 :func:`freeze_manifest` 封存）。

    各段由独立构造函数产出（便于逐段评审与单元测试）：源对 / 统计量 / 阈值 / 效应与 holdout /
    场景与 block / 实现数与子流 / 引用文件与校准器身份 / T5-T6 占位 / 差异登记 / 限制声明 / 谱系。

    Args:
        run_id: 本次校准身份。
        source: 已读取的冻结源对（提供两池与日期块证据）。
        prereg: ``{"path": ..., "sha256": ...}``（PSD 预登记）。
        psd_manifest: ``{"path": ..., "sha256": ...}``（T0.5b 冻结 manifest）。
        counts: ``{train_per_unit, holdout_per_unit, effect_per_scene_per_level, seed,
            realization_prefix}``。
        calibrator: 裁判代码身份（缺省 = 复算）。
        repo_root: 仓库根（用于把源对路径登记为仓库相对路径）。
        h: 持有期（默认 20）。
        lineage: 冻结谱系（缺省 = 首次冻结 sequence=1；重冻必须给前身哈希与原因）。

    Returns:
        合法（``validate_manifest`` 通过）的 manifest 文档。
    """
    root = Path(repo_root) if repo_root is not None else project_root()
    plan = make_look_plan(h)
    identity = dict(calibrator) if calibrator is not None else calibrator_identity()
    placeholders = drawdown_and_h6_section()
    referee = {
        "statistic": statistic_section(prereg),
        "sample": plan.as_dict(),
        "threshold": threshold_section(plan),
        **effect_and_holdout_section(),
    }
    doc = {
        "schema": MANIFEST_SCHEMA,
        "run_id": str(run_id),
        "created_at": "2026-11-02",
        "status": "frozen_before_evidence",
        "scope": "historical_testbed",
        "evidence": {
            "evidence_class": "calibration_only",
            "virtual_clock": True,
            "not_forward_evidence": True,
            "not_sla_evidence": True,
            "not_real_candidate": True,
            "no_ledger_entry": True,
            "row_domain": "structural",
            "schedule_mode": "replay",
        },
        "income_referee": referee,
        "scenes": scene_parameters(source.params),
        "calibration": {
            "dgp": source.params.as_dict(),
            "block_resample": {
                "method": "circular",
                "block_length": int(source.params.block_length),
                "cross_pool_sampling_forbidden": True,
            },
        },
        "block_resample": {
            "method": "circular",
            "block_length": int(source.params.block_length),
            "center": "each pool centred by its own mean",
            "wrap": "index modulo pool length",
            "truncation_forbidden": True,
            "cross_pool_sampling_forbidden": True,
            "date_key_alignment": ["split_index", "date"],
        },
        "counts": {
            "train_per_unit": int(counts["train_per_unit"]),
            "holdout_per_unit": int(counts["holdout_per_unit"]),
            "effect_per_scene_per_level": int(counts["effect_per_scene_per_level"]),
            "seed": int(counts["seed"]),
            "realization_prefix": str(counts["realization_prefix"]),
            "chunk": DEFAULT_CHUNK,
        },
        "substreams": implementation_ids(),
        "source_pair": source_pair_section(source, root),
        "files": {
            "prereg": {"path": str(prereg["path"]), "sha256": str(prereg["sha256"])},
            "psd_manifest": {
                "path": str(psd_manifest["path"]),
                "sha256": str(psd_manifest["sha256"]),
            },
            "calibrator": {
                "implementation_id": identity["implementation_id"],
                "sha256": identity["sha256"],
                "modules": dict(identity["modules"]),
                "covered_modules": list(CALIBRATOR_MODULES),
            },
        },
        "deviation_registry": deviation_registry_section(counts),
        "limitations": limitations_section(),
        "lineage": (dict(lineage) if lineage is not None else lineage_section()),
    }
    doc.update(placeholders)
    return doc


def _guard(campaign_root: Path, *, allow_external_root: bool) -> Path:
    return assert_write_scope(Path(campaign_root), allow_external_root=allow_external_root)


def freeze_calibration_manifest(
    campaign_root: Path,
    doc: Mapping[str, Any],
    *,
    apply: bool,
    allow_external_root: bool = False,
) -> Dict[str, Any]:
    """冻结 manifest 到 campaign ``reports/``（默认 dry-run 只返回计划）。"""
    root = _guard(campaign_root, allow_external_root=allow_external_root)
    target = artifacts.artifact_path(root, artifacts.ARTIFACT_MANIFEST)
    if not apply:
        return {"dry_run": True, "target": target.as_posix(), "written": False}
    frozen = freeze_manifest(doc, target)
    return {"dry_run": False, "target": target.as_posix(), "written": True, **frozen}


def load_calibration_manifest(
    campaign_root: Path,
    *,
    require_scale_floor: bool = True,
    verify_calibrator: bool = True,
    repo_manifest: Optional[Path] = None,
) -> Tuple[Dict[str, Any], str]:
    """读取 campaign 内的冻结 manifest（返回文档与文件 sha256）。

    Args:
        campaign_root: campaign 根。
        require_scale_floor: 是否强制 §5.3 实现数下限（生产 True）。
        verify_calibrator: 是否复算校准器身份并比对（生产 True）。
        repo_manifest: 可选的仓库冻结件路径；给出时**额外核验两份文件逐字节一致**，
            不一致即拒绝运行（防止本地冻结件与 campaign 副本漂移）。

    Raises:
        ManifestError: 字段/口径/实现数/校准器身份不符，或与仓库冻结件字节不一致。
    """
    target = artifacts.artifact_path(Path(campaign_root), artifacts.ARTIFACT_MANIFEST)
    doc = load_manifest(
        target,
        require_scale_floor=require_scale_floor,
        verify_calibrator=verify_calibrator,
    )
    if repo_manifest is not None:
        repo_path = Path(repo_manifest)
        if not repo_path.is_file():
            raise ManifestError(f"仓库冻结件不存在（禁止在未冻结状态下运行）: {repo_path}")
        if repo_path.read_bytes() != target.read_bytes():
            raise ManifestError(
                "仓库冻结件与 campaign 副本不是逐字节一致（拒绝运行）: " f"{repo_path} vs {target}"
            )
    return doc, manifest_sha256(target)


def train_calibration(
    campaign_root: Path,
    *,
    repo_root: Optional[Path] = None,
    repo_manifest: Optional[Path] = None,
    apply: bool = False,
    allow_external_root: bool = False,
    require_scale_floor: bool = True,
    chunk: Optional[int] = DEFAULT_CHUNK,
) -> Dict[str, Any]:
    """跑训练并写 ``calibration_thresholds.json``（默认 dry-run 只算不写）。

    Args:
        campaign_root: campaign 根。
        repo_root: 仓库根（源对定位）。
        repo_manifest: 可选的仓库冻结件；给出时核验与 campaign 副本逐字节一致。
        apply: ``False`` = dry-run 只算不写。
        allow_external_root: 仅测试放开写域白名单。
        require_scale_floor: 是否强制 §5.3 实现数下限（生产必须 True）。
        chunk: 分块路径数。

    Returns:
        ``{dry_run, passed, c_by_look, failures, determining_units, thresholds_sha256,
        machine_seconds, counts}``。
    """
    root = _guard(campaign_root, allow_external_root=allow_external_root)
    doc, doc_sha = load_calibration_manifest(
        root, require_scale_floor=require_scale_floor, repo_manifest=repo_manifest
    )
    identity = calibrator_identity()
    source = source_pair_for_manifest(doc, repo_root=repo_root)
    plan = make_look_plan(
        int(doc["income_referee"]["sample"]["H"]), int(doc["income_referee"]["sample"]["n_max"])
    )
    started = time.perf_counter()
    result = train_income_thresholds(
        plan=plan,
        registry=_registry(source, "train"),
        seed=int(doc["counts"]["seed"]),
        counts=doc["counts"],
        config=ThresholdConfig.from_manifest(doc),
        prefix=str(doc["counts"]["realization_prefix"]),
        statistic=str(doc["income_referee"]["statistic"]["name"]),
        delta=float(doc["income_referee"]["statistic"]["delta_log_daily"]),
        chunk=chunk,
    )
    seconds = time.perf_counter() - started
    payload = {
        **finite_safe(result.as_dict()),
        "run_id": str(doc["run_id"]),
        "manifest_sha256": doc_sha,
        "calibrator_sha256": identity["sha256"],
        "passed": bool(result.passed),
        "failure_detail": (
            "配置失败原因见 failures / units[*].rejection；c 列为 null 表示该 look 不可估计"
            if not result.passed
            else None
        ),
        "counts": {
            "train_per_unit": int(doc["counts"]["train_per_unit"]),
            "holdout_per_unit": int(doc["counts"]["holdout_per_unit"]),
            "effect_per_scene_per_level": int(doc["counts"]["effect_per_scene_per_level"]),
            "seed": int(doc["counts"]["seed"]),
            "realization_prefix": str(doc["counts"]["realization_prefix"]),
            "chunk": chunk,
        },
        "source": {
            "train_pool_rows": int(source.train_pool["rows"]),
            "holdout_pool_rows": int(source.holdout_pool["rows"]),
            "train_pool_mean": float(source.train_pool["mean"]),
        },
    }
    outcome: Dict[str, Any] = {
        "dry_run": not apply,
        "passed": bool(result.passed),
        # 配置失败时 c 为 nan/None：统一转 None，保证 CLI/JSON 输出合法且不误导
        "c_by_look": finite_safe([float(value) for value in result.thresholds]),
        "failures": [dict(item) for item in result.failures],
        "determining_units": list(result.determining_units),
        "machine_seconds": seconds,
        "counts": payload["counts"],
    }
    if not apply:
        return outcome
    written = artifacts.write_json_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_THRESHOLDS),
        payload,
        what="临界值产物",
    )
    outcome.update({"thresholds_sha256": written["sha256"], "path": written["path"]})
    return outcome


def _holdout_inputs(
    root: Path,
    *,
    repo_root: Optional[Path],
    repo_manifest: Optional[Path] = None,
    require_scale_floor: bool,
) -> Dict[str, Any]:
    """读入并校验 holdout 全部前置（manifest + 临界值 + 三环哈希 + 源对）。

    Raises:
        ManifestError: 哈希链断裂、临界值缺失或临界值未通过（config failure）。
    """
    doc, doc_sha = load_calibration_manifest(
        root,
        require_scale_floor=require_scale_floor,
        repo_manifest=repo_manifest,
    )
    identity = calibrator_identity()
    thresholds_path = artifacts.artifact_path(root, artifacts.ARTIFACT_THRESHOLDS)
    thresholds_doc = artifacts.load_artifact(
        thresholds_path, what="临界值产物", schema="p5b_testbed_calibration_thresholds/v1"
    )
    thresholds_sha = file_sha256(thresholds_path)
    verify_calibration_chain(
        manifest_doc=doc,
        manifest_file_sha256=doc_sha,
        calibrator=identity,
        thresholds=thresholds_doc,
        thresholds_sha256=thresholds_sha,
    )
    if not thresholds_doc.get("passed"):
        raise ManifestError(
            "临界值产物标记为未通过（config failure）："
            "不得进入 holdout 验收（禁止看 holdout 调 c_k）"
        )
    return {
        "doc": doc,
        "doc_sha": doc_sha,
        "identity": identity,
        "thresholds_doc": thresholds_doc,
        "thresholds_sha": thresholds_sha,
        "source": source_pair_for_manifest(doc, repo_root=repo_root),
        "plan": make_look_plan(
            int(doc["income_referee"]["sample"]["H"]),
            int(doc["income_referee"]["sample"]["n_max"]),
        ),
        "frozen": [float(value) for value in thresholds_doc["c_by_look"]],
    }


def _run_evaluation(inputs: Mapping[str, Any], *, chunk: Optional[int]) -> Dict[str, Any]:
    """跑 holdout + 功效（同一 holdout 池；功效的 MDE 登记依赖 A/A 结果）。"""
    doc = inputs["doc"]
    frozen = inputs["frozen"]
    plan = inputs["plan"]
    statistic = str(doc["income_referee"]["statistic"]["name"])
    delta = float(doc["income_referee"]["statistic"]["delta_log_daily"])
    seed = int(doc["counts"]["seed"])
    prefix = str(doc["counts"]["realization_prefix"])
    holdout_doc = evaluate_holdout(
        plan=plan,
        registry=_registry(inputs["source"], "holdout"),
        seed=seed,
        count=int(doc["counts"]["holdout_per_unit"]),
        thresholds=frozen,
        config=ThresholdConfig.from_manifest(doc),
        statistic=statistic,
        delta=delta,
        prefix=prefix,
        wilson_upper_max=float(doc["income_referee"]["holdout"]["wilson_upper_max"]),
        chunk=chunk,
    )
    power_doc = evaluate_power(
        plan=plan,
        registry=_registry(inputs["source"], "holdout"),
        seed=seed,
        count=int(doc["counts"]["effect_per_scene_per_level"]),
        thresholds=frozen,
        applicable=bool(holdout_doc["referee_usable"]),
        statistic=statistic,
        delta=delta,
        prefix=prefix,
        chunk=chunk,
        invalid_rate_per_look_max=float(
            doc["income_referee"]["threshold"]["invalid_rate_per_look_max"]
        ),
    )
    links = {
        "manifest_sha256": inputs["doc_sha"],
        "calibrator_sha256": inputs["identity"]["sha256"],
        "thresholds_sha256": inputs["thresholds_sha"],
    }
    return {**links, "holdout": {**holdout_doc, **links}, "power": {**power_doc, **links}}


def _write_holdout_outputs(
    root: Path,
    inputs: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    ledger_sample: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """写 holdout / power / verdict / 报告（独占创建，绝不覆盖）。"""
    doc = inputs["doc"]
    identity = inputs["identity"]
    holdout_doc = evaluation["holdout"]
    power_doc = evaluation["power"]
    holdout_written = artifacts.write_json_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_HOLDOUT),
        holdout_doc,
        what="holdout 产物",
    )
    power_written = artifacts.write_json_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_POWER), power_doc, what="功效产物"
    )
    verdict = build_referee_verdict(
        run_id=str(doc["run_id"]),
        campaign_id=root.name,
        manifest_sha256=inputs["doc_sha"],
        calibrator_sha256=identity["sha256"],
        thresholds_sha256=inputs["thresholds_sha"],
        holdout_sha256=holdout_written["sha256"],
        power_sha256=power_written["sha256"],
        holdout=holdout_doc,
        power=power_doc,
        implementation_id=IMPLEMENTATION_ID,
    )
    verdict_written = artifacts.write_json_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_VERDICT), verdict, what="判定表"
    )
    report = build_income_report(
        campaign_id=root.name,
        run_id=str(doc["run_id"]),
        manifest=doc,
        manifest_sha256=inputs["doc_sha"],
        calibrator=identity,
        thresholds=inputs["thresholds_doc"],
        thresholds_sha256=inputs["thresholds_sha"],
        holdout=holdout_doc,
        holdout_sha256=holdout_written["sha256"],
        power=power_doc,
        power_sha256=power_written["sha256"],
        verdict=verdict,
        verdict_sha256=verdict_written["sha256"],
        source=inputs["source"],
        ledger_sample=ledger_sample,
    )
    report_written = artifacts.write_json_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_REPORT), report, what="收益裁判报告"
    )
    markdown_written = artifacts.write_text_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_REPORT_MD),
        render_markdown(report),
        what="收益裁判报告（markdown）",
    )
    return {
        "holdout": holdout_written,
        "power": power_written,
        "verdict": verdict_written,
        "report": report_written,
        "report_md": markdown_written,
    }


def run_holdout(
    campaign_root: Path,
    *,
    repo_root: Optional[Path] = None,
    repo_manifest: Optional[Path] = None,
    apply: bool = False,
    allow_external_root: bool = False,
    chunk: Optional[int] = DEFAULT_CHUNK,
    ledger_sample: Optional[Mapping[str, Any]] = None,
    require_scale_floor: bool = True,
) -> Dict[str, Any]:
    """跑 holdout + 功效并写 holdout/power/verdict/报告（默认 dry-run）。

    Args:
        campaign_root: campaign 根。
        repo_root: 仓库根（源对/引用文件定位）。
        repo_manifest: 可选的仓库冻结件；给出时核验与 campaign 副本逐字节一致。
        apply: ``False`` = dry-run 只算不写。
        allow_external_root: 仅测试放开写域白名单。
        chunk: 分块路径数。
        ledger_sample: 可选的配对账本样本身份（只登记摘要）。
        require_scale_floor: 是否强制 §5.3 实现数下限（生产必须 True；测试可放开）。

    Raises:
        ManifestError: 哈希链断裂、临界值缺失或产物已存在（禁止覆盖）。
    """
    root = _guard(campaign_root, allow_external_root=allow_external_root)
    inputs = _holdout_inputs(
        root,
        repo_root=repo_root,
        repo_manifest=repo_manifest,
        require_scale_floor=require_scale_floor,
    )
    started = time.perf_counter()
    evaluation = _run_evaluation(inputs, chunk=chunk)
    seconds = time.perf_counter() - started
    holdout_doc = evaluation["holdout"]
    outcome: Dict[str, Any] = {
        "dry_run": not apply,
        "referee_usable": bool(holdout_doc["referee_usable"]),
        "failed_units": list(holdout_doc["failed_units"]),
        "unit_count": int(holdout_doc["unit_count"]),
        "mde": dict(evaluation["power"]["mde"]),
        "machine_seconds": seconds,
    }
    if not apply:
        return outcome
    outcome.update(_write_holdout_outputs(root, inputs, evaluation, ledger_sample=ledger_sample))
    return outcome


def register_ledger_sample(campaign_root: Path, sample: Mapping[str, Any]) -> Dict[str, Any]:
    """把（可选）配对账本样本身份登记进报告（只做摘要，不落 NAV/不改账本）。"""
    doc = artifacts.load_artifact(
        artifacts.artifact_path(Path(campaign_root), artifacts.ARTIFACT_REPORT),
        what="收益裁判报告",
    )
    doc["ledger_sample"] = dict(ledger_sample_identity(sample))
    return doc


def write_machine_time(campaign_root: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """落墙钟登记（不参与确定性比对；独占写）。"""
    return artifacts.write_json_once(
        artifacts.artifact_path(Path(campaign_root), artifacts.ARTIFACT_MACHINE_TIME),
        payload,
        what="墙钟登记",
        sealed=False,
    )
