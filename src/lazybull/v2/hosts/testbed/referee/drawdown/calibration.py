# -*- coding: utf-8 -*-
"""P5b T5 回撤裁判校准编排：冻结 manifest → 训练 c^W/c^I → holdout → 功效/MDE → 报告。

冻结顺序（顺序即纪律）
======================
1. ``freeze_drawdown_manifest``：**先**冻结 §5.5 回撤清单（读数/MDD/窗口/注入/反事实/求解/
   阈值/A/A/holdout/效应/实现数/子流/源对/上游哈希/H6 占位/差异/限制/谱系），之后不可改写；
   冻结阶段只读 B0（盲态选窗），**不读任何 A/A 或效应读数**；
2. ``train_drawdown_calibration``：只读 manifest + 训练池，产出 ``drawdown_thresholds.json``；
3. ``run_drawdown_holdout``：校验三环哈希后跑 holdout + 功效，产出 holdout/power/verdict/报告；
4. 墙钟单独落 ``machine_time.json``（不参与确定性比对）。

全部写操作都过试车场写域守卫（生产根硬禁）并独占创建（不覆盖既有产物）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import assert_write_scope, project_root
from src.lazybull.v2.hosts.testbed.referee.drawdown import artifacts
from src.lazybull.v2.hosts.testbed.referee.drawdown.crash_window import (
    CrashWindow,
    assert_windows_fixed,
    crash_window_mask,
    find_drawdown_episodes,
    select_crash_windows,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.holdout import (
    evaluate_drawdown_holdout,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.injection import (
    TOLERANCE_PP,
    InjectionSolution,
    apply_injection,
    injection_support,
    solve_injection,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.manifest import (
    CALIBRATOR_EXCLUSIONS,
    MANIFEST_SCHEMA,
    ManifestError,
    calibrator_identity,
    file_sha256,
    freeze_manifest,
    load_manifest,
    manifest_sha256,
    verify_calibration_chain,
    verify_upstream,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.mdd import (
    ANCHOR_HORIZON,
    DIRECTION_IMPROVE,
    DIRECTIONS,
    WINDOW_SAMPLES,
    baseline_percent_by_look,
    drawdown_look_plan,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.power import (
    EFFECT_GRID_PP,
    EffectCell,
    evaluate_drawdown_power,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.realizations import (
    SourceWindow,
    draw_arm_paths,
    load_source_window,
    window_identity,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.report import (
    build_drawdown_report,
    build_drawdown_verdict,
    render_drawdown_markdown,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.sampling import (
    IMPLEMENTATION_ID,
    implementation_ids,
    substream,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.sections import (
    deviation_registry_section,
    drawdown_referee_section,
    h6_section,
    limitations_section,
    lineage_section,
    repo_relative,
    source_pair_section,
)
from src.lazybull.v2.hosts.testbed.referee.drawdown.thresholds import (
    AaReadings,
    DrawdownThresholds,
    aa_readings_from_independent_paths,
    aa_summary,
    merge_aa_readings,
    reading_matrix,
    train_drawdown_thresholds,
)
from src.lazybull.v2.hosts.testbed.referee.scenes import (
    DgpParams,
    SourcePair,
    build_scene_registry,
    load_source_pair,
    pool_for_folds,
    scene_parameters,
)

__all__ = [
    "DEFAULT_CHUNK",
    "PreparedSource",
    "build_drawdown_manifest",
    "collect_aa_readings",
    "freeze_drawdown_manifest",
    "load_drawdown_manifest",
    "prepare_source",
    "run_drawdown_holdout",
    "source_from_pair_document",
    "train_drawdown_calibration",
    "write_machine_time",
]

#: 分块实现数（只影响峰值内存，不影响结果）
DEFAULT_CHUNK = 2000
#: B0 冻结窗口 MDD 与 manifest 登记值的对照容差（pp；证明"窗口未被重选"）
WINDOW_MDD_TOLERANCE = 1e-4


@dataclass(frozen=True)
class PreparedSource:
    """冻结源对 + 两池 + B0 冻结窗口 + 冻结崩盘窗 + DGP 参数。"""

    pair: SourcePair
    train_pool: Dict[str, Any]
    holdout_pool: Dict[str, Any]
    params: DgpParams
    window: SourceWindow
    windows: Tuple[CrashWindow, ...]

    def registry(self, role: str):
        """按角色取场景注册表（``train`` 池与 ``holdout`` 池互斥）。"""
        pool = self.train_pool if role == "train" else self.holdout_pool
        return build_scene_registry(pool["centred"], params=self.params, pool_role=role)

    def window_mask(self) -> np.ndarray:
        """冻结崩盘窗并集的布尔掩码（训练/holdout/效应共用同一组窗）。"""
        return crash_window_mask(self.window.samples, self.windows)

    def baseline_percent(self) -> np.ndarray:
        """基准臂 B0 在各 look 上的 MDD（pp；全实现共用同一组减数）。"""
        return baseline_percent_by_look(self.window.log_returns, drawdown_look_plan().look_days)


def source_from_pair_document(
    pair_doc: Mapping[str, Any],
    *,
    dgp_doc: Mapping[str, Any],
    repo_root: Optional[Path] = None,
    h: int = 20,
    window_samples: int = WINDOW_SAMPLES,
    expected_window_mdd: Optional[float] = None,
    mdd_tolerance: float = WINDOW_MDD_TOLERANCE,
) -> PreparedSource:
    """由（已核验的）源对声明读取 B0 窗口并识别崩盘窗（冻结阶段用；盲态）。"""
    root = Path(repo_root) if repo_root is not None else project_root()
    pair = load_source_pair(
        root / str(pair_doc["base_chain_nav"]),
        root / str(pair_doc["noise_chain_nav"]),
        expected_base_sha256=str(pair_doc["base_sha256"]),
        expected_noise_sha256=str(pair_doc["noise_sha256"]),
    )
    window = load_source_window(
        root / str(pair_doc["base_chain_nav"]),
        expected_sha256=str(pair_doc["base_sha256"]),
        window=int(window_samples),
        expected_mdd_percent=expected_window_mdd,
        mdd_tolerance=float(mdd_tolerance),
    )
    windows = select_crash_windows(find_drawdown_episodes(window.log_returns))
    return PreparedSource(
        pair=pair,
        train_pool=pool_for_folds(pair, pair_doc["train_folds"]),
        holdout_pool=pool_for_folds(pair, pair_doc["holdout_folds"]),
        params=DgpParams.from_manifest({"calibration": dict(dgp_doc)}, h=int(h)),
        window=window,
        windows=windows,
    )


def prepare_source(
    doc: Mapping[str, Any],
    *,
    repo_root: Optional[Path] = None,
    require_scale_floor: bool = True,
) -> PreparedSource:
    """按 manifest 读取源对与 B0 冻结窗口，并核验窗口/崩盘窗与冻结值逐字段一致。"""
    verify_upstream(
        doc,
        repo_root=Path(repo_root) if repo_root is not None else project_root(),
        require_scale_floor=require_scale_floor,
    )
    reading = doc["drawdown_referee"]["reading"]
    source = source_from_pair_document(
        doc["source_pair"],
        dgp_doc=doc["calibration"],
        repo_root=repo_root,
        h=int(reading["H"]),
        window_samples=int(doc["drawdown_referee"]["windows"]["window_samples"]),
        expected_window_mdd=float(doc["source_window"]["window"]["mdd_percent"]),
    )
    assert_windows_fixed(
        doc["drawdown_referee"]["windows"]["frozen_list"],
        source.windows,
        samples=source.window.samples,
    )
    return source


def build_drawdown_manifest(
    *,
    run_id: str,
    source: PreparedSource,
    windows: Tuple[CrashWindow, ...],
    counts: Mapping[str, Any],
    upstream: Mapping[str, Any],
    calibrator: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Path] = None,
    created_at: str = "2026-11-02",
    lineage: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """生成 §5.5 回撤清单 manifest（不含封存字段；由 ``freeze_manifest`` 封存）。"""
    root = Path(repo_root) if repo_root is not None else project_root()
    plan = drawdown_look_plan()
    identity = dict(calibrator) if calibrator is not None else calibrator_identity()
    parent_manifest = dict(upstream["parent_manifest"])
    parent_manifest["path"] = repo_relative(str(parent_manifest["path"]), root)
    parent_manifest["run_id"] = str(upstream["parent_run_id"])
    parent_manifest["lineage_sequence"] = int(upstream["parent_lineage_sequence"])
    parent_thresholds = dict(upstream["parent_thresholds"])
    parent_thresholds["path"] = repo_relative(str(parent_thresholds["path"]), root)
    source_window = window_identity(source.window, source.params)
    source_window["window"]["source_path"] = repo_relative(source.window.source_path, root)
    return {
        "schema": MANIFEST_SCHEMA,
        "run_id": str(run_id),
        "created_at": str(created_at),
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
        "drawdown_referee": drawdown_referee_section(plan, windows, samples=source.window.samples),
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
        "source_window": source_window,
        "files": {
            "calibrator": {
                "implementation_id": identity["implementation_id"],
                "sha256": identity["sha256"],
                "modules": dict(identity["modules"]),
                "covered_modules": list(identity["modules"]),
                "exclusions": dict(CALIBRATOR_EXCLUSIONS),
            },
            "parent_manifest": parent_manifest,
            "parent_thresholds": parent_thresholds,
        },
        "deviation_registry": deviation_registry_section(counts),
        "limitations": limitations_section(),
        "lineage": dict(lineage) if lineage is not None else lineage_section(),
        "h6_numeric": h6_section(),
    }


def freeze_drawdown_manifest(
    campaign_root: Path,
    doc: Mapping[str, Any],
    *,
    apply: bool,
    allow_external_root: bool = False,
) -> Dict[str, Any]:
    """冻结回撤 manifest 到 campaign ``reports/``（默认 dry-run 只返回计划）。

    dry-run 也执行完整校验（否则"预演通过"没有意义）。
    """
    from src.lazybull.v2.hosts.testbed.referee.drawdown.manifest import validate_manifest

    root = assert_write_scope(Path(campaign_root), allow_external_root=allow_external_root)
    target = artifacts.artifact_path(root, artifacts.ARTIFACT_MANIFEST)
    errors = validate_manifest(doc)
    if errors:
        raise ManifestError("manifest 不可冻结（dry-run 校验失败）: " + "; ".join(errors))
    if not apply:
        return {"dry_run": True, "target": target.as_posix(), "written": False, "errors": []}
    frozen = freeze_manifest(doc, target)
    return {"dry_run": False, "target": target.as_posix(), "written": True, **frozen}


def load_drawdown_manifest(
    campaign_root: Path,
    *,
    require_scale_floor: bool = True,
    verify_calibrator: bool = True,
    repo_manifest: Optional[Path] = None,
) -> Tuple[Dict[str, Any], str]:
    """读取 campaign 内冻结的回撤 manifest（给出 ``repo_manifest`` 时额外核验逐字节一致）。"""
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


def collect_aa_readings(
    source: PreparedSource,
    *,
    stage: str,
    scene: str,
    count: int,
    seed: int,
    prefix: str,
    chunk: Optional[int] = DEFAULT_CHUNK,
) -> AaReadings:
    """在一个单元内抽 A/A **独立**实现并转成 ``D_k`` 读数（分块不影响结果）。

    A/A 的 arm 是独立抽取的 ``B0 + n``，base 是冻结基准臂 ``B0``（不允许同一路径零注入）。
    """
    plan = drawdown_look_plan()
    item = substream(seed, stage, scene, null="aa", prefix=prefix)
    registry = source.registry("train" if stage == "train" else "holdout")
    total = int(count)
    if total <= 0:
        raise ManifestError(f"实现数必须为正，实际 {count!r}")
    size = int(chunk) if chunk else total
    blocks: List[AaReadings] = []
    produced = 0
    while produced < total:
        step = min(size, total - produced)
        arm = draw_arm_paths(source.window, registry, scene, item.generator, step)
        blocks.append(
            aa_readings_from_independent_paths(
                arm,
                source.window.log_returns,
                plan,
                scene=scene,
                realization_id=item.realization_id,
            )
        )
        produced += step
    return merge_aa_readings(blocks)


def _train_thresholds(
    source: PreparedSource, doc: Mapping[str, Any], *, chunk: Optional[int]
) -> Tuple[DrawdownThresholds, Dict[str, str]]:
    readings = [
        collect_aa_readings(
            source,
            stage="train",
            scene=scene,
            count=int(doc["counts"]["train_per_unit"]),
            seed=int(doc["counts"]["seed"]),
            prefix=str(doc["counts"]["realization_prefix"]),
            chunk=chunk,
        )
        for scene in sorted(doc["scenes"])
    ]
    config = doc["drawdown_referee"]["threshold"]
    thresholds = train_drawdown_thresholds(
        readings, drawdown_look_plan(), alpha_train=float(config["alpha_train"])
    )
    return (
        thresholds,
        {item.scene: item.realization_id for item in readings},
        aa_summary(readings),
    )


def train_drawdown_calibration(
    campaign_root: Path,
    *,
    repo_root: Optional[Path] = None,
    repo_manifest: Optional[Path] = None,
    apply: bool = False,
    allow_external_root: bool = False,
    require_scale_floor: bool = True,
    chunk: Optional[int] = DEFAULT_CHUNK,
) -> Dict[str, Any]:
    """跑训练并写 ``drawdown_thresholds.json``（默认 dry-run 只算不写）。"""
    root = assert_write_scope(Path(campaign_root), allow_external_root=allow_external_root)
    doc, doc_sha = load_drawdown_manifest(
        root, require_scale_floor=require_scale_floor, repo_manifest=repo_manifest
    )
    identity = calibrator_identity()
    source = prepare_source(doc, repo_root=repo_root, require_scale_floor=require_scale_floor)
    started = time.perf_counter()
    thresholds, realization_ids, aa_rows = _train_thresholds(source, doc, chunk=chunk)
    seconds = time.perf_counter() - started
    payload = {
        **thresholds.as_dict(),
        "run_id": str(doc["run_id"]),
        "manifest_sha256": doc_sha,
        "calibrator_sha256": identity["sha256"],
        "implementation_id": IMPLEMENTATION_ID,
        "passed": True,
        "realization_ids": dict(sorted(realization_ids.items())),
        "counts": {**dict(doc["counts"]), "executed_chunk": None if chunk is None else int(chunk)},
        "window": dict(doc["source_window"]["window"]),
        "aa_summary": aa_rows,
    }
    outcome: Dict[str, Any] = {
        "dry_run": not apply,
        "c_w_by_look": [float(value) for value in thresholds.worsen],
        "c_i_by_look": [float(value) for value in thresholds.improve],
        "determining_units_worsen": list(thresholds.determining_worsen),
        "determining_units_improve": list(thresholds.determining_improve),
        "machine_seconds": seconds,
        "counts": dict(doc["counts"]),
    }
    if not apply:
        return outcome
    written = artifacts.write_json_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_THRESHOLDS),
        payload,
        what="回撤临界值产物",
    )
    outcome.update({"thresholds_sha256": written["sha256"], "path": written["path"]})
    return outcome


def _thresholds_from_artifact(doc: Mapping[str, Any]) -> DrawdownThresholds:
    """从临界值产物重建阈值对象（只读；不重算分位、不回写）。"""
    return DrawdownThresholds(
        plan=drawdown_look_plan(),
        alpha_train=float(doc["alpha_train"]),
        alpha_per_look=float(doc["alpha_per_look"]),
        worsen=tuple(float(item) for item in doc["c_w_by_look"]),
        improve=tuple(float(item) for item in doc["c_i_by_look"]),
        worsen_by_scene={
            key: tuple(float(v) for v in values)
            for key, values in dict(doc["c_w_by_scene"]).items()
        },
        improve_by_scene={
            key: tuple(float(v) for v in values)
            for key, values in dict(doc["c_i_by_scene"]).items()
        },
        worsen_units=tuple(dict(item) for item in doc["units_worsen"]),
        improve_units=tuple(dict(item) for item in doc["units_improve"]),
        determining_worsen=tuple(doc["determining_units_worsen"]),
        determining_improve=tuple(doc["determining_units_improve"]),
    )


def _fill_block(
    paths: np.ndarray,
    solution: InjectionSolution,
    mask: np.ndarray,
    plan: Any,
    *,
    direction: str,
    baseline_pct: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把一次求解结果铺成逐 look 读数矩阵（不可构造行留 ``nan``，仍计入分母）。

    - 读数 ``D_k``（含噪观察效应）= ``MDD(arm,T_k) - MDD(B0,T_k)``（基准臂一次固定）；
    - ``pure`` = 锚上**纯注入效应** ``MDD(arm) - MDD(x)``（求解器目标，非读数）；
    - 不可构造行两个量都留 ``nan``，但该实现仍在主功效分母里（按未检出）。
    """
    rows = int(paths.shape[0])
    d_matrix = np.full((rows, int(plan.k)), np.nan, dtype=float)
    observed = np.full(rows, np.nan, dtype=float)
    ok = np.asarray(solution.constructible, dtype=bool)
    parameter = np.asarray(solution.parameter, dtype=float).copy()
    pure = np.asarray(solution.achieved_pp, dtype=float).copy()
    parameter[~ok] = np.nan
    pure[~ok] = np.nan
    if ok.any():
        arm = apply_injection(
            np.asarray(paths, dtype=float)[ok],
            injection_support(np.asarray(paths, dtype=float)[ok], mask),
            direction=direction,
            parameter=parameter[ok],
        )
        d_matrix[ok] = reading_matrix(arm, baseline_pct, plan)
        observed[ok] = d_matrix[ok, -1]
    return d_matrix, ok, parameter, pure, observed


def _effect_cell(
    source: PreparedSource,
    doc: Mapping[str, Any],
    *,
    direction: str,
    level_pp: float,
    scene: str,
    mask: np.ndarray,
    chunk: Optional[int],
) -> EffectCell:
    """一个「方向 × 档 × 场景」单元：求 λ/ζ → 注入一次 → 逐 look 读数 + 含噪观察效应。"""
    plan = drawdown_look_plan()
    item = substream(
        int(doc["counts"]["seed"]),
        "effect",
        scene,
        null="dd_i" if direction == DIRECTION_IMPROVE else "dd_w",
        level=float(level_pp),
        prefix=str(doc["counts"]["realization_prefix"]),
    )
    registry = source.registry("holdout")
    total = int(doc["counts"]["effect_per_scene_per_level"])
    size = int(chunk) if chunk else total
    target = float(level_pp) if direction == DIRECTION_IMPROVE else -float(level_pp)
    baseline_pct = source.baseline_percent()
    blocks = []
    produced = 0
    while produced < total:
        step = min(size, total - produced)
        noise = registry.build(scene, item.generator, step)
        paths = source.window.log_returns[None, :] + noise
        solution = solve_injection(
            paths,
            mask,
            direction=direction,
            target_pp=target,
            anchor=ANCHOR_HORIZON,
            tolerance=TOLERANCE_PP,
        )
        blocks.append(
            _fill_block(
                paths,
                solution,
                mask,
                plan,
                direction=direction,
                baseline_pct=baseline_pct,
            )
        )
        produced += step
    return EffectCell(
        direction=direction,
        level_pp=float(level_pp),
        scene=scene,
        d_matrix=np.vstack([item_[0] for item_ in blocks]),
        constructible=np.concatenate([item_[1] for item_ in blocks]),
        parameter=np.concatenate([item_[2] for item_ in blocks]),
        pure_pp=np.concatenate([item_[3] for item_ in blocks]),
        observed_pp=np.concatenate([item_[4] for item_ in blocks]),
        realization_id=item.realization_id,
    )


def reference_solves_on_b0(
    source: PreparedSource, mask: np.ndarray
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """B0 单路径（无噪声）的 λ/ζ 复算——与 T0.5b §5 预检读数的对照锚点。

    只读 B0 冻结窗口，不参与阈值训练、不参与 holdout；用于证明注入器/窗口/锚与已批预检一致
    （T0.5b：最后 500 日窗口 12pp 可由 ``λ≈0.49`` 达到）。
    """
    single = source.window.log_returns[None, :]
    reference: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for direction in DIRECTIONS:
        reference[direction] = {}
        for level in EFFECT_GRID_PP:
            target = float(level) if direction == DIRECTION_IMPROVE else -float(level)
            solution = solve_injection(
                single,
                mask,
                direction=direction,
                target_pp=target,
                anchor=ANCHOR_HORIZON,
                tolerance=TOLERANCE_PP,
            )
            ok = bool(np.asarray(solution.constructible)[0])
            reference[direction][str(float(level))] = {
                "parameter": float(np.asarray(solution.parameter)[0]) if ok else None,
                "pure_effect_pp": float(np.asarray(solution.achieved_pp)[0]),
                "constructible": ok,
                "status": solution.status[0],
            }
    return reference


def _run_evaluation(
    source: PreparedSource,
    doc: Mapping[str, Any],
    thresholds_doc: Mapping[str, Any],
    *,
    chunk: Optional[int],
) -> Dict[str, Any]:
    """跑 holdout + 功效（功效的 MDE 登记依赖两个方向各自的 A/A 结果）。"""
    thresholds = _thresholds_from_artifact(thresholds_doc)
    holdout_count = int(doc["counts"]["holdout_per_unit"])
    readings = [
        collect_aa_readings(
            source,
            stage="holdout",
            scene=scene,
            count=holdout_count,
            seed=int(doc["counts"]["seed"]),
            prefix=str(doc["counts"]["realization_prefix"]),
            chunk=chunk,
        )
        for scene in sorted(doc["scenes"])
    ]
    holdout = evaluate_drawdown_holdout(
        readings,
        thresholds,
        plan=drawdown_look_plan(),
        count=holdout_count,
        scenes=sorted(doc["scenes"]),
        wilson_upper_max=float(doc["drawdown_referee"]["holdout"]["wilson_upper_max"]),
    )
    holdout["counts"]["executed_chunk"] = None if chunk is None else int(chunk)
    mask = source.window_mask()
    cells = [
        _effect_cell(
            source, doc, direction=direction, level_pp=level, scene=scene, mask=mask, chunk=chunk
        )
        for direction in DIRECTIONS
        for level in EFFECT_GRID_PP
        for scene in sorted(doc["scenes"])
    ]
    power = evaluate_drawdown_power(
        cells,
        thresholds,
        plan=drawdown_look_plan(),
        applicable=dict(holdout["usable_by_direction"]),
        scenes=sorted(doc["scenes"]),
        chunk=chunk,
        per_scene_realizations=int(doc["counts"]["effect_per_scene_per_level"]),
    )
    power["reference_on_b0"] = reference_solves_on_b0(source, mask)
    return {"holdout": holdout, "power": power}


def _write_outputs(
    root: Path,
    doc: Mapping[str, Any],
    doc_sha: str,
    identity: Mapping[str, Any],
    thresholds_doc: Mapping[str, Any],
    thresholds_sha: str,
    evaluation: Mapping[str, Any],
    upstream: Mapping[str, Any],
    source: PreparedSource,
) -> Dict[str, Any]:
    """写 holdout / power / verdict / 报告（独占创建，绝不覆盖）。"""
    holdout_doc = dict(evaluation["holdout"])
    power_doc = dict(evaluation["power"])
    links = {
        "manifest_sha256": doc_sha,
        "calibrator_sha256": identity["sha256"],
        "thresholds_sha256": thresholds_sha,
    }
    holdout_doc.update(links)
    power_doc.update(links)
    holdout_written = artifacts.write_json_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_HOLDOUT),
        holdout_doc,
        what="回撤 holdout 产物",
    )
    power_written = artifacts.write_json_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_POWER),
        power_doc,
        what="回撤功效产物",
    )
    verdict = build_drawdown_verdict(
        run_id=str(doc["run_id"]),
        campaign_id=root.name,
        manifest_sha256=doc_sha,
        calibrator_sha256=identity["sha256"],
        thresholds_sha256=thresholds_sha,
        holdout_sha256=holdout_written["sha256"],
        power_sha256=power_written["sha256"],
        thresholds=thresholds_doc,
        holdout=holdout_doc,
        power=power_doc,
        implementation_id=IMPLEMENTATION_ID,
        windows=list(doc["drawdown_referee"]["windows"]["frozen_list"]),
    )
    verdict_written = artifacts.write_json_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_VERDICT), verdict, what="回撤判定表"
    )
    report = build_drawdown_report(
        campaign_id=root.name,
        run_id=str(doc["run_id"]),
        manifest=doc,
        manifest_sha256=doc_sha,
        calibrator=identity,
        thresholds=thresholds_doc,
        thresholds_sha256=thresholds_sha,
        holdout=holdout_doc,
        holdout_sha256=holdout_written["sha256"],
        power=power_doc,
        power_sha256=power_written["sha256"],
        verdict=verdict,
        verdict_sha256=verdict_written["sha256"],
        upstream=upstream,
        source=source,
    )
    # 不再静默把非有限值转 null：产物纪律是「禁 NaN/Inf」，写盘即失败（fail-closed）
    report_written = artifacts.write_json_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_REPORT), report, what="回撤裁判报告"
    )
    markdown_written = artifacts.write_text_once(
        artifacts.artifact_path(root, artifacts.ARTIFACT_REPORT_MD),
        render_drawdown_markdown(report),
        what="回撤裁判报告（markdown）",
    )
    return {
        "holdout": holdout_written,
        "power": power_written,
        "verdict": verdict_written,
        "report": report_written,
        "report_md": markdown_written,
    }


def run_drawdown_holdout(
    campaign_root: Path,
    *,
    repo_root: Optional[Path] = None,
    repo_manifest: Optional[Path] = None,
    apply: bool = False,
    allow_external_root: bool = False,
    chunk: Optional[int] = DEFAULT_CHUNK,
    require_scale_floor: bool = True,
) -> Dict[str, Any]:
    """跑 holdout + 功效并写 holdout/power/verdict/报告（默认 dry-run）。"""
    root = assert_write_scope(Path(campaign_root), allow_external_root=allow_external_root)
    doc, doc_sha = load_drawdown_manifest(
        root, require_scale_floor=require_scale_floor, repo_manifest=repo_manifest
    )
    identity = calibrator_identity()
    thresholds_path = artifacts.artifact_path(root, artifacts.ARTIFACT_THRESHOLDS)
    thresholds_doc = artifacts.load_artifact(
        thresholds_path, what="回撤临界值产物", schema="p5b_testbed_drawdown_thresholds/v1"
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
        raise ManifestError("回撤临界值产物标记为未通过：不得进入 holdout 验收")
    source = prepare_source(doc, repo_root=repo_root, require_scale_floor=require_scale_floor)
    upstream = verify_upstream(
        doc,
        repo_root=Path(repo_root) if repo_root else project_root(),
        require_scale_floor=require_scale_floor,
    )
    started = time.perf_counter()
    evaluation = _run_evaluation(source, doc, thresholds_doc, chunk=chunk)
    seconds = time.perf_counter() - started
    holdout_doc = evaluation["holdout"]
    outcome: Dict[str, Any] = {
        "dry_run": not apply,
        "referee_usable": bool(holdout_doc["referee_usable"]),
        "usable_by_direction": dict(holdout_doc["usable_by_direction"]),
        "failed_units": list(holdout_doc["failed_units"]),
        "unit_count": int(holdout_doc["unit_count"]),
        "mde_worsen": dict(evaluation["power"]["mde"]["MDE_drawdown_worsen"]),
        "mde_improve": dict(evaluation["power"]["mde"]["MDE_drawdown_improve"]),
        "machine_seconds": seconds,
    }
    if not apply:
        return outcome
    registered_upstream = dict(upstream)
    registered_upstream["parent_manifest"] = dict(doc["files"]["parent_manifest"])
    registered_upstream["parent_thresholds"] = dict(doc["files"]["parent_thresholds"])
    outcome.update(
        _write_outputs(
            root,
            doc,
            doc_sha,
            identity,
            thresholds_doc,
            thresholds_sha,
            evaluation,
            registered_upstream,
            source,
        )
    )
    return outcome


#: 阶段 → 墙钟产物名（分阶段独立落盘；holdout 为主批次）
MACHINE_TIME_STAGES: Mapping[str, str] = {
    "freeze": artifacts.ARTIFACT_MACHINE_TIME_FREEZE,
    "train": artifacts.ARTIFACT_MACHINE_TIME_TRAIN,
    "holdout": artifacts.ARTIFACT_MACHINE_TIME,
}


def write_machine_time(
    campaign_root: Path, payload: Mapping[str, Any], *, stage: str = "holdout"
) -> Dict[str, Any]:
    """落墙钟登记（不参与确定性比对；独占写；按阶段分文件）。"""
    if stage not in MACHINE_TIME_STAGES:
        raise ManifestError(f"未登记墙钟阶段: {stage!r}（允许 {sorted(MACHINE_TIME_STAGES)}）")
    return artifacts.write_json_once(
        artifacts.artifact_path(Path(campaign_root), MACHINE_TIME_STAGES[stage]),
        payload,
        what=f"墙钟登记（{stage}）",
        sealed=False,
    )
