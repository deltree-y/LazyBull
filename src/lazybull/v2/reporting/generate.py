# -*- coding: utf-8 -*-
"""驾驶舱生成编排（P6 T2：稳定快照 → 投影 → 视图 → 渲染 → manifest → 原子发布）。

单向数据流（设计 §2.1）：显式根映射 → 私有稳定快照 → 既有 loader/纯函数 → 白名单投影 →
静态 HTML → 自封存 manifest → 新 run 原子发布。本模块**不新增指标**、不读 `.env`、
不取 P4 写锁、不调用 ensure/恢复/runner，也不构造 Storage/Registry/日历。

证据域隔离（设计 §2.2）在生成本身 fail-closed：``production`` 混入 research/calibration
输入直接抛 ``EvidenceIsolationError``（CLI 映射为非零退出），不生成“有效混合报告”。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd
import pyarrow

from src.lazybull import PROJECT_ROOT
from src.lazybull.v2.reporting import (
    cockpit,
    dashboard_view,
    deps,
    evidence,
    experiment_csv,
    experiment_guard,
    experiment_projection,
    experiment_sources,
    experiment_view,
    manifest,
    publish,
    render,
    trend_csv,
    trend_projection,
    trend_sources,
    trend_view,
)
from src.lazybull.v2.reporting import snapshot as snapshot_module
from src.lazybull.v2.reporting.snapshot import StableSnapshot, capture_stable_snapshot

__all__ = [
    "DashboardRequest",
    "ExperimentRequest",
    "IMPLEMENTED_KINDS",
    "TrendRequest",
    "build_generator_record",
    "generate_dashboard",
    "generate_experiment",
    "generate_report",
    "generate_trend",
]

#: 质量长表来源（相对质量根；唯一读口）
QUALITY_RELATIVE = "latest_metrics.parquet"
QUALITY_ROOT_ID = "quality"
_MISSING_VERSION = "<unavailable>"
#: 已接入的报告类型（T2 驾驶舱 / T3 实验 / T4 趋势）。未接入的类型必须在**采集快照前**
#: 明确拒绝，否则会以驾驶舱内容发布到 ``research/experiment/`` 之类路径（T2-R3-04）。
IMPLEMENTED_KINDS: tuple[str, ...] = ("dashboard", "experiment", "trend")


@dataclass(frozen=True)
class TrendRequest:
    """一次趋势库生成请求（证据域 + 输入根 + 可选上一快照，全部显式）。"""

    reports_root: Path
    evidence_class: str
    generated_at: str
    attempt_id: str
    input_evidence_classes: tuple[str, ...]
    sources: tuple[str, ...] = ()
    paper_root: Path | None = None
    bundles: tuple[str, ...] = ()
    previous: Path | None = None
    report_kind: str = "trend"
    git_record: Mapping[str, Any] | None = field(default=None)
    publish_run: bool = True


@dataclass(frozen=True)
class ExperimentRequest:
    """一次实验报告生成请求（显式基线/臂；禁止隐式默认来源）。"""

    reports_root: Path
    evidence_class: str
    generated_at: str
    attempt_id: str
    input_evidence_classes: tuple[str, ...]
    arms: tuple[str, ...]
    baseline: str
    regime_json: Path | None = None
    splits: tuple[int, ...] | None = None
    report_kind: str = "experiment"
    git_record: Mapping[str, Any] | None = field(default=None)
    publish_run: bool = True


@dataclass(frozen=True)
class DashboardRequest:
    """一次驾驶舱生成请求（全部输入显式给出，禁止隐式默认）。"""

    paper_root: Path
    reports_root: Path
    evidence_class: str
    generated_at: str
    attempt_id: str
    input_evidence_classes: tuple[str, ...]
    quality_root: Path | None = None
    report_kind: str = "dashboard"
    three_ends: Mapping[str, Any] | None = field(default=None)
    git_record: Mapping[str, Any] | None = field(default=None)
    publish_run: bool = True


def _project_version() -> str:
    """项目版本（唯一来源 ``pyproject.toml``；读取失败显式标不可用，不猜值）。"""
    path = Path(PROJECT_ROOT) / "pyproject.toml"
    try:
        import tomllib

        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, ValueError, ImportError):
        return _MISSING_VERSION
    version = payload.get("tool", {}).get("poetry", {}).get("version")
    return str(version) if version else _MISSING_VERSION


def build_generator_record(git_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """manifest ``generator`` 字段：依赖精确版本 + 生成器/模板逐文件 SHA + git 状态。"""
    jinja2 = deps.load_jinja2()
    git = dict(git_record or {})
    dirty = git.get("git_dirty")
    return {
        "project": "lazybull",
        "project_version": _project_version(),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "jinja2": str(getattr(jinja2, "__version__", "") or ""),
        "git_commit": str(git.get("git_commit") or _MISSING_VERSION),
        "git_dirty": dirty if isinstance(dirty, bool) else None,
        "generator_files": render.generator_file_sha256(),
        "template_sha256": render.template_sha256(),
        "entrypoint": render.entrypoint_identity(),
    }


def _roots(request: DashboardRequest) -> dict[str, Path]:
    """显式根映射（质量根可选；全部先过受信根判定）。"""
    roots: dict[str, Path] = {cockpit.PAPER_ROOT_ID: Path(request.paper_root)}
    if request.quality_root is not None:
        roots[QUALITY_ROOT_ID] = Path(request.quality_root)
    return roots


def _capture(request: DashboardRequest) -> tuple[StableSnapshot, dict[str, Any]]:
    """采集一次稳定快照（纸面固定文件 + 指令目录 + 质量长表）。

    重试口径与 T1 一致（最多 2 次，即最多 3 次尝试）：全部 attempt 目录都在
    ``.staging/<attempt>/`` 下，故精确写白名单的四键绑定 ``attempt`` 仍然覆盖它们。
    """
    info: dict[str, Any] = {}

    def reader(snapshot: StableSnapshot) -> None:
        info.clear()
        info.update(cockpit.capture_paper_inputs(snapshot))
        if request.quality_root is not None:
            snapshot.capture(QUALITY_ROOT_ID, QUALITY_RELATIVE)

    retries = snapshot_module.MAX_RETRIES
    reports_root = Path(request.reports_root)
    publish.assert_attempt_id(request.attempt_id)
    staging_root = reports_root / ".staging" / request.attempt_id
    if staging_root.resolve().parent != (reports_root / ".staging").resolve():
        # 纵深防御：attempt 已校验，这里再断言 staging 只能落在 <reports_root>/.staging/ 之下
        raise publish.PublishError(
            f"staging 目录越出 .staging 根：{staging_root}（attempt={request.attempt_id!r}）"
        )
    snapshot_result, audit = capture_stable_snapshot(
        roots=_roots(request),
        staging_root=staging_root,
        attempt_ids=[f"r{index + 1}" for index in range(retries + 1)],
        reader=reader,
        max_retries=retries,
    )
    info["_audit"] = audit
    return snapshot_result, info


def _quality_rows(
    snapshot: StableSnapshot, request: DashboardRequest
) -> tuple[list[dict[str, Any]] | None, bool]:
    """从**副本字节**读质量长表（缺失即 ``None``/不存在，不补 NaN）。"""
    if request.quality_root is None or not snapshot.exists(QUALITY_ROOT_ID, QUALITY_RELATIVE):
        return None, False
    frame = pd.read_parquet(snapshot.copy_path(QUALITY_ROOT_ID, QUALITY_RELATIVE))
    return cockpit.load_quality_rows(frame), True


def _projection(
    snapshot: StableSnapshot, request: DashboardRequest, info: Mapping[str, Any]
) -> dict:
    """装配驾驶舱投影（质量行只从快照副本读入）。"""
    rows, present = _quality_rows(snapshot, request)
    return cockpit.build_dashboard_projection(
        snapshot=snapshot,
        generated_at=request.generated_at,
        evidence_class=request.evidence_class,
        capture_info=dict(info),
        quality_rows=rows,
        quality_present=present,
    )


def _report_id(
    inputs: Sequence[Mapping[str, Any]], request: DashboardRequest | ExperimentRequest
) -> str:
    """``p6_<kind>_<UTC>_<input_manifest_sha256 前 12 位>``（T1 唯一实现）。"""
    digest = manifest.input_manifest_sha256(manifest.build_inputs(inputs))
    return manifest.build_report_id(
        report_kind=request.report_kind,
        generated_at=request.generated_at,
        input_sha256=digest,
    )


def _json_bytes(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    """canonical 风格 JSON 字节（UTF-8/LF、sort_keys、allow_nan=False）。"""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _text(value: Any) -> str | None:
    """可选文本（``None`` 保持 ``None``，不写空串冒充）。"""
    return None if value is None else str(value)


def _attach_root_map(projection: dict[str, Any], request: DashboardRequest) -> dict[str, Any]:
    """把本次实际读取的根补进 ``root_map``（覆盖 ``inputs`` 中出现的全部 root_id）。

    T1 投影的 ``root_map`` 只登记纸面根；T2 额外读了质量长表，若不补登记，
    ``inputs`` 里的 ``quality:`` 前缀无法在 T5 复算时解析回具体根（T2-R1-02）。
    """
    root_map = dict(projection.get("root_map") or {})
    root_map.setdefault(cockpit.PAPER_ROOT_ID, "v2 纸面生产根（只读副本）")
    if request.quality_root is not None:
        root_map[QUALITY_ROOT_ID] = "质量长表来源（只读副本）"
    projection["root_map"] = {str(key): str(value) for key, value in sorted(root_map.items())}
    return projection


def _data_window(projection: Mapping[str, Any]) -> dict[str, Any]:
    """数据窗口（账户起始日 → 权威代日；均取原值，不外推）。"""
    config = dict(projection.get("config") or {})
    authority = dict(projection.get("authority") or {})
    return {
        "start": _text(config.get("account_start_date")),
        "end": _text(authority.get("trade_date")),
    }


def _source_watermarks(projection: Mapping[str, Any]) -> dict[str, Any]:
    """数据水位（纸面代 / NAV 日 / 账户日 / 质量指标行数；不读 mtime）。"""
    authority = dict(projection.get("authority") or {})
    nav = dict(projection.get("nav") or {})
    quality = dict(projection.get("quality") or {})
    account = (projection.get("account") or {}).get("account")
    account_map = account if isinstance(account, Mapping) else {}
    generation = None
    if authority.get("trade_date") and authority.get("run_kind"):
        generation = f"{authority['run_kind']}@{authority['trade_date']}"
    return {
        "paper_generation": generation,
        "nav_date": _text(nav.get("nav_date")),
        "account_day": _text(account_map.get("last_update")),
        "quality_metric_rows": quality.get("metric_count"),
    }


def _compose(
    snapshot: StableSnapshot, request: DashboardRequest, info: Mapping[str, Any]
) -> dict[str, Any]:
    """组装 run 文件（HTML/projection/manifest）与 manifest 记录。"""
    projection = _attach_root_map(_projection(snapshot, request, info), request)
    inputs = [receipt.as_dict() for receipt in snapshot.receipts.values()]
    report_id = _report_id(inputs, request)
    generator = build_generator_record(request.git_record)
    view = dashboard_view.build_dashboard_view(
        projection,
        report_id=report_id,
        generated_at=request.generated_at,
        input_evidence_classes=request.input_evidence_classes,
        generator=generator,
    )
    html = render.render_dashboard_bytes(view)
    projection_bytes = _json_bytes(projection)
    outputs = {
        "projection.json": sha256(projection_bytes).hexdigest(),
        "report.html": sha256(html).hexdigest(),
    }
    record = manifest.build_manifest(
        report_kind=request.report_kind,
        evidence_class=request.evidence_class,
        input_evidence_classes=request.input_evidence_classes,
        generated_at=request.generated_at,
        data_window=_data_window(projection),
        source_watermarks=_source_watermarks(projection),
        root_map=dict(projection.get("root_map") or {}),
        inputs=inputs,
        projection=projection,
        generator=generator,
        outputs=outputs,
        csv_schemas={},
        degradations=projection.get("degradations") or [],
    )
    files = {
        "projection.json": projection_bytes,
        "report.html": html,
        "report_manifest.json": _json_bytes(record),
    }
    return {
        "report_id": report_id,
        "projection": projection,
        "manifest": record,
        "view": view,
        "files": files,
        "outputs": outputs,
    }


def _safe_discard(reports_root: Path | str, attempt_id: str) -> dict[str, bool]:
    """清理本次 staging；**不得掩盖原始异常**（残留会被审计判 ``unexpected_inside``）。

    返回 ``{"staging_discard_ok", "staging_residual"}``：``discard_ok`` 要求
    ``discard_attempt`` 未抛异常**且**该 attempt 路径确实不存在（``rmtree(ignore_errors=True)``
    本身不会报错，旧版“恒真”字段不可否证——B-05）；``residual`` 为存在性复核结果。
    """
    try:
        publish.discard_attempt(reports_root, attempt_id)
    except (publish.PublishError, OSError):
        pass
    target = Path(reports_root) / ".staging" / attempt_id
    residual = target.exists()
    return {"staging_discard_ok": not residual, "staging_residual": residual}


class StagingOwnership(NamedTuple):
    """本次调用对 staging 的所有权边界（只有 owner 可以清理，且只清理自己创建的层）。"""

    path: Path
    created_staging: bool
    created_root: bool


def _claim_staging(reports_root: Path | str, attempt_id: str) -> StagingOwnership:
    """**原子**占用本次 staging 根（``mkdir(exist_ok=False)``），返回所有权记录。

    所有权只有两个结果：创建成功 = 本调用拥有（可清理），``FileExistsError`` = 别人拥有
    （原样保留、直接报错）。不再通过异常文本推断所有权，也不会出现两个调用共用同一份快照。

    白名单依据：``readonly_audit.compile_whitelist`` 的尾部 ``/**`` 同时匹配目录自身，
    且 ``data/reports/p6/.staging`` 在 T2 的结构性父目录放行表内，故该 ``mkdir`` 在审计轮
    同样被允许（不需要改 T1 判据）。

    ``created_staging``/``created_root`` 分别记录**两层父目录是否由本调用创建**：收尾只回收
    自己创建的空层，调用前已存在的空目录一律不得删除（T2-R3-N1）。
    """
    root = Path(reports_root)
    created_root = not root.exists()
    staging_parent = root / ".staging"
    created_staging = not staging_parent.exists()
    staging_parent.mkdir(parents=True, exist_ok=True)
    target = staging_parent / attempt_id
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise publish.PublishError(
            f"attempt 已被占用（拒绝复用既有 staging，既有现场原样保留）：{target}"
        ) from exc
    return StagingOwnership(target, created_staging, created_root)


def _release_staging(
    ownership: StagingOwnership, reports_root: Path | str, attempt_id: str
) -> dict[str, bool]:
    """**唯一的释放点**：删 attempt → 回收本调用创建的空父层，返回清理证据。

    成功与失败路径都只调用一次。删除 attempt 之后不再访问同名路径（目录可能已被新调用
    原子取得），父层回收用 ``rmdir`` 只删**空**目录，因此不会波及任何在用现场（T2-R3-01-R2）。
    """
    cleaned = _safe_discard(reports_root, attempt_id)
    if ownership.created_staging:
        try:
            ownership.path.parent.rmdir()
        except OSError:
            pass
    if ownership.created_root:
        try:
            Path(reports_root).rmdir()
        except OSError:
            pass
    return cleaned


#: 制度重排 JSON 的根标识（父目录为受信根，文件名作相对键）
REGIME_ROOT_ID = "regime"


def _experiment_roots(
    request: ExperimentRequest, specs: Sequence[experiment_sources.ArmSpec]
) -> dict[str, Path]:
    """实验面显式根映射：每个臂一个根 + 可选制度 JSON 父目录。"""
    roots: dict[str, Path] = {spec.root_id: Path(spec.path) for spec in specs}
    if request.regime_json is not None:
        roots[REGIME_ROOT_ID] = Path(request.regime_json).parent
    return roots


def _capture_experiment(
    request: ExperimentRequest,
) -> tuple[StableSnapshot, dict[str, Any]]:
    """采集一次实验面稳定快照（逐臂固定/逐折文件 + 制度 JSON）。

    重试口径与 T1 一致（最多 2 次重试）：全部 attempt 目录都在 ``.staging/<attempt>/``
    下，精确写白名单的 ``attempt`` 绑定仍然成立。
    """
    specs, baseline = experiment_sources.build_arm_specs(request.arms, request.baseline)
    info: dict[str, Any] = {"baseline": baseline, "specs": specs, "plans": {}}
    report_file = Path(request.regime_json).name if request.regime_json is not None else None

    def reader(snapshot: StableSnapshot) -> None:
        info["plans"].clear()
        for spec in specs:
            info["plans"][spec.arm_id] = experiment_sources.capture_arm_files(snapshot, spec)
        if report_file is not None:
            snapshot.capture(REGIME_ROOT_ID, report_file)

    retries = snapshot_module.MAX_RETRIES
    reports_root = Path(request.reports_root)
    publish.assert_attempt_id(request.attempt_id)
    staging_root = reports_root / ".staging" / request.attempt_id
    if staging_root.resolve().parent != (reports_root / ".staging").resolve():
        raise publish.PublishError(
            f"staging 目录越出 .staging 根：{staging_root}（attempt={request.attempt_id!r}）"
        )
    snapshot_result, audit = capture_stable_snapshot(
        roots=_experiment_roots(request, specs),
        staging_root=staging_root,
        attempt_ids=[f"r{index + 1}" for index in range(retries + 1)],
        reader=reader,
        max_retries=retries,
    )
    info["_audit"] = audit
    return snapshot_result, info


def _experiment_root_map(request: ExperimentRequest, info: Mapping[str, Any]) -> dict[str, str]:
    """root_map 语义（只存 root_id 与语义，不写绝对路径）。"""
    root_map: dict[str, str] = {}
    for spec in info.get("specs") or []:
        label = "规范化 runs 批次根（只读副本）" if spec.source_kind == "runs" else "旧 WF 运行根（只读副本）"
        root_map[spec.root_id] = f"臂 {spec.arm_id}：{label}"
    if request.regime_json is not None:
        root_map[REGIME_ROOT_ID] = "显式 P5a 制度重排 JSON（只读副本）"
    return root_map


def _compose_experiment(
    snapshot: StableSnapshot, request: ExperimentRequest, info: Mapping[str, Any]
) -> dict[str, Any]:
    """组装实验 run（HTML + 五 CSV + projection + manifest）。"""
    specs = list(info.get("specs") or [])
    arms = [
        experiment_sources.load_arm(snapshot, spec, dict(info["plans"].get(spec.arm_id) or {}))
        for spec in specs
    ]
    regime_payload: Mapping[str, Any] | None = None
    regime_error: str | None = None
    if request.regime_json is not None:
        payload, reason = snapshot.read_json(REGIME_ROOT_ID, Path(request.regime_json).name)
        regime_error = reason
        regime_payload = payload if isinstance(payload, Mapping) else None
    built = experiment_projection.build_experiment_projection(
        baseline=str(info.get("baseline")),
        arms=arms,
        generated_at=request.generated_at,
        evidence_class=request.evidence_class,
        root_map=_experiment_root_map(request, info),
        regime_payload=regime_payload,
        regime_error=regime_error,
        splits=request.splits,
    )
    projection = built["projection"]
    inputs = [receipt.as_dict() for receipt in snapshot.receipts.values()]
    report_id = _report_id(inputs, request)
    generator = build_generator_record(request.git_record)
    # 行数守恒硬门：期望值来自源侧计数（臂数/源链行/源 summary 行/源归因行），
    # 不是"同一行集长度自证"；缺行/多行都在生成阶段失败（A-07）
    bundle = experiment_csv.build_csv_bundle(
        built["csv_rows"], expected_rows=built["csv_expected_rows"]
    )
    view = experiment_view.build_experiment_view(
        projection,
        report_id=report_id,
        generated_at=request.generated_at,
        input_evidence_classes=request.input_evidence_classes,
        generator=generator,
        csv_rows=built["csv_rows"],
    )
    html = render.render_report_bytes(view, report_kind="experiment")
    projection_bytes = _json_bytes(projection)
    outputs = {
        "projection.json": sha256(projection_bytes).hexdigest(),
        "report.html": sha256(html).hexdigest(),
        **dict(bundle["digests"]),
    }
    record = manifest.build_manifest(
        report_kind=request.report_kind,
        evidence_class=request.evidence_class,
        input_evidence_classes=request.input_evidence_classes,
        generated_at=request.generated_at,
        data_window=projection.get("data_window") or {},
        source_watermarks=projection.get("source_watermarks") or {},
        root_map=dict(projection.get("root_map") or {}),
        inputs=inputs,
        projection=projection,
        generator=generator,
        outputs=outputs,
        csv_schemas=dict(bundle["schemas"]),
        degradations=projection.get("degradations") or [],
    )
    files: dict[str, bytes] = {
        "projection.json": projection_bytes,
        "report.html": html,
        "report_manifest.json": _json_bytes(record),
    }
    files.update(dict(bundle["files"]))
    return {
        "report_id": report_id,
        "projection": projection,
        "manifest": record,
        "view": view,
        "files": files,
        "outputs": outputs,
        "csv_schemas": bundle["schemas"],
    }


def generate_experiment(request: ExperimentRequest) -> dict[str, Any]:
    """生成并（可选）原子发布一份实验报告 run。"""
    evidence.assert_input_evidence(request.evidence_class, request.input_evidence_classes)
    publish.assert_attempt_id(request.attempt_id)
    if request.report_kind not in IMPLEMENTED_KINDS:
        raise publish.PublishError(
            f"未接入的报告类型：{request.report_kind!r}（已实现 {IMPLEMENTED_KINDS}）"
        )
    if request.report_kind != "experiment":
        raise publish.PublishError(f"实验生成入口只接受 experiment，收到 {request.report_kind!r}")
    # 生成链拦截：静态扫描（别名/动态取用）+ 执行期哨兵（实际调用即失败，B-02）
    guard_report = experiment_guard.assert_generation_chain(
        extra_paths=(Path(PROJECT_ROOT) / render.ENTRYPOINT_PATH,)
    )
    ownership = _claim_staging(request.reports_root, request.attempt_id)
    started = time.perf_counter()
    target: Path | None = None
    try:
        with experiment_guard.forbid_bootstrap_calls() as sentinel:
            snapshot, info = _capture_experiment(request)
            composed = _compose_experiment(snapshot, request, info)
        composed["bootstrap_guard"] = {
            "scanned": guard_report["scanned"],
            "scope": guard_report.get("scope"),
            "sentinel": sentinel,
        }
        if request.publish_run:
            target = publish.publish_new_run(
                reports_root=request.reports_root,
                evidence_class=request.evidence_class,
                report_kind=request.report_kind,
                report_id=composed["report_id"],
                files=composed["files"],
                attempt_id=request.attempt_id,
            )
    except BaseException:
        _release_staging(ownership, request.reports_root, request.attempt_id)
        raise
    cleaned = _release_staging(ownership, request.reports_root, request.attempt_id)
    return {
        **composed,
        "report_path": target,
        "snapshot_audit": info.get("_audit"),
        "cleanup": {
            "staging_discard_ok": bool(cleaned["staging_discard_ok"]),
            "staging_residual": bool(cleaned["staging_residual"]),
            "attempt_id": request.attempt_id,
        },
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


#: 趋势面来源根的语义标签（root_map 只存 root_id 与语义，不写绝对路径）
_TREND_ROOT_LABELS: dict[str, str] = {
    "runs": "规范化 runs 来源根（只读副本）",
    "legacy_wf": "旧 WF 运行根（只读副本）",
    "p4_paper": "v2 纸面生产根（只读副本）",
    "p5b_bundle": "P5b sealed bundle（calibration_only，只读副本）",
}


def _trend_roots(
    request: TrendRequest, specs: Sequence[trend_sources.SourceSpec]
) -> dict[str, Path]:
    """趋势面显式根映射：每个来源根 + 可选上一快照根。"""
    roots: dict[str, Path] = {spec.root_id: Path(spec.path) for spec in specs}
    if request.previous is not None:
        roots[trend_sources.PREVIOUS_ROOT_ID] = Path(request.previous)
    return roots


def _trend_root_map(
    request: TrendRequest,
    specs: Sequence[trend_sources.SourceSpec],
    previous_root_map: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """root_map（覆盖输出行引用的全部 root_id：本次来源 ∪ 上一快照 ∪ ``prev``）。

    继承行保留上一快照的 ``source_ref``（含其 root_id），因此必须把上一快照的
    root_map 一并继承，否则冻结包内出现“引用了未登记根”的断裂（T4-R1-02）。
    """
    mapping: dict[str, str] = {}
    for spec in specs:
        mapping[spec.root_id] = _TREND_ROOT_LABELS.get(spec.kind, "趋势来源根（只读副本）")
    if request.previous is not None:
        mapping[trend_sources.PREVIOUS_ROOT_ID] = "上一不可变趋势快照（只读副本）"
    for key, value in dict(previous_root_map or {}).items():
        mapping.setdefault(str(key), str(value))
    return mapping


def _assert_trend_root_refs(
    index_rows: Sequence[Mapping[str, Any]],
    trend_rows: Sequence[Mapping[str, Any]],
    root_map: Mapping[str, str],
) -> None:
    """fail-closed：输出行引用的 root_id 必须在 root_map 中登记（否则不可回溯）。"""
    referenced = {
        str(row.get("source_ref")).split(":", 1)[0]
        for row in list(index_rows) + list(trend_rows)
        if row.get("source_ref")
    }
    unmapped = sorted(referenced - set(root_map))
    if unmapped:
        raise trend_sources.TrendSourceError(
            f"输出引用了未登记的 root_id：{unmapped}（root_map 未覆盖继承来源）"
        )


def _capture_trend(request: TrendRequest) -> tuple[StableSnapshot, dict[str, Any]]:
    """采集一次趋势面稳定快照（上一快照 + 各类来源根的身份/指标文件）。"""
    specs = trend_sources.build_sources(
        request.evidence_class,
        raw_sources=request.sources,
        paper_root=request.paper_root,
        bundles=request.bundles,
    )
    info: dict[str, Any] = {"specs": specs, "plan": None, "previous": None, "fresh": []}

    def reader(snapshot: StableSnapshot) -> None:
        info["plan"] = None
        info["previous"] = None
        info["fresh"] = []
        previous = None
        if request.previous is not None:
            previous = trend_sources.load_previous_snapshot(
                snapshot, evidence_class=request.evidence_class
            )
        plan = trend_sources.plan_and_capture(snapshot, request.evidence_class, specs)
        inherited = (
            trend_sources.previous_run_keys(previous["index_rows"]) if previous else set()
        )
        info["previous"] = previous
        info["plan"] = plan
        info["fresh"] = trend_sources.capture_new_run_files(snapshot, plan, inherited)
        if request.evidence_class == "production":
            # 与 run 数量无关的权威门：零 run 时也要能判定“权威缺失”，而非直接合法空
            from src.lazybull.v2.reporting import trend_observe

            info["authority_gate"] = trend_observe._authority_record_gate(snapshot, plan)

    retries = snapshot_module.MAX_RETRIES
    reports_root = Path(request.reports_root)
    publish.assert_attempt_id(request.attempt_id)
    staging_root = reports_root / ".staging" / request.attempt_id
    if staging_root.resolve().parent != (reports_root / ".staging").resolve():
        raise publish.PublishError(
            f"staging 目录越出 .staging 根：{staging_root}（attempt={request.attempt_id!r}）"
        )
    snapshot_result, audit = capture_stable_snapshot(
        roots=_trend_roots(request, specs),
        staging_root=staging_root,
        attempt_ids=[f"r{index + 1}" for index in range(retries + 1)],
        reader=reader,
        max_retries=retries,
    )
    info["_audit"] = audit
    return snapshot_result, info


def _previous_run_dir(
    request: TrendRequest, previous: Mapping[str, Any]
) -> dict[str, Any]:
    """上一快照 run 目录的位置登记（仓库内给相对 POSIX；仓库外不写绝对路径）。"""
    report_id = previous.get("report_id")
    if not report_id:
        return {"run_dir": None, "run_dir_present": False, "inside_repo": False}
    candidate = Path(request.previous)
    # ``--previous`` 既可为 run 目录（含 manifest，可能被复制/改名）也可为集合根：
    # 只按“是否有 manifest”判定，不按目录名猜（目录名≠report_id 时不得拼错路径）。
    if not (candidate / "report_manifest.json").is_file():
        candidate = candidate / request.evidence_class / "trend" / str(report_id)
    present = candidate.is_dir()
    try:
        relative = candidate.resolve().relative_to(Path(PROJECT_ROOT).resolve()).as_posix()
    except (OSError, ValueError):
        return {
            "run_dir": None,
            "run_dir_present": present,
            "inside_repo": False,
            "note": "上一快照不在仓库内（临时/外部根）：不写绝对路径，仅登记存在性",
        }
    return {"run_dir": relative, "run_dir_present": present, "inside_repo": True}


def _compose_trend(
    snapshot: StableSnapshot, request: TrendRequest, info: Mapping[str, Any]
) -> dict[str, Any]:
    """组装趋势 run（HTML + run_index.csv + trend.csv + projection + manifest）。"""
    previous = info.get("previous")
    plan = dict(info.get("plan") or {})
    observations = (
        trend_sources.build_observations(snapshot, plan, list(info.get("fresh") or []))
        if info.get("fresh")
        else []
    )
    previous_index = [] if previous is None else list(previous.get("index_rows") or [])
    previous_trend = [] if previous is None else list(previous.get("trend_rows") or [])
    index_rows = trend_projection.merge_rows(
        previous_index, [item.index_row for item in observations], key_fields=("run_key",)
    )
    trend_rows = trend_projection.merge_rows(
        previous_trend,
        [row for item in observations for row in item.metric_rows],
        key_fields=("run_key", "metric_name"),
    )
    specs = list(info.get("specs") or [])
    previous_root_map = {} if previous is None else dict(previous.get("root_map") or {})
    root_map = _trend_root_map(request, specs, previous_root_map)
    if previous is not None:
        try:
            trend_projection.assert_no_identity_conflict(
                previous_index, [item.index_row for item in observations]
            )
        except ValueError as exc:
            raise trend_sources.TrendSourceError(str(exc)) from exc
    _assert_trend_root_refs(index_rows, trend_rows, root_map)
    current_roots = {spec.root_id for spec in specs}
    inherited_roots = sorted(set(previous_root_map) - current_roots - {"prev"})
    inherited_source = (
        None
        if previous is None
        else {
            "report_id": previous.get("report_id"),
            "report_manifest_sha256": previous.get("manifest_sha256"),
            "input_manifest_sha256": previous.get("input_manifest_sha256"),
            "previous": _previous_run_dir(request, previous),
            "note": (
                "继承行的来源字节由上一冻结包的 manifest/inputs 与本包继承的 root_map 共同封存；"
                "previous.run_dir 为上一快照的仓库相对位置（仓库外只登记存在性）"
            ),
        }
    )
    projection = trend_projection.build_trend_projection(
        evidence_class=request.evidence_class,
        generated_at=request.generated_at,
        root_map=root_map,
        index_rows=index_rows,
        trend_rows=trend_rows,
        observations=observations,
        previous=previous,
        inherited_roots=inherited_roots,
        inherited_source=inherited_source,
        authority_gate=(
            info.get("authority_gate") if request.evidence_class == "production" else None
        ),
    )
    inputs = [receipt.as_dict() for receipt in snapshot.receipts.values()]
    report_id = _report_id(inputs, request)
    generator = build_generator_record(request.git_record)
    bundle = trend_csv.build_trend_bundle(
        index_rows, trend_rows, expected_index_rows=len(index_rows)
    )
    view = trend_view.build_trend_view(
        projection,
        report_id=report_id,
        generated_at=request.generated_at,
        input_evidence_classes=request.input_evidence_classes,
        generator=generator,
    )
    html = render.render_report_bytes(view, report_kind="trend")
    projection_bytes = _json_bytes(projection)
    outputs = {
        "projection.json": sha256(projection_bytes).hexdigest(),
        "report.html": sha256(html).hexdigest(),
        **dict(bundle["digests"]),
    }
    record = manifest.build_manifest(
        report_kind=request.report_kind,
        evidence_class=request.evidence_class,
        input_evidence_classes=request.input_evidence_classes,
        generated_at=request.generated_at,
        data_window=projection.get("data_window") or {},
        source_watermarks=projection.get("source_watermarks") or {},
        root_map=dict(projection.get("root_map") or {}),
        inputs=inputs,
        projection=projection,
        generator=generator,
        outputs=outputs,
        csv_schemas=dict(bundle["schemas"]),
        degradations=projection.get("degradations") or [],
    )
    files: dict[str, bytes] = {
        "projection.json": projection_bytes,
        "report.html": html,
        "report_manifest.json": _json_bytes(record),
    }
    files.update(dict(bundle["files"]))
    return {
        "report_id": report_id,
        "projection": projection,
        "manifest": record,
        "view": view,
        "files": files,
        "outputs": outputs,
        "csv_schemas": bundle["schemas"],
    }


def generate_trend(request: TrendRequest) -> dict[str, Any]:
    """生成并（可选）原子发布一份趋势库 run（三类证据域物理隔离）。"""
    evidence.assert_input_evidence(request.evidence_class, request.input_evidence_classes)
    publish.assert_attempt_id(request.attempt_id)
    if request.report_kind != "trend":
        raise publish.PublishError(f"趋势生成入口只接受 trend，收到 {request.report_kind!r}")
    ownership = _claim_staging(request.reports_root, request.attempt_id)
    started = time.perf_counter()
    target: Path | None = None
    try:
        snapshot, info = _capture_trend(request)
        composed = _compose_trend(snapshot, request, info)
        if request.publish_run:
            target = publish.publish_new_run(
                reports_root=request.reports_root,
                evidence_class=request.evidence_class,
                report_kind=request.report_kind,
                report_id=composed["report_id"],
                files=composed["files"],
                attempt_id=request.attempt_id,
            )
    except BaseException:
        _release_staging(ownership, request.reports_root, request.attempt_id)
        raise
    cleaned = _release_staging(ownership, request.reports_root, request.attempt_id)
    return {
        **composed,
        "report_path": target,
        "snapshot_audit": info.get("_audit"),
        "cleanup": {
            "staging_discard_ok": bool(cleaned["staging_discard_ok"]),
            "staging_residual": bool(cleaned["staging_residual"]),
            "attempt_id": request.attempt_id,
        },
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def generate_report(request: Any) -> dict[str, Any]:
    """按请求类型分派（驾驶舱/实验/趋势；其它类型 fail-closed）。"""
    if isinstance(request, DashboardRequest):
        return generate_dashboard(request)
    if isinstance(request, ExperimentRequest):
        return generate_experiment(request)
    if isinstance(request, TrendRequest):
        return generate_trend(request)
    raise publish.PublishError(f"未支持的请求类型：{type(request).__name__}")


def generate_dashboard(request: DashboardRequest) -> dict[str, Any]:
    """生成并（可选）原子发布一份驾驶舱 run。

    采集、组装与发布**全部**在同一个 ``try`` 内：任何一步失败都清理**本调用占用**的 staging
    （含快照失败留下的 ``.staging/<attempt>/`` 父目录），不留半成品；既有 attempt 目录在被
    占用前就已经拒绝，绝不会被本调用的清理波及。

    成功返回的 ``cleanup.staging_discard_ok`` 是**成功路径**的清理证据字段（失败路径以异常
    本身为证据，不再包装返回载荷）。
    """
    evidence.assert_input_evidence(request.evidence_class, request.input_evidence_classes)
    publish.assert_attempt_id(request.attempt_id)
    if request.report_kind != "dashboard":
        raise publish.PublishError(
            f"驾驶舱生成入口只接受 dashboard；report_kind={request.report_kind!r} 必须走"
            f"对应生成器（已接入 {IMPLEMENTED_KINDS}，趋势/索引归 T4），拒绝以驾驶舱内容冒充"
        )
    if request.three_ends is not None:
        raise publish.PublishError(
            "本阶段不接受额外三端输入（不进入封存包、无法包内复算）：三端身份一律取自 "
            "projection（T2-R3-03）"
        )
    # 所有权：只有 mkdir 成功的调用可以清理；冲突调用在进入 try 之前就退出（不清理）
    ownership = _claim_staging(request.reports_root, request.attempt_id)
    started = time.perf_counter()
    target: Path | None = None
    try:
        snapshot, info = _capture(request)
        composed = _compose(snapshot, request, info)
        if request.publish_run:
            target = publish.publish_new_run(
                reports_root=request.reports_root,
                evidence_class=request.evidence_class,
                report_kind=request.report_kind,
                report_id=composed["report_id"],
                files=composed["files"],
                attempt_id=request.attempt_id,
            )
    except BaseException:
        _release_staging(ownership, request.reports_root, request.attempt_id)
        raise
    # 成功路径同样只释放一次：先删 attempt 再回头删同名目录会误伤新所有者（T2-R3-01-R2）
    cleaned = _release_staging(ownership, request.reports_root, request.attempt_id)
    return {
        **composed,
        "report_path": target,
        "snapshot_audit": info.get("_audit"),
        "cleanup": {
            "staging_discard_ok": bool(cleaned["staging_discard_ok"]),
            "staging_residual": bool(cleaned["staging_residual"]),
            "attempt_id": request.attempt_id,
        },
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
