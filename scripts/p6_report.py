# -*- coding: utf-8 -*-
"""P6 报告生成入口（只读、离线、确定性；T2 只含 ``dashboard``）。

用法::

    python scripts/p6_report.py generate --evidence-class research \\
        --paper-root data/paper --quality-root data/reports/quality \\
        --audit-out data/reports/p6/validations/research/<new_id>

写域与证据（设计 §2.5）：

- 允许写：本次 ``.staging/<attempt>/**`` 与最终 ``<evidence>/<kind>/<report_id>/**``；
  ``--audit-out`` 的 ``readonly_audit.json`` 落在**本次新 validation 目录**，且在审计判定
  之后写入（不属于 generate 的写集）。
- 带 ``--audit-out`` 时启用运行期 ``ReadonlyGuard``（写/mkdir/rename/子进程/网络拦截），
  绑定 ``attempt/evidence_class/kind`` 三个字面量；``report_id`` 需要输入哈希才能确定，
  其**精确集合**判定由 ``publish.audit_generate``（T1 ``audit_command``）承担——
  守卫是形状级兜底，审计是判定性证据，两者职责不同。
- 不带 ``--audit-out`` 时（性能计时/本地探索）不启用守卫，写域仍受 `publish_new_run`
  的“新目录 + 拒绝覆盖”约束；此类运行**不作为写域证据**。
- 证据域混合（``production`` 收 research/calibration 输入等）⇒ 非零退出，不生成混合报告。

退出码：``0`` 成功 / ``3`` 判据失败（证据域混合、写域审计失败、发布冲突、**快照协议失败**：
受信根不合格或读取期持续漂移、运行期只读守卫拦截、**文件系统失败** `stage=file_system`）
/ ``2`` 设置错误（路径不存在、validation 目录已存在、``--attempt-id`` 形态非法等参数问题）。
"""

# 复杂度豁免：阶段入口脚本（scripts/p6_report.py 不进生产训练链路，见 CLAUDE.md §7.5）

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import uuid
from hashlib import sha256

if __name__ == "__main__":  # 验收进程禁写 __pycache__（设计 §2.5）；pytest 导入时不改全局
    sys.dont_write_bytecode = True
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare.fold_subset import parse_split_spec  # noqa: E402
from src.lazybull.v2.reporting import (  # noqa: E402
    contracts,
    evidence,
    generate,
    publish,
    readonly_audit,
    validation,
)
from src.lazybull.v2.reporting.experiment_sources import ArmSourceError  # noqa: E402
from src.lazybull.v2.reporting.snapshot import SnapshotError  # noqa: E402
from src.lazybull.v2.reporting.trend_sources import TrendSourceError  # noqa: E402

__all__ = ["EXIT_CODES", "SetupError", "git_record", "main", "run_generate"]

EXIT_CODES = {"pass": 0, "policy_fail": 3, "setup_error": 2}
#: 默认纸面生产根 / 质量根
DEFAULT_PAPER_ROOT = "data/paper"
DEFAULT_QUALITY_ROOT = "data/reports/quality"


class SetupError(RuntimeError):
    """命令行前置条件不满足（路径形态、新目录、审计根不匹配）。"""


def _git(arguments: Sequence[str]) -> str:
    """执行 git 并返回标准输出（缺失 git 时返回空，不伪造值）。"""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return ""
    return completed.stdout or ""


def git_record() -> dict[str, Any]:
    """生成器 git 记录（只登记事实；取不到写 ``<unavailable>``）。"""
    commit = _git(["rev-parse", "HEAD"]).strip()
    status = _git(["status", "--porcelain"]).strip()
    return {"git_commit": commit or "<unavailable>", "git_dirty": bool(status)}


def _assert_canonical_reports_root(reports_root: Path) -> None:
    """带审计的生成必须写 repo 内的 ``data/reports/p6``（审计按 repo 相对路径判定）。"""
    expected = (ROOT / contracts.REPORTS_ROOT).resolve()
    if reports_root.resolve() != expected:
        raise SetupError(
            f"带审计的生成只允许写 {expected}（收到 {reports_root.resolve()}）："
            "精确写审计按 repo 相对路径判定，其他根无法取证"
        )


def _assert_new_validation_dir(audit_out: Path, evidence_class: str) -> None:
    """``--audit-out`` 必须是 ``validations/<class>/<new_id>`` 且不存在。"""
    validations_root = (ROOT / contracts.REPORTS_ROOT / "validations").resolve()
    target = audit_out.resolve()
    try:
        relative = target.relative_to(validations_root)
    except ValueError as exc:
        raise SetupError(f"审计输出目录不在 validation 根下：{target}") from exc
    parts = relative.parts
    if len(parts) != 2 or parts[0] != evidence_class:
        raise SetupError(
            "审计输出目录形态须为 <evidence_class>/<validation_id> 且与 --evidence-class 一致："
            f"{relative}"
        )
    validation.validate_validation_id(parts[1])
    if target.exists():
        raise SetupError(f"validation 目录已存在，拒绝覆盖：{target}")


#: 运行期守卫额外允许的**结构性父目录**（只允许建目录本身，不承载报告字节）：
#: ``Path.mkdir(parents=True)`` 会对每一层父目录发起 ``os.mkdir`` 审计事件，
#: 而契约白名单只覆盖 ``.staging/<attempt>/**`` 与 ``<evidence>/<kind>/<report_id>/**``。
#: 这三个目录不进入文件差异集合，判定性判据仍是 ``audit_command`` 的精确文件集合。
_STRUCTURAL_DIRS: tuple[str, ...] = (
    "data/reports/p6/.staging",
    "data/reports/p6/<evidence_class>",
    "data/reports/p6/<evidence_class>/<kind>",
)


@contextlib.contextmanager
def _guard(evidence_class: str, kind: str, attempt: str) -> Iterator[None]:
    """运行期只读守卫（形状级兜底；精确集合由 ``audit_generate`` 判定）。"""
    with readonly_audit.ReadonlyGuard(
        allowed_patterns=tuple(contracts.WRITE_WHITELIST["generate"]) + _STRUCTURAL_DIRS,
        worktree_root=ROOT,
        bindings={"attempt": attempt, "evidence_class": evidence_class, "kind": kind},
        deny_network=True,
        deny_subprocess=True,
    ):
        yield


def _evidence_class(args: argparse.Namespace) -> str:
    """证据域默认值：驾驶舱 production、实验 research；趋势面**必须显式**（三类互斥）。"""
    if args.evidence_class:
        return str(args.evidence_class)
    if args.kind == "trend":
        raise SetupError("趋势面必须显式给出 --evidence-class（production/research/calibration_only）")
    return "research" if args.kind == "experiment" else "production"


def _input_classes(args: argparse.Namespace, evidence_class: str) -> tuple[str, ...]:
    """输入证据类：只有**未给出**才回退为证据域本身（显式空串由隔离门拒绝）。"""
    raw_classes = (
        evidence_class if args.input_evidence_classes is None else args.input_evidence_classes
    )
    return tuple(item.strip() for item in raw_classes.split(",") if item.strip())


def _experiment_request(
    args: argparse.Namespace, *, generated_at: str, attempt: str
) -> generate.ExperimentRequest:
    """把命令行参数转成显式实验生成请求（臂清单与基线必须显式）。"""
    evidence_class = _evidence_class(args)
    if not args.arm:
        raise SetupError("实验报告必须显式给出至少一个 --arm arm_id=kind:path")
    if not args.baseline:
        raise SetupError("实验报告必须显式给出 --baseline（基线臂 id）")
    splits = tuple(parse_split_spec(args.splits)) if args.splits else None
    regime = Path(args.regime_json).resolve() if args.regime_json else None
    if regime is not None and not regime.is_file():
        raise SetupError(f"制度重排 JSON 不存在或不是文件：{regime}")
    return generate.ExperimentRequest(
        reports_root=Path(args.reports_root).resolve(),
        evidence_class=evidence_class,
        generated_at=generated_at,
        attempt_id=attempt,
        input_evidence_classes=_input_classes(args, evidence_class),
        arms=tuple(str(item) for item in args.arm),
        baseline=str(args.baseline),
        regime_json=regime,
        splits=splits,
        git_record=git_record(),
        publish_run=not args.no_publish,
    )


def _trend_request(
    args: argparse.Namespace, *, generated_at: str, attempt: str
) -> generate.TrendRequest:
    """把命令行参数转成显式趋势生成请求（输入白名单在 ``build_sources`` 内判定）。"""
    evidence_class = _evidence_class(args)
    previous = Path(args.previous).resolve() if args.previous else None
    if previous is not None and not previous.is_dir():
        raise SetupError(f"上一快照目录不存在或不是目录：{previous}")
    paper_root = Path(args.paper_root).resolve() if args.paper_root else None
    return generate.TrendRequest(
        reports_root=Path(args.reports_root).resolve(),
        evidence_class=evidence_class,
        generated_at=generated_at,
        attempt_id=attempt,
        input_evidence_classes=_input_classes(args, evidence_class),
        sources=tuple(str(item) for item in (args.source or ())),
        paper_root=paper_root,
        bundles=tuple(str(item) for item in (args.bundle or ())),
        previous=previous,
        git_record=git_record(),
        publish_run=not args.no_publish,
    )


def _request(
    args: argparse.Namespace, *, generated_at: str, attempt: str
) -> generate.DashboardRequest:
    """把命令行参数转成显式生成请求（不在别处补默认值）。

    ``--input-evidence-classes`` 只有**未给出**（``None``）才回退到 ``--evidence-class``；
    显式空串会得到空元组并由证据域隔离门报错（禁止静默改写为空声明豁免）。
    """
    quality_root = None if args.no_quality else Path(args.quality_root).resolve()
    evidence_class = _evidence_class(args)
    input_classes = _input_classes(args, evidence_class)
    paper_default = Path(args.paper_root) if args.paper_root else Path(DEFAULT_PAPER_ROOT)
    return generate.DashboardRequest(
        paper_root=paper_default.resolve(),
        quality_root=quality_root,
        reports_root=Path(args.reports_root).resolve(),
        evidence_class=evidence_class,
        generated_at=generated_at,
        attempt_id=attempt,
        input_evidence_classes=input_classes,
        report_kind=args.kind,
        git_record=git_record(),
        publish_run=not args.no_publish,
    )


def _assert_trend_inputs(request: generate.TrendRequest) -> None:
    """趋势面输入根形态检查（白名单判定在 ``trend_sources.build_sources`` 内）。"""
    for raw in request.sources:
        path = Path(str(raw).split(":", 1)[-1])
        if not path.is_dir():
            raise SetupError(f"趋势来源根不存在或不是目录：{path}")
    for bundle in request.bundles:
        if not Path(bundle).is_dir():
            raise SetupError(f"P5b bundle 目录不存在或不是目录：{bundle}")
    if request.paper_root is not None and not Path(request.paper_root).is_dir():
        raise SetupError(f"纸面根不存在或不是目录：{request.paper_root}")


def _attempt_id(args: argparse.Namespace, generated_at: str) -> str:
    """attempt_id：显式给出即用（空串也拒绝），否则按时间 + 短随机生成。"""
    attempt = (
        f"gen_{generated_at.lower().replace('-', '')}_{uuid.uuid4().hex[:6]}"
        if args.attempt_id is None
        else args.attempt_id
    )
    try:
        publish.assert_attempt_id(attempt)
    except publish.PublishError as exc:
        raise SetupError(str(exc)) from exc
    return attempt


def _extra_roots_for(request: Any) -> dict[str, Path]:
    """repo 之外的输入根（其任何变化都会被审计判为 foreign root）。"""
    repo = ROOT.resolve()
    roots: dict[str, Path] = {}

    def _add(name: str, path: Path) -> None:
        try:
            Path(path).resolve().relative_to(repo)
        except ValueError:
            roots[name] = Path(path).resolve()

    if isinstance(request, generate.TrendRequest):
        for index, raw in enumerate(request.sources):
            _add(f"trend_source_{index}", Path(str(raw).split(":", 1)[-1]))
        for index, bundle in enumerate(request.bundles):
            _add(f"trend_bundle_{index}", Path(bundle))
        if request.paper_root is not None:
            _add("trend_paper_input", request.paper_root)
        if request.previous is not None:
            _add("trend_previous_input", request.previous)
        return roots
    if hasattr(request, "paper_root"):
        _add("paper_input", request.paper_root)
        return roots
    for index, spec in enumerate(getattr(request, "arms", ()) or ()):
        raw = str(spec).split(":", 1)[-1]
        _add(f"arm_input_{index}", Path(raw))
    regime = getattr(request, "regime_json", None)
    if regime is not None:
        _add("regime_input", Path(regime).parent)
    return roots


def _run_requested(request: Any) -> Any:
    """按请求类型调用生成器（驾驶舱/实验同一入口）。"""
    return generate.generate_report(request)


def _execute(request: Any) -> tuple[dict[str, Any] | None, Any]:
    """执行生成并把登记内异常收敛为 ``(失败载荷, None)``（不抛裸 traceback）。"""
    try:
        return None, _run_requested(request)
    except evidence.EvidenceIsolationError as exc:
        return {"stage": "evidence_isolation", "detail": str(exc)}, None
    except TrendSourceError as exc:
        # 输入白名单/跨类/身份不可用：策略失败（不生成“有效混合报告”）
        return {"stage": "trend_source", "detail": str(exc)}, None
    except SnapshotError as exc:
        return {"stage": "stable_snapshot", "detail": f"{type(exc).__name__}: {exc}"}, None
    except readonly_audit.ReadonlyViolationError as exc:
        # 运行期只读守卫拦截（写域白名单外/网络/子进程）：与“写域审计失败 → 3”口径一致
        return {"stage": "readonly_guard", "detail": f"{type(exc).__name__}: {exc}"}, None
    except OSError as exc:
        # 文件系统级失败（权限/路径过长/盘满/设备名等）：结构性兜底，仍按登记退出码返回
        return {"stage": "file_system", "detail": f"{type(exc).__name__}: {exc}"}, None


def run_generate(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """执行一次 generate（可选审计）；返回 ``(退出码, 结果载荷)``。"""
    generated_at = args.generated_at or validation.utc_stamp()
    # attempt_id 走 T1 快照的 ROOT_ID_PATTERN（小写字母开头、仅小写/数字/下划线、<=32）；
    # 参数形态问题归 ``2``（SetupError），快照协议失败才归 ``3``（T2-R1-N6 / R2-N1）
    # 显式空串不得静默回退成随机 id（与 --input-evidence-classes 同口径：只有 None 才回退）
    attempt = _attempt_id(args, generated_at)
    evidence_class = _evidence_class(args)
    if args.kind == "experiment":
        request = _experiment_request(args, generated_at=generated_at, attempt=attempt)
    elif args.kind == "trend":
        request = _trend_request(args, generated_at=generated_at, attempt=attempt)
    else:
        request = _request(args, generated_at=generated_at, attempt=attempt)
    if args.kind == "dashboard":
        if not request.paper_root.is_dir():
            raise SetupError(f"纸面根不存在或不是目录：{request.paper_root}")
        if request.quality_root is not None and not request.quality_root.is_dir():
            raise SetupError(f"质量根不存在或不是目录：{request.quality_root}")
    if isinstance(request, generate.TrendRequest):
        _assert_trend_inputs(request)
    audit_out = Path(args.audit_out).resolve() if args.audit_out else None
    before = None
    if audit_out is not None:
        _assert_canonical_reports_root(request.reports_root)
        _assert_new_validation_dir(audit_out, evidence_class)
        try:
            before = publish.scan_audit_roots(
                repo_root=ROOT, extra_roots=_extra_roots_for(request)
            )
        except (OSError, validation.ValidationError) as exc:
            # 首次扫描与后扫同口径收敛：fail-closed 且返回登记退出码（R3-11）
            return EXIT_CODES["policy_fail"], {
                "verdict": "fail",
                "stage": "audit_scan",
                "detail": f"{type(exc).__name__}: {exc}",
                "evidence_class": evidence_class,
                "input_evidence_classes": list(request.input_evidence_classes),
                "attempt_id": request.attempt_id,
            }
    context = (
        _guard(evidence_class, args.kind, request.attempt_id)
        if audit_out is not None
        else contextlib.nullcontext()
    )
    payload_base = {
        "evidence_class": evidence_class,
        "input_evidence_classes": list(request.input_evidence_classes),
        "attempt_id": request.attempt_id,
    }
    with context:
        failure, result = _execute(request)
    if failure is not None:
        return EXIT_CODES["policy_fail"], {"verdict": "fail", **failure, **payload_base}
    payload: dict[str, Any] = {
        "verdict": "pass",
        "report_id": result["report_id"],
        "report_path": str(result["report_path"]) if result["report_path"] else None,
        "generated_at": request.generated_at,
        "attempt_id": request.attempt_id,
        "evidence_class": evidence_class,
        "report_kind": args.kind,
        "outputs": result["outputs"],
        # manifest 自身不在 outputs（自封存），单独登记字节哈希供“八件一致”判定（B-06）
        "manifest_sha256": sha256(result["files"]["report_manifest.json"]).hexdigest(),
        "file_bytes": {name: len(data) for name, data in sorted(result["files"].items())},
        "elapsed_ms": round(result["elapsed_ms"], 1),
        "cleanup": dict(result.get("cleanup") or {}),
        "git": dict(request.git_record or {}),
        "excluded_boxes": list(publish.AUDIT_SCAN_EXCLUDES),
    }
    if audit_out is None or before is None:
        return EXIT_CODES["pass"], payload
    return _finish_with_audit(
        payload=payload,
        before=before,
        audit_out=audit_out,
        extra_roots=_extra_roots_for(request),
        evidence_class=evidence_class,
        report_kind=args.kind,
        report_id=result["report_id"],
        attempt=request.attempt_id,
    )


def _finish_with_audit(
    *,
    payload: dict[str, Any],
    before: Any,
    audit_out: Path,
    extra_roots: Mapping[str, Path],
    evidence_class: str,
    report_kind: str,
    report_id: str,
    attempt: str,
) -> tuple[int, dict[str, Any]]:
    """事后整树比对 + 精确写判定，并落 ``readonly_audit.json``。

    审计前后半段（重扫整树 + 写审计文件）也在登记退出码内：该处 ``OSError`` 归
    ``stage=audit_scan``/``audit_write`` 并返回 ``3``（T2 R6-1 的 fail-closed 补足，
    不产生假 pass，也不抛裸 traceback）。
    """
    try:
        after = publish.scan_audit_roots(repo_root=ROOT, extra_roots=extra_roots)
    except (OSError, validation.ValidationError) as exc:
        payload["verdict"] = "fail"
        payload["stage"] = "audit_scan"
        payload["detail"] = f"{type(exc).__name__}: {exc}"
        return EXIT_CODES["policy_fail"], payload
    expected = [
        f"{contracts.REPORTS_ROOT}/{evidence_class}/{report_kind}/{report_id}/{name}"
        for name in publish.RUN_FILENAMES_BY_KIND[report_kind]
    ]
    report = publish.audit_generate(
        before=before,
        after=after,
        expected_paths=expected,
        bindings={
            "attempt": attempt,
            "evidence_class": evidence_class,
            "kind": report_kind,
            "report_id": report_id,
        },
        output_root="worktree",
        repo_root=ROOT,
    )
    try:
        validation.ensure_new_dir(audit_out)
        validation.write_json_new(audit_out / "readonly_audit.json", report)
    except (OSError, validation.ValidationError, publish.PublishError) as exc:
        payload["verdict"] = "fail"
        payload["stage"] = "audit_write"
        payload["detail"] = f"{type(exc).__name__}: {exc}"
        return EXIT_CODES["policy_fail"], payload
    payload["audit"] = {
        "path": (audit_out / "readonly_audit.json").as_posix(),
        "verdict": report.get("verdict"),
        "report_status": report.get("report_status"),
    }
    # 把判定性字段直接回读进载荷（单一载荷即可核验，不必再翻审计文件）
    scan_scope = dict(report.get("scan_scope") or {})
    payload["audit_evidence"] = {
        "verdict": report.get("verdict"),
        "uncovered": list(scan_scope.get("uncovered") or []),
        "reparse_checked": scan_scope.get("reparse_checked"),
        "reparse_points": list(scan_scope.get("reparse_points") or []),
        "missing_expected": list(report.get("missing_expected") or []),
        "unexpected_inside": list(report.get("unexpected_inside") or []),
        "outside_whitelist": list(report.get("outside_whitelist") or []),
        "foreign_root_paths": list(report.get("foreign_root_paths") or []),
        "overwrote_existing": list(report.get("overwrote_existing") or []),
    }
    # 保留 T2 冻结键名 ``scan_scope``（描述性）；判定性字段见 ``audit_evidence``
    payload["scan_scope"] = {
        "repo_root": str(ROOT),
        "includes": list(publish.AUDIT_SCAN_INCLUDES),
        "note": "扫描域为有界子树；数十 GB 级历史/临时树由运行期只读守卫承担",
    }
    if report.get("verdict") != "pass":
        payload["verdict"] = "fail"
        return EXIT_CODES["policy_fail"], payload
    return EXIT_CODES["pass"], payload


def build_parser() -> argparse.ArgumentParser:
    """命令行解析器（三类均已接入：``dashboard`` / ``experiment`` / ``trend``）。"""
    parser = argparse.ArgumentParser(description="P6 报告生成（只读、离线、确定性）")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("generate", help="生成并发布一份报告 run")
    command.add_argument("--kind", default="dashboard", choices=list(generate.IMPLEMENTED_KINDS))
    command.add_argument(
        "--evidence-class", default=None, choices=list(contracts.EVIDENCE_CLASSES)
    )
    command.add_argument(
        "--paper-root", default=None, help=f"纸面根（驾驶舱默认 {DEFAULT_PAPER_ROOT}；趋势面必须显式）"
    )
    command.add_argument("--quality-root", default=DEFAULT_QUALITY_ROOT)
    command.add_argument("--no-quality", action="store_true", help="不读质量长表")
    command.add_argument("--reports-root", default=contracts.REPORTS_ROOT)
    command.add_argument("--generated-at", default=None, help="UTC YYYYMMDDTHHMMSSZ")
    command.add_argument("--attempt-id", default=None)
    command.add_argument(
        "--input-evidence-classes",
        default=None,
        help="逗号分隔的输入证据类（默认与 --evidence-class 相同）",
    )
    command.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="ARM_ID=KIND:PATH",
        help="实验臂（可重复；KIND ∈ runs|legacy_wf）",
    )
    command.add_argument("--baseline", default=None, help="实验基线臂 id（必须出现在 --arm 内）")
    command.add_argument("--regime-json", default=None, help="显式 P5a 制度重排 JSON")
    command.add_argument("--splits", default=None, help="折子集（如 0-5,7；默认全部可用折）")
    command.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="KIND:PATH",
        help="趋势来源（可重复；KIND ∈ runs|legacy_wf，仅 research 面）",
    )
    command.add_argument(
        "--bundle",
        action="append",
        default=None,
        metavar="PATH",
        help="P5b sealed bundle 目录（可重复；仅 calibration_only 面）",
    )
    command.add_argument(
        "--previous", default=None, help="上一不可变趋势 run 目录（增量生成；同证据类）"
    )
    command.add_argument("--audit-out", default=None, help="本次新 validation 目录")
    command.add_argument("--no-publish", action="store_true")
    command.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口；退出码见模块 docstring。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code, payload = run_generate(args)
    except (SetupError, ArmSourceError) as exc:
        print(f"[P6] 设置错误：{exc}", file=sys.stderr)
        return EXIT_CODES["setup_error"]
    except evidence.EvidenceIsolationError as exc:
        print(f"[P6] 证据域混合，拒绝生成：{exc}", file=sys.stderr)
        return EXIT_CODES["policy_fail"]
    except publish.PublishError as exc:
        print(f"[P6] 发布失败：{exc}", file=sys.stderr)
        return EXIT_CODES["policy_fail"]
    print(
        # ``--json`` 供验收驱动**单行**消费（缩进会破坏逐行解析）；人读走默认摘要
        json.dumps(payload, ensure_ascii=False)
        if args.json
        else f"verdict={payload['verdict']} report_id={payload.get('report_id')}"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
