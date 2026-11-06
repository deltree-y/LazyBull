# -*- coding: utf-8 -*-
"""P5b ``calibration_only`` sealed bundle 校验（设计 §5.6，按**上游原生协议**读取）。

硬口径（与 2026-11-06 复审 R2-20261106-01 对齐）：
- 三族一律用**各族原生** ``load_artifact(..., sealed=True)`` 读口与原生 ``schema`` 常量，
  不自造字段、不改写 P5b 冻结字节；
- 链形状按原生语义解析：``hash_chain`` 的键是**语义名**（``manifest_sha256`` /
  ``calibrator_sha256`` / ``thresholds_sha256`` / ``<文件名>_sha256``），不是文件名白名单；
- 双向绑定来自原生：``report.identity.verdict_sha256`` 绑判定表字节，
  ``verdict.hash_chain.*`` 绑 manifest/临界值/holdout/power 字节；任一**部分改写**即断链；
- ``manifest.files`` 登记的上游引用（``calibrator`` / ``parent_manifest`` / ``parent_thresholds``）
  逐一核对：文件在 bundle 内则比对字节 SHA，否则登记为外部引用（``external_refs``）并计数；
- ``evidence`` 只从 manifest 的 ``evidence`` 块取（原生只有 manifest 有该块），
  ``evidence_class=calibration_only`` 且 ``not_forward_evidence=true``；报告/判定表若带
  ``evidence_class``/``not_forward_evidence`` 也须一致。

**诚实边界**：``record_sha256`` 是**无密钥自封存**。拥有 bundle 完整写权的对手可以改写全部
文件并重算所有回指，从而得到自洽结果；本模块检出的是**部分改写**（改值不改回指、改回指不改表、
断链、幽灵引用、逃逸、schema 不符）。真实性由上游产物根与证据类隔离承担，不得据本模块通过
宣称“该 calibration bundle 未被任何人重写过”。

**测试隔离**：pytest 的正向依据必须是**上游原生管线产出的合成 bundle**（``tests/p5b_t4_env``、
``tests/p5b_t5_env``、``tests/p5b_t6_env``），自造 JSON 不得作为正向依据；真实 campaign
（seq10/7/21）仍由 T5 的 ``check_real_golden`` 验收。本模块**不硬编码**任何真实 campaign 名
或 ``target_pp`` 数值。
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.lazybull.v2.hosts.testbed.referee import artifacts as t4_artifacts
from src.lazybull.v2.hosts.testbed.referee import manifest as t4_manifest
from src.lazybull.v2.hosts.testbed.referee import report as t4_report
from src.lazybull.v2.hosts.testbed.referee.drawdown import artifacts as t5_artifacts
from src.lazybull.v2.hosts.testbed.referee.drawdown import manifest as t5_manifest
from src.lazybull.v2.hosts.testbed.referee.drawdown import report as t5_report
from src.lazybull.v2.hosts.testbed.rehearsal import artifacts as t6_artifacts
from src.lazybull.v2.hosts.testbed.rehearsal import manifest as t6_manifest
from src.lazybull.v2.hosts.testbed.rehearsal import report as t6_report
from src.lazybull.v2.hosts.testbed.rehearsal import verdict as t6_verdict
from src.lazybull.v2.hosts.testbed.storage import TestbedError
from src.lazybull.v2.reporting import status

__all__ = [
    "BUNDLE_FAMILIES",
    "CALIBRATION_EVIDENCE_CLASS",
    "verify_p5b_bundle",
]

#: 证据类固定值（P5b 产物一律校准证据）
CALIBRATION_EVIDENCE_CLASS = "calibration_only"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _spec(
    *,
    manifest: str,
    report: str,
    verdict: str,
    aux: Mapping[str, str],
    loader: Any,
    manifest_schema: str,
    report_schema: str | None,
    verdict_schema: str | None,
    manifest_has_files: bool,
    required_chain: Mapping[str, Sequence[str]],
    linked: Sequence[str] = (),
) -> dict[str, Any]:
    """族描述（文件名、schema、必需链键全部来自上游模块，不手抄）。"""
    return {
        "manifest": manifest,
        "report": report,
        "verdict": verdict,
        "aux": dict(aux),
        "loader": loader,
        "schemas": {
            manifest: manifest_schema,
            **({report: report_schema} if report_schema else {}),
            **({verdict: verdict_schema} if verdict_schema else {}),
        },
        "manifest_has_files": bool(manifest_has_files),
        "required_chain": {str(k): tuple(v) for k, v in required_chain.items()},
        "linked": tuple(str(item) for item in linked),
    }


#: T4/T5 原生 ``hash_chain`` 的五个语义键（``referee/chain`` 与 ``drawdown/chain`` 同构）
_REFEREE_CHAIN_KEYS: tuple[str, ...] = (
    "manifest_sha256",
    "calibrator_sha256",
    "thresholds_sha256",
    "holdout_sha256",
    "power_sha256",
)


#: 三族 bundle 结构（文件名取自上游 ``ARTIFACT_*`` 常量，schema 取自上游常量）
BUNDLE_FAMILIES: dict[str, dict[str, Any]] = {
    "t4_seq10": _spec(
        manifest=t4_artifacts.ARTIFACT_MANIFEST,
        report=t4_artifacts.ARTIFACT_REPORT,
        verdict=t4_artifacts.ARTIFACT_VERDICT,
        aux={
            "thresholds": t4_artifacts.ARTIFACT_THRESHOLDS,
            "holdout": t4_artifacts.ARTIFACT_HOLDOUT,
            "power": t4_artifacts.ARTIFACT_POWER,
        },
        loader=t4_artifacts.load_artifact,
        manifest_schema=t4_manifest.MANIFEST_SCHEMA,
        report_schema=getattr(t4_report, "REPORT_SCHEMA", None),
        verdict_schema=getattr(t4_report, "VERDICT_SCHEMA", None),
        manifest_has_files=True,
        required_chain={
            t4_artifacts.ARTIFACT_REPORT: _REFEREE_CHAIN_KEYS,
            t4_artifacts.ARTIFACT_VERDICT: _REFEREE_CHAIN_KEYS,
        },
    ),
    "t5_seq7": _spec(
        manifest=t5_artifacts.ARTIFACT_MANIFEST,
        report=t5_artifacts.ARTIFACT_REPORT,
        verdict=t5_artifacts.ARTIFACT_VERDICT,
        aux={
            "thresholds": t5_artifacts.ARTIFACT_THRESHOLDS,
            "holdout": t5_artifacts.ARTIFACT_HOLDOUT,
            "power": t5_artifacts.ARTIFACT_POWER,
        },
        loader=t5_artifacts.load_artifact,
        manifest_schema=t5_manifest.MANIFEST_SCHEMA,
        report_schema=getattr(t5_report, "REPORT_SCHEMA", None),
        verdict_schema=getattr(t5_report, "VERDICT_SCHEMA", None),
        manifest_has_files=True,
        required_chain={
            t5_artifacts.ARTIFACT_REPORT: _REFEREE_CHAIN_KEYS,
            t5_artifacts.ARTIFACT_VERDICT: _REFEREE_CHAIN_KEYS,
        },
    ),
    "t6_seq21": _spec(
        manifest=t6_artifacts.ARTIFACT_MANIFEST,
        report=t6_artifacts.ARTIFACT_REPORT,
        verdict=t6_artifacts.ARTIFACT_VERDICT,
        aux={},
        loader=t6_artifacts.load_artifact,
        manifest_schema=t6_manifest.MANIFEST_SCHEMA,
        report_schema=getattr(t6_report, "REPORT_SCHEMA", None),
        verdict_schema=getattr(t6_verdict, "VERDICT_SCHEMA", None),
        manifest_has_files=False,
        required_chain={
            t6_artifacts.ARTIFACT_VERDICT: (
                "manifest_sha256",
                "calibrator_sha256",
                f"{t6_artifacts.ARTIFACT_REPORT}_sha256",
                f"{t6_artifacts.ARTIFACT_REPORT_MD}_sha256",
            )
        },
        linked=(t6_artifacts.ARTIFACT_REPORT_MD,),
    ),
}


def _sha256_file(path: Path) -> str:
    """文件字节 SHA-256（与 ``record_sha256`` 口径分离，禁止互换）。"""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(document: Mapping[str, Any], dotted: str) -> Any:
    """按点分路径取值；缺失返回 ``_MISSING``。"""
    current: Any = document
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


_MISSING: Any = object()


def _digest_problems(
    name: str,
    document: Mapping[str, Any],
    shas: Mapping[str, str],
    required: Sequence[str],
) -> list[str]:
    """核对 ``hash_chain``（原生语义键）：**必需键齐全 + 每个键都可解析且一致**。

    R2-20261106-01（复审）：只遍历“已提供的键”会让空链与幽灵键通过；因此
    ① 空链/缺必需键即失败；② 任何键若无载体（找不到对应字节来源）即失败。
    """
    chain = document.get("hash_chain")
    if not isinstance(chain, Mapping) or not chain:
        return [f"{name}: 缺 hash_chain 或链为空（原生协议必需，禁止空链放行）"]
    problems: list[str] = []
    missing = [key for key in required if key not in chain]
    if missing:
        problems.append(f"{name}: hash_chain 缺必需键 {missing}")
    for key, declared in sorted(chain.items()):
        if not isinstance(declared, str) or not _SHA256_PATTERN.match(declared):
            problems.append(f"{name}: hash_chain[{key}] 不是 64 位 SHA-256")
            continue
        actual = shas.get(str(key))
        if actual is None:
            problems.append(
                f"{name}: hash_chain[{key}] 在 bundle 内无载体（无对应字节来源，拒绝放行）"
            )
            continue
        if actual != declared:
            problems.append(
                f"{name}: hash_chain[{key}]={declared[:12]} 与实际 {actual[:12]} 不一致"
            )
    return problems


def _evidence_problems(name: str, document: Mapping[str, Any]) -> list[str]:
    """``evidence_class`` / ``not_forward_evidence`` 一致性（原生字段名）。"""
    problems: list[str] = []
    block = document.get("evidence")
    if isinstance(block, Mapping):
        if block.get("evidence_class") != CALIBRATION_EVIDENCE_CLASS:
            problems.append(
                f"{name}: evidence.evidence_class={block.get('evidence_class')!r} "
                f"不是 {CALIBRATION_EVIDENCE_CLASS}"
            )
        if block.get("not_forward_evidence") is not True:
            problems.append(f"{name}: evidence.not_forward_evidence 必须为 true")
    declared_class = document.get("evidence_class")
    if "evidence_class" in document and declared_class != CALIBRATION_EVIDENCE_CLASS:
        problems.append(
            f"{name}: evidence_class={declared_class!r} 不是 {CALIBRATION_EVIDENCE_CLASS}"
        )
    if "not_forward_evidence" in document and document.get("not_forward_evidence") is not True:
        problems.append(f"{name}: not_forward_evidence 必须为 true")
    return problems


def _resolve_shas(
    spec: Mapping[str, Any], directory: Path, manifest: Mapping[str, Any] | None
) -> dict[str, str]:
    """构建“语义键 → 实际字节 SHA”映射（仅登记**实际存在**的文件，不抛异常）。

    ``calibrator_sha256`` 没有独立文件，其载体是 ``manifest.files.calibrator.sha256``
    登记值——该键因此也进入映射，供 ``hash_chain`` 交叉核对。
    """
    shas: dict[str, str] = {}
    manifest_path = directory / str(spec["manifest"])
    if manifest_path.is_file():
        shas["manifest_sha256"] = _sha256_file(manifest_path)
    for key, artifact in dict(spec["aux"]).items():
        path = directory / str(artifact)
        if path.is_file():
            shas[f"{key}_sha256"] = _sha256_file(path)
    for path in sorted(directory.iterdir()):
        if path.is_file() and not path.is_symlink():
            shas[f"{path.name}_sha256"] = _sha256_file(path)
    recorded_calibrator = _lookup(manifest or {}, "files.calibrator.sha256")
    if isinstance(recorded_calibrator, str) and _SHA256_PATTERN.match(recorded_calibrator):
        shas["calibrator_sha256"] = recorded_calibrator
    return shas


def _files_problems(
    name: str, manifest: Mapping[str, Any], directory: Path, *, required: bool
) -> tuple[list[str], list[dict[str, Any]]]:
    """``manifest.files`` 上游引用核对（bundle 内比对字节，bundle 外登记外部引用）。"""
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return ([f"{name}: 缺 files 表（原生协议必需）"] if required else []), []
    problems: list[str] = []
    external: list[dict[str, Any]] = []
    for key, payload in sorted(files.items()):
        if not isinstance(payload, Mapping):
            problems.append(f"{name}: files[{key}] 不是对象")
            continue
        declared = payload.get("sha256")
        relative = payload.get("path")
        if not isinstance(declared, str) or not _SHA256_PATTERN.match(declared):
            problems.append(f"{name}: files[{key}].sha256 不是 64 位 SHA-256")
            continue
        candidate = directory / Path(str(relative or "")).name
        if isinstance(relative, str) and candidate.is_file():
            actual = _sha256_file(candidate)
            if actual != declared:
                problems.append(f"{name}: files[{key}] 字节 {actual[:12]} ≠ 登记 {declared[:12]}")
        else:
            external.append({"key": str(key), "path": relative, "sha256": declared})
    return problems, external


def _aux_binding_problems(
    spec: Mapping[str, Any], documents: Mapping[str, Mapping[str, Any]], manifest_sha: str
) -> list[str]:
    """辅助产物自带 ``manifest_sha256`` 的回指核对（原生 T4/T5 有该字段）。"""
    problems: list[str] = []
    for artifact in dict(spec["aux"]).values():
        document = documents.get(str(artifact))
        if document is None:
            continue
        declared = document.get("manifest_sha256")
        if not isinstance(declared, str) or not _SHA256_PATTERN.match(declared):
            continue
        if declared != manifest_sha:
            problems.append(
                f"{artifact}: manifest_sha256={declared[:12]} "
                f"与 manifest 字节 {manifest_sha[:12]} 不一致"
            )
    return problems


def _identity_problems(
    spec: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    shas: Mapping[str, str],
) -> list[str]:
    """报告/判定表身份回指核对（``identity.manifest_sha256`` / ``identity.verdict_sha256``）。"""
    manifest_sha = shas.get("manifest_sha256", "")
    problems: list[str] = []
    report = documents.get(str(spec["report"]))
    verdict = documents.get(str(spec["verdict"]))
    if report is not None:
        declared = _lookup(report, "identity.manifest_sha256")
        if declared is _MISSING:
            problems.append(f"{spec['report']}: 缺 identity.manifest_sha256")
        elif str(declared) != manifest_sha:
            problems.append(
                f"{spec['report']}: identity.manifest_sha256={str(declared)[:12]} "
                f"≠ manifest 字节 {manifest_sha[:12]}"
            )
        bound = _lookup(report, "identity.verdict_sha256")
        if bound is not _MISSING and verdict is not None:
            actual = shas.get(f"{spec['verdict']}_sha256", "")
            if str(bound) != actual:
                problems.append(
                    f"{spec['report']}: identity.verdict_sha256={str(bound)[:12]} "
                    f"≠ 判定表字节 {actual[:12]}"
                )
    if verdict is not None:
        declared = _lookup(verdict, "hash_chain.manifest_sha256")
        if declared is _MISSING:
            problems.append(f"{spec['verdict']}: 缺 hash_chain.manifest_sha256")
        elif str(declared) != manifest_sha:
            problems.append(
                f"{spec['verdict']}: hash_chain.manifest_sha256={str(declared)[:12]} "
                f"≠ manifest 字节 {manifest_sha[:12]}"
            )
    return problems


def _load_documents(
    spec: Mapping[str, Any], directory: Path
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    """按各族原生读口加载 manifest/report/verdict/aux（封存 + schema 一并校验）。"""
    documents: dict[str, Mapping[str, Any]] = {}
    problems: list[str] = []
    names = [str(spec["manifest"]), str(spec["report"]), str(spec["verdict"])]
    names.extend(str(item) for item in dict(spec["aux"]).values())
    for linked in spec.get("linked") or ():
        # 被链引用的非 JSON 产物（如 report.md）只做存在性检查
        if not (directory / str(linked)).is_file():
            problems.append(f"{linked}: 被 hash_chain 引用的文件缺失")
    for name in names:
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: 文件缺失")
            continue
        try:
            documents[name] = spec["loader"](
                path, what=f"P5b 产物 {name}", sealed=True, schema=dict(spec["schemas"]).get(name)
            )
        except (TestbedError, OSError, json.JSONDecodeError) as exc:
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
    return documents, problems


def verify_p5b_bundle(
    family: str, bundle_dir: Path | str, *, scope_base: Path | str
) -> dict[str, Any]:
    """校验单族 sealed bundle（按原生协议）；任一失败即 ``invalid``（fail-closed）。"""
    spec = BUNDLE_FAMILIES.get(str(family))
    if spec is None:
        raise ValueError(f"未知 P5b bundle 族：{family!r}（合法：{sorted(BUNDLE_FAMILIES)}）")
    directory = Path(bundle_dir)
    if not directory.is_dir():
        return status.make_status(
            "not_available", reason_code="no_source", detail=f"bundle 目录不存在：{directory}"
        )
    if not _trusted_dir(directory, Path(scope_base)):
        return status.make_status(
            "invalid", reason_code="bundle_path_escape", detail=f"bundle 目录不受信：{directory}"
        )
    documents, problems = _load_documents(spec, directory)
    shas = _resolve_shas(spec, directory, documents.get(str(spec["manifest"])))
    external: list[dict[str, Any]] = []
    manifest = documents.get(str(spec["manifest"]))
    if manifest is not None:
        problems.extend(_evidence_problems(str(spec["manifest"]), manifest))
        file_problems, external = _files_problems(
            str(spec["manifest"]), manifest, directory, required=bool(spec["manifest_has_files"])
        )
        problems.extend(file_problems)
    required_chain = dict(spec["required_chain"])
    for name, document in documents.items():
        if name != str(spec["manifest"]):
            problems.extend(_evidence_problems(name, document))
        if name in required_chain:
            problems.extend(
                _digest_problems(name, document, shas, required_chain[name])
            )
    problems.extend(_identity_problems(spec, documents, shas))
    problems.extend(_aux_binding_problems(spec, documents, shas.get("manifest_sha256", "")))
    return _bundle_result(
        family=str(family),
        directory=directory,
        shas=shas,
        documents=documents,
        external=external,
        problems=problems,
        spec=spec,
    )


def _campaign_id(manifest: Mapping[str, Any] | None, directory: Path) -> tuple[str | None, str]:
    """campaign id 与来源（``campaign.campaign_id`` → ``campaign_id`` → 目录名）。"""
    if manifest is not None:
        nested = _lookup(manifest, "campaign.campaign_id")
        if nested is not _MISSING:
            return str(nested), "manifest.campaign.campaign_id"
        flat = manifest.get("campaign_id")
        if flat is not None:
            return str(flat), "manifest.campaign_id"
    return directory.name, "bundle_dir_name"


def _bundle_result(
    *,
    family: str,
    directory: Path,
    shas: Mapping[str, str],
    documents: Mapping[str, Mapping[str, Any]],
    external: Sequence[Mapping[str, Any]],
    problems: Sequence[str],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """装配校验结果（campaign 身份 = 目录名 + manifest 字节哈希）。"""
    manifest = documents.get(str(spec["manifest"]))
    campaign_id, campaign_source = _campaign_id(manifest, directory)
    manifest_sha = shas.get("manifest_sha256", "")
    details = {
        "family": family,
        "campaign_identity": (
            f"{directory.name}@{manifest_sha[:12]}" if manifest_sha else directory.name
        ),
        "campaign_id": campaign_id,
        "campaign_id_source": campaign_source,
        "run_id": None if manifest is None else manifest.get("run_id"),
        "files": [{"name": name, "source_sha256": sha} for name, sha in sorted(shas.items())],
        "external_refs": [dict(item) for item in external],
        "protocol": "native_p5b_artifacts/v1",
        "failures": list(problems),
    }
    if problems:
        return status.make_status(
            "invalid",
            reason_code=_problem_reason(problems),
            detail="; ".join(list(problems)[:5]),
            extra=details,
        )
    return status.make_status(
        "available",
        detail=(
            f"{family} 原生 sealed bundle 校验通过（{len(documents)} 个文档、"
            f"{len(external)} 项外部引用）"
        ),
        verification_level="authority_verified",
        extra=details,
    )


def _problem_reason(problems: Sequence[str]) -> str:
    """按失败内容归类原因码（封存 → 证据类 → 链 → 逃逸 → 字段）。"""
    text = " | ".join(problems)
    if "封存哈希" in text or "record_sha256" in text:
        return "bundle_seal_mismatch"
    if "evidence" in text or "not_forward_evidence" in text:
        return "bundle_evidence_class"
    if "hash_chain" in text or "manifest_sha256" in text or "verdict_sha256" in text:
        return "bundle_chain_broken"
    if "逃逸" in text or "不受信" in text:
        return "bundle_path_escape"
    return "bundle_field_missing"


def _trusted_dir(target: Path, base: Path) -> bool:
    """bundle 目录是否受信（非重解析点、非符号链接、位于 scope 内）。"""
    from src.lazybull.v2.reporting.snapshot import is_reparse_point

    if target.is_symlink() or is_reparse_point(target):
        return False
    try:
        target.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return False
    return True
