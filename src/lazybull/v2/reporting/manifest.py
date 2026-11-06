# -*- coding: utf-8 -*-
"""canonical projection、manifest 自封存与源/projection 复算（设计 §8.1 / §8.2 / §8.3）。

硬口径：
- ``record_sha256`` 对**移除自身后**的全字段 canonical JSON 计算，**不与文件 SHA 混称**；
- 链序固定：原输入 SHA → ``input_manifest_sha256`` → projection SHA → manifest outputs →
  manifest 自 seal；
- ``verify`` **先验 schema + seal**，篡改 ``degradations``/字段路径同样失败；
- ``generated_at`` 是显式输入（复算读回原值），新墙钟即新 run；
- canonical JSON 复用 ``store_identity.canonical_json_sha256``（sort_keys + compact + UTF-8）。
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from src.lazybull.v2.reporting import contracts, status
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "MANIFEST_SCHEMA",
    "build_inputs",
    "build_manifest",
    "build_report_id",
    "input_manifest_sha256",
    "projection_sha256",
    "recompute_chain",
    "seal_manifest",
    "verify_manifest",
]

#: manifest schema（版本不符即 ``manifest_schema_unsupported``，不做隐式兼容）
MANIFEST_SCHEMA = "p6_report_manifest/v1"
_GENERATED_AT_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")
#: 64 位小写十六进制（文件/内容哈希的唯一形态）
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SEAL_FIELD = "record_sha256"


def build_inputs(receipts: Sequence[Any]) -> list[dict[str, Any]]:
    """输入身份清单（``root_id:relative`` + SHA + size，按路径稳定排序）。

    Raises:
        ValueError: 同一路径登记两次（会静默改变 ``input_manifest_sha256``）或 SHA 形态非法。
    """
    rows = [
        receipt.as_dict() if hasattr(receipt, "as_dict") else dict(receipt)
        for receipt in receipts
    ]
    ordered = sorted(rows, key=lambda row: str(row["path"]))
    seen: set[str] = set()
    for row in ordered:
        path = str(row["path"])
        if path in seen:
            raise ValueError(f"manifest 输入路径重复：{path}")
        seen.add(path)
        if not _SHA256_PATTERN.match(str(row.get("sha256"))):
            raise ValueError(f"manifest 输入 {path} 的 sha256 不是 64 位小写十六进制")
    return ordered


def input_manifest_sha256(inputs: Sequence[Mapping[str, Any]]) -> str:
    """输入清单 canonical 哈希（链序第二环）。"""
    return canonical_json_sha256([dict(row) for row in inputs])


def projection_sha256(projection: Mapping[str, Any]) -> str:
    """projection canonical 哈希（链序第三环）。"""
    return canonical_json_sha256(dict(projection))


def build_report_id(
    *, report_kind: str, generated_at: str, input_sha256: str
) -> str:
    """``p6_<kind>_<UTC YYYYMMDDTHHMMSSZ>_<input_manifest_sha256 前 12 位>``。"""
    if report_kind not in contracts.REPORT_KINDS:
        raise ValueError(f"未知 report_kind：{report_kind!r}（合法 {contracts.REPORT_KINDS}）")
    if not _GENERATED_AT_PATTERN.match(str(generated_at)):
        raise ValueError(f"generated_at 必须是 UTC YYYYMMDDTHHMMSSZ：{generated_at!r}")
    if not _SHA256_PATTERN.match(str(input_sha256)):
        raise ValueError(f"input_manifest_sha256 必须是 64 位小写十六进制：{input_sha256!r}")
    report_id = f"p6_{report_kind}_{generated_at}_{str(input_sha256)[:12]}"
    if not re.match(contracts.REPORT_ID_PATTERN, report_id):
        raise ValueError(f"report_id 形态不符冻结 pattern：{report_id!r}")
    return report_id


def _canonical_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """去自身封存字段后的 canonical 载荷。"""
    return {key: value for key, value in manifest.items() if key != _SEAL_FIELD}


def seal_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """给 manifest 加自封存 ``record_sha256``（移除自身后全字段 canonical 哈希）。"""
    sealed = dict(manifest)
    sealed.pop(_SEAL_FIELD, None)
    sealed[_SEAL_FIELD] = canonical_json_sha256(_canonical_payload(sealed))
    return sealed


def build_manifest(
    *,
    report_kind: str,
    evidence_class: str,
    input_evidence_classes: Sequence[str],
    generated_at: str,
    data_window: Mapping[str, Any],
    source_watermarks: Mapping[str, Any],
    root_map: Mapping[str, str],
    inputs: Sequence[Mapping[str, Any]],
    projection: Mapping[str, Any],
    generator: Mapping[str, Any],
    outputs: Mapping[str, str],
    csv_schemas: Mapping[str, Any],
    degradations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """装配并自封存 report manifest（必需字段集合见 ``contracts``）。"""
    if evidence_class not in contracts.EVIDENCE_CLASSES:
        raise ValueError(f"未知 evidence_class：{evidence_class!r}")
    unknown = [item for item in input_evidence_classes if item not in contracts.EVIDENCE_CLASSES]
    if unknown:
        raise ValueError(f"未声明的输入证据类：{unknown}")
    input_rows = build_inputs(inputs)
    input_sha = input_manifest_sha256(input_rows)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "report_id": build_report_id(
            report_kind=report_kind, generated_at=generated_at, input_sha256=input_sha
        ),
        "report_kind": report_kind,
        "evidence_class": evidence_class,
        "input_evidence_classes": sorted({str(item) for item in input_evidence_classes}),
        "generated_at": str(generated_at),
        "data_window": dict(data_window),
        "source_watermarks": dict(source_watermarks),
        "root_map": {str(key): str(value) for key, value in sorted(root_map.items())},
        "inputs": input_rows,
        "projection_sha256": projection_sha256(projection),
        "input_manifest_sha256": input_sha,
        "generator": dict(generator),
        "outputs": {str(key): str(value) for key, value in sorted(outputs.items())},
        "csv_schemas": dict(csv_schemas),
        "degradations": [dict(row) for row in degradations],
    }
    return seal_manifest(manifest)


def verify_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """只读校验：**先** schema + seal，再必需字段与 report_id 形态。"""
    if not isinstance(manifest, Mapping):
        return status.make_status(
            "invalid", reason_code="manifest_schema_unsupported", detail="manifest 顶层不是对象"
        )
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        return status.make_status(
            "invalid",
            reason_code="manifest_schema_unsupported",
            detail=f"schema_version={manifest.get('schema_version')!r} 不受支持",
        )
    recorded = manifest.get(_SEAL_FIELD)
    if not recorded:
        return status.make_status(
            "invalid", reason_code="manifest_seal_mismatch", detail="缺 record_sha256（禁止使用未封存记录）"
        )
    actual = canonical_json_sha256(_canonical_payload(manifest))
    if str(recorded) != actual:
        return status.make_status(
            "invalid",
            reason_code="manifest_seal_mismatch",
            detail=f"封存哈希不一致：记录 {recorded} 实际 {actual}",
        )
    missing = [name for name in contracts.MANIFEST_REQUIRED_FIELDS if name not in manifest]
    if missing:
        return status.make_status(
            "invalid",
            reason_code="missing_precondition",
            detail=f"manifest 缺必需字段 {missing}",
        )
    if not re.match(contracts.REPORT_ID_PATTERN, str(manifest.get("report_id"))):
        return status.make_status(
            "invalid",
            reason_code="missing_precondition",
            detail=f"report_id 形态不符冻结 pattern：{manifest.get('report_id')!r}",
        )
    shape_problems = _shape_problems(manifest)
    if shape_problems:
        return status.make_status(
            "invalid",
            reason_code="missing_precondition",
            detail="; ".join(shape_problems[:5]),
        )
    return status.make_status(
        "available", detail="schema + seal + 必需字段 + 哈希形态通过（先验 seal）"
    )


def _shape_problems(manifest: Mapping[str, Any]) -> list[str]:
    """输入/输出哈希与路径形态校验（非法即不可信，不做隐式修正）。"""
    problems: list[str] = []
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        return ["inputs 不是数组"]
    seen: set[str] = set()
    for index, row in enumerate(inputs):
        if not isinstance(row, Mapping):
            problems.append(f"inputs[{index}] 不是对象")
            continue
        path = str(row.get("path"))
        if not path or ":" not in path:
            problems.append(f"inputs[{index}].path 形态非法：{path!r}")
        if path in seen:
            problems.append(f"inputs[{index}].path 重复：{path}")
        seen.add(path)
        if not _SHA256_PATTERN.match(str(row.get("sha256"))):
            problems.append(f"inputs[{index}].sha256 不是 64 位小写十六进制")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return problems + ["outputs 不是对象"]
    for name, digest in outputs.items():
        if not _SHA256_PATTERN.match(str(digest)):
            problems.append(f"outputs[{name}] 不是 64 位小写十六进制")
    return problems


def recompute_chain(
    *,
    manifest: Mapping[str, Any],
    projection: Mapping[str, Any],
    outputs: Mapping[str, str],
    inputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """复算链核验：输入 → projection → outputs 与 manifest 登记**逐环双向**比对。

    ``inputs``/``projection`` 必填（缺环不得以“部分一致”冒充全链一致）；
    ``outputs`` 与 manifest 登记的键集合必须相等（多写/少写都 fail）。
    """
    row = verify_manifest(manifest)
    if not status.is_usable(row):
        return row
    checks: list[dict[str, Any]] = []

    def _check(link: str, expected: Any, actual: Any) -> None:
        checks.append(
            {"link": link, "expected": expected, "actual": actual, "ok": expected == actual}
        )

    _check(
        "input_manifest_sha256",
        manifest.get("input_manifest_sha256"),
        input_manifest_sha256(build_inputs(inputs)),
    )
    _check(
        "projection_sha256",
        manifest.get("projection_sha256"),
        projection_sha256(projection),
    )
    declared_outputs = manifest.get("outputs") or {}
    _check(
        "outputs.keys",
        sorted(str(name) for name in declared_outputs),
        sorted(str(name) for name in outputs),
    )
    for name in sorted(declared_outputs):
        _check(f"outputs.{name}", declared_outputs.get(name), outputs.get(name))
    broken = [item for item in checks if not item["ok"]]
    details = {"checks": checks, "ok": not broken}
    if broken:
        reason = (
            "input_drift"
            if any(item["link"].startswith("input_") for item in broken)
            else "projection_sha_mismatch"
            if any(item["link"].startswith("projection") for item in broken)
            else "output_sha_mismatch"
        )
        return status.make_status(
            "invalid",
            reason_code=reason,
            detail=f"复算链断裂：{[item['link'] for item in broken]}",
            extra=details,
        )
    return status.make_status(
        "available", detail=f"{len(checks)} 环复算一致", extra=details
    )
