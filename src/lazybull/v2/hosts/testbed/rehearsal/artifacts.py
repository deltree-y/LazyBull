# -*- coding: utf-8 -*-
"""P5b T6 演练产物落盘（独占创建、逐字节确定，不覆盖任何既有证据）。

与 T4/T5 的 ``artifacts`` 同口径但**自持实现**：产物字节由本模块决定，因此它必须在
演练器身份内（否则改字节不会被身份拦住）。共享层只用试车场的 ``storage``
（``seal_record`` / ``read_json`` / ``verify_seal``）。

- 产物内容**不含墙钟**（墙钟单独落 ``machine_time*.json``，不参与复现比对）；
- 序列化固定 ``ensure_ascii=False, indent=2, sort_keys=True``、行尾 ``\\n``；
- 数值一律转原生 Python 类型（禁 NaN/Inf）；写盘一律 ``x`` 模式。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.hosts.testbed.storage import read_json, seal_record, verify_seal

__all__ = [
    "ARTIFACT_MACHINE_TIME",
    "ARTIFACT_MACHINE_TIME_FREEZE",
    "ARTIFACT_MACHINE_TIME_RUN",
    "ARTIFACT_MANIFEST",
    "ARTIFACT_NAMES",
    "ARTIFACT_PRODUCTION_BEFORE",
    "ARTIFACT_REPORT",
    "ARTIFACT_REPORT_MD",
    "ARTIFACT_VERDICT",
    "ArtifactError",
    "artifact_path",
    "json_text",
    "load_artifact",
    "reports_dir",
    "sha256_file",
    "to_jsonable",
    "write_json_once",
    "write_text_once",
]

#: 演练 manifest（campaign 副本；仓库冻结件同内容）
ARTIFACT_MANIFEST = "rehearsal_manifest.json"
#: 判定表（H1/H6/H7 作者侧检查 + 三环哈希）
ARTIFACT_VERDICT = "rehearsal_verdict.json"
#: §7.2 演练报告
ARTIFACT_REPORT = "rehearsal_report.json"
#: 报告可读版
ARTIFACT_REPORT_MD = "rehearsal_report.md"
#: 冻结阶段墙钟
ARTIFACT_MACHINE_TIME_FREEZE = "machine_time_freeze.json"
#: 运行阶段墙钟（含单日与整批）
ARTIFACT_MACHINE_TIME_RUN = "machine_time_run.json"
#: 报告阶段墙钟
ARTIFACT_MACHINE_TIME = "machine_time.json"
#: register 阶段的生产根快照（生产零写入跨阶段比对的 before）
ARTIFACT_PRODUCTION_BEFORE = "production_before.json"
#: 允许的产物名（写盘白名单）
ARTIFACT_NAMES = (
    ARTIFACT_MANIFEST,
    ARTIFACT_VERDICT,
    ARTIFACT_REPORT,
    ARTIFACT_REPORT_MD,
    ARTIFACT_MACHINE_TIME,
    ARTIFACT_MACHINE_TIME_RUN,
    ARTIFACT_PRODUCTION_BEFORE,
)


class ArtifactError(TestbedError):
    """产物读写非法（已存在、封存不符、数值非有限、未登记产物名）。"""

    __test__ = False


def reports_dir(campaign_root: Path) -> Path:
    """campaign ``reports/`` 目录（§7.1 布局；不建目录，写时按需创建）。"""
    return Path(campaign_root) / "reports"


def artifact_path(campaign_root: Path, name: str) -> Path:
    """产物绝对路径（``name`` 必须在 :data:`ARTIFACT_NAMES` 内）。"""
    if name not in ARTIFACT_NAMES:
        raise ArtifactError(f"未登记产物名: {name!r}（允许 {list(ARTIFACT_NAMES)}）")
    return reports_dir(campaign_root) / name


def sha256_file(path: Path) -> str:
    """文件字节 sha256。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_jsonable(value: Any) -> Any:
    """转为严格 JSON 安全的原生类型（numpy → Python；非有限浮点即报错）。"""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError(f"产物含非有限浮点（禁止落盘）: {value!r}")
    return value


def json_text(payload: Mapping[str, Any]) -> str:
    """确定性 JSON 文本（换行由写盘函数补）。"""
    return json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)


def write_text_once(path: Path, text: str, *, what: str) -> Dict[str, Any]:
    """独占写文本产物；返回 ``{path, sha256}``。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise ArtifactError(f"{what} 已存在（禁止覆盖既有演练产物）: {target}") from exc
    return {"path": target.as_posix(), "sha256": sha256_file(target)}


def write_json_once(
    path: Path, payload: Mapping[str, Any], *, what: str, sealed: bool = True
) -> Dict[str, Any]:
    """独占写一个 JSON 产物（可选封存）；已存在即拒绝。"""
    body = seal_record(to_jsonable(payload)) if sealed else to_jsonable(payload)
    return write_text_once(path, json_text(body) + "\n", what=what)


def load_artifact(
    path: Path, *, what: str, sealed: bool = True, schema: Optional[str] = None
) -> Dict[str, Any]:
    """读取产物（缺省校验封存与 schema）。"""
    doc = read_json(Path(path), what=what)
    if sealed:
        verify_seal(doc, what=f"{what}（{path}）")
    if schema is not None and doc.get("schema") != schema:
        raise ArtifactError(f"{what} schema 非法: {doc.get('schema')!r}（期望 {schema}）")
    return doc
