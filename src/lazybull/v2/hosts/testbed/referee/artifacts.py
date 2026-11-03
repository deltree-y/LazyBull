# -*- coding: utf-8 -*-
"""P5b T4 校准产物落盘（只落 campaign ``reports/``；逐字节确定 + 独占写）。

确定性口径
==========
- 产物内容**不含墙钟**（墙钟单独落 ``machine_time.json``，不参与复现比对）；
- 序列化固定 ``ensure_ascii=False, indent=2, sort_keys=True``，行尾 ``\\n``；
- 数值一律转为原生 Python 类型（禁 NaN/Inf）——否则封存哈希与复现比对失效；
- 写盘一律**独占创建**（``x`` 模式）：重复运行必须换新 run 身份目录，绝不覆盖既有证据。
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
    "ARTIFACT_HOLDOUT",
    "ARTIFACT_MACHINE_TIME",
    "ARTIFACT_MANIFEST",
    "ARTIFACT_POWER",
    "ARTIFACT_REPORT",
    "ARTIFACT_REPORT_MD",
    "ARTIFACT_THRESHOLDS",
    "ARTIFACT_VERDICT",
    "ARTIFACT_NAMES",
    "ArtifactError",
    "artifact_path",
    "json_text",
    "load_artifact",
    "reports_dir",
    "sha256_file",
    "to_jsonable",
    "write_text_once",
    "write_json_once",
]

#: 冻结 manifest（campaign 报告副本；仓库冻结件同内容）
ARTIFACT_MANIFEST = "calibration_manifest.json"
#: 临界值产物（训练一次，之后只读）
ARTIFACT_THRESHOLDS = "calibration_thresholds.json"
#: holdout 逐单元验收
ARTIFACT_HOLDOUT = "income_holdout.json"
#: 功效与 MDE
ARTIFACT_POWER = "income_power.json"
#: 判定表（哈希链汇总 + 可读结论）
ARTIFACT_VERDICT = "referee_verdict.json"
#: §7.2 收益部分报告
ARTIFACT_REPORT = "income_referee_report.json"
#: 报告的可读版（markdown）
ARTIFACT_REPORT_MD = "income_referee_report.md"
#: 墙钟登记（**不参与**确定性比对）
ARTIFACT_MACHINE_TIME = "machine_time.json"
#: 产物名清单
ARTIFACT_NAMES = (
    ARTIFACT_MANIFEST,
    ARTIFACT_THRESHOLDS,
    ARTIFACT_HOLDOUT,
    ARTIFACT_POWER,
    ARTIFACT_VERDICT,
    ARTIFACT_REPORT,
    ARTIFACT_REPORT_MD,
    ARTIFACT_MACHINE_TIME,
)


class ArtifactError(TestbedError):
    """产物读写非法（已存在、封存不符、数值非有限）。"""

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
    """确定性 JSON 文本（后续加换行由写盘函数负责）。"""
    return json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)


def write_json_once(
    path: Path, payload: Mapping[str, Any], *, what: str, sealed: bool = True
) -> Dict[str, Any]:
    """独占写一个 JSON 产物（可选封存）；已存在即拒绝。"""
    body = seal_record(to_jsonable(payload)) if sealed else to_jsonable(payload)
    return write_text_once(path, json_text(body) + "\n", what=what)


def write_text_once(path: Path, text: str, *, what: str) -> Dict[str, Any]:
    """独占写文本产物；返回 ``{path, sha256}``。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise ArtifactError(f"{what} 已存在（禁止覆盖既有校准产物）: {target}") from exc
    return {"path": target.as_posix(), "sha256": sha256_file(target)}


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
