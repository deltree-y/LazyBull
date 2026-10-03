# -*- coding: utf-8 -*-
"""假设台账全量 schema 校验（F3 §8 技术债偿还：DataStore 落地后对全量条目跑 schema 校验回补）。

契约依据 `docs/contracts/hypothesis_ledger_schema.md`（F3）：
- §3.1 公共字段 / §3.2 prereg 必填 / §3.3 conclusion 必填；
- §8 + F3：占位 prereg（``retrospective_placeholder=true``）豁免
  ``arms / metrics / noise_band / stop_rule`` 空值；回填条目 ``alpha_budget`` 允许 null；
- F3 ②-07：结论类条目不带 ``vehicle / tags``（书实差异登记，分类字段不重复校验）；
- §7：结论类 ``evidence_refs`` 必须非空（每次状态转换必须有证据引用）。

**只登记不改写**（append-only）：本模块只读校验并产出报告，绝不回写台账文件。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["validate_ledger", "validate_entry"]

#: prereg payload 必填字段（schema §3.1 + §3.2）
PREREG_REQUIRED = (
    "title",
    "vehicle",
    "tags",
    "statement",
    "arms",
    "metrics",
    "decision_rule",
    "noise_band",
    "candidate_pool_denominator",
    "pool_description",
    "alpha_budget",
    "stop_rule",
    "reopen_condition",
)

#: 占位 prereg（retrospective_placeholder=true）允许空值的字段（F3 §8 豁免）
PLACEHOLDER_EMPTY_ALLOWED = ("arms", "metrics", "noise_band", "stop_rule")

#: conclusion payload 必填字段（schema §3.3 + F3 ②-07：结论类不带 vehicle/tags；title 沿 §3.1）
CONCLUSION_REQUIRED = (
    "title",
    "hypothesis_id",
    "verdict",
    "result_metrics",
    "evidence_refs",
    "supersedes",
    "notes",
)

VALID_KINDS = frozenset({"prereg", "conclusion"})
VALID_VERDICTS = frozenset({"pass", "fail", "terminated", "inconclusive", "not_adopted"})
VALID_VEHICLES = frozenset({"sleeve", "factor_family", "rule", "policy", "data_source"})
DATA_STATE_KEYS = ("raw_partitions", "features_partition_max", "config_digest", "code_digest")


def _is_empty(value: Any) -> bool:
    """空值口径：None / 空串 / 空列表 / 空字典。"""
    return value is None or value == "" or value == [] or value == {}


def _check_common(obj: dict[str, Any], reasons: list[str]) -> None:
    """① schema_version==1；② hypothesis_id 非空；③ kind 合法。"""
    if obj.get("schema_version") != 1:
        reasons.append(f"schema_version 应为 1，实得 {obj.get('schema_version')!r}")
    hypothesis_id = obj.get("hypothesis_id")
    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
        reasons.append(f"hypothesis_id 为空或类型非法: {hypothesis_id!r}")
    if obj.get("kind") not in VALID_KINDS:
        reasons.append(f"kind 非法: {obj.get('kind')!r}（合法值 {sorted(VALID_KINDS)}）")


def _check_prereg_field(key: str, value: Any, placeholder: bool, reasons: list[str]) -> None:
    """prereg 单字段口径：候选池分母 / α 预算 / 空值（占位豁免仅覆盖 F3 指定四字段）。"""
    if key == "candidate_pool_denominator":
        if not (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
            reasons.append(f"prereg candidate_pool_denominator 须为 ≥1 整数，实得 {value!r}")
    elif key == "alpha_budget":
        # §8：回填条目（占位 prereg）alpha_budget 记 null；正式 prereg 登记即锁定扣减，必须为数
        if value is None:
            if not placeholder:
                reasons.append("prereg alpha_budget 为 null（仅回填占位条目允许）")
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            reasons.append(f"prereg alpha_budget 类型非法: {value!r}")
    elif _is_empty(value) and not (placeholder and key in PLACEHOLDER_EMPTY_ALLOWED):
        reasons.append(f"prereg 字段为空: {key}（占位豁免={placeholder}）")


def _check_prereg_payload(payload: dict[str, Any], reasons: list[str]) -> None:
    """prereg 必填 + 占位豁免 + 字段级口径（vehicle 枚举 / tags 类型 / metrics 主判据）。"""
    missing = [k for k in PREREG_REQUIRED if k not in payload]
    if missing:
        reasons.append(f"prereg payload 缺字段: {missing}")
    placeholder = payload.get("retrospective_placeholder") is True
    for key in PREREG_REQUIRED:
        if key in payload:
            _check_prereg_field(key, payload[key], placeholder, reasons)
    vehicle = payload.get("vehicle")
    if vehicle is not None and vehicle not in VALID_VEHICLES:
        reasons.append(f"prereg vehicle 非法: {vehicle!r}（合法值 {sorted(VALID_VEHICLES)}）")
    if "tags" in payload and not isinstance(payload["tags"], list):
        reasons.append(f"prereg tags 须为 list，实得 {type(payload['tags']).__name__}")
    metrics = payload.get("metrics")
    if not placeholder and isinstance(metrics, dict) and metrics and "primary" not in metrics:
        reasons.append("prereg metrics 缺 primary 主判据键（§3.2）")


def _check_conclusion_payload(payload: dict[str, Any], reasons: list[str]) -> None:
    """conclusion 必填 + verdict 枚举 + 证据引用非空（§7）。"""
    missing = [k for k in CONCLUSION_REQUIRED if k not in payload]
    if missing:
        reasons.append(f"conclusion payload 缺字段: {missing}")
    verdict = payload.get("verdict")
    if "verdict" in payload and verdict not in VALID_VERDICTS:
        reasons.append(f"conclusion verdict 非法: {verdict!r}（合法值 {sorted(VALID_VERDICTS)}）")
    link = payload.get("hypothesis_id")
    if "hypothesis_id" in payload and (not isinstance(link, str) or not link.strip()):
        reasons.append(f"conclusion payload.hypothesis_id 为空（须挂接 prereg）: {link!r}")
    if "evidence_refs" in payload:
        refs = payload["evidence_refs"]
        if not isinstance(refs, list) or not refs:
            reasons.append("conclusion evidence_refs 须为非空 list（§7：状态转换必须有证据引用）")
    if "supersedes" in payload and not isinstance(payload["supersedes"], list):
        reasons.append("conclusion supersedes 须为 list（可空）")
    if "notes" in payload and _is_empty(payload["notes"]):
        reasons.append("conclusion notes 为空（须登记已知代价 / 解释风险）")
    if "result_metrics" in payload and not isinstance(payload["result_metrics"], dict):
        reasons.append("conclusion result_metrics 须为 object")
    if "title" in payload and _is_empty(payload["title"]):
        reasons.append("conclusion title 为空")


def _check_data_state(obj: dict[str, Any], reasons: list[str]) -> None:
    """data_state 四键齐全。"""
    data_state = obj.get("data_state")
    if not isinstance(data_state, dict):
        reasons.append(f"data_state 缺失或类型非法: {type(data_state).__name__}")
        return
    missing = [k for k in DATA_STATE_KEYS if k not in data_state]
    if missing:
        reasons.append(f"data_state 缺键: {missing}")


def _check_registered_at(obj: dict[str, Any], reasons: list[str]) -> None:
    """registered_at 可解析 ISO。"""
    value = obj.get("registered_at")
    if not isinstance(value, str) or not value:
        reasons.append(f"registered_at 缺失或类型非法: {value!r}")
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        reasons.append(f"registered_at 非合法 ISO 时间戳: {value!r}")


def validate_entry(obj: dict[str, Any]) -> list[str]:
    """校验单条台账条目，返回失败原因列表（空 = pass）。"""
    reasons: list[str] = []
    _check_common(obj, reasons)
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        reasons.append(f"payload 缺失或类型非法: {type(payload).__name__}")
    elif obj.get("kind") == "prereg":
        _check_prereg_payload(payload, reasons)
    elif obj.get("kind") == "conclusion":
        _check_conclusion_payload(payload, reasons)
    _check_data_state(obj, reasons)
    _check_registered_at(obj, reasons)
    return reasons


def validate_ledger(ledger_path: Path | str) -> dict[str, Any]:
    """全量校验台账 jsonl，返回报告 dict（逐条 pass/fail + 原因 + 汇总计数）。

    附台账级完整性检查：hypothesis_id 跨条目唯一（协议 §8 防冲突语义）。
    只读校验，不改写台账文件。
    """
    path = Path(ledger_path)
    entries: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            record = _validate_line(stripped, line_no, seen_ids)
            entries.append(record)
    failed = sum(1 for e in entries if e["status"] == "fail")
    return {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "ledger_path": str(path),
        "schema_ref": "docs/contracts/hypothesis_ledger_schema.md（F3，含 §8 占位豁免）",
        "total": len(entries),
        "passed": len(entries) - failed,
        "failed": failed,
        "entries": entries,
    }


def _validate_line(stripped: str, line_no: int, seen_ids: dict[str, int]) -> dict[str, Any]:
    """单行校验：JSON 可解析 ⇒ 条目级校验 ⇒ 重复 id 检查。"""
    record: dict[str, Any] = {
        "line": line_no,
        "hypothesis_id": None,
        "kind": None,
        "status": "pass",
        "reasons": [],
    }
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        record["status"] = "fail"
        record["reasons"] = [f"JSON 解析失败: {exc}"]
        return record
    if not isinstance(obj, dict):
        record["status"] = "fail"
        record["reasons"] = ["条目不是 JSON 对象"]
        return record
    record["hypothesis_id"] = obj.get("hypothesis_id")
    record["kind"] = obj.get("kind")
    reasons = validate_entry(obj)
    hypothesis_id = obj.get("hypothesis_id")
    if isinstance(hypothesis_id, str) and hypothesis_id.strip():
        if hypothesis_id in seen_ids:
            reasons.append(
                f"hypothesis_id 与第 {seen_ids[hypothesis_id]} 行重复（append-only 唯一性）"
            )
        else:
            seen_ids[hypothesis_id] = line_no
    record["reasons"] = reasons
    record["status"] = "fail" if reasons else "pass"
    return record
