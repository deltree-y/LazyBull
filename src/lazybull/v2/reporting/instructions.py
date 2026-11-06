# -*- coding: utf-8 -*-
"""未来指令、pending 与调仓状态投影（设计 §5.2 对应三行）。

硬口径：
- 未来指令只看 ``E > 当前权威日 D`` 且**所有行** ``source_date == D`` 的文件；历史
  ``E ≤ D`` 文件不参与未来推断；合法时恰一个 ``E``，且不存在其他 ``E' > D``；
- ``N=0`` 且无当前未来文件才 ``empty_valid``；正常 v2 有计数时要求行数 = N；
- **切换前记录形态**（同时缺 ``run_kind``/``engine``）固定
  ``not_available/run_record_pre_switch_shape``，不以当日已执行指令推断未来；
- **恢复代缺计数**不当 0 也不单独判坏：可由文件/权威核对则标
  ``recovered_generation_count_missing``，否则 ``not_available/recovery_counts_missing``；
- ``pending`` 非 list、``rebalance`` 非 dict 一律 ``invalid``；合法 ``[]`` 才是 ``empty_valid``；
- 远端只读来源不支持详细读取 → ``not_available/unsupported``。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from src.lazybull.v2.reporting import status

__all__ = [
    "INSTRUCTION_FILE_PATTERN",
    "project_instructions",
    "project_pending",
    "project_rebalance",
    "record_counts",
    "select_future_instruction_files",
]

#: 未来指令文件名形态（执行日 ``YYYYMMDD.parquet``）
INSTRUCTION_FILE_PATTERN = re.compile(r"^(\d{8})\.parquet$")


def select_future_instruction_files(
    names: Sequence[str], *, current_date: str
) -> tuple[list[str], list[str]]:
    """切分 ``instructions/`` 文件名：返回 ``(未来候选日期升序, 历史日期升序)``。"""
    future: list[str] = []
    history: list[str] = []
    for name in names:
        matched = INSTRUCTION_FILE_PATTERN.match(str(name))
        if not matched:
            continue
        day = matched.group(1)
        (future if day > current_date else history).append(day)
    return sorted(future), sorted(history)


def _rows_source_date_mismatch(
    rows: Sequence[Mapping[str, Any]], *, source_date: str
) -> str | None:
    """校验未来指令全部行 ``source_date == D``；返回首个不一致描述（无则 None）。"""
    for index, row in enumerate(rows):
        value = str(row.get("source_date") or "").strip()
        normalized = value[:10].replace("-", "") if len(value) >= 10 else value
        if normalized != source_date:
            return f"第 {index} 行 source_date={row.get('source_date')!r} ≠ 当前权威日 {source_date}"
    return None


def project_instructions(
    *,
    names: Sequence[str],
    current_date: str,
    record: Mapping[str, Any] | None,
    pre_switch_shape: bool,
    loader: Callable[[str], Any],
    generation_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """未来指令区投影（合法空/缺文件/歧义/跨代/恢复缺计数分别处理）。

    ``generation_gate``（R2-20261106-02 复审）：record 未通过完整代门禁时，
    指令区不得给出 ``authority_verified`` 的结论——``invalid`` 直接透传原因码，
    切换前形态仍走专用 ``not_available``。
    """
    future, history = select_future_instruction_files(names, current_date=current_date)
    counts = record_counts(record)
    details = {"future_files": future, "history_files": len(history)}
    gate_state = None if generation_gate is None else generation_gate.get("report_status")
    if gate_state == "invalid":
        reason = str(
            (generation_gate or {}).get("degradation_reason_code") or "authority_mismatch"
        )
        return status.make_status(
            "invalid",
            reason_code=reason,
            detail=(
                f"run record 未通过完整代门禁（{reason}）："
                f"未来指令文件 {future or '无'} 不得被认证为 authority_verified"
            ),
            extra={**details, "instruction_count": None, "gate_reason_code": reason},
        )
    if pre_switch_shape:
        return status.make_status(
            "not_available",
            reason_code="run_record_pre_switch_shape",
            detail="切换前记录形态：不以当日已执行指令推断未来（不伪造）",
            extra={**details, "instruction_count": None},
        )
    if not future:
        return _instructions_without_future(counts, details)
    if len(future) > 1:
        return status.make_status(
            "invalid",
            reason_code="instruction_future_ambiguous",
            detail=f"存在多个未来执行日指令文件：{future}",
            extra=details,
        )
    return _instructions_with_future(
        execution_date=future[0],
        counts=counts,
        details=details,
        loader=loader,
        current_date=current_date,
    )


def record_counts(record: Mapping[str, Any] | None) -> dict[str, int | None]:
    """从 run record 取计数字段；**缺键与 0 严格区分**（缺 = None）。"""
    if not isinstance(record, Mapping):
        return {"instructions_count": None, "fills_count": None,
                "pending_buys_count": None, "pending_sells_count": None}
    keys = ("instructions_count", "fills_count", "pending_buys_count", "pending_sells_count")
    return {key: (int(record[key]) if key in record and record[key] is not None else None)
            for key in keys}


def _instructions_without_future(
    counts: Mapping[str, int | None], details: Mapping[str, Any]
) -> dict[str, Any]:
    """无未来指令文件分支：按记录计数区分合法空/缺文件/恢复缺计数。"""
    declared = counts.get("instructions_count")
    if declared is None:
        return status.make_status(
            "not_available",
            reason_code="recovery_counts_missing",
            detail="记录缺 instructions_count 且无未来指令文件可核对",
            extra={**details, "instruction_count": None},
        )
    if declared == 0:
        return status.make_status(
            "empty_valid",
            detail="记录声明 0 条指令且无未来执行日文件",
            verification_level="authority_verified",
            extra={**details, "instruction_count": 0},
        )
    return status.make_status(
        "invalid",
        reason_code="instruction_missing_future",
        detail=f"记录声明 {declared} 条指令但缺 E>D 的指令文件",
        extra={**details, "instruction_count": declared},
    )


def _instructions_with_future(
    *,
    execution_date: str,
    counts: Mapping[str, int | None],
    details: Mapping[str, Any],
    loader: Callable[[str], Any],
    current_date: str,
) -> dict[str, Any]:
    """恰一个未来指令文件分支：校验跨代/行数/计数缺省。"""
    try:
        frame = loader(f"{execution_date}.parquet")
    except (OSError, ValueError) as exc:
        return status.make_status(
            "invalid",
            reason_code="invalid_json",
            detail=f"未来指令文件不可读：{type(exc).__name__}: {exc}",
            extra=details,
        )
    rows = [] if frame is None else list(frame.to_dict("records"))
    mismatch = _rows_source_date_mismatch(rows, source_date=current_date)
    if mismatch is not None:
        return status.make_status(
            "invalid",
            reason_code="instruction_source_date_mismatch",
            detail=mismatch,
            extra=details,
        )
    declared = counts.get("instructions_count")
    base = {**details, "execution_date": execution_date, "instruction_count": len(rows)}
    if declared is not None and declared != len(rows):
        return status.make_status(
            "invalid",
            reason_code="count_mismatch",
            detail=f"记录 instructions_count={declared} 与文件行数 {len(rows)} 不一致",
            extra=base,
        )
    if declared is None:
        base["instructions"] = rows
        return status.make_status(
            "degraded",
            reason_code="recovered_generation_count_missing",
            detail="恢复代缺 instructions_count；未来指令文件已按 source_date 核对",
            verification_level="authority_verified",
            extra=base,
        )
    if not rows:
        return status.make_status(
            "empty_valid",
            detail=f"未来执行日 {execution_date} 指令文件为合法空",
            verification_level="authority_verified",
            extra=base,
        )
    base["instructions"] = rows
    return status.make_status(
        "available",
        detail=f"未来执行日 {execution_date} 共 {len(rows)} 条指令",
        verification_level="authority_verified",
        extra=base,
    )


def project_pending(
    *,
    present: bool,
    payload: Any,
    parse_error: str | None,
    declared_count: int | None,
    label: str,
    supported: bool = True,
) -> dict[str, Any]:
    """pending 买/卖队列投影（非 list 与计数冲突 invalid；合法 [] 才 empty_valid）。"""
    if not supported:
        return status.make_status(
            "not_available", reason_code="unsupported", detail=f"{label}：远端来源不支持详细读取"
        )
    if not present:
        return status.make_status("not_available", reason_code="no_source", detail=f"{label} 文件缺失")
    if parse_error is not None:
        return status.make_status(
            "invalid", reason_code=parse_error, detail=f"{label} JSON 损坏"
        )
    if not isinstance(payload, list):
        return status.make_status(
            "invalid",
            reason_code="pending_not_list",
            detail=f"{label} 顶层类型为 {type(payload).__name__}，契约要求 list",
        )
    if declared_count is not None and declared_count != len(payload):
        return status.make_status(
            "invalid",
            reason_code="count_mismatch",
            detail=f"{label} 记录计数 {declared_count} 与队列长度 {len(payload)} 不一致",
        )
    if declared_count is None and payload:
        return status.make_status(
            "degraded",
            reason_code="recovered_generation_count_missing",
            detail=f"{label} 恢复代缺计数；队列本身为合法 list，行数 {len(payload)}",
            extra={"pending_count": len(payload), "pending": payload},
        )
    if not payload:
        return status.make_status(
            "empty_valid", detail=f"{label} 合法为空", extra={"pending_count": 0, "pending": []}
        )
    return status.make_status(
        "available",
        detail=f"{label} 共 {len(payload)} 条",
        extra={"pending_count": len(payload), "pending": payload},
    )


def project_rebalance(
    *, present: bool, payload: Any, parse_error: str | None, supported: bool = True
) -> dict[str, Any]:
    """调仓状态五字段投影（原值，不重算；非 dict invalid）。"""
    if not supported:
        return status.make_status(
            "not_available", reason_code="unsupported", detail="调仓状态：远端来源不支持详细读取"
        )
    if not present:
        return status.make_status(
            "not_available", reason_code="no_source", detail="缺 runs/rebalance_state.json"
        )
    if parse_error is not None:
        return status.make_status(
            "invalid", reason_code=parse_error, detail="rebalance_state.json JSON 损坏"
        )
    if not isinstance(payload, Mapping):
        return status.make_status(
            "invalid",
            reason_code="rebalance_not_dict",
            detail=f"rebalance_state 顶层类型为 {type(payload).__name__}，契约要求 dict",
        )
    return status.make_status(
        "available",
        detail="调仓状态原值（不重算）",
        extra={"rebalance": dict(payload)},
    )
