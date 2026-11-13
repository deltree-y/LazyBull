# -*- coding: utf-8 -*-
"""P7 T1 计数守恒、历史有效块、压力诊断与三向 verdict（设计 §6.2/§6.5）。

硬规则
======
- **计数守恒先行**：``intent_total = resolved_filled_count + unfilled_group_count +
  unresolved_count``；不守恒即拒绝生成经济 verdict（硬失败，不是冻结）；
- **可识别性先于经济判定**：任一核心格 ``unresolved_count > 0`` 即
  ``not_estimable / inconclusive_frozen``，压力情景与可解析子样本诊断不得覆盖；
- 5% 只是数据质量告警分级，不提供任何放行额度；
- 仅验收 P-1 组件门与功效；组件样本/功效不足一律冻结，不得记经济失败；
- 完整复合门（CP 上界、λ 选择、独立验证流）在本任务固定为 ``not_tested_in_T1``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

STATE_PASS = "pass"
STATE_FAIL = "fail"
STATE_FROZEN = "inconclusive_frozen"
STATE_REFUSED = "refused_count_nonconservation"

FULL_GATE_STATUS_T1 = "not_tested_in_T1"
TRADING_DAYS_PER_YEAR = 250.0
DATA_QUALITY_WARN_SHARE = 0.05


@dataclass
class BlockAssignment:
    """历史有效块归属（归块键 = 计划入场日 T+1）。"""

    block_index: np.ndarray
    in_guard: np.ndarray
    n_blocks: int
    block_start: np.ndarray
    block_end: np.ndarray
    guard_end: np.ndarray
    block_days: int = 20
    guard_days: int = 10


def assign_blocks(
    planned_entry: np.ndarray,
    *,
    timeline_start: int,
    timeline_end: int,
    block_days: int,
    guard_days: int,
) -> BlockAssignment:
    """把计划入场日映射到 (20 日块 + 10 日 guard) 网格。"""
    stride = int(block_days) + int(guard_days)
    length = int(timeline_end) - int(timeline_start) + 1
    n_blocks = 0 if length <= 0 else int(np.ceil(length / stride))
    planned = np.asarray(planned_entry, dtype=np.int64)
    relative = planned - int(timeline_start)
    block_index = np.full(planned.size, -1, dtype=np.int64)
    in_guard = np.zeros(planned.size, dtype=bool)
    inside = (relative >= 0) & (relative < length)
    cycle = np.where(inside, relative // stride, -1)
    offset = np.where(inside, relative % stride, -1)
    in_block = inside & (offset < int(block_days))
    block_index[in_block] = cycle[in_block]
    in_guard[inside & ~in_block] = True
    starts = np.arange(n_blocks, dtype=np.int64) * stride
    ends = starts + int(block_days) - 1
    guards = np.minimum(ends + int(guard_days), length - 1)
    return BlockAssignment(
        block_index=block_index,
        in_guard=in_guard,
        n_blocks=n_blocks,
        block_start=starts + int(timeline_start),
        block_end=ends + int(timeline_start),
        guard_end=guards + int(timeline_start),
        block_days=int(block_days),
        guard_days=int(guard_days),
    )


def effective_blocks(
    assignment: BlockAssignment,
    resolved_mask: np.ndarray,
    exit_pos: np.ndarray,
) -> Dict[str, Any]:
    """有效块 = 至少一个可解析匹配组，且其成熟窗口不越过本块 guard 末端的块。"""
    block_index = assignment.block_index
    resolved = np.asarray(resolved_mask, dtype=bool)
    usable = resolved & (block_index >= 0) & (np.asarray(exit_pos) >= 0)
    if not usable.any():
        return {
            "blocks_total": int(assignment.n_blocks),
            "effective_history_block": 0,
            "block_years": 0.0,
            "resolved_groups_in_blocks": 0,
            "resolved_groups_in_guard": int((resolved & assignment.in_guard).sum()),
            "resolved_groups_beyond_guard": 0,
        }
    covered = block_index[usable]
    exits = np.asarray(exit_pos, dtype=np.int64)[usable]
    guard_end = assignment.guard_end[covered]
    inside = exits <= guard_end
    counts = np.bincount(covered[inside], minlength=assignment.n_blocks)
    effective = int((counts > 0).sum())
    blocks_per_year = TRADING_DAYS_PER_YEAR / (assignment.block_days + assignment.guard_days)
    return {
        "blocks_total": int(assignment.n_blocks),
        "effective_history_block": effective,
        "block_years": round(effective / blocks_per_year, 3) if blocks_per_year else 0.0,
        "resolved_groups_in_blocks": int(usable.sum()),
        "resolved_groups_in_guard": int((resolved & assignment.in_guard).sum()),
        "resolved_groups_beyond_guard": int((~inside).sum()),
    }


def stress_statistics(
    status: np.ndarray,
    event_net: np.ndarray,
    control_net: np.ndarray,
) -> Dict[str, Any]:
    """两种有限压力情景与两种单侧诊断（`diagnostic_only=true`，不参与判定）。"""
    status = np.asarray(status)
    resolved = status == 0
    unfilled = status == 1
    unresolved = status == 2

    def statistic(event_value: Optional[float], control_value: Optional[float]) -> Optional[float]:
        """已解析保留实际、未成交保留真实 0、**只有未知部分**套用压力常数。

        压力情景只替换无法观测的一侧：未给出的那侧取 0（无压力），给出的那侧取情景值
        （T1 评审 R11）。
        """
        actual = np.asarray(event_net, dtype=np.float64) - np.asarray(control_net, dtype=np.float64)
        stressed = (0.0 if event_value is None else float(event_value)) - (
            0.0 if control_value is None else float(control_value)
        )
        values = np.where(resolved, actual, np.where(unfilled, 0.0, stressed))
        if values.size == 0 or not np.isfinite(values).all():
            return None
        return float(values.mean())

    def resolved_only(
        event_value: Optional[float] = None, control_value: Optional[float] = None
    ) -> Optional[float]:
        """仅已解析子样本的**实际**残差均值（不施加任何压力）。"""
        if not resolved.any():
            return None
        values = (
            np.asarray(event_net, dtype=np.float64) - np.asarray(control_net, dtype=np.float64)
        )[resolved]
        values = values[np.isfinite(values)]
        return float(values.mean()) if values.size else None

    return {
        "diagnostic_only": True,
        "intent_total": int(status.size),
        "unresolved_count": int(unresolved.sum()),
        "cases": {
            "event_minus_100__control_plus_100": statistic(-1.0, 1.0),
            "reversed_event_plus_100__control_minus_100": statistic(1.0, -1.0),
        },
        "single_side_diagnostics": {
            "event_minus_100_only": statistic(-1.0, None),
            "control_plus_100_only": statistic(None, 1.0),
            "event_plus_100_only": statistic(1.0, None),
            "control_minus_100_only": statistic(None, -1.0),
        },
        "resolved_only_diagnostic": {
            "note": "仅可解析子样本口径，不替代全意图裁决（存在不可解析组时仍为冻结）",
            "event_minus_100__control_plus_100": resolved_only(-1.0, 1.0),
            "reversed_event_plus_100__control_minus_100": resolved_only(1.0, -1.0),
            "control_plus_100_only": resolved_only(None, 1.0),
            "event_minus_100_only": resolved_only(-1.0, None),
        },
        "forbidden_uses": "不得作为识别上下界、不得参与 pass/fail、不得解除冻结、不得作为临界值依据",
    }


def cell_state(
    counts: Dict[str, int],
    *,
    effective_blocks: int,
    min_blocks: int,
    statistic_intent: Optional[float],
    critical_value: Optional[float],
    power: Dict[str, Any],
) -> Dict[str, Any]:
    """单格状态判定（先可识别性、再样本/功效、最后组件经济门）。"""
    conserved = counts["intent_total"] == (
        counts["resolved_filled_count"]
        + counts["unfilled_group_count"]
        + counts["unresolved_count"]
    )
    unresolved = int(counts["unresolved_count"])
    intents = int(counts["intent_total"])
    share = (unresolved / intents) if intents else None
    mde = power.get("mde")
    power_ok = mde is not None and not power.get("grid_exhausted", False)
    gate = (
        statistic_intent is not None
        and critical_value is not None
        and float(statistic_intent) > float(critical_value)
    )
    decisions = (
        (not conserved, STATE_REFUSED, "count_nonconservation"),
        (intents == 0, STATE_FROZEN, "empty_intent_cell"),
        (unresolved > 0, STATE_FROZEN, "unresolved_groups_present"),
        (effective_blocks < int(min_blocks), STATE_FROZEN, "insufficient_effective_blocks"),
        (statistic_intent is None, STATE_FROZEN, "statistic_not_estimable"),
        (not power_ok, STATE_FROZEN, "power_grid_exhausted"),
        (gate, STATE_PASS, "component_gate_pass"),
    )
    state = STATE_FAIL
    reasons: List[str] = ["component_gate_fail"]
    for condition, candidate, reason in decisions:
        if condition:
            state = candidate
            reasons = [reason]
            break
    return {
        "counts_conserved": bool(conserved),
        "missing_identification_status": (
            "complete" if unresolved == 0 and conserved else "unresolved"
        ),
        "unresolved_share": share,
        "data_quality_warning": (
            "warn" if share is not None and share > DATA_QUALITY_WARN_SHARE else "ok"
        ),
        "component_gate_pass": bool(gate),
        "min_blocks_ok": effective_blocks >= int(min_blocks),
        "power_ok": bool(power_ok),
        "mde": mde,
        "state": state,
        "reasons": reasons,
    }


def build_verdict(
    cells: Sequence[Dict[str, Any]],
    *,
    core_cells: Sequence[Tuple[str, int]],
    min_blocks: int,
    run_id: str,
    prereg_id: str,
    prereg_sha256: str,
    evidence_level: str = "H",
) -> Dict[str, Any]:
    """按 prereg 三向规则汇总 P-1 组件 verdict（完整复合门留待 T3）。"""
    by_key = {(cell["source"], int(cell["horizon"])): cell for cell in cells}
    core_keys = [(source, int(horizon)) for source, horizon in core_cells]
    missing = [key for key in core_keys if key not in by_key]
    if missing:
        raise ValueError(f"缺少核心格：{missing}")

    refused = [
        cell["source"] + f"/{cell['horizon']}"
        for cell in cells
        if cell["gate_state"] == STATE_REFUSED
    ]
    if refused:
        verdict = None
        state = STATE_REFUSED
        note = "计数不守恒：拒绝生成经济 verdict（硬失败，不是冻结）"
    else:
        core_states = [by_key[key]["gate_state"] for key in core_keys]
        if any(item == STATE_FROZEN for item in core_states):
            verdict, state = STATE_FROZEN, STATE_FROZEN
            note = "至少一个核心格不可估计/样本或功效不足：按可识别性优先冻结"
        elif all(item == STATE_PASS for item in core_states):
            verdict, state = STATE_PASS, STATE_PASS
            note = "三源核心格 P-1 组件门均通过（完整复合门未在 T1 检验）"
        else:
            verdict, state = STATE_FAIL, STATE_FAIL
            note = "功效足够且可识别，但至少一个核心格组件门不通过"

    payload = {
        "schema": "p7_t1_p1_verdict/v1",
        "run_id": run_id,
        "prereg_id": prereg_id,
        "prereg_sha256": prereg_sha256,
        "task": "P7-T1 P-1 同域静态可执行性",
        "verdict": verdict,
        "verdict_state": state,
        "verdict_note": note,
        "full_gate_status": FULL_GATE_STATUS_T1,
        "full_gate_note": "完整复合门（CP 上界、λ 选择、独立验证流）留待 T3，本任务不做任何通过声明",
        "evidence_level": evidence_level,
        "historical_virtual_clock": True,
        "not_forward_evidence": True,
        "not_production_authorization": True,
        "not_real_clock_evidence": True,
        "core_cells": [list(key) for key in core_keys],
        "counts_conserved_all": all(cell["counts_conserved"] for cell in cells),
        "cells": {cell["source"] + f"/{cell['horizon']}": cell for cell in cells},
    }
    return payload


def assert_counts_conserved(cells: Sequence[Dict[str, Any]]) -> None:
    """计数不守恒一律硬失败（拒绝生成经济 verdict）。"""
    for cell in cells:
        counts = cell["counts"]
        total = counts["intent_total"]
        parts = (
            counts["resolved_filled_count"]
            + counts["unfilled_group_count"]
            + counts["unresolved_count"]
        )
        if total != parts:
            raise ValueError(
                f"计数不守恒：{cell['source']}/{cell['horizon']} "
                f"intent_total={total} parts={parts}"
            )
