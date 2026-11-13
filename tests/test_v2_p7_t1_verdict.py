# -*- coding: utf-8 -*-
"""P7 T1 三向 verdict 与六条反例自测。

覆盖验收门 §4.8：
① 相同可观测收益与缺失率、不同缺失结果 ⇒ 冻结；
② 对照收益 +150% ⇒ 不得放行（有不可解析组仍冻结）；
③ 改变压力赋值或压力结果跨临界值 ⇒ 经济 verdict 不变；
④ 未知已持有头寸 vs 真实未建仓 0 的区分；
⑤ 零缺失的正常评价路径；
⑥ 计数不守恒拒绝生成经济 verdict。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v2_p7.t1 import execution, randomization, verdict  # noqa: E402

CORE_CELLS = (("forecast", 10), ("express", 10), ("fina_report", 10))


def _cell(
    source: str,
    horizon: int,
    *,
    intents: int,
    resolved: int,
    unfilled: int,
    unresolved: int,
    blocks: int = 120,
    statistic: float | None = 0.02,
    critical: float | None = 0.01,
    mde: float | None = 1.0,
    state: str | None = None,
    stress: dict | None = None,
) -> dict:
    counts = {
        "intent_total": intents,
        "resolved_filled_count": resolved,
        "unfilled_group_count": unfilled,
        "unresolved_count": unresolved,
    }
    power = {"mde": mde, "grid_exhausted": mde is None}
    info = verdict.cell_state(
        counts,
        effective_blocks=blocks,
        min_blocks=60,
        statistic_intent=statistic,
        critical_value=critical,
        power=power,
    )
    if state is not None:
        info["state"] = state
        info["reasons"] = [state]
    return {
        "source": source,
        "horizon": horizon,
        "counts": counts,
        "counts_conserved": intents == resolved + unfilled + unresolved,
        "gate_state": info["state"],
        "missing_identification_status": info["missing_identification_status"],
        "blocks": {"effective_history_block": blocks},
        "state": info,
        "stress": stress or {"diagnostic_only": True, "cases": {}},
    }


def _verdict(cells, *, min_blocks: int = 60) -> dict:
    return verdict.build_verdict(
        cells,
        core_cells=CORE_CELLS,
        min_blocks=min_blocks,
        run_id="unit",
        prereg_id="unit",
        prereg_sha256="0" * 64,
    )


def _healthy_cells() -> list[dict]:
    return [
        _cell(source, 10, intents=100, resolved=95, unfilled=5, unresolved=0)
        for source, _ in CORE_CELLS
    ] + [
        _cell(source, 5, intents=100, resolved=95, unfilled=5, unresolved=0)
        for source, _ in CORE_CELLS
    ]


def test_pass_requires_zero_missing_and_power() -> None:
    payload = _verdict(_healthy_cells())
    assert payload["verdict"] == verdict.STATE_PASS
    assert payload["full_gate_status"] == verdict.FULL_GATE_STATUS_T1
    assert payload["counts_conserved_all"] is True


def test_counterexample_1_same_rate_different_missing_still_frozen() -> None:
    """① 可观测收益与缺失率相同、缺失结果不同 ⇒ 两者都必须冻结。"""
    base = dict(intents=100, resolved=99, unfilled=0, unresolved=1)
    first = [_cell(source, 10, statistic=0.5, **base) for source, _ in CORE_CELLS]
    second = [_cell(source, 10, statistic=-0.5, **base) for source, _ in CORE_CELLS]
    for cells in (first, second):
        payload = _verdict(cells)
        assert payload["verdict"] == verdict.STATE_FROZEN
        assert "unresolved_groups_present" in payload["cells"]["forecast/10"]["state"]["reasons"]


def test_counterexample_2_extreme_control_return_never_rescues() -> None:
    """② 对照收益 +150% 不得放行；存在不可解析组时仍冻结。"""
    stressed = {
        "diagnostic_only": True,
        "cases": {"control_plus_100_only": 1.5, "event_minus_100_only": -1.0},
    }
    frozen = [
        _cell(source, 10, intents=100, resolved=99, unfilled=0, unresolved=1, stress=stressed)
        for source, _ in CORE_CELLS
    ]
    assert _verdict(frozen)["verdict"] == verdict.STATE_FROZEN

    identifiable = [
        _cell(
            source,
            10,
            intents=100,
            resolved=100,
            unfilled=0,
            unresolved=0,
            statistic=-0.01,
            critical=0.01,
            stress=stressed,
        )
        for source, _ in CORE_CELLS
    ]
    payload = _verdict(identifiable)
    assert payload["verdict"] == verdict.STATE_FAIL
    assert payload["verdict"] != verdict.STATE_PASS


def test_counterexample_3_stress_values_do_not_move_economic_verdict() -> None:
    """③ 改变压力赋值或压力结果跨临界值，经济 verdict 不变。"""

    def build(value: float) -> list[dict]:
        return [
            _cell(
                source,
                10,
                intents=100,
                resolved=100,
                unfilled=0,
                unresolved=0,
                stress={"diagnostic_only": True, "cases": {"stress": value}},
            )
            for source, _ in CORE_CELLS
        ]

    assert _verdict(build(-2.0))["verdict"] == _verdict(build(5.0))["verdict"] == verdict.STATE_PASS

    def build_failing(value: float) -> list[dict]:
        cells = build(value)
        for cell in cells:
            cell["gate_state"] = verdict.STATE_FAIL
            cell["state"]["state"] = verdict.STATE_FAIL
        return cells

    assert (
        _verdict(build_failing(-2.0))["verdict"]
        == _verdict(build_failing(5.0))["verdict"]
        == verdict.STATE_FAIL
    )


def test_counterexample_4_unknown_position_versus_real_zero() -> None:
    """④ 未知已持有头寸（unresolved）与真实未建仓 0（unfilled）必须分开。"""
    status = np.array(
        [execution.STATUS_RESOLVED, execution.STATUS_UNFILLED, execution.STATUS_UNRESOLVED],
        dtype=np.int8,
    )
    result = execution.HorizonResult(
        horizon=10,
        status=status,
        reason=["resolved", "no_common_entry_within_10d", "endpoint_unresolved"],
        entry_pos=np.array([1, -1, 2], dtype=np.int32),
        exit_pos=np.array([11, -1, -1], dtype=np.int32),
        entry_defer=np.array([0, -1, 0], dtype=np.int16),
        exit_defer=np.array([0, -1, -1], dtype=np.int16),
        gross=np.full((3, 6), np.nan),
        net=np.full((3, 6), np.nan),
        shares=np.zeros((3, 6), dtype=np.int64),
        sell_shares=np.zeros((3, 6), dtype=np.float64),
        buy_amount=np.full((3, 6), np.nan),
        sell_amount=np.full((3, 6), np.nan),
        buy_cost=np.full((3, 6), np.nan),
        sell_cost=np.full((3, 6), np.nan),
        residual=np.array([0.03, np.nan, np.nan]),
        gross_residual=np.array([0.03, np.nan, np.nan]),
        event_net=np.array([0.03, np.nan, np.nan]),
        control_net=np.array([0.0, np.nan, np.nan]),
        stock_entry_ok=np.zeros((3, 6), dtype=bool),
        stock_exit_ok=np.zeros((3, 6), dtype=bool),
        entry_gap=np.full((3, 6), np.nan),
        exit_gap=np.full((3, 6), np.nan),
        one_price_entry=np.zeros((3, 6), dtype=bool),
        one_price_exit=np.zeros((3, 6), dtype=bool),
    )
    values = execution.intent_residual(result)
    assert values[1] == 0.0
    assert np.isnan(values[2])
    statistics = randomization.observed_statistic(
        values,
        status == execution.STATUS_RESOLVED,
        status == execution.STATUS_UNFILLED,
    )
    assert statistics.statistic_intent is None
    counts = result.counts()
    assert counts == {
        "intent_total": 3,
        "resolved_filled_count": 1,
        "unfilled_group_count": 1,
        "unresolved_count": 1,
    }
    cell = verdict.cell_state(
        counts,
        effective_blocks=120,
        min_blocks=60,
        statistic_intent=0.03,
        critical_value=0.01,
        power={"mde": 1.0, "grid_exhausted": False},
    )
    assert cell["state"] == verdict.STATE_FROZEN
    assert cell["missing_identification_status"] == "unresolved"


def test_empty_intent_and_low_power_and_few_blocks_freeze() -> None:
    empty = verdict.cell_state(
        {
            "intent_total": 0,
            "resolved_filled_count": 0,
            "unfilled_group_count": 0,
            "unresolved_count": 0,
        },
        effective_blocks=0,
        min_blocks=60,
        statistic_intent=None,
        critical_value=None,
        power={"mde": None, "grid_exhausted": True},
    )
    assert empty["state"] == verdict.STATE_FROZEN
    assert "empty_intent_cell" in empty["reasons"]

    few = verdict.cell_state(
        {
            "intent_total": 10,
            "resolved_filled_count": 10,
            "unfilled_group_count": 0,
            "unresolved_count": 0,
        },
        effective_blocks=59,
        min_blocks=60,
        statistic_intent=0.05,
        critical_value=0.01,
        power={"mde": 1.0, "grid_exhausted": False},
    )
    assert few["state"] == verdict.STATE_FROZEN
    assert "insufficient_effective_blocks" in few["reasons"]

    weak = verdict.cell_state(
        {
            "intent_total": 10,
            "resolved_filled_count": 10,
            "unfilled_group_count": 0,
            "unresolved_count": 0,
        },
        effective_blocks=120,
        min_blocks=60,
        statistic_intent=0.05,
        critical_value=0.01,
        power={"mde": None, "grid_exhausted": True},
    )
    assert weak["state"] == verdict.STATE_FROZEN
    assert "power_grid_exhausted" in weak["reasons"]


def test_counterexample_6_count_nonconservation_refuses_verdict() -> None:
    """⑥ 计数不守恒 ⇒ 拒绝生成经济 verdict（硬失败，不是冻结）。"""
    cells = _healthy_cells()
    broken = _cell("forecast", 10, intents=100, resolved=95, unfilled=5, unresolved=0)
    broken["counts"]["unresolved_count"] = 1
    broken["counts_conserved"] = False
    broken["gate_state"] = verdict.STATE_REFUSED
    cells[0] = broken
    payload = _verdict(cells)
    assert payload["verdict"] is None
    assert payload["verdict_state"] == verdict.STATE_REFUSED
    with pytest.raises(ValueError):
        verdict.assert_counts_conserved(
            [{"source": "forecast", "horizon": 10, "counts": broken["counts"]}]
        )


def test_block_guard_prevents_window_sharing() -> None:
    """有效块要求成熟窗口不越过本块 guard 末端（不与邻块共享执行/成熟窗口）。"""
    assignment = verdict.assign_blocks(
        np.array([100, 101, 118, 119], dtype=np.int64),
        timeline_start=100,
        timeline_end=199,
        block_days=20,
        guard_days=10,
    )
    assert assignment.n_blocks == 4
    assert list(assignment.block_index) == [0, 0, 0, 0]
    assert not assignment.in_guard.any()
    resolved = np.ones(4, dtype=bool)
    inside = verdict.effective_blocks(assignment, resolved, np.array([110, 111, 128, 129]))
    outside = verdict.effective_blocks(assignment, resolved, np.array([130, 131, 140, 141]))
    assert inside["effective_history_block"] == 1
    assert inside["resolved_groups_beyond_guard"] == 0
    assert outside["effective_history_block"] == 0
    assert outside["resolved_groups_beyond_guard"] == 4

    guard = verdict.assign_blocks(
        np.array([120], dtype=np.int64),
        timeline_start=100,
        timeline_end=199,
        block_days=20,
        guard_days=10,
    )
    assert guard.block_index[0] == -1
    assert bool(guard.in_guard[0]) is True


def test_stress_statistics_are_diagnostic_only() -> None:
    # 压力只作用于未知组：已解析组保留实际残差，未成交组保留真实 0（T1 评审 R11）
    status = np.array(
        [
            execution.STATUS_RESOLVED,
            execution.STATUS_UNFILLED,
            execution.STATUS_UNRESOLVED,
        ]
    )
    payload = verdict.stress_statistics(
        status,
        np.array([0.10, np.nan, np.nan]),
        np.array([0.02, np.nan, np.nan]),
    )
    assert payload["diagnostic_only"] is True
    # 已解析 0.08 + 未成交 0 + 未知组压力 (−1 − 1 = −2) ⇒ 均值 −0.64
    assert payload["cases"]["event_minus_100__control_plus_100"] == pytest.approx(-0.64)
    # 反向情景：未知组取 (1 − (−1)) = +2 ⇒ 均值 (0.08 + 0 + 2) / 3
    assert payload["cases"]["reversed_event_plus_100__control_minus_100"] == pytest.approx(
        2.08 / 3
    )
    # 单侧：未知组取 (0 − 1) = −1 ⇒ 均值 (−0.92) / 3
    assert payload["single_side_diagnostics"]["control_plus_100_only"] == pytest.approx(-0.92 / 3)
    # 仅已解析子样本口径保持实际残差
    assert payload["resolved_only_diagnostic"][
        "event_minus_100__control_plus_100"
    ] == pytest.approx(0.08)
