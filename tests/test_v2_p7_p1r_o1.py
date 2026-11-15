# -*- coding: utf-8 -*-
"""P-1 O1 读数回归（R05 四项修正的复现反例）。

四个反例（对应 P7 整体检视 R05）
==============================
1. 回款必须用 ``event_net``：首笔 ``d_point`` 为负但 ``event_net`` 为正时，第二笔必须能买；
2. 占资必须等**实际** ``entry_pos``：计划入场日与真实入场日不同时，R2 按真实持有日算；
3. ``holding=OPEN`` 截止不得当已退出：``open_at_cutoff`` 与 R5 必须非零；
4. R3 = 逐日汇总超期仓位市值/本金，不是逐笔固定 ``min_ticket`` 分位。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest

from scripts.v2_p7.t1.p1_bounds import (
    FILL_CONFIRMED,
    HOLDING_CLOSED,
    HOLDING_NA,
    HOLDING_OPEN,
)
from scripts.v2_p7.t1.p1_o1 import constraint_readings


def _params(
    *, capital: float = 100.0, ticket: float = 100.0, slots: int = 6, horizon: int = 10
) -> Any:
    return SimpleNamespace(
        initial_capital=float(capital),
        min_ticket=float(ticket),
        max_concurrent_positions=int(slots),
        horizon_primary=int(horizon),
    )


def _intents(rows: List[dict]) -> Any:
    return SimpleNamespace(
        stock=np.asarray([row["stock"] for row in rows], dtype=np.int64),
        signal_pos=np.asarray([row.get("signal_pos", 0) for row in rows], dtype=np.int64),
        source_code=np.asarray([row.get("source", 0) for row in rows], dtype=np.int64),
        size=len(rows),
    )


def _result(rows: List[dict]) -> Any:
    return SimpleNamespace(
        fill=np.asarray([row.get("fill", FILL_CONFIRMED) for row in rows], dtype=np.int64),
        holding=np.asarray([row.get("holding", HOLDING_CLOSED) for row in rows], dtype=np.int64),
        entry_pos=np.asarray([row.get("entry", -1) for row in rows], dtype=np.int64),
        exit_pos=np.asarray([row.get("exit", -1) for row in rows], dtype=np.int64),
        event_net=np.asarray([row.get("event_net", 0.0) for row in rows], dtype=np.float64),
        d_point=np.asarray([row.get("d_point", 0.0) for row in rows], dtype=np.float64),
    )


def test_cash_return_uses_event_net_not_d_point() -> None:
    """反例①：``d_point=-0.5`` 但 ``event_net=+0.5`` 时，第二笔必须能买。"""
    rows = [
        {"stock": 1, "entry": 0, "exit": 2, "event_net": 0.5, "d_point": -0.5},
        {"stock": 2, "entry": 3, "exit": 5, "event_net": 0.0, "d_point": 0.0},
    ]
    readings = constraint_readings(
        _intents(rows), _result(rows), _params(), cutoff_pos=9
    )
    assert readings["counts"]["bought"] == 2
    assert readings["cash_return_basis"] == "event_net"


def test_entry_uses_actual_entry_pos() -> None:
    """反例②：实际入场晚于计划日时，占用率只从真实入场日起算。"""
    # 单一意图：signal_pos=0（计划 day1），真实 entry=2，exit=4；cutoff=4（5 个交易日）
    rows = [{"stock": 1, "signal_pos": 0, "entry": 2, "exit": 4, "event_net": 0.0}]
    readings = constraint_readings(
        _intents(rows), _result(rows), _params(capital=100.0, ticket=100.0), cutoff_pos=4
    )
    # 持仓日 = day2/3/4（day4 先释放再建仓，故当日不占资）⇒ 2/5 = 40%
    assert readings["R2_mean_capital_utilization"] == pytest.approx(2.0 / 5.0)
    assert readings["entry_basis"] == "entry_pos"


def test_open_position_is_not_treated_as_exited() -> None:
    """反例③：``holding=OPEN`` 且 exit=cutoff 时，R5 与 open_at_cutoff 必须非零。"""
    rows = [
        {
            "stock": 1,
            "entry": 2,
            "exit": 4,
            "event_net": -0.2,
            "holding": HOLDING_OPEN,
        }
    ]
    readings = constraint_readings(
        _intents(rows), _result(rows), _params(capital=100.0, ticket=100.0), cutoff_pos=4
    )
    assert readings["counts"]["open_at_cutoff"] == 1
    assert readings["R5_terminal_loss_stress"] == pytest.approx(1.0)  # 100/100 全按 0 估值
    assert readings["R5_open_capital"] == pytest.approx(100.0)
    # 未平仓的退出延迟只记已观察下界（4 − (2+10) < 0 → 0），不得记成实际退出
    assert readings["R4_exit_delay_p95_days"] == pytest.approx(0.0)


def test_r3_is_daily_aggregate_of_overdue_market_value() -> None:
    """反例④：六笔同时超期 ⇒ R3 为逐日汇总市值/本金（≈51.18%），不是 8.53% 的逐笔分位。"""
    capital = 300_000.0
    ticket = 25_588.536335721597
    rows = [
        {"stock": index, "entry": 0, "exit": 30, "event_net": 0.0}
        for index in range(1, 7)
    ]
    readings = constraint_readings(
        _intents(rows),
        _result(rows),
        _params(capital=capital, ticket=ticket, slots=6, horizon=10),
        cutoff_pos=40,
    )
    expected = 6.0 * ticket / capital
    assert readings["R3_overdue_lock_share_p95"] == pytest.approx(expected, rel=1e-9)
    assert readings["R3_overdue_days"] > 0
    assert readings["flags"]["R3_above_reference"] is True  # 51.18% > 10% 参考线（只标记不判）


def test_not_filled_and_na_holding_are_counted_separately() -> None:
    """未成交意图不进入约束账户；``holding=NA`` 不得混入 CLOSED/OPEN 计数。"""
    rows = [
        {"stock": 1, "entry": -1, "exit": -1, "fill": 1, "holding": HOLDING_NA},
        {"stock": 2, "entry": 1, "exit": 3, "event_net": 0.1},
    ]
    readings = constraint_readings(
        _intents(rows), _result(rows), _params(capital=100.0, ticket=100.0), cutoff_pos=5
    )
    assert readings["counts"]["not_filled"] == 1
    assert readings["counts"]["holding_na_intents"] == 1
    assert readings["counts"]["bought"] == 1
    assert readings["counts"]["open_at_cutoff"] == 0
