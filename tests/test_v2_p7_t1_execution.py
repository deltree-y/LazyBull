# -*- coding: utf-8 -*-
"""P7 T1 共同入离场、费用与总回报公式的定点回归。

口径来源：`docs/plans/v2/p7_design_20261111.md` §6.2~§6.3（冻结）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v2_p7.t1 import execution  # noqa: E402

COST = {
    "commission_rate": 0.0001954,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.0005,
}
MIN_TICKET = 25588.5363


class FakePanel:
    """仅供单元测试的最小面板（六只股票 × 若干交易日）。"""

    def __init__(self, n_days: int, n_stocks: int = 6) -> None:
        shape = (n_days, n_stocks)
        self.n_days = n_days
        self.open_u = np.full(shape, 10.0)
        self.open_adj = np.full(shape, 10.0)
        self.pre_close = np.full(shape, 10.0)
        self.high = np.full(shape, 10.1)
        self.low = np.full(shape, 9.9)
        self.suspended = np.zeros(shape, dtype=bool)
        self.limit_up = np.zeros(shape, dtype=bool)
        self.limit_down = np.zeros(shape, dtype=bool)
        self.share_mult = np.ones(shape)
        self.share_unknown = np.zeros(shape, dtype=np.int32)


def _run(panel: FakePanel, horizon: int = 2, members: np.ndarray | None = None):
    if members is None:
        members = np.tile(np.arange(6), (1, 1))
    signal = np.array([0], dtype=np.int64)
    return execution.simulate_horizon(
        panel,
        members,
        signal,
        np.array([0], dtype=np.int16),
        ["matched"],
        horizon=horizon,
        min_ticket=MIN_TICKET,
        cost=COST,
        lot_size=100,
        entry_max_defer=10,
        exit_max_defer=10,
    )


def test_common_entry_defers_until_all_six_executable() -> None:
    panel = FakePanel(30)
    panel.limit_up[1, 5] = True
    result = _run(panel)
    assert result.status[0] == execution.STATUS_RESOLVED
    assert int(result.entry_pos[0]) == 2
    assert int(result.entry_defer[0]) == 1
    assert int(result.exit_pos[0]) == 4
    assert int(result.exit_defer[0]) == 0


def test_no_common_entry_is_unfilled_with_true_zero() -> None:
    panel = FakePanel(30)
    panel.limit_up[:, 5] = True
    result = _run(panel)
    assert result.status[0] == execution.STATUS_UNFILLED
    assert result.reason[0] == "no_common_entry_within_10d"
    values = execution.intent_residual(result)
    assert values[0] == 0.0
    assert np.isnan(result.event_net[0])


def test_exit_defer_then_endpoint_unresolved() -> None:
    panel = FakePanel(40)
    panel.limit_down[3, 3] = True
    deferred = _run(panel)
    assert deferred.status[0] == execution.STATUS_RESOLVED
    assert int(deferred.exit_pos[0]) == 4
    assert int(deferred.exit_defer[0]) == 1

    stuck = FakePanel(40)
    stuck.limit_down[:, 3] = True
    unresolved = _run(stuck)
    assert unresolved.status[0] == execution.STATUS_UNRESOLVED
    assert unresolved.reason[0] == "endpoint_unresolved"
    assert np.isnan(execution.intent_residual(unresolved)[0])


def _expected_net_return(
    buy_price: float, sell_price: float, adj_ratio: float
) -> tuple[float, float, int]:
    """独立按契约公式复算：预算 = V + 100P + 预估买入费用，整手向下。"""
    budget = MIN_TICKET + 100 * buy_price
    estimate = max(budget * COST["commission_rate"], COST["min_commission"]) + (
        budget * COST["slippage"]
    )
    total_budget = budget + estimate
    shares = int(total_budget / buy_price / 100) * 100
    buy_amount = shares * buy_price
    buy_cost = max(buy_amount * COST["commission_rate"], COST["min_commission"]) + (
        buy_amount * COST["slippage"]
    )
    sell_amount = shares * sell_price
    sell_cost = (
        max(sell_amount * COST["commission_rate"], COST["min_commission"])
        + sell_amount * COST["stamp_tax"]
        + sell_amount * COST["slippage"]
    )
    exit_value = buy_amount * adj_ratio
    net = (exit_value - sell_cost) / (buy_amount + buy_cost) - 1.0
    gross = exit_value / buy_amount - 1.0
    return net, gross, shares


def test_cost_and_return_formula_matches_contract() -> None:
    panel = FakePanel(30)
    panel.open_u[:, :] = 10.0
    panel.open_adj[:, :] = 10.0
    panel.open_u[1, :] = 10.0
    panel.open_adj[1, :] = 10.0
    panel.open_u[3, :] = 11.0
    panel.open_adj[3, :] = 11.0
    result = _run(panel)
    expected_net, expected_gross, expected_shares = _expected_net_return(10.0, 11.0, 1.1)
    assert int(result.shares[0, 0]) == expected_shares
    assert result.shares[0, 0] % 100 == 0
    assert result.buy_amount[0, 0] >= MIN_TICKET
    assert result.net[0, 0] == pytest.approx(expected_net, rel=1e-12)
    assert result.gross[0, 0] == pytest.approx(expected_gross, rel=1e-12)
    # 六股同价同收益 ⇒ 净残差为 0
    assert result.residual[0] == pytest.approx(0.0, abs=1e-12)


def test_event_control_residual_uses_equal_weight_basket() -> None:
    panel = FakePanel(30)
    panel.open_u[1, :] = 10.0
    panel.open_adj[1, :] = 10.0
    panel.open_u[3, 0] = 12.0
    panel.open_adj[3, 0] = 12.0
    panel.open_u[3, 1:] = 11.0
    panel.open_adj[3, 1:] = 11.0
    result = _run(panel)
    event_net, event_gross, _ = _expected_net_return(10.0, 12.0, 1.2)
    control_net, _, _ = _expected_net_return(10.0, 11.0, 1.1)
    assert result.event_net[0] == pytest.approx(event_net, rel=1e-12)
    assert result.control_net[0] == pytest.approx(control_net, rel=1e-12)
    assert result.residual[0] == pytest.approx(event_net - control_net, rel=1e-12)
    assert result.gross_residual[0] == pytest.approx(event_gross - 0.1, rel=1e-12)


def test_missing_price_chain_is_not_zero_filled() -> None:
    panel = FakePanel(30)
    panel.open_u[3:14, 2] = np.nan
    panel.open_adj[3:14, 2] = np.nan
    result = _run(panel)
    assert result.status[0] == execution.STATUS_UNRESOLVED
    assert result.reason[0] == "endpoint_unresolved"
    assert np.isnan(execution.intent_residual(result)[0])
