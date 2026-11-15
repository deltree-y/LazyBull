# -*- coding: utf-8 -*-
"""P-1 重开核心反例测试（预登记 §14 的 1~6、8 条）。

合成面板不触碰真实 data/；每条对应用户可执行的验收反例。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.v2_p7.t1 import p1_artifacts, p1_bounds, p1_inputs
from scripts.v2_p7.t1.p1_inputs import BenchmarkBook, Extras, IntentTable
from scripts.v2_p7.t1.p1_params import CostRates, P1RParams


def _params(**overrides) -> P1RParams:
    base = dict(
        repo_root=SimpleNamespace(),
        prereg_path=SimpleNamespace(),
        prereg_sha256="a" * 64,
        prereg_id="p7-p1-redesign-20261113",
        criteria_digest="b" * 64,
        evidence_root=SimpleNamespace(),
        run_id="test",
        common_data_cutoff="20261029",
        evaluation_start="20111107",
        evaluation_end="20261029",
        evaluation_trading_days=130,
        horizon_primary=5,
        horizon_sensitivity=5,
        entry_max_defer=10,
        lot_size=100,
        min_ticket=25588.536335721597,
        cost=CostRates(0.0001954, 5.0, 0.0005, 0.0005),
        initial_capital=300000.0,
        max_concurrent_positions=6,
        maturity_marker_days=60,
        bootstrap_main_block=90,
        bootstrap_aux_block=90,
        bootstrap_replicates=100,
        bootstrap_sensitivity=(45, 180),
        calibration_samples=2,
        calibration_replicates=20,
        seeds={
            "main_stationary": 42,
            "aux_circular": 1042,
            "layer2_calibration_base": 2042,
            "pilot": 3042,
            "sensitivity": 4042,
        },
    )
    base.update(overrides)
    return P1RParams(**base)


def _panel(n_days: int, n_stocks: int = 2) -> SimpleNamespace:
    open_u = np.full((n_days, n_stocks), 10.0)
    open_adj = np.full((n_days, n_stocks), 10.0)
    close_adj = np.full((n_days, n_stocks), 10.0)
    suspended = np.zeros((n_days, n_stocks), dtype=bool)
    limit_up = np.zeros((n_days, n_stocks), dtype=bool)
    limit_down = np.zeros((n_days, n_stocks), dtype=bool)
    industry = np.full((n_days, n_stocks), 1, dtype=np.int32)
    return SimpleNamespace(
        n_days=n_days,
        n_stocks=n_stocks,
        open_u=open_u,
        open_adj=open_adj,
        close_adj=close_adj,
        suspended=suspended,
        limit_up=limit_up,
        limit_down=limit_down,
        industry_matrix=industry,
        share_mult=np.ones((n_days, n_stocks)),
        share_unknown=np.zeros((n_days, n_stocks), dtype=np.int32),
        has_row=np.ones((n_days, n_stocks), dtype=bool),
        listed=np.ones((n_days, n_stocks), dtype=bool),
    )


def _extras(panel: SimpleNamespace, **overrides) -> Extras:
    shape = (panel.n_days, panel.n_stocks)
    data = dict(
        has_row=panel.has_row,
        up_limit=np.full(shape, 11.0, dtype=np.float32),
        down_limit=np.full(shape, 9.0, dtype=np.float32),
        suspend_flag=np.zeros(shape, dtype=bool),
        stats={},
    )
    data.update(overrides)
    return Extras(**data)


def _benchmark(panel: SimpleNamespace) -> BenchmarkBook:
    members = {day: {1: np.array([0, 1], dtype=np.int32)} for day in range(panel.n_days)}
    return BenchmarkBook.build(panel, members)


def _intents(stocks, signals) -> IntentTable:
    return IntentTable(
        stock=np.asarray(stocks, dtype=np.int32),
        signal_pos=np.asarray(signals, dtype=np.int32),
        source_code=np.zeros(len(stocks), dtype=np.int8),
        fingerprint=["f"] * len(stocks),
        stats={},
    )


def test_late_exit_not_truncated_at_60_days():
    """反例 1：第 105 日才可卖不得被 60 日截断。"""
    panel = _panel(130)
    panel.open_u[105, 0] = 10.5
    panel.open_adj[105, 0] = 10.5
    panel.close_adj[105, 0] = 10.5
    for day in range(5, 105):
        panel.suspended[day, 0] = True
    extras = _extras(panel)
    branches = p1_bounds._exit_branches(panel, extras, 5, 0, 129)
    assert branches[0]["exit"] == 105
    assert branches[0]["cutoff"] is False


def test_tail_observation_not_zero_and_not_permanent():
    """反例 2：截止未平仓按估值计，不记 0，也不诊断为永久无法卖出。"""
    panel = _panel(20)
    panel.close_adj[19, 0] = 9.5
    panel.open_adj[19, 0] = 9.5
    for day in range(5, 20):
        panel.suspended[day, 0] = True
    extras = _extras(panel)
    params = _params()
    intents = _intents([0], [0])
    result = p1_bounds.evaluate_intents(
        panel, extras, _benchmark(panel), params, intents, max_paths=64
    )
    assert result.holding[0] == p1_bounds.HOLDING_OPEN
    assert np.isfinite(result.lo[0]) and np.isfinite(result.hi[0])
    assert result.lo[0] != 0.0


def test_uncovered_branch_is_unbounded_and_blocks_negative():
    """反例 3：未覆盖分支 ⇒ unbounded ⇒ 禁用 negative，但仍 continue。"""
    panel = _panel(20)
    panel.has_row[1, 0] = False
    for day in range(1, 12):
        panel.has_row[day, 0] = False
    extras = _extras(panel, up_limit=np.full((20, 2), np.nan), down_limit=np.full((20, 2), np.nan))
    params = _params()
    result = p1_bounds.evaluate_intents(
        panel, extras, _benchmark(panel), params, _intents([0], [0]), max_paths=64
    )
    assert result.unbounded[0]
    verdict = p1_bounds.decide_verdict(
        calibration_passed=True,
        method_sensitive=False,
        unbounded=True,
        no_valid_interval=False,
        final_lo=-0.01,
        final_hi=-0.005,
        negative_power=0.9,
        intent_total=1,
    )
    assert verdict["verdict"] == "continue_limited"


def test_single_exception_does_not_reject_batch():
    """反例 4：单条例外不得整批拒绝证据（结构与数据例外分离）。"""
    panel = _panel(20)
    panel.has_row[1, 0] = False
    for day in range(1, 12):
        panel.has_row[day, 0] = False
    extras = _extras(panel, up_limit=np.full((20, 2), np.nan), down_limit=np.full((20, 2), np.nan))
    params = _params()
    result = p1_bounds.evaluate_intents(
        panel, extras, _benchmark(panel), params, _intents([0, 1], [0, 0]), max_paths=64
    )
    assert result.unbounded[0]
    assert np.isfinite(result.lo[1]) and np.isfinite(result.hi[1])
    envelope = p1_bounds.aggregate_envelope(result)
    assert envelope["bounded"] == 1


def test_structural_sha_error_hard_fails(tmp_path):
    """反例 5：SHA/账本结构错误仍硬失败（verify 检出篡改）。"""
    target = tmp_path / "run"
    target.mkdir()
    (target / "verdict.json").write_text('{"verdict": "continue"}\n', encoding="utf-8")
    manifest = {"written_files_sha256": {"verdict.json": "0" * 64}}
    verify = p1_artifacts.verify_payload(target, manifest)
    assert not verify["all_equal"]
    assert verify["mismatches"]


def test_same_window_residual_order_reversed():
    """反例 6：事件收益排序与残存排序相反时必须被同期净残存覆盖。"""
    panel = _panel(20)
    # 意图 0：事件 10 -> 10.2（+2%），基准成员 10 -> 10.3（+3%）⇒ 残存约 -1%
    panel.open_u[1, 0] = 10.0
    panel.open_u[6, 0] = 10.2
    panel.open_adj[1, 0] = 10.0
    panel.open_adj[6, 0] = 10.2
    panel.open_u[1, 1] = 10.0
    panel.open_u[6, 1] = 10.3
    panel.open_adj[1, 1] = 10.0
    panel.open_adj[6, 1] = 10.3
    params = _params()
    result = p1_bounds.evaluate_intents(
        panel, _extras(panel), _benchmark(panel), params, _intents([0], [0]), max_paths=64
    )
    event_gross = 10.2 / 10.0 - 1.0
    benchmark_gross = 10.3 / 10.0 - 1.0
    assert event_gross > 0 and benchmark_gross > event_gross
    assert result.lo[0] < 0 and result.hi[0] < 0


def test_not_filled_holding_is_na_and_conserved():
    """收口项：未成交不得伪作 CLOSED/OPEN；守恒按 filled_* 拆分。"""
    panel = _panel(20)
    for day in range(1, 12):
        panel.suspended[day, 0] = True
    extras = _extras(panel)
    params = _params()
    result = p1_bounds.evaluate_intents(
        panel, extras, _benchmark(panel), params, _intents([0], [0]), max_paths=64
    )
    assert result.fill[0] == p1_bounds.FILL_PROVEN_NOT
    assert result.holding[0] == p1_bounds.HOLDING_NA
    assert result.stats["filled_na"] == 0
    assert result.stats["holding_na"] == 1
    assert result.stats["conservation_ok"]


def test_legs_reconcile_net_and_gross():
    """收口项：事件净收益必须与毛收益、四项成本一致。"""
    panel = _panel(20)
    panel.open_u[1, 0] = 10.0
    panel.open_u[6, 0] = 10.5
    panel.open_adj[1, 0] = 10.0
    panel.open_adj[6, 0] = 10.5
    params = _params()
    result = p1_bounds.evaluate_intents(
        panel, _extras(panel), _benchmark(panel), params, _intents([0], [0]), max_paths=64
    )
    a = params.cost.commission_rate + params.cost.slippage
    b = params.cost.commission_rate + params.cost.stamp_tax + params.cost.slippage
    expected = (1.0 + result.event_gross[0]) * (1.0 - b) / (1.0 + a) - 1.0
    assert abs(result.event_net[0] - expected) < 1e-12
    assert result.buy_cost[0] > 0 and result.sell_cost[0] > 0
    assert result.bench_members[0] == 1


def test_next_trading_day_helper():
    days = ["20200102", "20200103", "20200106"]
    assert p1_inputs._next_trading_day(days, "20200104") == "20200106"
    assert p1_inputs._next_trading_day(days, "20200103") == "20200103"
    assert p1_inputs._next_trading_day(days, "20200107") is None


def test_open_clock_uses_prev_close_not_today_close():
    """修正批次：开盘估值点不得使用当日收盘（时钟越界）。"""
    panel = _panel(3, n_stocks=1)
    panel.open_adj[1, 0] = np.nan
    panel.close_adj[0, 0] = 10.0
    panel.close_adj[1, 0] = 12.0
    book = BenchmarkBook.build(panel, {day: {1: np.array([0])} for day in range(3)}, mode="fixed")
    price, back = book._price(1, np.array([0]), clock="open")
    assert price[0] == 10.0
    assert back[0] == 1
    legacy_price, legacy_back = book._price(1, np.array([0]), clock="legacy")
    assert legacy_price[0] == 12.0
    assert legacy_back[0] == 0


def test_fill_possible_when_only_unknown_entry():
    """修正批次：只有未知日买入分支时，成交是“可能”而非“确认”。"""
    panel = _panel(20)
    for day in range(1, 12):
        panel.has_row[day, 0] = False
    extras = _extras(panel)
    params = _params()
    result = p1_bounds.evaluate_intents(
        panel, extras, _benchmark(panel), params, _intents([0], [0]), max_paths=64
    )
    assert result.fill[0] == p1_bounds.FILL_POSSIBLE
    assert result.holding[0] == p1_bounds.HOLDING_NA
    assert result.stats["filled_possible"] == 1
    assert result.stats["filled_confirmed"] == 0
    assert result.stats["conservation_ok"]


def test_next_open_shifts_non_trading_day_signal():
    """修正批次：非交易日公告顺延到首个交易日，且不先改日期再机械 T+1。"""
    days = ["20200103", "20200106", "20200107", "20200108", "20200109"]
    panel = _panel(len(days), n_stocks=1)
    panel.days = days
    panel.day_index = {day: index for index, day in enumerate(days)}
    panel.stock_index = {"000001.SZ": 0}
    panel.day_position = lambda day: panel.day_index[day]
    events = {
        "forecast": pd.DataFrame(
            {"ts_code": ["000001.SZ"], "knowledge_date": ["20200104"], "fingerprint": ["f"]}
        ),
        "express": pd.DataFrame(
            {"ts_code": [], "knowledge_date": [], "fingerprint": []}
        ),
        "fina_report": pd.DataFrame(
            {"ts_code": [], "knowledge_date": [], "fingerprint": []}
        ),
    }
    shifted = p1_inputs.build_intents(
        events,
        panel,
        evaluation_start="20200106",
        horizon=1,
        entry_max_defer=1,
        cutoff_pos=4,
        calendar_mode="next_open",
    )
    assert shifted.size == 1
    assert shifted.signal_pos[0] == 1
    assert shifted.knowledge[0] == "20200104"
    legacy = p1_inputs.build_intents(
        events,
        panel,
        evaluation_start="20200106",
        horizon=1,
        entry_max_defer=1,
        cutoff_pos=4,
        calendar_mode="legacy",
    )
    assert legacy.size == 0


def test_zero_denominator_replicate_counted_invalid():
    """反例 8：N*=0 的复制不得静默丢弃，须计入 invalid_rate。"""
    days = np.zeros(10, dtype=np.int64)
    values = np.full(10, 0.01)
    result = p1_bounds._bootstrap_values(
        days,
        values,
        50,
        block=1,
        replicates=200,
        seed=7,
        mode="stationary",
    )
    assert result["invalid"] > 0
    assert result["invalid_rate"] > 0
