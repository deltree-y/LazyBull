# -*- coding: utf-8 -*-
"""P2b-T4 合成端到端等价与真实合并路径验收。

验收依据：``docs/plans/v2/p2b_implementation_plan_20261016.md`` §5-T4、§6 门 4~7、
§7 合成测试矩阵，以及 T2 评审遗留的提前调仓计数义务。全部输入均为合成行情、
信号、特征与成本配置，不读取真实生产配置或数据。

覆盖默认/显式关闭/开启三态逐位一致、真实 merge 调用计数、半 Kelly + 限权 +
分两批、提前调仓接缝前早退/接受/拖尾拒绝回滚、条件卖出补位与 T+1 顺位替换、
买卖延迟订单及同日止损/到期顺序，并对绕过 merge、假计数和输出漂移做反变异。
"""

import copy
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest

from src.lazybull.universe.base import Universe
from src.lazybull.v2.common import cost as new_cost
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.rules.stop_loss import StopLossConfig
from src.lazybull.v2.common.types import TSCode, Weight
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger
from src.lazybull.v2.core.execution.ml_signal_feed import BacktestEngineML
from src.lazybull.v2.core.signal.base import Signal
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML
from src.lazybull.v2.hosts.backtest.replay import _run_replay_engine_and_collect
from src.lazybull.v2.hosts.backtest.runtime import (
    LEGACY_BYPASS,
    SINGLE_SLEEVE_PASSTHROUGH,
    get_merger_call_count,
)

_DATES = list(pd.bdate_range("2026-01-05", periods=25))
_STOCKS = ["A", "B", "C", "D", "E", "F"]
_BASE_PRICES = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0, "E": 50.0, "F": 60.0}
_COST_SETTINGS = {
    "commission_rate": 0.0003,
    "min_commission": 5.0,
    "stamp_tax": 0.0005,
    "slippage": 0.001,
}


@pytest.fixture()
def synthetic_cost_settings(monkeypatch):
    """CostModel 的无条件配置读取统一替换为合成成本。"""
    monkeypatch.setattr(new_cost, "get_cost_settings", lambda: dict(_COST_SETTINGS))


class _FakeStorage:
    """停牌数据恒为空，避免访问真实 Storage。"""

    def load_raw_by_date(self, dataset: str, trade_date: str):
        assert dataset == "suspend"
        return pd.DataFrame({"ts_code": [], "suspend_type": []})


class _StaticUniverse(Universe):
    def __init__(self, stocks: List[str]):
        super().__init__("p2b_t4_static")
        self._stocks = list(stocks)

    def get_stocks(self, date: pd.Timestamp, quote_data: Optional[pd.DataFrame] = None):
        return list(self._stocks)


class _DeterministicSignal(Signal):
    """每个臂独立持有候选计划，按日期返回确定性排名。"""

    def __init__(
        self,
        top_n: int,
        plan: Dict[str, List[Tuple[str, float]]],
        default: List[Tuple[str, float]],
    ):
        super().__init__("p2b_t4_deterministic")
        self.top_n = top_n
        self.plan = copy.deepcopy(plan)
        self.default = copy.deepcopy(default)

    def generate(self, date, universe, data):
        return dict(self.generate_ranked(date, universe, data))

    def generate_ranked(self, date, universe, data):
        ranked = self.plan.get(date.strftime("%Y%m%d"), self.default)
        universe_set = set(universe)
        return [(stock, score) for stock, score in ranked if stock in universe_set]


class _InstrumentedMixin:
    """只增加验收观测，不改变归一化、补位或合并数值路径。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_inputs = []
        self.refill_observations = []

    def _normalize_signals(self, signals, date):
        normalized = super()._normalize_signals(signals, date)
        self.normalize_inputs.append((date, dict(signals), dict(normalized)))
        return normalized

    def _queue_condition_sell_refill_signal(self, date, slot_weights, price_data, date_to_idx):
        before_count = get_merger_call_count(self)
        already_pending = date in self.pending_signals
        try:
            return super()._queue_condition_sell_refill_signal(
                date, slot_weights, price_data, date_to_idx
            )
        finally:
            self.refill_observations.append(
                {
                    "date": date,
                    "generated": not already_pending and date in self.pending_signals,
                    "merge_delta": get_merger_call_count(self) - before_count,
                }
            )


class _InstrumentedEngineML(_InstrumentedMixin, BacktestEngineML):
    pass


class _InstrumentedNativeEngineML(_InstrumentedMixin, NativeMergeBacktestEngineML):
    pass


def _flat_closes(stocks=_STOCKS) -> Dict[str, List[float]]:
    return {
        stock: [_BASE_PRICES[stock] * (1 + 0.001 * index) for index in range(len(_DATES))]
        for stock in stocks
    }


def _make_price_data(
    closes: Dict[str, List[float]],
    *,
    limit_up=frozenset(),
    limit_down=frozenset(),
    suspended=frozenset(),
) -> pd.DataFrame:
    rows = []
    for index, date in enumerate(_DATES):
        date_str = date.strftime("%Y%m%d")
        for stock, series in closes.items():
            close = series[index]
            rows.append(
                {
                    "ts_code": stock,
                    "trade_date": date_str,
                    "close": close,
                    "open": close * 0.999,
                    "close_adj": close,
                    "open_adj": close * 0.999,
                    "is_suspended": int((stock, date_str) in suspended),
                    "is_limit_up": int((stock, date_str) in limit_up),
                    "is_limit_down": int((stock, date_str) in limit_down),
                }
            )
    return pd.DataFrame(rows)


def _make_features() -> Dict[str, pd.DataFrame]:
    return {
        date.strftime("%Y%m%d"): pd.DataFrame(
            {"ts_code": _STOCKS, "atr_pct_14": [0.02, 0.03, 0.025, 0.035, 0.04, 0.045]}
        )
        for date in _DATES
    }


def _make_arm(
    mode: Optional[str],
    price_data: pd.DataFrame,
    *,
    plan: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    default: Optional[List[Tuple[str, float]]] = None,
    top_n: int = 2,
    **overrides,
):
    plan = plan or {}
    default = default if default is not None else [("A", 0.9), ("B", 0.8)]
    signal = _DeterministicSignal(top_n, plan, default)
    features = _make_features()
    kwargs = dict(
        universe=_StaticUniverse(_STOCKS),
        signal=signal,
        features_by_date=features,
        initial_capital=777_777.0,
        rebalance_freq=5,
        verbose=False,
        cost_model=CostModel(**_COST_SETTINGS),
        data_storage=_FakeStorage(),
    )
    kwargs.update(overrides)
    if mode != LEGACY_BYPASS:
        engine = _InstrumentedNativeEngineML(
            **kwargs, portfolio_merger=SingleSleevePortfolioMerger(), sleeve_id="value"
        )
    else:
        engine = _InstrumentedEngineML(**kwargs)
    engine.record_holdings_snapshot = True

    merge_calls = []
    port_returns = []
    account_builds = []
    if mode == SINGLE_SLEEVE_PASSTHROUGH:
        original_build = engine._build_current_account

        def _build_spy(total_value):
            account_builds.append({"total_value": total_value})
            return original_build(total_value)

        engine._build_current_account = _build_spy
        original_merge = engine.portfolio_merger.merge

        def _merge_spy(targets, current):
            target = targets["value"]
            merge_calls.append(
                {
                    "date": pd.Timestamp(target.as_of_date.value),
                    "weights": [
                        (code.value, weight.value) for code, weight in target.target_weights.items()
                    ],
                }
            )
            return original_merge(targets, current)

        engine.portfolio_merger.merge = _merge_spy
        original_port_merge = engine._merge_target_weights

        def _port_spy(normalized, date):
            result = original_port_merge(normalized, date)
            port_returns.append(list(result.items()))
            return result

        engine._merge_target_weights = _port_spy

    return SimpleNamespace(
        engine=engine,
        signal=signal,
        price_data=price_data,
        price_before=price_data.copy(deep=True),
        features_before=copy.deepcopy(features),
        merge_calls=merge_calls,
        port_returns=port_returns,
        account_builds=account_builds,
        nav=None,
    )


def _runtime_snapshot(arm) -> Dict:
    engine = arm.engine
    names = (
        "current_capital",
        "positions",
        "pending_signals",
        "pending_condition_sells",
        "pending_stop_loss_sells",
        "unfilled_slots",
        "_last_ranked_candidates",
        "_last_signal_date",
        "_last_rebalance_nav",
    )
    return {name: copy.deepcopy(getattr(engine, name)) for name in names}


def _assert_features_equal(left: Dict[str, pd.DataFrame], right: Dict[str, pd.DataFrame]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        pd.testing.assert_frame_equal(left[key], right[key], check_exact=True)


def _assert_independent_initial_state(closed, opened) -> None:
    assert closed.engine is not opened.engine
    assert closed.signal is not opened.signal
    assert closed.engine.features_by_date is not opened.engine.features_by_date
    assert closed.price_data is not opened.price_data
    assert _runtime_snapshot(closed) == _runtime_snapshot(opened)
    assert closed.signal.plan == opened.signal.plan
    assert closed.signal.default == opened.signal.default
    _assert_features_equal(closed.engine.features_by_date, opened.engine.features_by_date)
    pd.testing.assert_frame_equal(closed.price_data, opened.price_data, check_exact=True)


def _run_arm(arm, dates: List[pd.Timestamp]):
    arm.nav = arm.engine.run(dates[0], dates[-1], dates, arm.price_data)
    return arm


def _run_collect_arm(arm, dates: List[pd.Timestamp]):
    arm.metrics = _run_replay_engine_and_collect(
        arm.engine,
        dates[0].strftime("%Y%m%d"),
        dates[-1].strftime("%Y%m%d"),
        dates,
        arm.price_data,
        bt_top_n=2,
        split_num=0,
    )
    arm.nav = arm.metrics["_nav_curve"]
    return arm


def _make_pair(price_data: pd.DataFrame, **kwargs):
    closed = _make_arm(LEGACY_BYPASS, price_data.copy(deep=True), **kwargs)
    opened = _make_arm(SINGLE_SLEEVE_PASSTHROUGH, price_data.copy(deep=True), **kwargs)
    _assert_independent_initial_state(closed, opened)
    return closed, opened


def _run_pair(price_data: pd.DataFrame, dates: Optional[List[pd.Timestamp]] = None, **kwargs):
    closed, opened = _make_pair(price_data, **kwargs)
    dates = dates or _DATES[:12]
    _run_arm(closed, dates)
    _run_arm(opened, dates)
    _assert_arm_equivalent(closed, opened)
    return closed, opened


def _assert_inputs_unchanged(arm) -> None:
    pd.testing.assert_frame_equal(arm.price_data, arm.price_before, check_exact=True)
    _assert_features_equal(arm.engine.features_by_date, arm.features_before)


def _assert_arm_equivalent(closed, opened) -> None:
    """runs 对应的合成全产物与关键终态精确一致。"""
    pd.testing.assert_frame_equal(closed.nav, opened.nav, check_exact=True)
    pd.testing.assert_frame_equal(
        closed.engine.get_trades(), opened.engine.get_trades(), check_exact=True
    )
    pd.testing.assert_frame_equal(
        closed.engine.get_execution_attribution(),
        opened.engine.get_execution_attribution(),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        closed.engine.get_holdings_snapshot(),
        opened.engine.get_holdings_snapshot(),
        check_exact=True,
    )
    assert closed.engine.positions == opened.engine.positions
    assert closed.engine.pending_signals == opened.engine.pending_signals
    assert closed.engine.pending_condition_sells == opened.engine.pending_condition_sells
    assert closed.engine.pending_stop_loss_sells == opened.engine.pending_stop_loss_sells
    assert closed.engine.completion_stats == opened.engine.completion_stats
    assert (
        closed.engine.pending_order_manager.get_statistics()
        == opened.engine.pending_order_manager.get_statistics()
    )
    assert closed.engine.normalize_inputs == opened.engine.normalize_inputs
    assert closed.engine.refill_observations == opened.engine.refill_observations
    assert get_merger_call_count(closed.engine) == 0
    assert closed.merge_calls == [] and closed.port_returns == []
    assert closed.account_builds == []
    _assert_inputs_unchanged(closed)
    _assert_inputs_unchanged(opened)


def _assert_real_merge_path(closed, opened) -> None:
    """逐日核对真实调用次数，以及归一化→merge→端口返回的权重无损保序。"""
    expected = closed.engine.normalize_inputs
    expected_dates = [item[0] for item in expected]
    assert expected_dates, "前置锚定：场景必须至少生成一个当批目标"
    assert [item[0] for item in opened.engine.normalize_inputs] == expected_dates
    assert [call["date"] for call in opened.merge_calls] == expected_dates
    assert len(opened.port_returns) == len(expected_dates)
    assert len(opened.account_builds) == len(expected_dates)
    assert get_merger_call_count(opened.engine) == len(expected_dates)
    assert opened.engine.portfolio_merger.merge_call_count == len(expected_dates)
    for (_date, _raw, normalized), call, returned in zip(
        opened.engine.normalize_inputs, opened.merge_calls, opened.port_returns
    ):
        assert list(normalized.items()) == call["weights"]
        assert returned == call["weights"]


def _prepare_manual(arm, price_data: pd.DataFrame) -> None:
    arm.engine._prepare_price_index(price_data)
    arm.engine.price_data_cache = price_data
    arm.engine._trade_date_index = {date: index for index, date in enumerate(_DATES)}
    arm.engine._cycle_anchor_idx = 0
    arm.engine._deferred_day_logs = []
    arm.engine._reset_daily_warning_items()


def _drive_early(arm, idx: int, cycle_day: int, signal_dates: Dict):
    date = _DATES[idx]
    result = arm.engine._try_early_rebalance(
        idx,
        date,
        cycle_day,
        signal_dates,
        _DATES,
        arm.price_data,
        {date: index for index, date in enumerate(_DATES)},
        "separator",
    )
    return result


def test_default_closed_and_enabled_three_modes_exact(synthetic_cost_settings):
    """缺省 native 与显式 native 均真实合并，显式 legacy 为零合并且三臂产物一致。"""
    price_data = _make_price_data(_flat_closes())
    arms = [
        _run_arm(
            _make_arm(
                mode,
                price_data.copy(deep=True),
                rebalance_freq=20,
                enable_early_rebalance_on_empty=False,
            ),
            _DATES[:4],
        )
        for mode in (None, LEGACY_BYPASS, SINGLE_SLEEVE_PASSTHROUGH)
    ]
    default, closed, opened = arms
    assert default.engine.portfolio_merger.merge_call_count > 0
    assert "portfolio_merger" not in vars(closed.engine)
    assert "merge_port" not in vars(default.engine) and "merge_port" not in vars(closed.engine)
    _assert_arm_equivalent(closed, default)
    _assert_arm_equivalent(closed, opened)
    _assert_real_merge_path(closed, opened)


def test_native_close_replay_returns_and_matches_legacy(synthetic_cost_settings):
    """close 是公开合法配置：证据投影跳过，不得让正常原生回测失败或改值。"""
    price_data = _make_price_data(_flat_closes())
    closed, opened = _make_pair(
        price_data,
        rebalance_freq=3,
        sell_timing="close",
        enable_early_rebalance_on_empty=False,
    )
    for arm in (closed, opened):
        _run_collect_arm(arm, _DATES[:10])
    _assert_arm_equivalent(closed, opened)
    _assert_real_merge_path(closed, opened)
    assert len(opened.engine.trades) > 0
    assert opened.metrics["_merger_s6_role"] == "backtest_t3_evidence_skipped_sell_timing_close"
    assert opened.metrics["_merger_net_call_count"] == 0
    assert opened.metrics["_native_evidence_order_count"] == 0


def test_equal_regular_rebalance_and_condition_refill(synthetic_cost_settings):
    """equal 常规目标真实合并一次；到期补位是局部计划，merge 增量为零。"""
    closed, opened = _run_pair(
        _make_price_data(_flat_closes()),
        dates=_DATES[:9],
        default=[("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)],
        rebalance_freq=20,
        holding_period=5,
        enable_early_rebalance_on_empty=False,
    )
    _assert_real_merge_path(closed, opened)
    assert len(opened.merge_calls) == 1
    trades = opened.engine.get_trades()
    for arm in (closed, opened):
        snapshot = arm.engine.get_holdings_snapshot()
        assert not snapshot.empty, "合成持仓快照必须有非空内容，禁止空表等价空转"
        assert (snapshot["shares"] > 0).any() and (snapshot["market_value"] > 0).any()
    assert "holding_period" in set(trades["sell_type"].dropna())
    generated = [item for item in opened.engine.refill_observations if item["generated"]]
    assert generated, "前置锚定：必须实际生成持有期卖出补位计划"
    assert all(item["merge_delta"] == 0 for item in opened.engine.refill_observations)
    assert {"C", "D"} <= set(trades.loc[trades["action"] == "buy", "stock"])


def test_half_kelly_max_weight_and_two_tranches(synthetic_cost_settings):
    """半 Kelly + 限权 + 两批：merge 读取归一化后、批次限权前的当批相对权重。"""
    closes = _flat_closes()
    closes["A"] = [10.0 * (1 + 0.002 * index) for index in range(len(_DATES))]
    closes["B"] = [20.0 * (1 + (0.004 if index % 2 else -0.003)) for index in range(len(_DATES))]
    closed, opened = _run_pair(
        _make_price_data(closes),
        dates=_DATES[:10],
        default=[("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6), ("E", 0.5), ("F", 0.4)],
        top_n=3,
        rebalance_freq=4,
        stagger_tranches=2,
        position_sizing="half_kelly",
        kelly_vol_window=3,
        max_weight_per_stock=0.34,
        enable_early_rebalance_on_empty=False,
    )
    _assert_real_merge_path(closed, opened)
    schedule = closed.engine._get_rebalance_dates(_DATES[:10])
    tranches = {schedule[call["date"]] for call in opened.merge_calls}
    assert tranches == {0, 1}
    for call in opened.merge_calls:
        weights = dict(call["weights"])
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
        assert sum(weights.values()) != pytest.approx(0.5)
    # 两批资金比例为 0.5 时，0.34 的全局上限对应批内 cap=0.68；
    # 单目标批次的 merge 前权重为 1.0，证明限权尚未提前施加。
    assert any(max(dict(call["weights"]).values()) > 0.68 for call in opened.merge_calls)


@pytest.mark.parametrize("early_case", ["no_candidates", "no_t1"])
def test_early_rebalance_returns_before_seam(synthetic_cost_settings, early_case):
    """无候选或无 T+1：接缝前返回，不构造 Account、不调用 merge。"""
    default = [] if early_case == "no_candidates" else [("A", 0.9)]
    idx = 1 if early_case == "no_candidates" else len(_DATES) - 1
    price_data = _make_price_data(_flat_closes())
    closed, opened = _make_pair(price_data, default=default, rebalance_freq=20)
    for arm in (closed, opened):
        _prepare_manual(arm, arm.price_data)
        _drive_early(arm, idx, 2, {})
    assert closed.engine.pending_signals == opened.engine.pending_signals == {}
    assert closed.engine.normalize_inputs == opened.engine.normalize_inputs == []
    assert opened.merge_calls == [] and opened.account_builds == []
    assert get_merger_call_count(opened.engine) == 0


def test_early_rebalance_empty_position_accepted_counts_once(synthetic_cost_settings):
    """空仓提前调仓接受：真实 merge=1，双臂 pending 与节奏清理一致。"""
    price_data = _make_price_data(_flat_closes())
    closed, opened = _make_pair(price_data, rebalance_freq=20)
    results = []
    for arm in (closed, opened):
        _prepare_manual(arm, arm.price_data)
        signal_dates = {_DATES[3]: 0}
        results.append(_drive_early(arm, 1, 2, signal_dates))
        assert signal_dates == {}
        assert _DATES[1] in arm.engine.pending_signals
    assert results == [1, 1]
    _assert_real_merge_path(closed, opened)
    assert len(opened.merge_calls) == 1
    assert closed.engine.pending_signals == opened.engine.pending_signals


def test_early_rebalance_trailing_rejection_rolls_back_but_not_count(synthetic_cost_settings):
    """拖尾拒绝：生成后删除 pending 并回滚历史，但真实 merge 计数保持 1。"""
    price_data = _make_price_data(_flat_closes())
    closed, opened = _make_pair(
        price_data,
        default=[("B", 0.9), ("C", 0.8)],
        rebalance_freq=5,
        holding_period=5,
    )
    for arm in (closed, opened):
        _prepare_manual(arm, arm.price_data)
        arm.engine.current_capital = 100_000.0
        arm.engine.positions["A"] = {
            "shares": 10_000,
            "buy_date": _DATES[0],
            "signal_date": _DATES[0],
            "buy_trade_price": 10.0,
            "buy_pnl_price": 10.0,
            "buy_cost_cash": 100_000.0,
        }
        arm.engine._last_ranked_candidates = [("SENTINEL", 1.0)]
        arm.engine._last_signal_date = pd.Timestamp("1999-01-01")
        arm.engine._last_rebalance_nav = 123_456.0
        _drive_early(arm, 1, 5, {})
        assert _DATES[1] not in arm.engine.pending_signals
        assert arm.engine._last_ranked_candidates == [("SENTINEL", 1.0)]
        assert arm.engine._last_signal_date == pd.Timestamp("1999-01-01")
        assert arm.engine._last_rebalance_nav == 123_456.0
        labels = [item["label"] for item in arm.engine._daily_warning_items["early_rebalance"]]
        assert "拖尾拒绝" in labels
    _assert_real_merge_path(closed, opened)
    assert len(opened.merge_calls) == 1
    assert closed.engine.positions == opened.engine.positions


def test_t1_limit_up_candidate_replacement_adds_no_merge(synthetic_cost_settings):
    """T+1 涨停槽位按候选顺位替换，属于执行 continuation，不再次合并。"""
    limit_up = frozenset({("B", _DATES[1].strftime("%Y%m%d"))})
    closed, opened = _run_pair(
        _make_price_data(_flat_closes(), limit_up=limit_up),
        dates=_DATES[:3],
        default=[("A", 0.9), ("B", 0.8), ("C", 0.7)],
        rebalance_freq=20,
        enable_early_rebalance_on_empty=False,
    )
    _assert_real_merge_path(closed, opened)
    assert len(opened.merge_calls) == 1
    attribution = opened.engine.get_execution_attribution()
    replaced = attribution[attribution["planned_stock"] != attribution["actual_stock"]]
    assert len(replaced) == 1
    assert replaced.iloc[0]["planned_stock"] == "B"
    assert replaced.iloc[0]["actual_stock"] == "C"


def test_pending_buy_retry_success_and_two_expiry_paths(synthetic_cost_settings):
    """买入涨停入队后重试成功，超次/超期分别放弃；开启态不改变执行语义。"""
    d0, d1, d2, d3 = _DATES[:4]
    closes = {stock: [10.0] * len(_DATES) for stock in ("X", "Y", "Z")}
    limit_up = frozenset(
        {("Y", d0.strftime("%Y%m%d"))} | {("X", date.strftime("%Y%m%d")) for date in _DATES[:5]}
    )
    price_data = _make_price_data(closes, limit_up=limit_up)
    closed, opened = _make_pair(price_data, max_retry_count=2, max_retry_days=10)

    def _drive(arm):
        engine = arm.engine
        _prepare_manual(arm, arm.price_data)
        engine._buy_stock_with_status_check(d0, "X", 50_000.0, signal_date=d0)
        engine._buy_stock_with_status_check(d0, "Y", 50_000.0, signal_date=d0)
        engine._buy_stock_with_status_check(d0, "Z", 50_000.0, signal_date=d0)
        engine._process_pending_orders(d1)
        engine._process_pending_orders(d2)
        engine._process_pending_orders(d3)
        engine._buy_stock_with_status_check(d0, "W", 1_000.0, signal_date=d0)
        engine._process_pending_orders(d0 + pd.Timedelta(days=20))

    _drive(closed)
    _drive(opened)
    pd.testing.assert_frame_equal(
        closed.engine.get_trades(), opened.engine.get_trades(), check_exact=True
    )
    assert [trade["stock"] for trade in opened.engine.trades] == ["Z", "Y"]
    closed_stats = closed.engine.pending_order_manager.get_statistics()
    opened_stats = opened.engine.pending_order_manager.get_statistics()
    assert closed_stats == opened_stats
    assert opened_stats["total_succeeded"] == 1 and opened_stats["total_expired"] == 2
    expired = opened.engine._daily_warning_items["pending_order_expired"]
    by_stock = {item["stock"]: item for item in expired}
    assert len(expired) == 2 and set(by_stock) == {"X", "W"}
    assert by_stock["X"]["expire_type"] == "expired_retry"
    assert by_stock["X"]["retry_count"] == 3 and by_stock["X"]["max_retry_count"] == 2
    assert by_stock["W"]["expire_type"] == "expired_days"
    assert by_stock["W"]["delay_days"] == 20 and by_stock["W"]["max_retry_days"] == 10
    assert closed.engine._daily_warning_items == opened.engine._daily_warning_items
    assert opened.engine.normalize_inputs == []
    assert opened.merge_calls == [] and opened.account_builds == []
    assert get_merger_call_count(opened.engine) == 0


def test_sell_limit_down_and_suspension_retry_without_expiry(synthetic_cost_settings):
    """卖出跌停/停牌持续重试且不过期，恢复后成交；补位路径不增加 merge。"""
    limit_down = frozenset({("A", _DATES[6].strftime("%Y%m%d"))})
    suspended = frozenset({("A", _DATES[7].strftime("%Y%m%d"))})
    closed, opened = _run_pair(
        _make_price_data(_flat_closes(), limit_down=limit_down, suspended=suspended),
        dates=_DATES[:10],
        default=[("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)],
        rebalance_freq=20,
        holding_period=5,
        max_retry_count=0,
        max_retry_days=0,
        enable_early_rebalance_on_empty=False,
    )
    _assert_real_merge_path(closed, opened)
    stats = opened.engine.pending_order_manager.get_statistics()
    assert stats["total_succeeded"] >= 1
    assert stats["total_expired"] == 0 and stats["pending"] == 0
    sells_a = opened.engine.get_trades().query("action == 'sell' and stock == 'A'")
    assert len(sells_a) == 1 and sells_a.iloc[0]["date"] == _DATES[8]
    assert all(item["merge_delta"] == 0 for item in opened.engine.refill_observations)


def test_same_day_stop_loss_and_expiry_order_exact(synthetic_cost_settings):
    """同日止损与持有期到期：双臂均为 A/stop_loss 先于 B/holding_period。"""
    closes = _flat_closes()
    closes["A"] = [
        10.0 * (1 + 0.001 * index) if index < 5 else 9.0 * (1 + 0.0005 * (index - 5))
        for index in range(len(_DATES))
    ]
    closed, opened = _run_pair(
        _make_price_data(closes),
        dates=_DATES[:9],
        rebalance_freq=20,
        holding_period=5,
        stop_loss_config=StopLossConfig(enabled=True, drawdown_pct=5.0),
        enable_early_rebalance_on_empty=False,
    )
    _assert_real_merge_path(closed, opened)
    sells = opened.engine.get_trades().query("action == 'sell'").head(2)
    assert list(sells["stock"]) == ["A", "B"]
    assert list(sells["sell_type"]) == ["stop_loss", "holding_period"]
    assert sells.iloc[0]["date"] == sells.iloc[1]["date"] == _DATES[6]


def test_anti_mutation_bypass_and_false_count_detected(synthetic_cost_settings):
    """绕过 merge 或伪造计数时，真实调用证据必须失败，即使数值产物相同。"""
    price_data = _make_price_data(_flat_closes())
    closed, opened = _run_pair(
        price_data,
        dates=_DATES[:3],
        rebalance_freq=20,
        enable_early_rebalance_on_empty=False,
    )
    _assert_real_merge_path(closed, opened)
    opened.engine.portfolio_merger.merge_call_count += 1
    with pytest.raises(AssertionError):
        _assert_real_merge_path(closed, opened)

    bypass = _make_arm(
        SINGLE_SLEEVE_PASSTHROUGH,
        price_data.copy(deep=True),
        rebalance_freq=20,
        enable_early_rebalance_on_empty=False,
    )
    bypass.engine._merge_target_weights = lambda normalized, date: dict(normalized)
    _run_arm(bypass, _DATES[:3])
    pd.testing.assert_frame_equal(closed.nav, bypass.nav, check_exact=True)
    with pytest.raises(AssertionError):
        _assert_real_merge_path(closed, bypass)


def test_anti_mutation_output_drift_detected(synthetic_cost_settings):
    """任一 runs 数值产物漂移都必须被精确比较捕获。"""
    closed, opened = _run_pair(
        _make_price_data(_flat_closes()),
        dates=_DATES[:3],
        rebalance_freq=20,
        enable_early_rebalance_on_empty=False,
    )
    changed = copy.copy(opened)
    changed.nav = opened.nav.copy(deep=True)
    changed.nav.loc[changed.nav.index[0], "nav"] += 1e-12
    with pytest.raises(AssertionError):
        _assert_arm_equivalent(closed, changed)

    drifted_snapshot = opened.engine.get_holdings_snapshot().copy(deep=True)
    assert not drifted_snapshot.empty, "快照漂移反例必须基于非空快照"
    drifted_snapshot.loc[drifted_snapshot.index[0], "market_value"] += 1.0
    snapshot_changed = copy.copy(opened)
    snapshot_changed.engine = copy.copy(opened.engine)
    snapshot_changed.engine.get_holdings_snapshot = lambda: drifted_snapshot
    with pytest.raises(AssertionError):
        _assert_arm_equivalent(closed, snapshot_changed)


@pytest.mark.parametrize("mutation", ["reorder", "renormalize"])
def test_anti_mutation_weight_reorder_and_renormalize_detected(synthetic_cost_settings, mutation):
    """merge 内重排或额外归一化必须被权重保真断言捕获。

    非精确权重和仅用于让额外归一化变异产生可见差异，不表示生产合法输入。
    """
    price_data = _make_price_data(_flat_closes())
    closed = _make_arm(LEGACY_BYPASS, price_data.copy(deep=True))
    opened = _make_arm(SINGLE_SLEEVE_PASSTHROUGH, price_data.copy(deep=True))
    original_merge = opened.engine.portfolio_merger.merge

    def _mutating_merge(targets, current):
        target = original_merge(targets, current)
        items = [(code.value, weight.value) for code, weight in target.target_weights.items()]
        if mutation == "reorder":
            items = list(reversed(items))
        else:
            total = sum(weight for _code, weight in items)
            items = [(code, weight / total) for code, weight in items]
        return replace(
            target,
            target_weights=MappingProxyType(
                {TSCode(code): Weight(weight) for code, weight in items}
            ),
        )

    opened.engine.portfolio_merger.merge = _mutating_merge
    weights = {"A": 0.9, "B": 0.4}
    date = _DATES[0]
    for arm in (closed, opened):
        arm.engine.normalize_inputs.append((date, dict(weights), dict(weights)))
    opened.engine.current_capital = 100_000.0
    opened.engine._last_rebalance_nav = 200_000.0
    result = opened.engine._merge_target_weights(weights, date)
    if mutation == "reorder":
        assert list(result) != list(weights), "重排变异未实际生效"
    else:
        assert result != weights, "额外归一化变异未实际生效"
    with pytest.raises(AssertionError):
        _assert_real_merge_path(closed, opened)
