# -*- coding: utf-8 -*-
"""v2 P4 T2「信号/买入计划装配映射」黄金行为锁定（tests/ 全量豁免复杂度）。

任务卡映射小节的六项既有行为，用合成引擎（BacktestEngineML + 合成信号/行情 +
内存桩 Storage）逐项构造场景断言；引擎全内存语义、零真实 data/ 读取、零落盘
（隔离硬契约天然满足；落盘投影侧由 test_v2_p4_t2_state_store / host 覆盖）。

锁定项与实现落点（文件:行号，2026-10-27 实测）：
1. 一手买不起顺延/缩量——``buy_execution.py:973-995``：``compute_lot_shares``
   整手向下取整，0 手 ⇒ 跳过（:975）；总额超可用现金 ⇒ 按可用现金降手（:990）。
2. 已持仓排除——计划侧 ``signal_execution.py:137-141``（候选剔除已持仓）；
   执行侧 ``buy_execution.py:346-347``（槽位匹配 ``__held__`` 跳过 + 同日去重
   ``buy_plan.py:63-66``）；兜底 ``buy_execution.py:953-958``（重复建仓跳过）。
3. 行业上限——``signal_execution.py:118-129`` 调用
   ``decision/industry_constraint.py:142-159``（超限候选跳过）。
4. 补位上限 5 次——``pending_order.py:173``（``retry_count > max_retry_count``=5
   超次放弃，引擎装配 ``config_adapter.to_engine_kwargs`` 写死 5）；T2 指令映射侧
   ``state_store.py:470``（retry > DEFAULT_MAX_REPLACEMENT_ATTEMPTS=5 不再生成
   补位指令）。
5. 候选域 ×2——``buy_execution.py:535-549``（``candidate_buffer = 未成交槽位 × 2``
   at :542，排除已持仓、保序截断）。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import pytest

from scripts.v2_p4.t05b_spike_runner import (
    _COST_SETTINGS,
    _DeterministicSignal,
    _FakeStorage,
    _StaticUniverse,
    _make_price_data,
)
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.core.decision.sizing import compute_lot_shares
from src.lazybull.v2.core.execution.buy_execution import _build_completion_stocks_to_try
from src.lazybull.v2.core.execution.ml_signal_feed import BacktestEngineML
from src.lazybull.v2.hosts.paper.state_store import (
    DEFAULT_MAX_REPLACEMENT_ATTEMPTS,
    build_instruction_frame,
)
from src.lazybull.v2.core.execution.pending_order import PendingOrder

#: 8 个合成交易日（2026-01-05 周一起；覆盖 T0/T+1/补齐/重试窗口）
_DAYS: List[pd.Timestamp] = list(pd.bdate_range("2026-01-05", periods=8))


def _features_by_date(days: List[pd.Timestamp], codes: List[str]) -> Dict[str, pd.DataFrame]:
    """ML 引擎特征桩：每日非空截面（桩信号不消费内容，仅要求 _build_signal_data 非 None）。"""
    return {day.strftime("%Y%m%d"): pd.DataFrame({"ts_code": list(codes)}) for day in days}


def _make_ml_engine(
    stocks: List[str],
    signal,
    days: List[pd.Timestamp],
    **overrides,
) -> BacktestEngineML:
    """合成 BacktestEngineML（成本/停牌日历全内存桩；行为参数同 T2 装配模板）。"""
    kwargs = dict(
        features_by_date=_features_by_date(days, stocks),
        universe=_StaticUniverse(stocks),
        signal=signal,
        initial_capital=1_000_000.0,
        rebalance_freq=5,
        verbose=False,
        cost_model=CostModel(**_COST_SETTINGS),
        data_storage=_FakeStorage(),
    )
    kwargs.update(overrides)
    return BacktestEngineML(**kwargs)


def _wire_and_drive(
    engine: BacktestEngineML,
    days: List[pd.Timestamp],
    price_data: pd.DataFrame,
    upto: int,
) -> None:
    """回放形态接线（镜像 run() 初始化段）+ 逐日 run_step 前 upto 日。"""
    engine._trade_date_index = {d: i for i, d in enumerate(days)}
    engine._prepare_price_index(price_data)
    engine.price_data_cache = price_data
    engine._signal_dates = engine._get_rebalance_dates(days)
    engine._cycle_anchor_idx = 0
    engine._run_total_days = len(days)
    for day in days[:upto]:
        engine.run_step(day, days, price_data)


def _day_fills(engine: BacktestEngineML, day: pd.Timestamp) -> List[Dict]:
    """当日成交行（入账序）。"""
    return [t for t in engine.trades if t["date"] == day]


# ---------------------------------------------------------------------------
# 1. 一手买不起顺延 / 缩量（buy_execution.py:973-995）
# ---------------------------------------------------------------------------


class TestLotSizingPostpone:
    def test_unaffordable_single_lot_skips_buy(self):
        """现金不足一手（槽位预算 5000 < 一手 100×100=10000）⇒ 当日零成交、不建仓。"""
        stocks = ["A", "B"]
        plan = [("A", 0.9), ("B", 0.8)]
        signal = _DeterministicSignal(2, {}, plan)
        engine = _make_ml_engine(stocks, signal, _DAYS, initial_capital=10_000.0)
        price_data = _make_price_data(_DAYS, {s: [100.0] * len(_DAYS) for s in stocks})
        # 驱动 3 日（T0 + T+1 + 首个补齐尝试日）：任何路径都买不起一手
        _wire_and_drive(engine, _DAYS, price_data, upto=3)
        assert engine.trades == [], "买不起一手必须零成交（顺延而非碎股）"
        assert engine.positions == {}
        assert engine.current_capital == 10_000.0

    def test_insufficient_cash_shrinks_to_affordable_lots(self):
        """满仓序列下末笔按可用现金缩量（计划 200 股 ⇒ 成交 100 股），首笔不受影响。"""
        stocks = ["A", "B"]
        plan = [("A", 0.9), ("B", 0.8)]
        signal = _DeterministicSignal(2, {}, plan)
        # 40040 = 两槽位各计划 200 股（20_020/100 取整 200）；首笔成交 20_026 后
        # 余 20_014 < 第二笔 20_026 ⇒ 触发 buy_execution.py:983 资金不足分支降手
        engine = _make_ml_engine(stocks, signal, _DAYS, initial_capital=40_040.0)
        price_data = _make_price_data(_DAYS, {s: [100.0] * len(_DAYS) for s in stocks})
        _wire_and_drive(engine, _DAYS, price_data, upto=2)
        day1_fills = _day_fills(engine, _DAYS[1])
        assert [(t["stock"], t["shares"]) for t in day1_fills] == [("A", 200), ("B", 100)]
        # 缩量方向与整手性：B 成交 < 其槽位计划（200 股）且为整手
        assert compute_lot_shares(40_040.0 / 2, 100.0) == 200, "前提：槽位计划 200 股"
        assert day1_fills[1]["shares"] % 100 == 0
        assert engine.positions["A"]["shares"] == 200
        assert engine.positions["B"]["shares"] == 100


# ---------------------------------------------------------------------------
# 2. 已持仓排除（signal_execution.py:137-141 / buy_execution.py:346-347,953-958）
# ---------------------------------------------------------------------------


class TestHeldPositionExclusion:
    def _driven_engine_with_ab(self):
        """驱动 day0（T0）+ day1（T+1 买入 A/B）后的引擎（持仓 {A, B}）。"""
        stocks = ["A", "B", "C", "D"]
        plan = [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)]
        signal = _DeterministicSignal(2, {}, plan)
        engine = _make_ml_engine(stocks, signal, _DAYS)
        price_data = _make_price_data(_DAYS, {s: [10.0] * len(_DAYS) for s in stocks})
        _wire_and_drive(engine, _DAYS, price_data, upto=2)
        assert set(engine.positions) == {"A", "B"}
        return engine, price_data

    def _wired_engine_with_ab_seeded(self):
        """接线（不驱动）+ 直接播种持仓 {A, B}（保现金充裕，隔离执行侧断言与满仓缩量噪声）。

        持仓 dict 键集对齐快照字段注册表口径（shares/buy_date/signal_date/
        buy_trade_price/buy_pnl_price/buy_cost_cash）。
        """
        stocks = ["A", "B", "C", "D"]
        plan = [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)]
        signal = _DeterministicSignal(2, {}, plan)
        engine = _make_ml_engine(stocks, signal, _DAYS)
        price_data = _make_price_data(_DAYS, {s: [10.0] * len(_DAYS) for s in stocks})
        _wire_and_drive(engine, _DAYS, price_data, upto=0)
        for stock in ("A", "B"):
            engine.positions[stock] = {
                "shares": 1000,
                "buy_date": _DAYS[0],
                "signal_date": _DAYS[0],
                "buy_trade_price": 10.0,
                "buy_pnl_price": 10.0,
                "buy_cost_cash": 10_013.0,
            }
            engine.current_capital -= 10_013.0
        return engine, price_data

    def test_planning_side_excludes_held(self):
        """计划侧：已持仓 A/B 不进新买入计划（候选剔除 ⇒ 次日计划 = {C, D}）。"""
        engine, price_data = self._driven_engine_with_ab()
        day2 = _DAYS[2]
        # 直接调用生产 _generate_signal（freq=5 下 day2 非排期日，白盒直调锁定计划侧行为）
        engine._generate_signal(day2, _DAYS, price_data, engine._trade_date_index)
        plan_entry = engine.pending_signals[day2]
        assert set(plan_entry["signals"]) == {"C", "D"}, "已持仓 A/B 必须被排除出新计划"
        assert {stock for stock, _s in plan_entry["priority_candidates"]} == {"C", "D"}

    def test_execution_side_skips_held_and_dedups_same_day(self):
        """执行侧：槽位匹配跳过已持仓候选（__held__）+ 同日已买候选去重。"""
        engine, price_data = self._wired_engine_with_ab_seeded()
        day1, day2 = _DAYS[1], _DAYS[2]
        a_shares_before = engine.positions["A"]["shares"]
        # 注入「day1 信号日」计划（T+1=day2 执行）：槽位 [A, C]，候选 [A, C]
        engine.pending_signals[day1] = {
            "signals": {"A": 0.5, "C": 0.5},
            "ranked_candidates": [("A", 0.9), ("C", 0.85)],
            "priority_candidates": [("A", 0.9), ("C", 0.85)],
            "slot_weights": [
                {"stock": "A", "weight": 0.5},
                {"stock": "C", "weight": 0.5},
            ],
            "target_n": 2,
            "desired_position_count": 4,
            "tranche_idx": 0,
        }
        engine.run_step(day2, _DAYS, price_data)
        day2_fills = _day_fills(engine, day2)
        # 槽位 A：候选 A 已持仓（__held__）⇒ 顺位 C 成交；槽位 C：A 已持仓、
        # C 当日已买（REASON_ALREADY_BOUGHT）⇒ 未成交
        assert [(t["stock"], t["action"]) for t in day2_fills] == [("C", "buy")]
        assert engine.positions["A"]["shares"] == a_shares_before, "已持仓 A 不得重复买入"
        assert set(engine.positions) == {"A", "B", "C"}

    def test_duplicate_buy_direct_guard(self):
        """兜底：_buy_stock_direct 对已持仓股票直接返回（buy_execution.py:953）。"""
        engine, _price_data = self._wired_engine_with_ab_seeded()
        trades_before = len(engine.trades)
        a_shares_before = engine.positions["A"]["shares"]
        engine._buy_stock_direct(_DAYS[2], "A", 50_000.0, signal_date=_DAYS[1])
        assert len(engine.trades) == trades_before, "已持仓股票重复建仓必须被跳过"
        assert engine.positions["A"]["shares"] == a_shares_before


# ---------------------------------------------------------------------------
# 3. 行业上限（signal_execution.py:118-129 + industry_constraint.py:142-159）
# ---------------------------------------------------------------------------


class TestIndustryCap:
    def test_max_per_industry_filters_candidates(self):
        """X 行业 3 候选（A/B/C）上限 2 ⇒ C 被剔除；计划与成交 = {A, B, D, E}。"""
        stocks = ["A", "B", "C", "D", "E"]
        industries = {"A": "X", "B": "X", "C": "X", "D": "Y", "E": "Y"}
        ranked = [(s, 0.9 - 0.05 * i) for i, s in enumerate(stocks)]
        signal = _DeterministicSignal(4, {}, ranked)
        _FakeStorage.shenwan_table = pd.DataFrame(
            {"ts_code": stocks, "sw_l1": [industries[s] for s in stocks]}
        )
        try:
            engine = _make_ml_engine(
                stocks,
                signal,
                _DAYS,
                max_per_industry=2,
                stock_basic=pd.DataFrame(
                    {"ts_code": stocks, "industry": [industries[s] for s in stocks]}
                ),
                industry_reader=lambda: _FakeStorage.shenwan_table,
            )
            price_data = _make_price_data(_DAYS, {s: [10.0] * len(_DAYS) for s in stocks})
            _wire_and_drive(engine, _DAYS, price_data, upto=1)  # 只驱动 T0（计划未消费）
            plan_entry = engine.pending_signals.get(_DAYS[0])
            # 行业约束在候选层已剔除 C（X 满 2）；计划 = 约束后候选前 4
            assert plan_entry is not None, "T0 计划必须已入队"
            assert [s for s, _ in plan_entry["ranked_candidates"]] == ["A", "B", "D", "E"]
            assert set(plan_entry["signals"]) == {"A", "B", "D", "E"}
            engine.run_step(_DAYS[1], _DAYS, price_data)  # T+1 执行
        finally:
            _FakeStorage.shenwan_table = None
        day1_fills = _day_fills(engine, _DAYS[1])
        assert {t["stock"] for t in day1_fills} == {"A", "B", "D", "E"}
        assert "C" not in engine.positions, "行业上限外候选 C 不得成交"


# ---------------------------------------------------------------------------
# 4. 补位上限 5 次（pending_order.py:173 / state_store.py:470）
# ---------------------------------------------------------------------------


class TestReplacementCap:
    def test_pending_buy_expires_after_max_retry_count(self):
        """延迟买入订单连续涨停重试：retry_count 超 5（max_retry_count）⇒ 放弃。"""
        stocks = ["A", "B", "X"]
        signal = _DeterministicSignal(2, {}, [("A", 0.9), ("B", 0.8)])  # X 永不入选
        engine = _make_ml_engine(stocks, signal, _DAYS, max_retry_count=5, max_retry_days=10)
        limit_up = frozenset(("X", d.strftime("%Y%m%d")) for d in _DAYS)
        price_data = _make_price_data(
            _DAYS, {s: [10.0] * len(_DAYS) for s in stocks}, limit_up=limit_up
        )
        # 先驱动 day0（排期信号日），再补注 X 延迟买入订单（模拟其 T+1 首试失败落队）
        _wire_and_drive(engine, _DAYS, price_data, upto=1)
        engine.pending_order_manager.add_order(
            stock="X",
            action="buy",
            current_date=_DAYS[0],
            signal_date=_DAYS[0],
            target_value=50_000.0,
            reason="涨停",
        )
        # 逐日重试轨迹：day1..day5 重试后 retry_count = 2..6；day6 超次放弃
        expected_retry_after_day = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
        for i in range(1, 7):
            engine.run_step(_DAYS[i], _DAYS, price_data)
            order = engine.pending_order_manager.pending_orders.get(("X", "buy"))
            if i in expected_retry_after_day:
                assert (
                    order is not None and order.retry_count == expected_retry_after_day[i]
                ), f"day{i} 后订单应在队且 retry_count={expected_retry_after_day[i]}"
            else:
                assert order is None, "retry_count=6 > max_retry_count=5 ⇒ 必须放弃"
        manager = engine.pending_order_manager
        assert manager.total_expired == 1 and manager.total_succeeded == 0
        assert "X" not in engine.positions
        assert all(t["stock"] != "X" for t in engine.trades)
        # 放弃形态 = 超次（expired_retry）而非超期（expired_days）：day6-day0 = 8 自然日 ≤ 10
        expired_events = engine._daily_warning_items.get("pending_order_expired", [])
        assert expired_events and expired_events[0]["expire_type"] == "expired_retry"
        assert expired_events[0]["retry_count"] == 6
        assert expired_events[0]["max_retry_count"] == 5

    def test_instruction_projection_cap_at_five_retries(self):
        """T2 指令映射侧：retry_count > 5 不再生成补位买入指令（state_store.py:470）。"""
        day = pd.Timestamp("2026-01-09")
        base = dict(
            stock="X",
            action="buy",
            target_value=50_000.0,
            signal_date=pd.Timestamp("2026-01-05"),
            create_date=pd.Timestamp("2026-01-06"),
            last_reason="涨停",
        )
        within = PendingOrder(retry_count=DEFAULT_MAX_REPLACEMENT_ATTEMPTS, **base)
        over = PendingOrder(retry_count=DEFAULT_MAX_REPLACEMENT_ATTEMPTS + 1, **base)
        frame_within = build_instruction_frame(
            signal_date=day, delayed_orders=[within], positions={}, prices={"X": 10.0}
        )
        assert (
            len(frame_within) == 1 and frame_within.iloc[0]["ts_code"] == "X"
        ), "retry=5（上限内）必须仍生成补位买入指令"
        frame_over = build_instruction_frame(
            signal_date=day, delayed_orders=[over], positions={}, prices={"X": 10.0}
        )
        assert len(frame_over) == 0, "retry=6（超上限）不得生成补位买入指令"


# ---------------------------------------------------------------------------
# 5. 候选域 ×2（buy_execution.py:535-549）
# ---------------------------------------------------------------------------


class TestCompletionCandidateBuffer:
    def test_buffer_is_twice_unfilled_excluding_held(self):
        """buffer = 未成交槽位 × 2；排除已持仓；按候选顺序截断。"""
        candidates = [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6), ("E", 0.5)]
        positions = {"A": {"shares": 100}}
        # 未成交 2 ⇒ buffer 4；排除已持仓 A ⇒ [B, C, D, E]
        picked = _build_completion_stocks_to_try(positions, candidates, 2)
        assert picked == [("B", 0.8), ("C", 0.7), ("D", 0.6), ("E", 0.5)]
        # 未成交 1 ⇒ buffer 2 ⇒ [B, C]
        assert _build_completion_stocks_to_try(positions, candidates, 1) == [("B", 0.8), ("C", 0.7)]
        # 候选不足 buffer ⇒ 全部非持仓候选
        assert _build_completion_stocks_to_try(positions, candidates[:3], 5) == [
            ("B", 0.8),
            ("C", 0.7),
        ]
        # 部分持仓 ⇒ 只剔除持仓者（C/D/E 未持仓仍入选）
        assert _build_completion_stocks_to_try({"A": {}, "B": {}}, candidates, 2) == [
            ("C", 0.7),
            ("D", 0.6),
            ("E", 0.5),
        ]
        # 全部候选已持仓 ⇒ 空
        all_held = {s: {} for s, _score in candidates}
        assert _build_completion_stocks_to_try(all_held, candidates, 2) == []

    def test_engine_completion_uses_buffer_beyond_first_candidate(self):
        """引擎级 ×2 证据：首位补齐候选涨停时，buffer 内第二候选当日成交。

        若 buffer = 未成交×1，stocks_to_try 止于 C（当日涨停）⇒ B 买不进；
        B 在 day2 经补齐路径成交即证明 buffer ≥ 2。
        """
        stocks = ["A", "B", "C"]
        plan = {
            _DAYS[0].strftime("%Y%m%d"): [("A", 0.9), ("B", 0.8)],  # T0 计划
            # day1（补齐日 D-1 再生成候选）：A 已持仓，C 排在 B 前
            _DAYS[1].strftime("%Y%m%d"): [("A", 0.9), ("C", 0.85), ("B", 0.8)],
        }
        signal = _DeterministicSignal(2, plan, plan[_DAYS[0].strftime("%Y%m%d")])
        engine = _make_ml_engine(stocks, signal, _DAYS)
        limit_up = frozenset(
            {
                ("B", _DAYS[1].strftime("%Y%m%d")),  # T+1 主买入日 B 涨停 ⇒ 槽位未成交
                ("C", _DAYS[2].strftime("%Y%m%d")),  # 补齐日 C 涨停（首位候选不可用）
            }
        )
        price_data = _make_price_data(
            _DAYS, {s: [10.0] * len(_DAYS) for s in stocks}, limit_up=limit_up
        )
        _wire_and_drive(engine, _DAYS, price_data, upto=3)
        day1_fills = _day_fills(engine, _DAYS[1])
        assert [t["stock"] for t in day1_fills] == ["A"], "day1：A 成交、B 涨停未成交"
        day2_fills = _day_fills(engine, _DAYS[2])
        assert [t["stock"] for t in day2_fills] == [
            "B"
        ], "day2：C（首位候选）涨停被跳过，buffer 内第二候选 B 必须成交"
        assert "C" not in engine.positions
        assert engine.completion_stats["total_completed"] == 1
        assert engine.unfilled_slots == {}, "补齐成功后槽位必须出清"
        completion_records = [
            r
            for r in engine.execution_attribution_records
            if r.get("execution_stage") == "completion"
        ]
        assert len(completion_records) == 1
        assert completion_records[0]["planned_stock"] == "B"
        assert completion_records[0]["actual_stock"] == "B"
        assert completion_records[0]["status"] == "filled"
