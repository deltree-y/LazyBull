# -*- coding: utf-8 -*-
"""v2 P4 步进等价性测试（**T1 生产实现版**；阶段专用；tests/ 全量豁免复杂度）。

T1 验收增项①②：场景套件改打生产实现（生产 ``run_step`` / ``engine_state``
/ ``live_schedule``），spike 原型副本已删除；恢复契约 = **fail-fast**（缺字段/
schema 版本不符 ⇒ ``EngineStateError``），spike 宽松恢复形态退役。

覆盖：
- 生产快照导出/恢复/往返字节一致与 fail-fast 拒绝（D2 字段清单，生产版）；
- 连续 run vs 逐日步进+落盘恢复等价性（S1~S7、S9、S10 场景函数直驱，
  含旁路产物终态比对）；
- 变异自证：字段剔除（_signal_dates / stop_loss_monitor / live_anchor_state）
  ⇒ 恢复拒绝（fail-fast 检出）；值篡改（配置指纹/last_scheduled 预改）
  ⇒ 行为发散检出；且**变异结果入总判定**（R2-04）；
- live 末日 T0 计划（S7：末日前全字段比对 + 末日白名单 + 反例拒绝）；
- live 持久化锚定四场景（旧链生产函数）+ 隔离根 containment（前后快照口径，
  含越界写入可检出反例）；
- 门 2b 绑定采集 fail-closed（缺文件/缺字段 ⇒ errors + verdict=FAIL 反例）；
- 幂等探针计数机制（吞异常仍计数）与写守卫。

不依赖真实配置与真实数据：成本配置经 runner 的 ``synthetic_cost_settings``
打桩；PaperStorage 根与受保护根全部为 pytest ``tmp_path``；绑定采集用合成仓库。
"""

import json
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from scripts.v2_p4 import t05b_gate2b_binding as binding_mod
from scripts.v2_p4.t05b_gate2b_binding import collect_binding_evidence
from scripts.v2_p4.t05b_idempotency_probe import (
    CallRecorder,
    _NetworkCallError,
    _WriteGuardError,
    wrap_client_no_network,
    wrap_storage_no_write,
)
from scripts.v2_p4.t05b_live_anchor import (
    _AnchorHarness,
    run_anchor_scenarios,
    snapshot_protected_root,
    verify_isolated_root_containment,
)
from scripts.v2_p4.t05b_spike_runner import (
    _aggregate_verdict,
    _make_engine_factory,
    _make_price_data,
    _DeterministicSignal,
    scenario_s1_multi_cycle_with_holiday,
    scenario_s2_stagger_boundary,
    scenario_s3_pending_order_postpone,
    scenario_s4_stop_loss,
    scenario_s4b_stop_loss_limit_down_counter,
    scenario_s5_position_completion,
    scenario_s6_early_rebalance_signal_dates_mutation,
    scenario_s7_live_last_day_t0_plan,
    scenario_s9a_live_resume_anchor_before_window,
    scenario_s9b_live_config_change_rebuild,
    scenario_s9c_live_missed_batch_catchup,
    scenario_s9d_k1_delayed_success,
    scenario_s9e_live_early_rebalance_k2,
    scenario_s9f_first_failure_then_success,
    scenario_s9g_calendar_extension,
    scenario_s10_production_config,
    synthetic_cost_settings,
)
from scripts.v2_p4.t05b_step_driver import (
    RUNTIME_STATE_FIELDS,
    SteppedRunResult,
    compare_runs,
    export_blob,
    restore_blob,
    run_continuous_with_probes,
    run_stepped,
)
from src.lazybull.v2.core.accounting.engine_state import (
    EngineStateError,
    dumps_blob,
    export_state,
    loads_blob,
    restore_state,
    state_from_blob,
    state_to_blob,
)


@pytest.fixture(autouse=True)
def _synthetic_costs():
    with synthetic_cost_settings():
        yield


def _tiny_setup(n_days: int = 12):
    dates = list(pd.bdate_range("2026-01-05", periods=n_days))
    stocks = ["A", "B"]
    signal = _DeterministicSignal(2, {}, [("A", 0.9), ("B", 0.8)])
    factory = _make_engine_factory(stocks, signal)
    closes = {s: [10.0 * (1 + 0.001 * i) for i in range(n_days)] for s in stocks}
    price_data = _make_price_data(dates, closes)
    return dates, stocks, factory, price_data


class TestSnapshotRoundTrip:
    def test_export_restore_export_byte_identical(self):
        dates, _stocks, factory, price_data = _tiny_setup()
        engine = factory()
        engine._prepare_price_index(price_data)
        engine.price_data_cache = price_data
        engine._trade_date_index = {d: i for i, d in enumerate(dates)}
        engine._signal_dates = engine._get_rebalance_dates(dates)
        blob1 = state_to_blob(export_state(engine))

        engine2 = factory()
        engine2._prepare_price_index(price_data)
        restore_state(engine2, state_from_blob(loads_blob(dumps_blob(blob1))))
        assert dumps_blob(state_to_blob(export_state(engine2))) == dumps_blob(blob1)

    def test_snapshot_detached_from_engine(self):
        """恢复后改动新引擎不得污染已导出快照（pickle 往返切断引用）。"""
        dates, _stocks, factory, price_data = _tiny_setup()
        engine = factory()
        engine._prepare_price_index(price_data)
        engine._signal_dates = engine._get_rebalance_dates(dates)
        blob = export_blob(engine)
        engine2 = factory()
        engine2._prepare_price_index(price_data)
        restore_blob(engine2, blob)
        engine2.current_capital = -1.0
        assert blob["fields"]["current_capital"] != -1.0

    def test_missing_field_rejected_failfast(self):
        """生产恢复契约（D2）：快照缺字段 ⇒ fail-fast 拒绝（EngineStateError）。

        （spike 宽松恢复形态——缺字段回退工厂初值——已随原型副本一并退役；
        T1 验收增项②。）
        """
        dates, _stocks, factory, price_data = _tiny_setup()
        engine = factory()
        engine._prepare_price_index(price_data)
        engine._signal_dates = engine._get_rebalance_dates(dates)
        blob = export_blob(engine)
        blob["fields"].pop("current_capital")
        engine2 = factory()
        engine2._prepare_price_index(price_data)
        with pytest.raises(EngineStateError):
            restore_blob(engine2, blob)

    def test_frozen_field_list_covers_known_runtime_state(self):
        """字段清单防漂移：必含关键运行时状态，且不含输入接线/构造参数。"""
        fields = set(RUNTIME_STATE_FIELDS)
        for required in (
            "current_capital",
            "positions",
            "pending_signals",
            "pending_stop_loss_sells",
            "pending_condition_sells",
            "_cycle_anchor_idx",
            "portfolio_values",
            "trades",
            "execution_attribution_records",
            "holdings_snapshots",
            "unfilled_slots",
            "completion_stats",
            "_last_ranked_candidates",
            "_last_signal_date",
            "_last_rebalance_nav",
            "_signal_dates",  # T1 新增：run() 局部可变变量提升为引擎态
        ):
            assert required in fields
        for forbidden in (
            "price_data_cache",
            "trade_price_index",
            "pnl_price_index",
            "_trade_date_index",
            "_run_total_days",  # 驱动上下文（wiring），不入快照
            "universe",
            "signal",
            "cost_model",
            "data_storage",
            "_suspend_calendar",
            "record_holdings_snapshot",  # 配置型标志（host 注入，不入快照，P3-5）
        ):
            assert forbidden not in fields


class TestSteppedEquivalence:
    """连续 vs 逐日步进+落盘恢复：场景函数直驱（含 seam 说明）。"""

    def test_s1_multi_cycle_with_holiday(self):
        result = scenario_s1_multi_cycle_with_holiday()
        assert result["verdict_ok"], result["probe_mismatches"]
        assert result["n_days"] == 30  # ≥2 个完整调仓周期（freq=5 ⇒ 6 周期）
        assert result["n_trades"] > 0
        assert result["snapshot_roundtrip_ok"]
        assert result["sidecar_identical"]

    def test_s2_stagger_boundary(self):
        result = scenario_s2_stagger_boundary()
        assert result["verdict_ok"], result["probe_mismatches"]
        assert result["n_trades"] > 0

    def test_s3_pending_order_postpone(self):
        result = scenario_s3_pending_order_postpone()
        assert result["verdict_ok"], result["probe_mismatches"]
        assert result["n_trades"] > 0

    def test_s4_stop_loss(self):
        result = scenario_s4_stop_loss()
        assert result["verdict_ok"], result["probe_mismatches"]
        assert result["n_trades"] > 0

    def test_s5_position_completion(self):
        result = scenario_s5_position_completion()
        assert result["verdict_ok"], result["probe_mismatches"]

    def test_s6_early_rebalance_rewrites_signal_dates(self):
        result = scenario_s6_early_rebalance_signal_dates_mutation()
        assert result["verdict_ok"], result["probe_mismatches"]

    def test_s10_production_config(self):
        result = scenario_s10_production_config()
        assert result["verdict_ok"], result["probe_mismatches"]
        assert result["n_days"] == 60  # 3 个完整周期（freq=20, K=2）
        assert result["n_trades"] > 0


class TestMutationDetection:
    """变异自证：快照遗漏关键字段 ⇒ 与连续侧必发散（证明 harness 敏感性）。"""

    def test_drop_signal_dates_diverges(self):
        result = scenario_s6_early_rebalance_signal_dates_mutation()
        mutant = result["mutant"]
        assert mutant["expect_pass"] is False
        assert mutant["verdict_ok"], "漏 signal_dates 的变异未被判发散"
        assert not mutant["passed"]

    def test_drop_stop_loss_monitor_diverges(self):
        result = scenario_s4b_stop_loss_limit_down_counter()
        assert result["verdict_ok"], result["probe_mismatches"]
        mutant = result["mutant"]
        assert mutant["expect_pass"] is False
        assert mutant["verdict_ok"], "漏 stop_loss_monitor 的变异未被判发散"
        assert not mutant["passed"]

    def test_mutant_verdict_enters_overall_aggregation(self):
        """R2-04：嵌套变异失败必须拖垮总判定（合成回归：伪造 mutant 失效）。"""
        good = {"verdict_ok": True, "mutant": {"verdict_ok": True}}
        bad = {"verdict_ok": True, "mutant": {"verdict_ok": False}}
        anchor = {"passed": True}
        assert _aggregate_verdict([good], anchor) is True
        assert _aggregate_verdict([bad], anchor) is False
        assert _aggregate_verdict([{"verdict_ok": False}], anchor) is False
        assert _aggregate_verdict([good], {"passed": False}) is False


class TestLiveAnchorEngineWiring:
    """S9：live 锚定状态接入引擎步进驱动（R2-03 首轮 + 复审四子项处置）。

    复审口径：K=1 按实际成功日 / 配置变更提案-提交 / S9b~d 双侧比较 + 旧链
    oracle 逐日一致 / 锚定状态随快照同代落盘（blob live_anchor 键）。
    """

    def test_s9a_resume_with_anchor_before_window(self, tmp_path):
        result = scenario_s9a_live_resume_anchor_before_window(tmp_path)
        assert result["verdict_ok"], result["probe_mismatches"]
        # 信号日 = 锚定排期 0/5/10/15/20（anchor < 驱动首日 6）
        assert [d for d, _t in result["signal_day_trace"]] == [
            "20260105",
            "20260112",
            "20260119",
            "20260126",
            "20260202",
        ]
        mutant = result["mutant"]
        assert mutant["verdict_ok"], "锚定状态未落盘（重锚）变异未被判发散"

    def test_s9b_config_change_rebuilds_anchor(self, tmp_path):
        result = scenario_s9b_live_config_change_rebuild(tmp_path)
        assert result["verdict_ok"], result["detail"]
        # 双侧比较（live 连续 vs 步进恢复）+ 旧链 oracle 逐日一致 + 成交非空
        assert result["sidecar_identical"]
        assert result["n_trades"] > 0
        assert result["anchor_state_final"]["rebalance_freq"] == 4
        mutant = result["mutant"]
        assert mutant["verdict_ok"], "配置变更未检出变异未被判发散"

    def test_s9c_missed_batch_catchup_daily(self, tmp_path):
        result = scenario_s9c_live_missed_batch_catchup(tmp_path)
        assert result["verdict_ok"], result["detail"]
        # 漏批 6(t0)/9(t1)/12(t0) 在 11/12/13 逐日消化，此后正常排期
        phase2_trace = [x for x in result["signal_day_trace"] if x[0] >= "20260120"]
        assert phase2_trace[:3] == [("20260120", 0), ("20260121", 1), ("20260122", 0)]
        assert result["sidecar_identical"]
        mutant = result["mutant"]
        assert mutant["verdict_ok"], "漏批不追赶变异未被判发散"

    def test_s9d_k1_delayed_success_actual_day_rule(self, tmp_path):
        """K=1 按实际成功日判到期（reviewer2 实证案例：0119 不到期/0120 到期）。"""
        result = scenario_s9d_k1_delayed_success(tmp_path)
        assert result["verdict_ok"], result["detail"]
        days = [d for d, _t in result["signal_day_trace"]]
        assert days == ["20260105", "20260113", "20260120", "20260127", "20260203"]
        assert "20260119" not in days  # 距实际成功日 0113 仅 4 个交易日
        oracle_days = [d for d, _t, _p in result["oracle_trace"]]
        assert oracle_days == days  # 与旧链生产函数逐日一致

    def test_s9e_live_early_rebalance_k2(self, tmp_path):
        """P1-1：K>1 空仓提前调仓 ⇒ 锚前移当日 + 排期自该日重建（无删除回流）。"""
        result = scenario_s9e_live_early_rebalance_k2(tmp_path)
        assert result["verdict_ok"], result["detail"]
        trace = result["signal_day_trace"]
        assert ("20260119", 0) in trace  # 空仓提前调仓日（非初始排期表成员）
        assert ("20260122", 1) in trace  # 提前调仓后 +3 交易日 = 批1（自该日重建）
        oracle_pairs = [(d, _t) for d, _t, _p in result["oracle_trace"]]
        assert oracle_pairs[: len(trace)] == trace

    def test_s9f_first_failure_then_success(self, tmp_path):
        """R2-P4-T1-01：首次 T0 失败后成功 ⇒ 锚 = 成功日（非失败计划日）。"""
        result = scenario_s9f_first_failure_then_success(tmp_path)
        assert result["verdict_ok"], result["detail"]
        days = [d for d, _t in result["signal_day_trace"]]
        assert days[0] == "20260106"  # 首次成功日（0105 被门控阻断）
        assert days[1] == "20260109"  # 自成功日 +3 交易日 = 批1
        oracle_pairs = [(d, _t) for d, _t, _p in result["oracle_trace"]]
        assert oracle_pairs[: len(days)] == result["signal_day_trace"]

    def test_s9g_calendar_extension(self, tmp_path):
        """R2-P4-T1-02：日历延长 + 配置不变 ⇒ 排期自动扩展（不停止调仓）。"""
        result = scenario_s9g_calendar_extension(tmp_path)
        assert result["verdict_ok"], result["detail"]
        days = [d for d, _t in result["signal_day_trace"]]
        assert "20260121" in days  # 短日历（12 日）耗尽后的延长段仍有调仓
        oracle_pairs = [(d, _t) for d, _t, _p in result["oracle_trace"]]
        assert oracle_pairs[: len(days)] == result["signal_day_trace"]


class TestLiveLastDayT0Plan:
    def test_s7_last_quote_day_generates_next_day_plan(self):
        result = scenario_s7_live_last_day_t0_plan()
        assert result["verdict_ok"], result["probe_mismatches"]
        assert result["last_day_t0_plan_ok"]
        assert result["last_day_t0_plan"]["n_signals"] == 2
        assert result["continuous_last_day_plan_absent"]
        assert result["price_data_max_date"] < result["calendar_extends_to"]

    def test_s7_pre_last_day_corruption_rejected(self):
        """R2-05 反例：末日前日期的关键状态被改坏 ⇒ 比对必须拒绝。"""
        dates = list(pd.bdate_range("2026-01-05", periods=21))
        future = list(pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=3))
        full_calendar = dates + future
        stocks = ["A", "B"]
        plan = {}
        for i, d in enumerate(dates):
            plan[d.strftime("%Y%m%d")] = [("A", 0.9), ("B", 0.8)] if (i // 5) % 2 == 0 else [
                ("B", 0.9),
                ("A", 0.8),
            ]
        signal = _DeterministicSignal(2, plan, plan[dates[0].strftime("%Y%m%d")])
        factory = _make_engine_factory(stocks, signal)
        n = len(dates)
        closes = {s: [10.0 * (1 + 0.001 * i) for i in range(n)] for s in stocks}
        price_data = _make_price_data(dates, closes)
        cont_engine, _nav, cont_probes = run_continuous_with_probes(
            factory, dates[0], dates[-1], dates, price_data
        )
        stepped = run_stepped(
            factory, dates[0], dates[-1], full_calendar, price_data, live_full_calendar=True
        )
        # 改坏末日前（第 2 日）的最近候选
        stepped.probes[1].last_ranked_candidates = [("WRONG", -999.0)]
        pre_view = SteppedRunResult(
            engine_state=stepped.engine_state, probes=stepped.probes[:-1]
        )
        verdict = compare_runs(
            cont_engine,
            cont_probes[:-1],
            pre_view,
            signal_dates_max=dates[-1].strftime("%Y%m%d"),
        )
        assert not verdict.passed
        assert any("last_ranked_candidates" in m for m in verdict.probe_mismatches)


class TestEngineDayPhaseGoldenSequence:
    """引擎日相位黄金序列（P2-1 处置）：run_step 内相位调用顺序锁定。

    旧 11 步 ↔ run_t0/run_t1 ↔ 引擎日相位映射表见规划 §17.3；本测试锁定
    引擎侧相位顺序（P1 延迟订单 → P2 止损检查 → P3 信号日 → P4 当日执行
    序列 → P5 提前调仓 → P6 仓位补齐 → P7 估值 → P8 日终收口）——宿主侧
    run_t0/run_t1 编排与该顺序的对齐是 T2 前置硬项（§17.7 登记）。
    """

    def test_run_step_phase_order(self):
        dates, _stocks, factory, price_data = _tiny_setup()
        engine = factory()
        engine._trade_date_index = {d: i for i, d in enumerate(dates)}
        engine._prepare_price_index(price_data)
        engine.price_data_cache = price_data
        engine._run_total_days = len(dates)
        engine._signal_dates = engine._get_rebalance_dates(dates)

        calls: List[str] = []
        for phase in (
            "_process_pending_orders",
            "_check_stop_loss",
            "_run_signal_day",
            "_run_daily_trade_executions",
            "_try_early_rebalance",
            "_process_position_completion",
            "_finalize_day",
        ):
            original = getattr(engine, phase)

            def _make_spy(name, fn):
                def _spy(*args, **kwargs):
                    calls.append(name)
                    return fn(*args, **kwargs)

                return _spy

            setattr(engine, phase, _make_spy(phase, original))

        for date in dates[:3]:
            engine.run_step(date, dates, price_data)

        expected_per_day = [
            "_process_pending_orders",
            "_check_stop_loss",
            "_run_signal_day",
            "_run_daily_trade_executions",
            "_try_early_rebalance",
            "_process_position_completion",
            "_finalize_day",
        ]
        # 每日相位顺序逐日一致（引擎默认启用延迟订单/仓位补齐；无止损配置时
        # _check_stop_loss 不被调用——按实际启用集过滤期望；P7 估值为
        # _calculate_portfolio_value，被 P3 内部复用，不在顶层相位序列断言内）
        if engine.stop_loss_monitor is None:
            expected_per_day.remove("_check_stop_loss")
        assert calls[: len(expected_per_day)] == expected_per_day
        assert calls[len(expected_per_day): 2 * len(expected_per_day)] == expected_per_day


class TestLiveAnchorScenarios:
    def test_all_five_scenarios_on_isolated_root(self, tmp_path):
        trade_dates = [d.strftime("%Y%m%d") for d in pd.bdate_range("2026-01-05", periods=120)]
        protected = tmp_path / "protected_paper"
        protected.mkdir()
        (protected / "account.json").write_text("{}", encoding="utf-8")
        before = snapshot_protected_root(protected)
        results = run_anchor_scenarios(tmp_path / "iso", trade_dates, freq=20, k=3)
        by_name = {r.name: r for r in results}
        assert set(by_name) == {
            "A_first_run_tranche0",
            "B_stagger_hit",
            "C_config_change_rebuild_anchor",
            "D_missed_batch_catchup",
            "E_anchor_equals_window_start",
        }
        for r in results:
            assert r.passed, f"{r.name}: {r.detail}"
        ok, detail = verify_isolated_root_containment(tmp_path / "iso", protected, before)
        assert ok, detail

    def test_containment_detects_protected_root_change(self, tmp_path):
        """R2-06 反例：受保护根在两次快照间被写入 ⇒ 必检出。"""
        protected = tmp_path / "protected_paper"
        protected.mkdir()
        (protected / "account.json").write_text("{}", encoding="utf-8")
        before = snapshot_protected_root(protected)
        isolated = tmp_path / "iso"
        (isolated / "runs").mkdir(parents=True)
        (isolated / "runs" / "rebalance_state.json").write_text("{}", encoding="utf-8")
        # 越界写入（受保护根新增文件）
        (protected / "escape.json").write_text("x", encoding="utf-8")
        ok, detail = verify_isolated_root_containment(isolated, protected, before)
        assert not ok
        assert "escape.json" in detail
        # 内容篡改同样检出
        (protected / "escape.json").unlink()
        after_tamper = snapshot_protected_root(protected)
        (protected / "account.json").write_text('{"x": 1}', encoding="utf-8")
        ok2, _detail2 = verify_isolated_root_containment(isolated, protected, before)
        assert not ok2
        assert after_tamper == before  # 未篡改前快照与初始一致

    def test_anchor_equals_window_start_schedule(self):
        from src.lazybull.v2.common.rules.stagger import compute_tranche_schedule
        from src.lazybull.v2.common.rules.stagger import build_tranche_schedule_from_anchor

        dates = [d.strftime("%Y%m%d") for d in pd.bdate_range("2026-01-05", periods=60)]
        assert build_tranche_schedule_from_anchor(dates[0], dates, 20, 3) == compute_tranche_schedule(
            dates, 20, 3
        )

    def test_config_change_rebuilds_anchor_same_day(self, tmp_path):
        dates = [d.strftime("%Y%m%d") for d in pd.bdate_range("2026-01-05", periods=40)]
        harness = _AnchorHarness(tmp_path, dates)
        harness.paper_storage.save_rebalance_state(
            {
                "last_rebalance_date": dates[0],
                "tranche_anchor_date": dates[0],
                "last_scheduled_rebalance_date": dates[0],
                "rebalance_freq": 20,
                "stagger_tranches": 3,
            }
        )
        is_due, tranche = harness._check_rebalance_day(dates[5], 10, 3)
        assert (is_due, tranche) == (True, 0)
        assert harness._resolved_rebalance_plan_date == dates[5]


class TestIdempotencyProbeMechanics:
    def test_wrapped_client_counts_even_when_exception_swallowed(self):
        """R2-02 反例：调用方吞掉异常时，计数仍留痕（门禁不得假 PASS）。"""

        class _FakeClient:
            def get_stock_st(self, **kwargs):
                return "network"

            def query(self, api, **kwargs):
                return "network"

        recorder = CallRecorder()
        client = _FakeClient()
        wrapped = wrap_client_no_network(client, recorder)
        assert wrapped == ["get_stock_st", "query"]
        # 模拟 ensure 的 except Exception fail-soft 分支吞掉禁网异常
        for _ in range(2):
            try:
                client.get_stock_st(trade_date="20260731")
            except _NetworkCallError:
                pass
        assert len(recorder.events) == 2
        assert all(e["kind"] == "network" for e in recorder.events)
        assert recorder.events[0]["date_args"] == ["20260731"]

    def test_write_guard_counts_and_blocks(self):
        class _FakeStorage:
            def save_raw_by_date(self, df, dataset, trade_date):
                return "written"

            def load_raw(self, dataset):
                return None

        recorder = CallRecorder()
        storage = _FakeStorage()
        wrapped = wrap_storage_no_write(storage, recorder)
        assert wrapped == ["save_raw_by_date"]
        with pytest.raises(_WriteGuardError):
            storage.save_raw_by_date(None, "daily", "20260731")
        assert len(recorder.events) == 1 and recorder.events[0]["kind"] == "write"
        assert storage.load_raw("daily") is None  # 读方法不受影响


class TestGate2bBinding:
    def _make_synthetic_repo(self, root: Path) -> None:
        for batch in binding_mod.FROZEN_RUNS_BATCHES:
            meta_dir = root / "data" / "runs" / batch
            meta_dir.mkdir(parents=True)
            (meta_dir / "batch_meta.json").write_text(
                '{"batch_id": "%s", "host_mode": "backtest", '
                '"data_state": {"data_state_id": "abc123", "git_commit": "deadbeef"}}' % batch,
                encoding="utf-8",
            )
        raw_dir = root / binding_mod.B0_RAW_DIR
        raw_dir.mkdir(parents=True)
        (raw_dir / "placeholder.parquet").write_bytes(b"PAR1")
        # 数据态文件在 raw/ 子目录（P1-1 修正后的真实布局）
        (raw_dir / "data_state_wf_test.json").write_text(
            '{"data_state_id": "ds_test", "raw_latest_partitions": {"daily": "2026-07-31"}}',
            encoding="utf-8",
        )
        ev_dir = root / binding_mod.HISTORICAL_EVIDENCE_DIR
        ev_dir.mkdir(parents=True)
        (ev_dir / "regression_edge_verdict.json").write_text(
            '{"verdict": "PASS", "new_batch": "x", "frozen_batch": "y"}', encoding="utf-8"
        )
        (ev_dir / "consumption_before.json").write_text("{}", encoding="utf-8")
        for tool in binding_mod.COMPARE_TOOLS:
            tool_path = root / tool
            tool_path.parent.mkdir(parents=True, exist_ok=True)
            tool_path.write_text("# stub", encoding="utf-8")

    def test_collect_binding_evidence_synthetic(self, tmp_path):
        self._make_synthetic_repo(tmp_path)
        evidence = collect_binding_evidence(tmp_path)
        assert evidence["verdict"] == "PASS"
        assert evidence["errors"] == []
        assert evidence["binding_rule"] == binding_mod.BINDING_RULE
        assert len(evidence["frozen_runs_batches"]) == 3
        for batch in evidence["frozen_runs_batches"]:
            assert batch["data_state_id"] == "abc123"
            assert len(batch["batch_meta_sha256"]) == 64
        b0 = evidence["b0_raw_dir"]
        assert b0["file_count"] == 2
        assert b0["data_state_id"] == "ds_test"  # glob 定位 raw/ 内数据态文件
        assert b0["data_state_sha256"] and len(b0["data_state_sha256"]) == 64
        assert b0["raw_latest_partitions"] == {"daily": "2026-07-31"}
        assert evidence["historical_pass_evidence"]["verdict"] == "PASS"
        assert all(len(t["sha256"]) == 64 for t in evidence["compare_tools"])

    def test_missing_compare_tools_fail_closed(self, tmp_path):
        """fail-closed 反例：比较工具缺失 ⇒ errors 非空且 verdict=FAIL。"""
        self._make_synthetic_repo(tmp_path)
        for tool in binding_mod.COMPARE_TOOLS:
            (tmp_path / tool).unlink()
        evidence = collect_binding_evidence(tmp_path)
        assert evidence["verdict"] == "FAIL"
        assert any("比较工具不存在" in e for e in evidence["errors"])

    def test_missing_b0_data_state_file_fails_closed(self, tmp_path):
        """P1-1/R2-01 反例：B0 数据态文件缺失 ⇒ errors 非空且 verdict=FAIL。"""
        self._make_synthetic_repo(tmp_path)
        (tmp_path / binding_mod.B0_RAW_DIR / "data_state_wf_test.json").unlink()
        evidence = collect_binding_evidence(tmp_path)
        assert evidence["verdict"] == "FAIL"
        assert any("数据态文件" in e for e in evidence["errors"])

    def test_missing_batch_meta_propagates_to_top_errors(self, tmp_path):
        """R2-01 反例 2：批次缺失的子对象错误必须上冒顶层 errors。"""
        self._make_synthetic_repo(tmp_path)
        import shutil

        shutil.rmtree(tmp_path / "data" / "runs" / binding_mod.FROZEN_RUNS_BATCHES[0])
        evidence = collect_binding_evidence(tmp_path)
        assert evidence["verdict"] == "FAIL"
        assert any("batch_meta.json 不存在" in e for e in evidence["errors"])


class TestRunnerSmoke:
    def test_run_all_scenarios_writes_evidence(self, tmp_path, monkeypatch):
        from scripts.v2_p4 import t05b_spike_runner as runner_mod

        # 绑定采集指回合成仓库（不触真实 data/runs），证据输出落 tmp
        TestGate2bBinding()._make_synthetic_repo(tmp_path / "repo")
        monkeypatch.setattr(runner_mod, "_REPO_ROOT", tmp_path / "repo")
        summary = runner_mod.run_all_scenarios(
            tmp_path / "out", repo_root=tmp_path / "repo", include_probe=False
        )
        assert summary["verdict"] == "PASS"
        evidence_path = Path(summary["evidence_path"])
        assert evidence_path.exists()

        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert evidence["verdict"] == "PASS"
        assert len(evidence["step_equivalence_scenarios"]) == 16  # S1~S7 + S9a~S9g + S10
        assert evidence["live_anchor_scenarios"]["passed"]
        # 门 2b 绑定留证落盘且绑定合成仓库数据态（fail-closed 齐备）
        binding = json.loads(Path(summary["binding_path"]).read_text(encoding="utf-8"))
        assert binding["verdict"] == "PASS"
        assert binding["frozen_runs_batches"][0]["data_state_id"] == "abc123"
        assert binding["b0_raw_dir"]["data_state_id"] == "ds_test"
