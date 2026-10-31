# -*- coding: utf-8 -*-
"""v2 P4 T1 live 持久化锚定排期黄金测试（tests/ 全量豁免复杂度）。

锁定 D1 排期来源二分的 live 侧（F25 + T0.5b 复审形态的生产复刻）：
- K=1：按**实际成功日**（last_rebalance_date）计交易日判到期（首次运行放行；
  排期表对 K=1 不构成成员资格）；计划日未成功不推进状态；
- K>1：按计划日窗口 + 漏批逐日追赶最早未完成批次；
- 配置变更「提案—成功 T0 后提交」：T0 失败零提交，次日仍按旧配置检出变更；
- 锚定初始化仅限首次（anchor 不在日历内拒绝）；缺锚定状态的政策判定/结算拒绝；
- LiveAnchorState blob 投影封闭键集（缺键/多键 fail-fast）。

端到端黄金序列（anchor 早于驱动首日续跑 / 配置变更重建 / 漏批追赶 / K=1 延迟
成功）由生产实现版场景套件 S9a~S9d 承载（tests/test_v2_p4_t05b_spike.py）。
"""

import pandas as pd
import pytest

from src.lazybull.v2.core.execution.live_schedule import (
    LiveAnchorState,
    LiveScheduleError,
    apply_live_policy,
    init_live_anchor,
    settle_live_day,
)


class _StubEngine:
    """最小引擎桩（live 排期政策消费的引擎面）。"""

    def __init__(self, rebalance_freq: int = 5, stagger_tranches: int = 1) -> None:
        self.rebalance_freq = rebalance_freq
        self.stagger_tranches = stagger_tranches
        self._signal_dates = {}
        self._cycle_anchor_idx = 0
        self.live_anchor_state = None
        self.pending_signals = {}


def _calendar(n: int = 30) -> list:
    return list(pd.bdate_range("2026-01-05", periods=n))


def _mark_t0_success(engine: _StubEngine, date: pd.Timestamp, tranche_idx: int = 0) -> None:
    """模拟成功 T0：pending_signals 入队带 decision_trace 的调仓计划。"""
    engine.pending_signals[date] = {"signals": {"A": 1.0}, "tranche_idx": tranche_idx,
                                    "decision_trace": {"queued": True}}


class TestInitLiveAnchor:
    def test_first_run_tranche0_schedule_and_anchor(self):
        engine = _StubEngine(5, 1)
        calendar = _calendar()
        init_live_anchor(engine, calendar[0], calendar)
        assert engine.live_anchor_state.anchor_date == calendar[0]
        assert engine.live_anchor_state.last_rebalance_date is None
        assert engine.live_anchor_state.last_scheduled_rebalance_date is None
        assert engine._cycle_anchor_idx == 0
        # 排期 = 自锚每 5 日（0/5/10/15/20/25）
        assert sorted(calendar.index(d) for d in engine._signal_dates) == [0, 5, 10, 15, 20, 25]

    def test_anchor_not_in_calendar_rejected(self):
        engine = _StubEngine(5, 1)
        with pytest.raises(LiveScheduleError):
            init_live_anchor(engine, pd.Timestamp("2030-01-01"), _calendar())


class TestK1DueByActualSuccessDay:
    def test_first_run_due_then_due_by_actual_success(self):
        """K=1 黄金序列：首日放行；第 5 日（计划日）未成功不推进；第 6 日成功
        ⇒ 下一轮按实际成功日 +5 交易日到期（第 11 日到期、第 10 日不到期）。"""
        engine = _StubEngine(5, 1)
        calendar = _calendar()
        init_live_anchor(engine, calendar[0], calendar)
        # 首日：首次运行放行
        day0 = apply_live_policy(engine, calendar[0], calendar)
        assert day0.due == [calendar[0]]
        engine._signal_dates = day0.working
        _mark_t0_success(engine, calendar[0])
        settle_live_day(engine, calendar[0], day0)
        assert engine.live_anchor_state.last_rebalance_date == calendar[0]
        # 第 5 日（idx5）：计划日到期；但模拟 T0 失败（不入队）⇒ 不提交
        day5 = apply_live_policy(engine, calendar[5], calendar)
        assert day5.due == [calendar[5]]
        settle_live_day(engine, calendar[5], day5)  # pending_signals 无当日 ⇒ 零提交
        assert engine.live_anchor_state.last_rebalance_date == calendar[0]
        # 第 6 日（idx6）：仍到期（按实际成功日已过 5 个交易日？idx6-idx0=6>=5 ⇒ 到期）
        # ——注意：判到期看「距实际成功日」，0105 成功后 idx6 已满足 freq=5
        day6 = apply_live_policy(engine, calendar[6], calendar)
        assert day6.due == [calendar[6]]
        engine._signal_dates = day6.working
        _mark_t0_success(engine, calendar[6])
        settle_live_day(engine, calendar[6], day6)
        assert engine.live_anchor_state.last_rebalance_date == calendar[6]
        # 此后按实际成功日 idx6 计：idx10（差 4）不到期，idx11（差 5）到期
        assert apply_live_policy(engine, calendar[10], calendar).due == []
        assert apply_live_policy(engine, calendar[11], calendar).due == [calendar[11]]

    def test_k1_schedule_table_not_membership(self):
        """K=1 工作副本不含排期表成员资格（纯到期注入）：表内杂项条目不驱动，
        到期判定唯一来源 = 距实际成功日的交易日数。"""
        engine = _StubEngine(5, 1)
        calendar = _calendar()
        init_live_anchor(engine, calendar[0], calendar)
        # 首日成功 T0（实际成功日 = calendar[0]）
        day0 = apply_live_policy(engine, calendar[0], calendar)
        engine._signal_dates = day0.working
        _mark_t0_success(engine, calendar[0])
        settle_live_day(engine, calendar[0], day0)
        # 权威表注入杂项条目（calendar[1] 非排期日）：K=1 到期判定不得受其驱动
        engine._signal_dates = {calendar[1]: 0, calendar[5]: 0}
        day1 = apply_live_policy(engine, calendar[1], calendar)
        assert day1.working == {} and day1.due == []  # 表成员资格不生效
        # 到期日（距实际成功日 5 个交易日）：即使权威表无该日，仍注入
        engine._signal_dates = {}
        day5 = apply_live_policy(engine, calendar[5], calendar)
        assert day5.due == [calendar[5]] and day5.working == {calendar[5]: 0}


class TestK1PlusCatchUp:
    def test_missed_batches_drain_earliest_first(self):
        """K=2 漏批追赶：停机跨 3 计划日 ⇒ 重启后逐日消化最早未完成批次。"""
        engine = _StubEngine(6, 2)
        calendar = _calendar(30)
        init_live_anchor(engine, calendar[0], calendar)
        # 排期：freq=6, K=2 ⇒ 偏移 0/3（批次0: 0,6,12,18,24；批次1: 3,9,15,21,27）
        assert [ (calendar.index(d), t) for d, t in sorted(engine._signal_dates.items(), key=lambda kv: kv[0]) ][:4] == [
            (0, 0), (3, 1), (6, 0), (9, 1),
        ]
        # 首日批次0成功
        day0 = apply_live_policy(engine, calendar[0], calendar)
        engine._signal_dates = day0.working
        _mark_t0_success(engine, calendar[0], tranche_idx=0)
        settle_live_day(engine, calendar[0], day0)
        assert engine.live_anchor_state.last_scheduled_rebalance_date == calendar[0]
        # 停机至第 10 日（错过 idx3 批1 / idx6 批0 / idx9 批1）
        day10 = apply_live_policy(engine, calendar[10], calendar)
        assert day10.due == [calendar[3], calendar[6], calendar[9]]  # 窗口内全部漏批
        assert day10.working[calendar[10]] == 1  # 当日执行最早未完成批次（idx3 ⇒ tranche 1）
        engine._signal_dates = day10.working
        _mark_t0_success(engine, calendar[10], tranche_idx=1)
        settle_live_day(engine, calendar[10], day10)
        assert engine.live_anchor_state.last_scheduled_rebalance_date == calendar[3]
        # 次日继续追赶 idx6（批0）
        day11 = apply_live_policy(engine, calendar[11], calendar)
        assert day11.due == [calendar[6], calendar[9]]
        assert day11.working[calendar[11]] == 0


class TestConfigChangeProposalCommit:
    def test_proposal_commits_only_on_successful_t0(self):
        """配置变更提案-提交：T0 失败零提交（次日仍按旧配置检出变更）。"""
        engine = _StubEngine(5, 1)
        calendar = _calendar()
        init_live_anchor(engine, calendar[0], calendar)
        day0 = apply_live_policy(engine, calendar[0], calendar)
        engine._signal_dates = day0.working
        _mark_t0_success(engine, calendar[0])
        settle_live_day(engine, calendar[0], day0)

        # 配置变更 freq 5→4（次日生效提案）
        engine.rebalance_freq = 4
        day1 = apply_live_policy(engine, calendar[1], calendar)
        assert day1.proposal is not None and day1.proposal["rebalance_freq"] == 4
        assert day1.due == [calendar[1]]  # 变更当日即拟议调仓日
        engine._signal_dates = day1.working
        # T0 失败（不入队）⇒ 零提交：锚定指纹仍为旧配置
        settle_live_day(engine, calendar[1], day1)
        assert engine.live_anchor_state.rebalance_freq == 5
        # 次日仍按旧配置检出变更（再次提案）
        day2 = apply_live_policy(engine, calendar[2], calendar)
        assert day2.proposal is not None
        engine._signal_dates = day2.working
        _mark_t0_success(engine, calendar[2])
        settle_live_day(engine, calendar[2], day2)
        # 成功 T0 后提交：指纹 + 锚 + 权威表替换
        assert engine.live_anchor_state.rebalance_freq == 4
        assert engine.live_anchor_state.anchor_date == calendar[2]
        assert engine.live_anchor_state.last_rebalance_date == calendar[2]
        # 新表按 freq=4 自当日重建（K=1：每 4 交易日），具体键集断言（P3-2 处置）
        from src.lazybull.v2.common.rules.stagger import build_tranche_schedule_from_anchor

        expected = build_tranche_schedule_from_anchor(calendar[2], calendar, 4, 1)
        assert engine._signal_dates == expected
        assert calendar[2] in engine._signal_dates and calendar[6] in engine._signal_dates


class TestEarlyRebalanceAnchorRebuild:
    """P1-1 处置黄金测试：K>1 空仓提前调仓 ⇒ 锚前移当日 + 排期自该日重建。

    复核 reviewer1 的数值期望（freq=6/K=2，01-07 提前调仓成功 ⇒ 下一到期
    01-12 批1，而非持久表删除回流的 01-16）。
    """

    def test_early_rebalance_moves_anchor_and_rebuilds(self):
        engine = _StubEngine(6, 2)
        calendar = _calendar(20)  # 2026-01-05 起 20 个工作日
        init_live_anchor(engine, calendar[0], calendar)
        # day0 批次0成功
        day0 = apply_live_policy(engine, calendar[0], calendar)
        engine._signal_dates = day0.working
        _mark_t0_success(engine, calendar[0], tranche_idx=0)
        settle_live_day(engine, calendar[0], day0)
        # day2（01-07，非排期日）空仓提前调仓成功（due 为空）
        day2 = apply_live_policy(engine, calendar[2], calendar)
        assert day2.due == []  # 非到期日
        engine._signal_dates = day2.working
        _mark_t0_success(engine, calendar[2], tranche_idx=0)
        settle_live_day(engine, calendar[2], day2)
        # 锚前移当日 + last_scheduled = 当日
        assert engine.live_anchor_state.anchor_date == calendar[2]
        assert engine.live_anchor_state.last_scheduled_rebalance_date == calendar[2]
        # 排期自 01-07 重建：批次1 = +3 交易日 = 01-12（reviewer1 期望；持久表
        # 删除回流形态会等到 01-16）
        day5 = apply_live_policy(engine, calendar[5], calendar)
        assert calendar[5].strftime("%m%d") == "0112"
        assert day5.due == [calendar[5]] and day5.working[calendar[5]] == 1

    def test_no_persistent_table_deletion_reflow(self):
        """无提案日 settle 不把工作副本删除回流到权威语义——权威 = 锚 + 每日重建。"""
        engine = _StubEngine(6, 2)
        calendar = _calendar(20)
        init_live_anchor(engine, calendar[0], calendar)
        day0 = apply_live_policy(engine, calendar[0], calendar)
        engine._signal_dates = day0.working
        _mark_t0_success(engine, calendar[0], tranche_idx=0)
        settle_live_day(engine, calendar[0], day0)
        # 模拟工作副本在运行期被删条目（提前调仓清理）——次日重建不受影响
        day1 = apply_live_policy(engine, calendar[1], calendar)
        day1.working.pop(calendar[3], None)  # 删除下一批次1 计划日
        engine._signal_dates = day1.working
        settle_live_day(engine, calendar[1], day1)  # 无成功 T0
        # 次日：批次1 仍到期（锚未动，重建自锚）
        day3 = apply_live_policy(engine, calendar[3], calendar)
        assert day3.due == [calendar[3]] and day3.working[calendar[3]] == 1


class TestNeverSucceededK2:
    """R2-P4-T1-01 处置：K>1 尚无成功 T0 ⇒ 每日放行 tranche0，成功后锚 = 成功日。"""

    def test_daily_tranche0_until_first_success(self):
        engine = _StubEngine(6, 2)
        calendar = _calendar(20)
        init_live_anchor(engine, calendar[0], calendar)
        # day0：首次运行放行；T0 失败（不入队）⇒ 零提交
        day0 = apply_live_policy(engine, calendar[0], calendar)
        assert day0.due == [calendar[0]] and day0.working == {calendar[0]: 0}
        engine._signal_dates = day0.working
        settle_live_day(engine, calendar[0], day0)
        assert engine.live_anchor_state.last_rebalance_date is None
        # day1：仍放行（不把失败日当锚）
        day1 = apply_live_policy(engine, calendar[1], calendar)
        assert day1.due == [calendar[1]] and day1.working == {calendar[1]: 0}
        engine._signal_dates = day1.working
        _mark_t0_success(engine, calendar[1], tranche_idx=0)
        settle_live_day(engine, calendar[1], day1)
        # 首次成功 ⇒ 锚/last_scheduled/last_rebalance 全部 = 成功日
        anchor = engine.live_anchor_state
        assert anchor.anchor_date == calendar[1]
        assert anchor.last_scheduled_rebalance_date == calendar[1]
        assert anchor.last_rebalance_date == calendar[1]
        # 后续排期自成功日重建：批次1 = 0106+3 = 0109
        day4 = apply_live_policy(engine, calendar[4], calendar)
        assert calendar[4].strftime("%m%d") == "0109"
        assert day4.due == [calendar[4]] and day4.working[calendar[4]] == 1


class TestFailOpenAndCalendarEdge:
    """P2-4 处置：旧链 fail-open 兜底复刻 + 日历边界。"""

    def test_k1_anchor_date_not_in_calendar_fail_open(self):
        """K=1 锚定日不在注入日历（迁移/脏状态）⇒ fail-open 放行当日 tranche0
        （旧链 _check_single_rebalance_day 异常兜底语义），不抛异常。"""
        engine = _StubEngine(5, 1)
        calendar = _calendar(20)
        init_live_anchor(engine, calendar[0], calendar)
        # 模拟迁移脏状态：锚定实际成功日不在新日历内
        engine.live_anchor_state.last_rebalance_date = pd.Timestamp("2020-01-01")
        day = apply_live_policy(engine, calendar[1], calendar)
        assert day.due == [calendar[1]] and day.working == {calendar[1]: 0}

    def test_k2_anchor_not_in_calendar_not_due(self):
        """K>1 锚定日不在日历 ⇒ 排期表为空 ⇒ 非调仓日（旧链 due 空 ⇒ 非调仓日）。"""
        engine = _StubEngine(6, 2)
        calendar = _calendar(20)
        init_live_anchor(engine, calendar[0], calendar)
        anchor = engine.live_anchor_state
        anchor.last_rebalance_date = calendar[0]
        anchor.last_scheduled_rebalance_date = calendar[0]
        anchor.anchor_date = pd.Timestamp("2020-01-01")  # 锚不在日历
        day = apply_live_policy(engine, calendar[1], calendar)
        assert day.due == [] and day.working == {}

    def test_calendar_extension_keeps_schedule(self):
        """日历同前缀延长：恢复后以延长日历续跑，排期自动扩展（R2-P4-T1-02）。"""
        engine = _StubEngine(6, 2)
        calendar_short = _calendar(12)
        init_live_anchor(engine, calendar_short[0], calendar_short)
        day0 = apply_live_policy(engine, calendar_short[0], calendar_short)
        engine._signal_dates = day0.working
        _mark_t0_success(engine, calendar_short[0], tranche_idx=0)
        settle_live_day(engine, calendar_short[0], day0)
        # 日历延长至 26 日，配置不变 ⇒ 排期自动扩展：重建表覆盖短日历（12 日）
        # 耗尽段之后（含 idx18/24 等）——旧持久表形态在 idx11 后停止调仓
        calendar_long = _calendar(26)
        day18 = apply_live_policy(engine, calendar_long[18], calendar_long)
        assert max(day18.effective) > calendar_short[-1], "重建表未覆盖延长段"
        assert day18.due, "漏批窗口应为非空（中间批次未履行）"
        # 当日执行最早到期批次（漏批追赶）
        assert day18.working[calendar_long[18]] == day18.effective[day18.due[0]]


class TestBlobProjectionAndGuards:
    def test_anchor_blob_roundtrip_and_failfast(self):
        anchor = LiveAnchorState(
            anchor_date=pd.Timestamp("2026-01-05"),
            rebalance_freq=5,
            stagger_tranches=1,
            last_scheduled_rebalance_date=pd.Timestamp("2026-01-12"),
            last_rebalance_date=pd.Timestamp("2026-01-12"),
        )
        restored = LiveAnchorState.from_dict(anchor.to_dict())
        assert restored == anchor
        blob = anchor.to_dict()
        blob.pop("last_rebalance_date")
        with pytest.raises(LiveScheduleError):
            LiveAnchorState.from_dict(blob)
        blob2 = anchor.to_dict()
        blob2["extra"] = 1
        with pytest.raises(LiveScheduleError):
            LiveAnchorState.from_dict(blob2)

    def test_policy_and_settle_require_anchor(self):
        engine = _StubEngine(5, 1)  # live_anchor_state = None
        calendar = _calendar()
        with pytest.raises(LiveScheduleError):
            apply_live_policy(engine, calendar[0], calendar)
        with pytest.raises(LiveScheduleError):
            settle_live_day(engine, calendar[0], None)  # 锚检查先于 day 使用
