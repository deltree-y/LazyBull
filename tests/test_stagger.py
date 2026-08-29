"""分批调仓共享纯函数测试（trading.stagger）"""

import pandas as pd
import pytest

from src.lazybull.trading.stagger import (
    build_tranche_schedule_from_anchor,
    compute_tranche_schedule,
    get_tranche_capital_fraction,
    get_tranche_target_count,
)


# ── get_tranche_target_count ──


class TestGetTrancheTargetCount:
    def test_no_stagger_returns_full(self):
        assert get_tranche_target_count(0, 20, 1) == 20

    def test_even_split(self):
        # 20 / 4 = 5/5/5/5
        assert [get_tranche_target_count(i, 20, 4) for i in range(4)] == [5, 5, 5, 5]

    def test_uneven_split(self):
        # 30 / 4 = 8/8/7/7
        assert [get_tranche_target_count(i, 30, 4) for i in range(4)] == [8, 8, 7, 7]

    def test_remainder_goes_to_earlier_tranches(self):
        # 5 / 3 = 2/2/1
        assert [get_tranche_target_count(i, 5, 3) for i in range(3)] == [2, 2, 1]

    def test_top_n_less_than_k_raises(self):
        with pytest.raises(ValueError, match="不能超过总目标持仓数"):
            get_tranche_target_count(0, 2, 4)

    def test_sum_equals_total(self):
        for total in [5, 10, 20, 30, 33]:
            for k in [2, 3, 4, 5]:
                assert sum(get_tranche_target_count(i, total, k) for i in range(k)) == total


# ── get_tranche_capital_fraction ──


class TestGetTrancheCapitalFraction:
    def test_no_stagger_returns_one(self):
        assert get_tranche_capital_fraction(0, 20, 1) == 1.0

    def test_even_split(self):
        fractions = [get_tranche_capital_fraction(i, 20, 4) for i in range(4)]
        assert fractions == pytest.approx([0.25, 0.25, 0.25, 0.25])

    def test_uneven_split(self):
        # 30/4 → 8/8/7/7 → 8/30, 8/30, 7/30, 7/30
        fractions = [get_tranche_capital_fraction(i, 30, 4) for i in range(4)]
        assert fractions == pytest.approx([8 / 30, 8 / 30, 7 / 30, 7 / 30])

    def test_sum_equals_one(self):
        for total in [10, 20, 30]:
            for k in [2, 3, 4]:
                assert sum(
                    get_tranche_capital_fraction(i, total, k) for i in range(k)
                ) == pytest.approx(1.0)

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="总目标持仓数"):
            get_tranche_capital_fraction(0, 0, 3)


# ── compute_tranche_schedule ──


class TestComputeTrancheSchedule:
    def _make_dates(self, n=60):
        return pd.bdate_range("2026-01-05", periods=n).tolist()

    def test_no_stagger(self):
        dates = self._make_dates(40)
        schedule = compute_tranche_schedule(dates, 20, 1)
        assert len(schedule) == 2  # day 0, day 20
        assert all(v == 0 for v in schedule.values())

    def test_stagger_2_batches(self):
        dates = self._make_dates(40)
        schedule = compute_tranche_schedule(dates, 20, 2)
        # tranche 0: idx 0, 20; tranche 1: idx 10, 30
        assert schedule[dates[0]] == 0
        assert schedule[dates[10]] == 1
        assert schedule[dates[20]] == 0
        assert schedule[dates[30]] == 1

    def test_stagger_3_batches_20_days(self):
        dates = self._make_dates(60)
        schedule = compute_tranche_schedule(dates, 20, 3)
        # offsets: t0=0, t1=7, t2=13
        assert schedule[dates[0]] == 0
        assert schedule[dates[7]] == 1
        assert schedule[dates[13]] == 2
        # next cycle
        assert schedule[dates[20]] == 0
        assert schedule[dates[27]] == 1
        assert schedule[dates[33]] == 2

    def test_invalid_freq_raises(self):
        with pytest.raises(ValueError):
            compute_tranche_schedule(self._make_dates(), 0, 1)

    def test_tranches_cannot_exceed_rebalance_frequency(self):
        with pytest.raises(ValueError, match="不能超过调仓频率"):
            compute_tranche_schedule(self._make_dates(), 2, 4)


# ── build_tranche_schedule_from_anchor ──


class TestBuildTrancheScheduleFromAnchor:
    def _make_dates(self, n=60):
        return [d.strftime("%Y%m%d") for d in pd.bdate_range("2026-01-05", periods=n)]

    def test_no_stagger(self):
        dates = self._make_dates(40)
        schedule = build_tranche_schedule_from_anchor(dates[0], dates, 20, 1)
        assert schedule[dates[0]] == 0
        assert schedule[dates[20]] == 0

    def test_stagger_3_batches(self):
        dates = self._make_dates(60)
        schedule = build_tranche_schedule_from_anchor(dates[0], dates, 20, 3)
        assert schedule[dates[0]] == 0
        assert schedule[dates[7]] == 1
        assert schedule[dates[13]] == 2
        assert schedule[dates[20]] == 0

    def test_anchor_not_in_dates_returns_empty(self):
        dates = self._make_dates(40)
        schedule = build_tranche_schedule_from_anchor("20251231", dates, 20, 3)
        assert schedule == {}

    def test_anchor_in_middle(self):
        dates = self._make_dates(60)
        anchor = dates[10]
        schedule = build_tranche_schedule_from_anchor(anchor, dates, 20, 3)
        # tranche 0 starts at anchor (idx 10)
        assert schedule[dates[10]] == 0
        # tranche 1 offset: (2*1*20+3)//(2*3) = 43//6 = 7 → idx 17
        assert schedule[dates[17]] == 1
        # tranche 2 offset: (2*2*20+3)//(2*3) = 83//6 = 13 → idx 23
        assert schedule[dates[23]] == 2
