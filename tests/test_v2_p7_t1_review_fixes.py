# -*- coding: utf-8 -*-
"""T1 首轮评审（reviewer1）修复的定向回归：R1/R2/R3/R4/R6/R7/R8/R10/R12。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v2_p7 import event_builder  # noqa: E402
from scripts.v2_p7.t1 import execution, matching, randomization  # noqa: E402


class _Panel:
    """最小面板：含上市状态、行业归属日、送转倍数与不可解析计数。"""

    def __init__(self, n_days: int = 60, n_stocks: int = 6) -> None:
        shape = (n_days, n_stocks)
        self.n_days = n_days
        self.open_u = np.full(shape, 10.0)
        self.open_adj = np.full(shape, 10.0)
        self.close_adj = np.full(shape, 10.0)
        self.pre_close = np.full(shape, 10.0)
        self.high = np.full(shape, 10.1)
        self.low = np.full(shape, 9.9)
        self.total_mv = np.full(shape, 100000.0)
        self.ret20 = np.full(shape, 0.02)
        self.vol20 = np.full(shape, 0.03)
        self.is_st = np.zeros(shape, dtype=bool)
        self.suspended = np.zeros(shape, dtype=bool)
        self.limit_up = np.zeros(shape, dtype=bool)
        self.limit_down = np.zeros(shape, dtype=bool)
        self.announce = np.zeros(shape, dtype=bool)
        self.listed = np.ones(shape, dtype=bool)
        self.industry = np.zeros(n_stocks, dtype=np.int32)
        self.industry_since = np.zeros(n_stocks, dtype=np.int32)
        self.industry_matrix = None
        self.share_mult = np.ones(shape)
        self.share_unknown = np.zeros(shape, dtype=np.int32)
        self.days = [
            f"2020{(index // 28) + 1:02d}{(index % 28) + 1:02d}" for index in range(n_days)
        ]
        self.stocks = [f"00000{index}.SZ" for index in range(n_stocks)]
        self.stock_index = {code: index for index, code in enumerate(self.stocks)}


def _events(panel: _Panel, day_index: int, rows: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_source": ["forecast"] * rows,
            "ts_code": [panel.stocks[index] for index in range(rows)],
            "event_date": ["20200331"] * rows,
            "knowledge_date": [panel.days[day_index]] * rows,
            "revision_key": [f"k{index}" for index in range(rows)],
            "predicate_value": [1.0] * rows,
            "fingerprint": [f"f{index}" for index in range(rows)],
        }
    )


def test_r1_suspended_but_listed_event_stays_in_domain() -> None:
    """停牌（当日无行情行）不得被当作「未上市」剔出意图域。"""
    panel = _Panel()
    day = 5
    panel.open_u[day, 0] = np.nan
    panel.close_adj[day, 0] = np.nan
    panel.announce[day, 0] = True
    table, _, domain = matching.build_groups(
        _events(panel, day),
        panel,
        source="forecast",
        evaluation_start_pos=0,
        horizon=1,
    )
    assert table.size == 1
    assert domain["universe_excluded"]["not_listed_at_signal"] == 0
    # 停牌当日无行情行，但上市事实成立 → 保留意图；协变量可用时照常匹配
    assert table.status.tolist()[0] in (
        matching.STATUS_OK,
        matching.STATUS_EVENT_COVARIATE_MISSING,
    )


def test_r1_pre_listing_event_is_excluded_by_listing_fact() -> None:
    panel = _Panel()
    day = 5
    panel.listed[: day + 1, 0] = False
    panel.close_adj[: day + 1, 0] = np.nan
    table, _, domain = matching.build_groups(
        _events(panel, day),
        panel,
        source="forecast",
        evaluation_start_pos=0,
        horizon=1,
    )
    assert table.size == 0
    assert domain["universe_excluded"]["not_listed_at_signal"] == 1


def test_r2_tail_events_are_right_censored_not_unfilled() -> None:
    """信号日之后放不下完整观察窗的事件 → 右删失，不得记未成交（真实 0）。"""
    panel = _Panel(n_days=15)
    event_day = 14
    table, _, domain = matching.build_groups(
        _events(panel, event_day),
        panel,
        source="forecast",
        evaluation_start_pos=0,
        horizon=10,
    )
    assert table.size == 0
    assert domain["events_right_censored_tail"] == 1
    assert domain["accounting_ok"] is True


def test_r3_missing_adjustment_is_unresolved_not_unfilled() -> None:
    """复权链缺失只否认可解析性，不影响「买不进」的真实 0。"""
    panel = _Panel()
    panel.open_adj[1:, :] = np.nan
    result = execution.simulate_horizon(
        panel,
        np.tile(np.arange(6), (1, 1)),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int16),
        ["matched"],
        horizon=2,
        min_ticket=25588.5363,
        cost={
            "commission_rate": 0.0001954,
            "min_commission": 5.0,
            "stamp_tax": 0.0005,
            "slippage": 0.0005,
        },
        lot_size=100,
    )
    assert result.status[0] == execution.STATUS_UNRESOLVED
    assert result.reason[0] == "return_chain_unresolved"


def test_r4_share_split_updates_sell_shares_and_fees() -> None:
    """送转后卖出股数按真实倍数放大，卖出金额与费用用真实金额。"""
    panel = _Panel()
    entry, exit_ = 1, 3
    panel.open_u[entry:, :] = 10.0
    panel.open_adj[entry:, :] = 10.0
    panel.open_u[exit_:, :] = 5.0
    panel.share_mult[exit_:, :] = 2.0
    result = execution.simulate_horizon(
        panel,
        np.tile(np.arange(6), (1, 1)),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int16),
        ["matched"],
        horizon=2,
        min_ticket=25588.5363,
        cost={
            "commission_rate": 0.0001954,
            "min_commission": 5.0,
            "stamp_tax": 0.0005,
            "slippage": 0.0005,
        },
        lot_size=100,
    )
    assert result.status[0] == execution.STATUS_RESOLVED
    assert int(result.shares[0, 0]) == 2600
    assert int(result.sell_shares[0, 0]) == 5200
    assert result.sell_amount[0, 0] == pytest.approx(5200 * 5.0)
    assert result.gross[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_r8_ties_are_broken_globally_not_inside_truncated_subset() -> None:
    """全部候选同距离时必须按决胜键取前 k 个（不能只看 argpartition 子集）。"""
    covariates = np.zeros((1, 3))
    candidates = np.zeros((30, 3))
    industries = np.zeros(30, dtype=np.int32)
    result = matching.match_one_day(
        covariates,
        candidates,
        np.zeros(1, dtype=np.int32),
        industries,
        candidate_keys=np.arange(30),
    )
    assert result.controls.tolist() == [[0, 1, 2, 3, 4]]


def test_r10_industry_diagnostics_use_same_and_known_pairs() -> None:
    panel = _Panel(n_days=40, n_stocks=8)
    panel.stocks = [f"00000{index}.SZ" for index in range(8)]
    panel.stock_index = {code: index for index, code in enumerate(panel.stocks)}
    panel.industry = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    panel.industry_since = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    panel.industry_since[3] = 30  # 该股在信号日之后才纳入分类 → 未知
    panel.announce[5, 0] = True
    panel.announce[5, 1] = True
    table, _, _ = matching.build_groups(
        _events(panel, 5),
        panel,
        source="forecast",
        evaluation_start_pos=0,
        horizon=1,
    )
    quality = matching.matching_quality(table)
    assert quality["matched_groups"] == 1
    assert quality["industry_known_pair_share"] is not None
    # v2 协议 A（2026-11-11）：行业未知的候选不入对照池（stock3 在信号日尚未纳入分类），
    # 因此入选的 5 个对照必然全部已知；R10 的「不混口径」仍由 same/known 两个指标分别承担。
    assert table.industry_known[0].tolist() == [True, True, True, True, True]
    assert 3 not in table.members[0].tolist()
    assert quality["industry_known_pair_share"] == 1.0


def test_r12_empty_candidate_pool_yields_insufficient_controls() -> None:
    """事件协变量完整但全市场都在公告 → 无对照池时可正常落账，不得抛异常。"""
    panel = _Panel(n_days=40, n_stocks=6)
    day = 5
    panel.announce[day, :] = True
    table, _, _ = matching.build_groups(
        _events(panel, day),
        panel,
        source="forecast",
        evaluation_start_pos=0,
        horizon=1,
    )
    assert table.size == 1
    assert table.status.tolist() == [matching.STATUS_INSUFFICIENT_CONTROLS]


def test_r9_same_day_conflict_resolution_is_order_independent() -> None:
    """同键冲突的决胜不得依赖分区或行序。"""
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20200331", "20200331"],
            "knowledge_date": ["20200430", "20200430"],
            "type": ["预增", "预减"],
            "p_change_min": [10.0, -10.0],
            "p_change_max": [20.0, -20.0],
        }
    )
    forward = event_builder._dedup_same_day(frame.copy(), prefer_updated=False)
    backward = event_builder._dedup_same_day(
        frame.iloc[::-1].reset_index(drop=True), prefer_updated=False
    )
    assert len(forward) == len(backward) == 1
    assert forward["type"].tolist() == backward["type"].tolist()


def test_r6_block_axis_uses_full_trading_calendar() -> None:
    """相同矩阵、不同事件日间隔：完整日历口径下零假设分布必须不同。"""
    matrix = np.array([[1.0, -0.2, -0.2, -0.2, -0.2, -0.2]], dtype=np.float32)
    dense = randomization.null_distribution(
        matrix,
        np.array([0], dtype=np.int64),
        block_h=1,
        implementations=4000,
        seed=7,
        day_span=(0, 90),
    )
    dense_other = randomization.null_distribution(
        matrix,
        np.array([60], dtype=np.int64),
        block_h=1,
        implementations=4000,
        seed=7,
        day_span=(0, 90),
    )
    # 同一块轴下位置不同只影响多重性，不改变分布形状；但块轴长度不同必须改变分布
    short = randomization.null_distribution(
        matrix,
        np.array([0], dtype=np.int64),
        block_h=1,
        implementations=4000,
        seed=7,
        day_span=(0, 1),
    )
    assert dense.dates == 91
    assert short.dates == 2
    assert not np.allclose(np.sort(dense.values[:50]), np.sort(short.values[:50]))
    assert dense.values.shape == dense_other.values.shape


def test_r7_each_sampled_date_instance_permutes_independently() -> None:
    """两日各一组、h=1 时，均值统计量的零假设方差应为 0.1（而非复用标签的 0.15）。"""
    matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32)
    result = randomization.null_distribution(
        matrix,
        np.array([0, 1], dtype=np.int64),
        block_h=1,
        implementations=40000,
        seed=42,
        day_span=(0, 1),
    )
    assert float(np.var(result.values, ddof=0)) == pytest.approx(0.10, abs=0.012)
