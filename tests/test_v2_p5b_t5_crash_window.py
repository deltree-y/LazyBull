# -*- coding: utf-8 -*-
"""P5b T5：崩盘窗选择（§5.4.3 唯一规则：深度降序 / 最多 3 / 不重叠 / 同深取较早 peak）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

import p5b_t5_env  # noqa: F401
from src.lazybull.v2.hosts.testbed.referee.drawdown import (
    MAX_CRASH_WINDOWS,
    CrashWindow,
    CrashWindowError,
    assert_windows_fixed,
    crash_window_mask,
    find_drawdown_episodes,
    select_crash_windows,
    window_mdd_percent,
    window_section,
)


def _series(segments) -> np.ndarray:
    """按「(天数, 每日 log 收益)」段拼出序列。"""
    if not segments:
        return np.zeros(0)
    return np.concatenate([np.full(int(n), float(r)) for n, r in segments])


def test_episode_window_is_peak_plus_one_to_trough():
    series = _series([(5, 0.01), (10, -0.02), (5, 0.05), (20, 0.0)])
    episodes = find_drawdown_episodes(series)
    assert len(episodes) == 1
    window = episodes[0]
    # running max 在第 5 个收益处（cum 索引 5），trough 在第 15 个收益处（cum 索引 15）
    assert window.peak == 5
    assert window.trough == 15
    assert window.start == 5 and window.stop == 15
    assert window.length == 10
    assert math.isclose(window.depth_pp, (math.exp(-0.2) - 1.0) * 100.0, abs_tol=1e-9)


def test_selection_takes_top3_by_depth_and_never_overlaps():
    series = _series(
        [
            (3, 0.0),
            (5, -0.02),  # −10%
            (5, 0.03),  # 恢复
            (4, -0.05),  # −20%（最深）
            (6, 0.02),
            (3, -0.04),  # −12%
            (10, 0.0),
        ]
    )
    episodes = find_drawdown_episodes(series)
    chosen = select_crash_windows(episodes)
    assert len(chosen) <= MAX_CRASH_WINDOWS
    depths = [item.depth_pp for item in chosen]
    # 返回按 peak 升序；深度名次（rank）按深度降序
    assert [item.peak for item in chosen] == sorted(item.peak for item in chosen)
    assert chosen[0].depth_pp == min(depths) or any(
        item.rank == 0 and item.depth_pp == min(depths) for item in chosen
    )
    for left, right in zip(chosen, chosen[1:]):
        assert right.peak > left.trough


def test_selection_rejects_overlapping_windows():
    """手工构造与最深 episode 嵌套的候选 ⇒ 必须被拒（互不重叠）。"""
    deep = CrashWindow(peak=10, trough=40, depth_log=-0.3, rank=0)
    nested = CrashWindow(peak=20, trough=25, depth_log=-0.25)
    other = CrashWindow(peak=50, trough=60, depth_log=-0.1)
    chosen = select_crash_windows([deep, nested, other])
    assert [item.peak for item in chosen] == [10, 50]
    assert all(item.rank in (0, 1) for item in chosen)


def test_tie_breaks_on_earlier_peak():
    early = CrashWindow(peak=5, trough=15, depth_log=-0.2)
    late = CrashWindow(peak=40, trough=50, depth_log=-0.2)
    chosen = select_crash_windows([late, early])
    assert [item.peak for item in chosen] == [5, 40]
    assert chosen[0].rank == 0 and chosen[1].rank == 1


def test_no_episode_is_not_constructible():
    with pytest.raises(CrashWindowError):
        select_crash_windows(())
    assert find_drawdown_episodes(np.zeros(50)).__len__() == 0


def test_single_episode_is_enough():
    series = _series([(2, 0.01), (8, -0.01), (10, 0.0)])
    chosen = select_crash_windows(find_drawdown_episodes(series))
    assert len(chosen) == 1


def test_unfinished_tail_episode_is_recorded():
    series = _series([(2, 0.02), (10, -0.02)])
    episodes = find_drawdown_episodes(series)
    assert len(episodes) == 1
    assert episodes[0].trough == 12


def test_mask_covers_window_inclusive_of_peak():
    window = CrashWindow(peak=3, trough=8, depth_log=-0.1)
    mask = crash_window_mask(20, [window])
    assert mask[:3].sum() == 0
    assert mask[3:8].all()
    assert mask[8:].sum() == 0
    with pytest.raises(CrashWindowError):
        crash_window_mask(5, [window])


def test_windows_must_match_manifest_frozen_list():
    """运行期必须与冻结值逐字段一致（禁止按实现重选窗口）。"""
    window = CrashWindow(peak=3, trough=8, depth_log=-0.1)
    section = window_section([window], samples=20)
    assert_windows_fixed(section["frozen_list"], [window], samples=20)
    tampered = [dict(section["frozen_list"][0])]
    tampered[0]["trough"] = 7
    with pytest.raises(CrashWindowError):
        assert_windows_fixed(tampered, [window], samples=20)
    with pytest.raises(CrashWindowError):
        assert_windows_fixed(section["frozen_list"], [], samples=20)
    with pytest.raises(CrashWindowError):
        assert_windows_fixed([], [window], samples=20)


def test_window_section_fields_are_frozen_text():
    window = CrashWindow(peak=3, trough=8, depth_log=-0.1)
    section = window_section([window], samples=20)
    assert section["window_samples"] == 20
    assert section["max_windows"] == 3
    assert section["fixed_for_all_realizations"] is True
    assert section["selection_uses_only_b0"] is True
    assert section["full_chain_used_for_selection"] is False
    assert "深度降序" in section["selection_rule"]
    assert "较早" in section["tie_break"]
    assert section["frozen_list"][0]["window_start_ordinal"] == 3
    assert section["frozen_list"][0]["window_stop_ordinal"] == 8


def test_window_mdd_percent_matches_direct_mdd():
    series = _series([(5, 0.0), (10, -0.01), (5, 0.0)])
    assert math.isclose(window_mdd_percent(series), (math.exp(-0.1) - 1.0) * 100.0, abs_tol=1e-9)


def test_synthetic_repo_window_has_episodes(tmp_path):
    """合成的 B0 窗口必须可构造崩盘窗（环境自检）。"""
    env = p5b_t5_env.synthetic_repo(tmp_path)
    window = p5b_t5_env.source_window(env)
    assert window.samples == 500
    episodes = find_drawdown_episodes(window.log_returns)
    assert len(episodes) >= 1
    chosen = select_crash_windows(episodes)
    assert 1 <= len(chosen) <= MAX_CRASH_WINDOWS
    assert window.mdd_percent < 0
