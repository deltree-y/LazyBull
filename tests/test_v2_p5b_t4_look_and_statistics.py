# -*- coding: utf-8 -*-
"""P5b T4 look 映射与 block 统计量（设计 v4 §5.1/§5.2 + PSD 预登记）。

覆盖：H→look 映射表（含 H=60/80/84 的 K=2/拒绝）、block 构造（不截断）、
γ0/γ1 与 J−1 分母的手工公式逐值核对（J=6 最小样本）、V≤0/非有限/block 不足的
fail-closed（不否决、不产生有利结论）、family 否决事件。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.lazybull.v2.hosts.testbed.referee import (  # noqa: E402
    DELTA_LOG_DAILY,
    MIN_BLOCKS_PER_LOOK,
    MIN_LOOKS,
    REGISTERED_STATISTICS,
    STATISTIC_DESIGN_V4,
    STATISTIC_PSD,
    LookPlanError,
    StatisticError,
    block_means,
    make_look_plan,
    score_paths,
    veto_flags,
    variance_gamma0_plus_2gamma1,
    variance_psd_lag1,
)
from src.lazybull.v2.hosts.testbed.referee.statistics import (  # noqa: E402
    annualized_log_return,
    delta_annualized_ok,
)

GOLDEN_BLOCKS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_look_plan_h20_matches_design():
    plan = make_look_plan(20)
    assert list(plan.j_looks) == [6, 12, 18, 25]
    assert list(plan.look_days) == [120, 240, 360, 500]
    assert plan.k == 4
    assert plan.j_max == 25


def test_look_plan_h60_has_two_looks():
    plan = make_look_plan(60)
    assert list(plan.j_looks) == [6, 8]
    assert list(plan.look_days) == [360, 480]
    assert plan.k == 2


@pytest.mark.parametrize("h", [80, 84])
def test_look_plan_rejects_insufficient_looks(h):
    with pytest.raises(LookPlanError):
        make_look_plan(h)


def test_look_plan_h80_message_reports_k1():
    with pytest.raises(LookPlanError) as exc:
        make_look_plan(80)
    assert "K=1" in str(exc.value)


def test_look_plan_h84_message_reports_no_look():
    with pytest.raises(LookPlanError) as exc:
        make_look_plan(84)
    assert "K=0" in str(exc.value)


@pytest.mark.parametrize(
    ("h", "blocks", "look_days"),
    [
        (5, [24, 48, 72, 100], [120, 240, 360, 500]),
        (10, [12, 24, 36, 50], [120, 240, 360, 500]),
        (21, [6, 12, 18, 23], [126, 252, 378, 483]),
    ],
)
def test_look_plan_matches_t05b_table(h, blocks, look_days):
    plan = make_look_plan(h)
    assert list(plan.j_looks) == blocks
    assert list(plan.look_days) == look_days


def test_look_plan_rejects_non_positive_h():
    with pytest.raises(LookPlanError):
        make_look_plan(0)


def test_look_plan_rejects_cap_smaller_than_h():
    with pytest.raises(LookPlanError):
        make_look_plan(600, n_max=503)


def test_look_plan_constants_are_frozen():
    assert MIN_BLOCKS_PER_LOOK == 6
    assert MIN_LOOKS == 2
    assert make_look_plan(20).as_dict()["min_blocks_per_look"] == 6


def test_block_means_golden():
    path = np.arange(1.0, 13.0)[None, :]  # 1..12
    means = block_means(path, 6, 2)
    assert means.shape == (1, 6)
    assert np.allclose(means[0], [1.5, 3.5, 5.5, 7.5, 9.5, 11.5])


def test_block_means_rejects_short_path_no_truncation():
    path = np.zeros((2, 119))
    with pytest.raises(StatisticError):
        block_means(path, 6, 20)


def test_block_means_rejects_non_matrix():
    with pytest.raises(StatisticError):
        block_means(np.zeros(12), 6, 2)


def test_psd_variance_equals_gamma0_plus_gamma1():
    centered = GOLDEN_BLOCKS - GOLDEN_BLOCKS.mean()
    gamma0 = float(np.sum(centered**2) / 5)
    gamma1 = float(np.sum(centered[1:] * centered[:-1]) / 5)
    assert gamma0 == pytest.approx(3.5)
    assert gamma1 == pytest.approx(1.75)
    value = variance_psd_lag1(centered[None, :])
    assert float(value[0]) == pytest.approx(gamma0 + gamma1, rel=0, abs=1e-12)


def test_legacy_variance_equals_gamma0_plus_2gamma1():
    centered = GOLDEN_BLOCKS - GOLDEN_BLOCKS.mean()
    value = variance_gamma0_plus_2gamma1(centered[None, :])
    assert float(value[0]) == pytest.approx(7.0, rel=0, abs=1e-12)


def test_legacy_variance_can_be_negative():
    centered = np.array([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
    assert float(variance_psd_lag1(centered)[0]) > 0
    assert float(variance_gamma0_plus_2gamma1(centered)[0]) < 0


def test_registered_statistics_is_psd_only():
    assert REGISTERED_STATISTICS == (STATISTIC_PSD,)
    assert STATISTIC_DESIGN_V4 not in REGISTERED_STATISTICS


def test_score_paths_golden_six_blocks_manual():
    plan = make_look_plan(20)
    path = np.zeros((1, 500))
    path[0, :120] = np.repeat(GOLDEN_BLOCKS, 20)
    path[0, 120:] = 0.0
    result = score_paths(path, plan, statistic=STATISTIC_PSD)
    bar = GOLDEN_BLOCKS.mean()
    variance = 5.25
    expected = (bar + DELTA_LOG_DAILY) / math.sqrt(variance / 6)
    assert float(result.values[0, 0]) == pytest.approx(expected, rel=0, abs=1e-12)
    legacy = score_paths(path, plan, statistic=STATISTIC_DESIGN_V4)
    assert float(legacy.values[0, 0]) == pytest.approx(bar / math.sqrt(7.0 / 6), rel=0, abs=1e-12)


def test_score_paths_constant_zero_path_fails_closed():
    plan = make_look_plan(20)
    path = np.zeros((3, 500))
    result = score_paths(path, plan, statistic=STATISTIC_PSD)
    assert np.isnan(result.values).all()
    assert result.reason_counts(0) == {"non_positive_variance": 3}
    # fail-closed：不产生否决，也不产生有利结论
    assert not veto_flags(result.values, [0.0, 0.0, 0.0, 0.0]).any()
    assert result.invalid_rate(0) == 1.0


def test_constant_nonzero_path_numeric_edge_is_documented():
    """常数非零路径的 block 均值只差浮点舍入 ⇒ V 是极小正数，T 极大但不触发否决。

    这是**已登记**的数值边界（设计只允许 V<=0/非有限/block 不足判无效，禁止 eps 截断）：
    单侧下尾否决 + 下分位，因此极大正值既不影响 ``c_k`` 也不产生误否决。
    """
    plan = make_look_plan(20)
    path = np.full((2, 500), 0.001)
    result = score_paths(path, plan, statistic=STATISTIC_PSD)
    look_one = result.values[:, 1]
    # 常数路径：均值恰为 0（差分后），V 为舍入级正数 ⇒ T 极大；只要有限就不触发否决
    assert np.all(np.isfinite(look_one))
    assert np.all(look_one > 1e6)
    assert not veto_flags(result.values, [4.0, 4.0, 4.0, 4.0]).any()


def test_score_paths_non_finite_path_fails_closed():
    plan = make_look_plan(20)
    path = np.zeros((1, 500))
    path[0, 10] = np.nan
    result = score_paths(path, plan)
    assert np.isnan(result.values).all()
    assert result.reason_counts(0) == {"non_finite": 1}


def test_score_paths_insufficient_blocks_fails_closed():
    plan = make_look_plan(20)
    path = np.zeros((1, 100))  # 只够 J=5，全部 look 不足
    result = score_paths(path, plan)
    assert np.isnan(result.values).all()
    assert result.reason_counts(3) == {"insufficient_blocks": 1}
    assert result.invalid_rate(0) == 1.0


def test_score_paths_unknown_statistic_rejected():
    plan = make_look_plan(20)
    with pytest.raises(StatisticError):
        score_paths(np.zeros((1, 500)), plan, statistic="bartlett_fixed_bandwidth")


def test_veto_flags_any_look_with_invalid_excluded():
    values = np.array(
        [
            [10.0, 10.0, 10.0, 10.0],
            [-10.0, 10.0, 10.0, 10.0],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    flags = veto_flags(values, [3.0, 3.0, 3.0, 3.0])
    assert flags.tolist() == [False, True, False]


def test_veto_flags_rejects_threshold_length_mismatch():
    with pytest.raises(StatisticError):
        veto_flags(np.zeros((2, 4)), [1.0, 1.0])


def test_veto_flags_rejects_non_finite_threshold():
    with pytest.raises(StatisticError):
        veto_flags(np.zeros((2, 4)), [1.0, float("nan"), 1.0, 1.0])


def test_delta_matches_annualized_log_conversion():
    assert delta_annualized_ok(DELTA_LOG_DAILY)
    assert annualized_log_return(DELTA_LOG_DAILY) == pytest.approx(
        math.log(1.05), rel=1e-9, abs=0.0
    )
    assert math.exp(annualized_log_return(DELTA_LOG_DAILY)) - 1.0 == pytest.approx(0.05, rel=1e-9)
    assert not delta_annualized_ok(DELTA_LOG_DAILY * 1.01)


def test_score_block_shape_and_validity_matrix():
    plan = make_look_plan(20)
    paths = np.full((4, 500), 0.0003)
    paths[0, 5] = -0.01  # 仅第 0 行出现 block 间差异
    result = score_paths(paths, plan)
    assert result.values.shape == (4, 4)
    assert result.valid(0).sum() == 1
    assert result.reason_counts(3) == {"non_positive_variance": 3}
