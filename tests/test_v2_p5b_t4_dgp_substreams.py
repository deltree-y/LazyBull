# -*- coding: utf-8 -*-
"""P5b T4 场景 DGP、circular block 重采样与命名 seed 子流（设计 v4 §5.3/§5.5）。

覆盖：四类场景齐全（禁删困难场景）、block 重采样环回与不截断、MA/AR/regime 的统计特征、
真实块重采样只取池内值、子流可复现且训练/holdout 互斥、分块生成与单批逐值一致、
配对收益差 ``d_t`` 的手工核对与非法输入拒绝。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p5b_t4_env as env  # noqa: E402

from src.lazybull.v2.hosts.testbed.referee import (  # noqa: E402
    SCENE_NAMES,
    SCENE_REQUIRED,
    DgpParams,
    SceneError,
    SceneRegistry,
    SubstreamError,
    assert_distinct_streams,
    build_scene_registry,
    circular_block_resample,
    generate_scores,
    make_look_plan,
    paired_log_diff,
    stream_codes,
    substream,
)

PARAMS = env.dgp_params()


def _registry(pool_role: str = "train", params: DgpParams = PARAMS) -> SceneRegistry:
    pool = np.linspace(-0.02, 0.02, 200)
    return build_scene_registry(pool, params=params, pool_role=pool_role)


def test_scene_set_is_frozen_four():
    assert SCENE_NAMES == ("ma20", "ar1_06", "regime_hetero", "real_seed_block")
    assert SCENE_REQUIRED == SCENE_NAMES


def test_registry_requires_all_scenes():
    functions = {"ma20": lambda rng, n: np.zeros((n, 5))}
    with pytest.raises(SceneError):
        SceneRegistry(pool_role="train", functions=functions, params=PARAMS)


def test_registry_rejects_extra_scene():
    base = _registry()
    functions = dict(base.functions)
    functions["extra"] = lambda rng, n: np.zeros((n, 5))
    with pytest.raises(SceneError):
        SceneRegistry(pool_role="train", functions=functions, params=PARAMS)


def test_registry_rejects_unknown_pool_role():
    with pytest.raises(SceneError):
        SceneRegistry(pool_role="validation", functions=_registry().functions, params=PARAMS)


def test_registry_requires_pool():
    with pytest.raises(SceneError):
        build_scene_registry(None, params=PARAMS, pool_role="train")


def test_registry_build_rejects_unknown_scene():
    with pytest.raises(SceneError):
        _registry().build("unknown_scene", np.random.default_rng(0), 4)


def test_circular_block_resample_rejects_short_pool():
    with pytest.raises(SceneError):
        circular_block_resample(
            np.random.default_rng(0), 3, np.arange(5.0), days=7, block_length=10
        )


def test_circular_block_resample_wraps_without_truncation():
    pool = np.arange(5.0)
    paths = circular_block_resample(np.random.default_rng(7), 4, pool, days=7, block_length=3)
    assert paths.shape == (4, 7)
    assert set(np.unique(paths)).issubset(set(pool.tolist()))
    # 块内必须逐步 +1（末尾环回）：位置非块首 ⇒ 值 = 前值 +1 mod 池长
    for row in paths:
        for position in range(1, row.size):
            if position % 3 == 0:
                continue
            assert row[position] == pytest.approx((row[position - 1] + 1.0) % 5.0)


def test_circular_block_resample_is_deterministic():
    first = circular_block_resample(
        np.random.default_rng(3), 2, np.arange(50.0), days=20, block_length=5
    )
    second = circular_block_resample(
        np.random.default_rng(3), 2, np.arange(50.0), days=20, block_length=5
    )
    assert np.array_equal(first, second)


def test_gen_ma_shape_and_mean():
    paths = _registry().build("ma20", np.random.default_rng(11), 400)
    assert paths.shape == (400, PARAMS.path_days)
    assert np.isfinite(paths).all()
    # MA(20) 下整体均值的抽样 sd ≈ sd*sqrt(H/N) ≈ 3.7e-5，取 4 sigma 容差
    assert abs(float(paths.mean())) < 1.5e-4


def test_gen_ar1_has_rho_autocorrelation():
    paths = _registry().build("ar1_06", np.random.default_rng(12), 300)
    assert paths.shape == (300, PARAMS.path_days)
    lag1 = float(np.mean([np.corrcoef(row[:-1], row[1:])[0, 1] for row in paths[:120]]))
    assert 0.5 < lag1 < 0.7


def test_gen_regime_is_heteroskedastic():
    paths = _registry().build("regime_hetero", np.random.default_rng(13), 400)
    first = float(paths[:, :100].std())
    second = float(paths[:, 100:200].std())
    assert first < second
    assert 2.0 < second / first < 4.0


def test_gen_real_block_only_uses_pool_values():
    pool = np.linspace(-0.03, 0.03, 137)
    registry = build_scene_registry(pool, params=PARAMS, pool_role="holdout")
    paths = registry.build("real_seed_block", np.random.default_rng(21), 20)
    assert paths.shape == (20, PARAMS.path_days)
    assert set(np.unique(paths)).issubset(set(np.unique(pool).tolist()))


def test_dgp_params_reject_block_length_mismatch():
    doc = {
        "calibration": {
            "dgp": {
                "sd_daily": 0.0037,
                "path_days": 500,
                "ar_rho": 0.6,
                "regime_block_length": 100,
                "regime_multipliers": [0.5, 1.5],
                "block_length": 21,
            },
            "block_resample": {
                "method": "circular",
                "block_length": 21,
                "cross_pool_sampling_forbidden": True,
            },
        }
    }
    with pytest.raises(SceneError):
        DgpParams.from_manifest(doc, h=20)


def test_dgp_params_reject_non_circular_method():
    doc = {
        "calibration": {
            "dgp": {
                "sd_daily": 0.0037,
                "path_days": 500,
                "ar_rho": 0.6,
                "regime_block_length": 100,
                "regime_multipliers": [0.5, 1.5],
                "block_length": 20,
            },
            "block_resample": {
                "method": "moving_block",
                "block_length": 20,
                "cross_pool_sampling_forbidden": True,
            },
        }
    }
    with pytest.raises(SceneError):
        DgpParams.from_manifest(doc, h=20)


def test_substream_same_params_same_paths():
    first = substream(11, "train", "ma20", null="mu=0", prefix="p")
    second = substream(11, "train", "ma20", null="mu=0", prefix="p")
    assert np.array_equal(first.generator.normal(size=6), second.generator.normal(size=6))
    assert first.realization_id == second.realization_id


def test_substream_train_and_holdout_are_disjoint():
    train = substream(11, "train", "ma20", null="mu=0")
    holdout = substream(11, "holdout", "ma20", null="mu=0")
    assert train.codes != holdout.codes
    assert not np.array_equal(train.generator.normal(size=6), holdout.generator.normal(size=6))


def test_substream_effect_levels_are_disjoint():
    first = substream(11, "effect", "ma20", level=0.5)
    second = substream(11, "effect", "ma20", level=1.0)
    assert first.codes != second.codes
    assert first.realization_id != second.realization_id


def test_stream_codes_follow_t05b_stage_codes():
    assert stream_codes(5, "train", "ma20", "mu=0")[1] == 1
    assert stream_codes(5, "holdout", "ma20", "mu=0")[1] == 2
    assert stream_codes(5, "effect", "ma20", "", 1.5)[1] == 3
    assert stream_codes(5, "effect", "ma20", "", 1.5)[4] == 1500
    assert stream_codes(5, "train", "ar1_06", "mu=-delta")[3] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": "validation", "scene": "ma20"},
        {"stage": "train", "scene": "unknown"},
        {"stage": "train", "scene": "ma20", "null": "mu=+delta"},
        {"stage": "effect", "scene": "ma20", "level": float("inf")},
    ],
)
def test_stream_codes_reject_unknown_inputs(kwargs):
    with pytest.raises(SubstreamError):
        stream_codes(5, kwargs.pop("stage"), kwargs.pop("scene"), **kwargs)


def test_generate_scores_chunk_equivalence():
    plan = make_look_plan(20)
    registry = _registry()
    single, rid_single = generate_scores(
        registry,
        "ma20",
        seed=31,
        stage="train",
        null="mu=0",
        mu=0.0,
        count=60,
        plan=plan,
        prefix="p",
        chunk=None,
    )
    chunked, rid_chunked = generate_scores(
        registry,
        "ma20",
        seed=31,
        stage="train",
        null="mu=0",
        mu=0.0,
        count=60,
        plan=plan,
        prefix="p",
        chunk=17,
    )
    assert rid_single == rid_chunked
    assert np.array_equal(single.values, chunked.values, equal_nan=True)


@pytest.mark.parametrize("scene", ["real_seed_block", "ar1_06", "regime_hetero"])
def test_generate_scores_chunk_equivalence_all_samplers(scene):
    """分块与单批逐值等价：覆盖拒绝采样（rng.integers）与 AR 预热两类抽样。

    生产 holdout 每单元 10000 > chunk=2000，等价性是"分块只影响内存不影响读数"的前提。
    """
    plan = make_look_plan(20)
    registry = _registry()
    common = dict(
        seed=41,
        stage="holdout",
        null="mu=0",
        mu=0.0,
        count=45,
        plan=plan,
        prefix="p",
    )
    single, rid_single = generate_scores(registry, scene, chunk=None, **common)
    chunked, rid_chunked = generate_scores(registry, scene, chunk=11, **common)
    assert rid_single == rid_chunked
    assert np.array_equal(single.values, chunked.values, equal_nan=True)
    for index in range(plan.k):
        assert single.invalid_rate(index) == chunked.invalid_rate(index)
        assert single.reason_counts(index) == chunked.reason_counts(index)


def test_streams_are_mutually_exclusive_across_stages_and_scenes():
    """种子子流互斥不依赖调用顺序：3 阶段 × 4 场景全组合两两不同。"""
    pairs = tuple(
        (stage, scene)
        for stage in ("train", "holdout", "effect")
        for scene in ("ma20", "ar1_06", "regime_hetero", "real_seed_block")
    )
    assert assert_distinct_streams(20261103, pairs, prefix="p5b_psd_r3") == []
    codes = {substream(20261103, stage, scene, prefix="p5b_psd_r3").codes for stage, scene in pairs}
    assert len(codes) == len(pairs)


def test_generate_scores_rejects_zero_paths():
    with pytest.raises(Exception):
        generate_scores(
            _registry(),
            "ma20",
            seed=31,
            stage="train",
            null="mu=0",
            mu=0.0,
            count=0,
            plan=make_look_plan(20),
        )


def test_paired_log_diff_golden():
    nav_a = [1.0, 1.10, 1.21]
    nav_b = [1.0, 1.05, 1.10]
    result = paired_log_diff(nav_a, nav_b)
    assert result.shape == (2,)
    assert result[0] == pytest.approx(np.log(1.10) - np.log(1.05), rel=0, abs=1e-15)
    assert result[1] == pytest.approx(np.log(1.21 / 1.10) - np.log(1.10 / 1.05), rel=0, abs=1e-15)


@pytest.mark.parametrize(
    "first, second",
    [
        ([1.0], [1.0]),
        ([1.0, 1.1], [1.0]),
        ([1.0, 0.0], [1.0, 1.0]),
        ([1.0, float("nan")], [1.0, 1.0]),
    ],
)
def test_paired_log_diff_rejects_invalid(first, second):
    with pytest.raises(SceneError):
        paired_log_diff(first, second)
