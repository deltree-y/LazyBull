# -*- coding: utf-8 -*-
"""P5b T5：命名 seed 子流与 B0 冻结窗口实现（§5.5「路径数/种子」+「基线/噪声」行）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import p5b_t5_env
from src.lazybull.v2.hosts.testbed.referee.drawdown import (
    IMPLEMENTATION_ID,
    NAMESPACE_CODE,
    SamplingError,
    assert_independent_of_income,
    draw_arm_paths,
    draw_noise_paths,
    implementation_ids,
    load_source_window,
    stream_codes,
    substream,
)
from src.lazybull.v2.hosts.testbed.referee.scenes import (
    DgpParams,
    build_scene_registry,
    pool_for_folds,
    load_source_pair,
)


def test_stream_codes_include_namespace_and_are_stable():
    codes = stream_codes(20261103, "train", "ma20")
    assert codes[0] == 20261103
    assert codes[1] == NAMESPACE_CODE
    assert codes == stream_codes(20261103, "train", "ma20")
    assert stream_codes(20261103, "holdout", "ma20") != codes
    assert stream_codes(20261103, "train", "ar1_06") != codes
    assert stream_codes(20261103, "train", "ma20", null="dd_i") != codes
    assert stream_codes(20261103, "effect", "ma20", level=12.0) != stream_codes(
        20261103, "effect", "ma20", level=8.0
    )


def test_streams_are_independent_of_income_referee():
    pairs = tuple(
        (stage, scene)
        for stage in ("train", "holdout", "effect")
        for scene in ("ma20", "ar1_06", "regime_hetero", "real_seed_block")
    )
    assert assert_independent_of_income(20261103, pairs) == []


def test_substream_is_call_order_independent():
    first = substream(20261103, "train", "ma20", null="aa", prefix="p")
    second = substream(20261103, "train", "ma20", null="aa", prefix="p")
    assert first.realization_id == second.realization_id
    assert np.array_equal(first.generator.normal(size=5), second.generator.normal(size=5))


def test_aa_stream_is_shared_across_directions():
    """A/A 与两个方向用同一 ``null='aa'`` 子流 ⇒ 同一组独立路径（方向自检前提）。"""
    first = substream(20261103, "train", "ma20", null="aa")
    second = substream(20261103, "train", "ma20", null="aa")
    assert first.codes == second.codes
    assert stream_codes(20261103, "train", "ma20", null="dd_i") != first.codes


def test_substream_rejects_unknown_stage_or_direction():
    with pytest.raises(SamplingError):
        stream_codes(20261103, "validate", "ma20")
    with pytest.raises(SamplingError):
        stream_codes(20261103, "train", "unknown_scene")
    with pytest.raises(SamplingError):
        stream_codes(20261103, "train", "ma20", null="mu=0")


def test_implementation_ids_block_is_frozen():
    payload = implementation_ids()
    assert payload["implementation_id"] == IMPLEMENTATION_ID
    assert payload["namespace_code"] == NAMESPACE_CODE
    assert payload["independence"]["train_holdout_disjoint"] is True
    assert payload["independence"]["independent_of_income_referee"] is True
    assert payload["independence"]["aa_uses_independent_paths_not_zero_injection"] is True
    assert payload["direction_codes"] == {"aa": 0, "dd_i": 1, "dd_w": 2}


def test_source_window_is_last_500_samples(tmp_path):
    env = p5b_t5_env.synthetic_repo(tmp_path)
    window = load_source_window(env.base_path, window=500)
    assert window.samples == 500
    assert window.total_returns > 500
    assert window.fold_count == p5b_t5_env.FOLDS
    assert window.mdd_percent < 0
    assert window.history["usable_for_calibration"] is False
    assert window.history["window_samples"] == window.total_returns


def test_source_window_rejects_hash_and_mdd_mismatch(tmp_path):
    env = p5b_t5_env.synthetic_repo(tmp_path)
    window = load_source_window(env.base_path, window=500)
    with pytest.raises(Exception):
        load_source_window(env.base_path, expected_sha256="0" * 64, window=500)
    with pytest.raises(Exception):
        load_source_window(env.base_path, window=500, expected_mdd_percent=window.mdd_percent + 1.0)
    with pytest.raises(Exception):
        load_source_window(env.base_path, window=10_000)


def test_source_window_rejects_short_history(tmp_path):
    target = tmp_path / "short.csv"
    frame = pd.DataFrame(
        {
            "date": np.arange(30),
            "nav": np.cumprod(1.0 + np.full(30, 0.001)),
            "split_index": np.zeros(30, dtype=int),
        }
    )
    frame.to_csv(target, index=False)
    with pytest.raises(Exception):
        load_source_window(target, window=500)


def test_draw_arm_paths_requires_matching_days(tmp_path):
    env = p5b_t5_env.synthetic_repo(tmp_path)
    window = load_source_window(env.base_path, window=500)
    pair = load_source_pair(env.base_path, env.noise_path)
    params = DgpParams(
        sd_daily=0.0037,
        path_days=400,  # 与窗口 500 不符
        ar_rho=0.6,
        regime_block_length=100,
        regime_multipliers=(0.5, 1.5),
        block_length=20,
    )
    registry = build_scene_registry(
        pool_for_folds(pair, p5b_t5_env.TRAIN_FOLDS)["centred"],
        params=params,
        pool_role="train",
    )
    with pytest.raises(Exception):
        draw_arm_paths(window, registry, "ma20", np.random.default_rng(1), 3)


def test_draw_arm_paths_is_window_plus_noise(tmp_path):
    env = p5b_t5_env.synthetic_repo(tmp_path)
    window = load_source_window(env.base_path, window=500)
    pair = load_source_pair(env.base_path, env.noise_path)
    params = DgpParams(path_days=500, block_length=20)
    registry = build_scene_registry(
        pool_for_folds(pair, p5b_t5_env.TRAIN_FOLDS)["centred"],
        params=params,
        pool_role="train",
    )
    rng = np.random.default_rng(7)
    arms = draw_arm_paths(window, registry, "regime_hetero", rng, 4)
    noise = draw_noise_paths(registry, "regime_hetero", np.random.default_rng(7), 4)
    assert arms.shape == (4, 500)
    assert np.allclose(arms - noise, window.log_returns[None, :])
