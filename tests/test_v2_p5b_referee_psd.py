import math

import numpy as np
import pytest

from scripts.v2_p5b.referee_psd_calibration import (
    psd_variance,
    t_scores,
    wilson_interval,
)


def test_psd_variance_pathological_sequence_nonnegative():
    # The old gamma0+2*gamma1 is -0.8 for this sequence.
    assert psd_variance(np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])) == pytest.approx(0.2)


def test_psd_variance_is_nonnegative_and_zero_for_constant_blocks():
    rng = np.random.default_rng(7)
    for _ in range(100):
        value = psd_variance(rng.normal(size=10))
        assert math.isfinite(value)
        assert value >= 0.0
    assert psd_variance(np.ones(6)) == pytest.approx(0.0)


def test_t_scores_boundary_centered_and_invalid_zero_variance():
    # Constant block means -> PSD variance zero -> invalid (NaN), no numeric shortcut.
    paths = np.ones((3, 500))
    z = t_scores(paths, h=20, look_blocks=(6, 12, 18, 25), delta=0.0001936117626)
    assert np.isnan(z).all()


def test_t_scores_shape_and_finite_for_random_paths():
    rng = np.random.default_rng(11)
    paths = rng.normal(0, 0.0037, size=(20, 500))
    z = t_scores(paths, h=20, look_blocks=(6, 12, 18, 25), delta=0.0001936117626)
    assert z.shape == (20, 4)
    assert np.isfinite(z).any()


def test_wilson_interval_contains_point_and_is_bounded():
    lo, hi = wilson_interval(50, 200)
    assert 0.0 <= lo < 0.25 < hi <= 1.0


def test_wilson_interval_rejects_empty_sample():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
