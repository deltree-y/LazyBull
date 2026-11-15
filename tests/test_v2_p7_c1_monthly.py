# -*- coding: utf-8 -*-
"""C1 月度分组/残差最小自验（合成数组，不触碰真实数据）。"""

import numpy as np

from scripts.v2_p7.c1.monthly import _reference_groups, _residuals, _size_tiers


def test_size_tiers_full_cross_section_deterministic():
    mv = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 6.0])
    eligible = np.ones(6, dtype=bool)
    stocks = [f"{i:06d}.SZ" for i in range(6)]
    tiers = _size_tiers(mv, eligible, stocks)
    assert tiers.tolist() == [2, 1, 1, 0, 0, 2]


def test_thin_cell_merges_industry():
    # 一个行业 10 只，3/3/4 分档：任一格 <5 ⇒ 全部合为行业组
    l1 = np.zeros(10, dtype=np.int64)
    l2 = np.ones(10, dtype=np.int64)
    tiers = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    eligible = np.ones(10, dtype=bool)
    ref_type, ref_key = _reference_groups(l1, l2, tiers, eligible)
    assert set(ref_type.tolist()) == {2}
    assert len(set(ref_key.tolist())) == 1


def test_residual_excludes_self():
    ret = np.array([0.10, 0.00, 0.20, 0.00])
    ref_type = np.full(4, 2, dtype=np.int64)
    ref_key = np.array([1, 1, 2, 2], dtype=np.int64)
    l1 = np.zeros(4, dtype=np.int64)
    eligible = np.ones(4, dtype=bool)
    resid = _residuals(ret, eligible, ref_type, ref_key, l1)
    assert np.isclose(resid[0], 0.10)
    assert np.isclose(resid[1], -0.10)
    assert np.isclose(resid[2], 0.20)
    assert np.isclose(resid[3], -0.20)
