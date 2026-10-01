# -*- coding: utf-8 -*-
"""v2 信号层尺子（P5a-1 交付物④）单元测试：合成数据，不依赖真实批次。

覆盖：同日配对、分块自举区间存在、噪声带判据、topk 缺失降级。
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.v2.evidence.signal_ruler import (
    NOISE_BAND_RETURN_BPS,
    RELATIVE_GATE,
    ruler_verdict,
)


def _mk_topk(days, seed, ret_mean, topk=20):
    """合成一臂 topk_detail（每日 topk 只）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        for r in range(1, topk + 1):
            rows.append({
                "trade_date": d, "topk": topk, "rank": r,
                "ts_code": f"600{r:04d}.SH",
                "true_return": rng.normal(ret_mean, 0.02),
                "pred_score": 0.5, "score_column": "final_score",
                "ml_score": 0.5, "risk_score": 0.0, "final_score": 0.5,
            })
    return pd.DataFrame(rows)


class TestSignalRuler:
    def test_identical_arms_delta_zero(self):
        """同一臂 ⇒ Δ=0、区间跨 0、相对降幅 0、过尺（不触发阻断）。"""
        days = pd.date_range("2024-01-01", periods=40).strftime("%Y%m%d")
        a = _mk_topk(days, seed=1, ret_mean=0.001)
        v = ruler_verdict(a, a.copy(), topk=20, block_days=20, seed=1)
        assert v.delta_return_bps == pytest.approx(0.0, abs=1e-6)
        assert v.relative_drop == pytest.approx(0.0, abs=1e-9)
        assert v.passes is True
        assert v.n_signal_days == 40

    def test_better_arm_positive_delta(self):
        """臂明显更优（+5bps/日）⇒ Δ收益为正。"""
        days = pd.date_range("2024-01-01", periods=60).strftime("%Y%m%d")
        base = _mk_topk(days, seed=2, ret_mean=0.001)
        arm = _mk_topk(days, seed=2, ret_mean=0.0015)  # 同 seed 同噪声，仅均值高 0.05pp/日
        v = ruler_verdict(base, arm, topk=20, block_days=20, seed=3)
        assert v.delta_return_bps > 0
        assert v.relative_drop < 0  # 负 = 变好
        assert v.n_signal_days == 60

    def test_no_common_days_raises(self):
        """两臂无共同信号日 ⇒ 报错。"""
        base = _mk_topk(pd.date_range("2024-01-01", periods=10).strftime("%Y%m%d"), seed=1, ret_mean=0.001)
        arm = _mk_topk(pd.date_range("2024-03-01", periods=10).strftime("%Y%m%d"), seed=1, ret_mean=0.001)
        with pytest.raises(ValueError, match="无共同信号日"):
            ruler_verdict(base, arm, topk=20)

    def test_noise_band_gate(self):
        """相对降幅 ≥9% ⇒ 不过尺（阻断）。"""
        days = pd.date_range("2024-01-01", periods=60).strftime("%Y%m%d")
        base = _mk_topk(days, seed=4, ret_mean=0.005)  # 基线强
        arm = _mk_topk(days, seed=5, ret_mean=0.001)   # 臂弱（相对降幅大）
        v = ruler_verdict(base, arm, topk=20, block_days=20, seed=6)
        assert v.relative_drop > RELATIVE_GATE
        assert v.passes is False

    def test_ci_present(self):
        """95% 区间存在且为 (low, high) 有序对。"""
        days = pd.date_range("2024-01-01", periods=30).strftime("%Y%m%d")
        a = _mk_topk(days, seed=7, ret_mean=0.001)
        b = _mk_topk(days, seed=8, ret_mean=0.0012)
        v = ruler_verdict(a, b, topk=20, block_days=15, seed=9)
        assert len(v.delta_return_ci95) == 2
        assert v.delta_return_ci95[0] <= v.delta_return_ci95[1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
