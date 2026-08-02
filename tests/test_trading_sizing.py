"""trading.sizing 共享仓位规模逻辑测试"""

import numpy as np
import pytest

from src.lazybull.trading.sizing import (
    compute_kelly_weights,
    compute_lot_shares,
    compute_min_buy_value_threshold,
    estimate_variance_from_prices,
)


class TestComputeLotShares:
    """整手买入股数计算测试。"""

    def test_rounds_down_to_default_lot(self):
        assert compute_lot_shares(7_000, 33.33) == 200

    def test_supports_custom_lot_size(self):
        assert compute_lot_shares(1_050, 10, lot_size=10) == 100

    @pytest.mark.parametrize(
        "budget,price,lot_size",
        [(0, 10, 100), (1_000, 0, 100), (1_000, 10, 0), (None, None, None)],
    )
    def test_invalid_input_returns_zero(self, budget, price, lot_size):
        assert compute_lot_shares(budget, price, lot_size) == 0


class TestEstimateVarianceFromPrices:
    """方差估计口径测试"""

    def test_normal_prices_return_variance(self):
        """正常价格序列返回对数收益率方差"""
        prices = np.array([10.0 * (1.01**i) for i in range(30)])
        var = estimate_variance_from_prices(prices)
        assert var is not None
        # 恒定收益率的方差应接近 0
        assert var == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_prices_return_none(self):
        """价格点不足 10 个返回 None"""
        assert estimate_variance_from_prices(np.array([10.0] * 9)) is None

    def test_non_positive_prices_filtered(self):
        """非正价格与 NaN 被过滤"""
        prices = np.array([10.0, -1.0, np.nan, 0.0] + [10.5 + i * 0.1 for i in range(9)])
        # 过滤后剩 10 个有效价格
        var = estimate_variance_from_prices(prices)
        assert var is not None

    def test_all_invalid_returns_none(self):
        """全部无效价格返回 None"""
        assert estimate_variance_from_prices(np.array([np.nan, -1.0, 0.0])) is None


class TestComputeKellyWeights:
    """Kelly 权重共享实现测试"""

    def test_empty_signals(self):
        weights, fallback = compute_kelly_weights({}, variance_fn=lambda s: None)
        assert weights == {}
        assert fallback == 0

    def test_all_non_positive_scores_equal_weight(self):
        """全部非正分数回退等权"""
        signals = {"A": -0.1, "B": 0.0}
        weights, _ = compute_kelly_weights(signals, variance_fn=lambda s: 0.01)
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(0.5)

    def test_low_volatility_gets_higher_weight(self):
        """相同分数下低波动股票权重更高"""
        signals = {"LOW": 0.5, "HIGH": 0.5}
        variance = {"LOW": 0.0001, "HIGH": 0.01}
        weights, fallback = compute_kelly_weights(signals, variance_fn=lambda s: variance[s])
        assert weights["LOW"] > weights["HIGH"]
        assert fallback == 0
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_variance_fallback_counted(self):
        """无法估计方差的股票计入 fallback 数"""
        signals = {"A": 0.5, "B": 0.3}
        weights, fallback = compute_kelly_weights(
            signals, variance_fn=lambda s: 0.01 if s == "A" else None
        )
        assert fallback == 1
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_half_kelly_closer_to_equal(self):
        """半 Kelly 比全 Kelly 更接近等权"""
        signals = {"A": 0.9, "B": 0.1}
        variance = {"A": 0.0001, "B": 0.01}
        full, _ = compute_kelly_weights(signals, variance_fn=lambda s: variance[s])
        half, _ = compute_kelly_weights(signals, variance_fn=lambda s: variance[s], half=True)
        eq = 0.5
        assert abs(half["A"] - eq) < abs(full["A"] - eq)
        assert sum(half.values()) == pytest.approx(1.0)

    def test_max_leverage_caps_weight(self):
        """单股权重软上限生效（迭代重归一化，显著降低集中度）"""
        signals = {"A": 0.9, "B": 0.1, "C": 0.05}
        variance = {"A": 0.00001, "B": 0.01, "C": 0.01}
        uncapped, _ = compute_kelly_weights(signals, variance_fn=lambda s: variance[s])
        weights, _ = compute_kelly_weights(
            signals, variance_fn=lambda s: variance[s], max_leverage=0.5
        )
        # 最大权重显著低于未限制时的集中度，且总和保持 1.0
        assert max(weights.values()) < max(uncapped.values())
        assert sum(weights.values()) == pytest.approx(1.0)


class TestComputeMinBuyValueThreshold:
    """最小买入市值阈值测试"""

    def test_normal(self):
        # 100万总资产 / 20 目标持仓 * 0.2 = 1万
        assert compute_min_buy_value_threshold(1_000_000, 20, 0.2) == pytest.approx(10_000)

    def test_disabled_when_ratio_zero(self):
        assert compute_min_buy_value_threshold(1_000_000, 20, 0.0) == 0.0

    def test_disabled_when_target_count_zero(self):
        assert compute_min_buy_value_threshold(1_000_000, 0, 0.2) == 0.0

    def test_disabled_when_assets_non_positive(self):
        assert compute_min_buy_value_threshold(0.0, 20, 0.2) == 0.0

    def test_none_inputs_treated_as_disabled(self):
        assert compute_min_buy_value_threshold(None, None, None) == 0.0
