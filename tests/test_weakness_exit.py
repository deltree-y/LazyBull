"""测试持仓表现弱势退出功能"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.risk.weakness_exit import (
    WeaknessExitConfig,
    WeaknessExitMonitor,
    create_weakness_exit_config_from_dict,
)


class TestWeaknessExitConfig:
    """测试配置创建与验证"""

    def test_default_config(self):
        config = WeaknessExitConfig()
        assert config.enabled is False
        assert config.threshold == 0.6
        assert config.consecutive_days == 3
        assert config.min_holding_days == 5
        assert config.weights == (0.30, 0.25, 0.25, 0.20)

    def test_custom_config(self):
        config = WeaknessExitConfig(
            enabled=True,
            threshold=0.7,
            consecutive_days=4,
            min_holding_days=10,
            weights=(0.40, 0.20, 0.20, 0.20),
        )
        assert config.enabled is True
        assert config.threshold == 0.7
        assert config.consecutive_days == 4
        assert config.min_holding_days == 10

    def test_weight_normalization(self):
        """权重自动归一化"""
        config = WeaknessExitConfig(weights=(1.0, 1.0, 1.0, 1.0))
        w_sum = sum(config.weights)
        assert abs(w_sum - 1.0) < 0.01

    def test_weight_normalization_already_one(self):
        config = WeaknessExitConfig(weights=(0.30, 0.25, 0.25, 0.20))
        w_sum = sum(config.weights)
        assert abs(w_sum - 1.0) < 0.01

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            WeaknessExitConfig(threshold=1.5)

    def test_from_dict(self):
        d = {
            "weakness_exit_enabled": True,
            "weakness_exit_threshold": 0.65,
            "weakness_exit_consecutive_days": 4,
            "weakness_exit_min_holding_days": 7,
            "weakness_exit_weights": "40,20,20,20",
            "weakness_exit_industry_filter": True,
            "weakness_exit_industry_bottom_pct": 0.25,
        }
        config = create_weakness_exit_config_from_dict(d)
        assert config.enabled is True
        assert config.threshold == 0.65
        assert config.consecutive_days == 4
        assert config.industry_filter is True
        assert sum(config.weights) == pytest.approx(1.0, abs=0.02)


class TestPnlRankCalculation:
    """测试持仓收益排名计算"""

    def test_best_stock(self):
        monitor = WeaknessExitMonitor(WeaknessExitConfig())
        profits = {"A": 0.10, "B": 0.05, "C": -0.02, "D": -0.08}
        rank = monitor._calc_pnl_rank("A", profits)
        assert rank == pytest.approx(1.0)  # 最好

    def test_worst_stock(self):
        monitor = WeaknessExitMonitor(WeaknessExitConfig())
        profits = {"A": 0.10, "B": 0.05, "C": -0.02, "D": -0.08}
        rank = monitor._calc_pnl_rank("D", profits)
        assert rank == pytest.approx(0.0)  # 最差

    def test_middle_stock(self):
        monitor = WeaknessExitMonitor(WeaknessExitConfig())
        profits = {"A": 0.10, "B": 0.00, "C": -0.10}
        rank = monitor._calc_pnl_rank("B", profits)
        assert rank == pytest.approx(0.5)

    def test_single_position(self):
        monitor = WeaknessExitMonitor(WeaknessExitConfig())
        profits = {"A": 0.05}
        rank = monitor._calc_pnl_rank("A", profits)
        assert rank == 0.5


class TestDownStreak:
    """测试连续下跌天数计算"""

    def test_no_streak(self):
        prices = pd.Series([10.0, 10.5, 10.3, 10.8, 11.0])
        streak = WeaknessExitMonitor._calc_down_streak(prices, holding_days=5)
        assert streak == 0.0

    def test_three_day_streak(self):
        prices = pd.Series([10.0, 10.5, 10.3, 10.1, 9.8, 9.5, 9.2])
        # 最后3天都是下跌（10.1->9.8->9.5->9.2）
        streak = WeaknessExitMonitor._calc_down_streak(prices, holding_days=7)
        assert streak >= 0.4  # 3/max(7,5) ≈ 0.43

    def test_streak_broken(self):
        prices = pd.Series([10.0, 9.8, 9.5, 10.0, 9.8])
        # 最后一天下跌，但前一日上涨，streak=1
        streak = WeaknessExitMonitor._calc_down_streak(prices, holding_days=5)
        assert streak == pytest.approx(1.0 / 5.0)

    def test_short_series(self):
        prices = pd.Series([10.0])
        streak = WeaknessExitMonitor._calc_down_streak(prices, holding_days=1)
        assert streak == 0.0


class TestDrawdownCalculation:
    """测试回撤深度计算"""

    def test_no_drawdown(self):
        prices = pd.Series([10.0, 10.5, 11.0, 11.5, 12.0])
        dd = WeaknessExitMonitor._calc_drawdown(prices)
        assert dd == 0.0

    def test_drawdown_from_peak(self):
        prices = pd.Series([10.0, 12.0, 11.0, 10.5, 9.6])
        # peak=12.0, current=9.6, dd=(9.6-12)/12=-0.2
        dd = WeaknessExitMonitor._calc_drawdown(prices)
        assert dd == pytest.approx(0.2)

    def test_short_series(self):
        prices = pd.Series([10.0])
        dd = WeaknessExitMonitor._calc_drawdown(prices)
        assert dd == 0.0


class TestRecoveryRatio:
    """测试回升比率计算"""

    def test_full_recovery(self):
        """跌后完全收复"""
        prices = pd.Series([10.0, 8.0, 10.0, 10.5])
        ratio = WeaknessExitMonitor._calc_recovery_ratio(prices)
        assert ratio >= 1.0

    def test_no_recovery(self):
        """持续下跌无反弹"""
        prices = pd.Series([10.0, 9.0, 8.5, 8.0, 7.8])
        ratio = WeaknessExitMonitor._calc_recovery_ratio(prices)
        # 最低点在末尾，无反弹
        assert ratio == 0.0

    def test_partial_recovery(self):
        """跌后小幅反弹"""
        prices = pd.Series([10.0, 8.0, 8.5, 8.3, 8.4])
        ratio = WeaknessExitMonitor._calc_recovery_ratio(prices)
        # 从8.0反弹到8.5，回撤2.0，反弹0.5，比率=0.5/8.0/0.2≈0.31
        assert 0.1 < ratio < 1.0

    def test_short_series(self):
        prices = pd.Series([10.0, 10.5])
        ratio = WeaknessExitMonitor._calc_recovery_ratio(prices)
        assert ratio == 1.0  # 太短，默认为正常


class TestWeaknessScoreEvaluation:
    """测试综合弱势评分与触发"""

    def _make_monitor(self, threshold=0.6, consec=3, min_hold=5):
        config = WeaknessExitConfig(
            enabled=True,
            threshold=threshold,
            consecutive_days=consec,
            min_holding_days=min_hold,
        )
        return WeaknessExitMonitor(config)

    def test_strong_stock_low_score(self):
        """强势股评分低"""
        monitor = self._make_monitor()
        # 持续上涨的股票
        prices = pd.Series([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])
        profits = {"A": 0.30, "B": 0.10, "C": -0.05, "D": -0.10}
        score, breakdown = monitor.evaluate(
            stock="A",
            price_series=prices,
            all_positions_profit=profits,
            holding_days=7,
        )
        assert score < 0.5

    def test_weak_stock_high_score(self):
        """弱势股评分高"""
        monitor = self._make_monitor()
        # 持续下跌
        prices = pd.Series([10.0, 9.8, 9.5, 9.3, 9.0, 8.8, 8.5, 8.2])
        profits = {"A": 0.10, "B": 0.05, "C": -0.02, "D": -0.18}
        score, breakdown = monitor.evaluate(
            stock="D",
            price_series=prices,
            all_positions_profit=profits,
            holding_days=8,
        )
        assert score > 0.5

    def test_min_holding_days_guard(self):
        """持有不足最低天数时仍然计算评分，仅由调用方决定是否触发"""
        monitor = self._make_monitor(min_hold=5)
        prices = pd.Series([10.0, 9.5, 9.0])
        profits = {"A": -0.10}
        score, breakdown = monitor.evaluate(
            stock="A",
            price_series=prices,
            all_positions_profit=profits,
            holding_days=3,
        )
        # 评分仍正常计算（不再返回 {"skip": "min_holding"}）
        assert isinstance(score, float)
        assert "total" in breakdown

    def test_industry_filter_boost(self):
        """弱势行业叠加加分"""
        config = WeaknessExitConfig(
            enabled=True,
            threshold=0.6,
            industry_filter=True,
            industry_bottom_pct=0.5,
        )
        monitor = WeaknessExitMonitor(config)
        prices = pd.Series([10.0, 9.8, 9.5, 9.3, 9.0, 8.8])
        profits = {"A": 0.05, "B": -0.12}
        score_no_filter, _ = monitor.evaluate(
            stock="B",
            price_series=prices,
            all_positions_profit=profits,
            holding_days=6,
            industry_rank=0.8,  # 强行业，不应加分
        )
        score_with_filter, _ = monitor.evaluate(
            stock="B",
            price_series=prices,
            all_positions_profit=profits,
            holding_days=6,
            industry_rank=0.2,  # 弱行业，应加分
        )
        # 弱行业时评分更高
        assert score_with_filter >= score_no_filter


class TestConsecutiveWeakDays:
    """测试连续弱势天数跟踪"""

    def test_consecutive_accumulates(self):
        monitor = WeaknessExitMonitor(WeaknessExitConfig(threshold=0.6, consecutive_days=3))
        c1 = monitor.update("TEST", "20260301", 0.7)  # 触发
        assert c1 == 1
        c2 = monitor.update("TEST", "20260302", 0.8)  # 再次触发
        assert c2 == 2
        c3 = monitor.update("TEST", "20260303", 0.65)  # 第三次触发
        assert c3 == 3

    def test_resets_on_good_day(self):
        monitor = WeaknessExitMonitor(WeaknessExitConfig(threshold=0.6))
        monitor.update("TEST", "20260301", 0.7)  # 弱势
        monitor.update("TEST", "20260302", 0.7)  # 弱势
        c = monitor.update("TEST", "20260303", 0.3)  # 恢复
        assert c == 0

    def test_get_consecutive_days(self):
        monitor = WeaknessExitMonitor(WeaknessExitConfig(threshold=0.6))
        monitor.update("TEST", "20260301", 0.7)
        monitor.update("TEST", "20260302", 0.7)
        assert monitor.get_consecutive_days("TEST") == 2
        assert monitor.get_consecutive_days("NONEXIST") == 0

    def test_reset_clears_state(self):
        monitor = WeaknessExitMonitor(WeaknessExitConfig(threshold=0.6))
        monitor.update("TEST", "20260301", 0.7)
        monitor.update("TEST", "20260302", 0.7)
        monitor.reset("TEST")
        assert monitor.get_consecutive_days("TEST") == 0

    def test_below_threshold_no_accumulate(self):
        monitor = WeaknessExitMonitor(WeaknessExitConfig(threshold=0.6))
        c = monitor.update("TEST", "20260301", 0.5)  # 低于阈值
        assert c == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
