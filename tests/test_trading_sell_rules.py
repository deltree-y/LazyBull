"""trading.sell_rules 共享卖出规则测试"""

from src.lazybull.trading.sell_rules import (
    is_holding_period_expired,
    min_holding_days_for_rebalance_sell,
    select_rebalance_sell_candidates,
)


class TestHoldingPeriodExpired:
    def test_expired(self):
        assert is_holding_period_expired(5, 5)
        assert is_holding_period_expired(6, 5)

    def test_not_expired(self):
        assert not is_holding_period_expired(4, 5)


class TestMinHoldingDaysForRebalanceSell:
    def test_backtest_floor_zero(self):
        """回测口径：holding_period=1 时阈值为 0（全部触发）"""
        assert min_holding_days_for_rebalance_sell(1, floor=0) == 0
        assert min_holding_days_for_rebalance_sell(5, floor=0) == 4

    def test_paper_floor_one(self):
        """纸面口径：rebalance_freq=1 时阈值为 1（保护当日买入）"""
        assert min_holding_days_for_rebalance_sell(1, floor=1) == 1
        assert min_holding_days_for_rebalance_sell(5, floor=1) == 4


class TestSelectRebalanceSellCandidates:
    def test_basic_sell_order_preserved(self):
        """符合条件的持仓按输入顺序输出"""
        decision = select_rebalance_sell_candidates(
            {"A": 10, "B": 8, "C": 9},
            min_holding_days=4,
        )
        assert decision.sells == ["A", "B", "C"]

    def test_skip_queued_first(self):
        """已在其他卖出队列的持仓优先跳过"""
        decision = select_rebalance_sell_candidates(
            {"A": 10, "B": 10},
            min_holding_days=4,
            queued_codes={"A"},
        )
        assert decision.sells == ["B"]
        assert decision.skipped_queued == 1

    def test_skip_protected_and_target(self):
        decision = select_rebalance_sell_candidates(
            {"A": 10, "B": 10, "C": 10},
            min_holding_days=4,
            target_codes={"B"},
            protected_codes={"C"},
        )
        assert decision.sells == ["A"]
        assert decision.skipped_target == 1
        assert decision.skipped_protected == 1

    def test_skip_too_young(self):
        decision = select_rebalance_sell_candidates(
            {"A": 3, "B": 4},
            min_holding_days=4,
        )
        assert decision.sells == ["B"]
        assert decision.skipped_too_young == 1

    def test_none_holding_days_not_filtered_by_age(self):
        """持有天数无法计算时不做年轻过滤（与纸面无交易日列表时行为一致）"""
        decision = select_rebalance_sell_candidates(
            {"A": None},
            min_holding_days=4,
        )
        assert decision.sells == ["A"]

    def test_protected_priority_over_target(self):
        """同时受保护且在目标中的持仓计入保护跳过"""
        decision = select_rebalance_sell_candidates(
            {"A": 10},
            min_holding_days=0,
            target_codes={"A"},
            protected_codes={"A"},
        )
        assert decision.sells == []
        assert decision.skipped_protected == 1
        assert decision.skipped_target == 0
