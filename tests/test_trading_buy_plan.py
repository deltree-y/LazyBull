"""trading.buy_plan 共享槽位匹配骨架测试"""

from src.lazybull.trading.buy_plan import (
    REASON_ALREADY_BOUGHT,
    REASON_EXECUTION_FAILED,
    fill_slots_from_candidates,
)


def _slot(stock: str, weight: float = 0.1) -> dict:
    return {"stock": stock, "weight": weight}


class TestFillSlotsFromCandidates:
    def test_each_slot_takes_highest_available_candidate(self):
        """每个槽位从候选池头部顺位匹配，已成交候选自动去重"""
        slots = [_slot("A"), _slot("B")]
        result = fill_slots_from_candidates(
            slots,
            candidates=["X", "Y", "Z"],
            evaluate_candidate=lambda c, s: (True, ""),
            execute_buy=lambda c, s: True,
        )
        assert result.bought == ["X", "Y"]
        assert [f["stock"] for f in result.filled] == ["X", "Y"]
        assert result.unfilled == []

    def test_rejected_candidate_skipped_with_reason(self):
        """评估失败的候选被跳过并回报原因"""
        rejects = []
        result = fill_slots_from_candidates(
            [_slot("A")],
            candidates=["X", "Y"],
            evaluate_candidate=lambda c, s: (c != "X", "不可交易"),
            execute_buy=lambda c, s: True,
            on_reject=lambda s, c, r: rejects.append((c, r)),
        )
        assert result.bought == ["Y"]
        assert rejects == [("X", "不可交易")]

    def test_execution_failure_continues_to_next_candidate(self):
        """下单失败继续尝试下一候选并回报执行失败原因"""
        rejects = []
        result = fill_slots_from_candidates(
            [_slot("A")],
            candidates=["X", "Y"],
            evaluate_candidate=lambda c, s: (True, ""),
            execute_buy=lambda c, s: c == "Y",
            on_reject=lambda s, c, r: rejects.append((c, r)),
        )
        assert result.bought == ["Y"]
        assert rejects == [("X", REASON_EXECUTION_FAILED)]

    def test_exhausted_candidates_slot_unfilled(self):
        """候选耗尽的槽位进入未成交列表"""
        slot = _slot("A")
        result = fill_slots_from_candidates(
            [slot],
            candidates=["X"],
            evaluate_candidate=lambda c, s: (False, "无价格"),
            execute_buy=lambda c, s: True,
        )
        assert result.unfilled == [slot]
        assert result.bought == []

    def test_already_bought_reason_reported(self):
        """同日已被其他槽位买入的候选回报去重原因"""
        rejects = []
        result = fill_slots_from_candidates(
            [_slot("A"), _slot("B")],
            candidates=["X"],
            evaluate_candidate=lambda c, s: (True, ""),
            execute_buy=lambda c, s: True,
            on_reject=lambda s, c, r: rejects.append((s["stock"], c, r)),
        )
        assert result.bought == ["X"]
        assert len(result.unfilled) == 1
        assert result.unfilled[0]["stock"] == "B"
        assert ("B", "X", REASON_ALREADY_BOUGHT) in rejects
