# -*- coding: utf-8 -*-
"""P6.5 T2 费用瀑布（Q13-A）与股数意图上界（Q18-A）黄金值 + 反变异。

契约依据：``docs/contracts/netting_freeze.md`` F25 §5（费用瀑布/终端残差/
零成交短路/意图上界）与设计 §8.3 的逐步算例。

覆盖：
- S4 两袖算例 ``q=(1000,3000)``、``a=(500,1500)``、``F=5``、``c=(5,5)`` ⇒ ``(0,5)``；
- N 袖 ``q=(500,1000,1500)``、``F=5``、``c=(5,5,5)`` ⇒ ``(0,0,5)``；
- 一般金额 ``(100,700,1200)`` 股 @10.01 ⇒ ``(0, 1.0060000000000002, 4.999999999999999)``；
- 同额字典序受益、q/a 排序键区分（仅费用内核）、输入置换不变；
- 零成交短路（``Σa=0,F=0`` 不调用成本模型；``Σa=0,F≠0`` 拒绝）；
- ``F>Σc`` 比例分摊分支、终端残差闭合与非负/精确和守卫；
- Q18 ``167/167/167 → (167,167,166)``、超意图拒绝、容量跳过与零意图容量 0。

全部输入为合成值，无真实配置/行情依赖。
"""

from __future__ import annotations

import itertools
import math

import pytest

from src.lazybull.v2.core.accounting.settlement_batch import (
    _close_terminal_fee,
    _distribute_remainder,
    allocate_fees,
    allocate_shares,
)
from src.lazybull.v2.core.accounting.virtual_ledger import VirtualLedgerError


def _buy_cost(amount: float) -> float:
    """合成买入成本：佣金 ``max(0.0003×amount, 5)``，滑点 0。"""
    return max(amount * 0.0003, 5.0)


def social_sum(waterfall) -> float:
    """最终费用精确和（显式左到右，禁止用带补偿的 ``sum``）。"""
    total = 0.0
    for _sid, fee in waterfall.final_fees:
        total += fee
    return total


def _zero_forbidden(_amount: float) -> float:
    raise AssertionError("零成交短路不得调用成本模型")


def _fee_vector(waterfall) -> tuple:
    return tuple(waterfall.final_fees)


def _exact_sum(values) -> float:
    total = 0.0
    for value in values:
        total += value
    return total


# ────────────────────────── Q13 黄金值 ──────────────────────────


def test_two_sleeve_s4_golden_zero_and_five():
    waterfall = allocate_fees(
        intent_amounts={"A": 1000.0, "B": 3000.0},
        allocated_amounts={"A": 500.0, "B": 1500.0},
        physical_fee=5.0,
        buy_cost=_buy_cost,
    )
    assert waterfall.order == ("A", "B")
    assert waterfall.independent_costs == (("A", 5.0), ("B", 5.0))
    assert waterfall.provisional_fees == (("A", 0.0), ("B", 5.0))
    assert waterfall.terminal_sleeve == "B"
    assert waterfall.residual_correction == 0.0
    assert _fee_vector(waterfall) == (("A", 0.0), ("B", 5.0))
    assert social_sum(waterfall) == 5.0


def test_three_sleeve_golden_zero_zero_five():
    waterfall = allocate_fees(
        intent_amounts={"A": 500.0, "B": 1000.0, "C": 1500.0},
        allocated_amounts={"A": 500.0, "B": 1000.0, "C": 1500.0},
        physical_fee=5.0,
        buy_cost=_buy_cost,
    )
    assert waterfall.order == ("A", "B", "C")
    assert waterfall.provisional_fees == (("A", 0.0), ("B", 0.0), ("C", 5.0))
    assert _fee_vector(waterfall) == (("A", 0.0), ("B", 0.0), ("C", 5.0))
    assert social_sum(waterfall) == 5.0
    # 反变异：若按「节省归最小者」只看第一袖，会得到 (-5,5,5) 的负费用
    assert all(fee >= 0.0 for _sid, fee in waterfall.final_fees)


def test_general_amount_golden_terminal_residual():
    shares = {"A": 100, "B": 700, "C": 1200}
    price = 10.01
    amounts = {sid: count * price for sid, count in shares.items()}
    physical_fee = 2000 * price * 0.0003
    waterfall = allocate_fees(
        intent_amounts=amounts,
        allocated_amounts=amounts,
        physical_fee=physical_fee,
        buy_cost=_buy_cost,
    )
    assert waterfall.independent_costs == (("A", 5.0), ("B", 5.0), ("C", 5.0))
    assert waterfall.provisional_fees == (
        ("A", 0.0),
        ("B", 1.0060000000000002),
        ("C", 5.0),
    )
    assert waterfall.terminal_sleeve == "C"
    assert _fee_vector(waterfall) == (
        ("A", 0.0),
        ("B", 1.0060000000000002),
        ("C", 4.999999999999999),
    )
    assert social_sum(waterfall) == physical_fee
    # 终端费用即「F − 其他袖预备费用」的闭合结果（本例一次即闭合，无需修正）
    assert waterfall.residual_correction == 0.0


def test_terminal_residual_single_correction_path():
    """构造一次残差修正：首轮 Σf≠F，一次修正后精确相等。"""
    provisional = {
        "A": 3611.9888921207043,
        "B": 4630.333399994915,
        "C": 1240.2707675313252,
    }
    physical_fee = 14588.042903560152
    final, residual, total = _close_terminal_fee(
        ("A", "B", "C"), provisional, physical_fee, "B"
    )
    assert residual == 1.8189894035458565e-12
    assert total == physical_fee
    others = provisional["A"] + provisional["C"]
    assert final["B"] == (physical_fee - others) + residual
    assert final["A"] == provisional["A"] and final["C"] == provisional["C"]


def test_terminal_residual_correction_insufficient_is_rejected():
    """一次修正后仍不等即拒绝（不得二次修正或只登记余数）。"""
    provisional = {
        "A": 535.8820043066892,
        "B": 365.6889169125855,
        "C": 57.99892477470681,
    }
    with pytest.raises(VirtualLedgerError, match="精确和不成立"):
        _close_terminal_fee(("A", "B", "C"), provisional, 1101.3166622708163, "B")


def test_general_amount_all_input_permutations_are_invariant():
    shares = {"A": 100, "B": 700, "C": 1200}
    price = 10.01
    amounts = {sid: count * price for sid, count in shares.items()}
    physical_fee = 2000 * price * 0.0003
    baseline = allocate_fees(
        intent_amounts=amounts,
        allocated_amounts=amounts,
        physical_fee=physical_fee,
        buy_cost=_buy_cost,
    )
    expected = dict(baseline.final_fees)
    for permuted in itertools.permutations(("A", "B", "C")):
        result = allocate_fees(
            intent_amounts={sid: amounts[sid] for sid in permuted},
            allocated_amounts={sid: amounts[sid] for sid in permuted},
            physical_fee=physical_fee,
            buy_cost=_buy_cost,
        )
        assert result.order == baseline.order
        assert dict(result.final_fees) == expected
        assert social_sum(result) == physical_fee


def test_same_amount_uses_sleeve_id_lexicographic_order():
    waterfall = allocate_fees(
        intent_amounts={"B": 2000.0, "A": 2000.0},
        allocated_amounts={"B": 2000.0, "A": 2000.0},
        physical_fee=5.0,
        buy_cost=_buy_cost,
    )
    assert waterfall.order == ("A", "B")
    assert _fee_vector(waterfall) == (("A", 0.0), ("B", 5.0))
    reversed_input = allocate_fees(
        intent_amounts={"A": 2000.0, "B": 2000.0},
        allocated_amounts={"A": 2000.0, "B": 2000.0},
        physical_fee=5.0,
        buy_cost=_buy_cost,
    )
    assert _fee_vector(reversed_input) == (("A", 0.0), ("B", 5.0))


def test_benefit_order_uses_intent_amount_not_allocated_amount():
    """q/a 排序键区分例（仅费用内核，不是合法 S4 市场分摊输入）。"""
    waterfall = allocate_fees(
        intent_amounts={"A": 1000.0, "B": 2000.0, "C": 3000.0},
        allocated_amounts={"A": 500.0, "B": 1500.0, "C": 100.0},
        physical_fee=5.0,
        buy_cost=_buy_cost,
    )
    assert waterfall.order == ("A", "B", "C")
    assert _fee_vector(waterfall) == (("A", 0.0), ("B", 0.0), ("C", 5.0))
    # 误按 a 升序（C/A/B）会得到 (0,5,0)：断言当前结果不是该变异
    assert dict(waterfall.final_fees)["B"] == 0.0


# ────────────────────────── 零成交短路 ──────────────────────────


def test_zero_fill_zero_fee_short_circuits_without_cost_model():
    waterfall = allocate_fees(
        intent_amounts={"A": 1000.0, "B": 3000.0},
        allocated_amounts={"A": 0.0, "B": 0.0},
        physical_fee=0.0,
        buy_cost=_zero_forbidden,
    )
    assert _fee_vector(waterfall) == (("A", 0.0), ("B", 0.0))
    assert waterfall.terminal_sleeve == ""
    assert waterfall.provisional_fees == (("A", 0.0), ("B", 0.0))


def test_zero_fill_with_nonzero_fee_is_rejected():
    with pytest.raises(VirtualLedgerError, match="全零成交且费用非零"):
        allocate_fees(
            intent_amounts={"A": 1000.0},
            allocated_amounts={"A": 0.0},
            physical_fee=5.0,
            buy_cost=_zero_forbidden,
        )


def test_zero_allocated_sleeve_never_pays_minimum_commission():
    calls: list = []

    def tracked(amount: float) -> float:
        calls.append(amount)
        return max(amount * 0.0003, 5.0)

    waterfall = allocate_fees(
        intent_amounts={"A": 1000.0, "B": 3000.0},
        allocated_amounts={"A": 0.0, "B": 1500.0},
        physical_fee=5.0,
        buy_cost=tracked,
    )
    assert calls == [1500.0]
    assert dict(waterfall.final_fees)["A"] == 0.0


def test_zero_share_sleeve_cannot_receive_savings():
    """零成交份额不占用节省额度：节省只落在正份额袖上。"""
    waterfall = allocate_fees(
        intent_amounts={"A": 1.0, "B": 3000.0},
        allocated_amounts={"A": 0.0, "B": 1500.0},
        physical_fee=5.0,
        buy_cost=_buy_cost,
    )
    assert waterfall.order == ("A", "B")
    assert _fee_vector(waterfall) == (("A", 0.0), ("B", 5.0))


# ────────────────────────── 异常分支与守卫 ──────────────────────────


def test_physical_fee_above_independent_costs_uses_proportional_branch():
    waterfall = allocate_fees(
        intent_amounts={"A": 100.0, "B": 300.0},
        allocated_amounts={"A": 100.0, "B": 300.0},
        physical_fee=20.0,
        buy_cost=_buy_cost,
    )
    assert waterfall.proportional_branch is True
    assert waterfall.provisional_fees == (("A", 5.0), ("B", 15.0))
    assert _fee_vector(waterfall) == (("A", 5.0), ("B", 15.0))
    assert social_sum(waterfall) == 20.0


def test_terminal_negative_fee_is_rejected():
    with pytest.raises(VirtualLedgerError, match="终端袖 .* 费用为负"):
        _close_terminal_fee(("A", "B"), {"A": 6.0, "B": 1.0}, 5.0, "B")


def test_exact_sum_failure_is_rejected(monkeypatch: pytest.MonkeyPatch):
    import src.lazybull.v2.core.accounting.settlement_kernel as kernel_mod

    def broken(values):
        return 0.0

    monkeypatch.setattr(kernel_mod, "_left_to_right", broken)
    with pytest.raises(VirtualLedgerError, match="精确和不成立"):
        _close_terminal_fee(("A", "B"), {"A": 0.0, "B": 4.999}, 5.0, "B")


def test_invalid_inputs_are_rejected():
    with pytest.raises(VirtualLedgerError, match="键集"):
        allocate_fees(
            intent_amounts={"A": 100.0},
            allocated_amounts={"B": 100.0},
            physical_fee=0.0,
            buy_cost=_buy_cost,
        )
    with pytest.raises(VirtualLedgerError, match="分得金额"):
        allocate_fees(
            intent_amounts={"A": 100.0},
            allocated_amounts={"A": 200.0},
            physical_fee=0.0,
            buy_cost=_buy_cost,
        )
    with pytest.raises(VirtualLedgerError, match="有限非负"):
        allocate_fees(
            intent_amounts={"A": 100.0},
            allocated_amounts={"A": -1.0},
            physical_fee=0.0,
            buy_cost=_buy_cost,
        )
    with pytest.raises(VirtualLedgerError, match="有限非负"):
        allocate_fees(
            intent_amounts={"A": 100.0},
            allocated_amounts={"A": 100.0},
            physical_fee=-5.0,
            buy_cost=_buy_cost,
        )
    with pytest.raises(VirtualLedgerError, match="可调用"):
        allocate_fees(
            intent_amounts={"A": 100.0},
            allocated_amounts={"A": 100.0},
            physical_fee=1.0,
            buy_cost=None,  # type: ignore[arg-type]
        )


def test_non_finite_cost_model_is_rejected():
    def nan_cost(_amount: float) -> float:
        return float("nan")

    with pytest.raises(VirtualLedgerError, match="独立费用"):
        allocate_fees(
            intent_amounts={"A": 100.0},
            allocated_amounts={"A": 100.0},
            physical_fee=1.0,
            buy_cost=nan_cost,
        )


# ────────────────────────── Q18 股数分摊 ──────────────────────────


def test_q18_golden_capacity_waterfall():
    result = allocate_shares({"A": 167, "B": 167, "C": 167}, 500)
    assert [(item.sleeve_id, item.intent_shares, item.allocated_shares) for item in result] == [
        ("A", 167, 167),
        ("B", 167, 167),
        ("C", 167, 166),
    ]
    assert sum(item.allocated_shares for item in result) == 500


def test_q18_two_sleeve_ratio():
    result = allocate_shares({"A": 100, "B": 300}, 200)
    assert [item.allocated_shares for item in result] == [50, 150]


def test_q18_rejects_over_intent_fill():
    with pytest.raises(VirtualLedgerError, match="超意图成交"):
        allocate_shares({"A": 100, "B": 300}, 401)


def test_q18_zero_fill_returns_zeros():
    result = allocate_shares({"A": 100, "B": 300}, 0)
    assert [item.allocated_shares for item in result] == [0, 0]


def test_q18_capacity_skip_keeps_order_deterministic():
    """容量已满的候选跳过、不得超单袖增量；余数恰好落在前两袖。"""
    result = allocate_shares({"A": 1, "B": 1, "C": 1}, 2)
    assert [item.allocated_shares for item in result] == [1, 1, 0]
    # 余数大于候选容量（一轮分配后仍有余数）必须拒绝，不得进第二轮
    with pytest.raises(VirtualLedgerError, match="余数"):
        _distribute_remainder({"A": 1, "B": 1}, {"A": 1, "B": 1}, 1)


def test_q18_zero_intent_sleeve_never_receives_remainder():
    result = allocate_shares({"A": 0, "B": 3}, 3)
    assert [item.allocated_shares for item in result] == [0, 3]
    zero = allocate_shares({"A": 0, "B": 0}, 0)
    assert [item.allocated_shares for item in zero] == [0, 0]
    with pytest.raises(VirtualLedgerError, match="超意图成交"):
        allocate_shares({"A": 0, "B": 0}, 1)


def test_q18_each_sleeve_bounded_by_intent():
    result = allocate_shares({"A": 7, "B": 11, "C": 13}, 30)
    for item in result:
        assert 0 <= item.allocated_shares <= item.intent_shares
    assert sum(item.allocated_shares for item in result) == 30


def test_q18_does_not_use_float_floor_rounding():
    """三袖各 167 股必须得到 167/167/166（不是 168/166/166）。"""
    result = allocate_shares({"A": 167, "B": 167, "C": 167}, 500)
    assert [item.allocated_shares for item in result] == [167, 167, 166]
    assert math.floor(500 * 167 / 501) == 166  # 锚定 float 路径的原始基数
