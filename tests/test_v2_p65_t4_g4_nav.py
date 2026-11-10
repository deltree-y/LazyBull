# -*- coding: utf-8 -*-
"""P6.5 T4 G4 双 NAV 门：先合并总值再单次派生，与 I3 逐袖和分离（D2 1 ULP 见证）。

覆盖：
- D2 见证：``Σ NAV_i_contrib = 1.0039500000000001``（I3 容差通过）与物理
  ``100395/100000 = 1.00395`` 相差 **1 ULP**；G4 合并后单次派生与单袖位模式一致；
- G4 与 I3 运算顺序分离：把逐袖已舍入贡献 NAV 的和当 G4 字段、或用近似/容差兜底，
  必须被 D2 用例捕获（``G4NavMismatchError``）；
- 合并总值按袖 ID 字典序 float64 左到右（固定顺序，可区分求和次序）；
- 输入拒绝面：空袖快照、价格源跨日、初始资金非正、单袖 NAV 非有限。

全部为合成输入：不读真实配置/行情。
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.core.accounting.g4_nav import (
    G4NavMismatchError,
    assert_g4_nav_bitwise_equal,
    assert_g4_nav_matches,
    float64_bits,
    g4_merged_nav,
    merged_sleeve_total_value,
    ulp_distance,
)
from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.core.accounting.reconciliation import (
    NAV_TOLERANCE,
    IdentityContext,
    check_nav_identity,
)
from src.lazybull.v2.core.accounting.virtual_ledger import (
    PriceSource,
    VirtualLedger,
    VirtualLot,
)

_X = "600000.SH"
_D1 = pd.Timestamp("2026-01-06")
_D2 = pd.Timestamp("2026-01-07")
_D1_DATE = TradeDate(_D1.date())
_D2_DATE = TradeDate(_D2.date())
_INITIAL = 100_000.0
#: D2 逐袖货币总值（设计 §8.4 冻结黄金值）
_A_VALUE = 50_200.0
_B_VALUE = 50_195.0
_PHYSICAL_VALUE = 100_395.0


def _d2_ledger() -> VirtualLedger:
    """D2 状态：A 现金 48,000 + 200 股；B 现金 47,995 + 200 股（价 11）。"""
    ledger = VirtualLedger()
    ledger.open_account("A", 48_000.0, [VirtualLot("A:L1", _X, "L1", 200, _D1_DATE, 10.0, 0.0)])
    ledger.open_account("B", 47_995.0, [VirtualLot("B:L1", _X, "L1", 200, _D1_DATE, 10.0, 5.0)])
    return ledger


def _d2_source() -> PriceSource:
    return PriceSource(as_of=_D2_DATE, source_id="syn:t4g4", prices={_X: 11.0})


def _context() -> IdentityContext:
    return IdentityContext(
        trade_date=_D2_DATE,
        host="virtual_native_p65",
        run_id="T4G4",
        sleeve_ids=("A", "B"),
        price_source_id="syn:t4g4",
    )


def test_d2_witness_g4_bitwise_matches_single_arm_while_i3_uses_tolerance() -> None:
    """D2 见证：I3 逐袖和 1.0039500000000001 与物理 1.00395 相差 1 ULP，G4 位模式一致。"""
    ledger = _d2_ledger()
    states = ledger.snapshot()
    source = _d2_source()
    context = _context()

    # 逐袖货币总值与合并总值（固定顺序 A → B）
    assert states["A"].total_value(source) == _A_VALUE
    assert states["B"].total_value(source) == _B_VALUE
    merged, totals = merged_sleeve_total_value(states, source, context=context)
    assert totals == (("A", _A_VALUE), ("B", _B_VALUE))
    assert merged == _PHYSICAL_VALUE

    # I3：直接逐袖贡献 NAV 和（容差门通过），float64 上是 1 ULP 的 0.5 偏差
    i3 = check_nav_identity(
        virtual_states=states,
        physical_total_value=_PHYSICAL_VALUE,
        physical_initial_capital=_INITIAL,
        price_source=source,
        context=context,
    )
    assert i3.left == 1.0039500000000001
    assert i3.right == 1.00395
    assert abs(i3.diff) <= NAV_TOLERANCE
    assert ulp_distance(i3.left, i3.right) == 1
    assert i3.diff == 2.220446049250313e-16 == 2.0**-52

    # G4：先合并总值、单次 calculate_nav，与单袖位模式一致
    evidence = assert_g4_nav_matches(
        virtual_states=states,
        price_source=source,
        physical_initial_capital=_INITIAL,
        single_arm_nav=i3.right,
        context=context,
    )
    assert evidence.merged_nav == 1.00395
    assert evidence.merged_nav_bits == float64_bits(1.00395)
    assert evidence.single_arm_bits == float64_bits(1.00395)
    assert evidence.merged_total_bits == float64_bits(_PHYSICAL_VALUE)
    assert evidence.ulp_distance == 0
    # 1 ULP 见证：I3 的和与 G4 派生结果相差正好 1 ULP
    assert ulp_distance(i3.left, evidence.merged_nav) == 1


def test_g4_rejects_sum_of_sleeve_nav_as_its_own_field() -> None:
    """变异（逐袖 NAV 和 / 先除后加）必须被 G4 位模式门捕获。"""
    ledger = _d2_ledger()
    states = ledger.snapshot()
    source = _d2_source()
    context = _context()

    mutant = 0.0
    for sleeve_id in sorted(states):
        mutant += float(calculate_nav(states[sleeve_id].total_value(source), _INITIAL))
    assert mutant == 1.0039500000000001
    assert float64_bits(mutant) != float64_bits(calculate_nav(_PHYSICAL_VALUE, _INITIAL))

    with pytest.raises(G4NavMismatchError, match="位模式不同"):
        assert_g4_nav_matches(
            virtual_states=states,
            price_source=source,
            physical_initial_capital=_INITIAL,
            single_arm_nav=mutant,
            context=context,
        )
    with pytest.raises(G4NavMismatchError, match="ULP=1"):
        assert_g4_nav_matches(
            virtual_states=states,
            price_source=source,
            physical_initial_capital=_INITIAL,
            single_arm_nav=1.0039500000000001,
            context=context,
        )


def test_assert_g4_nav_bitwise_equal_is_the_single_g4_entry() -> None:
    """两臂已派生 NAV 的位比较入口（装配/证据链路复用同一判定，A-R2 口径）。"""
    left, right = assert_g4_nav_bitwise_equal(1.00395, 1.00395, context=_context())
    assert left == right == 1.00395
    with pytest.raises(G4NavMismatchError, match="两臂 NAV 位模式不同"):
        assert_g4_nav_bitwise_equal(
            1.00395, 1.0039500000000001, context=_context(), label="G4 单袖 vs 自拆合并"
        )
    with pytest.raises(G4NavMismatchError, match="ULP=1"):
        assert_g4_nav_bitwise_equal(1.00395, math.nextafter(1.00395, math.inf), context=_context())
    # 有符号零：数值相等但位模式不同，G4 必须判破缺（不得用 == 兜底）
    assert 0.0 == -0.0
    with pytest.raises(G4NavMismatchError, match="位模式不同"):
        assert_g4_nav_bitwise_equal(0.0, -0.0, context=_context())


def test_g4_has_no_tolerance_fallback() -> None:
    """G4 禁止近似比较：任何 1 ULP 差异都必须报错（不是 warning，不允许容差）。"""
    ledger = _d2_ledger()
    states = ledger.snapshot()
    context = _context()
    correct = 1.00395
    nudged = correct + 2.0**-52
    assert nudged != correct
    with pytest.raises(G4NavMismatchError):
        assert_g4_nav_matches(
            virtual_states=states,
            price_source=_d2_source(),
            physical_initial_capital=_INITIAL,
            single_arm_nav=nudged,
            context=context,
        )


def test_g4_merged_total_uses_fixed_sleeve_order() -> None:
    """合并按袖 ID 字典序 float64 左到右：求和次序改变结果，必须固定为 A→B→C。"""
    ledger = VirtualLedger()
    ledger.open_account("A", 1.0)
    ledger.open_account("B", 1.0)
    ledger.open_account("C", 1e16)
    source = PriceSource(as_of=_D2_DATE, source_id="syn:order", prices={_X: 1.0})
    merged, totals = merged_sleeve_total_value(ledger.snapshot(), source, context=_context())
    assert [sleeve_id for sleeve_id, _value in totals] == ["A", "B", "C"]
    expected = 1.0 + 1.0
    expected += 1e16
    assert merged == expected
    assert merged != 1e16  # 换成 1e16 先加会丢失两个 1.0（求和次序可区分）


def test_g4_input_guards_reject_bad_contexts() -> None:
    """空袖快照、跨日价源、初始资金非正、单袖 NAV 非有限一律拒绝。"""
    ledger = _d2_ledger()
    states = ledger.snapshot()
    source = _d2_source()
    context = _context()
    with pytest.raises(ValueError, match="非空袖快照"):
        g4_merged_nav({}, source, _INITIAL, context=context)
    with pytest.raises(ValueError, match="同日同价源"):
        g4_merged_nav(
            states,
            PriceSource(as_of=_D1_DATE, source_id="syn:cross", prices={_X: 11.0}),
            _INITIAL,
            context=context,
        )
    with pytest.raises(ValueError, match="初始资金"):
        g4_merged_nav(states, source, 0.0, context=context)
    with pytest.raises(G4NavMismatchError, match="非有限"):
        assert_g4_nav_matches(
            virtual_states=states,
            price_source=source,
            physical_initial_capital=_INITIAL,
            single_arm_nav=float("nan"),
            context=context,
        )


def test_ulp_distance_is_monotonic_across_sign_boundary() -> None:
    """ULP 距离辅助函数：相邻可表示值差 1，跨零不跳变，NaN 返回 -1。"""
    assert ulp_distance(1.0, 1.0 + 2.0**-52) == 1
    assert ulp_distance(1.0, 1.0 - 2.0**-53) == 1
    assert ulp_distance(-1.0, 1.0) > 1
    assert ulp_distance(0.0, -0.0) == 0
    assert ulp_distance(float("nan"), 1.0) == -1
    assert math.isfinite(calculate_nav(_PHYSICAL_VALUE, _INITIAL))
