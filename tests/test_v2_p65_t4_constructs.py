# -*- coding: utf-8 -*-
"""P6.5 T4 补充构造与反变异（设计 §8.4 补充构造表，六类全部可打红）。

覆盖：
1. 先合并再整手（母配额 3,000、价 10 → 母目标 300；子袖先各自整手 100+100=200 失败，
   并由**实际执行器**消费合并目标证明成交由母目标驱动）；
2. 多股票保序（插入序 600000.SH → 000001.SZ → 300750.SZ → 601318.SH，按代码排序失败）；
3. 不重复归一化（8/35,9/35,9/35,9/35 的 float 和 0.9999999999999999；二次归一化改位失败）；
4. 同额费用与置换（字典序受益 A=0/B=5；三袖瀑布 (0,0,5)；全置换结果不变）；
5. 多袖账户单袖发单（A 400 不发单 + B 100→300 ⇒ 物理目标 700 并由执行器成交；卖方向
   A 400 不发单 + B 100→0 ⇒ 物理目标 400，物理卖出零副作用拒绝）；
6. 目标与配额校验（math.fsum 通过、权重位模式不变、n 袖配额按 n×2^-52、精确 ==1 误拒、
   负值/NaN/Inf/ID 不符必须拒绝）。

另含 T3 收口 §6 P3 交接：``T3-A-R3-01``（计划源与订单交叉校验）。

全部为合成输入：不读真实配置/行情，不写 data/runs。
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Dict, Mapping

import pandas as pd
import pytest

from src.lazybull.universe.base import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.common.types import (
    Account,
    Money,
    Order,
    OrderReason,
    OrderSide,
    OrderType,
    Shares,
    TargetPortfolio,
    TSCode,
    TradeDate,
    Weight,
)
from src.lazybull.v2.core.accounting.physical_lot_view import PhysicalLot, PhysicalLotView
from src.lazybull.v2.core.accounting.settlement_kernel import allocate_fees
from src.lazybull.v2.core.accounting.virtual_ledger import (
    PriceSource,
    VirtualLedger,
    VirtualLot,
)
from src.lazybull.v2.core.decision.target_synthesis import (
    TargetSynthesisError,
    assert_merged_rounding,
    assert_single_normalized,
    normalize_positive_scores,
    synthesize_lot_targets,
)
from src.lazybull.v2.core.decision.virtual_merger import (
    MultiSleevePortfolioMerger,
    VirtualMergerError,
)
from src.lazybull.v2.core.signal.base import EqualWeightSignal
from src.lazybull.v2.hosts.order_chain import (
    OrderChainError,
    OrderChainInvariantError,
    assert_plan_integrity,
    build_sleeve_orders,
)
from src.lazybull.v2.hosts.self_split_arms import SelfSplitScenario, run_self_split_arms
from src.lazybull.v2.hosts.virtual_native_engine import VirtualNativeMergeBacktestEngineML

_X = "600000.SH"
_D0 = TradeDate(pd.Timestamp("2026-01-05").date())
_D1 = pd.Timestamp("2026-01-06")
_INSERTION_ORDER = ("600000.SH", "000001.SZ", "300750.SZ", "601318.SH")
_RAW_SCORES = (8.0, 9.0, 9.0, 9.0)
_EXPECTED_WEIGHTS = {code: score / 35.0 for code, score in zip(_INSERTION_ORDER, _RAW_SCORES)}
_CONFIG = {
    "bt_top_n": 1,
    "label_column": "syn",
    "stock_domain": "main",
    "wf_start_date": "20260105",
    "wf_end_date": "20260107",
}


def _cost_model() -> CostModel:
    return CostModel(commission_rate=0.0003, min_commission=5.0, stamp_tax=0.0005, slippage=0.0)


def _weights(pairs: Mapping[str, float]) -> MappingProxyType:
    return MappingProxyType({TSCode(code): Weight(value) for code, value in pairs.items()})


def _target(sleeve_id: str, pairs: Mapping[str, float]) -> TargetPortfolio:
    return TargetPortfolio(sleeve_id, _weights(pairs), _D0)


def _account(cash: float = 1_000_000.0) -> Account:
    return Account(cash=Money(cash), positions={}, total_value=Money(cash), nav=1.0)


def _source(price: float) -> PriceSource:
    return PriceSource(as_of=_D0, source_id="syn:t4c", prices={_X: price})


# ─────────────────── 1. 先合并再整手 ───────────────────


def test_merge_then_lot_rounding_once_only() -> None:
    """母配额 3,000、价 10：母目标 300（子袖各 150），子袖各自整手 100+100 必须失败。"""
    synthesis = synthesize_lot_targets({"A": 1500.0, "B": 1500.0}, price=10.0)
    assert synthesis.mother_value == 3000.0
    assert synthesis.mother_target_shares == 300
    assert synthesis.as_mapping() == {"A": 150, "B": 150}
    assert synthesis.allocated_total == 300

    with pytest.raises(TargetSynthesisError, match="先合并再整手的唯一结果"):
        assert_merged_rounding({"A": 1500.0, "B": 1500.0}, {"A": 100, "B": 100}, price=10.0)
    # 正确结果原样通过（守卫不是恒真：同一函数对唯一结果放行）
    assert (
        assert_merged_rounding({"A": 1500.0, "B": 1500.0}, {"A": 150, "B": 150}, price=10.0) == 300
    )


def test_merged_rounding_drives_physical_fill_end_to_end(tmp_path) -> None:
    """成交由合并目标驱动：两臂都由母目标 300 股下单（不是子袖整手 200 股）。"""
    scenario = SelfSplitScenario(combo_quota=3000.0)
    assembly = run_self_split_arms(
        scenario=scenario,
        run_id="t4c3000",
        base_dir=tmp_path,
        writer_config=_CONFIG,
        code_state={"git_commit": "syn", "git_dirty": False},
        data_state={"data_state_id": "syn_t4c"},
    )
    for arm in (assembly.single, assembly.split):
        assert len(arm.fills) == 1
        assert arm.fills[0].order_target == 300
        assert arm.fills[0].shares == 300
        assert arm.fills[0].fee == 5.0
        assert arm.physical_cash == 96_995.0
        assert arm.physical_positions == ((_X, 300),)
    # 变异（子袖先整手）会得到 200 股，与上面 300 股可区分
    assert 100 + 100 != 300


# ─────────────────── 2. 多股票保序 ───────────────────


def test_multi_stock_insertion_order_is_preserved_end_to_end() -> None:
    """插入序必须原样进入合并目标（按代码排序会改写键序）。"""
    scores = list(zip(_INSERTION_ORDER, _RAW_SCORES))
    weights = normalize_positive_scores(scores)
    assert list(weights) == list(_INSERTION_ORDER)
    assert list(weights) != sorted(_INSERTION_ORDER)
    assert sorted(_INSERTION_ORDER)[0] == "000001.SZ"  # 代码序与插入序可区分（非平凡）

    merger = MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5})
    merged = merger.merge({"A": _target("A", weights), "B": _target("B", weights)}, _account())
    merged_codes = [code.value for code in merged.target_weights]
    assert merged_codes == list(_INSERTION_ORDER)
    assert merged_codes != sorted(_INSERTION_ORDER)
    for code in _INSERTION_ORDER:
        # 两袖同权重同配额：合并结果位模式与原始单袖权重逐位相同（未重排/未重归一化）
        assert merged.target_weights[TSCode(code)].value == weights[code]


# ─────────────────── 3. 不重复归一化 ───────────────────


def test_single_normalization_bits_are_never_recomputed() -> None:
    """8/35 的浮点和为 0.9999999999999999：精确 ==1 会误拒，二次归一化会改位。"""
    scores = list(zip(_INSERTION_ORDER, _RAW_SCORES))
    weights = normalize_positive_scores(scores)
    assert weights["600000.SH"] == 8.0 / 35.0
    total = math.fsum(weights.values())
    assert total == 0.9999999999999999
    assert total != 1.0
    # 归一化后总和落在 n×2^-52 容差内（合法输入，不得被精确判定拒绝）
    assert abs(total - 1.0) <= 4 * 2.0**-52

    renormalized = {code: value / total for code, value in weights.items()}
    assert renormalized["600000.SH"] != weights["600000.SH"]  # 位模式被改写
    with pytest.raises(TargetSynthesisError, match="位模式被改写"):
        assert_single_normalized(scores, renormalized)
    assert_single_normalized(scores, weights)


def test_reordered_normalized_keys_are_rejected() -> None:
    """键序（插入序 = 分数输入序）属契约：重排键序必须失败（Codex-06）。"""
    scores = list(zip(_INSERTION_ORDER, _RAW_SCORES))
    weights = normalize_positive_scores(scores)
    reordered = dict(sorted(weights.items()))
    assert set(reordered) == set(weights)  # 键集与值完全不变，只有顺序不同
    assert list(reordered) != list(weights)
    with pytest.raises(TargetSynthesisError, match="键序被改写"):
        assert_single_normalized(scores, reordered)
    assert_single_normalized(scores, weights)


# ─────────────────── 4. 同额费用与置换 ───────────────────


def _fee_of(waterfall, sleeve_id: str) -> float:
    return dict(waterfall.final_fees)[sleeve_id]


def test_equal_intent_fee_beneficiary_follows_sleeve_id_under_all_permutations() -> None:
    """两袖意图金额均 2,000：受益按袖 ID 字典序（A=0/B=5），置换输入不改变结果。"""
    buy_cost = _cost_model().calculate_buy_cost
    results: Dict[tuple, tuple] = {}
    for keys in (("A", "B"), ("B", "A")):
        waterfall = allocate_fees(
            intent_amounts={key: 2000.0 for key in keys},
            allocated_amounts={key: 2000.0 for key in keys},
            physical_fee=5.0,
            buy_cost=buy_cost,
        )
        assert waterfall.order == ("A", "B")
        results[keys] = tuple(waterfall.final_fees)
    assert results[("A", "B")] == results[("B", "A")] == (("A", 0.0), ("B", 5.0))
    # 非字典序受益（B 先受益）会得到不同结果 ⇒ 本断言可区分变异
    assert ("B", 0.0) not in results[("A", "B")]


def test_three_sleeve_fee_waterfall_matches_design_golden_values() -> None:
    """三袖 500/1,000/1,500 元、F=5、c=(5,5,5)：瀑布为 (0,0,5)，不是 (-5,5,5)。"""
    waterfall = allocate_fees(
        intent_amounts={"A": 500.0, "B": 1000.0, "C": 1500.0},
        allocated_amounts={"A": 500.0, "B": 1000.0, "C": 1500.0},
        physical_fee=5.0,
        buy_cost=_cost_model().calculate_buy_cost,
    )
    assert tuple(waterfall.final_fees) == (("A", 0.0), ("B", 0.0), ("C", 5.0))
    assert math.fsum(value for _sid, value in waterfall.final_fees) == 5.0
    assert all(value >= 0.0 for _sid, value in waterfall.final_fees)


# ─────────────────── 5. 多袖账户单袖发单 ───────────────────


def _host(
    ledger: VirtualLedger, positions: Mapping[str, object], view: PhysicalLotView
) -> VirtualNativeMergeBacktestEngineML:
    universe = BasicUniverse(
        pd.DataFrame({"ts_code": [_X], "name": ["甲"], "industry": ["I"]}), verbose=False
    )
    host = VirtualNativeMergeBacktestEngineML(
        features_by_date={},
        universe=universe,
        signal=EqualWeightSignal(top_n=1),
        initial_capital=100_000.0,
        cost_model=_cost_model(),
        verbose=False,
        portfolio_merger=MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5}),
        virtual_ledger=ledger,
        physical_lot_view=view,
        run_id="T4C",
    )
    host.current_capital = 100_000.0
    host.positions = dict(positions)
    host._emit_immediate_log = lambda *args, **kwargs: None
    host._prepare_price_index(
        pd.DataFrame(
            {
                "ts_code": [_X],
                "trade_date": [_D1],
                "close": [10.0],
                "close_adj": [10.0],
                "open": [10.0],
                "open_adj": [10.0],
            }
        )
    )
    return host


def _two_sleeve_state() -> tuple[VirtualLedger, PhysicalLotView, Dict[str, object]]:
    """A 持 400（来源 L1）、B 持 100（来源 L2）；物理 500 股。"""
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 60_000.0, [VirtualLot("A:L1", _X, "L1", 400, _D0, 10.0)])
    ledger.open_account("B", 40_000.0, [VirtualLot("B:L2", _X, "L2", 100, _D0, 10.0)])
    view = PhysicalLotView([PhysicalLot(_X, "L1", 400), PhysicalLot(_X, "L2", 100)])
    positions = {
        _X: {
            "shares": 500,
            "buy_date": pd.Timestamp("2026-01-05"),
            "buy_cost_cash": 5000.0,
            "buy_trade_price": 10.0,
        }
    }
    return ledger, view, positions


def test_single_sleeve_issuer_restores_full_physical_target_and_executes() -> None:
    """A 400 不发单、B 100→300：物理目标必须是 400+300=700，且由执行器真实成交。"""
    ledger, view, positions = _two_sleeve_state()
    host = _host(ledger, positions, view)
    source = PriceSource(as_of=TradeDate(_D1.date()), source_id="syn:t4c5", prices={_X: 10.0})
    orders = build_sleeve_orders(targets={"A": {_X: 400}, "B": {_X: 300}}, ledger=ledger, date=_D1)
    plan = host.portfolio_merger.plan_netting(orders, ledger.as_virtual_accounts(source))
    assert plan.physical_current[_X] == 500
    assert plan.net_by_code[_X] == 200
    assert plan.physical_targets[_X] == 700
    # 变异：把发单袖目标原样当物理目标（300）或遗漏不动作袖基数（100+200=300）都可区分
    assert plan.physical_targets[_X] != 300
    assert plan.physical_targets[_X] != 100 + 200

    result = host.execute_order_chain(_D1, {"A": {_X: 400}, "B": {_X: 300}}, batch_id="T4C5")
    assert result.observed[0].order_target == 700
    assert result.observed[0].shares == 200
    assert host.positions[_X]["shares"] == 700
    assert host.current_capital == 97_995.0


def test_single_sleeve_sell_restores_full_physical_target_before_any_instruction() -> None:
    """卖方向：A 400 不发单、B 100→0 ⇒ 物理目标 400；物理卖出在 T3 零副作用拒绝。"""
    ledger, view, positions = _two_sleeve_state()
    host = _host(ledger, positions, view)
    source = PriceSource(as_of=TradeDate(_D1.date()), source_id="syn:t4c5", prices={_X: 10.0})
    orders = build_sleeve_orders(targets={"A": {_X: 400}, "B": {_X: 0}}, ledger=ledger, date=_D1)
    plan = host.portfolio_merger.plan_netting(orders, ledger.as_virtual_accounts(source))
    assert plan.net_by_code[_X] == -100
    assert plan.physical_targets[_X] == 400
    assert plan.physical_targets[_X] != 0  # 变异：把 B 的 target 0 原样当物理目标
    physical_orders = plan.physical_orders()
    assert len(physical_orders) == 1
    assert physical_orders[0].side is OrderSide.SELL
    assert physical_orders[0].target_shares.value == 400

    with pytest.raises(OrderChainError, match="状态机=S5/单袖卖出"):
        host.execute_order_chain(_D1, {"A": {_X: 400}, "B": {_X: 0}}, batch_id="T4C5S")
    assert host.positions[_X]["shares"] == 500
    assert host.current_capital == 100_000.0
    assert host.trades == []


# ─────────────────── 6. 目标与配额校验 ───────────────────


def test_target_weights_use_fsum_tolerance_and_keep_bit_pattern() -> None:
    """8/35.. 权重经 math.fsum 容差通过、位模式不变；精确 ==1 会误拒。"""
    merger = MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5})
    merged = merger.merge(
        {"A": _target("A", _EXPECTED_WEIGHTS), "B": _target("B", _EXPECTED_WEIGHTS)},
        _account(),
    )
    for code, value in _EXPECTED_WEIGHTS.items():
        assert merged.target_weights[TSCode(code)].value == value
    total = math.fsum(_EXPECTED_WEIGHTS.values())
    assert total == 0.9999999999999999
    assert total != 1.0
    # 二次归一化会改位（合并器必须原样保留权重位模式）
    assert _EXPECTED_WEIGHTS["600000.SH"] / total != _EXPECTED_WEIGHTS["600000.SH"]


def test_quota_sum_tolerance_is_n_times_two_to_the_minus_52() -> None:
    """配额总和按 n×2^-52：同一 3 ULP 偏差在 2 袖拒绝、4 袖通过（容差随袖数放大）。"""
    quotas = {code: score / 35.0 for code, score in zip(_INSERTION_ORDER, _RAW_SCORES)}
    accepted = MultiSleevePortfolioMerger(_INSERTION_ORDER, quotas)
    total = math.fsum(quotas.values())
    assert total == 0.9999999999999999
    assert total != 1.0  # 精确 ==1 判定会误拒这组合法配额
    assert abs(total - 1.0) <= 4 * 2.0**-52
    for code, value in quotas.items():
        assert accepted.quotas[code] == value  # 位模式不变（未重新归一化）

    deviation = 0.5 - 3 * 2.0**-52  # 与 0.5 相差 3 ULP
    assert math.fsum([0.5, deviation]) - 1.0 == -3 * 2.0**-52
    with pytest.raises(VirtualMergerError, match="构造期配额总和"):
        MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": deviation})
    relaxed = MultiSleevePortfolioMerger(
        ("A", "B", "C", "D"), {"A": 0.5, "B": deviation, "C": 0.0, "D": 0.0}
    )
    assert relaxed.quotas["B"] == deviation

    with pytest.raises(VirtualMergerError, match="容差="):
        MultiSleevePortfolioMerger(("A", "B", "C"), {"A": 0.5, "B": 0.3, "C": 0.3})


@pytest.mark.parametrize(
    "weights,expected",
    [
        ({_X: -0.1, "000001.SZ": 1.1}, "有限非负"),
        ({_X: float("nan")}, "有限非负"),
        ({_X: float("inf")}, "有限非负"),
        ({_X: 0.5}, "目标权重 fsum"),
    ],
)
def test_invalid_target_weights_are_rejected(weights, expected) -> None:
    """负值/NaN/Inf/总和越界必须拒绝（不得静默归一化）。"""
    merger = MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5})
    with pytest.raises(VirtualMergerError, match=expected):
        merger.merge({"A": _target("A", weights), "B": _target("B", weights)}, _account())


def test_quota_id_mismatch_is_rejected() -> None:
    """ID 不符（缺失/多余）必须零副作用拒绝。"""
    with pytest.raises(VirtualMergerError, match="配额键集"):
        MultiSleevePortfolioMerger(("A", "B"), {"A": 1.0})
    merger = MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5})
    with pytest.raises(VirtualMergerError, match="键集不一致"):
        merger.merge({"A": _target("A", _EXPECTED_WEIGHTS)}, _account())


# ─────────────────── T3-A-R3-01 交接：计划源交叉校验 ───────────────────


def _plan_for_two_sleeves() -> tuple[object, tuple[Order, ...]]:
    ledger = VirtualLedger(cost_model=_cost_model())
    ledger.open_account("A", 2500.0)
    ledger.open_account("B", 7500.0)
    orders = build_sleeve_orders(targets={"A": {_X: 100}, "B": {_X: 300}}, ledger=ledger, date=_D1)
    plan = MultiSleevePortfolioMerger(("A", "B"), {"A": 0.5, "B": 0.5}).plan_netting(
        orders, ledger.as_virtual_accounts(_source(10.0))
    )
    return plan, orders


def test_plan_source_must_match_intents_one_to_one() -> None:
    """T3-A-R3-01：计划源与 intents 必须一一对应，整源清空不得平凡自洽。"""
    from dataclasses import replace as dataclass_replace

    plan, orders = _plan_for_two_sleeves()
    assert_plan_integrity(plan, batch_id="OK", date=_D1, source_orders=orders)

    emptied = dataclass_replace(plan, intents=())
    with pytest.raises(OrderChainInvariantError, match="不是一一对应"):
        assert_plan_integrity(emptied, batch_id="A1", date=_D1, source_orders=orders)

    drifted = dataclass_replace(
        plan,
        intents=tuple(
            dataclass_replace(intent, target_shares=intent.target_shares + 100)
            for intent in plan.intents
        ),
    )
    with pytest.raises(OrderChainInvariantError, match="计划源目标股数与 intents 不一致"):
        assert_plan_integrity(drifted, batch_id="A2", date=_D1, source_orders=orders)

    extra_order = Order(
        ts_code=TSCode(_X),
        side=OrderSide.BUY,
        target_shares=Shares(500),
        order_type=OrderType.MARKET_OPEN,
        reason=OrderReason.REBALANCE,
        source_sleeve="C",
    )
    with pytest.raises(OrderChainInvariantError, match="不是一一对应"):
        assert_plan_integrity(plan, batch_id="A3", date=_D1, source_orders=(*orders, extra_order))


def test_plan_source_duplicate_sleeve_stock_is_rejected() -> None:
    """计划源同袖同股重复指令必须失败（禁止一个槽位承载两条目标）。"""
    plan, orders = _plan_for_two_sleeves()
    duplicated = (*orders, orders[0])
    with pytest.raises(OrderChainInvariantError, match="重复袖/股票指令"):
        assert_plan_integrity(plan, batch_id="A4", date=_D1, source_orders=duplicated)
