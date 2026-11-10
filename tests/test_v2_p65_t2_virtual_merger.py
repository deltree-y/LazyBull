# -*- coding: utf-8 -*-
"""P6.5 T2 多袖合并器：S1/S4/S6、物理全量目标、配额校验与拒绝面。

契约依据：``docs/contracts/netting_freeze.md`` F25 §2/§3/§5、
``docs/contracts/protocols.md`` F13 §3.4（C65-01 物理全量目标）、设计
``docs/plans/v2/p65_merger_reconciliation_design_20261108.md`` §4.4/§6.1/§6.2/§8.2/§8.3。

覆盖：
- S6 强等价（真正单袖：merge 返回对象本身、net 返回原序列对象）；
- S1 等量反向（多袖零残差）：零物理指令、确定性抵消配对；
- S4 同向买入：单一物理指令且 ``target_shares`` = 物理全量目标；
- F13 关键用例：A 持 X 400 不发单、B 持 100 目标 300 ⇒ 物理目标 700（不是 300/500）；
- 目标/配额校验（有限、非负、ID 集、``math.fsum`` + ``n×2^-52`` 容差、不重归一化）；
- 拒绝面（S2/S3/S5、缺上下文、ID 不符、重复袖/股票、非整数/负目标）零副作用。

全部输入为合成值对象，不依赖真实配置与真实行情。
"""

from __future__ import annotations

import copy
import inspect
import math
from types import MappingProxyType
from typing import Dict, Mapping, get_type_hints

import pandas as pd
import pytest

from src.lazybull.v2.common.protocols.core import PortfolioMerger
from src.lazybull.v2.common.types import (
    Account,
    Lot,
    Money,
    Order,
    OrderReason,
    OrderSide,
    OrderType,
    Shares,
    TargetPortfolio,
    TradeDate,
    TSCode,
    VirtualAccount,
    VirtualPosition,
    Weight,
)
from src.lazybull.v2.core.decision.virtual_merger import (
    MERGED_SLEEVE_ID,
    PHYSICAL_SLEEVE_ID,
    MultiSleevePortfolioMerger,
    NettingPlan,
    VirtualMergerError,
    _assert_merged_total,
    _tolerance,
)

_X = "600000.SH"
_Y = "000001.SZ"
_Z = "300750.SZ"
_D0 = TradeDate(pd.Timestamp("2026-01-05").date())
_D1 = TradeDate(pd.Timestamp("2026-01-06").date())


# ────────────────────────── 构造件 ──────────────────────────


def _weights(pairs: Mapping[str, float]) -> Dict[TSCode, Weight]:
    return {TSCode(code): Weight(value) for code, value in pairs.items()}


def _target(sleeve_id: str, pairs: Mapping[str, float], date: TradeDate = _D0):
    return TargetPortfolio(sleeve_id, _weights(pairs), date)


def _account(cash: float = 1_000_000.0) -> Account:
    return Account(cash=Money(cash), positions={}, total_value=Money(cash), nav=1.0)


def _vaccount(
    sleeve_id: str, cash: float, holdings: Mapping[str, int], price: float = 10.0
) -> VirtualAccount:
    positions = {
        TSCode(code): VirtualPosition(
            ts_code=TSCode(code),
            lots=(Lot(Shares(shares), _D0, Money(shares * price)),),
        )
        for code, shares in holdings.items()
        if shares
    }
    total = cash + sum(shares * price for shares in holdings.values())
    return VirtualAccount(
        sleeve_id=sleeve_id,
        cash=Money(cash),
        positions=MappingProxyType(positions),
        total_value=Money(total),
    )


def _order(
    sleeve_id: str,
    side: OrderSide,
    target: int,
    *,
    code: str = _X,
    reason: OrderReason = OrderReason.REBALANCE,
) -> Order:
    return Order(
        ts_code=TSCode(code),
        side=side,
        target_shares=Shares(target),
        order_type=OrderType.MARKET_OPEN,
        reason=reason,
        source_sleeve=sleeve_id,
    )


def _merger(ids=("A", "B"), quotas: Mapping[str, float] | None = None):
    values = quotas if quotas is not None else {sleeve_id: 1.0 / len(ids) for sleeve_id in ids}
    return MultiSleevePortfolioMerger(list(ids), values)


def _signature(accounts: Mapping[str, VirtualAccount]) -> tuple:
    """账户不可变签名（拒绝路径零副作用的可比对快照）。"""
    rows = []
    for sleeve_id in sorted(accounts):
        account = accounts[sleeve_id]
        holdings = tuple(
            sorted(
                (
                    code.value,
                    tuple((lot.shares.value, lot.cost_basis.value) for lot in position.lots),
                )
                for code, position in account.positions.items()
            )
        )
        rows.append((sleeve_id, account.cash.value, holdings))
    return tuple(rows)


def _plan(merger, orders, accounts) -> NettingPlan:
    return merger.plan_netting(orders, accounts)


# ────────────────────────── S6 真正单袖直通 ──────────────────────────


def test_s6_merge_returns_identical_target_object():
    merger = _merger(("A",), {"A": 1.0})
    weights = _weights({_X: 0.6, _Y: 0.4})
    target = TargetPortfolio("A", weights, _D0)
    merged = merger.merge({"A": target}, _account())
    assert merged is target
    assert merged.target_weights is weights
    assert merged.target_weights[TSCode(_X)].value == 0.6


def test_s6_net_returns_original_sequence_object():
    merger = _merger(("A",), {"A": 1.0})
    orders = [
        _order("A", OrderSide.SELL, 100),
        _order("A", OrderSide.BUY, 200, code=_Y),
    ]
    assert merger.net_across_sleeves(orders) is orders
    empty: list = []
    assert merger.net_across_sleeves(empty) is empty


def test_s6_rejects_other_sleeve_orders_and_multi_context():
    merger = _merger(("A",), {"A": 1.0})
    with pytest.raises(VirtualMergerError, match="S6"):
        merger.net_across_sleeves([_order("B", OrderSide.BUY, 100)])
    context = {"A": _vaccount("A", 0.0, {}), "B": _vaccount("B", 0.0, {})}
    with pytest.raises(VirtualMergerError, match="多袖/异 ID"):
        merger.net_across_sleeves([_order("A", OrderSide.BUY, 100)], context)


def test_s6_rejects_non_account_context_value():
    """B-R3-04：S6 分支也必须校验上下文值的类型与 sleeve_id（不得只看键集）。"""
    merger = _merger(("A",), {"A": 1.0})
    orders = [_order("A", OrderSide.BUY, 100)]
    with pytest.raises(VirtualMergerError, match="必须为 sleeve_id"):
        merger.net_across_sleeves(orders, {"A": object()})
    wrong = _vaccount("B", 0.0, {})
    with pytest.raises(VirtualMergerError, match="必须为 sleeve_id"):
        merger.net_across_sleeves(orders, {"A": wrong})
    assert merger.net_call_count == 0


def test_multi_sleeve_never_falls_back_to_s6_without_context():
    merger = _merger()
    with pytest.raises(VirtualMergerError, match="缺虚拟账户上下文"):
        merger.net_across_sleeves([_order("A", OrderSide.BUY, 100)])
    assert merger.net_call_count == 0


# ────────────────────────── S1 等量反向 ──────────────────────────


def test_s1_zero_turnover_has_no_market_orders():
    merger = _merger()
    accounts = {
        "A": _vaccount("A", 50000.0, {_X: 400}),
        "B": _vaccount("B", 50000.0, {}),
    }
    plan = _plan(
        merger,
        [_order("A", OrderSide.SELL, 300), _order("B", OrderSide.BUY, 100)],
        accounts,
    )
    assert plan.physical_orders() == ()
    assert dict(plan.net_by_code) == {_X: 0}
    assert dict(plan.physical_targets) == {_X: 400}
    assert dict(plan.physical_current) == {_X: 400}
    assert [(item.from_sleeve, item.to_sleeve, item.shares) for item in plan.offsets] == [
        ("A", "B", 100)
    ]
    intents = {item.sleeve_id: item.delta_shares for item in plan.intents}
    assert intents == {"A": -100, "B": 100}
    # 目标股数不是增量：A 的目标仍是 300
    assert {item.sleeve_id: item.target_shares for item in plan.intents} == {"A": 300, "B": 100}


def test_multi_seller_zero_residual_is_rejected_as_unregistered_s1_extension():
    """S1 契约只定义一对等量反向；"多卖一买"零残差属未登记推广（B-R1-05）。"""
    merger = _merger(("A", "B", "C"), {"A": 0.4, "B": 0.3, "C": 0.3})
    accounts = {
        "A": _vaccount("A", 1000.0, {_X: 300}),
        "B": _vaccount("B", 1000.0, {_X: 200}),
        "C": _vaccount("C", 1000.0, {}),
    }
    before = _signature(accounts)
    with pytest.raises(VirtualMergerError, match="只定义一对等量反向"):
        merger.plan_netting(
            [
                _order("A", OrderSide.SELL, 200),
                _order("B", OrderSide.SELL, 100),
                _order("C", OrderSide.BUY, 200),
            ],
            accounts,
        )
    assert _signature(accounts) == before


def test_multi_buyer_zero_residual_is_rejected_as_unregistered_s1_extension():
    merger = _merger(("A", "B", "C"), {"A": 0.4, "B": 0.3, "C": 0.3})
    accounts = {
        "A": _vaccount("A", 1000.0, {_X: 300}),
        "B": _vaccount("B", 1000.0, {}),
        "C": _vaccount("C", 1000.0, {}),
    }
    with pytest.raises(VirtualMergerError, match="只定义一对等量反向"):
        merger.plan_netting(
            [
                _order("A", OrderSide.SELL, 200),
                _order("B", OrderSide.BUY, 50),
                _order("C", OrderSide.BUY, 50),
            ],
            accounts,
        )


def test_s1_transfer_facts_require_price():
    merger = _merger()
    accounts = {
        "A": _vaccount("A", 50000.0, {_X: 400}),
        "B": _vaccount("B", 50000.0, {}),
    }
    plan = _plan(
        merger, [_order("A", OrderSide.SELL, 300), _order("B", OrderSide.BUY, 100)], accounts
    )
    with pytest.raises(VirtualMergerError, match="缺当日价源价格"):
        plan.transfer_facts({})
    facts = plan.transfer_facts({_X: 12.0})
    assert len(facts) == 1 and facts[0].price == 12.0 and facts[0].shares == 100


# ────────────────────────── S4 同向买入 ──────────────────────────


def test_s4_produces_single_physical_full_target_order():
    merger = _merger()
    accounts = {"A": _vaccount("A", 2500.0, {}), "B": _vaccount("B", 7500.0, {})}
    plan = _plan(
        merger,
        [_order("A", OrderSide.BUY, 100), _order("B", OrderSide.BUY, 300)],
        accounts,
    )
    orders = plan.physical_orders()
    assert len(orders) == 1
    order = orders[0]
    assert order.ts_code.value == _X
    assert order.side is OrderSide.BUY
    assert order.target_shares.value == 400
    assert order.source_sleeve == PHYSICAL_SLEEVE_ID
    assert dict(plan.net_by_code) == {_X: 400}


def test_s4_physical_target_includes_non_issuing_sleeve_holdings():
    """F13/C65-01 关键用例：A 持 400 不发单、B 持 100 目标 300 ⇒ 物理目标 700。"""
    merger = _merger()
    accounts = {
        "A": _vaccount("A", 10000.0, {_X: 400}),
        "B": _vaccount("B", 10000.0, {_X: 100}),
    }
    plan = _plan(merger, [_order("B", OrderSide.BUY, 300)], accounts)
    assert dict(plan.physical_current) == {_X: 500}
    assert dict(plan.physical_targets) == {_X: 700}
    order = plan.physical_orders()[0]
    assert order.side is OrderSide.BUY
    assert order.target_shares.value == 700


def test_s4_sell_side_full_target_after_other_sleeve_exits():
    merger = _merger()
    accounts = {
        "A": _vaccount("A", 10000.0, {_X: 400}),
        "B": _vaccount("B", 10000.0, {_X: 100}),
    }
    plan = _plan(merger, [_order("B", OrderSide.SELL, 0)], accounts)
    assert dict(plan.physical_targets) == {_X: 400}
    order = plan.physical_orders()[0]
    assert order.side is OrderSide.SELL
    assert order.target_shares.value == 400


def test_multi_stock_orders_are_deterministically_ordered():
    merger = _merger()
    accounts = {"A": _vaccount("A", 10000.0, {}), "B": _vaccount("B", 10000.0, {})}
    plan = _plan(
        merger,
        [
            _order("B", OrderSide.BUY, 200, code=_Z),
            _order("A", OrderSide.BUY, 100, code=_X),
            _order("B", OrderSide.BUY, 100, code=_X),
        ],
        accounts,
    )
    assert [order.ts_code.value for order in plan.physical_orders()] == sorted([_X, _Z])
    assert dict(plan.physical_targets) == {_X: 200, _Z: 200}


def test_plan_is_pure_and_repeatable():
    merger = _merger()
    accounts = {"A": _vaccount("A", 2500.0, {}), "B": _vaccount("B", 7500.0, {})}
    before = _signature(accounts)
    orders = [_order("A", OrderSide.BUY, 100), _order("B", OrderSide.BUY, 300)]
    orders_before = tuple(orders)
    first = _plan(merger, orders, accounts)
    second = _plan(merger, orders, accounts)
    assert first.orders == second.orders
    assert _signature(accounts) == before
    assert tuple(orders) == orders_before


# ────────────────────────── 目标合并与配额 ──────────────────────────


def test_merge_path_rejects_merged_total_beyond_tolerance():
    """B-R2-01：Σ=1 守卫必须在 merge 路径可打红（不只直测私有函数）。

    B-R3-02：这是**双边界复合效应**——逐袖门与配额门各自"恰好等于容差"时输入本身
    合法，但合成结果残差可达 2×n×2^-52；本守卫按 n×2^-52 判定，是更严的保守界
    （fail-closed），故该输入被拒。注释不得读成"输入本身越界"。
    """
    eps = 2.0**-51  # n=2 时逐袖/配额门容差 = 2×2^-52 = 2^-51
    quotas = {"A": 0.5, "B": 0.5 + eps}
    merger = MultiSleevePortfolioMerger(["A", "B"], quotas)
    targets = {
        "A": _target("A", {_X: 1.0 + eps, _Y: 0.0}),
        "B": _target("B", {_X: 1.0 + eps, _Y: 0.0}),
    }
    with pytest.raises(VirtualMergerError, match="合成后权重 fsum"):
        merger.merge(targets, _account())


def test_multi_merge_applies_quota_and_keeps_total_one():
    """B-R1-01：配额参与目标合成，合并后仍满足 Σ=1 的上游不变量。"""
    merger = _merger()
    targets = {
        "A": _target("A", {_X: 0.6, _Y: 0.4}),
        "B": _target("B", {_Y: 0.3, _Z: 0.7}),
    }
    merged = merger.merge(targets, _account())
    assert merged.sleeve_id == MERGED_SLEEVE_ID
    # 0.5×A + 0.5×B：X=0.3, Y=0.5×0.4+0.5×0.3=0.35, Z=0.35
    assert [(code.value, weight.value) for code, weight in merged.target_weights.items()] == [
        (_X, 0.3),
        (_Y, 0.35),
        (_Z, 0.35),
    ]
    total = math.fsum(weight.value for weight in merged.target_weights.values())
    assert total == 1.0
    # 各袖原始目标未被改写（不重归一化、不排序）
    assert targets["A"].target_weights[TSCode(_X)].value == 0.6
    assert targets["B"].target_weights[TSCode(_Z)].value == 0.7


def test_multi_merge_keeps_bit_patterns_with_asymmetric_quotas():
    fractions = [8 / 35, 9 / 35, 9 / 35, 9 / 35]
    codes = [_X, _Y, _Z, "601318.SH"]
    targets = {
        sleeve: _target(sleeve, dict(zip(codes, fractions)))
        for sleeve in ("A", "B")
    }
    merger = _merger()
    merged = merger.merge(targets, _account())
    # 0.5×f + 0.5×f == f（二进制除 2 精确）：位模式不因合成漂移
    for code, fraction in zip(codes, fractions):
        assert merged.target_weights[TSCode(code)].value == fraction
    merged_total = math.fsum(weight.value for weight in merged.target_weights.values())
    # 与合并守卫同口径（不用 pytest.approx：其默认容差远宽于 n×2^-52）
    assert abs(merged_total - 1.0) <= _tolerance(2)

    skewed = _merger(quotas={"A": 0.25, "B": 0.75})
    skewed_target = skewed.merge(targets, _account())
    for code, fraction in zip(codes, fractions):
        assert skewed_target.target_weights[TSCode(code)].value == 0.25 * fraction + 0.75 * fraction
    # 与合并守卫同口径（T2-P3-5：不用 pytest.approx 的默认宽容差）
    assert (
        abs(math.fsum(w.value for w in skewed_target.target_weights.values()) - 1.0)
        <= _tolerance(2)
    )


def test_merged_total_guard_rejects_non_unit_sum():
    with pytest.raises(VirtualMergerError, match="合成后权重 fsum"):
        _assert_merged_total({_X: 0.6, _Y: 0.7}, 2 * 2.0**-52)
    _assert_merged_total({_X: 0.6, _Y: 0.4}, 2 * 2.0**-52)


def test_quota_validation_rejects_bad_sums_and_values():
    with pytest.raises(VirtualMergerError, match="配额总和"):
        MultiSleevePortfolioMerger(["A", "B"], {"A": 0.5, "B": 0.6})
    with pytest.raises(VirtualMergerError, match="有限非负"):
        MultiSleevePortfolioMerger(["A", "B"], {"A": -0.5, "B": 1.5})
    with pytest.raises(VirtualMergerError, match="有限非负"):
        MultiSleevePortfolioMerger(["A", "B"], {"A": float("nan"), "B": 1.0})
    with pytest.raises(VirtualMergerError, match=r"超出 \[0,1\]"):
        MultiSleevePortfolioMerger(["A", "B"], {"A": 1.2, "B": -0.2})
    with pytest.raises(VirtualMergerError, match="配额键集"):
        MultiSleevePortfolioMerger(["A", "B"], {"A": 1.0})
    with pytest.raises(VirtualMergerError, match="重复"):
        MultiSleevePortfolioMerger(["A", "A"], {"A": 1.0})


def test_quota_sum_tolerance_is_n_times_eps():
    n = 3
    tolerance = n * 2.0**-52
    base = 1.0 / 3.0
    quotas = {"A": base, "B": base, "C": 1.0 - base - base}
    assert abs(math.fsum(quotas.values()) - 1.0) <= tolerance
    merger = MultiSleevePortfolioMerger(["A", "B", "C"], quotas)
    assert merger.quotas["C"] == quotas["C"]


def test_quota_tolerance_two_sided_boundary():
    """B-R1-04：n×2^-52 容差必须双侧锚定（恰好通过 / 2 倍即拒绝）。"""
    eps = 2.0**-52
    two = 2.0  # n = 2 ⇒ tolerance = 2×2^-52
    assert _tolerance(2) == two * eps
    at_boundary = {"A": 0.5, "B": 0.5 + 2 * eps}
    MultiSleevePortfolioMerger(["A", "B"], at_boundary)
    beyond = {"A": 0.5, "B": 0.5 + 4 * eps}
    with pytest.raises(VirtualMergerError, match="配额总和"):
        MultiSleevePortfolioMerger(["A", "B"], beyond)


def test_weight_tolerance_two_sided_boundary(monkeypatch: pytest.MonkeyPatch):
    """目标权重同容差双侧锚定 + 放宽容差的变异必须让用例变红。"""
    import src.lazybull.v2.core.decision.virtual_merger as merger_mod

    eps = 2.0**-52
    merger = _merger()
    at_boundary = {
        "A": _target("A", {_X: 1.0 + 2 * eps, _Y: 0.0}),
        "B": _target("B", {_X: 1.0, _Y: 0.0}),
    }
    # A 恰好落在容差边界（fsum = 1+2eps），通过
    merger.merge(at_boundary, _account())
    beyond = {
        "A": _target("A", {_X: 1.0 + 4 * eps, _Y: 0.0}),
        "B": _target("B", {_X: 1.0, _Y: 0.0}),
    }
    with pytest.raises(VirtualMergerError, match="目标权重 fsum"):
        merger.merge(beyond, _account())
    # 反变异（正向对照）：把容差放宽到 1e-3 后，越界输入必须不再被拒——
    # 证明上述两处拒绝确实由 n×2^-52 容差驱动，而不是别的原因顺带拒绝。
    monkeypatch.setattr(merger_mod, "_tolerance", lambda n: 1e-3)
    loosened = merger_mod.MultiSleevePortfolioMerger(["A", "B"], {"A": 0.5, "B": 0.5})
    merged = loosened.merge(beyond, _account())
    assert merged.target_weights[TSCode(_X)].value == 0.5 * (1.0 + 4 * eps) + 0.5 * 1.0


def test_protocol_structural_and_signature_compatibility():
    """B-R1-03：多袖实现与 PortfolioMerger 的结构兼容 + 有序签名等同。"""
    merger = MultiSleevePortfolioMerger(["A", "B"], {"A": 0.5, "B": 0.5})
    assert isinstance(merger, PortfolioMerger)

    def _full(func):
        hints = get_type_hints(func)
        params = tuple(
            (name, param.kind, param.default, hints.get(name))
            for name, param in inspect.signature(func).parameters.items()
        )
        return params, hints.get("return")

    assert _full(MultiSleevePortfolioMerger.merge) == _full(PortfolioMerger.merge)
    assert _full(MultiSleevePortfolioMerger.net_across_sleeves) == _full(
        PortfolioMerger.net_across_sleeves
    )


def test_merge_rejects_weight_sum_outside_tolerance():
    merger = _merger()
    bad = {"A": _target("A", {_X: 0.6, _Y: 0.3}), "B": _target("B", {_X: 1.0})}
    with pytest.raises(VirtualMergerError, match="目标权重 fsum"):
        merger.merge(bad, _account())


def test_merge_rejects_nan_negative_and_missing_sleeves():
    merger = _merger()
    with pytest.raises(VirtualMergerError, match="有限非负"):
        merger.merge(
            {"A": _target("A", {_X: float("nan")}), "B": _target("B", {_X: 1.0})}, _account()
        )
    with pytest.raises(VirtualMergerError, match="有限非负"):
        merger.merge(
            {"A": _target("A", {_X: -0.5, _Y: 1.5}), "B": _target("B", {_X: 1.0})}, _account()
        )
    with pytest.raises(VirtualMergerError, match="键集不一致"):
        merger.merge({"A": _target("A", {_X: 1.0})}, _account())
    with pytest.raises(VirtualMergerError, match="键集不一致"):
        merger.merge(
            {
                "A": _target("A", {_X: 1.0}),
                "B": _target("B", {_X: 1.0}),
                "C": _target("C", {_X: 1.0}),
            },
            _account(),
        )
    with pytest.raises(VirtualMergerError, match="as_of_date"):
        merger.merge(
            {"A": _target("A", {_X: 1.0}), "B": _target("B", {_X: 1.0}, _D1)}, _account()
        )
    with pytest.raises(VirtualMergerError, match="!= target.sleeve_id"):
        merger.merge({"A": _target("other", {_X: 1.0}), "B": _target("B", {_X: 1.0})}, _account())


def test_merge_counter_tracks_only_successful_calls():
    merger = _merger()
    targets = {"A": _target("A", {_X: 1.0}), "B": _target("B", {_X: 1.0})}
    merger.merge(targets, _account())
    assert merger.merge_call_count == 1
    with pytest.raises(VirtualMergerError):
        merger.merge({"A": targets["A"]}, _account())
    assert merger.merge_call_count == 1


# ────────────────────────── 拒绝面（零副作用） ──────────────────────────


def test_missing_or_mismatched_context_is_rejected_without_side_effects():
    merger = _merger()
    orders = [_order("A", OrderSide.BUY, 100)]
    with pytest.raises(VirtualMergerError, match="缺虚拟账户上下文"):
        merger.net_across_sleeves(orders)
    accounts = {"A": _vaccount("A", 1000.0, {}), "B": _vaccount("B", 1000.0, {})}
    before = _signature(accounts)
    with pytest.raises(VirtualMergerError, match="ID 集"):
        merger.net_across_sleeves(orders, {"A": accounts["A"]})
    with pytest.raises(VirtualMergerError, match="ID 集"):
        merger.net_across_sleeves(orders, {**accounts, "C": _vaccount("C", 0.0, {})})
    assert _signature(accounts) == before
    assert merger.net_call_count == 0


def test_duplicate_sleeve_stock_orders_are_rejected():
    merger = _merger()
    accounts = {"A": _vaccount("A", 1000.0, {}), "B": _vaccount("B", 1000.0, {})}
    before = _signature(accounts)
    with pytest.raises(VirtualMergerError, match="重复袖/股票指令"):
        merger.net_across_sleeves(
            [_order("A", OrderSide.BUY, 100), _order("A", OrderSide.BUY, 200)], accounts
        )
    assert _signature(accounts) == before


def test_non_integer_and_negative_targets_are_rejected():
    merger = _merger()
    accounts = {"A": _vaccount("A", 1000.0, {}), "B": _vaccount("B", 1000.0, {})}
    float_order = Order(
        ts_code=TSCode(_X),
        side=OrderSide.BUY,
        target_shares=Shares(100.0),  # type: ignore[arg-type]
        order_type=OrderType.MARKET_OPEN,
        reason=OrderReason.REBALANCE,
        source_sleeve="A",
    )
    with pytest.raises(VirtualMergerError, match="必须为整数股"):
        merger.net_across_sleeves([float_order], accounts)
    with pytest.raises(VirtualMergerError, match="必须非负"):
        merger.net_across_sleeves([_order("A", OrderSide.BUY, -1)], accounts)
    with pytest.raises(VirtualMergerError, match="不在构造期袖集合"):
        merger.net_across_sleeves([_order("C", OrderSide.BUY, 100)], accounts)


def test_side_must_match_resolved_increment_sign():
    """A-R2-07：指令方向与解析增量符号矛盾必须 fail-fast（不得产出不可结算的物理单）。"""
    merger = _merger()
    accounts = {
        "A": _vaccount("A", 10000.0, {_X: 400}),
        "B": _vaccount("B", 10000.0, {}),
    }
    # A 当前 400，目标 300 ⇒ Δ=-100，但声明 BUY：方向与增量矛盾
    with pytest.raises(VirtualMergerError, match="符号不一致"):
        merger.net_across_sleeves([_order("A", OrderSide.BUY, 300)], accounts)
    # A 当前 0，目标 100 ⇒ Δ=+100，但声明 SELL
    empty = {"A": _vaccount("A", 10000.0, {}), "B": _vaccount("B", 10000.0, {})}
    with pytest.raises(VirtualMergerError, match="符号不一致"):
        merger.net_across_sleeves([_order("A", OrderSide.SELL, 100)], empty)
    assert merger.net_call_count == 0


def test_state_machine_rejections_s2_s3_s5():
    merger = _merger()
    accounts = {
        "A": _vaccount("A", 10000.0, {_X: 300}),
        "B": _vaccount("B", 10000.0, {_X: 100}),
    }
    before = _signature(accounts)
    with pytest.raises(VirtualMergerError, match="状态机=S2"):
        merger.net_across_sleeves(
            [_order("A", OrderSide.SELL, 100), _order("B", OrderSide.BUY, 200)], accounts
        )
    with pytest.raises(VirtualMergerError, match="状态机=S3"):
        merger.net_across_sleeves(
            [_order("A", OrderSide.SELL, 200), _order("B", OrderSide.BUY, 300)], accounts
        )
    with pytest.raises(VirtualMergerError, match="状态机=S5"):
        merger.net_across_sleeves(
            [_order("A", OrderSide.SELL, 100), _order("B", OrderSide.SELL, 50)], accounts
        )
    assert _signature(accounts) == before
    assert merger.net_call_count == 0


def test_net_call_count_tracks_successful_multi_calls():
    merger = _merger()
    accounts = {"A": _vaccount("A", 2500.0, {}), "B": _vaccount("B", 7500.0, {})}
    merger.net_across_sleeves(
        [_order("A", OrderSide.BUY, 100), _order("B", OrderSide.BUY, 300)], accounts
    )
    assert merger.net_call_count == 1
    with pytest.raises(VirtualMergerError, match="重复袖/股票指令"):
        merger.net_across_sleeves(
            [_order("A", OrderSide.BUY, 100), _order("A", OrderSide.BUY, 200)], accounts
        )
    assert merger.net_call_count == 1


def test_merger_does_not_touch_inputs_even_on_success():
    merger = _merger()
    accounts = {
        "A": _vaccount("A", 2500.0, {_X: 400}),
        "B": _vaccount("B", 7500.0, {_X: 100}),
    }
    orders = [_order("A", OrderSide.SELL, 300), _order("B", OrderSide.BUY, 200)]
    accounts_before = copy.deepcopy(_signature(accounts))
    orders_before = tuple(orders)
    merger.net_across_sleeves(orders, accounts)
    assert _signature(accounts) == accounts_before
    assert tuple(orders) == orders_before
