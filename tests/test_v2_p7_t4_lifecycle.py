# -*- coding: utf-8 -*-
"""P7 T4 生命周期 / 价源 / 双口径 / 状态的纯函数与桩件回归。

覆盖点
======
- Q3=A 费用分摊：比例、终端袖、零份额、``A=0,F=0`` / ``A=0,F≠0``、``F>Σc`` 比例分支、
  有符号节省可为负、``Σf=F`` 精确闭合；
- 独立费用按 ``(sleeve, 日, 股票, 方向)`` 聚合后**只计一次**；
- S4 费用瀑布与受审 ``settlement_kernel.allocate_fees`` 同公式（差分回归）；
- 状态机归类 S1~S6、抵消腿超持仓拒绝、卖出按持仓截断、意图守恒；
- 价源：成交只认开盘（不回落收盘）、涨跌停方向阻断、内部转账只认开盘价、
  三级估值解析与停牌恒定性；
- 状态信封：同代重放无操作、跨代拒绝、同键异内容拒绝；
- 落账：S1 黄金例（无市场指令、I1/I2/I3 闭合）、S4 合并、S5 分摊、跨日残差恢复。

本文件不碰真实数据（除非显式 ``pytest.skip`` 的 T2.5 证据根回归）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import pytest

from scripts.v2_p7.t4 import fees as fee_mod
from scripts.v2_p7.t4.dual import run_daily_identities
from scripts.v2_p7.t4.ledger import LedgerError, execute_plan
from scripts.v2_p7.t4.netting import (
    Intent,
    NettingError,
    assert_intent_conservation,
    cap_sell_intents,
    classify_sides,
    intent_from_plan_row,
    plan_day,
)
from scripts.v2_p7.t4.params import sha256_file
from scripts.v2_p7.t4.price import (
    SOURCE_BUY_PRICE,
    SOURCE_DAY_CLOSE,
    SOURCE_LAST_KNOWN,
    PriceBook,
)
from scripts.v2_p7.t4.state import (
    P7Lot,
    PhysicalBook,
    PhysicalLot,
    SleeveState,
    StateEnvelope,
    StateError,
    stable_digest,
)
from src.lazybull.v2.common.cost import CostModel

#: 冻结成本参数（与 ``configs/base.yaml`` 当值一致；显式传入 => 不读默认配置）
COST = CostModel(commission_rate=0.0001954, min_commission=5.0, stamp_tax=0.0005, slippage=0.0005)
DAY = "20240103"


# ---------------------------------------------------------------- 桩件
class StubPrices:
    """价源桩件（``PriceBook`` 的最小接口；不读盘）。"""

    def __init__(self, quotes: Mapping[str, Mapping[str, Any]], day: str = DAY) -> None:
        self.day = day
        self.quotes = {code: dict(values) for code, values in quotes.items()}
        self.blocks: Dict[str, int] = {}
        self.valuations: List[Tuple[str, float, str]] = []
        self.tracked: Tuple[str, ...] = ()

    def begin_day(self, day: str, tracked: Optional[Sequence[str]] = None) -> None:
        self.day = str(day)
        self.tracked = tuple(tracked or ())

    def market_open(self, ts_code: str) -> Any:
        from scripts.v2_p7.t4.price import ExecutionLookup

        quote = self.quotes.get(str(ts_code))
        if quote is None:
            return ExecutionLookup(None, "row_absent")
        if quote.get("suspended"):
            return ExecutionLookup(None, "suspended")
        price = float(quote.get("open") or 0.0)
        if price <= 0.0:
            return ExecutionLookup(None, "no_valid_open")
        return ExecutionLookup(price, "ok")

    def execution_price(self, ts_code: str, side: str) -> Any:
        from scripts.v2_p7.t4.price import ExecutionLookup

        available = self.market_open(ts_code)
        if not available.ok:
            return available
        quote = self.quotes[str(ts_code)]
        if side == "buy" and quote.get("limit_up"):
            return ExecutionLookup(None, "limit_up_blocked")
        if side == "sell" and quote.get("limit_down"):
            return ExecutionLookup(None, "limit_down_blocked")
        return available

    def record_block(self, reason: str) -> None:
        self.blocks[reason] = self.blocks.get(reason, 0) + 1

    def valuation(self, ts_code: str, buy_price: float) -> Tuple[float, str, str]:
        quote = self.quotes.get(str(ts_code)) or {}
        close = float(quote.get("close") or 0.0)
        if close > 0.0:
            return close, self.day, SOURCE_DAY_CLOSE
        known = quote.get("last_known")
        if known:
            return float(known), str(quote.get("last_known_day") or ""), SOURCE_LAST_KNOWN
        if buy_price > 0:
            return float(buy_price), "", SOURCE_BUY_PRICE
        return 0.0, "", "unavailable"

    def note_valuation(self, ts_code: str, price: float, source: str, price_date: str = "") -> None:
        self.valuations.append((str(ts_code), float(price), str(source), str(price_date)))


def _sleeve(sleeve_id: str, cash: float, lots: Sequence[P7Lot] = ()) -> SleeveState:
    return SleeveState(sleeve_id=sleeve_id, cash=float(cash), credit=0.0, lots=tuple(lots))


def _lot(
    vid: str,
    code: str,
    source: str,
    shares: int,
    price: float,
    day: str = "20240102",
    due: str = "",
) -> P7Lot:
    return P7Lot(
        virtual_lot_id=vid,
        ts_code=code,
        source_lot_id=source,
        shares=int(shares),
        acquired_date=day,
        acquired_price=float(price),
        acquired_fee=0.0,
        buy_date=day,
        due_date=due,
        signal_date=day,
        origin_model_version=24008,
    )


def _intent(sleeve: str, code: str, side: str, shares: int, day: str = "20240102") -> Intent:
    return intent_from_plan_row(
        sleeve_id=sleeve,
        ts_code=code,
        side=side,
        shares=shares,
        signal_date=day,
        due_date=day,
        model_version=24008,
    )


# ---------------------------------------------------------------- Q3=A 分摊
def test_s5_split_is_proportional_and_closes_exactly() -> None:
    split = fee_mod.allocate_s5_fees(
        intent_shares={"v20": 100, "v10": 300},
        allocated_shares={"v20": 100, "v10": 300},
        price=10.0,
        physical_fee=5.0,
    )
    assert split.total_fee == 5.0
    assert split.fee_of("v20") == pytest.approx(1.25)
    assert split.fee_of("v10") == pytest.approx(3.75)


def test_s5_terminal_sleeve_is_largest_allocated_amount() -> None:
    split = fee_mod.allocate_s5_fees(
        intent_shares={"v20": 100, "v10": 900},
        allocated_shares={"v20": 100, "v10": 900},
        price=7.0,
        physical_fee=11.0,
    )
    assert split.terminal_sleeve == "v10"
    assert split.total_fee == 11.0


def test_s5_zero_share_sleeve_is_not_charged() -> None:
    split = fee_mod.allocate_s5_fees(
        intent_shares={"v20": 100, "v10": 300},
        allocated_shares={"v20": 0, "v10": 300},
        price=10.0,
        physical_fee=7.0,
    )
    assert split.fee_of("v20") == 0.0
    assert split.fee_of("v10") == 7.0


def test_s5_all_zero_fill_short_circuits_but_rejects_nonzero_fee() -> None:
    zero = fee_mod.allocate_s5_fees(
        intent_shares={"v20": 100},
        allocated_shares={"v20": 0},
        price=10.0,
        physical_fee=0.0,
    )
    assert zero.zero_fill is True
    assert zero.total_fee == 0.0
    with pytest.raises(fee_mod.FeeError):
        fee_mod.allocate_s5_fees(
            intent_shares={"v20": 100},
            allocated_shares={"v20": 0},
            price=10.0,
            physical_fee=0.05,
        )


def test_s5_rejects_allocation_above_intent() -> None:
    with pytest.raises(fee_mod.FeeError):
        fee_mod.allocate_s5_fees(
            intent_shares={"v20": 100},
            allocated_shares={"v20": 200},
            price=10.0,
            physical_fee=1.0,
        )


def test_s5_signed_saving_may_be_negative() -> None:
    split = fee_mod.allocate_s5_fees(
        intent_shares={"v20": 100, "v10": 100},
        allocated_shares={"v20": 100, "v10": 100},
        price=10.0,
        physical_fee=4.0,
    )
    zero_cost = 0.0
    savings = {sid: zero_cost - split.fee_of(sid) for sid in ("v20", "v10")}
    assert all(value < 0.0 for value in savings.values())
    assert sum(savings.values()) == pytest.approx(-4.0)


def test_s4_proportional_branch_when_fee_exceeds_independent_costs() -> None:
    constant = lambda _amount: 1.0  # noqa: E731 - 构造用固定成本
    waterfall = fee_mod.allocate_s4_fees(
        intent_amounts={"v20": 100.0, "v10": 900.0},
        allocated_amounts={"v20": 100.0, "v10": 900.0},
        physical_fee=5.0,
        buy_cost=constant,
    )
    assert waterfall.proportional_branch is True
    assert waterfall.total_fee == 5.0
    assert waterfall.fee_of("v20") == pytest.approx(0.5)
    assert waterfall.fee_of("v10") == pytest.approx(4.5)


@pytest.mark.parametrize(
    "intents,allocated,fee",
    [
        ({"a": 100, "b": 300}, {"a": 50, "b": 150}, 5.0),
        ({"a": 0, "b": 400}, {"a": 0, "b": 400}, 5.0),
        ({"a": 700, "b": 300}, {"a": 700, "b": 300}, 123.456789),
        ({"a": 1, "b": 99999}, {"a": 1, "b": 99999}, 61.0000000001),
    ],
)
def test_s4_matches_audited_waterfall_when_it_closes(
    intents: Mapping[str, int],
    allocated: Mapping[str, int],
    fee: float,
) -> None:
    """与受审 ``allocate_fees`` 同公式（差分回归；受审件拒绝时跳过该样例）。"""
    from src.lazybull.v2.core.accounting.settlement_kernel import allocate_fees

    scope = 10.0
    intent_amounts = {sid: intents[sid] * scope for sid in intents}
    allocated_amounts = {sid: allocated[sid] * scope for sid in allocated}
    mine = fee_mod.allocate_s4_fees(
        intent_amounts=intent_amounts,
        allocated_amounts=allocated_amounts,
        physical_fee=fee,
        buy_cost=COST.calculate_buy_cost,
    )
    # P7 侧实现必须**无条件**闭合（不静默跳过；复审 A-02）
    assert mine.total_fee == mine.physical_fee
    try:
        theirs = allocate_fees(
            intent_amounts=intent_amounts,
            allocated_amounts=allocated_amounts,
            physical_fee=fee,
            buy_cost=COST.calculate_buy_cost,
        )
    except Exception as error:  # noqa: BLE001 - 受审件在 1 ulp 尾差上会拒绝
        # 只有「精确和不成立」这一种拒绝被接受；其他异常一律视为回归
        assert "精确和不成立" in str(error), str(error)
        assert (
            abs(mine.declared_fee - mine.physical_fee) <= max(abs(mine.declared_fee), 1.0) * 2.3e-16
        )
        return
    assert mine.order == theirs.order
    assert mine.terminal_sleeve == theirs.terminal_sleeve
    assert dict(mine.final_fees) == dict(theirs.final_fees)


def test_p7_waterfall_differs_from_audited_only_on_ulp_closure() -> None:
    """差分回归的有效性锚点：至少 3 个样例与受审件逐袖一致，且受审件拒绝样例可复现（A-02）。"""
    from scripts.v2_p7.t4.scenarios import find_audited_waterfall_rejection
    from src.lazybull.v2.core.accounting.settlement_kernel import allocate_fees

    compared = 0
    rejected = 0
    for shares_a, shares_b, price in (
        (100, 300, 10.0),
        (700, 300, 7.77),
        (1300, 2100, 3.31),
        (100, 100, 61.19),
        (2100, 700, 12.34),
    ):
        intent_amounts = {"v20": shares_a * price, "v10": shares_b * price}
        physical_fee = float(COST.calculate_buy_cost((shares_a + shares_b) * price))
        mine = fee_mod.allocate_s4_fees(
            intent_amounts=intent_amounts,
            allocated_amounts=dict(intent_amounts),
            physical_fee=physical_fee,
            buy_cost=COST.calculate_buy_cost,
        )
        assert mine.total_fee == mine.physical_fee
        try:
            theirs = allocate_fees(
                intent_amounts=intent_amounts,
                allocated_amounts=dict(intent_amounts),
                physical_fee=physical_fee,
                buy_cost=COST.calculate_buy_cost,
            )
        except Exception as error:  # noqa: BLE001 - 只接受精确和拒绝
            assert "精确和不成立" in str(error)
            rejected += 1
            continue
        compared += 1
        assert mine.order == theirs.order
        assert dict(mine.final_fees) == dict(theirs.final_fees)
    assert compared >= 3, f"与受审件实际比对样例不足：{compared}"
    assert compared + rejected == 5
    sample = find_audited_waterfall_rejection()
    assert sample["found"] is True
    assert "精确和不成立" in sample["error_text"]


def test_fee_book_aggregates_independent_cost_once_per_key() -> None:
    book = fee_mod.FeeBook()
    event_a = fee_mod.FeeEvent(
        event_id="a",
        trade_date=DAY,
        ts_code="000001.SZ",
        side="buy",
        scenario="offset",
        physical_fee=0.0,
        price=10.0,
        allocations=(("v20", 100),),
    )
    event_b = fee_mod.FeeEvent(
        event_id="b",
        trade_date=DAY,
        ts_code="000001.SZ",
        side="buy",
        scenario="S3",
        physical_fee=float(COST.calculate_buy_cost(1000.0)),
        price=10.0,
        allocations=(("v20", 100),),
    )
    book.add_day(
        DAY,
        [(event_a, (("v20", 0.0),)), (event_b, (("v20", float(COST.calculate_buy_cost(1000.0))),))],
        cost_of_by_side={
            "buy": COST.calculate_buy_cost,
            "sell": COST.calculate_sell_cost,
        },
    )
    key = ("v20", DAY, "000001.SZ", "buy")
    assert book.independent_by_key[key] == pytest.approx(COST.calculate_buy_cost(2000.0))
    assert len(book.independent_by_key) == 1


def test_close_terminal_rejects_gap_beyond_one_ulp() -> None:
    """精度容差是**硬约束**：超出 100 ulp(F) 的残差不是舍入尾差，必须拒绝（CD-1 裁决）。

    容差口径只放过「浮点尾差」，不放过真错；此例用 1e12 量级挤压 1e-3 的费用，
    使两步修正 + 终端袖微调都无法闭合，残差 1e-3 远大于 ulp(1e-3)。
    """
    with pytest.raises(fee_mod.FeeError):
        fee_mod._close_terminal(  # noqa: SLF001 - 定向精度单测
            ("v20", "v10"), {"v20": 1e12, "v10": 1.0}, "v10", 1e-3, "精度测试"
        )


def test_fee_payload_declares_event_and_run_tolerance() -> None:
    """证据里必须能读到精度口径本身（逐事件 100 ulp、全窗 1e-9 元）。"""
    payload = fee_mod.FeeBook().payload()
    assert payload["float_closure"]["event_tolerance_ulp"] == 100
    assert payload["float_closure"]["run_tolerance"] == fee_mod.FEE_FLOAT_CLOSURE_TOLERANCE
    assert fee_mod.FEE_FLOAT_CLOSURE_TOLERANCE == 1e-9


def _fee_event(event_id: str, allocations, physical_fee: float) -> fee_mod.FeeEvent:
    return fee_mod.FeeEvent(
        event_id=event_id,
        trade_date=DAY,
        ts_code="000001.SZ",
        side="buy",
        scenario="S4",
        physical_fee=physical_fee,
        price=10.0,
        allocations=allocations,
    )


def test_fee_book_rejects_missing_sleeve_in_final_fees() -> None:
    """费用键集与分得股数键集不一致必须拒绝（禁止漏袖按 0 静默收费，复审 A-11）。"""
    book = fee_mod.FeeBook()
    event = _fee_event("missing_sleeve", (("v20", 100), ("v10", 100)), 5.0)
    with pytest.raises(fee_mod.FeeError):
        book.add_day(
            DAY,
            [(event, (("v20", 5.0),))],
            cost_of_by_side={
                "buy": COST.calculate_buy_cost,
                "sell": COST.calculate_sell_cost,
            },
        )


def test_fee_book_second_channel_detects_by_sleeve_gap() -> None:
    """按袖汇总通道与逐事件通道是两条独立校验（复审 A-06）。"""
    book = fee_mod.FeeBook()
    book.physical_fee_total = 10.0
    book.running_actual = 10.0
    book.actual_by_sleeve = {"v20": 4.0}
    with pytest.raises(fee_mod.FeeError):
        book.assert_closed()


def test_fee_book_rejects_zero_share_charge_and_unclosed_event() -> None:
    book = fee_mod.FeeBook()
    event = fee_mod.FeeEvent(
        event_id="x",
        trade_date=DAY,
        ts_code="000001.SZ",
        side="sell",
        scenario="S5",
        physical_fee=5.0,
        price=10.0,
        allocations=(("v20", 100),),
    )
    with pytest.raises(fee_mod.FeeError):
        book.add_day(
            DAY,
            [(event, (("v20", 0.0),))],
            cost_of_by_side={"buy": COST.calculate_buy_cost, "sell": COST.calculate_sell_cost},
        )


# ---------------------------------------------------------------- 状态机
@pytest.mark.parametrize(
    "per_sleeve,expected",
    [
        ({"v20": (0, 100), "v10": (100, 0)}, "S1"),
        ({"v20": (0, 300), "v10": (100, 0)}, "S2"),
        ({"v20": (0, 100), "v10": (300, 0)}, "S3"),
        ({"v20": (100, 0), "v10": (200, 0)}, "S4"),
        ({"v20": (0, 100), "v10": (0, 200)}, "S5"),
        ({"v20": (100, 0), "v10": (0, 0)}, "S6"),
        ({"v20": (0, 100), "v10": (0, 0)}, "S6"),
        ({"v20": (0, 0), "v10": (200, 0)}, "S6"),
    ],
)
def test_classify_sides_matches_state_machine_table(
    per_sleeve: Mapping[str, Tuple[int, int]], expected: str
) -> None:
    assert classify_sides(per_sleeve) == expected


def test_plan_day_builds_offset_leg_and_no_market_order_for_s1() -> None:
    code = "000001.SZ"
    intents = [_intent("v20", code, "sell", 500), _intent("v10", code, "buy", 500)]
    plan = plan_day(DAY, intents, {"v20": {code: 1000}, "v10": {}})
    group = plan.groups[0]
    assert group.scenario == "S1"
    assert group.offsets[0].shares == 500
    assert group.market is None
    assert plan.offset_shares() == 500
    assert plan.market_shares() == 0


def test_plan_day_assigns_net_leg_to_net_direction_only() -> None:
    code = "000001.SZ"
    intents = [_intent("v20", code, "sell", 900), _intent("v10", code, "buy", 400)]
    plan = plan_day(DAY, intents, {"v20": {code: 1000}, "v10": {}})
    group = plan.groups[0]
    assert group.scenario == "S2"
    assert group.offset_shares == 400
    assert group.market is not None
    assert group.market.side == "sell"
    assert group.market.shares == 500
    assert group.market.owners == (("v20", 500),)


def test_plan_day_caps_offset_at_actual_holding() -> None:
    """卖出意图超过实际持仓 => 先按持仓截断（原信号资格失效），抵消腿不得引用不存在的持仓。"""
    code = "000001.SZ"
    intents = [_intent("v20", code, "sell", 500), _intent("v10", code, "buy", 500)]
    plan = plan_day(DAY, intents, {"v20": {code: 100}, "v10": {}})
    assert plan.offset_shares() == 100
    assert plan.voided_rows and plan.voided_rows[0]["voided"] == 400
    # 截断后卖 100 / 买 500 => S3（净买 400 归买方袖）
    assert plan.groups[0].scenario == "S3"
    assert plan.groups[0].market is not None
    assert plan.groups[0].market.shares == 400


def test_plan_day_rejects_duplicate_intent_identity() -> None:
    code = "000001.SZ"
    intent = _intent("v20", code, "sell", 100)
    with pytest.raises(NettingError):
        plan_day(DAY, [intent, intent], {"v20": {code: 1000}, "v10": {}})


def test_cap_sell_intents_voids_excess_as_no_position() -> None:
    code = "000001.SZ"
    intents = [_intent("v20", code, "sell", 500)]
    capped, voided = cap_sell_intents(intents, {"v20": {code: 200}}, DAY)
    assert capped[0].shares == 200
    assert voided[0]["voided"] == 300
    assert voided[0]["original"] == 300


def test_intent_conservation_detects_violation() -> None:
    rows = [
        {"original": 100, "offset": 40, "filled": 60, "voided": 0, "carry": 0},
        {"original": 100, "offset": 40, "filled": 50, "voided": 0, "carry": 0},
    ]
    assert assert_intent_conservation(rows)["violations"] == 1


# ---------------------------------------------------------------- 价源
def _price_book(quotes: Mapping[str, Mapping[str, Any]]) -> PriceBook:
    class _Loader:
        def __init__(self, frame: Any) -> None:
            self.frame = frame

        def load_clean_daily_by_date(self, day: str, auto_ensure: bool = False) -> Any:
            del day, auto_ensure
            return self.frame

    import pandas as pd

    rows = []
    for code, values in quotes.items():
        rows.append(
            {
                "ts_code": code,
                "open": float(values.get("open", 0.0)),
                "close": float(values.get("close", 0.0)),
                "is_suspended": int(values.get("suspended", 0)),
                "is_limit_up": int(values.get("limit_up", 0)),
                "is_limit_down": int(values.get("limit_down", 0)),
            }
        )
    frame = pd.DataFrame(rows) if rows else None
    return PriceBook(_Loader(frame))


def test_execution_price_never_falls_back_to_close() -> None:
    prices = _price_book({"000001.SZ": {"open": 0.0, "close": 12.0}})
    prices.begin_day(DAY)
    lookup = prices.execution_price("000001.SZ", "buy")
    assert lookup.ok is False
    assert lookup.reason == "no_valid_open"


def test_execution_price_blocks_limit_up_buy_and_limit_down_sell() -> None:
    prices = _price_book(
        {
            "000001.SZ": {"open": 12.0, "close": 12.0, "limit_up": 1},
            "000002.SZ": {"open": 8.0, "close": 8.0, "limit_down": 1},
        }
    )
    prices.begin_day(DAY)
    assert prices.execution_price("000001.SZ", "buy").reason == "limit_up_blocked"
    assert prices.execution_price("000001.SZ", "sell").ok is True
    assert prices.execution_price("000002.SZ", "sell").reason == "limit_down_blocked"
    # 内部转账只要求"有真实开盘价"，涨跌停不阻断
    assert prices.market_open("000001.SZ").ok is True


def test_valuation_resolution_chain_and_suspension_constancy() -> None:
    """连续 2 日无真实有价日：每日估值恒等于停牌前价；恢复日回到当日收盘价。"""

    class _Loader:
        def load_clean_daily_by_date(self, day: str, auto_ensure: bool = False) -> Any:
            del auto_ensure
            import pandas as pd

            if day in ("20240103", "20240108"):
                return pd.DataFrame(
                    [
                        {
                            "ts_code": "000001.SZ",
                            "open": 10.0,
                            "close": 10.5 if day == "20240103" else 11.0,
                            "is_suspended": 0,
                            "is_limit_up": 0,
                            "is_limit_down": 0,
                        }
                    ]
                )
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000002.SZ",
                        "open": 5.0,
                        "close": 5.0,
                        "is_suspended": 0,
                        "is_limit_up": 0,
                        "is_limit_down": 0,
                    }
                ]
            )

    prices = PriceBook(_Loader())
    prices.begin_day("20240103", tracked=("000001.SZ",))
    price, day, source = prices.valuation("000001.SZ", 9.0)
    assert (price, day, source) == (10.5, "20240103", SOURCE_DAY_CLOSE)
    prices.note_valuation("000001.SZ", price, source)
    for suspended_day in ("20240104", "20240105"):
        prices.begin_day(suspended_day, tracked=("000001.SZ",))
        price, _day, source = prices.valuation("000001.SZ", 9.0)
        assert (price, source) == (10.5, SOURCE_LAST_KNOWN)
        prices.note_valuation("000001.SZ", price, source)
    prices.begin_day("20240108", tracked=("000001.SZ",))
    price, _day, source = prices.valuation("000001.SZ", 9.0)
    assert (price, source) == (11.0, SOURCE_DAY_CLOSE)
    payload = prices.payload()
    assert payload["suspension_valuation"]["drift"] == 0
    assert payload["suspension_valuation"]["constant_to_pre_gap"] == 2
    assert payload["suspension_episodes"][0]["gap_days"] == 2
    assert payload["suspension_episodes"][0]["resume_raw_close"] == 11.0


def test_valuation_last_resort_is_buy_price() -> None:
    prices = _price_book({"000003.SZ": {"open": 6.0, "close": 6.0}})
    prices.begin_day(DAY)
    price, day, source = prices.valuation("999999.SZ", 7.5)
    assert (price, day, source) == (7.5, "", SOURCE_BUY_PRICE)


# ---------------------------------------------------------------- 状态信封
def test_envelope_replay_is_noop_and_cross_generation_rejected() -> None:
    envelope = StateEnvelope(run_id="r1")
    sleeves = {"v20": _sleeve("v20", 1000.0)}
    physical = PhysicalBook(cash=1000.0)
    assert (
        envelope.commit(
            key="k1",
            generation=1,
            fingerprint="f1",
            sleeves=sleeves,
            physical=physical,
            as_of=DAY,
        )
        is True
    )
    before = envelope.state_digest()
    assert (
        envelope.commit(
            key="k1",
            generation=1,
            fingerprint="f1",
            sleeves=sleeves,
            physical=physical,
            as_of=DAY,
        )
        is False
    )
    assert envelope.state_digest() == before
    with pytest.raises(StateError):
        envelope.commit(
            key="k1",
            generation=2,
            fingerprint="f1",
            sleeves=sleeves,
            physical=physical,
            as_of=DAY,
        )
    with pytest.raises(StateError):
        envelope.commit(
            key="k2",
            generation=1,
            fingerprint="f2",
            sleeves=sleeves,
            physical=physical,
            as_of=DAY,
        )


def test_envelope_rejects_same_key_different_content() -> None:
    envelope = StateEnvelope(run_id="r1")
    sleeves = {"v20": _sleeve("v20", 1000.0)}
    physical = PhysicalBook(cash=1000.0)
    envelope.commit(
        key="k1",
        generation=1,
        fingerprint="f1",
        sleeves=sleeves,
        physical=physical,
        as_of=DAY,
    )
    with pytest.raises(StateError):
        envelope.commit(
            key="k1",
            generation=1,
            fingerprint="f2",
            sleeves=sleeves,
            physical=physical,
            as_of=DAY,
        )


def test_envelope_credit_balance_guard() -> None:
    envelope = StateEnvelope(run_id="r1")
    envelope.sleeves = {
        "v20": SleeveState(sleeve_id="v20", cash=10.0, credit=5.0),
        "v10": SleeveState(sleeve_id="v10", cash=10.0, credit=-5.0),
    }
    envelope.assert_credit_balanced()
    envelope.sleeves["v10"] = SleeveState(sleeve_id="v10", cash=10.0, credit=-4.0)
    with pytest.raises(StateError):
        envelope.assert_credit_balanced()


def test_stable_digest_is_bitwise_sensitive() -> None:
    assert stable_digest({"a": 1.0}) != stable_digest({"a": 1.0000000000000002})
    assert stable_digest({"a": 1.0}) == stable_digest({"a": 1.0})


# ---------------------------------------------------------------- 落账
def _base_s1() -> Tuple[Dict[str, SleeveState], PhysicalBook, StubPrices]:
    code = "000001.SZ"
    sleeves = {
        "v20": _sleeve("v20", 100_000.0, [_lot("L1", code, "P1", 1000, 10.0)]),
        "v10": _sleeve("v10", 100_000.0),
    }
    physical = PhysicalBook(
        cash=200_000.0, lots={"P1": PhysicalLot("P1", code, 1000, "20240102", 10.0, 0.0)}
    )
    prices = StubPrices({code: {"open": 12.0, "close": 12.5}})
    return sleeves, physical, prices


def test_s1_golden_case_has_zero_market_order_and_closed_identities() -> None:
    sleeves, physical, prices = _base_s1()
    code = "000001.SZ"
    intents = [_intent("v20", code, "sell", 100), _intent("v10", code, "buy", 100)]
    plan = plan_day(DAY, intents, {"v20": {code: 1000}, "v10": {}})
    outcome = execute_plan(
        plan=plan,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        cost_model=COST,
        lot_size=100,
    )
    assert outcome.scenario_counts == {"S1": 1}
    assert outcome.market_filled == {"buy": 0, "sell": 0}
    assert outcome.offset_shares == 100
    assert all(event.physical_fee == 0.0 for event, _ in outcome.fee_entries)
    assert outcome.sleeves["v20"].cash == pytest.approx(101_200.0)
    assert outcome.sleeves["v10"].cash == pytest.approx(98_800.0)
    report = run_daily_identities(
        trade_date=DAY,
        sleeves=outcome.sleeves,
        physical=outcome.physical,
        prices=prices,
        physical_initial=200_000.0,
        sleeve_capital={"v20": 100_000.0, "v10": 100_000.0},
        cumulative_savings={"v20": 0.0, "v10": 0.0},
    )
    assert report.ok
    assert outcome.physical.cash == pytest.approx(200_000.0)


def test_s4_merge_allocates_by_q18_and_closes_fees() -> None:
    code = "000002.SZ"
    sleeves = {"v20": _sleeve("v20", 50_000.0), "v10": _sleeve("v10", 50_000.0)}
    physical = PhysicalBook(cash=100_000.0)
    prices = StubPrices({code: {"open": 10.0, "close": 10.2}})
    intents = [_intent("v20", code, "buy", 1000), _intent("v10", code, "buy", 3000)]
    plan = plan_day(DAY, intents, {"v20": {}, "v10": {}})
    outcome = execute_plan(
        plan=plan,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        cost_model=COST,
        lot_size=100,
    )
    assert outcome.scenario_counts == {"S4": 1}
    assert outcome.market_filled["buy"] == 4000
    fees = dict(outcome.fee_entries[0][1])
    assert sum(fees.values()) == pytest.approx(outcome.fee_entries[0][0].physical_fee)
    assert set(fees) == {"v20", "v10"}
    report = run_daily_identities(
        trade_date=DAY,
        sleeves=outcome.sleeves,
        physical=outcome.physical,
        prices=prices,
        physical_initial=100_000.0,
        sleeve_capital={"v20": 50_000.0, "v10": 50_000.0},
        cumulative_savings={"v20": 0.0, "v10": 0.0},
    )
    assert report.ok


def test_blocked_buy_carries_and_fills_next_day_without_double_offset() -> None:
    code = "000003.SZ"
    sleeves = {"v20": _sleeve("v20", 100_000.0), "v10": _sleeve("v10", 100_000.0)}
    physical = PhysicalBook(cash=200_000.0)
    blocked = StubPrices({code: {"open": 9.0, "close": 9.0, "limit_up": 1}})
    intent = _intent("v20", code, "buy", 1000)
    plan1 = plan_day(DAY, [intent], {"v20": {}, "v10": {}})
    out1 = execute_plan(
        plan=plan1,
        sleeves=sleeves,
        physical=physical,
        prices=blocked,
        cost_model=COST,
        lot_size=100,
    )
    assert out1.market_filled["buy"] == 0
    residuals = out1.sleeves["v20"].residuals
    assert sum(item.shares for item in residuals) == 1000
    assert residuals[0].reason == "limit_up_blocked"
    for row in out1.conservation_rows:
        assert row["original"] == (row["offset"] + row["filled"] + row["voided"] + row["carry"])
    recovered = StubPrices({code: {"open": 9.0, "close": 9.1}})
    carried = [
        Intent(
            intent_key=item.intent_key,
            sleeve_id=item.sleeve_id,
            ts_code=item.ts_code,
            side=item.side,
            shares=item.shares,
            origin="carry",
            signal_date=item.signal_date,
            due_date=item.due_date,
        )
        for item in residuals
    ]
    plan2 = plan_day(DAY, carried, {"v20": {}, "v10": {}})
    out2 = execute_plan(
        plan=plan2,
        sleeves=out1.sleeves,
        physical=out1.physical,
        prices=recovered,
        cost_model=COST,
        lot_size=100,
    )
    assert out2.market_filled["buy"] == 1000
    assert not out2.sleeves["v20"].residuals
    for row in out2.conservation_rows:
        assert row["original"] == (row["offset"] + row["filled"] + row["voided"] + row["carry"])


def test_suspension_blocks_transfer_even_when_valuation_available() -> None:
    code = "000004.SZ"
    sleeves = {
        "v20": _sleeve("v20", 100_000.0, [_lot("L1", code, "P1", 1000, 10.0)]),
        "v10": _sleeve("v10", 100_000.0),
    }
    physical = PhysicalBook(
        cash=200_000.0, lots={"P1": PhysicalLot("P1", code, 1000, "20240102", 10.0, 0.0)}
    )
    prices = StubPrices({code: {"open": 0.0, "close": 12.0, "suspended": 1}})
    intents = [_intent("v20", code, "sell", 500), _intent("v10", code, "buy", 500)]
    plan = plan_day(DAY, intents, {"v20": {code: 1000}, "v10": {}})
    outcome = execute_plan(
        plan=plan,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        cost_model=COST,
        lot_size=100,
    )
    assert outcome.offset_shares == 0
    assert outcome.sleeves["v20"].cash == sleeves["v20"].cash
    assert outcome.sleeves["v10"].cash == sleeves["v10"].cash
    assert (
        sum(item.shares for state in outcome.sleeves.values() for item in state.residuals) == 1000
    )
    # 估值链本身可用（收盘价 12），但不得用于完成转账
    assert prices.valuation(code, 10.0)[0] == 12.0


def test_s4_remainder_is_voided_at_end_of_day() -> None:
    code = "000005.SZ"
    sleeves = {"v20": _sleeve("v20", 1_000.0), "v10": _sleeve("v10", 1_000.0)}
    physical = PhysicalBook(cash=2_000.0)
    prices = StubPrices({code: {"open": 10.0, "close": 10.0}})
    intents = [_intent("v20", code, "buy", 1000), _intent("v10", code, "buy", 1000)]
    plan = plan_day(DAY, intents, {"v20": {}, "v10": {}})
    outcome = execute_plan(
        plan=plan,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        cost_model=COST,
        lot_size=100,
    )
    assert outcome.voids.get("s4_remainder_voided_eod", 0) > 0
    assert all(not state.residuals for state in outcome.sleeves.values())
    for row in outcome.conservation_rows:
        assert row["original"] == (row["offset"] + row["filled"] + row["voided"] + row["carry"])


def test_nonpositive_equity_is_flagged_not_deleted() -> None:
    code = "000006.SZ"
    sleeves = {
        "v20": _sleeve("v20", 10.0, [_lot("L1", code, "P1", 100, 10.0)]),
        "v10": _sleeve("v10", 10.0),
    }
    physical = PhysicalBook(
        cash=20.0, lots={"P1": PhysicalLot("P1", code, 100, "20240102", 10.0, 0.0)}
    )
    prices = StubPrices({code: {"open": 10.0, "close": 10.0}})
    report = run_daily_identities(
        trade_date=DAY,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        physical_initial=20.0,
        sleeve_capital={"v20": 10.0, "v10": 10.0},
        cumulative_savings={"v20": 2000.0, "v10": 0.0},
    )
    assert report.ok
    assert report.invalid_sleeves.get("v20") == "invalid_nonpositive_equity"
    assert report.independent_nav["v20"] is None
    assert len(report.total_value) == 2  # 当日不删，仍逐袖登记


def test_identities_abort_on_broken_cash_identity() -> None:
    from scripts.v2_p7.t4.dual import DualCaliberError, assert_identities

    code = "000007.SZ"
    sleeves = {"v20": _sleeve("v20", 999.0)}
    physical = PhysicalBook(cash=1000.0)
    prices = StubPrices({code: {"open": 1.0, "close": 1.0}})
    report = run_daily_identities(
        trade_date=DAY,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        physical_initial=1000.0,
        sleeve_capital={"v20": 1000.0},
        cumulative_savings={},
    )
    assert report.ok is False
    with pytest.raises((DualCaliberError, LedgerError)):
        assert_identities(report)


# ---------------------------------------------------------------- 自然实例（数据门）
def test_natural_instances_recompute_matches_t25_when_evidence_present() -> None:
    """自然实例逐笔复算与 T2.5 证据根一致（缺证据根则跳过）。"""
    from pathlib import Path

    from scripts.v2_p7.common import REPO_ROOT
    from scripts.v2_p7.t4.params import FROZEN_INTENT_COUNTS, T25_ROOT_REL
    from scripts.v2_p7.t4.scenarios import recompute_natural

    root = Path(REPO_ROOT) / T25_ROOT_REL
    if not (root / "intent_precheck.json").exists():  # pragma: no cover - 数据缺件
        pytest.skip("T2.5 证据根不在工作区")
    import json

    precheck = json.loads((root / "intent_precheck.json").read_text(encoding="utf-8"))
    natural = recompute_natural(
        root=root,
        arms=("v20", "v10"),
        precheck=precheck,
        frozen_counts=dict(FROZEN_INTENT_COUNTS),
    )
    assert natural["matches_frozen_counts"] is True
    assert natural["matches_t25_precheck"] is True
    assert natural["natural_counts"]["S1"] == 0
    assert natural["s1_status"] == "observed_with_registered_gap"
    assert len(natural["instances"]["S2"]) == 20
    assert len(natural["instances"]["S4"]) == 695


# ---------------------------------------------------------------- C4-01 物理现金
def test_offset_capped_by_buyer_cash_never_overdraws_physical() -> None:
    """内部转账必须由买方袖自有现金支撑，账户不得透支（C4-01 / 口径 a）。

    复刻外部评审的隔离构造：物理现金 1,000；v10 持 X 200 股、现金 0；v20 现金 1,000。
    X 开盘 10 元时抵消 200 股需 2,000 元，超出 v20 全部现金 => 限额 100 股；
    随后 v10 拿回款买 Y 100 股 @15 元。修复前物理现金落到 -505.75 且恒等式仍判通过，
    修复后必须保持非负，且被限额的股数如实登记。
    """
    x, y = "600000.SH", "600001.SH"
    lot = _lot("L1", x, "S1", 200, 10.0)
    sleeves = {"v20": _sleeve("v20", 1_000.0), "v10": _sleeve("v10", 0.0, [lot])}
    physical = PhysicalBook(
        cash=1_000.0,
        lots={
            "S1": PhysicalLot(
                source_lot_id="S1",
                ts_code=x,
                shares=200,
                buy_date="20240102",
                buy_price=10.0,
                buy_fee=0.0,
            )
        },
    )
    prices = StubPrices({x: {"open": 10.0, "close": 10.0}, y: {"open": 15.0, "close": 15.0}})
    intents = [
        _intent("v10", x, "sell", 200),
        _intent("v20", x, "buy", 200),
        _intent("v10", y, "buy", 100),
    ]
    plan = plan_day(DAY, intents, {"v10": {x: 200}, "v20": {}})
    outcome = execute_plan(
        plan=plan,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        cost_model=COST,
        lot_size=100,
    )
    assert outcome.offset_capped_shares == 100
    assert outcome.blocks.get("transfer_cash_capped", 0) >= 1
    assert outcome.physical.cash >= 0.0
    assert all(float(state.cash) >= 0.0 for state in outcome.sleeves.values())


def test_apply_transfer_rejects_buyer_shortfall() -> None:
    """兜底断言：若限额逻辑失效，`_apply_transfer` 必须立刻拒绝而不是把现金扣成负数。"""
    from scripts.v2_p7.t4.ledger import _apply_transfer

    with pytest.raises(LedgerError):
        _apply_transfer(
            leg=_offset_leg("600000.SH", "v10", "v20", 200),
            price=10.0,
            day=DAY,
            lots={"v20": [], "v10": [_lot("L1", "600000.SH", "S1", 200, 10.0)]},
            cash={"v20": 500.0, "v10": 0.0},
            seq=0,
        )


def test_execute_buy_rejects_physical_cash_shortfall() -> None:
    """兜底断言：市场买入若会让物理现金转负，必须拒绝（非杠杆前提）。"""
    code = "600002.SH"
    # 故意构造「袖现金远大于物理现金」的不一致输入（正是转账不限额时会产生的那种状态）：
    # 袖级可负担性放行，但物理账户付不出 -> 必须由物理现金守卫拒绝。
    sleeves = {"v20": _sleeve("v20", 100_000.0), "v10": _sleeve("v10", 0.0)}
    physical = PhysicalBook(cash=1_000.0)
    prices = StubPrices({code: {"open": 10.0, "close": 10.0}})
    intents = [_intent("v20", code, "buy", 100)]
    plan = plan_day(DAY, intents, {"v20": {}, "v10": {}})
    with pytest.raises(LedgerError):
        execute_plan(
            plan=plan,
            sleeves=sleeves,
            physical=physical,
            prices=prices,
            cost_model=COST,
            lot_size=100,
        )


def _offset_leg(code: str, seller: str, buyer: str, shares: int):
    from scripts.v2_p7.t4.netting import OffsetLeg

    return OffsetLeg(
        ts_code=code,
        from_sleeve=seller,
        to_sleeve=buyer,
        shares=int(shares),
        from_intent_key=f"{seller}:{code}:sell",
        to_intent_key=f"{buyer}:{code}:buy",
    )


def test_envelope_rejects_state_differing_only_in_dates_or_physical_totals() -> None:
    """C4-04 复现 A：残差日期、物理累计三项必须进载荷，否则异状态会被当成无操作。

    评审的隔离复现：同键同代同指纹提交后，把残差到期日改成 20991231、信号日改成
    20000101、物理累计费用改成 123456.0 —— 修复前第二次 commit 返回 False（静默无操作），
    修复后必须抛 StateError。
    """
    from dataclasses import replace

    from scripts.v2_p7.t4.state import ResidualEntry

    envelope = StateEnvelope(run_id="c404")
    lot = _lot("L1", "600000.SH", "S1", 100, 10.0, due="20240131")
    residual = ResidualEntry(
        intent_key="v20:600000.SH:buy",
        sleeve_id="v20",
        ts_code="600000.SH",
        side="buy",
        shares=100,
        origin="signal_buy",
        reason="no_price",
        first_blocked_date=DAY,
        days_carried=1,
        signal_date=DAY,
        due_date="20240131",
    )
    sleeves = {
        "v20": SleeveState(
            sleeve_id="v20", cash=1_000.0, credit=0.0, lots=(lot,), residuals=(residual,)
        )
    }
    physical = PhysicalBook(cash=1_000.0, total_fee=1.0)
    assert (
        envelope.commit(
            key="k",
            generation=1,
            fingerprint="f",
            sleeves=sleeves,
            physical=physical,
            as_of=DAY,
        )
        is True
    )

    mutated_lot = replace(lot, due_date="20991231", signal_date="20000101")
    mutated_residual = replace(residual, due_date="20991231", signal_date="20000101")
    mutated_sleeves = {
        "v20": SleeveState(
            sleeve_id="v20",
            cash=1_000.0,
            credit=0.0,
            lots=(mutated_lot,),
            residuals=(mutated_residual,),
        )
    }
    mutated_physical = PhysicalBook(cash=1_000.0, total_fee=123_456.0)
    with pytest.raises(StateError):
        envelope.commit(
            key="k",
            generation=1,
            fingerprint="f",
            sleeves=mutated_sleeves,
            physical=mutated_physical,
            as_of=DAY,
        )


# ---------------------------------------------------------------- C4-02 / 05 / 06 / 03
def test_s4_allocation_base_is_frozen_original_intent() -> None:
    """C4-02：资金上限只做裁剪，Q18 分摊基数必须是冻结原始意图。

    v10 现金不足（上限 700 股），v20 不受限。若仍以「现金截断后的可执行量」为基数，
    两边会被等比压低；按原始意图（3000:1000）分摊时，v20 应拿到扣除 v10 上限后的绝大部分。
    """
    code = "000004.SZ"
    sleeves = {"v20": _sleeve("v20", 50_000.0), "v10": _sleeve("v10", 7_000.0)}
    physical = PhysicalBook(cash=57_000.0)
    prices = StubPrices({code: {"open": 10.0, "close": 10.0}})
    intents = [_intent("v20", code, "buy", 3000), _intent("v10", code, "buy", 1000)]
    plan = plan_day(DAY, intents, {"v20": {}, "v10": {}})
    outcome = execute_plan(
        plan=plan,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        cost_model=COST,
        lot_size=100,
    )
    trade = next(row for row in outcome.trades if row["kind"] == "market_buy")
    split = trade["sleeves"]
    executed = int(trade["shares"])
    assert executed == sum(split.values())
    assert split["v10"] <= 700
    assert split["v20"] > split["v10"]
    # 按原始意图分摊时 v20 应占主导；等比压低会把两者拉近
    # 按原始意图（3000:1000）分摊时两袖**达成率相同**；若以现金截断后的量
    # （3000:600）为基数，达成率会被拉成 0.667 / 0.4。
    # 冻结 Q18 向量的比例必须严格保持（3000:1000 = 3:1），物理 Q 由可行性反推：
    # v10 现金 7,000 元是唯一约束（整手 + 费用后），故 Q 上限 2,700，分得 2,025 / 675
    assert executed == 2_700
    assert split == {"v20": 2_025, "v10": 675}
    # 逐股精确对照受审分摊函数：最终向量 = allocate_shares(原始增量, 物理 Q)
    from src.lazybull.v2.core.accounting.settlement_kernel import allocate_shares

    assert trade["intent_shares"] == {"v20": 3000, "v10": 1000}
    audited = {
        item.sleeve_id: int(item.allocated_shares)
        for item in allocate_shares({"v20": 3000, "v10": 1000}, executed)
    }
    assert split == {sid: value for sid, value in audited.items() if value > 0}
    # 现金可行性必须是**缩小 Q 后**的结果，而不是落地后改分摊
    for sleeve, shares in split.items():
        assert shares * 10.0 + trade["allocated_fees"][sleeve] <= (
            50_000.0 if sleeve == "v20" else 7_000.0
        )


def test_residuals_keep_model_version_and_carry_age() -> None:
    """C4-05：顺延残差必须保留原模型版本、首阻断日与真实累计天数。"""
    from scripts.v2_p7.t4.ledger import _carried_days, _first_blocked
    from scripts.v2_p7.t4.netting import carried_intent

    class _Entry:
        intent_key = "v20:600000.SH:buy"
        sleeve_id = "v20"
        ts_code = "600000.SH"
        side = "buy"
        shares = 500
        signal_date = "20240102"
        due_date = "20240131"
        model_version = 24008
        first_blocked_date = "20240102"
        days_carried = 3

    intent = carried_intent(_Entry())
    assert intent.model_version == 24008
    assert _first_blocked(intent, "20240109") == "20240102"
    assert _carried_days(intent) == 3


def test_sleeve_metrics_invalid_on_nonpositive_equity_day() -> None:
    """C4-06：出现无效权益日即整袖标 invalid，不得跨日拼接收益。"""
    from scripts.v2_p7.t4.runner import _sleeve_metrics

    frame = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103", "20240104"],
            "independent_nav": [1.0, None, 1.1],
            "split_index": [0, 0, 0],
        }
    )
    metrics, status, invalid_dates = _sleeve_metrics(frame)
    assert metrics is None
    assert status == "invalid_nonpositive_equity_day"
    assert invalid_dates == ["20240103"]


def test_upstream_sha_verification_rejects_tampered_input(tmp_path) -> None:
    """C4-03：上游登记 SHA 与实际不符必须拒绝（不是只采集）。"""
    from scripts.v2_p7.t4.params import ParamsError, _verify_against_upstream

    root = tmp_path / "t25"
    root.mkdir()
    target = root / "targets"
    target.mkdir()
    for name in ("plans_v20.parquet", "holdings_v20.parquet", "daily_v20.parquet"):
        (target / name).write_bytes(name.encode("utf-8"))
    (root / "intent_precheck.json").write_bytes(b"{}")
    (root / "manifest.json").write_bytes(b"{}")
    (target / "calendar.json").write_bytes(b"[]")

    real = {
        f"targets/{name}": sha256_file(target / name)
        for name in ("plans_v20.parquet", "holdings_v20.parquet", "daily_v20.parquet")
    }
    real["intent_precheck.json"] = sha256_file(root / "intent_precheck.json")
    real["targets/calendar.json"] = sha256_file(target / "calendar.json")

    ok = _verify_against_upstream(
        root,
        real,
        {"v20": sha256_file(target / "plans_v20.parquet")},
        {"v20": sha256_file(target / "holdings_v20.parquet")},
        {"v20": sha256_file(target / "daily_v20.parquet")},
        target / "calendar.json",
    )
    assert ok["checked"] == 5
    assert not ok["mismatch"]

    tampered = dict(real)
    tampered["targets/plans_v20.parquet"] = "0" * 64
    with pytest.raises(ParamsError):
        _verify_against_upstream(
            root,
            tampered,
            {"v20": sha256_file(target / "plans_v20.parquet")},
            {"v20": sha256_file(target / "holdings_v20.parquet")},
            {"v20": sha256_file(target / "daily_v20.parquet")},
            target / "calendar.json",
        )


def test_capped_offset_keeps_sell_obligation_instead_of_voiding() -> None:
    """C4-R1-01：转账被资金截断后，仍有底仓的卖出义务必须顺延，不得记「无持仓作废」。

    评审的隔离构造：A 持 X 100 股、现金 0；B 现金 0；X 开盘/收盘 10 元；A 卖 100、B 买 100。
    修复前的产物是 ``voids = {insufficient_cash_voided_eod: 100, no_position_voided: 100}``
    且 ``trades = []`` —— 卖方明明有持仓却被作废，等于凭空取消到期退出。
    """
    code = "600000.SH"
    lot = _lot("L1", code, "S1", 100, 10.0)
    sleeves = {"v20": _sleeve("v20", 0.0), "v10": _sleeve("v10", 0.0, [lot])}
    physical = PhysicalBook(
        cash=0.0,
        lots={
            "S1": PhysicalLot(
                source_lot_id="S1",
                ts_code=code,
                shares=100,
                buy_date="20240102",
                buy_price=10.0,
                buy_fee=0.0,
            )
        },
    )
    prices = StubPrices({code: {"open": 10.0, "close": 10.0}})
    intents = [_intent("v10", code, "sell", 100), _intent("v20", code, "buy", 100)]
    plan = plan_day(DAY, intents, {"v10": {code: 100}, "v20": {}})
    outcome = execute_plan(
        plan=plan,
        sleeves=sleeves,
        physical=physical,
        prices=prices,
        cost_model=COST,
        lot_size=100,
    )
    # 关键 1：卖方必须改走**市场卖出**（不能既不卖也不留义务）
    assert outcome.market_filled["sell"] == 100
    # 关键 2：卖方义务不得被记成「无持仓作废」
    assert outcome.voids.get("no_position_voided", 0) == 0
    # 关键 3：买方没钱，其买入按资金不足处理，不产生成交
    assert outcome.market_filled["buy"] == 0


# ---------------------------------------------------------------- C4-04 续跑信封
def _resume(**overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "price_last_known": {"600000.SH": {"close": 12.0, "day": "20240102"}},
        "no_new_buy": [],
        "seq": 1,
    }
    payload.update(overrides)
    return payload


def test_commit_rejects_resume_state_drift_on_same_commit() -> None:
    """C4-04：同一次提交必须把续跑字段与累计字段**一起**校验。

    评审隔离复现：同键同代同指纹、账户不变，只把缓存价 12 改成 99 / 禁买 [] 改成 [A] /
    序号 1 改成 999 —— 修复前返回 False（当成无操作），修复后必须抛 ``StateError``。
    """
    envelope = StateEnvelope(run_id="c404-resume")
    sleeves = {"v20": SleeveState(sleeve_id="v20", cash=1_000.0, credit=0.0)}
    physical = PhysicalBook(cash=1_000.0)
    assert (
        envelope.commit(
            key="k",
            generation=1,
            fingerprint="f",
            sleeves=sleeves,
            physical=physical,
            as_of=DAY,
            resume_state=_resume(),
        )
        is True
    )
    for mutated in (
        _resume(price_last_known={"600000.SH": {"close": 99.0, "day": "20240102"}}),
        _resume(no_new_buy=["v20"]),
        _resume(seq=999),
    ):
        with pytest.raises(StateError):
            envelope.commit(
                key="k",
                generation=1,
                fingerprint="f",
                sleeves=sleeves,
                physical=physical,
                as_of=DAY,
                resume_state=mutated,
            )
    # 真正的重放（不传续跑态 = 沿用已登记值）仍是无操作
    assert (
        envelope.commit(
            key="k",
            generation=1,
            fingerprint="f",
            sleeves=sleeves,
            physical=physical,
            as_of=DAY,
        )
        is False
    )


def test_resume_advance_requires_next_generation_commit() -> None:
    """C4-04：序号推进走**下一代提交**，不能靠排除字段把新状态当重放。"""
    envelope = StateEnvelope(run_id="c404-advance")
    sleeves = {"v20": SleeveState(sleeve_id="v20", cash=1_000.0, credit=0.0)}
    physical = PhysicalBook(cash=1_000.0)
    assert envelope.commit(
        key="k",
        generation=1,
        fingerprint="f",
        sleeves=sleeves,
        physical=physical,
        as_of=DAY,
        resume_state=_resume(),
    )
    with pytest.raises(StateError):
        # 同键推进序号 => 跨代，必须拒绝
        envelope.commit(
            key="k",
            generation=2,
            fingerprint="f2",
            sleeves=sleeves,
            physical=physical,
            as_of=DAY,
            resume_state=_resume(seq=2),
        )
    assert envelope.commit(
        key="k2",
        generation=2,
        fingerprint="f2",
        sleeves=sleeves,
        physical=physical,
        as_of="20240104",
        resume_state=_resume(seq=2),
    )
    assert envelope.resume_state["seq"] == 2


def test_envelope_roundtrip_restores_generation_history_and_dual_caliber() -> None:
    """C4-04：完整版本化信封可 JSON 往返还原代际/幂等历史/双口径/续跑态。"""
    import json

    from scripts.v2_p7.t4.state import DualCaliber

    envelope = StateEnvelope(run_id="c404-roundtrip")
    for generation in (1, 2):
        envelope.commit(
            key=f"k{generation}",
            generation=generation,
            fingerprint=f"f{generation}",
            sleeves={"v20": SleeveState(sleeve_id="v20", cash=1_000.0 - generation, credit=0.0)},
            physical=PhysicalBook(cash=1_000.0 - generation),
            as_of=f"2024010{generation}",
            resume_state=_resume(seq=generation),
        )
    envelope.dual_caliber = DualCaliber(
        actual_fee={"v20": 1.5},
        independent_fee={"v20": 2.5},
        total_value={"v20": 999.0},
        contribution_nav={"v20": 0.999},
        independent_nav={"v20": 0.998},
        invalid_sleeves={},
    )
    text = json.dumps(envelope.to_payload(), ensure_ascii=False, sort_keys=True)
    restored = StateEnvelope(run_id="c404-roundtrip")
    restored.restore_from(json.loads(text))
    assert restored.generation == 2
    assert restored.applied == envelope.applied
    assert restored.resume_state == envelope.resume_state
    assert restored.dual_caliber == envelope.dual_caliber
    assert restored.state_digest() == envelope.state_digest()
    assert restored.digest() == envelope.digest()
    # 续跑必须接着第 3 代（不是新开 1 代）
    assert restored.commit(
        key="k3",
        generation=3,
        fingerprint="f3",
        sleeves={"v20": SleeveState(sleeve_id="v20", cash=996.0, credit=0.0)},
        physical=PhysicalBook(cash=996.0),
        as_of="20240103",
        resume_state=_resume(seq=3),
    )


def test_envelope_restore_rejects_unknown_schema_and_broken_history() -> None:
    """C4-04：续跑只认版本化信封；代际与幂等历史不闭合即拒绝。"""
    with pytest.raises(StateError):
        StateEnvelope(run_id="bad").restore_from({"generation": 0, "idempotency": {}})
    envelope = StateEnvelope(run_id="c404-broken")
    payload = envelope.to_payload()
    payload["generation"] = 3  # 只有 0 个幂等键
    with pytest.raises(StateError):
        StateEnvelope(run_id="c404-broken").restore_from(payload)


def test_fee_book_snapshot_roundtrip_is_bitwise_identical() -> None:
    """C4-04：费用台账累计态可 JSON 往返逐位还原（续跑不重置双通道累计）。"""
    import json

    book = fee_mod.FeeBook()
    event = fee_mod.FeeEvent(
        event_id="a",
        trade_date=DAY,
        ts_code="000001.SZ",
        side="buy",
        scenario="S3",
        physical_fee=float(COST.calculate_buy_cost(1000.0)),
        price=10.0,
        allocations=(("v20", 100),),
    )
    book.add_day(
        DAY,
        [(event, (("v20", float(COST.calculate_buy_cost(1000.0))),))],
        cost_of_by_side={"buy": COST.calculate_buy_cost, "sell": COST.calculate_sell_cost},
    )
    snapshot = json.loads(json.dumps(book.snapshot(), ensure_ascii=False))
    restored = fee_mod.FeeBook.restore(snapshot)
    assert restored.snapshot() == book.snapshot()
    restored.assert_closed()
    with pytest.raises(fee_mod.FeeError):
        fee_mod.FeeBook.restore({"schema": "unknown"})


# ---------------------------------------------------------------- C4-03 行情身份
def _market_fixture(tmp_path: Any) -> Tuple[Any, List[str]]:
    """构造最小行情/日历/主表目录（3 日窗口，末两日为 runoff 模拟）。"""
    days = ["20240102", "20240103", "20240104"]
    daily = tmp_path / "data" / "clean" / "daily"
    daily.mkdir(parents=True)
    for day in days:
        (daily / f"{day[:4]}-{day[4:6]}-{day[6:8]}.parquet").write_bytes(day.encode("utf-8"))
    (tmp_path / "data" / "clean" / "trade_cal.parquet").write_bytes(b"calendar")
    (tmp_path / "data" / "clean" / "stock_basic.parquet").write_bytes(b"basic")
    return tmp_path, days


def _market_expected(root: Any, days: Sequence[str]) -> Dict[str, Any]:
    from scripts.v2_p7.t4.params import market_identity

    payload = market_identity(root, days)
    return {
        "clean_daily": {
            "digest": payload["clean_daily"]["digest"],
            "partitions": payload["clean_daily"]["partitions"],
        },
        "clean_meta": {
            "data/clean/trade_cal.parquet": {
                "sha256": payload["clean_trade_cal"]["sha256"],
                "frozen_sha256": payload["clean_trade_cal"]["sha256"],
                "status": "match",
            },
            "data/clean/stock_basic.parquet": {
                "sha256": payload["clean_stock_basic"]["sha256"],
                "frozen_sha256": payload["clean_stock_basic"]["sha256"],
                "status": "match",
            },
        },
        "accepted_input_drift": [],
    }


def test_market_identity_covers_full_window_and_rejects_drift(tmp_path) -> None:
    """C4-03：身份覆盖全窗（含 runoff）；信号域/runoff/日历/主表任一被改即拒绝。"""
    from scripts.v2_p7.t4.params import ParamsError, market_identity

    root, days = _market_fixture(tmp_path)
    expected = _market_expected(root, days)
    payload = market_identity(root, days, expected=expected)
    assert payload["verification"]["ok"] is True
    assert payload["scope"] == {"first_day": days[0], "last_day": days[-1], "days": 3}
    assert payload["clean_daily"]["partitions"] == 3

    def _mutate(path: str, content: bytes) -> None:
        target = root / path
        target.write_bytes(content)

    daily_of = lambda day: f"data/clean/daily/{day[:4]}-{day[4:6]}-{day[6:8]}.parquet"  # noqa: E731
    # ① 信号域行情被改 / ② runoff 行情被改 / ③ 日历被改 / ④ 证券主表被改
    for index, path in enumerate(
        (
            daily_of(days[0]),
            daily_of(days[-1]),
            "data/clean/trade_cal.parquet",
            "data/clean/stock_basic.parquet",
        )
    ):
        original = (root / path).read_bytes()
        _mutate(path, b"tampered-%d" % index)
        with pytest.raises(ParamsError):
            market_identity(root, days, expected=expected)
        _mutate(path, original)
    # 分区数不一致也要拒绝（少读/多读不得静默）
    wrong = dict(expected)
    wrong["clean_daily"] = {"digest": expected["clean_daily"]["digest"], "partitions": 4}
    with pytest.raises(ParamsError):
        market_identity(root, days, expected=wrong)
    # 显式接受漂移才放行，并留下登记
    _mutate(daily_of(days[0]), b"drifted")
    allowed = market_identity(root, days, expected=expected, accept_drift=True)
    assert allowed["verification"]["ok"] is True
    assert any(row["status"] == "accepted_drift_now" for row in allowed["verification"]["checks"])
    assert allowed["verification"]["accepted_drift"]


def test_market_identity_requires_sorted_full_window(tmp_path) -> None:
    """C4-03：窗口必须升序且不得少读；缺件即拒绝（不得把缺件当 0 行）。"""
    from scripts.v2_p7.t4.params import ParamsError, market_identity

    root, days = _market_fixture(tmp_path)
    expected = _market_expected(root, days)
    with pytest.raises(ParamsError):
        market_identity(root, [days[1], days[0]], expected=expected)
    (root / "data/clean/daily/2024-01-04.parquet").unlink()
    with pytest.raises(ParamsError):
        market_identity(root, days, expected=expected)
