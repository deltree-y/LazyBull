# -*- coding: utf-8 -*-
"""P7 T5 回归：配额/授权、借用会计、召回与死局、信封扩展与逐袖非负门。

口径
====
全部用桩件（合成行情 + 合成参数），**不碰真实数据/证据根**；断言只针对机械量。

覆盖点（对应任务书 §6 验收门）
=============================
- ``R = max(0, ρG − MV) ∈ [0, D]``、整手后无额度、cap-binding 非真空；
- ``A ≤ min(D, R+L)`` 与 ``A + Fee(A) ≤ cash_open + L``（费用预留、零份额不收费）；
- ``credit_outstanding`` 不参与授权；``L`` 有符号（出借方为负）且只入账一次；
- FIFO 借款 lot、部分归还、T+1 召回、永久 ``deadlocked``、``late_recovery``；
- ``Σ credit = 0`` 精确、本金逐笔对消、本金转移零 P&L；
- ``cash_i + credit_i ≥ 0`` 逐日硬门（违约即拒绝）；
- ``p7_t5_state/v1`` 信封：schema 隔离、同键同代含 credit 状态校验、跨代拒绝；
- T4 兼容门（``credit=off & quota=off`` 与 T4 宿主逐位一致）与 VV-H credit 惰性。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import pytest

from scripts.v2_p7.t4.host import HistoricalVirtualSleeveHost
from scripts.v2_p7.t4.netting import Intent, intent_from_plan_row
from scripts.v2_p7.t4.price import PriceBook
from scripts.v2_p7.t4.state import SleeveState
from scripts.v2_p7.t5.borrow import (
    DEADLOCKED,
    OUTSTANDING,
    RETURNED,
    BorrowError,
    BorrowLot,
    fifo_order,
    next_due_date,
    principal_offsets,
)
from scripts.v2_p7.t5.credit import CreditBook, CreditError, CreditGrant, allocate_credit
from scripts.v2_p7.t5.host import (
    MODE_T4_PARITY,
    MODE_VVH,
    MODE_VVS,
    QuotaCreditHost,
)
from scripts.v2_p7.t5.quota import (
    QUOTA_NO_LOT_REASON,
    QuotaError,
    authorized_by_cash,
    decide_quota,
    funding_gap,
    holdings_market_value,
    quota_bounds,
)
from scripts.v2_p7.t5.recall import service_credit_day
from scripts.v2_p7.t5.state import (
    T5StateEnvelope,
    assert_nonnegative_gate,
    sleeve_balances,
    zero_pnl_check,
)
from src.lazybull.v2.common.cost import CostModel

COST = CostModel(commission_rate=0.0001954, min_commission=5.0, stamp_tax=0.0005, slippage=0.0005)
DAYS: Tuple[str, ...] = tuple(f"202401{idx:02d}" for idx in range(2, 32))
CODES = ("000001.SZ", "000002.SZ")


# ---------------------------------------------------------------- 桩件


class _Loader:
    def __init__(self, prices: Mapping[str, float]) -> None:
        self.prices = dict(prices)

    def load_clean_daily_by_date(self, day: str, auto_ensure: bool = False) -> Any:
        del auto_ensure
        return pd.DataFrame(
            [
                {
                    "ts_code": code,
                    "open": float(self.prices[code]),
                    "close": float(self.prices[code]),
                    "is_suspended": 0,
                    "is_limit_up": 0,
                    "is_limit_down": 0,
                }
                for code in CODES
            ]
        )


class _Params:
    def __init__(
        self,
        *,
        days: Sequence[str] = DAYS,
        capital: Mapping[str, float] = None,
        target: float = 1_000_000.0,
        ratios: Mapping[str, float] = None,
        lot_size: int = 100,
    ) -> None:
        self._capital = dict(capital or {"v20": 500_000.0, "v10": 500_000.0})
        self._ratios = dict(ratios or {"v20": 0.7, "v10": 0.3})
        self._target = float(target)
        self.arms = tuple(sorted(self._capital))
        self.sleeve_capital = dict(self._capital)
        self.physical_capital = float(sum(self._capital.values()))
        self.lot_size = int(lot_size)
        self.extended_days = tuple(str(day) for day in days)
        self.quota_ratio = dict(self._ratios)
        self.account_target = {day: self._target for day in self.extended_days}
        self.arm_target = {
            day: {arm: self._target * self._ratios[arm] for arm in self.arms}
            for day in self.extended_days
        }

        class _T25:
            folds = ((0, self.extended_days[0], self.extended_days[-1], 24008),)

        self.t25 = _T25()

    def ratio_of(self, sleeve_id: str) -> float:
        return float(self._ratios[str(sleeve_id)])

    def target_of(self, day: str) -> float:
        return float(self.account_target.get(str(day), 0.0))

    def arm_target_of(self, day: str, arm: str) -> float:
        return float(self.arm_target.get(str(day), {}).get(str(arm), 0.0))

    def split_of(self, day: str) -> int:
        del day
        return 0

    def model_of(self, day: str) -> int:
        del day
        return 24008


def _intents(
    plan: Sequence[Tuple[str, str, str, int, str]],
) -> Tuple[Dict[str, List[Intent]], Dict[str, List[Intent]]]:
    buys: Dict[str, List[Intent]] = {}
    sells: Dict[str, List[Intent]] = {}
    for day, sleeve, code, shares, due in plan:
        buys.setdefault(str(day), []).append(
            intent_from_plan_row(
                sleeve_id=sleeve,
                ts_code=code,
                side="buy",
                shares=int(shares),
                signal_date=str(day),
                due_date=str(due),
            )
        )
        sells.setdefault(str(due), []).append(
            intent_from_plan_row(
                sleeve_id=sleeve,
                ts_code=code,
                side="sell",
                shares=int(shares),
                signal_date=str(day),
                due_date=str(due),
            )
        )
    return buys, sells


def _run(
    *,
    params: Any,
    prices: Mapping[str, float],
    plan: Sequence[Tuple[str, str, str, int, str]],
    mode: Any,
) -> Any:
    buys, sells = _intents(plan)
    loader = _Loader(prices)
    return QuotaCreditHost(
        params=params,
        run_id=f"t5-{mode.label}",
        prices=PriceBook(loader),
        cost_model=COST,
        buy_index=buys,
        sell_index=sells,
        price_factory=lambda: PriceBook(loader),
        mode=mode,
    ).run()


# ---------------------------------------------------------------- 配额纯函数


def test_quota_bounds_never_exceed_demand() -> None:
    """``R = max(0, ρG−MV)`` 必须落在 ``[0, D]``（ρ≤1 的结构推论）。"""
    bounds = quota_bounds(gross_target=1_000_000.0, mv_hold=200_000.0, ratio=0.7)
    assert bounds.demand == pytest.approx(800_000.0)
    assert bounds.quota == pytest.approx(500_000.0)
    assert 0.0 <= bounds.quota <= bounds.demand
    empty = quota_bounds(gross_target=100_000.0, mv_hold=900_000.0, ratio=0.3)
    assert empty.quota == 0.0 and empty.demand == 0.0


def test_quota_bounds_reject_bad_inputs() -> None:
    with pytest.raises(QuotaError):
        quota_bounds(gross_target=-1.0, mv_hold=0.0, ratio=0.7)
    with pytest.raises(QuotaError):
        quota_bounds(gross_target=1.0, mv_hold=0.0, ratio=0.0)


def test_quota_with_credit_is_signed() -> None:
    """``L`` 有符号：出借方额度被压缩（``ΣL = 0 ⇒ ΣA ≤ ΣR``）。"""
    bounds = quota_bounds(gross_target=1_000_000.0, mv_hold=100_000.0, ratio=0.7)
    assert bounds.with_credit(+50_000.0) == pytest.approx(bounds.quota + 50_000.0)
    assert bounds.with_credit(-50_000.0) == pytest.approx(bounds.quota - 50_000.0)
    assert bounds.with_credit(-10_000_000.0) == 0.0


def _buy_intent(code: str, shares: int, day: str = DAYS[0]) -> Intent:
    return intent_from_plan_row(
        sleeve_id="v20", ts_code=code, side="buy", shares=shares, signal_date=day, due_date=day
    )


def test_authorized_by_cash_reserves_fee_and_rounds_lots() -> None:
    """``A + Fee(A) ≤ cash`` 且整手；现金不足时按整手回退（不放宽）。"""
    intent = _buy_intent("000001.SZ", 10_000)
    priced = [(intent, 10.0)]
    result = authorized_by_cash(
        priced=priced, cash=50_000.0, buy_cost=COST.calculate_buy_cost, lot_size=100, bound=1e9
    )
    shares = int(result["shares"])
    assert shares == 4_900
    assert result["notional"] + result["fee"] <= 50_000.0
    assert result["reason"] == "cash_binding"


def test_authorized_by_cash_zero_when_no_cash() -> None:
    intent = _buy_intent("000001.SZ", 1_000)
    result = authorized_by_cash(
        priced=[(intent, 10.0)], cash=0.0, buy_cost=COST.calculate_buy_cost, lot_size=100, bound=1e9
    )
    assert result["shares"] == 0
    assert result["fee"] == 0.0


def test_decide_quota_marks_no_lot_after_rounding() -> None:
    """``R`` 小于一手 ⇒ 授权 0 股并标 ``quota_no_lot_after_rounding``。"""
    sleeve = SleeveState(sleeve_id="v20", cash=500_000.0, credit=0.0)
    bounds = quota_bounds(gross_target=1_000.0, mv_hold=0.0, ratio=0.7)
    kept, decision = decide_quota(
        trade_date=DAYS[0],
        sleeve=sleeve,
        intents=[_buy_intent("000001.SZ", 1_000)],
        bounds=bounds,
        cash_open=500_000.0,
        credit_granted=0.0,
        credit_outstanding=0.0,
        price_of=lambda _code: 10.0,
        buy_cost=COST.calculate_buy_cost,
        lot_size=100,
    )
    assert decision.authorized_shares == 0
    assert decision.reason == QUOTA_NO_LOT_REASON
    assert decision.denied_by_intent
    assert kept == []


def test_decide_quota_disabled_is_identity() -> None:
    """``quota=off`` 必须逐位返回入参意图（T4 兼容门的前提）。"""
    sleeve = SleeveState(sleeve_id="v20", cash=1_000.0, credit=0.0)
    intent = _buy_intent("000001.SZ", 1_000)
    bounds = quota_bounds(gross_target=0.0, mv_hold=0.0, ratio=0.7)
    kept, decision = decide_quota(
        trade_date=DAYS[0],
        sleeve=sleeve,
        intents=[intent],
        bounds=bounds,
        cash_open=1_000.0,
        credit_granted=0.0,
        credit_outstanding=0.0,
        price_of=lambda _code: 10.0,
        buy_cost=COST.calculate_buy_cost,
        lot_size=100,
        enabled=False,
    )
    assert kept == [intent]
    assert decision.reason == "quota_disabled"


def test_decide_quota_rejects_authorization_beyond_bounds() -> None:
    """``A > min(D, R+L)`` 必须当场拒绝（不是事后告警）。"""
    sleeve = SleeveState(sleeve_id="v20", cash=10_000_000.0, credit=0.0)
    bounds = quota_bounds(gross_target=100_000.0, mv_hold=0.0, ratio=0.7)
    _kept, decision = decide_quota(
        trade_date=DAYS[0],
        sleeve=sleeve,
        intents=[_buy_intent("000001.SZ", 100_000)],
        bounds=bounds,
        cash_open=10_000_000.0,
        credit_granted=0.0,
        credit_outstanding=0.0,
        price_of=lambda _code: 10.0,
        buy_cost=COST.calculate_buy_cost,
        lot_size=100,
    )
    assert decision.authorized_notional <= min(decision.bounds.demand, decision.bounds.quota) + 1e-6
    assert funding_gap(decision, cash_open=10_000_000.0) == 0.0


def test_funding_gap_ignores_credit_outstanding() -> None:
    """缺口只按 ``cash_open`` 计（存量 credit 不得再进入授权）。"""
    sleeve = SleeveState(sleeve_id="v20", cash=50_000.0, credit=200_000.0)
    bounds = quota_bounds(gross_target=1_000_000.0, mv_hold=0.0, ratio=0.7)
    _kept, decision = decide_quota(
        trade_date=DAYS[0],
        sleeve=sleeve,
        intents=[_buy_intent("000001.SZ", 100_000)],
        bounds=bounds,
        cash_open=50_000.0,
        credit_granted=0.0,
        credit_outstanding=200_000.0,
        price_of=lambda _code: 10.0,
        buy_cost=COST.calculate_buy_cost,
        lot_size=100,
    )
    # 存量 credit 不参与授权 ⇒ 授权只到现金可负担的 4,900 股
    assert decision.authorized_shares == 4_900
    assert funding_gap(decision, cash_open=50_000.0) == 0.0


# ---------------------------------------------------------------- 借用纯函数


def test_allocate_credit_splits_by_idle_ratio_and_orders_borrowers() -> None:
    """多出借方按闲置比例分摊；多借方按缺口降序 + ID 升序决胜。"""
    grants, diag = allocate_credit(
        requests={"v20": 100.0, "v10": 300.0},
        idle={"a": 200.0, "b": 200.0},
        due_date="20240201",
    )
    assert diag["borrower_order"] == ["v10", "v20"]
    by_borrower: Dict[str, float] = {}
    for grant in grants:
        by_borrower[grant.borrower] = by_borrower.get(grant.borrower, 0.0) + grant.amount
    assert by_borrower["v10"] == pytest.approx(300.0)
    assert by_borrower["v20"] == pytest.approx(100.0)
    # 闲置比例分摊：两户各 200 ⇒ 各承担一半
    per_lender: Dict[str, float] = {}
    for grant in grants:
        per_lender[grant.lender] = per_lender.get(grant.lender, 0.0) + grant.amount
    assert per_lender["a"] == pytest.approx(200.0)
    assert per_lender["b"] == pytest.approx(200.0)


def test_allocate_credit_reports_unmet_when_no_idle() -> None:
    grants, diag = allocate_credit(
        requests={"v20": 500.0}, idle={"v20": 1_000.0}, due_date="20240201"
    )
    assert grants == []
    assert diag["unmet"]["v20"] == pytest.approx(500.0)


def test_borrow_lot_partial_return_and_late_recovery() -> None:
    lot = BorrowLot(
        borrow_id="L1",
        lender="v10",
        borrower="v20",
        amount=1_000.0,
        grant_date="20240102",
        due_date="20240103",
    )
    partial, event = lot.repay(400.0, "20240103")
    assert event["kind"] == "repay_partial"
    assert partial.outstanding == pytest.approx(600.0)
    assert partial.state == OUTSTANDING
    dead, dead_event = partial.deadlock("20240103")
    assert dead.state == DEADLOCKED and dead_event["outstanding"] == pytest.approx(600.0)
    closed, late_event = dead.repay(600.0, "20240105")
    assert closed.state == RETURNED
    assert late_event["kind"] == "repay_late_recovery"
    assert closed.late_recovery is True


def test_borrow_lot_rejects_overpayment_and_self_lending() -> None:
    lot = BorrowLot(
        borrow_id="L1",
        lender="v10",
        borrower="v20",
        amount=100.0,
        grant_date="20240102",
        due_date="20240103",
    )
    with pytest.raises(BorrowError):
        lot.repay(101.0, "20240103")
    with pytest.raises(BorrowError):
        BorrowLot(
            borrow_id="L2",
            lender="v20",
            borrower="v20",
            amount=10.0,
            grant_date="20240102",
            due_date="20240103",
        )


def test_next_due_date_is_next_trading_day_and_right_censored_at_end() -> None:
    assert next_due_date(DAYS, DAYS[0]) == DAYS[1]
    assert next_due_date(DAYS, DAYS[-1]) == ""


def test_principal_offsets_are_paired() -> None:
    lots = [
        BorrowLot(
            borrow_id="L1",
            lender="v10",
            borrower="v20",
            amount=100.0,
            grant_date="20240102",
            due_date="20240103",
        ),
        BorrowLot(
            borrow_id="L2",
            lender="v10",
            borrower="v20",
            amount=50.0,
            grant_date="20240103",
            due_date="20240104",
            returned_amount=50.0,
            state=RETURNED,
        ),
    ]
    payload = principal_offsets(lots)
    assert payload["balanced"] is True
    assert payload["borrowed_total"] == pytest.approx(100.0)
    assert payload["lent_total"] == pytest.approx(100.0)
    assert fifo_order(lots)[0].borrow_id == "L1"


def test_service_credit_day_recalls_repays_and_deadlocks() -> None:
    book = CreditBook()
    book.apply_grants(
        day=DAYS[0],
        grants=[CreditGrant(lender="v10", borrower="v20", amount=1_000.0, due_date=DAYS[1])],
    )
    # 到期日只有 300 现金 ⇒ 部分归还 + 余额死局
    service = service_credit_day(book=book, day=DAYS[1], cash={"v20": 300.0, "v10": 0.0})
    assert len(service["recall_queue"]) == 1
    assert service["recall_queue"][0]["borrow_id"] == book.lots[0].borrow_id
    assert service["recall_queue"][0]["outstanding"] == pytest.approx(1_000.0)
    assert service["cash"]["v20"] == pytest.approx(0.0)
    assert service["cash"]["v10"] == pytest.approx(300.0)
    assert service["repayments"][0]["kind"] == "repay_partial"
    assert service["deadlocks"][0]["kind"] == "deadlock"
    assert book.lots[0].state == DEADLOCKED
    # 后续自然现金归还 ⇒ late_recovery
    later = service_credit_day(book=book, day=DAYS[2], cash={"v20": 1_000.0, "v10": 0.0})
    assert later["late_recoveries"][0]["kind"] == "repay_late_recovery"
    assert book.lots[0].outstanding == pytest.approx(0.0)


def test_service_credit_day_skips_right_censored_lots() -> None:
    book = CreditBook()
    book.apply_grants(
        day=DAYS[0],
        grants=[CreditGrant(lender="v10", borrower="v20", amount=500.0, due_date="")],
    )
    service = service_credit_day(book=book, day=DAYS[1], cash={"v20": 0.0, "v10": 0.0})
    assert service["deadlocks"] == []
    assert service["right_censored"][0]["borrow_id"] == book.lots[0].borrow_id


def test_credit_book_balances_exactly_zero() -> None:
    book = CreditBook()
    book.apply_grants(
        day=DAYS[0],
        grants=[
            CreditGrant(lender="v10", borrower="v20", amount=1_234.5678, due_date=DAYS[1]),
            CreditGrant(lender="v10", borrower="v20", amount=87.65, due_date=DAYS[1]),
        ],
    )
    payload = book.assert_balanced(("v20", "v10"))
    assert sum(payload["credit_by_sleeve"].values()) == 0.0
    assert payload["credit_by_sleeve"]["v20"] == pytest.approx(1_322.2178)
    assert payload["credit_by_sleeve"]["v10"] == pytest.approx(-1_322.2178)
    assert payload["residue_adjustment"] == 0.0


def test_credit_book_restore_roundtrip() -> None:
    book = CreditBook()
    book.apply_grants(
        day=DAYS[0],
        grants=[CreditGrant(lender="v10", borrower="v20", amount=100.0, due_date=DAYS[1])],
    )
    restored = CreditBook.restore(json.loads(json.dumps(book.snapshot(), ensure_ascii=False)))
    assert restored.digest() == book.digest()
    with pytest.raises(CreditError):
        CreditBook.restore({"schema": "p7_t4_state/v2"})


# ---------------------------------------------------------------- 信封与非负门


def test_t5_envelope_rejects_t4_schema() -> None:
    envelope = T5StateEnvelope(run_id="x")
    with pytest.raises(Exception):
        envelope.restore_from({"schema": "p7_t4_state/v2", "generation": 0})


def test_t5_envelope_same_key_same_generation_checks_credit_state() -> None:
    """同键同代还必须比对 credit 台账（不能把不同 credit 态当无操作）。"""
    envelope = T5StateEnvelope(run_id="x")
    sleeves = {"v20": SleeveState(sleeve_id="v20", cash=100.0, credit=0.0)}
    physical = _physical(100.0)
    envelope.commit(
        key="k",
        generation=1,
        fingerprint="f",
        sleeves=sleeves,
        physical=physical,
        as_of=DAYS[0],
        resume_state={"credit_state": {"schema": "p7_t5_credit/v1", "lots": []}},
    )
    with pytest.raises(Exception):
        envelope.commit(
            key="k",
            generation=1,
            fingerprint="f",
            sleeves=sleeves,
            physical=physical,
            as_of=DAYS[0],
            resume_state={
                "credit_state": {
                    "schema": "p7_t5_credit/v1",
                    "lots": [{"borrow_id": "L1", "amount": 10.0}],
                }
            },
        )


def test_t5_envelope_payload_uses_own_schema_and_keeps_t4_view() -> None:
    envelope = T5StateEnvelope(run_id="x")
    envelope.commit(
        key="k",
        generation=1,
        fingerprint="f",
        sleeves={"v20": SleeveState(sleeve_id="v20", cash=1.0, credit=0.0)},
        physical=_physical(1.0),
        as_of=DAYS[0],
    )
    payload = envelope.to_payload()
    assert payload["schema"] == "p7_t5_state/v1"
    assert payload["t4_compatible_schema"] == "p7_t4_state/v2"
    assert set(envelope.state_payload()) == {"as_of", "sleeves", "physical"}


def _physical(cash: float) -> Any:
    from scripts.v2_p7.t4.state import PhysicalBook

    return PhysicalBook(cash=float(cash))


def test_nonnegative_gate_raises_on_violation() -> None:
    """``cash_i + credit_i < 0`` 必须拒绝（负现金没有 credit 背书）。"""
    rows = sleeve_balances(
        trade_date=DAYS[0],
        sleeves={"v20": SleeveState(sleeve_id="v20", cash=-100.0, credit=50.0)},
        prices=_StubPrices(),
    )
    with pytest.raises(Exception):
        assert_nonnegative_gate(rows)
    ok_rows = sleeve_balances(
        trade_date=DAYS[0],
        sleeves={"v20": SleeveState(sleeve_id="v20", cash=-100.0, credit=200.0)},
        prices=_StubPrices(),
    )
    payload = assert_nonnegative_gate(ok_rows)
    assert payload["ok"] is True
    assert payload["min_cash_plus_credit"] == pytest.approx(100.0)


class _StubPrices:
    def valuation(self, ts_code: str, buy_price: float) -> Tuple[float, str, str]:
        del ts_code
        return float(buy_price), "", "buy_price"

    def market_open(self, ts_code: str) -> Any:
        del ts_code
        raise AssertionError("本用例不应查询开盘价")


def test_zero_pnl_check_detects_principal_movement() -> None:
    prices = _StubPrices()
    before = sleeve_balances(
        trade_date=DAYS[0],
        sleeves={"v20": SleeveState(sleeve_id="v20", cash=100.0, credit=0.0)},
        prices=prices,
    )
    # 借入 50：cash 100→150、credit 0→50 ⇒ E 不变（本金转移零 P&L）
    after = sleeve_balances(
        trade_date=DAYS[0],
        sleeves={"v20": SleeveState(sleeve_id="v20", cash=150.0, credit=50.0)},
        prices=prices,
    )
    assert zero_pnl_check(before=before, after=after)["ok"] is True
    drifted = sleeve_balances(
        trade_date=DAYS[0],
        sleeves={"v20": SleeveState(sleeve_id="v20", cash=160.0, credit=50.0)},
        prices=prices,
    )
    assert zero_pnl_check(before=before, after=drifted)["ok"] is False


# ---------------------------------------------------------------- 宿主行为


def _parity_setup() -> Tuple[Any, List[Tuple[str, str, str, int, str]]]:
    params = _Params()
    plan = [
        (DAYS[0], "v20", "000001.SZ", 1_000, DAYS[4]),
        (DAYS[0], "v10", "000001.SZ", 1_000, DAYS[4]),
        (DAYS[2], "v20", "000002.SZ", 500, DAYS[6]),
    ]
    return params, plan


def test_reference_mode_matches_t4_host_bitwise() -> None:
    """T4 兼容门：``credit=off & quota=off`` 与 T4 宿主逐位一致。"""
    params, plan = _parity_setup()
    prices = {"000001.SZ": 10.0, "000002.SZ": 20.0}
    buys, sells = _intents(plan)
    loader = _Loader(prices)
    t4 = HistoricalVirtualSleeveHost(
        params=params,
        run_id="parity",
        prices=PriceBook(loader),
        cost_model=COST,
        buy_index=buys,
        sell_index=sells,
        price_factory=lambda: PriceBook(loader),
    ).run()
    t5 = _run(params=params, prices=prices, plan=plan, mode=MODE_T4_PARITY)
    assert [a[1] for a in t4.digest_series] == [b[1] for b in t5.digest_series]
    assert [a[1] for a in t4.fee_series] == [b[1] for b in t5.fee_series]
    assert t4.nav_series == t5.nav_series
    assert t4.daily_rows == t5.daily_rows
    assert t4.conservation_rows == t5.conservation_rows
    assert t4.trades == t5.trades
    assert t4.envelope.state_digest() == t5.envelope.state_digest()
    assert t5.credit_payload["totals"]["lots"] == 0
    assert t5.quota_rows == []


def test_vvh_keeps_credit_inert_and_records_requests() -> None:
    """VV-H：同配额、禁 credit —— 无 lot、有借用申请登记，逐袖非负门通过。"""
    params = _Params(target=4_000_000.0)
    plan = [
        (DAYS[0], "v20", "000001.SZ", 45_000, DAYS[6]),
        (DAYS[1], "v20", "000001.SZ", 20_000, DAYS[7]),
    ]
    result = _run(
        params=params, prices={"000001.SZ": 10.0, "000002.SZ": 20.0}, plan=plan, mode=MODE_VVH
    )
    assert result.credit_payload["totals"]["lots"] == 0
    assert result.credit_balance["credit_by_sleeve"] == {"v10": 0.0, "v20": 0.0}
    statuses = {item["status"] for item in result.credit_payload["requests"]}
    assert statuses <= {"credit_disabled", "below_one_lot", "blocked_no_new_buy"}
    assert result.gate_payload["ok"] is True
    assert result.gate_payload["violation_days"] == 0
    assert result.quota_rows  # 配额实际生效（决策逐日登记）


def test_vvs_borrows_deadlocks_and_recovers_late() -> None:
    """VV-S：授信 → T+1 召回 → 死局 → 自然卖出迟回收，且 Σcredit=0。"""
    params = _Params(target=4_000_000.0)
    plan = [
        (DAYS[0], "v20", "000001.SZ", 45_000, DAYS[6]),
        (DAYS[1], "v20", "000001.SZ", 20_000, DAYS[7]),
    ]
    result = _run(
        params=params, prices={"000001.SZ": 10.0, "000002.SZ": 20.0}, plan=plan, mode=MODE_VVS
    )
    events = [item["kind"] for item in result.credit_payload["events"]]
    assert "grant" in events and "deadlock" in events
    assert "repay_late_recovery" in events
    assert result.credit_balance["principal"]["balanced"] is True
    assert sum(result.credit_balance["credit_by_sleeve"].values()) == 0.0
    assert result.gate_payload["ok"] is True
    assert not result.zero_pnl["violations"]
    # 出借方在借出期间 credit 为负、借入方为正（符号约定）
    credits = [float(row["credit"]) for row in result.balance_rows]
    assert min(credits) < 0.0 < max(credits)


def test_vvs_quota_capping_registers_voids_and_conservation() -> None:
    """配额拒绝登记为日终作废，且逐日/跨日守恒零违例。"""
    params = _Params(target=1_000_000.0)
    plan = [(DAYS[0], "v20", "000001.SZ", 90_000, DAYS[8])]
    result = _run(
        params=params, prices={"000001.SZ": 10.0, "000002.SZ": 20.0}, plan=plan, mode=MODE_VVH
    )
    assert int(result.voids.get("quota_capped_voided_eod", 0)) > 0
    assert result.conservation["violations"] == 0
    assert result.cross_day["violations"] == 0
    row = next(item for item in result.quota_rows if item["wanted_notional"] > 0)
    assert row["authorized_shares"] < row["wanted_shares"]
    assert row["quota"] < row["demand"]


def test_host_restart_covers_credit_states() -> None:
    """续跑覆盖「借款未还中 / 召回当日 / deadlocked 中」三类 credit 检查点。"""
    params = _Params(target=4_000_000.0)
    plan = [
        (DAYS[0], "v20", "000001.SZ", 45_000, DAYS[6]),
        (DAYS[1], "v20", "000001.SZ", 20_000, DAYS[7]),
    ]
    result = _run(
        params=params, prices={"000001.SZ": 10.0, "000002.SZ": 20.0}, plan=plan, mode=MODE_VVS
    )
    restart = result.restart
    assert restart["checked"] is True
    assert restart["credit_all_identical"] is True
    coverage = restart["coverage"]
    assert coverage["borrow_outstanding"] is True
    assert coverage["recall_due"] is True
    assert coverage["deadlocked"] is True
    assert restart["idempotency"]["cross_generation_rejected"] is True


def test_host_reference_mode_still_passes_nonnegative_gate() -> None:
    params, plan = _parity_setup()
    result = _run(
        params=params,
        prices={"000001.SZ": 10.0, "000002.SZ": 20.0},
        plan=plan,
        mode=MODE_T4_PARITY,
    )
    assert result.gate_payload["ok"] is True
    assert result.gate_payload["violation_days"] == 0
    assert all(float(row["credit"]) == 0.0 for row in result.balance_rows)


def test_holdings_market_value_uses_open_price() -> None:
    class _Prices:
        def market_open(self, ts_code: str) -> Any:
            from scripts.v2_p7.t4.price import ExecutionLookup

            return ExecutionLookup(11.0 if ts_code == "000001.SZ" else 21.0, "ok")

        def valuation(self, ts_code: str, buy_price: float) -> Tuple[float, str, str]:
            del ts_code
            return float(buy_price), "", "buy_price"

    from scripts.v2_p7.t4.state import P7Lot

    lot = P7Lot(
        virtual_lot_id="l1",
        ts_code="000001.SZ",
        source_lot_id="s1",
        shares=100,
        acquired_date=DAYS[0],
        acquired_price=10.0,
        acquired_fee=0.0,
        buy_date=DAYS[0],
        due_date=DAYS[5],
        signal_date=DAYS[0],
        origin_model_version=1,
    )
    sleeve = SleeveState(sleeve_id="v20", cash=0.0, credit=0.0, lots=(lot,))
    value, detail = holdings_market_value(sleeve, _Prices())
    assert value == pytest.approx(1_100.0)
    assert detail["fallback_count"] == 0


# ---------------------------------------------------------------- T4 互证比较口径


def test_parity_diff_treats_missing_and_nan_as_equal() -> None:
    """Parquet 往返会给缺席字段补 NaN：比较时必须按「不存在」处理（不得误判）。"""
    from scripts.v2_p7.t5.runner import _diff_rows

    reference = [{"trade_date": "20240102", "filled": 0, "void_reason": float("nan")}]
    ours = [{"trade_date": "20240102", "filled": 0}]
    assert _diff_rows(reference, ours) == []


def test_parity_diff_detects_real_value_change() -> None:
    """真实数值变化必须被抓出（并给出定位）。"""
    from scripts.v2_p7.t5.runner import _diff_rows

    reference = [{"trade_date": "20240102", "filled": 100}]
    ours = [{"trade_date": "20240102", "filled": 200}]
    mismatched = _diff_rows(reference, ours)
    assert len(mismatched) == 1
    assert mismatched[0]["pos"] == 0


def test_parity_diff_treats_integral_float_as_int() -> None:
    """Parquet 混合列会把整数升成 float：整值浮点必须与整数等价（不得误判）。"""
    from scripts.v2_p7.t5.runner import _diff_rows

    reference = [{"trade_date": "20240102", "truncated_from": 3500.0}]
    ours = [{"trade_date": "20240102", "truncated_from": 3500}]
    assert _diff_rows(reference, ours) == []


def test_zero_share_authorization_charges_no_fee() -> None:
    """``A=0`` 必须 ``F=0``（``buy_cost(0)`` 会返回最小佣金，不得据此派生幽灵缺口）。"""
    intent = _buy_intent("000001.SZ", 1_000)
    result = authorized_by_cash(
        priced=[(intent, 10.0)],
        cash=50.0,
        buy_cost=COST.calculate_buy_cost,
        lot_size=100,
        bound=1e9,
    )
    assert result["shares"] == 0
    assert result["notional"] == 0.0
    assert result["fee"] == 0.0


def test_holdings_market_value_never_uses_same_day_close() -> None:
    """``MV_hold`` 回退不得取当日收盘（开盘时点不可知），只允许最后已知价/取得价。"""

    class _Prices:
        def market_open(self, ts_code: str) -> Any:
            from scripts.v2_p7.t4.price import ExecutionLookup

            return ExecutionLookup(None, "no_valid_open")

        def last_known(self, ts_code: str) -> Tuple[float, str]:
            return 9.0, "20231229"

        def valuation(self, ts_code: str, buy_price: float) -> Tuple[float, str, str]:
            raise AssertionError("MV_hold 不得走当日收盘估值链")

    from scripts.v2_p7.t4.state import P7Lot

    lot = P7Lot(
        virtual_lot_id="l1",
        ts_code="000001.SZ",
        source_lot_id="s1",
        shares=100,
        acquired_date=DAYS[0],
        acquired_price=10.0,
        acquired_fee=0.0,
        buy_date=DAYS[0],
        due_date=DAYS[5],
        signal_date=DAYS[0],
        origin_model_version=1,
    )
    sleeve = SleeveState(sleeve_id="v20", cash=0.0, credit=0.0, lots=(lot,))
    value, detail = holdings_market_value(sleeve, _Prices())
    assert value == pytest.approx(900.0)
    assert detail["fallback"][0]["source"] == "last_known"


# ---------------------------------------------------------- Codex 复审（2026-11-13）


def test_denials_merge_handles_multiple_intents() -> None:
    """C-T5-04：多笔被拒意图必须逐行并入，不得只处理首行后退出。"""
    from scripts.v2_p7.t5.denials import _append_denial_rows, _apply_denials_to_rows

    class _Outcome:
        def __init__(self) -> None:
            self.conservation_rows = []

    rows = [
        {"trade_date": "20240102", "intent_key": "k1", "original": 100, "filled": 100, "voided": 0},
        {"trade_date": "20240102", "intent_key": "k2", "original": 100, "filled": 100, "voided": 0},
    ]
    voids: dict = {}
    seen = _apply_denials_to_rows(rows, {"k1": 100, "k2": 100}, {}, voids)
    outcome = _Outcome()
    _append_denial_rows("20240102", outcome, {"k1": 100, "k2": 100}, {}, seen)
    assert seen == {"k1", "k2"}
    assert outcome.conservation_rows == []
    assert [row["original"] for row in rows] == [200, 200]
    assert [row["voided"] for row in rows] == [100, 100]


def test_overdue_partial_repayment_counts_as_late_recovery() -> None:
    """C-T5-06：逾期**部分**归还也必须计入迟回收（不只结清那一笔）。"""
    from scripts.v2_p7.t5.borrow import BorrowLot

    lot = BorrowLot(
        borrow_id="L1",
        borrower="v20",
        lender="v10",
        amount=1_000.0,
        grant_date=DAYS[0],
        due_date=DAYS[1],
    )
    partial, first = lot.repay(400.0, DAYS[2])
    assert first["kind"] == "repay_late_recovery"
    assert first["late_recovery"] is True
    assert partial.outstanding == pytest.approx(600.0)
    rest, second = partial.repay(600.0, DAYS[3])
    assert second["late_recovery"] is True
    assert rest.outstanding == pytest.approx(0.0)


def test_price_source_failure_blocks_publish() -> None:
    """C-T5-07：价源门失败必须进入硬失败集合（禁止发布）。"""
    from scripts.v2_p7.t5.checks import _stop_on_hard_failures
    from scripts.v2_p7.t5.errors import RunnerError

    ok = {"ok": True, "failures": []}
    checks = {
        key: dict(ok)
        for key in (
            "quota",
            "authorization",
            "credit_accounting",
            "price_source",
            "nonnegative_gate",
            "identities",
            "constructed_cases",
            "identity_restart",
            "legacy_sha",
            "conservation_unique",
            "denial_double_count",
            "credit_state_substitutes",
        )
    }
    checks["t4_parity"] = {"ok": True}
    checks["vvh_reference"] = {"ok": True}
    checks["nonnegative_gate"] = {"ok": True, "violation_days": 0, "min_cash_plus_credit": 0.0}
    checks["identities"] = {"ok": True, "failure_days": 0}
    checks["constructed_cases"] = {"ok": True, "failed": []}
    checks["legacy_sha"] = {"ok": True, "modified_in_worktree": []}
    _stop_on_hard_failures(checks)
    checks["price_source"] = {"ok": False, "failures": ["no_valid_open"]}
    with pytest.raises(RunnerError):
        _stop_on_hard_failures(checks)


def test_conservation_unique_gate_detects_duplicate_intent_rows() -> None:
    """C-T5-04：同一意图当日出现两行必须被门判失败。"""
    from scripts.v2_p7.t5.checks import _conservation_unique

    class _Result:
        def __init__(self, rows: list) -> None:
            self.conservation_rows = rows

    clean = {
        "vvs": _Result(
            [
                {"trade_date": "20240102", "intent_key": "k1"},
                {"trade_date": "20240102", "intent_key": "k2"},
            ]
        )
    }
    assert _conservation_unique(clean)["ok"] is True
    dirty = {
        "vvs": _Result(
            [
                {"trade_date": "20240102", "intent_key": "k1"},
                {"trade_date": "20240102", "intent_key": "k1"},
            ]
        )
    }
    payload = _conservation_unique(dirty)
    # 行重数是诊断量（同意图可按腿拆行），不做硬门；但必须如实计数
    assert payload["duplicate_rows_total"] == 1
    assert payload["arms"]["vvs"]["duplicate_rows"] == 1


def test_denial_double_count_gate_blocks_double_merge() -> None:
    """C-T5-04：同键两行都带 voided>0（重复并入）必须阻断发布。"""
    from scripts.v2_p7.t5.checks import _denial_double_count

    class _Result:
        def __init__(self, rows: list) -> None:
            self.conservation_rows = rows

    clean = {
        "vvs": _Result(
            [
                {"trade_date": "20240102", "intent_key": "k1", "voided": 100},
                {"trade_date": "20240102", "intent_key": "k2", "voided": 0},
            ]
        )
    }
    assert _denial_double_count(clean)["ok"] is True
    dirty = {
        "vvs": _Result(
            [
                {"trade_date": "20240102", "intent_key": "k1", "voided": 100},
                {"trade_date": "20240102", "intent_key": "k1", "voided": 100},
            ]
        )
    }
    payload = _denial_double_count(dirty)
    assert payload["ok"] is False
    assert payload["arms"]["vvs"]["multi_row_keys"] == 1


def test_denied_split_clears_fully_released_denials() -> None:
    """R1-03：授信完全解除某键的拒绝后，授权前拒绝量必须清零。"""

    class _Decision:
        def __init__(self, denied: dict) -> None:
            self.denied_by_intent = denied

    from scripts.v2_p7.t5.denials import _denied_split

    quota_only = {"v20": _Decision({"k1": 300})}
    assert _denied_split(quota_only, {"v20": _Decision({})}) == ({}, {})
    quota, cash = _denied_split(quota_only, {"v20": _Decision({"k1": 120})})
    assert quota == {"k1": 120}
    assert cash == {}
