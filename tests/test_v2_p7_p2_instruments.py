# -*- coding: utf-8 -*-
"""P7 P2 共用计量工具合成探针（不读真实行情/候选收益）。"""

from decimal import Decimal

import pytest

from scripts.v2_p7.p2.account import (
    AccountError,
    InsufficientCash,
    OrderStatus,
    ResearchAccount,
)
from scripts.v2_p7.p2.announcements import (
    AnnouncementStore,
    AnnouncementStoreError,
)
from scripts.v2_p7.p2.benchmark import FrozenWeightBasket, equal_weights
from scripts.v2_p7.p2.missing import (
    MissingPriceConflict,
    MissingPriceLedger,
    MissingPriceRecord,
)
from scripts.v2_p7.p2.types import (
    InstrumentError,
    MarketSnapshot,
    PriceObservation,
    PriceStatus,
)


def ok(price: str, source: str = "synthetic") -> PriceObservation:
    return PriceObservation(PriceStatus.OK, Decimal(price), source=source)


def missing(status: PriceStatus, source: str = "synthetic") -> PriceObservation:
    return PriceObservation(status, source=source)


def snapshot(day: str, quotes: dict[str, PriceObservation]) -> MarketSnapshot:
    return MarketSnapshot(day, quotes, source="synthetic")


def buy_100_at_10(account: ResearchAccount) -> None:
    account.mark(snapshot("20240102", {}))
    account.submit_order("B1", "AAA", "buy", 100, "20240102", "20240103")
    result = account.execute_fill("B1", trade_date="20240103", quantity=100, trade_price=10, fee=5)
    assert result.order_completed is True
    assert result.position_closed is False


def test_share_t1_blocks_same_day_sell_of_new_buy():
    account = ResearchAccount(100_000)
    buy_100_at_10(account)
    account.submit_order("S1", "AAA", "sell", 100, "20240102", "20240103")
    with pytest.raises(AccountError, match="股份 T\\+1"):
        account.execute_fill("S1", trade_date="20240103", quantity=100, trade_price=11)


def test_stock_dividend_adjusts_carry_price_and_cash():
    account = ResearchAccount(100_000)
    buy_100_at_10(account)
    account.apply_stock_dividend("S1", "AAA", 0.2, "20240104", 0.18)
    position = account.position("AAA")
    assert position.shares == 120
    assert position.last_price == Decimal("8.1833")
    assert position.last_price_date == "20240104"


def test_mark_never_uses_future_price_and_missing_keeps_last_valid():
    account = ResearchAccount(100_000)
    buy_100_at_10(account)
    day_one = account.mark(snapshot("20240103", {"AAA": ok("10")}))
    assert day_one.nav == Decimal("99995.00")

    day_two = account.mark(snapshot("20240104", {"AAA": missing(PriceStatus.SUSPENDED)}))
    assert day_two.nav == Decimal("99995.00")
    assert day_two.market_value == Decimal("1000.00")
    assert day_two.unresolved_weight == pytest.approx(1000 / 99995, abs=1e-4)
    assert len(day_two.missing_records) == 1
    assert day_two.missing_records[0].reason is PriceStatus.SUSPENDED

    day_three = account.mark(snapshot("20240105", {"AAA": ok("12")}))
    assert day_three.nav == Decimal("100195.00")
    assert day_three.missing_records == ()
    assert account.position("AAA").last_price_date == "20240105"


def test_partial_sell_order_completed_is_separate_from_position_closed():
    account = ResearchAccount(100_000)
    buy_100_at_10(account)
    account.mark(snapshot("20240103", {"AAA": ok("10")}))
    account.submit_order("S1", "AAA", "sell", 100, "20240103", "20240104")

    first = account.execute_fill("S1", trade_date="20240104", quantity=60, trade_price=10)
    assert first.order_completed is False
    assert first.position_closed is False
    assert first.position_shares_after == 40
    assert account.order("S1")["status"] == OrderStatus.PENDING.value
    account.mark(snapshot("20240104", {"AAA": ok("10")}))

    second = account.execute_fill("S1", trade_date="20240105", quantity=40, trade_price=10)
    assert second.order_completed is True
    assert second.position_closed is True
    assert second.position_shares_after == 0
    assert account.order("S1")["status"] == OrderStatus.FILLED.value
    assert account.position("AAA") is None
    assert account.active_orders() == ()
    account.mark(snapshot("20240105", {}))


def test_expired_partial_order_keeps_position_and_does_not_freeze_account():
    account = ResearchAccount(100_000)
    buy_100_at_10(account)
    account.mark(snapshot("20240103", {"AAA": ok("10")}))
    account.submit_order("S1", "AAA", "sell", 100, "20240103", "20240104")
    account.execute_fill("S1", trade_date="20240104", quantity=60, trade_price=10)
    account.expire_order("S1", reason="retry_limit")
    assert account.order("S1")["status"] == OrderStatus.EXPIRED.value
    assert account.position("AAA").shares == 40
    mark = account.mark(snapshot("20240104", {"AAA": ok("10")}))
    assert mark.nav > 0
    assert account.active_orders() == ()


def test_t1_timeline_lot_and_insufficient_cash_are_zero_side_effect():
    account = ResearchAccount(1_000)
    account.mark(snapshot("20240102", {}))
    with pytest.raises(AccountError, match="整数倍"):
        account.submit_order("BAD", "AAA", "buy", 150, "20240102", "20240103")
    account.submit_order("B1", "AAA", "buy", 100, "20240102", "20240103")
    with pytest.raises(AccountError, match="早于最早可交易日"):
        account.execute_fill("B1", trade_date="20240102", quantity=100, trade_price=10)
    with pytest.raises(InsufficientCash):
        account.execute_fill("B1", trade_date="20240103", quantity=100, trade_price=11)
    assert account.cash == Decimal("1000.00")
    assert account.position("AAA") is None
    assert account.order("B1")["status"] == OrderStatus.PENDING.value


def test_actual_cash_receivable_and_nav_are_separate_legs():
    account = ResearchAccount(1_000)
    account.mark(snapshot("20240102", {}))
    account.add_receivable("DIV1", "AAA", "100", "20240103", "20240104")
    day_two = account.mark(snapshot("20240103", {}))
    assert day_two.cash == Decimal("1000.00")
    assert day_two.receivable == Decimal("100.00")
    assert day_two.nav == Decimal("1100.00")
    day_three = account.mark(snapshot("20240104", {}))
    assert day_three.cash == Decimal("1100.00")
    assert day_three.receivable == Decimal("0.00")
    assert day_three.nav == Decimal("1100.00")
    with pytest.raises(AccountError, match="历史重复"):
        account.add_receivable("DIV1", "AAA", "100", "20240105", "20240106")


def _missing_record(day: str, code: str, value: str, streak: int = 1) -> MissingPriceRecord:
    return MissingPriceRecord(
        as_of=day,
        ts_code=code,
        reason=PriceStatus.NO_QUOTE,
        shares=100,
        carried_value=Decimal(value),
        last_price_date="20240101",
        days_since_last_quote=int(day[-2:]),
        missing_streak_days=streak,
        nav=Decimal("10000.00"),
        unresolved_weight=0.1,
        source="synthetic",
    )


def test_missing_price_ledger_dedupes_and_orders_by_amount_times_duration():
    ledger = MissingPriceLedger()
    first = _missing_record("20240102", "AAA", "1000")
    stored, duplicate = ledger.record(first)
    assert stored == first
    assert duplicate is False
    _, duplicate = ledger.record(first)
    assert duplicate is True
    assert len(ledger) == 1

    ledger.record(_missing_record("20240103", "AAA", "1000", streak=2))
    ledger.record(_missing_record("20240102", "BBB", "2000"))
    ledger.record(_missing_record("20240103", "BBB", "2000", streak=2))
    rows = ledger.top_sources()
    assert rows[0]["ts_code"] == "BBB"
    assert rows[0]["value_days"] == Decimal("4000.00")
    assert rows[0]["max_missing_streak_days"] == 2
    assert ledger.duplicate_hits == 1
    assert ledger.to_rows()[0]["reason"] == PriceStatus.NO_QUOTE.value

    with pytest.raises(MissingPriceConflict):
        ledger.record(_missing_record("20240102", "AAA", "999"))


def test_frozen_basket_keeps_weight_and_does_not_use_future_or_reweight():
    quotes = {"AAA": ok("10"), "BBB": ok("20")}
    basket = FrozenWeightBasket.freeze(
        as_of="20240102",
        weights=equal_weights(["AAA", "BBB"]),
        base_snapshot=snapshot("20240102", quotes),
    )
    day_two = basket.mark(
        snapshot(
            "20240103",
            {"AAA": missing(PriceStatus.SUSPENDED), "BBB": ok("20.01")},
        )
    )
    assert day_two.nav == Decimal("300075.00")
    assert day_two.return_since_freeze == Decimal("0.00025")
    assert day_two.missing_records[0].shares is None
    assert basket.weights == {"AAA": Decimal("0.5"), "BBB": Decimal("0.5")}

    day_three = basket.mark(snapshot("20240104", {"AAA": ok("30"), "BBB": ok("20")}))
    assert day_three.nav == Decimal("600000.00")
    assert day_three.return_since_freeze == Decimal("1.00")
    with pytest.raises(InstrumentError, match="必须递增"):
        basket.mark(snapshot("20240103", {"AAA": ok("30"), "BBB": ok("20")}))


def test_announcement_store_append_only_versions_and_pit_observation(tmp_path):
    store = AnnouncementStore(tmp_path)
    attachment = tmp_path / "名单.txt"
    attachment.write_text("AAA\nBBB\n", encoding="utf-8")
    first = store.capture(
        source="csindex",
        notice_id="100",
        publish_date="20240101",
        captured_at="20240102",
        payload={"items": ["AAA"]},
        attachments=[attachment],
    )
    assert first.record.version == 1
    assert first.duplicate is False
    assert first.verification["ok"] is True
    first_bytes = (tmp_path / first.record.payload_path).read_bytes()

    duplicate = store.capture(
        source="csindex",
        notice_id="100",
        publish_date="20240101",
        captured_at="20240103",
        payload={"items": ["AAA"]},
        attachments=[attachment],
    )
    assert duplicate.duplicate is True
    assert duplicate.record.version == 1
    assert len(store.versions(source="csindex", notice_id="100")) == 2

    second = store.capture(
        source="csindex",
        notice_id="100",
        publish_date="20240104",
        captured_at="20240104T12:00:00+00:00",
        payload={"items": ["AAA", "CCC"]},
        attachments=[attachment],
    )
    assert second.record.version == 2
    assert (tmp_path / first.record.payload_path).read_bytes() == first_bytes
    assert (
        store.latest_as_published(source="csindex", notice_id="100", as_of="20240101").version == 1
    )
    assert store.latest_as_observed(source="csindex", notice_id="100", as_of="20240101") is None
    assert (
        store.latest_as_observed(source="csindex", notice_id="100", as_of="20240103").version == 1
    )
    assert (
        store.latest_as_observed(source="csindex", notice_id="100", as_of="20240104").version == 2
    )
    assert store.verify()["ok"] is True

    with pytest.raises(AnnouncementStoreError):
        store.capture(
            source="../escape",
            notice_id="100",
            publish_date="20240101",
            captured_at="20240102",
            payload={"items": []},
        )

    (tmp_path / first.record.payload_path).write_text("{}", encoding="utf-8")
    tampered = store.verify()
    assert tampered["ok"] is False
    assert any("正文哈希不符" in item for item in tampered["errors"])
