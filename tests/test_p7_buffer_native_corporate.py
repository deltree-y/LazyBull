"""固定续持研究公司行为适配的合成回归，不读取真实配置或行情。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.v2_p7.buffer_native_corporate import (
    NativeCorporateAdapter,
    NativeCorporateBook,
    NativeCorporateMixin,
)
from scripts.v2_p7.t4.corporate import CorporateError

_DAYS = ("20260105", "20260106", "20260107", "20260108", "20260109")
_CODE = "000001.SZ"


def _row(**kwargs: Any) -> dict[str, Any]:
    row = {
        "ts_code": _CODE,
        "end_date": "20251231",
        "ann_date": "20260101",
        "div_proc": "实施",
        "stk_div": 0.0,
        "cash_div": 1.0,
        "cash_div_tax": 1.25,
        "record_date": "20260105",
        "ex_date": "20260106",
        "pay_date": "20260108",
        "imp_ann_date": "20260102",
        "div_listdate": "20260106",
    }
    return {**row, **kwargs}


def _book(tmp_path: Path, *rows: dict[str, Any]) -> NativeCorporateBook:
    pd.DataFrame(list(rows) or [_row(div_proc="预案")]).to_parquet(
        tmp_path / "2025-12-31.parquet", index=False
    )
    return NativeCorporateBook(calendar=_DAYS, dividend_dir=tmp_path)


def _position(shares: int = 100) -> dict[str, Any]:
    return {
        "shares": shares,
        "buy_date": pd.Timestamp("20260105"),
        "signal_date": pd.Timestamp("20260102"),
        "buy_trade_price": 10.0,
        "buy_pnl_price": 12.0,
        "buy_cost_cash": shares * 10.0 + 5.0,
        "last_known_price": 10.0,
        "renewal_count": 3,
    }


class _BaseEngine:
    """原生最小状态面；测试模拟交易仍从 mixin 下方运行。"""

    def __init__(self, *, positions: dict[str, Any] | None = None) -> None:
        self.current_capital = 1000.0
        self.positions = copy.deepcopy(positions or {})
        self.days: list[str] = []
        self.navs: list[float] = []
        self.sell_on = ""
        self.buy_on = ""

    def run_step(self, date: Any, trading_dates_full: Any, price_data: Any) -> None:
        del trading_dates_full, price_data
        day = pd.Timestamp(date).strftime("%Y%m%d")
        self.days.append(day)
        if self.sell_on == day:
            self.current_capital += sum(
                value["shares"] * value["last_known_price"] for value in self.positions.values()
            )
            self.positions.clear()
        if self.buy_on == day:
            self.positions[_CODE] = _position()
        self.navs.append(self._calculate_portfolio_value(date))

    def _calculate_portfolio_value(self, date: Any) -> float:
        del date
        return self.current_capital + sum(
            value["shares"] * value["last_known_price"] for value in self.positions.values()
        )


class _Engine(NativeCorporateMixin, _BaseEngine):
    pass


def _step(engine: Any, *days: str) -> None:
    for day in days:
        engine.run_step(pd.Timestamp(day), _DAYS, None)


def test_disabled_has_exact_native_state_without_corporate_fields() -> None:
    plain = _BaseEngine(positions={_CODE: _position()})
    disabled = _Engine(positions={_CODE: _position()})
    _step(plain, *_DAYS)
    _step(disabled, *_DAYS)
    assert vars(disabled) == vars(plain)


def test_enabled_no_event_preserves_native_values(tmp_path: Path) -> None:
    plain = _BaseEngine(positions={_CODE: _position()})
    enabled = _Engine(positions={_CODE: _position()}, corporate_book=_book(tmp_path))
    _step(plain, *_DAYS)
    _step(enabled, *_DAYS)
    assert {k: v for k, v in vars(enabled).items() if k != "native_corporate"} == vars(plain)


def test_cash_entitlement_survives_sale_and_is_not_spendable_until_pay(tmp_path: Path) -> None:
    engine = _Engine(positions={_CODE: _position()}, corporate_book=_book(tmp_path, _row()))
    engine.sell_on = "20260107"
    _step(engine, *_DAYS[:2])
    assert engine.current_capital == 1000.0
    assert engine.native_corporate.receivable == 100.0
    assert engine.navs == [2000.0, 2000.0]
    _step(engine, _DAYS[2])
    assert not engine.positions
    assert engine.native_corporate.receivable == 100.0
    _step(engine, _DAYS[3])
    assert engine.current_capital == 2000.0
    assert engine.native_corporate.receivable == 0.0
    assert engine.native_corporate.payload()["pay_count"] == 1


def test_record_day_close_purchase_gets_entitlement(tmp_path: Path) -> None:
    engine = _Engine(corporate_book=_book(tmp_path, _row()))
    engine.buy_on = _DAYS[0]
    _step(engine, *_DAYS[:2])
    assert engine.native_corporate.receivable == 100.0


def test_record_day_sale_has_no_entitlement(tmp_path: Path) -> None:
    engine = _Engine(positions={_CODE: _position()}, corporate_book=_book(tmp_path, _row()))
    engine.sell_on = _DAYS[0]
    _step(engine, *_DAYS[:2])
    assert engine.native_corporate.receivable == 0.0
    assert engine.native_corporate.rows == []


def test_bonus_preserves_buy_date_cost_and_missing_price_equity(tmp_path: Path) -> None:
    original = _position()
    engine = _Engine(
        positions={_CODE: original},
        corporate_book=_book(tmp_path, _row(stk_div=0.2, cash_div=0.0, cash_div_tax=0.0)),
    )
    _step(engine, *_DAYS[:2])
    position = engine.positions[_CODE]
    assert position["shares"] == 120
    assert position["buy_date"] == original["buy_date"]
    assert position["buy_cost_cash"] == original["buy_cost_cash"]
    assert position["renewal_count"] == 3
    assert engine.navs == pytest.approx([2000.0, 2000.0])


@pytest.mark.parametrize("pay_date,status", [(None, "unknown"), ("20260203", "after_window")])
def test_missing_or_after_window_payment_remains_receivable(
    tmp_path: Path, pay_date: str | None, status: str
) -> None:
    book = _book(tmp_path, _row(pay_date=pay_date))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS)
    assert engine.current_capital == 1000.0
    assert engine.native_corporate.receivable == 100.0
    assert engine.native_corporate.payload()["pay_count"] == 0
    assert engine.native_corporate.rows[0]["pay_status"] == status


def test_same_day_ex_and_pay_is_once_and_precedes_trades(tmp_path: Path) -> None:
    engine = _Engine(
        positions={_CODE: _position()}, corporate_book=_book(tmp_path, _row(pay_date=_DAYS[1]))
    )
    _step(engine, *_DAYS[:2])
    adapter = engine.native_corporate
    adapter.before_day(engine, _DAYS[1])
    assert engine.current_capital == 1100.0
    assert adapter.receivable == 0.0
    assert [row["kind"] for row in adapter.rows] == ["ex", "pay"]


def test_future_revision_is_ignored_in_favor_of_known_version(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(), _row(imp_ann_date="20260109", cash_div=9.0))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS[:2])
    assert engine.native_corporate.receivable == 100.0
    assert engine.native_corporate.rows[0]["future_versions_ignored"] == 1


@pytest.mark.parametrize(
    "row2",
    [
        _row(cash_div=9.0),
        _row(imp_ann_date="20260109"),
    ],
)
def test_conflicting_or_unavailable_announcement_blocks_only_actual_holder(
    tmp_path: Path, row2: dict[str, Any]
) -> None:
    rows = [_row(), row2] if row2["imp_ann_date"] == "20260102" else [row2]
    book = _book(tmp_path, *rows)
    empty = _Engine(corporate_book=book)
    _step(empty, *_DAYS[:2])
    held = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(held, _DAYS[0])
    with pytest.raises(CorporateError, match="实际持仓"):
        _step(held, _DAYS[1])
    assert held.native_corporate.issues


def test_unknown_bonus_listing_blocks_holder(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(stk_div=0.2, div_listdate=None))
    held = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(held, _DAYS[0])
    with pytest.raises(CorporateError, match="送转上市"):
        _step(held, _DAYS[1])
    assert held.positions[_CODE]["shares"] == 100
    empty = _Engine(corporate_book=book)
    _step(empty, *_DAYS[:2])


def test_known_late_listing_locks_bonus_but_preserves_ex_day_equity(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(stk_div=0.2, cash_div=0.0, cash_div_tax=0.0, div_listdate=_DAYS[3]))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS[:2])
    adapter = engine.native_corporate
    assert engine.positions[_CODE]["shares"] == 120
    assert adapter.locked_shares(_CODE) == 20
    assert adapter.sellable_shares(engine, _CODE) == 100
    assert engine.navs == pytest.approx([2000.0, 2000.0])
    assert engine.positions[_CODE]["buy_date"] == pd.Timestamp(_DAYS[0])
    assert engine.positions[_CODE]["buy_cost_cash"] == 1005.0
    before_cash = engine.current_capital
    _step(engine, *_DAYS[2:4])
    assert engine.positions[_CODE]["shares"] == 120
    assert adapter.sellable_shares(engine, _CODE) == 120
    assert adapter.locked_shares(_CODE) == 0
    assert engine.current_capital == before_cash
    assert adapter.payload()["lock_count"] == adapter.payload()["unlock_count"] == 1


def test_multiple_bonus_events_unlock_independently(tmp_path: Path) -> None:
    book = _book(
        tmp_path,
        _row(stk_div=0.2, cash_div=0.0, cash_div_tax=0.0, div_listdate=_DAYS[3]),
        _row(
            end_date="20250630",
            stk_div=0.1,
            cash_div=0.0,
            cash_div_tax=0.0,
            record_date=_DAYS[1],
            ex_date=_DAYS[2],
            div_listdate=_DAYS[4],
        ),
    )
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS[:3])
    adapter = engine.native_corporate
    assert engine.positions[_CODE]["shares"] == 132
    assert adapter.locked_shares(_CODE) == 32
    assert adapter.sellable_shares(engine, _CODE) == 100
    _step(engine, _DAYS[3])
    assert adapter.locked_shares(_CODE) == 12
    assert adapter.sellable_shares(engine, _CODE) == 120
    _step(engine, _DAYS[4])
    assert adapter.locked_shares(_CODE) == 0
    assert adapter.sellable_shares(engine, _CODE) == 132
    assert adapter.payload()["lock_count"] == adapter.payload()["unlock_count"] == 2


def test_listing_after_window_preserves_terminal_lock(tmp_path: Path) -> None:
    book = _book(
        tmp_path, _row(stk_div=0.2, cash_div=0.0, cash_div_tax=0.0, div_listdate="20260202")
    )
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS)
    payload = engine.native_corporate.payload()
    assert payload["locked_shares"] == {_CODE: 20}
    assert payload["unlock_count"] == 0
    assert payload["locks"][0]["unlock_trade_day"] == "20260202"
    assert engine.navs[-1] == pytest.approx(2000.0)


def test_lock_guard_rejects_full_sell_without_auto_partial_execution(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(stk_div=0.2, cash_div=0.0, cash_div_tax=0.0, div_listdate=_DAYS[3]))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS[:2])
    adapter = engine.native_corporate
    before = copy.deepcopy(engine.positions)
    adapter.check_sell(engine, _DAYS[1], _CODE, 100)
    with pytest.raises(CorporateError, match="超过可卖"):
        adapter.check_sell(engine, _DAYS[1], _CODE, 120)
    assert engine.positions == before
    assert engine.current_capital == 1000.0
    assert adapter.payload()["blocked_sell_count"] == 1
    evidence = next(row for row in adapter.rows if row["kind"] == "sell_blocked_locked")
    assert (
        evidence["requested_shares"],
        evidence["sellable_shares"],
        evidence["locked_shares"],
    ) == (120, 100, 20)


def test_cash_still_paid_if_sold_between_record_and_ex(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(ex_date="20260107"))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    engine.sell_on = "20260106"
    _step(engine, *_DAYS[:3])
    assert engine.native_corporate.receivable == 100.0
    assert not engine.positions


def test_previous_window_record_has_zero_rights_for_cash_start(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(record_date="20260102"))
    engine = _Engine(corporate_book=book)
    engine.buy_on = _DAYS[0]
    _step(engine, *_DAYS[:2])
    assert engine.native_corporate.receivable == 0.0


def test_missing_record_snapshot_in_window_is_visible(tmp_path: Path) -> None:
    engine = _BaseEngine(positions={_CODE: _position()})
    adapter = NativeCorporateAdapter(_book(tmp_path, _row()))
    with pytest.raises(CorporateError, match="快照缺失"):
        adapter.before_day(engine, _DAYS[1])


def test_missing_record_date_blocks_actual_holder(tmp_path: Path) -> None:
    engine = _Engine(
        positions={_CODE: _position()}, corporate_book=_book(tmp_path, _row(record_date=None))
    )
    _step(engine, _DAYS[0])
    with pytest.raises(CorporateError, match="登记日未知"):
        _step(engine, _DAYS[1])


def test_tax_fallback_uses_t4_rule_and_is_reported(tmp_path: Path) -> None:
    engine = _Engine(
        positions={_CODE: _position()},
        corporate_book=_book(tmp_path, _row(cash_div=None, cash_div_tax=2.0)),
    )
    _step(engine, *_DAYS[:2])
    assert engine.native_corporate.receivable == 160.0
    assert engine.native_corporate.payload()["cash_basis_fallback_held"] == 1


def test_accounts_do_not_share_mutable_corporate_state(tmp_path: Path) -> None:
    book = _book(tmp_path, _row())
    first = _Engine(positions={_CODE: _position()}, corporate_book=book)
    second = _Engine(corporate_book=book)
    _step(first, *_DAYS[:2])
    _step(second, *_DAYS[:2])
    assert first.native_corporate.receivable == 100.0
    assert second.native_corporate.receivable == 0.0
    assert book._snapshot_counts == {}


def test_skipped_day_fails_instead_of_losing_payment(tmp_path: Path) -> None:
    engine = _Engine(corporate_book=_book(tmp_path, _row()))
    _step(engine, _DAYS[0])
    with pytest.raises(CorporateError, match="连续历史驱动"):
        _step(engine, _DAYS[2])


def test_undated_old_record_does_not_block_current_holding(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(ex_date=None, record_date="20251230"))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS)
    assert engine.native_corporate.issues == []
    assert engine.native_corporate.receivable == 0.0
    assert book.undated_problems[0]["record_date"] == "20251230"
    assert book.undated_problems[0]["known_date"] == "20260102"


def test_undated_actual_record_entitlement_blocks_at_close(tmp_path: Path) -> None:
    engine = _Engine(
        positions={_CODE: _position()}, corporate_book=_book(tmp_path, _row(ex_date=None))
    )
    with pytest.raises(CorporateError, match="除权日未知"):
        _step(engine, _DAYS[0])
    # 只在真实收盘权益产生后阻断，不根据开盘前持仓猜测登记结果。
    assert engine.days == [_DAYS[0]]
    assert engine.native_corporate.payload()["held_issue_count"] == 1


def test_undated_record_day_sale_has_zero_rights(tmp_path: Path) -> None:
    engine = _Engine(
        positions={_CODE: _position()}, corporate_book=_book(tmp_path, _row(ex_date=None))
    )
    engine.sell_on = _DAYS[0]
    _step(engine, *_DAYS)
    assert engine.native_corporate.issues == []


def test_future_undated_announcement_does_not_block_earlier_days(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(ex_date=None, record_date=None, imp_ann_date=_DAYS[3]))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS[:3])
    assert engine.native_corporate.issues == []
    with pytest.raises(CorporateError, match="登记日也未知"):
        _step(engine, _DAYS[3])
    assert engine.days == list(_DAYS[:3])


def test_late_known_undated_problem_uses_past_record_rights_after_sale(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(ex_date=None, imp_ann_date=_DAYS[3]))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    engine.sell_on = _DAYS[1]
    _step(engine, *_DAYS[:3])
    assert engine.positions == {}
    with pytest.raises(CorporateError, match="除权日未知"):
        _step(engine, _DAYS[3])


def test_undated_future_record_waits_for_actual_close(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(ex_date=None, record_date=_DAYS[2]))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS[:2])
    assert engine.native_corporate.issues == []
    with pytest.raises(CorporateError, match="除权日未知"):
        _step(engine, _DAYS[2])
    assert engine.days == list(_DAYS[:3])


def test_undated_without_any_known_date_blocks_and_explains(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(ex_date=None, record_date=None, ann_date=None, imp_ann_date=None))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    with pytest.raises(CorporateError, match="无任何可知公告日期"):
        _step(engine, _DAYS[0])


def test_undated_records_preserve_separate_ownership_dates(tmp_path: Path) -> None:
    book = _book(
        tmp_path,
        _row(ex_date=None, record_date="20251230"),
        _row(ex_date=None, end_date="20250630", record_date=_DAYS[1]),
    )
    assert len(book.undated_problems) == 2
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, _DAYS[0])
    with pytest.raises(CorporateError, match="除权日未知"):
        _step(engine, _DAYS[1])


def test_t4_null_cash_zero_shares_do_not_create_missing_date_risk(tmp_path: Path) -> None:
    book = _book(
        tmp_path,
        _row(
            cash_div=None,
            cash_div_tax=None,
            stk_div=0.0,
            ex_date=None,
            record_date=None,
            pay_date=None,
        ),
    )
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS)
    assert book.events() == ()
    assert book.undated_problems == []
    assert engine.native_corporate.issues == []
    summary = engine.native_corporate.payload()["zero_effect_summary"]
    assert summary["input_zero_effect_rows"] == summary["missing_ex_rows"] == 1
    assert summary["both_cash_fields_missing_rows"] == 1
    example = summary["examples"][0]
    assert example["cash_basis"] == "none"
    assert example["source_cash_div"] is None and example["source_cash_div_tax"] is None
    assert example["source_stk_div"] == 0.0
    assert example["parsed_cash_per_share"] == example["parsed_share_ratio"] == 0.0


@pytest.mark.parametrize(
    "cash, tax, ratio",
    [
        (1.0, None, 0.0),
        (None, None, 0.2),
        (-1.0, 1.0, 0.0),
        (float("inf"), 1.0, 0.0),
    ],
)
def test_positive_t4_effect_with_missing_ex_still_blocks(
    tmp_path: Path, cash: float | None, tax: float | None, ratio: float
) -> None:
    book = _book(
        tmp_path,
        _row(cash_div=cash, cash_div_tax=tax, stk_div=ratio, ex_date=None, record_date=None),
    )
    assert book.zero_effect_summary["input_zero_effect_rows"] == 0
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    with pytest.raises(CorporateError, match="除权日未知"):
        _step(engine, _DAYS[0])


def test_zero_effect_latest_version_does_not_resurrect_old_positive_event(tmp_path: Path) -> None:
    book = _book(
        tmp_path,
        _row(),
        _row(imp_ann_date=_DAYS[1], cash_div=None, cash_div_tax=None, stk_div=0.0),
    )
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS)
    assert book.events() == ()
    assert book._stats["zero_effect_events"] == 1
    assert engine.native_corporate.receivable == 0.0
    assert engine.native_corporate.rows == []


def test_zero_effect_cannot_hide_same_time_positive_conflict(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(), _row(cash_div=None, cash_div_tax=None, stk_div=0.0))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, _DAYS[0])
    with pytest.raises(CorporateError, match="冲突实施版本"):
        _step(engine, _DAYS[1])


def test_known_ex_zero_effect_missing_record_does_not_block(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(cash_div=0.0, cash_div_tax=0.0, stk_div=0.0, record_date=None))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS)
    assert engine.native_corporate.issues == []
    assert book.zero_effect_summary["missing_record_rows"] == 1
    assert book.zero_effect_summary["both_cash_fields_missing_rows"] == 0


@pytest.mark.parametrize(
    "cash, ratio, pay, listing",
    [
        (None, 0.3, None, "20251231"),
        (1.0, 0.0, "20251231", None),
        (1.0, 0.3, "20251230", "20251231"),
    ],
)
def test_all_positive_components_settled_before_cash_start_have_zero_rights(
    tmp_path: Path, cash: float | None, ratio: float, pay: str | None, listing: str | None
) -> None:
    book = _book(
        tmp_path,
        _row(
            cash_div=cash,
            cash_div_tax=None,
            stk_div=ratio,
            ex_date=None,
            record_date=None,
            pay_date=pay,
            div_listdate=listing,
        ),
    )
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    _step(engine, *_DAYS)
    assert book.undated_problems == []
    assert engine.native_corporate.issues == []
    assert engine.native_corporate.rows == []
    payload = engine.native_corporate.payload()
    assert payload["settled_before_start_count"] == 1
    proof = payload["settled_before_start_proofs"][0]
    assert proof["window_cash_start"] == _DAYS[0]
    assert proof["cash_settled_date"] == (pay if cash else None)
    assert proof["shares_listed_date"] == (listing if ratio else None)
    assert proof["source_fields"]["cash_div"] == cash
    assert proof["source_fields"]["record_date"] == ""
    assert proof["source_fields"]["ex_date"] == ""


@pytest.mark.parametrize(
    "cash, ratio, pay, listing",
    [
        (1.0, 0.0, None, "20251231"),
        (1.0, 0.0, "20260105", None),
        (None, 0.3, None, "20260105"),
        (1.0, 0.3, "20251231", None),
        (1.0, 0.3, None, "20251231"),
        (1.0, 0.3, "20251231", "20260105"),
        (1.0, 0.3, "20260105", "20251231"),
        (1.0, 0.3, "20251231", "20260106"),
    ],
)
def test_any_positive_component_unsettled_unknown_or_same_start_remains_blocked(
    tmp_path: Path, cash: float | None, ratio: float, pay: str | None, listing: str | None
) -> None:
    book = _book(
        tmp_path,
        _row(
            cash_div=cash,
            cash_div_tax=None,
            stk_div=ratio,
            ex_date=None,
            record_date=None,
            pay_date=pay,
            div_listdate=listing,
            ann_date="20000101",
            imp_ann_date="20000102",
        ),
    )
    assert book.settled_before_start_proofs == []
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    with pytest.raises(CorporateError, match="登记日也未知"):
        _step(engine, _DAYS[0])


def test_known_record_date_is_not_overridden_by_old_settlement_fields(tmp_path: Path) -> None:
    book = _book(tmp_path, _row(ex_date=None, record_date=_DAYS[0], pay_date="20251231"))
    engine = _Engine(positions={_CODE: _position()}, corporate_book=book)
    assert book.settled_before_start_proofs == []
    with pytest.raises(CorporateError, match="除权日未知"):
        _step(engine, _DAYS[0])
