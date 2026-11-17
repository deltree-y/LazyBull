"""原生续持、公司行为与日终旁路组合的合成集成验证。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook, NativeCorporateMixin
from scripts.v2_p7.buffer_native_engine import BufferNativeEngine
from scripts.v2_p7.buffer_native_replay import (
    FrozenRankSignal,
    RecordNativeMixin,
    compare_exact,
    initialize_driver,
    result_payload,
)
from scripts.v2_p7.t4.corporate import CorporateError
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger
from src.lazybull.v2.core.execution.price_index import prepare_price_index

_CODES = [f"{index:06}.SZ" for index in range(1, 41)]


class _Universe:
    def get_stocks(self, date: Any, quote_data: Any = None) -> list[str]:
        return list(_CODES)


class _Storage:
    def load_raw_by_date(self, dataset: str, trade_date: Any) -> pd.DataFrame:
        assert dataset == "suspend"
        return pd.DataFrame(columns=["ts_code", "suspend_type"])


class _ResearchEngine(RecordNativeMixin, NativeCorporateMixin, BufferNativeEngine):
    pass


def _inputs(count: int = 65, start: str = "20260105") -> tuple[list, pd.DataFrame, dict]:
    dates = list(pd.bdate_range(start, periods=count))
    ranks = {
        day.strftime("%Y%m%d"): [(code, float(40 - rank)) for rank, code in enumerate(_CODES)]
        for day in dates
    }
    prices = pd.DataFrame(
        [
            {
                "trade_date": day.strftime("%Y%m%d"),
                "ts_code": code,
                "open": 10.0,
                "close": 10.0,
                "open_adj": 10.0,
                "close_adj": 10.0,
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
                "is_st": False,
                "tradable": True,
                "list_days": 1000,
                "vol": 1000000.0,
            }
            for day in dates
            for code in _CODES
        ]
    )
    return dates, prices, ranks


def _dividend(dates: list, code: str, record: int, ex: int, pay: int, **values: Any) -> dict:
    return {
        "ts_code": code,
        "end_date": "20251231",
        "ann_date": "20260101",
        "div_proc": "实施",
        "stk_div": 0.0,
        "cash_div": 1.0,
        "cash_div_tax": 1.25,
        "record_date": dates[record].strftime("%Y%m%d"),
        "ex_date": dates[ex].strftime("%Y%m%d"),
        "pay_date": dates[pay].strftime("%Y%m%d"),
        "imp_ann_date": "20260102",
        "div_listdate": dates[ex].strftime("%Y%m%d"),
        **values,
    }


def _book(tmp_path: Path, dates: list, rows: list[dict]) -> NativeCorporateBook:
    if not rows:
        rows = [_dividend(dates, _CODES[0], 0, 1, 2, div_proc="预案")]
    pd.DataFrame(rows).to_parquet(tmp_path / "2025-12-31.parquet", index=False)
    return NativeCorporateBook(
        calendar=[day.strftime("%Y%m%d") for day in dates], dividend_dir=tmp_path
    )


def _engine(ranks: dict, book: NativeCorporateBook | None) -> _ResearchEngine:
    return _ResearchEngine(
        portfolio_merger=SingleSleevePortfolioMerger(),
        universe=_Universe(),
        signal=FrozenRankSignal(ranks),
        features_by_date={day: pd.DataFrame({"x": [1]}) for day in ranks},
        initial_capital=1_000_000.0,
        cost_model=CostModel(0.0001954, 5.0, 0.0005, 0.0005),
        rebalance_freq=20,
        stagger_tranches=2,
        holding_period=20,
        sell_timing="open",
        verbose=False,
        position_sizing="equal",
        max_weight_per_stock=0.15,
        min_buy_value_ratio=0.0,
        data_storage=_Storage(),
        enable_pending_order=True,
        enable_position_completion=True,
        completion_window_days=5,
        enable_early_rebalance_on_empty=False,
        buffer_enabled=True,
        rank_provider=lambda date: ranks[date.strftime("%Y%m%d")],
        corporate_book=book,
    )


def _drive(engine: _ResearchEngine, dates: list, prices: pd.DataFrame) -> None:
    initialize_driver(engine, dates, prices, prepare_price_index(prices))
    for date in dates:
        engine.run_step(date, dates, prices)


def _check_daily_accounts(engine: _ResearchEngine, prices: pd.DataFrame) -> None:
    """独立按原生成交现金流与未复权持仓估值重算，不引用已算好的市值字段。"""
    paid = engine.native_corporate.payload()["cash_paid_by_day"]
    close = prices.set_index(["trade_date", "ts_code"])["close"].to_dict()
    previous = engine.initial_capital
    for row, state in zip(engine.research_daily, engine.research_states):
        day = row["trade_date"]
        expected_cash = (
            previous + row["sell_amount"] - row["buy_amount"] - row["fees"] + paid.get(day, 0.0)
        )
        assert row["cash"] == pytest.approx(expected_cash, abs=1e-7)
        market_value = sum(
            position["shares"] * close[(day, code)] for code, position in state["positions"].items()
        )
        assert row["market_value"] == pytest.approx(market_value, abs=1e-7)
        assert row["total_value"] == pytest.approx(
            expected_cash + market_value + row["receivable"], abs=1e-7
        )
        assert row["cash"] >= 0
        previous = row["cash"]


def test_renewals_cross_cash_dividend_bonus_and_post_sale_payment(tmp_path: Path) -> None:
    dates, prices, ranks = _inputs()
    cash_code, bonus_code = _CODES[:2]
    for date in dates[40:]:
        day = date.strftime("%Y%m%d")
        ranks[day] = ranks[day][1:] + ranks[day][:1]
    cash = _dividend(dates, cash_code, 35, 36, 45)
    bonus = _dividend(dates, bonus_code, 25, 26, 30, stk_div=0.2, cash_div=0.4)
    prices.loc[
        prices.ts_code.eq(cash_code) & prices.trade_date.ge(cash["ex_date"]), ["open", "close"]
    ] = 9.0
    prices.loc[
        prices.ts_code.eq(bonus_code) & prices.trade_date.ge(bonus["ex_date"]), ["open", "close"]
    ] = 8.0
    engine = _engine(ranks, _book(tmp_path, dates, [cash, bonus]))
    _drive(engine, dates, prices)
    _check_daily_accounts(engine, prices)

    cash_buys = [t for t in engine.trades if t["stock"] == cash_code and t["action"] == "buy"]
    cash_sells = [t for t in engine.trades if t["stock"] == cash_code and t["action"] == "sell"]
    assert len(cash_buys) == len(cash_sells) == 1
    assert cash_buys[0]["date"] == dates[1]
    assert cash_sells[0]["date"] == dates[41]
    assert cash_sells[0]["buy_date"] == dates[1]
    assert [
        row["renew"]
        for row in engine.events
        if row["kind"] == "expiry" and row["ts_code"] == cash_code
    ] == [True, False]
    assert cash_code not in engine.research_states[41]["positions"]
    assert engine.research_daily[41]["receivable"] == cash_buys[0]["shares"]
    assert engine.research_daily[44]["receivable"] == cash_buys[0]["shares"]
    assert engine.research_daily[45]["receivable"] == 0.0

    before = engine.research_states[25]["positions"][bonus_code]
    after = engine.research_states[26]["positions"][bonus_code]
    assert after["shares"] == int(before["shares"] * 1.2)
    assert after["buy_date"] == before["buy_date"] == dates[1]
    assert after["buy_cost_cash"] == before["buy_cost_cash"]
    assert engine._buffer_meta[bonus_code]["renewal_count"] == 3
    assert engine.native_corporate.payload()["held_issue_count"] == 0


def test_composed_empty_corporate_book_preserves_full_native_path(tmp_path: Path) -> None:
    dates, prices, ranks = _inputs(45)
    for date in dates[40:]:
        day = date.strftime("%Y%m%d")
        ranks[day] = ranks[day][20:] + ranks[day][:20]
    raw = _engine(ranks, None)
    economic = _engine(ranks, _book(tmp_path, dates, []))
    _drive(raw, dates, prices)
    _drive(economic, dates, prices)
    check = compare_exact(result_payload(raw), result_payload(economic))
    assert check["all_equal"], check
    assert raw.events == economic.events
    assert any(row["kind"] == "expiry" and row["renew"] for row in economic.events)
    assert any(row["action"] == "sell" for row in raw.trades)


def test_holdout_cash_anchor_records_rights_without_signals(tmp_path: Path) -> None:
    dates, prices, ranks = _inputs(25, "20260202")
    ranks.pop("20260202")
    dividend = _dividend(dates, _CODES[0], 0, 1, 3)
    engine = _engine(ranks, _book(tmp_path, dates, [dividend]))
    initialize_driver(engine, dates, prices, prepare_price_index(prices))
    engine._signal_dates = engine._get_rebalance_dates(dates[1:])
    adapter = engine.native_corporate
    adapter.before_day(engine, dates[0])
    adapter.after_day(engine, dates[0])
    assert engine.pending_signals == {}
    assert engine.trades == []
    assert engine.events == []
    assert engine.research_daily == []
    for date in dates[1:]:
        engine.run_step(date, dates, prices)
    assert min(row["date"] for row in engine.trades if row["action"] == "buy") == dates[2]
    assert not any(row["kind"] == "pay" for row in adapter.rows)
    assert adapter.payload()["record_snapshots"]["first_day"] == "20260202"
    assert engine.research_daily[0]["trade_date"] == "20260203"


def test_native_final_sell_entry_rejects_locked_bonus_then_sells_after_listing(
    tmp_path: Path,
) -> None:
    dates, prices, ranks = _inputs(8)
    code = _CODES[0]
    row = _dividend(
        dates,
        code,
        2,
        3,
        3,
        stk_div=0.2,
        cash_div=0.0,
        cash_div_tax=0.0,
        div_listdate=dates[5].strftime("%Y%m%d"),
    )
    prices.loc[
        prices.ts_code.eq(code) & prices.trade_date.ge(row["ex_date"]), ["open", "close"]
    ] = (10.0 / 1.2)
    engine = _engine(ranks, _book(tmp_path, dates, [row]))
    initialize_driver(engine, dates, prices, prepare_price_index(prices))
    for date in dates[:4]:
        engine.run_step(date, dates, prices)
    adapter = engine.native_corporate
    position = copy.deepcopy(engine.positions[code])
    previous_cash, previous_trades = engine.current_capital, copy.deepcopy(engine.trades)
    assert adapter.locked_shares(code) > 0
    with pytest.raises(CorporateError, match="超过可卖"):
        engine._sell_stock_direct(dates[3], code)
    assert engine.positions[code] == position
    assert engine.current_capital == previous_cash
    assert engine.trades == previous_trades
    assert adapter.payload()["blocked_sell_count"] == 1
    for date in dates[4:6]:
        engine.run_step(date, dates, prices)
    assert adapter.locked_shares(code) == 0
    assert engine.positions[code] == position
    engine._sell_stock_direct(dates[5], code)
    assert code not in engine.positions
    assert engine.trades[-1]["shares"] == position["shares"]
    assert engine.trades[-1]["buy_date"] == dates[1]
