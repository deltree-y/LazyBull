"""月度经营袖账户的合成检查；不依赖真实行情、配置或公司行动。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook
from scripts.v2_p7.operating_monthly_account import MonthlyAccountError, run_monthly_account

DAYS = tuple(day.strftime("%Y%m%d") for day in pd.bdate_range("2026-01-05", periods=8))
CODES = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ")


def quotes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": day,
                "ts_code": code,
                "open": 10.0,
                "close": 10.0,
                "is_suspended": 0,
                "is_limit_up": 0,
                "is_limit_down": 0,
                "is_st": 0,
                "tradable": 1,
            }
            for day in DAYS
            for code in CODES
        ]
    )


def signals(*items: tuple[int, tuple[str, ...]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"signal_date": DAYS[index], "ts_code": code, "eligible": code in selected}
            for index, selected in items
            for code in CODES
        ]
    )


def dividend(**updates: Any) -> dict[str, Any]:
    row = {
        "ts_code": CODES[0],
        "end_date": "20251231",
        "ann_date": "20260101",
        "imp_ann_date": "20260102",
        "div_proc": "实施",
        "cash_div": 1.0,
        "cash_div_tax": 1.25,
        "stk_div": 0.0,
        "record_date": DAYS[1],
        "ex_date": DAYS[2],
        "pay_date": DAYS[4],
        "div_listdate": DAYS[2],
    }
    return {**row, **updates}


def run(
    tmp_path: Path,
    *,
    data: pd.DataFrame | None = None,
    sig: pd.DataFrame | None = None,
    capital: float = 4100.0,
    events: list[dict[str, Any]] | None = None,
    haircut: float = 0.0,
) -> dict[str, Any]:
    directory = tmp_path / f"div_{len(list(tmp_path.iterdir()))}"
    directory.mkdir()
    pd.DataFrame(events or [dividend(div_proc="预案")]).to_parquet(directory / "2025.parquet")
    return run_monthly_account(
        calendar=DAYS,
        quotes=quotes() if data is None else data,
        signals=signals((0, (CODES[0],))) if sig is None else sig,
        capital=capital,
        corporate_book=NativeCorporateBook(calendar=DAYS, dividend_dir=directory),
        arm_name="synthetic",
        codes=CODES,
        dividend_haircut=haircut,
    )


def test_signal_then_next_day_close_buy_and_four_fixed_slots(tmp_path: Path) -> None:
    result = run(tmp_path)
    trades = result["trades"]
    assert len(trades) == 1
    assert trades.iloc[0]["trade_date"] == DAYS[1]
    assert trades.iloc[0]["signal_date"] == DAYS[0]
    assert trades.iloc[0]["price_kind"] == "close"
    assert trades.iloc[0]["shares"] == 100
    assert result["daily"].iloc[0]["equity"] == 4100
    assert result["daily"].iloc[-1]["cash"] == pytest.approx(3094.5)
    daily = result["daily"].set_index("trade_date")
    assert daily.total_value.equals(daily.equity)
    assert daily.loc[DAYS[1], "buy_amount"] == 1000
    assert daily.buy_amount.sum() == 1000
    assert daily.sell_amount.sum() == 0
    assert daily.fees.sum() == pytest.approx(5.5)


@pytest.mark.parametrize(
    "column,value", [("is_limit_up", 1), ("is_suspended", 1), ("is_st", 1), ("tradable", 0)]
)
def test_rejected_buy_is_not_retried_next_day(tmp_path: Path, column: str, value: int) -> None:
    data = quotes()
    data.loc[(data.trade_date == DAYS[1]) & (data.ts_code == CODES[0]), column] = value
    result = run(tmp_path, data=data)
    assert result["trades"].empty
    assert result["daily"].nav.eq(1.0).all()
    assert result["decisions"].kind.eq("buy_voided").sum() == 1


@pytest.mark.parametrize("column", ["is_suspended", "is_limit_down"])
def test_failed_sell_carries_until_executable_open(tmp_path: Path, column: str) -> None:
    data = quotes()
    data.loc[data.trade_date.isin(DAYS[3:5]) & (data.ts_code == CODES[0]), column] = 1
    result = run(tmp_path, data=data, sig=signals((0, (CODES[0],)), (2, ())))
    sell = result["trades"].query("side == 'sell'")
    assert list(sell.trade_date) == [DAYS[5]]
    assert list(sell.price_kind) == ["open"]
    assert result["decisions"].kind.eq("sell_blocked").sum() == 2
    assert result["final_state"]["pending_sells"] == {}


@pytest.mark.parametrize("column,value", [("is_st", 1), ("tradable", 0)])
def test_buy_domain_flags_do_not_block_exit(tmp_path: Path, column: str, value: int) -> None:
    data = quotes()
    data.loc[(data.trade_date == DAYS[3]) & (data.ts_code == CODES[0]), column] = value
    result = run(tmp_path, data=data, sig=signals((0, (CODES[0],)), (2, ())))
    sells = result["trades"].query("side == 'sell'")
    assert list(sells.trade_date) == [DAYS[3]]
    assert result["daily"].sell_amount.sum() == 1000


def test_minimum_commission_and_lots_cannot_overdraw_cash(tmp_path: Path) -> None:
    result = run(tmp_path, sig=signals((0, CODES)))
    assert list(result["trades"].shares) == [100] * 4
    assert (result["daily"].cash >= 0).all()
    assert result["daily"].iloc[-1].cash == pytest.approx(78)
    too_small = run(tmp_path, capital=4000, sig=signals((0, CODES)))
    assert too_small["trades"].empty


def test_open_sells_do_not_use_same_day_close_price(tmp_path: Path) -> None:
    data = quotes()
    data.loc[(data.trade_date >= DAYS[3]) & (data.ts_code == CODES[0]), "close"] = 20
    result = run(tmp_path, capital=40400, data=data, sig=signals((0, CODES), (2, CODES)))
    assert result["trades"].query("side == 'sell'").empty
    assert result["final_state"]["positions"][CODES[0]]["shares"] == 1000


def test_cash_dividend_is_receivable_then_paid_after_sale(tmp_path: Path) -> None:
    data = quotes()
    mask = (data.trade_date >= DAYS[2]) & (data.ts_code == CODES[0])
    data.loc[mask, ["open", "close"]] = 9.0
    result = run(tmp_path, data=data, events=[dividend()], sig=signals((0, (CODES[0],)), (2, ())))
    daily = result["daily"].set_index("trade_date")
    assert daily.loc[DAYS[2], "receivable"] == 100
    assert daily.loc[DAYS[3], "receivable"] == 100
    assert daily.loc[DAYS[3], "holding_count"] == 0
    assert daily.loc[DAYS[4], "receivable"] == 0
    assert daily.loc[DAYS[4], "cash"] - daily.loc[DAYS[3], "cash"] == pytest.approx(100)
    assert daily.loc[DAYS[2], "equity"] == pytest.approx(daily.loc[DAYS[1], "equity"])


def test_bonus_locks_then_sells_only_after_listing(tmp_path: Path) -> None:
    data = quotes()
    mask = (data.trade_date >= DAYS[2]) & (data.ts_code == CODES[0])
    data.loc[mask, ["open", "close"]] = 10.0 / 1.2
    result = run(
        tmp_path,
        data=data,
        sig=signals((0, (CODES[0],)), (2, ())),
        events=[dividend(cash_div=0, cash_div_tax=0, stk_div=0.2, div_listdate=DAYS[5])],
    )
    sells = result["trades"].query("side == 'sell'")
    assert list(sells.shares) == [100, 20]
    assert list(sells.trade_date) == [DAYS[3], DAYS[5]]
    assert result["corporate"]["lock_count"] == 1
    assert result["corporate"]["unlock_count"] == 1
    daily = result["daily"].set_index("trade_date")
    assert daily.loc[DAYS[2], "equity"] == pytest.approx(daily.loc[DAYS[1], "equity"])


def test_dividend_haircut_liability_starts_at_ex_day(tmp_path: Path) -> None:
    data = quotes()
    data.loc[(data.trade_date >= DAYS[2]) & (data.ts_code == CODES[0]), ["open", "close"]] = 9
    base = run(tmp_path, data=data, events=[dividend()])
    stress = run(tmp_path, data=data, events=[dividend()], haircut=0.2)
    b = base["daily"].set_index("trade_date")
    s = stress["daily"].set_index("trade_date")
    assert s.loc[DAYS[2], "dividend_haircut_liability"] == 20
    assert s.loc[DAYS[2], "cash"] == b.loc[DAYS[2], "cash"]
    assert b.loc[DAYS[2], "equity"] - s.loc[DAYS[2], "equity"] == 20
    assert s.loc[DAYS[4], "dividend_haircut_liability"] == 0
    assert b.loc[DAYS[4], "cash"] - s.loc[DAYS[4], "cash"] == 20
    assert (s.cash + s.market_value + s.receivable - s.dividend_haircut_liability).equals(s.equity)


def test_obligation_not_cancelled_by_new_eligible_signal(tmp_path: Path) -> None:
    data = quotes()
    data.loc[data.trade_date.isin(DAYS[3:5]) & (data.ts_code == CODES[0]), "is_suspended"] = 1
    result = run(tmp_path, data=data, sig=signals((0, (CODES[0],)), (2, ()), (3, (CODES[0],))))
    sells = result["trades"].query("side == 'sell'")
    assert list(sells.trade_date) == [DAYS[5]]
    assert result["final_state"]["positions"] == {}


def test_new_exit_extends_old_blocked_trim_to_whole_position(tmp_path: Path) -> None:
    data = quotes()
    mask = (data.trade_date >= DAYS[3]) & (data.ts_code == CODES[0])
    data.loc[mask, ["open", "close"]] = 20
    data.loc[data.trade_date.isin(DAYS[3:5]) & (data.ts_code == CODES[0]), "is_suspended"] = 1
    result = run(
        tmp_path, data=data, capital=40400, sig=signals((0, (CODES[0],)), (2, (CODES[0],)), (3, ()))
    )
    sells = result["trades"].query("side == 'sell'")
    assert list(sells.shares) == [1000]
    assert list(sells.trade_date) == [DAYS[5]]
    assert result["final_state"]["positions"] == {}


def test_missing_ex_day_quote_uses_corporate_adjusted_mark(tmp_path: Path) -> None:
    data = quotes()
    data = data.loc[~((data.trade_date == DAYS[2]) & (data.ts_code == CODES[0]))]
    data.loc[(data.trade_date > DAYS[2]) & (data.ts_code == CODES[0]), ["open", "close"]] = 9
    result = run(tmp_path, data=data, events=[dividend()])
    daily = result["daily"].set_index("trade_date")
    assert daily.loc[DAYS[2], "equity"] == pytest.approx(daily.loc[DAYS[1], "equity"])
    assert daily.loc[DAYS[2], "receivable"] == 100
    hold = result["holdings"].query("trade_date == @DAYS[2]")
    assert hold.iloc[0].price == 9


def test_missing_quote_carries_valuation_without_future_fill(tmp_path: Path) -> None:
    data = quotes()
    data = data.loc[~((data.trade_date == DAYS[2]) & (data.ts_code == CODES[0]))]
    data.loc[(data.trade_date >= DAYS[3]) & (data.ts_code == CODES[0]), ["open", "close"]] = 50
    result = run(tmp_path, data=data)
    hold = result["holdings"].query("trade_date == @DAYS[2]")
    assert hold.iloc[0].price == 10
    assert hold.iloc[0].valuation_source == "last_known_adjusted_for_corporate"


def test_invalid_signal_grid_and_unknown_clean_flags_fail(tmp_path: Path) -> None:
    with pytest.raises(MonthlyAccountError, match="四个"):
        run(tmp_path, sig=signals((0, CODES)).iloc[:3])
    data = quotes()
    data.loc[0, "is_limit_up"] = float("nan")
    with pytest.raises(MonthlyAccountError, match="未知"):
        run(tmp_path, data=data)
