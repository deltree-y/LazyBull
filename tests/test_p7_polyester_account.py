"""涤纶双槽与旧四槽执行等价合成检查，不依赖真实数据。

旧槽位方法快照来自改动前账户SHA：
0742b837c67877947f0311133d8d2f5c99064b626cd451d064c48ad82fa690bc。
只复制本次涉及的四个方法，其余执行方法未改，由原类继承。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

from scripts.v2_p7.buffer_native_corporate import NativeCorporateAdapter, NativeCorporateBook
from scripts.v2_p7.operating_monthly_account import (
    FROZEN_COSTS,
    MonthlyAccountError,
    _day,
    _MonthlyAccount,
    _positive,
    run_monthly_account,
)
from scripts.v2_p7.polyester_account import run_polyester_account
from src.lazybull.v2.common.cost import CostModel


class _LegacyFourSlotAccount(_MonthlyAccount):
    """改动前四个方法的静态快照，供四槽逐位比较。"""

    def __init__(
        self,
        calendar: Sequence[str],
        quotes: pd.DataFrame,
        signals: pd.DataFrame,
        capital: float,
        corporate_book: NativeCorporateBook,
        arm_name: str,
        codes: Sequence[str],
        dividend_haircut: float,
    ) -> None:
        self.days = tuple(_day(day) for day in calendar)
        self.codes = tuple(sorted(str(code) for code in codes))
        if not self.days or self.days != tuple(sorted(set(self.days))):
            raise MonthlyAccountError("日历必须非空、严格递增且不重复")
        if len(self.codes) != 4 or len(set(self.codes)) != 4:
            raise MonthlyAccountError("固定四槽必须提供四个不同股票代码")
        if tuple(corporate_book._days) != self.days:
            raise MonthlyAccountError("公司行为日历与账户日历不一致")
        if not _positive(capital) or not 0 <= dividend_haircut <= 1:
            raise MonthlyAccountError("初始资金须正且分红压力折扣须在[0,1]")
        required = {
            "trade_date",
            "ts_code",
            "open",
            "close",
            "is_suspended",
            "is_limit_up",
            "is_limit_down",
            "is_st",
            "tradable",
        }
        if not required.issubset(quotes.columns):
            raise MonthlyAccountError(f"行情缺列：{sorted(required - set(quotes.columns))}")
        data = quotes.copy()
        data["trade_date"] = data["trade_date"].map(_day)
        if data.duplicated(["trade_date", "ts_code"]).any():
            raise MonthlyAccountError("行情日期与代码重复")
        for column in ("is_suspended", "is_limit_up", "is_limit_down", "is_st", "tradable"):
            if not data[column].isin([0, 1, False, True]).all():
                raise MonthlyAccountError(f"clean 状态列含未知值：{column}")
        self.quotes = {
            (str(row["trade_date"]), str(row["ts_code"])): row for row in data.to_dict("records")
        }
        self.signals = self._validate_signals(signals)
        self.current_capital = float(capital)
        self.initial_capital = float(capital)
        self.arm_name = str(arm_name)
        self.positions: dict[str, dict[str, Any]] = {}
        self.corporate = NativeCorporateAdapter(corporate_book)
        self.cost = CostModel(**FROZEN_COSTS)
        self.haircut = float(dividend_haircut)
        self.tax_liability = 0.0
        self.tax_paid = 0.0
        self.total_fees = 0.0
        self.pending_sells: dict[str, dict[str, Any]] = {}
        self.plans: dict[str, list[dict[str, Any]]] = {}
        self.daily: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.holdings: list[dict[str, Any]] = []

    def _validate_signals(self, signals: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        required = {"signal_date", "ts_code", "eligible"}
        if not required.issubset(signals.columns):
            raise MonthlyAccountError(f"信号缺列：{sorted(required - set(signals.columns))}")
        result = {}
        for day, group in signals.groupby("signal_date", sort=True):
            day = _day(day)
            if day not in self.days:
                raise MonthlyAccountError(f"信号日不在日历：{day}")
            if len(group) != 4 or set(group["ts_code"]) != set(self.codes):
                raise MonthlyAccountError(f"{day} 信号必须恰有四个固定代码")
            if not group["eligible"].map(lambda value: isinstance(value, bool)).all():
                raise MonthlyAccountError(f"{day} eligible 必须显式为布尔，未知不能当 False")
            result[day] = group.sort_values("ts_code").to_dict("records")
        return result

    def _plan(self, day: str, index: int, equity: float) -> None:
        rows = self.signals.get(day, [])
        if not rows:
            return
        if index + 1 >= len(self.days):
            self._note(day, "", "right_censored_signal")
            return
        execution = self.days[index + 1]
        self.plans[execution] = [
            {
                "signal_date": day,
                "ts_code": row["ts_code"],
                "eligible": row["eligible"],
                "notional": equity / 4 if row["eligible"] else 0.0,
            }
            for row in rows
        ]
        for plan in self.plans[execution]:
            self._note(
                day,
                plan["ts_code"],
                "planned",
                execution_date=execution,
                signal_date=day,
                eligible=plan["eligible"],
                notional=plan["notional"],
            )

    def run(self) -> dict[str, Any]:
        for index, day in enumerate(self.days):
            trade_start = len(self.trades)
            self._corporate_day(day)
            plans = self.plans.get(day, [])
            self._queue_sells(day, plans)
            self._sell(day)
            self._buy(day, plans)
            equity = self._close(day, self.trades[trade_start:])
            self.corporate.after_day(self, day)
            self._plan(day, index, equity)
        return {
            "arm_name": self.arm_name,
            "initial_capital": self.initial_capital,
            "daily": pd.DataFrame(self.daily),
            "trades": pd.DataFrame(
                self.trades,
                columns=[
                    "trade_date",
                    "signal_date",
                    "ts_code",
                    "side",
                    "shares",
                    "price",
                    "amount",
                    "fee",
                    "price_kind",
                    "cash_after",
                ],
            ),
            "decisions": (
                pd.DataFrame(self.decisions)
                if self.decisions
                else pd.DataFrame(columns=["trade_date", "ts_code", "kind"])
            ),
            "holdings": pd.DataFrame(
                self.holdings,
                columns=[
                    "trade_date",
                    "ts_code",
                    "shares",
                    "price",
                    "market_value",
                    "valuation_source",
                    "locked_shares",
                ],
            ),
            "corporate": self.corporate.payload(),
            "final_state": {
                "cash": self.current_capital,
                "positions": copy.deepcopy(self.positions),
                "pending_sells": copy.deepcopy(self.pending_sells),
                "receivable": self.corporate.receivable,
                "dividend_haircut_liability": self.tax_liability,
            },
            "assumptions": {
                "costs": dict(FROZEN_COSTS),
                "slot_count": 4,
                "dividend_haircut": self.haircut,
                "dividend_haircut_basis": "额外扣减既有 T4 口径已入账红利的压力，非精确个人税",
                "cash_interest": 0.0,
                "not_production": True,
                "historical_only": True,
            },
        }


DAYS = tuple(day.strftime("%Y%m%d") for day in pd.bdate_range("2026-01-05", periods=8))
CODES = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ")


def _inputs(tmp_path: Path, count: int, *, one_selected: bool = False) -> dict:
    codes = CODES[:count]
    quotes = pd.DataFrame(
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
            for code in codes
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "signal_date": DAYS[0],
                "ts_code": code,
                "eligible": not one_selected or code == codes[0],
            }
            for code in codes
        ]
    )
    directory = tmp_path / "dividend"
    directory.mkdir(exist_ok=True)
    pd.DataFrame(
        [
            {
                "ts_code": codes[0],
                "end_date": "20251231",
                "ann_date": "20260101",
                "imp_ann_date": "20260102",
                "div_proc": "预案",
                "cash_div": 0.0,
                "cash_div_tax": 0.0,
                "stk_div": 0.0,
                "record_date": DAYS[1],
                "ex_date": DAYS[2],
                "pay_date": DAYS[4],
                "div_listdate": DAYS[2],
            }
        ]
    ).to_parquet(directory / "2025.parquet", index=False)
    return dict(
        calendar=DAYS,
        quotes=quotes,
        signals=signals,
        capital=4100.0,
        corporate_book=NativeCorporateBook(calendar=DAYS, dividend_dir=directory),
        arm_name="synthetic",
        codes=codes,
        dividend_haircut=0.0,
    )


def test_two_slots_use_half_account_and_real_fees(tmp_path: Path) -> None:
    result = run_polyester_account(**_inputs(tmp_path, 2))
    assert result["assumptions"]["slot_count"] == 2
    assert result["trades"].shares.tolist() == [200, 200]
    assert result["trades"].trade_date.tolist() == [DAYS[1], DAYS[1]]
    assert result["daily"].iloc[-1].cash == pytest.approx(88.0)
    assert result["daily"].cash.min() >= 0
    plans = result["decisions"].query("kind == 'planned'")
    assert plans.notional.tolist() == [2050.0, 2050.0]


def test_unselected_slot_keeps_cash_not_rescaled(tmp_path: Path) -> None:
    result = run_polyester_account(**_inputs(tmp_path, 2, one_selected=True))
    assert result["trades"].shares.tolist() == [200]
    assert result["daily"].iloc[-1].cash == pytest.approx(2094.0)


@pytest.mark.parametrize("count", [1, 3, 4])
def test_two_slot_entry_rejects_wrong_count(tmp_path: Path, count: int) -> None:
    with pytest.raises(MonthlyAccountError, match="固定2槽"):
        run_polyester_account(**_inputs(tmp_path, count))


def test_two_slot_signals_must_include_both_codes(tmp_path: Path) -> None:
    values = _inputs(tmp_path, 2)
    values["signals"] = values["signals"].iloc[:1]
    with pytest.raises(MonthlyAccountError, match="恰有2个"):
        run_polyester_account(**values)


@pytest.mark.parametrize("slots", [True, False, 0, -1, 2.0, "2", None])
def test_slot_count_rejects_nonpositive_or_noninteger(tmp_path: Path, slots: Any) -> None:
    with pytest.raises(MonthlyAccountError, match="正整数"):
        _MonthlyAccount(**_inputs(tmp_path, 2), expected_slots=slots)


def test_old_public_entry_still_requires_four(tmp_path: Path) -> None:
    with pytest.raises(MonthlyAccountError, match="四个"):
        run_monthly_account(**_inputs(tmp_path, 2))


def test_four_slots_match_prechange_snapshot_exactly(tmp_path: Path) -> None:
    values = _inputs(tmp_path, 4)
    # 同一个合成夹具含送股锁定、现金应收、跌停延期和新信号，不只检查空账户。
    directory = tmp_path / "dividend"
    pd.DataFrame(
        [
            {
                "ts_code": CODES[0],
                "end_date": "20251231",
                "ann_date": "20260101",
                "imp_ann_date": "20260102",
                "div_proc": "实施",
                "cash_div": 1.0,
                "cash_div_tax": 1.0,
                "stk_div": 0.2,
                "record_date": DAYS[1],
                "ex_date": DAYS[2],
                "pay_date": DAYS[5],
                "div_listdate": DAYS[5],
            }
        ]
    ).to_parquet(directory / "2025.parquet", index=False)
    values["corporate_book"] = NativeCorporateBook(calendar=DAYS, dividend_dir=directory)
    data = values["quotes"]
    data.loc[(data.ts_code == CODES[0]) & data.trade_date.ge(DAYS[2]), ["open", "close"]] = 7.5
    data.loc[(data.ts_code == CODES[0]) & data.trade_date.eq(DAYS[3]), "is_limit_down"] = 1
    values["signals"] = pd.concat(
        [
            values["signals"],
            pd.DataFrame(
                [{"signal_date": DAYS[2], "ts_code": code, "eligible": False} for code in CODES]
            ),
            pd.DataFrame(
                [{"signal_date": DAYS[4], "ts_code": code, "eligible": True} for code in CODES]
            ),
        ],
        ignore_index=True,
    )
    old = _LegacyFourSlotAccount(**values).run()
    new = run_monthly_account(**values)
    assert old.keys() == new.keys()
    for key in old:
        if isinstance(old[key], pd.DataFrame):
            pd.testing.assert_frame_equal(old[key], new[key], check_exact=True)
        else:
            assert old[key] == new[key], key
    assert new["corporate"]["lock_count"] > 0
    assert not new["trades"].empty
