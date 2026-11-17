"""合成账户守恒测试，覆盖交易、送转、应收与开放右端。"""

import copy

import pandas as pd
import pytest

from scripts.v2_p7.buffer_native_validation import validate_account


def _daily(day, cash, market, receivable, buy=0.0, sell=0.0, fees=0.0, positions=0):
    total = cash + market + receivable
    return {
        "trade_date": day,
        "cash": cash,
        "market_value": market,
        "receivable": receivable,
        "total_value": total,
        "nav": total / 10000.0,
        "buy_amount": buy,
        "sell_amount": sell,
        "fees": fees,
        "positions": positions,
    }


@pytest.fixture
def account():
    days = [pd.Timestamp("2026-02-02"), "20260203", "2026-02-04T00:00:00", "20260205"]
    daily = [
        _daily(days[0], 10000, 0, 0),
        _daily(days[1], 8999, 1000, 0, buy=1000, fees=1, positions=1),
        _daily(days[2], 8999, 990, 100, positions=1),
        _daily(days[3], 9188, 900, 0, sell=90, fees=1, positions=1),
    ]
    states = []
    for row, shares in zip(daily, [0, 100, 110, 100]):
        positions = (
            {}
            if not shares
            else {"000001.SZ": {"shares": shares, "buy_date": pd.Timestamp("2026-02-03")}}
        )
        states.append(
            {
                "trade_date": row["trade_date"],
                "cash": row["cash"],
                "positions": positions,
                "pending_condition_sells": {},
                "pending_stop_loss_sells": {},
            }
        )
    states[-1]["pending_condition_sells"] = {
        "000001.SZ": {"trigger_date": pd.Timestamp("2026-02-05")}
    }
    return {
        "daily": daily,
        "states": states,
        "trades": [
            {
                "date": pd.Timestamp("2026-02-03"),
                "stock": "000001.SZ",
                "action": "buy",
                "shares": 100,
                "amount": 1000,
                "cost": 1,
            },
            {
                "date": "2026-02-05T00:00:00",
                "stock": "000001.SZ",
                "action": "sell",
                "shares": 10,
                "amount": 90,
                "cost": 1,
            },
        ],
        "corporate": {
            "cash_paid_by_day": {"20260205": 100},
            "receivable": 0,
            "receivables": [],
            "rows": [
                {
                    "trade_date": "20260204",
                    "kind": "ex",
                    "ts_code": "000001.SZ",
                    "bonus_shares": 10,
                    "cash_receivable": 100,
                },
                {
                    "trade_date": pd.Timestamp("2026-02-05"),
                    "kind": "pay",
                    "ts_code": "000001.SZ",
                    "cash_paid": 100,
                },
            ],
        },
        "buffer_events": [],
        "ending": copy.deepcopy(states[-1]),
    }


def test_cash_shares_dividends_and_open_end_are_conserved(account):
    before = copy.deepcopy(account)
    result = validate_account(account, 10000)
    assert result["valid"] and result["all_passed"]
    assert result["initial_point"]["trade_date"] == "20260202"
    assert result["ending"]["positions"] == {"000001.SZ": 100}
    assert result["ending"]["pending"]["pending_condition_sells"]
    assert not result["ending"]["liquidation_required"]
    assert account == before


@pytest.mark.parametrize("field", ["cash", "shares"])
def test_corrupt_one_cash_or_share_record_is_rejected(account, field):
    if field == "cash":
        account["daily"][2]["cash"] += 1
    else:
        account["states"][2]["positions"]["000001.SZ"]["shares"] -= 1
    result = validate_account(account, 10000)
    assert not result["valid"]
    assert any(row["check"] == f"{field}_conservation" for row in result["violations"])


def test_receivable_rollforward_and_cash_paid_summary_must_match(account):
    account["corporate"]["cash_paid_by_day"]["20260205"] = 101
    result = validate_account(account, 10000)
    assert not result["valid"]
    checks = {item["check"] for item in result["violations"]}
    assert {"corporate_cash_paid_rows", "receivable_conservation", "cash_conservation"} <= checks


def test_explicit_initial_point_supports_first_actual_trading_day(account):
    point = account["daily"].pop(0)
    state = account["states"].pop(0)
    account["initial_point"] = {**point, "positions": state["positions"]}
    result = validate_account(account, 10000)
    assert result["valid"]
    assert result["initial_point"]["source"] == "explicit_initial_point"


def test_non_cash_first_day_without_anchor_is_not_invented(account):
    account["daily"].pop(0)
    account["states"].pop(0)
    result = validate_account(account, 10000)
    assert not result["valid"]
    assert any(row["check"] == "initial_cash" for row in result["violations"])


def test_unpaid_dividend_and_pending_sell_are_legal_at_right_endpoint(account):
    account["daily"].pop()
    account["states"].pop()
    account["trades"].pop()
    account["corporate"]["rows"].pop()
    account["corporate"]["cash_paid_by_day"] = {}
    account["corporate"]["receivable"] = 100
    account["corporate"]["receivables"] = [{"ts_code": "000001.SZ", "amount": 100}]
    account["ending"] = copy.deepcopy(account["states"][-1])
    result = validate_account(account, 10000)
    assert result["valid"]
    assert result["ending"]["receivable"] == 100


def test_missing_or_duplicate_daily_date_is_rejected(account):
    account["states"][2]["trade_date"] = "20260203"
    with pytest.raises(ValueError, match="日期"):
        validate_account(account, 10000)


def test_nonfinite_cash_cannot_pass(account):
    account["daily"][1]["cash"] = float("nan")
    with pytest.raises(ValueError, match="有限"):
        validate_account(account, 10000)


def test_renewal_observes_actual_buy_date_without_modifying_it(account):
    account["buffer_events"] = [
        {
            "trade_date": "20260204",
            "kind": "expiry",
            "renew": True,
            "ts_code": "000001.SZ",
            "buy_date": "20260203",
        }
    ]
    result = validate_account(account, 10000)
    assert result["valid"]
    assert result["holding_date_observations"]["matched_renewals"] == 1
    account["states"][2]["positions"]["000001.SZ"]["buy_date"] = "20260204"
    result = validate_account(account, 10000)
    assert not result["valid"]
    assert any(row["check"] == "renewal_preserves_buy_date" for row in result["violations"])


def test_unmatched_renewal_is_diagnostic_only(account):
    account["buffer_events"] = [
        {
            "trade_date": "20260204",
            "kind": "expiry",
            "renew": True,
            "ts_code": "000002.SZ",
            "buy_date": "20260203",
        }
    ]
    result = validate_account(account, 10000)
    assert result["valid"]
    assert result["holding_date_observations"]["unmatched_renewals"] == 1


def test_twenty_one_positions_are_rejected():
    initial = _daily("20260202", 10000, 0, 0)
    later = _daily("20260203", 9979, 21, 0, buy=21, positions=21)
    codes = [f"{index:06d}.SZ" for index in range(21)]
    positions = {code: {"shares": 1} for code in codes}
    result = validate_account(
        {
            "daily": [initial, later],
            "states": [
                {"trade_date": "20260202", "cash": 10000, "positions": {}},
                {"trade_date": "20260203", "cash": 9979, "positions": positions},
            ],
            "trades": [
                {
                    "date": "20260203",
                    "stock": code,
                    "action": "buy",
                    "shares": 1,
                    "amount": 1,
                    "cost": 0,
                }
                for code in codes
            ],
        },
        10000,
    )
    assert not result["valid"]
    assert any(row["check"] == "maximum_20_positions" for row in result["violations"])


@pytest.fixture
def locked_native_account(tmp_path, request):
    """完整原生账户 + 真实公司行为 API 产出锁事件，不手造 rows。"""
    from scripts.v2_p7.buffer_native_corporate import NativeCorporateBook, NativeCorporateMixin
    from scripts.v2_p7.buffer_native_replay import (
        FrozenRankSignal,
        RecordNativeMixin,
        result_payload,
    )
    from src.lazybull.v2.common.cost import CostModel
    from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger
    from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML

    class SyntheticUniverse:
        def get_stocks(self, date, quote_data=None):
            return ["000001.SZ"]

    class SyntheticStorage:
        def load_raw_by_date(self, dataset, trade_date):
            assert dataset == "suspend"
            return pd.DataFrame(columns=["ts_code", "suspend_type"])

    class IntegratedEngine(RecordNativeMixin, NativeCorporateMixin, NativeMergeBacktestEngineML):
        pass

    dates = list(pd.bdate_range("2026-01-05", periods=8))
    days = [date.strftime("%Y%m%d") for date in dates]
    dividend = []
    for record, ex, listing, ratio in [
        (days[2], days[3], days[5], 0.2),
        (days[3], days[4], days[6], 0.1),
    ]:
        dividend.append(
            {
                "ts_code": "000001.SZ",
                "end_date": "20251231",
                "ann_date": "20260101",
                "div_proc": "实施",
                "stk_div": ratio,
                "cash_div": 0.0,
                "cash_div_tax": 0.0,
                "record_date": record,
                "ex_date": ex,
                "pay_date": listing,
                "imp_ann_date": "20260102",
                "div_listdate": listing,
            }
        )
    pd.DataFrame(dividend).to_parquet(tmp_path / "2025-12-31.parquet", index=False)
    book = NativeCorporateBook(calendar=days, dividend_dir=tmp_path)
    prices = []
    for i, day in enumerate(days):
        price = 10.0 / (1.2 if i >= 3 else 1.0) / (1.1 if i >= 4 else 1.0)
        prices.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": day,
                "open": price,
                "close": price,
                "open_adj": 10.0,
                "close_adj": 10.0,
                "is_st": False,
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
                "tradable": True,
            }
        )
    ranked = {day: [("000001.SZ", 1.0)] for day in days}
    engine = IntegratedEngine(
        portfolio_merger=SingleSleevePortfolioMerger(),
        universe=SyntheticUniverse(),
        signal=FrozenRankSignal(ranked),
        features_by_date={day: pd.DataFrame({"x": [1]}) for day in days},
        initial_capital=1000000,
        cost_model=CostModel(0.0001954, 5.0, 0.0005, 0.0005),
        rebalance_freq=20,
        stagger_tranches=2,
        holding_period=20,
        sell_timing="open",
        verbose=False,
        position_sizing="equal",
        max_weight_per_stock=0.15,
        min_buy_value_ratio=0.2,
        data_storage=SyntheticStorage(),
        completion_window_days=5,
        corporate_book=book,
    )
    count = getattr(request, "param", len(dates))
    engine.run(dates[0], dates[count - 1], dates, pd.DataFrame(prices))
    return result_payload(engine), engine, dates[:count]


def test_real_native_corporate_v2_payload_accumulates_and_releases_locks(locked_native_account):
    payload, _, dates = locked_native_account
    assert payload["corporate"]["schema"] == "p7_buffer_native_corporate/v2"
    assert payload["corporate"]["lock_count"] == payload["corporate"]["unlock_count"] == 2
    result = validate_account(payload, 1000000)
    assert result["valid"], result["violations"]
    history = {
        row["trade_date"]: row["locked_shares"].get("000001.SZ", 0)
        for row in result["locked_shares_by_day"]
    }
    assert 0 < history[dates[3].strftime("%Y%m%d")] < history[dates[4].strftime("%Y%m%d")]
    assert 0 < history[dates[5].strftime("%Y%m%d")] < history[dates[4].strftime("%Y%m%d")]
    assert history[dates[6].strftime("%Y%m%d")] == 0
    assert result["ending"]["locked_shares"] == {}


def test_unlock_must_match_original_event_quantity(locked_native_account):
    payload, _, _ = locked_native_account
    row = next(row for row in payload["corporate"]["rows"] if row["kind"] == "unlock")
    row["shares"] += 1
    result = validate_account(payload, 1000000)
    assert not result["valid"]
    assert any(row["check"] == "unlock_matches_active_event" for row in result["violations"])


@pytest.mark.parametrize("locked_native_account", [5], indirect=True)
def test_terminal_locks_are_valid_without_forced_unlock(locked_native_account):
    payload, _, _ = locked_native_account
    assert len(payload["corporate"]["locks"]) == 2
    result = validate_account(payload, 1000000)
    assert result["valid"], result["violations"]
    assert result["ending"]["locked_shares"]["000001.SZ"] > 0


def test_daily_locked_quantity_cannot_exceed_total_position(locked_native_account):
    payload, _, _ = locked_native_account
    lock = next(row for row in payload["corporate"]["rows"] if row["kind"] == "lock")
    lock["shares"] = 100000000
    result = validate_account(payload, 1000000)
    assert not result["valid"]
    assert any(row["check"] == "locked_shares_within_total" for row in result["violations"])


def test_locked_stock_sale_is_rejected_even_when_total_shares_suffice(locked_native_account):
    payload, _, dates = locked_native_account
    day = dates[4].strftime("%Y%m%d")
    total = payload["states"][4]["positions"]["000001.SZ"]["shares"]
    payload["trades"].append(
        {
            "date": day,
            "stock": "000001.SZ",
            "action": "sell",
            "shares": total,
            "amount": 1.0,
            "cost": 0.0,
        }
    )
    result = validate_account(payload, 1000000)
    assert not result["valid"]
    assert any(row["check"] == "sell_within_unlocked_shares" for row in result["violations"])


def test_unlock_never_adds_or_removes_total_stock(locked_native_account):
    payload, _, _ = locked_native_account
    payload["states"][5]["positions"]["000001.SZ"]["shares"] += 1
    result = validate_account(payload, 1000000)
    assert not result["valid"]
    assert any(row["check"] == "shares_conservation" for row in result["violations"])


def test_actual_adapter_block_evidence_is_never_valid(locked_native_account):
    """用真实 check_sell 生成阻断 API 行，不手写新 kind 的字段。"""
    from scripts.v2_p7.t4.corporate import CorporateError

    payload, engine, dates = locked_native_account
    adapter = engine.native_corporate
    lock_row = next(row for row in adapter.rows if row["kind"] == "lock")
    adapter.locks = [
        {key: value for key, value in lock_row.items() if key not in {"kind", "trade_date"}}
    ]
    with pytest.raises(CorporateError):
        adapter.check_sell(engine, dates[-1], "000001.SZ", engine.positions["000001.SZ"]["shares"])
    payload["corporate"] = adapter.payload()
    result = validate_account(payload, 1000000)
    assert not result["valid"]
    assert any(row["check"] == "corporate_sell_blocked_locked" for row in result["violations"])
