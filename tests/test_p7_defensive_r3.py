"""DEF-ML-60-v1 R3 合成执行证据：相位、费用、行业上限、停牌顺延、锁股与守恒。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.defensive_ml import b_account, b_replay, spec

DAYS = [f"2026{month:02d}{day:02d}" for month in range(1, 9) for day in (2, 9, 16, 23)]
DAYS = sorted(DAYS)[:40]


def _frames(n: int, codes: list[str], price: float = 10.0) -> dict[str, pd.DataFrame]:
    days = DAYS[:n]
    make = lambda value: pd.DataFrame(  # noqa: E731
        np.full((len(days), len(codes)), value, dtype=float), index=days, columns=codes
    )
    return {
        "open": make(price),
        "close": make(price),
        "is_suspended": make(0.0),
        "is_limit_up": make(0.0),
        "is_limit_down": make(0.0),
        "tradable": make(1.0),
    }


def _ranks(n: int, codes: list[str]) -> pd.DataFrame:
    rows = []
    for day in DAYS[:n]:
        for rank, code in enumerate(codes, start=1):
            rows.append({"trade_date": day, "ts_code": code, "rank": rank})
    return pd.DataFrame(rows)


def _industry(n: int, codes: list[str], buckets: dict[str, str]) -> pd.DataFrame:
    rows = []
    for day in DAYS[:n]:
        for code in codes:
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": code,
                    "industry_bucket": buckets.get(code, "UNKNOWN"),
                }
            )
    return pd.DataFrame(rows)


def _config(**overrides) -> b_replay.ReplayConfig:
    base = dict(
        top_n=4,
        holding_period=5,
        rebalance=5,
        tranches=(2, 2),
        offsets=(0, 20),
        max_per_industry=5,
        min_scored=0,
        buy_retry_days=3,
        buys_per_day_retry=1,
    )
    base.update(overrides)
    return b_replay.ReplayConfig(**base)


def _run(prices, ranks, industry, capital=1_000_000.0, config=None):
    return b_replay.run_b_arm(
        prices=prices,
        ranks=ranks,
        industry=industry,
        calendar=DAYS[: len(prices["close"].index)],
        start=str(prices["close"].index[0]),
        end=str(prices["close"].index[-1]),
        capital=capital,
        arm="B_TEST",
        corporate=None,
        config=config or _config(),
    )


def test_buys_execute_at_next_day_close_with_fees() -> None:
    codes = ["A", "B", "C", "D"]
    prices = _frames(12, codes)
    result = _run(prices, _ranks(12, codes), _industry(12, codes, {}))
    trades = result.trades
    buys = trades[trades["action"] == "buy"]
    assert not buys.empty
    first_day = str(prices["close"].index[0])
    second_day = str(prices["close"].index[1])
    assert buys["trade_date"].min() == second_day, "信号日 T 不得当日买"
    assert buys["trade_date"].iloc[0] != first_day
    expected_price = 10.0 * (1.0 + spec.COSTS["slippage"])
    assert buys["price"].iloc[0] == pytest.approx(expected_price)
    assert buys["fee"].iloc[0] >= spec.COSTS["min_commission"]
    assert (buys["shares"] % 100 == 0).all()


def test_sells_execute_at_open_after_holding_period() -> None:
    codes = ["A"]
    prices = _frames(20, codes)
    prices["open"] = prices["open"] * 1.0
    prices["close"] = prices["close"] * 1.0
    cfg = _config(top_n=1, tranches=(1,), offsets=(0,), holding_period=5, rebalance=5)
    result = _run(prices, _ranks(20, codes), _industry(20, codes, {}), config=cfg)
    trades = result.trades
    sells = trades[trades["action"] == "sell"]
    assert len(sells) >= 1
    buy_day = trades[trades["action"] == "buy"]["trade_date"].iloc[0]
    sell_day = sells["trade_date"].iloc[0]
    buy_index = DAYS.index(buy_day)
    sell_index = DAYS.index(sell_day)
    assert sell_index - buy_index == 5
    assert sells["price"].iloc[0] == pytest.approx(10.0 * (1.0 - spec.COSTS["slippage"]))
    expected_fee = (
        max(spec.COSTS["commission_rate"] * sells["amount"].iloc[0], spec.COSTS["min_commission"])
        + spec.COSTS["stamp_tax"] * sells["amount"].iloc[0]
    )
    assert sells["fee"].iloc[0] == pytest.approx(expected_fee)


def test_industry_cap_limits_holdings() -> None:
    codes = ["A", "B", "C", "D"]
    prices = _frames(12, codes)
    industry = _industry(12, codes, {code: "SAME" for code in codes})
    cfg = _config(top_n=4, tranches=(4,), offsets=(0,), max_per_industry=2)
    result = _run(prices, _ranks(12, codes), industry, config=cfg)
    held = result.daily["positions"].max()
    assert held <= 2


def test_suspended_buy_deferred_then_filled() -> None:
    codes = ["A"]
    prices = _frames(12, codes)
    second = DAYS[1]
    prices["is_suspended"].loc[second, "A"] = 1.0
    cfg = _config(top_n=1, tranches=(1,), offsets=(0,), buy_retry_days=3)
    result = _run(prices, _ranks(12, codes), _industry(12, codes, {}), config=cfg)
    buys = result.trades[result.trades["action"] == "buy"]
    assert not buys.empty
    assert buys["trade_date"].iloc[0] == DAYS[2], "停牌日必须顺延，不得用停牌价成交"
    assert result.diagnostics["stats"]["buy_deferred"] >= 1


def test_limit_up_blocks_buy_until_next_day() -> None:
    codes = ["A"]
    prices = _frames(12, codes)
    prices["is_limit_up"].loc[DAYS[1], "A"] = 1.0
    cfg = _config(top_n=1, tranches=(1,), offsets=(0,), buy_retry_days=3)
    result = _run(prices, _ranks(12, codes), _industry(12, codes, {}), config=cfg)
    buys = result.trades[result.trades["action"] == "buy"]
    assert buys["trade_date"].iloc[0] == DAYS[2]


def test_conservation_and_cash_nonnegative() -> None:
    codes = ["A", "B", "C", "D"]
    prices = _frames(24, codes)
    result = _run(prices, _ranks(24, codes), _industry(24, codes, {}))
    daily = result.daily
    assert (daily["cash"] >= -1e-6).all()
    recomputed = daily["cash"] + daily["market_value"] + daily["receivable"]
    assert np.allclose(recomputed, daily["total_value"], atol=1e-6)
    assert daily["positions"].max() <= 4


def test_blocked_sell_keeps_obligation() -> None:
    codes = ["A"]
    prices = _frames(20, codes)
    cfg = _config(top_n=1, tranches=(1,), offsets=(0,), holding_period=3, rebalance=3)
    result = _run(prices, _ranks(20, codes), _industry(20, codes, {}), config=cfg)
    sell_days = result.trades[result.trades["action"] == "sell"]["trade_date"].tolist()
    if sell_days:
        prices_down = prices["is_limit_down"].copy()
        prices_down.loc[sell_days[0], "A"] = 1.0
        blocked = _frames(20, codes)
        blocked["is_limit_down"] = prices_down
        result2 = _run(blocked, _ranks(20, codes), _industry(20, codes, {}), config=cfg)
        first_sell = result2.trades[result2.trades["action"] == "sell"]["trade_date"].iloc[0]
        assert first_sell > sell_days[0], "跌停必须顺延卖出，义务不得消失"


def test_bonus_shares_locked_until_next_trading_day() -> None:
    account = b_account.BAccount(capital=100_000.0, arm="t")
    account.buy("20260105", "A", 1000, 10.0, tranche=0)

    class _Event:
        ts_code = "A"
        cash_per_share = 0.5
        share_ratio = 0.1
        end_date = "20251231"

    account.apply_ex_date("20260106", [_Event()], "20260107")
    lot = next(iter(account.lots.values()))
    assert lot.shares == 1100
    assert lot.sellable("20260106") == 1000
    assert lot.sellable("20260107") == 1100
    assert account.receivable == pytest.approx(500.0)
    paid = account.apply_pay_date("20260110", [_Event()])
    assert paid == pytest.approx(500.0)
    assert account.receivable == pytest.approx(0.0)


def test_receivable_settles_on_its_pay_day() -> None:
    account = b_account.BAccount(capital=100_000.0, arm="t")
    account.buy("20260105", "A", 1000, 10.0, tranche=0)

    class _Event:
        ts_code = "A"
        cash_per_share = 0.5
        share_ratio = 0.0
        end_date = "20251231"
        pay_trade_day = "20260112"

    account.apply_ex_date("20260106", [_Event()], "20260107")
    assert account.receivable == pytest.approx(500.0)
    assert account.settle_due("20260109") == 0.0
    assert account.settle_due("20260112") == pytest.approx(500.0)
    assert account.receivable == pytest.approx(0.0)
    # 买入价含 5bp 滑点：10.005 × 1000 + 5 佣金
    assert account.cash == pytest.approx(100_000.0 - 10_005.0 - 5.0 + 500.0)


def test_min_commission_applies_per_order() -> None:
    account = b_account.BAccount(capital=100_000.0, arm="t")
    record = account.buy("20260105", "A", 100, 10.0, tranche=0)
    assert record["fee"] == pytest.approx(spec.COSTS["min_commission"])
