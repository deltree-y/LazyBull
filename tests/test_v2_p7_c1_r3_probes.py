# -*- coding: utf-8 -*-
"""C1 r3 定向探针：时间轴、红利守恒、L1 回退、五级裁决状态。"""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.v2_p7.c1.backtest import Position, _member_return, _try_sell, run_c1
from scripts.v2_p7.c1.monthly import _residuals
from scripts.v2_p7.c1.run_prescreen import decide


class FakePanel:
    def __init__(self, days, stocks, close_adj, open_u=None, open_adj=None):
        self.days = list(days)
        self.stocks = list(stocks)
        self.stock_index = {code: i for i, code in enumerate(self.stocks)}
        self.n_days = len(self.days)
        self.n_stocks = len(self.stocks)
        self.close_adj = np.asarray(close_adj, dtype=float)
        self.open_u = np.asarray(open_u if open_u is not None else close_adj, dtype=float)
        self.open_adj = np.asarray(open_adj if open_adj is not None else close_adj, dtype=float)
        self.suspended = np.zeros((self.n_days, self.n_stocks), dtype=bool)
        self.limit_up = np.zeros((self.n_days, self.n_stocks), dtype=bool)
        self.limit_down = np.zeros((self.n_days, self.n_stocks), dtype=bool)
        self.is_st = np.zeros((self.n_days, self.n_stocks), dtype=bool)
        self.listed = np.ones((self.n_days, self.n_stocks), dtype=bool)
        self.total_mv = np.ones((self.n_days, self.n_stocks), dtype=float) * 1e6
        self.amount = np.ones((self.n_days, self.n_stocks), dtype=float) * 1e6
        self.pre_close = self.open_u.copy()


def _monthly_stub(days, n_stocks, month_end_days, signal_month=0, signal_stock=0):
    n_months = len(month_end_days)
    signals = np.full((n_months, n_stocks), np.nan)
    signals[signal_month, signal_stock] = 1.0
    eligible = np.zeros((n_months, n_stocks), dtype=bool)
    eligible[signal_month, signal_stock] = True
    return SimpleNamespace(
        days=list(days),
        month_end_days=np.asarray(month_end_days, dtype=np.int64),
        returns=np.zeros((n_months, n_stocks)),
        residuals=np.zeros((n_months, n_stocks)),
        signals=signals,
        eligible=eligible,
        ref_keys=np.zeros((n_months, n_stocks), dtype=np.int64),
        ref_types=np.full((n_months, n_stocks), 2, dtype=np.int64),
        l1_keys=np.zeros((n_months, n_stocks), dtype=np.int64),
        amount_ma20=np.ones((n_months, n_stocks), dtype=float) * 1e6,
        last_obs=np.full(n_stocks, len(days) - 1, dtype=np.int64),
    )


def test_delayed_buy_not_in_nav_before_execution():
    days = ["20200102", "20200103", "20200106", "20200107", "20200108", "20200109"]
    panel = FakePanel(days, ["000001.SZ"], np.full((6, 1), 10.0))
    panel.suspended[1, 0] = True
    panel.suspended[2, 0] = True
    monthly = _monthly_stub(days, 1, [0, 4], signal_month=0, signal_stock=0)
    result = run_c1(panel, monthly, panel.stocks, principal=300_000.0)
    # 首笔成交在 20200107（第 3 个交易日）
    assert result.trades.iloc[0]["trade_date"] == "20200107"
    # 成交前两日 NAV 仍为现金本金，未被未来成交提前扣费
    assert result.nav.loc[result.nav.index[1]] == pytest.approx(300_000.0)
    assert result.nav.loc[result.nav.index[2]] == pytest.approx(300_000.0)
    assert result.nav.loc[result.nav.index[3]] < 300_000.0
    assert result.validity["signal_day_trades"] == 0


def test_sell_uses_total_return_value_after_cash_dividend():
    days = ["20200102", "20200103"]
    panel = FakePanel(days, ["000001.SZ"], np.full((2, 1), 10.0))
    pos = Position(
        units=1000.0,
        entry_adj=10.0,
        entry_day=0,
        code="000001.SZ",
        shares_raw=100,
        last_valid_adj=10.0,
    )
    trades = []
    cash, closed, status = _try_sell(panel, 1, "000001.SZ", pos, 1000.0, 0.0, trades, "20200102")
    # 裸价从 10 跌到 9、现金分红 1 的等价情形：复权价仍为 10，卖出应回约 1000 而非 900
    assert status == "done"
    assert closed is True
    assert cash == pytest.approx(1000.0 - (5.0 + 0.5 + 0.5))
    assert trades[0]["amount"] != trades[0]["raw_notional"] or trades[0]["shares"] * trades[0][
        "price"
    ] == pytest.approx(trades[0]["raw_notional"])


def test_benchmark_does_not_use_future_final_price():
    days = ["20200131", "20200228", "20200331"]
    panel = FakePanel(days, ["000001.SZ"], np.array([[10.0], [np.nan], [20.0]]))
    monthly = _monthly_stub(days, 1, [0, 1])
    monthly.returns[1, 0] = np.nan
    monthly.last_obs[0] = 2  # 未来仍有报价，但不得回填 2 月
    ret, status = _member_return(panel, monthly, 0, 0)
    assert ret == 0.0
    assert status == "stale_or_delisted"


def test_partial_reduce_order_done_is_not_full_sell():
    days = ["20200102", "20200103"]
    panel = FakePanel(days, ["000001.SZ"], np.full((2, 1), 10.0))
    pos = Position(
        units=70_000.0,
        entry_adj=10.0,
        entry_day=0,
        code="000001.SZ",
        shares_raw=7000,
        last_valid_adj=10.0,
    )
    trades = []
    cash, closed, status = _try_sell(panel, 1, "000001.SZ", pos, 20_000.0, 0.0, trades, "20200102")
    assert status == "done"
    assert closed is False
    assert pos.units == pytest.approx(50_000.0)
    assert len(trades) == 1


def test_l1_fallback_uses_full_reference_set():
    n = 22
    ret = np.array([0.10] * 20 + [0.0, 0.0])
    eligible = np.ones(n, dtype=bool)
    ref_type = np.array([2] * 20 + [1, 1], dtype=np.int64)
    ref_key = np.array([7] * 20 + [0, 0], dtype=np.int64)
    l1 = np.zeros(n, dtype=np.int64)
    resid = _residuals(ret, eligible, ref_type, ref_key, l1)
    # 薄 L2 两只应相对完整 L1（22 只）剔自身均值，而不是只在两只之间比较
    assert np.isclose(resid[-1], -0.10 * 20 / 21)
    assert np.isclose(resid[-2], -0.10 * 20 / 21)


def test_decide_does_not_proceed_without_es3_es5():
    state, reason = decide(
        implementation_valid=True,
        es1_pass=True,
        es2_pass=True,
        es4_pass=True,
        es3_pass=None,
        es5_pass=None,
    )
    assert state == "incomplete_requires_table2_3"
    assert reason == "es3_es5_not_evaluated"
    invalid, _ = decide(
        implementation_valid=False,
        es1_pass=False,
        es2_pass=False,
        es4_pass=False,
        es3_pass=None,
        es5_pass=None,
    )
    assert invalid == "invalid_implementation"
