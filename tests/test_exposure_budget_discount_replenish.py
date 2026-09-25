# -*- coding: utf-8 -*-
"""建仓折扣回补（A3 v2 / P2-4 镜像缺口）单元测试。

硬契约（预登记 docs/plans/slot_refill_ab_prereg.md §2）：
1. **默认关闭逐位一致**：`exposure_budget_discount_replenish=False` 或 λ=1 时记账无任何副作用；
2. **记账口径**：折扣额 = 实际成交额 × (1/λ − 1)（λ<1 信号日）；
3. **必须与回补同开**：单开 `budget_discount_replenish` 直接报错（记了没人回补）；
4. **回补全链路复用 P2-4**：λ 恢复 + 释放额存在 + 容差外 → T+1 买回；新调仓计划日清零；
5. 释放额与减仓释放共用同一余额（`exposure_release_budget`），统计字段独立累计。
"""

import pandas as pd
import pytest

from src.lazybull.backtest.buy_execution import BacktestBuyExecutionMixin
from src.lazybull.backtest.exposure_override import BacktestExposureOverrideMixin
from src.lazybull.backtest.exposure_replenish import BacktestExposureReplenishMixin
from src.lazybull.backtest.exposure_trim import BacktestExposureTrimMixin
from src.lazybull.common.cost import CostModel

D0 = pd.Timestamp("2024-01-02")
D1 = pd.Timestamp("2024-01-03")
TRADING_DATES = [D0, D1]
DATE_TO_IDX = {D0: 0, D1: 1}


def _position(shares: int, buy_trade_price: float) -> dict:
    return {
        "shares": shares,
        "buy_date": D0,
        "signal_date": D0,
        "buy_trade_price": buy_trade_price,
        "buy_pnl_price": buy_trade_price,
        "buy_cost_cash": shares * buy_trade_price,
    }


class _StubEngine(
    BacktestExposureReplenishMixin,
    BacktestExposureTrimMixin,
    BacktestExposureOverrideMixin,
    BacktestBuyExecutionMixin,
):
    """承载记账与回补判定/执行的最小桩（沿 test_backtest_exposure_replenish.py 模式）。"""

    def __init__(self, positions, cash, prices):
        self.positions = positions
        self.current_capital = cash
        self.prices = prices
        self.pending_condition_sells = {}
        self.pending_stop_loss_sells = {}
        self.pending_signals = {}
        self.unfilled_slots = {}
        self.pending_order_manager = None
        self.stop_loss_monitor = None
        self.verbose = False
        self.sell_timing = "open"
        self.trades = []
        self.cost_model = CostModel()
        self.exposure_table = None
        self.price_data_cache = None
        self.enable_pending_order = False
        self._init_exposure_trim_state()
        self._init_exposure_replenish_state()

    def _get_trade_price(self, date, stock):
        return self.prices.get(stock, (None, None))[1]

    def _get_trade_price_open(self, date, stock):
        return self.prices.get(stock, (None, None))[0]

    def _get_pnl_price(self, date, stock):
        return self._get_trade_price(date, stock)

    def _get_pnl_price_open(self, date, stock):
        return self._get_trade_price_open(date, stock)

    def _position_market_value(self, date, stock):
        info = self.positions.get(stock)
        if not info:
            return 0.0
        price = self._get_trade_price(date, stock)
        if price is None or not (price > 0):
            return 0.0
        return float(info["shares"]) * float(price)

    def _calculate_portfolio_value(self, date):
        total = self.current_capital
        for stock, info in self.positions.items():
            total += info["shares"] * self._get_trade_price(date, stock)
        return total

    def _can_replenish_now(self, date, stock):
        return True, ""


def _engine(positions, cash, prices, table, replenish=True, budget=False):
    engine = _StubEngine(positions, cash, prices)
    engine.set_exposure_table(
        table, verbose=False, replenish=replenish, budget_discount_replenish=budget
    )
    return engine


HALF_TABLE = {"20240102": 0.5, "20240103": 1.0}


# ---------------------------------------------------------------- 记账契约
def test_disabled_no_side_effects():
    """开关关闭时 λ<1 成交也不记账（逐位一致）。"""
    engine = _engine(
        {"600000.SH": _position(1000, 10.0)}, 5000.0,
        {"600000.SH": (10.0, 10.0)}, HALF_TABLE, replenish=True, budget=False,
    )
    engine._record_budget_discount_release(D0, 100000.0)
    assert engine.exposure_release_budget == 0.0


def test_lambda_one_no_side_effects():
    """λ=1 信号日不记账（即使开关开）。"""
    engine = _engine(
        {"600000.SH": _position(1000, 10.0)}, 5000.0,
        {"600000.SH": (10.0, 10.0)}, HALF_TABLE, replenish=True, budget=True,
    )
    engine._record_budget_discount_release(D1, 100000.0)
    assert engine.exposure_release_budget == 0.0


def test_discount_formula():
    """折扣额 = 成交额 × (1/λ − 1)：λ=0.5、成交 100 → 记 100。"""
    engine = _engine(
        {"600000.SH": _position(1000, 10.0)}, 5000.0,
        {"600000.SH": (10.0, 10.0)}, HALF_TABLE, replenish=True, budget=True,
    )
    engine._record_budget_discount_release(D0, 100.0)
    assert engine.exposure_release_budget == pytest.approx(100.0)
    # 累计第二次（同 λ）→ 200
    engine._record_budget_discount_release(D0, 100.0)
    assert engine.exposure_release_budget == pytest.approx(200.0)
    stats = engine.exposure_replenish_stats
    assert stats["budget_discount_release_amount"] == pytest.approx(200.0)


def test_zero_amount_ignored():
    engine = _engine(
        {"600000.SH": _position(1000, 10.0)}, 5000.0,
        {"600000.SH": (10.0, 10.0)}, HALF_TABLE, replenish=True, budget=True,
    )
    engine._record_budget_discount_release(D0, 0.0)
    assert engine.exposure_release_budget == 0.0


def test_requires_replenish_flag():
    """单开 budget_discount_replenish（无 replenish）必须报错。"""
    engine = _StubEngine({}, 10000.0, {})
    with pytest.raises(ValueError, match="replenish"):
        engine.set_exposure_table(
            HALF_TABLE, verbose=False, replenish=False, budget_discount_replenish=True
        )


# ---------------------------------------------------------------- 回补链路
def test_replenish_buys_back_after_lambda_recovers():
    """λ 折半建仓记账 → λ 恢复 → 回补 T+1 买回（全链路复用 P2-4）。"""
    # D0 λ=0.5：建仓成交 1000 元 → 折扣 1000（未打折应买 2000）
    engine = _engine({}, 1000.0, {"600000.SH": (10.0, 10.0)}, HALF_TABLE,
                     replenish=True, budget=True)
    engine._record_budget_discount_release(D0, 1000.0)
    assert engine.exposure_release_budget == pytest.approx(1000.0)

    # D0 判定：λ=0.5（未恢复）不回补
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_replenishes == {}

    # D1 λ=1.0：恢复 + 释放额 1000 > 容差 → 排队回补；但无持仓时无可加仓对象（跳过）
    engine._queue_exposure_replenish(D1, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_replenishes == {}


def test_replenish_buys_back_with_position():
    """有持仓时 λ 恢复 → 回补按比例加仓（上界 = min(现金, 释放额, λ×总值−市值)）。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 1200.0, {"600000.SH": (10.0, 10.0)}, HALF_TABLE,
                     replenish=True, budget=True)
    # 建仓折扣 1000（λ=0.5 信号日成交 1000）
    engine._record_budget_discount_release(D0, 1000.0)
    # D1 λ=1：总值 = 1000×10 + 1200 = 11200；room = 1.0×11200 − 10000 = 1200；
    # 回补额 = min(现金 1200, 释放额 1000, room 1200) = 1000
    engine._queue_exposure_replenish(D1, TRADING_DATES, DATE_TO_IDX)
    queued = engine.pending_exposure_replenishes.get("600000.SH")
    assert queued is not None
    assert float(queued["amount"]) == pytest.approx(1000.0)

    # T+1 执行：加仓 1000 元（10 元/股 → 100 股）
    engine._execute_pending_exposure_replenishes(D1, TRADING_DATES, DATE_TO_IDX)
    assert engine.positions["600000.SH"]["shares"] == 1100
    # 释放额被消耗（回补成交后扣减；整手取整允许 <1 手金额的尾差）
    assert engine.exposure_release_budget == pytest.approx(0.0, abs=20.0)
    # 加仓不重置买入日
    assert engine.positions["600000.SH"]["buy_date"] == D0


def test_budget_cleared_on_new_signal_date():
    """新调仓计划生成日释放额清零（既有 P2-4 语义对建仓折扣同样生效）。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 1200.0, {"600000.SH": (10.0, 10.0)}, HALF_TABLE,
                     replenish=True, budget=True)
    engine._record_budget_discount_release(D0, 1000.0)
    # D1 为新计划生成日（pending_signals 非空）→ 判定日清零
    engine.pending_signals[D1] = {"signals": {}}
    engine._queue_exposure_replenish(D1, TRADING_DATES, DATE_TO_IDX)
    assert engine.exposure_release_budget == 0.0
    assert engine.exposure_replenish_stats["budget_reset_on_rebalance"] == 1
