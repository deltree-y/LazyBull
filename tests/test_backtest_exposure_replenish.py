"""对称回补（暴露门控 P2-4）单元测试。

硬契约：
1. **默认关闭逐位一致**：`exposure_replenish_enabled=False`（或表为 None）时判定与执行都无副作用；
2. **只在两次调仓之间生效**：`pending_signals` / `unfilled_slots` 非空时释放额归零、不干预；
3. **上界三选一**：回补额 = min(可用现金, 未回补释放额, λ×组合总值 − 持仓市值)，**不把 λ 当目标仓位**；
4. **按比例对全部持仓加仓**，不做个股选择；跳过正在排队卖出的股票；
5. 加仓**不重置买入日**、`buy_cost_cash` 累加；不足一手自然跳过；
6. 不可买入（停牌/涨停/无行情）⇒ 跳过且**额度退回**，次日重判。
"""

import inspect

import pandas as pd
import pytest

from src.lazybull.backtest.buy_execution import BacktestBuyExecutionMixin
from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.backtest.exposure_override import BacktestExposureOverrideMixin
from src.lazybull.backtest.exposure_replenish import (
    REPLENISH_BUY_TYPE,
    BacktestExposureReplenishMixin,
)
from src.lazybull.backtest.exposure_trim import BacktestExposureTrimMixin
from src.lazybull.backtest.sell_execution import BacktestSellExecutionMixin
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
    """只承载回补判定/执行与加仓账务的最小桩。"""

    def __init__(self, positions, cash, prices, blocked_buy=()):
        self.positions = positions
        self.current_capital = cash
        self.prices = prices  # {stock: (open, close)}
        self.blocked_buy = set(blocked_buy)
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
        """单股持仓市值（与引擎同口径：股数 × 当日收盘价）。"""
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
        if stock in self.blocked_buy:
            return False, "涨停"
        return True, ""


def _engine(positions, cash, prices, blocked_buy=(), replenish=True, table=None):
    engine = _StubEngine(positions, cash, prices, blocked_buy=blocked_buy)
    engine.set_exposure_table(
        {"20240102": 1.0, "20240103": 1.0} if table is None else table,
        verbose=False,
        replenish=replenish,
    )
    return engine


class _ChainEngine(BacktestSellExecutionMixin, _StubEngine):
    """走真实卖出链的桩：验证「减仓 → 释放额 → 回补」整链打通。"""

    def __init__(self, positions, cash, prices):
        super().__init__(positions, cash, prices)
        self.enable_pending_order = False
        self.price_data_cache = None

    def _sell_stock(self, date, stock, **kwargs):  # noqa: D401 - 使用真实卖出实现
        BacktestSellExecutionMixin._sell_stock(self, date, stock, **kwargs)


# --------------------------------------------------------------------- 契约
def test_mixin_is_mounted_on_engine():
    assert BacktestExposureReplenishMixin in BacktestEngine.__mro__


def test_add_to_position_signature():
    """契约：对称的部分买入入口必须存在且关键字稳定（防签名漂移）。"""
    params = set(inspect.signature(BacktestBuyExecutionMixin._add_to_position).parameters)
    assert {"date", "stock", "target_value", "buy_type", "buy_reason"} <= params


def test_disabled_has_no_side_effects():
    """开关关闭（replenish=False）时判定与执行都无副作用。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 5000.0, {"600000.SH": (10.0, 10.0)}, replenish=False)
    engine.exposure_release_budget = 3000.0
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_replenishes == {}
    engine._execute_pending_exposure_replenishes(D1, TRADING_DATES, DATE_TO_IDX)
    assert engine.positions["600000.SH"]["shares"] == 1000
    assert engine.current_capital == 5000.0


# --------------------------------------------------------------------- 判定
def test_no_budget_no_action():
    engine = _engine({"600000.SH": _position(1000, 10.0)}, 0.0, {"600000.SH": (10.0, 10.0)})
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_replenishes == {}
    assert engine.exposure_replenish_stats["skipped_days_no_budget"] == 1


def test_rebalance_in_flight_resets_budget():
    """新调仓计划生成日：不排队、释放额归零（由新计划重新分配那笔现金）。"""
    engine = _engine({"600000.SH": _position(1000, 10.0)}, 2000.0, {"600000.SH": (10.0, 10.0)})
    engine.exposure_release_budget = 1000.0
    engine.pending_signals[D0] = {"signals": {}}
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_replenishes == {}
    assert engine.exposure_release_budget == 0.0
    assert engine.exposure_replenish_stats["budget_reset_on_rebalance"] == 1


def test_unfilled_slots_do_not_block_replenish():
    """补齐未完成（unfilled_slots 持续存在）不得阻止回补——只在计划生成日清零。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 2000.0, {"600000.SH": (10.0, 10.0)})
    engine.unfilled_slots[D0] = {"unfilled_count": 1, "unfilled_slot_weights": []}
    engine.exposure_release_budget = 1000.0
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_replenishes, "补齐在途不应阻断回补"
    assert engine.exposure_replenish_stats["budget_reset_on_rebalance"] == 0


def test_cap_still_tight_uses_room_as_upper_bound():
    """λ 仍紧（room<=0）时不得回补（回补只在“上限放开”后发生）。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 0.0, {"600000.SH": (10.0, 10.0)}, table={"20240102": 0.5})
    engine.exposure_release_budget = 3000.0
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_replenishes == {}


def test_proportional_topup_is_no_selection():
    """按市值比例回补：两只持仓各补 1000 元（比例一致），且金额不超过释放额。"""
    positions = {
        "600000.SH": _position(1000, 10.0),  # 市值 10000
        "000001.SZ": _position(1000, 10.0),  # 市值 10000
    }
    engine = _engine(positions, 2000.0, {"600000.SH": (10.0, 10.0), "000001.SZ": (10.0, 10.0)})
    engine.exposure_release_budget = 2000.0
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    amounts = {s: i["amount"] for s, i in engine.pending_exposure_replenishes.items()}
    assert amounts == pytest.approx({"600000.SH": 1000.0, "000001.SZ": 1000.0})
    assert engine.exposure_release_budget == pytest.approx(0.0)


def test_sell_queue_stocks_are_skipped():
    """正在排队卖出的股票不得回补（不买回即将卖出的标的）。"""
    positions = {
        "600000.SH": _position(1000, 10.0),
        "000001.SZ": _position(1000, 10.0),
    }
    engine = _engine(positions, 2000.0, {"600000.SH": (10.0, 10.0), "000001.SZ": (10.0, 10.0)})
    engine.exposure_release_budget = 2000.0
    engine.pending_condition_sells["600000.SH"] = {"trigger_date": D0}
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    assert set(engine.pending_exposure_replenishes) == {"000001.SZ"}


def test_tolerance_blocks_small_topup():
    """差额未达容差（3%）不得回补。"""
    positions = {"600000.SH": _position(10_000, 100.0)}  # 市值 100 万
    engine = _engine(positions, 1000.0, {"600000.SH": (100.0, 100.0)})
    engine.exposure_release_budget = 1000.0  # 1000 < 3% × ~100.1 万 = 30030
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_replenishes == {}
    assert engine.exposure_replenish_stats["skipped_days_below_tolerance"] == 1


# --------------------------------------------------------------------- 执行
def test_execution_buys_proportionally_and_keeps_buy_date():
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 2000.0, {"600000.SH": (10.0, 10.0)})
    engine.exposure_release_budget = 1000.0  # 释放额 1000 ⇒ 回补 1 手（100 股）
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    engine._execute_pending_exposure_replenishes(D1, TRADING_DATES, DATE_TO_IDX)
    assert engine.positions["600000.SH"]["shares"] == 1100  # 加仓 100 股（1 手）
    assert engine.positions["600000.SH"]["buy_date"] == D0  # 买入日不重置
    assert engine.trades[0]["buy_type"] == REPLENISH_BUY_TYPE
    assert engine.exposure_replenish_stats["replenish_orders_filled"] == 1
    assert engine.current_capital < 2000.0


def test_blocked_buy_returns_budget_for_retry():
    """不可买入（涨停/停牌）：跳过且额度**全额退回**，次日重判。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 2000.0, {"600000.SH": (10.0, 10.0)}, blocked_buy={"600000.SH"})
    engine.exposure_release_budget = 1000.0
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.exposure_release_budget == pytest.approx(0.0)  # 已按计划额扣减
    engine._execute_pending_exposure_replenishes(D1, TRADING_DATES, DATE_TO_IDX)
    assert engine.positions["600000.SH"]["shares"] == 1000
    assert engine.exposure_replenish_stats["replenish_orders_skipped"] == 1
    assert engine.exposure_release_budget == pytest.approx(1000.0)  # 额度退回


def test_insufficient_cash_partially_fills_and_returns_remainder():
    """现金不足：按可用现金缩量成交，未花掉的额度退回释放额。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 1200.0, {"600000.SH": (10.0, 10.0)})
    engine.exposure_release_budget = 5000.0
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    planned = engine.pending_exposure_replenishes["600000.SH"]["amount"]
    engine._execute_pending_exposure_replenishes(D1, TRADING_DATES, DATE_TO_IDX)
    invested = engine.exposure_replenish_stats["replenish_bought_amount"]
    assert invested > 0  # 至少成交 1 手
    assert engine.exposure_replenish_stats["replenish_orders_filled"] == 1
    # 未花掉的额度已退回（释放额 = 原额度 − 计划额 + 未花掉部分）
    assert engine.exposure_release_budget == pytest.approx(5000.0 - invested, abs=1e-6)
    assert planned >= invested


def test_report_contains_budget_and_flag():
    engine = _engine({"600000.SH": _position(1000, 10.0)}, 0.0, {"600000.SH": (10.0, 10.0)})
    report = engine.get_exposure_replenish_report()
    assert report["enabled"] is True
    assert report["release_budget"] == 0.0
    assert "pending_orders" in report


# ----------------------------------------------------------- 整链：减仓 → 释放额 → 回补
def test_trim_then_replenish_chain():
    """减仓释放的现金必须先计入释放额，再在上限放开后按比例回补。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _ChainEngine(positions, 0.0, {"600000.SH": (10.0, 10.0)})
    engine.set_exposure_table({"20240102": 0.5, "20240103": 1.0}, verbose=False, replenish=True)
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    engine._execute_pending_exposure_trims(D1, TRADING_DATES, DATE_TO_IDX)
    assert engine.exposure_release_budget > 0.0  # 减仓释放额被累计
    sold = 1000 - positions["600000.SH"]["shares"]
    assert sold > 0
    # 次日上限放开：回补排队（额度 = 释放额）
    engine._queue_exposure_replenish(D1, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_replenishes, "上限放开后必须产生回补计划"
    planned = sum(i["amount"] for i in engine.pending_exposure_replenishes.values())
    assert 0 < planned <= engine.exposure_release_budget + planned + 1e-6


def test_flat_replenish_absent_when_no_trim():
    """全 1 表（无减仓）⇒ 释放额为 0 ⇒ 永不产生回补（对照臂逐位一致的前提）。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 5000.0, {"600000.SH": (10.0, 10.0)})
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_trims == {}
    assert engine.exposure_release_budget == 0.0
    engine._queue_exposure_replenish(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_replenishes == {}
