"""每日风控减仓（暴露门控主动减仓）单元测试。

硬契约：
1. **判定每日进行、执行走 T0/T1**：T0 生成卖单只入队列（当日不改持仓），T+1 开盘执行；
2. **按比例减仓不做个股选择**：全部持仓同一比例，且为部分卖出（保留剩余持仓）；
3. **开关关闭逐位一致**：未设置暴露系数表时不产生任何副作用；
4. 不可交易时不进延迟订单队列（避免被放大成整仓卖出），统计跳过并由次日重判。
"""

import inspect

import pandas as pd
import pytest

from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.backtest.exposure_override import BacktestExposureOverrideMixin
from src.lazybull.backtest.exposure_trim import (
    EXPOSURE_TRIM_TOLERANCE,
    TRIM_SELL_TYPE,
    TRIM_TRIGGER_TYPE,
    BacktestExposureTrimMixin,
)
from src.lazybull.backtest.sell_execution import BacktestSellExecutionMixin
from src.lazybull.common.cost import CostModel

D0 = pd.Timestamp("2024-01-02")
D1 = pd.Timestamp("2024-01-03")
TRADING_DATES = [D0, D1]
DATE_TO_IDX = {D0: 0, D1: 1}


def _position(
    shares: int,
    buy_trade_price: float,
    buy_cost_cash: float = None,
) -> dict:
    """构造持仓字典（字段与引擎一致）。"""
    return {
        "shares": shares,
        "buy_date": D0,
        "signal_date": D0,
        "buy_trade_price": buy_trade_price,
        "buy_pnl_price": buy_trade_price,
        "buy_cost_cash": (buy_cost_cash if buy_cost_cash is not None else shares * buy_trade_price),
    }


class _StubEngine(
    BacktestExposureTrimMixin,
    BacktestExposureOverrideMixin,
    BacktestSellExecutionMixin,
):
    """只承载减仓判定与部分卖出账务的最小桩（价格/成本可控）。"""

    def __init__(self, positions, cash, prices, blocked=()):
        self.positions = positions
        self.current_capital = cash
        self.prices = prices  # {stock: (open, close)}
        self.blocked = set(blocked)  # 模拟不可交易
        self.pending_condition_sells = {}
        self.pending_stop_loss_sells = {}
        self.pending_order_manager = None
        self.stop_loss_monitor = None
        self.verbose = False
        self.sell_timing = "open"
        self.trades = []
        self.cost_model = CostModel()
        self.exposure_table = None
        self.sell_calls = []
        self._init_exposure_trim_state()

    # ---------------- 价格桩 ----------------
    def _get_trade_price(self, date, stock):
        return self.prices.get(stock, (None, None))[1]

    def _get_trade_price_open(self, date, stock):
        return self.prices.get(stock, (None, None))[0]

    def _get_pnl_price(self, date, stock):
        return self._get_trade_price(date, stock)

    def _get_pnl_price_open(self, date, stock):
        return self._get_trade_price_open(date, stock)

    def _calculate_portfolio_value(self, date):
        total = self.current_capital
        for stock, info in self.positions.items():
            total += info["shares"] * self._get_trade_price(date, stock)
        return total

    # ---------------- 卖出入口桩（模拟可交易检查结果） ----------------
    def _sell_stock(self, date, stock, **kwargs):
        self.sell_calls.append((stock, kwargs))
        if stock in self.blocked:
            return
        direct_kwargs = {k: v for k, v in kwargs.items() if k != "allow_pending"}
        self._sell_stock_direct(date, stock, **direct_kwargs)


def _engine(positions, cash, prices, blocked=()):
    engine = _StubEngine(positions, cash, prices, blocked=blocked)
    engine.set_exposure_table({"20240102": 0.5, "20240103": 0.5}, verbose=False)
    return engine


class _RealSellEngine(_StubEngine):
    """不覆写 `_sell_stock` 的桩：走真实卖出链（`enable_pending_order=False`）。"""

    def __init__(self, positions, cash, prices):
        super().__init__(positions, cash, prices)
        self.enable_pending_order = False
        self.price_data_cache = None

    def _sell_stock(self, date, stock, **kwargs):  # noqa: D401 - 恢复真实实现
        BacktestSellExecutionMixin._sell_stock(self, date, stock, **kwargs)


# --------------------------------------------------------------------- 契约
@pytest.mark.parametrize(
    "method,expected",
    [
        ("_sell_stock", {"fraction", "allow_pending"}),
        ("_sell_stock_with_status_check", {"fraction", "allow_pending"}),
        ("_sell_stock_direct", {"fraction"}),
    ],
)
def test_sell_chain_signatures_accept_trim_kwargs(method, expected):
    """契约：减仓依赖的关键字必须在整条卖出链上都被接受（防签名漂移）。"""
    params = set(inspect.signature(getattr(BacktestSellExecutionMixin, method)).parameters)
    missing = expected - params
    assert not missing, f"{method} 缺少参数 {sorted(missing)}"


def test_real_sell_chain_executes_trim():
    """走真实 `_sell_stock` 链：执行减仓不得因关键字不匹配而报错。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _RealSellEngine(positions, 0.0, {"600000.SH": (10.0, 10.0)})
    engine.set_exposure_table({"20240102": 0.5, "20240103": 0.5}, verbose=False)
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    engine._execute_pending_exposure_trims(D1, TRADING_DATES, DATE_TO_IDX)
    assert engine.positions["600000.SH"]["shares"] == 500
    assert engine.exposure_trim_stats["trim_orders_sold"] == 1
    assert engine.trades[0]["sell_type"] == TRIM_SELL_TYPE


def test_mixin_is_mounted_on_engine():
    """契约：每日减仓由专用 mixin 提供，且引擎已挂载。"""
    assert BacktestExposureTrimMixin in BacktestEngine.__mro__


def test_tolerance_is_single_default():
    """契约：容差只有一个默认值（禁止多层回退）。"""
    engine = _StubEngine({}, 0.0, {})
    assert engine.exposure_trim_tolerance == EXPOSURE_TRIM_TOLERANCE == 0.03


def test_disabled_has_no_side_effects():
    """表为 None（开关关闭）时判定与执行都必须无副作用。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _StubEngine(positions, 0.0, {"600000.SH": (10.0, 10.0)})
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_trims == {}
    assert engine.exposure_trim_stats["trigger_days"] == 0
    assert engine.positions["600000.SH"]["shares"] == 1000
    engine._execute_pending_exposure_trims(D1, TRADING_DATES, DATE_TO_IDX)
    assert engine.sell_calls == []
    assert engine.positions["600000.SH"]["shares"] == 1000


# --------------------------------------------------------------------- 判定
def test_tolerance_blocks_small_excess():
    """超配未达容差不得减仓（避免反复微调）。"""
    positions = {"600000.SH": _position(10_000, 100.0)}
    engine = _engine(positions, 0.0, {"600000.SH": (100.0, 100.0)})
    engine.set_exposure_table({"20240102": 0.98}, verbose=False)
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_trims == {}
    assert engine.exposure_trim_stats["skipped_days_below_tolerance"] == 1
    assert engine.exposure_trim_stats["trigger_days"] == 0


def test_trigger_queues_every_position_without_selling():
    """T0 判定只入队列：全部持仓同一比例、当日不改持仓。"""
    positions = {
        "600000.SH": _position(1000, 10.0),
        "000001.SZ": _position(2000, 20.0),
    }
    prices = {"600000.SH": (10.0, 10.0), "000001.SZ": (20.0, 20.0)}
    engine = _engine(positions, 0.0, prices)
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)

    assert set(engine.pending_exposure_trims) == set(positions)
    fracs = {info["fraction"] for info in engine.pending_exposure_trims.values()}
    assert len(fracs) == 1 and fracs.pop() == pytest.approx(0.5)
    # T0 不改持仓、不产生成交
    assert engine.positions["600000.SH"]["shares"] == 1000
    assert engine.trades == []
    assert engine.exposure_trim_stats["trim_orders"] == 2


def test_recovery_day_does_not_trim():
    """暴露系数回到 1.0 时不再减仓（恢复靠常规调仓买回）。"""
    positions = {"600000.SH": _position(10_000, 100.0)}
    engine = _StubEngine(positions, 0.0, {"600000.SH": (100.0, 100.0)})
    engine.set_exposure_table({"20240102": 1.0}, verbose=False)
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.pending_exposure_trims == {}
    assert engine.exposure_trim_stats["trigger_days"] == 0


def test_skips_stock_already_queued_for_full_exit():
    """已在整仓卖出队列（到期/止损）的股票不进减仓队列，整仓卖出优先。"""
    positions = {"600000.SH": _position(1000, 10.0), "000001.SZ": _position(2000, 20.0)}
    prices = {"600000.SH": (10.0, 10.0), "000001.SZ": (20.0, 20.0)}
    engine = _engine(positions, 0.0, prices)
    engine.pending_condition_sells["600000.SH"] = {"signal_date": D0}
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    assert set(engine.pending_exposure_trims) == {"000001.SZ"}


def test_no_queue_when_already_pending_trim():
    """同一股票在队列中未执行前不得重复排队。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 0.0, {"600000.SH": (10.0, 10.0)})
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    engine.exposure_trim_stats["trim_orders"] = 0
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    assert engine.exposure_trim_stats["trim_orders"] == 0
    assert len(engine.pending_exposure_trims) == 1


# --------------------------------------------------------------------- 执行
def test_execute_trim_reduces_exposure_and_clears_queue():
    """T+1 执行：按比例部分卖出、队列清空、暴露向目标收敛。"""
    positions = {
        "600000.SH": _position(1000, 10.0),
        "000001.SZ": _position(2000, 20.0),
    }
    originals = {stock: info["shares"] for stock, info in positions.items()}
    prices = {"600000.SH": (10.0, 10.0), "000001.SZ": (20.0, 20.0)}
    engine = _engine(positions, 0.0, prices)
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    engine._execute_pending_exposure_trims(D1, TRADING_DATES, DATE_TO_IDX)

    assert engine.pending_exposure_trims == {}
    # 两笔均卖出，且是部分卖出（剩余持仓保留）
    assert engine.exposure_trim_stats["trim_orders_sold"] == 2
    assert engine.exposure_trim_stats["trim_orders_skipped"] == 0
    assert len(engine.trades) == 2
    for stock in originals:
        assert stock in engine.positions
        assert engine.positions[stock]["shares"] < originals[stock]
    # 暴露落到目标附近（容差内）
    portfolio_value = engine._calculate_portfolio_value(D1)
    position_value = portfolio_value - engine.current_capital
    assert position_value <= 0.5 * portfolio_value * (1 + EXPOSURE_TRIM_TOLERANCE)


def test_not_tradeable_skips_without_pending_order():
    """不可交易：不进延迟订单队列、只统计跳过（次日每日判定重试）。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 0.0, {"600000.SH": (10.0, 10.0)}, blocked={"600000.SH"})
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    engine._execute_pending_exposure_trims(D1, TRADING_DATES, DATE_TO_IDX)

    assert engine.pending_order_manager is None
    assert engine.exposure_trim_stats["trim_orders_sold"] == 0
    assert engine.exposure_trim_stats["trim_orders_skipped"] == 1
    assert engine.positions["600000.SH"]["shares"] == 1000
    assert engine.pending_exposure_trims == {}


def test_sell_call_carries_fraction_and_disables_pending():
    """执行时必须传 fraction 且关闭延迟队列（否则会被放大成整仓卖出）。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 0.0, {"600000.SH": (10.0, 10.0)}, blocked={"600000.SH"})
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    engine._execute_pending_exposure_trims(D1, TRADING_DATES, DATE_TO_IDX)

    assert len(engine.sell_calls) == 1
    _, kwargs = engine.sell_calls[0]
    assert kwargs["fraction"] == pytest.approx(0.5)
    assert kwargs["allow_pending"] is False
    assert kwargs["sell_type"] == TRIM_SELL_TYPE
    assert kwargs["trigger_type"] == TRIM_TRIGGER_TYPE


# --------------------------------------------------------------------- 账务
def test_partial_sell_accounting():
    """部分卖出：剩余持仓保留、每股成本不变、总现金支出按比例缩减。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _StubEngine(positions, 0.0, {"600000.SH": (12.0, 12.0)})
    engine._sell_stock_direct(
        D1,
        "600000.SH",
        sell_type=TRIM_SELL_TYPE,
        sell_reason="暴露门控减仓",
        trigger_type=TRIM_TRIGGER_TYPE,
        fraction=0.5,
    )

    assert engine.positions["600000.SH"]["shares"] == 500
    assert engine.positions["600000.SH"]["buy_trade_price"] == 10.0
    assert engine.positions["600000.SH"]["buy_cost_cash"] == pytest.approx(5000.0)
    assert engine.trades[0]["shares"] == 500
    assert engine.trades[0]["sell_type"] == TRIM_SELL_TYPE
    assert engine.trades[0]["sell_reason"] == "暴露门控减仓"
    assert engine.trades[0]["trigger_type"] == TRIM_TRIGGER_TYPE
    expected_proceeds = 500 * 12.0 - engine.trades[0]["cost"]
    assert engine.current_capital == pytest.approx(expected_proceeds)
    # 绩效收益按卖出部分计算（500 股 × (12-10) − 手续费）
    assert engine.trades[0]["pnl_profit_amount"] == pytest.approx(1000.0, abs=20.0)


def test_whole_sell_path_unchanged():
    """fraction=1.0（默认）时行为与改动前一致：持仓被删除。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _StubEngine(positions, 0.0, {"600000.SH": (12.0, 12.0)})
    engine._sell_stock_direct(D1, "600000.SH")
    assert "600000.SH" not in engine.positions
    assert engine.trades[0]["shares"] == 1000
    assert engine.trades[0]["sell_type"] == "holding_period"


@pytest.mark.parametrize(
    "total,fraction,expected",
    [
        (1000, 1.0, 1000),  # 整仓卖出
        (1000, 0.5, 500),  # 整手取整
        (300, 0.5, 100),  # 150 股取整到 100 股（整手）
        (105, 0.99, 105),  # 剩余不足一手 → 整仓卖出
        (105, 0.5, 0),  # 取整后不足 100 股 → 不执行
        (50, 0.5, 50),  # 小额持仓无法保留零股 → 整仓卖出
    ],
)
def test_resolve_trim_shares(total, fraction, expected):
    """股数取整：整手卖出，剩余不足一手则整仓卖出。"""
    engine = _StubEngine({}, 0.0, {})
    assert engine._resolve_trim_shares(total, fraction) == expected


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.2])
def test_invalid_fraction_rejected(bad):
    """非法比例必须报错（禁止凭空放大卖出量）。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _StubEngine(positions, 0.0, {"600000.SH": (10.0, 10.0)})
    with pytest.raises(ValueError, match="fraction 必须落于"):
        engine._sell_stock_direct(D1, "600000.SH", fraction=bad)


def test_report_contains_trim_stats():
    """报告必须暴露减仓统计与容差，便于事后归因。"""
    positions = {"600000.SH": _position(1000, 10.0)}
    engine = _engine(positions, 0.0, {"600000.SH": (10.0, 10.0)})
    engine._queue_exposure_trim(D0, TRADING_DATES, DATE_TO_IDX)
    report = engine.get_exposure_trim_report()
    assert report["tolerance"] == EXPOSURE_TRIM_TOLERANCE
    assert report["pending_orders"] == 1
    assert report["trim_orders"] == 1
