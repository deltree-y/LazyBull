"""回测持仓快照旁路测试（合成数据，不依赖真实配置与真实数据）。

覆盖：
1. 快照行内容（权重/持有天数/剩余持有交易日/到期执行日）与标签口径一致；
2. 开关关闭时零记录（默认关闭，不影响既有链路）；
3. 旁路**只读**：不写回持仓字典任何字段；
4. 引擎持有快照 mixin 且默认关闭（契约：逐日持仓快照必须由本 mixin 提供）。
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.backtest.engine import BacktestEngine
from src.lazybull.backtest.holdings_snapshot import BacktestHoldingsSnapshotMixin
from src.lazybull.common.sidecar_schema import SNAPSHOT_KEYS


class _StubEngine(BacktestHoldingsSnapshotMixin):
    """最小桩：只提供快照所需的价格取数与持仓状态。"""

    def __init__(self, prices, holding_period=5):
        self.holding_period = holding_period
        self.positions = {}
        self.holdings_snapshots = []
        self.record_holdings_snapshot = False
        self._prices = prices

    def _get_trade_price(self, date, stock):
        return self._prices.get((date, stock))


DATES = [pd.Timestamp(f"2024-01-{day:02d}") for day in range(2, 12)]


def _engine():
    prices = {}
    for date in DATES:
        for stock in ("A", "B"):
            prices[(date, stock)] = 10.0
    engine = _StubEngine(prices, holding_period=5)
    engine.positions = {
        "A": {
            "shares": 1000,
            "buy_date": DATES[1],  # 第 2 个交易日买入
            "signal_date": DATES[0],
            "buy_trade_price": 10.0,
        },
        "B": {
            "shares": 2000,
            "buy_date": DATES[3],
            "signal_date": DATES[2],
            "buy_trade_price": 10.0,
        },
    }
    return engine


def test_snapshot_rows_match_holding_arithmetic():
    engine = _engine()
    engine.record_holdings_snapshot = True
    date_to_idx = {date: idx for idx, date in enumerate(DATES)}
    engine._record_holdings_snapshot(
        DATES[5], portfolio_value=50_000.0, trading_dates=DATES, date_to_idx=date_to_idx
    )

    snapshot = engine.get_holdings_snapshot()
    assert list(snapshot.columns) == SNAPSHOT_KEYS
    assert len(snapshot) == 2
    row_a = snapshot[snapshot["ts_code"] == "A"].iloc[0]
    # A 在第 2 个交易日买入、当前为第 6 个交易日 → 持有 4 个交易日
    assert row_a["holding_days"] == 4
    # 到期执行日 = 买入位置 1 + 持有期 5 = 位置 6；h = 6 - 5 - 1 = 0
    assert row_a["planned_exit_date"] == DATES[6]
    assert row_a["remaining_intervals"] == 0
    assert row_a["market_value"] == pytest.approx(10_000.0)
    assert row_a["weight"] == pytest.approx(0.2)

    row_b = snapshot[snapshot["ts_code"] == "B"].iloc[0]
    # B 在位置 3 买入 → 到期执行位置 8；h = 8 - 5 - 1 = 2
    assert row_b["holding_days"] == 2
    assert row_b["planned_exit_date"] == DATES[8]
    assert row_b["remaining_intervals"] == 2


def test_snapshot_skipped_when_disabled_and_when_flat():
    engine = _engine()
    date_to_idx = {date: idx for idx, date in enumerate(DATES)}
    engine._record_holdings_snapshot(DATES[5], 50_000.0, DATES, date_to_idx)
    assert engine.get_holdings_snapshot().empty  # 默认关闭 → 零记录

    engine.record_holdings_snapshot = True
    engine.positions = {}
    engine._record_holdings_snapshot(DATES[5], 50_000.0, DATES, date_to_idx)
    assert engine.get_holdings_snapshot().empty  # 空仓不记录


def test_snapshot_is_read_only_for_positions():
    engine = _engine()
    engine.record_holdings_snapshot = True
    before = {code: dict(info) for code, info in engine.positions.items()}
    engine._record_holdings_snapshot(
        DATES[5], 50_000.0, DATES, {date: idx for idx, date in enumerate(DATES)}
    )
    assert engine.positions == before  # 旁路不得写回任何持仓字段


def test_snapshot_missing_calendar_position_raises():
    engine = _engine()
    engine.record_holdings_snapshot = True
    with pytest.raises(ValueError, match="交易日序列中的位置"):
        engine._record_holdings_snapshot(pd.Timestamp("2024-03-01"), 50_000.0, DATES, {})


def test_snapshot_out_of_window_defers_exit_fields():
    """到期执行日超出回测窗口 → 记空值，不猜测。"""
    engine = _engine()
    engine.record_holdings_snapshot = True
    engine.positions = {
        "A": {"shares": 1000, "buy_date": DATES[-1], "buy_trade_price": 10.0},
    }
    date_to_idx = {date: idx for idx, date in enumerate(DATES)}
    engine._record_holdings_snapshot(DATES[-1], 50_000.0, DATES, date_to_idx)
    row = engine.get_holdings_snapshot().iloc[0]
    assert pd.isna(row["planned_exit_date"])
    assert row["remaining_intervals"] is None
    # 持有天数仍可算（买入当日起算）
    assert row["holding_days"] == 0


def test_engine_exposes_snapshot_contract():
    """引擎必须内置快照 mixin，且默认关闭（不改变既有链路行为）。"""
    assert issubclass(BacktestEngine, BacktestHoldingsSnapshotMixin)
    assert hasattr(BacktestEngine, "get_holdings_snapshot")
