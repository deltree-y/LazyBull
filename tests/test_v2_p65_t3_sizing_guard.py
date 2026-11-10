# -*- coding: utf-8 -*-
"""P6.5 T3 加仓定仓等价守卫：`size_add_shares` 与 T3 前的内联实现逐值一致。

任务书 §3.2 要求"执行侧只加受控注入点或订单构造适配：默认生产路径（无注入）
行为逐位不变，且有守卫用例证明"。T3 把 `_add_to_position` 的内联定仓/缩量/费用
逻辑提取为 `core/decision/sizing.size_add_shares`（供预演复用），本文件以**冻结
参照实现**（T3 前的逐字逻辑）做同输入逐值对照，并用真实执行器验证落账结果一致。

全部为合成输入（合成费率、合成价格），不读真实配置/行情。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pytest

from src.lazybull.universe.base import BasicUniverse
from src.lazybull.v2.common.cost import CostModel
from src.lazybull.v2.core.accounting.ledger import BookingEntry, make_fill_key
from src.lazybull.v2.core.decision.portfolio_merger import SingleSleevePortfolioMerger
from src.lazybull.v2.core.decision.sizing import compute_lot_shares, size_add_shares
from src.lazybull.v2.core.signal.base import EqualWeightSignal
from src.lazybull.v2.hosts.native_engine import NativeMergeBacktestEngineML

_X = "600000.SH"
_D1 = pd.Timestamp("2026-01-06")


def _cost_model() -> CostModel:
    return CostModel(commission_rate=0.0003, min_commission=5.0, stamp_tax=0.0005, slippage=0.0)


def _reference_add_shares(
    *, target_value: float, trade_price: float, capital: float, cost_model: Any
) -> Optional[Tuple[int, float, float, float]]:
    """T3 前 `_add_to_position` 的内联实现（冻结参照；逐字复制，禁止改写）。"""
    shares = compute_lot_shares(target_value, trade_price)
    if shares == 0:
        return None

    amount = shares * trade_price
    cost = cost_model.calculate_buy_cost(amount)
    total_cost_cash = amount + cost
    if total_cost_cash > capital:
        if capital <= cost:
            return None
        shares = compute_lot_shares(capital - cost, trade_price)
        if shares == 0:
            return None
        amount = shares * trade_price
        cost = cost_model.calculate_buy_cost(amount)
        total_cost_cash = amount + cost
    return shares, amount, cost, total_cost_cash


_CASES = (
    # (target_value, trade_price, capital)
    (2000.0, 10.0, 100000.0),  # 正常整手
    (999.0, 10.0, 100000.0),  # 不足一手
    (2000.0, 10.0, 500.0),  # 缩量后仍不足一手
    (2000.0, 10.0, 1200.0),  # 缩量到 1 手
    (2000.0, 10.0, 5.0),  # 现金 <= 费用
    (2000.0, 10.0, 6.0),  # 现金略高于费用但仍不足一手
    (2000.0, 10.0, 2005.0),  # 恰好够全额（不进入缩量）
    (2000.0, 10.0, 2004.99),  # 差 0.01 元 ⇒ 缩量
    (2000.0, 10.5, 100000.0),  # 非整数手价格
    (0.0, 10.0, 100000.0),  # 零预算
)


def test_size_add_shares_matches_frozen_reference() -> None:
    cost_model = _cost_model()
    for target_value, price, capital in _CASES:
        reference = _reference_add_shares(
            target_value=target_value,
            trade_price=price,
            capital=capital,
            cost_model=cost_model,
        )
        actual = size_add_shares(
            target_value=target_value,
            trade_price=price,
            capital=capital,
            cost_model=cost_model,
        )
        assert actual == reference, (
            f"size_add_shares 与冻结参照不一致: target={target_value} price={price} "
            f"capital={capital} ⇒ {actual} != {reference}"
        )


def _engine(capital: float, positions: Dict[str, Dict[str, Any]]) -> NativeMergeBacktestEngineML:
    engine = NativeMergeBacktestEngineML(
        features_by_date={},
        universe=BasicUniverse(
            pd.DataFrame({"ts_code": [_X], "name": ["甲"], "industry": ["I1"]}),
            verbose=False,
        ),
        signal=EqualWeightSignal(top_n=5),
        initial_capital=capital,
        cost_model=_cost_model(),
        verbose=False,
        portfolio_merger=SingleSleevePortfolioMerger(),
    )
    engine.current_capital = capital
    engine.positions = {code: dict(info) for code, info in positions.items()}
    engine._emit_immediate_log = lambda *args, **kwargs: None
    engine._prepare_price_index(
        pd.DataFrame({"ts_code": [_X], "trade_date": [_D1], "close": [10.0]})
    )
    return engine


def _position(shares: int, price: float = 10.0) -> Dict[str, Any]:
    return {
        "shares": shares,
        "buy_date": pd.Timestamp("2025-12-31"),
        "buy_trade_price": price,
        "buy_pnl_price": price,
        "buy_cost_cash": shares * price,
    }


@pytest.mark.parametrize("capital", [1200.0, 3000.0, 100000.0])
def test_add_to_position_books_the_same_numbers_as_reference(capital: float) -> None:
    """真实执行器（`_add_to_position`）与冻结参照在同输入下落账结果一致。"""
    executed = _engine(capital, {_X: _position(100)})
    expected_engine = _engine(capital, {_X: _position(100)})

    executed._add_to_position(_D1, _X, 2000.0)

    sized = _reference_add_shares(
        target_value=2000.0,
        trade_price=10.0,
        capital=capital,
        cost_model=_cost_model(),
    )
    assert sized is not None
    shares, _amount, _cost, total_cost_cash = sized
    expected_engine.ledger.apply_fills(
        expected_engine,
        [
            BookingEntry(
                fill_key=make_fill_key(_D1, "buy", _X, 0),
                date=_D1,
                stock=_X,
                action="buy",
                cash_delta=-total_cost_cash,
                trade_record={},
                add_shares=int(shares),
                add_cost_cash=float(total_cost_cash),
            )
        ],
    )

    assert executed.current_capital == expected_engine.current_capital
    assert executed.positions[_X]["shares"] == expected_engine.positions[_X]["shares"]
    assert executed.positions[_X]["buy_cost_cash"] == expected_engine.positions[_X]["buy_cost_cash"]
