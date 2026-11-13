# -*- coding: utf-8 -*-
"""P7 T6 逐日序列装配：nav / equity / contribution / drawdown / turnover / fees / credit_state。

唯一入口
========
- ``nav``（物理 NAV）= ``calculate_nav(Σ_i total_value_i, PC)``（``v2/core/accounting/nav``）；
- ``equity``（袖级 credit 调整后权益）= ``cash + MV − credit``（T5 ``SleeveBalance.equity``，
  不在本模块重算）；
- ``contribution`` = ``equity / PC``（**保持有符号**，可加，Σ = nav − 1）；
- ``drawdown`` = ``nav / 累计峰值 − 1``（描述量；**MaxDD 数值仍只取**
  ``calculate_chain_metrics.max_drawdown``，并在 G4 互证）；
- ``turnover`` = 市场腿成交名义 / PC（内部抵消是袖间过户、不计成交额、费用恒 0）；
- ``fees`` = 逐日物理费用（由交易行的 ``fee`` 汇总；与 ``FeeBook.physical_fee_total``
  在 G4 逐位核对）；
- ``credit_state`` = 借入/出借/授予/归还/死局/迟回收/执行期现金保留的机械量。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from scripts.v2_p7.t5.host import T5HostResult
from scripts.v2_p7.t6.params import T6Params
from src.lazybull.v2.core.accounting.nav import calculate_nav

#: 市场腿类型（计成交额与费用）；内部抵消不计成交额
MARKET_KINDS: Tuple[str, ...] = ("market_buy", "market_sell")

#: 逐日帧 schema
DAILY_SCHEMA = "p7_t6_daily/v1"
SLEEVE_SCHEMA = "p7_t6_sleeve_daily/v1"


class FrameError(RuntimeError):
    """逐日帧装配的前置条件不满足（缺字段/日期不对齐），一律 fail-fast。"""


def nav_frame(result: T5HostResult, params: T6Params) -> pd.DataFrame:
    """物理 NAV 帧（``trade_date`` / ``nav`` / ``total_value`` / ``split_index``）。

    ``nav`` 经 ``calculate_nav``：分母为物理初资，分子为**两袖总值之和**
    （``Σ credit_i = 0`` 与 I1/I2 ⇒ 等于物理总值；T5 证据已逐日核验）。
    """
    rows = per_day_rows(result, params)
    frame = pd.DataFrame(
        [
            {
                "trade_date": row["trade_date"],
                "nav": row["nav"],
                "total_value": row["total_value"],
                "split_index": int(row["split_index"]),
            }
            for row in rows
        ]
    )
    return frame


def per_day_rows(result: T5HostResult, params: T6Params) -> List[Dict[str, Any]]:
    """逐日序列（两臂各一份；字段见模块 docstring）。"""
    series = result.nav_series
    if not series or any(not series.get(arm) for arm in params.arms):
        raise FrameError(f"NAV 序列缺失：{sorted(series)}")
    dates = [str(item["trade_date"]) for item in series[params.arms[0]]]
    for arm in params.arms:
        arm_dates = [str(item["trade_date"]) for item in series[arm]]
        if arm_dates != dates:
            raise FrameError(f"袖 {arm} 的 NAV 日期序列与其他袖不一致")
    balances = _by_day_sleeve(result.balance_rows)
    fees = _fee_by_day(result.trades)
    turnover = _turnover_by_day(result.trades)
    day_rows = {str(row["trade_date"]): row for row in result.daily_rows}
    credit_rows = {str(row["trade_date"]): row for row in result.credit_daily}
    out: List[Dict[str, Any]] = []
    peak = None
    fee_cum = 0.0
    for pos, day in enumerate(dates):
        total_value = float(sum(float(series[arm][pos]["total_value"]) for arm in params.arms))
        nav = float(calculate_nav(total_value, float(params.physical_capital)))
        peak = nav if peak is None else max(peak, nav)
        drawdown = float(nav / peak - 1.0) if peak else 0.0
        day_fee = float(fees.get(day, 0.0))
        fee_cum += day_fee
        out.append(
            _day_row(
                day=day,
                split_index=int(series[params.arms[0]][pos]["split_index"]),
                total_value=total_value,
                nav=nav,
                drawdown=drawdown,
                sleeve=balances.get(day) or {},
                arms=params.arms,
                physical_capital=float(params.physical_capital),
                credit_day=credit_rows.get(day) or {},
                mech=day_rows.get(day) or {},
                day_fee=day_fee,
                fee_cum=fee_cum,
                turnover=turnover.get(day) or {},
            )
        )
    return out


def _day_row(
    *,
    day: str,
    split_index: int,
    total_value: float,
    nav: float,
    drawdown: float,
    sleeve: Mapping[str, Mapping[str, Any]],
    arms: Sequence[str],
    physical_capital: float,
    credit_day: Mapping[str, Any],
    mech: Mapping[str, Any],
    day_fee: float,
    fee_cum: float,
    turnover: Mapping[str, Any],
) -> Dict[str, Any]:
    """单日序列行（把装配细节与循环拆开，便于逐字段核对）。"""
    if sorted(sleeve) != sorted(arms):
        raise FrameError(f"{day} 逐袖资产负债行缺失：{sorted(sleeve)}")
    equity = {sid: float(sleeve[sid]["equity"]) for sid in arms}
    credit = {sid: float(sleeve[sid]["credit"]) for sid in arms}
    cash = {sid: float(sleeve[sid]["cash"]) for sid in arms}
    gate_value = (credit_day.get("nonnegative_gate") or {}).get("min")
    market_notional = float(turnover.get("market") or 0.0)
    return {
        "trade_date": day,
        "split_index": int(split_index),
        "nav": nav,
        "total_value": total_value,
        "drawdown": drawdown,
        "sleeve_equity": equity,
        "sleeve_credit": credit,
        "sleeve_cash": cash,
        "contribution": {sid: float(equity[sid] / physical_capital) for sid in arms},
        "cash_sum": float(sum(cash.values())),
        "physical_cash": float(mech.get("physical_cash") or 0.0),
        "credit_sum": float(sum(credit.values())),
        "borrow_outstanding": float(sum(max(0.0, value) for value in credit.values())),
        "lent_outstanding": float(sum(max(0.0, -value) for value in credit.values())),
        "market_notional": market_notional,
        "offset_notional": float(turnover.get("offset") or 0.0),
        "turnover": market_notional / physical_capital,
        "fee_day": day_fee,
        "fee_cum": fee_cum,
        "trades": int(len(turnover.get("rows") or ())),
        "granted_total": float(
            sum(float(item["amount"]) for item in credit_day.get("grants") or ())
        ),
        "returned_total": float(
            sum(float(item["amount"]) for item in credit_day.get("repayments") or ())
        ),
        "deadlocked_total": float(
            sum(float(item["outstanding"]) for item in credit_day.get("deadlocks") or ())
        ),
        "late_recovery_total": float(
            sum(float(item["amount"]) for item in credit_day.get("late_recoveries") or ())
        ),
        "recall_due": int(len(credit_day.get("recall_queue") or ())),
        "right_censored": int(len(credit_day.get("right_censored") or ())),
        "credit_by_sleeve": dict(credit_day.get("credit_by_sleeve") or {}),
        "hold_by_sleeve": dict(credit_day.get("hold_by_sleeve") or {}),
        "due_reserve_by_sleeve": dict(credit_day.get("due_reserve_by_sleeve") or {}),
        "gate_min": float(gate_value) if gate_value is not None else None,
        "market_filled": dict(mech.get("market_filled") or {}),
        "scenarios": dict(mech.get("scenarios") or {}),
        "identity_ok": bool(mech.get("identity_ok", True)),
    }


def per_sleeve_rows(result: T5HostResult, params: T6Params) -> List[Dict[str, Any]]:
    """逐袖逐日资产负债（``equity = cash + MV − credit``；费用与成交额逐袖归集）。"""
    fees = _fee_by_sleeve_day(result.trades)
    turnover = _turnover_by_sleeve_day(result.trades)
    out: List[Dict[str, Any]] = []
    for row in result.balance_rows:
        day = str(row["trade_date"])
        sleeve = str(row["sleeve"])
        value = float(row["equity"])
        out.append(
            {
                "trade_date": day,
                "sleeve": sleeve,
                "cash": float(row["cash"]),
                "credit": float(row["credit"]),
                "market_value": float(row["market_value"]),
                "equity": value,
                "contribution": value / float(params.physical_capital),
                "cash_plus_credit": float(row["cash"]) + float(row["credit"]),
                "fee_day": float(fees.get((day, sleeve), 0.0)),
                "market_notional": float(turnover.get((day, sleeve), 0.0)),
            }
        )
    return out


def trade_rows(result: T5HostResult) -> List[Dict[str, Any]]:
    """交易行（保留 T4 原始字段；另加 ``market_notional`` 便于复核成交额）。"""
    rows: List[Dict[str, Any]] = []
    for trade in result.trades:
        row = dict(trade)
        kind = str(row.get("kind") or "")
        shares = int(row.get("shares") or 0)
        price = float(row.get("price") or 0.0)
        row["market_notional"] = float(shares * price) if kind in MARKET_KINDS else 0.0
        rows.append(row)
    return rows


def fee_total_from_trades(result: T5HostResult) -> float:
    """交易行费用合计（与 ``FeeBook.physical_fee_total`` 对账用）。"""
    return float(sum(float(trade.get("fee") or 0.0) for trade in result.trades))


def _by_day_sleeve(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        out.setdefault(str(row["trade_date"]), {})[str(row["sleeve"])] = row
    return out


def _fee_by_day(trades: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for trade in trades:
        day = str(trade["trade_date"])
        out[day] = out.get(day, 0.0) + float(trade.get("fee") or 0.0)
    return out


def _fee_by_sleeve_day(trades: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for trade in trades:
        for sleeve, amount in _sleeve_fee(trade).items():
            key = (str(trade["trade_date"]), sleeve)
            out[key] = out.get(key, 0.0) + float(amount)
    return out


def _turnover_by_day(trades: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for trade in trades:
        day = str(trade["trade_date"])
        item = out.setdefault(day, {"market": 0.0, "offset": 0.0, "rows": []})
        notional = float(int(trade.get("shares") or 0) * float(trade.get("price") or 0.0))
        if str(trade.get("kind") or "") in MARKET_KINDS:
            item["market"] += notional
        else:
            item["offset"] += notional
        item["rows"].append(trade)
    return out


def _turnover_by_sleeve_day(trades: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for trade in trades:
        if str(trade.get("kind") or "") not in MARKET_KINDS:
            continue
        day = str(trade["trade_date"])
        price = float(trade.get("price") or 0.0)
        for sleeve, shares in _sleeve_shares(trade).items():
            key = (day, sleeve)
            out[key] = out.get(key, 0.0) + float(shares * price)
    return out


def _sleeve_shares(trade: Mapping[str, Any]) -> Dict[str, int]:
    raw = trade.get("sleeves")
    if isinstance(raw, Mapping):
        return {str(key): int(value) for key, value in raw.items()}
    return {}


def _sleeve_fee(trade: Mapping[str, Any]) -> Dict[str, float]:
    for key in ("fee_split", "allocated_fees"):
        raw = trade.get(key)
        if isinstance(raw, Mapping):
            return {str(item): float(value) for item, value in raw.items()}
    return {}


def build_arm_frames(result: T5HostResult, params: T6Params) -> Dict[str, List[Dict[str, Any]]]:
    """单臂三帧（daily / sleeve / trades）。"""
    return {
        "daily": per_day_rows(result, params),
        "sleeve": per_sleeve_rows(result, params),
        "trades": trade_rows(result),
    }
