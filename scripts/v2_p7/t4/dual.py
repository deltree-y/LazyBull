# -*- coding: utf-8 -*-
"""P7 T4 双口径恒等式（I1/I2/I3 + 只读反事实 NAV）。

契约依据（``docs/contracts/netting_freeze.md`` §4 / F26 §P7 注记）
==============================================================
- **I1 现金**：``Σ cash_i == 物理现金``（≤``1e-6`` 元）；借用态另需
  ``Σ credit_i == 0``（本金逐笔对消，精确 0）；
- **I2 持仓**：逐 ``(ts_code, 物理来源 lot_id)`` 整数股**精确相等**（0 差）；
- **I3 NAV**：``Σ calculate_nav(total_value_i, 物理初资) == calculate_nav(物理总值, 物理初资)``
  （≤``1e-6`` 无量纲），并以总值和辅证；**允许 E 为零/负的有符号贡献**，
  只要求各分量与物理总值有限、物理初资 > 0；
- **反事实口径**：``E_i^nosave = total_value_i - 累计节省_i``，分母为**袖自身初资**；
  ``E_i ≤ 0`` 或分母非正 => 该袖独立指标标 ``invalid_nonpositive_equity``（**保留当日**，
  不删日、不跳过该袖）。

纪律
====
- NAV 只经 ``src.lazybull.v2.core.accounting.nav.calculate_nav``；
- 同一交易日内逐股只解析一次估值价（物理与虚拟**共享**），逐股登记来源；
- 恒等式破缺**报错停止**，不降级为告警（``assert_identities``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from scripts.v2_p7.t4.price import PriceBook
from scripts.v2_p7.t4.state import PhysicalBook, SleeveState

#: I1 货币绝对容差（元；F25/F26）
CASH_TOLERANCE = 1e-6
#: I3 直接 NAV 与总值辅证容差（无量纲 / 元；F25/F26）
NAV_TOLERANCE = 1e-6


class DualCaliberError(RuntimeError):
    """恒等式破缺或估值不可用（拒绝语义，不降级为告警）。"""


@dataclass(frozen=True)
class IdentityRow:
    """单条恒等式的证据行。"""

    identity: str
    left: float
    right: float
    diff: float
    tolerance: float
    unit: str


@dataclass(frozen=True)
class DailyLedgerReport:
    """单日双口径报告（I1/I2/I3 + 有符号贡献 + 反事实 NAV）。"""

    trade_date: str
    identities: Tuple[IdentityRow, ...]
    total_value: Mapping[str, float]
    contribution_nav: Mapping[str, float]
    independent_equity: Mapping[str, float]
    independent_nav: Mapping[str, Optional[float]]
    invalid_sleeves: Mapping[str, str]
    physical_total_value: float
    physical_nav: float

    @property
    def ok(self) -> bool:
        return all(abs(row.diff) <= row.tolerance for row in self.identities)


def run_daily_identities(
    *,
    trade_date: str,
    sleeves: Mapping[str, SleeveState],
    physical: PhysicalBook,
    prices: PriceBook,
    physical_initial: float,
    sleeve_capital: Mapping[str, float],
    cumulative_savings: Mapping[str, float],
) -> DailyLedgerReport:
    """逐日 I1/I2/I3（实际含节省账）与只读反事实 NAV（不含节省）。"""
    if not math.isfinite(float(physical_initial)) or float(physical_initial) <= 0.0:
        raise DualCaliberError(f"物理初资必须为正有限数：{physical_initial!r}")
    total_value, physical_value = _daily_valuations(sleeves, physical, prices)
    rows: List[IdentityRow] = [
        _cash_row(sleeves, physical),
        _credit_row(sleeves),
        _holdings_row(sleeves, physical, trade_date),
    ]
    contribution, phys_nav = _nav_rows(rows, total_value, physical_value, physical_initial)
    independent_equity, independent_nav, invalid = _independent_caliber(
        total_value, sleeve_capital, cumulative_savings
    )
    return DailyLedgerReport(
        trade_date=str(trade_date),
        identities=tuple(rows),
        total_value=dict(total_value),
        contribution_nav=contribution,
        independent_equity=independent_equity,
        independent_nav=independent_nav,
        invalid_sleeves=invalid,
        physical_total_value=physical_value,
        physical_nav=phys_nav,
    )


def _daily_valuations(
    sleeves: Mapping[str, SleeveState], physical: PhysicalBook, prices: PriceBook
) -> Tuple[Dict[str, float], float]:
    """逐袖与物理侧的当日货币总值（同一交易日逐股只解析一次估值价）。"""
    cache: Dict[str, float] = {}
    total_value: Dict[str, float] = {}
    for sleeve_id in sorted(sleeves):
        state = sleeves[sleeve_id]
        value = float(state.cash)
        for lot in state.lots:
            value += float(lot.shares) * _price_of(prices, lot.ts_code, lot.acquired_price, cache)
        if not math.isfinite(value):
            raise DualCaliberError(f"袖 {sleeve_id} 货币总值非有限：{value!r}")
        total_value[sleeve_id] = value
    physical_value = float(physical.cash)
    for lot in physical.lots.values():
        physical_value += float(lot.shares) * _price_of(prices, lot.ts_code, lot.buy_price, cache)
    if not math.isfinite(physical_value):
        raise DualCaliberError(f"物理总值非有限：{physical_value!r}")
    return total_value, physical_value


def _cash_row(sleeves: Mapping[str, SleeveState], physical: PhysicalBook) -> IdentityRow:
    """I1：``Σ 虚拟现金 == 物理现金``（≤1e-6 元）。"""
    cash_sum = 0.0
    for sleeve_id in sorted(sleeves):
        cash_sum += float(sleeves[sleeve_id].cash)
    return IdentityRow(
        "I1", cash_sum, float(physical.cash), cash_sum - float(physical.cash), CASH_TOLERANCE, "元"
    )


def _credit_row(sleeves: Mapping[str, SleeveState]) -> IdentityRow:
    """I1b：``Σ credit_i == 0``（借入/借出本金逐笔对消；精确）。"""
    credit_sum = 0.0
    for sleeve_id in sorted(sleeves):
        credit_sum += float(sleeves[sleeve_id].credit)
    return IdentityRow("I1b", credit_sum, 0.0, credit_sum, 0.0, "元")


def _holdings_row(
    sleeves: Mapping[str, SleeveState], physical: PhysicalBook, trade_date: str
) -> IdentityRow:
    """I2：逐 ``(ts_code, 物理来源 lot_id)`` 整数股精确相等（0 差）。"""
    virtual_shares: Dict[Tuple[str, str], int] = {}
    for sleeve_id in sorted(sleeves):
        for lot in sleeves[sleeve_id].lots:
            key = (lot.ts_code, lot.source_lot_id)
            virtual_shares[key] = virtual_shares.get(key, 0) + lot.shares
    physical_shares = {
        (lot.ts_code, lot.source_lot_id): lot.shares for lot in physical.lots.values()
    }
    missing = sorted(set(physical_shares) - set(virtual_shares))
    extra = sorted(set(virtual_shares) - set(physical_shares))
    if missing or extra:
        raise DualCaliberError(
            f"{trade_date} I2 来源键集不一致：物理有而虚拟缺={missing[:5]}，"
            f"虚拟有而物理无={extra[:5]}"
        )
    share_diff = 0
    for key in set(virtual_shares) | set(physical_shares):
        share_diff += abs(virtual_shares.get(key, 0) - physical_shares.get(key, 0))
    return IdentityRow(
        "I2",
        float(sum(virtual_shares.values())),
        float(sum(physical_shares.values())),
        float(share_diff),
        0.0,
        "股",
    )


def _nav_rows(
    rows: List[IdentityRow],
    total_value: Mapping[str, float],
    physical_value: float,
    physical_initial: float,
) -> Tuple[Dict[str, float], float]:
    """I3（贡献 NAV 和）+ I3b（总值和辅证）；NAV 只经 ``calculate_nav``。"""
    from src.lazybull.v2.core.accounting.nav import calculate_nav

    contribution = {
        sid: float(calculate_nav(total_value[sid], physical_initial)) for sid in total_value
    }
    phys_nav = float(calculate_nav(physical_value, physical_initial))
    nav_sum = 0.0
    for sleeve_id in sorted(contribution):
        nav_sum += contribution[sleeve_id]
    rows.append(IdentityRow("I3", nav_sum, phys_nav, nav_sum - phys_nav, NAV_TOLERANCE, "无量纲"))
    value_sum = 0.0
    for sleeve_id in sorted(total_value):
        value_sum += total_value[sleeve_id]
    rows.append(
        IdentityRow(
            "I3b", value_sum, physical_value, value_sum - physical_value, NAV_TOLERANCE, "元"
        )
    )
    return contribution, phys_nav


def _independent_caliber(
    total_value: Mapping[str, float],
    sleeve_capital: Mapping[str, float],
    cumulative_savings: Mapping[str, float],
) -> Tuple[Dict[str, float], Dict[str, Optional[float]], Dict[str, str]]:
    """只读反事实（不含节省）：``E_i = total_value_i - 累计节省_i``，分母为袖自身初资。"""
    from src.lazybull.v2.core.accounting.nav import calculate_nav

    independent_equity: Dict[str, float] = {}
    independent_nav: Dict[str, Optional[float]] = {}
    invalid: Dict[str, str] = {}
    for sleeve_id in sorted(total_value):
        equity = total_value[sleeve_id] - float(cumulative_savings.get(sleeve_id, 0.0))
        independent_equity[sleeve_id] = equity
        capital = float(sleeve_capital.get(sleeve_id, 0.0))
        if (
            not math.isfinite(equity)
            or equity <= 0.0
            or not math.isfinite(capital)
            or capital <= 0.0
        ):
            invalid[sleeve_id] = "invalid_nonpositive_equity"
            independent_nav[sleeve_id] = None
            continue
        independent_nav[sleeve_id] = float(calculate_nav(equity, capital))
    return independent_equity, independent_nav, invalid


def _price_of(prices: PriceBook, ts_code: str, buy_price: float, cache: Dict[str, float]) -> float:
    """同一交易日内逐股只解析一次估值价（物理与虚拟共享），并登记来源。"""
    cached = cache.get(str(ts_code))
    if cached is not None:
        return cached
    price, price_date, source = prices.valuation(ts_code, buy_price)
    if price <= 0.0 or not math.isfinite(float(price)):
        raise DualCaliberError(
            f"估值价三级解析全不可用：{ts_code}（buy_price={buy_price!r}，source={source}）"
        )
    prices.note_valuation(ts_code, price, source, price_date)
    cache[str(ts_code)] = float(price)
    return float(price)


def assert_identities(report: DailyLedgerReport) -> None:
    """恒等式破缺即报错停止（不是告警；F25 §4「验收」）。"""
    broken = [row for row in report.identities if abs(row.diff) > row.tolerance]
    if broken:
        detail = "；".join(
            f"[{row.identity}] 左={row.left!r} 右={row.right!r} 差={row.diff!r} "
            f"容差={row.tolerance} 单位={row.unit}"
            for row in broken
        )
        raise DualCaliberError(
            f"{report.trade_date} 恒等式破缺（全部编号={[r.identity for r in broken]}）：{detail}"
        )
