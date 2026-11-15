# -*- coding: utf-8 -*-
"""IDX-IN300 正式 T2/T3 的逐批面板与约束账户模拟器（零假设流与 A/A 校准共用）。

口径
====
- **候选池** = 每批主板新纳入且满足 P3 资格与 20 日成交额条件的股票（= P4 排名表
  里 ``selected`` ∪ ``not_selected_top6``）；零假设只在**组内**换标签，不跨批、不换日历；
- **执行** = 与 P4 实际账户同一经济规则：T+1 开盘买入（停牌/开盘涨停/缺价不成交，
  不重试）、整手 100 股、单槽预算 = 信号日 NAV/6、现金非负、同股抑制、``flow_date``
  收盘卖出（跌停/停牌顺延到下一可卖开盘价），费用同 P4 冻结费率；
- **持有期公司行为** = 现金分红在除权生效日计入现金（P3 §2.4 税后口径）、送转按
  ``floor(shares×(1+stk_div))`` 增股（成本总额不变）；
- 本模块只做机械模拟，不产生任何最终判定；判定在 ``formal``。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from scripts.v2_p7.p4.data import P4Data
from scripts.v2_p7.p4.engine import BatchPlan, _buy_fee, _sell_fee, max_lot_shares

#: 顺延卖出的最大观察交易日（超出记未平仓；这里只是面板边界，不是经济上限）
ROLLOVER_LIMIT = 60
#: 买入资格（P3 §2.2）：主板涨跌停按 10%
LIMIT_RATE = "0.10"


@dataclass(frozen=True)
class CorpEvent:
    """持有期公司行为（按面板内的日序表示）。"""

    day_index: int
    cash_per_share: float
    share_ratio: float


@dataclass
class BatchPanel:
    """单批候选池面板（候选股 × 窗口交易日的稠密数组）。"""

    notice_id: str
    signal_date: str
    entry_date: str
    flow_date: str
    codes: Tuple[str, ...]
    days: Tuple[str, ...]
    open_raw: np.ndarray
    close_raw: np.ndarray
    pre_close: np.ndarray
    suspended: np.ndarray
    limit_up: np.ndarray
    limit_down: np.ndarray
    has_row: np.ndarray
    is_st: np.ndarray
    entry_tradable: np.ndarray
    sellable_close: np.ndarray
    sell_open_ok: np.ndarray
    selected: Tuple[str, ...]
    bench_return: float
    corp: Tuple[Tuple[CorpEvent, ...], ...]

    @property
    def n_codes(self) -> int:
        return len(self.codes)

    @property
    def n_days(self) -> int:
        return len(self.days)


def _half_up(value: Decimal) -> Decimal:
    from decimal import ROUND_HALF_UP

    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _cash_basis(row: Any) -> float:
    for name, factor in (("cash_div", 1.0), ("cash_div_tax", 0.8)):
        value = getattr(row, name, None)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0.0:
            return float(number) * factor
    return 0.0


def build_panels(data: P4Data, plans: Mapping[str, BatchPlan]) -> Dict[str, BatchPanel]:
    """逐批构建候选池面板（只读；不读未来日）。"""
    panels: Dict[str, BatchPanel] = {}
    for notice_id in sorted(plans):
        plan = plans[notice_id]
        codes = _candidate_codes(plan)
        if not codes:
            continue
        window = _window_days(data, plan)
        arrays, flags = _collect_frames(data, codes, window)
        open_raw = arrays["open_raw"]
        close_raw = arrays["close_raw"]
        pre_close = arrays["pre_close"]
        suspended = flags["suspended"]
        limit_up = flags["limit_up"]
        limit_down = flags["limit_down"]
        has_row = flags["has_row"]
        is_st = flags["is_st"]
        panels[notice_id] = BatchPanel(
            notice_id=plan.notice_id,
            signal_date=plan.signal_date,
            entry_date=plan.entry_date,
            flow_date=plan.flow_date,
            codes=codes,
            days=window,
            open_raw=open_raw,
            close_raw=close_raw,
            pre_close=pre_close,
            suspended=suspended,
            limit_up=limit_up,
            limit_down=limit_down,
            has_row=has_row,
            is_st=is_st,
            entry_tradable=_entry_flags(open_raw, pre_close, has_row, suspended, limit_up, is_st),
            sellable_close=has_row & (~suspended) & (close_raw > 0.0) & (~limit_down),
            sell_open_ok=_sell_open_flags(
                open_raw, pre_close, has_row, suspended, limit_up, limit_down
            ),
            selected=tuple(str(item["ts_code"]) for item in plan.selected),
            bench_return=_bench_return(data, plan),
            corp=_corporate_by_code(data, codes, plan.entry_date, window),
        )
    return panels


def _candidate_codes(plan: BatchPlan) -> Tuple[str, ...]:
    """候选池 = 已选中 ∪ 通过资格但未进前 6（``not_selected_top6``）。"""
    tags = {
        str(row["ts_code"])
        for row in plan.selection_audit
        if row["status"] == "not_selected_top6"
    }
    return tuple(sorted({str(item["ts_code"]) for item in plan.selected} | tags))


def _collect_frames(
    data: P4Data, codes: Sequence[str], window: Sequence[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """按窗口逐日读取候选股行情（只读；缺行留 NaN/False）。"""
    n_codes, n_days = len(codes), len(window)
    arrays = {
        name: np.full((n_codes, n_days), np.nan)
        for name in ("open_raw", "close_raw", "pre_close")
    }
    flags = {
        name: np.zeros((n_codes, n_days), dtype=bool)
        for name in ("suspended", "limit_up", "limit_down", "has_row", "is_st")
    }
    position = {code: pos for pos, code in enumerate(codes)}
    for day_pos, day in enumerate(window):
        frame = data.read_codes(day, list(codes))
        if frame.empty:
            continue
        frame = frame.set_index("code")
        for code in codes:
            if code not in frame.index:
                continue
            row = frame.loc[code]
            pos = position[code]
            flags["has_row"][pos, day_pos] = True
            for name, column in (
                ("open_raw", "open"),
                ("close_raw", "close"),
                ("pre_close", "pre_close"),
            ):
                value = row.get(column)
                if value is not None and np.isfinite(float(value)):
                    arrays[name][pos, day_pos] = float(value)
            flags["suspended"][pos, day_pos] = bool(row.get("is_suspended", False))
            flags["limit_up"][pos, day_pos] = bool(row.get("is_limit_up", False))
            flags["limit_down"][pos, day_pos] = bool(row.get("is_limit_down", False))
            flags["is_st"][pos, day_pos] = bool(row.get("is_st", False))
    return arrays, flags


def _entry_flags(
    open_raw: np.ndarray,
    pre_close: np.ndarray,
    has_row: np.ndarray,
    suspended: np.ndarray,
    limit_up: np.ndarray,
    is_st: np.ndarray,
) -> np.ndarray:
    """入场可执行（T+1 开盘）：有行、未停牌、非 ST、开盘价有效且未一字涨停。"""
    n_codes = int(open_raw.shape[0])
    out = np.zeros(n_codes, dtype=bool)
    for pos in range(n_codes):
        if not has_row[pos, 0] or suspended[pos, 0] or is_st[pos, 0]:
            continue
        price = open_raw[pos, 0]
        pre = pre_close[pos, 0]
        if not np.isfinite(price) or price <= 0.0 or not np.isfinite(pre) or pre <= 0.0:
            continue
        if limit_up[pos, 0]:
            continue
        if price >= float(_half_up(Decimal(str(pre)) * (Decimal(1) + Decimal(LIMIT_RATE)))):
            continue
        out[pos] = True
    return out


def _sell_open_flags(
    open_raw: np.ndarray,
    pre_close: np.ndarray,
    has_row: np.ndarray,
    suspended: np.ndarray,
    limit_up: np.ndarray,
    limit_down: np.ndarray,
) -> np.ndarray:
    """顺延卖出的开盘可执行（未停牌、非一字跌停/涨停、开盘价有效）。"""
    n_codes, n_days = open_raw.shape
    out = has_row & (~suspended) & (open_raw > 0.0) & (pre_close > 0.0) & (~limit_up)
    for pos in range(n_codes):
        for day_pos in range(n_days):
            if not out[pos, day_pos]:
                continue
            rate = Decimal("0.05") if limit_down[pos, day_pos] else Decimal(LIMIT_RATE)
            floor = float(
                _half_up(Decimal(str(pre_close[pos, day_pos])) * (Decimal(1) - rate))
            )
            if open_raw[pos, day_pos] <= floor:
                out[pos, day_pos] = False
    return out


def _window_days(data: P4Data, plan: BatchPlan) -> Tuple[str, ...]:
    """窗口 = 入场日到 ``flow_date``，再加顺延观察日（到 CUTOFF 为止）。"""
    cutoff = str(getattr(data, "cutoff", data.dates[-1]))
    core = tuple(data.trade_days(plan.entry_date, min(plan.flow_date, cutoff)))
    tail = tuple(
        day for day in data.trade_days(plan.flow_date, cutoff) if day > plan.flow_date
    )[:ROLLOVER_LIMIT]
    return core + tail


def _corporate_by_code(
    data: P4Data,
    codes: Sequence[str],
    entry_date: str,
    window: Sequence[str],
) -> Tuple[Tuple[CorpEvent, ...], ...]:
    """窗口内公司行为（除权生效日 → 面板日序）。"""
    frame = data.dividends.loc[data.dividends["ts_code"].isin(set(codes))].copy()
    if frame.empty:
        return tuple(() for _ in codes)
    frame["effective"] = frame["ex_date"].map(
        lambda value: data.effective_trade_day(str(value)) or ""
    )
    frame = frame.loc[
        frame["effective"].gt(str(entry_date)) & frame["effective"].le(str(window[-1]))
    ]
    day_index = {str(day): pos for pos, day in enumerate(window)}
    position = {code: pos for pos, code in enumerate(codes)}
    out: List[List[CorpEvent]] = [[] for _ in codes]
    for row in frame.itertuples(index=False):
        pos = position.get(str(row.ts_code))
        day_pos = day_index.get(str(row.effective))
        if pos is None or day_pos is None:
            continue
        cash = _cash_basis(row)
        try:
            ratio = float(row.stk_div)
        except (TypeError, ValueError):
            ratio = 0.0
        if not np.isfinite(ratio) or ratio < 0.0:
            ratio = 0.0
        if cash <= 0.0 and ratio <= 0.0:
            continue
        out[pos].append(
            CorpEvent(day_index=day_pos, cash_per_share=max(cash, 0.0), share_ratio=ratio)
        )
    return tuple(tuple(sorted(items, key=lambda item: item.day_index)) for items in out)


def _bench_return(data: P4Data, plan: BatchPlan) -> float:
    """该批同域等权毛收益（P3 §3：T+1 ``open_adj`` → flow ``close_adj``，缺价保留现金）。"""
    members = list(plan.membership)
    if not members:
        return 0.0
    entry = data.read_codes(plan.entry_date, members).set_index("code")
    exit_frame = data.read_codes(plan.flow_date, members).set_index("code")
    total = 0.0
    for code in members:
        if code not in entry.index or code not in exit_frame.index:
            total += 1.0
            continue
        open_adj = float(entry.loc[code].get("open_adj", np.nan))
        close_adj = float(exit_frame.loc[code].get("close_adj", np.nan))
        if (
            not np.isfinite(open_adj)
            or open_adj <= 0.0
            or not np.isfinite(close_adj)
            or close_adj <= 0.0
        ):
            total += 1.0
            continue
        total += close_adj / open_adj
    return float(total / len(members) - 1.0)


class _Panels:
    """面板的索引缓存（避免逐日逐股线性查找）。"""

    def __init__(self, panels: Mapping[str, BatchPanel]) -> None:
        self.panels = dict(panels)
        self.code_pos = {
            notice: {code: pos for pos, code in enumerate(panel.codes)}
            for notice, panel in self.panels.items()
        }
        self.day_pos = {
            notice: {day: pos for pos, day in enumerate(panel.days)}
            for notice, panel in self.panels.items()
        }
        self.calendar = sorted({day for panel in self.panels.values() for day in panel.days})

    def price_mark(self, notice_id: str, code: str, day: str, last: Dict[str, float]) -> float:
        panel = self.panels[notice_id]
        pos = self.code_pos[notice_id][code]
        day_pos = self.day_pos[notice_id].get(day)
        if day_pos is None:
            return float(last.get(code, 0.0))
        close = float(panel.close_raw[pos, day_pos])
        if np.isfinite(close) and close > 0.0:
            last[code] = close
            return close
        fallback = float(last.get(code, 0.0))
        if fallback > 0.0:
            return fallback
        entry = float(panel.open_raw[pos, 0])
        if np.isfinite(entry) and entry > 0.0:
            last[code] = entry
            return entry
        return 0.0


def simulate(
    panels: Mapping[str, BatchPanel],
    selection: Mapping[str, Sequence[str]],
    *,
    principal: float,
    slots: int = 6,
) -> pd.DataFrame:
    """按批推进约束账户，返回逐日 ``(trade_date, nav, market_value)``。"""
    if not panels:
        raise RuntimeError("空面板")
    index = _Panels(panels)
    entries: Dict[str, List[str]] = {}
    for notice_id, panel in index.panels.items():
        entries.setdefault(panel.entry_date, []).append(notice_id)
    cash = float(principal)
    positions: Dict[str, Dict[str, Any]] = {}
    last_price: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    for day in index.calendar:
        # 信号日 NAV = 入场日开盘前（已完成前一日结算、未做本日卖出/买入）的账户值：
        # 与 P4 ``nav_signal[notice] = 信号日收盘 mark`` 等价（T+1 之间无交易）。
        nav_open = cash + _holdings_value(positions, index, day, last_price)
        cash += _corporate_step(day, positions, index, last_price)
        cash += _sell_step(day, positions, index)
        cash = _entry_step(
            day, entries, selection, index, positions, cash, nav_open, slots
        )
        market_value = _holdings_value(positions, index, day, last_price)
        rows.append(
            {
                "trade_date": day,
                "nav": float(cash + market_value),
                "market_value": float(market_value),
            }
        )
    return pd.DataFrame(rows)


def _holdings_value(
    positions: Mapping[str, Mapping[str, Any]],
    index: "_Panels",
    day: str,
    last_price: Dict[str, float],
) -> float:
    value = 0.0
    for code, item in positions.items():
        value += float(item["shares"]) * index.price_mark(
            str(item["notice_id"]), code, day, last_price
        )
    return float(value)


def _corporate_step(
    day: str,
    positions: Dict[str, Dict[str, Any]],
    index: "_Panels",
    last_price: Dict[str, float],
) -> float:
    """除权生效日：现金入账 + 送转增股 + 缺价 carry 下调（返回现金增量）。"""
    cash = 0.0
    for code in list(positions):
        item = positions[code]
        notice_id = str(item["notice_id"])
        panel = index.panels[notice_id]
        day_pos = index.day_pos[notice_id].get(day)
        if day_pos is None:
            continue
        pos = index.code_pos[notice_id][code]
        for event in panel.corp[pos]:
            if event.day_index != day_pos:
                continue
            shares = int(item["shares"])
            last = float(last_price.get(code, 0.0))
            if event.cash_per_share > 0.0:
                cash += float(shares) * float(event.cash_per_share)
            if event.share_ratio > 0.0:
                added = int(np.floor(float(shares) * float(event.share_ratio) + 1e-12))
                item["shares"] = int(shares) + added
            # 与 P4 一致：除权日同步下调缺价 carry（否则旧价 × 新股数会机械增值）
            if last > 0.0:
                adjusted = (last - float(event.cash_per_share)) / (
                    1.0 + float(event.share_ratio)
                )
                if adjusted > 0.0:
                    last_price[code] = float(adjusted)
    return float(cash)


def _sell_step(day: str, positions: Dict[str, Dict[str, Any]], index: "_Panels") -> float:
    """到期收盘卖出 / 顺延开盘卖出（返回现金增量）。"""
    cash = 0.0
    for code in list(positions):
        item = positions[code]
        notice_id = str(item["notice_id"])
        panel = index.panels[notice_id]
        day_pos = index.day_pos[notice_id].get(day)
        if day_pos is None:
            continue
        pos = index.code_pos[notice_id][code]
        price = None
        if day == panel.flow_date:
            if bool(panel.sellable_close[pos, day_pos]):
                price = float(panel.close_raw[pos, day_pos])
        elif day > panel.flow_date and bool(panel.sell_open_ok[pos, day_pos]):
            price = float(panel.open_raw[pos, day_pos])
        if price is None or not np.isfinite(price) or price <= 0.0:
            continue
        notional = Decimal(str(price)) * Decimal(int(item["shares"]))
        cash += float(notional) - float(_sell_fee(notional))
        positions.pop(code)
    return float(cash)


def _entry_step(
    day: str,
    entries: Mapping[str, Sequence[str]],
    selection: Mapping[str, Sequence[str]],
    index: "_Panels",
    positions: Dict[str, Dict[str, Any]],
    cash: float,
    nav_open: float,
    slots: int,
) -> float:
    """当日建仓（T+1 开盘、整手、单槽预算 = 信号日 NAV/6、现金非负）。"""
    for notice_id in sorted(entries.get(day, [])):
        panel = index.panels[notice_id]
        codes = list(selection.get(notice_id, ()))
        if not codes:
            continue
        target = Decimal(str(nav_open)) / Decimal(slots)
        for code in codes:
            if len(positions) >= slots:
                break
            if code in positions:
                continue
            pos = index.code_pos[notice_id].get(code, -1)
            if pos < 0 or not bool(panel.entry_tradable[pos]):
                continue
            price = float(panel.open_raw[pos, 0])
            if not np.isfinite(price) or price <= 0.0:
                continue
            budget = min(target, Decimal(str(cash)))
            shares = max_lot_shares(budget, Decimal(str(price)))
            if shares <= 0:
                continue
            notional = Decimal(str(price)) * Decimal(shares)
            cost = notional + _buy_fee(notional)
            if cost > Decimal(str(cash)):
                continue
            cash -= float(cost)
            positions[code] = {"notice_id": notice_id, "shares": int(shares), "entry": day}
    return float(cash)
