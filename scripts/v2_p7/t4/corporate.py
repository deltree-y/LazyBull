# -*- coding: utf-8 -*-
"""P7 T4/T5 公司行为入账（现金分红 / 送转 / 两者同时发生 / 停牌跨事件 / 复牌）。

契约依据
========
- ``docs/contracts/netting_freeze.md`` §3「公司行为拆分」：物理账户收到现金分红 ⇒ 按
  **除息日（权益登记口径）**登记各袖应收，**到账日兑付**；送股 / 拆股以**除权日持仓**为准
  （新增股数并入原 lot，不重置买入日）；
- ``docs/experiments/p7_idx_in300_p3_prereg_20261114.md`` §2.4：只处理
  ``div_proc="实施"``；``new_shares = floor(old_shares × (1+stk_div))``、成本总额不变；
  非交易日 ``ex_date`` 在首个 ``>= ex_date`` 的交易日开盘前处理；现金优先 ``cash_div``
  （税后），缺失回退 ``cash_div_tax × 0.8`` 并登记。

实现纪律
========
1. **登记日口径**：现金按除权生效日**开盘前**持仓确认（= 上一交易日收盘持仓），
   与「权益登记日」等价；若两者不等（非交易日映射），逐条登记 ``record_date_gap``。
2. **应收不是现金**：除息日进 ``SleeveState.receivable`` / ``PhysicalBook.receivable``，
   到账日一次性转现金；I1 仍是纯现金恒等式，另加 I1r 应收恒等式。
3. **物理与虚拟同步**：送转股数先按**物理来源 lot** 计算（``floor``），再按虚拟批次
   持有比例分配，余数按 ``(股数降序, virtual_lot_id)`` 逐股分配；两侧价格同除
   ``(1+ratio)``，成本总额守恒。
4. **缺价 carry 可比**：除权生效日**开盘前**由 ``PriceBook.apply_ex_adjustment`` 把
   「最后已知收盘」按理论除权价 ``(last − cash) / (1 + ratio)`` 调整，防止停牌期间
   「旧价 × 新股数」的机械增值（R03 同源问题在 T4/T5 的对应修正）。
5. 本模块只读原始分红分区与宿主状态，不写盘、不产生任何经济判定。
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.v2_p7.t4.price import PriceBook
from scripts.v2_p7.t4.state import (
    P7Lot,
    PhysicalBook,
    PhysicalLot,
    ReceivableEntry,
    SleeveState,
    StateError,
)

#: 分红实施状态（TuShare ``div_proc``）
DIV_PROC_IMPLEMENTED = "实施"
#: 现金回退口径：``cash_div`` 缺失时按 ``cash_div_tax × 0.8``（税前 → 保守 20% 税）
CASH_TAX_FACTOR = 0.8
#: 默认分红分区目录（相对仓库根）
DEFAULT_DIVIDEND_DIR = "data/raw/dividend"


class CorporateError(RuntimeError):
    """公司行为数据不可解析或与持仓状态冲突（fail-fast，不静默跳过）。"""


@dataclass(frozen=True)
class DividendEvent:
    """单条已实施的分红 / 送转事件（冻结字段 + 生效交易日映射）。"""

    ts_code: str
    end_date: str
    ex_date: str
    pay_date: str
    record_date: str
    cash_per_share: float
    share_ratio: float
    cash_basis: str
    #: 除权生效交易日（首个 ``>= ex_date`` 的交易日）
    ex_trade_day: str
    #: 支付生效交易日（首个 ``>= pay_date`` 的交易日；缺失回退 ``ex_trade_day``）
    pay_trade_day: str

    @property
    def event_id(self) -> str:
        return f"DIV-{self.ts_code}-{self.end_date}-{self.ex_date}"

    def payload(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ts_code": self.ts_code,
            "end_date": self.end_date,
            "ex_date": self.ex_date,
            "ex_trade_day": self.ex_trade_day,
            "pay_date": self.pay_date,
            "pay_trade_day": self.pay_trade_day,
            "record_date": self.record_date,
            "cash_per_share": float(self.cash_per_share),
            "share_ratio": float(self.share_ratio),
            "cash_basis": self.cash_basis,
        }


def _date8(value: Any) -> str:
    text = str(value if value is not None else "")
    if text in ("nan", "NaT", "None"):
        return ""
    return text.replace("-", "")[:8]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class CorporateActionBook:
    """分红 / 送转事件簿（按运行日历映射到生效交易日；只读）。"""

    def __init__(
        self,
        *,
        calendar: Sequence[str],
        dividend_dir: Path,
        cutoff: Optional[str] = None,
        tax_factor: float = CASH_TAX_FACTOR,
    ) -> None:
        days = tuple(str(day) for day in calendar)
        if not days:
            raise CorporateError("公司行为事件簿需要非空交易日历")
        self._days = days
        self._cutoff = str(cutoff) if cutoff else days[-1]
        self._tax_factor = float(tax_factor)
        self._by_ex: Dict[str, List[DividendEvent]] = {}
        self._by_pay: Dict[str, List[DividendEvent]] = {}
        self._events: List[DividendEvent] = []
        self._stats: Dict[str, Any] = {
            "rows_read": 0,
            "rows_implemented": 0,
            "events_in_window": 0,
            "events_before_window": 0,
            "events_after_cutoff": 0,
            "cash_basis_fallback": 0,
            "unparsable_ex_date": 0,
            "zero_effect_events": 0,
        }
        self._load(Path(dividend_dir))
        self._record_date_gaps: List[Dict[str, Any]] = []
        self._snapshot_counts: Dict[str, int] = {}

    # ────────────────────────── 装载 ──────────────────────────

    def _load(self, directory: Path) -> None:
        import pandas as pd

        if not directory.exists():
            raise CorporateError(f"分红分区目录缺件：{directory}")
        paths = sorted(directory.glob("*.parquet"))
        if not paths:
            raise CorporateError(f"分红分区为空：{directory}")
        columns = [
            "ts_code",
            "end_date",
            "ann_date",
            "div_proc",
            "stk_div",
            "cash_div",
            "cash_div_tax",
            "record_date",
            "ex_date",
            "pay_date",
            "imp_ann_date",
        ]
        frames = [pd.read_parquet(path, columns=columns) for path in paths]
        frame = pd.concat(frames, ignore_index=True)
        self._stats["rows_read"] = int(len(frame))
        frame = frame.loc[frame["div_proc"].astype(str).eq(DIV_PROC_IMPLEMENTED)].copy()
        self._stats["rows_implemented"] = int(len(frame))
        if frame.empty:
            return
        frame["ts_code"] = frame["ts_code"].astype(str)
        date_columns = (
            "end_date",
            "ann_date",
            "record_date",
            "ex_date",
            "pay_date",
            "imp_ann_date",
        )
        for column in date_columns:
            frame[column] = frame[column].map(_date8)
        frame["ex_date"] = frame["ex_date"].map(lambda value: value if len(value) == 8 else "")
        bad = int((frame["ex_date"] == "").sum())
        self._stats["unparsable_ex_date"] = bad
        frame = frame.loc[frame["ex_date"] != ""].copy()
        # 同一 (ts_code, ex_date) 多行：按 (imp_ann_date, ann_date, end_date) 取最后一行
        frame["sort_key"] = (
            frame["imp_ann_date"].fillna("")
            + "|"
            + frame["ann_date"].fillna("")
            + "|"
            + frame["end_date"].fillna("")
        )
        frame = frame.sort_values(["ts_code", "ex_date", "sort_key"], kind="mergesort")
        frame = frame.groupby(["ts_code", "ex_date"], as_index=False).tail(1)
        for row in frame.itertuples(index=False):
            event = self._build_event(row)
            if event is None:
                continue
            self._events.append(event)
            self._by_ex.setdefault(event.ex_trade_day, []).append(event)
            self._by_pay.setdefault(event.pay_trade_day, []).append(event)
        for key in self._by_ex:
            self._by_ex[key].sort(key=lambda item: (item.ts_code, item.ex_date))
        for key in self._by_pay:
            self._by_pay[key].sort(key=lambda item: (item.ts_code, item.ex_date))
        self._stats["events_in_window"] = len(self._events)

    def _build_event(self, row: Any) -> Optional[DividendEvent]:
        ex_date = str(row.ex_date)
        ex_day = self._effective_day(ex_date, allow_before=True)
        if ex_day is None:
            if ex_date < self._days[0]:
                self._stats["events_before_window"] += 1
            else:
                self._stats["events_after_cutoff"] += 1
            return None
        pay_date = _date8(getattr(row, "pay_date", ""))
        pay_day = self._effective_day(pay_date, allow_before=False) if pay_date else None
        if pay_day is None or pay_day < ex_day:
            pay_day = ex_day
        cash, basis = self._cash(row)
        ratio = float(row.stk_div) if _finite(row.stk_div) and float(row.stk_div) > 0 else 0.0
        if basis == "cash_div_tax_x0.8":
            self._stats["cash_basis_fallback"] += 1
        if cash <= 0.0 and ratio <= 0.0:
            self._stats["zero_effect_events"] += 1
            return None
        return DividendEvent(
            ts_code=str(row.ts_code),
            end_date=str(row.end_date),
            ex_date=ex_date,
            pay_date=pay_date,
            record_date=_date8(getattr(row, "record_date", "")),
            cash_per_share=float(cash),
            share_ratio=float(ratio),
            cash_basis=basis,
            ex_trade_day=str(ex_day),
            pay_trade_day=str(pay_day),
        )

    def _cash(self, row: Any) -> Tuple[float, str]:
        """现金口径：``cash_div``（税后）优先；缺失回退 ``cash_div_tax × 0.8``。"""
        if _finite(row.cash_div) and float(row.cash_div) > 0.0:
            return float(row.cash_div), "cash_div_after_tax"
        if _finite(row.cash_div_tax) and float(row.cash_div_tax) > 0.0:
            return float(row.cash_div_tax) * self._tax_factor, "cash_div_tax_x0.8"
        return 0.0, "none"

    def _effective_day(self, day: str, *, allow_before: bool) -> Optional[str]:
        """首个 ``>= day`` 的交易日（窗口外一律 ``None``，不把历史事件搬到首日）。

        ``allow_before`` 仅用于表达「窗口前事件」的登记语义：两种取值都返回 ``None``，
        差异只体现在统计口径（``events_before_window``），不影响入账。
        """
        text = str(day)
        if not text:
            return None
        if text > self._cutoff or text < self._days[0]:
            return None
        index = bisect_left(self._days, text)
        if index >= len(self._days):
            return None
        del allow_before
        return self._days[index]

    # ────────────────────────── 查询 ──────────────────────────

    def last_day(self) -> str:
        return self._days[-1]

    def ex_events(self, day: str) -> Tuple[DividendEvent, ...]:
        return tuple(self._by_ex.get(str(day), ()))

    def pay_events(self, day: str) -> Tuple[DividendEvent, ...]:
        return tuple(self._by_pay.get(str(day), ()))

    def has_events(self, day: str) -> bool:
        return str(day) in self._by_ex or str(day) in self._by_pay

    def events(self) -> Tuple[DividendEvent, ...]:
        return tuple(self._events)

    # ────────────────────────── 除息日 ──────────────────────────

    def apply_ex_day(
        self,
        *,
        day: str,
        sleeves: Dict[str, SleeveState],
        physical: PhysicalBook,
        prices: PriceBook,
    ) -> List[Dict[str, Any]]:
        """除权生效日开盘前：价格可比调整 + 应收登记 + 送转并入原 lot。

        Returns:
            逐事件逐袖证券变动行（证据用）；``[]`` 表示当日无事件或无可处置持仓。
        """
        events = self.ex_events(day)
        if not events:
            return []
        rows: List[Dict[str, Any]] = []
        for event in events:
            prices.apply_ex_adjustment(
                event.ts_code,
                cash_per_share=event.cash_per_share,
                share_ratio=event.share_ratio,
                ex_date=event.ex_trade_day,
            )
            if event.record_date and event.record_date >= event.ex_trade_day:
                self._record_date_gaps.append(
                    {
                        "ts_code": event.ts_code,
                        "record_date": event.record_date,
                        "ex_trade_day": event.ex_trade_day,
                        "reason": "record_date 不早于除权生效日",
                    }
                )
            # ① 物理来源批次先加股（唯一一次），保证 I2 右值先于虚拟侧
            added_by_source = _apply_bonus_to_physical(
                physical=physical, event=event, ratio=event.share_ratio
            )
            holders = [
                sleeve_id
                for sleeve_id in sorted(sleeves)
                if sleeves[sleeve_id].shares_of(event.ts_code) > 0
            ]
            if event.share_ratio > 0.0 and holders and not added_by_source:
                raise CorporateError(
                    f"{event.event_id} 袖持有 {event.ts_code} 但物理来源批次缺失"
                    f"（袖={holders}）"
                )
            allocation = (
                _allocate_bonus(sleeves, event.ts_code, added_by_source)
                if event.share_ratio > 0.0
                else {}
            )
            if not added_by_source and event.cash_per_share <= 0.0:
                continue
            # ② 逐袖：应收 + 按物理来源分配新股
            for sleeve_id in holders:
                state = sleeves[sleeve_id]
                shares = int(state.shares_of(event.ts_code))
                cash = _cash_amount(shares, event.cash_per_share)
                added = 0
                if event.share_ratio > 0.0:
                    state, added = _apply_bonus_to_sleeve(
                        sleeve=state,
                        ts_code=event.ts_code,
                        ratio=event.share_ratio,
                        allocation=allocation.get(sleeve_id, {}),
                    )
                if cash > 0.0:
                    entry = ReceivableEntry(
                        receivable_id=str(event.event_id),
                        ts_code=event.ts_code,
                        amount=float(cash),
                        recognized_date=str(day),
                        settle_date=str(event.pay_trade_day),
                        shares=int(shares),
                    )
                    state = replace(state, receivables=state.receivables + (entry,))
                    physical.receivable += cash
                sleeves[sleeve_id] = state
                rows.append(
                    {
                        "trade_date": str(day),
                        "ts_code": event.ts_code,
                        "event_id": event.event_id,
                        "sleeve": sleeve_id,
                        "shares_at_record": int(shares),
                        "cash_per_share": float(event.cash_per_share),
                        "cash_receivable": float(cash),
                        "share_ratio": float(event.share_ratio),
                        "bonus_shares": int(added),
                        "shares_after": int(state.shares_of(event.ts_code)),
                        "cash_basis": event.cash_basis,
                        "ex_date": event.ex_date,
                        "ex_trade_day": event.ex_trade_day,
                        "pay_trade_day": event.pay_trade_day,
                    }
                )
            self._snapshot_counts[event.event_id] = self._snapshot_counts.get(event.event_id, 0) + 1
            _assert_receivable_symmetry(
                day=str(day),
                sleeves=sleeves,
                physical=physical,
                context=f"除息 {event.event_id}",
            )
        return rows

    # ────────────────────────── 支付日 ──────────────────────────

    def apply_pay_day(
        self, *, day: str, sleeves: Dict[str, SleeveState], physical: PhysicalBook
    ) -> List[Dict[str, Any]]:
        """到账日：**逐笔**应收转现金（逐袖与物理侧同额，I1/I1r 同时闭合）。"""
        stamp = str(day)
        rows: List[Dict[str, Any]] = []
        total_paid = 0.0
        for sleeve_id in sorted(sleeves):
            state = sleeves[sleeve_id]
            due = tuple(item for item in state.receivables if item.settle_date == stamp)
            if not due:
                continue
            paid = float(sum(item.amount for item in due))
            remaining = tuple(item for item in state.receivables if item.settle_date != stamp)
            sleeves[sleeve_id] = replace(
                state, cash=state.cash + paid, receivables=remaining
            )
            total_paid += paid
            for item in due:
                rows.append(
                    {
                        "trade_date": stamp,
                        "sleeve": sleeve_id,
                        "receivable_id": item.receivable_id,
                        "ts_code": item.ts_code,
                        "cash_paid": float(item.amount),
                        "shares": int(item.shares),
                        "recognized_date": item.recognized_date,
                        "settle_date": item.settle_date,
                    }
                )
        if total_paid > 0.0:
            physical.cash += total_paid
            physical.receivable = float(physical.receivable) - total_paid
            if abs(float(physical.receivable)) <= 1e-6:
                physical.receivable = 0.0
            _assert_receivable_symmetry(
                day=stamp, sleeves=sleeves, physical=physical, context="到账兑付"
            )
        return rows

    # ────────────────────────── 证据 ──────────────────────────

    def payload(self) -> Dict[str, Any]:
        return {
            "schema": "p7_t4_corporate_actions/v1",
            "calendar": [self._days[0], self._days[-1]],
            "cutoff": self._cutoff,
            "cash_basis_rule": (
                "cash_div（每股分红，税后）优先；缺失回退 cash_div_tax × 0.8（税前按 20% 税，"
                "沿 P3 预登记 §2.4）；持有期差异税率不建模"
            ),
            "share_rule": (
                "new_shares = floor(shares × (1 + stk_div))，新增股数并入原 lot、不重置买入日；"
                "物理来源 lot 先算再按虚拟批次比例分配，价格同除 (1 + ratio)"
            ),
            "stats": dict(sorted(self._stats.items())),
            "record_date_gaps": list(self._record_date_gaps[:50]),
            "record_date_gap_count": len(self._record_date_gaps),
            "events": [item.payload() for item in self._events],
            "event_count": len(self._events),
            "ex_days": len(self._by_ex),
            "pay_days": len(self._by_pay),
        }


def _assert_receivable_symmetry(
    *,
    day: str,
    sleeves: Mapping[str, SleeveState],
    physical: PhysicalBook,
    context: str,
) -> None:
    """公司行为后立即校验 ``Σ 袖应收 == 物理应收``（不对称即定位到事件）。"""
    total = float(sum(state.receivable for state in sleeves.values()))
    actual = float(physical.receivable)
    if abs(total - actual) <= 1e-6:
        return
    detail = {
        sid: [
            {"id": item.receivable_id, "amount": float(item.amount), "settle": item.settle_date}
            for item in state.receivables
        ]
        for sid, state in sorted(sleeves.items())
    }
    raise CorporateError(
        f"{day} {context} 后应收不对称：Σ袖={total!r} 物理={actual!r} "
        f"差={total - actual!r}；明细={detail}"
    )


def _cash_amount(shares: int, cash_per_share: float) -> float:
    if shares <= 0 or cash_per_share <= 0.0:
        return 0.0
    return float(shares) * float(cash_per_share)


def _apply_bonus_to_physical(
    *, physical: PhysicalBook, event: DividendEvent, ratio: float
) -> Dict[str, int]:
    """物理来源批次加股（每事件唯一一次）：``floor(shares × ratio)``，价格同除。"""
    code = str(event.ts_code)
    lots: List[PhysicalLot] = [
        lot for lot in physical.lots.values() if lot.ts_code == code
    ]
    if not lots:
        return {}
    if float(ratio) <= 0.0:
        return {}
    added_by_source: Dict[str, int] = {}
    for lot in sorted(lots, key=lambda item: item.source_lot_id):
        added = int(math.floor(float(lot.shares) * float(ratio) + 1e-12))
        added_by_source[lot.source_lot_id] = added
        physical.lots[lot.source_lot_id] = replace(
            lot,
            shares=int(lot.shares) + added,
            buy_price=float(lot.buy_price) / (1.0 + float(ratio)),
        )
    return added_by_source


def _allocate_bonus(
    sleeves: Mapping[str, SleeveState],
    ts_code: str,
    added_by_source: Mapping[str, int],
) -> Dict[str, Dict[str, int]]:
    """把每个物理来源批次的新股**跨袖**按虚拟持有比例分配（Σ分配 == Σ物理新增）。"""
    code = str(ts_code)
    out: Dict[str, Dict[str, int]] = {sleeve_id: {} for sleeve_id in sleeves}
    for source_lot_id in sorted(added_by_source):
        added = int(added_by_source[source_lot_id])
        if added <= 0:
            continue
        entries: List[Tuple[str, P7Lot]] = [
            (sleeve_id, lot)
            for sleeve_id in sorted(sleeves)
            for lot in sleeves[sleeve_id].lots
            if lot.ts_code == code and lot.source_lot_id == source_lot_id
        ]
        held = sum(int(lot.shares) for _sid, lot in entries)
        if held <= 0:  # pragma: no cover - 无虚拟持有时物理不应有来源批次
            raise CorporateError(
                f"物理来源批次 {source_lot_id} 有新增股但无虚拟持有（{code}）"
            )
        assigned = 0
        for sleeve_id, lot in entries:
            share = int(math.floor(float(added) * float(lot.shares) / float(held) + 1e-12))
            out[sleeve_id][lot.virtual_lot_id] = share
            assigned += share
        remainder = added - assigned
        if remainder > 0:
            order = sorted(entries, key=lambda item: (-int(item[1].shares), item[1].virtual_lot_id))
            for position in range(remainder):
                sleeve_id, lot = order[position % len(order)]
                out[sleeve_id][lot.virtual_lot_id] += 1
    return out


def _apply_bonus_to_sleeve(
    *,
    sleeve: SleeveState,
    ts_code: str,
    ratio: float,
    allocation: Mapping[str, int],
) -> Tuple[SleeveState, int]:
    """单袖按已分配股数加股；价格同除 ``(1+ratio)``（成本总额守恒）。"""
    code = str(ts_code)
    total_added = 0
    new_lots: List[P7Lot] = []
    for lot in sleeve.lots:
        if lot.ts_code != code:
            new_lots.append(lot)
            continue
        added = int(allocation.get(lot.virtual_lot_id, 0))
        total_added += added
        new_lots.append(
            replace(
                lot,
                shares=int(lot.shares) + added,
                acquired_price=float(lot.acquired_price) / (1.0 + float(ratio)),
            )
        )
    return replace(sleeve, lots=tuple(new_lots)), total_added


def build_corporate_book(
    *,
    calendar: Sequence[str],
    repo_root: Path,
    cutoff: Optional[str] = None,
    dividend_dir: str = DEFAULT_DIVIDEND_DIR,
) -> CorporateActionBook:
    """构造事件簿（宿主工厂统一入口；目录可显式覆盖，测试用）。"""
    path = Path(dividend_dir)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return CorporateActionBook(calendar=calendar, dividend_dir=path, cutoff=cutoff)


def merge_corporate_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    """按 ``(trade_date, ts_code, event_id, sleeve)`` 去重的证据行（只用于汇总）。"""
    seen: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("trade_date")),
            str(row.get("ts_code")),
            str(row.get("event_id")),
            str(row.get("sleeve")),
        )
        seen.setdefault(key, dict(row))
    return tuple(seen[key] for key in sorted(seen))


def assert_corporate_state(sleeves: Mapping[str, SleeveState], physical: PhysicalBook) -> None:
    """应收恒等式的状态侧自检（``Σ receivable_i == physical.receivable``）。"""
    total = sum(float(state.receivable) for state in sleeves.values())
    if abs(total - float(physical.receivable)) > 1e-6:
        raise StateError(
            f"应收恒等式破缺：Σ袖应收={total!r} != 物理应收={physical.receivable!r}"
        )
