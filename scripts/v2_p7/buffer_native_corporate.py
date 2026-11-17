"""固定续持原生历史回放的公司行为桥接；不接生产、不承担实盘税务复刻。

复杂度豁免：阶段专用研究诊断，不入生产链路。
使用 T4 的事件数值解析、应收入账、到账与送转计算；独立限制公告可知性、
登记日权益和送转可卖日。无可靠证据的实际持仓事件报错，不以零分红降级。
研究状态位于适配器，原生状态信封不承载它；本版只支持连续历史回放。
"""

from __future__ import annotations

import copy
import math
from bisect import bisect_left
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd

from scripts.v2_p7.t4.corporate import CorporateActionBook, CorporateError, DividendEvent, _finite
from scripts.v2_p7.t4.state import P7Lot, PhysicalBook, PhysicalLot, ReceivableEntry, SleeveState

_SID = "NATIVE"
_UNKNOWN_PAY_DAY = "99991231"
_FIELDS = (
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
    "div_listdate",
)


def _day(value: Any) -> str:
    """仅接收可规范为 YYYYMMDD 的日期；空值保持未知。"""
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("-", "")[:8]
    return text if len(text) == 8 and text.isdigit() else ""


class NativeCorporateBook(CorporateActionBook):
    """只读事件簿：事件版本按除权生效日已公告的信息选择，不取未来修订。"""

    def __init__(self, *, calendar: Sequence[str], dividend_dir: Path) -> None:
        days = tuple(str(day) for day in calendar)
        if not days or days != tuple(sorted(set(days))):
            raise CorporateError("研究公司行为日历必须非空且严格递增")
        self.details: dict[str, dict[str, Any]] = {}
        self.problems: dict[str, list[dict[str, Any]]] = {}
        self.undated_problems: list[dict[str, Any]] = []
        self.zero_effect_summary: dict[str, Any] = {}
        self.settled_before_start_proofs: list[dict[str, Any]] = []
        super().__init__(calendar=days, dividend_dir=dividend_dir, cutoff=days[-1])

    def _load(self, directory: Path) -> None:
        paths = sorted(directory.glob("*.parquet"))
        if not paths:
            raise CorporateError(f"研究公司行为分区缺件：{directory}")
        frames = []
        for path in paths:
            frame = pd.read_parquet(path)
            missing = set(_FIELDS) - set(frame.columns)
            if missing - {"div_listdate"}:
                raise CorporateError(f"公司行为字段缺失 {path}: {sorted(missing)}")
            frames.append(frame.reindex(columns=_FIELDS))
        data = pd.concat(frames, ignore_index=True).drop_duplicates()
        self._stats["rows_read"] = len(data)
        data = data.loc[data["div_proc"].eq("实施")].copy()
        self._stats["rows_implemented"] = len(data)
        for name in _FIELDS:
            if name.endswith("date"):
                data[name] = data[name].map(_day)
        self._annotate_effects(data)
        # 缺除权日也保留逐事件日期，不把几十年前的旧权利压缩成永久股票黑名单。
        for row in data.loc[data["ex_date"].eq("") & ~data["numeric_zero_effect"]].to_dict(
            "records"
        ):
            known_date = max(str(row["ann_date"]), str(row["imp_ann_date"]))
            if not row["record_date"] and self._record_prestart_settlement(row, known_date):
                continue
            self.undated_problems.append(
                {
                    "ts_code": str(row["ts_code"]),
                    "end_date": str(row["end_date"]),
                    "record_date": str(row["record_date"]),
                    "ann_date": str(row["ann_date"]),
                    "imp_ann_date": str(row["imp_ann_date"]),
                    "known_date": known_date,
                    "pay_date": str(row["pay_date"]),
                    "div_listdate": str(row["div_listdate"]),
                    "cash_per_share": float(row["parsed_cash_per_share"]),
                    "share_ratio": float(row["parsed_share_ratio"]),
                    "cash_basis": str(row["parsed_cash_basis"]),
                    "reason": "除权日未知" if known_date else "除权日未知且无任何可知公告日期",
                }
            )
        self._stats["settled_before_start_missing_dates_rows"] = len(
            self.settled_before_start_proofs
        )
        data = data.loc[data["ex_date"].ne("")]
        for (code, ex_date), group in data.groupby(["ts_code", "ex_date"], sort=True):
            ex_day = self._effective_day(str(ex_date), allow_before=True)
            if ex_day is None:
                continue
            group = group.copy()
            group["known_date"] = group[["ann_date", "imp_ann_date"]].max(axis=1)
            known = group.loc[group["known_date"].ne("") & group["known_date"].le(ex_day)]
            context = {
                "ts_code": str(code),
                "ex_date": str(ex_date),
                "record_dates": sorted(set(group["record_date"]) - {""}),
            }
            if known.empty:
                self.problems.setdefault(ex_day, []).append(
                    {**context, "reason": "除权时无已公告实施版本"}
                )
                continue
            chosen = known.loc[known["known_date"].eq(known["known_date"].max())]
            # 不提前删零效果历史行：最新可知版本为零时，绝不复活较早的正权益版本。
            if chosen["numeric_zero_effect"].all():
                self._stats["zero_effect_events"] += 1
                continue
            economic_fields = [
                "end_date",
                "record_date",
                "pay_date",
                "stk_div",
                "cash_div",
                "cash_div_tax",
                "div_listdate",
            ]
            if len(chosen[economic_fields].drop_duplicates()) != 1:
                self.problems.setdefault(ex_day, []).append(
                    {**context, "reason": "同一公告时点存在冲突实施版本"}
                )
                continue
            row = chosen.iloc[0].to_dict()
            event = super()._build_event(SimpleNamespace(**row))
            if event is None:
                continue
            pay_date = str(row["pay_date"])
            index = bisect_left(self._days, pay_date) if pay_date else len(self._days)
            pay_day = self._days[index] if index < len(self._days) else pay_date
            # 未知支付日保持应收；窗口外支付绝不回落除权日。
            event = replace(event, pay_trade_day=pay_day or _UNKNOWN_PAY_DAY)
            issues = []
            if not event.record_date or event.record_date >= event.ex_trade_day:
                issues.append("登记日缺失或不早于除权生效日")
            if pay_date and pay_date < event.ex_date and event.cash_per_share > 0:
                issues.append("支付日早于除权日")
            list_day = str(row["div_listdate"])
            if event.share_ratio > 0 and not list_day:
                issues.append("送转上市日未知，无法确定受限股可卖日期")
            self.details[event.event_id] = {
                "known_date": row["known_date"],
                "div_listdate": list_day,
                "future_versions_ignored": int(group["known_date"].gt(ex_day).sum()),
                "pay_status": (
                    "unknown"
                    if not pay_date
                    else ("after_window" if pay_day > self._days[-1] else "scheduled")
                ),
                "issues": issues,
            }
            self._events.append(event)
            self._by_ex.setdefault(event.ex_trade_day, []).append(event)
            self._by_pay.setdefault(event.pay_trade_day, []).append(event)
        self._stats["events_in_window"] = len(self._events)

    def _annotate_effects(self, data: pd.DataFrame) -> None:
        """日期门之前复用 T4 数值解释；仅标记，不改变公告版本选择的原始候选行。"""
        effects = []
        parsed_cash, parsed_ratio, parsed_basis = [], [], []
        examples = []
        missing_ex = missing_record = missing_cash = zero_count = 0
        for row in data.itertuples(index=False):
            cash, basis = self._cash(row)
            # 与 T4 CorporateActionBook._build_event 同一有限正送转规则，不新增归零口径。
            ratio = float(row.stk_div) if _finite(row.stk_div) and float(row.stk_div) > 0 else 0.0
            parsed_cash.append(cash)
            parsed_ratio.append(ratio)
            parsed_basis.append(basis)
            zero = cash <= 0.0 and ratio <= 0.0
            effects.append(zero)
            if not zero:
                continue
            zero_count += 1
            missing_ex += not row.ex_date
            missing_record += not row.record_date
            both_missing = bool(pd.isna(row.cash_div) and pd.isna(row.cash_div_tax))
            missing_cash += both_missing
            if len(examples) < 50:
                examples.append(
                    {
                        "ts_code": str(row.ts_code),
                        "end_date": str(row.end_date),
                        "ann_date": str(row.ann_date),
                        "imp_ann_date": str(row.imp_ann_date),
                        "record_date": str(row.record_date),
                        "ex_date": str(row.ex_date),
                        "cash_basis": basis,
                        "parsed_cash_per_share": cash,
                        "parsed_share_ratio": ratio,
                        "both_cash_fields_missing": both_missing,
                        "source_cash_div": self._source_number(row.cash_div),
                        "source_cash_div_tax": self._source_number(row.cash_div_tax),
                        "source_stk_div": self._source_number(row.stk_div),
                    }
                )
        data["numeric_zero_effect"] = pd.Series(effects, index=data.index, dtype=bool)
        data["parsed_cash_per_share"] = parsed_cash
        data["parsed_share_ratio"] = parsed_ratio
        data["parsed_cash_basis"] = parsed_basis
        self._stats["zero_effect_rows"] = zero_count
        self._stats["zero_effect_missing_ex_rows"] = missing_ex
        self.zero_effect_summary = {
            "input_zero_effect_rows": zero_count,
            "missing_ex_rows": missing_ex,
            "missing_record_rows": missing_record,
            "both_cash_fields_missing_rows": missing_cash,
            "cash_basis_counts": {"none": zero_count} if zero_count else {},
            "rule": "沿 T4 有限正现金/送转解析为零；cash_basis=none 不表示原始现金明确为0",
            "examples": examples,
            "example_limit": 50,
        }

    def _record_prestart_settlement(self, row: Mapping[str, Any], known_date: str) -> bool:
        """现金空仓起步的旧权益证明；不依公告年份猜登记日或支付完成。"""
        first = self._days[0]
        cash, ratio = float(row["parsed_cash_per_share"]), float(row["parsed_share_ratio"])
        pay, listing = str(row["pay_date"]), str(row["div_listdate"])
        cash_done = cash <= 0.0 or bool(pay and pay < first)
        shares_done = ratio <= 0.0 or bool(listing and listing < first)
        if not (cash_done and shares_done):
            return False
        source = {
            name: (
                self._source_number(row[name])
                if name in {"cash_div", "cash_div_tax", "stk_div"}
                else row[name]
            )
            for name in _FIELDS
        }
        self.settled_before_start_proofs.append(
            {
                "ts_code": str(row["ts_code"]),
                "end_date": str(row["end_date"]),
                "window_cash_start": first,
                "known_date": known_date,
                "cash_per_share": cash,
                "share_ratio": ratio,
                "cash_basis": str(row["parsed_cash_basis"]),
                "cash_settled_date": pay if cash > 0.0 else None,
                "shares_listed_date": listing if ratio > 0.0 else None,
                "source_fields": source,
                "reason": "全部正权益分量已按明确支付/上市日期在现金空仓起点前结清",
            }
        )
        return True

    @staticmethod
    def _source_number(value: Any) -> Any:
        """摘要保留缺失与非有限/非数字的区别，不把未知源字段写成明确零。"""
        if value is None or pd.isna(value):
            return None
        return float(value) if _finite(value) else str(value)


class _EventBridge(CorporateActionBook):
    """每次调用独立 T4 入账器，避免多个账户共享事件簿统计的可变状态。"""

    def __init__(self, event: DividendEvent | None = None) -> None:
        self._by_ex = {} if event is None else {event.ex_trade_day: [event]}
        self._record_date_gaps: list[dict[str, Any]] = []
        self._snapshot_counts: dict[str, int] = {}


class _NativePriceBridge:
    """只调整停牌缺价时的上次已知价格，不写入当日行情或复权价格表。"""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def apply_ex_adjustment(
        self, code: str, *, cash_per_share: float, share_ratio: float, ex_date: str
    ) -> None:
        position = self.engine.positions.get(code)
        if not position:
            return
        value = position.get("last_known_price", position.get("buy_trade_price"))
        if value is None or not math.isfinite(float(value)) or float(value) <= cash_per_share:
            raise CorporateError(f"{ex_date} {code} 缺价除权基准不可用：{value!r}")
        position["last_known_price"] = (float(value) - cash_per_share) / (1.0 + share_ratio)


class NativeCorporateAdapter:
    """单账户公司行为状态；权益记录、现金到账与原生交易状态分开保存。"""

    def __init__(self, book: NativeCorporateBook) -> None:
        self.book = book
        self.receivables: tuple[ReceivableEntry, ...] = ()
        self.rows: list[dict[str, Any]] = []
        self.issues: list[dict[str, Any]] = []
        self.locks: list[dict[str, Any]] = []
        self._record_holdings: dict[str, dict[str, dict[str, Any]]] = {}
        self._before_days: set[str] = set()
        self._closed_days: set[str] = set()
        self._last_day = ""

    @property
    def receivable(self) -> float:
        """应收仅计权益，绝不加入原生可用现金。"""
        return float(sum(item.amount for item in self.receivables))

    def receivable_value(self) -> float:
        """日终账本校验的显式读取接口。"""
        return self.receivable

    def locked_shares(self, code: str) -> int:
        """独立受限股份，只限制成交，不从总股数或估值中扣除。"""
        return sum(int(lock["shares"]) for lock in self.locks if lock["ts_code"] == code)

    def sellable_shares(self, engine: Any, code: str) -> int:
        """可卖数量以真实总股数减独立锁定股数计算，不作整手缩量。"""
        total = int(engine.positions.get(code, {}).get("shares", 0))
        locked = self.locked_shares(code)
        if locked > total:
            self._fail(self._last_day, code, f"受限股 {locked} 超过真实总股数 {total}")
        return total - locked

    def check_sell(self, engine: Any, date: Any, code: str, requested: int) -> None:
        """原生最终卖出前校验；不替调用者缩量、挂单或更改部分成交规则。"""
        locked = self.locked_shares(code)
        if not locked:
            return
        available = self.sellable_shares(engine, code)
        if requested <= available:
            return
        day = _day(date)
        self.rows.append(
            {
                "kind": "sell_blocked_locked",
                "trade_date": day,
                "ts_code": code,
                "requested_shares": int(requested),
                "sellable_shares": available,
                "locked_shares": locked,
            }
        )
        self._fail(day, code, f"原生卖单 {requested} 股超过可卖 {available} 股，锁定 {locked} 股")

    def _unlock(self, day: str) -> None:
        """已知上市日开盘前释放独立锁，不产生任何现金或股数变化。"""
        due = [lock for lock in self.locks if lock["unlock_trade_day"] <= day]
        self.locks = [lock for lock in self.locks if lock["unlock_trade_day"] > day]
        self.rows.extend({**lock, "kind": "unlock", "trade_date": day} for lock in due)

    def _fail(self, day: str, code: str, reason: str) -> None:
        self.issues.append({"trade_date": day, "ts_code": code, "reason": reason})
        raise CorporateError(f"{day} 实际持仓 {code} 公司行为不可可靠处理：{reason}")

    def _has_ownership(self, engine: Any, problem: Mapping[str, Any]) -> bool:
        code = str(problem["ts_code"])
        if code in engine.positions:
            return True
        return any(
            code in self._record_holdings.get(day, {}) for day in problem.get("record_dates", ())
        )

    def _check_undated(self, engine: Any, day: str) -> None:
        """缺除权日按当时可知的登记权益判定；未来公告不得提前阻断。"""
        for problem in self.book.undated_problems:
            known = str(problem["known_date"])
            if known and known > day:
                continue
            code = str(problem["ts_code"])
            record = str(problem["record_date"])
            if record:
                # 研究窗口现金空仓起步，窗前权益恒为零；未来登记尚未产生权利。
                if record < self.book._days[0] or record > day:
                    continue
                snapshot = self._record_holdings.get(record)
                if snapshot is None:
                    if record == day:
                        # 当天收盘前不猜权益；after_day 将读取真实收盘持仓。
                        continue
                    if code in engine.positions:
                        self._fail(day, code, "缺除权日事件的登记日权益快照缺失")
                elif code in snapshot:
                    self._fail(day, code, str(problem["reason"]))
            elif code in engine.positions:
                self._fail(day, code, str(problem["reason"]) + "；登记日也未知")

    def before_day(self, engine: Any, date: Any) -> None:
        """当日交易前处理；重复调用无效，跳日/回退不静默漏支付。"""
        day = _day(date)
        if day not in self.book._days:
            raise CorporateError(f"公司行为日期不在冻结日历：{day}")
        if day in self._before_days:
            return
        if self._last_day:
            prior = self.book._days.index(self._last_day)
            if prior + 1 >= len(self.book._days) or self.book._days[prior + 1] != day:
                raise CorporateError(f"公司行为要求连续历史驱动：{self._last_day} → {day}")
            if self._last_day not in self._closed_days:
                raise CorporateError(f"公司行为缺少上一日收盘权益快照：{self._last_day}")
        self._check_undated(engine, day)
        for problem in self.book.problems.get(day, []):
            if self._has_ownership(engine, problem):
                self._fail(day, str(problem["ts_code"]), str(problem["reason"]))
        self._unlock(day)
        for event in self.book.ex_events(day):
            self._apply_event(engine, day, event)
        sleeves = {
            _SID: SleeveState(
                _SID, float(engine.current_capital), 0.0, receivables=self.receivables
            )
        }
        physical = PhysicalBook(float(engine.current_capital), receivable=self.receivable)
        paid = _EventBridge().apply_pay_day(day=day, sleeves=sleeves, physical=physical)
        if paid:
            engine.current_capital = physical.cash
        self.receivables = sleeves[_SID].receivables
        self.rows.extend({**row, "kind": "pay"} for row in paid)
        self._before_days.add(day)
        self._last_day = day

    def _apply_event(self, engine: Any, day: str, event: DividendEvent) -> None:
        code = event.ts_code
        snapshot = self._record_holdings.get(event.record_date)
        position = None if snapshot is None else snapshot.get(code)
        current = engine.positions.get(code)
        if current and not event.record_date:
            self._fail(day, code, "登记日未知，无法确认实际持仓权益")
        if position is None:
            if current and snapshot is None and event.record_date >= self.book._days[0]:
                self._fail(day, code, "登记日收盘权益快照缺失")
            return
        detail = self.book.details[event.event_id]
        if detail["issues"]:
            self._fail(day, code, "; ".join(detail["issues"]))
        if event.share_ratio > 0 and (
            not current
            or current["shares"] != position["shares"]
            or current.get("buy_date") != position.get("buy_date")
        ):
            self._fail(day, code, "登记日到除权日送转权益对应原生批次已变化")
        shares = int(position["shares"])
        buy_date = _day(position["buy_date"])
        price = float(position["buy_trade_price"])
        lot_id = f"{code}-{buy_date}"
        lot = P7Lot(
            lot_id,
            code,
            lot_id,
            shares,
            buy_date,
            price,
            0.0,
            buy_date,
            "99991231",
            _day(position.get("signal_date", buy_date)),
            0,
        )
        sleeves = {
            _SID: SleeveState(
                _SID, float(engine.current_capital), 0.0, lots=(lot,), receivables=self.receivables
            )
        }
        physical = PhysicalBook(float(engine.current_capital), receivable=self.receivable)
        physical.add_lot(PhysicalLot(lot_id, code, shares, buy_date, price, 0.0))
        rows = _EventBridge(event).apply_ex_day(
            day=day, sleeves=sleeves, physical=physical, prices=_NativePriceBridge(engine)
        )
        added = sleeves[_SID].shares_of(code) - shares
        if added:
            current["shares"] += added
            for name in ("buy_trade_price", "buy_pnl_price"):
                if name in current:
                    current[name] = float(current[name]) / (1.0 + event.share_ratio)
            listing = str(detail["div_listdate"])
            if listing > day:
                index = bisect_left(self.book._days, listing)
                unlock = self.book._days[index] if index < len(self.book._days) else listing
                lock = {
                    "event_id": event.event_id,
                    "ts_code": code,
                    "shares": int(added),
                    "recognized_date": day,
                    "div_listdate": listing,
                    "unlock_trade_day": unlock,
                }
                self.locks.append(lock)
                self.rows.append({**lock, "kind": "lock", "trade_date": day})
        self.receivables = sleeves[_SID].receivables
        self.rows.extend(
            {
                **row,
                **detail,
                "kind": "ex",
                "record_date": event.record_date,
                "pay_date": event.pay_date,
            }
            for row in rows
        )

    def after_day(self, engine: Any, date: Any) -> None:
        """保存登记日所需的真实收盘权益；快照只读、不参与买卖决策。"""
        day = _day(date)
        if day not in self._before_days:
            raise CorporateError(f"{day} 缺少公司行为交易前阶段")
        if day in self._closed_days:
            return
        self._record_holdings[day] = copy.deepcopy(engine.positions)
        self._closed_days.add(day)
        self._check_undated(engine, day)

    def payload(self) -> dict[str, Any]:
        """供研究报告引用的状态，不是原生磁盘恢复信封。"""
        return {
            "schema": "p7_buffer_native_corporate/v2",
            "tax_rule": "沿 T4 cash_div 优先、缺失 cash_div_tax×0.8；不复刻持有期差异税",
            "zero_effect_summary": copy.deepcopy(self.book.zero_effect_summary),
            "settled_before_start_count": len(self.book.settled_before_start_proofs),
            "settled_before_start_proofs": copy.deepcopy(self.book.settled_before_start_proofs),
            "receivable": self.receivable,
            "receivables": [asdict(item) for item in self.receivables],
            "rows": list(self.rows),
            "issues": list(self.issues),
            "held_issue_count": len(self.issues),
            "locks": copy.deepcopy(self.locks),
            "locked_shares": {
                code: self.locked_shares(code)
                for code in sorted({x["ts_code"] for x in self.locks})
            },
            "lock_count": sum(row["kind"] == "lock" for row in self.rows),
            "unlock_count": sum(row["kind"] == "unlock" for row in self.rows),
            "blocked_sell_count": sum(row["kind"] == "sell_blocked_locked" for row in self.rows),
            "record_snapshots": {
                "count": len(self._record_holdings),
                "first_day": min(self._record_holdings, default=None),
                "last_day": max(self._record_holdings, default=None),
                "held_code_days": sum(len(value) for value in self._record_holdings.values()),
            },
            "cash_paid_by_day": {
                day: sum(
                    float(row["cash_paid"])
                    for row in self.rows
                    if row["kind"] == "pay" and row["trade_date"] == day
                )
                for day in sorted({row["trade_date"] for row in self.rows if row["kind"] == "pay"})
            },
            "ex_count": sum(row["kind"] == "ex" for row in self.rows),
            "pay_count": sum(row["kind"] == "pay" for row in self.rows),
            "cash_basis_fallback_held": sum(
                row["kind"] == "ex" and row.get("cash_basis") == "cash_div_tax_x0.8"
                for row in self.rows
            ),
            "continuous_history_only": True,
        }


class NativeCorporateMixin:
    """研究引擎前置 Mixin；关闭时直接委托原生实现且不增加研究状态字段。"""

    def __init__(
        self, *args: Any, corporate_book: NativeCorporateBook | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if corporate_book is not None:
            self.native_corporate = NativeCorporateAdapter(corporate_book)

    def run_step(self, date: Any, trading_dates_full: Any, price_data: Any) -> None:
        adapter = getattr(self, "native_corporate", None)
        if adapter is None:
            return super().run_step(date, trading_dates_full, price_data)
        adapter.before_day(self, date)
        result = super().run_step(date, trading_dates_full, price_data)
        adapter.after_day(self, date)
        return result

    def _calculate_portfolio_value(self, date: Any) -> float:
        value = super()._calculate_portfolio_value(date)
        adapter = getattr(self, "native_corporate", None)
        return value if adapter is None else value + adapter.receivable

    def _sell_stock_direct(
        self,
        date: Any,
        stock: str,
        sell_type: str = "holding_period",
        sell_reason: str | None = None,
        trigger_type: str | None = None,
        fraction: float = 1.0,
    ) -> None:
        """最终原生卖出入口校验锁定股份；关闭/无锁时原样委托。"""
        adapter = getattr(self, "native_corporate", None)
        if adapter is not None and adapter.locked_shares(stock) and 0.0 < fraction <= 1.0:
            total = int(self.positions.get(stock, {}).get("shares", 0))
            requested = self._resolve_trim_shares(total, fraction)
            adapter.check_sell(self, date, stock, requested)
        return super()._sell_stock_direct(
            date, stock, sell_type, sell_reason, trigger_type, fraction=fraction
        )
