# -*- coding: utf-8 -*-
"""P7 T4 价源分离：成交价（只认真实 T+1 开盘）与估值价（三级解析链）。

契约依据（``docs/contracts/netting_freeze.md`` F26 §「成交价与估值价分离」）
=====================================================================
- **成交价**只认真实 ``T+1`` 开盘价；开盘不可用即**不成交并登记**
  （``reason=no_valid_open`` 等），**禁止任何回退到当日收盘**，不得伪造成交价；
- **估值价**用「当日价 -> 最后已知价 -> 买入价」解析链，物理与虚拟**共享**同一解析结果，
  逐股记录 ``price_date`` / ``price_source``；
- **缓存估值不得完成 B1 内部转账**：内部转账的价格必须来自当日**真实开盘价**
  （``execution_price`` 成功），估值链的兜底价一概不得进入转账；
- **连续停牌期间每日可估值，恢复后逐位一致**：停牌段内每日估值必须成功且价格恒等于
  停牌前最后已知价；恢复首个真实有价日必须回到 ``day_close`` 口径。

实现纪律
========
- 逐日**流式**读取 ``clean/daily``（不缓存全日历，内存占用与日历长度无关）；
- 查询（``execution_price`` / ``valuation``）与登记（``note_valuation``）分离：
  只有真正参与账本估值的查询才进入分布统计，重复查询不会污染证据；
- 本模块只做只读查询与证据登记，不改账本、不写盘。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: 估值价来源标签（证据枚举；顺序即解析优先级）
SOURCE_DAY_CLOSE = "day_close"
SOURCE_LAST_KNOWN = "last_known"
SOURCE_BUY_PRICE = "buy_price"
PRICE_SOURCES: Tuple[str, ...] = (SOURCE_DAY_CLOSE, SOURCE_LAST_KNOWN, SOURCE_BUY_PRICE)
SOURCE_UNAVAILABLE = "unavailable"

#: 停牌（连续无真实有价日）登记的最短段长
SUSPENSION_EPISODE_MIN_DAYS = 2
#: 停牌段登记上限（超出只计数，避免证据膨胀）
SUSPENSION_EPISODE_LIMIT = 500


class PriceError(RuntimeError):
    """价源帧缺列或价源职责被越界调用。"""


@dataclass(frozen=True)
class Quote:
    """单只股票单日的真实行情快照（成交可用性与估值价的唯一来源）。"""

    ts_code: str
    open: float
    close: float
    is_suspended: int
    is_limit_up: int
    is_limit_down: int

    @property
    def has_open(self) -> bool:
        return math.isfinite(self.open) and self.open > 0.0

    @property
    def has_close(self) -> bool:
        return math.isfinite(self.close) and self.close > 0.0


@dataclass(frozen=True)
class ExecutionLookup:
    """成交价查询结果（``price=None`` 时 ``reason`` 给出阻断原因）。"""

    price: Optional[float]
    reason: str

    @property
    def ok(self) -> bool:
        return self.price is not None


class PriceBook:
    """逐日价源（真实开盘价 + 三级估值解析 + 逐股来源登记）。"""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._day: Optional[str] = None
        self._today: Dict[str, Quote] = {}
        self._last_close: Dict[str, Tuple[float, str]] = {}
        self._active_gaps: Dict[str, Dict[str, Any]] = {}
        self._distribution: Dict[str, int] = {}
        self._per_stock: Dict[str, Dict[str, Any]] = {}
        self._episodes: List[Dict[str, Any]] = []
        self._episode_truncated = 0
        self._block_reasons: Dict[str, int] = {}
        self._suspension_checks: Dict[str, int] = {"checked": 0, "constant": 0, "drift": 0}
        self._drift_examples: List[Dict[str, Any]] = []
        self._days = 0
        self._tracked: Tuple[str, ...] = ()
        self._independent_cache: Dict[Tuple[str, str], Optional[float]] = {}
        self._resume_checks: Dict[str, Any] = {"checked": 0, "bitwise_equal": 0, "mismatch": []}

    # ────────────────────────── 续跑状态（C4-04） ──────────────────────────

    def snapshot_last_known(self) -> Dict[str, Any]:
        """导出「最后已知收盘」缓存（检查点用；重启必须从这里恢复，否则估值会退化）。"""
        return {
            code: {"close": float(close), "day": str(day)}
            for code, (close, day) in sorted(self._last_close.items())
        }

    def restore_last_known(self, snapshot: Mapping[str, Any]) -> None:
        """从快照恢复「最后已知收盘」缓存（缺失/非法条目跳过，不静默填 0）。"""
        for code in sorted(snapshot or {}):
            item = snapshot[code]
            close = float(item.get("close") or 0.0)
            if close <= 0.0:
                continue
            self._last_close[str(code)] = (close, str(item.get("day") or ""))

    # ────────────────────────── 逐日装载 ──────────────────────────

    def begin_day(self, day: str, tracked: Optional[Sequence[str]] = None) -> None:
        """装载当日 ``clean/daily`` 帧并推进「最后已知价」与停牌段状态。

        Args:
            day: 交易日（YYYYMMDD）
            tracked: 需要跟踪停牌段的股票（当日持仓与待执行意图的代码）。
                ``clean/daily`` 在停牌日**不产出行**，只有显式跟踪集合才能识别
                「连续无真实有价日」；不跟踪的股票不建立停牌段（避免为全市场
                数千只退市股登记无意义段落）。
        """
        self._day = str(day)
        self._days += 1
        self._tracked = tuple(sorted({str(code) for code in (tracked or ())}))
        frame = self._loader.load_clean_daily_by_date(day, auto_ensure=False)
        if frame is None or len(frame) == 0:
            self._today = {}
            self._advance_gaps(quotes={})
            return
        required = {"ts_code", "open", "close", "is_suspended", "is_limit_up", "is_limit_down"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise PriceError(f"{day} clean/daily 缺列：{missing}")
        quotes: Dict[str, Quote] = {}
        for row in frame.itertuples(index=False):
            quotes[str(row.ts_code)] = Quote(
                ts_code=str(row.ts_code),
                open=float(row.open) if _finite(row.open) else 0.0,
                close=float(row.close) if _finite(row.close) else 0.0,
                is_suspended=int(getattr(row, "is_suspended", 0) or 0),
                is_limit_up=int(getattr(row, "is_limit_up", 0) or 0),
                is_limit_down=int(getattr(row, "is_limit_down", 0) or 0),
            )
        self._today = quotes
        for code, quote in quotes.items():
            if quote.has_close:
                self._last_close[code] = (quote.close, self._day)
        self._advance_gaps(quotes=quotes)

    def _advance_gaps(self, quotes: Dict[str, Quote]) -> None:
        """维护「连续无真实收盘价」段（停牌 / 缺行 / close 为 NaN）。

        只对 ``begin_day(tracked=...)`` 显式跟踪的股票建档（持仓与待执行意图），
        因此段数上界由持仓数决定，不会为全市场退市股登记无意义段落。
        """
        for code in list(self._active_gaps):
            quote = quotes.get(code)
            if quote is not None and quote.has_close:
                self._close_gap(code)
        tracked = getattr(self, "_tracked", ())
        for code in tracked:
            if code in self._active_gaps:
                continue
            quote = quotes.get(code)
            if quote is not None and quote.has_close:
                continue
            known = self._last_close.get(code)
            self._active_gaps[code] = {
                "start": self._day,
                "days": 0,
                "pre_gap_price": float(known[0]) if known else None,
                "pre_gap_date": str(known[1]) if known else "",
            }
        for gap in self._active_gaps.values():
            gap["days"] += 1

    def _close_gap(self, code: str) -> None:
        """停牌段结束（当日回到真实收盘价）：登记段长、停牌前价格与恢复日一致性。

        「恢复后逐位一致」在此落地为**独立复读**比较：恢复日按真实收盘价的估值
        必须与重新读取原始 ``clean/daily`` 得到的收盘价**逐位相等**。
        """
        gap = self._active_gaps.pop(code)
        resume_close = float(self._today[code].close)
        independent = self._independent_close(str(self._day), code)
        self._resume_checks["checked"] += 1
        if independent is not None and float(independent) == resume_close:
            self._resume_checks["bitwise_equal"] += 1
        else:
            self._resume_checks["mismatch"].append(
                {
                    "ts_code": code,
                    "resume_date": self._day,
                    "expected_raw_close": independent,
                    "resume_raw_close": resume_close,
                }
            )
        if int(gap["days"]) >= SUSPENSION_EPISODE_MIN_DAYS:
            if len(self._episodes) < SUSPENSION_EPISODE_LIMIT:
                self._episodes.append(
                    {
                        "ts_code": code,
                        "gap_start": str(gap["start"]),
                        "gap_days": int(gap["days"]),
                        "resume_date": self._day,
                        "pre_gap_price": gap["pre_gap_price"],
                        "pre_gap_date": gap["pre_gap_date"],
                        "resume_raw_close": resume_close,
                        "resume_independent_close": independent,
                        "resume_source": SOURCE_DAY_CLOSE,
                        "resume_close_bitwise_equal": bool(
                            independent is not None and float(independent) == resume_close
                        ),
                    }
                )
            else:
                self._episode_truncated += 1

    # ────────────────────────── 成交价 ──────────────────────────

    def market_open(self, ts_code: str) -> ExecutionLookup:
        """当日真实开盘价是否可用（**不含**涨跌停方向判定）。

        内部转账（B1）只需要"当日有真实开盘价"即可完成；涨跌停只阻断市场指令，
        不阻断内部抵消（F26 / B4「退出不被剥夺」）。
        """
        quote = self._today.get(str(ts_code))
        if quote is None:
            return ExecutionLookup(None, "row_absent")
        if quote.is_suspended:
            return ExecutionLookup(None, "suspended")
        if not quote.has_open:
            return ExecutionLookup(None, "no_valid_open")
        return ExecutionLookup(float(quote.open), "ok")

    def execution_price(self, ts_code: str, side: str) -> ExecutionLookup:
        """真实 T+1 开盘价（市场成交唯一来源；**绝不回退当日收盘**）。

        Returns:
            ``ExecutionLookup``；``price=None`` 时 ``reason`` ∈
            ``{row_absent, suspended, no_valid_open, limit_up_blocked, limit_down_blocked}``。
        """
        if side not in ("buy", "sell"):
            raise PriceError(f"成交价查询方向必须为 buy/sell：{side!r}")
        available = self.market_open(ts_code)
        if not available.ok:
            return available
        quote = self._today[str(ts_code)]
        if side == "buy" and quote.is_limit_up:
            return ExecutionLookup(None, "limit_up_blocked")
        if side == "sell" and quote.is_limit_down:
            return ExecutionLookup(None, "limit_down_blocked")
        return ExecutionLookup(float(quote.open), "ok")

    def record_block(self, reason: str) -> None:
        """登记成交阻断原因计数（证据用；``ok`` 不登记）。"""
        if reason == "ok":
            return
        self._block_reasons[reason] = self._block_reasons.get(reason, 0) + 1

    # ────────────────────────── 估值价 ──────────────────────────

    def valuation(self, ts_code: str, buy_price: float) -> Tuple[float, str, str]:
        """三级估值解析：当日价 -> 最后已知价 -> 买入价（逐股返回日期与来源）。

        Returns:
            ``(price, price_date, source)``；三级全不可用时 ``price=0``、
            ``source=unavailable``（调用方必须登记为估值不可用，不得伪造价格）。
        """
        code = str(ts_code)
        quote = self._today.get(code)
        if quote is not None and quote.has_close:
            return float(quote.close), self._day or "", SOURCE_DAY_CLOSE
        known = self._last_close.get(code)
        if known is not None and float(known[0]) > 0.0:
            return float(known[0]), str(known[1]), SOURCE_LAST_KNOWN
        if _finite(buy_price) and float(buy_price) > 0.0:
            return float(buy_price), "", SOURCE_BUY_PRICE
        return 0.0, "", SOURCE_UNAVAILABLE

    def note_valuation(self, ts_code: str, price: float, source: str, price_date: str = "") -> None:
        """登记一次**实际参与账本估值**的取值（分布 + 逐股 + 日期 + 停牌恒定性核对）。

        停牌期的期望价来自**独立复读**：建段时记录的 ``pre_gap_date`` 对应的
        ``clean/daily`` 原始收盘价（``self._independent_close`` 现读，不复用
        ``_last_close`` 缓存），因此 ``constant_to_pre_gap`` 不是同源自证。
        """
        code = str(ts_code)
        self._distribution[source] = self._distribution.get(source, 0) + 1
        row = self._per_stock.get(code)
        if row is None:
            row = {
                "days": 0,
                SOURCE_DAY_CLOSE: 0,
                SOURCE_LAST_KNOWN: 0,
                SOURCE_BUY_PRICE: 0,
                SOURCE_UNAVAILABLE: 0,
                "dates_recorded": 0,
                "last_price_date": "",
                "last_source": "",
            }
            self._per_stock[code] = row
        row["days"] += 1
        row[source] = row.get(source, 0) + 1
        row["last_price_date"] = str(price_date)
        row["last_source"] = str(source)
        if str(price_date):
            row["dates_recorded"] += 1
        gap = self._active_gaps.get(code)
        if gap is None:
            return
        expected = self._independent_close(str(gap.get("pre_gap_date") or ""), code)
        self._suspension_checks["checked"] += 1
        if expected is not None and float(price) == float(expected) and source == SOURCE_LAST_KNOWN:
            self._suspension_checks["constant"] += 1
        else:
            self._suspension_checks["drift"] += 1
            if len(self._drift_examples) < 20:
                self._drift_examples.append(
                    {
                        "ts_code": code,
                        "trade_date": self._day,
                        "gap_start": str(gap["start"]),
                        "gap_days": int(gap["days"]),
                        "expected_pre_gap_close": expected,
                        "expected_source": SOURCE_LAST_KNOWN,
                        "actual_valuation_price": float(price),
                        "source": source,
                    }
                )

    def _independent_close(self, day: str, ts_code: str) -> Optional[float]:
        """独立复读某一交易日的原始收盘价（不经 ``_last_close`` 缓存，防同源自证）。"""
        if not day:
            return None
        key = (str(day), str(ts_code))
        cached = self._independent_cache.get(key)
        if cached is not None or key in self._independent_cache:
            return cached
        frame = self._loader.load_clean_daily_by_date(str(day), auto_ensure=False)
        value: Optional[float] = None
        if frame is not None and len(frame) > 0 and "ts_code" in frame.columns:
            matched = frame[frame["ts_code"].astype(str) == str(ts_code)]
            if len(matched) == 1:
                close = float(matched.iloc[0]["close"])
                if math.isfinite(close) and close > 0.0:
                    value = close
        self._independent_cache[key] = value
        return value

    def last_known(self, ts_code: str) -> Optional[Tuple[float, str]]:
        """最后已知真实收盘价与日期（只读；不参与成交与转账）。"""
        return self._last_close.get(str(ts_code))

    def active_gaps(self) -> Tuple[str, ...]:
        """当前仍无真实收盘价的股票（停牌/缺行）代码。"""
        return tuple(sorted(self._active_gaps))

    # ────────────────────────── 证据 ──────────────────────────

    def payload(self) -> Dict[str, Any]:
        """``price_source.json`` 载荷（分布 + 逐股摘要 + 停牌段 + 恒定性核对）。"""
        rows = [
            {
                "ts_code": code,
                "days": int(row["days"]),
                "day_close": int(row.get(SOURCE_DAY_CLOSE, 0)),
                "last_known": int(row.get(SOURCE_LAST_KNOWN, 0)),
                "buy_price": int(row.get(SOURCE_BUY_PRICE, 0)),
                "unavailable": int(row.get(SOURCE_UNAVAILABLE, 0)),
                "dates_recorded": int(row.get("dates_recorded", 0)),
                "last_price_date": str(row.get("last_price_date", "")),
                "last_source": str(row.get("last_source", "")),
            }
            for code, row in sorted(self._per_stock.items())
        ]
        total = sum(int(row["days"]) for row in rows)
        dated = sum(int(row["dates_recorded"]) for row in rows)
        return {
            "schema": "p7_t4_price_source/v1",
            "days": int(self._days),
            "stocks": len(rows),
            "valuation_queries": int(total),
            "valuation_dates_recorded": int(dated),
            "valuation_dates_missing": int(total - dated),
            "distribution": {key: int(value) for key, value in sorted(self._distribution.items())},
            "distribution_share": {
                key: (float(value) / float(total) if total else None)
                for key, value in sorted(self._distribution.items())
            },
            "execution_block_reasons": {
                key: int(value) for key, value in sorted(self._block_reasons.items())
            },
            "per_stock": rows,
            "suspension_episodes": list(self._episodes),
            "suspension_episode_limit": SUSPENSION_EPISODE_LIMIT,
            "suspension_episodes_truncated": int(self._episode_truncated),
            "suspension_valuation": {
                "checked": int(self._suspension_checks["checked"]),
                "constant_to_pre_gap": int(self._suspension_checks["constant"]),
                "drift": int(self._suspension_checks["drift"]),
                "drift_examples": list(self._drift_examples),
                "expectation_basis": (
                    "期望价来自独立复读 pre_gap_date 的 clean/daily 原始收盘价（不复用 last_known 缓存）；"
                    "来源必须为 last_known。"
                ),
            },
            "resume_consistency": {
                "checked": int(self._resume_checks["checked"]),
                "bitwise_equal": int(self._resume_checks["bitwise_equal"]),
                "mismatch": list(self._resume_checks["mismatch"]),
                "basis": "恢复日估值来源必须为 day_close，且价格与独立复读的原始收盘价逐位相等。",
            },
            "notes": {
                "execution": (
                    "成交价只取当日真实开盘价（open>0、未停牌、非反向涨跌停）；"
                    "开盘不可用即不成交并登记原因，禁止回退当日收盘。"
                ),
                "valuation": (
                    "估值价 = 当日价(close) -> 最后已知价 -> 买入价；物理与虚拟共用同一解析，"
                    "逐股登记 last_price_date / last_source。"
                ),
                "transfer": "内部转账价格只认当日真实开盘价；估值链兜底价不得完成 B1 转账。",
            },
        }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
