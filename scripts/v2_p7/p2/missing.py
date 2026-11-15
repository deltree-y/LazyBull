# -*- coding: utf-8 -*-
"""缺价明细账：逐证券/日期/原因去重，按金额×持续期排序。

口径：
- 每个持仓-估值日最多一条记录；同键重复登记必须内容全等，冲突不覆盖；
- 金额使用"沿用最后有效价的持仓市值"（carried_value），不是成本价；
- 影响度使用 carried_value × 1 个估值日，先排序最大金额×持续时间来源；
- 停牌、退市、无报价、缺行分状态，均保留持仓，不用未来价回填或归零。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from typing import Iterable

from scripts.v2_p7.p2.types import (
    InstrumentError,
    PriceStatus,
    money,
    normalize_date,
    share_count,
)


class MissingPriceConflict(InstrumentError):
    """同一 (代码, 日期, 原因) 被以不同事实重复登记。"""


@dataclass(frozen=True, slots=True)
class MissingPriceRecord:
    """单个持仓/基准成员在单个估值日的缺价事实。

    ``shares`` 对实际账户为持股数；冻结权重基准没有实际股数，记 ``None``，
    不得用 0 冒充持仓。
    """

    as_of: str
    ts_code: str
    reason: PriceStatus
    shares: int | None
    carried_value: Decimal
    last_price_date: str
    days_since_last_quote: int
    missing_streak_days: int
    nav: Decimal
    unresolved_weight: float
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", normalize_date(self.as_of))
        object.__setattr__(self, "last_price_date", normalize_date(self.last_price_date))
        code = str(self.ts_code).strip().upper()
        if not code:
            raise InstrumentError("缺价记录 ts_code 不得为空")
        object.__setattr__(self, "ts_code", code)
        reason = PriceStatus(self.reason)
        if reason is PriceStatus.OK:
            raise InstrumentError("OK 不是缺价原因")
        object.__setattr__(self, "reason", reason)
        if self.shares is not None:
            object.__setattr__(self, "shares", share_count(self.shares))
        object.__setattr__(self, "carried_value", money(self.carried_value))
        nav = money(self.nav)
        if nav < 0:
            raise InstrumentError(f"缺价记录 NAV 不得为负: {nav}")
        object.__setattr__(self, "nav", nav)
        if self.carried_value < 0:
            raise InstrumentError("carried_value 不得为负")
        for field_name in ("days_since_last_quote", "missing_streak_days"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InstrumentError(f"{field_name} 必须为非负整数: {value!r}")
        if self.missing_streak_days < 1:
            raise InstrumentError("missing_streak_days 至少为 1")
        if not 0.0 <= float(self.unresolved_weight) <= 1.0:
            raise InstrumentError(f"unresolved_weight 越界: {self.unresolved_weight}")
        if not str(self.source).strip():
            raise InstrumentError("缺价记录 source 不得为空")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.as_of, self.ts_code, self.reason.value)


class MissingPriceLedger:
    """缺价明细账；追加式，不覆盖历史事实。"""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], MissingPriceRecord] = {}
        self._duplicate_hits = 0

    @property
    def duplicate_hits(self) -> int:
        """完全相同的重复登记次数（去重而非重复计数）。"""
        return self._duplicate_hits

    def record(self, item: MissingPriceRecord) -> tuple[MissingPriceRecord, bool]:
        """登记一条事实；返回 (记录, 是否重复)。同键不同内容直接拒绝。"""
        existing = self._records.get(item.key)
        if existing is not None:
            if existing != item:
                raise MissingPriceConflict(
                    f"缺价键冲突 {item.key}: 已登记 {existing!r}，新事实 {item!r}"
                )
            self._duplicate_hits += 1
            return existing, True
        self._records[item.key] = item
        return item, False

    @property
    def records(self) -> tuple[MissingPriceRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def __len__(self) -> int:
        return len(self._records)

    def top_sources(self, limit: int = 5) -> list[dict]:
        """按 carried_value×估值日聚合，返回金额×持续期最大的少数来源。"""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise InstrumentError(f"limit 必须为正整数: {limit!r}")
        grouped: dict[tuple[str, str], dict] = {}
        for item in self._records.values():
            key = (item.ts_code, item.reason.value)
            current = grouped.setdefault(
                key,
                {
                    "ts_code": item.ts_code,
                    "reason": item.reason.value,
                    "valuation_days": 0,
                    "first_date": item.as_of,
                    "last_date": item.as_of,
                    "value_days": Decimal("0.00"),
                    "max_carried_value": Decimal("0.00"),
                    "max_unresolved_weight": 0.0,
                    "max_missing_streak_days": 0,
                },
            )
            current["valuation_days"] += 1
            current["first_date"] = min(current["first_date"], item.as_of)
            current["last_date"] = max(current["last_date"], item.as_of)
            current["value_days"] += item.carried_value
            current["max_carried_value"] = max(current["max_carried_value"], item.carried_value)
            current["max_unresolved_weight"] = max(
                current["max_unresolved_weight"], float(item.unresolved_weight)
            )
            current["max_missing_streak_days"] = max(
                current["max_missing_streak_days"], item.missing_streak_days
            )
        rows = list(grouped.values())
        rows.sort(key=lambda row: (-row["value_days"], row["ts_code"], row["reason"]))
        return rows[:limit]

    def to_rows(self) -> list[dict]:
        """稳定排序的逐日明细（便于 CSV 审计）。"""
        rows: list[dict] = []
        for item in self.records:
            row = asdict(item)
            row["reason"] = item.reason.value
            row["carried_value"] = str(item.carried_value)
            row["nav"] = str(item.nav)
            rows.append(row)
        return rows


def calendar_days_between(start: str, end: str) -> int:
    """自然日差（缺价持续时间披露用，不是交易日计数）。"""
    first = datetime.strptime(normalize_date(start), "%Y%m%d")
    last = datetime.strptime(normalize_date(end), "%Y%m%d")
    return (last - first).days


def value_days(records: Iterable[MissingPriceRecord]) -> Decimal:
    """一组逐日缺价记录的金额×估值日总和。"""
    return sum((item.carried_value for item in records), Decimal("0.00"))
