# -*- coding: utf-8 -*-
"""冻结权重毛基准：信号日定成员/权重，缺价保留份额，不重排、不回填未来价。

这是毛机会成本尺子，不是可执行账户：没有现金、整手或成交假设。
账户侧必须使用 :class:`ResearchAccount`，不得把本类 NAV 冒充可交易净值。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Mapping

from scripts.v2_p7.p2.missing import (
    MissingPriceLedger,
    MissingPriceRecord,
    calendar_days_between,
)
from scripts.v2_p7.p2.types import (
    InstrumentError,
    MarketSnapshot,
    PriceStatus,
    money,
    normalize_date,
)

ZERO = Decimal("0.00")
UNIT_SCALE = Decimal("0.000000000001")


def equal_weights(ts_codes: list[str]) -> dict[str, Decimal]:
    """等权映射（重复/空代码拒绝；总权重视为 1）。"""
    normalized = [str(code).strip().upper() for code in ts_codes]
    if not normalized or any(not code for code in normalized):
        raise InstrumentError("等权成员不得为空")
    if len(set(normalized)) != len(normalized):
        raise InstrumentError("等权成员包含重复代码")
    weight = Decimal(1) / Decimal(len(normalized))
    return {code: weight for code in normalized}


@dataclass(slots=True)
class _BasketMember:
    ts_code: str
    weight: Decimal
    units: Decimal
    last_price: Decimal
    last_price_date: str
    missing_streak_days: int = 0


@dataclass(frozen=True, slots=True)
class BasketMark:
    as_of: str
    base_nav: Decimal
    market_value: Decimal
    nav: Decimal
    return_since_freeze: Decimal
    unresolved_weight: float
    missing_records: tuple[MissingPriceRecord, ...]


class FrozenWeightBasket:
    """信号日冻结成员的毛收益基准。"""

    def __init__(
        self,
        *,
        as_of: str,
        base_nav: Decimal,
        members: Mapping[str, _BasketMember],
        missing_ledger: MissingPriceLedger,
    ) -> None:
        self._as_of = normalize_date(as_of)
        self._base_nav = money(base_nav)
        self._members = dict(members)
        self._missing_ledger = missing_ledger
        self._last_mark_date = self._as_of
        self._last_mark: BasketMark | None = None

    @classmethod
    def freeze(
        cls,
        *,
        as_of: object,
        weights: Mapping[str, object],
        base_snapshot: MarketSnapshot,
        base_nav: object = 300_000,
        missing_ledger: MissingPriceLedger | None = None,
    ) -> "FrozenWeightBasket":
        """冻结成员与权重；基准日所有成员必须有 OK 价格。"""
        day = normalize_date(as_of)
        if normalize_date(base_snapshot.as_of) != day:
            raise InstrumentError(f"冻结日 {day} 与基准价格快照日 {base_snapshot.as_of} 不一致")
        nav = money(base_nav)
        if nav <= 0:
            raise InstrumentError(f"基准初始净值必须为正: {nav}")
        member_map: dict[str, _BasketMember] = {}
        total_weight = Decimal(0)
        for raw_code, raw_weight in weights.items():
            code = str(raw_code).strip().upper()
            if not code or code in member_map:
                raise InstrumentError(f"基准成员为空或重复: {raw_code!r}")
            weight = Decimal(str(raw_weight))
            if not weight.is_finite() or weight <= 0:
                raise InstrumentError(f"{code} 权重必须为有限正数: {raw_weight!r}")
            observation = base_snapshot.observation(code)
            if observation.status is not PriceStatus.OK or observation.value is None:
                raise InstrumentError(
                    f"基准冻结日 {code} 不是 OK 价格（{observation.status.value}），禁止猜价建基"
                )
            units = (weight * nav / observation.value).quantize(UNIT_SCALE, rounding=ROUND_HALF_UP)
            member_map[code] = _BasketMember(
                ts_code=code,
                weight=weight,
                units=units,
                last_price=observation.value,
                last_price_date=day,
            )
            total_weight += weight
        if abs(total_weight - Decimal(1)) > Decimal("1e-9"):
            raise InstrumentError(f"基准权重合计必须为 1，实际 {total_weight}")
        if not member_map:
            raise InstrumentError("基准不得为空")
        return cls(
            as_of=day,
            base_nav=nav,
            members=member_map,
            missing_ledger=missing_ledger or MissingPriceLedger(),
        )

    @property
    def as_of(self) -> str:
        return self._as_of

    @property
    def weights(self) -> Mapping[str, Decimal]:
        return MappingProxyType({code: item.weight for code, item in self._members.items()})

    @property
    def missing_ledger(self) -> MissingPriceLedger:
        return self._missing_ledger

    @property
    def snapshot(self) -> BasketMark:
        if self._last_mark is None:
            raise InstrumentError("基准尚未完成冻结日之后的估值")
        return self._last_mark

    def mark(self, snapshot: MarketSnapshot) -> BasketMark:
        """只消费当日快照；缺价保留最后有效份额，绝不按剩余成员重排权重。"""
        day = normalize_date(snapshot.as_of)
        if day <= self._last_mark_date:
            raise InstrumentError(f"基准估值日必须递增: {day} 不晚于 {self._last_mark_date}")
        market_value = ZERO
        pending: list[tuple[_BasketMember, PriceStatus, Decimal]] = []
        for code in sorted(self._members):
            member = self._members[code]
            observation = snapshot.observation(code)
            if observation.status is PriceStatus.OK:
                assert observation.value is not None
                member.last_price = observation.value
                member.last_price_date = day
                member.missing_streak_days = 0
                market_value += money(member.units * member.last_price)
                continue
            member.missing_streak_days += 1
            carried = money(member.units * member.last_price)
            market_value += carried
            pending.append((member, observation.status, carried))

        nav = money(market_value)
        missing_records: list[MissingPriceRecord] = []
        for member, reason, carried in pending:
            weight = float(money(carried / nav * 100) / 100) if nav > 0 else 0.0
            record = MissingPriceRecord(
                as_of=day,
                ts_code=member.ts_code,
                reason=reason,
                shares=None,
                carried_value=carried,
                last_price_date=member.last_price_date,
                days_since_last_quote=calendar_days_between(member.last_price_date, day),
                missing_streak_days=member.missing_streak_days,
                nav=nav,
                unresolved_weight=weight,
                source=snapshot.source,
            )
            stored, _ = self._missing_ledger.record(record)
            missing_records.append(stored)
        unresolved = (
            float(
                money(sum((item.carried_value for item in missing_records), ZERO) / nav * 100) / 100
            )
            if nav > 0
            else 0.0
        )
        self._last_mark_date = day
        self._last_mark = BasketMark(
            as_of=day,
            base_nav=self._base_nav,
            market_value=nav,
            nav=nav,
            return_since_freeze=nav / self._base_nav - Decimal(1),
            unresolved_weight=unresolved,
            missing_records=tuple(missing_records),
        )
        return self._last_mark
