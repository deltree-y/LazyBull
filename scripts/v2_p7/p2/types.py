# -*- coding: utf-8 -*-
"""P7 IDX-IN300 P2 共用计量工具：基础类型与确定性数值口径。

这一层只定义"什么算合法价格/日期/金额"，不读行情、不判断候选。
内部金额使用 Decimal 并统一到分，股数使用 int，避免 float 现金账漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class InstrumentError(ValueError):
    """P2 共用工具的结构性输入错误（调用方不得静默吞掉）。"""


class PriceStatus(str, Enum):
    """估值时点的价格可观测状态（互斥）。"""

    OK = "ok"
    SUSPENDED = "suspended"
    DELISTED = "delisted"
    NO_QUOTE = "no_quote"
    MISSING_ROW = "missing_row"


def normalize_date(value: object) -> str:
    """归一为 YYYYMMDD；只接受 8 位数字或带连字符的同一日期。"""
    text = str(value).strip().replace("-", "")
    if len(text) != 8 or not text.isdigit():
        raise InstrumentError(f"日期必须为 YYYYMMDD，实际 {value!r}")
    return text


def money(value: object) -> Decimal:
    """金额转为分精度 Decimal（四舍五入 HALF_UP；NaN/Inf 拒绝）。"""
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise InstrumentError(f"金额不可解析: {value!r}") from exc
    if not number.is_finite():
        raise InstrumentError(f"金额必须有限: {value!r}")
    return number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def price(value: object) -> Decimal:
    """价格转为 4 位小数 Decimal；必须有限且为正。"""
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise InstrumentError(f"价格不可解析: {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise InstrumentError(f"价格必须为有限正数: {value!r}")
    return number.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def share_count(value: object, *, field: str = "shares") -> int:
    """股数严格校验：非 bool 整数且非负。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InstrumentError(f"{field} 必须为 int，实际 {type(value).__name__}={value!r}")
    if value < 0:
        raise InstrumentError(f"{field} 不得为负: {value}")
    return value


@dataclass(frozen=True, slots=True)
class PriceObservation:
    """单个证券在单个估值日的价格观测。

    ``OK`` 必须有正价格；其余状态一律不得携带价格，防止调用方把
    "停牌/退市/缺行" 伪装成正常报价。估值只能消费该日快照。
    """

    status: PriceStatus
    value: Decimal | None = None
    source: str = "unspecified"

    def __post_init__(self) -> None:
        status = PriceStatus(self.status)
        object.__setattr__(self, "status", status)
        if not str(self.source).strip():
            raise InstrumentError("价格观测 source 不得为空")
        if status is PriceStatus.OK:
            if self.value is None:
                raise InstrumentError("OK 价格观测必须携带正价格")
            object.__setattr__(self, "value", price(self.value))
        elif self.value is not None:
            raise InstrumentError(f"{status.value} 状态不得携带价格（避免伪造可交易价）")


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """同一估值日的只读价格快照；不接受未来日或重复证券。"""

    as_of: str
    quotes: Mapping[str, PriceObservation]
    source: str = "unspecified"

    def __post_init__(self) -> None:
        as_of = normalize_date(self.as_of)
        object.__setattr__(self, "as_of", as_of)
        if not str(self.source).strip():
            raise InstrumentError("行情快照 source 不得为空")
        normalized: dict[str, PriceObservation] = {}
        for raw_code, observation in self.quotes.items():
            code = str(raw_code).strip().upper()
            if not code:
                raise InstrumentError("行情快照包含空证券代码")
            if code in normalized:
                raise InstrumentError(f"行情快照包含重复证券代码: {code}")
            if not isinstance(observation, PriceObservation):
                raise InstrumentError(f"{code} 的价格观测类型非法: {type(observation).__name__}")
            normalized[code] = observation
        object.__setattr__(self, "quotes", MappingProxyType(normalized))

    def observation(self, ts_code: str) -> PriceObservation:
        """取单证券观测；缺失按 MISSING_ROW 返回，不猜价格。"""
        return self.quotes.get(
            str(ts_code).strip().upper(),
            PriceObservation(PriceStatus.MISSING_ROW, source=self.source),
        )
