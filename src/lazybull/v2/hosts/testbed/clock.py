# -*- coding: utf-8 -*-
"""P5b 虚拟时钟原语（D0 估值日 / D1 首个相邻收益）。

设计 v4 §4.1/§4.1.1/§5.1 的日期语义在此唯一成文：

- D0 只是**估值日**，不产生收益样本；
- D1 起才构造相邻日收益（``adjacent_return_pairs`` 从 D1 开始）；
- 虚拟日必须按交易日历**连续**，禁止跳日、禁止压缩缺失日、禁止填旧值。

评审处置：多日窗口**必须**提供交易日历并逐日核验连续（R1-T1-01 / R2-T1-10）；
单日窗口无相邻关系，标 ``vacuous_single_day``；日历身份（起止/条数/哈希）随
campaign 落盘，D0 与复核阶段据此复算，不再依赖调用方临时补参。

本模块只做日期序列校验与派生，不含任何 IO。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from src.lazybull.v2.hosts.testbed.paths import TestbedError
from src.lazybull.v2.store.store_identity import canonical_json_sha256

__all__ = [
    "CONTIGUITY_VACUOUS",
    "D0_ROLE",
    "VirtualClock",
    "build_virtual_clock",
    "calendar_identity",
    "normalize_day",
]

#: 虚拟日角色（D0 只估值，不参与收益样本）
D0_ROLE = "valuation_only"
#: 单日窗口的连续性依据（无相邻关系可校验）
CONTIGUITY_VACUOUS = "vacuous_single_day"


def normalize_day(value: Any, *, name: str = "virtual_day") -> str:
    """YYYYMMDD 形态与真实日期校验（fail-fast；不做宽松补零）。"""
    day = str(value)
    if len(day) != 8 or not day.isdigit():
        raise TestbedError(f"{name} 必须为 YYYYMMDD 字符串，实际: {value!r}")
    try:
        datetime.strptime(day, "%Y%m%d")
    except ValueError as exc:
        raise TestbedError(f"{name} 不是合法日期: {value!r}（{exc}）") from exc
    return day


def calendar_identity(calendar: Sequence[str]) -> Dict[str, Any]:
    """交易日历身份：起止、条数与整列哈希（落盘后供 D0/复核复算）。"""
    days = [normalize_day(day, name="calendar") for day in calendar]
    if not days:
        raise TestbedError("交易日历为空，禁止登记多日窗口")
    return {
        "first_day": days[0],
        "last_day": days[-1],
        "day_count": len(days),
        "calendar_sha256": canonical_json_sha256(days),
    }


@dataclass(frozen=True)
class VirtualClock:
    """虚拟时钟：历史交易日的连续序列（首个 = D0）。"""

    d0_date: str
    days: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.days:
            raise TestbedError("虚拟时钟至少需要 D0 一个交易日")
        if self.days[0] != self.d0_date:
            raise TestbedError(f"虚拟时钟首日必须等于 D0: days[0]={self.days[0]} d0={self.d0_date}")

    @property
    def first_return_date(self) -> Optional[str]:
        """第一个收益观测日（D1）；窗口只有 D0 时返回 None。"""
        return self.days[1] if len(self.days) > 1 else None

    @property
    def return_days(self) -> Tuple[str, ...]:
        """产生相邻日收益的日期（D1 起；D0 不在其中）。"""
        return self.days[1:]

    def adjacent_return_pairs(self) -> Tuple[Tuple[str, str], ...]:
        """相邻日配对 ``(t-1, t)``，首个配对从 D1 开始。"""
        return tuple(zip(self.days[:-1], self.days[1:]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d0_date": self.d0_date,
            "d0_role": D0_ROLE,
            "days": list(self.days),
            "n_return_days": len(self.return_days),
            "first_return_date": self.first_return_date,
            "return_days": list(self.return_days),
            "adjacent_return_pairs": [list(pair) for pair in self.adjacent_return_pairs()],
        }


def build_virtual_clock(
    *,
    d0_date: str,
    virtual_days: Optional[Sequence[str]] = None,
    calendar: Optional[Iterable[str]] = None,
    require_calendar: bool = True,
) -> VirtualClock:
    """构造并校验虚拟时钟。

    Args:
        d0_date: 历史 D0（估值日）。
        virtual_days: 已登记的虚拟日（含 D0）；缺省 = 只有 D0。
        calendar: 完整交易日历（YYYYMMDD）；提供时逐项校验连续性与边界。
        require_calendar: 多日窗口必须提供日历（"不跳日"不可静默省略）。

    Raises:
        TestbedError: 形态非法 / 首日不是 D0 / 重复或回退 / 跳日 / 日历外日期 /
            多日窗口缺日历。
    """
    d0 = normalize_day(d0_date, name="d0_date")
    raw_days = [normalize_day(day) for day in (virtual_days or [d0])]
    if raw_days[0] != d0:
        raise TestbedError(f"virtual_days 首日必须等于 d0_date: {raw_days[0]} != {d0}")
    if len(set(raw_days)) != len(raw_days):
        raise TestbedError(f"virtual_days 含重复日期: {raw_days}")
    if any(left >= right for left, right in zip(raw_days, raw_days[1:])):
        raise TestbedError(f"virtual_days 必须严格升序（禁止时钟回退）: {raw_days}")
    if calendar is None and require_calendar and len(raw_days) > 1:
        raise TestbedError(
            "多日虚拟窗口必须提供交易日历（禁止无条件接受稀疏日期）: " f"days={raw_days}"
        )
    clock = VirtualClock(d0_date=d0, days=tuple(raw_days))
    if calendar is not None:
        _assert_contiguous(clock, [normalize_day(day, name="calendar") for day in calendar])
    return clock


def _assert_contiguous(clock: VirtualClock, calendar: Sequence[str]) -> None:
    """虚拟日必须落在日历内且相邻两日之间无其他交易日（禁止跳日/压缩）。"""
    index = {day: position for position, day in enumerate(calendar)}
    missing = [day for day in clock.days if day not in index]
    if missing:
        raise TestbedError(f"虚拟日不在交易日历内: {missing}")
    for left, right in zip(clock.days, clock.days[1:]):
        gap = index[right] - index[left]
        if gap != 1:
            skipped = [day for day in calendar if index[left] < index[day] < index[right]]
            raise TestbedError(
                "虚拟日必须逐交易日连续（禁止跳日或压缩缺失日）: "
                f"{left} → {right} 跨 {gap} 个日历步，被跳过={skipped}"
            )
