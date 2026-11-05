# -*- coding: utf-8 -*-
"""v2 纸面估值口径的纯函数族（v0.209.9）。

两类物件的分工是硬契约：

- **逐业务日收盘估值** = ``engine_state.portfolio_values`` 的语义（NAV 曲线源；末日行 =
  「最后处理日」当日成交后收盘价），其业务日行必须与 ``nav.parquet`` **同日行**同值。
- **某一行情日的当前市值** = 只读旁路 ``state/valuation_preview.json``（本模块
  :func:`build_valuation_preview`），显式带 ``market_date`` / ``authority_last_date``，
  **禁止**写进 ``portfolio_values`` 或 ``nav.parquet``。

v0.209.9 缺陷背景：迁移工具把行情日 ``20261029`` 的收盘价写进了业务日 ``20260603`` 的
``portfolio_values`` 行，两载体差 98,754.00（现金/持仓一致，纯估值时点错位）。
:func:`nav_projection_divergence` 是该类缺陷的启动期出口（``PaperStateStore``
``check_consistency`` 第三腿），按业务日对齐后逐字段比对，**只暴露不修复**。

本模块为纯函数（无 I/O、无状态），拆分自 ``state_store.py`` 以守 1200 行硬上限。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from src.lazybull.v2.hosts.paper.instruction_projection import _yyyymmdd

__all__ = [
    "NAV_VALUE_TOLERANCE",
    "VALUATION_PREVIEW_FILENAME",
    "NavProjectionDayMissing",
    "build_valuation_preview",
    "nav_projection_divergence",
    "same_value",
]

#: 当前市值只读旁路文件名（显式带 market_date；**永不**回流 portfolio_values/NAV 曲线）
VALUATION_PREVIEW_FILENAME = "valuation_preview.json"
#: nav 投影 ↔ 权威同日行的数值比对容差（cash / position_value / total_value）
NAV_VALUE_TOLERANCE = 1e-6


class NavProjectionDayMissing(RuntimeError):
    """nav.parquet 末行业务日在权威快照 ``portfolio_values`` 中不存在（结构性冲突）。"""


def build_valuation_preview(
    *,
    market_date: Any,
    cash: float,
    market_value: float,
    mark_prices: Mapping[str, float],
    source: str,
    authority_last_date: Optional[Any] = None,
) -> Dict[str, Any]:
    """「当前市值」只读旁路投影（纯函数）：显式绑定行情日，**不进** NAV 曲线。"""
    return {
        "market_date": _yyyymmdd(market_date),
        "authority_last_date": _yyyymmdd(authority_last_date) or None,
        "source": str(source),
        "cash": float(cash),
        "market_value": float(market_value),
        "total_value": float(cash) + float(market_value),
        "mark_prices": {str(code): float(price) for code, price in sorted(mark_prices.items())},
    }


def same_value(left: Any, right: Any) -> bool:
    """两路数值在 :data:`NAV_VALUE_TOLERANCE` 内相等（缺失/非数值/NaN ⇒ False）。"""
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    if a != a or b != b:  # NaN：任一侧不可比即视为不一致
        return False
    return abs(a - b) <= NAV_VALUE_TOLERANCE


def nav_projection_divergence(
    portfolio_values: Sequence[Mapping[str, Any]],
    nav_row: Mapping[str, Any],
) -> Optional[str]:
    """``nav.parquet`` 末行 ↔ 权威**同日**行的逐字段差异描述（一致 ⇒ None）。

    跳空语义（nav 仅在有成交日追加、``portfolio_values`` 每处理日追加）⇒ nav 末行日期
    **可以早于**权威末日；此时按该日期在 ``portfolio_values`` 中找同日行比对。找不到同日
    行 = 结构性冲突，抛 :class:`NavProjectionDayMissing`。数值不一致只返回描述、不修复：
    两侧都可能承载已发布事实，静默改写任一侧都会伪造历史。
    """
    day = _yyyymmdd(nav_row.get("trade_date"))
    authority = {
        _yyyymmdd(item.get("date")): item for item in portfolio_values if isinstance(item, Mapping)
    }
    expected = authority.get(day)
    if expected is None:
        raise NavProjectionDayMissing(
            f"nav.parquet 末行业务日 {day} 在权威快照 portfolio_values 中不存在"
            "（两载体业务日集合不同，禁止静默处置；请核查迁移/回滚后的状态）"
        )
    diffs = [
        f"{label} nav={left!r} 权威={right!r}"
        for label, left, right in (
            ("cash", nav_row.get("cash"), expected.get("capital")),
            ("position_value", nav_row.get("position_value"), expected.get("market_value")),
            ("total_value", nav_row.get("total_value"), expected.get("portfolio_value")),
        )
        if not same_value(left, right)
    ]
    return "；".join(diffs) if diffs else None
