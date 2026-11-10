"""状态信封五项之间的**经济一致性**校验（同一套经济事实，而不只是同一代）。

与 ``state_envelope`` 的"同代 marker"和 ``state_owners`` 的"身份键一致"互补：

- 同代 marker 只证明"这五项是同一代人写的"；
- 身份键一致只证明"幂等键/结算键/批次键互相对得上"；
- 本模块证明"股数与现金说的是同一笔钱、同一批股票"，且**当日已保存的日终估值**
  仍对应待提交的状态。

缺了本模块，给旧代视图重算外层哈希与 marker 后，就能与另一代的 P4/虚拟账拼成一个
"形式合法但经济上不可能"的信封——落盘后恢复起点即 I2/I1/I3 破缺
（P6.5 T5-Codex-C1 / R1-01）。

纯函数、零副作用：输入是各所有者解码后的**投影**，不引用任何所有者数据类，
避免与 ``state_owners`` 形成循环导入。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from src.lazybull.v2.core.accounting.reconciliation import (
    CASH_TOLERANCE,
    HOLDINGS_TOLERANCE_SHARES,
)
from src.lazybull.v2.core.accounting.state_envelope import STAGE, StateEnvelopeError

#: 来源股数键：``(ts_code, source_lot_id)``——与 I2 的比较粒度**完全一致**。
SourceKey = Tuple[str, str]


def _error(message: str) -> Exception:
    """带阶段上下文的一致性破缺（与所有者层同一异常族）。"""
    return StateEnvelopeError(f"阶段={STAGE} 状态信封所有者 {message}")


def _valuation_date_key(value: Any) -> str:
    """日期 → ``YYYYMMDD``（接受 Timestamp / ``YYYY-MM-DD`` / ``YYYYMMDD``）。

    命名避开 P2a-T5 退役符号 ``_date_key``（旧暴露 mixin 的方法名，v2/core 内禁出现）。
    """
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits[:8]


def position_shares(positions: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    """P4 聚合持仓 → ``{代码: 整数股}``（按代码汇总，用于成本口径对照）。"""
    shares: Dict[str, int] = {}
    for code, position in positions.items():
        shares[str(code)] = shares.get(str(code), 0) + int(position["shares"])
    return shares


def physical_source_shares(
    lots: Iterable[Sequence[Any]],
) -> Dict[SourceKey, int]:
    """物理来源 lot（``(代码, lot_id, 股数, 成本)``）→ ``{(代码, 来源): 整数股}``。

    R1-01 路径 A：只按代码汇总会把"同一代码在两个来源之间一增一减"的错配掩盖掉，
    因此必须按 I2 的 ``(ts_code, source_lot_id)`` 粒度对账。
    """
    shares: Dict[SourceKey, int] = {}
    for code, lot_id, lot_shares, _cost in lots:
        key = (str(code), str(lot_id))
        shares[key] = shares.get(key, 0) + int(lot_shares)
    return shares


def virtual_source_shares(
    books: Iterable[Sequence[Any]],
) -> Dict[SourceKey, int]:
    """虚拟账各袖 ``(袖, 现金, 占用, lots)`` → 取得批次 ``{(代码, 来源): 整数股}``。"""
    shares: Dict[SourceKey, int] = {}
    for _sleeve_id, _cash, _credit, lots in books:
        for lot in lots:
            key = (str(lot.ts_code), str(lot.source_lot_id))
            shares[key] = shares.get(key, 0) + int(lot.shares)
    return shares


def virtual_cash_and_credit(books: Iterable[Sequence[Any]]) -> Tuple[float, Tuple[str, ...]]:
    """虚拟账 → ``(Σ(现金+占用), 非有限的袖列表)``（非有限值不得进入比较）。"""
    total = 0.0
    bad: list[str] = []
    for sleeve_id, cash, credit, _lots in books:
        value = float(cash) + float(credit)
        if not math.isfinite(value):
            bad.append(str(sleeve_id))
            continue
        total += value
    return total, tuple(bad)


def _diff_keys(
    left: Mapping[SourceKey, int], right: Mapping[SourceKey, int]
) -> Tuple[Tuple[SourceKey, int, int], ...]:
    """两侧来源键的差异（含键集差异，股数按右侧缺省 0 补）。"""
    keys = set(left) | set(right)
    return tuple(
        (key, left.get(key, 0), right.get(key, 0))
        for key in sorted(keys)
        if left.get(key, 0) != right.get(key, 0)
    )


def _totals_by_code(shares: Mapping[SourceKey, int]) -> Dict[str, int]:
    """``{(代码, 来源): 股数}`` → ``{代码: 股数}``（仅用于给出更易懂的差异提示）。"""
    totals: Dict[str, int] = {}
    for (code, _lot), value in shares.items():
        totals[code] = totals.get(code, 0) + value
    return totals


def _find_daily_snapshot(
    portfolio_values: Iterable[Mapping[str, Any]], trade_date: str
) -> Mapping[str, Any]:
    """取当日日终估值快照（缺失即拒绝：已提交的信封必须带当日的日终事实）。"""
    target = _valuation_date_key(trade_date)
    for entry in portfolio_values:
        if _valuation_date_key(entry["date"]) == target:
            return entry
    raise _error(
        f"缺少交易日 {trade_date} 的日终估值快照（portfolio_values）；"
        "提交的信封必须带当日估值事实，禁止以邻日快照冒充"
    )


def assert_economic_consistency(
    *,
    label: str,
    trade_date: Any,
    positions: Mapping[str, Mapping[str, Any]],
    physical_lots: Iterable[Sequence[Any]],
    virtual_books: Iterable[Sequence[Any]],
    physical_cash: float,
    portfolio_values: Iterable[Mapping[str, Any]],
) -> None:
    """五项经济一致性 + 当日日终估值一致性。

    1. **来源粒度**（R1-01 路径 A）：物理来源视图与全袖虚拟来源批次按
       ``(ts_code, source_lot_id)`` 逐键相等（整数股 tol=0，键集一致）；
    2. **现金**：Σ(虚拟现金+占用) == P4 现金（I1 容差）；
    3. **日终估值**（R1-01 路径 B）：当日快照的 ``capital`` 必须等于待提交现金，
       且 ``portfolio_value == capital + market_value``（单字段陈旧即被绑定暴露）。

    Args:
        label: 定位串（信封代号 + 交易日）。
        trade_date: 信封交易日。
        positions: P4 聚合持仓。
        physical_lots: 物理伴随视图批次（含来源 lot_id）。
        virtual_books: 虚拟账各袖（含现金、占用与取得批次）。
        physical_cash: P4 现金。
        portfolio_values: P4 逐日估值历史（含当日快照）。

    Raises:
        StateEnvelopeError: 来源键/股数、现金或日终估值互不相符，或出现非有限值。
    """
    physical = physical_source_shares(physical_lots)
    virtual = virtual_source_shares(virtual_books)
    mismatch = _diff_keys(physical, virtual)
    if mismatch:
        detail = [
            f"{code}/{lot}: 物理={left} 虚拟={right}" for (code, lot), left, right in mismatch[:3]
        ]
        totals_physical = _totals_by_code(physical)
        totals_virtual = _totals_by_code(virtual)
        hint = (
            "（各代码总量相同、差异仅发生在来源之间：这正是只比代码汇总会漏掉的错配）"
            if totals_physical == totals_virtual
            else ""
        )
        raise _error(
            f"{label}：来源批次股数不一致 {detail}（共 {len(mismatch)} 处；"
            f"按 (ts_code, source_lot_id) 逐键比较，整数股 tol={HOLDINGS_TOLERANCE_SHARES}）{hint}"
        )
    by_code = position_shares(positions)
    lot_by_code: Dict[str, int] = {}
    for (code, _lot), shares in physical.items():
        lot_by_code[code] = lot_by_code.get(code, 0) + shares
    if by_code != lot_by_code:
        raise _error(
            f"{label}：P4 聚合持仓 {by_code} != 物理来源 lot 视图汇总 {lot_by_code}"
            f"（整数股 tol={HOLDINGS_TOLERANCE_SHARES}）"
        )
    virtual_cash, bad_sleeves = virtual_cash_and_credit(virtual_books)
    if bad_sleeves:
        raise _error(f"{label}：袖 {list(bad_sleeves)} 的虚拟现金/占用非有限")
    cash = float(physical_cash)
    if not math.isfinite(cash):
        raise _error(f"{label}：P4 现金非有限（{physical_cash!r}）")
    if abs(virtual_cash - cash) > CASH_TOLERANCE:
        raise _error(
            f"{label}：Σ虚拟现金+占用 {virtual_cash!r} 与 P4 现金 {cash!r} 不符"
            f"（差={virtual_cash - cash!r} 元，容差={CASH_TOLERANCE}）；"
            "校验通过之后被改写同样在此被拒"
        )
    snapshot = _find_daily_snapshot(portfolio_values, trade_date)
    fields = ("capital", "portfolio_value", "market_value")
    for field in fields:
        if field not in snapshot:
            raise _error(
                f"{label}：当日日终快照缺少字段 {field!r}（现有字段 {sorted(snapshot)}）；"
                "估值快照必须可被完整复核"
            )
    # R2-01：三项**先显式验有限再比较**——``abs(NaN - cash) > tol`` 恒为 False，
    # 不能由不等式隐式承担 NaN 拒绝（否则 NaN 现金可提交并恢复）。
    values = {field: float(snapshot[field]) for field in fields}
    bad = sorted(field for field, value in values.items() if not math.isfinite(value))
    if bad:
        raise _error(
            f"{label}：当日日终快照字段 {bad} 非有限"
            f"（capital={values['capital']!r} / portfolio_value={values['portfolio_value']!r} / "
            f"market_value={values['market_value']!r}）；不可核验的历史值不得落盘"
        )
    snapshot_capital = values["capital"]
    if abs(snapshot_capital - cash) > CASH_TOLERANCE:
        raise _error(
            f"{label}：当日日终快照 capital={snapshot_capital!r} 与待提交现金 {cash!r} 不符"
            f"（差={snapshot_capital - cash!r} 元）；日终事实不可静默陈旧或被重写"
        )
    total = values["portfolio_value"]
    market_value = values["market_value"]
    if abs((snapshot_capital + market_value) - total) > CASH_TOLERANCE:
        raise _error(
            f"{label}：当日日终快照不自洽 portfolio_value={total!r} != "
            f"capital={snapshot_capital!r} + market_value={market_value!r}"
            f"（差={(snapshot_capital + market_value) - total!r} 元）；"
            "单字段被改写会在恒等式上暴露"
        )


__all__ = (
    "assert_economic_consistency",
    "physical_source_shares",
    "position_shares",
    "virtual_cash_and_credit",
    "virtual_source_shares",
)
