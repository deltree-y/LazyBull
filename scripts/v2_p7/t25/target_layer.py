# -*- coding: utf-8 -*-
"""P7 T2.5 目标层：内核同源选择 + 两档错峰相位 + 跨折连续持仓账。

硬约束
======
- **选择逻辑不重写**：tranche 槽位数、持仓排除、候选截断全部委托内核
  ``v2/core/execution/signal_execution``（``_filter_signal_candidates`` /
  ``_resolve_signal_target_n``）与 ``v2/core/decision/params.get_tranche_target_count``；
  本模块只提供它们需要的宿主属性（``signal`` / ``max_per_industry`` /
  ``_post_filter_candidates`` / ``_get_tranche_target_count``），不改 ``src/lazybull/**``；
- **分数层同源**：全排名由 ``MLSignal.generate_ranked`` 产出（与回测引擎同一实现），
  不使用锚批次的 ``topk_detail``（其缺失）或 trades；
- **相位只移日历**：V10 与 V20 共用同一分数、Top-N、20 日持期与两档拆分，只错后 10 个交易日；
- **跨折连续账户**：同一袖的持仓/lot 在折界不清零，模型版本只在各折 ``test_start`` 信号点切换。

时点口径（与引擎同日序一致）
============================
引擎单日次序 = 信号生成 → 到期卖出 → 买入执行。因此：

- 信号日 ``T`` 的「已持仓」= ``T-1`` 收盘后的持仓（当日到期但尚未卖出的仍计入排除集）；
- 买在 ``T+1`` 开盘，20 个交易日后（``T+21`` 开盘）到期卖出；
- 本模块的持仓为**目标级计划持仓**（假设 T+1 全额成交），不是实际成交账；
  真实成交价源、部分成交与残差归 T4/T5。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from scripts.v2_p7.t25.params import (
    HOLDING_DAYS,
    LOT_SIZE,
    STAGGER_TRANCHES,
    TOP_N,
    FoldRow,
    T25Params,
)

PLAN_COLUMNS: Tuple[str, ...] = (
    "arm",
    "signal_date",
    "execution_date",
    "due_date",
    "tranche_idx",
    "planned_rank",
    "ts_code",
    "score",
    "weight",
    "budget",
    "split_index",
    "model_version",
)

#: 冻结 sizing（锚批次 ``batch_meta.json/config``：half_kelly + 0.15 单股上限 + 60 日波动窗 + 0.2 杠杆上限）
SIZING_POSITION: str = "half_kelly"
SIZING_MAX_WEIGHT_PER_STOCK: float = 0.15
SIZING_KELLY_VOL_WINDOW: int = 60
SIZING_KELLY_MAX_LEVERAGE: float = 0.2

HOLDING_COLUMNS: Tuple[str, ...] = (
    "arm",
    "trade_date",
    "ts_code",
    "signal_date",
    "buy_date",
    "due_date",
    "origin_model_version",
    "current_model_version",
    "held_days",
    "shares",
    "buy_price",
)

DAILY_COLUMNS: Tuple[str, ...] = (
    "arm",
    "trade_date",
    "split_index",
    "model_version",
    "tranche_idx",
    "target_n",
    "candidate_count",
    "held_count",
    "new_count",
    "closed_count",
    "cash_open",
    "sell_proceeds",
    "buy_need",
    "target_notional",
    "mv_hold",
    "raw_demand",
    "funding_gap",
)


class TargetLayerError(RuntimeError):
    """目标层前置条件不满足（身份/日历/内核契约）。"""


class _TopNPort:
    """内核 ``_resolve_signal_target_n`` 只读取 ``signal.top_n``；此处显式传入冻结值。"""

    def __init__(self, top_n: int) -> None:
        self.top_n = int(top_n)


class PlannerKernel:
    """内核选择逻辑的只读宿主适配（不实现任何选择/截断语义）。

    仅提供 ``signal_execution`` 两个纯函数读取的宿主属性；``max_per_industry``
    保持 ``None``（锚批次 ``bt_max_per_industry`` 为空），因此行业约束分支不进入。
    """

    def __init__(self, top_n: int = TOP_N, stagger_tranches: int = STAGGER_TRANCHES) -> None:
        self.signal = _TopNPort(top_n)
        self.max_per_industry: Optional[int] = None
        self.verbose = False
        self._top_n = int(top_n)
        self._tranche_count = int(stagger_tranches)

    def _post_filter_candidates(self, ranked_candidates: list, date: pd.Timestamp) -> list:
        """与内核 ML 宿主同义（``v2/core/execution/ml_signal_feed.py`` 透传）。"""
        return ranked_candidates

    def _get_tranche_target_count(
        self, tranche_idx: int, target_count: Optional[int] = None
    ) -> int:
        """委托内核 ``v2/core/decision/params.get_tranche_target_count``。"""
        from src.lazybull.v2.core.decision.params import (
            get_tranche_target_count as kernel_tranche_target_count,
        )

        return kernel_tranche_target_count(
            tranche_idx,
            self._tranche_count,
            self.signal,
            {},
            target_count,
        )


def select_targets(
    kernel: PlannerKernel,
    ranked_candidates: Sequence[Tuple[str, float]],
    signal_date: str,
    tranche_idx: int,
    held: Sequence[str],
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """按内核逻辑从全排名中选出本档目标（排除当前持仓，同股不加仓）。

    Args:
        kernel: 内核适配宿主
        ranked_candidates: 当日全排名 ``[(ts_code, score), ...]``（``generate_ranked`` 输出）
        signal_date: 信号日（YYYYMMDD）
        tranche_idx: 批次索引（0-based）
        held: 信号时点的当前持仓代码

    Returns:
        ``(picks, info)``；``picks`` 为按内核顺序截断后的目标列表
    """
    from src.lazybull.v2.core.execution.signal_execution import (
        _filter_signal_candidates,
        _resolve_signal_target_n,
    )

    date = pd.Timestamp(signal_date)
    ranked, for_selection = _filter_signal_candidates(
        kernel, list(ranked_candidates), date, set(held)
    )
    target_n, overall_target_n = _resolve_signal_target_n(kernel, ranked, int(tranche_idx))
    picks = list(for_selection[: max(0, int(target_n))])
    info = {
        "target_n": int(target_n),
        "overall_target_n": int(overall_target_n),
        "candidate_count": len(ranked),
        "excluded_held": len(ranked) - len(for_selection),
    }
    return picks, info


@dataclass
class Lot:
    """目标级计划持仓批次（不是物理 lot，只用于计划账与模型版本登记）。"""

    ts_code: str
    signal_date: str
    buy_date: str
    buy_index: int
    due_index: int
    due_date: str
    shares: int
    buy_price: float
    origin_model_version: int
    budget: float = 0.0


@dataclass
class ArmResult:
    """单袖运行结果（计划帧 + 逐日持仓 + 逐日资金诊断）。"""

    arm: str
    plans: pd.DataFrame
    holdings: pd.DataFrame
    daily: pd.DataFrame
    integrity: Dict[str, Any] = field(default_factory=dict)


def _lot_shares(budget: float, price: float) -> int:
    """整手估算股数（100 股取整；无价或非正价返回 0）。"""
    if not price or price <= 0:
        return 0
    return int(budget // (price * LOT_SIZE)) * LOT_SIZE


class VarianceBook:
    """逐股 ``close_adj`` 序列（按日历索引），供内核 half_kelly 方差估计使用。

    单次顺序扫描全日历建表（两袖共用），``variance(code, day)`` 取该日及之前
    ``SIZING_KELLY_VOL_WINDOW`` 个点交给内核 ``estimate_variance_from_prices``。
    """

    def __init__(self, calendar: Sequence[str], loader: Any) -> None:
        import numpy as np

        self._calendar = list(calendar)
        self._index = {day: pos for pos, day in enumerate(self._calendar)}
        self._rows: Dict[str, Any] = {}
        # 「有行情行」与「行内价格为 NaN」必须分开：内核按**行数**过门槛，再由估计器过滤无效价
        self._present: Dict[str, Any] = {}
        n_days = len(self._calendar)
        for pos, day in enumerate(self._calendar):
            frame = loader(day)
            if frame is None or len(frame) == 0:
                continue
            for row in frame.itertuples(index=False):
                code = str(row.ts_code)
                series = self._rows.get(code)
                if series is None:
                    series = np.full(n_days, np.nan, dtype=np.float64)
                    self._rows[code] = series
                    self._present[code] = np.zeros(n_days, dtype=bool)
                series[pos] = float(row.close_adj)
                self._present[code][pos] = True
        self._np = np

    @property
    def codes(self) -> int:
        return len(self._rows)

    def variance(self, code: str, day: str) -> Optional[float]:
        """逐股取样语义与内核 ``estimate_stock_variance`` 一致。

        内核：先筛出该股在截止日前的**实际行情行**，再取最近 ``window`` 行。
        故此处先剔除缺失（该股当日无行情），再取尾部，而不是先截日历槽再看缺失。
        """
        from src.lazybull.v2.core.decision.sizing import estimate_variance_from_prices

        series = self._rows.get(str(code))
        present = self._present.get(str(code))
        pos = self._index.get(day)
        if series is None or present is None or pos is None:
            return None
        # 与内核同序：先按**实际行情行**计数并取末 window 行，行内无效价交给共享估计器过滤
        rows = self._np.flatnonzero(present[: pos + 1])[-SIZING_KELLY_VOL_WINDOW:]
        if len(rows) < 20:  # 与内核最小样本（行数）一致
            return None
        return estimate_variance_from_prices(series[rows])


def sleeve_weights(
    signals: Mapping[str, float],
    signal_day: str,
    variance_book: Optional[VarianceBook],
) -> Dict[str, float]:
    """按冻结 sizing 计算本批逐股**权重**（与引擎 ``_normalize_and_cap_signals`` 同序）。

    时点纪律：权重在**信号日 T** 定死（方差截止 T、样本 ``<= T``），不得读 T+1 任何行情；
    执行日只用开盘时已知的现金与开盘价换算股数。

    顺序：``compute_kelly_weights``（half_kelly，杠杆上限 0.2）→ ``cap_and_normalize_weights``。
    """
    from src.lazybull.v2.core.decision.sizing import compute_kelly_weights
    from src.lazybull.v2.core.decision.weight_processor import (
        cap_and_normalize_weights,
        resolve_tranche_weight_cap,
    )

    if not signals:
        return {}

    def _variance(code: str) -> Optional[float]:
        return variance_book.variance(code, signal_day) if variance_book else None

    if SIZING_POSITION == "half_kelly":
        raw, _fallback = compute_kelly_weights(
            dict(signals),
            _variance,
            half=True,
            max_leverage=SIZING_KELLY_MAX_LEVERAGE,
        )
    else:  # pragma: no cover - 冻结口径只有 half_kelly
        raise TargetLayerError(f"未支持的 sizing：{SIZING_POSITION}")
    cap = resolve_tranche_weight_cap(SIZING_MAX_WEIGHT_PER_STOCK, 1.0 / STAGGER_TRANCHES)
    capped = cap_and_normalize_weights(raw, max_weight_per_stock=cap, verbose=False)
    return {str(code): float(weight) for code, weight in capped.items()}


def _signal_step(
    *,
    arm: str,
    day: str,
    pos: int,
    calendar: Sequence[str],
    tranche_idx: Optional[int],
    kernel: PlannerKernel,
    ranked_lookup: Any,
    held_before: Sequence[str],
    plans: List[Dict[str, Any]],
    pending: Dict[int, List[Dict[str, Any]]],
    split_index: int,
    model_version: int,
    empty_ranking_days: List[str],
    variance_book: Optional[VarianceBook] = None,
) -> Optional[Dict[str, Any]]:
    """单日信号步：内核选择 → 计划行 → 次日待执行队列。"""
    if tranche_idx is None:
        return None
    picks, info = select_targets(
        kernel, ranked_lookup(day), day, int(tranche_idx), sorted(held_before)
    )
    if not picks:
        # 内核当日拒绝预测（特征质量门禁）⇒ 无候选、不建计划；显式登记不做补数
        empty_ranking_days.append(day)
    if picks and pos + 1 >= len(calendar):
        raise TargetLayerError(f"{arm} 末个再平衡日 {day} 之后无交易日，无法执行 T+1")
    if not picks:
        return info
    execution_date = calendar[pos + 1] if pos + 1 < len(calendar) else ""
    due_pos = pos + 1 + HOLDING_DAYS
    due_date = calendar[due_pos] if due_pos < len(calendar) else ""
    # 权重在**信号日 T** 定死（方差截止 T），执行日只做现金/开盘价换算，禁止读 T+1 行情
    weights = sleeve_weights({str(code): float(score) for code, score in picks}, day, variance_book)
    for rank, (code, score) in enumerate(picks, start=1):
        entry = {
            "arm": arm,
            "signal_date": day,
            "execution_date": execution_date,
            "due_date": due_date,
            "tranche_idx": int(tranche_idx),
            "planned_rank": rank,
            "ts_code": str(code),
            "score": float(score),
            "weight": float(weights.get(str(code), 0.0)),
            "split_index": int(split_index),
            "model_version": int(model_version),
        }
        plans.append(entry)
        pending.setdefault(pos + 1, []).append(entry)
    return info


def _settle_trades(
    *,
    day: str,
    pos: int,
    calendar: Sequence[str],
    book: Dict[str, Lot],
    due_index: Dict[int, List[str]],
    pending: Dict[int, List[Dict[str, Any]]],
    price_lookup: Any,
    cash_open: float,
    top_n: int,
    stagger_tranches: int,
    variance_book: Optional[VarianceBook],
) -> Tuple[float, float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """单日成交步：先按到期卖出，再按冻结 sizing（half_kelly）计算预算并买入。

    预算口径与引擎一致：NAV（成交当日市值 + 现金）× 本批资金比例 × 批内权重；
    权重 = ``compute_kelly_weights(half_kelly)`` → ``cap_and_normalize_weights``（0.15 上限）。
    """
    sell_proceeds = 0.0
    for code in list(due_index.get(pos, [])):
        lot = book.pop(code, None)
        if lot is None:
            continue
        price = _open_price(price_lookup, day, code, lot.buy_price)
        sell_proceeds += lot.shares * price

    new_entries = pending.pop(pos, [])
    if not new_entries:
        return sell_proceeds, 0.0, new_entries, []

    # NAV 只用**开盘时已知**的量：现金 + 当日卖出回款 + 存量持仓按当日**开盘价**计值
    mv_hold = 0.0
    for lot in book.values():
        mv_hold += lot.shares * _open_price(price_lookup, day, lot.ts_code, lot.buy_price)
    nav = float(cash_open) + sell_proceeds + mv_hold

    tranche_idx = int(new_entries[0]["tranche_idx"])
    from src.lazybull.v2.common.rules.stagger import get_tranche_capital_fraction

    fraction = get_tranche_capital_fraction(tranche_idx, int(top_n), int(stagger_tranches))
    batch_budget = nav * fraction

    buy_need = 0.0
    unpriced: List[Dict[str, Any]] = []
    for entry in new_entries:
        code = str(entry["ts_code"])
        budget = float(entry.get("weight") or 0.0) * batch_budget
        entry["budget"] = budget
        price = _open_price(price_lookup, day, code, 0.0)
        shares = _lot_shares(budget, price)
        if shares <= 0:
            # 开盘不可用 ⇒ 不买入（不用当日 close 伪装开盘价），显式登记为未定价
            entry["shares"] = 0
            unpriced.append({"trade_date": day, "ts_code": code, "reason": "no_valid_open"})
            continue
        buy_need += shares * price
        lot = Lot(
            ts_code=code,
            signal_date=entry["signal_date"],
            buy_date=day,
            buy_index=pos,
            due_index=pos + HOLDING_DAYS,
            due_date=(calendar[pos + HOLDING_DAYS] if pos + HOLDING_DAYS < len(calendar) else ""),
            shares=shares,
            buy_price=price,
            origin_model_version=int(entry["model_version"]),
            budget=budget,
        )
        book[lot.ts_code] = lot
        due_index.setdefault(lot.due_index, []).append(lot.ts_code)
        entry["shares"] = shares
        entry["buy_price"] = price
    return sell_proceeds, buy_need, new_entries, unpriced


def _holding_rows(
    arm: str, day: str, pos: int, book: Dict[str, Lot], model_version: int
) -> List[Dict[str, Any]]:
    """单日持仓登记行（含 ``origin/current_model_version`` 双版本）。"""
    rows: List[Dict[str, Any]] = []
    for code in sorted(book):
        lot = book[code]
        rows.append(
            {
                "arm": arm,
                "trade_date": day,
                "ts_code": code,
                "signal_date": lot.signal_date,
                "buy_date": lot.buy_date,
                "due_date": lot.due_date,
                "origin_model_version": int(lot.origin_model_version),
                "current_model_version": int(model_version),
                "held_days": pos - lot.buy_index,
                "shares": int(lot.shares),
                "buy_price": float(lot.buy_price),
            }
        )
    return rows


def _daily_row(
    *,
    arm: str,
    day: str,
    pos: int,
    split_index: int,
    model_version: int,
    tranche_idx: Optional[int],
    planned_info: Optional[Dict[str, Any]],
    held_before: Sequence[str],
    new_entries: Sequence[Dict[str, Any]],
    due_index: Dict[int, List[str]],
    cash_open: float,
    sell_proceeds: float,
    buy_need: float,
    book: Dict[str, Lot],
    price_lookup: Any,
) -> Dict[str, Any]:
    """单日资金/目标诊断行（T/MV/D 同取**成交后**同一时点）。

    目标槽位 = 成交前持仓 − 当日到期平仓 + 当日新买；市值 = 成交后 book 的当日开盘市值。
    两者同一时点，``raw_demand = max(0, G − MV)`` 才有意义（旧版混用成交前后，
    换仓日会把目标算成 30 槽、普通持仓日又把市值置零）。
    """
    # 目标名义 = 成交后各持仓的**目标预算**之和（冻结 sizing 下逐股预算不同，不能用 槽位×常数）
    target_notional = float(sum(lot.budget for lot in book.values()))
    mv_hold = sum(
        lot.shares * _open_price(price_lookup, day, lot.ts_code, lot.buy_price)
        for lot in book.values()
    )
    return {
        "arm": arm,
        "trade_date": day,
        "split_index": int(split_index),
        "model_version": int(model_version),
        "tranche_idx": -1 if tranche_idx is None else int(tranche_idx),
        "target_n": -1 if planned_info is None else int(planned_info["target_n"]),
        "candidate_count": (-1 if planned_info is None else int(planned_info["candidate_count"])),
        "held_count": len(held_before),
        "new_count": len(new_entries),
        "closed_count": len(due_index.get(pos, [])),
        "cash_open": float(cash_open),
        "sell_proceeds": float(sell_proceeds),
        "buy_need": float(buy_need),
        "target_notional": target_notional,
        "mv_hold": float(mv_hold),
        "raw_demand": float(max(0.0, target_notional - mv_hold)),
        # 缺口只在有买入意向的日子定义（无买入日不因现金为负而虚增）
        "funding_gap": float(max(0.0, buy_need - cash_open)) if buy_need > 0 else 0.0,
    }


def simulate_arm(
    arm: str,
    calendar: Sequence[str],
    schedule: Mapping[str, int],
    ranked_lookup: Any,
    price_lookup: Any,
    params: T25Params,
    capital: float,
    top_n: int = TOP_N,
    stagger_tranches: int = STAGGER_TRANCHES,
    variance_book: Optional[VarianceBook] = None,
) -> ArmResult:
    """单袖目标级账户模拟（跨折连续；折界不清零）。

    Args:
        arm: ``v20`` / ``v10``（或扫描用的候选名）
        calendar: 全信号域交易日（升序，YYYYMMDD）
        schedule: 本袖再平衡排期 ``{signal_date: tranche_idx}``
        ranked_lookup: ``day -> [(ts_code, score), ...]`` 全排名提供者（内核产出）
        price_lookup: ``day -> {ts_code: (open, close)}`` 估计价提供者
        params: 冻结参数（用于折/模型版本映射与 runoff 边界）
        capital: 袖虚拟初资
        top_n: 本袖总目标持仓数（默认冻结值；只读扫描可覆盖）
        stagger_tranches: 本袖分批档数（默认冻结值；只读扫描可覆盖）

    Returns:
        :class:`ArmResult`
    """
    kernel = PlannerKernel(top_n, stagger_tranches)

    book: Dict[str, Lot] = {}
    cash = float(capital)
    plans: List[Dict[str, Any]] = []
    holdings: List[Dict[str, Any]] = []
    daily: List[Dict[str, Any]] = []
    pending: Dict[int, List[Dict[str, Any]]] = {}
    due_index: Dict[int, List[str]] = {}
    empty_ranking_days: List[str] = []
    unpriced_buys: List[Dict[str, Any]] = []

    for pos, day in enumerate(calendar):
        model_version = params.current_model_version(day)
        split_index = params.current_split_index(day)
        held_before = set(book)
        cash_open = cash
        tranche_idx = schedule.get(day)

        # ── 信号步（先于当日成交，与引擎同日序一致）──
        planned_info = _signal_step(
            arm=arm,
            day=day,
            pos=pos,
            calendar=calendar,
            tranche_idx=tranche_idx,
            kernel=kernel,
            ranked_lookup=ranked_lookup,
            held_before=held_before,
            plans=plans,
            pending=pending,
            split_index=split_index,
            model_version=model_version,
            empty_ranking_days=empty_ranking_days,
            variance_book=variance_book,
        )

        # ── 成交步：先卖后买 ──
        sell_proceeds, buy_need, new_entries, unpriced_today = _settle_trades(
            day=day,
            pos=pos,
            calendar=calendar,
            book=book,
            due_index=due_index,
            pending=pending,
            price_lookup=price_lookup,
            cash_open=cash,
            top_n=top_n,
            stagger_tranches=stagger_tranches,
            variance_book=variance_book,
        )
        cash += sell_proceeds - buy_need
        unpriced_buys.extend(unpriced_today)

        holdings.extend(_holding_rows(arm, day, pos, book, model_version))
        daily.append(
            _daily_row(
                arm=arm,
                day=day,
                pos=pos,
                split_index=split_index,
                model_version=model_version,
                tranche_idx=tranche_idx,
                planned_info=planned_info,
                held_before=held_before,
                new_entries=new_entries,
                due_index=due_index,
                cash_open=cash_open,
                sell_proceeds=sell_proceeds,
                buy_need=buy_need,
                book=book,
                price_lookup=price_lookup,
            )
        )

    integrity = {
        "arm": arm,
        "rebalances": len(schedule),
        "plans": len(plans),
        "unpriced_buys": len(unpriced_buys),
        "unpriced_buy_examples": unpriced_buys[:5],
        "empty_ranking_days": sorted(empty_ranking_days),
        "holdings_rows": len(holdings),
        "final_positions": len(book),
        "open_obligations_at_end": int(
            sum(1 for lot in book.values() if lot.due_index >= len(calendar))
        ),
        "cash_end": float(cash),
        "min_cash": float(min(row["cash_open"] for row in daily)) if daily else 0.0,
    }
    return ArmResult(
        arm=arm,
        plans=pd.DataFrame(plans, columns=list(PLAN_COLUMNS) + ["shares", "buy_price"]),
        holdings=pd.DataFrame(holdings, columns=list(HOLDING_COLUMNS)),
        daily=pd.DataFrame(daily, columns=list(DAILY_COLUMNS)),
        integrity=integrity,
    )


def _open_price(price_lookup: Any, day: str, code: str, fallback: float) -> float:
    """T+1 开盘价的估计值；**开盘不可用时绝不回退当日收盘**。

    因果纪律：开盘那一刻当日 close 尚不可见，故无有效 open 时只用
    开盘时已知的最后价格（存量持仓的买入价）或 0（新买 → 股数 0，由调用方登记为未定价）。
    """
    quote = price_lookup(day) or {}
    pair = quote.get(code)
    if not pair:
        return float(fallback)
    price = float(pair[0] or 0.0)
    if price > 0:
        return price
    return float(fallback)


def cross_model_stats(holdings: pd.DataFrame, params: T25Params) -> Dict[str, Any]:
    """折界持仓占比与跨模型持有期占比（逐袖）。

    折界日 = 各折首个交易日（``test_start`` 信号点，模型在此切换）。
    """
    if holdings.empty:
        return {"arms": {}, "boundaries": []}
    arms: Dict[str, Any] = {}
    boundaries: List[Dict[str, Any]] = []
    for arm, frame in holdings.groupby("arm"):
        total_days = int(len(frame))
        cross_days = int((frame["origin_model_version"] != frame["current_model_version"]).sum())
        arms[str(arm)] = {
            "holding_days": total_days,
            "cross_model_holding_days": cross_days,
            "cross_model_holding_share": _share(cross_days, total_days),
            "positions": int(frame["ts_code"].nunique()),
        }
    for fold in params.folds:
        day = fold.signal_days[0]
        snapshot = holdings[holdings["trade_date"] == day]
        for arm, frame in snapshot.groupby("arm"):
            total = int(len(frame))
            cross = int((frame["origin_model_version"] != frame["current_model_version"]).sum())
            boundaries.append(
                {
                    "split_index": fold.split_index,
                    "trade_date": day,
                    "arm": str(arm),
                    "model_version": int(fold.model_version),
                    "positions": total,
                    "cross_model_positions": cross,
                    "cross_model_share": _share(cross, total),
                }
            )
    return {"arms": arms, "boundaries": boundaries}


def _share(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def target_overlap(plans: pd.DataFrame) -> Dict[str, Any]:
    """两袖同日目标集合重叠率（Jaccard），以及同日双向（一买一卖）股数。"""
    if plans.empty:
        return {"days": 0}
    pivoted: Dict[str, Dict[str, set]] = {}
    for row in plans.itertuples(index=False):
        pivoted.setdefault(str(row.signal_date), {}).setdefault(str(row.arm), set()).add(
            str(row.ts_code)
        )
    overlaps: List[float] = []
    for day, per_arm in sorted(pivoted.items()):
        left = per_arm.get("v20", set())
        right = per_arm.get("v10", set())
        if not left or not right:
            continue
        union = left | right
        overlaps.append(float(len(left & right)) / float(len(union)) if union else 0.0)
    return {
        "days": len(overlaps),
        "mean_jaccard": float(sum(overlaps) / len(overlaps)) if overlaps else None,
        "min_jaccard": float(min(overlaps)) if overlaps else None,
        "max_jaccard": float(max(overlaps)) if overlaps else None,
    }


def holdings_overlap(holdings: pd.DataFrame) -> Dict[str, Any]:
    """两袖逐日持仓集合重叠率（Jaccard）。"""
    if holdings.empty:
        return {"days": 0}
    per_day: Dict[str, Dict[str, set]] = {}
    for row in holdings.itertuples(index=False):
        per_day.setdefault(str(row.trade_date), {}).setdefault(str(row.arm), set()).add(
            str(row.ts_code)
        )
    values: List[float] = []
    for _, per_arm in sorted(per_day.items()):
        left = per_arm.get("v20", set())
        right = per_arm.get("v10", set())
        if not left and not right:
            continue
        union = left | right
        values.append(float(len(left & right)) / float(len(union)) if union else 0.0)
    return {
        "days": len(values),
        "mean_jaccard": float(sum(values) / len(values)) if values else None,
        "min_jaccard": float(min(values)) if values else None,
        "max_jaccard": float(max(values)) if values else None,
        "zero_overlap_days": int(sum(1 for value in values if value == 0.0)),
    }


def fold_boundary_continuity(
    plans: pd.DataFrame, holdings: pd.DataFrame, params: T25Params
) -> List[Dict[str, Any]]:
    """折界持仓承接核对：折首日持仓必须全部来自折前信号（不清零、不重建）。"""
    checks: List[Dict[str, Any]] = []
    for fold in params.folds:
        first_day = fold.signal_days[0]
        snapshot = holdings[holdings["trade_date"] == first_day]
        for arm, frame in snapshot.groupby("arm"):
            carried = int((frame["signal_date"].astype(str) < first_day).sum())
            checks.append(
                {
                    "split_index": fold.split_index,
                    "arm": str(arm),
                    "trade_date": first_day,
                    "positions": int(len(frame)),
                    "carried_from_previous_fold": carried,
                    "opened_in_fold": int(len(frame)) - carried,
                }
            )
    return checks


def plan_integrity(plans: pd.DataFrame, params: T25Params) -> Dict[str, Any]:
    """目标层自检：每档槽位数、T+1 执行、20 日到期、模型版本归属。"""
    if plans.empty:
        raise TargetLayerError("目标层为空，禁止产出")
    calendar = list(params.extended_days)
    index_of = {day: pos for pos, day in enumerate(calendar)}
    per_tranche: Dict[str, int] = {}
    bad_execution = 0
    bad_due = 0
    bad_model = 0
    for row in plans.itertuples(index=False):
        signal_pos = index_of[str(row.signal_date)]
        execution_pos = index_of.get(str(row.execution_date))
        if execution_pos != signal_pos + 1:
            bad_execution += 1
        if str(row.due_date):
            due_pos = index_of.get(str(row.due_date))
            if due_pos != execution_pos + HOLDING_DAYS:
                bad_due += 1
        if int(row.model_version) != params.model_version_of_day(str(row.signal_date)):
            bad_model += 1
        key = f"{row.arm}:{row.tranche_idx}"
        per_tranche[key] = max(per_tranche.get(key, 0), int(row.planned_rank))
    if bad_execution or bad_due or bad_model:
        raise TargetLayerError(
            f"目标层时点/模型归属破缺：execution={bad_execution} due={bad_due} model={bad_model}"
        )
    return {
        "rows": int(len(plans)),
        "days": int(plans["signal_date"].nunique()),
        "max_rank_by_arm_tranche": dict(sorted(per_tranche.items())),
        "execution_is_t_plus_1": True,
        "due_is_execution_plus_holding": True,
        "model_version_switch_at_test_start": True,
    }


def runoff_obligations(result: ArmResult, params: T25Params) -> Dict[str, Any]:
    """末折 20 日 runoff 输入清单（未决义务显式右删失，禁止折界清零）。

    口径：末折最后一个信号日之后的 20 个交易日只执行存量到期/残差/召回/估值；
    本 task 只登记**输入**（未决 lot 清单），不执行 runoff 成交。
    """
    last_signal = max(result.plans["signal_date"].astype(str))
    open_lots = result.holdings[(result.holdings["trade_date"] == params.signal_days[-1])].copy()
    obligations: List[Dict[str, Any]] = []
    for row in open_lots.itertuples(index=False):
        due_date = str(row.due_date)
        within = due_date in set(params.runoff_days)
        obligations.append(
            {
                "arm": result.arm,
                "ts_code": str(row.ts_code),
                "signal_date": str(row.signal_date),
                "buy_date": str(row.buy_date),
                "due_date": due_date,
                "origin_model_version": int(row.origin_model_version),
                "due_in_runoff": bool(within),
                "right_censored": bool(not within),
            }
        )
    return {
        "arm": result.arm,
        "last_signal_date": last_signal,
        "runoff_window": [params.runoff_days[0], params.runoff_days[-1]],
        "runoff_days": len(params.runoff_days),
        "obligations": sorted(obligations, key=lambda item: (item["due_date"], item["ts_code"])),
        "right_censored_count": int(sum(1 for item in obligations if item["right_censored"])),
        "note": "未决义务只登记输入，不在本 task 执行成交；右删失不得记为自然归还。",
    }


def model_switch_points(params: T25Params) -> List[Dict[str, Any]]:
    """模型切换点清单（只发生在各折 ``test_start`` 信号点）。"""
    points: List[Dict[str, Any]] = []
    for fold in params.folds:
        points.append(
            {
                "split_index": fold.split_index,
                "switch_at": fold.signal_days[0],
                "model_version": fold.model_version,
                "previous_model_version": (
                    params.folds[fold.split_index - 1].model_version
                    if fold.split_index > 0
                    else None
                ),
            }
        )
    return points


def fold_day_assignment(params: T25Params) -> Dict[str, FoldRow]:
    """信号日 → 折映射（唯一性由 :func:`params.build_params` 保证）。"""
    mapping: Dict[str, FoldRow] = {}
    for fold in params.folds:
        for day in fold.signal_days:
            if day in mapping:
                raise TargetLayerError(f"信号日重复归属两折：{day}")
            mapping[day] = fold
    return mapping
