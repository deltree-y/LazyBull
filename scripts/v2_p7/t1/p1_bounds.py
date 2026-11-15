# -*- coding: utf-8 -*-
"""P-1 重开核心：完整路径枚举、同期净残存包络、日历轴抽样与判定。

口径（v0.5 定稿）
================
- 路径集合 ``P_i`` 覆盖每个入场分支直到真实退出或研究截止；退出**不设 60/90 日上限**；
- 逐路径 ``d_i(p) = r_event_net(i,p) - r_bench_gross(i,p)``（同一 e→x 窗口两腿）；
- ``lo_i = inf d_i(p)``、``hi_i = sup d_i(p)``；任一未覆盖分支 ⇒ 该意图 ``unbounded``；
- 聚合 ``D_min = (K + Σ min(0,lo_i))/N``、``D_max = (K + Σ max(0,hi_i))/N``；
- 抽样为日历轴 stationary bootstrap（主）+ 定长圆形块（辅），区间取包络。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from scripts.v2_p7.t1 import execution
from scripts.v2_p7.t1.p1_inputs import BenchmarkBook, Extras
from scripts.v2_p7.t1.p1_params import P1RParams

TRI_NOT_TRADABLE = 0
TRI_TRADABLE = 1
TRI_UNKNOWN = -1

FILL_CONFIRMED = 0
FILL_PROVEN_NOT = 1
FILL_UNKNOWN = 2
FILL_POSSIBLE = 3
FILL_FILLED = FILL_CONFIRMED  # 兼容旧引用；语义见 _fill_status

VALUE_EXACT = 0
VALUE_FALLBACK = 1
VALUE_CHAIN_BROKEN = 2
VALUE_MISSING = 3

COVERAGE_FULL = 0
COVERAGE_THIN = 1
COVERAGE_UNAVAILABLE = 2

HOLDING_NA = 2
HOLDING_CLOSED = 0
HOLDING_OPEN = 1


@dataclass
class P1PathsResult:
    """逐意图路径界与主路径（供 O1 / 产物 / 抽样）。

    持仓状态只对 ``fill=FILLED`` 有意义；未成交一律 ``HOLDING_NA``，
    不得伪作 CLOSED 或 OPEN（r4 收口项）。
    """

    d_point: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    unbounded: np.ndarray
    holding: np.ndarray
    fill: np.ndarray
    value: np.ndarray
    coverage: np.ndarray
    entry_pos: np.ndarray
    exit_pos: np.ndarray
    entry_defer: np.ndarray
    exit_defer: np.ndarray
    paths_per_intent: np.ndarray
    event_gross: np.ndarray
    event_net: np.ndarray
    bench_gross: np.ndarray
    buy_cost: np.ndarray
    sell_cost: np.ndarray
    bench_members: np.ndarray
    bench_unvalued: np.ndarray
    bench_fallback: np.ndarray
    reason: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def _tri_buy(panel: Any, extras: Extras, day: int, stock: int) -> int:
    """入场三态：可买 / 有证据不可买 / 证据未知。"""
    row = bool(extras.has_row[day, stock])
    price = float(panel.open_u[day, stock])
    if (
        row
        and np.isfinite(price)
        and price > 0
        and not panel.suspended[day, stock]
        and not panel.limit_up[day, stock]
    ):
        return TRI_TRADABLE
    if extras.suspend_flag[day, stock] or panel.suspended[day, stock] or panel.limit_up[day, stock]:
        return TRI_NOT_TRADABLE
    return TRI_UNKNOWN


def _tri_sell(panel: Any, extras: Extras, day: int, stock: int) -> int:
    """退出三态：可卖 / 有证据不可卖 / 证据未知。"""
    row = bool(extras.has_row[day, stock])
    price = float(panel.open_u[day, stock])
    if (
        row
        and np.isfinite(price)
        and price > 0
        and not panel.suspended[day, stock]
        and not panel.limit_down[day, stock]
    ):
        return TRI_TRADABLE
    if (
        extras.suspend_flag[day, stock]
        or panel.suspended[day, stock]
        or panel.limit_down[day, stock]
    ):
        return TRI_NOT_TRADABLE
    return TRI_UNKNOWN


def _entry_branches(
    panel: Any, extras: Extras, signal_pos: int, stock: int, max_defer: int
) -> tuple[List[Dict], bool, bool]:
    """入场分支：可买即停；未知日保留“买入/继续”两分支。

    返回 ``(branches, not_filled, saw_unknown)``——``not_filled`` 只表示“没有可买的
    证明日”，``saw_unknown`` 才区分“未成交是证实的还是未知的”。
    """
    branches: List[Dict] = []
    saw_unknown = False
    last = min(panel.n_days - 1, signal_pos + max_defer)
    for day in range(signal_pos + 1, last + 1):
        state = _tri_buy(panel, extras, day, stock)
        if state == TRI_TRADABLE:
            branches.append({"entry": day, "entry_unknown": False})
            return branches, False, saw_unknown
        if state == TRI_UNKNOWN:
            saw_unknown = True
            branches.append({"entry": day, "entry_unknown": True})
    return branches, True, saw_unknown


def _exit_branches(
    panel: Any, extras: Extras, expiry: int, stock: int, cutoff_pos: int
) -> List[Dict]:
    """退出分支：到期后首个可卖日；未知日保留分支；截止未平仓单独一条。"""
    branches: List[Dict] = []
    for day in range(expiry, cutoff_pos + 1):
        state = _tri_sell(panel, extras, day, stock)
        if state == TRI_TRADABLE:
            branches.append({"exit": day, "exit_unknown": False, "cutoff": False})
            return branches
        if state == TRI_UNKNOWN:
            branches.append({"exit": day, "exit_unknown": True, "cutoff": False})
    branches.append({"exit": cutoff_pos, "exit_unknown": False, "cutoff": True})
    return branches


class _RatioBook:
    """复权因子比（open_adj/open_u）与最后可得索引；缺行日沿用最近有效比率。"""

    def __init__(self, panel: Any) -> None:
        open_u = np.asarray(panel.open_u)
        open_adj = np.asarray(panel.open_adj)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(
                np.isfinite(open_u) & (open_u > 0) & np.isfinite(open_adj) & (open_adj > 0),
                open_adj / open_u,
                np.nan,
            )
        valid = np.isfinite(ratio) & (ratio > 0)
        index = np.arange(panel.n_days, dtype=np.int32)[:, None]
        last = np.where(valid, index, -1).astype(np.int32)
        np.maximum.accumulate(last, axis=0, out=last)
        self.ratio = ratio
        self.last = last

    def at(self, day: int, stock: int) -> float:
        idx = int(self.last[day, stock])
        if idx < 0:
            return float("nan")
        return float(self.ratio[idx, stock])


def _limit_price(extras: Extras, day: int, stock: int, *, side: str, upper: bool) -> float:
    """涨跌停价端点；缺失返回 NaN。"""
    matrix = extras.up_limit if upper else extras.down_limit
    value = float(matrix[day, stock])
    if not np.isfinite(value) or value <= 0:
        return float("nan")
    return value


def _buy_candidates(
    panel: Any,
    extras: Extras,
    ratios: _RatioBook,
    stock: int,
    entry: int,
    entry_unknown: bool,
) -> tuple[List[tuple[float, float]], str]:
    """买入端点候选（未知日取涨跌停价 × 复权因子）；返回 (候选, 失败原因)。"""
    price_buy = float(panel.open_u[entry, stock])
    adj_buy = float(panel.open_adj[entry, stock])
    if not entry_unknown:
        if not (np.isfinite(price_buy) and price_buy > 0 and np.isfinite(adj_buy) and adj_buy > 0):
            return [], "entry_price_missing"
        return [(price_buy, adj_buy)], "ok"
    up = _limit_price(extras, entry, stock, side="buy", upper=True)
    down = _limit_price(extras, entry, stock, side="buy", upper=False)
    if not (np.isfinite(up) and np.isfinite(down)):
        return [], "entry_price_missing"
    factor = ratios.at(entry, stock)
    if not np.isfinite(factor):
        return [], "entry_adj_factor_missing"
    return [(up, up * factor), (down, down * factor)], "ok"


def _sell_candidates(
    panel: Any,
    extras: Extras,
    ratios: _RatioBook,
    last_valid: np.ndarray,
    stock: int,
    exit_: int,
    exit_unknown: bool,
    cutoff: bool,
) -> tuple[List[tuple[float, float]], int, str]:
    """卖出端点候选；返回 (候选, value_evidence, 失败原因)。"""
    if cutoff:
        last_idx = int(last_valid[exit_, stock])
        if last_idx < 0:
            return [], VALUE_MISSING, "cutoff_price_missing"
        adj_sell = float(panel.close_adj[last_idx, stock])
        if not (np.isfinite(adj_sell) and adj_sell > 0):
            return [], VALUE_MISSING, "cutoff_price_missing"
        flag = VALUE_EXACT if last_idx == exit_ else VALUE_FALLBACK
        return [(adj_sell, adj_sell)], flag, "ok"
    if not exit_unknown:
        price_sell = float(panel.open_u[exit_, stock])
        adj_sell = float(panel.open_adj[exit_, stock])
        if not (
            np.isfinite(price_sell) and price_sell > 0 and np.isfinite(adj_sell) and adj_sell > 0
        ):
            return [], VALUE_MISSING, "exit_price_missing"
        return [(price_sell, adj_sell)], VALUE_EXACT, "ok"
    up = _limit_price(extras, exit_, stock, side="sell", upper=True)
    down = _limit_price(extras, exit_, stock, side="sell", upper=False)
    if not (np.isfinite(up) and np.isfinite(down)):
        return [], VALUE_MISSING, "exit_price_missing"
    factor = ratios.at(exit_, stock)
    if not np.isfinite(factor):
        return [], VALUE_MISSING, "exit_adj_factor_missing"
    return [(up, up * factor), (down, down * factor)], VALUE_FALLBACK, "ok"


def _net_at_endpoints(
    params: P1RParams,
    ratio_share: float,
    buy_candidates: List[tuple[float, float]],
    sell_candidates: List[tuple[float, float]],
    *,
    cutoff: bool,
) -> List[float]:
    """端点组合的事件腿净收益（截止估值按复权经济价值扣一次卖出费用）。"""
    nets: List[float] = []
    for buy_price, buy_adj in buy_candidates:
        for sell_price, sell_adj in sell_candidates:
            leg = execution.single_leg_net(
                buy_price,
                buy_adj,
                sell_price,
                sell_adj,
                ratio_share,
                True,
                cost=params.cost.as_dict(),
                min_ticket=params.min_ticket,
                lot_size=params.lot_size,
            )
            if not leg.valid:
                continue
            if cutoff:
                net = execution.single_leg_net_from_value(
                    leg.buy_amount * (sell_adj / buy_adj),
                    leg.buy_amount,
                    leg.buy_cost,
                    cost=params.cost.as_dict(),
                )
            else:
                net = leg.net
            if np.isfinite(net):
                nets.append(float(net))
    return nets


def _event_bounds_for_path(
    panel: Any,
    extras: Extras,
    ratios: _RatioBook,
    last_valid: np.ndarray,
    params: P1RParams,
    stock: int,
    path: Dict,
) -> tuple[float, float, int, str]:
    """单条路径的事件腿净收益界（含未知日涨跌停端点）；返回 (lo, hi, value_evidence, reason)。"""
    entry = int(path["entry"])
    exit_ = int(path["exit"])
    cutoff = bool(path.get("cutoff", False))
    ratio_share = float(
        execution.share_ratio(panel, np.array([entry]), np.array([exit_]), np.array([stock]))[0]
    )
    chain_ok = bool(
        execution.share_chain_known(panel, np.array([entry]), np.array([exit_]), np.array([stock]))[
            0
        ]
    )
    if not chain_ok or not np.isfinite(ratio_share) or ratio_share <= 0:
        return float("nan"), float("nan"), VALUE_CHAIN_BROKEN, "share_chain_broken"

    buy_candidates, buy_reason = _buy_candidates(
        panel, extras, ratios, stock, entry, bool(path["entry_unknown"])
    )
    if not buy_candidates:
        return float("nan"), float("nan"), VALUE_MISSING, buy_reason
    sell_candidates, value_flag, sell_reason = _sell_candidates(
        panel,
        extras,
        ratios,
        last_valid,
        stock,
        exit_,
        bool(path["exit_unknown"]),
        cutoff,
    )
    if not sell_candidates:
        return float("nan"), float("nan"), VALUE_MISSING, sell_reason
    nets = _net_at_endpoints(params, ratio_share, buy_candidates, sell_candidates, cutoff=cutoff)
    if not nets:
        return float("nan"), float("nan"), VALUE_MISSING, "leg_not_valuable"
    return min(nets), max(nets), value_flag, "ok"


def _build_paths(
    panel: Any,
    extras: Extras,
    signal: int,
    stock: int,
    horizon: int,
    entry_max_defer: int,
    cutoff: int,
) -> tuple[List[Dict], List[Dict], bool, bool]:
    """构建某意图的全部完整路径（入场分支 × 退出分支 + 未成交分支）。"""
    entries, not_filled, saw_unknown = _entry_branches(
        panel, extras, signal, stock, entry_max_defer
    )
    paths: List[Dict] = []
    for entry_branch in entries:
        expiry = int(entry_branch["entry"]) + horizon
        for exit_branch in _exit_branches(panel, extras, expiry, stock, cutoff):
            paths.append({**entry_branch, **exit_branch})
    if not_filled:
        paths.append({"not_filled": True, "entry": signal + 1, "exit": signal + 1})
    return paths, entries, not_filled, saw_unknown


def _path_envelope(
    panel: Any,
    extras: Extras,
    ratios: _RatioBook,
    benchmark: BenchmarkBook,
    params: P1RParams,
    stock: int,
    signal: int,
    industry: int,
    paths: Sequence[Dict],
) -> tuple[Optional[float], Optional[float], int, int, Optional[str]]:
    """逐路径求事件腿界与基准腿，返回 (lo, hi, value_flag, coverage_flag, 失败原因)。"""
    low, high = float("inf"), float("-inf")
    value_flag = VALUE_EXACT
    cover_flag = COVERAGE_FULL
    for path in paths:
        if path.get("not_filled"):
            low, high = min(low, 0.0), max(high, 0.0)
            continue
        event_low, event_high, flag, reason = _event_bounds_for_path(
            panel,
            extras,
            ratios,
            benchmark.last_valid,
            params,
            stock,
            path,
        )
        if not (np.isfinite(event_low) and np.isfinite(event_high)):
            return None, None, VALUE_MISSING, COVERAGE_UNAVAILABLE, reason
        bench = benchmark.return_window(
            signal,
            industry,
            stock,
            int(path["entry"]),
            int(path["exit"]),
            exit_mode="close" if path.get("cutoff") else "open",
        )
        if bench["coverage"] == "unavailable" or not np.isfinite(bench["ret"]):
            return None, None, VALUE_MISSING, COVERAGE_UNAVAILABLE, "benchmark_unavailable"
        low = min(low, event_low - float(bench["ret"]))
        high = max(high, event_high - float(bench["ret"]))
        value_flag = max(value_flag, int(flag))
        cover_flag = max(cover_flag, COVERAGE_THIN if bench["coverage"] == "thin" else 0)
    if not (np.isfinite(low) and np.isfinite(high)):
        return None, None, VALUE_MISSING, COVERAGE_UNAVAILABLE, "no_path_value"
    return low, high, value_flag, cover_flag, None


def _proven_entry(entries: Sequence[Dict]) -> Optional[Dict]:
    """主路径入场：只返回已证明可买日；只有未知日买入分支时返回 None。"""
    return next((item for item in entries if not item["entry_unknown"]), None)


def _fill_status(entries: Sequence[Dict], saw_unknown: bool) -> int:
    """成交证据：确认成交 / 可能成交 / 证明未成交 / 未知。"""
    if _proven_entry(entries) is not None:
        return FILL_CONFIRMED
    if entries:
        return FILL_POSSIBLE
    return FILL_UNKNOWN if saw_unknown else FILL_PROVEN_NOT


def _main_exit(
    panel: Any,
    extras: Extras,
    entry: int,
    horizon: int,
    stock: int,
    cutoff: int,
) -> tuple[int, int, int]:
    """主路径退出：首个证明可卖日；否则截止未平仓。返回 (exit_pos, holding, defer)。"""
    expiry = entry + horizon
    exits = _exit_branches(panel, extras, expiry, stock, cutoff)
    proven = next((item for item in exits if not item["exit_unknown"] and not item["cutoff"]), None)
    if proven is None:
        item = exits[-1]
        return int(item["exit"]), HOLDING_OPEN, int(item["exit"]) - expiry
    return int(proven["exit"]), HOLDING_CLOSED, int(proven["exit"]) - expiry


def _main_path_values(
    panel: Any,
    extras: Extras,
    ratios: _RatioBook,
    benchmark: BenchmarkBook,
    params: P1RParams,
    stock: int,
    signal: int,
    industry: int,
    entry: int,
    exit_: int,
    cutoff: bool,
) -> Optional[Dict[str, Any]]:
    """主路径两腿读数：事件毛/净、买卖费用、基准毛与成员覆盖（供归因表）。"""
    buy, _ = _buy_candidates(panel, extras, ratios, stock, entry, False)
    sell, value_flag, _ = _sell_candidates(
        panel, extras, ratios, benchmark.last_valid, stock, exit_, False, cutoff
    )
    if not buy or not sell:
        return None
    ratio_share = float(
        execution.share_ratio(panel, np.array([entry]), np.array([exit_]), np.array([stock]))[0]
    )
    chain_ok = bool(
        execution.share_chain_known(panel, np.array([entry]), np.array([exit_]), np.array([stock]))[
            0
        ]
    )
    if not chain_ok or not np.isfinite(ratio_share) or ratio_share <= 0:
        return None
    buy_price, buy_adj = buy[0]
    sell_price, sell_adj = sell[0]
    leg = execution.single_leg_net(
        buy_price,
        buy_adj,
        sell_price,
        sell_adj,
        ratio_share,
        True,
        cost=params.cost.as_dict(),
        min_ticket=params.min_ticket,
        lot_size=params.lot_size,
    )
    if not leg.valid:
        return None
    if cutoff:
        value_exit = leg.buy_amount * (sell_adj / buy_adj)
        net = execution.single_leg_net_from_value(
            value_exit, leg.buy_amount, leg.buy_cost, cost=params.cost.as_dict()
        )
        sell_cost = float(execution._sell_cost(np.asarray([value_exit]), params.cost.as_dict())[0])
        gross = value_exit / leg.buy_amount - 1.0
    else:
        net = float(leg.net)
        gross = leg.value_exit / leg.buy_amount - 1.0
        sell_cost = float(leg.sell_cost)
    bench = benchmark.return_window(
        signal,
        industry,
        stock,
        entry,
        exit_,
        exit_mode="close" if cutoff else "open",
    )
    return {
        "event_gross": float(gross),
        "event_net": float(net),
        "bench_gross": float(bench["ret"]),
        "buy_cost": float(leg.buy_cost),
        "sell_cost": sell_cost,
        "bench_members": int(bench["members"]),
        "bench_unvalued": int(bench["unvalued"]),
        "bench_fallback": int(bench["max_fallback"]),
        "value_flag": int(value_flag),
    }


def _evaluate_one_intent(
    panel: Any,
    extras: Extras,
    ratios: _RatioBook,
    benchmark: BenchmarkBook,
    params: P1RParams,
    intents: Any,
    index: int,
    industry_matrix: np.ndarray,
    horizon: int,
    cutoff: int,
    max_paths: int,
) -> Dict[str, Any]:
    """单意图：路径枚举 → 包络 → 主路径与两腿；返回可写回数组的字典。"""
    stock = int(intents.stock[index])
    signal = int(intents.signal_pos[index])
    industry = int(industry_matrix[signal, stock])
    paths, entries, _, saw_unknown = _build_paths(
        panel, extras, signal, stock, horizon, params.entry_max_defer, cutoff
    )
    outcome: Dict[str, Any] = {
        "paths": len(paths),
        "unbounded": False,
        "reason": None,
        "lo": float("nan"),
        "hi": float("nan"),
        "d_point": float("nan"),
        "value": VALUE_MISSING,
        "coverage": COVERAGE_UNAVAILABLE,
        "fill": FILL_UNKNOWN,
        "holding": HOLDING_NA,
        "entry_pos": -1,
        "exit_pos": -1,
        "entry_defer": -1,
        "exit_defer": -1,
        "legs": None,
    }
    if not paths or len(paths) > max_paths:
        outcome["unbounded"] = True
        outcome["reason"] = "path_enumeration_exceeded" if paths else "no_path"
        outcome["fill"] = _fill_status(entries, saw_unknown)
        return outcome
    low, high, value_flag, cover_flag, reason = _path_envelope(
        panel, extras, ratios, benchmark, params, stock, signal, industry, paths
    )
    if low is None or high is None:
        outcome["unbounded"] = True
        outcome["reason"] = str(reason)
        outcome["fill"] = _fill_status(entries, saw_unknown)
        return outcome
    outcome.update(
        {
            "lo": low,
            "hi": high,
            "d_point": low if (high - low) < 1e-15 else 0.5 * (low + high),
            "value": value_flag,
            "coverage": cover_flag,
        }
    )
    proven = _proven_entry(entries)
    if proven is None:
        outcome["fill"] = _fill_status(entries, saw_unknown)
        return outcome
    entry = int(proven["entry"])
    exit_pos, holding, exit_defer = _main_exit(panel, extras, entry, horizon, stock, cutoff)
    outcome.update(
        {
            "fill": FILL_FILLED,
            "holding": holding,
            "entry_pos": entry,
            "exit_pos": exit_pos,
            "entry_defer": entry - (signal + 1),
            "exit_defer": exit_defer,
        }
    )
    outcome["legs"] = _main_path_values(
        panel,
        extras,
        ratios,
        benchmark,
        params,
        stock,
        signal,
        industry,
        entry,
        exit_pos,
        holding == HOLDING_OPEN,
    )
    return outcome


def evaluate_intents(
    panel: Any,
    extras: Extras,
    benchmark: BenchmarkBook,
    params: P1RParams,
    intents: Any,
    *,
    max_paths: int = 64,
) -> P1PathsResult:
    """逐意图枚举完整路径并求同期净残存界（含未知日端点与未成交分支）。"""
    n = intents.size
    d_point = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    unbounded = np.zeros(n, dtype=bool)
    holding = np.full(n, HOLDING_NA, dtype=np.int8)
    fill = np.full(n, FILL_UNKNOWN, dtype=np.int8)
    value = np.full(n, VALUE_MISSING, dtype=np.int8)
    coverage = np.full(n, COVERAGE_UNAVAILABLE, dtype=np.int8)
    entry_pos = np.full(n, -1, dtype=np.int32)
    exit_pos = np.full(n, -1, dtype=np.int32)
    entry_defer = np.full(n, -1, dtype=np.int16)
    exit_defer = np.full(n, -1, dtype=np.int16)
    paths_per_intent = np.zeros(n, dtype=np.int16)
    event_gross = np.full(n, np.nan)
    event_net = np.full(n, np.nan)
    bench_gross = np.full(n, np.nan)
    buy_cost = np.full(n, np.nan)
    sell_cost = np.full(n, np.nan)
    bench_members = np.full(n, -1, dtype=np.int32)
    bench_unvalued = np.full(n, -1, dtype=np.int32)
    bench_fallback = np.full(n, -1, dtype=np.int32)
    reasons: List[str] = []
    ratios = _RatioBook(panel)
    cutoff = panel.n_days - 1
    horizon = int(params.horizon_primary)
    industry_matrix = np.asarray(panel.industry_matrix)

    for index in range(n):
        item = _evaluate_one_intent(
            panel,
            extras,
            ratios,
            benchmark,
            params,
            intents,
            index,
            industry_matrix,
            horizon,
            cutoff,
            max_paths,
        )
        paths_per_intent[index] = int(item["paths"])
        if item["unbounded"]:
            unbounded[index] = True
            reasons.append(str(item["reason"]))
        else:
            lo[index] = item["lo"]
            hi[index] = item["hi"]
            d_point[index] = item["d_point"]
            value[index] = item["value"]
            coverage[index] = item["coverage"]
        fill[index] = item["fill"]
        holding[index] = item["holding"]
        if item["entry_pos"] >= 0:
            entry_pos[index] = item["entry_pos"]
            exit_pos[index] = item["exit_pos"]
            entry_defer[index] = item["entry_defer"]
            exit_defer[index] = item["exit_defer"]
        legs = item["legs"]
        if legs is not None:
            event_gross[index] = legs["event_gross"]
            event_net[index] = legs["event_net"]
            bench_gross[index] = legs["bench_gross"]
            buy_cost[index] = legs["buy_cost"]
            sell_cost[index] = legs["sell_cost"]
            bench_members[index] = legs["bench_members"]
            bench_unvalued[index] = legs["bench_unvalued"]
            bench_fallback[index] = legs["bench_fallback"]

    confirmed = fill == FILL_CONFIRMED
    possible = fill == FILL_POSSIBLE
    stats = {
        "intent_total": int(n),
        "filled_confirmed": int(confirmed.sum()),
        "filled_possible": int(possible.sum()),
        "filled_closed": int((confirmed & (holding == HOLDING_CLOSED)).sum()),
        "filled_open": int((confirmed & (holding == HOLDING_OPEN)).sum()),
        "filled_na": int((confirmed & (holding == HOLDING_NA)).sum()),
        "proven_not_filled": int((fill == FILL_PROVEN_NOT).sum()),
        "unknown_not_filled": int((fill == FILL_UNKNOWN).sum()),
        "holding_na": int((holding == HOLDING_NA).sum()),
        "bounded": int(np.isfinite(lo).sum()),
        "unbounded": int(unbounded.sum()),
        "paths_mean": float(paths_per_intent.mean()) if n else 0.0,
        "paths_p95": float(np.percentile(paths_per_intent, 95)) if n else 0.0,
        "paths_max": int(paths_per_intent.max()) if n else 0,
        "reasons": _reason_counts(reasons),
    }
    stats["conservation_ok"] = bool(
        stats["intent_total"]
        == stats["filled_confirmed"]
        + stats["filled_possible"]
        + stats["proven_not_filled"]
        + stats["unknown_not_filled"]
        and stats["filled_confirmed"]
        == stats["filled_closed"] + stats["filled_open"] + stats["filled_na"]
        and stats["holding_na"]
        == stats["filled_na"]
        + stats["filled_possible"]
        + stats["proven_not_filled"]
        + stats["unknown_not_filled"]
    )
    return P1PathsResult(
        d_point=d_point,
        lo=lo,
        hi=hi,
        unbounded=unbounded,
        holding=holding,
        fill=fill,
        value=value,
        coverage=coverage,
        entry_pos=entry_pos,
        exit_pos=exit_pos,
        entry_defer=entry_defer,
        exit_defer=exit_defer,
        paths_per_intent=paths_per_intent,
        event_gross=event_gross,
        event_net=event_net,
        bench_gross=bench_gross,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
        bench_members=bench_members,
        bench_unvalued=bench_unvalued,
        bench_fallback=bench_fallback,
        reason=reasons,
        stats=stats,
    )


def _reason_counts(reasons: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for reason in reasons:
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def aggregate_envelope(result: P1PathsResult) -> Dict[str, Any]:
    """D_min/D_max：已知意图进 K，未知意图逐笔取 min(0,lo)/max(0,hi)。

    任一意图 ``unbounded`` ⇒ 整个 D 记不可约束（``constrained=False``、d_min/d_max=None），
    不得用 bounded 子集冒充整体包络。
    """
    n = int(result.d_point.size)
    bounded = np.isfinite(result.lo) & np.isfinite(result.hi) & (~result.unbounded)
    stats: Dict[str, Any] = {
        "intent_total": n,
        "bounded": int(bounded.sum()),
        "unbounded": int(result.unbounded.sum()),
        "constrained": bool(n > 0 and not result.unbounded.any() and bounded.any()),
    }
    if not stats["constrained"]:
        stats.update(
            {
                "d_min": None,
                "d_max": None,
                "known_intents": int(bounded.sum()),
                "unknown_intents": 0,
                "known_sum": None,
            }
        )
        return stats
    known = bounded & ((result.hi - result.lo) < 1e-15)
    unknown = bounded & (~known)
    k_known = float(result.d_point[known].sum())
    d_min = (k_known + float(np.minimum(0.0, result.lo[unknown]).sum())) / n
    d_max = (k_known + float(np.maximum(0.0, result.hi[unknown]).sum())) / n
    stats.update(
        {
            "d_min": float(d_min),
            "d_max": float(d_max),
            "known_intents": int(known.sum()),
            "unknown_intents": int(unknown.sum()),
            "known_sum": k_known,
        }
    )
    return stats


def _circular_block_index(
    *, rng: np.random.Generator, size: int, days: int, block: int
) -> np.ndarray:
    """定长圆形块索引（每行拼接 ceil(days/block) 个环回块，截断到 days）。"""
    if block < 1 or days < 1:
        raise ValueError(f"块长/天数非法：block={block} days={days}")
    n_blocks = int(np.ceil(days / block))
    starts = rng.integers(0, days, size=(size, n_blocks))
    offsets = np.arange(block, dtype=np.int64)[None, None, :]
    index = (starts[:, :, None] + offsets) % days
    return index.reshape(size, -1)[:, :days]


def _bootstrap_values(
    day_pos: np.ndarray,
    values: np.ndarray,
    n_days: int,
    *,
    block: int,
    replicates: int,
    seed: int,
    mode: str,
) -> Dict[str, Any]:
    """日历轴重抽：D* = Σ d_i w / Σ w（索引计数等价于逐位置求和）。"""
    from scripts.v2_p7.t6.noise import _stationary_index

    finite = np.isfinite(values)
    day_pos = np.asarray(day_pos)[finite]
    values = np.asarray(values)[finite]
    sums = np.bincount(day_pos, weights=values, minlength=n_days)
    counts = np.bincount(day_pos, minlength=n_days).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    chunk = 1000
    collected: List[np.ndarray] = []
    invalid = 0
    remaining = int(replicates)
    while remaining > 0:
        size = min(chunk, remaining)
        remaining -= size
        if mode == "stationary":
            index = _stationary_index(rng=rng, size=size, days=n_days, block=block)
        elif mode == "circular":
            index = _circular_block_index(rng=rng, size=size, days=n_days, block=block)
        else:
            raise ValueError(f"未知重抽模式：{mode}")
        numerator = sums[index].sum(axis=1)
        denominator = counts[index].sum(axis=1)
        bad = denominator <= 0
        invalid += int(bad.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            collected.append(np.where(bad, np.nan, numerator / np.maximum(denominator, 1.0)))
    draws = np.concatenate(collected)
    finite_draws = draws[np.isfinite(draws)]
    if finite_draws.size == 0:
        return {
            "lo": None,
            "hi": None,
            "width": None,
            "invalid": invalid,
            "invalid_rate": invalid / max(int(replicates), 1),
            "replicates": int(replicates),
        }
    lo_v, hi_v = np.percentile(finite_draws, [2.5, 97.5])
    return {
        "lo": float(lo_v),
        "hi": float(hi_v),
        "width": float(hi_v - lo_v),
        "invalid": invalid,
        "invalid_rate": invalid / max(int(replicates), 1),
        "replicates": int(replicates),
    }


def sampling_envelope(
    result: P1PathsResult,
    intents: Any,
    params: P1RParams,
    *,
    n_days: int,
    replicates: Optional[int] = None,
) -> Dict[str, Any]:
    """双路抽样 + 敏感性 + 与识别界合成最终包络。"""
    replicates = int(replicates or params.bootstrap_replicates)
    bounded = np.isfinite(result.d_point) & (~result.unbounded)
    day_pos = np.asarray(intents.signal_pos)[bounded] + 1
    values = np.asarray(result.d_point)[bounded]
    main = _bootstrap_values(
        day_pos,
        values,
        n_days,
        block=params.bootstrap_main_block,
        replicates=replicates,
        seed=params.seeds["main_stationary"],
        mode="stationary",
    )
    aux = _bootstrap_values(
        day_pos,
        values,
        n_days,
        block=params.bootstrap_aux_block,
        replicates=replicates,
        seed=params.seeds["aux_circular"],
        mode="circular",
    )
    sensitivity = {
        str(block): _bootstrap_values(
            day_pos,
            values,
            n_days,
            block=int(block),
            replicates=replicates,
            seed=params.seeds["sensitivity"] + int(block),
            mode="stationary",
        )
        for block in params.bootstrap_sensitivity
    }
    widths = [item["width"] for item in sensitivity.values() if item["width"] is not None]
    method_sensitive = False
    if main["width"] and widths:
        method_sensitive = max(abs(width / main["width"] - 1.0) for width in widths) > 0.30
    envelope = aggregate_envelope(result)
    los = [item for item in (main["lo"], aux["lo"]) if item is not None]
    his = [item for item in (main["hi"], aux["hi"]) if item is not None]
    ci_lo = min(los) if los else None
    ci_hi = max(his) if his else None
    final_lo = None
    final_hi = None
    if envelope["constrained"] and envelope["d_min"] is not None and ci_lo is not None:
        final_lo = float(min(envelope["d_min"], ci_lo))
        final_hi = float(max(envelope["d_max"], ci_hi))
    invalid_rate = max(main["invalid_rate"], aux["invalid_rate"])
    return {
        "main": main,
        "aux": aux,
        "sensitivity": sensitivity,
        "method_sensitive": bool(method_sensitive),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "identification": envelope,
        "constraint_status": "constrained" if envelope["constrained"] else "unbounded",
        "final_lo": final_lo,
        "final_hi": final_hi,
        "invalid_rate": invalid_rate,
        "no_valid_interval": bool(invalid_rate > 0.05 or ci_lo is None),
    }


def decide_verdict(
    *,
    calibration_passed: bool,
    method_sensitive: bool,
    unbounded: bool,
    no_valid_interval: bool,
    final_lo: Optional[float],
    final_hi: Optional[float],
    negative_power: float,
    intent_total: int,
) -> Dict[str, Any]:
    """判定优先级：结构失败 > 校准 > 方法敏感 > 不可约束 > negative > continue。"""
    if intent_total <= 0:
        return {
            "verdict": "continue_limited",
            "reason_code": "empty_domain",
            "reason": "L1 为空：计量不可定义，继续但不否决",
        }
    if not calibration_passed:
        return {
            "verdict": "continue_limited",
            "reason_code": "calibration_not_passed",
            "reason": "层二校准未通过，禁用 negative",
        }
    if method_sensitive:
        return {
            "verdict": "continue_limited",
            "reason_code": "method_sensitive",
            "reason": "{45,180} 相对 90 的区间宽度变化 >30%",
        }
    if unbounded or no_valid_interval or final_lo is None:
        return {
            "verdict": "continue_limited",
            "reason_code": "unbounded",
            "reason": "存在未覆盖分支或无有效区间：继续但不否决",
        }
    if final_hi < 0 and negative_power >= 0.80:
        return {
            "verdict": "negative",
            "reason_code": "negative",
            "reason": "最终包络上界 < 0 且负向功效 ≥80%",
        }
    if final_lo <= 0.0 <= final_hi:
        return {
            "verdict": "continue_limited",
            "reason_code": "envelope_crosses_zero",
            "reason": "最终包络跨 0：继续但不否决",
        }
    return {"verdict": "continue", "reason_code": "continue", "reason": "未触发任何否决条件"}
