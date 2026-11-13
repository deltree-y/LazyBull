# -*- coding: utf-8 -*-
"""P7 T1 报告件：逐格执行诊断、构成/匹配质量表与匹配组账本帧。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import matching
from .execution import STATUS_RESOLVED, STATUS_UNFILLED, STATUS_UNRESOLVED

SLOT_CAPACITY = 6
INITIAL_CAPITAL_B0 = 195000.0


def _quantiles(values: np.ndarray, points: Sequence[float]) -> Dict[str, Optional[float]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {f"p{int(point * 100):02d}": None for point in points}
    return {f"p{int(point * 100):02d}": float(np.quantile(finite, point)) for point in points}


def _mean(values: np.ndarray) -> Optional[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def reason_breakdown(reasons: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for reason in reasons:
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def daily_concurrency(result: Any, n_days: int, start: int = 0) -> np.ndarray:
    """逐日并行持仓组数（仅已成交且已离场的组；只在评价窗内统计）。"""
    delta = np.zeros(n_days + 1, dtype=np.int64)
    filled = result.status == STATUS_RESOLVED
    entries = result.entry_pos[filled]
    exits = result.exit_pos[filled]
    valid = (entries >= int(start)) & (exits >= entries)
    np.add.at(delta, entries[valid], 1)
    np.add.at(delta, np.minimum(exits[valid] + 1, n_days), -1)
    return np.cumsum(delta[:n_days])[int(start) :]


def monthly_utilization(
    concurrency: np.ndarray,
    days: Sequence[str],
    *,
    min_ticket: float,
    initial_capital: float = INITIAL_CAPITAL_B0,
    slot_capacity: int = SLOT_CAPACITY,
) -> Dict[str, Any]:
    """月度槽位占用与资金占用（诊断口径，不是 E4-4 判据）。"""
    frame = pd.DataFrame({"day": list(days), "count": concurrency})
    frame["month"] = frame["day"].astype(str).str.slice(0, 6)
    frame["slot"] = np.minimum(frame["count"] / float(slot_capacity), 1.0)
    frame["capital"] = np.minimum(frame["count"] * float(min_ticket) / float(initial_capital), 1.0)
    grouped = frame.groupby("month")
    slot_median = grouped["slot"].median()
    capital_median = grouped["capital"].median()
    return {
        "diagnostic_only": True,
        "slot_capacity_assumption": int(slot_capacity),
        "initial_capital_assumption": float(initial_capital),
        "definition": "槽位占用 = min(并行组数/6, 1)；资金占用 = min(并行组数*V/195000, 1)",
        "months": int(grouped.ngroups),
        "slot_utilization_median": _mean(slot_median.to_numpy()),
        "capital_utilization_median": _mean(capital_median.to_numpy()),
        "slot_utilization_worst_month": (float(slot_median.min()) if len(slot_median) else None),
        "concurrency": _quantiles(concurrency, (0.5, 0.9, 0.99)),
    }


def composition_table(table: Any, panel: Any, result: Any) -> Dict[str, Any]:
    """事件/对照分组构成（ST、成交额、市值），只在已成交组的实际入场日取数。"""
    members = table.members
    width = members.shape[1]
    filled = result.status == STATUS_RESOLVED
    entry = np.clip(result.entry_pos, 0, panel.n_days - 1)
    safe = np.where(members >= 0, members, 0)
    payload: Dict[str, Any] = {}
    for role, columns in (("event", [0]), ("control", list(range(1, width)))):
        stocks = safe[:, columns].reshape(-1)
        days = np.repeat(entry[:, None], len(columns), axis=1).reshape(-1)
        valid = np.repeat(filled[:, None], len(columns), axis=1).reshape(-1) & (stocks >= 0)
        selected_stocks = stocks[valid]
        selected_days = days[valid]
        if selected_stocks.size:
            st_share = float(panel.is_st[selected_days, selected_stocks].mean())
            amount = panel.amount[selected_days, selected_stocks]
            market_value = panel.total_mv[selected_days, selected_stocks]
        else:
            st_share = None
            amount = np.array([])
            market_value = np.array([])
        payload[role] = {
            "observations": int(selected_stocks.size),
            "st_share": st_share,
            "amount_quantiles_yuan": {
                key: (value * 1000.0 if value is not None else None)
                for key, value in _quantiles(amount, (0.1, 0.5, 0.9)).items()
            },
            "total_mv_quantiles_wan": _quantiles(market_value, (0.1, 0.5, 0.9)),
        }
    return payload


def _distribution(values: np.ndarray) -> Dict[str, Optional[float]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"zero_share": None, "median": None, "max": None}
    return {
        "zero_share": float((finite == 0).mean()),
        "median": float(np.median(finite)),
        "max": float(finite.max()),
    }


def execution_report(
    result: Any,
    table: Any,
    panel: Any,
    *,
    matching_quality: Mapping[str, Any],
    matching_diagnostics: Mapping[str, Any],
    min_ticket: float,
    horizon: int,
    timeline_start: int = 0,
) -> Dict[str, Any]:
    """逐格执行诊断（覆盖验收门要求的全部报告项）。"""
    counts = result.counts()
    resolved = result.status == STATUS_RESOLVED
    unfilled = result.status == STATUS_UNFILLED
    unresolved = result.status == STATUS_UNRESOLVED
    entry_ok_groups = result.entry_pos >= 0
    concurrency = daily_concurrency(result, panel.n_days, start=int(timeline_start))
    utilized = monthly_utilization(
        concurrency, panel.days[int(timeline_start) :], min_ticket=min_ticket
    )
    event_side_missing = float((table.status == matching.STATUS_EVENT_COVARIATE_MISSING).mean())
    control_side_missing = float((table.status == matching.STATUS_INSUFFICIENT_CONTROLS).mean())
    endpoint_unresolved = int(
        sum(
            1
            for status, reason in zip(result.status, result.reason)
            if status == STATUS_UNRESOLVED and reason == "endpoint_unresolved"
        )
    )
    return {
        "horizon": int(horizon),
        "counts": counts,
        "status_breakdown": {
            "resolved_filled": int(resolved.sum()),
            "unfilled": int(unfilled.sum()),
            "unresolved": int(unresolved.sum()),
        },
        "reason_breakdown": reason_breakdown(result.reason),
        "common_entry_rate": float(entry_ok_groups.mean()) if result.size else None,
        "common_exit_rate": float(resolved.mean()) if result.size else None,
        "per_stock_entry_feasibility": (
            float(result.stock_entry_ok.mean()) if result.size else None
        ),
        "per_stock_exit_feasibility": (
            float(result.stock_exit_ok[entry_ok_groups].mean()) if entry_ok_groups.any() else None
        ),
        "entry_defer_days": _distribution(result.entry_defer[resolved]),
        "exit_defer_days": _distribution(result.exit_defer[resolved]),
        "entry_gap": _quantiles(result.entry_gap[resolved], (0.01, 0.5, 0.99)),
        "exit_gap": _quantiles(result.exit_gap[resolved], (0.01, 0.5, 0.99)),
        "one_price_board": {
            "entry_share": (
                float(result.one_price_entry[resolved].mean()) if resolved.any() else None
            ),
            "exit_share": (
                float(result.one_price_exit[resolved].mean()) if resolved.any() else None
            ),
        },
        "missing_rates": {
            "event_covariate_missing_share": event_side_missing,
            "insufficient_controls_share": control_side_missing,
            "event_side_missing_share": event_side_missing,
            "control_side_missing_share": control_side_missing,
            "endpoint_unresolved_count": endpoint_unresolved,
            "endpoint_unresolved_share": (
                endpoint_unresolved / counts["intent_total"] if counts["intent_total"] else None
            ),
        },
        "fees": {
            "buy_cost_mean": _mean(result.buy_cost[resolved]),
            "sell_cost_mean": _mean(result.sell_cost[resolved]),
            "total_cost_mean": _mean((result.buy_cost + result.sell_cost)[resolved]),
            "sell_amount_mean": _mean(result.sell_amount[resolved]),
        },
        "returns": {
            "event_net_quantiles": _quantiles(result.event_net[resolved], (0.05, 0.5, 0.95)),
            "control_net_quantiles": _quantiles(result.control_net[resolved], (0.05, 0.5, 0.95)),
            "residual_gross_mean": _mean(result.gross_residual[resolved]),
            "residual_net_mean": _mean(result.residual[resolved]),
            "filled_only_residual_mean": _mean(result.residual[resolved]),
        },
        "composition": composition_table(table, panel, result),
        "matching_quality": dict(matching_quality),
        "matching_diagnostics": dict(matching_diagnostics),
        "utilization": utilized,
        "capital_min_ticket": float(min_ticket),
    }


def group_frame(table: Any, result: Any, panel: Any, *, missing_status: str) -> pd.DataFrame:
    """匹配组级账本（每行一个六股组）。"""
    codes = np.asarray(panel.stocks, dtype=object)
    safe = np.where(table.members >= 0, table.members, 0)
    distances = np.where(np.isfinite(table.distances), table.distances, np.nan)
    finite = np.isfinite(distances)
    counts = finite.sum(axis=1)
    distance_min = np.where(counts > 0, np.where(finite, distances, np.inf).min(axis=1), np.nan)
    distance_mean = np.where(
        counts > 0, np.where(finite, distances, 0.0).sum(axis=1) / np.maximum(counts, 1), np.nan
    )
    frame: Dict[str, Any] = {
        "group_id": table.group_id,
        "source": np.asarray(table.source, dtype=object),
        "event_ts_code": codes[safe[:, 0]],
        "knowledge_date": table.knowledge_date,
        "event_date": table.event_date,
        "signal_day": [panel.days[p] for p in table.signal_pos],
        "planned_entry_day": [
            panel.days[p + 1] if p + 1 < panel.n_days else "" for p in table.signal_pos
        ],
        "entry_day": [panel.days[p] if p >= 0 else "" for p in result.entry_pos],
        "exit_day": [panel.days[p] if p >= 0 else "" for p in result.exit_pos],
        "entry_defer_days": result.entry_defer,
        "exit_defer_days": result.exit_defer,
        "status": result.status,
        "reason": result.reason,
        "distance_min": distance_min.astype(np.float32),
        "distance_mean": distance_mean.astype(np.float32),
        "industry_all_match": table.industry_match.all(axis=1),
        "industry_all_known": table.industry_known.all(axis=1),
        "event_net": result.event_net.astype(np.float32),
        "control_mean_net": result.control_net.astype(np.float32),
        "net_residual": result.residual.astype(np.float32),
        "gross_residual": result.gross_residual.astype(np.float32),
        "buy_cost_sum": np.nansum(result.buy_cost, axis=1).astype(np.float32),
        "sell_cost_sum": np.nansum(result.sell_cost, axis=1).astype(np.float32),
        "buy_amount_sum": np.nansum(result.buy_amount, axis=1).astype(np.float32),
        "sell_amount_sum": np.nansum(result.sell_amount, axis=1).astype(np.float32),
        "missing_identification_status": missing_status,
    }
    for offset in range(1, table.members.shape[1]):
        frame[f"control{offset}_ts_code"] = codes[safe[:, offset]]
        frame[f"control{offset}_distance"] = distances[:, offset - 1].astype(np.float32)
    return pd.DataFrame(frame)


def stock_frame(table: Any, result: Any, panel: Any) -> pd.DataFrame:
    """六股明细（协变量、距离、成交金额、费用、毛/净收益）。"""
    codes = np.asarray(panel.stocks, dtype=object)
    safe = np.where(table.members >= 0, table.members, 0)
    entry = np.clip(result.entry_pos, 0, panel.n_days - 1)
    exit_position = np.clip(result.exit_pos, 0, panel.n_days - 1)
    blocks: List[pd.DataFrame] = []
    for column in range(table.members.shape[1]):
        role = "event" if column == 0 else "control"
        distance = (
            np.zeros(table.size, dtype=np.float32)
            if column == 0
            else table.distances[:, column - 1]
        )
        blocks.append(
            pd.DataFrame(
                {
                    "group_id": table.group_id,
                    "role": role,
                    "ts_code": codes[safe[:, column]],
                    "buy_day": [panel.days[p] if p >= 0 else "" for p in result.entry_pos],
                    "sell_day": [panel.days[p] if p >= 0 else "" for p in result.exit_pos],
                    "distance": distance.astype(np.float32),
                    "buy_price": panel.open_u[entry, safe[:, column]],
                    "sell_price": panel.open_u[exit_position, safe[:, column]],
                    "shares": result.shares[:, column],
                    "sell_shares": result.sell_shares[:, column],
                    "buy_amount": result.buy_amount[:, column].astype(np.float32),
                    "sell_amount": result.sell_amount[:, column].astype(np.float32),
                    "buy_cost": result.buy_cost[:, column].astype(np.float32),
                    "sell_cost": result.sell_cost[:, column].astype(np.float32),
                    "gross_return": result.gross[:, column].astype(np.float32),
                    "net_return": result.net[:, column].astype(np.float32),
                    "entry_gap": result.entry_gap[:, column].astype(np.float32),
                    "exit_gap": result.exit_gap[:, column].astype(np.float32),
                    "one_price_entry": result.one_price_entry[:, column],
                    "one_price_exit": result.one_price_exit[:, column],
                    "stock_entry_feasible": result.stock_entry_ok[:, column],
                    "stock_exit_feasible": result.stock_exit_ok[:, column],
                    "total_mv_on_signal": panel.total_mv[table.signal_pos, safe[:, column]],
                    "ret20_on_signal": panel.ret20[table.signal_pos, safe[:, column]],
                    "vol20_on_signal": panel.vol20[table.signal_pos, safe[:, column]],
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def disclosure_seasons(table: Any) -> Dict[str, Any]:
    """披露季覆盖（报告期分布，只作样本描述）。"""
    periods = pd.Series([str(item) for item in table.event_date])
    years = periods.str.slice(0, 4)
    return {
        "distinct_report_periods": int(periods.nunique()),
        "distinct_report_years": int(years.nunique()),
        "events_by_year": {str(k): int(v) for k, v in years.value_counts().sort_index().items()},
        "report_period_top": {str(k): int(v) for k, v in periods.value_counts().head(8).items()},
    }
