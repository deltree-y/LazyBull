# -*- coding: utf-8 -*-
"""IDX-IN300 P4 一次性预筛：执行冻结规格并落三表与 ES 判决。"""

from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p7.p4.data import A_NAV, DATA, P4Data, sha256  # noqa: E402
from scripts.v2_p7.p4.engine import (  # noqa: E402
    BenchmarkRun,
    CandidateRun,
    prepare_plans,
    run_benchmark,
    run_candidate,
)
from scripts.v2_p7.p4.metrics import (  # noqa: E402
    compound_blocks,
    corr_line,
    monthly_returns,
    nav_metrics,
    tail_attribution,
    yearly_returns,
)

RUN_ID = os.environ.get("P4_RUN_ID", "p7_idx_in300_p4_20261114_r1")
OUT = DATA / "frozen_reference/v2_p7" / RUN_ID
P3_RUN = DATA / "frozen_reference/v2_p7/p7_idx_in300_p3_20261114_r1"
AUTH = OUT / "authorization.json"
CRASH_WINDOWS = json.loads(
    (DATA / "paper_shadow/p7_testbed/p7_t05_20261111_r6/specs/crash_windows_v6.json").read_text(
        encoding="utf-8"
    )
)["windows"]
COMMON_START = "20181126"
COMMON_END = "20260202"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Decimal):
        return float(value)
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _series_to_frame(series: pd.Series, column: str) -> pd.DataFrame:
    frame = series.rename(column).to_frame()
    frame.index.name = "trade_date"
    return frame


def _crash_returns(levels: pd.Series) -> dict[str, float | None]:
    result = {}
    for window in CRASH_WINDOWS:
        start = str(window["start_date"]).replace("-", "")
        end = str(window["end_date"]).replace("-", "")
        part = levels.loc[(levels.index >= start) & (levels.index <= end)]
        result[f"{window['year']}_{start}_{end}"] = (
            float(part.iloc[-1] / part.iloc[0] - 1.0) if len(part) >= 2 else None
        )
    return result


def _monthly_pair(
    candidate: CandidateRun, benchmark: BenchmarkRun
) -> tuple[pd.Series, pd.Series, pd.Series]:
    c = monthly_returns(candidate.levels, candidate.start_date)
    b = monthly_returns(benchmark.levels, benchmark.start_date)
    joined = pd.concat([c.rename("candidate"), b.rename("benchmark")], axis=1, join="inner")
    joined["excess"] = joined["candidate"] - joined["benchmark"]
    return joined["candidate"], joined["benchmark"], joined["excess"]


def _cost_breakdown(candidate: CandidateRun, metrics: dict) -> dict:
    fee = sum((Decimal(item["fee"]) for item in candidate.fills), Decimal("0"))
    buy_notional = sum(
        (Decimal(item["amount"]) for item in candidate.fills if item["side"] == "buy"),
        Decimal("0"),
    )
    sell_notional = sum(
        (Decimal(item["amount"]) for item in candidate.fills if item["side"] == "sell"),
        Decimal("0"),
    )
    years = metrics["calendar_days"] / 365.25
    mean_nav = float(candidate.levels.mean())
    cost_drag = float(fee) / mean_nav / years * 100 if years > 0 and mean_nav > 0 else None
    turnover = (
        float(buy_notional + sell_notional) / mean_nav / years
        if years > 0 and mean_nav > 0
        else None
    )
    return {
        "fees_cny": float(fee),
        "cost_drag_pp_per_year": cost_drag,
        "cost_proxy_9_454bp_times_G_pp": (
            0.0009454 * turnover * 100 if turnover is not None else None
        ),
        "turnover_G_per_year": turnover,
        "nav": {
            "buy_notional": float(buy_notional),
            "sell_notional": float(sell_notional),
            "mean_nav": mean_nav,
        },
    }


def _selection_summary(candidate: CandidateRun, plans: dict) -> tuple[dict, list[dict], list[dict]]:
    selection = [dict(item) for plan in plans.values() for item in plan.selected]
    audit = [dict(item) for plan in plans.values() for item in plan.selection_audit]
    status_counts = {}
    for item in candidate.exceptions:
        key = f"{item['side']}:{item['status']}"
        status_counts[key] = status_counts.get(key, 0) + 1
    fills = {
        "buy": int(sum(item["side"] == "buy" for item in candidate.fills)),
        "sell": int(sum(item["side"] == "sell" for item in candidate.fills)),
        "selected": len(selection),
        "exceptions": len(candidate.exceptions),
        "status_counts": status_counts,
    }
    selection_block = {
        "batches": len(plans),
        "selected": len(selection),
        "audit_rows": len(audit),
        "audit": audit,
    }
    return fills, selection, selection_block


def _capacity_missing(candidate: CandidateRun, plans: dict) -> tuple[dict, dict]:
    capacity_1pct = [
        float(plan.capacity_1pct) for plan in plans.values() if plan.capacity_1pct is not None
    ]
    capacity_5pct = [
        float(plan.capacity_5pct) for plan in plans.values() if plan.capacity_5pct is not None
    ]
    missing_max = max(
        (float(item.get("unresolved_weight", 0.0)) for item in candidate.missing_rows),
        default=0.0,
    )
    missing_ratio = (
        candidate.missing_position_days / candidate.position_days
        if candidate.position_days > 0
        else 0.0
    )
    capacity = {
        "1pct_cny": min(capacity_1pct) if capacity_1pct else None,
        "5pct_cny": min(capacity_5pct) if capacity_5pct else None,
    }
    missing = {
        "max_unresolved_weight": missing_max,
        "position_days": candidate.position_days,
        "missing_position_days": candidate.missing_position_days,
        "missing_position_day_ratio": missing_ratio,
        "records": len(candidate.missing_rows),
    }
    return capacity, missing


def _es1_block(candidate: CandidateRun, benchmark: BenchmarkRun) -> dict:
    c_monthly, b_monthly, excess = _monthly_pair(candidate, benchmark)
    monthly_t = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(len(excess)))
        if len(excess) > 1 and excess.std(ddof=1) > 0
        else None
    )
    return {
        "mean_monthly_excess": float(excess.mean()),
        "annualized_pp": 12.0 * float(excess.mean()) * 100.0,
        "months": int(len(excess)),
        "monthly_t": monthly_t,
        "monthly": [
            {
                "month": str(month),
                "candidate": float(candidate_value),
                "benchmark": float(benchmark_value),
                "excess": float(excess_value),
            }
            for month, candidate_value, benchmark_value, excess_value in zip(
                excess.index, c_monthly.loc[excess.index], b_monthly.loc[excess.index], excess
            )
        ],
    }


def _table1(
    data: P4Data,
    candidate: CandidateRun,
    benchmark: BenchmarkRun,
    plans: dict,
) -> dict:
    metrics = nav_metrics(candidate.levels)
    cost = _cost_breakdown(candidate, metrics)
    fills, _selection, selection_block = _selection_summary(candidate, plans)
    capacity, missing = _capacity_missing(candidate, plans)
    return {
        "metrics": metrics,
        **cost,
        "es1": _es1_block(candidate, benchmark),
        "yearly_returns": yearly_returns(candidate.levels),
        "crash_segment_returns": _crash_returns(candidate.levels),
        "selection": selection_block,
        "fills": fills,
        "capacity": capacity,
        "missing": missing,
        "final_positions": list(candidate.final_positions),
        "benchmark_events": list(benchmark.event_rows),
    }


def _common_series(data: P4Data, candidate: CandidateRun) -> tuple[pd.Series, pd.Series, pd.Series]:
    a = data.a_nav
    common = a.index[
        (a.index >= COMMON_START) & (a.index <= COMMON_END) & a.index.isin(candidate.levels.index)
    ]
    if len(common) < 2:
        raise RuntimeError("A/candidate 共同窗口不足")
    a_levels = a.loc[common]
    c_levels = candidate.levels.reindex(common).ffill()
    return a_levels, c_levels, pd.Index(common)


def _correlation_table(data: P4Data, candidate: CandidateRun) -> tuple[dict, dict]:
    a_levels, c_levels, common = _common_series(data, candidate)
    a_ret = a_levels.pct_change().dropna()
    c_ret = c_levels.pct_change().dropna()
    full = corr_line(a_ret, c_ret, block=10, min_observations=250)
    paired = pd.concat([a_ret.rename("a"), c_ret.rename("c")], axis=1, join="inner").dropna()
    a_blocks = compound_blocks(paired["a"].to_numpy(dtype=float), 10)
    c_blocks = compound_blocks(paired["c"].to_numpy(dtype=float), 10)
    rebalance = corr_line(
        pd.Series(a_blocks),
        pd.Series(c_blocks),
        block=1,
        min_observations=40,
    )
    crash_left, crash_right = [], []
    window_stats = {}
    for window in CRASH_WINDOWS:
        start = str(window["start_date"]).replace("-", "")
        end = str(window["end_date"]).replace("-", "")
        left = a_ret.loc[(a_ret.index >= start) & (a_ret.index <= end)]
        right = c_ret.loc[(c_ret.index >= start) & (c_ret.index <= end)]
        joined = pd.concat([left.rename("a"), right.rename("c")], axis=1, join="inner").dropna()
        key = f"{window['year']}_{start}_{end}"
        window_stats[key] = {
            "paired_days": int(len(joined)),
            "a_return": float(left.sum()) if len(left) else None,
            "candidate_return": float(right.sum()) if len(right) else None,
        }
        crash_left.extend(joined["a"].tolist())
        crash_right.extend(joined["c"].tolist())
    crash = corr_line(
        pd.Series(crash_left),
        pd.Series(crash_right),
        block=5,
        min_observations=60,
    )
    crash["pairable_windows"] = int(
        sum(1 for item in window_stats.values() if item["paired_days"] > 0)
    )
    if crash["pairable_windows"] < 2:
        crash["estimable"] = False
    table2 = {
        "common_window": [str(common[0]), str(common[-1])],
        "a_metrics_normalized": nav_metrics(a_levels / a_levels.iloc[0]),
        "candidate_metrics_common": nav_metrics(c_levels / c_levels.iloc[0]),
        "correlation": {
            "full_sample": full,
            "rebalance_10d": rebalance,
            "crash_segment": crash,
            "windows": window_stats,
        },
        "a_reference": {
            "path": str(A_NAV.relative_to(ROOT)),
            "sha256": sha256(A_NAV),
            "return_days": int(len(a_ret)),
        },
    }
    points = [
        full["point"],
        rebalance["point"],
        crash["point"],
    ]
    estimable = [full["estimable"], rebalance["estimable"], crash["estimable"]]
    finite_points = [float(item) for item in points if item is not None and np.isfinite(item)]
    es5 = {
        "point_max": max(finite_points) if finite_points else None,
        "estimable": bool(all(estimable)),
        "points": points,
        "interval_upper_975": [
            full["upper_975"],
            rebalance["upper_975"],
            crash["upper_975"],
        ],
    }
    return table2, es5


def _table3(data: P4Data, reset: CandidateRun) -> tuple[dict, dict]:
    a_levels, c_levels, common = _common_series(data, reset)
    a_growth = a_levels / a_levels.iloc[0]
    a_only = 1_000_000.0 * a_growth
    a_leg = 700_000.0 * a_growth
    combined = a_leg + c_levels
    a_only_metrics = nav_metrics(a_only)
    combined_metrics = nav_metrics(combined)
    candidate_metrics = nav_metrics(c_levels)
    d_pp = (combined_metrics["max_drawdown"] - a_only_metrics["max_drawdown"]) * 100.0
    p_pp = (combined_metrics["cagr"] - a_only_metrics["cagr"]) * 100.0
    if d_pp >= -1.0:
        es3_pass = True
    elif d_pp >= -3.0:
        es3_pass = bool(p_pp > 0)
    else:
        es3_pass = False
    tail = tail_attribution(a_only, combined, a_leg, c_levels)
    table3 = {
        "common_window": [str(common[0]), str(common[-1])],
        "combined_metrics": combined_metrics,
        "a_only_metrics": a_only_metrics,
        "candidate_leg_metrics": candidate_metrics,
        "delta": {
            "d_maxdd_pp": d_pp,
            "p_cagr_pp": p_pp,
            "pass": es3_pass,
        },
        "tail_attribution": tail,
        "weights": {"a": 0.7, "candidate": 0.3, "total_principal": 1_000_000.0},
    }
    levels = pd.DataFrame(
        {"a_only": a_only, "combined": combined, "a_leg": a_leg, "candidate_leg": c_levels}
    )
    levels.index.name = "trade_date"
    es3 = {
        "d_maxdd_pp": d_pp,
        "p_cagr_pp": p_pp,
        "pass": es3_pass,
    }
    return table3, {"es3": es3, "levels": levels}


def _es_block(table1: dict, es5: dict, es3: dict, flags: dict) -> dict:
    return {
        "ES1": {
            "value_pp_per_year": table1["es1"]["annualized_pp"],
            "threshold": 1.0,
            "pass": flags["es1"],
        },
        "ES2": {
            "value_pp_per_year": table1["cost_drag_pp_per_year"],
            "threshold": 2.0,
            "pass": flags["es2"],
        },
        "ES3": es3,
        "ES4": {
            "cagr": table1["metrics"]["cagr"],
            "threshold": 0.0,
            "pass": flags["es4"],
        },
        "ES5": {
            "point_max": es5["point_max"],
            "threshold": 0.3,
            "estimable": es5["estimable"],
            "pass": flags["es5"],
        },
        "capacity": {
            "1pct_cny": table1["capacity"]["1pct_cny"],
            "5pct_cny": table1["capacity"]["5pct_cny"],
            "gate": "not_applicable_personal_scale",
            "is_informational": True,
        },
        "missing_material": flags["missing"],
    }


def _verdict(flags: dict, es5: dict, es: dict) -> tuple[str, str, list[str]]:
    core_fail = [key for key in ("ES1", "ES2", "ES3", "ES4") if not es[key]["pass"]]
    if core_fail:
        return "stop_new_investment", "stop", core_fail
    if not es5["estimable"]:
        return "limited", "stop_additional_investment", ["ES5_not_estimable"]
    if not flags["es5"]:
        return "stop_new_investment", "stop", ["ES5"]
    if flags["missing"]:
        return "limited", "stop_additional_investment", ["missing_price_materiality"]
    return (
        "continue_to_formal_review",
        "propose_formal_T2_T3_only",
        ["all_es_pass_capacity_informational"],
    )


def _decide(table1: dict, es5: dict, es3: dict) -> dict:
    flags = {
        "es1": bool(table1["es1"]["annualized_pp"] >= 1.0),
        "es2": bool(
            table1["cost_drag_pp_per_year"] is not None and table1["cost_drag_pp_per_year"] <= 2.0
        ),
        "es4": bool(table1["metrics"]["cagr"] > 0),
        "es5": bool(es5["estimable"] and es5["point_max"] is not None and es5["point_max"] <= 0.3),
        "missing": bool(
            table1["missing"]["max_unresolved_weight"] > 0.05
            or table1["missing"]["missing_position_day_ratio"] > 0.05
        ),
    }
    es = _es_block(table1, es5, es3, flags)
    verdict, action, reasons = _verdict(flags, es5, es)
    return {
        "status": "complete",
        "verdict": verdict,
        "action": action,
        "reasons": reasons,
        "es": es,
        "alpha_consumed": False,
        "formal_t2_t3": False,
    }


def _summary(
    table1: dict,
    table2: dict,
    table3: dict,
    es5: dict,
    decision: dict,
) -> dict:
    return {
        "run_id": str(RUN_ID),
        "candidate": "IDX-IN300",
        "prereg_sha256": sha256(P3_RUN / "prereg/p7_idx_in300_prereg_v1.json"),
        "prereg_criteria_digest": json.loads(
            (P3_RUN / "prereg/p7_idx_in300_prereg_v1.json").read_text(encoding="utf-8")
        )["criteria_digest"],
        "authorization": str(AUTH.relative_to(ROOT)),
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "es5": es5,
        "decision": decision,
        "boundaries": {
            "exploratory_prescreen": True,
            "alpha_consumed": False,
            "formal_t2_t3": False,
            "full_universe_metrics_not_used": True,
        },
    }


def _write_outputs(
    plans: dict,
    candidate: CandidateRun,
    benchmark: BenchmarkRun,
    reset: CandidateRun,
    table2: dict,
    table3_extra: dict,
    summary: dict,
    decision: dict,
) -> None:
    _write_json(OUT / "summary.json", summary)
    _write_json(OUT / "decision.json", decision)
    _series_to_frame(candidate.levels, "candidate_nav").to_parquet(
        OUT / "candidate_daily_nav.parquet"
    )
    _series_to_frame(benchmark.levels, "benchmark_nav").to_parquet(
        OUT / "benchmark_daily_nav.parquet"
    )
    _series_to_frame(reset.levels, "candidate_table3_nav").to_parquet(
        OUT / "candidate_table3_daily_nav.parquet"
    )
    table3_extra["levels"].to_parquet(OUT / "table3_daily_nav.parquet")
    # E4-4：逐日持仓市值/NAV（空窗日按 0 计入；不删零事件月）
    pd.DataFrame(list(candidate.daily_utilization)).to_parquet(
        OUT / "daily_utilization.parquet"
    )
    c_monthly = monthly_returns(candidate.levels, candidate.start_date)
    b_monthly = monthly_returns(benchmark.levels, benchmark.start_date)
    pd.concat([c_monthly.rename("candidate"), b_monthly.rename("benchmark")], axis=1).to_csv(
        OUT / "monthly_returns.csv", encoding="utf-8-sig"
    )
    frames = {
        "selected.csv": [dict(item) for plan in plans.values() for item in plan.selected],
        "selection_audit.csv": [
            dict(item) for plan in plans.values() for item in plan.selection_audit
        ],
        "fills.csv": list(candidate.fills),
        "exceptions.csv": list(candidate.exceptions),
        "missing_prices.csv": list(candidate.missing_rows),
        "corporate_actions.csv": list(candidate.corporate_actions),
    }
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    _write_json(
        OUT / "run_meta.json",
        {
            "event_batches": len(plans),
            "candidate_start": candidate.start_date,
            "candidate_end": candidate.end_date,
            "benchmark_start": benchmark.levels.index[0],
            "benchmark_end": benchmark.levels.index[-1],
            "reset_start": reset.start_date,
            "a_common": [table2["common_window"][0], table2["common_window"][1]],
            "selection_rows": sum(len(plan.selected) for plan in plans.values()),
        },
    )


def main() -> None:
    if (OUT / "summary.json").exists() and not os.environ.get("P4_ALLOW_REPLAY"):
        raise RuntimeError(
            f"冻结根已存在，拒绝覆盖: {OUT}；请设置新的 P4_RUN_ID 重放到新根"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    data = P4Data()
    plans, cache = prepare_plans(data)
    start = min(plan.signal_date for plan in plans.values())
    candidate = run_candidate(data, plans, cache, principal=300_000, start_date=start, label="full")
    benchmark = run_benchmark(data, plans, cache, principal=300_000, start_date=start)
    table1 = _table1(data, candidate, benchmark, plans)
    table2, es5 = _correlation_table(data, candidate)
    reset = run_candidate(
        data, plans, cache, principal=300_000, start_date=COMMON_START, label="table3_reset"
    )
    table3, table3_extra = _table3(data, reset)
    decision = _decide(table1, es5, table3_extra["es3"])
    summary = _summary(table1, table2, table3, es5, decision)
    _write_outputs(plans, candidate, benchmark, reset, table2, table3_extra, summary, decision)
    print(json.dumps(_jsonable(decision), ensure_ascii=False))


if __name__ == "__main__":
    main()
