# -*- coding: utf-8 -*-
"""C1（RM1-IN）规则型探索性预筛：一次性交付 C1 独立收益与信号诊断。

边界：exploratory / not_prereg / not_admission；不消费 α；不构成 B 区转正、
生产多袖、实盘或前向/日历验证。按评审包 v3.1 的 ES1~ES5 与五级优先表出资源决定。

# 复杂度豁免：本文件为一次性探索性预筛入口（scripts/v2_p7/c1/**，不入生产链路）；
# main 184 行超硬限，若结论要求正式化，再拆为 load/run/report 三段。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p7.c1.backtest import (  # noqa: E402
    MIN_TICKET,
    TOP_N,
    benchmark_series,
    matched_benchmark_series,
    run_c1,
)
from scripts.v2_p7.c1.monthly import (  # noqa: E402
    FORMATION_MONTHS,
    MIN_VALID_MONTHS,
    build_monthly,
    top_codes,
)
from scripts.v2_p7.t1.dataplane import (  # noqa: E402
    load_panel,
    load_trading_days,
    metadata_digest,
)

DEFAULT_OUT = ROOT / "data" / "paper_shadow" / "p7_c1_prescreen"
BOUNDARIES = [
    "exploratory pre-screen；不是正式预登记、不是正式 P-1 仪器",
    "不构成 C1 经济准入、B 区转正、生产多袖、实盘可用、前向/日历验证",
    "alpha_consumed=false；与 B0 旧证据不可比",
    "C1-IN 是自定义行业×规模分组残差，不是 FF3 复现",
    "负结果只适用于本冻结算法的实现，不外推为所有残差动量无效",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)
    path.write_text(text + "\n", encoding="utf-8")
    return _sha256_file(path)


def _annualized_monthly(nav: pd.Series) -> Dict[str, float]:
    raw = nav.dropna().to_numpy(dtype=float)
    if len(raw) < 1:
        return {}
    # 补单位起点，纳入首月收益
    values = np.concatenate([[1.0], raw])
    n = len(values) - 1
    cagr = values[-1] ** (12.0 / n) - 1.0
    peak = np.maximum.accumulate(values)
    mdd = float(np.nanmin(values / peak - 1.0))
    rets = pd.Series(values).pct_change().dropna()
    vol = float(rets.std(ddof=1) * np.sqrt(12))
    sharpe = float(rets.mean() * 12 / vol) if vol > 0 else float("nan")
    return {
        "cagr": float(cagr),
        "max_drawdown": mdd,
        "volatility": vol,
        "sharpe": sharpe,
        "months": int(n),
    }


def _nw_t(x: np.ndarray, lags: int = 6) -> float:
    """Newey-West t（月度 IC 序列，简单 Bartlett 核）。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    err = x - mean
    gamma0 = float(np.dot(err, err) / n)
    s = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = float(np.dot(err[lag:], err[:-lag]) / n)
        s += 2.0 * weight * gamma
    se = np.sqrt(max(s, 0.0) / n)
    return mean / se if se > 0 else float("nan")


def _rank_ic(panel, monthly, signal_matrix: np.ndarray, label: str) -> Dict[str, float]:
    from scipy.stats import spearmanr

    values = []
    for m in range(FORMATION_MONTHS, len(monthly.month_end_days) - 1):
        signal = signal_matrix[m]
        ret = monthly.returns[m + 1]
        mask = monthly.eligible[m] & np.isfinite(signal) & np.isfinite(ret)
        if mask.sum() < 50:
            continue
        ic, _ = spearmanr(signal[mask], ret[mask])
        values.append(float(ic))
    arr = np.asarray(values)
    return {
        "label": label,
        "months": int(len(arr)),
        "mean_ic": float(np.mean(arr)),
        "t_stat": float(np.mean(arr) / (np.std(arr, ddof=1) / np.sqrt(len(arr)))),
        "t_stat_nw": float(_nw_t(arr)),
        "positive_months": int((arr > 0).sum()),
    }


def _signal_diagnostics(panel, monthly) -> Dict[str, Any]:
    n_months, n_stocks = monthly.residuals.shape
    mean_only = np.full((n_months, n_stocks), np.nan)
    raw = np.full((n_months, n_stocks), np.nan)
    rev = np.full((n_months, n_stocks), np.nan)
    for m in range(FORMATION_MONTHS, n_months):
        block = monthly.residuals[m - FORMATION_MONTHS : m]
        count = np.sum(np.isfinite(block), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(invalid="ignore"):
                mean_only[m] = np.where(
                    count >= MIN_VALID_MONTHS, np.nanmean(block, axis=0), np.nan
                )
    for m in range(12, n_months):
        end = int(monthly.month_end_days[m - 1])
        start = int(monthly.month_end_days[m - 12])
        with np.errstate(invalid="ignore", divide="ignore"):
            raw[m] = panel.close_adj[end] / panel.close_adj[start] - 1.0
        rev[m] = -monthly.returns[m]

    rows = []
    for m in range(11, len(monthly.month_end_days) - 1):
        selected = top_codes(monthly, panel.stocks, m, TOP_N)
        if not selected:
            continue
        eligible = monthly.eligible[m] & np.isfinite(monthly.returns[m + 1])
        if not eligible.any():
            continue
        idx = [panel.stock_index[c] for c in selected if c in panel.stock_index]
        if not idx:
            continue
        rows.append(
            {
                "signal_date": panel.days[int(monthly.month_end_days[m])],
                "selected": len(idx),
                "sel_ret": float(np.nanmean(monthly.returns[m + 1][idx])),
                "bench_ret": float(np.nanmean(monthly.returns[m + 1][eligible])),
            }
        )
    frame = pd.DataFrame(rows)
    frame["diff"] = frame["sel_ret"] - frame["bench_ret"]
    diff = frame["diff"].to_numpy()
    return {
        "rank_ic": {
            "mean_std_residual": _rank_ic(panel, monthly, monthly.signals, "mean/std residual"),
            "mean_residual": _rank_ic(panel, monthly, mean_only, "mean residual"),
            "raw_12_1": _rank_ic(panel, monthly, raw, "raw 12-1"),
            "one_month_reversal": _rank_ic(panel, monthly, rev, "1M reversal"),
        },
        "top6_vs_benchmark": {
            "months": int(len(diff)),
            "mean_monthly_diff": float(np.mean(diff)),
            "annualized_arithmetic": float(np.mean(diff) * 12),
            "t_stat": float(np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))),
            "positive_months": int((diff > 0).sum()),
        },
        "rows": frame,
    }


def _es1(c1_monthly: pd.DataFrame, benchmark: pd.DataFrame) -> Dict[str, float]:
    c1 = c1_monthly["ret"].copy()
    c1.index = c1.index.to_period("M")
    bench = benchmark.copy()
    bench["period"] = pd.to_datetime(bench["hold_date"], format="%Y%m%d").dt.to_period("M")
    bench = bench.set_index("period")["gross_return"]
    common = c1.index.intersection(bench.index)
    diff = (c1.loc[common] - bench.loc[common]).dropna()
    return {
        "months": int(len(diff)),
        "mean_monthly_net_excess": float(diff.mean()),
        "annualized_arithmetic": float(diff.mean() * 12),
        "t_stat": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))),
    }


def decide(
    *,
    implementation_valid: bool,
    es1_pass: bool,
    es2_pass: bool,
    es4_pass: bool,
    es3_pass: bool | None,
    es5_pass: bool | None,
) -> Tuple[str, str]:
    """五级优先裁决（ES3/ES5 未计算时不得前进）。"""
    if not implementation_valid:
        return "invalid_implementation", "accounting_or_time_axis"
    if not (es1_pass and es2_pass and es4_pass):
        return "stop_new_investment", "economic_support_not_met"
    if es3_pass is None or es5_pass is None:
        return "incomplete_requires_table2_3", "es3_es5_not_evaluated"
    if not es3_pass:
        return "stop_new_investment", "es3_not_met"
    if not es5_pass:
        return "stop_new_investment", "es5_not_met"
    return "proceed_to_formal_validation", "es_all_met"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C1 规则型探索性预筛")
    parser.add_argument("--run-id", default="p7_c1_prescreen_20261114_r1")
    parser.add_argument("--cutoff", default="20261029")
    parser.add_argument("--start", default="20111010")
    parser.add_argument("--principal", type=float, default=300_000.0)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    started = time.time()
    out_root = Path(args.out_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=False)

    calendar_days = load_trading_days(ROOT / "data", args.cutoff)
    days = [d for d in calendar_days if d >= args.start]
    panel = load_panel(ROOT / "data", days, cutoff=args.cutoff, industry_source="pit")
    monthly = build_monthly(panel, ROOT / "data", calendar_days=calendar_days)
    result = run_c1(panel, monthly, panel.stocks, principal=args.principal)
    benchmark = benchmark_series(panel, monthly)
    matched = matched_benchmark_series(panel, monthly, panel.stocks)
    diagnostics = _signal_diagnostics(panel, monthly)

    c1_metrics = result.metrics
    bench_metrics = _annualized_monthly(
        pd.Series(
            benchmark["nav"].to_numpy(),
            index=pd.to_datetime(benchmark["hold_date"], format="%Y%m%d"),
        )
    )
    es1 = _es1(result.monthly, benchmark)
    es2 = float(c1_metrics.get("cost_drag", float("nan")))
    es4 = float(c1_metrics.get("cagr", float("nan")))
    es1_pass = bool(es1["annualized_arithmetic"] >= 0.01)
    es2_pass = bool(es2 <= 0.02)
    es4_pass = bool(es4 > 0)
    implementation_valid = bool(result.validity.get("valid", False))
    bench_unresolved_mean = float(benchmark["unresolved_weight"].mean())
    bench_unresolved_max = float(benchmark["unresolved_weight"].max())
    missing_ratio = float(c1_metrics.get("missing_price_value_ratio", 0.0))
    limited = bool(missing_ratio > 0.05 or bench_unresolved_mean > 0.02)
    limitations = []
    if missing_ratio > 0.05:
        limitations.append(f"缺失价格估值占 NAV-日比例 {missing_ratio:.2%} > 5%")
    if bench_unresolved_mean > 0.02:
        limitations.append(
            f"基准未解析权重均值 {bench_unresolved_mean:.2%} > 2%（最大 {bench_unresolved_max:.2%}）"
        )
    decision, reason = decide(
        implementation_valid=implementation_valid,
        es1_pass=es1_pass,
        es2_pass=es2_pass,
        es4_pass=es4_pass,
        es3_pass=None,
        es5_pass=None,
    )

    # 落盘
    result.nav.rename("nav").to_csv(out_root / "c1_nav.csv", encoding="utf-8")
    result.monthly.to_csv(out_root / "c1_monthly.csv", encoding="utf-8")
    result.trades.to_csv(out_root / "c1_trades.csv", index=False, encoding="utf-8")
    benchmark.to_csv(out_root / "benchmark.csv", index=False, encoding="utf-8")
    matched.to_csv(out_root / "matched_benchmark.csv", index=False, encoding="utf-8")
    diagnostics["rows"].to_csv(out_root / "signal_test.csv", index=False, encoding="utf-8")
    _write_json(
        out_root / "metrics.json",
        {
            "c1": c1_metrics,
            "benchmark": bench_metrics,
            "matched_benchmark_rows": int(len(matched)),
            "validity": result.validity,
            "es1": es1,
            "cagr_gap_c1_minus_benchmark": float(es4 - bench_metrics.get("cagr", float("nan"))),
            "es2_cost_drag": es2,
            "es3": {
                "status": "not_evaluated_unknown",
                "reason": "ES1~ES4 未全过，按五级优先表停止；A/组合表未构建",
            },
            "es4_cagr": es4,
            "es5": {
                "status": "not_evaluated_unknown",
                "reason": "ES1~ES4 未全过，按五级优先表停止；A 相关未计算",
            },
            "benchmark_unresolved_weight_mean": bench_unresolved_mean,
            "benchmark_unresolved_weight_max": bench_unresolved_max,
            "scope_deviation": True,
            "scope_deviation_note": "v3.1 要求三张表；本 run 仅表一与诊断，表二/表三未交付",
        },
    )
    diagnostics_payload = {k: v for k, v in diagnostics.items() if k != "rows"}
    _write_json(out_root / "signal_diagnostics.json", diagnostics_payload)
    if not implementation_valid:
        note = "账务/时间轴有效性未通过；本 run 不作经济决定，先修实现。"
    elif decision == "stop_new_investment" and reason == "economic_support_not_met":
        note = (
            f"ES1={es1['annualized_arithmetic']:.4%}/年（阈值 +1.0pp，未达标）；"
            f"C1 CAGR={es4:.4%}，基准 CAGR={bench_metrics.get('cagr', float('nan')):.4%}，"
            f"CAGR 差={es4 - bench_metrics.get('cagr', float('nan')):.4%}；"
            f"信号层 top6 相对基准 {diagnostics['top6_vs_benchmark']['annualized_arithmetic']:.4%}/年"
            f"（t={diagnostics['top6_vs_benchmark']['t_stat']:.2f}），"
            f"RankIC={diagnostics['rank_ic']['mean_std_residual']['mean_ic']:.4f}"
            f"（NW t={diagnostics['rank_ic']['mean_std_residual']['t_stat_nw']:.2f}）。"
            "按五级优先表第 3 行作资源停止；ES3/ES5 与表二/表三未计算，"
            "状态记为未知，不冒充全 ES 达标。"
        )
    else:
        note = "ES1~ES4 达标，但 ES3/ES5 与表二/表三尚未计算；不得前进，标记待补。"
    verdict = {
        "schema": "p7_c1_prescreen_verdict/v1",
        "run_id": args.run_id,
        "decision": decision,
        "reason_code": reason,
        "implementation_valid": implementation_valid,
        "economic_result_valid": implementation_valid and not limited,
        "limited": limited,
        "limitations": limitations,
        "missing_price_value_ratio": missing_ratio,
        "benchmark_unresolved_weight_mean": bench_unresolved_mean,
        "implementation_deviation": False,
        "es1_annualized_net_excess": es1["annualized_arithmetic"],
        "es1_threshold": 0.01,
        "es1_pass": es1_pass,
        "es2_cost_drag": es2,
        "es2_threshold": 0.02,
        "es2_pass": es2_pass,
        "es4_cagr": es4,
        "es4_threshold": 0.0,
        "es4_pass": es4_pass,
        "es3_status": "not_evaluated_unknown",
        "es5_status": "not_evaluated_unknown",
        "alpha_consumed": False,
        "not_admission": True,
        "scope_deviation": True,
        "boundaries": BOUNDARIES,
        "note": note,
    }
    _write_json(out_root / "verdict.json", verdict)

    artifacts = [
        "c1_nav.csv",
        "c1_monthly.csv",
        "c1_trades.csv",
        "benchmark.csv",
        "matched_benchmark.csv",
        "signal_test.csv",
        "metrics.json",
        "signal_diagnostics.json",
        "verdict.json",
    ]
    shas = {name: _sha256_file(out_root / name) for name in artifacts}
    code_files = sorted((ROOT / "scripts" / "v2_p7" / "c1").glob("*.py"))
    code_sha = {path.name: _sha256_file(path) for path in code_files}
    dependency_sha = {
        "scripts/v2_p7/t1/dataplane.py": _sha256_file(ROOT / "scripts/v2_p7/t1/dataplane.py"),
        "scripts/v2_p7/t1/execution.py": _sha256_file(ROOT / "scripts/v2_p7/t1/execution.py"),
    }
    protocol_path = ROOT / "docs/plans/v2/p7_b_c1_rule_based_review_package_20261114.md"
    input_identity = {
        "clean/daily": metadata_digest(ROOT / "data/clean/daily"),
        "clean/daily_basic": metadata_digest(ROOT / "data/clean/daily_basic"),
    }
    dividend_dir = ROOT / "data/raw/dividend"
    if dividend_dir.exists():
        input_identity["raw/dividend"] = metadata_digest(dividend_dir)
    manifest = {
        "schema": "p7_c1_prescreen_manifest/v1",
        "run_id": args.run_id,
        "task": "P7 B 区 C1（RM1-IN）规则型探索性预筛",
        "exploratory": True,
        "not_prereg": True,
        "not_admission": True,
        "alpha_consumed": False,
        "params": {
            "cutoff": args.cutoff,
            "start": args.start,
            "principal": args.principal,
            "top_n": TOP_N,
            "formation_months": FORMATION_MONTHS,
            "min_valid_months": MIN_VALID_MONTHS,
            "min_ticket": MIN_TICKET,
            "cost": {
                "commission_rate": 0.0001954,
                "min_commission": 5.0,
                "stamp_tax": 0.0005,
                "slippage": 0.0005,
            },
            "benchmark": "same_domain_equal_weight_gross",
            "es1_threshold": 0.01,
            "es2_threshold": 0.02,
        },
        "data_state": {
            "panel_start": panel.days[0],
            "panel_end": panel.days[-1],
            "days": panel.n_days,
            "stocks": panel.n_stocks,
            "load_stats": panel.load_stats,
            "input_sha256": {
                "sw_industry_pit.parquet": _sha256_file(
                    ROOT / "data/clean/sw_industry_pit.parquet"
                ),
                "stock_basic.parquet": _sha256_file(ROOT / "data/raw/stock_basic.parquet"),
                "trade_cal.parquet": _sha256_file(ROOT / "data/raw/trade_cal.parquet"),
            },
        },
        "code_sha256": code_sha,
        "dependencies_sha256": dependency_sha,
        "protocol_sha256": _sha256_file(protocol_path),
        "input_identity": input_identity,
        "validity": result.validity,
        "scope_deviation": True,
        "elapsed_seconds": round(time.time() - started, 3),
        "written_files_sha256": shas,
        "verdict": decision,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(out_root / "manifest.json", manifest)
    verify = {
        "schema": "p7_c1_prescreen_verify/v1",
        "all_equal": True,
        "checked_files": len(shas),
        "mismatches": [],
        "missing": [],
    }
    for name, expected in shas.items():
        actual = _sha256_file(out_root / name)
        if actual != expected:
            verify["all_equal"] = False
            verify["mismatches"].append({"file": name, "expected": expected, "actual": actual})
    _write_json(out_root / "verify.json", verify)
    runbook = (
        "# C1 规则型探索性预筛 · 运行手册\n\n"
        f"- run_id: `{args.run_id}`\n"
        f"- 命令: `python scripts/v2_p7/c1/run_prescreen.py --run-id {args.run_id}`\n"
        f"- 窗口: {panel.days[0]} ~ {panel.days[-1]}；数据截止 {args.cutoff}\n"
        f"- 账户: 6 槽 / {args.principal:.0f} 元 / min_ticket {MIN_TICKET:.2f}\n"
        f"- 判定: {decision}（{reason}）\n"
        f"- 实现有效性: {implementation_valid}"
        f"（signal_day_trades={result.validity.get('signal_day_trades')}，"
        f"negative_cash_days={result.validity.get('negative_cash_days')}，"
        f"missing_price_days={result.validity.get('missing_price_days')}）\n"
        f"- ES1 年化净超额: {es1['annualized_arithmetic']:.4%}（阈值 +1.0pp）\n"
        f"- ES2 成本拖累: {es2:.4%}（阈值 ≤2.0pp）\n"
        f"- ES4 C1 CAGR: {es4:.4%}（阈值 >0）\n"
        "- 边界: exploratory / not_prereg / not_admission；alpha_consumed=false。\n"
        "- 表二/表三与 ES3/ES5: 未计算，状态未知；scope_deviation=true。\n"
        "- 若 ES1~ES4 未全过，按五级优先表停止；达标也不得在缺表二/表三时前进。\n"
    )
    (out_root / "runbook.md").write_text(runbook, encoding="utf-8")
    print(
        json.dumps(
            {"run_root": str(out_root), "decision": decision, "es1": es1, "es2": es2, "es4": es4},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
