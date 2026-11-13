# -*- coding: utf-8 -*-
"""P7 T6 只读校验器：**只用落盘产物**重算裁定所需的每个数字。

用途
====
1. runner 在产物落盘后调用（结果并入 G4：``artifact_recompute``），
   避免「用实现证明实现」的循环论证；
2. 外部复核者独立运行：``python scripts/v2_p7/t6/verify.py --run-root <path>``。

校验项（**证据可复算性**）
=========================
- **NAV**：``calculate_nav(total_value, PC)`` 逐日与 ``arms/*/daily_*.parquet`` 的 ``nav`` 逐位一致；
- **链式指标**：``calculate_chain_metrics`` 复算 == ``metrics.json`` 登记值（1e-12）；
- **自然日年化**：``calculate_annualized_return`` 复算 == 登记值（1e-9 百分点）；
- **费用/换手**：交易行汇总 == ``fees.total`` / ``turnover.turnover``（1e-6）；
- **配对差与归因**：``delta == treatment − control``、``ΔTR_gross = ΔTR − ΔTR_fee`` 等闭式复核；
- **分月/分年与 C5**：从 ``daily_*.parquet`` 的 NAV 独立重算月度/年度收益，再复算集中度、
  留出 top3 月的 Δ、分年符号三项，与 ``verdict.json`` 的 C5 载荷比对；
- **噪声带**：``build_noise``（同 seed/block/B）复算 == ``metrics.json.noise`` 主带（1e-12）。
  **注意**：该项是**同实现的确定性重放**（用于证明证据未被改写、随机源可复现），
  不构成对 bootstrap 公式的独立验证；公式与唯一入口的同源性由「同折掩码 + cagr 指数」的
  单测与 G4 的内存复算共同保证；
- **预登记**：代码常量摘要 == ``calibration.json.criteria_digest``，且 criteria/delta/band/stability 逐项相同；
- **产物 SHA**（manifest 存在时）：逐件复核 ``outputs``。

边界：本模块只读，不写任何文件；不接受窗口裁剪参数（重算必须覆盖全窗）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - 脚本入口
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from scripts.v2_p7.common import sha256_file  # noqa: E402
from scripts.v2_p7.t6 import calibration  # noqa: E402
from scripts.v2_p7.t6.noise import build_noise  # noqa: E402
from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics  # noqa: E402
from src.lazybull.v2.core.accounting.nav import calculate_nav  # noqa: E402
from src.lazybull.v2.hosts.paper.performance import (  # noqa: E402
    calculate_annualized_return,
)

VERIFY_SCHEMA = "p7_t6_verify/v1"

#: 复算容差（与 G4 同值；``0.0`` 表示要求逐位相等）
TOLERANCE: Mapping[str, float] = {
    "nav": 1e-12,
    "chain_metric": 1e-12,
    "annualized_pct": 1e-9,
    "fee_total": 1e-6,
    "turnover": 1e-12,
    "noise": 1e-12,
    "paired_delta": 1e-12,
    "period": 1e-12,
}

ARMS = ("vvh", "vvs")


def verify_run(run_root: Path) -> Dict[str, Any]:
    """对已完成证据根做只读复算（返回 ``{ok, checks, mismatches}``）。"""
    root = Path(run_root)
    calibration_doc = _read_json(root / "calibration.json")
    metrics = _read_json(root / "metrics.json")
    verdict = _read_json(root / "verdict.json")
    physical_capital = float(calibration.PHYSICAL_CAPITAL)
    frames = {arm: _read_arm(root, arm) for arm in ARMS}
    checks: Dict[str, Any] = {}
    checks["calibration"] = _verify_calibration(calibration_doc, verdict)
    checks["arms"] = {
        arm: _verify_arm(frames[arm], metrics["arms"][arm], physical_capital) for arm in ARMS
    }
    checks["paired"] = _verify_paired(metrics)
    checks["monthly_annual"] = _verify_periods(frames, metrics, verdict)
    checks["noise"] = _verify_noise(frames, metrics, calibration_doc)
    checks["manifest"] = _verify_manifest(root)
    mismatches = _collect(checks)
    return {
        "schema": VERIFY_SCHEMA,
        "run_root": str(root),
        "ok": not mismatches,
        "tolerance": dict(TOLERANCE),
        "checks": checks,
        "mismatches": mismatches,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"证据根缺件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_arm(root: Path, arm: str) -> Dict[str, Any]:
    daily = pd.read_parquet(root / "arms" / arm / f"daily_{arm}.parquet")
    trades = pd.read_parquet(root / "arms" / arm / f"trades_{arm}.parquet")
    return {"daily": daily, "trades": trades}


def _verify_calibration(doc: Mapping[str, Any], verdict: Mapping[str, Any]) -> Dict[str, Any]:
    payload = calibration.payload()
    digest = calibration.digest()
    same_criteria = list(doc.get("criteria") or ()) == list(payload["criteria"])
    same_delta = dict(doc.get("delta") or {}) == dict(payload["delta"])
    same_band = dict(doc.get("band") or {}) == dict(payload["band"])
    same_stability = dict(doc.get("stability") or {}) == dict(payload["stability"])
    return {
        "criteria_digest": str(doc.get("criteria_digest") or ""),
        "code_digest": digest,
        "digest_equal": str(doc.get("criteria_digest") or "") == digest,
        "criteria_equal": same_criteria,
        "delta_equal": same_delta,
        "band_equal": same_band,
        "stability_equal": same_stability,
        "verdict_digest": str(verdict.get("calibration_digest") or ""),
        "verdict_digest_equal": str(verdict.get("calibration_digest") or "") == digest,
        "ok": bool(
            str(doc.get("criteria_digest") or "") == digest
            and same_criteria
            and same_delta
            and same_band
            and same_stability
            and str(verdict.get("calibration_digest") or "") == digest
        ),
    }


def _verify_arm(frame: Mapping[str, Any], stored: Mapping[str, Any], pc: float) -> Dict[str, Any]:
    daily = frame["daily"]
    nav = [float(value) for value in daily["nav"]]
    recomputed_nav = [float(calculate_nav(value, pc)) for value in daily["total_value"]]
    nav_diff = max(abs(a - b) for a, b in zip(recomputed_nav, nav))
    chain = calculate_chain_metrics(
        pd.DataFrame(
            {
                "trade_date": daily["trade_date"],
                "nav": daily["nav"],
                "split_index": daily["split_index"],
            }
        )
    )
    chain_diff = {
        key: abs(float(chain[key]) - float(stored["chain_metrics"][key]))
        for key in ("total_return", "cagr", "max_drawdown", "volatility", "sharpe")
    }
    annualized = calculate_annualized_return(
        pc,
        float(daily["total_value"].iloc[-1]),
        str(daily["trade_date"].iloc[-1]),
        str(daily["trade_date"].iloc[0]),
    )
    annualized_diff = abs(float(annualized) - float(stored["annualized_natural_days_pct"]))
    fee_total = float(frame["trades"]["fee"].sum())
    fee_diff = abs(fee_total - float(stored["fees"]["total"]))
    turnover = float(frame["trades"]["market_notional"].sum()) / pc
    turnover_diff = abs(turnover - float(stored["turnover"]["turnover"]))
    ok = bool(
        nav_diff <= TOLERANCE["nav"]
        and max(chain_diff.values()) <= TOLERANCE["chain_metric"]
        and annualized_diff <= TOLERANCE["annualized_pct"]
        and fee_diff <= TOLERANCE["fee_total"]
        and turnover_diff <= TOLERANCE["turnover"]
    )
    return {
        "nav_max_abs_diff": nav_diff,
        "chain_metric_abs_diff": chain_diff,
        "annualized_pct_abs_diff": annualized_diff,
        "fee_total_abs_diff": fee_diff,
        "turnover_abs_diff": turnover_diff,
        "ok": ok,
    }


def _verify_paired(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """配对差与费用归因的闭式复算（防止换号/分母错而无门可发现）。"""
    paired = metrics["paired"]
    deltas: Dict[str, float] = {}
    for key in ("total_return", "max_drawdown", "volatility", "cagr", "sharpe", "nav_end"):
        item = paired.get(key) or {}
        expected = float(item.get("treatment") or 0.0) - float(item.get("control") or 0.0)
        deltas[key] = abs(expected - float(item.get("delta") or 0.0))
    attribution = paired["attribution"]
    pc = float(attribution["physical_capital"])
    fee_delta = float(paired["fees"]["treatment"]) - float(paired["fees"]["control"])
    closure = {
        "physical_capital_match": bool(abs(pc - float(calibration.PHYSICAL_CAPITAL)) <= 0.0),
        "delta_fee_over_pc": abs(float(attribution["delta_fee_over_pc"]) - fee_delta / pc),
        "delta_tr_fee_sign": abs(
            float(attribution["delta_tr_fee"]) + float(attribution["delta_fee_over_pc"])
        ),
        "delta_tr_gross": abs(
            float(attribution["delta_tr_gross"])
            - (float(attribution["delta_tr"]) - float(attribution["delta_tr_fee"]))
        ),
        "delta_turnover": abs(
            float(attribution["delta_turnover"])
            - (float(paired["turnover"]["treatment"]) - float(paired["turnover"]["control"]))
        ),
    }
    max_delta = max(deltas.values()) if deltas else 0.0
    max_closure = max(closure[key] for key in closure if key != "physical_capital_match")
    return {
        "delta_max_abs_diff": max_delta,
        "closure_abs_diff": closure,
        "ok": bool(
            max_delta <= TOLERANCE["paired_delta"]
            and closure["physical_capital_match"]
            and max_closure <= TOLERANCE["paired_delta"]
        ),
    }


def _verify_periods(
    frames: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    verdict: Mapping[str, Any],
) -> Dict[str, Any]:
    """分月/分年序列与 C5 三项判据的独立复算（与 verdict 模块不同代码路径）。"""
    monthly = {arm: _period_returns(frames[arm]["daily"], 6) for arm in ARMS}
    annual = {arm: _period_returns(frames[arm]["daily"], 4) for arm in ARMS}
    months = sorted(set(monthly["vvh"]) | set(monthly["vvs"]))
    monthly_delta = {
        month: float(monthly["vvs"].get(month, 0.0)) - float(monthly["vvh"].get(month, 0.0))
        for month in months
    }
    years = sorted(set(annual["vvh"]) | set(annual["vvs"]))
    annual_delta = {
        year: float(annual["vvs"].get(year, 0.0)) - float(annual["vvh"].get(year, 0.0))
        for year in years
    }
    stored_month = {str(row["month"]): float(row["delta"]) for row in metrics["monthly_paired"]}
    stored_year = {str(row["year"]): float(row["delta"]) for row in metrics["by_year_paired"]}
    month_diff = max(abs(monthly_delta[key] - stored_month.get(key, 0.0)) for key in monthly_delta)
    year_diff = max(abs(annual_delta[key] - stored_year.get(key, 0.0)) for key in annual_delta)
    c5 = next(item for item in verdict["criteria"] if item["id"] == "C5")["values"]
    magnitudes = sorted((abs(value) for value in monthly_delta.values()), reverse=True)
    total_abs = float(sum(magnitudes))
    concentration = float(sum(magnitudes[:3]) / total_abs) if total_abs > 0.0 else 0.0
    excluded = sorted(monthly_delta, key=lambda key: abs(monthly_delta[key]), reverse=True)[:3]
    keep = [month for month in months if month not in set(excluded)]
    control = 1.0
    treatment = 1.0
    for month in keep:
        control *= 1.0 + float(monthly["vvh"].get(month, 0.0))
        treatment *= 1.0 + float(monthly["vvs"].get(month, 0.0))
    positive = sum(1 for value in annual_delta.values() if value > 0.0)
    negative = sum(1 for value in annual_delta.values() if value < 0.0)
    non_zero = positive + negative
    share = max(positive, negative) / non_zero if non_zero else None
    checks = {
        "monthly_series_abs_diff": month_diff,
        "annual_series_abs_diff": year_diff,
        "concentration_abs_diff": abs(concentration - float(c5["concentration"])),
        "loo_delta_abs_diff": abs((treatment - control) - float(c5["loo_delta"])),
        "excluded_months_equal": sorted(excluded) == sorted(c5["excluded_months"]),
        "year_positive_equal": positive == int(c5["year"]["positive"]),
        "year_negative_equal": negative == int(c5["year"]["negative"]),
        "year_share_equal": share is not None
        and abs(float(share) - float(c5["year"]["same_sign_share"])) <= 1e-12,
    }
    ok = bool(
        month_diff <= TOLERANCE["period"]
        and year_diff <= TOLERANCE["period"]
        and checks["concentration_abs_diff"] <= TOLERANCE["period"]
        and checks["loo_delta_abs_diff"] <= TOLERANCE["period"]
        and checks["excluded_months_equal"]
        and checks["year_positive_equal"]
        and checks["year_negative_equal"]
        and checks["year_share_equal"]
    )
    return {**checks, "daily_counts": {arm: len(monthly[arm]) for arm in ARMS}, "ok": ok}


def _period_returns(daily: pd.DataFrame, digits: int) -> Dict[str, float]:
    """按 ``YYYYMM`` / ``YYYY`` 取期末净值并算收益率（首期以期内首个净值为基期）。"""
    frame = pd.DataFrame(
        {"key": daily["trade_date"].str[:digits], "nav": daily["nav"].astype(float)}
    )
    grouped = frame.groupby("key")["nav"]
    first = grouped.first()
    last = grouped.last()
    returns: Dict[str, float] = {}
    previous = None
    for key in last.index:
        base = float(first[key]) if previous is None else float(previous)
        returns[str(key)] = float(last[key]) / base - 1.0 if base != 0.0 else 0.0
        previous = last[key]
    return returns


def _verify_noise(
    frames: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    calibration_doc: Mapping[str, Any],
) -> Dict[str, Any]:
    band = dict(calibration_doc.get("band") or calibration.BAND)
    paired = metrics["paired"]
    recomputed = build_noise(
        nav={arm: [float(value) for value in frames[arm]["daily"]["nav"]] for arm in ARMS},
        split_index=[int(value) for value in frames["vvh"]["daily"]["split_index"]],
        point_deltas={
            "total_return": float(paired["total_return"]["delta"]),
            "max_drawdown": float(paired["max_drawdown"]["delta"]),
            "cagr": float(paired["cagr"]["delta"]),
            "volatility": float(paired["volatility"]["delta"]),
            "sharpe": float(paired["sharpe"]["delta"]),
        },
        calibration={**calibration.payload(), "band": band},
        deltas_by_year=[float(row["delta"]) for row in metrics["by_year_paired"]],
        deltas_by_month=[float(row["delta"]) for row in metrics["monthly_paired"]],
    )
    stored = metrics.get("noise") or {}
    diffs: Dict[str, float] = {}
    for name, item in (recomputed["primary"]["metrics"]).items():
        target = ((stored.get("primary") or {}).get("metrics") or {}).get(name) or {}
        for key in ("boot_mean", "low", "high", "se"):
            if target.get(key) is None or item.get(key) is None:
                diffs[f"{name}.{key}"] = float("inf")
                continue
            diffs[f"{name}.{key}"] = abs(float(item[key]) - float(target[key]))
    days_equal = dict(recomputed.get("days") or {}) == dict(stored.get("days") or {})
    max_diff = max(diffs.values()) if diffs else 0.0
    return {
        "max_abs_diff": max_diff,
        "diffs": diffs,
        "days_equal": days_equal,
        "stored_present": bool(stored),
        "ok": bool(stored and max_diff <= TOLERANCE["noise"] and days_equal),
    }


def _verify_manifest(root: Path) -> Dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        return {"present": False, "ok": True, "note": "manifest 尚未写出（runner 内调用）"}
    manifest = _read_json(path)
    outputs = manifest.get("outputs") or {}
    mismatches: List[Dict[str, Any]] = []
    for rel, expected in sorted(outputs.items()):
        target = root / str(rel)
        if not target.exists():
            mismatches.append({"file": str(rel), "status": "missing"})
            continue
        actual = sha256_file(target)
        if actual != str(expected):
            mismatches.append({"file": str(rel), "expected": str(expected), "actual": actual})
    return {
        "present": True,
        "outputs": len(outputs),
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def _collect(checks: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    if not checks["calibration"]["ok"]:
        problems.append("预登记复算不一致（digest/criteria/delta/band/stability/verdict 之一）")
    for arm, payload in sorted(checks["arms"].items()):
        if not payload["ok"]:
            diffs = {key: value for key, value in payload.items() if key != "ok"}
            problems.append(f"{arm} 落盘复算不一致：{diffs}")
    if not checks["paired"]["ok"]:
        problems.append(f"配对差/归因复算不一致：{checks['paired']}")
    if not checks["monthly_annual"]["ok"]:
        problems.append(f"分月/分年与 C5 复算不一致：{checks['monthly_annual']}")
    if not checks["noise"]["ok"]:
        problems.append(f"噪声带复算不一致：max_abs_diff={checks['noise']['max_abs_diff']!r}")
    if not checks["manifest"]["ok"]:
        problems.append(f"manifest outputs SHA 不符：{checks['manifest']['mismatches'][:3]}")
    return problems


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P7 T6 证据只读复算校验")
    parser.add_argument(
        "--run-root", required=True, help="证据根（data/paper_shadow/p7_testbed/<run_id>）"
    )
    args = parser.parse_args(argv)
    report = verify_run(Path(args.run_root))
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - 脚本入口
    raise SystemExit(main())
