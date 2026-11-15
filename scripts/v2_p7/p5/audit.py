# -*- coding: utf-8 -*-
"""P5 T2/T3 历史复核：只核账与组合恒等式，不新选股、不加新门槛。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v2_p7.p4.data import A_NAV  # noqa: E402  （与 P4 同一 A 参照口径）
from scripts.v2_p7.p4.metrics import nav_metrics  # noqa: E402

#: P4 证据根（R01 整改后必须指向**新**根；旧根只读保留）
_P4_ENV = Path(
    os.environ.get(
        "P7_P4_ROOT",
        str(ROOT / "data/frozen_reference/v2_p7/p7_idx_in300_p4_20261114_r1"),
    )
)
P4 = _P4_ENV if _P4_ENV.is_absolute() else ROOT / _P4_ENV
CLOSURE = (
    ROOT
    / "data/frozen_reference/v2_p7/p7_idx_in300_missing_data_closure_20261114_r1"
    / "closure.json"
)

_OUT_ENV = Path(
    os.environ.get(
        "P5_AUDIT_OUT",
        str(ROOT / "data/frozen_reference/v2_p7/p7_idx_in300_p5_20261114_r1"),
    )
)
OUT = _OUT_ENV if _OUT_ENV.is_absolute() else ROOT / _OUT_ENV


def money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def audit_t2(summary: dict) -> dict:
    fills = pd.read_csv(
        P4 / "fills.csv", dtype={"ts_code": str, "trade_price": str, "amount": str, "fee": str}
    )
    corp = pd.read_csv(P4 / "corporate_actions.csv", dtype={"ts_code": str, "cash_amount": str})
    for column in ("trade_price", "amount", "fee"):
        fills[column] = fills[column].map(Decimal)
    corp["cash_amount"] = corp["cash_amount"].map(Decimal)
    notional_errors = []
    for row in fills.itertuples():
        expected = money(row.trade_price * row.shares)
        if expected != row.amount:
            notional_errors.append(
                {"fill_id": row.fill_id, "expected": str(expected), "actual": str(row.amount)}
            )
    buy_shares = fills.loc[fills["side"].eq("buy")].groupby("ts_code")["shares"].sum()
    sell_shares = fills.loc[fills["side"].eq("sell")].groupby("ts_code")["shares"].sum()
    dividend_shares = (
        corp.assign(delta=corp["shares_after"] - corp["shares_before"])
        .groupby("ts_code")["delta"]
        .sum()
    )
    share_mismatch = {
        code: int(buy_shares.get(code, 0) + dividend_shares.get(code, 0) - sell_shares.get(code, 0))
        for code in set(buy_shares.index) | set(sell_shares.index)
        if int(buy_shares.get(code, 0) + dividend_shares.get(code, 0) - sell_shares.get(code, 0))
        != 0
    }
    buy_amount = fills.loc[fills["side"].eq("buy"), "amount"].sum()
    sell_amount = fills.loc[fills["side"].eq("sell"), "amount"].sum()
    fees = fills["fee"].sum()
    dividends = corp["cash_amount"].sum()
    cash_end = money(Decimal("300000") - buy_amount + sell_amount - fees + dividends)
    expected_end = Decimal(str(summary["table1"]["metrics"]["end_nav"]))
    fill_ids_unique = fills["fill_id"].is_unique
    final_positions = summary["table1"]["final_positions"]
    checks = {
        "fill_ids_unique": bool(fill_ids_unique),
        "notional_errors": notional_errors,
        "share_mismatch": share_mismatch,
        "all_positions_closed": final_positions == [],
        "cash_end": str(cash_end),
        "expected_end_nav": str(expected_end),
        "cash_identity_diff": str(money(cash_end - expected_end)),
        "buy_fills": int(fills["side"].eq("buy").sum()),
        "sell_fills": int(fills["side"].eq("sell").sum()),
        "fees_cny": str(money(fees)),
        "dividends_cash_cny": str(money(dividends)),
        "stock_dividend_shares": {
            str(k): int(v) for k, v in dividend_shares.items() if int(v) != 0
        },
    }
    checks["pass"] = bool(
        checks["fill_ids_unique"]
        and not notional_errors
        and not share_mismatch
        and checks["all_positions_closed"]
        and money(cash_end - expected_end) == Decimal("0.00")
    )
    return checks


def audit_t3(summary: dict) -> dict:
    levels = pd.read_parquet(P4 / "table3_daily_nav.parquet")
    identity_error = (levels["combined"] - levels["a_leg"] - levels["candidate_leg"]).abs().max()
    recomputed = {
        "combined": nav_metrics(levels["combined"]),
        "a_only": nav_metrics(levels["a_only"]),
        "candidate_leg": nav_metrics(levels["candidate_leg"]),
    }
    reported = summary["table3"]
    diffs = {}
    for name, key in (
        ("combined", "combined_metrics"),
        ("a_only", "a_only_metrics"),
        ("candidate_leg", "candidate_leg_metrics"),
    ):
        for metric in ("end_nav", "cagr", "max_drawdown", "annual_vol", "sharpe_rf0"):
            diffs[f"{name}.{metric}"] = abs(
                float(recomputed[name][metric]) - float(reported[key][metric])
            )
    d_pp = (recomputed["combined"]["max_drawdown"] - recomputed["a_only"]["max_drawdown"]) * 100.0
    p_pp = (recomputed["combined"]["cagr"] - recomputed["a_only"]["cagr"]) * 100.0
    es3_pass = d_pp >= -1.0 or (-3.0 <= d_pp < -1.0 and p_pp > 0.0)
    return {
        "max_identity_error": float(identity_error),
        "metric_abs_diff_max": max(diffs.values()),
        "recomputed_d_pp": d_pp,
        "recomputed_p_pp": p_pp,
        "recomputed_es3_pass": bool(es3_pass),
        "reported_d_pp": reported["delta"]["d_maxdd_pp"],
        "reported_p_pp": reported["delta"]["p_cagr_pp"],
        "pass": bool(
            identity_error <= 1e-9
            and max(diffs.values()) <= 1e-9
            and abs(d_pp - reported["delta"]["d_maxdd_pp"]) <= 1e-9
            and abs(p_pp - reported["delta"]["p_cagr_pp"]) <= 1e-9
        ),
    }


def evaluate_status(t2: dict, t3: dict, es: dict) -> dict:
    """P5 整体状态必须合取 T2、T3、ES1~ES5，并核对 ES3 复算一致性。"""
    required = ("ES1", "ES2", "ES3", "ES4", "ES5")
    es_pass = all(bool(es[key]["pass"]) for key in required)
    es3_consistent = bool(t3["recomputed_es3_pass"]) == bool(es["ES3"]["pass"])
    accounting_pass = bool(t2["pass"] and t3["pass"])
    return {
        "accounting_pass": accounting_pass,
        "es_pass": es_pass,
        "es3_consistent": es3_consistent,
        "status": "pass" if accounting_pass and es_pass and es3_consistent else "fail",
    }


def main() -> None:
    if (OUT / "audit.json").exists() and not os.environ.get("P5_ALLOW_REPLAY"):
        raise RuntimeError(f"P5审计根已存在，拒绝覆盖: {OUT}；请设置新的 P5_AUDIT_OUT")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((P4 / "summary.json").read_text(encoding="utf-8"))
    closure = json.loads(CLOSURE.read_text(encoding="utf-8"))
    t2 = audit_t2(summary)
    t3 = audit_t3(summary)
    es_from_p4 = {
        key: summary["decision"]["es"][key] for key in ("ES1", "ES2", "ES3", "ES4", "ES5")
    }
    status = evaluate_status(t2, t3, es_from_p4)
    checks = {
        "schema": "p7_idx_in300_p5_accounting/v3",
        "candidate": "IDX-IN300",
        "role": "accounting_only",
        "role_note": (
            "本根只做逐笔/期末账务与静态组合恒等式复核（T2/T3 的会计部分）；"
            "经济准入由正式 T2/T3 程序（scripts/v2_p7/f2）另行预登记与判定，"
            "不得把本根 status=pass 读作正式准入。"
        ),
        "p4_root": str(P4.relative_to(ROOT).as_posix()),
        "a_nav": str(Path(A_NAV).relative_to(ROOT).as_posix()),
        "a_nav_sha256": hashlib.sha256(Path(A_NAV).read_bytes()).hexdigest(),
        "status": status["status"],
        "status_parts": {
            "accounting_pass": status["accounting_pass"],
            "es_pass": status["es_pass"],
            "es3_consistent": status["es3_consistent"],
        },
        "t2": t2,
        "t3": t3,
        "known_suspension": {
            "records": closure["records"],
            "all_suspend_S": closure["all_records_are_suspend_S"],
            "nav_changed": closure["p4_nav_changed"],
            "accepted_by_user": True,
        },
        "es_from_p4": es_from_p4,
        "capacity": {
            "1pct_cny": summary["table1"]["capacity"]["1pct_cny"],
            "role": "informational_only",
            "no_gate": True,
        },
        "alpha_consumed": False,
        "note": "P5整体状态合取T2/T3账务与ES1~ES5；不再只看账务恒等式；容量仅报告。",
    }
    (OUT / "audit.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
