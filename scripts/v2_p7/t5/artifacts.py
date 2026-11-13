# -*- coding: utf-8 -*-
"""P7 T5 证据落盘（九件套 + 逐日帧）与各件载荷装配。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from scripts.v2_p7.common import sha256_file, write_json
from scripts.v2_p7.t4.runner import _flatten_row
from scripts.v2_p7.t5 import scenarios as scenario_mod
from scripts.v2_p7.t5.host import T5HostResult
from scripts.v2_p7.t5.params import T5Params, quota_ratio


def write_artifacts(
    *,
    params: T5Params,
    run_root: Path,
    results: Mapping[str, T5HostResult],
    cases: Sequence[Any],
    parity: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """写九件套 + 逐日帧（返回逐件 SHA/行数）。"""
    written: Dict[str, Dict[str, Any]] = {}
    written["quota.json"] = write_json_counted(run_root / "quota.json", quota_payload(results))
    written["credit_ledger.json"] = write_json_counted(
        run_root / "credit_ledger.json", credit_payload(results)
    )
    written["sleeve_balance.json"] = write_json_counted(
        run_root / "sleeve_balance.json", balance_payload(results)
    )
    written["dual_caliber.json"] = write_json_counted(
        run_root / "dual_caliber.json", dual_payload(params, results)
    )
    written["identity.json"] = write_json_counted(
        run_root / "identity.json", identity_payload(results)
    )
    written["t4_parity.json"] = write_json_counted(
        run_root / "t4_parity.json", {**dict(parity), "vvh_reference": dict(reference)}
    )
    written["legacy_sha.json"] = write_json_counted(
        run_root / "legacy_sha.json",
        {
            "schema": "p7_t5_legacy_sha/v1",
            "src_legacy": dict(params.legacy_sha),
            "t4_modules": {key: dict(value) for key, value in sorted(params.t4_module_sha.items())},
            "note": "T4 模块逐件 SHA 用于证明本 task 未改 T4 语义（不重跑、不覆盖 r31）。",
        },
    )
    written["scenarios/constructed.json"] = write_json_counted(
        run_root / "scenarios" / "constructed.json", scenario_mod.summarize_cases(cases)
    )
    written["scenarios/synthetic.json"] = write_json_counted(
        run_root / "scenarios" / "synthetic.json",
        {
            "schema": "p7_t5_scenarios/v1",
            "kind": scenario_mod.SYNTHETIC,
            "cases": [item.payload() for item in cases if item.kind == scenario_mod.SYNTHETIC],
        },
    )
    written["scenarios/constructed_on_history.json"] = write_json_counted(
        run_root / "scenarios" / "constructed_on_history.json",
        {
            "schema": "p7_t5_scenarios/v1",
            "kind": scenario_mod.ON_HISTORY,
            "cases": [item.payload() for item in cases if item.kind == scenario_mod.ON_HISTORY],
            "note": "本代构造例全部为 synthetic；历史口径扩展在此分列，不与合成例混报。",
        },
    )
    for name, result in sorted(results.items()):
        written[f"daily_{name}.parquet"] = write_frames(
            run_root / f"daily_{name}.parquet", result.daily_rows
        )
        written[f"conservation_{name}.parquet"] = write_frames(
            run_root / f"conservation_{name}.parquet", result.conservation_rows
        )
        written[f"trades_{name}.parquet"] = write_frames(
            run_root / f"trades_{name}.parquet", result.trades
        )
        written[f"quota_daily_{name}.parquet"] = write_frames(
            run_root / f"quota_daily_{name}.parquet", result.quota_daily
        )
        written[f"sleeve_balance_{name}.parquet"] = write_frames(
            run_root / f"sleeve_balance_{name}.parquet", result.balance_rows
        )
    vvs = results.get("vvs")
    if vvs is not None:
        written["credit_lots.parquet"] = write_frames(
            run_root / "credit_lots.parquet", vvs.credit_payload["lots"]
        )
        written["credit_events.parquet"] = write_frames(
            run_root / "credit_events.parquet", vvs.credit_payload["events"]
        )
    return written


def write_json_counted(path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    digest = write_json(path, dict(payload))
    return {"sha256": digest, "bytes": int(path.stat().st_size), "file": path.name}


def write_frames(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """写帧（嵌套 dict/list 列规范化为 JSON 字符串，Parquet 无空 struct）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([_flatten_row(row) for row in rows])
    frame.to_parquet(path, index=False)
    return {"sha256": sha256_file(path), "rows": int(len(frame)), "file": path.name}


def quota_payload(results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    arms: Dict[str, Any] = {}
    for name in ("vvh", "vvs", "parity"):
        result = results.get(name)
        if result is None:
            continue
        rows = result.quota_rows
        days = sorted({str(row["trade_date"]) for row in rows}) or list(result.days)
        binding_days = sorted({str(row["trade_date"]) for row in rows if row["cap_binding"]})
        arms[name] = {
            "mode": dict(result.mode),
            "sleeve_days": len(rows),
            "days": len(days),
            "cap_binding_sleeve_days": sum(1 for row in rows if row["cap_binding"]),
            "not_binding_days": len(days) - len(binding_days),
            "denied_shares_total": int(sum(int(row["denied_shares"]) for row in rows)),
            "rows": rows,
            "daily": [
                {
                    "trade_date": item["trade_date"],
                    "gross_target": item["gross_target"],
                    "arm_target": item["arm_target"],
                    "cap_binding_sleeves": item["cap_binding_sleeves"],
                    "quota_enabled": item["quota_enabled"],
                    "credit_enabled": item["credit_enabled"],
                }
                for item in result.quota_daily
            ],
        }
    return {
        "schema": "p7_t5_quota/v1",
        "quota_ratio": dict(sorted(quota_ratio().items())),
        "target_source": (
            "T2.5 ``targets/daily_{arm}.parquet`` 的 ``target_notional``；"
            "账户级 G = Σ_arm（同一 A 信号的双相位副本），ρ 为资本份额（Σρ=1）"
        ),
        "rule": (
            "G → D=max(0,G−MV) → R=max(0,ρG−MV) → 逐袖整手缩放 → A ≤ min(D, R+L) 且 "
            "A+Fee(A) ≤ cash_open+L；原始卖出不受配额约束"
        ),
        "arms": arms,
    }


def credit_payload(results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    arms: Dict[str, Any] = {}
    for name in ("vvs", "vvh", "parity"):
        result = results.get(name)
        if result is None:
            continue
        arms[name] = {
            "mode": dict(result.mode),
            "payload": result.credit_payload,
            "balance": result.credit_balance,
            "zero_pnl": result.zero_pnl,
            "daily": result.credit_daily,
        }
    return {
        "schema": "p7_t5_credit/v1",
        "arms": arms,
        "capacity_rule": {
            "idle_formula": (
                "出借闲置 = (cash_open − 自有买入负债 − 既有出借余额) / 2；"
                "系数 2 是 ``cash_i + credit_i >= 0`` 的代数推论（贷出 L 后 cash=cash_open−L、"
                "credit=−(reserve+L) ⇒ 2L <= cash_open − reserve − need），不是保守 haircut"
            ),
            "implication": (
                "出借方须以现金为其**全部**应收（含已死局债权）全额背书，"
                "故可贷额度上限为盈余的一半；两袖同日普遍同时缺口时 idle=0 ⇒ 机制可能近乎空转"
            ),
            "capacity_note": (
                "本代证据逐日登记申请额、授予额、剩余闲置与未满足缺口；"
                "借用规模**不得**被读作资金约束缓解的经济结论（alpha_consumed=false）"
            ),
        },
        "rule": (
            "借入为正、出借为负；L 只按本日转账一次入账；credit_outstanding 不参与授权；"
            "T+1 未还即永久 deadlocked；自然卖出只记 late_recovery"
        ),
        "no_interest": "本 task 无利息（逐 lot 计息为零）",
    }


def balance_payload(results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    from scripts.v2_p7.t5.state import summary_by_sleeve

    arms: Dict[str, Any] = {}
    for name, result in sorted(results.items()):
        credit_nav = [dict(row) for row in result.credit_nav_rows]
        arms[name] = {
            "mode": dict(result.mode),
            "summary": summary_by_sleeve(as_balances(result.balance_rows)),
            "gate": result.gate_payload,
            "rows": len(result.balance_rows),
            # 权威袖级序列：E = cash + MV − credit（T4 的 total_value 不含 credit）
            "credit_adjusted": {
                "rows": credit_nav,
                "invalid_nonpositive_equity_days": sum(
                    1 for row in credit_nav if bool(row.get("invalid_nonpositive_equity"))
                ),
                "note": (
                    "T4 兼容视图（dual_caliber.json 的 independent_nav）为 cash + MV，不含 credit；"
                    "T6 应使用本节的 contribution_credit_adjusted / independent_nav_credit_adjusted。"
                    "贡献始终保留**有符号**值（E/物理初资，可加）；仅当**实际 E 或调整后反事实"
                    "权益**非正时，独立指标置空并标 invalid_nonpositive_equity"
                    "（不删日、不溢出为正）。"
                    "credit=off 模式下 credit=0 ⇒ 本节与 T4 口径逐位一致。"
                ),
            },
        }
    return {
        "schema": "p7_t5_sleeve_balance/v1",
        "arms": arms,
        "rule": "E_i = cash_i + MV_i − credit_i；cash_minus_credit 与 cash_plus_credit 双字段落地",
    }


def as_balances(rows: Sequence[Mapping[str, Any]]) -> List[Any]:
    from scripts.v2_p7.t5.state import SleeveBalance

    return [
        SleeveBalance(
            trade_date=str(row["trade_date"]),
            sleeve=str(row["sleeve"]),
            cash=float(row["cash"]),
            credit=float(row["credit"]),
            market_value=float(row["market_value"]),
            equity=float(row["equity"]),
        )
        for row in rows
    ]


def dual_payload(params: T5Params, results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics

    arms: Dict[str, Any] = {}
    for name, result in sorted(results.items()):
        per_arm: Dict[str, Any] = {}
        for arm in params.arms:
            series = result.nav_series.get(arm, [])
            frame = pd.DataFrame(series)
            if frame.empty:
                continue
            valid = frame.dropna(subset=["nav"])
            metrics = (
                calculate_chain_metrics(
                    valid[["trade_date", "nav", "split_index"]].reset_index(drop=True)
                )
                if len(valid)
                else None
            )
            invalid_rows = frame[frame["independent_nav"].isna()]
            independent = (
                None
                if len(invalid_rows)
                else (
                    calculate_chain_metrics(
                        frame[["trade_date", "independent_nav", "split_index"]]
                        .rename(columns={"independent_nav": "nav"})
                        .reset_index(drop=True)
                    )
                    if len(frame)
                    else None
                )
            )
            per_arm[arm] = {
                "contribution_nav_metrics": metrics,
                "independent_nav_metrics": independent,
                "independent_status": (
                    "ok" if not len(invalid_rows) else "invalid_nonpositive_equity_day"
                ),
                "independent_invalid_dates": [
                    str(item) for item in invalid_rows["trade_date"].tolist()
                ],
                "days": int(len(frame)),
                "invalid_equity_days": int(result.invalid_days.get(arm, 0)),
            }
        arms[name] = {
            "mode": dict(result.mode),
            "sleeves": per_arm,
            "fee": result.fee_book.payload(),
            "identities": {
                "failure_days": len(result.identity_failures),
                "tolerance": {"I1": 1e-6, "I1b": 0.0, "I2": 0.0, "I3": 1e-6, "I3b": 1e-6},
                "examples": list(result.identity_failures[:3]),
            },
            "nav_series": {
                arm: [
                    {
                        "trade_date": item["trade_date"],
                        "nav": item["nav"],
                        "independent_nav": item["independent_nav"],
                        "split_index": item["split_index"],
                    }
                    for item in result.nav_series.get(arm, [])
                ]
                for arm in params.arms
            },
        }
    return {
        "schema": "p7_t5_dual_caliber/v1",
        "physical_initial_capital": float(params.physical_capital),
        "sleeve_capital": dict(params.sleeve_capital),
        "arms": arms,
        "notes": {
            "I1": "Σ cash_i ≡ 物理现金（≤1e-6 元）；I1b Σ credit_i = 0（精确 0）",
            "independent_equity_basis": (
                "independent_equity = total_value − 累计节省（T4 只读反事实口径，不含 credit）；"
                "含 credit 的权益见 sleeve_balance.json 的 equity（= cash + MV − credit）"
            ),
            "authorization_fee_basis": (
                "授权按申报费用判 A+Fee 不超 cash_open+L；执行扣实际费用，两者差在 100 ulp(F)/笔"
                "与全窗 1e-9 元门内（CD-1 已裁决）；最小现金余量见 sleeve_balance.summary.min_cash"
            ),
            "I3": "允许 E 为零/负的有符号贡献；NAV 只经 calculate_nav、链指标只经 calculate_chain_metrics",
            "verdict": "本 task 不产出任何经济判定（alpha_consumed=false）",
        },
    }


def identity_payload(results: Mapping[str, T5HostResult]) -> Dict[str, Any]:
    arms: Dict[str, Any] = {}
    for name, result in sorted(results.items()):
        envelope = result.envelope
        series = result.digest_series
        arms[name] = {
            "mode": dict(result.mode),
            "generation": int(envelope.generation),
            "idempotency_keys": int(len(envelope.applied)),
            "schema": "p7_t5_state/v1",
            "state_digest": envelope.state_digest(),
            "envelope_digest": envelope.digest(),
            "replay": dict(result.replay),
            "restart": dict(result.restart),
            "credit_series": {
                "days": len(result.credit_series),
                "first": result.credit_series[0] if result.credit_series else None,
                "last": result.credit_series[-1] if result.credit_series else None,
            },
            "digest_series": {
                "days": len(series),
                "first": series[0] if series else None,
                "middle": series[len(series) // 2] if series else None,
                "last": series[-1] if series else None,
            },
        }
    return {
        "schema": "p7_t5_identity/v1",
        "arms": arms,
        "cross_generation_policy": "同键同代重复提交 => 无操作；跨代或同键异内容 => 拒绝。",
        "credit_policy": (
            "借款 lot / 归还 / 死局与账本本体在**同一次提交**内校验（resume_state.credit_state）"
        ),
    }
