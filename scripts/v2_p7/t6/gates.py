# -*- coding: utf-8 -*-
"""P7 T6 机械门 G1–G7（全部**可失败**；任一不过即禁止发布）。

| 门 | 判据 |
|---|---|
| G1 上游未改 | T4 模块 / T2.5 输入 / T5 完成根逐件 SHA 与 T5 登记值一致 |
| G2 两臂同参 | 两臂参数摘要逐位相同（除 ``credit`` 开关），差异表只含该项 |
| G3 逐日闭合 | 两臂各自 I1/I1b/I2/I3 与逐袖 ``cash+credit ≥ 0`` **逐日**通过 |
| G4 公开口径 | NAV/年化/链式指标/MaxDD/费用与既有入口复算一致（容差登记） |
| G5 确定性 | 同参重跑终态摘要逐位一致；重放抽样一致 |
| G6 黄金例 | §4.6 三条全部通过 |
| G7 产物完整性 | 每个登记产物 SHA 可复核；manifest 自洽 |
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from scripts.v2_p7.common import sha256_file
from scripts.v2_p7.t6.params import T6Params, params_digest
from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics
from src.lazybull.v2.hosts.paper.performance import calculate_annualized_return

#: NAV / 年化 / 链式指标的复算容差（登记值）
TOLERANCE: Mapping[str, float] = {
    "nav": 1e-12,
    "cash_identity": 1e-6,
    "credit_identity": 0.0,
    "contribution_identity": 1e-9,
    "chain_metric": 1e-12,
    "annualized_pct": 1e-9,
    "fee_total": 1e-6,
    "episode_maxdd": 1e-12,
}


class GateError(RuntimeError):
    """机械门不通过（禁止发布；逐门明细见 ``gates`` 载荷）。"""


def build_gates(
    *,
    params: T6Params,
    frames: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    results: Mapping[str, Any],
    metrics: Mapping[str, Any],
    determinism: Mapping[str, Any],
    golden: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    run_root: Any,
    verification: Mapping[str, Any] | None = None,
    prereg: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """七道门一次判定（G7 与 G4 的落盘复算需在产物落盘后调用）。"""
    gates = {
        "G1": _g1(params),
        "G2": _g2(params, results),
        "G3": _g3(frames, results),
        "G4": _g4(params, frames, results, metrics, verification),
        "G5": _g5(determinism),
        "G6": _g6(golden),
        "G7": _g7(artifacts=artifacts, run_root=run_root, prereg=prereg),
    }
    failed = [key for key, value in sorted(gates.items()) if not value["ok"]]
    gates["summary"] = {
        "ok": not failed,
        "failed": failed,
        "total": 7,
        "passed": 7 - len(failed),
        "rule": "G1–G7 全部通过才允许发布 manifest；任一失败即停手报告",
    }
    return gates


def stop_on_hard_failures(gates: Mapping[str, Any]) -> None:
    summary = gates.get("summary") or {}
    if summary.get("ok"):
        return
    detail = "；".join(
        f"{key}: {(gates.get(key) or {}).get('failures')}" for key in summary.get("failed") or ()
    )
    raise GateError(f"T6 机械门未通过，禁止发布：{detail}")


# ────────────────────────── 各门 ──────────────────────────


def _g1(params: T6Params) -> Dict[str, Any]:
    """上游未改：T4 模块 / T2.5 输入 / T5 完成根逐件 SHA（**发布前重采**再核一次）。"""
    from scripts.v2_p7.t6.params import (
        ParamsError,
        load_t5_reference,
        module_identity,
        verify_t4_module,
        verify_t25_input,
    )

    failures: List[str] = []
    reference = params.t5_reference
    try:
        verify_t4_module(reference, module_identity(params.repo_root, "scripts/v2_p7/t4"))
    except ParamsError as error:
        failures.append(f"T4 模块变动：{error}")
    try:
        reread = load_t5_reference(params.repo_root)
        if reread.manifest_sha256 != reference.manifest_sha256:
            failures.append("T5 完成根 manifest 在运行期被改写")
    except ParamsError as error:
        failures.append(f"T5 完成根核验失败：{error}")
    try:
        verify_t25_input(reference, params.t25.payload())
    except ParamsError as error:
        failures.append(f"T2.5 输入身份变动：{error}")
    return {
        "ok": not failures,
        "criterion": "T4 模块 + T2.5 输入 + T5 完成根逐件 SHA 与 T5 r15 登记值一致（发布前重采）",
        "t4_modules": {
            "count": len(params.t4_module_sha),
            "source": "p7_t5_20261113_r15/legacy_sha.json#t4_modules",
            "reverified": True,
        },
        "t25_inputs": {
            "source": "p7_t5_20261113_r15/manifest.json#params.t4_params.t25_input",
            "reverified": True,
        },
        "t5_root": {
            "artifacts_verified": int(reference.self_check["artifacts_verified"]),
            "mismatch": list(reference.self_check["mismatch"]),
            "extra_files": list(reference.self_check["extra_files"]),
            "manifest_sha256": reference.manifest_sha256,
            "reverified": True,
        },
        "t5_module_baseline": {
            "count": len(params.t5_module_sha),
            "note": (
                "T5 完成根未登记 scripts/v2_p7/t5/** 自身 SHA（已登记缺口）："
                "本 task 首采为基线，后续 task 以此比对；不声称与 T5 登记值比对通过"
            ),
        },
        "failures": failures,
    }


def _g2(params: T6Params, results: Mapping[str, Any]) -> Dict[str, Any]:
    """两臂同参：参数摘要唯一；模式差异表只允许 ``credit``（``label`` 为非语义标签）。"""
    from scripts.v2_p7.t6.arms import mode_payload

    modes = mode_payload()
    differences = []
    for key in ("quota", "credit"):
        values = {name: bool(modes[name][key]) for name in sorted(modes)}
        if len(set(values.values())) > 1:
            differences.append(key)
    windows = {
        name: [str(result.days[0]), str(result.days[-1]), len(result.days)]
        for name, result in sorted(results.items())
    }
    same_window = len({tuple(value) for value in windows.values()}) == 1
    digest = params_digest(params)
    fee_rows = {name: int(result.fee_book.events) for name, result in sorted(results.items())}
    label_only = {name: str(modes[name]["label"]) for name in sorted(modes)}
    return {
        "ok": bool(differences == ["credit"] and same_window),
        "criterion": "两臂参数摘要逐位相同（唯一语义差异 = credit 开关），窗口/初资/目标/成本/日历一致",
        "params_digest": digest,
        "mode_payload": modes,
        "differences": differences,
        "non_semantic_fields": {
            "label": label_only,
            "note": "``label`` 是证据标签（VV-H / VV-S），不参与任何决策；语义差异只有 credit",
        },
        "windows": windows,
        "same_window": bool(same_window),
        "fee_ledger_events": fee_rows,
        "physical_capital": float(params.physical_capital),
        "sleeve_capital": dict(params.sleeve_capital),
        "cost_settings": dict(params.cost_settings),
        "calendar_days": len(params.extended_days),
        "failures": (
            []
            if (differences == ["credit"] and same_window)
            else [f"参数差异表异常：{differences}；窗口一致={same_window}"]
        ),
    }


def _g3(
    frames: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    results: Mapping[str, Any],
) -> Dict[str, Any]:
    """逐日闭合：I1 / I1b / I3 与逐袖非负门（I2 取宿主登记值）。"""
    arms: Dict[str, Any] = {}
    failures: List[str] = []
    for name in sorted(frames):
        daily = frames[name]["daily"]
        max_cash = max(abs(float(row["cash_sum"]) - float(row["physical_cash"])) for row in daily)
        max_credit = max(abs(float(row["credit_sum"])) for row in daily)
        max_contrib = max(
            abs(
                float(sum(float(value) for value in row["contribution"].values()))
                - float(row["nav"])
            )
            for row in daily
        )
        gate_rows = [row for row in daily if row["gate_min"] is not None]
        min_gate = min((float(row["gate_min"]) for row in gate_rows), default=None)
        identity_failures = int(len(results[name].identity_failures))
        arms[name] = {
            "days": len(daily),
            "max_cash_identity_diff": max_cash,
            "max_credit_sum": max_credit,
            "max_contribution_identity_diff": max_contrib,
            "min_gate": min_gate,
            "identity_failure_days": identity_failures,
            "gate_violation_days": int(results[name].gate_payload["violation_days"]),
        }
        if max_cash > TOLERANCE["cash_identity"]:
            failures.append(f"{name} I1 破缺：max|cash_sum−physical_cash|={max_cash!r}")
        if max_credit > TOLERANCE["credit_identity"]:
            failures.append(f"{name} I1b 破缺：max|Σcredit|={max_credit!r}")
        if max_contrib > TOLERANCE["contribution_identity"]:
            failures.append(f"{name} I3 破缺：max|Σcontribution−nav|={max_contrib!r}")
        if identity_failures:
            failures.append(f"{name} I2/双口径破缺日={identity_failures}")
        if int(results[name].gate_payload["violation_days"]):
            failures.append(
                f"{name} 逐袖非负门违例日={results[name].gate_payload['violation_days']}"
            )
    return {
        "ok": not failures,
        "criterion": "两臂各自 I1/I1b/I2/I3 与逐袖 cash+credit ≥ 0 逐日通过",
        "tolerances": dict(TOLERANCE),
        "arms": arms,
        "failures": failures,
    }


def _g4(
    params: T6Params,
    frames: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    results: Mapping[str, Any],
    metrics: Mapping[str, Any],
    verification: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """公开口径：NAV / 链式指标 / MaxDD / 年化 / 费用与既有入口复算一致（含落盘复算）。"""
    from src.lazybull.v2.core.accounting.nav import calculate_nav

    arms: Dict[str, Any] = {}
    failures: List[str] = []
    for name in sorted(frames):
        daily = frames[name]["daily"]
        frame = pd.DataFrame(
            {
                "nav": [float(row["nav"]) for row in daily],
                "split_index": [int(row["split_index"]) for row in daily],
                "total_value": [float(row["total_value"]) for row in daily],
            }
        )
        recomputed = calculate_chain_metrics(
            pd.DataFrame(
                {
                    "trade_date": [str(row["trade_date"]) for row in daily],
                    "nav": frame["nav"],
                    "split_index": frame["split_index"],
                }
            )
        )
        stored = metrics["arms"][name]["chain_metrics"]
        metric_diff = {
            key: abs(float(recomputed[key]) - float(stored[key]))
            for key in ("total_return", "cagr", "max_drawdown", "volatility", "sharpe")
        }
        nav_diff = float(
            max(
                abs(
                    float(calculate_nav(row["total_value"], float(params.physical_capital)))
                    - float(row["nav"])
                )
                for row in daily
            )
        )
        annualized = calculate_annualized_return(
            float(params.physical_capital),
            float(daily[-1]["total_value"]),
            str(daily[-1]["trade_date"]),
            str(daily[0]["trade_date"]),
        )
        annualized_diff = abs(
            float(annualized) - float(metrics["arms"][name]["annualized_natural_days_pct"])
        )
        fee_diff = abs(
            float(sum(float(trade.get("fee") or 0.0) for trade in frames[name]["trades"]))
            - float(results[name].fee_book.physical_fee_total)
        )
        episode_mdd = float(metrics["arms"][name]["drawdown"]["episode_series_max"])
        stored_mdd = float(metrics["arms"][name]["drawdown"]["max_drawdown"])
        arms[name] = {
            "nav_max_abs_diff": nav_diff,
            "chain_metric_max_abs_diff": metric_diff,
            "annualized_pct_abs_diff": annualized_diff,
            "fee_total_abs_diff": fee_diff,
            "maxdd_episode_vs_chain_abs_diff": abs(episode_mdd - stored_mdd),
        }
        if nav_diff > TOLERANCE["nav"]:
            failures.append(f"{name} NAV 复算不一致：{nav_diff!r}")
        if max(metric_diff.values()) > TOLERANCE["chain_metric"]:
            failures.append(f"{name} 链式指标复算不一致：{metric_diff}")
        if annualized_diff > TOLERANCE["annualized_pct"]:
            failures.append(f"{name} 年化复算不一致：{annualized_diff!r}")
        if fee_diff > TOLERANCE["fee_total"]:
            failures.append(f"{name} 费用台账与交易行不一致：{fee_diff!r}")
        if abs(episode_mdd - stored_mdd) > TOLERANCE["episode_maxdd"]:
            failures.append(f"{name} 回撤事件极值 ≠ 链式 MaxDD：{episode_mdd!r} vs {stored_mdd!r}")
    recompute = dict(verification or {})
    if verification is None:
        failures.append("缺落盘复算结果（verify_run 未接入）：禁止以内存对象自证")
    elif not recompute.get("ok"):
        failures.append(f"落盘复算不一致：{recompute.get('mismatches')}")
    return {
        "ok": not failures,
        "criterion": "NAV/链式指标/MaxDD/年化/费用与既有唯一入口逐位一致，且由**落盘产物**独立复算通过",
        "tolerances": dict(TOLERANCE),
        "arms": arms,
        "artifact_recompute": {
            "present": bool(verification is not None),
            "ok": bool(recompute.get("ok")),
            "mismatches": list(recompute.get("mismatches") or ()),
            "noise_max_abs_diff": ((recompute.get("checks") or {}).get("noise") or {}).get(
                "max_abs_diff"
            ),
            "calibration_recompute_ok": (
                (recompute.get("checks") or {}).get("calibration") or {}
            ).get("ok"),
        },
        "failures": failures,
    }


def _g5(determinism: Mapping[str, Any]) -> Dict[str, Any]:
    """G5：重跑逐位一致 **且** 两次运行各自的宿主 replay/restart 自检全为真（C6-01）。"""
    arms = dict(determinism.get("arms") or {})
    self_checks = {
        name: {
            "ok": bool((payload or {}).get("self_checks_ok")),
            "rows": dict((payload or {}).get("self_checks") or {}),
        }
        for name, payload in sorted(arms.items())
    }
    return {
        "ok": bool(determinism.get("ok")),
        "criterion": (
            "同参重跑终态摘要逐位一致；且两臂两次运行的 replay/restart 自检 "
            "checked 与 all_bitwise_identical 全为真"
        ),
        "self_checks": self_checks,
        "payload": dict(determinism),
        "failures": list(determinism.get("failures") or ()),
    }


def _g6(golden: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "ok": bool(golden.get("ok")),
        "criterion": "§4.6 三条黄金例全部通过",
        "failed": list(golden.get("failed") or ()),
        "cases": [str(case.get("name")) for case in golden.get("cases") or ()],
        "failures": list(golden.get("failed") or ()),
    }


def _g7(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    run_root: Any,
    prereg: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """产物完整性：逐件 SHA 可复核；目录内无未登记文件；跨 run 预登记一致。"""
    from pathlib import Path

    root = Path(run_root)
    registered = {str(name): str(item.get("sha256") or "") for name, item in artifacts.items()}
    mismatch: List[Dict[str, Any]] = []
    for name, expected in sorted(registered.items()):
        path = root / name
        if not path.exists():
            mismatch.append({"file": name, "status": "missing"})
            continue
        actual = sha256_file(path)
        if actual != expected:
            mismatch.append({"file": name, "expected": expected, "actual": actual})
    extra = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and path.relative_to(root).as_posix() not in registered
    )
    ordering = _pre_registration_ordering(root)
    stability = _prereg_stability_check(prereg)
    failures = ([f"SHA 不符/缺失：{mismatch[:3]}"] if mismatch else []) + (
        [f"未登记文件：{extra[:5]}"] if extra else []
    )
    if not ordering["ok"]:
        failures.append(f"预登记未先于两臂运行落盘：{ordering}")
    if not stability["ok"]:
        failures.append(f"跨 run 预登记摘要不一致：{stability}")
    return {
        "ok": bool(not mismatch and not extra and ordering["ok"] and stability["ok"]),
        "criterion": (
            "每个登记产物 SHA 可复核；目录内无未登记文件；预登记先于运行且跨 run 摘要一致"
            "（manifest 自洽）"
        ),
        "artifacts": len(registered),
        "mismatch": mismatch,
        "extra_files": extra,
        "pre_registration": ordering,
        "prereg_stability": stability,
        "failures": failures,
    }


def _prereg_stability_check(prereg: Mapping[str, Any] | None) -> Dict[str, Any]:
    """跨 run 预登记一致性（由 ``artifacts.prereg_stability`` 计算，缺失即失败）。"""
    if prereg is None:
        return {"ok": False, "reason": "缺 prereg_stability 载荷（不得省略跨 run 核验）"}
    runs = dict(prereg.get("runs") or {})
    return {
        "ok": bool(prereg.get("all_equal")) and int(prereg.get("covers_current_run") or 0) == 1,
        "runs": sorted(runs),
        "digests": list(prereg.get("digests") or ()),
        "covers_current_run": bool(prereg.get("covers_current_run")),
        "missing_runs_not_enumerated": (
            "删除旧 run 会使 all_equal 静默为真；本字段登记该扫描的覆盖边界"
        ),
    }


def _pre_registration_ordering(root: Any) -> Dict[str, Any]:
    """机械证据：``calibration.json`` 的 mtime 早于全部臂帧（预登记先于运行）。"""
    from pathlib import Path

    base = Path(root)
    calibration = base / "calibration.json"
    arm_frames = [base / "arms" / arm / f"daily_{arm}.parquet" for arm in ("vvh", "vvs")]
    if not calibration.exists() or any(not path.exists() for path in arm_frames):
        return {"ok": False, "reason": "缺 calibration.json 或臂帧"}
    calibration_mtime = calibration.stat().st_mtime
    first_arm_mtime = min(path.stat().st_mtime for path in arm_frames)
    return {
        "ok": bool(calibration_mtime <= first_arm_mtime),
        "calibration_mtime": calibration_mtime,
        "first_arm_mtime": first_arm_mtime,
        "rule": "预登记必须在两臂运行之前落盘（运行后修改即视为无效裁定）",
    }
