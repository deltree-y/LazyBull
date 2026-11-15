# -*- coding: utf-8 -*-
"""P-1 重开校准：层一机械核验（执行层）+ 层二净残存 DGP（推断层）。

关键纪律
========
- 层一常残存构造区间退化为 ``[θ,θ]``，覆盖率必然 100%，**不得作为统计证据**；
- 层二只验推断层，不经执行层；200 套独立样本、θ∈{0, −10bp}；
- 机器预期由冻结费率运行时计算，文中 ``−18.8949bp`` 只作显示；
- 容差分层：独立公式 1e-12 / 同函数同输入重放 0 / M2 反解 1e-9。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from scripts.v2_p7.t1 import execution
from scripts.v2_p7.t1.p1_bounds import _bootstrap_values
from scripts.v2_p7.t1.p1_params import P1RParams

BP = 1e-4
TOL_INDEPENDENT = 1e-12
TOL_REPLAY = 0.0
TOL_INVERSE = 1e-9


def _flat_net(params: P1RParams, price: float = 1000.0) -> float:
    """恒定价格、无公司行为、正常成交的单腿净收益（M1 机器预期）。"""
    leg = execution.single_leg_net(
        price,
        price,
        price,
        price,
        1.0,
        True,
        cost=params.cost.as_dict(),
        min_ticket=params.min_ticket,
        lot_size=params.lot_size,
    )
    if not leg.valid:
        raise RuntimeError("M1 平价腿不可估值")
    return float(leg.net)


def layer1_mechanical(params: P1RParams, frozen: Dict[str, Any]) -> Dict[str, Any]:
    """层一：M1 平价 / M2 精确反解 / M3 非线性数值反解。"""
    cost = params.cost.as_dict()
    a = cost["commission_rate"] + cost["slippage"]
    b = cost["commission_rate"] + cost["stamp_tax"] + cost["slippage"]
    r_flat_formula = (1.0 - b) / (1.0 + a) - 1.0
    r_flat_leg = _flat_net(params)
    m1_gap = abs(r_flat_leg - r_flat_formula)
    m1 = {
        "expected_formula": float(r_flat_formula),
        "expected_leg": float(r_flat_leg),
        "gap": float(m1_gap),
        "tolerance": TOL_INDEPENDENT,
        "passed": bool(m1_gap <= TOL_INDEPENDENT),
        "display_constant_never_asserted": True,
        "frozen_display_bp": float(frozen.get("display_bp", -18.894860513998565)),
    }
    replay_gap = abs(_flat_net(params) - r_flat_leg)
    m1["replay_gap"] = float(replay_gap)
    m1["replay_tolerance"] = TOL_REPLAY
    m1["replay_passed"] = bool(replay_gap <= TOL_REPLAY)

    m2_rows: List[Dict[str, Any]] = []
    for theta_bp, base_bp in ((0.0, 0.0), (10.0, 0.0), (-10.0, 0.0), (10.0, 5.0), (-10.0, -5.0)):
        theta = theta_bp * BP
        base = base_bp * BP
        gross = (1.0 + base + theta) * (1.0 + a) / (1.0 - b) - 1.0
        price_buy = 1000.0
        price_sell = price_buy * (1.0 + gross)
        leg = execution.single_leg_net(
            price_buy,
            price_buy,
            price_sell,
            price_sell,
            1.0,
            True,
            cost=cost,
            min_ticket=params.min_ticket,
            lot_size=params.lot_size,
        )
        measured = float(leg.net) - base
        m2_rows.append(
            {
                "theta_bp": theta_bp,
                "base_bp": base_bp,
                "gross": float(gross),
                "measured_bp": measured / BP,
                "gap": float(abs(measured - theta)),
                "passed": bool(abs(measured - theta) <= TOL_INVERSE),
            }
        )
    m2 = {
        "rows": m2_rows,
        "tolerance": TOL_INVERSE,
        "passed": all(row["passed"] for row in m2_rows),
    }

    small_amount = 100.0
    small_buy_cost = float(execution._buy_cost(np.asarray([small_amount]), cost)[0])
    small_sell_cost = float(execution._sell_cost(np.asarray([small_amount]), cost)[0])
    lot_leg = execution.single_leg_net(
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1.0,
        True,
        cost=cost,
        min_ticket=params.min_ticket,
        lot_size=params.lot_size,
    )
    split_leg = execution.single_leg_net(
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1.5,
        True,
        cost=cost,
        min_ticket=params.min_ticket,
        lot_size=params.lot_size,
    )
    m3 = {
        "min_commission_branch": {
            "amount": small_amount,
            "buy_cost": small_buy_cost,
            "sell_cost": small_sell_cost,
            "triggered": bool(small_buy_cost > small_amount * cost["commission_rate"] + 1e-15),
            "note": "B0 最小单笔预算保证 buy_amount ≥ V，最低佣金在常规腿不触发；本分支独立覆盖成本函数",
        },
        "lot_branch": {
            "price": 1000.0,
            "shares": int(lot_leg.shares),
            "buy_amount": float(lot_leg.buy_amount),
            "lot_size": int(params.lot_size),
            "integer_lots": bool(lot_leg.shares % params.lot_size == 0),
        },
        "split_branch": {
            "ratio": 1.5,
            "shares": int(split_leg.shares),
            "sell_shares": float(split_leg.sell_shares),
            "net": float(split_leg.net),
        },
        "method": "数值反解（不用闭式），偏差逐笔登记",
        "note": "M3 不设统一容差；上述读数为登记项",
    }

    return {
        "M1": m1,
        "M2": m2,
        "M3": m3,
        "a": float(a),
        "b": float(b),
        "layer1_passed": bool(m1["passed"] and m1["replay_passed"] and m2["passed"]),
        "not_statistical_evidence": True,
    }


def estimate_pilot(
    d_point: np.ndarray,
    day_pos: np.ndarray,
    n_days: int,
    *,
    floor_variance: float = 1e-12,
) -> Dict[str, Any]:
    """一次性只读 pilot：只估方差与自相关，不把均值当作 θ。"""
    finite = np.isfinite(d_point)
    values = np.asarray(d_point)[finite]
    days = np.asarray(day_pos)[finite]
    if values.size < 100:
        raise ValueError(f"pilot 样本不足：{values.size}")
    mean_value = float(values.mean())
    centered = values - mean_value
    day_sum = np.bincount(days, weights=centered, minlength=n_days)
    day_count = np.bincount(days, minlength=n_days)
    active = day_count > 0
    cluster = np.zeros(n_days)
    cluster[active] = day_sum[active] / day_count[active]
    series = cluster[active]
    if series.size < 30:
        raise ValueError(f"pilot 日簇样本不足：{series.size}")
    lag = series[:-1]
    lead = series[1:]
    var_lag = float(np.var(lag, ddof=1))
    phi = float(np.cov(lag, lead, ddof=1)[0, 1] / var_lag) if var_lag > 0 else 0.0
    phi = float(np.clip(phi, -0.95, 0.95))
    residual = lead - float(np.mean(lead)) - phi * (lag - float(np.mean(lag)))
    sigma_eta = float(np.std(residual, ddof=1))
    sigma_u = float(np.sqrt(max(sigma_eta**2 / max(1.0 - phi**2, 1e-6), floor_variance)))
    group_residual = values - cluster[days]
    sigma_eps = float(np.sqrt(max(float(np.var(group_residual, ddof=1)), floor_variance)))
    return {
        "phi": phi,
        "sigma_u": sigma_u,
        "sigma_eta": sigma_eta,
        "sigma_epsilon": sigma_eps,
        "n_intents": int(values.size),
        "n_active_days": int(active.sum()),
        "mean_registered_only_for_centering": mean_value,
        "mean_not_used_as_theta": True,
        "method": "日簇去均值残差 AR(1) OLS；σ_ε = 组内残差标准差",
    }


def _simulate_theta(
    *,
    theta: float,
    pilot: Dict[str, Any],
    day_pos: np.ndarray,
    n_days: int,
    samples: int,
    replicates: int,
    seed_base: int,
    params: P1RParams,
) -> Dict[str, Any]:
    """单个 θ 档的 200 套独立样本：生成 d_i 并跑完整推断管线（双路）。"""
    days = np.asarray(day_pos)
    phi = float(pilot["phi"])
    sigma_u = float(pilot["sigma_u"])
    sigma_eps = float(pilot["sigma_epsilon"])
    n = int(days.size)
    records: List[Dict[str, Any]] = []
    for sample in range(int(samples)):
        rng = np.random.default_rng(int(seed_base) + sample)
        shock = rng.normal(0.0, sigma_u * np.sqrt(max(1.0 - phi**2, 1e-6)), size=n_days)
        factor = np.zeros(n_days)
        for day in range(1, n_days):
            factor[day] = phi * factor[day - 1] + shock[day]
        epsilon = rng.normal(0.0, sigma_eps, size=n)
        values = theta + factor[days] + epsilon
        main = _bootstrap_values(
            days,
            values,
            n_days,
            block=params.bootstrap_main_block,
            replicates=replicates,
            seed=int(seed_base) + sample * 2 + 1,
            mode="stationary",
        )
        aux = _bootstrap_values(
            days,
            values,
            n_days,
            block=params.bootstrap_aux_block,
            replicates=replicates,
            seed=int(seed_base) + sample * 2 + 2,
            mode="circular",
        )
        los = [item for item in (main["lo"], aux["lo"]) if item is not None]
        his = [item for item in (main["hi"], aux["hi"]) if item is not None]
        if not los or not his:
            records.append({"lo": None, "hi": None, "width": None, "invalid": True, "point": None})
            continue
        lo = min(los)
        hi = max(his)
        records.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "width": float(hi - lo),
                "invalid": bool((main["invalid"] + aux["invalid"]) > 0),
                "point": float(values.mean()),
            }
        )
    return _summarize_theta(theta, records)


def _summarize_theta(theta: float, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    valid = [item for item in records if item["lo"] is not None and item["hi"] is not None]
    degenerate = [item for item in valid if item["width"] < 1e-12]
    covered = [item for item in valid if item["lo"] <= theta <= item["hi"]]
    negative = [item for item in valid if item["hi"] < 0]
    points = [item["point"] for item in valid if item.get("point") is not None]
    point_se = float(np.std(points, ddof=1)) if len(points) > 1 else None
    return {
        "theta": float(theta),
        "samples": total,
        "usable": len(valid),
        "coverage": (len(covered) / len(valid)) if valid else None,
        "negative_rate": (len(negative) / len(valid)) if valid else None,
        "degenerate_rate": (len(degenerate) / len(valid)) if valid else None,
        "point_se": point_se,
        "records": list(records),
    }


def layer2_statistical(
    pilot: Dict[str, Any],
    day_pos: np.ndarray,
    n_days: int,
    params: P1RParams,
    *,
    samples: int,
    replicates: int,
) -> Dict[str, Any]:
    """层二：θ=0 与 θ=−Δ 两档各 200 套独立样本。"""
    theta_grid = [0.0, -10.0 * BP]
    results = [
        _simulate_theta(
            theta=theta,
            pilot=pilot,
            day_pos=day_pos,
            n_days=n_days,
            samples=samples,
            replicates=replicates,
            seed_base=int(params.seeds["layer2_calibration_base"]),
            params=params,
        )
        for theta in theta_grid
    ]
    zero = results[0]
    negative = results[1]
    coverage_ok = all(item["coverage"] is not None and item["coverage"] >= 0.90 for item in results)
    false_trigger_ok = zero["negative_rate"] is not None and zero["negative_rate"] <= 0.05
    detection_ok = negative["negative_rate"] is not None and negative["negative_rate"] >= 0.80
    degenerate_ok = all(
        item["degenerate_rate"] is not None and item["degenerate_rate"] <= 0.05 for item in results
    )
    passed = bool(coverage_ok and false_trigger_ok and detection_ok and degenerate_ok)
    return {
        "theta_grid_bp": [item / BP for item in theta_grid],
        "samples_per_theta": int(samples),
        "replicates_per_sample": int(replicates),
        "results": results,
        "coverage_ok": bool(coverage_ok),
        "false_trigger_ok": bool(false_trigger_ok),
        "negative_detection_ok": bool(detection_ok),
        "degenerate_ok": bool(degenerate_ok),
        "negative_power": negative["negative_rate"],
        "passed": passed,
        "failure_consequence": "continue + 限制（校准未通过），禁用 negative",
        "calibration_replicates_disclosure": (
            "校准复制次数为实现参数（预登记未冻结）；报告另做 B 敏感性"
        ),
    }


def noise_fidelity(
    pilot: Dict[str, Any],
    day_pos: np.ndarray,
    n_days: int,
    sampling: Dict[str, Any],
    layer2: Dict[str, Any],
) -> Dict[str, Any]:
    """实际样本与合成 DGP 的噪声保真度对照（只登记，不作判据）。

    解析式：``Var(D) = σ_u²[Σ_tΣ_s w_t w_s φ^|t−s| − (Σ_t w_t φ^t)²] + σ_ε²/N``，
    ``w_t = n_t/N``；经验值取 200 套样本点估计的标准差。
    """
    counts = np.bincount(np.asarray(day_pos, dtype=np.int64), minlength=int(n_days)).astype(
        np.float64
    )
    total = float(counts.sum())
    weights = counts / total if total > 0 else counts
    phi = float(pilot["phi"])
    sigma_u = float(pilot["sigma_u"])
    sigma_eps = float(pilot["sigma_epsilon"])
    lag = np.arange(len(weights), dtype=np.float64)
    ac = np.correlate(weights, weights, mode="full")[len(weights) - 1 :]
    quad = float(ac[0] + 2.0 * np.sum(ac[1:] * (phi ** lag[1:])))
    tail = float(weights @ (phi**lag))
    variance = sigma_u**2 * (quad - tail**2) + sigma_eps**2 / total
    analytic_se_bp = float(np.sqrt(max(variance, 0.0)) * 1e4)
    empirical = [
        float(item["point_se"])
        for item in layer2.get("results", [])
        if item.get("point_se") is not None
    ]
    empirical_se_bp = float(np.mean(empirical) * 1e4) if empirical else None
    width = sampling.get("main", {}).get("width")
    sample_se_bp = float(width * 1e4 / 3.92) if width else None
    active = counts > 0
    return {
        "schema": "p7_p1r_noise_fidelity/v1",
        "actual": {
            "n_intents": int(total),
            "n_days": int(n_days),
            "active_days": int(active.sum()),
            "mean_intents_per_active_day": float(counts[active].mean()) if active.any() else None,
            "phi": phi,
            "sigma_u_bp": float(sigma_u * 1e4),
            "sigma_epsilon_bp": float(sigma_eps * 1e4),
        },
        "se_bp": {
            "sample_diagnostic": sample_se_bp,
            "dgp_analytic": analytic_se_bp,
            "dgp_empirical_200": empirical_se_bp,
        },
        "ratio_sample_over_dgp": (
            float(sample_se_bp / analytic_se_bp)
            if (sample_se_bp is not None and analytic_se_bp > 0)
            else None
        ),
        "checks_to_review": [
            "pilot 在去空日的日簇均值序列上拟合 AR(1)，再用于完整日历轴；空日与事件日间隔的相关结构可能不同",
            "日簇均值未显式分离小样本日的意图噪声 σ_ε²/n_t，n_t 小的日子方差被高估",
            "DGP 的 u_t 在每个日历日都递推；实际样本只有事件日可观测，二者有效独立信息量可能不同",
            "bootstrap 复制次数（2000/10000）只影响分位数 MC 精度，不改变上述信息量差异",
        ],
        "not_a_decision_gate": True,
        "note": "本表只解释样本 SE 与 DGP SE 的差别；不改本轮判据、不改 verdict",
    }


def calibration_passed(layer1: Dict[str, Any], layer2: Dict[str, Any]) -> bool:
    """两层都通过才算校准通过（层一只验执行层）。"""
    return bool(layer1.get("layer1_passed")) and bool(layer2.get("passed"))
