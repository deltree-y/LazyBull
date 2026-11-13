# -*- coding: utf-8 -*-
"""P7 T6 经济裁定：按**预登记**判据 C1–C5 逐条判定并给出 ``verdict``。

纪律
====
- 判据、阈值、噪声带、verdict 规则一律来自 :mod:`scripts.v2_p7.t6.calibration`
  （运行前冻结）；本模块**只做代入与比较**，不得引入新判据或事后挑选子集；
- 每条判据都必须给出：定义、阈值、两臂实测值、配对差、噪声带、是否越带、是否通过；
- ``verdict`` 只对本窗口、本机械与费用口径负责（表述边界见 calibration.BOUNDARY）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

VERDICT_SCHEMA = "p7_t6_verdict/v1"

#: 链式指标字段 -> 指标 key（与 metrics.paired 同名）
C3_TERMS = ("cagr", "max_drawdown", "volatility")


class VerdictError(RuntimeError):
    """裁定输入不完整或与预登记不一致（fail-fast，不产出 verdict）。"""


def build_verdict(
    *,
    calibration: Mapping[str, Any],
    metrics: Mapping[str, Any],
    noise: Mapping[str, Any],
) -> Dict[str, Any]:
    """C1–C5 → verdict（pass / fail / inconclusive）。"""
    criteria = [
        _c1(calibration, metrics, noise),
        _c2(calibration, metrics, noise),
        _c3(calibration, metrics, noise),
        _c4(calibration, metrics),
        _c5(calibration, metrics),
    ]
    verdict = _decide(criteria)
    return {
        "schema": VERDICT_SCHEMA,
        "task": "P7 T6",
        "verdict": verdict["verdict"],
        "verdict_reason": verdict["reason"],
        "alpha_consumed": True,
        "calibration_digest": _calibration_digest(calibration),
        "criteria": criteria,
        "passed_ids": [item["id"] for item in criteria if item["passed"]],
        "failed_ids": [item["id"] for item in criteria if not item["passed"]],
        "opposite_over_band_ids": [item["id"] for item in criteria if item["opposite_over_band"]],
        "verdict_rule": dict(calibration["verdict_rule"]),
        "boundary": list(calibration["boundary"]),
        "not_in_scope": list(calibration["not_in_scope"]),
    }


def _calibration_digest(calibration: Mapping[str, Any]) -> str:
    from scripts.v2_p7.t6.calibration import digest

    return digest()


# ────────────────────────── 单条判据 ──────────────────────────


def _c1(
    calibration: Mapping[str, Any], metrics: Mapping[str, Any], noise: Mapping[str, Any]
) -> Dict[str, Any]:
    return _directional(
        calibration=calibration,
        metrics=metrics,
        noise=noise,
        criterion_id="C1",
        metric="total_return",
        better="higher",
    )


def _c2(
    calibration: Mapping[str, Any], metrics: Mapping[str, Any], noise: Mapping[str, Any]
) -> Dict[str, Any]:
    return _directional(
        calibration=calibration,
        metrics=metrics,
        noise=noise,
        criterion_id="C2",
        metric="max_drawdown",
        better="higher",
    )


def _c3(
    calibration: Mapping[str, Any], metrics: Mapping[str, Any], noise: Mapping[str, Any]
) -> Dict[str, Any]:
    spec = _criterion(calibration, "C3")
    items: List[Dict[str, Any]] = []
    for item in calibration["c3_items"]:
        term = str(item["term"])
        better = str(item["better"])
        delta = float(metrics["paired"][term]["delta"])
        band = dict(noise["primary"]["metrics"][term])
        delta_threshold = float(calibration["delta"][str(item["delta_key"])])
        over = _over_band(delta=delta, band=band, threshold=delta_threshold)
        direction_ok = delta > 0.0 if better == "higher" else delta < 0.0
        items.append(
            {
                "term": term,
                "better": better,
                "control": float(metrics["paired"][term]["control"]),
                "treatment": float(metrics["paired"][term]["treatment"]),
                "delta": delta,
                "threshold": delta_threshold,
                "band": band,
                "over_band": bool(over),
                "direction_ok": bool(direction_ok),
                "passed": bool(over and direction_ok),
            }
        )
    opposite = any(item["over_band"] and not item["direction_ok"] for item in items)
    return {
        "id": "C3",
        "name": str(spec["name"]),
        "definition": str(spec["definition"]),
        # 只渲染 C3 的三个 term（预登记载荷里的 ``threshold`` 是 DELTA 全集，含无关键）
        "threshold": {
            str(spec_item["term"]): float(calibration["delta"][str(spec_item["delta_key"])])
            for spec_item in calibration["c3_items"]
        },
        "pass_rule": str(spec["pass_rule"]),
        "band_source": str(spec["band"]),
        "items": items,
        "passed": bool(all(item["passed"] for item in items)),
        "opposite_over_band": bool(opposite),
        "evidence": "metrics.paired[cagr,max_drawdown,volatility] + noise.primary.metrics",
        "shared_with": "C2 与 items[max_drawdown] 共用同一数字与同一带（预登记设计）",
    }


def _c4(calibration: Mapping[str, Any], metrics: Mapping[str, Any]) -> Dict[str, Any]:
    spec = _criterion(calibration, "C4")
    attribution = dict(metrics["paired"]["attribution"])
    delta_tr = float(attribution["delta_tr"])
    delta_fee_pp = float(attribution["delta_fee_over_pc"])
    delta_gross = float(attribution["delta_tr_gross"])
    delta_turnover = float(attribution["delta_turnover"])
    threshold = float(calibration["delta"]["total_return"])
    checks = {
        "gross_residual_within_delta": bool(abs(delta_gross) <= threshold),
        "fee_increment_within_delta": bool(abs(delta_fee_pp) <= threshold),
        "turnover_fee_same_sign": bool(attribution["same_sign_turnover_fee"]),
    }
    return {
        "id": "C4",
        "name": str(spec["name"]),
        "definition": str(spec["definition"]),
        "threshold": threshold,
        "pass_rule": str(spec["pass_rule"]),
        "band_source": "无（确定性归因）",
        "values": {
            "delta_tr": delta_tr,
            "delta_fee_over_pc": delta_fee_pp,
            "delta_tr_fee": float(attribution["delta_tr_fee"]),
            "delta_tr_gross": delta_gross,
            "delta_turnover": delta_turnover,
            "fee_control": float(metrics["paired"]["fees"]["control"]),
            "fee_treatment": float(metrics["paired"]["fees"]["treatment"]),
            "turnover_control": float(metrics["paired"]["turnover"]["control"]),
            "turnover_treatment": float(metrics["paired"]["turnover"]["treatment"]),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "opposite_over_band": False,
        "evidence": "metrics.paired.attribution",
        "note": (
            "当 |ΔTR| + |Δfee|/PC ≤ δ 时 gross 残差项**算术必然成立**，"
            "故 C4 的『通过』不构成独立证据；它只表示差异未被费用解释"
            "（本例费用仅解释一小部分，方向见 values.delta_fee_over_pc）"
        ),
    }


def _c5(calibration: Mapping[str, Any], metrics: Mapping[str, Any]) -> Dict[str, Any]:
    spec = _criterion(calibration, "C5")
    threshold = dict(spec["threshold"])
    monthly = list(metrics["monthly_paired"])
    full_delta = float(metrics["paired"]["total_return"]["delta"])
    concentration = _concentration(monthly)
    loo = _leave_out_top_months(monthly=monthly, top=3)
    year = _year_sign(
        metrics["by_year_paired"],
        float(threshold["year_sign_min_share"]),
        int((calibration.get("stability") or {}).get("year_sign_min_years", 3)),
    )
    loo_ok = _loo_ok(
        full_delta=full_delta, loo_delta=loo["delta"], retain=float(threshold["loo_retain_min"])
    )
    checks = {
        "concentration_within_max": bool(concentration <= float(threshold["concentration_max"])),
        "loo_same_sign_and_retained": bool(loo_ok),
        "year_sign_consistent": bool(year["passed"]),
    }
    return {
        "id": "C5",
        "name": str(spec["name"]),
        "definition": str(spec["definition"]),
        "threshold": threshold,
        "pass_rule": str(spec["pass_rule"]),
        "band_source": str(spec["band"]),
        "values": {
            "full_delta": full_delta,
            "concentration": concentration,
            "excluded_months": loo["excluded"],
            "loo_delta": loo["delta"],
            "year": year,
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "opposite_over_band": False,
        "evidence": "metrics.monthly_paired / metrics.by_year_paired",
    }


# ────────────────────────── 共用工具 ──────────────────────────


def _criterion(calibration: Mapping[str, Any], criterion_id: str) -> Mapping[str, Any]:
    for item in calibration["criteria"]:
        if str(item["id"]) == criterion_id:
            return item
    raise VerdictError(f"预登记缺判据 {criterion_id}")


def _directional(
    *,
    calibration: Mapping[str, Any],
    metrics: Mapping[str, Any],
    noise: Mapping[str, Any],
    criterion_id: str,
    metric: str,
    better: str,
) -> Dict[str, Any]:
    spec = _criterion(calibration, criterion_id)
    paired = metrics["paired"][metric]
    delta = float(paired["delta"])
    band = dict(noise["primary"]["metrics"][metric])
    threshold = float(calibration["delta"][str(spec["threshold_key"])])
    over = _over_band(delta=delta, band=band, threshold=threshold)
    direction_ok = delta > 0.0 if better == "higher" else delta < 0.0
    payload = {
        "id": str(spec["id"]),
        "name": str(spec["name"]),
        "definition": str(spec["definition"]),
        "threshold": threshold,
        "pass_rule": str(spec["pass_rule"]),
        "band_source": str(spec["band"]),
        "values": {
            "control": float(paired["control"]),
            "treatment": float(paired["treatment"]),
            "delta": delta,
            "relative_delta": paired.get("relative_delta"),
        },
        "band": band,
        "over_band": bool(over),
        "direction_ok": bool(direction_ok),
        "passed": bool(over and direction_ok),
        "opposite_over_band": bool(over and not direction_ok),
        "evidence": f"metrics.paired.{metric} + noise.primary.metrics.{metric}",
        "band_center_note": (
            "区间以 boot_mean 为中心（bias=boot_mean−point，mde=1.96·se）；" "越带只看区间是否含 0"
        ),
    }
    if metric == "max_drawdown":
        payload["shared_with"] = (
            "C3.items[max_drawdown]（同一数字、同一噪声带；属预登记设计，"
            "两条判据不构成相互独立的证据）"
        )
    return payload


def _over_band(*, delta: float, band: Mapping[str, Any], threshold: float) -> bool:
    """越带 ≡ |Δ| ≥ δ 且 95% 区间不含 0（预登记的合取判据）。"""
    return bool(abs(delta) >= float(threshold) and bool(band.get("excludes_zero")))


def _concentration(monthly: Sequence[Mapping[str, Any]]) -> float:
    values = sorted((abs(float(row["delta"])) for row in monthly), reverse=True)
    total = float(sum(values))
    if total <= 0.0:
        return 0.0
    return float(sum(values[:3]) / total)


def _leave_out_top_months(*, monthly: Sequence[Mapping[str, Any]], top: int) -> Dict[str, Any]:
    ordered = sorted(monthly, key=lambda row: abs(float(row["delta"])), reverse=True)
    excluded = [str(row["month"]) for row in ordered[:top]]
    keep = [row for row in monthly if str(row["month"]) not in set(excluded)]
    control = 1.0
    treatment = 1.0
    for row in keep:
        control *= 1.0 + float(row["control"])
        treatment *= 1.0 + float(row["treatment"])
    return {"excluded": excluded, "delta": float(treatment - control), "months": len(keep)}


def _loo_ok(*, full_delta: float, loo_delta: float, retain: float) -> bool:
    if full_delta == 0.0:
        return False
    same_sign = (loo_delta > 0.0) == (full_delta > 0.0)
    return bool(same_sign and abs(loo_delta) >= retain * abs(full_delta))


def _year_sign(
    rows: Sequence[Mapping[str, Any]], min_share: float, min_years: int
) -> Dict[str, Any]:
    from scripts.v2_p7.t6.noise import sign_test

    deltas = [float(row["delta"]) for row in rows]
    payload = sign_test(deltas, unit="year")
    share = payload["same_sign_share"]
    passed = bool(
        payload["non_zero"] >= int(min_years)
        and share is not None
        and float(share) >= float(min_share)
    )
    return {
        **payload,
        "min_share": float(min_share),
        "min_years": int(min_years),
        "passed": passed,
    }


def _decide(criteria: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """先判 fail（方向相反且越带）→ 再判 pass（全满足）→ 否则 inconclusive。"""
    opposite = [item for item in criteria if item["opposite_over_band"]]
    if opposite:
        detail = "；".join(f"{item['id']}：{_summary(item)}" for item in opposite)
        return {
            "verdict": "fail",
            "reason": (
                f"判据 {[item['id'] for item in opposite]} 出现「方向相反且越带」"
                f"（|Δ| ≥ δ 且 95% 噪声带不含 0）：{detail}"
            ),
        }
    failed = [item for item in criteria if not item["passed"]]
    c4_text = _summary(_by_id(criteria, "C4"))
    if not failed:
        return {
            "verdict": "pass",
            "reason": (
                "C1–C5 全部满足：收益/回撤/稳定性同向且越带，"
                f"差异不集中于少数月份（{_summary(criteria[0])}）；{c4_text}"
            ),
        }
    detail = "；".join(f"{item['id']} {_summary(item)}" for item in failed)
    passed_ids = [item["id"] for item in criteria if item["passed"]]
    return {
        "verdict": "inconclusive",
        "reason": (
            f"未满足：{detail}。无「方向相反且越带」的情形，按预登记规则既不能判 pass "
            f"也不能判 fail；通过项={passed_ids}。{c4_text}（C4 只表示「差异不由费用解释」，"
            "不表示费用机制被验证）。结论限于本窗口机械与费用口径"
        ),
    }


def _by_id(criteria: Sequence[Mapping[str, Any]], criterion_id: str) -> Mapping[str, Any]:
    for item in criteria:
        if str(item["id"]) == criterion_id:
            return item
    raise VerdictError(f"裁定载荷缺判据 {criterion_id}")


def _summary(item: Mapping[str, Any]) -> str:
    """单条判据的量化摘要（供 verdict_reason 引用，不引入新判据）。"""
    criterion_id = str(item["id"])
    values = item.get("values") or {}
    if criterion_id in ("C1", "C2"):
        return (
            f"Δ={float(values['delta']):+.4%}（δ={float(item['threshold']):.2%}，"
            f"越带={bool(item['over_band'])}，方向={'一致' if item['direction_ok'] else '相反'}）"
        )
    if criterion_id == "C3":
        parts = [
            f"{term['term']} Δ={float(term['delta']):+.6%}"
            f"(δ={float(term['threshold']):.2%},越带={bool(term['over_band'])},"
            f"方向={'一致' if term['direction_ok'] else '相反'})"
            for term in item.get("items") or ()
        ]
        return "；".join(parts)
    if criterion_id == "C4":
        share = _fee_share(values)
        return (
            f"ΔTR_gross={float(values['delta_tr_gross']):+.4%}"
            f"（δ={float(item['threshold']):.2%}）、费用解释占比={share:.1%}、"
            f"Δfee={float(values['fee_treatment']) - float(values['fee_control']):+.2f} 元、"
            f"Δturnover={float(values['delta_turnover']):+.4f}"
        )
    year = values.get("year") or {}
    return (
        f"集中度={float(values['concentration']):.4f}、留出 top3 月后 Δ={float(values['loo_delta']):+.4%}、"
        f"分年同号={year.get('positive')}正/{year.get('negative')}负"
        f"（{year.get('same_sign_share')}，阈值 {year.get('min_share')}）"
    )


def _fee_share(values: Mapping[str, Any]) -> float:
    """费用对配对差的解释占比（``ΔTR_fee / ΔTR``；ΔTR=0 时记 0）。"""
    delta_tr = float(values.get("delta_tr") or 0.0)
    if delta_tr == 0.0:
        return 0.0
    return float(values.get("delta_tr_fee") or 0.0) / delta_tr
