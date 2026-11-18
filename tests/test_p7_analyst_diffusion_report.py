"""报告与决策测试：资源建议四岔口、必含章节、禁用正式通过语汇。"""

from __future__ import annotations

import pytest

from scripts.v2_p7.analyst_diffusion_ml import report, spec


def _segment(segment: int, cagr: float, mdd_loss: float) -> list[dict]:
    base = {"segment": segment, "start": "20250102", "end": "20261029", "sharpe": 0.5}
    return [
        {**base, "arm": "A100", "cagr": 0.10, "max_drawdown": -0.20, "mdd_loss": 0.20},
        {**base, "arm": "A80_CASH20", "cagr": cagr - 0.02, "max_drawdown": -0.18,
         "mdd_loss": 0.18},
        {**base, "arm": "A80_AD20", "cagr": cagr, "max_drawdown": -0.18 - mdd_loss + 0.18,
         "mdd_loss": mdd_loss},
        {**base, "arm": "A80_AD_SIMPLE20", "cagr": cagr, "max_drawdown": -0.18,
         "mdd_loss": 0.18},
    ]


def _fixture(
    *,
    meets: bool = True,
    stress_ok: bool = True,
    engineering: dict | None = None,
    last_reversal: bool = False,
    unresolved: int = 0,
) -> dict:
    segments = _segment(1, 0.12, 0.18) + _segment(2, 0.09, 0.20) + _segment(3, 0.11, 0.22)
    if last_reversal:
        segments = segments[:-4] + [
            {**item, "cagr": 0.02 if item["arm"] == "A80_AD20" else item["cagr"],
             "mdd_loss": 0.30 if item["arm"] == "A80_AD20" else item["mdd_loss"]}
            for item in segments[-4:]
        ]
    main = {}
    for capital in ("1000000", "2000000"):
        main[capital] = {
            "metrics": [
                {"arm": "A100", "start": 1e6, "end": 2.0e6, "cagr": 0.10,
                 "max_drawdown": -0.20, "mdd_loss": 0.20, "sharpe": 0.8, "volatility": 0.15,
                 "trading_days": 1800},
                {"arm": "A80_CASH20", "start": 1e6, "end": 1.9e6, "cagr": 0.095,
                 "max_drawdown": -0.18, "mdd_loss": 0.18, "sharpe": 0.8, "volatility": 0.14,
                 "trading_days": 1800},
                {"arm": "A80_AD20", "start": 1e6, "end": 2.1e6,
                 "cagr": 0.106 if meets else 0.101, "max_drawdown": -0.19,
                 "mdd_loss": 0.19, "sharpe": 0.9, "volatility": 0.15, "trading_days": 1800},
                {"arm": "A80_AD_SIMPLE20", "start": 1e6, "end": 1.95e6, "cagr": 0.098,
                 "max_drawdown": -0.19, "mdd_loss": 0.19, "sharpe": 0.85, "volatility": 0.15,
                 "trading_days": 1800},
            ],
            "target_vs_a100": {
                "cagr_delta": 0.006 if meets else 0.001,
                "mdd_loss_delta": 0.0,
                "cagr_vs_cash": 0.011 if meets else 0.006,
                "meets_cagr": meets,
                "meets_mdd": True,
                "beats_cash": True,
                "all_three": meets,
            },
            "target_vs_cash": {"cagr_delta": 0.011, "all_three": True},
            "simple_vs_a100": {"cagr_delta": -0.002, "mdd_loss_delta": 0.0, "all_three": False},
            "segments": segments,
            "correlation": {"A100": {"A100": 1.0}},
            "bootstrap": {},
            "non_overlap_correlation": {"ml_vs_a100": 0.5},
            "sleeve": {
                label: {
                    "metrics": {"cagr": 0.12, "max_drawdown": -0.15, "mdd_loss": 0.15,
                                "sharpe": 0.9},
                    "turnover": {"turnover": 3.0, "fees": 1000.0},
                    "exposure": {"mean_exposure": 0.95},
                    "trades": 120,
                }
                for label in ("ML", "SIMPLE")
            },
            "engineering": [],
            "industry": {"max_bucket_count": 2, "cap": 2, "breach_days": 0},
            "zero_recovery": {"held_at_end": 6, "unresolved_positions": 0},
        }
    stress = {"report_lag_1d": {}, "slippage_doubled": {}, "cash_equity_discount_20": {}}
    for capital in ("1000000", "2000000"):
        for name in stress:
            stress[name][capital] = {
                "cagr_delta": 0.004 if stress_ok else -0.001,
                "mdd_loss_delta": 0.002 if stress_ok else 0.02,
            }
    stress["zero_recovery_unresolved"] = {"unresolved_positions": unresolved}
    return {
        "main": main,
        "stress": stress,
        "coverage": {
            "base_eligible_mean": 1350.0,
            "analyst_eligible_mean": 700.0,
            "analyst_eligible_min": 475,
            "scored_mean": 690.0,
            "scored_min": 470,
            "scored_days_below_100": 0,
        },
        "engineering": engineering or {"1000000": [], "2000000": []},
    }


def _closed_meta() -> dict:
    """证据链闭合的 meta（缺门时必须 fail-closed，故非 repair 用例都要带它）。"""
    return {
        "upstream": {
            "dataset_run": "R1d",
            "predictions_run": "R2c",
            "replay_run": "R3c",
            "diagnostics_run": "R5c",
            "models_gate": {"status": "pass"},
            "diagnostics_gates": {
                "input": {"status": "pass"},
                "models": {"status": "pass"},
                "accounts_run": "R3c",
            },
        }
    }


def test_decide_continue_when_all_targets_and_stress_hold() -> None:
    decision = report.decide(_fixture(), _closed_meta())
    assert decision["resource_recommendation"] == "continue_to_admission_preparation"
    assert decision["formal_admission"] == "not_evaluated"
    assert decision["verdict"] is None
    assert decision["evidence_blockers"] == []


def test_decide_stop_when_main_target_missing() -> None:
    decision = report.decide(_fixture(meets=False), _closed_meta())
    assert decision["resource_recommendation"] == "stop_v1_investment"
    assert "未满足第 2 节目标" in decision["reason"]


def test_decide_limited_when_stress_fails() -> None:
    decision = report.decide(_fixture(stress_ok=False), _closed_meta())
    assert decision["resource_recommendation"] == "limited_evidence_defer"


def test_decide_limited_when_last_segment_reverses() -> None:
    decision = report.decide(_fixture(last_reversal=True), _closed_meta())
    assert decision["resource_recommendation"] == "limited_evidence_defer"
    assert decision["last_segment_reversal"]["1000000"] is True


def test_decide_repair_takes_priority_over_economics() -> None:
    engineering = {"1000000": [{"issue": "negative_cash"}], "2000000": []}
    decision = report.decide(_fixture(engineering=engineering), _closed_meta())
    assert decision["resource_recommendation"] == "repair_required"


def test_decide_fail_closed_without_evidence_chain() -> None:
    """没有证据链（缺 meta）时不得给出经济结论。"""
    decision = report.decide(_fixture())
    assert decision["resource_recommendation"] == "repair_required"
    assert decision["evidence_blockers"]


def test_decide_repair_when_stress_missing() -> None:
    diagnostics = _fixture()
    diagnostics["stress"] = {}
    decision = report.decide(diagnostics, _closed_meta())
    assert decision["resource_recommendation"] == "repair_required"
    assert any("压力" in item for item in decision["evidence_blockers"])


def test_zero_recovery_gap_becomes_obstacle_but_not_auto_stop() -> None:
    decision = report.decide(_fixture(unresolved=1), _closed_meta())
    assert decision["zero_recovery_gap"] is True
    assert any("零回收" in item for item in decision["admission_obstacles"])
    assert decision["resource_recommendation"] == "continue_to_admission_preparation"


def test_report_and_summary_share_same_decision() -> None:
    """报告正文与机器摘要必须给出同一个资源建议（首版曾出现两者不一致）。"""
    diagnostics = _fixture(meets=False)
    meta = {
        "campaign": "adml20_v1_20261118r2",
        "protocol_sha256": "b" * 64,
        "generated_at": "2026-11-18T00:00:00Z",
        "identity": {"git": {"commit": "deadbeef", "dirty": False}, "closure": {}},
        "model_summary": {},
        "budget": {},
        "upstream": _closed_meta()["upstream"],
    }
    text = report.build_report(diagnostics, meta)
    summary = report.machine_summary(diagnostics, meta)
    recommendation = summary["decision"]["resource_recommendation"]
    assert recommendation == "stop_v1_investment"
    assert f"`{recommendation}`" in text
    # 封存语义：只封存规格，不判机制
    assert summary["disposition"]["disposition"] == "evidence_insufficient"
    assert "封存裁决" in text
    assert "不记为机制证伪" in text
    assert "不是对分析师扩散机制的否定" in text


def test_report_contains_required_sections_and_no_formal_pass() -> None:
    diagnostics = _fixture()
    meta = {
        "campaign": "adml20_v1_20261118",
        "protocol_sha256": "a" * 64,
        "generated_at": "2026-11-18T00:00:00Z",
        "identity": {"git": {"commit": "deadbeef", "dirty": False}, "closure": {}},
        "model_summary": {
            "folds": 16,
            "models_registered": 16,
            "reload_all_identical": True,
            "oos_metrics": {"mae": 0.05, "rank_ic": 0.02, "rank_ic_days": 1500},
        },
        "budget": {"dataset_seconds": 600, "train_seconds": 300},
    }
    text = report.build_report(diagnostics, meta)
    for header in (
        "## 1 一页结论",
        "## 2 两本金四臂主表",
        "## 3 分年、固定分段、启动阶段与持仓",
        "## 4 特征、扩散分母与模型质量",
        "## 5 固定四压力与未决事项",
        "## 6 身份、测试与资源",
        "## 7 未完成的正式 P7 缺口与复跑",
    ):
        assert header in text
    assert "not_evaluated" in text
    assert "verdict=pass" not in text
    assert "P7 即将通过" not in text
    assert "收益预测" not in text


def test_machine_summary_declares_prescreen() -> None:
    diagnostics = _fixture()
    summary = report.machine_summary(
        diagnostics, {"campaign": "c", "protocol_sha256": "b" * 64, "generated_at": "t"}
    )
    assert summary["formal_admission"] == "not_evaluated"
    assert summary["production_changed"] is False
    assert summary["not_virgin_market_holdout"] is True
    assert summary["candidate_id"] == spec.CANDIDATE_ID
    assert summary["decision"]["resource_recommendation"] in report.RECOMMENDATIONS


@pytest.mark.parametrize("recommendation", report.RECOMMENDATIONS)
def test_recommendations_are_closed_set(recommendation: str) -> None:
    assert recommendation in {
        "continue_to_admission_preparation",
        "limited_evidence_defer",
        "stop_v1_investment",
        "repair_required",
    }
