"""DEF-ML-60-v1 R6 测试：资源决策表三态与报告关键口径。"""

from __future__ import annotations

from typing import Any

from scripts.v2_p7.defensive_ml import report


def _target(all_three: bool, dd: float = 0.03, cagr: float = -0.008) -> dict[str, Any]:
    return {
        "dd_improvement": dd,
        "cagr_delta": cagr,
        "sharpe_delta": 0.0,
        "cagr_vs_cash": 0.01,
        "meets_dd": dd >= 0.03,
        "meets_cagr": cagr >= -0.01,
        "beats_cash": True,
        "all_three": all_three,
    }


def _arms() -> list[dict[str, Any]]:
    return [
        {"arm": "A100", "cagr": 0.164, "max_drawdown": -0.3174, "mdd_loss": 0.3174,
         "sharpe": 0.677, "end": 3_022_302.0},
        {"arm": "A80_CASH20", "cagr": 0.141, "max_drawdown": -0.2896, "mdd_loss": 0.2896,
         "sharpe": 0.637, "end": 2_611_876.0},
        {"arm": "A80_DEF20", "cagr": 0.156, "max_drawdown": -0.2873, "mdd_loss": 0.2873,
         "sharpe": 0.669, "end": 2_860_419.0},
        {"arm": "A80_LOWVOL20", "cagr": 0.145, "max_drawdown": -0.2929, "mdd_loss": 0.2929,
         "sharpe": 0.627, "end": 2_678_186.0},
    ]


def _diagnostics(both_meet: bool, stress_ok: bool, last_adverse: bool = False) -> dict[str, Any]:
    segments = [{"segment": 3, "start": "20250102", "end": "20261029", "arm": "A80_DEF20",
                 "cagr": -0.02 if last_adverse else 0.05, "max_drawdown": -0.1}]
    main = {}
    for capital in ("1000000", "2000000"):
        main[capital] = {
            "arms": _arms(),
            "target_vs_a100": _target(both_meet),
            "target_vs_cash": _target(True, dd=0.002),
            "lowvol_target_vs_a100": _target(False),
            "segments": segments,
            "sleeve_only": {
                "def": {"cagr": 0.113, "max_drawdown": -0.30},
                "lowvol": {"cagr": 0.039, "max_drawdown": -0.32},
            },
        }
    return {
        "main": main,
        "stress": {
            "slippage_doubled": {
                "1000000": _target(stress_ok),
                "2000000": _target(stress_ok),
                "config": {"stress_slippage": "0.001"},
            }
        },
        "correlation": {
            "1000000": {"A100": {"A100": 1.0, "A80_DEF20": 0.9},
                        "A80_DEF20": {"A100": 0.9, "A80_DEF20": 1.0}},
            "2000000": {"A100": {"A100": 1.0}, "A80_DEF20": {"A100": 0.9}},
        },
        "zero_recovery": {"applicable": False, "note": "无缺价持仓", "rows": []},
    }


META = {
    "campaign": "c",
    "generated_at": "2026-11-17T00:00:00Z",
    "protocol_sha256": "x",
    "spec_sha256": "y",
    "prediction": {"rows": 1, "days": 1, "crossed_rate": 0.0, "reload_all_identical": True},
    "machine_hours": 2.0,
    "over_budget": False,
    "gaps": ["g1"],
}


def test_continue_when_both_meet_and_stress_hold() -> None:
    decision = report.decide(_diagnostics(both_meet=True, stress_ok=True))
    assert decision["resource_recommendation"] == "continue_to_admission_preparation"
    assert decision["formal_admission"] == "not_evaluated"
    assert decision["verdict"] is None


def test_limited_when_stress_overturns() -> None:
    decision = report.decide(_diagnostics(both_meet=True, stress_ok=False))
    assert decision["resource_recommendation"] == "limited_evidence_defer"
    assert decision["stress_hold"] is False


def test_stop_when_main_window_fails_without_stress_flag() -> None:
    decision = report.decide(_diagnostics(both_meet=False, stress_ok=False))
    assert decision["resource_recommendation"] == "stop_v1_investment"


def test_decision_never_uses_formal_pass_language() -> None:
    for both in (True, False):
        for stress in (True, False):
            decision = report.decide(_diagnostics(both_meet=both, stress_ok=stress))
            assert decision["resource_recommendation"] in {
                "repair_required",
                "continue_to_admission_preparation",
                "limited_evidence_defer",
                "stop_v1_investment",
            }
            assert decision["verdict"] is None


def test_report_contains_mandatory_sections() -> None:
    text = report.build_report(_diagnostics(both_meet=True, stress_ok=False), META)
    for title in (
        "## 1 一页结论",
        "## 2 四臂主表",
        "## 3 固定分段与相关性",
        "## 4 分位诊断与低波对照",
        "## 5 固定压力",
        "## 6 身份与资源",
        "## 7 未完成项与复跑",
    ):
        assert title in text
    assert "回撤改善分解" in text
    assert "not_evaluated" in text
    assert "limited_evidence_defer" in text
