# -*- coding: utf-8 -*-
"""型 II（风险调整改善）判据测试（F29 草案 v5）。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.f2 import formal
from scripts.v2_p7.f2 import risk_adjusted as ra


def _returns(n: int = 320, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0004, 0.010, n)


def test_delta_sharpe_lower_tail_direction() -> None:
    """下尾分位方向：更好的路径 ⇒ 下界为正；更差的路径 ⇒ 下界为负。"""
    base = _returns()
    better = base + 0.0020
    worse = base - 0.0020
    good = ra.delta_sharpe_gate(base, better, alpha_c=0.05)
    bad = ra.delta_sharpe_gate(base, worse, alpha_c=0.05)
    assert good["lower"] < good["point"] < good["upper"]
    assert good["lower"] > 0.0 and good["pass"] is True
    assert bad["upper"] < 0.0 and bad["pass"] is False


def test_position_gate_thresholds() -> None:
    """完整窗平均 ≥10% 且最差 20% 交易日 ≥20%；缺记录/缺前日/缺市场 ⇒ not_estimable。"""
    full = [f"2022{i:04d}" for i in range(1, 102)]  # 首日为窗前一日
    frame = pd.DataFrame(
        {
            "trade_date": full,
            "nav": [100_000.0] * 101,
            "market_value": [15_000.0] + [30_000.0] * 20 + [11_250.0] * 80,
        }
    )
    days = full[1:]
    market = pd.Series(np.zeros(101), index=full)
    market.iloc[0:20] = -0.02  # 窗口最差 20 天
    windows = [{"start_date": days[0], "end_date": days[-1]}]
    ok = ra.position_gate(frame, windows, market)
    assert ok["estimable"] is True and ok["pass"] is True
    line = ok["lines"][list(ok["lines"])[0]]
    assert line["mean_position"] == pytest.approx(0.1504, abs=1e-3)  # 滞后一期
    assert line["worst_day_position"] > 0.25

    thin = frame.copy()
    thin["market_value"] = 3_000.0  # 平均 3%
    assert ra.position_gate(thin, windows, market)["pass"] is False

    # 缺前一日 ⇒ not_estimable（不得用更早记录替代）
    short = frame[frame["trade_date"] >= days[0]]
    gaps = ra.position_gate(short, windows, market)
    assert gaps["estimable"] is False
    assert gaps["reason"] == "missing_prior_position"

    # 缺窗内记录 ⇒ missing_position_rows
    hole = frame[frame["trade_date"] != days[10]]
    assert ra.position_gate(hole, windows, market)["reason"] == "missing_position_rows"

    # 市场覆盖不足 ⇒ market_coverage_insufficient
    thin_market = pd.Series([np.nan] * 101, index=full)
    assert (
        ra.position_gate(frame, windows, thin_market)["reason"]
        == "market_coverage_insufficient"
    )


def test_position_gate_hard_fail_wins_over_missing() -> None:
    """V03：一窗已确认 fail、另一窗缺证据 ⇒ 整体 fail（不被遮蔽）。"""
    full = [f"2022{i:04d}" for i in range(1, 102)]
    frame = pd.DataFrame(
        {
            "trade_date": full,
            "nav": [100_000.0] * 101,
            "market_value": [0.0] * 101,  # 全空仓 ⇒ 两窗都 fail
        }
    )
    market = pd.Series(np.zeros(101), index=full)
    windows = [
        {"start_date": full[1], "end_date": full[50]},
        {"start_date": full[60], "end_date": full[100]},
    ]
    gate = ra.position_gate(frame, windows, market)
    assert gate["estimable"] is True
    assert gate["pass"] is False
    assert len(gate["failed_windows"]) == 2

    # 把第二窗做成缺证据：整体仍应以硬失败为准
    hole = frame[frame["trade_date"] != full[70]]
    gate2 = ra.position_gate(hole, windows, market)
    assert gate2["pass"] is False
    assert gate2["not_estimable_windows"]


def test_noncrisis_gate_uses_composite_counterfactual() -> None:
    """非危机期对照 = A 腿 + 现金腿（不是纯现金）；不拖累才通过。"""
    days = [f"2019{i:04d}" for i in range(1, 201)]
    a_leg = pd.Series(np.linspace(700_000.0, 900_000.0, 200), index=days)
    # 袖子腿 = 30 万本金按年化 5% 增长（不是固定金额，否则增长率会被稀释）
    growth = 300_000.0 * (1.0 + 0.05) ** (np.arange(200) / 252.0)
    good = ra.noncrisis_gate(a_leg, a_leg + growth, [], calendar=days)
    assert good["estimable"] is True
    assert good["delta_cagr_pp"] > 0.0 and good["pass"] is True

    # 同等本金的袖子腿负增长 ⇒ 相对现金腿拖累
    bad_growth = 300_000.0 * (1.0 - 0.05) ** (np.arange(200) / 252.0)
    drag = ra.noncrisis_gate(a_leg, a_leg + bad_growth, [], calendar=days)
    assert drag["pass"] is False


def test_windows_excluded_from_noncrisis() -> None:
    """崩盘窗内的收益先计算、再按掩码剔除（不删日期桥接）。"""
    days = [f"2020{i:04d}" for i in range(1, 201)]
    a_leg = pd.Series(np.linspace(700_000.0, 780_000.0, 200), index=days)
    windows = [{"start_date": "20200050", "end_date": "20200120"}]
    full = ra.noncrisis_gate(a_leg, a_leg + 320_000.0, [], calendar=days)
    masked = ra.noncrisis_gate(a_leg, a_leg + 320_000.0, windows, calendar=days)
    assert masked["noncrisis_days"] < full["noncrisis_days"]


def _component(estimable: bool, passed: Any, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"estimable": estimable, "pass": passed}
    payload.update(extra)
    return payload


def test_composite_gate_risk_adjusted() -> None:
    """型 II 三向 verdict、优先级与 not_applicable 跳过。"""
    ok = {
        "delta_sharpe": _component(True, True, calibrated=True, lower=0.02),
        "delta_maxdd": _component(True, True, calibrated=True, lower=0.5),
        "correlation": _component(True, True, upper_max=0.2),
        "E4": _component(True, True, vetoes=[], not_estimable=[]),
        "position": _component(
            True, True, calibrated=True, mean_position=0.15, worst_day_position=0.25
        ),
        "noncrisis": _component(True, True, calibrated=True, delta_cagr_pp=1.0),
        "E1": {"state": "not_applicable"},
        "Q1": {"state": "not_applicable"},
    }
    gate = formal.composite_gate(ok, sleeve_type="risk_adjusted")
    assert gate["verdict"] == "pass"
    assert "E1" not in gate["applicable"]

    failed = {**ok, "delta_sharpe": _component(True, False, calibrated=True, lower=-0.01)}
    gate = formal.composite_gate(failed, sleeve_type="risk_adjusted")
    assert gate["verdict"] == "fail"
    assert "delta_sharpe" in gate["failed_components"]

    frozen = {**ok, "position": _component(False, None, calibrated=True, reason="missing")}
    gate = formal.composite_gate(frozen, sleeve_type="risk_adjusted")
    assert gate["verdict"] == "inconclusive_frozen"

    with pytest.raises(ValueError):
        formal.composite_gate(ok, sleeve_type="unknown")


def test_composite_gate_enhancement_unchanged() -> None:
    """型 I 默认行为不变（六条 AND）。"""
    ok = {name: _component(True, True) for name in ("E1", "E2", "T2", "Q1", "correlation", "E4")}
    ok["E1"]["power"] = {"detectable_any": True}
    ok["Q1"]["aa_calibration"] = {"ok": True}
    ok["Q1"]["mde"] = {"power_ok": True}
    ok["E4"]["vetoes"] = []
    assert formal.composite_gate(ok)["verdict"] == "pass"
    assert formal.composite_gate(ok)["sleeve_type"] == "enhancement"


def test_e4_exempts_utilization_for_risk_adjusted() -> None:
    """型 II 的 E4 剔除 E4-4，其余否决线保留。"""
    e4 = {
        "lines": {"E4_4_utilization": {"pass": False}},
        "vetoes": ["E4_4_utilization"],
        "not_estimable": [],
        "pass": False,
    }
    out = ra._e4_for_risk_adjusted(e4)
    assert out["vetoes"] == []
    assert out["pass"] is True
    assert out["exempt"] == ["E4_4_utilization"]

    blocked = ra._e4_for_risk_adjusted(
        {**e4, "vetoes": ["E4_4_utilization", "E4_3_capacity"]}
    )
    assert blocked["vetoes"] == ["E4_3_capacity"]
    assert blocked["pass"] is False


def test_all_not_applicable_cannot_pass() -> None:
    """C01：必需组件全被标 not_applicable ⇒ frozen，不是 pass。"""
    skip = {
        name: {"state": "not_applicable"}
        for name in ("delta_sharpe", "delta_maxdd", "correlation", "E4", "position", "noncrisis")
    }
    gate = formal.composite_gate(skip, sleeve_type="risk_adjusted")
    assert gate["verdict"] == "inconclusive_frozen"
    assert "no_applicable_component" in gate["frozen_components"]
    assert len(gate["invalid_not_applicable"]) == 6


def test_uncalibrated_component_is_frozen_not_fail() -> None:
    """D02：新统计门未校准 ⇒ frozen，不得据此判经济失败。"""
    ok = {
        "delta_sharpe": {"estimable": True, "pass": False, "calibrated": False, "lower": -0.05},
        "delta_maxdd": {"estimable": True, "pass": True, "calibrated": False, "lower": 1.0},
        "correlation": {"estimable": True, "pass": True, "upper_max": 0.2},
        "E4": {"estimable": True, "pass": True, "vetoes": [], "not_estimable": []},
        "position": {"estimable": True, "pass": True, "calibrated": False},
        "noncrisis": {"estimable": True, "pass": True, "calibrated": False},
    }
    gate = formal.composite_gate(ok, sleeve_type="risk_adjusted")
    assert gate["verdict"] == "inconclusive_frozen"
    assert "delta_sharpe_uncalibrated" in gate["frozen_components"]


def test_noncrisis_rejects_common_missing_day() -> None:
    """W01-A：两条路径同时缺同一交易日 ⇒ 不得桥接，直接 not_estimable。"""
    days = [f"2019{i:04d}" for i in range(1, 201)]
    a_leg = pd.Series(np.linspace(700_000.0, 900_000.0, 200), index=days)
    combined = a_leg + 300_000.0 * (1.0 + 0.05) ** (np.arange(200) / 252.0)
    ok = ra.noncrisis_gate(a_leg, combined, [], calendar=days)
    assert ok["estimable"] is True
    hole = [d for d in days if d != days[100]]
    assert (
        ra.noncrisis_gate(a_leg.loc[hole], combined.loc[hole], [], calendar=days)["reason"]
        == "missing_calendar_days"
    )
    assert ra.noncrisis_gate(a_leg, combined, [])["reason"] == "calendar_required"


def test_position_gate_invalid_prior_day() -> None:
    """W01-B：窗前一日 NAV=0/NaN/Inf 不得进入统计。"""
    full = [f"2022{i:04d}" for i in range(1, 102)]
    market = pd.Series(np.zeros(101), index=full)
    windows = [{"start_date": full[1], "end_date": full[-1]}]
    base = pd.DataFrame(
        {"trade_date": full, "nav": [100_000.0] * 101, "market_value": [30_000.0] * 101}
    )
    for bad_nav, bad_mv in ((0.0, 30_000.0), (np.nan, 30_000.0), (100_000.0, np.inf)):
        frame = base.copy()
        frame.loc[0, "nav"] = bad_nav
        frame.loc[0, "market_value"] = bad_mv
        gate = ra.position_gate(frame, windows, market)
        assert gate["estimable"] is False
        assert gate["reason"] == "invalid_prior_position"
