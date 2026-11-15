# -*- coding: utf-8 -*-
"""P7 正式 T2/T3 机械件回归：统计量、面板模拟器与复合门优先级。

覆盖点
======
- 年化夏普 / 块自举下界 / 配对 ΔMaxDD 自举的方向与符号；
- E2 制度重排三态（pass / fail / frozen）；
- 约束账户模拟器：整手、现金非负、到期卖出、零假设换标签不改日历；
- 复合门优先级：**功效足够的经济失败 ⇒ fail**；否则不可估计/校准失败 ⇒ frozen。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.f2 import formal
from scripts.v2_p7.f2.panel import BatchPanel, CorpEvent, simulate


def _panel(
    *,
    codes=("A", "B"),
    days=("20240102", "20240103", "20240104"),
    opens=(10.0, 10.0),
    closes=((10.0, 11.0, 12.0), (10.0, 9.0, 9.0)),
    selected=("A", "B"),
    entry_tradable=(True, True),
    flow_date="20240104",
) -> BatchPanel:
    n_codes, n_days = len(codes), len(days)
    open_raw = np.asarray(opens, dtype=float)[:, None] * np.ones((1, n_days))
    close_raw = np.asarray(closes, dtype=float)
    pre_close = np.vstack([open_raw[:, 0], close_raw[:, :-1]]).T
    return BatchPanel(
        notice_id="N1",
        signal_date="20231229",
        entry_date=days[0],
        flow_date=flow_date,
        codes=tuple(codes),
        days=tuple(days),
        open_raw=open_raw,
        close_raw=close_raw,
        pre_close=pre_close,
        suspended=np.zeros((n_codes, n_days), dtype=bool),
        limit_up=np.zeros((n_codes, n_days), dtype=bool),
        limit_down=np.zeros((n_codes, n_days), dtype=bool),
        has_row=np.ones((n_codes, n_days), dtype=bool),
        is_st=np.zeros((n_codes, n_days), dtype=bool),
        entry_tradable=np.asarray(entry_tradable, dtype=bool),
        sellable_close=np.ones((n_codes, n_days), dtype=bool),
        sell_open_ok=np.ones((n_codes, n_days), dtype=bool),
        selected=tuple(selected),
        bench_return=0.0,
        corp=tuple(() for _ in codes),
    )


def test_simulate_lot_rounding_and_exit() -> None:
    """整手买入、到期卖出、期末无持仓；现金非负。"""
    panel = _panel()
    frame = simulate({"N1": panel}, {"N1": ["A"]}, principal=100_000.0)
    assert list(frame.columns) == ["trade_date", "nav", "market_value"]
    first = frame.iloc[0]
    assert first["market_value"] > 0.0
    # 单槽预算 100000/6=16666.67 → 1600 股 @10 元
    assert first["market_value"] == pytest.approx(16_000.0)
    assert frame.iloc[-1]["market_value"] == 0.0  # 到期已卖出
    assert frame.iloc[-1]["nav"] > 100_000.0


def test_simulate_unfillable_entry_stays_cash() -> None:
    """入场不可交易（涨停/停牌）⇒ 不建仓、不重试。"""
    panel = _panel(entry_tradable=(False, False))
    frame = simulate({"N1": panel}, {"N1": ["A"]}, principal=100_000.0)
    assert (frame["market_value"] == 0.0).all()
    assert frame["nav"].iloc[-1] == pytest.approx(100_000.0)


def test_event_excess_includes_unfilled_as_cash() -> None:
    """未成交事件的残存 = 0 − 基准收益（不得剔除）。"""
    panel = _panel(entry_tradable=(False, True))
    out = formal.event_excess({"N1": panel}, {"N1": ["A", "B"]})
    rows = {row["ts_code"]: row for row in out["rows"]}
    assert rows["A"]["filled"] is False
    assert rows["A"]["excess"] == pytest.approx(-0.0)
    assert rows["B"]["filled"] is True
    assert rows["B"]["excess"] == pytest.approx(9.0 / 10.0 - 1.0)


def test_annualized_sharpe_and_bootstrap() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0006, 0.01, size=500)
    point = formal.annualized_sharpe(returns)
    assert point is not None
    line = formal.block_bootstrap_sharpe_lower(returns, resamples=500)
    assert line["lower"] is not None and line["lower"] < point < line["upper"]
    assert formal.annualized_sharpe(np.zeros(100)) is None


def test_delta_maxdd_bootstrap_positive_for_improvement() -> None:
    """组合回撤更浅 ⇒ ΔMaxDD 点估计与下界为正（改善为正）。"""
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 0.01, size=600)
    base[100:130] -= 0.02
    arm = base.copy() * 0.6
    out = formal.block_bootstrap_delta_maxdd(base, arm, resamples=500, block=10)
    assert out["point"] > 0.0
    assert out["lower"] > 0.0


def test_e2_regime_three_states() -> None:
    dates = pd.bdate_range("2018-01-01", periods=400, freq="B").strftime("%Y%m%d")
    index = pd.Index(dates)
    base = pd.Series(np.full(len(index), 100.0), index=index)
    # 候选每月稳定跑赢 0.5% ⇒ pass
    growth = np.cumprod(np.full(len(index), 1.0 + 0.005 / 21.0))
    positive = pd.Series(100.0 * growth, index=index)
    pass_state = formal.e2_regime(positive, base, str(index[0]))
    assert pass_state["state"] == "pass"
    # 候选每月跑输 ⇒ fail
    negative = pd.Series(100.0 / growth, index=index)
    fail_state = formal.e2_regime(negative, base, str(index[0]))
    assert fail_state["state"] == "fail"


def _component(pass_flag: bool, **extra: Any) -> Dict[str, Any]:
    payload = {"estimable": True, "pass": bool(pass_flag)}
    payload.update(extra)
    return payload


def test_composite_gate_precedence() -> None:
    """功效足够的经济失败 ⇒ fail；校准失败 ⇒ frozen；全过 ⇒ pass。"""
    ok: Dict[str, Dict[str, Any]] = {
        "E1": _component(True, power={"detectable_any": True}),
        "E2": _component(True, state="pass"),
        "T2": _component(True),
        "Q1": _component(
            True, aa_calibration={"ok": True}, mde={"power_ok": True}
        ),
        "correlation": _component(True),
        "E4": _component(True, vetoes=[], not_estimable=[]),
    }
    assert formal.composite_gate(ok)["verdict"] == "pass"

    failed_e1: Dict[str, Dict[str, Any]] = {
        **ok,
        "E1": _component(False, power={"detectable_any": True}),
    }
    gate = formal.composite_gate(failed_e1)
    assert gate["verdict"] == "fail"
    assert "E1" in gate["failed_components"]

    underpowered: Dict[str, Dict[str, Any]] = {
        **ok,
        "E1": _component(False, power={"detectable_any": False}),
    }
    gate = formal.composite_gate(underpowered)
    assert gate["verdict"] == "inconclusive_frozen"
    assert "E1_underpowered" in gate["frozen_components"]

    aa_bad: Dict[str, Dict[str, Any]] = {
        **ok,
        "Q1": _component(False, aa_calibration={"ok": False}, mde={"power_ok": True}),
    }
    assert formal.composite_gate(aa_bad)["verdict"] == "inconclusive_frozen"

    # A/A 通过 + 功效足够但门不过 ⇒ 经济失败（不是证据不足）
    q1_economic: Dict[str, Dict[str, Any]] = {
        **ok,
        "Q1": _component(
            False, gate_raw=False, aa_calibration={"ok": True}, mde={"power_ok": True}
        ),
    }
    gate = formal.composite_gate(q1_economic)
    assert gate["verdict"] == "fail"
    assert "Q1" in gate["failed_components"]

    veto: Dict[str, Dict[str, Any]] = {
        **ok,
        "E4": _component(False, vetoes=["E4_4_utilization"], not_estimable=[]),
    }
    assert formal.composite_gate(veto)["verdict"] == "fail"

    missing: Dict[str, Dict[str, Any]] = {**ok, "T2": {"estimable": False}}
    assert formal.composite_gate(missing)["verdict"] == "inconclusive_frozen"


def test_sim_start_guard() -> None:
    """A/A 分母守卫：模拟器首行（首日收盘）偏离期初本金 >20% 即判不可估计。"""
    assert formal._sim_start_ok(292_918.43, 300_000.0)
    assert not formal._sim_start_ok(100_000.0, 300_000.0)
    assert not formal._sim_start_ok(0.0, 300_000.0)
    assert not formal._sim_start_ok(300_000.0, 0.0)


def test_corp_event_adjusts_carry_and_shares() -> None:
    """公司行为：送转增股 + 现金入账；缺价 carry 按理论除权价下调（权益不机械增值）。"""
    panel = _panel(closes=((10.0, 10.0, 12.0), (10.0, 10.0, 10.0)))
    # 除权日（day 1）无行情 ⇒ 走 carry；carry 必须按 (10−0.5)/1.2 下调
    panel.has_row[0, 1] = False
    panel.close_raw[0, 1] = np.nan
    panel.corp = ((CorpEvent(day_index=1, cash_per_share=0.5, share_ratio=0.2),), ())
    frame = simulate({"N1": panel}, {"N1": ["A"]}, principal=100_000.0)
    # 1600 股 × (1+0.2) = 1920 股，价格 (10−0.5)/1.2 ⇒ 市值 15,200；现金 +800
    assert frame.iloc[1]["market_value"] == pytest.approx(15_200.0, rel=1e-9)
    # 除权当日权益与除权前一致（只差费用），不产生机械增值
    assert frame.iloc[1]["nav"] == pytest.approx(frame.iloc[0]["nav"], rel=1e-12)
    assert frame.iloc[-1]["nav"] > 100_000.0
