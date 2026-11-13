# -*- coding: utf-8 -*-
"""P7 T6 逐日帧与指标口径回归（桩件驱动，不碰真实数据/证据根）。

覆盖点
======
- ``per_day_rows``：NAV 只经 ``calculate_nav``；换手只计市场腿；费用累计与当日分列；
  credit_state 逐日字段（借入/出借/授予/归还/死局/迟回收）齐全；
- ``per_sleeve_rows``：``equity = cash + MV − credit``，费用与成交额逐袖归集；
- ``build_metrics``：链式指标与 ``calculate_chain_metrics`` 逐位一致、自然日年化与
  ``calculate_annualized_return`` 一致、回撤事件极值 == 链式 MaxDD、配对差与归因符号正确。
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List

import pandas as pd
import pytest

from scripts.v2_p7.t6.daily import build_arm_frames, per_day_rows
from scripts.v2_p7.t6.metrics import build_metrics, drawdown_episodes
from src.lazybull.v2.common.chain_metrics import calculate_chain_metrics
from src.lazybull.v2.core.accounting.nav import calculate_nav
from src.lazybull.v2.hosts.paper.performance import calculate_annualized_return

DAYS = ("20240102", "20240103", "20240104", "20240105", "20240108")
CAPITAL = {"v20": 700_000.0, "v10": 300_000.0}


def _params() -> Any:
    return SimpleNamespace(arms=("v20", "v10"), physical_capital=1_000_000.0)


def _host(scale: float = 1.0, credit: bool = False) -> Any:
    """两天建仓、两天回撤、一天回升的极小宿主结果（两袖各 2 只持仓由价格体现）。"""
    nav_series: Dict[str, List[Dict[str, Any]]] = {"v20": [], "v10": []}
    balance_rows: List[Dict[str, Any]] = []
    daily_rows: List[Dict[str, Any]] = []
    credit_daily: List[Dict[str, Any]] = []
    prices = {"v20": [1.0, 1.02, 0.98, 0.99, 1.01], "v10": [1.0, 1.01, 0.99, 1.00, 1.02]}
    for sleeve in ("v20", "v10"):
        # 半仓结构：现金 C/2 + 持仓 C/2 手，价格上涨按 hold 计值（cash + MV = 袖总值）
        cash = CAPITAL[sleeve] * 0.5
        holdings = CAPITAL[sleeve] * 0.5
        for pos, day in enumerate(DAYS):
            level = float(prices[sleeve][pos])
            factor = scale**pos  # 首日因子为 1（两臂同起点），逐日按 scale 复利累加
            scaled_cash = cash * factor
            mv = holdings * level * factor
            value = scaled_cash + mv
            sleeve_credit = 1000.0 if (credit and sleeve == "v20" and pos == 2) else 0.0
            if credit and sleeve == "v10" and pos == 2:
                sleeve_credit = -1000.0
            # 借入/出借把 credit 同时搬进 cash 与 credit：E = cash + MV − credit = value（零 P&L）
            nav_series[sleeve].append(
                {
                    "trade_date": day,
                    "nav": value / 1_000_000.0,
                    "total_value": value,
                    "split_index": 0,
                }
            )
            balance_rows.append(
                {
                    "trade_date": day,
                    "sleeve": sleeve,
                    "cash": scaled_cash + sleeve_credit,
                    "credit": sleeve_credit,
                    "market_value": mv,
                    "equity": value,
                }
            )
    for pos, day in enumerate(DAYS):
        daily_rows.append(
            {
                "trade_date": day,
                "physical_cash": 400_000.0,
                "market_filled": {"buy": 0, "sell": 0},
                "scenarios": {},
                "identity_ok": True,
            }
        )
        credit_daily.append(
            {
                "trade_date": day,
                "grants": [],
                "repayments": [],
                "deadlocks": [],
                "late_recoveries": [],
                "recall_queue": [],
                "right_censored": [],
                "credit_by_sleeve": {"v10": 0.0, "v20": 0.0},
                "hold_by_sleeve": {},
                "due_reserve_by_sleeve": {},
                "nonnegative_gate": {"min": 1000.0 * (pos + 1)},
            }
        )
    trades = [
        {
            "trade_date": "20240104",
            "kind": "market_sell",
            "shares": 100,
            "price": 10.0,
            "fee": 6.0,
            "sleeves": {"v20": 60, "v10": 40},
            "fee_split": {"v20": 3.6, "v10": 2.4},
        },
        {
            "trade_date": "20240104",
            "kind": "internal_offset",
            "shares": 200,
            "price": 10.0,
            "fee": 0.0,
            "from_sleeve": "v20",
            "to_sleeve": "v10",
        },
    ]
    fee_total = sum(float(row["fee"]) for row in trades)
    return SimpleNamespace(
        nav_series=nav_series,
        balance_rows=balance_rows,
        daily_rows=daily_rows,
        credit_daily=credit_daily,
        trades=trades,
        identity_failures=[],
        gate_payload={"violation_days": 0},
        fee_book=SimpleNamespace(physical_fee_total=fee_total),
    )


def test_per_day_rows_uses_single_nav_entry() -> None:
    params = _params()
    rows = per_day_rows(_host(), params)
    assert [row["trade_date"] for row in rows] == list(DAYS)
    for row in rows:
        expected = calculate_nav(row["total_value"], 1_000_000.0)
        assert row["nav"] == pytest.approx(expected, abs=1e-15)
        assert sum(row["contribution"].values()) == pytest.approx(row["nav"], abs=1e-12)
        assert row["credit_sum"] == 0.0


def test_per_day_rows_turnover_and_fees_exclude_offset() -> None:
    rows = {row["trade_date"]: row for row in per_day_rows(_host(), _params())}
    day = rows["20240104"]
    assert day["market_notional"] == pytest.approx(100 * 10.0)
    assert day["offset_notional"] == pytest.approx(200 * 10.0)
    assert day["turnover"] == pytest.approx(1000.0 / 1_000_000.0)
    assert day["fee_day"] == pytest.approx(6.0)
    assert day["fee_cum"] == pytest.approx(6.0)
    assert rows["20240105"]["fee_cum"] == pytest.approx(6.0)


def test_per_day_rows_drawdown_is_underwater_curve() -> None:
    rows = per_day_rows(_host(), _params())
    peak = None
    for row in rows:
        peak = row["nav"] if peak is None else max(peak, row["nav"])
        assert row["drawdown"] == pytest.approx(row["nav"] / peak - 1.0, abs=1e-15)
    assert min(row["drawdown"] for row in rows) < 0.0


def test_per_day_rows_credit_state_fields() -> None:
    rows = per_day_rows(_host(credit=True), _params())
    assert rows[2]["borrow_outstanding"] == pytest.approx(1000.0)
    assert rows[2]["lent_outstanding"] == pytest.approx(1000.0)
    assert rows[2]["credit_sum"] == 0.0
    assert rows[2]["gate_min"] == pytest.approx(3000.0)


def test_per_sleeve_rows_attribution() -> None:
    frames = build_arm_frames(_host(credit=True), _params())
    sleeve = {(row["trade_date"], row["sleeve"]): row for row in frames["sleeve"]}
    row = sleeve[("20240104", "v20")]
    assert row["fee_day"] == pytest.approx(3.6)
    assert row["market_notional"] == pytest.approx(600.0)
    assert row["cash_plus_credit"] == pytest.approx(row["cash"] + row["credit"])
    assert row["contribution"] == pytest.approx(row["equity"] / 1_000_000.0)
    assert row["equity"] == pytest.approx(
        row["cash"] + row["market_value"] - row["credit"], abs=1e-9
    )


def test_build_metrics_matches_public_entries() -> None:
    params = _params()
    frames = {
        "vvh": build_arm_frames(_host(scale=1.0), params),
        "vvs": build_arm_frames(_host(scale=1.0), params),
    }
    metrics = build_metrics(params=params, frames=frames)
    daily = frames["vvh"]["daily"]
    chain = calculate_chain_metrics(
        pd.DataFrame(
            {
                "trade_date": [row["trade_date"] for row in daily],
                "nav": [row["nav"] for row in daily],
                "split_index": [row["split_index"] for row in daily],
            }
        )
    )
    stored = metrics["arms"]["vvh"]["chain_metrics"]
    for key in ("total_return", "cagr", "max_drawdown", "volatility", "sharpe"):
        assert stored[key] == pytest.approx(chain[key], abs=1e-12)
    annualized = calculate_annualized_return(
        1_000_000.0, float(daily[-1]["total_value"]), DAYS[-1], DAYS[0]
    )
    assert metrics["arms"]["vvh"]["annualized_natural_days_pct"] == pytest.approx(annualized)
    assert metrics["arms"]["vvh"]["drawdown"]["episode_series_max"] == pytest.approx(
        chain["max_drawdown"], abs=1e-12
    )


def test_build_metrics_paired_identity_and_attribution() -> None:
    params = _params()
    frames = {
        "vvh": build_arm_frames(_host(scale=1.0), params),
        "vvs": build_arm_frames(_host(scale=1.001), params),
    }
    metrics = build_metrics(params=params, frames=frames)
    paired = metrics["paired"]
    assert paired["total_return"]["delta"] > 0.0
    assert paired["turnover"]["control"] == paired["turnover"]["treatment"]
    attribution = paired["attribution"]
    assert attribution["delta_tr_gross"] == pytest.approx(
        attribution["delta_tr"] - attribution["delta_tr_fee"]
    )
    assert attribution["same_sign_turnover_fee"] is True


def test_attribution_scales_with_physical_capital() -> None:
    params = SimpleNamespace(arms=("v20", "v10"), physical_capital=500_000.0)
    frames = {arm: build_arm_frames(_host(), params) for arm in ("vvh", "vvs")}
    metrics = build_metrics(params=params, frames=frames)
    attribution = metrics["paired"]["attribution"]
    fee_delta = float(metrics["paired"]["fees"]["delta"])
    assert attribution["physical_capital"] == pytest.approx(500_000.0)
    assert attribution["delta_fee_over_pc"] == pytest.approx(fee_delta / 500_000.0)
    assert attribution["delta_tr_fee"] == pytest.approx(-attribution["delta_fee_over_pc"])


def test_drawdown_episodes_shapes() -> None:
    nav = pd.Series([1.0, 1.1, 1.0, 1.05, 1.2, 1.15, 1.2])
    dates = [f"2024010{index}" for index in range(1, 8)]
    daily = [
        {"fee_cum": 0.0, "market_notional": 100.0},
        {"fee_cum": 5.0, "market_notional": 100.0},
        {"fee_cum": 9.0, "market_notional": 50.0},
        {"fee_cum": 9.0, "market_notional": 0.0},
        {"fee_cum": 12.0, "market_notional": 80.0},
        {"fee_cum": 12.0, "market_notional": 0.0},
        {"fee_cum": 12.0, "market_notional": 10.0},
    ]
    episodes = drawdown_episodes(nav=nav, dates=dates, daily=daily, physical_capital=1_000_000.0)
    assert len(episodes) == 2
    worst = episodes[0]
    assert worst["depth"] == pytest.approx(1.0 / 1.1 - 1.0)
    assert worst["start_date"] == "20240102"
    assert worst["trough_date"] == "20240103"
    assert worst["end_date"] == "20240105"
    assert worst["fee_share_of_loss"] is not None
    assert all(math.isfinite(item["depth"]) for item in episodes)


def test_fee_share_of_loss_uses_peak_to_trough_amount() -> None:
    """C6-03：费用占**峰谷资产损失金额**，不是费用/初资再除以 depth。"""
    from scripts.v2_p7.t6.metrics import _episode

    values = pd.Series([2.0, 1.8, 2.0]).to_numpy(dtype=float)
    drawdown = pd.Series([0.0, -0.1, 0.0]).to_numpy(dtype=float)
    fee_cum = pd.Series([0.0, 100.0, 100.0]).to_numpy(dtype=float)
    notional = pd.Series([0.0, 0.0, 0.0]).to_numpy(dtype=float)
    row = _episode(
        0,
        2,
        values,
        drawdown,
        ["20240101", "20240102", "20240103"],
        fee_cum,
        notional,
        1_000.0,
    )
    # 峰谷损失 = 1000 × (2.0 − 1.8) = 200，费用 100 ⇒ 50%（旧实现会给 ~100%）
    assert row["loss_amount"] == pytest.approx(200.0)
    assert row["fee_share_of_loss"] == pytest.approx(0.5)
    assert row["fee_pp_to_trough"] == pytest.approx(0.1)


def test_fee_metrics_gross_adds_back_fees() -> None:
    """C6-04：毛利润 = 期末资产 − 初资 + 累计费用；净利非正不影响 fees/毛利。"""
    from scripts.v2_p7.t6.metrics import _fee_metrics

    params = SimpleNamespace(physical_capital=1_000.0)
    daily = [
        {"trade_date": "20200101", "total_value": 1_090.0, "fee_day": 4.0},
        {"trade_date": "20200102", "total_value": 1_090.0, "fee_day": 6.0},
    ]
    trades = [{"fee": 4.0, "kind": "market_buy"}, {"fee": 6.0, "kind": "market_sell"}]
    fees = _fee_metrics(params, trades, daily)
    assert fees["net_profit"] == pytest.approx(90.0)
    assert fees["gross_profit"] == pytest.approx(100.0)
    assert fees["over_gross_profit"] == pytest.approx(0.1)
    assert fees["over_net_profit"] == pytest.approx(10.0 / 90.0)

    loss_daily = [
        {"trade_date": "20200101", "total_value": 995.0, "fee_day": 20.0},
    ]
    loss_fees = _fee_metrics(params, [{"fee": 20.0, "kind": "market_buy"}], loss_daily)
    assert loss_fees["net_profit"] == pytest.approx(-5.0)
    assert loss_fees["gross_profit"] == pytest.approx(15.0)
    assert loss_fees["over_gross_profit"] == pytest.approx(20.0 / 15.0)
    assert loss_fees["over_net_profit"] is None


def test_stationary_sampler_matches_registered_method() -> None:
    """C6-02：抽样机制必须是 stationary（几何块长），不是定长环形块。"""
    import numpy as np

    from scripts.v2_p7.t6.noise import _stationary_index

    rng = np.random.default_rng(7)
    index = _stationary_index(rng=rng, size=4, days=200, block=1)
    assert index.shape == (4, 200)
    assert int(index.min()) >= 0 and int(index.max()) < 200
    adjacent = np.abs(np.diff(index, axis=1) - 1.0) < 1e-12
    assert adjacent.mean() < 0.1  # block=1 ⇒ 近似 i.i.d.

    rng = np.random.default_rng(7)
    long_blocks = _stationary_index(rng=rng, size=4, days=200, block=200)
    adjacent = np.abs(np.diff(long_blocks, axis=1) - 1.0) < 1e-12
    assert adjacent.mean() > 0.5  # 大块长 ⇒ 大量 +1 连续

    again = _stationary_index(rng=np.random.default_rng(7), size=4, days=200, block=1)
    assert np.array_equal(index, again)


def test_bootstrap_is_paired_when_arms_identical() -> None:
    """配对性：两臂收益相同时，全部重采样 Δ 必须严格为 0。"""
    import numpy as np

    from scripts.v2_p7.t6.noise import _boot_deltas

    rng = np.random.default_rng(11)
    returns = rng.normal(0.0005, 0.01, size=120)
    deltas, invalid = _boot_deltas(
        control=returns,
        treatment=returns.copy(),
        block=20,
        replicates=200,
        seed=42,
        metrics=("total_return", "max_drawdown", "volatility"),
        cagr_days=120,
    )
    assert all(int(value) == 0 for value in invalid.values())
    for name, values in deltas.items():
        assert np.all(np.abs(values) < 1e-15), name
