# -*- coding: utf-8 -*-
"""P7 候选一 PRG-M1-v1 的合成验收（协议 §9 必测案例）。

全部用人工样本，不依赖真实配置与生产数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.peer_return_gap import clock, graph, inputs, score, spec, targets
from scripts.v2_p7.peer_return_gap.common import ResearchError, round12

SPRING = {f"202102{day:02d}" for day in (11, 12, 15, 16, 17)}


def trading_days(start: str, end: str, skip: set[str] | None = None) -> list[str]:
    frames = pd.bdate_range(start=start, end=end)
    days = [item.strftime("%Y%m%d") for item in frames]
    return [day for day in days if day not in (skip or set())]


def correlated_excess(
    codes: list[str], n_days: int = 260, seed: int = 7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=n_days)
    values = {
        code: factor + 0.05 * rng.normal(size=n_days) for code in codes
    }
    return pd.DataFrame(values, index=[f"D{i:04d}" for i in range(n_days)])


def test_clock_spring_short_month_and_year_end() -> None:
    days = trading_days("2019-11-01", "2021-03-31", SPRING | {"20200101"})
    rows = clock.month_clock(days, "20191202", "20210301", anchor="20191231")
    months = [row["month"] for row in rows]
    assert months[0] == "201912"
    assert months[-1] == "202102"
    assert "202103" not in months
    december = next(row for row in rows if row["month"] == "201912")
    assert december["signal_date"] == "20191231"
    assert december["exec_date"] == "20200102"
    assert december["formation_date"] == "20191129"
    february = next(row for row in rows if row["month"] == "202102")
    assert february["signal_date"] == "20210226"
    meta = clock.describe_clock(rows, days)
    assert meta["short_months"]["202102"] == 15
    assert "201912" in meta["cross_year_months"]


def test_clock_right_censoring_does_not_fake_month_end() -> None:
    days = trading_days("2020-01-01", "2021-03-31", SPRING)
    rows = clock.month_clock(days, "20200102", "20210226", anchor="20200123")
    last = rows[-1]
    assert last["month"] == "202101"
    assert last["exec_date"] == "20210201"
    with pytest.raises(ResearchError):
        clock.month_clock(days, "20200102", "20210227", anchor="20200123")


def test_clock_left_window_never_overlaps_signal_month() -> None:
    days = trading_days("2019-01-01", "2021-12-31", SPRING)
    rows = clock.month_clock(days, "20191202", "20210301", anchor="20191231")
    for row in rows:
        start = max(0, int(row["formation_index"]) - spec.CORR_WINDOW + 1)
        left = set(days[start:int(row["formation_index"]) + 1])
        signal = set(days[int(row["formation_index"]) + 1:int(row["signal_index"]) + 1])
        assert not (left & signal)


def test_score_simple_example_and_arm_order() -> None:
    beta_frame = pd.DataFrame(
        [
            {"center": "A", "beta": 1.2, "pair_n": 200.0},
            {"center": "B", "beta": 1.0, "pair_n": 200.0},
        ]
    )
    month_frame = pd.DataFrame(
        [
            {"ts_code": "A", "L": -0.01, "valid": True},
            {"ts_code": "B", "L": -0.02, "valid": True},
        ]
    )
    excess = {"A": -0.01, "B": -0.02}
    peers = {"A": [], "B": []}
    for index in range(10):
        excess[f"PA{index}"] = 0.04
        excess[f"PB{index}"] = -0.01
        peers["A"].append(f"PA{index}")
        peers["B"].append(f"PB{index}")
    scored, _meta = score.score_centers(
        beta_frame=beta_frame,
        month_frame=month_frame,
        excess_month_by_code=excess,
        peers_by_center=peers,
        base_codes=["A", "B"],
    )
    values = scored.set_index("ts_code")["score"].to_dict()
    assert round12(values["A"]) == round12(0.058)
    assert round12(values["B"]) == round12(0.010)
    industry = {"A": "电子", "B": "电子"}
    peer_arm = score.select_arm(scored, key="score", industry=industry)
    assert [row["ts_code"] for row in peer_arm] == ["A", "B"]
    rev_arm = score.select_arm(scored, key="L", industry=industry, ascending=True)
    assert [row["ts_code"] for row in rev_arm] == ["B", "A"]


def test_formation_requires_ten_peers_and_forbids_self() -> None:
    codes = [f"S{index:02d}" for index in range(14)]
    excess = correlated_excess(codes)
    factor = excess["S02"].to_numpy()
    opposite = pd.DataFrame(
        {
            "S00": -factor + 0.01 * np.random.default_rng(1).normal(size=len(factor)),
            "S01": -factor + 0.01 * np.random.default_rng(2).normal(size=len(factor)),
        },
        index=excess.index,
    )
    mixed = excess.copy()
    mixed[["S00", "S01"]] = opposite
    edges, betas, meta = graph.formation(mixed, codes)
    assert "S00" not in set(edges["center"])
    assert not (edges["center"] == edges["peer"]).any()
    center_beta = betas.set_index("center").loc["S00"]
    assert not bool(center_beta["peer_valid"])
    assert "peers=" in str(center_beta["reason"])


def test_peer_selection_tie_breaks_by_code() -> None:
    rho = pd.Series({"C": 1.0, "B": 0.5, "A": 0.5, "D": 0.5})
    count = pd.Series({"C": 252, "A": 200, "B": 200, "D": 200})
    picked = graph.select_peers(rho, count, "C")
    assert [item["peer"] for item in picked] == ["A", "B", "D"]


def test_peer_month_requires_eight_valid_of_ten() -> None:
    peers = [f"P{index}" for index in range(10)]
    values = pd.Series({peer: 0.02 for peer in peers[:7]})
    value, meta = score.peer_month_value(values, peers)
    assert not np.isfinite(value)
    assert meta["valid"] == 7
    values = pd.Series({peer: 0.02 for peer in peers[:8]})
    value, meta = score.peer_month_value(values, peers)
    assert value == round12(0.02)
    assert meta["valid"] == 8


def test_pairwise_blocked_matches_bruteforce_with_missing_and_reversed_order() -> None:
    rng = np.random.default_rng(11)
    codes = [f"K{index:02d}" for index in range(14)]
    values = rng.normal(size=(240, len(codes)))
    values[10:60, 3] = np.nan
    values[:, 8] = 0.0
    frame = pd.DataFrame(values, index=range(240), columns=codes)
    count_a, rho_a = graph.pairwise_correlation(frame)
    count_b, rho_b = graph.pairwise_correlation(frame.iloc[::-1, ::-1])
    count_c, rho_c = graph.pairwise_correlation(frame, brute_force=True)
    count_b = count_b.reindex(index=count_a.index, columns=count_a.columns)
    rho_b = rho_b.reindex(index=rho_a.index, columns=rho_a.columns)
    assert np.allclose(count_a.to_numpy(), count_b.to_numpy())
    assert np.nanmax(np.abs(rho_a.to_numpy() - rho_c.to_numpy())) < 1e-12
    assert rho_b.loc["K12", "K13"] == pytest.approx(rho_a.loc["K12", "K13"], abs=1e-12)
    assert np.isnan(rho_a.loc["K00", "K08"]) and np.isnan(rho_c.loc["K08", "K00"])


def test_pair_count_179_rejects_and_180_accepts() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=(200, 2))
    values[0:21, 0] = np.nan
    frame = pd.DataFrame(values, columns=["A", "B"])
    count, rho = graph.pairwise_correlation(frame)
    assert int(count.loc["A", "B"]) == 179
    assert graph.select_peers(rho["A"], count["A"], "A") == []
    values2 = values.copy()
    values2[20, 0] = 0.3
    frame2 = pd.DataFrame(values2, columns=["A", "B"])
    count2, rho2 = graph.pairwise_correlation(frame2)
    assert int(count2.loc["A", "B"]) == 180
    assert len(graph.select_peers(rho2["A"], count2["A"], "A")) == 1


def test_fit_beta_rejects_seven_valid_peers() -> None:
    rng = np.random.default_rng(5)
    peers = [f"P{index}" for index in range(10)]
    factor = rng.normal(size=220)
    frame = pd.DataFrame(
        {peer: factor + 0.05 * rng.normal(size=220) for peer in peers}
    )
    frame["C"] = 1.5 * factor + 0.05 * rng.normal(size=220)
    for peer in peers[7:]:
        frame[peer] = np.nan
    record = graph.fit_beta(frame, "C", peers)
    assert not bool(record["peer_valid"])
    for peer in peers[7:]:
        frame[peer] = factor + 0.05 * rng.normal(size=220)
    record = graph.fit_beta(frame, "C", peers)
    assert record["peer_valid"] and np.isfinite(record["beta"])


def test_amount_units_and_paused_days() -> None:
    days = [f"D{index:02d}" for index in range(20)]
    rows = []
    for day in days:
        rows.append(
            {
                "trade_date": day,
                "ts_code": "A",
                "amount": 50000.0,
                "is_suspended": 0,
            }
        )
    frame = pd.DataFrame(rows)
    stats = inputs.amount_statistics(frame, ["A"])
    assert stats.loc["A", "mean_amount_yuan"] == pytest.approx(5.0e7)
    frame.loc[0:4, "is_suspended"] = 1
    frame.loc[0:4, "amount"] = np.nan
    stats = inputs.amount_statistics(frame, ["A"])
    assert stats.loc["A", "amount_known_days"] == 20
    assert stats.loc["A", "mean_amount_yuan"] == pytest.approx(3.75e7)
    short = frame.iloc[0:15]
    stats = inputs.amount_statistics(short, ["A"])
    assert not np.isfinite(stats.loc["A", "mean_amount_yuan"])


def test_select_arm_slots_and_sector_cap() -> None:
    rows = [
        {"ts_code": f"C{index:02d}", "score": 1.0 - index * 0.01, "L": -0.01}
        for index in range(12)
    ]
    industry = {
        f"C{index:02d}": ("电子" if index < 4 else "医药" if index < 8 else "机械")
        for index in range(12)
    }
    picked = score.select_arm(pd.DataFrame(rows), key="score", industry=industry)
    assert len(picked) == 6
    sectors = [industry[row["ts_code"]] for row in picked]
    assert sectors == ["电子", "电子", "医药", "医药", "机械", "机械"]


def test_match_unique_and_empty_when_out_of_tolerance() -> None:
    codes = ["A", "B", "C", "D", "E", "F", "G", "H"]
    base = pd.DataFrame(
        {
            "ts_code": codes,
            "L": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.5],
            "total_mv_yuan": [1.0e10 * (1.0 + 0.005 * index) for index in range(8)],
        }
    )
    scored = pd.DataFrame(
        {
            "ts_code": codes,
            "score": [1.0 - 0.01 * index for index in range(8)],
        }
    )
    industry = {code: "电子" for code in codes}
    industry["H"] = "医药"
    matches = score.match_targets(
        peer_targets=[{"ts_code": "A"}],
        scored=scored,
        base_frame=base,
        industry=industry,
    )
    assert matches.iloc[0]["match"] == "B"
    second = score.match_targets(
        peer_targets=[{"ts_code": "A"}, {"ts_code": "B"}],
        scored=scored,
        base_frame=base,
        industry=industry,
    )
    assert second.iloc[0]["match"] != second.iloc[1]["match"]
    empty = score.match_targets(
        peer_targets=[{"ts_code": "H"}],
        scored=scored,
        base_frame=base,
        industry=industry,
    )
    assert empty.iloc[0]["match"] is None
    assert empty.iloc[0]["reason"] == "no_candidate_within_tolerance"


def test_affordable_shares_lot_min_notional_and_capital_scaling() -> None:
    shares, notional = targets.affordable_shares(10.0, 30000.0)
    assert shares == 2900
    assert notional == 29000.0
    shares, notional = targets.affordable_shares(10.0, 25000.0)
    assert (shares, notional) == (0, 0.0)
    shares, _ = targets.affordable_shares(10.0, 25588.54)
    assert shares == 0
    assert targets.account_params(1_000_000)["b_capital"] == 200000.0
    assert targets.account_params(2_000_000)["b_capital"] == 400000.0
    assert spec.MIN_NOTIONAL == pytest.approx(max(25000.0, 5.0 / 0.0001954), abs=1e-9)


def test_price_scale_invariance_of_excess_returns() -> None:
    days = [f"202001{index:02d}" for index in range(1, 21)]
    rng = np.random.default_rng(9)
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(scale=0.01, size=(20, 4)), axis=0)),
        index=days,
        columns=list("ABCD"),
    )
    eligible = pd.DataFrame(True, index=days, columns=list("ABCD"))
    suspended = pd.DataFrame(False, index=days, columns=list("ABCD"))
    excess, median, _ = graph.build_excess_returns(close, eligible, suspended)
    scaled, median_scaled, _ = graph.build_excess_returns(
        close * 3.7, eligible, suspended
    )
    assert np.allclose(excess.to_numpy(), scaled.to_numpy(), equal_nan=True)
    assert np.allclose(median.to_numpy(), median_scaled.to_numpy(), equal_nan=True)


def test_month_return_ignores_rows_after_signal() -> None:
    days = trading_days("2020-01-01", "2020-03-31")
    signal_days = [day for day in days if "20200203" <= day <= "20200228"]
    rows = [
        {
            "ts_code": "A",
            "trade_date": "20200131",
            "close": 10.0,
            "close_adj": 10.0,
            "is_suspended": 0,
        }
    ]
    for day in signal_days:
        value = 11.0 if day == "20200228" else 10.5
        rows.append(
            {
                "ts_code": "A",
                "trade_date": day,
                "close": value,
                "close_adj": value,
                "is_suspended": 0,
            }
        )
        rows.append(
            {
                "ts_code": "B",
                "trade_date": day,
                "close": 10.0,
                "close_adj": 10.0,
                "is_suspended": 0,
            }
        )
    panel = pd.DataFrame(rows)
    median = pd.Series(0.0, index=signal_days)
    frame = score.month_return_frame(
        panel, formation="20200131", signal="20200228", median=median, days=days
    )
    value = float(frame.set_index("ts_code").loc["A", "L"])
    assert value == pytest.approx(np.log(1.1), abs=1e-12)
    after = panel.copy()
    after.loc[after["trade_date"] > "20200228", "close"] = 999.0
    frame_after = score.month_return_frame(
        after, formation="20200131", signal="20200228", median=median, days=days
    )
    assert (
        frame_after.set_index("ts_code").loc["A", "L"]
        == frame.set_index("ts_code").loc["A", "L"]
    )


def test_round12_half_even() -> None:
    assert round12(0.0000000000005) == 0.0
    assert round12(0.0000000000015) == 2e-12
    assert round12(1.23456789012345) == 1.234567890123
