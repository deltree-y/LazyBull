"""扩散状态测试：每分析师一票、分母、90 日半开窗口与 breadth 上界。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.analyst_diffusion_ml import diffusion


def _events(eps: list[dict], target: list[dict] | None = None, rating: list[dict] | None = None):
    eps_columns = [
        "ts_code", "analyst_team", "report_date", "absolute_fy", "eps_new", "eps_old",
        "eps_delta", "eps_rel", "direction", "gap_days",
    ]
    target_columns = ["ts_code", "analyst_team", "report_date", "target_rel", "direction", "gap_days"]
    rating_columns = ["ts_code", "analyst_team", "report_date", "rating_delta", "direction", "gap_days"]
    return {
        "eps_events": pd.DataFrame(eps, columns=eps_columns) if eps else pd.DataFrame(columns=eps_columns),
        "target_events": pd.DataFrame(target, columns=target_columns)
        if target
        else pd.DataFrame(columns=target_columns),
        "rating_events": pd.DataFrame(rating, columns=rating_columns)
        if rating
        else pd.DataFrame(columns=rating_columns),
    }


def _eps_row(day: str, team: str, direction: str, rel: float) -> dict:
    return {
        "ts_code": "000001.SZ",
        "analyst_team": team,
        "report_date": day,
        "absolute_fy": 2025,
        "eps_new": 1.0 + rel,
        "eps_old": 1.0,
        "eps_delta": rel,
        "eps_rel": rel,
        "direction": direction,
        "gap_days": 30,
    }


def _run(events: dict, days: list[str], n_active_value: int = 3) -> pd.DataFrame:
    dates = days
    codes = ["000001.SZ"]
    mask = np.ones((len(dates), 1), dtype=bool)
    n_active = np.full((len(dates), 1), n_active_value, dtype=np.int16)
    age = np.zeros((len(dates), 1), dtype=np.float32)
    return diffusion.compute_features(events, dates, codes, mask, n_active, age)


def test_denominator_zero_writes_nan() -> None:
    frame = _run(_events([]), ["20240110"])
    assert np.isnan(frame["ad_eps_net_diffusion_90"].iloc[0])
    assert np.isnan(frame["ad_target_net_diffusion_90"].iloc[0])
    assert frame["ad_n_eps_90"].iloc[0] == 0
    assert frame["ad_active_analysts_90_log"].iloc[0] == pytest.approx(np.log1p(3))


def test_latest_vote_per_analyst_and_direction_counts() -> None:
    events = _events(
        [
            _eps_row("20240105", "A", "down", -0.2),
            _eps_row("20240201", "A", "up", 0.1),  # 同一分析师最新一票覆盖旧的
            _eps_row("20240205", "B", "flat", 0.0),
            _eps_row("20240206", "C", "down", -0.3),
        ]
    )
    frame = _run(events, ["20240301"])
    assert frame["ad_n_eps_90"].iloc[0] == 3
    assert frame["ad_n_eps_up_90"].iloc[0] == 1
    assert frame["ad_n_eps_down_90"].iloc[0] == 1
    assert frame["ad_n_eps_flat_90"].iloc[0] == 1
    assert frame["ad_eps_net_diffusion_90"].iloc[0] == pytest.approx(0.0)
    assert frame["ad_eps_up_ratio_90"].iloc[0] == pytest.approx(1.0 / 3.0)
    assert frame["ad_eps_revision_median_90"].iloc[0] == pytest.approx(0.0)


def test_window_is_half_open_90_days() -> None:
    base = pd.Timestamp("2024-04-01")
    day = base.strftime("%Y%m%d")
    inside = (base - pd.Timedelta(days=89)).strftime("%Y%m%d")
    boundary = (base - pd.Timedelta(days=90)).strftime("%Y%m%d")
    events = _events([_eps_row(inside, "A", "up", 0.1), _eps_row(boundary, "B", "down", -0.1)])
    frame = _run(events, [day])
    assert frame["ad_n_eps_90"].iloc[0] == 1
    assert frame["ad_n_eps_up_90"].iloc[0] == 1


def test_breadth_never_exceeds_one() -> None:
    events = _events([_eps_row("20240110", f"A{i}", "up", 0.1) for i in range(4)])
    frame = _run(events, ["20240120"], n_active_value=4)
    assert frame["ad_eps_revision_breadth_90"].iloc[0] == pytest.approx(1.0)
    assert 0.0 <= frame["ad_eps_revision_breadth_90"].iloc[0] <= 1.0


def test_target_and_rating_net_diffusion() -> None:
    events = _events(
        [],
        target=[
            {"ts_code": "000001.SZ", "analyst_team": "A", "report_date": "20240110",
             "target_rel": 0.2, "direction": "up", "gap_days": 20},
            {"ts_code": "000001.SZ", "analyst_team": "B", "report_date": "20240111",
             "target_rel": -0.2, "direction": "down", "gap_days": 20},
        ],
        rating=[
            {"ts_code": "000001.SZ", "analyst_team": "A", "report_date": "20240110",
             "rating_delta": 0.0, "direction": "flat", "gap_days": 20},
        ],
    )
    frame = _run(events, ["20240120"])
    assert frame["ad_target_net_diffusion_90"].iloc[0] == pytest.approx(0.0)
    assert frame["ad_n_target_90"].iloc[0] == 2
    assert frame["ad_rating_net_diffusion_90"].iloc[0] == pytest.approx(0.0)
    assert frame["ad_n_rating_90"].iloc[0] == 1


def test_mask_controls_output_rows() -> None:
    events = _events([_eps_row("20240110", "A", "up", 0.1)])
    dates = ["20240110", "20240120"]
    mask = np.array([[True], [False]])
    n_active = np.array([[1], [1]], dtype=np.int16)
    age = np.zeros((2, 1), dtype=np.float32)
    frame = diffusion.compute_features(
        events, dates, ["000001.SZ"], mask, n_active, age
    )
    assert len(frame) == 1
    assert frame["trade_date"].iloc[0] == "20240110"
