"""研报事件层测试：身份、时点、FY 配对、数值容差与冲突隔离。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.analyst_diffusion_ml import report_events


def _frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "ts_code": "000001.SZ",
        "report_date": "20240110",
        "report_title": "T",
        "org_name": "券商A",
        "author_name": "张三",
        "quarter": "2024Q4",
        "eps": 1.0,
        "max_price": np.nan,
        "min_price": np.nan,
        "rating": "买入",
    }
    out = []
    for row in rows:
        item = dict(base)
        item.update(row)
        out.append(item)
    return pd.DataFrame(out)


def test_missing_author_does_not_vote() -> None:
    frame = _frame(
        [
            {"report_date": "20240105", "author_name": "", "eps": 1.0},
            {"report_date": "20240110", "author_name": "", "eps": 1.2},
        ]
    )
    events = report_events.build_events(frame)
    assert events["eps_events"].empty
    assert len(events["activity"]) == 2  # 两个报告日各一行（仍计年龄）
    assert not events["activity"]["named_analyst"].any()


def test_multi_name_author_is_one_team() -> None:
    frame = _frame(
        [
            {"report_date": "20240105", "author_name": "张三,李四", "eps": 1.0},
            {"report_date": "20240110", "author_name": "张三,李四", "eps": 1.2},
        ]
    )
    events = report_events.build_events(frame)
    assert len(events["eps_events"]) == 1
    assert events["eps_events"]["direction"].iloc[0] == "up"


def test_same_report_multiple_fy_counts_once() -> None:
    """同一篇研报同时给 FY0/FY1，只按 FY1 形成一次投票。"""
    frame = _frame(
        [
            {"report_date": "20240105", "quarter": "2024Q4", "eps": 1.0},
            {"report_date": "20240105", "quarter": "2024Q4", "eps": 1.0},
            {"report_date": "20240105", "quarter": "2025Q4", "eps": 1.5},
            {"report_date": "20240110", "quarter": "2024Q4", "eps": 1.2},
            {"report_date": "20240110", "quarter": "2025Q4", "eps": 1.8},
        ]
    )
    events = report_events.build_events(frame)
    primary = events["primary"]
    assert len(primary) == 2  # 两个报告日各一条
    latest = primary[primary["report_date"] == "20240110"].iloc[0]
    assert latest["absolute_fy"] == 2025
    # 主预测期是 FY1，因此只有 FY2025 的修正进入事件
    assert set(events["eps_events"]["absolute_fy"]) == {2025}


def test_eps_pair_requires_same_absolute_fy() -> None:
    frame = _frame(
        [
            {"report_date": "20240105", "quarter": "2024Q4", "eps": 1.0},
            {"report_date": "20240110", "quarter": "2025Q4", "eps": 2.0},
        ]
    )
    events = report_events.build_events(frame)
    assert events["eps_events"].empty  # 不同 FY 不配对


def test_first_coverage_is_not_upgrade() -> None:
    frame = _frame([{"report_date": "20240110", "eps": 1.5}])
    events = report_events.build_events(frame)
    assert events["eps_events"].empty


def test_lookback_boundary_180_days() -> None:
    inside = _frame(
        [
            {"report_date": "20240101", "epoch": None, "eps": 1.0},
            {"report_date": "20240629", "eps": 1.2},  # 180 天
        ]
    )
    outside = _frame(
        [
            {"report_date": "20240101", "eps": 1.0},
            {"report_date": "20240630", "eps": 1.2},  # 181 天
        ]
    )
    # 用真实日历差计算：2024-01-01 -> 2024-06-29 是 180 天
    assert len(report_events.build_events(inside)["eps_events"]) == 1
    assert len(report_events.build_events(outside)["eps_events"]) == 0


def test_unordered_input_does_not_leak_future() -> None:
    rows = [
        {"report_date": "20240110", "eps": 1.2},
        {"report_date": "20240105", "eps": 1.0},
    ]
    forward = report_events.build_events(_frame(rows))["eps_events"]
    backward = report_events.build_events(_frame(list(reversed(rows))))["eps_events"]
    assert len(forward) == 1 and len(backward) == 1
    assert forward["report_date"].iloc[0] == "20240110"
    assert forward["eps_old"].iloc[0] == pytest.approx(1.0)
    assert forward["eps_new"].iloc[0] == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("old", "new", "direction"),
    [
        (-1.0, -0.5, "up"),
        (-1.0, -1.5, "down"),
        (1.0, 1.005, "flat"),  # 容差 = max(0.01, 0.01*|old|) = 0.01
        (1.0, 1.02, "up"),
        (0.001, 0.02, "up"),
    ],
)
def test_eps_direction_and_tolerance(old: float, new: float, direction: str) -> None:
    frame = _frame(
        [
            {"report_date": "20240105", "eps": old},
            {"report_date": "20240110", "eps": new},
        ]
    )
    events = report_events.build_events(frame)["eps_events"]
    assert events["direction"].iloc[0] == direction


def test_eps_relative_clipped_and_floor_applied() -> None:
    frame = _frame(
        [
            {"report_date": "20240105", "eps": 0.001},
            {"report_date": "20240110", "eps": 5.0},
        ]
    )
    events = report_events.build_events(frame)["eps_events"]
    assert events["eps_rel"].iloc[0] == pytest.approx(2.0)  # clip 到 +2


def test_target_and_rating_pairing() -> None:
    frame = _frame(
        [
            {"report_date": "20240105", "target": None, "max_price": 10.0, "min_price": np.nan,
             "rating": "增持"},
            {"report_date": "20240110", "max_price": 11.0, "min_price": 10.0, "rating": "买入"},
        ]
    )
    events = report_events.build_events(frame)
    target = events["target_events"]
    assert len(target) == 1
    assert target["direction"].iloc[0] == "up"
    # 目标价 = 有效 max/min 的均值：(11+10)/2 = 10.5 对 10.0 -> +5%
    assert target["target_rel"].iloc[0] == pytest.approx(10.5 / 10.0 - 1.0)
    rating = events["rating_events"]
    assert len(rating) == 1
    assert rating["direction"].iloc[0] == "up"


def test_target_single_sided_value_used() -> None:
    frame = _frame([{"report_date": "20240110", "max_price": 12.0, "min_price": np.nan}])
    prepared = report_events.prepare_fields(report_events.normalize(frame)[0])
    assert prepared["target_value"].iloc[0] == pytest.approx(12.0)


def test_unknown_rating_is_nan_not_vote() -> None:
    frame = _frame([{"report_date": "20240110", "rating": "外星评级"}])
    prepared = report_events.prepare_fields(report_events.normalize(frame)[0])
    assert np.isnan(prepared["rating_value"].iloc[0])


def test_conflict_group_is_isolated_with_row_sha() -> None:
    frame = _frame(
        [
            {"report_date": "20240110", "eps": 1.0},
            {"report_date": "20240110", "eps": 1.3},
        ]
    )
    normalized, audit = report_events.normalize(frame)
    assert len(normalized) == 0
    assert audit["conflict_identity_groups"] == 1
    assert audit["conflict_row_sha"]  # 原行 SHA 留证


def test_visibility_maps_to_first_trading_day_and_lag() -> None:
    days = ["20240105", "20240108", "20240109", "20240110"]
    dates = pd.Series(["20240106", "20240108", "20240110"])
    base = report_events.visibility_map(dates, days, 0)
    lagged = report_events.visibility_map(dates, days, 1)
    assert list(base) == ["20240108", "20240108", "20240110"]
    assert list(lagged) == ["20240109", "20240109", ""]


def test_report_date_on_t_day_enters_t_but_excluded_when_lagged() -> None:
    days = ["20240109", "20240110", "20240111"]
    frame = _frame([{"report_date": "20240110", "eps": 1.0}])
    events_base = report_events.build_events(frame, trading_days=days, lag_trading_days=0)
    events_lag = report_events.build_events(frame, trading_days=days, lag_trading_days=1)
    assert events_base["activity"]["report_date"].iloc[0] == "20240110"
    assert events_lag["activity"]["report_date"].iloc[0] == "20240111"
