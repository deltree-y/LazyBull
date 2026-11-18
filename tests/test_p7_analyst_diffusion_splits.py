"""切分测试：刷新时钟、采样相位、label_end purge 与折硬门。"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.v2_p7.analyst_diffusion_ml import splits


def _business_calendar(start: str, end: str) -> list[str]:
    days = pd.bdate_range(start, end)
    return [day.strftime("%Y%m%d") for day in days]


def test_refresh_dates_anchor_and_step() -> None:
    calendar = _business_calendar("2019-03-25", "2020-12-31")
    points = splits.refresh_dates(calendar)
    assert points[0] == "20190401"
    assert "20191001" in points
    assert all(day >= "20190329" for day in points)


def test_refresh_dates_respect_limit() -> None:
    calendar = _business_calendar("2019-03-25", "2020-12-31")
    points = splits.refresh_dates(calendar, end="20190501")
    assert points == ["20190401"]


def test_sampled_dates_phase_is_fixed() -> None:
    calendar = _business_calendar("2010-01-01", "2010-02-28")
    sampled = splits.sampled_dates(calendar)
    ordered = [day for day in calendar if day in sampled]
    assert ordered[0] == "20100104"
    gaps = {calendar.index(day) - calendar.index(ordered[0]) for day in ordered}
    assert all(gap % 5 == 0 for gap in gaps)


def test_purge_requires_label_maturity() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240101", "20240102", "20240103"],
            "label_end_date": ["20240120", "20240125", "20240210"],
        }
    )
    mask = splits.purge_mask(frame, "20240131")
    assert mask.tolist() == [True, True, False]


def test_split_masks_combine_window_purge_and_status() -> None:
    fold = {"train_days": ["20240101"], "val_days": ["20240102"], "V": "20240131", "C": "20240215"}
    frame = pd.DataFrame(
        {
            "trade_date": ["20240101", "20240101", "20240102"],
            "label_end_date": ["20240120", "20240301", "20240125"],
            "label_status": ["valid", "valid", "right_censored"],
        }
    )
    train, val = splits.split_masks(frame, fold)
    assert train.tolist() == [True, False, False]
    assert val.tolist() == [False, False, False]
    train_loose, _ = splits.split_masks(frame, fold, require_valid=False)
    assert train_loose.tolist() == [True, False, False]


def test_check_fold_requirements_flags_short_windows() -> None:
    short = {
        "fold_id": "f99",
        "L": "20230101",
        "V": "20230601",
        "n_train_days": 20,
        "n_val_days": 3,
    }
    problems = splits.check_fold_requirements(short)
    assert any("train_span" in item for item in problems)
    assert any("train_days" in item for item in problems)
    assert any("val_days" in item for item in problems)

    ok = {"fold_id": "f01", "L": "20130101", "V": "20180701", "n_train_days": 200, "n_val_days": 20}
    assert splits.check_fold_requirements(ok) == []


def test_fold_table_drops_date_lists() -> None:
    folds = [
        {
            "fold_id": "f01",
            "L": "20130401",
            "V": "20181008",
            "C": "20190401",
            "train_days": ["20130401"],
            "val_days": ["20181008"],
            "oos_days": ["20190401"],
            "n_train_days": 1,
            "n_val_days": 1,
            "n_oos_days": 1,
        }
    ]
    table = splits.fold_table(folds)
    assert "train_days" not in table.columns
    assert table["fold_id"].tolist() == ["f01"]
