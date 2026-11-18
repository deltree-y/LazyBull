"""DEF-ML-60-v1 R1 测试：标签端点、缺口状态、purge、采样相位、资格与行业桶。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.defensive_ml import features, labels, panel, splits, spec, universe

DAYS = [f"2026010{i}" for i in range(1, 9)]


def _wide(values: dict[str, list[float | None]], days: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=days)


def _prices(close: pd.DataFrame, open_px: pd.DataFrame | None = None) -> dict:
    index = close.index
    columns = close.columns
    frame = lambda value: pd.DataFrame(value, index=index, columns=columns)  # noqa: E731
    return {
        "close_adj": close,
        "open_adj": open_px if open_px is not None else close,
        "amount": frame(1000.0),
        "is_st": frame(0.0),
        "is_suspended": frame(0.0),
        "is_limit_up": frame(0.0),
        "is_limit_down": frame(0.0),
        "tradable": frame(1.0),
    }


def test_label_uses_t_plus_1_close_and_t_plus_61_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(spec.LABEL, "horizon", 2)
    monkeypatch.setitem(spec.LABEL, "entry_offset", 1)
    monkeypatch.setitem(spec.LABEL, "exit_offset", 3)
    close = _wide({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}, DAYS[:5])
    open_px = _wide({"A": [10.0, 11.0, 12.0, 13.0, 15.0]}, DAYS[:5])
    value, book = labels.build_labels(_prices(close, open_px), DAYS[:5])
    # T=0: B=1(close 11) E=3(open 13) -> 13/11-1
    assert value.iloc[0, 0] == pytest.approx(13.0 / 11.0 - 1.0)
    assert book["label_status"].iloc[0, 0] == labels.STATUS_CODE["valid"]
    # T=1: B=2(close 12) E=4(open 15) -> 15/12-1
    assert value.iloc[1, 0] == pytest.approx(15.0 / 12.0 - 1.0)
    # 尾部越界 -> right_censored
    assert book["label_status"].iloc[3, 0] == labels.STATUS_CODE["right_censored"]
    assert book["label_status"].iloc[4, 0] == labels.STATUS_CODE["right_censored"]


def test_label_flags_suspension_and_delisting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(spec.LABEL, "horizon", 2)
    monkeypatch.setitem(spec.LABEL, "entry_offset", 1)
    monkeypatch.setitem(spec.LABEL, "exit_offset", 3)
    close = _wide(
        {
            "A": [10.0, 10.5, 11.0, 11.5, 12.0],  # 正常
            "B": [10.0, 10.5, 11.0, None, 12.0],  # E 日停牌
            "C": [10.0, 10.5, 11.0, None, None],  # 之后终止报价
        },
        DAYS[:5],
    )
    prices = _prices(close)
    prices["is_suspended"] = _wide(
        {"A": [0.0] * 5, "B": [0.0, 0.0, 0.0, 1.0, 0.0], "C": [0.0] * 5}, DAYS[:5]
    ).fillna(0.0)
    _, book = labels.build_labels(prices, DAYS[:5])
    status = book["label_status"]
    assert status.iloc[0]["B"] == labels.STATUS_CODE["exit_suspended"]
    assert status.iloc[0]["C"] == labels.STATUS_CODE["exit_delisted"]
    assert book["exit_suspended"].iloc[0]["B"]
    assert book["exit_delisted"].iloc[0]["C"]


def test_label_reports_entry_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(spec.LABEL, "horizon", 2)
    monkeypatch.setitem(spec.LABEL, "entry_offset", 1)
    monkeypatch.setitem(spec.LABEL, "exit_offset", 3)
    close = _wide({"A": [10.0, None, 11.0, 11.5, 12.0]}, DAYS[:5])
    _, book = labels.build_labels(_prices(close), DAYS[:5])
    assert book["label_status"].iloc[0, 0] == labels.STATUS_CODE["entry_missing"]


def test_purge_requires_label_maturity_before_cutoff() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103", "20240104"],
            "label_end_date": ["20240302", "20240410", "20240220"],
            "label_status": ["valid"] * 3,
        }
    )
    mask = splits.purge_mask(frame, "20240401")
    assert list(mask) == [True, False, True]


def test_split_masks_apply_window_and_purge() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20230103", "20230109", "20230601", "20230602"],
            "label_end_date": ["20230303", "20230309", "20230901", "20230902"],
            "label_status": ["valid", "valid", "valid", "right_censored"],
        }
    )
    fold = {
        "train_days": ["20230103", "20230109"],
        "val_days": ["20230601", "20230602"],
        "V": "20230401",
        "C": "20230901",
    }
    train, val = splits.split_masks(frame, fold)
    assert list(train) == [True, True, False, False]
    assert list(val) == [False, False, False, False]


def test_sampled_dates_follow_fixed_phase() -> None:
    calendar = [f"2024010{i}" for i in range(1, 10)] + [f"2024011{i}" for i in range(0, 6)]
    sampled = splits.sampled_dates(calendar)
    picked = [i for i, day in enumerate(calendar) if day in sampled]
    assert picked[:3] == [0, 5, 10]
    assert all((i - picked[0]) % 5 == 0 for i in picked)


def test_refresh_dates_are_semiannual() -> None:
    calendar = [f"{year}{month:02d}{day:02d}" for year in range(2019, 2027)
                for month in range(1, 13) for day in (1, 15)]
    calendar = sorted(d for d in calendar if d <= "20261231")
    points = splits.refresh_dates(calendar, "20261029")
    assert points[0] == "20190401"
    assert len(points) == 16
    assert points[1] == "20191001"


def test_list_age_uses_full_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    basic = pd.DataFrame(
        {"ts_code": ["OLD.SZ", "NEW.SZ"], "market": ["主板", "主板"],
         "list_date": ["20100104", "20231229"]}
    )
    full = [f"20{year}{month:02d}01" for year in range(10, 26) for month in range(1, 13)]
    full = sorted(full)
    dates = full[-300:]
    age, missing = universe._list_age(basic, dates, ["OLD.SZ", "NEW.SZ"], full)
    assert not missing.any()
    # OLD.SZ 上市日 20100104 落在全历第 2 个位置（20100201）
    expected = len(full) - 1 - int(np.searchsorted(np.array(full), "20100104"))
    assert age[-1, 0] == expected
    # NEW.SZ 于 2023-12-29 上市：窗口内年龄明显小于 OLD
    assert 0 <= age[-1, 1] < age[-1, 0]


def test_industry_bucket_marks_financial_and_unknown() -> None:
    """F01 修复后的口径：可核一级映射优先；只有彻底无映射的旧码才是 UNKNOWN。"""
    pit = pd.DataFrame(
        {
            "ts_code": ["A", "B", "C", "D"],
            "in_date": ["20200101"] * 4,
            "out_date": [None, None, "20210101", None],
            "l1_code": ["480000", "220000", "220000", "470000"],
            "l1_name": ["银行", None, "农林牧渔", None],
        }
    )
    dates = ["20200601", "20210601"]
    matrix, names = universe.industry_bucket_matrix(pit, dates, ["A", "B", "C", "D"])
    assert names[matrix[0, 0] - 1] == "FINANCIAL"
    # B 的名称缺失，但 220000 已由 C 提供可核名称 ⇒ 判为非金融并归入该一级桶
    assert names[matrix[0, 1] - 1] == "220000"
    assert names[matrix[0, 2] - 1] == "220000"
    # D 的 470000 在代码表与 PIT 中都没有名称 ⇒ 身份未知
    assert names[matrix[0, 3] - 1] == "UNKNOWN"
    assert matrix[1, 2] == 0


def test_rolling_mdd_matches_hand_calculation() -> None:
    close = pd.DataFrame(
        {"A": [10.0, 12.0, 9.0, 11.0], "B": [10.0, 9.0, 8.0, 7.0]}, index=DAYS[:4]
    )
    mdd = features.rolling_mdd120(close, window=4, min_valid=4)
    assert mdd[3, 0] == pytest.approx(1 - 9.0 / 12.0)
    assert mdd[3, 1] == pytest.approx(1 - 7.0 / 10.0)
    assert np.isnan(mdd[1, 0])


def test_return_windows_do_not_fill_missing_as_zero() -> None:
    close = pd.DataFrame({"A": [10.0, None, 12.0, 13.0]}, index=DAYS[:4])
    ret = panel.daily_returns(close)
    assert np.isnan(ret.iloc[1, 0])
    assert np.isnan(ret.iloc[2, 0])
    assert ret.iloc[3, 0] == pytest.approx(13.0 / 12.0 - 1.0)
    window = features._ret_window(close, 1)
    assert np.isnan(window.iloc[1, 0])


def test_market_series_ignores_missing_returns() -> None:
    returns = pd.DataFrame(
        {"A": [0.10, 0.02], "B": [np.nan, 0.04], "C": [0.02, np.nan]}, index=DAYS[:2]
    )
    mask = pd.DataFrame(True, index=returns.index, columns=returns.columns)
    bench = panel.market_series(returns, mask)
    assert bench["n_valid"].tolist() == [2, 2]
    assert bench["mkt_ret"].iloc[0] == pytest.approx(0.06)
    assert bench["mkt_ret"].iloc[1] == pytest.approx(0.03)


def test_norm_day_unifies_partition_and_row_formats() -> None:
    assert panel.norm_day("2026-10-29") == "20261029"
    assert panel.norm_day("20261029") == "20261029"
