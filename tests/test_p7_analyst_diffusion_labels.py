"""标签测试：T+1 close / T+21 open、行业中性只取基础域、跨态与删失。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v2_p7.analyst_diffusion_ml import labels


def _calendar(n: int = 30) -> list[str]:
    return [f"2024{m:02d}{d:02d}" for m, d in zip([1] * n, range(1, n + 1))]


def _prices(values: dict[str, list[float]], n: int) -> dict[str, pd.DataFrame]:
    index = _calendar(n)

    def _frame(series: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"000001.SZ": series}, index=index)

    out: dict[str, pd.DataFrame] = {}
    for name, series in values.items():
        out[name] = _frame(series)
    for name, fill in (
        ("is_suspended", 0.0),
        ("is_limit_down", 0.0),
        ("is_limit_up", 0.0),
        ("tradable", 1.0),
    ):
        out.setdefault(name, _frame([fill] * n))
    return out


def test_manual_label_endpoints() -> None:
    n = 30
    close = [float(10 + i) for i in range(n)]
    open_px = [float(10 + i) for i in range(n)]
    open_px[21] = 22.0
    close[1] = 11.0
    prices = _prices({"close_adj": close, "open_adj": open_px}, n)
    raw, book = labels.build_labels(prices, _calendar(n))
    assert raw.iloc[0, 0] == pytest.approx(22.0 / 11.0 - 1.0)
    assert book["label_status"].iloc[0, 0] == labels.STATUS_CODE["valid"]


def test_right_censored_at_tail() -> None:
    n = 30
    close = [10.0] * n
    open_px = [10.0] * n
    prices = _prices({"close_adj": close, "open_adj": open_px}, n)
    raw, book = labels.build_labels(prices, _calendar(n))
    assert book["label_status"].iloc[-1, 0] == labels.STATUS_CODE["right_censored"]
    assert np.isnan(raw.iloc[-1, 0])


def test_entry_missing_when_entry_price_absent() -> None:
    n = 30
    close = [10.0] * n
    close[1] = np.nan
    open_px = [10.0] * n
    prices = _prices({"close_adj": close, "open_adj": open_px}, n)
    _, book = labels.build_labels(prices, _calendar(n))
    assert book["label_status"].iloc[0, 0] == labels.STATUS_CODE["entry_missing"]


def test_exit_suspended_keeps_price_proxy_but_marks_blocked() -> None:
    """端点有报价时标签仍是价格代理，停牌只写未来执行诊断位（协议 §7.3）。"""
    n = 30
    close = [10.0] * n
    open_px = [10.0] * n
    prices = _prices({"close_adj": close, "open_adj": open_px}, n)
    prices["is_suspended"].iloc[21, 0] = 1.0
    raw, book = labels.build_labels(prices, _calendar(n))
    assert book["label_status"].iloc[0, 0] == labels.STATUS_CODE["valid"]
    assert bool(book["exit_suspended"].iloc[0, 0]) is True
    assert bool(book["exit_execution_blocked"].iloc[0, 0]) is True
    assert raw.iloc[0, 0] == pytest.approx(0.0)


def test_exit_missing_price_with_suspension_is_exit_suspended() -> None:
    """端点缺价且可归因停牌 -> exit_suspended（不压缩日历）。"""
    n = 30
    close = [10.0] * n
    open_px = [10.0] * n
    open_px[21] = np.nan
    prices = _prices({"close_adj": close, "open_adj": open_px}, n)
    prices["is_suspended"].iloc[21, 0] = 1.0
    _, book = labels.build_labels(prices, _calendar(n))
    assert book["label_status"].iloc[0, 0] == labels.STATUS_CODE["exit_suspended"]


def test_industry_median_uses_base_domain_and_floor() -> None:
    """行业基准来自基础合格域；不足 5 只的行业整组标 insufficient。"""
    dates = ["20240102"]
    codes = [f"00000{i}.SZ" for i in range(1, 7)]
    raw = pd.DataFrame(
        [[0.10, 0.20, 0.30, 0.40, 0.50, 0.90]], index=dates, columns=codes
    )
    status = pd.DataFrame(
        [[labels.STATUS_CODE["valid"]] * 6], index=dates, columns=codes
    )
    bucket = np.array([[1, 1, 1, 1, 1, 2]], dtype=np.int16)
    base_mask = np.ones((1, 6), dtype=bool)
    neutral, medians = labels.industry_neutralize(raw, status, base_mask, bucket)

    assert medians[0, 0] == pytest.approx(0.30)
    assert neutral[0, 0] == pytest.approx(0.10 - 0.30)
    assert np.isnan(medians[0, 5])  # 单只股票不构成行业基准

    frame = labels.extract_label_frame(
        raw, status, neutral, medians, np.ones((1, 6), dtype=bool), dates, codes, dates
    )
    last = frame[frame["ts_code"] == "000006.SZ"].iloc[0]
    assert last["label_status"] == "industry_label_insufficient"
    assert frame[frame["ts_code"] == "000001.SZ"].iloc[0]["label_status"] == "valid"


def test_mask_does_not_change_industry_median() -> None:
    """掩码只决定输出行，不参与中位数（防止用持仓子集重排）。"""
    dates = ["20240102"]
    codes = [f"00000{i}.SZ" for i in range(1, 7)]
    raw = pd.DataFrame([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], index=dates, columns=codes)
    status = pd.DataFrame([[labels.STATUS_CODE["valid"]] * 6], index=dates, columns=codes)
    bucket = np.ones((1, 6), dtype=np.int16)
    base_mask = np.ones((1, 6), dtype=bool)
    neutral, medians = labels.industry_neutralize(raw, status, base_mask, bucket)
    assert medians[0, 0] == pytest.approx(0.35)

    mask = np.array([[True, False, False, False, False, False]])
    frame = labels.extract_label_frame(raw, status, neutral, medians, mask, dates, codes, dates)
    assert len(frame) == 1
    assert frame["industry_median"].iloc[0] == pytest.approx(0.35)


def test_insufficient_group_excluded_from_median() -> None:
    """不足 5 只的行业不参与自身中位数，也不产生中性标签。"""
    dates = ["20240102"]
    codes = [f"00000{i}.SZ" for i in range(1, 8)]
    raw = pd.DataFrame([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], index=dates, columns=codes)
    status = pd.DataFrame([[labels.STATUS_CODE["valid"]] * 7], index=dates, columns=codes)
    bucket = np.array([[1, 1, 1, 1, 1, 2, 2]], dtype=np.int16)
    neutral, medians = labels.industry_neutralize(raw, status, np.ones((1, 7), bool), bucket)
    assert medians[0, 0] == pytest.approx(0.3)
    assert np.isnan(medians[0, 5])
    assert np.isnan(neutral[0, 5])
