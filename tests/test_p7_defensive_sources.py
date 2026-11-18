"""DEF-ML-60-v1 输入审计测试：单位断言、行业 PIT、退市追溯、预热窗口（合成数据）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.v2_p7.defensive_ml import sources

CODES = ["000001.SZ", "600000.SH", "600087.SH"]


def _daily_frame(date8: str, amount_is_yuan: bool = False) -> pd.DataFrame:
    """close=10、vol=1000 手、amount=1000 千元，与 low/high 自洽。"""
    amount = 1000.0 * (1000.0 if amount_is_yuan else 1.0)
    return pd.DataFrame(
        {
            "ts_code": CODES,
            "trade_date": [date8] * len(CODES),
            "open": [10.0] * len(CODES),
            "high": [10.2] * len(CODES),
            "low": [9.8] * len(CODES),
            "close": [10.0] * len(CODES),
            "vol": [1000.0] * len(CODES),
            "amount": [amount] * len(CODES),
            "adj_factor": [2.0] * len(CODES),
            "close_adj": [20.0] * len(CODES),
            "is_st": [0] * len(CODES),
            "tradable": [1] * len(CODES),
        }
    )


def _basic_frame(date8: str, mv_is_yuan: bool = False) -> pd.DataFrame:
    """total_share=10000 万股、total_mv=100000 万元（= close * total_share）。"""
    total_mv = 100000.0 * (10000.0 if mv_is_yuan else 1.0)
    return pd.DataFrame(
        {
            "ts_code": CODES,
            "trade_date": [date8] * len(CODES),
            "close": [10.0] * len(CODES),
            "total_mv": [total_mv] * len(CODES),
            "circ_mv": [total_mv] * len(CODES),
            "total_share": [10000.0] * len(CODES),
            "float_share": [8000.0] * len(CODES),
            "turnover_rate": [3.0] * len(CODES),
        }
    )


def _write_day(root: Path, date_dash: str, **kwargs: bool) -> None:
    date8 = date_dash.replace("-", "")
    daily_dir = root / "data" / "clean" / "daily"
    basic_dir = root / "data" / "clean" / "daily_basic"
    daily_dir.mkdir(parents=True, exist_ok=True)
    basic_dir.mkdir(parents=True, exist_ok=True)
    _daily_frame(date8, kwargs.get("amount_is_yuan", False)).to_parquet(
        daily_dir / f"{date_dash}.parquet"
    )
    _basic_frame(date8, kwargs.get("mv_is_yuan", False)).to_parquet(
        basic_dir / f"{date_dash}.parquet"
    )


def test_verify_units_passes_on_consistent_units(tmp_path: Path) -> None:
    _write_day(tmp_path, "2019-04-01")
    result = sources.verify_units(tmp_path, dates=("2019-04-01",))
    assert result["status"] == "pass"
    by_name = {c["check"]: c for c in result["checks"]}
    assert by_name["daily_amount_unit_qian_yuan"]["ratio_inside_low_high"] == 1.0
    assert by_name["daily_basic_total_mv_unit_wan_yuan"]["status"] == "pass"
    assert by_name["close_adj_equals_close_times_factor"]["status"] == "pass"


def test_verify_units_flags_amount_already_in_yuan(tmp_path: Path) -> None:
    _write_day(tmp_path, "2019-04-01", amount_is_yuan=True)
    result = sources.verify_units(tmp_path, dates=("2019-04-01",))
    assert result["status"] == "fail"
    by_name = {c["check"]: c for c in result["checks"]}
    assert by_name["daily_amount_unit_qian_yuan"]["status"] == "fail"


def test_verify_units_flags_total_mv_already_in_yuan(tmp_path: Path) -> None:
    _write_day(tmp_path, "2019-04-01", mv_is_yuan=True)
    result = sources.verify_units(tmp_path, dates=("2019-04-01",))
    by_name = {c["check"]: c for c in result["checks"]}
    assert by_name["daily_basic_total_mv_unit_wan_yuan"]["status"] == "fail"
    assert result["status"] == "fail"


def test_inventory_hashing_optional(tmp_path: Path) -> None:
    _write_day(tmp_path, "2019-04-01")
    hashed = sources.inventory(tmp_path, hash_content=True)
    plain = sources.inventory(tmp_path, hash_content=False)
    assert hashed["clean_daily"]["n_partitions"] == plain["clean_daily"]["n_partitions"] == 1
    assert "content_sha256" in hashed["clean_daily"]
    assert "content_sha256" not in plain["clean_daily"]
    assert hashed["clean_daily"]["first"] == "2019-04-01"


def test_industry_coverage_uses_open_interval_sentinel(tmp_path: Path) -> None:
    _write_day(tmp_path, "2019-04-01")
    pit = pd.DataFrame(
        {
            "code6": ["000001", "600000", "600087"],
            "ts_code": CODES,
            "in_date": ["20140101", "20140101", "20140101"],
            "out_date": [None, "20180102", None],
            "ind_code": ["801780", "801780", "801780"],
            "l1_code": ["801780", "801780", None],
            "l2_code": ["801781", "801781", None],
            "l1_name": ["银行", "银行", None],
            "l2_name": ["银行", "银行", None],
            "l3_name": ["银行", "银行", None],
            "index_code": ["x", "x", None],
            "sw_version": ["SW2021", "SW2021", "UNMAPPED"],
            "source_updated_at": pd.Timestamp("2026-11-12"),
            "code_alias": [True, True, False],
        }
    )
    pit_dir = tmp_path / "data" / "clean"
    pit_dir.mkdir(parents=True, exist_ok=True)
    pit.to_parquet(pit_dir / "sw_industry_pit.parquet")

    result = sources.industry_coverage(tmp_path, dates=("2019-04-01", "2019-04-02"))
    row = result["rows"][0]
    # 000001 开放区间命中；600000 区间已结束；600087 无 L1 -> 1/3
    assert row["mapped_rows"] == 1
    assert row["l1_ratio"] == pytest.approx(1 / 3)
    assert result["rows"][1]["status"] == "missing_partition"
    assert result["unmapped_pit_rows"] == 1
    assert result["open_interval_rows"] == 2


def test_delisted_presence_counts_days_and_last(tmp_path: Path) -> None:
    for date_dash in ("2014-06-03", "2014-06-04", "2019-04-01"):
        _write_day(tmp_path, date_dash)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ts_code": CODES[:2],
            "symbol": ["000001", "600000"],
            "name": ["a", "b"],
            "area": ["x", "x"],
            "industry": ["y", "y"],
            "market": ["主板", "主板"],
            "list_date": ["19910403", "19991110"],
        }
    ).to_parquet(raw_dir / "stock_basic.parquet")

    result = sources.delisted_presence(tmp_path, probes=(("600087.SH", "退市样例"),))
    assert result["probes"]["600087.SH"]["days"] == 3
    assert result["probes"]["600087.SH"]["last"] == "2019-04-01"
    assert result["has_independent_delist_registry"] is False


def test_warmup_requires_252_partitions(tmp_path: Path) -> None:
    for date_dash in ("2011-10-10", "2012-01-04", "2019-04-01"):
        _write_day(tmp_path, date_dash)
    result = sources.warmup_availability(tmp_path)
    assert result["warmup_ok"] is False
    assert result["partitions_before_first_fold"] == 2
    assert result["oldest_partition"] == "2011-10-10"


def test_calendar_span_reports_anchor_presence(tmp_path: Path) -> None:
    cal = pd.DataFrame(
        {
            "exchange": ["SSE"] * 3,
            "cal_date": ["20130401", "20190401", "20261029"],
            "is_open": [1, 1, 1],
        }
    )
    cal_dir = tmp_path / "data" / "clean"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(cal_dir / "trade_cal.parquet")
    result = sources.calendar_span(tmp_path)
    assert result["open_days"] == 3
    assert result["anchor_present"]["20190401"] is True
    assert result["anchor_present"]["20100104"] is False
