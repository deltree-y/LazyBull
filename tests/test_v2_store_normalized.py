# -*- coding: utf-8 -*-
"""v2 normalized 转换器测试（tmp_path 合成 mini raw，不触碰真实 data/）。

覆盖：convert_calendar / convert_stock_basic / convert_daily（与 build_clean.py 逐日段
同序的直算结果逐值一致）、convert_daily 强制依赖缺失报错、reconcile_daily_vs_clean
报告结构与篡改检出。
"""

import pandas as pd
import pytest

from src.lazybull.data.cleaner import DataCleaner
from src.lazybull.data.loader import DataLoader
from src.lazybull.data.storage import Storage
from src.lazybull.v2.store.normalized import (
    convert_calendar,
    convert_daily,
    convert_stock_basic,
    reconcile_daily_vs_clean,
)

_CODES = ["600000.SH", "000001.SZ"]
_DATES = ["20240102", "20240103", "20240104"]


def _dashed(date: str) -> str:
    return f"{date[:4]}-{date[4:6]}-{date[6:8]}"


def _write_raw(root, name: str, df: pd.DataFrame, date: str | None = None) -> None:
    """写合成 raw（单文件或 YYYY-MM-DD 日分区，与真实 raw 形态一致）。"""
    if date is None:
        df.to_parquet(root / "raw" / f"{name}.parquet", index=False)
    else:
        part = root / "raw" / name
        part.mkdir(parents=True, exist_ok=True)
        df.to_parquet(part / f"{_dashed(date)}.parquet", index=False)


@pytest.fixture
def mini_raw(tmp_path):
    """合成 mini raw：2 股票 × 3 日五表 + trade_cal/stock_basic 单文件。"""
    root = tmp_path
    (root / "raw").mkdir()
    _write_raw(
        root,
        "trade_cal",
        pd.DataFrame(
            {
                "exchange": "SSE",
                "cal_date": [*_DATES, "20240105"],
                "is_open": [1, 1, 1, 0],
                "pretrade_date": ["20240101", "20240102", "20240103", "20240104"],
            }
        ),
    )
    _write_raw(
        root,
        "stock_basic",
        pd.DataFrame(
            {
                "ts_code": _CODES,
                "symbol": ["600000", "000001"],
                "name": ["浦发银行", "平安银行"],
                "area": ["上海", "深圳"],
                "industry": ["银行", "银行"],
                "market": ["主板", "主板"],
                "list_date": ["19991110", "19910403"],
            }
        ),
    )
    for i, date in enumerate(_DATES):
        _write_raw(
            root,
            "daily",
            pd.DataFrame(
                {
                    "ts_code": _CODES,
                    "trade_date": [date] * 2,
                    "open": [10.0 + i, 20.0],
                    "high": [10.5 + i, 20.5],
                    "low": [9.5 + i, 19.5],
                    "close": [10.2 + i, 20.2],
                    "pre_close": [10.1 + i, 20.1],
                    "change": [0.1, 0.1],
                    "pct_chg": [1.0, 0.5],
                    "vol": [1000.0, 2000.0],
                    "amount": [10200.0, 40400.0],
                }
            ),
            date,
        )
        _write_raw(
            root,
            "adj_factor",
            pd.DataFrame({"ts_code": _CODES, "trade_date": [date] * 2, "adj_factor": [2.0, 3.0]}),
            date,
        )
        _write_raw(
            root,
            "suspend",
            (
                pd.DataFrame(
                    {
                        "ts_code": [_CODES[1]],
                        "trade_date": [date],
                        "suspend_timing": "全天停牌",
                        "suspend_type": "S",
                    }
                )
                if i == 0
                else pd.DataFrame(
                    columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"]
                )
            ),
            date,
        )
        _write_raw(
            root,
            "stk_limit",
            pd.DataFrame(
                {
                    "trade_date": [date] * 2,
                    "ts_code": _CODES,
                    "up_limit": [11.0, 22.0],
                    "down_limit": [9.0, 18.0],
                }
            ),
            date,
        )
        _write_raw(
            root,
            "stock_st",
            pd.DataFrame(columns=["ts_code", "name", "trade_date", "type", "type_name"]),
            date,
        )
    return root


def _direct_clean_daily(loader: DataLoader, cleaner: DataCleaner, date: str) -> pd.DataFrame:
    """与 build_clean.py 逐日段同序的直算（测试基准）。"""
    storage = loader.storage
    daily_clean = cleaner.clean_daily(
        storage.load_raw_by_date("daily", date), storage.load_raw_by_date("adj_factor", date)
    )
    stock_basic_clean = cleaner.clean_stock_basic(storage.load_raw("stock_basic"))
    suspend_raw = storage.load_raw_by_date("suspend", date)
    limit_raw = storage.load_raw_by_date("stk_limit", date)
    st_raw = storage.load_raw_by_date("stock_st", date)
    return cleaner.add_tradable_universe_flag(
        daily_clean,
        stock_basic_clean,
        stock_st_df=cleaner.clean_stock_st(st_raw) if st_raw is not None and len(st_raw) else None,
        suspend_info_df=(
            cleaner.clean_suspend_info(suspend_raw)
            if suspend_raw is not None and len(suspend_raw)
            else None
        ),
        limit_info_df=(
            cleaner.clean_limit_info(limit_raw)
            if limit_raw is not None and len(limit_raw)
            else None
        ),
        min_list_days=365,
    )


class TestConvertSingles:
    def test_convert_calendar(self, mini_raw):
        cleaner = DataCleaner()
        out = convert_calendar(mini_raw / "raw", cleaner, mini_raw / "normalized")
        df = pd.read_parquet(out)
        assert out.name == "trade_cal.parquet"
        assert df["cal_date"].tolist() == sorted(_DATES + ["20240105"])
        assert df["is_open"].dtype.kind == "i"

    def test_convert_stock_basic(self, mini_raw):
        cleaner = DataCleaner()
        out = convert_stock_basic(mini_raw / "raw", cleaner, mini_raw / "normalized")
        df = pd.read_parquet(out)
        assert df["ts_code"].tolist() == sorted(_CODES)

    def test_convert_single_missing_raw_raises(self, mini_raw):
        with pytest.raises(ValueError, match="缺少 raw"):
            convert_calendar(mini_raw / "raw" / "nope", DataCleaner(), mini_raw / "normalized")


class TestConvertDaily:
    def test_matches_direct_pipeline_exactly(self, mini_raw):
        """convert_daily 与 build_clean 逐日段同序直算逐值一致（同一实现 ⇒ 完全相等）。"""
        storage = Storage(root_path=str(mini_raw))
        loader = DataLoader(storage)
        cleaner = DataCleaner()
        for date in _DATES:
            out = convert_daily(date, loader, cleaner, mini_raw / "normalized")
            assert out.name == f"{date}.parquet"  # 契约命名无横线
            converted = pd.read_parquet(out)
            direct = _direct_clean_daily(loader, cleaner, date)
            pd.testing.assert_frame_equal(
                converted.reset_index(drop=True),
                direct.reset_index(drop=True),
                check_dtype=False,
            )

    def test_overwrite_allowed(self, mini_raw):
        """normalized 是缓存：重跑覆盖不报错。"""
        loader = DataLoader(Storage(root_path=str(mini_raw)))
        cleaner = DataCleaner()
        convert_daily(_DATES[0], loader, cleaner, mini_raw / "normalized")
        convert_daily(_DATES[0], loader, cleaner, mini_raw / "normalized")

    def test_missing_daily_raises(self, mini_raw):
        loader = DataLoader(Storage(root_path=str(mini_raw)))
        with pytest.raises(ValueError, match="daily"):
            convert_daily("20240108", loader, DataCleaner(), mini_raw / "normalized")

    def test_missing_adj_factor_raises(self, mini_raw):
        (mini_raw / "raw" / "adj_factor" / "2024-01-03.parquet").unlink()
        loader = DataLoader(Storage(root_path=str(mini_raw)))
        with pytest.raises(ValueError, match="复权因子"):
            convert_daily("20240103", loader, DataCleaner(), mini_raw / "normalized")


class TestReconcile:
    def _build_clean_side(self, mini_raw):
        """用同一直算链路铺 clean 侧分区（YYYY-MM-DD 命名）。"""
        loader = DataLoader(Storage(root_path=str(mini_raw)))
        cleaner = DataCleaner()
        for date in _DATES:
            df = _direct_clean_daily(loader, cleaner, date)
            part = mini_raw / "clean" / "daily"
            part.mkdir(parents=True, exist_ok=True)
            df.to_parquet(part / f"{_dashed(date)}.parquet", index=False)
        return loader, cleaner

    def test_reconcile_pass_on_identical(self, mini_raw):
        loader, cleaner = self._build_clean_side(mini_raw)
        for date in _DATES:
            convert_daily(date, loader, cleaner, mini_raw / "normalized")
        report = reconcile_daily_vs_clean(_DATES, mini_raw)
        assert report["summary"] == {
            "total": 3,
            "passed": 3,
            "failed": 0,
            "worst_max_abs_diff": 0.0,
        }
        day = report["dates"]["20240102"]
        assert day["rows"] == 2
        assert day["mismatched_columns"] == []

    def test_reconcile_detects_tampering(self, mini_raw):
        loader, cleaner = self._build_clean_side(mini_raw)
        for date in _DATES:
            convert_daily(date, loader, cleaner, mini_raw / "normalized")
        tampered = pd.read_parquet(mini_raw / "clean" / "daily" / "2024-01-02.parquet")
        tampered.loc[0, "close"] = float(tampered.loc[0, "close"]) + 1.0
        tampered.to_parquet(mini_raw / "clean" / "daily" / "2024-01-02.parquet", index=False)
        report = reconcile_daily_vs_clean(["20240102"], mini_raw)
        day = report["dates"]["20240102"]
        assert day["status"] == "fail"
        assert "close" in day["mismatched_columns"]
        assert day["rows_over_tolerance"] == 1
        assert day["max_abs_diff"] == pytest.approx(1.0)
        assert report["summary"]["failed"] == 1

    def test_reconcile_missing_partition(self, mini_raw):
        loader = DataLoader(Storage(root_path=str(mini_raw)))
        convert_daily(_DATES[0], loader, DataCleaner(), mini_raw / "normalized")
        report = reconcile_daily_vs_clean([_DATES[0]], mini_raw)
        assert report["dates"][_DATES[0]]["status"] == "fail"
        assert "分区缺失" in report["dates"][_DATES[0]]["note"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
