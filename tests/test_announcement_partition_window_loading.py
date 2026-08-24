#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公告型季度分区窗口读取回归测试。"""

import tempfile
import warnings

import pandas as pd

from src.lazybull.data import DataLoader, Storage


def test_loader_load_fina_indicator_reads_only_needed_quarter_partitions():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        storage.save_raw(
            pd.DataFrame(
                {
                    "ts_code": ["LEGACY.SZ"],
                    "ann_date": ["20210101"],
                    "end_date": ["20201231"],
                }
            ),
            "fina_indicator",
            is_force=True,
        )
        storage.save_raw_by_date(
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "ann_date": ["20240430"],
                    "end_date": ["20240331"],
                }
            ),
            "fina_indicator",
            "20240331",
        )
        storage.save_raw_by_date(
            pd.DataFrame(
                {
                    "ts_code": ["000002.SZ"],
                    "ann_date": ["20241030"],
                    "end_date": ["20240930"],
                }
            ),
            "fina_indicator",
            "20240930",
        )
        storage.save_raw_by_date(
            pd.DataFrame(
                {
                    "ts_code": ["000003.SZ"],
                    "ann_date": ["20250430"],
                    "end_date": ["20250331"],
                }
            ),
            "fina_indicator",
            "20250331",
        )

        loader = DataLoader(storage=storage)
        df = loader.load_fina_indicator(start_date="20250115", end_date="20250131")

        assert df is not None
        assert sorted(df["ts_code"].tolist()) == ["000001.SZ", "000002.SZ"]
        assert sorted(df["end_date"].tolist()) == ["20240331", "20240930"]


def test_loader_load_cashflow_reads_only_needed_quarter_partitions():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        storage.save_raw_by_date(
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "ann_date": ["20240430"],
                    "end_date": ["20240331"],
                    "f_ann_date": ["20240501"],
                }
            ),
            "cashflow",
            "20240331",
        )
        storage.save_raw_by_date(
            pd.DataFrame(
                {
                    "ts_code": ["000002.SZ"],
                    "ann_date": ["20241030"],
                    "end_date": ["20240930"],
                    "f_ann_date": ["20241031"],
                }
            ),
            "cashflow",
            "20240930",
        )
        storage.save_raw_by_date(
            pd.DataFrame(
                {
                    "ts_code": ["000003.SZ"],
                    "ann_date": ["20250430"],
                    "end_date": ["20250331"],
                    "f_ann_date": ["20250501"],
                }
            ),
            "cashflow",
            "20250331",
        )

        loader = DataLoader(storage=storage)
        df = loader.load_cashflow(start_date="20250115", end_date="20250131")

        assert df is not None
        assert sorted(df["ts_code"].tolist()) == ["000001.SZ", "000002.SZ"]
        assert sorted(df["ann_date"].tolist()) == ["20240430", "20241030"]
        assert sorted(df["f_ann_date"].tolist()) == ["20240501", "20241031"]


def test_loader_cashflow_suppresses_all_na_concat_warning(monkeypatch):
    """窗口内外现金流分区合并不泄露告警，且全 NA 列原样保留。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        rows_by_period = {
            "20230930": ("000001.SZ", "20231030"),
            "20231231": ("000002.SZ", "20240430"),
            "20240331": ("000003.SZ", "20240501"),
        }
        for period, (ts_code, ann_date) in rows_by_period.items():
            storage.save_raw_by_date(
                pd.DataFrame(
                    {
                        "ts_code": [ts_code],
                        "ann_date": [ann_date],
                        "end_date": [period],
                        "optional_metric": [None],
                    }
                ),
                "cashflow",
                period,
            )

        original_concat = pd.concat

        def concat_with_warning(*args, **kwargs):
            warnings.warn(
                "The behavior of DataFrame concatenation with empty or all-NA entries "
                "is deprecated.",
                FutureWarning,
                stacklevel=2,
            )
            return original_concat(*args, **kwargs)

        monkeypatch.setattr(pd, "concat", concat_with_warning)
        loader = DataLoader(storage=storage)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = loader.load_cashflow(start_date="20250115", end_date="20250131")

        assert result is not None and len(result) == 3
        assert sorted(result["ts_code"].tolist()) == [
            "000001.SZ",
            "000002.SZ",
            "000003.SZ",
        ]
        assert "optional_metric" in result.columns
        assert result["optional_metric"].isna().all()
        assert not any(
            issubclass(item.category, FutureWarning)
            and "DataFrame concatenation with empty or all-NA entries" in str(item.message)
            for item in caught
        )
