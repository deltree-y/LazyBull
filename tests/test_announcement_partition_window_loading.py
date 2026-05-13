#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公告型季度分区窗口读取回归测试。"""

import tempfile

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
