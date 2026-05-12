#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""新增因子接线回归测试。"""

import pandas as pd

from src.lazybull.data.loader import DataLoader
from src.lazybull.features.ensure import _REQUIRED_FACTOR_COLS


class _StubStorage:
    """最小存储桩：仅覆盖 DataLoader 测试需要的方法。"""

    def __init__(self, payload):
        self._payload = payload

    def load_raw(self, name):
        return self._payload.get(name)


def test_required_factor_cols_include_new_cashflow_and_consensus_revision_fields():
    assert "cashflow_freshness_days" in _REQUIRED_FACTOR_COLS
    assert "cons_revision_freshness_days" in _REQUIRED_FACTOR_COLS
    assert "grossprofit_margin" in _REQUIRED_FACTOR_COLS


def test_loader_cashflow_normalizes_date_columns():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "ann_date": ["2024-05-01"],
            "end_date": ["2024-03-31"],
            "f_ann_date": ["2024-05-02"],
        }
    )
    loader = DataLoader(storage=_StubStorage({"cashflow": raw}))

    df = loader.load_cashflow()

    assert df is not None
    assert df.loc[0, "ann_date"] == "20240501"
    assert df.loc[0, "end_date"] == "20240331"
    assert df.loc[0, "f_ann_date"] == "20240502"
