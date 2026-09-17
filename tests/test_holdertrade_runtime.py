#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""股东增减持因子运行时派生（不写回 cs_train / cs_infer）专项测试。"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.holdertrade import (
    HOLDERTRADE_COLS,
    HOLDERTRADE_FRESHNESS_COL,
    HOLDERTRADE_SCHEMA_VERSION,
    HOLDERTRADE_VERSION_COL,
    build_holdertrade_lookup_by_date,
    derive_holdertrade_columns,
    load_holdertrade_lookup,
)
from src.lazybull.ml.train_core.constants import HOLDERTRADE_FEATURE_COLUMNS

DATES = ["20240111", "20240216"]


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("000001.SZ", "20240110", "G", "IN", 0.5),
            ("000001.SZ", "20240110", "G", "IN", 0.3),
            ("000001.SZ", "20240215", "C", "DE", 0.2),
            ("000002.SZ", "20240115", "P", "DE", 2.0),
        ],
        columns=["ts_code", "ann_date", "holder_type", "in_de", "change_ratio"],
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20240111", "20240111", "20240216", "20240216"],
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "other": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture(scope="module")
def lookup():
    return build_holdertrade_lookup_by_date(_raw(), DATES)


def test_derive_fills_window_values_and_sentinel(lookup):
    frame = _frame()
    derived = derive_holdertrade_columns(frame, lookup, wanted=HOLDERTRADE_FEATURE_COLUMNS)
    assert set(derived) == set(HOLDERTRADE_FEATURE_COLUMNS)
    # 第 1 行：20240111 有 0.8 增持
    assert frame.loc[0, "ht_net_ratio_30d"] == pytest.approx(0.8, rel=1e-6)
    # 第 2 行：20240111 的 000002.SZ 在 T−30 窗口内有 1/15 减持？1/15 晚于 1/11 ⇒ 当日无事件
    assert frame.loc[1, "ht_net_ratio_30d"] == pytest.approx(0.0)
    # 第 3 行：20240216 同 000001.SZ（1/20 前无事件⇒30 日窗口内只有 2/15 减持？）
    assert frame.loc[3, "ht_net_ratio_30d"] == pytest.approx(0.0)
    # 哨兵列全为当前版本（含窗口外股票）
    assert frame[HOLDERTRADE_VERSION_COL].eq(HOLDERTRADE_SCHEMA_VERSION).all()
    # 新鲜度：有事件的股票非空，无事件的为 NaN
    assert not pd.isna(frame.loc[0, HOLDERTRADE_FRESHNESS_COL])
    assert pd.isna(frame.loc[1, HOLDERTRADE_FRESHNESS_COL])


def test_derive_does_not_override_existing_columns(lookup):
    frame = _frame()
    frame["ht_net_ratio_30d"] = 123.0  # 模拟特征分区已带本族列
    derived = derive_holdertrade_columns(frame, lookup, wanted=HOLDERTRADE_FEATURE_COLUMNS)
    assert "ht_net_ratio_30d" not in derived
    assert frame["ht_net_ratio_30d"].eq(123.0).all()


def test_derive_handles_missing_lookup_and_unknown_days():
    frame = _frame()
    derived = derive_holdertrade_columns(frame, None, wanted=HOLDERTRADE_FEATURE_COLUMNS)
    assert set(derived) == set(HOLDERTRADE_FEATURE_COLUMNS)
    for col in HOLDERTRADE_COLS:
        assert frame[col].eq(0.0).all()  # 无查询表 ⇒ 全部按 0（语义 = 无事件）
    assert frame[HOLDERTRADE_FRESHNESS_COL].isna().all()
    assert frame[HOLDERTRADE_VERSION_COL].eq(HOLDERTRADE_SCHEMA_VERSION).all()


def test_derive_wanted_subset_only(lookup):
    frame = _frame()
    derived = derive_holdertrade_columns(
        frame, lookup, wanted=["ht_net_ratio_30d", HOLDERTRADE_VERSION_COL]
    )
    assert set(derived) == {"ht_net_ratio_30d", HOLDERTRADE_VERSION_COL}
    assert "ht_net_ratio_90d" not in frame.columns


def test_derive_requires_key_columns_and_unique_index(lookup):
    with pytest.raises(ValueError, match="trade_date"):
        derive_holdertrade_columns(_frame().drop(columns=["trade_date"]), lookup)
    with pytest.raises(ValueError, match="ts_code"):
        derive_holdertrade_columns(_frame().drop(columns=["ts_code"]), lookup)
    duplicated = pd.concat([_frame(), _frame()], ignore_index=False)
    with pytest.raises(ValueError, match="索引唯一"):
        derive_holdertrade_columns(duplicated, lookup)


def test_derive_is_noop_when_wanted_absent(lookup):
    frame = _frame()
    assert derive_holdertrade_columns(frame, lookup, wanted=["not_a_holdertrade_col"]) == []
    assert not any(col.startswith("ht_") for col in frame.columns)


def test_derive_records_no_future_leakage(lookup):
    """同一交易日的派生值只取决于该日及之前的公告（对照：截断未来事件）。"""
    frame = _frame()
    derive_holdertrade_columns(frame, lookup, wanted=HOLDERTRADE_FEATURE_COLUMNS)
    truncated_lookup = build_holdertrade_lookup_by_date(
        _raw()[_raw()["ann_date"] <= "20240216"], DATES
    )
    other = _frame()
    derive_holdertrade_columns(other, truncated_lookup, wanted=HOLDERTRADE_FEATURE_COLUMNS)
    for col in HOLDERTRADE_FEATURE_COLUMNS:
        assert np.allclose(
            frame[col].to_numpy(dtype="float64"),
            other[col].to_numpy(dtype="float64"),
            equal_nan=True,
        )


def test_load_lookup_from_loader_handles_empty_raw():
    class _Loader:
        def __init__(self, raw):
            self._raw = raw

        def load_stk_holdertrade(self):
            return self._raw

    assert load_holdertrade_lookup(_Loader(None), DATES) == {}
    assert load_holdertrade_lookup(_Loader(pd.DataFrame()), DATES) == {}
    loaded = load_holdertrade_lookup(_Loader(_raw()), DATES)
    assert set(loaded) == {"20240111", "20240216"}
