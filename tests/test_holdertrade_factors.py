#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""股东增减持因子（factors/holdertrade.py）与 handler 的专项测试。

事件样例（000001.SZ）：
- 20240110：高管（G）增持 0.5% + 0.3%（同日多股东求和）
- 20240120：公司股东（C）增持 1.0%
- 20240215：高管（G）减持 0.2%
窗口为**自然日**开区间 (T−W, T]。
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.holdertrade import (
    HOLDERTRADE_COLS,
    HOLDERTRADE_FRESHNESS_COL,
    HOLDERTRADE_SCHEMA_VERSION,
    HOLDERTRADE_VERSION_COL,
    aggregate_holdertrade_events,
    available_holdertrade_columns,
    build_holdertrade_lookup_by_date,
)
from src.lazybull.features.factor_handlers import HoldertradeFactorHandler
from src.lazybull.ml.train_core.constants import HOLDERTRADE_FEATURE_COLUMNS

TEST_DATES = ["20240111", "20240116", "20240210", "20240211", "20240216", "20240331", "20240520"]


def _raw_rows() -> pd.DataFrame:
    """逐股东明细样例（含非法 ann_date 与同日多笔）。"""
    return pd.DataFrame(
        [
            ("000001.SZ", "20240110", "G", "IN", 0.5),
            ("000001.SZ", "20240110", "G", "IN", 0.3),
            ("000001.SZ", "20240120", "C", "IN", 1.0),
            ("000001.SZ", "20240215", "G", "DE", 0.2),
            ("000002.SZ", "20240115", "P", "DE", 2.0),
            ("000002.SZ", None, "P", "DE", 9.9),  # 非法日期：必须剔除
        ],
        columns=["ts_code", "ann_date", "holder_type", "in_de", "change_ratio"],
    )


@pytest.fixture(scope="module")
def lookup():
    return build_holdertrade_lookup_by_date(_raw_rows(), TEST_DATES)


def _row(lookup, trade_date: str, ts_code: str) -> pd.Series:
    frame = lookup.get(trade_date)
    assert frame is not None, f"{trade_date} 无任何活跃股票（预期至少一只）"
    indexed = frame.set_index("ts_code")
    assert ts_code in indexed.index
    return indexed.loc[ts_code]


# ---------------------------------------------------------------- 事件聚合


def test_aggregate_sums_same_day_ratio_and_sets_presence_flags():
    events = aggregate_holdertrade_events(_raw_rows())
    row = events[(events["ts_code"] == "000001.SZ") & (events["ann_date"] == "20240110")].iloc[0]
    assert row["ratio_net"] == pytest.approx(0.8)  # 同日两笔求和
    assert row["ratio_net_exec"] == pytest.approx(0.8)  # holder_type=G
    assert row["ratio_net_other"] == pytest.approx(0.0)
    assert row["has_in"] == 1.0 and row["has_de"] == 0.0
    # 事件单位是 (ts_code, ann_date)：000001.SZ 有 3 个事件日，000002.SZ 1 个
    assert len(events) == 4


def test_aggregate_nets_in_and_de_on_same_day():
    mixed = pd.DataFrame(
        [
            ("000003.SZ", "20240110", "G", "IN", 0.6),
            ("000003.SZ", "20240110", "C", "DE", 0.2),
        ],
        columns=["ts_code", "ann_date", "holder_type", "in_de", "change_ratio"],
    )
    row = aggregate_holdertrade_events(mixed).iloc[0]
    assert row["ratio_net"] == pytest.approx(0.4)
    assert row["has_in"] == 1.0 and row["has_de"] == 1.0


def test_aggregate_drops_invalid_ann_date_and_requires_columns():
    events = aggregate_holdertrade_events(_raw_rows())
    assert set(events["ts_code"]) == {"000001.SZ", "000002.SZ"}
    assert not events["ann_date"].isna().any()

    with pytest.raises(ValueError, match="缺少必要列"):
        aggregate_holdertrade_events(pd.DataFrame({"ts_code": ["x"]}))


def test_aggregate_merges_duplicated_rows_into_same_day():
    raw = pd.concat([_raw_rows(), _raw_rows().iloc[[0]]], ignore_index=True)
    events = aggregate_holdertrade_events(raw)
    # 去重责任在 raw 层，这里只要求"不静默"：重复行会被并入同一天（0.5 变 1.0）
    row = events[(events["ts_code"] == "000001.SZ") & (events["ann_date"] == "20240110")].iloc[0]
    assert row["ratio_net"] == pytest.approx(1.3)


# ---------------------------------------------------------------- 窗口与 PIT


def test_short_window_boundary_is_natural_days(lookup):
    # 20240210：距 1/10 已 31 个自然日 ⇒ 出 30 日窗口；1/20 仍在窗口内
    row = _row(lookup, "20240210", "000001.SZ")
    assert row["ht_net_ratio_30d"] == pytest.approx(1.0, rel=1e-5)
    assert row["ht_net_ratio_90d"] == pytest.approx(1.8, rel=1e-5)  # 1.0 + 0.8
    assert row["ht_buy_count_30d"] == 1
    assert row["ht_net_count_90d"] == 2  # 两个增持披露日，无减持
    assert row[HOLDERTRADE_FRESHNESS_COL] == pytest.approx(21)  # 1/20 → 2/10


def test_exec_split_counts_and_accel(lookup):
    # 20240216：30 日窗口 (1/17, 2/16] ⇒ 1/20 增持 1.0 与 2/15 减持 0.2
    row = _row(lookup, "20240216", "000001.SZ")
    assert row["ht_net_ratio_30d"] == pytest.approx(0.8, rel=1e-5)
    assert row["ht_net_ratio_30d_exec"] == pytest.approx(-0.2, rel=1e-5)  # 高管净额
    assert row["ht_net_ratio_30d_other"] == pytest.approx(1.0, rel=1e-5)  # 非高管净额
    assert row["ht_buy_count_30d"] == 1 and row["ht_sell_count_30d"] == 1
    assert row["ht_net_count_90d"] == 1  # 90 日内 2 个增持日 − 1 个减持日
    assert row["ht_net_ratio_accel"] == pytest.approx(0.8 - 1.6 / 3, rel=1e-5)
    assert row[HOLDERTRADE_FRESHNESS_COL] == pytest.approx(1)
    assert row[HOLDERTRADE_VERSION_COL] == HOLDERTRADE_SCHEMA_VERSION


def test_windows_decay_to_zero_and_row_disappears_after_90d(lookup):
    # 20240331：全部事件已出 30 日窗口 ⇒ 0；但 2/15 仍在 90 日内
    row = _row(lookup, "20240331", "000001.SZ")
    assert row["ht_net_ratio_30d"] == pytest.approx(0.0)
    assert row["ht_net_ratio_30d_exec"] == pytest.approx(0.0)
    assert row["ht_buy_count_30d"] == 0 and row["ht_sell_count_30d"] == 0
    assert row["ht_net_ratio_90d"] == pytest.approx(1.6, rel=1e-5)
    assert row["ht_net_ratio_accel"] == pytest.approx(-1.6 / 3, rel=1e-5)
    assert row[HOLDERTRADE_FRESHNESS_COL] == pytest.approx(45)  # 2/15 → 3/31

    # 20240520：T−90 = 2/20 > 2/15 ⇒ 无活跃事件，该股票不再出现在查询表（消费侧填 0）
    frame = lookup.get("20240520")
    if frame is not None:
        assert "000001.SZ" not in set(frame["ts_code"])
        assert "000002.SZ" not in set(frame["ts_code"])


def test_no_future_leakage_when_later_events_removed():
    raw = _raw_rows()
    for target in TEST_DATES:
        full = build_holdertrade_lookup_by_date(raw, TEST_DATES).get(target)
        truncated = build_holdertrade_lookup_by_date(
            raw[raw["ann_date"].fillna("99999999") <= target], TEST_DATES
        ).get(target)
        if full is None:
            assert truncated is None
            continue
        a = full.set_index("ts_code").sort_index()
        b = truncated.set_index("ts_code").sort_index() if truncated is not None else None
        assert b is not None and list(a.index) == list(b.index)
        for col in HOLDERTRADE_COLS + [HOLDERTRADE_FRESHNESS_COL]:
            assert np.allclose(a[col].to_numpy(), b[col].to_numpy(), equal_nan=True)


def test_empty_inputs_return_empty_lookup():
    empty = pd.DataFrame(columns=["ts_code", "ann_date", "in_de", "holder_type", "change_ratio"])
    assert build_holdertrade_lookup_by_date(empty, TEST_DATES) == {}
    assert build_holdertrade_lookup_by_date(_raw_rows(), []) == {}


def test_available_columns_match_train_constants():
    assert set(available_holdertrade_columns()) == set(HOLDERTRADE_FEATURE_COLUMNS)
    assert set(HOLDERTRADE_COLS).issubset(set(HOLDERTRADE_FEATURE_COLUMNS))


# ---------------------------------------------------------------- handler


def _features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"ts_code": ["000001.SZ", "000002.SZ", "999999.SZ"], "close": [1.0, 2.0, 3.0]}
    )


def test_handler_returns_empty_when_family_disabled():
    handler = HoldertradeFactorHandler()
    assert handler.apply(_features_frame(), None, "20240216", pd.DataFrame()) == {}


def test_handler_zero_fills_missing_and_writes_sentinel():
    handler = HoldertradeFactorHandler()
    features = _features_frame()
    data = build_holdertrade_lookup_by_date(_raw_rows(), TEST_DATES)["20240216"]
    result = handler.apply(features, data, "20240216", pd.DataFrame())
    assert set(result) == set(HOLDERTRADE_FEATURE_COLUMNS)
    # 有事件的股票：真实值；无事件的股票：0（语义 = 窗口内无增减持）
    assert result["ht_net_ratio_30d"].iloc[0] == pytest.approx(0.8, rel=1e-5)
    assert result["ht_net_ratio_30d"].iloc[1] == pytest.approx(0.0)
    assert result["ht_net_ratio_30d"].iloc[2] == pytest.approx(0.0)
    assert result["ht_sell_count_30d"].iloc[0] == pytest.approx(1.0)
    assert result["ht_sell_count_30d"].iloc[1] == pytest.approx(0.0)
    assert pd.isna(result[HOLDERTRADE_FRESHNESS_COL].iloc[2])  # 无事件 ⇒ NaN
    assert result[HOLDERTRADE_VERSION_COL].eq(HOLDERTRADE_SCHEMA_VERSION).all()
    assert result["ht_net_ratio_30d"].dtype == np.float32


def test_handler_empty_day_keeps_schema_consistent():
    handler = HoldertradeFactorHandler()
    features = _features_frame()
    result = handler.apply(features, pd.DataFrame(), "20240216", pd.DataFrame())
    assert set(result) == set(HOLDERTRADE_FEATURE_COLUMNS)
    for col in HOLDERTRADE_COLS:
        assert result[col].eq(0.0).all()
    assert result[HOLDERTRADE_FRESHNESS_COL].isna().all()
    assert result[HOLDERTRADE_VERSION_COL].eq(HOLDERTRADE_SCHEMA_VERSION).all()
