#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""股票回购因子（factors/repurchase.py）与 handler 的专项测试。

事件样例（`amount` 为累计口径 ⇒ 强度列取**增量**）：
- 000001.SZ 20240110：实施 1.0e8（首执行行 ⇒ 增量 1.0e8）
- 000001.SZ 20240220：实施 1.6e8（累计 ⇒ 增量 0.6e8）
- 000002.SZ 20240115：完成，amount 缺失（**不兜底** ⇒ 增量 0，不更新基线）
- 000003.SZ 20240120：预案×2（**计划金额不进强度列**，只贡献 high_limit）
- 000004.SZ ann_date 缺失：必须剔除

窗口为**自然日**开区间 (T−W, T]；PIT 锚点只有 ann_date。
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.repurchase import (
    REPURCHASE_COLS,
    REPURCHASE_FRESHNESS_COL,
    REPURCHASE_SCHEMA_VERSION,
    REPURCHASE_VERSION_COL,
    aggregate_repurchase_events,
    available_repurchase_columns,
    build_repurchase_feature_frame,
    build_repurchase_lookup_by_date,
    derive_repurchase_columns,
)
from src.lazybull.features.factor_handlers import RepurchaseFactorHandler
from src.lazybull.ml.train_core.constants import REPURCHASE_FEATURE_COLUMNS

TEST_DATES = [
    "20240109",  # 任何事件之前：不得出现（PIT 无前视）
    "20240110",  # 公告当日：可见
    "20240111",
    "20240408",  # 20240110 已 89 天：90 日窗口内
    "20240409",  # 20240110 恰好 90 天：开区间 ⇒ 排除
    "20240410",
    "20240520",  # 20240220 恰好 90 天：90 日窗口不可见，180 日窗口仍在
    "20240707",  # 20240110 已 179 天：180 日窗口内
    "20240708",  # 20240110 恰好 180 天：开区间 ⇒ 排除
    "20240810",
    "20240910",  # 20240220 已 203 天：全部窗口外
]


def _raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("000001.SZ", "20240110", "实施", 1.0e8, 1.0e6, 10.0),
            ("000001.SZ", "20240220", "实施", 1.6e8, 1.0e6, 12.0),
            ("000002.SZ", "20240115", "完成", np.nan, 5.0e5, None),
            ("000003.SZ", "20240120", "预案", 1.0e8, None, 8.0),
            ("000003.SZ", "20240120", "预案", 0.5e8, None, 7.0),
            ("000004.SZ", None, "预案", 1.0e7, None, None),  # 非法日期：必须剔除
        ],
        columns=["ts_code", "ann_date", "proc", "amount", "vol", "high_limit"],
    )


@pytest.fixture(scope="module")
def lookup():
    return build_repurchase_lookup_by_date(_raw_rows(), TEST_DATES)


def _row(lookup, trade_date: str, ts_code: str) -> pd.Series:
    frame = lookup.get(trade_date)
    assert frame is not None, f"{trade_date} 无任何活跃股票（预期至少一只）"
    indexed = frame.set_index("ts_code")
    assert ts_code in indexed.index, f"{trade_date} 缺少 {ts_code}"
    return indexed.loc[ts_code]


# ---------------------------------------------------------------- 事件聚合


def test_aggregate_sums_amount_and_takes_max_limit_same_day():
    """同日多行：计划金额不进强度列、exec 取 max、limit 取非空最大。"""
    events = aggregate_repurchase_events(_raw_rows())
    row = events[(events["ts_code"] == "000001.SZ") & (events["ann_date"] == "20240110")].iloc[0]
    assert row["amt"] == pytest.approx(1.0e8)  # 首执行行 ⇒ 增量 = 本行金额
    assert row["exec_flag"] == 1.0
    assert row["limit_price"] == pytest.approx(10.0)
    # 000004.SZ 非法日期被剔除 ⇒ 4 个事件行（000003 同日两行聚合成 1 行）
    assert len(events) == 4
    assert "000004.SZ" not in set(events["ts_code"])
    # 000003.SZ 预案×2 同日：金额不进强度、limit 取非空最大
    plan = events[(events["ts_code"] == "000003.SZ")].iloc[0]
    assert plan["amt"] == 0.0 and plan["exec_flag"] == 0.0
    assert plan["limit_price"] == pytest.approx(8.0)


def test_aggregate_executed_amount_uses_cumulative_increment():
    """执行类 `amount` 是累计口径 ⇒ 必须取增量，不得直接求和。"""
    rows = pd.DataFrame(
        [
            ("000020.SZ", "20240110", "实施", 1.0e8),
            ("000020.SZ", "20240210", "实施", 1.6e8),
            ("000020.SZ", "20240310", "完成", 2.0e8),
        ],
        columns=["ts_code", "ann_date", "proc", "amount"],
    )
    events = aggregate_repurchase_events(rows)
    amt = events.sort_values("ann_ord")["amt"].tolist()
    assert amt == [pytest.approx(1.0e8), pytest.approx(0.6e8), pytest.approx(0.4e8)]
    assert events["exec_flag"].tolist() == [1.0, 1.0, 1.0]


def test_aggregate_executed_amount_resets_on_new_plan():
    """累计值回落 ⇒ 视为新计划，增量 = 本行金额（不得出现负增量）。"""
    rows = pd.DataFrame(
        [
            ("000021.SZ", "20240110", "完成", 2.0e8),
            ("000021.SZ", "20240610", "实施", 0.3e8),
            ("000021.SZ", "20240710", "实施", 0.5e8),
        ],
        columns=["ts_code", "ann_date", "proc", "amount"],
    )
    events = aggregate_repurchase_events(rows).sort_values("ann_ord")
    assert events["amt"].tolist() == [
        pytest.approx(2.0e8),
        pytest.approx(0.3e8),
        pytest.approx(0.2e8),
    ]
    assert (events["amt"] >= 0).all()


def test_aggregate_orders_same_day_rows_by_end_date():
    """同日多行必须按 `end_date`（进度报告期）次级排序，否则"回落"会被误判为新计划。

    复刻真实案例 300138.SZ 20240604：同日先报 20240603 期（3.65867e8）、后报 20240531 期
    （3.461217e8），文件行序与报告期相反 ⇒ 不排序会虚增一笔 3.46e8 的"新计划"增量。
    """
    rows = pd.DataFrame(
        [
            ("000030.SZ", "20240509", "20240507", "完成", 3.066505e8),
            ("000030.SZ", "20240604", "20240603", "完成", 3.658670e8),
            ("000030.SZ", "20240604", "20240531", "完成", 3.461217e8),
        ],
        columns=["ts_code", "ann_date", "end_date", "proc", "amount"],
    )
    events = aggregate_repurchase_events(rows).sort_values("ann_ord")
    assert events["amt"].tolist() == [
        pytest.approx(3.066505e8),  # 首行 ⇒ 增量 = 本行金额
        pytest.approx(0.592165e8),  # 同日两期：0.394712e8 + 0.197453e8
    ]
    # 若不按 end_date 排序，同日事件会变成 0.592e8 + 3.461e8（误判为新计划）


def test_aggregate_plan_stage_amount_not_in_intensity():
    """预案/股东大会通过的 `amount` 是计划金额且跨期重复 ⇒ 不进强度列。"""
    rows = pd.DataFrame(
        [
            ("000022.SZ", "20240110", "预案", 2.0e8),
            ("000022.SZ", "20240210", "股东大会通过", 2.0e8),
            ("000022.SZ", "20240310", "实施", 0.5e8),
        ],
        columns=["ts_code", "ann_date", "proc", "amount"],
    )
    events = aggregate_repurchase_events(rows).sort_values("ann_ord")
    assert events["amt"].tolist() == [0.0, 0.0, pytest.approx(0.5e8)]
    assert events["exec_flag"].tolist() == [0.0, 0.0, 1.0]


def test_aggregate_missing_amount_is_not_backfilled():
    """`amount` 缺失不兜底：增量记 0（不用 vol × high_limit 近似），且不更新累计基线。"""
    rows = pd.DataFrame(
        [
            ("000010.SZ", "20240110", "实施", 1.0e8, 1.0e6, 20.0),
            ("000010.SZ", "20240210", "实施", np.nan, 1.0e6, 20.0),
            ("000010.SZ", "20240310", "实施", 1.4e8, 1.0e6, 20.0),
        ],
        columns=["ts_code", "ann_date", "proc", "amount", "vol", "high_limit"],
    )
    events = aggregate_repurchase_events(rows).sort_values("ann_ord")
    # 1.0e8（首行）→ NaN 行 0（基线保持 1.0e8）→ 1.4e8 − 1.0e8 = 0.4e8
    assert events["amt"].tolist() == [pytest.approx(1.0e8), 0.0, pytest.approx(0.4e8)]


def test_aggregate_all_nan_amount_raises():
    rows = pd.DataFrame(
        [("000010.SZ", "20240110", "实施", np.nan)],
        columns=["ts_code", "ann_date", "proc", "amount"],
    )
    with pytest.raises(ValueError, match="amount 整列为空"):
        aggregate_repurchase_events(rows)


def test_aggregate_missing_required_column_raises():
    rows = pd.DataFrame([("000010.SZ", "20240110")], columns=["ts_code", "ann_date"])
    with pytest.raises(ValueError, match="缺少必要列: proc"):
        aggregate_repurchase_events(rows)


def test_aggregate_unknown_proc_ignored_for_stage_semantics():
    rows = pd.DataFrame(
        [("000011.SZ", "20240110", "未知阶段", 5.0e7)],
        columns=["ts_code", "ann_date", "proc", "amount"],
    )
    events = aggregate_repurchase_events(rows)
    assert events.iloc[0]["amt"] == 0.0  # 未登记取值：不计金额、不算已执行
    assert events.iloc[0]["exec_flag"] == 0.0


# ---------------------------------------------------------------- 查询表 PIT


def test_lookup_has_no_entry_before_first_announcement(lookup):
    assert "20240109" not in lookup


def test_lookup_window_is_open_left_interval(lookup):
    # 20240408：距 20240110 已 89 天 ⇒ 90 日窗口内
    assert _row(lookup, "20240408", "000001.SZ")["rp_amt_90d"] == pytest.approx(1.6e8)
    # 20240409：恰好 90 天 ⇒ 开区间排除（只剩 20240220 的增量 0.6e8）
    row = _row(lookup, "20240409", "000001.SZ")
    assert row["rp_amt_90d"] == pytest.approx(0.6e8)
    assert row["rp_amt_180d"] == pytest.approx(1.6e8)
    # 20240707：距 20240110 已 179 天 ⇒ 180 日窗口内
    assert _row(lookup, "20240707", "000001.SZ")["rp_amt_180d"] == pytest.approx(1.6e8)
    # 20240708：恰好 180 天 ⇒ 开区间排除
    assert _row(lookup, "20240708", "000001.SZ")["rp_amt_180d"] == pytest.approx(0.6e8)


def test_lookup_exec_flag_decays_faster_than_intensity(lookup):
    # 20240520：距 20240220 恰好 90 天 ⇒ 90 日窗口全空，但 180 日强度仍在
    row = _row(lookup, "20240520", "000001.SZ")
    assert row["rp_amt_90d"] == pytest.approx(0.0)
    assert row["rp_exec_90d"] == pytest.approx(0.0)
    assert row["rp_amt_180d"] == pytest.approx(1.6e8)


def test_lookup_windows_and_freshness(lookup):
    row = _row(lookup, "20240111", "000001.SZ")
    assert row["rp_amt_90d"] == pytest.approx(1.0e8)
    assert row["rp_amt_180d"] == pytest.approx(1.0e8)
    assert row["rp_exec_90d"] == pytest.approx(1.0)
    assert row["rp_freshness_days"] == pytest.approx(1.0)

    row = _row(lookup, "20240410", "000001.SZ")
    assert row["rp_amt_90d"] == pytest.approx(0.6e8)  # 91 天前的增量已出 90 日窗
    assert row["rp_amt_180d"] == pytest.approx(1.6e8)
    assert row["rp_exec_90d"] == pytest.approx(1.0)
    assert row["rp_limit_price"] == pytest.approx(12.0)  # 窗口内最新带上限的公告


def test_lookup_zero_intensity_for_plan_only_and_missing_amount(lookup):
    # 000003.SZ 仅有预案（计划金额不进强度），但价格上限可见
    row = _row(lookup, "20240520", "000003.SZ")
    assert row["rp_amt_90d"] == 0.0 and row["rp_amt_180d"] == 0.0
    assert row["rp_exec_90d"] == 0.0
    assert row["rp_limit_price"] == pytest.approx(8.0)
    # 000002.SZ 完成行缺 amount ⇒ 增量为 0 但已执行标志为 1（20240410 距公告 86 天，90 日窗口内）
    row = _row(lookup, "20240410", "000002.SZ")
    assert row["rp_amt_90d"] == 0.0 and row["rp_amt_180d"] == 0.0
    assert row["rp_exec_90d"] == pytest.approx(1.0)
    assert np.isnan(row["rp_limit_price"])


def test_lookup_stock_leaves_window_after_180_days(lookup):
    assert "20240910" not in lookup  # 000001 的 20240220 已 203 天


def test_lookup_does_not_expose_sentinel_less_rows(lookup):
    for frame in lookup.values():
        assert REPURCHASE_VERSION_COL in frame.columns
        assert (frame[REPURCHASE_VERSION_COL] == REPURCHASE_SCHEMA_VERSION).all()


# ---------------------------------------------------------------- 特征折算


def _features_frame(ts_codes) -> pd.DataFrame:
    """当日截面：circ_mv 万元、amount 千元、vol 手（VWAP = amount × 10 ÷ vol = 10 元）。"""
    return pd.DataFrame(
        {
            "trade_date": ["20240410"] * len(ts_codes),
            "ts_code": list(ts_codes),
            "circ_mv": [500000.0] * len(ts_codes),  # 万元 → 5e9 元
            "amount": [50000.0] * len(ts_codes),  # 千元
            "vol": [50000.0] * len(ts_codes),  # 手 → VWAP 10 元
        }
    )


def test_build_feature_frame_converts_units_and_headroom(lookup):
    features = _features_frame(["000001.SZ", "000002.SZ"])
    day = lookup["20240410"]
    merged = features.merge(day, on="ts_code", how="left")
    merged.index = features.index
    frame = build_repurchase_feature_frame(features, merged)

    row = frame.loc[0]
    assert row["rp_amount_to_mv_90d"] == pytest.approx(0.6e8 / 5.0e9, rel=1e-6)
    assert row["rp_amount_to_mv_180d"] == pytest.approx(1.6e8 / 5.0e9, rel=1e-6)
    assert row["rp_exec_flag_90d"] == pytest.approx(1.0)
    assert row["rp_price_headroom"] == pytest.approx(12.0 / 10.0 - 1.0, rel=1e-6)
    assert frame.loc[1, "rp_amount_to_mv_90d"] == 0.0  # 000002 当日不在查询表内


def test_build_feature_frame_zero_fills_when_no_event():
    features = _features_frame(["000002.SZ"])
    frame = build_repurchase_feature_frame(features, None)
    for col in REPURCHASE_COLS:
        assert frame.loc[0, col] == 0.0
    assert np.isnan(frame.loc[0, REPURCHASE_FRESHNESS_COL])


def test_build_feature_frame_zero_fills_on_missing_price(lookup):
    """金额为 0（缺失不兜底）且无 VWAP ⇒ 值列仍为 0，不得变 NaN。"""
    features = _features_frame(["000002.SZ"])
    features.loc[0, "amount"] = 0.0
    features.loc[0, "vol"] = 0.0
    day = lookup["20240410"]
    merged = features.merge(day, on="ts_code", how="left")
    merged.index = features.index
    frame = build_repurchase_feature_frame(features, merged)
    assert frame.loc[0, "rp_price_headroom"] == 0.0
    assert frame.loc[0, "rp_amount_to_mv_90d"] == 0.0


def test_build_feature_frame_requires_market_value_and_price_columns():
    features = _features_frame(["000001.SZ"])
    with pytest.raises(ValueError, match="circ_mv"):
        build_repurchase_feature_frame(features.drop(columns=["circ_mv"]), None)
    with pytest.raises(ValueError, match="amount"):
        build_repurchase_feature_frame(features.drop(columns=["amount"]), None)


# ---------------------------------------------------------------- handler


def test_handler_returns_empty_when_family_disabled():
    features = _features_frame(["000001.SZ"])
    assert RepurchaseFactorHandler().apply(features, None, "20240410", pd.DataFrame()) == {}


def test_handler_zero_fills_on_empty_day_data(lookup):
    features = _features_frame(["000002.SZ"])
    result = RepurchaseFactorHandler().apply(features, pd.DataFrame(), "20240910", pd.DataFrame())
    for col in REPURCHASE_COLS:
        assert (result[col] == 0.0).all()
    assert result[REPURCHASE_FRESHNESS_COL].isna().all()
    assert (result[REPURCHASE_VERSION_COL] == REPURCHASE_SCHEMA_VERSION).all()


def test_handler_writes_all_market_coverage(lookup):
    """窗口外股票（不在查询表内）必须显式填 0，避免被 0.6 缺失率门禁删除。"""
    features = _features_frame(["000001.SZ", "999999.SZ"])
    result = RepurchaseFactorHandler().apply(
        features, lookup["20240410"], "20240410", pd.DataFrame()
    )
    for col in REPURCHASE_COLS:
        assert result[col].notna().all()
    assert result["rp_amount_to_mv_180d"].iloc[1] == 0.0


# ---------------------------------------------------------------- 运行时派生


def test_derive_adds_columns_and_sentinel(lookup):
    features = _features_frame(["000001.SZ", "000002.SZ"])
    added = derive_repurchase_columns(features, lookup)
    assert sorted(added) == sorted(available_repurchase_columns())
    assert (features[REPURCHASE_VERSION_COL] == REPURCHASE_SCHEMA_VERSION).all()
    assert features["rp_exec_flag_90d"].notna().all()


def test_derive_does_not_overwrite_existing_columns(lookup):
    features = _features_frame(["000001.SZ"])
    features["rp_exec_flag_90d"] = -7.0
    added = derive_repurchase_columns(features, lookup)
    assert "rp_exec_flag_90d" not in added
    assert features["rp_exec_flag_90d"].iloc[0] == -7.0


def test_derive_respects_wanted_columns(lookup):
    features = _features_frame(["000001.SZ"])
    added = derive_repurchase_columns(features, lookup, wanted=["rp_amount_to_mv_90d"])
    assert added == ["rp_amount_to_mv_90d"]


def test_derive_without_lookup_fills_zero(lookup):
    features = _features_frame(["000001.SZ"])
    derive_repurchase_columns(features, None)
    assert features["rp_amount_to_mv_90d"].iloc[0] == 0.0


def test_derive_requires_key_columns(lookup):
    features = _features_frame(["000001.SZ"]).drop(columns=["trade_date"])
    with pytest.raises(ValueError, match="trade_date"):
        derive_repurchase_columns(features, lookup)


def test_derive_requires_unique_index(lookup):
    features = _features_frame(["000001.SZ", "000001.SZ"])
    features.index = [0, 0]
    with pytest.raises(ValueError, match="索引唯一"):
        derive_repurchase_columns(features, lookup)


def test_feature_columns_consistent_with_factor_columns():
    """列集双处单一来源：factors 与 train_core.constants 必须一致（有测试锁定）。"""
    assert REPURCHASE_FEATURE_COLUMNS == available_repurchase_columns()
