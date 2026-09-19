#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""十大流通股东因子（factors/top10_floatholders.py）与 handler 的专项测试。

口径要点（见 `docs/top10_floatholders_pit_audit.md` §6 口径定稿）：
- 同 `(ts_code, end_date, ann_date, holder_name)` 多行 ⇒ 取 `hold_amount` 最大行；
- 组内按 `hold_amount` 降序取前 10 行；
- 长线机构按 `holder_type` **整串相等**白名单（`金融机构—证券公司`/`公益基金`/`保险公司` 不得误纳）；
- `hold_float_ratio` 缺失行按 0 贡献跳过（不整组置 NaN）；`tfh_top1_ratio` 取组内 max；
- PIT：T 日可见 = `ann_date ≤ T` 的最新一次披露；同日披露多个报告期取 `end_ord` 更大者；
- 未披露股票 ⇒ 值列 NaN（**禁止 0 填充**），哨兵列恒写当前版本。
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.top10_floatholders import (
    TOP10FH_ANN_ORD_COL,
    TOP10FH_COLS,
    TOP10FH_CONCENTRATION_CHG_COL,
    TOP10FH_END_ORD_COL,
    TOP10FH_FRESHNESS_COL,
    TOP10FH_INST_COUNT_COL,
    TOP10FH_INST_RATIO_COL,
    TOP10FH_SCHEMA_VERSION,
    TOP10FH_SOCIAL_FLAG_COL,
    TOP10FH_TOP1_RATIO_COL,
    TOP10FH_TOP10_RATIO_COL,
    TOP10FH_VERSION_COL,
    aggregate_top10fh_periods,
    available_top10fh_columns,
    build_top10fh_day_frame,
    build_top10fh_feature_frame,
    build_top10fh_lookup_by_date,
    build_top10fh_panel,
    top10fh_feature_columns,
)
from src.lazybull.features.factor_handlers import Top10FhFactorHandler
from src.lazybull.ml.train_core.constants import TOP10FH_FEATURE_COLUMNS

RAW_COLUMNS = [
    "ts_code",
    "end_date",
    "ann_date",
    "holder_name",
    "hold_amount",
    "hold_ratio",
    "hold_float_ratio",
    "hold_change",
    "holder_type",
]


def _rows():
    """构造样例（金额单位任意，仅需相对大小）。"""
    return [
        # 000001.SZ 20231231（ann 20240120）：10 名持有人，含社保与保险
        (
            "000001.SZ",
            "20231231",
            "20240120",
            "社保A",
            50.0,
            5.0,
            30.0,
            np.nan,
            "社保基金、社保机构",
        ),
        ("000001.SZ", "20231231", "20240120", "保险B", 30.0, 3.0, 18.0, np.nan, "保险投资组合"),
        (
            "000001.SZ",
            "20231231",
            "20240120",
            "券商C",
            20.0,
            2.0,
            12.0,
            np.nan,
            "金融机构—证券公司",
        ),
        ("000001.SZ", "20231231", "20240120", "自然人D", 10.0, 1.0, 6.0, np.nan, "自然人"),
        # 000001.SZ 20240331（ann 20240425）：集中度上升、社保退出
        (
            "000001.SZ",
            "20240331",
            "20240425",
            "社保A",
            60.0,
            6.0,
            40.0,
            np.nan,
            "社保基金、社保机构",
        ),
        ("000001.SZ", "20240331", "20240425", "基金会E", 20.0, 2.0, 12.0, np.nan, "公益基金"),
        # 同键同名多行（修订版并存）：取金额最大行（60.0 而非 40.0）
        ("000001.SZ", "20240331", "20240425", "自然人D", 40.0, 4.0, 8.0, np.nan, "自然人"),
        ("000001.SZ", "20240331", "20240425", "自然人D", 60.0, 6.0, 9.0, np.nan, "自然人"),
        # 000002.SZ 20231231（ann 20240131）：13 行 ⇒ 取金额前 10；
        # 金额最大行的 hold_float_ratio 缺失 ⇒ 按 0 贡献跳过（不整组置 NaN）
        ("000002.SZ", "20231231", "20240131", "缺失F", 500.0, 50.0, np.nan, np.nan, "自然人"),
        *[
            (
                "000002.SZ",
                "20231231",
                "20240131",
                f"H{i}",
                100.0 - i,
                1.0,
                10.0 - i,
                np.nan,
                "一般企业",
            )
            for i in range(12)
        ],
        # 000003.SZ：同日披露两个报告期（20231231 与 20240331）⇒ 取更近报告期
        ("000003.SZ", "20231231", "20240426", "甲A", 10.0, 1.0, 5.0, np.nan, "自然人"),
        ("000003.SZ", "20240331", "20240426", "甲B", 20.0, 2.0, 11.0, np.nan, "自然人"),
        # 000004.SZ：ann_date 非法 ⇒ 必须剔除
        ("000004.SZ", "20231231", "20241349", "非法G", 1.0, 0.1, 0.5, np.nan, "自然人"),
    ]


def _raw() -> pd.DataFrame:
    return pd.DataFrame(_rows(), columns=RAW_COLUMNS)


@pytest.fixture(scope="module")
def panel():
    return build_top10fh_panel(_raw())


def _period(panel, ts_code: str, end_date: str) -> pd.Series:
    sub = panel[(panel["ts_code"] == ts_code) & (panel["end_date"] == end_date)]
    assert len(sub) == 1, f"{ts_code} {end_date} 报告期行数应为 1，实际 {len(sub)}"
    return sub.iloc[0]


# ---------------------------------------------------------------- 报告期聚合


def test_aggregate_top10_ratio_and_top1(panel):
    """20231231：前 10 合计 = 30+18+12+6 = 66；第一大 = 30（社保）。"""
    row = _period(panel, "000001.SZ", "20231231")
    assert row[TOP10FH_TOP10_RATIO_COL] == pytest.approx(66.0)
    assert row[TOP10FH_TOP1_RATIO_COL] == pytest.approx(30.0)


def test_institutional_whitelist_is_exact_match(panel):
    """券商与公益基金不得计入长线机构；社保 + 保险 = 48。"""
    row = _period(panel, "000001.SZ", "20231231")
    assert row[TOP10FH_INST_RATIO_COL] == pytest.approx(48.0)
    assert row[TOP10FH_INST_COUNT_COL] == pytest.approx(2.0)
    assert row[TOP10FH_SOCIAL_FLAG_COL] == pytest.approx(1.0)


def test_same_name_multiple_rows_take_max_amount(panel):
    """同名两行（40.0 / 60.0）⇒ 取 60.0 对应行（流通比 9）。"""
    row = _period(panel, "000001.SZ", "20240331")
    # 40（社保）+ 12（公益基金，不计机构）+ 9（自然人） = 61
    assert row[TOP10FH_TOP10_RATIO_COL] == pytest.approx(61.0)
    assert row[TOP10FH_INST_RATIO_COL] == pytest.approx(40.0)
    assert row[TOP10FH_INST_COUNT_COL] == pytest.approx(1.0)
    assert row[TOP10FH_SOCIAL_FLAG_COL] == pytest.approx(1.0)


def test_top_n_cap_and_missing_ratio_skipped(panel):
    """13 行取金额前 10：缺比例的最大行占一席（跳过贡献），其余为 H0..H8 ⇒ 合计 54。"""
    row = _period(panel, "000002.SZ", "20231231")
    assert row[TOP10FH_TOP10_RATIO_COL] == pytest.approx(54.0)
    # 缺比例行既不计贡献、也不把整组拉成 NaN；top1 取组内 max（H0 = 10）
    assert row[TOP10FH_TOP1_RATIO_COL] == pytest.approx(10.0)


def test_concentration_change_aligned_to_previous_period(panel):
    """环比 = 本期 − 上一已存报告期；首期为 NaN。"""
    first = _period(panel, "000001.SZ", "20231231")
    second = _period(panel, "000001.SZ", "20240331")
    assert np.isnan(first[TOP10FH_CONCENTRATION_CHG_COL])
    assert second[TOP10FH_CONCENTRATION_CHG_COL] == pytest.approx(61.0 - 66.0)


def test_invalid_ann_date_dropped(panel):
    assert panel[panel["ts_code"] == "000004.SZ"].empty


def test_missing_required_columns_raise():
    with pytest.raises(ValueError, match="缺少必要列"):
        aggregate_top10fh_periods(pd.DataFrame({"ts_code": ["A"], "end_date": ["20231231"]}))
    no_ratio = _raw().drop(columns=["hold_float_ratio"])
    with pytest.raises(ValueError, match="hold_float_ratio"):
        aggregate_top10fh_periods(no_ratio)
    no_type = _raw().drop(columns=["holder_type"])
    with pytest.raises(ValueError, match="holder_type"):
        aggregate_top10fh_periods(no_type)


def test_empty_input_returns_empty_panel():
    panel = build_top10fh_panel(pd.DataFrame(columns=RAW_COLUMNS))
    assert panel.empty
    assert list(panel.columns) == [
        "ts_code",
        "end_date",
        TOP10FH_ANN_ORD_COL,
        TOP10FH_END_ORD_COL,
        *TOP10FH_COLS,
    ]


# ---------------------------------------------------------------- 单日截面（PIT）


def test_day_frame_pit_visibility():
    """公告日前不可见、公告当日起可见（PIT 无前视）。"""
    lookups = build_top10fh_lookup_by_date(_raw(), ["20240120", "20240119", "20240426", "20240425"])
    before = lookups["20240119"]
    assert "000001.SZ" not in set(before["ts_code"])  # 任何公告之前
    on_day = lookups["20240120"].set_index("ts_code")
    assert on_day.loc["000001.SZ", TOP10FH_TOP10_RATIO_COL] == pytest.approx(66.0)
    # 20240425 = 000001.SZ 第二期公告日 ⇒ 可见新期（61）
    assert lookups["20240425"].set_index("ts_code").loc[
        "000001.SZ", TOP10FH_TOP10_RATIO_COL
    ] == pytest.approx(61.0)
    # 同日披露两个报告期（000003.SZ ann=20240426）⇒ 取更近报告期 20240331（11）
    assert lookups["20240426"].set_index("ts_code").loc[
        "000003.SZ", TOP10FH_TOP10_RATIO_COL
    ] == pytest.approx(11.0)


def test_day_frame_keeps_latest_disclosure_only():
    """跨期只保留最近一次披露（20240331 之后不再回落到 20231231）。"""
    day = build_top10fh_day_frame(build_top10fh_panel(_raw()), "20240630").set_index("ts_code")
    assert day.loc["000001.SZ", TOP10FH_TOP10_RATIO_COL] == pytest.approx(61.0)
    assert day.loc["000001.SZ", TOP10FH_ANN_ORD_COL] > 0


def test_lookup_guard_for_long_range():
    long_range = [f"2024{m:02d}{d:02d}" for m in range(1, 13) for d in range(1, 26)]
    assert len(long_range) == 300
    with pytest.raises(ValueError, match="仅支持"):
        build_top10fh_lookup_by_date(_raw(), long_range)
    assert build_top10fh_lookup_by_date(_raw(), []) == {}


def test_invalid_trade_date_raises():
    with pytest.raises(ValueError, match="非法交易日"):
        build_top10fh_day_frame(build_top10fh_panel(_raw()), "2024-01-20")


# ---------------------------------------------------------------- handler / 派生


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {"ts_code": ["000001.SZ", "000002.SZ", "999999.SZ"], "close": [1.0, 2.0, 3.0]}
    )


def test_handler_returns_empty_when_family_disabled():
    assert Top10FhFactorHandler().apply(_features(), None, "20240122", pd.DataFrame()) == {}


def test_handler_writes_sentinel_and_nan_for_undisclosed(panel):
    """未披露股票 ⇒ NaN（**禁止 0 填充**）；哨兵列恒写当前版本。"""
    day = build_top10fh_day_frame(panel, "20240122")
    merged = _features().merge(day, on="ts_code", how="left")
    frame = build_top10fh_feature_frame(_features(), merged, "20240122")
    assert frame.loc[2, TOP10FH_TOP10_RATIO_COL] != frame.loc[2, TOP10FH_TOP10_RATIO_COL]  # NaN
    assert frame.loc[0, TOP10FH_TOP10_RATIO_COL] == pytest.approx(66.0)
    assert frame.loc[0, TOP10FH_FRESHNESS_COL] == pytest.approx(2.0)  # 20240120 → 20240122
    result = Top10FhFactorHandler().apply(_features(), day, "20240122", pd.DataFrame())
    assert result[TOP10FH_VERSION_COL].eq(TOP10FH_SCHEMA_VERSION).all()


def test_handler_empty_day_frame_keeps_schema(panel):
    result = Top10FhFactorHandler().apply(_features(), pd.DataFrame(), "20240122", pd.DataFrame())
    assert set(TOP10FH_COLS).issubset(result.keys())
    assert result[TOP10FH_TOP10_RATIO_COL].isna().all()
    assert result[TOP10FH_VERSION_COL].eq(TOP10FH_SCHEMA_VERSION).all()


def test_derive_matches_day_frame_path(panel):
    """训练侧运行时派生与 handler 逐日路径**逐值一致**（同一数值实现）。"""
    from src.lazybull.factors.top10_floatholders import derive_top10fh_columns

    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ", "999999.SZ"],
            "trade_date": ["20240122", "20240426", "20240426", "20240426"],
        },
        index=[10, 11, 12, 13],
    )
    derived = derive_top10fh_columns(frame, panel)
    assert set(derived) == set(available_top10fh_columns())
    # 20240122 可见 20231231 期（66），新鲜度 = 2 天
    assert frame.loc[10, TOP10FH_TOP10_RATIO_COL] == pytest.approx(66.0)
    assert frame.loc[10, TOP10FH_FRESHNESS_COL] == pytest.approx(2.0)
    # 20240426 起可见 20240331 期（61）
    assert frame.loc[11, TOP10FH_TOP10_RATIO_COL] == pytest.approx(61.0)
    # 未披露股票 ⇒ 全 NaN
    assert frame.loc[13, TOP10FH_COLS].isna().all()
    # 与 handler 逐日路径逐值一致
    day = build_top10fh_day_frame(panel, "20240122")
    one = frame.loc[[10]].reset_index(drop=True)
    result = Top10FhFactorHandler().apply(one, day, "20240122", pd.DataFrame())
    for col in TOP10FH_COLS + [TOP10FH_FRESHNESS_COL]:
        assert result[col].iloc[0] == pytest.approx(frame.loc[10, col], nan_ok=True)


def test_derive_empty_panel_writes_nan_and_sentinel():
    from src.lazybull.factors.top10_floatholders import derive_top10fh_columns

    frame = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20240122"]}, index=[0])
    derive_top10fh_columns(frame, None)
    assert frame[TOP10FH_TOP10_RATIO_COL].isna().all()
    assert frame[TOP10FH_FRESHNESS_COL].isna().all()
    assert frame[TOP10FH_VERSION_COL].eq(TOP10FH_SCHEMA_VERSION).all()


def test_derive_does_not_overwrite_existing_columns(panel):
    from src.lazybull.factors.top10_floatholders import derive_top10fh_columns

    frame = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": ["20240122"], TOP10FH_TOP10_RATIO_COL: [123.0]},
        index=[0],
    )
    derive_top10fh_columns(frame, panel)
    assert frame.loc[0, TOP10FH_TOP10_RATIO_COL] == pytest.approx(123.0)


def test_derive_requires_keys_and_unique_index(panel):
    from src.lazybull.factors.top10_floatholders import derive_top10fh_columns

    with pytest.raises(ValueError, match="缺少必要列"):
        derive_top10fh_columns(pd.DataFrame({"ts_code": ["A"]}), panel)
    with pytest.raises(ValueError, match="缺少必要列"):
        derive_top10fh_columns(pd.DataFrame({"trade_date": ["20240122"]}), panel)
    dup = pd.DataFrame(
        {"ts_code": ["A", "A"], "trade_date": ["20240122", "20240123"]}, index=[0, 0]
    )
    with pytest.raises(ValueError, match="索引唯一"):
        derive_top10fh_columns(dup, panel)
    bad_date = pd.DataFrame({"ts_code": ["A"], "trade_date": ["2024-01-22"]}, index=[0])
    with pytest.raises(ValueError, match="非法 trade_date"):
        derive_top10fh_columns(bad_date, panel)


# ---------------------------------------------------------------- 列清单单一来源


def test_available_columns_match_train_constants():
    assert set(available_top10fh_columns()) == set(TOP10FH_FEATURE_COLUMNS)
    assert set(TOP10FH_COLS).issubset(set(TOP10FH_FEATURE_COLUMNS))
    assert set(top10fh_feature_columns()) == set(TOP10FH_FEATURE_COLUMNS)


def test_concentration_feature_set_is_single_value_column():
    """单列臂：仅 `tfh_concentration_chg` + 哨兵（**不含 freshness** —— 与既有 freshness ρ=0.999）。"""
    cols = top10fh_feature_columns("concentration")
    assert cols == [TOP10FH_CONCENTRATION_CHG_COL, TOP10FH_VERSION_COL]
    assert TOP10FH_FRESHNESS_COL not in cols
    with pytest.raises(ValueError, match="未知 top10fh 列集"):
        top10fh_feature_columns("unknown-set")
