# -*- coding: utf-8 -*-
"""top_inst（龙虎榜机构席位）因子测试。

覆盖：清洗三规则（列互换修正 / 三元组去重 / 单日榜优先）、机构聚合、
查询表窗口语义、特征帧归一化与 0 填充、handler 与运行时派生、列清单双处一致性。
所有测试自造数据，不依赖真实生产数据。
"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.top_inst import (
    TOP_INST_COLS,
    TOP_INST_VERSION_COL,
    aggregate_inst_daily,
    build_top_inst_feature_frame,
    build_top_inst_lookup_by_date,
    clean_top_inst,
    derive_top_inst_columns,
    top_inst_feature_columns,
)
from src.lazybull.features.factor_handlers import TopInstFactorHandler

_RAW_COLS = ["ts_code", "trade_date", "exalter", "side", "buy", "sell", "net_buy", "reason"]


def _raw(rows):
    return pd.DataFrame(rows, columns=_RAW_COLS)


def _row(code="A.SZ", day="20240102", exalter="机构专用", side="0",
         buy=1_000_000.0, sell=0.0, net_buy=1_000_000.0,
         reason="日涨幅偏离值达到7%的前5只证券"):
    return dict(ts_code=code, trade_date=day, exalter=exalter, side=side,
                buy=buy, sell=sell, net_buy=net_buy, reason=reason)


# ---------------------------------------------------------------- 清洗三规则

def test_clean_fixes_swapped_buy_sell():
    """S1：|sell−buy−net|<1 且 |buy−sell−net|≥1 → 交换（旧格式行）。"""
    df = _raw([_row(buy=0.0, sell=1_000_000.0, net_buy=1_000_000.0,
                    reason="涨幅偏离值达7%的证券")])
    out = clean_top_inst(df)
    assert out.iloc[0]["buy"] == 1_000_000.0
    assert out.iloc[0]["sell"] == 0.0
    assert abs(out.iloc[0]["buy"] - out.iloc[0]["sell"] - out.iloc[0]["net_buy"]) < 1.0


def test_clean_dedups_repeated_disclosure():
    """S2：同一事实（多 reason × 多 side 笛卡尔展开）→ 按修正后金额三元组去重。"""
    rows = [
        _row(side="0", reason="换手率达20%的证券"),
        _row(side="0", reason="涨幅偏离值达7%的证券"),
        _row(side="1", reason="换手率达20%的证券"),
        _row(side="1", reason="涨幅偏离值达7%的证券"),
        _row(side="0", buy=2_000_000.0, net_buy=2_000_000.0,
             reason="日换手率达到20%的前五只证券"),
    ]
    out = clean_top_inst(_raw(rows))
    # 前 4 行 → 2 行（side 0/1 各一）；第 5 行不同金额保留
    assert len(out) == 3


def test_clean_prefers_daily_reason():
    """S3：有单日榜行时剔除连续榜行；仅连续榜时保留。"""
    rows = [
        _row(code="A.SZ", reason="日涨幅偏离值达到7%的前5只证券", buy=1_000_000.0,
             net_buy=1_000_000.0),
        _row(code="A.SZ", reason="连续三个交易日内，涨幅偏离值累计达到20%的证券",
             buy=9_000_000.0, net_buy=9_000_000.0),
        _row(code="B.SZ", reason="连续三个交易日内，涨幅偏离值累计达到20%的证券",
             buy=5_000_000.0, net_buy=5_000_000.0),
    ]
    out = clean_top_inst(_raw(rows))
    a = out[out["ts_code"] == "A.SZ"]
    b = out[out["ts_code"] == "B.SZ"]
    assert len(a) == 1 and a.iloc[0]["net_buy"] == 1_000_000.0
    assert len(b) == 1 and b.iloc[0]["net_buy"] == 5_000_000.0


def test_clean_requires_columns():
    with pytest.raises(ValueError, match="缺少必需列"):
        clean_top_inst(pd.DataFrame({"ts_code": ["A"]}))


# ---------------------------------------------------------------- 机构聚合

def test_aggregate_sums_multiple_inst_rows():
    """同榜单下多行机构（多机构分列披露）→ sum；非机构行排除。"""
    rows = [
        _row(buy=1_000_000.0, sell=200_000.0, net_buy=800_000.0),
        _row(buy=500_000.0, sell=100_000.0, net_buy=400_000.0,
             reason="日换手率达到20%的前五只证券"),
        _row(exalter="中信证券股份有限公司杭州庆春路证券营业部", buy=9_999.0, net_buy=9_999.0),
    ]
    daily = aggregate_inst_daily(clean_top_inst(_raw(rows)))
    assert len(daily) == 1
    r = daily.iloc[0]
    # 前两行金额不同（非重复披露）→ 多机构明细保留 → sum
    assert r["inst_buy"] == 1_500_000.0
    assert r["inst_sell"] == 300_000.0
    assert r["inst_net"] == 1_200_000.0
    assert r["inst_rows"] == 2


# ---------------------------------------------------------------- 查询表窗口

def test_lookup_window_20_trading_days():
    dates = [f"202401{d:02d}" for d in range(2, 27)]  # 25 个"交易日"
    daily = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240107"],
            "ts_code": ["A.SZ", "B.SZ"],
            "inst_buy": [1_000_000.0, 500_000.0],
            "inst_sell": [0.0, 0.0],
            "inst_net": [1_000_000.0, 500_000.0],
            "inst_rows": [1, 1],
        }
    )
    lookup = build_top_inst_lookup_by_date(daily, dates)
    # A：窗口 [t-19, t] → t=0..19 输出（t=0 窗口仅含 0 日）
    a_days = [d for d in dates if d in lookup and (lookup[d]["ts_code"] == "A.SZ").any()]
    assert dates[0] in a_days and dates[19] in a_days and dates[20] not in a_days
    # B：记录在 index 5 → 仅 index 5..24 输出
    b_days = [d for d in dates if d in lookup and (lookup[d]["ts_code"] == "B.SZ").any()]
    assert dates[4] not in b_days and dates[5] in b_days

    day0 = lookup[dates[0]]
    a0 = day0[day0["ts_code"] == "A.SZ"].iloc[0]
    assert a0["ti_inst_days_20"] == 1.0
    assert a0["ti_inst_net_amt"] == 1_000_000.0
    assert a0["ti_inst_net_sum_20_amt"] == 1_000_000.0

    # A 在 index 8（当日无记录）→ 当日值 0、days 仍为 1
    day8 = lookup[dates[8]]
    a8 = day8[day8["ts_code"] == "A.SZ"].iloc[0]
    assert a8["ti_inst_net_amt"] == 0.0
    assert a8["ti_inst_net_sum_20_amt"] == 1_000_000.0
    assert a8["ti_inst_days_20"] == 1.0


# ---------------------------------------------------------------- 特征帧

def _features(n=2):
    return pd.DataFrame({"ts_code": ["A.SZ", "B.SZ"][:n], "circ_mv": [100.0, 200.0][:n]})


def test_feature_frame_ratio_and_fill():
    feats = _features()
    merged = pd.DataFrame(
        {
            "ti_inst_buy_amt": [500_000.0, np.nan],
            "ti_inst_sell_amt": [0.0, np.nan],
            "ti_inst_net_amt": [500_000.0, np.nan],
            "ti_inst_net_sum_20_amt": [1_000_000.0, np.nan],
            "ti_inst_days_20": [2.0, np.nan],
        },
        index=feats.index,
    )
    frame = build_top_inst_feature_frame(feats, merged)
    # circ_mv 100 万元 = 1e6 元 → 500_000 / 1e6 = 0.5
    assert frame["ti_inst_net_ratio"].iloc[0] == pytest.approx(0.5)
    assert frame["ti_inst_net_sum_20"].iloc[0] == pytest.approx(1.0)
    assert frame["ti_inst_days_20"].iloc[0] == 2.0
    # 缺失行 → 0 填充（含全部值列）
    assert (frame.iloc[1][TOP_INST_COLS] == 0.0).all()


def test_feature_frame_requires_circ_mv():
    feats = pd.DataFrame({"ts_code": ["A.SZ"]})
    with pytest.raises(ValueError, match="circ_mv"):
        build_top_inst_feature_frame(feats, None)


# ---------------------------------------------------------------- handler 与派生

def test_handler_disabled_and_empty_semantics():
    feats = _features()
    handler = TopInstFactorHandler()
    # data=None ⇒ 未启用本族（不输出列）
    assert handler.apply(feats, None, "20240102", pd.DataFrame()) == {}
    # 空表 ⇒ 0 填充 + 哨兵恒写
    res = handler.apply(feats, pd.DataFrame(), "20240102", pd.DataFrame())
    assert (res["ti_inst_net_ratio"] == 0.0).all()
    assert (res[TOP_INST_VERSION_COL] == 1).all()


def test_derive_columns_and_no_overwrite():
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["A.SZ", "A.SZ"],
            "circ_mv": [100.0, 100.0],
            "ti_inst_net_ratio": [999.0, 999.0],  # 已存在列（不被覆盖）
        }
    )
    day = pd.DataFrame(
        {
            "ts_code": ["A.SZ"],
            "ti_inst_buy_amt": [500_000.0],
            "ti_inst_sell_amt": [0.0],
            "ti_inst_net_amt": [500_000.0],
            "ti_inst_net_sum_20_amt": [1_500_000.0],
            "ti_inst_days_20": [3.0],
        }
    )
    lookup = {"20240102": day}
    derived = derive_top_inst_columns(frame, lookup)
    assert "ti_inst_buy_ratio" in derived
    assert "ti_inst_net_ratio" not in derived  # 已存在 → 不覆盖
    assert frame.loc[0, "ti_inst_buy_ratio"] == pytest.approx(0.5)
    assert frame.loc[0, "ti_inst_days_20"] == 3.0
    # 查询表缺该日 → 0 填充
    assert frame.loc[1, "ti_inst_buy_ratio"] == 0.0
    assert (frame[TOP_INST_VERSION_COL] == 1).all()


def test_derive_requires_columns():
    frame = pd.DataFrame({"ts_code": ["A.SZ"]})
    with pytest.raises(ValueError, match="缺少必要列"):
        derive_top_inst_columns(frame, {})


# ---------------------------------------------------------------- 列清单双处一致

def test_feature_columns_single_source():
    from src.lazybull.ml.train_core.constants import TOP_INST_FEATURE_COLUMNS

    assert top_inst_feature_columns() == TOP_INST_FEATURE_COLUMNS
