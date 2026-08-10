"""龙虎榜因子单元测试"""

import pandas as pd

from src.lazybull.factors.lhb import LHB_COLS, build_lhb_lookup_by_date


def _make_top_list_df() -> pd.DataFrame:
    # 3 天, 2 只股票, 其中 000001 第 1 天有 2 条上榜理由
    return pd.DataFrame(
        [
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 1_000_000.0, "net_rate": 0.02,
                "amount_rate": 0.15, "reason": "日涨幅偏离值达 7%",
            },
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 500_000.0, "net_rate": 0.01,
                "amount_rate": 0.10, "reason": "换手率达 20%",
            },
            {
                "trade_date": "20240102", "ts_code": "000002.SZ",
                "net_amount": -800_000.0, "net_rate": -0.03,
                "amount_rate": 0.12, "reason": "日跌幅偏离值",
            },
            {
                "trade_date": "20240103", "ts_code": "000001.SZ",
                "net_amount": 200_000.0, "net_rate": 0.005,
                "amount_rate": 0.08, "reason": "日涨幅偏离值",
            },
            {
                "trade_date": "20240104", "ts_code": "000002.SZ",
                "net_amount": 300_000.0, "net_rate": 0.01,
                "amount_rate": 0.09, "reason": "日涨幅偏离值",
            },
        ]
    )


def _cal(start: str = "20240101", periods: int = 60) -> list:
    """构造连续交易日历（工作日频率）。"""
    return pd.date_range(start, periods=periods, freq="B").strftime("%Y%m%d").tolist()


def test_lhb_lookup_basic():
    df = _make_top_list_df()
    trading_dates = ["20240102", "20240103", "20240104"]
    result = build_lhb_lookup_by_date(df, trading_dates)

    assert set(result.keys()) == set(trading_dates)
    for td, frame in result.items():
        assert "ts_code" in frame.columns
        for col in LHB_COLS:
            assert col in frame.columns, f"{td} 缺列 {col}"


def test_lhb_same_day_multiple_reasons_take_abs_max():
    """同日多理由取净买入绝对值最大的一条（而非求和, 避免同周期重复放大）。"""
    df = _make_top_list_df()
    result = build_lhb_lookup_by_date(df, ["20240102", "20240103", "20240104"])
    day0 = result["20240102"]
    row = day0[day0["ts_code"] == "000001.SZ"].iloc[0]
    # 两条上榜理由的净买入应取绝对值最大的一条, 而不是相加
    assert row["lhb_net_amount"] == 1_000_000.0
    # 去重后的上榜理由数
    assert row["lhb_reason_count"] == 2
    assert row["lhb_on_list"] == 1.0
    # 两条理由均不含"连续", 非连续异动上榜
    assert row["lhb_cont_on_list"] == 0.0


def test_lhb_rolling_up_days_keeps_timeline_continuity():
    """滚动按交易日窗口, 且未上榜日保留近 20 日历史累计（时序连续性）。"""
    df = _make_top_list_df()
    trading_dates = ["20240102", "20240103", "20240104"]
    result = build_lhb_lookup_by_date(df, trading_dates)
    # 000001 连续两日上榜, 第 2 日 up_days_20 应为 2
    row = result["20240103"][result["20240103"]["ts_code"] == "000001.SZ"].iloc[0]
    assert row["lhb_up_days_20"] == 2.0
    # 000001 第 3 日未上榜, 但近 20 交易日内有上榜, 仍应出现在 lookup 中
    row3 = result["20240104"][result["20240104"]["ts_code"] == "000001.SZ"]
    assert len(row3) == 1
    assert row3.iloc[0]["lhb_up_days_20"] == 2.0
    assert row3.iloc[0]["lhb_on_list"] == 0.0


def test_lhb_rolling_window_decays_after_20_days():
    """两次上榜间隔超过 20 个交易日时, 第一次不再计入（真正按交易日窗口）。"""
    calendar = _cal("20240101", 60)
    df = pd.DataFrame(
        [
            {
                "trade_date": calendar[0], "ts_code": "600000.SH",
                "net_amount": 1e8, "net_rate": 0.02, "amount_rate": 0.1,
                "reason": "日涨幅偏离值达 7%",
            },
            {
                "trade_date": calendar[30], "ts_code": "600000.SH",
                "net_amount": 2e8, "net_rate": 0.03, "amount_rate": 0.15,
                "reason": "日涨幅偏离值达 7%",
            },
        ]
    )
    result = build_lhb_lookup_by_date(
        df, [calendar[0], calendar[30]], calendar_dates=calendar
    )
    row0 = result[calendar[0]][result[calendar[0]]["ts_code"] == "600000.SH"].iloc[0]
    assert row0["lhb_up_days_20"] == 1.0
    # 第二次上榜时第一次已在 20 交易日窗口外
    row = result[calendar[30]][result[calendar[30]]["ts_code"] == "600000.SH"].iloc[0]
    assert row["lhb_up_days_20"] == 1.0
    assert row["lhb_net_sum_5"] == 2e8


def test_lhb_excludes_continuous_reason():
    """同日同时存在单日榜与连续类记录时, 优先取单日榜（避免重叠周期重复放大）。"""
    df = pd.DataFrame(
        [
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 1e8, "net_rate": 0.02, "amount_rate": 0.1,
                "reason": "日涨幅偏离值达 7%",
            },
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 5e8, "net_rate": 0.1, "amount_rate": 0.2,
                "reason": "连续三个交易日内，涨幅偏离值累计达到20%的证券",
            },
        ]
    )
    result = build_lhb_lookup_by_date(df, ["20240102"])
    row = result["20240102"][result["20240102"]["ts_code"] == "000001.SZ"].iloc[0]
    # 优先单日榜的 1e8, 而不是求和 6e8
    assert row["lhb_net_amount"] == 1e8
    assert row["lhb_reason_count"] == 1
    # 当日同时存在连续类理由, 连续异动信号记为 1（事件级, 与选中主记录无关）
    assert row["lhb_cont_on_list"] == 1.0


def test_lhb_only_continuous_reason_kept():
    """当日只有连续类理由时必须保留（不能误标为未上榜）。"""
    df = pd.DataFrame(
        [
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 5e8, "net_rate": 0.1, "amount_rate": 0.2,
                "reason": "连续三个交易日内，涨幅偏离值累计达到20%的证券",
            },
        ]
    )
    result = build_lhb_lookup_by_date(df, ["20240102"])
    row = result["20240102"][result["20240102"]["ts_code"] == "000001.SZ"].iloc[0]
    assert row["lhb_on_list"] == 1.0
    assert row["lhb_net_amount"] == 5e8
    assert row["lhb_reason_count"] == 1
    assert row["lhb_cont_on_list"] == 1.0


def test_lhb_all_na_net_amount_no_crash():
    """净买入全为 NaN 的股票日不应崩溃（idxmax 返回 NaN 索引的边界）。"""
    df = pd.DataFrame(
        [
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": float("nan"), "net_rate": 0.02,
                "amount_rate": 0.1, "reason": "日涨幅偏离值达 7%",
            },
            {
                "trade_date": "20240102", "ts_code": "000002.SZ",
                "net_amount": float("nan"), "net_rate": float("nan"),
                "amount_rate": float("nan"), "reason": "日涨幅偏离值达 7%",
            },
        ]
    )
    result = build_lhb_lookup_by_date(df, ["20240102"])
    codes = result["20240102"]["ts_code"].tolist()
    assert "000001.SZ" in codes
    assert "000002.SZ" in codes
    row = result["20240102"][result["20240102"]["ts_code"] == "000001.SZ"].iloc[0]
    assert row["lhb_on_list"] == 1.0
    assert row["lhb_net_amount"] == 0.0


def test_lhb_calendar_warmup():
    """日历早于输出首日时, 输出首日也能得到历史累计（批量构建预热一致性）。"""
    calendar = _cal("20240101", 60)
    df = pd.DataFrame(
        [
            {
                "trade_date": calendar[0], "ts_code": "600000.SH",
                "net_amount": 1e8, "net_rate": 0.02, "amount_rate": 0.1,
                "reason": "日涨幅偏离值达 7%",
            },
        ]
    )
    # 输出交易日从 calendar[10] 开始（前 10 个交易日是预热）
    output_dates = calendar[10:20]
    result = build_lhb_lookup_by_date(df, output_dates, calendar_dates=calendar)
    # 首日 calendar[10] 距上榜 10 个交易日 (< 20), 应保留历史累计
    row = result[output_dates[0]][result[output_dates[0]]["ts_code"] == "600000.SH"].iloc[0]
    assert row["lhb_up_days_20"] == 1.0
    # 对照: 不传 calendar_dates 时默认日历=输出日期, 预热被 reindex 丢弃
    result2 = build_lhb_lookup_by_date(df, output_dates)
    assert output_dates[0] not in result2


def test_lhb_missing_ts_code_returns_empty():
    """历史残留的空占位（仅 trade_date 列）不应崩溃, 应返回空。"""
    df = pd.DataFrame({"trade_date": ["20240102"]})
    assert build_lhb_lookup_by_date(df, ["20240102"]) == {}


def test_lhb_no_field_cross_join():
    """组内不同列 NaN 分布时, 应保留第一整行而非拼接不同记录的字段。"""
    df = pd.DataFrame(
        [
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 1e8, "net_rate": float("nan"),
                "amount_rate": 0.1, "reason": "日涨幅偏离值达 7%",
            },
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 5e7, "net_rate": 0.2,
                "amount_rate": float("nan"), "reason": "日涨幅偏离值达 7%",
            },
        ]
    )
    result = build_lhb_lookup_by_date(df, ["20240102"])
    row = result["20240102"][result["20240102"]["ts_code"] == "000001.SZ"].iloc[0]
    # 选中 net_amount 最大(1e8)的那条整行: net_rate 应保持 NaN(填充 0),
    # 而不是把另一条记录的 0.2 拼接进来
    assert row["lhb_net_amount"] == 1e8
    assert row["lhb_net_rate"] == 0.0
    assert row["lhb_amount_rate"] == 0.1


def test_lhb_cont_on_list_event_signal():
    """lhb_cont_on_list 是事件级信号: 当日任一条 reason 含"连续"即记为 1,
    即使主记录选中单日榜; 仅单日类理由时记为 0。"""
    df = pd.DataFrame(
        [
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 1e8, "net_rate": 0.02, "amount_rate": 0.1,
                "reason": "日涨幅偏离值达 7%",
            },
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 5e8, "net_rate": 0.1, "amount_rate": 0.2,
                "reason": "连续三个交易日内，涨幅偏离值累计达到20%的证券",
            },
            {
                "trade_date": "20240103", "ts_code": "000002.SZ",
                "net_amount": 3e7, "net_rate": 0.01, "amount_rate": 0.05,
                "reason": "日涨幅偏离值达 7%",
            },
        ]
    )
    result = build_lhb_lookup_by_date(df, ["20240102", "20240103"])
    # 当日单日榜与连续类并存: 主记录选中单日榜(1e8), 但连续异动信号=1
    row = result["20240102"][result["20240102"]["ts_code"] == "000001.SZ"].iloc[0]
    assert row["lhb_net_amount"] == 1e8
    assert row["lhb_cont_on_list"] == 1.0
    # 仅单日类理由: cont=0
    row2 = result["20240103"][result["20240103"]["ts_code"] == "000002.SZ"].iloc[0]
    assert row2["lhb_cont_on_list"] == 0.0


def test_lhb_cont_on_list_non_listed_day_zero():
    """未上榜但近 20 日上过榜的股票, cont 标记为 0（仅上榜当日为 1）。"""
    df = pd.DataFrame(
        [
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 5e8, "net_rate": 0.1, "amount_rate": 0.2,
                "reason": "连续三个交易日内，涨幅偏离值累计达到20%的证券",
            },
        ]
    )
    result = build_lhb_lookup_by_date(df, ["20240102", "20240103"])
    row = result["20240102"][result["20240102"]["ts_code"] == "000001.SZ"].iloc[0]
    assert row["lhb_cont_on_list"] == 1.0
    # 第 2 日未上榜: cont=0
    row2 = result["20240103"][result["20240103"]["ts_code"] == "000001.SZ"].iloc[0]
    assert row2["lhb_on_list"] == 0.0
    assert row2["lhb_cont_on_list"] == 0.0


def test_lhb_cont_on_list_missing_reason_column():
    """reason 列缺失时 cont 标记恒为 0（不崩溃）。"""
    df = pd.DataFrame(
        [
            {
                "trade_date": "20240102", "ts_code": "000001.SZ",
                "net_amount": 1e8, "net_rate": 0.02, "amount_rate": 0.1,
            },
        ]
    )
    result = build_lhb_lookup_by_date(df, ["20240102"])
    row = result["20240102"][result["20240102"]["ts_code"] == "000001.SZ"].iloc[0]
    assert row["lhb_cont_on_list"] == 0.0
    assert row["lhb_cont_up_days_5"] == 0.0
    assert row["lhb_cont_up_days_20"] == 0.0


def test_lhb_cont_up_days_rolling():
    """近 5/20 交易日连续异动上榜次数滚动累计（含 5 日窗口衰减与 20 日保留）。"""
    calendar = _cal("20240101", 60)
    df = pd.DataFrame(
        [
            {
                "trade_date": calendar[i], "ts_code": "600000.SH",
                "net_amount": 1e8, "net_rate": 0.02, "amount_rate": 0.1,
                "reason": "连续三个交易日内，涨幅偏离值累计达到20%的证券",
            }
            for i in range(6)
        ]
    )
    output_dates = calendar[2:15]
    result = build_lhb_lookup_by_date(df, output_dates, calendar_dates=calendar)

    def _row_at(td):
        return result[td][result[td]["ts_code"] == "600000.SH"].iloc[0]

    # 第 2 日（第 3 个连续异动上榜日）: 近 5 日累计 3 次, 近 20 日累计 3 次
    r = _row_at(calendar[2])
    assert r["lhb_cont_up_days_5"] == 3.0
    assert r["lhb_cont_up_days_20"] == 3.0
    # 第 5 日（第 6 个连续异动上榜日）: 近 5 日累计封顶 5 次, 近 20 日累计 6 次
    r = _row_at(calendar[5])
    assert r["lhb_cont_up_days_5"] == 5.0
    assert r["lhb_cont_up_days_20"] == 6.0
    # 第 10 日（未上榜）: 最近上榜在第 5 日, 距 5 日窗口之外 → 5 日累计归 0;
    # 但仍在 20 日窗口内 → 20 日累计保留 6（时序连续性）
    r = _row_at(calendar[10])
    assert r["lhb_on_list"] == 0.0
    assert r["lhb_cont_up_days_5"] == 0.0
    assert r["lhb_cont_up_days_20"] == 6.0


def test_lhb_empty():
    assert build_lhb_lookup_by_date(pd.DataFrame(), ["20240102"]) == {}
    assert build_lhb_lookup_by_date(None, ["20240102"]) == {}
