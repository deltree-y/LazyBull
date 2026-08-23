"""测试筹码胜率因子模块"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.cyq_perf import build_cyq_perf_lookup_by_date


@pytest.fixture
def mock_cyq_perf_data():
    """构造筹码胜率原始数据

    包含 2 只股票各 25 个交易日的数据，覆盖 diff(5) / diff(20) 计算。
    """
    dates = pd.bdate_range("2023-01-02", periods=25).strftime("%Y%m%d").tolist()
    stocks = ["000001.SZ", "600000.SH"]

    rows = []
    for i, d in enumerate(dates):
        for j, code in enumerate(stocks):
            base_wr = 50.0 + i * 0.5 + j * 5  # 胜率递增
            wa = 10.0 + j * 2  # 加权成本
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": d,
                    "winner_rate": base_wr,
                    "weight_avg": wa,
                    "cost_5pct": wa * 0.90,
                    "cost_15pct": wa * 0.92,
                    "cost_50pct": wa * 1.00,
                    "cost_85pct": wa * 1.10,
                    "cost_95pct": wa * 1.15,
                    "his_low": wa * 0.80,
                    "his_high": wa * 1.30,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trading_dates_25():
    """25 个交易日列表"""
    return pd.bdate_range("2023-01-02", periods=25).strftime("%Y%m%d").tolist()


class TestBuildCyqPerfLookup:
    """build_cyq_perf_lookup_by_date 单元测试"""

    def test_basic_lookup_structure(self, mock_cyq_perf_data, trading_dates_25):
        """验证返回字典基本结构：key 是交易日，value 是 DataFrame"""
        lookup = build_cyq_perf_lookup_by_date(mock_cyq_perf_data, trading_dates_25)

        assert isinstance(lookup, dict)
        assert len(lookup) == 25  # 每个交易日都应有数据

        # 任取一天验证列
        first_day = trading_dates_25[0]
        df = lookup[first_day]
        assert "ts_code" in df.columns
        assert "winner_rate" in df.columns
        assert "cost_concentration" in df.columns
        assert "weight_avg" in df.columns

    def test_winner_rate_values(self, mock_cyq_perf_data, trading_dates_25):
        """验证 winner_rate 值正确传递"""
        lookup = build_cyq_perf_lookup_by_date(mock_cyq_perf_data, trading_dates_25)
        day0 = lookup[trading_dates_25[0]]
        row = day0[day0["ts_code"] == "000001.SZ"].iloc[0]
        assert row["winner_rate"] == pytest.approx(50.0, abs=0.01)

    def test_cost_concentration(self, mock_cyq_perf_data, trading_dates_25):
        """验证筹码集中度 = (cost_85 - cost_15) / weight_avg"""
        lookup = build_cyq_perf_lookup_by_date(mock_cyq_perf_data, trading_dates_25)
        day0 = lookup[trading_dates_25[0]]
        row = day0[day0["ts_code"] == "000001.SZ"].iloc[0]
        wa = 10.0
        expected = (wa * 1.10 - wa * 0.92) / wa  # 0.18
        assert row["cost_concentration"] == pytest.approx(expected, abs=0.001)

    def test_winner_rate_chg_5(self, mock_cyq_perf_data, trading_dates_25):
        """验证 5 日胜率变化 = diff(5)"""
        lookup = build_cyq_perf_lookup_by_date(mock_cyq_perf_data, trading_dates_25)
        # 前 5 天的 chg_5 应为 NaN
        day3 = lookup[trading_dates_25[3]]
        row = day3[day3["ts_code"] == "000001.SZ"].iloc[0]
        assert np.isnan(row["winner_rate_chg_5"])

        # 第 6 天（index=5）的 chg_5 应有值
        day5 = lookup[trading_dates_25[5]]
        row5 = day5[day5["ts_code"] == "000001.SZ"].iloc[0]
        # winner_rate 每天 +0.5，5 日变化 = 2.5
        assert row5["winner_rate_chg_5"] == pytest.approx(2.5, abs=0.01)

    def test_winner_rate_chg_20(self, mock_cyq_perf_data, trading_dates_25):
        """验证 20 日胜率变化 = diff(20)"""
        lookup = build_cyq_perf_lookup_by_date(mock_cyq_perf_data, trading_dates_25)
        # 第 21 天（index=20）的 chg_20 应有值
        day20 = lookup[trading_dates_25[20]]
        row = day20[day20["ts_code"] == "000001.SZ"].iloc[0]
        # winner_rate 每天 +0.5，20 日变化 = 10.0
        assert row["winner_rate_chg_20"] == pytest.approx(10.0, abs=0.01)

    def test_winner_rate_chg_5_missing_day_alignment(self, mock_cyq_perf_data, trading_dates_25):
        """缺失数据日不静默跨期：对齐的 5 个交易日前缺数据时 chg_5 为 NaN。

        删除 000001.SZ 第 11 天（index=10）数据后：
        - index=15 的 chg_5 应对齐 index=10（缺失）→ NaN，
          行级 diff 会错误地取 index=9（跨 6 个日历日）；
        - index=16 的 chg_5 正确对齐 index=11。
        """
        dropped_date = trading_dates_25[10]
        df = mock_cyq_perf_data[
            ~(
                (mock_cyq_perf_data["ts_code"] == "000001.SZ")
                & (mock_cyq_perf_data["trade_date"] == dropped_date)
            )
        ].copy()
        lookup = build_cyq_perf_lookup_by_date(df, trading_dates_25)

        # 缺失日仅剩 600000.SH，000001.SZ 不应出现在该日截面中
        day_dropped = lookup[dropped_date]
        assert "000001.SZ" not in day_dropped["ts_code"].tolist()

        # 5 个交易日前（index=10）缺失 → chg_5 为 NaN，而非跨到 index=9
        day15 = lookup[trading_dates_25[15]]
        row15 = day15[day15["ts_code"] == "000001.SZ"].iloc[0]
        assert np.isnan(row15["winner_rate_chg_5"])

        # index=16 的 chg_5 正确对齐 index=11：0.5 * 5 = 2.5
        day16 = lookup[trading_dates_25[16]]
        row16 = day16[day16["ts_code"] == "000001.SZ"].iloc[0]
        assert row16["winner_rate_chg_5"] == pytest.approx(2.5, abs=0.01)

        # 20 日对齐不受缺失日影响（index=20 对齐 index=0）
        day20 = lookup[trading_dates_25[20]]
        row20 = day20[day20["ts_code"] == "000001.SZ"].iloc[0]
        assert row20["winner_rate_chg_20"] == pytest.approx(10.0, abs=0.01)

        # 另一只股票不受影响
        day15_other = lookup[trading_dates_25[15]]
        row_other = day15_other[day15_other["ts_code"] == "600000.SH"].iloc[0]
        assert row_other["winner_rate_chg_5"] == pytest.approx(2.5, abs=0.01)

    def test_winner_rate_chg_5_missing_market_day_alignment(
        self, mock_cyq_perf_data, trading_dates_25
    ):
        """全市场缺失某交易日时，日历应保留空位而非位置压缩。

        删除两只股票在 index=10 的全部数据（模拟该日 cyq 分区缺失）：
        - index=14 的 chg_5 仍应严格对齐 5 个交易日前的 index=9；
        - index=15 的 chg_5 对齐 index=10（空位）→ NaN。
        """
        dropped_date = trading_dates_25[10]
        df = mock_cyq_perf_data[mock_cyq_perf_data["trade_date"] != dropped_date].copy()
        lookup = build_cyq_perf_lookup_by_date(df, trading_dates_25)

        # 该日截面整体缺失
        assert dropped_date not in lookup

        # index=14 对齐 index=9：0.5 * 5 = 2.5（位置压缩会错误地取 index=8）
        day14 = lookup[trading_dates_25[14]]
        row14 = day14[day14["ts_code"] == "000001.SZ"].iloc[0]
        assert row14["winner_rate_chg_5"] == pytest.approx(2.5, abs=0.01)

        # index=15 对齐 index=10（全市场空位）→ NaN
        day15 = lookup[trading_dates_25[15]]
        row15 = day15[day15["ts_code"] == "000001.SZ"].iloc[0]
        assert np.isnan(row15["winner_rate_chg_5"])

    def test_winner_rate_chg_5_missing_market_day_with_calendar_dates(
        self, mock_cyq_perf_data, trading_dates_25
    ):
        """ensure 链路：输出仅单日、日历补传完整交易日，全市场缺失日同样对齐。"""
        dropped_date = trading_dates_25[10]
        df = mock_cyq_perf_data[mock_cyq_perf_data["trade_date"] != dropped_date].copy()
        # 仅输出单日（模拟 ensure 的 factor_output_dates=[trade_date]）
        output_dates = [trading_dates_25[15]]
        lookup = build_cyq_perf_lookup_by_date(df, output_dates, calendar_dates=trading_dates_25)
        assert set(lookup.keys()) == set(output_dates)
        day15 = lookup[trading_dates_25[15]]
        row15 = day15[day15["ts_code"] == "000001.SZ"].iloc[0]
        # 5 个交易日前（index=10）全市场缺失 → NaN
        assert np.isnan(row15["winner_rate_chg_5"])
        # 20 个交易日前在数据起点之外 → NaN（与完整数据路径行为一致）
        assert np.isnan(row15["winner_rate_chg_20"])

    def test_date_normalization(self, trading_dates_25):
        """验证日期带横线也能正确匹配"""
        df = pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["2023-01-02"],  # 带横线
                "winner_rate": [55.0],
                "weight_avg": [10.0],
                "cost_15pct": [9.2],
                "cost_85pct": [11.0],
            }
        )
        lookup = build_cyq_perf_lookup_by_date(df, trading_dates_25)
        assert "20230102" in lookup

    def test_empty_input(self, trading_dates_25):
        """空数据应返回空字典"""
        df = pd.DataFrame(columns=["ts_code", "trade_date", "winner_rate"])
        lookup = build_cyq_perf_lookup_by_date(df, trading_dates_25)
        assert lookup == {}

    def test_filtered_by_trading_dates(self, mock_cyq_perf_data):
        """只返回指定交易日的数据"""
        subset = ["20230102", "20230103"]
        lookup = build_cyq_perf_lookup_by_date(mock_cyq_perf_data, subset)
        assert set(lookup.keys()).issubset(set(subset))

    def test_two_stocks_per_day(self, mock_cyq_perf_data, trading_dates_25):
        """每个交易日应有 2 只股票"""
        lookup = build_cyq_perf_lookup_by_date(mock_cyq_perf_data, trading_dates_25)
        day0 = lookup[trading_dates_25[0]]
        assert len(day0) == 2
