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
            rows.append({
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
            })
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

    def test_date_normalization(self, trading_dates_25):
        """验证日期带横线也能正确匹配"""
        df = pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "trade_date": ["2023-01-02"],  # 带横线
            "winner_rate": [55.0],
            "weight_avg": [10.0],
            "cost_15pct": [9.2],
            "cost_85pct": [11.0],
        })
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
