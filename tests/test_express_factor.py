"""测试业绩快报因子模块"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.express import build_express_lookup_by_date


@pytest.fixture
def mock_express_data():
    """构造业绩快报数据

    包含 2 只股票，各有 2 个报告期（Q3/Q4），不同 ann_date 发布。
    """
    return pd.DataFrame([
        # 000001.SZ：Q3 公告日 2023-10-28，Q4 公告日 2024-01-20
        {
            "ts_code": "000001.SZ", "ann_date": "20231028",
            "end_date": "20230930",
            "revenue_yoy": 15.5, "n_income_yoy": 20.3, "roe": 12.0,
        },
        {
            "ts_code": "000001.SZ", "ann_date": "20240120",
            "end_date": "20231231",
            "revenue_yoy": 18.0, "n_income_yoy": 22.5, "roe": 13.0,
        },
        # 600000.SH：仅 Q3
        {
            "ts_code": "600000.SH", "ann_date": "20231030",
            "end_date": "20230930",
            "revenue_yoy": -5.0, "n_income_yoy": -8.0, "roe": 6.5,
        },
    ])


@pytest.fixture
def mock_forecast_data():
    """构造预告数据，用于计算 express_surprise"""
    return pd.DataFrame([
        {
            "ts_code": "000001.SZ", "ann_date": "20231015",
            "end_date": "20230930",
            "p_change_min": 15.0, "p_change_max": 25.0,
        },
        {
            "ts_code": "000001.SZ", "ann_date": "20240110",
            "end_date": "20231231",
            "p_change_min": 10.0, "p_change_max": 20.0,
        },
    ])


@pytest.fixture
def trading_dates_express():
    """横跨 Q3/Q4 公告期的交易日"""
    return [
        "20231025",  # 所有公告前
        "20231029",  # 000001 Q3 已公告
        "20231031",  # 600000 Q3 也已公告
        "20240115",  # Q4 公告前
        "20240121",  # 000001 Q4 已公告
    ]


class TestBuildExpressLookup:
    """build_express_lookup_by_date 单元测试"""

    def test_point_in_time_before_any(self, mock_express_data, trading_dates_express):
        """公告前不应有数据"""
        lookup = build_express_lookup_by_date(mock_express_data, trading_dates_express)
        assert "20231025" not in lookup

    def test_point_in_time_after_q3(self, mock_express_data, trading_dates_express):
        """Q3 公告后应有 000001 数据"""
        lookup = build_express_lookup_by_date(mock_express_data, trading_dates_express)
        df = lookup["20231029"]
        assert len(df) == 1
        row = df[df["ts_code"] == "000001.SZ"].iloc[0]
        assert row["express_revenue_yoy"] == pytest.approx(15.5)
        assert row["express_profit_yoy"] == pytest.approx(20.3)
        assert row["express_roe"] == pytest.approx(12.0)

    def test_point_in_time_both_announced(self, mock_express_data, trading_dates_express):
        """两只股票的 Q3 都公告后"""
        lookup = build_express_lookup_by_date(mock_express_data, trading_dates_express)
        df = lookup["20231031"]
        assert len(df) == 2

    def test_q4_overwrites_q3(self, mock_express_data, trading_dates_express):
        """Q4 公告后应覆盖 Q3"""
        lookup = build_express_lookup_by_date(mock_express_data, trading_dates_express)
        df = lookup["20240121"]
        row = df[df["ts_code"] == "000001.SZ"].iloc[0]
        assert row["express_revenue_yoy"] == pytest.approx(18.0)
        assert row["express_profit_yoy"] == pytest.approx(22.5)

    def test_express_surprise_with_forecast(
        self, mock_express_data, mock_forecast_data, trading_dates_express
    ):
        """验证 express_surprise = 实际增速 - 预告中位数"""
        lookup = build_express_lookup_by_date(
            mock_express_data, trading_dates_express, forecast_df=mock_forecast_data
        )
        # Q3: n_income_yoy=20.3, forecast_mid=(15+25)/2=20 → surprise=0.3
        df_q3 = lookup["20231029"]
        row = df_q3[df_q3["ts_code"] == "000001.SZ"].iloc[0]
        assert row["express_surprise"] == pytest.approx(0.3, abs=0.01)

        # Q4: n_income_yoy=22.5, forecast_mid=(10+20)/2=15 → surprise=7.5
        df_q4 = lookup["20240121"]
        row = df_q4[df_q4["ts_code"] == "000001.SZ"].iloc[0]
        assert row["express_surprise"] == pytest.approx(7.5, abs=0.01)

    def test_surprise_nan_without_forecast(self, mock_express_data, trading_dates_express):
        """无预告数据时 surprise 应为 NaN"""
        lookup = build_express_lookup_by_date(mock_express_data, trading_dates_express)
        df = lookup["20231029"]
        row = df[df["ts_code"] == "000001.SZ"].iloc[0]
        assert np.isnan(row["express_surprise"])

    def test_empty_input(self, trading_dates_express):
        """空数据返回空字典"""
        lookup = build_express_lookup_by_date(pd.DataFrame(), trading_dates_express)
        assert lookup == {}

    def test_none_input(self, trading_dates_express):
        """None 输入返回空字典"""
        lookup = build_express_lookup_by_date(None, trading_dates_express)
        assert lookup == {}

    def test_date_with_dashes(self, trading_dates_express):
        """日期带横线也能正确处理"""
        df = pd.DataFrame([{
            "ts_code": "000001.SZ", "ann_date": "2023-10-28",
            "end_date": "2023-09-30",
            "revenue_yoy": 15.5, "n_income_yoy": 20.3, "roe": 12.0,
        }])
        lookup = build_express_lookup_by_date(df, trading_dates_express)
        assert "20231029" in lookup

    def test_output_columns(self, mock_express_data, trading_dates_express):
        """验证输出包含所有必要列"""
        lookup = build_express_lookup_by_date(mock_express_data, trading_dates_express)
        df = lookup["20231031"]
        for col in ["ts_code", "express_revenue_yoy", "express_profit_yoy",
                     "express_roe", "express_surprise"]:
            assert col in df.columns
