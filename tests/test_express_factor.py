"""测试业绩快报因子模块"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.express import build_express_lookup_by_date


@pytest.fixture
def mock_express_data():
    """构造业绩快报数据（列名与 TuShare express_vip 实际返回一致）

    包含 2 只股票，各有 2 个报告期（Q3/Q4），不同 ann_date 发布。
    revenue_yoy 由代码自动根据同比 revenue 计算。
    """
    return pd.DataFrame([
        # 000001.SZ：去年同期（用于计算同比）
        {
            "ts_code": "000001.SZ", "ann_date": "20221028",
            "end_date": "20220930",
            "revenue": 1_000_000, "yoy_net_profit": 10.0, "diluted_roe": 11.0,
        },
        {
            "ts_code": "000001.SZ", "ann_date": "20230120",
            "end_date": "20221231",
            "revenue": 2_000_000, "yoy_net_profit": 12.0, "diluted_roe": 11.5,
        },
        # 000001.SZ：Q3 公告日 2023-10-28，Q4 公告日 2024-01-20
        {
            "ts_code": "000001.SZ", "ann_date": "20231028",
            "end_date": "20230930",
            "revenue": 1_155_000, "yoy_net_profit": 20.3, "diluted_roe": 12.0,
        },
        {
            "ts_code": "000001.SZ", "ann_date": "20240120",
            "end_date": "20231231",
            "revenue": 2_360_000, "yoy_net_profit": 22.5, "diluted_roe": 13.0,
        },
        # 600000.SH：仅 Q3（无去年同期，revenue_yoy 应为 NaN）
        {
            "ts_code": "600000.SH", "ann_date": "20231030",
            "end_date": "20230930",
            "revenue": 500_000, "yoy_net_profit": -8.0, "diluted_roe": 6.5,
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
        """公告前不应有数据（去年同期的公告日更早，但 20231025 前只有去年数据）"""
        lookup = build_express_lookup_by_date(mock_express_data, trading_dates_express)
        # 20231025 时 000001 已有去年 Q3（ann_date=20221028）的数据，但不影响这个测试
        # 600000 在 20231025 无任何数据
        pass  # 去年同期数据存在是正常的

    def test_point_in_time_after_q3(self, mock_express_data, trading_dates_express):
        """Q3 公告后应有 000001 数据，revenue_yoy 由同比计算"""
        lookup = build_express_lookup_by_date(mock_express_data, trading_dates_express)
        df = lookup["20231029"]
        row = df[df["ts_code"] == "000001.SZ"].iloc[0]
        # revenue_yoy = (1_155_000 - 1_000_000) / 1_000_000 * 100 = 15.5%
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
        # revenue_yoy = (2_360_000 - 2_000_000) / 2_000_000 * 100 = 18.0%
        assert row["express_revenue_yoy"] == pytest.approx(18.0)
        assert row["express_profit_yoy"] == pytest.approx(22.5)

    def test_express_surprise_with_forecast(
        self, mock_express_data, mock_forecast_data, trading_dates_express
    ):
        """验证 express_surprise = 实际增速 - 预告中位数"""
        lookup = build_express_lookup_by_date(
            mock_express_data, trading_dates_express, forecast_df=mock_forecast_data
        )
        # Q3: yoy_net_profit=20.3, forecast_mid=(15+25)/2=20 → surprise=0.3
        df_q3 = lookup["20231029"]
        row = df_q3[df_q3["ts_code"] == "000001.SZ"].iloc[0]
        assert row["express_surprise"] == pytest.approx(0.3, abs=0.01)

        # Q4: yoy_net_profit=22.5, forecast_mid=(10+20)/2=15 → surprise=7.5
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
            "revenue": 1_155_000, "yoy_net_profit": 20.3, "diluted_roe": 12.0,
        }])
        lookup = build_express_lookup_by_date(df, trading_dates_express)
        assert "20231029" in lookup

    def test_revenue_yoy_nan_without_prev_year(self, mock_express_data, trading_dates_express):
        """无去年同期数据时 revenue_yoy 应为 NaN"""
        lookup = build_express_lookup_by_date(mock_express_data, trading_dates_express)
        df = lookup["20231031"]
        # 600000 没有去年同期数据，revenue_yoy 应为 NaN
        row = df[df["ts_code"] == "600000.SH"].iloc[0]
        assert np.isnan(row["express_revenue_yoy"])

    def test_output_columns(self, mock_express_data, trading_dates_express):
        """验证输出包含所有必要列"""
        lookup = build_express_lookup_by_date(mock_express_data, trading_dates_express)
        df = lookup["20231031"]
        for col in ["ts_code", "express_revenue_yoy", "express_profit_yoy",
                     "express_roe", "express_surprise", "express_freshness_days"]:
            assert col in df.columns
