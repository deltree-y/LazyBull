"""测试基金持仓因子模块"""

import numpy as np
import pandas as pd
import pytest

from src.lazybull.factors.fund_portfolio import (
    _aggregate_fund_portfolio,
    _symbol_to_ts_code,
    build_fund_portfolio_lookup_by_date,
)


@pytest.fixture
def mock_fund_portfolio_data():
    """构造基金持仓原始数据

    3 只基金分别在 Q2 和 Q3 持仓 000001 / 600000:
    - Q2 (end_date=20230630, ann_date=20230820): 2 只基金持仓 000001, 1 只持仓 600000
    - Q3 (end_date=20230930, ann_date=20231020): 3 只基金持仓 000001, 0 只持仓 600000
    """
    return pd.DataFrame([
        # Q2: 基金 A 持仓 000001
        {
            "ts_code": "000001.OF", "symbol": "000001", "ann_date": "20230820",
            "end_date": "20230630", "stk_float_ratio": 1.5, "mkv": 5000, "amount": 500,
        },
        # Q2: 基金 B 持仓 000001
        {
            "ts_code": "000002.OF", "symbol": "000001", "ann_date": "20230825",
            "end_date": "20230630", "stk_float_ratio": 2.0, "mkv": 8000, "amount": 800,
        },
        # Q2: 基金 A 持仓 600000
        {
            "ts_code": "000001.OF", "symbol": "600000", "ann_date": "20230820",
            "end_date": "20230630", "stk_float_ratio": 0.8, "mkv": 3000, "amount": 300,
        },
        # Q3: 基金 A 持仓 000001
        {
            "ts_code": "000001.OF", "symbol": "000001", "ann_date": "20231020",
            "end_date": "20230930", "stk_float_ratio": 1.8, "mkv": 6000, "amount": 600,
        },
        # Q3: 基金 B 持仓 000001
        {
            "ts_code": "000002.OF", "symbol": "000001", "ann_date": "20231025",
            "end_date": "20230930", "stk_float_ratio": 2.5, "mkv": 9000, "amount": 900,
        },
        # Q3: 基金 C 持仓 000001
        {
            "ts_code": "000003.OF", "symbol": "000001", "ann_date": "20231028",
            "end_date": "20230930", "stk_float_ratio": 1.0, "mkv": 4000, "amount": 400,
        },
    ])


@pytest.fixture
def trading_dates_fund():
    """跨越 Q2/Q3 公告日的交易日"""
    return [
        "20230815",  # Q2 公告前
        "20230821",  # 基金 A 的 Q2 已公告
        "20230826",  # 基金 B 的 Q2 也已公告（Q2 完整）
        "20231015",  # Q3 公告前
        "20231029",  # Q3 大部分基金已公告
    ]


class TestSymbolToTsCode:
    """_symbol_to_ts_code 单元测试"""

    def test_sz_codes(self):
        assert _symbol_to_ts_code("000001") == "000001.SZ"
        assert _symbol_to_ts_code("300001") == "300001.SZ"

    def test_sh_codes(self):
        assert _symbol_to_ts_code("600000") == "600000.SH"
        assert _symbol_to_ts_code("688001") == "688001.SH"

    def test_bj_codes(self):
        assert _symbol_to_ts_code("830001") == "830001.BJ"
        assert _symbol_to_ts_code("430001") == "430001.BJ"

    def test_zero_padding(self):
        """短代码应补零"""
        assert _symbol_to_ts_code("1") == "000001.SZ"

    def test_already_has_suffix_sz(self):
        """已含 .SZ 后缀的 symbol 应直接返回"""
        assert _symbol_to_ts_code("000001.SZ") == "000001.SZ"

    def test_already_has_suffix_sh(self):
        """已含 .SH 后缀的 symbol 应直接返回"""
        assert _symbol_to_ts_code("600820.SH") == "600820.SH"

    def test_already_has_suffix_bj(self):
        """已含 .BJ 后缀的 symbol 应直接返回"""
        assert _symbol_to_ts_code("830001.BJ") == "830001.BJ"

    def test_unknown_prefix(self):
        """无法识别的前缀返回 None"""
        assert _symbol_to_ts_code("100001") is None


class TestAggregateFundPortfolio:
    """_aggregate_fund_portfolio 单元测试"""

    def test_aggregation_sum_ratio(self, mock_fund_portfolio_data):
        """验证 fund_hold_ratio 是 stk_float_ratio 的求和"""
        agg = _aggregate_fund_portfolio(mock_fund_portfolio_data)
        # Q2 000001: 1.5 + 2.0 = 3.5
        q2_001 = agg[(agg["symbol"] == "000001") & (agg["end_date"] == "20230630")]
        assert len(q2_001) == 1
        assert q2_001.iloc[0]["fund_hold_ratio"] == pytest.approx(3.5)

    def test_aggregation_count_funds(self, mock_fund_portfolio_data):
        """验证 fund_count 是持仓基金数量"""
        agg = _aggregate_fund_portfolio(mock_fund_portfolio_data)
        # Q2 000001: 2 只基金
        q2_001 = agg[(agg["symbol"] == "000001") & (agg["end_date"] == "20230630")]
        assert q2_001.iloc[0]["fund_count"] == 2

        # Q3 000001: 3 只基金
        q3_001 = agg[(agg["symbol"] == "000001") & (agg["end_date"] == "20230930")]
        assert q3_001.iloc[0]["fund_count"] == 3

    def test_ann_date_takes_max(self, mock_fund_portfolio_data):
        """ann_date 取最晚公告日"""
        agg = _aggregate_fund_portfolio(mock_fund_portfolio_data)
        # Q2 000001 公告日：max(20230820, 20230825) = 20230825
        q2_001 = agg[(agg["symbol"] == "000001") & (agg["end_date"] == "20230630")]
        assert q2_001.iloc[0]["ann_date"] == "20230825"

    def test_aggregation_with_minimal_columns(self, mock_fund_portfolio_data):
        """只保留聚合必要列时仍能正确汇总。"""
        slim = mock_fund_portfolio_data[["ts_code", "symbol", "ann_date", "end_date", "stk_float_ratio"]]
        agg = _aggregate_fund_portfolio(slim)
        q3_001 = agg[(agg["symbol"] == "000001") & (agg["end_date"] == "20230930")]
        assert len(q3_001) == 1
        assert q3_001.iloc[0]["fund_hold_ratio"] == pytest.approx(5.3)
        assert q3_001.iloc[0]["fund_count"] == 3


class TestBuildFundPortfolioLookup:
    """build_fund_portfolio_lookup_by_date 单元测试"""

    def test_point_in_time_before_any(self, mock_fund_portfolio_data, trading_dates_fund):
        """所有 Q2 公告前不应有数据"""
        lookup = build_fund_portfolio_lookup_by_date(
            mock_fund_portfolio_data, trading_dates_fund
        )
        assert "20230815" not in lookup

    def test_point_in_time_after_q2(self, mock_fund_portfolio_data, trading_dates_fund):
        """Q2 完整公告后应有数据"""
        lookup = build_fund_portfolio_lookup_by_date(
            mock_fund_portfolio_data, trading_dates_fund
        )
        assert "20230826" in lookup
        df = lookup["20230826"]
        # 000001 应有 Q2 数据
        row = df[df["ts_code"] == "000001.SZ"]
        assert len(row) == 1
        assert row.iloc[0]["fund_hold_ratio"] == pytest.approx(3.5)

    def test_q3_overwrites_q2(self, mock_fund_portfolio_data, trading_dates_fund):
        """Q3 公告后 000001 应更新为 Q3 数据"""
        lookup = build_fund_portfolio_lookup_by_date(
            mock_fund_portfolio_data, trading_dates_fund
        )
        df = lookup["20231029"]
        row = df[df["ts_code"] == "000001.SZ"].iloc[0]
        # Q3 ratio: 1.8 + 2.5 + 1.0 = 5.3
        assert row["fund_hold_ratio"] == pytest.approx(5.3)
        assert row["fund_count"] == 3

    def test_hold_ratio_chg(self, mock_fund_portfolio_data, trading_dates_fund):
        """验证季度环比变化"""
        lookup = build_fund_portfolio_lookup_by_date(
            mock_fund_portfolio_data, trading_dates_fund
        )
        df = lookup["20231029"]
        row = df[df["ts_code"] == "000001.SZ"].iloc[0]
        # Q3 ratio(5.3) - Q2 ratio(3.5) = 1.8
        assert row["fund_hold_ratio_chg"] == pytest.approx(1.8)
        # Q3 count(3) - Q2 count(2) = 1
        assert row["fund_count_chg"] == pytest.approx(1)

    def test_first_quarter_chg_is_nan(self, mock_fund_portfolio_data, trading_dates_fund):
        """第一个季度没有前值，变化应为 NaN"""
        lookup = build_fund_portfolio_lookup_by_date(
            mock_fund_portfolio_data, trading_dates_fund
        )
        df = lookup["20230826"]
        row = df[df["ts_code"] == "000001.SZ"].iloc[0]
        assert np.isnan(row["fund_hold_ratio_chg"])
        assert np.isnan(row["fund_count_chg"])

    def test_empty_input(self, trading_dates_fund):
        """空数据返回空字典"""
        lookup = build_fund_portfolio_lookup_by_date(pd.DataFrame(), trading_dates_fund)
        assert lookup == {}

    def test_none_input(self, trading_dates_fund):
        """None 输入返回空字典"""
        lookup = build_fund_portfolio_lookup_by_date(None, trading_dates_fund)
        assert lookup == {}

    def test_symbol_with_suffix(self, trading_dates_fund):
        """symbol 已含交易所后缀时（TuShare 实际格式）也能正确处理"""
        data = pd.DataFrame([
            {
                "ts_code": "000001.OF", "symbol": "000001.SZ", "ann_date": "20230820",
                "end_date": "20230630", "stk_float_ratio": 1.5, "mkv": 5000, "amount": 500,
            },
            {
                "ts_code": "000002.OF", "symbol": "600000.SH", "ann_date": "20230820",
                "end_date": "20230630", "stk_float_ratio": 2.0, "mkv": 8000, "amount": 800,
            },
        ])
        lookup = build_fund_portfolio_lookup_by_date(data, trading_dates_fund)
        df = lookup["20230826"]
        assert len(df) == 2
        assert set(df["ts_code"]) == {"000001.SZ", "600000.SH"}

    def test_output_columns(self, mock_fund_portfolio_data, trading_dates_fund):
        """验证输出包含所有必要列"""
        lookup = build_fund_portfolio_lookup_by_date(
            mock_fund_portfolio_data, trading_dates_fund
        )
        df = lookup["20230826"]
        for col in ["ts_code", "fund_hold_ratio", "fund_hold_ratio_chg",
                     "fund_count", "fund_count_chg"]:
            assert col in df.columns
