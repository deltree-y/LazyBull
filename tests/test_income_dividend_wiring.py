# -*- coding: utf-8 -*-
"""利润表归母净利润到分红支付率的数据链路测试。"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from scripts.raw_download.income import download_income
from src.lazybull.data import DataLoader, Storage, TushareClient
from src.lazybull.data.financial_statement_versions import (
    INCOME_VERSION_DEDUP_COLS,
    deduplicate_prefer_latest_update_flag,
)
from src.lazybull.features.ensure.bulk import _API_PAGE_LIMITS
from src.lazybull.features.pipeline import build_features_data


def test_income_client_default_fields_include_attributable_profit():
    client = object.__new__(TushareClient)

    with patch.object(client, "query", return_value=pd.DataFrame()) as query_mock:
        client.get_income_by_period("20231231")

    kwargs = query_mock.call_args.kwargs
    assert query_mock.call_args.args == ("income_vip",)
    assert kwargs["period"] == "20231231"
    assert "n_income_attr_p" in kwargs["fields"].split(",")
    assert "f_ann_date" in kwargs["fields"].split(",")
    assert "update_flag" in kwargs["fields"].split(",")


def test_load_income_reads_quarter_partition_and_normalizes_dates(tmp_path):
    storage = Storage(str(tmp_path))
    storage.save_raw_by_date(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["2024-04-30"],
                "f_ann_date": ["2024-05-06"],
                "end_date": ["2023-12-31"],
                "n_income_attr_p": [100_000_000.0],
            }
        ),
        "income",
        "2023-12-31",
    )

    result = DataLoader(storage).load_income("20240101", "20241231")

    assert result is not None
    assert result.loc[0, "ann_date"] == "20240430"
    assert result.loc[0, "f_ann_date"] == "20240506"
    assert result.loc[0, "end_date"] == "20231231"


def test_load_income_default_lookback_reads_exactly_six_years(tmp_path):
    """loader 接收目标区间后只在内部展开一次六年预热。"""
    storage = Storage(str(tmp_path))
    storage.save_raw_by_date(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20180430"],
                "f_ann_date": ["20180506"],
                "end_date": ["20171231"],
                "n_income_attr_p": [100_000_000.0],
            }
        ),
        "income",
        "2018-03-31",
    )

    with patch.object(
        storage,
        "load_raw_by_date_range",
        wraps=storage.load_raw_by_date_range,
    ) as load_range:
        DataLoader(storage).load_income("20240102", "20241231")

    assert load_range.call_args.args[:3] == (
        "income",
        "2018-01-01",
        "2024-12-31",
    )


def test_income_download_uses_consolidated_statement_and_safe_pagination():
    class _StorageStub:
        def list_partitions(self, layer, name):
            return []

        def load_sync_watermark(self, name):
            return None

        def save_sync_watermark(self, name, value):
            self.saved_watermark = (name, value)

    captured = {}

    def _fake_download_by_period(*args, **kwargs):
        captured.update(kwargs)
        return True

    storage = _StorageStub()
    with patch(
        "scripts.raw_download.income.download_by_period",
        side_effect=_fake_download_by_period,
    ):
        download_income(object(), storage, "20230101", "20241231")

    assert _API_PAGE_LIMITS["income_vip"] == 5000
    assert captured["api_name"] == "income_vip"
    assert captured["page_limit"] == 5000
    assert captured["query_kwargs"] == {"report_type": "1"}
    assert captured["dedup_cols"] == ["ts_code", "end_date", "f_ann_date"]


def test_income_version_key_falls_back_to_announcement_date_before_dedup():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231"],
            "ann_date": ["20240401", "20240430"],
            "f_ann_date": [None, ""],
            "n_income_attr_p": [100.0, 120.0],
        }
    )

    result = deduplicate_prefer_latest_update_flag(raw, INCOME_VERSION_DEDUP_COLS)

    assert len(result) == 2
    assert set(result["f_ann_date"]) == {"20240401", "20240430"}


def test_dividend_pipeline_delegates_income_lookback_to_loader(monkeypatch):
    """批量调用方只传目标区间，六年回看由 income loader 单点控制。"""
    trade_date = "20250102"

    class _Loader:
        def __init__(self):
            self.income_calls = []

        def load_clean_trade_cal(self):
            return pd.DataFrame({"cal_date": [trade_date], "is_open": [1]})

        def load_clean_stock_basic(self):
            return pd.DataFrame({"ts_code": ["000001.SZ"], "list_date": ["19910403"]})

        def get_trading_dates(self, start_date, end_date):
            return [trade_date]

        def load_clean_daily(self, start_date=None, end_date=None):
            return pd.DataFrame(
                {
                    "trade_date": [trade_date],
                    "ts_code": ["000001.SZ"],
                    "close": [10.0],
                    "close_adj": [10.0],
                }
            )

        def load_clean_daily_basic(self, start_date=None, end_date=None):
            return None

        def load_clean_moneyflow(self, start_date=None, end_date=None):
            return None

        def load_dividend(self):
            return pd.DataFrame({"ts_code": ["000001.SZ"]})

        def load_income(self, start_date=None, end_date=None):
            self.income_calls.append((start_date, end_date))
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "end_date": ["20231231"],
                    "f_ann_date": ["20240430"],
                    "n_income_attr_p": [100_000_000.0],
                }
            )

    monkeypatch.setattr(
        "src.lazybull.factors.dividend.build_dividend_lookup_by_date",
        lambda *args, **kwargs: {trade_date: pd.DataFrame()},
    )
    loader = _Loader()
    storage = Mock()
    storage.is_feature_exists.return_value = False
    builder = Mock()
    builder.build_features_for_day.return_value = pd.DataFrame()

    build_features_data(
        storage=storage,
        loader=loader,
        builder=builder,
        start_date=trade_date,
        end_date=trade_date,
        enable_dividend_policy=True,
    )

    assert loader.income_calls == [(trade_date, trade_date)]


def test_dividend_pipeline_rejects_missing_income():
    """显式启用分红政策时，缺失 income 不得静默写出全空支付率。"""
    trade_date = "20250102"
    loader = Mock()
    loader.load_clean_trade_cal.return_value = pd.DataFrame(
        {"cal_date": [trade_date], "is_open": [1]}
    )
    loader.load_clean_stock_basic.return_value = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "list_date": ["19910403"]}
    )
    loader.get_trading_dates.return_value = [trade_date]
    loader.load_clean_daily.return_value = pd.DataFrame(
        {
            "trade_date": [trade_date],
            "ts_code": ["000001.SZ"],
            "close": [10.0],
            "close_adj": [10.0],
        }
    )
    loader.load_clean_daily_basic.return_value = None
    loader.load_clean_moneyflow.return_value = None
    loader.load_dividend.return_value = pd.DataFrame({"ts_code": ["000001.SZ"]})
    loader.load_income.return_value = None
    storage = Mock()
    storage.is_feature_exists.return_value = False

    with pytest.raises(ValueError, match="raw/income"):
        build_features_data(
            storage=storage,
            loader=loader,
            builder=Mock(),
            start_date=trade_date,
            end_date=trade_date,
            enable_dividend_policy=True,
        )


def test_dividend_pipeline_rejects_income_without_valid_annual_report():
    """非空 income 若没有有效年报公告，也不得通过支付率门禁。"""
    from src.lazybull.factors.dividend import validate_income_for_dividend_payout

    income = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20231231"],
            "f_ann_date": [None],
            "n_income_attr_p": [100_000_000.0],
        }
    )

    with pytest.raises(ValueError, match="有效合并年报"):
        validate_income_for_dividend_payout(income)
