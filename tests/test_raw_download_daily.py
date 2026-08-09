"""日频 raw 批量下载契约测试。"""

from unittest.mock import Mock

import pandas as pd

from scripts.raw_download.daily import download_daily_data


def test_download_daily_adj_factor_empty_does_not_save_partial_day():
    """adj_factor 空响应时整日失败，已获取子集也不得部分落盘。"""
    trade_date = "20230103"
    trade_cal = pd.DataFrame({"cal_date": [trade_date], "is_open": [1]})
    client = Mock()
    client.get_daily.return_value = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": [trade_date], "close": [10.0]}
    )
    client.get_daily_basic.return_value = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": [trade_date], "pb": [1.0]}
    )
    client.get_adj_factor.return_value = pd.DataFrame()
    storage = Mock()
    storage.is_data_exists.return_value = False

    download_daily_data(client, storage, trade_cal, trade_date, trade_date)

    storage.save_raw_by_date.assert_not_called()
