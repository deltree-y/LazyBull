# -*- coding: utf-8 -*-
"""特征流水线缓存预筛顺序测试。"""

from unittest.mock import Mock, patch

import pandas as pd

from src.lazybull.features.pipeline import build_features_data


def test_all_cached_dates_return_before_loading_daily_data():
    """所有目标分区命中时，不应加载 clean 日线或构建任何因子 lookup。"""
    trade_date = "20250102"

    class _Loader:
        def load_clean_trade_cal(self):
            return pd.DataFrame({"cal_date": [trade_date], "is_open": [1]})

        def load_clean_stock_basic(self):
            return pd.DataFrame({"ts_code": ["000001.SZ"]})

        def get_trading_dates(self, start_date, end_date):
            return [trade_date]

        def load_clean_daily(self, *args, **kwargs):
            raise AssertionError("缓存预筛应先于 clean 日线加载")

    storage = Mock()
    storage.is_feature_exists.return_value = True
    with patch("src.lazybull.features.pipeline._check_features_schema", return_value=True):
        build_features_data(
            storage=storage,
            loader=_Loader(),
            builder=Mock(),
            start_date=trade_date,
            end_date=trade_date,
        )
