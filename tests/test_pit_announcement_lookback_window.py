# -*- coding: utf-8 -*-
"""PIT 公告类因子加载窗口回归测试（质押）。

质押是季频 PIT 前向填充数据集：T 日的取值可能来自数千天前的公告
（实测 2024-07 的质押率来自 2023-06-30）。若调用方只按批次预热窗口
（start_date − 7 个月）传下界，窗口内可能一个季度分区都没有 →
质押列整列消失（各日期 schema 不一致）、pledge_high_flag/pledge_delta
被静默零填充。本测试锁定两处调用点都必须使用全量历史起点：

- 离线批量构建：scripts/build_clean_features.py -> features/pipeline.py
- 纸面交易自动补齐：features/ensure/factor_load.py

限售解禁（share_float）为年分区且按公告日 PIT 选取，不在此修复范围：
其窗口变化会改变 days_to_unlock/unlock_ratio 取值，属另一类需先决策的问题。
"""

from unittest.mock import Mock

import pandas as pd

from src.lazybull.data.loader_announcement import ANNOUNCEMENT_PIT_LOOKBACK_START
from src.lazybull.features.pipeline import build_features_data


class _StubLoader:
    """build_features_data 最小 stub：仅公告类路径所需方法。"""

    def __init__(self, cal_dates):
        self._cal_dates = cal_dates
        self._cal = pd.DataFrame({"cal_date": cal_dates, "is_open": [1] * len(cal_dates)})
        self.pledge_calls = []
        self.share_float_calls = []
        self.block_trade_calls = []

    def load_clean_trade_cal(self):
        return self._cal.copy()

    def load_clean_stock_basic(self):
        return pd.DataFrame({"ts_code": [], "name": []})

    def get_trading_dates(self, start_date, end_date):
        start = str(start_date).replace("-", "")
        end = str(end_date).replace("-", "")
        return [d for d in self._cal_dates if start <= d <= end]

    def load_clean_daily(self, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "trade_date": self._cal_dates,
                "ts_code": ["000001.SZ"] * len(self._cal_dates),
                "close": [10.0] * len(self._cal_dates),
                "close_adj": [10.0] * len(self._cal_dates),
            }
        )

    def load_clean_daily_basic(self, start_date=None, end_date=None):
        return None

    def load_clean_moneyflow(self, start_date=None, end_date=None):
        return None

    def load_pledge_stat(self, start_date=None, end_date=None):
        self.pledge_calls.append((start_date, end_date))
        return pd.DataFrame(
            {"ts_code": ["000001.SZ"], "end_date": ["20230630"], "pledge_ratio": [1.0]}
        )

    def load_share_float(self, start_date=None, end_date=None):
        self.share_float_calls.append((start_date, end_date))
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20240101"],
                "float_date": ["20240801"],
                "float_share": [1e6],
            }
        )

    def load_block_trade(self, start_date=None, end_date=None):
        self.block_trade_calls.append((start_date, end_date))
        return None


def _stub_builder(monkeypatch, captured_dates):
    """把三个公告类 lookup 构造替换为记录输出日期并返回空表的桩。"""

    def _factory(name):
        def _fake(df, trading_dates, *args, **kwargs):
            captured_dates[name] = list(trading_dates)
            return {d: pd.DataFrame() for d in trading_dates}

        return _fake

    monkeypatch.setattr(
        "src.lazybull.factors.risk.announcement_lookup.build_pledge_lookup_by_date",
        _factory("pledge"),
    )
    monkeypatch.setattr(
        "src.lazybull.factors.risk.announcement_lookup.build_share_float_lookup_by_date",
        _factory("share_float"),
    )
    monkeypatch.setattr(
        "src.lazybull.factors.risk.announcement_lookup.build_block_trade_lookup_by_date",
        _factory("block_trade"),
    )


def test_pipeline_announcement_data_uses_full_history_lookback(monkeypatch):
    """局部重建时质押/解禁必须从全量历史起点加载，上界仍受 end_date 约束。"""
    all_dates = pd.date_range("20250101", periods=60, freq="B").strftime("%Y%m%d").tolist()
    loader = _StubLoader(all_dates)
    start_date = "20250121"
    end_date = "20250228"

    captured_dates = {}
    _stub_builder(monkeypatch, captured_dates)
    monkeypatch.setattr(
        "src.lazybull.features.pipeline._check_features_schema", lambda *a, **k: True
    )

    builder = Mock()
    builder.build_features_for_day.return_value = pd.DataFrame()
    storage = Mock()
    storage.is_feature_exists.side_effect = lambda trade_date: trade_date != start_date

    build_features_data(
        storage=storage,
        loader=loader,
        builder=builder,
        start_date=start_date,
        end_date=end_date,
        force=False,
        enable_announcement_risk=True,
    )

    # 质押：下界为全量历史起点（不随批次预热窗口收缩）
    assert loader.pledge_calls, "质押数据未被加载"
    assert loader.pledge_calls[0][0] == ANNOUNCEMENT_PIT_LOOKBACK_START
    # 上界不得越界到批次之后（避免未来分区泄露）
    end_dt_str = loader.pledge_calls[0][1]
    assert end_dt_str > end_date
    assert pd.to_datetime(end_dt_str, format="%Y%m%d") <= pd.to_datetime(
        end_date, format="%Y%m%d"
    ) + pd.DateOffset(months=1)


class _NullLoader:
    """未显式定义的加载方法统一返回 None（等价于"该数据集本地不可用"）。"""

    def __init__(self):
        self.pledge_calls = []

    def load_pledge_stat(self, start_date=None, end_date=None):
        self.pledge_calls.append((start_date, end_date))
        return None

    def __getattr__(self, name):
        def _none(*args, **kwargs):
            return None

        return _none


def test_ensure_factor_load_announcement_data_uses_full_history_lookback(monkeypatch):
    """纸面交易按日补齐时，质押/解禁同样使用全量历史起点（上界为 end_date）。"""
    from src.lazybull.features.ensure import factor_load as ensure_factor_load

    # 下载/补齐函数全部置空：本测试只关注公告类数据加载入参。
    for name in list(dir(ensure_factor_load)):
        if name.startswith("_try_download_") or name.startswith("_try_ensure_"):
            monkeypatch.setattr(ensure_factor_load, name, lambda *a, **k: None)

    loader = _NullLoader()
    storage = Mock()
    storage.load_sync_watermark.return_value = None

    factor_output_dates = ["20240802"]
    ensure_factor_load._load_factor_data(
        loader=loader,
        client=Mock(),
        storage=storage,
        trade_date="20240802",
        trading_dates_str=list(factor_output_dates),
        start_date="20240722",
        end_date="20240802",
    )

    assert loader.pledge_calls == [(ANNOUNCEMENT_PIT_LOOKBACK_START, "20240802")]
