"""龙虎榜关键接线测试（二轮审计问题 2 补充）

覆盖:
- features/pipeline.py 批量构建龙虎榜必须传入含预热期的交易日历
- scripts/raw_download/alt.py download_top_list 对近期空占位分区重新查询覆盖
"""

import tempfile
from unittest.mock import Mock

import pandas as pd

from src.lazybull.data import Storage, TushareClient
from src.lazybull.features.pipeline import build_features_data

_EMPTY_TOP_LIST = pd.DataFrame(
    columns=[
        "trade_date",
        "ts_code",
        "net_amount",
        "net_rate",
        "amount_rate",
        "reason",
    ]
)


class _StubLoader:
    """build_features_data 最小 stub: 仅 enable_lhb 路径所需方法。"""

    def __init__(self, cal_dates: list):
        self._cal_dates = cal_dates
        self._cal = pd.DataFrame({"cal_date": cal_dates, "is_open": [1] * len(cal_dates)})

    def load_clean_trade_cal(self):
        return self._cal.copy()

    def load_clean_stock_basic(self):
        return pd.DataFrame({"ts_code": [], "name": []})

    def get_trading_dates(self, start_date, end_date):
        # 模拟真实行为: 按日期范围筛选交易日
        start = str(start_date).replace("-", "")
        end = str(end_date).replace("-", "")
        return [d for d in self._cal_dates if start <= d <= end]

    def load_clean_daily(self, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "trade_date": self._cal_dates,
                "ts_code": ["000001.SZ"] * len(self._cal_dates),
                "close_adj": [10.0] * len(self._cal_dates),
            }
        )

    def load_clean_daily_basic(self, start_date=None, end_date=None):
        return None

    def load_clean_moneyflow(self, start_date=None, end_date=None):
        return None

    def load_top_list(self, start_date=None, end_date=None):
        return pd.DataFrame(
            {
                "trade_date": [self._cal_dates[0]],
                "ts_code": ["000001.SZ"],
                "net_amount": [1e8],
                "net_rate": [0.02],
                "amount_rate": [0.1],
                "reason": ["日涨幅偏离值达 7%"],
            }
        )


def test_build_features_lhb_passes_warmup_calendar(monkeypatch):
    """批量构建龙虎榜必须把含预热期的完整日历传给 build_lhb_lookup_by_date。"""
    all_dates = pd.date_range("20250101", periods=60, freq="B").strftime("%Y%m%d").tolist()
    loader = _StubLoader(all_dates)

    captured = {}

    def fake_build(df, trading_dates, calendar_dates=None):
        captured["trading_dates"] = list(trading_dates)
        captured["calendar_dates"] = list(calendar_dates) if calendar_dates is not None else None
        return {d: pd.DataFrame() for d in trading_dates}

    monkeypatch.setattr("src.lazybull.factors.lhb.build_lhb_lookup_by_date", fake_build)
    # 仅保留首日待构建，其余日期命中缓存；lookup 仍需完整输出区间和预热日历
    monkeypatch.setattr(
        "src.lazybull.features.pipeline._check_features_schema", lambda *a, **k: True
    )

    builder = Mock()
    builder.build_features_for_day.return_value = pd.DataFrame()
    storage = Mock()
    storage.is_feature_exists.side_effect = lambda trade_date: trade_date != start_date

    start_date = "20250121"
    end_date = "20250228"
    build_features_data(
        storage=storage,
        loader=loader,
        builder=builder,
        start_date=start_date,
        end_date=end_date,
        force=False,
        enable_lhb=True,
    )

    assert captured.get("calendar_dates") is not None
    # 日历包含早于 start_date 的预热日期
    assert captured["calendar_dates"][0] == all_dates[0]
    assert captured["calendar_dates"][0] < start_date
    # 输出交易日仍从 start_date 开始
    assert captured["trading_dates"][0] == start_date


def test_download_top_list_redownloads_recent_empty_placeholder(monkeypatch):
    """download_top_list 对近期已存在的空占位分区重新查询并覆盖。"""
    fixed = pd.Timestamp("2026-08-08 12:00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed))

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        # 已有近期空占位（模拟假空）
        storage.save_raw_by_date(_EMPTY_TOP_LIST.copy(), "top_list", "20260806")

        from scripts.raw_download.alt import download_top_list

        client = Mock(spec=TushareClient)
        data = pd.DataFrame(
            {
                "trade_date": ["20260806"],
                "ts_code": ["000001.SZ"],
                "net_amount": [1e8],
                "net_rate": [0.02],
                "amount_rate": [0.1],
                "reason": ["日涨幅偏离值达 7%"],
            }
        )
        client.get_top_list.return_value = data
        trade_cal = pd.DataFrame({"cal_date": ["20260806"], "is_open": [1]})

        download_top_list(client, storage, trade_cal, "20260801", "20260808")

        saved = storage.load_raw_by_date("top_list", "20260806")
        assert saved is not None and len(saved) == 1
        client.get_top_list.assert_called_once_with(trade_date="20260806")


def test_download_top_list_skips_old_empty_placeholder(monkeypatch):
    """download_top_list 对非近期空占位分区不重新查询（保持防重复下载）。"""
    fixed = pd.Timestamp("2026-08-08 12:00:00")
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed))

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Storage(tmpdir)
        storage.save_raw_by_date(_EMPTY_TOP_LIST.copy(), "top_list", "20250106")

        from scripts.raw_download.alt import download_top_list

        client = Mock(spec=TushareClient)
        data = pd.DataFrame(
            {
                "trade_date": ["20250106"],
                "ts_code": ["000001.SZ"],
                "net_amount": [1e8],
                "net_rate": [0.02],
                "amount_rate": [0.1],
                "reason": ["日涨幅偏离值达 7%"],
            }
        )
        client.get_top_list.return_value = data
        trade_cal = pd.DataFrame({"cal_date": ["20250106"], "is_open": [1]})

        download_top_list(client, storage, trade_cal, "20250101", "20250110")

        client.get_top_list.assert_not_called()
        saved = storage.load_raw_by_date("top_list", "20250106")
        assert saved is not None and len(saved) == 0
