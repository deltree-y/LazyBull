"""龙虎榜空响应防假空守卫测试

覆盖:
- is_recent_date_str 近期窗口判断
- _try_ensure_historical_top_list 对近期空响应不落盘（延迟重试）,
  对历史空响应落盘 0 行 6 列占位（schema 与下载脚本一致）
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from src.lazybull.common.date_utils import is_recent_date_str
from src.lazybull.data import Storage, TushareClient
from src.lazybull.features.ensure.historical_assets import _try_ensure_historical_top_list

_PLACEHOLDER_COLS = {
    "trade_date",
    "ts_code",
    "net_amount",
    "net_rate",
    "amount_rate",
    "reason",
}


@pytest.fixture
def storage(tmp_path):
    return Storage(str(tmp_path))


class TestIsRecentDateStr:
    @pytest.fixture
    def fixed_now(self, monkeypatch):
        fixed = pd.Timestamp("2026-08-08 12:00:00")
        monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed))
        return fixed

    def test_recent_date_in_window(self, fixed_now):
        assert is_recent_date_str("20260806") is True  # 2 天前
        assert is_recent_date_str("20260808") is True  # 今天

    def test_old_date_out_of_window(self, fixed_now):
        assert is_recent_date_str("20260801") is False  # 7 天前
        assert is_recent_date_str("20240101") is False

    def test_invalid_date(self):
        assert is_recent_date_str("") is False
        assert is_recent_date_str("not-a-date") is False


class TestEnsureHistoricalTopList:
    def test_recent_empty_response_not_persisted(self, monkeypatch, storage):
        """近期空响应不落盘, 下次运行重试（防假空永久缓存）。"""
        fixed = pd.Timestamp("2026-08-08 12:00:00")
        monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed))
        client = Mock(spec=TushareClient)
        client.get_top_list.return_value = pd.DataFrame()

        result = _try_ensure_historical_top_list(client, storage, ["20260806", "20260807"])

        assert result is None
        assert not storage.is_data_exists("raw", "top_list", "20260806")
        assert not storage.is_data_exists("raw", "top_list", "20260807")

    def test_old_empty_response_persisted_as_placeholder(self, monkeypatch, storage):
        """历史日期空响应落盘 0 行 6 列占位（与下载脚本 schema 一致）。"""
        fixed = pd.Timestamp("2026-08-08 12:00:00")
        monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed))
        client = Mock(spec=TushareClient)
        client.get_top_list.return_value = pd.DataFrame()

        result = _try_ensure_historical_top_list(client, storage, ["20250106", "20250107"])

        assert result is None
        for d in ["20250106", "20250107"]:
            assert storage.is_data_exists("raw", "top_list", d)
            df = storage.load_raw_by_date("top_list", d)
            assert df is not None and len(df) == 0
            assert set(df.columns) == _PLACEHOLDER_COLS

    def test_data_downloaded_and_returned(self, storage):
        """有数据时落盘并返回合并结果。"""
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

        result = _try_ensure_historical_top_list(client, storage, ["20250106"])

        assert result is not None and len(result) == 1
        assert storage.is_data_exists("raw", "top_list", "20250106")

    def test_recent_empty_placeholder_redownloaded(self, monkeypatch, storage):
        """已存在的近期空占位分区会被重新查询并覆盖为真实数据（修复已落盘假空）。"""
        fixed = pd.Timestamp("2026-08-08 12:00:00")
        monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed))
        # 先落盘一个 0 行空占位（模拟假空）
        storage.save_raw_by_date(
            pd.DataFrame(
                columns=[
                    "trade_date", "ts_code", "net_amount", "net_rate",
                    "amount_rate", "reason",
                ]
            ),
            "top_list",
            "20260806",
        )
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

        result = _try_ensure_historical_top_list(client, storage, ["20260806"])

        assert result is not None and len(result) == 1
        saved = storage.load_raw_by_date("top_list", "20260806")
        assert saved is not None and len(saved) == 1

    def test_old_empty_placeholder_not_redownloaded(self, monkeypatch, storage):
        """已存在的非近期空占位分区不会被重新查询（保持防重复下载）。"""
        fixed = pd.Timestamp("2026-08-08 12:00:00")
        monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed))
        storage.save_raw_by_date(
            pd.DataFrame(
                columns=[
                    "trade_date", "ts_code", "net_amount", "net_rate",
                    "amount_rate", "reason",
                ]
            ),
            "top_list",
            "20250106",
        )
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

        result = _try_ensure_historical_top_list(client, storage, ["20250106"])

        assert result is None
        client.get_top_list.assert_not_called()
        saved = storage.load_raw_by_date("top_list", "20250106")
        assert saved is not None and len(saved) == 0
