# -*- coding: utf-8 -*-
"""公告型因子增量补齐链路审计修复回归测试。

覆盖 2026-08-09 审计修复：
- 高危1: 门控从"记录数量充足"改为"最新公告日是否覆盖目标交易日"（_has_announcement_gap）
- 高危2: fina_indicator 增量补齐按季度分区写入（而非写单文件被遮蔽）
- 中危: 季度窗口外股票保留"旧值 + 大 freshness"（_load_pre_window_latest_rows）
- 低危1: 同日多公告稳定排序（mergesort），PIT 取最新报告期
- 低危2: 同 (ts_code,end_date,ann_date) 多次修订优先保留 update_flag 修正版
"""

import tempfile
from unittest.mock import Mock

import pandas as pd
import pytest

from src.lazybull.data import DataLoader, Storage, TushareClient
from src.lazybull.factors.announcement_utils import build_latest_announcement_lookup_by_date
from src.lazybull.features.ensure.downloads import _try_download_fina_indicator
from src.lazybull.features.ensure.factor_load import _has_announcement_gap
from src.lazybull.features.ensure.incremental import (
    _drop_duplicates_keep_updated,
    _incremental_catchup_by_calendar_date,
)


@pytest.fixture
def temp_storage():
    """临时存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Storage(tmpdir)


# ═══════════════════════════════════════════════════════════════
# 高危1: 门控语义 —— 最新公告日覆盖判断
# ═══════════════════════════════════════════════════════════════


class TestHasAnnouncementGap:
    def test_none_or_empty_is_gap(self, temp_storage):
        assert (
            _has_announcement_gap(temp_storage, None, "fina_indicator", "ann_date", "20260430")
            is True
        )
        assert (
            _has_announcement_gap(
                temp_storage, pd.DataFrame(), "fina_indicator", "ann_date", "20260430"
            )
            is True
        )

    def test_missing_date_col_is_gap(self, temp_storage):
        # 无日期列时无法判断覆盖, 视为缺口触发下载
        df = pd.DataFrame({"ts_code": ["000001.SZ"]})
        assert (
            _has_announcement_gap(temp_storage, df, "fina_indicator", "ann_date", "20260430")
            is True
        )

    def test_latest_ann_date_below_target_is_gap(self, temp_storage):
        df = pd.DataFrame({"ann_date": ["20260401"]})
        assert (
            _has_announcement_gap(temp_storage, df, "fina_indicator", "ann_date", "20260430")
            is True
        )

    def test_latest_ann_date_covering_target_is_not_gap(self, temp_storage):
        df = pd.DataFrame({"ann_date": ["20260430"]})
        assert (
            _has_announcement_gap(temp_storage, df, "fina_indicator", "ann_date", "20260430")
            is False
        )
        df2 = pd.DataFrame({"ann_date": ["20260501"]})
        assert (
            _has_announcement_gap(temp_storage, df2, "fina_indicator", "ann_date", "20260430")
            is False
        )

    def test_report_date_col_supported(self, temp_storage):
        df = pd.DataFrame({"report_date": ["20260401"]})
        assert (
            _has_announcement_gap(temp_storage, df, "report_rc", "report_date", "20260430")
            is True
        )
        df2 = pd.DataFrame({"report_date": ["20260430"]})
        assert (
            _has_announcement_gap(temp_storage, df2, "report_rc", "report_date", "20260430")
            is False
        )

    def test_watermark_covers_gap_only_when_data_present(self, temp_storage):
        """数据非空时水位可覆盖空日；数据完全缺失时不能仅凭水位跳过（需重建）。"""
        temp_storage.save_sync_watermark("fina_indicator", "20260430")
        # 数据缺失（None/空）：即使水位高也判定缺口（parquet 可能被删/损坏）
        assert (
            _has_announcement_gap(temp_storage, None, "fina_indicator", "ann_date", "20260430")
            is True
        )
        assert (
            _has_announcement_gap(
                temp_storage, pd.DataFrame(), "fina_indicator", "ann_date", "20260430"
            )
            is True
        )
        # 数据存在但最新公告日落后于目标：水位兜底视为已覆盖（空日不再重复下载）
        df = pd.DataFrame({"ann_date": ["20260420"]})
        assert (
            _has_announcement_gap(temp_storage, df, "fina_indicator", "ann_date", "20260430")
            is False
        )
        assert (
            _has_announcement_gap(temp_storage, df, "fina_indicator", "ann_date", "20260501")
            is True
        )

    def test_watermark_below_target_still_gap(self, temp_storage):
        temp_storage.save_sync_watermark("fina_indicator", "20260420")
        df = pd.DataFrame({"ann_date": ["20260420"]})
        assert (
            _has_announcement_gap(temp_storage, df, "fina_indicator", "ann_date", "20260430")
            is True
        )

    def test_watermark_not_overridden_by_latest_data_past_failed_day(self, temp_storage):
        """反例：watermark=11（连续前缀）、数据已有 13 的公告、target=13 时仍为缺口。"""
        temp_storage.save_sync_watermark("fina_indicator", "20260411")
        df = pd.DataFrame({"ann_date": ["20260413"]})
        # 12 可能失败/缺口：数据最新公告日 13 不得越过水位 11 判定已覆盖
        assert (
            _has_announcement_gap(temp_storage, df, "fina_indicator", "ann_date", "20260413")
            is True
        )
        # 无水位时，才以数据最新公告日作为前缀
        fresh = Storage(tempfile.mkdtemp())
        assert (
            _has_announcement_gap(fresh, df, "fina_indicator", "ann_date", "20260413")
            is False
        )
        assert (
            _has_announcement_gap(fresh, df, "fina_indicator", "ann_date", "20260414")
            is True
        )


class TestSyncWatermarkAdvance:
    def _existing(self):
        return pd.DataFrame(
            [{"ts_code": "000001.SZ", "ann_date": "20260410", "end_date": "20260331"}]
        )

    def test_catchup_advances_watermark_through_empty_days(self, temp_storage):
        """无公告的空白日也会推进同步水位，避免下次重复查询。"""
        existing = self._existing()
        temp_storage.save_raw_by_date(existing, "fina_indicator", "20260331")

        queried = []

        def _fetch(ann_date):
            queried.append(ann_date)
            return pd.DataFrame()  # 区间内无任何公告

        _incremental_catchup_by_calendar_date(
            storage=temp_storage,
            dataset_name="fina_indicator",
            existing_df=existing,
            trade_date="20260413",
            date_col="ann_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
            fetch_by_date=_fetch,
        )

        assert queried == ["20260411", "20260412", "20260413"]
        assert temp_storage.load_sync_watermark("fina_indicator") == "20260413"

    def test_catchup_stops_at_first_failure_and_advances_prefix(self, temp_storage):
        """遇到首个失败立即停止，水位只推进到最后一个成功日；失败日不被跳过。"""
        existing = self._existing()
        temp_storage.save_raw_by_date(existing, "fina_indicator", "20260331")

        queried = []

        def _fetch(ann_date):
            queried.append(ann_date)
            if ann_date == "20260412":
                raise RuntimeError("网络抖动")
            return pd.DataFrame()

        _incremental_catchup_by_calendar_date(
            storage=temp_storage,
            dataset_name="fina_indicator",
            existing_df=existing,
            trade_date="20260413",
            date_col="ann_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
            fetch_by_date=_fetch,
        )

        # 11 成功（空）、12 失败即停，13 不查询
        assert queried == ["20260411", "20260412"]
        # 水位推进到最后一个成功日 11；12 失败，下次从 12 重试
        assert temp_storage.load_sync_watermark("fina_indicator") == "20260411"

    def test_failed_day_not_skipped_by_later_success(self, temp_storage):
        """故障注入：中间日失败、后续日成功落盘时，失败日不被 latest_data 跨过。"""
        existing = self._existing()
        temp_storage.save_raw_by_date(existing, "fina_indicator", "20260331")

        def _fetch(ann_date):
            if ann_date == "20260411":
                raise RuntimeError("网络抖动")
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000002.SZ",
                        "ann_date": ann_date,
                        "end_date": "20260331",
                    }
                ]
            )

        # 第一次：11 失败即停（12 不查），水位不推进
        _incremental_catchup_by_calendar_date(
            storage=temp_storage,
            dataset_name="fina_indicator",
            existing_df=existing,
            trade_date="20260413",
            date_col="ann_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
            fetch_by_date=_fetch,
        )
        assert temp_storage.load_sync_watermark("fina_indicator") is None

        # 第二次：仍从 11 开始重试（失败日不被跳过）
        retry_calls = []

        def _fetch_retry(ann_date):
            retry_calls.append(ann_date)
            return pd.DataFrame()

        _incremental_catchup_by_calendar_date(
            storage=temp_storage,
            dataset_name="fina_indicator",
            existing_df=existing,
            trade_date="20260413",
            date_col="ann_date",
            dedup_cols=["ts_code", "end_date", "ann_date"],
            fetch_by_date=_fetch_retry,
        )
        assert "20260411" in retry_calls

    def test_watermark_not_advanced_when_save_fails(self, temp_storage, monkeypatch):
        """新数据落盘失败时水位不推进（避免水位已提交但数据缺失）。"""
        existing = self._existing()
        temp_storage.save_raw_by_date(existing, "fina_indicator", "20260331")

        import src.lazybull.features.ensure.incremental as ensure_incremental

        def _boom(*args, **kwargs):
            raise OSError("磁盘写入失败")

        monkeypatch.setattr(ensure_incremental, "_append_and_save_raw", _boom)

        def _fetch(ann_date):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000002.SZ",
                        "ann_date": ann_date,
                        "end_date": "20260331",
                    }
                ]
            )

        with pytest.raises(OSError):
            _incremental_catchup_by_calendar_date(
                storage=temp_storage,
                dataset_name="fina_indicator",
                existing_df=existing,
                trade_date="20260411",
                date_col="ann_date",
                dedup_cols=["ts_code", "end_date", "ann_date"],
                fetch_by_date=_fetch,
            )

        # 落盘抛异常 → 水位未提交，下次从原水位之后重查
        assert temp_storage.load_sync_watermark("fina_indicator") is None


# ═══════════════════════════════════════════════════════════════
# 高危2: fina_indicator 增量补齐写季度分区（而非单文件被遮蔽）
# ═══════════════════════════════════════════════════════════════


def test_try_download_fina_indicator_incremental_writes_quarter_partition(
    monkeypatch, temp_storage
):
    """增量补齐的新公告应路由写入对应季度分区，加载时分区优先可读。"""
    import src.lazybull.features.ensure.downloads as ensure_downloads

    monkeypatch.setattr(ensure_downloads, "_MIN_FINA_RECORDS", 1)

    existing = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "ann_date": "20260410",
                "end_date": "20260331",
                "update_flag": None,
                "roe_waa": 12.0,
                "q_gr_yoy": 5.0,
                "q_ocf_to_sales": 0.2,
                "int_to_talcap": 3.0,
                "inv_turn": 1.1,
            }
        ]
    )
    temp_storage.save_raw_by_date(existing, "fina_indicator", "20260331")

    new_row = pd.DataFrame(
        [
            {
                "ts_code": "000002.SZ",
                "ann_date": "20260411",
                "end_date": "20260331",
                "update_flag": None,
                "roe_waa": 8.0,
                "q_gr_yoy": 3.0,
                "q_ocf_to_sales": 0.1,
                "int_to_talcap": 2.0,
                "inv_turn": 0.9,
            }
        ]
    )

    def _fetch_by_date(ann_date):
        if ann_date == "20260411":
            return new_row
        return pd.DataFrame()

    client = Mock(spec=TushareClient)
    client.get_fina_indicator_by_date.side_effect = _fetch_by_date

    result = _try_download_fina_indicator(client, temp_storage, "20260412")

    assert result is not None
    loader = DataLoader(temp_storage)
    loaded = loader.load_fina_indicator(start_date="20260412", end_date="20260412")
    assert loaded is not None
    assert set(loaded["ts_code"].tolist()) == {"000001.SZ", "000002.SZ"}
    # 确认新公告落在季度分区内（分区优先读取），而非被遮蔽的单文件
    partitions = temp_storage.list_partitions("raw", "fina_indicator")
    assert partitions == ["2026-03-31"]
    # 增量区间无失败 → 同步水位推进到目标日（避免下次重复查询）
    assert temp_storage.load_sync_watermark("fina_indicator") == "20260412"


# ═══════════════════════════════════════════════════════════════
# 中危: 窗口外股票保留"旧值 + 大 freshness"，而非硬缺失 NaN
# ═══════════════════════════════════════════════════════════════


class TestLoadPreWindowLatestRows:
    def test_window_outside_stock_keeps_latest_pre_window_announcement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(tmpdir)
            # 窗口内分区（2025-03-31），覆盖 000001.SZ
            storage.save_raw_by_date(
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ"],
                        "ann_date": ["20250430"],
                        "end_date": ["20250331"],
                    }
                ),
                "fina_indicator",
                "20250331",
            )
            # 窗口前最近分区（2023-03-31），含 000001.SZ（窗口内已覆盖）与 000002.SZ
            storage.save_raw_by_date(
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ", "000002.SZ"],
                        "ann_date": ["20230428", "20230429"],
                        "end_date": ["20230331", "20230331"],
                    }
                ),
                "fina_indicator",
                "20230331",
            )
            # 窗口前更早分区（2022-03-31），只有 000002.SZ
            storage.save_raw_by_date(
                pd.DataFrame(
                    {
                        "ts_code": ["000002.SZ"],
                        "ann_date": ["20220429"],
                        "end_date": ["20220331"],
                    }
                ),
                "fina_indicator",
                "20220331",
            )

            loader = DataLoader(storage=storage)
            # 窗口 [2024-01-01, 2025-04-30]
            df = loader.load_fina_indicator(
                start_date="20250430", end_date="20250430"
            )

            assert df is not None
            assert set(df["ts_code"].tolist()) == {"000001.SZ", "000002.SZ"}
            # 000002.SZ 在窗口外，应保留窗口前最近一条公告（end_date=20230331）
            row_000002 = df[df["ts_code"] == "000002.SZ"].iloc[0]
            assert row_000002["end_date"] == "20230331"

    def test_no_pre_window_partitions_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(tmpdir)
            storage.save_raw_by_date(
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ"],
                        "ann_date": ["20250430"],
                        "end_date": ["20250331"],
                    }
                ),
                "fina_indicator",
                "20250331",
            )
            loader = DataLoader(storage=storage)
            df = loader.load_fina_indicator(
                start_date="20250430", end_date="20250430"
            )
            assert df is not None
            assert df["ts_code"].tolist() == ["000001.SZ"]

    def test_window_empty_still_backfills_from_pre_window(self):
        """目标窗口内无任何分区时，仍从窗口前分区补充股票最新公告，而非直接返回 None。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Storage(tmpdir)
            # 只有窗口前分区（2023-03-31），窗口 [2024-01-01, 2025-04-30] 内无分区
            storage.save_raw_by_date(
                pd.DataFrame(
                    {
                        "ts_code": ["000001.SZ", "000002.SZ"],
                        "ann_date": ["20230428", "20230429"],
                        "end_date": ["20230331", "20230331"],
                    }
                ),
                "fina_indicator",
                "20230331",
            )
            loader = DataLoader(storage=storage)
            df = loader.load_fina_indicator(
                start_date="20250430", end_date="20250430"
            )
            # 窗口内无数据时，返回窗口前最近分区（旧值 + 大 freshness），而非 None
            assert df is not None
            assert set(df["ts_code"].tolist()) == {"000001.SZ", "000002.SZ"}
            assert set(df["end_date"].tolist()) == {"20230331"}


# ═══════════════════════════════════════════════════════════════
# 低危1: 同日多公告稳定排序 —— PIT 取最新报告期
# ═══════════════════════════════════════════════════════════════


class TestSameDayMultiAnnouncementStableSort:
    def test_pit_picks_latest_end_date_on_same_ann_date(self):
        """同一天披露"旧报告期更正 + 新季报"时，稳定排序保证选中新报告期。"""
        # 模拟 fundamental.py 上游已按 [ts_code, end_date, ann_date] 排序
        factor_df = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20260430",
                    "end_date": "20251231",
                    "val": 10.0,  # 旧年报更正
                },
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20260430",
                    "end_date": "20260331",
                    "val": 20.0,  # 新季报
                },
            ]
        )

        lookup = build_latest_announcement_lookup_by_date(
            factor_df,
            ["20260501"],
            value_cols=["val", "end_date"],
        )

        row = lookup["20260501"].iloc[0]
        assert row["val"] == 20.0
        assert row["end_date"] == "20260331"


# ═══════════════════════════════════════════════════════════════
# 低危2: 同 (ts_code,end_date,ann_date) 多次修订优先保留 update_flag 修正版
# ═══════════════════════════════════════════════════════════════


class TestDropDuplicatesKeepUpdated:
    def test_keeps_updated_flag_record(self):
        df = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "ann_date": ["20240501", "20240501"],
                "end_date": ["20240331", "20240331"],
                "update_flag": [None, "1"],
                "val": [10.0, 20.0],
            }
        )
        result = _drop_duplicates_keep_updated(df, ["ts_code", "end_date", "ann_date"])
        assert len(result) == 1
        assert result["val"].iloc[0] == 20.0

    def test_non_one_update_flag_not_treated_as_updated(self):
        """仅显式识别 update_flag == "1" 为修正版；其他非空值（如 "0"）不视为修正。"""
        df = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "ann_date": ["20240501", "20240501"],
                "end_date": ["20240331", "20240331"],
                "update_flag": ["0", "1"],
                "val": [10.0, 20.0],
            }
        )
        result = _drop_duplicates_keep_updated(df, ["ts_code", "end_date", "ann_date"])
        assert len(result) == 1
        assert result["val"].iloc[0] == 20.0

        # 两个都非 "1" 时，回退到 keep='last'（输入顺序稳定）
        df2 = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "ann_date": ["20240501", "20240501"],
                "end_date": ["20240331", "20240331"],
                "update_flag": ["0", "0"],
                "val": [10.0, 20.0],
            }
        )
        result2 = _drop_duplicates_keep_updated(df2, ["ts_code", "end_date", "ann_date"])
        assert len(result2) == 1
        assert result2["val"].iloc[0] == 20.0

    def test_without_update_flag_falls_back_to_keep_last(self):
        df = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "ann_date": ["20240501", "20240501"],
                "end_date": ["20240331", "20240331"],
                "val": [10.0, 20.0],
            }
        )
        result = _drop_duplicates_keep_updated(df, ["ts_code", "end_date", "ann_date"])
        assert len(result) == 1
        assert result["val"].iloc[0] == 20.0

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "ann_date": ["20240501", "20240502"],
                "end_date": ["20240331", "20240331"],
                "update_flag": [None, "1"],
            }
        )
        result = _drop_duplicates_keep_updated(df, ["ts_code", "end_date", "ann_date"])
        assert len(result) == 2
