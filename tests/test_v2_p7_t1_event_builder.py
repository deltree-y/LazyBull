# -*- coding: utf-8 -*-
"""P7 事件构建器（T1/T2 共用）的合成数据回归。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.v2_p7 import event_builder  # noqa: E402


def _write(raw_root: Path, name: str, frame: pd.DataFrame) -> None:
    path = raw_root / name / "2020-12-31.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """返回 `data/raw` 根（事件构建器的输入根）。"""
    raw_root = tmp_path / "data" / "raw"
    forecast = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "ann_date": ["20200430", "20200430", "20200430", "", "20200430"],
            "end_date": ["20200331"] * 5,
            "type": ["预增", "预增", "预减", "预增", "不确定"],
            "p_change_min": [50.0, 50.0, -50.0, 10.0, 0.0],
            "p_change_max": [80.0, 80.0, -30.0, 20.0, 0.0],
        }
    )
    express = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000005.SZ"],
            "ann_date": ["20200430", "20200430", "20200501"],
            "end_date": ["20200331"] * 3,
            "yoy_net_profit": [70.0, 10.0, 30.0],
        }
    )
    fina = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000002.SZ"],
            "ann_date": ["20200430", "20200430", "20200831", "20200430"],
            "end_date": ["20200331", "20200331", "20200630", "20200331"],
            "or_yoy": [10.0, 12.0, 5.0, -1.0],
            "netprofit_yoy": [20.0, 25.0, 6.0, 5.0],
            "update_flag": ["0", "1", "1", "1"],
        }
    )
    _write(raw_root, "forecast", forecast)
    _write(raw_root, "express", express)
    _write(raw_root, "fina_indicator", fina)
    return raw_root


def test_forecast_positive_predicate_and_dedup(data_root: Path) -> None:
    events, stats = event_builder.build_events(data_root)
    forecast = events["forecast"]
    assert forecast["ts_code"].tolist() == ["000001.SZ"]
    assert forecast["predicate_value"].tolist() == [1.0]
    assert forecast["revision_key"].tolist() == ["000001.SZ|20200331|20200430"]
    assert stats["sources"]["forecast"]["invalid_date_rows"] == 1
    assert stats["sources"]["forecast"]["unmapped_type_rows"] == 0


def test_express_surprise_uses_point_in_time_forecast(data_root: Path) -> None:
    events, _ = event_builder.build_events(data_root)
    express = events["express"].sort_values("ts_code").reset_index(drop=True)
    assert express["ts_code"].tolist() == ["000001.SZ", "000002.SZ"]
    assert express["predicate_value"].tolist() == pytest.approx([5.0, 50.0])


def test_fina_keeps_all_revisions_and_prefers_updated_same_day(data_root: Path) -> None:
    events, _ = event_builder.build_events(data_root)
    fina = events["fina_report"].sort_values("knowledge_date").reset_index(drop=True)
    assert len(fina) == 2
    assert fina["event_date"].tolist() == ["20200331", "20200630"]
    assert fina["predicate_value"].tolist() == [25.0, 6.0]
    assert fina["revision_key"].tolist() == [
        "000001.SZ|20200331|20200430|1",
        "000001.SZ|20200630|20200831|1",
    ]


def test_cutoff_filters_knowledge_date(data_root: Path) -> None:
    events, _ = event_builder.build_events(data_root, cutoff="20200601")
    assert events["fina_report"]["knowledge_date"].tolist() == ["20200430"]


def test_announcement_index_keeps_non_event_records(data_root: Path) -> None:
    index = event_builder.build_announcement_index(data_root)
    codes = set(index["ts_code"])
    assert {"000002.SZ", "000004.SZ", "000005.SZ"} <= codes
    assert "000003.SZ" not in codes  # 公告日非法，无法归属任何交易日
    assert index.duplicated(subset=["ts_code", "knowledge_date"]).sum() == 0


def test_build_events_is_deterministic(data_root: Path) -> None:
    first, _ = event_builder.build_events(data_root)
    second, _ = event_builder.build_events(data_root)
    for name in event_builder.EVENT_SOURCES:
        pd.testing.assert_frame_equal(first[name], second[name])
    for name in event_builder.EVENT_SOURCES:
        assert first[name]["fingerprint"].str.len().eq(64).all()


def test_fina_watermark_scan(data_root: Path) -> None:
    payload = event_builder.scan_fina_watermark(data_root)
    assert payload["partition_count"] == 1
    assert payload["rows"] == 4
    assert payload["max_ann_date"] == "20200831"
    assert payload["max_end_date"] == "20200630"
    assert payload["invalid_ann_date_rows"] == 0


def test_event_builder_identity_is_stable() -> None:
    identity = event_builder.event_builder_identity()
    assert identity.sha256 == event_builder.event_builder_identity().sha256
    assert identity.version == event_builder.BUILDER_VERSION
    assert len(identity.sha256) == 64
