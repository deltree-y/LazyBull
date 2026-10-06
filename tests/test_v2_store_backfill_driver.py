# -*- coding: utf-8 -*-
"""v2 P1 单元 4：回填驱动测试（keep_dates 剔除已写日 / 冷区已写判定）。

合成数据 + monkeypatch 捕获驱动（不触碰真实 data/ 与生产管线）。
"""

from datetime import date, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from scripts.v2_p1.backfill_panel import _written_days
from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.panel_builder import V2PanelBuilder, bootstrap_manifest

_GROUPS = (
    "core",
    "fundamental",
    "moneyflow",
    "technical",
    "announcement",
    "risk",
    "market_state",
    "neutralized",
)
#: 每族取 1 列（族一致性校验要求列与登记族匹配）
_GROUP_COL = {
    "core": "ret_1",
    "fundamental": "roe_dt",
    "moneyflow": "net_mf_amount",
    "technical": "rsi_14",
    "announcement": "pledge_ratio",
    "risk": "amihud_illiq_20",
    "market_state": "mkt_ret_vol_20",
    "neutralized": "zscore_pb",
}


def _day_df(date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["600000.SH", "000001.SZ"],
            "ret_1": [0.01, -0.02],
        }
    )


def _group_day_df(group: str, date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date, date],
            "ts_code": ["600000.SH", "000001.SZ"],
            _GROUP_COL[group]: [0.01, -0.02],
        }
    )


def _write_hot(store: PanelDataStore, date: str, groups: tuple[str, ...] = _GROUPS) -> None:
    for group in groups:
        store.append_features(TradeDate.from_str(date), group, _group_day_df(group, date))


def _write_cold_month(store: PanelDataStore, month: str, groups: tuple[str, ...] = _GROUPS) -> None:
    date = month.replace("-", "") + "03"
    for group in groups:
        store.append_archive_features(month, group, _group_day_df(group, date))


class TestWrittenDays:
    def test_hot_day_all_groups_written(self, tmp_path):
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        _write_hot(store, "20260603")
        assert _written_days(store, ["20260603"]) == {"20260603"}

    def test_hot_day_partial_groups_not_written(self, tmp_path):
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        _write_hot(store, "20260603", groups=_GROUPS[:7])  # 缺一族
        assert _written_days(store, ["20260603"]) == set()

    def test_hot_day_registered_but_file_missing(self, tmp_path):
        """热区 8 族全登记但盘上文件丢失 ⇒ 按未写处理（D-10 镜像教训）。"""
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        _write_hot(store, "20260603")
        (store.panel_dir / "20260603" / "core.parquet").unlink()
        assert _written_days(store, ["20260603"]) == set()

    def test_cold_month_day_precision(self, tmp_path):
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        _write_cold_month(store, "2025-05")  # 仅落 20250503
        # 冷区日级精度（D-10 防复发）：仅 core 族分区实际含有的日视为已写，
        # 月文件已登记但日内缺日的日期不算已写（分块末标签不成熟缺口事件教训）
        assert _written_days(store, ["20250503"]) == {"20250503"}
        assert _written_days(store, ["20250505", "20250529"]) == set()
        assert _written_days(store, ["20250602"]) == set()

    def test_cold_month_partial_not_written(self, tmp_path):
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        _write_cold_month(store, "2025-05", groups=_GROUPS[:7])
        assert _written_days(store, ["20250505"]) == set()

    def test_labels_not_counted(self, tmp_path):
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        store.append_labels(
            "y_ret_20",
            pd.DataFrame(
                {
                    "trade_date": ["20260603"],
                    "ts_code": ["600000.SH"],
                    "label_value": [0.01],
                    "maturity_status": ["forming"],
                }
            ),
        )
        assert _written_days(store, ["20260603"]) == set()  # labels 不算已写


def _stub_builder(monkeypatch, captured, calendar=None):
    builder = V2PanelBuilder(loader=Mock())
    # stub 尊重 keep_dates（与真实 _run_capture 的 sink 过滤同语义）
    monkeypatch.setattr(
        builder,
        "_run_capture",
        lambda s, e, keep_dates=None: [
            (d, df) for d, df in captured if keep_dates is None or d in keep_dates
        ],
    )
    # 日历默认取捕获日并集；调用方须保证覆盖 backfill 区间（否则 range_dates 为空先报错）
    stub_cal = calendar if calendar is not None else sorted({d for d, _ in captured})
    monkeypatch.setattr(builder, "_full_calendar", lambda: stub_cal)
    monkeypatch.setattr(builder, "_data_horizon", lambda: "20991231")  # 数据水位打桩
    monkeypatch.setattr(
        "src.lazybull.v2.store.panel_builder.load_top_inst_lookup", lambda loader, dates: {}
    )
    return builder


def _big_day(date: str, ret: float = 0.01) -> pd.DataFrame:
    """含 6 标签列 + circ_mv（ti 分母）的合成单日大表。"""
    df = _day_df(date)
    df["ret_1"] = [ret, ret]
    for col in ("y_ret_5", "y_ret_10", "y_ret_20"):
        df[col] = 0.01
    for col in ("neu_y_ret_5", "neu_y_ret_10", "neu_y_ret_20"):
        df[col] = 0.005
    df["circ_mv"] = 1e6
    return df


class TestBackfillKeepDates:
    def test_keep_dates_filters_writes(self, tmp_path, monkeypatch):
        """分块重叠场景（EMA 长尾陷阱）：已写日在重叠块重建值不同，
        keep_dates 剔除 ⇒ 不触发指纹冲突且盘上内容不变。"""
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        _write_hot(store, "20260601")  # 已写日（ret_1=0.01）
        before = store.manifest.partition_fingerprint("panel/20260601/core.parquet")
        captured = [
            ("20260601", _big_day("20260601", ret=9.9)),  # 重叠重建值不同（尾差模拟）
            ("20260602", _big_day("20260602")),
            ("20260603", _big_day("20260603")),
        ]
        builder = _stub_builder(monkeypatch, captured)
        builder.backfill(
            TradeDate.from_str("20260601"),
            TradeDate.from_str("20260603"),
            ["core"],
            store,
            keep_dates={"20260602", "20260603"},
        )
        assert store.manifest.partition_fingerprint("panel/20260601/core.parquet") == before
        assert (tmp_path / "features/panel/20260602/core.parquet").exists()
        assert (tmp_path / "features/panel/20260603/core.parquet").exists()

    def test_overlap_without_keep_raises_fingerprint_conflict(self, tmp_path, monkeypatch):
        """对照：keep_dates 不剔除已写日 ⇒ 重建值不同必触发指纹冲突（陷阱实证）。"""
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        _write_hot(store, "20260601")
        captured = [("20260601", _big_day("20260601", ret=9.9))]
        builder = _stub_builder(monkeypatch, captured)
        with pytest.raises(RuntimeError, match="指纹冲突"):
            builder.backfill(
                TradeDate.from_str("20260601"),
                TradeDate.from_str("20260601"),
                ["core"],
                store,
                keep_dates={"20260601"},
            )

    def test_keep_dates_empty_skips_build(self, tmp_path, monkeypatch):
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        builder = _stub_builder(monkeypatch, [], calendar=["20260601", "20260602", "20260603"])
        builder.backfill(
            TradeDate.from_str("20260601"),
            TradeDate.from_str("20260603"),
            ["core"],
            store,
            keep_dates=set(),
        )  # 不报错 = 跳过构建

    def test_keep_dates_out_of_range_raises(self, tmp_path, monkeypatch):
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        builder = _stub_builder(monkeypatch, [], calendar=["20260601", "20260602", "20260603"])
        with pytest.raises(ValueError, match="区间外"):
            builder.backfill(
                TradeDate.from_str("20260601"),
                TradeDate.from_str("20260603"),
                ["core"],
                store,
                keep_dates={"20260608"},
            )

    def test_missing_keep_day_raises(self, tmp_path, monkeypatch):
        """keep 日未被捕获（管线吞异常）⇒ 显式报错。"""
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        # 25 天日历（T+21 端点在界内 ⇒ 两日均为可建日，缺捕获才触发断言）
        cal_25 = [(date(2026, 6, 1) + timedelta(days=i)).strftime("%Y%m%d") for i in range(25)]
        builder = _stub_builder(
            monkeypatch,
            [("20260601", _big_day("20260601"))],
            calendar=cal_25,
        )
        with pytest.raises(RuntimeError, match="缺日"):
            builder.backfill(
                TradeDate.from_str("20260601"),
                TradeDate.from_str("20260602"),
                ["core"],
                store,
                keep_dates={"20260601", "20260602"},
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
