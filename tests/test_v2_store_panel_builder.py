# -*- coding: utf-8 -*-
"""v2 panel 构建器测试（tmp_path 合成数据 + mock 捕获，不触碰真实 data/ 生产分区）。

覆盖：_split_groups 族归属/键列/标签剥离/缺列补 NaN；_ArchiveMonthBuffer 月切换 flush；
append_archive_features 两阶段/冲突/no-op/校验；bootstrap_manifest 登记与幂等；
backfill 冷热路由（stub 捕获驱动，20250630 边界两侧）与缺日兜底。
"""

from datetime import date, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from src.lazybull.v2.store.column_groups import MATERIALIZED_COLUMNS, PANEL_GROUPS
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.panel_builder import (
    V2PanelBuilder,
    _ArchiveMonthBuffer,
    _split_groups,
    bootstrap_manifest,
)

_CODES = ["600000.SH", "000001.SZ", "000002.SZ"]
#: 60 天真实日期序列（20250620 起），跨冷热边界 20250630 且覆盖 T+21 标签端点
_CALENDAR = [(date(2025, 6, 20) + timedelta(days=i)).strftime("%Y%m%d") for i in range(60)]


def _big_day(date: str) -> pd.DataFrame:
    """合成单日大表：8 族各 1~2 列 + 6 标签列 + circ_mv（ti 分母）+ 部分物化列。"""
    n = len(_CODES)
    return pd.DataFrame(
        {
            "trade_date": [date] * n,
            "ts_code": _CODES,
            "ret_1": [0.01, -0.02, 0.0],  # core
            "circ_mv": [1e6, 2e6, 3e6],  # core（ti 市值分母）
            "roe_dt": [0.1, 0.2, 0.15],  # fundamental
            "net_mf_amount": [1.0, 2.0, 3.0],  # moneyflow
            "rsi_14": [55.0, 60.0, 50.0],  # technical
            "pledge_ratio": [0.1, 0.0, 0.2],  # announcement（基础 10）
            "amihud_illiq_20": [0.5, 0.6, 0.7],  # risk
            "mkt_ret_vol_20": [0.02, 0.02, 0.02],  # market_state
            "zscore_pb": [0.1, -0.1, 0.0],  # neutralized
            "ht_net_ratio_30d": [0.0, 0.1, 0.0],  # 物化 ht
            "cons_analyst_count_30d": [3.0, 0.0, 5.0],  # has_cons_coverage 来源列
            "rzye_chg_5": [100.0, 200.0, 0.0],  # has_margin_balance 来源列
            "fund_hold_ratio": [0.5, 0.0, 1.0],  # has_fund_holding 来源列
            "express_profit_yoy": [0.2, 0.0, 0.3],  # has_express_data 来源列
            "y_ret_5": [0.01] * n,
            "y_ret_10": [0.02] * n,
            "y_ret_20": [0.03] * n,
            "neu_y_ret_5": [0.005] * n,
            "neu_y_ret_10": [0.01] * n,
            "neu_y_ret_20": [0.015] * n,
        }
    )


class TestSplitGroups:
    def test_group_assignment_and_keys(self):
        frames = _split_groups(_big_day("20250627"))
        assert set(frames) == set(PANEL_GROUPS)
        for group, frame in frames.items():
            assert list(frame.columns[:2]) == ["trade_date", "ts_code"]
            assert len(frame) == 3
        assert "ret_1" in frames["core"].columns and "roe_dt" not in frames["core"].columns
        assert "roe_dt" in frames["fundamental"].columns
        assert "net_mf_amount" in frames["moneyflow"].columns
        assert "rsi_14" in frames["technical"].columns
        assert "amihud_illiq_20" in frames["risk"].columns
        assert "mkt_ret_vol_20" in frames["market_state"].columns
        assert "zscore_pb" in frames["neutralized"].columns

    def test_announcement_has_base_plus_materialized(self):
        frames = _split_groups(_big_day("20250627"))
        announcement = frames["announcement"]
        assert len(announcement.columns) == 2 + 44  # 键 + 10 基础 + 34 物化
        assert announcement["ht_net_ratio_30d"].tolist() == [0.0, 0.1, 0.0]
        assert announcement["ti_inst_net_ratio"].isna().all()  # 未喂入的物化列 ⇒ NaN

    def test_labels_stripped(self):
        frames = _split_groups(_big_day("20250627"))
        for frame in frames.values():
            assert not [c for c in frame.columns if c.startswith(("y_ret_", "neu_y_ret_"))]

    def test_missing_group_column_filled_nan(self):
        df = _big_day("20250627").drop(columns=["rsi_14"])
        frames = _split_groups(df)
        assert frames["technical"]["rsi_14"].isna().all()
        assert list(frames["technical"].columns) == ["trade_date", "ts_code"] + [
            c for c in PANEL_GROUPS["technical"] if c not in ("trade_date", "ts_code")
        ]

    def test_unregistered_column_raises(self):
        """列集闭合断言：列新增/改名未登记 ⇒ 显式失败而非静默 reindex 丢列。"""
        df = _big_day("20250627").assign(mystery_new_col=1.0)
        with pytest.raises(RuntimeError, match="未登记列"):
            _split_groups(df)


class TestBootstrapManifest:
    def test_register_all_columns_and_dependencies(self, tmp_path):
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        snapshot = store.get_manifest()
        columns = snapshot["columns"]
        assert len(columns) == 411  # 377 + 34
        assert columns["ret_1"]["group"] == "core" and columns["ret_1"]["status"] == "active"
        assert columns["ht_net_ratio_30d"]["group"] == "announcement"
        assert columns["ht_net_ratio_30d"]["status"] == "deprecated"
        assert columns["trade_date"]["group"] == "core"
        deps = snapshot["dependencies"]
        assert "daily" in deps["raw_datasets"] and "top_inst" in deps["raw_datasets"]
        assert "top_inst" in deps["factor_functions"]

    def test_idempotent_rerun(self, tmp_path):
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        bootstrap_manifest(store)  # 幂等重跑不报错
        assert len(store.get_manifest()["columns"]) == 411

    def test_existing_governance_metadata_preserved(self, tmp_path):
        """R3-07：bootstrap 重跑不覆盖已登记列的治理元数据。"""
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        store.manifest.register_column(
            "ret_1",
            "core",
            source="clean/daily",
            available_from="20130101",
            backfilled_at="2026-10-01T00:00:00",
            status="deprecated",
        )
        bootstrap_manifest(store)
        meta = store.manifest.column_meta("ret_1")
        assert meta["available_from"] == "20130101"
        assert meta["backfilled_at"] == "2026-10-01T00:00:00"
        assert meta["status"] == "deprecated"
        assert len(store.get_manifest()["columns"]) == 411  # 仍幂等

    def test_conflict_with_existing_registration_raises(self, tmp_path):
        store = PanelDataStore(tmp_path)
        store.manifest.register_column("ret_1", "core", "src", definition_version="v2")
        store.manifest.save()
        with pytest.raises(ValueError, match="冲突"):
            bootstrap_manifest(store)


@pytest.fixture
def bootstrapped_store(tmp_path):
    store = PanelDataStore(tmp_path)
    bootstrap_manifest(store)
    return store


def _archive_frame(month_days: list[str], col: str = "ret_1") -> pd.DataFrame:
    dates = [d for d in month_days for _ in _CODES]
    return pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": _CODES * len(month_days),
            col: [0.01] * len(dates),
        }
    )


class TestAppendArchiveFeatures:
    def test_two_phase_success(self, bootstrapped_store, tmp_path):
        bootstrapped_store.append_archive_features("2025-05", "core", _archive_frame(["20250506"]))
        final = tmp_path / "features" / "panel_archive" / "2025-05" / "core.parquet"
        assert final.exists()
        meta = bootstrapped_store.manifest.partition_meta("panel_archive/2025-05/core.parquet")
        assert meta["zone"] == "archive"
        assert meta["rows"] == 3
        assert list(final.parent.glob("*.tmp")) == []

    def test_identical_noop(self, bootstrapped_store):
        df = _archive_frame(["20250506"])
        bootstrapped_store.append_archive_features("2025-05", "core", df)
        bootstrapped_store.append_archive_features("2025-05", "core", df)  # no-op 不报错

    def test_conflict_raises(self, bootstrapped_store):
        bootstrapped_store.append_archive_features("2025-05", "core", _archive_frame(["20250506"]))
        with pytest.raises(RuntimeError, match="禁止原地修改"):
            bootstrapped_store.append_archive_features(
                "2025-05", "core", _archive_frame(["20250506", "20250507"])
            )

    def test_out_of_month_raises(self, bootstrapped_store):
        with pytest.raises(RuntimeError, match="月外日期"):
            bootstrapped_store.append_archive_features(
                "2025-05", "core", _archive_frame(["20250601"])
            )

    def test_bad_month_format_raises(self, bootstrapped_store):
        with pytest.raises(RuntimeError, match="YYYY-MM"):
            bootstrapped_store.append_archive_features(
                "202505", "core", _archive_frame(["20250506"])
            )

    def test_unregistered_column_raises(self, bootstrapped_store):
        df = _archive_frame(["20250506"]).assign(mystery=1.0)
        with pytest.raises(RuntimeError, match="未登记"):
            bootstrapped_store.append_archive_features("2025-05", "core", df)


class TestArchiveMonthBuffer:
    def test_month_switch_auto_flush(self, bootstrapped_store, tmp_path):
        buffer = _ArchiveMonthBuffer(bootstrapped_store, ["core", "moneyflow"])
        day1 = {
            "core": _archive_frame(["20250529"]),
            "moneyflow": _archive_frame(["20250529"], "net_mf_amount"),
        }
        day2 = {
            "core": _archive_frame(["20250530"]),
            "moneyflow": _archive_frame(["20250530"], "net_mf_amount"),
        }
        day3 = {  # 月切换 ⇒ 自动 flush 2025-05
            "core": _archive_frame(["20250602"]),
            "moneyflow": _archive_frame(["20250602"], "net_mf_amount"),
        }
        buffer.add("20250529", day1)
        buffer.add("20250530", day2)
        buffer.add("20250602", day3)
        buffer.flush()
        may = pd.read_parquet(tmp_path / "features" / "panel_archive" / "2025-05" / "core.parquet")
        june = pd.read_parquet(tmp_path / "features" / "panel_archive" / "2025-06" / "core.parquet")
        assert sorted(may["trade_date"].unique()) == ["20250529", "20250530"]
        assert june["trade_date"].unique().tolist() == ["20250602"]
        assert len(may) == 6 and len(june) == 3

    def test_flush_without_buffer_noop(self, bootstrapped_store):
        _ArchiveMonthBuffer(bootstrapped_store, ["core"]).flush()  # 不报错


class TestBuildDailyLabelStarved:
    """build_daily 标签不可终 fail-fast（R3-03 防御性收口）。"""

    def _stub(self, monkeypatch, captured, horizon):
        builder = V2PanelBuilder(loader=Mock())
        monkeypatch.setattr(
            builder,
            "_run_capture",
            lambda s, e, keep_dates=None: [
                (d, df) for d, df in captured if keep_dates is None or d in keep_dates
            ],
        )
        monkeypatch.setattr(builder, "_full_calendar", lambda: list(_CALENDAR))
        monkeypatch.setattr(builder, "_data_horizon", lambda: horizon)
        monkeypatch.setattr(
            "src.lazybull.v2.store.panel_builder.load_top_inst_lookup", lambda loader, dates: {}
        )
        return builder

    def test_unsealable_day_fail_fast(self, bootstrapped_store, tmp_path, monkeypatch):
        """T+21 端点超数据水位 ⇒ RuntimeError（文案含 R3-03），不落盘不报误导性缺日。"""
        from src.lazybull.v2.common.types import TradeDate

        day = _CALENDAR[0]  # 端点 _CALENDAR[21]=20250711 > 水位 _CALENDAR[20]=20250710
        builder = self._stub(monkeypatch, [(day, _big_day(day))], horizon=_CALENDAR[20])
        with pytest.raises(RuntimeError, match="R3-03"):
            builder.build_daily(TradeDate.from_str(day), ["core"], bootstrapped_store)
        assert not (tmp_path / "features" / "panel" / day).exists()  # 未落盘

    def test_buildable_day_proceeds(self, bootstrapped_store, tmp_path, monkeypatch):
        from src.lazybull.v2.common.types import TradeDate

        day = _CALENDAR[0]
        builder = self._stub(monkeypatch, [(day, _big_day(day))], horizon="20991231")
        builder.build_daily(TradeDate.from_str(day), ["core"], bootstrapped_store)
        assert (tmp_path / "features" / "panel" / day / "core.parquet").exists()
        assert (tmp_path / "labels" / "y_ret_20" / f"{day}.parquet").exists()


class TestBackfillRouting:
    def _builder_with_stub(self, monkeypatch, captured):
        builder = V2PanelBuilder(loader=Mock())
        monkeypatch.setattr(
            builder,
            "_run_capture",
            lambda s, e, keep_dates=None: [
                (d, df) for d, df in captured if keep_dates is None or d in keep_dates
            ],
        )
        monkeypatch.setattr(builder, "_full_calendar", lambda: list(_CALENDAR))
        monkeypatch.setattr(builder, "_data_horizon", lambda: "20991231")  # 数据水位打桩
        monkeypatch.setattr(
            "src.lazybull.v2.store.panel_builder.load_top_inst_lookup", lambda loader, dates: {}
        )
        return builder

    def test_cold_hot_boundary_routing(self, bootstrapped_store, tmp_path, monkeypatch):
        captured = [(d, _big_day(d)) for d in ("20250629", "20250630", "20250701")]
        builder = self._builder_with_stub(monkeypatch, captured)
        from src.lazybull.v2.common.types import TradeDate

        builder.backfill(
            TradeDate.from_str("20250629"),
            TradeDate.from_str("20250701"),
            ["core"],
            bootstrapped_store,
        )
        archive = pd.read_parquet(
            tmp_path / "features" / "panel_archive" / "2025-06" / "core.parquet"
        )
        assert sorted(archive["trade_date"].unique()) == ["20250629", "20250630"]  # ≤ 边界入冷区
        hot = pd.read_parquet(tmp_path / "features" / "panel" / "20250701" / "core.parquet")
        assert hot["trade_date"].unique().tolist() == ["20250701"]  # > 边界入热区
        labels = pd.read_parquet(tmp_path / "labels" / "y_ret_20" / "20250701.parquet")
        assert "label_value" in labels.columns  # labels 逐日随落

    def test_missing_day_raises(self, bootstrapped_store, monkeypatch):
        builder = self._builder_with_stub(monkeypatch, [("20250629", _big_day("20250629"))])
        from src.lazybull.v2.common.types import TradeDate

        with pytest.raises(RuntimeError, match="缺日"):
            builder.backfill(
                TradeDate.from_str("20250629"),
                TradeDate.from_str("20250630"),
                ["core"],
                bootstrapped_store,
            )

    def test_materialize_day_adds_ti_and_has(self, monkeypatch):
        builder = V2PanelBuilder(loader=Mock())
        out = builder._materialize_day(_big_day("20250627"), "20250627", ti_lookup={})
        for col in MATERIALIZED_COLUMNS:
            if col.startswith(("ti_", "has_")):
                assert col in out.columns, col
        assert out["ti_inst_net_ratio"].tolist() == [0.0] * 3  # 空查询表 ⇒ 0 填充
        assert out["ti_schema_v1"].tolist() == [1] * 3  # 哨兵恒写


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
