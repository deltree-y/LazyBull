# -*- coding: utf-8 -*-
"""v2 P1 单元 5：新增列演练（可执行操作剧本，合成数据，不触碰真实 data/）。

演练目的：验证「新增因子列」全生命周期语义（契约 §4.1 补丁 1 + 冻结文档 §4）——
① manifest 登记（含 available_from）是写入前提；
② 新增列**禁止追溯改写已写分区**（热区日：指纹冲突拒绝；封存月：原地修改拒绝）；
③ 新增列只写登记后的新分区，旧分区逐位不变（指纹零变化）；
④ 早于登记的分区读取时该列整列 NaN（_load_group_frame reindex 补齐）；
⑤ available_from 的 PIT 查询治理（越界查询拒绝）。
"""

import pandas as pd
import pytest

from src.lazybull.v2.common.types import FeatureQuery, TradeDate
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.panel_builder import bootstrap_manifest

_DRILL_COL = "drill_momentum_3d"
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


def _day_df(group: str, date: str, extra_col: str | None = None) -> pd.DataFrame:
    data = {
        "trade_date": [date, date],
        "ts_code": ["600000.SH", "000001.SZ"],
        _GROUP_COL[group]: [0.01, -0.02],
    }
    if extra_col:
        data[extra_col] = [0.001, 0.002]
    return pd.DataFrame(data)


def _write_day(store: PanelDataStore, date: str, with_drill: bool = False) -> None:
    for group in _GROUPS:
        extra = _DRILL_COL if (with_drill and group == "technical") else None
        store.append_features(TradeDate.from_str(date), group, _day_df(group, date, extra))


@pytest.fixture
def store(tmp_path):
    s = PanelDataStore(tmp_path)
    bootstrap_manifest(s)
    return s


class TestNewColumnDrill:
    def test_full_lifecycle(self, store):
        # 前置：D1 全日 8 族已写（演练基线）
        _write_day(store, "20260603")
        fp_d1_before = store.manifest.partition_fingerprint("panel/20260603/technical.parquet")

        # ① 登记（technical 族，available_from = D1——登记晚于 D1 写入 ⇒ D1 分区无该列）
        store.manifest.register_column(
            _DRILL_COL, "technical", source="单元 5 新增列演练", definition_version="v1",
            available_from="20260603",
        )

        # ② 追溯改写 D1（含新列）⇒ 指纹冲突拒绝，D1 逐位不变
        with pytest.raises(RuntimeError, match="指纹冲突"):
            store.append_features(
                TradeDate.from_str("20260603"),
                "technical",
                _day_df("technical", "20260603", extra_col=_DRILL_COL),
            )
        assert store.manifest.partition_fingerprint("panel/20260603/technical.parquet") == fp_d1_before

        # ③ D2 全日写入（technical 含新列）⇒ 成功；D1 指纹仍不变
        _write_day(store, "20260604", with_drill=True)
        assert store.manifest.partition_fingerprint("panel/20260603/technical.parquet") == fp_d1_before
        assert store.manifest.partition_fingerprint("panel/20260604/technical.parquet") is not None

        # ④ 登记前已写的 D1 分区读取：新列整列 NaN（reindex 补齐），其余列正常
        frame = store.load_features(
            FeatureQuery(
                columns=[_GROUP_COL["technical"], _DRILL_COL],
                start_date=TradeDate.from_str("20260603"),
                end_date=TradeDate.from_str("20260604"),
            )
        )
        df = frame.df.reset_index()
        d1 = df[df["trade_date"].astype(str) == "20260603"]
        d2 = df[df["trade_date"].astype(str) == "20260604"]
        assert d1[_DRILL_COL].isna().all()
        assert d1[_GROUP_COL["technical"]].notna().all()
        assert d2[_DRILL_COL].notna().all()
        # available_from 元数据随查询返回（PIT 治理元数据）
        assert str(frame.available_from[_DRILL_COL]) == "20260603"

        # ⑤ available_from 越界查询 ⇒ ValueError（晚登记列查更早区间）
        store.manifest.register_column(
            "drill_late_col", "technical", source="单元 5 新增列演练", definition_version="v1",
            available_from="20260701",
        )
        with pytest.raises(ValueError, match="available_from"):
            store.load_features(
                FeatureQuery(
                    columns=["drill_late_col"],
                    start_date=TradeDate.from_str("20260603"),
                    end_date=TradeDate.from_str("20260604"),
                )
            )

    def test_sealed_archive_month_blocks_retroactive_add(self, store):
        # 封存月已写 ⇒ 热区写入同月日期即拒（分区已封存），冷区原地修改亦拒
        month, date = "2025-05", "20250506"
        for group in _GROUPS:
            store.append_archive_features(month, group, _day_df(group, date))
        store.manifest.register_column(
            _DRILL_COL, "technical", source="单元 5 新增列演练", definition_version="v1"
        )
        with pytest.raises(RuntimeError, match="已封存"):
            store.append_features(
                TradeDate.from_str(date), "technical", _day_df("technical", date, extra_col=_DRILL_COL)
            )
        with pytest.raises(RuntimeError, match="禁止原地修改"):
            store.append_archive_features(
                month, "technical", _day_df("technical", date, extra_col=_DRILL_COL)
            )
        # 新月份写入含新列 ⇒ 成功（新增列只写新文件）
        store.append_archive_features("2025-06", "technical", _day_df("technical", "20250603", extra_col=_DRILL_COL))
        assert store.manifest.partition_fingerprint("panel_archive/2025-06/technical.parquet") is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
