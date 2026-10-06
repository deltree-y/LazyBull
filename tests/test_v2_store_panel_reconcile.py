# -*- coding: utf-8 -*-
"""v2 P1 单元 4：panel 对账门测试（合成数据，不触碰真实 data/）。

覆盖：严格门全零通过 / 注入差异失败；归因门 L1/L2 日期边界与越界检出、
D-04② 日期窗口、D-05 应为零、D-04①③④ 列族放行；行集差异检出；
指纹不变断言（store 级 no-op）。
"""

import pandas as pd
import pytest

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.panel_builder import bootstrap_manifest
from src.lazybull.v2.store.panel_reconcile import (
    _classify_attrib_column,
    _classify_strict_column,
    reconcile_panel,
)

_CODES = ["600000.SH", "000001.SZ"]
#: 每族取 1~2 列（合成小表；unlock_ratio/macd_hist 服务 D-12/D-11 严格门测试）
_GROUP_COLS = {
    "core": ["ret_1"],
    "moneyflow": ["net_mf_amount"],
    "technical": ["rsi_14", "macd_hist"],
    "fundamental": ["roe_dt"],
    "announcement": ["pledge_ratio", "unlock_ratio"],
    "risk": ["amihud_illiq_20"],
    "market_state": ["mkt_ret_vol_20"],
    "neutralized": ["zscore_pb"],
}


def _feature_frames(date: str, value: float = 0.01) -> dict[str, pd.DataFrame]:
    frames = {}
    for group, cols in _GROUP_COLS.items():
        data = {"trade_date": [date] * 2, "ts_code": list(_CODES)}
        for col in cols:
            data[col] = [value, value]
        frames[group] = pd.DataFrame(data)
    return frames


def _write_day(store: PanelDataStore, date: str, value: float = 0.01) -> None:
    for group, frame in _feature_frames(date, value).items():
        store.append_features(TradeDate.from_str(date), group, frame)
    for label in ("y_ret_5", "y_ret_10", "y_ret_20"):
        store.append_labels(
            label,
            pd.DataFrame(
                {
                    "trade_date": [date] * 2,
                    "ts_code": list(_CODES),
                    "label_value": [value, value],
                    "neu_label_value": [value / 2, value / 2],
                    "maturity_status": ["forming"] * 2,
                }
            ),
        )


def _write_cold_month(store: PanelDataStore, month: str, date: str, value: float = 0.01) -> None:
    for group, frame in _feature_frames(date, value).items():
        store.append_archive_features(month, group, frame)
    for label in ("y_ret_5", "y_ret_10", "y_ret_20"):
        store.append_labels(
            label,
            pd.DataFrame(
                {
                    "trade_date": [date] * 2,
                    "ts_code": list(_CODES),
                    "label_value": [value, value],
                    "neu_label_value": [value / 2, value / 2],
                    "maturity_status": ["forming"] * 2,
                }
            ),
        )


def _ref_day(
    ref_dir,
    date: str,
    value: float = 0.01,
    extra_codes: list[str] | None = None,
    overrides: dict[str, float] | None = None,
) -> None:
    """合成参照日分区（特征列 + 6 标签列；overrides 单列注值）。"""
    codes = list(_CODES) + (extra_codes or [])
    n = len(codes)
    overrides = overrides or {}
    data = {"trade_date": [date] * n, "ts_code": codes}
    for cols in _GROUP_COLS.values():
        for col in cols:
            data[col] = [overrides.get(col, value)] * n
    for col in ("y_ret_5", "y_ret_10", "y_ret_20"):
        data[col] = [value] * n
        data[f"neu_{col}"] = [value / 2] * n
    ref_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(ref_dir / f"{date}.parquet", index=False)


@pytest.fixture
def store_and_ref(tmp_path):
    store = PanelDataStore(tmp_path / "panel_root")
    bootstrap_manifest(store)
    return store, tmp_path / "ref"


class TestStrictGate:
    def test_all_zero_passes(self, store_and_ref):
        store, ref_dir = store_and_ref
        _write_day(store, "20260603", 0.01)
        _ref_day(ref_dir, "20260603", 0.01)
        report = reconcile_panel(store.root, ref_dir, ["20260603"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is True
        assert report["summary"]["columns_with_diff"] == 0

    def test_injected_diff_fails(self, store_and_ref):
        store, ref_dir = store_and_ref
        _write_day(store, "20260603", 0.01)
        _ref_day(ref_dir, "20260603", 0.02)  # ret_1 等全列差 0.01
        report = reconcile_panel(store.root, ref_dir, ["20260603"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is False
        assert "ret_1" in report["verdict"]["offenders"]
        assert report["columns"]["ret_1"]["attribution"].startswith("未归因")
        assert report["columns"]["ret_1"]["unattributed"] is True

    def test_d12_exempt_family_passes(self, store_and_ref):
        """D-12 豁免族（裁决 2026-10-06）：unlock_ratio 差异全归因登记 ⇒ 严格门放行。"""
        store, ref_dir = store_and_ref
        _write_day(store, "20260603", 0.01)
        _ref_day(ref_dir, "20260603", 0.01, overrides={"unlock_ratio": 0.02})
        report = reconcile_panel(store.root, ref_dir, ["20260603"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is True
        assert report["columns"]["unlock_ratio"]["attribution"].startswith("D-12")
        assert "unlock_ratio" in report["verdict"]["attributed_columns"]

    def test_d11_anchor_window_passes(self, store_and_ref):
        """D-11 分块锚定：长记忆列差异日 ⊆ 锚定影响窗（20170103~20170731）⇒ 归因放行。"""
        store, ref_dir = store_and_ref
        _write_cold_month(store, "2017-01", "20170105", 0.01)
        _ref_day(ref_dir, "20170105", 0.01, overrides={"macd_hist": 0.02})
        report = reconcile_panel(store.root, ref_dir, ["20170105"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is True
        assert report["columns"]["macd_hist"]["attribution"].startswith("D-11")

    def test_d11_outside_window_fails(self, store_and_ref):
        """锚定影响窗外 + 非 L1 期的长记忆列差异 ⇒ 未归因 fail。"""
        store, ref_dir = store_and_ref
        _write_cold_month(store, "2018-01", "20180105", 0.01)
        _ref_day(ref_dir, "20180105", 0.01, overrides={"macd_hist": 0.02})
        report = reconcile_panel(store.root, ref_dir, ["20180105"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is False
        assert "macd_hist" in report["verdict"]["offenders"]

    def test_row_set_diff_fails(self, store_and_ref):
        store, ref_dir = store_and_ref
        _write_day(store, "20260603", 0.01)
        _ref_day(ref_dir, "20260603", 0.01, extra_codes=["000002.SZ"])
        report = reconcile_panel(store.root, ref_dir, ["20260603"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is False
        assert report["days"]["row_set_diff_days"][0]["sample_only_ref"] == ["000002.SZ"]


class TestAttribGate:
    def test_l1_zone_diff_passes(self, store_and_ref):
        """2012 年内差异 ⇒ L1 血缘，归因门放行。"""
        store, ref_dir = store_and_ref
        _write_cold_month(store, "2012-01", "20120105", 0.01)
        _ref_day(ref_dir, "20120105", 0.02)
        report = reconcile_panel(store.root, ref_dir, ["20120105"], mode="attrib", n_jobs=1)
        assert report["verdict"]["pass"] is True
        assert report["columns"]["ret_1"]["attribution"].startswith("L1")

    def test_out_of_boundary_diff_fails(self, store_and_ref):
        """待调查类列在 ≥2016 出差异 ⇒ 越界失败。"""
        store, ref_dir = store_and_ref
        _write_day(store, "20260603", 0.01)
        _ref_day(ref_dir, "20260603", 0.02)
        report = reconcile_panel(store.root, ref_dir, ["20260603"], mode="attrib", n_jobs=1)
        assert report["verdict"]["pass"] is False
        assert "ret_1" in report["verdict"]["boundary_violations"]


class TestClassifyBoundaries:
    """归因门类判定（单元 4 复核后语义：diff_days = [(date, over_rows, day_rows)]）。

    L1 = 全部差异日 ≤2012 年底；L1+数据态孤立单点 = L1 期 + 期外零星日（单日 ≤1%、
    期外累计 ≤0.1%）归因不越界；D-04 族按事件日登记不设日期边界；D-04② 命中窗口
    外 ⇒ 越界；D-05 名命中（skew/kurt）在归因门按日期模式归类（参照已修复）。
    """

    def test_l1_zone_diff_passes(self):
        klass, violated = _classify_attrib_column("ret_5", [("20120105", 100, 5000)])
        assert klass.startswith("L1") and not violated

    def test_l1_plus_scattered_drift_passes(self):
        # L1 期 5000 行 + 期外 2 日各 1 行：单日 1/5000=0.02% ≤1%、期外累计 2/5002≈0.04% ≤0.1% ⇒ 归因
        days = [("20120105", 5000, 5000)] + [(d, 1, 5000) for d in ("20220513", "20220516")]
        klass, violated = _classify_attrib_column("downside_vol_20", days)
        assert "孤立单点" in klass and not violated

    def test_l1_plus_dense_late_days_violate(self):
        # 期外单日 200/5000=4% >1% ⇒ 越界
        days = [("20120105", 100, 5000), ("20220513", 200, 5000)]
        _, violated = _classify_attrib_column("downside_vol_20", days)
        assert violated

    def test_d04_family_passthrough(self):
        assert not _classify_attrib_column("unlock_ratio", [("20260601", 10, 5000)])[1]  # D-04①
        assert not _classify_attrib_column("dividend_payout_ratio", [("20260601", 10, 5000)])[1]  # D-04③
        assert not _classify_attrib_column("rzye_chg_5", [("20260601", 10, 5000)])[1]  # D-04④

    def test_d04_2_date_window(self):
        assert not _classify_attrib_column("downside_corr_20", [("20240117", 10, 5000)])[1]
        assert _classify_attrib_column("downside_corr_20", [("20250101", 10, 5000)])[1]

    def test_d05_named_columns_classified_by_date_pattern(self):
        # skew/kurt 差异日全在 L1 期 ⇒ L1 归因、不判越界（参照侧 D-05 已修复）
        klass, violated = _classify_attrib_column("skewness_20", [("20120105", 100, 5000)])
        assert klass.startswith("L1") and not violated


class TestAttribSpecial:
    """D-14 专项登记（skew/kurt 数值尾、zscore_macd EMA 长尾、downside_corr 血缘块）。"""

    def test_skew_kurt_numeric_tail_attributed(self):
        # L1 主体 + 晚日 ≤25 股且 ≤1.5%（2015 停牌簇量级）⇒ 归因
        days = [("20120201", 1680, 1683), ("20150522", 23, 1752), ("20260122", 1, 5054)]
        klass, violated = _classify_attrib_column("kurtosis_20", days)
        assert "D-05" in klass and not violated

    def test_skew_kurt_dense_late_day_violates(self):
        days = [("20120201", 1680, 1683), ("20150522", 200, 1752)]
        _, violated = _classify_attrib_column("kurtosis_20", days)
        assert violated

    def test_zscore_macd_ema_tail_and_d11_attributed(self):
        # EMA 长尾（≤2013-09，允许密集）+ D-11 窗内复牌簇 + 稀疏孤立单点 ⇒ 归因
        days = [
            ("20130729", 1809, 2049),
            ("20130902", 23, 2066),
            ("20220505", 3058, 3872),
            ("20220826", 1, 4093),
        ]
        klass, violated = _classify_attrib_column("zscore_macd_hist_sz", days)
        assert "EMA" in klass and not violated

    def test_zscore_macd_dense_outside_bounds_violates(self):
        _, violated = _classify_attrib_column("zscore_macd_hist", [("20140105", 100, 5000)])
        assert violated

    def test_downside_corr_lineage_block_attributed(self):
        # 2013-09/10 血缘块 + D-04② 窗 + 2021-12 稀疏单点 ⇒ 归因
        days = [("20120104", 2014, 2059), ("20131017", 2014, 2059), ("20240117", 10, 4000), ("20211216", 1, 3705)]
        klass, violated = _classify_attrib_column("downside_corr_20", days)
        assert "血缘" in klass and not violated

    def test_downside_corr_dense_outside_bounds_violates(self):
        _, violated = _classify_attrib_column("downside_corr_20", [("20140105", 100, 5000)])
        assert violated


class TestStrictClassify:
    """严格门类判定（出口 = 残差 100% 归因：L1/D-04/D-11/D-12 豁免/纯孤立单点外 ⇒ 未归因）。"""

    def test_d12_family_any_date_attributed(self):
        for col in ("days_to_unlock", "unlock_ratio", "unlock_risk_flag"):
            klass, unattributed = _classify_strict_column(col, [("20140103", 5000, 5000)], 89.8)
            assert klass.startswith("D-12") and not unattributed

    def test_d04_family_passthrough(self):
        assert not _classify_strict_column("dividend_payout_ratio", [("20260601", 10, 5000)], 0.0)[1]
        assert not _classify_strict_column("rzye_chg_5", [("20260601", 10, 5000)], 0.0)[1]

    def test_d11_window_anchor_sensitive(self):
        # 锚定影响窗内（2017-01 / 2022-01 起 ~9 个月）长记忆列 ⇒ D-11 归因
        klass, unattributed = _classify_strict_column("macd_hist", [("20170205", 10, 5000)], 0.01)
        assert klass.startswith("D-11") and not unattributed
        klass, unattributed = _classify_strict_column(
            "zscore_macd_hist_sz", [("20220513", 10, 5000)], 0.08
        )
        assert klass.startswith("D-11") and not unattributed
        # 复牌股 EMA 收敛尾（实证 002260.SZ 2022-05 簇）可延伸至窗末
        klass, unattributed = _classify_strict_column("macd_hist", [("20220902", 1, 3894)], 0.02)
        assert klass.startswith("D-11") and not unattributed

    def test_d11_window_companion_sparse_point(self):
        # 窗内非长记忆列的稀疏微量单点（复牌股锚定事件伴随）⇒ D-11 伴随稀疏单点
        klass, unattributed = _classify_strict_column("amplitude", [("20220505", 1, 3872)], 0.0)
        assert klass.startswith("D-11") and not unattributed

    def test_d11_window_dense_non_anchor_unattributed(self):
        # 同窗内但非长记忆列且密集差异（4% > 1%）⇒ 不落 D-11，未归因
        _, unattributed = _classify_strict_column("ret_1", [("20170205", 200, 5000)], 0.01)
        assert unattributed

    def test_d11_outside_window_unattributed(self):
        # 窗外非孤立单点量级（30 行 > 纯单点阈 20 行）的长记忆列差异 ⇒ 未归因
        _, unattributed = _classify_strict_column("macd_hist", [("20180105", 30, 5000)], 0.01)
        assert unattributed

    def test_l1_zone_attributed(self):
        klass, unattributed = _classify_strict_column("ret_5", [("20120105", 100, 5000)], 0.01)
        assert klass.startswith("L1") and not unattributed

    def test_l1_plus_scattered_attributed(self):
        days = [("20120105", 5000, 5000)] + [(d, 1, 5000) for d in ("20220513", "20220516")]
        klass, unattributed = _classify_strict_column("downside_vol_20", days, 0.01)
        assert "孤立单点" in klass and not unattributed

    def test_pure_scatter_point_attributed(self):
        # 百分位秩单步翻转：全日 1 行稀疏、总量 ≤20、日数 ≤10、max|Δ| ≤0.05 ⇒ 纯孤立单点
        days = [(d, 1, 4000) for d in ("20191119", "20200902", "20231212")]
        klass, unattributed = _classify_strict_column("vol_regime_percentile", days, 0.008)
        assert "孤立单点" in klass and not unattributed

    def test_pure_scatter_point_over_magnitude_unattributed(self):
        # 单点但 max|Δ| > 0.05 ⇒ 超幅一律调查
        _, unattributed = _classify_strict_column("vol_regime_percentile", [("20191119", 1, 4000)], 0.5)
        assert unattributed


class TestImmutableFingerprints:
    def test_store_level_noop_and_conflict(self, tmp_path):
        """闸门③ store 级语义：同内容重放 no-op（指纹不变），不同内容冲突 raise。"""
        store = PanelDataStore(tmp_path)
        bootstrap_manifest(store)
        _write_day(store, "20260603", 0.01)
        _write_cold_month(store, "2025-05", "20250506", 0.01)
        hot_fp = store.manifest.partition_fingerprint("panel/20260603/core.parquet")
        cold_fp = store.manifest.partition_fingerprint("panel_archive/2025-05/core.parquet")
        _write_day(store, "20260603", 0.01)  # no-op
        _write_cold_month(store, "2025-05", "20250506", 0.01)  # no-op
        assert store.manifest.partition_fingerprint("panel/20260603/core.parquet") == hot_fp
        assert store.manifest.partition_fingerprint("panel_archive/2025-05/core.parquet") == cold_fp
        with pytest.raises(RuntimeError, match="指纹冲突|禁止原地修改"):
            _write_day(store, "20260603", 0.02)
        with pytest.raises(RuntimeError, match="禁止原地修改"):
            _write_cold_month(store, "2025-05", "20250506", 0.02)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
