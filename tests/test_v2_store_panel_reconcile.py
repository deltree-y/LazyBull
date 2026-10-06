# -*- coding: utf-8 -*-
"""v2 P1 单元 4：panel 对账门测试（合成数据，不触碰真实 data/）。

覆盖：严格门全零通过 / 注入差异失败；归因门 L1/L2 日期边界与越界检出、
D-04② 日期窗口、D-05 应为零、D-04①③④ 收紧（纯 NaN 型 + 登记窗 / 差异日下界）、
D-14 专项三上限、D-11 超幅上限；行集差异检出；结构合法性判定（missing_ref /
cols_only 白名单）；R3-04 全窗口分母；派生连锁跨列一致性；指纹不变断言。

合成数据约定：panel 侧 375 特征列 + 6 标签列全写、参照侧同列集——结构合法性
（fail-closed）判定要求两侧列集对齐，只写子集会误触 cols_only 越界。
"""

import pandas as pd
import pytest

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.store.column_groups import PANEL_GROUPS
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.panel_builder import bootstrap_manifest
from src.lazybull.v2.store.panel_reconcile import (
    CYQ5_COLUMNS,
    PANEL_FEATURE_COLUMNS,
    _classify_attrib_column,
    _classify_strict_column,
    _merge_and_judge,
    reconcile_panel,
)

_CODES = ["600000.SH", "000001.SZ"]


def _feature_frames(date: str, value: float = 0.01) -> dict[str, pd.DataFrame]:
    """全列族特征帧（375 特征列全写——与参照侧列集对齐，结构合法性判定前提）。"""
    frames = {}
    for group, cols in PANEL_GROUPS.items():
        data = {"trade_date": [date] * 2, "ts_code": list(_CODES)}
        for col in cols:
            if col in ("trade_date", "ts_code"):
                continue
            data[col] = [value, value]
        frames[group] = pd.DataFrame(data)
    return frames


def _label_frame(date: str, value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date] * 2,
            "ts_code": list(_CODES),
            "label_value": [value, value],
            "neu_label_value": [value / 2, value / 2],
            "maturity_status": ["forming"] * 2,
        }
    )


def _write_day(store: PanelDataStore, date: str, value: float = 0.01) -> None:
    for group, frame in _feature_frames(date, value).items():
        store.append_features(TradeDate.from_str(date), group, frame)
    for label in ("y_ret_5", "y_ret_10", "y_ret_20"):
        store.append_labels(label, _label_frame(date, value))


def _write_cold_month(store: PanelDataStore, month: str, date: str, value: float = 0.01) -> None:
    for group, frame in _feature_frames(date, value).items():
        store.append_archive_features(month, group, frame)
    for label in ("y_ret_5", "y_ret_10", "y_ret_20"):
        store.append_labels(label, _label_frame(date, value))


def _ref_day(
    ref_dir,
    date: str,
    value: float = 0.01,
    extra_codes: list[str] | None = None,
    overrides: dict[str, float] | None = None,
    drop_cols: tuple[str, ...] = (),
    extra_cols: dict[str, float] | None = None,
) -> None:
    """合成参照日分区（375 特征列 + 6 标签列；overrides 单列注值；drop_cols 模拟
    cols_only_panel 场景；extra_cols 模拟 cols_only_ref 场景）。"""
    codes = list(_CODES) + (extra_codes or [])
    n = len(codes)
    overrides = overrides or {}
    data = {"trade_date": [date] * n, "ts_code": codes}
    for col in PANEL_FEATURE_COLUMNS:
        if col in drop_cols:
            continue
        data[col] = [overrides.get(col, value)] * n
    for col in ("y_ret_5", "y_ret_10", "y_ret_20"):
        data[col] = [value] * n
        data[f"neu_{col}"] = [value / 2] * n
    for col, val in (extra_cols or {}).items():
        data[col] = [val] * n
    ref_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(ref_dir / f"{date}.parquet", index=False)


def _stat(over: int, rows: int = 5000, over_nan: int = 0, max_abs: float = 0.0) -> dict:
    """_merge_and_judge 测试用日级列统计。"""
    return {
        "rows": rows,
        "over_rows": over,
        "over_nan_rows": over_nan,
        "max_abs_diff": max_abs,
        "samples": [],
    }


def _day_result(date: str, col_stats: dict, n_rows: int = 5000) -> dict:
    """_merge_and_judge 测试用日级结果（行集/列集无差异）。"""
    return {
        "date": date,
        "status": "ok",
        "n_only_ref": 0,
        "n_only_panel": 0,
        "sample_only_ref": [],
        "sample_only_panel": [],
        "cols_only_ref": [],
        "cols_only_panel": [],
        "col_rows": {col: n_rows for col in col_stats},
        "columns": col_stats,
    }


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
        """D-11 分块锚定：长记忆列差异日 ⊆ 锚定影响窗（20170103~20170831）⇒ 归因放行。"""
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
        """2012 年内差异（非 D-04 列族）⇒ L1 血缘，归因门放行。"""
        store, ref_dir = store_and_ref
        _write_cold_month(store, "2012-01", "20120105", 0.01)
        _ref_day(ref_dir, "20120105", 0.01, overrides={"ret_1": 0.02})
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


class TestStructuralVerdict:
    """P1-1 整改：verdict 结构合法性判定（fail-closed：missing_ref / cols_only 白名单）。"""

    def test_missing_ref_day_fails(self, store_and_ref):
        store, ref_dir = store_and_ref
        _write_day(store, "20260603", 0.01)
        _write_day(store, "20260604", 0.01)
        _ref_day(ref_dir, "20260603", 0.01)  # 20260604 参照缺日
        report = reconcile_panel(
            store.root, ref_dir, ["20260603", "20260604"], mode="strict", n_jobs=1
        )
        assert report["verdict"]["pass"] is False
        assert any("missing_ref_days" in item for item in report["verdict"]["structural_offenders"])

    def test_cols_only_panel_cyq5_whitelist_passes(self, store_and_ref):
        """CYQ5 列且末日 ≤20171229 ⇒ 合法放行（实证：恰 5 列 × 20120104~20171229）。"""
        store, ref_dir = store_and_ref
        _write_cold_month(store, "2015-01", "20150105", 0.01)
        _ref_day(ref_dir, "20150105", 0.01, drop_cols=tuple(sorted(CYQ5_COLUMNS)))
        report = reconcile_panel(store.root, ref_dir, ["20150105"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is True
        assert set(report["cols_only_panel"]) == set(CYQ5_COLUMNS)

    def test_cols_only_panel_cyq5_late_fails(self, store_and_ref):
        """CYQ5 列但末日 >20171229 ⇒ 越出白名单边界，fail。"""
        store, ref_dir = store_and_ref
        _write_cold_month(store, "2018-01", "20180105", 0.01)
        _ref_day(ref_dir, "20180105", 0.01, drop_cols=("winner_rate",))
        report = reconcile_panel(store.root, ref_dir, ["20180105"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is False
        assert any("winner_rate" in item for item in report["verdict"]["structural_offenders"])

    def test_cols_only_panel_non_whitelist_fails(self, store_and_ref):
        """非 CYQ5 列且末日 >20121231 ⇒ fail 并列入 offenders。"""
        store, ref_dir = store_and_ref
        _write_day(store, "20260603", 0.01)
        _ref_day(ref_dir, "20260603", 0.01, drop_cols=("roe_dt",))
        report = reconcile_panel(store.root, ref_dir, ["20260603"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is False
        assert any("roe_dt" in item for item in report["verdict"]["structural_offenders"])
        assert any("roe_dt" in item for item in report["verdict"]["offenders"])

    def test_cols_only_panel_2012_schema_evolution_passes(self, store_and_ref):
        """任意列末日 ≤20121231（2012 冷启动 schema 演进）⇒ 合法放行。"""
        store, ref_dir = store_and_ref
        _write_cold_month(store, "2012-01", "20120105", 0.01)
        _ref_day(ref_dir, "20120105", 0.01, drop_cols=("roe_dt", "fund_hold_ratio"))
        report = reconcile_panel(store.root, ref_dir, ["20120105"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is True
        assert set(report["cols_only_panel"]) == {"roe_dt", "fund_hold_ratio"}

    def test_cols_only_ref_2012_legal(self, store_and_ref):
        """cols_only_ref 全部登记日 ≤20121231 ⇒ 合法放行。"""
        store, ref_dir = store_and_ref
        _write_cold_month(store, "2012-01", "20120105", 0.01)
        _ref_day(ref_dir, "20120105", 0.01, extra_cols={"legacy_col": 0.01})
        report = reconcile_panel(store.root, ref_dir, ["20120105"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is True
        assert report["cols_only_ref_days"] == [{"date": "20120105", "cols": ["legacy_col"]}]

    def test_cols_only_ref_late_fails(self, store_and_ref):
        """cols_only_ref 登记日 >20121231 ⇒ fail。"""
        store, ref_dir = store_and_ref
        _write_day(store, "20260603", 0.01)
        _ref_day(ref_dir, "20260603", 0.01, extra_cols={"legacy_col": 0.01})
        report = reconcile_panel(store.root, ref_dir, ["20260603"], mode="strict", n_jobs=1)
        assert report["verdict"]["pass"] is False
        assert any("cols_only_ref" in item for item in report["verdict"]["structural_offenders"])

    def test_attrib_mode_structure_checked_too(self, store_and_ref):
        """归因门同样追加结构合法性判定。"""
        store, ref_dir = store_and_ref
        _write_day(store, "20260603", 0.01)
        _ref_day(ref_dir, "20260603", 0.01, drop_cols=("roe_dt",))
        report = reconcile_panel(store.root, ref_dir, ["20260603"], mode="attrib", n_jobs=1)
        assert report["verdict"]["pass"] is False
        assert any("roe_dt" in item for item in report["verdict"]["structural_offenders"])


class TestFullWindowDenominator:
    """R3-04 整改：over_share 分母 = 全窗口该列实际可比日的可比行数合计。"""

    def test_over_share_uses_full_window(self, store_and_ref):
        """反例：20 日 × 2 行仅 1 日差异 ⇒ 分母含零差异日（40 行而非差异日 2 行）。"""
        store, ref_dir = store_and_ref
        dates = [f"202606{d:02d}" for d in range(1, 21)]  # 20260601~20260620
        for date in dates:
            _write_day(store, date, 0.01)
            _ref_day(ref_dir, date, 0.01, overrides={"ret_1": 0.02} if date == "20260610" else None)
        report = reconcile_panel(store.root, ref_dir, dates, mode="strict", n_jobs=1)
        ret1 = report["columns"]["ret_1"]
        assert ret1["over_rows"] == 2
        assert ret1["total_rows"] == 40  # 20 日 × 2 行全窗口分母
        assert ret1["over_share"] == pytest.approx(0.05)

    def test_denominator_counts_only_comparable_days(self, store_and_ref):
        """列在部分日不可比（不在当日 common_cols）⇒ 分母只计实际可比日。"""
        store, ref_dir = store_and_ref
        _write_day(store, "20260603", 0.01)
        _write_day(store, "20260604", 0.01)
        _ref_day(ref_dir, "20260603", 0.01, overrides={"roe_dt": 0.02})
        _ref_day(ref_dir, "20260604", 0.01, drop_cols=("roe_dt",))  # 当日 roe_dt 不可比
        report = reconcile_panel(
            store.root, ref_dir, ["20260603", "20260604"], mode="strict", n_jobs=1
        )
        roe = report["columns"]["roe_dt"]
        assert roe["over_rows"] == 2
        assert roe["total_rows"] == 2  # 只计 20260603 可比日
        assert roe["over_share"] == pytest.approx(1.0)


class TestClassifyBoundaries:
    """归因门类判定（diff_days = [(date, over_rows, day_rows, over_nan_rows)]）。

    L1 = 全部差异日 ≤2012 年底；L1+数据态孤立单点 = L1 期 + 期外零星日（单日 ≤1%、
    期外累计 ≤0.1%）归因不越界；D-04①/③/④ 收紧（③/④ 基础列纯 NaN 型 + 登记窗
    [20161201, 20260702]，① 差异日 ≥20260101；派生连锁单列免纯 NaN 要求）；
    D-04② 命中窗口外 ⇒ 越界；D-05 名命中（skew/kurt）在归因门按日期模式归类。
    """

    def test_l1_zone_diff_passes(self):
        klass, violated = _classify_attrib_column("ret_5", [("20120105", 100, 5000, 0)], 0.01)
        assert klass.startswith("L1") and not violated

    def test_l1_plus_scattered_drift_passes(self):
        # L1 期 5000 行 + 期外 2 日各 1 行：单日 1/5000=0.02% ≤1%、期外累计 2/5002≈0.04% ≤0.1% ⇒ 归因
        days = [("20120105", 5000, 5000, 0)] + [(d, 1, 5000, 0) for d in ("20220513", "20220516")]
        klass, violated = _classify_attrib_column("downside_vol_20", days, 0.01)
        assert "孤立单点" in klass and not violated

    def test_l1_plus_dense_late_days_violate(self):
        # 期外单日 200/5000=4% >1% ⇒ 越界
        days = [("20120105", 100, 5000, 0), ("20220513", 200, 5000, 0)]
        _, violated = _classify_attrib_column("downside_vol_20", days, 0.01)
        assert violated

    def test_d04_family_pure_nan_in_window_passes(self):
        assert not _classify_attrib_column("unlock_ratio", [("20260601", 10, 5000, 0)], 0.0)[
            1
        ]  # D-12
        # D-04③/④：纯 NaN 型（over_nan == over）+ 登记窗内 ⇒ 放行
        days = [("20260601", 10, 5000, 10)]
        assert not _classify_attrib_column("dividend_payout_ratio", days, 0.0)[1]  # D-04③
        assert not _classify_attrib_column("fund_hold_ratio", days, 0.0)[1]  # D-04③ fund_ 前缀
        assert not _classify_attrib_column("rzye_chg_5", days, 0.0)[1]  # D-04④

    def test_d04_3_value_misalignment_violates(self):
        """值错位型（over_nan < over，max_abs>0）⇒ 越界（P1-2：D-04③ 不再无条件放行）。"""
        _, violated = _classify_attrib_column(
            "dividend_payout_ratio", [("20200105", 10, 5000, 5)], 0.1
        )
        assert violated

    def test_d04_3_outside_window_violates(self):
        """纯 NaN 型但差异日越出登记窗 [20161201, 20260702] ⇒ 越界。"""
        assert _classify_attrib_column("dividend_payout_ratio", [("20150105", 10, 5000, 10)], 0.0)[
            1
        ]
        assert _classify_attrib_column("cf_nm", [("20260703", 10, 5000, 10)], 0.0)[1]

    def test_d04_1_date_lower_bound(self):
        """D-04①：差异日全部 ≥20260101 ⇒ 放行；任一更早 ⇒ 越界。"""
        assert not _classify_attrib_column("lhb_net_amount", [("20260201", 10, 5000, 0)], 0.0)[1]
        assert _classify_attrib_column("lhb_net_amount", [("20251231", 10, 5000, 0)], 0.0)[1]

    def test_d04_derived_single_column_exempts_pure_nan(self):
        """派生连锁单列阶段：同族日期规则、免除纯 NaN 要求（实证 max_abs=3.4）。"""
        klass, violated = _classify_attrib_column(
            "zscore_dividend_payout_ratio_sz", [("20200105", 10, 5000, 0)], 3.4
        )
        assert klass.startswith("D-04③") and "派生连锁" in klass and not violated

    def test_d04_2_date_window(self):
        assert not _classify_attrib_column("downside_corr_20", [("20240117", 10, 5000, 0)], 0.5)[1]
        assert _classify_attrib_column("downside_corr_20", [("20250101", 10, 5000, 0)], 0.5)[1]

    def test_d05_named_columns_classified_by_date_pattern(self):
        # skew/kurt 差异日全在 L1 期 ⇒ L1 归因、不判越界（参照侧 D-05 已修复）
        klass, violated = _classify_attrib_column("skewness_20", [("20120105", 100, 5000, 0)], 0.01)
        assert klass.startswith("L1") and not violated


class TestAttribSpecial:
    """D-14 专项登记（skew/kurt 数值尾、zscore_macd EMA 长尾、downside_corr 血缘块）
    + P1-3 三上限（天数 / 累计 over / max|Δ|，越任一 ⇒ 越界）。"""

    def test_skew_kurt_numeric_tail_attributed(self):
        # L1 主体 + 晚日 ≤25 股且 ≤1.5%（2015 停牌簇量级）⇒ 归因
        days = [("20120201", 1680, 1683, 0), ("20150522", 23, 1752, 0), ("20260122", 1, 5054, 0)]
        klass, violated = _classify_attrib_column("kurtosis_20", days, 35.0)
        assert "D-05" in klass and not violated

    def test_skew_kurt_dense_late_day_violates(self):
        days = [("20120201", 1680, 1683, 0), ("20150522", 200, 1752, 0)]
        _, violated = _classify_attrib_column("kurtosis_20", days, 1.0)
        assert violated

    def test_zscore_macd_ema_tail_and_d11_attributed(self):
        # EMA 长尾（≤2013-09，允许密集）+ D-11 窗内复牌簇 + 稀疏孤立单点 ⇒ 归因
        days = [
            ("20130729", 1809, 2049, 0),
            ("20130902", 23, 2066, 0),
            ("20220505", 3058, 3872, 0),
            ("20220826", 1, 4093, 0),
        ]
        klass, violated = _classify_attrib_column("zscore_macd_hist_sz", days, 14.0)
        assert "EMA" in klass and not violated

    def test_zscore_macd_dense_outside_bounds_violates(self):
        _, violated = _classify_attrib_column("zscore_macd_hist", [("20140105", 100, 5000, 0)], 1.0)
        assert violated

    def test_downside_corr_lineage_block_attributed(self):
        # 2013-09/10 血缘块 + D-04② 窗 + 2021-12 稀疏单点 ⇒ 归因
        days = [
            ("20120104", 2014, 2059, 0),
            ("20131017", 2014, 2059, 0),
            ("20240117", 10, 4000, 0),
            ("20211216", 1, 3705, 0),
        ]
        klass, violated = _classify_attrib_column("downside_corr_20", days, 1.0)
        assert "血缘" in klass and not violated

    def test_downside_corr_dense_outside_bounds_violates(self):
        _, violated = _classify_attrib_column("downside_corr_20", [("20140105", 100, 5000, 0)], 1.0)
        assert violated

    def test_skew_max_abs_cap_violates(self):
        """三上限之 max|Δ|：skewness_20 上限 5.0（实证 3.121），6.0 ⇒ 越界。"""
        days = [("20120201", 100, 1683, 0), ("20150522", 3, 1752, 0)]
        _, violated = _classify_attrib_column("skewness_20", days, 6.0)
        assert violated

    def test_kurt_late_days_cap_violates(self):
        """三上限之天数：kurtosis_20 late 天数上限 750（实证 658），751 ⇒ 越界。"""
        days = [("20120201", 100, 1683, 0)] + [(f"2014{i:04d}", 1, 5000, 0) for i in range(751)]
        _, violated = _classify_attrib_column("kurtosis_20", days, 1.0)
        assert violated

    def test_zscore_macd_cumulative_over_cap_violates(self):
        """三上限之累计 over：zscore_macd 上限 600000（实证 471326），80 万 ⇒ 越界。"""
        days = [(f"2013{i:04d}", 2000, 5000, 0) for i in range(400)]  # 全部 ≤20130930 合法
        _, violated = _classify_attrib_column("zscore_macd_hist", days, 1.0)
        assert violated

    def test_downside_corr_max_abs_cap_violates(self):
        """三上限之 max|Δ|：downside_corr_20 上限 1.5（实证 1.116），2.0 ⇒ 越界。"""
        days = [("20120104", 100, 2059, 0), ("20240117", 10, 4000, 0)]
        _, violated = _classify_attrib_column("downside_corr_20", days, 2.0)
        assert violated


class TestStrictClassify:
    """严格门类判定（出口 = 残差 100% 归因：L1/D-04/D-11/D-12 豁免/纯孤立单点外 ⇒ 未归因）。"""

    def test_d12_family_any_date_attributed(self):
        for col in ("days_to_unlock", "unlock_ratio", "unlock_risk_flag"):
            klass, unattributed = _classify_strict_column(col, [("20140103", 5000, 5000, 0)], 89.8)
            assert klass.startswith("D-12") and not unattributed

    def test_d04_family_pure_nan_in_window_attributed(self):
        days = [("20260601", 10, 5000, 10)]
        assert not _classify_strict_column("dividend_payout_ratio", days, 0.0)[1]
        assert not _classify_strict_column("rzye_chg_5", days, 0.0)[1]

    def test_d04_value_misalignment_unattributed(self):
        """值错位型（max_abs>0、over_nan<over）⇒ 未归因（P1-2）。"""
        _, unattributed = _classify_strict_column(
            "dividend_payout_ratio", [("20200105", 10, 5000, 5)], 0.1
        )
        assert unattributed

    def test_d04_outside_window_unattributed(self):
        """窗外日期 ⇒ 未归因。"""
        assert _classify_strict_column("dividend_payout_ratio", [("20150105", 10, 5000, 10)], 0.0)[
            1
        ]
        assert _classify_strict_column("ocf_to_profit", [("20260703", 10, 5000, 10)], 0.0)[1]

    def test_d04_1_date_lower_bound(self):
        assert not _classify_strict_column("lhb_net_amount", [("20260201", 10, 5000, 0)], 0.0)[1]
        assert _classify_strict_column("block_discount_days_10d", [("20250101", 10, 5000, 0)], 0.0)[
            1
        ]

    def test_d11_window_anchor_sensitive(self):
        # 锚定影响窗内（2017-01 / 2022-01 起 ~9 个月）长记忆列 ⇒ D-11 归因
        klass, unattributed = _classify_strict_column(
            "macd_hist", [("20170205", 10, 5000, 0)], 0.01
        )
        assert klass.startswith("D-11") and not unattributed
        klass, unattributed = _classify_strict_column(
            "zscore_macd_hist_sz", [("20220513", 10, 5000, 0)], 0.08
        )
        assert klass.startswith("D-11") and not unattributed
        # 复牌股 EMA 收敛尾（实证 002260.SZ 2022-05 簇）可延伸至窗末
        klass, unattributed = _classify_strict_column("macd_hist", [("20220902", 1, 3894, 0)], 0.02)
        assert klass.startswith("D-11") and not unattributed

    def test_d11_window_dense_any_density_passes(self):
        """R2-1 语义：D-11 窗内任意密度放行（实证 zscore_macd_hist_sz 窗内
        23857/51327≈46% 密集差异合法）——只要 max|Δ| ≤ 5.0（实证窗内最大 2.326）。"""
        klass, unattributed = _classify_strict_column(
            "zscore_macd_hist_sz", [("20220513", 23857, 51327, 0)], 2.326
        )
        assert klass.startswith("D-11") and not unattributed

    def test_d11_window_over_magnitude_unattributed(self):
        """R2-1：D-11 超幅 >5.0 ⇒ 拒（未归因）。"""
        _, unattributed = _classify_strict_column(
            "zscore_macd_hist_sz", [("20220513", 100, 51327, 0)], 6.0
        )
        assert unattributed

    def test_d11_window_companion_sparse_point(self):
        # 窗内非长记忆列的稀疏微量单点（复牌股锚定事件伴随）⇒ D-11 伴随稀疏单点
        klass, unattributed = _classify_strict_column("amplitude", [("20220505", 1, 3872, 0)], 0.0)
        assert klass.startswith("D-11") and not unattributed

    def test_d11_window_dense_non_anchor_unattributed(self):
        # 同窗内但非长记忆列且密集差异（4% > 1%）⇒ 不落 D-11，未归因
        _, unattributed = _classify_strict_column("ret_1", [("20170205", 200, 5000, 0)], 0.01)
        assert unattributed

    def test_d11_outside_window_unattributed(self):
        # 窗外非孤立单点量级（30 行 > 纯单点阈 20 行）的长记忆列差异 ⇒ 未归因
        _, unattributed = _classify_strict_column("macd_hist", [("20180105", 30, 5000, 0)], 0.01)
        assert unattributed

    def test_l1_zone_attributed(self):
        klass, unattributed = _classify_strict_column("ret_5", [("20120105", 100, 5000, 0)], 0.01)
        assert klass.startswith("L1") and not unattributed

    def test_l1_plus_scattered_attributed(self):
        days = [("20120105", 5000, 5000, 0)] + [(d, 1, 5000, 0) for d in ("20220513", "20220516")]
        klass, unattributed = _classify_strict_column("downside_vol_20", days, 0.01)
        assert "孤立单点" in klass and not unattributed

    def test_pure_scatter_point_attributed(self):
        # 百分位秩单步翻转：全日 1 行稀疏、总量 ≤20、日数 ≤10、max|Δ| ≤0.05 ⇒ 纯孤立单点
        days = [(d, 1, 4000, 0) for d in ("20191119", "20200902", "20231212")]
        klass, unattributed = _classify_strict_column("vol_regime_percentile", days, 0.008)
        assert "孤立单点" in klass and not unattributed

    def test_pure_scatter_point_over_magnitude_unattributed(self):
        # 单点但 max|Δ| > 0.05 ⇒ 超幅一律调查
        _, unattributed = _classify_strict_column(
            "vol_regime_percentile", [("20191119", 1, 4000, 0)], 0.5
        )
        assert unattributed


class TestDerivedChain:
    """P1-2 派生连锁跨列一致性（合并后第二阶段）：母列须在差异列集合中、
    且派生列差异日 ⊆ 母列差异日；不满足 ⇒ 改判未归因（strict）/ 越界（attrib）。"""

    def test_parent_missing_fails_attrib(self):
        merged = _merge_and_judge(
            [
                _day_result(
                    "20200105",
                    {"zscore_dividend_payout_ratio": _stat(10, over_nan=0, max_abs=3.4)},
                )
            ],
            "attrib",
        )
        agg = merged["columns"]["zscore_dividend_payout_ratio"]
        assert "派生连锁" in agg["attribution"]
        assert agg["boundary_violation"] is True

    def test_parent_missing_fails_strict(self):
        merged = _merge_and_judge(
            [
                _day_result(
                    "20200105",
                    {"zscore_dividend_payout_ratio": _stat(10, over_nan=0, max_abs=3.4)},
                )
            ],
            "strict",
        )
        assert merged["columns"]["zscore_dividend_payout_ratio"]["unattributed"] is True

    def test_days_subset_of_parent_passes(self):
        """实证形态：zscore_cf_nm 840 天 ⊆ cf_nm 840 天 ⇒ 放行。"""
        days = [
            _day_result(
                "20200105",
                {
                    "dividend_payout_ratio": _stat(10, over_nan=10),
                    "zscore_dividend_payout_ratio": _stat(10, over_nan=0, max_abs=3.4),
                },
            ),
            _day_result("20200106", {"dividend_payout_ratio": _stat(5, over_nan=5)}),
        ]
        merged = _merge_and_judge(days, "strict")
        assert merged["columns"]["dividend_payout_ratio"]["unattributed"] is False
        assert merged["columns"]["zscore_dividend_payout_ratio"]["unattributed"] is False

    def test_days_not_subset_of_parent_fails(self):
        """派生列差异日越出母列差异日 ⇒ 改判。"""
        days = [
            _day_result("20200105", {"dividend_payout_ratio": _stat(10, over_nan=10)}),
            _day_result(
                "20200107",
                {"zscore_dividend_payout_ratio": _stat(10, over_nan=0, max_abs=3.4)},
            ),
        ]
        merged = _merge_and_judge(days, "strict")
        assert merged["columns"]["dividend_payout_ratio"]["unattributed"] is False
        assert merged["columns"]["zscore_dividend_payout_ratio"]["unattributed"] is True


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
