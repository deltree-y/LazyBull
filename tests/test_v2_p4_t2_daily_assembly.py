# -*- coding: utf-8 -*-
"""v2 P4 T2 纸面宿主当日数据组装 + 量化就绪门（D4）测试。

全合成数据 + ``tmp_path``（真实 PanelDataStore 落在临时根；Storage/DataLoader
落在临时根），禁止读真实 ``data/``。就绪门 6 条逐条覆盖正例 + 反例：

- 门 1（当日 panel 分区存在）：全就绪正例 / 缺日反例；
- 门 2（行数带 0.9~1.1）：全就绪 + 下界恰过正例 / 掉出带外反例；
- 门 3（逐列来源清单，F6 显式来源制）：全 panel 正例 + 派生族声明正例 /
  未覆盖列反例 / 未登记哨兵列反例 / 哨兵列登记在场正例 / 派生形态无声明反例；
- 门 4（缺失率 0.6）：阈值恰过正例 / 超阈反例 / 期望列缺席反例（桩 store）
  / 59% vs 61% 边界锁定（P2-4 登记）/ 事件 freshness 闭包在场判定（F4）；
- 门 5（派生族一致性）：双空 + 启用一致正例 / 双向不一致反例 / 未知族 fail-fast；
- 门 6（clean/daily 当日存在）：全就绪正例 / 缺当日 + 全空反例 /
  sell_price=open 追加校验（F9）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from src.lazybull.data import DataLoader, Storage
from src.lazybull.v2.common.freshness import FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY
from src.lazybull.v2.common.types import FeatureQuery, PanelFrame, TradeDate
from src.lazybull.v2.core.signal.ml_signal import MLSignal
from src.lazybull.v2.hosts.paper.daily_assembly import (
    COLUMN_MISSING_RATE_MAX,
    PRICE_DESIRED_COLUMNS,
    SELECTION_FILTER_COLUMNS,
    SPARSE_TOLERATED_COLUMNS,
    _check_missing_rates,
    assemble_daily_data,
    compare_cs_infer_vs_panel,
)
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.store.panel_builder import bootstrap_manifest

# 12 只合成股票（行数带边界用例需要 >5 行）
_CODES = [f"0000{i:02d}.SZ" for i in range(12)]
#: 合成模型列（全部落在 column_groups core 族，bootstrap 后必然登记）
_MODEL_COLUMNS = ["ret_1", "ret_5", "ret_20", "bp", "turnover_rate"]
#: 30 个合成交易日（2026-07-06 起的工作日；structural 行域段内）
_CALENDAR = pd.bdate_range("2026-07-06", periods=30).strftime("%Y%m%d").tolist()
_TARGET = _CALENDAR[-1]


def _panel_day_frame(
    day: str, n_rows: int = 5, nan_column: str | None = None, nan_count: int = 0
) -> pd.DataFrame:
    """单日 core 族 panel 帧（模型列 + 选股过滤列，全非空；可按列注入 NaN）。"""
    n = n_rows
    data = {
        "trade_date": [day] * n,
        # n <= 12 时与 _CODES[:n] 逐位一致；n > 12（缺失率边界用例）按同形态扩展
        "ts_code": [f"{i:06d}.SZ" for i in range(n)],
        "ret_1": np.linspace(0.10, 0.14, n),
        "ret_5": np.linspace(0.20, 0.24, n),
        "ret_20": np.linspace(-0.10, -0.06, n),
        "bp": np.linspace(0.01, 0.05, n),
        "turnover_rate": np.linspace(1.0, 1.4, n),
        "amount_ma20": np.full(n, 60000.0),
        "total_mv": np.full(n, 800000.0),
        "sw_l1_code": ["801780"] * n,
    }
    if nan_column is not None:
        values = np.linspace(0.01, 0.05, n)
        values[:nan_count] = np.nan
        data[nan_column] = values
    return pd.DataFrame(data)


def _make_store(
    tmp_path,
    *,
    skip_days: frozenset | set = frozenset(),
    day_rows: dict | None = None,
    nan_spec: tuple | None = None,
) -> PanelDataStore:
    """临时根真实 PanelDataStore：bootstrap manifest + 逐日写 core 分区。

    ``nan_spec`` = (列名, NaN 行数)，仅作用于当日（``_TARGET``）。
    """
    store = PanelDataStore(tmp_path / "store")
    bootstrap_manifest(store)
    rows_by_day = dict(day_rows or {})
    for day in _CALENDAR:
        if day in skip_days:
            continue
        nan_column, nan_count = nan_spec if (nan_spec and day == _TARGET) else (None, 0)
        frame = _panel_day_frame(
            day, n_rows=rows_by_day.get(day, 5), nan_column=nan_column, nan_count=nan_count
        )
        store.append_features(TradeDate.from_str(day), "core", frame)
    return store


def _make_loader(
    tmp_path,
    *,
    skip_days: frozenset | set = frozenset(),
    drop_columns: Sequence[str] = (),
    open_nan_days: frozenset | set = frozenset(),
) -> DataLoader:
    """临时根 DataLoader：合成 clean/daily 全 14 个 replay 口径列。

    ``drop_columns`` 从行情帧整列剔除（F9 缺 open 列用例）；``open_nan_days``
    内日期的 open 列写全 NaN（F9 当日 open 全 NaN 用例）。
    """
    storage = Storage(root_path=str(tmp_path / "io"))
    daily_dir = storage.clean_path / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    n = 5
    for day in _CALENDAR:
        if day in skip_days:
            continue
        frame = pd.DataFrame(
            {
                "trade_date": [day] * n,
                "ts_code": _CODES[:n],
                "close": np.full(n, 10.0),
                "close_adj": np.full(n, 10.0),
                "open": np.full(n, np.nan if day in open_nan_days else 9.9),
                "open_adj": np.full(n, 9.9),
                "is_suspended": [0] * n,
                "is_limit_up": [0] * n,
                "is_limit_down": [0] * n,
                "vol": np.full(n, 1000.0),
                "pct_chg": np.full(n, 0.5),
                "is_st": [0] * n,
                "list_days": [1000] * n,
                "tradable": [1] * n,
                "unused_extra_col": np.full(n, -1.0),  # 验证裁剪口径（按存在交集）
            }
        ).drop(columns=list(drop_columns))
        frame.to_parquet(daily_dir / f"{day[:4]}-{day[4:6]}-{day[6:8]}.parquet", index=False)
    return DataLoader(storage=storage)


def _make_store_with_extra_columns(tmp_path, extra_columns: dict) -> PanelDataStore:
    """登记（若未登记）并逐日写入额外列的 store（F4 freshness / F6 哨兵在场正例用）。

    ``extra_columns`` = {列名: 填充值}；已在 column_groups 单一来源登记的列按其
    已登记族写入对应族分区（族不一致会被写前校验拒绝），未登记列登记进 core 族。
    """
    store = PanelDataStore(tmp_path / "store")
    bootstrap_manifest(store)
    registered = set(store.manifest.snapshot()["columns"])
    for col in extra_columns:
        if col not in registered:
            store.manifest.register_column(col, "core", source="test")
    group_of = {col: str(store.manifest.column_group(col)) for col in extra_columns}
    for day in _CALENDAR:
        core_frame = _panel_day_frame(day)
        side_columns: dict[str, dict] = {}
        for col, value in extra_columns.items():
            if group_of[col] == "core":
                core_frame[col] = value
            else:
                side_columns.setdefault(group_of[col], {})[col] = value
        store.append_features(TradeDate.from_str(day), "core", core_frame)
        for group, columns in side_columns.items():
            n = len(core_frame)
            frame = pd.DataFrame(
                {
                    "trade_date": [day] * n,
                    "ts_code": core_frame["ts_code"].tolist(),
                    **{col: [value] * n for col, value in columns.items()},
                }
            )
            store.append_features(TradeDate.from_str(day), group, frame)
    return store


def _assemble(store, loader, **overrides):
    """默认全就绪形态的组装调用（覆写参数走关键字）。"""
    params = {
        "date": _TARGET,
        "store": store,
        "loader": loader,
        "trading_dates": list(_CALENDAR),
        "feature_columns": list(_MODEL_COLUMNS),
        "train_params": {},
    }
    params.update(overrides)
    return assemble_daily_data(**params)


class _StubStore:
    """最小 store 桩：manifest 登记面与 load_features 返回帧可分别构造。

    用于真实 store 不可达的分支（load_features 的 reindex 语义保证请求列
    必然在场，「期望列缺席」只能经桩构造）。
    """

    def __init__(self, flat_frame: pd.DataFrame, registered: set) -> None:
        self._flat_frame = flat_frame
        self._registered = set(registered)

    @property
    def manifest(self):
        registered = self._registered
        return SimpleNamespace(snapshot=lambda: {"columns": {c: {} for c in registered}})

    def load_features(self, query: FeatureQuery) -> PanelFrame:
        flat = self._flat_frame
        return PanelFrame(df=flat.set_index(["trade_date", "ts_code"]), manifest_version="1")


def _flat_all_days(drop_columns: Sequence[str] = ()) -> pd.DataFrame:
    frames = [_panel_day_frame(day).drop(columns=list(drop_columns)) for day in _CALENDAR]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 全就绪（门 1~6 联合正例）
# ---------------------------------------------------------------------------


class TestAllReady:
    def test_all_ready_ok_and_fields_filled(self, tmp_path):
        assembly = _assemble(_make_store(tmp_path), _make_loader(tmp_path))
        report = assembly.readiness
        assert report.ok is True
        assert report.gaps == []
        assert assembly.date == _TARGET == report.date
        # 门 1/2：当日行数与行数带（20 日中位数 5 → [4.5, 5.5]）
        assert report.row_count == 5
        assert report.row_count_band == pytest.approx((4.5, 5.5))
        # 门 4：缺失率覆盖模型 panel 列 + 选股过滤列，全 0
        assert set(report.column_missing_rates) == set(_MODEL_COLUMNS) | set(
            SELECTION_FILTER_COLUMNS
        )
        assert all(rate == 0.0 for rate in report.column_missing_rates.values())
        assert report.uncovered_columns == []
        # 门 5：双空一致
        assert report.derived_family_note.endswith("核对结论=一致")
        # 特征：按日平铺、裁到近 10 个交易日（含当日）、含 ts_code 与过滤列
        assert sorted(assembly.features_by_date) == sorted(_CALENDAR[-10:])
        day = assembly.features_by_date[_TARGET]
        assert len(day) == 5
        assert "ts_code" in day.columns
        assert set(day["trade_date"]) == {_TARGET}
        assert set(_MODEL_COLUMNS) | set(SELECTION_FILTER_COLUMNS) <= set(day.columns)
        # 行情：窗口含当日、trade_date 为 YYYYMMDD 字符串、按 replay 口径裁剪
        assert _TARGET in set(assembly.price_data["trade_date"])
        assert set(assembly.price_data.columns) <= set(PRICE_DESIRED_COLUMNS)
        assert "close" in assembly.price_data.columns
        assert "unused_extra_col" not in assembly.price_data.columns


# ---------------------------------------------------------------------------
# 门 1：当日 panel 分区存在
# ---------------------------------------------------------------------------


class TestDayPartition:
    def test_missing_day_partition_fails(self, tmp_path):
        store = _make_store(tmp_path, skip_days={_TARGET})
        assembly = _assemble(store, _make_loader(tmp_path))
        report = assembly.readiness
        assert report.ok is False
        assert any("当日 panel 分区缺失" in gap for gap in report.gaps)
        assert _TARGET not in assembly.features_by_date
        assert report.row_count == 0
        # 行数带基准仍存在（窗口内其余 19 日），行数 0 掉出带外同时登记
        assert report.row_count_band == pytest.approx((4.5, 5.5))


# ---------------------------------------------------------------------------
# 门 2：行数带
# ---------------------------------------------------------------------------


class TestRowBand:
    def test_row_count_out_of_band_fails(self, tmp_path):
        store = _make_store(tmp_path, day_rows={_TARGET: 1})
        assembly = _assemble(store, _make_loader(tmp_path))
        report = assembly.readiness
        assert report.ok is False
        assert report.row_count == 1
        assert any("行数带" in gap for gap in report.gaps)

    def test_row_count_at_lower_edge_passes(self, tmp_path):
        # 其余日 10 行 ⇒ 中位数 10、带 [9.0, 11.0]；当日 9 行 = 下界恰通过
        day_rows = {day: 10 for day in _CALENDAR[:-1]}
        day_rows[_TARGET] = 9
        store = _make_store(tmp_path, day_rows=day_rows)
        assembly = _assemble(store, _make_loader(tmp_path))
        report = assembly.readiness
        assert report.row_count == 9
        assert report.row_count_band == pytest.approx((9.0, 11.0))
        assert report.ok is True


# ---------------------------------------------------------------------------
# 门 3：逐列来源清单
# ---------------------------------------------------------------------------


class TestColumnSources:
    def test_uncovered_column_fails(self, tmp_path):
        assembly = _assemble(
            _make_store(tmp_path),
            _make_loader(tmp_path),
            feature_columns=[*_MODEL_COLUMNS, "mystery_factor_x"],
        )
        report = assembly.readiness
        assert report.ok is False
        assert any("未覆盖列" in gap and "mystery_factor_x" in gap for gap in report.gaps)
        assert "mystery_factor_x" in report.uncovered_columns

    def test_derived_declared_family_without_producer_fails(self, tmp_path):
        assembly = _assemble(
            _make_store(tmp_path),
            _make_loader(tmp_path),
            feature_columns=[*_MODEL_COLUMNS, "ht_unregistered_probe"],
            train_params={"enable_holdertrade_features": True},
            derived_families_provided=["holdertrade"],
        )
        assert assembly.readiness.ok is False
        assert "ht_unregistered_probe" in assembly.readiness.uncovered_columns
        assert any("无明确生产者" in gap for gap in assembly.readiness.gaps)

    def test_derived_exact_producer_with_actual_output_passes(self, tmp_path):
        def produce(date, frame):
            assert date == _TARGET
            return pd.Series(np.arange(len(frame), dtype=float), index=frame.index)

        assembly = _assemble(
            _make_store(tmp_path),
            _make_loader(tmp_path),
            feature_columns=[*_MODEL_COLUMNS, "ht_synthetic_x"],
            train_params={"enable_holdertrade_features": True},
            derived_families_provided=["holdertrade"],
            derived_producers={"ht_synthetic_x": produce},
        )
        assert assembly.readiness.ok is True
        assert assembly.readiness.uncovered_columns == []
        assert assembly.features_by_date[_TARGET]["ht_synthetic_x"].tolist() == [0, 1, 2, 3, 4]

    def test_derived_producer_without_actual_output_fails(self, tmp_path):
        assembly = _assemble(
            _make_store(tmp_path),
            _make_loader(tmp_path),
            feature_columns=[*_MODEL_COLUMNS, "ht_synthetic_x"],
            train_params={"enable_holdertrade_features": True},
            derived_families_provided=["holdertrade"],
            derived_producers={"ht_synthetic_x": lambda date, frame: None},
        )
        assert assembly.readiness.ok is False
        assert "ht_synthetic_x" in assembly.readiness.uncovered_columns
        assert any("未在最终输入实际产出" in gap for gap in assembly.readiness.gaps)

    def test_derived_shape_column_without_declared_family_fails(self, tmp_path):
        # F6 显式来源制反例：派生形态列无宿主声明（模型未启用、宿主未提供）
        # ⇒ 运行时无产出者 ⇒ 未覆盖（旧行为按形态名放行造成假就绪，已废止）
        assembly = _assemble(
            _make_store(tmp_path),
            _make_loader(tmp_path),
            feature_columns=[*_MODEL_COLUMNS, "ht_synthetic_x"],
        )
        report = assembly.readiness
        assert report.ok is False
        assert "ht_synthetic_x" in report.uncovered_columns
        assert any("无声明来源" in gap and "ht_synthetic_x" in gap for gap in report.gaps)

    def test_unknown_sentinel_column_fails(self, tmp_path):
        # F6 反例（评审实测形态）：unknown_schema_v999 无任何产出者，旧行为按
        # 哨兵形态名放行（ok=True 但当日报文实际无该列）；现按普通 store 列
        # 处理——未登记 ⇒ 未覆盖、未就绪且缺口含该列名
        assembly = _assemble(
            _make_store(tmp_path),
            _make_loader(tmp_path),
            feature_columns=[*_MODEL_COLUMNS, "unknown_schema_v999"],
        )
        report = assembly.readiness
        assert report.ok is False
        assert "unknown_schema_v999" in report.uncovered_columns
        assert any(
            "哨兵形态列未登记" in gap and "unknown_schema_v999" in gap for gap in report.gaps
        )

    def test_sentinel_column_registered_and_present_passes(self, tmp_path):
        # F6 正例：哨兵列按普通 store 列处理——登记 manifest 且逐日写入 ⇒
        # 归 panel 类纳入查询、实际读出 ⇒ 就绪（缺失率记录证明其进了查询列集）
        store = _make_store_with_extra_columns(tmp_path, {"cashflow_quality_schema_v2": 2.0})
        assembly = _assemble(
            store,
            _make_loader(tmp_path),
            feature_columns=[*_MODEL_COLUMNS, "cashflow_quality_schema_v2"],
        )
        report = assembly.readiness
        assert report.ok is True
        assert report.uncovered_columns == []
        assert report.column_missing_rates["cashflow_quality_schema_v2"] == pytest.approx(0.0)
        assert "cashflow_quality_schema_v2" in assembly.features_by_date[_TARGET].columns


# ---------------------------------------------------------------------------
# 门 4：每列当日缺失率阈值
# ---------------------------------------------------------------------------


class TestMissingRate:
    def test_column_missing_rate_over_threshold_fails(self, tmp_path):
        # 当日 bp 列 4/5 行 NaN = 0.8 > 0.6
        store = _make_store(tmp_path, nan_spec=("bp", 4))
        assembly = _assemble(store, _make_loader(tmp_path))
        report = assembly.readiness
        assert report.ok is False
        assert report.column_missing_rates["bp"] == pytest.approx(0.8)
        assert any("缺失率超阈" in gap and "bp" in gap for gap in report.gaps)
        # 其余列缺失率仍记录且未超阈
        assert report.column_missing_rates["ret_1"] == pytest.approx(0.0)

    def test_column_missing_rate_at_threshold_passes(self, tmp_path):
        # 当日 bp 列 3/5 行 NaN = 0.6 ≤ 阈值 0.6，恰通过
        store = _make_store(tmp_path, nan_spec=("bp", 3))
        assembly = _assemble(store, _make_loader(tmp_path))
        assert assembly.readiness.column_missing_rates["bp"] == pytest.approx(
            COLUMN_MISSING_RATE_MAX
        )
        assert assembly.readiness.ok is True

    def test_expected_column_absent_from_assembled_frame_fails(self, tmp_path):
        # 桩 store：bp 已登记但返回帧缺该列（真实 reindex 语义不可达的分支）
        registered = set(_MODEL_COLUMNS) | set(SELECTION_FILTER_COLUMNS)
        stub = _StubStore(_flat_all_days(drop_columns=["bp"]), registered)
        assembly = _assemble(stub, _make_loader(tmp_path))
        report = assembly.readiness
        assert report.ok is False
        assert "bp" in report.uncovered_columns
        assert any("未出现在当日组装结果" in gap for gap in report.gaps)
        # 在场列缺失率仍正常记录
        assert report.column_missing_rates["ret_1"] == pytest.approx(0.0)

    def test_unregistered_filter_column_fails(self, tmp_path):
        # 桩 store：sw_l1_code 未登记 manifest ⇒ 过滤列未覆盖（防静默跳过陷阱）
        registered = set(_MODEL_COLUMNS) | {"amount_ma20", "total_mv"}
        stub = _StubStore(_flat_all_days(), registered)
        assembly = _assemble(stub, _make_loader(tmp_path))
        report = assembly.readiness
        assert report.ok is False
        assert "sw_l1_code" in report.uncovered_columns
        assert any("选股过滤列未登记" in gap for gap in report.gaps)


# ---------------------------------------------------------------------------
# 门 4 例外：已知稀疏列白名单（M-P4-1 裁决 C，2026-10-29）
# ---------------------------------------------------------------------------


class TestSparseToleratedColumn:
    """``cons_target_upside`` 超阈放行，但全空 / 缺席仍失败（对齐旧链两级语义）。"""

    def test_whitelist_is_exact_registry(self):
        # 白名单为契约登记物：扩列必须同步规划 §23.8 与契约，禁止静默增长
        assert SPARSE_TOLERATED_COLUMNS == frozenset({"cons_target_upside"})

    def test_tolerated_column_over_threshold_passes_rate_still_recorded(self):
        frame = pd.DataFrame({"cons_target_upside": [np.nan] * 4 + [0.1]})
        rates, missing, gaps = _check_missing_rates(frame, ["cons_target_upside"], 0.6)
        assert gaps == []
        assert missing == []
        # 缺失率仍写入报告（证据留证口径不变）
        assert rates["cons_target_upside"] == pytest.approx(0.8)

    def test_non_tolerated_column_over_threshold_still_fails(self):
        frame = pd.DataFrame({"bp": [np.nan] * 4 + [0.01]})
        rates, _, gaps = _check_missing_rates(frame, ["bp"], 0.6)
        assert any("缺失率超阈" in gap and "bp" in gap for gap in gaps)
        assert rates["bp"] == pytest.approx(0.8)

    def test_tolerated_column_all_nan_still_fails(self):
        # 全空 = 真数据断链（旧链 ml_signal 亦硬拒），白名单不放行 100%
        frame = pd.DataFrame({"cons_target_upside": [np.nan] * 5})
        rates, _, gaps = _check_missing_rates(frame, ["cons_target_upside"], 0.6)
        assert any("cons_target_upside" in gap and "缺失率超阈" in gap for gap in gaps)
        assert rates["cons_target_upside"] == pytest.approx(1.0)

    def test_tolerated_column_absent_still_fails(self):
        frame = pd.DataFrame({"bp": [0.01] * 5})
        _, missing, gaps = _check_missing_rates(frame, ["cons_target_upside", "bp"], 0.6)
        assert missing == ["cons_target_upside"]
        assert any("未出现在当日组装结果" in gap for gap in gaps)

    def test_assemble_end_to_end_tolerated_column_ok(self, tmp_path):
        # cons_* 入选 ⇒ 事件 freshness 闭包同时要求 consensus_freshness_days 在场
        store = _make_store_with_extra_columns(
            tmp_path, {"cons_target_upside": 0.2, "consensus_freshness_days": 30.0}
        )
        assembly = _assemble(
            store,
            _make_loader(tmp_path),
            feature_columns=[*_MODEL_COLUMNS, "cons_target_upside"],
        )
        report = assembly.readiness
        assert report.ok is True
        assert report.column_missing_rates["cons_target_upside"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 门 5：派生族开关与模型 metadata 一致
# ---------------------------------------------------------------------------


class TestDerivedConsistency:
    def test_model_enabled_but_host_not_provided_fails(self, tmp_path):
        assembly = _assemble(
            _make_store(tmp_path),
            _make_loader(tmp_path),
            train_params={"enable_holdertrade_features": True},
        )
        report = assembly.readiness
        assert report.ok is False
        assert any("启用而宿主未提供" in gap for gap in report.gaps)
        assert report.derived_family_note.endswith("核对结论=不一致")
        assert "holdertrade" in report.derived_family_note

    def test_host_provided_but_model_not_enabled_fails(self, tmp_path):
        assembly = _assemble(
            _make_store(tmp_path),
            _make_loader(tmp_path),
            derived_families_provided=["top_inst"],
        )
        report = assembly.readiness
        assert report.ok is False
        assert any("宿主提供而模型 metadata 未启用" in gap for gap in report.gaps)

    def test_consistent_enabled_family_passes(self, tmp_path):
        assembly = _assemble(
            _make_store(tmp_path),
            _make_loader(tmp_path),
            feature_columns=[*_MODEL_COLUMNS, "rp_synthetic_x"],
            train_params={"enable_repurchase_features": True},
            derived_families_provided=["repurchase"],
            derived_producers={
                "rp_synthetic_x": lambda date, frame: pd.Series(1.0, index=frame.index)
            },
        )
        report = assembly.readiness
        assert report.ok is True
        assert report.derived_family_note.endswith("核对结论=一致")
        assert "repurchase" in report.derived_family_note

    def test_unknown_derived_family_raises(self, tmp_path):
        with pytest.raises(ValueError, match="派生族未知"):
            _assemble(
                _make_store(tmp_path),
                _make_loader(tmp_path),
                derived_families_provided=["martian"],
            )


# ---------------------------------------------------------------------------
# 门 6：clean/daily 当日分区存在
# ---------------------------------------------------------------------------


class TestPriceToday:
    def test_missing_price_today_fails(self, tmp_path):
        loader = _make_loader(tmp_path, skip_days={_TARGET})
        assembly = _assemble(_make_store(tmp_path), loader)
        report = assembly.readiness
        assert report.ok is False
        assert any("clean/daily 当日分区缺失" in gap for gap in report.gaps)
        # 窗口其余日仍在
        assert not assembly.price_data.empty

    def test_empty_price_data_fails(self, tmp_path):
        loader = _make_loader(tmp_path, skip_days=set(_CALENDAR))
        assembly = _assemble(_make_store(tmp_path), loader)
        report = assembly.readiness
        assert report.ok is False
        assert any("clean/daily 行情为空" in gap for gap in report.gaps)
        assert assembly.price_data.empty


# ---------------------------------------------------------------------------
# F4 评审处置：state_keep_event_decay 事件 freshness 推理闭包（门 4 追加段）
# ---------------------------------------------------------------------------


class TestFreshnessClosure:
    """freshness_strategy=state_keep_event_decay 时 freshness 辅助列的查询与在场判定。"""

    #: 合成元数据：事件衰减策略 + 含 forecast_type_score（forecast_freshness_days
    #: 的服务值列）的模型特征列
    _DECAY_PARAMS = {"freshness_strategy": FRESHNESS_STRATEGY_STATE_KEEP_EVENT_DECAY}
    _FEATURES = [*_MODEL_COLUMNS, "forecast_type_score"]

    def test_freshness_column_present_passes_and_is_queried(self, tmp_path):
        # (a) freshness 列在场（登记 manifest + 逐日写入）⇒ 就绪，且查询列集
        # 含 forecast_freshness_days（按日拆分结果含该列 = 经 store 查询读出；
        # 其缺失率不走门 4——freshness 列就绪判定唯一归属 4+ 在场段）
        store = _make_store_with_extra_columns(
            tmp_path, {"forecast_type_score": 1.5, "forecast_freshness_days": 10.0}
        )
        assembly = _assemble(
            store,
            _make_loader(tmp_path),
            feature_columns=self._FEATURES,
            train_params=self._DECAY_PARAMS,
        )
        report = assembly.readiness
        assert report.ok is True
        assert "forecast_freshness_days" in assembly.features_by_date[_TARGET].columns
        assert "forecast_freshness_days" not in report.column_missing_rates

    def test_freshness_column_missing_both_sides_fails(self, tmp_path):
        # (b) panel 未登记该列且 cs_infer 当日分区也不含 ⇒ 未就绪，缺口含
        # 列名与「事件 freshness 辅助列缺失」「state_keep_event_decay 推理必需」
        store = _make_store_with_extra_columns(tmp_path, {"forecast_type_score": 1.5})
        assembly = _assemble(
            store,
            _make_loader(tmp_path),
            feature_columns=self._FEATURES,
            train_params=self._DECAY_PARAMS,
        )
        report = assembly.readiness
        assert report.ok is False
        assert any(
            "事件 freshness 辅助列缺失" in gap
            and "state_keep_event_decay 推理必需" in gap
            and "forecast_freshness_days" in gap
            for gap in report.gaps
        )
        assert "forecast_freshness_days" in report.uncovered_columns

    @pytest.mark.parametrize("panel_has_nan_column", [False, True])
    def test_freshness_cs_infer_values_merge_into_model_input(self, tmp_path, panel_has_nan_column):
        """panel 未登记或全 NaN 时，cs_infer 值都必须进入最终推理帧并实际衰减。"""
        extra = {"forecast_type_score": 2.0}
        if panel_has_nan_column:
            extra["forecast_freshness_days"] = np.nan
        store = _make_store_with_extra_columns(tmp_path, extra)
        loader = _make_loader(tmp_path)
        cs_infer = pd.DataFrame(
            {
                "trade_date": [_TARGET] * 5,
                "ts_code": _CODES[:5],
                "forecast_type_score": [999.0] * 5,
                "forecast_freshness_days": [120.0] * 5,
            }
        )
        loader.storage.save_cs_train_day(cs_infer, _TARGET, subdir="cs_infer")
        params = dict(self._DECAY_PARAMS, event_freshness_half_life_days=120.0)
        assembly = _assemble(store, loader, feature_columns=self._FEATURES, train_params=params)
        assert assembly.readiness.ok is True
        day = assembly.features_by_date[_TARGET]
        assert day["forecast_freshness_days"].tolist() == [120.0] * 5
        assert day["forecast_type_score"].tolist() == [2.0] * 5, "只合入辅助列，不覆盖 panel 值列"
        signal = MLSignal.__new__(MLSignal)
        signal.metadata = {"train_params": params}
        decayed = signal._apply_serving_event_decay(day.copy())
        assert decayed["forecast_type_score"].tolist() == pytest.approx([1.0] * 5)

    def test_non_decay_strategy_does_not_require_freshness(self, tmp_path):
        cases = (
            {"freshness_strategy": "drop_all"},
            {"freshness_strategy": "state_keep_event_no_decay"},
        )
        for index, params in enumerate(cases):
            case_dir = tmp_path / f"case_{index}"
            case_dir.mkdir()
            store = _make_store_with_extra_columns(case_dir, {"forecast_type_score": 1.5})
            assembly = _assemble(
                store,
                _make_loader(case_dir),
                feature_columns=self._FEATURES,
                train_params=params,
            )
            assert assembly.readiness.ok is True

    def test_missing_strategy_uses_production_decay_default(self, tmp_path):
        store = _make_store_with_extra_columns(
            tmp_path, {"forecast_type_score": 2.0, "forecast_freshness_days": 45.0}
        )
        assembly = _assemble(
            store, _make_loader(tmp_path), feature_columns=self._FEATURES, train_params={}
        )
        assert assembly.readiness.ok is True
        day = assembly.features_by_date[_TARGET]
        signal = MLSignal.__new__(MLSignal)
        signal.metadata = {"train_params": {}}
        decayed = signal._apply_serving_event_decay(day.copy())
        assert decayed["forecast_type_score"].tolist() == pytest.approx([1.0] * 5)

    def test_missing_strategy_and_missing_freshness_fails(self, tmp_path):
        store = _make_store_with_extra_columns(tmp_path, {"forecast_type_score": 2.0})
        assembly = _assemble(
            store, _make_loader(tmp_path), feature_columns=self._FEATURES, train_params={}
        )
        assert assembly.readiness.ok is False
        assert "forecast_freshness_days" in assembly.readiness.uncovered_columns

    def test_ensemble_members_use_own_strategy_and_columns(self, tmp_path):
        store = _make_store_with_extra_columns(
            tmp_path, {"forecast_type_score": 2.0, "express_revenue_yoy": 4.0}
        )
        loader = _make_loader(tmp_path)
        cs_infer = pd.DataFrame(
            {
                "trade_date": [_TARGET] * 5,
                "ts_code": _CODES[:5],
                "express_freshness_days": [45.0] * 5,
            }
        )
        loader.storage.save_cs_train_day(cs_infer, _TARGET, subdir="cs_infer")
        assembly = _assemble(
            store,
            loader,
            feature_columns=[*_MODEL_COLUMNS, "forecast_type_score", "express_revenue_yoy"],
            train_params={},
            freshness_members=[
                (
                    ["forecast_type_score"],
                    {"freshness_strategy": "state_keep_event_no_decay"},
                ),
                (["express_revenue_yoy"], dict(self._DECAY_PARAMS)),
            ],
        )
        assert assembly.readiness.ok is True
        day = assembly.features_by_date[_TARGET]
        assert "forecast_freshness_days" not in day.columns, "A 臂 no-decay 不得放大依赖"
        assert day["express_freshness_days"].tolist() == [45.0] * 5
        signal_a = MLSignal.__new__(MLSignal)
        signal_a.metadata = {"train_params": {"freshness_strategy": "state_keep_event_no_decay"}}
        assert (
            signal_a._apply_serving_event_decay(day.copy())["forecast_type_score"].tolist()
            == [2.0] * 5
        )
        signal_b = MLSignal.__new__(MLSignal)
        signal_b.metadata = {"train_params": dict(self._DECAY_PARAMS)}
        decayed_b = signal_b._apply_serving_event_decay(day.copy())
        assert decayed_b["express_revenue_yoy"].tolist() == pytest.approx([2.0] * 5)

    def test_decay_strategy_without_event_value_columns_passes(self, tmp_path):
        # 闭包按「服务值列 ∩ 模型列」计算：模型列不含任何事件值列 ⇒ 闭包空，
        # 即使策略为 state_keep_event_decay 也不要求 freshness 列
        assembly = _assemble(
            _make_store(tmp_path),
            _make_loader(tmp_path),
            train_params=self._DECAY_PARAMS,
        )
        assert assembly.readiness.ok is True


# ---------------------------------------------------------------------------
# F9 评审处置：sell_price=open 时门 6 追加 open 列在场与非空校验
# ---------------------------------------------------------------------------


class TestSellPriceOpen:
    def test_sell_price_open_missing_open_column_fails(self, tmp_path):
        loader = _make_loader(tmp_path, drop_columns=["open"])
        assembly = _assemble(_make_store(tmp_path), loader, sell_price_type="open")
        report = assembly.readiness
        assert report.ok is False
        assert any("缺 open 列" in gap and "sell_price=open" in gap for gap in report.gaps)

    def test_sell_price_open_all_nan_today_fails(self, tmp_path):
        # 当日 open 全 NaN（窗口其余日正常）⇒ 未就绪
        loader = _make_loader(tmp_path, open_nan_days={_TARGET})
        assembly = _assemble(_make_store(tmp_path), loader, sell_price_type="open")
        report = assembly.readiness
        assert report.ok is False
        assert any("open 列全为 NaN" in gap and "sell_price=open" in gap for gap in report.gaps)

    def test_sell_price_open_normal_passes(self, tmp_path):
        assembly = _assemble(_make_store(tmp_path), _make_loader(tmp_path), sell_price_type="open")
        assert assembly.readiness.ok is True

    def test_sell_price_default_keeps_legacy_behavior(self, tmp_path):
        # 不传参（None）⇒ 与旧行为一致：缺 open 列不拦截
        loader = _make_loader(tmp_path, drop_columns=["open"])
        assembly = _assemble(_make_store(tmp_path), loader)
        assert assembly.readiness.ok is True

    def test_sell_price_close_keeps_legacy_behavior(self, tmp_path):
        # 显式 "close" ⇒ 与旧行为一致：缺 open 列不拦截
        loader = _make_loader(tmp_path, drop_columns=["open"])
        assembly = _assemble(_make_store(tmp_path), loader, sell_price_type="close")
        assert assembly.readiness.ok is True


# ---------------------------------------------------------------------------
# P2-4 登记：COLUMN_MISSING_RATE_MAX=0.6 门敏感性边界锁定（只锁测试，不改阈值）
# ---------------------------------------------------------------------------


class TestMissingRateBoundary:
    def test_missing_rate_59_percent_passes(self, tmp_path):
        # 当日 bp 缺失率 59/100 = 0.59 ≤ 0.6 ⇒ 该列不触发未就绪
        day_rows = {day: 100 for day in _CALENDAR}
        store = _make_store(tmp_path, day_rows=day_rows, nan_spec=("bp", 59))
        assembly = _assemble(store, _make_loader(tmp_path))
        report = assembly.readiness
        assert report.column_missing_rates["bp"] == pytest.approx(0.59)
        assert report.ok is True

    def test_missing_rate_61_percent_fails(self, tmp_path):
        # 当日 bp 缺失率 61/100 = 0.61 > 0.6 ⇒ 触发未就绪且缺口含该列
        day_rows = {day: 100 for day in _CALENDAR}
        store = _make_store(tmp_path, day_rows=day_rows, nan_spec=("bp", 61))
        assembly = _assemble(store, _make_loader(tmp_path))
        report = assembly.readiness
        assert report.column_missing_rates["bp"] == pytest.approx(0.61)
        assert report.ok is False
        assert any("缺失率超阈" in gap and "bp" in gap for gap in report.gaps)


# ---------------------------------------------------------------------------
# 只读冒烟辅助：compare_cs_infer_vs_panel
# ---------------------------------------------------------------------------


class TestCompareCsInferVsPanel:
    def test_structure_and_content(self, tmp_path):
        store = _make_store(tmp_path)
        loader = _make_loader(tmp_path)
        cs_infer = pd.DataFrame(
            {
                "trade_date": [_TARGET, _TARGET],
                "ts_code": _CODES[:2],
                "ret_1": [0.1, np.nan],
                "custom_infer_col": [1.0, 2.0],
            }
        )
        loader.storage.save_cs_train_day(cs_infer, _TARGET, subdir="cs_infer")
        result = compare_cs_infer_vs_panel(_TARGET, store, loader)
        assert set(result) == {
            "date",
            "cs_infer_cols",
            "panel_cols",
            "common_cols",
            "only_cs_infer",
            "only_panel",
            "nonnull_rate_cs_infer",
            "nonnull_rate_panel",
        }
        assert result["date"] == _TARGET
        # 列覆盖对照
        assert "custom_infer_col" in result["only_cs_infer"]
        assert "ret_1" in result["common_cols"]
        assert "volatility_5" in result["only_panel"]  # 已登记但合成店只写 core
        assert "amount_ma20" in result["only_panel"]  # cs_infer 帧未含该过滤列
        assert result["only_cs_infer"] == sorted(result["only_cs_infer"])
        # 非空率对照（各侧本侧实到列）
        assert result["nonnull_rate_cs_infer"]["ret_1"] == pytest.approx(0.5)
        assert result["nonnull_rate_cs_infer"]["custom_infer_col"] == pytest.approx(1.0)
        assert result["nonnull_rate_panel"]["ret_1"] == pytest.approx(1.0)
        # 族缺日整族 NaN：volatility_5 已登记但无 technical 分区 ⇒ 非空率 0
        assert result["nonnull_rate_panel"]["volatility_5"] == pytest.approx(0.0)

    def test_missing_cs_infer_side_outputs_empty(self, tmp_path):
        store = _make_store(tmp_path)
        loader = _make_loader(tmp_path)
        result = compare_cs_infer_vs_panel(_TARGET, store, loader)
        assert result["cs_infer_cols"] == []
        assert result["nonnull_rate_cs_infer"] == {}
        assert result["panel_cols"]  # panel 侧正常输出
        assert set(result["only_cs_infer"]) == set()


# ---------------------------------------------------------------------------
# 入参与守卫
# ---------------------------------------------------------------------------


class TestInputGuards:
    def test_date_not_in_calendar_raises(self, tmp_path):
        with pytest.raises(ValueError, match="不在注入的交易日历"):
            _assemble(_make_store(tmp_path), _make_loader(tmp_path), date="20260901")

    def test_unsorted_calendar_raises(self, tmp_path):
        calendar = list(_CALENDAR)
        calendar[-1], calendar[-2] = calendar[-2], calendar[-1]
        with pytest.raises(ValueError, match="升序且无重复"):
            _assemble(_make_store(tmp_path), _make_loader(tmp_path), trading_dates=calendar)

    def test_empty_feature_columns_raises(self, tmp_path):
        with pytest.raises(ValueError, match="不能为空"):
            _assemble(_make_store(tmp_path), _make_loader(tmp_path), feature_columns=[])

    def test_panel_query_contract_failure_becomes_gap(self, tmp_path):
        # available_from PIT 越界 ⇒ load_features ValueError ⇒ 转就绪缺口（不穿透）
        store = _make_store(tmp_path)
        store.manifest.register_column("late_col", "core", source="test", available_from="20990101")
        assembly = _assemble(
            store, _make_loader(tmp_path), feature_columns=[*_MODEL_COLUMNS, "late_col"]
        )
        report = assembly.readiness
        assert report.ok is False
        assert any("panel 读取契约失败" in gap for gap in report.gaps)
        assert assembly.features_by_date == {}
