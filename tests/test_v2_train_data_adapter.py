# -*- coding: utf-8 -*-
"""v2 P3 T1 训练数据只读适配层测试（合成数据，不触碰真实 data/）。

覆盖（P3 规划 §4 / §4.1 / §7.6）：
- 配置校验：冻结段窗口上界拒绝、窗口倒置、标签列白名单、列集纯净（禁键列/标签列）、
  未知模式、panel 模式缺 store；
- panel 模式 join 语义：六标签原名恢复与列序、缺表（含隔离根未接入 labels）/ 缺分区 /
  日级半成品 / 键集不一致 / 意外增删键 / 重复键 fail-fast、universe 过滤、未登记列拒绝；
- 成熟度与水位：forming 尾部标签值保留（E_5 ≤ W_in < E_20 正例）、E_20 ≤ W_in 仍
  forming 硬失败（负例）、水位三对象分列记录不混用、逐日 readiness 状态登记；
- cs_train 验证源模式：默认关、用途声明白名单、生产/纸面调用栈拒绝、与现役
  ``load_features_data`` 逐值一致（含六标签与键集）、分区覆盖硬失败、universe 拒绝、
  四族默认关与「启用未传 lookup ⇒ 报错」、panel 模式禁 derive 双写、生产路径静态不可达。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.lazybull.v2.common.types import TradeDate
from src.lazybull.v2.store.column_groups import (
    LABEL_TABLES,
    LABEL_VALUE_COLUMNS,
    PANEL_GROUPS,
)
from src.lazybull.v2.store.data_store import PanelDataStore
from src.lazybull.v2.train import data_adapter
from src.lazybull.v2.train.data_adapter import (
    SOURCE_CS_TRAIN_VERIFY,
    SOURCE_PANEL,
    AdapterConfig,
    load_training_frame,
    resolve_input_watermark,
    resolve_trading_dates,
)

_CODES = ["600000.SH", "000001.SZ", "000002.SZ"]
_FEATURE_COLUMNS = ("ret_1", "turnover_rate")
#: 标签分区协议列序（label_refresh._LABEL_COLUMNS；readiness 强校验）
_LABEL_COLUMNS = ["ts_code", "trade_date", "label_value", "neu_label_value", "maturity_status"]


# ---------------------------------------------------------------------------
# 合成数据根夹具
# ---------------------------------------------------------------------------


def _calendar_days(n: int = 70, start: str = "2026-01-05") -> list[str]:
    return pd.date_range(start, periods=n, freq="B").strftime("%Y%m%d").tolist()


def _write_calendar(root: Path, days: list[str]) -> None:
    clean = root / "clean"
    clean.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cal_date": days, "is_open": [1] * len(days)}).to_parquet(
        clean / "trade_cal.parquet", index=False
    )


def _write_daily_partitions(root: Path, days: list[str]) -> None:
    daily = root / "clean" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for day in days:
        pd.DataFrame({"trade_date": [day], "ts_code": [_CODES[0]], "close": [1.0]}).to_parquet(
            daily / f"{day[:4]}-{day[4:6]}-{day[6:]}.parquet", index=False
        )


def _make_store(root: Path) -> PanelDataStore:
    store = PanelDataStore(root)
    store.manifest.register_columns(
        {col: {"group": "core", "source": "test"} for col in _FEATURE_COLUMNS}
    )
    store.manifest.save()
    return store


def _append_panel_day(
    store: PanelDataStore, day: str, *, missing_groups: tuple[str, ...] = (), value: float = 0.01
) -> None:
    for group in PANEL_GROUPS:
        if group in missing_groups:
            continue
        payload: dict[str, list] = {"trade_date": [day] * len(_CODES), "ts_code": list(_CODES)}
        if group == "core":
            for col in _FEATURE_COLUMNS:
                payload[col] = [value + i * 0.001 for i in range(len(_CODES))]
        store.append_features(TradeDate.from_str(day), group, pd.DataFrame(payload))


def _append_label_day(
    store: PanelDataStore,
    day: str,
    *,
    maturity: str = "sealed",
    nan_tables: tuple[str, ...] = (),
    drop_code: str | None = None,
) -> None:
    codes = [code for code in _CODES if code != drop_code]
    for offset, (name, _) in enumerate(LABEL_TABLES.items()):
        nan = name in nan_tables
        frame = pd.DataFrame(
            {
                "ts_code": codes,
                "trade_date": [day] * len(codes),
                "label_value": [float("nan") if nan else 0.01 * (offset + 1)] * len(codes),
                "neu_label_value": [float("nan") if nan else 0.02 * (offset + 1)] * len(codes),
                "maturity_status": [maturity] * len(codes),
            },
            columns=_LABEL_COLUMNS,
        )
        store.append_labels(name, frame)


def _build_panel_root(
    tmp_path: Path,
    days: list[str],
    *,
    maturity: str = "sealed",
    watermark_day: str | None = None,
    nan_tables: tuple[str, ...] = (),
    drop_label_code: str | None = None,
    missing_groups: tuple[str, ...] = (),
    skip_label_tables: bool = False,
) -> PanelDataStore:
    """构建最小数据根：日历 + clean/daily 水位分区 + panel 八族 + 三表标签。"""
    calendar = _calendar_days()
    _write_calendar(tmp_path, calendar)
    watermark_day = watermark_day or calendar[-1]
    _write_daily_partitions(tmp_path, [day for day in calendar if day <= watermark_day])
    store = _make_store(tmp_path)
    for day in days:
        _append_panel_day(store, day, missing_groups=missing_groups)
        if not skip_label_tables:
            _append_label_day(
                store, day, maturity=maturity, nan_tables=nan_tables, drop_code=drop_label_code
            )
    return store


def _panel_config(days: list[str], **overrides) -> AdapterConfig:
    params = {
        "start_date": days[0],
        "end_date": days[-1],
        "feature_columns": _FEATURE_COLUMNS,
    }
    params.update(overrides)
    return AdapterConfig(**params)


def _write_cs_train_day(root: Path, day: str, *, with_labels: bool = True) -> pd.DataFrame:
    cs_train = root / "features" / "cs_train"
    cs_train.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list] = {
        "trade_date": [day] * len(_CODES),
        "ts_code": list(_CODES),
        "ret_1": [0.01, 0.02, 0.03],
        "turnover_rate": [1.1, 1.2, 1.3],
        "sw_industry": ["银行", "地产", "银行"],
    }
    if with_labels:
        for col in LABEL_VALUE_COLUMNS:
            payload[col] = [0.05, float("nan"), -0.02]
    frame = pd.DataFrame(payload)
    frame.to_parquet(cs_train / f"{day}.parquet", index=False)
    return frame


def _build_cs_train_root(tmp_path: Path, days: list[str]) -> None:
    calendar = _calendar_days()
    _write_calendar(tmp_path, calendar)
    _write_daily_partitions(tmp_path, calendar[:30])
    for day in days:
        _write_cs_train_day(tmp_path, day)


def _cs_train_config(days: list[str], **overrides) -> AdapterConfig:
    params = {
        "start_date": days[0],
        "end_date": days[-1],
        "source_mode": SOURCE_CS_TRAIN_VERIFY,
    }
    params.update(overrides)
    return AdapterConfig(**params)


def _old_path_objects(root: Path):
    from src.lazybull.data.loader import DataLoader
    from src.lazybull.data.storage import Storage

    storage = Storage(root_path=str(root))
    return storage, DataLoader(storage)


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_default_mode_is_panel(self):
        config = AdapterConfig(start_date="20260105", end_date="20260106")
        assert config.source_mode == SOURCE_PANEL
        assert config.runtime_families == frozenset()

    def test_frozen_window_upper_bound_rejected(self, tmp_path):
        store = _build_panel_root(tmp_path, ["20260105"])
        config = _panel_config(["20260105"], end_date="20260703")
        with pytest.raises(ValueError, match="冻结段"):
            load_training_frame(config, store=store)

    def test_custom_frozen_through(self, tmp_path):
        store = _build_panel_root(tmp_path, ["20260105"])
        # 窗口终点 = 上界 ⇒ 放行；上界前移一日 ⇒ 越界拒绝
        config = _panel_config(["20260105"], frozen_through="20260105")
        load_training_frame(config, store=store)
        config = _panel_config(["20260105"], frozen_through="20260104")
        with pytest.raises(ValueError, match="越界"):
            load_training_frame(config, store=store)

    def test_start_after_end_rejected(self):
        config = AdapterConfig(start_date="20260106", end_date="20260105")
        with pytest.raises(ValueError, match="倒置"):
            load_training_frame(config, store=None)

    def test_bad_label_column_rejected(self):
        config = AdapterConfig(start_date="20260105", end_date="20260106", label_column="y_ret_7")
        with pytest.raises(ValueError, match="六标签原名"):
            load_training_frame(config, store=None)

    def test_feature_columns_with_label_or_key_rejected(self):
        with pytest.raises(ValueError, match="标签列"):
            load_training_frame(
                AdapterConfig(
                    start_date="20260105",
                    end_date="20260106",
                    feature_columns=("ret_1", "y_ret_20"),
                ),
                store=None,
            )
        with pytest.raises(ValueError, match="键列"):
            load_training_frame(
                AdapterConfig(
                    start_date="20260105",
                    end_date="20260106",
                    feature_columns=("trade_date",),
                ),
                store=None,
            )

    def test_unknown_source_mode_rejected(self):
        config = AdapterConfig(start_date="20260105", end_date="20260106", source_mode="raw")
        with pytest.raises(ValueError, match="未知数据源模式"):
            load_training_frame(config)

    def test_panel_mode_requires_store(self):
        config = AdapterConfig(
            start_date="20260105", end_date="20260106", feature_columns=_FEATURE_COLUMNS
        )
        with pytest.raises(ValueError, match="PanelDataStore"):
            load_training_frame(config)


# ---------------------------------------------------------------------------
# panel 模式：join 语义与 fail-fast
# ---------------------------------------------------------------------------


class TestPanelModeJoin:
    def test_happy_path_restores_six_labels(self, tmp_path):
        days = ["20260105", "20260106", "20260107"]
        store = _build_panel_root(tmp_path, days)
        result = load_training_frame(_panel_config(days), store=store)
        df = result.df
        assert result.trade_days == 3
        assert len(df) == 3 * len(_CODES)
        expected_cols = ["trade_date", "ts_code", *_FEATURE_COLUMNS, *LABEL_VALUE_COLUMNS]
        assert list(df.columns) == expected_cols
        # 六标签原名恢复且 raw/neu 变体取值正确（LABEL_TABLES 顺序 + 夹具步长）
        first = df.iloc[0]
        assert first["y_ret_5"] == pytest.approx(0.01)
        assert first["neu_y_ret_5"] == pytest.approx(0.02)
        assert first["y_ret_10"] == pytest.approx(0.02)
        assert first["neu_y_ret_20"] == pytest.approx(0.06)
        # 特征值来自 core 分区（输出按 (trade_date, ts_code) 排序，首行 000001.SZ 为 0.011）
        assert df[df["ts_code"] == "600000.SH"]["ret_1"].iloc[0] == pytest.approx(0.01)
        # maturity_status 不进特征矩阵
        assert "maturity_status" not in df.columns
        # 逐日 readiness 状态登记
        assert [entry["status"] for entry in result.day_readiness] == ["sealed"] * 3
        assert result.watermarks.label_sealed_through["y_ret_20"] == days[-1]

    def test_missing_labels_dir_fails(self, tmp_path):
        """隔离根未接入 labels（无标签目录）⇒ 明确失败（R2-01 负例，readiness 先行拦截）。"""
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days, skip_label_tables=True)
        with pytest.raises(RuntimeError, match="缺标签分区"):
            load_training_frame(_panel_config(days), store=store)

    def test_missing_panel_day_fails(self, tmp_path):
        days = ["20260105", "20260106"]
        store = _make_store(tmp_path)
        calendar = _calendar_days()
        _write_calendar(tmp_path, calendar)
        _write_daily_partitions(tmp_path, calendar)
        _append_panel_day(store, "20260105")
        _append_label_day(store, "20260105")
        with pytest.raises(RuntimeError, match="缺 panel 八族分区"):
            load_training_frame(_panel_config(days), store=store)

    def test_missing_group_fails(self, tmp_path):
        """日级半成品（当日缺一族分区）⇒ readiness 硬失败。"""
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days, missing_groups=("moneyflow",))
        with pytest.raises(RuntimeError, match="缺 panel 八族分区"):
            load_training_frame(_panel_config(days), store=store)

    def test_label_key_mismatch_fails(self, tmp_path):
        """标签键集与 panel core 不一致（删键）⇒ fail-fast。"""
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days, drop_label_code=_CODES[0])
        with pytest.raises(RuntimeError, match="键集不一致"):
            load_training_frame(_panel_config(days), store=store)

    def test_join_unexpected_extra_key_fails(self, tmp_path, monkeypatch):
        """标签侧多出 core 之外的键（增键）⇒ 意外增删键 fail-fast（P3-T2 收敛路径）。

        注入点随 §16.5-② 收敛移至 ``validate_label_day_readiness_and_collect``
        （适配层不再经 ``store.load_labels`` 双通道读取）。
        """
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days)
        original = data_adapter.validate_label_day_readiness_and_collect

        def patched(store_, day, **kwargs):
            readiness, frames = original(store_, day, **kwargs)
            frames = dict(frames)
            extra = frames["y_ret_5"].iloc[:1].copy()
            extra["ts_code"] = "999999.XX"
            frames["y_ret_5"] = pd.concat([frames["y_ret_5"], extra], ignore_index=True)
            return readiness, frames

        monkeypatch.setattr(data_adapter, "validate_label_day_readiness_and_collect", patched)
        with pytest.raises(RuntimeError, match="意外增删键"):
            load_training_frame(_panel_config(days), store=store)

    def test_join_duplicate_keys_fail(self, tmp_path, monkeypatch):
        """标签键重复 ⇒ fail-fast（P3-T2 收敛路径，装配层重复键检查）。"""
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days)
        original = data_adapter.validate_label_day_readiness_and_collect

        def patched(store_, day, **kwargs):
            readiness, frames = original(store_, day, **kwargs)
            frames = dict(frames)
            frames["y_ret_5"] = pd.concat(
                [frames["y_ret_5"], frames["y_ret_5"].iloc[:1]], ignore_index=True
            )
            return readiness, frames

        monkeypatch.setattr(data_adapter, "validate_label_day_readiness_and_collect", patched)
        with pytest.raises(RuntimeError, match="重复"):
            load_training_frame(_panel_config(days), store=store)

    def test_load_labels_not_called_on_panel_path(self, tmp_path, monkeypatch):
        """P3-T2 收敛去重证据（§16.5-②）：panel 模式 ``store.load_labels`` 调用计数为 0。"""
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days)
        calls: list = []
        original = store.load_labels

        def spy(query):
            calls.append(query)
            return original(query)

        monkeypatch.setattr(store, "load_labels", spy)
        result = load_training_frame(_panel_config(days), store=store)
        assert calls == []
        assert len(result.df) == len(_CODES)

    def test_universe_filter(self, tmp_path):
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days)
        config = _panel_config(days, universe=(_CODES[0],))
        result = load_training_frame(config, store=store)
        assert set(result.df["ts_code"]) == {_CODES[0]}

    def test_unregistered_feature_column_fails(self, tmp_path):
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days)
        config = _panel_config(days, feature_columns=("not_registered_col",))
        with pytest.raises(ValueError, match="未登记 manifest"):
            load_training_frame(config, store=store)

    def test_frame_restricted_to_validated_days(self, tmp_path):
        """reviewer2-N1 余项：区间读取带入的非日历日数据 ⇒ 剔除并登记 excluded_days。

        store 含周六 20260110（不在交易日历）的完整 panel+labels；查询区间覆盖该日，
        已校验日域仅工作日 ⇒ 输出帧不得携带该日，excluded_days 显式登记。
        """
        days = ["20260105", "20260106", "20260107", "20260108", "20260109"]
        store = _build_panel_root(tmp_path, days)
        _append_panel_day(store, "20260110")  # 周六，非交易日
        _append_label_day(store, "20260110")
        config = _panel_config(days, end_date="20260110")
        result = load_training_frame(config, store=store)
        assert sorted(result.df["trade_date"].unique()) == days
        assert result.excluded_days == ("20260110",)
        assert {entry["trade_date"] for entry in result.day_readiness} == set(days)


# ---------------------------------------------------------------------------
# 成熟度与水位（§4.1 正反例）
# ---------------------------------------------------------------------------


class TestMaturityAndWatermark:
    def test_forming_tail_values_preserved(self, tmp_path):
        """正例：E_5 ≤ W_in < E_20 的 forming 日，y5 合法非空可消费、y20 NaN 不清空。"""
        calendar = _calendar_days()
        day = calendar[19]  # E_5 = calendar[25]，E_20 = calendar[41]
        watermark = calendar[25]
        store = _build_panel_root(
            tmp_path, [day], maturity="forming", watermark_day=watermark, nan_tables=("y_ret_20",)
        )
        result = load_training_frame(_panel_config([day]), store=store)
        row = result.df.iloc[0]
        assert row["y_ret_5"] == pytest.approx(0.01)
        assert row["neu_y_ret_5"] == pytest.approx(0.02)
        assert pd.isna(row["y_ret_20"])  # 禁止对 forming 行标签值统一清空
        assert result.day_readiness[0]["status"] == "forming"

    def test_expired_forming_day_fails(self, tmp_path):
        """负例：E_20 ≤ W_in 的日仍 forming ⇒ 硬失败。"""
        calendar = _calendar_days()
        day = calendar[30]  # E_20 = calendar[51] ≤ watermark calendar[60]
        store = _build_panel_root(tmp_path, [day], maturity="forming", watermark_day=calendar[60])
        with pytest.raises(RuntimeError, match="E20 已到水位仍为"):
            load_training_frame(_panel_config([day]), store=store)

    def test_watermark_objects_recorded_separately(self, tmp_path):
        """水位三对象分列：W_in / sealed_through / 最新分区日各自取值，禁止 min 合并。"""
        calendar = _calendar_days()
        days = ["20260105", "20260106"]
        store = _build_panel_root(tmp_path, days, watermark_day=calendar[40])
        result = load_training_frame(_panel_config(days), store=store)
        marks = result.watermarks
        assert marks.input_watermark == calendar[40]
        assert marks.label_sealed_through == {name: days[-1] for name in LABEL_TABLES}
        assert marks.label_latest_partition == {name: days[-1] for name in LABEL_TABLES}
        assert marks.panel_latest_partition == days[-1]
        # 混用拒绝：W_in（更晚）不得被 min 合并压回 sealed_through
        assert marks.input_watermark > marks.label_sealed_through["y_ret_20"]

    def test_watermark_ignores_unregistered_orphan(self, tmp_path):
        """水位单一来源 = manifest 登记（reviewer1-P3-4）：未登记孤儿分区不影响最新分区日。"""
        days = ["20260105", "20260106"]
        store = _build_panel_root(tmp_path, days)
        orphan_dir = tmp_path / "labels" / "y_ret_20"
        pd.DataFrame(
            {
                "ts_code": _CODES,
                "trade_date": ["20261231"] * len(_CODES),
                "label_value": [0.5] * len(_CODES),
                "neu_label_value": [0.6] * len(_CODES),
                "maturity_status": ["forming"] * len(_CODES),
            }
        ).to_parquet(orphan_dir / "20261231.parquet", index=False)
        result = load_training_frame(_panel_config(days), store=store)
        assert result.watermarks.label_latest_partition["y_ret_20"] == days[-1]
        assert result.watermarks.panel_latest_partition == days[-1]

    def test_resolve_helpers(self, tmp_path):
        calendar = _calendar_days()
        _write_calendar(tmp_path, calendar)
        _write_daily_partitions(tmp_path, calendar[:30])
        assert resolve_trading_dates(tmp_path) == calendar
        assert resolve_input_watermark(tmp_path) == calendar[29]

    def test_resolve_watermark_missing_dir_fails(self, tmp_path):
        _write_calendar(tmp_path, _calendar_days())
        with pytest.raises(ValueError, match="clean/daily"):
            resolve_input_watermark(tmp_path)


# ---------------------------------------------------------------------------
# cs_train 验证源模式（验收专用过渡例外）
# ---------------------------------------------------------------------------


class TestCsTrainVerifyMode:
    def test_purpose_required(self, tmp_path):
        days = ["20260105"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        config = _cs_train_config(days)
        with pytest.raises(ValueError, match="验收用途"):
            load_training_frame(config, storage=storage, loader=loader)
        with pytest.raises(ValueError, match="验收用途"):
            load_training_frame(
                config, storage=storage, loader=loader, acceptance_purpose="production"
            )

    @staticmethod
    def _forge_stack(*filenames: str):
        """构造假调用栈：首帧 = 适配层自身文件（被跳过），其后为给定调用方文件。"""
        adapter_file = str(Path(data_adapter.__file__).resolve())
        return [SimpleNamespace(filename=name) for name in (adapter_file, *filenames)]

    def test_denied_caller_rejected(self, tmp_path, monkeypatch):
        """生产/纸面文件调用 ⇒ 拒绝（调用方文件白名单，reviewer1-P2-1）。"""
        repo_root = Path(data_adapter.__file__).resolve().parents[4]
        fake = self._forge_stack(str(repo_root / "src" / "lazybull" / "paper" / "runner.py"))
        monkeypatch.setattr(inspect, "stack", lambda: fake)
        days = ["20260105"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        with pytest.raises(RuntimeError, match="白名单|仅允许"):
            load_training_frame(
                _cs_train_config(days),
                storage=storage,
                loader=loader,
                acceptance_purpose="g_code_parity",
            )

    def test_main_entry_production_script_rejected(self, tmp_path, monkeypatch):
        """以 ``__main__`` 运行的生产训练入口按文件名同样被拒绝（reviewer1-P2-1）。"""
        repo_root = Path(data_adapter.__file__).resolve().parents[4]
        fake = self._forge_stack(str(repo_root / "scripts" / "train_ml_model.py"))
        monkeypatch.setattr(inspect, "stack", lambda: fake)
        days = ["20260105"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        with pytest.raises(RuntimeError, match="白名单|仅允许"):
            load_training_frame(
                _cs_train_config(days),
                storage=storage,
                loader=loader,
                acceptance_purpose="g_code_parity",
            )

    def test_non_file_caller_rejected(self, tmp_path, monkeypatch):
        """REPL / ``python -c`` 非文件调用栈 ⇒ fail-closed 拒绝。"""
        monkeypatch.setattr(inspect, "stack", lambda: [SimpleNamespace(filename="<stdin>")])
        days = ["20260105"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        with pytest.raises(RuntimeError, match="无法判定调用方"):
            load_training_frame(
                _cs_train_config(days),
                storage=storage,
                loader=loader,
                acceptance_purpose="g_code_parity",
            )

    def test_engine_internal_call_allowed(self, tmp_path, monkeypatch):
        """reviewer1-R1 方案 A：v2/train 引擎内部帧 + 验收入口帧 ⇒ 放行（G-code 接线合法）。"""
        repo_root = Path(data_adapter.__file__).resolve().parents[4]
        fake = self._forge_stack(
            str(repo_root / "src" / "lazybull" / "v2" / "train" / "walk_forward" / "runner.py"),
            str(repo_root / "scripts" / "v2_p3" / "gcode_driver.py"),
        )
        monkeypatch.setattr(inspect, "stack", lambda: fake)
        days = ["20260105"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        result = load_training_frame(
            _cs_train_config(days),
            storage=storage,
            loader=loader,
            acceptance_purpose="g_code_parity",
        )
        assert len(result.df) == 3

    def test_engine_without_entry_rejected(self, tmp_path, monkeypatch):
        """reviewer1-R1：引擎帧之后不是验收入口（生产脚本）⇒ 拒绝。"""
        repo_root = Path(data_adapter.__file__).resolve().parents[4]
        fake = self._forge_stack(
            str(repo_root / "src" / "lazybull" / "v2" / "train" / "walk_forward" / "runner.py"),
            str(repo_root / "scripts" / "train_ml_model.py"),
        )
        monkeypatch.setattr(inspect, "stack", lambda: fake)
        days = ["20260105"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        with pytest.raises(RuntimeError, match="验收入口"):
            load_training_frame(
                _cs_train_config(days),
                storage=storage,
                loader=loader,
                acceptance_purpose="g_code_parity",
            )

    def test_frozen_later_bound_rejected(self, tmp_path):
        """R2-P3-T1-02：自定义更晚上界 ⇒ 拒绝（两模式同一权威边界）；更早边界仍有效。"""
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days)
        with pytest.raises(ValueError, match="权威冻结边界"):
            load_training_frame(_panel_config(days, frozen_through="20260731"), store=store)
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        with pytest.raises(ValueError, match="权威冻结边界"):
            load_training_frame(
                _cs_train_config(days, frozen_through="20260731"),
                storage=storage,
                loader=loader,
                acceptance_purpose="g_code_parity",
            )

    def test_universe_empty_or_str_rejected(self, tmp_path):
        """空 tuple / str 的 universe ⇒ 拒绝（None 才是全市场；禁止 fail-open 静默退化）。"""
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days)
        with pytest.raises(ValueError, match="空 tuple"):
            load_training_frame(_panel_config(days, universe=()), store=store)
        with pytest.raises(ValueError, match="tuple"):
            load_training_frame(_panel_config(days, universe="600000.SH"), store=store)

    def test_cs_train_feature_columns_rejected(self, tmp_path):
        """cs_train 验证源模式传非空 feature_columns ⇒ 拒绝（静默忽略禁止）。"""
        days = ["20260105"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        with pytest.raises(ValueError, match="全 schema"):
            load_training_frame(
                _cs_train_config(days, feature_columns=("ret_1",)),
                storage=storage,
                loader=loader,
                acceptance_purpose="g_code_parity",
            )

    def test_parity_with_legacy_loader(self, tmp_path):
        """G-code 数据前提：与现役 load_features_data 逐值完全一致（含六标签与键集）。"""
        days = ["20260105", "20260106", "20260107"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        from src.lazybull.ml.train_core.split import load_features_data

        reference, n_days = load_features_data(storage, loader, days[0], days[-1])
        result = load_training_frame(
            _cs_train_config(days),
            storage=storage,
            loader=loader,
            acceptance_purpose="g_code_parity",
        )
        pd.testing.assert_frame_equal(result.df, reference, check_exact=True)
        assert result.trade_days == n_days == 3
        assert result.watermarks.cs_train_latest_partition == days[-1]
        # 回显语义（reviewer1-P3-5）：无 store ⇒ fingerprint=None；feature_columns=实际加载列
        assert result.manifest_fingerprint is None
        assert result.feature_columns == ("ret_1", "turnover_rate", "sw_industry")

    def test_coverage_missing_partition_fails(self, tmp_path):
        """窗口内交易日缺 cs_train 分区 ⇒ 硬失败（现役静默跳过，适配层拒绝）。"""
        days = ["20260105", "20260106"]
        _build_cs_train_root(tmp_path, days[:1])  # 只写首日
        storage, loader = _old_path_objects(tmp_path)
        with pytest.raises(RuntimeError, match="分区覆盖不全"):
            load_training_frame(
                _cs_train_config(days),
                storage=storage,
                loader=loader,
                acceptance_purpose="g_code_parity",
            )

    def test_universe_rejected(self, tmp_path):
        days = ["20260105"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        with pytest.raises(ValueError, match="universe"):
            load_training_frame(
                _cs_train_config(days, universe=(_CODES[0],)),
                storage=storage,
                loader=loader,
                acceptance_purpose="g_code_parity",
            )

    def test_runtime_families_lookup_required(self, tmp_path):
        """四族默认关；cs_train 模式启用而未传同源 lookup ⇒ 报错（S5）。"""
        days = ["20260105"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        config = _cs_train_config(days, runtime_families=frozenset({"holdertrade"}))
        with pytest.raises(ValueError, match="holdertrade_lookup"):
            load_training_frame(
                config, storage=storage, loader=loader, acceptance_purpose="g_code_parity"
            )
        # 传入同源 lookup 后放行并透传
        lookups = {"holdertrade_lookup": {"20260105": pd.DataFrame()}}
        result = load_training_frame(
            _cs_train_config(
                days, runtime_families=frozenset({"holdertrade"}), runtime_lookups=lookups
            ),
            storage=storage,
            loader=loader,
            acceptance_purpose="g_code_parity",
        )
        assert result.runtime_lookups is lookups

    def test_unknown_family_rejected(self, tmp_path):
        days = ["20260105"]
        _build_cs_train_root(tmp_path, days)
        storage, loader = _old_path_objects(tmp_path)
        with pytest.raises(ValueError, match="未知运行时派生族"):
            load_training_frame(
                _cs_train_config(days, runtime_families=frozenset({"top_inst"})),
                storage=storage,
                loader=loader,
                acceptance_purpose="g_code_parity",
            )

    def test_panel_mode_derive_forbidden(self, tmp_path):
        """panel 模式：四族已物化直读，传 runtime_families / lookups ⇒ 报错（防双写）。"""
        days = ["20260105"]
        store = _build_panel_root(tmp_path, days)
        with pytest.raises(ValueError, match="禁止再走 derive"):
            load_training_frame(
                _panel_config(days, runtime_families=frozenset({"holdertrade"})),
                store=store,
            )
        with pytest.raises(ValueError, match="禁止再走 derive"):
            load_training_frame(
                _panel_config(days, runtime_lookups={"holdertrade_lookup": {}}),
                store=store,
            )

    def test_missing_label_columns_fail(self, tmp_path):
        days = ["20260105"]
        calendar = _calendar_days()
        _write_calendar(tmp_path, calendar)
        _write_daily_partitions(tmp_path, calendar[:30])
        _write_cs_train_day(tmp_path, days[0], with_labels=False)
        storage, loader = _old_path_objects(tmp_path)
        with pytest.raises(RuntimeError, match="缺标签列"):
            load_training_frame(
                _cs_train_config(days),
                storage=storage,
                loader=loader,
                acceptance_purpose="g_code_parity",
            )


class TestProductionUnreachableStatic:
    """生产/纸面代码路径静态不可达：禁止任何生产模块引用 cs_train 验证源模式。"""

    def test_no_production_reference(self):
        root = Path(__file__).resolve().parents[1]
        denied_globs = [
            "src/lazybull/paper/**/*.py",
            "src/lazybull/live/**/*.py",
            "src/lazybull/v2/hosts/**/*.py",
            "src/lazybull/v2/core/**/*.py",
            "src/lazybull/ml/**/*.py",
            "scripts/batch/**/*.py",
        ]
        denied_files = [
            "scripts/paper_trade.py",
            "scripts/bot_service.py",
            "scripts/train_ml_model.py",
            "scripts/walk_forward.py",
        ]
        # P3-T3 豁免登记（codex-R06 处置后精确化）：仅豁免
        # ``src/lazybull/v2/hosts/backtest/replay.py``（验收宿主，承载 cs_train 例外
        # 守卫本体；运行时白名单 fail-closed + 区分力测试覆盖）——按完整仓库相对
        # 路径匹配，同名文件不豁免。
        exempt_paths = {"src/lazybull/v2/hosts/backtest/replay.py"}
        offenders = []
        for pattern in denied_globs:
            for path in root.glob(pattern):
                if path.relative_to(root).as_posix() in exempt_paths:
                    continue
                offenders.extend(self._scan(path))
        for rel in denied_files:
            path = root / rel
            if path.is_file():
                offenders.extend(self._scan(path))
        assert not offenders, f"生产/纸面路径引用 cs_train 验证源模式: {offenders}"

    @staticmethod
    def _scan(path: Path) -> list[str]:
        tokens = ("cs_train_verify", "SOURCE_CS_TRAIN_VERIFY", "load_training_frame")
        text = path.read_text(encoding="utf-8")
        return [f"{path.name}:{token}" for token in tokens if token in text]
