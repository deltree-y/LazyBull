# -*- coding: utf-8 -*-
"""v2 P3 T5 panel 源预测指纹打分测试（t05b_fingerprint 双模式扩展）。

覆盖：
- 跨源误用守卫（reviewer1-S7 处置）：未知模式 / cs_train_verify 传 store 或
  feature_columns / panel 缺 store / panel 缺 feature_columns（None 与空列表）
  各 ValueError；
- panel 模式端到端：合成 PanelDataStore（core 族 f1/f2 + 过滤列，labels 表不落盘
  ——panel 行集无标签非空隐含条件）× 替身 MLSignal ⇒ 指纹产出、行集 =
  panel 当日行 ∩ 主板 ∩ 非 ST/停牌/涨停、source_mode / row_set_semantics 字段、
  逐种子诊断标签；
- panel 打分日域：list_panel_score_days 从 panel 分区日域取（禁 glob cs_train，
  tmp 根无 features/cs_train 构造即证）、resolve_panel_score_days 缺日列出失败；
- strict_days 语义：score_days 缺 panel 分区日 ⇒ strict 失败 / 非 strict 跳过
  （与 cs_train 模式同语义）；
- cs_train_verify 模式回归：结果新增 source_mode / row_set_semantics 字段，
  行集与指纹语义不变。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3.t05b_fingerprint import (  # noqa: E402
    ROW_SET_SEMANTICS,
    SOURCE_MODE_CS_TRAIN_VERIFY,
    SOURCE_MODE_PANEL,
    fingerprint_prediction_rows,
    list_panel_score_days,
    resolve_panel_score_days,
    score_model_fingerprint,
)
from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402

_DAYS = ("20240102", "20240103")
_MAIN_CODES = {"000001.SZ", "000002.SZ"}
_FEATURE_COLUMNS = ["f1", "f2"]
_FILTER_COLUMNS = ("is_st", "is_suspended", "is_limit_up")


class _FakeSubModel:
    def __init__(self, seed: int, offset: float):
        self._seed = seed
        self._offset = offset

    def get_params(self):
        return {"random_state": self._seed}

    def predict(self, X):
        return np.full(len(X), self._offset, dtype=float)


class _FakeEnsemble:
    def __init__(self, sub_models):
        self.models = sub_models


class _FakeSignal:
    """替身 MLSignal：请求种子 [42, 61]、实际聚合顺序 [61, 42]；主预测 = arange。"""

    def __init__(self):
        self.metadata = {"train_params": {"ensemble_seeds": [42, 61]}}
        self.feature_columns = list(_FEATURE_COLUMNS)
        self.model = _FakeEnsemble([_FakeSubModel(61, 1.0), _FakeSubModel(42, 2.0)])

    def _load_model(self):
        return None

    def _build_model_input(self, date, df):
        return df, df[list(_FEATURE_COLUMNS)]

    def _predict_scores(self, X, before, after, ranked):
        return np.arange(len(X), dtype=float)


def _patch_signal(monkeypatch: pytest.MonkeyPatch, signal: _FakeSignal) -> None:
    monkeypatch.setattr("src.lazybull.v2.core.signal.ml_signal.MLSignal", lambda **kwargs: signal)


def _panel_day_frame(
    day: str,
    is_st=(0, 1, 0),
    is_suspended=(0, 0, 0),
    is_limit_up=(0, 0, 0),
) -> pd.DataFrame:
    """合成 panel core 日帧：默认 000002.SZ 为 ST（验证过滤）、000003.SZ 非主板（验证域过滤）。"""
    return pd.DataFrame(
        {
            "trade_date": [day, day, day],
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
            "is_st": list(is_st),
            "is_suspended": list(is_suspended),
            "is_limit_up": list(is_limit_up),
        }
    )


def _make_panel_store(root: Path, days=_DAYS, **frame_kwargs) -> PanelDataStore:
    """合成 panel store（core 族特征列 + 过滤列登记；不落 labels 表）。"""
    store = PanelDataStore(root)
    store.manifest.register_columns(
        {
            col: {"group": "core", "source": "test"}
            for col in _FEATURE_COLUMNS + list(_FILTER_COLUMNS)
        }
    )
    store.manifest.save()
    for day in days:
        store.append_features(
            TradeDate.from_str(day), "core", _panel_day_frame(day, **frame_kwargs)
        )
    return store


def _score_panel(store: PanelDataStore, tmp_path: Path, days=_DAYS, strict_days: bool = True):
    return score_model_fingerprint(
        models_dir=tmp_path,
        model_version=7,
        data_root=tmp_path,
        score_days=list(days),
        main_board_codes=set(_MAIN_CODES),
        strict_days=strict_days,
        source_mode=SOURCE_MODE_PANEL,
        store=store,
        feature_columns=list(_FEATURE_COLUMNS),
    )


class TestSourceModeGuards:
    """跨源误用守卫（reviewer1-S7 处置）：四类误用组合 + 未知模式各 ValueError。"""

    def test_unknown_source_mode_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="未知打分源模式"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102"],
                main_board_codes=set(),
                source_mode="cs_infer",
            )

    def test_cs_train_mode_rejects_store(self, tmp_path):
        with pytest.raises(ValueError, match="禁止传 store/feature_columns"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102"],
                main_board_codes=set(),
                source_mode=SOURCE_MODE_CS_TRAIN_VERIFY,
                store=object(),
            )

    def test_cs_train_mode_rejects_feature_columns(self, tmp_path):
        with pytest.raises(ValueError, match="禁止传 store/feature_columns"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102"],
                main_board_codes=set(),
                source_mode=SOURCE_MODE_CS_TRAIN_VERIFY,
                feature_columns=list(_FEATURE_COLUMNS),
            )

    def test_panel_mode_requires_store(self, tmp_path):
        with pytest.raises(ValueError, match="必须提供 PanelDataStore"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102"],
                main_board_codes=set(),
                source_mode=SOURCE_MODE_PANEL,
                feature_columns=list(_FEATURE_COLUMNS),
            )

    def test_panel_mode_requires_feature_columns(self, tmp_path):
        for bad_columns in (None, []):
            with pytest.raises(ValueError, match="必须提供非空 feature_columns"):
                score_model_fingerprint(
                    models_dir=tmp_path,
                    model_version=1,
                    data_root=tmp_path,
                    score_days=["20240102"],
                    main_board_codes=set(),
                    source_mode=SOURCE_MODE_PANEL,
                    store=object(),
                    feature_columns=bad_columns,
                )


class TestPanelScoring:
    """panel 源端到端打分（合成 store × 替身 signal）。"""

    def test_panel_end_to_end_row_set_and_result_fields(self, tmp_path, monkeypatch):
        store = _make_panel_store(tmp_path / "store")
        _patch_signal(monkeypatch, _FakeSignal())
        result = _score_panel(store, tmp_path)
        # 行集口径：每日仅 000001.SZ 存活（000002.SZ 被 is_st 过滤、000003.SZ 非主板）
        assert result["n_days"] == 2
        assert result["n_rows"] == 2
        assert result["score_days"] == list(_DAYS)
        expected_rows = [(day, "000001.SZ", 0.0) for day in _DAYS]
        assert result["fingerprint"] == fingerprint_prediction_rows(expected_rows)
        assert result["source_mode"] == SOURCE_MODE_PANEL
        assert result["row_set_semantics"] == ROW_SET_SEMANTICS[SOURCE_MODE_PANEL]
        assert "无标签非空隐含条件" in result["row_set_semantics"]
        assert [item["seed"] for item in result["per_seed_diagnostic"]] == [61, 42]
        assert result["requested_ensemble_seeds"] == [42, 61]

    def test_panel_tradable_filter_uses_panel_columns(self, tmp_path, monkeypatch):
        """过滤列来源 = panel core 族登记列：停牌/涨停行同样被剔除（非仅 is_st）。"""
        store = _make_panel_store(tmp_path / "store", is_suspended=(1, 0, 0), is_limit_up=(0, 1, 0))
        _patch_signal(monkeypatch, _FakeSignal())
        # 000001.SZ 停牌、000002.SZ 涨停被剔除；000003.SZ 洁净且纳入主板 ⇒ 每日唯一存活
        result = score_model_fingerprint(
            models_dir=tmp_path,
            model_version=7,
            data_root=tmp_path,
            score_days=list(_DAYS),
            main_board_codes={"000001.SZ", "000002.SZ", "000003.SZ"},
            source_mode=SOURCE_MODE_PANEL,
            store=store,
            feature_columns=list(_FEATURE_COLUMNS),
        )
        expected_rows = [(day, "000003.SZ", 0.0) for day in _DAYS]
        assert result["n_rows"] == 2
        assert result["fingerprint"] == fingerprint_prediction_rows(expected_rows)

    def test_panel_empty_section_strict_rejected(self, tmp_path, monkeypatch):
        store = _make_panel_store(tmp_path / "store", is_st=(1, 1, 1))
        _patch_signal(monkeypatch, _FakeSignal())
        with pytest.raises(RuntimeError, match="过滤后截面为空"):
            _score_panel(store, tmp_path)

    def test_panel_missing_partition_day_strict_rejected(self, tmp_path, monkeypatch):
        store = _make_panel_store(tmp_path / "store", days=("20240102",))
        _patch_signal(monkeypatch, _FakeSignal())
        with pytest.raises(RuntimeError, match="无 panel 分区数据"):
            _score_panel(store, tmp_path, days=("20240102", "20240103"))

    def test_panel_missing_partition_day_nonstrict_skipped(self, tmp_path, monkeypatch):
        store = _make_panel_store(tmp_path / "store", days=("20240102",))
        _patch_signal(monkeypatch, _FakeSignal())
        result = _score_panel(store, tmp_path, days=("20240102", "20240103"), strict_days=False)
        assert result["n_days"] == 1
        assert result["score_days"] == ["20240102"]
        assert result["n_rows"] == 1

    def test_panel_empty_score_days_rejected(self, tmp_path, monkeypatch):
        store = _make_panel_store(tmp_path / "store")
        _patch_signal(monkeypatch, _FakeSignal())
        with pytest.raises(ValueError, match="score_days 为空"):
            _score_panel(store, tmp_path, days=())


class TestPanelScoreDays:
    """panel 打分日域枚举（禁 glob cs_train；tmp 根无 features/cs_train 即证）。"""

    def test_list_panel_score_days_from_partition_domain(self, tmp_path):
        store = _make_panel_store(tmp_path / "store")
        assert list_panel_score_days(store, "20240101", "20240131") == list(_DAYS)
        assert list_panel_score_days(store, "20240103", "20240131") == ["20240103"]

    def test_list_panel_score_days_empty_window_rejected(self, tmp_path):
        store = _make_panel_store(tmp_path / "store")
        with pytest.raises(ValueError, match="无分区日"):
            list_panel_score_days(store, "20250101", "20250131")

    def test_resolve_panel_score_days_missing_listed(self, tmp_path):
        store = _make_panel_store(tmp_path / "store", days=("20240102",))
        with pytest.raises(RuntimeError, match="缺 panel 分区"):
            resolve_panel_score_days(["20240102", "20240103"], store, "20240101", "20240131")
        assert resolve_panel_score_days(["20240102"], store, "20240101", "20240131") == ["20240102"]


class _FakeStorage:
    def __init__(self, frames):
        self._frames = frames

    def load_cs_train_day(self, day: str):
        return self._frames.get(day)


def _fake_cs_train_frame(day: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "is_st": [False, False],
            "is_suspended": [False, False],
            "is_limit_up": [False, False],
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
        }
    )


class TestCsTrainModeRegression:
    """cs_train_verify 模式回归：结果新增字段 + 行集/指纹语义不变。"""

    def test_cs_train_result_marks_source_mode(self, tmp_path, monkeypatch):
        frames = {day: _fake_cs_train_frame(day) for day in _DAYS}
        monkeypatch.setattr(
            "src.lazybull.data.storage.Storage", lambda root_path: _FakeStorage(frames)
        )
        _patch_signal(monkeypatch, _FakeSignal())
        result = score_model_fingerprint(
            models_dir=tmp_path,
            model_version=1,
            data_root=tmp_path,
            score_days=list(_DAYS),
            main_board_codes=set(_MAIN_CODES),
        )
        assert result["source_mode"] == SOURCE_MODE_CS_TRAIN_VERIFY
        assert result["row_set_semantics"] == ROW_SET_SEMANTICS[SOURCE_MODE_CS_TRAIN_VERIFY]
        assert "隐含标签非空" in result["row_set_semantics"]
        assert result["n_days"] == 2
        assert result["n_rows"] == 4
