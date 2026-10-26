# -*- coding: utf-8 -*-
"""v2 P3 T5 R1 指纹预测行/信号 topk 持久化测试（R2-T5A-06 + 复审 R2-T5A-R1-01 处置）。

覆盖：
- persist 开启 ⇒ parquet 产物落盘：列集/键序/类型正确（trade_date str / ts_code str /
  score float64，``(trade_date, ts_code)`` 升序）、``-0.0`` 归一 ``0.0``（符号位清除）、
  结果 JSON 的 ``predictions_path`` / ``predictions_sha256`` / ``predictions_rows``
  与文件一致、回读行再序列化与指纹逐位一致（与 S1 文本口径同源）；panel 模式同兑现；
- NaN/Inf 预测 ⇒ 拒绝且不写文件（父目录/tmp 残留均无，校验先于任何文件创建）；
- 同输入双写 ⇒ 文件字节一致（确定性：固定 pyarrow + snappy + index=False）；
- persist 默认关 ⇒ 结果无新键、目录无产物（cs_train_verify 与 panel 两模式各一例）；
  参数误用守卫（开 persist 缺 path / 关 persist 传 path 各 ValueError，打分前生效）；
- standalone CLI 冒烟：``--persist-predictions`` 缺 ``--predictions-path`` ⇒
  SystemExit 非零；开启 + 显式 path ⇒ 正常返回且文件落盘、结果 JSON 增键；
- 信号路径 top-k（R2-T5A-R1-01，真实 ``_apply_selection_filters`` unbound 复用）：
  金融股剔除/成交额门槛/市值门槛各一例 ⇒ 产物只含过滤后行且 top-N 正确（score 降序、
  同分 ts_code 升序、rank 自 1 起）；列/序/类型/双写字节一致/-0.0 归一；
  信号行 NaN/Inf ⇒ 两产物均不落盘；零选股日登记 ``signal_topk_empty_days``；
  锚点登记相对 POSIX；panel 模式过滤消费列并入读取契约（缺列 ⇒ store 契约报错）；
  守卫：signal_selection 缺 persist / 缺键 / 未知键各 ValueError；默认关零变化回归。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3.t05b_fingerprint import (  # noqa: E402
    SOURCE_MODE_PANEL,
    fingerprint_prediction_rows,
    main,
    score_model_fingerprint,
)
from src.lazybull.v2.common.types import TradeDate  # noqa: E402
from src.lazybull.v2.core.signal.ml_signal import MLSignal as _RealMLSignal  # noqa: E402
from src.lazybull.v2.store.data_store import PanelDataStore  # noqa: E402

_DAYS = ("20240102", "20240103")
_MAIN_CODES = {"000001.SZ", "000002.SZ"}
_FEATURE_COLUMNS = ["f1", "f2"]
_FILTER_COLUMNS = ("is_st", "is_suspended", "is_limit_up")
_PERSIST_KEYS = ("predictions_path", "predictions_sha256", "predictions_rows")
_SIGNAL_KEYS = (
    "signal_topk_path",
    "signal_topk_sha256",
    "signal_topk_rows",
    "signal_topk_days",
    "signal_topk_empty_days",
)
_SIGNAL_TOPK_NAME = "signal_topk.parquet"
# 主预测计划：逐打分日一组（日1 含 -0.0；日2 正常值），行序 = 输入帧行序
_SCORES_PLAN = [[1.5, -0.0], [0.25, 2.0]]


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
    """替身 MLSignal：主预测按调用次序取 ``scores_plan``（每打分日一组，逐日一调用）。"""

    def __init__(self, scores_plan):
        self.metadata = {"train_params": {"ensemble_seeds": [42, 61]}}
        self.feature_columns = list(_FEATURE_COLUMNS)
        self.model = _FakeEnsemble([_FakeSubModel(61, 1.0), _FakeSubModel(42, 2.0)])
        self._scores_plan = [np.asarray(scores, dtype=float) for scores in scores_plan]
        self._calls = 0

    def _load_model(self):
        return None

    def _build_model_input(self, date, df):
        return df, df[list(_FEATURE_COLUMNS)]

    def _predict_scores(self, X, before, after, ranked):
        scores = self._scores_plan[self._calls]
        self._calls += 1
        if len(scores) != len(X):
            raise AssertionError(f"替身主预测长度 {len(scores)} != 输入行数 {len(X)}")
        return scores


def _patch_signal(monkeypatch: pytest.MonkeyPatch, signal: _FakeSignal) -> None:
    monkeypatch.setattr("src.lazybull.v2.core.signal.ml_signal.MLSignal", lambda **kwargs: signal)
    if hasattr(signal, "_score_by_code"):
        codes = list(signal._score_by_code)

        class _UniverseLoader:
            def load_clean_stock_basic(self):
                return pd.DataFrame(
                    {
                        "ts_code": codes,
                        "name": ["NORMAL"] * len(codes),
                        "market": ["主板"] * len(codes),
                        "list_date": ["20000101"] * len(codes),
                    }
                )

            def load_clean_daily_by_date(self, day, auto_ensure=False):
                assert auto_ensure is False
                return pd.DataFrame(
                    {
                        "ts_code": codes,
                        "trade_date": [day] * len(codes),
                        "is_suspended": [False] * len(codes),
                    }
                )

        monkeypatch.setattr(
            "src.lazybull.data.loader.DataLoader", lambda storage: _UniverseLoader()
        )


def _fake_cs_train_frame(day: str) -> pd.DataFrame:
    """合成 cs_train 日帧：键序故意乱序（000002 在 000001 前）以验证产物排序。"""
    return pd.DataFrame(
        {
            "ts_code": ["000002.SZ", "000001.SZ"],
            "is_st": [False, False],
            "is_suspended": [False, False],
            "is_limit_up": [False, False],
            "f1": [2.0, 1.0],
            "f2": [4.0, 3.0],
        }
    )


class _FakeStorage:
    def __init__(self, frames):
        self._frames = frames

    def load_cs_train_day(self, day: str):
        return self._frames.get(day)


def _patch_cs_train_stack(
    monkeypatch: pytest.MonkeyPatch, signal: _FakeSignal, days=_DAYS, frame_fn=None
) -> None:
    frame_fn = frame_fn or _fake_cs_train_frame
    frames = {day: frame_fn(day) for day in days}
    monkeypatch.setattr("src.lazybull.data.storage.Storage", lambda root_path: _FakeStorage(frames))
    _patch_signal(monkeypatch, signal)


def _panel_day_frame(day: str) -> pd.DataFrame:
    """合成 panel core 日帧：000002.SZ 为 ST（验证过滤）、000003.SZ 非主板（验证域过滤）。"""
    return pd.DataFrame(
        {
            "trade_date": [day, day, day],
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
            "is_st": [0, 1, 0],
            "is_suspended": [0, 0, 0],
            "is_limit_up": [0, 0, 0],
        }
    )


def _make_panel_store(root: Path, days=_DAYS) -> PanelDataStore:
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
        store.append_features(TradeDate.from_str(day), "core", _panel_day_frame(day))
    return store


def _score_cs_train(tmp_path: Path, **overrides):
    kwargs = {
        "models_dir": tmp_path,
        "model_version": 1,
        "data_root": tmp_path,
        "score_days": list(_DAYS),
        "main_board_codes": set(_MAIN_CODES),
    }
    kwargs.update(overrides)
    if kwargs.get("signal_selection") is not None:
        kwargs.setdefault("signal_universe_config", _signal_universe_config())
    return score_model_fingerprint(**kwargs)


class TestPersistPredictionsCsTrain:
    """cs_train_verify 模式 persist 开启：产物列/序/类型/归一/结果键一致。"""

    def test_persist_writes_sorted_parquet_and_result_keys(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_cs_train_stack(monkeypatch, _FakeSignal(_SCORES_PLAN))
        rel_path = Path("folds/split00/predictions.parquet")
        result = _score_cs_train(tmp_path, persist_predictions=True, predictions_path=rel_path)
        out_file = tmp_path / rel_path
        assert out_file.is_file()
        # 结果键：登记值 = 传入路径 POSIX 形态（相对臂 run 目录口径由调用方传入兑现）
        assert result["predictions_path"] == "folds/split00/predictions.parquet"
        assert result["predictions_rows"] == 4
        assert result["predictions_sha256"] == hashlib.sha256(out_file.read_bytes()).hexdigest()

        frame = pd.read_parquet(out_file)
        assert list(frame.columns) == ["trade_date", "ts_code", "score"]
        assert pd.api.types.is_string_dtype(frame["trade_date"])
        assert pd.api.types.is_string_dtype(frame["ts_code"])
        assert frame["score"].dtype == np.float64
        expected = [
            ("20240102", "000001.SZ", 0.0),
            ("20240102", "000002.SZ", 1.5),
            ("20240103", "000001.SZ", 2.0),
            ("20240103", "000002.SZ", 0.25),
        ]
        assert list(frame.itertuples(index=False, name=None)) == expected
        # -0.0 归一为 0.0（符号位清除）
        assert not np.signbit(frame["score"].to_numpy()).any()
        # 与 S1 文本序列化口径同源：回读行再序列化 ⇒ 指纹逐位一致
        read_rows = [tuple(item) for item in frame.itertuples(index=False, name=None)]
        assert fingerprint_prediction_rows(read_rows) == result["fingerprint"]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_scores_rejected_without_file(self, tmp_path, monkeypatch, bad):
        monkeypatch.chdir(tmp_path)
        _patch_cs_train_stack(monkeypatch, _FakeSignal([[1.5, bad], [0.25, 2.0]]))
        rel_path = Path("folds/split00/predictions.parquet")
        with pytest.raises(ValueError, match="NaN/Inf"):
            _score_cs_train(tmp_path, persist_predictions=True, predictions_path=rel_path)
        assert not (tmp_path / rel_path).exists()
        # 校验先于任何文件/目录创建：父目录与 tmp 残留均不存在
        assert not (tmp_path / "folds").exists()

    def test_double_write_byte_identical(self, tmp_path, monkeypatch):
        """同一输入两次持久化 ⇒ 文件字节一致（固定 pyarrow + snappy + index=False）。"""
        monkeypatch.chdir(tmp_path)
        contents = []
        for name in ("run_a", "run_b"):
            _patch_cs_train_stack(monkeypatch, _FakeSignal(_SCORES_PLAN))
            result = _score_cs_train(
                tmp_path,
                persist_predictions=True,
                predictions_path=Path(name) / "predictions.parquet",
            )
            content = (tmp_path / name / "predictions.parquet").read_bytes()
            assert result["predictions_sha256"] == hashlib.sha256(content).hexdigest()
            contents.append(content)
        assert contents[0] == contents[1]


class TestPersistAnchor:
    """predictions_anchor（臂 run 目录）口径：登记相对锚点 POSIX 路径（runner↔归因契约锁）。"""

    def test_anchor_registers_relative_path(self, tmp_path, monkeypatch):
        _patch_cs_train_stack(monkeypatch, _FakeSignal(_SCORES_PLAN))
        arm_dir = tmp_path / "arm_run"
        abs_path = arm_dir / "folds" / "split00" / "predictions.parquet"
        result = _score_cs_train(
            tmp_path,
            persist_predictions=True,
            predictions_path=abs_path,
            predictions_anchor=arm_dir,
        )
        assert result["predictions_path"] == "folds/split00/predictions.parquet"
        assert abs_path.is_file()
        assert result["predictions_sha256"] == hashlib.sha256(abs_path.read_bytes()).hexdigest()

    def test_anchor_containment_violation_rejected(self, tmp_path, monkeypatch):
        _patch_cs_train_stack(monkeypatch, _FakeSignal(_SCORES_PLAN))
        arm_dir = tmp_path / "arm_run"
        outside = tmp_path / "elsewhere" / "predictions.parquet"
        with pytest.raises(ValueError, match="越出锚点目录"):
            _score_cs_train(
                tmp_path,
                persist_predictions=True,
                predictions_path=outside,
                predictions_anchor=arm_dir,
            )


class TestPersistPredictionsPanel:
    """panel 模式 persist 开启：两模式通用（合成 store 先例）。"""

    def test_panel_persist_writes_parquet(self, tmp_path, monkeypatch):
        store = _make_panel_store(tmp_path / "store")
        _patch_signal(monkeypatch, _FakeSignal([[0.5], [-0.0]]))
        out_path = tmp_path / "folds" / "split00" / "predictions.parquet"
        result = score_model_fingerprint(
            models_dir=tmp_path,
            model_version=7,
            data_root=tmp_path,
            score_days=list(_DAYS),
            main_board_codes={"000001.SZ"},
            source_mode=SOURCE_MODE_PANEL,
            store=store,
            feature_columns=list(_FEATURE_COLUMNS),
            persist_predictions=True,
            predictions_path=out_path,
        )
        # 行集 = panel 当日行 ∩ 主板 ∩ 非 ST/停牌/涨停 ⇒ 每日仅 000001.SZ 存活
        frame = pd.read_parquet(out_path)
        expected = [("20240102", "000001.SZ", 0.5), ("20240103", "000001.SZ", 0.0)]
        assert list(frame.itertuples(index=False, name=None)) == expected
        assert not np.signbit(frame["score"].to_numpy()).any()
        assert result["predictions_rows"] == 2
        # 绝对路径传入 ⇒ 登记绝对 POSIX 形态（docstring 口径）
        assert result["predictions_path"] == out_path.as_posix()
        assert result["predictions_sha256"] == hashlib.sha256(out_path.read_bytes()).hexdigest()


class TestPersistDefaultOff:
    """persist 默认关：结果无新键、目录无产物（两模式各一例；现状逐位不变）。"""

    def test_cs_train_mode_unchanged_without_persist(self, tmp_path, monkeypatch):
        _patch_cs_train_stack(monkeypatch, _FakeSignal(_SCORES_PLAN))
        result = _score_cs_train(tmp_path)
        for key in _PERSIST_KEYS:
            assert key not in result
        assert list(tmp_path.rglob("*.parquet")) == []

    def test_panel_mode_unchanged_without_persist(self, tmp_path, monkeypatch):
        store = _make_panel_store(tmp_path / "store")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        monkeypatch.chdir(run_dir)  # 若误写相对路径产物只会落在 CWD
        _patch_signal(monkeypatch, _FakeSignal([[0.5], [1.5]]))
        result = score_model_fingerprint(
            models_dir=tmp_path,
            model_version=7,
            data_root=tmp_path,
            score_days=list(_DAYS),
            main_board_codes={"000001.SZ"},
            source_mode=SOURCE_MODE_PANEL,
            store=store,
            feature_columns=list(_FEATURE_COLUMNS),
        )
        for key in _PERSIST_KEYS:
            assert key not in result
        assert list(run_dir.rglob("*.parquet")) == []


class TestPersistArgGuards:
    """参数误用守卫：打分前生效（无需替身打分栈即拒绝）。"""

    def test_persist_without_path_rejected_before_scoring(self, tmp_path):
        with pytest.raises(ValueError, match="必须显式提供 predictions_path"):
            _score_cs_train(tmp_path, persist_predictions=True)

    def test_path_without_persist_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="禁止传 predictions_path"):
            _score_cs_train(tmp_path, predictions_path=tmp_path / "predictions.parquet")


class TestCliPersistSmoke:
    """standalone CLI 冒烟：缺 path ⇒ SystemExit 非零；显式 path ⇒ 落盘 + JSON 增键。"""

    def _base_argv(self, tmp_path: Path) -> list:
        return [
            "t05b_fingerprint.py",
            "--models-dir",
            str(tmp_path),
            "--model-version",
            "1",
            "--data-root",
            str(tmp_path),
            "--test-start",
            "20240102",
            "--test-end",
            "20240103",
            "--out",
            str(tmp_path / "fingerprint.json"),
        ]

    def _patch_cli_externals(self, monkeypatch: pytest.MonkeyPatch, signal) -> None:
        _patch_cs_train_stack(monkeypatch, signal)
        monkeypatch.setattr(
            "scripts.v2_p3.t05b_spike_runner.list_expected_oos_days",
            lambda data_root, test_start, test_end: list(_DAYS),
        )
        monkeypatch.setattr(
            "scripts.v2_p3.t05b_spike_runner.resolve_score_days",
            lambda expected_days, features_root: list(expected_days),
        )
        monkeypatch.setattr(
            "scripts.v2_p3.t05b_fingerprint.load_main_board_codes",
            lambda data_root, stock_domain="main": set(_MAIN_CODES),
        )

    def test_persist_without_path_cli_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._base_argv(tmp_path) + ["--persist-predictions"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    def test_persist_with_path_cli_writes_file(self, tmp_path, monkeypatch):
        self._patch_cli_externals(monkeypatch, _FakeSignal(_SCORES_PLAN))
        predictions = tmp_path / "folds" / "split00" / "predictions.parquet"
        monkeypatch.setattr(
            sys,
            "argv",
            self._base_argv(tmp_path)
            + ["--persist-predictions", "--predictions-path", str(predictions)],
        )
        main()
        assert predictions.is_file()
        payload = json.loads((tmp_path / "fingerprint.json").read_text(encoding="utf-8"))
        assert payload["predictions_path"] == predictions.as_posix()
        assert payload["predictions_rows"] == 4
        assert payload["predictions_sha256"] == (
            hashlib.sha256(predictions.read_bytes()).hexdigest()
        )
        assert list(pd.read_parquet(predictions).columns) == ["trade_date", "ts_code", "score"]


# ---------------------------------------------------------------------------
# 信号路径 top-k 产物（R2-T5A-R1-01：真实 _apply_selection_filters 同一实现）
# ---------------------------------------------------------------------------


class _FakeSignalWithFilters:
    """替身 MLSignal（信号路径场景）：选择过滤委托**真实** ``_apply_selection_filters``
    （unbound 复用，禁止复制语义）；主预测按 ts_code 分数表产出（与调用次序无关）。
    """

    _FINANCIAL_SW_L1_CODES = _RealMLSignal._FINANCIAL_SW_L1_CODES

    def __init__(self, score_by_code):
        self.metadata = {"train_params": {"ensemble_seeds": [42, 61]}}
        self.feature_columns = list(_FEATURE_COLUMNS)
        self.model = _FakeEnsemble([_FakeSubModel(61, 1.0)])
        self._score_by_code = {str(code): float(score) for code, score in score_by_code.items()}
        self._last_df = None

    def _load_model(self):
        return None

    def _apply_selection_filters(self, df):
        return _RealMLSignal._apply_selection_filters(self, df)

    def _build_model_input(self, date, df):
        self._last_df = df
        return df, df[list(_FEATURE_COLUMNS)]

    def _predict_scores(self, X, before, after, ranked):
        codes = [str(code) for code in self._last_df["ts_code"].tolist()]
        return np.array([self._score_by_code[code] for code in codes], dtype=float)


_SIGNAL_SCORES = {
    "600000.SH": 0.95,  # 金融股：分数最高但被剔除 ⇒ 证明过滤先于排序取头
    "000001.SZ": 0.9,
    "000004.SZ": 0.85,  # 低成交额剔除
    "000002.SZ": 0.8,
    "000005.SZ": 0.7,  # 市值过小剔除
    "000006.SZ": 0.6,  # 市值过大剔除
    "000003.SZ": -0.0,  # 归一 0.0
}
_TIE_SCORES = {"000001.SZ": 0.5, "000002.SZ": 0.5, "000003.SZ": 0.9}


def _signal_selection(top_n=3, **overrides):
    selection = {
        "top_n": top_n,
        "min_amount_ma20": 50000.0,
        "min_total_mv": 500000.0,
        "max_total_mv": 15000000.0,
        "exclude_financial": True,
    }
    selection.update(overrides)
    return selection


def _signal_universe_config():
    return {"exclude_st": True, "min_list_days": 365, "markets": ("主板",)}


def _signal_cs_train_frame(day: str) -> pd.DataFrame:
    """信号过滤场景日帧（7 码乱序）：金融/低成交额/市值出带各剔除行 + 三 NORMAL 存活。"""
    return pd.DataFrame(
        {
            "ts_code": [
                "600000.SH",  # sw_l1 801780 金融股
                "000004.SZ",  # amount_ma20 低于门槛
                "000001.SZ",
                "000005.SZ",  # total_mv 低于下限
                "000002.SZ",
                "000006.SZ",  # total_mv 高于上限
                "000003.SZ",
            ],
            "is_st": [False] * 7,
            "is_suspended": [False] * 7,
            "is_limit_up": [False] * 7,
            "f1": [1.0] * 7,
            "f2": [2.0] * 7,
            "amount_ma20": [100000.0, 1000.0] + [100000.0] * 5,
            "total_mv": [1000000.0] * 3 + [100.0] + [1000000.0, 20000000.0, 1000000.0],
            "sw_l1_code": ["801780"] + ["801120"] * 6,
        }
    )


def _tie_cs_train_frame(day: str) -> pd.DataFrame:
    """同分定秩场景：000001 与 000002 同分 0.5（行序故意颠倒），000003 最高 0.9。"""
    return pd.DataFrame(
        {
            "ts_code": ["000002.SZ", "000001.SZ", "000003.SZ"],
            "is_st": [False] * 3,
            "is_suspended": [False] * 3,
            "is_limit_up": [False] * 3,
            "f1": [1.0] * 3,
            "f2": [2.0] * 3,
            "amount_ma20": [100000.0] * 3,
            "total_mv": [1000000.0] * 3,
            "sw_l1_code": ["801120"] * 3,
        }
    )


def _excluded_cs_train_frame(day: str) -> pd.DataFrame:
    """全剔除日帧：金融股 + 低成交额两码，信号路径零存活。"""
    return pd.DataFrame(
        {
            "ts_code": ["600000.SH", "000004.SZ"],
            "is_st": [False, False],
            "is_suspended": [False, False],
            "is_limit_up": [False, False],
            "f1": [1.0, 1.0],
            "f2": [2.0, 2.0],
            "amount_ma20": [100000.0, 1000.0],
            "total_mv": [1000000.0, 1000000.0],
            "sw_l1_code": ["801780", "801120"],
        }
    )


def _nan_signal_frame(day: str) -> pd.DataFrame:
    """涨停股可进入 T 日信号候选，但被主门矩阵过滤：验证两产物写前拒绝纪律。"""
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000007.SZ"],
            "is_st": [False, False],
            "is_suspended": [False, False],
            "is_limit_up": [False, True],
            "f1": [1.0, 1.0],
            "f2": [2.0, 2.0],
            "amount_ma20": [100000.0, 100000.0],
            "total_mv": [1000000.0, 1000000.0],
            "sw_l1_code": ["801120", "801120"],
        }
    )


def _panel_signal_frame(day: str) -> pd.DataFrame:
    """panel 信号场景日帧：金融股最高分被剔除、低成交额被剔除，仅 000001.SZ 存活。"""
    return pd.DataFrame(
        {
            "trade_date": [day, day, day],
            "ts_code": ["600000.SH", "000001.SZ", "000004.SZ"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
            "is_st": [0, 0, 0],
            "is_suspended": [0, 0, 0],
            "is_limit_up": [0, 0, 0],
            "amount_ma20": [100000.0, 100000.0, 1000.0],
            "total_mv": [1000000.0] * 3,
            "sw_l1_code": ["801780", "801120", "801120"],
        }
    )


_PANEL_SIGNAL_COLUMNS = [
    *_FEATURE_COLUMNS,
    *_FILTER_COLUMNS,
    "amount_ma20",
    "total_mv",
    "sw_l1_code",
]


def _make_panel_signal_store(root: Path, days=_DAYS) -> PanelDataStore:
    """合成 panel store（信号过滤消费列一并登记的 core 族）。"""
    store = PanelDataStore(root)
    store.manifest.register_columns(
        {col: {"group": "core", "source": "test"} for col in _PANEL_SIGNAL_COLUMNS}
    )
    store.manifest.save()
    for day in days:
        store.append_features(TradeDate.from_str(day), "core", _panel_signal_frame(day))
    return store


class TestSignalTopkCsTrain:
    """信号路径 top-k（cs_train 模式）：真实选择过滤 + 定秩/top-N + 产物纪律。"""

    def _score(self, tmp_path, monkeypatch, frame_fn, scores, **overrides):
        monkeypatch.chdir(tmp_path)
        _patch_cs_train_stack(monkeypatch, _FakeSignalWithFilters(scores), frame_fn=frame_fn)
        kwargs = {
            "main_board_codes": set(scores),
            "persist_predictions": True,
            "predictions_path": Path("folds/split00/predictions.parquet"),
            "signal_selection": _signal_selection(),
        }
        kwargs.update(overrides)
        return _score_cs_train(tmp_path, **kwargs)

    def test_filters_apply_and_topk_written(self, tmp_path, monkeypatch):
        result = self._score(tmp_path, monkeypatch, _signal_cs_train_frame, _SIGNAL_SCORES)
        topk_file = tmp_path / "folds" / "split00" / _SIGNAL_TOPK_NAME
        assert topk_file.is_file()
        # 预测行产物同目录并存（signal_selection 开启必须 persist 开启）
        assert (tmp_path / "folds" / "split00" / "predictions.parquet").is_file()
        assert result["signal_topk_path"] == f"folds/split00/{_SIGNAL_TOPK_NAME}"
        assert result["signal_topk_rows"] == 6
        assert result["signal_topk_days"] == list(_DAYS)
        assert result["signal_topk_empty_days"] == []
        assert result["signal_topk_sha256"] == (hashlib.sha256(topk_file.read_bytes()).hexdigest())

        frame = pd.read_parquet(topk_file)
        assert list(frame.columns) == ["trade_date", "rank", "ts_code", "score"]
        assert pd.api.types.is_string_dtype(frame["trade_date"])
        assert frame["rank"].dtype == np.int64
        assert pd.api.types.is_string_dtype(frame["ts_code"])
        assert frame["score"].dtype == np.float64
        expected = [
            ("20240102", 1, "000001.SZ", 0.9),
            ("20240102", 2, "000002.SZ", 0.8),
            ("20240102", 3, "000003.SZ", 0.0),
            ("20240103", 1, "000001.SZ", 0.9),
            ("20240103", 2, "000002.SZ", 0.8),
            ("20240103", 3, "000003.SZ", 0.0),
        ]
        assert list(frame.itertuples(index=False, name=None)) == expected
        # 金融股（0.95 最高分）/低成交额/市值出带均被剔除 ⇒ 不在产物中
        assert set(frame["ts_code"]) == {"000001.SZ", "000002.SZ", "000003.SZ"}
        # -0.0 归一为 0.0（符号位清除）
        assert not np.signbit(frame["score"].to_numpy()).any()

    def test_tie_break_by_ts_code(self, tmp_path, monkeypatch):
        self._score(tmp_path, monkeypatch, _tie_cs_train_frame, _TIE_SCORES)
        frame = pd.read_parquet(tmp_path / "folds" / "split00" / _SIGNAL_TOPK_NAME)
        day_one_rows = [tuple(item) for item in frame.itertuples(index=False, name=None)][:3]
        assert day_one_rows == [
            ("20240102", 1, "000003.SZ", 0.9),
            ("20240102", 2, "000001.SZ", 0.5),
            ("20240102", 3, "000002.SZ", 0.5),
        ]

    def test_empty_selection_day_registered(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        frames = {
            "20240102": _signal_cs_train_frame("20240102"),
            "20240103": _excluded_cs_train_frame("20240103"),
        }
        monkeypatch.setattr(
            "src.lazybull.data.storage.Storage", lambda root_path: _FakeStorage(frames)
        )
        _patch_signal(monkeypatch, _FakeSignalWithFilters(_SIGNAL_SCORES))
        result = _score_cs_train(
            tmp_path,
            main_board_codes=set(_SIGNAL_SCORES),
            persist_predictions=True,
            predictions_path=Path("folds/split00/predictions.parquet"),
            signal_selection=_signal_selection(),
            signal_universe_config=_signal_universe_config(),
        )
        assert result["signal_topk_days"] == ["20240102"]
        assert result["signal_topk_empty_days"] == ["20240103"]
        assert result["signal_topk_rows"] == 3
        frame = pd.read_parquet(tmp_path / "folds" / "split00" / _SIGNAL_TOPK_NAME)
        assert set(frame["trade_date"]) == {"20240102"}

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_signal_scores_rejected_neither_artifact_written(
        self, tmp_path, monkeypatch, bad
    ):
        scores = {"000001.SZ": 0.9, "000007.SZ": bad}
        monkeypatch.chdir(tmp_path)
        _patch_cs_train_stack(
            monkeypatch, _FakeSignalWithFilters(scores), frame_fn=_nan_signal_frame
        )
        with pytest.raises(ValueError, match="NaN/Inf"):
            _score_cs_train(
                tmp_path,
                main_board_codes=set(scores),
                persist_predictions=True,
                predictions_path=Path("folds/split00/predictions.parquet"),
                signal_selection=_signal_selection(top_n=10),
            )
        # 两产物写前拒绝纪律：集成行洁净（000007 被可交易过滤剔除）但信号行含非法值
        # ⇒ 预测行与信号产物均不落盘（父目录亦无残留）
        assert not (tmp_path / "folds").exists()

    def test_double_write_byte_identical(self, tmp_path, monkeypatch):
        """同一输入两次持久化 ⇒ 信号产物文件字节一致（pyarrow + snappy + index=False）。"""
        monkeypatch.chdir(tmp_path)
        contents = []
        for name in ("run_a", "run_b"):
            _patch_cs_train_stack(
                monkeypatch,
                _FakeSignalWithFilters(_SIGNAL_SCORES),
                frame_fn=_signal_cs_train_frame,
            )
            _score_cs_train(
                tmp_path,
                main_board_codes=set(_SIGNAL_SCORES),
                persist_predictions=True,
                predictions_path=Path(name) / "predictions.parquet",
                signal_selection=_signal_selection(),
                signal_universe_config=_signal_universe_config(),
            )
            contents.append((tmp_path / name / _SIGNAL_TOPK_NAME).read_bytes())
        assert contents[0] == contents[1]

    def test_anchor_registers_relative_signal_path(self, tmp_path, monkeypatch):
        _patch_cs_train_stack(
            monkeypatch,
            _FakeSignalWithFilters(_SIGNAL_SCORES),
            frame_fn=_signal_cs_train_frame,
        )
        arm_dir = tmp_path / "arm_run"
        result = _score_cs_train(
            tmp_path,
            main_board_codes=set(_SIGNAL_SCORES),
            persist_predictions=True,
            predictions_path=arm_dir / "folds" / "split00" / "predictions.parquet",
            predictions_anchor=arm_dir,
            signal_selection=_signal_selection(),
        )
        assert result["signal_topk_path"] == f"folds/split00/{_SIGNAL_TOPK_NAME}"
        assert (arm_dir / "folds" / "split00" / _SIGNAL_TOPK_NAME).is_file()


class TestSignalTopkPanel:
    """panel 模式：过滤消费列并入读取契约；缺列 ⇒ store 契约明确报错。"""

    def test_panel_signal_topk_with_extended_contract(self, tmp_path, monkeypatch):
        scores = {"600000.SH": 0.95, "000001.SZ": 0.9, "000004.SZ": 0.85}
        store = _make_panel_signal_store(tmp_path / "store")
        _patch_signal(monkeypatch, _FakeSignalWithFilters(scores))
        result = score_model_fingerprint(
            models_dir=tmp_path,
            model_version=7,
            data_root=tmp_path,
            score_days=list(_DAYS),
            main_board_codes=set(scores),
            source_mode=SOURCE_MODE_PANEL,
            store=store,
            feature_columns=list(_FEATURE_COLUMNS),
            persist_predictions=True,
            predictions_path=tmp_path / "folds" / "split00" / "predictions.parquet",
            signal_selection=_signal_selection(),
            signal_universe_config=_signal_universe_config(),
        )
        topk_file = tmp_path / "folds" / "split00" / _SIGNAL_TOPK_NAME
        frame = pd.read_parquet(topk_file)
        expected = [
            ("20240102", 1, "000001.SZ", 0.9),
            ("20240103", 1, "000001.SZ", 0.9),
        ]
        assert list(frame.itertuples(index=False, name=None)) == expected
        assert result["signal_topk_rows"] == 2
        assert result["signal_topk_days"] == list(_DAYS)
        assert result["signal_topk_empty_days"] == []

    def test_panel_missing_filter_column_rejected(self, tmp_path, monkeypatch):
        """过滤消费列未登记 ⇒ 读取契约明确报错（禁止静默降级跳过过滤规则）。"""
        store = _make_panel_store(tmp_path / "store")  # 仅特征列 + 可交易过滤列
        scores = {"000001.SZ": 0.9}
        _patch_signal(monkeypatch, _FakeSignalWithFilters(scores))
        with pytest.raises(ValueError, match="特征列未登记"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=7,
                data_root=tmp_path,
                score_days=list(_DAYS),
                main_board_codes=set(scores),
                source_mode=SOURCE_MODE_PANEL,
                store=store,
                feature_columns=list(_FEATURE_COLUMNS),
                persist_predictions=True,
                predictions_path=tmp_path / "predictions.parquet",
                signal_selection=_signal_selection(),
                signal_universe_config=_signal_universe_config(),
            )


class TestSignalSelectionGuards:
    """signal_selection 误用守卫：打分前生效（无需替身打分栈即拒绝）。"""

    def test_signal_selection_requires_persist(self, tmp_path):
        with pytest.raises(ValueError, match="必须 persist_predictions=True"):
            _score_cs_train(tmp_path, signal_selection=_signal_selection())

    @pytest.mark.parametrize(
        "missing_key",
        ["top_n", "min_amount_ma20", "min_total_mv", "max_total_mv", "exclude_financial"],
    )
    def test_missing_key_rejected(self, tmp_path, missing_key):
        selection = _signal_selection()
        selection.pop(missing_key)
        with pytest.raises(ValueError, match="缺键"):
            _score_cs_train(
                tmp_path,
                persist_predictions=True,
                predictions_path=tmp_path / "predictions.parquet",
                signal_selection=selection,
            )

    def test_unknown_key_rejected(self, tmp_path):
        selection = _signal_selection()
        selection["min_turnover"] = 0.0
        with pytest.raises(ValueError, match="未知键"):
            _score_cs_train(
                tmp_path,
                persist_predictions=True,
                predictions_path=tmp_path / "predictions.parquet",
                signal_selection=selection,
            )


class TestSignalTopkDefaultOff:
    """signal_selection 默认 None：persist 开启但无信号键、无信号产物（现状逐位不变）。"""

    def test_persist_on_without_signal_selection_has_no_signal_keys(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _patch_cs_train_stack(monkeypatch, _FakeSignal(_SCORES_PLAN))
        result = _score_cs_train(
            tmp_path,
            persist_predictions=True,
            predictions_path=Path("folds/split00/predictions.parquet"),
        )
        for key in _SIGNAL_KEYS:
            assert key not in result
        assert not (tmp_path / "folds" / "split00" / _SIGNAL_TOPK_NAME).exists()
