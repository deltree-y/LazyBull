# -*- coding: utf-8 -*-
"""v2 P3 T0.5b spike 工具单元测试（含 T0.5b 评审 + 复核处置补验）。

覆盖：
- S1 指纹序列化（§13.3.3 冻结规格）：键升序、-0.0 归一、NaN/Inf 拒绝、
  ``%.17g``、UTF-8 + LF、sha256 稳定性；
- 隔离运行根构建（§13.3.4）：四只读联接 + 四全新目录、models 禁止联接、
  非空复用拒绝、生产根方向拒绝、S5 只读目录必须为联接且指向生产根；
- 生产模型目录守卫（§7.4-3）：零漂移、新增/删除/内容变化/索引文件哈希漂移；
- **R2-T05B-01 + N1 消费集合硬门**：按层真实命名（cs_train=``YYYYMMDD``、
  clean/daily 与 raw/suspend=``YYYY-MM-DD``）归一解析、daily/suspend 纳入且
  daily 计数 > 0、冻结段外排除、零覆盖拒绝、内容变更拒绝、集合外追加通过；
- **R2-T05B-02 实际训练边界**：日志切分行解析（三种子一致性）、双口径分列、
  val-raw 交叉校验、实际子集不得跨入 val-raw；
- **N3 独立预期日**：trade_cal 开市日为预期源、缺分区即失败并列出缺失日；
- **R2-T05B-03 覆盖/键域完整性**：单日失败/部分日失败/全日失败/空批次拒绝、
  预测长度 != 冻结输入行数拒绝（2 进 1 出 / 2 进 3 出）、ts_code 键不唯一拒绝；
- **R2-T05B-04 续跑**：身份一致可复用、代码/环境/配置/输入摘要变化拒绝、
  **端到端反例**（基线漂移/模型改写/基线缺失/快照不被覆盖）；
- **R2-T05B-05 逐种子诊断标签**：实际 random_state 标注、请求/实际顺序分列；
- S2/B6 特征名断言：逐列同名同序，乱序/缺列即失败；
- spike 命令组装：镜像批处理生产口径、配置摘要路径无关；
- OOS 打分日枚举：测试窗内 cs_train 分区日过滤与排序。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v2_p3.t05b_baselines import (  # noqa: E402
    enumerate_consumption_files,
    flatten_consumption_files,
)
from scripts.v2_p3.t05b_fingerprint import (  # noqa: E402
    assert_feature_names_aligned,
    environment_identity,
    fingerprint_prediction_rows,
    list_oos_score_days,
    resolve_sub_model_seed,
    score_model_fingerprint,
    serialize_prediction_rows,
    sha256_text,
)
from scripts.v2_p3.t05b_isolated_root import (  # noqa: E402
    FRESH_DIRS,
    READONLY_LINK_DIRS,
    assert_readonly_links,
    build_isolated_root,
    collect_consumption_snapshot,
    snapshot_production_models_guard,
    verify_consumption_snapshot,
    verify_production_models_guard,
)
from scripts.v2_p3.t05b_spike_runner import (  # noqa: E402
    _assert_coverage,
    _assert_identity_compatible,
    build_producer_identity,
    build_spike_walk_forward_args,
    collect_model_identity,
    list_expected_oos_days,
    parse_actual_train_boundary,
    read_split0_boundary,
    resolve_score_days,
    run_spike,
    spike_config_digest,
)


class TestS1Serialization:
    """S1 冻结序列化规格（-0.0 归一 / 拒绝 NaN、Inf / 键升序 / %.17g / UTF-8+LF）。"""

    def test_rows_sorted_by_key(self):
        rows = [("20240103", "000002.SZ", 1.5), ("20240102", "000001.SZ", 0.25)]
        text = serialize_prediction_rows(rows)
        assert text.splitlines() == ["20240102,000001.SZ,0.25", "20240103,000002.SZ,1.5"]

    def test_same_date_sorted_by_ts_code(self):
        rows = [("20240102", "000002.SZ", 1.0), ("20240102", "000001.SZ", 2.0)]
        text = serialize_prediction_rows(rows)
        assert text.splitlines()[0].endswith("000001.SZ,2")

    def test_negative_zero_normalized(self):
        text = serialize_prediction_rows([("20240102", "000001.SZ", -0.0)])
        assert text == "20240102,000001.SZ,0\n"

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN/Inf"):
            serialize_prediction_rows([("20240102", "000001.SZ", float("nan"))])

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="NaN/Inf"):
            serialize_prediction_rows([("20240102", "000001.SZ", float("inf"))])
        with pytest.raises(ValueError, match="NaN/Inf"):
            serialize_prediction_rows([("20240102", "000001.SZ", float("-inf"))])

    def test_lf_ending_and_utf8(self):
        text = serialize_prediction_rows([("20240102", "000001.SZ", 1.0)])
        assert text.endswith("\n") and "\r" not in text

    def test_float17g_roundtrip(self):
        value = 0.12345678901234567890
        text = serialize_prediction_rows([("20240102", "000001.SZ", value)])
        rendered = text.strip().rsplit(",", 1)[1]
        assert float(rendered) == value

    def test_fingerprint_stable(self):
        rows = [("20240102", "000001.SZ", 1.0), ("20240103", "000002.SZ", -2.5)]
        assert fingerprint_prediction_rows(rows) == fingerprint_prediction_rows(
            list(reversed(rows))
        )
        assert fingerprint_prediction_rows(rows) == sha256_text(serialize_prediction_rows(rows))

    def test_empty_rows(self):
        assert serialize_prediction_rows([]) == ""


class TestIsolatedRoot:
    """隔离运行根构建（四只读联接 + 四全新目录；models 禁止联接；S5 链接断言）。"""

    def _make_production(self, tmp_path: Path) -> Path:
        production = tmp_path / "data"
        for name in READONLY_LINK_DIRS:
            (production / name).mkdir(parents=True)
        return production

    def test_build_structure(self, tmp_path):
        production = self._make_production(tmp_path)
        run_dir = tmp_path / "run1"
        links: list[tuple[Path, Path]] = []
        root = build_isolated_root(
            run_dir,
            production,
            link_fn=lambda link, target: links.append((link, target)),
            verify_links=False,
        )
        assert root == run_dir.resolve() / "root"
        assert [link.name for link, _ in links] == list(READONLY_LINK_DIRS)
        assert all(target.parent == production.resolve() for _, target in links)
        for name in FRESH_DIRS:
            assert (root / name).is_dir()
        assert "models" not in [link.name for link, _ in links]

    def test_refuse_non_empty_reuse(self, tmp_path):
        production = self._make_production(tmp_path)
        run_dir = tmp_path / "run1"
        (run_dir / "root").mkdir(parents=True)
        (run_dir / "root" / "stale.txt").write_text("x", encoding="utf-8")
        with pytest.raises(FileExistsError, match="禁止复用"):
            build_isolated_root(run_dir, production, link_fn=lambda link, target: None)

    def test_refuse_production_inside_run_dir(self, tmp_path):
        run_dir = tmp_path / "run1"
        production = run_dir / "data"
        for name in READONLY_LINK_DIRS:
            (production / name).mkdir(parents=True)
        with pytest.raises(ValueError, match="不得位于隔离运行目录内"):
            build_isolated_root(run_dir, production, link_fn=lambda link, target: None)

    def test_missing_production_subdir(self, tmp_path):
        production = tmp_path / "data"
        production.mkdir()
        with pytest.raises(FileNotFoundError, match="无法只读接入"):
            build_isolated_root(tmp_path / "run1", production, link_fn=lambda link, target: None)

    def test_s5_real_dir_rejected(self, tmp_path):
        """S5：既有真实空目录冒充只读接入（_ensure_link 静默跳过情形）必须拒绝。"""
        production = self._make_production(tmp_path)
        root = tmp_path / "run1" / "root"
        for name in READONLY_LINK_DIRS:
            (root / name).mkdir(parents=True)
        with pytest.raises(ValueError, match="必须是联接"):
            assert_readonly_links(root, production)

    def test_s5_wrong_target_rejected(self, tmp_path):
        """S5：联接指向错误目标必须拒绝（有链接能力时验证，否则跳过）。"""
        production = self._make_production(tmp_path)
        other = self._make_production(tmp_path / "other")
        root = tmp_path / "run1" / "root"
        root.mkdir(parents=True)
        try:
            from scripts.v2_p3.t05b_isolated_root import _ensure_link

            for name in READONLY_LINK_DIRS:
                _ensure_link(root / name, production / name)
        except Exception as exc:  # pragma: no cover - 环境无链接能力
            pytest.skip(f"环境不支持目录联接: {exc}")
        with pytest.raises(ValueError, match="指向错误目标"):
            assert_readonly_links(root, other)
        assert_readonly_links(root, production)


class TestProductionModelsGuard:
    """生产模型目录守卫：清单 + 索引文件哈希双口径。"""

    def _make_models_dir(self, tmp_path: Path) -> Path:
        base = tmp_path / "data" / "models" / "stock_selection"
        base.mkdir(parents=True)
        (base / "ml_train_runs.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (base / "model_registry.json").write_text("{}", encoding="utf-8")
        (base / "v1_model.joblib").write_bytes(b"model-bytes")
        (base / "v1_metadata.json").write_text("{}", encoding="utf-8")
        return tmp_path / "data"

    def test_no_drift(self, tmp_path):
        production = self._make_models_dir(tmp_path)
        snapshot = snapshot_production_models_guard(production)
        assert verify_production_models_guard(snapshot) == []

    def test_detect_new_file(self, tmp_path):
        production = self._make_models_dir(tmp_path)
        snapshot = snapshot_production_models_guard(production)
        (production / "models" / "stock_selection" / "v2_model.joblib").write_bytes(b"new")
        drift = verify_production_models_guard(snapshot)
        assert any("新增文件" in item and "v2_model.joblib" in item for item in drift)

    def test_detect_deleted_file(self, tmp_path):
        production = self._make_models_dir(tmp_path)
        snapshot = snapshot_production_models_guard(production)
        (production / "models" / "stock_selection" / "v1_model.joblib").unlink()
        drift = verify_production_models_guard(snapshot)
        assert any("删除文件" in item for item in drift)

    def test_detect_index_content_change_same_size(self, tmp_path):
        production = self._make_models_dir(tmp_path)
        snapshot = snapshot_production_models_guard(production)
        target = production / "models" / "stock_selection" / "ml_train_runs.csv"
        target.write_text("a,b\n1,3\n", encoding="utf-8")
        drift = verify_production_models_guard(snapshot)
        assert any("索引文件内容变化" in item for item in drift)


class TestConsumptionGate:
    """R2-T05B-01 + N1 消费集合硬门（真实命名形态：daily/suspend=YYYY-MM-DD）。"""

    def _make_production(self, tmp_path: Path) -> Path:
        production = tmp_path / "data"
        (production / "features" / "cs_train").mkdir(parents=True)
        (production / "clean" / "daily").mkdir(parents=True)
        (production / "raw" / "suspend").mkdir(parents=True)
        (production / "raw").mkdir(parents=True, exist_ok=True)
        for day in ("20181123", "20190524", "20260702", "20260703"):
            (production / "features" / "cs_train" / f"{day}.parquet").write_bytes(
                b"cs" + day.encode()
            )
        for day in ("2018-11-23", "2019-05-24", "2026-07-02", "2026-07-03"):
            (production / "clean" / "daily" / f"{day}.parquet").write_bytes(b"d" + day.encode())
        for day in ("2018-11-23", "2026-07-02", "2026-07-03"):
            (production / "raw" / "suspend" / f"{day}.parquet").write_bytes(b"s" + day.encode())
        (production / "clean" / "stock_basic.parquet").write_bytes(b"sb")
        (production / "raw" / "trade_cal.parquet").write_bytes(b"tc")
        return production

    def test_enumerate_real_naming(self, tmp_path):
        production = self._make_production(tmp_path)
        by_dir = enumerate_consumption_files(production)
        # 真实 YYYY-MM-DD 命名的 daily / suspend 全部纳入（N1 修复前被静默丢弃）
        assert len(by_dir["clean/daily"]) == 3
        assert len(by_dir["raw/suspend"]) == 2
        assert len(by_dir["features/cs_train"]) == 3
        assert {p.name for p in by_dir["（单文件）"]} == {
            "stock_basic.parquet",
            "trade_cal.parquet",
        }
        # 冻结段外（2026-07-03 / 20260703）日更追加不入集合
        names = {p.name for p in flatten_consumption_files(by_dir)}
        assert "2026-07-03.parquet" not in names and "20260703.parquet" not in names

    def test_enumerate_zero_coverage_rejected(self, tmp_path):
        """任一按日分区目录冻结段内零覆盖即拒绝（零覆盖须可复核理由）。"""
        production = self._make_production(tmp_path)
        for path in (production / "raw" / "suspend").glob("*.parquet"):
            path.unlink()
        with pytest.raises(ValueError, match="零覆盖"):
            enumerate_consumption_files(production)

    def test_enumerate_missing_dir_rejected(self, tmp_path):
        (tmp_path / "data").mkdir()
        with pytest.raises(FileNotFoundError, match="消费集合目录缺失"):
            enumerate_consumption_files(tmp_path / "data")

    def _snapshot(self, tmp_path: Path):
        production = self._make_production(tmp_path)
        files = flatten_consumption_files(enumerate_consumption_files(production))
        snapshot_path = tmp_path / "snapshot.json"
        collect_consumption_snapshot(files, base_dir=tmp_path, out_path=snapshot_path)
        return production, snapshot_path

    def test_daily_content_change_rejected(self, tmp_path):
        """daily 分区内容变更 ⇒ 漂移（N1 补验用例）。"""
        production, snapshot_path = self._snapshot(tmp_path)
        assert verify_consumption_snapshot(snapshot_path, base_dir=tmp_path) == []
        target = production / "clean" / "daily" / "2018-11-23.parquet"
        target.write_bytes(b"tampered")
        drift = verify_consumption_snapshot(snapshot_path, base_dir=tmp_path)
        assert any("2018-11-23" in item for item in drift)

    def test_suspend_content_change_rejected(self, tmp_path):
        production, snapshot_path = self._snapshot(tmp_path)
        target = production / "raw" / "suspend" / "2018-11-23.parquet"
        target.write_bytes(b"tampered")
        drift = verify_consumption_snapshot(snapshot_path, base_dir=tmp_path)
        assert any("2018-11-23" in item for item in drift)

    def test_out_of_set_append_passes(self, tmp_path):
        _, snapshot_path = self._snapshot(tmp_path)
        production = tmp_path / "data"
        (production / "features" / "cs_train" / "20260704.parquet").write_bytes(b"new")
        (production / "clean" / "daily" / "2026-07-04.parquet").write_bytes(b"new")
        assert verify_consumption_snapshot(snapshot_path, base_dir=tmp_path) == []


class TestExpectedDays:
    """N3 独立预期日：trade_cal 开市日为预期源，缺分区即失败。"""

    def _stub_trade_cal(self, monkeypatch: pytest.MonkeyPatch, days: list[str]):
        cal = pd.DataFrame(
            {
                "exchange": ["SSE"] * len(days),
                "cal_date": days,
                "is_open": [1] * len(days),
                "pretrade_date": [None] * len(days),
            }
        )

        class _Loader:
            def load_clean_trade_cal(self):
                return cal

            def load_trade_cal(self):
                return cal

        monkeypatch.setattr("src.lazybull.data.loader.DataLoader", lambda storage: _Loader())
        monkeypatch.setattr("src.lazybull.data.storage.Storage", lambda root_path: object())

    def test_expected_from_trade_cal(self, tmp_path, monkeypatch):
        self._stub_trade_cal(monkeypatch, ["20240101", "20240102", "20240103", "20240110"])
        days = list_expected_oos_days(tmp_path, "20240102", "20240105")
        assert days == ["20240102", "20240103"]

    def test_missing_partition_rejected(self, tmp_path):
        """trade_cal 有 3 日、cs_train 只有 2 日 ⇒ 必须失败并列出缺失日（N3 补验）。"""
        cs_train = tmp_path / "cs_train"
        cs_train.mkdir()
        for day in ("20240102", "20240103"):
            (cs_train / f"{day}.parquet").write_bytes(b"x")
        with pytest.raises(RuntimeError, match="缺 cs_train 分区"):
            resolve_score_days(["20240102", "20240103", "20240104"], tmp_path)

    def test_resolve_passes_with_dash_names(self, tmp_path):
        cs_train = tmp_path / "cs_train"
        cs_train.mkdir()
        for day in ("20240102", "20240103"):
            (cs_train / f"{day}.parquet").write_bytes(b"x")
        assert resolve_score_days(["20240102", "20240103"], tmp_path) == ["20240102", "20240103"]


class TestActualTrainBoundary:
    """R2-T05B-02 实际训练边界：日志切分行解析 + 双口径登记。"""

    _LOG_TEMPLATE = (
        "18:13:05 | INFO | 按 trade_date 粒度切分（共 1460 个交易日，delta=21 天间隔）:\n"
        "18:13:05 | INFO |   训练集: {start} 至 {end}（{n} 个交易日，{m} 条样本）\n"
        "18:13:05 | INFO |   验证集: {vstart} 至 {vend}（{vn} 个交易日，{vm} 条样本）\n"
    )

    def _write_log(self, path: Path, blocks: int = 3, mismatch: bool = False) -> None:
        text = ""
        for idx in range(blocks):
            end = "20170811" if not mismatch or idx == 0 else "20170812"
            text += self._LOG_TEMPLATE.format(
                start="20121123",
                end=end,
                n=1147,
                m=1940423,
                vstart="20171018",
                vend="20181123",
                vn=271,
                vm=577980,
            )
        path.write_text(text, encoding="utf-8")

    def test_parse_three_seeds_agree(self, tmp_path):
        log_path = tmp_path / "run.log"
        self._write_log(log_path)
        actual = parse_actual_train_boundary(log_path)
        assert actual["train_actual_start"] == "20121123"
        assert actual["train_actual_end"] == "20170811"
        assert actual["train_actual_n_dates"] == 1147
        assert actual["train_actual_samples"] == 1940423
        assert actual["val_raw_log_start"] == "20171018"
        assert actual["val_raw_log_samples"] == 577980

    def test_parse_seed_disagree_rejected(self, tmp_path):
        log_path = tmp_path / "run.log"
        self._write_log(log_path, mismatch=True)
        with pytest.raises(ValueError, match="边界不一致"):
            parse_actual_train_boundary(log_path)

    def test_parse_missing_rejected(self, tmp_path):
        log_path = tmp_path / "run.log"
        log_path.write_text("no boundary here", encoding="utf-8")
        with pytest.raises(ValueError, match="未找到"):
            parse_actual_train_boundary(log_path)

    def _make_root(self, tmp_path: Path) -> Path:
        root = tmp_path / "root"
        models = root / "models" / "stock_selection"
        models.mkdir(parents=True)
        row = {
            "split_index": 0,
            "start_date": 20121123,
            "end_date": 20181123,
            "test_start_date": 20181126,
            "test_end_date": 20190524,
            "val_raw_start_date": 20171018,
            "val_raw_end_date": 20181123,
            "val_raw_n_dates": 271,
            "val_raw_samples": 577980,
            "val_es_start_date": 20171018,
            "val_es_end_date": 20180720,
            "val_es_n_dates": 187,
            "val_es_samples": 384202,
            "val_calib_start_date": 20180723,
            "val_calib_end_date": 20181025,
            "val_calib_n_dates": 63,
            "val_calib_samples": 144287,
            "val_embargo_start_date": 20181026,
            "val_embargo_end_date": 20181123,
            "val_embargo_n_dates": 21,
            "val_embargo_samples": 49491,
            "train_samples": 1940423,
            "val_samples": 384202,
            "total_samples": 2639970,
            "samples_after_filter": 2600131,
            "trade_days_count": 1460,
            "best_iteration": 490,
            "n_jobs": 20,
        }
        pd.DataFrame([row]).to_csv(models / "ml_train_runs.csv", index=False, encoding="utf-8-sig")
        return root

    def test_dual_register(self, tmp_path):
        root = self._make_root(tmp_path)
        log_path = tmp_path / "run.log"
        self._write_log(log_path)
        boundary = read_split0_boundary(root, log_path)
        assert boundary["outer_train_start"] == "20121123"
        assert boundary["outer_train_end"] == "20181123"
        assert boundary["train_actual_end"] == "20170811"
        assert boundary["train_actual_end"] < boundary["val_raw_start"]
        assert boundary["train_actual_n_dates"] == 1147
        assert boundary["train_actual_samples"] == boundary["train_samples"]
        assert boundary["samples_after_filter"] == 2600131
        assert boundary["val_es_samples_alias"] == boundary["val_es_samples"]

    def test_val_raw_crosscheck_rejected(self, tmp_path):
        root = self._make_root(tmp_path)
        log_path = tmp_path / "run.log"
        self._write_log(log_path)
        text = log_path.read_text(encoding="utf-8").replace("20181123（271", "20181124（271")
        log_path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="val-raw 与台账不一致"):
            read_split0_boundary(root, log_path)


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
    """替身 MLSignal：请求种子 [42,61,82]、实际聚合顺序 [61,42,82]。"""

    def __init__(
        self,
        reject_days: set[str] | None = None,
        main_predict_len: int | None = None,
        sub_predict_len: int | None = None,
    ):
        self._reject_days = reject_days or set()
        self._main_predict_len = main_predict_len
        self._sub_predict_len = sub_predict_len
        self.metadata = {"train_params": {"ensemble_seeds": [42, 61, 82]}}
        self.feature_columns = ["f1", "f2"]
        self.model = _FakeEnsemble(
            [_FakeSubModel(61, 1.0), _FakeSubModel(42, 2.0), _FakeSubModel(82, 3.0)]
        )
        if sub_predict_len is not None:
            for sub in self.model.models:
                sub.predict = lambda X, n=sub_predict_len: np.full(n, 1.0, dtype=float)

    def _load_model(self):
        return None

    def _build_model_input(self, date, df):
        day = date.strftime("%Y%m%d")
        if day in self._reject_days:
            return None
        return df, df[["f1", "f2"]]

    def _predict_scores(self, X, before, after, ranked):
        n = self._main_predict_len if self._main_predict_len is not None else len(X)
        return np.arange(n, dtype=float)


def _fake_cs_train(day: str, all_st: bool = False, dup_keys: bool = False) -> pd.DataFrame:
    ts_codes = ["000001.SZ", "000001.SZ"] if dup_keys else ["000001.SZ", "000002.SZ"]
    return pd.DataFrame(
        {
            "ts_code": ts_codes,
            "is_st": [all_st, all_st],
            "is_suspended": [False, False],
            "is_limit_up": [False, False],
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
        }
    )


class _FakeStorage:
    def __init__(self, frames: dict[str, pd.DataFrame | None]):
        self._frames = frames

    def load_cs_train_day(self, day: str):
        return self._frames.get(day)


def _patch_signal_stack(monkeypatch: pytest.MonkeyPatch, frames, signal: _FakeSignal) -> None:
    storage = _FakeStorage(frames)
    monkeypatch.setattr("src.lazybull.data.storage.Storage", lambda root_path: storage)
    monkeypatch.setattr("src.lazybull.v2.core.signal.ml_signal.MLSignal", lambda **kwargs: signal)


def _score(tmp_path, signal, days=("20240102",), codes=None):
    frames = {day: _fake_cs_train(day) for day in days}
    return frames, signal


class TestCoverageAndKeyDomain:
    """R2-T05B-03 覆盖 + 键域完整性门。"""

    def test_full_coverage_pass_with_actual_seed_labels(self, tmp_path, monkeypatch):
        frames = {day: _fake_cs_train(day) for day in ("20240102", "20240103")}
        _patch_signal_stack(monkeypatch, frames, _FakeSignal())
        result = score_model_fingerprint(
            models_dir=tmp_path,
            model_version=1,
            data_root=tmp_path,
            score_days=["20240102", "20240103"],
            main_board_codes={"000001.SZ", "000002.SZ"},
        )
        assert result["n_days"] == 2 and result["n_rows"] == 4
        assert [item["seed"] for item in result["per_seed_diagnostic"]] == [61, 42, 82]
        assert result["requested_ensemble_seeds"] == [42, 61, 82]
        assert result["actual_seed_order"] == [61, 42, 82]

    def test_missing_partition_rejected(self, tmp_path, monkeypatch):
        frames: dict[str, pd.DataFrame | None] = {"20240102": _fake_cs_train("20240102")}
        _patch_signal_stack(monkeypatch, frames, _FakeSignal())
        with pytest.raises(RuntimeError, match="覆盖完整性门拒绝"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102", "20240103"],
                main_board_codes={"000001.SZ"},
            )

    def test_empty_section_rejected(self, tmp_path, monkeypatch):
        frames = {
            "20240102": _fake_cs_train("20240102"),
            "20240103": _fake_cs_train("20240103", all_st=True),
        }
        _patch_signal_stack(monkeypatch, frames, _FakeSignal())
        with pytest.raises(RuntimeError, match="过滤后截面为空"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102", "20240103"],
                main_board_codes={"000001.SZ"},
            )

    def test_quality_gate_reject_day_rejected(self, tmp_path, monkeypatch):
        frames = {day: _fake_cs_train(day) for day in ("20240102", "20240103")}
        _patch_signal_stack(monkeypatch, frames, _FakeSignal(reject_days={"20240103"}))
        with pytest.raises(RuntimeError, match="质量门禁拒绝"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102", "20240103"],
                main_board_codes={"000001.SZ"},
            )

    def test_all_days_failed_empty_batch_rejected(self, tmp_path, monkeypatch):
        frames: dict[str, pd.DataFrame | None] = {}
        _patch_signal_stack(monkeypatch, frames, _FakeSignal())
        with pytest.raises(RuntimeError, match="覆盖完整性门拒绝"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102"],
                main_board_codes={"000001.SZ"},
            )

    def test_short_main_prediction_rejected(self, tmp_path, monkeypatch):
        """输入两行、主预测一行 ⇒ 拒绝（R2-T05B-03 复核补验：禁止 zip 截断）。"""
        frames = {"20240102": _fake_cs_train("20240102")}
        _patch_signal_stack(monkeypatch, frames, _FakeSignal(main_predict_len=1))
        with pytest.raises(RuntimeError, match="主预测长度"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102"],
                main_board_codes={"000001.SZ", "000002.SZ"},
            )

    def test_long_main_prediction_rejected(self, tmp_path, monkeypatch):
        """输入两行、主预测三行 ⇒ 拒绝。"""
        frames = {"20240102": _fake_cs_train("20240102")}
        _patch_signal_stack(monkeypatch, frames, _FakeSignal(main_predict_len=3))
        with pytest.raises(RuntimeError, match="主预测长度"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102"],
                main_board_codes={"000001.SZ", "000002.SZ"},
            )

    def test_short_sub_prediction_rejected(self, tmp_path, monkeypatch):
        frames = {"20240102": _fake_cs_train("20240102")}
        _patch_signal_stack(monkeypatch, frames, _FakeSignal(sub_predict_len=1))
        with pytest.raises(RuntimeError, match="子模型"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102"],
                main_board_codes={"000001.SZ", "000002.SZ"},
            )

    def test_duplicate_keys_rejected(self, tmp_path, monkeypatch):
        frames = {"20240102": _fake_cs_train("20240102", dup_keys=True)}
        _patch_signal_stack(monkeypatch, frames, _FakeSignal())
        with pytest.raises(RuntimeError, match="键不唯一"):
            score_model_fingerprint(
                models_dir=tmp_path,
                model_version=1,
                data_root=tmp_path,
                score_days=["20240102"],
                main_board_codes={"000001.SZ"},
            )

    def test_runner_coverage_assert(self):
        fingerprint = {"n_days": 1, "n_rows": 10, "fingerprint": "abc", "score_days": ["20240102"]}
        with pytest.raises(RuntimeError, match="独立预期日不一致"):
            _assert_coverage(fingerprint, ["20240102", "20240103"], run_index=1)
        empty_fp = {
            "n_days": 1,
            "n_rows": 0,
            "fingerprint": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "score_days": ["20240102"],
        }
        with pytest.raises(RuntimeError, match="空预测批次"):
            _assert_coverage(empty_fp, ["20240102"], run_index=2)
        _assert_coverage(
            {"n_days": 1, "n_rows": 5, "fingerprint": "abc", "score_days": ["20240102"]},
            ["20240102"],
            1,
        )


class TestIdentityReuse:
    """R2-T05B-04 续跑复用前生产者身份核验（含输入基线摘要）。"""

    def _identity(self, source: str = "s1", config: str = "c1", digest: str = "d1") -> dict:
        return {
            "run_index": 1,
            "source_digest": source,
            "config_digest": config,
            "input_snapshot_digest": digest,
            "environment_identity": {"python": "3.12.10", "cpu_count": 20},
        }

    def test_identical_passes(self):
        _assert_identity_compatible(self._identity(), self._identity())

    def test_source_change_rejected(self):
        with pytest.raises(ValueError, match="禁止复用"):
            _assert_identity_compatible(self._identity(source="old"), self._identity(source="new"))

    def test_input_digest_change_rejected(self):
        with pytest.raises(ValueError, match="禁止复用"):
            _assert_identity_compatible(self._identity(digest="old"), self._identity(digest="new"))

    def test_env_change_rejected(self):
        saved = self._identity()
        current = self._identity()
        current["environment_identity"]["cpu_count"] = 16
        with pytest.raises(ValueError, match="禁止复用"):
            _assert_identity_compatible(saved, current)

    def test_config_digest_path_independent(self):
        digest1 = spike_config_digest()
        digest2 = spike_config_digest()
        assert digest1 == digest2 and len(digest1) == 64


class _ResumeFixture:
    """端到端续跑夹具：真实基线快照 + 伪造指纹/模型文件 + 替身训练与打分。"""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.production = self._make_production()
        self.spike_base = tmp_path / "spike"
        self.run_dir = self.spike_base / "r"
        self.run_dir.mkdir(parents=True)
        self._write_baselines()
        self._write_runs()
        self._patch_externals()

    def _make_production(self) -> Path:
        production = self.tmp_path / "data"
        (production / "features" / "cs_train").mkdir(parents=True)
        (production / "clean" / "daily").mkdir(parents=True)
        (production / "raw" / "suspend").mkdir(parents=True)
        (production / "models" / "stock_selection").mkdir(parents=True)
        (production / "features" / "cs_train" / "20240102.parquet").write_bytes(b"cs")
        (production / "clean" / "daily" / "2024-01-02.parquet").write_bytes(b"d")
        (production / "raw" / "suspend" / "2024-01-02.parquet").write_bytes(b"s")
        (production / "clean" / "stock_basic.parquet").write_bytes(b"sb")
        (production / "raw" / "trade_cal.parquet").write_bytes(b"tc")
        (production / "models" / "stock_selection" / "v9_model.joblib").write_bytes(b"prod")
        return production

    def _write_baselines(self) -> None:
        guard = snapshot_production_models_guard(self.production)
        (self.run_dir / "before_guard.json").write_text(
            json.dumps(guard, ensure_ascii=False), encoding="utf-8"
        )
        files = flatten_consumption_files(enumerate_consumption_files(self.production))
        collect_consumption_snapshot(
            files, base_dir=self.tmp_path, out_path=self.run_dir / "consumption_snapshot.json"
        )

    def _write_runs(self) -> None:
        digest = self._input_digest()
        for run_index in (1, 2):
            root = self.run_dir / f"run{run_index}" / "root"
            models = root / "models" / "stock_selection"
            models.mkdir(parents=True)
            for suffix in ("_model.joblib", "_metadata.json", "_features.json"):
                (models / f"v1{suffix}").write_bytes(b"model" + str(run_index).encode())
            identity = build_producer_identity(run_index, digest)
            saved = {
                "run_index": run_index,
                "reused": False,
                "model_version": 1,
                "model_identity": collect_model_identity(root, 1),
                "fingerprint": {
                    "model_version": 1,
                    "n_days": 1,
                    "n_rows": 2,
                    "score_days": ["20240102"],
                    "fingerprint": "f" * 64,
                    "per_seed_diagnostic": [{"seed": 61, "fingerprint": "e" * 64}],
                    "producer_identity": identity,
                },
            }
            (self.run_dir / f"run{run_index}").mkdir(parents=True, exist_ok=True)
            (self.run_dir / f"run{run_index}" / "fingerprint.json").write_text(
                json.dumps(saved, ensure_ascii=False), encoding="utf-8"
            )

    def _input_digest(self) -> str:
        import hashlib

        return hashlib.sha256((self.run_dir / "consumption_snapshot.json").read_bytes()).hexdigest()

    def _patch_externals(self) -> None:
        def _no_training(*args, **kwargs):
            raise AssertionError("续跑路径不得触发训练")

        self.monkeypatch.setattr("scripts.v2_p3.t05b_spike_runner._run_walk_forward", _no_training)
        self.monkeypatch.setattr(
            "scripts.v2_p3.t05b_spike_runner.build_isolated_root",
            lambda sub_dir, production, **kw: (_ for _ in ()).throw(
                AssertionError("续跑路径不得重建隔离根")
            ),
        )

    def resume(self):
        return run_spike(
            "r",
            self.production,
            skip_existing=True,
            spike_base_dir=self.spike_base,
        )


class TestResumeEndToEnd:
    """R2-T05B-04 复核余项：端到端续跑反例（先核验后复用，禁止覆盖基线）。"""

    def test_clean_resume_passes_without_overwrite(self, tmp_path, monkeypatch):
        fixture = _ResumeFixture(tmp_path, monkeypatch)
        guard_before = (fixture.run_dir / "before_guard.json").read_bytes()
        consumption_before = (fixture.run_dir / "consumption_snapshot.json").read_bytes()
        evidence = fixture.resume()
        assert evidence["comparison"]["verdict"] == "PASS"
        assert [run["reused"] for run in evidence["runs"]] == [True, True]
        # 基线快照不得被改写（先核验后复用；append-only）
        assert (fixture.run_dir / "before_guard.json").read_bytes() == guard_before
        assert (fixture.run_dir / "consumption_snapshot.json").read_bytes() == consumption_before

    def test_tampered_consumption_rejected(self, tmp_path, monkeypatch):
        fixture = _ResumeFixture(tmp_path, monkeypatch)
        (fixture.production / "features" / "cs_train" / "20240102.parquet").write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match="基线漂移"):
            fixture.resume()

    def test_tampered_production_models_rejected(self, tmp_path, monkeypatch):
        fixture = _ResumeFixture(tmp_path, monkeypatch)
        (fixture.production / "models" / "stock_selection" / "v9_model.joblib").write_bytes(b"x")
        with pytest.raises(RuntimeError, match="基线漂移"):
            fixture.resume()

    def test_tampered_isolated_model_rejected(self, tmp_path, monkeypatch):
        fixture = _ResumeFixture(tmp_path, monkeypatch)
        target = (
            fixture.run_dir / "run1" / "root" / "models" / "stock_selection" / "v1_model.joblib"
        )
        target.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="模型文件身份"):
            fixture.resume()

    def test_missing_baseline_rejected(self, tmp_path, monkeypatch):
        fixture = _ResumeFixture(tmp_path, monkeypatch)
        (fixture.run_dir / "consumption_snapshot.json").unlink()
        with pytest.raises(RuntimeError, match="基线缺失"):
            fixture.resume()

    def test_both_baselines_missing_with_cache_rejected(self, tmp_path, monkeypatch):
        """复核二余项：已有缓存但两份基线同时缺失 ⇒ 拒绝，且不得重建替代快照。"""
        fixture = _ResumeFixture(tmp_path, monkeypatch)
        (fixture.run_dir / "before_guard.json").unlink()
        (fixture.run_dir / "consumption_snapshot.json").unlink()
        with pytest.raises(RuntimeError, match="原始基线缺失"):
            fixture.resume()
        assert not (fixture.run_dir / "before_guard.json").exists()
        assert not (fixture.run_dir / "consumption_snapshot.json").exists()

    def test_both_baselines_missing_and_production_changed_rejected(self, tmp_path, monkeypatch):
        """复核二反例：双基线同时缺失 + 生产模型变化 ⇒ 仍拒绝（禁止重置基线后复用）。"""
        fixture = _ResumeFixture(tmp_path, monkeypatch)
        (fixture.production / "models" / "stock_selection" / "v9_model.joblib").write_bytes(b"x")
        (fixture.run_dir / "before_guard.json").unlink()
        (fixture.run_dir / "consumption_snapshot.json").unlink()
        with pytest.raises(RuntimeError, match="原始基线缺失"):
            fixture.resume()
        assert not (fixture.run_dir / "before_guard.json").exists()

    def test_no_skip_launch_with_cache_and_missing_baselines_rejected(self, tmp_path, monkeypatch):
        """复核三反例：不带 --skip-existing 启动（缓存存在 + 双基线缺失）⇒ 在任何
        快照写入前拒绝；缺失基线检查独立于 skip_existing。"""
        fixture = _ResumeFixture(tmp_path, monkeypatch)
        (fixture.run_dir / "before_guard.json").unlink()
        (fixture.run_dir / "consumption_snapshot.json").unlink()
        with pytest.raises(RuntimeError, match="原始基线缺失"):
            run_spike(
                "r",
                fixture.production,
                skip_existing=False,
                spike_base_dir=fixture.spike_base,
            )
        # 拒绝先于任何快照写入：两份基线不得被重建
        assert not (fixture.run_dir / "before_guard.json").exists()
        assert not (fixture.run_dir / "consumption_snapshot.json").exists()

    def test_tampered_source_identity_rejected(self, tmp_path, monkeypatch):
        fixture = _ResumeFixture(tmp_path, monkeypatch)
        path = fixture.run_dir / "run1" / "fingerprint.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["fingerprint"]["producer_identity"]["source_digest"] = "tampered"
        path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError, match="禁止复用"):
            fixture.resume()


class TestFeatureNamesAssertion:
    """S2/B6：特征名逐列同名同序，禁止交集降级。"""

    def test_aligned_passes(self):
        assert_feature_names_aligned(["a", "b", "c"], ["a", "b", "c"], context="t")

    def test_reorder_rejected(self):
        with pytest.raises(ValueError, match="同名同序断言失败"):
            assert_feature_names_aligned(["a", "b"], ["b", "a"], context="t")

    def test_subset_rejected(self):
        with pytest.raises(ValueError, match="同名同序断言失败"):
            assert_feature_names_aligned(["a", "b", "c"], ["a", "b"], context="t")

    def test_resolve_seed_from_wrapper(self):
        assert resolve_sub_model_seed(_FakeSubModel(61, 0.0)) == 61

        class Wrapper:
            def __init__(self):
                self.base_model = _FakeSubModel(42, 0.0)

        assert resolve_sub_model_seed(Wrapper()) == 42
        assert resolve_sub_model_seed(object()) is None


class TestSpikeCommand:
    """spike 命令镜像批处理生产口径（关键参数逐项在列）。"""

    def test_core_production_signature(self, tmp_path):
        args = build_spike_walk_forward_args(tmp_path / "root", tmp_path / "summary.csv")

        def value_of(flag: str) -> str:
            return args[args.index(flag) + 1]

        assert value_of("--algorithm") == "xgboost"
        assert value_of("--split-count") == "14"
        assert value_of("--final-date") == "20260105"
        assert value_of("--train-window-years") == "6"
        assert value_of("--test-window-months") == "6"
        assert value_of("--val-ratio") == "0.2"
        assert value_of("--label") == "neu_y_ret_20"
        assert value_of("--task") == "regression"
        assert value_of("--label-transform") == "cs_zscore"
        assert value_of("--objective") == "mse"
        assert value_of("--n-estimators") == "500"
        assert value_of("--max-depth") == "5"
        assert value_of("--learning-rate") == "0.03"
        assert value_of("--min-child-weight") == "200"
        assert value_of("--colsample-bytree") == "0.3"
        assert value_of("--subsample") == "0.8"
        assert value_of("--reg-alpha") == "0.03"
        assert value_of("--reg-lambda") == "7"
        assert value_of("--gamma") == "0.5"
        assert value_of("--early-stopping-rounds") == "50"
        assert value_of("--early-stopping-metric") == "rank_ic_daily"
        assert value_of("--min-best-iteration") == "30"
        assert value_of("--rank-weight-topk") == "120"
        assert value_of("--rank-weight") == "100"
        assert value_of("--rank-weight-topk-weight-mode") == "linear_decay"
        assert value_of("--freshness-strategy") == "state_keep_event_decay"
        assert value_of("--event-freshness-half-life-days") == "120"
        assert value_of("--ensemble-seeds") == "42,61,82"
        assert value_of("--stock-domain") == "main"

    def test_spike_specific_switches(self, tmp_path):
        args = build_spike_walk_forward_args(tmp_path / "root", tmp_path / "summary.csv")
        assert args[args.index("--selected-split-indices") + 1] == "0"
        assert "--no-deploy-train" in args
        assert args[args.index("--data-root") + 1] == str(tmp_path / "root")
        for flag in (
            "--enable-fundamental-features",
            "--enable-alt-features",
            "--enable-margin-features",
            "--enable-cyq-features",
            "--enable-fund-features",
            "--enable-express-features",
            "--enable-north-features",
            "--enable-lhb-features",
            "--enable-consensus-features",
            "--enable-enhanced-features",
            "--enable-cashflow-quality-features",
        ):
            assert flag in args
        for absent in (
            "--enable-consensus-revision-features",
            "--enable-dividend-policy-features",
            "--factor-prune",
            "--feature-stability-filter",
            "--exposure-policy",
            "--downside-penalty",
        ):
            assert absent not in args


class TestScoreDays:
    """OOS 打分日枚举（历史辅助函数，现由 N3 预期日流程接管）。"""

    def test_filter_and_sort(self, tmp_path):
        cs_train = tmp_path / "cs_train"
        cs_train.mkdir()
        for day in ("20240105", "20240102", "20240103", "20231229", "20240109"):
            (cs_train / f"{day}.parquet").write_bytes(b"x")
        days = list_oos_score_days(tmp_path, "20240102", "20240105")
        assert days == ["20240102", "20240103", "20240105"]

    def test_empty_window_rejected(self, tmp_path):
        (tmp_path / "cs_train").mkdir()
        with pytest.raises(ValueError, match="无 cs_train 分区"):
            list_oos_score_days(tmp_path, "20240102", "20240105")


class TestEnvironmentIdentity:
    def test_required_keys(self):
        identity = environment_identity()
        for key in ("python", "pandas", "numpy", "pyarrow", "xgboost", "cpu_count", "env_vars"):
            assert key in identity
        assert "OMP_NUM_THREADS" in identity["env_vars"]
