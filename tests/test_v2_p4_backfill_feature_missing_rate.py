# -*- coding: utf-8 -*-
"""旧模型 feature_missing_rate 补录工具测试（B′：schema v2 分布摘要）。

锁定要点：
- `build_addendum` 输出**三键**（schema v2 / 纯投影分布 dict / stat）——首版误把
  纯投影 dict 展开成 train_params 顶层键（生产已回滚重写），此处以断言防回归；
- 端到端 scan → 逐日率 → 分布摘要 → 校验 → 双份写入 → 读侧校验通过；
- 幂等保护、备份存在、fail-closed 校验分支（结构/池化健全性/键集）。
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.v2_p4 import backfill_feature_missing_rate as tool
from src.lazybull.data import DataLoader, Storage
from src.lazybull.v2.common.model_schema import (
    FEATURE_MISSING_RATE_SCHEMA_VERSION,
    SELECTION_FILTER_COLUMNS,
    read_feature_missing_rate,
)

_DAYS = ["20260105", "20260106"]
_MAIN_CODES = ["600000.SH", "600001.SH"]
_GEM_CODE = "300001.SZ"
_FEATURES = ["f1", "f2"]


def _write_root(root: Path, *, label_nan_main: bool = False) -> Storage:
    (root / "clean").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "cal_date": _DAYS,
            "is_open": [1, 1],
            "pretrade_date": ["20260102", "20260105"],
        }
    ).to_parquet(root / "clean" / "trade_cal.parquet", index=False)
    pd.DataFrame(
        {
            "ts_code": [*_MAIN_CODES, _GEM_CODE],
            "market": ["主板", "主板", "创业板"],
            "name": ["A", "B", "C"],
        }
    ).to_parquet(root / "clean" / "stock_basic.parquet", index=False)

    cs = root / "features" / "cs_train"
    cs.mkdir(parents=True, exist_ok=True)
    for day in _DAYS:
        codes = [*_MAIN_CODES, _GEM_CODE]
        payload = {
            "trade_date": [day] * len(codes),
            "ts_code": codes,
            "f1": [0.1, float("nan"), 0.3],
            "f2": [float("nan"), 0.2, 0.3],
            "amount_ma20": [1.0e4] * 3,
            "total_mv": [1.0e8] * 3,
            "sw_l1_code": ["801780"] * 3,
            "is_st": [0, 0, 0],
            "is_suspended": [0, 0, 0],
            "is_limit_up": [0, 0, 0],
            "neu_y_ret_20": [
                0.01,
                float("nan") if label_nan_main else 0.02,
                float("nan"),
            ],
            "y_ret_20": [0.01, 0.02, 0.03],
        }
        pd.DataFrame(payload).to_parquet(cs / f"{day}.parquet", index=False)
    return Storage(root_path=str(root))


def _write_model(models_dir: Path, version: int = 99) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": version,
        "version_str": f"v{version}",
        "model_type": "xgboost_regression_wf",
        "model_file": f"v{version}_model.joblib",
        "features_file": f"v{version}_features.json",
        "train_start_date": _DAYS[0],
        "train_end_date": _DAYS[-1],
        "feature_count": len(_FEATURES),
        "label_column": "neu_y_ret_20",
        "n_samples": 2,
        "train_params": {"learning_rate": 0.03, "is_deploy": True},
        "performance_metrics": {},
    }
    (models_dir / f"v{version}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (models_dir / f"v{version}_features.json").write_text(
        json.dumps(_FEATURES, ensure_ascii=False), encoding="utf-8"
    )
    registry = {"models": [dict(metadata)], "next_version": version + 1}
    (models_dir / "model_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _scan(tmp_path, *, label_nan_main: bool = False):
    storage = _write_root(tmp_path, label_nan_main=label_nan_main)
    loader = DataLoader(storage)
    models_dir = tmp_path / "models"
    _write_model(models_dir)
    spec = tool.load_spec(models_dir, 99)
    counts, data_state = tool.scan(spec, storage, loader)
    return models_dir, spec, counts, data_state


def _build(models_dir, spec, counts, data_state) -> dict:
    distribution = tool.compute_distribution(counts, spec)
    pooled = tool.compute_pooled(counts, spec)
    tool.validate(distribution, pooled, spec)
    return tool.build_addendum(distribution, spec, counts, data_state, "evidence.json")


class TestBuildAddendum:
    def test_three_keys_and_pure_projection(self, tmp_path):
        models_dir, spec, counts, data_state = _scan(tmp_path)
        addendum = _build(models_dir, spec, counts, data_state)
        assert set(addendum) == {
            "feature_missing_rate_schema_version",
            "feature_missing_rate",
            "feature_missing_rate_stat",
        }
        assert (
            addendum["feature_missing_rate_schema_version"]
            == FEATURE_MISSING_RATE_SCHEMA_VERSION
            == 2
        )
        rate = addendum["feature_missing_rate"]
        assert set(rate) == set(_FEATURES) | set(SELECTION_FILTER_COLUMNS)
        sample = rate["f1"]
        assert {"p50", "p90", "p95", "p99", "max", "mean", "n_days", "p99_unstable"} <= set(sample)
        assert sample["n_days"] == 2
        assert sample["p99_unstable"] is True  # 短窗（2 日）⇒ p99 退化为 p95
        stat = addendum["feature_missing_rate_stat"]
        assert stat["source"] == "backfilled"
        assert stat["coverage_n_samples"] == (
            stat["n_samples_after_filter"] - stat["n_dropped_label_nan"]
        )
        coverage, error = read_feature_missing_rate(addendum)
        assert error is None and coverage == rate


class TestScanAndDistribution:
    def test_domain_filter_and_daily_rate_sequence(self, tmp_path):
        _models_dir, spec, counts, data_state = _scan(tmp_path)
        assert counts.n_samples_after_filter == 4  # 域过滤后 2 日 × 2 行
        assert counts.n_rows == 4
        assert counts.n_dropped_label_nan == 0
        assert data_state["n_trade_days"] == 2
        assert len(counts.daily_rates["f1"]) == 2  # 逐日率序列
        distribution = tool.compute_distribution(counts, spec)
        assert distribution["f1"]["p95"] == pytest.approx(0.5)
        assert distribution["amount_ma20"]["p95"] == pytest.approx(0.0)
        pooled = tool.compute_pooled(counts, spec)
        assert pooled["f1"] == pytest.approx(0.5)

    def test_label_nan_rows_dropped_from_denominator(self, tmp_path):
        _models_dir, spec, counts, data_state = _scan(tmp_path, label_nan_main=True)
        assert counts.n_samples_after_filter == 4
        assert counts.n_rows == 2
        assert counts.n_dropped_label_nan == 2
        # reviewer3 P2-1：actual_data_end / n_trade_days 按**标签可用**口径
        assert data_state["actual_data_end"] == _DAYS[-1]
        assert data_state["n_trade_days"] == 2

    def test_verify_backfill_reports_columns_and_detects_split_brain(self, tmp_path):
        models_dir, spec, counts, data_state = _scan(tmp_path)
        addendum = _build(models_dir, spec, counts, data_state)
        tool.write_backfill(models_dir, spec, addendum, {"feature_missing_rate_sha256": "x"})
        results = tool.verify_backfill(models_dir, [spec.version])
        assert results[spec.version]["n_columns"] == len(addendum["feature_missing_rate"])
        # 模拟 registry 单边被改（split-brain）⇒ 全量校验必须失败
        registry_path = models_dir / "model_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        entry = next(i for i in registry["models"] if int(i["version"]) == spec.version)
        entry["train_params"]["feature_missing_rate"]["f1"]["p95"] = 0.99
        registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(RuntimeError, match="双份不一致"):
            tool.verify_backfill(models_dir, [spec.version])


class TestValidateFailClosed:
    def test_keyset_mismatch_rejected(self, tmp_path):
        _models_dir, spec, counts, _ = _scan(tmp_path)
        distribution = tool.compute_distribution(counts, spec)
        pooled = tool.compute_pooled(counts, spec)
        distribution.pop("f1")
        with pytest.raises(ValueError, match="键集"):
            tool.validate(distribution, pooled, spec)

    def test_invalid_distribution_rejected(self, tmp_path):
        _models_dir, spec, counts, _ = _scan(tmp_path)
        distribution = tool.compute_distribution(counts, spec)
        pooled = tool.compute_pooled(counts, spec)
        distribution["f1"]["p95"] = 0.9  # p95 > p99 ⇒ 分位序非法
        with pytest.raises(ValueError, match="结构非法"):
            tool.validate(distribution, pooled, spec)

    def test_pooled_over_limit_rejected(self, tmp_path):
        _models_dir, spec, counts, _ = _scan(tmp_path)
        distribution = tool.compute_distribution(counts, spec)
        pooled = tool.compute_pooled(counts, spec)
        pooled["f1"] = 0.61
        with pytest.raises(ValueError, match="池化缺失率 > 0.6"):
            tool.validate(distribution, pooled, spec)


class TestWriteBackfill:
    def test_write_and_readback_consistent(self, tmp_path):
        models_dir, spec, counts, data_state = _scan(tmp_path)
        addendum = _build(models_dir, spec, counts, data_state)
        result = tool.write_backfill(
            models_dir, spec, addendum, {"feature_missing_rate_sha256": "x"}
        )
        assert result["readback_equal"] is True
        assert Path(result["sidecar_backup"]).exists()
        assert Path(result["registry_backup"]).exists()
        sidecar = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))
        params = sidecar["train_params"]
        assert params["learning_rate"] == 0.03
        assert "f1" not in params and "amount_ma20" not in params  # 防顶层键污染
        coverage, error = read_feature_missing_rate(params)
        assert error is None and set(coverage) == set(addendum["feature_missing_rate"])
        registry = json.loads(Path(result["registry"]).read_text(encoding="utf-8"))
        entry = next(i for i in registry["models"] if int(i["version"]) == spec.version)
        assert entry["train_params"]["feature_missing_rate"] == addendum["feature_missing_rate"]

    def test_second_write_is_idempotent_noop(self, tmp_path):
        models_dir, spec, counts, data_state = _scan(tmp_path)
        addendum = _build(models_dir, spec, counts, data_state)
        tool.write_backfill(models_dir, spec, addendum, {"feature_missing_rate_sha256": "x"})
        before = (models_dir / "v99_metadata.json").read_bytes()
        result = tool.write_backfill(
            models_dir, spec, addendum, {"feature_missing_rate_sha256": "x"}
        )
        assert result["noop"] is True
        assert (models_dir / "v99_metadata.json").read_bytes() == before
        assert not (models_dir / "backfill_journal_v99.json").exists()

    def test_crash_between_writes_is_repaired_on_rerun(self, tmp_path, monkeypatch):
        # reviewer3 P1-5：registry 写失败 ⇒ journal 残留；重跑先修复再幂等 no-op
        models_dir, spec, counts, data_state = _scan(tmp_path)
        addendum = _build(models_dir, spec, counts, data_state)
        real_write = tool._atomic_write_json

        def _fail_registry(path, payload):
            if path.name == "model_registry.json":
                raise OSError("synthetic registry write failure")
            return real_write(path, payload)

        monkeypatch.setattr(tool, "_atomic_write_json", _fail_registry)
        with pytest.raises(OSError, match="synthetic registry write failure"):
            tool.write_backfill(models_dir, spec, addendum, {"feature_missing_rate_sha256": "x"})
        assert (models_dir / "backfill_journal_v99.json").exists()

        monkeypatch.setattr(tool, "_atomic_write_json", real_write)
        result = tool.write_backfill(
            models_dir, spec, addendum, {"feature_missing_rate_sha256": "x"}
        )
        assert result["readback_equal"] is True
        assert not (models_dir / "backfill_journal_v99.json").exists()
        sidecar_params = json.loads((models_dir / "v99_metadata.json").read_text(encoding="utf-8"))[
            "train_params"
        ]
        registry_params = next(
            item
            for item in json.loads(
                (models_dir / "model_registry.json").read_text(encoding="utf-8")
            )["models"]
            if int(item["version"]) == spec.version
        )["train_params"]
        assert sidecar_params["feature_missing_rate"] == registry_params["feature_missing_rate"]

    def test_v1_payload_is_overwritten(self, tmp_path):
        # A5 裁决：v2 重算覆盖 v1（v1 单值 μ 形态 ⇒ 可写）
        models_dir, spec, counts, data_state = _scan(tmp_path)
        sidecar_path = models_dir / "v99_metadata.json"
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        metadata["train_params"].update(
            {
                "feature_missing_rate_schema_version": 1,
                "feature_missing_rate": {"f1": 0.5},
                "feature_missing_rate_stat": {"source": "backfilled"},
            }
        )
        sidecar_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        addendum = _build(models_dir, spec, counts, data_state)
        result = tool.write_backfill(
            models_dir, spec, addendum, {"feature_missing_rate_sha256": "x"}
        )
        params = json.loads(Path(result["sidecar"]).read_text(encoding="utf-8"))["train_params"]
        assert params["feature_missing_rate_schema_version"] == 2
        coverage, error = read_feature_missing_rate(params)
        assert error is None and isinstance(coverage["f1"], dict)
