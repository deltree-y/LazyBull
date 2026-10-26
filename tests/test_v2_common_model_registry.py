# -*- coding: utf-8 -*-
"""v2 P3-T2 ModelRegistry 整移测试（注册表回读四类 + 读路径 S3 无回填；tmp_path 隔离）。

覆盖（唯一来源 ``src.lazybull.v2.common.model_registry.ModelRegistry``）：
- 普通模型注册回读：metadata 字段 / feature_columns / latest 旁路 / list_models；
- 现金流质量列模型：train_params 缺 schema 版本 ⇒ 「现金流质量列 + train/serve 语义偏差」
  WARNING；记录正确版本 3 ⇒ 无该 WARNING；
- 分红政策列模型：缺版本 ⇒ 「分红政策列」WARNING；记录版本 3 ⇒ 无；
- 旧 joblib 模型：无 save_model 属性对象走 joblib 路径，model_file 以 .joblib 结尾且可回读；
- S3 读路径无回填：sidecar 缺失 ⇒ load_model / get_latest_version 抛 FileNotFoundError，
  且目录零新文件（旧实现会静默补写 sidecar，新实现禁止）；
- check_feature_consistency：缺列 ⇒ ValueError，齐全 ⇒ 通过；
- strict_version_check：sidecar 缺 train_params ⇒ True 抛 ValueError、False 放行。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from loguru import logger as loguru_logger

from src.lazybull.v2.common.model_registry import ModelRegistry

_FEATURE_COLUMNS = ["feat_a", "feat_b", "feat_c"]


def _small_xgb_model():
    """CPU 小模型（n_estimators=5，tree_method=hist + device=cpu，不依赖 GPU）。"""
    import xgboost as xgb

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((40, 3)), columns=_FEATURE_COLUMNS)
    y = pd.Series(rng.standard_normal(40) * 0.05)
    model = xgb.XGBRegressor(n_estimators=5, tree_method="hist", device="cpu", random_state=42)
    model.fit(X, y)
    return model


def _register(
    registry: ModelRegistry,
    model=None,
    feature_columns: Optional[List[str]] = None,
    train_params: Optional[Dict] = None,
) -> int:
    """以固定元数据注册一个模型，返回版本号。"""
    return registry.register_model(
        model if model is not None else _small_xgb_model(),
        model_type="xgboost",
        train_start_date="20240102",
        train_end_date="20241231",
        feature_columns=feature_columns or list(_FEATURE_COLUMNS),
        label_column="neu_y_ret_20",
        n_samples=40,
        train_params=train_params if train_params is not None else {"n_estimators": 5},
        performance_metrics={"mse": 0.01},
    )


def _capture_warning_messages() -> tuple:
    """挂 loguru WARNING sink 捕获消息（返回 (messages, sink_id)，用后须 remove）。"""
    messages: List[str] = []
    sink_id = loguru_logger.add(lambda msg: messages.append(str(msg)), level="WARNING")
    return messages, sink_id


class TestRegisterAndLoadRoundtrip:
    def test_register_and_load_roundtrip(self, tmp_path):
        """普通模型：注册 ⇒ 回读，metadata 字段 / 特征列 / latest 旁路 / list_models 全对。"""
        registry = ModelRegistry(models_dir=str(tmp_path))
        version = _register(registry)
        assert version == 1

        model, metadata = registry.load_model(version)
        assert metadata["version_str"] == "v1"
        assert metadata["model_type"] == "xgboost"
        assert metadata["train_start_date"] == "20240102"
        assert metadata["train_end_date"] == "20241231"
        assert metadata["feature_count"] == len(_FEATURE_COLUMNS)
        assert metadata["label_column"] == "neu_y_ret_20"
        assert metadata["n_samples"] == 40
        assert metadata["train_params"] == {"n_estimators": 5}
        assert metadata["performance_metrics"] == {"mse": 0.01}
        assert metadata["feature_columns"] == _FEATURE_COLUMNS

        # latest_model_version.txt 指向该版本
        latest_file = tmp_path / "latest_model_version.txt"
        assert latest_file.read_text(encoding="utf-8").strip() == "1"
        assert registry.get_latest_version() == 1
        # list_models 含该版本
        assert [entry["version"] for entry in registry.list_models()] == [1]
        # 原生 JSON 回读的模型可正常 predict
        probe = pd.DataFrame(np.zeros((2, 3)), columns=_FEATURE_COLUMNS)
        assert model.predict(probe).shape == (2,)


class TestLegacySchemaWarnings:
    def test_cashflow_columns_warn_without_schema_version(self, tmp_path):
        """现金流列模型：train_params 缺 cashflow_quality_schema_version ⇒ WARNING。"""
        registry = ModelRegistry(models_dir=str(tmp_path / "legacy"))
        version = _register(
            registry,
            feature_columns=["feat_a", "zscore_ocf_to_profit"],
            train_params={"n_estimators": 5},
        )
        messages, sink_id = _capture_warning_messages()
        try:
            registry.load_model(version)
        finally:
            loguru_logger.remove(sink_id)
        cashflow_warnings = [m for m in messages if "现金流质量列" in m]
        assert cashflow_warnings, "缺版本未触发现金流质量列 WARNING"
        assert any("train/serve 语义偏差" in m for m in cashflow_warnings)

        # 记录正确版本 3 ⇒ 无该 WARNING
        registry_ok = ModelRegistry(models_dir=str(tmp_path / "versioned"))
        version_ok = _register(
            registry_ok,
            feature_columns=["feat_a", "zscore_ocf_to_profit"],
            train_params={"n_estimators": 5, "cashflow_quality_schema_version": 3},
        )
        messages_ok, sink_id_ok = _capture_warning_messages()
        try:
            registry_ok.load_model(version_ok)
        finally:
            loguru_logger.remove(sink_id_ok)
        assert not any("现金流质量列" in m for m in messages_ok)

    def test_dividend_columns_warn_without_schema_version(self, tmp_path):
        """分红列模型：train_params 缺 dividend_policy_schema_version ⇒ WARNING。"""
        registry = ModelRegistry(models_dir=str(tmp_path / "legacy"))
        version = _register(
            registry,
            feature_columns=["feat_a", "zscore_dividend_payout_ratio"],
            train_params={"n_estimators": 5},
        )
        messages, sink_id = _capture_warning_messages()
        try:
            registry.load_model(version)
        finally:
            loguru_logger.remove(sink_id)
        assert any("分红政策列" in m for m in messages), "缺版本未触发分红政策列 WARNING"

        # 记录正确版本 3 ⇒ 无该 WARNING
        registry_ok = ModelRegistry(models_dir=str(tmp_path / "versioned"))
        version_ok = _register(
            registry_ok,
            feature_columns=["feat_a", "zscore_dividend_payout_ratio"],
            train_params={"n_estimators": 5, "dividend_policy_schema_version": 3},
        )
        messages_ok, sink_id_ok = _capture_warning_messages()
        try:
            registry_ok.load_model(version_ok)
        finally:
            loguru_logger.remove(sink_id_ok)
        assert not any("分红政策列" in m for m in messages_ok)


class TestJoblibModelPath:
    def test_joblib_model_roundtrip(self, tmp_path):
        """无 save_model 属性对象 ⇒ joblib 路径：model_file 以 .joblib 结尾且可回读。"""
        registry = ModelRegistry(models_dir=str(tmp_path))
        plain_model = {"weights": [1.0, 2.0, 3.0], "bias": 0.5}
        version = _register(registry, model=plain_model)

        sidecar = tmp_path / f"v{version}_metadata.json"
        sidecar_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        assert sidecar_metadata["model_file"].endswith(".joblib")

        loaded_model, metadata = registry.load_model(version)
        assert metadata["model_file"].endswith(".joblib")
        assert loaded_model == plain_model


class TestS3NoSidecarBackfill:
    def test_missing_sidecar_raises_and_writes_nothing(self, tmp_path):
        """S3：sidecar 缺失 ⇒ FileNotFoundError，且读路径零新文件（不补写 sidecar）。"""
        registry = ModelRegistry(models_dir=str(tmp_path))
        version = _register(registry)

        # 删除 v{N}_metadata.json（保留 model_registry.json）
        (tmp_path / f"v{version}_metadata.json").unlink()
        files_before = {p.name for p in tmp_path.iterdir()}
        assert f"v{version}_metadata.json" not in files_before

        with pytest.raises(FileNotFoundError):
            registry.load_model(version)
        with pytest.raises(FileNotFoundError):
            registry.get_latest_version()

        files_after = {p.name for p in tmp_path.iterdir()}
        assert files_after == files_before, "S3 违背：读路径产生了新文件（sidecar 回填）"


class TestFeatureConsistency:
    def test_check_feature_consistency(self, tmp_path):
        """available_features 缺列 ⇒ ValueError；齐全（含多余列）⇒ 通过。"""
        registry = ModelRegistry(models_dir=str(tmp_path))
        version = _register(registry)
        _, metadata = registry.load_model(version)

        with pytest.raises(ValueError, match="缺少模型训练时使用的特征列"):
            registry.check_feature_consistency(metadata, ["feat_a", "feat_b"])
        # 齐全（多余列允许）
        registry.check_feature_consistency(metadata, ["feat_a", "feat_b", "feat_c", "extra_col"])


class TestStrictVersionCheck:
    def test_missing_train_params_sidecar(self, tmp_path):
        """sidecar 缺 train_params：strict_version_check=True ⇒ ValueError；False ⇒ 放行。"""
        registry = ModelRegistry(models_dir=str(tmp_path))
        version = _register(registry)

        # 手工改 sidecar 去掉 train_params
        sidecar: Path = tmp_path / f"v{version}_metadata.json"
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata.pop("train_params")
        sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        with pytest.raises(ValueError, match="缺少新版本必需的元数据字段"):
            registry.load_model(version, strict_version_check=True)

        model, relaxed_metadata = registry.load_model(version, strict_version_check=False)
        assert relaxed_metadata["version_str"] == f"v{version}"
        assert model is not None
