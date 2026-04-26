"""ML 模块测试"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import pytest

from src.lazybull.ml import ModelRegistry


class MockModel:
    """模拟机器学习模型"""
    
    def __init__(self):
        self.feature_importances_ = None
    
    def predict(self, X):
        """模拟预测"""
        return np.random.randn(len(X))


@pytest.fixture
def temp_models_dir():
    """创建临时模型目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_model_registry_init(temp_models_dir):
    """测试模型注册表初始化"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    assert registry.models_dir == Path(temp_models_dir)
    # 注册表文件在首次保存时创建，初始化时不一定存在
    assert registry.get_next_version() == 1
    assert len(registry.list_models()) == 0


def test_register_model(temp_models_dir):
    """测试注册模型"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    # 创建模拟模型
    model = MockModel()
    
    # 注册模型
    version = registry.register_model(
        model=model,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["feature1", "feature2", "feature3"],
        label_column="y_ret_5",
        n_samples=1000,
        train_params={"n_estimators": 100, "max_depth": 6},
        performance_metrics={"mse": 0.01, "r2": 0.85}
    )
    
    assert version == 1
    assert registry.get_next_version() == 2
    
    # 验证模型文件存在
    model_file = Path(temp_models_dir) / "v1_model.joblib"
    features_file = Path(temp_models_dir) / "v1_features.json"
    metadata_file = Path(temp_models_dir) / "v1_metadata.json"
    latest_version_file = Path(temp_models_dir) / "latest_model_version.txt"
    assert model_file.exists()
    assert features_file.exists()
    assert metadata_file.exists()
    assert latest_version_file.exists()
    
    # 验证特征文件内容
    with open(features_file, 'r') as f:
        features = json.load(f)
    assert features == ["feature1", "feature2", "feature3"]
    
    # 验证注册表内容
    models = registry.list_models()
    assert len(models) == 1
    assert models[0]["version"] == 1
    assert models[0]["model_type"] == "xgboost"
    assert models[0]["n_samples"] == 1000


def test_load_specific_version_without_registry_file_uses_metadata_sidecar(temp_models_dir):
    """测试指定版本可仅依赖旁路元数据加载。"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    model = MockModel()

    registry.register_model(
        model=model,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["feature1", "feature2"],
        label_column="y_ret_5",
        n_samples=1000,
        train_params={"n_estimators": 100, "max_depth": 6},
    )

    registry_file = Path(temp_models_dir) / "model_registry.json"
    registry_file.unlink()

    reloaded_registry = ModelRegistry(models_dir=temp_models_dir)
    loaded_model, metadata = reloaded_registry.load_model(version=1)

    assert loaded_model is not None
    assert metadata["version"] == 1
    assert metadata["feature_columns"] == ["feature1", "feature2"]


def test_load_specific_version_streams_large_registry_without_full_load(temp_models_dir, monkeypatch):
    """测试旧模型在无旁路文件时可按版本流式读取注册表。"""
    models_dir = Path(temp_models_dir)
    model = MockModel()
    joblib.dump(model, models_dir / "v1_model.joblib")
    joblib.dump(model, models_dir / "v2_model.joblib")

    with open(models_dir / "v1_features.json", 'w', encoding='utf-8') as f:
        json.dump(["feature1", "feature2"], f, ensure_ascii=False, indent=2)
    with open(models_dir / "v2_features.json", 'w', encoding='utf-8') as f:
        json.dump(["feature1", "feature2", "feature3"], f, ensure_ascii=False, indent=2)

    metadata_v1 = {
        "version": 1,
        "version_str": "v1",
        "model_type": "xgboost",
        "model_file": "v1_model.joblib",
        "features_file": "v1_features.json",
        "train_start_date": "20230101",
        "train_end_date": "20231231",
        "feature_count": 2,
        "label_column": "y_ret_5",
        "n_samples": 1000,
        "train_params": {"n_estimators": 100, "task": "regression"},
        "performance_metrics": {},
        "created_at": "2026-04-25 21:22:30",
    }
    metadata_v2 = {
        "version": 2,
        "version_str": "v2",
        "model_type": "xgboost",
        "model_file": "v2_model.joblib",
        "features_file": "v2_features.json",
        "train_start_date": "20240101",
        "train_end_date": "20241231",
        "feature_count": 3,
        "label_column": "y_ret_5",
        "n_samples": 1200,
        "train_params": {"n_estimators": 120, "task": "regression"},
        "performance_metrics": {},
        "created_at": "2026-04-25 21:25:30",
    }
    with open(models_dir / "model_registry.json", 'w', encoding='utf-8') as f:
        json.dump(
            {"models": [metadata_v1, metadata_v2], "next_version": 3},
            f,
            ensure_ascii=False,
            indent=2,
        )

    registry = ModelRegistry(models_dir=temp_models_dir)

    def _unexpected_full_load() -> Dict:
        raise AssertionError("指定版本加载不应回退到完整注册表加载")

    monkeypatch.setattr(registry, "_ensure_registry_loaded", _unexpected_full_load)

    loaded_model, loaded_metadata = registry.load_model(version=1)

    assert loaded_model is not None
    assert loaded_metadata["version"] == 1
    assert loaded_metadata["feature_columns"] == ["feature1", "feature2"]


def test_load_latest_version_uses_registry_tail_without_full_load(temp_models_dir, monkeypatch):
    """测试加载最新版本可先通过注册表尾部读取版本号。"""
    models_dir = Path(temp_models_dir)
    model = MockModel()
    joblib.dump(model, models_dir / "v2_model.joblib")

    with open(models_dir / "v2_features.json", 'w', encoding='utf-8') as f:
        json.dump(["feature1", "feature2", "feature3"], f, ensure_ascii=False, indent=2)

    metadata_v1 = {
        "version": 1,
        "version_str": "v1",
        "model_type": "xgboost",
        "model_file": "v1_model.joblib",
        "features_file": "v1_features.json",
        "train_start_date": "20220101",
        "train_end_date": "20221231",
        "feature_count": 1,
        "label_column": "y_ret_5",
        "n_samples": 100,
        "train_params": {"n_estimators": 50, "task": "regression"},
        "performance_metrics": {},
        "created_at": "2026-04-25 21:19:01",
    }
    metadata_v2 = {
        "version": 2,
        "version_str": "v2",
        "model_type": "xgboost",
        "model_file": "v2_model.joblib",
        "features_file": "v2_features.json",
        "train_start_date": "20230101",
        "train_end_date": "20231231",
        "feature_count": 3,
        "label_column": "y_ret_5",
        "n_samples": 1000,
        "train_params": {"n_estimators": 100, "task": "regression"},
        "performance_metrics": {},
        "created_at": "2026-04-25 21:22:30",
    }
    with open(models_dir / "model_registry.json", 'w', encoding='utf-8') as f:
        json.dump(
            {"models": [metadata_v1, metadata_v2], "next_version": 3},
            f,
            ensure_ascii=False,
            indent=2,
        )

    registry = ModelRegistry(models_dir=temp_models_dir)

    def _unexpected_full_load() -> Dict:
        raise AssertionError("加载最新版本不应先回退到完整注册表加载")

    monkeypatch.setattr(registry, "_ensure_registry_loaded", _unexpected_full_load)

    loaded_model, loaded_metadata = registry.load_model()

    assert loaded_model is not None
    assert loaded_metadata["version"] == 2
    assert loaded_metadata["feature_columns"] == ["feature1", "feature2", "feature3"]


def test_register_multiple_models(temp_models_dir):
    """测试注册多个模型（版本递增）"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    # 注册第一个模型
    model1 = MockModel()
    version1 = registry.register_model(
        model=model1,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20230630",
        feature_columns=["f1", "f2"],
        label_column="y_ret_5",
        n_samples=500,
        train_params={"n_estimators": 100}
    )
    
    # 注册第二个模型
    model2 = MockModel()
    version2 = registry.register_model(
        model=model2,
        model_type="xgboost",
        train_start_date="20230701",
        train_end_date="20231231",
        feature_columns=["f1", "f2", "f3"],
        label_column="y_ret_5",
        n_samples=600,
        train_params={"n_estimators": 100}
    )
    
    assert version1 == 1
    assert version2 == 2
    assert registry.get_next_version() == 3
    
    # 验证注册表
    models = registry.list_models()
    assert len(models) == 2


def test_load_latest_model(temp_models_dir):
    """测试加载最新模型"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    # 注册两个模型
    model1 = MockModel()
    registry.register_model(
        model=model1,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20230630",
        feature_columns=["f1", "f2"],
        label_column="y_ret_5",
        n_samples=500,
        train_params={"n_estimators": 100}
    )
    
    model2 = MockModel()
    registry.register_model(
        model=model2,
        model_type="xgboost",
        train_start_date="20230701",
        train_end_date="20231231",
        feature_columns=["f1", "f2", "f3"],
        label_column="y_ret_5",
        n_samples=600,
        train_params={"n_estimators": 100}
    )
    
    # 加载最新模型（不指定版本）
    loaded_model, metadata = registry.load_model()
    
    assert metadata["version"] == 2
    assert metadata["feature_columns"] == ["f1", "f2", "f3"]
    assert metadata["n_samples"] == 600


def test_load_specific_version(temp_models_dir):
    """测试加载指定版本模型"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    # 注册两个模型
    model1 = MockModel()
    registry.register_model(
        model=model1,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20230630",
        feature_columns=["f1", "f2"],
        label_column="y_ret_5",
        n_samples=500,
        train_params={"n_estimators": 100}
    )
    
    model2 = MockModel()
    registry.register_model(
        model=model2,
        model_type="xgboost",
        train_start_date="20230701",
        train_end_date="20231231",
        feature_columns=["f1", "f2", "f3"],
        label_column="y_ret_5",
        n_samples=600,
        train_params={"n_estimators": 100}
    )
    
    # 加载版本1
    loaded_model, metadata = registry.load_model(version=1)
    
    assert metadata["version"] == 1
    assert metadata["feature_columns"] == ["f1", "f2"]
    assert metadata["n_samples"] == 500


def test_load_nonexistent_version(temp_models_dir):
    """测试加载不存在的版本"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    # 注册一个模型
    model = MockModel()
    registry.register_model(
        model=model,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["f1"],
        label_column="y_ret_5",
        n_samples=100,
        train_params={"n_estimators": 100}
    )
    
    # 尝试加载不存在的版本
    with pytest.raises(ValueError, match="未找到版本"):
        registry.load_model(version=99)


def test_load_model_no_models(temp_models_dir):
    """测试加载模型时没有已注册模型"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    # 尝试加载模型
    with pytest.raises(ValueError, match="没有已注册的模型"):
        registry.load_model()


def test_get_latest_version(temp_models_dir):
    """测试获取最新版本号"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    # 没有模型时
    assert registry.get_latest_version() is None
    
    # 注册一个模型
    model = MockModel()
    registry.register_model(
        model=model,
        model_type="xgboost",
        train_start_date="20230101",
        train_end_date="20231231",
        feature_columns=["f1"],
        label_column="y_ret_5",
        n_samples=100,
        train_params={"n_estimators": 100}
    )
    
    assert registry.get_latest_version() == 1


def test_list_models(temp_models_dir):
    """测试列出所有模型"""
    registry = ModelRegistry(models_dir=temp_models_dir)
    
    # 初始时没有模型
    assert len(registry.list_models()) == 0
    
    # 注册三个模型
    for i in range(3):
        model = MockModel()
        registry.register_model(
            model=model,
            model_type="xgboost",
            train_start_date="20230101",
            train_end_date="20231231",
            feature_columns=[f"f{i}"],
            label_column="y_ret_5",
            n_samples=100 * (i + 1),
            train_params={"n_estimators": 100}
        )
    
    models = registry.list_models()
    assert len(models) == 3
    assert models[0]["version"] == 1
    assert models[1]["version"] == 2
    assert models[2]["version"] == 3
